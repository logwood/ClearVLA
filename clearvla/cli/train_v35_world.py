from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    print_context,
    resolve_device,
)
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.dynamic_world_lab.dataset import PairedDynamicWorldDataset
from clearvla.experiments.dynamic_world_lab.pairing import LocalPairTable, build_local_pair_table
from clearvla.experiments.dynamic_world_lab.shared_runtime import encode_current_tokens
from clearvla.experiments.observed_state_lab.dataset import (
    CurrentEvidenceViewDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from clearvla.experiments.observed_state_lab.world_model import (
    V35ObservedStateWorldModel,
    V35WorldConfig,
)
from clearvla.experiments.observed_state_lab.world_objectives import V35WorldLossConfig
from clearvla.experiments.observed_state_lab.world_runtime import (
    V35WorldTrainerConfig,
    train_v35_world,
)


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V35 observed-state segment-recurrent latent dynamics."
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pair-index-dir", type=Path, default=None)
    parser.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")
    parser.add_argument("--world-horizon", type=int, default=48)
    parser.add_argument("--policy-horizon", type=int, default=24)
    parser.add_argument("--segment-length", type=int, default=4)
    parser.add_argument("--history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    parser.add_argument("--executed-action-offsets", nargs="+", type=int, default=[-8, -4, -1])
    parser.add_argument("--target-history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    parser.add_argument("--state-offset", type=int, default=0)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--action-offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--control-hz", type=float, default=30.0)

    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--debug-token-dim", type=int, default=64)
    parser.add_argument("--debug-patches-per-camera", type=int, default=16)

    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--world-tokens", type=int, default=32)
    parser.add_argument("--global-tokens", type=int, default=8)
    parser.add_argument("--interaction-tokens", type=int, default=12)
    parser.add_argument("--motion-tokens", type=int, default=12)
    parser.add_argument("--evidence-read-depth", type=int, default=2)
    parser.add_argument("--latent-mix-depth", type=int, default=2)
    parser.add_argument("--action-depth", type=int, default=2)
    parser.add_argument("--transition-depth", type=int, default=4)
    parser.add_argument("--transition-unique-blocks", type=int, default=2)
    parser.add_argument("--inverse-depth", type=int, default=2)
    parser.add_argument("--descriptor-projection-dim", type=int, default=32)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--gripper-open-value", type=float, default=0.0)
    parser.add_argument("--gripper-close-value", type=float, default=1.7459820890426636)
    parser.add_argument(
        "--rollout-checkpoint",
        action="store_true",
        help="Checkpoint each segment-recurrent rollout step during training to reduce activation memory.",
    )
    parser.add_argument("--consequence-slots", type=int, default=3)
    parser.add_argument("--consequence-feedback-depth", type=int, default=1)
    parser.add_argument("--consequence-temperature", type=float, default=0.7)
    parser.add_argument(
        "--no-rollout-checkpoint-preserve-rng-state",
        dest="rollout_checkpoint_preserve_rng_state",
        action="store_false",
        help="Do not preserve RNG state inside rollout checkpoint recomputation. Leave default on unless the transition has no stochastic layers.",
    )
    parser.set_defaults(rollout_checkpoint_preserve_rng_state=True)

    defaults = V35WorldLossConfig()
    for field in V35WorldLossConfig.__dataclass_fields__:
        value = getattr(defaults, field)
        parser.add_argument(
            "--" + field.replace("_", "-"), dest=field, type=type(value), default=value
        )

    parser.add_argument("--pair-candidates", type=int, default=96)
    parser.add_argument("--pair-min-action-distance", type=float, default=1.0)
    parser.add_argument("--pair-min-future-distance", type=float, default=0.75)
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--pair-build-batch-size", type=int, default=32)

    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    trainer = V35WorldTrainerConfig()
    for name in (
        "epochs",
        "encoder_lr",
        "dynamics_lr",
        "auxiliary_lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "grad_clip",
        "warmup_steps",
        "action_warmup_steps",
        "stability_warmup_steps",
        "min_lr_ratio",
        "ema_decay_start",
        "ema_decay_end",
        "camera_drop_prob",
        "state_mask_prob",
        "executed_action_mask_prob",
        "patch_mask_prob",
        "checkpoint_predictive_slack",
        "checkpoint_hold_ratio_max",
        "checkpoint_min_embedding_std",
        "checkpoint_zero_world_max",
        "log_every",
        "max_train_batches",
        "max_val_batches",
        "eval_ablation_batches",
    ):
        value = getattr(trainer, name)
        parser.add_argument(
            "--" + name.replace("_", "-"), dest=name, type=type(value), default=value
        )
    return parser.parse_args()


@torch.no_grad()
def pair_descriptors(
    dataset,
    *,
    conditioner,
    model,
    cameras,
    device,
    dtype,
    batch_size,
    workers,
    gripper_open_value: float,
    gripper_close_value: float,
):
    loader = DataLoader(
        CurrentEvidenceViewDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    visual_rows, state_rows, executed_rows = [], [], []
    for batch in loader:
        tokens = encode_current_tokens(
            batch,
            conditioner=conditioner,
            model_config=model.config,
            camera_names=cameras,
            device=device,
            dtype=dtype,
        )
        region = model.fixed_region_descriptor(tokens).reshape(len(tokens), -1).cpu().numpy()
        visual_rows.append(region)
        state_rows.append(batch["state"].numpy().reshape(len(tokens), -1))
        executed_rows.append(batch["executed_action_history"].numpy().reshape(len(tokens), -1))
    visual = np.concatenate(visual_rows)
    state = np.concatenate(state_rows)
    executed = np.concatenate(executed_rows)
    metadata = [dataset.descriptor_metadata(i) for i in range(len(dataset))]
    action = np.stack([row["action_summary"] for row in metadata])
    future = np.stack([row["future_summary"] for row in metadata])
    episode = np.asarray([row["episode_idx"] for row in metadata], dtype=np.int64)
    state_raw = np.stack([row["state_raw"] for row in metadata])
    gripper_midpoint = 0.5 * (float(gripper_open_value) + float(gripper_close_value))
    gripper = (state_raw[:, model.config.gripper_index] >= gripper_midpoint).astype(np.int64)
    return np.concatenate([state, executed, visual], axis=1), action, future, episode, gripper


def build_or_load_pairs(pair_dir: Path, bases: dict[str, ObservedStateWindowDataset], **kwargs):
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: pair_dir / f"{name}_local_pairs.npz" for name in bases}
    manifest_path = pair_dir / "pair_index_manifest.json"
    expected = {
        "schema": "clearvla-v35-local-pair-index-v1",
        "train_windows": len(bases["train"]),
        "val_windows": len(bases["val"]),
        "test_windows": len(bases["test"]),
        "segment_offsets": list(bases["train"].config.future_offsets),
        "world_horizon": bases["train"].config.world_horizon,
    }
    rebuild = bool(kwargs["rebuild"])
    if not rebuild and all(path.is_file() for path in paths.values()) and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == expected:
            return {name: LocalPairTable.load(path) for name, path in paths.items()}
    descriptor_kwargs = {
        key: kwargs[key]
        for key in (
            "conditioner",
            "model",
            "cameras",
            "device",
            "dtype",
            "batch_size",
            "workers",
            "gripper_open_value",
            "gripper_close_value",
        )
    }
    rows = {name: pair_descriptors(dataset, **descriptor_kwargs) for name, dataset in bases.items()}
    tables = {}
    for name, (condition, action, future, episode, gripper) in rows.items():
        tables[name] = build_local_pair_table(
            condition_descriptor=condition,
            action_summary=action,
            future_summary=future,
            episode_ids=episode,
            gripper_state=gripper,
            candidate_count=kwargs["candidate_count"],
            min_action_distance=kwargs["min_action_distance"],
            min_future_distance=kwargs["min_future_distance"],
        )
        tables[name].save(paths[name])
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return tables


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    cameras = tuple(str(x) for x in args.cameras)
    dataset_config = ObservedStateDatasetConfig(
        world_horizon=args.world_horizon,
        policy_horizon=args.policy_horizon,
        segment_length=args.segment_length,
        history_offsets=tuple(args.history_offsets),
        executed_action_offsets=tuple(args.executed_action_offsets),
        target_history_offsets=tuple(args.target_history_offsets),
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        return_images=args.condition_mode != "dinov2-cache",
    )
    dataset_config.validate()
    min_length = (
        args.world_horizon + abs(min(args.history_offsets + args.executed_action_offsets)) + 2
    )
    episodes, train_ids, val_ids, test_ids, action_norm, state_norm, image_store, skipped = (
        load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    )
    bases = {
        name: ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_norm,
            action_normalizer=action_norm,
            config=dataset_config,
        )
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=args.debug_token_dim,
        debug_patches_per_camera=args.debug_patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if patches is None:
        latent_dim, patches = infer_dense_geometry(
            conditioner, bases["train"][0], camera_names=cameras
        )
    model_config = V35WorldConfig(
        latent_dim=latent_dim,
        action_dim=int(action_norm.scale.shape[-1]),
        state_dim=int(state_norm.scale.shape[-1]),
        world_horizon=args.world_horizon,
        segment_length=args.segment_length,
        history_length=len(args.history_offsets),
        executed_history_length=len(args.executed_action_offsets),
        num_cameras=len(cameras),
        patches_per_camera=patches,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
        world_tokens=args.world_tokens,
        global_tokens=args.global_tokens,
        interaction_tokens=args.interaction_tokens,
        motion_tokens=args.motion_tokens,
        evidence_read_depth=args.evidence_read_depth,
        latent_mix_depth=args.latent_mix_depth,
        action_depth=args.action_depth,
        transition_depth=args.transition_depth,
        transition_unique_blocks=args.transition_unique_blocks,
        inverse_depth=args.inverse_depth,
        descriptor_projection_dim=args.descriptor_projection_dim,
        gripper_dim_index=args.gripper_dim_index,
        rollout_checkpoint=bool(args.rollout_checkpoint),
        consequence_slots=args.consequence_slots,
        consequence_feedback_depth=args.consequence_feedback_depth,
        consequence_temperature=args.consequence_temperature,
        rollout_checkpoint_preserve_rng_state=bool(args.rollout_checkpoint_preserve_rng_state),
    )
    model = V35ObservedStateWorldModel(model_config).to(device=device, dtype=torch.float32)
    pair_dir = args.pair_index_dir or args.out_dir / "pair_index"
    tables = build_or_load_pairs(
        pair_dir,
        bases,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        batch_size=args.pair_build_batch_size,
        workers=args.num_workers,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        candidate_count=args.pair_candidates,
        min_action_distance=args.pair_min_action_distance,
        min_future_distance=args.pair_min_future_distance,
        rebuild=args.rebuild_pairs,
    )
    datasets = {
        name: PairedDynamicWorldDataset(
            base,
            pair_index=tables[name].pair_index,
            pair_valid=tables[name].pair_valid,
            pair_distance=tables[name].pair_distance,
            action_distance=tables[name].action_distance,
            future_distance=tables[name].future_distance,
        )
        for name, base in bases.items()
    }
    train_loader = make_loader(
        datasets["train"],
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        datasets["val"],
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    loss_config = V35WorldLossConfig(
        **{name: getattr(args, name) for name in V35WorldLossConfig.__dataclass_fields__}
    )
    trainer = V35WorldTrainerConfig(
        **{name: getattr(args, name) for name in V35WorldTrainerConfig.__dataclass_fields__}
    )
    context = {
        "schema": "clearvla-v35-observed-state-world-context-v1",
        "args": vars(args),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "dataset": asdict(dataset_config),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "trainer": asdict(trainer),
        "train_windows": len(bases["train"]),
        "val_windows": len(bases["val"]),
        "test_windows": len(bases["test"]),
        "pair_index_dir": pair_dir,
        "parameter_report": model.parameter_report(),
        "architecture_contract": "action-independent observed-state encoder + shared segment-recurrent full-width world-action dynamics",
        "causal_scope": "action-conditioned observational dynamics; strict causal claims require intervention data",
    }
    print_context(context)
    train_v35_world(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        out_dir=args.out_dir,
        trainer=trainer,
        loss_config=loss_config,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        context=context,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
