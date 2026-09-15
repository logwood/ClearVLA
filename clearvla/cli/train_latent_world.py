from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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
from clearvla.experiments.dynamic_world_lab.dataset import (
    CurrentHistoryViewDataset,
    DynamicWorldDatasetConfig,
    DynamicWorldWindowDataset,
    PairedDynamicWorldDataset,
)
from clearvla.experiments.dynamic_world_lab.latent_world_model import (
    LatentWorldConfig,
    LatentWorldModel,
)
from clearvla.experiments.dynamic_world_lab.latent_world_objectives import LatentWorldLossConfig
from clearvla.experiments.dynamic_world_lab.latent_world_runtime import (
    LatentWorldTrainerConfig,
    train_latent_world,
)
from clearvla.experiments.dynamic_world_lab.pairing import (
    LocalPairTable,
    build_local_pair_table,
    nearest_support,
)
from clearvla.experiments.dynamic_world_lab.shared_runtime import encode_current_tokens


def _parse_offsets(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train V34.1 in one continuous run: a direct full-evidence World Perceiver, "
            "dense segment-correct Joint-AdaLN dynamics, shared actual/hold counterfactuals, "
            "FP32 master/EMA parameters and latent-only readouts."
        )
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")
    parser.add_argument("--action-horizon", type=int, default=48)
    parser.add_argument("--history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    parser.add_argument("--future-offsets", nargs="+", type=int, default=[8, 24, 48])
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

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument(
        "--encoder-depth",
        type=int,
        default=1,
        help="legacy config field; V34.1 Perceiver does not use the old encoder",
    )
    parser.add_argument("--perceiver-depth", type=int, default=4)
    parser.add_argument("--action-depth", type=int, default=3)
    parser.add_argument("--dynamics-depth", type=int, default=6)
    parser.add_argument("--dynamics-unique-blocks", type=int, default=3)
    parser.add_argument("--latent-stride", type=int, default=4)
    parser.add_argument("--state-decoder-depth", type=int, default=2)
    parser.add_argument("--inverse-depth", type=int, default=2)
    parser.add_argument("--action-probe-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--dynamic-tokens", type=int, default=16)
    parser.add_argument("--descriptor-projection-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--gripper-open-value", type=float, default=0.0)
    parser.add_argument("--gripper-close-value", type=float, default=1.7459820890426636)

    # V34.1 losses.  val_full retains the legacy scene+dynamic metric.
    defaults = LatentWorldLossConfig()
    for field, spec in LatentWorldLossConfig.__dataclass_fields__.items():
        value = getattr(defaults, field)
        option = "--" + field.replace("_", "-")
        parser.add_argument(option, dest=field, type=type(value), default=value)

    parser.add_argument("--pair-index-dir", type=Path, default=None)
    parser.add_argument("--pair-candidates", type=int, default=96)
    parser.add_argument("--pair-min-action-distance", type=float, default=1.0)
    parser.add_argument("--pair-min-future-distance", type=float, default=0.75)
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--pair-build-batch-size", type=int, default=32)

    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="bf16",
        help="forward autocast dtype; parameters and EMA remain FP32",
    )
    trainer_defaults = LatentWorldTrainerConfig()
    parser.add_argument("--epochs", type=int, default=trainer_defaults.epochs)
    parser.add_argument("--perceiver-lr", type=float, default=trainer_defaults.perceiver_lr)
    parser.add_argument("--dynamics-lr", type=float, default=trainer_defaults.dynamics_lr)
    parser.add_argument("--auxiliary-lr", type=float, default=trainer_defaults.auxiliary_lr)
    parser.add_argument("--weight-decay", type=float, default=trainer_defaults.weight_decay)
    parser.add_argument("--beta1", type=float, default=trainer_defaults.beta1)
    parser.add_argument("--beta2", type=float, default=trainer_defaults.beta2)
    parser.add_argument("--adam-eps", type=float, default=trainer_defaults.eps)
    parser.add_argument("--grad-clip", type=float, default=trainer_defaults.grad_clip)
    parser.add_argument("--warmup-steps", type=int, default=trainer_defaults.warmup_steps)
    parser.add_argument(
        "--action-warmup-steps", type=int, default=trainer_defaults.action_warmup_steps
    )
    parser.add_argument(
        "--stability-warmup-steps", type=int, default=trainer_defaults.stability_warmup_steps
    )
    parser.add_argument("--min-lr-ratio", type=float, default=trainer_defaults.min_lr_ratio)
    parser.add_argument("--ema-decay-start", type=float, default=trainer_defaults.ema_decay_start)
    parser.add_argument("--ema-decay-end", type=float, default=trainer_defaults.ema_decay_end)
    parser.add_argument("--camera-drop-prob", type=float, default=trainer_defaults.camera_drop_prob)
    parser.add_argument("--state-mask-prob", type=float, default=trainer_defaults.state_mask_prob)
    parser.add_argument("--patch-mask-prob", type=float, default=trainer_defaults.patch_mask_prob)
    parser.add_argument(
        "--checkpoint-predictive-slack",
        type=float,
        default=trainer_defaults.checkpoint_predictive_slack,
    )
    parser.add_argument(
        "--checkpoint-hold-ratio-max",
        type=float,
        default=trainer_defaults.checkpoint_hold_ratio_max,
    )
    parser.add_argument(
        "--checkpoint-min-embedding-std",
        type=float,
        default=trainer_defaults.checkpoint_min_embedding_std,
    )
    parser.add_argument("--log-every", type=int, default=trainer_defaults.log_every)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--eval-ablation-batches", type=int, default=trainer_defaults.eval_ablation_batches
    )
    return parser.parse_args()


@torch.no_grad()
def _pair_descriptors(
    dataset: DynamicWorldWindowDataset,
    *,
    conditioner,
    model: LatentWorldModel,
    cameras,
    device,
    dtype,
    batch_size: int,
    workers: int,
    gripper_midpoint: float,
):
    loader = DataLoader(
        CurrentHistoryViewDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    condition_rows, action_rows, future_rows, episode_rows, gripper_rows = [], [], [], [], []
    offsets = np.asarray(model.config.future_offsets, dtype=np.int64) - 1
    for batch in loader:
        current = encode_current_tokens(
            batch,
            conditioner=conditioner,
            model_config=model.config,
            camera_names=cameras,
            device=device,
            dtype=dtype,
        )
        dynamic_descriptor = model.fixed_dynamic_descriptor(current).cpu().numpy()
        static = current.float()[:, -1].mean(dim=2) @ model.descriptor_projection.float()
        static = (
            torch.nn.functional.normalize(static, dim=-1).reshape(len(current), -1).cpu().numpy()
        )
        state = batch["state"].numpy().reshape(len(current), -1)
        action_state = batch["action_state"].numpy().reshape(len(current), -1)
        condition_rows.append(np.concatenate([state, static, dynamic_descriptor], axis=1))

        action = batch["action"].numpy()
        boundary = np.concatenate([action_state[:, None], action[:, :-1]], axis=1)
        velocity = action - boundary
        sampled = action[:, offsets]
        action_rows.append(
            np.concatenate(
                [
                    sampled.reshape(len(action), -1),
                    velocity.mean(1),
                    velocity.std(1),
                    action[:, -1] - action_state,
                ],
                axis=1,
            )
        )

        future = batch["future_state"].numpy()
        future_boundary = np.concatenate([state[:, None], future[:, :-1]], axis=1)
        future_velocity = future - future_boundary
        future_sampled = future[:, offsets]
        future_rows.append(
            np.concatenate(
                [
                    future_sampled.reshape(len(future), -1),
                    future_velocity.mean(1),
                    future_velocity.std(1),
                    future[:, -1] - state,
                ],
                axis=1,
            )
        )
        episode_rows.append(batch["episode_idx"].numpy())
        raw_g = batch["state_raw"].numpy()[:, model.config.gripper_index]
        gripper_rows.append((raw_g > float(gripper_midpoint)).astype(np.int64))
    return tuple(
        np.concatenate(rows)
        for rows in (condition_rows, action_rows, future_rows, episode_rows, gripper_rows)
    )


def _build_or_load_pairs(
    *,
    pair_dir: Path,
    bases: dict[str, DynamicWorldWindowDataset],
    conditioner,
    model: LatentWorldModel,
    cameras,
    device,
    dtype,
    args,
):
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: pair_dir / f"{name}_local_pairs.npz" for name in ("train", "val", "test")}
    paths.update(
        {
            "val_support": pair_dir / "val_support_distance.npy",
            "val_support_index": pair_dir / "val_support_index.npy",
            "test_support": pair_dir / "test_support_distance.npy",
            "test_support_index": pair_dir / "test_support_index.npy",
            "manifest": pair_dir / "pair_index_manifest.json",
        }
    )
    expected = {
        "schema": "clearvla-v34.1-local-pair-index-v1",
        "train_windows": len(bases["train"]),
        "val_windows": len(bases["val"]),
        "test_windows": len(bases["test"]),
        "train_episode_ids": list(bases["train"].episode_ids),
        "val_episode_ids": list(bases["val"].episode_ids),
        "test_episode_ids": list(bases["test"].episode_ids),
        "history_length": model.config.history_length,
        "future_offsets": list(model.config.future_offsets),
        "action_horizon": model.config.action_horizon,
        "descriptor_seed": model.config.descriptor_seed,
        "descriptor_projection_dim": model.config.descriptor_projection_dim,
        "candidate_count": int(args.pair_candidates),
        "min_action_distance": float(args.pair_min_action_distance),
        "min_future_distance": float(args.pair_min_future_distance),
        "gripper_midpoint": float(0.5 * (args.gripper_open_value + args.gripper_close_value)),
    }
    required = [
        paths["train"],
        paths["val"],
        paths["test"],
        paths["val_support"],
        paths["val_support_index"],
        paths["test_support"],
        paths["test_support_index"],
        paths["manifest"],
    ]
    if not args.rebuild_pairs and all(path.exists() for path in required):
        saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if saved == expected:
            return (
                LocalPairTable.load(paths["train"]),
                LocalPairTable.load(paths["val"]),
                LocalPairTable.load(paths["test"]),
                np.load(paths["val_support_index"]),
                np.load(paths["val_support"]),
                np.load(paths["test_support_index"]),
                np.load(paths["test_support"]),
            )
        print("[latent-world] stale pair manifest; rebuilding", flush=True)

    midpoint = 0.5 * (args.gripper_open_value + args.gripper_close_value)
    rows = {
        name: _pair_descriptors(
            dataset,
            conditioner=conditioner,
            model=model,
            cameras=cameras,
            device=device,
            dtype=dtype,
            batch_size=args.pair_build_batch_size,
            workers=args.num_workers,
            gripper_midpoint=midpoint,
        )
        for name, dataset in bases.items()
    }
    tables = {}
    for name in ("train", "val", "test"):
        condition, action, future, episode, gripper = rows[name]
        tables[name] = build_local_pair_table(
            condition_descriptor=condition,
            action_summary=action,
            future_summary=future,
            episode_ids=episode,
            gripper_state=gripper,
            candidate_count=args.pair_candidates,
            min_action_distance=args.pair_min_action_distance,
            min_future_distance=args.pair_min_future_distance,
        )
        tables[name].save(paths[name])
    val_support_index, val_support = nearest_support(
        query_descriptor=rows["val"][0], reference_descriptor=rows["train"][0]
    )
    test_support_index, test_support = nearest_support(
        query_descriptor=rows["test"][0], reference_descriptor=rows["train"][0]
    )
    np.save(paths["val_support_index"], val_support_index)
    np.save(paths["val_support"], val_support)
    np.save(paths["test_support_index"], test_support_index)
    np.save(paths["test_support"], test_support)
    paths["manifest"].write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return (
        tables["train"],
        tables["val"],
        tables["test"],
        val_support_index,
        val_support,
        test_support_index,
        test_support,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype)
    cameras = tuple(str(value) for value in args.cameras)
    history_offsets = _parse_offsets(args.history_offsets)
    future_offsets = _parse_offsets(args.future_offsets)
    target_history_offsets = _parse_offsets(args.target_history_offsets)
    dataset_config = DynamicWorldDatasetConfig(
        action_horizon=args.action_horizon,
        history_offsets=history_offsets,
        future_offsets=future_offsets,
        target_history_offsets=target_history_offsets,
        state_offset=args.state_offset,
        image_offset=args.image_offset,
        action_offset=args.action_offset,
        stride=args.stride,
        return_images=args.condition_mode != "dinov2-cache",
    )
    dataset_config.validate()
    max_extent = max(args.action_horizon, max(future_offsets) - min(target_history_offsets))
    min_length = max_extent + abs(min(history_offsets)) + 2
    (
        episodes,
        train_ids,
        val_ids,
        test_ids,
        action_normalizer,
        state_normalizer,
        image_store,
        skipped,
    ) = load_data(args, min_length=min_length, normalizer_mode=args.normalizer)
    bases = {
        name: DynamicWorldWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
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

    model_config = LatentWorldConfig(
        latent_dim=latent_dim,
        action_dim=int(action_normalizer.scale.shape[-1]),
        state_dim=int(state_normalizer.scale.shape[-1]),
        action_horizon=args.action_horizon,
        history_length=len(history_offsets),
        num_cameras=len(cameras),
        patches_per_camera=patches,
        future_offsets=future_offsets,
        hidden_size=args.hidden_size,
        encoder_depth=args.encoder_depth,
        predictor_depth=1,
        action_depth=args.action_depth,
        num_heads=args.heads,
        context_tokens=args.context_tokens,
        dynamic_tokens=args.dynamic_tokens,
        descriptor_projection_dim=args.descriptor_projection_dim,
        dropout=args.dropout,
        input_mode="full",
        gripper_dim_index=args.gripper_dim_index,
        perceiver_depth=args.perceiver_depth,
        dynamics_depth=args.dynamics_depth,
        dynamics_unique_blocks=args.dynamics_unique_blocks,
        state_decoder_depth=args.state_decoder_depth,
        inverse_depth=args.inverse_depth,
        latent_stride=args.latent_stride,
        action_probe_depth=args.action_probe_depth,
    )
    model = LatentWorldModel(model_config).to(device=device, dtype=torch.float32)

    pair_dir = args.pair_index_dir or (args.out_dir / "pair_index")
    (
        train_pairs,
        val_pairs,
        test_pairs,
        val_support_index,
        val_support,
        test_support_index,
        test_support,
    ) = _build_or_load_pairs(
        pair_dir=pair_dir,
        bases=bases,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        args=args,
    )
    train_dataset = PairedDynamicWorldDataset(
        bases["train"],
        pair_index=train_pairs.pair_index,
        pair_valid=train_pairs.pair_valid,
        pair_distance=train_pairs.pair_distance,
        action_distance=train_pairs.action_distance,
        future_distance=train_pairs.future_distance,
    )
    val_dataset = PairedDynamicWorldDataset(
        bases["val"],
        pair_index=val_pairs.pair_index,
        pair_valid=val_pairs.pair_valid,
        pair_distance=val_pairs.pair_distance,
        action_distance=val_pairs.action_distance,
        future_distance=val_pairs.future_distance,
        support_distance=val_support,
        support_base=bases["train"],
        support_index=val_support_index,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
        device=device,
    )

    loss_config = LatentWorldLossConfig(
        **{field: getattr(args, field) for field in LatentWorldLossConfig.__dataclass_fields__}
    )
    trainer = LatentWorldTrainerConfig(
        epochs=args.epochs,
        perceiver_lr=args.perceiver_lr,
        dynamics_lr=args.dynamics_lr,
        auxiliary_lr=args.auxiliary_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        action_warmup_steps=args.action_warmup_steps,
        stability_warmup_steps=args.stability_warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay_start=args.ema_decay_start,
        ema_decay_end=args.ema_decay_end,
        camera_drop_prob=args.camera_drop_prob,
        state_mask_prob=args.state_mask_prob,
        patch_mask_prob=args.patch_mask_prob,
        checkpoint_predictive_slack=args.checkpoint_predictive_slack,
        checkpoint_hold_ratio_max=args.checkpoint_hold_ratio_max,
        checkpoint_min_embedding_std=args.checkpoint_min_embedding_std,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        eval_ablation_batches=args.eval_ablation_batches,
    )
    context = {
        "schema": "clearvla-v34.1-latent-world-context-v1",
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
        "train_pair_valid_fraction": float(train_pairs.pair_valid.mean()),
        "val_pair_valid_fraction": float(val_pairs.pair_valid.mean()),
        "test_pair_valid_fraction": float(test_pairs.pair_valid.mean()),
        "pair_index_dir": pair_dir,
        "future_seconds": [float(offset) / float(args.control_hz) for offset in future_offsets],
        "parameter_report": model.parameter_report(),
        "architecture_contract": (
            "direct full-evidence action-independent World Perceiver; dense local-interval full-width "
            "Latent Dynamics shared by actual/hold; metadata is key-only and every value/output comes from latent content"
        ),
        "training_contract": (
            "one model, one optimizer and one continuous run; no representation checkpoint, prior phase, "
            "effect phase or alignment phase"
        ),
    }
    print_context(context)
    train_latent_world(
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
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        context=context,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
