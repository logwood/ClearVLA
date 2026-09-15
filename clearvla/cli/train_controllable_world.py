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

from clearvla.cli.train_dynamic_world import (
    _load_frozen_representation,
    _read_representation_checkpoint,
    _validate_representation_data_contract,
)
from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    print_context,
    resolve_device,
    serializable,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.controllable_model import (
    ControllableDynamicWorld,
    ControllableWorldConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.controllable_objectives import (
    ControllableWorldLossConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.controllable_runtime import (
    ControllableWorldTrainerConfig,
    train_controllable_world,
)
from clearvla.experiments.dynamic_world_lab.dataset import (
    CurrentHistoryViewDataset,
    DynamicWorldDatasetConfig,
    DynamicWorldWindowDataset,
    PairedDynamicWorldDataset,
)
from clearvla.experiments.dynamic_world_lab.pairing import (
    LocalPairTable,
    build_local_pair_table,
    nearest_support,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.runtime import encode_current_tokens


def _parse_offsets(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train V33.6 joint-AdaLN controllable world modelling with an action-free prior, "
            "deep full-token world/action modulation, inverse alignment, and EMA targets."
        )
    )
    add_data_args(p, default_resize=(336, 336))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--representation-checkpoint", type=Path, required=True)
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")
    p.add_argument("--action-horizon", type=int, default=48)
    p.add_argument("--history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    p.add_argument("--future-offsets", nargs="+", type=int, default=[8, 24, 48])
    p.add_argument("--target-history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--control-hz", type=float, default=30.0)

    p.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    p.add_argument("--dinov2-model", default="facebook/dinov2-base")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--debug-token-dim", type=int, default=64)
    p.add_argument("--debug-patches-per-camera", type=int, default=16)

    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=3)
    p.add_argument("--predictor-depth", type=int, default=3)
    p.add_argument("--action-depth", type=int, default=3)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--context-tokens", type=int, default=8)
    p.add_argument("--dynamic-tokens", type=int, default=16)
    p.add_argument("--descriptor-projection-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--input-mode", choices=["full", "current-only", "action-only"], default="full")
    p.add_argument("--gripper-dim-index", type=int, default=-1)
    p.add_argument("--gripper-open-value", type=float, default=0.0)
    p.add_argument("--gripper-close-value", type=float, default=1.7459820890426636)

    p.add_argument("--adapter-depth", type=int, default=2)
    p.add_argument("--inverse-depth", type=int, default=2)
    p.add_argument("--prior-depth", type=int, default=3)
    p.add_argument("--effect-depth", type=int, default=3)
    p.add_argument("--adapter-layer-scale", type=float, default=0.02)
    p.add_argument("--prior-layer-scale", type=float, default=0.10)
    p.add_argument("--effect-layer-scale", type=float, default=1.0)

    p.add_argument("--predictive-weight", type=float, default=1.0)
    p.add_argument("--scene-predictive-weight", type=float, default=0.25)
    p.add_argument("--direction-weight", type=float, default=0.25)
    p.add_argument("--amplitude-weight", type=float, default=0.10)
    p.add_argument("--increment-weight", type=float, default=0.50)
    p.add_argument("--scene-increment-weight", type=float, default=0.10)
    p.add_argument("--teacher-forced-weight", type=float, default=0.20)
    p.add_argument("--descriptor-weight", type=float, default=0.40)
    p.add_argument("--state-path-weight", type=float, default=0.10)
    p.add_argument("--prior-state-path-weight", type=float, default=0.05)
    p.add_argument("--residual-weight", type=float, default=1.0)
    p.add_argument("--residual-direction-weight", type=float, default=0.25)
    p.add_argument("--necessity-weight", type=float, default=0.25)
    p.add_argument("--necessity-margin", type=float, default=0.005)
    p.add_argument("--informative-residual-threshold", type=float, default=0.02)
    p.add_argument("--inverse-action-weight", type=float, default=0.20)
    p.add_argument("--inverse-delta-weight", type=float, default=0.10)
    p.add_argument("--inverse-gripper-weight", type=float, default=0.10)
    p.add_argument("--local-effect-weight", type=float, default=0.25)
    p.add_argument("--local-effect-direction-weight", type=float, default=0.15)
    p.add_argument("--swap-rank-weight", type=float, default=0.05)
    p.add_argument("--swap-margin", type=float, default=0.01)
    p.add_argument("--representation-anchor-weight", type=float, default=0.20)
    p.add_argument("--adapter-delta-weight", type=float, default=0.005)
    p.add_argument("--variance-weight", type=float, default=0.02)
    p.add_argument("--embedding-std-target", type=float, default=0.05)
    p.add_argument("--gripper-transition-boost", type=float, default=3.0)
    p.add_argument("--gripper-transition-threshold", type=float, default=0.10)
    p.add_argument("--gripper-transition-radius", type=int, default=1)

    p.add_argument("--pair-index-dir", type=Path, default=None)
    p.add_argument("--pair-candidates", type=int, default=96)
    p.add_argument("--pair-min-action-distance", type=float, default=1.0)
    p.add_argument("--pair-min-future-distance", type=float, default=0.75)
    p.add_argument("--rebuild-pairs", action="store_true")
    p.add_argument("--pair-build-batch-size", type=int, default=32)

    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--prior-warmup-epochs", type=int, default=2)
    p.add_argument("--effect-warmup-epochs", type=int, default=5)
    p.add_argument("--prior-lr", type=float, default=1e-4)
    p.add_argument("--effect-lr", type=float, default=1e-4)
    p.add_argument("--adapter-lr", type=float, default=1e-5)
    p.add_argument("--encoder-lr", type=float, default=2e-6)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--unfreeze-dynamic-blocks", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--eval-ablation-batches", type=int, default=64)
    return p.parse_args()


@torch.no_grad()
def _pair_descriptors(
    dataset: DynamicWorldWindowDataset,
    *,
    conditioner,
    model: ControllableDynamicWorld,
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
        condition_rows.append(np.concatenate([state, static, dynamic_descriptor], axis=1))

        action = batch["action"].numpy()
        boundary = np.concatenate([state[:, None], action[:, :-1]], axis=1)
        velocity = action - boundary
        sampled = action[:, offsets]
        action_rows.append(
            np.concatenate(
                [
                    sampled.reshape(len(action), -1),
                    velocity.mean(1),
                    velocity.std(1),
                    action[:, -1] - state,
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
    return (
        np.concatenate(condition_rows),
        np.concatenate(action_rows),
        np.concatenate(future_rows),
        np.concatenate(episode_rows),
        np.concatenate(gripper_rows),
    )


def _build_or_load_pairs(
    *,
    pair_dir: Path,
    train_dataset,
    val_dataset,
    test_dataset,
    conditioner,
    model,
    cameras,
    device,
    dtype,
    args,
    representation_provenance,
):
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": pair_dir / "train_local_pairs.npz",
        "val": pair_dir / "val_local_pairs.npz",
        "test": pair_dir / "test_local_pairs.npz",
        "val_support": pair_dir / "val_support_distance.npy",
        "val_support_index": pair_dir / "val_support_index.npy",
        "test_support": pair_dir / "test_support_distance.npy",
        "test_support_index": pair_dir / "test_support_index.npy",
        "descriptors": pair_dir / "pair_descriptors.npz",
        "manifest": pair_dir / "pair_index_manifest.json",
    }
    expected = {
        "schema": "clearvla-v33.6-local-pair-index-v1",
        "representation_sha256": representation_provenance["sha256"],
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "train_episode_ids": list(train_dataset.episode_ids),
        "val_episode_ids": list(val_dataset.episode_ids),
        "test_episode_ids": list(test_dataset.episode_ids),
        "history_length": model.config.history_length,
        "future_offsets": list(model.config.future_offsets),
        "action_horizon": model.config.action_horizon,
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
        print("[controllable-world] stale pair manifest; rebuilding", flush=True)

    midpoint = 0.5 * (args.gripper_open_value + args.gripper_close_value)
    rows = {}
    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        rows[name] = _pair_descriptors(
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
    np.savez_compressed(
        paths["descriptors"],
        train_condition=rows["train"][0],
        train_action=rows["train"][1],
        train_future=rows["train"][2],
        val_condition=rows["val"][0],
        val_action=rows["val"][1],
        val_future=rows["val"][2],
        test_condition=rows["test"][0],
        test_action=rows["test"][1],
        test_future=rows["test"][2],
    )
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
    cameras = tuple(str(x) for x in args.cameras)
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

    representation_checkpoint, representation_provenance = _read_representation_checkpoint(
        args.representation_checkpoint
    )
    action_normalizer, state_normalizer, splits = _validate_representation_data_contract(
        checkpoint=representation_checkpoint, args=args, dataset_config=dataset_config
    )
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
    ) = load_data(
        args,
        min_length=min_length,
        normalizer_mode=args.normalizer,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        splits=splits,
    )
    bases = {
        "train": DynamicWorldWindowDataset(
            episodes,
            train_ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=dataset_config,
        ),
        "val": DynamicWorldWindowDataset(
            episodes,
            val_ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=dataset_config,
        ),
        "test": DynamicWorldWindowDataset(
            episodes,
            test_ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=dataset_config,
        ),
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

    model_config = ControllableWorldConfig(
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
        predictor_depth=args.predictor_depth,
        action_depth=args.action_depth,
        num_heads=args.heads,
        context_tokens=args.context_tokens,
        dynamic_tokens=args.dynamic_tokens,
        descriptor_projection_dim=args.descriptor_projection_dim,
        dropout=args.dropout,
        input_mode=args.input_mode,
        gripper_dim_index=args.gripper_dim_index,
        adapter_depth=args.adapter_depth,
        inverse_depth=args.inverse_depth,
        prior_depth=args.prior_depth,
        effect_depth=args.effect_depth,
        adapter_layer_scale=args.adapter_layer_scale,
        prior_layer_scale=args.prior_layer_scale,
        effect_layer_scale=args.effect_layer_scale,
    )
    model = ControllableDynamicWorld(model_config).to(device=device, dtype=dtype)
    representation_provenance = _load_frozen_representation(
        model, representation_checkpoint, representation_provenance
    )

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
        train_dataset=bases["train"],
        val_dataset=bases["val"],
        test_dataset=bases["test"],
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        args=args,
        representation_provenance=representation_provenance,
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

    loss_config = ControllableWorldLossConfig(
        predictive_weight=args.predictive_weight,
        scene_predictive_weight=args.scene_predictive_weight,
        direction_weight=args.direction_weight,
        amplitude_weight=args.amplitude_weight,
        increment_weight=args.increment_weight,
        scene_increment_weight=args.scene_increment_weight,
        teacher_forced_weight=args.teacher_forced_weight,
        descriptor_weight=args.descriptor_weight,
        state_path_weight=args.state_path_weight,
        prior_state_path_weight=args.prior_state_path_weight,
        residual_weight=args.residual_weight,
        residual_direction_weight=args.residual_direction_weight,
        necessity_weight=args.necessity_weight,
        necessity_margin=args.necessity_margin,
        informative_residual_threshold=args.informative_residual_threshold,
        inverse_action_weight=args.inverse_action_weight,
        inverse_delta_weight=args.inverse_delta_weight,
        inverse_gripper_weight=args.inverse_gripper_weight,
        local_effect_weight=args.local_effect_weight,
        local_effect_direction_weight=args.local_effect_direction_weight,
        swap_rank_weight=args.swap_rank_weight,
        swap_margin=args.swap_margin,
        representation_anchor_weight=args.representation_anchor_weight,
        adapter_delta_weight=args.adapter_delta_weight,
        variance_weight=args.variance_weight,
        embedding_std_target=args.embedding_std_target,
        gripper_transition_boost=args.gripper_transition_boost,
        gripper_transition_threshold=args.gripper_transition_threshold,
        gripper_transition_radius=args.gripper_transition_radius,
    )
    trainer = ControllableWorldTrainerConfig(
        epochs=args.epochs,
        prior_warmup_epochs=args.prior_warmup_epochs,
        effect_warmup_epochs=args.effect_warmup_epochs,
        prior_lr=args.prior_lr,
        effect_lr=args.effect_lr,
        adapter_lr=args.adapter_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay=args.ema_decay,
        unfreeze_dynamic_blocks=args.unfreeze_dynamic_blocks,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        eval_ablation_batches=args.eval_ablation_batches,
    )
    context = {
        "schema": "clearvla-v33.6-controllable-world-context-v1",
        "args": serializable(vars(args)),
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
        "future_seconds": [float(x / args.control_hz) for x in future_offsets],
        "parameter_count": model.parameter_count(),
        "representation": representation_provenance,
        "anti_shortcut_contract": (
            "the current-only prior never receives action and is frozen before action-effect training. "
            "actual-minus-hold action tokens are the only action source. every AdaLN shift/scale/gate "
            "is produced by a bias-free bilinear product of full world content and an action-derived "
            "signal; hold action and zero world content therefore yield exact zero residual. no target "
            "future enters prediction."
        ),
        "expression_contract": (
            "all 48 action tokens, all scene/dynamics tokens, and hidden width are preserved. each effect "
            "block first conditions action tokens on the complete world, then conditions the complete world "
            "on those action tokens, and applies per-token AdaLN-Zero self-attention/FFN modulation. the "
            "full-width adapter remains identity-residual, action-independent, and EMA-targeted."
        ),
        "alignment_contract": (
            "prior warmup -> frozen-prior action residual -> low-LR adapter/last-dynamic-block alignment; "
            "future target encoder/adapter are EMA stop-gradient targets."
        ),
    }
    print_context(context)
    train_controllable_world(
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
    )


if __name__ == "__main__":
    main()
