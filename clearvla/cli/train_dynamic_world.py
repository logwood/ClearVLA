from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import random
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
    serializable,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import (
    build_dense_conditioner,
    infer_dense_geometry,
)
from clearvla.experiments.dynamic_world_lab.dataset import (
    DynamicWorldDatasetConfig,
    DynamicWorldWindowDataset,
    PairedDynamicWorldDataset,
    CurrentHistoryViewDataset,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.model import (
    DynamicPredictiveWorld,
    DynamicPredictiveWorldConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.objectives import DynamicWorldLossConfig
from clearvla.experiments.dynamic_world_lab.pairing import (
    LocalPairTable,
    build_local_pair_table,
    nearest_support,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.runtime import (
    DynamicWorldTrainerConfig,
    encode_current_tokens,
    train_dynamic_world,
)



_REPRESENTATION_CONFIG_FIELDS = (
    "latent_dim",
    "action_dim",
    "state_dim",
    "action_horizon",
    "history_length",
    "num_cameras",
    "patches_per_camera",
    "future_offsets",
    "hidden_size",
    "encoder_depth",
    "num_heads",
    "context_tokens",
    "dynamic_tokens",
    "descriptor_projection_dim",
    "dropout",
    "gripper_dim_index",
    "descriptor_seed",
)

_REPRESENTATION_DATASET_FIELDS = (
    "action_horizon",
    "history_offsets",
    "future_offsets",
    "target_history_offsets",
    "state_offset",
    "image_offset",
    "action_offset",
    "stride",
)

_REPRESENTATION_INPUT_FIELDS = (
    "cameras",
    "dinov2_model",
    "cache_resize",
    "cache_crop",
    "action_key",
    "state_key",
    "top_key",
    "wrist_key",
)


def _load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_representation_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"representation checkpoint not found: {checkpoint_path}")
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    if checkpoint.get("schema") != "clearvla-v33.4-dynamic-representation-checkpoint-v1":
        raise ValueError(
            "predictor requires a V33.4 action-independent representation checkpoint; "
            f"got schema={checkpoint.get('schema')!r}"
        )
    required = {
        "representation",
        "model_config",
        "context",
        "action_normalizer",
        "state_normalizer",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"representation checkpoint is missing fields: {missing}")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    provenance = {
        "path": str(checkpoint_path.resolve()),
        "sha256": digest,
        "schema": checkpoint["schema"],
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
    }
    return checkpoint, provenance


def _validate_representation_data_contract(
    *,
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    dataset_config: DynamicWorldDatasetConfig,
) -> tuple[ArrayNormalizer, ArrayNormalizer, dict[str, list[int]]]:
    context = checkpoint["context"]
    if not isinstance(context, dict):
        raise TypeError("representation checkpoint context must be a dictionary")
    saved_dataset = context.get("dataset")
    saved_args = context.get("args")
    saved_splits = context.get("splits")
    if not isinstance(saved_dataset, dict) or not isinstance(saved_args, dict):
        raise KeyError("representation checkpoint is missing dataset/args contracts")
    if not isinstance(saved_splits, dict):
        raise KeyError("representation checkpoint is missing fixed episode splits")

    current_dataset = asdict(dataset_config)
    dataset_mismatches = {
        field: (saved_dataset.get(field), current_dataset.get(field))
        for field in _REPRESENTATION_DATASET_FIELDS
        if saved_dataset.get(field) != current_dataset.get(field)
    }
    if dataset_mismatches:
        raise ValueError(
            "predictor temporal dataset contract differs from representation pretraining: "
            f"{dataset_mismatches}"
        )

    current_args = vars(args)
    input_mismatches = {}
    for field in _REPRESENTATION_INPUT_FIELDS:
        saved = saved_args.get(field)
        current = current_args.get(field)
        if field in {"cameras", "cache_resize", "cache_crop"}:
            saved = None if saved is None else tuple(saved)
            current = None if current is None else tuple(current)
        if saved != current:
            input_mismatches[field] = (saved, current)
    if input_mismatches:
        raise ValueError(
            "predictor visual/input contract differs from representation pretraining: "
            f"{input_mismatches}"
        )

    action_normalizer = ArrayNormalizer.from_dict(checkpoint["action_normalizer"])
    state_normalizer = ArrayNormalizer.from_dict(checkpoint["state_normalizer"])
    if args.normalizer != action_normalizer.mode or args.normalizer != state_normalizer.mode:
        raise ValueError(
            "--normalizer must match the representation checkpoint exactly: "
            f"requested={args.normalizer!r}, action={action_normalizer.mode!r}, "
            f"state={state_normalizer.mode!r}"
        )
    splits = {
        name: [int(index) for index in saved_splits[name]]
        for name in ("train", "val", "test")
    }
    return action_normalizer, state_normalizer, splits


def _load_frozen_representation(
    model: DynamicPredictiveWorld,
    checkpoint: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    saved_config = checkpoint["model_config"]
    current_config = asdict(model.config)
    mismatches = {
        field: (saved_config.get(field), current_config.get(field))
        for field in _REPRESENTATION_CONFIG_FIELDS
        if saved_config.get(field) != current_config.get(field)
    }
    if mismatches:
        raise ValueError(f"representation geometry/config mismatch: {mismatches}")
    model.load_representation_state_dict(checkpoint["representation"], freeze=True)
    return dict(provenance)

def _parse_offsets(text: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in text)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the V33.4 standalone dynamic-predictive world model. "
            "This entry point never constructs or updates a policy."
        )
    )
    add_data_args(p, default_resize=(336, 336))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--representation-checkpoint",
        type=Path,
        required=True,
        help=(
            "Action-independent representation checkpoint produced by "
            "clearvla.cli.train_dynamic_representation. All predictor baselines "
            "must use the same file."
        ),
    )
    p.add_argument("--normalizer", choices=["identity", "limits", "zscore"], default="zscore")
    p.add_argument("--action-horizon", type=int, default=48)
    p.add_argument("--history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    p.add_argument("--future-offsets", nargs="+", type=int, default=[8, 24, 48])
    p.add_argument("--target-history-offsets", nargs="+", type=int, default=[-8, -4, 0])
    p.add_argument("--state-offset", type=int, default=0)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--action-offset", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--control-hz", type=float, default=30.0, help="Reporting only; indices remain explicit")

    p.add_argument(
        "--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache"
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

    p.add_argument("--predictive-weight", type=float, default=1.0)
    p.add_argument("--scene-predictive-weight", type=float, default=0.25)
    p.add_argument("--direction-weight", type=float, default=0.25)
    p.add_argument("--amplitude-weight", type=float, default=0.10)
    p.add_argument("--increment-weight", type=float, default=0.50)
    p.add_argument("--scene-increment-weight", type=float, default=0.10)
    p.add_argument("--teacher-forced-weight", type=float, default=0.25)
    p.add_argument("--scene-teacher-forced-weight", type=float, default=0.10)
    p.add_argument("--descriptor-weight", type=float, default=0.50)
    p.add_argument("--encoder-anchor-weight", type=float, default=0.0)
    p.add_argument("--state-path-weight", type=float, default=0.10)
    p.add_argument("--local-effect-weight", type=float, default=0.25)
    p.add_argument("--local-effect-direction-weight", type=float, default=0.10)
    p.add_argument("--swap-rank-weight", type=float, default=0.0)
    p.add_argument("--swap-margin", type=float, default=0.02)
    p.add_argument("--variance-weight", type=float, default=0.0)
    p.add_argument("--embedding-std-target", type=float, default=0.05)
    p.add_argument("--gripper-transition-boost", type=float, default=3.0)
    p.add_argument("--gripper-transition-threshold", type=float, default=0.10)
    p.add_argument("--gripper-transition-radius", type=int, default=1)

    p.add_argument("--pair-index-dir", type=Path, default=None)
    p.add_argument("--pair-candidates", type=int, default=64)
    p.add_argument("--pair-min-action-distance", type=float, default=1.0)
    p.add_argument("--rebuild-pairs", action="store_true")
    p.add_argument("--pair-build-batch-size", type=int, default=32)

    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
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
    model: DynamicPredictiveWorld,
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
    condition_rows, action_rows, episode_rows, gripper_rows = [], [], [], []
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
        static = torch.nn.functional.normalize(static, dim=-1).reshape(len(current), -1).cpu().numpy()
        state = batch["state"].numpy().reshape(len(current), -1)
        condition_rows.append(np.concatenate([state, static, dynamic_descriptor], axis=1))

        action = batch["action"].numpy()
        boundary = np.concatenate([state[:, None, :], action[:, :-1]], axis=1)
        velocity = action - boundary
        sampled = action[:, np.asarray(model.config.future_offsets) - 1]
        action_rows.append(
            np.concatenate(
                [sampled.reshape(len(action), -1), velocity.mean(1), velocity.std(1), action[:, -1] - state],
                axis=1,
            )
        )
        episode_rows.append(batch["episode_idx"].numpy())
        raw_g = batch["state_raw"].numpy()[:, model.config.gripper_index]
        gripper_rows.append((raw_g > float(gripper_midpoint)).astype(np.int64))
    return (
        np.concatenate(condition_rows),
        np.concatenate(action_rows),
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
    train_path = pair_dir / "train_local_pairs.npz"
    val_path = pair_dir / "val_local_pairs.npz"
    support_path = pair_dir / "val_support_distance.npy"
    support_index_path = pair_dir / "val_support_index.npy"
    test_path = pair_dir / "test_local_pairs.npz"
    test_support_path = pair_dir / "test_support_distance.npy"
    test_support_index_path = pair_dir / "test_support_index.npy"
    descriptors_path = pair_dir / "pair_descriptors.npz"
    manifest_path = pair_dir / "pair_index_manifest.json"
    expected_manifest = {
        "schema": "clearvla-v33.4-local-pair-index-v1",
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
        "gripper_midpoint": float(
            0.5 * (args.gripper_open_value + args.gripper_close_value)
        ),
    }
    cache_files_exist = (
        train_path.exists()
        and val_path.exists()
        and support_path.exists()
        and support_index_path.exists()
        and test_path.exists()
        and test_support_path.exists()
        and test_support_index_path.exists()
        and manifest_path.exists()
    )
    if not args.rebuild_pairs and cache_files_exist:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest == expected_manifest:
            return (
                LocalPairTable.load(train_path), LocalPairTable.load(val_path),
                LocalPairTable.load(test_path), np.load(support_index_path),
                np.load(support_path), np.load(test_support_index_path),
                np.load(test_support_path)
            )
        print(
            "[dynamic-world] pair index manifest is stale; rebuilding pair/support indices",
            flush=True,
        )

    train_desc, train_action, train_episode, train_gripper = _pair_descriptors(
        train_dataset,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        batch_size=args.pair_build_batch_size,
        workers=args.num_workers,
        gripper_midpoint=0.5 * (args.gripper_open_value + args.gripper_close_value),
    )
    val_desc, val_action, val_episode, val_gripper = _pair_descriptors(
        val_dataset,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        batch_size=args.pair_build_batch_size,
        workers=args.num_workers,
        gripper_midpoint=0.5 * (args.gripper_open_value + args.gripper_close_value),
    )
    test_desc, test_action, test_episode, test_gripper = _pair_descriptors(
        test_dataset,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        batch_size=args.pair_build_batch_size,
        workers=args.num_workers,
        gripper_midpoint=0.5 * (args.gripper_open_value + args.gripper_close_value),
    )
    train_table = build_local_pair_table(
        condition_descriptor=train_desc,
        action_summary=train_action,
        episode_ids=train_episode,
        gripper_state=train_gripper,
        candidate_count=args.pair_candidates,
        min_action_distance=args.pair_min_action_distance,
    )
    val_table = build_local_pair_table(
        condition_descriptor=val_desc,
        action_summary=val_action,
        episode_ids=val_episode,
        gripper_state=val_gripper,
        candidate_count=args.pair_candidates,
        min_action_distance=args.pair_min_action_distance,
    )
    test_table = build_local_pair_table(
        condition_descriptor=test_desc,
        action_summary=test_action,
        episode_ids=test_episode,
        gripper_state=test_gripper,
        candidate_count=args.pair_candidates,
        min_action_distance=args.pair_min_action_distance,
    )
    support_index, support = nearest_support(
        query_descriptor=val_desc, reference_descriptor=train_desc
    )
    test_support_index, test_support = nearest_support(
        query_descriptor=test_desc, reference_descriptor=train_desc
    )
    train_table.save(train_path)
    val_table.save(val_path)
    test_table.save(test_path)
    np.save(support_index_path, support_index)
    np.save(support_path, support)
    np.save(test_support_index_path, test_support_index)
    np.save(test_support_path, test_support)
    np.savez_compressed(
        descriptors_path,
        train_condition=train_desc,
        train_action=train_action,
        val_condition=val_desc,
        val_action=val_action,
        test_condition=test_desc,
        test_action=test_action,
    )
    manifest_path.write_text(
        json.dumps(expected_manifest, indent=2), encoding="utf-8"
    )
    return (
        train_table, val_table, test_table, support_index, support,
        test_support_index, test_support
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
    (
        checkpoint_action_normalizer,
        checkpoint_state_normalizer,
        checkpoint_splits,
    ) = _validate_representation_data_contract(
        checkpoint=representation_checkpoint,
        args=args,
        dataset_config=dataset_config,
    )
    max_extent = max(
        args.action_horizon,
        max(future_offsets) - min(target_history_offsets),
    )
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
        action_normalizer=checkpoint_action_normalizer,
        state_normalizer=checkpoint_state_normalizer,
        splits=checkpoint_splits,
    )
    train_base = DynamicWorldWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    val_base = DynamicWorldWindowDataset(
        episodes,
        val_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    test_base = DynamicWorldWindowDataset(
        episodes,
        test_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
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
            conditioner, train_base[0], camera_names=cameras
        )
    model_config = DynamicPredictiveWorldConfig(
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
    )
    model = DynamicPredictiveWorld(model_config).to(device=device, dtype=dtype)
    representation_provenance = _load_frozen_representation(
        model, representation_checkpoint, representation_provenance
    )
    if args.encoder_anchor_weight != 0.0 or args.variance_weight != 0.0:
        raise ValueError(
            "predictor training freezes the shared representation; "
            "--encoder-anchor-weight and --variance-weight must both be 0"
        )

    pair_dir = args.pair_index_dir or (args.out_dir / "pair_index")
    (
        train_pairs, val_pairs, test_pairs, val_support_index, val_support,
        test_support_index, test_support,
    ) = _build_or_load_pairs(
        pair_dir=pair_dir,
        train_dataset=train_base,
        val_dataset=val_base,
        test_dataset=test_base,
        conditioner=conditioner,
        model=model,
        cameras=cameras,
        device=device,
        dtype=dtype,
        args=args,
        representation_provenance=representation_provenance,
    )
    train_dataset = PairedDynamicWorldDataset(
        train_base,
        pair_index=train_pairs.pair_index,
        pair_valid=train_pairs.pair_valid,
        pair_distance=train_pairs.pair_distance,
        action_distance=train_pairs.action_distance,
    )
    val_dataset = PairedDynamicWorldDataset(
        val_base,
        pair_index=val_pairs.pair_index,
        pair_valid=val_pairs.pair_valid,
        pair_distance=val_pairs.pair_distance,
        action_distance=val_pairs.action_distance,
        support_distance=val_support,
        support_base=train_base,
        support_index=val_support_index,
    )
    train_loader = make_loader(
        train_dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device
    )
    val_loader = make_loader(
        val_dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device
    )

    loss_config = DynamicWorldLossConfig(
        predictive_weight=args.predictive_weight,
        scene_predictive_weight=args.scene_predictive_weight,
        direction_weight=args.direction_weight,
        amplitude_weight=args.amplitude_weight,
        increment_weight=args.increment_weight,
        scene_increment_weight=args.scene_increment_weight,
        teacher_forced_weight=args.teacher_forced_weight,
        scene_teacher_forced_weight=args.scene_teacher_forced_weight,
        descriptor_weight=args.descriptor_weight,
        encoder_anchor_weight=args.encoder_anchor_weight,
        state_path_weight=args.state_path_weight,
        local_effect_weight=args.local_effect_weight,
        local_effect_direction_weight=args.local_effect_direction_weight,
        swap_rank_weight=args.swap_rank_weight,
        swap_margin=args.swap_margin,
        variance_weight=args.variance_weight,
        embedding_std_target=args.embedding_std_target,
        gripper_transition_boost=args.gripper_transition_boost,
        gripper_transition_threshold=args.gripper_transition_threshold,
        gripper_transition_radius=args.gripper_transition_radius,
    )
    trainer = DynamicWorldTrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.adam_eps,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        eval_ablation_batches=args.eval_ablation_batches,
    )
    context = {
        "schema": "clearvla-v33.4-dynamic-predictive-world-context-v3",
        "args": serializable(vars(args)),
        "splits": {"train": train_ids, "val": val_ids, "test": test_ids},
        "skipped": skipped,
        "dataset": asdict(dataset_config),
        "model": asdict(model_config),
        "loss": asdict(loss_config),
        "trainer": asdict(trainer),
        "train_windows": len(train_base),
        "val_windows": len(val_base),
        "test_windows": len(test_base),
        "train_pair_valid_fraction": float(train_pairs.pair_valid.mean()),
        "val_pair_valid_fraction": float(val_pairs.pair_valid.mean()),
        "test_pair_valid_fraction": float(test_pairs.pair_valid.mean()),
        "future_seconds": [float(x / args.control_hz) for x in future_offsets],
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "representation": representation_provenance,
        "representation_contract": (
            "the temporal visual representation is loaded from one action-independent checkpoint, "
            "frozen, and shared unchanged by full/current-only/action-only predictors"
        ),
        "closed_loop_contract": (
            "world-internal autoregressive rollout of both compact scene and dynamics states; "
            "no policy construction, no policy gradient, and teacher forcing is diagnostic only"
        ),
    }
    print_context(context)
    train_dynamic_world(
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
