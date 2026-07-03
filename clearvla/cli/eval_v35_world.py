from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args, load_data, make_loader, preprocessing_from_args, resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.dynamic_world_lab.dataset import PairedDynamicWorldDataset
from clearvla.experiments.dynamic_world_lab.pairing import LocalPairTable
from clearvla.experiments.observed_state_lab.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset
from clearvla.experiments.observed_state_lab.world_model import V35ObservedStateWorldModel, V35WorldConfig
from clearvla.experiments.observed_state_lab.world_objectives import V35WorldLossConfig
from clearvla.experiments.observed_state_lab.world_runtime import evaluate_v35_world, jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V35 world checkpoint.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--ablation-batches", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("unsupported V35 checkpoint")
    model_config = V35WorldConfig(**checkpoint["model_config"])
    context = checkpoint["context"]
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    action_norm = ArrayNormalizer.from_dict(checkpoint["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(checkpoint["state_normalizer"])
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)
    min_length = dataset_config.world_horizon + abs(
        min(dataset_config.history_offsets + dataset_config.executed_action_offsets)
    ) + 2
    episodes, train_ids, val_ids, test_ids, _, _, image_store, _ = load_data(
        args, min_length=min_length, normalizer_mode=action_norm.mode,
        action_normalizer=action_norm, state_normalizer=state_norm, splits=context["splits"],
    )
    effective = ObservedStateDatasetConfig(**{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    ids = val_ids if args.split == "val" else test_ids
    base = ObservedStateWindowDataset(
        episodes, ids, image_store=image_store, camera_names=cameras,
        state_normalizer=state_norm, action_normalizer=action_norm, config=effective,
    )
    manifest_path = args.pair_index_dir / "pair_index_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "clearvla-v35-local-pair-index-v1",
        "train_windows": int(context["train_windows"]),
        "val_windows": int(context["val_windows"]),
        "test_windows": int(context["test_windows"]),
        "segment_offsets": list(dataset_config.future_offsets),
        "world_horizon": int(dataset_config.world_horizon),
    }
    if manifest != expected_manifest:
        raise ValueError("pair-index manifest does not match the checkpoint dataset contract")
    pair_path = args.pair_index_dir / f"{args.split}_local_pairs.npz"
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    table = LocalPairTable.load(pair_path)
    dataset = PairedDynamicWorldDataset(
        base, pair_index=table.pair_index, pair_valid=table.pair_valid,
        pair_distance=table.pair_distance, action_distance=table.action_distance,
        future_distance=table.future_distance,
    )
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras,
        preprocessing=preprocessing_from_args(args), dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=model_config.latent_dim, debug_patches_per_camera=model_config.patches_per_camera,
        device=device, dtype=dtype,
    )
    if latent_dim != model_config.latent_dim or (patches is not None and patches != model_config.patches_per_camera):
        raise ValueError("conditioner geometry does not match checkpoint")
    model = V35ObservedStateWorldModel(model_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"], strict=True)
    loss_payload = {**V35WorldLossConfig().__dict__, **checkpoint.get("loss_config", {})}
    metrics = evaluate_v35_world(
        model=model, loader=loader, conditioner=conditioner, device=device, dtype=dtype,
        camera_names=cameras, loss_config=V35WorldLossConfig(**loss_payload),
        state_normalizer=state_norm, action_normalizer=action_norm,
        max_batches=args.max_batches, ablation_batches=args.ablation_batches,
    )
    print(json.dumps(jsonable(metrics), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
