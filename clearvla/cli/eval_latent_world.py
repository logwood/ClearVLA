from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.dynamic_world_lab.dataset import (
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
    _jsonable,
    evaluate_latent_world,
)
from clearvla.experiments.dynamic_world_lab.pairing import LocalPairTable


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one V34.1 latent-world checkpoint.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
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
    if not str(checkpoint.get("schema", "")).startswith("clearvla-v34.1"):
        raise ValueError(f"unsupported checkpoint schema={checkpoint.get('schema')!r}")
    model_config = LatentWorldConfig(**checkpoint["model_config"])
    context = checkpoint["context"]
    dataset_config = DynamicWorldDatasetConfig(**context["dataset"])
    action_normalizer = ArrayNormalizer.from_dict(checkpoint["action_normalizer"])
    state_normalizer = ArrayNormalizer.from_dict(checkpoint["state_normalizer"])
    splits = context["splits"]

    device = resolve_device(args.device)
    dtype = _dtype(args.dtype)
    cameras = tuple(str(value) for value in args.cameras)
    max_extent = max(
        dataset_config.action_horizon,
        max(dataset_config.future_offsets) - min(dataset_config.target_history_offsets),
    )
    min_length = max_extent + abs(min(dataset_config.history_offsets)) + 2
    (
        episodes,
        train_ids,
        val_ids,
        test_ids,
        _,
        _,
        image_store,
        _,
    ) = load_data(
        args,
        min_length=min_length,
        normalizer_mode=action_normalizer.mode,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        splits=splits,
    )
    effective_config = DynamicWorldDatasetConfig(
        **{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"}
    )
    train_base = DynamicWorldWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=effective_config,
    )
    ids = val_ids if args.split == "val" else test_ids
    base = DynamicWorldWindowDataset(
        episodes,
        ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=effective_config,
    )
    prefix = args.split
    pair_path = args.pair_index_dir / f"{prefix}_local_pairs.npz"
    support_path = args.pair_index_dir / f"{prefix}_support_distance.npy"
    support_index_path = args.pair_index_dir / f"{prefix}_support_index.npy"
    manifest_path = args.pair_index_dir / "pair_index_manifest.json"
    missing = [
        str(path)
        for path in (pair_path, support_path, support_index_path, manifest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing formal V34 pair/support contract: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "clearvla-v34.1-local-pair-index-v1":
        raise ValueError(f"unsupported pair-index schema={manifest.get('schema')!r}")
    expected = {
        "train_windows": len(train_base),
        f"{prefix}_windows": len(base),
        "history_length": model_config.history_length,
        "future_offsets": list(model_config.future_offsets),
        "action_horizon": model_config.action_horizon,
        "descriptor_seed": model_config.descriptor_seed,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(f"pair-index contract mismatch: {mismatch}")

    pair_table = LocalPairTable.load(pair_path)
    dataset = PairedDynamicWorldDataset(
        base,
        pair_index=pair_table.pair_index,
        pair_valid=pair_table.pair_valid,
        pair_distance=pair_table.pair_distance,
        action_distance=pair_table.action_distance,
        future_distance=pair_table.future_distance,
        support_distance=np.load(support_path),
        support_base=train_base,
        support_index=np.load(support_index_path),
    )
    loader = make_loader(
        dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device
    )
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=model_config.latent_dim,
        debug_patches_per_camera=model_config.patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if latent_dim != model_config.latent_dim:
        raise ValueError("conditioner latent dimension does not match checkpoint")
    if patches is not None and patches != model_config.patches_per_camera:
        raise ValueError("conditioner patch geometry does not match checkpoint")

    model = LatentWorldModel(model_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"], strict=True)
    loss_config = LatentWorldLossConfig(**checkpoint["loss_config"])
    metrics = evaluate_latent_world(
        model=model,
        loader=loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        loss_config=loss_config,
        state_normalizer=state_normalizer,
        max_batches=args.max_batches,
        ablation_batches=args.ablation_batches,
    )
    print(json.dumps(_jsonable(metrics), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
