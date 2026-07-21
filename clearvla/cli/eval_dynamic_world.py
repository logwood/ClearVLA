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
from clearvla.experiments.legacy_v33.dynamic_world_lab.model import (
    DynamicPredictiveWorld,
    DynamicPredictiveWorldConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.objectives import DynamicWorldLossConfig
from clearvla.experiments.dynamic_world_lab.pairing import LocalPairTable
from clearvla.experiments.legacy_v33.dynamic_world_lab.runtime import evaluate_dynamic_world


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a V33.4 dynamic-predictive world checkpoint")
    add_data_args(p, default_resize=(336, 336))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pair-index-dir", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    p.add_argument("--dinov2-model", default="facebook/dinov2-base")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--ablation-batches", type=int, default=64)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "clearvla-v33.4-dynamic-predictive-world-checkpoint-v3":
        raise ValueError(
            "expected a V33.4 dynamic-predictive World checkpoint-v3; "
            f"got schema={checkpoint.get('schema')!r}"
        )
    model_config = DynamicPredictiveWorldConfig(**checkpoint["model_config"])
    context = checkpoint["context"]
    saved_args = context.get("args", {})
    current_args = vars(args)
    input_fields = (
        "cameras",
        "dinov2_model",
        "cache_resize",
        "cache_crop",
        "action_key",
        "state_key",
        "top_key",
        "wrist_key",
    )
    input_mismatches = {}
    for field in input_fields:
        saved = saved_args.get(field)
        current = current_args.get(field)
        if field in {"cameras", "cache_resize", "cache_crop"}:
            saved = None if saved is None else tuple(saved)
            current = None if current is None else tuple(current)
        if saved != current:
            input_mismatches[field] = (saved, current)
    if input_mismatches:
        raise ValueError(f"evaluation input contract mismatch: {input_mismatches}")
    dataset_config = DynamicWorldDatasetConfig(**context["dataset"])
    state_normalizer = ArrayNormalizer.from_dict(checkpoint["state_normalizer"])
    action_normalizer = ArrayNormalizer.from_dict(checkpoint["action_normalizer"])
    splits = context["splits"]
    max_extent = max(
        model_config.action_horizon,
        max(model_config.future_offsets) - min(dataset_config.target_history_offsets),
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
    cameras = tuple(str(x) for x in args.cameras)
    ids = val_ids if args.split == "val" else test_ids
    effective_dataset_config = DynamicWorldDatasetConfig(
        **{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"}
    )
    train_base = DynamicWorldWindowDataset(
        episodes,
        train_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=effective_dataset_config,
    )
    base = DynamicWorldWindowDataset(
        episodes,
        ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=effective_dataset_config,
    )
    pair_name = "val_local_pairs.npz" if args.split == "val" else "test_local_pairs.npz"
    support_name = (
        "val_support_distance.npy" if args.split == "val" else "test_support_distance.npy"
    )
    support_index_name = (
        "val_support_index.npy" if args.split == "val" else "test_support_index.npy"
    )
    pair_path = args.pair_index_dir / pair_name
    support_path = args.pair_index_dir / support_name
    support_index_path = args.pair_index_dir / support_index_name
    manifest_path = args.pair_index_dir / "pair_index_manifest.json"
    required_paths = [pair_path, support_path, support_index_path, manifest_path]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "formal evaluation requires the exact saved pair/support index contract; "
            f"missing={missing}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "clearvla-v33.4-local-pair-index-v1":
        raise ValueError(f"unsupported pair-index schema={manifest.get('schema')!r}")
    representation = context.get("representation", {})
    expected_manifest = {
        "representation_sha256": representation.get("sha256"),
        "train_windows": len(train_base),
        f"{args.split}_windows": len(base),
        "history_length": model_config.history_length,
        "future_offsets": list(model_config.future_offsets),
        "action_horizon": model_config.action_horizon,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_manifest.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"pair/support index contract mismatch: {mismatches}")
    pair_table = LocalPairTable.load(pair_path)
    support = np.load(support_path)
    support_index = np.load(support_index_path)
    dataset = PairedDynamicWorldDataset(
        base,
        pair_index=pair_table.pair_index,
        pair_valid=pair_table.pair_valid,
        pair_distance=pair_table.pair_distance,
        action_distance=pair_table.action_distance,
        support_distance=support,
        support_base=train_base,
        support_index=support_index,
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
        raise ValueError("conditioner patch count does not match checkpoint")
    model = DynamicPredictiveWorld(model_config).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_representation()
    loss_config = DynamicWorldLossConfig(**checkpoint["loss_config"])
    metrics = evaluate_dynamic_world(
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
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
