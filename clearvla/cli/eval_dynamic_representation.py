from __future__ import annotations

import argparse
import json
from pathlib import Path

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
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.model import (
    DynamicPredictiveWorld,
    DynamicPredictiveWorldConfig,
)
from clearvla.experiments.legacy_v33.dynamic_world_lab.representation import (
    DynamicRepresentationLossConfig,
    evaluate_dynamic_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a V33.4 action-independent dynamics representation checkpoint"
    )
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16}[name]


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = _dtype(args.dtype)
    checkpoint = _load_checkpoint(args.checkpoint)
    if checkpoint.get("schema") != "clearvla-v33.4-dynamic-representation-checkpoint-v1":
        raise ValueError(f"unsupported representation schema={checkpoint.get('schema')!r}")
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
    saved_dataset = context["dataset"]
    action_normalizer = ArrayNormalizer.from_dict(checkpoint["action_normalizer"])
    state_normalizer = ArrayNormalizer.from_dict(checkpoint["state_normalizer"])
    dataset_config = DynamicWorldDatasetConfig(
        **{**saved_dataset, "return_images": args.condition_mode != "dinov2-cache"}
    )
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
        splits=context["splits"],
    )
    cameras = tuple(str(name) for name in args.cameras)
    split_ids = val_ids if args.split == "val" else test_ids
    dataset = DynamicWorldWindowDataset(
        episodes,
        split_ids,
        image_store=image_store,
        camera_names=cameras,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        config=dataset_config,
    )
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.num_workers,
        shuffle=False,
        device=device,
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
    model.load_representation_state_dict(checkpoint["representation"], freeze=True)
    loss_config = DynamicRepresentationLossConfig(**checkpoint["loss_config"])
    metrics = evaluate_dynamic_representation(
        model=model,
        loader=loader,
        conditioner=conditioner,
        device=device,
        dtype=dtype,
        camera_names=cameras,
        loss_config=loss_config,
        max_batches=args.max_batches,
    )
    output = {
        "schema": "clearvla-v33.4-dynamic-representation-eval-v1",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_windows": len(dataset),
        "metrics": metrics,
    }
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
