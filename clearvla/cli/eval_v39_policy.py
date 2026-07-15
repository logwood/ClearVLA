from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, load_data, make_loader, preprocessing_from_args, resolve_device
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset, PolicyWindowDataset
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
    evaluate_v39_policy,
)
from clearvla.experiments.observed_state_lab.world_runtime import jsonable
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.system import V39PolicySystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V39 staged mid-cut temporal policy.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--eval-inference-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_schema = payload.get("schema")
    if checkpoint_schema not in POLICY_CHECKPOINT_SCHEMAS:
        raise ValueError("--checkpoint must be a V39/V40 policy checkpoint")
    context = payload["context"]
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    visual_geometry = context.get("visual_geometry")
    if visual_geometry is None and "source_world_model" in context:
        # Backward compatibility with the first V38 package.
        source_world = context["source_world_model"]
        visual_geometry = {
            "history_length": int(source_world["history_length"]),
            "future_count": int(source_world["num_future"]),
            "num_cameras": int(source_world["num_cameras"]),
            "patches_per_camera": int(source_world["patches_per_camera"]),
            "latent_dim": int(source_world["latent_dim"]),
        }
    if visual_geometry is None:
        raise ValueError("checkpoint context is missing visual_geometry")
    class Geometry:
        history_length = int(visual_geometry["history_length"])
        num_future = int(visual_geometry["future_count"])
        num_cameras = int(visual_geometry["num_cameras"])
        patches_per_camera = int(visual_geometry["patches_per_camera"])
        latent_dim = int(visual_geometry["latent_dim"])
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    min_length = dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2
    episodes, train_ids, val_ids, test_ids, _, _, image_store, skipped = load_data(
        args, min_length=min_length, normalizer_mode=action_norm.mode,
        action_normalizer=action_norm, state_normalizer=state_norm, splits=context["splits"],
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    effective = ObservedStateDatasetConfig(**{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    dataset = PolicyWindowDataset(ObservedStateWindowDataset(episodes, split_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=effective))
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras, preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model, dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir, debug_token_dim=Geometry.latent_dim,
        debug_patches_per_camera=Geometry.patches_per_camera, device=device, dtype=dtype,
    )
    if latent_dim != Geometry.latent_dim or (patches is not None and patches != Geometry.patches_per_camera):
        raise ValueError("conditioner geometry does not match checkpoint")
    policy_config = V39PolicyConfig(**payload["policy_config"])
    trainer = V39PolicyTrainerConfig(**payload["trainer_config"])
    if args.eval_inference_steps is not None:
        trainer = V39PolicyTrainerConfig(**{**asdict(trainer), "eval_inference_steps": int(args.eval_inference_steps)})
    system = V39PolicySystem(policy_config)
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)
    metrics = evaluate_v39_policy(
        system=system, loader=loader, conditioner=conditioner, device=device, dtype=dtype,
        camera_names=cameras, action_normalizer=action_norm, trainer=trainer, max_batches=args.max_val_batches,
    )
    eval_schema = (
        "clearvla-v40-policy-eval-v1"
        if checkpoint_schema == "clearvla-v40-policy-checkpoint-v1"
        else "clearvla-v39-policy-eval-v1"
    )
    out = {"schema": eval_schema, "split": args.split, "checkpoint": str(args.checkpoint), "metrics": metrics, "skipped": skipped}
    print(json.dumps(jsonable(out), indent=2), flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
