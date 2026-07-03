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
from clearvla.experiments.observed_state_lab.policy_v36_2 import V362PolicyConfig, V362PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v36_2 import V362PolicyTrainerConfig, evaluate_v362_policy
from clearvla.experiments.observed_state_lab.world_model import V35WorldConfig, WorldEvidenceEncoder
from clearvla.experiments.observed_state_lab.world_runtime import jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V36.2 physical-action-flow policy checkpoint.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--eval-inference-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v36-2-policy-checkpoint-v1":
        raise ValueError("checkpoint is not V36.2 policy")
    world_config = V35WorldConfig(**payload["world_config"])
    policy_config = V362PolicyConfig(**payload["policy_config"])
    trainer = V362PolicyTrainerConfig(**payload["trainer_config"])
    if args.eval_inference_steps is not None:
        trainer = V362PolicyTrainerConfig(**{**asdict(trainer), "eval_inference_steps": int(args.eval_inference_steps)})
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    context = payload["context"]
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    cameras = tuple(str(x) for x in args.cameras)
    episodes, train_ids, val_ids, test_ids, _, _, image_store, _ = load_data(
        args, min_length=dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2,
        normalizer_mode=action_norm.mode, action_normalizer=action_norm, state_normalizer=state_norm,
        splits=context["splits"],
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    effective = ObservedStateDatasetConfig(**{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    base = ObservedStateWindowDataset(episodes, split_ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=effective)
    loader = make_loader(PolicyWindowDataset(base), batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras, preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model, dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir, debug_token_dim=world_config.latent_dim,
        debug_patches_per_camera=world_config.patches_per_camera, device=device, dtype=dtype,
    )
    if latent_dim != world_config.latent_dim or (patches is not None and patches != world_config.patches_per_camera):
        raise ValueError("conditioner geometry does not match checkpoint")
    system = V362PolicySystem(world_config, policy_config, WorldEvidenceEncoder(world_config))
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)
    metrics = evaluate_v362_policy(system=system, loader=loader, conditioner=conditioner, device=device, dtype=dtype,
                                   camera_names=cameras, action_normalizer=action_norm, trainer=trainer,
                                   max_batches=args.max_val_batches)
    report = {"schema": "clearvla-v36-2-policy-eval-v1", "split": args.split, "checkpoint": str(args.checkpoint), "metrics": metrics}
    text = json.dumps(jsonable(report), indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
