from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args, load_data, make_loader, preprocessing_from_args, resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.arm_motion_table import (
    ArmMotionTableConfig,
    collect_arm_motion_predictions_for_episode,
    write_arm_motion_tables,
)
from clearvla.experiments.observed_state_lab.dataset import (
    ObservedStateDatasetConfig, ObservedStateWindowDataset, PolicyWindowDataset,
)
from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    POLICY_CHECKPOINT_SCHEMAS,
    V39PolicyTrainerConfig,
)
from clearvla.experiments.observed_state_lab.policy_v39 import V39PolicyConfig, V39PolicySystem
from clearvla.experiments.observed_state_lab.world_runtime import jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump V39 arm-motion and optional FK task-space tables for one episode.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--episode-idx", type=int, default=None, help="Raw episode index. Defaults to first episode of selected split.")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--first-k", type=int, default=4)
    parser.add_argument("--motion-window", type=int, default=8)
    parser.add_argument("--motion-source", choices=["auto", "ee", "joint"], default="auto")
    parser.add_argument("--motion-quantile", type=float, default=0.80)
    parser.add_argument("--motion-min-value", type=float, default=None)
    parser.add_argument("--phase-merge-gap", type=int, default=8)
    parser.add_argument("--ratio-eps", type=float, default=1e-6)
    parser.add_argument("--disable-fk", action="store_true")
    parser.add_argument("--urdf-path", type=Path, default=None)
    parser.add_argument("--urdf-variant", default="gripper_50mm", choices=["gripper_50mm", "gripper_100mm"])
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--end-link", default="tool0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
        raise ValueError("unsupported checkpoint; expected V39/V40 policy checkpoint")
    policy_config = V39PolicyConfig(**payload["policy_config"])
    trainer = V39PolicyTrainerConfig(**payload["trainer_config"])
    context = payload["context"]
    dataset_config = ObservedStateDatasetConfig(**context["dataset"])
    visual_geometry = context["visual_geometry"]
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
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
    effective = ObservedStateDatasetConfig(
        **{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"}
    )
    ids = val_ids if args.split == "val" else test_ids
    if not ids:
        raise ValueError(f"selected split {args.split!r} is empty")
    episode_idx = int(ids[0] if args.episode_idx is None else args.episode_idx)
    if episode_idx not in set(int(x) for x in ids):
        raise ValueError(f"episode_idx={episode_idx} is not in {args.split} split: {ids}")
    base = ObservedStateWindowDataset(
        episodes, ids, image_store=image_store, camera_names=cameras,
        state_normalizer=state_norm, action_normalizer=action_norm, config=effective,
    )
    loader = make_loader(
        PolicyWindowDataset(base), batch_size=args.batch_size, workers=args.num_workers,
        shuffle=False, device=device,
    )
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras,
        preprocessing=preprocessing_from_args(args), dinov2_model=args.dinov2_model,
        dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir,
        debug_token_dim=int(visual_geometry["latent_dim"]),
        debug_patches_per_camera=int(visual_geometry["patches_per_camera"]),
        device=device, dtype=dtype,
    )
    if latent_dim != int(visual_geometry["latent_dim"]) or (patches is not None and patches != int(visual_geometry["patches_per_camera"])):
        raise ValueError("conditioner mismatch")
    system = V39PolicySystem(policy_config)
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32)

    table_config = ArmMotionTableConfig(
        episode_idx=episode_idx,
        action_offset=dataset_config.action_offset,
        policy_horizon=dataset_config.policy_horizon,
        gripper_index=policy_config.gripper_index,
        first_k=args.first_k,
        inference_steps=trainer.eval_inference_steps,
        motion_window=args.motion_window,
        motion_source=args.motion_source,
        motion_quantile=args.motion_quantile,
        motion_min_value=args.motion_min_value,
        phase_merge_gap=args.phase_merge_gap,
        ratio_eps=args.ratio_eps,
        urdf_path=str(args.urdf_path) if args.urdf_path else None,
        urdf_variant=args.urdf_variant,
        base_link=args.base_link,
        end_link=args.end_link,
        enable_fk=not bool(args.disable_fk),
    )
    window_rows, episode_rows, fk_meta = collect_arm_motion_predictions_for_episode(
        system=system, loader=loader, conditioner=conditioner, device=device, dtype=dtype,
        camera_names=cameras, action_normalizer=action_norm, config=table_config,
        max_batches=args.max_batches,
    )
    if not window_rows:
        raise ValueError(f"no windows collected for episode_idx={episode_idx}")
    summary = write_arm_motion_tables(
        out_prefix=args.out_prefix, config=table_config,
        window_rows=window_rows, episode_rows=episode_rows, fk_meta=fk_meta,
    )
    print(json.dumps(jsonable(summary), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
