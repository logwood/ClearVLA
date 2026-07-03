from __future__ import annotations

import argparse
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args, load_data, make_loader, preprocessing_from_args, print_context, resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset, PolicyWindowDataset
from clearvla.experiments.observed_state_lab.policy_v36_3 import V363PolicyConfig, V363PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v36_3 import V363PolicyTrainerConfig, train_v363_policy
from clearvla.experiments.observed_state_lab.world_model import V35ObservedStateWorldModel, V35WorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V36.3 transition-aware action latent policy.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-policy-checkpoint", type=Path, default=None, help="Optional V36.2/V36.3 policy checkpoint for model-only warm start.")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6, help="typed latent DiT planner depth")
    parser.add_argument("--action-decoder-depth", type=int, default=4, help="planner-conditioned physical action decoder depth")
    parser.add_argument("--proposal-depth", type=int, default=2)
    parser.add_argument("--proposal-dropout", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--event-tokens", type=int, default=3)
    parser.add_argument("--inference-steps", type=int, default=5)
    parser.add_argument("--gripper-dim-index", type=int, default=-1)
    parser.add_argument("--first-execution-steps", type=int, default=4)
    parser.add_argument("--mid-execution-steps", type=int, default=8)
    parser.add_argument("--physical-decode-delta-blend", type=float, default=0.25)
    parser.add_argument("--transition-fusion-dropout", type=float, default=0.05)
    parser.add_argument("--transition-event-dropout", type=float, default=0.10)

    defaults = V363PolicyTrainerConfig()
    for field in V363PolicyTrainerConfig.__dataclass_fields__:
        value = getattr(defaults, field)
        parser.add_argument("--" + field.replace("_", "-"), dest=field, type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)

    world_payload = torch.load(args.world_checkpoint, map_location="cpu", weights_only=False)
    if world_payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("--world-checkpoint must be a V35 world checkpoint")
    world_config = V35WorldConfig(**world_payload["model_config"])
    world_context = world_payload["context"]
    dataset_config = ObservedStateDatasetConfig(**world_context["dataset"])
    action_norm = ArrayNormalizer.from_dict(world_payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(world_payload["state_normalizer"])
    min_length = dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2
    episodes, train_ids, val_ids, test_ids, _, _, image_store, skipped = load_data(
        args, min_length=min_length, normalizer_mode=action_norm.mode,
        action_normalizer=action_norm, state_normalizer=state_norm, splits=world_context["splits"],
    )
    effective = ObservedStateDatasetConfig(**{**world_context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    bases = {
        name: ObservedStateWindowDataset(episodes, ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=effective)
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))
    }
    train_loader = make_loader(PolicyWindowDataset(bases["train"]), batch_size=args.batch_size, workers=args.num_workers, shuffle=True, device=device)
    val_loader = make_loader(PolicyWindowDataset(bases["val"]), batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras, preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model, dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir, debug_token_dim=world_config.latent_dim,
        debug_patches_per_camera=world_config.patches_per_camera, device=device, dtype=dtype,
    )
    if latent_dim != world_config.latent_dim or (patches is not None and patches != world_config.patches_per_camera):
        raise ValueError("conditioner geometry does not match world checkpoint")
    world_model = V35ObservedStateWorldModel(world_config)
    world_model.load_state_dict(world_payload["model"], strict=True)
    policy_config = V363PolicyConfig(
        action_dim=int(action_norm.scale.shape[-1]), state_dim=int(state_norm.scale.shape[-1]),
        action_horizon=dataset_config.policy_horizon, executed_history_length=len(dataset_config.executed_action_offsets),
        hidden_size=args.hidden_size, num_heads=args.heads, depth=args.depth,
        action_decoder_depth=args.action_decoder_depth, proposal_depth=args.proposal_depth,
        proposal_dropout=args.proposal_dropout, dropout=args.dropout, event_tokens=args.event_tokens,
        gripper_dim_index=args.gripper_dim_index, inference_steps=args.inference_steps,
        first_execution_steps=args.first_execution_steps, mid_execution_steps=args.mid_execution_steps,
        physical_decode_delta_blend=args.physical_decode_delta_blend,
        transition_fusion_dropout=args.transition_fusion_dropout,
        transition_event_dropout=args.transition_event_dropout,
    )
    system = V363PolicySystem(world_config, policy_config, deepcopy(world_model.online_encoder))
    init_report = None
    if args.init_policy_checkpoint is not None:
        init_payload = torch.load(args.init_policy_checkpoint, map_location="cpu", weights_only=False)
        if init_payload.get("schema") not in {"clearvla-v36-2-policy-checkpoint-v1", "clearvla-v36-3-policy-checkpoint-v1"}:
            raise ValueError("--init-policy-checkpoint must be a V36.2 or V36.3 policy checkpoint")
        incompatible = system.load_state_dict(init_payload["model"], strict=False)
        init_report = {
            "checkpoint": str(args.init_policy_checkpoint),
            "schema": init_payload.get("schema"),
            "epoch": init_payload.get("epoch"),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    trainer = V363PolicyTrainerConfig(**{name: getattr(args, name) for name in V363PolicyTrainerConfig.__dataclass_fields__})
    context = {
        "schema": "clearvla-v36-3-transition-aware-action-latent-context-v1", "args": vars(args),
        "world_checkpoint": str(args.world_checkpoint), "world_checkpoint_epoch": world_payload["epoch"],
        "splits": world_context["splits"], "dataset": asdict(dataset_config), "world_model": asdict(world_config),
        "policy_model": asdict(policy_config), "trainer": asdict(trainer), "parameter_report": system.parameter_report(),
        "init_policy_checkpoint": init_report,
        "skipped": skipped,
        "policy_contract": (
            "frozen V35 world encoder; V36.3 transition-aware action latent inside V36.2 typed physical action flow; "
            "event/transition readout and physical velocity share the fused action latent; existing physical velocity head remains the only action output; "
            "native Alicia-D 7-D decoder; no target event labels in forward path"
        ),
    }
    print_context(context)
    train_v363_policy(system=system, train_loader=train_loader, val_loader=val_loader, conditioner=conditioner,
                      device=device, dtype=dtype, camera_names=cameras, action_normalizer=action_norm,
                      state_normalizer=state_norm, trainer=trainer, out_dir=args.out_dir, context=context,
                      resume=args.resume)


if __name__ == "__main__":
    main()
