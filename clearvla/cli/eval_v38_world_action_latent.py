from __future__ import annotations

"""Offline diagnostics for V38.6.2 action-centered controlled-residual rollout use.

This checks whether the rollout latent/effect target is sensitive to action and
visual perturbations.  Future observations are encoded once as a fixed target
for the normal sample and are never fed to the model as inputs.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from clearvla.experiments.classic_policy_lab.cli_common import add_data_args, load_data, make_loader, preprocessing_from_args, resolve_device
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset, PolicyWindowDataset
from clearvla.experiments.observed_state_lab.policy_v38 import V38PolicyConfig, V38PolicySystem
from clearvla.experiments.observed_state_lab.policy_runtime_v38 import (
    V38PolicyTrainerConfig,
    prepare_v38_policy_sample,
    rollout_contrast_loss,
    rollout_diagnostics,
    rollout_dynamics_loss,
)
from clearvla.experiments.observed_state_lab.world_runtime import autocast_context, jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V38.6.2 action-centered controlled-residual rollout latent dynamics diagnostics.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--condition-mode", choices=["dinov2", "dinov2-cache", "debug-dense"], default="dinov2-cache")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def _run(
    system: V38PolicySystem,
    sample: dict[str, torch.Tensor],
    *,
    mode: str,
    fixed_rollout_pack: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    local = dict(sample)
    b = local["policy_action"].shape[0]
    if mode == "shuffle_action" and b > 1:
        perm = torch.arange(b - 1, -1, -1, device=local["policy_action"].device)
        local["policy_action"] = local["policy_action"][perm]
    elif mode == "hold_action":
        local["policy_action"] = local["action_state"][:, None].expand_as(local["policy_action"])
    elif mode == "shuffle_visual" and b > 1:
        perm = torch.arange(b - 1, -1, -1, device=local["visual"].device)
        local["visual"] = local["visual"][perm]
    return system.flow_training_forward(
        local["visual"], local["history_state"], local["executed_action_history"], local["state"], local["policy_action"],
        action_state=local["action_state"], rollout_target_pack=fixed_rollout_pack, proposal_dropout=0.0, make_counterfactuals=False,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v38-policy-checkpoint-v1":
        raise ValueError("--checkpoint must be a V38/V38.5/V38.6/V38.6.2 policy checkpoint")
    context = payload["context"]
    dataset_config = ObservedStateDatasetConfig(**{**context["dataset"], "return_images": args.condition_mode != "dinov2-cache"})
    geom = context.get("visual_geometry")
    if geom is None:
        raise ValueError("checkpoint context is missing visual geometry")
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    min_length = dataset_config.world_horizon + abs(min(dataset_config.history_offsets + dataset_config.executed_action_offsets)) + 2
    episodes, train_ids, val_ids, test_ids, _, _, image_store, skipped = load_data(
        args, min_length=min_length, normalizer_mode=action_norm.mode,
        action_normalizer=action_norm, state_normalizer=state_norm, splits=context["splits"],
    )
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]
    ds = PolicyWindowDataset(ObservedStateWindowDataset(episodes, ids, image_store=image_store, camera_names=cameras, state_normalizer=state_norm, action_normalizer=action_norm, config=dataset_config))
    loader = make_loader(ds, batch_size=args.batch_size, workers=args.num_workers, shuffle=False, device=device)
    conditioner, latent_dim, patches = build_dense_conditioner(
        mode=args.condition_mode, episodes=episodes, camera_names=cameras, preprocessing=preprocessing_from_args(args),
        dinov2_model=args.dinov2_model, dinov2_local_files_only=args.dinov2_local_files_only,
        dinov2_token_cache_dir=args.dinov2_token_cache_dir, debug_token_dim=int(geom["latent_dim"]),
        debug_patches_per_camera=int(geom["patches_per_camera"]), device=device, dtype=dtype,
    )
    if latent_dim != int(geom["latent_dim"]) or (patches is not None and patches != int(geom["patches_per_camera"])):
        raise ValueError("conditioner geometry does not match checkpoint")
    system = V38PolicySystem(V38PolicyConfig(**payload["policy_config"]))
    system.load_state_dict(payload["model"], strict=True)
    system.to(device=device, dtype=torch.float32).eval()
    modes = ["normal", "shuffle_action", "hold_action", "shuffle_visual"]
    rows = {m: [] for m in modes}
    trainer = V38PolicyTrainerConfig(**payload["trainer_config"])
    for batch_idx, batch in enumerate(loader, start=1):
        if args.max_batches and batch_idx > args.max_batches:
            break
        sample = prepare_v38_policy_sample(batch, conditioner=conditioner, system=system, camera_names=cameras, device=device, dtype=dtype, include_target_visual=True)
        with autocast_context(device, dtype):
            fixed_pack = system.build_rollout_target_pack(sample["visual"], sample["target_visual"])
            for mode in modes:
                torch.manual_seed(38100 + batch_idx)
                out = _run(system, sample, mode=mode, fixed_rollout_pack=fixed_pack)
                diag = rollout_diagnostics(out)
                row = {
                    "rollout_dynamics": float(rollout_dynamics_loss(out).detach().float().cpu()),
                    "rollout_contrast": float(rollout_contrast_loss(out, margin=float(trainer.rollout_contrast_margin)).detach().float().cpu()),
                    "rollout_mse": float(torch.mean((out["rollout_effect_pred"].float() - out["rollout_effect_target"].float()) ** 2).detach().cpu()),
                    "gate_self": float(out["gate_self"].detach().float().cpu()),
                    "gate_visual": float(out["gate_visual"].detach().float().cpu()),
                    "gate_rollout": float(out.get("gate_rollout", torch.zeros((), device=device)).detach().float().cpu()),
                    "gate_ffn": float(out["gate_ffn"].detach().float().cpu()),
                    "mod_content_to_time": float(out.get("mod_content_to_time", torch.zeros((), device=device)).detach().float().cpu()),
                }
                row.update({k: float(v.detach().float().cpu()) for k, v in diag.items()})
                rows[mode].append(row)
    metrics = {mode: _mean(rows[mode]) for mode in modes}
    if metrics.get("normal"):
        base = metrics["normal"]
        metrics["deltas_vs_normal"] = {
            mode: {k: float(v - base[k]) for k, v in vals.items() if k in base}
            for mode, vals in metrics.items() if mode != "normal"
        }
    out = {"schema": "clearvla-v38-6-2-action-centered-residual-diagnostics-v1", "split": args.split, "checkpoint": str(args.checkpoint), "metrics": metrics, "skipped": skipped}
    print(json.dumps(jsonable(out), indent=2), flush=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
