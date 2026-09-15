from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clearvla.experiments.classic_policy_lab.cli_common import (
    add_data_args,
    load_data,
    make_loader,
    preprocessing_from_args,
    resolve_device,
)
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.dynamic_world_lab.conditioning import build_dense_conditioner
from clearvla.experiments.observed_state_lab.intervention import (
    InterventionBranchDataset,
    validate_intervention_groups,
)
from clearvla.experiments.observed_state_lab.world_model import (
    V35ObservedStateWorldModel,
    V35WorldConfig,
)
from clearvla.experiments.observed_state_lab.world_runtime import (
    prepare_v35_sample,
    forward_v35,
    jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V35 on short intervention branches.")
    add_data_args(parser, default_resize=(336, 336))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--intervention-npz", type=Path, required=True)
    parser.add_argument(
        "--condition-mode",
        choices=["dinov2", "dinov2-cache", "debug-dense"],
        default="dinov2-cache",
    )
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--dinov2-token-cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "clearvla-v35-world-checkpoint-v1":
        raise ValueError("checkpoint is not V35 world")
    cfg = V35WorldConfig(**payload["model_config"])
    context = payload["context"]
    action_norm = ArrayNormalizer.from_dict(payload["action_normalizer"])
    state_norm = ArrayNormalizer.from_dict(payload["state_normalizer"])
    device = resolve_device(args.device)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    cameras = tuple(str(x) for x in args.cameras)
    episodes, *_ = load_data(
        args,
        min_length=2,
        normalizer_mode=action_norm.mode,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        splits=context["splits"],
    )
    dataset = InterventionBranchDataset(
        args.intervention_npz,
        action_normalizer=action_norm,
        state_normalizer=state_norm,
        policy_horizon=context["dataset"]["policy_horizon"],
    )
    validate_intervention_groups(dataset.data["branch_group"])
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
        debug_token_dim=cfg.latent_dim,
        debug_patches_per_camera=cfg.patches_per_camera,
        device=device,
        dtype=dtype,
    )
    if latent_dim != cfg.latent_dim or (patches is not None and patches != cfg.patches_per_camera):
        raise ValueError("conditioner mismatch")
    model = V35ObservedStateWorldModel(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    outputs = []
    for batch in loader:
        sample = prepare_v35_sample(
            batch,
            conditioner=conditioner,
            model=model,
            camera_names=cameras,
            device=device,
            dtype=dtype,
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=device.type == "cuda" and dtype != torch.float32,
            ),
        ):
            out = forward_v35(model, sample)
        for row in range(sample["state"].shape[0]):
            outputs.append(
                {
                    "group": int(batch["branch_group"][row]),
                    "initial": out["target_initial_world"][row].float().cpu(),
                    "pred": out["pred_world"][row].float().cpu(),
                    "target": out["target_world"][row].float().cpu(),
                    "state_pred": out["pred_segment_state"][row].float().cpu(),
                    "state_target": sample["segment_state"][row].float().cpu(),
                }
            )
    groups: dict[int, list[dict]] = {}
    for row in outputs:
        groups.setdefault(row["group"], []).append(row)
    cosines, magnitude_ratios, state_rmse = [], [], []
    for rows in groups.values():
        for left, right in combinations(rows, 2):
            pred_delta = (left["pred"] - left["initial"][None]) - (
                right["pred"] - right["initial"][None]
            )
            target_delta = (left["target"] - left["initial"][None]) - (
                right["target"] - right["initial"][None]
            )
            cosines.append(
                float(F.cosine_similarity(pred_delta.flatten(), target_delta.flatten(), dim=0))
            )
            magnitude_ratios.append(float(pred_delta.norm() / target_delta.norm().clamp_min(1e-8)))
        for row in rows:
            state_rmse.append(
                float((row["state_pred"] - row["state_target"]).square().mean().sqrt())
            )
    metrics = {
        "groups": len(groups),
        "branches": len(outputs),
        "branch_pairs": len(cosines),
        "intervention_effect_cosine": float(np.mean(cosines)),
        "intervention_effect_magnitude_ratio": float(np.mean(magnitude_ratios)),
        "intervention_segment_state_rmse_normalized": float(np.mean(state_rmse)),
    }
    print(json.dumps(jsonable(metrics), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
