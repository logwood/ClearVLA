"""Save deterministic TargetFact forward tensors for source parity checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import (
    CurrentObservation,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy


def make_input(config, batch: int, device: torch.device) -> OnlinePolicyInput:
    dims = config.dimensions
    side = config.data.cache_side
    return OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=torch.randn(
                batch, dims.visual_history_length, dims.num_cameras,
                dims.patches_per_camera, dims.visual_token_dim,
                device=device, dtype=torch.bfloat16,
            ),
            raw_rgb=torch.randn(
                batch, dims.visual_history_length, dims.num_cameras, 3,
                side, side, device=device,
            ),
        ),
        history=ObservableHistory(
            state=torch.randn(batch, dims.state_dim, device=device),
            action_state=torch.randn(batch, dims.action_dim, device=device),
            codec_gripper_boundary=torch.randn(batch, 1, device=device),
            state_history=torch.randn(
                batch, dims.state_history_length, dims.state_dim, device=device,
            ),
            executed_action_history=torch.randn(
                batch, dims.executed_history_length, dims.action_dim,
                device=device,
            ),
        ),
        goal=GoalCondition(
            tokens=torch.randn(
                batch, dims.goal_max_tokens, dims.goal_token_dim, device=device,
            ),
            mask=torch.ones(
                batch, dims.goal_max_tokens, device=device, dtype=torch.bool,
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    config = load_config("configs/mainline/calvin_push_color6_target_fact_v1.json")
    device = torch.device("cuda")
    torch.manual_seed(17_022)
    model = ClearVLAMainlinePolicy(config).to(device).eval()
    zeros = np.zeros((2, config.dimensions.action_dim), dtype=np.float32)
    model.configure_action_normalizer(ArrayNormalizer.fit_identity([zeros]))
    policy_input = make_input(config, args.batch, device)
    noisy = torch.randn(
        args.batch, config.dimensions.action_horizon,
        model.outlet_adapter.physical_dim, device=device,
    )
    time = torch.full((args.batch,), 0.5, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        cache, _, _ = model.encode_online(policy_input)
        output = model.velocity(cache, noisy_action_field=noisy, time=time)
    plan = output.compiled.plan
    payload = {
        "factual_protected": cache.factual_dock.protected_detail.float().cpu(),
        "factual_scene": (
            None if cache.factual_dock.scene_detail is None
            else cache.factual_dock.scene_detail.float().cpu()
        ),
        "bottom_velocity": output.bottom.physical_velocity.float().cpu(),
        "effect_target_semantic": output.compiled.effect.target.semantic.float().cpu(),
        "effect_target_geometry": output.compiled.effect.target.geometry.float().cpu(),
        "consequence_protected": output.compiled.consequence.protected_consequence.float().cpu(),
        "plan_protected_base": plan.protected_base.float().cpu(),
        "plan_policy_precision": plan.protected_policy_precision.float().cpu(),
        "plan_temporal": plan.temporal.float().cpu(),
        "plan_state_change": plan.state_change.float().cpu(),
    }
    if plan.scene is not None:
        payload["plan_scene"] = plan.scene.float().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.output} keys={sorted(payload)}")


if __name__ == "__main__":
    main()
