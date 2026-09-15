from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from clearvla.experiments.observed_state_lab.policy_v36_2 import V362PolicyConfig
from clearvla.experiments.observed_state_lab.world_model import V35WorldConfig, WorldEvidenceEncoder
from clearvla.experiments.observed_state_lab.policy_v36_2 import V362PolicySystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static bottleneck audit for V36.2 physical-action-flow policy."
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--world-hidden", type=int, default=512)
    parser.add_argument("--world-tokens", type=int, default=16)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = V362PolicyConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_horizon=args.horizon,
        hidden_size=args.hidden_size,
        num_heads=args.heads,
    )
    world = V35WorldConfig(
        latent_dim=64,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        world_horizon=12,
        segment_length=12,
        history_length=3,
        executed_history_length=3,
        num_cameras=2,
        patches_per_camera=8,
        hidden_size=args.world_hidden,
        num_heads=args.heads,
        world_tokens=args.world_tokens,
        global_tokens=max(1, args.world_tokens // 4),
        interaction_tokens=max(1, args.world_tokens // 2),
        motion_tokens=args.world_tokens
        - max(1, args.world_tokens // 4)
        - max(1, args.world_tokens // 2),
    )
    system = V362PolicySystem(world, policy, WorldEvidenceEncoder(world))
    report = {
        "schema": "clearvla-v36-2-static-bottleneck-audit-v1",
        "policy_config": asdict(policy),
        "shape_contract": {
            "native_action_dim": policy.action_dim,
            "arm_dim": policy.arm_dim,
            "physical_action_dim": policy.physical_action_dim,
            "flow_coordinate": "arm_abs + arm_delta + gripper_value + gripper_delta",
            "flow_expansion_ratio_vs_raw_action": policy.physical_action_dim / policy.action_dim,
            "horizon": policy.action_horizon,
            "first_execution_steps": policy.first_execution_steps,
            "mid_execution_steps": policy.mid_execution_steps,
        },
        "bottleneck_decisions": [
            {
                "site": "noisy action lift",
                "v36_1": "single Linear(7 -> hidden)",
                "v36_2": "typed component lift over 14-D physical action: arm_abs/arm_delta/grip/grip_delta",
                "status": "resolved for action-coordinate bottleneck",
            },
            {
                "site": "flow output head",
                "v36_1": "single LayerNorm+Linear(hidden -> 7)",
                "v36_2": "typed physical velocity heads hidden -> arm_abs/arm_delta/grip/grip_delta",
                "status": "resolved for raw 7-D velocity choke point",
            },
            {
                "site": "horizon role",
                "v36_1": "position embedding only",
                "v36_2": "explicit execution/mid/tail role embeddings",
                "status": "stage-resolved; train/eval must verify first4/tail behavior",
            },
            {
                "site": "world-policy interface",
                "v36_1": "frozen V35 world evidence tokens",
                "v36_2": "intentionally unchanged",
                "status": "deferred by design to isolate action-flow changes",
            },
        ],
        "parameter_report": system.parameter_report(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
