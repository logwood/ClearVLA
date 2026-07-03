from __future__ import annotations

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import argparse
import json

import torch

from clearvla.cli.train_rdt2_grounded_motor import PRESETS
from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import RDT2FMReferenceConfig, estimate_rdt2_fm_parameter_count
from clearvla.experiments.classic_policy_lab.rdt2_grounded_motor import GroundedMotorRDT2FM, GroundedMotorRDT2FMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the v20 grounded shallow motor policy")
    p.add_argument("--model-size", choices=["small", "medium", "wide"], default="medium")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--prediction-horizon", type=int, default=24)
    p.add_argument("--dense-token-dim", type=int, default=768)
    return p.parse_args()


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/profile_rdt2_grounded_motor.py")
    args = parse_args()
    preset = PRESETS[args.model_size]
    cfg = GroundedMotorRDT2FMConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        prediction_horizon=args.prediction_horizon,
        hidden_size=preset["hidden_size"],
        first_depth=preset["first_depth"],
        tail_depth=preset["tail_depth"],
        num_heads=preset["heads"],
        num_kv_heads=preset["kv_heads"],
        multiple_of=preset["multiple_of"],
        dense_token_dim=args.dense_token_dim,
        grounding_depth=preset["grounding_depth"],
        grounding_queries=preset["grounding_queries"],
        history_hidden_size=preset["history_hidden_size"],
        motion_tokens=preset["motion_tokens"],
        use_flash_attn=False,
    )
    model = GroundedMotorRDT2FM(cfg, dtype=torch.float32)
    reference = RDT2FMReferenceConfig(
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        prediction_horizon=args.prediction_horizon,
        hidden_size=cfg.hidden_size,
        depth=14,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        multiple_of=cfg.multiple_of,
        use_flash_attn=False,
        lang_adaptor="linear",
        lang_token_dim=args.dense_token_dim,
    )
    payload = {
        "model": "GroundedMotorRDT2FM",
        "model_size": args.model_size,
        "parameters": model.parameter_count(),
        "reference_same_width_depth14_parameters": estimate_rdt2_fm_parameter_count(reference),
        "ratio_vs_reference_same_width_depth14": model.parameter_count() / estimate_rdt2_fm_parameter_count(reference),
        "config": model.config_dict(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
