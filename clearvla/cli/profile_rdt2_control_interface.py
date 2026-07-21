from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

import torch
from torch import nn

from clearvla.experiments.classic_policy_lab.rdt2_control_interface import (
    ControlInterfaceRDT2FMConfig,
    RDT2ControlInterface,
)
from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import RDT, RDT2FMReference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile v21 RDT2-FM control-interface parameter groups"
    )
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def reference_dense_condition_count(config: ControlInterfaceRDT2FMConfig) -> int:
    """Count the v18 dense-condition reference without allocating ~2 GiB."""
    rdt_cfg = {
        "hidden_size": config.hidden_size,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "num_kv_heads": config.num_kv_heads,
        "num_register_tokens": config.num_register_tokens,
        "norm_eps": config.norm_eps,
        "multiple_of": config.multiple_of,
        "ffn_dim_multiplier": config.ffn_dim_multiplier,
        "use_flash_attn": config.use_flash_attn,
    }
    with torch.device("meta"):
        core = RDT(
            horizon=config.prediction_horizon,
            output_size=config.action_dim,
            config=rdt_cfg,
            x_pos_emb_config=[
                ("action", config.prediction_horizon),
                ("register", config.num_register_tokens),
            ],
            dtype=torch.float32,
        )
        lang = RDT2FMReference._build_adapter("linear", config.dense_token_dim, config.hidden_size)
        action = RDT2FMReference._build_adapter("mlp3x_silu", config.action_dim, config.hidden_size)
        state = RDT2FMReference._build_adapter("mlp3x_silu", config.state_dim, config.hidden_size)
    assert lang is not None and action is not None and state is not None
    return count(core) + count(lang) + count(action) + count(state)


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/profile_rdt2_control_interface.py")
    args = parse_args()
    base = ControlInterfaceRDT2FMConfig()
    with torch.device("meta"):
        model = RDT2ControlInterface(base, dtype=torch.float32)
    dynamic_specific = (
        count(model.control.summary)
        + count(model.control.state_in)
        + count(model.control.time)
        + count(model.control.dynamic_bias)
    )
    groups = model.parameter_groups()
    total = model.parameter_count()
    rows = []
    for mode in ("static", "dynamic"):
        rows.append(
            {
                "variant": mode,
                "parameters_allocated": total,
                "parameters_active": total - dynamic_specific if mode == "static" else total,
                "parameter_groups": groups,
                "dynamic_specific_parameters": dynamic_specific,
                "model": replace(base, interface_mode=mode).__dict__,
            }
        )
    report = {
        "schema": "clearvla-rdt2-control-interface-profile-v1",
        "controlled_motor_core": {
            "hidden_size": base.hidden_size,
            "depth": base.depth,
            "action_horizon": base.prediction_horizon,
            "inference_steps": base.num_inference_timesteps,
        },
        "reference_dense_condition_parameters": reference_dense_condition_count(base),
        "variants": rows,
        "notes": [
            "Both static and dynamic checkpoints allocate the same modules for state-dict parity.",
            "Static mode deliberately leaves dynamic-specific query-bias modules inactive.",
            "The experiment is not a compression experiment: the 14-layer motor core is retained.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
