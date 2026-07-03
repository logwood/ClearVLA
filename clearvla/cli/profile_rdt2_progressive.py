from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.experiments.classic_policy_lab.legacy_guard import require_legacy_rdt2_cli

from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import RDT2FMReferenceConfig, estimate_rdt2_fm_parameter_count
from clearvla.experiments.classic_policy_lab.rdt2_progressive import ProgressiveRDT2FM, ProgressiveRDT2FMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile a v19 progressive RDT2-FM preset")
    p.add_argument("--model-size", choices=["small", "medium", "official"], default="medium")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--state-dim", type=int, default=7)
    p.add_argument("--dense-token-dim", type=int, default=768)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def config_for(name: str, *, action_dim: int, state_dim: int, dense_token_dim: int) -> ProgressiveRDT2FMConfig:
    common = dict(action_dim=action_dim, state_dim=state_dim, lang_adaptor="mlp2x_silu", lang_token_dim=dense_token_dim)
    if name == "small":
        return ProgressiveRDT2FMConfig(**common, hidden_size=256, depth=6, num_heads=4, num_kv_heads=2, history_hidden_size=64, fast_exit_layer=2, prefix_exit_layer=4, visual_start_layer=2, modulation_rank=64)
    if name == "medium":
        return ProgressiveRDT2FMConfig(**common, hidden_size=512, depth=8, num_heads=8, num_kv_heads=4, history_hidden_size=128, fast_exit_layer=2, prefix_exit_layer=4, visual_start_layer=2, modulation_rank=128)
    return ProgressiveRDT2FMConfig(**common, hidden_size=1024, depth=14, num_heads=8, num_kv_heads=4, history_hidden_size=256, fast_exit_layer=4, prefix_exit_layer=8, visual_start_layer=4, modulation_rank=256)


def main() -> None:
    require_legacy_rdt2_cli("clearvla/cli/profile_rdt2_progressive.py")
    args = parse_args()
    cfg = config_for(args.model_size, action_dim=args.action_dim, state_dim=args.state_dim, dense_token_dim=args.dense_token_dim)
    model = ProgressiveRDT2FM(cfg)
    progressive = model.parameter_count()
    reference = estimate_rdt2_fm_parameter_count(RDT2FMReferenceConfig(action_dim=args.action_dim, state_dim=args.state_dim, lang_adaptor="mlp2x_silu", lang_token_dim=args.dense_token_dim))
    report = {
        "model": "ProgressiveRDT2FM",
        "model_size": args.model_size,
        "parameter_count": progressive,
        "reference_parameter_count_same_local_interface": reference,
        "parameter_reduction": 1.0 - progressive / reference,
        "config": cfg.__dict__,
        "network_changes": [
            "history-only learned trajectory prior",
            "prior-relative residual flow bridge",
            "first-action and near-prefix native exits",
            "prefix-priority loss",
            "stage-shared low-rank modulation",
            "pooled dense-token first-action visual correction",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
