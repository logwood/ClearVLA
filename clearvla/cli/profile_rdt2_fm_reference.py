from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.experiments.classic_policy_lab.rdt2_fm_reference import (
    RDT2FMReferenceConfig,
    estimate_rdt2_fm_parameter_count,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile the released RDT2-FM action-expert tensor contract without allocating the full model"
    )
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RDT2FMReferenceConfig()
    report = {
        "model": "RDT2FMReference",
        "upstream_compatible_shape": cfg.upstream_compatible,
        "parameter_count": estimate_rdt2_fm_parameter_count(cfg),
        "tensor_count": 292,
        "bf16_raw_mib": estimate_rdt2_fm_parameter_count(cfg) * 2 / 1024 / 1024,
        "config": cfg.__dict__,
        "active_condition_path": "RDT2-VQ per-layer KV cache",
        "flow_matching": {
            "objective": "velocity MSE",
            "inference_steps": cfg.num_inference_timesteps,
            "solver": "first-order Euler",
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
