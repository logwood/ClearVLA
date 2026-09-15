from __future__ import annotations

import argparse
import json

from clearvla.experiments.classic_policy_lab.act_reference import ACTReference, ACTReferenceConfig
from clearvla.experiments.classic_policy_lab.dp_reference import DPReference, DPReferenceConfig


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print ACT and Diffusion Policy reference parameter counts"
    )
    p.add_argument("--compact", action="store_true")
    args = p.parse_args()
    if args.compact:
        act_cfg = ACTReferenceConfig(
            hidden_dim=256,
            ffn_dim=1024,
            transformer_encoder_layers=2,
            transformer_decoder_layers=4,
            style_encoder_layers=2,
        )
        dp_cfg = DPReferenceConfig(down_dims=(256, 512, 1024))
    else:
        act_cfg = ACTReferenceConfig()
        dp_cfg = DPReferenceConfig()
    act = ACTReference(act_cfg)
    dp = DPReference(dp_cfg)
    print(
        json.dumps(
            {
                "act_parameters": act.parameter_count(),
                "dp_parameters": dp.parameter_count(),
                "act_config": act_cfg.to_dict(),
                "dp_config": dp_cfg.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
