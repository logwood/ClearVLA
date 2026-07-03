from __future__ import annotations

import argparse
import json

from clearvla.experiments.classic_policy_lab.rdt_small_reference import RDTSmallReference, RDTSmallReferenceConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the released RDT-170M / RDT-small policy core")
    p.add_argument("--robot-dim", type=int, default=7)
    p.add_argument("--state-indices", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 10])
    return p.parse_args()


def count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def main() -> None:
    args = parse_args()
    cfg = RDTSmallReferenceConfig(robot_dim=args.robot_dim, state_indices=tuple(args.state_indices))
    model = RDTSmallReference(cfg)
    report = {
        "schema": "clearvla-rdt-small-reference-profile-v1",
        "marketed_name": "RDT-170M / RDT-small",
        "policy_parameters": model.parameter_count(),
        "official_170m_shape": model.architecture_is_official_170m(),
        "breakdown": {
            "rdt_core": count(model.model),
            "language_adapter": count(model.lang_adaptor),
            "image_adapter": count(model.img_adaptor),
            "state_action_adapter": count(model.state_adaptor),
            "unified_mapper": count(model.mapper),
        },
        "config": cfg.to_dict(),
        "excluded_from_policy_count": [
            "frozen google/siglip-so400m-patch14-384 vision tower",
            "precomputed google/t5-v1_1-xxl language encoder",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
