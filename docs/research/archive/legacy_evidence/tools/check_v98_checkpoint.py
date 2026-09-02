#!/usr/bin/env python3
"""Print the immutable metadata needed by the V98 address probe preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["policy_config"]
    print(
        json.dumps(
            {
                "schema": payload.get("schema"),
                "epoch": payload.get("epoch"),
                "global_step": payload.get("global_step"),
                "raw": config.get("flow_jepa_raw_image_enabled"),
                "guard": config.get("flow_jepa_zero_flow_guard", 0),
                "midcut": config.get("midcut_layer"),
                "role_hierarchy": config.get("flow_jepa_role_hierarchy"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
