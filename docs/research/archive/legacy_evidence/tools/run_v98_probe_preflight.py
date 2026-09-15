#!/usr/bin/env python3
"""Run the V98 probe test without pytest and print checkpoint metadata."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch


TEST_NAME = (
    "test_v98_transient_address_intervention_preserves_checkpoint_and_camera_identity"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    namespace = runpy.run_path(str(args.test_file))
    namespace[TEST_NAME]()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["policy_config"]
    print(
        json.dumps(
            {
                "targeted_test": "pass",
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
