#!/usr/bin/env python3
"""Print the compact action/normalizer contract from a CALVIN result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    health = payload.get("bridge_health_after", {})
    deployment = health.get("deployment", {}) if isinstance(health, dict) else {}
    action = deployment.get("action", {}) if isinstance(deployment, dict) else {}
    normalizers = action.get("normalizers", {}) if isinstance(action, dict) else {}
    print(json.dumps({
        "result_action_contract": payload.get("action_contract"),
        "normalizers": normalizers,
        "checkpoint": payload.get("checkpoint"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
