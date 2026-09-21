#!/usr/bin/env python3
"""Find action-contract fields in a serialized run context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def walk(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if any(token in str(key).lower() for token in ("normalizer", "profile", "action_chart", "physical_chart")):
                print(child, "=", json.dumps(item, sort_keys=True))
            walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.context.read_text(encoding="utf-8"))
    print("top_keys", sorted(payload) if isinstance(payload, dict) else type(payload).__name__)
    walk(payload)


if __name__ == "__main__":
    main()
