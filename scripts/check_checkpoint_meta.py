#!/usr/bin/env python3
"""Read scalar checkpoint metadata without materializing model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="meta", weights_only=False)
    print("type", type(payload).__name__)
    if not isinstance(payload, dict):
        return
    print("keys", sorted(str(key) for key in payload.keys()))
    for key, value in payload.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            print(f"{key}={value!r}")
        elif isinstance(value, dict):
            scalar = {
                str(k): v
                for k, v in value.items()
                if isinstance(v, (int, float, str, bool)) or v is None
            }
            if scalar:
                print(f"{key}_scalar={scalar!r}")


if __name__ == "__main__":
    main()
