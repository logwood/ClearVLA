#!/usr/bin/env python3
"""Mechanically embed JPEG assets into an existing visualization fragment."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


MARKER = "/*__EMBEDDED_ASSETS__*/ {}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    parser.add_argument("asset_dir", type=Path)
    args = parser.parse_args()

    mapping = {
        path.name: "data:image/jpeg;base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
        for path in sorted(args.asset_dir.glob("*.jpg"))
    }
    source = args.html.read_text(encoding="utf-8")
    if MARKER not in source:
        raise SystemExit(f"marker not found in {args.html}")
    replacement = json.dumps(mapping, separators=(",", ":"))
    args.html.write_text(source.replace(MARKER, replacement), encoding="utf-8")


if __name__ == "__main__":
    main()
