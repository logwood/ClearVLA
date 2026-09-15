#!/usr/bin/env python3
"""Read-only summary of CALVIN language annotation coverage."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def _summary(path: Path) -> dict[str, object]:
    payload = np.load(path, allow_pickle=True).item()
    language = payload["language"]
    annotations = [str(value) for value in language["ann"]]
    counts = collections.Counter(annotations)
    colors = ("blue", "red", "pink", "green", "yellow")
    return {
        "path": str(path),
        "annotations": len(annotations),
        "unique": len(counts),
        "color_counts": {
            color: {
                "rows": sum(count for text, count in counts.items() if color in text.lower()),
                "unique": sum(1 for text in counts if color in text.lower()),
                "examples": sorted(
                    ((text, int(count)) for text, count in counts.items() if color in text.lower()),
                    key=lambda item: (-item[1], item[0]),
                )[:12],
            }
            for color in colors
        },
        "target_exact_counts": {
            text: int(counts[text])
            for text in (
                "go push the blue block right",
                "go push the red block right",
                "go push the pink block right",
                "go push the blue block left",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([_summary(path) for path in args.paths], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
