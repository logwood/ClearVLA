#!/usr/bin/env python3
"""Export compact image samples for the image-stage probe visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


TEST_EPISODES = {
    68: 121,
    69: 132,
    70: 73,
    71: 80,
    72: 103,
}
PROGRESS = (0.0, 0.25, 0.5, 0.75, 1.0)
EVENT_OFFSETS = (-12, 0, 12)


def save_frame(array: np.ndarray, index: int, destination: Path) -> None:
    frame = Image.fromarray(array[index])
    frame.thumbnail((180, 180), Image.Resampling.LANCZOS)
    frame.save(destination, format="JPEG", quality=72, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"episodes": {}}
    for episode_id, anchor in TEST_EPISODES.items():
        stem = f"episode_{episode_id:06d}"
        episode_dir = args.image_cache / stem
        top = np.load(episode_dir / "top.uint8.npy", mmap_mode="r")
        wrist = np.load(episode_dir / "wrist.uint8.npy", mmap_mode="r")
        length = min(len(top), len(wrist))
        episode: dict[str, object] = {
            "length": length,
            "anchor": anchor,
            "anchor_progress": anchor / max(length - 1, 1),
            "progress": [],
            "event": [],
        }

        for fraction in PROGRESS:
            index = round(fraction * (length - 1))
            item = {"fraction": fraction, "index": index}
            for camera, array in (("top", top), ("wrist", wrist)):
                name = f"ep{episode_id}_{camera}_p{round(fraction * 100):03d}.jpg"
                save_frame(array, index, args.output / name)
                item[camera] = name
            episode["progress"].append(item)

        for offset in EVENT_OFFSETS:
            index = min(max(anchor + offset, 0), length - 1)
            name = f"ep{episode_id}_wrist_event_{offset:+03d}.jpg"
            save_frame(wrist, index, args.output / name)
            episode["event"].append({"offset": offset, "index": index, "wrist": name})

        manifest["episodes"][str(episode_id)] = episode

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
