from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.schema import parse_camera_key_overrides
from clearvla.vision.decoded_image_store import build_all_decoded_caches
from clearvla.vision.preprocessing import PreprocessConfig, parse_hw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode HDF5 images once into strict mmap-backed uint8 episode caches"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")
    p.add_argument(
        "--camera-key",
        action="append",
        default=[],
        metavar="NAME=HDF5/PATH",
        help="Repeatable explicit key for any ordered camera name.",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional explicit resize. Default preserves native decoded resolution.",
    )
    p.add_argument("--crop", type=int, nargs=2, metavar=("H", "W"), default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--allow-skipped", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cameras = tuple(str(x) for x in args.cameras)
    camera_keys = parse_camera_key_overrides(args.camera_key)
    unknown = sorted(set(camera_keys) - set(cameras))
    if unknown:
        raise ValueError(f"camera key assignments name unselected cameras: {unknown}")
    camera_keys.setdefault("top", args.top_key)
    camera_keys.setdefault("wrist", args.wrist_key)
    preprocessing = PreprocessConfig(resize_hw=parse_hw(args.resize), crop_hw=parse_hw(args.crop))
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=1,
        action_key=args.action_key,
        camera_key_overrides=camera_keys,
    )
    if skipped and not args.allow_skipped:
        raise RuntimeError(f"decoded cache inventory has skipped episodes: {skipped[:5]}")
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    metas = build_all_decoded_caches(
        episodes,
        cache_dir=args.cache_dir,
        camera_names=cameras,
        preprocessing=preprocessing,
        rebuild=args.rebuild,
    )
    payload = {
        "schema": "clearvla-decoded-image-cache-report-v1",
        "episodes": len(episodes),
        "skipped": skipped,
        "cache_dir": str(args.cache_dir),
        "preprocessing": preprocessing.to_dict(),
        "cameras": list(cameras),
        "camera_shapes_hwc": {
            meta.episode_stem: meta.to_dict()["camera_shapes_hwc"] for meta in metas
        },
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "cache_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
