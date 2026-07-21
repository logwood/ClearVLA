from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearvla.data.hdf5_episode import load_episodes
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
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional explicit resize. Default preserves native decoded resolution.",
    )
    p.add_argument("--crop", type=int, nargs=2, metavar=("H", "W"), default=None)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cameras = tuple(str(x) for x in args.cameras)
    preprocessing = PreprocessConfig(resize_hw=parse_hw(args.resize), crop_hw=parse_hw(args.crop))
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=1,
        action_key=args.action_key,
        camera_key_overrides={"top": args.top_key, "wrist": args.wrist_key},
    )
    metas = build_all_decoded_caches(
        episodes,
        cache_dir=args.cache_dir,
        camera_names=cameras,
        preprocessing=preprocessing,
        rebuild=args.rebuild,
    )
    payload = {
        "episodes": len(episodes),
        "skipped": skipped,
        "cache_dir": str(args.cache_dir),
        "preprocessing": preprocessing.to_dict(),
        "cameras": list(cameras),
        "camera_shapes_hwc": {
            meta.episode_stem: meta.to_dict()["camera_shapes_hwc"] for meta in metas
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
