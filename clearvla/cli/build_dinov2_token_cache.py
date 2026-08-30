from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.schema import parse_camera_key_overrides
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import DinoV2DenseConditioner
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import (
    DinoV2TokenStore,
    episode_tokens_exist,
    load_episode_token_meta,
    save_episode_tokens,
)
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.online_store import OnlineVisualStore
from clearvla.vision.preprocessing import PreprocessConfig, parse_hw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Encode decoded RGB caches once into strict mmap-backed DINOv2 patch-token caches"
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--glob", default="*.hdf5")
    p.add_argument(
        "--decoded-image-cache-dir",
        type=Path,
        default=None,
        help="Optional decoded mmap cache; omit to decode JPEG rows directly with a bounded LRU.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    p.add_argument("--action-key", default="action")
    p.add_argument("--state-key", default="qpos")
    p.add_argument("--top-key", default="observations/images/cam_high")
    p.add_argument("--wrist-key", default="observations/images/cam_right_wrist")
    p.add_argument(
        "--camera-key",
        action="append",
        default=[],
        metavar="NAME=HDF5/PATH",
        help="Repeatable explicit key for any ordered camera name.",
    )
    p.add_argument("--cache-resize", nargs=2, type=int, default=[224, 224], metavar=("H", "W"))
    p.add_argument("--cache-crop", nargs=2, type=int, default=None, metavar=("H", "W"))
    p.add_argument("--dinov2-model", default="facebook/dinov2-base")
    p.add_argument("--dinov2-local-files-only", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--max-episodes", type=int, default=0)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--allow-skipped", action="store_true")
    p.add_argument("--frame-lru-capacity", type=int, default=512)
    p.add_argument("--open-file-capacity", type=int, default=8)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device)
    if args.dtype == "bf16" and device.type != "cuda":
        raise RuntimeError("--dtype bf16 requires CUDA; use --dtype fp32 on CPU")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    cameras = tuple(str(x) for x in args.cameras)
    camera_keys = parse_camera_key_overrides(args.camera_key)
    unknown = sorted(set(camera_keys) - set(cameras))
    if unknown:
        raise ValueError(f"camera key assignments name unselected cameras: {unknown}")
    camera_keys.setdefault("top", args.top_key)
    camera_keys.setdefault("wrist", args.wrist_key)
    preprocessing = PreprocessConfig(
        resize_hw=parse_hw(args.cache_resize), crop_hw=parse_hw(args.cache_crop)
    )
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=1,
        action_key=args.action_key,
        state_key=args.state_key,
        camera_key_overrides=camera_keys,
    )
    if skipped and not args.allow_skipped:
        raise RuntimeError(f"DINO cache inventory has skipped episodes: {skipped[:5]}")
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if args.decoded_image_cache_dir is None:
        image_store: DecodedImageStore | OnlineVisualStore = OnlineVisualStore(
            camera_names=cameras,
            preprocessing=preprocessing,
            frame_lru_capacity=args.frame_lru_capacity,
            open_file_capacity=args.open_file_capacity,
        )
    else:
        image_store = DecodedImageStore(
            args.decoded_image_cache_dir,
            camera_names=cameras,
            preprocessing=preprocessing,
        )
    for episode in episodes:
        image_store.validate_episode(episode)
    conditioner = DinoV2DenseConditioner(
        args.dinov2_model, local_files_only=args.dinov2_local_files_only
    )
    conditioner = conditioner.to(device=device, dtype=dtype).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode_idx, episode in enumerate(episodes):
        if episode_tokens_exist(args.out_dir, episode.cache_key) and not args.rebuild:
            meta = load_episode_token_meta(args.out_dir, episode.cache_key)
            rows.append(meta.to_dict())
            print(
                f"[dinov2-cache] episode={episode_idx + 1:03d}/{len(episodes):03d} "
                f"id={episode.episode_id} status=reuse",
                flush=True,
            )
            continue
        token_batches = []
        for start in range(0, episode.length, args.batch_size):
            end = min(episode.length, start + args.batch_size)
            indices = np.arange(start, end, dtype=np.int64)
            frames = image_store.load_window(episode, indices)
            images = (
                torch.stack([frames[camera] for camera in cameras], dim=1).to(torch.float32) / 255.0
            )
            images = images.to(device=device, non_blocking=True)
            with torch.no_grad():
                dense = conditioner.encode(images, camera_names=cameras).dense_tokens
            if dense is None:
                raise AssertionError("DINOv2 conditioner did not return dense tokens")
            if dense.shape[1] % len(cameras) != 0:
                raise ValueError(
                    f"flattened token count={dense.shape[1]} is not divisible by cameras={len(cameras)}"
                )
            patches = dense.shape[1] // len(cameras)
            token_batches.append(
                dense.reshape(end - start, len(cameras), patches, dense.shape[2])
                .float()
                .cpu()
                .numpy()
            )
        tokens = np.concatenate(token_batches, axis=0)
        meta = save_episode_tokens(
            cache_dir=args.out_dir,
            episode=episode,
            camera_names=cameras,
            preprocessing=preprocessing,
            dinov2_model=args.dinov2_model,
            tokens=tokens,
            rebuild=args.rebuild,
        )
        rows.append(meta.to_dict())
        print(
            f"[dinov2-cache] episode={episode_idx + 1:03d}/{len(episodes):03d} "
            f"id={episode.episode_id} shape={tokens.shape}",
            flush=True,
        )
    # Re-open strictly before reporting success.
    store = DinoV2TokenStore(
        args.out_dir,
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing,
        dinov2_model=args.dinov2_model,
    )
    report = {
        "schema": "clearvla-rdt2-dinov2-token-cache-v1",
        "episodes": len(episodes),
        "skipped": skipped,
        "out_dir": str(args.out_dir),
        "decoded_image_cache_dir": (
            None
            if args.decoded_image_cache_dir is None
            else str(args.decoded_image_cache_dir)
        ),
        "image_source_mode": (
            "hdf5-direct" if args.decoded_image_cache_dir is None else "decoded-cache"
        ),
        "decoded_preprocessing": preprocessing.to_dict(),
        "cameras": list(cameras),
        "dinov2_model": args.dinov2_model,
        "token_dim": store.token_dim,
        "tokens_per_camera": store.tokens_per_camera,
        "dtype": "float16",
        "episode_meta": rows,
    }
    (args.out_dir / "cache_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
