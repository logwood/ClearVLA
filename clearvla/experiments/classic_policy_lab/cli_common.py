from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from clearvla.data.hdf5_episode import LoadedEpisode, load_episodes
from clearvla.data.split import resolve_episode_ids
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.preprocessing import PreprocessConfig, parse_hw
from .normalizer import ArrayNormalizer


def add_data_args(parser: argparse.ArgumentParser, *, default_resize: tuple[int, int]) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--decoded-image-cache-dir", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["top", "wrist"])
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-key", default="qpos")
    parser.add_argument("--top-key", default="observations/images/cam_high")
    parser.add_argument("--wrist-key", default="observations/images/cam_right_wrist")
    parser.add_argument(
        "--cache-resize", nargs=2, type=int, default=list(default_resize), metavar=("H", "W")
    )
    parser.add_argument("--cache-crop", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--episode-split-mode",
        choices=["random-frac", "random-fraction", "ordered-counts"],
        default="random-frac",
        help=(
            "random-frac preserves the legacy seeded split; ordered-counts uses "
            "naturally ordered train prefix, validation immediately before the test suffix, "
            "and the final test suffix"
        ),
    )
    parser.add_argument("--train-episode-count", type=int, default=None)
    parser.add_argument("--val-episode-count", type=int, default=None)
    parser.add_argument("--test-episode-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--device", default="auto")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def preprocessing_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        resize_hw=parse_hw(args.cache_resize), crop_hw=parse_hw(args.cache_crop)
    )


def load_data(
    args: argparse.Namespace,
    *,
    min_length: int,
    normalizer_mode: str,
    action_normalizer: ArrayNormalizer | None = None,
    state_normalizer: ArrayNormalizer | None = None,
    splits: dict[str, list[int]] | None = None,
) -> tuple[
    list[LoadedEpisode],
    list[int],
    list[int],
    list[int],
    ArrayNormalizer,
    ArrayNormalizer,
    DecodedImageStore,
    list[tuple[str, str]],
]:
    cameras = tuple(str(x) for x in args.cameras)
    episodes, skipped = load_episodes(
        args.data_root,
        args.glob,
        cameras=cameras,
        min_length=min_length,
        action_key=args.action_key,
        state_key=args.state_key,
        camera_key_overrides={"top": args.top_key, "wrist": args.wrist_key},
    )
    if splits is None:
        train_ids, val_ids, test_ids = resolve_episode_ids(
            len(episodes),
            mode=getattr(args, "episode_split_mode", "random-frac"),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            train_episode_count=getattr(args, "train_episode_count", None),
            val_episode_count=getattr(args, "val_episode_count", None),
            test_episode_count=getattr(args, "test_episode_count", None),
            episode_names=[episode.stem for episode in episodes],
        )
    else:
        train_ids = [int(x) for x in splits["train"]]
        val_ids = [int(x) for x in splits["val"]]
        test_ids = [int(x) for x in splits["test"]]
    if action_normalizer is None or state_normalizer is None:
        if normalizer_mode == "zscore":
            fit = ArrayNormalizer.fit_zscore
        elif normalizer_mode == "limits":
            fit = ArrayNormalizer.fit_limits
        elif normalizer_mode == "identity":
            fit = ArrayNormalizer.fit_identity
        else:
            raise ValueError(f"unknown normalizer mode: {normalizer_mode}")
        action_normalizer = fit([episodes[index].actions_raw for index in train_ids])
        state_normalizer = fit([episodes[index].states_raw for index in train_ids])
    image_store = DecodedImageStore(
        args.decoded_image_cache_dir,
        camera_names=cameras,
        preprocessing=preprocessing_from_args(args),
    )
    for episode in episodes:
        image_store.validate_episode(episode)
    return (
        episodes,
        train_ids,
        val_ids,
        test_ids,
        action_normalizer,
        state_normalizer,
        image_store,
        skipped,
    )


def make_loader(
    dataset, *, batch_size: int, workers: int, shuffle: bool, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serializable(x) for x in value]
    if isinstance(value, list):
        return [serializable(x) for x in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    return value


def print_context(payload: dict[str, Any]) -> None:
    print(json.dumps(serializable(payload), indent=2), flush=True)
