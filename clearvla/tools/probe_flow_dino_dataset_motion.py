"""Measure matchable motion in a DINO cache without loading a policy checkpoint.

The probe mirrors the V95 Flow-DINO geometry: cached DINO patch tokens are
average-pooled to the configured grid, L2-normalized, and matched with the
same squared normalized-feature error used by the learned warp objective.
It compares the identity (zero-flow) correspondence with a radius-bounded
integer-patch oracle and a temperature-scaled global soft correspondence.
This is a dataset identifiability probe, not an evaluation of a learned
optical-flow checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from clearvla.data.hdf5_episode import find_hdf5_files


@dataclass(frozen=True)
class MatchBatch:
    identity_cosine: Tensor
    oracle_cosine: Tensor
    identity_warp_error: Tensor
    oracle_warp_error: Tensor
    soft_warp_error: Tensor
    displacement: Tensor
    soft_displacement: Tensor
    soft_entropy: Tensor
    soft_margin: Tensor
    nonzero: Tensor
    mutual: Tensor


class MatchCollector:
    def __init__(self) -> None:
        self._values: dict[str, list[np.ndarray]] = {
            "identity_cosine": [],
            "oracle_cosine": [],
            "identity_warp_error": [],
            "oracle_warp_error": [],
            "soft_warp_error": [],
            "displacement": [],
            "soft_displacement": [],
            "soft_entropy": [],
            "soft_margin": [],
            "nonzero": [],
            "mutual": [],
            "camera_id": [],
            "episode_id": [],
        }

    def append(self, batch: MatchBatch, *, episode_id: int) -> None:
        shape = tuple(batch.identity_cosine.shape)
        if len(shape) != 3:
            raise ValueError(f"match batch must be [B,C,P], got {shape}")
        batch_size, cameras, patches = shape
        for name in (
            "identity_cosine",
            "oracle_cosine",
            "identity_warp_error",
            "oracle_warp_error",
            "soft_warp_error",
            "displacement",
            "soft_displacement",
            "soft_entropy",
            "soft_margin",
            "nonzero",
            "mutual",
        ):
            value = getattr(batch, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape={tuple(value.shape)} does not match {shape}")
            self._values[name].append(value.detach().cpu().reshape(-1).numpy())
        camera_id = np.broadcast_to(
            np.arange(cameras, dtype=np.int16)[None, :, None],
            (batch_size, cameras, patches),
        )
        self._values["camera_id"].append(camera_id.reshape(-1).copy())
        self._values["episode_id"].append(
            np.full(batch_size * cameras * patches, episode_id, dtype=np.int16)
        )

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._values["identity_cosine"]:
            raise RuntimeError("no match rows were collected")
        return {name: np.concatenate(rows) for name, rows in self._values.items()}


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in text.replace(",", " ").split())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--train-episodes", type=int, default=63)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--history-offsets", type=_parse_ints, default=(-8, -4, 0))
    parser.add_argument(
        "--action-history-offsets",
        type=_parse_ints,
        default=(-24, -16, -12, -8, -6, -4, -2, -1),
    )
    parser.add_argument("--window-offsets", type=_parse_ints, default=(4, 12, 24))
    parser.add_argument("--stage-offset", type=int, default=48)
    parser.add_argument("--extra-target-offsets", type=_parse_ints, default=(1,))
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--correlation-radius", type=int, default=2)
    parser.add_argument("--correlation-temperature", type=float, default=0.07)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--motion-top-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _validate_offsets(name: str, values: Iterable[int], *, end_at_zero: bool = False) -> None:
    offsets = tuple(int(value) for value in values)
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError(f"{name} must be strictly increasing, got {offsets}")
    if end_at_zero and offsets[-1] != 0:
        raise ValueError(f"{name} must end at zero, got {offsets}")


def configured_pairs(
    history_offsets: tuple[int, ...],
    window_offsets: tuple[int, ...],
    stage_offset: int,
    extra_target_offsets: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    pairs = list(zip(history_offsets[:-1], history_offsets[1:]))
    pairs.extend((0, offset) for offset in extra_target_offsets)
    pairs.extend((0, offset) for offset in window_offsets)
    pairs.append((0, stage_offset))
    unique: list[tuple[int, int]] = []
    for pair in pairs:
        if pair[1] <= pair[0]:
            raise ValueError(f"probe pair must move forward in time, got {pair}")
        if pair not in unique:
            unique.append(pair)
    return tuple(unique)


def _select_files(
    files: list[Path],
    *,
    split: str,
    train_episodes: int,
    val_episodes: int,
    test_episodes: int,
    max_episodes: int,
) -> list[Path]:
    counts = (train_episodes, val_episodes, test_episodes)
    if any(value < 0 for value in counts) or min(train_episodes, val_episodes) < 1:
        raise ValueError("episode counts must be non-negative and train/val must be positive")
    total = sum(counts)
    if len(files) < total:
        raise ValueError(f"need {total} ordered episodes, found {len(files)}")
    slices = {
        "train": (0, train_episodes),
        "val": (train_episodes, train_episodes + val_episodes),
        "test": (train_episodes + val_episodes, total),
        "all": (0, total),
    }
    start, stop = slices[split]
    selected = files[start:stop]
    if max_episodes > 0:
        selected = selected[:max_episodes]
    if not selected:
        raise ValueError(f"split={split} selected no episodes")
    return selected


def _load_meta(cache_root: Path, stem: str) -> dict[str, Any]:
    path = cache_root / stem / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing DINO cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _token_array(cache_root: Path, stem: str) -> np.ndarray:
    path = cache_root / stem / "tokens.float16.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing DINO token cache: {path}")
    return np.load(path, mmap_mode="r")


def pool_cached_tokens(tokens: np.ndarray, indices: np.ndarray, grid_size: int) -> Tensor:
    """Pool cached tokens [T,C,P,D] to float32 [B,C,G,G,D]."""

    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [T,C,P,D], got {tokens.shape}")
    patch_grid = math.isqrt(int(tokens.shape[2]))
    if patch_grid * patch_grid != int(tokens.shape[2]):
        raise ValueError(f"patch count must be square, got {tokens.shape[2]}")
    if grid_size < 1 or patch_grid % grid_size:
        raise ValueError(f"grid_size={grid_size} must divide cached patch grid={patch_grid}")
    block = patch_grid // grid_size
    values = np.asarray(tokens[indices], dtype=np.float32).reshape(
        indices.size,
        tokens.shape[1],
        grid_size,
        block,
        grid_size,
        block,
        tokens.shape[3],
    )
    pooled = values.mean(axis=(3, 5), dtype=np.float32)
    return torch.from_numpy(pooled)


def local_oracle_match(
    source: Tensor, target: Tensor, radius: int, *, temperature: float = 0.07
) -> MatchBatch:
    """Compare identity matching with the best integer match inside ``radius``."""

    if source.shape != target.shape or source.ndim != 5:
        raise ValueError(
            f"source/target must share [B,C,G,G,D], got {source.shape}/{target.shape}"
        )
    if radius < 0:
        raise ValueError("correlation radius must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("correlation temperature must be finite and positive")
    batch, cameras, height, width, dim = source.shape
    if height != width:
        raise ValueError("the Flow-DINO probe requires a square pooled grid")
    count = batch * cameras
    patches = height * width
    source_flat = F.normalize(source.float().reshape(count, patches, dim), dim=-1)
    target_flat = F.normalize(target.float().reshape(count, patches, dim), dim=-1)
    correlation = torch.matmul(source_flat, target_flat.transpose(1, 2)).clamp(-1.0, 1.0)

    y, x = torch.meshgrid(
        torch.arange(height, device=source.device),
        torch.arange(width, device=source.device),
        indexing="ij",
    )
    coordinates = torch.stack((x, y), dim=-1).reshape(patches, 2)
    displacement_table = coordinates[None] - coordinates[:, None]
    legal = displacement_table.abs().amax(dim=-1) <= radius

    legal_correlation = correlation.masked_fill(~legal[None], float("-inf"))
    oracle_cosine, oracle_index = legal_correlation.max(dim=-1)
    identity_cosine = correlation.diagonal(dim1=-2, dim2=-1)
    displacement_xy = coordinates[oracle_index] - coordinates[None, :, :]
    displacement = displacement_xy.float().square().sum(dim=-1).sqrt()
    nonzero = displacement > 0.0

    backward = correlation.transpose(1, 2).masked_fill(~legal.t()[None], float("-inf"))
    _, backward_index = backward.max(dim=-1)
    source_index = torch.arange(patches, device=source.device)[None].expand(count, -1)
    mutual = backward_index.gather(1, oracle_index) == source_index

    # The learned SEA-RAFT path initializes its continuous flow with a global
    # temperature-scaled correlation expectation. Repeating that construction
    # on raw normalized DINO tokens exposes sub-patch dataset motion without
    # pretending to reproduce the trainable feature encoder or residual update.
    probability = torch.softmax(correlation / float(temperature), dim=-1)
    coordinates_float = coordinates.float()
    expected_coordinates = torch.matmul(probability, coordinates_float)
    soft_flow = expected_coordinates - coordinates_float[None]
    soft_displacement = soft_flow.square().sum(dim=-1).sqrt()
    normalized_grid = expected_coordinates.reshape(count, height, width, 2).clone()
    normalized_grid[..., 0] = (
        2.0 * normalized_grid[..., 0] / float(max(width - 1, 1)) - 1.0
    )
    normalized_grid[..., 1] = (
        2.0 * normalized_grid[..., 1] / float(max(height - 1, 1)) - 1.0
    )
    target_image = target.float().reshape(count, height, width, dim).permute(0, 3, 1, 2)
    warped_target = F.grid_sample(
        target_image,
        normalized_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1).reshape(count, patches, dim)
    soft_warp_error = (
        source_flat - F.normalize(warped_target, dim=-1)
    ).square().mean(dim=-1)
    soft_entropy = -(
        probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()
    ).sum(dim=-1) / math.log(float(max(patches, 2)))
    top2 = probability.topk(k=min(2, patches), dim=-1).values
    soft_margin = top2[..., 0] - (top2[..., 1] if patches > 1 else 0.0)

    # This is exactly the mean squared error between L2-normalized patch
    # features used by the learned warp term; cosine form avoids gathering D.
    identity_warp_error = (2.0 - 2.0 * identity_cosine).clamp_min(0.0) / float(dim)
    oracle_warp_error = (2.0 - 2.0 * oracle_cosine).clamp_min(0.0) / float(dim)

    def restore(value: Tensor) -> Tensor:
        return value.reshape(batch, cameras, patches)

    return MatchBatch(
        identity_cosine=restore(identity_cosine),
        oracle_cosine=restore(oracle_cosine),
        identity_warp_error=restore(identity_warp_error),
        oracle_warp_error=restore(oracle_warp_error),
        soft_warp_error=restore(soft_warp_error),
        displacement=restore(displacement),
        soft_displacement=restore(soft_displacement),
        soft_entropy=restore(soft_entropy),
        soft_margin=restore(soft_margin),
        nonzero=restore(nonzero),
        mutual=restore(mutual),
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_matches(
    arrays: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    motion_top_fraction: float,
) -> dict[str, Any]:
    count = int(arrays["identity_cosine"].size)
    selected = np.ones(count, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != (count,) or not selected.any():
        raise ValueError("summary mask must select at least one match")

    identity_error = arrays["identity_warp_error"][selected].astype(np.float64)
    oracle_error = arrays["oracle_warp_error"][selected].astype(np.float64)
    soft_error = arrays["soft_warp_error"][selected].astype(np.float64)
    identity_mean = float(identity_error.mean())
    oracle_mean = float(oracle_error.mean())
    soft_mean = float(soft_error.mean())
    gain = identity_mean - oracle_mean
    soft_gain = identity_mean - soft_mean
    result: dict[str, Any] = {
        "patches": int(selected.sum()),
        "identity_cosine_mean": float(arrays["identity_cosine"][selected].mean()),
        "oracle_cosine_mean": float(arrays["oracle_cosine"][selected].mean()),
        "match_cosine_gain": float(
            (arrays["oracle_cosine"][selected] - arrays["identity_cosine"][selected]).mean()
        ),
        "identity_warp_error_mean": identity_mean,
        "oracle_warp_error_mean": oracle_mean,
        "soft_warp_error_mean": soft_mean,
        "warp_gain_mean": gain,
        "warp_gain_ratio": gain / max(identity_mean, 1e-12),
        "soft_warp_gain_mean": soft_gain,
        "soft_warp_gain_ratio": soft_gain / max(identity_mean, 1e-12),
        "soft_correlation_entropy_mean": float(arrays["soft_entropy"][selected].mean()),
        "soft_correlation_margin_mean": float(arrays["soft_margin"][selected].mean()),
        "nonzero_match_fraction": float(arrays["nonzero"][selected].mean()),
        "mutual_match_fraction": float(arrays["mutual"][selected].mean()),
        "displacement_patch_units": _quantiles(arrays["displacement"][selected]),
        "soft_displacement_patch_units": _quantiles(
            arrays["soft_displacement"][selected]
        ),
        "sufficient_statistics": {
            "identity_warp_error_sum": float(identity_error.sum()),
            "oracle_warp_error_sum": float(oracle_error.sum()),
            "soft_warp_error_sum": float(soft_error.sum()),
        },
    }

    mutual = selected & arrays["mutual"].astype(bool)
    if mutual.any():
        mutual_identity = arrays["identity_warp_error"][mutual].astype(np.float64)
        mutual_oracle = arrays["oracle_warp_error"][mutual].astype(np.float64)
        mutual_gain = float(mutual_identity.mean() - mutual_oracle.mean())
        result["mutual_only"] = {
            "patches": int(mutual.sum()),
            "warp_gain_mean": mutual_gain,
            "warp_gain_ratio": mutual_gain / max(float(mutual_identity.mean()), 1e-12),
            "nonzero_match_fraction": float(arrays["nonzero"][mutual].mean()),
            "displacement_patch_units": _quantiles(arrays["displacement"][mutual]),
        }

    selected_index = np.flatnonzero(selected)
    top_count = max(1, int(math.ceil(selected_index.size * motion_top_fraction)))
    local_error = arrays["identity_warp_error"][selected_index]
    top_local = np.argpartition(local_error, -top_count)[-top_count:]
    top_index = selected_index[top_local]
    top_identity = arrays["identity_warp_error"][top_index].astype(np.float64)
    top_oracle = arrays["oracle_warp_error"][top_index].astype(np.float64)
    top_soft = arrays["soft_warp_error"][top_index].astype(np.float64)
    top_gain = float(top_identity.mean() - top_oracle.mean())
    top_soft_gain = float(top_identity.mean() - top_soft.mean())
    result["motion_top"] = {
        "fraction": motion_top_fraction,
        "patches": int(top_index.size),
        "identity_warp_error_mean": float(top_identity.mean()),
        "oracle_warp_error_mean": float(top_oracle.mean()),
        "soft_warp_error_mean": float(top_soft.mean()),
        "warp_gain_mean": top_gain,
        "warp_gain_ratio": top_gain / max(float(top_identity.mean()), 1e-12),
        "soft_warp_gain_mean": top_soft_gain,
        "soft_warp_gain_ratio": top_soft_gain / max(float(top_identity.mean()), 1e-12),
        "nonzero_match_fraction": float(arrays["nonzero"][top_index].mean()),
        "mutual_match_fraction": float(arrays["mutual"][top_index].mean()),
        "displacement_patch_units": _quantiles(arrays["displacement"][top_index]),
        "soft_displacement_patch_units": _quantiles(
            arrays["soft_displacement"][top_index]
        ),
    }
    return result


def _bootstrap_gain_ratio(
    episode_summaries: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, float] | None:
    if samples <= 0 or len(episode_summaries) < 2:
        return None
    identity = np.asarray(
        [row["sufficient_statistics"]["identity_warp_error_sum"] for row in episode_summaries],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [row["sufficient_statistics"]["oracle_warp_error_sum"] for row in episode_summaries],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(episode_summaries), size=(samples, len(episode_summaries)))
    identity_sum = identity[draws].sum(axis=1)
    oracle_sum = oracle[draws].sum(axis=1)
    ratios = (identity_sum - oracle_sum) / np.maximum(identity_sum, 1e-12)
    return {
        "samples": samples,
        "p2_5": float(np.quantile(ratios, 0.025)),
        "p50": float(np.quantile(ratios, 0.50)),
        "p97_5": float(np.quantile(ratios, 0.975)),
    }


def _episode_contract(meta: dict[str, Any], tokens: np.ndarray) -> dict[str, Any]:
    if tokens.ndim != 4:
        raise ValueError(f"cached tokens must be [T,C,P,D], got {tokens.shape}")
    contract = {
        "cameras": list(meta["cameras"]),
        "dinov2_model": str(meta["dinov2_model"]),
        "tokens_per_camera": int(meta["tokens_per_camera"]),
        "token_dim": int(meta["token_dim"]),
        "cache_version": str(meta["cache_version"]),
    }
    expected = (
        len(contract["cameras"]),
        contract["tokens_per_camera"],
        contract["token_dim"],
    )
    if tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"cache tensor shape={tokens.shape[1:]}, metadata={expected}")
    return contract


def _pair_key(pair: tuple[int, int]) -> str:
    return f"{pair[0]:+d}:{pair[1]:+d}"


def main() -> int:
    args = _parse_args()
    _validate_offsets("history_offsets", args.history_offsets, end_at_zero=True)
    _validate_offsets("action_history_offsets", args.action_history_offsets)
    _validate_offsets("window_offsets", args.window_offsets)
    if args.action_history_offsets[-1] >= 0:
        raise ValueError("action history offsets must precede the current frame")
    if args.stage_offset <= args.window_offsets[-1]:
        raise ValueError("stage offset must follow every window offset")
    if min(args.grid_size, args.stride, args.batch_size) < 1 or args.correlation_radius < 0:
        raise ValueError("grid/stride/batch must be positive and radius non-negative")
    if not math.isfinite(args.correlation_temperature) or args.correlation_temperature <= 0.0:
        raise ValueError("correlation-temperature must be finite and positive")
    if not 0.0 < args.motion_top_fraction <= 1.0:
        raise ValueError("motion-top-fraction must be in (0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    files = find_hdf5_files(args.data_root, args.glob)
    selected = _select_files(
        files,
        split=args.split,
        train_episodes=args.train_episodes,
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        max_episodes=args.max_episodes,
    )
    pairs = configured_pairs(
        args.history_offsets,
        args.window_offsets,
        args.stage_offset,
        args.extra_target_offsets,
    )
    min_offset = min((*args.history_offsets, *args.action_history_offsets, 0))
    max_offset = max(
        (*args.window_offsets, args.stage_offset, *args.extra_target_offsets, 0)
    )
    print(
        f"[flow-data-probe] split={args.split} episodes={len(selected)} device={device} "
        f"grid={args.grid_size} radius={args.correlation_radius} "
        f"temperature={args.correlation_temperature} "
        f"center_support={min_offset:+d}:{max_offset:+d} pairs={pairs}",
        flush=True,
    )

    cache_contract: dict[str, Any] | None = None
    episode_cache: list[tuple[Path, np.ndarray]] = []
    for path in selected:
        meta = _load_meta(args.dino_cache, path.stem)
        tokens = _token_array(args.dino_cache, path.stem)
        contract = _episode_contract(meta, tokens)
        if cache_contract is None:
            cache_contract = contract
        elif contract != cache_contract:
            raise ValueError(f"DINO cache contract changed at {path.stem}")
        episode_cache.append((path, tokens))
    assert cache_contract is not None

    pair_results: dict[str, Any] = {}
    for pair_id, pair in enumerate(pairs):
        collector = MatchCollector()
        centers_total = 0
        for episode_id, (path, tokens) in enumerate(episode_cache):
            center_start = -min_offset
            center_stop = int(tokens.shape[0]) - max_offset
            centers = np.arange(center_start, center_stop, args.stride, dtype=np.int64)
            if centers.size == 0:
                raise ValueError(
                    f"{path.stem}: no centers support offsets {min_offset:+d}:{max_offset:+d}"
                )
            centers_total += int(centers.size)
            for start in range(0, int(centers.size), args.batch_size):
                center = centers[start : start + args.batch_size]
                source = pool_cached_tokens(tokens, center + pair[0], args.grid_size).to(
                    device=device, non_blocking=True
                )
                target = pool_cached_tokens(tokens, center + pair[1], args.grid_size).to(
                    device=device, non_blocking=True
                )
                with torch.no_grad():
                    matched = local_oracle_match(
                        source,
                        target,
                        args.correlation_radius,
                        temperature=args.correlation_temperature,
                    )
                collector.append(matched, episode_id=episode_id)
            print(
                f"[flow-data-probe] pair={_pair_key(pair)} "
                f"episode={episode_id + 1}/{len(episode_cache)} stem={path.stem} "
                f"centers={centers.size}",
                flush=True,
            )

        arrays = collector.arrays()
        overall = summarize_matches(
            arrays, motion_top_fraction=args.motion_top_fraction
        )
        cameras = {
            camera: summarize_matches(
                arrays,
                mask=arrays["camera_id"] == camera_id,
                motion_top_fraction=args.motion_top_fraction,
            )
            for camera_id, camera in enumerate(cache_contract["cameras"])
        }
        episodes = []
        for episode_id, (path, _) in enumerate(episode_cache):
            summary = summarize_matches(
                arrays,
                mask=arrays["episode_id"] == episode_id,
                motion_top_fraction=args.motion_top_fraction,
            )
            summary["stem"] = path.stem
            episodes.append(summary)
        bootstrap = _bootstrap_gain_ratio(
            episodes,
            samples=args.bootstrap_samples,
            seed=args.seed + pair_id,
        )
        pair_results[_pair_key(pair)] = {
            "source_offset": pair[0],
            "target_offset": pair[1],
            "delta": pair[1] - pair[0],
            "centers": centers_total,
            "overall": overall,
            "cameras": cameras,
            "episodes": episodes,
            "episode_bootstrap_warp_gain_ratio": bootstrap,
        }
        displacement = overall["displacement_patch_units"]
        soft_displacement = overall["soft_displacement_patch_units"]
        top = overall["motion_top"]
        ci_text = ""
        if bootstrap is not None:
            ci_text = f" ci95={bootstrap['p2_5']:.4f}:{bootstrap['p97_5']:.4f}"
        print(
            f"[flow-data-result] pair={_pair_key(pair)} patches={overall['patches']} "
            f"identity_warp={overall['identity_warp_error_mean']:.6f} "
            f"oracle_warp={overall['oracle_warp_error_mean']:.6f} "
            f"oracle_gain={overall['warp_gain_ratio']:.4f}{ci_text} "
            f"soft_warp={overall['soft_warp_error_mean']:.6f} "
            f"soft_gain={overall['soft_warp_gain_ratio']:.4f} "
            f"nonzero={overall['nonzero_match_fraction']:.3f} "
            f"mutual={overall['mutual_match_fraction']:.3f} "
            f"disp=p50:{displacement['p50']:.3f}/p90:{displacement['p90']:.3f}/p95:{displacement['p95']:.3f} "
            f"soft_disp=p50:{soft_displacement['p50']:.3f}/p90:{soft_displacement['p90']:.3f}/p95:{soft_displacement['p95']:.3f} "
            f"top_motion_gain={top['warp_gain_ratio']:.4f} "
            f"top_motion_nonzero={top['nonzero_match_fraction']:.3f}",
            flush=True,
        )

    payload = {
        "kind": "flow_dino_dataset_motion_probe_v1",
        "interpretation": (
            "A radius-bounded integer-patch oracle plus a temperature-scaled global soft "
            "correspondence prior over normalized cached DINO features. It measures matchable "
            "dataset motion and zero-flow headroom; it does not measure the learned Flow-DINO "
            "checkpoint."
        ),
        "config": {
            "data_root": str(args.data_root),
            "dino_cache": str(args.dino_cache),
            "split": args.split,
            "episode_counts": {
                "train": args.train_episodes,
                "val": args.val_episodes,
                "test": args.test_episodes,
            },
            "selected_episodes": [path.stem for path in selected],
            "history_offsets": list(args.history_offsets),
            "action_history_offsets": list(args.action_history_offsets),
            "window_offsets": list(args.window_offsets),
            "stage_offset": args.stage_offset,
            "extra_target_offsets": list(args.extra_target_offsets),
            "grid_size": args.grid_size,
            "correlation_radius": args.correlation_radius,
            "correlation_temperature": args.correlation_temperature,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "motion_top_fraction": args.motion_top_fraction,
            "seed": args.seed,
            "device": str(device),
        },
        "cache_contract": cache_contract,
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[flow-data-probe] wrote={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
