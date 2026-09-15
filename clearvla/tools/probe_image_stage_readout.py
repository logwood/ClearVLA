from __future__ import annotations

"""Probe implicit task-stage information in cached robot images.

The probe is intentionally read-only with respect to training artifacts. It
uses fixed episode splits, frozen cached DINO features or low-resolution RGB,
and linear readouts. Statistical summaries treat episodes, not frames, as the
independent unit and include within-episode permutation nulls.
"""

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from clearvla.data.schema import STATE_ALIASES, list_hdf5_datasets, resolve_key


@dataclass(frozen=True)
class Episode:
    path: Path
    qpos: np.ndarray

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def length(self) -> int:
        return int(self.qpos.shape[0])


@dataclass(frozen=True)
class ProbeData:
    dino: np.ndarray
    raw: np.ndarray
    target: np.ndarray
    episode_ids: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--state-key", default="qpos")
    parser.add_argument("--train-episodes", type=int, default=63)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--progress-stride", type=int, default=4)
    parser.add_argument("--stage-bins", type=int, default=5)
    parser.add_argument("--dino-spatial-pool", type=int, default=2)
    parser.add_argument("--raw-spatial-pool", type=int, default=12)
    parser.add_argument("--gripper-event-threshold-deg", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_episodes(root: Path, pattern: str, state_key: str, limit: int) -> list[Episode]:
    files = sorted(root.glob(pattern))
    files = [path for path in files if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}]
    if limit > 0:
        files = files[:limit]
    episodes: list[Episode] = []
    for path in files:
        datasets = list_hdf5_datasets(str(path))
        resolved = resolve_key(datasets, state_key, STATE_ALIASES, required=True)
        assert resolved is not None
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle[resolved], dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] < 2 or not np.isfinite(qpos).all():
            raise ValueError(f"{path}: invalid qpos shape/content {qpos.shape}")
        episodes.append(Episode(path, qpos))
    if not episodes:
        raise RuntimeError(f"no HDF5 episodes under {root} with glob={pattern!r}")
    return episodes


def _dino_pool(values: np.ndarray, output_grid: int) -> np.ndarray:
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError(f"expected DINO [N,2,P,D], got {values.shape}")
    patch_grid = math.isqrt(int(values.shape[2]))
    if patch_grid * patch_grid != int(values.shape[2]) or patch_grid % output_grid:
        raise ValueError(f"cannot pool {values.shape[2]} patches to {output_grid}x{output_grid}")
    block = patch_grid // output_grid
    values = np.asarray(values, dtype=np.float32).reshape(
        values.shape[0], 2, output_grid, block, output_grid, block, values.shape[3]
    )
    return values.mean(axis=(3, 5), dtype=np.float32)


def _raw_pool(values: np.ndarray, output_grid: int) -> np.ndarray:
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"expected RGB [N,H,W,3], got {values.shape}")
    height, width = int(values.shape[1]), int(values.shape[2])
    if height % output_grid or width % output_grid:
        raise ValueError(f"raw image {height}x{width} not divisible by pool={output_grid}")
    bh, bw = height // output_grid, width // output_grid
    values = np.asarray(values, dtype=np.float32).reshape(
        values.shape[0], output_grid, bh, output_grid, bw, 3
    )
    return values.mean(axis=(2, 4), dtype=np.float32) / 255.0


def _load_feature_rows(
    stem: str,
    indices: np.ndarray,
    dino_cache: Path,
    image_cache: Path,
    dino_pool: int,
    raw_pool: int,
) -> tuple[np.ndarray, np.ndarray]:
    dino_path = dino_cache / stem / "tokens.float16.npy"
    dino_array = np.load(dino_path, mmap_mode="r")
    dino = _dino_pool(dino_array[indices], dino_pool).astype(np.float16)
    cameras = []
    for camera in ("top", "wrist"):
        array = np.load(image_cache / stem / f"{camera}.uint8.npy", mmap_mode="r")
        cameras.append(_raw_pool(array[indices], raw_pool))
    raw = np.stack(cameras, axis=1).astype(np.float16)
    return dino, raw


def _progress_data(
    episodes: list[Episode],
    dino_cache: Path,
    image_cache: Path,
    stride: int,
    stage_bins: int,
    dino_pool: int,
    raw_pool: int,
) -> tuple[ProbeData, np.ndarray]:
    dino_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    progress_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    stage_rows: list[np.ndarray] = []
    for episode_id, episode in enumerate(episodes):
        indices = np.arange(0, episode.length, stride, dtype=np.int64)
        progress = indices.astype(np.float32) / max(episode.length - 1, 1)
        stage = np.minimum((progress * stage_bins).astype(np.int64), stage_bins - 1)
        dino, raw = _load_feature_rows(
            episode.stem, indices, dino_cache, image_cache, dino_pool, raw_pool
        )
        dino_rows.append(dino)
        raw_rows.append(raw)
        progress_rows.append(progress)
        stage_rows.append(stage)
        episode_rows.append(np.full(indices.size, episode_id, dtype=np.int32))
        print(
            f"[progress] episode={episode_id + 1:03d}/{len(episodes):03d} "
            f"stem={episode.stem} samples={indices.size}",
            flush=True,
        )
    return (
        ProbeData(
            np.concatenate(dino_rows),
            np.concatenate(raw_rows),
            np.concatenate(progress_rows),
            np.concatenate(episode_rows),
        ),
        np.concatenate(stage_rows),
    )


def _event_stage_data(
    episodes: list[Episode],
    dino_cache: Path,
    image_cache: Path,
    threshold_deg: float,
    dino_pool: int,
    raw_pool: int,
) -> tuple[ProbeData, list[dict[str, Any]]]:
    dino_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    anchors: list[dict[str, Any]] = []
    threshold = math.radians(threshold_deg)
    windows = ((-16, -8), (-4, 4), (8, 16))
    for episode_id, episode in enumerate(episodes):
        delta = np.diff(episode.qpos[:, -1])
        candidates = np.flatnonzero(delta >= threshold)
        if candidates.size == 0:
            anchors.append({"episode_id": episode_id, "stem": episode.stem, "status": "no_close"})
            continue
        anchor = int(candidates[np.argmax(delta[candidates])]) + 1
        if anchor + windows[0][0] < 0 or anchor + windows[-1][1] > episode.length:
            anchors.append(
                {"episode_id": episode_id, "stem": episode.stem, "status": "boundary", "anchor": anchor}
            )
            continue
        indices = np.concatenate(
            [np.arange(anchor + start, anchor + stop, dtype=np.int64) for start, stop in windows]
        )
        targets = np.repeat(np.arange(3, dtype=np.int64), [stop - start for start, stop in windows])
        dino, raw = _load_feature_rows(
            episode.stem, indices, dino_cache, image_cache, dino_pool, raw_pool
        )
        dino_rows.append(dino)
        raw_rows.append(raw)
        target_rows.append(targets)
        episode_rows.append(np.full(indices.size, episode_id, dtype=np.int32))
        anchors.append(
            {
                "episode_id": episode_id,
                "stem": episode.stem,
                "status": "used",
                "anchor": anchor,
                "close_delta_deg": float(math.degrees(float(delta[anchor - 1]))),
            }
        )
    if not dino_rows:
        raise RuntimeError("no usable close-event episodes")
    return (
        ProbeData(
            np.concatenate(dino_rows),
            np.concatenate(raw_rows),
            np.concatenate(target_rows),
            np.concatenate(episode_rows),
        ),
        anchors,
    )


def _interfaces(dino: np.ndarray, raw: np.ndarray) -> dict[str, np.ndarray]:
    dino_mean = dino.astype(np.float32).mean(axis=(2, 3))
    return {
        "dino_mean_top": dino_mean[:, 0],
        "dino_mean_wrist": dino_mean[:, 1],
        "dino_mean_both": dino_mean.reshape(dino_mean.shape[0], -1),
        "dino_spatial_both": dino.reshape(dino.shape[0], -1),
        "raw_lowres_top": raw[:, 0].reshape(raw.shape[0], -1),
        "raw_lowres_wrist": raw[:, 1].reshape(raw.shape[0], -1),
        "raw_lowres_both": raw.reshape(raw.shape[0], -1),
    }


def _split_masks(
    episode_ids: np.ndarray, train_episodes: int, val_episodes: int, test_episodes: int
) -> dict[str, np.ndarray]:
    val_start = train_episodes
    test_start = train_episodes + val_episodes
    stop = test_start + test_episodes
    return {
        "train": episode_ids < val_start,
        "val": (episode_ids >= val_start) & (episode_ids < test_start),
        "test": (episode_ids >= test_start) & (episode_ids < stop),
    }


def _fit_linear(
    features: np.ndarray,
    target: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    kind: str,
    classes: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed_everything(seed)
    x_train = features[masks["train"]].astype(np.float32)
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-5] = 1.0
    x_split = {
        split: torch.from_numpy((features[mask].astype(np.float32) - x_mean) / x_std)
        for split, mask in masks.items()
    }
    if kind == "regression":
        y_train_np = target[masks["train"]].astype(np.float32).reshape(-1, 1)
        y_mean = y_train_np.mean(axis=0)
        y_std = y_train_np.std(axis=0)
        y_std[y_std < 1e-6] = 1.0
        y_split = {
            split: torch.from_numpy(
                ((target[mask].astype(np.float32).reshape(-1, 1) - y_mean) / y_std)
            )
            for split, mask in masks.items()
        }
        model = nn.Linear(features.shape[1], 1).to(device)
        loss_fn: Callable[[Tensor, Tensor], Tensor] = nn.MSELoss()
    elif kind == "classification":
        y_mean = np.asarray([0.0], dtype=np.float32)
        y_std = np.asarray([1.0], dtype=np.float32)
        y_split = {
            split: torch.from_numpy(target[mask].astype(np.int64)) for split, mask in masks.items()
        }
        model = nn.Linear(features.shape[1], classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"unknown kind={kind!r}")
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, Tensor] | None = None
    best_val = math.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_split["train"].shape[0], generator=generator)
        train_sum = 0.0
        train_count = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            x = x_split["train"][index].to(device)
            y = y_split["train"][index].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            count = int(index.numel())
            train_sum += float(loss.detach()) * count
            train_count += count
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for start in range(0, x_split["val"].shape[0], batch_size):
                x = x_split["val"][start : start + batch_size].to(device)
                y = y_split["val"][start : start + batch_size].to(device)
                loss = loss_fn(model(x), y)
                count = int(x.shape[0])
                val_sum += float(loss) * count
                val_count += count
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("probe failed to select a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, x_split["test"].shape[0], batch_size):
            x = x_split["test"][start : start + batch_size].to(device)
            chunks.append(model(x).cpu().numpy())
    output = np.concatenate(chunks)
    if kind == "regression":
        prediction = (output * y_std + y_mean).reshape(-1)
    else:
        prediction = output.argmax(axis=1).astype(np.int64)
    return prediction, {
        "feature_dim": int(features.shape[1]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x, y = _rankdata(left), _rankdata(right)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    denominator = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": 1.0 - float(np.square(error).sum()) / max(denominator, 1e-12),
        "spearman": _spearman(prediction, target),
    }


def _classification_metrics(
    prediction: np.ndarray, target: np.ndarray, classes: int
) -> dict[str, float]:
    recalls, f1s = [], []
    for label in range(classes):
        positive = target == label
        predicted = prediction == label
        tp = int(np.sum(positive & predicted))
        fp = int(np.sum(~positive & predicted))
        fn = int(np.sum(positive & ~predicted))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recalls.append(recall)
        f1s.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "accuracy": float(np.mean(prediction == target)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def _bootstrap(values: np.ndarray, reps: int, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(reps, values.size))].mean(axis=1)
    return {
        "episodes": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def _episode_summary(
    prediction: np.ndarray,
    target: np.ndarray,
    episode_ids: np.ndarray,
    episode_names: list[str],
    *,
    kind: str,
    classes: int,
    permutations: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    metric_fn = (
        _regression_metrics
        if kind == "regression"
        else lambda pred, truth: _classification_metrics(pred, truth, classes)
    )
    rows = []
    unique = np.unique(episode_ids)
    for episode_id in unique:
        mask = episode_ids == episode_id
        row = {
            "episode_id": int(episode_id),
            "stem": episode_names[int(episode_id)],
            "samples": int(mask.sum()),
            **metric_fn(prediction[mask], target[mask]),
        }
        rows.append(row)
    metric_names = [key for key in rows[0] if key not in {"episode_id", "stem", "samples"}]
    aggregate = {
        name: _bootstrap(
            np.asarray([row[name] for row in rows]), bootstrap_reps, seed + index
        )
        for index, name in enumerate(metric_names)
    }
    rng = np.random.default_rng(seed + 1000)
    null: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(permutations):
        per_episode: dict[str, list[float]] = {name: [] for name in metric_names}
        for episode_id in unique:
            mask = episode_ids == episode_id
            shuffled = prediction[mask][rng.permutation(int(mask.sum()))]
            metrics = metric_fn(shuffled, target[mask])
            for name in metric_names:
                per_episode[name].append(metrics[name])
        for name in metric_names:
            null[name].append(float(np.mean(per_episode[name])))
    lower_is_better = {"mae", "rmse"}
    permutation_summary = {}
    for name in metric_names:
        values = np.asarray(null[name], dtype=np.float64)
        observed = float(aggregate[name]["mean"])
        extreme = values <= observed if name in lower_is_better else values >= observed
        permutation_summary[name] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "p_value": float((int(extreme.sum()) + 1) / (values.size + 1)),
        }
    return {"episodes": rows, "aggregate": aggregate, "permutation_null": permutation_summary}


def _run_task(
    data: ProbeData,
    target: np.ndarray,
    episodes: list[Episode],
    masks: dict[str, np.ndarray],
    *,
    kind: str,
    classes: int,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    test_episode_ids = data.episode_ids[masks["test"]]
    test_target = target[masks["test"]]
    for index, (name, features) in enumerate(_interfaces(data.dino, data.raw).items()):
        print(f"[fit] task={kind} interface={name} dim={features.shape[1]}", flush=True)
        prediction, fit = _fit_linear(
            features,
            target,
            masks,
            kind=kind,
            classes=classes,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + seed_offset + index,
        )
        result[name] = {
            **fit,
            "test": _episode_summary(
                prediction,
                test_target,
                test_episode_ids,
                [episode.stem for episode in episodes],
                kind=kind,
                classes=classes,
                permutations=args.permutations,
                bootstrap_reps=args.bootstrap_reps,
                seed=args.seed + 10_000 + seed_offset + index,
            ),
        }
    return result


def main() -> int:
    args = _parse_args()
    requested = args.train_episodes + args.val_episodes + args.test_episodes
    limit = args.max_episodes if args.max_episodes > 0 else requested
    episodes = _load_episodes(args.data_root, args.glob, args.state_key, limit)
    if len(episodes) < requested:
        raise ValueError(f"need {requested} episodes, found {len(episodes)}")
    episodes = episodes[:requested]
    progress, progress_stage = _progress_data(
        episodes,
        args.dino_cache,
        args.image_cache,
        args.progress_stride,
        args.stage_bins,
        args.dino_spatial_pool,
        args.raw_spatial_pool,
    )
    event, anchors = _event_stage_data(
        episodes,
        args.dino_cache,
        args.image_cache,
        args.gripper_event_threshold_deg,
        args.dino_spatial_pool,
        args.raw_spatial_pool,
    )
    progress_masks = _split_masks(
        progress.episode_ids, args.train_episodes, args.val_episodes, args.test_episodes
    )
    event_masks = _split_masks(
        event.episode_ids, args.train_episodes, args.val_episodes, args.test_episodes
    )
    for name, masks in (("progress", progress_masks), ("event", event_masks)):
        if any(int(mask.sum()) == 0 for mask in masks.values()):
            raise ValueError(f"{name} data has an empty train/val/test split")
    device = _device(args.device)
    print(
        f"[probe] device={device} progress_samples={progress.target.size} "
        f"event_samples={event.target.size}",
        flush=True,
    )
    report = {
        "schema": "clearvla-image-stage-readout-v1",
        "contract": {
            "data_root": str(args.data_root),
            "dino_cache": str(args.dino_cache),
            "image_cache": str(args.image_cache),
            "episode_split": {
                "train": args.train_episodes,
                "val": args.val_episodes,
                "test": args.test_episodes,
            },
            "progress_stride": args.progress_stride,
            "stage_bins": args.stage_bins,
            "dino_spatial_pool": args.dino_spatial_pool,
            "raw_spatial_pool": args.raw_spatial_pool,
            "gripper_event_threshold_deg": args.gripper_event_threshold_deg,
            "event_windows_relative_to_close": {
                "pre": [-16, -8],
                "event": [-4, 4],
                "post": [8, 16],
            },
            "statistics": {
                "independent_unit": "episode",
                "bootstrap_reps": args.bootstrap_reps,
                "within_episode_permutations": args.permutations,
            },
            "seed": args.seed,
        },
        "samples": {
            "progress": int(progress.target.size),
            "event_relative": int(event.target.size),
            "event_anchor_status": anchors,
        },
        "progress_regression": _run_task(
            progress,
            progress.target,
            episodes,
            progress_masks,
            kind="regression",
            classes=1,
            args=args,
            device=device,
            seed_offset=0,
        ),
        "progress_stage_classification": _run_task(
            progress,
            progress_stage,
            episodes,
            progress_masks,
            kind="classification",
            classes=args.stage_bins,
            args=args,
            device=device,
            seed_offset=100,
        ),
        "event_relative_stage_classification": _run_task(
            event,
            event.target,
            episodes,
            event_masks,
            kind="classification",
            classes=3,
            args=args,
            device=device,
            seed_offset=200,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
