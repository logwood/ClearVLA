"""Probe action/control information in an existing DINOv2 patch-token cache.

This tool deliberately does not instantiate a DINO model.  It reads the
episode-aligned mmap cache produced by ``build_dinov2_token_cache``, constructs
small fixed feature interfaces, and fits one linear multi-output readout per
interface.  Episode splits prevent adjacent-frame leakage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from clearvla.data.hdf5_episode import find_hdf5_files
from clearvla.data.schema import ACTION_ALIASES, STATE_ALIASES, list_hdf5_datasets, resolve_key


@dataclass(frozen=True)
class EpisodeArrays:
    stem: str
    state: np.ndarray
    action: np.ndarray

    @property
    def length(self) -> int:
        return int(self.state.shape[0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-key", default="qpos")
    parser.add_argument("--train-episodes", type=int, default=63)
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--spatial-pool", type=int, default=2)
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--dynamic-lags", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gripper-event-threshold-deg", type=float, default=5.0)
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=0)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_episode(path: Path, action_key: str, state_key: str) -> EpisodeArrays:
    datasets = list_hdf5_datasets(str(path))
    resolved_action = resolve_key(datasets, action_key, ACTION_ALIASES, required=True)
    resolved_state = resolve_key(datasets, state_key, STATE_ALIASES, required=True)
    assert resolved_action is not None and resolved_state is not None
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle[resolved_action], dtype=np.float32)
        state = np.asarray(handle[resolved_state], dtype=np.float32)
    if action.ndim != 2 or state.shape != action.shape:
        raise ValueError(
            f"{path}: expected aligned state/action [T,D], got {state.shape} and {action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: state/action contains non-finite values")
    return EpisodeArrays(path.stem, state, action)


def _load_meta(cache_root: Path, stem: str) -> dict[str, Any]:
    path = cache_root / stem / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _token_array(cache_root: Path, stem: str) -> np.ndarray:
    path = cache_root / stem / "tokens.float16.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing token cache: {path}")
    return np.load(path, mmap_mode="r")


def _spatial_pool(tokens: np.ndarray, output_grid: int) -> np.ndarray:
    """Equal-area average pool [T,C,P,D] square patch tokens to [T,C,G,G,D]."""
    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [T,C,P,D], got {tokens.shape}")
    patch_grid = math.isqrt(int(tokens.shape[2]))
    if patch_grid * patch_grid != int(tokens.shape[2]):
        raise ValueError(f"patch count must be square, got {tokens.shape[2]}")
    if output_grid <= 0 or patch_grid % output_grid:
        raise ValueError(f"spatial_pool={output_grid} must divide patch grid={patch_grid}")
    block = patch_grid // output_grid
    values = np.asarray(tokens, dtype=np.float32).reshape(
        tokens.shape[0], tokens.shape[1], output_grid, block, output_grid, block, tokens.shape[3]
    )
    return values.mean(axis=(3, 5), dtype=np.float32)


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
    x = _rankdata(left)
    y = _rankdata(right)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def _collect_features(
    episodes: list[EpisodeArrays],
    cache_root: Path,
    *,
    stride: int,
    spatial_pool: int,
    future_horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
]:
    spatial_rows: list[np.ndarray] = []
    future_spatial_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    episode_rows: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    cache_contract: dict[str, Any] | None = None
    for episode_id, episode in enumerate(episodes):
        meta = _load_meta(cache_root, episode.stem)
        tokens = _token_array(cache_root, episode.stem)
        expected = (
            episode.length,
            len(meta["cameras"]),
            int(meta["tokens_per_camera"]),
            int(meta["token_dim"]),
        )
        if tuple(tokens.shape) != expected:
            raise ValueError(f"{episode.stem}: token shape={tokens.shape}, expected={expected}")
        if episode.length <= future_horizon:
            continue
        indices = np.arange(0, episode.length - future_horizon, stride, dtype=np.int64)
        sampled = _spatial_pool(tokens[indices], spatial_pool).astype(np.float16)
        future_sampled = _spatial_pool(tokens[indices + future_horizon], spatial_pool).astype(
            np.float16
        )
        state = episode.state[indices]
        action = episode.action[indices]
        state_delta = episode.state[indices + future_horizon] - state
        target = np.concatenate((state, action, action - state, state_delta), axis=1)
        spatial_rows.append(sampled)
        future_spatial_rows.append(future_sampled)
        target_rows.append(target.astype(np.float32, copy=False))
        episode_rows.append(np.full(indices.size, episode_id, dtype=np.int32))
        sample_rows.append(
            {
                "episode_id": episode_id,
                "stem": episode.stem,
                "frames": episode.length,
                "samples": int(indices.size),
            }
        )
        contract = {
            "cameras": list(meta["cameras"]),
            "dinov2_model": str(meta["dinov2_model"]),
            "token_dim": int(meta["token_dim"]),
            "tokens_per_camera": int(meta["tokens_per_camera"]),
            "cache_version": str(meta["cache_version"]),
        }
        if cache_contract is None:
            cache_contract = contract
        elif contract != cache_contract:
            raise ValueError(f"cache contract changed at {episode.stem}: {contract} != {cache_contract}")
        print(
            f"[collect] episode={episode_id + 1:03d}/{len(episodes):03d} "
            f"stem={episode.stem} samples={indices.size}",
            flush=True,
        )
    if not spatial_rows or cache_contract is None:
        raise RuntimeError("no probe samples collected")
    return (
        np.concatenate(spatial_rows, axis=0),
        np.concatenate(future_spatial_rows, axis=0),
        np.concatenate(target_rows, axis=0),
        np.concatenate(episode_rows, axis=0),
        sample_rows,
        cache_contract,
    )


def _feature_interfaces(spatial: np.ndarray) -> dict[str, np.ndarray]:
    if spatial.ndim != 5 or spatial.shape[1] != 2:
        raise ValueError(f"expected pooled features [N,2,G,G,D], got {spatial.shape}")
    mean = spatial.astype(np.float32).mean(axis=(2, 3))
    return {
        "mean_top": mean[:, 0],
        "mean_wrist": mean[:, 1],
        "mean_both": mean.reshape(mean.shape[0], -1),
        "spatial_top": spatial[:, 0].reshape(spatial.shape[0], -1),
        "spatial_wrist": spatial[:, 1].reshape(spatial.shape[0], -1),
        "spatial_both": spatial.reshape(spatial.shape[0], -1),
    }


def _split_masks(
    episode_ids: np.ndarray, train_episodes: int, val_episodes: int, test_episodes: int
) -> dict[str, np.ndarray]:
    val_start = train_episodes
    test_start = train_episodes + val_episodes
    total = train_episodes + val_episodes + test_episodes
    if int(episode_ids.max()) + 1 < total:
        raise ValueError(
            f"requested {total} episodes but only {int(episode_ids.max()) + 1} were collected"
        )
    return {
        "train": episode_ids < val_start,
        "val": (episode_ids >= val_start) & (episode_ids < test_start),
        "test": (episode_ids >= test_start) & (episode_ids < total),
    }


def _group_slices(action_dim: int) -> dict[str, tuple[int, int]]:
    grip = action_dim - 1
    return {
        "state_all": (0, action_dim),
        "state_arm": (0, grip),
        "state_gripper": (grip, action_dim),
        "action_all": (action_dim, 2 * action_dim),
        "action_arm": (action_dim, action_dim + grip),
        "action_gripper": (action_dim + grip, 2 * action_dim),
        "command_residual_all": (2 * action_dim, 3 * action_dim),
        "command_residual_arm": (2 * action_dim, 2 * action_dim + grip),
        "command_residual_gripper": (2 * action_dim + grip, 3 * action_dim),
        "future_state_delta_all": (3 * action_dim, 4 * action_dim),
        "future_state_delta_arm": (3 * action_dim, 3 * action_dim + grip),
        "future_state_delta_gripper": (3 * action_dim + grip, 4 * action_dim),
    }


def _metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    train_target: np.ndarray,
    action_dim: int,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, (start, stop) in _group_slices(action_dim).items():
        pred = prediction[:, start:stop].astype(np.float64)
        truth = target[:, start:stop].astype(np.float64)
        train = train_target[:, start:stop].astype(np.float64)
        error = pred - truth
        rmse = float(np.sqrt(np.square(error).mean()))
        scale = float(np.sqrt(np.square(train - train.mean(axis=0, keepdims=True)).mean()))
        denominator = float(np.square(truth - train.mean(axis=0, keepdims=True)).sum())
        r2 = 1.0 - float(np.square(error).sum()) / max(denominator, 1e-12)
        result[name] = {
            "rmse": rmse,
            "nrmse_train_rms": rmse / max(scale, 1e-12),
            "r2": r2,
        }
    return result


def _delta_metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    train_target: np.ndarray,
    action_dim: int,
) -> dict[str, dict[str, float]]:
    groups = {
        "delta_all": (0, action_dim),
        "delta_arm": (0, action_dim - 1),
        "delta_gripper": (action_dim - 1, action_dim),
    }
    result: dict[str, dict[str, float]] = {}
    for name, (start, stop) in groups.items():
        pred = prediction[:, start:stop].astype(np.float64)
        truth = target[:, start:stop].astype(np.float64)
        train = train_target[:, start:stop].astype(np.float64)
        error = pred - truth
        rmse = float(np.sqrt(np.square(error).mean()))
        train_mean = train.mean(axis=0, keepdims=True)
        scale = float(np.sqrt(np.square(train - train_mean).mean()))
        denominator = float(np.square(truth - train_mean).sum())
        r2 = 1.0 - float(np.square(error).sum()) / max(denominator, 1e-12)
        result[name] = {
            "rmse": rmse,
            "nrmse_train_rms": rmse / max(scale, 1e-12),
            "r2": r2,
        }
    return result


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = _rankdata(scores) + 1.0
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(precision[ordered == 1].sum() / positives)


def _binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    prediction = logits >= 0.0
    positive = labels == 1
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & ~positive))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & ~positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "samples": int(labels.size),
        "positives": int(positive.sum()),
        "positive_rate": float(positive.mean()),
        "auroc": _binary_auc(logits, labels),
        "average_precision": _average_precision(logits, labels),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "balanced_accuracy_at_0_5": 0.5 * (recall + specificity),
        "accuracy_at_0_5": (tp + tn) / max(labels.size, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _train_binary_probe(
    features: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    x_train = features[masks["train"]].astype(np.float32)
    y_train = labels[masks["train"]].astype(np.float32)
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-5] = 1.0
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("binary probe training split must contain positive and negative examples")
    pos_weight = negatives / positives
    model = nn.Linear(features.shape[1], 1, bias=True).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    generator = torch.Generator().manual_seed(seed)
    x_train_tensor = torch.from_numpy((x_train - x_mean) / x_std)
    y_train_tensor = torch.from_numpy(y_train)
    x_val_tensor = torch.from_numpy(
        (features[masks["val"]].astype(np.float32) - x_mean) / x_std
    )
    y_val_tensor = torch.from_numpy(labels[masks["val"]].astype(np.float32))
    best_state: dict[str, Tensor] | None = None
    best_val = math.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.shape[0], generator=generator)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            x = x_train_tensor[index].to(device, non_blocking=True)
            y = y_train_tensor[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * int(index.numel())
            train_count += int(index.numel())
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for start in range(0, x_val_tensor.shape[0], batch_size):
                x = x_val_tensor[start : start + batch_size].to(device)
                y = y_val_tensor[start : start + batch_size].to(device)
                loss = loss_fn(model(x).squeeze(-1), y)
                count = int(x.shape[0])
                val_loss_sum += float(loss) * count
                val_count += count
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history.append({"epoch": epoch + 1, "train_bce": train_loss, "val_bce": val_loss})
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
        raise RuntimeError("binary probe never produced a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics: dict[str, Any] = {}
    for split in ("val", "test"):
        x = (features[masks[split]].astype(np.float32) - x_mean) / x_std
        chunks = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.from_numpy(x[start : start + batch_size]).to(device)
                chunks.append(model(batch).squeeze(-1).cpu().numpy())
        metrics[split] = _binary_metrics(
            np.concatenate(chunks), labels[masks[split]]
        )
    return {
        "feature_dim": int(features.shape[1]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "positive_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_bce": best_val,
        "history": history,
        "metrics": metrics,
    }


def _train_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    action_dim: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    metric_kind: str = "bundle",
) -> dict[str, Any]:
    if metric_kind not in {"bundle", "delta"}:
        raise ValueError(f"unsupported metric_kind={metric_kind!r}")
    _seed_everything(seed)
    x_train = features[masks["train"]].astype(np.float32)
    y_train = targets[masks["train"]].astype(np.float32)
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-5] = 1.0
    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = y_train.std(axis=0, dtype=np.float64).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0

    model = nn.Linear(features.shape[1], targets.shape[1], bias=True).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    x_train_tensor = torch.from_numpy((x_train - x_mean) / x_std)
    y_train_tensor = torch.from_numpy((y_train - y_mean) / y_std)
    x_val_tensor = torch.from_numpy(
        (features[masks["val"]].astype(np.float32) - x_mean) / x_std
    )
    y_val_tensor = torch.from_numpy(
        (targets[masks["val"]].astype(np.float32) - y_mean) / y_std
    )
    best_state: dict[str, Tensor] | None = None
    best_val = math.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train_tensor.shape[0], generator=generator)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            x = x_train_tensor[index].to(device, non_blocking=True)
            y = y_train_tensor[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * int(index.numel())
            train_count += int(index.numel())
        model.eval()
        with torch.inference_mode():
            val_loss_sum = 0.0
            val_count = 0
            for start in range(0, x_val_tensor.shape[0], batch_size):
                x = x_val_tensor[start : start + batch_size].to(device, non_blocking=True)
                y = y_val_tensor[start : start + batch_size].to(device, non_blocking=True)
                loss = torch.nn.functional.mse_loss(model(x), y)
                count = int(x.shape[0])
                val_loss_sum += float(loss) * count
                val_count += count
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history.append({"epoch": epoch + 1, "train_mse_z": train_loss, "val_mse_z": val_loss})
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
        raise RuntimeError("linear probe never produced a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        x = ((features[masks[split]].astype(np.float32) - x_mean) / x_std)
        chunks = []
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.from_numpy(x[start : start + batch_size]).to(device)
                chunks.append(model(batch).cpu().numpy())
        prediction_z = np.concatenate(chunks, axis=0)
        prediction = prediction_z * y_std + y_mean
        predictions[split] = prediction
        metric_fn = _metric_rows if metric_kind == "bundle" else _delta_metric_rows
        metrics[split] = metric_fn(prediction, targets[masks[split]], y_train, action_dim)
    constant = np.broadcast_to(y_mean, targets[masks["test"]].shape)
    metric_fn = _metric_rows if metric_kind == "bundle" else _delta_metric_rows
    return {
        "feature_dim": int(features.shape[1]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": best_epoch,
        "best_val_mse_z": best_val,
        "history": history,
        "metrics": metrics,
        "constant_test_metrics": metric_fn(
            constant, targets[masks["test"]], y_train, action_dim
        ),
    }


def _dynamic_sensitivity(
    episodes: list[EpisodeArrays],
    cache_root: Path,
    *,
    lags: list[int],
    stride: int,
    episode_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selected = episodes[:episode_limit]
    for lag in lags:
        measures: dict[str, list[np.ndarray]] = {
            "top_mean": [],
            "wrist_mean": [],
            "top_patch": [],
            "wrist_patch": [],
            "state_arm": [],
            "state_gripper": [],
            "action_arm": [],
            "action_gripper": [],
        }
        for episode in selected:
            if episode.length <= lag:
                continue
            tokens = _token_array(cache_root, episode.stem)
            left_index = np.arange(0, episode.length - lag, stride, dtype=np.int64)
            right_index = left_index + lag
            left = np.asarray(tokens[left_index], dtype=np.float32)
            right = np.asarray(tokens[right_index], dtype=np.float32)
            delta = right - left
            mean_delta = delta.mean(axis=2)
            patch_rms = np.sqrt(np.square(delta, dtype=np.float32).mean(axis=(2, 3)))
            measures["top_mean"].append(np.linalg.norm(mean_delta[:, 0], axis=1))
            measures["wrist_mean"].append(np.linalg.norm(mean_delta[:, 1], axis=1))
            measures["top_patch"].append(patch_rms[:, 0])
            measures["wrist_patch"].append(patch_rms[:, 1])
            state_delta = episode.state[right_index] - episode.state[left_index]
            action_delta = episode.action[right_index] - episode.action[left_index]
            measures["state_arm"].append(np.linalg.norm(state_delta[:, :-1], axis=1))
            measures["state_gripper"].append(np.abs(state_delta[:, -1]))
            measures["action_arm"].append(np.linalg.norm(action_delta[:, :-1], axis=1))
            measures["action_gripper"].append(np.abs(action_delta[:, -1]))
        joined = {name: np.concatenate(rows) for name, rows in measures.items()}
        correlation: dict[str, dict[str, float]] = {}
        for feature_name in ("top_mean", "wrist_mean", "top_patch", "wrist_patch"):
            correlation[feature_name] = {
                target_name: _spearman(joined[feature_name], joined[target_name])
                for target_name in ("state_arm", "state_gripper", "action_arm", "action_gripper")
            }
        result[str(lag)] = {
            "samples": int(joined["top_mean"].size),
            "spearman": correlation,
        }
    return result


def main() -> int:
    args = _parse_args()
    if args.stride <= 0 or args.spatial_pool <= 0 or args.future_horizon <= 0:
        raise ValueError("stride, spatial-pool, and future-horizon must be positive")
    requested = args.train_episodes + args.val_episodes + args.test_episodes
    files = find_hdf5_files(args.data_root, args.glob)
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
        requested = min(requested, len(files))
    if len(files) < requested:
        raise ValueError(f"need {requested} episodes, found {len(files)}")
    episodes = [
        _load_episode(path, action_key=args.action_key, state_key=args.state_key)
        for path in files[:requested]
    ]
    action_dim = int(episodes[0].action.shape[1])
    if action_dim < 2 or any(episode.action.shape[1] != action_dim for episode in episodes):
        raise ValueError("all episodes must share an action dimension >= 2")
    spatial, future_spatial, targets, episode_ids, sample_rows, cache_contract = _collect_features(
        episodes,
        args.dino_cache,
        stride=args.stride,
        spatial_pool=args.spatial_pool,
        future_horizon=args.future_horizon,
    )
    masks = _split_masks(
        episode_ids, args.train_episodes, args.val_episodes, args.test_episodes
    )
    device = _device(args.device)
    print(
        f"[probe] device={device} samples={targets.shape[0]} action_dim={action_dim} "
        f"spatial_shape={tuple(spatial.shape)}",
        flush=True,
    )
    probes: dict[str, Any] = {}
    if not args.event_only:
        for name, features in _feature_interfaces(spatial).items():
            print(f"[probe] interface={name} dim={features.shape[1]}", flush=True)
            probes[name] = _train_linear_probe(
                features,
                targets,
                masks,
                action_dim=action_dim,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
    delta_targets = targets[:, 3 * action_dim : 4 * action_dim]
    delta_spatial = future_spatial.astype(np.float32) - spatial.astype(np.float32)
    delta_interfaces = _feature_interfaces(delta_spatial)
    delta_probe_names = ("mean_top", "mean_wrist", "mean_both", "spatial_both")
    delta_probes: dict[str, Any] = {}
    if not args.event_only:
        for name in delta_probe_names:
            features = delta_interfaces[name]
            print(f"[delta-probe] interface={name} dim={features.shape[1]}", flush=True)
            delta_probes[name] = _train_linear_probe(
                features,
                delta_targets,
                masks,
                action_dim=action_dim,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                metric_kind="delta",
            )
    event_threshold = math.radians(float(args.gripper_event_threshold_deg))
    event_labels = (np.abs(delta_targets[:, -1]) >= event_threshold).astype(np.int64)
    event_probes: dict[str, Any] = {}
    for name in delta_probe_names:
        features = delta_interfaces[name]
        print(
            f"[event-probe] interface={name} dim={features.shape[1]} "
            f"event_rate={event_labels.mean():.4f}",
            flush=True,
        )
        event_probes[name] = _train_binary_probe(
            features,
            event_labels,
            masks,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    dynamic = (
        {}
        if args.event_only
        else _dynamic_sensitivity(
            episodes,
            args.dino_cache,
            lags=list(args.dynamic_lags),
            stride=args.stride,
            episode_limit=args.train_episodes + args.val_episodes,
        )
    )
    report = {
        "schema": "clearvla-dino-control-readout-v1",
        "contract": {
            "data_root": str(args.data_root),
            "dino_cache": str(args.dino_cache),
            "episode_split": {
                "train": args.train_episodes,
                "val": args.val_episodes,
                "test": args.test_episodes,
            },
            "stride": args.stride,
            "spatial_pool": args.spatial_pool,
            "future_horizon": args.future_horizon,
            "gripper_event_threshold_deg": args.gripper_event_threshold_deg,
            "dynamic_lags": list(args.dynamic_lags),
            "action_dim": action_dim,
            "seed": args.seed,
            "optimizer": {
                "name": "AdamW",
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "cache": cache_contract,
            "limitations": [
                "cache contains final-layer patch tokens only; CLS and intermediate layers are unavailable",
                "HDF5 labels expose qpos and action but not contact or object pose",
                "action readout measures dataset-conditional predictability, not causal action evidence",
            ],
        },
        "samples": {
            "total": int(targets.shape[0]),
            "train": int(masks["train"].sum()),
            "val": int(masks["val"].sum()),
            "test": int(masks["test"].sum()),
            "episodes": sample_rows,
        },
        "data_audit": {
            "state_action_residual_max_abs": float(
                np.abs(targets[:, 2 * action_dim : 3 * action_dim]).max()
            ),
            "state_action_residual_rms": float(
                np.sqrt(
                    np.square(targets[:, 2 * action_dim : 3 * action_dim], dtype=np.float32).mean()
                )
            ),
            "state_action_exact_equal_fraction": float(
                np.all(
                    targets[:, 2 * action_dim : 3 * action_dim] == 0.0,
                    axis=1,
                ).mean()
            ),
            "gripper_event_rate": float(event_labels.mean()),
        },
        "probes": probes,
        "delta_probes": delta_probes,
        "event_probes": event_probes,
        "dynamic_sensitivity": dynamic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
