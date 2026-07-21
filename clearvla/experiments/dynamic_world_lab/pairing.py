from __future__ import annotations

"""Local cross-episode pair construction.

Pairs are selected by current-condition proximity and future-action difference.
They are real trajectories, not synthetic negatives.  The implementation uses
scikit-learn when available and a deterministic chunked Torch fallback.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class LocalPairTable:
    pair_index: np.ndarray
    pair_valid: np.ndarray
    pair_distance: np.ndarray
    action_distance: np.ndarray
    future_distance: np.ndarray | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pair_index=self.pair_index,
            pair_valid=self.pair_valid,
            pair_distance=self.pair_distance,
            action_distance=self.action_distance,
            future_distance=(
                np.zeros_like(self.action_distance, dtype=np.float32)
                if self.future_distance is None
                else self.future_distance
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "LocalPairTable":
        data = np.load(path)
        return cls(
            pair_index=np.asarray(data["pair_index"], dtype=np.int64),
            pair_valid=np.asarray(data["pair_valid"], dtype=np.bool_),
            pair_distance=np.asarray(data["pair_distance"], dtype=np.float32),
            action_distance=np.asarray(data["action_distance"], dtype=np.float32),
            future_distance=(
                np.asarray(data["future_distance"], dtype=np.float32)
                if "future_distance" in data.files
                else np.zeros_like(np.asarray(data["action_distance"], dtype=np.float32))
            ),
        )


def _standardize(
    value: np.ndarray, *, mean: np.ndarray | None = None, std: np.ndarray | None = None
):
    value = np.asarray(value, dtype=np.float32)
    if mean is None:
        mean = value.mean(axis=0, keepdims=True)
    if std is None:
        std = np.maximum(value.std(axis=0, keepdims=True), 1e-4)
    return (value - mean) / std, mean, std


def _neighbours(
    query: np.ndarray, reference: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(count), len(reference))
    try:
        from sklearn.neighbors import NearestNeighbors

        search = NearestNeighbors(n_neighbors=count, algorithm="auto", metric="euclidean")
        search.fit(reference)
        distance, index = search.kneighbors(query, return_distance=True)
        return index.astype(np.int64), distance.astype(np.float32)
    except Exception:
        q = torch.from_numpy(np.asarray(query, dtype=np.float32))
        r = torch.from_numpy(np.asarray(reference, dtype=np.float32))
        all_index, all_distance = [], []
        for start in range(0, len(q), 256):
            distance = torch.cdist(q[start : start + 256], r)
            values, indices = torch.topk(distance, k=count, largest=False, dim=1)
            all_index.append(indices.cpu().numpy())
            all_distance.append(values.cpu().numpy())
        return np.concatenate(all_index), np.concatenate(all_distance)


def build_local_pair_table(
    *,
    condition_descriptor: np.ndarray,
    action_summary: np.ndarray,
    episode_ids: np.ndarray,
    gripper_state: np.ndarray,
    candidate_count: int = 64,
    min_action_distance: float = 1.0,
    future_summary: np.ndarray | None = None,
    min_future_distance: float = 0.0,
) -> LocalPairTable:
    condition, _, _ = _standardize(condition_descriptor)
    action, _, _ = _standardize(action_summary)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if future_summary is None:
        future = np.zeros((len(condition), 1), dtype=np.float32)
    else:
        future, _, _ = _standardize(future_summary)
    gripper_state = np.asarray(gripper_state, dtype=np.int64)
    candidates, distances = _neighbours(condition, condition, candidate_count)
    pair_index = np.arange(len(condition), dtype=np.int64)
    pair_valid = np.zeros(len(condition), dtype=np.bool_)
    pair_distance = np.full(len(condition), np.inf, dtype=np.float32)
    action_distance = np.zeros(len(condition), dtype=np.float32)
    future_distance = np.zeros(len(condition), dtype=np.float32)
    for row in range(len(condition)):
        fallback = None
        for candidate, distance in zip(candidates[row], distances[row], strict=True):
            candidate = int(candidate)
            if candidate == row or episode_ids[candidate] == episode_ids[row]:
                continue
            if gripper_state[candidate] != gripper_state[row]:
                continue
            current_action_distance = float(np.linalg.norm(action[row] - action[candidate]))
            current_future_distance = float(np.linalg.norm(future[row] - future[candidate]))
            if fallback is None:
                fallback = (
                    candidate,
                    float(distance),
                    current_action_distance,
                    current_future_distance,
                )
            if current_action_distance >= float(
                min_action_distance
            ) and current_future_distance >= float(min_future_distance):
                pair_index[row] = candidate
                pair_valid[row] = True
                pair_distance[row] = float(distance)
                action_distance[row] = current_action_distance
                future_distance[row] = current_future_distance
                break
        if not pair_valid[row] and fallback is not None:
            (
                pair_index[row],
                pair_distance[row],
                action_distance[row],
                future_distance[row],
            ) = fallback
    return LocalPairTable(pair_index, pair_valid, pair_distance, action_distance, future_distance)


def nearest_support(
    *,
    query_descriptor: np.ndarray,
    reference_descriptor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference, mean, std = _standardize(reference_descriptor)
    query, _, _ = _standardize(query_descriptor, mean=mean, std=std)
    index, distance = _neighbours(query, reference, 1)
    return index[:, 0].astype(np.int64), distance[:, 0].astype(np.float32)


def nearest_support_distance(
    *,
    query_descriptor: np.ndarray,
    reference_descriptor: np.ndarray,
) -> np.ndarray:
    return nearest_support(
        query_descriptor=query_descriptor, reference_descriptor=reference_descriptor
    )[1]


__all__ = [
    "LocalPairTable",
    "build_local_pair_table",
    "nearest_support",
    "nearest_support_distance",
]
