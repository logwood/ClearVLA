from __future__ import annotations

"""Window datasets for the standalone dynamic predictive world model.

This module deliberately does not import the policy.  A sample contains:

* a short visual history ending at the current frame;
* a longer action trajectory;
* future visual-history windows at several physical offsets;
* the complete future low-dimensional state path used only as an anchor;
* stable episode/frame keys for online or cached DINOv2 encoding.

The target histories make dynamics observable at each prediction time.  They are
not fed back to the closed-loop predictor during rollout.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer


@dataclass(frozen=True)
class DynamicWorldDatasetConfig:
    action_horizon: int = 48
    history_offsets: tuple[int, ...] = (-8, -4, 0)
    future_offsets: tuple[int, ...] = (8, 24, 48)
    target_history_offsets: tuple[int, ...] = (-8, -4, 0)
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    return_images: bool = True

    def validate(self) -> None:
        if self.action_horizon <= 0 or self.stride <= 0:
            raise ValueError("action_horizon and stride must be positive")
        if not self.history_offsets or self.history_offsets[-1] != 0:
            raise ValueError("history_offsets must be non-empty and end at 0")
        if tuple(sorted(set(self.history_offsets))) != self.history_offsets:
            raise ValueError("history_offsets must be strictly increasing and unique")
        if not self.future_offsets or any(int(x) <= 0 for x in self.future_offsets):
            raise ValueError("future_offsets must be non-empty positive integers")
        if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
            raise ValueError("future_offsets must be strictly increasing and unique")
        if max(self.future_offsets) > self.action_horizon:
            raise ValueError("max future offset cannot exceed action_horizon")
        if not self.target_history_offsets or self.target_history_offsets[-1] != 0:
            raise ValueError("target_history_offsets must be non-empty and end at 0")
        if tuple(sorted(set(self.target_history_offsets))) != self.target_history_offsets:
            raise ValueError("target_history_offsets must be strictly increasing and unique")
        if min(self.history_offsets) > 0 or min(self.target_history_offsets) > 0:
            raise ValueError("history offsets must include current/past frames only")
        if len(self.target_history_offsets) != len(self.history_offsets):
            raise ValueError("target_history_offsets must match history_offsets length")


@dataclass(frozen=True)
class DynamicWindowRef:
    episode_idx: int
    center: int


def _camera_stack(frames: Mapping[str, torch.Tensor], camera_names: Sequence[str]) -> torch.Tensor:
    rows = []
    for name in camera_names:
        if name not in frames:
            raise KeyError(f"camera {name!r} missing from decoded frame batch")
        rows.append(frames[name])
    return torch.stack(rows, dim=1)  # [T,Cam,3,H,W]


class DynamicWorldWindowDataset(Dataset):
    """Cross-episode-safe windows for action-conditioned visual dynamics."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: DynamicWorldDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = [int(x) for x in episode_ids]
        self.image_store = image_store
        self.camera_names = tuple(camera_names)
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[DynamicWindowRef] = []

        min_history = min(config.history_offsets)
        min_target_history = min(config.target_history_offsets)
        max_future = max(config.future_offsets)
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            # All indices below are relative to center.  State/action arrays and
            # decoded images share the same synchronized frame index contract.
            min_target_relative = min(config.future_offsets) + min_target_history
            low = max(
                -config.image_offset - min_history,
                -config.image_offset - min_target_relative,
                -config.state_offset - min_history,
                -config.state_offset - min_target_relative,
                -config.state_offset,
                -config.action_offset,
                0,
            )
            high = min(
                episode.length - 1 - config.image_offset - max_future,
                episode.length - 1 - config.state_offset - max_future,
                episode.length - 1 - config.state_offset - config.action_horizon,
                episode.length - config.action_offset - config.action_horizon,
            )
            for center in range(int(low), int(high) + 1, config.stride):
                self.refs.append(DynamicWindowRef(episode_idx=episode_idx, center=center))
        if not self.refs:
            raise ValueError("dynamic-world dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def _history_indices(self, center: int) -> np.ndarray:
        base = center + self.config.image_offset
        return np.asarray(
            [base + int(offset) for offset in self.config.history_offsets], dtype=np.int64
        )

    def _target_indices(self, center: int) -> np.ndarray:
        base = center + self.config.image_offset
        return np.asarray(
            [
                [
                    base + int(future) + int(history)
                    for history in self.config.target_history_offsets
                ]
                for future in self.config.future_offsets
            ],
            dtype=np.int64,
        )

    def descriptor_metadata(self, index: int) -> dict[str, np.ndarray | int]:
        """Cheap metadata used by local-pair indexing before image loading."""
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_index = ref.center + cfg.state_offset
        action_start = ref.center + cfg.action_offset
        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.action_horizon], dtype=np.float32
        )
        sampled = action_raw[np.asarray(cfg.future_offsets, dtype=np.int64) - 1]
        boundary = np.concatenate([state_raw[None], action_raw[:-1]], axis=0)
        velocity = action_raw - boundary
        summary = np.concatenate(
            [
                sampled.reshape(-1),
                velocity.mean(axis=0),
                velocity.std(axis=0),
                action_raw[-1] - state_raw,
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "episode_idx": int(ref.episode_idx),
            "center": int(ref.center),
            "state": self.state_normalizer.encode(state_raw).astype(np.float32),
            "state_raw": state_raw,
            "action_summary": summary,
            "history_keys": np.stack(
                [
                    np.full(len(self.config.history_offsets), ref.episode_idx),
                    self._history_indices(ref.center),
                ],
                axis=1,
            ).astype(np.int64),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_index = ref.center + cfg.state_offset
        action_start = ref.center + cfg.action_offset

        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.action_horizon], dtype=np.float32
        )
        # Future state path is one state per action step.  This unified physical
        # anchor is decoded from shared world tokens; no arm/gripper branch is
        # allowed to bypass the world representation.
        future_state_raw = np.asarray(
            episode.states_raw[state_index + 1 : state_index + cfg.action_horizon + 1],
            dtype=np.float32,
        )
        if future_state_raw.shape[0] != cfg.action_horizon:
            raise RuntimeError("future state path length violates dataset bounds")

        history_indices = self._history_indices(ref.center)
        target_indices = self._target_indices(ref.center)
        state_base = ref.center + cfg.state_offset
        history_state_indices = np.asarray(
            [state_base + int(offset) for offset in cfg.history_offsets], dtype=np.int64
        )
        target_history_state_indices = np.asarray(
            [
                [state_base + int(future) + int(history) for history in cfg.target_history_offsets]
                for future in cfg.future_offsets
            ],
            dtype=np.int64,
        )
        history_state_raw = np.asarray(episode.states_raw[history_state_indices], dtype=np.float32)
        target_history_state_raw = np.asarray(
            episode.states_raw[target_history_state_indices], dtype=np.float32
        )
        history_keys = np.stack(
            [np.full(len(history_indices), ref.episode_idx), history_indices], axis=1
        ).astype(np.int64)
        target_keys = np.stack(
            [np.stack([np.full(len(row), ref.episode_idx), row], axis=1) for row in target_indices],
            axis=0,
        ).astype(np.int64)

        sample: dict[str, torch.Tensor] = {
            "sample_index": torch.tensor(int(index), dtype=torch.long),
            "episode_idx": torch.tensor(ref.episode_idx, dtype=torch.long),
            "center": torch.tensor(ref.center, dtype=torch.long),
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "action_state": torch.from_numpy(self.action_normalizer.encode(state_raw)),
            "state_raw": torch.from_numpy(state_raw.copy()),
            "action": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "action_raw": torch.from_numpy(action_raw.copy()),
            "future_state": torch.from_numpy(self.state_normalizer.encode(future_state_raw)),
            "future_state_raw": torch.from_numpy(future_state_raw.copy()),
            "history_state": torch.from_numpy(self.state_normalizer.encode(history_state_raw)),
            "history_state_raw": torch.from_numpy(history_state_raw.copy()),
            "target_history_state": torch.from_numpy(
                self.state_normalizer.encode(target_history_state_raw)
            ),
            "target_history_state_raw": torch.from_numpy(target_history_state_raw.copy()),
            "history_keys": torch.from_numpy(history_keys),
            "target_history_keys": torch.from_numpy(target_keys),
            "future_offsets": torch.tensor(cfg.future_offsets, dtype=torch.long),
        }
        if cfg.return_images:
            history_frames = self.image_store.load_window(episode, history_indices)
            history_images = (
                _camera_stack(history_frames, self.camera_names).to(torch.float32) / 255.0
            )
            flat_target = target_indices.reshape(-1)
            target_frames = self.image_store.load_window(episode, flat_target)
            target_images = (
                _camera_stack(target_frames, self.camera_names).to(torch.float32) / 255.0
            )
            target_images = target_images.reshape(
                len(cfg.future_offsets), len(cfg.target_history_offsets), *target_images.shape[1:]
            )
            sample["history_obs_image"] = history_images
            sample["target_history_obs_image"] = target_images
        return sample


class PairedDynamicWorldDataset(Dataset):
    """Attach a real cross-episode local neighbour to every sample.

    Pair indices are built from current-state descriptors and require a minimum
    future-action difference.  The pair is a second real trajectory, not a
    synthetic corruption.  Invalid rows remain available for ordinary
    predictive training but are masked out of local-effect losses.
    """

    def __init__(
        self,
        base: DynamicWorldWindowDataset,
        *,
        pair_index: np.ndarray,
        pair_valid: np.ndarray,
        pair_distance: np.ndarray,
        action_distance: np.ndarray,
        future_distance: np.ndarray | None = None,
        support_distance: np.ndarray | None = None,
        support_base: DynamicWorldWindowDataset | None = None,
        support_index: np.ndarray | None = None,
    ) -> None:
        self.base = base
        self.pair_index = np.asarray(pair_index, dtype=np.int64)
        self.pair_valid = np.asarray(pair_valid, dtype=np.bool_)
        self.pair_distance = np.asarray(pair_distance, dtype=np.float32)
        self.action_distance = np.asarray(action_distance, dtype=np.float32)
        self.future_distance = (
            np.zeros(len(base), dtype=np.float32)
            if future_distance is None
            else np.asarray(future_distance, dtype=np.float32)
        )
        self.support_distance = (
            np.zeros(len(base), dtype=np.float32)
            if support_distance is None
            else np.asarray(support_distance, dtype=np.float32)
        )
        self.support_base = support_base
        self.support_index = (
            None if support_index is None else np.asarray(support_index, dtype=np.int64)
        )
        if (self.support_base is None) != (self.support_index is None):
            raise ValueError("support_base and support_index must be supplied together")
        expected = (len(base),)
        validation = {
            "pair_index": self.pair_index,
            "pair_valid": self.pair_valid,
            "pair_distance": self.pair_distance,
            "action_distance": self.action_distance,
            "future_distance": self.future_distance,
            "support_distance": self.support_distance,
        }
        if self.support_index is not None:
            validation["support_index"] = self.support_index
        for name, value in validation.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair_idx = int(self.pair_index[int(index)])
        if pair_idx < 0 or pair_idx >= len(self.base):
            pair_idx = int(index)
        out: dict[str, object] = {
            "primary": self.base[int(index)],
            "pair": self.base[pair_idx],
            "pair_valid": torch.tensor(bool(self.pair_valid[int(index)]), dtype=torch.bool),
            "pair_distance": torch.tensor(
                float(self.pair_distance[int(index)]), dtype=torch.float32
            ),
            "action_distance": torch.tensor(
                float(self.action_distance[int(index)]), dtype=torch.float32
            ),
            "future_distance": torch.tensor(
                float(self.future_distance[int(index)]), dtype=torch.float32
            ),
            "support_distance": torch.tensor(
                float(self.support_distance[int(index)]), dtype=torch.float32
            ),
        }
        if self.support_base is not None and self.support_index is not None:
            support_idx = int(self.support_index[int(index)])
            if support_idx < 0 or support_idx >= len(self.support_base):
                raise IndexError("support index outside support dataset")
            out["support"] = self.support_base[support_idx]
            out["support_index"] = torch.tensor(support_idx, dtype=torch.long)
        return out


__all__ = [
    "DynamicWorldDatasetConfig",
    "DynamicWindowRef",
    "DynamicWorldWindowDataset",
    "PairedDynamicWorldDataset",
]


class CurrentHistoryViewDataset(Dataset):
    """Lightweight view used only while constructing local-pair descriptors.

    Online DINO pair indexing should not decode every future target frame.  This
    view exposes the current history, state and action summary inputs only.
    """

    def __init__(self, base: DynamicWorldWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.base.refs[int(index)]
        episode = self.base.episodes[ref.episode_idx]
        cfg = self.base.config
        state_index = ref.center + cfg.state_offset
        action_start = ref.center + cfg.action_offset
        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.action_horizon], dtype=np.float32
        )
        history_indices = self.base._history_indices(ref.center)
        history_keys = np.stack(
            [np.full(len(history_indices), ref.episode_idx), history_indices], axis=1
        ).astype(np.int64)
        future_state_raw = np.asarray(
            episode.states_raw[state_index + 1 : state_index + cfg.action_horizon + 1],
            dtype=np.float32,
        )
        sample = {
            "sample_index": torch.tensor(int(index), dtype=torch.long),
            "episode_idx": torch.tensor(ref.episode_idx, dtype=torch.long),
            "state": torch.from_numpy(self.base.state_normalizer.encode(state_raw)),
            "action_state": torch.from_numpy(self.base.action_normalizer.encode(state_raw)),
            "state_raw": torch.from_numpy(state_raw.copy()),
            "action": torch.from_numpy(self.base.action_normalizer.encode(action_raw)),
            "future_state": torch.from_numpy(self.base.state_normalizer.encode(future_state_raw)),
            "future_state_raw": torch.from_numpy(future_state_raw.copy()),
            "history_keys": torch.from_numpy(history_keys),
        }
        if cfg.return_images:
            history_frames = self.base.image_store.load_window(episode, history_indices)
            sample["history_obs_image"] = (
                _camera_stack(history_frames, self.base.camera_names).to(torch.float32) / 255.0
            )
        return sample


__all__.append("CurrentHistoryViewDataset")
