"""Minimal observed-state dataset owned by the capability mainline.

Only values consumed by :class:`TrainingBatch` are materialized.  Future RGB,
target-history tensors and ancestry-only descriptor views are deliberately
absent: the teacher consumes twelve cached DINO supports at offsets 4..48.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.decoded_image_store import DecodedImageStore

from .normalizer import ArrayNormalizer
from .token_store import DinoV2TokenStore


@dataclass(frozen=True)
class ObservedStateDatasetConfig:
    world_horizon: int = 48
    policy_horizon: int = 24
    support_stride: int = 4
    state_history_offsets: tuple[int, ...] = (-8, -4, 0)
    raw_pair_offsets: tuple[int, ...] = (-4, 0)
    executed_action_offsets: tuple[int, ...] = (-24, -16, -12, -8, -6, -4, -2, -1)
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1

    def validate(self) -> None:
        if (
            min(
                self.world_horizon,
                self.policy_horizon,
                self.support_stride,
                self.stride,
            )
            <= 0
        ):
            raise ValueError("horizons and strides must be positive")
        if self.world_horizon % self.support_stride:
            raise ValueError("world_horizon must be divisible by support_stride")
        if self.policy_horizon > self.world_horizon:
            raise ValueError("policy_horizon cannot exceed world_horizon")
        if (
            not self.state_history_offsets
            or self.state_history_offsets[-1] != 0
            or tuple(sorted(set(self.state_history_offsets))) != self.state_history_offsets
        ):
            raise ValueError("state_history_offsets must be increasing and end at zero")
        if self.raw_pair_offsets != (-4, 0):
            raise ValueError("the learned-flow input is exactly the previous/current raw pair")
        if not self.executed_action_offsets or max(self.executed_action_offsets) >= 0:
            raise ValueError("executed_action_offsets must contain only past actions")
        if tuple(sorted(set(self.executed_action_offsets))) != self.executed_action_offsets:
            raise ValueError("executed_action_offsets must be strictly increasing")

    @property
    def future_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.support_stride, self.world_horizon + 1, self.support_stride))


@dataclass(frozen=True)
class ObservedWindowRef:
    episode_idx: int
    center: int


def _camera_stack(frames: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.stack([frames[name] for name in names], dim=1)


class ObservedStateWindowDataset(Dataset):
    """Return current observable evidence plus disjoint action/future targets."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: ObservedStateDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = [int(value) for value in episode_ids]
        self.image_store = image_store
        self.camera_names = tuple(camera_names)
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[ObservedWindowRef] = []

        min_rel = min(
            min(config.raw_pair_offsets) + config.image_offset,
            min(config.state_history_offsets) + config.state_offset,
            min(config.executed_action_offsets) + config.action_offset,
        )
        max_rel = max(
            config.world_horizon + config.action_offset - 1,
            config.world_horizon + config.state_offset,
            max(config.future_offsets) + config.image_offset,
        )
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(0, -int(min_rel))
            high = int(episode.length - 1 - max_rel)
            for center in range(low, high + 1, config.stride):
                self.refs.append(ObservedWindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("mainline dataset has no valid 48-frame windows")

    def __len__(self) -> int:
        return len(self.refs)

    def training_information_signals(
        self,
        *,
        gripper_index: int,
        event_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the formal sampling strata without image/cache reads."""

        motion = np.empty((len(self.refs),), dtype=np.float32)
        event = np.zeros((len(self.refs),), dtype=bool)
        cfg = self.config
        grip_index = int(gripper_index)
        action_dim = int(self.action_normalizer.minimum.shape[-1])
        if grip_index < 0:
            grip_index += action_dim
        if not 0 <= grip_index < action_dim:
            raise ValueError("gripper_index is outside the action normalizer dimension")
        if float(event_threshold) < 0.0:
            raise ValueError("event_threshold must be non-negative")
        for index, ref in enumerate(self.refs):
            episode = self.episodes[ref.episode_idx]
            states_raw = episode.states_raw
            actions_raw = episode.actions_raw
            if states_raw is None or actions_raw is None:
                raise ValueError("mainline policy episodes require state and action arrays")
            state_raw = np.asarray(
                states_raw[ref.center + cfg.state_offset], dtype=np.float32
            )
            action_raw = np.asarray(
                actions_raw[
                    ref.center
                    + cfg.action_offset : ref.center
                    + cfg.action_offset
                    + cfg.policy_horizon
                ],
                dtype=np.float32,
            )
            action = self.action_normalizer.encode(action_raw).astype(np.float32)
            state = self.action_normalizer.encode(state_raw).astype(np.float32)
            boundary = np.concatenate((state[None], action[:-1]), axis=0)
            delta = action - boundary
            motion[index] = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            raw_boundary = np.concatenate((state_raw[None], action_raw[:-1]), axis=0)
            gripper_delta = action_raw[:, grip_index] - raw_boundary[:, grip_index]
            event[index] = bool(np.any(np.abs(gripper_delta) >= float(event_threshold)))
        return motion, event

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        states_raw = episode.states_raw
        actions_raw = episode.actions_raw
        if states_raw is None or actions_raw is None:
            raise ValueError("mainline policy episodes require state and action arrays")
        cfg = self.config
        center = ref.center
        state_index = center + cfg.state_offset
        action_start = center + cfg.action_offset

        state_raw = np.asarray(states_raw[state_index], dtype=np.float32)
        history_state_indices = np.asarray(
            [center + cfg.state_offset + offset for offset in cfg.state_history_offsets],
            dtype=np.int64,
        )
        history_image_indices = np.asarray(
            [center + cfg.image_offset + offset for offset in cfg.raw_pair_offsets],
            dtype=np.int64,
        )
        executed_indices = np.asarray(
            [center + cfg.action_offset + offset for offset in cfg.executed_action_offsets],
            dtype=np.int64,
        )
        future_image_indices = np.asarray(
            [center + cfg.image_offset + offset for offset in cfg.future_offsets],
            dtype=np.int64,
        )
        history_state_raw = np.asarray(states_raw[history_state_indices], dtype=np.float32)
        executed_action_raw = np.asarray(actions_raw[executed_indices], dtype=np.float32)
        future_action_raw = np.asarray(
            actions_raw[action_start : action_start + cfg.world_horizon],
            dtype=np.float32,
        )
        future_state_raw = np.asarray(
            states_raw[state_index + 1 : state_index + cfg.world_horizon + 1],
            dtype=np.float32,
        )

        history_frames = self.image_store.load_window(episode, history_image_indices)
        history_rgb = _camera_stack(history_frames, self.camera_names).float() / 255.0
        current_key = np.asarray(
            (ref.episode_idx, center + cfg.image_offset),
            dtype=np.int64,
        )
        future_keys = np.stack(
            (
                np.full(len(future_image_indices), ref.episode_idx, dtype=np.int64),
                future_image_indices,
            ),
            axis=1,
        )
        return {
            "sample_index": torch.tensor(int(index), dtype=torch.long),
            "episode_idx": torch.tensor(ref.episode_idx, dtype=torch.long),
            "frame_progress": torch.tensor(
                float(center + cfg.image_offset) / float(max(int(episode.length) - 1, 1)),
                dtype=torch.float32,
            ),
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "state_raw": torch.from_numpy(state_raw.copy()),
            "action_state": torch.from_numpy(self.action_normalizer.encode(state_raw)),
            "history_state": torch.from_numpy(self.state_normalizer.encode(history_state_raw)),
            "executed_action_history": torch.from_numpy(
                self.action_normalizer.encode(executed_action_raw)
            ),
            "action": torch.from_numpy(self.action_normalizer.encode(future_action_raw)),
            "policy_action": torch.from_numpy(
                self.action_normalizer.encode(future_action_raw[: cfg.policy_horizon])
            ),
            "policy_action_raw": torch.from_numpy(future_action_raw[: cfg.policy_horizon].copy()),
            "future_state": torch.from_numpy(self.state_normalizer.encode(future_state_raw)),
            "future_offsets": torch.tensor(cfg.future_offsets, dtype=torch.long),
            "history_obs_image": history_rgb,
            "current_key": torch.from_numpy(current_key),
            "future_keys": torch.from_numpy(future_keys),
        }


class CachedTokenPolicyWindowDataset(Dataset):
    """Load current and future DINO mmap rows inside DataLoader workers."""

    def __init__(
        self,
        base: ObservedStateWindowDataset,
        *,
        token_store: DinoV2TokenStore,
    ) -> None:
        self.base = base
        self.token_store = token_store
        if len(base.config.future_offsets) <= 0:
            raise ValueError("future support set cannot be empty")

    def __len__(self) -> int:
        return len(self.base)

    def training_information_signals(
        self,
        *,
        gripper_index: int,
        event_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base.training_information_signals(
            gripper_index=gripper_index,
            event_threshold=event_threshold,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[int(index)]
        current_key = sample.pop("current_key")
        future_keys = sample.pop("future_keys")
        # Current and future keys normally hit the same episode mmap.  One
        # grouped read avoids allocating and dispatching two independent
        # result buffers in every DataLoader sample while preserving the
        # exact current/future ownership split in the returned mapping.
        token_rows = self.token_store.load_batch(torch.cat((current_key[None], future_keys), dim=0))
        sample["current_dinov2_tokens"] = token_rows[0]
        sample["target_future_dinov2_tokens"] = token_rows[1:]
        sample["target_future_offsets"] = sample.pop("future_offsets")
        return sample


__all__ = [
    "CachedTokenPolicyWindowDataset",
    "ObservedStateDatasetConfig",
    "ObservedStateWindowDataset",
    "ObservedWindowRef",
]
