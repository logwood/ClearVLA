from __future__ import annotations

"""V35 observed-state datasets.

The current world observation contains image/state history and *executed* action
history only. Candidate future actions are returned separately and never enter
perception. Dense future targets are aligned to the recurrent segment grid.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.vision.decoded_image_store import DecodedImageStore


@dataclass(frozen=True)
class ObservedStateDatasetConfig:
    world_horizon: int = 48
    policy_horizon: int = 24
    segment_length: int = 4
    history_offsets: tuple[int, ...] = (-8, -4, 0)
    executed_action_offsets: tuple[int, ...] = (-8, -4, -1)
    target_history_offsets: tuple[int, ...] = (-8, -4, 0)
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    return_images: bool = True

    def validate(self) -> None:
        if min(self.world_horizon, self.policy_horizon, self.segment_length, self.stride) <= 0:
            raise ValueError("horizons, segment_length and stride must be positive")
        if self.world_horizon % self.segment_length:
            raise ValueError("world_horizon must be divisible by segment_length")
        if self.policy_horizon > self.world_horizon:
            raise ValueError("policy_horizon cannot exceed world_horizon")
        if not self.history_offsets or self.history_offsets[-1] != 0:
            raise ValueError("history_offsets must be increasing and end at 0")
        if tuple(sorted(set(self.history_offsets))) != self.history_offsets:
            raise ValueError("history_offsets must be strictly increasing")
        if not self.executed_action_offsets or max(self.executed_action_offsets) >= 0:
            raise ValueError("executed_action_offsets must contain only past actions")
        if tuple(sorted(set(self.executed_action_offsets))) != self.executed_action_offsets:
            raise ValueError("executed_action_offsets must be strictly increasing")
        if not self.target_history_offsets or self.target_history_offsets[-1] != 0:
            raise ValueError("target_history_offsets must end at 0")
        if tuple(sorted(set(self.target_history_offsets))) != self.target_history_offsets:
            raise ValueError("target_history_offsets must be strictly increasing")
        if len(self.target_history_offsets) != len(self.history_offsets):
            raise ValueError("target_history_offsets must match history length")

    @property
    def future_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.segment_length, self.world_horizon + 1, self.segment_length))


@dataclass(frozen=True)
class ObservedWindowRef:
    episode_idx: int
    center: int


def _camera_stack(frames: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.stack([frames[name] for name in names], dim=1)


class ObservedStateWindowDataset(Dataset):
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
        self.episode_ids = [int(x) for x in episode_ids]
        self.image_store = image_store
        self.camera_names = tuple(camera_names)
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[ObservedWindowRef] = []

        min_rel = min(
            min(config.history_offsets) + config.image_offset,
            min(config.history_offsets) + config.state_offset,
            min(config.executed_action_offsets) + config.action_offset,
            min(config.future_offsets) + min(config.target_history_offsets) + config.image_offset,
            min(config.future_offsets) + min(config.target_history_offsets) + config.state_offset,
        )
        max_rel = max(
            config.world_horizon + config.action_offset - 1,
            config.world_horizon + config.state_offset,
            max(config.future_offsets) + config.image_offset,
            max(config.future_offsets) + config.state_offset,
        )
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(0, -int(min_rel))
            high = int(episode.length - 1 - max_rel)
            for center in range(low, high + 1, config.stride):
                self.refs.append(ObservedWindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("V35 dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def _history_indices(self, center: int) -> np.ndarray:
        return np.asarray(
            [center + self.config.image_offset + int(x) for x in self.config.history_offsets],
            dtype=np.int64,
        )

    def _target_indices(self, center: int) -> np.ndarray:
        return np.asarray(
            [
                [
                    center + self.config.image_offset + int(future) + int(history)
                    for history in self.config.target_history_offsets
                ]
                for future in self.config.future_offsets
            ],
            dtype=np.int64,
        )

    def descriptor_metadata(self, index: int) -> dict[str, np.ndarray | int]:
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_raw = np.asarray(episode.states_raw[ref.center + cfg.state_offset], dtype=np.float32)
        action_raw = np.asarray(
            episode.actions_raw[
                ref.center + cfg.action_offset : ref.center + cfg.action_offset + cfg.world_horizon
            ],
            dtype=np.float32,
        )
        future_state = np.asarray(
            episode.states_raw[
                ref.center + cfg.state_offset + 1 : ref.center + cfg.state_offset + cfg.world_horizon + 1
            ],
            dtype=np.float32,
        )
        boundary = np.concatenate([state_raw[None], action_raw[:-1]], axis=0)
        velocity = action_raw - boundary
        sampled = action_raw[np.asarray(cfg.future_offsets) - 1]
        future_sampled = future_state[np.asarray(cfg.future_offsets) - 1]
        return {
            "episode_idx": int(ref.episode_idx),
            "center": int(ref.center),
            "state": self.state_normalizer.encode(state_raw).astype(np.float32),
            "state_raw": state_raw,
            "action_summary": np.concatenate(
                [sampled.reshape(-1), velocity.mean(0), velocity.std(0), action_raw[-1] - state_raw]
            ).astype(np.float32),
            "future_summary": np.concatenate(
                [future_sampled.reshape(-1), (future_sampled - state_raw).reshape(-1)]
            ).astype(np.float32),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        center = ref.center
        state_index = center + cfg.state_offset
        action_start = center + cfg.action_offset

        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        history_state_indices = np.asarray(
            [center + cfg.state_offset + int(x) for x in cfg.history_offsets], dtype=np.int64
        )
        executed_indices = np.asarray(
            [center + cfg.action_offset + int(x) for x in cfg.executed_action_offsets], dtype=np.int64
        )
        history_state_raw = np.asarray(episode.states_raw[history_state_indices], dtype=np.float32)
        executed_action_raw = np.asarray(episode.actions_raw[executed_indices], dtype=np.float32)
        future_action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.world_horizon], dtype=np.float32
        )
        future_state_raw = np.asarray(
            episode.states_raw[state_index + 1 : state_index + cfg.world_horizon + 1],
            dtype=np.float32,
        )
        target_state_indices = np.asarray(
            [
                [center + cfg.state_offset + int(f) + int(h) for h in cfg.target_history_offsets]
                for f in cfg.future_offsets
            ],
            dtype=np.int64,
        )
        target_history_state_raw = np.asarray(
            episode.states_raw[target_state_indices], dtype=np.float32
        )
        target_executed_indices = np.asarray(
            [
                [center + cfg.action_offset + int(f) + int(h) for h in cfg.executed_action_offsets]
                for f in cfg.future_offsets
            ],
            dtype=np.int64,
        )
        target_executed_action_raw = np.asarray(
            episode.actions_raw[target_executed_indices], dtype=np.float32
        )
        history_indices = self._history_indices(center)
        target_indices = self._target_indices(center)
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
            "center": torch.tensor(center, dtype=torch.long),
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "state_raw": torch.from_numpy(state_raw.copy()),
            "action_state": torch.from_numpy(self.action_normalizer.encode(state_raw)),
            "history_state": torch.from_numpy(self.state_normalizer.encode(history_state_raw)),
            "history_state_raw": torch.from_numpy(history_state_raw.copy()),
            "executed_action_history": torch.from_numpy(
                self.action_normalizer.encode(executed_action_raw)
            ),
            "executed_action_history_raw": torch.from_numpy(executed_action_raw.copy()),
            "action": torch.from_numpy(self.action_normalizer.encode(future_action_raw)),
            "action_raw": torch.from_numpy(future_action_raw.copy()),
            "policy_action": torch.from_numpy(
                self.action_normalizer.encode(future_action_raw[: cfg.policy_horizon])
            ),
            "policy_action_raw": torch.from_numpy(future_action_raw[: cfg.policy_horizon].copy()),
            "future_state": torch.from_numpy(self.state_normalizer.encode(future_state_raw)),
            "future_state_raw": torch.from_numpy(future_state_raw.copy()),
            "segment_state": torch.from_numpy(
                self.state_normalizer.encode(future_state_raw[np.asarray(cfg.future_offsets) - 1])
            ),
            "segment_state_raw": torch.from_numpy(
                future_state_raw[np.asarray(cfg.future_offsets) - 1].copy()
            ),
            "target_history_state": torch.from_numpy(
                self.state_normalizer.encode(target_history_state_raw)
            ),
            "target_history_state_raw": torch.from_numpy(target_history_state_raw.copy()),
            "target_executed_action_history": torch.from_numpy(
                self.action_normalizer.encode(target_executed_action_raw)
            ),
            "target_executed_action_history_raw": torch.from_numpy(
                target_executed_action_raw.copy()
            ),
            "history_keys": torch.from_numpy(history_keys),
            "target_history_keys": torch.from_numpy(target_keys),
            "future_offsets": torch.tensor(cfg.future_offsets, dtype=torch.long),
        }
        if cfg.return_images:
            current_frames = self.image_store.load_window(episode, history_indices)
            current = _camera_stack(current_frames, self.camera_names).float() / 255.0
            flat_target = target_indices.reshape(-1)
            target_frames = self.image_store.load_window(episode, flat_target)
            target = _camera_stack(target_frames, self.camera_names).float() / 255.0
            sample["history_obs_image"] = current
            sample["target_history_obs_image"] = target.reshape(
                len(cfg.future_offsets), len(cfg.target_history_offsets), *target.shape[1:]
            )
        return sample




class CurrentEvidenceViewDataset(Dataset):
    """Cheap current-only view for pair/support indexing."""

    def __init__(self, base: ObservedStateWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[int(index)]
        keep = {
            "sample_index", "episode_idx", "state", "state_raw", "action_state", "history_state",
            "executed_action_history", "action", "action_raw", "future_state", "future_state_raw",
            "segment_state", "segment_state_raw", "history_keys", "history_obs_image",
        }
        return {key: value for key, value in sample.items() if key in keep}


class PolicyWindowDataset(Dataset):
    def __init__(self, base: ObservedStateWindowDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[int(index)]
        keys = {
            "sample_index", "episode_idx", "center", "state", "state_raw", "action_state",
            "history_state", "executed_action_history", "executed_action_history_raw",
            "policy_action", "policy_action_raw", "history_keys", "history_obs_image",
            "target_history_keys", "target_history_obs_image", "target_history_state", "target_history_state_raw",
        }
        return {key: value for key, value in sample.items() if key in keys}


class CachedTokenPolicyWindowDataset(PolicyWindowDataset):
    """Policy dataset that loads DINO cache tokens inside DataLoader workers.

    The normal policy dataset returns episode/frame keys and the training loop
    reads token mmap files synchronously in the main process.  That makes
    ``num_workers`` mostly useless for cached-DINO training.  This wrapper moves
    current and compact future-token loading into ``__getitem__`` so worker
    prefetch and pinned-memory transfer can hide most token I/O.

    Only the future anchors actually used by V38 are loaded, and only the last
    target-history frame for each future offset is needed by the residual
    future-flow objective.
    """

    def __init__(self, base: ObservedStateWindowDataset, *, token_store, future_anchors: int) -> None:
        super().__init__(base)
        self.token_store = token_store
        self.future_anchors = int(future_anchors)
        if self.future_anchors <= 0:
            raise ValueError("future_anchors must be positive")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        if "history_obs_image" in sample:
            return sample
        history_keys = sample["history_keys"]
        sample["history_dinov2_tokens"] = self.token_store.load_batch(history_keys).reshape(
            history_keys.shape[0],
            len(self.base.camera_names),
            self.token_store.tokens_per_camera,
            self.token_store.token_dim,
        )
        if "target_history_keys" in sample:
            target_keys = sample["target_history_keys"]
            anchors = min(self.future_anchors, int(target_keys.shape[0]))
            compact_keys = target_keys[:anchors, -1, :]
            sample["target_future_dinov2_tokens"] = self.token_store.load_batch(compact_keys).reshape(
                anchors,
                len(self.base.camera_names),
                self.token_store.tokens_per_camera,
                self.token_store.token_dim,
            )
        return sample


__all__ = [
    "ObservedStateDatasetConfig",
    "ObservedWindowRef",
    "ObservedStateWindowDataset",
    "CurrentEvidenceViewDataset",
    "PolicyWindowDataset",
    "CachedTokenPolicyWindowDataset",
]
