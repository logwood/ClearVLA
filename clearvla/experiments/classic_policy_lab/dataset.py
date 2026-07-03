from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.vision.decoded_image_store import DecodedImageStore
from .normalizer import ArrayNormalizer


@dataclass(frozen=True)
class WindowRef:
    episode_idx: int
    center: int


def _camera_stack(frames: dict[str, torch.Tensor], camera_names: tuple[str, ...]) -> torch.Tensor:
    # Store returns [H,C,H,W]. Stack camera axis after observation time.
    return torch.stack([frames[name] for name in camera_names], dim=1)


def _hold_prior(raw_actions: np.ndarray, center: int, horizon: int) -> np.ndarray:
    index = max(0, min(int(center) - 1, len(raw_actions) - 1))
    return np.repeat(raw_actions[index:index + 1], horizon, axis=0).astype(np.float32)


@dataclass(frozen=True)
class ACTDatasetConfig:
    chunk_len: int = 25
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    include_tail_padding: bool = False

    def validate(self) -> None:
        if self.chunk_len <= 0 or self.stride <= 0:
            raise ValueError("chunk_len and stride must be positive")


class ACTWindowDataset(Dataset):
    """ACT observations with explicit timing and optional official-style tail padding."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: ACTDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.image_store = image_store
        self.camera_names = camera_names
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[WindowRef] = []
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(-config.state_offset, -config.image_offset, -config.action_offset, 1 - config.action_offset)
            if config.include_tail_padding:
                high = min(episode.length - 1 - config.state_offset, episode.length - 1 - config.image_offset, episode.length - 1 - config.action_offset)
            else:
                high = min(
                    episode.length - 1 - config.state_offset,
                    episode.length - 1 - config.image_offset,
                    episode.length - config.chunk_len - config.action_offset,
                )
            for center in range(int(low), int(high) + 1, config.stride):
                self.refs.append(WindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("ACT dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_index = ref.center + cfg.state_offset
        image_index = ref.center + cfg.image_offset
        action_start = ref.center + cfg.action_offset
        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        available = max(0, min(cfg.chunk_len, episode.length - action_start))
        action_raw = np.zeros((cfg.chunk_len, episode.actions_raw.shape[1]), dtype=np.float32)
        if available:
            action_raw[:available] = episode.actions_raw[action_start:action_start + available]
        is_pad = np.zeros((cfg.chunk_len,), dtype=bool)
        is_pad[available:] = True
        frames = self.image_store.load_window(episode, np.asarray([image_index], dtype=np.int64))
        image = _camera_stack(frames, self.camera_names)[0].to(torch.float32) / 255.0
        past_start = max(0, action_start - cfg.chunk_len)
        past_raw = np.asarray(episode.actions_raw[past_start:action_start], dtype=np.float32)
        if len(past_raw) < cfg.chunk_len:
            pad = np.repeat(episode.actions_raw[:1], cfg.chunk_len - len(past_raw), axis=0)
            past_raw = np.concatenate([pad, past_raw], axis=0)
        prior_raw = _hold_prior(episode.actions_raw, action_start, cfg.chunk_len)
        return {
            "qpos": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "image": image,
            "actions": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "is_pad": torch.from_numpy(is_pad),
            "target_raw": torch.from_numpy(action_raw),
            "past_raw": torch.from_numpy(past_raw),
            "prior_raw": torch.from_numpy(prior_raw),
        }


@dataclass(frozen=True)
class DPDatasetConfig:
    prediction_horizon: int = 16
    obs_horizon: int = 2
    action_steps: int = 8
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1

    def validate(self) -> None:
        if min(self.prediction_horizon, self.obs_horizon, self.action_steps, self.stride) <= 0:
            raise ValueError("horizons and stride must be positive")
        if self.obs_horizon > self.prediction_horizon:
            raise ValueError("obs_horizon cannot exceed prediction_horizon")
        if self.obs_horizon - 1 + self.action_steps > self.prediction_horizon:
            raise ValueError("execution slice exceeds prediction horizon")


class DPWindowDataset(Dataset):
    """Diffusion Policy windows matching the official observation/action alignment.

    The target trajectory starts at the first observation time. The executable
    action slice starts at ``obs_horizon - 1``, exactly as in the official image
    policy's ``predict_action`` method.
    """

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: DPDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.image_store = image_store
        self.camera_names = camera_names
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[WindowRef] = []
        for episode_idx in episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(config.obs_horizon - 1 - config.state_offset, config.obs_horizon - 1 - config.image_offset, config.obs_horizon - 1 - config.action_offset)
            high = min(
                episode.length - 1 - config.state_offset,
                episode.length - 1 - config.image_offset,
                episode.length - config.prediction_horizon + config.obs_horizon - 1 - config.action_offset,
            )
            for center in range(int(low), int(high) + 1, config.stride):
                self.refs.append(WindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("DP dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        obs_start_state = ref.center + cfg.state_offset - cfg.obs_horizon + 1
        obs_start_image = ref.center + cfg.image_offset - cfg.obs_horizon + 1
        trajectory_start = ref.center + cfg.action_offset - cfg.obs_horizon + 1
        state_raw = np.asarray(episode.states_raw[obs_start_state:obs_start_state + cfg.obs_horizon], dtype=np.float32)
        action_raw = np.asarray(episode.actions_raw[trajectory_start:trajectory_start + cfg.prediction_horizon], dtype=np.float32)
        image_indices = np.arange(obs_start_image, obs_start_image + cfg.obs_horizon, dtype=np.int64)
        frames = self.image_store.load_window(episode, image_indices)
        images = _camera_stack(frames, self.camera_names).to(torch.float32) / 255.0
        exec_start = cfg.obs_horizon - 1
        exec_raw = action_raw[exec_start:exec_start + cfg.action_steps]
        past_start = max(0, ref.center + cfg.action_offset - cfg.action_steps)
        past_raw = np.asarray(episode.actions_raw[past_start:ref.center + cfg.action_offset], dtype=np.float32)
        if len(past_raw) < cfg.action_steps:
            pad = np.repeat(episode.actions_raw[:1], cfg.action_steps - len(past_raw), axis=0)
            past_raw = np.concatenate([pad, past_raw], axis=0)
        prior_raw = _hold_prior(episode.actions_raw, ref.center + cfg.action_offset, cfg.action_steps)
        return {
            "obs_state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "obs_image": images,
            "action": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "target_raw": torch.from_numpy(exec_raw.copy()),
            "past_raw": torch.from_numpy(past_raw.copy()),
            "prior_raw": torch.from_numpy(prior_raw.copy()),
        }


@dataclass(frozen=True)
class RDTSmallDatasetConfig:
    """RDT-170M windows adapted to ClearVLA episodes.

    RDT consumes the latest proprioceptive state and a short image history, then
    predicts a future action chunk beginning at the current control step.
    """

    prediction_horizon: int = 64
    image_history: int = 2
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    control_frequency: float = 25.0

    def validate(self) -> None:
        if min(self.prediction_horizon, self.image_history, self.stride) <= 0:
            raise ValueError("prediction_horizon, image_history, and stride must be positive")
        if self.control_frequency < 0:
            raise ValueError("control_frequency must be non-negative")


class RDTSmallWindowDataset(Dataset):
    """RDT-small windows with two-frame image history and a 64-step target."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: RDTSmallDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.image_store = image_store
        self.camera_names = camera_names
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[WindowRef] = []
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(
                -config.state_offset,
                config.image_history - 1 - config.image_offset,
                1 - config.action_offset,
            )
            high = min(
                episode.length - 1 - config.state_offset,
                episode.length - 1 - config.image_offset,
                episode.length - config.prediction_horizon - config.action_offset,
            )
            for center in range(int(low), int(high) + 1, config.stride):
                self.refs.append(WindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("RDT-small dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_index = ref.center + cfg.state_offset
        image_end = ref.center + cfg.image_offset
        image_start = image_end - cfg.image_history + 1
        action_start = ref.center + cfg.action_offset
        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.prediction_horizon],
            dtype=np.float32,
        )
        image_indices = np.arange(image_start, image_end + 1, dtype=np.int64)
        frames = self.image_store.load_window(episode, image_indices)
        images = _camera_stack(frames, self.camera_names).to(torch.float32) / 255.0
        past_start = max(0, action_start - cfg.prediction_horizon)
        past_raw = np.asarray(episode.actions_raw[past_start:action_start], dtype=np.float32)
        if len(past_raw) < cfg.prediction_horizon:
            pad = np.repeat(episode.actions_raw[:1], cfg.prediction_horizon - len(past_raw), axis=0)
            past_raw = np.concatenate([pad, past_raw], axis=0)
        prior_raw = _hold_prior(episode.actions_raw, action_start, cfg.prediction_horizon)
        return {
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "obs_image": images,
            "action": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "target_raw": torch.from_numpy(action_raw.copy()),
            "past_raw": torch.from_numpy(past_raw.copy()),
            "prior_raw": torch.from_numpy(prior_raw.copy()),
            "ctrl_freq": torch.tensor(cfg.control_frequency, dtype=torch.float32),
        }


@dataclass(frozen=True)
class RDT2FMDatasetConfig:
    """RDT2-FM windows for a pluggable visual or vision-language condition source.

    The released RDT2-FM expert predicts a 24-step relative action chunk from
    one synchronized multi-camera observation.  ClearVLA keeps the source action
    representation explicit: local joint-space experiments can use the recorded
    dimensions directly; official UMI-20 checkpoint loading requires an external
    dataset conversion into the released 20-dimensional relative TCP format.
    """

    prediction_horizon: int = 24
    image_history: int = 1
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    zero_state: bool = False
    future_latent_offsets: tuple[int, ...] = ()
    return_future_images: bool = False

    def validate(self) -> None:
        if min(self.prediction_horizon, self.image_history, self.stride) <= 0:
            raise ValueError("prediction_horizon, image_history, and stride must be positive")
        if self.image_history != 1:
            raise ValueError("released RDT2-FM uses one image timestep; use image_history=1")
        if self.future_latent_offsets:
            if any(int(offset) <= 0 for offset in self.future_latent_offsets):
                raise ValueError("future_latent_offsets must contain positive integers")
            if tuple(sorted(set(self.future_latent_offsets))) != tuple(self.future_latent_offsets):
                raise ValueError("future_latent_offsets must be strictly increasing and unique")
            if max(self.future_latent_offsets) > self.prediction_horizon:
                raise ValueError("future_latent_offsets cannot exceed prediction_horizon")


class RDT2FMWindowDataset(Dataset):
    """Single-observation, 24-step flow-matching windows for RDT2-FM."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: RDT2FMDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.image_store = image_store
        self.camera_names = camera_names
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.config = config
        self.refs: list[WindowRef] = []
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            image_store.validate_episode(episode)
            low = max(-config.state_offset, -config.image_offset, 1 - config.action_offset)
            future_image_limit = episode.length - 1 - config.image_offset
            if config.future_latent_offsets:
                future_image_limit -= max(config.future_latent_offsets)
            high = min(
                episode.length - 1 - config.state_offset,
                future_image_limit,
                episode.length - config.prediction_horizon - config.action_offset,
            )
            for center in range(int(low), int(high) + 1, config.stride):
                self.refs.append(WindowRef(episode_idx, center))
        if not self.refs:
            raise ValueError("RDT2-FM dataset has no valid windows")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        cfg = self.config
        state_index = ref.center + cfg.state_offset
        image_index = ref.center + cfg.image_offset
        action_start = ref.center + cfg.action_offset
        state_raw = np.asarray(episode.states_raw[state_index], dtype=np.float32)
        if cfg.zero_state:
            state_raw = np.zeros_like(state_raw)
        action_raw = np.asarray(
            episode.actions_raw[action_start : action_start + cfg.prediction_horizon],
            dtype=np.float32,
        )
        frames = self.image_store.load_window(episode, np.asarray([image_index], dtype=np.int64))
        images = _camera_stack(frames, self.camera_names)[0].to(torch.float32) / 255.0
        past_start = max(0, action_start - cfg.prediction_horizon)
        past_raw = np.asarray(episode.actions_raw[past_start:action_start], dtype=np.float32)
        if len(past_raw) < cfg.prediction_horizon:
            pad = np.repeat(episode.actions_raw[:1], cfg.prediction_horizon - len(past_raw), axis=0)
            past_raw = np.concatenate([pad, past_raw], axis=0)
        prior_raw = _hold_prior(episode.actions_raw, action_start, cfg.prediction_horizon)
        future_indices = np.asarray(
            [image_index + int(offset) for offset in cfg.future_latent_offsets],
            dtype=np.int64,
        )
        sample = {
            "state": torch.from_numpy(self.state_normalizer.encode(state_raw)),
            "obs_image": images,
            "action": torch.from_numpy(self.action_normalizer.encode(action_raw)),
            "target_raw": torch.from_numpy(action_raw.copy()),
            "past_raw": torch.from_numpy(past_raw.copy()),
            "prior_raw": torch.from_numpy(prior_raw.copy()),
            # Normalized history inputs for progressive history-anchored policies.
            # Existing v18 reference callers continue to use the raw keys above.
            "past": torch.from_numpy(self.action_normalizer.encode(past_raw)),
            "prior": torch.from_numpy(self.action_normalizer.encode(prior_raw)),
            # Stable cache and ablation lookup key: [global episode index, frame index].
            "episode_idx": torch.tensor(ref.episode_idx, dtype=torch.long),
            "image_index": torch.tensor(image_index, dtype=torch.long),
            "future_image_indices": torch.from_numpy(future_indices),
        }
        if cfg.return_future_images and len(future_indices):
            future_frames = self.image_store.load_window(episode, future_indices)
            sample["future_obs_image"] = (
                _camera_stack(future_frames, self.camera_names).to(torch.float32) / 255.0
            )
        return sample

    def load_images_for_keys(self, sample_keys: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Load observations for explicit ``[episode_idx, image_index]`` pairs."""
        keys = np.asarray(sample_keys, dtype=np.int64)
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(f"sample_keys must be [B,2], got {keys.shape}")
        rows = []
        for episode_idx, image_index in keys.tolist():
            if int(episode_idx) < 0 or int(episode_idx) >= len(self.episodes):
                raise IndexError(f"episode_idx={episode_idx} outside available episodes")
            episode = self.episodes[int(episode_idx)]
            if int(image_index) < 0 or int(image_index) >= episode.length:
                raise IndexError(f"image_index={image_index} outside episode length={episode.length}")
            frames = self.image_store.load_window(episode, np.asarray([int(image_index)], dtype=np.int64))
            rows.append(_camera_stack(frames, self.camera_names)[0].to(torch.float32) / 255.0)
        return torch.stack(rows, dim=0)

    def cross_episode_keys(self, sample_keys: torch.Tensor | np.ndarray, *, seed: int = 0) -> torch.Tensor:
        """Select deterministic observations from another validation episode.

        This preserves approximate temporal position while breaking the visual
        correspondence to the action target.  It is stronger than batch shuffling
        and still works with evaluation batch size one.
        """
        keys = np.asarray(sample_keys, dtype=np.int64)
        if keys.ndim != 2 or keys.shape[1] != 2:
            raise ValueError(f"sample_keys must be [B,2], got {keys.shape}")
        # Use the full loaded episode set rather than only the evaluated split.
        # A validation split may contain one episode; replacing its image with an
        # image from another episode is still a valid counterfactual ablation.
        pool = list(range(len(self.episodes)))
        if len(pool) < 2:
            raise ValueError("shuffle-episode ablation requires at least two loaded episodes")
        position = {int(episode_idx): idx for idx, episode_idx in enumerate(pool)}
        rows = []
        shift = 1 + (int(seed) % (len(pool) - 1))
        for row, (episode_idx, image_index) in enumerate(keys.tolist()):
            if int(episode_idx) not in position:
                raise ValueError(f"episode_idx={episode_idx} is not a loaded episode")
            source_pos = position[int(episode_idx)]
            target_idx = int(pool[(source_pos + shift + row) % len(pool)])
            if target_idx == int(episode_idx):
                target_idx = int(pool[(source_pos + shift + row + 1) % len(pool)])
            target_episode = self.episodes[target_idx]
            # Retain normalized phase instead of silently clipping late frames.
            source_episode = self.episodes[int(episode_idx)]
            phase = 0.0 if source_episode.length <= 1 else float(image_index) / float(source_episode.length - 1)
            target_frame = int(round(phase * max(target_episode.length - 1, 0)))
            rows.append([target_idx, target_frame])
        return torch.tensor(rows, dtype=torch.long)
