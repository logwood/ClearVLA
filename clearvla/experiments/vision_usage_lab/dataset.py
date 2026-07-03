from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.data.action_prior import make_prior_np
from .latent_cache import VisionLatentCacheStore


class LabVisualMode(str, Enum):
    CORRECT = "correct"
    ZERO = "zero"
    SAME_EPISODE_SHIFT = "same_episode_shift"
    CROSS_EPISODE = "cross_episode"


@dataclass(frozen=True)
class LabWindowRef:
    episode_idx: int
    center: int


@dataclass(frozen=True)
class LabEventScoreConfig:
    prior_residual_weight: float = 1.0
    motion_weight: float = 0.35
    acceleration_weight: float = 0.35
    gripper_weight: float = 0.20
    event_quantile: float = 0.70
    gripper_index: int = -1

    def validate(self) -> None:
        if any(value < 0 for value in (
            self.prior_residual_weight,
            self.motion_weight,
            self.acceleration_weight,
            self.gripper_weight,
        )):
            raise ValueError("event-score weights must be non-negative")
        if not 0.0 < self.event_quantile < 1.0:
            raise ValueError("event_quantile must be in (0,1)")


@dataclass(frozen=True)
class LabEventScores:
    raw_score: np.ndarray
    normalized_score: np.ndarray
    is_event: np.ndarray
    threshold: float
    components: dict[str, np.ndarray]

    def validate(self, expected_size: int | None = None) -> None:
        n = int(self.raw_score.shape[0])
        if expected_size is not None and n != int(expected_size):
            raise ValueError(f"score size={n} != expected={expected_size}")
        if self.normalized_score.shape != (n,) or self.is_event.shape != (n,):
            raise ValueError("event-score arrays must be flat and aligned")
        if not np.isfinite(self.raw_score).all() or not np.isfinite(self.normalized_score).all():
            raise ValueError("event-score arrays must be finite")
        if ((self.normalized_score < 0) | (self.normalized_score > 1)).any():
            raise ValueError("normalized event scores must be in [0,1]")
        for name, values in self.components.items():
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError(f"invalid event component={name!r}")


def _rms(value: np.ndarray, axis: tuple[int, ...] | int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=axis)).astype(np.float32)


def _robust_unit_interval(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    lo = float(np.quantile(value, 0.05))
    hi = float(np.quantile(value, 0.95))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite quantile in event score")
    if hi <= lo + 1e-12:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class VisionUsageLabDataset(Dataset):
    """Action windows paired with frozen patch tokens and future latent targets.

    Counterfactual negatives are explicit: prefer another episode, otherwise use
    a temporally distant frame from the same episode.  Batch-local ``roll`` is
    intentionally avoided because trajectory-block batches contain neighboring
    windows and therefore produce weak negatives.
    """

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        latent_store: VisionLatentCacheStore,
        chunk_len: int,
        past_len: int,
        obs_horizon: int,
        future_visual_horizons: tuple[int, ...] = (1, 4, 8),
        stride: int = 1,
        prior: str = "blend",
        prior_beta: float = 0.5,
        velocity_mode: str = "ema",
        ema_decay: float = 0.75,
        visual_mode: LabVisualMode = LabVisualMode.CORRECT,
        visual_shift: int = 8,
        visual_pool_episode_ids: list[int] | None = None,
        include_negative_visual: bool = False,
        negative_visual_min_shift: int = 8,
        include_future_visual_delta: bool = True,
    ) -> None:
        if min(chunk_len, past_len, obs_horizon, stride) <= 0:
            raise ValueError("chunk_len, past_len, obs_horizon, and stride must be positive")
        if not future_visual_horizons or any(int(x) <= 0 for x in future_visual_horizons):
            raise ValueError("future_visual_horizons must contain positive offsets")
        if tuple(sorted(set(int(x) for x in future_visual_horizons))) != tuple(int(x) for x in future_visual_horizons):
            raise ValueError("future_visual_horizons must be sorted and unique")
        if visual_shift <= 0 or negative_visual_min_shift <= 0:
            raise ValueError("visual shifts must be positive")
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.visual_pool_episode_ids = list(self.episode_ids if visual_pool_episode_ids is None else visual_pool_episode_ids)
        if not self.episode_ids:
            raise ValueError("episode_ids must be non-empty")
        if not self.visual_pool_episode_ids:
            raise ValueError("visual_pool_episode_ids must be non-empty")
        all_ids = set(self.episode_ids) | set(self.visual_pool_episode_ids)
        if any(idx < 0 or idx >= len(episodes) for idx in all_ids):
            raise IndexError("episode id outside episodes list")
        self.latent_store = latent_store
        self.chunk_len = int(chunk_len)
        self.past_len = int(past_len)
        self.obs_horizon = int(obs_horizon)
        self.future_visual_horizons = tuple(int(x) for x in future_visual_horizons)
        self.stride = int(stride)
        self.prior = str(prior)
        self.prior_beta = float(prior_beta)
        self.velocity_mode = str(velocity_mode)
        self.ema_decay = float(ema_decay)
        self.visual_mode = LabVisualMode(visual_mode)
        self.visual_shift = int(visual_shift)
        self.include_negative_visual = bool(include_negative_visual)
        self.negative_visual_min_shift = int(negative_visual_min_shift)
        self.include_future_visual_delta = bool(include_future_visual_delta)
        if self.visual_mode == LabVisualMode.CROSS_EPISODE:
            missing = [idx for idx in self.episode_ids if not any(pool_idx != idx for pool_idx in self.visual_pool_episode_ids)]
            if missing:
                raise ValueError(
                    "cross-episode visual counterfactual requires at least two episodes in visual_pool_episode_ids"
                )
        self.refs: list[LabWindowRef] = []
        self.event_score: np.ndarray | None = None
        self.event_flag: np.ndarray | None = None
        self.demand_target: np.ndarray | None = None

        for episode_idx in sorted(all_ids):
            episode = episodes[episode_idx]
            if episode.actions_norm is None:
                raise ValueError(f"episode {episode.path} has no normalized actions")
            latent_store.validate_episode(episode)

        max_visual_horizon = max(self.future_visual_horizons) if self.include_future_visual_delta else 0
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            start = max(self.past_len, self.obs_horizon - 1)
            stop = min(episode.length - self.chunk_len + 1, episode.length - max_visual_horizon)
            for center in range(start, stop, self.stride):
                self.refs.append(LabWindowRef(episode_idx=episode_idx, center=center))
        if not self.refs:
            raise ValueError("no valid vision-usage lab windows")

    def __len__(self) -> int:
        return len(self.refs)

    def attach_event_scores(self, scores: LabEventScores) -> None:
        """Attach sampling diagnostics and the continuous correction-demand target.

        Demand supervision intentionally uses only the normalized prior residual.
        The broader event score may include velocity, acceleration and gripper
        changes for sampling and reporting, but it must not blur the operational
        meaning of "how much correction does the history prior need?"
        """
        scores.validate(expected_size=len(self.refs))
        self.event_score = scores.normalized_score.copy()
        self.event_flag = scores.is_event.copy()
        self.demand_target = scores.components["prior_residual"].copy()

    def _indices_at(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        episode = self.episodes[episode_idx]
        clipped = int(np.clip(center, self.obs_horizon - 1, episode.length - 1))
        indices = np.arange(clipped - self.obs_horizon + 1, clipped + 1, dtype=np.int64)
        return int(episode_idx), indices

    def _cross_episode_indices(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        candidates = [idx for idx in self.visual_pool_episode_ids if idx != episode_idx]
        if not candidates:
            raise ValueError(
                "cross-episode visual counterfactual requires at least two episodes in visual_pool_episode_ids"
            )
        target_episode_idx = candidates[(int(center) + int(episode_idx)) % len(candidates)]
        return self._indices_at(target_episode_idx, center)

    def _shifted_indices(self, episode_idx: int, center: int, *, min_shift: int) -> tuple[int, np.ndarray]:
        episode = self.episodes[episode_idx]
        lower = self.obs_horizon - 1
        upper = episode.length - 1
        shift = max(int(min_shift), self.obs_horizon)
        candidates = [center + shift, center - shift, upper, lower]
        candidates = [int(x) for x in candidates if lower <= int(x) <= upper and abs(int(x) - center) >= shift]
        if not candidates:
            raise ValueError(
                f"episode {episode.path} is too short for a same-episode negative with min_shift={shift}"
            )
        target_center = max(candidates, key=lambda value: abs(value - center))
        return self._indices_at(episode_idx, target_center)

    def _input_visual_indices(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        if self.visual_mode == LabVisualMode.SAME_EPISODE_SHIFT:
            return self._shifted_indices(episode_idx, center, min_shift=self.visual_shift)
        if self.visual_mode == LabVisualMode.CROSS_EPISODE:
            return self._cross_episode_indices(episode_idx, center)
        return self._indices_at(episode_idx, center)

    def _negative_visual_indices(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        candidates = [idx for idx in self.visual_pool_episode_ids if idx != episode_idx]
        if candidates:
            return self._cross_episode_indices(episode_idx, center)
        return self._shifted_indices(episode_idx, center, min_shift=self.negative_visual_min_shift)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        assert episode.actions_norm is not None
        action = episode.actions_norm
        center = ref.center
        past = np.asarray(action[center - self.past_len:center], dtype=np.float32)
        future = np.asarray(action[center:center + self.chunk_len], dtype=np.float32)
        state_action = episode.states_norm if episode.states_norm is not None else action
        past_state = np.asarray(state_action[center - self.past_len:center], dtype=np.float32)
        prior = make_prior_np(
            past[None],
            self.chunk_len,
            prior=self.prior,
            prior_beta=self.prior_beta,
            velocity_mode=self.velocity_mode,
            ema_decay=self.ema_decay,
        )[0].astype(np.float32)

        input_episode_idx, input_indices = self._input_visual_indices(ref.episode_idx, center)
        input_tokens = self.latent_store.load_tokens(self.episodes[input_episode_idx], input_indices)
        if self.visual_mode == LabVisualMode.ZERO:
            input_tokens = np.zeros_like(input_tokens)

        sample: dict[str, torch.Tensor] = {
            "past": torch.from_numpy(past.copy()),
            "past_state": torch.from_numpy(past_state.copy()),
            "future": torch.from_numpy(future.copy()),
            "prior": torch.from_numpy(prior.copy()),
            "visual_tokens": torch.from_numpy(input_tokens.copy()),
        }
        if self.include_future_visual_delta:
            future_indices = np.asarray([center + h for h in self.future_visual_horizons], dtype=np.int64)
            future_tokens = self.latent_store.load_tokens(episode, future_indices)
            current_tokens = self.latent_store.load_tokens(episode, np.asarray([center], dtype=np.int64))[0]
            future_delta_tokens = future_tokens - current_tokens[None]
            sample["future_visual_delta_tokens"] = torch.from_numpy(future_delta_tokens.astype(np.float32))
        if self.include_negative_visual:
            negative_episode_idx, negative_indices = self._negative_visual_indices(ref.episode_idx, center)
            negative_tokens = self.latent_store.load_tokens(self.episodes[negative_episode_idx], negative_indices)
            sample["negative_visual_tokens"] = torch.from_numpy(negative_tokens.copy())
        if self.event_flag is not None:
            sample["event_flag"] = torch.tensor(float(self.event_flag[index]), dtype=torch.float32)
        if self.demand_target is not None:
            sample["demand_target"] = torch.tensor(float(self.demand_target[index]), dtype=torch.float32)
        return sample


def compute_lab_event_scores(
    dataset: VisionUsageLabDataset,
    config: LabEventScoreConfig = LabEventScoreConfig(),
) -> LabEventScores:
    config.validate()
    n = len(dataset.refs)
    prior_residual = np.zeros((n,), dtype=np.float32)
    motion = np.zeros((n,), dtype=np.float32)
    acceleration = np.zeros((n,), dtype=np.float32)
    gripper = np.zeros((n,), dtype=np.float32)
    for index, ref in enumerate(dataset.refs):
        episode = dataset.episodes[ref.episode_idx]
        assert episode.actions_norm is not None
        action = episode.actions_norm
        center = ref.center
        past = np.asarray(action[center - dataset.past_len:center], dtype=np.float32)
        future = np.asarray(action[center:center + dataset.chunk_len], dtype=np.float32)
        prior = make_prior_np(
            past[None],
            dataset.chunk_len,
            prior=dataset.prior,
            prior_beta=dataset.prior_beta,
            velocity_mode=dataset.velocity_mode,
            ema_decay=dataset.ema_decay,
        )[0].astype(np.float32)
        prior_residual[index] = float(_rms(future - prior, axis=(0, 1)))
        if future.shape[0] >= 2:
            velocity = np.diff(np.concatenate([past[-1:], future], axis=0), axis=0)
            motion[index] = float(_rms(velocity, axis=(0, 1)))
        if future.shape[0] >= 3:
            accel = np.diff(np.diff(np.concatenate([past[-1:], future], axis=0), axis=0), axis=0)
            acceleration[index] = float(_rms(accel, axis=(0, 1)))
        gi = config.gripper_index if config.gripper_index >= 0 else future.shape[1] + config.gripper_index
        if 0 <= gi < future.shape[1]:
            values = np.concatenate([past[-1:, gi], future[:, gi]], axis=0)
            gripper[index] = float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
    components = {
        "prior_residual": _robust_unit_interval(prior_residual),
        "motion": _robust_unit_interval(motion),
        "acceleration": _robust_unit_interval(acceleration),
        "gripper": _robust_unit_interval(gripper),
    }
    raw = (
        config.prior_residual_weight * components["prior_residual"]
        + config.motion_weight * components["motion"]
        + config.acceleration_weight * components["acceleration"]
        + config.gripper_weight * components["gripper"]
    ).astype(np.float32)
    normalized = _robust_unit_interval(raw)
    threshold = float(np.quantile(normalized, config.event_quantile))
    is_event = normalized >= threshold
    if bool(is_event.all()) and len(is_event) > 1:
        is_event[np.argmin(normalized)] = False
    if not bool(is_event.any()) and len(is_event):
        is_event[np.argmax(normalized)] = True
    out = LabEventScores(raw, normalized, is_event.astype(bool), threshold, components)
    out.validate(expected_size=n)
    return out
