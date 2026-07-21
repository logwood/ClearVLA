from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.action_prior import make_prior_np
from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.experiments.vision_usage_lab.dataset import (
    LabEventScoreConfig,
    LabEventScores,
    LabVisualMode,
    LabWindowRef,
)
from clearvla.experiments.vision_usage_lab.latent_cache import VisionLatentCacheStore
from .codec import RDTLiteCodecs, _valid_centers


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


@dataclass(frozen=True)
class RDTLiteDatasetConfig:
    chunk_len: int = 25
    past_len: int = 25
    state_history_len: int = 1
    obs_horizon: int = 2
    stride: int = 1
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    prior: str = "blend"
    prior_beta: float = 0.5
    velocity_mode: str = "ema"
    ema_decay: float = 0.75
    visual_shift: int = 8

    def validate(self) -> None:
        if (
            min(
                self.chunk_len, self.past_len, self.state_history_len, self.obs_horizon, self.stride
            )
            <= 0
        ):
            raise ValueError("lengths and stride must be positive")
        if self.visual_shift <= 0:
            raise ValueError("visual_shift must be positive")


class RDTLiteDataset(Dataset):
    """Center-aligned direct-action windows for the lightweight RDT reference.

    ``center`` denotes the nominal current action time.  State and image offsets
    are explicit and independently configurable.  The default therefore has a
    clear meaning:

        state[center], images[center] -> actions[center:center+H]

    The previous v13 path indirectly sliced ``past_state`` and then consumed its
    last element, which made the default state token one frame stale.
    """

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        latent_store: VisionLatentCacheStore,
        codecs: RDTLiteCodecs,
        config: RDTLiteDatasetConfig,
        visual_mode: LabVisualMode = LabVisualMode.CORRECT,
        visual_pool_episode_ids: list[int] | None = None,
    ) -> None:
        config.validate()
        codecs.validate()
        if not episode_ids:
            raise ValueError("episode_ids must be non-empty")
        self.episodes = episodes
        self.episode_ids = list(episode_ids)
        self.visual_pool_episode_ids = list(
            self.episode_ids if visual_pool_episode_ids is None else visual_pool_episode_ids
        )
        if not self.visual_pool_episode_ids:
            raise ValueError("visual_pool_episode_ids must be non-empty")
        self.latent_store = latent_store
        self.codecs = codecs
        self.config = config
        self.visual_mode = LabVisualMode(visual_mode)
        self.refs: list[LabWindowRef] = []
        self.event_score: np.ndarray | None = None
        self.event_flag: np.ndarray | None = None

        all_ids = set(self.episode_ids) | set(self.visual_pool_episode_ids)
        if any(index < 0 or index >= len(episodes) for index in all_ids):
            raise IndexError("episode id outside episodes list")
        for index in sorted(all_ids):
            episode = episodes[index]
            if (
                episode.actions_norm is None
                or episode.states_norm is None
                or episode.states_raw is None
            ):
                raise ValueError(f"episode {episode.path} is not normalized for RDT-lite")
            latent_store.validate_episode(episode)
        for episode_idx in self.episode_ids:
            episode = episodes[episode_idx]
            for center in _valid_centers(
                episode,
                chunk_len=config.chunk_len,
                past_len=config.past_len,
                state_history_len=config.state_history_len,
                obs_horizon=config.obs_horizon,
                state_offset=config.state_offset,
                image_offset=config.image_offset,
                action_offset=config.action_offset,
                stride=config.stride,
            ):
                self.refs.append(LabWindowRef(episode_idx=episode_idx, center=int(center)))
        if not self.refs:
            raise ValueError("no valid RDT-lite windows")

    @property
    def chunk_len(self) -> int:
        return self.config.chunk_len

    @property
    def past_len(self) -> int:
        return self.config.past_len

    @property
    def prior(self) -> str:
        return self.config.prior

    @property
    def prior_beta(self) -> float:
        return self.config.prior_beta

    @property
    def velocity_mode(self) -> str:
        return self.config.velocity_mode

    @property
    def ema_decay(self) -> float:
        return self.config.ema_decay

    def __len__(self) -> int:
        return len(self.refs)

    def attach_event_scores(self, scores: LabEventScores) -> None:
        scores.validate(expected_size=len(self.refs))
        self.event_score = scores.normalized_score.copy()
        self.event_flag = scores.is_event.copy()

    def _visual_indices_at(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        episode = self.episodes[episode_idx]
        clipped = int(np.clip(center, self.config.obs_horizon - 1, episode.length - 1))
        indices = np.arange(clipped - self.config.obs_horizon + 1, clipped + 1, dtype=np.int64)
        return int(episode_idx), indices

    def _cross_episode_visual(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        candidates = [index for index in self.visual_pool_episode_ids if index != episode_idx]
        if not candidates:
            raise ValueError("cross-episode visual counterfactual requires at least two episodes")
        target = candidates[(int(center) + int(episode_idx)) % len(candidates)]
        return self._visual_indices_at(target, center)

    def _shifted_visual(self, episode_idx: int, center: int) -> tuple[int, np.ndarray]:
        episode = self.episodes[episode_idx]
        lower = self.config.obs_horizon - 1
        upper = episode.length - 1
        shift = max(int(self.config.visual_shift), self.config.obs_horizon)
        candidates = [center + shift, center - shift, upper, lower]
        candidates = [
            value
            for value in candidates
            if lower <= value <= upper and abs(value - center) >= shift
        ]
        if not candidates:
            # Short synthetic episodes and tiny debugging subsets may not offer
            # the requested minimum shift.  Use the furthest distinct valid
            # frame rather than failing validation entirely.
            candidates = [value for value in (lower, upper) if value != center]
        if not candidates:
            raise ValueError(
                f"episode {episode.path} does not contain a distinct shifted visual frame"
            )
        return self._visual_indices_at(
            episode_idx, max(candidates, key=lambda value: abs(value - center))
        )

    def _resolve_visual(self, episode_idx: int, visual_center: int) -> tuple[int, np.ndarray]:
        if self.visual_mode == LabVisualMode.SAME_EPISODE_SHIFT:
            return self._shifted_visual(episode_idx, visual_center)
        if self.visual_mode == LabVisualMode.CROSS_EPISODE:
            return self._cross_episode_visual(episode_idx, visual_center)
        return self._visual_indices_at(episode_idx, visual_center)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[index]
        episode = self.episodes[ref.episode_idx]
        assert episode.actions_norm is not None
        assert episode.states_norm is not None
        assert episode.states_raw is not None
        cfg = self.config
        action_center = ref.center + cfg.action_offset
        state_center = ref.center + cfg.state_offset
        visual_center = ref.center + cfg.image_offset

        past = np.asarray(
            episode.actions_norm[action_center - cfg.past_len : action_center], dtype=np.float32
        )
        future_abs = np.asarray(
            episode.actions_norm[action_center : action_center + cfg.chunk_len], dtype=np.float32
        )
        future_raw = np.asarray(
            episode.actions_raw[action_center : action_center + cfg.chunk_len], dtype=np.float32
        )
        current_state_raw = np.asarray(episode.states_raw[state_center], dtype=np.float32)
        state_history = np.asarray(
            episode.states_norm[state_center - cfg.state_history_len + 1 : state_center + 1],
            dtype=np.float32,
        )
        target_actions = self.codecs.encode_target(future_raw, current_state_raw)
        prior = make_prior_np(
            past[None],
            cfg.chunk_len,
            prior=cfg.prior,
            prior_beta=cfg.prior_beta,
            velocity_mode=cfg.velocity_mode,
            ema_decay=cfg.ema_decay,
        )[0].astype(np.float32)

        visual_episode_idx, visual_indices = self._resolve_visual(ref.episode_idx, visual_center)
        tokens = self.latent_store.load_tokens(self.episodes[visual_episode_idx], visual_indices)
        if self.visual_mode == LabVisualMode.ZERO:
            tokens = np.zeros_like(tokens)

        sample: dict[str, torch.Tensor] = {
            "past": torch.from_numpy(past.copy()),
            "state_history": torch.from_numpy(state_history.copy()),
            "current_state_raw": torch.from_numpy(current_state_raw.copy()),
            "future": torch.from_numpy(future_abs.copy()),
            "target_actions": torch.from_numpy(target_actions.copy()),
            "prior": torch.from_numpy(prior.copy()),
            "visual_tokens": torch.from_numpy(tokens.copy()),
        }
        if self.event_flag is not None:
            sample["event_flag"] = torch.tensor(float(self.event_flag[index]), dtype=torch.float32)
        return sample


def compute_rdt_lite_event_scores(
    dataset: RDTLiteDataset,
    config: LabEventScoreConfig = LabEventScoreConfig(),
) -> LabEventScores:
    config.validate()
    n = len(dataset.refs)
    prior_residual = np.zeros((n,), dtype=np.float32)
    motion = np.zeros((n,), dtype=np.float32)
    acceleration = np.zeros((n,), dtype=np.float32)
    gripper = np.zeros((n,), dtype=np.float32)
    cfg = dataset.config
    for index, ref in enumerate(dataset.refs):
        episode = dataset.episodes[ref.episode_idx]
        assert episode.actions_norm is not None
        action_center = ref.center + cfg.action_offset
        action = episode.actions_norm
        past = np.asarray(action[action_center - cfg.past_len : action_center], dtype=np.float32)
        future = np.asarray(action[action_center : action_center + cfg.chunk_len], dtype=np.float32)
        prior = make_prior_np(
            past[None],
            cfg.chunk_len,
            prior=cfg.prior,
            prior_beta=cfg.prior_beta,
            velocity_mode=cfg.velocity_mode,
            ema_decay=cfg.ema_decay,
        )[0].astype(np.float32)
        prior_residual[index] = float(_rms(future - prior, axis=(0, 1)))
        if future.shape[0] >= 2:
            velocity = np.diff(np.concatenate([past[-1:], future], axis=0), axis=0)
            motion[index] = float(_rms(velocity, axis=(0, 1)))
        if future.shape[0] >= 3:
            accel = np.diff(np.diff(np.concatenate([past[-1:], future], axis=0), axis=0), axis=0)
            acceleration[index] = float(_rms(accel, axis=(0, 1)))
        gi = (
            config.gripper_index
            if config.gripper_index >= 0
            else future.shape[1] + config.gripper_index
        )
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
