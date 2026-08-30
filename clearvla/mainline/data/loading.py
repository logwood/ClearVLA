"""One explicit data path from cached episodes to typed mainline batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.data.hdf5_episode import LoadedEpisode, load_episodes
from clearvla.data.samplers import (
    InformationBalancedBatchSampler,
    InformationBalancedSamplerConfig,
)
from clearvla.data.split import resolve_episode_ids
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.preprocessing import PreprocessConfig

from ..config import ExperimentConfig
from ..interfaces import (
    ActionSupervision,
    AuditMetadata,
    CurrentObservation,
    FutureSupervision,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
    TrainingBatch,
)
from .dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from .language import load_t5_condition_bank
from .normalizer import ArrayNormalizer
from .token_store import DinoV2TokenStore


@dataclass(frozen=True)
class GoalTemplate:
    tokens: Tensor  # CPU float32 [N,L,D]
    mask: Tensor  # CPU bool [N,L]
    metadata: dict[str, object]
    episode_condition_indices: Tensor | None = None  # CPU long [episodes]


@dataclass(frozen=True)
class MainlineDataBundle:
    episodes: tuple[LoadedEpisode, ...]
    splits: dict[str, tuple[int, ...]]
    datasets: dict[str, CachedTokenPolicyWindowDataset]
    action_normalizer: ArrayNormalizer
    state_normalizer: ArrayNormalizer
    goal: GoalTemplate
    skipped: tuple[tuple[str, str], ...]
    sampling_seed: int
    information_uniform_fraction: float
    information_event_fraction: float
    information_motion_quantile: float
    gripper_event_threshold: float

    def loader(
        self,
        split: str,
        *,
        batch_size: int,
        workers: int,
        device: torch.device,
        shuffle: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"unknown data split {split!r}")
        if batch_size <= 0 or workers < 0:
            raise ValueError("batch size must be positive and workers non-negative")
        do_shuffle = split == "train" if shuffle is None else bool(shuffle)
        common = {
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": workers > 0,
            "generator": generator,
        }
        dataset = self.datasets[split]
        if split == "train" and do_shuffle:
            if not isinstance(dataset, CachedTokenPolicyWindowDataset):
                raise TypeError("formal training requires the cached-token window dataset")
            motion_score, is_event = dataset.training_information_signals(
                gripper_index=-1,
                event_threshold=self.gripper_event_threshold,
            )
            sampler = InformationBalancedBatchSampler(
                motion_score,
                is_event,
                InformationBalancedSamplerConfig(
                    batch_size=batch_size,
                    uniform_fraction=self.information_uniform_fraction,
                    event_fraction=self.information_event_fraction,
                    motion_quantile=self.information_motion_quantile,
                    seed=self.sampling_seed,
                ),
            )
            return DataLoader(dataset, batch_sampler=sampler, **common)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=do_shuffle,
            **common,
        )


def _normalizers(
    episodes: list[LoadedEpisode],
    train_ids: list[int],
    *,
    mode: str,
) -> tuple[ArrayNormalizer, ArrayNormalizer]:
    if mode != "zscore":
        raise ValueError("mainline only accepts the established z-score chart")
    action_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    for index in train_ids:
        action = episodes[index].actions_raw
        state = episodes[index].states_raw
        if action is None or state is None:
            raise ValueError("mainline normalization requires state and action arrays")
        action_rows.append(action)
        state_rows.append(state)
    actions = ArrayNormalizer.fit_zscore(action_rows)
    states = ArrayNormalizer.fit_zscore(state_rows)
    return actions, states


def load_mainline_data(
    config: ExperimentConfig,
    *,
    allow_null_goal: bool = False,
) -> MainlineDataBundle:
    """Build formal train/val/test datasets without importing a legacy lab."""

    config.validate()
    data = config.data
    dims = config.dimensions
    obs = config.observation
    cameras = tuple(data.camera_names)
    dataset_config = ObservedStateDatasetConfig(
        world_horizon=48,
        policy_horizon=dims.action_horizon,
        support_stride=4,
        state_history_offsets=(-8, -4, 0),
        visual_history_offsets=(-2 * obs.flow_reference_frames, -obs.flow_reference_frames, 0),
        executed_action_offsets=(-24, -16, -12, -8, -6, -4, -2, -1),
        stride=data.stride,
    )
    dataset_config.validate()
    min_length = 48 + 8 + 2
    episodes, skipped = load_episodes(
        Path(data.raw_hdf5_root),
        data.hdf5_glob,
        cameras=cameras,
        min_length=min_length,
        action_key=data.action_key,
        state_key=data.state_key,
        camera_key_overrides={
            "top": data.top_camera_key,
            "wrist": data.wrist_camera_key,
        },
    )
    train_ids, val_ids, test_ids = resolve_episode_ids(
        len(episodes),
        mode=data.split_mode,
        train_frac=0.8,
        val_frac=0.1,
        seed=data.seed,
        train_episode_count=data.train_episodes,
        val_episode_count=data.val_episodes,
        test_episode_count=data.test_episodes,
        episode_names=[episode.episode_id for episode in episodes],
    )
    action_normalizer, state_normalizer = _normalizers(
        episodes,
        train_ids,
        mode=data.normalizer,
    )
    preprocessing = PreprocessConfig(resize_hw=(data.cache_side, data.cache_side), crop_hw=None)
    image_store = DecodedImageStore(
        Path(data.decoded_cache),
        camera_names=cameras,
        preprocessing=preprocessing,
    )
    token_store = DinoV2TokenStore(
        Path(data.dino_cache),
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing,
        dinov2_model=data.dinov2_model,
    )
    if token_store.token_dim != dims.visual_token_dim:
        raise ValueError(
            f"DINO cache width {token_store.token_dim} != model width {dims.visual_token_dim}"
        )
    if token_store.tokens_per_camera != dims.patches_per_camera:
        raise ValueError(
            "DINO cache has "
            f"{token_store.tokens_per_camera} patches/camera, but the configured "
            f"native chart expects {dims.patches_per_camera}"
        )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    datasets: dict[str, CachedTokenPolicyWindowDataset] = {}
    for name, ids in split_ids.items():
        base = ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=dataset_config,
        )
        datasets[name] = CachedTokenPolicyWindowDataset(
            base,
            token_store=token_store,
        )
    goal_bank = load_t5_condition_bank(
        data.t5_condition,
        max_tokens=dims.goal_max_tokens,
        expected_width=dims.goal_token_dim,
        allow_null=allow_null_goal,
    )
    episode_condition_indices = goal_bank.condition_indices(
        [episode.instruction for episode in episodes]
    )
    return MainlineDataBundle(
        episodes=tuple(episodes),
        splits={name: tuple(ids) for name, ids in split_ids.items()},
        datasets=datasets,
        action_normalizer=action_normalizer,
        state_normalizer=state_normalizer,
        goal=GoalTemplate(
            goal_bank.tokens,
            goal_bank.mask,
            goal_bank.metadata,
            episode_condition_indices,
        ),
        skipped=tuple((str(path), str(reason)) for path, reason in skipped),
        sampling_seed=data.seed,
        information_uniform_fraction=data.information_uniform_fraction,
        information_event_fraction=data.information_event_fraction,
        information_motion_quantile=data.information_motion_quantile,
        gripper_event_threshold=config.objectives.gripper_event_threshold,
    )


def _device_tensor(
    batch: Mapping[str, Tensor],
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if name not in batch:
        raise KeyError(f"dataset batch is missing required field {name!r}")
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"dataset field {name!r} is not a tensor")
    return value.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")


def _audit_tensor(
    batch: Mapping[str, Tensor],
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Keep audit-only dataset metadata off the accelerator hot path."""

    if name not in batch:
        raise KeyError(f"dataset batch is missing required field {name!r}")
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"dataset field {name!r} is not a tensor")
    return value.detach().to(device="cpu", dtype=dtype)


def to_training_batch(
    batch: Mapping[str, Tensor],
    *,
    goal: GoalTemplate,
    config: ExperimentConfig,
    device: torch.device,
) -> TrainingBatch:
    """Convert a worker batch to the three disjoint model/training planes."""

    dino_history = _device_tensor(batch, "history_dinov2_tokens", device=device)
    batch_size = int(dino_history.shape[0])
    if goal.episode_condition_indices is None:
        if int(goal.tokens.shape[0]) != 1:
            raise ValueError("multi-row goal template requires episode condition indices")
        condition_indices = torch.zeros(batch_size, dtype=torch.long)
    else:
        episode_indices = _audit_tensor(batch, "episode_idx").to(dtype=torch.long)
        if tuple(episode_indices.shape) != (batch_size,):
            raise ValueError("episode_idx must be one scalar per batch row")
        if episode_indices.numel() and (
            int(episode_indices.min()) < 0
            or int(episode_indices.max()) >= len(goal.episode_condition_indices)
        ):
            raise IndexError("episode_idx is outside the goal-condition mapping")
        condition_indices = goal.episode_condition_indices.index_select(0, episode_indices)
    goal_tokens = goal.tokens.index_select(0, condition_indices).to(
        device=device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )
    goal_mask = goal.mask.index_select(0, condition_indices).to(
        device=device,
        non_blocking=device.type == "cuda",
    )
    online = OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=dino_history,
            raw_rgb=_device_tensor(batch, "history_obs_image", device=device, dtype=torch.float32),
        ),
        history=ObservableHistory(
            state=_device_tensor(batch, "state", device=device, dtype=torch.float32),
            action_state=_device_tensor(batch, "action_state", device=device, dtype=torch.float32),
            state_history=_device_tensor(
                batch, "history_state", device=device, dtype=torch.float32
            ),
            executed_action_history=_device_tensor(
                batch,
                "executed_action_history",
                device=device,
                dtype=torch.float32,
            ),
        ),
        goal=GoalCondition(tokens=goal_tokens, mask=goal_mask),
    )
    future = FutureSupervision(
        dino_supports=_device_tensor(
            batch,
            "target_future_dinov2_tokens",
            device=device,
        ),
        action_sequence=_device_tensor(batch, "action", device=device, dtype=torch.float32),
        state_sequence=_device_tensor(batch, "future_state", device=device, dtype=torch.float32),
        offsets=_device_tensor(batch, "target_future_offsets", device=device).long(),
    )
    action = ActionSupervision(
        normalized=_device_tensor(batch, "policy_action", device=device, dtype=torch.float32),
        raw_units=_device_tensor(
            batch, "policy_action_raw", device=device, dtype=torch.float32
        ),
        current_raw_units=_device_tensor(batch, "state_raw", device=device, dtype=torch.float32),
    )
    audit = AuditMetadata(
        sample_index=(
            None if "sample_index" not in batch else _audit_tensor(batch, "sample_index")
        ),
        episode_index=(None if "episode_idx" not in batch else _audit_tensor(batch, "episode_idx")),
        frame_progress=(
            None
            if "frame_progress" not in batch
            else _audit_tensor(batch, "frame_progress", dtype=torch.float32)
        ),
    )
    result = TrainingBatch(online=online, action_target=action, future=future, audit=audit)
    result.validate(config)
    return result


__all__ = [
    "GoalTemplate",
    "MainlineDataBundle",
    "load_mainline_data",
    "to_training_batch",
]
