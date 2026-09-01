"""One explicit data path from cached episodes to typed mainline batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.data.action_chart import project_episodes, resolve_action_state_profile
from clearvla.data.hdf5_episode import (
    LoadedEpisode,
    load_episodes,
    load_hdf5_instruction,
    resolve_too_short_episode_exclusions,
)
from clearvla.data.multitask_selection import (
    RDT_MULTITASK_INTERNAL_SPLITS,
    load_rdt_multitask_selection_manifest,
)
from clearvla.data.samplers import (
    InformationBalancedBatchSampler,
    InformationBalancedSamplerConfig,
    TaskBalancedInformationBatchSampler,
    TaskStratifiedBatchSampler,
)
from clearvla.data.split import (
    RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
    load_rdt_split_manifest,
    resolve_episode_ids,
)
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.online_store import OnlineVisualStore
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
from .language import (
    load_t5_condition_bank,
    source_instruction_inventory_sha256,
)
from .normalizer import ArrayNormalizer
from .normalizer_artifact import load_shared_normalizers
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
    # Exact union of the episode indices materialized by the model-facing
    # datasets.  ``episodes`` remains the complete verified source inventory;
    # unselected and external-only rows must not silently become cache or
    # checkpoint dependencies after split selection.
    materialized_episode_indices: tuple[int, ...]
    action_normalizer: ArrayNormalizer
    state_normalizer: ArrayNormalizer
    goal: GoalTemplate
    skipped: tuple[tuple[str, str], ...]
    sampling_seed: int
    information_uniform_fraction: float
    information_event_fraction: float
    information_motion_quantile: float
    gripper_event_threshold: float | None
    gripper_indices: tuple[int, ...] = (-1,)
    task_order: tuple[str, ...] = ()
    episode_task_indices: tuple[int, ...] = ()
    data_profile_metadata: dict[str, object] = field(default_factory=dict)
    split_metadata: dict[str, object] = field(default_factory=dict)
    normalizer_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def is_multitask(self) -> bool:
        return bool(self.task_order)

    def task_indices_for_episodes(self, episode_indices: Tensor) -> Tensor:
        """Resolve detached episode identities to CPU-only task indices."""

        if not self.is_multitask:
            raise ValueError("the selected data bundle has no multitask registry")
        if episode_indices.ndim != 1:
            raise ValueError("episode indices must be a flat batch vector")
        values = episode_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        if any(index < 0 or index >= len(self.episode_task_indices) for index in values):
            raise IndexError("episode index is outside the multitask registry")
        result = torch.tensor(
            [self.episode_task_indices[index] for index in values],
            dtype=torch.long,
        )
        if bool((result < 0).any()):
            raise ValueError("a model-facing split contains an unregistered task")
        return result

    def dataset_task_indices(self, split: str) -> np.ndarray:
        if split not in self.datasets:
            raise KeyError(f"unknown data split {split!r}")
        refs = self.datasets[split].base.refs
        episode_indices = torch.tensor(
            [int(ref.episode_idx) for ref in refs],
            dtype=torch.long,
        )
        return self.task_indices_for_episodes(episode_indices).numpy()

    def task_registry_summary(self) -> dict[str, object] | None:
        if not self.is_multitask:
            return None
        episode_counts: dict[str, dict[str, int]] = {}
        window_counts: dict[str, dict[str, int]] = {}
        for split, episode_ids in self.splits.items():
            counts = [0 for _ in self.task_order]
            for episode_id in episode_ids:
                task = self.episode_task_indices[int(episode_id)]
                if task >= 0:
                    counts[task] += 1
            episode_counts[split] = {
                name: counts[index]
                for index, name in enumerate(self.task_order)
            }
        for split in self.datasets:
            values = self.dataset_task_indices(split)
            counts = np.bincount(values, minlength=len(self.task_order))
            window_counts[split] = {
                name: int(counts[index])
                for index, name in enumerate(self.task_order)
            }
        return {
            "schema": "clearvla-cpu-task-registry-v1",
            "task_order": list(self.task_order),
            "task_count": len(self.task_order),
            "usage": "sampling_validation_logging_only",
            "model_conditioning": False,
            "episode_counts": episode_counts,
            "window_counts": window_counts,
        }

    def loader(
        self,
        split: str,
        *,
        batch_size: int,
        workers: int,
        device: torch.device,
        shuffle: bool | None = None,
        generator: torch.Generator | None = None,
        task_panel_max_batches: int | None = None,
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
            if self.gripper_event_threshold is None:
                raise ValueError(
                    "the selected data profile has no adopted gripper-event threshold; "
                    "use a deterministic unshuffled loader-only smoke or configure an "
                    "explicit source-chart threshold before training"
                )
            motion_score, is_event = dataset.training_information_signals(
                gripper_indices=self.gripper_indices,
                event_threshold=self.gripper_event_threshold,
            )
            sampler_config = InformationBalancedSamplerConfig(
                batch_size=batch_size,
                uniform_fraction=self.information_uniform_fraction,
                event_fraction=self.information_event_fraction,
                motion_quantile=self.information_motion_quantile,
                seed=self.sampling_seed,
            )
            if self.is_multitask:
                sampler = TaskBalancedInformationBatchSampler(
                    motion_score,
                    is_event,
                    self.dataset_task_indices(split),
                    self.task_order,
                    sampler_config,
                )
            else:
                sampler = InformationBalancedBatchSampler(
                    motion_score,
                    is_event,
                    sampler_config,
                )
            return DataLoader(dataset, batch_sampler=sampler, **common)
        if (
            self.is_multitask
            and not do_shuffle
            and task_panel_max_batches is not None
            and int(task_panel_max_batches) > 0
        ):
            sample_slots = int(task_panel_max_batches) * int(batch_size)
            samples_per_task = sample_slots // len(self.task_order)
            if samples_per_task <= 0:
                raise ValueError(
                    "bounded multitask validation must have at least one sample "
                    "slot per task"
                )
            sampler = TaskStratifiedBatchSampler(
                self.dataset_task_indices(split),
                self.task_order,
                samples_per_task=samples_per_task,
                batch_size=batch_size,
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


def _cpu_task_registry(
    episodes: list[LoadedEpisode],
    split_ids: Mapping[str, list[int]],
    split_metadata: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    selection = split_metadata.get("task_selection")
    if selection is None:
        return (), ()
    if not isinstance(selection, Mapping):
        raise TypeError("task selection metadata must be a mapping")
    raw_order = selection.get("task_order")
    if not isinstance(raw_order, list):
        raise TypeError("task selection metadata has no ordered task registry")
    task_order = tuple(str(value) for value in raw_order)
    if not task_order or len(set(task_order)) != len(task_order):
        raise ValueError("task selection order must be non-empty and unique")
    lookup = {name: index for index, name in enumerate(task_order)}
    episode_task_indices = tuple(
        lookup.get(str(episode.task_id), -1) for episode in episodes
    )
    for split in RDT_MULTITASK_INTERNAL_SPLITS:
        if split not in split_ids:
            raise ValueError(f"task selection is missing internal split {split!r}")
        missing = [
            episodes[index].episode_id
            for index in split_ids[split]
            if episode_task_indices[int(index)] < 0
        ]
        if missing:
            raise ValueError(
                f"selected split {split!r} contains unregistered tasks: {missing[:3]}"
            )
    return task_order, episode_task_indices


def _load_mainline_data(
    config: ExperimentConfig,
    *,
    allow_null_goal: bool = False,
    materialized_splits: tuple[str, ...] | None = None,
    max_episodes_per_materialized_split: int | None = None,
) -> MainlineDataBundle:
    """Build the formal inventory, optionally materializing bounded datasets."""

    config.validate()
    if materialized_splits is None:
        if max_episodes_per_materialized_split is not None:
            raise ValueError("an episode limit requires explicit materialized splits")
    else:
        if not materialized_splits or len(set(materialized_splits)) != len(
            materialized_splits
        ):
            raise ValueError("materialized split names must be non-empty and unique")
        if (
            max_episodes_per_materialized_split is None
            or int(max_episodes_per_materialized_split) <= 0
        ):
            raise ValueError("loader-only materialization requires a positive episode limit")
    data = config.data
    dims = config.dimensions
    obs = config.observation
    cameras = tuple(data.camera_names)
    profile = resolve_action_state_profile(data.data_profile)
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
    if data.split_mode == "manifest":
        min_length = dataset_config.minimum_episode_length
        if min_length != RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH:
            raise AssertionError(
                "RDT manifest minimum length no longer matches the typed window ABI"
            )
    else:
        # Preserve the exact Pen discovery/filter behavior.  Its formal source
        # inventory and 63/5/5 membership are not changed by the RDT adapter.
        min_length = 48 + 8 + 2
    raw_root = Path(data.raw_hdf5_root)
    episodes, skipped = load_episodes(
        raw_root,
        data.hdf5_glob,
        cameras=cameras,
        min_length=min_length,
        action_key=data.action_key,
        state_key=data.state_key,
        camera_key_overrides=data.camera_key_map(),
    )
    episode_names = [episode.episode_id for episode in episodes]
    if data.split_mode == "manifest":
        excluded_too_short = resolve_too_short_episode_exclusions(
            raw_root,
            skipped,
            expected_minimum_length=min_length,
        )
        split_ids, split_metadata = load_rdt_split_manifest(
            data.split_manifest,
            episode_names=episode_names,
            expected_pattern=data.hdf5_glob,
            excluded_too_short=excluded_too_short,
            expected_minimum_episode_length=min_length,
        )
        if data.task_selection_manifest:
            split_ids, selection_metadata = load_rdt_multitask_selection_manifest(
                data.task_selection_manifest,
                episode_names=episode_names,
                task_names=[episode.task_id for episode in episodes],
                instructions=[episode.instruction for episode in episodes],
                base_splits=split_ids,
                base_split_metadata=split_metadata,
            )
            split_metadata = {
                **split_metadata,
                "task_selection": selection_metadata,
            }
    else:
        train_ids, val_ids, test_ids = resolve_episode_ids(
            len(episodes),
            mode=data.split_mode,
            train_frac=0.8,
            val_frac=0.1,
            seed=data.seed,
            train_episode_count=data.train_episodes,
            val_episode_count=data.val_episodes,
            test_episode_count=data.test_episodes,
            episode_names=episode_names,
        )
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
        split_metadata = {
            "schema": "clearvla-ordered-counts-split-v1",
            "split_counts": {name: len(values) for name, values in split_ids.items()},
        }
    if materialized_splits is None:
        materialized_names = (
            RDT_MULTITASK_INTERNAL_SPLITS
            if data.task_selection_manifest
            else tuple(split_ids)
        )
        dataset_episode_ids = {
            name: list(split_ids[name]) for name in materialized_names
        }
    else:
        unknown_splits = sorted(set(materialized_splits) - set(split_ids))
        if unknown_splits:
            raise ValueError(
                f"loader-only materialization names unknown splits: {unknown_splits}"
            )
        assert max_episodes_per_materialized_split is not None
        dataset_episode_ids = {
            name: list(split_ids[name][: int(max_episodes_per_materialized_split)])
            for name in materialized_splits
        }
        empty_splits = [name for name, ids in dataset_episode_ids.items() if not ids]
        if empty_splits:
            raise ValueError(
                f"loader-only materialization selected empty splits: {empty_splits}"
            )
    required_token_episode_ids = sorted(
        {index for ids in dataset_episode_ids.values() for index in ids}
    )
    episodes = project_episodes(episodes, profile)
    action_normalizer, state_normalizer = _normalizers(
        episodes,
        split_ids["train"],
        mode=data.normalizer,
    )
    normalizer_metadata: dict[str, object] = {
        "source": "fresh_train_only_fit",
        "train_episode_count": len(split_ids["train"]),
    }
    if data.normalizer_artifact:
        selection_metadata = split_metadata.get("task_selection")
        if not isinstance(selection_metadata, dict):
            raise ValueError("a shared normalizer artifact requires task selection metadata")
        action_normalizer, state_normalizer, normalizer_metadata = load_shared_normalizers(
            data.normalizer_artifact,
            expected_selection_sha256=str(
                selection_metadata.get("selection_sha256", "")
            ),
            expected_profile_sha256=profile.digest(),
            expected_train_episode_ids=[
                episodes[index].episode_id for index in split_ids["train"]
            ],
            computed_action=action_normalizer,
            computed_state=state_normalizer,
        )
    preprocessing = PreprocessConfig(resize_hw=(data.cache_side, data.cache_side), crop_hw=None)
    if data.image_store_mode == "decoded-cache":
        image_store: DecodedImageStore | OnlineVisualStore = DecodedImageStore(
            Path(data.decoded_cache),
            camera_names=cameras,
            preprocessing=preprocessing,
        )
    else:
        image_store = OnlineVisualStore(
            camera_names=cameras,
            preprocessing=preprocessing,
            frame_lru_capacity=data.image_frame_lru_capacity,
            open_file_capacity=data.image_open_file_capacity,
        )
    token_store = DinoV2TokenStore(
        Path(data.dino_cache),
        episodes=episodes,
        camera_names=cameras,
        preprocessing=preprocessing,
        dinov2_model=data.dinov2_model,
        required_episode_indices=required_token_episode_ids,
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
    datasets: dict[str, CachedTokenPolicyWindowDataset] = {}
    for name, ids in dataset_episode_ids.items():
        base = ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=dataset_config,
            gripper_transition_boundary=profile.gripper_transition_boundary,
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
    if data.split_mode == "manifest":
        if not goal_bank.is_instruction_bank:
            raise ValueError(
                "RDT manifest data requires a per-instruction T5 condition bank"
            )
        eligible_instructions = [episode.instruction for episode in episodes]
        if any(value is None for value in eligible_instructions):
            raise ValueError("every RDT episode must own an HDF5 instruction")
        excluded_instructions = [
            load_hdf5_instruction(Path(path)) for path, _reason in skipped
        ]
        if any(value is None for value in excluded_instructions):
            raise ValueError("every excluded RDT source episode must own an instruction")
        source_instructions = [
            str(value) for value in (*eligible_instructions, *excluded_instructions)
        ]
        source_episode_count = int(
            goal_bank.metadata.get("source_episode_count", -1)
        )
        if source_episode_count != len(source_instructions):
            raise ValueError(
                "T5 instruction bank source episode count differs from the live RDT inventory"
            )
        source_digest = source_instruction_inventory_sha256(source_instructions)
        if (
            str(
                goal_bank.metadata.get(
                    "source_instruction_inventory_sha256", ""
                )
            )
            != source_digest
        ):
            raise ValueError(
                "T5 instruction bank source instruction inventory differs from live RDT data"
            )
    episode_condition_indices = goal_bank.condition_indices(
        [episode.instruction for episode in episodes]
    )
    task_order, episode_task_indices = _cpu_task_registry(
        episodes,
        split_ids,
        split_metadata,
    )
    return MainlineDataBundle(
        episodes=tuple(episodes),
        splits={name: tuple(ids) for name, ids in split_ids.items()},
        datasets=datasets,
        materialized_episode_indices=tuple(required_token_episode_ids),
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
        gripper_event_threshold=(
            config.objectives.gripper_event_threshold
            if profile.name == "identity_7d_pen"
            and data.sampling_gripper_event_threshold is None
            else data.sampling_gripper_event_threshold
        ),
        gripper_indices=profile.gripper_indices,
        task_order=task_order,
        episode_task_indices=episode_task_indices,
        data_profile_metadata={
            **profile.as_dict(),
            "sha256": profile.digest(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        split_metadata=split_metadata,
        normalizer_metadata=normalizer_metadata,
    )


def load_mainline_data(
    config: ExperimentConfig,
    *,
    allow_null_goal: bool = False,
) -> MainlineDataBundle:
    """Build every formal dataset; training and validation use only this path."""

    return _load_mainline_data(config, allow_null_goal=allow_null_goal)


def load_mainline_data_for_smoke(
    config: ExperimentConfig,
    *,
    split: str,
    episode_limit: int = 1,
    allow_null_goal: bool = False,
) -> MainlineDataBundle:
    """Verify the full inventory while requiring cache rows for a bounded lane."""

    return _load_mainline_data(
        config,
        allow_null_goal=allow_null_goal,
        materialized_splits=(str(split),),
        max_episodes_per_materialized_split=int(episode_limit),
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
        current_raw_units=_device_tensor(
            batch, "action_state_raw", device=device, dtype=torch.float32
        ),
        gripper_transition_boundary=_device_tensor(
            batch,
            "gripper_transition_boundary",
            device=device,
            dtype=torch.float32,
        ),
        gripper_transition_boundary_raw_units=_device_tensor(
            batch,
            "gripper_transition_boundary_raw",
            device=device,
            dtype=torch.float32,
        ),
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
    "load_mainline_data_for_smoke",
    "to_training_batch",
]
