"""One explicit data path from cached episodes to typed mainline batches."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.benchmarks.calvin_raw import (
    CalvinRawReader,
    virtualize_calvin_cached_prefix,
)
from clearvla.data.action_chart import project_episodes, resolve_action_state_profile
from clearvla.data.hdf5_episode import (
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
    LoadedEpisode,
    load_episodes,
    load_hdf5_instruction,
    resolve_too_short_episode_exclusions,
)
from clearvla.data.history_clock import HISTORY_TIMING_KEYS
from clearvla.data.libero_retarget import (
    LIBERO_RETARGET_NORMALIZER_POLICY,
    load_libero_retarget_overlay_contract,
    merge_libero_retarget_training_overlay,
)
from clearvla.data.multitask_selection import (
    RDT_MULTITASK_INTERNAL_SPLITS,
    load_rdt_multitask_selection_manifest,
)
from clearvla.data.samplers import (
    BoundaryAwareInformationBatchSampler,
    EvenlySpacedPanelBatchSampler,
    InformationBalancedBatchSampler,
    InformationBalancedSamplerConfig,
    TaskBalancedInformationBatchSampler,
    TaskStratifiedBatchSampler,
)
from clearvla.data.split import (
    RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
    load_episode_split_manifest,
    load_episode_split_manifest_inventory,
    load_rdt_split_manifest,
    resolve_episode_ids,
)
from clearvla.data.window_boundaries import (
    CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    CAUSAL_PREFIX_V1,
    PREFIX_REGION,
    STRICT_COMPLETE_V1,
    TAIL_REGION,
)
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.online_store import OnlineVisualStore
from clearvla.vision.preprocessing import PreprocessConfig

from ..config import ExperimentConfig
from ..gripper_contract import is_binary_gripper_mode
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
from ..temporal import TIMED_HISTORY_ENCODING, HistoryTiming
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


def _configure_worker_tensor_sharing(workers: int) -> None:
    """Use path-backed tensor sharing when multiprocessing workers are active.

    The mainline batches contain many independent tensor storages.  PyTorch's
    default ``file_descriptor`` strategy keeps one descriptor per shared
    storage in each worker, which can exhaust the server's 1024-FD soft limit
    during a long CUDA run even with a bounded prefetch queue.  The
    ``file_system`` strategy keeps the same tensor ABI while moving that
    bookkeeping out of the worker descriptor table.  Fail closed if the
    runtime cannot provide it; silently falling back would recreate the
    original EMFILE failure.
    """

    if workers <= 0:
        return
    desired = "file_system"
    try:
        current = torch.multiprocessing.get_sharing_strategy()
        if current != desired:
            torch.multiprocessing.set_sharing_strategy(desired)
        if torch.multiprocessing.get_sharing_strategy() != desired:
            raise RuntimeError("PyTorch did not retain the requested sharing strategy")
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "mainline multiprocessing requires PyTorch file_system tensor sharing"
        ) from exc


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
    information_batches_per_epoch: int | None = None
    sampling_arm_motion: str = "adjacent_action_delta"
    sampling_gripper_event_scope: str = "window_any"
    release_first_action_fraction: float = 0.0
    gripper_indices: tuple[int, ...] = (-1,)
    gripper_open_direction: int = -1
    window_boundary_contract: str = STRICT_COMPLETE_V1
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
                name: counts[index] for index, name in enumerate(self.task_order)
            }
        for split in self.datasets:
            values = self.dataset_task_indices(split)
            counts = np.bincount(values, minlength=len(self.task_order))
            window_counts[split] = {
                name: int(counts[index]) for index, name in enumerate(self.task_order)
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
        coverage_panel_max_batches: int | None = None,
    ) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"unknown data split {split!r}")
        if batch_size <= 0 or workers < 0:
            raise ValueError("batch size must be positive and workers non-negative")
        do_shuffle = split == "train" if shuffle is None else bool(shuffle)
        _configure_worker_tensor_sharing(workers)
        # The training loop owns a long-lived iterator, so persistent workers
        # are useful there.  Validation is also used by the startup preflight
        # through ``next(iter(val_loader))``; keeping those workers persistent
        # would let them prefetch the entire validation set after that one
        # batch is consumed, retaining one shared-storage FD per tensor until
        # the soft limit is reached (PyTorch then raises ``EMFILE`` in the
        # worker feeder).  Validation iterators must therefore be
        # self-terminating, especially when the preflight iterator is dropped.
        persistent_workers = workers > 0 and split == "train"
        common = {
            "num_workers": workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": persistent_workers,
            "generator": generator,
        }
        if workers > 0:
            # A policy sample contains many independent tensor storages.  The
            # default prefetch factor of two can therefore consume most of a
            # 1024-FD worker budget even when every storage is released
            # correctly.  One in-flight batch per worker keeps the queue
            # bounded without changing sample order or model semantics.
            common["prefetch_factor"] = 1
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
            signal_kwargs = {
                "gripper_indices": self.gripper_indices,
                "event_threshold": self.gripper_event_threshold,
                "arm_motion": self.sampling_arm_motion,
            }
            # Keep lightweight test/dataset doubles compatible with the
            # historical method while making the non-default experiment
            # explicit on the real cached-token dataset.
            if self.sampling_gripper_event_scope != "window_any":
                signal_kwargs["event_scope"] = self.sampling_gripper_event_scope
            motion_score, is_event = dataset.training_information_signals(
                **signal_kwargs
            )
            release_first_events = None
            if self.release_first_action_fraction > 0.0:
                release_signal = getattr(
                    dataset, "training_release_first_signals", None
                )
                if not callable(release_signal):
                    raise TypeError(
                        "release-first sampling requires a dataset-owned "
                        "training_release_first_signals implementation"
                    )
                release_first_events = release_signal(
                    gripper_indices=self.gripper_indices,
                    event_threshold=self.gripper_event_threshold,
                    open_direction=self.gripper_open_direction,
                )
            sampler_config = InformationBalancedSamplerConfig(
                batch_size=batch_size,
                uniform_fraction=self.information_uniform_fraction,
                event_fraction=self.information_event_fraction,
                motion_quantile=self.information_motion_quantile,
                batches_per_epoch=self.information_batches_per_epoch,
                seed=self.sampling_seed,
                event_scope=self.sampling_gripper_event_scope,
            )
            if self.window_boundary_contract in {
                CAUSAL_PREFIX_V1,
                CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
            }:
                sampler = BoundaryAwareInformationBatchSampler(
                    motion_score,
                    is_event,
                    dataset.boundary_regions,
                    contract=self.window_boundary_contract,
                    config=sampler_config,
                    release_first_events=release_first_events,
                    release_first_fraction=self.release_first_action_fraction,
                )
            elif self.is_multitask:
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
                    "bounded multitask validation must have at least one sample slot per task"
                )
            sampler = TaskStratifiedBatchSampler(
                self.dataset_task_indices(split),
                self.task_order,
                samples_per_task=samples_per_task,
                batch_size=batch_size,
            )
            return DataLoader(dataset, batch_sampler=sampler, **common)
        if (
            not do_shuffle
            and coverage_panel_max_batches is not None
            and int(coverage_panel_max_batches) > 0
        ):
            sampler = EvenlySpacedPanelBatchSampler(
                len(dataset),
                batch_size=batch_size,
                max_batches=int(coverage_panel_max_batches),
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
        episode = episodes[index]
        action = episode.actions_raw
        state = episode.states_raw
        if action is None or state is None:
            raise ValueError("mainline normalization requires state and action arrays")
        # Synthetic absorbing rows exist only to close a fixed 48-frame
        # Teacher horizon around a shorter terminal task.  They must never
        # change the physical source chart fitted from real demonstrations.
        if episode.state_normalizer_reference_raw is not None:
            # Causal LIBERO conversion changes observation time alignment but
            # model-only E8 initialization must retain the exact E8 affine
            # chart.  This numeric-only reference is copied from the audited
            # legacy real rows by the converter and is never sampled as model
            # evidence.
            reference = episode.state_normalizer_reference_raw
            if episode.terminal_state_index is None:
                action_rows.append(action)
            elif (
                episode.terminal_padding_mode
                == LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                and episode.source_action_count is not None
            ):
                action_rows.append(action[: int(episode.source_action_count)])
            else:
                raise ValueError(
                    "a state normalizer reference is valid only for causal LIBERO data"
                )
            state_rows.append(reference)
        elif episode.terminal_state_index is None:
            action_rows.append(action)
            state_rows.append(state)
        elif (
            episode.terminal_padding_mode
            == LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
        ):
            if episode.source_action_count is None:
                raise ValueError(
                    "LIBERO terminal replay is missing source_action_count"
                )
            source_count = int(episode.source_action_count)
            if int(episode.terminal_state_index) != source_count:
                raise ValueError(
                    "LIBERO terminal replay source/terminal indices disagree"
                )
            # The replayed terminal observation is genuine simulator evidence,
            # but it was not part of the E8 source chart.  Fit both affine
            # normalizers from exactly the original T pre-action rows so Test B
            # remains numerically comparable to E8 and Test A.
            action_rows.append(action[:source_count])
            state_rows.append(state[:source_count])
        elif (
            episode.terminal_padding_mode
            == RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
        ):
            terminal = int(episode.terminal_state_index)
            action_rows.append(action[:terminal])
            state_rows.append(state[: terminal + 1])
        else:
            raise ValueError(
                "an episode with terminal_state_index has an unknown padding contract"
            )
    actions = ArrayNormalizer.fit_zscore(action_rows)
    states = ArrayNormalizer.fit_zscore(state_rows)
    return actions, states


def _cpu_task_registry(
    episodes: list[LoadedEpisode],
    split_ids: Mapping[str, list[int]],
    split_metadata: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    selection = split_metadata.get("task_selection")
    # RDT selection manifests nest their registry under ``task_selection``;
    # CALVIN's trajectory manifest carries the same immutable registry at the
    # top level.  Both are CPU-only bookkeeping and never enter model
    # conditioning.
    if selection is None:
        raw_order = split_metadata.get("task_order")
    else:
        if not isinstance(selection, Mapping):
            raise TypeError("task selection metadata must be a mapping")
        raw_order = selection.get("task_order")
    if not isinstance(raw_order, list):
        return (), ()
    task_order = tuple(str(value) for value in raw_order)
    if not task_order or len(set(task_order)) != len(task_order):
        raise ValueError("task selection order must be non-empty and unique")
    lookup = {name: index for index, name in enumerate(task_order)}
    episode_task_indices = tuple(lookup.get(str(episode.task_id), -1) for episode in episodes)
    required_splits = (
        RDT_MULTITASK_INTERNAL_SPLITS
        if selection is not None
        else ("train", "val", "test")
    )
    for split in required_splits:
        if split not in split_ids:
            raise ValueError(f"task selection is missing internal split {split!r}")
        missing = [
            episodes[index].episode_id
            for index in split_ids[split]
            if episode_task_indices[int(index)] < 0
        ]
        if missing:
            raise ValueError(f"selected split {split!r} contains unregistered tasks: {missing[:3]}")
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
        if not materialized_splits or len(set(materialized_splits)) != len(materialized_splits):
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
    strict_dataset_config = ObservedStateDatasetConfig(
        emit_history_timing=config.top.history_encoding_mode == TIMED_HISTORY_ENCODING,
        world_horizon=48,
        policy_horizon=dims.action_horizon,
        support_stride=4,
        state_history_offsets=(-8, -4, 0),
        visual_history_offsets=(-2 * obs.flow_reference_frames, -obs.flow_reference_frames, 0),
        executed_action_offsets=(-24, -16, -12, -8, -6, -4, -2, -1),
        stride=data.stride,
        causal_reset_padding=profile.name == "maniskill_pd_ee_delta_pose_7d_v2",
        window_boundary_contract=STRICT_COMPLETE_V1,
    )
    strict_dataset_config.validate()
    train_dataset_config = (
        strict_dataset_config
        if data.window_boundary_contract == STRICT_COMPLETE_V1
        else replace(
            strict_dataset_config,
            causal_reset_padding=True,
            window_boundary_contract=data.window_boundary_contract,
        )
    )
    train_dataset_config.validate()
    if data.split_mode in {"manifest", "episode-manifest"}:
        min_length = strict_dataset_config.minimum_episode_length
        if data.split_mode == "manifest" and min_length != RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH:
            raise AssertionError(
                "RDT manifest minimum length no longer matches the typed window ABI"
            )
    else:
        # Preserve the exact Pen discovery/filter behavior.  Its formal source
        # inventory and 63/5/5 membership are not changed by the RDT adapter.
        min_length = 48 + 8 + 2
    raw_root = Path(data.raw_hdf5_root)
    maniskill_audit = None
    if profile.name in {"maniskill_pd_ee_delta_pose_7d_v1", "maniskill_pd_ee_delta_pose_7d_v2"}:
        from clearvla.simulation.admission import audit_stackcube_experts

        if data.action_state_key != "action_state" or data.state_key != "state":
            raise ValueError("ManiSkill requires explicit state/action_state dataset keys")
        maniskill_audit = audit_stackcube_experts(
            raw_root,
            minimum_length=strict_dataset_config.minimum_episode_length,
            profile=profile.name,
            require_binary_gripper=is_binary_gripper_mode(
                config.bottom.gripper_output_mode
            ),
        )
    episode_inventory = None
    manifest_payload: dict[str, object] | None = None
    if data.split_mode == "episode-manifest":
        _manifest_path, _manifest_payload, episode_inventory = (
            load_episode_split_manifest_inventory(data.split_manifest)
        )
        manifest_payload = _manifest_payload
    # The old converter-v1 CALVIN prefix ends at the annotation terminal and
    # therefore is shorter than the typed 48-frame window.  Raw-overlay mode
    # appends the official absorbing suffix in memory, so only require a
    # non-empty physical prefix during HDF5 discovery.  The virtualized
    # episode is checked against the normal typed-window bounds below.
    load_min_length = 1 if data.calvin_raw_source.strip() else min_length
    episodes, skipped = load_episodes(
        raw_root,
        data.hdf5_glob,
        cameras=cameras,
        min_length=load_min_length,
        action_key=data.action_key,
        action_state_key=data.action_state_key or None,
        state_key=data.state_key,
        camera_key_overrides=data.camera_key_map(),
        episode_names=episode_inventory,
    )
    if data.task_filter:
        requested_task = str(data.task_filter)
        filtered = [episode for episode in episodes if episode.task_id == requested_task]
        if not filtered:
            observed = sorted({episode.task_id for episode in episodes if episode.task_id})
            raise ValueError(
                f"data.task_filter={requested_task!r} matched no episodes; "
                f"observed examples={observed[:12]}"
            )
        episodes = filtered
    overlay_report: dict[str, object] | None = None
    raw_reader_report: dict[str, object] | None = None
    if data.calvin_raw_source.strip():
        if profile.name != "calvin_relative_7d_v1":
            raise ValueError("CALVIN raw overlay requires calvin_relative_7d_v1")
        if data.split_mode != "episode-manifest" or manifest_payload is None:
            raise ValueError("CALVIN raw overlay requires an episode split manifest")
        raw_source = Path(data.calvin_raw_source)
        manifest_source = str(manifest_payload.get("raw_source", "")).strip()
        if manifest_source and Path(manifest_source).resolve() != raw_source.resolve():
            raise ValueError(
                "CALVIN split manifest raw_source differs from data.calvin_raw_source"
            )
        manifest_splits = manifest_payload.get("splits")
        if not isinstance(manifest_splits, Mapping):
            raise ValueError("CALVIN raw overlay manifest has no train/val/test splits")
        raw_episode_map = manifest_payload.get("raw_episode_map")
        if raw_episode_map is not None and not isinstance(raw_episode_map, Mapping):
            raise ValueError("CALVIN raw overlay manifest raw_episode_map must be a mapping")
        split_seed = int(manifest_payload.get("split_seed", data.seed))
        val_fraction = float(
            manifest_payload.get("train_validation_fraction", 0.1)
        )
        raw_reader = CalvinRawReader(
            raw_source,
            val_fraction=val_fraction,
            split_seed=split_seed,
            task_filter=data.task_filter or None,
        )
        episodes, overlay_report = virtualize_calvin_cached_prefix(
            episodes,
            raw_reader,
            expected_splits=manifest_splits,
            raw_episode_map=raw_episode_map,
        )
        raw_reader_report = raw_reader.report()
    if profile.name == "calvin_relative_7d_v1":
        invalid_terminal_contract = [
            episode.episode_id
            for episode in episodes
            if (
                episode.terminal_state_index is None
                or episode.terminal_padding_mode
                != RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
            )
        ]
        if invalid_terminal_contract:
            raise ValueError(
                "CALVIN training requires converter-v2 absorbing terminal padding; "
                "old converted episodes omit the success tail from direct policy "
                f"supervision: {invalid_terminal_contract[:8]}"
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
    elif data.split_mode == "episode-manifest":
        train_ids, val_ids, test_ids, split_metadata = load_episode_split_manifest(
            data.split_manifest,
            episode_names=episode_names,
        )
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
        manifest_task_filter = str(split_metadata.get("task_filter", ""))
        if manifest_task_filter != data.task_filter:
            raise ValueError("episode split manifest task_filter differs from data.task_filter")
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
    if overlay_report is not None:
        # The split manifest is the immutable membership contract; the report
        # below records what was actually overlaid and keeps raw eligibility
        # versus cached-prefix coverage explicit in the run context.
        split_metadata = {
            **split_metadata,
            "calvin_raw_overlay": overlay_report,
            "calvin_raw_reader": raw_reader_report,
            "calvin_overlay_active": True,
        }
    elif data.calvin_raw_source.strip():
        raise AssertionError("CALVIN raw overlay source was configured but not applied")

    # The immutable base split exclusively owns both affine normalizers.  Fit
    # them before loading a retarget HDF5 so an implementation reorder cannot
    # silently let synthetic translations redefine the action/state chart.
    episodes = project_episodes(episodes, profile)
    base_train_ids = tuple(int(value) for value in split_ids["train"])
    action_normalizer, state_normalizer = _normalizers(
        episodes,
        base_train_ids,
        mode=data.normalizer,
    )
    normalizer_metadata: dict[str, object] = {
        "source": "fresh_train_only_fit",
        "train_episode_count": len(base_train_ids),
    }
    if data.normalizer_artifact:
        selection_metadata = split_metadata.get("task_selection")
        if not isinstance(selection_metadata, dict):
            raise ValueError("a shared normalizer artifact requires task selection metadata")
        action_normalizer, state_normalizer, normalizer_metadata = load_shared_normalizers(
            data.normalizer_artifact,
            expected_selection_sha256=str(selection_metadata.get("selection_sha256", "")),
            expected_profile_sha256=profile.digest(),
            expected_train_episode_ids=[
                episodes[index].episode_id for index in base_train_ids
            ],
            computed_action=action_normalizer,
            computed_state=state_normalizer,
        )

    if data.libero_retarget_overlay_root.strip():
        contract = load_libero_retarget_overlay_contract(
            data.libero_retarget_overlay_root,
            data.libero_retarget_overlay_manifest,
            base_split_manifest=data.split_manifest,
            base_causal_root=raw_root,
            base_train_episode_ids=[
                episodes[index].episode_id for index in base_train_ids
            ],
        )
        overlay_episodes, overlay_skipped = load_episodes(
            contract.root,
            "*.hdf5",
            cameras=cameras,
            min_length=strict_dataset_config.minimum_episode_length,
            action_key=data.action_key,
            action_state_key=data.action_state_key or None,
            state_key=data.state_key,
            camera_key_overrides=data.camera_key_map(),
            episode_names=contract.episode_ids,
        )
        if overlay_skipped:
            raise RuntimeError(
                "LIBERO retarget overlay contains unusable HDF5 episodes: "
                f"{overlay_skipped[:3]}"
            )
        merge = merge_libero_retarget_training_overlay(
            episodes,
            split_ids,
            project_episodes(overlay_episodes, profile),
            contract,
        )
        episodes = list(merge.episodes)
        split_ids = {name: list(values) for name, values in merge.splits.items()}
        split_metadata = {
            **split_metadata,
            "libero_retarget_overlay_active": True,
            "libero_retarget_overlay": merge.metadata,
        }
        normalizer_metadata = {
            **normalizer_metadata,
            "libero_retarget_overlay_excluded": True,
            "libero_retarget_overlay_episode_count": len(
                merge.overlay_episode_indices
            ),
            "libero_retarget_normalizer_policy": LIBERO_RETARGET_NORMALIZER_POLICY,
            "effective_train_episode_count": len(split_ids["train"]),
        }

    if materialized_splits is None:
        materialized_names = (
            RDT_MULTITASK_INTERNAL_SPLITS if data.task_selection_manifest else tuple(split_ids)
        )
        dataset_episode_ids = {name: list(split_ids[name]) for name in materialized_names}
    else:
        unknown_splits = sorted(set(materialized_splits) - set(split_ids))
        if unknown_splits:
            raise ValueError(f"loader-only materialization names unknown splits: {unknown_splits}")
        assert max_episodes_per_materialized_split is not None
        dataset_episode_ids = {
            name: list(split_ids[name][: int(max_episodes_per_materialized_split)])
            for name in materialized_splits
        }
        empty_splits = [name for name, ids in dataset_episode_ids.items() if not ids]
        if empty_splits:
            raise ValueError(f"loader-only materialization selected empty splits: {empty_splits}")
    required_token_episode_ids = sorted(
        {index for ids in dataset_episode_ids.values() for index in ids}
    )
    preprocessing = PreprocessConfig(resize_hw=(data.cache_side, data.cache_side), crop_hw=None)
    if data.image_store_mode == "decoded-cache":
        image_store: DecodedImageStore | OnlineVisualStore = DecodedImageStore(
            Path(data.decoded_cache),
            camera_names=cameras,
            preprocessing=preprocessing,
            read_backend=data.visual_cache_read_backend,
            max_open_arrays=data.visual_pread_max_open_files,
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
        read_backend=data.visual_cache_read_backend,
        max_open_arrays=data.visual_pread_max_open_files,
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

    def materialize_dataset(
        name: str,
        ids: list[int],
        *,
        selected_config: ObservedStateDatasetConfig,
        allowed_regions: tuple[str, ...] | None = None,
    ) -> None:
        base = ObservedStateWindowDataset(
            episodes,
            ids,
            image_store=image_store,
            camera_names=cameras,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            config=selected_config,
            gripper_transition_boundary=profile.gripper_transition_boundary,
            allowed_boundary_regions=allowed_regions,
        )
        datasets[name] = CachedTokenPolicyWindowDataset(
            base,
            token_store=token_store,
        )

    for name, ids in dataset_episode_ids.items():
        materialize_dataset(
            name,
            ids,
            selected_config=(
                train_dataset_config if name == "train" else strict_dataset_config
            ),
        )
    # Keep the primary validation/test surfaces strict and therefore directly
    # comparable with E8.  Boundary rows get separate, deployment-only panels;
    # they can never select the best checkpoint or dilute the strict metric.
    if "val" in dataset_episode_ids and data.window_boundary_contract in {
        CAUSAL_PREFIX_V1,
        CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    }:
        materialize_dataset(
            "val_prefix",
            dataset_episode_ids["val"],
            selected_config=train_dataset_config,
            allowed_regions=(PREFIX_REGION,),
        )
        if data.window_boundary_contract == CAUSAL_PREFIX_TERMINAL_SUFFIX_V2:
            materialize_dataset(
                "val_tail",
                dataset_episode_ids["val"],
                selected_config=train_dataset_config,
                allowed_regions=(TAIL_REGION,),
            )
    goal_bank = load_t5_condition_bank(
        data.t5_condition,
        max_tokens=dims.goal_max_tokens,
        expected_width=dims.goal_token_dim,
        allow_null=allow_null_goal,
    )
    if data.split_mode in {"manifest", "episode-manifest"}:
        if not goal_bank.is_instruction_bank:
            raise ValueError(
                "manifest-backed benchmark data requires a per-instruction T5 condition bank"
            )
    if data.split_mode == "manifest":
        eligible_instructions = [episode.instruction for episode in episodes]
        if any(value is None for value in eligible_instructions):
            raise ValueError("every RDT episode must own an HDF5 instruction")
        excluded_instructions = [load_hdf5_instruction(Path(path)) for path, _reason in skipped]
        if any(value is None for value in excluded_instructions):
            raise ValueError("every excluded RDT source episode must own an instruction")
        source_instructions = [
            str(value) for value in (*eligible_instructions, *excluded_instructions)
        ]
        source_episode_count = int(goal_bank.metadata.get("source_episode_count", -1))
        if source_episode_count != len(source_instructions):
            raise ValueError(
                "T5 instruction bank source episode count differs from the live RDT inventory"
            )
        source_digest = source_instruction_inventory_sha256(source_instructions)
        if str(goal_bank.metadata.get("source_instruction_inventory_sha256", "")) != source_digest:
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
        information_batches_per_epoch=data.information_batches_per_epoch,
        gripper_event_threshold=(
            config.objectives.gripper_event_threshold
            if profile.name == "identity_7d_pen" and data.sampling_gripper_event_threshold is None
            else data.sampling_gripper_event_threshold
        ),
        sampling_arm_motion=profile.sampling_arm_motion,
        sampling_gripper_event_scope=data.sampling_gripper_event_scope,
        release_first_action_fraction=data.release_first_action_fraction,
        gripper_indices=profile.gripper_indices,
        gripper_open_direction=profile.gripper_open_direction,
        window_boundary_contract=data.window_boundary_contract,
        task_order=task_order,
        episode_task_indices=episode_task_indices,
        data_profile_metadata={
            **profile.as_dict(),
            "sha256": profile.digest(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
            "gripper_open_direction": profile.gripper_open_direction,
            **({"simulator_environment": maniskill_audit["environment"],
                "expert_reset_seeds": maniskill_audit["reset_seeds"],
                "expert_admission": maniskill_audit["expert_admission"]}
               if maniskill_audit is not None else {}),
            **(
                {"sampling_arm_motion": profile.sampling_arm_motion}
                if profile.sampling_arm_motion != "adjacent_action_delta"
                else {}
            ),
            "sampling_gripper_event_scope": data.sampling_gripper_event_scope,
            # Physical units/ranges are metadata-only.  They deliberately do
            # not enter the numeric action-profile digest or alter z-score
            # normalization, projection, decoding, or model conditioning.
            "physical_chart": profile.physical_chart.as_dict(),
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
    action_state = _device_tensor(
        batch,
        "action_state",
        device=device,
        dtype=torch.float32,
    )
    gripper_transition_boundary = _device_tensor(
        batch,
        "gripper_transition_boundary",
        device=device,
        dtype=torch.float32,
    )
    timing = None
    if config.top.history_encoding_mode == TIMED_HISTORY_ENCODING:
        timing = HistoryTiming.from_mapping(
            {name: _device_tensor(batch, name, device=device) for name in HISTORY_TIMING_KEYS}
        )
        timing.validate(
            batch=batch_size,
            states=config.dimensions.state_history_length,
            actions=config.dimensions.executed_history_length,
            device=dino_history.device,
            strict=True,
        )
    online = OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=dino_history,
            raw_rgb=_device_tensor(batch, "history_obs_image", device=device, dtype=torch.float32),
        ),
        history=ObservableHistory(
            state=_device_tensor(batch, "state", device=device, dtype=torch.float32),
            action_state=action_state,
            timing=timing,
            codec_gripper_boundary=gripper_transition_boundary[:, -1:],
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
        raw_units=_device_tensor(batch, "policy_action_raw", device=device, dtype=torch.float32),
        current_raw_units=_device_tensor(
            batch, "action_state_raw", device=device, dtype=torch.float32
        ),
        gripper_transition_boundary=gripper_transition_boundary,
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
