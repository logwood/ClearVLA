"""Minimal observed-state dataset owned by the capability mainline.

Only values consumed by :class:`TrainingBatch` are materialized.  Future RGB,
target-history tensors and ancestry-only descriptor views are deliberately
absent: the teacher consumes source-declared cached DINO supports every four
steps (4..48 in the legacy layout, 4..24 in the control-aligned layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from clearvla.data.annotation_endpoint import resolve_annotation_endpoint
from clearvla.data.future_clock import OBSERVED_FUTURE_CONTRACT, future_source_rows
from clearvla.data.hdf5_episode import LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING, LoadedEpisode
from clearvla.data.history_clock import sparse_history_clock
from clearvla.data.state_features import (
    NATIVE_AFFINE_STATE,
    encode_state_features,
    state_feature_width,
    validate_state_feature_profile,
)
from clearvla.data.window_boundaries import (
    BOUNDARY_REGION_TO_INDEX,
    BOUNDARY_REGIONS,
    CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
    CAUSAL_PREFIX_V1,
    OBSERVED_TAIL_V1,
    STRICT_COMPLETE_V1,
    WindowBoundaryMetadata,
    WindowBoundaryPlan,
    policy_action_union_coverage,
    resolve_window_boundary_plan,
)
from clearvla.vision.decoded_image_store import DecodedImageStore
from clearvla.vision.online_store import OnlineVisualStore

from ..executed_world import EXECUTED_WORLD_FEEDBACK
from ..instruction_reference import INSTRUCTION_START_REFERENCE
from .normalizer import ArrayNormalizer
from .token_store import DinoV2TokenStore


@dataclass(frozen=True)
class ObservedStateDatasetConfig:
    world_horizon: int = 48
    policy_horizon: int = 24
    support_stride: int = 4
    state_history_offsets: tuple[int, ...] = (-8, -4, 0)
    visual_history_offsets: tuple[int, ...] = (-8, -4, 0)
    executed_action_offsets: tuple[int, ...] = (-24, -16, -12, -8, -6, -4, -2, -1)
    state_offset: int = 0
    image_offset: int = 0
    action_offset: int = 0
    stride: int = 1
    causal_reset_padding: bool = False
    emit_history_timing: bool = False
    instruction_reference_mode: str = "none"
    annotation_goal_mode: str = "none"
    robot_feedback_mode: str = "none"
    world_feedback_mode: str = "none"
    state_feature_mode: str = NATIVE_AFFINE_STATE
    state_profile: str = "identity_7d_pen"
    window_boundary_contract: str = STRICT_COMPLETE_V1

    def validate(self) -> None:
        if self.annotation_goal_mode not in {"none", "annotated_endpoint_relation_v1"}:
            raise ValueError("unknown annotated endpoint dataset mode")
        if self.annotation_goal_mode != "none" and (
            self.instruction_reference_mode != INSTRUCTION_START_REFERENCE
            or self.window_boundary_contract != OBSERVED_TAIL_V1
        ):
            raise ValueError("annotated endpoints require real-tail instruction provenance")
        if self.world_feedback_mode not in {"none",EXECUTED_WORLD_FEEDBACK}:
            raise ValueError("unknown executed world source producer")
        if self.world_feedback_mode!="none" and (
            not self.emit_history_timing or self.state_profile!="calvin_relative_7d_v1"
            or self.world_horizon!=24 or self.support_stride!=4 or self.state_history_offsets!=(-8,-4,0)):
            raise ValueError("executed world source requires aligned CALVIN history and W endpoints")
        if self.robot_feedback_mode not in {"none", "one_step_proprioceptive_v1"}:
            raise ValueError("unknown robot feedback producer")
        if self.robot_feedback_mode != "none" and (not self.emit_history_timing or
                self.state_profile != "calvin_relative_7d_v1" or self.executed_action_offsets[-1] != -1):
            raise ValueError("robot feedback requires pre-action CALVIN and exact last recorded command")
        validate_state_feature_profile(self.state_feature_mode, self.state_profile)
        if self.instruction_reference_mode not in {"none", INSTRUCTION_START_REFERENCE}:
            raise ValueError("unknown instruction reference producer")
        if self.instruction_reference_mode != "none" and (
            self.state_profile != "calvin_relative_7d_v1" or not self.emit_history_timing
        ):
            raise ValueError("instruction reference requires declared, aligned CALVIN source times")
        if self.state_feature_mode != NATIVE_AFFINE_STATE and not self.emit_history_timing:
            raise ValueError("state feature chart requires source-timed history")
        if type(self.emit_history_timing) is not bool:
            raise ValueError("emit_history_timing must be boolean")
        if self.emit_history_timing and (
            self.state_offset or self.image_offset or self.action_offset
        ):
            raise ValueError("timestamped history requires an aligned pre-action observation chart")
        if type(self.causal_reset_padding) is not bool:
            raise ValueError("causal_reset_padding must be boolean")
        if self.causal_reset_padding and (self.state_offset or self.image_offset or self.action_offset):
            raise ValueError("reset padding requires aligned zero offsets")
        if self.window_boundary_contract not in {
            STRICT_COMPLETE_V1,
            OBSERVED_TAIL_V1,
            CAUSAL_PREFIX_V1,
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        }:
            raise ValueError(
                f"unknown window boundary contract {self.window_boundary_contract!r}"
            )
        if self.window_boundary_contract == OBSERVED_TAIL_V1 and not self.emit_history_timing:
            raise ValueError("observed-tail windows require explicit history timing")
        if self.window_boundary_contract != STRICT_COMPLETE_V1 and not self.causal_reset_padding:
            raise ValueError("causal boundary contracts require reset-prefix padding")
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
        if self.visual_history_offsets != (-8, -4, 0):
            raise ValueError("the causal visual history is exactly -8/-4/0")
        if not self.executed_action_offsets or max(self.executed_action_offsets) >= 0:
            raise ValueError("executed_action_offsets must contain only past actions")
        if tuple(sorted(set(self.executed_action_offsets))) != self.executed_action_offsets:
            raise ValueError("executed_action_offsets must be strictly increasing")

    @property
    def future_offsets(self) -> tuple[int, ...]:
        return tuple(range(self.support_stride, self.world_horizon + 1, self.support_stride))

    @property
    def minimum_episode_length(self) -> int:
        """Shortest episode that owns at least one complete typed window."""

        min_relative = min(
            min(self.visual_history_offsets) + self.image_offset,
            min(self.state_history_offsets) + self.state_offset,
            min(self.executed_action_offsets) + self.action_offset,
        )
        max_relative = max(
            self.world_horizon + self.action_offset - 1,
            self.world_horizon + self.state_offset,
            max(self.future_offsets) + self.image_offset,
        )
        prefix = 0 if self.causal_reset_padding else max(0, -int(min_relative))
        return prefix + int(max_relative) + 1


@dataclass(frozen=True)
class ObservedWindowRef:
    episode_idx: int
    center: int
    boundary_region: str = "strict"


def _camera_stack(frames: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    return torch.stack([frames[name] for name in names], dim=1)


class ObservedStateWindowDataset(Dataset):
    """Return current observable evidence plus disjoint action/future targets."""

    def __init__(
        self,
        episodes: list[LoadedEpisode],
        episode_ids: list[int],
        *,
        image_store: DecodedImageStore | OnlineVisualStore,
        camera_names: tuple[str, ...],
        state_normalizer: ArrayNormalizer,
        action_normalizer: ArrayNormalizer,
        config: ObservedStateDatasetConfig,
        gripper_transition_boundary: str = "current_action_state",
        allowed_boundary_regions: tuple[str, ...] | None = None,
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
        if gripper_transition_boundary not in {
            "current_action_state",
            "previous_command",
        }:
            raise ValueError(
                "gripper transition boundary must be current_action_state or previous_command"
            )
        self.gripper_transition_boundary = str(gripper_transition_boundary)
        if allowed_boundary_regions is not None:
            selected_regions = tuple(str(value) for value in allowed_boundary_regions)
            if not selected_regions or len(set(selected_regions)) != len(selected_regions):
                raise ValueError("allowed boundary regions must be non-empty and unique")
            unknown_regions = sorted(set(selected_regions) - set(BOUNDARY_REGIONS))
            if unknown_regions:
                raise ValueError(f"unknown allowed boundary regions: {unknown_regions}")
            self.allowed_boundary_regions = selected_regions
        else:
            self.allowed_boundary_regions = None
        self.refs: list[ObservedWindowRef] = []
        self.boundary_plans: dict[int, WindowBoundaryPlan] = {}
        self.instruction_starts: dict[int, int] = {}

        min_rel = min(
            min(config.visual_history_offsets) + config.image_offset,
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
            if config.state_feature_mode != NATIVE_AFFINE_STATE:
                if episode.data_profile != config.state_profile:
                    raise ValueError("state feature profile differs from the projected episode")
                if episode.states_raw is None:
                    raise ValueError("state feature chart requires native observed state")
                state_feature_width(config.state_feature_mode, int(episode.states_raw.shape[-1]))
            if config.instruction_reference_mode == INSTRUCTION_START_REFERENCE:
                if episode.source_start is None or episode.context_start is None:
                    raise ValueError("instruction reference needs declared source_start/context_start")
                start = int(episode.source_start) - int(episode.context_start)
                if not 0 <= start < episode.length:
                    raise ValueError("declared instruction start is outside causal source rows")
                self.instruction_starts[episode_idx] = start
            image_store.validate_episode(episode)
            complete_history_start = max(0, -int(min_rel))
            strict_computed_start = (
                complete_history_start
                if config.window_boundary_contract != STRICT_COMPLETE_V1
                else 0
                if config.causal_reset_padding
                else complete_history_start
            )
            plan = resolve_window_boundary_plan(
                WindowBoundaryMetadata(
                    length=episode.length,
                    valid_center_start=episode.valid_center_start,
                    valid_center_end=episode.valid_center_end,
                    strict_valid_center_start=episode.strict_valid_center_start,
                    strict_valid_center_end=episode.strict_valid_center_end,
                    terminal_state_index=episode.terminal_state_index,
                    source_action_count=episode.source_action_count,
                    terminal_padding_mode=episode.terminal_padding_mode,
                ),
                contract=config.window_boundary_contract,
                strict_computed_start=strict_computed_start,
                complete_future_end=int(
                    (
                        episode.terminal_state_index
                        if config.window_boundary_contract == OBSERVED_TAIL_V1
                        and episode.terminal_state_index is not None
                        else episode.length - 1
                    )
                    - max_rel
                ),
                expected_terminal_padding_mode=(
                    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                    if config.window_boundary_contract == CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
                    else None
                ),
            )
            self.boundary_plans[episode_idx] = plan
            for center, region in plan.iter_centers(
                stride=config.stride,
                allowed_regions=self.allowed_boundary_regions,
            ):
                if episode_idx in self.instruction_starts and center < self.instruction_starts[episode_idx]:
                    raise ValueError("training center precedes its declared instruction")
                self.refs.append(ObservedWindowRef(episode_idx, center, region))
        if not self.refs:
            raise ValueError("mainline dataset has no valid 48-frame windows")

    @property
    def boundary_regions(self) -> np.ndarray:
        return np.asarray([ref.boundary_region for ref in self.refs], dtype=object)

    def boundary_summary(self) -> dict[str, object]:
        region_counts = {name: 0 for name in BOUNDARY_REGIONS}
        centers_by_episode: dict[int, list[int]] = {
            int(index): [] for index in self.episode_ids
        }
        source_action_counts: dict[int, int] = {}
        for episode_idx in self.episode_ids:
            plan = self.boundary_plans[int(episode_idx)]
            source_action_counts[int(episode_idx)] = int(
                getattr(plan, "source_action_count")
            )
        for ref in self.refs:
            region_counts[ref.boundary_region] += 1
            centers_by_episode[int(ref.episode_idx)].append(int(ref.center))
        coverage = policy_action_union_coverage(
            centers_by_episode,
            source_action_counts=source_action_counts,
            policy_horizon=self.config.policy_horizon,
            action_offset=self.config.action_offset,
        )
        summary: dict[str, object] = {
            "schema": "clearvla-window-boundary-summary-v1",
            "contract": self.config.window_boundary_contract,
            "allowed_regions": (
                list(BOUNDARY_REGIONS)
                if self.allowed_boundary_regions is None
                else list(self.allowed_boundary_regions)
            ),
            "windows": len(self.refs),
            "region_window_counts": region_counts,
            **coverage,
        }
        if self.config.window_boundary_contract == OBSERVED_TAIL_V1:
            summary.update(
                {
                    "source_current_action_rows": sum(
                        len(set(centers)) for centers in centers_by_episode.values()
                    ),
                    "supervision_support": "real_source_only",
                    "future_label_contract": OBSERVED_FUTURE_CONTRACT,
                    "eligible_labeled_current_rows": sum(
                        int(self.episodes[index].terminal_state_index or 0)
                        - int(self.episodes[index].valid_center_start or 0)
                        for index in self.episode_ids
                    ),
                }
            )
        if self.config.annotation_goal_mode != "none":
            endpoints = {i: resolve_annotation_endpoint(self.episodes[i]) for i in self.episode_ids}
            statuses: dict[str, int] = {}
            for endpoint in endpoints.values():
                statuses[endpoint.status] = statuses.get(endpoint.status, 0) + 1
            summary["annotation_endpoint_episode_statuses"] = statuses
            labeled_windows = 0
            for ref in self.refs:
                endpoint_index = endpoints[ref.episode_idx].index
                if endpoint_index is not None and ref.center <= endpoint_index:
                    labeled_windows += 1
            summary["annotation_endpoint_labeled_windows"] = labeled_windows
            summary["annotation_endpoint_is_success_label"] = False
        return summary

    def _gripper_transition_boundary_raw(
        self,
        episode: LoadedEpisode,
        *,
        state_index: int,
        action_start: int,
    ) -> np.ndarray:
        """Return the producer-owned boundary for gripper command changes.

        Pen declares current qpos and command to share this boundary. RDT
        declares continuous command-to-command transitions instead: qpos
        remains separately observable and anchors the arm chart, while the
        previous executed command anchors the complete gripper codec path.
        """

        action_states_raw = episode.action_states_raw
        actions_raw = episode.actions_raw
        if action_states_raw is None or actions_raw is None:
            raise ValueError("mainline policy episodes require action-state and action arrays")
        if self.gripper_transition_boundary == "current_action_state":
            value = action_states_raw[state_index]
        else:
            previous_index = int(action_start) - 1
            if previous_index < 0:
                if not self.config.causal_reset_padding or action_start != 0:
                    raise IndexError("previous-command gripper boundary precedes the episode")
                value = action_states_raw[0]
            else:
                value = actions_raw[previous_index]
        result = np.asarray(value, dtype=np.float32)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("gripper transition boundary must be one finite action row")
        return result

    def __len__(self) -> int:
        return len(self.refs)

    def training_information_signals(
        self,
        *,
        gripper_indices: Sequence[int],
        event_threshold: float,
        arm_motion: str = "adjacent_action_delta",
        event_scope: str = "window_any",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute source-semantic sampling strata without image/cache reads."""

        motion = np.empty((len(self.refs),), dtype=np.float32)
        event = np.zeros((len(self.refs),), dtype=bool)
        cfg = self.config
        action_dim = int(self.action_normalizer.minimum.shape[-1])
        grip_indices = []
        for value in gripper_indices:
            grip_index = int(value)
            if grip_index < 0:
                grip_index += action_dim
            if not 0 <= grip_index < action_dim:
                raise ValueError("gripper index is outside the action normalizer dimension")
            grip_indices.append(grip_index)
        if not grip_indices or len(set(grip_indices)) != len(grip_indices):
            raise ValueError("gripper indices must be non-empty and unique")
        if float(event_threshold) < 0.0:
            raise ValueError("event_threshold must be non-negative")
        if arm_motion not in {
            "adjacent_action_delta",
            "relative_command_magnitude",
        }:
            raise ValueError(
                "arm_motion must be adjacent_action_delta or "
                "relative_command_magnitude"
            )
        if event_scope not in {"window_any", "first_action"}:
            raise ValueError("event_scope must be window_any or first_action")
        arm_indices = tuple(
            index for index in range(action_dim) if index not in set(grip_indices)
        )
        if not arm_indices:
            raise ValueError("sampling motion requires at least one arm coordinate")
        normalized_zero = (
            self.action_normalizer.encode(
                np.zeros((1, action_dim), dtype=np.float32)
            )[0]
            if arm_motion == "relative_command_magnitude"
            else None
        )
        for index, ref in enumerate(self.refs):
            episode = self.episodes[ref.episode_idx]
            states_raw = episode.states_raw
            action_states_raw = episode.action_states_raw
            actions_raw = episode.actions_raw
            if states_raw is None or action_states_raw is None or actions_raw is None:
                raise ValueError("mainline policy episodes require state and action arrays")
            action_state_raw = np.asarray(
                action_states_raw[ref.center + cfg.state_offset], dtype=np.float32
            )
            action_start = ref.center + cfg.action_offset
            gripper_boundary_raw = self._gripper_transition_boundary_raw(
                episode,
                state_index=ref.center + cfg.state_offset,
                action_start=action_start,
            )
            stop = action_start + cfg.policy_horizon
            if cfg.window_boundary_contract == OBSERVED_TAIL_V1:
                if episode.terminal_state_index is None:
                    raise ValueError("observed-tail sampler lost terminal provenance")
                stop = min(stop, episode.terminal_state_index)
            action_raw = np.asarray(actions_raw[action_start:stop], dtype=np.float32)
            action = self.action_normalizer.encode(action_raw).astype(np.float32)
            state = self.action_normalizer.encode(action_state_raw).astype(np.float32)
            gripper_boundary = self.action_normalizer.encode(gripper_boundary_raw).astype(
                np.float32
            )
            first_boundary = state.copy()
            first_boundary[grip_indices] = gripper_boundary[grip_indices]
            boundary = np.concatenate((first_boundary[None], action[:-1]), axis=0)
            delta = action - boundary
            if arm_motion == "relative_command_magnitude":
                # CALVIN rel_actions already describe per-step TCP motion.
                # Subtract the normalized representation of raw zero before
                # scoring so a non-zero z-score mean cannot become fake motion.
                assert normalized_zero is not None
                centered_arm = action[:, arm_indices] - normalized_zero[None, arm_indices]
                motion[index] = float(
                    np.sqrt(np.mean(np.square(centered_arm), dtype=np.float64))
                )
            else:
                # Preserve the historical Pen/RDT formula bit-for-bit.
                motion[index] = float(
                    np.sqrt(np.mean(np.square(delta), dtype=np.float64))
                )
            raw_boundary = np.concatenate((gripper_boundary_raw[None], action_raw[:-1]), axis=0)
            gripper_delta = action_raw[:, grip_indices] - raw_boundary[:, grip_indices]
            if event_scope == "first_action":
                event[index] = bool(
                    np.any(np.abs(gripper_delta[0]) >= float(event_threshold))
                )
            else:
                event[index] = bool(np.any(np.abs(gripper_delta) >= float(event_threshold)))
        return motion, event

    def training_release_first_signals(
        self,
        *,
        gripper_indices: Sequence[int],
        event_threshold: float,
        open_direction: int = -1,
    ) -> np.ndarray:
        """Mark windows whose first target row opens the gripper.

        This is a sampler-only signal.  It is intentionally computed from the
        same previous-command boundary used by ``__getitem__`` and never
        enters model inputs or the action normalizer.  LIBERO/robosuite uses
        a negative command delta for opening, hence the default direction.
        Keeping this directional lane separate from the historical
        ``window_any`` event flag prevents close events from consuming the
        release quota in the terminal tail.
        """

        if int(open_direction) not in {-1, 1}:
            raise ValueError("open_direction must be -1 or +1")
        if float(event_threshold) < 0.0:
            raise ValueError("event_threshold must be non-negative")
        action_dim = int(self.action_normalizer.minimum.shape[-1])
        grip_indices: list[int] = []
        for value in gripper_indices:
            index = int(value)
            if index < 0:
                index += action_dim
            if not 0 <= index < action_dim:
                raise ValueError("gripper index is outside the action normalizer dimension")
            grip_indices.append(index)
        if not grip_indices or len(set(grip_indices)) != len(grip_indices):
            raise ValueError("gripper indices must be non-empty and unique")
        result = np.zeros((len(self.refs),), dtype=bool)
        for index, ref in enumerate(self.refs):
            episode = self.episodes[ref.episode_idx]
            actions_raw = episode.actions_raw
            action_states_raw = episode.action_states_raw
            if actions_raw is None or action_states_raw is None:
                raise ValueError("mainline policy episodes require action-state and action arrays")
            action_start = int(ref.center + self.config.action_offset)
            if action_start < 0 or action_start >= int(actions_raw.shape[0]):
                raise IndexError("release-first target row is outside the episode")
            boundary = self._gripper_transition_boundary_raw(
                episode,
                state_index=int(ref.center + self.config.state_offset),
                action_start=action_start,
            )
            delta = (
                np.asarray(actions_raw[action_start, grip_indices], dtype=np.float32)
                - np.asarray(boundary[grip_indices], dtype=np.float32)
            )
            result[index] = bool(
                np.any(float(open_direction) * delta >= float(event_threshold))
            )
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.refs[int(index)]
        episode = self.episodes[ref.episode_idx]
        states_raw = episode.states_raw
        action_states_raw = episode.action_states_raw
        actions_raw = episode.actions_raw
        if states_raw is None or action_states_raw is None or actions_raw is None:
            raise ValueError("mainline policy episodes require state and action arrays")
        cfg = self.config
        center = ref.center
        state_index = center + cfg.state_offset
        action_start = center + cfg.action_offset

        state_raw = np.asarray(states_raw[state_index], dtype=np.float32)
        action_state_raw = np.asarray(action_states_raw[state_index], dtype=np.float32)
        gripper_transition_boundary_raw = self._gripper_transition_boundary_raw(
            episode,
            state_index=state_index,
            action_start=action_start,
        )
        history_state_indices = np.asarray(
            [center + cfg.state_offset + offset for offset in cfg.state_history_offsets],
            dtype=np.int64,
        )
        history_image_indices = np.asarray(
            [center + cfg.image_offset + offset for offset in cfg.visual_history_offsets],
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
        if cfg.causal_reset_padding:
            # Only pre-episode history is padded, exactly like CausalHistory.
            # Never use action[0] as an unexecuted previous command.
            history_state_indices = np.maximum(history_state_indices, 0)
            history_image_indices = np.maximum(history_image_indices, 0)
        history_state_raw = np.asarray(states_raw[history_state_indices], dtype=np.float32)
        if cfg.causal_reset_padding:
            executed_action_raw = np.asarray(actions_raw[np.maximum(executed_indices, 0)], dtype=np.float32).copy()
            executed_action_raw[executed_indices < 0] = action_states_raw[0]
        else:
            executed_action_raw = np.asarray(actions_raw[executed_indices], dtype=np.float32)
        future_support: dict[str, torch.Tensor] = {}
        if cfg.window_boundary_contract == OBSERVED_TAIL_V1:
            if episode.terminal_state_index is None:
                raise ValueError("observed-tail source lost its real terminal observation")
            source = future_source_rows(
                center,
                episode.terminal_state_index,
                world_horizon=cfg.world_horizon,
                future_offsets=cfg.future_offsets,
            )
            future_action_raw = np.asarray(
                actions_raw[source.action_indices], dtype=np.float32
            ).copy()
            future_state_raw = np.asarray(states_raw[source.state_indices], dtype=np.float32).copy()
            # Transport fillers are never targets. Keep neutral finite payloads
            # and carry real-label support independently all the way to losses.
            future_action_raw[~source.action_observed] = 0.0
            future_state_raw[~source.state_observed] = 0.0
            future_image_indices = source.visual_indices
            future_support = {
                name: torch.from_numpy(mask) for name, mask in source.support_mapping().items()
            }
        else:
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
        history_keys = np.stack(
            (
                np.full(len(history_image_indices), ref.episode_idx, dtype=np.int64),
                history_image_indices,
            ),
            axis=1,
        )
        future_keys = np.stack(
            (
                np.full(len(future_image_indices), ref.episode_idx, dtype=np.int64),
                future_image_indices,
            ),
            axis=1,
        )
        frame_progress = float(center + cfg.image_offset) / float(max(int(episode.length) - 1, 1))
        if cfg.window_boundary_contract == OBSERVED_TAIL_V1:
            # Audit-only progress uses the real labelled segment, not appended
            # storage. The terminal observation is progress 1, never a command.
            label_start = int(episode.valid_center_start or 0)
            label_end = int(episode.terminal_state_index or 0)
            frame_progress = float(center - label_start) / max(label_end - label_start, 1)
        future_state_features = encode_state_features(
            future_state_raw, self.state_normalizer,
            mode=cfg.state_feature_mode, profile=cfg.state_profile,
        )
        if cfg.window_boundary_contract == OBSERVED_TAIL_V1:
            # A zero native Euler filler encodes a real identity rotation. It
            # is still missing data, so quarantine AFTER the nonlinear chart
            # as well as before it. Never export invented terminal targets.
            future_state_features[~future_support["future_state_observed"].numpy()] = 0.0
        step_fields: dict[str, torch.Tensor] = {}
        if cfg.robot_feedback_mode != "none":
            # Prefix rows in the raw overlay are actual causal observations;
            # instruction boundary is not an environment reset boundary.
            available = state_index > 0 and (
                episode.terminal_state_index is None or state_index <= episode.terminal_state_index
            )
            previous = encode_state_features(
                np.asarray(states_raw[max(state_index - 1, 0)], dtype=np.float32), self.state_normalizer,
                mode=cfg.state_feature_mode, profile=cfg.state_profile,
            )
            command = self.action_normalizer.encode(np.asarray(actions_raw[max(action_start - 1, 0)], dtype=np.float32))
            step_fields = {
                "robot_step_previous_state": torch.from_numpy(previous if available else np.zeros_like(previous)),
                "robot_step_command": torch.from_numpy(command if available else np.zeros_like(command)),
                "robot_step_observed": torch.tensor(available, dtype=torch.bool),
                "robot_step_offsets": torch.tensor([-1, -1, 0] if available else [0, 0, 0], dtype=torch.long),
            }
        world_fields: dict[str,torch.Tensor]={}
        if cfg.world_feedback_mode!="none":
            available=center>=4 and (episode.terminal_state_index is None or center<=episode.terminal_state_index)
            anchor=max(center-4,0)
            indices=np.maximum(anchor+np.asarray((-8,-4,0),dtype=np.int64),0)
            oldest=self.image_store.load_window(episode,indices[:1])
            old_rgb=torch.cat((_camera_stack(oldest,self.camera_names).float()/255.,history_rgb[:2]),0)
            old_state=encode_state_features(np.asarray(states_raw[anchor],dtype=np.float32),self.state_normalizer,
                mode=cfg.state_feature_mode,profile=cfg.state_profile)
            boundary=self.action_normalizer.encode(np.asarray(action_states_raw[anchor],dtype=np.float32))
            commands=(self.action_normalizer.encode(np.asarray(actions_raw[center-4:center],dtype=np.float32))
                      if available else np.zeros((4,actions_raw.shape[-1]),dtype=np.float32))
            world_fields={
                "executed_world_keys":torch.from_numpy(np.stack((np.full(3,ref.episode_idx,dtype=np.int64),indices),1)),
                "executed_world_rgb":old_rgb if available else torch.zeros_like(old_rgb),
                "executed_world_state":torch.from_numpy(old_state if available else np.zeros_like(old_state)),
                "executed_world_action_state":torch.from_numpy(boundary if available else np.zeros_like(boundary)),
                "executed_world_commands":torch.from_numpy(commands),
                "executed_world_observed":torch.tensor(available,dtype=torch.bool),
                "executed_world_offsets":torch.from_numpy(sparse_history_clock(anchor)["history_state_offsets"]),
            }
        reference_fields: dict[str, torch.Tensor] = {}
        if cfg.instruction_reference_mode == INSTRUCTION_START_REFERENCE:
            start = self.instruction_starts[ref.episode_idx]
            reference_fields = {
                "instruction_reference_key": torch.tensor([[ref.episode_idx, start]], dtype=torch.long),
                "instruction_reference_age": torch.tensor(center - start, dtype=torch.long),
                "instruction_reference_state": torch.from_numpy(encode_state_features(
                    np.asarray(states_raw[start], dtype=np.float32), self.state_normalizer,
                    mode=cfg.state_feature_mode, profile=cfg.state_profile,
                )),
            }
        endpoint_fields: dict[str, torch.Tensor] = {}
        if cfg.annotation_goal_mode != "none":
            source = resolve_annotation_endpoint(episode)
            idx = source.index
            available = idx is not None and center <= idx
            # Unknown endpoint still uses a safe observed row for transport;
            # its mask is false and no endpoint label is fabricated.
            endpoint_index = center if idx is None else idx
            fields = [-1] * 5
            if available:
                assert episode.source_annotation_index is not None
                assert episode.context_start is not None and episode.source_start is not None and episode.source_end is not None
                fields = [episode.source_annotation_index, episode.context_start,
                          episode.source_start, episode.source_end, center]
            endpoint_state = encode_state_features(
                np.asarray(states_raw[endpoint_index], dtype=np.float32), self.state_normalizer,
                mode=cfg.state_feature_mode, profile=cfg.state_profile,
            ) if available else np.zeros_like(future_state_features[0])
            endpoint_fields = {
                "annotation_endpoint_key": torch.tensor([[ref.episode_idx, endpoint_index]], dtype=torch.long),
                "annotation_endpoint_state": torch.from_numpy(endpoint_state),
                "annotation_endpoint_declared": torch.tensor(available, dtype=torch.bool),
                "annotation_endpoint_state_observed": torch.tensor(available, dtype=torch.bool),
                "annotation_endpoint_source_indices": torch.tensor(fields, dtype=torch.long),
                "annotation_endpoint_offset_steps": torch.tensor(endpoint_index - center if available else 0, dtype=torch.long),
            }
        return {
            **endpoint_fields,
            **world_fields,
            **reference_fields,
            **step_fields,
            **future_support,
            **(
                {
                    name: torch.from_numpy(value)
                    for name, value in sparse_history_clock(
                        center,
                        state_offsets=cfg.state_history_offsets,
                        action_offsets=cfg.executed_action_offsets,
                    ).items()
                }
                if cfg.emit_history_timing
                else {}
            ),
            "sample_index": torch.tensor(int(index), dtype=torch.long),
            "episode_idx": torch.tensor(ref.episode_idx, dtype=torch.long),
            "center_index": torch.tensor(center, dtype=torch.long),
            "boundary_region": torch.tensor(
                BOUNDARY_REGION_TO_INDEX[ref.boundary_region], dtype=torch.long
            ),
            "frame_progress": torch.tensor(
                frame_progress,
                dtype=torch.float32,
            ),
            "state": torch.from_numpy(encode_state_features(state_raw, self.state_normalizer,
                mode=cfg.state_feature_mode, profile=cfg.state_profile)),
            "state_raw": torch.from_numpy(state_raw.copy()),
            "action_state": torch.from_numpy(self.action_normalizer.encode(action_state_raw)),
            "action_state_raw": torch.from_numpy(action_state_raw.copy()),
            "gripper_transition_boundary": torch.from_numpy(
                self.action_normalizer.encode(gripper_transition_boundary_raw)
            ),
            "gripper_transition_boundary_raw": torch.from_numpy(
                gripper_transition_boundary_raw.copy()
            ),
            "history_state": torch.from_numpy(encode_state_features(history_state_raw, self.state_normalizer,
                mode=cfg.state_feature_mode, profile=cfg.state_profile)),
            "executed_action_history": torch.from_numpy(
                self.action_normalizer.encode(executed_action_raw)
            ),
            "action": torch.from_numpy(self.action_normalizer.encode(future_action_raw)),
            "policy_action": torch.from_numpy(
                self.action_normalizer.encode(future_action_raw[: cfg.policy_horizon])
            ),
            "policy_action_raw": torch.from_numpy(future_action_raw[: cfg.policy_horizon].copy()),
            "future_state": torch.from_numpy(future_state_features),
            "future_offsets": torch.tensor(cfg.future_offsets, dtype=torch.long),
            "history_obs_image": history_rgb,
            "history_keys": torch.from_numpy(history_keys),
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

    @property
    def boundary_regions(self) -> np.ndarray:
        return self.base.boundary_regions

    def boundary_summary(self) -> dict[str, object]:
        return self.base.boundary_summary()

    def training_information_signals(
        self,
        *,
        gripper_indices: Sequence[int],
        event_threshold: float,
        arm_motion: str = "adjacent_action_delta",
        event_scope: str = "window_any",
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base.training_information_signals(
            gripper_indices=gripper_indices,
            event_threshold=event_threshold,
            arm_motion=arm_motion,
            event_scope=event_scope,
        )

    def training_release_first_signals(
        self,
        *,
        gripper_indices: Sequence[int],
        event_threshold: float,
        open_direction: int = -1,
    ) -> np.ndarray:
        return self.base.training_release_first_signals(
            gripper_indices=gripper_indices,
            event_threshold=event_threshold,
            open_direction=open_direction,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[int(index)]
        history_keys = sample.pop("history_keys")
        future_keys = sample.pop("future_keys")
        # Current and future keys normally hit the same episode mmap.  One
        # grouped read avoids allocating and dispatching two independent
        # result buffers in every DataLoader sample while preserving the
        # exact current/future ownership split in the returned mapping.
        reference_key = sample.pop("instruction_reference_key", None)
        executed_keys=sample.pop("executed_world_keys",None)
        endpoint_key = sample.pop("annotation_endpoint_key", None)
        keys=[history_keys,future_keys]
        if reference_key is not None:
            keys.append(reference_key)
        if executed_keys is not None:
            keys.append(executed_keys)
        if endpoint_key is not None:
            keys.append(endpoint_key)
        token_rows=self.token_store.load_batch(torch.cat(keys,dim=0))
        history_rows = int(history_keys.shape[0])
        future_end = history_rows + int(future_keys.shape[0])
        sample["history_dinov2_tokens"] = token_rows[:history_rows]
        sample["target_future_dinov2_tokens"] = token_rows[history_rows:future_end]
        if reference_key is not None:
            sample["instruction_reference_dino"] = token_rows[future_end]
            sample["instruction_reference_observed"] = torch.ones(token_rows[future_end].shape[:-1], dtype=torch.bool)
        if executed_keys is not None:
            first=future_end+(0 if reference_key is None else int(reference_key.shape[0]))
            sample["executed_world_dino"]=token_rows[first:first+int(executed_keys.shape[0])]
            if not bool(sample["executed_world_observed"]):
                sample["executed_world_dino"]=torch.zeros_like(sample["executed_world_dino"])
        if endpoint_key is not None:
            endpoint_tokens = token_rows[-1]
            support = torch.full(endpoint_tokens.shape[:-1], bool(sample["annotation_endpoint_declared"]), dtype=torch.bool)
            sample["annotation_endpoint_dino"] = torch.where(support[..., None], endpoint_tokens, 0.0)
            sample["annotation_endpoint_visual_observed"] = support
        sample["target_future_offsets"] = sample.pop("future_offsets")
        return sample


__all__ = [
    "CachedTokenPolicyWindowDataset",
    "ObservedStateDatasetConfig",
    "ObservedStateWindowDataset",
    "ObservedWindowRef",
]
