"""Typed external boundaries for the capability-named ClearVLA mainline.

The old runtime passed one mutable ``dict[str, Tensor]`` through data loading,
teacher construction, training and deployment.  A boolean then decided
whether future tensors were legal.  Here deployment input and future
supervision are different Python types: an online model cannot receive a
teacher by accident because :class:`OnlinePolicyInput` has no future field.

These validators intentionally check metadata only (shape, dtype and device
agreement).  They do not reduce tensor values, so calling them on CUDA tensors
does not add synchronization to the five-step deployment path.  Finite-value
audits belong to the training/preflight boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ExperimentConfig


def _floating(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}")


def _shape(value: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must be {expected}, got {tuple(value.shape)}")


def _batch(value: Tensor, expected: int, name: str) -> None:
    if value.ndim == 0 or int(value.shape[0]) != expected:
        raise ValueError(f"{name} must have batch {expected}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class GoalCondition:
    """One explicit precomputed T5 condition selected for each batch row."""

    tokens: Tensor  # [B,L,D_goal]
    mask: Tensor  # bool [B,L]

    @property
    def batch(self) -> int:
        return int(self.tokens.shape[0])

    def validate(self, config: ExperimentConfig) -> None:
        dims = config.dimensions
        if self.tokens.ndim != 3:
            raise ValueError("goal tokens must be [B,L,D]")
        batch, length, width = self.tokens.shape
        if not 1 <= int(length) <= int(dims.goal_max_tokens):
            raise ValueError("goal token length is outside the configured bound")
        if int(width) != int(dims.goal_token_dim):
            raise ValueError("goal token width does not match goal_token_dim")
        _shape(self.mask, (batch, length), "goal mask")
        if self.mask.dtype != torch.bool:
            raise TypeError("goal mask must be boolean")
        _floating(self.tokens, "goal tokens")
        if self.tokens.device != self.mask.device:
            raise ValueError("goal tokens and mask must share a device")


@dataclass(frozen=True)
class CurrentObservation:
    """Causal observation history; no future support is representable."""

    dino_history: Tensor  # causal [-8,-4,0], [B,3,C,P,D_visual]
    raw_rgb: Tensor  # causal [-8,-4,0], [B,3,C,3,R,R], normalized float RGB

    @property
    def batch(self) -> int:
        return int(self.dino_history.shape[0])

    @property
    def raw_side(self) -> int:
        return int(self.raw_rgb.shape[-1])

    def validate(self, config: ExperimentConfig) -> None:
        dims = config.dimensions
        if self.dino_history.ndim != 5:
            raise ValueError("causal DINO history must be [B,H,C,P,D]")
        batch = self.batch
        _shape(
            self.dino_history,
            (
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
            ),
            "causal DINO history",
        )
        if self.raw_rgb.ndim != 6:
            raise ValueError("current raw RGB must be [B,H,C,3,R,R]")
        if tuple(self.raw_rgb.shape[:4]) != (
            batch,
            dims.visual_history_length,
            dims.num_cameras,
            3,
        ):
            raise ValueError("raw RGB must retain the causal -8/-4/0 camera history")
        if int(self.raw_rgb.shape[-2]) != self.raw_side:
            raise ValueError("current raw RGB must be square")
        if self.raw_side < 32 or self.raw_side % 16:
            raise ValueError("current raw RGB side must be >=32 and divisible by 16")
        _floating(self.dino_history, "causal DINO history")
        _floating(self.raw_rgb, "current raw RGB")
        if self.dino_history.device != self.raw_rgb.device:
            raise ValueError("causal DINO and raw RGB must share a device")


@dataclass(frozen=True)
class ObservableHistory:
    """Causal proprioception and already-executed actions available online."""

    state: Tensor  # [B,S]
    action_state: Tensor  # [B,A], current state in the action chart
    state_history: Tensor  # [B,Hs,S]
    executed_action_history: Tensor  # [B,Ha,A]

    @property
    def batch(self) -> int:
        return int(self.state.shape[0])

    def validate(self, config: ExperimentConfig) -> None:
        dims = config.dimensions
        batch = self.batch
        _shape(self.state, (batch, dims.state_dim), "current state")
        _shape(self.action_state, (batch, dims.action_dim), "current action-state")
        _shape(
            self.state_history,
            (batch, dims.state_history_length, dims.state_dim),
            "state history",
        )
        _shape(
            self.executed_action_history,
            (batch, dims.executed_history_length, dims.action_dim),
            "executed action history",
        )
        for name in (
            "state",
            "action_state",
            "state_history",
            "executed_action_history",
        ):
            _floating(getattr(self, name), name.replace("_", " "))
        devices = {
            value.device
            for value in (
                self.state,
                self.action_state,
                self.state_history,
                self.executed_action_history,
            )
        }
        if len(devices) != 1:
            raise ValueError("observable history tensors must share a device")


@dataclass(frozen=True)
class OnlinePolicyInput:
    """The complete deployment input.

    Deliberately absent: target action, future state/action, future offsets and
    future DINO supports.  Teacher isolation is therefore an API property, not
    a runtime flag.
    """

    observation: CurrentObservation
    history: ObservableHistory
    goal: GoalCondition

    @property
    def batch(self) -> int:
        return self.observation.batch

    @property
    def device(self) -> torch.device:
        return self.observation.dino_history.device

    def validate(self, config: ExperimentConfig) -> None:
        self.observation.validate(config)
        self.history.validate(config)
        self.goal.validate(config)
        if self.history.batch != self.batch or self.goal.batch != self.batch:
            raise ValueError("online input components must share a batch size")
        if not (
            self.observation.dino_history.device
            == self.history.state.device
            == self.goal.tokens.device
        ):
            raise ValueError("online input components must share a device")


@dataclass(frozen=True)
class FutureSupervision:
    """Training-only future evidence, never accepted by the online model."""

    dino_supports: Tensor  # cached float16 [B,F,C,P,D_visual]
    action_sequence: Tensor  # float32 [B,Tw,A]
    state_sequence: Tensor  # float32 [B,Tw,S]
    offsets: Tensor  # int64 [B,F]

    @property
    def batch(self) -> int:
        return int(self.dino_supports.shape[0])

    @property
    def supports(self) -> int:
        return int(self.dino_supports.shape[1])

    def validate(self, config: ExperimentConfig) -> None:
        dims = config.dimensions
        if self.dino_supports.ndim != 5:
            raise ValueError("future DINO supports must be [B,F,C,P,D]")
        batch, supports = self.dino_supports.shape[:2]
        if int(supports) != int(dims.future_supports):
            raise ValueError("future DINO support count does not match the config")
        if tuple(self.dino_supports.shape[2:]) != (
            dims.num_cameras,
            dims.patches_per_camera,
            dims.visual_token_dim,
        ):
            raise ValueError("future DINO camera/patch/width contract is invalid")
        if self.action_sequence.ndim != 3 or tuple(self.action_sequence.shape[:1]) != (batch,):
            raise ValueError("future action sequence must be [B,Tw,A]")
        world_horizon = int(self.action_sequence.shape[1])
        if world_horizon < 48 or int(self.action_sequence.shape[-1]) != dims.action_dim:
            raise ValueError("future action sequence must cover at least 48 action steps")
        _shape(
            self.state_sequence,
            (batch, world_horizon, dims.state_dim),
            "future state sequence",
        )
        _shape(self.offsets, (batch, supports), "future support offsets")
        if self.offsets.dtype != torch.long:
            raise TypeError("future support offsets must be int64")
        if self.dino_supports.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError("future DINO supports must use a floating cache dtype")
        for name in ("action_sequence", "state_sequence"):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"future supervision {name} must be float32")
        devices = {
            value.device
            for value in (
                self.dino_supports,
                self.action_sequence,
                self.state_sequence,
                self.offsets,
            )
        }
        if len(devices) != 1:
            raise ValueError("future supervision tensors must share a device")


@dataclass(frozen=True)
class ActionSupervision:
    """The policy-horizon action target kept separate from online evidence."""

    normalized: Tensor  # float32 [B,T,A]
    raw_units: Tensor  # float32 [B,T,A], native dataset units
    # Observed qpos projected into the same raw command chart as ``raw_units``.
    # It is byte-equivalent to native state only when the selected data
    # profile declares the qpos and command coordinates identical.
    current_raw_units: Tensor  # float32 [B,A], current raw action-chart state
    # Dataset-profile-owned first boundary for gripper command transitions.
    # Pen uses current action-state; RDT uses the previous executed command.
    gripper_transition_boundary: Tensor  # float32 [B,A], normalized action chart
    gripper_transition_boundary_raw_units: Tensor  # float32 [B,A], raw action chart

    @property
    def batch(self) -> int:
        return int(self.normalized.shape[0])

    def validate(self, config: ExperimentConfig) -> None:
        dims = config.dimensions
        _shape(
            self.normalized,
            (self.batch, dims.action_horizon, dims.action_dim),
            "normalized action target",
        )
        if self.normalized.dtype != torch.float32:
            raise TypeError("normalized action target must be float32")
        _shape(self.raw_units, tuple(self.normalized.shape), "raw-unit action target")
        _shape(
            self.current_raw_units,
            (self.batch, dims.action_dim),
            "current raw-unit action state",
        )
        _shape(
            self.gripper_transition_boundary,
            (self.batch, dims.action_dim),
            "normalized gripper transition boundary",
        )
        _shape(
            self.gripper_transition_boundary_raw_units,
            (self.batch, dims.action_dim),
            "raw-unit gripper transition boundary",
        )
        for name in (
            "raw_units",
            "current_raw_units",
            "gripper_transition_boundary",
            "gripper_transition_boundary_raw_units",
        ):
            if getattr(self, name).dtype != torch.float32:
                raise TypeError(f"action supervision {name} must be float32")
        devices = {
            self.normalized.device,
            self.raw_units.device,
            self.current_raw_units.device,
            self.gripper_transition_boundary.device,
            self.gripper_transition_boundary_raw_units.device,
        }
        if len(devices) != 1:
            raise ValueError("normalized and raw-unit action supervision must share a device")


@dataclass(frozen=True)
class AuditMetadata:
    """Detached dataset metadata that is never an argument to model forward."""

    sample_index: Tensor | None = None
    episode_index: Tensor | None = None
    frame_progress: Tensor | None = None

    def validate(self, batch: int) -> None:
        for name in ("sample_index", "episode_index", "frame_progress"):
            value = getattr(self, name)
            if value is None:
                continue
            _shape(value, (batch,), f"audit {name}")
            if value.device.type != "cpu":
                raise ValueError(f"audit {name} must stay on CPU")
            if value.requires_grad:
                raise ValueError(f"audit {name} cannot require gradients")
        if self.frame_progress is not None and self.frame_progress.dtype != torch.float32:
            raise TypeError("audit frame_progress must be float32")


@dataclass(frozen=True)
class TrainingBatch:
    """Training engine input; only the engine can see all three partitions."""

    online: OnlinePolicyInput
    action_target: ActionSupervision
    future: FutureSupervision
    audit: AuditMetadata = AuditMetadata()

    def validate(self, config: ExperimentConfig) -> None:
        self.online.validate(config)
        self.action_target.validate(config)
        self.future.validate(config)
        if not (self.online.batch == self.action_target.batch == self.future.batch):
            raise ValueError("training batch partitions must share a batch size")
        if not (
            self.online.device
            == self.action_target.normalized.device
            == self.future.dino_supports.device
        ):
            raise ValueError("training batch partitions must share a device")
        self.audit.validate(self.online.batch)


__all__ = [
    "ActionSupervision",
    "AuditMetadata",
    "CurrentObservation",
    "FutureSupervision",
    "GoalCondition",
    "ObservableHistory",
    "OnlinePolicyInput",
    "TrainingBatch",
]
