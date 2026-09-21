"""Typed simulator/policy boundary with evaluator information kept separate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

ACTION_DIM = 7
STATE_DIM = 7
CAMERA_NAMES = ("top", "wrist")


def _finite_vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (int(size),):
        raise ValueError(f"{name} must have shape [{size}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _rgb(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError(f"{name} must be one HWC RGB image, got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {array.dtype}")
    return array


@dataclass(frozen=True)
class EnvironmentDescriptor:
    """Serializable semantics of one concrete simulator configuration."""

    backend: str
    benchmark: str
    task: str
    robot: str
    simulator_version: str
    state_semantics: str
    action_semantics: str
    action_state_semantics: str
    camera_semantics: Mapping[str, str]
    control_hz: float
    max_episode_steps: int
    kinematic_proxy: bool = False
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        for name in (
            "backend",
            "benchmark",
            "task",
            "robot",
            "simulator_version",
            "state_semantics",
            "action_semantics",
            "action_state_semantics",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"environment descriptor {name} must be non-empty")
        if tuple(self.camera_semantics) != CAMERA_NAMES:
            raise ValueError(
                f"policy cameras must be ordered exactly as {CAMERA_NAMES}, "
                f"got {tuple(self.camera_semantics)}"
            )
        if any(not str(value).strip() for value in self.camera_semantics.values()):
            raise ValueError("camera semantics must be explicit")
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["camera_semantics"] = dict(self.camera_semantics)
        value["notes"] = list(self.notes)
        return value


@dataclass(frozen=True)
class PolicyObservation:
    """Only evidence that may cross into an online policy call."""

    rgb: Mapping[str, np.ndarray]
    state: np.ndarray
    action_state: np.ndarray

    def validate(self) -> None:
        if tuple(self.rgb) != CAMERA_NAMES:
            raise ValueError(
                f"policy RGB cameras must be ordered as {CAMERA_NAMES}, got {tuple(self.rgb)}"
            )
        for camera in CAMERA_NAMES:
            _rgb(self.rgb[camera], name=f"{camera} RGB")
        # Camera-native spatial shapes are part of the evidence boundary.
        # They are deliberately allowed to differ; every stream is resized
        # exactly once by the checkpoint-owned preprocessing contract.
        _finite_vector(self.state, size=STATE_DIM, name="policy state")
        _finite_vector(
            self.action_state,
            size=ACTION_DIM,
            name="policy action-state",
        )

    def copied(self) -> "PolicyObservation":
        self.validate()
        return PolicyObservation(
            rgb={name: np.asarray(self.rgb[name]).copy() for name in CAMERA_NAMES},
            state=np.asarray(self.state, dtype=np.float32).copy(),
            action_state=np.asarray(self.action_state, dtype=np.float32).copy(),
        )


@dataclass(frozen=True)
class EvaluationState:
    """Privileged simulator facts that must never be passed to the policy."""

    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResetResult:
    observation: PolicyObservation
    evaluation: EvaluationState = EvaluationState()


@dataclass(frozen=True)
class StepResult:
    observation: PolicyObservation
    reward: float
    terminated: bool
    truncated: bool
    evaluation: EvaluationState = EvaluationState()

    def validate(self) -> None:
        self.observation.validate()
        if not np.isfinite(self.reward):
            raise ValueError("environment reward must be finite")


@runtime_checkable
class SimulationEnvironment(Protocol):
    descriptor: EnvironmentDescriptor

    def reset(self, *, seed: int) -> ResetResult: ...

    def step(self, action: np.ndarray) -> StepResult: ...

    def clip_action(self, action: np.ndarray) -> np.ndarray: ...

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    def hold_action(self, observation: PolicyObservation) -> np.ndarray: ...

    def sample_action(self, rng: np.random.Generator) -> np.ndarray: ...

    def close(self) -> None: ...


__all__ = [
    "ACTION_DIM",
    "CAMERA_NAMES",
    "STATE_DIM",
    "EnvironmentDescriptor",
    "EvaluationState",
    "PolicyObservation",
    "ResetResult",
    "SimulationEnvironment",
    "StepResult",
]

