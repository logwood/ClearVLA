"""Exact online counterpart of the mainline sparse causal history windows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from clearvla.data.history_clock import (
    EXECUTED_ACTION_OFFSETS,
    STATE_OFFSETS,
    VISUAL_OFFSETS,
    sparse_history_clock,
)

from .contracts import ACTION_DIM, CAMERA_NAMES, STATE_DIM, PolicyObservation


@dataclass(frozen=True)
class HistorySnapshot:
    """One deployment input before normalization and DINO encoding."""

    time_index: int
    rgb_history: Mapping[str, np.ndarray]  # camera -> [3,H,W,3], uint8
    state: np.ndarray  # [7]
    action_state: np.ndarray  # [7]
    state_history: np.ndarray  # [3,7]
    executed_action_history: np.ndarray  # [8,7]

    def timing_arrays(self) -> dict[str, np.ndarray]:
        return sparse_history_clock(self.time_index)

    def validate(self) -> None:
        if self.time_index < 0:
            raise ValueError("history time index must be non-negative")
        if tuple(self.rgb_history) != CAMERA_NAMES:
            raise ValueError("RGB history camera order must be top,wrist")
        for camera in CAMERA_NAMES:
            value = np.asarray(self.rgb_history[camera])
            if (
                value.ndim != 4
                or value.shape[0] != len(VISUAL_OFFSETS)
                or value.shape[-1] != 3
                or min(value.shape[1:3]) <= 0
                or value.dtype != np.uint8
            ):
                raise ValueError(
                    f"RGB history {camera} must be uint8 [3,H,W,3], got {value.shape}"
                )
        expected = {
            "state": (STATE_DIM,),
            "action_state": (ACTION_DIM,),
            "state_history": (len(STATE_OFFSETS), STATE_DIM),
            "executed_action_history": (
                len(EXECUTED_ACTION_OFFSETS),
                ACTION_DIM,
            ),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"history {name} must be finite with shape {shape}")


class CausalHistory:
    """Append-only observation/action timeline with explicit reset padding.

    At time zero the model has no pre-episode actions.  Those unavailable rows
    are padded with the environment's safe reset action-state, while visual and
    proprioceptive rows repeat the reset observation.  No future or evaluator
    state is retained here.
    """

    def __init__(self) -> None:
        self._observations: deque[PolicyObservation] = deque(
            maxlen=1 - min(VISUAL_OFFSETS + STATE_OFFSETS)
        )
        self._actions: deque[np.ndarray] = deque(maxlen=-min(EXECUTED_ACTION_OFFSETS))
        self._time_index = 0
        self._reset_observation: PolicyObservation | None = None
        self._reset_action: np.ndarray | None = None

    @property
    def time_index(self) -> int:
        if not self._observations:
            raise RuntimeError("causal history has not been reset")
        return self._time_index

    def reset(
        self,
        observation: PolicyObservation,
        *,
        reset_action: np.ndarray | None = None,
    ) -> None:
        copied = observation.copied()
        boundary = copied.action_state if reset_action is None else reset_action
        boundary = np.asarray(boundary, dtype=np.float32)
        if boundary.shape != (ACTION_DIM,) or not np.isfinite(boundary).all():
            raise ValueError("reset action must be one finite [7] vector")
        self._observations.clear()
        self._observations.append(copied)
        self._actions.clear()
        self._time_index = 0
        self._reset_observation = copied
        self._reset_action = boundary.copy()

    def append(self, executed_action: np.ndarray, observation: PolicyObservation) -> None:
        if not self._observations or self._reset_action is None:
            raise RuntimeError("causal history must be reset before append")
        action = np.asarray(executed_action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("executed action must be one finite [7] vector")
        copied = observation.copied()
        self._actions.append(action.copy())
        self._observations.append(copied)
        self._time_index += 1

    def _observation_at(self, index: int) -> PolicyObservation:
        if index < 0:
            assert self._reset_observation is not None
            return self._reset_observation
        first = self._time_index - len(self._observations) + 1
        if not first <= index <= self._time_index:
            raise IndexError("requested observation is outside retained physical history")
        return self._observations[index - first]

    def _action_at(self, index: int) -> np.ndarray:
        assert self._reset_action is not None
        if index < 0:
            return self._reset_action
        first = self._time_index - len(self._actions)
        if not first <= index < self._time_index:
            raise IndexError("requested action is unexecuted or outside retained physical history")
        return self._actions[index - first]

    def snapshot(self) -> HistorySnapshot:
        if not self._observations or self._reset_action is None:
            raise RuntimeError("causal history has not been reset")
        now = self.time_index
        rgb = {
            camera: np.stack(
                [self._observation_at(now + offset).rgb[camera] for offset in VISUAL_OFFSETS],
                axis=0,
            )
            for camera in CAMERA_NAMES
        }
        state_history = np.stack(
            [self._observation_at(now + offset).state for offset in STATE_OFFSETS],
            axis=0,
        ).astype(np.float32)
        executed = np.stack(
            [self._action_at(now + offset) for offset in EXECUTED_ACTION_OFFSETS],
            axis=0,
        ).astype(np.float32)
        current = self._observations[-1]
        result = HistorySnapshot(
            time_index=now,
            rgb_history=rgb,
            state=np.asarray(current.state, dtype=np.float32).copy(),
            action_state=np.asarray(current.action_state, dtype=np.float32).copy(),
            state_history=state_history,
            executed_action_history=executed,
        )
        result.validate()
        return result


__all__ = [
    "EXECUTED_ACTION_OFFSETS",
    "STATE_OFFSETS",
    "VISUAL_OFFSETS",
    "CausalHistory",
    "HistorySnapshot",
]

