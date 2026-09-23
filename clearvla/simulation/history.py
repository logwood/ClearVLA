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
class ExecutedWorldSnapshot:
    """Older physical observation and acknowledged controls, not proposals."""
    anchor_index: int
    rgb_history: Mapping[str,np.ndarray]
    state: np.ndarray
    action_state: np.ndarray
    commands: np.ndarray
    visual_offsets: np.ndarray
    observed: bool

    def validate(self, *, now: int) -> None:
        if type(self.observed) is not bool or type(self.anchor_index) is not int or self.anchor_index!=max(now-4,0):
            raise ValueError("executed world source has invalid anchor")
        if self.observed and now<4:
            raise ValueError("executed world cannot observe pre-reset controls")
        if not np.array_equal(self.visual_offsets,sparse_history_clock(self.anchor_index)["history_state_offsets"]):
            raise ValueError("executed world source clock differs")
        if tuple(self.rgb_history)!=CAMERA_NAMES:
            raise ValueError("executed world source camera identity differs")
        for camera in CAMERA_NAMES:
            image=self.rgb_history[camera]
            if image.ndim!=4 or image.shape[0]!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
                raise ValueError("executed world needs three raw RGB observations")
        for value,shape in ((self.state,(STATE_DIM,)),(self.action_state,(ACTION_DIM,)),(self.commands,(4,ACTION_DIM))):
            if value.shape!=shape or (self.observed and not np.isfinite(value).all()):
                raise ValueError("executed world state/commands malformed")

@dataclass(frozen=True)
class HistorySnapshot:
    """One deployment input before normalization and DINO encoding."""

    time_index: int
    rgb_history: Mapping[str, np.ndarray]  # camera -> [3,H,W,3], uint8
    state: np.ndarray  # [7]
    action_state: np.ndarray  # [7]
    state_history: np.ndarray  # [3,7]
    executed_action_history: np.ndarray  # [8,7]
    previous_state: np.ndarray | None = None  # exact o[t-1], not sparse -4/-8
    executed_world: ExecutedWorldSnapshot | None = None

    def timing_arrays(self) -> dict[str, np.ndarray]:
        return sparse_history_clock(self.time_index)

    def validate(self) -> None:
        if self.executed_world is not None:
            self.executed_world.validate(now=self.time_index)
            w=self.executed_world
            if w.observed:
                if not np.array_equal(w.state,self.state_history[1]):
                    raise ValueError("executed world state disagrees with current history")
                for camera in CAMERA_NAMES:
                    if not np.array_equal(w.rgb_history[camera][1:],self.rgb_history[camera][:2]):
                        raise ValueError("executed world images disagree with current history")
                for j,offset in enumerate((-4,-3,-2,-1)):
                    if offset in EXECUTED_ACTION_OFFSETS and not np.array_equal(w.commands[j],self.executed_action_history[EXECUTED_ACTION_OFFSETS.index(offset)]):
                        raise ValueError("executed world command differs from confirmed history")
        if self.previous_state is not None:
            if self.time_index == 0 or self.previous_state.shape != (STATE_DIM,) or not np.isfinite(self.previous_state).all():
                raise ValueError("previous robot state requires a real one-step predecessor")
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

    def __init__(self, *, executed_world: bool = False) -> None:
        if type(executed_world) is not bool:
            raise TypeError("executed world history selection must be boolean")
        self.executed_world=executed_world
        self._observations: deque[PolicyObservation] = deque(
            maxlen=13 if executed_world else 1-min(VISUAL_OFFSETS+STATE_OFFSETS)
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
        world=None
        if self.executed_world:
            anchor=max(now-4,0)
            old=self._observation_at(anchor)
            world=ExecutedWorldSnapshot(anchor_index=anchor,
                rgb_history={camera:np.stack([self._observation_at(anchor+offset).rgb[camera] for offset in VISUAL_OFFSETS],0) for camera in CAMERA_NAMES},
                state=np.asarray(old.state,dtype=np.float32).copy(),
                action_state=np.asarray(old.action_state,dtype=np.float32).copy(),
                commands=(np.stack([self._action_at(t) for t in range(now-4,now)],0).copy() if now>=4 else np.zeros((4,ACTION_DIM),dtype=np.float32)),
                visual_offsets=sparse_history_clock(anchor)["history_state_offsets"],observed=now>=4)
        result = HistorySnapshot(
            executed_world=world,
            time_index=now,
            previous_state=None if now == 0 else np.asarray(self._observation_at(now - 1).state, dtype=np.float32).copy(),
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
    "ExecutedWorldSnapshot",
]

