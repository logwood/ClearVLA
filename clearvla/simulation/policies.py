"""Small policy protocol and dependency-free simulator smoke policies."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import ACTION_DIM
from .history import HistorySnapshot


class RolloutPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, history: HistorySnapshot, instruction: str) -> np.ndarray: ...


class HoldPolicy:
    def __init__(self, action: np.ndarray, *, horizon: int = 24) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("hold action must be one finite [7] vector")
        if horizon <= 0:
            raise ValueError("policy horizon must be positive")
        self.action = value.copy()
        self.horizon = int(horizon)

    def reset(self) -> None:
        pass

    def act(self, history: HistorySnapshot, instruction: str) -> np.ndarray:
        history.validate()
        if not instruction.strip():
            raise ValueError("rollout instruction must be non-empty")
        return np.repeat(self.action[None], self.horizon, axis=0)


class EnvironmentRandomPolicy:
    """Random policy whose sampler remains owned by the environment backend."""

    def __init__(self, sampler, *, seed: int, horizon: int = 24) -> None:
        self._sampler = sampler
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.horizon = int(horizon)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, history: HistorySnapshot, instruction: str) -> np.ndarray:
        history.validate()
        value = np.asarray(self._sampler(self._rng), dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("environment sampler returned a non-finite [7] action")
        return np.repeat(value[None], self.horizon, axis=0)


__all__ = ["EnvironmentRandomPolicy", "HoldPolicy", "RolloutPolicy"]


