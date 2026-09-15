"""Bounded transition replay with explicit post-action state and timeouts."""

import copy

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, feature_dim: int, *, seed: int) -> None:
        if capacity <= 0 or feature_dim <= 0:
            raise ValueError("replay dimensions must be positive")
        self.capacity, self.feature_dim = capacity, feature_dim
        self.size = self.cursor = 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {
            "features": np.zeros((capacity, feature_dim), np.float32),
            "next_features": np.zeros((capacity, feature_dim), np.float32),
            "residual": np.zeros((capacity, 7), np.float32),
            "executed": np.zeros((capacity, 7), np.float32),
            "reward": np.zeros((capacity, 1), np.float32),
            "terminated": np.zeros((capacity, 1), np.float32),
            "truncated": np.zeros((capacity, 1), np.float32),
        }

    def add(
        self,
        *,
        features,
        next_features,
        residual,
        executed,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("termination flags must be bool")
        row = dict(
            features=features,
            next_features=next_features,
            residual=residual,
            executed=executed,
            reward=[reward],
            terminated=[terminated],
            truncated=[truncated],
        )
        validated = {}
        for name, value in row.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape != self.arrays[name].shape[1:] or not np.isfinite(array).all():
                raise ValueError(f"invalid replay field {name}")
            validated[name] = array
        if np.any(np.abs(validated["residual"]) > 1):
            raise ValueError("replay residual is outside [-1,1]")
        # Validate everything before mutating the ring.
        for name, value in validated.items():
            self.arrays[name][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("not enough transitions for a complete replay batch")
        indices = self.rng.integers(self.size, size=batch_size)
        return {
            name: torch.from_numpy(value[indices]).to(device) for name, value in self.arrays.items()
        }

    def state_dict(self) -> dict:
        return dict(
            capacity=self.capacity,
            feature_dim=self.feature_dim,
            size=self.size,
            cursor=self.cursor,
            rng=copy.deepcopy(self.rng.bit_generator.state),
            arrays={k: v[: self.size].copy() for k, v in self.arrays.items()},
        )

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity or state["feature_dim"] != self.feature_dim:
            raise ValueError("replay contract differs")
        size, cursor = state["size"], state["cursor"]
        if (
            type(size) is not int
            or not 0 <= size <= self.capacity
            or type(cursor) is not int
            or not 0 <= cursor < self.capacity
            or (size < self.capacity and cursor != size)
        ):
            raise ValueError("invalid replay position")
        if set(state["arrays"]) != set(self.arrays):
            raise ValueError("incomplete replay state")
        for name, target in self.arrays.items():
            value = state["arrays"][name]
            if value.shape != target[:size].shape or not np.isfinite(value).all():
                raise ValueError(f"invalid saved replay field {name}")
        for name in ("terminated", "truncated"):
            if not np.isin(state["arrays"][name], (0, 1)).all():
                raise ValueError("invalid saved replay termination flags")
        if np.any(np.abs(state["arrays"]["residual"]) > 1):
            raise ValueError("invalid saved replay residual")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        for name, target in self.arrays.items():
            target[:size] = state["arrays"][name]
        self.size, self.cursor, self.rng = size, cursor, rng
