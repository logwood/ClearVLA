from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZScoreNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, arrays: list[np.ndarray], eps: float = 1e-6) -> "ZScoreNormalizer":
        if not arrays:
            raise ValueError("Cannot fit normalizer from an empty array list")
        x = np.concatenate(arrays, axis=0).astype(np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected [N,D] arrays, got concatenated shape={x.shape}")
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(x.std(axis=0, keepdims=True).astype(np.float32), eps)
        return cls(mean=mean, std=std)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[list[float]]]) -> "ZScoreNormalizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
        )
