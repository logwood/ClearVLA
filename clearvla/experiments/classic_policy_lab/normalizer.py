from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayNormalizer:
    """Explicit per-field affine normalizer used by classic policy references."""

    offset: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    mode: str

    @classmethod
    def fit_zscore(cls, arrays: list[np.ndarray], *, min_std: float = 1e-2) -> "ArrayNormalizer":
        x = _concat(arrays)
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(x.std(axis=0, keepdims=True).astype(np.float32), float(min_std))
        return cls(
            offset=(-mean / std).astype(np.float32),
            scale=(1.0 / std).astype(np.float32),
            mean=mean,
            std=std,
            minimum=x.min(axis=0, keepdims=True).astype(np.float32),
            maximum=x.max(axis=0, keepdims=True).astype(np.float32),
            mode="zscore",
        )

    @classmethod
    def fit_identity(cls, arrays: list[np.ndarray]) -> "ArrayNormalizer":
        """Keep raw physical units while recording statistics for evaluation."""
        x = _concat(arrays)
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(x.std(axis=0, keepdims=True).astype(np.float32), 1e-8)
        zeros = np.zeros_like(mean, dtype=np.float32)
        ones = np.ones_like(mean, dtype=np.float32)
        return cls(
            offset=zeros,
            scale=ones,
            mean=mean,
            std=std,
            minimum=x.min(axis=0, keepdims=True).astype(np.float32),
            maximum=x.max(axis=0, keepdims=True).astype(np.float32),
            mode="identity",
        )

    @classmethod
    def fit_limits(cls, arrays: list[np.ndarray], *, eps: float = 1e-4) -> "ArrayNormalizer":
        """Match Diffusion Policy's default limits mapping into [-1, 1]."""
        x = _concat(arrays)
        minimum = x.min(axis=0, keepdims=True).astype(np.float32)
        maximum = x.max(axis=0, keepdims=True).astype(np.float32)
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(x.std(axis=0, keepdims=True).astype(np.float32), 1e-8)
        value_range = maximum - minimum
        safe_range = value_range.copy()
        ignore = safe_range < float(eps)
        safe_range[ignore] = 2.0
        scale = (2.0 / safe_range).astype(np.float32)
        offset = (-1.0 - scale * minimum).astype(np.float32)
        offset[ignore] = -minimum[ignore]
        return cls(
            offset=offset,
            scale=scale,
            mean=mean,
            std=std,
            minimum=minimum,
            maximum=maximum,
            mode="limits",
        )

    def encode(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        shape = array.shape
        flat = array.reshape(-1, self.scale.shape[-1])
        return (flat * self.scale + self.offset).astype(np.float32).reshape(shape)

    def decode(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        shape = array.shape
        flat = array.reshape(-1, self.scale.shape[-1])
        return ((flat - self.offset) / self.scale).astype(np.float32).reshape(shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "offset": self.offset.tolist(),
            "scale": self.scale.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArrayNormalizer":
        return cls(
            offset=np.asarray(data["offset"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            minimum=np.asarray(data["minimum"], dtype=np.float32),
            maximum=np.asarray(data["maximum"], dtype=np.float32),
            mode=str(data["mode"]),
        )


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("cannot fit normalizer from empty arrays")
    x = np.concatenate([np.asarray(value, dtype=np.float32) for value in arrays], axis=0)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f"normalizer expects finite [N,D] data, got shape={x.shape}")
    return x
