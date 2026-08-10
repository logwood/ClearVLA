"""One executable numerical policy for training and deployment."""

from __future__ import annotations

import torch

from ..config import ExperimentConfig


def resolve_compute_dtype(
    config: ExperimentConfig,
    override: torch.dtype | None = None,
) -> torch.dtype:
    """Resolve the serialized dtype and reject an inconsistent live override."""

    expected = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[config.runtime.compute_dtype]
    if override is not None and override != expected:
        raise ValueError("live compute dtype differs from the serialized runtime configuration")
    return expected


__all__ = ["resolve_compute_dtype"]
