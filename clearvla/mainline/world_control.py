"""Source-owned domain of a candidate-conditioned world prediction.

Control completeness is neither visual validity nor learned confidence.  A
24-row proposal does not supply the controls at 25..48, even when W emits
values at those horizons.  Keep that distinction until the actual P2 read.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .manifest import INTERVALS

LEGACY_WORLD_CONTROL = "legacy_extrapolation_v1"
KNOWN_PREFIX_WORLD_CONTROL = "known_prefix_v1"


def world_control_metadata() -> dict[str, object]:
    return {
        "schema": "candidate-control-domain-v1",
        "mode": KNOWN_PREFIX_WORLD_CONTROL,
        "source": "candidate-action-prefix",
        "interval_bounds": [list(pair) for pair in INTERVALS],
        "online_known_prefix": 24,
        "admission": "all-controls-through-interval-upper-known",
        "unknown": "excluded-before-value-key-and-common-reduction",
        "observation_support": "unchanged",
    }


@dataclass(frozen=True)
class CandidateControlDomain:
    """Static control domain; no tensor mutation or ODE-time CUDA checks.

    Constructed by W from the actual candidate action condition, not S or P.
    The existing online ABI has a common dense prefix length for each batch.
    Training-only observed controls use their separate per-sample label masks.
    """

    known_prefix_steps: int
    interval_bounds: tuple[tuple[int, int], ...] = INTERVALS

    def validate(self, *, intervals: int = 4) -> None:
        if type(self.known_prefix_steps) is not int or self.known_prefix_steps < 0:
            raise ValueError("candidate known control prefix must be a nonnegative integer")
        if self.interval_bounds != INTERVALS or len(self.interval_bounds) != intervals:
            raise ValueError("candidate control domain must retain physical interval bounds")

    @property
    def admitted(self) -> tuple[bool, ...]:
        return tuple(upper <= self.known_prefix_steps for _, upper in self.interval_bounds)

    def mask_for(self, value: Tensor) -> Tensor:
        if value.ndim < 2 or value.shape[1] != len(self.interval_bounds):
            raise ValueError("control-domain value must retain the physical interval axis")
        return torch.tensor(self.admitted, device=value.device, dtype=torch.bool).reshape(
            1, len(self.interval_bounds), *([1] * (value.ndim - 2))
        )

    def quarantine(self, value: Tensor) -> Tensor:
        """Mask before arithmetic: an unknown NaN is not a zero command."""
        return torch.where(self.mask_for(value), value, torch.zeros_like(value))

    def common(self, value: Tensor) -> Tensor:
        safe = self.quarantine(value.float())
        return safe.sum(dim=1) / max(sum(self.admitted), 1)
