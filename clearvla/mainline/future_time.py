"""One physical clock for S, W, Teacher and policy consequence consumers.

This is future control time, not diffusion time or elapsed task progress.
The legacy layout is kept only as an explicitly identified control graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

LEGACY_FUTURE_TIME = "legacy_48_v1"
CONTROL_ALIGNED_FUTURE_TIME = "control_aligned_24_v1"
LEGACY_INTERVALS = ((4, 8), (8, 16), (16, 32), (32, 48))
CONTROL_INTERVALS = ((0, 4), (4, 8), (8, 16), (16, 24))


@dataclass(frozen=True)
class FutureTimeGrid:
    """Immutable resolved physical schedule, never inferred from array length."""

    mode: str = LEGACY_FUTURE_TIME

    def __post_init__(self) -> None:
        if self.mode not in {LEGACY_FUTURE_TIME, CONTROL_ALIGNED_FUTURE_TIME}:
            raise ValueError(f"unknown future time grid: {self.mode!r}")

    @property
    def aligned(self) -> bool:
        return self.mode == CONTROL_ALIGNED_FUTURE_TIME

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        return CONTROL_INTERVALS if self.aligned else LEGACY_INTERVALS

    @property
    def endpoints(self) -> tuple[int, ...]:
        return tuple(upper for _, upper in self.bounds)

    @property
    def horizon(self) -> int:
        return self.endpoints[-1]

    @property
    def centers(self) -> tuple[float, ...]:
        return tuple((lo + hi) / 2 for lo, hi in self.bounds)

    @property
    def support_offsets(self) -> tuple[int, ...]:
        return tuple(range(4, self.horizon + 1, 4))

    def action_slices(self, length: int) -> tuple[slice, ...]:
        if length < 1 or (self.aligned and length < self.horizon):
            raise ValueError("future sequence does not cover the declared time grid")
        if self.aligned:
            return tuple(slice(lo, hi) for lo, hi in self.bounds)
        # Historical clipping remains bit-exact only in the legacy control.
        rows: list[slice] = []
        for lo, hi in self.bounds:
            start = min(max(lo - 1, 0), length - 1)
            rows.append(slice(start, min(max(hi, start + 1), length)))
        return tuple(rows)

    def selection(self, offsets: Tensor) -> Tensor:
        """[B,I,F] samples of o[t+h] in each physical interval."""
        if offsets.ndim != 2:
            raise ValueError("future offsets require [B,F]")
        return torch.stack([
            ((offsets > lo) if self.aligned else (offsets >= lo)) & (offsets <= hi)
            for lo, hi in self.bounds
        ], dim=1)

    def validate_offsets(self, offsets: Tensor) -> None:
        if offsets.ndim != 2 or offsets.dtype != torch.long:
            raise ValueError("future time offsets must be int64 [B,F]")
        if self.aligned:
            expected = offsets.new_tensor(self.support_offsets)[None].expand(offsets.shape[0], -1)
            if not torch.equal(offsets, expected):
                raise ValueError("future supports differ from the declared physical time grid")

    def interval_observed(self, offsets: Tensor, observed: Tensor) -> Tensor:
        self.validate_offsets(offsets)
        if observed.dtype != torch.bool or observed.shape != offsets.shape or observed.device != offsets.device:
            raise ValueError("future observation support must match its offset chart")
        selected = self.selection(offsets)
        return ((~selected) | observed[:, None]).all(-1) & selected.any(-1)

    def interval_encoding(self, width: int) -> Tensor:
        """Deterministic endpoint/width codes, using a shared 24-step unit.

        Computed on CPU at construction and stored by the consumer. No RNG,
        trainable timing gate, host synchronization or task-phase clock.
        """
        points = torch.tensor(self.bounds, dtype=torch.float32) / 24.0
        freq = torch.exp(-math.log(10000.0) * torch.arange(max((width + 3)//4, 1)).float()
                         / max((width + 3)//4, 1))
        angles = points[..., None] * freq
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :width]

    def row_interpolation(self, horizon: int = 24) -> Tensor:
        """Interpolate the four learned coarse knots at their PHYSICAL centers."""
        if not self.aligned or horizon != self.horizon:
            raise ValueError("physical coarse interpolation requires the aligned control horizon")
        centers = torch.tensor(self.centers)
        rows = torch.arange(horizon).float() + 0.5
        right = torch.searchsorted(centers, rows).clamp(1, len(centers)-1)
        left = right - 1
        weight = ((rows-centers[left])/(centers[right]-centers[left])).clamp(0, 1)
        result = torch.zeros(horizon, len(centers))
        result.scatter_(1, left[:, None], (1-weight)[:, None])
        result.scatter_add_(1, right[:, None], weight[:, None])
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "physical-future-time-grid-v1", "mode": self.mode,
            "unit": "control-step", "bounds": [list(x) for x in self.bounds],
            "action_rows": "[lower,upper)" if self.aligned else "legacy-clipped-inclusive",
            "successor_observations": "(lower,upper]" if self.aligned else "legacy-inclusive",
            "support_offsets": list(self.support_offsets), "control_time_scale": 24,
            "target": "uniform-mean-over-declared-observed-supports",
            "w1_intervals": [0, 1], "w2_intervals": [2, 3],
            "not_task_phase_or_ode_time": True,
        }


def resolve_future_time(mode: str = LEGACY_FUTURE_TIME) -> FutureTimeGrid:
    return FutureTimeGrid(mode)
