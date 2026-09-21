"""Training-only label provenance and masked reductions.

Observed support is owned by the dataset, not predicted confidence. Online
inputs, world candidates and ODE caches never contain this record. The action
and future supervision partitions share the same immutable support object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from clearvla.data.future_clock import FUTURE_SUPPORT_KEYS


def quarantine(value: Tensor, valid: Tensor) -> Tensor:
    """Remove unsupported values before nonlinearities (NaN * 0 is not safe)."""
    if tuple(value.shape[: valid.ndim]) != tuple(valid.shape):
        raise ValueError("label support must match leading value axes")
    mask = valid.bool().reshape(*valid.shape, *((1,) * (value.ndim - valid.ndim)))
    return torch.where(mask, value, torch.zeros_like(value))


def supported_mean(value: Tensor, valid: Tensor | None) -> Tensor:
    """Average over supported entries; absent labels give an attached zero."""
    if valid is None:
        return value.mean()
    safe = quarantine(value, valid)
    width = value.numel() // max(valid.numel(), 1)
    return safe.sum() / (valid.float().sum() * width).clamp_min(1.0)


@dataclass(frozen=True)
class FutureLabelSupport:
    action: Tensor  # bool [B,Tw], real source command a[t+h]
    state: Tensor  # bool [B,Tw], real successor observation o[t+h+1]
    visual: Tensor  # bool [B,F], real observation at the declared support offset

    @classmethod
    def from_mapping(cls, values: Mapping[str, Tensor]) -> FutureLabelSupport:
        missing = [name for name in FUTURE_SUPPORT_KEYS if name not in values]
        if missing:
            raise ValueError(f"observed-future supervision lacks source masks: {missing}")
        return cls(*(values[name] for name in FUTURE_SUPPORT_KEYS))

    def validate(
        self,
        *,
        batch: int,
        horizon: int,
        offsets: Tensor,
        device: torch.device,
        strict: bool = False,
    ) -> None:
        if offsets.ndim != 2 or int(offsets.shape[0]) != batch:
            raise ValueError("support offsets must be [B,F]")
        if offsets.dtype != torch.long or offsets.device != device:
            raise ValueError("support offsets require int64 on the supervision device")
        for name, rows in (
            ("action", horizon),
            ("state", horizon),
            ("visual", int(offsets.shape[1])),
        ):
            value = getattr(self, name)
            if tuple(value.shape) != (batch, rows) or value.dtype != torch.bool:
                raise ValueError(f"future label {name} support must be bool [B,{rows}]")
            if value.device != device:
                raise ValueError("future support and labels must share a device")
        if not strict:
            return
        if not bool(self.action[:, 0].all()):
            raise ValueError("an admitted current state must own a real first action")
        for mask in (self.action, self.state, self.visual):
            if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
                raise ValueError("observed-tail support must be a contiguous real prefix")
        if bool(((offsets < 1) | (offsets > horizon)).any()):
            raise ValueError("future observation offset outside the world horizon")
        if bool((offsets[:, 1:] <= offsets[:, :-1]).any()):
            raise ValueError("future observation offsets must increase strictly")
        if not torch.equal(self.action, self.state):
            raise ValueError("each observed command must have its real successor observation")
        if not torch.equal(self.visual, self.state.gather(1, offsets - 1)):
            raise ValueError("visual labels must refer to the same real source observations")

    def interval_observed(self, offsets: Tensor, bounds: tuple[tuple[int, int], ...]) -> Tensor:
        """A fixed interval target exists only when ALL of its supports exist.

        Averaging a subset would silently change a fixed interval's target
        meaning. A partially observed interval therefore owns no W target.
        """
        selected = torch.stack([(offsets >= lo) & (offsets <= hi) for lo, hi in bounds], dim=1)
        return ((~selected) | self.visual[:, None]).all(dim=-1) & selected.any(dim=-1)

    def select_batch(self, indices: Tensor) -> FutureLabelSupport:
        return FutureLabelSupport(
            *(x.index_select(0, indices) for x in (self.action, self.state, self.visual))
        )
