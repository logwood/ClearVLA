"""Typed physical history time: producer provenance, not learned confidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor

from clearvla.data.history_clock import HISTORY_TIMING_CONTRACT, HISTORY_TIMING_KEYS

LEGACY_HISTORY_ENCODING = "paired_rows_v1"
TIMED_HISTORY_ENCODING = "timestamped_streams_v1"


@dataclass(frozen=True)
class HistoryTiming:
    """Relative physical-step clocks; masks distinguish reset padding from data.

    state_offsets identifies the source of each state row (padding repeats the
    reset source). action_offsets identifies a requested command; only entries
    with action_executed=True describe commands that were actually executed.
    Value validation belongs at producer/preflight boundaries, not ODE nodes.
    """

    state_offsets: Tensor
    action_offsets: Tensor
    state_observed: Tensor
    action_executed: Tensor

    @classmethod
    def from_mapping(cls, values: Mapping[str, Tensor]) -> HistoryTiming:
        missing = [name for name in HISTORY_TIMING_KEYS if name not in values]
        if missing:
            raise ValueError(f"timestamped history is missing source metadata: {missing}")
        return cls(*(values[name] for name in HISTORY_TIMING_KEYS))

    def validate(
        self,
        *,
        batch: int,
        states: int,
        actions: int,
        device: torch.device,
        strict: bool = False,
    ) -> None:
        for name, value, rows, dtype in (
            ("state_offsets", self.state_offsets, states, torch.long),
            ("action_offsets", self.action_offsets, actions, torch.long),
            ("state_observed", self.state_observed, states, torch.bool),
            ("action_executed", self.action_executed, actions, torch.bool),
        ):
            if not isinstance(value, Tensor) or tuple(value.shape) != (batch, rows):
                raise ValueError(f"history {name} must be [B,{rows}]")
            if value.dtype != dtype or value.device != device:
                raise ValueError(f"history {name} has an invalid dtype/device")
        if min(batch, states, actions) < 1:
            raise ValueError("history timing axes must be nonempty")
        if not strict:
            return
        if bool((self.state_offsets > 0).any()) or bool((self.action_offsets >= 0).any()):
            raise ValueError("history contains a future state or an unexecuted current action")
        if bool((self.state_offsets[:, -1] != 0).any()) or not bool(
            self.state_observed[:, -1].all()
        ):
            raise ValueError("last history state must be the real current observation")
        ds = self.state_offsets[:, 1:] - self.state_offsets[:, :-1]
        da = self.action_offsets[:, 1:] - self.action_offsets[:, :-1]
        same_time = self.state_offsets[:, :, None] == self.state_offsets[:, None, :]
        both_real = self.state_observed[:, :, None] & self.state_observed[:, None, :]
        duplicate_real = torch.triu(same_time & both_real, diagonal=1)
        if bool((ds < 0).any()) or bool(duplicate_real.any()) or bool((da <= 0).any()):
            raise ValueError("history time must be ordered; real rows cannot duplicate time")

    def without_actions(self, keep: Tensor) -> HistoryTiming:
        """Condition dropout removes evidence, never marks zero as execution."""
        if tuple(keep.shape) != (self.action_executed.shape[0],):
            raise ValueError("history keep mask must be [B]")
        return replace(self, action_executed=self.action_executed & keep.bool()[:, None])

    def as_mapping(self) -> dict[str, Tensor]:
        return dict(
            zip(
                HISTORY_TIMING_KEYS,
                (
                    self.state_offsets,
                    self.action_offsets,
                    self.state_observed,
                    self.action_executed,
                ),
                strict=True,
            )
        )


__all__ = [
    "HistoryTiming",
    "HISTORY_TIMING_CONTRACT",
    "LEGACY_HISTORY_ENCODING",
    "TIMED_HISTORY_ENCODING",
]
