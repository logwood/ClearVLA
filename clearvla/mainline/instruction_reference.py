"""Causal instruction-start observation; never a future target or phase clock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import ExperimentConfig

NO_INSTRUCTION_REFERENCE = "none"
INSTRUCTION_START_REFERENCE = "instruction_start_observation_v1"


def instruction_reference_metadata() -> dict[str, object]:
    return {
        "schema": "causal-instruction-start-observation-v1",
        "source": "declared-instruction-start-not-window-or-prefix-start",
        "values": ["cached-observed-DINO", "shared-model-state-features"],
        "matching": "shared-query-reference-current-soft-reads-per-view",
        "age": "admission-only-never-neural-feature",
        "reset": "episode-reset-or-explicit-new-instruction-event",
        "lifetime": "one-immutable-observed-anchor-per-instruction",
        "physical_object_ids": False,
        "cache_compute_boundary": "storage-values-cast-to-neural-parameter-dtype",
    }


@dataclass(frozen=True)
class InstructionReference:
    dino: Tensor  # [B,C,P,D], actual start observation, never a learned target
    state: Tensor  # [B,S], same feature chart as current/history states
    observed: Tensor  # bool [B,C,P], source availability, not model confidence
    age_steps: Tensor  # Long[B], checked for causality but never consumed by S

    def validate(self, config: ExperimentConfig, *, batch: int, device: torch.device) -> None:
        d = config.dimensions
        shape = (batch, d.num_cameras, d.patches_per_camera, d.visual_token_dim)
        if self.dino.shape != shape or self.state.shape != (batch, d.state_dim):
            raise ValueError("instruction reference lost declared observation/state axes")
        if self.observed.shape != shape[:-1] or self.observed.dtype != torch.bool:
            raise ValueError("instruction reference needs Boolean source-patch support")
        if self.age_steps.shape != (batch,) or self.age_steps.dtype != torch.long:
            raise ValueError("instruction reference age must be physical integer steps")
        if not self.dino.is_floating_point() or not self.state.is_floating_point():
            raise TypeError("instruction reference values must be floating point")
        if any(x.device != device for x in (self.dino, self.state, self.observed, self.age_steps)):
            raise ValueError("instruction reference and current input must share a device")
        if bool((self.age_steps < 0).any()):
            raise ValueError("instruction reference may not originate in the future")
        if not bool(torch.isfinite(self.state).all()):
            raise ValueError("instruction reference state must be a finite observation")
        supported = torch.where(self.observed[..., None], self.dino, torch.zeros_like(self.dino))
        if not bool(torch.isfinite(supported).all()):
            raise ValueError("observed instruction reference tokens must be finite")

    def owned_copy(self) -> InstructionReference:
        """External inputs/returned diagnostics cannot mutate the stored anchor."""
        return InstructionReference(
            *(x.detach().clone() for x in (self.dino, self.state, self.observed, self.age_steps))
        )
