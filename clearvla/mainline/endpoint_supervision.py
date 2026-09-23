"""Clean-endpoint command training is a supervision plane, not a solver update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .model.policy import PolicyStepOutput
    from .model.types import FlowStepContext
    from .training.losses import FlowMatchingState

NO_ENDPOINT_SUPERVISION = "interior_heads_v1"
CLEAN_ENDPOINT_SUPERVISION = "clean_command_v1"


def validate_endpoint_mode(mode: str) -> None:
    if mode not in {NO_ENDPOINT_SUPERVISION, CLEAN_ENDPOINT_SUPERVISION}:
        raise ValueError(f"unknown endpoint supervision mode: {mode!r}")


def endpoint_supervision_metadata() -> dict[str, object]:
    return {
        "schema": "clean-endpoint-command-supervision-v1",
        "mode": CLEAN_ENDPOINT_SUPERVISION,
        "time": "t=1,dt=0,index=1,endpoint=1-if-schedule-context-enabled",
        "input": "detached-clean-arm-labels;unknown-rows-source-noise;binary-gripper-source-noise",
        "heads": "existing-command-CE-and-motion-budget-at-endpoint-only",
        "velocity": "interior-flow-target-only-no-endpoint-velocity-loss",
        "execution": "interior-execution-value-supervision-only",
        "cache": "same-causal-online-cache-no-future-world-or-encoder-rebuild",
        "limits": "teacher-forced-endpoint-not-guaranteed-rollout-state-distribution",
        "deployment": "same-two-head-calls-no-new-integration-or-world-build",
    }


def endpoint_condition(
    reference: Tensor, *, context_enabled: bool
) -> tuple[Tensor, FlowStepContext | None]:
    """One numerical endpoint definition used by training and both sampling passes."""
    from .model.types import FlowStepContext

    time = torch.ones(reference.shape[0], device=reference.device, dtype=torch.float32)
    context = (
        FlowStepContext(
            time=time,
            step_size=torch.zeros_like(time),
            normalized_index=torch.ones_like(time),
            endpoint=torch.ones_like(time),
        )
        if context_enabled
        else None
    )
    return time, context


def clean_endpoint_field(flow: FlowMatchingState, *, arm_dim: int) -> Tensor:
    """Binary command lanes cannot leak labels or invent additional executed steps."""
    if flow.target_physical.shape != flow.source_physical_noise.shape:
        raise ValueError("endpoint source noise and target field mismatch")
    if arm_dim < 1 or flow.target_physical.shape[-1] <= 2 * arm_dim:
        raise ValueError("endpoint requires the binary outlet physical field")
    arm = flow.target_physical[..., : 2 * arm_dim].detach()
    noise = flow.source_physical_noise.detach()
    if flow.row_valid is not None:
        if flow.row_valid.dtype != torch.bool or flow.row_valid.shape != arm.shape[:2]:
            raise ValueError("endpoint labels require source-owned [B,T] support")
        arm = torch.where(flow.row_valid[..., None], arm, noise[..., : 2 * arm_dim])
    # Binary gripper dynamics remain frozen noise in the solver and are
    # neutralized at the existing outlet's model-input boundary, in BOTH paths.
    return torch.cat((arm, noise[..., 2 * arm_dim :]), dim=-1)


@dataclass(frozen=True)
class EndpointHeadSupervision:
    output: PolicyStepOutput
    time: Tensor
    context: FlowStepContext | None

    def validate(self) -> None:
        if (
            self.time.ndim != 1
            or self.time.dtype != torch.float32
            or not torch.equal(self.time, torch.ones_like(self.time))
        ):
            raise ValueError("command supervision requires an exact clean endpoint")
        if self.context is not None:
            c = self.context
            if not (
                torch.equal(c.time, self.time)
                and torch.equal(c.step_size, torch.zeros_like(self.time))
                and torch.equal(c.normalized_index, torch.ones_like(self.time))
                and torch.equal(c.endpoint, torch.ones_like(self.time))
            ):
                raise ValueError("command supervision endpoint context mismatch")
