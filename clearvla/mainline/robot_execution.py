"""Causal one-step robot response, NOT object progress or interval-W error.

The input owns o[t-1] and the controller command actually recorded for t-1;
ObservableHistory owns o[t]. No generated action plan or future label is legal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

NO_ROBOT_FEEDBACK = "none"
ONE_STEP_ROBOT_FEEDBACK = "one_step_proprioceptive_v1"


def robot_execution_metadata() -> dict[str, object]:
    return {
        "schema": "one-step-robot-response-v1",
        "scope": "proprioceptive-response-not-object-contact-or-task-progress",
        "source": "recorded-controller-command-not-generated-unexecuted-plan",
        "clock": "o[t-1],a[t-1],o[t];exactly-one-control-step",
        "state": "configured-model-state-feature-difference-not-physical-twist",
        "command": "configured-normalized-native-command-chart",
        "prediction": "previous-state-and-executed-command-only",
        "feedback": "detached-observed-minus-predicted-feature-delta",
        "training": "observed-response-loss-only-trains-predictor",
        "lifecycle": "recomputed-once-per-online-encode;read-only-through-ode",
        "unknown": "no-predecessor-or-dropped-history-means-zero-feedback-and-loss",
        "limit": "model-innovation-not-calibrated-controller-error-or-risk",
    }


@dataclass(frozen=True)
class ExecutedRobotStep:
    previous_state: Tensor  # [B,S], model feature chart
    command: Tensor  # [B,A], normalized RECORDED native controller command
    observed: Tensor  # bool [B], source-owned pair support
    offsets: Tensor  # int64 [B,3] for previous observation, command, current observation

    def validate(
        self,
        *,
        batch: int,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        strict: bool = False,
    ) -> None:
        for name, shape in (
            ("previous_state", (batch, state_dim)),
            ("command", (batch, action_dim)),
        ):
            x = getattr(self, name)
            if x.shape != shape or not x.is_floating_point() or x.device != device:
                raise ValueError(f"executed robot {name} lost its feature/command chart")
        if (
            self.observed.shape != (batch,)
            or self.observed.dtype != torch.bool
            or self.observed.device != device
        ):
            raise ValueError("robot step support must be bool [B]")
        if (
            self.offsets.shape != (batch, 3)
            or self.offsets.dtype != torch.long
            or self.offsets.device != device
        ):
            raise ValueError("robot step offsets must be int64 [B,3]")
        if strict:
            expected = self.offsets.new_tensor([-1, -1, 0])[None].expand(batch, -1)
            if not torch.equal(
                self.offsets,
                torch.where(self.observed[:, None], expected, torch.zeros_like(expected)),
            ):
                raise ValueError(
                    "robot response needs an exact one-step causal transition or absent pair"
                )
            for value in (self.previous_state, self.command):
                if not torch.isfinite(torch.where(self.observed[:, None], value, 0.0)).all():
                    raise ValueError("observed robot step payload must be finite")

    def without_actions(self, keep: Tensor) -> ExecutedRobotStep:
        if keep.shape != self.observed.shape or keep.device != self.observed.device:
            raise ValueError("robot history keep must be [B] on the source device")
        observed = self.observed & keep.bool()
        return replace(
            self,
            observed=observed,
            offsets=torch.where(observed[:, None], self.offsets, torch.zeros_like(self.offsets)),
        )


@dataclass(frozen=True)
class RobotResponseFeedback:
    innovation: Tensor  # detached [B,S]; no auxiliary graph retained in deployment
    observed: Tensor  # bool [B]
    source_step: ExecutedRobotStep | None = None
    current_state: Tensor | None = None

    def validate(self, *, batch: int, state_dim: int, device: torch.device) -> None:
        if (
            self.innovation.shape != (batch, state_dim)
            or not self.innovation.is_floating_point()
            or self.innovation.device != device
        ):
            raise ValueError("robot response feedback feature chart differs")
        if self.innovation.requires_grad:
            raise ValueError("task gradients cannot rewrite the robot response predictor")
        if (
            self.observed.shape != (batch,)
            or self.observed.dtype != torch.bool
            or self.observed.device != device
        ):
            raise ValueError("robot feedback support must be bool [B]")
