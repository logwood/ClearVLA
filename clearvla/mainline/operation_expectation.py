"""S-owned expected demonstration outcomes, never observed progress or a W rollout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time

if TYPE_CHECKING:
    from .model.target_binding import TargetBinding

POSTERIOR_INTENT = "posterior_distillation_v1"
OBJECT_OUTCOME_INTENT = "object_outcome_v1"
OPERATION_INTENT_MODES = (POSTERIOR_INTENT, OBJECT_OUTCOME_INTENT)


def operation_expectation_metadata(camera_names: tuple[str, ...]) -> dict[str, object]:
    if (
        not camera_names
        or len(set(camera_names)) != len(camera_names)
        or any(not isinstance(n, str) or not n for n in camera_names)
    ):
        raise ValueError("operation expectation requires unique named cameras")
    return {
        "schema": "object-operation-expectation-v1",
        "mode": OBJECT_OUTCOME_INTENT,
        "camera_names": list(camera_names),
        "role_basis": sorted(camera_names),
        "time_grid": resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME).metadata(),
        "source": "current-G-state-and-causal-S-task-no-candidate-action-or-future-input",
        "values": [
            "per-object-DINO-delta",
            "per-object-per-view-image-displacement",
            "robot-state-feature-delta",
        ],
        "targets": "observed-demonstration-interval-statistics-not-goal-or-success-oracle",
        "loss": "separate-source-supported-Huber-all-current-objects-no-target-mass-or-confidence-weight",
        "recognizer": "replaced-by-parameter-free-outcome-supervision-no-detached-learned-latent-target",
        "read": "separate-value-projections-named-view-before-pooling-shared-K-plus-null",
        "consumers": "S-public-carrier-and-independent-P3-plan-read-not-observed-change-lane",
        "comparison": "no-subtraction-from-start-to-current-change-or-one-step-robot-innovation",
        "state": "one-online-encode-read-only-through-ODE",
    }


@dataclass(frozen=True)
class OperationExpectation:
    semantic_delta: Tensor  # FP32 [B,4,K,D], expected change from current object facts
    image_delta: Tensor  # FP32 [B,4,K,C,2], not world-frame coordinates
    robot_delta: Tensor  # FP32 [B,4,S], feature differences, not velocity/twist
    view_observed: Tensor  # bool [B,K,C], current source support only
    binding: TargetBinding
    current_state: Tensor
    current_content: Tensor
    camera_names: tuple[str, ...]
    time_grid_mode: str = CONTROL_ALIGNED_FUTURE_TIME

    def validate(self, *, strict: bool = False) -> None:
        if self.time_grid_mode != CONTROL_ALIGNED_FUTURE_TIME:
            raise ValueError("operation expectation physical time mismatch")
        if self.semantic_delta.ndim != 4 or self.semantic_delta.shape[1] != 4:
            raise ValueError("operation content expectation must retain [B,4,K,D]")
        b, _, k, d = self.semantic_delta.shape
        c = len(self.camera_names)
        operation_expectation_metadata(self.camera_names)
        if self.current_state.ndim != 2 or self.current_state.shape[0] != b:
            raise ValueError("operation expectation has another current-state chart")
        if self.current_content.shape != (b, k, d):
            raise ValueError("operation expectation lost current object provenance")
        if self.image_delta.shape != (b, 4, k, c, 2) or self.robot_delta.shape != (
            b,
            4,
            self.current_state.shape[-1],
        ):
            raise ValueError("operation expectation lost typed object/view/robot axes")
        if self.view_observed.shape != (b, k, c) or self.view_observed.dtype != torch.bool:
            raise ValueError("operation expectation source support must be bool [B,K,C]")
        if any(
            x.device != self.current_state.device
            for x in (
                self.semantic_delta,
                self.image_delta,
                self.robot_delta,
                self.view_observed,
                self.current_content,
            )
        ):
            raise ValueError("operation expectation source device mismatch")
        if any(
            x.dtype != torch.float32
            for x in (self.semantic_delta, self.image_delta, self.robot_delta)
        ):
            raise TypeError("operation expectation value charts must remain FP32")
        self.binding.validate(batch=b, objects=k, device=self.current_state.device)
        if strict:
            for value, mask in (
                (self.semantic_delta, self.view_observed.any(-1)[:, None, :, None]),
                (self.image_delta, self.view_observed[:, None, :, :, None]),
                (
                    self.robot_delta,
                    torch.ones((b, 1, 1), dtype=torch.bool, device=self.current_state.device),
                ),
            ):
                if not bool(torch.isfinite(torch.where(mask, value, 0.0)).all()):
                    raise ValueError("nonfinite supported operation expectation")

    def permute(self, index: Tensor, binding: TargetBinding) -> OperationExpectation:
        return replace(
            self,
            semantic_delta=self.semantic_delta[:, :, index],
            image_delta=self.image_delta[:, :, index],
            view_observed=self.view_observed[:, index],
            binding=binding,
            current_content=self.current_content[:, index],
        )
