"""Typed comparison of two causal observations, not a goal-progress oracle.

Visual correspondence is soft and image-relative. Match status, apparent target
change and robot feature change have separate owners and downstream semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .instruction_reference import InstructionReference
    from .model.target_binding import TargetBinding

MIXED_REFERENCE_CHANGE = "mixed_reference_v1"
TYPED_REFERENCE_CHANGE = "typed_reference_v1"
INSTRUCTION_CHANGE_MODES = (MIXED_REFERENCE_CHANGE, TYPED_REFERENCE_CHANGE)


def instruction_change_metadata(camera_names: tuple[str, ...]) -> dict[str, object]:
    if (
        not camera_names
        or len(set(camera_names)) != len(camera_names)
        or any(not n for n in camera_names)
    ):
        raise ValueError("instruction change requires unique named camera charts")
    return {
        "schema": "typed-instruction-observation-change-v1",
        "mode": TYPED_REFERENCE_CHANGE,
        "camera_names": list(camera_names),
        "role_basis": sorted(camera_names),
        "matching": "same-current-object-content-query-on-both-images-no-task-query",
        "patch_support": "source-observed-intersection-not-learned-match-confidence",
        "values": [
            "DINO-content-difference",
            "image-centroid-difference",
            "robot-state-feature-difference",
            "separate-correspondence-status",
        ],
        "view_fusion": "source-specific-role-conditioned-features-before-view-pooling",
        "target": "same-K-plus-null-binding-applied-after-view-read",
        "S": "separate-source-projections-before-task-conditioning",
        "P3": "status-conditions-change-read-but-cannot-create-change-values",
        "zero": "identical-comparable-observations-have-zero-change-not-zero-status",
        "clock": "instruction-start-to-current-no-age-feature-no-ODE-state-update",
        "limits": "apparent-image-change-not-world-displacement-contact-success-or-persistent-ID",
    }


@dataclass(frozen=True)
class InstructionChangeEvidence:
    content_delta: Tensor  # FP32 [B,K,C,D]
    image_delta: Tensor  # FP32 [B,K,C,2], centroid difference in each image chart
    robot_delta: Tensor  # FP32 [B,S], feature change, not twist/object motion
    match_status: Tensor  # FP32 [B,K,C,6], ambiguity and source-availability facts
    view_observed: Tensor  # bool [B,K,C], comparable source support
    view_weight: Tensor  # FP32 [B,K,C], within-object source-view read
    binding: TargetBinding
    current_state: Tensor
    reference: InstructionReference
    camera_names: tuple[str, ...]

    def validate(self, *, hidden_content: int | None = None, strict: bool = False) -> None:
        if self.content_delta.ndim != 4:
            raise ValueError("instruction content change requires [B,K,C,D]")
        batch, objects, cameras, width = self.content_delta.shape
        if min(batch, objects, cameras, width) < 1 or (
            hidden_content is not None and width != hidden_content
        ):
            raise ValueError("instruction change lost content axes")
        if cameras != len(self.camera_names) or len(set(self.camera_names)) != cameras:
            raise ValueError("instruction change lost named camera chart")
        if self.image_delta.shape != (batch, objects, cameras, 2):
            raise ValueError("instruction image change lost per-object/per-view axes")
        if self.match_status.shape != (batch, objects, cameras, 6):
            raise ValueError("instruction match status must remain separate [B,K,C,6]")
        if (
            self.view_observed.shape != (batch, objects, cameras)
            or self.view_observed.dtype != torch.bool
        ):
            raise ValueError("instruction comparison requires Boolean view support")
        if self.view_weight.shape != self.view_observed.shape:
            raise ValueError("instruction view weights differ from support")
        if (
            self.robot_delta.ndim != 2
            or self.robot_delta.shape != self.current_state.shape
            or self.robot_delta.shape[0] != batch
        ):
            raise ValueError("instruction robot change lost state-feature chart")
        if self.binding.supported.shape != (batch, objects):
            raise ValueError("instruction change binding lost current objects")
        values = (
            self.content_delta,
            self.image_delta,
            self.robot_delta,
            self.match_status,
            self.view_weight,
        )
        if any(x.dtype != torch.float32 for x in values):
            raise TypeError("instruction change measurements must remain FP32")
        if any(
            x.device != self.current_state.device
            for x in (*values, self.view_observed, self.binding.log_probability)
        ):
            raise ValueError("instruction change belongs to another observation device")
        if strict:
            self.binding.validate(batch=batch, objects=objects, device=self.current_state.device)
            masked = (
                torch.where(self.view_observed[..., None], self.content_delta, 0.0),
                torch.where(self.view_observed[..., None], self.image_delta, 0.0),
                self.robot_delta,
                torch.where(self.binding.supported[..., None, None], self.match_status, 0.0),
                self.view_weight,
            )
            if any(not bool(torch.isfinite(x).all()) for x in masked):
                raise ValueError("instruction change has nonfinite supported evidence")
            if bool((self.view_weight < 0).any()) or bool(
                (self.view_weight[~self.view_observed] != 0).any()
            ):
                raise ValueError("instruction view read assigned invalid mass")
            expected = self.view_observed.any(-1).float()
            if not torch.allclose(self.view_weight.sum(-1), expected, atol=2e-5, rtol=0):
                raise ValueError("instruction view mass must normalize only inside each object")

    def permute(self, index: Tensor, binding: TargetBinding) -> InstructionChangeEvidence:
        return replace(
            self,
            content_delta=self.content_delta[:, index],
            image_delta=self.image_delta[:, index],
            match_status=self.match_status[:, index],
            view_observed=self.view_observed[:, index],
            view_weight=self.view_weight[:, index],
            binding=binding,
        )
