"""Causal, source-owned full-posterior instruction evidence and static reads.

No persistent object IDs, desired outcomes, future labels or progress numbers
are represented here. Current and reference are both observed image charts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from clearvla.vision.entity_chart import ObjectImageReadSource

if TYPE_CHECKING:
    from .instruction_change import InstructionChangeEvidence


@dataclass(frozen=True)
class InstructionPosterior:
    current: Tensor               # [B,C,N,D], original source reference
    reference: Tensor             # [B,C,N,D]
    observed: Tensor              # bool [B,C,N], comparable source support
    coordinates: Tensor           # FP32 [N,2], native image chart
    source_probability: Tensor    # FP32 [B,K,C,N], current G3 source law
    current_probability: Tensor   # FP32 [B,K,C,N], real mass, null retained
    reference_probability: Tensor # FP32 [B,K,C,N]
    current_null: Tensor          # FP32 [B,K,C]
    reference_null: Tensor        # FP32 [B,K,C]
    source: ObjectImageReadSource

    def validate(self, *, strict: bool = False) -> None:
        if self.current.ndim != 4 or self.reference.shape != self.current.shape:
            raise ValueError("posterior images must keep the same [B,C,N,D] chart")
        batch, cameras, patches, _ = self.current.shape
        shape = self.current_probability.shape
        if len(shape) != 4 or shape[0] != batch or shape[2:] != (cameras, patches):
            raise ValueError("posterior law must preserve [B,K,C,N]")
        if self.reference_probability.shape != shape or self.source_probability.shape != shape:
            raise ValueError("posterior current/reference/source law axes differ")
        if self.observed.shape != (batch, cameras, patches) or self.observed.dtype != torch.bool:
            raise ValueError("posterior source support must remain Boolean [B,C,N]")
        if self.coordinates.shape != (patches, 2):
            raise ValueError("posterior coordinates lost the complete native chart")
        if self.current_null.shape != shape[:-1] or self.reference_null.shape != shape[:-1]:
            raise ValueError("posterior null law lost object/view axes")
        values = (
            self.coordinates, self.source_probability, self.current_probability,
            self.reference_probability, self.current_null, self.reference_null,
        )
        if any(value.dtype != torch.float32 for value in values):
            raise TypeError("posterior measures must remain FP32")
        if any(value.device != self.current.device for value in (*values, self.reference, self.observed)):
            raise ValueError("posterior sources and laws require one device")
        if self.source.log_measure.shape[:3] != shape[:3]:
            raise ValueError("posterior law belongs to another G3 object/view source")
        if strict:
            if any(not bool(torch.isfinite(value).all()) for value in values):
                raise ValueError("posterior laws and coordinates must be finite")
            if any(bool((value < 0).any()) for value in values[1:]):
                raise ValueError("posterior probabilities must be nonnegative")
            mask = self.observed[:, None].expand(shape)
            if any(bool((value[~mask] != 0).any()) for value in values[1:4]):
                raise ValueError("posterior assigned mass outside observed support")
            mass = self.source_probability.sum(-1)
            for probability, null in (
                (self.current_probability, self.current_null),
                (self.reference_probability, self.reference_null),
            ):
                if not torch.allclose(probability.sum(-1) + null, mass, atol=2e-5, rtol=0):
                    raise ValueError("posterior real plus null mass differs from source")
            for raw in (self.current, self.reference):
                if not bool(torch.isfinite(torch.where(self.observed[..., None], raw, 0.0)).all()):
                    raise ValueError("posterior contains nonfinite observed content")

    def permute(self, index: Tensor) -> InstructionPosterior:
        return replace(
            self,
            source_probability=self.source_probability[:, index],
            current_probability=self.current_probability[:, index],
            reference_probability=self.reference_probability[:, index],
            current_null=self.current_null[:, index],
            reference_null=self.reference_null[:, index],
            source=self.source.permute(index),
        )


@dataclass(frozen=True)
class InstructionChangeValues:
    """Reader-owned projections built once; no image reopening inside P3/ODE."""

    content: Tensor                     # FP32 [B,K,C,H]
    image: Tensor                       # FP32 [B,K,C,H]
    robot: Tensor                       # FP32 [B,H], target law applied at read
    status: Tensor                      # FP32 [B,K,C,H], not a change value
    joint: Tensor | None                # FP32 [B,K,C,H], content-position law
    evidence: InstructionChangeEvidence
    reader_identity: int                # runtime module owner, not checkpoint

    def validate(self, *, hidden: int) -> None:
        evidence = self.evidence
        shape = (*evidence.content_delta.shape[:3], hidden)
        if self.content.shape != shape or self.image.shape != shape or self.status.shape != shape:
            raise ValueError("prepared instruction reads lost per-object/per-view axes")
        if self.robot.shape != (shape[0], hidden):
            raise ValueError("prepared instruction robot read lost its own chart")
        if (self.joint is not None) != (evidence.posterior is not None):
            raise ValueError("prepared posterior read lost joint evidence")
        if self.joint is not None and self.joint.shape != shape:
            raise ValueError("prepared joint read lost per-object/per-view axes")
        tensors: tuple[Tensor, ...] = (self.content, self.image, self.robot, self.status)
        if self.joint is not None:
            tensors = (*tensors, self.joint)
        if any(value.dtype != torch.float32 or value.device != evidence.current_state.device for value in tensors):
            raise ValueError("prepared instruction values require same-device FP32")

    def permute(
        self, index: Tensor, evidence: InstructionChangeEvidence
    ) -> InstructionChangeValues:
        return replace(
            self,
            content=self.content[:, index],
            image=self.image[:, index],
            status=self.status[:, index],
            joint=None if self.joint is None else self.joint[:, index],
            evidence=evidence,
        )
