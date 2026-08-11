"""Small typed boundary shared by the active V120 observation path.

This module intentionally contains no visual encoder, flow estimator or G
network.  Keeping the online evidence dataclasses and the one FP32 spatial
sampler here prevents the active mainline from importing the superseded
independent observation implementation merely to borrow utility definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..v120_core.flow_dino_evidence import ProgressiveFineCandidates
from .types import LocalFactSet


def _coordinate_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _sample_feature_chart(feature: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Sample ``[B,C,D,H,W]`` at ``[B,C,Y,X,M,2]`` in FP32."""

    if feature.ndim != 5 or coordinates.ndim != 6:
        raise ValueError("feature sampling requires [B,C,D,H,W] and [B,C,Y,X,M,2]")
    batch, cameras, channels = feature.shape[:3]
    if tuple(coordinates.shape[:2]) != (batch, cameras) or int(coordinates.shape[-1]) != 2:
        raise ValueError("feature chart and coordinate prefixes do not align")
    rows, columns, hypotheses = coordinates.shape[2:5]
    flat_feature = feature.reshape(batch * cameras, channels, *feature.shape[-2:])
    flat_grid = coordinates.reshape(batch * cameras, rows, columns * hypotheses, 2)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        sampled = F.grid_sample(
            flat_feature.float(),
            flat_grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    sampled = sampled.reshape(batch, cameras, channels, rows, columns, hypotheses)
    sampled = sampled.permute(0, 1, 3, 4, 5, 2).contiguous()
    valid = (coordinates.abs() <= 1.0).all(dim=-1, keepdim=True)
    return sampled, valid


@dataclass(frozen=True)
class PatchFlowField:
    """Learned flow in normalized coordinates, indexed on destination charts."""

    forward: Tensor
    backward: Tensor | None
    confidence: Tensor
    uncertainty: Tensor
    occlusion: Tensor
    refinement_sequence: tuple[Tensor, ...]

    def validate(self) -> None:
        if self.forward.ndim != 5 or int(self.forward.shape[2]) != 2:
            raise ValueError("patch flow must be [B,C,2,H,W]")
        if self.backward is not None and tuple(self.backward.shape) != tuple(self.forward.shape):
            raise ValueError("forward and backward flow charts must align")
        scalar_shape = (*self.forward.shape[:2], 1, *self.forward.shape[-2:])
        for name in ("confidence", "uncertainty", "occlusion"):
            if tuple(getattr(self, name).shape) != scalar_shape:
                raise ValueError(f"flow {name} chart does not align")
        if not self.refinement_sequence:
            raise ValueError("flow refinement sequence cannot be empty")


@dataclass(frozen=True)
class ObservationEvidence:
    """Current facts plus the sole high-resolution bank available to P1."""

    local_facts: LocalFactSet
    progressive_candidates: ProgressiveFineCandidates
    detail_features: Tensor
    previous_detail_features: Tensor
    earlier_detail_features: Tensor | None
    literal_rgb: Tensor
    previous_literal_rgb: Tensor
    earlier_literal_rgb: Tensor
    flow: PatchFlowField
    earlier_flow: PatchFlowField
    context_mask: Tensor
    native_flow_losses: dict[str, Tensor] | None = None

    def validate(self) -> None:
        self.local_facts.validate()
        batch = self.local_facts.batch
        cameras = int(self.local_facts.public_scene_base.shape[1])
        if self.detail_features.ndim != 5 or tuple(self.detail_features.shape[:2]) != (
            batch,
            cameras,
        ):
            raise ValueError("detail features must be [B,C,F,H,W]")
        if tuple(self.previous_detail_features.shape) != tuple(self.detail_features.shape):
            raise ValueError("previous/current detail features must align")
        if self.earlier_detail_features is not None and tuple(
            self.earlier_detail_features.shape
        ) != tuple(self.detail_features.shape):
            raise ValueError("causal detail history must align")
        if self.literal_rgb.ndim != 5 or tuple(self.literal_rgb.shape[:2]) != (
            batch,
            cameras,
        ):
            raise ValueError("literal RGB must be [B,C,3,R,R]")
        if tuple(self.previous_literal_rgb.shape) != tuple(self.literal_rgb.shape):
            raise ValueError("previous/current literal RGB charts must align")
        if tuple(self.earlier_literal_rgb.shape) != tuple(self.literal_rgb.shape):
            raise ValueError("causal literal RGB history must align")
        if tuple(self.context_mask.shape) != (batch, cameras, 8, 8):
            raise ValueError("context mask must preserve [B,C,8,8]")
        self.flow.validate()
        self.earlier_flow.validate()
        candidates = self.progressive_candidates
        candidate_prefix = (
            batch,
            cameras,
            8,
            8,
            self.local_facts.local_hypotheses,
        )
        if candidates.learned_detail.ndim != 7 or tuple(
            candidates.learned_detail.shape[:5]
        ) != candidate_prefix:
            raise ValueError(
                "progressive fine candidates must preserve [B,C,8,8,M,N,*]"
            )
        fine_prefix = tuple(candidates.learned_detail.shape[:-1])
        if tuple(candidates.valid.shape) != fine_prefix:
            raise ValueError("progressive fine validity lost the N candidate axis")
        if tuple(candidates.current_coordinates.shape) != (*fine_prefix, 2):
            raise ValueError("progressive fine coordinates are misaligned")
        for name in (
            "semantic_keys",
            "appearance_keys",
            "geometry_keys",
            "literal_rgb",
        ):
            value = getattr(candidates, name)
            if value is None or tuple(value.shape[:-1]) != fine_prefix:
                raise ValueError(f"progressive candidate {name} is missing or misaligned")
        if self.native_flow_losses is not None:
            required = {
                "flow_jepa_warp_loss",
                "flow_jepa_cycle_loss",
                "flow_jepa_smoothness_loss",
                "flow_jepa_uncertainty_nll",
                "flow_jepa_refinement_sequence_loss",
                "flow_jepa_identity_advantage_loss",
                "flow_jepa_static_identity_loss",
            }
            missing = required - set(self.native_flow_losses)
            if missing:
                raise ValueError(
                    "native V120 flow ledger is incomplete: " + ", ".join(sorted(missing))
                )
            if any(value.ndim != 0 for value in self.native_flow_losses.values()):
                raise ValueError("native V120 flow losses must be scalar")
        elif self.earlier_detail_features is None:
            raise ValueError(
                "explicit fallback flow losses require a real earlier detail feature"
            )


__all__ = [
    "ObservationEvidence",
    "PatchFlowField",
    "_coordinate_grid",
    "_sample_feature_chart",
]
