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

from ..v120_core.flow_dino_evidence import (
    LateRawDetailEvidence,
    ProgressiveGroundingAddressState,
    SoftAddressLatticeBank,
)
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
class GroundingObservationBank:
    """Complete pre-G observation bank; no local fact is materialized yet."""

    address_bank: SoftAddressLatticeBank
    late_detail: LateRawDetailEvidence
    visual_memory: Tensor
    visual_value_memory: Tensor
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
        if self.address_bank.dense_current_dino_content is None:
            raise ValueError("grounding bank lost the full current DINO chart")
        if self.late_detail.address_bank is not self.address_bank:
            raise ValueError("late detail and grounding address bank must be identical")
        batch, cameras = self.address_bank.dense_current_dino_content.shape[:2]
        if self.visual_memory.ndim != 3 or tuple(self.visual_memory.shape) != tuple(
            self.visual_value_memory.shape
        ):
            raise ValueError("grounding selector/value memory must align as [B,N,H]")
        if int(self.visual_memory.shape[0]) != batch:
            raise ValueError("grounding memory batch does not align with the address bank")
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


@dataclass(frozen=True)
class ObservationEvidence:
    """Completed G3 facts plus the one lossless V120 precision bank."""

    grounding: GroundingObservationBank
    progressive_state: ProgressiveGroundingAddressState
    local_facts: LocalFactSet

    def validate(self) -> None:
        self.grounding.validate()
        self.local_facts.validate()
        state = self.progressive_state
        if state.stage != 3 or state.grounded_fact_set is None:
            raise ValueError("observation evidence requires completed G1/G2/G3 facts")
        required = (
            state.dynamic_fine_values,
            state.dynamic_fine_valid,
            state.dynamic_fine_coordinates,
            state.dynamic_semantic_keys,
            state.dynamic_appearance_keys,
            state.dynamic_geometry_keys,
            state.dynamic_literal_rgb,
        )
        if not all(torch.is_tensor(value) for value in required):
            raise ValueError("completed G3 state lost its N=49 precision candidates")
        assert state.dynamic_fine_values is not None
        if int(state.dynamic_fine_values.shape[-2]) != 49:
            raise ValueError("V120 P1 requires the complete N=49 candidate axis")

    @property
    def detail_features(self) -> Tensor:
        return self.grounding.detail_features

    @property
    def previous_detail_features(self) -> Tensor:
        return self.grounding.previous_detail_features

    @property
    def earlier_detail_features(self) -> Tensor | None:
        return self.grounding.earlier_detail_features

    @property
    def literal_rgb(self) -> Tensor:
        return self.grounding.literal_rgb

    @property
    def previous_literal_rgb(self) -> Tensor:
        return self.grounding.previous_literal_rgb

    @property
    def earlier_literal_rgb(self) -> Tensor:
        return self.grounding.earlier_literal_rgb

    @property
    def flow(self) -> PatchFlowField:
        return self.grounding.flow

    @property
    def earlier_flow(self) -> PatchFlowField:
        return self.grounding.earlier_flow

    @property
    def context_mask(self) -> Tensor:
        return self.grounding.context_mask

    @property
    def native_flow_losses(self) -> dict[str, Tensor] | None:
        return self.grounding.native_flow_losses


__all__ = [
    "GroundingObservationBank",
    "ObservationEvidence",
    "PatchFlowField",
    "_coordinate_grid",
    "_sample_feature_chart",
]
