"""V120 visual evidence compiler with the object-mainline typed boundary.

The previous independent mainline accidentally replaced the mature V120
Flow-DINO/raw-address path with a small local ConvGRU and a second set of G
blocks.  This module extracts the actual V120 observation compiler and adapts
its lossless camera/cell/local-slot address bank to :class:`LocalFactSet`.

The complete pre-G bank is compiled first.  The exact V120 progressive updater
then rematerializes the N=49 candidates after G2 and emits the final local fact
set after G3. Camera, 8x8 cell and M=4 axes are never pooled or recreated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import CurrentObservation
from ..v120_core.flow_dino_evidence import (
    FlowDINOEvidenceEncoder,
    FlowDINOEvidencePack,
    ProgressiveGroundingAddressState,
)
from ..v120_core.profile import build_v120_visual_config
from .observation_contract import (
    GroundingObservationBank,
    ObservationEvidence,
    PatchFlowField,
)
from .types import LocalFactSet

_NON_OWNER_V120_VISUAL_PARAMETERS = {
    # The object-intent export uses the G3 block output to drive the bounded
    # owner residual, but its completed GroundedFactSet deliberately does not
    # consume the parallel generic route query.  Keeping this one matrix out
    # of the optimizer is exact no-op behavior and avoids a dead trainable
    # parameter.  Do not broaden this list to active pre-G/Teacher modules.
    "progressive_grounding_address.query_projections.2.weight",
}


@dataclass(frozen=True)
class _PreparedV120Observation:
    observation: CurrentObservation
    pack: FlowDINOEvidencePack
    training_mask: bool


def _high_frequency(value: Tensor, grid: int) -> Tensor:
    """Match the V120 address compiler's fixed low-frequency subtraction."""

    batch, cameras, channels, height, width = value.shape
    flat = value.reshape(batch * cameras, channels, height, width).float()
    low = F.adaptive_avg_pool2d(flat, (grid, grid))
    low = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=True)
    return (flat - low).reshape_as(value).to(dtype=value.dtype)


def _v120_flow_field(pack: FlowDINOEvidencePack, index: int) -> PatchFlowField:
    """Adapt V120's source-indexed 8x8-chart flow to the mainline contract.

    V120 stores ``patch_flow_forward`` as previous-to-current displacement on
    previous/source cells and ``patch_flow_backward`` as current-to-previous
    displacement on current/source cells.  ``PatchFlowField`` deliberately
    uses destination indexing instead: its online ``forward`` is
    previous-to-current on current cells.  Swapping the two directed solves
    and negating them is therefore required; merely relabelling the tensors
    silently attaches motion to the wrong frame. Before the pack is returned,
    V120 explicitly rescales raw-flow pixels by
    ``(chart_side - 1) / (high_side - 1)``. The exported displacement is
    therefore already measured in 8x8 chart cells and converts to
    ``grid_sample`` coordinates with ``2 / (chart_side - 1)``.
    """

    forward_source = pack.patch_flow_forward[:, index]
    backward_source = pack.patch_flow_backward[:, index]
    if (
        forward_source.ndim != 5
        or tuple(forward_source.shape) != tuple(backward_source.shape)
        or int(forward_source.shape[-3]) != 2
        or int(forward_source.shape[-2]) != int(forward_source.shape[-1])
        or int(forward_source.shape[-1]) < 2
    ):
        raise ValueError("V120 exported patch flow must be a square [B,C,2,Y,X] chart")
    chart_side = int(forward_source.shape[-1])
    scale = 2.0 / float(chart_side - 1)
    forward = -backward_source * scale
    backward = -forward_source * scale
    confidence = pack.flow_confidence[:, index]
    occlusion = pack.flow_occlusion[:, index]
    uncertainty = torch.zeros_like(confidence)
    field = PatchFlowField(
        forward=forward,
        backward=backward,
        confidence=confidence,
        uncertainty=uncertainty,
        occlusion=occlusion,
        refinement_sequence=(forward,),
    )
    field.validate()
    return field


class RestoredV120ObservationCompiler(nn.Module):
    """Compile current evidence through the source-resolved V120 visual path."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.v120_config = build_v120_visual_config(config)
        self.encoder = FlowDINOEvidenceEncoder(self.v120_config)
        self.grid = int(config.observation.grid_size)
        if self.encoder.raw_flow is None:
            raise RuntimeError("the restored V120 compiler has no raw-flow pyramid")
        self.detail_dim = int(self.encoder.raw_flow.pyramid.high_channels)
        for name, parameter in self.encoder.named_parameters():
            if name in _NON_OWNER_V120_VISUAL_PARAMETERS:
                parameter.requires_grad_(False)

    @property
    def flow(self) -> nn.Module:
        """Expose the actual V120 SEA-RAFT core for profiling/tests."""

        return self.encoder.flow

    @torch.no_grad()
    def teacher_supports(self, tokens: Tensor) -> Tensor:
        return self.encoder.object_teacher_supports(tokens)

    def prepare(
        self,
        observation: CurrentObservation,
        *,
        context_mask: Tensor | None = None,
        training_mask: bool = False,
        geometry_supervision: bool = True,
        collect_diagnostics: bool = False,
    ) -> _PreparedV120Observation:
        del geometry_supervision  # V120 constructs both directed pair objectives together.
        observation.validate(self.config)
        if context_mask is not None:
            raise ValueError(
                "the restored V120 compiler owns its early mask; an arbitrary post-hoc "
                "context mask would not reproduce the masked computation"
            )

        # V120 used module.training to select its early mask.  The independent
        # API makes that decision explicit so diagnostic calls on a train-mode
        # model do not silently mask unless ``training_mask`` is requested.
        previous_training = self.encoder.training
        self.encoder.training = bool(training_mask)
        try:
            pack = self.encoder(
                observation.dino_history,
                raw_visual=observation.raw_rgb,
            )
        finally:
            self.encoder.training = previous_training
        return _PreparedV120Observation(
            observation=observation,
            pack=pack,
            training_mask=bool(training_mask),
        )

    def build_grounding_bank(
        self,
        prepared: _PreparedV120Observation,
        grounding_canvas: Tensor,
        slices: dict[str, slice],
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[GroundingObservationBank, dict[str, Tensor]]:
        """Compile the address bank from the real shared G canvas."""

        pack = prepared.pack
        observation = prepared.observation
        raw_context = pack.raw_context
        previous_training = self.encoder.training
        self.encoder.training = prepared.training_mask
        try:
            refined = self.encoder.refine_raw_evidence(
                pack,
                grounding_canvas,
                slices,
                return_late_detail=True,
            )
            if len(refined) != 4:
                raise RuntimeError("V120 late-detail request returned no detail boundary")
            visual_memory, visual_value_memory, refine_metrics, detail = refined
        finally:
            self.encoder.training = previous_training
        if detail is None or detail.address_bank is None:
            raise RuntimeError("the restored V120 path did not produce its soft address bank")
        bank = detail.address_bank
        patch_count = int(observation.dino_history.shape[-2])
        native_side = int(round(math.sqrt(patch_count)))
        if native_side * native_side != patch_count:
            raise ValueError("V120 DINO history patch count must form a square chart")
        assert bank.dense_target_detail is not None
        assert bank.dense_current_rgb is not None
        current_detail = bank.dense_target_detail
        if raw_context is not None:
            previous_detail = _high_frequency(raw_context.high_features[:, 0], self.grid)
        else:
            previous_detail = torch.zeros_like(current_detail)
        if tuple(previous_detail.shape) != tuple(current_detail.shape):
            previous_detail = F.interpolate(
                previous_detail.reshape(
                    -1,
                    previous_detail.shape[2],
                    previous_detail.shape[-2],
                    previous_detail.shape[-1],
                ),
                size=current_detail.shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).reshape_as(current_detail)
        literal = bank.dense_current_rgb
        previous_literal = 2.0 * observation.raw_rgb[:, -2].float() - 1.0
        earlier_literal = 2.0 * observation.raw_rgb[:, -3].float() - 1.0
        grounding = GroundingObservationBank(
            address_bank=bank,
            late_detail=detail,
            visual_memory=visual_memory,
            visual_value_memory=visual_value_memory,
            detail_features=current_detail,
            previous_detail_features=previous_detail,
            # V120's retained high-resolution cache contains current and
            # previous detail only.  Do not label the previous tensor as a
            # distinct earlier frame; the complete native two-pair flow loss
            # ledger below owns the real -8/-4/0 geometry supervision.
            earlier_detail_features=None,
            literal_rgb=literal,
            previous_literal_rgb=previous_literal.to(dtype=literal.dtype),
            earlier_literal_rgb=earlier_literal.to(dtype=literal.dtype),
            flow=_v120_flow_field(pack, -1),
            earlier_flow=_v120_flow_field(pack, 0),
            context_mask=pack.context_dropout_mask[:, -1],
            native_flow_losses=pack.losses,
        )
        grounding.validate()
        if not collect_diagnostics:
            return grounding, {}
        source_metrics = {**pack.metrics, **refine_metrics}
        aliases = {
            "flow_jepa_patch_flow_magnitude": "observation_flow_grid_cell_magnitude",
            "flow_jepa_native_flow_magnitude": "observation_flow_native_patch_magnitude",
            "flow_jepa_confidence_mean": "observation_flow_confidence",
            "flow_jepa_occlusion_fraction": "observation_flow_occlusion",
            "flow_jepa_correlation_entropy": "observation_flow_correlation_entropy",
            "flow_jepa_correlation_margin": "observation_flow_correlation_margin",
            "flow_jepa_detail_gate_mean": "observation_detail_gate_mean",
            "flow_jepa_detail_effective_comparisons": "observation_detail_effective_comparisons",
            "flow_jepa_address_flow_mass": "observation_address_flow_mass",
            "flow_jepa_address_fallback_mass": "observation_address_fallback_mass",
            "flow_jepa_address_entropy": "observation_address_entropy",
            "flow_jepa_raw_detail_emphasis_mean": "observation_raw_detail_emphasis",
            "flow_jepa_raw_detail_precision_mean": "observation_raw_detail_precision",
            "flow_jepa_raw_address_flow_mass": "observation_raw_address_flow_mass",
            "flow_jepa_raw_address_fallback_mass": "observation_raw_address_fallback_mass",
            "flow_jepa_raw_address_entropy": "observation_raw_address_entropy",
            "flow_jepa_address_coarse_variance_min": (
                "flow_jepa_address_coarse_variance_min"
            ),
            "flow_jepa_address_coarse_std_dino_rms": (
                "flow_jepa_address_coarse_std_dino_rms"
            ),
            "flow_jepa_address_coarse_std_gain_max": (
                "flow_jepa_address_coarse_std_gain_max"
            ),
        }
        semantic_source_metrics = {
            target: source_metrics[source]
            for source, target in aliases.items()
            if source in source_metrics
        }
        metrics = {
            **semantic_source_metrics,
            "observation_flow_dino_active": current_detail.new_ones((), dtype=torch.float32),
            "observation_soft_address_active": current_detail.new_ones((), dtype=torch.float32),
            "observation_detail_width": current_detail.new_tensor(float(self.detail_dim), dtype=torch.float32),
            "observation_context_mask_fraction": pack.context_dropout_mask[:, -1].detach().float().mean(),
        }
        return grounding, metrics

    def begin_progressive_grounding(
        self, bank: GroundingObservationBank
    ) -> ProgressiveGroundingAddressState:
        bank.validate()
        state = self.encoder.begin_progressive_grounding_address(bank.address_bank)
        if state is None:
            raise RuntimeError("V120 progressive G is disabled in the active profile")
        return state

    def advance_progressive_grounding(
        self,
        state: ProgressiveGroundingAddressState,
        rollout: Tensor,
        *,
        stage: int,
        collect_diagnostics: bool = False,
    ) -> ProgressiveGroundingAddressState:
        return self.encoder.update_progressive_grounding_address(
            state,
            rollout,
            stage=stage,
            collect_diagnostics=collect_diagnostics,
        )

    def finalize_grounding(
        self,
        bank: GroundingObservationBank,
        state: ProgressiveGroundingAddressState,
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[ObservationEvidence, dict[str, Tensor]]:
        """Adapt the literal V120 G3 fact set without recomputing candidates."""

        bank.validate()
        grounded = state.grounded_fact_set
        if state.stage != 3 or grounded is None:
            raise RuntimeError("G1/G2/G3 did not produce a completed fact set")
        grounded.validate()
        target = bank.address_bank.dense_current_dino_content
        if target is None:
            raise RuntimeError("completed G3 lost the current DINO target")
        owner_logs = (
            grounded.semantic_owner_log_probs,
            grounded.appearance_owner_log_probs,
            grounded.geometry_owner_log_probs,
        )
        if any(value is None for value in owner_logs):
            raise RuntimeError("completed G3 lost its FP32 owner log probabilities")
        local = LocalFactSet(
            public_scene_base=grounded.public_scene_base,
            target_dino_content=target.detach(),
            cell_observed=(~bank.context_mask)[..., None],
            content_slots=grounded.content_slots,
            semantic_slots=grounded.semantic_slots,
            appearance_slots=grounded.appearance_slots,
            geometry_slots=grounded.geometry_slots,
            semantic_owner_probs=grounded.semantic_owner_probs,
            appearance_owner_probs=grounded.appearance_owner_probs,
            geometry_owner_probs=grounded.geometry_owner_probs,
            semantic_owner_log_probs=grounded.semantic_owner_log_probs,
            appearance_owner_log_probs=grounded.appearance_owner_log_probs,
            geometry_owner_log_probs=grounded.geometry_owner_log_probs,
            slot_coordinates=grounded.slot_coordinates,
            slot_support=grounded.slot_support,
            slot_validity=grounded.slot_validity,
            slot_transport_prior=grounded.slot_transport_prior,
        )
        evidence = ObservationEvidence(
            grounding=bank,
            progressive_state=state,
            local_facts=local,
        )
        evidence.validate()
        if not collect_diagnostics:
            return evidence, {}
        source = dict(state.metrics or {})
        aliases = {
            "flow_jepa_progressive_g2_dynamic_candidate_valid": "observation_progressive_candidate_valid",
            "flow_jepa_progressive_g2_dynamic_center_distance": "observation_progressive_center_distance",
            "flow_jepa_progressive_current_coordinate_rms": "observation_progressive_coordinate_rms",
            "flow_jepa_progressive_literal_rgb_candidate_rms": "observation_progressive_literal_rgb_rms",
            "flow_jepa_progressive_g2_input_variance_min": "flow_jepa_progressive_g2_input_variance_min",
            "flow_jepa_progressive_g2_input_std_rms": "flow_jepa_progressive_g2_input_std_rms",
            "flow_jepa_progressive_g2_input_std_gain_max": "flow_jepa_progressive_g2_input_std_gain_max",
            "flow_jepa_progressive_g2_aligned_variance_min": "flow_jepa_progressive_g2_aligned_variance_min",
            "flow_jepa_progressive_g2_correction_scale_min": "flow_jepa_progressive_g2_correction_scale_min",
            "flow_jepa_progressive_g2_correction_std_gain_max": "flow_jepa_progressive_g2_correction_std_gain_max",
        }
        metrics = {target_name: source[name] for name, target_name in aliases.items() if name in source}
        metrics.update(
            {
                "observation_g1_g2_g3_completed": grounded.public_scene_base.new_ones(
                    (), dtype=torch.float32
                ),
                "observation_g3_parent_semantic_l1": source.get(
                    "grounded_g2_g3_semantic_owner_l1",
                    grounded.public_scene_base.new_zeros((), dtype=torch.float32),
                ),
            }
        )
        return evidence, metrics


__all__ = ["RestoredV120ObservationCompiler"]
