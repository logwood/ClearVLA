"""V120 visual evidence compiler with the object-mainline typed boundary.

The previous independent mainline accidentally replaced the mature V120
Flow-DINO/raw-address path with a small local ConvGRU and a second set of G
blocks.  This module extracts the actual V120 observation compiler and adapts
its lossless camera/cell/local-slot address bank to :class:`LocalFactSet`.

Only one new boundary exists here: the V120 49-point fine lattice is reduced
softly *inside each existing local slot*.  Camera, 8x8 cell and M=4 axes are
never pooled or recreated.  The global K objects and the sole three-block G
host remain owned by ``ObjectIntentDynamicsTop``.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import CurrentObservation
from ..v120_core.flow_dino_evidence import (
    FlowDINOEvidenceEncoder,
    FlowDINOEvidencePack,
    ProgressiveFineCandidates,
    SoftAddressLatticeBank,
)
from ..v120_core.profile import build_v120_visual_config
from .observation_contract import (
    ObservationEvidence,
    PatchFlowField,
    _sample_feature_chart,
)
from .types import LocalFactSet

_UNUSED_V120_VISUAL_PARAMETERS = {
    "history_type",
    "camera_type",
    "spatial_type",
    "evidence_type",
    "future_query",
    "future_anchor_type",
}

_UNUSED_V120_VISUAL_PREFIXES = (
    "motion_key.",
    "organized_key.",
    "early_masked_raw_context.",
    "future_motion.",
    "future_history_score.",
    "future_transition.",
    # The independent object top owns G1-G3.  Keeping this historical G stack
    # trainable would silently restore the duplicate-G failure.
    "progressive_grounding_address.",
)


def _masked_softmax(score: Tensor, valid: Tensor, *, dim: int) -> Tensor:
    """FP32 softmax with a real all-invalid zero state."""

    score = score.float().masked_fill(~valid, -1.0e4)
    probability = torch.softmax(score, dim=dim) * valid.float()
    return probability / probability.sum(dim=dim, keepdim=True).clamp_min(1.0)


def _slot_probability(score: Tensor, valid: Tensor) -> Tensor:
    """Convert fine-candidate evidence into a conditional M posterior."""

    fine_count = valid.float().sum(dim=-1)
    evidence = torch.logsumexp(score.float().masked_fill(~valid, -1.0e4), dim=-1)
    evidence = evidence - fine_count.clamp_min(1.0).log()
    slot_valid = valid.any(dim=-1)
    probability = torch.softmax(evidence.masked_fill(~slot_valid, -1.0e4), dim=-1)
    probability = probability * slot_valid.float()
    return probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0)


def _high_frequency(value: Tensor, grid: int) -> Tensor:
    """Match the V120 address compiler's fixed low-frequency subtraction."""

    batch, cameras, channels, height, width = value.shape
    flat = value.reshape(batch * cameras, channels, height, width).float()
    low = F.adaptive_avg_pool2d(flat, (grid, grid))
    low = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=True)
    return (flat - low).reshape_as(value).to(dtype=value.dtype)


def _align_chart_to_later_frame(value: Tensor, forward_flow: Tensor) -> Tensor:
    """Transport a source-indexed coarse chart onto the later frame.

    ``forward_flow`` is a previous-to-current displacement indexed on the
    *current* destination chart and expressed in normalized grid coordinates.
    An output coordinate therefore samples the source at ``x - flow``.  V120
    natively returns the opposite indexing convention; ``_v120_flow_field``
    performs that explicit conversion before this helper is called.
    """

    if value.ndim != 5 or forward_flow.ndim != 5:
        raise ValueError("coarse chart alignment requires [B,C,Y,X,H] and [B,C,2,Y,X]")
    batch, cameras, rows, columns, hidden = value.shape
    if rows != columns or tuple(forward_flow.shape) != (
        batch,
        cameras,
        2,
        rows,
        columns,
    ):
        raise ValueError("coarse chart and V120 flow grid do not align")
    axis = torch.linspace(-1.0, 1.0, rows, device=value.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].expand(batch * cameras, -1, -1, -1)
    displacement = forward_flow.float().reshape(
        batch * cameras, 2, rows, columns
    ).permute(0, 2, 3, 1)
    source = value.permute(0, 1, 4, 2, 3).reshape(
        batch * cameras, hidden, rows, columns
    )
    with torch.autocast(device_type=value.device.type, enabled=False):
        aligned = F.grid_sample(
            source.float(),
            base - displacement,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return aligned.reshape(batch, cameras, hidden, rows, columns).permute(
        0, 1, 3, 4, 2
    ).to(dtype=value.dtype)


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
        self.hidden = int(config.dimensions.hidden_size)
        self.content_dim = int(config.dimensions.visual_token_dim)
        self.route_dim = int(config.observation.address_route_dim)
        self.grid = int(config.observation.grid_size)
        self.cameras = int(config.dimensions.num_cameras)
        self.history = int(config.dimensions.visual_history_length)
        self.hypotheses = int(config.observation.local_hypotheses)
        if self.encoder.raw_flow is None:
            raise RuntimeError("the restored V120 compiler has no raw-flow pyramid")
        self.detail_dim = int(self.encoder.raw_flow.pyramid.high_channels)
        self._freeze_non_mainline_branches()

    @property
    def flow(self) -> nn.Module:
        """Expose the actual V120 SEA-RAFT core for profiling/tests."""

        return self.encoder.flow

    def _freeze_non_mainline_branches(self) -> None:
        """Freeze extracted heads whose consumer belonged to the old monolith."""

        for name, parameter in self.encoder.named_parameters():
            if name in _UNUSED_V120_VISUAL_PARAMETERS or name.startswith(
                _UNUSED_V120_VISUAL_PREFIXES
            ):
                parameter.requires_grad_(False)

    @staticmethod
    def _required_bank(bank: SoftAddressLatticeBank) -> None:
        required: Iterable[Tensor | None] = (
            bank.coarse_candidate_coordinates,
            bank.coarse_flow_centers,
            bank.coarse_source_centers,
            bank.dense_target_detail,
            bank.dense_current_rgb,
            bank.dense_current_dino_content,
        )
        if not all(torch.is_tensor(value) for value in required):
            raise RuntimeError("V120 address bank is missing an active mainline field")

    def _fine_candidates(
        self,
        bank: SoftAddressLatticeBank,
        *,
        collect_diagnostics: bool,
    ) -> tuple[ProgressiveFineCandidates, Tensor, Tensor, Tensor]:
        self._required_bank(bank)
        if self.encoder.soft_address_compiler is None:
            raise RuntimeError("V120 soft-address compiler is inactive")
        assert bank.coarse_candidate_coordinates is not None
        assert bank.dense_target_detail is not None
        dino_side = int(round(math.sqrt(bank.coarse_candidate_coordinates.shape[-2])))
        if dino_side * dino_side != int(bank.coarse_candidate_coordinates.shape[-2]):
            raise ValueError("V120 coarse address candidates do not form a square chart")
        raw_side = int(bank.dense_target_detail.shape[-1])
        centers = 2.0 * bank.coarse_centers.float() / float(max(dino_side - 1, 1)) - 1.0
        variance = bank.coarse_variance.float() * (
            2.0 / float(max(dino_side - 1, 1))
        ) ** 2
        support = 2.0 * bank.fine_radius.float() / float(max(raw_side - 1, 1))
        candidates = self.encoder.soft_address_compiler.progressive_fine_candidates(
            bank,
            centers=centers.to(dtype=bank.coarse_keys.dtype),
            support=support.to(dtype=bank.coarse_keys.dtype),
            variance=variance.to(dtype=bank.coarse_keys.dtype),
            aligned_keys=bank.coarse_keys,
            collect_diagnostics=collect_diagnostics,
        )
        if any(
            value is None
            for value in (
                candidates.semantic_keys,
                candidates.appearance_keys,
                candidates.geometry_keys,
            )
        ):
            raise RuntimeError("V120 progressive candidates lost typed evidence")
        return candidates, centers, variance, support

    def _local_facts(
        self,
        *,
        pack: FlowDINOEvidencePack,
        bank: SoftAddressLatticeBank,
        candidates: ProgressiveFineCandidates,
        support: Tensor,
    ) -> tuple[LocalFactSet, dict[str, Tensor]]:
        assert candidates.semantic_keys is not None
        assert candidates.appearance_keys is not None
        assert candidates.geometry_keys is not None
        assert bank.dense_current_dino_content is not None
        assert bank.coarse_flow_centers is not None
        assert bank.coarse_source_centers is not None
        compiler = self.encoder.soft_address_compiler
        if compiler is None:
            raise RuntimeError("typed local facts require the V120 address compiler")

        query = compiler.query_norm(bank.coarse_keys)
        typed_keys = {
            "semantic": candidates.semantic_keys,
            "appearance": candidates.appearance_keys,
            "geometry": candidates.geometry_keys,
        }
        scores: dict[str, Tensor] = {}
        probabilities: dict[str, Tensor] = {}
        owners: dict[str, Tensor] = {}
        for name, key in typed_keys.items():
            normalized_key = compiler.key_norm(key)
            score = torch.einsum(
                "bcyxmr,bcyxmnr->bcyxmn",
                query.float(),
                normalized_key.float(),
            ) / math.sqrt(float(self.route_dim))
            scores[name] = score
            probabilities[name] = _masked_softmax(
                score,
                candidates.valid,
                dim=-1,
            )
            owners[name] = _slot_probability(score, candidates.valid)

        physical_score = torch.stack(tuple(scores.values()), dim=0).sum(
            dim=0
        ) / math.sqrt(3.0)
        physical_probability = _masked_softmax(
            physical_score,
            candidates.valid,
            dim=-1,
        )
        coordinates = torch.einsum(
            "bcyxmn,bcyxmnd->bcyxmd",
            physical_probability,
            candidates.current_coordinates.float(),
        )
        slot_valid = candidates.valid.any(dim=-1)
        fallback = 2.0 * bank.coarse_centers.float()
        candidate_count = int(bank.coarse_base_logits.shape[-1]) if bank.coarse_base_logits is not None else 0
        dino_side = int(round(math.sqrt(candidate_count))) if candidate_count else self.grid
        fallback = fallback / float(max(dino_side - 1, 1)) - 1.0
        coordinates = torch.where(slot_valid[..., None], coordinates, fallback)

        typed_slots = {
            name: torch.einsum(
                "bcyxmn,bcyxmnr->bcyxmr",
                probabilities[name],
                typed_keys[name].float(),
            ).to(dtype=bank.coarse_keys.dtype)
            for name in typed_keys
        }
        dino_map = bank.dense_current_dino_content.permute(0, 1, 4, 2, 3)
        content, content_valid = _sample_feature_chart(
            dino_map,
            coordinates.to(dtype=dino_map.dtype),
        )
        slot_valid = slot_valid & content_valid[..., 0]

        value_tokens = pack.value_tokens
        content_rows = self.history * self.cameras * self.grid * self.grid
        if int(value_tokens.shape[1]) < content_rows:
            raise ValueError("V120 online evidence lost current content rows")
        content_history = value_tokens[:, :content_rows].reshape(
            value_tokens.shape[0],
            self.history,
            self.cameras,
            self.grid,
            self.grid,
            self.hidden,
        )
        motion_rows = (self.history - 1) * self.cameras * self.grid * self.grid
        motion_start = content_rows
        motion_stop = motion_start + motion_rows
        if int(value_tokens.shape[1]) < motion_stop:
            raise ValueError("V120 online evidence lost adjacent-motion rows")
        motion_history = value_tokens[:, motion_start:motion_stop].reshape(
            value_tokens.shape[0],
            self.history - 1,
            self.cameras,
            self.grid,
            self.grid,
            self.hidden,
        )
        if self.history != 3 or int(motion_history.shape[1]) != 2:
            raise ValueError("the restored V120 chart requires -8/-4/0 and two motions")
        earlier_flow = _v120_flow_field(pack, 0)
        recent_flow = _v120_flow_field(pack, -1)
        current_content = content_history[:, -1]
        previous_content = content_history[:, -2]
        earlier_content = content_history[:, -3]
        previous_on_current = _align_chart_to_later_frame(
            previous_content,
            recent_flow.forward,
        )
        earlier_on_previous = _align_chart_to_later_frame(
            earlier_content,
            earlier_flow.forward,
        )
        earlier_delta_on_current = _align_chart_to_later_frame(
            previous_content - earlier_on_previous,
            recent_flow.forward,
        )
        visible = (~pack.context_dropout_mask[:, -1])[..., None]
        visual_history_innovation = torch.where(
            visible,
            current_content - previous_on_current + 0.5 * earlier_delta_on_current,
            torch.zeros_like(current_content),
        )
        # V120 motion rows are indexed on the earlier/source frame of each
        # pair.  Put both rows on the current chart before they enter G; the
        # former adapter used a source-indexed row as if it already belonged
        # to the destination and skipped one of the two transports for -8/-4.
        recent_motion = _align_chart_to_later_frame(
            motion_history[:, -1],
            recent_flow.forward,
        )
        earlier_motion_on_previous = _align_chart_to_later_frame(
            motion_history[:, 0],
            earlier_flow.forward,
        )
        earlier_motion_on_current = _align_chart_to_later_frame(
            earlier_motion_on_previous,
            recent_flow.forward,
        )
        earlier_flow_on_current = _align_chart_to_later_frame(
            earlier_flow.forward.permute(0, 1, 3, 4, 2),
            recent_flow.forward,
        ).permute(0, 1, 4, 2, 3)
        flow_acceleration = recent_flow.forward.float() - earlier_flow_on_current.float()
        # V120 exposed all three content rows and both motion rows.  Preserve
        # that information at the single G public-address boundary: absolute
        # current content, flow-aligned visual change, recent motion and the
        # earlier motion transported into current coordinates.  The fixed
        # variance-preserving sum cannot learn to erase one of these sources.
        public = (
            current_content
            + visual_history_innovation
            + recent_motion
            + earlier_motion_on_current
        ) / 2.0
        context_mask = pack.context_dropout_mask[:, -1]
        transport = bank.coarse_flow_centers.float()[..., None, :] - (
            bank.coarse_source_centers.float()[..., None, :]
        )
        transport = transport.expand(-1, -1, -1, -1, self.hypotheses, -1)
        local = LocalFactSet(
            public_scene_base=public,
            target_dino_content=bank.dense_current_dino_content.detach(),
            cell_observed=(~context_mask)[..., None],
            content_slots=content,
            semantic_slots=typed_slots["semantic"],
            appearance_slots=typed_slots["appearance"],
            geometry_slots=typed_slots["geometry"],
            semantic_owner_probs=owners["semantic"].to(dtype=public.dtype),
            appearance_owner_probs=owners["appearance"].to(dtype=public.dtype),
            geometry_owner_probs=owners["geometry"].to(dtype=public.dtype),
            slot_coordinates=coordinates.to(dtype=public.dtype),
            slot_support=support.to(dtype=public.dtype),
            slot_validity=slot_valid[..., None].to(dtype=public.dtype),
            slot_transport_prior=transport.to(dtype=public.dtype),
        )
        local.validate()
        metrics = {
            "observation_typed_slot_valid_fraction": slot_valid.detach().float().mean(),
            "observation_typed_slot_coordinate_rms": coordinates.detach().float().square().mean().sqrt(),
            "observation_typed_semantic_owner_max": owners["semantic"].detach().float().amax(dim=-1).mean(),
            "observation_typed_appearance_owner_max": owners["appearance"].detach().float().amax(dim=-1).mean(),
            "observation_typed_geometry_owner_max": owners["geometry"].detach().float().amax(dim=-1).mean(),
            "observation_typed_owner_pair_gap": (
                (owners["semantic"] - owners["appearance"]).abs().mean()
                + (owners["semantic"] - owners["geometry"]).abs().mean()
                + (owners["appearance"] - owners["geometry"]).abs().mean()
            ) / 3.0,
            "observation_visual_history_innovation_rms": (
                visual_history_innovation.detach().float().square().mean().sqrt()
            ),
            "observation_recent_motion_rms": (
                recent_motion.detach().float().square().mean().sqrt()
            ),
            "observation_earlier_motion_aligned_rms": (
                earlier_motion_on_current.detach().float().square().mean().sqrt()
            ),
            "observation_retained_motion_pairs": public.new_tensor(
                2.0, dtype=torch.float32
            ),
            "observation_flow_rms": recent_flow.forward.detach().float().square().mean().sqrt(),
            "observation_earlier_flow_rms": earlier_flow.forward.detach().float().square().mean().sqrt(),
            "observation_flow_acceleration_rms": flow_acceleration.detach().square().mean().sqrt(),
        }
        return local, metrics

    @torch.no_grad()
    def teacher_supports(self, tokens: Tensor) -> Tensor:
        return self.encoder.object_teacher_supports(tokens)

    def forward(
        self,
        observation: CurrentObservation,
        *,
        context_mask: Tensor | None = None,
        training_mask: bool = False,
        geometry_supervision: bool = True,
        collect_diagnostics: bool = False,
    ) -> tuple[ObservationEvidence, dict[str, Tensor]]:
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
            raw_context = pack.raw_context
            grounding = pack.value_tokens.new_zeros(
                pack.value_tokens.shape[0],
                int(self.v120_config.future_token_count),
                self.hidden,
            )
            refined = self.encoder.refine_raw_evidence(
                pack,
                grounding,
                {"rollout": slice(0, grounding.shape[1])},
                return_late_detail=True,
            )
            if len(refined) != 4:
                raise RuntimeError("V120 late-detail request returned no detail boundary")
            _, _, refine_metrics, detail = refined
        finally:
            self.encoder.training = previous_training
        if detail is None or detail.address_bank is None:
            raise RuntimeError("the restored V120 path did not produce its soft address bank")
        bank = detail.address_bank
        patch_count = int(observation.dino_history.shape[-2])
        native_side = int(round(math.sqrt(patch_count)))
        if native_side * native_side != patch_count:
            raise ValueError("V120 DINO history patch count must form a square chart")
        candidates, _, _, support = self._fine_candidates(
            bank,
            collect_diagnostics=collect_diagnostics,
        )
        local, adapter_metrics = self._local_facts(
            pack=pack,
            bank=bank,
            candidates=candidates,
            support=support,
        )
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
        earlier_detail = previous_detail
        literal = bank.dense_current_rgb
        previous_literal = 2.0 * observation.raw_rgb[:, -2].float() - 1.0
        earlier_literal = 2.0 * observation.raw_rgb[:, -3].float() - 1.0
        evidence = ObservationEvidence(
            local_facts=local,
            detail_features=current_detail,
            previous_detail_features=previous_detail,
            earlier_detail_features=earlier_detail,
            literal_rgb=literal,
            previous_literal_rgb=previous_literal.to(dtype=literal.dtype),
            earlier_literal_rgb=earlier_literal.to(dtype=literal.dtype),
            flow=_v120_flow_field(pack, -1),
            earlier_flow=_v120_flow_field(pack, 0),
            context_mask=pack.context_dropout_mask[:, -1],
            native_flow_losses=pack.losses,
        )
        evidence.validate()
        if not collect_diagnostics:
            return evidence, {}
        source_metrics = {**pack.metrics, **refine_metrics, **candidates.metrics}
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
            "flow_jepa_progressive_g2_dynamic_candidate_valid": "observation_progressive_candidate_valid",
            "flow_jepa_progressive_g2_dynamic_center_distance": "observation_progressive_center_distance",
            "flow_jepa_progressive_current_coordinate_rms": "observation_progressive_coordinate_rms",
            "flow_jepa_progressive_literal_rgb_candidate_rms": "observation_progressive_literal_rgb_rms",
        }
        semantic_source_metrics = {
            target: source_metrics[source]
            for source, target in aliases.items()
            if source in source_metrics
        }
        metrics = {
            **semantic_source_metrics,
            **adapter_metrics,
            "observation_flow_dino_active": current_detail.new_ones((), dtype=torch.float32),
            "observation_soft_address_active": current_detail.new_ones((), dtype=torch.float32),
            "observation_detail_width": current_detail.new_tensor(float(self.detail_dim), dtype=torch.float32),
            "observation_context_mask_fraction": pack.context_dropout_mask[:, -1].detach().float().mean(),
        }
        return evidence, metrics


__all__ = ["RestoredV120ObservationCompiler"]
