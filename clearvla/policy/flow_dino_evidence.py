"""Flow/DINO/raw-image evidence and future-only JEPA targets.

The legacy V95/V96 motion estimator is a patch-grid adaptation of SEA-RAFT
over cached DINO features.  The raw-grounding path retains a learned RGB
pyramid at 1/4 resolution, seeds it with an identity-safe DINO correspondence,
then learns 1/8 and 1/4 RGB residual flow.  The resulting flow is a continuous
address prior for a late raw-detail read.  Both paths keep the parts that
matter for this policy:

* global correlation and a direct differentiable initial correspondence;
* a multi-level correlation lookup around the current coordinates;
* shared iterative ConvNeXt refinement of flow and uncertainty;
* forward/backward consistency, differentiable warping, and bounded geometry.

The implementation is mechanism-compatible with the BSD-3-Clause SEA-RAFT
release (https://github.com/princeton-vl/SEA-RAFT), uses only stock PyTorch
operators, and copies neither pretrained weights nor repository code.

Important masking contract: cached final-layer DINO tokens have already mixed
same-frame patches through DINO attention.  Consequently, history masking here
is named context dropout, not I-JEPA masking.  The actual JEPA boundary is
future-only: future DINO never enters :meth:`forward`; a learned future query is
created before the DiT, and a separate no-grad teacher pack selects future
positions for supervision.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .differential_intent_effect import (
    DifferentialWindowEffectBank,
    DifferentialWindowRouteCompiler,
    IntentWindowView,
)
from .grounded_intent_effect import (
    FutureEffectField as GroundedFutureEffectField,
)
from .grounded_intent_effect import (
    GroundedFactSet,
    GroundedWorldEffectCompiler,
    GroundedWorldWorkingState,
    bounded_owner_update,
    sample_spatial_slots,
)
from .grounded_intent_effect import (
    StatelessIntentState as GroundedIntentState,
)
from .role_delta_attnres import (
    AffineVarianceFlooredCenteredNorm,
    RoleDeltaAttnRes,
    VarianceFlooredCenteredNorm,
    rms_floored_l2_normalize,
    smooth_rms_contract,
    variance_floored_centered_norm,
)


@dataclass
class PatchFlowEstimate:
    """Bidirectional-ready semantic patch correspondence for one frame pair."""

    flow: Tensor
    information: Tensor
    uncertainty: Tensor
    correlation_entropy: Tensor
    correlation_margin: Tensor
    iterations: tuple[Tensor, ...]
    boundary_compression: Tensor | None = None
    correlation_feature_rms_min: Tensor | None = None
    correlation_norm_denominator_min: Tensor | None = None
    correlation_norm_gain_max: Tensor | None = None


@dataclass
class RawGroundingContext:
    """High-resolution observed RGB state retained until grounding is mature."""

    high_features: Tensor
    flow_forward: Tensor
    flow_backward: Tensor
    confidence: Tensor
    occlusion: Tensor
    uncertainty: Tensor
    correlation_entropy: Tensor
    correlation_margin: Tensor
    cycle_error: Tensor
    warp_error: Tensor
    observable_motion: Tensor
    # Native latest-pair DINO charts, retained only for the opt-in soft
    # address lattice. Shape: [B,2,camera,S,S,D].
    dino_features: Tensor | None = None
    # Literal latest observed RGB pair.  This remains observation-only and is
    # retained only for the coordinate-typed V110 precision bank.  It is never
    # replaced by a future teacher or an action-conditioned crop.
    # Shape: [B,2,camera,3,R,R].
    raw_rgb_pair: Tensor | None = None


@dataclass
class SoftAddressLatticeBank:
    """Observation-only multi-slot geometry and low-width precision values.

    The bank is safe to reuse across ODE steps.  It contains no trajectory,
    policy query, future teacher, or final address posterior.  Query-dependent
    fine/camera selection happens in the policy reader.
    """

    coarse_keys: Tensor
    fine_keys: Tensor
    fine_values: Tensor
    fine_valid: Tensor
    coarse_centers: Tensor
    coarse_variance: Tensor
    fine_radius: Tensor
    # Optional V109 observation scaffold. These retain selector evidence and
    # geometry only; no policy query, future teacher, or high-resolution value
    # posterior is cached here.
    coarse_base_logits: Tensor | None = None
    coarse_candidate_keys: Tensor | None = None
    coarse_candidate_coordinates: Tensor | None = None
    coarse_flow_centers: Tensor | None = None
    coarse_confidence: Tensor | None = None
    coarse_uncertainty: Tensor | None = None
    coarse_occlusion: Tensor | None = None
    coarse_cycle_error: Tensor | None = None
    fine_coordinates: Tensor | None = None
    coarse_source_centers: Tensor | None = None
    dense_source_raw_keys: Tensor | None = None
    dense_target_raw_keys: Tensor | None = None
    dense_target_dino_keys: Tensor | None = None
    dense_target_detail: Tensor | None = None
    dense_confidence: Tensor | None = None
    dense_uncertainty: Tensor | None = None
    dense_occlusion: Tensor | None = None
    # V110 bounded literal current RGB chart in [-1,1].  Coordinates remain
    # normalized image coordinates, so this chart can be sampled without a
    # fixed 8x8 -> patch ownership assumption.
    dense_current_rgb: Tensor | None = None
    # Frozen target-space projection of the current DINO chart.  This is
    # observation-only and sampled once by G3 at its continuous object-slot
    # coordinates.  It prevents W from learning a free current-reference head.
    dense_current_dino_content: Tensor | None = None


@dataclass
class ProgressiveFineCandidates:
    """Typed G2 candidates materialized at exact current-image coordinates.

    ``combined_keys`` preserves the V109 compatibility interface.  V110 never
    treats it as the source of truth: semantic, appearance and geometry keys,
    literal RGB, learned detail and coordinates remain independently available
    until P2.
    """

    combined_keys: Tensor
    learned_detail: Tensor
    valid: Tensor
    current_coordinates: Tensor
    metrics: dict[str, Tensor]
    semantic_keys: Tensor | None = None
    appearance_keys: Tensor | None = None
    geometry_keys: Tensor | None = None
    literal_rgb: Tensor | None = None
    source_coordinates: Tensor | None = None


@dataclass(frozen=True)
class FutureTeacherTrackPack:
    """Loss-only, G-aligned future consequences for current canonical slots."""

    stable_successor: Tensor
    semantic_delta: Tensor
    endpoint_delta: Tensor
    transport_mean: Tensor
    transport_covariance: Tensor
    path_envelope: Tensor
    persistence: Tensor
    visibility: Tensor
    uncertainty: Tensor
    reliability: Tensor
    association_entropy: Tensor
    semantic_advantage: Tensor
    effective_support: Tensor
    support_count: Tensor
    current_content: Tensor

    def validate(self) -> None:
        if self.stable_successor.ndim != 7:
            raise ValueError(
                "future teacher successor must be [B,A,C,G,G,M,H]"
            )
        prefix = tuple(self.stable_successor.shape[:-1])
        hidden = int(self.stable_successor.shape[-1])
        expected = {
            "semantic_delta": (*prefix, hidden),
            "endpoint_delta": (*prefix, hidden),
            "transport_mean": (*prefix, 2),
            "transport_covariance": (*prefix, 3),
            "path_envelope": (*prefix, 1),
            "persistence": (*prefix, 1),
            "visibility": (*prefix, 1),
            "uncertainty": (*prefix, 1),
            "reliability": (*prefix, 1),
            "association_entropy": (*prefix, 1),
            "semantic_advantage": (*prefix, 1),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"future teacher {name} must be {shape}, "
                    f"got {tuple(value.shape)}"
                )
        current_shape = (
            int(prefix[0]),
            int(prefix[2]),
            int(prefix[3]),
            int(prefix[4]),
            int(prefix[5]),
            hidden,
        )
        if tuple(self.current_content.shape) != current_shape:
            raise ValueError(
                "future teacher current content does not align with G slots"
            )

    def slot_reduced(
        self,
        slot_weights: Tensor,
    ) -> dict[str, Tensor]:
        """Expose legacy camera/cell targets without restoring cell identity."""

        self.validate()
        batch, anchors, cameras, grid_y, grid_x, slots, _ = (
            self.stable_successor.shape
        )
        if tuple(slot_weights.shape) != (
            batch,
            cameras,
            grid_y,
            grid_x,
            slots,
        ):
            raise ValueError(
                "future teacher slot weights do not align with current G3"
            )
        weights = slot_weights.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights[:, None, ..., None]

        def reduce(value: Tensor) -> Tensor:
            return (value.float() * weights).sum(dim=-2)

        def flatten(value: Tensor) -> Tensor:
            return value.reshape(
                batch,
                anchors * cameras * grid_y * grid_x,
                int(value.shape[-1]),
            ).detach()

        current_weights = weights[:, 0]
        current = (
            self.current_content.float() * current_weights
        ).sum(dim=-2).reshape(
            batch,
            cameras * grid_y * grid_x,
            int(self.current_content.shape[-1]),
        )
        return {
            "flow_jepa_future_target": flatten(
                reduce(self.stable_successor)
            ),
            "flow_jepa_interval_progress_target": flatten(
                reduce(self.semantic_delta)
            ),
            "flow_jepa_interval_endpoint_target": flatten(
                reduce(self.endpoint_delta)
            ),
            "flow_jepa_interval_current_target": current.detach(),
            "flow_jepa_future_transport_target": flatten(
                reduce(self.transport_mean)
            ),
            "flow_jepa_future_transport_covariance_target": flatten(
                reduce(self.transport_covariance)
            ),
            "flow_jepa_future_persistence_target": flatten(
                reduce(self.persistence)
            ),
            "flow_jepa_future_visibility_target": flatten(
                reduce(self.visibility)
            ),
            "flow_jepa_future_uncertainty_target": flatten(
                reduce(self.uncertainty)
            ),
            "flow_jepa_future_reliability": flatten(
                reduce(self.reliability)
            ),
            "flow_jepa_future_association_entropy": flatten(
                reduce(self.association_entropy)
            ),
            "flow_jepa_future_semantic_advantage": flatten(
                reduce(self.semantic_advantage)
            ),
            "flow_jepa_interval_effective_support": (
                self.effective_support.detach()
            ),
            "flow_jepa_interval_support_count": self.support_count.detach(),
        }


@dataclass(frozen=True)
class WindowTeacherTrackPack(FutureTeacherTrackPack):
    """V117 loss-only near/mid/late teacher with fixed slot ownership."""

    slot_names: tuple[str, str, str] = ("near", "mid", "late")

    def validate(self) -> None:
        super().validate()
        if int(self.stable_successor.shape[1]) != 3:
            raise ValueError("window teacher must contain near/mid/late slots")


@dataclass(frozen=True)
class FutureEffectField:
    """The one online W effect state supervised and consumed downstream."""

    semantic_delta: Tensor
    transport_mean: Tensor
    transport_covariance: Tensor
    persistence: Tensor
    visibility: Tensor
    uncertainty: Tensor
    # V115 ancestry only. V116 forbids an unsupervised route-width carrier at
    # P ingress and instead exposes current/successor content with direct
    # teacher ownership.
    state_innovation: Tensor | None = None
    current_content: Tensor | None = None
    successor_content: Tensor | None = None

    def validate(self) -> None:
        if self.semantic_delta.ndim != 7:
            raise ValueError(
                "future effect field must preserve [B,A,C,G,G,M,H]"
            )
        prefix = tuple(self.semantic_delta.shape[:-1])
        expected = {
            "transport_mean": (*prefix, 2),
            "transport_covariance": (*prefix, 3),
            "persistence": (*prefix, 1),
            "visibility": (*prefix, 1),
            "uncertainty": (*prefix, 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"future effect field {name} is misaligned")
        if self.state_innovation is not None and tuple(
            self.state_innovation.shape[:-1]
        ) != prefix:
            raise ValueError("future effect state innovation is misaligned")
        supervised = (
            self.current_content is not None
            and self.successor_content is not None
        )
        if supervised:
            for name in ("current_content", "successor_content"):
                if tuple(getattr(self, name).shape) != tuple(
                    self.semantic_delta.shape
                ):
                    raise ValueError(
                        f"future effect field {name} is misaligned"
                    )
            if self.state_innovation is not None:
                raise ValueError(
                    "supervised future effect cannot carry state_innovation"
                )
        elif self.state_innovation is None:
            raise ValueError(
                "future effect requires supervised content or legacy state innovation"
            )


@dataclass(frozen=True)
class WindowEffectBank(FutureEffectField):
    """The sole V117 W->P object with near/mid/late spatial effects."""

    slot_valid: Tensor | None = None
    slot_names: tuple[str, str, str] = ("near", "mid", "late")

    def validate(self) -> None:
        super().validate()
        if int(self.semantic_delta.shape[1]) != 3:
            raise ValueError("window effect bank must contain near/mid/late slots")
        if self.slot_valid is None or tuple(self.slot_valid.shape) != (3,):
            raise ValueError("window effect slot_valid must be present as [3]")
        if not bool(torch.isfinite(self.slot_valid).all()):
            raise ValueError("window effect slot_valid is non-finite")


@dataclass
class ProgressiveGroundingAddressState:
    """Query-owned G1/G2/G3 selector state over an observation-only bank."""

    bank: SoftAddressLatticeBank
    stage: int = 0
    coarse_logits: Tensor | None = None
    coarse_probability: Tensor | None = None
    aligned_centers: Tensor | None = None
    aligned_variance: Tensor | None = None
    aligned_keys: Tensor | None = None
    fine_logits: Tensor | None = None
    fine_probability: Tensor | None = None
    rectified_centers: Tensor | None = None
    rectified_support: Tensor | None = None
    rectified_keys: Tensor | None = None
    dynamic_fine_keys: Tensor | None = None
    dynamic_fine_values: Tensor | None = None
    dynamic_fine_valid: Tensor | None = None
    dynamic_fine_coordinates: Tensor | None = None
    dynamic_source_coordinates: Tensor | None = None
    dynamic_semantic_keys: Tensor | None = None
    dynamic_appearance_keys: Tensor | None = None
    dynamic_geometry_keys: Tensor | None = None
    dynamic_literal_rgb: Tensor | None = None
    # V111 G2 responsibilities.  Semantic hypotheses, appearance
    # verification and geometric rectification are distinct posteriors over
    # the same lossless candidate set; only the appearance+geometry posterior
    # becomes the precision fine prior.
    g2_semantic_probability: Tensor | None = None
    g2_appearance_probability: Tensor | None = None
    g2_geometry_probability: Tensor | None = None
    # Grounded mainline: owner probability across the object-slot axis.  These
    # are distinct from the within-slot fine-candidate posterior above.
    g2_semantic_slot_probability: Tensor | None = None
    g2_appearance_slot_probability: Tensor | None = None
    g2_geometry_slot_probability: Tensor | None = None
    canonical_coarse_bias: Tensor | None = None
    canonical_fine_bias: Tensor | None = None
    canonical_slot_keys: Tensor | None = None
    canonical_semantic_keys: Tensor | None = None
    canonical_appearance_keys: Tensor | None = None
    canonical_geometry_keys: Tensor | None = None
    canonical_summary_tokens: Tensor | None = None
    canonical_semantic_slot_weights: Tensor | None = None
    canonical_appearance_slot_weights: Tensor | None = None
    canonical_geometry_slot_weights: Tensor | None = None
    grounded_fact_set: GroundedFactSet | None = None
    # W-owned, horizon-specific selector state.  The teacher-facing relevance
    # chart and the source-state bias are two marginals of the same W/G3
    # compatibility tensor; only the latter is consumed by the P value read.
    world_teacher_relevance_logits: Tensor | None = None
    world_source_bias: Tensor | None = None
    world_semantic_source_bias: Tensor | None = None
    world_appearance_source_bias: Tensor | None = None
    world_geometry_source_bias: Tensor | None = None
    world_public_query: Tensor | None = None
    world_horizon_innovation: Tensor | None = None
    # Post-V111 pre-value ownership state.  These route-width charts remain
    # separate through every configured W block and are fused only as a
    # bounded write into the
    # shared rollout carrier.  The final states also parameterize the W/P
    # selector posterior; none of them contains a raw/DINO value read.
    world_semantic_state: Tensor | None = None
    world_appearance_state: Tensor | None = None
    world_geometry_state: Tensor | None = None
    world_interval_state: Tensor | None = None
    # V113: the hidden-width interval candidate presented to the same typed
    # W router that writes the online carrier.  The interval teacher supervises
    # this tensor directly; it is not produced by a post-W3 prediction sidecar.
    world_interval_progress_prediction: Tensor | None = None
    world_future_effect_w1_field: FutureEffectField | None = None
    world_future_effect_field: FutureEffectField | None = None
    # V117 compact private transition state. It never crosses W->P; P sees
    # only ``world_future_effect_field`` after the supervised decoders.
    world_window_effect_route_state: Tensor | None = None
    # Differential intent/effect mainline.  The current reference is stored
    # once and only slotwise changes cross W->P.
    world_differential_effect_w1_field: DifferentialWindowEffectBank | None = None
    world_differential_effect_field: DifferentialWindowEffectBank | None = None
    world_differential_effect_route_state: Tensor | None = None
    world_grounded_working_state: GroundedWorldWorkingState | None = None
    world_grounded_effect_w1_field: GroundedFutureEffectField | None = None
    world_grounded_effect_field: GroundedFutureEffectField | None = None
    world_appearance_fine_query: Tensor | None = None
    world_owner_depth: int = -1
    world_interval_offset_delta: Tensor | None = None
    world_interval_log_scale_delta: Tensor | None = None
    world_future_offset: Tensor | None = None
    world_future_scale: Tensor | None = None
    world_future_visibility: Tensor | None = None
    world_future_uncertainty: Tensor | None = None
    world_future_centers: Tensor | None = None
    metrics: dict[str, Tensor] | None = None


_GROUNDED_G3_SLOT_INTERVENTIONS = {
    "address_g3_slot_permute",
    "address_g3_slot_mean",
}


def _intervene_grounded_fact_slots(
    facts: GroundedFactSet,
    mode: str,
) -> tuple[GroundedFactSet, Tensor]:
    """Change only G3's object sidecar while preserving the public/P1 base.

    The intervention is deliberately narrower than ``address_g3_zero``.  It
    leaves ``public_scene_base`` and the completed P1 address lattice intact,
    so a deployed-action change cannot be attributed to damaging the sole
    high-resolution factual read.  ``slot_permute`` reindexes every object
    field consistently within each sample/camera/cell; ``slot_mean`` removes
    only object-slot distinctions while preserving each field's slot mean.
    """

    normalized = str(mode).strip().lower()
    if normalized not in _GROUNDED_G3_SLOT_INTERVENTIONS:
        raise ValueError(f"unsupported grounded G3 slot intervention: {mode}")
    facts.validate()

    def changed(value: Tensor, *, slot_dim: int) -> Tensor:
        resolved_dim = slot_dim if slot_dim >= 0 else value.ndim + slot_dim
        if int(value.shape[resolved_dim]) <= 1:
            return value
        if normalized == "address_g3_slot_permute":
            return value.roll(shifts=1, dims=resolved_dim)
        mean = value.float().mean(dim=resolved_dim, keepdim=True)
        return mean.to(dtype=value.dtype).expand_as(value)

    intervened = replace(
        facts,
        content_slots=changed(facts.content_slots, slot_dim=-2),
        semantic_slots=changed(facts.semantic_slots, slot_dim=-2),
        appearance_slots=changed(facts.appearance_slots, slot_dim=-2),
        geometry_slots=changed(facts.geometry_slots, slot_dim=-2),
        semantic_owner_probs=changed(
            facts.semantic_owner_probs,
            slot_dim=-1,
        ),
        appearance_owner_probs=changed(
            facts.appearance_owner_probs,
            slot_dim=-1,
        ),
        geometry_owner_probs=changed(
            facts.geometry_owner_probs,
            slot_dim=-1,
        ),
        slot_coordinates=changed(facts.slot_coordinates, slot_dim=-2),
        slot_support=changed(facts.slot_support, slot_dim=-1),
        slot_validity=changed(facts.slot_validity, slot_dim=-2),
        slot_transport_prior=(
            changed(facts.slot_transport_prior, slot_dim=-2)
            if facts.slot_transport_prior is not None
            else None
        ),
    )
    intervened.validate()
    delta_rows = tuple(
        (
            getattr(intervened, name).detach().float()
            - getattr(facts, name).detach().float()
        )
        .square()
        .mean()
        .sqrt()
        for name in (
            "content_slots",
            "semantic_slots",
            "appearance_slots",
            "geometry_slots",
            "semantic_owner_probs",
            "appearance_owner_probs",
            "geometry_owner_probs",
            "slot_coordinates",
            "slot_support",
            "slot_validity",
        )
    )
    return intervened, torch.stack(delta_rows).mean()


@dataclass
class LateRawDetailEvidence:
    """Flow-addressed high-frequency evidence retained for the policy boundary.

    The selector carries camera/spatial identity plus the exact post-reader
    high-frequency selector residual.  The value is only the corresponding
    high-frequency value residual; low-frequency DINO/world content remains on
    its own route and is not duplicated here.
    """

    selector_tokens: Tensor
    value_tokens: Tensor
    address_bank: SoftAddressLatticeBank | None = None
    progressive_address: ProgressiveGroundingAddressState | None = None


@dataclass
class FlowDINOEvidencePack:
    """Online-only evidence.  It intentionally contains no future feature."""

    selector_tokens: Tensor
    value_tokens: Tensor
    key_bias: Tensor
    stage_query: Tensor
    future_queries: Tensor
    context_dropout_mask: Tensor
    future_target_mask: Tensor
    patch_flow_forward: Tensor
    patch_flow_backward: Tensor
    flow_confidence: Tensor
    flow_occlusion: Tensor
    losses: dict[str, Tensor]
    metrics: dict[str, Tensor]
    raw_context: RawGroundingContext | None = None
    late_raw_detail: LateRawDetailEvidence | None = None
    late_raw_detail_metrics: dict[str, Tensor] | None = None


def _sinusoidal_offsets(offsets: tuple[int, ...], hidden: int) -> Tensor:
    """Encode real frame offsets without treating anchor ordinal as time."""

    if not offsets or hidden < 1:
        raise ValueError("horizon encoding requires offsets and a positive hidden size")
    position = torch.as_tensor(offsets, dtype=torch.float32)[:, None]
    pair_dim = max((hidden + 1) // 2, 1)
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(pair_dim, dtype=torch.float32)
        / float(max(pair_dim - 1, 1))
    )[None]
    phase = position * frequency
    encoded = torch.cat((phase.sin(), phase.cos()), dim=-1)[:, :hidden]
    if int(encoded.shape[-1]) < hidden:
        encoded = F.pad(encoded, (0, hidden - int(encoded.shape[-1])))
    return encoded


def _grid_coordinates(
    batch: int, height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0)[None].expand(batch, -1, -1, -1)


def _normalize_grid(coordinates: Tensor, height: int, width: int) -> Tensor:
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    x = 2.0 * x / float(max(width - 1, 1)) - 1.0
    y = 2.0 * y / float(max(height - 1, 1)) - 1.0
    return torch.stack((x, y), dim=-1)


def _smooth_bound_flow_to_image(flow: Tensor) -> tuple[Tensor, Tensor]:
    """Represent displacement in a smooth source-relative in-image chart.

    A validity mask is useful for genuine sampling boundaries, but it must not
    let a learned flow erase its own photometric/cycle supervision by sending
    every coordinate off screen.  Each signed displacement is therefore
    parameterized against the available distance from its source cell to the
    corresponding image boundary.  ``tanh`` is identity-like near zero,
    asymptotically approaches the legal boundary, and keeps ordinary autograd
    throughout; this is neither a hard clamp nor a discrete route.

    The second return value is a per-cell relative compression diagnostic.
    It is factual only and is never used as a loss or gate.
    """

    if flow.ndim != 4 or int(flow.shape[1]) != 2:
        raise ValueError("bounded flow must be [B,2,H,W]")
    batch, _, height, width = flow.shape
    base = _grid_coordinates(
        batch,
        int(height),
        int(width),
        device=flow.device,
        dtype=torch.float32,
    )
    maximum = flow.new_tensor(
        (float(max(int(width) - 1, 0)), float(max(int(height) - 1, 0))),
        dtype=torch.float32,
    )[None, :, None, None]
    positive_limit = (maximum - base).clamp_min(0.0)
    negative_limit = base.clamp_min(0.0)
    proposal = flow.float()

    def signed_chart(value: Tensor, limit: Tensor) -> Tensor:
        safe_limit = limit.clamp_min(1e-6)
        return limit * torch.tanh(value / safe_limit)

    bounded_positive = signed_chart(proposal, positive_limit)
    bounded_negative = signed_chart(proposal, negative_limit)
    bounded = torch.where(proposal >= 0.0, bounded_positive, bounded_negative)
    change = _stable_vector_norm(proposal - bounded, dim=1, keepdim=True)
    magnitude = _stable_vector_norm(proposal, dim=1, keepdim=True)
    compression = change / magnitude.clamp_min(1e-6)
    compression = torch.where(
        magnitude > 1e-6, compression, torch.zeros_like(compression)
    )
    return bounded.to(dtype=flow.dtype), compression


def _normalize_flow_evidence(flow: Tensor) -> Tensor:
    """Put flow evidence in a resolution-independent bounded coordinate chart."""

    if flow.ndim < 4 or int(flow.shape[-3]) != 2:
        raise ValueError("flow evidence must have a two-channel displacement axis")
    height = int(flow.shape[-2])
    width = int(flow.shape[-1])
    scale = flow.new_tensor(
        (float(max(width - 1, 1)), float(max(height - 1, 1))),
        dtype=torch.float32,
    )
    view = (1,) * (flow.ndim - 3) + (2, 1, 1)
    return flow.float() / scale.reshape(view)


def _stable_sqrt(value: Tensor, *, epsilon: float = 1e-12) -> Tensor:
    """Square root with a bounded zero-point derivative and exact zero value.

    ``sqrt(x)`` is finite at ``x == 0`` but its derivative is not. Flow
    consistency, confidence, and routing all encounter exact zeros during
    identity motion, so a direct square root can leave every forward metric
    finite while producing Inf/NaN gradients during backward. This smooth
    form preserves the semantic zero and has a finite derivative there.
    """

    value_f = value.float().clamp_min(0.0)
    epsilon_f = float(epsilon)
    root = torch.sqrt(value_f.clamp_min(epsilon_f))
    linear = value_f / math.sqrt(epsilon_f)
    return torch.where(value_f < epsilon_f, linear, root)


def _stable_vector_norm(
    value: Tensor, *, dim: int, keepdim: bool = False, epsilon: float = 1e-12
) -> Tensor:
    """FP32 Euclidean magnitude with a finite subgradient at zero."""

    squared = value.float().square().sum(dim=dim, keepdim=keepdim)
    return _stable_sqrt(squared, epsilon=epsilon)


def warp_patch_grid(value: Tensor, flow: Tensor) -> tuple[Tensor, Tensor]:
    """Sample ``value`` at ``base + flow``; flow is measured in patch units."""

    if value.ndim != 4 or flow.ndim != 4 or int(flow.shape[1]) != 2:
        raise ValueError("warp_patch_grid expects value [B,C,H,W] and flow [B,2,H,W]")
    if tuple(value.shape[0:1] + value.shape[2:]) != tuple(flow.shape[0:1] + flow.shape[2:]):
        raise ValueError("warp value and flow geometry must match")
    batch, _, height, width = flow.shape
    base = _grid_coordinates(batch, height, width, device=flow.device, dtype=flow.dtype)
    coordinates = (base + flow).permute(0, 2, 3, 1)
    grid = _normalize_grid(coordinates, height, width)
    with torch.autocast(device_type=value.device.type, enabled=False):
        sampled = F.grid_sample(
            value.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(dtype=value.dtype)
    valid = (
        (coordinates[..., 0] >= 0.0)
        & (coordinates[..., 0] <= float(width - 1))
        & (coordinates[..., 1] >= 0.0)
        & (coordinates[..., 1] <= float(height - 1))
    )
    return sampled, valid[:, None]


def _fixed_raw_motion_descriptor(image: Tensor, side: int) -> Tensor:
    """Build a non-trainable local RGB/census chart for flow supervision.

    The learned raw pyramid is a policy value representation and is therefore
    free to become temporally invariant. Using it as its own warp target lets
    the descriptor and flow jointly reduce the loss without discovering a
    correspondence. This chart depends only on observed RGB: local colour
    contrast, luminance gradients, and eight soft census comparisons. It is
    detached by construction while remaining differentiable with respect to
    the coordinates used to sample it.
    """

    if image.ndim != 4 or int(image.shape[1]) != 3:
        raise ValueError("fixed raw motion descriptor expects [B,3,R,R]")
    if int(side) < 2:
        raise ValueError("fixed raw motion descriptor requires side >= 2")
    value = F.interpolate(
        image.detach().float(), size=(int(side), int(side)), mode="area"
    ).clamp(0.0, 1.0)
    local_mean = F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
    contrast = value - local_mean
    local_scale = F.avg_pool2d(
        contrast.abs().mean(dim=1, keepdim=True), kernel_size=3, stride=1, padding=1
    )
    contrast = torch.tanh(contrast / (local_scale + 0.02))
    gray = (
        0.2989 * value[:, 0:1]
        + 0.5870 * value[:, 1:2]
        + 0.1140 * value[:, 2:3]
    )
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    census_rows: list[Tensor] = []
    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        neighbour = padded[
            :, :, 1 + dy : 1 + dy + int(side), 1 + dx : 1 + dx + int(side)
        ]
        census_rows.append(torch.tanh((neighbour - gray) / 0.03))
    gradient_x = F.pad(gray[..., 2:] - gray[..., :-2], (1, 1, 0, 0))
    gradient_y = F.pad(gray[..., 2:, :] - gray[..., :-2, :], (0, 0, 1, 1))
    descriptor = torch.cat((contrast, gradient_x, gradient_y, *census_rows), dim=1)
    return F.normalize(descriptor, dim=1, eps=1e-6).detach()


def _fixed_observable_motion(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    """Return identity error and a detached continuous observable-motion map."""

    if tuple(first.shape) != tuple(second.shape):
        raise ValueError("fixed motion descriptors must align")
    identity_error = _stable_sqrt(
        (first.float() - second.float()).square().mean(dim=1, keepdim=True),
        epsilon=1e-8,
    )
    scale = identity_error.flatten(2).mean(dim=-1, keepdim=True)[..., None]
    relative_visibility = identity_error / (identity_error + scale + 1e-4)
    # The relative term alone turns arbitrarily small sensor/codec noise into
    # an apparent 50% motion signal when the whole frame is nearly static.
    # Retain a continuous absolute observability gate in descriptor units so
    # small residual noise cannot receive the sparse-motion budget.
    absolute_visibility = identity_error / (identity_error + 0.02)
    observable = relative_visibility * absolute_visibility
    return identity_error.detach(), observable.detach().clamp(0.0, 1.0)


def _continuous_cycle_visibility(
    valid: Tensor,
    cycle_error_squared: Tensor,
    threshold: Tensor,
    *,
    transition_fraction: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return continuous online visibility and a detached hard audit.

    The historical boolean threshold made occlusion jump from zero to one
    while providing no gradient explaining that jump.  Here the same local
    consistency threshold is retained, but online consumers receive a sigmoid
    transition whose width is a fixed fraction of the detached local
    threshold.  Detaching only the width prevents the optimizer from widening
    its own transition; gradients still reach both the cycle error and the
    physically meaningful threshold.
    """

    fraction = float(transition_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("cycle-visibility transition fraction must be in (0,1]")
    threshold_f = threshold.float().clamp_min(1e-6)
    width = (fraction * threshold_f.detach()).clamp_min(1e-4)
    soft_visibility = valid.float() * torch.sigmoid(
        (threshold_f - cycle_error_squared.float()) / width
    )
    hard_visibility = (
        valid & (cycle_error_squared.float() < threshold_f)
    ).float().detach()
    # d sigmoid(x) / dx <= 1/4. This is the exact local upper bound with
    # respect to the squared consistency error, expressed in its own units.
    gain_bound = (0.25 / width).amax().detach()
    return (
        soft_visibility,
        hard_visibility,
        width.amin().detach(),
        gain_bound,
    )


class _ConvNeXtPatchBlock(nn.Module):
    def __init__(self, channels: int, *, expansion: int = 4) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, expansion * channels)
        self.contract = nn.Linear(expansion * channels, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-3))

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.depthwise(value).permute(0, 2, 3, 1)
        value = self.contract(F.gelu(self.expand(self.norm(value))))
        value = value * self.layer_scale.to(device=value.device, dtype=value.dtype)
        return residual + value.permute(0, 3, 1, 2)


class _DINOFlowFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, feature_dim)
        self.spatial = nn.Sequential(
            _ConvNeXtPatchBlock(feature_dim),
            _ConvNeXtPatchBlock(feature_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("DINO flow features must be [B,H,W,D]")
        value = self.project(self.norm(value)).permute(0, 3, 1, 2)
        return self.spatial(value)


class _CorrelationPyramid:
    """RAFT-style all-pairs correlation with differentiable local lookup."""

    def __init__(
        self,
        first: Tensor,
        second: Tensor,
        *,
        levels: int,
        radius: int,
        normalization_floor: float | None = None,
    ) -> None:
        if tuple(first.shape) != tuple(second.shape) or first.ndim != 4:
            raise ValueError("correlation inputs must be shape-aligned [B,C,H,W]")
        self.levels = int(levels)
        self.radius = int(radius)
        batch, channels, height, width = first.shape
        self.batch = batch
        self.height = height
        self.width = width
        with torch.autocast(device_type=first.device.type, enabled=False):
            if normalization_floor is None:
                first_n = F.normalize(first.float(), dim=1)
                second_n = F.normalize(second.float(), dim=1)
                one = first.new_ones((), dtype=torch.float32)
                self.feature_rms_min = one
                self.normalization_denominator_min = one
                self.normalization_gain_max = one
            else:
                first_n, first_denominator = rms_floored_l2_normalize(
                    first, normalization_floor, dim=1
                )
                second_n, second_denominator = rms_floored_l2_normalize(
                    second, normalization_floor, dim=1
                )
                first_rms = first.float().square().mean(dim=1, keepdim=True).sqrt()
                second_rms = second.float().square().mean(dim=1, keepdim=True).sqrt()
                self.feature_rms_min = torch.minimum(
                    first_rms.amin(), second_rms.amin()
                ).detach()
                self.normalization_denominator_min = torch.minimum(
                    first_denominator.amin(), second_denominator.amin()
                ).detach()
                self.normalization_gain_max = (
                    1.0 / self.normalization_denominator_min.clamp_min(1e-8)
                ).detach()
            matrix = torch.einsum("bchw,bcij->bhwij", first_n, second_n)
        self.matrix = matrix.reshape(batch, height * width, height * width)
        corr = matrix.reshape(batch * height * width, 1, height, width)
        self.pyramid: list[Tensor] = []
        for level in range(self.levels):
            self.pyramid.append(corr)
            if level + 1 < self.levels:
                if min(int(corr.shape[-2]), int(corr.shape[-1])) < 2:
                    raise ValueError("flow grid is too small for the requested correlation levels")
                corr = F.avg_pool2d(corr, kernel_size=2, stride=2)

    @property
    def channels(self) -> int:
        return self.levels * (2 * self.radius + 1) ** 2

    def lookup(self, coordinates: Tensor) -> Tensor:
        if tuple(coordinates.shape) != (self.batch, 2, self.height, self.width):
            raise ValueError("correlation lookup coordinates have invalid geometry")
        radius = self.radius
        delta_y, delta_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=coordinates.device, dtype=torch.float32),
            torch.arange(-radius, radius + 1, device=coordinates.device, dtype=torch.float32),
            indexing="ij",
        )
        delta = torch.stack((delta_x, delta_y), dim=-1)
        center = coordinates.float().permute(0, 2, 3, 1).reshape(
            self.batch * self.height * self.width, 1, 1, 2
        )
        rows: list[Tensor] = []
        for level, corr in enumerate(self.pyramid):
            level_center = center / float(2**level)
            sample_coordinates = level_center + delta[None]
            grid = _normalize_grid(sample_coordinates, int(corr.shape[-2]), int(corr.shape[-1]))
            with torch.autocast(device_type=corr.device.type, enabled=False):
                sampled = F.grid_sample(
                    corr.float(),
                    grid.float(),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            rows.append(
                sampled.reshape(
                    self.batch,
                    self.height,
                    self.width,
                    (2 * radius + 1) ** 2,
                ).permute(0, 3, 1, 2)
            )
        return torch.cat(rows, dim=1)


class _PatchMotionUpdate(nn.Module):
    """SEA-RAFT-style shared ConvNeXt update of correlation, flow and scale."""

    def __init__(self, hidden: int, corr_channels: int) -> None:
        super().__init__()
        self.corr = nn.Sequential(
            nn.Conv2d(corr_channels, 2 * hidden, 1),
            nn.GELU(),
            nn.Conv2d(2 * hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.flow = nn.Sequential(
            nn.Conv2d(2, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
        )
        self.info = nn.Sequential(nn.Conv2d(4, hidden // 2, 3, padding=1), nn.GELU())
        self.input = nn.Conv2d(4 * hidden, hidden, 1)
        self.blocks = nn.Sequential(
            _ConvNeXtPatchBlock(hidden),
            _ConvNeXtPatchBlock(hidden),
        )

    def forward(
        self, state: Tensor, context: Tensor, correlation: Tensor, flow: Tensor, info: Tensor
    ) -> Tensor:
        motion = torch.cat((self.corr(correlation), self.flow(flow), self.info(info)), dim=1)
        return self.blocks(self.input(torch.cat((state, context, motion), dim=1)))


class LatentSeaRaft(nn.Module):
    """SEA-RAFT mechanism adapted to semantic DINO patch correspondence."""

    def __init__(
        self,
        config: Any,
        *,
        input_dim: int | None = None,
        feature_dim: int | None = None,
        identity_centered_initialization: bool = False,
    ) -> None:
        super().__init__()
        dim = int(config.flow_jepa_feature_dim if feature_dim is None else feature_dim)
        self.dim = dim
        self.iterations = int(config.flow_jepa_flow_iters)
        self.levels = int(config.flow_jepa_corr_levels)
        self.radius = int(config.flow_jepa_corr_radius)
        self.uncertainty_floor = float(config.flow_jepa_uncertainty_floor)
        self.identity_centered_initialization = bool(identity_centered_initialization)
        self.bounded_coordinates = bool(
            int(getattr(config, "flow_jepa_bounded_flow_coordinates", 0))
        )
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        self.correlation_rms_floor = float(
            getattr(config, "flow_jepa_correlation_rms_floor", 0.10)
        )
        self.encoder = _DINOFlowFeatureEncoder(
            int(config.visual_token_dim if input_dim is None else input_dim), dim
        )
        self.context = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 3, padding=1),
            nn.GELU(),
            _ConvNeXtPatchBlock(2 * dim),
        )
        corr_channels = self.levels * (2 * self.radius + 1) ** 2
        self.update = _PatchMotionUpdate(dim, corr_channels)
        self.delta_head = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * dim, 6, 3, padding=1),
        )
        self.initial_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, 3, padding=1),
        )
        self.correlation_temperature_log = nn.Parameter(torch.tensor(math.log(0.07)))

    def _uncertainty(self, information: Tensor) -> Tensor:
        mixture = torch.softmax(information[:, :2].float(), dim=1)
        scale = F.softplus(information[:, 2:].float()) + self.uncertainty_floor
        return (mixture * scale).sum(dim=1, keepdim=True)

    def forward(self, first: Tensor, second: Tensor) -> PatchFlowEstimate:
        """Estimate source-to-target displacement from [B,G,G,D] features."""

        first_feature = self.encoder(first.float())
        second_feature = self.encoder(second.float())
        correlation = _CorrelationPyramid(
            first_feature,
            second_feature,
            levels=self.levels,
            radius=self.radius,
            normalization_floor=(
                self.correlation_rms_floor
                if self.complete_numerical_contract
                else None
            ),
        )
        batch, _, height, width = first_feature.shape
        base = _grid_coordinates(batch, height, width, device=first.device, dtype=torch.float32)

        temperature = self.correlation_temperature_log.exp().clamp(0.02, 0.50)
        with torch.autocast(device_type=first.device.type, enabled=False):
            content_logits = correlation.matrix.float() / temperature.float()
            diagnostic_probability = torch.softmax(content_logits, dim=-1)
            if self.identity_centered_initialization:
                # Content-free global correlation must mean "no displacement",
                # not "move every source patch towards the image centre".  A
                # local geometric prior is subtracted from the content-aware
                # expectation, so uniform content has exactly zero flow while
                # informative correspondence remains fully differentiable.
                source_axis = _grid_coordinates(
                    1, height, width, device=first.device, dtype=torch.float32
                ).permute(0, 2, 3, 1).reshape(1, height * width, 1, 2)
                target_axis = _grid_coordinates(
                    1, height, width, device=first.device, dtype=torch.float32
                ).permute(0, 2, 3, 1).reshape(1, 1, height * width, 2)
                distance_square = (target_axis - source_axis).square().sum(dim=-1)
                prior_sigma = float(max(self.radius * self.levels, 1))
                prior_logits = -0.5 * distance_square / (prior_sigma * prior_sigma)
                prior_probability = torch.softmax(prior_logits, dim=-1)
                probability = torch.softmax(content_logits + prior_logits, dim=-1)
            else:
                prior_probability = None
                probability = diagnostic_probability
        target_coordinates = base.permute(0, 2, 3, 1).reshape(batch, height * width, 2)
        expected = torch.matmul(probability, target_coordinates)
        source_coordinates = target_coordinates
        if prior_probability is None:
            soft_flow = expected - source_coordinates
        else:
            prior_expected = torch.matmul(
                prior_probability.expand(batch, -1, -1), target_coordinates
            )
            soft_flow = expected - prior_expected
        soft_flow = soft_flow.reshape(batch, height, width, 2).permute(0, 3, 1, 2)
        entropy = -(
            diagnostic_probability.clamp_min(1e-8)
            * diagnostic_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(max(height * width, 2)))
        top2 = diagnostic_probability.topk(k=min(2, height * width), dim=-1).values
        margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else 0.0)
        entropy = entropy.reshape(batch, 1, height, width)
        margin = margin.reshape(batch, 1, height, width)

        context_pair = self.context(torch.cat((first_feature, second_feature), dim=1))
        state, context = context_pair.chunk(2, dim=1)
        flow_proposal = soft_flow + 0.25 * torch.tanh(self.initial_residual(state))
        if self.bounded_coordinates:
            flow, boundary_compression = _smooth_bound_flow_to_image(flow_proposal)
        else:
            flow = flow_proposal
            boundary_compression = flow.new_zeros(
                (batch, 1, height, width), dtype=torch.float32
            )
        information = self.delta_head(state)[:, 2:]
        predictions: list[Tensor] = [flow]
        for _ in range(self.iterations):
            coordinates = base + flow
            local_correlation = correlation.lookup(coordinates)
            state = self.update(state, context, local_correlation, flow, information)
            delta = self.delta_head(state)
            flow_proposal = flow + delta[:, :2]
            if self.bounded_coordinates:
                flow, row_compression = _smooth_bound_flow_to_image(flow_proposal)
                boundary_compression = torch.maximum(
                    boundary_compression.float(), row_compression.float()
                )
            else:
                flow = flow_proposal
            information = delta[:, 2:]
            predictions.append(flow)
        return PatchFlowEstimate(
            flow=flow,
            information=information,
            uncertainty=self._uncertainty(information),
            correlation_entropy=entropy,
            correlation_margin=margin,
            iterations=tuple(predictions),
            boundary_compression=boundary_compression,
            correlation_feature_rms_min=correlation.feature_rms_min,
            correlation_norm_denominator_min=(
                correlation.normalization_denominator_min
            ),
            correlation_norm_gain_max=correlation.normalization_gain_max,
        )


class _DenseDINOOrganizer(nn.Module):
    """Mix space, time and cameras on the full-image coarse chart.

    Spatial mixing is local and translation-aware; temporal and camera mixing
    are factorized at each chart coordinate.  Native patches are retained in a
    separate late-read lane, so this global organizer does not pretend that a
    pooled cell contains contact-scale detail.
    """

    def __init__(self, input_dim: int, hidden: int, *, depth: int, heads: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, hidden) * 0.02)
        self.spatial = nn.ModuleList(
            [_ConvNeXtPatchBlock(hidden) for _ in range(int(depth))]
        )
        self.temporal_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.temporal = nn.MultiheadAttention(
            hidden, heads, batch_first=True, dropout=0.0
        )
        self.camera_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.camera = nn.MultiheadAttention(
            hidden, heads, batch_first=True, dropout=0.0
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.temporal_scale = nn.Parameter(torch.tensor(0.10))
        self.camera_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, visual: Tensor, mask: Tensor | None = None) -> Tensor:
        if visual.ndim != 6:
            raise ValueError("dense DINO organizer expects [B,T,C,S,S,D]")
        batch, history, cameras, side, side_b, _ = visual.shape
        if side != side_b:
            raise ValueError("dense DINO organizer requires a square patch chart")
        value = self.input_proj(self.input_norm(visual))
        if mask is not None:
            if tuple(mask.shape) != tuple(value.shape[:-1]):
                raise ValueError("dense DINO context mask must align with [B,T,C,S,S]")
            value = torch.where(
                mask[..., None],
                self.mask_token.to(device=value.device, dtype=value.dtype),
                value,
            )
        spatial = value.reshape(batch * history * cameras, side, side, self.hidden)
        spatial = spatial.permute(0, 3, 1, 2)
        for block in self.spatial:
            spatial = block(spatial)
        value = spatial.permute(0, 2, 3, 1).reshape(
            batch, history, cameras, side, side, self.hidden
        )

        temporal = value.permute(0, 2, 3, 4, 1, 5).reshape(
            batch * cameras * side * side, history, self.hidden
        )
        temporal_n = self.temporal_norm(temporal)
        temporal_update, _ = self.temporal(
            temporal_n, temporal_n, temporal_n, need_weights=False
        )
        temporal = temporal + self.temporal_scale.tanh().to(
            device=value.device, dtype=value.dtype
        ) * temporal_update
        value = temporal.reshape(batch, cameras, side, side, history, self.hidden).permute(
            0, 4, 1, 2, 3, 5
        )

        camera = value.permute(0, 1, 3, 4, 2, 5).reshape(
            batch * history * side * side, cameras, self.hidden
        )
        camera_n = self.camera_norm(camera)
        camera_update, _ = self.camera(camera_n, camera_n, camera_n, need_weights=False)
        camera = camera + self.camera_scale.tanh().to(
            device=value.device, dtype=value.dtype
        ) * camera_update
        value = camera.reshape(batch, history, side, side, cameras, self.hidden).permute(
            0, 1, 4, 2, 3, 5
        )
        return self.output_norm(value)


class _EarlyMaskedRawContextEncoder(nn.Module):
    """Build a local RGB chart after hiding cells, before learned mixing.

    The mask is expanded in the original image chart first.  Only then is the
    image reduced to a 2G chart and processed by trainable spatial blocks.  Two
    raw images that differ exclusively inside masked cells therefore produce
    exactly the same output.  A binary mask channel distinguishes hidden
    pixels from genuinely mean-coloured observations.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: int,
        grid: int,
        *,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        feature_dim = int(feature_dim)
        self.grid = int(grid)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.mask_rgb = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.local = nn.Sequential(
            nn.Conv2d(4, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(feature_dim), feature_dim),
            nn.GELU(),
            _ConvNeXtPatchBlock(feature_dim),
            _ConvNeXtPatchBlock(feature_dim),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(feature_dim), feature_dim),
            nn.GELU(),
            _ConvNeXtPatchBlock(feature_dim),
            _ConvNeXtPatchBlock(feature_dim),
            nn.Conv2d(feature_dim, hidden, 1, bias=False),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.register_buffer(
            "rgb_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "rgb_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None],
            persistent=False,
        )

    def _run(self, value: Tensor) -> Tensor:
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self.local, value, use_reentrant=False)
        return self.local(value)

    def forward(self, raw_visual: Tensor, mask: Tensor) -> Tensor:
        if raw_visual.ndim != 6 or int(raw_visual.shape[3]) != 3:
            raise ValueError("early masked raw context expects [B,T,C,3,R,R]")
        if mask.ndim != 5 or tuple(mask.shape[:3]) != tuple(raw_visual.shape[:3]):
            raise ValueError("early raw mask must align as [B,T,C,G,G]")
        if tuple(mask.shape[-2:]) != (self.grid, self.grid):
            raise ValueError("early raw mask does not match the configured grid")
        batch, history, cameras, channels, side, side_b = raw_visual.shape
        if side != side_b:
            raise ValueError("early masked raw context requires square images")
        flat = raw_visual.reshape(batch * history * cameras, channels, side, side)
        with torch.autocast(device_type=flat.device.type, enabled=False):
            normalized = (flat.float() - self.rgb_mean.float()) / self.rgb_std.float()
            pixel_mask = F.interpolate(
                mask.reshape(batch * history * cameras, 1, self.grid, self.grid).float(),
                size=(side, side),
                mode="nearest",
            )
            masked = torch.where(
                pixel_mask.bool(),
                self.mask_rgb.float().expand_as(normalized),
                normalized,
            )
            # 2G keeps within-cell structure before the learned stride-2
            # transition creates the chart consumed by future queries.
            local_input = F.adaptive_avg_pool2d(masked, (2 * self.grid, 2 * self.grid))
            mask_channel = F.adaptive_max_pool2d(
                pixel_mask, (2 * self.grid, 2 * self.grid)
            )
            local_input = torch.cat((local_input, mask_channel), dim=1)
        encoded = self._run(local_input.to(dtype=raw_visual.dtype))
        if tuple(encoded.shape[-2:]) != (self.grid, self.grid):
            raise RuntimeError("early masked raw encoder produced the wrong chart size")
        encoded = encoded.permute(0, 2, 3, 1).reshape(
            batch, history, cameras, self.grid, self.grid, int(encoded.shape[1])
        )
        return self.output_norm(encoded)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class _RawImagePyramid(nn.Module):
    """Shared RGB pyramid that preserves 1/4 detail until grounded reading.

    For the production 336 input the returned maps are 84 and 42 pixels.  The
    8x8 global address comes from DINO, so this module deliberately has no raw
    global-correlation stage.  No semantic or motion gate is allowed to
    discard the 84x84 map here.
    """

    def __init__(self, base: int, *, activation_checkpoint: bool) -> None:
        super().__init__()
        base = int(base)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.high_channels = base + base // 2
        self.mid_channels = 2 * base

        def down(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_group_count(cout), cout),
                nn.GELU(),
                _ConvNeXtPatchBlock(cout),
                _ConvNeXtPatchBlock(cout),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(base), base),
            nn.GELU(),
            _ConvNeXtPatchBlock(base),
        )
        self.high = down(base, self.high_channels)
        self.mid = down(self.high_channels, self.mid_channels)
        self.register_buffer(
            "rgb_mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "rgb_std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None],
            persistent=False,
        )

    def _run(self, module: nn.Module, value: Tensor) -> Tensor:
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, value, use_reentrant=False)
        return module(value)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError("raw image pyramid expects [B,3,R,R]")
        if int(image.shape[-2]) != int(image.shape[-1]) or int(image.shape[-1]) % 16:
            raise ValueError("raw image pyramid requires a square side divisible by 16")
        value = (image.float() - self.rgb_mean) / self.rgb_std
        value = value.to(dtype=image.dtype)
        high = self._run(self.high, self._run(self.stem, value))
        mid = self._run(self.mid, high)
        return high, mid


class _DenseRawFlowRefiner(nn.Module):
    """Dense local RAFT update at one raw-pyramid resolution."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        *,
        radius: int,
        uncertainty_floor: float,
        activation_checkpoint: bool,
        preserve_uncertain_seed: bool = False,
        bounded_coordinates: bool = False,
        normalization_floor: float | None = None,
    ) -> None:
        super().__init__()
        self.radius = int(radius)
        self.uncertainty_floor = float(uncertainty_floor)
        self.activation_checkpoint = bool(activation_checkpoint)
        self.preserve_uncertain_seed = bool(preserve_uncertain_seed)
        self.bounded_coordinates = bool(bounded_coordinates)
        self.normalization_floor = (
            None if normalization_floor is None else float(normalization_floor)
        )
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim, hidden, 1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.GELU(),
            _ConvNeXtPatchBlock(hidden),
        )
        self.update = nn.Sequential(
            nn.Conv2d(3 * hidden + 6, 2 * hidden, 3, padding=1),
            nn.GELU(),
            _ConvNeXtPatchBlock(2 * hidden),
            nn.Conv2d(2 * hidden, 3, 3, padding=1),
        )
        nn.init.normal_(self.update[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)
        self.temperature_log = nn.Parameter(torch.tensor(math.log(0.10)))

    @staticmethod
    def _sample(value: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        side = int(value.shape[-1])
        grid = _normalize_grid(coordinates, side, side)
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.float(), grid.float(), mode="bilinear", padding_mode="zeros",
                align_corners=True,
            )
        valid = (
            (coordinates[..., 0] >= 0.0)
            & (coordinates[..., 0] <= float(side - 1))
            & (coordinates[..., 1] >= 0.0)
            & (coordinates[..., 1] <= float(side - 1))
        )
        return sampled, valid

    def _forward_tensors(
        self,
        first: Tensor,
        second: Tensor,
        coarse_flow: Tensor,
        coarse_reliability: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        if first.ndim != 4 or tuple(first.shape) != tuple(second.shape):
            raise ValueError("raw flow refinement features must align as [B,C,S,S]")
        batch, _, side, side_b = first.shape
        if side != side_b or coarse_flow.ndim != 4 or int(coarse_flow.shape[1]) != 2:
            raise ValueError("raw flow refinement received invalid geometry")
        previous_side = int(coarse_flow.shape[-1])
        if int(coarse_flow.shape[-2]) != previous_side or int(coarse_flow.shape[0]) != batch:
            raise ValueError("raw coarse flow must be square and batch aligned")
        scale = float(max(side - 1, 1)) / float(max(previous_side - 1, 1))
        up_flow = F.interpolate(
            coarse_flow.float(), size=(side, side), mode="bilinear", align_corners=True
        ) * scale
        reliability = F.interpolate(
            coarse_reliability.float(), size=(side, side), mode="bilinear", align_corners=True
        ).clamp(0.0, 1.0)
        if not self.preserve_uncertain_seed:
            up_flow = up_flow * reliability
        first_feature = self.feature(first)
        second_feature = self.feature(second)
        if self.normalization_floor is None:
            source = F.normalize(first_feature.float(), dim=1)
            one = first_feature.new_ones((), dtype=torch.float32)
            correlation_feature_rms_min = one
            correlation_norm_denominator_min = one
            correlation_norm_gain_max = one
        else:
            source, source_denominator = rms_floored_l2_normalize(
                first_feature, self.normalization_floor, dim=1
            )
            correlation_feature_rms_min = (
                first_feature.float().square().mean(dim=1).sqrt().amin().detach()
            )
            correlation_norm_denominator_min = source_denominator.amin().detach()
            correlation_norm_gain_max = (
                1.0 / correlation_norm_denominator_min.clamp_min(1e-8)
            ).detach()
        base = _grid_coordinates(batch, side, side, device=first.device, dtype=torch.float32)
        center = (base + up_flow).permute(0, 2, 3, 1)
        offsets = [
            (dx, dy)
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
        ]
        correlations: list[Tensor] = []
        valid_rows: list[Tensor] = []
        sampled_rows: list[Tensor] = []
        offset_rows: list[Tensor] = []
        search_scale = 1.0 + 0.5 * (1.0 - reliability)
        for dx, dy in offsets:
            if self.preserve_uncertain_seed:
                offset = torch.cat(
                    (float(dx) * search_scale, float(dy) * search_scale), dim=1
                )
            else:
                offset = up_flow.new_tensor((float(dx), float(dy)))[None, :, None, None]
                offset = offset.expand(batch, -1, side, side)
            coordinates = center + offset.permute(0, 2, 3, 1)
            sampled, valid = self._sample(second_feature, coordinates)
            if self.normalization_floor is None:
                sampled_normalized = F.normalize(sampled.float(), dim=1)
            else:
                sampled_normalized, sampled_denominator = rms_floored_l2_normalize(
                    sampled, self.normalization_floor, dim=1
                )
                sampled_rms = sampled.float().square().mean(dim=1).sqrt()
                # Grid-sample padding is exact zero by construction and is
                # already excluded from the correlation softmax. Do not label
                # that factual invalid support as learned feature collapse.
                sampled_rms_min = sampled_rms.masked_fill(
                    ~valid, torch.inf
                ).amin().detach()
                correlation_feature_rms_min = torch.minimum(
                    correlation_feature_rms_min, sampled_rms_min
                )
                correlation_norm_denominator_min = torch.minimum(
                    correlation_norm_denominator_min,
                    sampled_denominator.amin().detach(),
                )
                correlation_norm_gain_max = torch.maximum(
                    correlation_norm_gain_max,
                    (
                        1.0
                        / sampled_denominator.amin().detach().clamp_min(1e-8)
                    ),
                )
            correlations.append(
                (source * sampled_normalized).sum(dim=1)
            )
            valid_rows.append(valid)
            sampled_rows.append(sampled)
            offset_rows.append(offset)
        logits = torch.stack(correlations, dim=1)
        valid = torch.stack(valid_rows, dim=1)
        logits = logits / self.temperature_log.exp().clamp(0.03, 0.50)
        valid_any = valid.any(dim=1, keepdim=True)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        logits = torch.where(valid_any, logits, torch.zeros_like(logits))
        probability = torch.softmax(logits, dim=1)
        # Subtract the content-free distribution over the same valid support.
        # This removes the inward boundary drift without hard-selecting any
        # candidate: equal correlations produce exactly zero local residual.
        support_prior = valid.float() / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        offset_field = torch.stack(offset_rows, dim=1).float()
        expected = (
            (probability - support_prior)[:, :, None] * offset_field
        ).sum(dim=1)
        aligned = torch.zeros_like(first_feature, dtype=torch.float32)
        for index, sampled in enumerate(sampled_rows):
            aligned = aligned + probability[:, index : index + 1] * sampled.float()
        entropy = -(
            probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True) / math.log(float(max(len(offsets), 2)))
        top2 = probability.topk(k=min(2, len(offsets)), dim=1).values
        margin = top2[:, :1] - (top2[:, 1:2] if int(top2.shape[1]) > 1 else 0.0)
        normalizer = float(max(side - 1, 1))
        update_input = torch.cat(
            (
                first_feature.float(),
                aligned,
                first_feature.float() - aligned,
                up_flow / normalizer,
                expected / normalizer,
                entropy,
                margin,
            ),
            dim=1,
        )
        learned = self.update(update_input.to(dtype=first_feature.dtype)).float()
        flow_proposal = up_flow + expected + 0.25 * torch.tanh(learned[:, :2])
        if self.bounded_coordinates:
            flow, boundary_compression = _smooth_bound_flow_to_image(flow_proposal)
        else:
            flow = flow_proposal
            boundary_compression = flow.new_zeros(
                (batch, 1, side, side), dtype=torch.float32
            )
        uncertainty = F.softplus(learned[:, 2:3]) + self.uncertainty_floor
        information = torch.cat(
            (entropy, margin, uncertainty, valid_any.float()), dim=1
        )
        return (
            flow,
            information,
            uncertainty,
            entropy,
            margin,
            up_flow,
            boundary_compression,
            correlation_feature_rms_min,
            correlation_norm_denominator_min,
            correlation_norm_gain_max,
        )

    def forward(
        self,
        first: Tensor,
        second: Tensor,
        coarse_flow: Tensor,
        coarse_reliability: Tensor | None = None,
    ) -> PatchFlowEstimate:
        if coarse_reliability is None:
            coarse_reliability = coarse_flow.new_ones(
                (
                    int(coarse_flow.shape[0]),
                    1,
                    int(coarse_flow.shape[-2]),
                    int(coarse_flow.shape[-1]),
                )
            )
        if tuple(coarse_reliability.shape) != (
            int(coarse_flow.shape[0]), 1, int(coarse_flow.shape[-2]), int(coarse_flow.shape[-1])
        ):
            raise ValueError("raw coarse reliability must align with coarse flow")
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            rows = checkpoint(
                self._forward_tensors,
                first,
                second,
                coarse_flow,
                coarse_reliability,
                use_reentrant=False,
            )
        else:
            rows = self._forward_tensors(first, second, coarse_flow, coarse_reliability)
        (
            flow,
            information,
            uncertainty,
            entropy,
            margin,
            up_flow,
            boundary_compression,
            correlation_feature_rms_min,
            correlation_norm_denominator_min,
            correlation_norm_gain_max,
        ) = rows
        return PatchFlowEstimate(
            flow=flow,
            information=information,
            uncertainty=uncertainty,
            correlation_entropy=entropy,
            correlation_margin=margin,
            iterations=(up_flow, flow),
            boundary_compression=boundary_compression,
            correlation_feature_rms_min=correlation_feature_rms_min,
            correlation_norm_denominator_min=correlation_norm_denominator_min,
            correlation_norm_gain_max=correlation_norm_gain_max,
        )


class _RawPyramidFlow(nn.Module):
    """DINO-seeded dense 1/8 and 1/4 raw refinement."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.zero_flow_guard = bool(
            int(getattr(config, "flow_jepa_zero_flow_guard", 0))
        )
        self.bounded_coordinates = bool(
            int(getattr(config, "flow_jepa_bounded_flow_coordinates", 0))
        )
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        self.visibility_transition_fraction = float(
            getattr(config, "flow_jepa_visibility_transition_fraction", 0.10)
        )
        correlation_rms_floor = (
            float(getattr(config, "flow_jepa_correlation_rms_floor", 0.10))
            if self.complete_numerical_contract
            else None
        )
        activation_checkpoint = bool(
            int(getattr(config, "flow_jepa_raw_activation_checkpoint", 1))
        )
        self.pyramid = _RawImagePyramid(
            int(config.flow_jepa_raw_base_channels),
            activation_checkpoint=activation_checkpoint,
        )
        dim = int(config.flow_jepa_feature_dim)
        self.mid = _DenseRawFlowRefiner(
            self.pyramid.mid_channels,
            dim,
            radius=int(config.flow_jepa_raw_mid_radius),
            uncertainty_floor=float(config.flow_jepa_uncertainty_floor),
            activation_checkpoint=activation_checkpoint,
            preserve_uncertain_seed=self.zero_flow_guard,
            bounded_coordinates=self.bounded_coordinates,
            normalization_floor=correlation_rms_floor,
        )
        self.high = _DenseRawFlowRefiner(
            self.pyramid.high_channels,
            max(dim // 2, 32),
            radius=int(config.flow_jepa_raw_high_radius),
            uncertainty_floor=float(config.flow_jepa_uncertainty_floor),
            activation_checkpoint=activation_checkpoint,
            preserve_uncertain_seed=self.zero_flow_guard,
            bounded_coordinates=self.bounded_coordinates,
            normalization_floor=correlation_rms_floor,
        )

    @staticmethod
    def _smoothness(flow: Tensor, feature: Tensor) -> Tensor:
        dx = flow[..., :, 1:] - flow[..., :, :-1]
        dy = flow[..., 1:, :] - flow[..., :-1, :]
        fx = feature[..., :, 1:] - feature[..., :, :-1]
        fy = feature[..., 1:, :] - feature[..., :-1, :]
        wx = torch.exp(-fx.float().abs().mean(dim=1, keepdim=True))
        wy = torch.exp(-fy.float().abs().mean(dim=1, keepdim=True))
        return (dx.float().abs() * wx).mean() + (dy.float().abs() * wy).mean()

    @staticmethod
    def _paired_mean(
        first: Tensor,
        first_valid: Tensor,
        second: Tensor,
        second_valid: Tensor,
        first_emphasis: Tensor | None = None,
        second_emphasis: Tensor | None = None,
    ) -> Tensor:
        fw = first_valid.float()
        bw = second_valid.float()
        if first_emphasis is not None:
            fw = fw * first_emphasis.float()
        if second_emphasis is not None:
            bw = bw * second_emphasis.float()
        return ((first * fw).sum() + (second * bw).sum()) / (
            fw.sum() + bw.sum()
        ).clamp_min(1.0)

    @staticmethod
    def _boundary_excess(flow: Tensor) -> Tensor:
        batch, _, side, _ = flow.shape
        base = _grid_coordinates(
            batch, side, side, device=flow.device, dtype=torch.float32
        )
        coordinates = base + flow.float()
        lower = F.relu(-coordinates)
        upper = F.relu(coordinates - float(side - 1))
        return (lower + upper).sum(dim=1, keepdim=True)

    @staticmethod
    def _motion_emphasis(
        coarse_flow: Tensor, coarse_reliability: Tensor, side: int
    ) -> Tensor:
        motion = _stable_vector_norm(coarse_flow, dim=1, keepdim=True)
        motion = motion * coarse_reliability.float().clamp(0.0, 1.0)
        motion = F.interpolate(
            motion, size=(side, side), mode="bilinear", align_corners=True
        )
        mean = motion.flatten(2).mean(dim=-1, keepdim=True)[..., None]
        # Every location remains supervised.  Motion only raises the relative
        # weight continuously from 1x towards 2x; it never selects or drops a
        # spatial region.
        return 1.0 + motion / (motion + mean + 1e-4)

    @staticmethod
    def _fixed_motion_emphasis(observable_motion: Tensor) -> Tensor:
        """Give sparse visible changes a fixed budget without selecting pixels."""

        return 1.0 + 3.0 * observable_motion.detach().float().clamp(0.0, 1.0)

    def forward(
        self,
        raw_visual: Tensor,
        coarse_forward: Tensor,
        coarse_backward: Tensor,
        coarse_reliability_forward: Tensor,
        coarse_reliability_backward: Tensor,
    ) -> tuple[RawGroundingContext, dict[str, Tensor], dict[str, Tensor]]:
        if raw_visual.ndim != 6 or int(raw_visual.shape[3]) != 3:
            raise ValueError("raw_visual must be [B,T,C,3,R,R]")
        batch, history, cameras, channels, side, side_b = raw_visual.shape
        if history < 2 or side != side_b:
            raise ValueError("raw flow requires at least two square RGB history frames")
        flat = raw_visual.reshape(batch * history * cameras, channels, side, side)
        high, mid = self.pyramid(flat)

        def restore(value: Tensor) -> Tensor:
            return value.reshape(batch, history, cameras, *value.shape[1:])

        high_u, mid_u = restore(high), restore(mid)
        if self.zero_flow_guard:
            fixed_high = _fixed_raw_motion_descriptor(flat, int(high.shape[-1]))
            fixed_mid = F.normalize(
                F.interpolate(
                    fixed_high,
                    size=(int(mid.shape[-1]), int(mid.shape[-1])),
                    mode="area",
                ),
                dim=1,
                eps=1e-6,
            ).detach()
            fixed_high_u = restore(fixed_high)
            fixed_mid_u = restore(fixed_mid)
        else:
            fixed_high_u = None
            fixed_mid_u = None
        pair_count = history - 1
        pair_batch = batch * pair_count * cameras
        coarse_shape = (
            pair_batch,
            2,
            int(coarse_forward.shape[-2]),
            int(coarse_forward.shape[-1]),
        )
        if tuple(coarse_forward.shape) != coarse_shape or tuple(coarse_backward.shape) != coarse_shape:
            raise ValueError("DINO coarse flow does not align with raw frame pairs")
        reliability_shape = (pair_batch, 1, coarse_shape[-2], coarse_shape[-1])
        if (
            tuple(coarse_reliability_forward.shape) != reliability_shape
            or tuple(coarse_reliability_backward.shape) != reliability_shape
        ):
            raise ValueError("DINO coarse reliability does not align with raw frame pairs")

        def pair(value: Tensor) -> tuple[Tensor, Tensor]:
            return (
                value[:, :-1].reshape(batch * pair_count * cameras, *value.shape[3:]),
                value[:, 1:].reshape(batch * pair_count * cameras, *value.shape[3:]),
            )

        first_h, second_h = pair(high_u)
        first_m, second_m = pair(mid_u)
        if fixed_high_u is not None and fixed_mid_u is not None:
            first_fixed_h, second_fixed_h = pair(fixed_high_u)
            first_fixed_m, second_fixed_m = pair(fixed_mid_u)
        else:
            first_fixed_h = second_fixed_h = None
            first_fixed_m = second_fixed_m = None
        joined_coarse_flow = torch.cat((coarse_forward, coarse_backward), dim=0)
        joined_coarse_reliability = torch.cat(
            (coarse_reliability_forward, coarse_reliability_backward), dim=0
        )
        joined_mid = self.mid(
            torch.cat((first_m, second_m), dim=0),
            torch.cat((second_m, first_m), dim=0),
            joined_coarse_flow,
            joined_coarse_reliability,
        )
        joined_high = self.high(
            torch.cat((first_h, second_h), dim=0),
            torch.cat((second_h, first_h), dim=0),
            joined_mid.flow,
        )

        def split(value: Tensor) -> tuple[Tensor, Tensor]:
            return value[:pair_batch], value[pair_batch:]

        forward, backward = split(joined_high.flow)
        uncertainty_f, uncertainty_b = split(joined_high.uncertainty)
        entropy_f, entropy_b = split(joined_high.correlation_entropy)
        margin_f, margin_b = split(joined_high.correlation_margin)
        warped_backward, forward_cycle_valid = warp_patch_grid(backward, forward)
        warped_forward, backward_cycle_valid = warp_patch_grid(forward, backward)
        forward_cycle = forward + warped_backward
        backward_cycle = backward + warped_forward
        forward_cycle_error = _stable_vector_norm(forward_cycle, dim=1, keepdim=True)
        backward_cycle_error = _stable_vector_norm(backward_cycle, dim=1, keepdim=True)
        unit_scale = float(max(coarse_shape[-1] - 1, 1)) / float(
            max(int(forward.shape[-1]) - 1, 1)
        )
        if first_fixed_h is not None and second_fixed_h is not None:
            warp_source, warp_target = first_fixed_h, second_fixed_h
            forward_identity_error, observable_motion = _fixed_observable_motion(
                warp_source, warp_target
            )
            backward_identity_error = forward_identity_error
        else:
            warp_source = F.normalize(first_h.float(), dim=1)
            warp_target = F.normalize(second_h.float(), dim=1)
            forward_identity_error = _stable_sqrt(
                (warp_source - warp_target).square().mean(dim=1, keepdim=True),
                epsilon=1e-8,
            ).detach()
            backward_identity_error = forward_identity_error
            observable_motion = None
        warped_second, forward_warp_valid = warp_patch_grid(warp_target, forward)
        warped_first, backward_warp_valid = warp_patch_grid(warp_source, backward)
        forward_warp_error = (
            warp_source.float() - warped_second.float()
        ).square().mean(dim=1, keepdim=True)
        backward_warp_error = (
            warp_target.float() - warped_first.float()
        ).square().mean(dim=1, keepdim=True)
        confidence = (
            torch.exp(-uncertainty_f.float())
            * torch.exp(-forward_cycle_error * unit_scale)
            * (1.0 - entropy_f.float()).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        forward_grid = forward.float() * unit_scale
        backward_warped_grid = warped_backward.float() * unit_scale
        cycle_error_squared = (forward_cycle_error * unit_scale).square()
        visibility_threshold = (
            0.01
            * (forward_grid.square().sum(dim=1, keepdim=True)
               + backward_warped_grid.square().sum(dim=1, keepdim=True))
            + 0.5
        )
        if self.complete_numerical_contract:
            (
                visible,
                hard_visibility,
                visibility_transition_width_min,
                visibility_gain_bound_max,
            ) = _continuous_cycle_visibility(
                forward_cycle_valid,
                cycle_error_squared,
                visibility_threshold,
                transition_fraction=self.visibility_transition_fraction,
            )
            occlusion = 1.0 - visible
        else:
            hard_visibility = (
                forward_cycle_valid
                & (cycle_error_squared < visibility_threshold)
            ).float()
            visible = hard_visibility
            occlusion = 1.0 - visible
            visibility_transition_width_min = forward.new_zeros(
                (), dtype=torch.float32
            )
            visibility_gain_bound_max = forward.new_zeros(
                (), dtype=torch.float32
            )
        if observable_motion is not None:
            forward_motion_emphasis = self._fixed_motion_emphasis(observable_motion)
            backward_motion_emphasis = forward_motion_emphasis
        else:
            forward_motion_emphasis = self._motion_emphasis(
                coarse_forward, coarse_reliability_forward, int(forward.shape[-1])
            )
            backward_motion_emphasis = self._motion_emphasis(
                coarse_backward, coarse_reliability_backward, int(backward.shape[-1])
            )
            observable_motion = (forward_motion_emphasis - 1.0).detach().clamp(0.0, 1.0)
        forward_warp_residual = _stable_sqrt(forward_warp_error, epsilon=1e-8)
        backward_warp_residual = _stable_sqrt(backward_warp_error, epsilon=1e-8)
        warp_loss = self._paired_mean(
            forward_warp_residual,
            forward_warp_valid,
            backward_warp_residual,
            backward_warp_valid,
            forward_motion_emphasis,
            backward_motion_emphasis,
        )
        if self.zero_flow_guard:
            # Require an observable-motion correspondence to improve over the
            # exact identity baseline. The margin scales with the evidence, so
            # identical/static pixels never receive an artificial nonzero-flow
            # target.
            forward_deficit = F.relu(
                forward_warp_residual - 0.95 * forward_identity_error
            )
            backward_deficit = F.relu(
                backward_warp_residual - 0.95 * backward_identity_error
            )
            identity_advantage_loss = self._paired_mean(
                forward_deficit,
                forward_warp_valid,
                backward_deficit,
                backward_warp_valid,
                observable_motion,
                observable_motion,
            )
            # Static evidence has a different contract: do not demand motion,
            # only prevent the learned warp from being worse than the exact
            # identity address.  This closes the zero-flow-collapse loophole
            # without introducing a nonzero-flow target on unchanged pixels.
            static_weight = (1.0 - observable_motion).clamp(0.0, 1.0)
            static_identity_loss = self._paired_mean(
                F.relu(forward_warp_residual - forward_identity_error),
                forward_warp_valid,
                F.relu(backward_warp_residual - backward_identity_error),
                backward_warp_valid,
                static_weight,
                static_weight,
            )
        else:
            identity_advantage_loss = warp_loss * 0.0
            static_identity_loss = warp_loss * 0.0
        cycle_core = unit_scale * self._paired_mean(
            torch.sqrt(forward_cycle_error.square() + 1e-6),
            forward_cycle_valid,
            torch.sqrt(backward_cycle_error.square() + 1e-6),
            backward_cycle_valid,
        )
        boundary_penalty = 0.5 * unit_scale * (
            self._boundary_excess(forward).mean()
            + self._boundary_excess(backward).mean()
        )
        # Boundary escape used to disappear behind the validity mask.  Keep the
        # historical loss key but make its raw-path geometry explicit: core
        # forward/backward consistency plus a small continuous boundary term.
        cycle_loss = cycle_core + 0.25 * boundary_penalty
        smooth_source = first_fixed_h if first_fixed_h is not None else first_h
        smooth_target = second_fixed_h if second_fixed_h is not None else second_h
        smoothness = 0.5 * unit_scale * (
            self._smoothness(forward, smooth_source)
            + self._smoothness(backward, smooth_target)
        )
        uncertainty_nll = self._paired_mean(
            forward_warp_error.detach().sqrt() / uncertainty_f.clamp_min(1e-4)
            + uncertainty_f.clamp_min(1e-4).log(),
            forward_warp_valid,
            backward_warp_error.detach().sqrt() / uncertainty_b.clamp_min(1e-4)
            + uncertainty_b.clamp_min(1e-4).log(),
            backward_warp_valid,
        )
        mid_forward, mid_backward = split(joined_mid.flow)
        mid_source = first_fixed_m if first_fixed_m is not None else F.normalize(
            first_m.float(), dim=1
        )
        mid_target = second_fixed_m if second_fixed_m is not None else F.normalize(
            second_m.float(), dim=1
        )
        mid_second, mid_valid_f = warp_patch_grid(mid_target, mid_forward)
        mid_first, mid_valid_b = warp_patch_grid(mid_source, mid_backward)
        mid_error_f = _stable_sqrt(
            (mid_source.float() - mid_second.float()).square().mean(dim=1, keepdim=True)
        )
        mid_error_b = _stable_sqrt(
            (mid_target.float() - mid_first.float()).square().mean(dim=1, keepdim=True)
        )
        sequence_loss = self._paired_mean(
            mid_error_f, mid_valid_f, mid_error_b, mid_valid_b
        )
        forward_gain = forward_identity_error - forward_warp_residual.detach()
        backward_gain = backward_identity_error - backward_warp_residual.detach()
        warp_gain_over_zero = self._paired_mean(
            forward_gain,
            forward_warp_valid,
            backward_gain,
            backward_warp_valid,
        )
        moving_warp_gain = self._paired_mean(
            forward_gain,
            forward_warp_valid,
            backward_gain,
            backward_warp_valid,
            observable_motion,
            observable_motion,
        )
        moving_correlation_entropy = self._paired_mean(
            entropy_f.float(),
            forward_warp_valid,
            entropy_b.float(),
            backward_warp_valid,
            observable_motion,
            observable_motion,
        )
        moving_correlation_margin = self._paired_mean(
            margin_f.float(),
            forward_warp_valid,
            margin_b.float(),
            backward_warp_valid,
            observable_motion,
            observable_motion,
        )
        static_weight = (1.0 - observable_motion).clamp(0.0, 1.0)
        static_warp_gain = self._paired_mean(
            forward_gain,
            forward_warp_valid,
            backward_gain,
            backward_warp_valid,
            static_weight,
            static_weight,
        )

        def unflatten(value: Tensor) -> Tensor:
            return value.reshape(batch, pair_count, cameras, *value.shape[1:])

        context = RawGroundingContext(
            # Only the last observed pair is needed by the post-grounding raw
            # reader.  Earlier high-resolution activations are not retained
            # across the eight-block DiT path.
            high_features=high_u[:, -2:],
            flow_forward=unflatten(forward),
            flow_backward=unflatten(backward),
            confidence=unflatten(confidence),
            occlusion=unflatten(occlusion),
            uncertainty=unflatten(uncertainty_f),
            correlation_entropy=unflatten(entropy_f),
            correlation_margin=unflatten(margin_f),
            cycle_error=unflatten(forward_cycle_error),
            warp_error=unflatten(_stable_sqrt(forward_warp_error)),
            observable_motion=unflatten(observable_motion),
        )
        losses = {
            "flow_jepa_warp_loss": warp_loss,
            "flow_jepa_identity_advantage_loss": identity_advantage_loss,
            "flow_jepa_static_identity_loss": static_identity_loss,
            "flow_jepa_cycle_loss": cycle_loss,
            "flow_jepa_smoothness_loss": smoothness,
            "flow_jepa_uncertainty_nll": uncertainty_nll,
            "flow_jepa_refinement_sequence_loss": sequence_loss,
        }
        metrics = {
            "flow_jepa_raw_high_grid_size": forward.new_tensor(float(forward.shape[-1])),
            "flow_jepa_raw_mid_grid_size": forward.new_tensor(float(mid_forward.shape[-1])),
            "flow_jepa_raw_coarse_grid_size": forward.new_tensor(
                float(coarse_shape[-1])
            ),
            "flow_jepa_raw_flow_magnitude": _stable_vector_norm(
                forward, dim=1
            ).mean().detach(),
            "flow_jepa_raw_flow_grid_magnitude": (
                _stable_vector_norm(forward, dim=1).mean() * unit_scale
            ).detach(),
            "flow_jepa_raw_seed_reliability": 0.5 * (
                coarse_reliability_forward.float().mean()
                + coarse_reliability_backward.float().mean()
            ).detach(),
            "flow_jepa_raw_mid_residual_magnitude": _stable_vector_norm(
                joined_mid.flow.float() - joined_mid.iterations[0].float(),
                dim=1,
            ).mean().detach(),
            "flow_jepa_raw_high_residual_magnitude": (
                _stable_vector_norm(
                    joined_high.flow.float() - joined_high.iterations[0].float(),
                    dim=1,
                ).mean().detach()
            ),
            "flow_jepa_raw_mid_boundary_compression": (
                joined_mid.boundary_compression.float().mean().detach()
                if joined_mid.boundary_compression is not None
                else forward.new_zeros(())
            ),
            "flow_jepa_raw_high_boundary_compression": (
                joined_high.boundary_compression.float().mean().detach()
                if joined_high.boundary_compression is not None
                else forward.new_zeros(())
            ),
            "flow_jepa_raw_cycle_core": cycle_core.detach(),
            "flow_jepa_raw_boundary_penalty": boundary_penalty.detach(),
            "flow_jepa_raw_valid_fraction": 0.5 * (
                forward_warp_valid.float().mean() + backward_warp_valid.float().mean()
            ).detach(),
            "flow_jepa_raw_confidence_mean": confidence.float().mean().detach(),
            "flow_jepa_raw_occlusion_fraction": occlusion.float().mean().detach(),
            "flow_jepa_raw_hard_occlusion_fraction": (
                1.0 - hard_visibility.float().mean()
            ).detach(),
            "flow_jepa_visibility_transition_width_min": (
                visibility_transition_width_min
            ),
            "flow_jepa_visibility_gain_bound_max": visibility_gain_bound_max,
            "flow_jepa_complete_numerical_contract": forward.new_tensor(
                float(self.complete_numerical_contract), dtype=torch.float32
            ),
            "flow_jepa_correlation_feature_rms_min": torch.minimum(
                joined_mid.correlation_feature_rms_min,
                joined_high.correlation_feature_rms_min,
            ).detach(),
            "flow_jepa_correlation_norm_denominator_min": torch.minimum(
                joined_mid.correlation_norm_denominator_min,
                joined_high.correlation_norm_denominator_min,
            ).detach(),
            "flow_jepa_correlation_norm_gain_max": torch.maximum(
                joined_mid.correlation_norm_gain_max,
                joined_high.correlation_norm_gain_max,
            ).detach(),
            "flow_jepa_raw_identity_warp_error": 0.5
            * (
                forward_identity_error.float().mean()
                + backward_identity_error.float().mean()
            ),
            "flow_jepa_raw_warp_gain_over_zero": warp_gain_over_zero.detach(),
            "flow_jepa_raw_moving_warp_gain": moving_warp_gain.detach(),
            "flow_jepa_raw_static_warp_gain": static_warp_gain.detach(),
            "flow_jepa_raw_moving_correlation_entropy": moving_correlation_entropy.detach(),
            "flow_jepa_raw_moving_correlation_margin": moving_correlation_margin.detach(),
            "flow_jepa_raw_observable_motion_fraction": observable_motion.float()
            .mean()
            .detach(),
            "flow_jepa_zero_flow_guard": forward.new_tensor(
                float(self.zero_flow_guard), dtype=torch.float32
            ),
            "flow_jepa_bounded_flow_coordinates": forward.new_tensor(
                float(self.bounded_coordinates), dtype=torch.float32
            ),
            "flow_jepa_raw_image_enabled": forward.new_ones((), dtype=torch.float32),
        }
        return context, losses, metrics


class _RawDeformableAddressReader(nn.Module):
    """Grounding-conditioned raw reader with no early spatial pooling."""

    def __init__(
        self,
        raw_dim: int,
        hidden: int,
        grid: int,
        *,
        radius: int,
        heads: int,
        nonduplicate_fallback: bool = False,
        complementary_detail: bool = False,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("raw address reader hidden size must be divisible by heads")
        self.hidden = int(hidden)
        self.grid = int(grid)
        self.radius = int(radius)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.nonduplicate_fallback = bool(nonduplicate_fallback)
        self.complementary_detail = bool(complementary_detail)
        if self.complementary_detail and not self.nonduplicate_fallback:
            raise ValueError(
                "complementary raw detail requires the non-duplicate identity-safe reader"
            )
        self.source_proj = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden))
        self.key_proj = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden))
        self.value_proj = nn.Sequential(
            nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        if self.complementary_detail:
            # Content and detail are different signals, not two candidates in
            # one router. Bias-free detail projections preserve an exact zero
            # when the high-frequency residual is absent.
            self.base_key_proj = nn.Sequential(
                nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden, bias=False)
            )
            self.base_value_proj = nn.Sequential(
                nn.LayerNorm(raw_dim),
                nn.Linear(raw_dim, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, hidden, bias=False),
            )
            self.detail_key_proj = nn.Sequential(
                nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden, bias=False)
            )
            self.detail_value_proj = nn.Sequential(
                nn.LayerNorm(raw_dim),
                nn.Linear(raw_dim, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, hidden, bias=False),
            )
        else:
            self.base_key_proj = None
            self.base_value_proj = None
            self.detail_key_proj = None
            self.detail_value_proj = None
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.selector_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.value_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.query = nn.Parameter(torch.randn(1, grid, grid, hidden) * 0.02)
        self.grounding_scale = nn.Parameter(torch.tensor(0.25))
        self.flow_prior_strength = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _sample(value: Tensor, coordinates: Tensor) -> Tensor:
        side = int(value.shape[-1])
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.float(), _normalize_grid(coordinates, side, side).float(),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )
        return sampled.permute(0, 2, 3, 1)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        flow: Tensor,
        confidence: Tensor,
        grounding_query: Tensor,
        detail_gate: Tensor,
        *,
        post_reader_detail_intervention: str = "none",
        return_detail_residual: bool = False,
    ) -> (
        tuple[Tensor, Tensor, dict[str, Tensor]]
        | tuple[
            Tensor,
            Tensor,
            dict[str, Tensor],
            Tensor | None,
            Tensor | None,
        ]
    ):
        if source.ndim != 4 or tuple(source.shape) != tuple(target.shape):
            raise ValueError("raw address source/target maps must align as [B,C,S,S]")
        batch, _, side, side_b = source.shape
        if side != side_b or tuple(flow.shape) != (batch, 2, side, side):
            raise ValueError("raw address flow must align with the high-resolution map")
        if tuple(confidence.shape) != (batch, 1, side, side):
            raise ValueError("raw address confidence must align with high-resolution flow")
        if tuple(grounding_query.shape) != (batch, self.grid, self.grid, self.hidden):
            raise ValueError("raw grounding query must be [B,G,G,H]")
        if tuple(detail_gate.shape) != (batch, 1, self.grid, self.grid):
            raise ValueError("raw detail gate must be [B,1,G,G]")
        detail_intervention = str(post_reader_detail_intervention).strip().lower()
        if detail_intervention not in {
            "none",
            "measure",
            "zero",
            "spatial_shuffle",
        }:
            raise ValueError(
                "post-reader detail intervention must be "
                "none/measure/zero/spatial_shuffle"
            )
        if detail_intervention != "none" and not self.complementary_detail:
            raise RuntimeError(
                "post-reader detail intervention requires complementary raw detail"
            )
        axis = torch.linspace(0.0, float(side - 1), self.grid, device=source.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
        flow_grid = F.interpolate(
            flow.float(), size=(self.grid, self.grid), mode="bilinear", align_corners=True
        ).permute(0, 2, 3, 1)
        flow_center = base + flow_grid
        # ``radius`` controls samples per coarse reader cell, not a fixed raw
        # pixel radius.  At 336px input the high map is 84x84 while the reader
        # chart is normally 8x8; fixed +/-3-pixel offsets would leave gaps
        # between adjacent cells.  Scaling the offsets to half a cell keeps
        # the same candidate count while covering the complete high-resolution
        # map before the late GxG bottleneck.
        cell_stride = float(max(side - 1, 1)) / float(max(self.grid - 1, 1))
        offset_step = 0.5 * cell_stride / float(max(self.radius, 1))
        offsets = [
            (float(dx) * offset_step, float(dy) * offset_step)
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
        ]
        if self.complementary_detail:
            low_frequency = F.adaptive_avg_pool2d(
                target.float(), (self.grid, self.grid)
            )
            low_frequency_high = F.interpolate(
                low_frequency,
                size=(side, side),
                mode="bilinear",
                align_corners=True,
            )
            candidate_source = (target.float() - low_frequency_high).to(
                dtype=target.dtype
            )
        else:
            low_frequency = None
            candidate_source = target
        candidate_rows: list[Tensor] = []
        valid_rows: list[Tensor] = []
        lane_rows: list[float] = []
        centers = (
            ((True, flow_center),)
            if self.nonduplicate_fallback
            else ((False, base), (True, flow_center))
        )
        for is_flow, center in centers:
            for dx, dy in offsets:
                coordinates = center + center.new_tensor((dx, dy))
                candidate_rows.append(self._sample(candidate_source, coordinates))
                valid_rows.append(
                    (coordinates[..., 0] >= 0.0)
                    & (coordinates[..., 0] <= float(side - 1))
                    & (coordinates[..., 1] >= 0.0)
                    & (coordinates[..., 1] <= float(side - 1))
                )
                lane_rows.append(1.0 if is_flow else 0.0)
        candidate_raw = torch.stack(candidate_rows, dim=3)
        candidate_valid = torch.stack(valid_rows, dim=3)
        source_raw = self._sample(source, base)
        query = (
            self.query.to(device=source.device, dtype=source.dtype)
            + self.source_proj(source_raw.to(dtype=source.dtype))
            + self.grounding_scale.tanh().to(device=source.device, dtype=source.dtype)
            * grounding_query.to(dtype=source.dtype)
        )
        query = self.query_norm(query).reshape(
            batch, self.grid, self.grid, self.heads, self.head_dim
        )
        if self.complementary_detail:
            assert self.detail_key_proj is not None and self.detail_value_proj is not None
            candidate_key = self.detail_key_proj(candidate_raw.to(dtype=source.dtype))
            candidate_value = self.detail_value_proj(candidate_raw.to(dtype=source.dtype))
        else:
            candidate_key = self.key_proj(candidate_raw.to(dtype=source.dtype))
            candidate_value = self.value_proj(candidate_raw.to(dtype=source.dtype))
        key = self.key_norm(candidate_key).reshape(
            batch, self.grid, self.grid, len(lane_rows), self.heads, self.head_dim
        )
        logits = torch.einsum("bijhd,bijkhd->bijhk", query.float(), key.float())
        detail_precision = 0.75 + 0.50 * detail_gate[:, 0].float()
        logits = (
            logits
            * (float(self.head_dim) ** -0.5)
            * detail_precision[:, :, :, None, None]
        )
        confidence_grid = F.interpolate(
            confidence.float(), size=(self.grid, self.grid), mode="bilinear", align_corners=True
        )[:, 0]
        if self.complementary_detail:
            assert low_frequency is not None
            assert self.base_key_proj is not None and self.base_value_proj is not None
            valid_any = candidate_valid.any(dim=3)
            logits = logits.masked_fill(
                ~candidate_valid[:, :, :, None, :], torch.finfo(logits.dtype).min
            )
            logits = torch.where(
                valid_any[:, :, :, None, None], logits, torch.zeros_like(logits)
            )
            detail_weights = torch.softmax(logits, dim=-1)
            detail_value_heads = candidate_value.float().reshape(
                batch, self.grid, self.grid, len(lane_rows), self.heads, self.head_dim
            )
            detail_key_heads = candidate_key.float().reshape_as(detail_value_heads)
            detail_value = torch.einsum(
                "bijhk,bijkhd->bijhd", detail_weights, detail_value_heads
            ).reshape(batch, self.grid, self.grid, self.hidden)
            detail_key = torch.einsum(
                "bijhk,bijkhd->bijhd", detail_weights, detail_key_heads
            ).reshape_as(detail_value)
            base_raw = low_frequency.permute(0, 2, 3, 1).to(dtype=source.dtype)
            base_key = self.base_key_proj(base_raw).float()
            base_value = self.base_value_proj(base_raw).float()
            # Mechanism-level anti-collapse: complete low-frequency content and
            # flow-addressed high-frequency detail are additive complements.
            # There is no lane softmax and no amplitude gate that can delete
            # one route to make the other an easier shortcut.
            read_key = base_key + detail_key
            read_value = base_value + detail_value
            base_norm = _stable_vector_norm(base_value, dim=-1)
            detail_norm = _stable_vector_norm(detail_value, dim=-1)
            detail_share = detail_norm / (base_norm + detail_norm).clamp_min(1e-6)
            flow_mass = detail_share.mean()
            entropy = -(
                detail_weights.clamp_min(1e-8)
                * detail_weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(max(len(lane_rows), 2)))
            lane_value_difference = detail_norm.mean()
            # Invalid candidates carry ``finfo.min`` for softmax masking.
            # Including those sentinels in a plain mean makes max-minus-mean
            # overflow to +inf even though the attention itself is finite.
            candidate_mask = candidate_valid[:, :, :, None, :]
            valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
            valid_mean = torch.where(
                candidate_mask,
                logits,
                torch.zeros_like(logits),
            ).sum(dim=-1) / valid_count
            lane_logit_advantage = (
                logits.max(dim=-1).values - valid_mean
            ).mean()
            candidate_count = len(lane_rows) + 1
        elif self.nonduplicate_fallback:
            # Attend inside the flow-centred neighbourhood first, then compare
            # that precise read with one spatially pooled fallback. Candidate
            # count can no longer make a lane look important, and at zero flow
            # the two lanes are not coordinate-identical copies.
            valid_any = candidate_valid.any(dim=3)
            logits = logits.masked_fill(
                ~candidate_valid[:, :, :, None, :], torch.finfo(logits.dtype).min
            )
            logits = torch.where(
                valid_any[:, :, :, None, None], logits, torch.zeros_like(logits)
            )
            within_flow = torch.softmax(logits, dim=-1)
            value_heads = candidate_value.float().reshape(
                batch, self.grid, self.grid, len(lane_rows), self.heads, self.head_dim
            )
            key_heads = candidate_key.float().reshape_as(value_heads)
            flow_value = torch.einsum("bijhk,bijkhd->bijhd", within_flow, value_heads)
            flow_key = torch.einsum("bijhk,bijkhd->bijhd", within_flow, key_heads)
            fallback_raw = F.adaptive_avg_pool2d(
                target.float(), (self.grid, self.grid)
            ).permute(0, 2, 3, 1)
            fallback_key_flat = self.key_proj(
                fallback_raw.to(dtype=source.dtype)
            ).float()
            fallback_value_flat = self.value_proj(
                fallback_raw.to(dtype=source.dtype)
            ).float()
            fallback_key = fallback_key_flat.reshape(
                batch, self.grid, self.grid, self.heads, self.head_dim
            )
            fallback_value = fallback_value_flat.reshape_as(fallback_key)
            lane_keys = torch.stack((flow_key, fallback_key), dim=3)
            lane_logits = torch.einsum(
                "bijhd,bijkhd->bijhk",
                F.normalize(query.float(), dim=-1),
                F.normalize(lane_keys.float(), dim=-1),
            )
            lane_logits = lane_logits * detail_precision[:, :, :, None, None]
            flow_lane_logit = lane_logits[..., 0] + (
                self.flow_prior_strength.tanh().float()
                * confidence_grid[:, :, :, None]
                * (0.5 + 0.5 * detail_gate[:, 0, :, :, None].float())
            )
            flow_lane_logit = torch.where(
                valid_any[:, :, :, None],
                flow_lane_logit,
                flow_lane_logit.new_full((), -1e4),
            )
            lane_logits = torch.stack((flow_lane_logit, lane_logits[..., 1]), dim=-1)
            lane_weights = torch.softmax(lane_logits, dim=-1)
            lane_values = torch.stack((flow_value, fallback_value), dim=3)
            read_value_heads = torch.einsum(
                "bijhk,bijkhd->bijhd", lane_weights, lane_values
            )
            read_key_heads = torch.einsum(
                "bijhk,bijkhd->bijhd", lane_weights, lane_keys
            )
            read_value = read_value_heads.reshape(
                batch, self.grid, self.grid, self.hidden
            )
            read_key = read_key_heads.reshape(batch, self.grid, self.grid, self.hidden)
            flow_mass = lane_weights[..., 0].mean()
            entropy = -(
                lane_weights.clamp_min(1e-8) * lane_weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(2.0)
            lane_value_difference = _stable_vector_norm(
                flow_value.reshape(batch, self.grid, self.grid, self.hidden)
                - fallback_value_flat,
                dim=-1,
            ).mean()
            lane_logit_delta = lane_logits[..., 0] - lane_logits[..., 1]
            valid_lane_weight = valid_any[:, :, :, None].float()
            lane_logit_advantage = (
                lane_logit_delta * valid_lane_weight
            ).sum() / valid_lane_weight.sum().clamp_min(1.0) / float(self.heads)
            candidate_count = len(lane_rows) + 1
        else:
            lane = source.new_tensor(lane_rows, dtype=torch.float32)[None, None, None, :]
            logits = logits + (
                lane
                * self.flow_prior_strength.tanh().float()
                * confidence_grid[:, :, :, None]
                * (0.5 + 0.5 * detail_gate[:, 0, :, :, None].float())
            )[:, :, :, None]
            logits = logits.masked_fill(
                ~candidate_valid[:, :, :, None, :], torch.finfo(logits.dtype).min
            )
            weights = torch.softmax(logits, dim=-1)
            value_heads = candidate_value.float().reshape(
                batch, self.grid, self.grid, len(lane_rows), self.heads, self.head_dim
            )
            key_heads = candidate_key.float().reshape_as(value_heads)
            read_value = torch.einsum("bijhk,bijkhd->bijhd", weights, value_heads).reshape(
                batch, self.grid, self.grid, self.hidden
            )
            read_key = torch.einsum("bijhk,bijkhd->bijhd", weights, key_heads).reshape(
                batch, self.grid, self.grid, self.hidden
            )
            flow_lane = source.new_tensor(lane_rows, dtype=torch.bool)
            flow_mass = weights[..., flow_lane].sum(dim=-1).mean()
            entropy = -(
                weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(max(len(lane_rows), 2)))
            lane_value_difference = read_value.new_zeros(())
            lane_logit_advantage = read_value.new_zeros(())
            candidate_count = len(lane_rows)
        center_separation = _stable_vector_norm(flow_grid, dim=-1).mean() / max(
            cell_stride, 1e-6
        )
        selector_output = self.selector_out(read_key.to(dtype=source.dtype))
        value_output = self.value_out(read_value.to(dtype=source.dtype))
        output_metrics = {
            "flow_mass": flow_mass.detach(),
            "fallback_mass": (1.0 - flow_mass).detach(),
            "entropy": entropy.mean().detach(),
            "candidate_count": flow_mass.new_tensor(float(candidate_count)),
            "detail_precision": detail_precision.mean().detach(),
            "center_separation": center_separation.detach(),
            "lane_value_difference": lane_value_difference.detach(),
            "lane_logit_advantage": lane_logit_advantage.detach(),
            "additive_detail_path": flow_mass.new_tensor(
                float(self.complementary_detail)
            ),
        }
        selector_detail_residual: Tensor | None = None
        value_detail_residual: Tensor | None = None
        if detail_intervention != "none" or bool(return_detail_residual):
            # Define the post-reader detail residual at the actual interface
            # consumed by DINO fusion.  This keeps the complete base output and
            # all output-projection nonlinearities fixed:
            #   detail_residual := full_reader_output - base_only_reader_output.
            # The intervention therefore cannot accidentally delete DINO/base
            # content or reinterpret a pre-projection hidden component.
            assert self.complementary_detail
            base_selector_output = self.selector_out(base_key.to(dtype=source.dtype))
            base_value_output = self.value_out(base_value.to(dtype=source.dtype))
            selector_detail_residual = selector_output - base_selector_output
            value_detail_residual = value_output - base_value_output
            output_metrics["post_reader_detail_selector_residual_norm"] = (
                _stable_vector_norm(
                    selector_detail_residual.detach().float(), dim=-1
                ).mean()
            )
            output_metrics["post_reader_detail_value_residual_norm"] = (
                _stable_vector_norm(value_detail_residual.detach().float(), dim=-1).mean()
            )
            if detail_intervention != "none":
                original_selector = selector_output
                original_value = value_output
                if detail_intervention == "zero":
                    selector_output = base_selector_output
                    value_output = base_value_output
                elif detail_intervention == "spatial_shuffle":
                    shifts = (
                        max(int(selector_detail_residual.shape[1]) // 2, 1),
                        max(int(selector_detail_residual.shape[2]) // 3, 1),
                    )
                    selector_output = base_selector_output + selector_detail_residual.roll(
                        shifts=shifts, dims=(1, 2)
                    )
                    value_output = base_value_output + value_detail_residual.roll(
                        shifts=shifts, dims=(1, 2)
                    )
                output_metrics["post_reader_detail_selector_intervention_delta"] = (
                    _stable_vector_norm(
                        (original_selector - selector_output).detach().float(), dim=-1
                    ).mean()
                )
                output_metrics["post_reader_detail_value_intervention_delta"] = (
                    _stable_vector_norm(
                        (original_value - value_output).detach().float(), dim=-1
                    ).mean()
                )
                # The policy-boundary payload must represent the intervention
                # actually consumed downstream, not the pre-intervention probe
                # value used only to measure the causal delta.
                selector_detail_residual = selector_output - base_selector_output
                value_detail_residual = value_output - base_value_output
        if bool(return_detail_residual):
            return (
                selector_output,
                value_output,
                output_metrics,
                selector_detail_residual,
                value_detail_residual,
            )
        return selector_output, value_output, output_metrics


class _SparseFineFlowRefiner(nn.Module):
    """Native-DINO local correlation only at the bounded reader chart.

    The coarse flow is converted to native-patch coordinates.  A soft
    importance gate controls only the fine residual, so low-importance regions
    retain a valid coarse route and gradients are never cut by hard top-k.
    """

    def __init__(
        self, input_dim: int, hidden: int, *, radius: int, grid: int, uncertainty_floor: float
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.radius = int(radius)
        self.grid = int(grid)
        self.uncertainty_floor = float(uncertainty_floor)
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(3 * hidden + 5),
            nn.Linear(3 * hidden + 5, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, 3),
        )
        nn.init.normal_(self.update[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)

    @staticmethod
    def _sample(value: Tensor, coordinates: Tensor) -> Tensor:
        side = int(value.shape[-1])
        grid = _normalize_grid(coordinates, side, side)
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.float(), grid.float(), mode="bilinear", padding_mode="zeros", align_corners=True
            )
        return sampled.permute(0, 2, 3, 1)

    def forward(
        self,
        first: Tensor,
        second: Tensor,
        coarse_flow: Tensor,
        importance: Tensor,
    ) -> PatchFlowEstimate:
        if first.ndim != 4 or tuple(first.shape) != tuple(second.shape):
            raise ValueError("sparse fine flow inputs must align as [B,S,S,D]")
        batch, side, side_b, _ = first.shape
        if side != side_b or tuple(coarse_flow.shape) != (batch, 2, self.grid, self.grid):
            raise ValueError("sparse fine flow coarse chart has invalid geometry")
        if tuple(importance.shape) != (batch, 1, self.grid, self.grid):
            raise ValueError("sparse fine flow importance must be [B,1,G,G]")
        first_feature = self.feature(first).permute(0, 3, 1, 2)
        second_feature = self.feature(second).permute(0, 3, 1, 2)
        axis = torch.linspace(0.0, float(side - 1), self.grid, device=first.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
        scale = float(max(side - 1, 1)) / float(max(self.grid - 1, 1))
        coarse_native = coarse_flow.float() * scale
        center = base + coarse_native.permute(0, 2, 3, 1)
        source = self._sample(first_feature, base)
        offsets = [
            (dx, dy)
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
        ]
        candidates: list[Tensor] = []
        valid_rows: list[Tensor] = []
        for dx, dy in offsets:
            coordinates = center + center.new_tensor((float(dx), float(dy)))
            candidates.append(self._sample(second_feature, coordinates))
            valid_rows.append(
                (coordinates[..., 0] >= 0.0)
                & (coordinates[..., 0] <= float(side - 1))
                & (coordinates[..., 1] >= 0.0)
                & (coordinates[..., 1] <= float(side - 1))
            )
        candidate = torch.stack(candidates, dim=3)
        valid = torch.stack(valid_rows, dim=3)
        logits = (
            F.normalize(source.float(), dim=-1)[:, :, :, None]
            * F.normalize(candidate.float(), dim=-1)
        ).sum(dim=-1)
        valid_any = valid.any(dim=3, keepdim=True)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        # A badly initialized or genuinely off-screen coarse displacement can
        # put every fine candidate outside the image.  A symmetric uniform
        # fallback has zero expected residual and stays finite; the coarse
        # address remains intact and its downstream reader can use identity.
        logits = torch.where(valid_any, logits, torch.zeros_like(logits))
        probability = torch.softmax(logits, dim=3)
        offset_tensor = coarse_flow.new_tensor(offsets, dtype=torch.float32)
        expected = torch.einsum("bijk,kd->bijd", probability, offset_tensor)
        aligned = (probability[..., None] * candidate.float()).sum(dim=3)
        entropy = -(
            probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log()
        ).sum(dim=3) / math.log(float(max(len(offsets), 2)))
        top2 = probability.topk(k=min(2, len(offsets)), dim=3).values
        margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else 0.0)
        importance_hw = importance[:, 0].float().clamp(0.0, 1.0)
        update_input = torch.cat(
            (
                source.float(),
                aligned,
                source.float() - aligned,
                coarse_native.permute(0, 2, 3, 1) / float(max(side - 1, 1)),
                entropy[..., None],
                margin[..., None],
                importance_hw[..., None],
            ),
            dim=-1,
        )
        learned = self.update(update_input.to(dtype=source.dtype)).float()
        fine_residual = expected + 0.25 * torch.tanh(learned[..., :2])
        flow_native = coarse_native + (
            importance_hw[..., None] * fine_residual
        ).permute(0, 3, 1, 2)
        uncertainty = (
            F.softplus(learned[..., 2:3]).permute(0, 3, 1, 2) + self.uncertainty_floor
        )
        entropy_map = entropy[:, None]
        margin_map = margin[:, None]
        return PatchFlowEstimate(
            flow=flow_native,
            information=torch.cat(
                (entropy_map, margin_map, uncertainty, importance.float()), dim=1
            ),
            uncertainty=uncertainty,
            correlation_entropy=entropy_map,
            correlation_margin=margin_map,
            iterations=(coarse_native, flow_native),
        )


class _SoftFlowAddressReader(nn.Module):
    """Late local reader with flow-addressed and identity fallback candidates."""

    def __init__(self, hidden: int, grid: int, *, radius: int, heads: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("soft address reader hidden size must be divisible by heads")
        self.hidden = int(hidden)
        self.grid = int(grid)
        self.radius = int(radius)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.query = nn.Parameter(torch.randn(1, grid, grid, hidden) * 0.02)
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.selector_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.value_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.flow_prior_strength = nn.Parameter(torch.tensor(1.0))
        self.identity_prior = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _sample(value: Tensor, coordinates: Tensor) -> Tensor:
        side = int(value.shape[-1])
        grid = _normalize_grid(coordinates, side, side)
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.float(),
                grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        return sampled.permute(0, 2, 3, 1)

    def forward(
        self,
        source_key: Tensor,
        target_key: Tensor,
        target_value: Tensor,
        flow: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if (
            source_key.ndim != 4
            or tuple(source_key.shape) != tuple(target_key.shape)
            or tuple(target_key.shape) != tuple(target_value.shape)
        ):
            raise ValueError(
                "soft address source/target key and value maps must align as [B,H,S,S]"
            )
        batch, hidden, side, side_b = source_key.shape
        if hidden != self.hidden or side != side_b:
            raise ValueError("soft address reader received invalid map geometry")
        if tuple(flow.shape) != (batch, 2, self.grid, self.grid):
            raise ValueError("soft address flow must be [B,2,G,G] in native-patch units")
        if tuple(confidence.shape) != (batch, 1, self.grid, self.grid):
            raise ValueError("soft address confidence must be [B,1,G,G]")
        axis = torch.linspace(0.0, float(side - 1), self.grid, device=source_key.device)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
        flow_at_base = flow.permute(0, 2, 3, 1).float()
        flow_center = base + flow_at_base
        offsets = [
            (dx, dy)
            for dy in range(-self.radius, self.radius + 1)
            for dx in range(-self.radius, self.radius + 1)
        ]
        sampled_keys: list[Tensor] = []
        sampled_values: list[Tensor] = []
        sampled_confidence: list[Tensor] = []
        valid_rows: list[Tensor] = []
        lane_rows: list[float] = []
        for is_flow, center in ((False, base), (True, flow_center)):
            for dx, dy in offsets:
                coordinates = center + center.new_tensor((float(dx), float(dy)))
                sampled_keys.append(self._sample(target_key, coordinates))
                sampled_values.append(self._sample(target_value, coordinates))
                sampled_confidence.append(
                    confidence.permute(0, 2, 3, 1).float()
                )
                valid_rows.append(
                    (coordinates[..., 0] >= 0.0)
                    & (coordinates[..., 0] <= float(side - 1))
                    & (coordinates[..., 1] >= 0.0)
                    & (coordinates[..., 1] <= float(side - 1))
                )
                lane_rows.append(1.0 if is_flow else 0.0)
        candidate_key = torch.stack(sampled_keys, dim=3)
        candidate_value = torch.stack(sampled_values, dim=3)
        candidate_confidence = torch.stack(sampled_confidence, dim=3).clamp(0.0, 1.0)
        candidate_valid = torch.stack(valid_rows, dim=3)
        lane = source_key.new_tensor(lane_rows, dtype=torch.float32)[None, None, None, :, None]
        # The address query belongs to the source frame.  Constructing it from
        # the target identity location would create an architectural zero-flow
        # shortcut before the learned flow has any chance to act as an address.
        pooled_key = F.adaptive_avg_pool2d(
            source_key.float(), (self.grid, self.grid)
        ).permute(
            0, 2, 3, 1
        )
        query = self.query.to(
            device=source_key.device, dtype=source_key.dtype
        ) + pooled_key.to(dtype=source_key.dtype)
        query = self.query_norm(query).reshape(
            batch, self.grid, self.grid, self.heads, self.head_dim
        )
        candidate_key_n = self.key_norm(candidate_key.to(dtype=source_key.dtype)).reshape(
            batch,
            self.grid,
            self.grid,
            len(lane_rows),
            self.heads,
            self.head_dim,
        )
        logits = torch.einsum("bijhd,bijkhd->bijhk", query.float(), candidate_key_n.float())
        logits = logits * (float(self.head_dim) ** -0.5)
        lane_bias = (
            lane[..., 0]
            * self.flow_prior_strength.to(device=source_key.device, dtype=torch.float32).tanh()
            * candidate_confidence[..., 0]
            + (1.0 - lane[..., 0])
            * self.identity_prior.to(device=source_key.device, dtype=torch.float32).tanh()
        )
        logits = logits + lane_bias[..., None, :]
        logits = logits.masked_fill(
            ~candidate_valid[:, :, :, None, :], torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=-1)
        value_heads = candidate_value.float().reshape(
            batch,
            self.grid,
            self.grid,
            len(lane_rows),
            self.heads,
            self.head_dim,
        )
        key_heads = candidate_key.float().reshape_as(value_heads)
        read_value = torch.einsum("bijhk,bijkhd->bijhd", weights, value_heads).reshape(
            batch, self.grid, self.grid, hidden
        )
        read_key = torch.einsum("bijhk,bijkhd->bijhd", weights, key_heads).reshape(
            batch, self.grid, self.grid, hidden
        )
        flow_lane = source_key.new_tensor(lane_rows, dtype=torch.float32).bool()
        flow_mass = weights[..., flow_lane].sum(dim=-1).mean()
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(float(max(len(lane_rows), 2)))
        return (
            self.selector_out(read_key.to(dtype=target_key.dtype)),
            self.value_out(read_value.to(dtype=target_value.dtype)),
            {
                "flow_mass": flow_mass.detach(),
                "fallback_mass": (1.0 - flow_mass).detach(),
                "entropy": entropy.mean().detach(),
            },
        )


class _SoftMultiResolutionAddressCompiler(nn.Module):
    """Compile a reusable multi-slot DINO/raw address bank.

    Unlike the historical readers, this module never emits one final detail
    vector per 8x8 cell.  It keeps several global DINO hypotheses per source
    cell and all continuous high-resolution candidates around those
    hypotheses.  Flow contributes a soft geometric prior only.
    """

    COARSE_GEOMETRY_DIM = 11
    FINE_GEOMETRY_DIM = 13

    def __init__(self, config: Any, *, raw_dim: int) -> None:
        super().__init__()
        self.grid = int(config.flow_jepa_grid_size)
        self.cameras = int(config.num_cameras)
        self.slots = int(config.flow_jepa_address_slots)
        self.route_dim = int(config.flow_jepa_address_route_dim)
        self.radius = int(config.flow_jepa_raw_reader_radius)
        self.raw_dim = int(raw_dim)
        self.flow_prior_floor = float(
            getattr(config, "flow_jepa_address_flow_prior_floor", 0.0)
        )
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        self.progressive_grounding_address = bool(
            int(getattr(config, "flow_jepa_progressive_grounding_address", 0))
        )
        self.coordinate_typed_raw_detail = bool(
            int(getattr(config, "flow_jepa_coordinate_typed_raw_detail", 0))
        )
        normalization_floor = float(
            getattr(config, "flow_jepa_routing_norm_floor", 0.25)
        )
        dino_dim = int(config.visual_token_dim)
        route = self.route_dim

        def affine_route_norm(width: int) -> nn.Module:
            if self.complete_numerical_contract:
                return AffineVarianceFlooredCenteredNorm(
                    width,
                    normalization_floor,
                    affine_maximum=4.0,
                )
            return nn.LayerNorm(width)

        self.source_dino = nn.Sequential(
            affine_route_norm(dino_dim), nn.Linear(dino_dim, route, bias=False)
        )
        self.target_dino_key = nn.Sequential(
            affine_route_norm(dino_dim), nn.Linear(dino_dim, route, bias=False)
        )
        # V109 keeps one semantic key chart and lets the typed G stages build
        # their own selector state.  The separate V103-V108 coarse value
        # projection only fed the retired compiler-owned coarse key, so keeping
        # it trainable in V109 would create an optimizer-owned dead branch.
        self.target_dino_value = (
            None
            if self.progressive_grounding_address
            else nn.Sequential(
                affine_route_norm(dino_dim), nn.Linear(dino_dim, route, bias=False)
            )
        )
        self.fine_dino_key = nn.Sequential(
            affine_route_norm(dino_dim), nn.Linear(dino_dim, route, bias=False)
        )
        self.raw_key = nn.Sequential(
            nn.GroupNorm(_group_count(self.raw_dim), self.raw_dim),
            nn.Conv2d(self.raw_dim, route, 1, bias=False),
        )
        self.source_raw_key = nn.Sequential(
            nn.GroupNorm(_group_count(self.raw_dim), self.raw_dim),
            nn.Conv2d(self.raw_dim, route, 1, bias=False),
        )
        self.raw_pair_key = nn.Sequential(
            affine_route_norm(4 * route),
            nn.Linear(4 * route, 2 * route, bias=False),
            nn.GELU(),
            nn.Linear(2 * route, route, bias=False),
        )
        self.coarse_geometry = (
            None
            if self.progressive_grounding_address
            else nn.Sequential(
                affine_route_norm(self.COARSE_GEOMETRY_DIM),
                nn.Linear(self.COARSE_GEOMETRY_DIM, route, bias=False),
            )
        )
        self.fine_geometry = nn.Sequential(
            affine_route_norm(self.FINE_GEOMETRY_DIM),
            nn.Linear(self.FINE_GEOMETRY_DIM, route, bias=False),
        )
        self.slot_query = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.slots, route) * 0.02
        )
        self.camera_query = nn.Parameter(
            torch.randn(1, self.cameras, 1, 1, 1, route) * 0.02
        )
        self.cell_query = nn.Parameter(
            torch.randn(1, 1, self.grid, self.grid, 1, route) * 0.02
        )
        self.query_norm = (
            VarianceFlooredCenteredNorm(normalization_floor)
            if self.complete_numerical_contract
            else nn.LayerNorm(route, elementwise_affine=False)
        )
        self.key_norm = (
            VarianceFlooredCenteredNorm(normalization_floor)
            if self.complete_numerical_contract
            else nn.LayerNorm(route, elementwise_affine=False)
        )
        self.content_log_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        learnable_prior_init = max(1.0 - self.flow_prior_floor, 1e-3)
        self.flow_prior_log_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(learnable_prior_init)))
        )
        axis = torch.linspace(-1.0, 1.0, 2 * self.radius + 1)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer(
            "fine_offset_lattice",
            torch.stack((xx, yy), dim=-1).reshape(-1, 2),
            persistent=False,
        )

    @staticmethod
    def _sample_chart(value: Tensor, coordinates: Tensor) -> Tensor:
        """Sample [B,C,D,S,S] maps at [...,K,2] pixel coordinates."""

        if value.ndim != 5 or coordinates.ndim != 7:
            raise ValueError(
                "soft address sampling expects map [B,C,D,S,S] and "
                "coordinates [B,C,G,G,M,K,2]"
            )
        batch, cameras, channels, side, side_b = value.shape
        if side != side_b or tuple(coordinates.shape[:2]) != (batch, cameras):
            raise ValueError("soft address sampling geometry is not aligned")
        grid_shape = coordinates.shape[2:-1]
        flat_grid = _normalize_grid(coordinates, side, side).reshape(
            batch * cameras,
            int(math.prod(grid_shape[:-1])),
            int(grid_shape[-1]),
            2,
        )
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.reshape(batch * cameras, channels, side, side).float(),
                flat_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).to(dtype=value.dtype)
        return sampled.reshape(
            batch,
            cameras,
            channels,
            *grid_shape,
        ).permute(0, 1, 3, 4, 5, 6, 2)

    @staticmethod
    def _sample_normalized_chart(value: Tensor, coordinates: Tensor) -> Tensor:
        """Sample maps at normalized continuous G1/G2 candidate coordinates."""

        if value.ndim != 5 or coordinates.ndim != 7:
            raise ValueError(
                "progressive address sampling expects [B,C,D,S,S] maps and "
                "[B,C,G,G,M,K,2] normalized coordinates"
            )
        batch, cameras, channels, side, side_b = value.shape
        if side != side_b or tuple(coordinates.shape[:2]) != (batch, cameras):
            raise ValueError("progressive address chart geometry is not aligned")
        grid_shape = coordinates.shape[2:-1]
        flat_grid = coordinates.reshape(
            batch * cameras,
            int(math.prod(grid_shape[:-1])),
            int(grid_shape[-1]),
            2,
        )
        with torch.autocast(device_type=value.device.type, enabled=False):
            sampled = F.grid_sample(
                value.reshape(batch * cameras, channels, side, side).float(),
                flat_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).to(dtype=value.dtype)
        return sampled.reshape(
            batch,
            cameras,
            channels,
            *grid_shape,
        ).permute(0, 1, 3, 4, 5, 6, 2)

    def progressive_fine_candidates(
        self,
        bank: SoftAddressLatticeBank,
        *,
        centers: Tensor,
        support: Tensor,
        variance: Tensor,
        aligned_keys: Tensor,
        collect_diagnostics: bool = True,
    ) -> ProgressiveFineCandidates:
        """Materialize G2 candidates around the corrected G1 modes.

        This samples candidate keys and narrow raw-detail values but performs
        no query-dependent value aggregation.  The latter remains owned by
        the W->P reader.
        """

        required = (
            bank.coarse_source_centers,
            bank.coarse_flow_centers,
            bank.dense_source_raw_keys,
            bank.dense_target_raw_keys,
            bank.dense_target_dino_keys,
            bank.dense_target_detail,
            bank.dense_confidence,
            bank.dense_uncertainty,
            bank.dense_occlusion,
        )
        if self.coordinate_typed_raw_detail:
            required = (*required, bank.dense_current_rgb)
        if not all(torch.is_tensor(value) for value in required):
            raise RuntimeError(
                "progressive fine resampling requires the retained dense observation charts"
            )
        assert bank.coarse_source_centers is not None
        assert bank.coarse_flow_centers is not None
        assert bank.dense_source_raw_keys is not None
        assert bank.dense_target_raw_keys is not None
        assert bank.dense_target_dino_keys is not None
        assert bank.dense_target_detail is not None
        assert bank.dense_confidence is not None
        assert bank.dense_uncertainty is not None
        assert bank.dense_occlusion is not None
        if centers.ndim != 6 or int(centers.shape[-1]) != 2:
            raise ValueError("progressive fine centers must be [B,C,G,G,M,2]")
        if tuple(support.shape) != tuple(centers.shape[:-1]):
            raise ValueError("progressive fine support does not align with centers")
        if tuple(variance.shape) != tuple(centers.shape):
            raise ValueError("progressive fine variance does not align with centers")
        if tuple(aligned_keys.shape[:-1]) != tuple(centers.shape[:-1]):
            raise ValueError("progressive aligned keys do not align with centers")

        offsets = self.fine_offset_lattice.to(
            device=centers.device, dtype=centers.dtype
        )
        offset_shape = (1, 1, 1, 1, 1, int(offsets.shape[0]), 2)
        normalized_offsets = offsets.reshape(offset_shape)
        target_coordinates = (
            centers[..., None, :]
            + support[..., None, None] * normalized_offsets
        )
        source_centers = bank.coarse_source_centers.to(
            device=centers.device, dtype=centers.dtype
        )[..., None, :].expand_as(centers)
        source_coordinates = (
            source_centers[..., None, :]
            + support[..., None, None] * normalized_offsets
        )
        target_valid = (
            (target_coordinates[..., 0] >= -1.0)
            & (target_coordinates[..., 0] <= 1.0)
            & (target_coordinates[..., 1] >= -1.0)
            & (target_coordinates[..., 1] <= 1.0)
        )
        source_valid = (
            (source_coordinates[..., 0] >= -1.0)
            & (source_coordinates[..., 0] <= 1.0)
            & (source_coordinates[..., 1] >= -1.0)
            & (source_coordinates[..., 1] <= 1.0)
        )
        valid = target_valid & source_valid
        target_raw_key = self._sample_normalized_chart(
            bank.dense_target_raw_keys, target_coordinates
        )
        source_raw_key = self._sample_normalized_chart(
            bank.dense_source_raw_keys, source_coordinates
        )
        raw_pair_key = self.raw_pair_key(
            torch.cat(
                (
                    source_raw_key,
                    target_raw_key,
                    source_raw_key * target_raw_key,
                    target_raw_key - source_raw_key,
                ),
                dim=-1,
            )
        )
        dino_key = self._sample_normalized_chart(
            bank.dense_target_dino_keys, target_coordinates
        )
        metadata = torch.cat(
            (
                bank.dense_confidence,
                bank.dense_uncertainty,
                bank.dense_occlusion,
            ),
            dim=2,
        )
        sampled_metadata = self._sample_normalized_chart(
            metadata, target_coordinates
        ).float()
        raw_side = int(bank.dense_target_detail.shape[-1])
        normalized_std = variance.float().clamp_min(0.0).sqrt()[..., None, :].expand(
            -1, -1, -1, -1, -1, int(offsets.shape[0]), -1
        )
        normalized_flow_delta = (
            centers.float()
            - bank.coarse_flow_centers.float()[..., None, :]
        )[..., None, :].expand_as(normalized_std)
        geometry = torch.cat(
            (
                target_coordinates.float(),
                normalized_offsets.expand_as(target_coordinates).float(),
                sampled_metadata[..., 0:1],
                sampled_metadata[..., 1:2]
                / float(max(raw_side - 1, 1)),
                sampled_metadata[..., 2:3],
                normalized_std,
                normalized_flow_delta,
                target_valid[..., None].float(),
                source_valid[..., None].float(),
            ),
            dim=-1,
        )
        if int(geometry.shape[-1]) != self.FINE_GEOMETRY_DIM:
            raise RuntimeError("progressive fine geometry width changed unexpectedly")
        semantic_keys = dino_key + aligned_keys[..., None, :]
        appearance_keys = target_raw_key + raw_pair_key
        geometry_keys = self.fine_geometry(
            geometry.to(dtype=target_raw_key.dtype)
        )
        fine_keys = semantic_keys + appearance_keys + geometry_keys
        fine_values = self._sample_normalized_chart(
            bank.dense_target_detail, target_coordinates
        )
        fine_values = fine_values * valid[..., None].to(dtype=fine_values.dtype)
        literal_rgb = (
            self._sample_normalized_chart(
                bank.dense_current_rgb, target_coordinates
            )
            if bank.dense_current_rgb is not None
            else None
        )
        if literal_rgb is not None:
            literal_rgb = literal_rgb * valid[..., None].to(dtype=literal_rgb.dtype)
        return ProgressiveFineCandidates(
            combined_keys=fine_keys,
            learned_detail=fine_values,
            valid=valid,
            current_coordinates=target_coordinates,
            source_coordinates=source_coordinates,
            semantic_keys=semantic_keys if self.coordinate_typed_raw_detail else None,
            appearance_keys=(
                appearance_keys if self.coordinate_typed_raw_detail else None
            ),
            geometry_keys=geometry_keys if self.coordinate_typed_raw_detail else None,
            literal_rgb=literal_rgb if self.coordinate_typed_raw_detail else None,
            metrics={
                "flow_jepa_progressive_g2_dynamic_candidate_valid": valid.detach()
                .float()
                .mean(),
                "flow_jepa_progressive_g2_dynamic_center_distance": (
                    target_coordinates.detach().float()
                    - bank.fine_coordinates.float()
                )
                .square()
                .sum(dim=-1)
                .sqrt()
                .mean()
                if bank.fine_coordinates is not None
                and tuple(bank.fine_coordinates.shape)
                == tuple(target_coordinates.shape)
                else centers.new_zeros((), dtype=torch.float32),
                "flow_jepa_progressive_current_coordinate_rms": (
                    target_coordinates.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_progressive_literal_rgb_candidate_rms": (
                    literal_rgb.detach().float().square().mean().sqrt()
                    if literal_rgb is not None
                    else centers.new_zeros((), dtype=torch.float32)
                ),
            }
            if collect_diagnostics
            else {},
        )

    @staticmethod
    def _grid_axis(
        side: int, grid: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        axis = torch.linspace(0.0, float(side - 1), grid, device=device, dtype=dtype)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((x, y), dim=-1)

    @staticmethod
    def _normalize_xy(value: Tensor, side: int) -> Tensor:
        return 2.0 * value / float(max(side - 1, 1)) - 1.0

    def forward(
        self,
        *,
        source_dino: Tensor,
        target_dino: Tensor,
        source_raw: Tensor,
        target_raw: Tensor,
        current_rgb: Tensor | None = None,
        flow: Tensor,
        confidence: Tensor,
        uncertainty: Tensor,
        occlusion: Tensor,
        cycle_error: Tensor | None = None,
    ) -> tuple[SoftAddressLatticeBank, dict[str, Tensor]]:
        if source_dino.ndim != 5 or tuple(source_dino.shape) != tuple(target_dino.shape):
            raise ValueError("soft address DINO charts must align as [B,C,S,S,D]")
        if source_raw.ndim != 5 or tuple(source_raw.shape) != tuple(target_raw.shape):
            raise ValueError("soft address raw charts must align as [B,C,D,R,R]")
        batch, cameras, dino_side, dino_side_b, _ = source_dino.shape
        raw_batch, raw_cameras, raw_dim, raw_side, raw_side_b = source_raw.shape
        if (
            cameras != self.cameras
            or dino_side != dino_side_b
            or (raw_batch, raw_cameras, raw_dim)
            != (batch, cameras, self.raw_dim)
            or raw_side != raw_side_b
        ):
            raise ValueError("soft address observation bank geometry is invalid")
        expected_map = (batch, cameras, 1, raw_side, raw_side)
        if tuple(confidence.shape) != expected_map or tuple(uncertainty.shape) != expected_map:
            raise ValueError("soft address confidence/uncertainty maps are invalid")
        if tuple(occlusion.shape) != expected_map:
            raise ValueError("soft address occlusion map is invalid")
        if cycle_error is not None and tuple(cycle_error.shape) != expected_map:
            raise ValueError("soft address cycle-error map is invalid")
        if tuple(flow.shape) != (batch, cameras, 2, raw_side, raw_side):
            raise ValueError("soft address flow map is invalid")
        if self.coordinate_typed_raw_detail:
            if current_rgb is None:
                raise ValueError("coordinate-typed address bank requires current RGB")
            if (
                current_rgb.ndim != 5
                or tuple(current_rgb.shape[:3]) != (batch, cameras, 3)
                or int(current_rgb.shape[-2]) != int(current_rgb.shape[-1])
            ):
                raise ValueError(
                    "coordinate-typed current RGB must be [B,C,3,R,R]"
                )
            # Keep the original RGB sampling chart.  Normalized address
            # coordinates are resolution independent, so downsampling this
            # three-channel value owner to the learned raw-feature side would
            # throw away precisely the sub-cell detail the late read exists to
            # preserve.  Raw inputs are dataset pixels in [0,1]; this fixed
            # affine chart is bounded, zero-centred, and has no learned gate.
            literal_rgb_chart = (2.0 * current_rgb.float() - 1.0).to(
                dtype=target_raw.dtype
            )
        else:
            literal_rgb_chart = None

        source_nchw = source_dino.permute(0, 1, 4, 2, 3).reshape(
            batch * cameras, int(source_dino.shape[-1]), dino_side, dino_side
        )
        source_grid = F.adaptive_avg_pool2d(
            source_nchw.float(), (self.grid, self.grid)
        ).reshape(
            batch, cameras, int(source_dino.shape[-1]), self.grid, self.grid
        ).permute(0, 1, 3, 4, 2).to(dtype=source_dino.dtype)
        source_key = self.source_dino(source_grid)
        coarse_query = (
            source_key[:, :, :, :, None]
            + self.slot_query.to(device=source_key.device, dtype=source_key.dtype)
            + self.camera_query.to(device=source_key.device, dtype=source_key.dtype)
            + self.cell_query.to(device=source_key.device, dtype=source_key.dtype)
        )
        target_key = self.target_dino_key(target_dino).reshape(
            batch, cameras, dino_side * dino_side, self.route_dim
        )
        target_value = (
            target_key
            if self.target_dino_value is None
            else self.target_dino_value(target_dino).reshape_as(target_key)
        )
        with torch.autocast(device_type=source_dino.device.type, enabled=False):
            content_logits = torch.einsum(
                "bcijmr,bcpr->bcijmp",
                self.query_norm(coarse_query).float(),
                self.key_norm(target_key).float(),
            )
            content_scale = self.content_log_scale.float().exp().clamp(1.0, 16.0)
            content_logits = (
                content_logits
                * content_scale
                * float(self.route_dim) ** -0.5
            )

            flow_grid = F.interpolate(
                flow.reshape(batch * cameras, 2, raw_side, raw_side).float(),
                size=(self.grid, self.grid),
                mode="bilinear",
                align_corners=True,
            ).reshape(batch, cameras, 2, self.grid, self.grid).permute(0, 1, 3, 4, 2)
            source_high = self._grid_axis(
                raw_side,
                self.grid,
                device=source_dino.device,
                dtype=torch.float32,
            )[None, None]
            flow_center_high = source_high + flow_grid
            flow_center_dino = (
                flow_center_high
                * float(max(dino_side - 1, 1))
                / float(max(raw_side - 1, 1))
            )
            target_axis = self._grid_axis(
                dino_side,
                dino_side,
                device=source_dino.device,
                dtype=torch.float32,
            ).reshape(1, 1, 1, 1, 1, dino_side * dino_side, 2)
            distance_square = (
                target_axis - flow_center_dino[:, :, :, :, None, None]
            ).square().sum(dim=-1)

            def pooled_scalar(value: Tensor) -> Tensor:
                return F.interpolate(
                    value.reshape(batch * cameras, 1, raw_side, raw_side).float(),
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=True,
                ).reshape(batch, cameras, self.grid, self.grid)

            confidence_grid = pooled_scalar(confidence).clamp(0.0, 1.0)
            uncertainty_grid = pooled_scalar(uncertainty).clamp_min(0.0)
            occlusion_grid = pooled_scalar(occlusion).clamp(0.0, 1.0)
            cycle_grid = (
                pooled_scalar(cycle_error).clamp_min(0.0)
                if cycle_error is not None
                else torch.zeros_like(occlusion_grid)
            )
            uncertainty_dino = (
                uncertainty_grid
                * float(max(dino_side - 1, 1))
                / float(max(raw_side - 1, 1))
            )
            sigma = (
                0.75
                + 2.0 * (1.0 - confidence_grid)
                + 2.0 * occlusion_grid
                + uncertainty_dino
            ).clamp(0.75, float(max(dino_side, 1)))
            adaptive_flow_prior = (
                -0.5
                * distance_square
                / sigma[:, :, :, :, None, None].square()
            )
            # A coefficient floor on the adaptive prior is not a real spatial
            # floor: confidence/uncertainty can widen sigma until that prior is
            # nearly constant. Keep one mild confidence-independent geometric
            # expert at roughly one coarse-cell width, then let the learned
            # adaptive expert widen when correspondence is ambiguous.
            floor_sigma = max(
                float(dino_side) / float(max(self.grid, 1)),
                0.75,
            )
            floor_flow_prior = (
                -0.5 * distance_square / float(floor_sigma**2)
            )
            learned_prior_scale = F.softplus(self.flow_prior_log_scale.float())
            adaptive_prior_scale = learned_prior_scale.clamp(
                0.0,
                max(4.0 - self.flow_prior_floor, 0.0),
            )
            prior_scale = self.flow_prior_floor + adaptive_prior_scale
            floor_flow_bias = self.flow_prior_floor * floor_flow_prior
            adaptive_flow_bias = adaptive_prior_scale * adaptive_flow_prior
            coarse_logits = (
                content_logits
                + floor_flow_bias
                + adaptive_flow_bias
            )
            coarse_probability = torch.softmax(coarse_logits, dim=-1)

            target_coordinates = target_axis.reshape(
                1, 1, dino_side * dino_side, 2
            )
            coarse_centers = torch.einsum(
                "bcijmp,bcpd->bcijmd",
                coarse_probability,
                target_coordinates.expand(batch, cameras, -1, -1),
            )
            centered = (
                target_axis
                - coarse_centers[:, :, :, :, :, None]
            )
            coarse_variance = torch.einsum(
                "bcijmp,bcijmpd->bcijmd",
                coarse_probability,
                centered.square(),
            )
            coarse_content = torch.einsum(
                "bcijmp,bcpr->bcijmr",
                coarse_probability,
                target_value.float(),
            )

        source_coordinates_dino = self._grid_axis(
            dino_side,
            self.grid,
            device=source_dino.device,
            dtype=torch.float32,
        )[None, None, :, :, None].expand(
            batch, cameras, -1, -1, self.slots, -1
        )
        flow_delta = coarse_centers - flow_center_dino[:, :, :, :, None]
        coarse_geometry = torch.cat(
            (
                self._normalize_xy(source_coordinates_dino, dino_side),
                self._normalize_xy(coarse_centers, dino_side),
                coarse_variance.clamp_min(0.0).sqrt()
                / float(max(dino_side - 1, 1)),
                flow_delta / float(max(dino_side - 1, 1)),
                confidence_grid[:, :, :, :, None, None].expand(
                    -1, -1, -1, -1, self.slots, -1
                ),
                uncertainty_dino[:, :, :, :, None, None].expand(
                    -1, -1, -1, -1, self.slots, -1
                )
                / float(max(dino_side - 1, 1)),
                occlusion_grid[:, :, :, :, None, None].expand(
                    -1, -1, -1, -1, self.slots, -1
                ),
            ),
            dim=-1,
        )
        if int(coarse_geometry.shape[-1]) != self.COARSE_GEOMETRY_DIM:
            raise RuntimeError("coarse address geometry width changed unexpectedly")
        coarse_geometry_key = (
            torch.zeros_like(coarse_content, dtype=source_key.dtype)
            if self.coarse_geometry is None
            else self.coarse_geometry(
                coarse_geometry.to(dtype=source_key.dtype)
            )
        )
        coarse_keys = (
            coarse_content.to(dtype=source_key.dtype)
            + source_key[:, :, :, :, None]
            + self.slot_query.to(device=source_key.device, dtype=source_key.dtype)
            + self.camera_query.to(device=source_key.device, dtype=source_key.dtype)
            + self.cell_query.to(device=source_key.device, dtype=source_key.dtype)
            + coarse_geometry_key
        )

        center_high = (
            coarse_centers
            * float(max(raw_side - 1, 1))
            / float(max(dino_side - 1, 1))
        )
        coarse_std_high = (
            coarse_variance.clamp_min(0.0).sqrt()
            * float(max(raw_side - 1, 1))
            / float(max(dino_side - 1, 1))
        )
        cell_stride = float(max(raw_side - 1, 1)) / float(max(self.grid - 1, 1))
        fine_radius = (
            0.50 * cell_stride
            + coarse_std_high.mean(dim=-1)
            + 0.50 * uncertainty_grid[:, :, :, :, None]
        ).clamp(0.50 * cell_stride, 2.50 * cell_stride)
        offsets = self.fine_offset_lattice.to(
            device=center_high.device, dtype=center_high.dtype
        )
        fine_coordinates = (
            center_high[:, :, :, :, :, None]
            + fine_radius[:, :, :, :, :, None, None] * offsets
        )
        fine_valid = (
            (fine_coordinates[..., 0] >= 0.0)
            & (fine_coordinates[..., 0] <= float(raw_side - 1))
            & (fine_coordinates[..., 1] >= 0.0)
            & (fine_coordinates[..., 1] <= float(raw_side - 1))
        )

        target_raw_flat = target_raw.reshape(
            batch * cameras, self.raw_dim, raw_side, raw_side
        )
        source_raw_flat = source_raw.reshape(
            batch * cameras, self.raw_dim, raw_side, raw_side
        )
        raw_key_map = self.raw_key(target_raw_flat).reshape(
            batch, cameras, self.route_dim, raw_side, raw_side
        )
        source_raw_key_map = self.source_raw_key(source_raw_flat).reshape(
            batch, cameras, self.route_dim, raw_side, raw_side
        )
        dino_key_map = self.fine_dino_key(target_dino).permute(0, 1, 4, 2, 3)
        sampled_raw_key = self._sample_chart(raw_key_map, fine_coordinates)
        # Preserve source-side sub-cell appearance until the policy query.  A
        # weak flow is only a geometric centre: source/target raw evidence can
        # still correct the fine posterior instead of inheriting the flow
        # estimate as an unchallengeable address.
        source_fine_coordinates = (
            source_high[:, :, :, :, None, None]
            + 0.50
            * cell_stride
            * offsets.reshape(1, 1, 1, 1, 1, -1, 2)
        ).expand(batch, cameras, -1, -1, self.slots, -1, -1)
        source_fine_valid = (
            (source_fine_coordinates[..., 0] >= 0.0)
            & (source_fine_coordinates[..., 0] <= float(raw_side - 1))
            & (source_fine_coordinates[..., 1] >= 0.0)
            & (source_fine_coordinates[..., 1] <= float(raw_side - 1))
        )
        sampled_source_raw_key = self._sample_chart(
            source_raw_key_map,
            source_fine_coordinates,
        )
        raw_pair_key = self.raw_pair_key(
            torch.cat(
                (
                    sampled_source_raw_key,
                    sampled_raw_key,
                    sampled_source_raw_key * sampled_raw_key,
                    sampled_raw_key - sampled_source_raw_key,
                ),
                dim=-1,
            )
        )
        dino_coordinates = (
            fine_coordinates
            * float(max(dino_side - 1, 1))
            / float(max(raw_side - 1, 1))
        )
        sampled_dino_key = self._sample_chart(dino_key_map, dino_coordinates)
        metadata_map = torch.cat((confidence, uncertainty, occlusion), dim=2)
        sampled_metadata = self._sample_chart(metadata_map, fine_coordinates).float()
        normalized_coordinate = self._normalize_xy(fine_coordinates, raw_side)
        normalized_offset = offsets.reshape(
            1, 1, 1, 1, 1, -1, 2
        ).expand(batch, cameras, self.grid, self.grid, self.slots, -1, -1)
        normalized_std = (
            coarse_std_high[:, :, :, :, :, None]
            / float(max(raw_side - 1, 1))
        ).expand(-1, -1, -1, -1, -1, int(offsets.shape[0]), -1)
        normalized_flow_delta = (
            flow_delta
            * float(max(raw_side - 1, 1))
            / float(max(dino_side - 1, 1))
            / float(max(raw_side - 1, 1))
        )[:, :, :, :, :, None].expand_as(normalized_std)
        fine_geometry = torch.cat(
            (
                normalized_coordinate,
                normalized_offset,
                sampled_metadata[..., 0:1],
                sampled_metadata[..., 1:2]
                / float(max(raw_side - 1, 1)),
                sampled_metadata[..., 2:3],
                normalized_std,
                normalized_flow_delta,
                fine_valid[..., None].float(),
                source_fine_valid[..., None].float(),
            ),
            dim=-1,
        )
        if int(fine_geometry.shape[-1]) != self.FINE_GEOMETRY_DIM:
            raise RuntimeError("fine address geometry width changed unexpectedly")
        fine_keys = (
            sampled_raw_key
            + raw_pair_key
            + sampled_dino_key
            + coarse_keys[:, :, :, :, :, None]
            + self.fine_geometry(fine_geometry.to(dtype=sampled_raw_key.dtype))
        )

        low_frequency = F.adaptive_avg_pool2d(
            target_raw_flat.float(), (self.grid, self.grid)
        )
        low_frequency_high = F.interpolate(
            low_frequency,
            size=(raw_side, raw_side),
            mode="bilinear",
            align_corners=True,
        )
        target_detail = (
            target_raw_flat.float() - low_frequency_high
        ).to(dtype=target_raw.dtype).reshape(
            batch, cameras, self.raw_dim, raw_side, raw_side
        )
        fine_values = self._sample_chart(target_detail, fine_coordinates)
        fine_values = fine_values * fine_valid[..., None].to(dtype=fine_values.dtype)
        raw_pair_cosine = F.cosine_similarity(
            sampled_source_raw_key.float(),
            sampled_raw_key.float(),
            dim=-1,
        )
        raw_pair_valid = (fine_valid & source_fine_valid).float()
        raw_pair_cosine_mean = (
            raw_pair_cosine * raw_pair_valid
        ).sum() / raw_pair_valid.sum().clamp_min(1.0)

        coarse_entropy = -(
            coarse_probability.clamp_min(1e-8)
            * coarse_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(max(dino_side * dino_side, 2)))
        if self.slots > 1:
            flattened_centers = coarse_centers.reshape(
                -1, self.slots, 2
            ).float()
            pair_distance = torch.cdist(
                flattened_centers,
                flattened_centers,
            )
            off_diagonal = ~torch.eye(
                self.slots, device=pair_distance.device, dtype=torch.bool
            )[None]
            slot_distance = pair_distance[off_diagonal.expand_as(pair_distance)].mean()
            # Center separation alone can miss two broad posteriors that have
            # the same expectation but retain different hypotheses.  Measure
            # the full categorical posterior with a bounded Hellinger
            # distance without materializing an [M,M,P] difference tensor.
            flattened_probability = coarse_probability.reshape(
                -1,
                self.slots,
                dino_side * dino_side,
            ).float()
            root_probability = flattened_probability.clamp_min(0.0).sqrt()
            slot_affinity = torch.einsum(
                "nmp,nkp->nmk",
                root_probability,
                root_probability,
            )
            slot_posterior_distance = (
                1.0 - slot_affinity.clamp(0.0, 1.0)
            ).clamp_min(0.0).sqrt()
            slot_posterior_distance = slot_posterior_distance[
                off_diagonal.expand_as(slot_posterior_distance)
            ].mean()
        else:
            slot_distance = coarse_centers.new_zeros(())
            slot_posterior_distance = coarse_centers.new_zeros(())
        chart_diagonal = math.sqrt(2.0) * float(max(dino_side - 1, 1))
        bank = SoftAddressLatticeBank(
            coarse_keys=coarse_keys,
            fine_keys=fine_keys,
            fine_values=fine_values,
            fine_valid=fine_valid,
            coarse_centers=coarse_centers.to(dtype=source_dino.dtype),
            coarse_variance=coarse_variance.to(dtype=source_dino.dtype),
            fine_radius=fine_radius.to(dtype=source_dino.dtype),
            coarse_base_logits=(
                coarse_logits.to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_candidate_keys=(
                target_key if self.progressive_grounding_address else None
            ),
            coarse_candidate_coordinates=(
                self._normalize_xy(
                    target_coordinates.expand(batch, cameras, -1, -1),
                    dino_side,
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_flow_centers=(
                self._normalize_xy(
                    flow_center_dino,
                    dino_side,
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_confidence=(
                confidence_grid.to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_uncertainty=(
                (
                    uncertainty_dino / float(max(dino_side - 1, 1))
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_occlusion=(
                occlusion_grid.to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_cycle_error=(
                (
                    cycle_grid / float(max(raw_side - 1, 1))
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            fine_coordinates=(
                self._normalize_xy(
                    fine_coordinates,
                    raw_side,
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            coarse_source_centers=(
                self._normalize_xy(
                    source_high.expand(batch, cameras, -1, -1, -1),
                    raw_side,
                ).to(dtype=source_dino.dtype)
                if self.progressive_grounding_address
                else None
            ),
            dense_source_raw_keys=(
                source_raw_key_map if self.progressive_grounding_address else None
            ),
            dense_target_raw_keys=(
                raw_key_map if self.progressive_grounding_address else None
            ),
            dense_target_dino_keys=(
                dino_key_map if self.progressive_grounding_address else None
            ),
            dense_target_detail=(
                target_detail if self.progressive_grounding_address else None
            ),
            dense_confidence=(
                confidence if self.progressive_grounding_address else None
            ),
            dense_uncertainty=(
                uncertainty if self.progressive_grounding_address else None
            ),
            dense_occlusion=(
                occlusion if self.progressive_grounding_address else None
            ),
            dense_current_rgb=(
                literal_rgb_chart if self.coordinate_typed_raw_detail else None
            ),
        )
        metrics = {
            "flow_jepa_address_lattice_enabled": coarse_centers.new_ones(()),
            "flow_jepa_address_slot_count": coarse_centers.new_tensor(
                float(self.slots)
            ),
            "flow_jepa_address_fine_candidates_per_slot": coarse_centers.new_tensor(
                float(offsets.shape[0])
            ),
            "flow_jepa_address_coarse_entropy": coarse_entropy.mean().detach(),
            "flow_jepa_address_coarse_max": coarse_probability.max(
                dim=-1
            ).values.mean().detach(),
            "flow_jepa_address_coarse_variance": coarse_variance.float()
            .mean()
            .detach(),
            "flow_jepa_address_slot_pair_distance": slot_distance.detach(),
            "flow_jepa_address_slot_pair_distance_normalized": (
                slot_distance / chart_diagonal
            ).detach(),
            "flow_jepa_address_slot_posterior_hellinger": (
                slot_posterior_distance.detach()
            ),
            "flow_jepa_address_fine_radius": fine_radius.float().mean().detach(),
            "flow_jepa_address_highres_valid_fraction": fine_valid.float()
            .mean()
            .detach(),
            "flow_jepa_address_source_raw_match_active": coarse_centers.new_ones(
                ()
            ),
            "flow_jepa_address_source_raw_pair_cosine": (
                raw_pair_cosine_mean.detach()
            ),
            "flow_jepa_coordinate_typed_raw_detail": coarse_centers.new_tensor(
                float(self.coordinate_typed_raw_detail), dtype=torch.float32
            ),
            "flow_jepa_literal_rgb_chart_rms": (
                literal_rgb_chart.detach().float().square().mean().sqrt()
                if literal_rgb_chart is not None
                else coarse_centers.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_address_flow_prior_scale": prior_scale.detach(),
            "flow_jepa_address_flow_prior_floor": coarse_centers.new_tensor(
                self.flow_prior_floor
            ),
            "flow_jepa_address_flow_prior_floor_sigma": coarse_centers.new_tensor(
                floor_sigma
            ),
            "flow_jepa_address_flow_prior_floor_logit_span": (
                floor_flow_bias.amax(dim=-1) - floor_flow_bias.amin(dim=-1)
            ).mean().detach(),
            "flow_jepa_address_flow_prior_adaptive_logit_span": (
                adaptive_flow_bias.amax(dim=-1)
                - adaptive_flow_bias.amin(dim=-1)
            ).mean().detach(),
            "flow_jepa_address_flow_prior_learned": adaptive_prior_scale.detach(),
        }
        return bank, metrics


class _ProgressiveGroundingAddressOrganizer(nn.Module):
    """Give G1/G2/G3 distinct selector-state transitions without reading values.

    G1 scores the complete DINO chart, G2 rectifies the continuous candidates,
    and G3 compiles selector priors plus low-rank observation summaries.  The
    raw fine values remain untouched until the existing W->P policy reader.
    """

    GEOMETRY_WIDTH = 8

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden = int(config.hidden_size)
        self.route_dim = int(config.flow_jepa_address_route_dim)
        self.grid = int(config.flow_jepa_grid_size)
        self.cameras = int(config.num_cameras)
        self.anchors = int(config.future_anchors)
        self.slots = int(config.flow_jepa_address_slots)
        self.world_blocks = int(config.flow_jepa_world_blocks)
        self.coordinate_typed_raw_detail = bool(
            int(getattr(config, "flow_jepa_coordinate_typed_raw_detail", 0))
        )
        self.structured_ownership = bool(
            int(getattr(config, "flow_jepa_structured_ownership_bottleneck", 0))
        )
        self.pre_value_owner_routing = bool(
            int(getattr(config, "flow_jepa_pre_value_owner_routing", 0))
        )
        self.functional_mainline_routing = bool(
            int(getattr(config, "flow_jepa_functional_mainline_routing", 0))
        )
        self.g_aligned_future_effect = bool(
            int(getattr(config, "flow_jepa_g_aligned_future_effect", 0))
        )
        self.supervised_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_supervised_effect_mainline",
                    0,
                )
            )
        )
        self.window_effect_bank = bool(
            int(getattr(config, "flow_jepa_window_effect_bank", 0))
        )
        self.differential_intent_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_differential_intent_effect_mainline",
                    0,
                )
            )
        )
        self.grounded_intent_effect_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_grounded_intent_effect_mainline",
                    0,
                )
            )
        )
        self.object_intent_dynamics_mainline = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_object_intent_dynamics_mainline",
                    0,
                )
            )
        )
        self.exports_grounded_facts = bool(
            self.grounded_intent_effect_mainline
            or self.object_intent_dynamics_mainline
        )
        self.effect_slots = int(
            getattr(config, "flow_jepa_future_slots", self.anchors)
        )
        self.window_successor_cell: nn.Module | None = None
        self.window_late_cell: nn.Module | None = None
        self.differential_window_compiler: (
            DifferentialWindowRouteCompiler | None
        ) = None
        self.grounded_world_compiler: GroundedWorldEffectCompiler | None = None
        self.pre_value_owner_update_scale = float(
            getattr(config, "flow_jepa_pre_value_owner_update_scale", 0.10)
        )
        floor = float(getattr(config, "flow_jepa_routing_norm_floor", 0.25))
        variance_safe = bool(
            int(getattr(config, "flow_jepa_variance_safe_routing", 0))
        )

        def norm(width: int, *, affine: bool = False) -> nn.Module:
            if variance_safe:
                if affine:
                    return AffineVarianceFlooredCenteredNorm(
                        width, floor, affine_maximum=4.0
                    )
                return VarianceFlooredCenteredNorm(floor)
            return nn.LayerNorm(width, elementwise_affine=affine)

        self.query_norms = nn.ModuleList([norm(self.hidden) for _ in range(3)])
        self.query_projections = nn.ModuleList(
            [nn.Linear(self.hidden, self.route_dim, bias=False) for _ in range(3)]
        )
        self.candidate_norm = norm(self.route_dim)
        self.slot_identity = nn.Parameter(
            torch.randn(1, 1, 1, 1, self.slots, self.route_dim) * 0.02
        )
        self.g2_rectifier = nn.Sequential(
            norm(self.route_dim + self.GEOMETRY_WIDTH, affine=True),
            nn.Linear(self.route_dim + self.GEOMETRY_WIDTH, self.route_dim),
            nn.SiLU(),
            nn.Linear(self.route_dim, 4, bias=False),
        )
        self.g3_slot_score = nn.Sequential(
            norm(self.route_dim, affine=True),
            nn.Linear(self.route_dim, 1, bias=False),
        )
        self.g3_summary_out = nn.Sequential(
            norm(2 * self.route_dim + 4, affine=True),
            nn.Linear(2 * self.route_dim + 4, self.hidden, bias=False),
        )
        if self.coordinate_typed_raw_detail:
            self.g2_typed_rectifier = nn.Sequential(
                norm(2 * self.route_dim + self.GEOMETRY_WIDTH, affine=True),
                nn.Linear(
                    2 * self.route_dim + self.GEOMETRY_WIDTH,
                    self.route_dim,
                ),
                nn.SiLU(),
                # dx, dy, support log-scale, and geometric-prior strength.
                # All four channels are consumed in update(stage=2).
                nn.Linear(self.route_dim, 4, bias=False),
            )
            self.g2_typed_query = nn.ModuleDict(
                {
                    name: nn.Linear(self.route_dim, self.route_dim, bias=False)
                    for name in ("semantic", "appearance", "geometry")
                }
            )
            # Keep the G->W handoff typed.  A single projection over the
            # concatenated semantic/appearance/geometry state would preserve
            # the tensors in the side state while still presenting W with one
            # irreversible information soup.  Instead each type gets its own
            # token and projection; the W blocks decide how to combine them.
            self.g3_typed_summary_out = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        norm(self.route_dim + 4, affine=True),
                        nn.Linear(self.route_dim + 4, self.hidden, bias=False),
                    )
                    for name in ("semantic", "appearance", "geometry")
                }
            )
            self.g3_typed_slot_score = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        norm(self.route_dim, affine=True),
                        nn.Linear(self.route_dim, 1, bias=False),
                    )
                    for name in ("semantic", "appearance", "geometry")
                }
            )
            self.g3_owner_residual = (
                nn.ModuleDict(
                    {
                        name: nn.Sequential(
                            norm(self.route_dim, affine=True),
                            nn.Linear(self.route_dim, 1, bias=False),
                        )
                        for name in ("semantic", "appearance", "geometry")
                    }
                )
                if self.exports_grounded_facts
                else None
            )
            if self.g3_owner_residual is not None:
                for residual in self.g3_owner_residual.values():
                    nn.init.zeros_(residual[-1].weight)
            self.world_typed_query = nn.ModuleDict(
                {
                    name: nn.Linear(self.route_dim, self.route_dim, bias=False)
                    for name in ("semantic", "appearance", "geometry")
                }
            )
            transport_width = 4 * self.route_dim + 4
            self.future_transport = nn.Sequential(
                norm(transport_width, affine=True),
                nn.Linear(transport_width, 2 * self.route_dim, bias=False),
                nn.SiLU(),
                nn.Linear(2 * self.route_dim, 5, bias=False),
            )
            nn.init.zeros_(self.future_transport[-1].weight)
            if self.pre_value_owner_routing:
                # The public chart is projected from the clean G3 query plus
                # owner-neutral geometry.  Private semantic/appearance/
                # geometry summaries never enter this projection.
                self.g3_public_summary_out = nn.Sequential(
                    norm(self.route_dim + 4, affine=True),
                    nn.Linear(
                        self.route_dim + 4,
                        self.hidden,
                        bias=False,
                    ),
                )
                owner_names = (
                    "semantic",
                    "appearance",
                    "geometry",
                    "interval",
                )
                # One transition at the G3->W entry and one after each
                # configured W block. States stay in route space; only their
                # bounded deltas are reconstructed into the shared W carrier.
                self.world_owner_transitions = nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                name: nn.Sequential(
                                    norm(3 * self.route_dim, affine=True),
                                    nn.Linear(
                                        3 * self.route_dim,
                                        2 * self.route_dim,
                                        bias=False,
                                    ),
                                    nn.SiLU(),
                                    nn.Linear(
                                        2 * self.route_dim,
                                        self.route_dim,
                                        bias=False,
                                    ),
                                )
                                for name in owner_names
                            }
                        )
                        for _ in range(self.world_blocks + 1)
                    ]
                )
                self.world_owner_writes = nn.ModuleDict(
                    {
                        name: nn.Linear(
                            self.route_dim,
                            self.hidden,
                            bias=False,
                        )
                        for name in owner_names
                    }
                )
                for transition_bank in self.world_owner_transitions:
                    for transition in transition_bank.values():
                        nn.init.normal_(
                            transition[-1].weight,
                            mean=0.0,
                            std=1e-3,
                        )
                if self.functional_mainline_routing:
                    self.world_owner_route_attnres = nn.ModuleList(
                        [
                            RoleDeltaAttnRes(
                                self.route_dim,
                                self.route_dim,
                                max_sources=len(owner_names),
                                max_value_rms=0.75,
                                normalization_floor=floor,
                            )
                            for _ in range(self.world_blocks + 1)
                        ]
                    )
                    self.world_owner_fused_writes = nn.ModuleList(
                        [
                            nn.Linear(
                                self.route_dim,
                                self.hidden,
                                bias=False,
                            )
                            for _ in range(self.world_blocks + 1)
                        ]
                    )
                    self.world_horizon_condition = nn.ModuleList(
                        [
                            nn.Linear(
                                self.hidden,
                                self.route_dim,
                                bias=False,
                            )
                            for _ in range(self.world_blocks + 1)
                        ]
                    )
                    for write in self.world_owner_fused_writes:
                        nn.init.normal_(write.weight, mean=0.0, std=2e-2)
                    # The full-width projections are retained only for
                    # checkpoint ancestry. V113+ routes the compact owner bank
                    # and reconstructs hidden width once.
                    self.world_owner_writes.requires_grad_(False)
                    if self.g_aligned_future_effect:
                        effect_width = (
                            self.route_dim
                            if self.supervised_effect_mainline
                            else 4 * self.route_dim
                        )
                        self.future_effect_semantic = nn.Sequential(
                            norm(effect_width, affine=True),
                            nn.Linear(
                                effect_width,
                                2 * self.route_dim,
                                bias=False,
                            ),
                            nn.SiLU(),
                            nn.Linear(
                                2 * self.route_dim,
                                self.hidden,
                                bias=False,
                            ),
                        )
                        self.future_effect_geometry = nn.Sequential(
                            norm(effect_width, affine=True),
                            nn.Linear(
                                effect_width,
                                2 * self.route_dim,
                                bias=False,
                            ),
                            nn.SiLU(),
                            nn.Linear(
                                2 * self.route_dim,
                                8,
                                bias=False,
                            ),
                        )
                        nn.init.normal_(
                            self.future_effect_semantic[-1].weight,
                            mean=0.0,
                            std=1e-3,
                        )
                        nn.init.normal_(
                            self.future_effect_geometry[-1].weight,
                            mean=0.0,
                            std=1e-3,
                        )
                        self.future_effect_current = (
                            nn.Sequential(
                                norm(self.route_dim, affine=True),
                                nn.Linear(
                                    self.route_dim,
                                    self.hidden,
                                    bias=False,
                                ),
                            )
                            if self.supervised_effect_mainline
                            else None
                        )
                        if self.future_effect_current is not None:
                            nn.init.normal_(
                                self.future_effect_current[-1].weight,
                                mean=0.0,
                                std=1e-3,
                            )
                        if self.window_effect_bank:
                            if self.effect_slots != 3 or self.anchors != 4:
                                raise ValueError(
                                    "V117 window effects require 3 slots over 4 anchors"
                                )
                            self.window_successor_cell = nn.Sequential(
                                norm(2 * self.route_dim, affine=True),
                                nn.Linear(
                                    2 * self.route_dim,
                                    2 * self.route_dim,
                                    bias=False,
                                ),
                                nn.SiLU(),
                                nn.Linear(
                                    2 * self.route_dim,
                                    self.route_dim,
                                    bias=False,
                                ),
                            )
                            self.window_late_cell = nn.Sequential(
                                norm(2 * self.route_dim, affine=True),
                                nn.Linear(
                                    2 * self.route_dim,
                                    2 * self.route_dim,
                                    bias=False,
                                ),
                                nn.SiLU(),
                                nn.Linear(
                                    2 * self.route_dim,
                                    self.route_dim,
                                    bias=False,
                                ),
                            )
                            nn.init.normal_(
                                self.window_successor_cell[-1].weight,
                                mean=0.0,
                                std=1e-2,
                            )
                            nn.init.normal_(
                                self.window_late_cell[-1].weight,
                                mean=0.0,
                                std=1e-2,
                            )
                    else:
                        self.future_effect_semantic = None
                        self.future_effect_geometry = None
                        self.future_effect_current = None
                else:
                    self.world_owner_route_attnres = None
                    self.world_owner_fused_writes = None
                    self.world_horizon_condition = None
                    self.future_effect_semantic = None
                    self.future_effect_geometry = None
            else:
                self.g3_public_summary_out = None
                self.world_owner_transitions = None
                self.world_owner_writes = None
                self.world_owner_route_attnres = None
                self.world_owner_fused_writes = None
                self.world_horizon_condition = None
                self.future_effect_semantic = None
                self.future_effect_geometry = None
                self.future_effect_current = None
            # Retain the V109 modules in the state dict for ancestry, but do
            # not give the optimizer dead owners once the typed V110 modules
            # replace their forward semantics.
            self.g2_rectifier.requires_grad_(False)
            self.g3_slot_score.requires_grad_(False)
            self.g3_summary_out.requires_grad_(False)
            if self.pre_value_owner_routing:
                # V112's public projection is explicitly independent of the
                # V110/V111 private hidden summaries.  Retain those parameters
                # for checkpoint ancestry without presenting dead owners to
                # the optimizer.
                self.g3_typed_summary_out.requires_grad_(False)
        else:
            self.g2_typed_rectifier = None
            self.g2_typed_query = None
            self.g3_typed_summary_out = None
            self.g3_typed_slot_score = None
            self.g3_owner_residual = None
            self.world_typed_query = None
            self.future_transport = None
            self.g3_public_summary_out = None
            self.world_owner_transitions = None
            self.world_owner_writes = None
            self.world_owner_route_attnres = None
            self.world_owner_fused_writes = None
            self.world_horizon_condition = None
            self.future_effect_semantic = None
            self.future_effect_geometry = None
            self.future_effect_current = None
        if self.differential_intent_effect_mainline:
            if not (
                self.window_effect_bank
                and self.supervised_effect_mainline
                and self.functional_mainline_routing
                and self.effect_slots == 3
                and self.anchors == 4
            ):
                raise ValueError(
                    "differential W requires the complete V117 four-anchor "
                    "three-slot supervised routing parent"
                )
            self.differential_window_compiler = DifferentialWindowRouteCompiler(
                route_dim=self.route_dim,
                hidden=self.hidden,
                heads=4,
                slots_per_cell=self.slots,
            )
            # These V116/V117 decoders stay serializable for ancestry, but the
            # differential graph must not optimizer-own dead parallel effects.
            for legacy_module in (
                self.future_effect_semantic,
                self.future_effect_geometry,
                self.future_effect_current,
                self.window_successor_cell,
                self.window_late_cell,
            ):
                if legacy_module is not None:
                    legacy_module.requires_grad_(False)
        if self.grounded_intent_effect_mainline:
            if not (
                self.g_aligned_future_effect
                and self.supervised_effect_mainline
                and self.functional_mainline_routing
                and self.pre_value_owner_routing
                and self.effect_slots == 4
                and self.anchors == 4
                and self.coordinate_typed_raw_detail
                and self.structured_ownership
            ):
                raise ValueError(
                    "grounded W requires typed G3 facts, four anchors/effects, "
                    "and the observable supervised 3-2-3 parent"
                )
            self.grounded_world_compiler = GroundedWorldEffectCompiler(
                hidden=self.hidden,
                fact_dim=self.route_dim,
                route_dim=self.route_dim,
                content_dim=int(config.visual_token_dim),
                heads=4,
            )
            if self.g3_typed_slot_score is not None:
                self.g3_typed_slot_score.requires_grad_(False)
            # The grounded compiler replaces the historical shared owner
            # transition/AttnRes soup and every parallel effect decoder.
            for legacy_module in (
                self.world_owner_transitions,
                self.world_owner_writes,
                self.world_owner_route_attnres,
                self.world_owner_fused_writes,
                self.future_effect_semantic,
                self.future_effect_geometry,
                self.future_effect_current,
                self.window_successor_cell,
                self.window_late_cell,
            ):
                if legacy_module is not None:
                    legacy_module.requires_grad_(False)
        self.horizon_query_norm = norm(self.hidden)
        self.horizon_query_proj = nn.Linear(
            self.hidden, self.route_dim, bias=False
        )
        if self.grounded_intent_effect_mainline:
            # G3 ownership is inherited through bounded residuals, while W is
            # owned by GroundedWorldEffectCompiler. These serialized ancestor
            # modules may still be evaluated to retain the parent state shape,
            # but they are not optimization owners in the sibling graph.
            self.query_projections[2].requires_grad_(False)
            for ancestor in (
                self.g3_public_summary_out,
                self.world_horizon_condition,
                self.horizon_query_norm,
                self.horizon_query_proj,
            ):
                if ancestor is not None:
                    ancestor.requires_grad_(False)
        if self.g_aligned_future_effect:
            # These modules own V109-V114's quadratic W target/source
            # posterior. V115 addresses P1 from protected G3 facts and carries
            # successor geometry in FutureEffectField, so the legacy posterior
            # is neither executed nor optimizer-owned.
            if self.world_typed_query is not None:
                self.world_typed_query.requires_grad_(False)
            if self.future_transport is not None:
                self.future_transport.requires_grad_(False)
        if self.object_intent_dynamics_mainline:
            # The capability reuses G1/G2/G3's typed local-fact construction
            # and the V114 P1 lattice only.  Historical W owner transitions,
            # horizon queries and effect decoders are never called by the
            # object graph; leaving them optimizer-owned would create a large
            # dead parameter group and misleading zero-gradient diagnostics.
            for ancestor in (
                self.g3_public_summary_out,
                self.g3_typed_slot_score,
                self.world_owner_transitions,
                self.world_owner_writes,
                self.world_owner_route_attnres,
                self.world_owner_fused_writes,
                self.world_horizon_condition,
                self.future_effect_semantic,
                self.future_effect_geometry,
                self.future_effect_current,
                self.window_successor_cell,
                self.window_late_cell,
                self.world_typed_query,
                self.future_transport,
                self.horizon_query_norm,
                self.horizon_query_proj,
            ):
                if ancestor is not None:
                    ancestor.requires_grad_(False)

    def _decode_supervised_effect_routes(
        self,
        route_state: Tensor,
        state: ProgressiveGroundingAddressState,
        *,
        rollout_dtype: torch.dtype,
    ) -> tuple[FutureEffectField, Tensor]:
        """Decode only the W-owned slots presented by one causal stage."""

        if (
            self.future_effect_semantic is None
            or self.future_effect_geometry is None
            or self.future_effect_current is None
            or state.canonical_semantic_keys is None
        ):
            raise RuntimeError("supervised effect route decoder is incomplete")
        if route_state.ndim != 6 or int(route_state.shape[-1]) != self.route_dim:
            raise ValueError("effect route state must be [B,W,C,G,G,R]")
        slot_count = int(route_state.shape[1])
        route_slots = route_state[..., None, :].expand(
            -1, -1, -1, -1, -1, self.slots, -1
        )
        semantic_slots = state.canonical_semantic_keys[:, None].expand(
            -1, slot_count, -1, -1, -1, -1, -1
        )
        raw_semantic_delta = self.future_effect_semantic(route_slots)
        semantic_delta, semantic_contract = smooth_rms_contract(
            raw_semantic_delta, 0.50
        )
        raw_geometry = self.future_effect_geometry(route_slots)
        transport_mean = 0.50 * torch.tanh(raw_geometry[..., :2])
        variance_diag = 0.01 + 0.99 * torch.sigmoid(raw_geometry[..., 2:4])
        covariance_cross = (
            0.50
            * torch.tanh(raw_geometry[..., 4:5])
            * variance_diag.prod(dim=-1, keepdim=True).sqrt()
        )
        transport_covariance = torch.cat(
            (variance_diag, covariance_cross), dim=-1
        )
        persistence = torch.sigmoid(raw_geometry[..., 5:6])
        visibility = torch.sigmoid(raw_geometry[..., 6:7])
        uncertainty = 0.05 + 3.95 * torch.sigmoid(
            raw_geometry[..., 7:8] - 1.5
        )
        current_content = self.future_effect_current(semantic_slots)
        current_content, _ = smooth_rms_contract(current_content, 0.75)
        effect = FutureEffectField(
            semantic_delta=semantic_delta.to(dtype=rollout_dtype),
            transport_mean=transport_mean.to(dtype=rollout_dtype),
            transport_covariance=transport_covariance.to(dtype=rollout_dtype),
            persistence=persistence.to(dtype=rollout_dtype),
            visibility=visibility.to(dtype=rollout_dtype),
            uncertainty=uncertainty.to(dtype=rollout_dtype),
            current_content=current_content.to(dtype=rollout_dtype),
            successor_content=(current_content + semantic_delta).to(
                dtype=rollout_dtype
            ),
        )
        effect.validate()
        return effect, semantic_contract

    @staticmethod
    def _concat_effect_fields(
        first: FutureEffectField,
        second: FutureEffectField,
        *,
        slot_valid: Tensor,
    ) -> WindowEffectBank:
        """Concatenate independently owned effect slots without a hidden bypass."""

        first.validate()
        second.validate()
        if first.current_content is None or first.successor_content is None:
            raise ValueError("first effect field is not supervised")
        if second.current_content is None or second.successor_content is None:
            raise ValueError("second effect field is not supervised")

        def combine(name: str) -> Tensor:
            return torch.cat((getattr(first, name), getattr(second, name)), dim=1)

        bank = WindowEffectBank(
            semantic_delta=combine("semantic_delta"),
            transport_mean=combine("transport_mean"),
            transport_covariance=combine("transport_covariance"),
            persistence=combine("persistence"),
            visibility=combine("visibility"),
            uncertainty=combine("uncertainty"),
            current_content=combine("current_content"),
            successor_content=combine("successor_content"),
            slot_valid=slot_valid,
        )
        bank.validate()
        return bank

    @staticmethod
    def _slice_effect_field(
        field: FutureEffectField, window: slice
    ) -> FutureEffectField:
        field.validate()
        if field.current_content is None or field.successor_content is None:
            raise ValueError("cannot slice a legacy unsupervised effect field")
        sliced = FutureEffectField(
            semantic_delta=field.semantic_delta[:, window],
            transport_mean=field.transport_mean[:, window],
            transport_covariance=field.transport_covariance[:, window],
            persistence=field.persistence[:, window],
            visibility=field.visibility[:, window],
            uncertainty=field.uncertainty[:, window],
            current_content=field.current_content[:, window],
            successor_content=field.successor_content[:, window],
        )
        sliced.validate()
        return sliced

    def begin(self, bank: SoftAddressLatticeBank) -> ProgressiveGroundingAddressState:
        required = (
            bank.coarse_base_logits,
            bank.coarse_candidate_keys,
            bank.coarse_candidate_coordinates,
            bank.coarse_flow_centers,
            bank.coarse_confidence,
            bank.coarse_uncertainty,
            bank.coarse_occlusion,
            bank.coarse_cycle_error,
            bank.fine_coordinates,
            bank.coarse_source_centers,
            bank.dense_source_raw_keys,
            bank.dense_target_raw_keys,
            bank.dense_target_dino_keys,
            bank.dense_target_detail,
            bank.dense_confidence,
            bank.dense_uncertainty,
            bank.dense_occlusion,
        )
        if not all(torch.is_tensor(value) for value in required):
            raise RuntimeError(
                "progressive grounding address requires the complete observation scaffold"
            )
        return ProgressiveGroundingAddressState(bank=bank, metrics={})

    def _clean_query(self, rollout: Tensor) -> Tensor:
        expected = self.anchors * self.cameras * self.grid * self.grid
        if rollout.ndim != 3 or int(rollout.shape[1]) != expected:
            raise ValueError(
                "progressive grounding query lost anchor/camera/cell geometry"
            )
        return rollout.reshape(
            int(rollout.shape[0]),
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
            self.hidden,
        ).mean(dim=1)

    @staticmethod
    def _intervene(
        value: Tensor,
        mode: str | None,
        *,
        zero_name: str,
        shuffle_name: str,
    ) -> Tensor:
        if mode == zero_name:
            return torch.zeros_like(value)
        if mode != shuffle_name:
            return value
        if int(value.shape[0]) > 1:
            return value.roll(shifts=1, dims=0)
        if value.ndim >= 4:
            return value.roll(shifts=1, dims=2)
        return value.roll(shifts=1, dims=-1)

    def update(
        self,
        state: ProgressiveGroundingAddressState,
        rollout: Tensor,
        *,
        stage: int,
        intervention: str | None = None,
        candidate_sampler: Any | None = None,
        collect_diagnostics: bool = True,
    ) -> ProgressiveGroundingAddressState:
        if stage != state.stage + 1 or stage not in (1, 2, 3):
            raise RuntimeError(
                f"progressive address expected stage {state.stage + 1}, got {stage}"
            )
        bank = state.bank
        clean = self._clean_query(rollout)
        query = self.query_projections[stage - 1](
            self.query_norms[stage - 1](clean)
        )[:, :, :, :, None]
        query = query + self.slot_identity.to(
            device=query.device, dtype=query.dtype
        )
        metrics = dict(state.metrics or {})
        scale = float(self.route_dim) ** -0.5

        if stage == 1:
            assert bank.coarse_base_logits is not None
            assert bank.coarse_candidate_keys is not None
            assert bank.coarse_candidate_coordinates is not None
            candidate_key = self.candidate_norm(bank.coarse_candidate_keys)
            with torch.autocast(device_type=rollout.device.type, enabled=False):
                delta_logits = torch.einsum(
                    "bcijmr,bcpr->bcijmp",
                    query.float(),
                    candidate_key.float(),
                ) * scale
                delta_logits = self._intervene(
                    delta_logits,
                    intervention,
                    zero_name="address_g1_zero",
                    shuffle_name="address_g1_shuffle",
                )
                logits = bank.coarse_base_logits.float() + delta_logits
                probability = torch.softmax(logits, dim=-1)
                coordinates = bank.coarse_candidate_coordinates.float()
                centers = torch.einsum(
                    "bcijmp,bcpd->bcijmd", probability, coordinates
                )
                variance = torch.einsum(
                    "bcijmp,bcijmpd->bcijmd",
                    probability,
                    (
                        coordinates[:, :, None, None, None]
                        - centers[..., None, :]
                    ).square(),
                )
                aligned_keys = torch.einsum(
                    "bcijmp,bcpr->bcijmr",
                    probability,
                    bank.coarse_candidate_keys.float(),
                )
            if collect_diagnostics:
                entropy = -(
                    probability.detach().float().clamp_min(1e-8)
                    * probability.detach().float().clamp_min(1e-8).log()
                ).sum(dim=-1) / math.log(
                    float(max(int(probability.shape[-1]), 2))
                )
                metrics.update({
                    "flow_jepa_progressive_grounding_address": rollout.new_ones(
                        (), dtype=torch.float32
                    ),
                    "flow_jepa_progressive_g1_logit_update_rms": delta_logits.detach()
                    .square()
                    .mean()
                    .sqrt(),
                    "flow_jepa_progressive_g1_coarse_entropy": entropy.detach().mean(),
                    "flow_jepa_progressive_g1_coarse_max": probability.detach()
                    .amax(dim=-1)
                    .mean(),
                    "flow_jepa_progressive_g1_center_flow_delta": (
                        centers.detach().float()
                        - bank.coarse_flow_centers.float()[:, :, :, :, None]
                    )
                    .square()
                    .sum(dim=-1)
                    .sqrt()
                    .mean(),
                })
            state.stage = 1
            state.coarse_logits = logits.to(dtype=rollout.dtype)
            state.coarse_probability = probability.to(dtype=rollout.dtype)
            state.aligned_centers = centers.to(dtype=rollout.dtype)
            state.aligned_variance = variance.to(dtype=rollout.dtype)
            state.aligned_keys = aligned_keys.to(dtype=rollout.dtype)
            state.metrics = metrics
            return state

        if state.aligned_centers is None or state.aligned_variance is None:
            raise RuntimeError("G2/G3 address update has no G1 alignment state")
        if state.aligned_keys is None:
            raise RuntimeError("G2/G3 address update has no aligned selector keys")

        if stage == 2:
            assert bank.fine_coordinates is not None
            assert bank.coarse_flow_centers is not None
            assert bank.coarse_confidence is not None
            assert bank.coarse_uncertainty is not None
            assert bank.coarse_occlusion is not None
            assert bank.coarse_cycle_error is not None
            geometry = torch.cat(
                (
                    state.aligned_centers.float()
                    - bank.coarse_flow_centers.float()[:, :, :, :, None],
                    state.aligned_variance.float().clamp_min(0.0).sqrt(),
                    bank.coarse_confidence.float()[..., None, None].expand(
                        -1, -1, -1, -1, self.slots, -1
                    ),
                    bank.coarse_uncertainty.float()[..., None, None].expand(
                        -1, -1, -1, -1, self.slots, -1
                    ),
                    bank.coarse_occlusion.float()[..., None, None].expand(
                        -1, -1, -1, -1, self.slots, -1
                    ),
                    bank.coarse_cycle_error.float()[..., None, None].expand(
                        -1, -1, -1, -1, self.slots, -1
                    ),
                ),
                dim=-1,
            )
            if self.coordinate_typed_raw_detail:
                if self.g2_typed_rectifier is None:
                    raise RuntimeError("typed G2 rectifier was not constructed")
                rectifier_input = torch.cat(
                    (
                        query,
                        state.aligned_keys.to(dtype=query.dtype),
                        geometry.to(dtype=query.dtype),
                    ),
                    dim=-1,
                )
                rectifier = self.g2_typed_rectifier(rectifier_input).float()
            else:
                rectifier = self.g2_rectifier(
                    torch.cat(
                        (
                            query + state.aligned_keys.to(dtype=query.dtype),
                            geometry.to(dtype=query.dtype),
                        ),
                        dim=-1,
                    )
                ).float()
            correction_proposal = 0.25 * torch.tanh(rectifier[..., :2]) * (
                state.aligned_variance.float().clamp_min(1e-4).sqrt()
            )
            original_fine_coordinates = bank.fine_coordinates.float()
            base_center = original_fine_coordinates.mean(dim=-2)
            base_support = (
                (original_fine_coordinates - base_center[..., None, :])
                .square()
                .sum(dim=-1)
                .mean(dim=-1)
                .sqrt()
                .clamp_min(0.02)
            )
            support = base_support * torch.exp(0.5 * torch.tanh(rectifier[..., 2]))
            if intervention == "address_g2_zero":
                correction_proposal = torch.zeros_like(correction_proposal)
                support = base_support
            elif intervention == "address_g2_shuffle":
                correction_proposal = self._intervene(
                    correction_proposal,
                    intervention,
                    zero_name="address_g2_zero",
                    shuffle_name="address_g2_shuffle",
                )
                support = self._intervene(
                    support,
                    intervention,
                    zero_name="address_g2_zero",
                    shuffle_name="address_g2_shuffle",
                )
            if self.coordinate_typed_raw_detail:
                # If x is in [-1,1] and |p| <= .25 then
                # x + (1-x^2)p also lies in [-1,1].  This source-relative map
                # has no division by a vanishing edge margin, unlike V109's
                # limiter, so its reverse gain stays finite at image borders.
                edge_margin = (
                    1.0 - state.aligned_centers.float().square()
                ).clamp_min(0.0)
                correction = edge_margin * correction_proposal
            else:
                positive_limit = (
                    1.0 - state.aligned_centers.float()
                ).clamp_min(0.0)
                negative_limit = (
                    1.0 + state.aligned_centers.float()
                ).clamp_min(0.0)
                correction_limit = torch.where(
                    correction_proposal >= 0.0,
                    positive_limit,
                    negative_limit,
                )
                correction = correction_limit * torch.tanh(
                    correction_proposal / correction_limit.clamp_min(1e-6)
                )
            correction_compression = (
                correction_proposal - correction
            ).abs().mean()
            rectified_centers = state.aligned_centers.float() + correction
            prior_strength = 0.25 + 1.75 * torch.sigmoid(rectifier[..., 3])
            if candidate_sampler is None:
                raise RuntimeError(
                    "G2 progressive address has no dynamic candidate sampler"
                )
            candidates = candidate_sampler(
                bank,
                centers=rectified_centers,
                support=support,
                variance=state.aligned_variance.float(),
                aligned_keys=state.aligned_keys,
                collect_diagnostics=collect_diagnostics,
            )
            if not isinstance(candidates, ProgressiveFineCandidates):
                raise TypeError("G2 candidate sampler returned an invalid contract")
            dynamic_fine_keys = candidates.combined_keys
            dynamic_fine_values = candidates.learned_detail
            dynamic_fine_valid = candidates.valid
            fine_coordinates = candidates.current_coordinates
            candidate_metrics = candidates.metrics
            fine_key = (
                None
                if self.coordinate_typed_raw_detail
                else self.candidate_norm(dynamic_fine_keys)
            )
            typed_query_rows: dict[str, Tensor] = {}
            typed_key_rows: dict[str, Tensor] = {}
            g2_slot_probability: dict[str, Tensor] = {}
            if self.coordinate_typed_raw_detail:
                if (
                    self.g2_typed_query is None
                    or candidates.semantic_keys is None
                    or candidates.appearance_keys is None
                    or candidates.geometry_keys is None
                ):
                    raise RuntimeError("typed G2 candidates are incomplete")
                # Learned modules must stay in the surrounding autocast domain.
                # Only their projected activations enter the explicit FP32
                # similarity island below.  Calling a Float32 Linear from that
                # island with a BF16 query bypasses autocast and is invalid.
                for name, candidate_key in (
                    ("semantic", candidates.semantic_keys),
                    ("appearance", candidates.appearance_keys),
                    ("geometry", candidates.geometry_keys),
                ):
                    typed_query_rows[name] = self.g2_typed_query[name](query)
                    typed_key_rows[name] = self.candidate_norm(candidate_key)
            with torch.autocast(device_type=rollout.device.type, enabled=False):
                if self.coordinate_typed_raw_detail:
                    typed_logits: dict[str, Tensor] = {}
                    for name, typed_key in typed_key_rows.items():
                        typed_logits[name] = torch.einsum(
                            "bcijmr,bcijmkr->bcijmk",
                            typed_query_rows[name].float(),
                            typed_key.float(),
                        ) * scale
                    if self.structured_ownership:
                        # G1 already owns semantic hypothesis alignment.  G2
                        # therefore lets appearance verify local content and
                        # geometry rectify local coordinates.  Semantic logits
                        # remain available as their own sidecar posterior, but
                        # cannot silently take ownership of fine localization.
                        content_logits = (
                            typed_logits["appearance"]
                            + typed_logits["geometry"]
                        ) / math.sqrt(2.0)
                    else:
                        # V110 compatibility ensemble.
                        content_logits = sum(typed_logits.values()) / math.sqrt(3.0)
                else:
                    typed_logits = {}
                    assert fine_key is not None
                    content_logits = torch.einsum(
                        "bcijmr,bcijmkr->bcijmk",
                        query.float(),
                        fine_key.float(),
                    ) * scale
                distance = (
                    fine_coordinates - rectified_centers[..., None, :]
                ).square().sum(dim=-1)
                spatial_prior = -0.5 * prior_strength[..., None] * (
                    distance / support[..., None].square().clamp_min(1e-4)
                )
                fine_logits = content_logits + spatial_prior
                fine_logits = fine_logits.masked_fill(
                    ~dynamic_fine_valid,
                    torch.finfo(fine_logits.dtype).min,
                )
                any_valid = dynamic_fine_valid.any(dim=-1, keepdim=True)
                safe_logits = torch.where(any_valid, fine_logits, torch.zeros_like(fine_logits))
                fine_probability = torch.softmax(safe_logits, dim=-1)
                fine_probability = fine_probability * dynamic_fine_valid.float()
                fine_probability = fine_probability / fine_probability.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0)
                if self.structured_ownership:
                    def owner_probability(owner_logits: Tensor) -> Tensor:
                        owner_logits = (owner_logits + spatial_prior).masked_fill(
                            ~dynamic_fine_valid,
                            torch.finfo(owner_logits.dtype).min,
                        )
                        safe_owner_logits = torch.where(
                            any_valid,
                            owner_logits,
                            torch.zeros_like(owner_logits),
                        )
                        probability = torch.softmax(safe_owner_logits, dim=-1)
                        probability = probability * dynamic_fine_valid.float()
                        return probability / probability.sum(
                            dim=-1, keepdim=True
                        ).clamp_min(1.0)

                    semantic_probability = owner_probability(
                        typed_logits["semantic"]
                    )
                    appearance_probability = owner_probability(
                        typed_logits["appearance"]
                    )
                    geometry_probability = owner_probability(
                        typed_logits["geometry"]
                    )
                    if self.exports_grounded_facts:
                        valid_count = dynamic_fine_valid.float().sum(
                            dim=-1
                        ).clamp_min(1.0)
                        for name in ("semantic", "appearance", "geometry"):
                            owner_candidate_logit = (
                                typed_logits[name] + spatial_prior
                            ).masked_fill(
                                ~dynamic_fine_valid,
                                torch.finfo(typed_logits[name].dtype).min,
                            )
                            safe_candidate_logit = torch.where(
                                any_valid,
                                owner_candidate_logit,
                                torch.zeros_like(owner_candidate_logit),
                            )
                            slot_evidence = torch.logsumexp(
                                safe_candidate_logit,
                                dim=-1,
                            ) - valid_count.log()
                            slot_evidence = torch.where(
                                any_valid[..., 0],
                                slot_evidence,
                                torch.zeros_like(slot_evidence),
                            )
                            g2_slot_probability[name] = torch.softmax(
                                slot_evidence,
                                dim=-1,
                            )
                else:
                    semantic_probability = fine_probability
                    appearance_probability = fine_probability
                    geometry_probability = fine_probability
                if intervention == "address_g2_zero":
                    fine_probability = dynamic_fine_valid.float()
                    fine_probability = fine_probability / fine_probability.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                    semantic_probability = fine_probability
                    appearance_probability = fine_probability
                    geometry_probability = fine_probability
                    if g2_slot_probability:
                        g2_slot_probability = {
                            name: torch.full_like(
                                probability,
                                1.0 / float(max(self.slots, 1)),
                            )
                            for name, probability in g2_slot_probability.items()
                        }
                elif intervention == "address_g2_shuffle":
                    fine_probability = self._intervene(
                        fine_probability,
                        intervention,
                        zero_name="address_g2_zero",
                        shuffle_name="address_g2_shuffle",
                    )
                    fine_probability = (
                        fine_probability * dynamic_fine_valid.float()
                    )
                    fine_probability = fine_probability / fine_probability.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1.0)
                    if self.structured_ownership:
                        semantic_probability = self._intervene(
                            semantic_probability,
                            intervention,
                            zero_name="address_g2_zero",
                            shuffle_name="address_g2_shuffle",
                        )
                        geometry_probability = self._intervene(
                            geometry_probability,
                            intervention,
                            zero_name="address_g2_zero",
                            shuffle_name="address_g2_shuffle",
                        )
                        semantic_probability = (
                            semantic_probability * dynamic_fine_valid.float()
                        )
                        semantic_probability = semantic_probability / (
                            semantic_probability.sum(dim=-1, keepdim=True).clamp_min(1.0)
                        )
                        geometry_probability = (
                            geometry_probability * dynamic_fine_valid.float()
                        )
                        geometry_probability = geometry_probability / (
                            geometry_probability.sum(dim=-1, keepdim=True).clamp_min(1.0)
                        )
                        appearance_probability = self._intervene(
                            appearance_probability,
                            intervention,
                            zero_name="address_g2_zero",
                            shuffle_name="address_g2_shuffle",
                        )
                        appearance_probability = (
                            appearance_probability * dynamic_fine_valid.float()
                        )
                        appearance_probability = appearance_probability / (
                            appearance_probability.sum(
                                dim=-1, keepdim=True
                            ).clamp_min(1.0)
                        )
                        if g2_slot_probability:
                            g2_slot_probability = {
                                name: probability.roll(shifts=1, dims=-1)
                                for name, probability in g2_slot_probability.items()
                            }
                rectified_keys = torch.einsum(
                    "bcijmk,bcijmkr->bcijmr",
                    fine_probability,
                    dynamic_fine_keys.float(),
                )
                center_from_fine = torch.einsum(
                    "bcijmk,bcijmkd->bcijmd",
                    geometry_probability,
                    fine_coordinates,
                )
            if collect_diagnostics:
                fine_entropy = -(
                    fine_probability.detach().float().clamp_min(1e-8)
                    * fine_probability.detach().float().clamp_min(1e-8).log()
                ).sum(dim=-1) / math.log(
                    float(max(int(fine_probability.shape[-1]), 2))
                )
                metrics.update({
                    **candidate_metrics,
                    "flow_jepa_progressive_g2_center_correction_rms": correction.detach()
                    .square()
                    .mean()
                    .sqrt(),
                    "flow_jepa_progressive_g2_correction_compression": (
                        correction_compression.detach()
                    ),
                    "flow_jepa_progressive_g2_fine_entropy": fine_entropy.detach().mean(),
                    "flow_jepa_progressive_g2_fine_max": fine_probability.detach()
                    .amax(dim=-1)
                    .mean(),
                    "flow_jepa_progressive_g2_support": support.detach().mean(),
                    "flow_jepa_progressive_g2_geometry_prior_strength": (
                        prior_strength.detach().mean()
                    ),
                    "flow_jepa_progressive_g2_center_shift": (
                        center_from_fine.detach() - state.aligned_centers.float()
                    )
                    .square()
                    .sum(dim=-1)
                    .sqrt()
                    .mean(),
                    "flow_jepa_progressive_g2_semantic_logit_rms": (
                        typed_logits["semantic"].detach().square().mean().sqrt()
                        if typed_logits
                        else rollout.new_zeros((), dtype=torch.float32)
                    ),
                    "flow_jepa_progressive_g2_appearance_logit_rms": (
                        typed_logits["appearance"].detach().square().mean().sqrt()
                        if typed_logits
                        else rollout.new_zeros((), dtype=torch.float32)
                    ),
                    "flow_jepa_progressive_g2_geometry_logit_rms": (
                        typed_logits["geometry"].detach().square().mean().sqrt()
                        if typed_logits
                        else rollout.new_zeros((), dtype=torch.float32)
                    ),
                    "flow_jepa_progressive_g2_semantic_appearance_posterior_l1": (
                        0.5
                        * (
                            semantic_probability.detach()
                            - appearance_probability.detach()
                        ).abs().sum(dim=-1).mean()
                    ),
                    "flow_jepa_progressive_g2_appearance_geometry_posterior_l1": (
                        0.5
                        * (
                            appearance_probability.detach()
                            - geometry_probability.detach()
                        ).abs().sum(dim=-1).mean()
                    ),
                })
            state.stage = 2
            state.fine_logits = fine_logits.to(dtype=rollout.dtype)
            state.fine_probability = fine_probability.to(dtype=rollout.dtype)
            state.rectified_centers = center_from_fine.to(dtype=rollout.dtype)
            state.rectified_support = support.to(dtype=rollout.dtype)
            state.rectified_keys = rectified_keys.to(dtype=rollout.dtype)
            state.dynamic_fine_keys = dynamic_fine_keys
            state.dynamic_fine_values = dynamic_fine_values
            state.dynamic_fine_valid = dynamic_fine_valid
            state.dynamic_fine_coordinates = fine_coordinates
            state.dynamic_source_coordinates = candidates.source_coordinates
            state.dynamic_semantic_keys = candidates.semantic_keys
            state.dynamic_appearance_keys = candidates.appearance_keys
            state.dynamic_geometry_keys = candidates.geometry_keys
            state.dynamic_literal_rgb = candidates.literal_rgb
            if self.structured_ownership:
                state.g2_semantic_probability = semantic_probability.to(
                    dtype=rollout.dtype
                )
                state.g2_appearance_probability = appearance_probability.to(
                    dtype=rollout.dtype
                )
                state.g2_geometry_probability = geometry_probability.to(
                    dtype=rollout.dtype
                )
                if self.exports_grounded_facts:
                    if set(g2_slot_probability) != {
                        "semantic",
                        "appearance",
                        "geometry",
                    }:
                        raise RuntimeError(
                            "grounded G2 did not construct typed slot ownership"
                        )
                    state.g2_semantic_slot_probability = (
                        g2_slot_probability["semantic"].to(dtype=rollout.dtype)
                    )
                    state.g2_appearance_slot_probability = (
                        g2_slot_probability["appearance"].to(dtype=rollout.dtype)
                    )
                    state.g2_geometry_slot_probability = (
                        g2_slot_probability["geometry"].to(dtype=rollout.dtype)
                    )
            state.metrics = metrics
            return state

        if state.coarse_logits is None or state.fine_probability is None:
            raise RuntimeError("G3 canonicalization has no G1/G2 posterior")
        if state.rectified_keys is None or state.rectified_centers is None:
            raise RuntimeError("G3 canonicalization has no rectified selector state")
        if state.rectified_support is None:
            raise RuntimeError("G3 canonicalization has no rectified support")
        canonical_semantic: Tensor | None = None
        canonical_appearance: Tensor | None = None
        canonical_geometry: Tensor | None = None
        owner_slot_scores: dict[str, Tensor] = {}
        if self.coordinate_typed_raw_detail:
            if (
                state.dynamic_semantic_keys is None
                or state.dynamic_appearance_keys is None
                or state.dynamic_geometry_keys is None
                or self.g3_typed_summary_out is None
                or self.g3_typed_slot_score is None
            ):
                raise RuntimeError("G3 has no typed G2 candidate state")
            if self.structured_ownership:
                if (
                    state.g2_semantic_probability is None
                    or state.g2_appearance_probability is None
                    or state.g2_geometry_probability is None
                ):
                    raise RuntimeError("G3 has no functional G2 owner posteriors")
                semantic_probability = state.g2_semantic_probability.float()
                appearance_probability = state.g2_appearance_probability.float()
                geometry_probability = state.g2_geometry_probability.float()
            else:
                semantic_probability = state.fine_probability.float()
                appearance_probability = state.fine_probability.float()
                geometry_probability = state.fine_probability.float()
            # Each dynamic semantic candidate already contains the G1
            # complete-chart aligned key plus its G2 local DINO key.  Adding
            # aligned_keys again here would count the coarse semantic owner
            # twice while appearance and geometry are counted once.
            canonical_semantic = torch.einsum(
                "bcijmk,bcijmkr->bcijmr",
                semantic_probability,
                state.dynamic_semantic_keys.float(),
            ).to(dtype=query.dtype)
            canonical_appearance = torch.einsum(
                "bcijmk,bcijmkr->bcijmr",
                appearance_probability,
                state.dynamic_appearance_keys.float(),
            ).to(dtype=query.dtype)
            canonical_geometry = torch.einsum(
                "bcijmk,bcijmkr->bcijmr",
                geometry_probability,
                state.dynamic_geometry_keys.float(),
            ).to(dtype=query.dtype)
            # Compatibility key only.  The three source keys remain stored
            # independently and are scored independently by W/P.
            slot_keys = (
                query
                + canonical_semantic
                + canonical_appearance
                + canonical_geometry
            ) * 0.5
            owner_slot_scores = (
                {}
                if self.exports_grounded_facts
                else {
                    "semantic": self.g3_typed_slot_score["semantic"](
                        canonical_semantic
                    ).squeeze(-1).float(),
                    "appearance": self.g3_typed_slot_score["appearance"](
                        canonical_appearance
                    ).squeeze(-1).float(),
                    "geometry": self.g3_typed_slot_score["geometry"](
                        canonical_geometry
                    ).squeeze(-1).float(),
                }
            )
            if self.exports_grounded_facts:
                if self.g3_owner_residual is None:
                    raise RuntimeError("grounded G3 has no bounded owner residual")
                parent_owner = {
                    "semantic": state.g2_semantic_slot_probability,
                    "appearance": state.g2_appearance_slot_probability,
                    "geometry": state.g2_geometry_slot_probability,
                }
                if any(value is None for value in parent_owner.values()):
                    raise RuntimeError(
                        "grounded G3 cannot inherit missing G2 slot ownership"
                    )
                updated_owner: dict[str, Tensor] = {}
                for name, canonical_key in (
                    ("semantic", canonical_semantic),
                    ("appearance", canonical_appearance),
                    ("geometry", canonical_geometry),
                ):
                    parent = parent_owner[name]
                    assert parent is not None
                    residual = self.g3_owner_residual[name](
                        canonical_key
                    ).squeeze(-1)
                    updated_owner[name] = bounded_owner_update(
                        parent,
                        residual,
                        maximum_residual=0.50,
                    ).float()
                # Store log probabilities as the score interface so the
                # downstream softmax recovers the inherited/refined posterior
                # exactly, rather than training a second independent owner.
                owner_slot_scores = {
                    name: probability.clamp_min(1e-8).log()
                    for name, probability in updated_owner.items()
                }
                slot_score = 0.5 * (
                    owner_slot_scores["semantic"]
                    + owner_slot_scores["geometry"]
                )
            elif self.structured_ownership:
                # Source/slot ownership is a semantic hypothesis constrained
                # by geometry.  Appearance is retained for local verification
                # and cannot seize the coarse source decision.
                slot_score = (
                    owner_slot_scores["semantic"]
                    + owner_slot_scores["geometry"]
                ) / math.sqrt(2.0)
            else:
                slot_score = sum(owner_slot_scores.values()) / math.sqrt(3.0)
        else:
            slot_keys = (
                query
                + state.aligned_keys.to(dtype=query.dtype)
                + state.rectified_keys.to(dtype=query.dtype)
            )
            slot_score = self.g3_slot_score(slot_keys).squeeze(-1).float()
        if intervention in {"address_g3_zero", "address_g3_shuffle"}:
            slot_keys = self._intervene(
                slot_keys,
                intervention,
                zero_name="address_g3_zero",
                shuffle_name="address_g3_shuffle",
            )
            if canonical_semantic is not None:
                canonical_semantic = self._intervene(
                    canonical_semantic,
                    intervention,
                    zero_name="address_g3_zero",
                    shuffle_name="address_g3_shuffle",
                )
                canonical_appearance = self._intervene(
                    canonical_appearance,
                    intervention,
                    zero_name="address_g3_zero",
                    shuffle_name="address_g3_shuffle",
                )
                canonical_geometry = self._intervene(
                    canonical_geometry,
                    intervention,
                    zero_name="address_g3_zero",
                    shuffle_name="address_g3_shuffle",
                )
                slot_score = self._intervene(
                    slot_score,
                    intervention,
                    zero_name="address_g3_zero",
                    shuffle_name="address_g3_shuffle",
                )
                if self.structured_ownership:
                    owner_slot_scores = {
                        name: self._intervene(
                            value,
                            intervention,
                            zero_name="address_g3_zero",
                            shuffle_name="address_g3_shuffle",
                        )
                        for name, value in owner_slot_scores.items()
                    }
        coarse_evidence = torch.logsumexp(state.coarse_logits.float(), dim=-1)
        coarse_evidence = coarse_evidence - math.log(
            float(max(int(state.coarse_logits.shape[-1]), 1))
        )
        evidence_mean = coarse_evidence.mean(dim=(2, 3, 4), keepdim=True)
        evidence_std = coarse_evidence.std(
            dim=(2, 3, 4), keepdim=True, unbiased=False
        ).clamp_min(0.25)
        coarse_bias = (coarse_evidence - evidence_mean) / evidence_std + slot_score
        coarse_bias = coarse_bias - coarse_bias.mean(
            dim=(1, 2, 3, 4), keepdim=True
        )
        coarse_bias, coarse_bias_scale = smooth_rms_contract(
            coarse_bias, 1.0
        )
        fine_bias = (
            state.fine_probability.float().clamp_min(1e-8).log()
            + math.log(float(max(int(state.fine_probability.shape[-1]), 1)))
        )
        if state.dynamic_fine_valid is None:
            raise RuntimeError("G3 canonicalization has no dynamic fine validity")
        fine_valid = state.dynamic_fine_valid
        valid_float = fine_valid.float()
        fine_bias_mean = (fine_bias * valid_float).sum(
            dim=-1, keepdim=True
        ) / valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        fine_bias = fine_bias - fine_bias_mean
        fine_bias = torch.where(
            fine_valid,
            fine_bias,
            torch.zeros_like(fine_bias),
        )
        fine_bias, fine_bias_scale = smooth_rms_contract(fine_bias, 1.0)
        fine_entropy = -(
            state.fine_probability.float().clamp_min(1e-8)
            * state.fine_probability.float().clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(
            float(max(int(state.fine_probability.shape[-1]), 2))
        )
        if self.coordinate_typed_raw_detail:
            assert canonical_semantic is not None
            assert canonical_appearance is not None
            assert canonical_geometry is not None
            assert self.g3_typed_summary_out is not None
            shared_geometry = (
                state.rectified_centers.float(),
                state.rectified_support.float()[..., None],
                fine_entropy[..., None],
            )
            summary_by_type = (
                {}
                if self.pre_value_owner_routing and self.structured_ownership
                else {
                    name: self.g3_typed_summary_out[name](
                        torch.cat((value.float(), *shared_geometry), dim=-1).to(
                            dtype=rollout.dtype
                        )
                    )
                    for name, value in (
                        ("semantic", canonical_semantic),
                        ("appearance", canonical_appearance),
                        ("geometry", canonical_geometry),
                    )
                }
            )
        else:
            summary_input = torch.cat(
                (
                    state.aligned_keys.float(),
                    state.rectified_keys.float(),
                    state.rectified_centers.float(),
                    state.rectified_support.float()[..., None],
                    fine_entropy[..., None],
                ),
                dim=-1,
            ).to(dtype=rollout.dtype)
            summary_by_slot = self.g3_summary_out(summary_input)
        slot_weights = torch.softmax(slot_score, dim=-1).to(dtype=rollout.dtype)
        owner_slot_weights: dict[str, Tensor] = {}
        if self.coordinate_typed_raw_detail:
            # V110 appended all three private summaries to generic visual
            # memory. V111 instead keeps the lower-width canonical typed keys
            # as the private sidecar actually consumed by W/P, and exposes only
            # one bounded camera-spatial public chart to the generic W memory.
            if self.structured_ownership:
                for name, canonical_key in (
                    ("semantic", canonical_semantic),
                    ("appearance", canonical_appearance),
                    ("geometry", canonical_geometry),
                ):
                    owner_slot_weights[name] = torch.softmax(
                        owner_slot_scores[name], dim=-1
                    ).to(dtype=rollout.dtype)
                    if collect_diagnostics:
                        metrics[
                            f"flow_jepa_progressive_g3_{name}_owner_sidecar_rms"
                        ] = canonical_key.detach().float().square().mean().sqrt()
                if self.pre_value_owner_routing:
                    if self.exports_grounded_facts:
                        # The grounded public base is the completed G3 chart
                        # itself.  Projecting the typed slot summaries and then
                        # averaging their object axis would recreate the public
                        # information soup that GroundedFactSet is designed to
                        # avoid.  ``clean`` retains camera/x/y identity and is
                        # already in the model hidden width.
                        summary, _ = smooth_rms_contract(clean, 0.35)
                        if collect_diagnostics:
                            metrics[
                                "grounded_g3_public_base_direct"
                            ] = summary.new_ones((), dtype=torch.float32)
                    else:
                        if self.g3_public_summary_out is None:
                            raise RuntimeError(
                                "pre-value owner routing has no public G3 projector"
                            )
                        # V112-V118 ancestry: retain the historical public
                        # projection exactly outside the grounded capability.
                        public_input = torch.cat(
                            (
                                query.float(),
                                state.rectified_centers.float(),
                                state.rectified_support.float()[..., None],
                                fine_entropy[..., None],
                            ),
                            dim=-1,
                        ).to(dtype=rollout.dtype)
                        summary = self.g3_public_summary_out(public_input).mean(dim=4)
                        if collect_diagnostics:
                            metrics[
                                "flow_jepa_progressive_g3_public_input_rms"
                            ] = public_input.detach().float().square().mean().sqrt()
                            private_reference = sum(
                                value.mean(dim=4)
                                for value in (
                                    canonical_semantic,
                                    canonical_appearance,
                                    canonical_geometry,
                                )
                            ) / math.sqrt(3.0)
                            metrics[
                                "flow_jepa_progressive_g3_query_private_cosine"
                            ] = F.cosine_similarity(
                                query.detach().float().mean(dim=4),
                                private_reference.detach().float(),
                                dim=-1,
                            ).mean()
                else:
                    # V111 compatibility path.  It is intentionally retained
                    # byte-for-byte when the post-V111 contract is disabled.
                    public_rows = [
                        summary_by_type[name].mean(dim=4)
                        for name in ("semantic", "appearance", "geometry")
                    ]
                    summary = sum(public_rows) / math.sqrt(3.0)
                summary, _ = smooth_rms_contract(summary, 0.35)
            else:
                typed_summary_rows = []
                for name in ("semantic", "appearance", "geometry"):
                    typed_summary = (
                        slot_weights[..., None] * summary_by_type[name]
                    ).sum(dim=4)
                    typed_summary, _ = smooth_rms_contract(typed_summary, 0.50)
                    typed_summary_rows.append(typed_summary)
                    if collect_diagnostics:
                        metrics[
                            f"flow_jepa_progressive_g3_{name}_summary_rms"
                        ] = typed_summary.detach().float().square().mean().sqrt()
                owner_summary = torch.stack(typed_summary_rows, dim=4)
                summary = owner_summary
        else:
            summary = (slot_weights[..., None] * summary_by_slot).sum(dim=4)
            summary, _ = smooth_rms_contract(summary, 0.50)
        if intervention in {"address_g3_zero", "address_g3_shuffle"}:
            coarse_bias = self._intervene(
                coarse_bias,
                intervention,
                zero_name="address_g3_zero",
                shuffle_name="address_g3_shuffle",
            )
            fine_bias = self._intervene(
                fine_bias,
                intervention,
                zero_name="address_g3_zero",
                shuffle_name="address_g3_shuffle",
            )
            summary = self._intervene(
                summary,
                intervention,
                zero_name="address_g3_zero",
                shuffle_name="address_g3_shuffle",
            )
        if collect_diagnostics:
            metrics.update({
                "flow_jepa_progressive_g3_coarse_bias_rms": coarse_bias.detach()
                .square()
                .mean()
                .sqrt(),
                "flow_jepa_progressive_g3_fine_bias_rms": fine_bias.detach()
                .square()
                .mean()
                .sqrt(),
                "flow_jepa_progressive_g3_summary_rms": summary.detach()
                .square()
                .mean()
                .sqrt(),
                "flow_jepa_progressive_g3_coarse_prior_contract_min": (
                    coarse_bias_scale.detach().float().amin()
                ),
                "flow_jepa_progressive_g3_fine_prior_contract_min": (
                    fine_bias_scale.detach().float().amin()
                ),
                "flow_jepa_progressive_g3_slot_entropy": (
                    -(
                        slot_weights.float().clamp_min(1e-8)
                        * slot_weights.float().clamp_min(1e-8).log()
                    ).sum(dim=-1)
                    / math.log(float(max(self.slots, 2)))
                )
                .detach()
                .mean(),
                "flow_jepa_progressive_g3_summary_camera_variation": summary.detach()
                .float()
                .std(dim=1, unbiased=False)
                .mean(),
                "flow_jepa_structured_ownership_bottleneck": summary.new_tensor(
                    float(self.structured_ownership), dtype=torch.float32
                ),
                "flow_jepa_pre_value_owner_routing": summary.new_tensor(
                    float(self.pre_value_owner_routing), dtype=torch.float32
                ),
            })
        if collect_diagnostics and self.structured_ownership:
            for name, weights in owner_slot_weights.items():
                metrics[f"flow_jepa_progressive_g3_{name}_slot_entropy"] = (
                    -(
                        weights.float().clamp_min(1e-8)
                        * weights.float().clamp_min(1e-8).log()
                    ).sum(dim=-1)
                    / math.log(float(max(self.slots, 2)))
                ).detach().mean()
            metrics[
                "flow_jepa_progressive_g3_semantic_appearance_slot_l1"
            ] = 0.5 * (
                owner_slot_weights["semantic"].detach()
                - owner_slot_weights["appearance"].detach()
            ).abs().sum(dim=-1).mean()
            metrics[
                "flow_jepa_progressive_g3_appearance_geometry_slot_l1"
            ] = 0.5 * (
                owner_slot_weights["appearance"].detach()
                - owner_slot_weights["geometry"].detach()
            ).abs().sum(dim=-1).mean()
            if self.exports_grounded_facts:
                parent_rows = {
                    "semantic": state.g2_semantic_slot_probability,
                    "appearance": state.g2_appearance_slot_probability,
                    "geometry": state.g2_geometry_slot_probability,
                }
                for name, parent in parent_rows.items():
                    if parent is None:
                        raise RuntimeError(
                            "grounded G3 diagnostics lost G2 ownership"
                        )
                    metrics[
                        f"grounded_g2_g3_{name}_owner_l1"
                    ] = 0.5 * (
                        owner_slot_weights[name].detach().float()
                        - parent.detach().float()
                    ).abs().sum(dim=-1).mean()
        if (
            intervention in _GROUNDED_G3_SLOT_INTERVENTIONS
            and not self.exports_grounded_facts
        ):
            raise RuntimeError(
                "G3 object-slot intervention requires grounded_intent_effect_323"
            )
        grounded_facts: GroundedFactSet | None = None
        if self.exports_grounded_facts:
            if (
                state.bank.dense_current_dino_content is None
                or state.dynamic_fine_valid is None
                or canonical_semantic is None
                or canonical_appearance is None
                or canonical_geometry is None
            ):
                raise RuntimeError(
                    "grounded G3 cannot form a complete object fact set"
                )
            content_slots = sample_spatial_slots(
                state.bank.dense_current_dino_content,
                state.rectified_centers,
            )
            slot_validity = state.dynamic_fine_valid.any(
                dim=-1,
                keepdim=True,
            ).to(dtype=rollout.dtype)
            grounded_facts = GroundedFactSet(
                public_scene_base=summary.to(dtype=rollout.dtype),
                content_slots=content_slots.to(dtype=rollout.dtype),
                semantic_slots=canonical_semantic.to(dtype=rollout.dtype),
                appearance_slots=canonical_appearance.to(dtype=rollout.dtype),
                geometry_slots=canonical_geometry.to(dtype=rollout.dtype),
                semantic_owner_probs=owner_slot_weights["semantic"].to(
                    dtype=rollout.dtype
                ),
                appearance_owner_probs=owner_slot_weights["appearance"].to(
                    dtype=rollout.dtype
                ),
                geometry_owner_probs=owner_slot_weights["geometry"].to(
                    dtype=rollout.dtype
                ),
                slot_coordinates=state.rectified_centers.to(
                    dtype=rollout.dtype
                ),
                slot_support=state.rectified_support.to(dtype=rollout.dtype),
                slot_validity=slot_validity,
                slot_transport_prior=(
                    (
                        state.bank.coarse_flow_centers
                        - state.bank.coarse_source_centers
                    )[..., None, :]
                    .expand_as(state.rectified_centers)
                    .to(dtype=rollout.dtype)
                    if (
                        state.bank.coarse_flow_centers is not None
                        and state.bank.coarse_source_centers is not None
                        and tuple(state.bank.coarse_flow_centers.shape)
                        == tuple(state.bank.coarse_source_centers.shape)
                        == tuple(state.rectified_centers.shape[:-2]) + (2,)
                    )
                    else torch.zeros_like(state.rectified_centers)
                ),
            )
            grounded_facts.validate()
            if intervention in _GROUNDED_G3_SLOT_INTERVENTIONS:
                grounded_facts, slot_delta = _intervene_grounded_fact_slots(
                    grounded_facts,
                    str(intervention),
                )
                # Keep the exported typed sidecar aligned with the exact
                # GroundedFactSet consumed by S/W.  The generic canonical
                # address key, coarse/fine priors and dynamic value lattice
                # remain untouched so P1 is an exact control path.
                canonical_semantic = grounded_facts.semantic_slots
                canonical_appearance = grounded_facts.appearance_slots
                canonical_geometry = grounded_facts.geometry_slots
                owner_slot_weights = {
                    "semantic": grounded_facts.semantic_owner_probs,
                    "appearance": grounded_facts.appearance_owner_probs,
                    "geometry": grounded_facts.geometry_owner_probs,
                }
                metrics[
                    "grounded_g3_slot_intervention_delta_norm"
                ] = slot_delta
                metrics[
                    "grounded_g3_slot_intervention_public_base_delta_norm"
                ] = (
                    grounded_facts.public_scene_base.detach().float()
                    - summary.detach().float()
                ).square().mean().sqrt()
        state.stage = 3
        state.canonical_coarse_bias = coarse_bias.to(dtype=rollout.dtype)
        state.canonical_fine_bias = fine_bias.to(dtype=rollout.dtype)
        state.canonical_slot_keys = slot_keys.to(dtype=rollout.dtype)
        state.canonical_semantic_keys = (
            canonical_semantic.to(dtype=rollout.dtype)
            if canonical_semantic is not None
            else None
        )
        state.canonical_appearance_keys = (
            canonical_appearance.to(dtype=rollout.dtype)
            if canonical_appearance is not None
            else None
        )
        state.canonical_geometry_keys = (
            canonical_geometry.to(dtype=rollout.dtype)
            if canonical_geometry is not None
            else None
        )
        state.canonical_summary_tokens = summary.reshape(
            int(summary.shape[0]), -1, self.hidden
        )
        state.canonical_semantic_slot_weights = owner_slot_weights.get(
            "semantic"
        )
        state.canonical_appearance_slot_weights = owner_slot_weights.get(
            "appearance"
        )
        state.canonical_geometry_slot_weights = owner_slot_weights.get(
            "geometry"
        )
        state.grounded_fact_set = grounded_facts
        state.metrics = metrics
        return state

    def advance_world_owner_state(
        self,
        rollout: Tensor,
        state: ProgressiveGroundingAddressState,
        *,
        depth: int,
        intervention: str | None = None,
        horizon_query_context: Tensor | None = None,
        intent_window_view: IntentWindowView | None = None,
        grounded_intent_state: GroundedIntentState | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance private selector state before values are available.

        ``depth=0`` is the G3->W entry; later depths are the configured
        post-W transitions.  Owner evidence remains in route space and is written to
        the shared rollout only through a small, bounded reconstruction.  This
        preserves a common W carrier without turning it into the sole owner of
        semantic, appearance, geometry, or interval information.
        """

        if not self.pre_value_owner_routing:
            return rollout, {}
        if self.grounded_intent_effect_mainline:
            if (
                self.grounded_world_compiler is None
                or state.grounded_fact_set is None
            ):
                raise RuntimeError(
                    "grounded W has no compiler or completed G3 fact set"
                )
            if state.world_owner_depth != int(depth) - 1:
                raise RuntimeError(
                    "grounded W boundaries must advance exactly once; "
                    f"previous={state.world_owner_depth}, requested={depth}"
                )
            batch = int(rollout.shape[0])
            if int(depth) == 0:
                if (
                    horizon_query_context is None
                    or tuple(horizon_query_context.shape)
                    != (batch, self.anchors, self.hidden)
                ):
                    raise ValueError(
                        "grounded W entry requires one [B,4,H] clean proposal"
                    )
                state.world_grounded_working_state = (
                    self.grounded_world_compiler.initialize(
                        horizon_query_context
                    )
                )
                state.world_owner_depth = 0
                return rollout, {
                    "grounded_w_clean_proposal_built_once": rollout.new_ones(
                        (),
                        dtype=torch.float32,
                    )
                }
            if grounded_intent_state is None:
                raise RuntimeError("grounded W lost its stateless intent state")
            working = state.world_grounded_working_state
            if working is None:
                raise RuntimeError("grounded W lost its private entry state")
            world_tokens = rollout.reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid,
                self.grid,
                self.hidden,
            )
            if int(depth) == 1:
                working, metrics = self.grounded_world_compiler.forward_w1(
                    world_tokens=world_tokens,
                    facts=state.grounded_fact_set,
                    intent=grounded_intent_state,
                    working=working,
                    output_dtype=rollout.dtype,
                    collect_diagnostics=collect_diagnostics,
                )
                state.world_grounded_effect_w1_field = working.effect_w1
            elif int(depth) == self.world_blocks:
                working, metrics = self.grounded_world_compiler.forward_w2(
                    world_tokens=world_tokens,
                    facts=state.grounded_fact_set,
                    intent=grounded_intent_state,
                    working=working,
                    output_dtype=rollout.dtype,
                    collect_diagnostics=collect_diagnostics,
                )
                state.world_grounded_effect_field = working.effect
            else:
                raise RuntimeError(
                    "grounded effect decoding is owned only by W1/W2"
                )
            state.world_grounded_working_state = working
            state.world_owner_depth = int(depth)
            return rollout, metrics
        if (
            self.world_owner_transitions is None
            or self.world_owner_writes is None
        ):
            raise RuntimeError("pre-value owner routing modules are missing")
        if not 0 <= int(depth) < len(self.world_owner_transitions):
            raise ValueError(
                "world owner depth is outside the configured G3/W schedule"
            )
        if state.stage != 3:
            raise RuntimeError("world owner routing requires the completed G3 state")
        required = {
            "semantic": (
                state.canonical_semantic_keys,
                state.canonical_semantic_slot_weights,
            ),
            "appearance": (
                state.canonical_appearance_keys,
                state.canonical_appearance_slot_weights,
            ),
            "geometry": (
                state.canonical_geometry_keys,
                state.canonical_geometry_slot_weights,
            ),
        }
        if any(
            key is None or weight is None
            for key, weight in required.values()
        ):
            raise RuntimeError("world owner routing has an incomplete G3 sidecar")
        if state.world_owner_depth != int(depth) - 1:
            raise RuntimeError(
                "world owner routing must advance exactly once at each "
                f"G3/W boundary; got previous={state.world_owner_depth}, "
                f"requested={depth}"
            )

        batch = int(rollout.shape[0])
        query = self.horizon_query_proj(
            self.horizon_query_norm(rollout)
        ).reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
            self.route_dim,
        )
        condition_rms = query.new_zeros((), dtype=torch.float32)
        if self.functional_mainline_routing:
            if (
                self.world_horizon_condition is None
                or horizon_query_context is None
                or tuple(horizon_query_context.shape)
                != (batch, self.anchors, self.hidden)
            ):
                raise ValueError(
                    "functional W routing requires [B,anchor,H] online "
                    "phase/goal/history selector context"
                )
            condition = self.world_horizon_condition[int(depth)](
                horizon_query_context.to(
                    device=query.device, dtype=rollout.dtype
                )
            )
            condition, _ = smooth_rms_contract(condition, 0.35)
            query = query + condition[:, :, None, None, None]
            if collect_diagnostics:
                condition_rms = (
                    condition.detach().float().square().mean().sqrt()
                )
        owner_evidence: dict[str, Tensor] = {}
        for name, (key, weight) in required.items():
            assert key is not None and weight is not None
            collapsed = (
                weight.to(dtype=key.dtype)[..., None] * key
            ).sum(dim=4)
            owner_evidence[name] = collapsed[:, None].expand(
                -1,
                self.anchors,
                -1,
                -1,
                -1,
                -1,
            )

        # A causal interval innovation: the first state is anchored to the
        # first real horizon query; every later state sees only its immediate
        # predecessor, never a later anchor or a future teacher.
        interval_evidence = torch.cat(
            (
                query[:, :1],
                query[:, 1:] - query[:, :-1],
            ),
            dim=1,
        )
        owner_evidence["interval"] = interval_evidence
        owner_names = (
            "semantic",
            "appearance",
            "geometry",
            "interval",
        )
        field_by_owner = {
            "semantic": "world_semantic_state",
            "appearance": "world_appearance_state",
            "geometry": "world_geometry_state",
            "interval": "world_interval_state",
        }
        transition_bank = self.world_owner_transitions[int(depth)]
        written_rows: list[Tensor] = []
        route_rows: list[Tensor] = []
        metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            metrics["flow_jepa_pre_value_owner_routing"] = rollout.new_ones(
                (), dtype=torch.float32
            )
        for name in owner_names:
            evidence = owner_evidence[name].to(dtype=query.dtype)
            previous = getattr(state, field_by_owner[name])
            if previous is None:
                if int(depth) != 0:
                    raise RuntimeError(
                        f"W{depth} has no previous {name} selector state"
                    )
                previous = evidence
            elif tuple(previous.shape) != tuple(query.shape):
                raise ValueError(f"{name} world owner state lost chart geometry")

            transition_input = torch.cat(
                (query, evidence, previous.to(dtype=query.dtype)),
                dim=-1,
            )
            raw_delta = transition_bank[name](transition_input)
            delta, delta_contract = smooth_rms_contract(raw_delta, 0.35)
            updated, state_contract = smooth_rms_contract(
                previous.to(dtype=delta.dtype) + delta,
                0.75,
            )
            mode = "" if intervention is None else str(intervention)
            interval_zero = (
                self.functional_mainline_routing
                and name == "interval"
                and mode == "interval_stage_zero"
            )
            interval_shuffle = (
                self.functional_mainline_routing
                and name == "interval"
                and mode == "interval_stage_shuffle"
            )
            if mode == f"{name}_owner_zero" or interval_zero:
                updated = torch.zeros_like(updated)
                delta = torch.zeros_like(delta)
            elif mode == f"{name}_owner_shuffle" or interval_shuffle:
                shifts = (
                    max(self.grid // 2, 1),
                    max(self.grid // 3, 1),
                )
                updated = updated.roll(shifts=shifts, dims=(3, 4))
                delta = delta.roll(shifts=shifts, dims=(3, 4))
            setattr(state, field_by_owner[name], updated.to(dtype=rollout.dtype))

            write_source = updated if int(depth) == 0 else delta
            if self.functional_mainline_routing:
                route_rows.append(write_source)
                owner_write = write_source
                write_contract = write_source.new_ones(
                    (*write_source.shape[:-1], 1), dtype=torch.float32
                )
            else:
                owner_write, write_contract = smooth_rms_contract(
                    self.world_owner_writes[name](write_source),
                    0.35,
                )
                written_rows.append(owner_write)
            if collect_diagnostics:
                prefix = f"flow_jepa_pre_value_w{depth}_{name}"
                metrics[f"{prefix}_state_rms"] = (
                    updated.detach().float().square().mean().sqrt()
                )
                metrics[f"{prefix}_delta_rms"] = (
                    delta.detach().float().square().mean().sqrt()
                )
                metrics[f"{prefix}_write_rms"] = (
                    owner_write.detach().float().square().mean().sqrt()
                )
                metrics[f"{prefix}_delta_contract_min"] = (
                    delta_contract.detach().float().amin()
                )
                metrics[f"{prefix}_state_contract_min"] = (
                    state_contract.detach().float().amin()
                )
                metrics[f"{prefix}_write_contract_min"] = (
                    write_contract.detach().float().amin()
                )

        if self.functional_mainline_routing:
            if (
                self.world_owner_route_attnres is None
                or self.world_owner_fused_writes is None
                or len(route_rows) != len(owner_names)
            ):
                raise RuntimeError("functional W owner router is incomplete")
            route_values = torch.stack(route_rows, dim=-2)
            selected_route, route_metrics = self.world_owner_route_attnres[
                int(depth)
            ](
                query,
                route_values,
                collect_diagnostics=collect_diagnostics,
            )
            mode = "" if intervention is None else str(intervention)
            baseline_selected_route = selected_route
            if mode == f"functional_w{int(depth)}_route_zero":
                selected_route = torch.zeros_like(selected_route)
            elif mode == f"functional_w{int(depth)}_route_shuffle":
                # Keep the intervention invariant to probe batch size.  This
                # tests whether the selected owner write is attached to the
                # correct spatial address without replacing it with another
                # episode's representation when B > 1.
                selected_route = selected_route.roll(
                    shifts=(
                        max(self.grid // 2, 1),
                        max(self.grid // 3, 1),
                    ),
                    dims=(3, 4),
                )
            if mode in {
                f"functional_w{int(depth)}_route_zero",
                f"functional_w{int(depth)}_route_shuffle",
            }:
                metrics[
                    f"flow_jepa_functional_w{depth}_route_"
                    "intervention_delta_norm"
                ] = (
                    selected_route.detach().float()
                    - baseline_selected_route.detach().float()
                ).norm(dim=-1).mean()
            combined_write, combined_contract = smooth_rms_contract(
                self.world_owner_fused_writes[int(depth)](selected_route),
                0.50,
            )
            interval_prediction, _ = smooth_rms_contract(
                self.world_owner_fused_writes[int(depth)](route_rows[-1]),
                0.50,
            )
            state.world_interval_progress_prediction = (
                interval_prediction.to(dtype=rollout.dtype)
            )
            decode_effect = bool(
                self.g_aligned_future_effect
                and (
                    int(depth) == self.world_blocks
                    or (
                        self.supervised_effect_mainline
                        and 1 <= int(depth) <= self.world_blocks
                    )
                )
            )
            if (
                decode_effect
                and self.window_effect_bank
                and self.differential_intent_effect_mainline
            ):
                if (
                    self.differential_window_compiler is None
                    or intent_window_view is None
                    or state.canonical_semantic_keys is None
                ):
                    raise RuntimeError(
                        "differential W lost its compiler, intent view, or "
                        "protected current G3 keys"
                    )
                if int(depth) == 1:
                    (
                        effect_field,
                        route_state,
                        differential_metrics,
                    ) = self.differential_window_compiler.forward_w1(
                        selected_route,
                        state.canonical_semantic_keys,
                        intent_window_view,
                        output_dtype=rollout.dtype,
                        collect_diagnostics=collect_diagnostics,
                    )
                    state.world_differential_effect_w1_field = effect_field
                    state.world_differential_effect_route_state = route_state
                elif int(depth) == self.world_blocks:
                    w1_field = state.world_differential_effect_w1_field
                    route_state = state.world_differential_effect_route_state
                    if w1_field is None or route_state is None:
                        raise RuntimeError(
                            "differential W2 has no completed near/mid state"
                        )
                    (
                        effect_field,
                        route_state,
                        differential_metrics,
                    ) = self.differential_window_compiler.forward_w2(
                        selected_route,
                        state.canonical_semantic_keys,
                        intent_window_view,
                        w1_bank=w1_field,
                        w1_route_state=route_state,
                        output_dtype=rollout.dtype,
                        collect_diagnostics=collect_diagnostics,
                    )
                    state.world_differential_effect_field = effect_field
                    state.world_differential_effect_route_state = route_state
                else:
                    raise RuntimeError(
                        "differential effect decoding is owned only by W1/W2"
                    )
                metrics.update(differential_metrics)
                semantic_contract = selected_route.new_ones(
                    (),
                    dtype=torch.float32,
                )
                if collect_diagnostics:
                    semantic_interval = (
                        effect_field.semantic_delta.detach().float().mean(
                            dim=(2, 3, 4, 5)
                        )
                    )
                    transport_interval = (
                        effect_field.transport_mean.detach().float().mean(
                            dim=(2, 3, 4, 5)
                        )
                    )
                    metrics.update(
                        {
                            "flow_jepa_differential_effect_bank_active": (
                                rollout.new_ones((), dtype=torch.float32)
                            ),
                            "flow_jepa_future_effect_semantic_rms": (
                                effect_field.semantic_delta.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_transport_rms": (
                                effect_field.transport_mean.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_pred_adjacent_cosine": (
                                F.cosine_similarity(
                                    semantic_interval[:, 1:],
                                    semantic_interval[:, :-1],
                                    dim=-1,
                                    eps=1e-6,
                                ).mean()
                                if int(semantic_interval.shape[1]) > 1
                                else semantic_interval.new_zeros(())
                            ),
                            "flow_jepa_future_effect_pred_interval_variation": (
                                semantic_interval.std(
                                    dim=1,
                                    unbiased=False,
                                ).mean()
                            ),
                            "flow_jepa_future_effect_pred_transport_variation": (
                                transport_interval.std(
                                    dim=1,
                                    unbiased=False,
                                ).mean()
                            ),
                        }
                    )
                decode_effect = False
            elif decode_effect and self.window_effect_bank:
                if (
                    self.window_successor_cell is None
                    or self.window_late_cell is None
                    or not self.supervised_effect_mainline
                ):
                    raise RuntimeError("V117 window-effect cells are incomplete")
                zero_route = torch.zeros_like(selected_route[:, 0])
                if int(depth) == 1:
                    near_route = self.window_successor_cell(
                        torch.cat((selected_route[:, 0], zero_route), dim=-1)
                    )
                    near_route, _ = smooth_rms_contract(near_route, 0.75)
                    mid_route = self.window_successor_cell(
                        torch.cat((selected_route[:, 1], near_route), dim=-1)
                    )
                    mid_route, _ = smooth_rms_contract(mid_route, 0.75)
                    route_state = torch.stack((near_route, mid_route), dim=1)
                    near_mid_field, semantic_contract = (
                        self._decode_supervised_effect_routes(
                            route_state,
                            state,
                            rollout_dtype=rollout.dtype,
                        )
                    )
                    placeholder_field, _ = self._decode_supervised_effect_routes(
                        zero_route[:, None],
                        state,
                        rollout_dtype=rollout.dtype,
                    )
                    effect_field = self._concat_effect_fields(
                        near_mid_field,
                        placeholder_field,
                        slot_valid=selected_route.new_tensor(
                            (1.0, 1.0, 0.0), dtype=torch.float32
                        ),
                    )
                    state.world_window_effect_route_state = route_state
                    state.world_future_effect_w1_field = effect_field
                elif int(depth) == self.world_blocks:
                    route_state = state.world_window_effect_route_state
                    w1_field = state.world_future_effect_w1_field
                    if (
                        route_state is None
                        or int(route_state.shape[1]) != 2
                        or w1_field is None
                    ):
                        raise RuntimeError(
                            "W2 has no causally completed near/mid effect state"
                        )
                    late_seed = selected_route[:, 2:].mean(dim=1)
                    near_mid_summary = route_state.mean(dim=1)
                    late_route = self.window_late_cell(
                        torch.cat((late_seed, near_mid_summary), dim=-1)
                    )
                    late_route, _ = smooth_rms_contract(late_route, 0.75)
                    late_field, semantic_contract = (
                        self._decode_supervised_effect_routes(
                            late_route[:, None],
                            state,
                            rollout_dtype=rollout.dtype,
                        )
                    )
                    effect_field = self._concat_effect_fields(
                        self._slice_effect_field(w1_field, slice(0, 2)),
                        late_field,
                        slot_valid=selected_route.new_ones(3, dtype=torch.float32),
                    )
                    state.world_window_effect_route_state = torch.cat(
                        (route_state, late_route[:, None]), dim=1
                    )
                    state.world_future_effect_field = effect_field
                else:
                    raise RuntimeError("V117 effect decoding is owned only by W1/W2")
                if collect_diagnostics:
                    semantic_interval = (
                        effect_field.semantic_delta.detach().float().mean(
                            dim=(2, 3, 4, 5)
                        )
                    )
                    transport_interval = (
                        effect_field.transport_mean.detach().float().mean(
                            dim=(2, 3, 4, 5)
                        )
                    )
                    metrics.update(
                        {
                            "flow_jepa_window_effect_bank_active": rollout.new_ones(
                                (), dtype=torch.float32
                            ),
                            "flow_jepa_future_effect_field_active": rollout.new_ones(
                                (), dtype=torch.float32
                            ),
                            "flow_jepa_future_effect_semantic_rms": (
                                effect_field.semantic_delta.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_transport_rms": (
                                effect_field.transport_mean.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_semantic_contract_min": (
                                semantic_contract.detach().float().amin()
                            ),
                            "flow_jepa_future_effect_pred_adjacent_cosine": (
                                F.cosine_similarity(
                                    semantic_interval[:, 1:],
                                    semantic_interval[:, :-1],
                                    dim=-1,
                                    eps=1e-6,
                                ).mean()
                            ),
                            "flow_jepa_future_effect_pred_interval_variation": (
                                semantic_interval.std(dim=1, unbiased=False).mean()
                            ),
                            "flow_jepa_future_effect_pred_transport_variation": (
                                transport_interval.std(dim=1, unbiased=False).mean()
                            ),
                        }
                    )
                    for slot_index, slot_name in enumerate(
                        ("near", "mid", "late")
                    ):
                        metrics[
                            f"flow_jepa_window_effect_{slot_name}_semantic_rms"
                        ] = (
                            effect_field.semantic_delta[:, slot_index]
                            .detach()
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                        )
                # The V116 decoder below would decode every anchor again and
                # overwrite W1-owned near/mid slots. V117 has already produced
                # the complete stage-owned interface.
                decode_effect = False
            if decode_effect:
                if (
                    self.future_effect_semantic is None
                    or self.future_effect_geometry is None
                    or state.canonical_semantic_keys is None
                    or (
                        not self.supervised_effect_mainline
                        and (
                            state.canonical_appearance_keys is None
                            or state.canonical_geometry_keys is None
                        )
                    )
                ):
                    raise RuntimeError(
                        "completed W state has no V115 future-effect compiler"
                    )
                route_slots = selected_route[..., None, :].expand(
                    -1, -1, -1, -1, -1, self.slots, -1
                )
                semantic_slots = state.canonical_semantic_keys[:, None].expand(
                    -1, self.anchors, -1, -1, -1, -1, -1
                )
                appearance_slots = None
                geometry_slots = None
                if not self.supervised_effect_mainline:
                    assert state.canonical_appearance_keys is not None
                    assert state.canonical_geometry_keys is not None
                    appearance_slots = (
                        state.canonical_appearance_keys[:, None].expand_as(
                            semantic_slots
                        )
                    )
                    geometry_slots = (
                        state.canonical_geometry_keys[:, None].expand_as(
                            semantic_slots
                        )
                    )
                effect_input = (
                    route_slots
                    if self.supervised_effect_mainline
                    else torch.cat(
                        (
                            route_slots,
                            semantic_slots,
                            appearance_slots,
                            geometry_slots,
                        ),
                        dim=-1,
                    )
                )
                raw_semantic_delta = self.future_effect_semantic(
                    effect_input
                )
                semantic_delta, semantic_contract = smooth_rms_contract(
                    raw_semantic_delta, 0.50
                )
                raw_geometry = self.future_effect_geometry(effect_input)
                transport_mean = 0.50 * torch.tanh(raw_geometry[..., :2])
                variance_diag = (
                    0.01
                    + 0.99 * torch.sigmoid(raw_geometry[..., 2:4])
                )
                covariance_cross = (
                    0.50
                    * torch.tanh(raw_geometry[..., 4:5])
                    * variance_diag.prod(dim=-1, keepdim=True).sqrt()
                )
                transport_covariance = torch.cat(
                    (variance_diag, covariance_cross), dim=-1
                )
                persistence = torch.sigmoid(raw_geometry[..., 5:6])
                visibility = torch.sigmoid(raw_geometry[..., 6:7])
                uncertainty = (
                    0.05
                    + 3.95
                    * torch.sigmoid(raw_geometry[..., 7:8] - 1.5)
                )
                if self.supervised_effect_mainline:
                    if self.future_effect_current is None:
                        raise RuntimeError(
                            "V116 effect decoder has no current-content projection"
                        )
                    # G3 supplies only the protected current reference. The
                    # successor delta and every geometric consequence are
                    # decoded solely from the selected W innovation.
                    current_content = self.future_effect_current(
                        semantic_slots
                    )
                    current_content, _ = smooth_rms_contract(
                        current_content, 0.75
                    )
                    successor_content = current_content + semantic_delta
                    effect_field = FutureEffectField(
                        semantic_delta=semantic_delta.to(dtype=rollout.dtype),
                        transport_mean=transport_mean.to(dtype=rollout.dtype),
                        transport_covariance=transport_covariance.to(
                            dtype=rollout.dtype
                        ),
                        persistence=persistence.to(dtype=rollout.dtype),
                        visibility=visibility.to(dtype=rollout.dtype),
                        uncertainty=uncertainty.to(dtype=rollout.dtype),
                        current_content=current_content.to(dtype=rollout.dtype),
                        successor_content=successor_content.to(
                            dtype=rollout.dtype
                        ),
                    )
                else:
                    effect_field = FutureEffectField(
                        semantic_delta=semantic_delta.to(dtype=rollout.dtype),
                        transport_mean=transport_mean.to(dtype=rollout.dtype),
                        transport_covariance=transport_covariance.to(
                            dtype=rollout.dtype
                        ),
                        persistence=persistence.to(dtype=rollout.dtype),
                        visibility=visibility.to(dtype=rollout.dtype),
                        uncertainty=uncertainty.to(dtype=rollout.dtype),
                        # V115 ancestry: the selected route crossed as an
                        # unsupervised state carrier.
                        state_innovation=route_slots.to(dtype=rollout.dtype),
                    )
                effect_field.validate()
                if self.supervised_effect_mainline and int(depth) == 1:
                    state.world_future_effect_w1_field = effect_field
                if int(depth) == self.world_blocks:
                    state.world_future_effect_field = effect_field
                if collect_diagnostics:
                    semantic_interval = semantic_delta.detach().float().mean(
                        dim=(2, 3, 4, 5)
                    )
                    transport_interval = transport_mean.detach().float().mean(
                        dim=(2, 3, 4, 5)
                    )
                    metrics.update(
                        {
                            "flow_jepa_future_effect_field_active": (
                                rollout.new_ones((), dtype=torch.float32)
                            ),
                            "flow_jepa_future_effect_semantic_rms": (
                                semantic_delta.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_transport_rms": (
                                transport_mean.detach()
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            "flow_jepa_future_effect_visibility_mean": (
                                visibility.detach().float().mean()
                            ),
                            "flow_jepa_future_effect_uncertainty_mean": (
                                uncertainty.detach().float().mean()
                            ),
                            "flow_jepa_future_effect_semantic_contract_min": (
                                semantic_contract.detach().float().amin()
                            ),
                            "flow_jepa_future_effect_pred_adjacent_cosine": (
                                F.cosine_similarity(
                                    semantic_interval[:, 1:],
                                    semantic_interval[:, :-1],
                                    dim=-1,
                                    eps=1e-6,
                                ).mean()
                            ),
                            "flow_jepa_future_effect_pred_interval_variation": (
                                semantic_interval.std(
                                    dim=1, unbiased=False
                                ).mean()
                            ),
                            "flow_jepa_future_effect_pred_transport_variation": (
                                transport_interval.std(
                                    dim=1, unbiased=False
                                ).mean()
                            ),
                        }
                    )
            written = combined_write
            if collect_diagnostics:
                for key, value in route_metrics.items():
                    if key == "source_mass":
                        for owner_index, owner_name in enumerate(
                            ("semantic", "appearance", "geometry", "interval")
                        ):
                            metrics[
                                f"flow_jepa_functional_w{depth}_{owner_name}_route_mass"
                            ] = value[owner_index].detach()
                    elif int(value.numel()) == 1:
                        metrics[
                            f"flow_jepa_functional_w{depth}_route_{key}"
                        ] = value.detach()
                metrics[
                    f"flow_jepa_functional_w{depth}_selected_route_rms"
                ] = selected_route.detach().float().square().mean().sqrt()
                metrics[
                    f"flow_jepa_functional_w{depth}_interval_prediction_rms"
                ] = (
                    interval_prediction.detach().float().square().mean().sqrt()
                )
        else:
            combined_write = sum(written_rows) / math.sqrt(
                float(len(written_rows))
            )
            combined_write, combined_contract = smooth_rms_contract(
                combined_write,
                0.50,
            )
            written = self.pre_value_owner_update_scale * combined_write
        refined = rollout + written.reshape_as(rollout)
        if collect_diagnostics:
            carrier_rms = rollout.detach().float().square().mean().sqrt()
            write_rms = written.detach().float().square().mean().sqrt()
            metrics.update(
                {
                    f"flow_jepa_pre_value_w{depth}_combined_write_rms": (
                        write_rms
                    ),
                    f"flow_jepa_pre_value_w{depth}_carrier_ratio": (
                        write_rms / carrier_rms.clamp_min(1e-8)
                    ),
                    f"flow_jepa_pre_value_w{depth}_combined_contract_min": (
                        combined_contract.detach().float().amin()
                    ),
                    f"flow_jepa_functional_w{depth}_condition_rms": condition_rms,
                    "flow_jepa_functional_mainline_routing": rollout.new_tensor(
                        float(self.functional_mainline_routing),
                        dtype=torch.float32,
                    ),
                }
            )
        state.world_owner_depth = int(depth)
        if state.metrics is None:
            state.metrics = {}
        state.metrics.update(metrics)
        return refined, metrics

    def _predict_future_transport(
        self,
        query: Tensor,
        state: ProgressiveGroundingAddressState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Predict a bounded soft transport and make it spatially consequential.

        The returned compatibility has shape
        ``[B,A,C,target-cell,source-cell,slot]``.  It is a finite additive
        logit, not a mask or selector: every valid source remains reachable.
        """

        if (
            self.future_transport is None
            or state.canonical_semantic_keys is None
            or state.canonical_appearance_keys is None
            or state.canonical_geometry_keys is None
            or state.rectified_centers is None
            or state.rectified_support is None
            or state.aligned_variance is None
        ):
            raise RuntimeError("typed W future-transport state is incomplete")
        batch = int(query.shape[0])
        query_context = query.mean(dim=3)[:, :, :, None, None, None]
        transport_shape = (
            batch,
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
            self.slots,
            self.route_dim,
        )
        query_context = query_context.expand(transport_shape)
        semantic = state.canonical_semantic_keys[:, None].expand(transport_shape)
        appearance = state.canonical_appearance_keys[:, None].expand(
            transport_shape
        )
        geometry = state.canonical_geometry_keys[:, None].expand(transport_shape)
        current_centers = state.rectified_centers.float()[:, None].expand(
            *transport_shape[:-1], 2
        )
        support = state.rectified_support.float()[:, None, ..., None].expand(
            *transport_shape[:-1], 1
        )
        variance = (
            state.aligned_variance.float()
            .clamp_min(0.0)
            .mean(dim=-1, keepdim=True)
            .sqrt()
        )[:, None].expand(*transport_shape[:-1], 1)
        transport_input = torch.cat(
            (
                query_context,
                semantic,
                (
                    torch.zeros_like(appearance)
                    if self.structured_ownership
                    else appearance
                ),
                geometry,
                current_centers.to(dtype=semantic.dtype),
                support.to(dtype=semantic.dtype),
                variance.to(dtype=semantic.dtype),
            ),
            dim=-1,
        )
        transport_raw = self.future_transport(transport_input).float()
        if self.structured_ownership:
            # The four W states describe chronological interval innovations,
            # not four unrelated endpoint guesses.  Composition is bounded
            # after each prefix, so later horizons can accumulate evidence
            # without an unbounded recurrent multiplier.
            interval_offset_delta = 0.12 * torch.tanh(
                transport_raw[..., :2]
            )
            cumulative_offset = interval_offset_delta.cumsum(dim=1)
            offset = 0.35 * torch.tanh(cumulative_offset / 0.35)
            interval_log_scale_delta = 0.15 * torch.tanh(
                transport_raw[..., 2:3]
            )
            cumulative_log_scale = interval_log_scale_delta.cumsum(dim=1)
            future_scale = torch.exp(
                0.5 * torch.tanh(cumulative_log_scale / 0.5)
            )
            visibility_logit = (
                0.75 * torch.tanh(transport_raw[..., 3:4])
            ).cumsum(dim=1)
            future_visibility = torch.sigmoid(visibility_logit)
            uncertainty_increment = support * (
                0.25 + 0.25 * F.softplus(transport_raw[..., 4:5])
            )
            future_uncertainty = torch.sqrt(
                variance.square()
                + uncertainty_increment.square().cumsum(dim=1)
                + 1e-6
            )
            state.world_interval_offset_delta = interval_offset_delta.to(
                dtype=query.dtype
            )
            state.world_interval_log_scale_delta = (
                interval_log_scale_delta.to(dtype=query.dtype)
            )
        else:
            interval_offset_delta = None
            interval_log_scale_delta = None
            offset = 0.35 * torch.tanh(transport_raw[..., :2])
            future_scale = torch.exp(0.5 * torch.tanh(transport_raw[..., 2:3]))
            future_visibility = torch.sigmoid(transport_raw[..., 3:4])
            future_uncertainty = torch.sqrt(
                variance.square()
                + (
                    support * (0.5 + F.softplus(transport_raw[..., 4:5]))
                ).square()
                + 1e-6
            )
        future_centers = current_centers + (
            1.0 - current_centers.square()
        ).clamp_min(0.0) * offset
        state.world_future_offset = offset.to(dtype=query.dtype)
        state.world_future_scale = future_scale.to(dtype=query.dtype)
        state.world_future_visibility = future_visibility.to(dtype=query.dtype)
        state.world_future_uncertainty = future_uncertainty.to(dtype=query.dtype)
        state.world_future_centers = future_centers.to(dtype=query.dtype)

        axis = torch.linspace(
            -1.0,
            1.0,
            self.grid,
            device=query.device,
            dtype=torch.float32,
        )
        target_y, target_x = torch.meshgrid(axis, axis, indexing="ij")
        target_coordinates = torch.stack(
            (target_x.reshape(-1), target_y.reshape(-1)), dim=-1
        ).reshape(1, 1, 1, self.grid * self.grid, 1, 1, 2)
        source_centers = future_centers.reshape(
            batch,
            self.anchors,
            self.cameras,
            1,
            self.grid * self.grid,
            self.slots,
            2,
        )
        source_scale = future_scale.reshape(
            batch,
            self.anchors,
            self.cameras,
            1,
            self.grid * self.grid,
            self.slots,
            1,
        )
        source_visibility = future_visibility.reshape_as(source_scale)
        source_uncertainty = future_uncertainty.reshape_as(source_scale)
        width = (0.05 + source_scale * source_uncertainty).clamp(0.05, 1.0)
        distance = (
            (target_coordinates - source_centers) / width
        ).square().sum(dim=-1)
        # Fixed, bounded geometry is a structural prior, not a learnable gate.
        # Its [-2.125, .125] range cannot delete semantic/appearance evidence.
        spatial_compatibility = 0.5 * (
            (-0.5 * distance).clamp_min(-4.0)
            + 0.25 * (2.0 * source_visibility[..., 0] - 1.0)
        )
        metrics = {
            "flow_jepa_progressive_future_transport_offset_rms": (
                offset.detach().square().mean().sqrt()
            ),
            "flow_jepa_progressive_future_transport_scale_mean": (
                future_scale.detach().mean()
            ),
            "flow_jepa_progressive_future_transport_visibility_mean": (
                future_visibility.detach().mean()
            ),
            "flow_jepa_progressive_future_transport_uncertainty_mean": (
                future_uncertainty.detach().mean()
            ),
            "flow_jepa_progressive_future_transport_horizon_variation": (
                (future_centers[:, 1:] - future_centers[:, :-1]).abs().mean()
                if self.anchors > 1
                else future_centers.new_zeros(())
            ),
            "flow_jepa_progressive_future_transport_spatial_logit_rms": (
                spatial_compatibility.detach().square().mean().sqrt()
            ),
            "flow_jepa_progressive_future_transport_spatial_logit_span": (
                spatial_compatibility.detach().amax(dim=(-2, -1))
                - spatial_compatibility.detach().amin(dim=(-2, -1))
            ).mean(),
            "flow_jepa_progressive_future_interval_offset_delta_rms": (
                interval_offset_delta.detach().square().mean().sqrt()
                if interval_offset_delta is not None
                else offset.new_zeros(())
            ),
            "flow_jepa_progressive_future_interval_scale_delta_rms": (
                interval_log_scale_delta.detach().square().mean().sqrt()
                if interval_log_scale_delta is not None
                else offset.new_zeros(())
            ),
            "flow_jepa_progressive_future_uncertainty_horizon_variation": (
                (
                    future_uncertainty[:, 1:]
                    - future_uncertainty[:, :-1]
                ).abs().mean()
                if self.anchors > 1
                else future_uncertainty.new_zeros(())
            ),
        }
        return spatial_compatibility, metrics

    def score_horizon_posterior(
        self,
        rollout: Tensor,
        state: ProgressiveGroundingAddressState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if state.stage != 3 or state.canonical_slot_keys is None:
            raise RuntimeError("horizon posterior requires a canonical G3 state")
        if state.canonical_coarse_bias is None:
            raise RuntimeError("horizon posterior requires canonical coarse bias")
        batch = int(rollout.shape[0])
        query = self.horizon_query_proj(
            self.horizon_query_norm(rollout)
        ).reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid * self.grid,
            self.route_dim,
        )
        if self.structured_ownership:
            public_query = query.mean(dim=1, keepdim=True)
            horizon_innovation = query - public_query
            # Public task/scene context may condition every owner, but the
            # horizon-varying innovation carries four times its amplitude so
            # a shared direction cannot erase interval identity again.
            owner_query = (
                horizon_innovation + 0.25 * public_query
            ) / math.sqrt(1.0 + 0.25**2)
            state.world_public_query = public_query.to(dtype=rollout.dtype)
            state.world_horizon_innovation = horizon_innovation.to(
                dtype=rollout.dtype
            )
        else:
            public_query = query.mean(dim=1, keepdim=True)
            horizon_innovation = query - public_query
            owner_query = query
        owner_query_rows: dict[str, Tensor] = {}
        transport_query = query
        if self.pre_value_owner_routing:
            if state.world_owner_depth != self.world_blocks:
                raise RuntimeError(
                    "P posterior requires the final configured W owner state; "
                    f"got depth={state.world_owner_depth}"
                )
            private_rows = {
                "semantic": state.world_semantic_state,
                "appearance": state.world_appearance_state,
                "geometry": state.world_geometry_state,
                "interval": state.world_interval_state,
            }
            if any(value is None for value in private_rows.values()):
                raise RuntimeError("P posterior has an incomplete W private bundle")
            for name in ("semantic", "appearance", "geometry"):
                private = private_rows[name]
                assert private is not None
                private = private.reshape(
                    batch,
                    self.anchors,
                    self.cameras,
                    self.grid * self.grid,
                    self.route_dim,
                )
                owner_query_rows[name] = (
                    owner_query + private.to(dtype=owner_query.dtype)
                ) / math.sqrt(2.0)
            interval = private_rows["interval"]
            assert interval is not None
            transport_query = (
                query
                + interval.reshape(
                    batch,
                    self.anchors,
                    self.cameras,
                    self.grid * self.grid,
                    self.route_dim,
                ).to(dtype=query.dtype)
            ) / math.sqrt(2.0)
        key = self.candidate_norm(state.canonical_slot_keys).reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
            self.route_dim,
        )
        typed_key_rows: dict[str, Tensor] = {}
        typed_query_rows: dict[str, Tensor] = {}
        if self.coordinate_typed_raw_detail:
            if self.world_typed_query is None:
                raise RuntimeError("typed W query projections are missing")
            for name, value in (
                ("semantic", state.canonical_semantic_keys),
                ("appearance", state.canonical_appearance_keys),
                ("geometry", state.canonical_geometry_keys),
            ):
                if value is None:
                    raise RuntimeError(f"W posterior has no canonical {name} key")
                typed_key_rows[name] = self.candidate_norm(value).reshape(
                    batch,
                    self.cameras,
                    self.grid * self.grid,
                    self.slots,
                    self.route_dim,
                )
                # Keep parameterized projections in autocast; the following
                # disabled-autocast block is reserved for FP32 similarity and
                # probability arithmetic over already-projected activations.
                typed_query_rows[name] = self.world_typed_query[name](
                    owner_query_rows.get(name, owner_query)
                )
        bias = state.canonical_coarse_bias.float().reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
        )
        future_spatial_compatibility: Tensor | None = None
        future_metrics: dict[str, Tensor] = {}
        if self.coordinate_typed_raw_detail:
            (
                future_spatial_compatibility,
                future_metrics,
            ) = self._predict_future_transport(transport_query, state)
        with torch.autocast(device_type=rollout.device.type, enabled=False):
            # [B,A,C,target-cell,source-cell,slot].  Keeping both spatial axes
            # here is what lets one computation serve the JEPA target marginal
            # and the source-state prior consumed by P without reading values.
            if typed_key_rows:
                typed_world_logits = {
                    name: torch.einsum(
                        "bactr,bcsur->bactsu",
                        typed_query_rows[name].float(),
                        typed_key.float(),
                    )
                    * (float(self.route_dim) ** -0.5)
                    for name, typed_key in typed_key_rows.items()
                }
                if future_spatial_compatibility is None:
                    raise RuntimeError("typed W logits have no future transport")
                if self.structured_ownership:
                    # W coarse ownership is semantic relevance constrained by
                    # transported geometry.  Appearance remains available to
                    # P's fine verifier and does not redundantly vote on the
                    # source chart here.
                    logits = (
                        typed_world_logits["semantic"]
                        + typed_world_logits["geometry"]
                    ) / math.sqrt(2.0)
                    logits = logits + future_spatial_compatibility.float()
                else:
                    logits = sum(typed_world_logits.values()) / math.sqrt(3.0)
                    logits = logits + future_spatial_compatibility.float()
                if self.pre_value_owner_routing:
                    # Turn the W appearance state into a source/slot-aligned
                    # verifier query without reading any value.  Soft target
                    # aggregation retains the all-target/all-source address
                    # contract; P1 later compares this query with every local
                    # appearance candidate, so W appearance can change the
                    # within-slot fine posterior rather than only a late P2
                    # condition.
                    appearance_target_logit = (
                        typed_world_logits["appearance"]
                        + future_spatial_compatibility.float()
                    )
                    appearance_target_weight = torch.softmax(
                        appearance_target_logit,
                        dim=3,
                    )
                    appearance_fine_query = torch.einsum(
                        "bactsu,bactr->bacsur",
                        appearance_target_weight,
                        typed_query_rows["appearance"].float(),
                    )
                    state.world_appearance_fine_query = (
                        appearance_fine_query.reshape(
                            batch,
                            self.anchors,
                            self.cameras,
                            self.grid,
                            self.grid,
                            self.slots,
                            self.route_dim,
                        ).to(dtype=rollout.dtype)
                    )
            else:
                typed_world_logits = {}
                logits = torch.einsum(
                    "bactr,bcsur->bactsu", query.float(), key.float()
                ) * (float(self.route_dim) ** -0.5)
            logits = logits + bias[:, None, :, None]
            relevance = torch.logsumexp(logits, dim=(-2, -1)) - math.log(
                float(max(self.grid * self.grid * self.slots, 1))
            )
            source_bias = torch.logsumexp(logits, dim=3) - math.log(
                float(max(self.grid * self.grid, 1))
            )
            source_bias = source_bias - source_bias.mean(
                dim=(-2, -1), keepdim=True
            )
            flat_source_bias, source_contract_scale = smooth_rms_contract(
                source_bias.flatten(-2), 1.0
            )
            source_bias = flat_source_bias.reshape_as(source_bias)
            probability = torch.softmax(relevance, dim=-1)
            source_probability = torch.softmax(
                source_bias.flatten(-2), dim=-1
            )
            owner_source_biases: dict[str, Tensor] = {}
            owner_source_probabilities: dict[str, Tensor] = {}
            owner_slot_contract_scales: list[Tensor] = []
            owner_source_contract_scales: list[Tensor] = []
            if self.structured_ownership and typed_world_logits:
                owner_slot_rows = {
                    "semantic": state.canonical_semantic_slot_weights,
                    "appearance": state.canonical_appearance_slot_weights,
                    "geometry": state.canonical_geometry_slot_weights,
                }
                for name, owner_logits in typed_world_logits.items():
                    owner_slot_weight = owner_slot_rows[name]
                    if owner_slot_weight is None:
                        raise RuntimeError(
                            f"typed W owner {name} has no G3 slot posterior"
                        )
                    owner_slot_prior = (
                        owner_slot_weight.float().clamp_min(1e-8).log()
                        + math.log(float(max(self.slots, 1)))
                    )
                    owner_slot_prior = owner_slot_prior - owner_slot_prior.mean(
                        dim=-1, keepdim=True
                    )
                    contracted_prior, contract_scale = smooth_rms_contract(
                        owner_slot_prior.flatten(-2), 1.0
                    )
                    owner_slot_contract_scales.append(contract_scale)
                    owner_slot_prior = contracted_prior.reshape(
                        batch,
                        self.cameras,
                        self.grid * self.grid,
                        self.slots,
                    )
                    # G3's typed slot decision remains private but is not
                    # audit-only: it directly conditions the matching W owner
                    # source posterior and therefore P's typed local read.
                    owner_logits = owner_logits + owner_slot_prior[
                        :, None, :, None
                    ]
                    if name == "geometry":
                        owner_logits = (
                            owner_logits + future_spatial_compatibility.float()
                        )
                    owner_bias = torch.logsumexp(owner_logits, dim=3) - math.log(
                        float(max(self.grid * self.grid, 1))
                    )
                    owner_bias = owner_bias - owner_bias.mean(
                        dim=(-2, -1), keepdim=True
                    )
                    contracted_bias, owner_source_scale = smooth_rms_contract(
                        owner_bias.flatten(-2), 1.0
                    )
                    owner_source_contract_scales.append(owner_source_scale)
                    owner_bias = contracted_bias.reshape_as(owner_bias)
                    owner_source_biases[name] = owner_bias
                    owner_source_probabilities[name] = torch.softmax(
                        owner_bias.flatten(-2), dim=-1
                    )
            entropy = -(
                probability.clamp_min(1e-8)
                * probability.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(
                float(max(self.grid * self.grid, 2))
            )
            if self.anchors > 1:
                variation = (
                    probability[:, 1:] - probability[:, :-1]
                ).abs().sum(dim=-1).mul(0.5).mean()
                source_variation = (
                    source_probability[:, 1:]
                    - source_probability[:, :-1]
                ).abs().sum(dim=-1).mul(0.5).mean()
            else:
                variation = probability.new_zeros(())
                source_variation = source_probability.new_zeros(())
        teacher_relevance = relevance.reshape(
            batch, self.anchors, self.cameras, self.grid, self.grid
        )
        state.world_teacher_relevance_logits = teacher_relevance.to(
            dtype=rollout.dtype
        )
        state.world_source_bias = source_bias.reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
            self.slots,
        ).to(dtype=rollout.dtype)
        if self.structured_ownership:
            state.world_semantic_source_bias = owner_source_biases[
                "semantic"
            ].reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid,
                self.grid,
                self.slots,
            ).to(dtype=rollout.dtype)
            state.world_appearance_source_bias = owner_source_biases[
                "appearance"
            ].reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid,
                self.grid,
                self.slots,
            ).to(dtype=rollout.dtype)
            state.world_geometry_source_bias = owner_source_biases[
                "geometry"
            ].reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid,
                self.grid,
                self.slots,
            ).to(dtype=rollout.dtype)
        private_metrics: dict[str, Tensor] = {}
        if self.pre_value_owner_routing:
            private_states = (
                state.world_semantic_state,
                state.world_appearance_state,
                state.world_geometry_state,
                state.world_interval_state,
            )
            if any(value is None for value in private_states):
                raise RuntimeError("final W private bundle is incomplete")
            private_rms = torch.stack(
                [
                    value.detach().float().square().mean().sqrt()
                    for value in private_states
                    if value is not None
                ]
            ).mean()
            public_rms = public_query.detach().float().square().mean().sqrt()
            private_metrics = {
                "flow_jepa_progressive_world_private_state_rms": private_rms,
                "flow_jepa_progressive_world_public_private_ratio": (
                    public_rms / (public_rms + private_rms).clamp_min(1e-8)
                ),
            }
        return teacher_relevance, {
            "flow_jepa_progressive_world_posterior_entropy": entropy.detach().mean(),
            "flow_jepa_progressive_world_posterior_max": probability.detach()
            .amax(dim=-1)
            .mean(),
            "flow_jepa_progressive_world_horizon_variation": variation.detach(),
            "flow_jepa_progressive_world_source_prior_entropy": (
                -(
                    source_probability.clamp_min(1e-8)
                    * source_probability.clamp_min(1e-8).log()
                ).sum(dim=-1)
                / math.log(
                    float(max(self.grid * self.grid * self.slots, 2))
                )
            ).detach().mean(),
            "flow_jepa_progressive_world_source_prior_max": (
                source_probability.detach().amax(dim=-1).mean()
            ),
            "flow_jepa_progressive_world_source_prior_rms": (
                source_bias.detach().square().mean().sqrt()
            ),
            "flow_jepa_progressive_world_source_horizon_variation": (
                source_variation.detach()
            ),
            "flow_jepa_progressive_world_source_contract_min": (
                source_contract_scale.detach().float().amin()
            ),
            "flow_jepa_progressive_world_semantic_logit_rms": (
                typed_world_logits["semantic"].detach().square().mean().sqrt()
                if typed_world_logits
                else rollout.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_appearance_logit_rms": (
                typed_world_logits["appearance"].detach().square().mean().sqrt()
                if typed_world_logits
                else rollout.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_geometry_logit_rms": (
                typed_world_logits["geometry"].detach().square().mean().sqrt()
                if typed_world_logits
                else rollout.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_public_query_rms": (
                public_query.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_progressive_world_horizon_innovation_rms": (
                horizon_innovation.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_progressive_world_public_ratio": (
                public_query.detach().float().square().mean().sqrt()
                / (
                    public_query.detach().float().square().mean().sqrt()
                    + horizon_innovation.detach().float().square().mean().sqrt()
                ).clamp_min(1e-8)
            ),
            "flow_jepa_progressive_world_semantic_appearance_source_l1": (
                0.5
                * (
                    owner_source_probabilities["semantic"]
                    - owner_source_probabilities["appearance"]
                ).abs().sum(dim=-1).mean().detach()
                if owner_source_probabilities
                else rollout.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_semantic_geometry_source_l1": (
                0.5
                * (
                    owner_source_probabilities["semantic"]
                    - owner_source_probabilities["geometry"]
                ).abs().sum(dim=-1).mean().detach()
                if owner_source_probabilities
                else rollout.new_zeros((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_owner_slot_contract_min": (
                torch.stack(
                    [value.detach().float().amin() for value in owner_slot_contract_scales]
                ).amin()
                if owner_slot_contract_scales
                else rollout.new_ones((), dtype=torch.float32)
            ),
            "flow_jepa_progressive_world_owner_source_contract_min": (
                torch.stack(
                    [
                        value.detach().float().amin()
                        for value in owner_source_contract_scales
                    ]
                ).amin()
                if owner_source_contract_scales
                else rollout.new_ones((), dtype=torch.float32)
            ),
            **private_metrics,
            **future_metrics,
        }


class _HorizonSoftAddressJEPA(nn.Module):
    """Read one observation-only address bank with horizon-specific W queries.

    The 8x8 rollout chart is a query lattice, not a fixed image partition.
    Every target cell may read every source cell/slot in the same camera, and
    each source hypothesis retains its continuous high-resolution candidates.
    Future teachers never enter this module; they supervise only the resulting
    relevance distribution in the loss.
    """

    def __init__(self, config: Any, *, raw_dim: int) -> None:
        super().__init__()
        self.hidden = int(config.hidden_size)
        self.grid = int(config.flow_jepa_grid_size)
        self.cameras = int(config.num_cameras)
        self.anchors = int(config.future_anchors)
        self.slots = int(config.flow_jepa_address_slots)
        self.route_dim = int(config.flow_jepa_address_route_dim)
        self.raw_dim = int(raw_dim)
        self.update_scale = float(
            getattr(config, "flow_jepa_horizon_address_update_scale", 0.10)
        )
        self.cell_specific_fine_address = bool(
            int(getattr(config, "flow_jepa_horizon_cell_fine_address", 0))
        )
        self.query_chunk = int(
            getattr(config, "flow_jepa_address_query_chunk", 4)
        )
        self.variance_safe = bool(
            int(getattr(config, "flow_jepa_variance_safe_routing", 0))
        )
        self.normalization_floor = float(
            getattr(config, "flow_jepa_routing_norm_floor", 0.25)
        )
        self.value_max_rms = float(
            getattr(config, "flow_jepa_horizon_value_max_rms", 0.50)
        )
        self.query_norm = (
            nn.Identity()
            if self.variance_safe
            else nn.LayerNorm(self.hidden)
        )
        self.query_proj = nn.Linear(self.hidden, self.route_dim, bias=False)
        self.key_norm = (
            nn.Identity()
            if self.variance_safe
            else nn.LayerNorm(
                self.route_dim, elementwise_affine=False
            )
        )
        # A bias here would let the auxiliary branch learn a constant future
        # residual even when every observation-owned fine value is zero.
        self.value_norm = (
            nn.Identity()
            if self.variance_safe
            else nn.LayerNorm(
                self.raw_dim,
                elementwise_affine=False,
            )
        )
        self.value_out = nn.Linear(self.raw_dim, self.hidden, bias=False)
        if not self.cell_specific_fine_address:
            # Exact V105/V106 compatibility.  V107 instead keeps PyTorch's
            # variance-preserving bias-free initialization so the observable
            # value owner is not numerically silent next to the full carrier.
            nn.init.normal_(self.value_out.weight, mean=0.0, std=1e-3)

    def _read_cell_specific_fine_address(
        self,
        query: Tensor,
        coarse_key: Tensor,
        fine_key: Tensor,
        fine_value: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Read continuous candidates without pooling the target-cell query.

        Target cells are processed in small chunks.  This retains the full
        ``target cell -> source cell -> slot -> fine candidate`` relation while
        keeping the largest FP32 routing tensor independent of the 8x8 target
        chart size.
        """

        batch = int(query.shape[0])
        target_cells = int(query.shape[3])
        candidates = int(fine_key.shape[-2])
        valid_any = valid.any(dim=-1)
        state_count = self.grid * self.grid * self.slots
        scale = float(self.route_dim) ** -0.5
        address_rows: list[Tensor] = []
        relevance_rows: list[Tensor] = []
        route_entropy_rows: list[Tensor] = []
        route_max_rows: list[Tensor] = []
        fine_entropy_rows: list[Tensor] = []
        fine_max_rows: list[Tensor] = []
        cross_distance_rows: list[Tensor] = []
        axis = torch.linspace(
            -1.0,
            1.0,
            self.grid,
            device=query.device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        source_xy = torch.stack((xx, yy), dim=-1).reshape(
            1, 1, 1, 1, self.grid * self.grid, 2
        )
        target_xy_all = torch.stack((xx, yy), dim=-1).reshape(
            1, 1, 1, target_cells, 2
        )
        for start in range(0, target_cells, self.query_chunk):
            stop = min(start + self.query_chunk, target_cells)
            query_row = query[:, :, :, start:stop].float()
            fine_logits = torch.einsum(
                "bactr,bcsukr->bactsuk",
                query_row,
                fine_key.float(),
            ) * scale
            candidate_mask = valid[:, None, :, None]
            fine_logits = fine_logits.masked_fill(
                ~candidate_mask,
                torch.finfo(fine_logits.dtype).min,
            )
            safe_fine_logits = torch.where(
                valid_any[:, None, :, None, :, :, None],
                fine_logits,
                torch.zeros_like(fine_logits),
            )
            fine_weights = torch.softmax(safe_fine_logits, dim=-1)
            fine_weights = fine_weights * candidate_mask.float()
            fine_weights = fine_weights / fine_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            local_values = torch.einsum(
                "bactsuk,bcsukv->bactsuv",
                fine_weights,
                fine_value.float(),
            )
            valid_count = candidate_mask.float().sum(dim=-1).clamp_min(1.0)
            fine_evidence = (
                torch.logsumexp(safe_fine_logits, dim=-1)
                - valid_count.log()
            )
            fine_evidence = torch.where(
                valid_any[:, None, :, None],
                fine_evidence,
                fine_evidence.new_full((), -1e4),
            )
            route_logits = torch.einsum(
                "bactr,bcsur->bactsu",
                query_row,
                coarse_key.float(),
            ) * scale
            route_logits = route_logits + fine_evidence
            route_logits = route_logits.masked_fill(
                ~valid_any[:, None, :, None],
                torch.finfo(route_logits.dtype).min,
            )
            flat_logits = route_logits.reshape(
                batch,
                self.anchors,
                self.cameras,
                stop - start,
                state_count,
            )
            flat_valid = valid_any.reshape(
                batch, self.cameras, state_count
            )[:, None, :, None]
            any_valid = flat_valid.any(dim=-1, keepdim=True)
            safe_flat_logits = torch.where(
                any_valid,
                flat_logits,
                torch.zeros_like(flat_logits),
            )
            route_weights = torch.softmax(safe_flat_logits, dim=-1)
            route_weights = route_weights * flat_valid.float()
            route_weights = route_weights / route_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            route_grid = route_weights.reshape(
                batch,
                self.anchors,
                self.cameras,
                stop - start,
                self.grid * self.grid,
                self.slots,
            )
            address_rows.append(
                torch.einsum(
                    "bactsu,bactsuv->bactv",
                    route_grid,
                    local_values,
                )
            )
            valid_state_count = flat_valid.float().sum(dim=-1).clamp_min(1.0)
            relevance = (
                torch.logsumexp(safe_flat_logits, dim=-1)
                - valid_state_count.log()
            )
            relevance_rows.append(
                torch.where(
                    any_valid.squeeze(-1),
                    relevance,
                    torch.zeros_like(relevance),
                )
            )
            route_entropy_rows.append(
                -(
                    route_weights.clamp_min(1e-8)
                    * route_weights.clamp_min(1e-8).log()
                ).sum(dim=-1)
                / math.log(float(max(state_count, 2)))
            )
            route_max_rows.append(route_weights.max(dim=-1).values)
            fine_entropy_rows.append(
                -(
                    fine_weights.clamp_min(1e-8)
                    * fine_weights.clamp_min(1e-8).log()
                ).sum(dim=-1)
                / math.log(float(max(candidates, 2)))
            )
            fine_max_rows.append(fine_weights.max(dim=-1).values)
            source_mass = route_grid.sum(dim=-1)
            expected_source = (
                source_mass[..., None] * source_xy
            ).sum(dim=-2)
            cross_distance_rows.append(
                (
                    expected_source - target_xy_all[:, :, :, start:stop]
                ).square().sum(dim=-1).sqrt()
            )
        return (
            torch.cat(address_rows, dim=3),
            torch.cat(relevance_rows, dim=3),
            torch.cat(route_entropy_rows, dim=3),
            torch.cat(route_max_rows, dim=3),
            torch.cat(fine_entropy_rows, dim=3),
            torch.cat(fine_max_rows, dim=3),
            torch.cat(cross_distance_rows, dim=3).mean(),
        )

    def forward(
        self,
        future_tokens: Tensor,
        bank: SoftAddressLatticeBank,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if future_tokens.ndim != 3:
            raise ValueError("future address queries must be [B,N,H]")
        batch, tokens, hidden = future_tokens.shape
        expected_tokens = (
            self.anchors * self.cameras * self.grid * self.grid
        )
        if tokens != expected_tokens or hidden != self.hidden:
            raise ValueError(
                "future address queries must preserve "
                f"[anchor={self.anchors},camera={self.cameras},"
                f"grid={self.grid}x{self.grid},hidden={self.hidden}]"
            )
        coarse_keys = bank.coarse_keys
        fine_keys = bank.fine_keys
        fine_values = bank.fine_values
        fine_valid = bank.fine_valid
        candidates = int(fine_keys.shape[-2])
        expected_coarse = (
            batch,
            self.cameras,
            self.grid,
            self.grid,
            self.slots,
            self.route_dim,
        )
        if tuple(coarse_keys.shape) != expected_coarse:
            raise ValueError("future address coarse bank has invalid geometry")
        if tuple(fine_keys.shape) != (
            *expected_coarse[:-1],
            candidates,
            self.route_dim,
        ):
            raise ValueError("future address fine keys do not align with coarse slots")
        if tuple(fine_values.shape) != (
            *expected_coarse[:-1],
            candidates,
            self.raw_dim,
        ):
            raise ValueError("future address values have an invalid width")
        if tuple(fine_valid.shape) != tuple(fine_keys.shape[:-1]):
            raise ValueError("future address validity does not align with fine keys")

        if self.variance_safe:
            query_input, query_denominator = variance_floored_centered_norm(
                future_tokens,
                self.normalization_floor,
            )
            coarse_key_input, coarse_key_denominator = (
                variance_floored_centered_norm(
                    coarse_keys,
                    self.normalization_floor,
                )
            )
            fine_key_input, fine_key_denominator = (
                variance_floored_centered_norm(
                    fine_keys,
                    self.normalization_floor,
                )
            )
        else:
            query_input = self.query_norm(future_tokens)
            coarse_key_input = self.key_norm(coarse_keys)
            fine_key_input = self.key_norm(fine_keys)
            query_denominator = future_tokens.new_ones(
                (*future_tokens.shape[:-1], 1), dtype=torch.float32
            )
            coarse_key_denominator = coarse_keys.new_ones(
                (*coarse_keys.shape[:-1], 1), dtype=torch.float32
            )
            fine_key_denominator = fine_keys.new_ones(
                (*fine_keys.shape[:-1], 1), dtype=torch.float32
            )
        query = self.query_proj(query_input).reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid * self.grid,
            self.route_dim,
        )
        coarse_key = coarse_key_input.reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
            self.route_dim,
        )
        fine_key = fine_key_input.reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
            candidates,
            self.route_dim,
        )
        fine_value = fine_values.reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
            candidates,
            self.raw_dim,
        )
        valid = fine_valid.reshape(
            batch,
            self.cameras,
            self.grid * self.grid,
            self.slots,
            candidates,
        )
        valid_any = valid.any(dim=-1)
        state_count = self.grid * self.grid * self.slots
        scale = float(self.route_dim) ** -0.5
        if self.cell_specific_fine_address:
            with torch.autocast(
                device_type=future_tokens.device.type,
                enabled=False,
            ):
                (
                    address_value,
                    relevance_logits,
                    route_entropy,
                    route_max,
                    fine_entropy,
                    fine_max,
                    cross_cell_distance,
                ) = self._read_cell_specific_fine_address(
                    query.float(),
                    coarse_key.float(),
                    fine_key.float(),
                    fine_value.float(),
                    valid,
                )
            address_value_model = address_value.to(dtype=future_tokens.dtype)
            address_value_rms = (
                address_value.detach().float().square().mean().sqrt()
            )
            address_value_channel_std = address_value.detach().float().std(
                dim=-1,
                unbiased=False,
            )
            if self.variance_safe:
                address_value_input, address_value_scale = smooth_rms_contract(
                    address_value_model,
                    self.value_max_rms,
                )
            else:
                address_value_input = self.value_norm(address_value_model)
                address_value_scale = address_value.new_ones(
                    (*address_value.shape[:-1], 1), dtype=torch.float32
                )
            address_update = self.value_out(address_value_input)
            address_update, _ = smooth_rms_contract(address_update, 0.50)
            address_update = address_update.reshape_as(future_tokens)
            refined = future_tokens + self.update_scale * address_update
            relevance_probability = torch.softmax(
                relevance_logits.reshape(batch, self.anchors, -1).float(),
                dim=-1,
            )
            relevance_entropy = -(
                relevance_probability.clamp_min(1e-8)
                * relevance_probability.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(
                float(max(self.cameras * self.grid * self.grid, 2))
            )
            if self.anchors > 1:
                horizon_variation = (
                    relevance_probability[:, 1:]
                    - relevance_probability[:, :-1]
                ).abs().sum(dim=-1).mul(0.5).mean()
            else:
                horizon_variation = relevance_probability.new_zeros(())
            metrics = {
                "flow_jepa_horizon_soft_address": future_tokens.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_horizon_cell_fine_address": future_tokens.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_horizon_address_update_scale": future_tokens.new_tensor(
                    self.update_scale, dtype=torch.float32
                ),
                "flow_jepa_horizon_address_raw_update_rms": (
                    address_update.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_horizon_address_update_rms": (
                    self.update_scale
                    * address_update.detach().float().square().mean().sqrt()
                ),
                "flow_jepa_horizon_address_update_ratio": (
                    self.update_scale
                    * address_update.detach().float().square().mean().sqrt()
                    / future_tokens.detach().float().square().mean().sqrt().clamp_min(1e-8)
                ),
                "flow_jepa_horizon_address_route_entropy": route_entropy.mean().detach(),
                "flow_jepa_horizon_address_route_max": route_max.mean().detach(),
                "flow_jepa_horizon_address_fine_entropy": fine_entropy.mean().detach(),
                "flow_jepa_horizon_address_fine_max": fine_max.mean().detach(),
                "flow_jepa_horizon_address_relevance_entropy": relevance_entropy.mean().detach(),
                "flow_jepa_horizon_address_relevance_max": relevance_probability.max(
                    dim=-1
                ).values.mean().detach(),
                "flow_jepa_horizon_address_variation": horizon_variation.detach(),
                "flow_jepa_horizon_address_cross_cell_distance": cross_cell_distance.detach(),
                "flow_jepa_horizon_address_variance_safe": future_tokens.new_tensor(
                    float(self.variance_safe), dtype=torch.float32
                ),
                "flow_jepa_horizon_address_value_precontract_rms": address_value_rms,
                "flow_jepa_horizon_address_value_channel_std": address_value_channel_std.mean(),
                "flow_jepa_horizon_address_value_channel_std_min": address_value_channel_std.amin(),
                "flow_jepa_horizon_address_value_contraction": (
                    1.0 - address_value_scale.detach().float()
                ).mean(),
                "flow_jepa_horizon_address_query_norm_denominator_min": query_denominator.detach().float().amin(),
                "flow_jepa_horizon_address_coarse_key_denominator_min": coarse_key_denominator.detach().float().amin(),
                "flow_jepa_horizon_address_fine_key_denominator_min": fine_key_denominator.detach().float().amin(),
            }
            return refined, relevance_logits.reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid,
                self.grid,
            ), metrics
        with torch.autocast(device_type=future_tokens.device.type, enabled=False):
            query_f = query.float()
            # Fine offsets describe one observed source hypothesis. Pooling
            # only target xy here keeps memory bounded; horizon and camera
            # identity remain explicit and coarse routing is still target-cell
            # specific.
            fine_query = query_f.mean(dim=3)
            fine_logits = torch.einsum(
                "bacr,bcsukr->bacsuk",
                fine_query,
                fine_key.float(),
            ) * scale
            fine_logits = fine_logits.masked_fill(
                ~valid[:, None],
                torch.finfo(fine_logits.dtype).min,
            )
            safe_fine_logits = torch.where(
                valid_any[:, None, :, :, :, None],
                fine_logits,
                torch.zeros_like(fine_logits),
            )
            fine_weights = torch.softmax(safe_fine_logits, dim=-1)
            fine_weights = fine_weights * valid[:, None].float()
            fine_weights = fine_weights / fine_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            local_values = torch.einsum(
                "bacsuk,bcsukv->bacsuv",
                fine_weights,
                fine_value.float(),
            )
            valid_count = valid[:, None].float().sum(dim=-1).clamp_min(1.0)
            fine_evidence = (
                torch.logsumexp(safe_fine_logits, dim=-1)
                - valid_count.log()
            )
            fine_evidence = torch.where(
                valid_any[:, None],
                fine_evidence,
                fine_evidence.new_full((), -1e4),
            )

            route_logits = torch.einsum(
                "bactr,bcsur->bactsu",
                query_f,
                coarse_key.float(),
            ) * scale
            route_logits = route_logits + fine_evidence[:, :, :, None]
            route_logits = route_logits.masked_fill(
                ~valid_any[:, None, :, None],
                torch.finfo(route_logits.dtype).min,
            )
            flat_logits = route_logits.reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid * self.grid,
                state_count,
            )
            flat_valid = valid_any.reshape(
                batch, self.cameras, state_count
            )[:, None, :, None]
            any_valid = flat_valid.any(dim=-1, keepdim=True)
            safe_flat_logits = torch.where(
                any_valid, flat_logits, torch.zeros_like(flat_logits)
            )
            route_weights = torch.softmax(safe_flat_logits, dim=-1)
            route_weights = route_weights * flat_valid.float()
            route_weights = route_weights / route_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            route_weights_grid = route_weights.reshape(
                batch,
                self.anchors,
                self.cameras,
                self.grid * self.grid,
                self.grid * self.grid,
                self.slots,
            )
            address_value = torch.einsum(
                "bactsu,bacsuv->bactv",
                route_weights_grid,
                local_values,
            )
            valid_state_count = flat_valid.float().sum(dim=-1).clamp_min(1.0)
            relevance_logits = (
                torch.logsumexp(safe_flat_logits, dim=-1)
                - valid_state_count.log()
            )
            relevance_logits = torch.where(
                any_valid.squeeze(-1),
                relevance_logits,
                torch.zeros_like(relevance_logits),
            )

            route_entropy = -(
                route_weights.clamp_min(1e-8)
                * route_weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(max(state_count, 2)))
            fine_entropy = -(
                fine_weights.clamp_min(1e-8)
                * fine_weights.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(float(max(candidates, 2)))
            source_mass = route_weights_grid.sum(dim=-1)
            axis = torch.linspace(
                -1.0,
                1.0,
                self.grid,
                device=future_tokens.device,
                dtype=torch.float32,
            )
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            source_xy = torch.stack((xx, yy), dim=-1).reshape(
                1, 1, 1, 1, self.grid * self.grid, 2
            )
            target_xy = torch.stack((xx, yy), dim=-1).reshape(
                1, 1, 1, self.grid * self.grid, 2
            )
            expected_source = (
                source_mass[..., None] * source_xy
            ).sum(dim=-2)
            cross_cell_distance = (
                expected_source - target_xy
            ).square().sum(dim=-1).sqrt().mean()

        address_value_model = address_value.to(dtype=future_tokens.dtype)
        address_value_rms = address_value.detach().float().square().mean().sqrt()
        address_value_channel_std = address_value.detach().float().std(
            dim=-1,
            unbiased=False,
        )
        if self.variance_safe:
            address_value_input, address_value_scale = smooth_rms_contract(
                address_value_model,
                self.value_max_rms,
            )
        else:
            address_value_input = self.value_norm(address_value_model)
            address_value_scale = address_value.new_ones(
                (*address_value.shape[:-1], 1), dtype=torch.float32
            )
        address_update = self.value_out(address_value_input)
        address_update, _ = smooth_rms_contract(address_update, 0.50)
        address_update = address_update.reshape_as(future_tokens)
        refined = future_tokens + self.update_scale * address_update
        relevance_probability = torch.softmax(
            relevance_logits.reshape(batch, self.anchors, -1).float(),
            dim=-1,
        )
        relevance_entropy = -(
            relevance_probability.clamp_min(1e-8)
            * relevance_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(
            float(max(self.cameras * self.grid * self.grid, 2))
        )
        if self.anchors > 1:
            horizon_variation = (
                relevance_probability[:, 1:] - relevance_probability[:, :-1]
            ).abs().sum(dim=-1).mul(0.5).mean()
        else:
            horizon_variation = relevance_probability.new_zeros(())
        metrics = {
            "flow_jepa_horizon_soft_address": future_tokens.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_horizon_cell_fine_address": future_tokens.new_zeros(
                (), dtype=torch.float32
            ),
            "flow_jepa_horizon_address_update_scale": future_tokens.new_tensor(
                self.update_scale, dtype=torch.float32
            ),
            "flow_jepa_horizon_address_raw_update_rms": address_update.detach().float()
            .square()
            .mean()
            .sqrt(),
            "flow_jepa_horizon_address_update_rms": (
                self.update_scale
                * address_update.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_horizon_address_update_ratio": (
                self.update_scale
                * address_update.detach().float().square().mean().sqrt()
                / future_tokens.detach().float().square().mean().sqrt().clamp_min(1e-8)
            ),
            "flow_jepa_horizon_address_route_entropy": route_entropy.mean().detach(),
            "flow_jepa_horizon_address_route_max": route_weights.max(
                dim=-1
            ).values.mean().detach(),
            "flow_jepa_horizon_address_fine_entropy": fine_entropy.mean().detach(),
            "flow_jepa_horizon_address_fine_max": fine_weights.max(
                dim=-1
            ).values.mean().detach(),
            "flow_jepa_horizon_address_relevance_entropy": (
                relevance_entropy.mean().detach()
            ),
            "flow_jepa_horizon_address_relevance_max": relevance_probability.max(
                dim=-1
            ).values.mean().detach(),
            "flow_jepa_horizon_address_variation": horizon_variation.detach(),
            "flow_jepa_horizon_address_cross_cell_distance": (
                cross_cell_distance.detach()
            ),
            "flow_jepa_horizon_address_variance_safe": future_tokens.new_tensor(
                float(self.variance_safe), dtype=torch.float32
            ),
            "flow_jepa_horizon_address_value_precontract_rms": address_value_rms,
            "flow_jepa_horizon_address_value_channel_std": (
                address_value_channel_std.mean()
            ),
            "flow_jepa_horizon_address_value_channel_std_min": (
                address_value_channel_std.amin()
            ),
            "flow_jepa_horizon_address_value_contraction": (
                1.0 - address_value_scale.detach().float()
            ).mean(),
            "flow_jepa_horizon_address_query_norm_denominator_min": (
                query_denominator.detach().float().amin()
            ),
            "flow_jepa_horizon_address_coarse_key_denominator_min": (
                coarse_key_denominator.detach().float().amin()
            ),
            "flow_jepa_horizon_address_fine_key_denominator_min": (
                fine_key_denominator.detach().float().amin()
            ),
        }
        return refined, relevance_logits.reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
        ), metrics


class _IntervalStageDeltaOrganizer(nn.Module):
    """Causally organize four interval stages without touching fine values.

    The module operates independently at every camera/xy location and attends
    only along the four chronological horizon tokens.  Its bounded residual is
    written to the coarse W rollout at the W->P boundary, where it can
    condition typed depth selection and the late precision query.  It never
    receives or rewrites the observation-owned fine-value bank.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden = int(config.hidden_size)
        self.anchors = int(config.future_anchors)
        self.cameras = int(config.num_cameras)
        self.grid = int(config.flow_jepa_grid_size)
        self.update_scale = float(
            getattr(config, "flow_jepa_interval_stage_update_scale", 0.10)
        )
        self.normalization_floor = float(
            getattr(config, "flow_jepa_routing_norm_floor", 0.25)
        )
        heads = int(config.num_heads)
        if self.hidden % heads:
            raise ValueError(
                "interval-stage hidden size must be divisible by num_heads"
            )
        self.temporal_attention = nn.MultiheadAttention(
            self.hidden,
            heads,
            batch_first=True,
            bias=False,
        )
        self.temporal_ffn = nn.Sequential(
            nn.Linear(self.hidden, 2 * self.hidden, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.hidden, self.hidden, bias=False),
        )
        self.delta_out = nn.Linear(self.hidden, self.hidden, bias=False)
        nn.init.normal_(self.temporal_ffn[-1].weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.delta_out.weight, mean=0.0, std=1e-3)
        boundaries = tuple(
            int(value)
            for value in config.flow_jepa_effective_interval_boundaries
        )
        self.horizon_labels = tuple(
            int(value)
            for value in config.flow_jepa_effective_window_offsets
        )
        if len(self.horizon_labels) != self.anchors:
            raise ValueError(
                "interval-stage labels must align with future anchors"
            )
        durations = tuple(
            boundaries[index + 1] - boundaries[index]
            for index in range(self.anchors)
        )
        self.register_buffer(
            "interval_encoding",
            _sinusoidal_offsets(durations, self.hidden)[None],
            persistent=False,
        )

    def forward(
        self,
        rollout_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if rollout_tokens.ndim != 3:
            raise ValueError("interval-stage rollout must be [B,N,H]")
        batch, token_count, hidden = rollout_tokens.shape
        expected = self.anchors * self.cameras * self.grid * self.grid
        if token_count != expected or hidden != self.hidden:
            raise ValueError(
                "interval-stage rollout must preserve "
                f"[anchor={self.anchors},camera={self.cameras},"
                f"grid={self.grid}x{self.grid},hidden={self.hidden}]"
            )
        grouped = rollout_tokens.reshape(
            batch,
            self.anchors,
            self.cameras,
            self.grid,
            self.grid,
            self.hidden,
        )
        temporal = grouped.permute(0, 2, 3, 4, 1, 5).reshape(
            batch * self.cameras * self.grid * self.grid,
            self.anchors,
            self.hidden,
        )
        normalized, denominator = variance_floored_centered_norm(
            temporal,
            self.normalization_floor,
        )
        temporal_input = normalized + self.interval_encoding.to(
            device=normalized.device,
            dtype=normalized.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                self.anchors,
                self.anchors,
                device=temporal.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        attention, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        stage_features = attention + self.temporal_ffn(attention)
        raw_delta = self.delta_out(stage_features)
        bounded_delta, delta_scale = smooth_rms_contract(raw_delta, 0.50)
        refined = temporal + self.update_scale * bounded_delta
        # The supervised progression is the exact bounded value written into
        # the action path (up to the fixed update scale).  A separate learned
        # readout could fit the teacher while the real W->P write stayed near
        # zero, recreating the representation-without-action shortcut.
        progress_prediction = bounded_delta

        refined_grouped = refined.reshape(
            batch,
            self.cameras,
            self.grid,
            self.grid,
            self.anchors,
            self.hidden,
        ).permute(0, 4, 1, 2, 3, 5)
        progress_grouped = progress_prediction.reshape(
            batch,
            self.cameras,
            self.grid,
            self.grid,
            self.anchors,
            self.hidden,
        ).permute(0, 4, 1, 2, 3, 5)
        written = self.update_scale * bounded_delta.detach().float()
        carrier_rms = temporal.detach().float().square().mean().sqrt()
        metrics: dict[str, Tensor] = {
            "flow_jepa_interval_stage_active": rollout_tokens.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_interval_stage_raw_delta_rms": (
                raw_delta.detach().float().square().mean().sqrt()
            ),
            "flow_jepa_interval_stage_written_delta_rms": (
                written.square().mean().sqrt()
            ),
            "flow_jepa_interval_stage_carrier_ratio": (
                written.square().mean().sqrt() / carrier_rms.clamp_min(1e-8)
            ),
            "flow_jepa_interval_stage_contraction": (
                1.0 - delta_scale.detach().float()
            ).mean(),
            "flow_jepa_interval_stage_norm_denominator_min": (
                denominator.detach().float().amin()
            ),
        }
        bounded_grouped = bounded_delta.detach().float().reshape(
            batch,
            self.cameras,
            self.grid,
            self.grid,
            self.anchors,
            self.hidden,
        )
        for index, label in enumerate(self.horizon_labels):
            metrics[
                f"flow_jepa_interval_stage_horizon_{label}_write_rms"
            ] = (
                self.update_scale
                * bounded_grouped[..., index, :].square().mean().sqrt()
            )
        return (
            refined_grouped.reshape_as(rollout_tokens),
            progress_grouped.reshape_as(rollout_tokens),
            metrics,
        )


class FlowDINOEvidenceEncoder(nn.Module):
    """Compile DINO content and learned patch motion into typed K/V evidence."""

    MOTION_DIM = 10

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        # Evaluation-only causal probe state.  This is deliberately a plain
        # Python attribute: it is not a parameter, buffer, config field, or
        # checkpoint key and therefore cannot change the trained V98 contract.
        self._raw_address_eval_intervention: str | None = None
        self._raw_address_eval_metrics: dict[str, Tensor] = {}
        self.late_bottleneck = bool(int(getattr(config, "flow_jepa_late_bottleneck", 0)))
        self.raw_enabled = bool(int(getattr(config, "flow_jepa_raw_image_enabled", 0)))
        self.soft_address_lattice_enabled = bool(
            int(getattr(config, "flow_jepa_soft_address_lattice", 0))
        )
        self.predictive_change_contract = bool(
            int(getattr(config, "flow_jepa_predictive_change_contract", 0))
        )
        self.sequential_horizon_memory = bool(
            int(getattr(config, "flow_jepa_sequential_horizon_memory", 0))
        )
        self.horizon_soft_address_enabled = bool(
            int(getattr(config, "flow_jepa_horizon_soft_address", 0))
        )
        self.progressive_grounding_address_enabled = bool(
            int(getattr(config, "flow_jepa_progressive_grounding_address", 0))
        )
        self.pre_value_owner_routing_enabled = bool(
            int(getattr(config, "flow_jepa_pre_value_owner_routing", 0))
        )
        self.functional_mainline_routing_enabled = bool(
            int(getattr(config, "flow_jepa_functional_mainline_routing", 0))
        )
        self.g_aligned_future_effect_enabled = bool(
            int(getattr(config, "flow_jepa_g_aligned_future_effect", 0))
        )
        self.supervised_effect_mainline_enabled = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_supervised_effect_mainline",
                    0,
                )
            )
        )
        self.window_effect_bank_enabled = bool(
            int(getattr(config, "flow_jepa_window_effect_bank", 0))
        )
        self.differential_intent_effect_mainline_enabled = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_differential_intent_effect_mainline",
                    0,
                )
            )
        )
        self.grounded_intent_effect_mainline_enabled = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_grounded_intent_effect_mainline",
                    0,
                )
            )
        )
        self.object_intent_dynamics_mainline_enabled = bool(
            int(
                getattr(
                    config,
                    "flow_jepa_object_intent_dynamics_mainline",
                    0,
                )
            )
        )
        self.exports_grounded_facts_enabled = bool(
            self.grounded_intent_effect_mainline_enabled
            or self.object_intent_dynamics_mainline_enabled
        )
        self.teacher_g_ema_decay = float(
            getattr(config, "flow_jepa_teacher_g_ema_decay", 0.995)
        )
        self._teacher_g_build_count = 0
        self.online_horizon_address_enabled = bool(
            int(getattr(config, "flow_jepa_online_horizon_address", 0))
        ) and not self.progressive_grounding_address_enabled
        self.interval_stage_enabled = bool(
            int(getattr(config, "flow_jepa_interval_stage_delta", 0))
        )
        self.complete_numerical_contract = bool(
            int(getattr(config, "flow_jepa_complete_numerical_contract", 0))
        )
        self.correlation_rms_floor = float(
            getattr(config, "flow_jepa_correlation_rms_floor", 0.10)
        )
        self.visibility_transition_fraction = float(
            getattr(config, "flow_jepa_visibility_transition_fraction", 0.10)
        )
        self.grid_size = int(config.flow_jepa_grid_size)
        self.history = int(config.visual_history_length)
        self.cameras = int(config.num_cameras)
        self.hidden = int(config.hidden_size)
        self.mask_ratio = float(config.flow_jepa_mask_ratio)
        self.mask_block = int(config.flow_jepa_mask_block_size)
        self.motion_mask_fraction = float(config.flow_jepa_motion_mask_fraction)
        self.history_offsets = tuple(int(value) for value in config.flow_jepa_history_offsets)
        self.window_offsets = tuple(
            int(value) for value in config.flow_jepa_effective_window_offsets
        )
        self.stage_offset = int(config.flow_jepa_effective_stage_offset)
        self.stage_target_scale = math.sqrt(
            float(self.cameras * self.grid_size * self.grid_size)
        )
        self.flow = LatentSeaRaft(
            config,
            # Uniform or weak correspondence must mean identity for both the
            # semantic-only lane and the raw-refinement lane. Previously V96
            # alone interpreted a uniform match as motion to image centre.
            identity_centered_initialization=True,
        )
        dino_dim = int(config.visual_token_dim)
        h = self.hidden

        def affine_evidence_norm(width: int) -> nn.Module:
            if self.complete_numerical_contract:
                return AffineVarianceFlooredCenteredNorm(
                    width,
                    float(
                        getattr(
                            config, "flow_jepa_routing_norm_floor", 0.25
                        )
                    ),
                    affine_maximum=4.0,
                )
            return nn.LayerNorm(width)

        self.content_key = nn.Sequential(
            affine_evidence_norm(dino_dim), nn.Linear(dino_dim, h)
        )
        self.content_value = nn.Sequential(
            affine_evidence_norm(dino_dim), nn.Linear(dino_dim, h)
        )
        self.motion_key = nn.Sequential(
            affine_evidence_norm(self.MOTION_DIM),
            nn.Linear(self.MOTION_DIM, h),
        )
        self.motion_value = nn.Sequential(
            affine_evidence_norm(self.MOTION_DIM),
            nn.Linear(self.MOTION_DIM, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )
        self.warp_key = nn.Sequential(affine_evidence_norm(h), nn.Linear(h, h))
        self.warp_value = nn.Sequential(
            affine_evidence_norm(2 * h), nn.Linear(2 * h, h)
        )
        if self.late_bottleneck:
            dense_dim = int(config.flow_jepa_feature_dim)
            self.coarse_organizer = _DenseDINOOrganizer(
                dino_dim,
                dense_dim,
                depth=int(config.flow_jepa_dense_depth),
                heads=8,
            )
            self.organized_key = nn.Sequential(
                affine_evidence_norm(dense_dim), nn.Linear(dense_dim, h)
            )
            self.organized_value = nn.Sequential(
                affine_evidence_norm(dense_dim),
                nn.Linear(dense_dim, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
            if self.raw_enabled:
                self.sparse_fine_flow = None
                self.address_reader = None
                self.detail_router = None
                self.raw_flow = _RawPyramidFlow(config)
                self.early_masked_raw_context = (
                    _EarlyMaskedRawContextEncoder(
                        int(config.flow_jepa_feature_dim),
                        h,
                        self.grid_size,
                        activation_checkpoint=bool(
                            int(config.flow_jepa_raw_activation_checkpoint)
                        ),
                    )
                    if self.predictive_change_contract
                    else None
                )
                self.raw_address_reader = _RawDeformableAddressReader(
                    self.raw_flow.pyramid.high_channels,
                    h,
                    self.grid_size,
                    radius=int(config.flow_jepa_raw_reader_radius),
                    heads=int(config.flow_jepa_raw_reader_heads),
                    nonduplicate_fallback=bool(
                        int(getattr(config, "flow_jepa_zero_flow_guard", 0))
                    ),
                    complementary_detail=bool(
                        int(getattr(config, "flow_jepa_complementary_raw_detail", 0))
                    ),
                )
                self.soft_address_compiler = (
                    _SoftMultiResolutionAddressCompiler(
                        config,
                        raw_dim=self.raw_flow.pyramid.high_channels,
                    )
                    if self.soft_address_lattice_enabled
                    else None
                )
                detail_hidden = max(int(config.flow_jepa_feature_dim) // 2, 32)
                self.raw_detail_query = nn.Sequential(
                    nn.LayerNorm(h), nn.Linear(h, detail_hidden), nn.SiLU(),
                    nn.Linear(detail_hidden, 1),
                )
                self.raw_detail_motion = nn.Sequential(
                    nn.LayerNorm(5), nn.Linear(5, detail_hidden), nn.SiLU(),
                    nn.Linear(detail_hidden, 1),
                )
                for router in (self.raw_detail_query, self.raw_detail_motion):
                    nn.init.normal_(router[-1].weight, mean=0.0, std=1e-3)
                    nn.init.constant_(router[-1].bias, -0.75)
                if bool(
                    int(getattr(config, "flow_jepa_late_policy_detail", 0))
                ):
                    # V102 deliberately removes action-conditioned grounding
                    # from early raw compilation. Keep legacy parameters in the
                    # state dict, but do not allocate optimizer state for two
                    # controls that are mathematically absent from this graph.
                    self.raw_detail_query.requires_grad_(False)
                    self.raw_address_reader.grounding_scale.requires_grad_(False)
                    # These are retained only for state-dict compatibility.
                    # The complementary reader uses its dedicated base/detail
                    # projections and has no fallback lane or learned flow
                    # prior, so the legacy projections are absent from the
                    # V102 graph.
                    self.raw_address_reader.key_proj.requires_grad_(False)
                    self.raw_address_reader.value_proj.requires_grad_(False)
                    self.raw_address_reader.flow_prior_strength.requires_grad_(False)
                    # Raw-grounding uses the dense organizer and motion bank;
                    # the semantic-only content/warp projections belong to the
                    # non-raw encoder path and must not appear trainable in the
                    # V102 optimizer contract.
                    for legacy_module in (
                        self.content_key,
                        self.content_value,
                        self.warp_key,
                        self.warp_value,
                    ):
                        legacy_module.requires_grad_(False)
                    if self.soft_address_lattice_enabled:
                        # The lattice replaces the compressed raw reader, not
                        # its checkpoint layout. Keep dormant modules for
                        # flags-off compatibility without optimizer ownership.
                        self.raw_address_reader.requires_grad_(False)
                        self.raw_detail_query.requires_grad_(False)
                        self.raw_detail_motion.requires_grad_(False)
                self.raw_evidence_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
                self.raw_mask_token = nn.Parameter(
                    torch.randn(1, self.raw_flow.pyramid.high_channels, 1, 1) * 0.02
                )
            else:
                self.early_masked_raw_context = None
                self.sparse_fine_flow = _SparseFineFlowRefiner(
                    dino_dim,
                    int(config.flow_jepa_feature_dim),
                    radius=int(config.flow_jepa_fine_radius),
                    grid=self.grid_size,
                    uncertainty_floor=float(config.flow_jepa_uncertainty_floor),
                )
                self.address_reader = _SoftFlowAddressReader(
                    h,
                    self.grid_size,
                    radius=int(config.flow_jepa_reader_radius),
                    heads=int(config.flow_jepa_reader_heads),
                )
                detail_hidden = max(int(config.flow_jepa_feature_dim) // 2, 16)
                self.detail_router = nn.Sequential(
                    nn.LayerNorm(5),
                    nn.Linear(5, detail_hidden),
                    nn.SiLU(),
                    nn.Linear(detail_hidden, 1),
                )
                nn.init.normal_(self.detail_router[-1].weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.detail_router[-1].bias, -1.734601055)
                self.raw_flow = None
                self.raw_address_reader = None
                self.raw_detail_query = None
                self.raw_detail_motion = None
                self.soft_address_compiler = None
                self.register_parameter("raw_evidence_type", None)
                self.register_parameter("raw_mask_token", None)
            self.detail_gate_floor_logit = nn.Parameter(torch.tensor(-1.5))
            if self.soft_address_lattice_enabled:
                self.detail_gate_floor_logit.requires_grad_(False)
                if self.raw_evidence_type is not None:
                    self.raw_evidence_type.requires_grad_(False)
        else:
            self.early_masked_raw_context = None
            self.coarse_organizer = None
            self.organized_key = None
            self.organized_value = None
            self.sparse_fine_flow = None
            self.address_reader = None
            self.detail_router = None
            self.raw_flow = None
            self.raw_address_reader = None
            self.raw_detail_query = None
            self.raw_detail_motion = None
            self.soft_address_compiler = None
            self.register_parameter("raw_evidence_type", None)
            self.register_parameter("raw_mask_token", None)
            self.register_parameter("detail_gate_floor_logit", None)
        self.context_mask_token = nn.Parameter(torch.randn(1, 1, 1, 1, 1, h) * 0.02)
        if self.raw_enabled:
            # Raw Flow-JEPA owns masking through ``raw_mask_token`` and the
            # early masked raw encoder. ``context_mask_token`` belongs to the
            # semantic-only encoder path; raw grounding may still materialize
            # compatibility intermediates that mention it, but none enter the
            # returned evidence bank or an active objective.
            self.context_mask_token.requires_grad_(False)
        self.history_type = nn.Parameter(torch.randn(1, self.history, 1, 1, 1, h) * 0.02)
        self.camera_type = nn.Parameter(torch.randn(1, 1, self.cameras, 1, 1, h) * 0.02)
        self.spatial_type = nn.Parameter(
            torch.randn(1, 1, 1, self.grid_size, self.grid_size, h) * 0.02
        )
        self.evidence_type = nn.Parameter(torch.randn(1, 3, h) * 0.02)
        self.future_query = nn.Parameter(
            torch.randn(
                1,
                int(config.future_anchors),
                self.cameras,
                self.grid_size,
                self.grid_size,
                h,
            )
            * 0.02
        )
        self.future_anchor_type = nn.Parameter(
            torch.randn(1, int(config.future_anchors), 1, 1, 1, h) * 0.02
        )
        self.stage_query_token = (
            None if self.late_bottleneck else nn.Parameter(torch.randn(1, 1, h) * 0.02)
        )
        self.stage_type = (
            None if self.late_bottleneck else nn.Parameter(torch.randn(1, 1, h) * 0.02)
        )
        self.future_motion = nn.Sequential(
            affine_evidence_norm(self.MOTION_DIM),
            nn.Linear(self.MOTION_DIM, h),
        )
        if self.sequential_horizon_memory:
            history_hidden = max(h // 2, 32)
            self.future_history_score = nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, history_hidden),
                nn.SiLU(),
                nn.Linear(history_hidden, 1, bias=False),
            )
            self.future_memory_norm = (
                VarianceFlooredCenteredNorm(
                    float(
                        getattr(
                            config, "flow_jepa_routing_norm_floor", 0.25
                        )
                    )
                )
                if self.complete_numerical_contract
                else nn.LayerNorm(h, elementwise_affine=False)
            )
            self.future_transition = nn.Sequential(
                nn.Linear(3 * h, 2 * h),
                nn.SiLU(),
                nn.Linear(2 * h, h, bias=False),
            )
            nn.init.normal_(
                self.future_transition[-1].weight,
                mean=0.0,
                std=1e-3,
            )
        else:
            self.future_history_score = None
            self.future_memory_norm = None
            self.future_transition = None
        self.stage_motion = (
            None
            if self.late_bottleneck
            else nn.Sequential(
                affine_evidence_norm(self.MOTION_DIM),
                nn.Linear(self.MOTION_DIM, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
        )
        self.stage_motion_scale = (
            None
            if self.late_bottleneck
            else nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        )
        self.future_prediction = nn.Sequential(
            (
                VarianceFlooredCenteredNorm(
                    float(
                        getattr(
                            config,
                            "flow_jepa_routing_norm_floor",
                            0.25,
                        )
                    )
                )
                if bool(
                    int(
                        getattr(
                            config,
                            "flow_jepa_variance_safe_routing",
                            0,
                        )
                    )
                )
                else nn.LayerNorm(h)
            ),
            nn.Linear(h, h),
        )
        if self.g_aligned_future_effect_enabled:
            # V115 exposes the JEPA delta from the same FutureEffectField that
            # P consumes.  The ancestral rollout-side decoder is retained for
            # state-dict ancestry but cannot remain a trainable sidecar.
            self.future_prediction.requires_grad_(False)
        self.horizon_address_jepa = (
            _HorizonSoftAddressJEPA(
                config,
                raw_dim=self.raw_flow.pyramid.high_channels,
            )
            if self.horizon_soft_address_enabled
            and self.raw_flow is not None
            and self.soft_address_compiler is not None
            and not self.progressive_grounding_address_enabled
            else None
        )
        self.progressive_grounding_address = (
            _ProgressiveGroundingAddressOrganizer(config)
            if self.progressive_grounding_address_enabled
            else None
        )
        self.interval_stage_organizer = (
            _IntervalStageDeltaOrganizer(config)
            if self.interval_stage_enabled
            else None
        )
        if (
            self.functional_mainline_routing_enabled
            and self.interval_stage_organizer is not None
        ):
            # V113 supervises the interval candidate already consumed by the
            # online W owner router.  Retain the V106-V112 module only for
            # checkpoint ancestry; a trainable post-W3 organizer would restore
            # the representation-without-action shortcut.
            self.interval_stage_organizer.requires_grad_(False)
        self.stage_prediction = (
            None
            if self.late_bottleneck
            else nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h))
        )
        self.register_buffer(
            "future_horizon_encoding",
            _sinusoidal_offsets(self.window_offsets, h)[None, :, None, None, None],
            persistent=False,
        )
        incremental_offsets = (
            self.window_offsets[0],
            *(
                self.window_offsets[index] - self.window_offsets[index - 1]
                for index in range(1, len(self.window_offsets))
            ),
        )
        self.register_buffer(
            "future_step_encoding",
            _sinusoidal_offsets(incremental_offsets, h)[
                None, :, None, None, None
            ],
            persistent=False,
        )
        self.register_buffer(
            "future_history_encoding",
            _sinusoidal_offsets(self.history_offsets[1:], h)[
                None, :, None, None, None
            ],
            persistent=False,
        )
        if self.late_bottleneck:
            self.register_buffer("stage_horizon_encoding", torch.empty(1, 0, h), persistent=False)
            self.register_buffer(
                "stage_teacher_position_code", torch.empty(1, 0, h), persistent=False
            )
        else:
            self.register_buffer(
                "stage_horizon_encoding",
                _sinusoidal_offsets((self.stage_offset,), h)[None],
                persistent=False,
            )
            stage_position_code = _sinusoidal_offsets(
                tuple(range(1, self.cameras * 4 + 1)), h
            )
            stage_position_code = stage_position_code - stage_position_code.mean(
                dim=0, keepdim=True
            )
            stage_position_code = stage_position_code / stage_position_code.square().mean(
                dim=0, keepdim=True
            ).sqrt().clamp_min(1e-3)
            self.register_buffer(
                "stage_teacher_position_code",
                stage_position_code.reshape(1, 1, self.cameras, 2, 2, h),
                persistent=False,
            )
        self.teacher_norm = nn.LayerNorm(dino_dim, elementwise_affine=False)
        self.teacher_projection = nn.Linear(dino_dim, h, bias=False)
        nn.init.orthogonal_(self.teacher_projection.weight)
        self.teacher_projection.weight.requires_grad_(False)
        self.teacher_norm.requires_grad_(False)
        if self.g_aligned_future_effect_enabled:
            if self.soft_address_compiler is None:
                raise RuntimeError(
                    "G-aligned teacher requires the soft address compiler"
                )
            self.teacher_g_semantic_projection = copy.deepcopy(
                self.soft_address_compiler.target_dino_key
            )
            self.teacher_g_semantic_projection.requires_grad_(False)
        else:
            self.teacher_g_semantic_projection = None

    def set_raw_address_eval_intervention(self, mode: str) -> None:
        """Temporarily intervene on address experts or protected raw values."""

        normalized = str(mode).strip().lower()
        if normalized not in {
            "none",
            "zero",
            "shuffle",
            "spatial_shuffle",
            "detail_zero",
            "detail_spatial_shuffle",
            "literal_rgb_zero",
            "literal_rgb_spatial_shuffle",
            "dino_key_spatial_shuffle",
            "source_raw_key_zero",
            "source_raw_key_spatial_shuffle",
            "joint_address_key_spatial_shuffle",
            "current_context_masked",
        }:
            raise ValueError(
                "raw intervention must be none/zero/shuffle/spatial_shuffle/"
                "detail_zero/detail_spatial_shuffle/dino_key_spatial_shuffle/"
                "literal_rgb_zero/literal_rgb_spatial_shuffle/"
                "source_raw_key_zero/source_raw_key_spatial_shuffle/"
                "joint_address_key_spatial_shuffle/current_context_masked"
            )
        if self.training:
            raise RuntimeError("raw address intervention is evaluation-only")
        if not self.raw_enabled or getattr(self, "raw_address_reader", None) is None:
            raise RuntimeError("raw address intervention requires the V98 raw reader")
        if normalized in {
            "dino_key_spatial_shuffle",
            "source_raw_key_zero",
            "source_raw_key_spatial_shuffle",
            "joint_address_key_spatial_shuffle",
        } and not self.soft_address_lattice_enabled:
            raise RuntimeError(
                "key-level address intervention requires the soft address lattice"
            )
        if normalized == "current_context_masked" and (
            not self.predictive_change_contract
            or not self.soft_address_lattice_enabled
        ):
            raise RuntimeError(
                "current-context mask intervention requires predictive JEPA "
                "and the soft address lattice"
            )
        self._raw_address_eval_intervention = normalized
        self._raw_address_eval_metrics = {}

    def clear_raw_address_eval_intervention(self) -> None:
        self._raw_address_eval_intervention = None
        self._raw_address_eval_metrics = {}

    def raw_address_eval_metrics(self) -> dict[str, float]:
        return {
            key: float(value.detach().float().cpu())
            for key, value in self._raw_address_eval_metrics.items()
        }

    def _intervened_raw_address_flow(
        self,
        flow: Tensor,
        *,
        batch: int,
        mode: str,
    ) -> tuple[Tensor, bool]:
        """Return reader coordinates while keeping camera identity intact."""

        if mode == "none":
            return flow, False
        if mode == "zero":
            return torch.zeros_like(flow), False
        if mode == "spatial_shuffle":
            shaped = flow.reshape(batch, self.cameras, 2, *flow.shape[-2:])
            shifts = (
                max(int(flow.shape[-2]) // 2, 1),
                max(int(flow.shape[-1]) // 2, 1),
            )
            # Always invalidate spatial addresses, including when an ordered
            # validation batch contains only adjacent windows from one episode.
            return shaped.roll(shifts=shifts, dims=(-2, -1)).reshape_as(flow), False
        if mode != "shuffle":
            raise ValueError(f"unknown raw address intervention: {mode!r}")
        shaped = flow.reshape(batch, self.cameras, 2, *flow.shape[-2:])
        if batch > 1:
            # Roll samples, not the flattened B*C axis: top remains top and
            # wrist remains wrist.
            return shaped.roll(1, dims=0).reshape_as(flow), False
        # A single-sample batch has no cross-sample donor.  Preserve camera
        # identity and deterministically misalign the spatial flow field.
        return shaped.roll(shifts=(1, 1), dims=(-2, -1)).reshape_as(flow), True

    def _pool(self, visual: Tensor) -> Tensor:
        if visual.ndim != 5:
            raise ValueError("visual must be [B,H,C,P,D]")
        batch, history, cameras, patches, dim = visual.shape
        if (history, cameras) != (self.history, self.cameras):
            raise ValueError("visual history/camera geometry does not match Flow-DINO config")
        side = int(round(float(patches) ** 0.5))
        if side * side != int(patches):
            raise ValueError("Flow-DINO requires a square DINO patch grid")
        grid = self.grid_size
        value = visual.reshape(batch * history * cameras, side, side, dim)
        value = F.adaptive_avg_pool2d(value.permute(0, 3, 1, 2).float(), (grid, grid))
        return value.permute(0, 2, 3, 1).reshape(batch, history, cameras, grid, grid, dim)

    def _structured_mask(self, score: Tensor, *, stochastic: bool) -> Tensor:
        """Return an exact-quota block-biased mask using detached saliency."""

        if score.ndim != 4:
            raise ValueError("mask score must be [B,T,G,G]")
        batch, rows, height, width = score.shape
        count = int(round(self.mask_ratio * height * width))
        if count <= 0:
            return torch.zeros_like(score, dtype=torch.bool)
        random = (
            torch.rand_like(score.float())
            if stochastic
            else torch.zeros_like(score, dtype=torch.float32)
        )
        saliency = score.detach().float()
        flat = saliency.flatten(-2)
        lo = flat.amin(dim=-1, keepdim=True)
        hi = flat.amax(dim=-1, keepdim=True)
        saliency = ((flat - lo) / (hi - lo).clamp_min(1e-6)).reshape_as(saliency)
        mixed = self.motion_mask_fraction * saliency + (1.0 - self.motion_mask_fraction) * random
        if self.mask_block > 1:
            kernel = min(self.mask_block, height, width)
            pad_before = (kernel - 1) // 2
            pad_after = kernel // 2
            block_input = F.pad(
                mixed.reshape(batch * rows, 1, height, width),
                (pad_before, pad_after, pad_before, pad_after),
                mode="replicate",
            )
            mixed = F.avg_pool2d(
                block_input,
                kernel_size=kernel,
                stride=1,
            ).reshape(batch, rows, height, width)
        selected = mixed.flatten(-2).topk(k=count, dim=-1).indices
        mask = torch.zeros(batch, rows, height * width, device=score.device, dtype=torch.bool)
        mask.scatter_(-1, selected, True)
        return mask.reshape(batch, rows, height, width)

    def _compose_future_queries(
        self,
        identity_queries: Tensor,
        motion_history: Tensor,
        context_seed: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Build future anchors from observation-only perceptual history.

        The legacy path predicts all horizons in parallel from the latest
        frame pair.  The sequential contract instead forms one externally
        stateless memory inside this forward call:

        ``observed history -> M0 -> M+4 -> M+12 -> M+24 -> M+48``.

        No future teacher, target mask content, action target, detach, hard
        route, or recurrent state survives beyond the call.
        """

        if identity_queries.ndim != 6:
            raise ValueError(
                "future identity queries must be [B,A,C,G,G,H]"
            )
        if motion_history.ndim != 6 or int(motion_history.shape[-1]) != self.MOTION_DIM:
            raise ValueError(
                "future motion history must be [B,pair,C,G,G,M]"
            )
        if context_seed.ndim != 6 or int(context_seed.shape[1]) != 1:
            raise ValueError(
                "future context seed must be [B,1,C,G,G,H]"
            )
        batch, anchors, cameras, grid, grid_b, hidden = identity_queries.shape
        if (
            anchors != len(self.window_offsets)
            or cameras != self.cameras
            or grid != self.grid_size
            or grid_b != self.grid_size
            or hidden != self.hidden
        ):
            raise ValueError("future identity queries do not match encoder geometry")
        if tuple(motion_history.shape[:2]) != (
            batch,
            len(self.history_offsets) - 1,
        ) or tuple(motion_history.shape[2:5]) != (
            cameras,
            grid,
            grid,
        ):
            raise ValueError("future motion history does not match observed history")
        if tuple(context_seed.shape) != (
            batch,
            1,
            cameras,
            grid,
            grid,
            hidden,
        ):
            raise ValueError("future context seed does not match future geometry")

        if not self.sequential_horizon_memory:
            latest_motion = motion_history[:, -1]
            latest_dt = float(
                self.history_offsets[-1] - self.history_offsets[-2]
            )
            anchor_scale = torch.as_tensor(
                [
                    math.sqrt(float(offset) / latest_dt)
                    for offset in self.window_offsets
                ],
                device=identity_queries.device,
                dtype=identity_queries.dtype,
            )[None, :, None, None, None, None]
            queries = (
                identity_queries
                + anchor_scale
                * self.future_motion(
                    latest_motion.to(dtype=identity_queries.dtype)
                )[:, None]
                + 0.10 * context_seed
            )
            zero = queries.new_zeros((), dtype=torch.float32)
            return queries, {
                "flow_jepa_sequential_horizon_memory": zero,
                "flow_jepa_perceptual_history_entropy": zero,
                "flow_jepa_perceptual_history_latest_mass": queries.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_horizon_transition_update_rms": zero,
                "flow_jepa_horizon_transition_state_delta": zero,
            }

        if (
            self.future_history_score is None
            or self.future_memory_norm is None
            or self.future_transition is None
        ):
            raise RuntimeError("sequential horizon memory modules are missing")
        motion_features = self.future_motion(
            motion_history.to(dtype=identity_queries.dtype)
        )
        motion_features = motion_features + self.future_history_encoding.to(
            device=motion_features.device,
            dtype=motion_features.dtype,
        )
        history_logits = self.future_history_score(motion_features).float()
        history_weights = torch.softmax(history_logits, dim=1).to(
            dtype=motion_features.dtype
        )
        history_state = (history_weights * motion_features).sum(dim=1)
        memory = self.future_memory_norm(
            context_seed[:, 0] + history_state
        )

        step_condition = identity_queries + (
            self.future_step_encoding - self.future_horizon_encoding
        ).to(device=identity_queries.device, dtype=identity_queries.dtype)
        outputs: list[Tensor] = []
        update_rows: list[Tensor] = []
        state_delta_rows: list[Tensor] = []
        for index in range(anchors):
            condition = step_condition[:, index]
            transition_input = torch.cat(
                (
                    self.future_memory_norm(memory),
                    self.future_memory_norm(history_state),
                    self.future_memory_norm(condition),
                ),
                dim=-1,
            )
            raw_update = self.future_transition(transition_input)
            update, _ = smooth_rms_contract(raw_update, 0.50)
            next_memory = self.future_memory_norm(
                memory + update + 0.10 * condition
            )
            outputs.append(next_memory + identity_queries[:, index])
            update_rows.append(
                update.detach().float().square().mean(dim=-1).sqrt().mean()
            )
            state_delta_rows.append(
                (next_memory.detach().float() - memory.detach().float())
                .square()
                .mean(dim=-1)
                .sqrt()
                .mean()
            )
            memory = next_memory

        queries = torch.stack(outputs, dim=1)
        history_probability = history_weights.detach().float()[..., 0]
        if int(history_probability.shape[1]) > 1:
            history_entropy = -(
                history_probability
                * history_probability.clamp_min(1e-8).log()
            ).sum(dim=1) / math.log(float(history_probability.shape[1]))
            history_entropy = history_entropy.mean()
        else:
            history_entropy = queries.new_zeros((), dtype=torch.float32)
        metrics: dict[str, Tensor] = {
            "flow_jepa_sequential_horizon_memory": queries.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_perceptual_history_entropy": history_entropy,
            "flow_jepa_perceptual_history_latest_mass": (
                history_probability[:, -1].mean()
            ),
            "flow_jepa_horizon_transition_update_rms": torch.stack(
                update_rows
            ).mean(),
            "flow_jepa_horizon_transition_state_delta": torch.stack(
                state_delta_rows
            ).mean(),
        }
        for index, offset in enumerate(self.window_offsets):
            metrics[
                f"flow_jepa_horizon_transition_update_{offset}_rms"
            ] = update_rows[index]
            metrics[
                f"flow_jepa_horizon_transition_{offset}_state_delta"
            ] = state_delta_rows[index]
        if anchors > 1:
            metrics["flow_jepa_future_query_adjacent_cosine"] = (
                F.cosine_similarity(
                    queries[:, 1:].detach().float(),
                    queries[:, :-1].detach().float(),
                    dim=-1,
                ).mean()
            )
        return queries, metrics

    @staticmethod
    def _smoothness(flow: Tensor, feature: Tensor) -> Tensor:
        dx = flow[..., :, 1:] - flow[..., :, :-1]
        dy = flow[..., 1:, :] - flow[..., :-1, :]
        fx = feature[..., :, 1:] - feature[..., :, :-1]
        fy = feature[..., 1:, :] - feature[..., :-1, :]
        wx = torch.exp(-fx.float().abs().mean(dim=-3, keepdim=True))
        wy = torch.exp(-fy.float().abs().mean(dim=-3, keepdim=True))
        return (dx.float().abs() * wx).mean() + (dy.float().abs() * wy).mean()

    def _estimate_pairs(
        self, pooled: Tensor
    ) -> tuple[PatchFlowEstimate, PatchFlowEstimate, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if self.flow is None:
            raise RuntimeError("DINO patch flow is disabled by the raw-image flow contract")
        batch, history, cameras, grid, _, dim = pooled.shape
        first = pooled[:, :-1].reshape(batch * (history - 1) * cameras, grid, grid, dim)
        second = pooled[:, 1:].reshape(batch * (history - 1) * cameras, grid, grid, dim)
        joined_first = torch.cat((first, second), dim=0)
        joined_second = torch.cat((second, first), dim=0)
        estimate = self.flow(joined_first, joined_second)
        pair_batch = int(first.shape[0])

        def split(value: Tensor) -> tuple[Tensor, Tensor]:
            return value[:pair_batch], value[pair_batch:]

        forward_flow, backward_flow = split(estimate.flow)
        forward_info, backward_info = split(estimate.information)
        forward_uncertainty, backward_uncertainty = split(estimate.uncertainty)
        forward_entropy, backward_entropy = split(estimate.correlation_entropy)
        forward_margin, backward_margin = split(estimate.correlation_margin)
        forward_iterations = tuple(value[:pair_batch] for value in estimate.iterations)
        backward_iterations = tuple(value[pair_batch:] for value in estimate.iterations)
        forward = PatchFlowEstimate(
            forward_flow,
            forward_info,
            forward_uncertainty,
            forward_entropy,
            forward_margin,
            forward_iterations,
            correlation_feature_rms_min=estimate.correlation_feature_rms_min,
            correlation_norm_denominator_min=(
                estimate.correlation_norm_denominator_min
            ),
            correlation_norm_gain_max=estimate.correlation_norm_gain_max,
        )
        backward = PatchFlowEstimate(
            backward_flow,
            backward_info,
            backward_uncertainty,
            backward_entropy,
            backward_margin,
            backward_iterations,
            correlation_feature_rms_min=estimate.correlation_feature_rms_min,
            correlation_norm_denominator_min=(
                estimate.correlation_norm_denominator_min
            ),
            correlation_norm_gain_max=estimate.correlation_norm_gain_max,
        )
        warped_backward, forward_cycle_valid = warp_patch_grid(backward_flow, forward_flow)
        warped_forward, backward_cycle_valid = warp_patch_grid(forward_flow, backward_flow)
        forward_cycle = forward_flow + warped_backward
        backward_cycle = backward_flow + warped_forward
        consistency = _stable_vector_norm(forward_cycle, dim=1, keepdim=True)
        scale = (
            forward_flow.float().square().sum(dim=1, keepdim=True)
            + warped_backward.float().square().sum(dim=1, keepdim=True)
        )
        visibility_threshold = 0.01 * scale + 0.5
        if self.complete_numerical_contract:
            visible, _, _, _ = _continuous_cycle_visibility(
                forward_cycle_valid,
                consistency.square(),
                visibility_threshold,
                transition_fraction=self.visibility_transition_fraction,
            )
        else:
            visible = (
                forward_cycle_valid
                & (consistency.square() < visibility_threshold)
            ).float()
        confidence = (
            torch.exp(-forward_uncertainty.float())
            * torch.exp(-consistency)
            * forward_cycle_valid.float()
            * (1.0 - forward_entropy.float()).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        occlusion = 1.0 - visible

        first_feature = first.permute(0, 3, 1, 2).float()
        second_feature = second.permute(0, 3, 1, 2).float()
        warped_second, forward_warp_valid = warp_patch_grid(second_feature, forward_flow)
        warped_first, backward_warp_valid = warp_patch_grid(first_feature, backward_flow)

        def normalize_feature(value: Tensor) -> Tensor:
            if self.complete_numerical_contract:
                normalized, _ = rms_floored_l2_normalize(
                    value, self.correlation_rms_floor, dim=1
                )
                return normalized
            return F.normalize(value, dim=1)

        normalized_first = normalize_feature(first_feature)
        normalized_second = normalize_feature(second_feature)
        forward_warp_error = (
            normalized_first - normalize_feature(warped_second.float())
        ).square().mean(dim=1, keepdim=True)
        backward_warp_error = (
            normalized_second - normalize_feature(warped_first.float())
        ).square().mean(dim=1, keepdim=True)

        def paired_masked_mean(
            first_value: Tensor,
            first_valid: Tensor,
            second_value: Tensor,
            second_valid: Tensor,
        ) -> Tensor:
            first_weight = first_valid.float()
            second_weight = second_valid.float()
            numerator = (first_value * first_weight).sum() + (second_value * second_weight).sum()
            denominator = (first_weight.sum() + second_weight.sum()).clamp_min(1.0)
            return numerator / denominator

        warp_loss = paired_masked_mean(
            torch.sqrt(forward_warp_error + 1e-6),
            forward_warp_valid,
            torch.sqrt(backward_warp_error + 1e-6),
            backward_warp_valid,
        )
        consistency_loss = paired_masked_mean(
            torch.sqrt(forward_cycle.float().square().sum(dim=1, keepdim=True) + 1e-6),
            forward_cycle_valid,
            torch.sqrt(backward_cycle.float().square().sum(dim=1, keepdim=True) + 1e-6),
            backward_cycle_valid,
        )
        smoothness = 0.5 * (
            self._smoothness(forward_flow, first_feature)
            + self._smoothness(backward_flow, second_feature)
        )
        forward_nll = (
            forward_warp_error.detach().sqrt() / forward_uncertainty.clamp_min(1e-4)
            + forward_uncertainty.clamp_min(1e-4).log()
        )
        backward_nll = (
            backward_warp_error.detach().sqrt() / backward_uncertainty.clamp_min(1e-4)
            + backward_uncertainty.clamp_min(1e-4).log()
        )
        uncertainty_nll = paired_masked_mean(
            forward_nll,
            forward_warp_valid,
            backward_nll,
            backward_warp_valid,
        )

        # The final iteration already owns ``warp_loss``.  Sequence supervision
        # is intentionally restricted to intermediate estimates so the loss
        # ledger does not count the same terminal error twice under two names.
        sequence_rows: list[Tensor] = []
        for forward_row, backward_row in zip(
            forward_iterations[:-1], backward_iterations[:-1]
        ):
            forward_intermediate, forward_valid = warp_patch_grid(
                second_feature, forward_row
            )
            backward_intermediate, backward_valid = warp_patch_grid(
                first_feature, backward_row
            )
            forward_error = torch.sqrt(
                (
                    normalize_feature(forward_intermediate.float()) - normalized_first
                ).square().mean(dim=1, keepdim=True)
                + 1e-6
            )
            backward_error = torch.sqrt(
                (
                    normalize_feature(backward_intermediate.float()) - normalized_second
                ).square().mean(dim=1, keepdim=True)
                + 1e-6
            )
            sequence_rows.append(
                paired_masked_mean(
                    forward_error,
                    forward_valid,
                    backward_error,
                    backward_valid,
                )
            )
        if sequence_rows:
            sequence_loss = torch.stack(sequence_rows)
            decay = torch.linspace(
                0.5, 1.0, steps=len(sequence_rows), device=sequence_loss.device
            )
            sequence_loss = (sequence_loss * decay).sum() / decay.sum()
        else:
            sequence_loss = warp_loss * 0.0
        losses = {
            "flow_jepa_warp_loss": warp_loss,
            "flow_jepa_cycle_loss": consistency_loss,
            "flow_jepa_smoothness_loss": smoothness,
            "flow_jepa_uncertainty_nll": uncertainty_nll,
            "flow_jepa_refinement_sequence_loss": sequence_loss,
        }
        return forward, backward, confidence, occlusion, forward_warp_error, losses

    @staticmethod
    def _semantic_seed_reliability(estimate: PatchFlowEstimate) -> Tensor:
        """Continuous evidence that a DINO displacement is spatially identified.

        This is deliberately not a route gate. Legacy raw refinement used it
        to attenuate an uncertain seed; the V99 zero-flow guard instead uses
        it only to widen the local search while preserving the seed coordinate.
        In both cases the value remains continuous and differentiable.
        """

        correlation_certainty = (1.0 - estimate.correlation_entropy.float()).clamp(
            0.0, 1.0
        )
        scale_certainty = torch.exp(-estimate.uncertainty.float().clamp_min(0.0))
        # This square root acts directly on certainty rather than on a squared
        # vector magnitude. A wider linear neighbourhood bounds its slope at
        # 100 instead of recreating an effectively singular reliability gate.
        return _stable_sqrt(correlation_certainty, epsilon=1e-4) * scale_certainty

    def _forward_raw_grounding(
        self, visual: Tensor, raw_visual: Tensor
    ) -> FlowDINOEvidencePack:
        if any(
            module is None
            for module in (
                self.coarse_organizer,
                self.organized_key,
                self.organized_value,
                self.raw_flow,
                self.raw_address_reader,
                self.raw_detail_query,
                self.raw_detail_motion,
            )
        ):
            raise RuntimeError("raw-grounding Flow-DINO modules are incomplete")
        pooled = self._pool(visual)
        batch, history, cameras, grid, _, _ = pooled.shape
        if tuple(raw_visual.shape[:3]) != (batch, history, cameras):
            raise ValueError("raw RGB and DINO history/camera axes must align")
        (
            coarse_forward,
            coarse_backward,
            coarse_confidence,
            coarse_occlusion,
            _,
            coarse_losses,
        ) = self._estimate_pairs(pooled)
        coarse_reliability_forward = self._semantic_seed_reliability(coarse_forward)
        coarse_reliability_backward = self._semantic_seed_reliability(coarse_backward)
        raw_context, raw_losses, raw_metrics = self.raw_flow(
            raw_visual,
            coarse_forward.flow,
            coarse_backward.flow,
            coarse_reliability_forward,
            coarse_reliability_backward,
        )
        if (
            coarse_forward.correlation_feature_rms_min is not None
            and coarse_forward.correlation_norm_denominator_min is not None
            and coarse_forward.correlation_norm_gain_max is not None
        ):
            raw_metrics["flow_jepa_correlation_feature_rms_min"] = torch.minimum(
                raw_metrics["flow_jepa_correlation_feature_rms_min"],
                coarse_forward.correlation_feature_rms_min,
            ).detach()
            raw_metrics[
                "flow_jepa_correlation_norm_denominator_min"
            ] = torch.minimum(
                raw_metrics["flow_jepa_correlation_norm_denominator_min"],
                coarse_forward.correlation_norm_denominator_min,
            ).detach()
            raw_metrics["flow_jepa_correlation_norm_gain_max"] = torch.maximum(
                raw_metrics["flow_jepa_correlation_norm_gain_max"],
                coarse_forward.correlation_norm_gain_max,
            ).detach()
        # Both semantic and RGB geometry supervise the same physical object.
        # Averaging keeps every historical objective weight on its established
        # scale instead of silently doubling the representation budget.
        losses = {
            name: 0.5 * (coarse_losses[name] + raw_losses[name])
            for name in coarse_losses
        }
        for name, value in raw_losses.items():
            if name not in losses:
                losses[name] = value
        for name, value in coarse_losses.items():
            raw_metrics[f"{name}_semantic_component"] = value.detach()
            raw_metrics[f"{name}_raw_component"] = raw_losses[name].detach()
        raw_metrics.update(
            {
                "flow_jepa_semantic_seed_confidence": coarse_confidence.float().mean().detach(),
                "flow_jepa_semantic_seed_occlusion": coarse_occlusion.float().mean().detach(),
                "flow_jepa_semantic_seed_reliability": 0.5
                * (
                    coarse_reliability_forward.float().mean()
                    + coarse_reliability_backward.float().mean()
                ).detach(),
                "flow_jepa_multiscale_loss_blend": coarse_forward.flow.new_ones(
                    (), dtype=torch.float32
                ),
            }
        )
        pair_count = history - 1
        high_side = int(raw_context.flow_forward.shape[-1])

        def to_grid(value: Tensor) -> Tensor:
            channels = int(value.shape[3])
            flat = value.reshape(batch * pair_count * cameras, channels, high_side, high_side)
            flat = F.interpolate(
                flat.float(), size=(grid, grid), mode="bilinear", align_corners=True
            )
            return flat.reshape(batch, pair_count, cameras, channels, grid, grid)

        raw_flow_grid = to_grid(raw_context.flow_forward) * (
            float(max(grid - 1, 1)) / float(max(high_side - 1, 1))
        )
        confidence_grid = to_grid(raw_context.confidence)
        occlusion_grid = to_grid(raw_context.occlusion)
        entropy_grid = to_grid(raw_context.correlation_entropy)
        margin_grid = to_grid(raw_context.correlation_margin)
        cycle_grid = to_grid(raw_context.cycle_error) * (
            float(max(grid - 1, 1)) / float(max(high_side - 1, 1))
        )
        warp_error_grid = to_grid(raw_context.warp_error)
        observable_motion_grid = to_grid(raw_context.observable_motion).clamp(0.0, 1.0)
        magnitude = _stable_vector_norm(raw_flow_grid, dim=3)
        motion_score = torch.zeros(
            batch, history, cameras, grid, grid, device=visual.device, dtype=torch.float32
        )
        if bool(int(getattr(self.config, "flow_jepa_zero_flow_guard", 0))):
            observed_pair_motion = observable_motion_grid[:, :, :, 0]
            motion_score[:, 0] = observed_pair_motion[:, 0]
            motion_score[:, -1] = observed_pair_motion[:, -1]
            if history > 2:
                # An interior observation belongs to both its incoming and
                # outgoing pair. Keep whichever side contains more visible
                # change instead of assigning the preceding pair twice.
                motion_score[:, 1:-1] = torch.maximum(
                    observed_pair_motion[:, :-1], observed_pair_motion[:, 1:]
                )
        else:
            motion_score[:, 0] = magnitude[:, 0]
            motion_score[:, 1:] = magnitude
        future_motion_score = (
            observable_motion_grid[:, -1, :, 0]
            if bool(int(getattr(self.config, "flow_jepa_zero_flow_guard", 0)))
            else magnitude[:, -1]
        )
        if self.training:
            context_dropout = self._structured_mask(
                motion_score.reshape(batch, history * cameras, grid, grid), stochastic=True
            ).reshape(batch, history, cameras, grid, grid)
        else:
            context_dropout = torch.zeros_like(motion_score, dtype=torch.bool)
        if self.predictive_change_contract:
            # One observation-only spatial mask owns both sides of the JEPA
            # boundary.  Reusing it across real horizons keeps the latest
            # online RGB/DINO context blind to every supervised coordinate
            # during training without revealing any future-teacher-derived
            # mask pattern.  Evaluation/deployment keeps the observation
            # context complete; masking is a training objective, not an
            # inference-time information deletion.
            future_spatial_mask = self._structured_mask(
                future_motion_score,
                stochastic=self.training,
            )
            if self.training:
                context_dropout = context_dropout.clone()
                context_dropout[:, -1] = future_spatial_mask
            future_mask = future_spatial_mask[:, None].expand(
                -1,
                int(self.config.future_anchors),
                -1,
                -1,
                -1,
            )
            if self._raw_address_eval_intervention == "current_context_masked":
                if self.training:
                    raise RuntimeError(
                        "current-context mask intervention is evaluation-only"
                    )
                # Matched evaluation reuses the exact deterministic,
                # observation-derived target mask. Only the latest online
                # RGB/DINO context changes; model mode, weights, action noise,
                # target coordinates and all future teachers remain fixed.
                context_dropout = context_dropout.clone()
                context_dropout[:, -1] = future_spatial_mask
        else:
            if self._raw_address_eval_intervention == "current_context_masked":
                raise RuntimeError(
                    "current-context mask intervention requires predictive JEPA"
                )
            future_score = future_motion_score[:, None].expand(
                -1, int(self.config.future_anchors), -1, -1, -1
            )
            future_mask = self._structured_mask(
                future_score.reshape(
                    batch,
                    int(self.config.future_anchors) * cameras,
                    grid,
                    grid,
                ),
                stochastic=self.training,
            ).reshape(
                batch,
                int(self.config.future_anchors),
                cameras,
                grid,
                grid,
            )

        early_raw_context = None
        if self.predictive_change_contract:
            if self.early_masked_raw_context is None:
                raise RuntimeError(
                    "predictive-change Flow-JEPA is missing its early masked RGB encoder"
                )
            early_raw_context = self.early_masked_raw_context(
                raw_visual, context_dropout
            )
        organized = self.coarse_organizer(pooled, context_dropout)
        content_selector = self.organized_key(organized)
        content_values = self.organized_value(organized)
        identity = (
            self.history_type.to(device=visual.device, dtype=content_selector.dtype)
            + self.camera_type.to(device=visual.device, dtype=content_selector.dtype)
            + self.spatial_type.to(device=visual.device, dtype=content_selector.dtype)
        )
        content_selector = content_selector + identity + self.evidence_type[:, 0:1].to(
            device=visual.device, dtype=content_selector.dtype
        )[:, :, None, None, None]
        pair_dt = torch.as_tensor(
            [
                self.history_offsets[index + 1] - self.history_offsets[index]
                for index in range(pair_count)
            ],
            device=visual.device,
            dtype=torch.float32,
        )[None, :, None, None, None, None]
        # Geometry stays in native/grid units for address compilation and
        # diagnostics.  The learned motion K/V lane receives a
        # resolution-independent coordinate in [-1,1], with confidence,
        # occlusion and uncertainty kept as separate factual channels.  It is
        # therefore impossible for an uncertain large pixel displacement to
        # dominate the world carrier merely through units.
        normalized_flow_grid = _normalize_flow_evidence(raw_flow_grid)
        normalized_cycle_grid = cycle_grid / float(max(grid - 1, 1))
        flow_xy = normalized_flow_grid.permute(0, 1, 2, 4, 5, 3)
        motion_raw = torch.cat(
            (
                flow_xy,
                flow_xy / pair_dt,
                confidence_grid.permute(0, 1, 2, 4, 5, 3),
                occlusion_grid.permute(0, 1, 2, 4, 5, 3),
                entropy_grid.permute(0, 1, 2, 4, 5, 3),
                margin_grid.permute(0, 1, 2, 4, 5, 3),
                normalized_cycle_grid.permute(0, 1, 2, 4, 5, 3),
                warp_error_grid.permute(0, 1, 2, 4, 5, 3),
            ),
            dim=-1,
        )
        motion_selector = self.motion_key(motion_raw.to(dtype=content_selector.dtype))
        motion_values = self.motion_value(motion_raw.to(dtype=content_values.dtype))
        pair_identity = identity[:, 1:]
        motion_selector = motion_selector + pair_identity + self.evidence_type[:, 1:2].to(
            device=visual.device, dtype=content_selector.dtype
        )[:, :, None, None, None]
        selector = torch.cat(
            (content_selector.flatten(1, 4), motion_selector.flatten(1, 4)), dim=1
        )
        values = torch.cat(
            (content_values.flatten(1, 4), motion_values.flatten(1, 4)), dim=1
        )
        future_queries = self.future_query.to(device=visual.device, dtype=selector.dtype)
        future_queries = future_queries + self.future_anchor_type.to(
            device=visual.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.future_horizon_encoding.to(
            device=visual.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.camera_type[:, :1, :, :1, :1].to(
            device=visual.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.spatial_type[:, :1, :1].to(
            device=visual.device, dtype=selector.dtype
        )
        if self.predictive_change_contract:
            if early_raw_context is None:
                raise RuntimeError("early masked raw context was not constructed")
            # The absolute latest-DINO value used by V96-V102 is deliberately
            # absent here.  The future query starts from local RGB whose hidden
            # cells were removed before trainable mixing, plus observed motion.
            future_context_seed = early_raw_context[:, -1][:, None].to(
                dtype=selector.dtype
            )
        else:
            future_context_seed = content_values[:, -1][:, None]
        future_queries, horizon_metrics = self._compose_future_queries(
            future_queries.expand(batch, -1, -1, -1, -1, -1),
            motion_raw,
            future_context_seed,
        )
        future_queries = future_queries.flatten(1, 4)
        metrics = {
            **raw_metrics,
            **horizon_metrics,
            "flow_jepa_patch_flow_magnitude": magnitude.mean().detach(),
            "flow_jepa_motion_evidence_flow_magnitude": _stable_vector_norm(
                normalized_flow_grid, dim=3
            ).mean().detach(),
            "flow_jepa_motion_evidence_normalized": selector.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_native_flow_magnitude": _stable_vector_norm(
                raw_context.flow_forward, dim=3
            ).mean().detach(),
            "flow_jepa_confidence_mean": raw_context.confidence.float().mean().detach(),
            "flow_jepa_occlusion_fraction": raw_context.occlusion.float().mean().detach(),
            "flow_jepa_correlation_entropy": raw_context.correlation_entropy.float().mean().detach(),
            "flow_jepa_correlation_margin": raw_context.correlation_margin.float().mean().detach(),
            "flow_jepa_context_dropout_fraction": context_dropout.float().mean().detach(),
            "flow_jepa_current_context_mask_fraction": (
                context_dropout[:, -1].float().mean().detach()
            ),
            "flow_jepa_future_target_fraction": future_mask.float().mean().detach(),
            "flow_jepa_predictive_change_contract": selector.new_tensor(
                float(self.predictive_change_contract), dtype=torch.float32
            ),
            "flow_jepa_early_raw_mask_before_mixing": selector.new_tensor(
                float(self.predictive_change_contract), dtype=torch.float32
            ),
            "flow_jepa_future_absolute_dino_seed": selector.new_tensor(
                float(not self.predictive_change_contract), dtype=torch.float32
            ),
            "flow_jepa_context_target_mask_aligned": selector.new_tensor(
                float(
                    self.predictive_change_contract
                    and (
                        self.training
                        or self._raw_address_eval_intervention
                        == "current_context_masked"
                    )
                ),
                dtype=torch.float32,
            ),
            "flow_jepa_future_shared_spatial_mask": selector.new_tensor(
                float(self.predictive_change_contract), dtype=torch.float32
            ),
            "flow_jepa_deploy_context_unmasked": selector.new_tensor(
                float(
                    self.predictive_change_contract
                    and not self.training
                    and self._raw_address_eval_intervention
                    != "current_context_masked"
                ),
                dtype=torch.float32,
            ),
            "flow_jepa_evidence_token_count": selector.new_tensor(float(selector.shape[1])).float(),
            "flow_jepa_horizon_min": selector.new_tensor(float(self.window_offsets[0])).float(),
            "flow_jepa_horizon_max": selector.new_tensor(float(self.window_offsets[-1])).float(),
            "flow_jepa_horizon_count": selector.new_tensor(float(len(self.window_offsets))).float(),
            "flow_jepa_coarse_grid_size": selector.new_tensor(float(grid)).float(),
            "flow_jepa_native_grid_size": selector.new_tensor(
                float(round(float(visual.shape[3]) ** 0.5))
            ).float(),
            "flow_jepa_role_hierarchy": selector.new_ones((), dtype=torch.float32),
            "flow_jepa_grounding_block_count": selector.new_tensor(
                float(self.config.flow_jepa_grounding_blocks)
            ).float(),
            "flow_jepa_world_block_count": selector.new_tensor(
                float(self.config.flow_jepa_world_blocks)
            ).float(),
            "flow_jepa_policy_block_count": selector.new_tensor(
                float(self.config.flow_jepa_policy_blocks)
            ).float(),
            "flow_jepa_late_bottleneck": selector.new_ones((), dtype=torch.float32),
        }
        if early_raw_context is not None:
            metrics["flow_jepa_early_raw_context_norm"] = (
                early_raw_context.detach().float().norm(dim=-1).mean()
            )
        reader_context = RawGroundingContext(
            high_features=raw_context.high_features,
            flow_forward=raw_context.flow_forward[:, -1:],
            flow_backward=raw_context.flow_backward[:, -1:],
            confidence=raw_context.confidence[:, -1:],
            occlusion=raw_context.occlusion[:, -1:],
            uncertainty=raw_context.uncertainty[:, -1:],
            correlation_entropy=raw_context.correlation_entropy[:, -1:],
            correlation_margin=raw_context.correlation_margin[:, -1:],
            cycle_error=raw_context.cycle_error[:, -1:],
            warp_error=raw_context.warp_error[:, -1:],
            observable_motion=raw_context.observable_motion[:, -1:],
            dino_features=(
                visual[:, -2:].reshape(
                    batch,
                    2,
                    cameras,
                    int(round(float(visual.shape[3]) ** 0.5)),
                    int(round(float(visual.shape[3]) ** 0.5)),
                    int(visual.shape[-1]),
                )
                if self.soft_address_lattice_enabled
                else None
            ),
            raw_rgb_pair=(
                raw_visual[:, -2:]
                if bool(
                    int(
                        getattr(
                            self.config,
                            "flow_jepa_coordinate_typed_raw_detail",
                            0,
                        )
                    )
                )
                else None
            ),
        )
        return FlowDINOEvidencePack(
            selector_tokens=selector,
            value_tokens=values,
            key_bias=torch.zeros(selector.shape[1], device=visual.device, dtype=torch.float32),
            stage_query=selector.new_empty(batch, 0, self.hidden),
            future_queries=future_queries,
            context_dropout_mask=context_dropout,
            future_target_mask=future_mask.flatten(1, 4),
            patch_flow_forward=raw_flow_grid,
            patch_flow_backward=to_grid(raw_context.flow_backward)
            * (float(max(grid - 1, 1)) / float(max(high_side - 1, 1))),
            flow_confidence=confidence_grid,
            flow_occlusion=occlusion_grid,
            losses=losses,
            metrics=metrics,
            raw_context=reader_context,
        )

    def refine_raw_evidence(
        self,
        pack: FlowDINOEvidencePack,
        grounding_canvas: Tensor,
        slices: dict[str, slice],
        *,
        return_late_detail: bool = False,
    ) -> (
        tuple[Tensor, Tensor, dict[str, Tensor]]
        | tuple[
            Tensor,
            Tensor,
            dict[str, Tensor],
            LateRawDetailEvidence | None,
        ]
    ):
        """Compile raw detail at the grounding boundary.

        Historical configurations return the fused DINO/raw memory consumed by
        world blocks.  The opt-in late-policy contract instead returns an
        action-local high-frequency payload while leaving world visual memory
        unchanged.  In that mode it caches only the compressed observation-only
        payload. A training-time no-grad preview is explicitly not cached, so
        it cannot consume the shared RGB context before the trainable action
        forward sees it; counterfactuals and ODE steps then reuse the same
        trainable/evaluation payload.
        """

        late_policy_detail = bool(
            int(getattr(self.config, "flow_jepa_late_policy_detail", 0))
        )
        if (
            late_policy_detail
            and pack.late_raw_detail is not None
            and pack.late_raw_detail_metrics is not None
        ):
            cached = (
                pack.selector_tokens,
                pack.value_tokens,
                pack.late_raw_detail_metrics,
                pack.late_raw_detail,
            )
            return cached if bool(return_late_detail) else cached[:3]
        context = pack.raw_context
        if context is None:
            if bool(return_late_detail):
                return pack.selector_tokens, pack.value_tokens, {}, None
            return pack.selector_tokens, pack.value_tokens, {}
        if any(
            module is None
            for module in (
                self.raw_address_reader,
                self.raw_detail_query,
                self.raw_detail_motion,
            )
        ) or self.raw_mask_token is None or self.raw_evidence_type is None:
            raise RuntimeError("raw detail refinement modules are missing")
        batch = int(grounding_canvas.shape[0])
        grid = self.grid_size
        rollout = grounding_canvas[:, slices["rollout"]].reshape(
            batch,
            int(self.config.future_anchors),
            self.cameras,
            grid,
            grid,
            self.hidden,
        )
        grounding_query = rollout.mean(dim=1)
        high = context.high_features
        source = high[:, -2].reshape(
            batch * self.cameras, int(high.shape[3]), int(high.shape[-2]), int(high.shape[-1])
        )
        target = high[:, -1].reshape_as(source)
        high_side = int(source.shape[-1])
        source_mask = F.interpolate(
            pack.context_dropout_mask[:, -2].reshape(
                batch * self.cameras, 1, grid, grid
            ).float(),
            size=(high_side, high_side),
            mode="nearest",
        ).bool()
        target_mask = F.interpolate(
            pack.context_dropout_mask[:, -1].reshape(
                batch * self.cameras, 1, grid, grid
            ).float(),
            size=(high_side, high_side),
            mode="nearest",
        ).bool()
        mask_token = self.raw_mask_token.to(device=source.device, dtype=source.dtype)
        source = torch.where(source_mask, mask_token, source)
        target = torch.where(target_mask, mask_token, target)
        flow = context.flow_forward[:, -1].reshape(
            batch * self.cameras, 2, high_side, high_side
        )

        def latest_grid(value: Tensor) -> Tensor:
            flat = value[:, -1].reshape(
                batch * self.cameras, int(value.shape[3]), high_side, high_side
            )
            return F.interpolate(
                flat.float(), size=(grid, grid), mode="bilinear", align_corners=True
            )

        confidence = context.confidence[:, -1].reshape(
            batch * self.cameras, 1, high_side, high_side
        )
        intervention = self._raw_address_eval_intervention
        if intervention is not None and self.training:
            raise RuntimeError("raw address intervention is evaluation-only")
        coordinate_intervention = (
            intervention
            if intervention in {"none", "zero", "shuffle", "spatial_shuffle"}
            else "spatial_shuffle"
            if intervention == "joint_address_key_spatial_shuffle"
            else "none"
        )
        if intervention == "detail_zero":
            detail_intervention = "zero"
        elif intervention == "detail_spatial_shuffle":
            detail_intervention = "spatial_shuffle"
        elif intervention is not None:
            detail_intervention = "measure"
        else:
            detail_intervention = "none"
        address_flow, single_sample_shuffle_fallback = self._intervened_raw_address_flow(
            flow,
            batch=batch,
            mode="none" if coordinate_intervention is None else coordinate_intervention,
        )
        flow_intervention_delta = (
            address_flow - flow
        ).detach().float().norm(dim=1).mean()
        if self.soft_address_lattice_enabled:
            if not late_policy_detail or self.soft_address_compiler is None:
                raise RuntimeError(
                    "soft address lattice requires the late-policy compiler"
                )
            if context.dino_features is None:
                raise RuntimeError("soft address lattice has no native DINO charts")
            dino = context.dino_features
            dino_side = int(dino.shape[3])
            if (
                dino.ndim != 6
                or tuple(dino.shape[:3]) != (batch, 2, self.cameras)
                or int(dino.shape[4]) != dino_side
            ):
                raise ValueError(
                    "native DINO context must be [B,2,camera,S,S,D]"
                )
            dino_mask = F.interpolate(
                pack.context_dropout_mask[:, -2:].reshape(
                    batch * 2 * self.cameras, 1, grid, grid
                ).float(),
                size=(dino_side, dino_side),
                mode="nearest",
            ).bool().reshape(batch, 2, self.cameras, 1, dino_side, dino_side)
            dino_nchw = dino.permute(0, 1, 2, 5, 3, 4)
            dino_nchw = torch.where(
                dino_mask,
                torch.zeros((), device=dino.device, dtype=dino.dtype),
                dino_nchw,
            )
            dino = dino_nchw.permute(0, 1, 2, 4, 5, 3)
            dino_key_delta = source.new_zeros((), dtype=torch.float32)
            if intervention in {
                "dino_key_spatial_shuffle",
                "joint_address_key_spatial_shuffle",
            }:
                original_target_dino = dino[:, 1]
                shifted_target_dino = original_target_dino.roll(
                    shifts=(
                        max(dino_side // 2, 1),
                        max(dino_side // 3, 1),
                    ),
                    dims=(2, 3),
                )
                dino = torch.stack((dino[:, 0], shifted_target_dino), dim=1)
                dino_key_delta = (
                    shifted_target_dino - original_target_dino
                ).detach().float().norm(dim=-1).mean()
            source_raw_for_address = source.reshape(
                batch, self.cameras, int(source.shape[1]), high_side, high_side
            )
            original_source_raw_for_address = source_raw_for_address
            if intervention == "source_raw_key_zero":
                source_raw_for_address = torch.zeros_like(
                    source_raw_for_address
                )
            elif intervention in {
                "source_raw_key_spatial_shuffle",
                "joint_address_key_spatial_shuffle",
            }:
                source_raw_for_address = source_raw_for_address.roll(
                    shifts=(
                        max(high_side // 2, 1),
                        max(high_side // 3, 1),
                    ),
                    dims=(-2, -1),
                )
            source_raw_key_delta = (
                source_raw_for_address - original_source_raw_for_address
            ).detach().float().norm(dim=2).mean()
            current_rgb_for_address: Tensor | None = None
            if bool(
                int(
                    getattr(
                        self.config,
                        "flow_jepa_coordinate_typed_raw_detail",
                        0,
                    )
                )
            ):
                if context.raw_rgb_pair is None:
                    raise RuntimeError(
                        "coordinate-typed raw detail has no literal RGB pair"
                    )
                current_rgb_for_address = context.raw_rgb_pair[:, 1]
                rgb_side = int(current_rgb_for_address.shape[-1])
                current_rgb_mask = F.interpolate(
                    pack.context_dropout_mask[:, -1].reshape(
                        batch * self.cameras, 1, grid, grid
                    ).float(),
                    size=(rgb_side, rgb_side),
                    mode="nearest",
                ).bool().reshape(batch, self.cameras, 1, rgb_side, rgb_side)
                # Neutral grey maps to exact zero in the compiler's fixed
                # [-1,1] literal chart and cannot reveal a JEPA-masked cell.
                current_rgb_for_address = torch.where(
                    current_rgb_mask,
                    current_rgb_for_address.new_tensor(0.5),
                    current_rgb_for_address,
                )
            address_bank, lattice_metrics = self.soft_address_compiler(
                source_dino=dino[:, 0],
                target_dino=dino[:, 1],
                source_raw=source_raw_for_address,
                target_raw=target.reshape(
                    batch, self.cameras, int(target.shape[1]), high_side, high_side
                ),
                current_rgb=current_rgb_for_address,
                flow=address_flow.reshape(
                    batch, self.cameras, 2, high_side, high_side
                ),
                confidence=confidence.reshape(
                    batch, self.cameras, 1, high_side, high_side
                ),
                uncertainty=context.uncertainty[:, -1],
                occlusion=context.occlusion[:, -1],
                cycle_error=context.cycle_error[:, -1],
            )
            if self.exports_grounded_facts_enabled:
                current_visual = dino[:, 1].reshape(
                    batch,
                    self.cameras,
                    dino_side * dino_side,
                    int(dino.shape[-1]),
                )[:, None]
                # Grounded content stays in the complete normalized DINO
                # width. Association keys use their own low-rank projection;
                # the teacher/prediction content must not be compressed into
                # the common policy hidden width before object supervision.
                address_bank.dense_current_dino_content = (
                    self._teacher_content_grid(current_visual)[:, 0].to(
                        dtype=target.dtype
                    )
                )
            original_fine_values = address_bank.fine_values
            original_dense_detail = address_bank.dense_target_detail
            original_literal_rgb = address_bank.dense_current_rgb
            if detail_intervention == "zero":
                address_bank.fine_values = torch.zeros_like(address_bank.fine_values)
                if address_bank.dense_target_detail is not None:
                    address_bank.dense_target_detail = torch.zeros_like(
                        address_bank.dense_target_detail
                    )
            elif detail_intervention == "spatial_shuffle":
                address_bank.fine_values = address_bank.fine_values.roll(
                    shifts=(max(grid // 2, 1), max(grid // 2, 1)),
                    dims=(2, 3),
                )
                if address_bank.dense_target_detail is not None:
                    dense_side = int(address_bank.dense_target_detail.shape[-1])
                    address_bank.dense_target_detail = (
                        address_bank.dense_target_detail.roll(
                            shifts=(
                                max(dense_side // 2, 1),
                                max(dense_side // 3, 1),
                            ),
                            dims=(-2, -1),
                        )
                    )
            if intervention == "literal_rgb_zero":
                if address_bank.dense_current_rgb is None:
                    raise RuntimeError("literal RGB intervention has no current RGB chart")
                address_bank.dense_current_rgb = torch.zeros_like(
                    address_bank.dense_current_rgb
                )
            elif intervention == "literal_rgb_spatial_shuffle":
                if address_bank.dense_current_rgb is None:
                    raise RuntimeError("literal RGB intervention has no current RGB chart")
                rgb_side = int(address_bank.dense_current_rgb.shape[-1])
                address_bank.dense_current_rgb = address_bank.dense_current_rgb.roll(
                    shifts=(max(rgb_side // 2, 1), max(rgb_side // 3, 1)),
                    dims=(-2, -1),
                )
            detail_intervention_delta = (
                address_bank.fine_values - original_fine_values
            ).detach().float().norm(dim=-1).mean()
            dense_detail_intervention_delta = (
                source.new_zeros((), dtype=torch.float32)
                if original_dense_detail is None
                or address_bank.dense_target_detail is None
                else (
                    address_bank.dense_target_detail - original_dense_detail
                )
                .detach()
                .float()
                .norm(dim=2)
                .mean()
            )
            literal_rgb_intervention_delta = (
                source.new_zeros((), dtype=torch.float32)
                if original_literal_rgb is None
                or address_bank.dense_current_rgb is None
                else (
                    address_bank.dense_current_rgb - original_literal_rgb
                )
                .detach()
                .float()
                .norm(dim=2)
                .mean()
            )
            empty = source.new_empty(batch, 0, self.hidden)
            late_detail = LateRawDetailEvidence(
                selector_tokens=empty,
                value_tokens=empty,
                address_bank=address_bank,
            )
            metrics = {
                **lattice_metrics,
                "flow_jepa_raw_detail_deferred_to_policy": source.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_raw_detail_action_independent_compile": source.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_raw_detail_token_count": source.new_tensor(
                    float(
                        self.cameras
                        * grid
                        * grid
                        * int(self.config.flow_jepa_address_slots)
                        * int(address_bank.fine_keys.shape[-2])
                    ),
                    dtype=torch.float32,
                ),
                "flow_jepa_refined_evidence_token_count": source.new_tensor(
                    float(pack.selector_tokens.shape[1]), dtype=torch.float32
                ),
                "flow_jepa_grounding_refinement": source.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_strict_role_visual_path": source.new_ones(
                    (), dtype=torch.float32
                ),
                "flow_jepa_dino_key_intervention_delta_norm": dino_key_delta,
                "flow_jepa_source_raw_key_intervention_delta_norm": (
                    source_raw_key_delta
                ),
                "flow_jepa_raw_flow_intervention_delta_norm": (
                    flow_intervention_delta
                ),
                "flow_jepa_raw_value_intervention_delta_norm": (
                    detail_intervention_delta
                ),
                "flow_jepa_dense_raw_value_intervention_delta_norm": (
                    dense_detail_intervention_delta
                ),
                "flow_jepa_literal_rgb_intervention_delta_norm": (
                    literal_rgb_intervention_delta
                ),
                "flow_jepa_current_context_mask_fraction": pack.metrics[
                    "flow_jepa_current_context_mask_fraction"
                ],
                "flow_jepa_current_context_mask_intervention_delta": (
                    pack.context_dropout_mask[:, -1]
                    .detach()
                    .float()
                    .mean()
                    if intervention == "current_context_masked"
                    else source.new_zeros((), dtype=torch.float32)
                ),
            }
            if intervention is not None:
                self._raw_address_eval_metrics = {
                    key: value.detach()
                    for key, value in metrics.items()
                    if isinstance(value, Tensor)
                }
                self._raw_address_eval_metrics[
                    "flow_jepa_raw_address_intervention_code"
                ] = source.new_tensor(
                    {
                        "none": 0.0,
                        "zero": 1.0,
                        "shuffle": 2.0,
                        "spatial_shuffle": 3.0,
                        "detail_zero": 4.0,
                        "detail_spatial_shuffle": 5.0,
                        "dino_key_spatial_shuffle": 6.0,
                        "source_raw_key_zero": 7.0,
                        "source_raw_key_spatial_shuffle": 8.0,
                        "joint_address_key_spatial_shuffle": 9.0,
                        "literal_rgb_zero": 10.0,
                        "literal_rgb_spatial_shuffle": 11.0,
                        "current_context_masked": 12.0,
                    }[intervention],
                    dtype=torch.float32,
                )
                self._raw_address_eval_metrics[
                    "flow_jepa_raw_address_shuffle_spatial_fallback"
                ] = source.new_tensor(
                    float(single_sample_shuffle_fallback), dtype=torch.float32
                )
            cache_late_detail = bool(torch.is_grad_enabled() or not self.training)
            if cache_late_detail:
                pack.late_raw_detail = late_detail
                pack.late_raw_detail_metrics = metrics
                pack.raw_context = None
            if bool(return_late_detail):
                return (
                    pack.selector_tokens,
                    pack.value_tokens,
                    metrics,
                    late_detail,
                )
            return pack.selector_tokens, pack.value_tokens, metrics
        magnitude = _stable_vector_norm(flow, dim=1, keepdim=True)
        magnitude_grid = F.interpolate(
            magnitude, size=(grid, grid), mode="bilinear", align_corners=True
        )
        if bool(int(getattr(self.config, "flow_jepa_zero_flow_guard", 0))):
            spatial_motion = latest_grid(context.observable_motion).clamp(0.0, 1.0)
        else:
            spatial_motion = magnitude_grid
        magnitude_scale = spatial_motion.flatten(2).mean(dim=-1, keepdim=True)[..., None]
        motion_features = torch.cat(
            (
                spatial_motion / magnitude_scale.clamp_min(1e-4),
                latest_grid(context.confidence),
                latest_grid(context.occlusion),
                latest_grid(context.correlation_entropy),
                latest_grid(context.uncertainty),
            ),
            dim=1,
        ).permute(0, 2, 3, 1)
        query_flat = grounding_query.reshape(
            batch * self.cameras, grid, grid, self.hidden
        )
        if late_policy_detail:
            # V102 compiles an observation-only raw bank.  Future/rollout
            # grounding tokens can read the noisy action canvas, so using them
            # here would create action -> raw carrier -> action feedback before
            # the explicit late policy read.  The source RGB projection inside
            # the reader plus raw motion determines the bank; action/world
            # conditioning happens once, after the final world block.
            reader_query_flat = torch.zeros_like(query_flat)
            detail_logit = self.raw_detail_motion(
                motion_features.to(dtype=query_flat.dtype)
            )
        else:
            reader_query_flat = query_flat
            detail_logit = self.raw_detail_query(
                query_flat
            ) + self.raw_detail_motion(
                motion_features.to(dtype=query_flat.dtype)
            )
        floor = 0.05 + 0.15 * torch.sigmoid(self.detail_gate_floor_logit.float())
        detail_gate = floor + (1.0 - floor) * torch.sigmoid(
            detail_logit.float()
        ).permute(0, 3, 1, 2)
        reader_output = self.raw_address_reader(
            source,
            target,
            address_flow,
            confidence,
            reader_query_flat,
            detail_gate,
            post_reader_detail_intervention=detail_intervention,
            return_detail_residual=late_policy_detail,
        )
        if late_policy_detail:
            (
                raw_selector,
                raw_value,
                reader_metrics,
                detail_selector_residual,
                detail_value_residual,
            ) = reader_output
        else:
            # Preserve the historical three-value reader contract exactly.
            # V98-V101 callers neither materialize nor consume the late
            # post-reader residual.
            raw_selector, raw_value, reader_metrics = reader_output
            detail_selector_residual = None
            detail_value_residual = None
        if (
            (
                bool(int(getattr(self.config, "flow_jepa_zero_flow_guard", 0)))
                or intervention == "none"
            )
            and not self.training
        ):
            with torch.no_grad():
                _, zero_flow_value, _ = self.raw_address_reader(
                    source,
                    target,
                    torch.zeros_like(flow),
                    confidence,
                    reader_query_flat,
                    detail_gate,
                )
                reader_metrics["zero_flow_value_delta"] = _stable_vector_norm(
                    raw_value.detach().float() - zero_flow_value.float(), dim=-1
                ).mean()
                shuffled_flow, _ = self._intervened_raw_address_flow(
                    flow,
                    batch=batch,
                    mode="shuffle",
                )
                _, shuffled_flow_value, _ = self.raw_address_reader(
                    source,
                    target,
                    shuffled_flow,
                    confidence,
                    reader_query_flat,
                    detail_gate,
                )
                reader_metrics["shuffled_flow_value_delta"] = _stable_vector_norm(
                    raw_value.detach().float() - shuffled_flow_value.float(), dim=-1
                ).mean()
        raw_selector = raw_selector.reshape(
            batch, self.cameras, grid, grid, self.hidden
        )
        raw_value = raw_value.reshape_as(raw_selector)
        complementary_detail = bool(
            int(getattr(self.config, "flow_jepa_complementary_raw_detail", 0))
        )
        source_aligned = bool(
            complementary_detail
            and int(getattr(self.config, "flow_jepa_source_aligned_raw_fusion", 0))
        )
        raw_type = self.raw_evidence_type.to(
            device=raw_selector.device, dtype=raw_selector.dtype
        )[:, :, None, None]
        late_detail: LateRawDetailEvidence | None = None
        if late_policy_detail:
            if not complementary_detail:
                raise RuntimeError(
                    "late policy detail requires the complementary raw reader"
                )
            if detail_selector_residual is None or detail_value_residual is None:
                raise RuntimeError("late policy detail residual was not materialized")
            detail_selector = detail_selector_residual.reshape_as(raw_selector)
            detail_values = detail_value_residual.reshape_as(raw_value)
            raw_identity = (
                self.camera_type[:, 0].to(
                    device=detail_selector.device, dtype=detail_selector.dtype
                )
                + self.spatial_type[:, 0, 0].to(
                    device=detail_selector.device, dtype=detail_selector.dtype
                )[:, None]
            )
            # Selection retains the source-chart camera/xy/type identity.  The
            # value remains the exact high-frequency residual, so zero detail
            # is an exact zero update in the downstream bias-free reader.
            late_detail = LateRawDetailEvidence(
                selector_tokens=(
                    detail_selector + raw_identity + raw_type
                ).flatten(1, 3),
                value_tokens=detail_values.flatten(1, 3),
            )
            selector = pack.selector_tokens
            values = pack.value_tokens
        elif complementary_detail:
            # The forward-flow reader is indexed by the preceding/source grid:
            # each source cell reads the current detail at x + flow(x).  V101
            # therefore fuses with the matching source DINO chart.  Historical
            # V100 remains available for controlled reproduction.
            raw_selector = raw_selector + raw_type
            raw_selector_flat = raw_selector.flatten(1, 3)
            raw_value_flat = raw_value.flatten(1, 3)
            spatial_tokens = self.cameras * grid * grid
            fusion_history_index = int(self.config.visual_history_length) - (
                2 if source_aligned else 1
            )
            fusion_start = fusion_history_index * spatial_tokens
            fusion_stop = fusion_start + spatial_tokens
            if fusion_history_index < 0 or fusion_stop > int(pack.selector_tokens.shape[1]):
                raise RuntimeError("matching DINO chart is absent from the raw fusion bank")
            fusion_scale = float(2.0**-0.5)
            fused_selector = fusion_scale * (
                pack.selector_tokens[:, fusion_start:fusion_stop] + raw_selector_flat
            )
            fused_values = fusion_scale * (
                pack.value_tokens[:, fusion_start:fusion_stop] + raw_value_flat
            )
            selector = torch.cat(
                (
                    pack.selector_tokens[:, :fusion_start],
                    fused_selector,
                    pack.selector_tokens[:, fusion_stop:],
                ),
                dim=1,
            )
            values = torch.cat(
                (
                    pack.value_tokens[:, :fusion_start],
                    fused_values,
                    pack.value_tokens[:, fusion_stop:],
                ),
                dim=1,
            )
        else:
            raw_identity = (
                self.camera_type[:, 0].to(
                    device=raw_selector.device, dtype=raw_selector.dtype
                )
                + self.spatial_type[:, 0, 0].to(
                    device=raw_selector.device, dtype=raw_selector.dtype
                )[:, None]
            )
            raw_selector = raw_selector + raw_identity + raw_type
            selector = torch.cat(
                (pack.selector_tokens, raw_selector.flatten(1, 3)), dim=1
            )
            values = torch.cat((pack.value_tokens, raw_value.flatten(1, 3)), dim=1)
        metrics = {
            "flow_jepa_raw_detail_emphasis_mean": detail_gate.mean().detach(),
            "flow_jepa_raw_detail_precision_mean": reader_metrics[
                "detail_precision"
            ],
            "flow_jepa_raw_address_flow_mass": reader_metrics["flow_mass"],
            "flow_jepa_raw_address_fallback_mass": reader_metrics["fallback_mass"],
            "flow_jepa_raw_address_entropy": reader_metrics["entropy"],
            "flow_jepa_raw_candidates_per_cell": reader_metrics["candidate_count"],
            "flow_jepa_raw_address_center_separation": reader_metrics[
                "center_separation"
            ],
            "flow_jepa_raw_address_lane_value_difference": reader_metrics[
                "lane_value_difference"
            ],
            "flow_jepa_raw_address_logit_advantage": reader_metrics[
                "lane_logit_advantage"
            ],
            "flow_jepa_raw_additive_detail_path": reader_metrics[
                "additive_detail_path"
            ],
            "flow_jepa_raw_detail_fused_with_latest_dino": selector.new_tensor(
                float(
                    complementary_detail
                    and not source_aligned
                    and not late_policy_detail
                ),
                dtype=torch.float32,
            ),
            "flow_jepa_raw_detail_fused_with_source_dino": selector.new_tensor(
                float(
                    complementary_detail
                    and source_aligned
                    and not late_policy_detail
                ),
                dtype=torch.float32,
            ),
            "flow_jepa_raw_detail_deferred_to_policy": selector.new_tensor(
                float(late_policy_detail), dtype=torch.float32
            ),
            "flow_jepa_raw_detail_action_independent_compile": selector.new_tensor(
                float(late_policy_detail), dtype=torch.float32
            ),
            "flow_jepa_refined_evidence_token_count": selector.new_tensor(
                float(selector.shape[1]), dtype=torch.float32
            ),
            "flow_jepa_raw_detail_token_count": selector.new_tensor(
                float(raw_selector.shape[1] * raw_selector.shape[2] * raw_selector.shape[3])
            ).float(),
            "flow_jepa_grounding_refinement": selector.new_ones((), dtype=torch.float32),
            "flow_jepa_strict_role_visual_path": selector.new_tensor(
                float(
                    bool(
                        int(
                            getattr(
                                self.config,
                                "flow_jepa_strict_role_visual_path",
                                0,
                            )
                        )
                    )
                ),
                dtype=torch.float32,
            ),
        }
        if "zero_flow_value_delta" in reader_metrics:
            metrics["flow_jepa_raw_address_zero_flow_value_delta"] = reader_metrics[
                "zero_flow_value_delta"
            ]
        if "shuffled_flow_value_delta" in reader_metrics:
            metrics["flow_jepa_raw_address_shuffled_flow_value_delta"] = reader_metrics[
                "shuffled_flow_value_delta"
            ]
        for key in (
            "post_reader_detail_selector_residual_norm",
            "post_reader_detail_value_residual_norm",
            "post_reader_detail_selector_intervention_delta",
            "post_reader_detail_value_intervention_delta",
        ):
            if key in reader_metrics:
                metrics[f"flow_jepa_raw_{key}"] = reader_metrics[key]
        if intervention is not None:
            required = (
                "flow_jepa_raw_flow_grid_magnitude",
                "flow_jepa_raw_seed_reliability",
                "flow_jepa_correlation_entropy",
                "flow_jepa_correlation_margin",
            )
            captured = {
                key: pack.metrics[key].detach()
                for key in required
                if key in pack.metrics
            }
            captured.update(
                {
                    key: value.detach()
                    for key, value in metrics.items()
                    if key.startswith("flow_jepa_raw_address_")
                    or key.startswith("flow_jepa_raw_post_reader_detail_")
                }
            )
            captured["flow_jepa_raw_address_intervention_code"] = flow.new_tensor(
                {
                    "none": 0.0,
                    "zero": 1.0,
                    "shuffle": 2.0,
                    "spatial_shuffle": 3.0,
                    "detail_zero": 4.0,
                    "detail_spatial_shuffle": 5.0,
                }[intervention]
            ).float()
            captured["flow_jepa_raw_address_shuffle_spatial_fallback"] = flow.new_tensor(
                float(single_sample_shuffle_fallback)
            ).float()
            self._raw_address_eval_metrics = captured
        if late_policy_detail:
            # The V102 detail bank is observation-only, so one compressed bank
            # is valid for the main action, counterfactuals, and every ODE step.
            # Do not cache a training-time no-grad preview: doing so would make
            # the subsequent trainable forward reuse detached detail.
            cache_late_detail = bool(torch.is_grad_enabled() or not self.training)
            if cache_late_detail:
                if late_detail is None:
                    raise RuntimeError("late raw detail cache has no payload")
                pack.late_raw_detail = late_detail
                pack.late_raw_detail_metrics = metrics
                pack.raw_context = None
        else:
            # Preserve the historical V98-V101 lifetime and replay behavior for
            # controlled reproduction.  The V102 path above is deliberately
            # separately cached because its bank is observation-only.
            pack.raw_context = None
        if bool(return_late_detail):
            return selector, values, metrics, late_detail
        return selector, values, metrics

    def _forward_late_bottleneck(self, visual: Tensor) -> FlowDINOEvidencePack:
        if any(
            module is None
            for module in (
                self.coarse_organizer,
                self.organized_key,
                self.organized_value,
                self.sparse_fine_flow,
                self.address_reader,
                self.detail_router,
            )
        ):
            raise RuntimeError("late-bottleneck Flow-DINO modules are incomplete")
        pooled = self._pool(visual)
        batch, history, cameras, grid, _, dim = pooled.shape
        pair_count = history - 1
        coarse_forward, coarse_backward, coarse_confidence, _, _, coarse_losses = (
            self._estimate_pairs(pooled)
        )

        def unflatten(value: Tensor) -> Tensor:
            return value.reshape(batch, pair_count, cameras, *value.shape[1:])

        coarse_flow = unflatten(coarse_forward.flow)
        coarse_magnitude = _stable_vector_norm(coarse_flow, dim=3)
        motion_score = torch.zeros(
            batch, history, cameras, grid, grid, device=visual.device, dtype=torch.float32
        )
        motion_score[:, 0] = coarse_magnitude[:, 0]
        motion_score[:, 1:] = coarse_magnitude
        if self.training:
            context_dropout = self._structured_mask(
                motion_score.reshape(batch, history * cameras, grid, grid),
                stochastic=True,
            ).reshape(batch, history, cameras, grid, grid)
        else:
            context_dropout = torch.zeros_like(motion_score, dtype=torch.bool)

        pair_batch = batch * pair_count * cameras
        magnitude_flat = _stable_vector_norm(
            coarse_forward.flow, dim=1, keepdim=True
        )
        magnitude_scale = magnitude_flat.flatten(2).mean(dim=-1, keepdim=True)[..., None]
        router_features = torch.cat(
            (
                magnitude_flat / magnitude_scale.clamp_min(1e-4),
                coarse_confidence,
                coarse_forward.correlation_entropy.float(),
                coarse_forward.correlation_margin.float(),
                coarse_forward.uncertainty.float(),
            ),
            dim=1,
        ).permute(0, 2, 3, 1)
        router = torch.sigmoid(self.detail_router(router_features).permute(0, 3, 1, 2))
        mask_hint = context_dropout[:, 1:].reshape(pair_batch, 1, grid, grid).float()
        floor = 0.05 + 0.20 * torch.sigmoid(self.detail_gate_floor_logit.float())
        importance = floor + (1.0 - floor) * (
            1.0 - (1.0 - router) * (1.0 - 0.75 * mask_hint)
        )

        side = int(round(float(visual.shape[3]) ** 0.5))
        native = visual.reshape(batch, history, cameras, side, side, dim)
        first = native[:, :-1].reshape(pair_batch, side, side, dim)
        second = native[:, 1:].reshape(pair_batch, side, side, dim)
        joined_fine = self.sparse_fine_flow(
            torch.cat((first, second), dim=0),
            torch.cat((second, first), dim=0),
            torch.cat((coarse_forward.flow, coarse_backward.flow), dim=0),
            torch.cat((importance, importance), dim=0),
        )

        def split(value: Tensor) -> tuple[Tensor, Tensor]:
            return value[:pair_batch], value[pair_batch:]

        fine_flow_forward, fine_flow_backward = split(joined_fine.flow)
        fine_uncertainty_forward, fine_uncertainty_backward = split(joined_fine.uncertainty)
        fine_entropy_forward, fine_entropy_backward = split(joined_fine.correlation_entropy)
        fine_margin_forward, fine_margin_backward = split(joined_fine.correlation_margin)
        native_to_grid = float(max(grid - 1, 1)) / float(max(side - 1, 1))
        fine_grid_forward = fine_flow_forward * native_to_grid
        fine_grid_backward = fine_flow_backward * native_to_grid
        warped_backward, forward_cycle_valid = warp_patch_grid(
            fine_grid_backward, fine_grid_forward
        )
        warped_forward, backward_cycle_valid = warp_patch_grid(
            fine_grid_forward, fine_grid_backward
        )
        forward_cycle = fine_grid_forward + warped_backward
        backward_cycle = fine_grid_backward + warped_forward
        forward_cycle_error = _stable_vector_norm(forward_cycle, dim=1, keepdim=True)
        fine_confidence = (
            torch.exp(-fine_uncertainty_forward.float())
            * torch.exp(-forward_cycle_error)
            * forward_cycle_valid.float()
            * (1.0 - fine_entropy_forward.float()).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        fine_occlusion = (~forward_cycle_valid).float()

        axis = torch.linspace(0.0, float(side - 1), grid, device=visual.device)
        base_y, base_x = torch.meshgrid(axis, axis, indexing="ij")
        native_base = torch.stack((base_x, base_y), dim=-1)[None].expand(
            pair_batch, -1, -1, -1
        )
        first_map = first.permute(0, 3, 1, 2)
        second_map = second.permute(0, 3, 1, 2)
        first_sample = self.address_reader._sample(first_map, native_base)
        second_sample = self.address_reader._sample(second_map, native_base)
        forward_coordinates = native_base + fine_flow_forward.permute(0, 2, 3, 1)
        backward_coordinates = native_base + fine_flow_backward.permute(0, 2, 3, 1)
        forward_warp = self.address_reader._sample(second_map, forward_coordinates)
        backward_warp = self.address_reader._sample(first_map, backward_coordinates)
        forward_valid = (
            (forward_coordinates[..., 0] >= 0.0)
            & (forward_coordinates[..., 0] <= float(side - 1))
            & (forward_coordinates[..., 1] >= 0.0)
            & (forward_coordinates[..., 1] <= float(side - 1))
        )[:, None]
        backward_valid = (
            (backward_coordinates[..., 0] >= 0.0)
            & (backward_coordinates[..., 0] <= float(side - 1))
            & (backward_coordinates[..., 1] >= 0.0)
            & (backward_coordinates[..., 1] <= float(side - 1))
        )[:, None]
        forward_warp_error = (
            F.normalize(first_sample.float(), dim=-1)
            - F.normalize(forward_warp.float(), dim=-1)
        ).square().mean(dim=-1)[:, None]
        backward_warp_error = (
            F.normalize(second_sample.float(), dim=-1)
            - F.normalize(backward_warp.float(), dim=-1)
        ).square().mean(dim=-1)[:, None]

        def paired_mean(
            first_value: Tensor,
            first_valid: Tensor,
            second_value: Tensor,
            second_valid: Tensor,
        ) -> Tensor:
            first_weight = first_valid.float()
            second_weight = second_valid.float()
            numerator = (first_value * first_weight).sum() + (
                second_value * second_weight
            ).sum()
            denominator = (first_weight.sum() + second_weight.sum()).clamp_min(1.0)
            return numerator / denominator

        fine_warp_loss = paired_mean(
            torch.sqrt(forward_warp_error + 1e-6),
            forward_valid,
            torch.sqrt(backward_warp_error + 1e-6),
            backward_valid,
        )
        fine_cycle_loss = paired_mean(
            torch.sqrt(forward_cycle.float().square().sum(dim=1, keepdim=True) + 1e-6),
            forward_cycle_valid,
            torch.sqrt(backward_cycle.float().square().sum(dim=1, keepdim=True) + 1e-6),
            backward_cycle_valid,
        )
        fine_smoothness = 0.5 * (
            self._smoothness(fine_grid_forward, first_sample.permute(0, 3, 1, 2))
            + self._smoothness(fine_grid_backward, second_sample.permute(0, 3, 1, 2))
        )
        fine_uncertainty_nll = paired_mean(
            forward_warp_error.detach().sqrt() / fine_uncertainty_forward.clamp_min(1e-4)
            + fine_uncertainty_forward.clamp_min(1e-4).log(),
            forward_valid,
            backward_warp_error.detach().sqrt() / fine_uncertainty_backward.clamp_min(1e-4)
            + fine_uncertainty_backward.clamp_min(1e-4).log(),
            backward_valid,
        )
        losses = {
            "flow_jepa_warp_loss": fine_warp_loss,
            "flow_jepa_cycle_loss": fine_cycle_loss,
            "flow_jepa_smoothness_loss": fine_smoothness,
            "flow_jepa_uncertainty_nll": fine_uncertainty_nll,
            "flow_jepa_refinement_sequence_loss": coarse_losses[
                "flow_jepa_refinement_sequence_loss"
            ],
        }

        organized = self.coarse_organizer(pooled, context_dropout)
        content_selector = self.organized_key(organized)
        content_values = self.organized_value(organized)
        identity = (
            self.history_type.to(device=content_selector.device, dtype=content_selector.dtype)
            + self.camera_type.to(device=content_selector.device, dtype=content_selector.dtype)
            + self.spatial_type.to(device=content_selector.device, dtype=content_selector.dtype)
        )
        content_selector = content_selector + identity + self.evidence_type[:, 0:1].to(
            device=content_selector.device, dtype=content_selector.dtype
        )[:, :, None, None, None]

        pair_dt = torch.as_tensor(
            [
                self.history_offsets[index + 1] - self.history_offsets[index]
                for index in range(pair_count)
            ],
            device=visual.device,
            dtype=torch.float32,
        )[None, :, None, None, None, None]
        fine_flow_u = unflatten(fine_flow_forward)
        fine_grid_u = unflatten(fine_grid_forward)
        fine_confidence_u = unflatten(fine_confidence)
        fine_occlusion_u = unflatten(fine_occlusion)
        fine_entropy_u = unflatten(fine_entropy_forward)
        fine_margin_u = unflatten(fine_margin_forward)
        cycle_u = unflatten(forward_cycle_error)
        warp_error_u = unflatten(forward_warp_error)
        fine_xy = fine_flow_u.permute(0, 1, 2, 4, 5, 3).float()
        motion_raw = torch.cat(
            (
                fine_xy,
                fine_xy / pair_dt,
                fine_confidence_u.permute(0, 1, 2, 4, 5, 3),
                fine_occlusion_u.permute(0, 1, 2, 4, 5, 3),
                fine_entropy_u.permute(0, 1, 2, 4, 5, 3),
                fine_margin_u.permute(0, 1, 2, 4, 5, 3),
                cycle_u.permute(0, 1, 2, 4, 5, 3),
                warp_error_u.permute(0, 1, 2, 4, 5, 3),
            ),
            dim=-1,
        )
        motion_selector = self.motion_key(motion_raw.to(dtype=content_selector.dtype))
        motion_values = self.motion_value(motion_raw.to(dtype=content_values.dtype))
        pair_identity = identity[:, 1:]
        motion_selector = motion_selector + pair_identity + self.evidence_type[:, 1:2].to(
            device=content_selector.device, dtype=content_selector.dtype
        )[:, :, None, None, None]

        native_key = self.content_key(native.to(dtype=content_selector.dtype))
        native_value = self.content_value(native.to(dtype=content_values.dtype))
        high_mask = F.interpolate(
            context_dropout.reshape(batch * history * cameras, 1, grid, grid).float(),
            size=(side, side),
            mode="nearest",
        ).reshape(batch, history, cameras, side, side).bool()
        native_key = torch.where(
            high_mask[..., None],
            self.context_mask_token.to(device=native_key.device, dtype=native_key.dtype),
            native_key,
        )
        native_value = torch.where(
            high_mask[..., None],
            self.context_mask_token.to(device=native_value.device, dtype=native_value.dtype),
            native_value,
        )
        source_key = native_key[:, :-1].reshape(
            pair_batch, side, side, self.hidden
        ).permute(0, 3, 1, 2)
        target_key = native_key[:, 1:].reshape(pair_batch, side, side, self.hidden).permute(
            0, 3, 1, 2
        )
        target_value = native_value[:, 1:].reshape(
            pair_batch, side, side, self.hidden
        ).permute(0, 3, 1, 2)
        addressed_key, addressed_value, reader_metrics = self.address_reader(
            source_key,
            target_key,
            target_value,
            fine_flow_forward,
            fine_confidence,
        )
        addressed_key = addressed_key.reshape(
            batch, pair_count, cameras, grid, grid, self.hidden
        )
        addressed_value = addressed_value.reshape_as(addressed_key)
        warp_selector = self.warp_key(addressed_key)
        warp_values = self.warp_value(
            torch.cat((content_values[:, :-1], addressed_value), dim=-1)
        )
        warp_reliability = fine_confidence_u.permute(0, 1, 2, 4, 5, 3).to(
            device=warp_values.device, dtype=warp_values.dtype
        )
        warp_values = warp_values * warp_reliability
        warp_selector = warp_selector + pair_identity + self.evidence_type[:, 2:3].to(
            device=content_selector.device, dtype=content_selector.dtype
        )[:, :, None, None, None]

        selector = torch.cat(
            (
                content_selector.flatten(1, 4),
                motion_selector.flatten(1, 4),
                warp_selector.flatten(1, 4),
            ),
            dim=1,
        )
        values = torch.cat(
            (
                content_values.flatten(1, 4),
                motion_values.flatten(1, 4),
                warp_values.flatten(1, 4),
            ),
            dim=1,
        )
        key_bias = torch.zeros(selector.shape[1], device=selector.device, dtype=torch.float32)

        latest_motion = motion_raw[:, -1]
        future_queries = self.future_query.to(device=selector.device, dtype=selector.dtype)
        future_queries = future_queries + self.future_anchor_type.to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.future_horizon_encoding.to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.camera_type[:, :1, :, :1, :1].to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.spatial_type[:, :1, :1].to(
            device=selector.device, dtype=selector.dtype
        )
        motion_seed = self.future_motion(latest_motion.to(dtype=selector.dtype))[:, None]
        latest_dt = float(self.history_offsets[-1] - self.history_offsets[-2])
        anchor_scale = torch.as_tensor(
            [math.sqrt(float(offset) / latest_dt) for offset in self.window_offsets],
            device=selector.device,
            dtype=selector.dtype,
        )[None, :, None, None, None, None]
        local_seed = warp_values[:, -1][:, None]
        future_queries = (
            future_queries.expand(batch, -1, -1, -1, -1, -1)
            + anchor_scale * motion_seed
            + 0.10 * local_seed
        ).flatten(1, 4)
        stage_query = selector.new_empty(batch, 0, self.hidden)

        future_score = _stable_vector_norm(fine_grid_u, dim=3)[:, -1]
        future_score = future_score[:, None].expand(
            -1, int(self.config.future_anchors), -1, -1, -1
        )
        future_mask = self._structured_mask(
            future_score.reshape(batch, int(self.config.future_anchors) * cameras, grid, grid),
            stochastic=self.training,
        ).reshape(batch, int(self.config.future_anchors), cameras, grid, grid)
        local_candidates = 2 * (2 * int(self.config.flow_jepa_reader_radius) + 1) ** 2
        fine_candidates = (2 * int(self.config.flow_jepa_fine_radius) + 1) ** 2
        metrics = {
            "flow_jepa_patch_flow_magnitude": _stable_vector_norm(
                fine_grid_u, dim=3
            ).mean().detach(),
            "flow_jepa_native_flow_magnitude": _stable_vector_norm(
                fine_flow_u, dim=3
            ).mean().detach(),
            "flow_jepa_confidence_mean": fine_confidence_u.float().mean().detach(),
            "flow_jepa_occlusion_fraction": fine_occlusion_u.float().mean().detach(),
            "flow_jepa_correlation_entropy": fine_entropy_u.float().mean().detach(),
            "flow_jepa_correlation_margin": fine_margin_u.float().mean().detach(),
            "flow_jepa_context_dropout_fraction": context_dropout.float().mean().detach(),
            "flow_jepa_future_target_fraction": future_mask.float().mean().detach(),
            "flow_jepa_evidence_token_count": selector.new_tensor(float(selector.shape[1])).float(),
            "flow_jepa_horizon_min": selector.new_tensor(float(self.window_offsets[0])).float(),
            "flow_jepa_horizon_max": selector.new_tensor(float(self.window_offsets[-1])).float(),
            "flow_jepa_horizon_count": selector.new_tensor(float(len(self.window_offsets))).float(),
            "flow_jepa_coarse_grid_size": selector.new_tensor(float(grid)).float(),
            "flow_jepa_native_grid_size": selector.new_tensor(float(side)).float(),
            "flow_jepa_detail_gate_mean": importance.float().mean().detach(),
            "flow_jepa_detail_mask_hint_fraction": mask_hint.mean().detach(),
            "flow_jepa_detail_effective_comparisons": (
                importance.float().mean() * float(fine_candidates * grid * grid)
            ).detach(),
            "flow_jepa_detail_candidate_comparisons": selector.new_tensor(
                float(fine_candidates * grid * grid)
            ).float(),
            "flow_jepa_address_flow_mass": reader_metrics["flow_mass"],
            "flow_jepa_address_fallback_mass": reader_metrics["fallback_mass"],
            "flow_jepa_address_entropy": reader_metrics["entropy"],
            "flow_jepa_address_candidates_per_cell": selector.new_tensor(
                float(local_candidates)
            ).float(),
            "flow_jepa_late_bottleneck": selector.new_ones((), dtype=torch.float32),
        }
        return FlowDINOEvidencePack(
            selector_tokens=selector,
            value_tokens=values,
            key_bias=key_bias,
            stage_query=stage_query,
            future_queries=future_queries,
            context_dropout_mask=context_dropout,
            future_target_mask=future_mask.flatten(1, 4),
            patch_flow_forward=fine_flow_u,
            patch_flow_backward=unflatten(fine_flow_backward),
            flow_confidence=fine_confidence_u,
            flow_occlusion=fine_occlusion_u,
            losses=losses,
            metrics=metrics,
        )

    def forward(
        self, visual: Tensor, *, raw_visual: Tensor | None = None
    ) -> FlowDINOEvidencePack:
        if self.raw_enabled:
            if raw_visual is None:
                raise ValueError("raw-image Flow-JEPA requires raw_visual")
            return self._forward_raw_grounding(visual, raw_visual)
        if raw_visual is not None:
            raise ValueError("raw_visual was supplied while raw-image Flow-JEPA is disabled")
        if self.late_bottleneck:
            return self._forward_late_bottleneck(visual)
        pooled = self._pool(visual)
        batch, history, cameras, grid, _, _ = pooled.shape
        forward, backward, confidence, occlusion, warp_error, losses = self._estimate_pairs(pooled)
        pair_count = history - 1

        def unflatten(value: Tensor) -> Tensor:
            return value.reshape(batch, pair_count, cameras, *value.shape[1:])

        flow_forward = unflatten(forward.flow)
        flow_backward = unflatten(backward.flow)
        confidence = unflatten(confidence)
        occlusion = unflatten(occlusion)
        entropy = unflatten(forward.correlation_entropy)
        margin = unflatten(forward.correlation_margin)
        warp_error = unflatten(warp_error)
        flow_magnitude = _stable_vector_norm(flow_forward, dim=3)
        motion_score = torch.zeros(
            batch, history, cameras, grid, grid, device=visual.device, dtype=torch.float32
        )
        motion_score[:, 0] = flow_magnitude[:, 0]
        motion_score[:, 1:] = flow_magnitude
        if self.training:
            context_dropout = self._structured_mask(
                motion_score.reshape(batch, history * cameras, grid, grid),
                stochastic=True,
            ).reshape(batch, history, cameras, grid, grid)
        else:
            context_dropout = torch.zeros_like(motion_score, dtype=torch.bool)

        content_key = self.content_key(pooled.to(dtype=next(self.parameters()).dtype))
        content_value = self.content_value(pooled.to(dtype=next(self.parameters()).dtype))
        mask_token = self.context_mask_token.to(device=content_value.device, dtype=content_value.dtype)
        content_value = torch.where(context_dropout[..., None], mask_token, content_value)
        identity = (
            self.history_type.to(device=content_key.device, dtype=content_key.dtype)
            + self.camera_type.to(device=content_key.device, dtype=content_key.dtype)
            + self.spatial_type.to(device=content_key.device, dtype=content_key.dtype)
        )
        content_selector = content_key + identity + self.evidence_type[:, 0:1].to(
            device=content_key.device, dtype=content_key.dtype
        )[:, :, None, None, None]
        content_values = content_value

        pair_dt = torch.as_tensor(
            [
                self.history_offsets[index + 1] - self.history_offsets[index]
                for index in range(pair_count)
            ],
            device=flow_forward.device,
            dtype=torch.float32,
        )[None, :, None, None, None, None]
        flow_xy = flow_forward.permute(0, 1, 2, 4, 5, 3).float()
        motion_raw = torch.cat(
            (
                flow_xy,
                flow_xy / pair_dt,
                confidence.permute(0, 1, 2, 4, 5, 3),
                occlusion.permute(0, 1, 2, 4, 5, 3),
                entropy.permute(0, 1, 2, 4, 5, 3),
                margin.permute(0, 1, 2, 4, 5, 3),
                _stable_vector_norm(
                    flow_forward
                    + warp_patch_grid(
                        flow_backward.reshape(
                            batch * pair_count * cameras, 2, grid, grid
                        ),
                        flow_forward.reshape(
                            batch * pair_count * cameras, 2, grid, grid
                        ),
                    )[0].reshape_as(flow_forward),
                    dim=3,
                )[..., None],
                warp_error.permute(0, 1, 2, 4, 5, 3),
            ),
            dim=-1,
        )
        motion_selector = self.motion_key(motion_raw.to(dtype=content_key.dtype))
        motion_values = self.motion_value(motion_raw.to(dtype=content_value.dtype))
        pair_identity = identity[:, 1:]
        motion_selector = motion_selector + pair_identity + self.evidence_type[:, 1:2].to(
            device=content_key.device, dtype=content_key.dtype
        )[:, :, None, None, None]

        previous_visible = content_value[:, :-1].reshape(
            batch * pair_count * cameras, grid, grid, self.hidden
        ).permute(0, 3, 1, 2)
        warped_previous, _ = warp_patch_grid(
            previous_visible,
            flow_backward.reshape(batch * pair_count * cameras, 2, grid, grid),
        )
        warped_previous = warped_previous.permute(0, 2, 3, 1).reshape(
            batch, pair_count, cameras, grid, grid, self.hidden
        )
        current_visible = content_value[:, 1:]
        warp_selector = self.warp_key(warped_previous)
        warp_values = self.warp_value(torch.cat((warped_previous, current_visible), dim=-1))
        warp_reliability = confidence.permute(0, 1, 2, 4, 5, 3).to(
            device=warp_values.device, dtype=warp_values.dtype
        )
        # Uncertain or cycle-inconsistent correspondences remain addressable
        # through their selector/motion metadata, but cannot inject a full-
        # amplitude aligned-content value into the action reader.
        warp_values = warp_values * warp_reliability
        warp_selector = warp_selector + pair_identity + self.evidence_type[:, 2:3].to(
            device=content_key.device, dtype=content_key.dtype
        )[:, :, None, None, None]

        selector = torch.cat(
            (
                content_selector.flatten(1, 4),
                motion_selector.flatten(1, 4),
                warp_selector.flatten(1, 4),
            ),
            dim=1,
        )
        values = torch.cat(
            (content_values.flatten(1, 4), motion_values.flatten(1, 4), warp_values.flatten(1, 4)),
            dim=1,
        )
        # OwnedEvidenceMemoryBank currently carries a global [N] prior.  Keep
        # this lane neutral and put per-sample reliability into selector tokens;
        # the action reader is still free to learn sample-specific addressing.
        key_bias = torch.zeros(selector.shape[1], device=selector.device, dtype=torch.float32)

        latest_motion = motion_raw[:, -1]
        future_queries = self.future_query.to(device=selector.device, dtype=selector.dtype)
        future_queries = future_queries + self.future_anchor_type.to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.future_horizon_encoding.to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.camera_type[:, :1, :, :1, :1].to(
            device=selector.device, dtype=selector.dtype
        )
        future_queries = future_queries + self.spatial_type[:, :1, :1].to(
            device=selector.device, dtype=selector.dtype
        )
        motion_seed = self.future_motion(latest_motion.to(dtype=selector.dtype))[:, None]
        latest_dt = float(self.history_offsets[-1] - self.history_offsets[-2])
        # Current flow is local evidence, not a promise of linear motion all
        # the way to the far anchor.  A square-root scale retains horizon
        # sensitivity without extrapolating a 4-frame displacement sixfold.
        anchor_scale = torch.as_tensor(
            [math.sqrt(float(offset) / latest_dt) for offset in self.window_offsets],
            device=selector.device,
            dtype=selector.dtype,
        )[None, :, None, None, None, None]
        future_queries = future_queries.expand(batch, -1, -1, -1, -1, -1) + anchor_scale * motion_seed
        future_queries = future_queries.flatten(1, 4)

        stage_motion = self.stage_motion(
            latest_motion.float().mean(dim=(1, 2, 3)).to(dtype=selector.dtype)
        )[:, None]
        stage_scale = self.stage_motion_scale.to(
            device=selector.device, dtype=selector.dtype
        ).tanh()
        stage_query = (
            self.stage_query_token.to(device=selector.device, dtype=selector.dtype)
            + self.stage_type.to(device=selector.device, dtype=selector.dtype)
            + self.stage_horizon_encoding.to(device=selector.device, dtype=selector.dtype)
            + stage_scale * stage_motion
        ).expand(batch, -1, -1)

        future_score = flow_magnitude[:, -1]
        future_score = future_score[:, None].expand(-1, int(self.config.future_anchors), -1, -1, -1)
        future_mask = self._structured_mask(
            future_score.reshape(batch, int(self.config.future_anchors) * cameras, grid, grid),
            stochastic=self.training,
        ).reshape(batch, int(self.config.future_anchors), cameras, grid, grid)
        metrics = {
            "flow_jepa_patch_flow_magnitude": flow_magnitude.mean().detach(),
            "flow_jepa_confidence_mean": confidence.float().mean().detach(),
            "flow_jepa_occlusion_fraction": occlusion.float().mean().detach(),
            "flow_jepa_correlation_entropy": entropy.float().mean().detach(),
            "flow_jepa_correlation_margin": margin.float().mean().detach(),
            "flow_jepa_context_dropout_fraction": context_dropout.float().mean().detach(),
            "flow_jepa_future_target_fraction": future_mask.float().mean().detach(),
            "flow_jepa_evidence_token_count": torch.as_tensor(
                selector.shape[1], device=selector.device, dtype=torch.float32
            ),
            "flow_jepa_window_horizon_min": torch.as_tensor(
                self.window_offsets[0], device=selector.device, dtype=torch.float32
            ),
            "flow_jepa_window_horizon_max": torch.as_tensor(
                self.window_offsets[-1], device=selector.device, dtype=torch.float32
            ),
            "flow_jepa_stage_horizon": torch.as_tensor(
                self.stage_offset, device=selector.device, dtype=torch.float32
            ),
            "flow_jepa_stage_target_scale": torch.as_tensor(
                self.stage_target_scale, device=selector.device, dtype=torch.float32
            ),
        }
        return FlowDINOEvidencePack(
            selector_tokens=selector,
            value_tokens=values,
            key_bias=key_bias,
            stage_query=stage_query,
            future_queries=future_queries,
            context_dropout_mask=context_dropout,
            future_target_mask=future_mask.flatten(1, 4),
            patch_flow_forward=flow_forward,
            patch_flow_backward=flow_backward,
            flow_confidence=confidence,
            flow_occlusion=occlusion,
            losses=losses,
            metrics=metrics,
        )

    def predict_future(self, future_tokens: Tensor) -> Tensor:
        return self.future_prediction(future_tokens)

    def organize_interval_stage(
        self,
        future_tokens: Tensor,
    ) -> tuple[Tensor, Tensor | None, dict[str, Tensor]]:
        """Apply the V106 W->P interval delta without any teacher input."""

        if not self.interval_stage_enabled:
            return future_tokens, None, {}
        if self.interval_stage_organizer is None:
            raise RuntimeError("interval-stage organizer is missing")
        return self.interval_stage_organizer(future_tokens)

    def organize_horizon_address(
        self,
        future_tokens: Tensor,
        address_bank: SoftAddressLatticeBank | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the single V108 online G3 -> W1 address write.

        This method owns no teacher and performs no future prediction.  It
        returns the refined version of the existing rollout carrier so every
        downstream W/P/action consumer sees the same causal state.
        """

        if not self.online_horizon_address_enabled:
            raise RuntimeError("online horizon address is not enabled")
        if not self.horizon_soft_address_enabled or self.horizon_address_jepa is None:
            raise RuntimeError("online horizon address has no soft-address owner")
        if address_bank is None:
            raise RuntimeError(
                "online horizon address requires the observation-only address bank"
            )
        refined, relevance_logits, metrics = self.horizon_address_jepa(
            future_tokens,
            address_bank,
        )
        return refined, {
            **metrics,
            "flow_jepa_online_horizon_address": future_tokens.new_ones(
                (), dtype=torch.float32
            ),
            "flow_jepa_horizon_address_logits": relevance_logits,
        }

    def begin_progressive_grounding_address(
        self,
        address_bank: SoftAddressLatticeBank | None,
    ) -> ProgressiveGroundingAddressState:
        """Create the V109 observation scaffold before G1.

        This operation owns no rollout/policy query and performs no value read;
        it only validates and wraps the observation-only lattice.
        """

        if not self.progressive_grounding_address_enabled:
            raise RuntimeError("progressive grounding address is not enabled")
        if self.progressive_grounding_address is None:
            raise RuntimeError("progressive grounding address organizer is missing")
        if address_bank is None:
            raise RuntimeError("progressive grounding address has no observation bank")
        return self.progressive_grounding_address.begin(address_bank)

    def update_progressive_grounding_address(
        self,
        state: ProgressiveGroundingAddressState,
        future_tokens: Tensor,
        *,
        stage: int,
        intervention: str | None = None,
        collect_diagnostics: bool = True,
    ) -> ProgressiveGroundingAddressState:
        """Advance exactly one of the typed G1/G2/G3 selector transitions."""

        if not self.progressive_grounding_address_enabled:
            raise RuntimeError("progressive grounding address is not enabled")
        if self.progressive_grounding_address is None:
            raise RuntimeError("progressive grounding address organizer is missing")
        return self.progressive_grounding_address.update(
            state,
            future_tokens,
            stage=stage,
            intervention=intervention,
            collect_diagnostics=collect_diagnostics,
            candidate_sampler=(
                self.soft_address_compiler.progressive_fine_candidates
                if stage == 2 and self.soft_address_compiler is not None
                else None
            ),
        )

    def score_progressive_horizon_posterior(
        self,
        future_tokens: Tensor,
        state: ProgressiveGroundingAddressState,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Form the W-owned target/source posterior before P reads raw values."""

        if not self.progressive_grounding_address_enabled:
            raise RuntimeError("progressive grounding address is not enabled")
        if self.progressive_grounding_address is None:
            raise RuntimeError("progressive grounding address organizer is missing")
        return self.progressive_grounding_address.score_horizon_posterior(
            future_tokens,
            state,
        )

    def advance_progressive_world_owner_state(
        self,
        future_tokens: Tensor,
        state: ProgressiveGroundingAddressState,
        *,
        depth: int,
        intervention: str | None = None,
        horizon_query_context: Tensor | None = None,
        intent_window_view: IntentWindowView | None = None,
        grounded_intent_state: GroundedIntentState | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Advance the G3-entry and configured-W private selector bundle."""

        if not self.pre_value_owner_routing_enabled:
            return future_tokens, {}
        if self.progressive_grounding_address is None:
            raise RuntimeError("pre-value owner routing has no progressive organizer")
        return self.progressive_grounding_address.advance_world_owner_state(
            future_tokens,
            state,
            depth=depth,
            intervention=intervention,
            horizon_query_context=horizon_query_context,
            intent_window_view=intent_window_view,
            grounded_intent_state=grounded_intent_state,
            collect_diagnostics=collect_diagnostics,
        )

    def progressive_interval_prediction(
        self,
        state: ProgressiveGroundingAddressState,
    ) -> Tensor:
        """Return the supervised interval candidate from the online W path."""

        if not self.functional_mainline_routing_enabled:
            raise RuntimeError(
                "online W interval prediction requires functional mainline routing"
            )
        if self.grounded_intent_effect_mainline_enabled:
            effect_field = state.world_grounded_effect_field
            facts = state.grounded_fact_set
            if effect_field is None or facts is None:
                raise RuntimeError(
                    "grounded interval prediction has no completed effect field"
                )
            effect_field.validate()
            weights = facts.semantic_owner_probs.float()
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            prediction = (
                effect_field.semantic_delta.float()
                * weights[:, None, ..., None]
            ).sum(dim=-2)
            batch, anchors, cameras, rows, columns, content_dim = prediction.shape
            if anchors != int(self.config.future_anchors):
                raise RuntimeError(
                    "grounded W prediction lost one of four real intervals"
                )
            # Even the slot-reduced compatibility diagnostic stays in the
            # complete DINO content space. It is audit-only in this capability
            # and must not silently redefine the object-level target width.
            return prediction.reshape(
                batch,
                anchors * cameras * rows * columns,
                content_dim,
            ).to(dtype=effect_field.semantic_delta.dtype)
        if self.differential_intent_effect_mainline_enabled:
            effect_field = state.world_differential_effect_field
            slot_weights = state.canonical_semantic_slot_weights
            if effect_field is None or slot_weights is None:
                raise RuntimeError(
                    "differential interval prediction has no completed effect "
                    "bank or canonical slot weights"
                )
            effect_field.validate(expected_slots=3)
            weights = slot_weights.float()
            weights = weights / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            window_prediction = (
                effect_field.semantic_delta.float()
                * weights[:, None, ..., None]
            ).sum(dim=-2)
            batch, _, cameras, rows, columns, hidden = window_prediction.shape
            flattened = window_prediction.permute(
                0,
                2,
                3,
                4,
                5,
                1,
            ).reshape(-1, hidden, 3)
            prediction = F.interpolate(
                flattened,
                size=int(self.config.future_anchors),
                mode="linear",
                align_corners=True,
            ).reshape(
                batch,
                cameras,
                rows,
                columns,
                hidden,
                int(self.config.future_anchors),
            ).permute(0, 5, 1, 2, 3, 4).to(
                dtype=effect_field.semantic_delta.dtype
            )
        elif self.g_aligned_future_effect_enabled and not self.window_effect_bank_enabled:
            effect_field = state.world_future_effect_field
            slot_weights = state.canonical_semantic_slot_weights
            if effect_field is None or slot_weights is None:
                raise RuntimeError(
                    "G-aligned interval prediction has no FutureEffectField"
                )
            effect_field.validate()
            weights = slot_weights.float()
            weights = weights / weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            prediction = (
                effect_field.semantic_delta.float()
                * weights[:, None, ..., None]
            ).sum(dim=-2).to(dtype=effect_field.semantic_delta.dtype)
        else:
            # V117 deliberately keeps the inherited four-anchor JEPA target
            # separate from the three-slot near/mid/late effect bank.  The
            # former remains an auxiliary W prediction and is never consumed
            # by P2; collapsing it onto the three effect slots would silently
            # change the old JEPA loss geometry.
            prediction = state.world_interval_progress_prediction
        expected_depth = int(self.config.flow_jepa_world_blocks)
        if prediction is None or state.world_owner_depth != expected_depth:
            raise RuntimeError(
                "online W interval prediction requires the completed final W "
                "owner state"
            )
        expected = (
            int(prediction.shape[0]),
            int(self.config.future_anchors),
            int(self.config.num_cameras),
            int(self.config.flow_jepa_grid_size),
            int(self.config.flow_jepa_grid_size),
            int(self.config.hidden_size),
        )
        if tuple(prediction.shape) != expected:
            raise ValueError(
                "online W interval prediction lost anchor/camera/spatial ownership"
            )
        return prediction.reshape(
            int(prediction.shape[0]),
            int(self.config.future_anchors)
            * int(self.config.num_cameras)
            * int(self.config.flow_jepa_grid_size)
            * int(self.config.flow_jepa_grid_size),
            int(self.config.hidden_size),
        )

    def predict_future_with_address(
        self,
        future_tokens: Tensor,
        address_bank: SoftAddressLatticeBank | None,
        *,
        enable_address: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.horizon_soft_address_enabled or not enable_address:
            return self.predict_future(future_tokens), {}
        if self.horizon_address_jepa is None:
            raise RuntimeError("horizon soft-address JEPA module is missing")
        if address_bank is None:
            raise RuntimeError(
                "horizon soft-address JEPA requires the observation-only address bank"
            )
        refined, relevance_logits, metrics = self.horizon_address_jepa(
            future_tokens,
            address_bank,
        )
        return self.future_prediction(refined), {
            **metrics,
            "flow_jepa_horizon_address_logits": relevance_logits,
        }

    def predict_stage(self, stage_tokens: Tensor) -> Tensor:
        if self.late_bottleneck:
            if stage_tokens.ndim != 3 or int(stage_tokens.shape[1]) != 0:
                raise ValueError("late-bottleneck Flow-DINO has no separate stage tokens")
            return stage_tokens
        if stage_tokens.ndim != 3 or int(stage_tokens.shape[1]) != 1:
            raise ValueError("stage tokens must be [B,1,H]")
        if self.stage_prediction is None:
            raise RuntimeError("stage prediction module is missing")
        return self.stage_prediction(stage_tokens)

    def _teacher_pool_grid(self, visual: Tensor) -> Tensor:
        if visual.ndim != 5:
            raise ValueError("teacher grid input must be [B,F,C,P,D]")
        batch, frames, cameras, patches, dim = visual.shape
        side = int(round(float(patches) ** 0.5))
        if side * side != patches:
            raise ValueError("future DINO target requires a square patch grid")
        grid = self.grid_size
        with torch.autocast(device_type=visual.device.type, enabled=False):
            pooled = F.adaptive_avg_pool2d(
                visual.reshape(batch * frames * cameras, side, side, dim)
                .permute(0, 3, 1, 2)
                .float(),
                (grid, grid),
            ).permute(0, 2, 3, 1)
        return pooled.reshape(
            batch, frames, cameras, grid, grid, dim
        )

    def _teacher_project_grid(self, visual: Tensor) -> Tensor:
        content = self._teacher_content_grid(visual)
        batch, frames, cameras, grid, _, _ = content.shape
        with torch.autocast(device_type=visual.device.type, enabled=False):
            projected = F.linear(
                content.to(
                    device=self.teacher_projection.weight.device, dtype=torch.float32
                ),
                self.teacher_projection.weight.float(),
            )
        return projected.reshape(batch, frames, cameras, grid, grid, self.hidden)

    def _teacher_content_grid(self, visual: Tensor) -> Tensor:
        """Return full-width normalized DINO content for grounded targets."""

        pooled = self._teacher_pool_grid(visual)
        with torch.autocast(
            device_type=visual.device.type,
            enabled=False,
        ):
            return self.teacher_norm(pooled).float()

    @torch.no_grad()
    def object_teacher_supports(self, visual: Tensor) -> Tensor:
        """Public full-width future support chart for the new teacher only.

        ``visual`` is the existing cached future-DINO tensor [B,F,C,P,D].
        The returned chart is detached, built once per training batch, and is
        never called by deployment sampling.
        """

        return self._teacher_content_grid(visual).detach()

    @torch.no_grad()
    def _update_teacher_g_ema(self) -> None:
        if not self.g_aligned_future_effect_enabled:
            return
        if (
            self.teacher_g_semantic_projection is None
            or self.soft_address_compiler is None
        ):
            raise RuntimeError("Teacher-G EMA modules are incomplete")
        if not self.training:
            return
        decay = float(self.teacher_g_ema_decay)
        for teacher, online in zip(
            self.teacher_g_semantic_projection.parameters(),
            self.soft_address_compiler.target_dino_key.parameters(),
            strict=True,
        ):
            teacher.mul_(decay).add_(
                online.detach().to(
                    device=teacher.device, dtype=teacher.dtype
                ),
                alpha=1.0 - decay,
            )

    @torch.no_grad()
    def _teacher_weighted_feature_dispersion(
        self,
        association: Tensor,
        support_content: Tensor,
        matched_content: Tensor,
    ) -> Tensor:
        """Return weighted feature RMS without materializing query/support pairs.

        ``association`` owns the current-cell/slot to future-cell posterior:
        [B,S,C,I,J,M,U,V].  Expanding both feature charts would append H to
        that complete Cartesian product and costs 6 GiB for the production
        B8/S12/C2/G8/M4/H512 contract.  The second-moment identity computes
        exactly the same weighted squared distance while reducing H before
        the current/future spatial product is formed.
        """

        if association.ndim != 8:
            raise ValueError(
                "Teacher-G association must be [B,S,C,I,J,M,U,V]"
            )
        if support_content.ndim != 6:
            raise ValueError(
                "Teacher-G support content must be [B,S,C,U,V,H]"
            )
        if matched_content.ndim != 7:
            raise ValueError(
                "Teacher-G matched content must be [B,S,C,I,J,M,H]"
            )
        if tuple(association.shape[:3]) != tuple(support_content.shape[:3]):
            raise ValueError(
                "Teacher-G association/support batch axes do not align"
            )
        if tuple(association.shape[-2:]) != tuple(
            support_content.shape[-3:-1]
        ):
            raise ValueError(
                "Teacher-G association/support spatial axes do not align"
            )
        if tuple(association.shape[:-2]) != tuple(
            matched_content.shape[:-1]
        ):
            raise ValueError(
                "Teacher-G association/matched query axes do not align"
            )
        if int(support_content.shape[-1]) != int(
            matched_content.shape[-1]
        ):
            raise ValueError(
                "Teacher-G support/matched feature widths do not align"
            )

        support_second_moment = (
            support_content.float().square().mean(dim=-1)
        )
        expected_second_moment = torch.einsum(
            "bscijmuv,bscuv->bscijm",
            association.float(),
            support_second_moment,
        )
        matched_second_moment = (
            matched_content.float().square().mean(dim=-1)
        )
        return (
            expected_second_moment - matched_second_moment
        ).clamp_min(0.0).sqrt()

    @torch.no_grad()
    def _teacher_g_v116_track_pack(
        self,
        target_visual: Tensor,
        current_visual: Tensor,
        current_state: ProgressiveGroundingAddressState | None,
    ) -> tuple[FutureTeacherTrackPack, Tensor]:
        """Symmetric, ordered and streamed Teacher-G association.

        Current and future semantic keys use the same EMA projection. G3 owns
        only current slot weights/support/coordinates. Supports are processed
        in time order, so the next association is conditioned on the previous
        soft match without constructing a support-by-spatial-by-hidden tensor.
        """

        if self.teacher_g_semantic_projection is None:
            raise RuntimeError("V116 Teacher-G projection is missing")
        support_offsets = tuple(
            int(value)
            for value in self.config.flow_jepa_effective_interval_support_offsets
        )
        if target_visual.ndim != 6 or current_visual.ndim != 5:
            raise ValueError("V116 Teacher-G inputs have invalid ranks")
        if int(target_visual.shape[1]) != len(support_offsets):
            raise ValueError("V116 Teacher-G support count does not match offsets")
        self._update_teacher_g_ema()
        self._teacher_g_build_count += 1

        support_pooled = self._teacher_pool_grid(target_visual[:, :, -1]).float()
        current_pooled = self._teacher_pool_grid(current_visual[:, -1:]).float()
        with torch.autocast(device_type=support_pooled.device.type, enabled=False):
            normalized_support_content = self.teacher_norm(
                support_pooled
            ).float()
            normalized_current_content = self.teacher_norm(
                current_pooled
            ).float()
            if self.grounded_intent_effect_mainline_enabled:
                support_content = normalized_support_content
                current_content_grid = normalized_current_content[:, 0]
            else:
                # Exact V116 ancestry: only the grounded sibling preserves
                # the full DINO content width.
                support_content = F.linear(
                    normalized_support_content,
                    self.teacher_projection.weight.float(),
                )
                current_content_grid = F.linear(
                    normalized_current_content,
                    self.teacher_projection.weight.float(),
                )[:, 0]
            if self.grounded_intent_effect_mainline_enabled:
                # Association content and association keys are separate
                # spaces. Both sides of the semantic comparison use the same
                # frozen/EMA key projection over the same normalized DINO
                # domain; the full-width content remains untouched.
                support_semantic = self.teacher_g_semantic_projection(
                    normalized_support_content
                ).float()
                current_semantic_grid = (
                    self.teacher_g_semantic_projection(
                        normalized_current_content
                    ).float()[:, 0]
                )
            else:
                support_semantic = self.teacher_g_semantic_projection(
                    support_pooled
                ).float()
                current_semantic_grid = self.teacher_g_semantic_projection(
                    current_pooled
                ).float()[:, 0]

        batch = int(current_visual.shape[0])
        supports = len(support_offsets)
        cameras = int(self.cameras)
        grid = int(self.grid_size)
        slots = int(self.config.flow_jepa_address_slots)
        axis = torch.linspace(
            -1.0,
            1.0,
            grid,
            device=current_visual.device,
            dtype=torch.float32,
        )
        coordinate_y, coordinate_x = torch.meshgrid(axis, axis, indexing="ij")
        grid_coordinates = torch.stack((coordinate_x, coordinate_y), dim=-1)
        fallback_centers = grid_coordinates.reshape(
            1, 1, grid, grid, 1, 2
        ).expand(batch, cameras, -1, -1, slots, -1)
        fallback_support = current_content_grid.new_full(
            (batch, cameras, grid, grid, slots),
            2.0 / float(max(grid - 1, 1)),
        )
        fallback_slot_weights = current_content_grid.new_full(
            (batch, cameras, grid, grid, slots),
            1.0 / float(slots),
        )
        current_centers = fallback_centers
        current_support = fallback_support
        slot_weights = fallback_slot_weights
        flow_delta = torch.zeros_like(current_centers)
        if (
            current_state is not None
            and int(current_state.stage) == 3
            and current_state.rectified_centers is not None
            and current_state.rectified_support is not None
        ):
            current_centers = current_state.rectified_centers.detach().float()
            current_support = current_state.rectified_support.detach().float()
            if current_state.canonical_semantic_slot_weights is not None:
                slot_weights = (
                    current_state.canonical_semantic_slot_weights.detach().float()
                )
            source_centers = current_state.bank.coarse_source_centers
            flow_centers = current_state.bank.coarse_flow_centers
            if (
                source_centers is not None
                and flow_centers is not None
                and tuple(source_centers.shape)
                == (batch, cameras, grid, grid, 2)
                and tuple(flow_centers.shape)
                == (batch, cameras, grid, grid, 2)
            ):
                flow_delta = (
                    flow_centers.detach().float()
                    - source_centers.detach().float()
                )[..., None, :].expand_as(current_centers)

        grounded_facts = (
            current_state.grounded_fact_set
            if (
                self.grounded_intent_effect_mainline_enabled
                and current_state is not None
                and int(current_state.stage) == 3
            )
            else None
        )
        if grounded_facts is not None:
            grounded_facts.validate()
            current_centers = grounded_facts.slot_coordinates.detach().float()
            current_support = grounded_facts.slot_support.detach().float()
            slot_weights = grounded_facts.semantic_owner_probs.detach().float()
            current_semantic = self.teacher_g_semantic_projection(
                grounded_facts.content_slots.detach().float()
            ).float()
            # The teacher's current reference is the exact completed-G3
            # object content seen by the online path. Re-sampling an unmasked
            # copy of the source frame here would make successor and delta
            # losses disagree by the context-mask residual.
            current_content = grounded_facts.content_slots.detach().float()
            slot_validity = grounded_facts.slot_validity.detach().float()[..., 0]
        else:
            current_semantic = current_semantic_grid[..., None, :].expand(
                -1, -1, -1, -1, slots, -1
            )
            current_content = current_content_grid[..., None, :].expand(
                -1, -1, -1, -1, slots, -1
            )
            slot_validity = current_content.new_ones(
                batch,
                cameras,
                grid,
                grid,
                slots,
            )
        track_semantic = current_semantic
        track_center = current_centers
        offset_tensor = torch.as_tensor(
            support_offsets,
            device=current_visual.device,
            dtype=torch.float32,
        )
        max_offset = float(max(support_offsets[-1], 1))
        candidate_coordinate = grid_coordinates.reshape(
            1, 1, 1, 1, 1, grid, grid, 2
        )

        successor_rows: list[Tensor] = []
        transport_rows: list[Tensor] = []
        covariance_rows: list[Tensor] = []
        persistence_rows: list[Tensor] = []
        visibility_rows: list[Tensor] = []
        uncertainty_rows: list[Tensor] = []
        reliability_rows: list[Tensor] = []
        entropy_rows: list[Tensor] = []
        advantage_rows: list[Tensor] = []
        previous_fraction = 0.0
        for support_index, offset in enumerate(support_offsets):
            fraction = float(offset) / max_offset
            flow_increment = math.tanh(2.0 * fraction) - math.tanh(
                2.0 * previous_fraction
            )
            previous_fraction = fraction
            prior_center = (track_center + flow_increment * flow_delta).clamp(
                -1.0, 1.0
            )
            query = F.normalize(track_semantic, dim=-1, eps=1e-4)
            key = F.normalize(
                support_semantic[:, support_index], dim=-1, eps=1e-4
            )
            semantic_logit = torch.einsum(
                "bcijmr,bcuvr->bcijmuv", query, key
            )
            coordinate_delta = candidate_coordinate - prior_center[
                ..., None, None, :
            ]
            width = (
                current_support[..., None, None]
                + 0.08
                + 0.20 * fraction
            ).clamp(0.05, 1.5)
            geometry_logit = (
                -0.5
                * coordinate_delta.square().sum(dim=-1)
                / width.square()
            ).clamp(min=-8.0, max=0.0)
            association = torch.softmax(
                (2.0 * semantic_logit + geometry_logit).flatten(-2), dim=-1
            ).reshape_as(semantic_logit)
            entropy = -(
                association.clamp_min(1e-8)
                * association.clamp_min(1e-8).log()
            ).sum(dim=(-2, -1)) / math.log(float(max(grid * grid, 2)))
            association_max = association.flatten(-2).amax(dim=-1)
            uniform_max = 1.0 / float(grid * grid)
            concentration = (
                (association_max - uniform_max) / max(1.0 - uniform_max, 1e-6)
            ).clamp(0.0, 1.0)
            semantic_expected = torch.einsum(
                "bcijmuv,bcijmuv->bcijm", association, semantic_logit
            )
            semantic_background = semantic_logit.mean(dim=(-2, -1))
            advantage = (
                0.5 * (semantic_expected - semantic_background)
            ).clamp(0.0, 1.0)
            reliability = (
                (1.0 - entropy).clamp(0.0, 1.0)
                * concentration
                * advantage
                * semantic_expected.clamp(0.0, 1.0)
            ).clamp_min(0.0).sqrt() * slot_validity
            matched_content = torch.einsum(
                "bcijmuv,bcuvh->bcijmh",
                association,
                support_content[:, support_index],
            )
            matched_semantic = torch.einsum(
                "bcijmuv,bcuvr->bcijmr",
                association,
                support_semantic[:, support_index],
            )
            matched_coordinate = torch.einsum(
                "bcijmuv,uvd->bcijmd", association, grid_coordinates
            )
            centered = candidate_coordinate - matched_coordinate[
                ..., None, None, :
            ]
            covariance_xy = torch.einsum(
                "bcijmuv,bcijmuvd->bcijmd",
                association,
                centered.square(),
            )
            covariance_cross = torch.einsum(
                "bcijmuv,bcijmuv->bcijm",
                association,
                centered[..., 0] * centered[..., 1],
            )[..., None]
            covariance = torch.cat((covariance_xy, covariance_cross), dim=-1)
            if self.grounded_intent_effect_mainline_enabled:
                # A failed match is a zero covariance change, not a free
                # positive geometry value that P2 could use as a shortcut.
                covariance = covariance * reliability[..., None]
            reliability_value = reliability[..., None]
            successor = (
                reliability_value * matched_content
                + (1.0 - reliability_value) * current_content
            )
            transport = reliability_value * (
                matched_coordinate - current_centers
            )
            persistence = (
                0.5
                * (
                    1.0
                    + F.cosine_similarity(
                        matched_content, current_content, dim=-1
                    )
                ).clamp(0.0, 2.0)
                * reliability
            )[..., None]
            dispersion = self._teacher_weighted_feature_dispersion(
                association[:, None],
                support_content[:, support_index : support_index + 1],
                matched_content[:, None],
            )[:, 0]
            uncertainty = (
                4.0
                * torch.tanh(
                    (dispersion + entropy + (1.0 - reliability)) / 4.0
                )
            )[..., None]
            successor_rows.append(successor)
            transport_rows.append(transport)
            covariance_rows.append(covariance)
            persistence_rows.append(persistence)
            visibility_rows.append(reliability_value)
            uncertainty_rows.append(uncertainty)
            reliability_rows.append(reliability_value)
            entropy_rows.append(entropy[..., None])
            advantage_rows.append(advantage[..., None])
            # Detached soft recurrent track. Low reliability preserves the
            # previous hypothesis rather than jumping to a random support.
            track_semantic = (
                reliability_value * matched_semantic
                + (1.0 - reliability_value) * track_semantic
            )
            track_center = (
                reliability_value * matched_coordinate
                + (1.0 - reliability_value) * prior_center
            ).clamp(-1.0, 1.0)

        successor_support = torch.stack(successor_rows, dim=1)
        transport_support = torch.stack(transport_rows, dim=1)
        covariance_support = torch.stack(covariance_rows, dim=1)
        persistence_support = torch.stack(persistence_rows, dim=1)
        visibility_support = torch.stack(visibility_rows, dim=1)
        uncertainty_support = torch.stack(uncertainty_rows, dim=1)
        reliability_support = torch.stack(reliability_rows, dim=1)
        entropy_support = torch.stack(entropy_rows, dim=1)
        advantage_support = torch.stack(advantage_rows, dim=1)

        aggregated: dict[str, list[Tensor]] = {
            name: []
            for name in (
                "successor",
                "delta",
                "endpoint",
                "transport",
                "covariance",
                "envelope",
                "persistence",
                "visibility",
                "uncertainty",
                "reliability",
                "entropy",
                "advantage",
                "effective",
            )
        }
        for start, end in self.config.flow_jepa_interval_windows:
            selected = [
                index
                for index, offset in enumerate(support_offsets)
                if int(start) <= offset <= int(end)
            ]
            if len(selected) < 2:
                raise ValueError(
                    f"V116 G-aligned interval [{start},{end}] needs two supports"
                )
            times = offset_tensor[selected]
            relative = (times - float(start)) / float(end - start)
            if len(selected) == 2:
                base_weight = torch.full_like(relative, 0.5)
            else:
                base_weight = torch.empty_like(relative)
                base_weight[0] = 0.5 * (relative[1] - relative[0])
                base_weight[-1] = 0.5 * (relative[-1] - relative[-2])
                base_weight[1:-1] = 0.5 * (relative[2:] - relative[:-2])
                base_weight = base_weight / base_weight.sum().clamp_min(1e-8)
            preliminary = (
                successor_support[:, selected]
                * base_weight.reshape(1, len(selected), 1, 1, 1, 1, 1)
            ).sum(dim=1)
            residual = (
                successor_support[:, selected] - preliminary[:, None]
            ).square().mean(dim=-1).sqrt()
            robust_scale = (
                residual.square()
                * base_weight.reshape(1, len(selected), 1, 1, 1, 1)
            ).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
            robust = torch.rsqrt(1.0 + (residual / robust_scale).square())
            robust_weight = robust * base_weight.reshape(
                1, len(selected), 1, 1, 1, 1
            )
            robust_weight = robust_weight / robust_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

            def aggregate(value: Tensor) -> Tensor:
                return (
                    value[:, selected] * robust_weight[..., None]
                ).sum(dim=1)

            aggregated["successor"].append(aggregate(successor_support))
            # The semantic delta is not an independent teacher object.  The
            # robust weights sum to one, so deriving it from the aggregated
            # successor is exact and avoids retaining another production-size
            # [B,S,C,G,G,M,H] buffer (about 100 MiB at B8/S12/H512).
            interval_successor = aggregated["successor"][-1]
            aggregated["delta"].append(
                interval_successor - current_content
            )
            aggregated["endpoint"].append(
                successor_support[:, selected[-1]] - current_content
            )
            aggregated["transport"].append(aggregate(transport_support))
            aggregated["covariance"].append(aggregate(covariance_support))
            aggregated["envelope"].append(
                transport_support[:, selected].norm(dim=-1).amax(dim=1)[..., None]
            )
            aggregated["persistence"].append(aggregate(persistence_support))
            aggregated["visibility"].append(aggregate(visibility_support))
            aggregated["uncertainty"].append(aggregate(uncertainty_support))
            aggregated["reliability"].append(aggregate(reliability_support))
            aggregated["entropy"].append(aggregate(entropy_support))
            aggregated["advantage"].append(aggregate(advantage_support))
            aggregated["effective"].append(
                robust_weight.square().sum(dim=1).reciprocal()
            )

        pack = FutureTeacherTrackPack(
            stable_successor=torch.stack(aggregated["successor"], dim=1).detach(),
            semantic_delta=torch.stack(aggregated["delta"], dim=1).detach(),
            endpoint_delta=torch.stack(aggregated["endpoint"], dim=1).detach(),
            transport_mean=torch.stack(aggregated["transport"], dim=1).detach(),
            transport_covariance=torch.stack(
                aggregated["covariance"], dim=1
            ).detach(),
            path_envelope=torch.stack(aggregated["envelope"], dim=1).detach(),
            persistence=torch.stack(aggregated["persistence"], dim=1).detach(),
            visibility=torch.stack(aggregated["visibility"], dim=1).detach(),
            uncertainty=torch.stack(aggregated["uncertainty"], dim=1).detach(),
            reliability=torch.stack(aggregated["reliability"], dim=1).detach(),
            association_entropy=torch.stack(
                aggregated["entropy"], dim=1
            ).detach(),
            semantic_advantage=torch.stack(
                aggregated["advantage"], dim=1
            ).detach(),
            effective_support=torch.stack(
                aggregated["effective"], dim=1
            ).mean().detach(),
            support_count=current_content.new_tensor(float(supports)),
            current_content=current_content.detach(),
        )
        pack.validate()
        return pack, slot_weights.detach()

    @torch.no_grad()
    def teacher_g_aligned_track_pack(
        self,
        target_visual: Tensor,
        current_visual: Tensor,
        current_state: ProgressiveGroundingAddressState | None,
    ) -> tuple[FutureTeacherTrackPack, Tensor]:
        """Associate future supports to current G facts without fixed cells."""

        if not self.g_aligned_future_effect_enabled:
            raise RuntimeError("G-aligned future teacher is disabled")
        if self.supervised_effect_mainline_enabled:
            return self._teacher_g_v116_track_pack(
                target_visual,
                current_visual,
                current_state,
            )
        if self.teacher_g_semantic_projection is None:
            raise RuntimeError("G-aligned future teacher projection is missing")
        if target_visual.ndim != 6:
            raise ValueError(
                "G-aligned target_visual must be [B,F,H,C,P,D]"
            )
        if current_visual.ndim != 5:
            raise ValueError(
                "G-aligned current_visual must be [B,H,C,P,D]"
            )
        support_offsets = tuple(
            int(value)
            for value in self.config.flow_jepa_effective_interval_support_offsets
        )
        if int(target_visual.shape[1]) != len(support_offsets):
            raise ValueError(
                "G-aligned future support count does not match offsets"
            )
        self._update_teacher_g_ema()
        self._teacher_g_build_count += 1

        support_pooled = self._teacher_pool_grid(
            target_visual[:, :, -1]
        ).float()
        current_pooled = self._teacher_pool_grid(
            current_visual[:, -1:]
        ).float()
        with torch.autocast(
            device_type=support_pooled.device.type, enabled=False
        ):
            support_content = F.linear(
                self.teacher_norm(support_pooled).float(),
                self.teacher_projection.weight.float(),
            )
            current_content_grid = F.linear(
                self.teacher_norm(current_pooled).float(),
                self.teacher_projection.weight.float(),
            )[:, 0]
            support_semantic = self.teacher_g_semantic_projection(
                support_pooled
            ).float()
            fallback_semantic = self.teacher_g_semantic_projection(
                current_pooled
            ).float()[:, 0]

        batch = int(current_visual.shape[0])
        supports = len(support_offsets)
        cameras = int(self.cameras)
        grid = int(self.grid_size)
        slots = int(self.config.flow_jepa_address_slots)
        route_dim = int(fallback_semantic.shape[-1])
        axis = torch.linspace(
            -1.0,
            1.0,
            grid,
            device=current_visual.device,
            dtype=torch.float32,
        )
        coordinate_y, coordinate_x = torch.meshgrid(
            axis, axis, indexing="ij"
        )
        grid_coordinates = torch.stack(
            (coordinate_x, coordinate_y), dim=-1
        )
        fallback_centers = grid_coordinates.reshape(
            1, 1, grid, grid, 1, 2
        ).expand(batch, cameras, -1, -1, slots, -1)
        fallback_support = current_content_grid.new_full(
            (batch, cameras, grid, grid, slots),
            2.0 / float(max(grid - 1, 1)),
        )
        fallback_slot_weights = current_content_grid.new_full(
            (batch, cameras, grid, grid, slots),
            1.0 / float(slots),
        )
        fallback_semantic_slots = fallback_semantic[:, :, :, :, None].expand(
            -1, -1, -1, -1, slots, -1
        )

        completed_g3 = bool(
            current_state is not None
            and int(current_state.stage) == 3
            and current_state.canonical_semantic_keys is not None
            and current_state.rectified_centers is not None
            and current_state.rectified_support is not None
        )
        if completed_g3:
            assert current_state is not None
            assert current_state.canonical_semantic_keys is not None
            assert current_state.rectified_centers is not None
            assert current_state.rectified_support is not None
            current_semantic = (
                current_state.canonical_semantic_keys.detach().float()
            )
            current_centers = current_state.rectified_centers.detach().float()
            current_support = current_state.rectified_support.detach().float()
            slot_weights = current_state.canonical_semantic_slot_weights
            if slot_weights is None:
                slot_weights = fallback_slot_weights
            else:
                slot_weights = slot_weights.detach().float()
            source_centers = current_state.bank.coarse_source_centers
            flow_centers = current_state.bank.coarse_flow_centers
            if (
                source_centers is not None
                and flow_centers is not None
                and tuple(source_centers.shape)
                == (batch, cameras, grid, grid, 2)
                and tuple(flow_centers.shape)
                == (batch, cameras, grid, grid, 2)
            ):
                flow_delta = (
                    flow_centers.detach().float()
                    - source_centers.detach().float()
                )[..., None, :].expand_as(current_centers)
            else:
                flow_delta = torch.zeros_like(current_centers)
        else:
            current_semantic = fallback_semantic_slots
            current_centers = fallback_centers
            current_support = fallback_support
            slot_weights = fallback_slot_weights
            flow_delta = torch.zeros_like(current_centers)

        expected_semantic = (
            batch,
            cameras,
            grid,
            grid,
            slots,
            route_dim,
        )
        if tuple(current_semantic.shape) != expected_semantic:
            raise ValueError(
                "current G semantic slots do not match Teacher-G route space"
            )
        current_content = current_content_grid[
            :, :, :, :, None
        ].expand(-1, -1, -1, -1, slots, -1)
        query = F.normalize(
            current_semantic, dim=-1, eps=1e-4
        )
        key = F.normalize(
            support_semantic, dim=-1, eps=1e-4
        )
        semantic_logit = torch.einsum(
            "bcijmr,bscuvr->bscijmuv",
            query,
            key,
        )
        offset_tensor = torch.as_tensor(
            support_offsets,
            device=current_visual.device,
            dtype=torch.float32,
        )
        relative_offset = offset_tensor / float(max(support_offsets[-1], 1))
        flow_factor = torch.tanh(2.0 * relative_offset).reshape(
            1, supports, 1, 1, 1, 1, 1
        )
        prior_center = (
            current_centers[:, None]
            + flow_factor * flow_delta[:, None]
        ).clamp(-1.0, 1.0)
        candidate_coordinate = grid_coordinates.reshape(
            1, 1, 1, 1, 1, 1, grid, grid, 2
        )
        coordinate_delta = candidate_coordinate - prior_center[
            ..., None, None, :
        ]
        width = (
            current_support[:, None, ..., None, None]
            + 0.08
            + 0.20
            * relative_offset.reshape(1, supports, 1, 1, 1, 1, 1, 1)
        ).clamp(0.05, 1.5)
        geometry_logit = (
            -0.5
            * coordinate_delta.square().sum(dim=-1)
            / width.square()
        ).clamp(min=-8.0, max=0.0)
        association_logit = 2.0 * semantic_logit + geometry_logit
        association = torch.softmax(
            association_logit.flatten(-2), dim=-1
        ).reshape_as(association_logit)
        association_entropy = -(
            association.clamp_min(1e-8)
            * association.clamp_min(1e-8).log()
        ).sum(dim=(-2, -1)) / math.log(float(max(grid * grid, 2)))
        association_max = association.flatten(-2).max(dim=-1).values
        uniform_max = 1.0 / float(grid * grid)
        concentration = (
            (association_max - uniform_max)
            / float(max(1.0 - uniform_max, 1e-6))
        ).clamp(0.0, 1.0)
        spatial_reliability = (
            (1.0 - association_entropy).clamp(0.0, 1.0)
            * concentration
        ).clamp_min(0.0).sqrt()
        semantic_expected = torch.einsum(
            "bscijmuv,bscijmuv->bscijm",
            association,
            semantic_logit,
        )
        semantic_background = semantic_logit.mean(dim=(-2, -1))
        # A geometric prior can concentrate the posterior even when every
        # future semantic key is equally incompatible.  Reliability therefore
        # requires both positive semantic agreement and an advantage over the
        # same-camera candidate background.  Flat semantics yields exactly
        # zero reliability rather than a disguised fixed-cell teacher.
        semantic_advantage_support = (
            0.5 * (semantic_expected - semantic_background)
        ).clamp(0.0, 1.0)
        semantic_positive = semantic_expected.clamp(0.0, 1.0)
        semantic_reliability = (
            semantic_advantage_support * semantic_positive
        ).clamp_min(0.0).sqrt()
        reliability_support = (
            spatial_reliability * semantic_reliability
        ).clamp(0.0, 1.0)
        matched_content = torch.einsum(
            "bscijmuv,bscuvh->bscijmh",
            association,
            support_content,
        )
        matched_coordinate = torch.einsum(
            "bscijmuv,uvd->bscijmd",
            association,
            grid_coordinates,
        )
        centered_coordinate = (
            grid_coordinates.reshape(1, 1, 1, 1, 1, 1, grid, grid, 2)
            - matched_coordinate[..., None, None, :]
        )
        covariance_xy = torch.einsum(
            "bscijmuv,bscijmuvd->bscijmd",
            association,
            centered_coordinate.square(),
        )
        covariance_cross = torch.einsum(
            "bscijmuv,bscijmuv->bscijm",
            association,
            centered_coordinate[..., 0] * centered_coordinate[..., 1],
        )[..., None]
        covariance = torch.cat((covariance_xy, covariance_cross), dim=-1)
        reliability_expanded = reliability_support[..., None]
        successor_support = (
            reliability_expanded * matched_content
            + (1.0 - reliability_expanded) * current_content[:, None]
        )
        delta_support = (
            reliability_expanded
            * (matched_content - current_content[:, None])
        )
        transport_support = reliability_expanded * (
            matched_coordinate - current_centers[:, None]
        )
        persistence_support = (
            (
                0.5
                * (
                1.0
                + F.cosine_similarity(
                    matched_content,
                    current_content[:, None],
                    dim=-1,
                )
                )
            ).clamp(0.0, 1.0)
            * reliability_support
        )[..., None]
        visibility_support = reliability_expanded
        teacher_dispersion_support = (
            self._teacher_weighted_feature_dispersion(
                association,
                support_content,
                matched_content,
            )
        )
        raw_uncertainty_support = (
            teacher_dispersion_support
            + association_entropy
            + (1.0 - reliability_support)
        )
        uncertainty_support = (
            4.0 * torch.tanh(raw_uncertainty_support / 4.0)
        )[..., None]

        successors: list[Tensor] = []
        deltas: list[Tensor] = []
        endpoint_deltas: list[Tensor] = []
        transports: list[Tensor] = []
        covariances: list[Tensor] = []
        envelopes: list[Tensor] = []
        persistences: list[Tensor] = []
        visibilities: list[Tensor] = []
        uncertainties: list[Tensor] = []
        reliabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        semantic_advantages: list[Tensor] = []
        effective_supports: list[Tensor] = []
        for start, end in self.config.flow_jepa_interval_windows:
            selected = [
                index
                for index, offset in enumerate(support_offsets)
                if int(start) <= offset <= int(end)
            ]
            if len(selected) < 2:
                raise ValueError(
                    f"G-aligned interval [{start},{end}] needs two supports"
                )
            times = offset_tensor[selected]
            relative = (times - float(start)) / float(end - start)
            if int(relative.numel()) == 2:
                base_weight = torch.full_like(relative, 0.5)
            else:
                base_weight = torch.empty_like(relative)
                base_weight[0] = 0.5 * (relative[1] - relative[0])
                base_weight[-1] = 0.5 * (relative[-1] - relative[-2])
                base_weight[1:-1] = 0.5 * (
                    relative[2:] - relative[:-2]
                )
                base_weight = base_weight / base_weight.sum().clamp_min(1e-8)
            interval_successor = successor_support[:, selected]
            weight_shape = (
                1,
                len(selected),
                1,
                1,
                1,
                1,
            )
            preliminary = (
                interval_successor
                * base_weight.reshape(*weight_shape, 1)
            ).sum(dim=1)
            residual = (
                interval_successor - preliminary[:, None]
            ).square().mean(dim=-1).sqrt()
            robust_scale = (
                residual.square()
                * base_weight.reshape(weight_shape)
            ).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
            robust = torch.rsqrt(
                1.0 + (residual / robust_scale).square()
            )
            robust_weight = (
                robust * base_weight.reshape(weight_shape)
            )
            robust_weight = robust_weight / robust_weight.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)

            def aggregate(value: Tensor) -> Tensor:
                return (
                    value[:, selected]
                    * robust_weight[..., None]
                ).sum(dim=1)

            successors.append(aggregate(successor_support))
            deltas.append(aggregate(delta_support))
            endpoint_deltas.append(
                successor_support[:, selected[-1]] - current_content
            )
            transports.append(aggregate(transport_support))
            covariances.append(aggregate(covariance))
            envelopes.append(
                transport_support[:, selected]
                .norm(dim=-1)
                .amax(dim=1)[..., None]
            )
            persistences.append(aggregate(persistence_support))
            visibilities.append(aggregate(visibility_support))
            uncertainties.append(aggregate(uncertainty_support))
            reliabilities.append(aggregate(reliability_expanded))
            entropies.append(
                aggregate(association_entropy[..., None])
            )
            semantic_advantages.append(
                aggregate(semantic_advantage_support[..., None])
            )
            effective_supports.append(
                robust_weight.square().sum(dim=1).reciprocal()
            )

        pack = FutureTeacherTrackPack(
            stable_successor=torch.stack(successors, dim=1).detach(),
            semantic_delta=torch.stack(deltas, dim=1).detach(),
            endpoint_delta=torch.stack(endpoint_deltas, dim=1).detach(),
            transport_mean=torch.stack(transports, dim=1).detach(),
            transport_covariance=torch.stack(covariances, dim=1).detach(),
            path_envelope=torch.stack(envelopes, dim=1).detach(),
            persistence=torch.stack(persistences, dim=1).detach(),
            visibility=torch.stack(visibilities, dim=1).detach(),
            uncertainty=torch.stack(uncertainties, dim=1).detach(),
            reliability=torch.stack(reliabilities, dim=1).detach(),
            association_entropy=torch.stack(entropies, dim=1).detach(),
            semantic_advantage=torch.stack(
                semantic_advantages, dim=1
            ).detach(),
            effective_support=torch.stack(
                effective_supports, dim=1
            ).mean().detach(),
            support_count=current_content.new_tensor(float(supports)),
            current_content=current_content.detach(),
        )
        pack.validate()
        return pack, slot_weights.detach()

    def _teacher_stage_summary(self, projected: Tensor) -> Tensor:
        """Pool to one token while retaining coarse camera/spatial movement."""

        if projected.ndim != 6:
            raise ValueError("projected stage grid must be [B,F,C,G,G,H]")
        batch, frames, cameras, grid, _, hidden = projected.shape
        with torch.autocast(device_type=projected.device.type, enabled=False):
            coarse = F.adaptive_avg_pool2d(
                projected.float().reshape(batch * frames * cameras, grid, grid, hidden)
                .permute(0, 3, 1, 2),
                (2, 2),
            ).permute(0, 2, 3, 1)
            coarse = coarse.reshape(batch, frames, cameras, 2, 2, hidden)
            global_summary = projected.float().mean(dim=(2, 3, 4))
            spatial_summary = (
                coarse
                * self.stage_teacher_position_code.to(
                    device=coarse.device, dtype=torch.float32
                )
            ).mean(dim=(2, 3, 4))
        return global_summary + 0.5 * spatial_summary

    @torch.no_grad()
    def teacher_current(self, current_visual: Tensor) -> Tensor:
        """Frozen current-DINO chart used only to measure future change."""

        if current_visual.ndim != 5:
            raise ValueError("current_visual must be [B,H,C,P,D]")
        projected = self._teacher_project_grid(current_visual[:, -1:])
        return projected.reshape(
            int(current_visual.shape[0]),
            self.cameras * self.grid_size * self.grid_size,
            self.hidden,
        ).detach()

    @staticmethod
    @torch.no_grad()
    def _window_teacher_track_pack(
        pack: FutureTeacherTrackPack,
    ) -> WindowTeacherTrackPack:
        """Compile four legacy intervals into fixed near/mid/late targets.

        The original four-interval pack remains available to the inherited
        JEPA objective. Only the supervised W effect interface receives this
        three-slot view. The late slot preserves both 16-32 and 32-48 moments
        through an equal, student-independent robust aggregate.
        """

        pack.validate()
        if int(pack.stable_successor.shape[1]) != 4:
            raise ValueError("V117 window teacher requires four parent intervals")

        def mean_late(value: Tensor) -> Tensor:
            return 0.5 * (value[:, 2:3] + value[:, 3:4])

        def near_mid_late(value: Tensor) -> Tensor:
            return torch.cat((value[:, :2], mean_late(value)), dim=1)

        successor = near_mid_late(pack.stable_successor)
        current = pack.current_content
        semantic = successor - current[:, None]
        endpoint = torch.cat(
            (pack.endpoint_delta[:, :2], pack.endpoint_delta[:, 3:4]),
            dim=1,
        )
        envelope = torch.cat(
            (
                pack.path_envelope[:, :2],
                torch.maximum(
                    pack.path_envelope[:, 2:3],
                    pack.path_envelope[:, 3:4],
                ),
            ),
            dim=1,
        )
        window = WindowTeacherTrackPack(
            stable_successor=successor.detach(),
            semantic_delta=semantic.detach(),
            endpoint_delta=endpoint.detach(),
            transport_mean=near_mid_late(pack.transport_mean).detach(),
            transport_covariance=near_mid_late(
                pack.transport_covariance
            ).detach(),
            path_envelope=envelope.detach(),
            persistence=near_mid_late(pack.persistence).detach(),
            visibility=near_mid_late(pack.visibility).detach(),
            uncertainty=near_mid_late(pack.uncertainty).detach(),
            reliability=near_mid_late(pack.reliability).detach(),
            association_entropy=near_mid_late(
                pack.association_entropy
            ).detach(),
            semantic_advantage=near_mid_late(
                pack.semantic_advantage
            ).detach(),
            effective_support=pack.effective_support.detach(),
            support_count=pack.support_count.detach(),
            current_content=current.detach(),
        )
        window.validate()
        return window

    @torch.no_grad()
    def teacher_interval_targets(
        self,
        target_visual: Tensor,
        current_visual: Tensor,
        current_state: ProgressiveGroundingAddressState | None = None,
    ) -> dict[str, Tensor]:
        """Build spatial interval content, progression, and endpoint targets.

        Every reduction is temporal only.  Camera and 8x8 cell ownership stay
        explicit, so stage supervision cannot replace the precision address
        with a pooled global target.  The robust content summary is accompanied
        by a signed least-squares progression and a separate endpoint
        increment; this is deliberately not a plain frame mean.
        """

        if not self.interval_stage_enabled:
            raise RuntimeError("interval targets requested while V106 is disabled")
        if target_visual.ndim != 6:
            raise ValueError("interval target_visual must be [B,F,H,C,P,D]")
        if current_visual.ndim != 5:
            raise ValueError("interval current_visual must be [B,H,C,P,D]")
        support_offsets = tuple(
            int(value)
            for value in self.config.flow_jepa_effective_interval_support_offsets
        )
        if int(target_visual.shape[1]) != len(support_offsets):
            raise ValueError(
                "interval target frame count does not match serialized support offsets"
            )
        if self.g_aligned_future_effect_enabled:
            track_pack, slot_weights = self.teacher_g_aligned_track_pack(
                target_visual,
                current_visual,
                current_state,
            )
            targets = track_pack.slot_reduced(slot_weights)
            effect_pack: FutureTeacherTrackPack = (
                self._window_teacher_track_pack(track_pack)
                if self.window_effect_bank_enabled
                else track_pack
            )
            targets.update(
                {
                    "flow_jepa_future_effect_successor_target_slots": (
                        effect_pack.stable_successor
                    ),
                    "flow_jepa_future_effect_semantic_target_slots": (
                        effect_pack.semantic_delta
                    ),
                    "flow_jepa_future_effect_transport_target_slots": (
                        effect_pack.transport_mean
                    ),
                    "flow_jepa_future_effect_transport_covariance_target_slots": (
                        effect_pack.transport_covariance
                    ),
                    "flow_jepa_future_effect_persistence_target_slots": (
                        effect_pack.persistence
                    ),
                    "flow_jepa_future_effect_visibility_target_slots": (
                        effect_pack.visibility
                    ),
                    "flow_jepa_future_effect_uncertainty_target_slots": (
                        effect_pack.uncertainty
                    ),
                    "flow_jepa_future_effect_reliability_target_slots": (
                        effect_pack.reliability
                    ),
                    "flow_jepa_g_aligned_future_teacher": (
                        track_pack.support_count.new_ones(())
                    ),
                    "flow_jepa_g_aligned_teacher_used_completed_g3": (
                        track_pack.support_count.new_tensor(
                            float(
                                current_state is not None
                                and int(current_state.stage) == 3
                            )
                        )
                    ),
                    "flow_jepa_teacher_g_builds_this_pack": (
                        track_pack.support_count.new_ones(())
                    ),
                    "flow_jepa_future_effect_teacher_reliability_mean": (
                        effect_pack.reliability.float().mean()
                    ),
                    "flow_jepa_future_effect_teacher_association_entropy": (
                        effect_pack.association_entropy.float().mean()
                    ),
                    "flow_jepa_future_effect_teacher_semantic_advantage": (
                        effect_pack.semantic_advantage.float().mean()
                    ),
                    "flow_jepa_future_effect_target_adjacent_cosine": (
                        F.cosine_similarity(
                            effect_pack.semantic_delta.float().mean(
                                dim=(2, 3, 4, 5)
                            )[:, 1:],
                            effect_pack.semantic_delta.float().mean(
                                dim=(2, 3, 4, 5)
                            )[:, :-1],
                            dim=-1,
                            eps=1e-6,
                        ).mean()
                    ),
                    "flow_jepa_future_effect_target_interval_variation": (
                        effect_pack.semantic_delta.float().mean(
                            dim=(2, 3, 4, 5)
                        ).std(dim=1, unbiased=False).mean()
                    ),
                    "flow_jepa_future_effect_target_transport_variation": (
                        effect_pack.transport_mean.float().mean(
                            dim=(2, 3, 4, 5)
                        ).std(dim=1, unbiased=False).mean()
                    ),
                    "flow_jepa_window_teacher_slots": (
                        effect_pack.support_count.new_tensor(
                            float(effect_pack.stable_successor.shape[1])
                        )
                    ),
                }
            )
            if self.grounded_intent_effect_mainline_enabled:
                if int(effect_pack.stable_successor.shape[1]) != 4:
                    raise RuntimeError(
                        "grounded teacher must preserve all four intervals"
                    )
                # The grounded FutureEffect interface is algebraically
                # zero-centred.  A currently visible/persistent fact is the
                # identity state (1.0), so only its future change may cross
                # W -> P2.  Keep the absolute quantities inside the detached
                # teacher pack for diagnostics; expose changes to the student.
                targets[
                    "flow_jepa_future_effect_persistence_target_slots"
                ] = effect_pack.persistence - 1.0
                targets[
                    "flow_jepa_future_effect_visibility_target_slots"
                ] = effect_pack.visibility - 1.0
                targets[
                    "flow_jepa_future_effect_current_reference_target"
                ] = effect_pack.current_content
                targets["grounded_intent_effect_active"] = (
                    effect_pack.support_count.new_ones(())
                )
                for interval_index, interval_name in enumerate(
                    ("h4_8", "h8_16", "h16_32", "h32_48")
                ):
                    targets[
                        f"grounded_future_effect_target_{interval_name}_rms"
                    ] = (
                        effect_pack.semantic_delta[:, interval_index]
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                    )
                    targets[
                        f"grounded_future_effect_teacher_reliability_{interval_name}"
                    ] = (
                        effect_pack.reliability[:, interval_index]
                        .float()
                        .mean()
                    )
            elif self.differential_intent_effect_mainline_enabled:
                targets[
                    "flow_jepa_future_effect_intent_summary_target_slots"
                ] = effect_pack.semantic_delta.float().mean(
                    dim=(2, 3, 4, 5)
                ).to(dtype=effect_pack.semantic_delta.dtype)
                targets[
                    "flow_jepa_future_effect_current_reference_target"
                ] = effect_pack.current_content
                for slot_index, slot_name in enumerate(
                    ("near", "mid", "late")
                ):
                    targets[
                        f"flow_jepa_future_effect_target_{slot_name}_rms"
                    ] = (
                        effect_pack.semantic_delta[:, slot_index]
                        .float()
                        .square()
                        .mean()
                        .sqrt()
                    )
                    targets[
                        f"flow_jepa_future_effect_teacher_reliability_{slot_name}"
                    ] = (
                        effect_pack.reliability[:, slot_index]
                        .float()
                        .mean()
                    )
            else:
                targets[
                    "flow_jepa_future_effect_current_target_slots"
                ] = effect_pack.current_content[:, None].expand_as(
                    effect_pack.stable_successor
                )
            return {key: value.detach() for key, value in targets.items()}
        support_grid = self._teacher_project_grid(
            target_visual[:, :, -1]
        ).float()
        current_grid = self._teacher_project_grid(
            current_visual[:, -1:]
        ).float()[:, 0]
        contents: list[Tensor] = []
        progressions: list[Tensor] = []
        endpoints: list[Tensor] = []
        effective_supports: list[Tensor] = []
        for start, end in self.config.flow_jepa_interval_windows:
            selected_indices = [
                index
                for index, offset in enumerate(support_offsets)
                if int(start) <= offset <= int(end)
            ]
            if len(selected_indices) < 2:
                raise ValueError(
                    f"interval [{start},{end}] needs at least two support frames"
                )
            values = support_grid[:, selected_indices]
            times = torch.as_tensor(
                [support_offsets[index] for index in selected_indices],
                device=values.device,
                dtype=torch.float32,
            )
            relative = (times - float(start)) / float(end - start)
            # Trapezoid weights honor the real frame spacing.  Robust
            # reweighting is cell-specific and suppresses an isolated teacher
            # outlier without erasing its signed progression target.
            if int(relative.numel()) == 2:
                base_weight = torch.full_like(relative, 0.5)
            else:
                base_weight = torch.empty_like(relative)
                base_weight[0] = 0.5 * (relative[1] - relative[0])
                base_weight[-1] = 0.5 * (relative[-1] - relative[-2])
                base_weight[1:-1] = 0.5 * (
                    relative[2:] - relative[:-2]
                )
                base_weight = base_weight / base_weight.sum().clamp_min(1e-8)
            weight_shape = (1, int(relative.numel()), 1, 1, 1, 1)
            preliminary = (
                values * base_weight.reshape(weight_shape)
            ).sum(dim=1)
            residual_rms = (
                values - preliminary[:, None]
            ).square().mean(dim=-1).sqrt()
            robust_scale = torch.sqrt(
                (
                    residual_rms.square()
                    * base_weight.reshape(1, -1, 1, 1, 1)
                ).sum(dim=1, keepdim=True)
            ).clamp_min(1e-3)
            robust = torch.rsqrt(
                1.0 + (residual_rms / robust_scale).square()
            )
            robust_weight = (
                robust
                * base_weight.reshape(1, -1, 1, 1, 1)
            )
            robust_weight = robust_weight / robust_weight.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            content = (
                values * robust_weight[..., None]
            ).sum(dim=1)

            temporal_center = (
                base_weight * relative
            ).sum()
            centered_time = relative - temporal_center
            slope_denominator = (
                base_weight * centered_time.square()
            ).sum().clamp_min(1e-8)
            progression = (
                values
                * (
                    base_weight * centered_time
                ).reshape(weight_shape)
            ).sum(dim=1) / slope_denominator
            endpoint = values[:, -1] - values[:, 0]
            contents.append(content)
            progressions.append(progression)
            endpoints.append(endpoint)
            effective_supports.append(
                robust_weight.square().sum(dim=1).reciprocal()
            )

        def flatten(rows: list[Tensor]) -> Tensor:
            stacked = torch.stack(rows, dim=1)
            expected_shape = (
                int(target_visual.shape[0]),
                len(self.window_offsets),
                self.cameras,
                self.grid_size,
                self.grid_size,
                self.hidden,
            )
            if tuple(stacked.shape) != expected_shape:
                raise ValueError(
                    "interval teacher rows lost anchor/camera/spatial "
                    f"ownership: got {tuple(stacked.shape)}, "
                    f"expected {expected_shape}"
                )
            anchor_count = int(stacked.shape[1])
            return stacked.reshape(
                int(stacked.shape[0]),
                anchor_count
                * self.cameras
                * self.grid_size
                * self.grid_size,
                self.hidden,
            ).detach()

        return {
            "flow_jepa_future_target": flatten(contents),
            "flow_jepa_interval_progress_target": flatten(progressions),
            "flow_jepa_interval_endpoint_target": flatten(endpoints),
            "flow_jepa_interval_current_target": current_grid.reshape(
                int(current_grid.shape[0]),
                self.cameras * self.grid_size * self.grid_size,
                self.hidden,
            ).detach(),
            "flow_jepa_interval_effective_support": torch.stack(
                effective_supports,
                dim=1,
            ).mean().detach(),
            "flow_jepa_interval_support_count": current_grid.new_tensor(
                float(len(support_offsets))
            ),
        }

    @torch.no_grad()
    def teacher_target(
        self, target_visual: Tensor, current_visual: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return local patch targets and one coarse far-stage delta target.

        ``target_visual`` is compact and ordered as all window anchors followed
        by the single stage horizon.  The current frame is used only on this
        no-grad teacher side to remove static scene content from the stage
        target.  It is never supplied to the online stage/window predictor.
        """

        if target_visual.ndim != 6:
            raise ValueError("target_visual must be [B,F,H,C,P,D]")
        if self.interval_stage_enabled:
            if current_visual is None:
                raise ValueError(
                    "interval-stage teacher requires the frozen current chart"
                )
            targets = self.teacher_interval_targets(
                target_visual,
                current_visual,
            )
            local = targets["flow_jepa_future_target"]
            empty_stage = local.new_empty(
                int(target_visual.shape[0]), 0, self.hidden
            )
            return local.detach(), empty_stage.detach()
        anchors = int(self.config.future_anchors)
        required = anchors if self.late_bottleneck else anchors + 1
        if int(target_visual.shape[1]) < required:
            raise ValueError(
                f"Flow-DINO target_visual must contain at least {required} configured targets"
            )
        future = target_visual[:, :anchors, -1]
        local_projected = self._teacher_project_grid(future)
        local = local_projected.reshape(
            int(target_visual.shape[0]),
            anchors * int(target_visual.shape[3]) * self.grid_size * self.grid_size,
            self.hidden,
        )
        if self.late_bottleneck:
            empty_stage = local.new_empty(int(target_visual.shape[0]), 0, self.hidden)
            return local.detach(), empty_stage.detach()
        stage_future = target_visual[:, anchors : anchors + 1, -1]
        stage_projected = self._teacher_stage_summary(
            self._teacher_project_grid(stage_future)
        )
        if current_visual is None:
            # Shape-safe compatibility for direct teacher tests.  Production
            # training always supplies current_visual and therefore learns a
            # change target rather than a copyable absolute scene summary.
            stage_delta = stage_projected
        else:
            if current_visual.ndim != 5:
                raise ValueError("current_visual must be [B,H,C,P,D]")
            current = current_visual[:, -1:, :, :, :]
            current_projected = self._teacher_stage_summary(
                self._teacher_project_grid(current)
            )
            stage_delta = stage_projected - current_projected
        # Preserve magnitude: normalizing every delta would amplify almost
        # static far targets to the same energy as genuine stage transitions.
        stage_delta = stage_delta.float() * self.stage_target_scale
        return local.detach(), stage_delta.detach()
