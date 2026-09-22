"""Recovered V120 P2 consequence read and five-lane P3 compiler."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .action_codec import ACTION_BAND_ENDS
from .routing import (
    PolicyRoleDeltaBank,
    register_gradient_axis_rms_metrics,
    register_gradient_rms_metric,
    smooth_rms_contract,
)
from .types import (
    CandidateWorld,
    FutureObjectDynamics,
    PhysicalActionCondition,
    PhysicalActionSequenceCondition,
    PolicyIntentDock,
    WorldActionCondition,
    normalized_entropy,
)


@dataclass(frozen=True)
class ObjectTypedEffect:
    """Semantic and physical-geometry P2 values before parameter-free fusion."""

    semantic: Tensor
    geometry: Tensor

    def validate(self) -> None:
        if self.semantic.ndim != 4:
            raise ValueError("typed P2 effect must be [B,T,Q,H]")
        if tuple(self.geometry.shape) != tuple(self.semantic.shape):
            raise ValueError("semantic and geometry P2 effects must align")

    def combined(self) -> Tensor:
        self.validate()
        return self.semantic + self.geometry

    def scaled(self, scale: Tensor) -> "ObjectTypedEffect":
        self.validate()
        if tuple(scale.shape) != (*self.semantic.shape[:-1], 1):
            raise ValueError("typed P2 effect scale must be [B,T,Q,1]")
        return ObjectTypedEffect(
            semantic=self.semantic * scale.to(dtype=self.semantic.dtype),
            geometry=self.geometry * scale.to(dtype=self.geometry.dtype),
        )


@dataclass(frozen=True)
class ObjectEffectBundle:
    """Target-owned effect plus a disjoint non-target scene consequence."""

    target: ObjectTypedEffect
    scene: ObjectTypedEffect
    scene_enabled: bool

    def validate(self) -> None:
        self.target.validate()
        self.scene.validate()
        if tuple(self.scene.semantic.shape) != tuple(self.target.semantic.shape):
            raise ValueError("target and scene P2 effects must align")


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    scene_factual_base: Tensor | None
    effect: ObjectTypedEffect
    interaction: ObjectTypedEffect
    scene_effect: ObjectTypedEffect
    scene_enabled: bool
    protected_consequence: Tensor

    def validate(self) -> None:
        self.effect.validate()
        self.interaction.validate()
        self.scene_effect.validate()
        expected = tuple(self.factual_base.shape)
        if tuple(self.effect.semantic.shape) != expected:
            raise ValueError("typed P2 effect and factual base must align")
        if tuple(self.interaction.semantic.shape) != expected:
            raise ValueError("typed consequence interaction and fact must align")
        if tuple(self.scene_effect.semantic.shape) != expected:
            raise ValueError("scene consequence and factual base must align")
        if self.scene_factual_base is not None and tuple(
            self.scene_factual_base.shape
        ) != expected:
            raise ValueError("scene factual base and target factual base must align")
        if tuple(self.protected_consequence.shape) != expected:
            raise ValueError("protected consequence lost [B,T,Q,H]")

    def innovation(self) -> Tensor:
        self.validate()
        return self.effect.combined() + self.interaction.combined()

    def scene_innovation(self) -> Tensor:
        self.validate()
        scene = self.scene_effect.combined()
        if self.scene_factual_base is not None:
            scene = scene + self.scene_factual_base
        return scene


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """Three optional innovations around two disjoint protected carriers."""

    protected_base: Tensor
    protected_policy_precision: Tensor
    scene: Tensor | None
    temporal: Tensor
    state_change: Tensor

    @property
    def source_names(self) -> tuple[str, ...]:
        names = ("p3_temporal", "p3_state_change")
        if self.scene is None:
            return names
        return ("p3_scene_consequence", *names)

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        if tuple(self.protected_policy_precision.shape) != expected:
            raise ValueError("protected policy precision lost [B,T,Q,H]")
        for name in ("temporal", "state_change"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")
        if self.scene is not None and tuple(self.scene.shape) != expected:
            raise ValueError("object policy scene lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        values = (
            (self.temporal, self.state_change)
            if self.scene is None
            else (self.scene, self.temporal, self.state_change)
        )
        return PolicyRoleDeltaBank(
            values=torch.stack(values, dim=1),
            source_names=self.source_names,
            # ``values`` has exactly one axis entry per declared optional
            # source.  The protected carriers are separate lanes and must not
            # inflate the source-depth metadata cardinality.
            source_depths=(int(source_depth),) * len(self.source_names),
            protected_detail=self.protected_base,
            protected_policy_precision=self.protected_policy_precision,
        )


@dataclass(frozen=True)
class SelectedIntervalEvidence:
    """W evidence after type-local spatial selection and before I removal."""

    key: Tensor  # [B,T,Q,I,Z,H]
    scene_key: Tensor  # [B,T,Q,I,Z,H]
    semantic_value: Tensor  # [B,T,Q,I,D]
    semantic_common_value: Tensor  # [B,T,Q,I,D]
    semantic_residual_value: Tensor  # [B,T,Q,I,D]
    geometry_value: Tensor  # [B,T,Q,I,2]
    geometry_common_value: Tensor  # [B,T,Q,I,2]
    geometry_residual_value: Tensor  # [B,T,Q,I,2]
    semantic_scene_value: Tensor  # [B,T,Q,I,D]
    semantic_scene_common_value: Tensor  # [B,T,Q,I,D]
    semantic_scene_residual_value: Tensor  # [B,T,Q,I,D]
    geometry_scene_value: Tensor  # [B,T,Q,I,2]
    geometry_scene_common_value: Tensor  # [B,T,Q,I,2]
    geometry_scene_residual_value: Tensor  # [B,T,Q,I,2]
    selected_s_context: Tensor  # [B,T,Q,I,Z,H]
    scene_s_context: Tensor  # [B,T,Q,I,Z,H]
    support: Tensor  # bool [B,I,Z]
    scene_support: Tensor  # bool [B,I,Z]

    def validate(self) -> None:
        if self.key.ndim != 6 or int(self.key.shape[-2]) != 2:
            raise ValueError("selected P2 evidence must be [B,T,Q,I,2,H]")
        expected = tuple(self.key.shape)
        if tuple(self.scene_key.shape) != expected:
            raise ValueError("selected P2 scene key lost an identity axis")
        if tuple(self.selected_s_context.shape) != expected:
            raise ValueError("selected P2 S context lost an identity axis")
        if tuple(self.scene_s_context.shape) != expected:
            raise ValueError("selected P2 scene context lost an identity axis")
        if tuple(self.support.shape) != (expected[0], expected[3], expected[4]):
            raise ValueError("selected P2 support must be [B,I,2]")
        if tuple(self.scene_support.shape) != tuple(self.support.shape):
            raise ValueError("selected P2 scene support must be [B,I,2]")
        if self.support.dtype != torch.bool or self.scene_support.dtype != torch.bool:
            raise TypeError("selected P2 support must be boolean")
        value_prefix = expected[:4]
        for name in (
            "semantic_value",
            "semantic_common_value",
            "semantic_residual_value",
            "semantic_scene_value",
            "semantic_scene_common_value",
            "semantic_scene_residual_value",
        ):
            value = getattr(self, name)
            if value.ndim != 5 or tuple(value.shape[:4]) != value_prefix:
                raise ValueError(f"selected P2 {name} lost [B,T,Q,I]")
        for name in (
            "geometry_value",
            "geometry_common_value",
            "geometry_residual_value",
            "geometry_scene_value",
            "geometry_scene_common_value",
            "geometry_scene_residual_value",
        ):
            value = getattr(self, name)
            if tuple(value.shape) != (*value_prefix, 2):
                raise ValueError(f"selected P2 {name} must retain physical 2D")
        # This is the fail-closed admission boundary for the physical-I
        # terminal.  Invalid producer rows have already been quarantined to
        # exact zero by the spatial reader, so every selected carrier must now
        # be finite irrespective of whether expensive diagnostics were
        # requested.  In particular, a supported non-finite typed-S row must
        # never be allowed to modulate the terminal or reach deployment.
        finite_fields = (
            ("key", self.key),
            ("scene key", self.scene_key),
            ("S context", self.selected_s_context),
            ("scene context", self.scene_s_context),
            ("semantic value", self.semantic_value),
            ("semantic common value", self.semantic_common_value),
            ("semantic residual value", self.semantic_residual_value),
            ("geometry value", self.geometry_value),
            ("geometry common value", self.geometry_common_value),
            ("geometry residual value", self.geometry_residual_value),
            ("semantic scene value", self.semantic_scene_value),
            ("semantic scene common value", self.semantic_scene_common_value),
            ("semantic scene residual value", self.semantic_scene_residual_value),
            ("geometry scene value", self.geometry_scene_value),
            ("geometry scene common value", self.geometry_scene_common_value),
            ("geometry scene residual value", self.geometry_scene_residual_value),
        )
        all_finite = torch.stack(
            tuple(torch.isfinite(value).all() for _, value in finite_fields)
        ).all()
        if not bool(all_finite):
            for name, value in finite_fields:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"selected P2 {name} is non-finite")
        if not torch.equal(
            self.semantic_value,
            self.semantic_common_value + self.semantic_residual_value,
        ):
            raise ValueError("selected semantic common/residual identity failed")
        if not torch.equal(
            self.geometry_value,
            self.geometry_common_value + self.geometry_residual_value,
        ):
            raise ValueError("selected geometry common/residual identity failed")
        if not torch.equal(
            self.semantic_scene_value,
            self.semantic_scene_common_value + self.semantic_scene_residual_value,
        ):
            raise ValueError("selected semantic scene common/residual identity failed")
        if not torch.equal(
            self.geometry_scene_value,
            self.geometry_scene_common_value + self.geometry_scene_residual_value,
        ):
            raise ValueError("selected geometry scene common/residual identity failed")


def _safe_masked_log_softmax(
    logit: Tensor,
    support: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor]:
    """Return FP32 probability/log pairs with exact-zero invalid rows."""

    if tuple(logit.shape) != tuple(support.shape):
        raise ValueError("masked softmax logit and support must align")
    has_support = support.any(dim=dim, keepdim=True)
    finite_logit = torch.where(support, logit.float(), torch.zeros_like(logit.float()))
    finite_logit = torch.where(has_support, finite_logit, torch.zeros_like(finite_logit))
    masked = finite_logit.masked_fill(~support & has_support, -torch.inf)
    log_probability = torch.log_softmax(masked, dim=dim)
    active = support & has_support
    log_probability = torch.where(
        active,
        log_probability,
        torch.zeros_like(log_probability),
    )
    probability = torch.where(
        active,
        log_probability.exp(),
        torch.zeros_like(log_probability),
    )
    return probability, log_probability


def _safe_masked_softmax(logit: Tensor, support: Tensor, *, dim: int) -> Tensor:
    """Return finite probabilities and exact zero on all-invalid rows."""

    probability, _ = _safe_masked_log_softmax(logit, support, dim=dim)
    return probability


class ObjectFutureEffectReader(nn.Module):
    """Select W spatial evidence before the no-null physical-I terminal."""

    TYPE_NAMES = ("semantic", "geometry")
    S_TYPE_INDEX_BY_P2 = (0, 2)
    INTERVENTION_MODES = (
        "semantic_far_zero",
        "geometry_value_all_zero",
        "geometry_address_neutral",
        "geometry_value_and_address_zero",
        "target_address_neutral",
    )
    _GEOMETRY_VALUE_ZERO_MODES = frozenset(
        ("geometry_value_all_zero", "geometry_value_and_address_zero")
    )
    _GEOMETRY_ADDRESS_NEUTRAL_MODES = frozenset(
        ("geometry_address_neutral", "geometry_value_and_address_zero")
    )
    _TARGET_ADDRESS_NEUTRAL_MODES = frozenset(("target_address_neutral",))

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        action_dim: int = 7,
        spatial_intent_mode: str = "post_pool_only",
    ) -> None:
        super().__init__()
        if spatial_intent_mode not in {
            "post_pool_only",
            "shared_target_prior_v1",
            "target_action_bottleneck_v1",
        }:
            raise ValueError("unknown P2 spatial intent mode")
        self.hidden = int(hidden)
        self.spatial_intent_mode = str(spatial_intent_mode)
        source_query = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        # Spatial K/K*C selection and physical-I termination are different
        # candidate decisions. Start them as the exact R1 function without
        # consuming initialization RNG, then let their ordinary task gradients
        # separate the two owners.
        self.terminal_query = deepcopy(source_query)
        if self.spatial_intent_mode == "target_action_bottleneck_v1":
            # Semantic K is owned by TargetFact in this mode.  Retain the
            # geometry query, but do not register a dead trainable query whose
            # only historical purpose was to re-softmax across semantic K.
            source_query[0] = nn.Identity()
        self.source_query = source_query
        self.source_key = nn.ModuleList(
            (
                nn.Linear(content_dim, hidden, bias=False),
                nn.Linear(2, hidden, bias=False),
            )
        )
        self.public_interval_key: nn.Linear | None = (
            None
            if self.spatial_intent_mode == "target_action_bottleneck_v1"
            else nn.Linear(hidden, hidden, bias=False)
        )
        self.action_interval_key: nn.Linear | None = (
            nn.Linear(int(action_dim), hidden, bias=False)
            if self.spatial_intent_mode == "target_action_bottleneck_v1"
            else None
        )
        self.typed_intent_key = nn.ModuleList(
            nn.Linear(route_dim, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.coordinate_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.semantic_scene_value: nn.Linear | None = None
        self.transport_scene_value: nn.Linear | None = None
        if self.spatial_intent_mode == "target_action_bottleneck_v1":
            # Non-target consequences are a separate physical scene role, not
            # another operated-object selector.  Exact-zero projections keep
            # the migrated target effect unchanged while ordinary action loss
            # opens the scene residual as soon as W supplies a non-zero raw
            # consequence (immediately for a trained migrated W).
            semantic_scene_value = nn.Linear(content_dim, hidden, bias=False)
            transport_scene_value = nn.Linear(2, hidden, bias=False)
            nn.init.zeros_(semantic_scene_value.weight)
            nn.init.zeros_(transport_scene_value.weight)
            self.semantic_scene_value = semantic_scene_value
            self.transport_scene_value = transport_scene_value
        self.temperature_logit = nn.Parameter(torch.zeros(3))
        # Plain evaluation state: it is neither a parameter nor a persistent
        # buffer and therefore cannot alter checkpoint identity. Matched
        # validation may remove one named value/address seam; ordinary
        # training and deployment keep the exact neutral string.
        self._eval_intervention = "none"

    def set_eval_intervention(self, mode: str) -> None:
        """Select one matched P2 value/address counterfactual for evaluation."""

        if self.training:
            raise ValueError("P2 interventions are evaluation-only")
        if mode not in self.INTERVENTION_MODES:
            raise ValueError(
                "P2 intervention must be one of "
                + ", ".join(self.INTERVENTION_MODES)
            )
        self._eval_intervention = mode

    def clear_eval_intervention(self) -> None:
        self._eval_intervention = "none"

    def _intervened_values(
        self,
        selected: SelectedIntervalEvidence,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        mode = self._eval_intervention
        if mode == "none":
            # Preserve the primary path without even an identity multiply.
            return (
                selected.semantic_common_value,
                selected.semantic_residual_value,
                selected.geometry_common_value,
                selected.geometry_residual_value,
                selected.semantic_scene_common_value,
                selected.semantic_scene_residual_value,
                selected.geometry_scene_common_value,
                selected.geometry_scene_residual_value,
            )
        if self.training:
            raise ValueError("P2 interventions are evaluation-only")
        if mode not in self.INTERVENTION_MODES:
            raise RuntimeError("P2 intervention state is invalid")
        semantic_mask = selected.semantic_common_value.new_ones(
            (1, 1, 1, selected.key.shape[3], 1)
        )
        geometry_mask = selected.geometry_common_value.new_ones(
            (1, 1, 1, selected.key.shape[3], 1)
        )
        # Value interventions happen only after both spatial posteriors have
        # formed. Address-neutral mode is handled at the semantic K logits and
        # therefore leaves both value masks untouched here.
        if mode == "semantic_far_zero":
            semantic_mask[..., (2, 3), :] = 0.0
        if mode in self._GEOMETRY_VALUE_ZERO_MODES:
            geometry_mask.zero_()
        return (
            selected.semantic_common_value * semantic_mask,
            selected.semantic_residual_value * semantic_mask,
            selected.geometry_common_value * geometry_mask,
            selected.geometry_residual_value * geometry_mask,
            selected.semantic_scene_common_value * semantic_mask,
            selected.semantic_scene_residual_value * semantic_mask,
            selected.geometry_scene_common_value * geometry_mask,
            selected.geometry_scene_residual_value * geometry_mask,
        )

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        value_f = value.float()
        return value_f / (
            value_f.square().sum(dim=-1, keepdim=True) + float(norm_floor) ** 2
        ).sqrt()

    @classmethod
    def _temporal_posterior_band_metrics(
        cls,
        posterior: Tensor,
        interval_support: Tensor,
    ) -> dict[str, Tensor]:
        """Describe the physical-I posterior without removing T or type first.

        ``posterior`` is the already executed terminal measure
        ``[B,T,Q,I,type]``.  Metrics are conditional on a sample having at
        least one observable interval for the requested type; the companion
        support fraction distinguishes an unavailable type from a real
        interval-0 selection.  No value is fed back into the forward path.
        """

        if posterior.ndim != 5:
            raise ValueError("P2 temporal posterior must be [B,T,Q,I,type]")
        batch, horizon, basis, intervals, types = posterior.shape
        if horizon != ACTION_BAND_ENDS[-1]:
            raise ValueError("P2 temporal diagnostics require the 24-row action horizon")
        if intervals != 4 or types != len(cls.TYPE_NAMES):
            raise ValueError("P2 temporal diagnostics require four intervals and two types")
        if tuple(interval_support.shape) != (batch, intervals, types):
            raise ValueError("P2 temporal support must be [B,I,type]")
        if interval_support.dtype != torch.bool:
            raise TypeError("P2 temporal support must be boolean")
        if batch < 1 or basis < 1:
            raise ValueError("P2 temporal diagnostics require non-empty batch and basis axes")

        posterior_f = posterior.detach().float()
        type_support = interval_support.any(dim=1)
        interval_index = torch.arange(
            intervals,
            device=posterior.device,
            dtype=torch.float32,
        )
        metrics: dict[str, Tensor] = {}
        band_distributions: dict[str, list[Tensor]] = {
            name: [] for name in cls.TYPE_NAMES
        }
        start = 0
        for end in ACTION_BAND_ENDS:
            band_name = f"{start + 1}_{end}"
            band_rows = end - start
            for type_index, type_name in enumerate(cls.TYPE_NAMES):
                active = type_support[:, type_index]
                denominator = active.float().sum() * float(band_rows * basis)
                mass = posterior_f[
                    :, start:end, :, :, type_index
                ].sum(dim=(0, 1, 2)) / denominator.clamp_min(1.0)
                band_distributions[type_name].append(mass)
                for interval_index_value in range(intervals):
                    metrics[
                        f"object_p2_{type_name}_band_{band_name}_interval_"
                        f"{interval_index_value}_mass"
                    ] = mass[interval_index_value]
                metrics[
                    f"object_p2_{type_name}_band_{band_name}_expected_interval"
                ] = (mass * interval_index).sum()
            start = end

        for type_index, type_name in enumerate(cls.TYPE_NAMES):
            metrics[f"object_p2_{type_name}_temporal_support_fraction"] = (
                type_support[:, type_index].detach().float().mean()
            )
            distributions = band_distributions[type_name]
            # One compact differentiation scalar: mean total variation over
            # the three unordered pairs of action-horizon bands.
            pairwise_total_variation = [
                0.5 * (distributions[left] - distributions[right]).abs().sum()
                for left in range(len(distributions))
                for right in range(left + 1, len(distributions))
            ]
            metrics[
                f"object_p2_{type_name}_band_pair_total_variation"
            ] = torch.stack(pairwise_total_variation).mean()
        return metrics

    @staticmethod
    def _covariance_aware_distance(delta: Tensor, covariance: Tensor) -> Tensor:
        """Retain the active I+Sigma positive-definite coordinate metric."""

        covariance_f = covariance.float()
        metric_xx = 1.0 + covariance_f[..., 0]
        metric_xy = covariance_f[..., 1]
        metric_yy = 1.0 + covariance_f[..., 2]
        determinant = metric_xx * metric_yy - metric_xy.square()
        dx, dy = delta[..., 0], delta[..., 1]
        return (
            metric_yy * dx.square()
            - 2.0 * metric_xy * dx * dy
            + metric_xx * dy.square()
        ) / determinant

    def _target_fact_spatial_select(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        action_condition: WorldActionCondition,
        collect_diagnostics: bool,
    ) -> tuple[SelectedIntervalEvidence, dict[str, Tensor]]:
        """Read W while preserving TargetFact's K marginal exactly.

        Semantic evidence has no camera axis and is summed directly under the
        one target posterior. Geometry may choose a camera conditionally inside
        each K, but neither action compatibility nor physical readability may
        transfer target mass to another object.
        """

        if intent.target_posterior is None or intent.target_support is None:
            raise ValueError(
                "TargetFact P2 requires the direct target posterior and producer support"
            )
        batch, horizon, basis, _ = action_query.shape
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        posterior = intent.target_posterior.float()
        if tuple(posterior.shape) != (batch, objects):
            raise ValueError("TargetFact P2 posterior must be [B,K]")
        if intent.target_posterior.dtype != torch.float32:
            raise TypeError("TargetFact P2 posterior must remain FP32")
        producer_support = intent.target_support
        if tuple(producer_support.shape) != (batch, objects):
            raise ValueError("TargetFact P2 producer support must be [B,K]")
        if producer_support.dtype != torch.bool:
            raise TypeError("TargetFact P2 producer support must be boolean")
        if not bool(torch.isfinite(posterior).all()):
            raise ValueError("TargetFact P2 posterior must be finite")
        if bool((posterior < 0.0).any().item()) or bool(
            (posterior.masked_select(~producer_support) != 0.0).any().item()
        ):
            raise ValueError("TargetFact P2 posterior violates producer support")
        producer_count = producer_support.float().sum(dim=-1, keepdim=True)
        expected_mass = (producer_count > 0.0).float()
        if not bool(
            torch.allclose(
                posterior.sum(dim=-1, keepdim=True),
                expected_mass,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        ):
            raise ValueError("TargetFact P2 posterior must carry one supported target")

        # Identity and physical readability are separate producer-owned
        # quantities.  Keep the normalized posterior for identity/scene
        # routing, while action-facing physical values consume the continuous
        # p*validity masses exported by S.  Compatibility fixtures may omit
        # the new fields; in that case the legacy posterior is the exact
        # fallback rather than silently changing their ABI.
        physical_object_mass = (
            posterior
            if intent.target_physical_mass is None
            else intent.target_physical_mass.float()
        )
        if tuple(physical_object_mass.shape) != (batch, objects):
            raise ValueError("TargetFact P2 physical object mass must be [B,K]")
        if not bool(torch.isfinite(physical_object_mass).all()) or bool(
            ((physical_object_mass < 0.0) | (physical_object_mass > 1.0)).any()
        ):
            raise ValueError("TargetFact P2 physical object mass is invalid")
        if bool((physical_object_mass.masked_select(~producer_support) != 0.0).any()):
            raise ValueError("TargetFact P2 physical object mass violates producer support")
        if bool((physical_object_mass > posterior + 1.0e-6).any()):
            raise ValueError("TargetFact P2 physical object mass exceeds identity mass")
        physical_camera_mass = intent.target_camera_mass
        if physical_camera_mass is not None:
            physical_camera_mass = physical_camera_mass.float()
            if physical_camera_mass.ndim != 3 or tuple(
                physical_camera_mass.shape[:2]
            ) != (batch, objects):
                raise ValueError("TargetFact P2 physical camera mass must be [B,K,C]")
            if not bool(torch.isfinite(physical_camera_mass).all()) or bool(
                ((physical_camera_mass < 0.0) | (physical_camera_mass > 1.0)).any()
            ):
                raise ValueError("TargetFact P2 physical camera mass is invalid")
            if int(physical_camera_mass.shape[-1]) != int(
                dynamics.camera_chart_availability.shape[-2]
            ):
                raise ValueError(
                    "TargetFact P2 physical camera mass has the wrong camera axis"
                )
            if bool(
                (physical_camera_mass > posterior[:, :, None] + 1.0e-6).any()
            ):
                raise ValueError(
                    "TargetFact P2 physical camera mass exceeds identity mass"
                )

        # ``scene_posterior`` is not another learned selector.  It is the
        # complement expectation for one uncertain primary target over the
        # same producer-owned K domain.  The fixed n-1 denominator prevents a
        # consumer with missing W evidence from reallocating that mass to a
        # different object.  With fewer than two legal objects there is no
        # non-target scene lane.
        scene_denominator = (producer_count - 1.0).clamp_min(1.0)
        scene_posterior = torch.where(
            producer_count >= 2.0,
            producer_support.float() * (1.0 - posterior) / scene_denominator,
            torch.zeros_like(posterior),
        )

        object_support = dynamics.chart_availability[..., 0].float() > 0.0
        camera_support = (
            object_support[..., None]
            & (dynamics.camera_chart_availability[..., 0].float() > 0.0)
        )
        if self._eval_intervention in self._TARGET_ADDRESS_NEUTRAL_MODES:
            if self.training:
                raise ValueError("P2 interventions are evaluation-only")
            posterior = torch.where(
                producer_support,
                producer_support.float() / producer_count.clamp_min(1.0),
                torch.zeros_like(posterior),
            )
            scene_posterior = torch.where(
                producer_count >= 2.0,
                producer_support.float() * (1.0 - posterior) / scene_denominator,
                torch.zeros_like(posterior),
            )

        def supported(value: Tensor, support: Tensor, name: str) -> Tensor:
            expanded = support
            while expanded.ndim < value.ndim:
                expanded = expanded.unsqueeze(-1)
            if collect_diagnostics and bool(
                (~torch.isfinite(value) & expanded).any().item()
            ):
                raise ValueError(f"TargetFact P2 {name} is non-finite on support")
            return torch.where(
                expanded,
                value,
                torch.zeros_like(value),
            )

        semantic_full = supported(
            dynamics.semantic_delta,
            object_support[:, None],
            "semantic value",
        )
        semantic_common = supported(
            dynamics.semantic_common,
            object_support,
            "semantic common value",
        )
        semantic_residual = supported(
            dynamics.semantic_interval_innovation,
            object_support[:, None],
            "semantic residual value",
        )
        semantic_key = self._bounded_unit(
            self.source_key[0](semantic_full)
        )
        semantic_typed_route = (
            intent.typed_common_value[..., self.S_TYPE_INDEX_BY_P2[0], :][
                :, None
            ]
            + intent.typed_interval_residual_value[
                ..., self.S_TYPE_INDEX_BY_P2[0], :
            ]
        )
        # Quarantine producer-invalid S rows before the learned projection.
        # Masking only its output is forward-safe but leaves a 0*NaN VJP in
        # the Linear weight gradient.
        semantic_typed_route = supported(
            semantic_typed_route,
            object_support[:, None],
            "semantic S context",
        )
        semantic_typed = self.typed_intent_key[0](semantic_typed_route).float()
        semantic_selected_key = torch.einsum(
            "bk,bikh->bih", posterior, semantic_key.float()
        )
        semantic_selected_common = torch.einsum(
            "bk,bkv->bv", physical_object_mass, semantic_common.float()
        )[:, None].expand(-1, intervals, -1)
        semantic_selected_residual = torch.einsum(
            "bk,bikv->biv", physical_object_mass, semantic_residual.float()
        )
        semantic_selected_typed = torch.einsum(
            "bk,bikh->bih", physical_object_mass, semantic_typed.float()
        )
        semantic_scene_key = torch.einsum(
            "bk,bikh->bih", scene_posterior, semantic_key.float()
        )
        semantic_scene_common = torch.einsum(
            "bk,bkv->bv", scene_posterior, semantic_common.float()
        )[:, None].expand(-1, intervals, -1)
        semantic_scene_residual = torch.einsum(
            "bk,bikv->biv", scene_posterior, semantic_residual.float()
        )
        semantic_scene_typed = torch.einsum(
            "bk,bikh->bih", scene_posterior, semantic_typed.float()
        )

        temperature = self._temperatures().to(device=action_query.device)
        geometry_query = self._bounded_unit(self.source_query[1](action_query))
        transport_full = supported(
            dynamics.transport_mean,
            camera_support[:, None],
            "geometry value",
        )
        geometry_key = self._bounded_unit(
            self.source_key[1](transport_full)
        )
        geometry_source_score = torch.einsum(
            "btqh,bikch->btqikc",
            geometry_query,
            geometry_key,
        )
        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        camera_coordinate = supported(
            dynamics.camera_coordinates,
            camera_support,
            "current camera coordinate",
        )
        transport_covariance = supported(
            dynamics.transport_covariance,
            camera_support[:, None],
            "transport covariance",
        )
        future_coordinate = (
            camera_coordinate[:, None].float() + transport_full.float()
        ).clamp(-1.0, 1.0)
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )
        coordinate_distance = self._covariance_aware_distance(
            coordinate_delta,
            transport_covariance[:, None, None],
        )
        coordinate_score = (-0.25 * coordinate_distance).clamp(-1.0, 0.0)
        camera_log_measure = torch.where(
            camera_support,
            dynamics.log_camera_chart_availability[..., 0].float(),
            torch.zeros_like(dynamics.camera_chart_availability[..., 0].float()),
        )
        camera_logit = (
            temperature[0] * geometry_source_score
            + temperature[2] * coordinate_score
            + camera_log_measure[:, None, None, None]
        )
        camera_posterior = _safe_masked_softmax(
            camera_logit,
            camera_support[:, None, None, None].expand_as(camera_logit),
            dim=-1,
        )
        if physical_camera_mass is None:
            physical_camera_mass = posterior[:, :, None].expand(
                -1, -1, int(camera_posterior.shape[-1])
            )
        joint_geometry_weight = (
            physical_camera_mass[:, None, None, None, :, :] * camera_posterior
        )
        scene_joint_geometry_weight = (
            scene_posterior[:, None, None, None, :, None] * camera_posterior
        )
        geometry_selected_key = torch.einsum(
            "btqikc,bikch->btqih",
            joint_geometry_weight,
            geometry_key.float(),
        )
        transport_common = supported(
            dynamics.transport_common,
            camera_support,
            "geometry common value",
        )
        transport_residual = supported(
            dynamics.transport_interval_innovation,
            camera_support[:, None],
            "geometry residual value",
        )
        geometry_selected_common = torch.einsum(
            "btqikc,bkcv->btqiv",
            joint_geometry_weight,
            transport_common.float(),
        )
        geometry_selected_residual = torch.einsum(
            "btqikc,bikcv->btqiv",
            joint_geometry_weight,
            transport_residual.float(),
        )
        geometry_scene_key = torch.einsum(
            "btqikc,bikch->btqih",
            scene_joint_geometry_weight,
            geometry_key.float(),
        )
        geometry_scene_common = torch.einsum(
            "btqikc,bkcv->btqiv",
            scene_joint_geometry_weight,
            transport_common.float(),
        )
        geometry_scene_residual = torch.einsum(
            "btqikc,bikcv->btqiv",
            scene_joint_geometry_weight,
            transport_residual.float(),
        )
        geometry_typed_route = (
            intent.typed_common_value[..., self.S_TYPE_INDEX_BY_P2[1], :][
                :, None
            ]
            + intent.typed_interval_residual_value[
                ..., self.S_TYPE_INDEX_BY_P2[1], :
            ]
        )
        geometry_typed_route = supported(
            geometry_typed_route,
            camera_support.any(dim=-1)[:, None],
            "geometry S context",
        )
        geometry_typed = self.typed_intent_key[1](geometry_typed_route).float()
        geometry_selected_typed = torch.einsum(
            "bk,bikh->bih", physical_object_mass, geometry_typed
        )
        geometry_scene_typed = torch.einsum(
            "bk,bikh->bih", scene_posterior, geometry_typed
        )

        def expand_interval(value: Tensor) -> Tensor:
            return value[:, None, None].expand(-1, horizon, basis, -1, -1)

        semantic_key_expanded = expand_interval(semantic_selected_key)
        semantic_common_expanded = expand_interval(semantic_selected_common)
        semantic_residual_expanded = expand_interval(semantic_selected_residual)
        semantic_typed_expanded = expand_interval(semantic_selected_typed)
        semantic_scene_key_expanded = expand_interval(semantic_scene_key)
        semantic_scene_common_expanded = expand_interval(semantic_scene_common)
        semantic_scene_residual_expanded = expand_interval(semantic_scene_residual)
        semantic_scene_typed_expanded = expand_interval(semantic_scene_typed)
        if self.action_interval_key is None:
            raise RuntimeError("target-action P2 has no physical interval key")
        if isinstance(action_condition, PhysicalActionSequenceCondition):
            physical_intervals = PhysicalActionCondition.from_horizon_action(
                action_condition.source_action,
                action_condition.source_action[:, 0],
            ).interval_action
        else:
            physical_intervals = action_condition.source_interval_action
        if tuple(physical_intervals.shape[:2]) != (batch, intervals):
            raise ValueError("target-action P2 physical intervals lost the W ABI")
        public_interval = self.action_interval_key(physical_intervals).float()
        public_expanded = public_interval[:, None, None].expand(
            -1, horizon, basis, -1, -1
        )
        geometry_typed_expanded = expand_interval(geometry_selected_typed)
        geometry_scene_typed_expanded = expand_interval(geometry_scene_typed)
        semantic_resolved = (
            physical_object_mass * object_support.float()
        ).sum(dim=-1)
        geometry_resolved = (
            physical_camera_mass * camera_support.float()
        ).sum(dim=(-1, -2))
        semantic_scene_resolved = (
            scene_posterior * object_support.float()
        ).sum(dim=-1)
        geometry_scene_resolved = (
            scene_posterior * camera_support.any(dim=-1).float()
        ).sum(dim=-1)
        semantic_interval_support = (semantic_resolved > 0.0)[:, None].expand(
            -1, intervals
        )
        geometry_interval_support = (geometry_resolved > 0.0)[:, None].expand(
            -1, intervals
        )
        semantic_scene_interval_support = (
            semantic_scene_resolved > 0.0
        )[:, None].expand(-1, intervals)
        geometry_scene_interval_support = (
            geometry_scene_resolved > 0.0
        )[:, None].expand(-1, intervals)

        selected = SelectedIntervalEvidence(
            key=torch.stack(
                (semantic_key_expanded, geometry_selected_key), dim=4
            ),
            scene_key=torch.stack(
                (semantic_scene_key_expanded, geometry_scene_key), dim=4
            ),
            semantic_value=semantic_common_expanded + semantic_residual_expanded,
            semantic_common_value=semantic_common_expanded,
            semantic_residual_value=semantic_residual_expanded,
            geometry_value=geometry_selected_common + geometry_selected_residual,
            geometry_common_value=geometry_selected_common,
            geometry_residual_value=geometry_selected_residual,
            semantic_scene_value=(
                semantic_scene_common_expanded + semantic_scene_residual_expanded
            ),
            semantic_scene_common_value=semantic_scene_common_expanded,
            semantic_scene_residual_value=semantic_scene_residual_expanded,
            geometry_scene_value=(
                geometry_scene_common + geometry_scene_residual
            ),
            geometry_scene_common_value=geometry_scene_common,
            geometry_scene_residual_value=geometry_scene_residual,
            selected_s_context=torch.stack(
                (
                    public_expanded + semantic_typed_expanded,
                    public_expanded + geometry_typed_expanded,
                ),
                dim=4,
            ),
            scene_s_context=torch.stack(
                (semantic_scene_typed_expanded, geometry_scene_typed_expanded),
                dim=4,
            ),
            support=torch.stack(
                (semantic_interval_support, geometry_interval_support), dim=-1
            ),
            scene_support=torch.stack(
                (
                    semantic_scene_interval_support,
                    geometry_scene_interval_support,
                ),
                dim=-1,
            ),
        )
        selected.validate()
        if not collect_diagnostics:
            return selected, {}

        gradient_metrics: dict[str, Tensor] = {}
        if self.training:
            register_gradient_rms_metric(
                intent.target_posterior,
                gradient_metrics,
                "gradient_tensor_p2_target_posterior_rms",
            )
        geometry_k_marginal = joint_geometry_weight.sum(dim=-1)
        scene_geometry_k_marginal = scene_joint_geometry_weight.sum(dim=-1)
        expected_geometry_k = (
            (physical_camera_mass * camera_support.float()).sum(dim=-1)
            [:, None, None, None]
        )
        expected_scene_geometry_k = (
            scene_posterior[:, None, None, None]
            * camera_support.any(dim=-1).float()[:, None, None, None]
        )
        return selected, {
            **gradient_metrics,
            "object_p2_target_address_mode_enabled": action_query.new_ones(
                (), dtype=torch.float32
            ),
            "object_p2_target_identity_entropy": normalized_entropy(
                posterior, dim=-1
            ).detach().mean(),
            "object_p2_target_semantic_effective_mass": semantic_resolved.detach().mean(),
            "object_p2_target_geometry_effective_mass": geometry_resolved.detach().mean(),
            "object_p2_scene_semantic_effective_mass": semantic_scene_resolved.detach().mean(),
            "object_p2_scene_geometry_effective_mass": geometry_scene_resolved.detach().mean(),
            "object_p2_target_unresolved_mass": (
                1.0 - torch.minimum(semantic_resolved, geometry_resolved)
            ).detach().mean(),
            "object_p2_target_geometry_k_marginal_error": (
                geometry_k_marginal - expected_geometry_k
            ).detach().abs().amax(),
            "object_p2_scene_geometry_k_marginal_error": (
                scene_geometry_k_marginal - expected_scene_geometry_k
            ).detach().abs().amax(),
            "object_p2_scene_identity_entropy": normalized_entropy(
                scene_posterior, dim=-1
            ).detach().mean(),
            "object_p2_geometry_camera_conditional_entropy": normalized_entropy(
                camera_posterior, dim=-1
            ).detach().mean(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_selected_w_key_rms": selected.key.detach().float().square().mean().sqrt(),
            "object_p2_selected_s_context_rms": selected.selected_s_context.detach().float().square().mean().sqrt(),
            "object_p2_scene_selected_w_key_rms": selected.scene_key.detach().float().square().mean().sqrt(),
            "object_p2_scene_selected_context_rms": selected.scene_s_context.detach().float().square().mean().sqrt(),
            "object_p2_semantic_selected_candidate_rms": selected.semantic_value.detach().float().square().mean().sqrt(),
            "object_p2_geometry_selected_physical_rms": selected.geometry_value.detach().float().square().mean().sqrt(),
            "object_p2_scene_semantic_selected_candidate_rms": selected.semantic_scene_value.detach().float().square().mean().sqrt(),
            "object_p2_scene_geometry_selected_physical_rms": selected.geometry_scene_value.detach().float().square().mean().sqrt(),
        }

    def spatial_select(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        action_condition: WorldActionCondition | None = None,
        collect_diagnostics: bool,
    ) -> tuple[SelectedIntervalEvidence, dict[str, Tensor]]:
        dynamics.validate()
        if action_query.ndim != 4:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        batch, horizon, basis, hidden = action_query.shape
        if hidden != self.hidden:
            raise ValueError("P2 action query hidden width is invalid")
        intent.validate(horizon=horizon, hidden=self.hidden)
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        if tuple(intent.typed_common_value.shape[:3]) != (batch, objects, 3):
            raise ValueError("P2 typed common S metadata lost K/type")
        if tuple(intent.typed_interval_residual_value.shape[:4]) != (
            batch,
            intervals,
            objects,
            3,
        ):
            raise ValueError("P2 typed residual S metadata lost I/K/type")
        if tuple(intent.target_object_address_logit.shape) != (
            batch,
            intervals,
            objects,
        ):
            raise ValueError("P2 target object address lost I/K identity")
        if intent.target_object_address_logit.dtype != torch.float32:
            raise TypeError("P2 target object address must remain FP32")

        if self.spatial_intent_mode == "target_action_bottleneck_v1":
            if action_condition is None:
                raise ValueError("target-action P2 requires the physical A0 condition")
            return self._target_fact_spatial_select(
                action_query,
                dynamics,
                intent,
                action_condition=action_condition,
                collect_diagnostics=collect_diagnostics,
            )

        temperature = self._temperatures().to(device=action_query.device)
        object_measure = dynamics.chart_availability[..., 0].float().clamp(0.0, 1.0)
        semantic_support = object_measure > 0.0
        semantic_log_measure = torch.where(
            semantic_support,
            dynamics.log_chart_availability[..., 0].float(),
            torch.zeros_like(object_measure),
        )
        camera_measure = dynamics.camera_chart_availability[..., 0].float().clamp(
            0.0, 1.0
        )
        camera_support = semantic_support[..., None] & (camera_measure > 0.0)
        camera_log_measure = torch.where(
            camera_support,
            dynamics.log_camera_chart_availability[..., 0].float(),
            torch.zeros_like(camera_measure),
        )
        conditional_camera, conditional_camera_log = _safe_masked_log_softmax(
            camera_log_measure,
            camera_support,
            dim=-1,
        )
        geometry_log_measure = torch.where(
            camera_support,
            semantic_log_measure[..., None] + conditional_camera_log,
            torch.zeros_like(conditional_camera),
        )

        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        current_coordinate = dynamics.camera_coordinates[:, None].float().clamp(
            -1.0, 1.0
        )
        future_coordinate = (
            dynamics.camera_coordinates[:, None].float()
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )
        coordinate_distance = self._covariance_aware_distance(
            coordinate_delta,
            dynamics.transport_covariance[:, None, None],
        )
        coordinate_score = (-0.25 * coordinate_distance).clamp(-1.0, 0.0)
        current_coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - current_coordinate[:, None, None]
        )
        current_coordinate_distance = self._covariance_aware_distance(
            current_coordinate_delta,
            dynamics.transport_covariance[:, None, None],
        )
        current_coordinate_score = (-0.25 * current_coordinate_distance).clamp(
            -1.0, 0.0
        )

        # Geometry contributes to semantic K selection only through the
        # transport-specific change in coordinate compatibility.  Current
        # position is subtracted under the same covariance, then legal cameras
        # are aggregated with their producer-owned conditional measure.  A
        # valid-K-centred correction cannot create support, interval votes or
        # semantic value amplitude.  Subtracting the first legal K value before
        # the mean makes a K-uniform row exact zero while preserving ordinary
        # gradients for non-uniform perturbations at the zero boundary.
        camera_weight = conditional_camera[:, None, None, None]
        address_change = (
            (coordinate_score - current_coordinate_score) * camera_weight
        ).sum(dim=-1)
        semantic_interval_support = semantic_support[:, None].expand(
            -1, intervals, -1
        )
        first_legal_k = semantic_interval_support.to(dtype=torch.int64).argmax(
            dim=-1, keepdim=True
        )
        address_reference = address_change.gather(
            -1,
            first_legal_k[:, None, None].expand(
                batch, horizon, basis, intervals, 1
            ),
        )
        address_relative = address_change - address_reference
        expanded_semantic_support = semantic_interval_support[:, None, None]
        legal_k_count = expanded_semantic_support.float().sum(
            dim=-1, keepdim=True
        )
        legal_k_mean = (
            torch.where(
                expanded_semantic_support,
                address_relative,
                torch.zeros_like(address_relative),
            ).sum(dim=-1, keepdim=True)
            / legal_k_count.clamp_min(1.0)
        )
        geometry_address_centered = torch.where(
            expanded_semantic_support,
            address_relative - legal_k_mean,
            torch.zeros_like(address_relative),
        )
        geometry_address_correction_primary = torch.tanh(
            geometry_address_centered
        )
        geometry_address_correction = geometry_address_correction_primary
        if self._eval_intervention in self._GEOMETRY_ADDRESS_NEUTRAL_MODES:
            if self.training:
                raise ValueError("P2 interventions are evaluation-only")
            geometry_address_correction = torch.zeros_like(
                geometry_address_correction_primary
            )

        if self.spatial_intent_mode == "target_action_bottleneck_v1":
            if intent.target_log_prior is None:
                raise ValueError(
                    "TargetFact P2 requires the direct target_log_prior field"
                )
            target_address_primary = intent.target_log_prior[:, None, :].expand(
                -1, intervals, -1
            ).float()
            compatibility_address = intent.target_object_address_logit.float()
            if not torch.equal(target_address_primary, compatibility_address):
                raise ValueError(
                    "TargetFact compatibility target address diverged from target_log_prior"
                )
        else:
            target_address_primary = intent.target_object_address_logit.float()
        if self.spatial_intent_mode in {
            "shared_target_prior_v1",
            "target_action_bottleneck_v1",
        }:
            invalid_supported_target = semantic_interval_support & ~torch.isfinite(
                target_address_primary
            )
            if bool(invalid_supported_target.any().item()):
                raise ValueError(
                    "P2 target object address contains non-finite values on supported K"
                )
            target_address_condition = torch.where(
                semantic_interval_support,
                target_address_primary,
                torch.zeros_like(target_address_primary),
            )
        else:
            target_address_condition = torch.zeros_like(target_address_primary)
        if self._eval_intervention in self._TARGET_ADDRESS_NEUTRAL_MODES:
            if self.training:
                raise ValueError("P2 interventions are evaluation-only")
            target_address_condition = torch.zeros_like(target_address_primary)

        common_fields = (
            dynamics.semantic_common,
            dynamics.transport_common,
        )
        residual_fields = (
            dynamics.semantic_interval_innovation,
            dynamics.transport_interval_innovation,
        )
        full_fields = (dynamics.semantic_delta, dynamics.transport_mean)
        if self.public_interval_key is None:
            raise RuntimeError("legacy P2 has no public interval key")
        public_interval = self.public_interval_key(intent.interval_key).float()

        selected_keys: list[Tensor] = []
        selected_common_values: list[Tensor] = []
        selected_residual_values: list[Tensor] = []
        selected_s_contexts: list[Tensor] = []
        interval_supports: list[Tensor] = []
        spatial_posteriors: list[Tensor] = []
        source_scores: list[Tensor] = []
        spatial_query_by_type = torch.stack(
            [
                self._bounded_unit(self.source_query[index](action_query))
                for index in range(len(self.TYPE_NAMES))
            ],
            dim=3,
        )
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics and self.training:
            register_gradient_axis_rms_metrics(
                spatial_query_by_type,
                gradient_metrics,
                (
                    "gradient_tensor_p2_semantic_spatial_query_rms",
                    "gradient_tensor_p2_geometry_spatial_query_rms",
                ),
                dim=3,
            )
            register_gradient_rms_metric(
                geometry_address_correction,
                gradient_metrics,
                "gradient_tensor_p2_geometry_address_correction_rms",
            )
            if self.spatial_intent_mode in {
                "shared_target_prior_v1",
                "target_action_bottleneck_v1",
            }:
                register_gradient_rms_metric(
                    target_address_condition,
                    gradient_metrics,
                    "gradient_tensor_p2_target_object_address_logit_rms",
                )

        for type_index in range(len(self.TYPE_NAMES)):
            query = spatial_query_by_type.select(3, type_index)
            source_key = self._bounded_unit(
                self.source_key[type_index](full_fields[type_index])
            )
            typed_route = (
                intent.typed_common_value[..., self.S_TYPE_INDEX_BY_P2[type_index], :][
                    :, None
                ]
                + intent.typed_interval_residual_value[
                    ..., self.S_TYPE_INDEX_BY_P2[type_index], :
                ]
            )
            typed_candidate = self.typed_intent_key[type_index](typed_route).float()

            if type_index == 0:
                source_score = torch.einsum(
                    "btqh,bikh->btqik",
                    query,
                    source_key,
                )
                support = semantic_support[:, None].expand(-1, intervals, -1)
                log_measure = semantic_log_measure[:, None].expand_as(support)
                candidate_logit = temperature[0] * source_score + log_measure[
                    :, None, None
                ] + geometry_address_correction + target_address_condition[
                    :, None, None
                ]
                candidate_support = support[:, None, None].expand_as(candidate_logit)
                candidate_typed = typed_candidate
            else:
                source_score = torch.einsum(
                    "btqh,bikch->btqikc",
                    query,
                    source_key,
                )
                support = camera_support[:, None].expand(
                    -1, intervals, -1, -1
                )
                log_measure = geometry_log_measure[:, None].expand_as(support)
                candidate_logit = (
                    temperature[0] * source_score
                    + temperature[2] * coordinate_score
                    + log_measure[:, None, None]
                    + target_address_condition[:, None, None, :, :, None]
                )
                candidate_support = support[:, None, None].expand_as(candidate_logit)
                candidate_typed = typed_candidate[..., None, :].expand_as(source_key)

            flat_logit = candidate_logit.reshape(batch, horizon, basis, intervals, -1)
            flat_support = candidate_support.reshape_as(flat_logit)
            posterior = _safe_masked_softmax(flat_logit, flat_support, dim=-1)
            interval_support = support.flatten(start_dim=2).any(dim=-1)
            flat_key = source_key.reshape(batch, intervals, -1, self.hidden)
            flat_typed = candidate_typed.reshape(batch, intervals, -1, self.hidden)
            value_width = int(full_fields[type_index].shape[-1])
            flat_common = common_fields[type_index].reshape(
                batch, -1, value_width
            )[:, None].expand(-1, intervals, -1, -1)
            flat_residual = residual_fields[type_index].reshape(
                batch,
                intervals,
                -1,
                value_width,
            )
            posterior_for_key = posterior.to(dtype=flat_key.dtype)
            selected_keys.append(
                torch.einsum("btqin,binh->btqih", posterior_for_key, flat_key)
            )
            selected_common_values.append(
                torch.einsum(
                    "btqin,binv->btqiv",
                    posterior.to(dtype=flat_common.dtype),
                    flat_common,
                )
            )
            selected_residual_values.append(
                torch.einsum(
                    "btqin,binv->btqiv",
                    posterior.to(dtype=flat_residual.dtype),
                    flat_residual,
                )
            )
            selected_typed = torch.einsum(
                "btqin,binh->btqih",
                posterior.to(dtype=flat_typed.dtype),
                flat_typed,
            )
            selected_s_contexts.append(
                public_interval[:, None, None] + selected_typed.float()
            )
            interval_supports.append(interval_support)
            spatial_posteriors.append(posterior)
            source_scores.append(source_score)

        selected = SelectedIntervalEvidence(
            key=torch.stack(selected_keys, dim=4),
            scene_key=torch.zeros_like(torch.stack(selected_keys, dim=4)),
            semantic_value=(
                selected_common_values[0] + selected_residual_values[0]
            ),
            semantic_common_value=selected_common_values[0],
            semantic_residual_value=selected_residual_values[0],
            geometry_value=(
                selected_common_values[1] + selected_residual_values[1]
            ),
            geometry_common_value=selected_common_values[1],
            geometry_residual_value=selected_residual_values[1],
            semantic_scene_value=torch.zeros_like(
                selected_common_values[0] + selected_residual_values[0]
            ),
            semantic_scene_common_value=torch.zeros_like(selected_common_values[0]),
            semantic_scene_residual_value=torch.zeros_like(selected_residual_values[0]),
            geometry_scene_value=torch.zeros_like(
                selected_common_values[1] + selected_residual_values[1]
            ),
            geometry_scene_common_value=torch.zeros_like(selected_common_values[1]),
            geometry_scene_residual_value=torch.zeros_like(selected_residual_values[1]),
            selected_s_context=torch.stack(selected_s_contexts, dim=4),
            scene_s_context=torch.zeros_like(torch.stack(selected_s_contexts, dim=4)),
            support=torch.stack(interval_supports, dim=-1),
            scene_support=torch.zeros_like(
                torch.stack(interval_supports, dim=-1), dtype=torch.bool
            ),
        )
        selected.validate()
        if not collect_diagnostics:
            return selected, {}
        return selected, {
            **gradient_metrics,
            "object_p2_content_score_abs": torch.stack(
                [score.detach().float().abs().mean() for score in source_scores]
            ).mean(),
            "object_p2_content_score_max_abs": torch.stack(
                [score.detach().float().abs().amax() for score in source_scores]
            ).amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_geometry_address_correction_rms": (
                geometry_address_correction_primary.detach()
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "object_p2_geometry_address_correction_max_abs": (
                geometry_address_correction_primary.detach().float().abs().amax()
            ),
            "object_p2_geometry_address_k_center_error": (
                (
                    geometry_address_centered.detach().float()
                    * expanded_semantic_support.detach().float()
                ).sum(dim=-1)
                / legal_k_count.detach().squeeze(-1).clamp_min(1.0)
            )
            .abs()
            .amax(),
            "object_p2_target_address_mode_enabled": action_query.new_tensor(
                float(
                    self.spatial_intent_mode
                    in {
                        "shared_target_prior_v1",
                        "target_action_bottleneck_v1",
                    }
                ),
                dtype=torch.float32,
            ),
            "object_p2_target_address_logit_rms": target_address_primary.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_target_address_active_rms": target_address_condition.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_target_address_k_center_error": target_address_primary.detach()
            .sum(dim=-1)
            .abs()
            .amax(),
            "object_p2_spatial_posterior_entropy": torch.stack(
                [normalized_entropy(value, dim=-1).detach().mean() for value in spatial_posteriors]
            ).mean(),
            "object_p2_spatial_posterior_max": torch.stack(
                [value.detach().amax(dim=-1).mean() for value in spatial_posteriors]
            ).mean(),
            "object_p2_spatial_selector_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_selected_w_key_rms": selected.key.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_s_context_rms": selected.selected_s_context.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_semantic_selected_candidate_rms": selected.semantic_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_selected_physical_rms": selected.geometry_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_spatial_common_residual_identity_error": torch.maximum(
                (
                    selected.semantic_value.detach().float()
                    - (
                        selected.semantic_common_value.detach()
                        + selected.semantic_residual_value.detach()
                    ).float()
                )
                .abs()
                .amax(),
                (
                    selected.geometry_value.detach().float()
                    - (
                        selected.geometry_common_value.detach()
                        + selected.geometry_residual_value.detach()
                    ).float()
                )
                .abs()
                .amax(),
            ),
        }

    def temporal_terminal(
        self,
        action_query: Tensor,
        selected: SelectedIntervalEvidence,
        *,
        collect_diagnostics: bool,
    ) -> tuple[ObjectEffectBundle, dict[str, Tensor]]:
        selected.validate()
        batch, horizon, basis, intervals, types, hidden = selected.key.shape
        if tuple(action_query.shape) != (batch, horizon, basis, hidden):
            raise ValueError("P2 terminal action query lost [B,T,Q,H]")
        action_by_type = torch.stack(
            [
                self._bounded_unit(self.terminal_query[index](action_query))
                for index in range(types)
            ],
            dim=3,
        )
        terminal_query_gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics and self.training:
            register_gradient_axis_rms_metrics(
                action_by_type,
                terminal_query_gradient_metrics,
                (
                    "gradient_tensor_p2_semantic_terminal_query_rms",
                    "gradient_tensor_p2_geometry_terminal_query_rms",
                ),
                dim=3,
            )
        temperature = self._temperatures().to(device=action_query.device)

        def terminal_posterior(
            key: Tensor,
            context: Tensor,
            interval_support: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            normalized_key = self._bounded_unit(key)
            normalized_context = torch.tanh(self._bounded_unit(context))
            conditioned = normalized_key + normalized_key * normalized_context
            score = torch.einsum(
                "btqzh,btqizh->btqiz",
                action_by_type,
                conditioned,
            )
            expanded_support = interval_support[:, None, None].expand_as(score)
            probability = _safe_masked_softmax(
                temperature[1] * score,
                expanded_support,
                dim=3,
            )
            return probability, score, normalized_key

        posterior, interval_score, selected_key = terminal_posterior(
            selected.key,
            selected.selected_s_context,
            selected.support,
        )
        scene_posterior, scene_interval_score, scene_selected_key = terminal_posterior(
            selected.scene_key,
            selected.scene_s_context,
            selected.scene_support,
        )
        support = selected.support[:, None, None].expand_as(interval_score)
        scene_support = selected.scene_support[:, None, None].expand_as(
            scene_interval_score
        )
        (
            semantic_common_source,
            semantic_residual_source,
            geometry_common_source,
            geometry_residual_source,
            semantic_scene_common_source,
            semantic_scene_residual_source,
            geometry_scene_common_source,
            geometry_scene_residual_source,
        ) = self._intervened_values(selected)
        semantic_posterior = posterior[..., 0]
        geometry_posterior = posterior[..., 1]
        semantic_common = torch.einsum(
            "btqi,btqid->btqd",
            semantic_posterior.to(dtype=semantic_common_source.dtype),
            semantic_common_source,
        )
        semantic_residual = torch.einsum(
            "btqi,btqid->btqd",
            semantic_posterior.to(dtype=semantic_residual_source.dtype),
            semantic_residual_source,
        )
        geometry_common = torch.einsum(
            "btqi,btqic->btqc",
            geometry_posterior.to(dtype=geometry_common_source.dtype),
            geometry_common_source,
        )
        geometry_residual = torch.einsum(
            "btqi,btqic->btqc",
            geometry_posterior.to(dtype=geometry_residual_source.dtype),
            geometry_residual_source,
        )
        scene_semantic_posterior = scene_posterior[..., 0]
        scene_geometry_posterior = scene_posterior[..., 1]
        semantic_scene_common = torch.einsum(
            "btqi,btqid->btqd",
            scene_semantic_posterior.to(dtype=semantic_scene_common_source.dtype),
            semantic_scene_common_source,
        )
        semantic_scene_residual = torch.einsum(
            "btqi,btqid->btqd",
            scene_semantic_posterior.to(dtype=semantic_scene_residual_source.dtype),
            semantic_scene_residual_source,
        )
        geometry_scene_common = torch.einsum(
            "btqi,btqic->btqc",
            scene_geometry_posterior.to(dtype=geometry_scene_common_source.dtype),
            geometry_scene_common_source,
        )
        geometry_scene_residual = torch.einsum(
            "btqi,btqic->btqc",
            scene_geometry_posterior.to(dtype=geometry_scene_residual_source.dtype),
            geometry_scene_residual_source,
        )
        semantic_selected = semantic_common + semantic_residual
        geometry_selected = geometry_common + geometry_residual
        semantic_scene_selected = semantic_scene_common + semantic_scene_residual
        geometry_scene_selected = geometry_scene_common + geometry_scene_residual
        target_effect = ObjectTypedEffect(
            semantic=self.semantic_value(semantic_selected),
            geometry=self.transport_value(geometry_selected),
        )
        semantic_scene_projection = self.semantic_scene_value
        geometry_scene_projection = self.transport_scene_value
        if semantic_scene_projection is None or geometry_scene_projection is None:
            scene_enabled = False
            scene_effect = ObjectTypedEffect(
                semantic=torch.zeros_like(target_effect.semantic),
                geometry=torch.zeros_like(target_effect.geometry),
            )
        else:
            scene_enabled = True
            scene_effect = ObjectTypedEffect(
                semantic=semantic_scene_projection(semantic_scene_selected),
                geometry=geometry_scene_projection(geometry_scene_selected),
            )
        effect = ObjectEffectBundle(
            target=target_effect,
            scene=scene_effect,
            scene_enabled=scene_enabled,
        )
        effect.validate()
        value_by_type = torch.stack(
            (target_effect.semantic, target_effect.geometry), dim=3
        )
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics and self.training:
            # Observe the exact tensors consumed by consequence.  A temporary
            # stack has no downstream consumer and would report false zeros.
            register_gradient_rms_metric(
                target_effect.semantic,
                gradient_metrics,
                "gradient_tensor_p2_semantic_effect_rms",
            )
            register_gradient_rms_metric(
                target_effect.geometry,
                gradient_metrics,
                "gradient_tensor_p2_geometry_effect_rms",
            )
            register_gradient_rms_metric(
                scene_effect.semantic,
                gradient_metrics,
                "gradient_tensor_p2_scene_semantic_effect_rms",
            )
            register_gradient_rms_metric(
                scene_effect.geometry,
                gradient_metrics,
                "gradient_tensor_p2_scene_geometry_effect_rms",
            )
        if not collect_diagnostics:
            return effect, {}
        neutral_key_score = torch.einsum(
            "btqzh,btqizh->btqiz",
            action_by_type,
            selected_key,
        )
        neutral_posterior = _safe_masked_softmax(
            temperature[1] * neutral_key_score,
            support,
            dim=3,
        )
        neutral_scene_key_score = torch.einsum(
            "btqzh,btqizh->btqiz",
            action_by_type,
            scene_selected_key,
        )
        neutral_scene_posterior = _safe_masked_softmax(
            temperature[1] * neutral_scene_key_score,
            scene_support,
            dim=3,
        )
        projection_delta_terms: list[Tensor] = []
        semantic_spatial_query_retired = action_query.new_zeros(
            (), dtype=torch.float32
        )
        for type_index, (spatial, terminal) in enumerate(zip(
            self.source_query,
            self.terminal_query,
            strict=True,
        )):
            if not isinstance(terminal, nn.Linear):
                raise TypeError("P2 query projections must remain linear")
            if (
                type_index == 0
                and self.spatial_intent_mode
                == "target_action_bottleneck_v1"
                and isinstance(spatial, nn.Identity)
            ):
                # TargetFact owns semantic K selection, so this mode has no
                # semantic spatial query to compare against the independent
                # terminal-I query.  Keep the historical finite metric ABI,
                # and expose the structural retirement explicitly.
                projection_delta_terms.append(
                    action_query.new_zeros((), dtype=torch.float32)
                )
                semantic_spatial_query_retired = action_query.new_ones(
                    (), dtype=torch.float32
                )
                continue
            if not isinstance(spatial, nn.Linear):
                raise TypeError("P2 query projections must remain linear")
            projection_delta = (
                terminal.weight.detach().float() - spatial.weight.detach().float()
            ).square().mean().sqrt()
            projection_delta_terms.append(projection_delta)
        raw_effect = target_effect.combined()
        raw_scene_effect = scene_effect.combined()
        metrics: dict[str, Tensor] = {
            "object_p2_intent_score_abs": (
                interval_score.detach() - neutral_key_score.detach()
            )
            .abs()
            .mean(),
            "object_p2_intent_score_max_abs": (
                interval_score.detach() - neutral_key_score.detach()
            )
            .abs()
            .amax(),
            "object_p2_combined_logit_max_abs": interval_score.detach().abs().amax(),
            "object_p2_scene_combined_logit_max_abs": scene_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_posterior_entropy": normalized_entropy(
                posterior.movedim(3, -1), dim=-1
            )
            .detach()
            .mean(),
            "object_p2_posterior_max": posterior.detach().amax(dim=3).mean(),
            "object_p2_null_mass": action_query.new_zeros((), dtype=torch.float32),
            "object_p2_terminal_has_null": action_query.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_s_condition_posterior_l1": (
                posterior.detach() - neutral_posterior.detach()
            )
            .abs()
            .mean(),
            "object_p2_scene_context_posterior_l1": (
                scene_posterior.detach() - neutral_scene_posterior.detach()
            )
            .abs()
            .mean(),
            "object_p2_effect_precontract_rms": raw_effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_semantic_effect_rms": value_by_type[..., 0, :]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_effect_rms": value_by_type[..., 1, :]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_scene_effect_precontract_rms": raw_scene_effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_scene_semantic_effect_rms": scene_effect.semantic.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_scene_geometry_effect_rms": scene_effect.geometry.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_semantic_terminal_selected_rms": semantic_selected.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_terminal_physical_rms": geometry_selected.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_scene_semantic_terminal_selected_rms": (
                semantic_scene_selected.detach().float().square().mean().sqrt()
            ),
            "object_p2_scene_geometry_terminal_physical_rms": (
                geometry_scene_selected.detach().float().square().mean().sqrt()
            ),
            "object_p2_semantic_value_weight_rms": self.semantic_value.weight.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_value_weight_rms": self.transport_value.weight.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_scene_semantic_value_weight_rms": (
                self.semantic_scene_value.weight.detach().float().square().mean().sqrt()
                if self.semantic_scene_value is not None
                else action_query.new_zeros((), dtype=torch.float32)
            ),
            "object_p2_scene_geometry_value_weight_rms": (
                self.transport_scene_value.weight.detach().float().square().mean().sqrt()
                if self.transport_scene_value is not None
                else action_query.new_zeros((), dtype=torch.float32)
            ),
            "object_p2_terminal_common_residual_identity_error": torch.maximum(
                (
                    semantic_selected.detach().float()
                    - (semantic_common.detach() + semantic_residual.detach()).float()
                )
                .abs()
                .amax(),
                (
                    geometry_selected.detach().float()
                    - (geometry_common.detach() + geometry_residual.detach()).float()
                )
                .abs()
                .amax(),
            ),
            "object_p2_terminal_query_delta_rms": torch.stack(
                projection_delta_terms
            ).mean(),
            "object_p2_semantic_terminal_query_delta_rms": projection_delta_terms[0],
            "object_p2_geometry_terminal_query_delta_rms": projection_delta_terms[1],
            "object_p2_semantic_spatial_query_retired": (
                semantic_spatial_query_retired
            ),
        }
        metrics.update(terminal_query_gradient_metrics)
        metrics.update(gradient_metrics)
        # The band matrix is a deployment-path validation diagnostic.  A
        # training flow-time posterior has a different conditioning point and
        # would create a misleading second semantic under the same name.
        if not self.training:
            metrics.update(
                self._temporal_posterior_band_metrics(
                    posterior,
                    selected.support,
                )
            )
        interval_mass = posterior.detach().float().mean(dim=4)
        for index in range(intervals):
            metrics[f"object_p2_interval_{index}_mass"] = interval_mass[
                ..., index
            ].mean()
        return effect, metrics

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        action_condition: WorldActionCondition | None = None,
        collect_diagnostics: bool,
    ) -> tuple[ObjectEffectBundle, dict[str, Tensor]]:
        selected, spatial_metrics = self.spatial_select(
            action_query,
            dynamics,
            intent,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        value, terminal_metrics = self.temporal_terminal(
            action_query,
            selected,
            collect_diagnostics=collect_diagnostics,
        )
        if not collect_diagnostics:
            return value, {}
        return value, {**spatial_metrics, **terminal_metrics}

    def forward_candidate(
        self,
        action_query: Tensor,
        candidate_world: CandidateWorld,
        intent: PolicyIntentDock,
        *,
        action_condition: WorldActionCondition,
        collect_diagnostics: bool,
    ) -> tuple[ObjectEffectBundle, dict[str, Tensor]]:
        """Consume an explicitly action-tagged world at the P2 boundary."""

        if action_condition is not candidate_world.action_condition:
            raise ValueError(
                "P2 refused a candidate world with a stale action fingerprint"
            )
        candidate_world.validate(action_dim=action_condition.action_dim)
        candidate_world.assert_action_identity(action_condition)
        # Route through ``__call__`` so ordinary module hooks continue to see
        # the exact P2 query used by the live consumer.  Calling ``forward``
        # directly would bypass the source-backed ingress probe.
        effect, metrics = self(
            action_query,
            candidate_world.dynamics,
            intent,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            metrics = {
                **metrics,
                "object_p2_candidate_world_action_identity_error": action_query.new_zeros(
                    (), dtype=torch.float32
                ),
                "object_p2_candidate_world_tagged": action_query.new_ones(
                    (), dtype=torch.float32
                ),
            }
        return effect, metrics


class ZeroPreservingObjectConsequence(nn.Module):
    """Exact identity when the P2 effect is zero."""

    INTERVENTION_MODES = ("effect_neutral",)

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fact = nn.Linear(hidden, hidden, bias=False)
        interaction = nn.Linear(hidden, hidden, bias=False)
        self.semantic_interaction = interaction
        # Preserve the exact old shared interaction at construction, without
        # consuming another initialization draw.  Ordinary gradients can then
        # assign independent semantic and geometry responsibilities.
        self.geometry_interaction = deepcopy(interaction)
        # Plain evaluation state only.  It is deliberately neither a parameter
        # nor a persistent buffer, so validation attribution cannot change the
        # checkpoint/optimizer/RNG identity of the Schema28 graph.
        self._eval_intervention = "none"

    def set_eval_intervention(self, mode: str) -> None:
        """Remove only the P2-owned consequence innovation during evaluation."""

        if self.training:
            raise ValueError("consequence interventions are evaluation-only")
        if mode not in self.INTERVENTION_MODES:
            raise ValueError(
                "consequence intervention must be one of "
                + ", ".join(self.INTERVENTION_MODES)
            )
        self._eval_intervention = mode

    def clear_eval_intervention(self) -> None:
        self._eval_intervention = "none"

    def forward(
        self,
        *,
        factual_base: Tensor,
        scene_factual_base: Tensor | None = None,
        effect: ObjectEffectBundle,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectConsequenceState, dict[str, Tensor]]:
        effect.validate()
        if tuple(factual_base.shape) != tuple(effect.target.semantic.shape):
            raise ValueError("factual base and typed effect must align")
        if scene_factual_base is not None and tuple(scene_factual_base.shape) != tuple(
            factual_base.shape
        ):
            raise ValueError("scene factual base and target factual base must align")
        fact_gate = torch.tanh(self.fact(factual_base))
        primary_interaction = ObjectTypedEffect(
            semantic=self.semantic_interaction(fact_gate * effect.target.semantic),
            geometry=self.geometry_interaction(fact_gate * effect.target.geometry),
        )
        primary_interaction.validate()
        mode = self._eval_intervention
        if mode == "none":
            # Preserve the primary path without an identity multiply, clone or
            # reconstruction.  This is what makes explicit-none bit-exact.
            consumed_effect = effect.target
            interaction = primary_interaction
            scene_effect = effect.scene
        else:
            if self.training:
                raise ValueError("consequence interventions are evaluation-only")
            if mode != "effect_neutral":
                raise RuntimeError("consequence intervention state is invalid")
            # The fact gate and both typed interaction networks have already
            # executed.  Neutralization happens only where their P2-owned
            # innovation enters the consequence state; factual P1 remains the
            # exact original tensor and dynamic P1 precision remains outside
            # this object under its own owner.
            consumed_effect = ObjectTypedEffect(
                semantic=torch.zeros_like(effect.target.semantic),
                geometry=torch.zeros_like(effect.target.geometry),
            )
            interaction = ObjectTypedEffect(
                semantic=torch.zeros_like(primary_interaction.semantic),
                geometry=torch.zeros_like(primary_interaction.geometry),
            )
            scene_effect = ObjectTypedEffect(
                semantic=torch.zeros_like(effect.scene.semantic),
                geometry=torch.zeros_like(effect.scene.geometry),
            )
        consumed_effect.validate()
        interaction.validate()
        scene_effect.validate()
        # This is the sole type-removal point.  Fusion is parameter-free and
        # happens only after both typed zero-preserving interactions complete.
        combined_effect = consumed_effect.combined()
        combined_interaction = interaction.combined()
        protected = factual_base + combined_effect + combined_interaction
        state = ObjectConsequenceState(
            factual_base=factual_base,
            scene_factual_base=scene_factual_base,
            effect=consumed_effect,
            interaction=interaction,
            scene_effect=scene_effect,
            scene_enabled=(effect.scene_enabled or scene_factual_base is not None),
            protected_consequence=protected,
        )
        state.validate()
        if not collect_diagnostics:
            return state, {}
        primary_combined_effect = effect.target.combined()
        primary_scene_effect = effect.scene.combined()
        primary_combined_interaction = primary_interaction.combined()
        return state, {
            "object_consequence_effect_rms": combined_effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_semantic_interaction_rms": interaction.semantic.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_geometry_interaction_rms": interaction.geometry.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_interaction_rms": combined_interaction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_scene_effect_rms": scene_effect.combined()
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_scene_factual_rms": (
                scene_factual_base.detach().float().square().mean().sqrt()
                if scene_factual_base is not None
                else factual_base.new_zeros((), dtype=torch.float32)
            ),
            "object_consequence_ratio": (
                (combined_effect + combined_interaction)
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
                / factual_base.detach().float().square().mean().sqrt().clamp_min(1e-6)
            ),
            "object_consequence_intervention_active": factual_base.new_tensor(
                float(mode != "none"), dtype=torch.float32
            ),
            "object_consequence_intervention_effect_delta_rms": (
                primary_combined_effect.detach().float()
                - combined_effect.detach().float()
            )
            .square()
            .mean()
            .sqrt(),
            "object_consequence_intervention_interaction_delta_rms": (
                primary_combined_interaction.detach().float()
                - combined_interaction.detach().float()
            )
            .square()
            .mean()
            .sqrt(),
            "object_consequence_intervention_scene_delta_rms": (
                primary_scene_effect.detach().float()
                - scene_effect.combined().detach().float()
            )
            .square()
            .mean()
            .sqrt(),
            "object_consequence_intervention_first_boundary_delta_rms": (
                (
                    primary_combined_effect.detach().float()
                    + primary_combined_interaction.detach().float()
                )
                - (combined_effect.detach().float() + combined_interaction.detach().float())
            )
            .square()
            .mean()
            .sqrt(),
            "object_consequence_intervention_factual_identity_max_abs": (
                state.factual_base.detach().float() - factual_base.detach().float()
            )
            .abs()
            .amax(),
        }


class ObjectPolicyPlanCompiler(nn.Module):
    """Compile the two P3 innovations without duplicating protected owners."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
        action_dim: int = 7,
        target_action_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.target_action_bottleneck = bool(target_action_bottleneck)
        # Consume the six removed alias projections' historical draws so the
        # retained P3 weights and every subsequently constructed module keep
        # their R1f fresh-run initialization stream. These temporary tensors
        # are never registered, serialized, moved or executed.
        removed_aliases = tuple(
            nn.Linear(hidden, hidden, bias=False) for _ in range(6)
        )
        self.temporal_action = nn.Linear(hidden, hidden, bias=False)
        self.temporal_effect = nn.Linear(hidden, hidden, bias=False)
        self.temporal_lane = nn.Linear(hidden, hidden, bias=False)
        self.state_change_action = nn.Linear(hidden, hidden, bias=False)
        self.state_change_temporal = nn.Linear(hidden, hidden, bias=False)
        self.state_change_lane = nn.Linear(hidden, hidden, bias=False)
        self.action_proposal_value: nn.Linear | None = (
            nn.Linear(int(action_dim), hidden, bias=False)
            if self.target_action_bottleneck
            else None
        )
        del removed_aliases

    def forward(
        self,
        *,
        p1_policy_residual: Tensor,
        consequence: ObjectConsequenceState,
        intent: PolicyIntentDock,
        action_query: Tensor,
        action_proposal: Tensor | None = None,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if tuple(action_query.shape) != expected or tuple(
            p1_policy_residual.shape
        ) != expected:
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        consequence.validate()
        for name in ("factual_base", "protected_consequence"):
            if tuple(getattr(consequence, name).shape) != expected:
                raise ValueError(f"P3 consequence {name} lost [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        proposal_residual = torch.zeros_like(p1_policy_residual)
        if self.target_action_bottleneck:
            if self.action_proposal_value is None or action_proposal is None:
                raise ValueError("target-action P3 requires the physical A0 proposal")
            if tuple(action_proposal.shape) != (
                expected[0],
                self.horizon,
                int(self.action_proposal_value.in_features),
            ):
                raise ValueError("target-action P3 A0 must be [B,T,A]")
            proposal_context, _ = smooth_rms_contract(
                self.action_proposal_value(action_proposal),
                0.35,
            )
            proposal_residual = proposal_context[:, :, None].expand(
                -1, -1, self.basis, -1
            )
            temporal_context = proposal_residual
        else:
            proposal_context = intent.temporal_control
            temporal_context = proposal_context[:, :, None].expand(
                -1, -1, self.basis, -1
            )
        if tuple(temporal_context.shape) != expected:
            raise ValueError("P3 temporal context lost the policy schema")
        consequence_innovation = consequence.innovation()
        scene_raw = consequence.scene_innovation()
        temporal_private = temporal_context + self.temporal_effect(
            consequence_innovation
        )
        temporal_raw = self.temporal_lane(
            temporal_private * torch.tanh(self.temporal_action(action_query))
        )
        state_change_source = intent.state_change_evidence[:, None, None].expand(
            -1, self.horizon, self.basis, -1
        )
        state_change_modulation = torch.tanh(
            self.state_change_action(action_query)
            + self.state_change_temporal(temporal_context)
        )
        state_change_raw = self.state_change_lane(
            state_change_source * state_change_modulation
        )
        temporal, temporal_scale = smooth_rms_contract(temporal_raw, 0.35)
        scene, scene_scale = smooth_rms_contract(scene_raw, 0.35)
        state_change, state_change_scale = smooth_rms_contract(
            state_change_raw,
            0.35,
        )
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            protected_policy_precision=(p1_policy_residual + proposal_residual),
            scene=(scene if consequence.scene_enabled else None),
            temporal=temporal,
            state_change=state_change,
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        metrics = {
            "object_p3_protected_policy_precision_rms": p1_policy_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_consequence_innovation_rms": consequence_innovation.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_physical_action_proposal_rms": proposal_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_scene_consequence_rms": scene.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_scene_contract_min": scene_scale.detach().float().amin(),
            "object_p3_temporal_private_rms": temporal_private.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_rms": temporal.detach().float().square().mean().sqrt(),
            "object_p3_temporal_contract_min": temporal_scale.detach().float().amin(),
            "object_p3_state_change_rms": state_change.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_state_change_contract_min": state_change_scale.detach()
            .float()
            .amin(),
        }
        if self.training:
            register_gradient_rms_metric(
                p1_policy_residual,
                metrics,
                "gradient_tensor_p1_protected_policy_precision_rms",
            )
            register_gradient_rms_metric(
                scene,
                metrics,
                "gradient_tensor_p3_scene_consequence_rms",
            )
            register_gradient_rms_metric(
                temporal,
                metrics,
                "gradient_tensor_p3_temporal_rms",
            )
            register_gradient_rms_metric(
                state_change,
                metrics,
                "gradient_tensor_p3_state_change_rms",
            )
        return bank, metrics


__all__ = [
    "ObjectConsequenceState",
    "ObjectEffectBundle",
    "ObjectFutureEffectReader",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "ObjectTypedEffect",
    "SelectedIntervalEvidence",
    "ZeroPreservingObjectConsequence",
]
