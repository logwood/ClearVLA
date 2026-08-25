"""Typed future-effect reads and consequence-conditioned P3 compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import (
    PolicyRoleDeltaBank,
    register_gradient_axis_rms_metrics,
    register_gradient_rms_metric,
    smooth_rms_contract,
    variance_floored_centered_norm,
)
from .types import FutureObjectDynamics, PolicyIntentDock, normalized_entropy


@dataclass(frozen=True)
class TypedP2EffectRead:
    """P2 effect with semantic/geometry ownership intact.

    The type axis is ordered exactly as :attr:`ObjectFutureEffectReader.TYPE_NAMES`.
    ``physical_sum`` is the only untyped view and is formed by a literal sum;
    no selector or averaging is allowed at this boundary.
    """

    effect_by_type: Tensor  # [B,T,Q,2,H], semantic then geometry

    def validate(self) -> None:
        if self.effect_by_type.ndim != 5:
            raise ValueError("typed P2 effect must be [B,T,Q,type,H]")
        if int(self.effect_by_type.shape[-2]) != 2:
            raise ValueError("typed P2 effect must retain semantic/geometry")

    @property
    def semantic(self) -> Tensor:
        self.validate()
        return self.effect_by_type[..., 0, :]

    @property
    def geometry(self) -> Tensor:
        self.validate()
        return self.effect_by_type[..., 1, :]

    @property
    def physical_sum(self) -> Tensor:
        self.validate()
        return self.effect_by_type.sum(dim=-2)


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    effect_by_type: Tensor
    interaction_by_type: Tensor
    protected_consequence: Tensor

    @property
    def effect(self) -> Tensor:
        """Physical semantic-plus-geometry effect."""

        return self.effect_by_type.sum(dim=-2)

    @property
    def interaction(self) -> Tensor:
        """Physical semantic-plus-geometry interaction."""

        return self.interaction_by_type.sum(dim=-2)

    def typed_effect(self) -> Tensor:
        """Return the mandatory live typed effect sidecar."""

        return self.effect_by_type

    def typed_interaction(self) -> Tensor:
        """Return the live typed interaction sidecar."""

        return self.interaction_by_type

    def validate(self) -> None:
        expected = tuple(self.factual_base.shape)
        if len(expected) != 4:
            raise ValueError("object consequence must be [B,T,Q,H]")
        if tuple(self.protected_consequence.shape) != expected:
            raise ValueError("object protected consequence lost [B,T,Q,H]")
        typed_effect = self.typed_effect()
        typed_interaction = self.typed_interaction()
        typed_expected = (*expected[:-1], 2, expected[-1])
        if tuple(typed_effect.shape) != typed_expected:
            raise ValueError("object consequence effect lost its type axis")
        if tuple(typed_interaction.shape) != typed_expected:
            raise ValueError("object consequence interaction lost its type axis")


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """Six optional typed innovations around one protected consequence."""

    protected_base: Tensor
    precision: Tensor
    effect_semantic: Tensor
    effect_geometry: Tensor
    temporal_semantic: Tensor
    temporal_geometry: Tensor
    state_change: Tensor

    @property
    def effect(self) -> Tensor:
        """Compatibility view; the routed bank retains the two typed lanes."""

        return self.effect_semantic + self.effect_geometry

    @property
    def temporal(self) -> Tensor:
        """Compatibility view; the routed bank retains the two typed lanes."""

        return self.temporal_semantic + self.temporal_geometry

    @property
    def source_names(self) -> tuple[str, ...]:
        return (
            "p3_precision",
            "p3_effect_semantic",
            "p3_effect_geometry",
            "p3_temporal_semantic",
            "p3_temporal_geometry",
            "p3_state_change",
        )

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        for name in (
            "precision",
            "effect_semantic",
            "effect_geometry",
            "temporal_semantic",
            "temporal_geometry",
            "state_change",
        ):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack(
                (
                    self.precision,
                    self.effect_semantic,
                    self.effect_geometry,
                    self.temporal_semantic,
                    self.temporal_geometry,
                    self.state_change,
                ),
                dim=1,
            ),
            source_names=self.source_names,
            source_depths=(int(source_depth),) * 6,
            protected_detail=self.protected_base,
        )


class ObjectFutureEffectReader(nn.Module):
    """Read complete typed W fields through W-anchored interval relations.

    Each semantic/geometry owner has one four-interval-plus-null simplex. S can
    condition the selected W key, but cannot contribute an independent time
    logit or nonzero value. Common and interval innovation are routed together
    as one complete field. Status remains absent because it has no independent
    observable target or legal action value.
    """

    TYPE_NAMES = ("semantic", "geometry")
    # S/W retain semantic/appearance/geometry in that order.  P2 consumes only
    # semantic and geometry values; semantic maps to S semantic (0), while
    # geometry maps to S geometry (2).  Never rely on matching integer
    # positions across these differently named type systems.
    S_TYPE_INDEX_BY_P2 = (0, 2)
    # A sparse active owner must retain its native hidden unit. Semantic and
    # geometry are therefore contracted only after their physical sum exists;
    # the one shared scale is copied back onto both typed sidecars.
    COMPLEMENTARY_VALUE_MAX_RMS = 0.35

    def __init__(self, *, hidden: int, content_dim: int, route_dim: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.source_query = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.source_key = nn.ModuleList(
            (
                nn.Linear(content_dim, hidden, bias=False),
                nn.Linear(2, hidden, bias=False),
            )
        )
        # Schema37 serialized two independent S-vote queries plus one public
        # vote query. Schema38 deliberately owns no such parameters. Consume
        # their historical initialization draws without registering modules so
        # subsequent live weights retain fresh-run RNG comparability.
        for _ in range(len(self.TYPE_NAMES) + 1):
            nn.Linear(hidden, hidden, bias=False)
        self.public_interval_key = nn.Linear(hidden, hidden, bias=False)
        self.common_intent_key = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.typed_intent_key = nn.ModuleList(
            nn.Linear(route_dim, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.coordinate_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.temperature_logit = nn.Parameter(torch.zeros(3))

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        value = value.float()
        return value / (
            value.square().sum(dim=-1, keepdim=True) + float(norm_floor) ** 2
        ).sqrt()

    def _fuse_complementary_values(
        self,
        selected_type_value: Tensor,
    ) -> tuple[TypedP2EffectRead, Tensor, Tensor]:
        """Physically sum both types once and share its one RMS contract.

        Applying the returned scale to the still-typed values preserves their
        relative physical contribution exactly.  It also prevents either a
        per-type limiter or a later untyped limiter from changing the typed
        sidecar seen by consequence/P3.
        """

        if selected_type_value.ndim < 2 or int(selected_type_value.shape[-2]) != len(
            self.TYPE_NAMES
        ):
            raise ValueError("P2 complementary values must retain the active type axis")
        raw_combined = selected_type_value.sum(dim=-2)
        _, shared_scale = smooth_rms_contract(
            raw_combined,
            self.COMPLEMENTARY_VALUE_MAX_RMS,
        )
        contracted_by_type = selected_type_value * shared_scale.to(
            dtype=selected_type_value.dtype
        )[..., None, :]
        read = TypedP2EffectRead(effect_by_type=contracted_by_type)
        read.validate()
        return read, raw_combined, shared_scale

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[TypedP2EffectRead, dict[str, Tensor]]:
        dynamics.validate()
        if action_query.ndim != 4:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        batch, horizon, basis, hidden = action_query.shape
        if hidden != self.hidden:
            raise ValueError("P2 action query hidden width is invalid")
        intent.validate(horizon=horizon, hidden=self.hidden)
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        typed_common_intent = intent.typed_common_object_value
        typed_residual_intent = intent.typed_interval_residual_value
        if tuple(typed_common_intent.shape[:3]) != (batch, objects, 3):
            raise ValueError("P2 typed common intent lost object/type identity")
        if tuple(typed_residual_intent.shape[:4]) != (
            batch,
            intervals,
            objects,
            3,
        ):
            raise ValueError("P2 typed residual intent lost interval/object/type identity")
        camera_coordinate = dynamics.camera_coordinates.float()
        camera_availability = dynamics.camera_chart_availability.float()[..., 0].clamp(
            0.0, 1.0
        )
        # ``camera_weights`` is the producer-owned physical evidence measure
        # and already includes chart availability. Mask illegal cameras but do
        # not multiply that measure a second time.
        camera_weight = torch.where(
            camera_availability > 0.0,
            dynamics.camera_weights.float()[..., 0].clamp_min(0.0),
            torch.zeros_like(camera_availability),
        )
        camera_count = int(camera_coordinate.shape[2])
        camera_mass = camera_weight.sum(dim=-1, keepdim=True)
        camera_has_support = camera_mass[..., 0] > 1.0e-8
        uniform_camera = torch.full_like(
            camera_weight, 1.0 / float(max(camera_count, 1))
        )
        normalized_camera_weight = torch.where(
            camera_mass > 1.0e-8,
            camera_weight / camera_mass.clamp_min(1.0e-8),
            uniform_camera,
        )
        # Schema38 reads one complete W field per interval. Common and
        # interval innovation remain an exact decomposition for supervision and
        # diagnostics, but neither component owns an independent action route.
        # In particular, the common field is no longer written outside the
        # interval/null competition.
        transport_common = dynamics.transport_common
        transport_residual = dynamics.transport_interval_residual
        common_source_fields = (
            dynamics.semantic_common,
            transport_common,
        )
        residual_source_fields = (
            dynamics.semantic_interval_residual,
            transport_residual,
        )
        full_source_fields = (
            dynamics.semantic_delta,
            dynamics.transport_mean,
        )
        common_projected_values = (
            self.semantic_value(common_source_fields[0]),
            self.transport_value(common_source_fields[1]),
        )
        residual_projected_values = (
            self.semantic_value(residual_source_fields[0]),
            self.transport_value(residual_source_fields[1]),
        )

        # Camera is a physical geometry hypothesis axis, not a preprocessing
        # nuisance axis. Keep KxC through the action-conditioned posterior.
        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        camera_coordinate_mean = (
            normalized_camera_weight[..., None] * camera_coordinate
        ).sum(dim=2)
        camera_coordinate_variation = (
            normalized_camera_weight[..., None]
            * (
                camera_coordinate
                - camera_coordinate_mean[:, :, None]
            ).square()
        ).sum(dim=2).mean(dim=-1).sqrt()
        camera_coordinate_variation = (
            camera_coordinate_variation * camera_has_support.float()
        ).mean()
        future_coordinate = (
            camera_coordinate[:, None]
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )

        def covariance_aware_distance(delta: Tensor, covariance: Tensor) -> Tensor:
            variance_floor = 1.0 / float(7 * 7)
            xx = covariance[..., 0].float().clamp_min(0.0) + variance_floor
            xy = covariance[..., 1].float()
            yy = covariance[..., 2].float().clamp_min(0.0) + variance_floor
            determinant = (xx * yy - xy.square()).clamp_min(variance_floor**2)
            dx, dy = delta[..., 0], delta[..., 1]
            return variance_floor * (
                yy * dx.square() - 2.0 * xy * dx * dy + xx * dy.square()
            ) / determinant

        coordinate_distance = covariance_aware_distance(
            coordinate_delta,
            dynamics.transport_covariance.float()[:, None, None],
        )
        coordinate_score = (1.0 - 0.5 * coordinate_distance).clamp(-1.0, 1.0)
        coordinate_score = torch.where(
            camera_has_support[:, None, None, None, :, None],
            coordinate_score,
            torch.zeros_like(coordinate_score),
        )
        current_validity = dynamics.chart_availability.float().squeeze(
            -1
        ).clamp(0.0, 1.0)
        geometry_current_validity = (
            current_validity[..., None]
            * normalized_camera_weight
            * camera_has_support[..., None].float()
        )
        full_validities = (
            current_validity[:, None].expand(-1, intervals, -1),
            geometry_current_validity[:, None].expand(
                -1, intervals, -1, -1
            ),
        )

        temperature = self._temperatures().to(device=action_query.device)
        public_interval_key = self._bounded_unit(
            self.public_interval_key(intent.interval_residual_key)
        )

        # Geometry is complementary spatial evidence for semantic object
        # identity. Marginalize its observable KxC coordinate likelihood to K,
        # remove the K-common offset, and use the bounded remainder only inside
        # the semantic K posterior. Missing cameras and uniform geometry produce
        # exact zero, so the semantic-only reader is an algebraic fallback.
        # Semantic already consumes ``current_validity`` as its K prior below.
        # The complementary geometry correction may therefore use K validity
        # only as a boolean support boundary; multiplying its magnitude into
        # the camera marginal would vote for the same object support twice.
        # Within every legal K, use only the conditional camera measure.
        geometry_k_legal = current_validity > 0.0
        geometry_camera_support = (
            geometry_k_legal[..., None]
            & (camera_weight > 0.0)
            & camera_has_support[..., None]
        )
        geometry_support = geometry_camera_support[:, None].expand(
            -1, intervals, -1, -1
        )
        conditional_camera_measure = normalized_camera_weight[:, None].expand(
            -1, intervals, -1, -1
        )
        geometry_measure_log = torch.where(
            geometry_support,
            conditional_camera_measure.clamp_min(1.0e-6).log(),
            torch.zeros_like(conditional_camera_measure),
        )
        geometry_coordinate_logit = (
            temperature[2] * coordinate_score
            + geometry_measure_log[:, None, None]
        ).masked_fill(~geometry_support[:, None, None], -torch.inf)
        geometry_k_support = geometry_support.any(dim=-1)
        # Never differentiate through logsumexp(all -inf). Unsupported K rows
        # take a finite bookkeeping branch before reduction and are forced to
        # exact zero afterwards, so both their value and backward are finite.
        geometry_coordinate_logit = torch.where(
            geometry_k_support[:, None, None, :, :, None],
            geometry_coordinate_logit,
            torch.zeros_like(geometry_coordinate_logit),
        )
        geometry_k_evidence = torch.logsumexp(
            geometry_coordinate_logit,
            dim=-1,
        )
        geometry_k_evidence = torch.where(
            geometry_k_support[:, None, None],
            geometry_k_evidence,
            torch.zeros_like(geometry_k_evidence),
        )
        geometry_k_weight = geometry_k_support[:, None, None].float()
        geometry_k_common = (
            geometry_k_evidence * geometry_k_weight
        ).sum(dim=-1, keepdim=True) / geometry_k_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        semantic_geometry_correction = torch.tanh(
            geometry_k_evidence - geometry_k_common
        ) * geometry_k_weight

        full_source_scores: list[Tensor] = []
        common_intent_scores: list[Tensor] = []
        residual_intent_scores: list[Tensor] = []
        interval_public_scores: list[Tensor] = []
        interval_typed_scores: list[Tensor] = []
        interval_w_scores: list[Tensor] = []
        selected_w_key_rms_values: list[Tensor] = []
        w_key_interval_variation_values: list[Tensor] = []
        typed_s_key_delta_rms_values: list[Tensor] = []
        public_s_key_delta_rms_values: list[Tensor] = []
        combined_s_key_delta_rms_values: list[Tensor] = []
        s_condition_pre_tanh_rms_values: list[Tensor] = []
        s_condition_saturation_values: list[Tensor] = []
        conditioned_interval_scores: list[Tensor] = []
        full_bounded_logits: list[Tensor] = []
        residual_object_posteriors: list[Tensor] = []
        interval_supports: list[Tensor] = []
        selected_common_values_by_interval: list[Tensor] = []
        selected_residual_values_by_interval: list[Tensor] = []
        selected_full_values_by_interval: list[Tensor] = []
        semantic_geometry_neutral_posterior_l1 = action_query.new_zeros(
            (), dtype=torch.float32
        )
        semantic_geometry_neutral_output_l1 = action_query.new_zeros(
            (), dtype=torch.float32
        )

        for type_index in range(len(self.TYPE_NAMES)):
            intent_type_index = self.S_TYPE_INDEX_BY_P2[type_index]
            query = self._bounded_unit(
                self.source_query[type_index](action_query)
            )
            full_source_key = self._bounded_unit(
                self.source_key[type_index](full_source_fields[type_index])
            )
            common_public_key = self.common_intent_key[type_index](
                intent.common_key
            )
            common_typed_route = typed_common_intent[..., intent_type_index, :]
            residual_typed_route = typed_residual_intent[
                ..., intent_type_index, :
            ]
            full_typed_route = (
                common_typed_route[:, None] + residual_typed_route
            )
            full_typed_key = self._bounded_unit(
                self.typed_intent_key[type_index](full_typed_route)
            )
            if type_index == 0:
                full_source_score = torch.einsum(
                    "btqh,bikh->btqik", query, full_source_key
                )
                candidate_typed_key = full_typed_key
            else:
                full_source_score = torch.einsum(
                    "btqh,bikch->btqikc", query, full_source_key
                )
                candidate_typed_key = full_typed_key[..., None, :].expand_as(
                    full_source_key
                )

            # Object/KxC identity is selected by action-conditioned W evidence
            # only (plus the explicit geometry address correction below). S is
            # read after this posterior and cannot independently select an
            # object hypothesis or a future interval.
            full_bounded_logit = temperature[0] * full_source_score
            full_bounded_logit_without_geometry = full_bounded_logit
            if type_index == 0:
                full_bounded_logit = (
                    full_bounded_logit + semantic_geometry_correction
                )
            else:
                full_bounded_logit = (
                    full_bounded_logit + temperature[2] * coordinate_score
                )

            full_validity = full_validities[type_index]
            full_support = full_validity > 0.0
            interval_has_support = full_support.flatten(start_dim=2).any(dim=-1)
            full_logit = full_bounded_logit + torch.where(
                full_support,
                full_validity.clamp_min(1.0e-6).log(),
                torch.zeros_like(full_validity),
            )[:, None, None]
            full_logit = full_logit.masked_fill(
                ~full_support[:, None, None],
                -torch.inf,
            ).reshape(batch, horizon, basis, intervals, -1)
            full_logit = torch.where(
                interval_has_support[:, None, None, :, None],
                full_logit,
                torch.zeros_like(full_logit),
            )
            object_posterior = torch.softmax(full_logit, dim=-1)
            object_posterior = (
                object_posterior
                * interval_has_support[:, None, None, :, None].to(
                    dtype=object_posterior.dtype
                )
            )
            semantic_geometry_neutral_posterior: Tensor | None = None
            if collect_diagnostics and type_index == 0:
                neutral_full_logit = full_bounded_logit_without_geometry + torch.where(
                    full_support,
                    full_validity.clamp_min(1.0e-6).log(),
                    torch.zeros_like(full_validity),
                )[:, None, None]
                neutral_full_logit = neutral_full_logit.masked_fill(
                    ~full_support[:, None, None],
                    -torch.inf,
                ).reshape(batch, horizon, basis, intervals, -1)
                neutral_full_logit = torch.where(
                    interval_has_support[:, None, None, :, None],
                    neutral_full_logit,
                    torch.zeros_like(neutral_full_logit),
                )
                semantic_geometry_neutral_posterior = torch.softmax(
                    neutral_full_logit,
                    dim=-1,
                ) * interval_has_support[:, None, None, :, None].to(
                    dtype=object_posterior.dtype
                )
                semantic_geometry_neutral_posterior_l1 = (
                    object_posterior.float()
                    - semantic_geometry_neutral_posterior.float()
                ).abs().mean()

            full_key_flat = full_source_key.reshape(
                batch, intervals, -1, self.hidden
            )
            typed_key_flat = candidate_typed_key.reshape(
                batch, intervals, -1, self.hidden
            )
            common_value_flat = common_projected_values[type_index].reshape(
                batch, -1, self.hidden
            )[:, None].expand(-1, intervals, -1, -1)
            residual_value_flat = residual_projected_values[type_index].reshape(
                batch, intervals, -1, self.hidden
            )
            selected_w_key = torch.einsum(
                "btqin,binh->btqih",
                object_posterior.to(dtype=full_key_flat.dtype),
                full_key_flat,
            )
            selected_typed_key = torch.einsum(
                "btqin,binh->btqih",
                object_posterior.to(dtype=typed_key_flat.dtype),
                typed_key_flat,
            )
            selected_common_by_interval = torch.einsum(
                "btqin,binh->btqih",
                object_posterior.to(dtype=common_value_flat.dtype),
                common_value_flat,
            )
            selected_residual_by_interval = torch.einsum(
                "btqin,binh->btqih",
                object_posterior.to(dtype=residual_value_flat.dtype),
                residual_value_flat,
            )
            selected_full_by_interval = (
                selected_common_by_interval + selected_residual_by_interval
            )
            if semantic_geometry_neutral_posterior is not None:
                neutral_common_by_interval = torch.einsum(
                    "btqin,binh->btqih",
                    semantic_geometry_neutral_posterior.to(
                        dtype=common_value_flat.dtype
                    ),
                    common_value_flat,
                )
                neutral_residual_by_interval = torch.einsum(
                    "btqin,binh->btqih",
                    semantic_geometry_neutral_posterior.to(
                        dtype=residual_value_flat.dtype
                    ),
                    residual_value_flat,
                )
                semantic_geometry_neutral_output_l1 = (
                    selected_full_by_interval.float()
                    - (
                        neutral_common_by_interval.float()
                        + neutral_residual_by_interval.float()
                    )
                ).abs().mean()

            # S is selected by the W-owned object posterior and can only
            # multiplicatively condition that nonzero W key. With neutral W,
            # the conditioned key and future value remain exactly zero.
            public_s_context = (
                common_public_key[:, None, None, None]
                + public_interval_key[:, None, None]
            )
            typed_s_context = selected_typed_key.float()
            combined_s_context = public_s_context + typed_s_context
            public_s_condition = torch.tanh(
                temperature[1] * self._bounded_unit(public_s_context)
            )
            typed_s_condition = torch.tanh(
                temperature[1] * self._bounded_unit(typed_s_context)
            )
            s_condition_pre_tanh = (
                temperature[1] * self._bounded_unit(combined_s_context)
            )
            s_condition = torch.tanh(s_condition_pre_tanh)
            s_key_delta = selected_w_key.float() * s_condition
            public_s_key_delta = selected_w_key.float() * public_s_condition
            typed_s_key_delta = selected_w_key.float() * typed_s_condition
            conditioned_w_key = selected_w_key.float() + s_key_delta
            conditioned_interval_score = torch.einsum(
                "btqh,btqih->btqi",
                query,
                self._bounded_unit(conditioned_w_key),
            )
            if collect_diagnostics:
                interval_w_score = torch.einsum(
                    "btqh,btqih->btqi",
                    query,
                    self._bounded_unit(selected_w_key),
                )
                interval_typed_score = torch.einsum(
                    "btqh,btqih->btqi",
                    query,
                    self._bounded_unit(typed_s_key_delta),
                )
                public_interval_score = torch.einsum(
                    "btqh,btqih->btqi",
                    query,
                    self._bounded_unit(public_s_key_delta),
                )
                # These diagnostics measure conditional W-key corrections;
                # neither score is an independent S likelihood.
                full_source_scores.append(full_source_score)
                common_intent_scores.append(public_interval_score)
                residual_intent_scores.append(interval_typed_score)
                interval_public_scores.append(public_interval_score)
                interval_typed_scores.append(interval_typed_score)
                interval_w_scores.append(interval_w_score)
                selected_w_key_rms_values.append(
                    selected_w_key.float().square().mean().sqrt()
                )
                w_key_interval_variation_values.append(
                    (
                        selected_w_key.float()
                        - selected_w_key.float().mean(dim=3, keepdim=True)
                    ).square().mean().sqrt()
                )
                typed_s_key_delta_rms_values.append(
                    typed_s_key_delta.square().mean().sqrt()
                )
                public_s_key_delta_rms_values.append(
                    public_s_key_delta.square().mean().sqrt()
                )
                combined_s_key_delta_rms_values.append(
                    s_key_delta.float().square().mean().sqrt()
                )
                s_condition_pre_tanh_rms_values.append(
                    s_condition_pre_tanh.float().square().mean().sqrt()
                )
                s_condition_saturation_values.append(
                    (s_condition.detach().abs() > 0.95).float().mean()
                )
            conditioned_interval_scores.append(conditioned_interval_score)
            if collect_diagnostics:
                full_bounded_logits.append(full_bounded_logit)
            residual_object_posteriors.append(object_posterior)
            interval_supports.append(interval_has_support)
            selected_common_values_by_interval.append(
                selected_common_by_interval
            )
            selected_residual_values_by_interval.append(
                selected_residual_by_interval
            )
            selected_full_values_by_interval.append(selected_full_by_interval)

        # Each type owns one I+null competition over complete W fields. The
        # null value is exactly zero; common and interval innovation can no
        # longer be accepted/rejected through different probability simplices.
        type_interval_precontract_score = torch.stack(
            conditioned_interval_scores,
            dim=3,
        )
        type_interval_score = type_interval_precontract_score
        type_interval_bounded_logit = temperature[0] * type_interval_score
        type_interval_has_support = torch.stack(interval_supports, dim=1)
        type_interval_logit = type_interval_bounded_logit.masked_fill(
            ~type_interval_has_support[:, None, None],
            -torch.inf,
        )
        # One type-local null is an equal fifth candidate, not a reward for
        # accepting one of K (or KxC) inner hypotheses. Candidate count belongs
        # only to the conditional object posterior above.
        type_null_logit = type_interval_logit.new_zeros(
            batch, horizon, basis, len(self.TYPE_NAMES), 1
        )
        type_interval_posterior = torch.softmax(
            torch.cat((type_interval_logit, type_null_logit), dim=-1),
            dim=-1,
        )
        type_interval_mass = type_interval_posterior[..., :-1]

        selected_common_values: list[Tensor] = []
        selected_residual_values: list[Tensor] = []
        selected_full_values: list[Tensor] = []
        residual_posteriors: list[Tensor] = []
        common_posteriors: list[Tensor] = []
        for type_index, object_posterior in enumerate(
            residual_object_posteriors
        ):
            interval_mass_for_type = type_interval_mass[..., type_index, :]
            selected_common_values.append(
                torch.einsum(
                    "btqi,btqih->btqh",
                    interval_mass_for_type.to(
                        dtype=selected_common_values_by_interval[type_index].dtype
                    ),
                    selected_common_values_by_interval[type_index],
                )
            )
            selected_residual_values.append(
                torch.einsum(
                    "btqi,btqih->btqh",
                    interval_mass_for_type.to(
                        dtype=selected_residual_values_by_interval[type_index].dtype
                    ),
                    selected_residual_values_by_interval[type_index],
                )
            )
            selected_full_values.append(
                selected_common_values[-1] + selected_residual_values[-1]
            )
            if collect_diagnostics:
                joint_real = (
                    interval_mass_for_type[..., None] * object_posterior
                ).flatten(-2)
                residual_posteriors.append(
                    torch.cat(
                        (
                            joint_real,
                            type_interval_posterior[..., type_index, -1:],
                        ),
                        dim=-1,
                    )
                )
                real_mass = interval_mass_for_type.sum(dim=-1, keepdim=True)
                common_posteriors.append(
                    (
                        interval_mass_for_type[..., None] * object_posterior
                    ).sum(dim=3)
                    / real_mass.clamp_min(1.0e-8)
                )

        selected_type_value = torch.stack(selected_full_values, dim=3)
        effect_read, raw_value, shared_effect_scale = (
            self._fuse_complementary_values(selected_type_value)
        )
        if not collect_diagnostics:
            return effect_read, {}
        residual_value_by_interval_and_type = torch.stack(
            selected_residual_values_by_interval,
            dim=4,
        )
        selected_common_type_value = torch.stack(selected_common_values, dim=3)
        selected_residual_type_value = torch.stack(
            selected_residual_values,
            dim=3,
        )
        public_interval_score_by_type = torch.stack(
            interval_public_scores,
            dim=3,
        )
        public_interval_score = public_interval_score_by_type.mean(dim=3)
        interval_typed_score_by_type = torch.stack(
            interval_typed_scores,
            dim=3,
        )
        interval_w_score_by_type = torch.stack(
            interval_w_scores,
            dim=3,
        )
        # Counterfactual diagnostic only: remove every S multiplicative
        # correction while retaining the identical W object posterior, support
        # and equal fifth null. This quantifies S's actual conditional effect
        # without creating a second forward route.
        neutral_interval_logit = (
            temperature[0] * interval_w_score_by_type
        ).masked_fill(
            ~type_interval_has_support[:, None, None],
            -torch.inf,
        )
        neutral_interval_posterior = torch.softmax(
            torch.cat(
                (
                    neutral_interval_logit,
                    neutral_interval_logit.new_zeros(
                        batch,
                        horizon,
                        basis,
                        len(self.TYPE_NAMES),
                        1,
                    ),
                ),
                dim=-1,
            ),
            dim=-1,
        )
        s_condition_neutral_posterior_l1 = (
            type_interval_posterior.detach().float()
            - neutral_interval_posterior.detach().float()
        ).abs().mean()
        common_value = selected_common_type_value.sum(dim=3)
        residual_value = selected_residual_type_value.sum(dim=3)
        common_semantic_value = selected_common_type_value[..., 0, :]
        common_geometry_value = selected_common_type_value[..., 1, :]
        residual_semantic_value = selected_residual_type_value[..., 0, :]
        residual_geometry_value = selected_residual_type_value[..., 1, :]
        interval_mass_by_type = type_interval_mass
        interval_mass = interval_mass_by_type.detach().mean(dim=3)
        inner_support_weight = type_interval_has_support[
            :, None, None
        ].detach().float()
        inner_support_denominator = (
            inner_support_weight.sum()
            * float(horizon * basis)
        ).clamp_min(1.0)
        inner_entropy = torch.stack(
            tuple(
                normalized_entropy(posterior, dim=-1).detach()
                for posterior in residual_object_posteriors
            ),
            dim=3,
        )
        inner_max = torch.stack(
            tuple(
                posterior.detach().amax(dim=-1)
                for posterior in residual_object_posteriors
            ),
            dim=3,
        )
        inner_raw_entropy = torch.stack(
            tuple(
                -(
                    posterior.detach().float()
                    * posterior.detach().float().clamp_min(1.0e-8).log()
                ).sum(dim=-1)
                for posterior in residual_object_posteriors
            ),
            dim=3,
        )
        interval_type_rms = residual_value_by_interval_and_type.detach().float()
        interval_type_rms = interval_type_rms.square().mean(dim=-1).sqrt()
        selected_type_rms = selected_residual_type_value.detach().float()
        selected_type_rms = selected_type_rms.square().mean(dim=-1).sqrt()
        weighted_interval_type_rms = (
            type_interval_mass.detach().float().transpose(-1, -2)
            * interval_type_rms
        ).sum(dim=3)
        cancellation_support = weighted_interval_type_rms > 1.0e-8
        retained_per_supported = torch.where(
            cancellation_support,
            (
                selected_type_rms
                / weighted_interval_type_rms.clamp_min(1.0e-8)
            ).clamp(0.0, 1.0),
            torch.zeros_like(selected_type_rms),
        )
        cancellation_support_count = cancellation_support.float().sum().clamp_min(1.0)
        residual_retained_rms_ratio = (
            retained_per_supported * cancellation_support.float()
        ).sum() / cancellation_support_count
        residual_cancelled_rms_fraction = (
            (1.0 - retained_per_supported) * cancellation_support.float()
        ).sum() / cancellation_support_count
        source_score_abs = torch.stack(
            tuple(score.detach().abs().mean() for score in full_source_scores)
        ).mean()
        source_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in full_source_scores)
        ).amax()
        intent_score_abs = torch.stack(
            tuple(score.detach().abs().mean() for score in (
                *common_intent_scores,
                *residual_intent_scores,
            ))
        ).mean()
        intent_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in (
                *common_intent_scores,
                *residual_intent_scores,
            ))
        ).amax()
        common_posterior_entropy = torch.stack(
            tuple(
                normalized_entropy(posterior, dim=-1).detach().mean()
                for posterior in common_posteriors
            )
        ).mean()
        common_posterior_max = torch.stack(
            tuple(
                posterior.detach().amax(dim=-1).mean()
                for posterior in common_posteriors
            )
        ).mean()
        residual_posterior_entropy = torch.stack(
            tuple(
                normalized_entropy(posterior, dim=-1).detach().mean()
                for posterior in residual_posteriors
            )
        ).mean()
        residual_posterior_max = torch.stack(
            tuple(
                posterior.detach().amax(dim=-1).mean()
                for posterior in residual_posteriors
            )
        ).mean()
        residual_null_mass = torch.stack(
            tuple(
                posterior.detach()[..., -1].mean()
                for posterior in residual_posteriors
            )
        ).mean()
        metrics: dict[str, Tensor] = {
            "object_p2_content_score_abs": source_score_abs,
            "object_p2_content_score_max_abs": source_score_max,
            "object_p2_s_condition_only_score_abs": intent_score_abs,
            "object_p2_s_condition_only_score_max_abs": intent_score_max,
            "object_p2_coordinate_score_abs": coordinate_score.detach()
            .abs()
            .mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach()
            .abs()
            .amax(),
            "object_p2_camera_mixture_effective_count": torch.exp(
                -(
                    normalized_camera_weight.detach().clamp_min(1.0e-8)
                    * normalized_camera_weight.detach().clamp_min(1.0e-8).log()
                ).sum(dim=-1)
            ).mul(camera_has_support.detach().float()).mean(),
            "object_p2_camera_support_fraction": camera_has_support.detach()
            .float()
            .mean(),
            "object_p2_camera_coordinate_variation": (
                camera_coordinate_variation.detach()
            ),
            "object_p2_combined_logit_max_abs": torch.stack(
                tuple(logit.detach().abs().amax() for logit in (
                    *full_bounded_logits,
                    type_interval_bounded_logit,
                ))
            ).amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_complete_field_collapsed_object_posterior_entropy": (
                common_posterior_entropy
            ),
            "object_p2_complete_field_collapsed_object_posterior_max": (
                common_posterior_max
            ),
            "object_p2_complete_field_joint_posterior_entropy": (
                residual_posterior_entropy
            ),
            "object_p2_complete_field_joint_posterior_max": residual_posterior_max,
            "object_p2_complete_field_null_mass": residual_null_mass,
            "object_p2_public_s_condition_only_score_abs": public_interval_score.detach()
            .abs()
            .mean(),
            "object_p2_public_s_condition_only_score_max_abs": public_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_type_interval_score_abs": type_interval_score.detach()
            .abs()
            .mean(),
            "object_p2_type_interval_score_max_abs": type_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_typed_s_condition_only_score_abs": (
                interval_typed_score_by_type.detach().abs().mean()
            ),
            "object_p2_unconditioned_w_interval_score_abs": (
                interval_w_score_by_type.detach().abs().mean()
            ),
            "object_p2_w_anchored_interval_score_abs": (
                type_interval_score.detach().abs().mean()
            ),
            "object_p2_w_anchored_interval_score_centered_rms": (
                type_interval_score.detach().float()
                - type_interval_score.detach().float().mean(dim=-1, keepdim=True)
            )
            .square()
            .mean()
            .sqrt(),
            # Contract flag: public/typed S have no additive interval-logit
            # term in Schema38. They only modulate a selected W key.
            "object_p2_independent_s_interval_vote": raw_value.new_zeros(
                (), dtype=torch.float32
            ),
            "object_p2_s_condition_neutral_posterior_l1": (
                s_condition_neutral_posterior_l1
            ),
            "object_p2_s_condition_pre_tanh_rms": torch.stack(
                s_condition_pre_tanh_rms_values
            ).mean().detach(),
            "object_p2_s_condition_saturation_fraction": torch.stack(
                s_condition_saturation_values
            ).mean().detach(),
            "object_p2_selected_w_key_rms": torch.stack(
                selected_w_key_rms_values
            ).mean().detach(),
            "object_p2_w_key_interval_centered_variation_rms": torch.stack(
                w_key_interval_variation_values
            ).mean().detach(),
            "object_p2_typed_s_w_key_delta_rms": torch.stack(
                typed_s_key_delta_rms_values
            ).mean().detach(),
            "object_p2_public_s_w_key_delta_rms": torch.stack(
                public_s_key_delta_rms_values
            ).mean().detach(),
            "object_p2_combined_s_w_key_delta_rms": torch.stack(
                combined_s_key_delta_rms_values
            ).mean().detach(),
            "object_p2_geometry_to_semantic_k_correction_rms": (
                semantic_geometry_correction.detach()
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "object_p2_geometry_condition_neutral_semantic_posterior_l1": (
                semantic_geometry_neutral_posterior_l1.detach()
            ),
            "object_p2_geometry_condition_neutral_semantic_output_l1": (
                semantic_geometry_neutral_output_l1.detach()
            ),
            "object_p2_conditioned_w_interval_score_max_abs": (
                type_interval_precontract_score.detach().abs().amax()
            ),
            "object_p2_type_interval_posterior_entropy": normalized_entropy(
                type_interval_posterior,
                dim=-1,
            ).detach().mean(),
            "object_p2_type_interval_posterior_max": (
                type_interval_posterior.detach().amax(dim=-1).mean()
            ),
            "object_p2_type_interval_null_mass": (
                type_interval_posterior.detach()[..., -1].mean()
            ),
            "object_p2_type_interval_effective_count": torch.exp(
                -(
                    type_interval_posterior.detach().float()
                    * type_interval_posterior.detach()
                    .float()
                    .clamp_min(1.0e-8)
                    .log()
                ).sum(dim=-1)
            ).mean(),
            "object_p2_type_interval_horizon_variation": (
                type_interval_mass.detach()
                .float()
                .std(dim=1, unbiased=False)
                .mean()
            ),
            "object_p2_within_interval_object_posterior_entropy": (
                (inner_entropy * inner_support_weight).sum()
                / inner_support_denominator
            ),
            "object_p2_within_interval_object_posterior_max": (
                (inner_max * inner_support_weight).sum()
                / inner_support_denominator
            ),
            "object_p2_within_interval_object_effective_count": (
                (torch.exp(inner_raw_entropy) * inner_support_weight).sum()
                / inner_support_denominator
            ),
            "object_p2_type_interval_disagreement_max_abs": (
                interval_mass_by_type.detach()
                - interval_mass.detach()[:, :, :, None, :]
            ).abs().amax(),
            "object_p2_complete_field_residual_retained_rms_ratio": (
                residual_retained_rms_ratio
            ),
            "object_p2_complete_field_residual_cancelled_rms_fraction": (
                residual_cancelled_rms_fraction
            ),
            "object_p2_complete_field_residual_cancellation_support_fraction": (
                cancellation_support.detach().float().mean()
            ),
            "object_p2_selected_complete_field_common_rms": common_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_complete_field_residual_rms": residual_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_complete_field_identity_error": (
                selected_type_value.detach().float()
                - (
                    selected_common_type_value.detach().float()
                    + selected_residual_type_value.detach().float()
                )
            ).abs().amax(),
            "object_p2_selected_complete_field_residual_to_common_rms_ratio": (
                residual_value.detach().float().square().mean().sqrt()
                / common_value.detach()
                .float()
                .square()
                .mean()
                .sqrt()
                .clamp_min(1.0e-8)
            ),
            "object_p2_effect_precontract_rms": raw_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_effect_contract_min": shared_effect_scale.detach()
            .float()
            .amin(),
            "object_p2_effect_postcontract_rms": effect_read.physical_sum.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_shared_effect_contract_scale_mean": shared_effect_scale.detach()
            .float()
            .mean(),
            "object_p2_shared_effect_contract_compression": (
                1.0 - shared_effect_scale.detach().float()
            ).mean(),
            "object_p2_semantic_postcontract_rms": effect_read.semantic.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_postcontract_rms": effect_read.geometry.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_complete_field_semantic_common_component_rms": common_semantic_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_complete_field_geometry_common_component_rms": common_geometry_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_complete_field_semantic_residual_component_rms": residual_semantic_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_selected_complete_field_geometry_residual_component_rms": residual_geometry_value.detach()
            .square()
            .mean()
            .sqrt(),
        }
        for type_index, name in enumerate(self.TYPE_NAMES):
            metrics[f"object_p2_{name}_score_max_abs"] = (
                full_source_scores[type_index].detach().abs().amax()
            )
            metrics[f"object_p2_{name}_public_s_condition_only_score_abs"] = (
                interval_public_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_typed_s_condition_only_score_abs"] = (
                interval_typed_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_unconditioned_w_interval_score_abs"] = (
                interval_w_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_interval_posterior_entropy"] = (
                normalized_entropy(
                    type_interval_posterior[..., type_index, :],
                    dim=-1,
                )
                .detach()
                .mean()
            )
            metrics[f"object_p2_{name}_interval_posterior_max"] = (
                type_interval_posterior[..., type_index, :]
                .detach()
                .amax(dim=-1)
                .mean()
            )
            metrics[f"object_p2_{name}_complete_field_null_mass"] = residual_posteriors[
                type_index
            ].detach()[..., -1].mean()
            common_selected_value = selected_common_type_value[
                ..., type_index, :
            ].detach().float()
            residual_selected_value = selected_residual_type_value[
                ..., type_index, :
            ].detach().float()
            metrics[f"object_p2_{name}_selected_complete_field_common_rms"] = (
                common_selected_value.square().mean().sqrt()
            )
            metrics[f"object_p2_{name}_selected_complete_field_residual_rms"] = (
                residual_selected_value.square().mean().sqrt()
            )
            metrics[f"object_p2_{name}_complete_field_common_candidate_rms"] = (
                common_projected_values[type_index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
            metrics[f"object_p2_{name}_complete_field_residual_candidate_rms"] = (
                residual_projected_values[type_index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
            metrics[f"object_p2_{name}_complete_field_shared_contract_scale_mean"] = (
                shared_effect_scale.detach().float().mean()
            )
            metrics[f"object_p2_{name}_complete_field_collapsed_posterior_entropy"] = (
                normalized_entropy(common_posteriors[type_index], dim=-1)
                .detach()
                .mean()
            )
            metrics[f"object_p2_{name}_complete_field_collapsed_posterior_max"] = (
                common_posteriors[type_index].detach().amax(dim=-1).mean()
            )
            type_support_weight = inner_support_weight[..., type_index, :]
            type_support_denominator = (
                type_support_weight.sum() * float(horizon * basis)
            ).clamp_min(1.0)
            metrics[f"object_p2_{name}_within_interval_object_posterior_entropy"] = (
                (
                    inner_entropy[..., type_index, :]
                    * type_support_weight
                ).sum()
                / type_support_denominator
            )
            metrics[f"object_p2_{name}_within_interval_object_posterior_max"] = (
                (
                    inner_max[..., type_index, :]
                    * type_support_weight
                ).sum()
                / type_support_denominator
            )
        metrics["object_p2_geometry_complete_field_collapsed_kc_posterior_entropy"] = metrics[
            "object_p2_geometry_complete_field_collapsed_posterior_entropy"
        ]
        metrics["object_p2_geometry_complete_field_collapsed_kc_posterior_max"] = metrics[
            "object_p2_geometry_complete_field_collapsed_posterior_max"
        ]
        metrics["object_p2_geometry_complete_field_joint_kc_posterior_entropy"] = (
            (
                inner_entropy[..., 1, :]
                * inner_support_weight[..., 1, :]
            ).sum()
            / (
                inner_support_weight[..., 1, :].sum() * float(horizon * basis)
            ).clamp_min(1.0)
        )
        metrics["object_p2_geometry_complete_field_joint_kc_posterior_max"] = (
            (
                inner_max[..., 1, :]
                * inner_support_weight[..., 1, :]
            ).sum()
            / (
                inner_support_weight[..., 1, :].sum() * float(horizon * basis)
            ).clamp_min(1.0)
        )
        metrics["object_p2_geometry_joint_kc_candidate_count"] = raw_value.new_tensor(
            float(objects * camera_count),
            dtype=torch.float32,
        )
        for index in range(intervals):
            metrics[f"object_p2_complete_field_interval_{index}_mass"] = (
                interval_mass[..., index].float().mean()
            )
            for type_index, name in enumerate(self.TYPE_NAMES):
                metrics[f"object_p2_{name}_complete_field_interval_{index}_mass"] = (
                    interval_mass_by_type[..., type_index, index]
                    .detach()
                    .float()
                    .mean()
                )
        if self.training:
            register_gradient_axis_rms_metrics(
                effect_read.effect_by_type,
                metrics,
                (
                    "gradient_tensor_p2_semantic_effect_rms",
                    "gradient_tensor_p2_geometry_effect_rms",
                ),
                dim=-2,
            )
        return effect_read, metrics


class ZeroPreservingObjectConsequence(nn.Module):
    """Exact identity when the P2 effect is zero."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fact = nn.Linear(hidden, hidden, bias=False)
        self.interaction = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        *,
        factual_base: Tensor,
        effect: TypedP2EffectRead,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectConsequenceState, dict[str, Tensor]]:
        if not isinstance(effect, TypedP2EffectRead):
            raise TypeError("Schema38 consequence requires TypedP2EffectRead")
        effect.validate()
        effect_by_type = effect.effect_by_type
        expected_typed = (*factual_base.shape[:-1], 2, factual_base.shape[-1])
        if tuple(effect_by_type.shape) != tuple(expected_typed):
            raise ValueError("factual base and effect must align")
        factual_interaction = torch.tanh(self.fact(factual_base))[..., None, :]
        interaction_by_type = self.interaction(
            factual_interaction * effect_by_type
        )
        physical_effect = effect_by_type.sum(dim=-2)
        interaction = interaction_by_type.sum(dim=-2)
        protected = factual_base + physical_effect + interaction
        state = ObjectConsequenceState(
            factual_base=factual_base,
            effect_by_type=effect_by_type,
            interaction_by_type=interaction_by_type,
            protected_consequence=protected,
        )
        state.validate()
        if not collect_diagnostics:
            return state, {}
        metrics = {
            "object_consequence_effect_rms": physical_effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_consequence_interaction_rms": interaction.detach().float().square().mean().sqrt(),
            "object_consequence_ratio": (
                (physical_effect + interaction).detach().float().square().mean().sqrt()
                / factual_base.detach().float().square().mean().sqrt().clamp_min(1e-6)
            ),
        }
        for type_index, name in enumerate(ObjectFutureEffectReader.TYPE_NAMES):
            metrics[f"object_consequence_{name}_effect_rms"] = effect_by_type[
                ..., type_index, :
            ].detach().float().square().mean().sqrt()
            metrics[f"object_consequence_{name}_interaction_rms"] = (
                interaction_by_type[..., type_index, :]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
        return state, metrics


class ObjectPolicyPlanCompiler(nn.Module):
    """Consequence-conditioned precision/effect/temporal/state-change P3."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.precision_action = nn.Linear(hidden, hidden, bias=False)
        self.precision_innovation = nn.Linear(hidden, hidden, bias=False)
        self.precision_lane = nn.Linear(hidden, hidden, bias=False)
        self.effect_lane = nn.Linear(hidden, hidden, bias=False)
        self.temporal_source = nn.Linear(hidden, hidden, bias=False)
        self.temporal_consequence = nn.Linear(hidden, hidden, bias=False)
        self.temporal_action = nn.Linear(hidden, hidden, bias=False)
        self.temporal_lane = nn.Linear(hidden, hidden, bias=False)
        self.state_change_action = nn.Linear(hidden, hidden, bias=False)
        self.state_change_temporal = nn.Linear(hidden, hidden, bias=False)
        self.state_change_lane = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        *,
        p1_factual_detail: Tensor,
        consequence: ObjectConsequenceState,
        intent: PolicyIntentDock,
        action_query: Tensor,
        p1_policy_residual: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if (
            tuple(action_query.shape) != expected
            or tuple(p1_factual_detail.shape) != expected
        ):
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        if tuple(p1_policy_residual.shape) != expected:
            raise ValueError("P3 policy precision residual must align as [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        consequence.validate()
        # The static factual consequence is already the protected base.
        # Optional lanes may only encode source-exclusive zero-centred
        # innovations; neither static detail nor the live policy residual is
        # copied into another protected carrier. Precision is the sole owner
        # of the cached P1 factual signal below.
        # V120's live P1 write legitimately conditioned both the P2 query and
        # P3 precision. Schema37 removed the precision consumer together with
        # the unsafe protected-fact write. Restore only the legal consumer: a
        # one-sided contracted dynamic residual can refine an existing factual
        # precision feature, but cannot synthesize precision when that fact is
        # zero and never enters the protected consequence.
        static_precision = self.precision_innovation(p1_factual_detail)
        contracted_policy_residual, precision_residual_scale = smooth_rms_contract(
            p1_policy_residual,
            0.35,
        )
        precision_fact_gate, precision_fact_denominator = (
            variance_floored_centered_norm(static_precision, 0.25)
        )
        precision_dynamic_interaction = (
            torch.tanh(precision_fact_gate) * contracted_policy_residual
        )
        precision_source = static_precision + precision_dynamic_interaction
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query))
            * precision_source
        )
        typed_effect = consequence.typed_effect() + consequence.typed_interaction()
        effect_semantic = self.effect_lane(typed_effect[..., 0, :])
        effect_geometry = self.effect_lane(typed_effect[..., 1, :])
        temporal_effect = typed_effect.sum(dim=-2)
        temporal_source = intent.temporal_control[:, :, None].expand(
            -1, -1, self.basis, -1
        )
        # Temporal is a W-effect relation, not a second factual carrier.
        # Requiring S, W effect and action makes neutral W an exact temporal
        # null while leaving the independently owned state-change lane intact.
        temporal_source_value = self.temporal_source(temporal_source)
        temporal_action_value = torch.tanh(self.temporal_action(action_query))

        def typed_temporal_lane(type_index: int) -> Tensor:
            return self.temporal_lane(
                temporal_source_value
                * torch.tanh(
                    self.temporal_consequence(typed_effect[..., type_index, :])
                )
                * temporal_action_value
            )

        temporal_semantic = typed_temporal_lane(0)
        temporal_geometry = typed_temporal_lane(1)
        state_change_source = intent.state_change_evidence[:, None, None].expand(
            -1, self.horizon, self.basis, -1
        )
        state_change_modulation = torch.tanh(
            (
                self.state_change_action(action_query)
                + self.state_change_temporal(temporal_source)
            )
            / math.sqrt(2.0)
        )
        state_change = 0.05 * self.state_change_lane(
            state_change_source * state_change_modulation
        )
        lanes = [
            precision,
            effect_semantic,
            effect_geometry,
            temporal_semantic,
            temporal_geometry,
            state_change,
        ]
        lanes = [smooth_rms_contract(value, 0.35)[0] for value in lanes]
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            precision=lanes[0],
            effect_semantic=lanes[1],
            effect_geometry=lanes[2],
            temporal_semantic=lanes[3],
            temporal_geometry=lanes[4],
            state_change=lanes[5],
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        metrics = {
            "object_p3_precision_factual_input_rms": p1_factual_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_static_projected_rms": static_precision.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_combined_source_rms": precision_source.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_dynamic_input_rms": p1_policy_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_dynamic_contracted_rms": (
                contracted_policy_residual.detach().float().square().mean().sqrt()
            ),
            "object_p3_precision_dynamic_interaction_rms": (
                precision_dynamic_interaction.detach().float().square().mean().sqrt()
            ),
            "object_p3_precision_fact_gate_rms": (
                precision_fact_gate.detach().float().square().mean().sqrt()
            ),
            "object_p3_precision_fact_denominator_min": (
                precision_fact_denominator.detach().float().amin()
            ),
            "object_p3_precision_dynamic_contract_min": (
                precision_residual_scale.detach().float().amin()
            ),
            "object_p3_precision_rms": lanes[0].detach().float().square().mean().sqrt(),
            "object_p3_effect_rms": bank.effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_effect_semantic_rms": lanes[1].detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_effect_geometry_rms": lanes[2].detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_source_rms": temporal_source.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_consequence_rms": temporal_effect
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_rms": bank.temporal.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_semantic_rms": lanes[3].detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_temporal_geometry_rms": lanes[4].detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_state_change_rms": lanes[5].detach().float().square().mean().sqrt(),
        }
        if self.training:
            for name, lane in zip(bank.source_names, lanes, strict=True):
                register_gradient_rms_metric(
                    lane,
                    metrics,
                    f"gradient_tensor_{name}_rms",
                )
        return bank, metrics


__all__ = [
    "ObjectConsequenceState",
    "ObjectFutureEffectReader",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "TypedP2EffectRead",
    "ZeroPreservingObjectConsequence",
]
