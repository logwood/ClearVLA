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
    """Bounded P2 reads with a public time prior and matched typed owners.

    Semantic and geometry are complementary effect values.  They share one
    protected public-S temporal prior, then each adds only its matching typed-S
    and supervised-W evidence before selecting an interval/null and an object.
    Status is deliberately absent: without independent visibility labels it
    has neither an active target nor a legal action value or route vote.
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
        self.intent_query = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.public_interval_query = nn.Linear(hidden, hidden, bias=False)
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
        # Camera is a physical geometry hypothesis axis, not a preprocessing
        # nuisance axis.  Keep KxC through the action-conditioned posterior;
        # only the selected geometry value below may reduce camera.
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
        common_projected_values = (
            self.semantic_value(dynamics.semantic_common),
            self.transport_value(transport_common),
        )
        residual_projected_values = (
            self.semantic_value(dynamics.semantic_interval_residual),
            self.transport_value(transport_residual),
        )
        # Do not contract typed candidates independently.  P2 owns one shared
        # physical effect budget after both conditional reads complete.
        common_source_values = common_projected_values
        residual_source_values = residual_projected_values
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
        common_coordinate = (
            camera_coordinate
            + dynamics.transport_common.float()
        ).clamp(-1.0, 1.0)
        future_coordinate = (
            camera_coordinate[:, None]
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        common_coordinate_delta = (
            coordinate_query[:, :, :, None, None]
            - common_coordinate[:, None, None]
        )
        coordinate_delta = (
            coordinate_query[:, :, :, None, None, None]
            - future_coordinate[:, None, None]
        )

        def covariance_aware_distance(delta: Tensor, covariance: Tensor) -> Tensor:
            # One half-cell standard deviation in normalized 8x8 coordinates
            # is the identity metric.  Therefore zero predicted covariance
            # exactly recovers the former Euclidean squared distance, while a
            # larger PSD covariance broadens uncertainty without moving its
            # centre or changing the value path.
            variance_floor = 1.0 / float(7 * 7)
            xx = covariance[..., 0].float().clamp_min(0.0) + variance_floor
            xy = covariance[..., 1].float()
            yy = covariance[..., 2].float().clamp_min(0.0) + variance_floor
            determinant = (xx * yy - xy.square()).clamp_min(variance_floor**2)
            dx, dy = delta[..., 0], delta[..., 1]
            return variance_floor * (
                yy * dx.square() - 2.0 * xy * dx * dy + xx * dy.square()
            ) / determinant

        common_coordinate_distance = covariance_aware_distance(
            common_coordinate_delta,
            dynamics.transport_covariance.float().mean(dim=1)[:, None, None],
        )
        coordinate_distance = covariance_aware_distance(
            coordinate_delta,
            dynamics.transport_covariance.float()[:, None, None],
        )
        # Exact coordinate agreement is positive evidence; the old [-1,0]
        # term could only punish and therefore could not establish a geometry
        # owner when semantic scores were diffuse.
        common_camera_score = (
            1.0 - 0.5 * common_coordinate_distance
        ).clamp(-1.0, 1.0)
        camera_score = (1.0 - 0.5 * coordinate_distance).clamp(-1.0, 1.0)
        # These scores stay on KxC.  Camera evidence enters as the physical
        # posterior measure below, after the noisy action has scored every
        # joint hypothesis; there is no pre-read log-mixture or value mean.
        common_coordinate_score = common_camera_score
        coordinate_score = camera_score
        # Uniform weights above are a finite arithmetic fallback only.  They
        # must not turn missing camera evidence into a semantic geometry
        # prior.  Other P2 fields may still use the valid object; the geometry
        # contribution itself is exactly neutral without an observed camera.
        common_coordinate_score = torch.where(
            camera_availability[:, None, None] > 0.0,
            common_coordinate_score,
            torch.zeros_like(common_coordinate_score),
        )
        coordinate_score = torch.where(
            camera_availability[:, None, None, None] > 0.0,
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
        common_validities = (
            current_validity,
            geometry_current_validity,
        )
        residual_validities = (
            current_validity[:, None].expand(-1, intervals, -1),
            geometry_current_validity[:, None].expand(
                -1, intervals, -1, -1
            ),
        )
        temperature = self._temperatures().to(device=action_query.device)
        public_interval_query = self._bounded_unit(
            self.public_interval_query(action_query)
        )
        public_interval_key = self._bounded_unit(
            self.public_interval_key(intent.interval_residual_key)
        )
        # This is the one protected temporal prior shared by all active effect
        # owners.  It contains no typed/W value and therefore cannot let one
        # owner select another owner's future interval.
        public_interval_score = torch.einsum(
            "btqh,bih->btqi",
            public_interval_query,
            public_interval_key,
        )
        common_source_scores: list[Tensor] = []
        residual_source_scores: list[Tensor] = []
        common_intent_scores: list[Tensor] = []
        residual_intent_scores: list[Tensor] = []
        interval_public_scores: list[Tensor] = []
        interval_typed_scores: list[Tensor] = []
        interval_w_scores: list[Tensor] = []
        common_bounded_logits: list[Tensor] = []
        residual_bounded_logits: list[Tensor] = []
        common_posteriors: list[Tensor] = []
        residual_object_posteriors: list[Tensor] = []
        interval_supports: list[Tensor] = []
        selected_common_values: list[Tensor] = []
        selected_residual_values_by_interval: list[Tensor] = []
        for type_index in range(len(self.TYPE_NAMES)):
            intent_type_index = self.S_TYPE_INDEX_BY_P2[type_index]
            query = self._bounded_unit(
                self.source_query[type_index](action_query)
            )
            common_source_key = self._bounded_unit(
                self.source_key[type_index](common_source_fields[type_index])
            )
            residual_source_key = self._bounded_unit(
                self.source_key[type_index](residual_source_fields[type_index])
            )
            if type_index == 0:
                common_source_score = torch.einsum(
                    "btqh,bkh->btqk", query, common_source_key
                )
                residual_source_score = torch.einsum(
                    "btqh,bikh->btqik", query, residual_source_key
                )
            else:
                common_source_score = torch.einsum(
                    "btqh,bkch->btqkc", query, common_source_key
                )
                residual_source_score = torch.einsum(
                    "btqh,bikch->btqikc", query, residual_source_key
                )
            common_public_key = self.common_intent_key[type_index](
                intent.common_key
            )[:, None]
            common_typed_key = self.typed_intent_key[type_index](
                typed_common_intent[..., intent_type_index, :]
            )
            common_intent_key = self._bounded_unit(
                common_public_key + common_typed_key
            )
            residual_typed_key = self._bounded_unit(
                self.typed_intent_key[type_index](
                    typed_residual_intent[..., intent_type_index, :]
                )
            )
            intent_query = self._bounded_unit(
                self.intent_query[type_index](action_query)
            )
            common_intent_score = torch.einsum(
                "btqh,bkh->btqk", intent_query, common_intent_key
            )
            residual_intent_score = torch.einsum(
                "btqh,bikh->btqik", intent_query, residual_typed_key
            )
            if type_index == 1:
                common_intent_score = common_intent_score[..., None].expand_as(
                    common_source_score
                )
                residual_intent_score = residual_intent_score[..., None].expand_as(
                    residual_source_score
                )
            common_bounded_logit = (
                temperature[0] * common_source_score
                + temperature[1] * common_intent_score
            )
            residual_bounded_logit = (
                temperature[0] * residual_source_score
                + temperature[1] * residual_intent_score
            )
            if type_index == 1:
                common_bounded_logit = (
                    common_bounded_logit
                    + temperature[2] * common_coordinate_score
                )
                residual_bounded_logit = (
                    residual_bounded_logit
                    + temperature[2] * coordinate_score
                )

            common_validity = common_validities[type_index]
            residual_validity = residual_validities[type_index]
            current_support = common_validity > 0.0
            residual_support = residual_validity > 0.0
            common_has_support = current_support.flatten(start_dim=1).any(dim=-1)
            interval_has_support = residual_support.flatten(start_dim=2).any(dim=-1)
            common_logit = common_bounded_logit + torch.where(
                current_support,
                common_validity.clamp_min(1e-6).log(),
                torch.zeros_like(common_validity),
            )[:, None, None]
            common_logit = common_logit.masked_fill(
                ~current_support[:, None, None],
                -torch.inf,
            )
            # Softmax over an all-invalid row is undefined.  The replacement
            # logits are bookkeeping only; multiplying by has_support makes
            # the returned common posterior and value algebraically zero.
            common_logit = torch.where(
                common_has_support[:, None, None, None],
                common_logit.flatten(start_dim=3),
                torch.zeros_like(common_logit.flatten(start_dim=3)),
            )
            common_posterior = torch.softmax(common_logit, dim=-1)
            common_posterior = common_posterior * common_has_support[
                :, None, None, None
            ].to(dtype=common_posterior.dtype)
            common_source_value = common_source_values[type_index].reshape(
                batch, -1, self.hidden
            )
            common_selected = torch.einsum(
                "btqn,bnh->btqh",
                common_posterior.to(dtype=common_source_value.dtype),
                common_source_value,
            )

            residual_logit = residual_bounded_logit + torch.where(
                residual_support,
                residual_validity.clamp_min(1e-6).log(),
                torch.zeros_like(residual_validity),
            )[:, None, None]
            residual_logit = residual_logit.masked_fill(
                ~residual_support[:, None, None],
                -torch.inf,
            )
            residual_logit = residual_logit.reshape(
                batch, horizon, basis, intervals, -1
            )
            # Object ownership is normalized only inside its already chosen
            # interval. Geometry's candidate is the joint KxC hypothesis, so
            # camera compression happens only after this action-conditioned
            # posterior is complete.
            residual_logit = torch.where(
                interval_has_support[:, None, None, :, None],
                residual_logit,
                torch.zeros_like(residual_logit),
            )
            residual_object_posterior = torch.softmax(
                residual_logit,
                dim=-1,
            )
            residual_object_posterior = (
                residual_object_posterior
                * interval_has_support[:, None, None, :, None].to(
                    dtype=residual_object_posterior.dtype
                )
            )
            residual_source_value = residual_source_values[type_index].reshape(
                batch, intervals, -1, self.hidden
            )
            residual_selected_by_interval = torch.einsum(
                "btqin,binh->btqih",
                residual_object_posterior.to(
                    dtype=residual_source_value.dtype
                ),
                residual_source_value,
            )
            # The shared temporal owner reads the compatibility of the object
            # that this type would actually use inside each interval.  This
            # retains I/K/type evidence until the named temporal consumer while
            # keeping raw coordinate distance out of the time score.
            interval_w_score = (
                residual_object_posterior.float()
                * residual_source_score.float().reshape(
                    batch, horizon, basis, intervals, -1
                )
            ).sum(dim=-1)
            interval_typed_score = (
                residual_object_posterior.float()
                * residual_intent_score.float().reshape(
                    batch, horizon, basis, intervals, -1
                )
            ).sum(dim=-1)
            common_source_scores.append(common_source_score)
            residual_source_scores.append(residual_source_score)
            common_intent_scores.append(common_intent_score)
            residual_intent_scores.append(residual_intent_score)
            interval_public_scores.append(public_interval_score)
            interval_typed_scores.append(interval_typed_score)
            interval_w_scores.append(interval_w_score)
            common_bounded_logits.append(common_bounded_logit)
            residual_bounded_logits.append(residual_bounded_logit)
            common_posteriors.append(common_posterior)
            residual_object_posteriors.append(residual_object_posterior)
            interval_supports.append(interval_has_support)
            selected_common_values.append(common_selected)
            selected_residual_values_by_interval.append(
                residual_selected_by_interval
            )

        # Public S is the shared temporal prior.  Each active owner adds only
        # its matching typed-S and supervised-W likelihood before selecting a
        # future interval or exact-zero null.  This is a product-of-evidence
        # factorization, not an outer type competition: semantic cannot choose
        # geometry's value and a neutral status field has no vote at all.
        interval_typed_score_by_type = torch.stack(interval_typed_scores, dim=3)
        interval_w_score_by_type = torch.stack(interval_w_scores, dim=3)
        type_interval_precontract_score = (
            public_interval_score[..., None, :]
            + interval_typed_score_by_type
            + interval_w_score_by_type
        )
        type_interval_score = torch.tanh(type_interval_precontract_score)
        type_interval_bounded_logit = temperature[1] * type_interval_score
        type_interval_has_support = torch.stack(interval_supports, dim=1)
        type_interval_logit = type_interval_bounded_logit.masked_fill(
            ~type_interval_has_support[:, None, None],
            -torch.inf,
        )
        # Each type owns one null competing with I interval measures, each of
        # which already owns a conditional semantic-K or geometry-KxC
        # posterior. The type-specific offset preserves the corresponding
        # neutral 1/(I*N+1) prior without turning candidate count into a time
        # score.
        type_candidate_count = type_interval_logit.new_tensor(
            (float(max(objects, 1)), float(max(objects * camera_count, 1)))
        )
        type_null_logit = -type_candidate_count.log()[None, None, None, :, None]
        type_null_logit = type_null_logit.expand(
            batch, horizon, basis, len(self.TYPE_NAMES), 1
        )
        type_interval_posterior = torch.softmax(
            torch.cat((type_interval_logit, type_null_logit), dim=-1),
            dim=-1,
        )
        type_interval_mass = type_interval_posterior[..., :-1]

        selected_residual_values: list[Tensor] = []
        residual_posteriors: list[Tensor] = []
        for type_index, (
            residual_selected_by_interval,
            residual_object_posterior,
        ) in enumerate(
            zip(
                selected_residual_values_by_interval,
                residual_object_posteriors,
            )
        ):
            interval_mass_for_type = type_interval_mass[..., type_index, :]
            selected_residual_values.append(
                torch.einsum(
                    "btqi,btqih->btqh",
                    interval_mass_for_type.to(
                        dtype=residual_selected_by_interval.dtype
                    ),
                    residual_selected_by_interval,
                )
            )
            # Retain the historical flattened posterior shape for diagnostics
            # and consumers of lossless metrics.  Its factorization is exact:
            # p_z(i,k)=pi_z(i)*rho_z(k|i), with one typed null at the end.
            joint_real = (
                interval_mass_for_type[..., None] * residual_object_posterior
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
        residual_value_by_interval_and_type = torch.stack(
            selected_residual_values_by_interval,
            dim=4,
        )
        selected_common_type_value = torch.stack(selected_common_values, dim=3)
        selected_residual_type_value = torch.stack(selected_residual_values, dim=3)
        selected_type_value = (
            selected_common_type_value + selected_residual_type_value
        )
        effect_read, raw_value, shared_effect_scale = (
            self._fuse_complementary_values(selected_type_value)
        )
        if not collect_diagnostics:
            return effect_read, {}
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
            tuple(score.detach().abs().mean() for score in (
                *common_source_scores,
                *residual_source_scores,
            ))
        ).mean()
        source_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in (
                *common_source_scores,
                *residual_source_scores,
            ))
        ).amax()
        intent_score_abs = torch.stack(
            tuple(score.detach().abs().mean() for score in (
                *common_intent_scores,
                *residual_intent_scores,
                public_interval_score,
                type_interval_score,
            ))
        ).mean()
        intent_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in (
                *common_intent_scores,
                *residual_intent_scores,
                public_interval_score,
                type_interval_score,
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
            "object_p2_intent_score_abs": intent_score_abs,
            "object_p2_intent_score_max_abs": intent_score_max,
            "object_p2_coordinate_score_abs": 0.5 * (
                common_coordinate_score.detach().abs().mean()
                + coordinate_score.detach().abs().mean()
            ),
            "object_p2_coordinate_score_max_abs": torch.maximum(
                common_coordinate_score.detach().abs().amax(),
                coordinate_score.detach().abs().amax(),
            ),
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
                    *common_bounded_logits,
                    *residual_bounded_logits,
                    type_interval_bounded_logit,
                ))
            ).amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_common_posterior_entropy": common_posterior_entropy,
            "object_p2_common_posterior_max": common_posterior_max,
            "object_p2_residual_posterior_entropy": residual_posterior_entropy,
            "object_p2_residual_posterior_max": residual_posterior_max,
            "object_p2_residual_null_mass": residual_null_mass,
            "object_p2_public_interval_score_abs": public_interval_score.detach()
            .abs()
            .mean(),
            "object_p2_public_interval_score_max_abs": public_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_type_interval_score_abs": type_interval_score.detach()
            .abs()
            .mean(),
            "object_p2_type_interval_score_max_abs": type_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_type_interval_typed_score_abs": (
                interval_typed_score_by_type.detach().abs().mean()
            ),
            "object_p2_type_interval_w_score_abs": (
                interval_w_score_by_type.detach().abs().mean()
            ),
            "object_p2_type_interval_precontract_score_max_abs": (
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
            "object_p2_residual_retained_rms_ratio": residual_retained_rms_ratio,
            "object_p2_residual_cancelled_rms_fraction": (
                residual_cancelled_rms_fraction
            ),
            "object_p2_residual_cancellation_support_fraction": (
                cancellation_support.detach().float().mean()
            ),
            "object_p2_protected_common_rms": common_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_optional_residual_rms": residual_value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_to_common_rms_ratio": (
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
            "object_p2_common_semantic_component_rms": common_semantic_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_common_geometry_component_rms": common_geometry_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_semantic_component_rms": residual_semantic_value.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_geometry_component_rms": residual_geometry_value.detach()
            .square()
            .mean()
            .sqrt(),
        }
        for type_index, name in enumerate(self.TYPE_NAMES):
            metrics[f"object_p2_{name}_score_max_abs"] = torch.maximum(
                common_source_scores[type_index].detach().abs().amax(),
                residual_source_scores[type_index].detach().abs().amax(),
            )
            metrics[f"object_p2_{name}_interval_public_score_abs"] = (
                interval_public_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_interval_typed_score_abs"] = (
                interval_typed_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_interval_w_score_abs"] = (
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
            metrics[f"object_p2_{name}_residual_null_mass"] = residual_posteriors[
                type_index
            ].detach()[..., -1].mean()
            common_selected_value = selected_common_type_value[
                ..., type_index, :
            ].detach().float()
            residual_selected_value = selected_residual_type_value[
                ..., type_index, :
            ].detach().float()
            metrics[f"object_p2_{name}_common_selected_value_rms"] = (
                common_selected_value.square().mean().sqrt()
            )
            metrics[f"object_p2_{name}_residual_selected_value_rms"] = (
                residual_selected_value.square().mean().sqrt()
            )
            metrics[f"object_p2_{name}_common_projected_candidate_value_rms"] = (
                common_projected_values[type_index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
            metrics[f"object_p2_{name}_residual_projected_candidate_value_rms"] = (
                residual_projected_values[type_index]
                .detach()
                .float()
                .square()
                .mean()
                .sqrt()
            )
            metrics[f"object_p2_{name}_common_value_contract_scale_mean"] = (
                shared_effect_scale.detach().float().mean()
            )
            metrics[f"object_p2_{name}_residual_value_contract_scale_mean"] = (
                shared_effect_scale.detach().float().mean()
            )
            metrics[f"object_p2_{name}_common_posterior_entropy"] = (
                normalized_entropy(common_posteriors[type_index], dim=-1)
                .detach()
                .mean()
            )
            metrics[f"object_p2_{name}_common_posterior_max"] = (
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
        metrics["object_p2_geometry_common_joint_kc_posterior_entropy"] = metrics[
            "object_p2_geometry_common_posterior_entropy"
        ]
        metrics["object_p2_geometry_common_joint_kc_posterior_max"] = metrics[
            "object_p2_geometry_common_posterior_max"
        ]
        metrics["object_p2_geometry_residual_joint_kc_posterior_entropy"] = (
            (
                inner_entropy[..., 1, :]
                * inner_support_weight[..., 1, :]
            ).sum()
            / (
                inner_support_weight[..., 1, :].sum() * float(horizon * basis)
            ).clamp_min(1.0)
        )
        metrics["object_p2_geometry_residual_joint_kc_posterior_max"] = (
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
            metrics[f"object_p2_residual_interval_{index}_mass"] = (
                interval_mass[..., index].float().mean()
            )
            for type_index, name in enumerate(self.TYPE_NAMES):
                metrics[f"object_p2_{name}_residual_interval_{index}_mass"] = (
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
            raise TypeError("Schema37 consequence requires TypedP2EffectRead")
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
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if (
            tuple(action_query.shape) != expected
            or tuple(p1_factual_detail.shape) != expected
        ):
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        consequence.validate()
        # The static factual consequence is already the protected base.
        # Optional lanes may only encode source-exclusive zero-centred
        # innovations; neither static detail nor the live policy residual is
        # copied into another protected carrier. Precision is the sole owner
        # of the cached P1 factual signal below.
        # P3 precision owns the cached high-resolution factual detail.  The
        # live V120 policy-block residual is a P2 query refinement, not another
        # precision value.  Feeding it here gave one dynamic tensor two action
        # exits and let Schema35 drive both the P1 block and this lane to their
        # amplitude limits while bypassing W.  The action query still provides
        # the legal time/noisy-action modulation.
        static_precision = self.precision_innovation(p1_factual_detail)
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query))
            * static_precision
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
            "object_p3_precision_input_rms": p1_factual_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_static_input_rms": p1_factual_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
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
