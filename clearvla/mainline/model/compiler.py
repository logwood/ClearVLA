"""Typed future-effect reads and consequence-conditioned P3 compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import PolicyRoleDeltaBank, smooth_rms_contract
from .types import FutureObjectDynamics, PolicyIntentDock, normalized_entropy


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    effect: Tensor
    interaction: Tensor
    protected_consequence: Tensor


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """Four optional innovations around one protected consequence."""

    protected_base: Tensor
    precision: Tensor
    effect: Tensor
    temporal: Tensor
    state_change: Tensor

    @property
    def source_names(self) -> tuple[str, ...]:
        return (
            "p3_precision",
            "p3_effect",
            "p3_temporal",
            "p3_state_change",
        )

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        for name in ("precision", "effect", "temporal", "state_change"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack(
                (
                    self.precision,
                    self.effect,
                    self.temporal,
                    self.state_change,
                ),
                dim=1,
            ),
            source_names=self.source_names,
            source_depths=(int(source_depth),) * 4,
            protected_detail=self.protected_base,
        )


class ObjectFutureEffectReader(nn.Module):
    """Bounded P2 reads with shared time and typed object ownership.

    Protected common effects keep their per-type K reads.  Optional residual
    effects first select one shared future interval (or one exact-zero null),
    then semantic/geometry/status independently select an object inside that
    interval.  Public S, typed S and supervised W compatibility jointly decide
    *when*; object count and raw coordinate distance cannot silently take over
    that shared temporal decision.
    """

    TYPE_NAMES = ("semantic", "geometry", "status")
    # S/W retain semantic/appearance/geometry in that order.  P2 consumes
    # semantic/geometry/status, where status is decoded from W appearance and
    # transport is decoded from W geometry.  Never rely on matching integer
    # positions across these two differently named type systems.
    S_TYPE_INDEX_BY_P2 = (0, 2, 1)
    # Each complementary lane owns an equal share of the established 0.35
    # hidden-update RMS budget.  This is a one-sided contract: it can prevent
    # native-unit magnitude from monopolising the protected sum, but it never
    # expands a weak or exactly-zero lane.
    COMPLEMENTARY_VALUE_MAX_RMS = 0.35 / math.sqrt(3.0)

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
                nn.Linear(2, hidden, bias=False),
            )
        )
        self.intent_query = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.public_intent_key = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.typed_intent_key = nn.ModuleList(
            nn.Linear(route_dim, hidden, bias=False) for _ in self.TYPE_NAMES
        )
        self.coordinate_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.status_value = nn.Linear(2, hidden, bias=False)
        # The three sources are complementary facts, not alternatives.  Keep
        # their variance-preserving symmetric sum as a protected path and
        # learn only a low-rank correction from type *contrasts*.  This avoids
        # both the old type selector and the later fixed /3 amplitude loss.
        contrast_rank = max(4, int(hidden) // 8)
        self.type_contrast_down = nn.Linear(
            len(self.TYPE_NAMES) * hidden,
            contrast_rank,
            bias=False,
        )
        self.type_contrast_up = nn.Linear(
            contrast_rank,
            hidden,
            bias=False,
        )
        self.type_contrast_scale = nn.Parameter(
            torch.full((hidden,), 1.0e-4, dtype=torch.float32)
        )
        self.type_contrast_activation = nn.GELU()
        self.temperature_logit = nn.Parameter(torch.zeros(3))

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @classmethod
    def _contract_complementary_value(
        cls,
        value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the one-sided, zero-preserving P2 value-unit contract.

        Keeping this boundary named makes two properties executable rather
        than documentary: weak values are never amplified, and exact zero
        remains exact zero.  The operation is applied before any soft read so
        a high-native-unit candidate cannot win the protected sum merely by
        being selected more often.
        """

        return smooth_rms_contract(value, cls.COMPLEMENTARY_VALUE_MAX_RMS)

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        value = value.float()
        return value / (
            value.square().sum(dim=-1, keepdim=True) + float(norm_floor) ** 2
        ).sqrt()

    def _fuse_complementary_values(
        self,
        selected_type_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fuse typed P2 values through an anchored contrast residual.

        Each type has already made its own interval/object/null decision.  A
        second learned softmax would turn those complementary decisions back
        into mutually exclusive alternatives and let a high-mass null type
        suppress the other two values.  Their symmetric sum divided by
        ``sqrt(3)`` therefore remains outside the learnable branch.  Unlike a
        fixed mean, this preserves the scale of independent complementary
        channels at initialization.  A small LayerScale-style residual reads
        only pairwise contrasts, providing cross-type capacity without
        learning another selector or a common-carrier bypass.

        Bias-free projections make the all-null state an exact algebraic zero.
        Identical type values also have zero contrast, so the learned branch
        cannot rewrite information that all three owners already agree on.
        """

        if selected_type_value.ndim < 2 or int(selected_type_value.shape[-2]) != len(
            self.TYPE_NAMES
        ):
            raise ValueError("P2 complementary values must retain the three-type axis")
        # Candidate values have already crossed their per-type unit contract
        # before the soft read.  Fusion therefore only combines comparable,
        # bounded owner values and does not perform a second normalization.
        typed_value = selected_type_value.float()
        base = typed_value.sum(dim=-2) / math.sqrt(float(len(self.TYPE_NAMES)))
        # Cyclic pairwise differences are exactly zero for bit-identical type
        # values, unlike subtracting a floating-point three-way mean.
        contrast = torch.stack(
            (
                typed_value[..., 0, :] - typed_value[..., 1, :],
                typed_value[..., 1, :] - typed_value[..., 2, :],
                typed_value[..., 2, :] - typed_value[..., 0, :],
            ),
            dim=-2,
        )
        flat_contrast = contrast.flatten(-2).to(dtype=selected_type_value.dtype)
        residual = self.type_contrast_up(
            self.type_contrast_activation(self.type_contrast_down(flat_contrast))
        )
        # Keep the protected base and LayerScale multiply in FP32.  The
        # resulting tensor returns to the active policy dtype at the boundary.
        scaled_residual = self.type_contrast_scale.float() * residual.float()
        fused = base + scaled_residual
        return (
            fused.to(dtype=selected_type_value.dtype),
            base,
            contrast,
            scaled_residual,
        )

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: PolicyIntentDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
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
        camera_weight = (
            dynamics.camera_weights.float()[..., 0].clamp_min(0.0)
            * camera_availability
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
        # Geometry values remain object-level complementary effects, but the
        # same physical camera measure used by coordinate scoring performs the
        # reduction.  The camera-specific transport/covariance themselves stay
        # intact through the coordinate consumer below.
        transport_object = (
            dynamics.transport_mean.float()
            * normalized_camera_weight[:, None, :, :, None]
        ).sum(dim=3)
        transport_common = transport_object.mean(dim=1)
        transport_residual = transport_object - transport_common[:, None]
        status_common = torch.cat(
            (dynamics.visibility_common, dynamics.persistence_common), dim=-1
        )
        status_residual = torch.cat(
            (
                dynamics.visibility_interval_residual,
                dynamics.persistence_interval_residual,
            ),
            dim=-1,
        )
        common_source_fields = (
            dynamics.semantic_common,
            transport_common,
            status_common,
        )
        residual_source_fields = (
            dynamics.semantic_interval_residual,
            transport_residual,
            status_residual,
        )
        common_projected_values = (
            self.semantic_value(dynamics.semantic_common),
            self.transport_value(transport_common),
            self.status_value(status_common),
        )
        residual_projected_values = (
            self.semantic_value(dynamics.semantic_interval_residual),
            self.transport_value(transport_residual),
            self.status_value(status_residual),
        )
        common_value_contracts = tuple(
            self._contract_complementary_value(value)
            for value in common_projected_values
        )
        residual_value_contracts = tuple(
            self._contract_complementary_value(value)
            for value in residual_projected_values
        )
        common_source_values = tuple(value for value, _ in common_value_contracts)
        residual_source_values = tuple(value for value, _ in residual_value_contracts)
        common_value_contract_scales = tuple(
            scale for _, scale in common_value_contracts
        )
        residual_value_contract_scales = tuple(
            scale for _, scale in residual_value_contracts
        )
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
        camera_log_weight = normalized_camera_weight.clamp_min(1.0e-8).log()
        # A log mixture scores real camera hypotheses without first averaging
        # their normalized-image coordinates into a point that belongs to no
        # view.  Since weights sum to one and every component score lies in
        # [-1,1], both mixed scores preserve the same bounded contract.
        common_coordinate_score = torch.logsumexp(
            common_camera_score + camera_log_weight[:, None, None],
            dim=-1,
        ).clamp(-1.0, 1.0)
        coordinate_score = torch.logsumexp(
            camera_score + camera_log_weight[:, None, None, None],
            dim=-1,
        ).clamp(-1.0, 1.0)
        # Uniform weights above are a finite arithmetic fallback only.  They
        # must not turn missing camera evidence into a semantic geometry
        # prior.  Other P2 fields may still use the valid object; the geometry
        # contribution itself is exactly neutral without an observed camera.
        common_coordinate_score = torch.where(
            camera_has_support[:, None, None],
            common_coordinate_score,
            torch.zeros_like(common_coordinate_score),
        )
        coordinate_score = torch.where(
            camera_has_support[:, None, None, None],
            coordinate_score,
            torch.zeros_like(coordinate_score),
        )
        current_validity = dynamics.chart_availability.float().squeeze(
            -1
        ).clamp(0.0, 1.0)
        residual_validity = current_validity[:, None].expand(-1, intervals, -1)
        temperature = self._temperatures().to(device=action_query.device)
        current_support = current_validity > 0.0
        common_has_support = current_support.any(dim=-1)
        residual_support = residual_validity > 0.0
        interval_has_support = residual_support.any(dim=-1)
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
        selected_common_values: list[Tensor] = []
        selected_residual_values_by_interval: list[Tensor] = []
        for type_index in range(3):
            intent_type_index = self.S_TYPE_INDEX_BY_P2[type_index]
            query = self._bounded_unit(
                self.source_query[type_index](action_query)
            )
            common_source_key = self._bounded_unit(
                self.source_key[type_index](common_source_fields[type_index])
            )
            common_source_score = torch.einsum(
                "btqh,bkh->btqk", query, common_source_key
            )
            residual_source_key = self._bounded_unit(
                self.source_key[type_index](residual_source_fields[type_index])
            )
            residual_source_score = torch.einsum(
                "btqh,bikh->btqik", query, residual_source_key
            )
            common_public_key = self.public_intent_key[type_index](
                intent.common_key
            )[:, None]
            common_typed_key = self.typed_intent_key[type_index](
                typed_common_intent[..., intent_type_index, :]
            )
            common_intent_key = self._bounded_unit(
                common_public_key + common_typed_key
            )
            interval_public_key = self._bounded_unit(
                self.public_intent_key[type_index](
                    intent.interval_residual_key
                )
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
            interval_public_score = torch.einsum(
                "btqh,bih->btqi", intent_query, interval_public_key
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

            common_logit = common_bounded_logit + torch.where(
                current_support,
                current_validity.clamp_min(1e-6).log(),
                torch.zeros_like(current_validity),
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
                common_logit,
                torch.zeros_like(common_logit),
            )
            common_posterior = torch.softmax(common_logit, dim=-1)
            common_posterior = common_posterior * common_has_support[
                :, None, None, None
            ].to(dtype=common_posterior.dtype)
            common_selected = torch.einsum(
                "btqk,bkh->btqh",
                common_posterior.to(dtype=common_source_values[type_index].dtype),
                common_source_values[type_index],
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
            # Object ownership is normalized only inside its already chosen
            # interval.  An all-invalid interval is bookkeeping-zero here and
            # is structurally excluded by the shared outer posterior below.
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
            residual_selected_by_interval = torch.einsum(
                "btqik,bikh->btqih",
                residual_object_posterior.to(
                    dtype=residual_source_values[type_index].dtype
                ),
                residual_source_values[type_index],
            )
            # The shared temporal owner reads the compatibility of the object
            # that this type would actually use inside each interval.  This
            # retains I/K/type evidence until the named temporal consumer while
            # keeping raw coordinate distance out of the time score.
            interval_w_score = (
                residual_object_posterior.float()
                * residual_source_score.float()
            ).sum(dim=-1)
            interval_typed_score = (
                residual_object_posterior.float()
                * residual_intent_score.float()
            ).sum(dim=-1)
            common_source_scores.append(common_source_score)
            residual_source_scores.append(residual_source_score)
            common_intent_scores.append(common_intent_score)
            residual_intent_scores.append(residual_intent_score)
            interval_public_scores.append(interval_public_score)
            interval_typed_scores.append(interval_typed_score)
            interval_w_scores.append(interval_w_score)
            common_bounded_logits.append(common_bounded_logit)
            residual_bounded_logits.append(residual_bounded_logit)
            common_posteriors.append(common_posterior)
            residual_object_posteriors.append(residual_object_posterior)
            selected_common_values.append(common_selected)
            selected_residual_values_by_interval.append(
                residual_selected_by_interval
            )

        # One temporal posterior remains shared by all complementary values,
        # but its evidence is no longer public-S-only.  For every P2 type it
        # combines the matching public S compatibility with the object-
        # conditioned typed-S and supervised-W compatibilities.  All nine
        # terms are already in [-1, 1]; sqrt(9)=3 is the symmetric variance
        # scale and tanh keeps the final score constructionally bounded.  The
        # K posterior is normalized inside each interval, so object count
        # cannot masquerade as temporal evidence.  Coordinate distance still
        # selects geometry's object only and never enters this score directly.
        shared_public_score = torch.stack(interval_public_scores, dim=0).sum(
            dim=0
        ) / 3.0
        shared_typed_score = torch.stack(interval_typed_scores, dim=0).sum(
            dim=0
        ) / 3.0
        shared_w_score = torch.stack(interval_w_scores, dim=0).sum(dim=0) / 3.0
        shared_interval_precontract_score = (
            shared_public_score + shared_typed_score + shared_w_score
        )
        shared_interval_score = torch.tanh(shared_interval_precontract_score)
        shared_interval_bounded_logit = temperature[1] * shared_interval_score
        shared_interval_logit = shared_interval_bounded_logit
        shared_interval_logit = shared_interval_logit.masked_fill(
            ~interval_has_support[:, None, None],
            -torch.inf,
        )
        # One null competes with four interval measures, each of which owns K
        # object candidates.  The fixed -log(K) correction preserves the old
        # neutral 1/(I*K+1) null prior without making K scores or active counts
        # part of temporal selection.  The null value itself is exact zero.
        shared_null_logit = torch.full_like(
            shared_interval_logit[..., :1],
            -math.log(float(max(objects, 1))),
        )
        shared_interval_posterior = torch.softmax(
            torch.cat((shared_interval_logit, shared_null_logit), dim=-1),
            dim=-1,
        )
        shared_interval_mass = shared_interval_posterior[..., :-1]

        selected_residual_values: list[Tensor] = []
        residual_posteriors: list[Tensor] = []
        for residual_selected_by_interval, residual_object_posterior in zip(
            selected_residual_values_by_interval,
            residual_object_posteriors,
        ):
            selected_residual_values.append(
                torch.einsum(
                    "btqi,btqih->btqh",
                    shared_interval_mass.to(
                        dtype=residual_selected_by_interval.dtype
                    ),
                    residual_selected_by_interval,
                )
            )
            # Retain the historical flattened posterior shape for diagnostics
            # and consumers of lossless metrics.  Its factorization is exact:
            # p(i,k)=pi(i)*rho(k|i), with one shared null at the final entry.
            joint_real = (
                shared_interval_mass[..., None] * residual_object_posterior
            ).flatten(-2)
            residual_posteriors.append(
                torch.cat(
                    (joint_real, shared_interval_posterior[..., -1:]),
                    dim=-1,
                )
            )
        common_posterior_by_type = torch.stack(common_posteriors, dim=3)
        residual_posterior_by_type = torch.stack(residual_posteriors, dim=3)
        residual_object_posterior_by_type = torch.stack(
            residual_object_posteriors,
            dim=3,
        )
        residual_value_by_interval_and_type = torch.stack(
            selected_residual_values_by_interval,
            dim=4,
        )
        selected_common_type_value = torch.stack(selected_common_values, dim=3)
        selected_residual_type_value = torch.stack(selected_residual_values, dim=3)
        (
            common_value,
            common_fusion_base,
            common_fusion_contrast,
            common_fusion_residual,
        ) = self._fuse_complementary_values(selected_common_type_value)
        (
            residual_value,
            residual_fusion_base,
            residual_fusion_contrast,
            residual_fusion_residual,
        ) = self._fuse_complementary_values(selected_residual_type_value)
        value = common_value + residual_value
        if not collect_diagnostics:
            return value, {}
        interval_mass_by_type = residual_posterior_by_type[..., :-1].reshape(
            batch, horizon, basis, 3, intervals, objects
        ).sum(dim=-1)
        interval_mass = shared_interval_mass.detach()
        inner_support_weight = interval_has_support[
            :, None, None, None, :
        ].detach().float()
        inner_support_denominator = (
            inner_support_weight.sum()
            * float(horizon * basis * len(self.TYPE_NAMES))
        ).clamp_min(1.0)
        inner_entropy = normalized_entropy(
            residual_object_posterior_by_type,
            dim=-1,
        ).detach()
        inner_max = residual_object_posterior_by_type.detach().amax(dim=-1)
        inner_raw_entropy = -(
            residual_object_posterior_by_type.detach().float()
            * residual_object_posterior_by_type.detach()
            .float()
            .clamp_min(1.0e-8)
            .log()
        ).sum(dim=-1)
        interval_type_rms = residual_value_by_interval_and_type.detach().float()
        interval_type_rms = interval_type_rms.square().mean(dim=-1).sqrt()
        selected_type_rms = selected_residual_type_value.detach().float()
        selected_type_rms = selected_type_rms.square().mean(dim=-1).sqrt()
        weighted_interval_type_rms = (
            shared_interval_mass.detach().float()[..., None]
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
                shared_interval_score,
            ))
        ).mean()
        intent_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in (
                *common_intent_scores,
                *residual_intent_scores,
                shared_interval_score,
            ))
        ).amax()
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
                    shared_interval_bounded_logit,
                ))
            ).amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_common_posterior_entropy": normalized_entropy(
                common_posterior_by_type, dim=-1
            ).detach().mean(),
            "object_p2_common_posterior_max": common_posterior_by_type.detach()
            .amax(dim=-1)
            .mean(),
            "object_p2_residual_posterior_entropy": normalized_entropy(
                residual_posterior_by_type, dim=-1
            ).detach().mean(),
            "object_p2_residual_posterior_max": residual_posterior_by_type.detach()
            .amax(dim=-1)
            .mean(),
            "object_p2_residual_null_mass": residual_posterior_by_type.detach()[
                ..., -1
            ].mean(),
            "object_p2_shared_interval_score_abs": shared_interval_score.detach()
            .abs()
            .mean(),
            "object_p2_shared_interval_score_max_abs": shared_interval_score.detach()
            .abs()
            .amax(),
            "object_p2_shared_interval_public_score_abs": shared_public_score.detach()
            .abs()
            .mean(),
            "object_p2_shared_interval_typed_score_abs": shared_typed_score.detach()
            .abs()
            .mean(),
            "object_p2_shared_interval_w_score_abs": shared_w_score.detach()
            .abs()
            .mean(),
            "object_p2_shared_interval_precontract_score_max_abs": (
                shared_interval_precontract_score.detach().abs().amax()
            ),
            "object_p2_shared_interval_posterior_entropy": normalized_entropy(
                shared_interval_posterior,
                dim=-1,
            ).detach().mean(),
            "object_p2_shared_interval_posterior_max": (
                shared_interval_posterior.detach().amax(dim=-1).mean()
            ),
            "object_p2_shared_interval_null_mass": (
                shared_interval_posterior.detach()[..., -1].mean()
            ),
            "object_p2_shared_interval_effective_count": torch.exp(
                -(
                    shared_interval_posterior.detach().float()
                    * shared_interval_posterior.detach()
                    .float()
                    .clamp_min(1.0e-8)
                    .log()
                ).sum(dim=-1)
            ).mean(),
            "object_p2_shared_interval_horizon_variation": (
                shared_interval_mass.detach()
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
            "object_p2_effect_precontract_rms": value.detach().float().square().mean().sqrt(),
            "object_p2_common_fusion_base_rms": common_fusion_base.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_common_fusion_residual_rms": common_fusion_residual.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_fusion_base_rms": residual_fusion_base.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_fusion_residual_rms": residual_fusion_residual.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_common_fusion_contrast_rms": common_fusion_contrast.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_residual_fusion_contrast_rms": residual_fusion_contrast.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_fusion_scale_abs_mean": self.type_contrast_scale.detach()
            .float()
            .abs()
            .mean(),
            "object_p2_fusion_scale_abs_max": self.type_contrast_scale.detach()
            .float()
            .abs()
            .amax(),
        }
        for type_index, name in enumerate(self.TYPE_NAMES):
            metrics[f"object_p2_{name}_score_max_abs"] = torch.maximum(
                common_source_scores[type_index].detach().abs().amax(),
                residual_source_scores[type_index].detach().abs().amax(),
            )
            metrics[f"object_p2_{name}_shared_interval_public_score_abs"] = (
                interval_public_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_shared_interval_typed_score_abs"] = (
                interval_typed_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_shared_interval_w_score_abs"] = (
                interval_w_scores[type_index].detach().float().abs().mean()
            )
            metrics[f"object_p2_{name}_residual_null_mass"] = residual_posterior_by_type[
                ..., type_index, -1
            ].detach().mean()
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
                common_value_contract_scales[type_index].detach()
                .float()
                .mean()
            )
            metrics[f"object_p2_{name}_residual_value_contract_scale_mean"] = (
                residual_value_contract_scales[type_index].detach()
                .float()
                .mean()
            )
            type_support_weight = inner_support_weight[..., 0, :]
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
        return value, metrics


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
        effect: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectConsequenceState, dict[str, Tensor]]:
        if tuple(factual_base.shape) != tuple(effect.shape):
            raise ValueError("factual base and effect must align")
        interaction = self.interaction(torch.tanh(self.fact(factual_base)) * effect)
        protected = factual_base + effect + interaction
        state = ObjectConsequenceState(
            factual_base=factual_base,
            effect=effect,
            interaction=interaction,
            protected_consequence=protected,
        )
        if not collect_diagnostics:
            return state, {}
        return state, {
            "object_consequence_effect_rms": effect.detach().float().square().mean().sqrt(),
            "object_consequence_interaction_rms": interaction.detach().float().square().mean().sqrt(),
            "object_consequence_ratio": (
                (effect + interaction).detach().float().square().mean().sqrt()
                / factual_base.detach().float().square().mean().sqrt().clamp_min(1e-6)
            ),
        }


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
        p1_policy_residual: Tensor,
        consequence: ObjectConsequenceState,
        intent: PolicyIntentDock,
        action_query: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if (
            tuple(action_query.shape) != expected
            or tuple(p1_factual_detail.shape) != expected
            or tuple(p1_policy_residual.shape) != expected
        ):
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        # The static factual consequence is already the protected base.
        # Optional lanes may only encode source-exclusive zero-centred
        # innovations; neither static detail nor the live policy residual is
        # copied into another protected carrier. Precision is the sole owner
        # of both P1 signals below.
        # Reuse the established precision projection for both P1-owned
        # signals.  Separate calls preserve the old static-detail path exactly
        # when the live residual is zero, add no parameter, and make either
        # source independently intervenable.  Bias-free linearity keeps the
        # joint-zero state algebraically zero.
        static_precision = self.precision_innovation(p1_factual_detail)
        dynamic_precision = self.precision_innovation(p1_policy_residual)
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query))
            * (static_precision + dynamic_precision)
        )
        effect = self.effect_lane(consequence.effect + consequence.interaction)
        temporal_effect = consequence.effect + consequence.interaction
        temporal_source = intent.temporal_control[:, :, None].expand(
            -1, -1, self.basis, -1
        )
        # Temporal is a W-effect relation, not a second factual carrier.
        # Requiring S, W effect and action makes neutral W an exact temporal
        # null while leaving the independently owned state-change lane intact.
        temporal = self.temporal_lane(
            self.temporal_source(temporal_source)
            * torch.tanh(
                self.temporal_consequence(temporal_effect)
            )
            * torch.tanh(self.temporal_action(action_query))
        )
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
        lanes = [precision, effect, temporal, state_change]
        lanes = [smooth_rms_contract(value, 0.35)[0] for value in lanes]
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            precision=lanes[0],
            effect=lanes[1],
            temporal=lanes[2],
            state_change=lanes[3],
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        return bank, {
            "object_p3_precision_input_rms": (
                p1_factual_detail + p1_policy_residual
            )
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_static_input_rms": p1_factual_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_dynamic_input_rms": p1_policy_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p3_precision_rms": lanes[0].detach().float().square().mean().sqrt(),
            "object_p3_effect_rms": lanes[1].detach().float().square().mean().sqrt(),
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
            "object_p3_temporal_rms": lanes[2].detach().float().square().mean().sqrt(),
            "object_p3_state_change_rms": lanes[3].detach().float().square().mean().sqrt(),
        }


__all__ = [
    "ObjectConsequenceState",
    "ObjectFutureEffectReader",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "ZeroPreservingObjectConsequence",
]
