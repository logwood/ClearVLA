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
    """Per-type bounded interval-by-object P2 reads with explicit nulls."""

    TYPE_NAMES = ("semantic", "geometry", "status")

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
            dynamics.transport_common,
            status_common,
        )
        residual_source_fields = (
            dynamics.semantic_interval_residual,
            dynamics.transport_interval_residual,
            status_residual,
        )
        common_source_values = (
            self.semantic_value(dynamics.semantic_common),
            self.transport_value(dynamics.transport_common),
            self.status_value(status_common),
        )
        residual_source_values = (
            self.semantic_value(dynamics.semantic_interval_residual),
            self.transport_value(dynamics.transport_interval_residual),
            self.status_value(status_residual),
        )
        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        common_coordinate = (
            dynamics.object_coordinates.float() + dynamics.transport_common
        ).clamp(-1.0, 1.0)
        future_coordinate = (
            dynamics.object_coordinates[:, None].float() + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        common_coordinate_distance = (
            coordinate_query[:, :, :, None] - common_coordinate[:, None, None]
        ).square().sum(dim=-1)
        coordinate_distance = (
            coordinate_query[:, :, :, None, None]
            - future_coordinate[:, None, None]
        ).square().sum(dim=-1)
        # Exact coordinate agreement is positive evidence; the old [-1,0]
        # term could only punish and therefore could not establish a geometry
        # owner when semantic scores were diffuse.
        common_coordinate_score = (
            1.0 - 0.5 * common_coordinate_distance
        ).clamp(-1.0, 1.0)
        coordinate_score = (1.0 - 0.5 * coordinate_distance).clamp(-1.0, 1.0)
        current_validity = dynamics.current_selector_validity.float().squeeze(
            -1
        ).clamp(0.0, 1.0)
        residual_validity = current_validity[:, None].expand(-1, intervals, -1)
        temperature = self._temperatures().to(device=action_query.device)
        common_source_scores: list[Tensor] = []
        residual_source_scores: list[Tensor] = []
        common_intent_scores: list[Tensor] = []
        residual_intent_scores: list[Tensor] = []
        common_bounded_logits: list[Tensor] = []
        residual_bounded_logits: list[Tensor] = []
        common_posteriors: list[Tensor] = []
        residual_posteriors: list[Tensor] = []
        selected_common_values: list[Tensor] = []
        selected_residual_values: list[Tensor] = []
        for type_index in range(3):
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
                typed_common_intent[..., type_index, :]
            )
            common_intent_key = self._bounded_unit(
                common_public_key + common_typed_key
            )
            residual_public_key = self.public_intent_key[type_index](
                intent.interval_residual_key
            )[:, :, None]
            residual_typed_key = self.typed_intent_key[type_index](
                typed_residual_intent[..., type_index, :]
            )
            residual_intent_key = self._bounded_unit(
                residual_public_key + residual_typed_key
            )
            intent_query = self._bounded_unit(
                self.intent_query[type_index](action_query)
            )
            common_intent_score = torch.einsum(
                "btqh,bkh->btqk", intent_query, common_intent_key
            )
            residual_intent_score = torch.einsum(
                "btqh,bikh->btqik", intent_query, residual_intent_key
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

            current_support = current_validity > 0.0
            common_has_support = current_support.any(dim=-1)
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

            residual_support = residual_validity > 0.0
            residual_logit = residual_bounded_logit + torch.where(
                residual_support,
                residual_validity.clamp_min(1e-6).log(),
                torch.zeros_like(residual_validity),
            )[:, None, None]
            residual_logit = residual_logit.masked_fill(
                ~residual_support[:, None, None],
                -torch.inf,
            )
            flat_logit = residual_logit.flatten(-2)
            residual_posterior = torch.softmax(
                torch.cat((flat_logit, torch.zeros_like(flat_logit[..., :1])), dim=-1),
                dim=-1,
            )
            flat_value = residual_source_values[type_index].reshape(
                batch, intervals * objects, hidden
            )
            residual_selected = torch.einsum(
                "btqn,bnh->btqh",
                residual_posterior[..., :-1].to(dtype=flat_value.dtype),
                flat_value,
            )
            common_source_scores.append(common_source_score)
            residual_source_scores.append(residual_source_score)
            common_intent_scores.append(common_intent_score)
            residual_intent_scores.append(residual_intent_score)
            common_bounded_logits.append(common_bounded_logit)
            residual_bounded_logits.append(residual_bounded_logit)
            common_posteriors.append(common_posterior)
            residual_posteriors.append(residual_posterior)
            selected_common_values.append(common_selected)
            selected_residual_values.append(residual_selected)
        common_posterior_by_type = torch.stack(common_posteriors, dim=3)
        residual_posterior_by_type = torch.stack(residual_posteriors, dim=3)
        selected_common_type_value = torch.stack(selected_common_values, dim=3)
        selected_residual_type_value = torch.stack(selected_residual_values, dim=3)
        common_value, common_fusion_base, common_fusion_contrast, common_fusion_residual = (
            self._fuse_complementary_values(selected_common_type_value)
        )
        residual_value, residual_fusion_base, residual_fusion_contrast, residual_fusion_residual = (
            self._fuse_complementary_values(selected_residual_type_value)
        )
        value = common_value + residual_value
        if not collect_diagnostics:
            return value, {}
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
            ))
        ).mean()
        intent_score_max = torch.stack(
            tuple(score.detach().abs().amax() for score in (
                *common_intent_scores,
                *residual_intent_scores,
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
            "object_p2_combined_logit_max_abs": torch.stack(
                tuple(logit.detach().abs().amax() for logit in (
                    *common_bounded_logits,
                    *residual_bounded_logits,
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
        interval_mass_by_type = residual_posterior_by_type[..., :-1].reshape(
            batch, horizon, basis, 3, intervals, objects
        ).sum(dim=-1)
        interval_mass = interval_mass_by_type.detach().mean(dim=3)
        for index in range(intervals):
            metrics[f"object_p2_residual_interval_{index}_mass"] = (
                interval_mass[..., index].detach().float().mean()
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
        p1_fact: Tensor,
        p1_precision_innovation: Tensor,
        consequence: ObjectConsequenceState,
        intent: PolicyIntentDock,
        action_query: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if (
            tuple(action_query.shape) != expected
            or tuple(p1_fact.shape) != expected
            or tuple(p1_precision_innovation.shape) != expected
        ):
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        # The complete factual consequence is already the protected base.
        # Optional lanes may only encode source-exclusive zero-centred
        # innovations; duplicating that base gives the bottom selector several
        # interchangeable ways to reconstruct the same fact.
        # This is the cached V120 factual-reader write,
        # ``updated_trajectory - clean_action_basis``.  It retains the
        # 24-query/N=49/3x3 evidence without copying the completed P1
        # self-write or W consequence back into an optional lane.  Action can
        # select this evidence but cannot synthesize it.
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query))
            * self.precision_innovation(p1_precision_innovation)
        )
        effect = self.effect_lane(consequence.effect + consequence.interaction)
        temporal_source = intent.temporal_control[:, :, None].expand(
            -1, -1, self.basis, -1
        )
        # Temporal is an optional relation, not a second public action
        # adapter.  Requiring all three observable operands closes the direct
        # S -> bottom lane while preserving ordinary autograd and exact-zero
        # semantics at every missing boundary.
        temporal = self.temporal_lane(
            self.temporal_source(temporal_source)
            * torch.tanh(
                self.temporal_consequence(consequence.protected_consequence)
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
            "object_p3_precision_input_rms": p1_precision_innovation.detach()
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
            "object_p3_temporal_consequence_rms": consequence.protected_consequence
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
