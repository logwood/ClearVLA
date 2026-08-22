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
        typed_intent = intent.typed_interval_object_value
        if tuple(typed_intent.shape[:4]) != (batch, intervals, objects, 3):
            raise ValueError("P2 typed intent lost interval/object/type identity")
        status_source = torch.cat(
            (dynamics.visibility, dynamics.persistence), dim=-1
        )
        source_fields = (
            dynamics.semantic_delta,
            dynamics.transport_mean,
            status_source,
        )
        source_values = (
            self.semantic_value(dynamics.semantic_delta),
            self.transport_value(dynamics.transport_mean),
            self.status_value(status_source),
        )
        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        future_coordinate = (
            dynamics.object_coordinates[:, None].float()
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_distance = (
            coordinate_query[:, :, :, None, None]
            - future_coordinate[:, None, None]
        ).square().sum(dim=-1)
        # Exact coordinate agreement is positive evidence; the old [-1,0]
        # term could only punish and therefore could not establish a geometry
        # owner when semantic scores were diffuse.
        coordinate_score = (1.0 - 0.5 * coordinate_distance).clamp(-1.0, 1.0)
        future_validity = dynamics.future_selector_validity.float().squeeze(
            -1
        ).clamp(0.0, 1.0)
        current_validity = dynamics.current_selector_validity.float().squeeze(
            -1
        ).clamp(0.0, 1.0)[:, None].expand(-1, intervals, -1)
        temperature = self._temperatures().to(device=action_query.device)
        source_scores: list[Tensor] = []
        intent_scores: list[Tensor] = []
        bounded_logits: list[Tensor] = []
        posteriors: list[Tensor] = []
        selected_values: list[Tensor] = []
        for type_index in range(3):
            query = self._bounded_unit(
                self.source_query[type_index](action_query)
            )
            source_key = self._bounded_unit(
                self.source_key[type_index](source_fields[type_index])
            )
            source_score = torch.einsum(
                "btqh,bikh->btqik", query, source_key
            )
            public_key = self.public_intent_key[type_index](
                intent.interval_key
            )[:, :, None]
            typed_key = self.typed_intent_key[type_index](
                typed_intent[..., type_index, :]
            )
            intent_key = self._bounded_unit(public_key + typed_key)
            intent_query = self._bounded_unit(
                self.intent_query[type_index](action_query)
            )
            intent_score = torch.einsum(
                "btqh,bikh->btqik", intent_query, intent_key
            )
            bounded_logit = (
                temperature[0] * source_score
                + temperature[1] * intent_score
            )
            if type_index == 1:
                bounded_logit = bounded_logit + temperature[2] * coordinate_score
            validity = current_validity if type_index == 2 else future_validity
            logit = bounded_logit + validity.clamp_min(1e-6).log()[:, None, None]
            flat_logit = logit.flatten(-2)
            posterior = torch.softmax(
                torch.cat((flat_logit, torch.zeros_like(flat_logit[..., :1])), dim=-1),
                dim=-1,
            )
            flat_value = source_values[type_index].reshape(
                batch, intervals * objects, hidden
            )
            selected = torch.einsum(
                "btqn,bnh->btqh",
                posterior[..., :-1].to(dtype=flat_value.dtype),
                flat_value,
            )
            source_scores.append(source_score)
            intent_scores.append(intent_score)
            bounded_logits.append(bounded_logit)
            posteriors.append(posterior)
            selected_values.append(selected)
        posterior_by_type = torch.stack(posteriors, dim=3)
        selected_type_value = torch.stack(selected_values, dim=3)
        value, fusion_base, fusion_contrast, fusion_residual = (
            self._fuse_complementary_values(selected_type_value)
        )
        if not collect_diagnostics:
            return value, {}
        metrics: dict[str, Tensor] = {
            "object_p2_content_score_abs": torch.stack(source_scores).detach().abs().mean(),
            "object_p2_content_score_max_abs": torch.stack(source_scores).detach().abs().amax(),
            "object_p2_intent_score_abs": torch.stack(intent_scores).detach().abs().mean(),
            "object_p2_intent_score_max_abs": torch.stack(intent_scores).detach().abs().amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_combined_logit_max_abs": torch.stack(bounded_logits).detach().abs().amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_posterior_entropy": normalized_entropy(
                posterior_by_type, dim=-1
            ).detach().mean(),
            "object_p2_posterior_max": posterior_by_type.detach().amax(dim=-1).mean(),
            "object_p2_null_mass": posterior_by_type.detach()[..., -1].mean(),
            "object_p2_effect_precontract_rms": value.detach().float().square().mean().sqrt(),
            "object_p2_fusion_base_rms": fusion_base.detach().square().mean().sqrt(),
            "object_p2_fusion_contrast_rms": fusion_contrast.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_fusion_residual_rms": fusion_residual.detach()
            .square()
            .mean()
            .sqrt(),
            "object_p2_fusion_residual_to_base": fusion_residual.detach()
            .square()
            .mean()
            .sqrt()
            / fusion_base.detach().square().mean().sqrt().clamp_min(1.0e-8),
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
            metrics[f"object_p2_{name}_score_max_abs"] = source_scores[
                type_index
            ].detach().abs().amax()
            metrics[f"object_p2_{name}_null_mass"] = posterior_by_type[
                ..., type_index, -1
            ].detach().mean()
            selected_value = selected_type_value[..., type_index, :].detach().float()
            metrics[f"object_p2_{name}_selected_value_rms"] = (
                selected_value.square().mean().sqrt()
            )
            metrics[f"object_p2_{name}_anchor_contribution_rms"] = (
                (
                    selected_value / math.sqrt(float(len(self.TYPE_NAMES)))
                ).square().mean().sqrt()
            )
        interval_mass_by_type = posterior_by_type[..., :-1].reshape(
            batch, horizon, basis, 3, intervals, objects
        ).sum(dim=-1)
        interval_mass = interval_mass_by_type.detach().mean(dim=3)
        for index in range(intervals):
            metrics[f"object_p2_interval_{index}_mass"] = (
                interval_mass[..., index].detach().float().mean()
            )
            for type_index, name in enumerate(self.TYPE_NAMES):
                metrics[f"object_p2_{name}_interval_{index}_mass"] = (
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
