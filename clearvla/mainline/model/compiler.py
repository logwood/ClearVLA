"""Recovered V120 P2 consequence read and five-lane P3 compiler."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import PolicyRoleDeltaBank, smooth_rms_contract
from .types import FutureObjectDynamics, ObjectIntentState, normalized_entropy


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    effect: Tensor
    interaction: Tensor
    protected_consequence: Tensor


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """V120 five typed lanes around one protected consequence."""

    protected_base: Tensor
    factual: Tensor
    precision: Tensor
    effect: Tensor
    temporal: Tensor
    state_change: Tensor

    @property
    def source_names(self) -> tuple[str, ...]:
        return (
            "p3_factual",
            "p3_precision",
            "p3_effect",
            "p3_temporal",
            "p3_state_change",
        )

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        for name in ("factual", "precision", "effect", "temporal", "state_change"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        return PolicyRoleDeltaBank(
            values=torch.stack(
                (
                    self.factual,
                    self.precision,
                    self.effect,
                    self.temporal,
                    self.state_change,
                ),
                dim=1,
            ),
            source_names=self.source_names,
            source_depths=(int(source_depth),) * 5,
            protected_detail=self.protected_base,
        )


class ObjectFutureEffectReader(nn.Module):
    """V120 bounded interval-by-object P2 read with a null value."""

    def __init__(self, *, hidden: int, content_dim: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.query_key = nn.Linear(hidden, hidden, bias=False)
        self.effect_key = nn.Linear(content_dim, hidden, bias=False)
        self.intent_query = nn.Linear(hidden, hidden, bias=False)
        self.intent_key = nn.Linear(hidden, hidden, bias=False)
        self.coordinate_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.status_value = nn.Linear(2, hidden, bias=False)
        self.type_query = nn.Linear(hidden, 3, bias=False)
        self.temperature_logit = nn.Parameter(torch.zeros(3))

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        value = value.float()
        return value / (
            value.square().sum(dim=-1, keepdim=True) + float(norm_floor) ** 2
        ).sqrt()

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: ObjectIntentState,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        dynamics.validate()
        if action_query.ndim != 4:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        batch, horizon, basis, hidden = action_query.shape
        if hidden != self.hidden:
            raise ValueError("P2 action query hidden width is invalid")
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        query = self._bounded_unit(self.query_key(action_query))
        effect_key = self._bounded_unit(self.effect_key(dynamics.semantic_delta))
        content_score = torch.einsum("btqh,bikh->btqik", query, effect_key)
        intent_query = self._bounded_unit(self.intent_query(action_query))
        intent_key = self._bounded_unit(self.intent_key(intent.interval_queries))
        intent_score = torch.einsum("btqh,bih->btqi", intent_query, intent_key)
        coordinate_query = torch.tanh(self.coordinate_query(action_query).float())
        future_coordinate = (
            dynamics.object_coordinates[:, None].float()
            + dynamics.transport_mean.float()
        ).clamp(-1.0, 1.0)
        coordinate_score = -0.25 * (
            coordinate_query[:, :, :, None, None]
            - future_coordinate[:, None, None]
        ).square().sum(dim=-1)
        coordinate_score = coordinate_score.clamp(-1.0, 0.0)
        validity = (
            dynamics.future_selector_validity.float().squeeze(-1).clamp(0.0, 1.0)
        )
        temperature = self._temperatures().to(device=action_query.device)
        bounded_logit = (
            temperature[0] * content_score
            + temperature[1] * intent_score[..., None]
            + temperature[2] * coordinate_score
        )
        logit = bounded_logit + validity.clamp_min(1e-6).log()[:, None, None]
        flat_logit = logit.flatten(-2)
        posterior = torch.softmax(
            torch.cat((flat_logit, torch.zeros_like(flat_logit[..., :1])), dim=-1),
            dim=-1,
        )
        typed_source_value = torch.stack(
            (
                self.semantic_value(dynamics.semantic_delta),
                self.transport_value(dynamics.transport_mean),
                self.status_value(
                    torch.cat((dynamics.visibility, dynamics.persistence), dim=-1)
                ),
            ),
            dim=-2,
        ).reshape(batch, intervals * objects, 3, hidden)
        selected_type_value = torch.einsum(
            "btqn,bnsh->btqsh",
            posterior[..., :-1].to(dtype=typed_source_value.dtype),
            typed_source_value,
        )
        type_weight = torch.softmax(self.type_query(action_query).float(), dim=-1)
        value = torch.einsum(
            "btqs,btqsh->btqh",
            type_weight.to(dtype=selected_type_value.dtype),
            selected_type_value,
        )
        if not collect_diagnostics:
            return value, {}
        metrics: dict[str, Tensor] = {
            "object_p2_content_score_abs": content_score.detach().abs().mean(),
            "object_p2_content_score_max_abs": content_score.detach().abs().amax(),
            "object_p2_intent_score_abs": intent_score.detach().abs().mean(),
            "object_p2_intent_score_max_abs": intent_score.detach().abs().amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_combined_logit_max_abs": bounded_logit.detach().abs().amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_posterior_entropy": normalized_entropy(
                posterior, dim=-1
            ).detach().mean(),
            "object_p2_posterior_max": posterior.detach().amax(dim=-1).mean(),
            "object_p2_null_mass": posterior.detach()[..., -1].mean(),
            "object_p2_effect_precontract_rms": value.detach().float().square().mean().sqrt(),
            "object_p2_semantic_value_mass": type_weight.detach()[..., 0].mean(),
            "object_p2_geometry_value_mass": type_weight.detach()[..., 1].mean(),
            "object_p2_status_value_mass": type_weight.detach()[..., 2].mean(),
        }
        interval_mass = posterior[..., :-1].reshape(
            batch, horizon, basis, intervals, objects
        ).sum(dim=-1)
        for index in range(intervals):
            metrics[f"object_p2_interval_{index}_mass"] = (
                interval_mass[..., index].detach().float().mean()
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
    """V120 factual/precision/effect/temporal/state-change P3."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.factual_lane = nn.Linear(hidden, hidden, bias=False)
        self.precision_action = nn.Linear(hidden, hidden, bias=False)
        self.precision_fact = nn.Linear(hidden, hidden, bias=False)
        self.precision_consequence = nn.Linear(hidden, hidden, bias=False)
        self.precision_lane = nn.Linear(hidden, hidden, bias=False)
        self.effect_lane = nn.Linear(hidden, hidden, bias=False)
        self.temporal_action = nn.Linear(hidden, hidden, bias=False)
        self.temporal_consequence = nn.Linear(hidden, hidden, bias=False)
        self.temporal_lane = nn.Linear(hidden, hidden, bias=False)
        self.state_change_action = nn.Linear(hidden, hidden, bias=False)
        self.state_change_temporal = nn.Linear(hidden, hidden, bias=False)
        self.state_change_lane = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        *,
        p1_fact: Tensor,
        consequence: ObjectConsequenceState,
        intent: ObjectIntentState,
        action_query: Tensor,
        collect_diagnostics: bool = True,
    ) -> tuple[ObjectPolicyPlanDeltaBank, dict[str, Tensor]]:
        expected = (int(action_query.shape[0]), self.horizon, self.basis, self.hidden)
        if tuple(action_query.shape) != expected or tuple(p1_fact.shape) != expected:
            raise ValueError("P3 inputs must align as [B,T,Q,H]")
        factual = self.factual_lane(consequence.factual_base)
        precision_condition = (
            self.precision_fact(p1_fact)
            + self.precision_consequence(consequence.protected_consequence)
        ) / math.sqrt(2.0)
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query)) * precision_condition
        )
        effect = self.effect_lane(consequence.effect + consequence.interaction)
        temporal_source = intent.temporal_queries[:, :, None].expand(
            -1, -1, self.basis, -1
        )
        temporal_condition = (
            temporal_source
            + self.temporal_consequence(consequence.protected_consequence)
        ) / math.sqrt(2.0)
        temporal = self.temporal_lane(
            temporal_condition * torch.tanh(self.temporal_action(action_query))
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
        lanes = [factual, precision, effect, temporal, state_change]
        lanes = [smooth_rms_contract(value, 0.35)[0] for value in lanes]
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            factual=lanes[0],
            precision=lanes[1],
            effect=lanes[2],
            temporal=lanes[3],
            state_change=lanes[4],
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        return bank, {
            "object_p3_factual_rms": lanes[0].detach().float().square().mean().sqrt(),
            "object_p3_precision_rms": lanes[1].detach().float().square().mean().sqrt(),
            "object_p3_effect_rms": lanes[2].detach().float().square().mean().sqrt(),
            "object_p3_temporal_rms": lanes[3].detach().float().square().mean().sqrt(),
            "object_p3_state_change_rms": lanes[4].detach().float().square().mean().sqrt(),
        }


__all__ = [
    "ObjectConsequenceState",
    "ObjectFutureEffectReader",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "ZeroPreservingObjectConsequence",
]
