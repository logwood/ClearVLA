"""Object-anchored P2 consequence read and innovation-only P3 compilation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..role_delta_attnres import PolicyRoleDeltaBank, smooth_rms_contract
from .types import (
    FutureObjectDynamics,
    ObjectFactualDock,
    ObjectIntentState,
    normalized_entropy,
)


@dataclass(frozen=True)
class ObjectConsequenceState:
    factual_base: Tensor
    effect: Tensor
    interaction: Tensor
    protected_consequence: Tensor


@dataclass(frozen=True)
class ObjectPolicyPlanDeltaBank:
    """Three optional innovations around one protected consequence.

    P1 facts and the selected P2 effect are already present exactly once in
    ``protected_base``.  Re-projecting them into separately named factual and
    effect lanes would give the bottom decoder duplicate algebraic access to
    the same evidence, so this bank contains only the remaining P3 jobs.
    """

    protected_base: Tensor
    precision: Tensor
    temporal: Tensor
    state_change: Tensor

    @property
    def source_names(self) -> tuple[str, ...]:
        return (
            "p3_precision",
            "p3_temporal",
            "p3_state_change",
        )

    def validate(self) -> None:
        expected = tuple(self.protected_base.shape)
        if len(expected) != 4:
            raise ValueError("object policy plan must be [B,T,Q,H]")
        for name in ("precision", "temporal", "state_change"):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"object policy {name} lost [B,T,Q,H]")

    def as_policy_role_bank(self, *, source_depth: int) -> PolicyRoleDeltaBank:
        self.validate()
        values = torch.stack(
            (
                self.precision,
                self.temporal,
                self.state_change,
            ),
            dim=1,
        )
        return PolicyRoleDeltaBank(
            values=values,
            source_names=self.source_names,
            source_depths=(int(source_depth),) * len(self.source_names),
            protected_detail=self.protected_base,
        )


class ObjectFutureEffectReader(nn.Module):
    """Typed P2 read over the exact global-object basis produced by P1.

    Semantic change and geometric transport have different keys, values and
    posteriors.  P1's K+null posterior supplies the object prior and its chart
    read supplies the source coordinate.  W's full future address supplies
    the destination coordinate.  No pooled tensor is expanded to recreate an
    object axis, and visibility/persistence/uncertainty calibrate selection
    once instead of becoming a third non-zero policy value.
    """

    def __init__(self, *, hidden: int, content_dim: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.semantic_query = nn.Linear(hidden, hidden, bias=False)
        self.semantic_fact = nn.Linear(hidden, hidden, bias=False)
        self.semantic_key = nn.Linear(content_dim, hidden, bias=False)
        self.geometry_query = nn.Linear(hidden, hidden, bias=False)
        self.geometry_fact = nn.Linear(hidden, hidden, bias=False)
        self.geometry_key = nn.Linear(5, hidden, bias=False)
        self.intent_query = nn.Linear(hidden, hidden, bias=False)
        self.intent_key = nn.Linear(hidden, hidden, bias=False)
        self.transport_query = nn.Linear(hidden, 2, bias=False)
        self.semantic_value = nn.Linear(content_dim, hidden, bias=False)
        self.transport_value = nn.Linear(2, hidden, bias=False)
        self.type_query = nn.Linear(hidden, hidden, bias=False)
        self.temperature_logit = nn.Parameter(torch.zeros(3))

    def _temperatures(self) -> Tensor:
        return 0.25 + 3.75 * torch.sigmoid(self.temperature_logit.float())

    @staticmethod
    def _bounded_unit(value: Tensor, *, norm_floor: float = 0.25) -> Tensor:
        """Smooth unit-vector map with a finite Jacobian at zero.

        W's semantic delta is deliberately initialized at exact zero.  A
        conventional ``F.normalize`` would therefore expose a ``1 / eps``
        derivative precisely when the future path is weakest.  The quadratic
        floor keeps the score in [-1, 1] while bounding the local gain by
        ``1 / norm_floor``.
        """

        value = value.float()
        denominator = (
            value.square().sum(dim=-1, keepdim=True)
            + float(norm_floor) ** 2
        ).sqrt()
        return value / denominator

    def forward(
        self,
        action_query: Tensor,
        dynamics: FutureObjectDynamics,
        intent: ObjectIntentState,
        factual_dock: ObjectFactualDock,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        dynamics.validate()
        factual_dock.validate()
        if action_query.ndim != 4:
            raise ValueError("P2 action query must be [B,T,Q,H]")
        batch, horizon, basis, hidden = action_query.shape
        if hidden != self.hidden:
            raise ValueError("P2 action query hidden width is invalid")
        intervals, objects = dynamics.semantic_delta.shape[1:3]
        if tuple(factual_dock.fact_by_object.shape[:4]) != (
            batch,
            horizon,
            basis,
            objects,
        ):
            raise ValueError("P2 factual dock and W object basis do not align")
        if tuple(factual_dock.chart_posterior.shape[-3:]) != tuple(
            dynamics.future_address.shape[-3:]
        ):
            raise ValueError("P2 current and future object charts do not align")

        semantic_query = self._bounded_unit(
            self.semantic_query(action_query)[:, :, :, None]
            + self.semantic_fact(factual_dock.fact_by_object)
        )
        semantic_key = self._bounded_unit(
            self.semantic_key(dynamics.semantic_delta)
        )
        semantic_score = torch.einsum(
            "btqkh,bikh->btqik", semantic_query, semantic_key
        )

        geometry_source = torch.cat(
            (dynamics.transport_mean, dynamics.transport_covariance), dim=-1
        )
        geometry_query = self._bounded_unit(
            self.geometry_query(action_query)[:, :, :, None]
            + self.geometry_fact(factual_dock.fact_by_object)
        )
        geometry_key = self._bounded_unit(self.geometry_key(geometry_source))
        geometry_score = torch.einsum(
            "btqkh,bikh->btqik", geometry_query, geometry_key
        )

        intent_query = self._bounded_unit(self.intent_query(action_query))
        intent_key = self._bounded_unit(
            self.intent_key(intent.interval_queries)
        )
        intent_score = torch.einsum("btqh,bih->btqi", intent_query, intent_key)

        # Full future-address distributions contribute through their spatial
        # moments.  The score compares the P1-observed source coordinate plus
        # W transport with that destination, so legitimate motion is not
        # penalized as if source and future pixels had to be identical.
        cameras, rows, columns = dynamics.future_address.shape[-3:]
        axis_y = torch.linspace(
            -1.0, 1.0, rows, device=action_query.device, dtype=torch.float32
        )
        axis_x = torch.linspace(
            -1.0, 1.0, columns, device=action_query.device, dtype=torch.float32
        )
        coordinate_y, coordinate_x = torch.meshgrid(
            axis_y, axis_x, indexing="ij"
        )
        chart_coordinate = torch.stack((coordinate_x, coordinate_y), dim=-1)
        chart_coordinate = chart_coordinate[None].expand(cameras, -1, -1, -1)
        future_address = dynamics.future_address.float()
        future_address = future_address / future_address.sum(
            dim=(-3, -2, -1), keepdim=True
        ).clamp_min(1e-6)
        future_coordinate = torch.einsum(
            "bikcyx,cyxd->bikd", future_address, chart_coordinate
        )
        transported_source = (
            factual_dock.coordinates.float()[:, :, :, None]
            + dynamics.transport_mean.float()[:, None, None]
        ).clamp(-1.0, 1.0)
        address_score = -0.5 * (
            transported_source
            - future_coordinate[:, None, None]
        ).square().sum(dim=-1)
        address_score = address_score.clamp(-1.0, 0.0)
        requested_transport = torch.tanh(
            self.transport_query(action_query).float()
        )
        transport_score = 1.0 - 0.5 * (
            requested_transport[:, :, :, None, None]
            - dynamics.transport_mean.float()[:, None, None]
        ).square().sum(dim=-1)
        transport_score = transport_score.clamp(-1.0, 1.0)
        coordinate_score = (0.5 * address_score + 0.5 * transport_score).clamp(
            -1.0, 1.0
        )

        temperature = self._temperatures().to(device=action_query.device)
        semantic_logit = (
            temperature[0] * semantic_score
            + temperature[1] * intent_score[..., None]
            + temperature[2] * coordinate_score
        )
        geometry_logit = (
            temperature[0] * geometry_score
            + temperature[1] * intent_score[..., None]
            + temperature[2] * coordinate_score
        )

        # One selector calibration.  Visibility/persistence/uncertainty never
        # become additive policy values and never multiply the selected value.
        calibration = torch.sigmoid(
            1.5 * dynamics.visibility.float().squeeze(-1)
            + 1.5 * dynamics.persistence.float().squeeze(-1)
            - dynamics.uncertainty.float().squeeze(-1)
        )
        physical_validity = dynamics.validity.float().squeeze(-1).clamp(0.0, 1.0)
        object_prior = factual_dock.object_posterior.float() / float(intervals)
        candidate_prior = (
            object_prior[:, :, :, None]
            * physical_validity[:, None, None]
            * calibration[:, None, None]
        )
        null_prior = factual_dock.null_posterior.float().clamp_min(1e-8)

        def typed_posterior(logit: Tensor) -> Tensor:
            flat_logit = logit.flatten(-2)
            flat_prior = candidate_prior.flatten(-2)
            candidate_log_prior = torch.where(
                flat_prior > 0.0,
                flat_prior.clamp_min(1e-30).log(),
                torch.full_like(flat_prior, -1.0e4),
            )
            null_logit = null_prior.log()
            return torch.softmax(
                torch.cat((flat_logit + candidate_log_prior, null_logit), dim=-1),
                dim=-1,
            )

        semantic_posterior = typed_posterior(semantic_logit)
        geometry_posterior = typed_posterior(geometry_logit)
        semantic_source_value = self.semantic_value(
            dynamics.semantic_delta
        ).reshape(batch, intervals * objects, hidden)
        geometry_source_value = self.transport_value(
            dynamics.transport_mean
        ).reshape(batch, intervals * objects, hidden)
        selected_semantic = torch.einsum(
            "btqn,bnh->btqh",
            semantic_posterior[..., :-1].to(dtype=semantic_source_value.dtype),
            semantic_source_value,
        )
        selected_geometry = torch.einsum(
            "btqn,bnh->btqh",
            geometry_posterior[..., :-1].to(dtype=geometry_source_value.dtype),
            geometry_source_value,
        )
        selected_semantic_key = torch.einsum(
            "btqn,bnh->btqh",
            semantic_posterior[..., :-1],
            semantic_key.reshape(batch, intervals * objects, hidden),
        )
        selected_geometry_key = torch.einsum(
            "btqn,bnh->btqh",
            geometry_posterior[..., :-1],
            geometry_key.reshape(batch, intervals * objects, hidden),
        )
        type_query = self._bounded_unit(self.type_query(action_query))
        type_logit = torch.stack(
            (
                (type_query * selected_semantic_key).sum(dim=-1),
                (type_query * selected_geometry_key).sum(dim=-1),
            ),
            dim=-1,
        )
        type_weight = torch.softmax(type_logit, dim=-1)
        value = (
            type_weight[..., :1].to(dtype=selected_semantic.dtype)
            * selected_semantic
            + type_weight[..., 1:].to(dtype=selected_geometry.dtype)
            * selected_geometry
        )
        if not collect_diagnostics:
            # P2 is action-query dependent and therefore runs at every ODE
            # step.  None of the reductions below participates in selection,
            # the consequence value, or training losses; keeping them on the
            # deploy hot path would add FP32 reductions and entropy kernels to
            # every sampling step for values that are immediately discarded.
            return value, {}
        metrics: dict[str, Tensor] = {
            "object_p2_semantic_score_abs": semantic_score.detach().abs().mean(),
            "object_p2_semantic_score_max_abs": semantic_score.detach().abs().amax(),
            "object_p2_geometry_score_abs": geometry_score.detach().abs().mean(),
            "object_p2_geometry_score_max_abs": geometry_score.detach().abs().amax(),
            "object_p2_intent_score_abs": intent_score.detach().abs().mean(),
            "object_p2_intent_score_max_abs": intent_score.detach().abs().amax(),
            "object_p2_coordinate_score_abs": coordinate_score.detach().abs().mean(),
            "object_p2_coordinate_score_max_abs": coordinate_score.detach().abs().amax(),
            "object_p2_address_score_abs": address_score.detach().abs().mean(),
            "object_p2_transport_score_abs": transport_score.detach().abs().mean(),
            "object_p2_semantic_logit_max_abs": semantic_logit.detach().abs().amax(),
            "object_p2_geometry_logit_max_abs": geometry_logit.detach().abs().amax(),
            "object_p2_temperature_content": temperature[0].detach(),
            "object_p2_temperature_intent": temperature[1].detach(),
            "object_p2_temperature_coordinate": temperature[2].detach(),
            "object_p2_semantic_posterior_entropy": normalized_entropy(
                semantic_posterior, dim=-1
            ).detach().mean(),
            "object_p2_geometry_posterior_entropy": normalized_entropy(
                geometry_posterior, dim=-1
            ).detach().mean(),
            "object_p2_semantic_posterior_max": semantic_posterior.detach().amax(dim=-1).mean(),
            "object_p2_geometry_posterior_max": geometry_posterior.detach().amax(dim=-1).mean(),
            "object_p2_semantic_null_mass": semantic_posterior.detach()[..., -1].mean(),
            "object_p2_geometry_null_mass": geometry_posterior.detach()[..., -1].mean(),
            "object_p2_selector_calibration": calibration.detach().mean(),
            "object_p2_effect_precontract_rms": value.detach().float().square().mean().sqrt(),
            "object_p2_semantic_value_mass": type_weight.detach()[..., 0].mean(),
            "object_p2_geometry_value_mass": type_weight.detach()[..., 1].mean(),
        }
        semantic_interval_mass = semantic_posterior[..., :-1].reshape(
            batch, horizon, basis, intervals, objects
        ).sum(dim=-1)
        geometry_interval_mass = geometry_posterior[..., :-1].reshape(
            batch, horizon, basis, intervals, objects
        ).sum(dim=-1)
        for index in range(intervals):
            metrics[f"object_p2_semantic_interval_{index}_mass"] = (
                semantic_interval_mass[..., index].detach().float().mean()
            )
            metrics[f"object_p2_geometry_interval_{index}_mass"] = (
                geometry_interval_mass[..., index].detach().float().mean()
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
        interaction = self.interaction(
            torch.tanh(self.fact(factual_base)) * effect
        )
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
    """P3 compiles three innovations around one protected consequence."""

    def __init__(self, *, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.precision_action = nn.Linear(hidden, hidden, bias=False)
        self.precision_fact = nn.Linear(hidden, hidden, bias=False)
        self.precision_consequence = nn.Linear(hidden, hidden, bias=False)
        self.precision_lane = nn.Linear(hidden, hidden, bias=False)
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
        precision_condition = (
            self.precision_fact(p1_fact)
            + self.precision_consequence(consequence.protected_consequence)
        ) / math.sqrt(2.0)
        precision = self.precision_lane(
            torch.tanh(self.precision_action(action_query))
            * precision_condition
        )
        temporal_source = intent.temporal_queries[:, :, None].expand(-1, -1, self.basis, -1)
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
        # The third optional lane is a weak, time/basis-specific modulation.  Its
        # multiplicative form preserves exact zero when no observable change
        # exists; temporal/action context can shape but cannot synthesize it.
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
        lanes = [precision, temporal, state_change]
        lanes = [smooth_rms_contract(value, 0.35)[0] for value in lanes]
        bank = ObjectPolicyPlanDeltaBank(
            protected_base=consequence.protected_consequence,
            precision=lanes[0],
            temporal=lanes[1],
            state_change=lanes[2],
        )
        bank.validate()
        if not collect_diagnostics:
            return bank, {}
        metrics = {
            "object_p3_precision_rms": lanes[0].detach().float().square().mean().sqrt(),
            "object_p3_temporal_rms": lanes[1].detach().float().square().mean().sqrt(),
            "object_p3_state_change_rms": lanes[2].detach().float().square().mean().sqrt(),
        }
        return bank, metrics
