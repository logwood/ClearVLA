"""Explicit G/S/W/P composition for the mainline top.

This module replaces the old eight-thousand-line planner dispatch for the
active capability.  It has three calls with non-overlapping authority:

``build_online_context``
    Current observation/history/language only.  Produces G, S and W once.
``build_training_targets``
    Training-only no-grad Teacher-G plus recognizer/loss targets.  It cannot
    mutate or replace the online context.
``compile_policy``
    ODE-step-dependent P2/P3 read from one already materialized P1 dock.

P1 is built exactly once by the independent factual reader and enters this
composer only through its typed dock; P2/P3 cannot reopen a visual bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ZeroPreservingObjectConsequence,
)
from .dynamics import ObjectFutureDynamicsCompiler
from .grounding import DenseObjectGrounder
from .intent import (
    CoarseActionIntent,
    FuturePlanRecognizer,
    StatelessObjectIntentOrganizer,
)
from .role_hosts import TypedGroundingRoleHost
from .teacher import ObjectFutureTeacher
from .types import (
    CoarseActionIntentState,
    FutureObjectDynamics,
    LocalFactSet,
    ObjectFactSet,
    ObjectFactualDock,
    ObjectIntentState,
    ObjectTopTrainingTargets,
)


@dataclass(frozen=True)
class OnlineTopContext:
    """Static online G/S/W state cached once for five-step deployment."""

    facts: ObjectFactSet
    intent: ObjectIntentState
    coarse_action: CoarseActionIntentState
    predicted_dynamics: FutureObjectDynamics

    def deployment_cache(self) -> "DeploymentTopCache":
        return DeploymentTopCache(
            intent=self.intent,
            predicted_dynamics=self.predicted_dynamics,
        )

    def validate(self, *, hidden: int, horizon: int) -> None:
        self.facts.validate()
        self.intent.validate(horizon=horizon, hidden=hidden)
        self.predicted_dynamics.validate()
        expected = (self.facts.batch, 4, hidden)
        if tuple(self.coarse_action.innovations.shape) != expected:
            raise ValueError("coarse action innovations lost the interval axis")
        if self.coarse_action.target is not None:
            raise ValueError("online context cannot carry a future action target")
        if self.coarse_action.loss.ndim != 0:
            raise ValueError("coarse action online placeholder loss must be scalar")


@dataclass(frozen=True)
class DeploymentTopCache:
    """Only S/W values consumed by dynamic P2/P3 deployment."""

    intent: ObjectIntentState
    predicted_dynamics: FutureObjectDynamics

    def validate(self, *, hidden: int, horizon: int) -> None:
        self.intent.validate(horizon=horizon, hidden=hidden)
        self.predicted_dynamics.validate()


@dataclass(frozen=True)
class CompiledPolicyState:
    """Dynamic P2/P3 state consumed by the bottom action model."""

    effect: Tensor
    consequence: ObjectConsequenceState
    plan: ObjectPolicyPlanDeltaBank

    def validate(self) -> None:
        if tuple(self.effect.shape) != tuple(self.consequence.factual_base.shape):
            raise ValueError("P2 effect and consequence schemas do not align")
        self.plan.validate()
        if tuple(self.plan.protected_base.shape) != tuple(self.effect.shape):
            raise ValueError("P3 plan and P2 effect schemas do not align")


class ObjectIntentDynamicsTop(nn.Module):
    """Single owner of the active G3/S/W2/P3 top graph."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        goal_dim: int,
        state_dim: int,
        action_dim: int,
        horizon: int,
        basis: int,
        heads: int,
        objects: int = 4,
        grounder_iterations: int = 3,
        teacher_key_dim: int = 64,
        flow_reference_frames: int = 4,
        role_host_depth: int = 3,
        role_host_expansion: float = 4.0,
        role_host_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.basis = int(basis)
        if int(objects) != 4:
            raise ValueError("the active object top requires K=4")
        self.grounding_host = TypedGroundingRoleHost(
            hidden=hidden,
            content_dim=content_dim,
            route_dim=route_dim,
            state_dim=state_dim,
            heads=heads,
            depth=role_host_depth,
            expansion=role_host_expansion,
            dropout=role_host_dropout,
        )
        self.grounder = DenseObjectGrounder(
            hidden=hidden,
            content_dim=content_dim,
            route_dim=route_dim,
            objects=objects,
            iterations=grounder_iterations,
        )
        self.intent = StatelessObjectIntentOrganizer(
            hidden=hidden,
            goal_dim=goal_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            content_dim=content_dim,
            route_dim=route_dim,
            horizon=horizon,
            heads=heads,
        )
        self.coarse_action = CoarseActionIntent(
            hidden=hidden,
            action_dim=action_dim,
            heads=heads,
        )
        self.dynamics = ObjectFutureDynamicsCompiler(
            hidden=hidden,
            content_dim=content_dim,
            heads=heads,
        )
        self.teacher = ObjectFutureTeacher(
            content_dim=content_dim,
            key_dim=teacher_key_dim,
            flow_reference_frames=flow_reference_frames,
        )
        self.recognizer = FuturePlanRecognizer(
            hidden=hidden,
            action_dim=action_dim,
            state_dim=state_dim,
            content_dim=content_dim,
            heads=heads,
        )
        self.effect_reader = ObjectFutureEffectReader(
            hidden=hidden,
            content_dim=content_dim,
        )
        self.consequence = ZeroPreservingObjectConsequence(hidden)
        self.plan_compiler = ObjectPolicyPlanCompiler(
            hidden=hidden,
            horizon=horizon,
            basis=basis,
        )

    def build_online_context(
        self,
        *,
        local_facts: LocalFactSet,
        goal_tokens: Tensor,
        goal_mask: Tensor,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[OnlineTopContext, dict[str, Tensor]]:
        """Build current G/S/W without a future-capable argument."""

        hosted_local_facts, host_metrics = self.grounding_host(
            local_facts,
            state=state,
            collect_diagnostics=collect_diagnostics,
        )
        facts, ground_metrics = self.grounder(
            hosted_local_facts,
            collect_diagnostics=collect_diagnostics,
        )
        intent, intent_metrics = self.intent(
            goal_tokens=goal_tokens,
            goal_mask=goal_mask,
            state_history=state_history,
            state=state,
            executed_history=executed_history,
            facts=facts,
            collect_diagnostics=collect_diagnostics,
        )
        coarse = self.coarse_action(intent)
        _, w1_state, w1_metrics = self.dynamics.forward_w1(
            facts=facts,
            intent=intent,
            action=coarse,
            collect_diagnostics=collect_diagnostics,
        )
        predicted, w2_metrics = self.dynamics.forward_w2(
            facts=facts,
            intent=intent,
            action=coarse,
            w1_state=w1_state,
            collect_diagnostics=collect_diagnostics,
        )
        context = OnlineTopContext(
            facts=facts,
            intent=intent,
            coarse_action=coarse,
            predicted_dynamics=predicted,
        )
        context.validate(hidden=self.hidden, horizon=self.horizon)
        if not collect_diagnostics:
            return context, {}
        metrics = {
            **host_metrics,
            **ground_metrics,
            **intent_metrics,
            **w1_metrics,
            **w2_metrics,
            "object_coarse_action_rms": coarse.action_prediction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        return context, metrics

    @staticmethod
    def _object_match(
        online: Tensor,
        target: Tensor,
        validity: Tensor,
    ) -> Tensor:
        row = F.smooth_l1_loss(
            online.float(),
            target.detach().float(),
            reduction="none",
        ).mean(dim=-1, keepdim=True)
        weight = validity.detach().float()
        return (row * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _interval_endpoint_summary(sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or int(sequence.shape[1]) < 48:
            raise ValueError("future sequence must be [B,T>=48,D]")
        rows: list[Tensor] = []
        for lower, upper in ((4, 8), (8, 16), (16, 32), (32, 48)):
            segment = sequence[:, lower - 1 : upper]
            start = segment[:, 0]
            end = segment[:, -1]
            rows.append(torch.cat((start, end, end - start), dim=-1))
        return torch.stack(rows, dim=1)

    def build_training_targets(
        self,
        context: OnlineTopContext,
        *,
        future_supports: Tensor,
        future_offsets: Tensor,
        future_action: Tensor,
        future_state: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectTopTrainingTargets, dict[str, Tensor]]:
        """Build the sole training plane without changing online values."""

        context.validate(hidden=self.hidden, horizon=self.horizon)
        teacher, teacher_metrics = self.teacher(
            facts=context.facts,
            future_supports=future_supports,
            future_offsets=future_offsets,
            collect_diagnostics=collect_diagnostics,
        )
        recognition = self.recognizer(
            future_action=future_action,
            future_state=future_state,
            teacher=teacher,
        )
        action_match = F.smooth_l1_loss(
            context.intent.interval_action_innovations.float(),
            recognition.action_targets.detach().float(),
        )
        state_match = F.smooth_l1_loss(
            context.intent.interval_state_innovations.float(),
            recognition.state_targets.detach().float(),
        )
        object_key_match = self._object_match(
            context.intent.interval_object_keys,
            recognition.object_key_targets,
            recognition.object_validity,
        )
        object_value_match = self._object_match(
            context.intent.interval_object_values,
            recognition.object_value_targets,
            recognition.object_validity,
        )
        online_intent_loss = 0.25 * (
            action_match + state_match + object_key_match + object_value_match
        )
        coarse_target = self._interval_endpoint_summary(future_action).detach()
        coarse_loss = F.mse_loss(
            context.coarse_action.action_prediction.float(),
            coarse_target.float(),
        )
        targets = ObjectTopTrainingTargets(
            teacher_dynamics=teacher,
            plan_recognition=recognition,
            online_intent_loss=online_intent_loss,
            plan_recognition_loss=recognition.reconstruction_loss,
            coarse_action_loss=coarse_loss,
            history_proposal_loss=coarse_loss.new_zeros(()),
            object_reconstruction_loss=context.facts.reconstruction_error,
        )
        if not collect_diagnostics:
            return targets, {}
        metrics = {
            **teacher_metrics,
            "object_intent_action_match_loss": action_match.detach(),
            "object_intent_state_match_loss": state_match.detach(),
            "object_intent_object_key_match_loss": object_key_match.detach(),
            "object_intent_object_value_match_loss": object_value_match.detach(),
            "object_intent_online_match_loss": online_intent_loss.detach(),
            "object_plan_recognition_loss": recognition.reconstruction_loss.detach(),
            "object_coarse_action_loss": coarse_loss.detach(),
        }
        return targets, metrics

    def compile_policy(
        self,
        context: DeploymentTopCache,
        *,
        factual_dock: ObjectFactualDock,
        action_query: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompiledPolicyState, dict[str, Tensor]]:
        """Run dynamic P2/P3; no observation or teacher input is accepted."""

        context.validate(hidden=self.hidden, horizon=self.horizon)
        effect, effect_metrics = self.effect_reader(
            action_query,
            context.predicted_dynamics,
            context.intent,
            factual_dock,
            collect_diagnostics=collect_diagnostics,
        )
        consequence, consequence_metrics = self.consequence(
            factual_base=factual_dock.aggregate_fact,
            effect=effect,
            collect_diagnostics=collect_diagnostics,
        )
        plan, plan_metrics = self.plan_compiler(
            factual_dock=factual_dock,
            consequence=consequence,
            intent=context.intent,
            action_query=action_query,
            collect_diagnostics=collect_diagnostics,
        )
        state = CompiledPolicyState(
            effect=effect,
            consequence=consequence,
            plan=plan,
        )
        state.validate()
        if not collect_diagnostics:
            return state, {}
        return state, {**effect_metrics, **consequence_metrics, **plan_metrics}


__all__ = [
    "CompiledPolicyState",
    "ObjectIntentDynamicsTop",
    "OnlineTopContext",
]
