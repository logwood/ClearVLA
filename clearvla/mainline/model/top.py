"""Explicit G/S/W/P composition for the mainline top.

This module replaces the old eight-thousand-line planner dispatch for the
active capability.  It has three calls with non-overlapping authority:

``build_online_context``
    Current observation/history/language only.  Produces G, S and W once.
``build_training_targets``
    Training-only no-grad Teacher-G plus the S-owned observable-state target.
    It cannot
    mutate or replace the online context.
``compile_policy``
    ODE-step-dependent P2/P3 read from the typed live P1 policy state.

P1's expensive current-detail read is built exactly once by the independent
factual reader. The live policy write is completed at each ODE step but remains
separate from the factual base; P2/P3 cannot reopen a visual bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from ..v120_core.flow_dino_evidence import ProgressiveGroundingAddressState
from ..v120_core.role_delta_attnres import AffineVarianceFlooredCenteredNorm
from ..v120_core.trunk_primitives import TemporalDynamicsBoundDiTBlock
from .compiler import (
    ObjectConsequenceState,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ZeroPreservingObjectConsequence,
)
from .dynamics import ObjectFutureDynamicsCompiler
from .effect_terminal import ObjectFutureEffectTerminal
from .grounding import DenseObjectGrounder
from .intent import (
    CoarseActionIntent,
    ObservableIntentStateSupervisor,
    StatelessObjectIntentOrganizer,
)
from .teacher import ObjectFutureTeacher
from .types import (
    CoarseActionIntentState,
    CompletedP1PolicyState,
    FutureObjectDynamics,
    LocalFactSet,
    ObjectFactSet,
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
        if tuple(self.coarse_action.tokens.shape) != expected:
            raise ValueError("coarse action tokens lost the interval axis")
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
        self.consequence.validate()
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
        core_config=None,
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
        del role_host_expansion, role_host_dropout
        if core_config is None:
            raise ValueError("the V120 G stack requires its resolved core configuration")
        if int(role_host_depth) != 3:
            raise ValueError("the active progressive grounding path requires G1/G2/G3")
        self.grounding_blocks = nn.ModuleList(
            [
                TemporalDynamicsBoundDiTBlock(core_config, role="grounding")
                for _ in range(3)
            ]
        )
        self.grounding_content_mod = nn.Sequential(
            AffineVarianceFlooredCenteredNorm(
                2 * hidden,
                float(core_config.flow_jepa_routing_norm_floor),
                affine_maximum=4.0,
            ),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.normal_(self.grounding_content_mod[-1].weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.grounding_content_mod[-1].bias)
        self.grounding_content_mod_scale = nn.Parameter(torch.tensor(0.10))
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
            route_dim=route_dim,
            heads=heads,
        )
        self.dynamics = ObjectFutureDynamicsCompiler(
            hidden=hidden,
            content_dim=content_dim,
            route_dim=route_dim,
            heads=heads,
        )
        self.teacher = ObjectFutureTeacher(
            content_dim=content_dim,
            key_dim=teacher_key_dim,
            flow_reference_frames=flow_reference_frames,
        )
        self.intent_supervisor = ObservableIntentStateSupervisor(
            hidden=hidden,
            state_dim=state_dim,
        )
        self.effect_reader = ObjectFutureEffectTerminal(
            hidden=hidden,
            content_dim=content_dim,
            route_dim=route_dim,
        )
        self.consequence = ZeroPreservingObjectConsequence(hidden)
        self.plan_compiler = ObjectPolicyPlanCompiler(
            hidden=hidden,
            horizon=horizon,
            basis=basis,
        )

    def run_progressive_grounding(
        self,
        *,
        canvas: Tensor,
        slices: dict[str, slice],
        visual_memory: Tensor,
        visual_value_memory: Tensor,
        state: ProgressiveGroundingAddressState,
        advance: Callable[..., ProgressiveGroundingAddressState],
        collect_diagnostics: bool = False,
    ) -> tuple[ProgressiveGroundingAddressState, Tensor, dict[str, Tensor]]:
        """Execute the literal V120 G1/G2/G3 block/update alternation."""

        clean = torch.cat(
            (canvas[:, slices["state"]], canvas[:, slices["registers"]]), dim=1
        )
        if int(clean.shape[1]) < 1:
            raise ValueError("grounding modulation requires state/register rows")
        summary = torch.cat((clean.mean(dim=1), visual_memory.mean(dim=1)), dim=-1)
        content_delta = self.grounding_content_mod(summary)
        content_delta = content_delta * self.grounding_content_mod_scale.to(
            device=canvas.device, dtype=canvas.dtype
        )
        # G is a cached current fact. Its exact endpoint is t_v120=0, hence
        # the V120 time embedding is algebraically zero at this static boundary.
        modulation = content_delta
        metrics: dict[str, Tensor] = {}
        for stage, block in enumerate(self.grounding_blocks, start=1):
            before = canvas[:, slices["rollout"]]
            canvas, block_metrics = block(
                canvas,
                visual_memory,
                modulation,
                slices,
                visual_value_memory=visual_value_memory,
                collect_diagnostics=collect_diagnostics,
            )
            rollout = canvas[:, slices["rollout"]]
            state = advance(
                state,
                rollout,
                stage=stage,
                collect_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics:
                metrics[f"grounding_g{stage}_update_rms"] = (
                    rollout.detach().float() - before.detach().float()
                ).square().mean().sqrt()
                metrics.update(
                    {f"grounding_g{stage}_{name}": value for name, value in block_metrics.items()}
                )
        if state.stage != 3 or state.grounded_fact_set is None:
            raise RuntimeError("progressive grounding did not complete G3")
        if collect_diagnostics:
            metrics["grounding_clean_endpoint_t_v120"] = canvas.new_zeros(
                (), dtype=torch.float32
            )
        return state, canvas, metrics

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

        facts, ground_metrics = self.grounder(
            local_facts,
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
        action_intent = intent.action_dock()
        world_intent = intent.world_dock()
        coarse = self.coarse_action(action_intent)
        w1_diagnostic_field, w1_state, w1_metrics = self.dynamics.forward_w1(
            facts=facts,
            intent=world_intent,
            action=coarse,
            collect_diagnostics=collect_diagnostics,
        )
        # The optional two-interval decode exists only to materialize W1
        # diagnostics.  Its metrics are detached inside ``forward_w1``; keeping
        # the field alive across W2 needlessly overlaps a second decoder
        # autograd graph with the final four-interval decode on logging batches.
        del w1_diagnostic_field
        predicted, w2_metrics = self.dynamics.forward_w2(
            facts=facts,
            intent=world_intent,
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
            **ground_metrics,
            **intent_metrics,
            **w1_metrics,
            **w2_metrics,
            "object_w_prediction_interval_variation": predicted.semantic_delta.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_w_prediction_common_effect_rms": predicted.semantic_common.detach()
            .square()
            .mean()
            .sqrt(),
            "object_w_prediction_interval_residual_rms": predicted.semantic_interval_residual
            .detach()
            .square()
            .mean()
            .sqrt(),
            "object_coarse_action_rms": coarse.action_prediction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        return context, metrics

    def build_training_targets(
        self,
        context: OnlineTopContext,
        *,
        future_supports: Tensor,
        future_offsets: Tensor,
        future_action: Tensor,
        current_state: Tensor,
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
        supervision = self.intent_supervisor(
            intent=context.intent,
            current_state=current_state,
            future_state=future_state,
        )
        supervised_coarse = self.coarse_action.attach_training_target(
            context.coarse_action,
            future_action,
        )
        coarse_loss = supervised_coarse.loss
        targets = ObjectTopTrainingTargets(
            teacher_dynamics=teacher,
            current_loss_support=(
                context.facts.camera_chart_availability.detach().float()
            ),
            intent_supervision=supervision,
            public_intent_loss=supervision.loss,
            coarse_action_loss=coarse_loss,
            history_proposal_loss=coarse_loss.new_zeros(()),
            object_reconstruction_loss=context.facts.reconstruction_error,
        )
        if not collect_diagnostics:
            return targets, {}
        metrics = {
            **teacher_metrics,
            "object_intent_public_future_increment_loss": supervision.loss.detach(),
            "object_intent_future_increment_prediction_rms": supervision.state_prediction
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_future_increment_target_rms": supervision.state_target
            .detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_intent_future_cumulative_reconstruction_error": (
                supervision.state_prediction.detach().float().cumsum(dim=1)
                - supervision.state_target.detach().float().cumsum(dim=1)
            )
            .square()
            .mean()
            .sqrt(),
            "object_coarse_action_loss": coarse_loss.detach(),
        }
        return targets, metrics

    def compile_policy(
        self,
        context: DeploymentTopCache,
        *,
        p1_state: CompletedP1PolicyState,
        action_query: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompiledPolicyState, dict[str, Tensor]]:
        """Run dynamic P2/P3; no observation or teacher input is accepted."""

        context.validate(hidden=self.hidden, horizon=self.horizon)
        p1_state.validate(
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if tuple(p1_state.factual_base.shape) != tuple(action_query.shape):
            raise ValueError("completed P1 policy state and action query must align")
        # Preserve V120's post-P1 P2 query exactly while keeping the live
        # policy-block write outside the observation-owned factual base.
        p2_query = p1_state.p2_dock(action_query).combined()
        typed_effect, effect_metrics = self.effect_reader(
            p2_query,
            context.predicted_dynamics,
            context.intent.policy_dock(),
            collect_diagnostics=collect_diagnostics,
        )
        effect = typed_effect.physical_sum
        consequence, consequence_metrics = self.consequence(
            factual_base=p1_state.factual_base,
            effect=typed_effect,
            collect_diagnostics=collect_diagnostics,
        )
        # The protected consequence carries only the static P1 fact plus W
        # effect. The live P1 write may refine P2 and P3 precision, but never
        # becomes a factual/protected value.
        p3_action_query = action_query
        plan, plan_metrics = self.plan_compiler(
            p1_factual_detail=p1_state.factual_base,
            p1_policy_residual=p1_state.policy_query_residual,
            consequence=consequence,
            intent=context.intent.policy_dock(),
            action_query=p3_action_query,
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
        return state, {
            **effect_metrics,
            **consequence_metrics,
            **plan_metrics,
        }


__all__ = [
    "CompiledPolicyState",
    "ObjectIntentDynamicsTop",
    "OnlineTopContext",
]
