"""Explicit G/S/W/P composition for the mainline top.

This module replaces the old eight-thousand-line planner dispatch for the
active capability.  It has three calls with non-overlapping authority:

``build_online_context``
    Current observation/history/language only.  Produces G, S and W once.
``build_training_targets``
    Training-only no-grad Teacher-G plus recognizer/loss targets.  It cannot
    mutate or replace the online context.
``compile_policy``
    ODE-step-dependent P2/P3 read from the completed dynamic P1 fact.

P1's expensive current-detail read is built exactly once by the independent
factual reader.  The live policy write is completed at each ODE step before it
enters this composer; P2/P3 cannot reopen a visual bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
from .routing import smooth_rms_contract
from .teacher import ObjectFutureTeacher
from .types import (
    CoarseActionIntentState,
    FutureObjectDynamics,
    LocalFactSet,
    ObjectFactSet,
    ObjectIntentState,
    ObjectTopTrainingTargets,
)
from ..v120_core.flow_dino_evidence import ProgressiveGroundingAddressState
from ..v120_core.role_delta_attnres import AffineVarianceFlooredCenteredNorm
from ..v120_core.trunk_primitives import TemporalDynamicsBoundDiTBlock


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
            **ground_metrics,
            **intent_metrics,
            **w1_metrics,
            **w2_metrics,
            "object_w_prediction_interval_variation": predicted.semantic_delta.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
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
            current_loss_support=context.facts.camera_validity,
        )
        online_intent_loss = F.smooth_l1_loss(
            context.intent.interval_queries.float(),
            recognition.interval_targets.detach().float(),
        )
        supervised_coarse = self.coarse_action(
            context.intent,
            future_action=future_action,
        )
        coarse_loss = supervised_coarse.loss
        targets = ObjectTopTrainingTargets(
            teacher_dynamics=teacher,
            current_loss_support=context.facts.camera_validity.detach().float(),
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
            "object_intent_online_match_loss": online_intent_loss.detach(),
            "object_plan_recognition_loss": recognition.reconstruction_loss.detach(),
            "object_coarse_action_loss": coarse_loss.detach(),
        }
        return targets, metrics

    def compile_policy(
        self,
        context: DeploymentTopCache,
        *,
        p1_fact: Tensor,
        action_query: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompiledPolicyState, dict[str, Tensor]]:
        """Run dynamic P2/P3; no observation or teacher input is accepted."""

        context.validate(hidden=self.hidden, horizon=self.horizon)
        # V120's P2 query was the live trajectory *after* the P1 factual
        # write, not the untouched noisy-action seed.  Keeping that residual
        # order is important: it lets the future-effect reader ask questions
        # in the factual chart that P1 actually selected.
        if tuple(p1_fact.shape) != tuple(action_query.shape):
            raise ValueError("completed P1 fact and action query must align")
        p1_action_query = action_query + p1_fact
        raw_effect, effect_metrics = self.effect_reader(
            p1_action_query,
            context.predicted_dynamics,
            context.intent,
            collect_diagnostics=collect_diagnostics,
        )
        # The original object path contracts the P2 write at the caller
        # boundary before it enters the zero-preserving consequence.  Applying
        # the bound only inside P3 (or omitting it) changes both the trajectory
        # seen by P3 and the controlled-transition coefficient geometry.
        effect, effect_contract = smooth_rms_contract(raw_effect, 0.35)
        consequence, consequence_metrics = self.consequence(
            factual_base=p1_fact,
            effect=effect,
            collect_diagnostics=collect_diagnostics,
        )
        # P3 likewise read the trajectory after the P2 write.  The protected
        # consequence is the complete P1+P2 residual, so adding it to the
        # original seed reconstructs that exact boundary without rebuilding a
        # generic canvas.
        p3_action_query = action_query + consequence.protected_consequence
        plan, plan_metrics = self.plan_compiler(
            p1_fact=p1_fact,
            consequence=consequence,
            intent=context.intent,
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
            "object_p2_effect_contract_min": effect_contract.detach().float().amin(),
            "object_p2_effect_postcontract_rms": effect.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }


__all__ = [
    "CompiledPolicyState",
    "ObjectIntentDynamicsTop",
    "OnlineTopContext",
]
