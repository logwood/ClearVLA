"""Explicit G/S/W/P composition for the mainline top.

This module replaces the old eight-thousand-line planner dispatch for the
active capability.  It has three calls with non-overlapping authority:

``build_online_context``
    Current observation/history/language only.  Produces G, S and W once.
``build_training_targets``
    Training-only no-grad Teacher-G plus recognizer/loss targets.  It cannot
    mutate or replace the online context.
``compile_policy``
    ODE-step-dependent P2/P3 read from the typed P1 policy state.

P1's expensive current-detail read is built exactly once by the independent
factual reader.  Its static fact and per-step policy residual remain distinct
when they enter this composer; P2/P3 cannot reopen a visual bank.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..v120_core.flow_dino_evidence import ProgressiveGroundingAddressState
from ..v120_core.role_delta_attnres import AffineVarianceFlooredCenteredNorm
from ..v120_core.trunk_primitives import TemporalDynamicsBoundDiTBlock
from .compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ObjectTypedEffect,
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
    CandidateWorld,
    CoarseActionIntentState,
    CompletedP1PolicyState,
    FutureObjectDynamics,
    LocalFactSet,
    ObjectFactSet,
    ObjectIntentState,
    ObjectTopTrainingTargets,
    ObjectWorldBelief,
    PhysicalActionCondition,
)


@dataclass(frozen=True)
class OnlineTopContext:
    """Static online G/S/W state cached once for five-step deployment."""

    facts: ObjectFactSet
    intent: ObjectIntentState
    coarse_action: CoarseActionIntentState
    candidate_world: CandidateWorld

    @property
    def action_condition(self) -> PhysicalActionCondition:
        """Read the action tag from the atomically cached world."""

        return self.candidate_world.action_condition

    @property
    def predicted_dynamics(self) -> FutureObjectDynamics:
        """Compatibility view; callers cannot replace it independently."""

        return self.candidate_world.dynamics

    def deployment_cache(self) -> "DeploymentTopCache":
        return DeploymentTopCache(
            belief=self.facts.world_belief(),
            intent=self.intent,
            candidate_world=self.candidate_world,
        )

    def validate(self, *, hidden: int, horizon: int) -> None:
        self.facts.validate()
        self.intent.validate(horizon=horizon, hidden=hidden)
        self.candidate_world.validate(
            action_dim=int(self.coarse_action.action_prediction.shape[-1])
        )
        self.action_condition.validate(
            action_dim=int(self.coarse_action.action_prediction.shape[-1])
        )
        expected = (self.facts.batch, 4, hidden)
        if tuple(self.coarse_action.tokens.shape) != expected:
            raise ValueError("coarse action tokens lost the interval axis")
        if self.coarse_action.target is not None:
            raise ValueError("online context cannot carry a future action target")
        if self.coarse_action.loss.ndim != 0:
            raise ValueError("coarse action online placeholder loss must be scalar")
        if tuple(self.coarse_action.action_prediction.shape) != tuple(
            self.action_condition.interval_action.shape
        ):
            raise ValueError("coarse proposal and physical W condition do not align")
        if self.action_condition.interval_action is not self.coarse_action.action_prediction:
            raise ValueError("physical W condition must retain the exact proposal tensor")


@dataclass(frozen=True)
class DeploymentTopCache:
    """Static S plus compact belief and one action-tagged W candidate."""

    belief: ObjectWorldBelief
    intent: ObjectIntentState
    candidate_world: CandidateWorld

    @property
    def action_condition(self) -> PhysicalActionCondition:
        """Read the action tag from the atomically cached world."""

        return self.candidate_world.action_condition

    @property
    def predicted_dynamics(self) -> FutureObjectDynamics:
        """Compatibility view; callers cannot replace it independently."""

        return self.candidate_world.dynamics

    def validate(self, *, hidden: int, horizon: int) -> None:
        self.belief.validate()
        self.intent.validate(horizon=horizon, hidden=hidden)
        self.candidate_world.validate(
            action_dim=int(self.candidate_world.action_condition.action_dim)
        )


@dataclass(frozen=True)
class CompiledPolicyState:
    """Dynamic P2/P3 state consumed by the bottom action model."""

    effect: ObjectTypedEffect
    consequence: ObjectConsequenceState
    plan: ObjectPolicyPlanDeltaBank

    def validate(self) -> None:
        self.effect.validate()
        self.consequence.validate()
        if tuple(self.effect.semantic.shape) != tuple(
            self.consequence.factual_base.shape
        ):
            raise ValueError("P2 effect and consequence schemas do not align")
        self.plan.validate()
        if tuple(self.plan.protected_base.shape) != tuple(self.effect.semantic.shape):
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
            action_dim=action_dim,
            heads=heads,
            normalization_floor=float(
                core_config.flow_jepa_routing_norm_floor
            ),
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
            route_dim=route_dim,
        )
        self.consequence = ZeroPreservingObjectConsequence(hidden)
        self.plan_compiler = ObjectPolicyPlanCompiler(
            hidden=hidden,
            horizon=horizon,
            basis=basis,
        )

    @staticmethod
    def _merge_world_diagnostics(
        w1_metrics: dict[str, Tensor],
        w2_metrics: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Keep one compact typed-normalization summary for a W materialization."""

        merged_w1 = dict(w1_metrics)
        merged_w2 = dict(w2_metrics)
        metrics: dict[str, Tensor] = {}
        for suffix, reduction in (
            ("denominator_min", torch.amin),
            ("gain_max", torch.amax),
            ("output_input_rms_ratio_max", torch.amax),
        ):
            values = []
            for source, name in (
                (merged_w1, "object_w1_typed_norm_"),
                (merged_w2, "object_w2_typed_norm_"),
            ):
                key = f"{name}{suffix}"
                if key in source:
                    values.append(source.pop(key))
            if values:
                metrics[f"object_w_typed_norm_{suffix}"] = reduction(
                    torch.stack(values)
                )
        metrics.update(merged_w1)
        metrics.update(merged_w2)
        return metrics

    def build_candidate_world(
        self,
        *,
        belief: ObjectFactSet | ObjectWorldBelief,
        action_condition: PhysicalActionCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[CandidateWorld, dict[str, Tensor]]:
        """Materialize exactly one W prediction for one physical proposal."""

        belief.validate()
        action_condition.validate(action_dim=self.action_dim)
        _, w1_state, w1_metrics = self.dynamics.forward_w1(
            facts=belief,
            action=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        predicted, w2_metrics = self.dynamics.forward_w2(
            facts=belief,
            w1_state=w1_state,
            collect_diagnostics=collect_diagnostics,
        )
        world = CandidateWorld(
            action_condition=action_condition,
            dynamics=predicted,
        )
        world.validate(action_dim=self.action_dim)
        if not collect_diagnostics:
            return world, {}
        return world, self._merge_world_diagnostics(w1_metrics, w2_metrics)

    def refine_deployment_world(
        self,
        context: DeploymentTopCache,
        *,
        action_condition: PhysicalActionCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[DeploymentTopCache, dict[str, Tensor]]:
        """Recompute W once for a decoded outer candidate action.

        G and S stay fixed for this observation.  Only the compact belief and
        the explicit physical action condition cross the refinement boundary;
        no teacher, dense source chart or new optimizer path is introduced.
        """

        context.validate(hidden=self.hidden, horizon=self.horizon)
        action_condition.validate(action_dim=self.action_dim)
        world, metrics = self.build_candidate_world(
            belief=context.belief,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        refined = replace(
            context,
            candidate_world=world,
        )
        refined.validate(hidden=self.hidden, horizon=self.horizon)
        if not collect_diagnostics:
            return refined, {}
        old_action = context.action_condition.interval_action.detach().float()
        new_action = action_condition.interval_action.detach().float()
        old_delta = context.action_condition.interval_delta.detach().float()
        new_delta = action_condition.interval_delta.detach().float()
        old_dynamics = context.predicted_dynamics
        new_dynamics = refined.predicted_dynamics
        old_semantic = old_dynamics.semantic_delta.detach().float()
        new_semantic = new_dynamics.semantic_delta.detach().float()
        old_transport = old_dynamics.transport_mean.detach().float()
        new_transport = new_dynamics.transport_mean.detach().float()
        metrics = {
            **metrics,
            "object_action_world_refinement_count": action_condition.interval_action.new_ones(
                (), dtype=torch.float32
            ),
            # Keep the proposal and the re-materialized condition visible as
            # separate quantities, without duplicate compatibility aliases.
            "object_action_world_refinement_pre_action_interval_rms": old_action.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_post_action_interval_rms": new_action.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_action_interval_delta_rms": (
                new_action - old_action
            ).square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_pre_action_delta_rms": old_delta.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_post_action_delta_rms": new_delta.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_pre_semantic_delta_rms": old_semantic.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_post_semantic_delta_rms": new_semantic.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_semantic_delta_change_rms": (
                new_semantic - old_semantic
            ).square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_pre_transport_rms": old_transport.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_post_transport_rms": new_transport.square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_transport_change_rms": (
                new_transport - old_transport
            ).square()
            .mean()
            .sqrt(),
            "object_action_world_refinement_tag_identity_error": action_condition.interval_action.new_zeros(
                (), dtype=torch.float32
            ),
        }
        return refined, metrics

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
        action_state: Tensor,
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
        coarse = self.coarse_action(action_intent)
        action_condition = PhysicalActionCondition.from_interval_action(
            coarse.action_prediction,
            action_state,
        )
        world, world_metrics = self.build_candidate_world(
            belief=facts,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        predicted = world.dynamics
        context = OnlineTopContext(
            facts=facts,
            intent=intent,
            coarse_action=coarse,
            candidate_world=world,
        )
        context.validate(hidden=self.hidden, horizon=self.horizon)
        if not collect_diagnostics:
            return context, {}
        metrics = {
            **ground_metrics,
            **intent_metrics,
            **world_metrics,
            "object_w_prediction_interval_variation": predicted.semantic_delta.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "object_coarse_action_rms": coarse.action_prediction.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_action_world_initial_materialization_count": predicted.semantic_delta.new_ones(
                (), dtype=torch.float32
            ),
            "object_action_world_initial_tag_identity_error": (
                action_condition.interval_action.detach().float()
                - coarse.action_prediction.detach().float()
            )
            .abs()
            .amax(),
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
        # The action condition is carried beside its world prediction through
        # the complete cache.  P2 cannot receive an untagged/stale W tensor by
        # API construction; a later second outer refinement must create a new
        # DeploymentTopCache rather than mutating this pair.
        context.action_condition.validate(action_dim=self.action_dim)
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
            context.intent.public_interval_carrier.float(),
            recognition.interval_targets.detach().float(),
        )
        supervised_coarse = self.coarse_action(
            context.intent.action_dock(),
            # The runtime action condition is a deterministic projection of
            # the deployable 24-row candidate.  Supervise the online proposal
            # from that exact same prefix so one [B,4,A] ABI never means
            # 48 rows during training and 24 rows during deployment.
            future_action=future_action[:, : self.horizon],
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
        # P2 alone consumes the complete post-P1 query. Keep the three owners
        # named until this exact read so the live policy write cannot acquire
        # a false factual identity elsewhere in the graph.
        p1_action_query = p1_state.p2_dock(action_query).combined()
        candidate_world = context.candidate_world
        candidate_world.validate(action_dim=self.action_dim)
        candidate_world.assert_action_identity(context.action_condition)
        raw_effect, effect_metrics = self.effect_reader.forward_candidate(
            p1_action_query,
            candidate_world,
            context.intent.policy_dock(),
            action_condition=context.action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        # The original object path contracts the P2 write at the caller
        # boundary before it enters the zero-preserving consequence.  Applying
        # the bound only inside P3 (or omitting it) changes both the trajectory
        # seen by P3 and the controlled-transition coefficient geometry.
        contracted_combined, effect_contract = smooth_rms_contract(
            raw_effect.combined(),
            0.35,
        )
        # Keep semantic/geometry identity through consequence while preserving
        # the exact caller-owned aggregate RMS boundary.  Both carriers receive
        # the same parameter-free scale computed from their combined value.
        effect = raw_effect.scaled(effect_contract)
        consequence, consequence_metrics = self.consequence(
            factual_base=p1_state.factual_base,
            effect=effect,
            collect_diagnostics=collect_diagnostics,
        )
        # P3 likewise read the trajectory after the P2 write.  The protected
        # consequence is the complete P1+P2 residual, so adding it to the
        # original seed reconstructs that exact boundary without rebuilding a
        # generic canvas.
        p3_action_query = action_query + consequence.protected_consequence
        plan, plan_metrics = self.plan_compiler(
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
            "object_p2_effect_contract_min": effect_contract.detach().float().amin(),
            "object_p2_effect_contract_identity_error": (
                effect.combined().detach().float()
                - contracted_combined.detach().float()
            )
            .abs()
            .amax(),
            "object_p2_effect_postcontract_rms": effect.combined().detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_semantic_effect_postcontract_rms": effect.semantic.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "object_p2_geometry_effect_postcontract_rms": effect.geometry.detach()
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
