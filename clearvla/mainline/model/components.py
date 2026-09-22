"""Registered component owners for the Schema30 mainline composition.

This module contains no alternate execution graph.  Each owner receives the
already-constructed child modules from the composition root and exposes only
the stage boundary it owns.  The source monoliths remain useful as construction
helpers, but are never registered beneath :class:`ClearVLAMainlinePolicy`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..future_time import LEGACY_FUTURE_TIME, resolve_future_time
from ..gripper_contract import is_binary_gripper_selection
from ..interfaces import CurrentObservation, ObservableHistory, OnlinePolicyInput
from ..supervision import FutureLabelSupport, supported_mean
from ..v120_core.flow_dino_evidence import ProgressiveGroundingAddressState
from ..v120_core.primitives import TimeEmbedding
from ..v120_core.role_delta_attnres import PolicyRoleDeltaBank
from ..v120_core.trunk_primitives import TemporalDynamicsBoundDiTBlock
from .action_codec import (
    PhysicalActionFieldCodec,
    PhysicalActionFieldParts,
    binary_gripper_command_from_logits,
)
from .action_contract import ActionQueryEncoder, BottomDecoderOutput, V120SeedContext
from .compiler import (
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ZeroPreservingObjectConsequence,
)
from .component_contracts import OutletActionOutput
from .grounding import DenseObjectGrounder
from .intent import CoarseActionIntent, FuturePlanRecognizer, StatelessObjectIntentOrganizer
from .observation_contract import GroundingObservationBank, ObservationEvidence
from .proposal import HistoryActionProposal
from .restored_observation import (
    RestoredV120ObservationCompiler,
    _PreparedV120Observation,
)
from .routing import smooth_rms_contract
from .target_binding import BoundTargetRead, TargetBinding, TargetEvidence
from .teacher import ObjectFutureTeacher
from .top import (
    CompiledPolicyState,
    DeploymentTopCache,
    OnlineTopContext,
)
from .types import (
    ABSOLUTE_ACTION_SEQUENCE_CHART,
    RELATIVE_COMMAND_SEQUENCE_CHART,
    CandidateWorld,
    CompletedP1PolicyState,
    ControlledTransitionState,
    FactualPrecisionDock,
    FlowStepContext,
    LocalFactSet,
    ObjectFactSet,
    ObjectIntentState,
    ObjectTopTrainingTargets,
    ObjectWorldBelief,
    ObservedActionSequenceCondition,
    OutletPhysicalActionCondition,
    PhysicalActionCondition,
    PhysicalActionSequenceCondition,
    SupervisedWorld,
    WorldActionCondition,
    intersect_object_camera_validity,
    physical_action_normalizer_fingerprint,
)


class ConditioningStage(nn.Module):
    """Own goal/history masks and the auxiliary history proposal."""

    def __init__(self, history_proposal: HistoryActionProposal) -> None:
        super().__init__()
        self.history_proposal = history_proposal

    def prepare(
        self,
        policy_input: OnlinePolicyInput,
        *,
        config: Any,
        training: bool,
        training_mask: bool,
        condition_generator: torch.Generator | None,
    ) -> tuple[OnlinePolicyInput, Any, Tensor, Tensor]:
        batch = policy_input.batch
        condition_is_training = bool(training and training_mask)

        def condition_keep(dropout: float, *, dtype: torch.dtype) -> Tensor:
            if not condition_is_training or float(dropout) <= 0.0:
                return torch.ones(batch, device=policy_input.device, dtype=dtype)
            return (
                torch.rand(
                    batch,
                    device=policy_input.device,
                    generator=condition_generator,
                )
                >= float(dropout)
            ).to(dtype=dtype)

        goal_keep = condition_keep(
            config.top.goal_condition_dropout,
            dtype=policy_input.goal.tokens.dtype,
        )
        history_keep = condition_keep(
            config.top.action_history_condition_dropout,
            dtype=policy_input.history.executed_action_history.dtype,
        )
        history_proposal = self.history_proposal(
            policy_input.history.executed_action_history,
            **(
                {"timing": policy_input.history.timing}
                if config.top.history_encoding_mode == "timestamped_streams_v1"
                else {}
            ),
        )
        if config.top.history_encoding_mode == "timestamped_streams_v1":
            assert policy_input.history.timing is not None
            history_keep = history_keep * policy_input.history.timing.action_executed.any(dim=1).to(
                history_keep.dtype
            )
        state_history = policy_input.history.state_history
        executed_history = policy_input.history.executed_action_history
        timing = policy_input.history.timing
        if config.top.history_encoding_mode == "timestamped_streams_v1":
            assert timing is not None
            # Quarantine padded values before *all* consumers, not only S.
            # The current state has one authoritative row across the graph.
            state_history = torch.cat(
                (state_history[:, :-1], policy_input.history.state[:, None]), dim=1
            )
            state_history = torch.where(
                timing.state_observed[..., None], state_history, torch.zeros_like(state_history)
            )
            executed_history = torch.where(
                timing.action_executed[..., None],
                executed_history,
                torch.zeros_like(executed_history),
            )
        conditioned_history = replace(
            policy_input.history,
            timing=None if timing is None else timing.without_actions(history_keep),
            state_history=state_history,
            executed_action_history=executed_history * history_keep[:, None, None],
        )
        conditioned_goal = replace(
            policy_input.goal,
            tokens=policy_input.goal.tokens * goal_keep[:, None, None],
        )
        conditioned = replace(
            policy_input,
            history=conditioned_history,
            goal=conditioned_goal,
        )
        return conditioned, history_proposal, goal_keep, history_keep


class ObservationStage(nn.Module):
    """Registered owner of the observation compiler and grounding bank."""

    def __init__(self, compiler: RestoredV120ObservationCompiler) -> None:
        super().__init__()
        self.compiler = compiler

    @property
    def v120_config(self):
        return self.compiler.v120_config

    def teacher_supports(self, tokens: Tensor) -> Tensor:
        return self.compiler.teacher_supports(tokens)

    def prepare(
        self,
        observation: CurrentObservation,
        *,
        context_mask: Tensor | None = None,
        source_offsets: Tensor | None = None,
        training_mask: bool = False,
        geometry_supervision: bool = True,
        collect_diagnostics: bool = False,
    ) -> _PreparedV120Observation:
        return self.compiler.prepare(
            observation,
            context_mask=context_mask,
            source_offsets=source_offsets,
            training_mask=training_mask,
            geometry_supervision=geometry_supervision,
            collect_diagnostics=collect_diagnostics,
        )

    def build_grounding_bank(
        self,
        prepared: _PreparedV120Observation,
        grounding_canvas: Tensor,
        slices: dict[str, slice],
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[GroundingObservationBank, dict[str, Tensor]]:
        return self.compiler.build_grounding_bank(
            prepared,
            grounding_canvas,
            slices,
            collect_diagnostics=collect_diagnostics,
        )

    def begin_progressive_grounding(
        self,
        bank: GroundingObservationBank,
    ) -> ProgressiveGroundingAddressState:
        return self.compiler.begin_progressive_grounding(bank)

    def advance_progressive_grounding(
        self,
        state: ProgressiveGroundingAddressState,
        rollout: Tensor,
        *,
        stage: int,
        collect_diagnostics: bool = False,
    ) -> ProgressiveGroundingAddressState:
        return self.compiler.advance_progressive_grounding(
            state,
            rollout,
            stage=stage,
            collect_diagnostics=collect_diagnostics,
        )

    def finalize_grounding(
        self,
        bank: GroundingObservationBank,
        state: ProgressiveGroundingAddressState,
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[ObservationEvidence, dict[str, Tensor]]:
        return self.compiler.finalize_grounding(
            bank,
            state,
            collect_diagnostics=collect_diagnostics,
        )


class BridgeStage(nn.Module):
    """Shared role/query bridge used by both grounding and dynamic P1."""

    def __init__(self, query_encoder: ActionQueryEncoder) -> None:
        super().__init__()
        self.query_encoder = query_encoder

    def action_query(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        return self.query_encoder(noisy_action_field, time)

    def action_and_context(
        self,
        noisy_action_field: Tensor,
        time: Tensor,
        history: ObservableHistory,
        *,
        executed_memory: Tensor,
        action_history_keep: Tensor,
        role: Tensor | None = None,
    ) -> tuple[Tensor, V120SeedContext]:
        return self.query_encoder.forward_with_context(
            noisy_action_field,
            time,
            history,
            executed_memory=executed_memory,
            action_history_keep=action_history_keep,
            role=role,
        )

    def sample_role_context(self, reference: Tensor) -> Tensor:
        return self.query_encoder.sample_role_table(reference)

    def build_grounding_seed(
        self,
        *,
        state: Tensor,
        rollout_init: Tensor,
        role: Tensor,
    ) -> tuple[Tensor, dict[str, slice]]:
        return self.query_encoder.grounding_canvas(
            state=state,
            rollout_init=rollout_init,
            role=role,
        )

    def clean_action_basis(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.query_encoder.clean_action_basis_tokens(
            batch,
            device=device,
            dtype=dtype,
        )


class GroundingStage(nn.Module):
    """G1/G2/G3 progressive grounding and current fact materialization."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        blocks: nn.ModuleList,
        content_mod: nn.Sequential,
        content_mod_scale: nn.Parameter,
        grounder: DenseObjectGrounder,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.content_dim = int(content_dim)
        # Canonical registered names are intentionally short and stage-local.
        self.blocks = blocks
        self.content_mod = content_mod
        self.content_mod_scale = content_mod_scale
        self.grounder = grounder

    def materialize_facts(
        self,
        local_facts: LocalFactSet,
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectFactSet, dict[str, Tensor]]:
        return self.grounder(
            local_facts,
            collect_diagnostics=collect_diagnostics,
        )

    def build_current(
        self,
        *,
        canvas: Tensor,
        slices: dict[str, slice],
        visual_memory: Tensor,
        visual_value_memory: Tensor,
        visual_memory_observed: Tensor | None = None,
        state: ProgressiveGroundingAddressState,
        advance: Callable[..., ProgressiveGroundingAddressState],
        collect_diagnostics: bool = False,
    ) -> tuple[ProgressiveGroundingAddressState, Tensor, dict[str, Tensor]]:
        clean = torch.cat(
            (canvas[:, slices["state"]], canvas[:, slices["registers"]]), dim=1
        )
        if int(clean.shape[1]) < 1:
            raise ValueError("grounding modulation requires state/register rows")
        if visual_memory_observed is None:
            visual_summary = visual_memory.mean(dim=1)
        else:
            if tuple(visual_memory_observed.shape) != tuple(visual_memory.shape[:2]) or visual_memory_observed.dtype != torch.bool:
                raise ValueError("grounding visual support must be Boolean [B,N]")
            if not bool(visual_memory_observed.any(dim=1).all()):
                raise ValueError("grounding requires at least the real current observation")
            safe = torch.where(visual_memory_observed[..., None], visual_memory, torch.zeros_like(visual_memory))
            visual_summary = safe.sum(dim=1) / visual_memory_observed.sum(dim=1, keepdim=True).clamp_min(1)
        summary = torch.cat((clean.mean(dim=1), visual_summary), dim=-1)
        content_delta = self.content_mod(summary)
        content_delta = content_delta * self.content_mod_scale.to(
            device=canvas.device, dtype=canvas.dtype
        )
        # G is a cached current fact. Its exact endpoint is t_v120=0, hence
        # the V120 time embedding is algebraically zero at this static boundary.
        modulation = content_delta
        metrics: dict[str, Tensor] = {}
        for stage, block in enumerate(self.blocks, start=1):
            before = canvas[:, slices["rollout"]]
            canvas, block_metrics = block(
                canvas,
                visual_memory,
                modulation,
                slices,
                visual_value_memory=visual_value_memory,
                visual_key_padding_mask=None if visual_memory_observed is None else ~visual_memory_observed,
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
                    {
                        f"grounding_g{stage}_{name}": value
                        for name, value in block_metrics.items()
                    }
                )
        if state.stage != 3 or state.grounded_fact_set is None:
            raise RuntimeError("progressive grounding did not complete G3")
        if collect_diagnostics:
            metrics["grounding_clean_endpoint_t_v120"] = canvas.new_zeros(
                (), dtype=torch.float32
            )
        return state, canvas, metrics


class IntentStage(nn.Module):
    """S intent organization and the four-interval coarse action proposal."""

    def __init__(nself, organizer: StatelessObjectIntentOrganizer, coarse_action: CoarseActionIntent) -> None:
        super().__init__()
        nself.organizer = organizer
        nself.coarse_action = coarse_action

    def organize(self, **kwargs: Any):
        return self.organizer(**kwargs)

    def propose_action(self, intent: Any, **kwargs: Any):
        return self.coarse_action(intent, **kwargs)


class WorldStage(nn.Module):
    """W1/W2 candidate-world materialization and one-shot refinement."""

    def __init__(
        self,
        *,
        hidden: int,
        action_dim: int,
        horizon: int,
        basis: int,
        dynamics: Any,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.dynamics = dynamics

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

    def materialize(
        self,
        *,
        belief: ObjectFactSet | ObjectWorldBelief,
        action_condition: WorldActionCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[CandidateWorld, dict[str, Tensor]]:
        belief.validate()
        if not isinstance(action_condition, (PhysicalActionCondition, PhysicalActionSequenceCondition)):
            raise TypeError("online world rejects training-only observed controls")
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

    def materialize_supervised(
        self, *, belief: ObjectFactSet | ObjectWorldBelief,
        action_condition: ObservedActionSequenceCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[SupervisedWorld, dict[str, Tensor]]:
        """Share W parameters, not the online candidate's physical controls."""
        if not isinstance(action_condition, ObservedActionSequenceCondition):
            raise TypeError("supervised world requires source-observed controls")
        belief.validate()
        action_condition.validate(action_dim=self.action_dim)
        _, state, near_metrics = self.dynamics.forward_w1(
            facts=belief, action=action_condition, collect_diagnostics=collect_diagnostics,
        )
        prediction, far_metrics = self.dynamics.forward_w2(
            facts=belief, w1_state=state, collect_diagnostics=collect_diagnostics,
        )
        result = SupervisedWorld(action_condition, prediction)
        result.validate(action_dim=self.action_dim)
        metrics = self._merge_world_diagnostics(near_metrics, far_metrics) if collect_diagnostics else {}
        # Keep observed-control and candidate-control diagnostics distinct.
        return result, {"supervised_" + key: value for key, value in metrics.items()}

    def refine_deployment_world(
        self,
        context: DeploymentTopCache,
        *,
        action_condition: WorldActionCondition,
        collect_diagnostics: bool = False,
    ) -> tuple[DeploymentTopCache, dict[str, Tensor]]:
        context.validate(hidden=self.hidden, horizon=self.horizon)
        action_condition.validate(action_dim=self.action_dim)
        old_condition = context.action_condition
        if isinstance(action_condition, PhysicalActionSequenceCondition):
            if not isinstance(old_condition, PhysicalActionSequenceCondition):
                raise TypeError("world refinement cannot change action condition schema")
            if old_condition.metadata_identity != action_condition.metadata_identity:
                raise ValueError("world refinement action sequence metadata changed")
        elif isinstance(action_condition, PhysicalActionCondition):
            if not isinstance(old_condition, PhysicalActionCondition):
                raise TypeError("world refinement cannot change action condition schema")
        else:
            raise TypeError("world refinement received an unknown action condition schema")
        world, metrics = self.materialize(
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
        if isinstance(action_condition, PhysicalActionSequenceCondition):
            old_action = old_condition.canonical_value.detach().float()
            new_action = action_condition.canonical_value.detach().float()
            old_delta = old_condition.canonical_delta.detach().float()
            new_delta = action_condition.canonical_delta.detach().float()
            reference = action_condition.source_action
            action_metrics = {
                "object_action_world_refinement_pre_action_sequence_rms": old_action.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_post_action_sequence_rms": new_action.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_action_sequence_change_rms": (
                    new_action - old_action
                )
                .square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_pre_action_sequence_delta_rms": old_delta.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_post_action_sequence_delta_rms": new_delta.square()
                .mean()
                .sqrt(),
            }
        else:
            old_action = old_condition.interval_action.detach().float()
            new_action = action_condition.interval_action.detach().float()
            old_delta = old_condition.interval_delta.detach().float()
            new_delta = action_condition.interval_delta.detach().float()
            reference = action_condition.interval_action
            action_metrics = {
                "object_action_world_refinement_pre_action_interval_rms": old_action.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_post_action_interval_rms": new_action.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_action_interval_delta_rms": (
                    new_action - old_action
                )
                .square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_pre_action_delta_rms": old_delta.square()
                .mean()
                .sqrt(),
                "object_action_world_refinement_post_action_delta_rms": new_delta.square()
                .mean()
                .sqrt(),
            }
        old_dynamics = context.predicted_dynamics
        new_dynamics = refined.predicted_dynamics
        old_semantic = old_dynamics.semantic_delta.detach().float()
        new_semantic = new_dynamics.semantic_delta.detach().float()
        old_transport = old_dynamics.transport_mean.detach().float()
        new_transport = new_dynamics.transport_mean.detach().float()
        metrics = {
            **metrics,
            **action_metrics,
            "object_action_world_refinement_count": reference.new_ones(
                (), dtype=torch.float32
            ),
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
            "object_action_world_refinement_tag_identity_error": reference.new_zeros(
                (), dtype=torch.float32
            ),
        }
        return refined, metrics




class P1Stage(nn.Module):
    """Static factual detail and dynamic P1 policy residual."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
        factual_reader: nn.Module,
        target_reader: BoundTargetRead | None = None,
        dynamic_time: TimeEmbedding,
        dynamic_content_mod: nn.Sequential,
        dynamic_content_mod_scale: nn.Parameter,
        dynamic_policy_block: TemporalDynamicsBoundDiTBlock,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.factual_reader = factual_reader
        self.target_read = target_reader
        self.dynamic_time = dynamic_time
        self.dynamic_content_mod = dynamic_content_mod
        self.dynamic_content_mod_scale = dynamic_content_mod_scale
        self.dynamic_policy_block = dynamic_policy_block

    def build_static(
        self,
        *,
        clean_trajectory: Tensor,
        g3_rollout: Tensor,
        detail: Any,
        phase_context: Tensor,
        condition_query_context: Tensor,
        history_query_context: Tensor,
        target_binding: TargetBinding | None = None,
        target_evidence: TargetEvidence | None = None,
        clean_basis_tokens: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[FactualPrecisionDock, dict[str, Tensor]]:
        updated, metrics = self.factual_reader(
            clean_trajectory,
            g3_rollout,
            detail,
            phase_context=phase_context,
            condition_query_context=condition_query_context,
            history_query_context=history_query_context,
            clean_basis_tokens=clean_basis_tokens,
            collect_diagnostics=collect_diagnostics,
        )
        if self.target_read is not None:
            if target_binding is None or target_evidence is None:
                raise ValueError("shared factual read requires the current target bundle")
            # Add AFTER subtracting the clean basis: its S-owned common mass is
            # physical evidence, not a query offset that a residual can cancel.
            target_query = clean_basis_tokens + phase_context.mean(1)[:, None, None]
            target_detail = self.target_read(target_query, target_evidence, target_binding)
        else:
            if target_binding is not None or target_evidence is not None:
                raise ValueError("legacy factual read cannot ignore target evidence")
            target_detail = None
        batch = int(clean_trajectory.shape[0])
        factual = FactualPrecisionDock(
            protected_detail=(updated - clean_trajectory).reshape(
                batch,
                self.horizon,
                self.basis,
                self.hidden,
            )
        )
        if target_detail is not None:
            factual = FactualPrecisionDock(factual.protected_detail + target_detail)
        factual.validate(horizon=self.horizon, basis=self.basis)
        return factual, metrics

    def update_dynamic(
        self,
        *,
        action_query: Tensor,
        factual: FactualPrecisionDock,
        time: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompletedP1PolicyState, dict[str, Tensor]]:
        protected_detail = factual.protected_detail
        expected = (
            int(action_query.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected or tuple(protected_detail.shape) != expected:
            raise ValueError("dynamic P1 inputs must align as [B,T,Q,H]")
        if tuple(time.shape) != (expected[0],):
            raise ValueError("dynamic P1 time must be [B]")
        trajectory = action_query + protected_detail
        canvas = trajectory.flatten(1, 2)
        rows = int(canvas.shape[1])
        empty_before = slice(0, 0)
        empty_after = slice(rows, rows)
        slices = {
            "task": empty_before,
            "state": empty_before,
            "state_history": empty_before,
            "executed": empty_before,
            "proposal": empty_before,
            "trajectory": slice(0, rows),
            "stage": empty_after,
            "rollout": empty_after,
            "registers": empty_after,
        }
        trajectory_summary = canvas.mean(dim=1)
        content_delta = self.dynamic_content_mod(
            torch.cat((trajectory_summary, trajectory_summary), dim=-1)
        ) * self.dynamic_content_mod_scale.to(
            device=canvas.device,
            dtype=canvas.dtype,
        )
        time_input = time.to(device=canvas.device, dtype=canvas.dtype)
        mod_embed = self.dynamic_time(time_input) + content_delta
        updated, block_metrics = self.dynamic_policy_block(
            canvas,
            canvas[:, :0],
            mod_embed,
            slices,
            collect_diagnostics=collect_diagnostics,
        )
        dynamic_delta = (updated - canvas).reshape(expected)
        state = CompletedP1PolicyState(
            factual_base=protected_detail,
            policy_query_residual=dynamic_delta,
        )
        state.validate(
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return state, {}
        completed_policy_trajectory = protected_detail + dynamic_delta
        metrics = {
            "p1_protected_detail_rms": protected_detail.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_dynamic_delta_rms": dynamic_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            # Retain the historical scalar name for longitudinal tooling. The
            # represented sum is no longer exported under a factual identity.
            "p1_completed_fact_rms": completed_policy_trajectory.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_factual_base_rms": state.factual_base.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_policy_query_residual_rms": state.policy_query_residual.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "p1_policy_content_mod_rms": content_delta.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
        }
        for source, target in (
            ("residual_self_written_rms", "p1_policy_self_written_rms"),
            ("residual_ffn_written_rms", "p1_policy_ffn_written_rms"),
        ):
            value = block_metrics.get(source)
            if value is not None:
                metrics[target] = value
        return state, metrics


class PolicyCompilerStage(nn.Module):
    """Dynamic P2 effect/consequence and P3 execution-plan compilation."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
        action_dim: int,
        effect_reader: ObjectFutureEffectReader,
        consequence: ZeroPreservingObjectConsequence,
        plan_compiler: ObjectPolicyPlanCompiler,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.action_dim = int(action_dim)
        self.effect_reader = effect_reader
        self.consequence = consequence
        self.plan_compiler = plan_compiler

    def compile(
        self,
        context: DeploymentTopCache,
        *,
        p1_state: CompletedP1PolicyState,
        action_query: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[CompiledPolicyState, dict[str, Tensor]]:
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
        # Legacy P3 expects the summed post-consequence trajectory. Typed P3
        # instead receives the original action query and all named owners,
        # so factual/effect content is not counted twice on its value ingress.
        p3_action_query = (
            action_query if self.plan_compiler.coordinator is not None
            else action_query + consequence.protected_consequence
        )
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


class TrainingTargetsStage(nn.Module):
    """Teacher-G and recognizer targets; inaccessible from deployment APIs."""

    def __init__(
        self,
        *,
        teacher: ObjectFutureTeacher,
        recognizer: FuturePlanRecognizer,
        hidden: int,
        horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.recognizer = recognizer
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)

    def build(
        self,
        context: OnlineTopContext,
        *,
        future_supports: Tensor,
        future_offsets: Tensor,
        future_action: Tensor,
        future_state: Tensor,
        coarse_action: CoarseActionIntent,
        label_support: FutureLabelSupport | None = None,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectTopTrainingTargets, dict[str, Tensor]]:
        context.validate(hidden=self.hidden, horizon=self.horizon)
        context.action_condition.validate(action_dim=self.action_dim)
        teacher, teacher_metrics = self.teacher(
            facts=context.facts,
            future_supports=future_supports,
            future_offsets=future_offsets,
            **({"future_observed": label_support.visual} if label_support is not None else {}),
            collect_diagnostics=collect_diagnostics,
        )
        if context.intent.time_grid_mode != self.teacher.time_grid.mode or self.recognizer.time_grid.mode != self.teacher.time_grid.mode:
            raise ValueError("training S/Teacher/recognizer time grid mismatch")
        interval_valid = (
            None
            if label_support is None
            else self.teacher.time_grid.interval_observed(future_offsets, label_support.visual)
        )
        current_loss_support = intersect_object_camera_validity(
            context.facts.validity,
            context.facts.camera_validity,
        )
        recognition = self.recognizer(
            future_action=future_action,
            future_state=future_state,
            teacher=teacher,
            current_loss_support=current_loss_support,
            **(
                {"label_support": label_support, "future_interval_valid": interval_valid}
                if label_support is not None
                else {}
            ),
        )
        online_intent_loss = supported_mean(
            F.smooth_l1_loss(
                context.intent.public_interval_carrier.float(),
                recognition.interval_targets.detach().float(),
                reduction="none",
            ),
            recognition.interval_valid,
        )
        supervised_coarse = coarse_action(
            context.intent.action_dock(),
            future_action=future_action[:, : self.horizon],
            **(
                {"future_action_valid": label_support.action[:, : self.horizon]}
                if label_support is not None
                else {}
            ),
            collect_diagnostics=collect_diagnostics,
        )
        coarse_loss = supervised_coarse.loss
        targets = ObjectTopTrainingTargets(
            teacher_dynamics=teacher,
            current_loss_support=current_loss_support,
            plan_recognition=recognition,
            online_intent_loss=online_intent_loss,
            plan_recognition_loss=recognition.reconstruction_loss,
            coarse_action_loss=coarse_loss,
            history_proposal_loss=coarse_loss.new_zeros(()),
            object_reconstruction_loss=context.facts.reconstruction_error,
            future_interval_valid=interval_valid,
        )
        if not collect_diagnostics:
            return targets, {}
        return targets, {
            **teacher_metrics,
            "object_intent_online_match_loss": online_intent_loss.detach(),
            "object_plan_recognition_loss": recognition.reconstruction_loss.detach(),
            "object_coarse_action_loss": coarse_loss.detach(),
        }


class ExecutionBottomStage(nn.Module):
    """Typed execution bottom and terminal-controller-owned decoder."""

    def __init__(
        self,
        *,
        hidden: int,
        horizon: int,
        basis: int,
        physical_action_dim: int,
        core_config: Any,
        layer_contract_heads: nn.ModuleList,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.physical_action_dim = int(physical_action_dim)
        self.core_config = core_config
        self.layer_contract_heads = layer_contract_heads
        self.decoder = decoder

    @property
    def blocks(self) -> nn.ModuleList:
        return self.decoder.blocks

    @property
    def capacity(self) -> nn.ModuleList:
        return self.decoder.operator_contractions

    @property
    def execution(self) -> nn.Module | None:
        return self.decoder.execution_controller

    def set_training_step(self, global_step: int) -> float:
        return self.decoder.set_execution_training_step(global_step)

    def _state_memory(self, seed: V120SeedContext) -> tuple[Tensor, Tensor, Tensor]:
        seed.validate(
            hidden=self.hidden,
            state_history=int(self.core_config.visual_history_length),
            executed=int(self.core_config.action_history_token_count),
        )
        # Match the active V120 object path: current state plus the final
        # causal state-history row, and only the final compressed execution
        # row in the compact intent bank.
        return seed.state, seed.state_history[:, -1:], seed.executed[:, -1:]

    def _neutral_trajectory_memory(self, plan: ObjectPolicyPlanDeltaBank) -> Tensor:
        """Restore V120's neutral generic proposal ingress."""

        plan.validate()
        return plan.protected_base.new_zeros(
            int(plan.protected_base.shape[0]),
            self.horizon,
            self.hidden,
        )

    def _transition_event_context(self, transition: ControlledTransitionState) -> Tensor:
        """Apply V120's spatial-anchor pooling to the centered transition."""

        transition.validate(hidden=self.hidden)
        batch, rows, hidden = transition.value.shape
        grid = (
            int(self.core_config.num_cameras)
            * int(self.core_config.future_grid_size) ** 2
        )
        anchors = int(self.core_config.future_anchors)
        if rows != anchors * grid:
            raise ValueError("event context requires the complete V120 transition chart")
        milestones = transition.value.reshape(batch, anchors, grid, hidden).mean(dim=2)
        boundaries = tuple(
            int(value) for value in self.core_config.flow_jepa_action_offsets
        )
        milestones = milestones[:, : len(boundaries)]
        rows_out: list[Tensor] = []
        lower = 0
        for index, upper in enumerate(boundaries):
            if upper <= lower or upper > self.horizon:
                raise ValueError("V120 event milestone boundaries are invalid")
            rows_out.append(
                milestones[:, index : index + 1].expand(-1, upper - lower, -1)
            )
            lower = upper
        if lower != self.horizon:
            raise ValueError("V120 event milestones do not cover the action horizon")
        return torch.cat(rows_out, dim=1)

    def _role_bank(self, plan: ObjectPolicyPlanDeltaBank) -> PolicyRoleDeltaBank:
        return plan.as_policy_role_bank(source_depth=7)

    def _layer_contract_canvas(
        self,
        *,
        rollout: Tensor,
        seed: V120SeedContext,
    ) -> tuple[Tensor, dict[str, slice]]:
        """Build the live rows read by the position-wise terminal contract."""

        batch = int(rollout.shape[0])
        if tuple(rollout.shape) != (
            batch,
            int(self.core_config.future_token_count),
            self.hidden,
        ):
            raise ValueError("V120 layer-contract rollout has invalid shape")
        empty = rollout[:, :0]
        parts = (
            ("state", seed.state),
            ("state_history", seed.state_history),
            ("executed", seed.executed),
            ("proposal", empty),
            ("trajectory", empty),
            ("rollout", rollout),
            ("registers", empty),
        )
        slices: dict[str, slice] = {}
        offset = 0
        for name, value in parts:
            slices[name] = slice(offset, offset + int(value.shape[1]))
            offset += int(value.shape[1])
        return torch.cat([value for _, value in parts], dim=1), slices

    def _layer_contracts(
        self,
        *,
        rollout: Tensor,
        seed: V120SeedContext,
    ) -> list[dict[str, Tensor]]:
        contracts: list[dict[str, Tensor]] = []
        for head in self.layer_contract_heads:
            canvas, slices = self._layer_contract_canvas(
                rollout=rollout,
                seed=seed,
            )
            contracts.append(head(canvas, slices))
        return contracts

    @staticmethod
    def _intent_memory(
        intent: ObjectIntentState,
        state_tokens: Tensor,
        executed_tokens: Tensor,
    ) -> dict[str, Tensor]:
        del intent
        return {
            "state": state_tokens,
            "executed": executed_tokens,
        }

    def _set_eval_intervention(self, mode: str) -> None:
        if mode == "learned":
            self.decoder.clear_execution_eval_ablation()
            return
        if self.training:
            raise ValueError("V120 execution interventions are evaluation-only")
        if mode == "spine_zero":
            if getattr(self.decoder, "spine", None) is None:
                raise ValueError(
                    "spine_zero requires the selected Schema31 B-spine execution bottom"
                )
            self.decoder.clear_execution_eval_ablation()
            return
        if mode == "no_updates":
            self.decoder.set_execution_eval_ablation(
                policy="neutral",
                capacity_gate=1.0,
            )
            return
        if mode == "hard":
            self.decoder.set_execution_eval_ablation(
                policy="hard",
                capacity_gate=None,
            )
            return
        if mode == "neutral":
            self.decoder.set_execution_eval_ablation(
                policy="neutral",
                capacity_gate=1.0,
            )
            return
        if mode == "full_capacity":
            self.decoder.set_execution_eval_ablation(
                policy="soft",
                capacity_gate=1.0,
            )
            return
        if mode == "three_basis_reduction":
            rank = max(
                int(self.core_config.latent_cvae_mmdit_operator_rank),
                1,
            )
            self.decoder.set_execution_eval_ablation(
                policy="soft",
                capacity_gate=max(float(rank - 3), 1.0) / float(rank),
            )
            return
        raise ValueError(
            "bottom execution_mode must be learned/no_updates/hard/neutral/"
            "full_capacity/three_basis_reduction/spine_zero"
        )

    def compile_evidence_view(
        self,
        *,
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
    ):
        state_tokens, state_history_tokens, executed_tokens = self._state_memory(seed)
        trajectory = self._neutral_trajectory_memory(plan)
        event_context = self._transition_event_context(transition)
        layer_contracts = self._layer_contracts(
            rollout=transition.selector,
            seed=seed,
        )
        return self.decoder.evidence_adapter(
            trajectory_tokens=trajectory,
            rollout_tokens=transition.selector,
            transition_memory=[transition.value, event_context],
            event_evidence=layer_contracts[-1]["event_logits"],
            state_memory=[state_tokens, state_history_tokens],
            layer_contracts=layer_contracts,
            intent_memory=self._intent_memory(intent, state_tokens, executed_tokens),
            visual_selector_tokens=None,
            visual_value_tokens=None,
            visual_key_bias=None,
        )

    def step(
        self,
        *,
        noisy_action_field: Tensor,
        time: Tensor,
        flow_step_context: FlowStepContext | None = None,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        intent: ObjectIntentState,
        seed: V120SeedContext,
        transition: ControlledTransitionState,
        execution_mode: str = "learned",
        deployment_fastpath: bool = False,
        require_execution_supervision: bool = False,
        collect_diagnostics: bool = False,
    ) -> tuple[BottomDecoderOutput, dict[str, Tensor]]:
        expected_query = (
            int(noisy_action_field.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected_query:
            raise ValueError("bottom and P2/P3 must share one action query")
        if flow_step_context is not None:
            flow_step_context.validate(
                batch=int(noisy_action_field.shape[0]),
                device=noisy_action_field.device,
                time=time,
                # The scheduler validates numerical bounds once at the host
                # boundary.  This node-level adapter only needs the cheap
                # shape/device seam; strict tensor reductions here would add
                # a host synchronisation to every ODE update.
                strict=False,
            )
        plan.validate()
        intent.validate(horizon=self.horizon, hidden=self.hidden)
        transition.validate(hidden=self.hidden)
        state_tokens, state_history_tokens, executed_tokens = self._state_memory(seed)
        role_bank = self._role_bank(plan)
        role_bank.validate(hidden_size=self.hidden, horizon=self.horizon)
        trajectory = self._neutral_trajectory_memory(plan)
        event_context = self._transition_event_context(transition)
        layer_contracts = self._layer_contracts(
            rollout=transition.selector,
            seed=seed,
        )
        event_evidence = layer_contracts[-1]["event_logits"]
        run_diagnostics = bool(
            collect_diagnostics
            or self.training
            or require_execution_supervision
            or execution_mode != "learned"
        )

        self._set_eval_intervention(execution_mode)
        try:
            raw = self.decoder(
                noisy_physical=noisy_action_field,
                time=time,
                flow_step_context=flow_step_context,
                trajectory_tokens=trajectory,
                trajectory_workspace_tokens=trajectory,
                policy_action_tokens=None,
                policy_role_delta_bank=role_bank,
                execution_terminal_probability=None,
                execution_terminal_uncertainty=None,
                rollout_tokens=transition.selector,
                transition_memory=[transition.value, event_context],
                event_evidence=event_evidence,
                state_memory=[state_tokens, state_history_tokens],
                layer_contracts=layer_contracts,
                intent_memory=self._intent_memory(
                    intent,
                    state_tokens,
                    executed_tokens,
                ),
                # P1 already owns the only high-resolution read.
                visual_selector_tokens=None,
                visual_value_tokens=None,
                visual_key_bias=None,
                collect_diagnostics=run_diagnostics,
                # Execution-value supervision needs the decoder's candidate
                # tensors on every training batch, but the R2 gripper state/
                # VJP observation is an outer logging concern.  Keep that
                # output surface on the existing bounded cadence.
                collect_gripper_diagnostics=collect_diagnostics,
                evidence_scale=1.0,
                noisy_scale=1.0,
                spine_zero=execution_mode == "spine_zero",
                deployment_fastpath=deployment_fastpath,
            )
        finally:
            if execution_mode != "learned":
                self.decoder.clear_execution_eval_ablation()

        prefix = raw.get("evidence_mmd_it_prefix_pred_velocity")
        block_updates: tuple[Tensor, ...] = ()
        if isinstance(prefix, Tensor) and prefix.ndim == 5:
            # Defensive compatibility for older candidate charts.
            prefix = prefix.mean(dim=2)
        if isinstance(prefix, Tensor) and prefix.ndim == 4:
            block_updates = tuple(
                prefix[:, index + 1] - prefix[:, index]
                for index in range(int(prefix.shape[1]) - 1)
            )
        tensor_output = {
            name: value for name, value in raw.items() if isinstance(value, Tensor)
        }
        physical_velocity = raw["pred_velocity"]
        if execution_mode == "no_updates":
            if not isinstance(prefix, Tensor) or prefix.ndim != 4:
                raise RuntimeError(
                    "true no-update ablation requires the V120 prefix velocity chart"
                )
            # Prefix row zero is the organized/noisy-action prediction before
            # any host Evidence-MMDiT block executes.  Capacity=0 is not a
            # no-op in V120 (it removes only the owned low-rank subspace), so
            # selecting this row is the only behaviorally exact ablation.
            physical_velocity = prefix[:, 0]
        if "event_logits" in raw:
            raise RuntimeError("active bottom cannot expose a hidden-state event bypass")
        output = BottomDecoderOutput(
            physical_velocity=physical_velocity,
            motion_logits=raw["motion_logits"],
            action_query=action_query,
            block_updates=block_updates,
            evidence_tokens=transition.value,
            decoder_tensors=tensor_output,
            gripper_command_logits=raw.get("gripper_command_logits"),
        )
        output.validate(
            action_dim=self.physical_action_dim,
            horizon=self.horizon,
            basis=self.basis,
            hidden=self.hidden,
        )
        if not collect_diagnostics:
            return output, {}
        metrics = {
            name: value
            for name, value in tensor_output.items()
            if value.ndim == 0
        }
        if "evidence_mmd_it_capacity_ratio" in tensor_output:
            metrics["bottom_capacity_mean"] = tensor_output[
                "evidence_mmd_it_capacity_ratio"
            ]
        if "evidence_mmd_it_dwell_expected" in tensor_output:
            metrics["bottom_expected_dwell"] = tensor_output[
                "evidence_mmd_it_dwell_expected"
            ]
        metrics["bottom_restored_v120_decoder"] = noisy_action_field.new_ones(
            (), dtype=torch.float32
        )
        metrics["bottom_retained_transition_rows"] = noisy_action_field.new_tensor(
            float(transition.value.shape[1]), dtype=torch.float32
        )
        metrics["bottom_rollout_selector_only"] = noisy_action_field.new_ones(
            (), dtype=torch.float32
        )
        metrics["bottom_protected_consequence_value_writes"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_generic_trajectory_neutral"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_event_from_terminal_layer_contract"] = (
            noisy_action_field.new_ones((), dtype=torch.float32)
        )
        metrics["bottom_terminal_layer_contract_count"] = (
            noisy_action_field.new_tensor(
                float(len(layer_contracts)), dtype=torch.float32
            )
        )
        metrics["bottom_execution_output_block_count"] = noisy_action_field.new_tensor(
            0.0 if execution_mode == "no_updates" else float(len(self.blocks)),
            dtype=torch.float32,
        )
        return output, metrics


class OutletAdapter(nn.Module):
    """Selected outlet's physical field codec and native action boundary."""

    def __init__(
        self,
        codec: PhysicalActionFieldCodec,
        *,
        selection: str,
        world_action_condition_mode: str = "interval_mean_v1",
        future_time_grid_mode: str = LEGACY_FUTURE_TIME,
    ) -> None:
        super().__init__()
        self.time_grid = resolve_future_time(future_time_grid_mode)
        self.codec = codec
        self.selection = str(selection)
        self.world_action_condition_mode = str(world_action_condition_mode)
        if self.world_action_condition_mode not in {
            "interval_mean_v1",
            "sequence_prefix_v1",
        }:
            raise ValueError("unknown outlet W action condition mode")
        # Dataset normalization is an ordinary Python sidecar.  Keeping these
        # values out of parameters and buffers preserves model/state/RNG
        # identity for Pen and RDT while allowing relative/binary outlet
        # semantics to be converted at the native action boundary.
        self._action_normalizer_offset: tuple[float, ...] | None = None
        self._action_normalizer_scale: tuple[float, ...] | None = None
        self._action_normalizer_fingerprint: str | None = None

    @property
    def action_dim(self) -> int:
        return self.codec.action_dim

    @property
    def horizon(self) -> int:
        return self.codec.horizon

    @property
    def gripper_field_dim(self) -> int:
        return self.codec.gripper_field_dim

    @property
    def physical_dim(self) -> int:
        return self.codec.physical_dim

    @property
    def arm_dim(self) -> int:
        return self.codec.arm_dim

    @property
    def decode_delta_blend(self) -> float:
        return self.codec.decode_delta_blend

    @property
    def arm_flow_mode(self) -> str:
        return "relative_command_adapter" if self.is_relative_command else "legacy_independent"

    @property
    def is_relative_command(self) -> bool:
        """Whether the outlet's six arm channels are relative TCP commands.

        This is intentionally independent from the gripper controller: CALVIN
        and ManiSkill can combine relative arms with a binary command head,
        while LIBERO and legacy ManiSkill combine the same arm adapter with the
        ordinary continuous gripper field.  Keeping the predicates separate
        prevents one outlet's gripper policy from silently selecting another
        outlet's arm chart.
        """

        return self.selection in {
            "calvin_7d_binary_v1",
            "libero_7d_continuous_v1",
            "maniskill_7d_continuous_v1",
            "maniskill_7d_continuous_v2",
            "maniskill_7d_binary_v2",
        }

    @property
    def is_binary_command(self) -> bool:
        return is_binary_gripper_selection(self.selection)

    def configure_action_normalizer(self, normalizer: Any) -> None:
        """Attach the verified affine action chart without model state.

        Relative-command outlets need the normalized representation of native
        zero to remove a z-score offset from TCP commands before W sees their
        cumulative displacement.  Binary outlets additionally use the affine
        chart to express their discrete gripper command in W coordinates.
        Absolute Pen/RDT outlets never read this sidecar.
        """

        if (
            not self.is_relative_command
            and self.world_action_condition_mode == "interval_mean_v1"
        ):
            # Pen/RDT never read or retain a relative-command sidecar.
            return
        try:
            offset = torch.as_tensor(normalizer.offset, dtype=torch.float32).reshape(-1)
            scale = torch.as_tensor(normalizer.scale, dtype=torch.float32).reshape(-1)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("action normalizer must expose finite offset and scale") from error
        if int(offset.numel()) != self.action_dim or int(scale.numel()) != self.action_dim:
            raise ValueError("action normalizer width does not match the outlet action chart")
        if not bool(torch.isfinite(offset).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("action normalizer offset/scale must be finite")
        if bool((scale <= 0.0).any()):
            raise ValueError("action normalizer scale must be strictly positive")
        self._action_normalizer_offset = tuple(float(value) for value in offset.tolist())
        self._action_normalizer_scale = tuple(float(value) for value in scale.tolist())
        self._action_normalizer_fingerprint = (
            physical_action_normalizer_fingerprint(offset, scale)
        )

    def _normalizer_chart(
        self,
        reference: Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self._action_normalizer_offset is None or self._action_normalizer_scale is None:
            raise RuntimeError(
                "relative-command outlet adapter requires the checkpoint/data action normalizer"
            )
        chart_dtype = reference.dtype if dtype is None else dtype
        offset = reference.new_tensor(
            self._action_normalizer_offset, dtype=chart_dtype
        ).reshape(1, 1, -1)
        scale = reference.new_tensor(
            self._action_normalizer_scale, dtype=chart_dtype
        ).reshape(1, 1, -1)
        return offset, scale

    def _normalizer_identity(self) -> str:
        if self._action_normalizer_fingerprint is None:
            raise RuntimeError(
                "sequence action condition requires the checkpoint/data action normalizer"
            )
        return self._action_normalizer_fingerprint

    def validate_world_condition(
        self,
        condition: WorldActionCondition,
    ) -> None:
        """Bind a cached W condition to this configured outlet unconditionally."""

        if self.world_action_condition_mode == "interval_mean_v1":
            if not isinstance(condition, PhysicalActionCondition):
                raise TypeError(
                    "interval-mean outlet rejected a sequence action condition"
                )
            condition.validate(action_dim=self.action_dim)
            return
        if not isinstance(condition, PhysicalActionSequenceCondition):
            raise TypeError(
                "sequence-prefix outlet rejected an interval action condition"
            )
        condition.validate(action_dim=self.action_dim)
        expected_chart = (
            RELATIVE_COMMAND_SEQUENCE_CHART
            if self.is_relative_command
            else ABSOLUTE_ACTION_SEQUENCE_CHART
        )
        if condition.outlet_profile != self.selection:
            raise ValueError("sequence action condition outlet profile changed")
        if condition.chart_version != expected_chart:
            raise ValueError("sequence action condition chart version changed")
        if condition.arm_dim != self.arm_dim:
            raise ValueError("sequence action condition arm width changed")
        if condition.normalizer_fingerprint != self._normalizer_identity():
            raise ValueError("sequence action condition normalizer identity changed")

    def world_condition_from_interval_action(
        self,
        interval_action: Tensor,
        current_action: Tensor,
    ) -> PhysicalActionCondition:
        """Convert an outlet-native four-row proposal to W's physical chart."""

        if self.world_action_condition_mode != "interval_mean_v1":
            raise RuntimeError(
                "four-row world condition is unavailable in sequence-prefix mode"
            )

        if not self.is_relative_command:
            # This exact factory is the established Pen/RDT path.
            return PhysicalActionCondition.from_interval_action(
                interval_action,
                current_action,
            )
        if interval_action.ndim != 3 or tuple(interval_action.shape[1:]) != (
            4,
            self.action_dim,
        ):
            raise ValueError("relative-command W proposal must be [B,4,7]")
        batch = int(interval_action.shape[0])
        if tuple(current_action.shape) != (batch, self.action_dim):
            raise ValueError("relative-command W current action must be [B,7]")
        offset, _scale = self._normalizer_chart(interval_action)
        source_current = current_action.to(
            device=interval_action.device,
            dtype=interval_action.dtype,
        )
        # Relative arm rows are interval TCP commands.  Remove the affine
        # chart's raw-zero offset, then expose cumulative displacement as W
        # value and the centered command itself as W physical delta.
        arm_command = interval_action[..., : self.arm_dim] - offset[..., : self.arm_dim]
        canonical_arm = torch.cumsum(arm_command, dim=1)
        gripper = interval_action[..., self.arm_dim :]
        gripper_boundary = torch.cat(
            (source_current[:, None, self.arm_dim :], gripper[:, :-1]),
            dim=1,
        )
        canonical_interval = torch.cat((canonical_arm, gripper), dim=-1)
        canonical_delta = torch.cat(
            (arm_command, gripper - gripper_boundary),
            dim=-1,
        )
        canonical_current = torch.cat(
            (
                torch.zeros_like(source_current[..., : self.arm_dim]),
                source_current[..., self.arm_dim :],
            ),
            dim=-1,
        )
        condition = OutletPhysicalActionCondition(
            interval_action=canonical_interval,
            interval_delta=canonical_delta,
            current_action=canonical_current,
            source_action=interval_action,
        )
        condition.validate(action_dim=self.action_dim)
        return condition

    def world_condition_from_action_prediction(
        self,
        action_prediction: Tensor,
        current_action: Tensor,
    ) -> WorldActionCondition:
        """Route the online coarse proposal through the configured W ABI."""

        if self.world_action_condition_mode == "interval_mean_v1":
            return self.world_condition_from_interval_action(
                action_prediction,
                current_action,
            )
        return self._sequence_condition_from_horizon_action(
            action_prediction,
            current_action,
        )

    def _sequence_condition_from_horizon_action(
        self,
        action: Tensor,
        current_action: Tensor,
    ) -> PhysicalActionSequenceCondition:
        """Build one complete 24-row condition with outlet-owned semantics."""

        if self.world_action_condition_mode != "sequence_prefix_v1":
            raise RuntimeError("sequence condition requested from interval-mean outlet")
        if action.ndim != 3 or tuple(action.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("sequence-prefix W proposal must be [B,24,A]")
        batch = int(action.shape[0])
        if tuple(current_action.shape) != (batch, self.action_dim):
            raise ValueError("sequence-prefix W current action does not align")
        normalizer_identity = self._normalizer_identity()
        # Identity describes the checkpoint chart, independently of AMP's
        # action dtype. The condition factory casts only the arithmetic view.
        offset, scale = self._normalizer_chart(action, dtype=torch.float32)
        if not self.is_relative_command:
            if self.selection not in {
                "pen_7d_continuous_v1",
                "rdt_right_arm_7d_v1",
            }:
                raise ValueError(
                    "sequence-prefix W has no absolute chart for this outlet"
                )
            condition = PhysicalActionSequenceCondition.from_absolute_action(
                action,
                current_action,
                outlet_profile=self.selection,
                normalizer_offset=offset,
                normalizer_scale=scale,
                normalizer_fingerprint=normalizer_identity,
            )
        else:
            condition = PhysicalActionSequenceCondition.from_relative_command(
                action,
                current_action,
                outlet_profile=self.selection,
                arm_dim=self.arm_dim,
                normalizer_offset=offset,
                normalizer_scale=scale,
                normalizer_fingerprint=normalizer_identity,
            )
        if condition.normalizer_fingerprint != normalizer_identity:
            raise RuntimeError("sequence action normalizer identity changed in factory")
        self.validate_world_condition(condition)
        return condition

    @torch.no_grad()
    def observed_world_condition(
        self, action: Tensor, current_action: Tensor, observed: Tensor | None = None,
    ) -> ObservedActionSequenceCondition:
        """Canonicalize labelled controls using the SAME outlet's 24-row chart.

        Compose complete chart blocks with the previous physical boundary.
        Joining relative-command blocks also carries the cumulative arm value;
        it does not restart the displacement at row 24. This is supervision,
        not an extension of the 24-row deployment proposal contract.
        """
        if self.world_action_condition_mode != "sequence_prefix_v1":
            raise ValueError("observed W supervision requires sequence-prefix mode")
        horizon = self.time_grid.horizon
        if action.ndim != 3 or action.shape[1] < horizon or action.shape[2] != self.action_dim:
            raise ValueError("observed W labels must cover the full world horizon")
        if observed is None:
            observed = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        if observed.dtype != torch.bool or observed.device != action.device or tuple(observed.shape) != tuple(action.shape[:2]):
            raise ValueError("observed W control support must be bool [B,Tw]")
        mask = observed[:, :horizon].detach().clone()
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("observed W controls require a causal prefix without holes")
        labels = action[:, :horizon].detach()
        if not bool(torch.isfinite(labels[mask]).all()) or not bool(torch.isfinite(current_action).all()):
            raise ValueError("observed W source controls/boundary must be finite")
        offset, _ = self._normalizer_chart(labels, dtype=torch.float32)
        # Native-zero padding is only an arithmetic placeholder. It is never
        # declared executed and cannot advance the masked W recurrence.
        safe = torch.where(mask[..., None], labels, offset.to(dtype=labels.dtype)).clone()
        values: list[Tensor] = []
        deltas: list[Tensor] = []
        identity: PhysicalActionSequenceCondition | None = None
        for start in range(0, horizon, self.horizon):
            chunk = safe[:, start:start + self.horizon]
            if chunk.shape[1] != self.horizon:
                raise ValueError("world horizon must compose full outlet chart blocks")
            boundary = current_action.detach() if start == 0 else safe[:, start - 1]
            identity = self._sequence_condition_from_horizon_action(chunk, boundary)
            value = identity.canonical_value
            if start and self.is_relative_command:
                value = torch.cat((
                    value[..., :self.arm_dim] + values[-1][:, -1:, :self.arm_dim],
                    value[..., self.arm_dim:],
                ), dim=-1)
            values.append(value)
            deltas.append(identity.canonical_delta)
        assert identity is not None
        keep = mask[..., None]
        condition = ObservedActionSequenceCondition(
            time_grid_mode=self.time_grid.mode,
            source_action=torch.where(keep, safe, torch.zeros_like(safe)),
            canonical_value=torch.where(keep, torch.cat(values, dim=1), torch.zeros_like(safe)),
            canonical_delta=torch.where(keep, torch.cat(deltas, dim=1), torch.zeros_like(safe)),
            observed=mask,
            row_end=torch.arange(1, horizon + 1, device=action.device, dtype=action.dtype)[None, :, None].expand(action.shape[0], -1, -1),
            outlet_profile=identity.outlet_profile, chart_version=identity.chart_version,
            normalizer_fingerprint=identity.normalizer_fingerprint,
        )
        condition.validate(action_dim=self.action_dim)
        return condition

    def world_condition_action_from_deployed(
        self,
        deployed_action: Tensor,
        *,
        command: Tensor | None,
    ) -> Tensor:
        """Return the normalized action view used only for a W rebuild."""

        if not self.is_binary_command:
            if command is not None:
                raise ValueError("continuous outlet cannot carry a binary command")
            return deployed_action
        if command is None:
            raise RuntimeError(
                "binary gripper W rebuild requires the deployed command tensor"
            )
        if tuple(command.shape) != tuple(deployed_action.shape[:2]):
            raise ValueError("binary gripper command does not align with deployed action")
        offset, scale = self._normalizer_chart(deployed_action)
        result = deployed_action.clone()
        result[..., -1] = (
            command.to(device=result.device, dtype=result.dtype) * scale[..., -1]
            + offset[..., -1]
        )
        return result

    def world_condition_from_horizon_action(
        self,
        action: Tensor,
        current_action: Tensor,
    ) -> WorldActionCondition:
        """Build the configured W condition from one deployed 24-row action."""

        if self.world_action_condition_mode == "sequence_prefix_v1":
            return self._sequence_condition_from_horizon_action(
                action,
                current_action,
            )

        source = PhysicalActionCondition.from_horizon_action(
            action,
            current_action,
        )
        if not self.is_relative_command:
            return source
        return self.world_condition_from_interval_action(
            source.interval_action,
            current_action,
        )

    def prepare_model_input(self, field: Tensor) -> Tensor:
        """Apply the selected outlet's dynamic-consumer input boundary."""

        if self.is_binary_command:
            return self.codec.binary_command_model_input(field)
        return field

    def conditioning_metrics(
        self,
        original_field: Tensor,
        model_field: Tensor,
        *,
        collect_diagnostics: bool,
    ) -> dict[str, Tensor]:
        if not self.is_binary_command or not collect_diagnostics:
            return {}
        arm_channels = 2 * int(self.arm_dim)
        removed_rms = (
            original_field[..., arm_channels:].detach().float().square().mean().sqrt()
        )
        max_abs = model_field[..., arm_channels:].detach().float().abs().max()
        neutral = original_field.new_ones((), dtype=torch.float32)
        metrics = {
            "bottom_binary_gripper_condition_removed_rms": removed_rms,
            "bottom_binary_gripper_condition_max_abs": max_abs,
            "bottom_binary_gripper_condition_neutral": neutral,
        }
        if self.selection == "calvin_7d_binary_v1":
            # Preserve the established CALVIN metric ABI for old dashboards.
            metrics.update(
                {
                    "bottom_calvin_binary_gripper_condition_removed_rms": removed_rms,
                    "bottom_calvin_binary_gripper_condition_max_abs": max_abs,
                    "bottom_calvin_binary_gripper_condition_neutral": neutral,
                }
            )
        elif self.selection == "maniskill_7d_binary_v2":
            metrics.update(
                {
                    "bottom_maniskill_binary_gripper_condition_removed_rms": removed_rms,
                    "bottom_maniskill_binary_gripper_condition_max_abs": max_abs,
                    "bottom_maniskill_binary_gripper_condition_neutral": neutral,
                }
            )
        return metrics

    def finalize(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None,
        command_logits: Tensor | None,
    ) -> OutletActionOutput:
        """Produce the selected outlet's deployed native action."""

        continuous_action = self.decode(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        ).float()
        if self.is_binary_command:
            if command_logits is None:
                raise RuntimeError(
                    "binary gripper deployment requires gripper command logits from the bottom"
                )
            command = binary_gripper_command_from_logits(command_logits.float())
            if tuple(command.shape) != tuple(continuous_action.shape[:2]):
                raise ValueError(
                    "binary gripper command shape does not match decoded action"
                )
            action = continuous_action.clone()
            action[..., -1] = command.to(dtype=action.dtype)
            world_action = self.world_condition_action_from_deployed(
                action,
                command=command,
            )
            return OutletActionOutput(
                deployed_action=action,
                world_condition_action=world_action,
                continuous_action=continuous_action,
                command_logits=command_logits.float(),
                command=command,
            )
        if command_logits is not None:
            raise RuntimeError(
                "continuous deployment unexpectedly exposed a gripper command head"
            )
        return OutletActionOutput(
            deployed_action=continuous_action,
            world_condition_action=continuous_action,
            continuous_action=None,
            command_logits=None,
            command=None,
        )

    def sampling_metrics(self, output: OutletActionOutput) -> dict[str, Tensor]:
        action = output.deployed_action
        metrics = {
            "sampling_gripper_output_mode_code": action.new_tensor(
                1.0 if self.is_binary_command else 0.0,
                dtype=torch.float32,
            )
        }
        if output.command is not None:
            command = output.command
            metrics.update(
                {
                    "sampling_gripper_command_positive_rate": (
                        (command > 0).float().mean()
                    ),
                    "sampling_gripper_command_switch_rate": (
                        (command[:, 1:] != command[:, :-1]).float().mean()
                        if int(command.shape[1]) > 1
                        else command.new_zeros(())
                    ),
                }
            )
        return metrics

    def encode(
        self,
        action: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        return self.codec.encode(
            action,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def decode(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        return self.codec.decode(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def sample_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.codec.sample_noise(
            batch,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def binary_command_model_input(self, field: Tensor) -> Tensor:
        return self.codec.binary_command_model_input(field)

    def split(self, field: Tensor) -> PhysicalActionFieldParts:
        return self.codec.split(field)

    def gripper_decode_branches(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.codec.gripper_decode_branches(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def delta_consistency(
        self,
        field: Tensor,
        action_state: Tensor,
        decoded_action: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        return self.codec.delta_consistency(
            field,
            action_state,
            decoded_action,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor]:
        return self.codec.project_arm_tangent(arm_field)

    def arm_motion_magnitude(self, action: Tensor, action_state: Tensor) -> Tensor:
        if self.is_relative_command:
            if action.ndim != 3 or int(action.shape[-1]) != self.action_dim:
                raise ValueError("relative-command motion action must be [B,T,7]")
            offset, _scale = self._normalizer_chart(action)
            return (
                action[..., : self.arm_dim] - offset[..., : self.arm_dim]
            ).float().norm(dim=-1)
        return self.codec.arm_motion_magnitude(action, action_state)


__all__ = [
    "BridgeStage",
    "ConditioningStage",
    "ExecutionBottomStage",
    "GroundingStage",
    "IntentStage",
    "ObservationStage",
    "OutletAdapter",
    "P1Stage",
    "PolicyCompilerStage",
    "TrainingTargetsStage",
    "WorldStage",
]
