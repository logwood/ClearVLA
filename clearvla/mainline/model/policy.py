"""End-to-end online policy with separate training supervision plane."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import FutureSupervision, ObservableHistory, OnlinePolicyInput
from .action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from .action_contract import BottomOutput
from .observation_contract import ObservationEvidence
from .proposal import HistoryActionProposal
from .restored_bottom import RestoredV120EvidenceBottom
from .restored_observation import RestoredV120ObservationCompiler
from .routing import register_gradient_rms_metric
from .top import (
    CompiledPolicyState,
    DeploymentTopCache,
    ObjectIntentDynamicsTop,
    OnlineTopContext,
)
from .transition import ControlledTransitionDynamics
from .types import (
    ControlledTransitionSource,
    FactualPrecisionDock,
    HistoryActionProposalState,
    ObjectTopTrainingTargets,
)
from .v120_p1 import LateRawDetailPolicyReader


@dataclass(frozen=True)
class OnlinePolicyCache:
    """Current-only state reused by every deployment ODE step."""

    top: DeploymentTopCache
    factual_dock: FactualPrecisionDock
    transition_source: ControlledTransitionSource
    history: ObservableHistory
    executed_memory: Tensor
    action_history_keep: Tensor
    role_table: Tensor

    def validate(self, config: ExperimentConfig) -> None:
        self.history.validate(config)
        self.top.validate(
            hidden=config.dimensions.hidden_size,
            horizon=config.dimensions.action_horizon,
        )
        self.factual_dock.validate(
            horizon=config.dimensions.action_horizon,
            basis=config.dimensions.action_basis_tokens,
        )
        self.transition_source.validate(hidden=config.dimensions.hidden_size)
        expected_memory = (
            int(self.history.state.shape[0]),
            config.top.proposal_summary_tokens + config.top.proposal_recent_tokens,
            config.dimensions.hidden_size,
        )
        if tuple(self.executed_memory.shape) != expected_memory:
            raise ValueError("online cache lost compressed executed-action memory")
        if tuple(self.action_history_keep.shape) != (expected_memory[0],):
            raise ValueError("online cache action-history keep mask must be [B]")
        if tuple(self.role_table.shape) != (8, config.dimensions.hidden_size):
            raise ValueError("online cache lost the shared V120 role table")


@dataclass(frozen=True)
class OnlineTrainingState:
    """Ephemeral training plane released before deployment ODE steps."""

    observation: ObservationEvidence
    top: OnlineTopContext
    history_proposal: HistoryActionProposalState

    def validate(self, config: ExperimentConfig) -> None:
        self.observation.validate()
        self.top.validate(
            hidden=config.dimensions.hidden_size,
            horizon=config.dimensions.action_horizon,
        )
        self.history_proposal.validate(
            horizon=config.dimensions.action_horizon,
            hidden=config.dimensions.hidden_size,
            action_dim=config.dimensions.action_dim,
            history_tokens=(
                config.top.proposal_recent_tokens
                + config.top.proposal_summary_tokens
            ),
        )


@dataclass(frozen=True)
class PolicyStepOutput:
    bottom: BottomOutput
    compiled: CompiledPolicyState
    metrics: dict[str, Tensor]


class ClearVLAMainlinePolicy(nn.Module):
    """Single owner of observation -> G/S/W/P -> Evidence-MMDiT."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dims = config.dimensions
        obs = config.observation
        top = config.top
        self.observation = RestoredV120ObservationCompiler(config)
        self.action_codec = PhysicalActionFieldCodec(
            action_dim=dims.action_dim,
            horizon=dims.action_horizon,
            gripper_field_dim=config.bottom.gripper_field_dim,
            decode_delta_blend=config.bottom.physical_decode_delta_blend,
        )
        self.top = ObjectIntentDynamicsTop(
            hidden=dims.hidden_size,
            content_dim=dims.visual_token_dim,
            route_dim=obs.address_route_dim,
            goal_dim=dims.goal_token_dim,
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            heads=dims.num_heads,
            objects=top.object_slots,
            grounder_iterations=top.grounder_iterations,
            teacher_key_dim=top.teacher_key_dim,
            flow_reference_frames=obs.flow_reference_frames,
            role_host_depth=top.role_host_depth,
            role_host_expansion=top.role_host_ffn_expansion,
            role_host_dropout=top.role_host_dropout,
            core_config=self.observation.v120_config,
        )
        self.history_proposal = HistoryActionProposal(
            action_dim=dims.action_dim,
            hidden=dims.hidden_size,
            heads=dims.num_heads,
            horizon=dims.action_horizon,
            history_length=dims.executed_history_length,
            recent_tokens=top.proposal_recent_tokens,
            summary_tokens=top.proposal_summary_tokens,
            depth=top.proposal_depth,
            expansion=top.role_host_ffn_expansion,
        )
        self.factual_reader = LateRawDetailPolicyReader(
            self.observation.v120_config
        )
        self.transition = ControlledTransitionDynamics(
            hidden=dims.hidden_size,
            content_dim=dims.visual_token_dim,
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            cameras=dims.num_cameras,
            heads=dims.num_heads,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            rank=config.bottom.controlled_delta_rank,
            action_tokens=config.bottom.controlled_action_tokens,
            normalization_floor=config.bottom.normalization_floor,
            dropout=config.bottom.controlled_delta_dropout,
        )
        self.bottom = RestoredV120EvidenceBottom(
            config,
            physical_action_dim=self.action_codec.physical_dim,
        )

    def set_training_step(self, global_step: int) -> float:
        """Advance the serialized V120 execution warm-up/transition schedule."""

        return self.bottom.set_training_step(global_step)

    def encode_online(
        self,
        policy_input: OnlinePolicyInput,
        *,
        training_mask: bool = False,
        geometry_supervision: bool = True,
        context_mask: Tensor | None = None,
        collect_diagnostics: bool = False,
        condition_generator: torch.Generator | None = None,
    ) -> tuple[OnlinePolicyCache, OnlineTrainingState, dict[str, Tensor]]:
        """Build every ODE-invariant value exactly once."""

        policy_input.validate(self.config)
        batch = policy_input.batch
        condition_is_training = bool(self.training and training_mask)

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

        # These three masks are the fixed formal-launcher semantics inherited
        # by the V120 reference from V103.  Build the proposal from complete
        # observable history so its auxiliary target remains fully supervised;
        # exact-null only the values that enter the policy graph.  Deployment
        # and validation never sample these masks and therefore always keep 1.
        goal_keep = condition_keep(
            self.config.top.goal_condition_dropout,
            dtype=policy_input.goal.tokens.dtype,
        )
        history_keep = condition_keep(
            self.config.top.action_history_condition_dropout,
            dtype=policy_input.history.executed_action_history.dtype,
        )
        proposal_keep = condition_keep(
            self.config.top.proposal_condition_dropout,
            dtype=policy_input.history.executed_action_history.dtype,
        )
        history_proposal = self.history_proposal(
            policy_input.history.executed_action_history
        )
        conditioned_history = replace(
            policy_input.history,
            executed_action_history=(
                policy_input.history.executed_action_history
                * history_keep[:, None, None]
            ),
        )
        conditioned_goal = replace(
            policy_input.goal,
            tokens=policy_input.goal.tokens * goal_keep[:, None, None],
        )
        conditioned_policy_input = replace(
            policy_input,
            history=conditioned_history,
            goal=conditioned_goal,
        )
        prepared = self.observation.prepare(
            conditioned_policy_input.observation,
            context_mask=context_mask,
            training_mask=training_mask,
            geometry_supervision=geometry_supervision,
        )
        role_table = self.bottom.sample_role_table(prepared.pack.value_tokens)
        grounding_canvas, grounding_slices = self.bottom.grounding_canvas(
            state=conditioned_policy_input.history.state,
            rollout_init=prepared.pack.future_queries,
            role=role_table,
        )
        grounding_bank, observation_metrics = self.observation.build_grounding_bank(
            prepared,
            grounding_canvas,
            grounding_slices,
            collect_diagnostics=collect_diagnostics,
        )
        progressive_state = self.observation.begin_progressive_grounding(
            grounding_bank
        )
        progressive_state, grounding_canvas, grounding_metrics = (
            self.top.run_progressive_grounding(
                canvas=grounding_canvas,
                slices=grounding_slices,
                visual_memory=grounding_bank.visual_memory,
                visual_value_memory=grounding_bank.visual_value_memory,
                state=progressive_state,
                advance=self.observation.advance_progressive_grounding,
                collect_diagnostics=collect_diagnostics,
            )
        )
        evidence, grounding_fact_metrics = self.observation.finalize_grounding(
            grounding_bank,
            progressive_state,
            collect_diagnostics=collect_diagnostics,
        )
        context, top_metrics = self.top.build_online_context(
            local_facts=evidence.local_facts,
            goal_tokens=conditioned_policy_input.goal.tokens,
            goal_mask=conditioned_policy_input.goal.mask,
            state_history=conditioned_policy_input.history.state_history,
            state=conditioned_policy_input.history.state,
            action_state=conditioned_policy_input.history.action_state,
            executed_history=conditioned_policy_input.history.executed_action_history,
            collect_diagnostics=collect_diagnostics,
        )
        factual_intent = context.intent.factual_dock()
        clean_action_basis = self.bottom.clean_action_basis_tokens(
            batch,
            device=factual_intent.phase_context.device,
            dtype=factual_intent.phase_context.dtype,
        )
        clean_trajectory = clean_action_basis.reshape(
            batch,
            self.config.dimensions.action_horizon
            * self.config.dimensions.action_basis_tokens,
            self.config.dimensions.hidden_size,
        )
        p1_detail = replace(
            evidence.grounding.late_detail,
            progressive_address=evidence.progressive_state,
        )
        g3_rollout = grounding_canvas[:, grounding_slices["rollout"]]
        updated_trajectory, p1_metrics = self.factual_reader(
            clean_trajectory,
            g3_rollout,
            p1_detail,
            phase_context=factual_intent.phase_context,
            condition_query_context=factual_intent.condition_query_context,
            history_query_context=factual_intent.history_query_context,
            clean_basis_tokens=clean_action_basis,
            collect_diagnostics=collect_diagnostics,
        )
        factual_dock = FactualPrecisionDock(
            protected_detail=(updated_trajectory - clean_trajectory).reshape(
                batch,
                self.config.dimensions.action_horizon,
                self.config.dimensions.action_basis_tokens,
                self.config.dimensions.hidden_size,
            )
        )
        factual_dock.validate(
            horizon=self.config.dimensions.action_horizon,
            basis=self.config.dimensions.action_basis_tokens,
        )
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            register_gradient_rms_metric(
                context.intent.public_interval_carrier,
                gradient_metrics,
                "gradient_tensor_s_public_interval_carrier_rms",
            )
            register_gradient_rms_metric(
                context.intent.typed_common_value,
                gradient_metrics,
                "gradient_tensor_s_typed_common_rms",
            )
            register_gradient_rms_metric(
                context.intent.typed_interval_residual_value,
                gradient_metrics,
                "gradient_tensor_s_typed_interval_residual_rms",
            )
            register_gradient_rms_metric(
                factual_dock.protected_detail,
                gradient_metrics,
                "gradient_tensor_p1_static_fact_rms",
            )
        p1_aliases = {
            "flow_jepa_p1_query_rows": "p1_query_rows",
            "flow_jepa_p1_query_chunk": "p1_query_chunk",
            "flow_jepa_p1_shared_factual": "p1_shared_factual",
            "flow_jepa_p1_g3_only_factual_address": "p1_g3_only_address",
            "flow_jepa_p1_clean_basis_entropy": "p1_clean_basis_entropy",
            "flow_jepa_typed_p1_micro_grid": "p1_microgrid_side",
            "flow_jepa_typed_p1_micro_token_count": "p1_microgrid_tokens",
            "flow_jepa_typed_p1_micro_value_rms": "p1_microgrid_value_rms",
            "flow_jepa_typed_p1_spatial_variation": "p1_spatial_variation",
            "flow_jepa_typed_p1_activation_checkpoint_active": (
                "p1_activation_checkpoint_active"
            ),
        }
        transition_source, transition_metrics = self.transition.build_source(
            g3_rollout=g3_rollout,
            collect_diagnostics=collect_diagnostics,
        )
        cache = OnlinePolicyCache(
            top=context.deployment_cache(),
            factual_dock=factual_dock,
            transition_source=transition_source,
            history=conditioned_policy_input.history,
            executed_memory=history_proposal.history_tokens,
            action_history_keep=history_keep,
            role_table=role_table,
        )
        training_state = OnlineTrainingState(
            observation=evidence,
            top=context,
            history_proposal=history_proposal,
        )
        cache.validate(self.config)
        training_state.validate(self.config)
        if not collect_diagnostics:
            return cache, training_state, {}
        return cache, training_state, {
            **observation_metrics,
            **grounding_metrics,
            **grounding_fact_metrics,
            **top_metrics,
            **p1_metrics,
            **{
                target: p1_metrics[source]
                for source, target in p1_aliases.items()
                if source in p1_metrics
            },
            **transition_metrics,
            **gradient_metrics,
            "condition_goal_keep": goal_keep.detach().float().mean(),
            "condition_action_history_keep": history_keep.detach().float().mean(),
            "condition_proposal_keep": proposal_keep.detach().float().mean(),
        }

    def build_training_targets(
        self,
        training_state: OnlineTrainingState,
        future: FutureSupervision,
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[ObjectTopTrainingTargets, dict[str, Tensor]]:
        """Run Teacher-G on a type that the online graph cannot accept."""

        training_state.validate(self.config)
        future.validate(self.config)
        if training_state.top.facts.batch != future.batch:
            raise ValueError("online cache and future supervision batch do not align")
        targets, metrics = self.top.build_training_targets(
            training_state.top,
            future_supports=self.observation.teacher_supports(future.dino_supports),
            future_offsets=future.offsets,
            future_action=future.action_sequence,
            future_state=future.state_sequence,
            collect_diagnostics=collect_diagnostics,
        )
        proposal_rows = F.smooth_l1_loss(
            training_state.history_proposal.action_prediction.float(),
            future.action_sequence[:, : self.config.dimensions.action_horizon]
            .detach()
            .float(),
            reduction="none",
        ).mean(dim=-1)
        proposal_weight = anchor_horizon_weights(
            horizon=self.config.dimensions.action_horizon,
            tail_emphasis=self.config.objectives.horizon_tail_emphasis,
            first_step_protection=self.config.objectives.horizon_first_step_protection,
            device=proposal_rows.device,
        )
        proposal_loss = (proposal_rows * proposal_weight[None]).mean()
        targets = replace(targets, history_proposal_loss=proposal_loss)
        if collect_diagnostics:
            metrics = {
                **metrics,
                "history_action_proposal_loss": proposal_loss.detach(),
            }
        return targets, metrics

    def velocity(
        self,
        cache: OnlinePolicyCache,
        *,
        noisy_action_field: Tensor,
        time: Tensor,
        execution_mode: str = "learned",
        require_execution_supervision: bool = False,
        collect_diagnostics: bool = False,
    ) -> PolicyStepOutput:
        """Run only the ODE-dependent P2/P3 and action bottom."""

        cache.validate(self.config)
        action_query, seed_context = self.bottom.action_and_context(
            noisy_action_field,
            time,
            cache.history,
            executed_memory=cache.executed_memory,
            action_history_keep=cache.action_history_keep,
            role=cache.role_table,
        )
        p1_state, p1_metrics = self.bottom.complete_p1_fact(
            action_query=action_query,
            protected_detail=cache.factual_dock.protected_detail,
            time=time,
            collect_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            register_gradient_rms_metric(
                p1_state.policy_query_residual,
                p1_metrics,
                "gradient_tensor_p1_dynamic_query_residual_rms",
            )
        compiled, top_metrics = self.top.compile_policy(
            cache.top,
            p1_state=p1_state,
            action_query=action_query,
            collect_diagnostics=collect_diagnostics,
        )
        transition, transition_metrics = self.transition(
            source=cache.transition_source,
            action_query=action_query,
            plan=compiled.plan,
            seed=seed_context,
            collect_diagnostics=collect_diagnostics,
        )
        bottom_core, bottom_metrics = self.bottom(
            noisy_action_field=noisy_action_field,
            time=time,
            action_query=action_query,
            plan=compiled.plan,
            intent=cache.top.intent,
            seed=seed_context,
            transition=transition,
            execution_mode=execution_mode,
            require_execution_supervision=require_execution_supervision,
            collect_diagnostics=collect_diagnostics,
        )
        bottom = BottomOutput(
            physical_velocity=bottom_core.physical_velocity,
            motion_logits=bottom_core.motion_logits,
            action_query=bottom_core.action_query,
            block_updates=bottom_core.block_updates,
            evidence_tokens=bottom_core.evidence_tokens,
            decoder_tensors=bottom_core.decoder_tensors,
        )
        bottom.validate(
            action_dim=self.action_codec.physical_dim,
            horizon=self.config.dimensions.action_horizon,
            basis=self.config.dimensions.action_basis_tokens,
            hidden=self.config.dimensions.hidden_size,
        )
        return PolicyStepOutput(
            bottom=bottom,
            compiled=compiled,
            metrics={
                **p1_metrics,
                **top_metrics,
                **transition_metrics,
                **bottom_metrics,
            },
        )

    @torch.no_grad()
    def proposal_ablation_cache(
        self,
        cache: OnlinePolicyCache,
        training_state: OnlineTrainingState,
    ) -> OnlinePolicyCache:
        """Return the unchanged V120 object cache for a proposal intervention.

        In the recovered object path the auxiliary 24-row future proposal is
        not a P1 query and is not the controlled action.  S still reads
        observable executed history, the V120 seed retains the separately
        compressed history memory, and transition reads the current noisy
        action at each ODE step.  Treating proposal-zero as either boundary
        would recreate the schema-20 alias this repair removes.
        """

        if self.training:
            raise ValueError("proposal ablation cache is evaluation-only")
        cache.validate(self.config)
        training_state.validate(self.config)
        return cache


__all__ = [
    "ClearVLAMainlinePolicy",
    "OnlinePolicyCache",
    "OnlineTrainingState",
    "PolicyStepOutput",
]
