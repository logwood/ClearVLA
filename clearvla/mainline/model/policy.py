"""End-to-end online policy with separate training supervision plane."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..interfaces import FutureSupervision, ObservableHistory, OnlinePolicyInput
from .action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from .bottom import BottomOutput, EvidenceMMDiTBottom
from .factual_reader import ObjectFactualReader
from .observation import CurrentObservationCompiler, ObservationEvidence
from .proposal import HistoryActionProposal
from .top import (
    CompiledPolicyState,
    DeploymentTopCache,
    ObjectIntentDynamicsTop,
    OnlineTopContext,
)
from .transition import ControlledTransitionDynamics
from .types import (
    ControlledTransitionState,
    HistoryActionProposalState,
    ObjectFactualDock,
    ObjectTopTrainingTargets,
)


@dataclass(frozen=True)
class OnlinePolicyCache:
    """Current-only state reused by every deployment ODE step."""

    top: DeploymentTopCache
    factual_dock: ObjectFactualDock
    transition: ControlledTransitionState
    history: ObservableHistory

    def validate(self, config: ExperimentConfig) -> None:
        self.history.validate(config)
        self.top.validate(
            hidden=config.dimensions.hidden_size,
            horizon=config.dimensions.action_horizon,
        )
        self.factual_dock.validate()
        self.transition.validate(hidden=config.dimensions.hidden_size)


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
        self.observation = CurrentObservationCompiler(config)
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
        self.factual_reader = ObjectFactualReader(
            hidden=dims.hidden_size,
            content_dim=dims.visual_token_dim,
            raw_dim=obs.feature_dim,
            route_dim=obs.address_route_dim,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            cameras=dims.num_cameras,
            heads=dims.num_heads,
            host_expansion=top.role_host_ffn_expansion,
            host_dropout=top.role_host_dropout,
            microgrid_side=obs.microgrid_side,
        )
        self.transition = ControlledTransitionDynamics(
            hidden=dims.hidden_size,
            content_dim=dims.visual_token_dim,
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            cameras=dims.num_cameras,
            heads=dims.num_heads,
            rank=config.bottom.controlled_delta_rank,
            action_tokens=config.bottom.controlled_action_tokens,
            neutral_tokens=config.bottom.controlled_neutral_tokens,
            dropout=config.bottom.controlled_delta_dropout,
        )
        self.bottom = EvidenceMMDiTBottom(
            config,
            physical_action_dim=self.action_codec.physical_dim,
        )

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
        # from V103 by the V122 script chain.  Build the proposal from complete
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
        proposal_policy_keep = history_keep * proposal_keep
        conditioned_history_proposal = replace(
            history_proposal,
            tokens=(
                history_proposal.tokens
                * proposal_policy_keep.to(dtype=history_proposal.tokens.dtype)[
                    :, None, None
                ]
            ),
            history_tokens=(
                history_proposal.history_tokens
                * history_keep.to(dtype=history_proposal.history_tokens.dtype)[
                    :, None, None
                ]
            ),
        )
        evidence, observation_metrics = self.observation(
            conditioned_policy_input.observation,
            context_mask=context_mask,
            training_mask=training_mask,
            geometry_supervision=geometry_supervision,
            collect_diagnostics=collect_diagnostics,
        )
        context, top_metrics = self.top.build_online_context(
            local_facts=evidence.local_facts,
            goal_tokens=conditioned_policy_input.goal.tokens,
            goal_mask=conditioned_policy_input.goal.mask,
            state_history=conditioned_policy_input.history.state_history,
            state=conditioned_policy_input.history.state,
            executed_history=conditioned_policy_input.history.executed_action_history,
            collect_diagnostics=collect_diagnostics,
        )
        factual_dock, p1_metrics = self.factual_reader(
            evidence=evidence,
            facts=context.facts,
            intent=context.intent,
            coarse_action=context.coarse_action,
            history_proposal=conditioned_history_proposal,
            collect_diagnostics=collect_diagnostics,
        )
        transition, transition_metrics = self.transition(
            dynamics=context.predicted_dynamics,
            proposal=conditioned_history_proposal,
            history=conditioned_policy_input.history,
            collect_diagnostics=collect_diagnostics,
        )
        cache = OnlinePolicyCache(
            top=context.deployment_cache(),
            factual_dock=factual_dock,
            transition=transition,
            history=conditioned_policy_input.history,
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
            **top_metrics,
            **p1_metrics,
            **transition_metrics,
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
        collect_diagnostics: bool = False,
    ) -> PolicyStepOutput:
        """Run only the ODE-dependent P2/P3 and action bottom."""

        cache.validate(self.config)
        action_query = self.bottom.action_query(noisy_action_field, time)
        compiled, top_metrics = self.top.compile_policy(
            cache.top,
            factual_dock=cache.factual_dock,
            action_query=action_query,
            collect_diagnostics=collect_diagnostics,
        )
        bottom, bottom_metrics = self.bottom(
            noisy_action_field=noisy_action_field,
            time=time,
            action_query=action_query,
            plan=compiled.plan,
            history=cache.history,
            transition=cache.transition,
            collect_diagnostics=collect_diagnostics,
        )
        return PolicyStepOutput(
            bottom=bottom,
            compiled=compiled,
            metrics={**top_metrics, **bottom_metrics},
        )


__all__ = [
    "ClearVLAMainlinePolicy",
    "OnlinePolicyCache",
    "OnlineTrainingState",
    "PolicyStepOutput",
]
