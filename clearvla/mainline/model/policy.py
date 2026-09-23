"""End-to-end online policy with separate training supervision plane."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ExperimentConfig
from ..executed_world import ExecutedWorldFeedback, ExecutedWorldPlanValues, ExecutedWorldWindow
from ..instruction_change import POSTERIOR_REFERENCE_CHANGE, TYPED_CHANGE_MODES
from ..instruction_reference import InstructionReference
from ..interfaces import CurrentObservation, FutureSupervision, ObservableHistory, OnlinePolicyInput
from ..operation_expectation import OBJECT_OUTCOME_INTENT
from ..p2_geometry import VIEW_CONDITIONED_TRANSPORT
from ..robot_execution import RobotResponseFeedback
from ..supervision import quarantine, supported_mean
from ..world_robot import OBSERVED_ROBOT_VIEWS, RobotWorldObservation
from .action_codec import PhysicalActionFieldCodec, anchor_horizon_weights
from .action_contract import BottomOutput
from .component_contracts import ComponentSelection, modular_to_legacy_name
from .components import (
    BridgeStage,
    ConditioningStage,
    ExecutionBottomStage,
    GroundingStage,
    IntentStage,
    ObservationStage,
    OutletAdapter,
    P1Stage,
    PolicyCompilerStage,
    TrainingTargetsStage,
    WorldStage,
)
from .intent import FuturePlanRecognizer
from .observation_contract import ObservationEvidence
from .operation_expectation import OperationExpectationSupervisor
from .proposal import HistoryActionProposal
from .restored_bottom import RestoredV120EvidenceBottom
from .restored_observation import RestoredV120ObservationCompiler
from .routing import register_gradient_rms_metric
from .target_binding import SHARED_TARGET_BINDING, BoundTargetRead
from .teacher import ObjectFutureTeacher
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
    FlowStepContext,
    HistoryActionProposalState,
    ObjectTopTrainingTargets,
    PhysicalActionCondition,
    PhysicalActionSequenceCondition,
    WorldActionCondition,
)
from .v120_p1 import LateRawDetailPolicyReader


def _detach_registered(source: nn.Module, name: str) -> object:
    """Take one already-constructed child out of a temporary source owner."""

    if name in source._modules:
        value = source._modules.pop(name)
        return value
    if name in source._parameters:
        value = source._parameters.pop(name)
        return value
    if name in source._buffers:
        value = source._buffers.pop(name)
        return value
    raise KeyError(f"temporary source has no registered child {name!r}")


def _temporary_source_legacy_name(name: str) -> str:
    """Flatten the newly injected controller only in the old-order ledger."""

    for modular, legacy in (
        ("terminal_controller.optional_command_head", "gripper_command_head"),
        ("terminal_controller.optional_event_head", "event_head"),
        ("terminal_controller.action_norm", "action_norm"),
        ("terminal_controller.velocity_head", "velocity_head"),
        ("terminal_controller.motion_head", "motion_head"),
    ):
        name = name.replace(modular, legacy)
    return name


def _validate_configured_world_action_condition(
    condition: WorldActionCondition,
    config: ExperimentConfig,
) -> None:
    """Bind a cached W action tag to the serialized experiment mode.

    This validator deliberately uses config-only metadata so cache containers
    can reject a schema/profile swap before they reach any module.  The policy
    additionally asks its concrete outlet adapter to verify the chart and
    normalizer identity at every real consumption boundary.
    """

    condition.validate(action_dim=config.dimensions.action_dim)
    mode = config.top.world_action_condition_mode
    if mode == "interval_mean_v1":
        if not isinstance(condition, PhysicalActionCondition):
            raise TypeError(
                "interval-mean policy cache rejected a sequence action condition"
            )
        return
    if mode != "sequence_prefix_v1":
        raise ValueError("unknown configured W action condition mode")
    if not isinstance(condition, PhysicalActionSequenceCondition):
        raise TypeError(
            "sequence-prefix policy cache rejected an interval action condition"
        )
    expected_profile = ComponentSelection.from_config(config).outlet_adapter
    if condition.outlet_profile != expected_profile:
        raise ValueError("policy cache action condition outlet profile changed")


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
    robot_feedback: RobotResponseFeedback | None = None
    world_feedback: ExecutedWorldPlanValues | None = None
    instruction_reference: InstructionReference | None = None

    def validate(self, config: ExperimentConfig) -> None:
        self.history.validate(config)
        if (self.world_feedback is not None)!=(config.top.world_feedback_mode!="none"):
            raise ValueError("executed world feedback cache differs from selected graph")
        if self.world_feedback is not None:
            fb=self.world_feedback
            fb.feedback.validate()
            if fb.current_facts.content is not self.top.belief.content:
                raise ValueError("executed world feedback belongs to another current chart")
            if fb.feedback.window is not self.history.executed_world_window:
                raise ValueError("executed world feedback belongs to another source window")
            if fb.binding is not self.top.intent.target_binding:
                raise ValueError("executed world feedback belongs to another operated target")
        op = self.top.intent.operation_expectation
        if (op is not None) != (config.top.operation_intent_mode == OBJECT_OUTCOME_INTENT):
            raise ValueError("operation expectation cache differs from configured mode")
        if op is not None:
            op.validate()
            if op.current_state is not self.history.state or op.current_content is not self.top.belief.content:
                raise ValueError("operation expectation belongs to another current observation")
            if op.binding is not self.top.intent.target_binding or op.camera_names != tuple(config.data.camera_names):
                raise ValueError("operation expectation target or named camera provenance differs")
        change = self.top.intent.instruction_change
        if (change is not None) != (config.top.instruction_change_mode in TYPED_CHANGE_MODES):
            raise ValueError("instruction change cache differs from configured mode")
        if (self.instruction_reference is not None) != (change is not None):
            raise ValueError("instruction change cache lost its reference owner")
        prepared=self.top.intent.instruction_plan_values
        if (prepared is not None) != (config.top.instruction_change_mode == POSTERIOR_REFERENCE_CHANGE):
            raise ValueError("instruction posterior cache is not prepared for the selected graph")
        if change is not None:
            if (change.posterior is not None) != (config.top.instruction_change_mode == POSTERIOR_REFERENCE_CHANGE):
                raise ValueError("instruction posterior evidence differs from configured mode")
            if prepared is not None and prepared.evidence is not change:
                raise ValueError("instruction prepared cache belongs to another source")
            if change.reference is not self.instruction_reference:
                raise ValueError("instruction change cache belongs to another instruction reference")
            if change.current_state is not self.history.state:
                raise ValueError("instruction change cache belongs to another current observation")
            if change.binding is not self.top.intent.target_binding:
                raise ValueError("instruction change cache uses another operated-object law")
            if change.camera_names != tuple(config.data.camera_names):
                raise ValueError("instruction change cache camera identity mismatch")
        if (self.robot_feedback is not None) != (config.top.robot_feedback_mode != "none"):
            raise ValueError("robot feedback cache differs from configured mode")
        if self.robot_feedback is not None:
            self.robot_feedback.validate(batch=self.history.batch, state_dim=config.dimensions.state_dim, device=self.history.state.device)
            if self.robot_feedback.current_state is not self.history.state or self.robot_feedback.source_step is not self.history.executed_robot_step:
                raise ValueError("robot feedback cache is paired with another observation/command")
            if self.history.executed_robot_step is None or self.robot_feedback.observed is not self.history.executed_robot_step.observed:
                raise ValueError("robot feedback cache support has another owner")
        self.top.validate(
            hidden=config.dimensions.hidden_size,
            horizon=config.dimensions.action_horizon,
        )
        _validate_configured_world_action_condition(self.top.action_condition, config)
        if config.top.p2_geometry_mode == VIEW_CONDITIONED_TRANSPORT and self.top.predicted_dynamics.camera_names != tuple(config.data.camera_names):
            raise ValueError("cached W camera charts differ from P2 configuration")
        if self.top.predicted_dynamics.time_grid_mode != config.top.future_time_grid_mode or self.top.intent.time_grid_mode != config.top.future_time_grid_mode:
            raise ValueError("cached intent/world time grid differs from selected graph")
        has_domain = self.top.predicted_dynamics.control_domain is not None
        if has_domain != (config.top.world_control_mode == "known_prefix_v1"):
            raise ValueError("policy cache control domain differs from configured world mode")
        robot = self.top.belief.robot_observation
        if (robot is not None) != (config.top.world_robot_condition_mode == OBSERVED_ROBOT_VIEWS):
            raise ValueError("policy cache robot observation differs from configured W")
        if robot is not None:
            robot.validate(batch=self.history.state.shape[0], device=self.history.state.device,
                           state_dim=config.dimensions.state_dim, feature_mode=config.top.state_feature_mode)
            if robot.state is not self.history.state:
                raise ValueError("W cache must retain the same current robot state as policy")
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
    robot_response_loss: Tensor | None = None

    def validate(self, config: ExperimentConfig) -> None:
        if (self.robot_response_loss is not None) != (config.top.robot_feedback_mode != "none"):
            raise ValueError("robot response training objective differs from configured mode")
        if self.robot_response_loss is not None and self.robot_response_loss.ndim != 0:
            raise ValueError("robot response training objective must be scalar")
        self.observation.validate()
        self.top.validate(
            hidden=config.dimensions.hidden_size,
            horizon=config.dimensions.action_horizon,
        )
        _validate_configured_world_action_condition(self.top.action_condition, config)
        if config.top.p2_geometry_mode == VIEW_CONDITIONED_TRANSPORT and self.top.predicted_dynamics.camera_names != tuple(config.data.camera_names):
            raise ValueError("cached W camera charts differ from P2 configuration")
        if self.top.predicted_dynamics.time_grid_mode != config.top.future_time_grid_mode or self.top.intent.time_grid_mode != config.top.future_time_grid_mode:
            raise ValueError("cached intent/world time grid differs from selected graph")
        has_domain = self.top.predicted_dynamics.control_domain is not None
        if has_domain != (config.top.world_control_mode == "known_prefix_v1"):
            raise ValueError("policy cache control domain differs from configured world mode")
        belief = self.top.current_world_belief
        robot = None if belief is None else belief.robot_observation
        if (robot is not None) != (config.top.world_robot_condition_mode == OBSERVED_ROBOT_VIEWS):
            raise ValueError("training cache robot observation differs from configured W")
        if robot is not None:
            robot.validate(batch=self.top.facts.batch, device=self.top.facts.content.device,
                           state_dim=config.dimensions.state_dim, feature_mode=config.top.state_feature_mode)
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
        selection = ComponentSelection.from_config(config)
        self.selection = selection

        # Construction is intentionally performed in the frozen pre-modular
        # order.  The temporary source owners are never attached to this
        # policy; their already-initialized children are moved exactly once to
        # the final registered component hierarchy below.
        raw_observation = RestoredV120ObservationCompiler(config)
        raw_codec = PhysicalActionFieldCodec(
            action_dim=dims.action_dim,
            horizon=dims.action_horizon,
            gripper_field_dim=config.bottom.gripper_field_dim,
            decode_delta_blend=config.bottom.physical_decode_delta_blend,
        )
        raw_top = ObjectIntentDynamicsTop(
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
            camera_names=config.data.camera_names,
            world_camera_condition_mode=top.world_camera_condition_mode,
            world_action_condition_mode=top.world_action_condition_mode,
            future_time_grid_mode=top.future_time_grid_mode,
            world_control_mode=top.world_control_mode,
            world_robot_condition_mode=top.world_robot_condition_mode,
            state_feature_mode=top.state_feature_mode,
            p2_spatial_intent_mode=top.p2_spatial_intent_mode,
            p2_geometry_mode=top.p2_geometry_mode,
            p3_coordination_mode=top.p3_coordination_mode,
            robot_feedback_mode=top.robot_feedback_mode,
            world_feedback_mode=top.world_feedback_mode,
            target_binding_mode=top.target_binding_mode,
            instruction_reference_mode=top.instruction_reference_mode,
            instruction_change_mode=top.instruction_change_mode,
            operation_intent_mode=top.operation_intent_mode,
            history_encoding_mode=top.history_encoding_mode,
            entity_context_mode=top.entity_context_mode,
            entity_chart_mode=top.entity_chart_mode,
            entity_history_mode=top.entity_history_mode,
            entity_motion_mode=top.entity_motion_mode,
            core_config=raw_observation.v120_config,
        )
        raw_history_proposal = HistoryActionProposal(
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
        raw_factual_reader = LateRawDetailPolicyReader(raw_observation.v120_config)
        raw_transition = ControlledTransitionDynamics(
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
            condition_mode=config.bottom.transition_condition_mode,
            dropout=config.bottom.controlled_delta_dropout,
        )
        raw_bottom = RestoredV120EvidenceBottom(
            config,
            physical_action_dim=raw_codec.physical_dim,
        )

        raw_target_reader = (BoundTargetRead(dims.hidden_size, dims.num_heads)
                             if top.target_binding_mode == SHARED_TARGET_BINDING else None)

        # Capture the exact old traversal before changing registrations.  This
        # ledger is consumed by optimizer/clipping code and by checkpoint
        # migration; the new human-facing module order is deliberately not
        # allowed to become an order-sensitive behavior change.
        old_parameter_objects: list[tuple[str, nn.Parameter]] = []
        old_state_names: list[str] = []
        old_buffer_names: list[str] = []
        for prefix, owner in (
            ("observation", raw_observation),
            ("action_codec", raw_codec),
            ("top", raw_top),
            ("history_proposal", raw_history_proposal),
            ("factual_reader", raw_factual_reader),
            ("transition", raw_transition),
            ("bottom", raw_bottom),
            *((("operated_target_reader", raw_target_reader),) if raw_target_reader is not None else ()),
        ):
            for name in owner.state_dict().keys():
                logical_state = _temporary_source_legacy_name(f"{prefix}.{name}")
                old_state_names.append(logical_state)
            for name, _buffer in owner.named_buffers():
                logical_buffer = _temporary_source_legacy_name(f"{prefix}.{name}")
                old_buffer_names.append(logical_buffer)
            for name, parameter in owner.named_parameters():
                logical = _temporary_source_legacy_name(f"{prefix}.{name}")
                # The decoder source is already terminal-controller based at
                # construction time.  Flatten only this temporary ledger to
                # the exact pre-modular logical names used by old checkpoints.
                old_parameter_objects.append((logical, parameter))

        observation = ObservationStage(raw_observation)
        codec = OutletAdapter(
            raw_codec,
            selection=selection.outlet_adapter,
            world_action_condition_mode=top.world_action_condition_mode,
            future_time_grid_mode=top.future_time_grid_mode,
        )
        # Move every top child without constructing a second parameterized
        # implementation.  Direct parameters are detached from the temporary
        # owner in the same way as modules.
        grounding_blocks = _detach_registered(raw_top, "grounding_blocks")
        grounding_content_mod = _detach_registered(raw_top, "grounding_content_mod")
        grounding_content_mod_scale = _detach_registered(raw_top, "grounding_content_mod_scale")
        grounder = _detach_registered(raw_top, "grounder")
        organizer = _detach_registered(raw_top, "intent")
        coarse_action = _detach_registered(raw_top, "coarse_action")
        dynamics = _detach_registered(raw_top, "dynamics")
        teacher = _detach_registered(raw_top, "teacher")
        recognizer = _detach_registered(raw_top, "recognizer")
        if not isinstance(teacher, ObjectFutureTeacher) or not isinstance(recognizer, (FuturePlanRecognizer, OperationExpectationSupervisor)):
            raise TypeError("training target owners must implement the declared physical time grid")
        effect_reader = _detach_registered(raw_top, "effect_reader")
        consequence = _detach_registered(raw_top, "consequence")
        plan_compiler = _detach_registered(raw_top, "plan_compiler")

        # Bottom children are detached only after their construction order has
        # been recorded.  The decoder already owns its terminal controller.
        query_encoder = _detach_registered(raw_bottom, "query_encoder")
        p1_time = _detach_registered(raw_bottom, "p1_time")
        p1_content_mod = _detach_registered(raw_bottom, "p1_content_mod")
        p1_content_mod_scale = _detach_registered(raw_bottom, "p1_content_mod_scale")
        p1_policy_block = _detach_registered(raw_bottom, "p1_policy_block")
        layer_contract_heads = _detach_registered(raw_bottom, "layer_contract_heads")
        decoder = _detach_registered(raw_bottom, "decoder")

        # Final registered hierarchy.  Each child appears under one owner only.
        self.conditioning = ConditioningStage(raw_history_proposal)
        self.observation = observation
        self.bridge = BridgeStage(query_encoder)
        self.grounding = GroundingStage(
            hidden=dims.hidden_size,
            content_dim=dims.visual_token_dim,
            route_dim=obs.address_route_dim,
            blocks=grounding_blocks,
            content_mod=grounding_content_mod,
            content_mod_scale=grounding_content_mod_scale,
            grounder=grounder,
        )
        self.intent = IntentStage(organizer, coarse_action)
        self.world = WorldStage(
            hidden=dims.hidden_size,
            action_dim=dims.action_dim,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            dynamics=dynamics,
        )
        self.p1 = P1Stage(
            hidden=dims.hidden_size,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            factual_reader=raw_factual_reader,
            target_reader=raw_target_reader,
            dynamic_time=p1_time,
            dynamic_content_mod=p1_content_mod,
            dynamic_content_mod_scale=p1_content_mod_scale,
            dynamic_policy_block=p1_policy_block,
        )
        self.policy_compiler = PolicyCompilerStage(
            hidden=dims.hidden_size,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            action_dim=dims.action_dim,
            effect_reader=effect_reader,
            consequence=consequence,
            plan_compiler=plan_compiler,
        )
        self.transition = raw_transition
        self.execution_bottom = ExecutionBottomStage(
            hidden=dims.hidden_size,
            horizon=dims.action_horizon,
            basis=dims.action_basis_tokens,
            physical_action_dim=codec.physical_dim,
            core_config=raw_bottom.core_config,
            layer_contract_heads=layer_contract_heads,
            decoder=decoder,
            transition_condition_mode=config.bottom.transition_condition_mode,
        )
        self.training_targets = TrainingTargetsStage(
            teacher=teacher,
            recognizer=recognizer,
            hidden=dims.hidden_size,
            horizon=dims.action_horizon,
            action_dim=dims.action_dim,
        )
        self.outlet_adapter = codec

        # Resolve final names by identity, then retain the frozen legacy order.
        final_by_id = {id(parameter): name for name, parameter in self.named_parameters()}
        self._legacy_parameter_order = tuple(
            (final_by_id[id(parameter)], old_name)
            for old_name, parameter in old_parameter_objects
        )
        if len(self._legacy_parameter_order) != len(final_by_id):
            raise RuntimeError("modular registration lost or duplicated a parameter")
        final_state_names = {
            modular_to_legacy_name(name): name for name in self.state_dict()
        }
        try:
            self._legacy_state_order = tuple(
                (final_state_names[legacy_name], legacy_name)
                for legacy_name in old_state_names
            )
        except KeyError as error:
            raise RuntimeError(
                f"modular registration lost state key {error.args[0]!r}"
            ) from error
        if len(self._legacy_state_order) != len(final_state_names):
            raise RuntimeError("modular registration lost or duplicated a state key")
        final_buffer_names = {
            modular_to_legacy_name(name): name for name, _ in self.named_buffers()
        }
        try:
            self._legacy_buffer_order = tuple(
                (final_buffer_names[legacy_name], legacy_name)
                for legacy_name in old_buffer_names
            )
        except KeyError as error:
            raise RuntimeError(
                f"modular registration lost buffer {error.args[0]!r}"
            ) from error
        if len(self._legacy_buffer_order) != len(final_buffer_names):
            raise RuntimeError("modular registration lost or duplicated a buffer")

    def set_training_step(self, global_step: int) -> float:
        """Advance the serialized V120 execution warm-up/transition schedule."""

        return self.execution_bottom.set_training_step(global_step)

    def configure_action_normalizer(self, normalizer: object) -> None:
        """Attach the data/checkpoint-owned outlet chart sidecar."""

        self.outlet_adapter.configure_action_normalizer(normalizer)

    def _admit_executed_world_source(self, policy_input: OnlinePolicyInput,
                                      window: ExecutedWorldWindow) -> None:
        """Source agreement before neural/RNG work; never per ODE node."""
        window.validate(self.config,batch=policy_input.batch,device=policy_input.device,strict=True)
        timing=policy_input.history.timing
        if timing is None:
            raise ValueError("executed world comparison requires physical times")
        observed=window.observed
        if bool((observed & (timing.state_offsets[:,1]!=-4)).any()):
            raise ValueError("executed world anchor precedes actual episode")
        if bool((observed & (window.visual_offsets[:,1]-4!=timing.state_offsets[:,0])).any()):
            raise ValueError("executed world visual anchor and current clock disagree")
        def same(left: Tensor,right: Tensor,name: str) -> None:
            mask=observed.reshape(observed.shape[0],*([1]*(left.ndim-1)))
            if not torch.equal(torch.where(mask,left,0.),torch.where(mask,right,0.)):
                raise ValueError(f"executed world {name} disagrees with observed history")
        same(window.state,policy_input.history.state_history[:,1],"state")
        same(window.dino_history[:,1:],policy_input.observation.dino_history[:,:2],"DINO source")
        same(window.raw_rgb[:,1:],policy_input.observation.raw_rgb[:,:2],"RGB source")
        for index,offset in enumerate((-4,-3,-2,-1)):
            matches=timing.action_offsets==offset
            admitted=matches & timing.action_executed & observed[:,None]
            if bool((observed & matches.any(1) & ~admitted.any(1)).any()):
                raise ValueError("executed world command claims unavailable history")
            left=window.commands[:,index,None].expand_as(policy_input.history.executed_action_history)
            if not torch.equal(torch.where(admitted[...,None],left,0.),
                torch.where(admitted[...,None],policy_input.history.executed_action_history,0.)):
                raise ValueError("executed world commands disagree with sparse history")

    @torch.no_grad()
    def _replay_executed_world(self, policy_input: OnlinePolicyInput,
                               role: Tensor) -> ExecutedWorldFeedback:
        """Current-weight retrodiction from past facts and confirmed controls.

        Not a saved ex-ante forecast. Current images only enter the subsequent
        observed measurement. Future label aggregation/recognizer is not used.
        """
        window=policy_input.history.executed_world_window
        if window is None:
            raise ValueError("executed world replay lost source window")
        batch=policy_input.batch
        def safe(x: Tensor) -> Tensor:
            return torch.where(window.observed.reshape(batch,*([1]*(x.ndim-1))),x,0.)
        prepared=self.observation.prepare(CurrentObservation(safe(window.dino_history),safe(window.raw_rgb)),
            source_offsets=window.visual_offsets,training_mask=False,geometry_supervision=False,collect_diagnostics=False)
        canvas,slices=self.bridge.build_grounding_seed(state=safe(window.state),
            rollout_init=prepared.pack.future_queries,role=role.detach())
        bank,_=self.observation.build_grounding_bank(prepared,canvas,slices,collect_diagnostics=False)
        progressive=self.observation.begin_progressive_grounding(bank)
        progressive,_,_=self.grounding.build_current(canvas=canvas,slices=slices,
            visual_memory=bank.visual_memory,visual_value_memory=bank.visual_value_memory,
            visual_memory_observed=bank.visual_memory_observed,state=progressive,
            advance=self.observation.advance_progressive_grounding,collect_diagnostics=False)
        evidence,_=self.observation.finalize_grounding(bank,progressive,collect_diagnostics=False)
        past_facts,_=self.grounding.materialize_facts(evidence.local_facts,collect_diagnostics=False)
        controls=self.outlet_adapter.executed_world_condition(safe(window.commands),safe(window.action_state),window.observed)
        predicted=self.world.dynamics.predict_executed_endpoint(
            facts=past_facts.world_belief(robot_observation=RobotWorldObservation(
                state=safe(window.state),feature_mode=self.config.top.state_feature_mode)),action=controls)
        current=policy_input.observation.dino_history
        grid=self.observation.observation_supports(safe(current[:,-1:]))
        measured=self.training_targets.teacher.measure_observations(facts=past_facts,observations=grid,
            relative_offsets=torch.full((batch,1),4,device=current.device,dtype=torch.long),observed=window.observed[:,None])
        views=(measured.camera_legal[...,0]>0) & window.observed[:,None,None]
        semantic=measured.successor_per_support[:,0]-measured.current_reference[:,0]-predicted.semantic.float()
        image=measured.transport_per_support[:,0]-predicted.image.float()
        feedback=ExecutedWorldFeedback(
            semantic=torch.where(views.any(-1)[...,None],semantic,0.).float(),
            image=torch.where(views[...,None],image,0.).float(),
            covariance=torch.where(views[...,None],measured.covariance_per_support[:,0]+predicted.covariance.float(),0.),
            null=measured.null_probability[:,0].float(),posterior=measured.candidate_posterior[:,0].flatten(-2).float(),
            past_content=torch.where(views.any(-1)[...,None],past_facts.content,0.).float(),
            view_observed=views,window=window,current_dino=current,camera_names=tuple(self.config.data.camera_names),
            measurement_shape=(int(grid.shape[-3]),int(grid.shape[-2])))
        feedback.validate()
        return feedback

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
        if self.config.top.history_encoding_mode == "timestamped_streams_v1":
            assert policy_input.history.timing is not None
            policy_input.history.timing.validate(
                batch=batch,
                states=self.config.dimensions.state_history_length,
                actions=self.config.dimensions.executed_history_length,
                device=policy_input.device,
                strict=True,
            )
        executed_window=policy_input.history.executed_world_window
        if executed_window is not None:
            self._admit_executed_world_source(policy_input,executed_window)
        step = policy_input.history.executed_robot_step
        if step is not None:
            step.validate(batch=batch, state_dim=self.config.dimensions.state_dim,
                          action_dim=self.config.dimensions.action_dim, device=policy_input.device, strict=True)
            timing = policy_input.history.timing
            if timing is None or not torch.equal(step.observed, timing.action_executed[:, -1]):
                raise ValueError("robot pair support disagrees with last executed command")
            if not torch.all(timing.action_offsets[:, -1] == -1):
                raise ValueError("robot feedback command timestamp must be exactly minus one")
            if not torch.equal(torch.where(step.observed[:, None], step.command, 0.0),
                               torch.where(step.observed[:, None], policy_input.history.executed_action_history[:, -1], 0.0)):
                raise ValueError("robot step command is not the recorded last executed command")
        # Conditioning owns the two masks and the auxiliary proposal.  It is
        # called once, preserving the original RNG draw order.
        conditioned_policy_input, history_proposal, goal_keep, history_keep = (
            self.conditioning.prepare(
                policy_input,
                config=self.config,
                training=self.training,
                training_mask=training_mask,
                condition_generator=condition_generator,
            )
        )
        robot_feedback, robot_response_loss = None, None
        observer = self.policy_compiler.plan_compiler.robot_observer
        if observer is not None:
            executed_step = conditioned_policy_input.history.executed_robot_step
            if executed_step is None:
                raise ValueError("robot response lost its one-step observation pair")
            robot_feedback, robot_response_loss = observer.observe(executed_step, conditioned_policy_input.history.state)
        source_offsets = None
        if self.config.observation.source_time_mode == "source_history_steps_v1":
            source_clock = conditioned_policy_input.history.timing
            if source_clock is None:
                raise ValueError("source-timed observation requires a physical source clock")
            source_offsets = source_clock.state_offsets
        prepared = self.observation.prepare(
            conditioned_policy_input.observation,
            source_offsets=source_offsets,
            context_mask=context_mask,
            training_mask=training_mask,
            geometry_supervision=geometry_supervision,
        )
        role_table = self.bridge.sample_role_context(prepared.pack.value_tokens)
        grounding_canvas, grounding_slices = self.bridge.build_grounding_seed(
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
            self.grounding.build_current(
                canvas=grounding_canvas,
                slices=grounding_slices,
                visual_memory=grounding_bank.visual_memory,
                visual_value_memory=grounding_bank.visual_value_memory,
                visual_memory_observed=grounding_bank.visual_memory_observed,
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
        facts, grounding_context_metrics = self.grounding.materialize_facts(
            evidence.local_facts,
            collect_diagnostics=collect_diagnostics,
        )
        intent, intent_metrics = self.intent.organize(
            goal_tokens=conditioned_policy_input.goal.tokens,
            goal_mask=conditioned_policy_input.goal.mask,
            state_history=conditioned_policy_input.history.state_history,
            state=conditioned_policy_input.history.state,
            executed_history=conditioned_policy_input.history.executed_action_history,
            history_timing=conditioned_policy_input.history.timing,
            instruction_reference=conditioned_policy_input.instruction_reference,
            current_dino=conditioned_policy_input.observation.dino_history[:, -1],
            facts=facts,
            collect_diagnostics=collect_diagnostics,
        )
        intent = self.policy_compiler.plan_compiler.prepare_instruction_values(intent)
        world_feedback=None
        reader=self.policy_compiler.plan_compiler.world_feedback_read
        if reader is not None:
            if intent.target_binding is None:
                raise ValueError("executed world feedback lost current target")
            feedback=self._replay_executed_world(conditioned_policy_input,role_table)
            world_feedback=reader.prepare(feedback,facts,intent.target_binding)
        action_intent = intent.action_dock()
        coarse = self.intent.propose_action(
            action_intent,
            collect_diagnostics=collect_diagnostics,
        )
        action_condition = self.outlet_adapter.world_condition_from_action_prediction(
            coarse.action_prediction,
            conditioned_policy_input.history.action_state,
        )
        self.outlet_adapter.validate_world_condition(action_condition)
        world_belief = (
            facts.world_belief(robot_observation=RobotWorldObservation(
                state=conditioned_policy_input.history.state,
                feature_mode=self.config.top.state_feature_mode,
            )) if self.config.top.world_robot_condition_mode == OBSERVED_ROBOT_VIEWS else None
        )
        world, world_metrics = self.world.materialize(
            belief=facts if world_belief is None else world_belief,
            action_condition=action_condition,
            collect_diagnostics=collect_diagnostics,
        )
        context = OnlineTopContext(
            facts=facts,
            intent=intent,
            coarse_action=coarse,
            candidate_world=world,
            current_world_belief=world_belief,
        )
        context.validate(hidden=self.config.dimensions.hidden_size, horizon=self.config.dimensions.action_horizon)
        top_metrics = {
            **grounding_context_metrics,
            **intent_metrics,
            **world_metrics,
        }
        if collect_diagnostics:
            predicted = world.dynamics
            top_metrics.update(
                {
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
                        (
                            action_condition.source_action
                            if isinstance(
                                action_condition,
                                PhysicalActionSequenceCondition,
                            )
                            else action_condition.source_interval_action
                        )
                        .detach()
                        .float()
                        - coarse.action_prediction.detach().float()
                    )
                    .abs()
                    .amax(),
                }
            )
        factual_intent = context.intent.factual_dock()
        clean_action_basis = self.bridge.clean_action_basis(
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
        factual_dock, p1_metrics = self.p1.build_static(
            clean_trajectory=clean_trajectory,
            g3_rollout=g3_rollout,
            detail=p1_detail,
            phase_context=factual_intent.phase_context,
            target_binding=factual_intent.target_binding,
            target_evidence=factual_intent.target_evidence,
            condition_query_context=factual_intent.condition_query_context,
            history_query_context=factual_intent.history_query_context,
            clean_basis_tokens=clean_action_basis,
            collect_diagnostics=collect_diagnostics,
        )
        gradient_metrics: dict[str, Tensor] = {}
        if collect_diagnostics:
            register_gradient_rms_metric(
                context.intent.public_interval_carrier,
                gradient_metrics,
                "gradient_tensor_s_public_interval_carrier_rms",
            )
            register_gradient_rms_metric(
                context.intent.object_tokens,
                gradient_metrics,
                "gradient_tensor_s_public_object_memory_rms",
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
            world_feedback=world_feedback,
            robot_feedback=robot_feedback,
            instruction_reference=(conditioned_policy_input.instruction_reference
                                   if self.config.top.instruction_change_mode in TYPED_CHANGE_MODES else None),
            top=context.deployment_cache(),
            factual_dock=factual_dock,
            transition_source=transition_source,
            history=conditioned_policy_input.history,
            executed_memory=history_proposal.history_tokens,
            action_history_keep=history_keep,
            role_table=role_table,
        )
        training_state = OnlineTrainingState(
            robot_response_loss=robot_response_loss,
            observation=evidence,
            top=context,
            history_proposal=history_proposal,
        )
        cache.validate(self.config)
        training_state.validate(self.config)
        self.outlet_adapter.validate_world_condition(
            training_state.top.action_condition
        )
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
        self.outlet_adapter.validate_world_condition(
            training_state.top.action_condition
        )
        future.validate(self.config)
        if training_state.top.facts.batch != future.batch:
            raise ValueError("online cache and future supervision batch do not align")
        source_dino = future.dino_supports
        source_action = future.action_sequence
        source_state = future.state_sequence
        if future.support is not None:
            source_dino = quarantine(source_dino, future.support.visual)
            source_action = quarantine(source_action, future.support.action)
            source_state = quarantine(source_state, future.support.state)
        targets, metrics = self.training_targets.build(
            training_state.top,
            future_supports=self.observation.teacher_supports(source_dino),
            future_offsets=future.offsets,
            future_action=source_action,
            future_state=source_state,
            coarse_action=self.intent.coarse_action,
            label_support=future.support,
            collect_diagnostics=collect_diagnostics,
        )
        proposal_rows = F.smooth_l1_loss(
            training_state.history_proposal.action_prediction.float(),
            source_action[:, : self.config.dimensions.action_horizon].detach().float(),
            reduction="none",
        ).mean(dim=-1)
        proposal_weight = anchor_horizon_weights(
            horizon=self.config.dimensions.action_horizon,
            tail_emphasis=self.config.objectives.horizon_tail_emphasis,
            first_step_protection=self.config.objectives.horizon_first_step_protection,
            device=proposal_rows.device,
        )
        proposal_loss = supported_mean(
            proposal_rows * proposal_weight[None],
            None if future.support is None else future.support.action[:, : proposal_rows.shape[1]],
        )
        targets = replace(targets, history_proposal_loss=proposal_loss)
        if self.config.top.world_supervision_mode == "matched_observed_sequence_v1":
            controls = self.outlet_adapter.observed_world_condition(
                source_action, training_state.top.action_condition.current_action,
                None if future.support is None else future.support.action,
            )
            supervised, world_metrics = self.world.materialize_supervised(
                belief=(training_state.top.facts if training_state.top.current_world_belief is None
                        else training_state.top.current_world_belief), action_condition=controls,
                collect_diagnostics=collect_diagnostics,
            )
            targets = replace(targets, supervised_world=supervised)
            metrics = {**metrics, **world_metrics}
            if collect_diagnostics:
                metrics["supervised_world_action_interval_fraction"] = controls.interval_observed.float().mean()
        if collect_diagnostics:
            metrics = {
                **metrics,
                "history_action_proposal_loss": proposal_loss.detach(),
            }
        if self.config.top.robot_feedback_mode != "none":
            if training_state.robot_response_loss is None:
                raise ValueError("robot response objective missing from online training plane")
            targets = replace(targets, robot_response_loss=training_state.robot_response_loss)
        return targets, metrics

    def velocity(
        self,
        cache: OnlinePolicyCache,
        *,
        noisy_action_field: Tensor,
        time: Tensor,
        flow_step_context: FlowStepContext | None = None,
        execution_mode: str = "learned",
        deployment_fastpath: bool = False,
        require_execution_supervision: bool = False,
        collect_diagnostics: bool = False,
    ) -> PolicyStepOutput:
        """Run only the ODE-dependent P2/P3 and action bottom."""

        cache.validate(self.config)
        self.outlet_adapter.validate_world_condition(cache.top.action_condition)
        if flow_step_context is not None:
            flow_step_context.validate(
                batch=int(noisy_action_field.shape[0]),
                device=noisy_action_field.device,
                time=time,
                # The sampler/training bridge constructs this context from a
                # host-validated schedule.  Keep the public shape/device seam
                # checked here, but do not perform tensor-to-Python finite or
                # range checks at every ODE node.
                strict=False,
            )
        # A binary outlet's legacy continuous gripper field is shape/audit
        # compatibility only. During training it contains a future-target
        # mixture, while its zero deployment velocity leaves it as source
        # noise. Sanitize it once before *any* dynamic consumer so neither the
        # command head nor an indirect P/transition/bottom route can learn that
        # unavailable shortcut. Continuous outlets retain the original tensor
        # object and values.
        model_action_field = self.outlet_adapter.prepare_model_input(
            noisy_action_field
        )
        action_query, seed_context = self.bridge.action_and_context(
            model_action_field,
            time,
            cache.history,
            executed_memory=cache.executed_memory,
            action_history_keep=cache.action_history_keep,
            role=cache.role_table,
        )
        p1_state, p1_metrics = self.p1.update_dynamic(
            action_query=action_query,
            factual=cache.factual_dock,
            time=time,
            collect_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            register_gradient_rms_metric(
                p1_state.policy_query_residual,
                p1_metrics,
                "gradient_tensor_p1_dynamic_query_residual_rms",
            )
        compiled, top_metrics = self.policy_compiler.compile(
            cache.top,
            robot_feedback=cache.robot_feedback,
            world_feedback=cache.world_feedback,
            p1_state=p1_state,
            action_query=action_query,
            collect_diagnostics=collect_diagnostics,
        )
        transition, transition_metrics = self.transition(
            source=cache.transition_source,
            action_query=action_query,
            plan=compiled.plan,
            consequence=compiled.consequence,
            seed=seed_context,
            collect_diagnostics=collect_diagnostics,
        )
        bottom_core, bottom_metrics = self.execution_bottom.step(
            noisy_action_field=model_action_field,
            time=time,
            flow_step_context=flow_step_context,
            action_query=action_query,
            plan=compiled.plan,
            intent=cache.top.intent,
            seed=seed_context,
            transition=transition,
            execution_mode=execution_mode,
            deployment_fastpath=deployment_fastpath,
            require_execution_supervision=require_execution_supervision,
            collect_diagnostics=collect_diagnostics,
        )
        conditioning_metrics = self.outlet_adapter.conditioning_metrics(
            noisy_action_field,
            model_action_field,
            collect_diagnostics=collect_diagnostics,
        )
        bottom = BottomOutput(
            physical_velocity=bottom_core.physical_velocity,
            motion_logits=bottom_core.motion_logits,
            action_query=bottom_core.action_query,
            block_updates=bottom_core.block_updates,
            evidence_tokens=bottom_core.evidence_tokens,
            decoder_tensors=bottom_core.decoder_tensors,
            gripper_command_logits=bottom_core.gripper_command_logits,
        )
        bottom.validate(
            action_dim=self.outlet_adapter.physical_dim,
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
                **conditioning_metrics,
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
