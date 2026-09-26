"""The structural branch's Pen and CALVIN outlet adapters remain explicit."""

from __future__ import annotations

from pathlib import Path

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.model.component_contracts import ComponentSelection


ROOT = Path(__file__).resolve().parents[1]
PEN_CONFIG = ROOT / "configs/mainline/structural_rebuild_pen_current.json"
CALVIN_CONFIG = ROOT / "configs/mainline/structural_rebuild_m6n_calvin.json"


def test_structural_pen_config_selects_portable_candidate_graph() -> None:
    config = load_config(PEN_CONFIG)
    assert config.data.data_profile == "identity_7d_pen"
    assert config.data.visual_cache_read_backend == "pread"
    assert config.data.window_boundary_contract == "strict_complete_v1"
    assert config.dimensions.state_dim == 7
    assert config.dimensions.future_supports == 6

    assert config.observation.source_time_mode == "source_history_steps_v1"
    assert config.observation.candidate_support_mode == "full_posterior_lattice_v1"
    assert config.observation.local_ownership_mode == "coupled_observation_v1"

    top = config.top
    assert top.world_camera_condition_mode == "coordinate_role_v1"
    assert top.world_action_condition_mode == "sequence_prefix_v1"
    assert top.future_time_grid_mode == "control_aligned_24_v1"
    assert top.world_supervision_mode == "matched_observed_sequence_v1"
    assert top.world_control_mode == "known_prefix_v1"
    assert top.world_robot_condition_mode == "observed_state_views_v1"
    assert top.p2_geometry_mode == "view_conditioned_transport_v1"
    assert top.p3_coordination_mode == "typed_horizon_v1"
    assert top.history_encoding_mode == "timestamped_streams_v1"
    assert top.entity_context_mode == "completed_g3_v1"
    assert top.entity_chart_mode == "current_image_support_v1"
    assert top.entity_history_mode == "flow_pulled_history_v1"
    assert top.entity_motion_mode == "current_entity_support_v1"
    assert top.target_binding_mode == "shared_operation_v1"
    assert top.operation_intent_mode == "object_outcome_v1"

    selection = ComponentSelection.from_config(config)
    assert selection.conditioning == "timestamped_observable_history_v1"
    assert selection.intent == "timestamped_object_intent_v1"
    assert selection.outlet_adapter == "pen_7d_continuous_v1"
    assert selection.terminal_controller == "continuous_physical_v1"


def test_structural_pen_keeps_calvin_only_producers_off() -> None:
    config = load_config(PEN_CONFIG)
    assert config.data.calvin_raw_source == ""
    assert config.top.state_feature_mode == "native_affine_v1"
    assert config.top.instruction_reference_mode == "none"
    assert config.top.robot_feedback_mode == "none"
    assert config.top.world_feedback_mode == "none"
    assert config.top.annotation_goal_mode == "none"
    assert config.bottom.gripper_output_mode == "continuous"
    assert config.bottom.endpoint_supervision_mode == "interior_heads_v1"
    assert config.bottom.arm_flow_mode == "legacy_independent"
    assert config.bottom.evidence_value_mode == "magnitude_preserving_v1"
    assert config.bottom.transition_condition_mode == "typed_plan_v1"
    assert config.bottom.controller_value_mode == "separate_magnitude_v1"

    schedule = config.runtime.deployment_flow_schedule
    assert isinstance(schedule, dict)
    proposal = schedule["proposal"]
    assert isinstance(proposal, dict)
    parameters = proposal["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["candidate_id"] == "Q5"
    assert parameters["steps"] == 5


def test_structural_calvin_adapter_remains_the_relative_binary_outlet() -> None:
    config = load_config(CALVIN_CONFIG)
    assert config.data.data_profile == "calvin_relative_7d_v1"
    assert config.data.calvin_raw_source
    assert config.data.split_mode == "episode-manifest"
    assert config.dimensions.state_dim == 10
    assert config.dimensions.future_supports == 6
    assert config.top.state_feature_mode == "calvin_tcp_rotation6d_v1"
    assert config.top.history_encoding_mode == "timestamped_streams_v1"
    assert config.top.entity_context_mode == "completed_g3_v1"
    assert config.top.entity_chart_mode == "current_image_support_v1"
    assert config.top.entity_history_mode == "flow_pulled_history_v1"
    assert config.top.entity_motion_mode == "current_entity_support_v1"
    assert config.top.target_binding_mode == "shared_operation_v1"
    assert config.top.world_camera_condition_mode == "coordinate_role_v1"
    assert config.top.world_action_condition_mode == "sequence_prefix_v1"
    assert config.top.future_time_grid_mode == "control_aligned_24_v1"
    assert config.top.world_supervision_mode == "matched_observed_sequence_v1"
    assert config.top.world_control_mode == "known_prefix_v1"
    assert config.top.world_robot_condition_mode == "observed_state_views_v1"
    assert config.top.p2_geometry_mode == "view_conditioned_transport_v1"
    assert config.top.p3_coordination_mode == "typed_horizon_v1"
    assert config.top.instruction_reference_mode == "instruction_start_observation_v1"
    assert config.top.robot_feedback_mode == "one_step_proprioceptive_v1"
    assert config.bottom.arm_flow_mode == "relative_command_adapter"
    assert config.bottom.gripper_output_mode == "calvin_binary_command"
    assert config.bottom.endpoint_supervision_mode == "clean_command_v1"
    assert config.bottom.transition_condition_mode == "typed_plan_v1"

    selection = ComponentSelection.from_config(config)
    assert selection.conditioning == "timestamped_observable_history_v1"
    assert selection.intent == "timestamped_object_intent_v1"
    assert selection.outlet_adapter == "calvin_7d_binary_v1"
    assert selection.terminal_controller == "calvin_binary_command_v1"


def test_structural_pen_config_round_trips_without_falling_back_to_legacy_modes() -> None:
    config = load_config(PEN_CONFIG)
    restored = config_from_mapping(config.as_dict())
    assert restored == config
    assert restored.top.world_action_condition_mode == "sequence_prefix_v1"
    assert restored.top.history_encoding_mode == "timestamped_streams_v1"
    assert restored.top.target_binding_mode == "shared_operation_v1"
