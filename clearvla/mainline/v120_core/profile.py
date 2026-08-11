"""Resolved V120 model profile used by the restored mainline.

This is not another version contract.  It is the exact model configuration
serialized by the reference V120 run whose source commit is 0b92d35.  Only
values differing from the V39 dataclass defaults are stored here; derived
dataset/model dimensions are already resolved to the formal 63/5/5 run.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import V39PolicyConfig

V120_REFERENCE_SOURCE_COMMIT = "0b92d35"
V120_REFERENCE_NORMALIZER_FINGERPRINT = "32a3a4d7f21f"

V120_MODEL_OVERRIDES: dict[str, object] = {
    "action_history_condition_dropout": 0.1,
    "action_history_condition_exact_null": 1,
    "action_history_enabled": 1,
    "action_history_proposal_detach": 0,
    "adaptive_cvae_function_adapters": 0,
    "adaptive_cvae_function_rank": 32,
    "adaptive_cvae_micro_control": 0,
    "adaptive_cvae_micro_kd_init": 0.04,
    "adaptive_cvae_micro_kd_max": 0.3,
    "adaptive_cvae_micro_kp_init": 0.22,
    "adaptive_cvae_micro_refine_block": 0,
    "adaptive_cvae_micro_refine_block_scale": 0.0,
    "adaptive_cvae_micro_step_init": 0.14,
    "adaptive_cvae_refine_steps": 6,
    "adaptive_cvae_route_entropy_floor_ratio": 0.15,
    "controlled_base_mode": 'fixed_zero',
    "depth": 8,
    "executed_action_offsets": (-24, -16, -12, -8, -6, -4, -2, -1),
    "executed_history_length": 8,
    "final_action_decoder": 'evidence_latent_mmdit_action',
    "flow_jepa_action_free_world_factual": 1,
    "flow_jepa_address_flow_prior_floor": 0.25,
    "flow_jepa_bounded_flow_coordinates": 1,
    "flow_jepa_complementary_raw_detail": 1,
    "flow_jepa_complete_numerical_contract": 1,
    "flow_jepa_coordinate_typed_raw_detail": 1,
    "flow_jepa_enabled": 1,
    "flow_jepa_functional_mainline_routing": 1,
    "flow_jepa_g_aligned_future_effect": 1,
    "flow_jepa_horizon_cell_fine_address": 1,
    "flow_jepa_horizon_soft_address": 1,
    "flow_jepa_interval_boundaries": (4, 8, 16, 32, 48),
    "flow_jepa_interval_stage_delta": 1,
    "flow_jepa_interval_stage_typed_value": 1,
    "flow_jepa_interval_support_offsets": (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
    "flow_jepa_late_bottleneck": 1,
    "flow_jepa_late_policy_detail": 1,
    "flow_jepa_object_intent_dynamics_mainline": 1,
    "flow_jepa_online_horizon_address": 1,
    "flow_jepa_p1_mixed_precision": 1,
    "flow_jepa_policy_blocks": 3,
    "flow_jepa_policy_multi_glimpse_address": 1,
    "flow_jepa_policy_plan_compiler": 1,
    "flow_jepa_policy_workspace_horizon_pool": 1,
    "flow_jepa_pre_value_owner_routing": 1,
    "flow_jepa_predictive_change_contract": 1,
    "flow_jepa_progressive_grounding_address": 1,
    "flow_jepa_raw_image_enabled": 1,
    "flow_jepa_role_hierarchy": 1,
    "flow_jepa_sequential_horizon_memory": 1,
    "flow_jepa_shared_factual_glimpse_bank": 1,
    "flow_jepa_soft_address_lattice": 1,
    "flow_jepa_source_aligned_raw_fusion": 1,
    "flow_jepa_stage_tokens": 0,
    "flow_jepa_stateless_goal_phase_machine": 1,
    "flow_jepa_strict_role_visual_path": 1,
    "flow_jepa_structured_ownership_bottleneck": 1,
    "flow_jepa_top_role_schedule": '3-2-3',
    "flow_jepa_utility_precision_mainline": 1,
    "flow_jepa_variance_safe_routing": 1,
    "flow_jepa_window_offsets": (4, 12, 24, 48),
    "flow_jepa_world_blocks": 2,
    "flow_jepa_zero_flow_guard": 1,
    "flow_matching_time_distribution": 'beta_1_5_1',
    "future_grid_size": 8,
    "goal_condition_dropout": 0.05,
    "goal_condition_exact_null": 1,
    "goal_conditioning_enabled": 1,
    "goal_language_dim": 4096,
    "gripper_field_dim": 6,
    "latent_cvae_layer_scan": 1,
    "latent_cvae_min_std": 0.6,
    "latent_cvae_mmdit_decoder": 1,
    "latent_cvae_mmdit_dwell_mode": 'learned',
    "latent_cvae_mmdit_dynamic_block_route": 1,
    "latent_cvae_mmdit_execution_controller": 1,
    "latent_cvae_mmdit_operator_capacity": 1,
    "latent_cvae_mmdit_operator_depth_logit_init": 2.268683541,
    "latent_cvae_mu_bound": 1.25,
    "latent_cvae_noisy_gate": 1,
    "latent_cvae_noisy_gate_min": 0.08,
    "latent_cvae_noisy_gate_power": 1.0,
    "latent_cvae_variational": 0,
    "latent_cvae_z_probe": 1,
    "layer_consequence_steps": 3,
    "layer_contract_adapters": 1,
    "layer_state_counterfactual": 1,
    "midcut_layer": 6,
    "patches_per_camera": 256,
    "role_attnres_enabled": 1,
    "role_attnres_ground_to_world": 1,
    "role_attnres_policy_to_mmdit": 1,
    "role_attnres_world_to_policy": 1,
    "role_residual_amplitude_contract": 1,
    "role_residual_contract_after_gate": 1,
    "stateless_phase_enabled": 1,
}


def build_v120_policy_config() -> V39PolicyConfig:
    """Construct and validate the exact extracted V120 model profile."""

    config = replace(V39PolicyConfig(), **V120_MODEL_OVERRIDES)
    config.validate()
    return config


def build_v120_visual_config(mainline_config: Any) -> V39PolicyConfig:
    """Resolve the extracted V120 visual compiler to one mainline shape.

    The production values remain exactly those serialized by V120.  The
    explicit shape projection exists so the small executable mainline tests
    exercise the same Flow-DINO/raw-address implementation instead of a toy
    substitute.  Activation checkpointing is a computational policy only and
    is disabled for the tiny FP32 graph; it is retained for the 512-wide
    production graph.
    """

    mainline_config.validate()
    dims = mainline_config.dimensions
    observation = mainline_config.observation
    config = replace(
        build_v120_policy_config(),
        action_dim=int(dims.action_dim),
        state_dim=int(dims.state_dim),
        action_horizon=int(dims.action_horizon),
        executed_history_length=int(dims.executed_history_length),
        hidden_size=int(dims.hidden_size),
        num_heads=int(dims.num_heads),
        visual_token_dim=int(dims.visual_token_dim),
        visual_history_length=int(dims.visual_history_length),
        num_cameras=int(dims.num_cameras),
        patches_per_camera=int(dims.patches_per_camera),
        target_future_count=int(dims.future_supports),
        action_basis_tokens=int(dims.action_basis_tokens),
        goal_language_dim=int(dims.goal_token_dim),
        flow_jepa_grid_size=int(observation.grid_size),
        flow_jepa_feature_dim=int(observation.feature_dim),
        flow_jepa_flow_iters=int(observation.flow_iterations),
        flow_jepa_corr_radius=int(observation.correlation_radius),
        flow_jepa_mask_ratio=float(observation.mask_ratio),
        flow_jepa_mask_block_size=int(observation.mask_block_size),
        flow_jepa_motion_mask_fraction=float(observation.motion_mask_fraction),
        flow_jepa_uncertainty_floor=float(observation.uncertainty_floor),
        flow_jepa_raw_base_channels=int(observation.raw_base_channels),
        flow_jepa_address_slots=int(observation.local_hypotheses),
        flow_jepa_address_route_dim=int(observation.address_route_dim),
        flow_jepa_raw_micro_grid=int(observation.microgrid_side),
        flow_jepa_raw_reader_heads=min(4, int(dims.num_heads)),
        flow_jepa_raw_activation_checkpoint=int(dims.hidden_size >= 128),
    )
    config.validate()
    return config


__all__ = [
    "V120_MODEL_OVERRIDES",
    "V120_REFERENCE_NORMALIZER_FINGERPRINT",
    "V120_REFERENCE_SOURCE_COMMIT",
    "build_v120_policy_config",
    "build_v120_visual_config",
]
