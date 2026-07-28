"""Audit ClearVLA training logs without importing the training stack.

The parser accepts the historical ``[v39-layer]`` format, the compact V94-V102
formats (including representation-only V95 Stage1), pretty-printed run
contexts, and epoch JSON/JSONL records.
Its summaries deliberately separate raw metrics, weighted objective
contributions, validation behavior, structural/controller gauges, gradients,
and observability quality.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOKEN_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_.-]*)=(?P<value>[^\s]+)")
HEADER_RE = re.compile(r"^\[(?P<name>v\d+(?:-[^\]]+)?)\]\s+(?P<body>.*)$")
INIT_COUNT_RE = re.compile(r"^\[v39-init\]\s+(?P<label>.*?)(?:\s+count=(?P<count>\d+))$")
UNHANDLED_EXCEPTION_RE = re.compile(
    r"^(?P<type>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception)):\s*(?P<message>.*)$"
)


# Only legacy slash groups that carry cross-version decision value are decoded
# here. Unknown groups remain available under their raw token name.
LEGACY_GROUPS: dict[str, tuple[str, ...]] = {
    "evexec": (
        "evidence_mmd_it_execution_progress",
        "evidence_mmd_it_capacity_ratio",
        "evidence_mmd_it_dwell_expected",
        "evidence_mmd_it_execution_cost",
    ),
    "evroute": (
        "evidence_mmd_it_dynamic_route_next_fraction",
        "evidence_mmd_it_hard_route_next_fraction",
    ),
    "evdwell": (
        "evidence_mmd_it_dwell_expected",
        "evidence_mmd_it_hard_dwell_expected",
    ),
    "evctrl": (
        "evidence_mmd_it_controller_slot_pair_cosine",
        "evidence_mmd_it_controller_slot_common_mode_ratio",
        "evidence_mmd_it_controller_slot_private_energy_ratio",
    ),
    "evval": (
        "evidence_mmd_it_execution_value_loss",
        "evidence_mmd_it_execution_value_target_spread",
        "evidence_mmd_it_execution_value_predicted_spread",
        "evidence_mmd_it_execution_value_decision_accuracy",
        "evidence_mmd_it_execution_value_common_mode_ratio",
    ),
    "evcap": (
        "evidence_mmd_it_effective_depth",
        "evidence_mmd_it_removed_channel_fraction",
        "evidence_mmd_it_nonexpansive_violation",
    ),
    "evsel": (
        "evidence_mmd_it_execution_selection_entropy",
        "evidence_mmd_it_learned_selection_entropy",
        "evidence_mmd_it_execution_selection_max_probability",
    ),
    "evgrad": (
        "grad_evidence_view_adapter",
        "grad_evidence_condition_organizer",
        "grad_evidence_mmdit_evidence_reader",
        "grad_evidence_mmdit_action_state",
        "grad_evidence_mmdit_blocks",
    ),
    "evcgrad": (
        "grad_evidence_mmdit_execution_controller",
        "grad_evidence_mmdit_operator_basis",
        "grad_evidence_mmdit_execution_value_reader",
    ),
    "anstd": ("arm_noise_abs_std", "arm_noise_delta_std"),
    "atstd": ("arm_target_abs_std", "arm_target_delta_std"),
}


LEGACY_ALIASES: dict[str, str] = {
    "loss": "loss",
    "pflow": "physical_flow",
    "pflowu": "physical_flow_uniform",
    "pfn": "physical_flow_native",
    "pfnu": "physical_flow_native_uniform",
    "afmd": "arm_fm_per_dim",
    "gfmf": "gripper_fm_field",
    "gfar": "gripper_arm_fm_ratio",
    "decode": "decoded_action",
    "rollout": "rollout_dynamics",
    "delta": "rollout_delta",
    "contrast": "rollout_contrast",
    "rvar": "rollout_variance",
    "rnorm": "rollout_norm",
    "rstep": "rollout_milestone_delta_match",
    "first8": "first8_physical_flow",
    "tail": "tail_physical_flow",
    "event": "event",
    "stdr": "rollout_pred_std_ratio",
    "dnratio": "rollout_milestone_delta_norm_ratio",
    "zzero": "latent_cvae_z_zero_delta",
    "zshuf": "latent_cvae_z_shuffle_delta",
    "grad": "grad",
    "lr": "learning_rate",
    "spb": "seconds_per_batch",
}


V94_ALIASES: dict[str, dict[str, str]] = {
    "train": {
        "loss": "loss",
        "loss_total": "loss",
        "loss_representation": "loss",
        "pflow": "physical_flow",
        "flow_loss": "physical_flow",
        "pfn": "physical_flow_native",
        "native_flow": "physical_flow_native",
        "arm": "arm_fm_per_dim",
        "arm_flow": "arm_fm_per_dim",
        "grip": "gripper_fm_field",
        "grip_flow": "gripper_fm_field",
        "decode": "decoded_action",
        "decode_loss": "decoded_action",
        "first8": "first8_physical_flow",
        "flow_first8": "first8_physical_flow",
        "tail": "tail_physical_flow",
        "flow_tail": "tail_physical_flow",
        "event": "event",
        "event_loss": "event",
        "motion": "motion",
        "motion_loss": "motion",
        "proposal": "proposal",
        "proposal_loss": "proposal",
        "roll": "rollout_dynamics",
        "rollout_loss": "rollout_dynamics",
        "rstep": "rollout_milestone_delta_match",
        "rollout_step": "rollout_milestone_delta_match",
        "contrast": "rollout_contrast",
        "rollout_contrast": "rollout_contrast",
        "stdr": "rollout_pred_std_ratio",
        "rollout_std_ratio": "rollout_pred_std_ratio",
        "dnratio": "rollout_milestone_delta_norm_ratio",
        "step_norm_ratio": "rollout_milestone_delta_norm_ratio",
        "ledger_residual": "loss_ledger_residual",
        "ledger_gap": "loss_ledger_residual",
    },
    "exec": {
        "progress": "evidence_mmd_it_execution_progress",
        "exec_progress": "evidence_mmd_it_execution_progress",
        "capacity": "evidence_mmd_it_capacity_ratio",
        "soft_capacity": "evidence_mmd_it_capacity_ratio",
        "capacity_gate_mass": "evidence_mmd_it_capacity_gate_mass",
        "depth": "evidence_mmd_it_effective_depth",
        "effective_rank": "evidence_mmd_it_effective_depth",
        "effective_basis_mass": "evidence_mmd_it_effective_basis_mass",
        "depth_ratio": "evidence_mmd_it_depth_ratio",
        "rank_ratio": "evidence_mmd_it_depth_ratio",
        "contraction": "evidence_mmd_it_contraction_ratio",
        "contraction_ratio": "evidence_mmd_it_contraction_ratio",
        "removed": "evidence_mmd_it_removed_channel_fraction",
        "soft_removed": "evidence_mmd_it_removed_channel_fraction",
        "selected_groups": "evidence_mmd_it_selected_active_group_fraction",
        "group_gate_mean": "evidence_mmd_it_selected_active_group_fraction",
        "selected_depth": "evidence_mmd_it_selected_effective_depth",
        "selected_rank": "evidence_mmd_it_selected_effective_depth",
        "cost_audit": "evidence_mmd_it_execution_cost",
        "cost_proxy": "evidence_mmd_it_execution_cost",
        "workload_audit": "evidence_mmd_it_execution_cost",
        "nonexp": "evidence_mmd_it_nonexpansive_violation",
        "nonexp_violation": "evidence_mmd_it_nonexpansive_violation",
        "selection_H": "evidence_mmd_it_execution_selection_entropy",
        "selection_entropy": "evidence_mmd_it_execution_selection_entropy",
        "selection_max": "evidence_mmd_it_execution_selection_max_probability",
        "terminal_prior": "evidence_mmd_it_terminal_prior_weight",
        "terminal_probability": "evidence_mmd_it_terminal_probability",
        "hard_terminal_fraction": "evidence_mmd_it_hard_terminal_fraction",
        "top_policy_scale": "evidence_top_policy_workspace_scale",
        "top_policy_update": "evidence_top_policy_workspace_update_norm",
        "top_policy_fixed_fusion": "evidence_top_policy_workspace_fixed_fusion",
        "operation_probability": "evidence_mmd_it_operation_probability",
        "value": "evidence_mmd_it_execution_value_loss",
        "value_loss": "evidence_mmd_it_execution_value_loss",
        "target_spread": "evidence_mmd_it_execution_value_target_spread",
        "value_target_spread": "evidence_mmd_it_execution_value_target_spread",
        "pred_spread": "evidence_mmd_it_execution_value_predicted_spread",
        "value_pred_spread": "evidence_mmd_it_execution_value_predicted_spread",
        "value_corr": "evidence_mmd_it_execution_value_correlation",
        "value_pair": "evidence_mmd_it_execution_value_pairwise_accuracy",
        "value_pair_acc": "evidence_mmd_it_execution_value_pairwise_accuracy",
        "value_decision": "evidence_mmd_it_execution_value_decision_accuracy",
        "value_top1_acc": "evidence_mmd_it_execution_value_decision_accuracy",
        "value_coverage": "evidence_mmd_it_execution_candidate_coverage",
        "candidate_coverage": "evidence_mmd_it_execution_candidate_coverage",
        "value_common": "evidence_mmd_it_execution_value_common_mode_ratio",
        "value_common_ratio": "evidence_mmd_it_execution_value_common_mode_ratio",
        "terminal_target_margin": "evidence_mmd_it_terminal_target_cost_margin",
        "terminal_pred_margin": "evidence_mmd_it_terminal_predicted_cost_margin",
        "terminal_target_preferred": "evidence_mmd_it_terminal_target_preferred_fraction",
        "terminal_identity_error": "evidence_mmd_it_terminal_identity_velocity_error",
        "layer_contract": "layer_contract",
        "layer_loss_raw": "layer_contract",
        "layer_scale": "layer_contract_aux_scale",
        "layer_contrib": "loss_contrib_layer_contract",
    },
    "grad": {
        "view": "grad_evidence_view_adapter",
        "view_adapter": "grad_evidence_view_adapter",
        "organizer": "grad_evidence_condition_organizer",
        "reader": "grad_evidence_mmdit_evidence_reader",
        "evidence_reader": "grad_evidence_mmdit_evidence_reader",
        "state": "grad_evidence_mmdit_action_state",
        "action_state": "grad_evidence_mmdit_action_state",
        "top_policy_lift": "grad_evidence_top_policy_workspace_lift",
        "blocks": "grad_evidence_mmdit_blocks",
        "mmdit_blocks": "grad_evidence_mmdit_blocks",
        "controller": "grad_evidence_mmdit_execution_controller",
        "exec_controller": "grad_evidence_mmdit_execution_controller",
        "cap_control": "grad_evidence_mmdit_capacity_control",
        "capacity_control": "grad_evidence_mmdit_capacity_control",
        "operator_cap": "grad_evidence_mmdit_operator_capacity",
        "operator_capacity": "grad_evidence_mmdit_operator_capacity",
        "basis": "grad_evidence_mmdit_operator_basis",
        "operator_basis": "grad_evidence_mmdit_operator_basis",
        "value_reader": "grad_evidence_mmdit_execution_value_reader",
        "layer_adapter": "grad_layer_contract_adapters",
        "consequence": "grad_layer_consequence_cell",
        "dynamics": "grad_controlled_dynamics",
        "dit": "grad_dit_blocks",
        "dit_blocks": "grad_dit_blocks",
        "flow_dino": "grad_flow_dino_evidence",
        "coarse_flow": "grad_flow_dino_coarse_flow",
        "fine_flow": "grad_flow_dino_sparse_fine",
        "detail_router": "grad_flow_dino_detail_router",
        "address_reader": "grad_flow_dino_address_reader",
        "future_predictor": "grad_flow_dino_future_predictor",
        "raw_pyramid": "grad_flow_dino_raw_pyramid",
        "raw_coarse_flow": "grad_flow_dino_raw_coarse_flow",
        "semantic_coarse_flow": "grad_flow_dino_semantic_coarse_flow",
        "raw_mid_flow": "grad_flow_dino_raw_mid_flow",
        "raw_high_flow": "grad_flow_dino_raw_high_flow",
        "raw_detail_router": "grad_flow_dino_raw_detail_router",
        "raw_address_reader": "grad_flow_dino_raw_address_reader",
        "late_detail_reader": "grad_late_raw_detail_reader",
        "grounding_blocks": "grad_dit_grounding_blocks",
        "world_blocks": "grad_dit_world_blocks",
        "policy_blocks": "grad_dit_policy_blocks",
        "goal_tokens": "grad_goal_resampler",
        "action_history": "grad_action_history_encoder",
        "heads": "grad_final_policy_heads",
        "policy_heads": "grad_final_policy_heads",
        "global": "grad",
        "global_preclip": "grad",
        "lr": "learning_rate",
        "spb": "seconds_per_batch",
        "sec_per_batch": "seconds_per_batch",
    },
    "val": {
        "jepa_window": "flow_jepa_future_prediction",
        "jepa_future": "flow_jepa_future_prediction",
        "jepa_change": "flow_jepa_future_change_direction",
        "change_obj": "flow_jepa_future_change",
        "jepa_stage": "flow_jepa_stage_prediction",
        "patch_warp": "flow_jepa_warp_loss",
        "identity_adv": "flow_jepa_identity_advantage_loss",
        "static_identity": "flow_jepa_static_identity_loss",
        "patch_cycle": "flow_jepa_cycle_loss",
        "patch_flow": "flow_jepa_patch_flow_magnitude",
        "patch_conf": "flow_jepa_confidence_mean",
        "patch_occ": "flow_jepa_occlusion_fraction",
        "raw_high_grid": "flow_jepa_raw_high_grid_size",
        "raw_flow": "flow_jepa_raw_flow_magnitude",
        "raw_flow_grid": "flow_jepa_raw_flow_grid_magnitude",
        "seed_reliability": "flow_jepa_raw_seed_reliability",
        "raw_boundary": "flow_jepa_raw_boundary_penalty",
        "raw_valid": "flow_jepa_raw_valid_fraction",
        "zero_warp": "flow_jepa_raw_identity_warp_error",
        "warp_gain": "flow_jepa_raw_warp_gain_over_zero",
        "moving_gain": "flow_jepa_raw_moving_warp_gain",
        "static_gain": "flow_jepa_raw_static_warp_gain",
        "moving_corr_entropy": "flow_jepa_raw_moving_correlation_entropy",
        "moving_corr_margin": "flow_jepa_raw_moving_correlation_margin",
        "motion_visible": "flow_jepa_raw_observable_motion_fraction",
        "raw_detail_gate": "flow_jepa_raw_detail_gate_mean",
        "raw_emphasis": "flow_jepa_raw_detail_emphasis_mean",
        "raw_precision": "flow_jepa_raw_detail_precision_mean",
        "raw_address_flow": "flow_jepa_raw_address_flow_mass",
        "raw_detail_share": "flow_jepa_raw_address_flow_mass",
        "raw_address_fallback": "flow_jepa_raw_address_fallback_mass",
        "raw_base_share": "flow_jepa_raw_address_fallback_mass",
        "detail_address_entropy": "flow_jepa_raw_address_entropy",
        "address_separation": "flow_jepa_raw_address_center_separation",
        "address_value_delta": "flow_jepa_raw_address_lane_value_difference",
        "address_logit_gain": "flow_jepa_raw_address_logit_advantage",
        "detail_address_concentration": "flow_jepa_raw_address_logit_advantage",
        "address_zero_delta": "flow_jepa_raw_address_zero_flow_value_delta",
        "address_shuffle_delta": "flow_jepa_raw_address_shuffled_flow_value_delta",
        "world_xy_residual": "flow_jepa_world_spatial_residual_norm",
        "world_anchor_residual": "flow_jepa_world_anchor_camera_residual_norm",
        "late_detail_entropy": "flow_jepa_late_detail_attention_entropy",
        "late_detail_max": "flow_jepa_late_detail_attention_max",
        "late_detail_update": "flow_jepa_late_detail_update_norm",
        "late_detail_ratio": "flow_jepa_late_detail_trajectory_ratio",
        "late_detail_scale": "flow_jepa_late_detail_fixed_scale",
        "late_detail_tokens": "flow_jepa_late_detail_token_count",
        "horizon_cos": "flow_jepa_horizon_adjacent_cosine",
        "rmse": "full_rmse",
        "action_rmse": "full_rmse",
        "first": "first_rmse",
        "first_rmse": "first_rmse",
        "first8": "first8_rmse",
        "first8_rmse": "first8_rmse",
        "tail": "tail_rmse",
        "tail_rmse": "tail_rmse",
        "tail_first": "tail_first_ratio",
        "tail_first_ratio": "tail_first_ratio",
        "arm": "arm_full_rmse",
        "arm_rmse": "arm_full_rmse",
        "grip": "gripper_full_rmse",
        "grip_rmse": "gripper_full_rmse",
        "event_ratio": "gripper_event_ratio",
        "grip_event_ratio": "gripper_event_ratio",
        "gripper_pred_events": "gripper_pred_events",
        "grip_events_pred": "gripper_pred_events",
        "gripper_target_events": "gripper_target_events",
        "grip_events_target": "gripper_target_events",
        "event_head_pred_events": "event_head_pred_events",
        "event_head_events_pred": "event_head_pred_events",
        "event_head_target_events": "event_head_target_events",
        "event_head_events_target": "event_head_target_events",
        "event_head_minus_decoded_f1": "event_head_minus_decoded_gripper_f1",
        "proposal_gain": "proposal_utility_mse_gain",
        "proposal_mse_gain": "proposal_utility_mse_gain",
        "proposal_coverage": "eval_proposal_ablation_coverage",
        "proposal_batch_cov": "eval_proposal_ablation_coverage",
        "score": "balanced_score",
        "balanced_score": "balanced_score",
        "deploy": "deploy_eligible",
        "deploy_gate": "deploy_eligible",
    },
    "probe": {
        "z_zero": "sample_evidence_z_zero_condition_delta",
        "z_zero_cond_delta": "sample_evidence_z_zero_condition_delta",
        "z_shuffle": "sample_evidence_z_shuffle_condition_delta",
        "z_shuffle_cond_delta": "sample_evidence_z_shuffle_condition_delta",
        "capacity": "sample_evidence_mmd_it_capacity_ratio",
        "soft_capacity": "sample_evidence_mmd_it_capacity_ratio",
        "depth": "sample_evidence_mmd_it_effective_depth",
        "effective_rank": "sample_evidence_mmd_it_effective_depth",
        "depth_ratio": "sample_evidence_mmd_it_depth_ratio",
        "rank_ratio": "sample_evidence_mmd_it_depth_ratio",
        "removed": "sample_evidence_mmd_it_removed_channel_fraction",
        "soft_removed": "sample_evidence_mmd_it_removed_channel_fraction",
        "selected_groups": "sample_evidence_mmd_it_selected_active_group_fraction",
        "group_gate_mean": "sample_evidence_mmd_it_selected_active_group_fraction",
        "selected_depth": "sample_evidence_mmd_it_selected_effective_depth",
        "selected_rank": "sample_evidence_mmd_it_selected_effective_depth",
        "route_soft": "sample_evidence_mmd_it_dynamic_route_next_fraction",
        "route_hard": "sample_evidence_mmd_it_hard_route_next_fraction",
        "dwell_soft": "sample_evidence_mmd_it_dwell_expected",
        "dwell_hard": "sample_evidence_mmd_it_hard_dwell_expected",
        "nonexp": "sample_evidence_mmd_it_nonexpansive_violation",
        "nonexp_violation": "sample_evidence_mmd_it_nonexpansive_violation",
        "sample_coverage": "eval_sampling_diagnostic_coverage",
        "probe_batch_cov": "eval_sampling_diagnostic_coverage",
        "capacity_gate_mass": "sample_evidence_mmd_it_capacity_gate_mass",
        "effective_basis_mass": "sample_evidence_mmd_it_effective_basis_mass",
        "terminal_prior": "sample_evidence_mmd_it_terminal_prior_weight",
        "terminal_probability": "sample_evidence_mmd_it_terminal_probability",
        "hard_terminal_fraction": "sample_evidence_mmd_it_hard_terminal_fraction",
        "late_detail_update": "sample_flow_jepa_late_detail_update_norm",
        "late_detail_ratio": "sample_flow_jepa_late_detail_trajectory_ratio",
        "late_detail_entropy": "sample_flow_jepa_late_detail_attention_entropy",
        "late_detail_max": "sample_flow_jepa_late_detail_attention_max",
        "late_detail_scale": "sample_flow_jepa_late_detail_fixed_scale",
        "late_detail_tokens": "sample_flow_jepa_late_detail_token_count",
        "world_xy_residual": "sample_flow_jepa_world_spatial_residual_norm",
        "world_anchor_residual": "sample_flow_jepa_world_anchor_camera_residual_norm",
    },
    "balance": {
        "flow_without_info_balance": "physical_flow_no_information_balance",
        "trajectory_info": "trajectory_information_score",
        "info_weight_min": "trajectory_information_weight_min",
        "info_weight_max": "trajectory_information_weight_max",
        "info_effective_fraction": "trajectory_information_effective_fraction",
        "horizon_weight_first": "action_horizon_weight_first",
        "horizon_weight_tail": "action_horizon_weight_tail",
        "history_keep": "condition_action_history_keep",
        "goal_keep": "condition_goal_keep",
        "proposal_keep": "condition_proposal_keep",
        "teacher_past_quota": "flow_jepa_teacher_mask_past_fraction",
        "teacher_change_quota": "flow_jepa_teacher_mask_change_fraction",
        "teacher_uniform_quota": "flow_jepa_teacher_mask_uniform_fraction",
        "selected_change_ratio": "flow_jepa_teacher_mask_selected_change_ratio",
        "action_h1_4": "action_band_1_4_physical_flow",
        "action_h5_12": "action_band_5_12_physical_flow",
        "action_h13_24": "action_band_13_24_physical_flow",
    },
    "repr": {
        "window_pred": "flow_jepa_future_prediction",
        "future_pred": "flow_jepa_future_prediction",
        "change_dir": "flow_jepa_future_change_direction",
        "change_obj": "flow_jepa_future_change",
        "stage_pred": "flow_jepa_stage_prediction",
        "warp": "flow_jepa_warp_loss",
        "identity_adv": "flow_jepa_identity_advantage_loss",
        "static_identity": "flow_jepa_static_identity_loss",
        "cycle": "flow_jepa_cycle_loss",
        "smooth": "flow_jepa_smoothness_loss",
        "uncert_nll": "flow_jepa_uncertainty_nll",
        "refine_seq": "flow_jepa_refinement_sequence_loss",
        "flow_mag": "flow_jepa_patch_flow_magnitude",
        "confidence": "flow_jepa_confidence_mean",
        "occlusion": "flow_jepa_occlusion_fraction",
        "corr_entropy": "flow_jepa_correlation_entropy",
        "corr_margin": "flow_jepa_correlation_margin",
        "context_drop": "flow_jepa_context_dropout_fraction",
        "target_mask": "flow_jepa_future_target_fraction",
        "window_hmax": "flow_jepa_window_horizon_max",
        "stage_h": "flow_jepa_stage_horizon",
        "stage_norm": "flow_jepa_stage_token_norm",
        "stage_target_norm": "flow_jepa_stage_target_norm",
        "stage_prediction_norm": "flow_jepa_stage_prediction_norm",
        "stage_window_cos": "flow_jepa_stage_window_cosine",
        "stage_window_gate": "flow_jepa_stage_to_window_gate",
        "stage_window_update": "flow_jepa_stage_to_window_update_norm",
        "goal_norm": "flow_jepa_goal_condition_norm",
        "goal_pair_cos": "flow_jepa_goal_pair_cosine",
        "action_mem_norm": "flow_jepa_action_condition_norm",
        "goal_action_cos": "flow_jepa_goal_action_cosine",
        "repr_batch_cov": "eval_representation_coverage",
        "horizon_count": "flow_jepa_horizon_count",
        "horizon_max": "flow_jepa_horizon_max",
        "native_grid": "flow_jepa_native_grid_size",
        "coarse_grid": "flow_jepa_coarse_grid_size",
        "dino_grid": "flow_jepa_native_grid_size",
        "reader_grid": "flow_jepa_coarse_grid_size",
        "native_flow": "flow_jepa_native_flow_magnitude",
        "detail_gate": "flow_jepa_detail_gate_mean",
        "detail_gate_mean": "flow_jepa_detail_gate_mean",
        "detail_cmp": "flow_jepa_detail_effective_comparisons",
        "detail_weighted_cmp": "flow_jepa_detail_effective_comparisons",
        "detail_candidates": "flow_jepa_detail_candidate_comparisons",
        "detail_candidate_cmp": "flow_jepa_detail_candidate_comparisons",
        "address_flow": "flow_jepa_address_flow_mass",
        "address_flow_mass": "flow_jepa_address_flow_mass",
        "address_fallback": "flow_jepa_address_fallback_mass",
        "address_fallback_mass": "flow_jepa_address_fallback_mass",
        "address_entropy": "flow_jepa_address_entropy",
        "horizon_cos": "flow_jepa_horizon_adjacent_cosine",
        "horizon_adj_cos": "flow_jepa_horizon_adjacent_cosine",
        "far_norm": "flow_jepa_far_horizon_norm",
        "far_horizon_norm": "flow_jepa_far_horizon_norm",
        "raw_high_grid": "flow_jepa_raw_high_grid_size",
        "raw_mid_grid": "flow_jepa_raw_mid_grid_size",
        "raw_coarse_grid": "flow_jepa_raw_coarse_grid_size",
        "raw_flow": "flow_jepa_raw_flow_magnitude",
        "raw_flow_grid": "flow_jepa_raw_flow_grid_magnitude",
        "seed_reliability": "flow_jepa_raw_seed_reliability",
        "mid_residual": "flow_jepa_raw_mid_residual_magnitude",
        "high_residual": "flow_jepa_raw_high_residual_magnitude",
        "raw_cycle_core": "flow_jepa_raw_cycle_core",
        "raw_boundary": "flow_jepa_raw_boundary_penalty",
        "raw_valid": "flow_jepa_raw_valid_fraction",
        "zero_warp": "flow_jepa_raw_identity_warp_error",
        "warp_gain": "flow_jepa_raw_warp_gain_over_zero",
        "moving_gain": "flow_jepa_raw_moving_warp_gain",
        "static_gain": "flow_jepa_raw_static_warp_gain",
        "moving_corr_entropy": "flow_jepa_raw_moving_correlation_entropy",
        "moving_corr_margin": "flow_jepa_raw_moving_correlation_margin",
        "motion_visible": "flow_jepa_raw_observable_motion_fraction",
        "raw_conf": "flow_jepa_raw_confidence_mean",
        "raw_occ": "flow_jepa_raw_occlusion_fraction",
        "raw_detail_gate": "flow_jepa_raw_detail_gate_mean",
        "raw_emphasis": "flow_jepa_raw_detail_emphasis_mean",
        "raw_precision": "flow_jepa_raw_detail_precision_mean",
        "raw_address_flow": "flow_jepa_raw_address_flow_mass",
        "raw_detail_share": "flow_jepa_raw_address_flow_mass",
        "raw_address_fallback": "flow_jepa_raw_address_fallback_mass",
        "raw_base_share": "flow_jepa_raw_address_fallback_mass",
        "raw_address_entropy": "flow_jepa_raw_address_entropy",
        "detail_address_entropy": "flow_jepa_raw_address_entropy",
        "address_separation": "flow_jepa_raw_address_center_separation",
        "address_value_delta": "flow_jepa_raw_address_lane_value_difference",
        "address_logit_gain": "flow_jepa_raw_address_logit_advantage",
        "detail_address_concentration": "flow_jepa_raw_address_logit_advantage",
        "address_zero_delta": "flow_jepa_raw_address_zero_flow_value_delta",
        "address_shuffle_delta": "flow_jepa_raw_address_shuffled_flow_value_delta",
        "raw_candidates": "flow_jepa_raw_candidates_per_cell",
        "raw_detail_tokens": "flow_jepa_raw_detail_token_count",
        "raw_dino_fused": "flow_jepa_raw_detail_fused_with_latest_dino",
        "raw_source_dino_fused": "flow_jepa_raw_detail_fused_with_source_dino",
        "world_xy_residual": "flow_jepa_world_spatial_residual_norm",
        "world_anchor_residual": "flow_jepa_world_anchor_camera_residual_norm",
        "late_detail_entropy": "flow_jepa_late_detail_attention_entropy",
        "late_detail_max": "flow_jepa_late_detail_attention_max",
        "late_detail_update": "flow_jepa_late_detail_update_norm",
        "late_detail_ratio": "flow_jepa_late_detail_trajectory_ratio",
        "late_detail_scale": "flow_jepa_late_detail_fixed_scale",
        "late_detail_tokens": "flow_jepa_late_detail_token_count",
        "refined_visual_tokens": "flow_jepa_refined_evidence_token_count",
        "grounding_blocks": "flow_jepa_grounding_block_count",
        "world_blocks": "flow_jepa_world_block_count",
        "policy_blocks": "flow_jepa_policy_block_count",
    },
}


CORE_BATCH_KEYS = (
    "loss",
    "flow_jepa_future_prediction",
    "flow_jepa_future_change",
    "flow_jepa_future_change_direction",
    "flow_jepa_stage_prediction",
    "flow_jepa_warp_loss",
    "flow_jepa_identity_advantage_loss",
    "flow_jepa_static_identity_loss",
    "flow_jepa_cycle_loss",
    "physical_flow",
    "physical_flow_native",
    "arm_fm_per_dim",
    "gripper_fm_field",
    "decoded_action",
    "first8_physical_flow",
    "tail_physical_flow",
    "rollout_dynamics",
    "rollout_milestone_delta_match",
    "event",
)

STRUCTURE_KEYS = (
    "evidence_mmd_it_execution_progress",
    "evidence_mmd_it_capacity_ratio",
    "evidence_mmd_it_capacity_gate_mass",
    "evidence_mmd_it_effective_depth",
    "evidence_mmd_it_effective_basis_mass",
    "evidence_mmd_it_selected_effective_depth",
    "evidence_mmd_it_removed_channel_fraction",
    "evidence_mmd_it_dwell_expected",
    "evidence_mmd_it_hard_dwell_expected",
    "evidence_mmd_it_dynamic_route_next_fraction",
    "evidence_mmd_it_hard_route_next_fraction",
    "evidence_mmd_it_execution_selection_entropy",
    "evidence_mmd_it_execution_selection_max_probability",
    "evidence_mmd_it_execution_value_target_spread",
    "evidence_mmd_it_execution_value_predicted_spread",
    "evidence_mmd_it_execution_value_correlation",
    "evidence_mmd_it_execution_value_pairwise_accuracy",
    "evidence_mmd_it_execution_value_decision_accuracy",
    "evidence_mmd_it_execution_value_common_mode_ratio",
    "evidence_mmd_it_terminal_prior_weight",
    "evidence_mmd_it_terminal_probability",
    "evidence_mmd_it_hard_terminal_fraction",
    "evidence_mmd_it_terminal_target_cost_margin",
    "evidence_mmd_it_terminal_predicted_cost_margin",
    "evidence_mmd_it_terminal_target_preferred_fraction",
    "evidence_mmd_it_terminal_identity_velocity_error",
    "flow_jepa_world_spatial_residual_norm",
    "flow_jepa_world_anchor_camera_residual_norm",
    "flow_jepa_late_detail_attention_entropy",
    "flow_jepa_late_detail_attention_max",
    "flow_jepa_late_detail_update_norm",
    "flow_jepa_late_detail_trajectory_ratio",
    "flow_jepa_late_detail_fixed_scale",
    "flow_jepa_late_detail_token_count",
)

GRADIENT_KEYS = (
    "grad_evidence_view_adapter",
    "grad_evidence_condition_organizer",
    "grad_evidence_mmdit_evidence_reader",
    "grad_evidence_mmdit_action_state",
    "grad_evidence_mmdit_blocks",
    "grad_evidence_mmdit_execution_controller",
    "grad_evidence_mmdit_capacity_control",
    "grad_evidence_mmdit_operator_capacity",
    "grad_evidence_mmdit_operator_basis",
    "grad_evidence_mmdit_execution_value_reader",
    "grad_layer_contract_adapters",
    "grad_layer_consequence_cell",
    "grad_controlled_dynamics",
    "grad_dit_blocks",
    "grad_flow_dino_evidence",
    "grad_flow_dino_coarse_flow",
    "grad_flow_dino_sparse_fine",
    "grad_flow_dino_detail_router",
    "grad_flow_dino_address_reader",
    "grad_flow_dino_future_predictor",
    "grad_late_raw_detail_reader",
    "grad_goal_resampler",
    "grad_action_history_encoder",
    "grad_final_policy_heads",
    "grad",
)

VALIDATION_KEYS = (
    "loss",
    "flow_jepa_future_prediction",
    "flow_jepa_stage_prediction",
    "flow_jepa_warp_loss",
    "flow_jepa_cycle_loss",
    "flow_jepa_stage_window_cosine",
    "flow_jepa_goal_pair_cosine",
    "eval_representation_coverage",
    "full_rmse",
    "first_rmse",
    "first8_rmse",
    "tail_rmse",
    "tail_first_ratio",
    "arm_full_rmse",
    "arm_tail_rmse",
    "gripper_full_rmse",
    "gripper_precision",
    "gripper_recall",
    "gripper_f1",
    "gripper_event_ratio",
    "event_head_accuracy",
    "event_head_precision",
    "event_head_recall",
    "event_head_f1",
    "event_head_minus_decoded_gripper_f1",
    "motion_head_precision",
    "motion_head_recall",
    "motion_head_f1",
    "proposal_utility_mse_gain",
    "eval_proposal_ablation_coverage",
    "eval_sampling_diagnostic_coverage",
    "eval_execution_ablation_coverage",
    "execution_ablation_primary_full_rmse",
    "execution_ablation_hard_full_rmse",
    "execution_ablation_neutral_full_rmse",
    "execution_ablation_full_capacity_full_rmse",
    "execution_ablation_three_basis_reduction_full_rmse",
    "balanced_score",
    "deploy_eligible",
    "sample_evidence_z_zero_condition_delta",
    "sample_evidence_z_shuffle_condition_delta",
)


@dataclass
class BatchPoint:
    epoch: int
    batch: int
    metrics: dict[str, float]
    source: str


@dataclass
class Finding:
    severity: str
    category: str
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRun:
    path: Path
    label: str
    headers: list[str] = field(default_factory=list)
    header_config: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    init_counts: dict[str, int] = field(default_factory=dict)
    batch_points: list[BatchPoint] = field(default_factory=list)
    epoch_records: list[dict[str, Any]] = field(default_factory=list)
    malformed_json: int = 0
    unclosed_json: int = 0
    traceback_count: int = 0
    fatal_errors: list[str] = field(default_factory=list)


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).rstrip(",;"))
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _relative_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or abs(first) <= 1e-12:
        return None
    return (last - first) / abs(first)


def _format_number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _scan_json_state(
    text: str, depth: int, in_string: bool, escaped: bool
) -> tuple[int, bool, bool]:
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
    return depth, in_string, escaped


def _parse_header_body(body: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for match in TOKEN_RE.finditer(body):
        key = match.group("key").replace("-", "_")
        raw = match.group("value").rstrip(",;")
        numeric = _number(raw)
        values[key] = numeric if numeric is not None else raw
    return values


def _parse_legacy_tokens(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in TOKEN_RE.finditer(line):
        key = match.group("key")
        raw = match.group("value").rstrip(",;")
        if "/" in raw and key in LEGACY_GROUPS:
            parts = raw.split("/")
            for name, part in zip(LEGACY_GROUPS[key], parts, strict=False):
                value = _number(part)
                if value is not None:
                    metrics[name] = value
            continue
        value = _number(raw)
        if value is not None:
            metrics[LEGACY_ALIASES.get(key, key)] = value
    return metrics


def _parse_v94_tokens(line: str, family: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    aliases = V94_ALIASES[family]
    for match in TOKEN_RE.finditer(line):
        key = match.group("key")
        raw = match.group("value").rstrip(",;")
        if key in {"lossgrp", "loss_groups", "losscontrib", "top_contrib", "contrib"}:
            for item in raw.split("/"):
                if ":" not in item:
                    continue
                name, value_raw = item.split(":", 1)
                value = _number(value_raw)
                if value is not None:
                    prefix = "loss_group_" if key in {"lossgrp", "loss_groups"} else "loss_contrib_"
                    metrics[f"{prefix}{name}"] = value
            continue
        if family == "val" and key in {
            "gripper",
            "grip_event",
            "event_head",
            "motion_head",
        }:
            prefix = "gripper" if key in {"gripper", "grip_event"} else key
            component_names = {"p": "precision", "r": "recall", "f1": "f1"}
            for item in raw.split("/"):
                if ":" not in item:
                    continue
                name, value_raw = item.split(":", 1)
                value = _number(value_raw)
                if value is not None and name in component_names:
                    metrics[f"{prefix}_{component_names[name]}"] = value
            continue
        if family == "val" and key == "action_band_rmse":
            for item in raw.split("/"):
                if ":" not in item:
                    continue
                name, value_raw = item.split(":", 1)
                value = _number(value_raw)
                if value is not None:
                    metrics[f"action_band_{name}_rmse"] = value
            continue
        if key in {"route", "dwell"} and "soft:" in raw:
            parsed: dict[str, float] = {}
            for item in raw.split("/"):
                if ":" not in item:
                    continue
                name, value_raw = item.split(":", 1)
                value = _number(value_raw)
                if value is not None:
                    parsed[name] = value
            if key == "route":
                soft_key = "evidence_mmd_it_dynamic_route_next_fraction"
                hard_key = "evidence_mmd_it_hard_route_next_fraction"
                gap_key = "evidence_mmd_it_route_soft_hard_gap"
            else:
                soft_key = "evidence_mmd_it_dwell_expected"
                hard_key = "evidence_mmd_it_hard_dwell_expected"
                gap_key = "evidence_mmd_it_dwell_soft_hard_gap"
            if "soft" in parsed:
                metrics[soft_key] = parsed["soft"]
            if "hard" in parsed:
                metrics[hard_key] = parsed["hard"]
            if "gap" in parsed:
                metrics[gap_key] = parsed["gap"]
            continue
        value = _number(raw)
        if value is not None:
            canonical = aliases.get(key, key)
            if key.startswith("future_h") and key.removeprefix("future_h").isdigit():
                canonical = f"flow_jepa_future_horizon_{key.removeprefix('future_h')}"
            metrics[canonical] = value
    return metrics


def _ingest_json(run: ParsedRun, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if "epoch" in payload and any(key in payload for key in ("train", "val", "validation")):
        run.epoch_records.append(payload)
        return
    if any(key in payload for key in ("schema", "args", "trainer", "policy_model")):
        run.context = payload


def _dedupe_epoch_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    order: list[tuple[Any, Any]] = []
    for record in records:
        key = (record.get("epoch"), record.get("global_step"))
        if key not in merged:
            merged[key] = dict(record)
            order.append(key)
            continue
        target = merged[key]
        for name, value in record.items():
            if isinstance(value, Mapping) and isinstance(target.get(name), Mapping):
                section = dict(target[name])
                section.update(value)
                target[name] = section
            elif value is not None:
                target[name] = value
    return [merged[key] for key in order]


def parse_log(path: Path, *, label: str | None = None) -> ParsedRun:
    run = ParsedRun(path=path, label=label or path.parent.name or path.stem)
    pending_v94: BatchPoint | None = None
    pending_epoch: dict[str, Any] | None = None
    json_buffer: list[str] | None = None
    json_depth = 0
    json_in_string = False
    json_escaped = False
    traceback_active = False

    def flush_v94() -> None:
        nonlocal pending_v94
        if pending_v94 is not None:
            run.batch_points.append(pending_v94)
            pending_v94 = None

    def flush_epoch() -> None:
        nonlocal pending_epoch
        if pending_epoch is not None:
            run.epoch_records.append(pending_epoch)
            pending_epoch = None

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if json_buffer is not None:
                json_buffer.append(raw_line)
                json_depth, json_in_string, json_escaped = _scan_json_state(
                    raw_line, json_depth, json_in_string, json_escaped
                )
                if json_depth <= 0 and not json_in_string:
                    text = "".join(json_buffer)
                    try:
                        _ingest_json(run, json.loads(text))
                    except json.JSONDecodeError:
                        run.malformed_json += 1
                    json_buffer = None
                continue
            if not line:
                continue
            if line.startswith("Traceback (most recent call last):"):
                run.traceback_count += 1
                traceback_active = True
                continue
            if traceback_active:
                exception_match = UNHANDLED_EXCEPTION_RE.match(line)
                if exception_match:
                    error_text = (
                        f"{exception_match.group('type')}: "
                        f"{exception_match.group('message')}"
                    )
                    run.fatal_errors.append(error_text[:1000])
                    traceback_active = False
            if line.startswith("{"):
                flush_epoch()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    json_buffer = [raw_line]
                    json_depth, json_in_string, json_escaped = _scan_json_state(
                        raw_line, 0, False, False
                    )
                else:
                    _ingest_json(run, payload)
                continue
            if line.startswith("[v39-layer]"):
                flush_v94()
                metrics = _parse_legacy_tokens(line)
                epoch = int(metrics.pop("epoch", 0.0))
                batch = int(metrics.pop("batch", 0.0))
                if epoch or batch:
                    run.batch_points.append(BatchPoint(epoch, batch, metrics, "v39-layer"))
                continue
            if line.startswith(
                (
                    "[v94-train]",
                    "[v95-train]",
                    "[v96-train]",
                    "[v97-train]",
                    "[v98-train]",
                    "[v99-train]",
                    "[v100-train]",
                    "[v101-train]",
                    "[v102-train]",
                    "[v95-stage1-train]",
                )
            ):
                flush_v94()
                metrics = _parse_v94_tokens(line, "train")
                epoch = int(metrics.pop("epoch", 0.0))
                batch = int(metrics.pop("batch", 0.0))
                if line.startswith("[v95-stage1-train]") and "loss" in metrics:
                    metrics.setdefault("loss_group_representation", metrics["loss"])
                    source = "v95-stage1"
                else:
                    source = (
                        "v102"
                        if line.startswith("[v102-")
                        else "v101"
                        if line.startswith("[v101-")
                        else "v100"
                        if line.startswith("[v100-")
                        else "v99"
                        if line.startswith("[v99-")
                        else "v98"
                        if line.startswith("[v98-")
                        else "v97"
                        if line.startswith("[v97-")
                        else "v96"
                        if line.startswith("[v96-")
                        else "v95"
                        if line.startswith("[v95-")
                        else "v94"
                    )
                pending_v94 = BatchPoint(epoch, batch, metrics, source)
                continue
            if line.startswith(
                (
                    "[v94-exec]",
                    "[v95-exec]",
                    "[v96-exec]",
                    "[v97-exec]",
                    "[v98-exec]",
                    "[v99-exec]",
                    "[v100-exec]",
                    "[v101-exec]",
                    "[v102-exec]",
                )
            ) and pending_v94 is not None:
                pending_v94.metrics.update(_parse_v94_tokens(line, "exec"))
                continue
            if line.startswith(
                (
                    "[v94-grad]",
                    "[v95-grad]",
                    "[v96-grad]",
                    "[v97-grad]",
                    "[v98-grad]",
                    "[v99-grad]",
                    "[v100-grad]",
                    "[v101-grad]",
                    "[v102-grad]",
                    "[v95-stage1-grad]",
                )
            ) and (
                pending_v94 is not None
            ):
                pending_v94.metrics.update(_parse_v94_tokens(line, "grad"))
                continue
            if line.startswith(
                (
                    "[v95-repr]",
                    "[v96-repr]",
                    "[v97-repr]",
                    "[v98-repr]",
                    "[v99-repr]",
                    "[v100-repr]",
                    "[v101-repr]",
                    "[v102-repr]",
                    "[v95-stage1-repr]",
                )
            ) and (
                pending_v94 is not None
            ):
                pending_v94.metrics.update(_parse_v94_tokens(line, "repr"))
                continue
            if line.startswith(("[v101-balance]", "[v102-balance]")) and pending_v94 is not None:
                pending_v94.metrics.update(_parse_v94_tokens(line, "balance"))
                continue
            if line.startswith("[v95-stage1-epoch]"):
                flush_epoch()
                metrics = _parse_v94_tokens(line, "repr")
                epoch = int(metrics.pop("epoch", 0.0))
                global_step = int(metrics.pop("step", 0.0))
                train_loss = metrics.pop("train_representation", None)
                val_loss = metrics.pop("val_representation", None)
                train_metrics: dict[str, float] = {}
                if train_loss is not None:
                    train_metrics["loss"] = train_loss
                    train_metrics["loss_group_representation"] = train_loss
                val_metrics = dict(metrics)
                if val_loss is not None:
                    val_metrics["loss"] = val_loss
                pending_epoch = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                continue
            if line.startswith(
                (
                    "[v94-epoch]",
                    "[v95-epoch]",
                    "[v96-epoch]",
                    "[v97-epoch]",
                    "[v98-epoch]",
                    "[v99-epoch]",
                    "[v100-epoch]",
                    "[v101-epoch]",
                    "[v102-epoch]",
                )
            ):
                flush_epoch()
                metrics = _parse_v94_tokens(line, "train")
                epoch = int(metrics.pop("epoch", 0.0))
                global_step = int(metrics.pop("step", 0.0))
                pending_epoch = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train": metrics,
                    "val": {},
                }
                continue
            if line.startswith(
                (
                    "[v94-val]",
                    "[v95-val]",
                    "[v96-val]",
                    "[v97-val]",
                    "[v98-val]",
                    "[v99-val]",
                    "[v100-val]",
                    "[v101-val]",
                    "[v102-val]",
                )
            ) and pending_epoch is not None:
                pending_epoch["val"].update(_parse_v94_tokens(line, "val"))
                continue
            if line.startswith(
                (
                    "[v94-probe]",
                    "[v95-probe]",
                    "[v96-probe]",
                    "[v97-probe]",
                    "[v98-probe]",
                    "[v99-probe]",
                    "[v100-probe]",
                    "[v101-probe]",
                    "[v102-probe]",
                )
            ) and pending_epoch is not None:
                pending_epoch["val"].update(_parse_v94_tokens(line, "probe"))
                continue
            init_match = INIT_COUNT_RE.match(line)
            if init_match:
                run.init_counts[init_match.group("label")] = int(init_match.group("count"))
                continue
            header_match = HEADER_RE.match(line)
            if header_match and not line.startswith(
                (
                    "[v39-init]",
                    "[v94-",
                    "[v95-",
                    "[v96-",
                    "[v97-",
                    "[v98-",
                    "[v99-",
                    "[v100-",
                    "[v101-",
                    "[v102-",
                )
            ):
                run.headers.append(line)
                config = _parse_header_body(header_match.group("body"))
                for key, value in config.items():
                    run.header_config.setdefault(key, value)
    flush_v94()
    flush_epoch()
    if json_buffer is not None:
        run.unclosed_json += 1
    run.epoch_records = _dedupe_epoch_records(run.epoch_records)
    return run


def _numeric_section(record: Mapping[str, Any], name: str) -> dict[str, float]:
    section = record.get(name)
    if not isinstance(section, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in section.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _train_section(record: Mapping[str, Any]) -> dict[str, float]:
    return _numeric_section(record, "train")


def _val_section(record: Mapping[str, Any]) -> dict[str, float]:
    for name in ("val", "validation", "eval", "evaluation"):
        section = _numeric_section(record, name)
        if section:
            return section
    return {}


def _config_value(run: ParsedRun, *keys: str) -> Any:
    normalized = {key.replace("-", "_") for key in keys}
    for section_name in ("trainer", "args", "policy_model", "performance_contract"):
        section = run.context.get(section_name)
        if isinstance(section, Mapping):
            for key in normalized:
                if key in section:
                    return section[key]
    for key in normalized:
        if key in run.header_config:
            return run.header_config[key]
    return None


def _series(run: ParsedRun, key: str, *, minimum_progress: float | None = None) -> list[float]:
    result: list[float] = []
    for point in run.batch_points:
        if minimum_progress is not None:
            progress = point.metrics.get("evidence_mmd_it_execution_progress")
            if progress is None or progress < minimum_progress:
                continue
        value = point.metrics.get(key)
        if value is not None and math.isfinite(value):
            result.append(value)
    return result


def _window_stats(run: ParsedRun, key: str, tail: int) -> dict[str, Any]:
    values = _series(run, key)
    if not values:
        return {"count": 0, "first": None, "last": None, "tail_median": None, "change": None}
    window = max(1, min(tail, max(len(values) // 4, 1)))
    first = _median(values[:window])
    last = _median(values[-window:])
    return {
        "count": len(values),
        "first": first,
        "last": last,
        "tail_median": _median(values[-tail:]),
        "change": _relative_change(first, last),
    }


LOSS_COMPONENTS: tuple[tuple[str, str, str | None, str], ...] = (
    ("flow", "physical_flow", None, "action"),
    ("proposal", "proposal", "proposal_loss_weight", "action"),
    ("event", "event", "event_loss_weight", "action"),
    ("motion", "motion", "arm_motion_loss_weight", "action"),
    ("gripper_transition", "transition_l1", "gripper_transition_l1_weight", "action"),
    ("smooth_delta", "smooth_delta", "smooth_delta_weight", "action"),
    ("decoded_action", "decoded_action", "decoded_action_loss_weight", "action"),
    (
        "physical_delta_consistency",
        "physical_delta_consistency",
        "physical_delta_consistency_weight",
        "action",
    ),
    (
        "transition_gripper_flow",
        "transition_gripper_flow",
        "transition_gripper_flow_weight",
        "action",
    ),
    (
        "event_delta_consistency",
        "event_delta_consistency",
        "event_delta_consistency_weight",
        "action",
    ),
    ("event_magnitude", "event_magnitude", "event_magnitude_weight", "action"),
    ("event_off_delta", "event_off_delta", "event_off_delta_weight", "action"),
    ("rollout_dynamics", "rollout_dynamics", "rollout_dynamics_loss_weight", "rollout"),
    ("rollout_delta", "rollout_delta", "rollout_delta_loss_weight", "rollout"),
    ("rollout_contrast", "rollout_contrast", "rollout_contrast_loss_weight", "rollout"),
    ("rollout_variance", "rollout_variance", "rollout_variance_loss_weight", "rollout"),
    ("rollout_norm", "rollout_norm", "rollout_norm_loss_weight", "rollout"),
    (
        "rollout_milestone",
        "rollout_milestone_delta_match",
        "rollout_milestone_delta_match_weight",
        "rollout",
    ),
    (
        "execution_value",
        "evidence_mmd_it_execution_value_loss",
        "latent_cvae_mmdit_execution_value_loss_weight",
        "execution",
    ),
    ("latent_kl", "latent_cvae_kl", "latent_cvae_kl_weight", "latent"),
    (
        "latent_posterior_recon",
        "latent_cvae_posterior_recon",
        "latent_cvae_posterior_recon_weight",
        "latent",
    ),
    (
        "latent_adaptive_regularizer",
        "latent_cvae_adaptive_regularizer",
        "latent_cvae_adaptive_regularizer_weight",
        "latent",
    ),
    (
        "latent_route_entropy",
        "latent_cvae_adaptive_route_entropy_regularizer",
        "latent_cvae_adaptive_route_entropy_weight",
        "latent",
    ),
)


def _loss_budget(run: ParsedRun) -> dict[str, Any]:
    latest_train = _train_section(run.epoch_records[-1]) if run.epoch_records else {}
    latest_batch = run.batch_points[-1].metrics if run.batch_points else {}
    source = (
        latest_batch
        if any(key.startswith("loss_group_") for key in latest_batch)
        else latest_train or latest_batch
    )
    exact_groups = {
        key.removeprefix("loss_group_"): value
        for key, value in source.items()
        if key.startswith("loss_group_") and math.isfinite(value)
    }
    if exact_groups:
        exact_components = {
            key.removeprefix("loss_contrib_"): value
            for key, value in source.items()
            if key.startswith("loss_contrib_") and math.isfinite(value)
        }
        total = sum(exact_groups.values())
        residual = source.get("loss_ledger_residual")
        return {
            "mode": "exact-ledger",
            "components": exact_components,
            "groups": exact_groups,
            "total": total,
            "reference_loss": source.get("loss"),
            "residual": residual,
            "shares": {
                key: value / total for key, value in exact_groups.items() if abs(total) > 1e-12
            },
        }

    contributions: dict[str, float] = {}
    groups: dict[str, float] = {}
    for name, metric, weight_key, group in LOSS_COMPONENTS:
        raw = source.get(metric)
        if raw is None or not math.isfinite(raw):
            continue
        if weight_key is None:
            weight = 1.0
        else:
            configured = _config_value(run, weight_key)
            weight = float(configured) if isinstance(configured, (int, float)) else 0.0
        if weight <= 0.0:
            continue
        contribution = raw * weight
        contributions[name] = contribution
        groups[group] = groups.get(group, 0.0) + contribution
    layer_contribution = source.get("loss_contrib_layer_contract")
    if layer_contribution is None:
        raw_layer = source.get("aux_layer_contract_loss", source.get("layer_contract"))
        layer_scale = source.get("layer_contract_aux_scale", source.get("midcut_aux_scale"))
        if raw_layer is not None and layer_scale is not None:
            layer_contribution = raw_layer * layer_scale
    if layer_contribution is not None and math.isfinite(layer_contribution):
        contributions["layer_contract"] = layer_contribution
        groups["layer"] = groups.get("layer", 0.0) + layer_contribution
    total = sum(contributions.values())
    reference = source.get("loss")
    return {
        "mode": "estimated-known-terms",
        "components": contributions,
        "groups": groups,
        "total": total,
        "reference_loss": reference,
        "residual": None if reference is None else reference - total,
        "shares": {key: value / total for key, value in groups.items() if abs(total) > 1e-12},
    }


def _observability(run: ParsedRun) -> dict[str, Any]:
    by_key: dict[str, list[float]] = {}
    for point in run.batch_points:
        for key, value in point.metrics.items():
            if math.isfinite(value):
                by_key.setdefault(key, []).append(value)
    eligible = {key: values for key, values in by_key.items() if len(values) >= 5}
    all_zero = sorted(
        key for key, values in eligible.items() if max(abs(value) for value in values) <= 1e-12
    )
    constant = sorted(
        key
        for key, values in eligible.items()
        if key not in all_zero
        and max(values) - min(values) <= max(1e-10, 1e-8 * max(abs(value) for value in values))
    )
    representation_stage = any(point.source == "v95-stage1" for point in run.batch_points)
    required_core = (
        ("loss", "flow_jepa_future_prediction", "flow_jepa_stage_prediction")
        if representation_stage
        else ("loss", "physical_flow")
    )
    missing_core = [key for key in required_core if key not in by_key]
    return {
        "batch_metric_count": len(by_key),
        "always_zero_count": len(all_zero),
        "always_zero_examples": all_zero[:20],
        "constant_count": len(constant),
        "constant_examples": constant[:20],
        "placeholder_fraction": len(all_zero) / max(len(eligible), 1),
        "missing_core": missing_core,
        "malformed_json": run.malformed_json,
        "unclosed_json": run.unclosed_json,
        "traceback_count": run.traceback_count,
        "fatal_errors": run.fatal_errors[:8],
    }


def _add_finding(
    findings: list[Finding],
    severity: str,
    category: str,
    code: str,
    message: str,
    **evidence: Any,
) -> None:
    findings.append(Finding(severity, category, code, message, evidence))


def _find_duplicate_series(run: ParsedRun, findings: list[Finding]) -> None:
    pairs = (
        ("rollout_dynamics", "rollout_delta"),
        ("rollout_delta", "rollout_milestone_delta_match"),
    )
    for left, right in pairs:
        aligned = [
            (point.metrics[left], point.metrics[right])
            for point in run.batch_points
            if left in point.metrics and right in point.metrics
        ]
        for record in run.epoch_records:
            train = _train_section(record)
            if left in train and right in train:
                aligned.append((train[left], train[right]))
        if len(aligned) < 5:
            continue
        scale = max(max(abs(a), abs(b)) for a, b in aligned)
        max_gap = max(abs(a - b) for a, b in aligned)
        if max_gap > max(1e-8, scale * 1e-6):
            continue
        left_weight = _config_value(run, f"{left}_loss_weight")
        right_weight = _config_value(run, f"{right}_loss_weight")
        both_active = all(
            isinstance(value, (int, float)) and float(value) > 0.0
            for value in (left_weight, right_weight)
        )
        severity = "warning" if both_active else "info"
        suffix = (
            "且两项权重均启用，构成重复监督。"
            if both_active
            else "当前至少一项未启用，主要是日志别名。"
        )
        _add_finding(
            findings,
            severity,
            "loss",
            "duplicate-series",
            f"{left} 与 {right} 在 {len(aligned)} 个 batch 上逐项相同；{suffix}",
            left=left,
            right=right,
            count=len(aligned),
            max_gap=max_gap,
            left_weight=left_weight,
            right_weight=right_weight,
        )


def _health_findings(run: ParsedRun, observability: Mapping[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    if run.traceback_count:
        nonfinite = [
            error
            for error in run.fatal_errors
            if any(token in error.lower() for token in ("non-finite", "nan", "inf"))
        ]
        out_of_memory = [
            error for error in run.fatal_errors if "outofmemoryerror" in error.lower()
        ]
        code = (
            "non-finite-backward"
            if nonfinite
            else "out-of-memory"
            if out_of_memory
            else "unhandled-exception"
        )
        category = "numerics" if nonfinite else "memory" if out_of_memory else "runtime"
        _add_finding(
            findings,
            "critical",
            category,
            code,
            "The run terminated with an unhandled exception; trailing finite rows do not prove health.",
            traceback_count=run.traceback_count,
            errors=run.fatal_errors[:4],
            exception_truncated=run.traceback_count > len(run.fatal_errors),
        )
    if observability["malformed_json"] or observability["unclosed_json"]:
        _add_finding(
            findings,
            "warning",
            "logging",
            "json-integrity",
            "日志包含无法解析或未闭合的 JSON，epoch/config 结论可能不完整。",
            malformed=observability["malformed_json"],
            unclosed=observability["unclosed_json"],
        )
    if observability["placeholder_fraction"] > 0.50:
        _add_finding(
            findings,
            "warning",
            "logging",
            "placeholder-heavy",
            "超过一半的重复 batch 指标恒为零，活动分支信息被大量占位项稀释。",
            placeholder_fraction=observability["placeholder_fraction"],
            always_zero_count=observability["always_zero_count"],
        )
    if run.init_counts:
        missing = sum(value for key, value in run.init_counts.items() if "missing" in key)
        skipped = sum(value for key, value in run.init_counts.items() if "skipped" in key)
        if missing or skipped:
            _add_finding(
                findings,
                "info",
                "initialization",
                "partial-initialization",
                "Stage1 不是严格完整恢复；迁移结论必须结合缺失/跳过参数规模解释。",
                missing=missing,
                skipped=skipped,
                details=run.init_counts,
            )

    finite_failures: list[tuple[int, int, str, float]] = []
    for point in run.batch_points:
        for key, value in point.metrics.items():
            if not math.isfinite(value):
                finite_failures.append((point.epoch, point.batch, key, value))
    if finite_failures:
        _add_finding(
            findings,
            "critical",
            "numerics",
            "non-finite",
            "训练日志出现 NaN/Inf。",
            examples=finite_failures[:8],
            count=len(finite_failures),
        )

    ledger = _series(run, "loss_ledger_residual")
    if ledger:
        max_abs = max(abs(value) for value in ledger)
        if max_abs > 1e-5:
            _add_finding(
                findings,
                "critical",
                "loss",
                "ledger-open",
                "loss ledger 无法重建实际 backward scalar，存在未登记 objective 或错误权重。",
                max_abs_residual=max_abs,
            )

    if len(run.epoch_records) >= 2:
        first_train = _train_section(run.epoch_records[0])
        last_train = _train_section(run.epoch_records[-1])
        first_val = _val_section(run.epoch_records[0])
        last_val = _val_section(run.epoch_records[-1])
        train_change = _relative_change(
            first_train.get("physical_flow"), last_train.get("physical_flow")
        )
        val_change = _relative_change(first_val.get("full_rmse"), last_val.get("full_rmse"))
        if train_change is not None and val_change is not None and train_change < -0.30:
            if val_change > -0.05:
                _add_finding(
                    findings,
                    "warning",
                    "generalization",
                    "train-val-decoupling",
                    "训练 pflow 大幅下降，但验证 full RMSE 几乎没有同步改善。",
                    train_pflow_change=train_change,
                    val_rmse_change=val_change,
                )
        arm_change = _relative_change(first_val.get("arm_full_rmse"), last_val.get("arm_full_rmse"))
        if arm_change is not None and arm_change > 0.0:
            _add_finding(
                findings,
                "warning",
                "generalization",
                "arm-regression",
                "验证 arm RMSE 随 epoch 变差。",
                change=arm_change,
            )

    latest_val = _val_section(run.epoch_records[-1]) if run.epoch_records else {}
    tail_ratio = latest_val.get("tail_first_ratio")
    if tail_ratio is None and latest_val.get("tail_rmse") is not None:
        tail_ratio = latest_val["tail_rmse"] / max(latest_val.get("first_rmse", 0.0), 1e-8)
    if tail_ratio is not None and tail_ratio > 2.0:
        _add_finding(
            findings,
            "warning",
            "horizon",
            "tail-gap",
            "长时域 tail RMSE 明显高于 first RMSE。",
            tail_first_ratio=tail_ratio,
        )
    event_ratio = latest_val.get("gripper_event_ratio")
    if event_ratio is not None and (event_ratio < 0.7 or event_ratio > 1.8):
        severity = "critical" if event_ratio > 3.0 else "warning"
        _add_finding(
            findings,
            severity,
            "gripper",
            "event-rate-mismatch",
            "最终 decoded gripper 的事件率与目标严重不匹配。",
            gripper_event_ratio=event_ratio,
        )
    event_accuracy = latest_val.get("event_head_accuracy")
    event_f1 = latest_val.get("event_head_f1")
    event_decoded_gap = latest_val.get("event_head_minus_decoded_gripper_f1")
    if event_decoded_gap is not None and abs(event_decoded_gap) >= 0.20:
        _add_finding(
            findings,
            "warning",
            "gripper",
            "event-head-decoded-gap",
            "The auxiliary event head and decoded gripper behavior have materially different F1.",
            f1_gap=event_decoded_gap,
        )
    if (
        event_accuracy is not None
        and event_f1 is not None
        and event_accuracy > 0.9
        and event_f1 < 0.4
    ):
        _add_finding(
            findings,
            "warning",
            "gripper",
            "accuracy-imbalance",
            "event-head accuracy 很高但 F1 很低，accuracy 主要反映类别不平衡。",
            accuracy=event_accuracy,
            f1=event_f1,
        )
    proposal_gain = latest_val.get("proposal_utility_mse_gain")
    proposal_coverage = latest_val.get("eval_proposal_ablation_coverage")
    if proposal_gain is not None and proposal_gain <= 0.0:
        _add_finding(
            findings,
            "warning",
            "proposal",
            "proposal-no-gain",
            "proposal ablation 没有显示正向 MSE 收益。",
            gain=proposal_gain,
            coverage=proposal_coverage,
        )
    if proposal_coverage is not None and proposal_coverage < 0.2:
        _add_finding(
            findings,
            "info",
            "coverage",
            "proposal-low-coverage",
            "proposal ablation 覆盖率较低，收益结论置信度有限。",
            coverage=proposal_coverage,
        )

    execution_coverage = latest_val.get("eval_execution_ablation_coverage")
    primary_execution_rmse = latest_val.get("execution_ablation_primary_full_rmse")
    neutral_execution_rmse = latest_val.get("execution_ablation_neutral_full_rmse")
    hard_execution_rmse = latest_val.get("execution_ablation_hard_full_rmse")
    full_capacity_rmse = latest_val.get("execution_ablation_full_capacity_full_rmse")
    if (
        primary_execution_rmse is not None
        and neutral_execution_rmse is not None
        and neutral_execution_rmse < 0.99 * primary_execution_rmse
    ):
        _add_finding(
            findings,
            "warning",
            "execution",
            "neutral-execution-better",
            "Matched-noise neutral host execution outperforms the learned soft controller.",
            primary_rmse=primary_execution_rmse,
            neutral_rmse=neutral_execution_rmse,
            coverage=execution_coverage,
        )
    if (
        primary_execution_rmse is not None
        and hard_execution_rmse is not None
        and hard_execution_rmse > 1.05 * primary_execution_rmse
    ):
        _add_finding(
            findings,
            "warning",
            "execution",
            "hard-soft-validation-gap",
            "Matched-noise hard execution is materially worse than the trained soft contract.",
            primary_rmse=primary_execution_rmse,
            hard_rmse=hard_execution_rmse,
            coverage=execution_coverage,
        )
    if (
        primary_execution_rmse is not None
        and full_capacity_rmse is not None
        and full_capacity_rmse < 0.99 * primary_execution_rmse
    ):
        _add_finding(
            findings,
            "warning",
            "execution",
            "capacity-gate-hurts",
            "The full-capacity matched-noise ablation outperforms learned capacity gating.",
            primary_rmse=primary_execution_rmse,
            full_capacity_rmse=full_capacity_rmse,
            coverage=execution_coverage,
        )

    open_points = [
        point
        for point in run.batch_points
        if point.metrics.get("evidence_mmd_it_execution_progress", -1.0) >= 0.9
    ]
    if open_points:
        capacities = [
            point.metrics.get(
                "evidence_mmd_it_capacity_gate_mass",
                point.metrics.get("evidence_mmd_it_capacity_ratio"),
            )
            for point in open_points
            if (
                "evidence_mmd_it_capacity_gate_mass" in point.metrics
                or "evidence_mmd_it_capacity_ratio" in point.metrics
            )
        ]
        depths = [
            point.metrics.get(
                "evidence_mmd_it_effective_basis_mass",
                point.metrics.get("evidence_mmd_it_effective_depth"),
            )
            for point in open_points
            if (
                "evidence_mmd_it_effective_basis_mass" in point.metrics
                or "evidence_mmd_it_effective_depth" in point.metrics
            )
        ]
        capacity = _median(capacities[-20:])
        depth = _median(depths[-20:])
        rank = _config_value(run, "rank", "latent_cvae_mmdit_operator_rank")
        if (
            capacity is not None
            and capacity >= 0.995
            and depth is not None
            and isinstance(rank, (int, float))
            and depth >= float(rank) - 0.25
        ):
            _add_finding(
                findings,
                "warning",
                "execution",
                "capacity-saturated",
                "execution 完全打开后仍保持满容量/满 depth。",
                capacity=capacity,
                depth=depth,
                rank=rank,
            )
        cap_grad = _median(
            point.metrics["grad_evidence_mmdit_capacity_control"]
            for point in open_points[-20:]
            if "grad_evidence_mmdit_capacity_control" in point.metrics
        )
        if cap_grad is None and run.epoch_records:
            cap_grad = _train_section(run.epoch_records[-1]).get(
                "grad_evidence_mmdit_capacity_control"
            )
        if cap_grad is not None and cap_grad <= 1e-6:
            _add_finding(
                findings,
                "warning",
                "gradient",
                "capacity-gradient-dead",
                "execution 打开后的 capacity-controller 梯度接近零。",
                median_gradient=cap_grad,
            )
        common = _median(
            point.metrics["evidence_mmd_it_execution_value_common_mode_ratio"]
            for point in open_points[-20:]
            if "evidence_mmd_it_execution_value_common_mode_ratio" in point.metrics
        )
        decision = _median(
            point.metrics["evidence_mmd_it_execution_value_decision_accuracy"]
            for point in open_points[-20:]
            if "evidence_mmd_it_execution_value_decision_accuracy" in point.metrics
        )
        if common is not None and decision is not None and common >= 0.9 and decision <= 0.6:
            _add_finding(
                findings,
                "warning",
                "execution",
                "value-common-mode",
                "value reader 主要输出候选共有分量，最终候选选择接近随机。",
                common_mode_ratio=common,
                decision_accuracy=decision,
            )
        selection_max = _median(
            point.metrics["evidence_mmd_it_execution_selection_max_probability"]
            for point in open_points[-20:]
            if "evidence_mmd_it_execution_selection_max_probability" in point.metrics
        )
        if selection_max is not None and selection_max <= 0.35:
            _add_finding(
                findings,
                "info",
                "execution",
                "high-selection-entropy",
                "execution 完全打开后候选分布仍然接近高熵。",
                median_max_probability=selection_max,
            )
        terminal_identity_error = _median(
            point.metrics["evidence_mmd_it_terminal_identity_velocity_error"]
            for point in open_points[-20:]
            if "evidence_mmd_it_terminal_identity_velocity_error" in point.metrics
        )
        if terminal_identity_error is not None and terminal_identity_error > 1e-5:
            _add_finding(
                findings,
                "critical",
                "execution",
                "terminal-not-identity",
                "The terminal candidate no longer matches the committed prefix velocity.",
                identity_velocity_error=terminal_identity_error,
            )

    depth_logit = _config_value(
        run, "depth_logit_init", "latent_cvae_mmdit_operator_depth_logit_init"
    )
    groups = _config_value(run, "groups", "latent_cvae_mmdit_operator_groups")
    if isinstance(depth_logit, (int, float)) and isinstance(groups, (int, float)):
        raw_ratio = 1.0 / (1.0 + math.exp(-float(depth_logit)))
        initial_hard_groups = math.ceil(raw_ratio * int(groups) - 1e-12)
        if initial_hard_groups >= int(groups):
            _add_finding(
                findings,
                "warning",
                "configuration",
                "full-depth-init",
                "depth logit 初始化经 hard group 取整后直接选择全部 group，容易形成满容量饱和。",
                depth_logit=depth_logit,
                ratio=raw_ratio,
                groups=groups,
                hard_groups=initial_hard_groups,
            )

    z_probe = _config_value(run, "z_probe", "latent_cvae_z_probe")
    if run.epoch_records and isinstance(z_probe, (int, float)) and int(z_probe) == 1:
        z_zero = latest_val.get("sample_evidence_z_zero_condition_delta")
        z_shuffle = latest_val.get("sample_evidence_z_shuffle_condition_delta")
        if z_zero is None or z_shuffle is None:
            _add_finding(
                findings,
                "warning",
                "logging",
                "z-probe-missing",
                "配置启用了 active-path z probe，但 epoch validation 没有给出完整 z-zero/z-shuffle 指标。",
                z_zero=z_zero,
                z_shuffle=z_shuffle,
            )
        elif abs(z_zero) <= 1e-12 and abs(z_shuffle) <= 1e-12:
            _add_finding(
                findings,
                "warning",
                "conditioning",
                "z-probe-zero",
                "z-zero 与 z-shuffle 干预均无可测影响，需要区分路径未使用与 probe 实现失效。",
                z_zero=z_zero,
                z_shuffle=z_shuffle,
            )

    _find_duplicate_series(run, findings)
    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda item: (order.get(item.severity, 9), item.category, item.code))
    return findings


def build_summary(run: ParsedRun, *, tail: int = 20) -> dict[str, Any]:
    observability = _observability(run)
    findings = _health_findings(run, observability)
    latest_train = _train_section(run.epoch_records[-1]) if run.epoch_records else {}

    def stats_with_epoch_fallback(key: str) -> dict[str, Any] | None:
        values = _series(run, key)
        if values:
            return _window_stats(run, key, tail)
        value = latest_train.get(key)
        if value is None or not math.isfinite(value):
            return None
        return {
            "count": 1,
            "first": value,
            "last": value,
            "tail_median": value,
            "change": None,
            "source": "latest-epoch",
        }

    trajectories = {
        key: _window_stats(run, key, tail) for key in CORE_BATCH_KEYS if _series(run, key)
    }
    structure = {
        key: stats
        for key in STRUCTURE_KEYS
        if (stats := stats_with_epoch_fallback(key)) is not None
    }
    gradients = {
        key: stats for key in GRADIENT_KEYS if (stats := stats_with_epoch_fallback(key)) is not None
    }
    all_batch_keys = sorted({key for point in run.batch_points for key in point.metrics})
    metric_index = {key: _window_stats(run, key, tail) for key in all_batch_keys}
    epochs: list[dict[str, Any]] = []
    for record in run.epoch_records:
        train = _train_section(record)
        val = _val_section(record)
        epochs.append(
            {
                "epoch": record.get("epoch"),
                "global_step": record.get("global_step"),
                "train": {
                    key: train[key]
                    for key in (
                        "loss",
                        "flow_jepa_future_prediction",
                        "flow_jepa_stage_prediction",
                        "flow_jepa_warp_loss",
                        "flow_jepa_cycle_loss",
                        "physical_flow",
                        "physical_flow_native",
                        "rollout_dynamics",
                        "decoded_action",
                        "event",
                        "loss_ledger_residual",
                    )
                    if key in train
                },
                "val": {key: val[key] for key in VALIDATION_KEYS if key in val},
            }
        )
    return {
        "label": run.label,
        "path": str(run.path),
        "coverage": {
            "batch_rows": len(run.batch_points),
            "epoch_records": len(run.epoch_records),
            "headers": run.headers,
            "init_counts": run.init_counts,
            "context_schema": run.context.get("schema"),
            "traceback_count": run.traceback_count,
            "fatal_errors": run.fatal_errors[:8],
            "batch_range": (
                {
                    "first": [run.batch_points[0].epoch, run.batch_points[0].batch],
                    "last": [run.batch_points[-1].epoch, run.batch_points[-1].batch],
                }
                if run.batch_points
                else None
            ),
        },
        "manifest": {
            "decoder": _config_value(run, "decoder", "final_action_decoder"),
            "training_stage": _config_value(run, "training_stage", "experiment_stage"),
            "seed": _config_value(run, "seed"),
            "batch_size": _config_value(run, "batch_size"),
            "data_root": _config_value(run, "data_root"),
            "train_episode_count": _config_value(run, "train_episode_count"),
            "val_episode_count": _config_value(run, "val_episode_count"),
            "condition_mode": _config_value(run, "condition_mode"),
            "stage1_checkpoint": _config_value(run, "stage1_checkpoint"),
            "action_normalizer_fingerprint": _config_value(run, "action_normalizer_fingerprint"),
            "rank": _config_value(run, "rank", "latent_cvae_mmdit_operator_rank"),
            "groups": _config_value(run, "groups", "latent_cvae_mmdit_operator_groups"),
            "depth_logit_init": _config_value(
                run, "depth_logit_init", "latent_cvae_mmdit_operator_depth_logit_init"
            ),
            "warmup": _config_value(run, "warmup", "latent_cvae_mmdit_execution_warmup_steps"),
            "transition": _config_value(
                run, "transition", "latent_cvae_mmdit_execution_transition_steps"
            ),
            "contract_grad_scale": _config_value(
                run, "contract_grad_scale", "layer_contract_grad_scale"
            ),
            "layer_grad_scale": _config_value(
                run, "layer_grad_scale", "latent_cvae_layer_grad_scale"
            ),
            "layer_detach": _config_value(run, "layer_detach", "latent_cvae_layer_detach"),
            "transition_detach": _config_value(
                run, "transition_detach", "latent_cvae_transition_detach"
            ),
            "z_probe": _config_value(run, "z_probe", "latent_cvae_z_probe"),
        },
        "loss_budget": _loss_budget(run),
        "trajectories": trajectories,
        "structure": structure,
        "gradients": gradients,
        "metric_index": metric_index,
        "epochs": epochs,
        "latest_epoch_metrics": (
            {
                "train": _train_section(run.epoch_records[-1]),
                "val": _val_section(run.epoch_records[-1]),
            }
            if run.epoch_records
            else None
        ),
        "observability": observability,
        "findings": [asdict(finding) for finding in findings],
    }


def _render_run_text(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    lines = [
        f"=== {summary['label']} ===",
        f"path: {summary['path']}",
        "coverage: "
        f"batch_rows={coverage['batch_rows']} epoch_records={coverage['epoch_records']} "
        f"range={coverage['batch_range']}",
    ]
    manifest = {key: value for key, value in summary["manifest"].items() if value is not None}
    if manifest:
        lines.append("manifest: " + " ".join(f"{key}={value}" for key, value in manifest.items()))
    budget = summary["loss_budget"]
    if budget.get("groups"):
        groups = " ".join(
            f"{key}={_format_number(value, 5)}" for key, value in budget["groups"].items()
        )
        lines.append(
            f"loss budget ({budget['mode']}): {groups} "
            f"residual={_format_number(budget.get('residual'), 3)}"
        )
        components = budget.get("components", {})
        if components:
            ranked = sorted(components.items(), key=lambda item: abs(float(item[1])), reverse=True)[
                :8
            ]
            lines.append(
                "  top weighted contributions: "
                + " ".join(f"{key}={_format_number(value, 5)}" for key, value in ranked)
            )
    if summary["trajectories"]:
        lines.append("training trajectory (window median first -> last, relative):")
        for key, stats in summary["trajectories"].items():
            change = stats.get("change")
            change_text = "-" if change is None else f"{change:+.1%}"
            lines.append(
                f"  {key}: {_format_number(stats['first'])} -> "
                f"{_format_number(stats['last'])} ({change_text}, n={stats['count']})"
            )
    if summary["epochs"]:
        lines.append("epoch validation:")
        for record in summary["epochs"]:
            val = record["val"]
            selected = (
                "loss",
                "flow_jepa_future_prediction",
                "flow_jepa_stage_prediction",
                "flow_jepa_stage_window_cosine",
                "flow_jepa_goal_pair_cosine",
                "eval_representation_coverage",
                "full_rmse",
                "arm_full_rmse",
                "gripper_full_rmse",
                "tail_first_ratio",
                "gripper_event_ratio",
                "event_head_f1",
                "proposal_utility_mse_gain",
            )
            metrics = " ".join(
                f"{key}={_format_number(val[key])}" for key in selected if key in val
            )
            lines.append(
                f"  epoch={record['epoch']} step={record['global_step']} {metrics}".rstrip()
            )
    if summary["structure"]:
        lines.append("structure/controller tail medians:")
        for key, stats in summary["structure"].items():
            lines.append(f"  {key}={_format_number(stats['tail_median'])}")
    if summary["gradients"]:
        lines.append("gradient tail medians:")
        for key, stats in summary["gradients"].items():
            lines.append(f"  {key}={_format_number(stats['tail_median'], 3)}")
    obs = summary["observability"]
    lines.append(
        "observability: "
        f"metrics={obs['batch_metric_count']} always_zero={obs['always_zero_count']} "
        f"constant={obs['constant_count']} placeholder_fraction={obs['placeholder_fraction']:.1%}"
    )
    if obs["always_zero_examples"]:
        lines.append("  zero examples: " + ", ".join(obs["always_zero_examples"][:10]))
    lines.append("findings:")
    if not summary["findings"]:
        lines.append(
            "  [info] no rule-based anomaly detected; absence of findings is not proof of health"
        )
    else:
        for finding in summary["findings"]:
            lines.append(
                f"  [{finding['severity']}] {finding['category']}/{finding['code']}: "
                f"{finding['message']}"
            )
    return "\n".join(lines)


def _comparison(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        latest_epoch = summary["epochs"][-1] if summary["epochs"] else {}
        latest_val = latest_epoch.get("val", {})
        pflow = summary["trajectories"].get("physical_flow", {})
        rows.append(
            {
                "label": summary["label"],
                "batch_rows": summary["coverage"]["batch_rows"],
                "epoch": latest_epoch.get("epoch"),
                "train_pflow_tail": pflow.get("tail_median"),
                "val_full_rmse": latest_val.get("full_rmse"),
                "val_arm_rmse": latest_val.get("arm_full_rmse"),
                "val_gripper_rmse": latest_val.get("gripper_full_rmse"),
                "tail_first_ratio": latest_val.get("tail_first_ratio"),
                "gripper_event_ratio": latest_val.get("gripper_event_ratio"),
                "decoder": summary["manifest"].get("decoder"),
                "action_normalizer_fingerprint": summary["manifest"].get(
                    "action_normalizer_fingerprint"
                ),
                "critical": sum(
                    finding["severity"] == "critical" for finding in summary["findings"]
                ),
                "warnings": sum(
                    finding["severity"] == "warning" for finding in summary["findings"]
                ),
            }
        )
    return rows


def _merge_runs(runs: Sequence[ParsedRun], *, path: Path, label: str) -> ParsedRun:
    merged = ParsedRun(path=path, label=label)
    for run in runs:
        merged.headers.extend(header for header in run.headers if header not in merged.headers)
        merged.header_config.update(
            {
                key: value
                for key, value in run.header_config.items()
                if key not in merged.header_config
            }
        )
        if len(run.context) > len(merged.context):
            merged.context = run.context
        merged.init_counts.update(run.init_counts)
        merged.batch_points.extend(run.batch_points)
        merged.epoch_records.extend(run.epoch_records)
        merged.malformed_json += run.malformed_json
        merged.unclosed_json += run.unclosed_json
        merged.traceback_count += run.traceback_count
        merged.fatal_errors.extend(run.fatal_errors)
    unique_points: dict[tuple[int, int, str], BatchPoint] = {}
    for point in merged.batch_points:
        unique_points[(point.epoch, point.batch, point.source)] = point
    merged.batch_points = sorted(
        unique_points.values(), key=lambda point: (point.epoch, point.batch)
    )
    merged.epoch_records = _dedupe_epoch_records(merged.epoch_records)
    merged.epoch_records.sort(
        key=lambda record: (int(record.get("epoch", 0)), int(record.get("global_step", 0)))
    )
    return merged


def parse_run_input(path: Path) -> ParsedRun:
    if path.is_file():
        return parse_log(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = ("nohup.log", "v39_policy_epochs.jsonl", "train.log")
    matched = [path / name for name in candidates if (path / name).is_file()]
    if not matched:
        matched = sorted(path.glob("*.log")) + sorted(path.glob("*.txt"))
    if not matched:
        raise FileNotFoundError(f"no supported logs under {path}")
    return _merge_runs(
        [parse_log(item, label=path.name) for item in matched],
        path=path,
        label=path.name,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ClearVLA policy logs across legacy and compact formats."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="log files or run directories")
    parser.add_argument("--tail", type=int, default=20, help="tail window in logged batches")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write report instead of stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tail < 1:
        raise SystemExit("--tail must be positive")
    try:
        runs = [parse_run_input(path) for path in args.logs]
    except FileNotFoundError as exc:
        print(f"log path does not exist: {exc}", file=sys.stderr)
        return 2
    if not runs:
        print("no supported log files found", file=sys.stderr)
        return 2
    summaries = [build_summary(run, tail=args.tail) for run in runs]
    payload = {"runs": summaries, "comparison": _comparison(summaries)}
    if args.format == "json":
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True)
    else:
        rendered = "\n\n".join(_render_run_text(summary) for summary in summaries)
        if len(summaries) > 1:
            rendered += "\n\n=== comparison ===\n"
            for row in payload["comparison"]:
                rendered += (
                    f"{row['label']}: pflow={_format_number(row['train_pflow_tail'])} "
                    f"val_rmse={_format_number(row['val_full_rmse'])} "
                    f"tail/first={_format_number(row['tail_first_ratio'])} "
                    f"event_ratio={_format_number(row['gripper_event_ratio'])} "
                    f"critical={row['critical']} warnings={row['warnings']}\n"
                )
            rendered = rendered.rstrip()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
