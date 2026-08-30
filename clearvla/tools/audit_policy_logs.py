"""Audit ClearVLA training logs without importing the training stack.

The parser accepts the historical ``[v39-layer]`` format, the compact V94-V121
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
MAINLINE_HEADER_RE = re.compile(r"^\[mainline\]\s+(?P<body>.*)$")
INIT_COUNT_RE = re.compile(r"^\[v39-init\]\s+(?P<label>.*?)(?:\s+count=(?P<count>\d+))$")
UNHANDLED_EXCEPTION_RE = re.compile(
    r"^(?P<type>(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception)):\s*(?P<message>.*)$"
)
COMPACT_POLICY_VERSIONS = tuple(range(94, 123))

# A recovery run is allowed moderate implementation overhead, but cannot hide
# the four-times-slower regressions that motivated the independent mainline.
# These are release gates, not optimization losses or model-side constraints.
RECOVERY_MEDIAN_SECONDS_RATIO_MAX = 1.5
RECOVERY_P90_SECONDS_RATIO_MAX = 2.0
RECOVERY_CUDA_PROCESS_GIB_MAX = 22.0


def _compact_prefixes(family: str) -> tuple[str, ...]:
    return tuple(f"[v{version}-{family}]" for version in COMPACT_POLICY_VERSIONS)


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
        "intent_goal": "grad_intent_goal_program",
        "intent_history": "grad_intent_history_encoder",
        "intent_history_write": "grad_intent_history_write",
        "intent_grounding": "grad_intent_grounding_write",
        "intent_ordered": "grad_intent_ordered_refinement",
        "intent_window": "grad_intent_window_read",
        "intent_predictive": "grad_intent_predictive_effect",
        "intent_terminal": "grad_intent_terminal",
        "w_clean_proposal": (
            "grad_differential_clean_proposal_world_condition"
        ),
        "intent_g_to_p_query": "grad_intent_canonical_g_to_p_query",
        "intent_p1_query": "grad_intent_canonical_p1_query",
        "consequence_organizer": "grad_consequence_plan_organizer",
        "grounded_w_inputs": "grad_grounded_world_shared_inputs",
        "grounded_w1_blocks": "grad_grounded_world_w1_blocks",
        "grounded_w2_blocks": "grad_grounded_world_w2_blocks",
        "grounded_w_shared_heads": "grad_grounded_world_shared_heads",
        # Preserve the actual names from already-completed V119 logs.  Their
        # W1 group is known to include shared inputs, so it remains a legacy
        # key rather than being relabeled as the corrected blocks-only group.
        "grounded_w1": "grad_grounded_world_w1",
        "grounded_w2": "grad_grounded_world_w2",
        "grounded_w_heads": "grad_grounded_world_effect_heads",
        "differential_w1": "grad_differential_w1_near_mid_transition",
        "differential_w2": "grad_differential_w2_late_transition",
        "effect_decoder": "grad_differential_effect_decoder",
        "current_reference": "grad_differential_current_reference_bridge",
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
        "effect_w1_current_loss": "flow_jepa_future_effect_w1_current_loss",
        "effect_w1_successor_loss": "flow_jepa_future_effect_w1_successor_loss",
        "effect_w1_semantic_loss": "flow_jepa_future_effect_w1_semantic_loss",
        "effect_w2_current_loss": "flow_jepa_future_effect_w2_current_loss",
        "effect_w2_successor_loss": "flow_jepa_future_effect_w2_successor_loss",
        "effect_semantic_loss": "flow_jepa_future_effect_semantic_loss",
        "effect_successor_loss": "flow_jepa_future_effect_successor_loss",
        "effect_semantic_near_loss": "flow_jepa_future_effect_semantic_near_loss",
        "effect_semantic_mid_loss": "flow_jepa_future_effect_semantic_mid_loss",
        "effect_semantic_late_loss": "flow_jepa_future_effect_semantic_late_loss",
        "effect_intent_summary_loss": "flow_jepa_future_effect_intent_summary_loss",
        "effect_near_contrib": "flow_jepa_future_effect_effective_near_loss",
        "effect_mid_contrib": "flow_jepa_future_effect_effective_mid_loss",
        "effect_late_contrib": "flow_jepa_future_effect_effective_late_loss",
        "effect_transport_loss": "flow_jepa_future_effect_transport_loss",
        "effect_cov_loss": "flow_jepa_future_effect_transport_covariance_loss",
        "effect_persist_loss": "flow_jepa_future_effect_persistence_loss",
        "effect_visible_loss": "flow_jepa_future_effect_visibility_loss",
        "effect_uncert_loss": "flow_jepa_future_effect_uncertainty_loss",
        "w1_effect_cos": "flow_jepa_differential_w1_adjacent_cosine",
        "w1_effect_var": "flow_jepa_differential_w1_slot_variation",
        "w2_effect_cos": "flow_jepa_differential_w2_adjacent_cosine",
        "w2_effect_var": "flow_jepa_differential_w2_slot_variation",
        "effect_pred_near": "flow_jepa_differential_w2_near_effect_rms",
        "effect_pred_mid": "flow_jepa_differential_w2_mid_effect_rms",
        "effect_pred_late": "flow_jepa_differential_w2_late_effect_rms",
        "effect_target_near": "flow_jepa_future_effect_target_near_rms",
        "effect_target_mid": "flow_jepa_future_effect_target_mid_rms",
        "effect_target_late": "flow_jepa_future_effect_target_late_rms",
        "effect_rel_near": "flow_jepa_future_effect_teacher_reliability_near",
        "effect_rel_mid": "flow_jepa_future_effect_teacher_reliability_mid",
        "effect_rel_late": "flow_jepa_future_effect_teacher_reliability_late",
        "p2_diff_read": "flow_jepa_p2_effect_read_rms",
        "p2_diff_content_score": "flow_jepa_p2_effect_content_score_rms",
        "p2_diff_intent_score": "flow_jepa_p2_effect_intent_score_rms",
        "p2_diff_coordinate_score": "flow_jepa_p2_effect_coordinate_score_rms",
        "p2_diff_entropy": "flow_jepa_p2_effect_entropy",
        "p2_effect_read": "flow_jepa_p2_structured_effect_read_rms",
        "p2_effect_entropy": "flow_jepa_p2_structured_effect_entropy",
        "p2_effect_interval_var": "flow_jepa_p2_structured_effect_interval_rms_variation",
        "consequence_effect": "flow_jepa_consequence_effect_base_rms",
        "consequence_organized": "flow_jepa_consequence_organized_delta_rms",
        "plan_protected_base": "flow_jepa_policy_plan_protected_base_rms",
        "plan_precision": "flow_jepa_policy_plan_precision_rms",
        "plan_temporal": "flow_jepa_policy_plan_temporal_rms",
        "intent_program_cos": "flow_jepa_intent_program_adjacent_cosine",
        "intent_attention_entropy": "flow_jepa_intent_program_attention_entropy",
        "intent_predictive_effect": "flow_jepa_intent_predictive_effect_rms",
        "intent_language_innovation": "flow_jepa_intent_language_innovation_rms",
        "intent_history_innovation": "flow_jepa_intent_history_innovation_rms",
        "intent_grounding_innovation": "flow_jepa_intent_grounding_innovation_rms",
        "intent_ordered_innovation": "flow_jepa_intent_ordered_innovation_rms",
        "intent_near_program": "flow_jepa_intent_near_program_argmax",
        "intent_mid_program": "flow_jepa_intent_mid_program_argmax",
        "intent_late_program": "flow_jepa_intent_late_program_argmax",
        "w0_proposal_mass": "flow_jepa_w0_typed_condition_proposal_mass",
        "w1_proposal_mass": "flow_jepa_w1_typed_condition_proposal_mass",
        "w2_proposal_mass": "flow_jepa_w2_typed_condition_proposal_mass",
        "w0_clean_proposal": "flow_jepa_w0_clean_proposal_context_rms",
        "w1_clean_proposal": "flow_jepa_w1_clean_proposal_context_rms",
        "w2_clean_proposal": "flow_jepa_w2_clean_proposal_context_rms",
        "w0_direct_intent_bypass": "flow_jepa_w0_direct_intent_bypass",
        "w1_direct_intent_bypass": "flow_jepa_w1_direct_intent_bypass",
        "w2_direct_intent_bypass": "flow_jepa_w2_direct_intent_bypass",
        "p1_intent_query": "flow_jepa_phase_detail_query_norm",
        "p1_direct_condition_bypass": (
            "flow_jepa_differential_p1_direct_condition_bypass"
        ),
        "g_to_p_intent_query": "attnres_world_to_policy_phase_query_norm",
        "g_to_p_goal_bypass": (
            "attnres_world_to_policy_condition_query_norm"
        ),
        "g_to_p_history_bypass": (
            "attnres_world_to_policy_history_query_norm"
        ),
        "phase_terminal": "flow_jepa_phase_terminal_mass",
        "execution_terminal": "flow_jepa_execution_terminal_probability",
        "execution_terminal_uncert": "flow_jepa_execution_terminal_uncertainty",
        "execution_terminal_bias": "evidence_execution_terminal_external_bias",
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


# V119 has capability-specific compact rows in addition to the inherited
# train/repr/exec/grad lines.  Keep their display names separate instead of
# overloading the already-large ancestry ``repr`` alias table.
V94_ALIASES.update(
    {
        "ground": {
            "active": "grounded_intent_effect_active",
            "g2_fine_H": "flow_jepa_progressive_g2_fine_entropy",
            "g2_sem_app_l1": (
                "flow_jepa_progressive_g2_semantic_appearance_posterior_l1"
            ),
            "g2_app_geo_l1": (
                "flow_jepa_progressive_g2_appearance_geometry_posterior_l1"
            ),
            "g3_sem_H": "flow_jepa_progressive_g3_semantic_slot_entropy",
            "g3_app_H": "flow_jepa_progressive_g3_appearance_slot_entropy",
            "g3_geo_H": "flow_jepa_progressive_g3_geometry_slot_entropy",
            "g2g3_sem": "grounded_g2_g3_semantic_owner_l1",
            "g2g3_app": "grounded_g2_g3_appearance_owner_l1",
            "g2g3_geo": "grounded_g2_g3_geometry_owner_l1",
            "p1_app_geo_l1": (
                "flow_jepa_typed_p1_appearance_geometry_fine_l1"
            ),
            "p1_sem_app_l1": (
                "flow_jepa_typed_p1_semantic_appearance_route_l1"
            ),
            "current_ref_align": (
                "grounded_future_effect_current_reference_alignment_rms"
            ),
        },
        "intent": {
            "goal_attention_H": "grounded_s_goal_attention_entropy",
            "interval_goal_H": "grounded_s_interval_goal_attention_entropy",
            "source_attention_H": "grounded_s_interval_source_entropy",
            "interval_cos": "grounded_s_interval_adjacent_cosine",
            "interval_var": "grounded_s_interval_variation",
            "achieved": "grounded_s_achieved_rms",
            "remaining": "grounded_s_remaining_rms",
            "completion": "grounded_s_completion_probability",
            "source_null": "grounded_s_interval_null_mass",
            "source_observable": "grounded_s_interval_observable_mass",
            "source_history": "grounded_s_interval_history_mass",
            "source_semantic": "grounded_s_interval_semantic_mass",
            "source_appearance": "grounded_s_interval_appearance_mass",
            "source_geometry": "grounded_s_interval_geometry_mass",
            **{
                f"{interval}_goal_H": (
                    f"grounded_s_{interval}_goal_attention_entropy"
                )
                for interval in ("h4_8", "h8_16", "h16_32", "h32_48")
            },
            **{
                f"{interval}_source_H": (
                    f"grounded_s_{interval}_source_attention_entropy"
                )
                for interval in ("h4_8", "h8_16", "h16_32", "h32_48")
            },
        },
        "effect": {
            "w1_sem": "grounded_w1_semantic_rms",
            "w1_transport": "grounded_w1_transport_rms",
            "w1_interval_var": "grounded_w1_interval_variation",
            "w1_object_var": "grounded_w1_object_variation",
            "w1_cos": "grounded_w1_adjacent_cosine",
            "w2_sem": "grounded_w2_semantic_rms",
            "w2_transport": "grounded_w2_transport_rms",
            "w2_interval_var": "grounded_w2_interval_variation",
            "w2_object_var": "grounded_w2_object_variation",
            "w2_cos": "grounded_w2_adjacent_cosine",
            "pred_cos": "grounded_future_effect_prediction_adjacent_cosine",
            "target_cos": "grounded_future_effect_target_adjacent_cosine",
            "pred_var": "grounded_future_effect_prediction_interval_variation",
            "target_var": "grounded_future_effect_target_interval_variation",
            "transport_pred_var": (
                "grounded_future_effect_prediction_transport_variation"
            ),
            "transport_target_var": (
                "grounded_future_effect_target_transport_variation"
            ),
            "loss_successor": "flow_jepa_future_effect_successor_loss",
            "loss_semantic": "flow_jepa_future_effect_semantic_loss",
            "loss_transport": "flow_jepa_future_effect_transport_loss",
            "loss_covariance": (
                "flow_jepa_future_effect_transport_covariance_loss"
            ),
            "loss_persistence_change": (
                "flow_jepa_future_effect_persistence_change_loss"
            ),
            "loss_visibility_change": (
                "flow_jepa_future_effect_visibility_change_loss"
            ),
            "loss_uncertainty_calibration": (
                "flow_jepa_future_effect_uncertainty_calibration_loss"
            ),
            "loss_reliability_calibration": (
                "flow_jepa_future_effect_reliability_calibration_loss"
            ),
            "loss_interval_transition": (
                "flow_jepa_future_effect_relative_transition_loss"
            ),
        },
        "policy": {
            "effect_read": "grounded_p2_effect_read_rms",
            "content_score_max": "grounded_p2_content_score_abs_max",
            "intent_score_max": "grounded_p2_intent_score_abs_max",
            "coordinate_score_max": "grounded_p2_coordinate_score_abs_max",
            "query_coordinate_std": "grounded_p2_query_coordinate_std",
            "posterior_max": "grounded_p2_posterior_max",
            "posterior_H": "grounded_p2_posterior_entropy",
            "tau_content": "grounded_p2_content_temperature",
            "tau_intent": "grounded_p2_intent_temperature",
            "tau_coordinate": "grounded_p2_coordinate_temperature",
            "mass_h4_8": "grounded_p2_h4_8_mass",
            "mass_h8_16": "grounded_p2_h8_16_mass",
            "mass_h16_32": "grounded_p2_h16_32_mass",
            "mass_h32_48": "grounded_p2_h32_48_mass",
            "consequence_effect": "grounded_consequence_effect_rms",
            "consequence_interaction": "grounded_consequence_interaction_rms",
            "p3_precision": "grounded_p3_precision_rms",
            "p3_temporal": "grounded_p3_temporal_rms",
        },
    }
)

# V120 uses a capability-owned schema.  Keep separate alias families so its
# concise display names cannot overwrite the different V119 meanings of
# ``interval_var``, ``intent_score`` or ``p3_*``.
V94_ALIASES.update(
    {
        "object_ground": {
            "reconstruction": "object_grounding_reconstruction_mse",
            "prototype_mse": "object_grounding_prototype_mse",
            "spatial_refine_mse": (
                "object_grounding_spatial_refinement_mse"
            ),
            "typed_consistency": "object_grounding_typed_consistency",
            "semantic_consistency": "object_grounding_semantic_consistency",
            "appearance_consistency": "object_grounding_appearance_consistency",
            "geometry_consistency": "object_grounding_geometry_consistency",
            "existence": "object_grounding_existence_mean",
            "validity": "object_grounding_validity_mean",
            "allocation": "object_grounding_allocation_share_mean",
            "null": "object_grounding_null_mass",
            "mass_error": "object_grounding_mass_conservation_error",
            "owner_H": "object_grounding_candidate_owner_entropy",
            "local_prior_H": "object_grounding_local_prior_entropy",
            "chart_H": "object_grounding_chart_entropy",
            "g3_parent_l1": "object_grounding_g3_parent_l1",
            "object_pair_cos": "object_grounding_object_content_pair_cosine",
            "chart_pair_overlap": "object_grounding_object_chart_pair_overlap",
            "sem_app_post_l1": (
                "object_grounding_semantic_appearance_posterior_l1"
            ),
            "sem_geo_post_l1": (
                "object_grounding_semantic_geometry_posterior_l1"
            ),
            "app_geo_post_l1": (
                "object_grounding_appearance_geometry_posterior_l1"
            ),
            "flow_prior": "object_grounding_transport_prior_rms",
            "camera_coord_var": "object_grounding_camera_coordinate_variation",
        },
        "object_intent": {
            "goal_H": "object_intent_goal_attention_entropy",
            "interval_goal_H": "object_intent_interval_goal_entropy",
            "history_H": "object_intent_interval_history_entropy",
            "object_H": "object_intent_interval_object_entropy",
            "semantic_H": "object_intent_interval_semantic_entropy",
            "appearance_H": "object_intent_interval_appearance_entropy",
            "geometry_H": "object_intent_interval_geometry_entropy",
            "object_sim_H": "object_intent_interval_object_audit_similarity_entropy",
            "semantic_sim_H": "object_intent_interval_semantic_audit_similarity_entropy",
            "appearance_sim_H": "object_intent_interval_appearance_audit_similarity_entropy",
            "geometry_sim_H": "object_intent_interval_geometry_audit_similarity_entropy",
            "interval_var": "object_intent_interval_variation",
            "public_interval_var": "object_intent_public_interval_variation",
            "policy_interval_var": "object_intent_policy_interval_variation",
            "state_interval_var": "object_intent_interval_state_variation",
            "object_key_var": "object_intent_interval_object_key_variation",
            "object_value_var": "object_intent_interval_object_value_variation",
            "temporal_var": "object_intent_temporal_variation",
            "goal_innov": "object_intent_goal_innovation_rms",
            "history_innov": "object_intent_history_innovation_rms",
            "object_innov": "object_intent_object_innovation_rms",
            "typed_innov": "object_intent_typed_innovation_rms",
            "typed_action": "object_intent_typed_action_context_rms",
            "sem_rel": "object_intent_semantic_relevance_mass",
            "sem_null": "object_intent_semantic_null_mass",
            "sem_value": "object_intent_semantic_selected_value_rms",
            "sem_k_var": "object_intent_semantic_object_variation",
            "sem_i_var": "object_intent_semantic_interval_variation",
            "app_rel": "object_intent_appearance_relevance_mass",
            "app_null": "object_intent_appearance_null_mass",
            "app_value": "object_intent_appearance_selected_value_rms",
            "app_k_var": "object_intent_appearance_object_variation",
            "app_i_var": "object_intent_appearance_interval_variation",
            "geo_rel": "object_intent_geometry_relevance_mass",
            "geo_null": "object_intent_geometry_null_mass",
            "geo_value": "object_intent_geometry_selected_value_rms",
            "geo_k_var": "object_intent_geometry_object_variation",
            "geo_i_var": "object_intent_geometry_interval_variation",
            "action_innov": "object_intent_action_innovation_rms",
            "state_innov": "object_intent_state_innovation_rms",
            "temporal_innov": "object_intent_temporal_innovation_rms",
            "state_delta": "object_intent_observed_state_delta_rms",
            "transport": "object_intent_observed_transport_rms",
            "change_history": "object_intent_state_change_history_rms",
            "change_transport": "object_intent_state_change_transport_rms",
            "state_change": "object_intent_state_change_evidence_rms",
            "state_change_H": "object_intent_state_change_attention_entropy",
            "online_match": "object_intent_online_match_loss",
            "action_match": "object_intent_action_match_loss",
            "state_match": "object_intent_state_match_loss",
            "object_key_match": "object_intent_object_key_match_loss",
            "object_value_match": "object_intent_object_value_match_loss",
            "recognizer": "object_plan_recognition_loss",
            "coarse_action": "object_coarse_action_loss",
            "coarse_innov": "object_coarse_action_innovation_rms",
            "coarse_var": "object_coarse_action_interval_variation",
            "coarse_cos": "object_coarse_action_adjacent_cosine",
            "coarse_target_error": "object_coarse_action_target_normalized_error",
        },
        "object_dynamics": {
            "goal_H": "object_w_goal_attention_entropy",
            "goal_innov": "object_w_goal_innovation_rms",
            "physical_action": "object_w_physical_action_condition_rms",
            "physical_delta": "object_w_physical_action_delta_rms",
            "physical_var": "object_w_physical_action_interval_variation",
            "goal_direct": "object_w_goal_direct_ingress",
            "coarse_hidden_direct": "object_w_coarse_hidden_direct_ingress",
            "world_initial_materialization_count": "object_action_world_initial_materialization_count",
            "world_initial_tag_error": "object_action_world_initial_tag_identity_error",
            "world_refinement_count": "object_action_world_refinement_count",
            "world_refinement_pre_action_interval": "object_action_world_refinement_pre_action_interval_rms",
            "world_refinement_post_action_interval": "object_action_world_refinement_post_action_interval_rms",
            "world_refinement_action_interval_delta": "object_action_world_refinement_action_interval_delta_rms",
            "world_refinement_pre_semantic": "object_action_world_refinement_pre_semantic_delta_rms",
            "world_refinement_post_semantic": "object_action_world_refinement_post_semantic_delta_rms",
            "world_refinement_semantic_change": "object_action_world_refinement_semantic_delta_change_rms",
            "world_refinement_pre_transport": "object_action_world_refinement_pre_transport_rms",
            "world_refinement_post_transport": "object_action_world_refinement_post_transport_rms",
            "world_refinement_transport_change": "object_action_world_refinement_transport_change_rms",
            "world_refinement_tag_error": "object_action_world_refinement_tag_identity_error",
            "intent_innov": "object_w_interval_innovation_rms",
            "action_innov": "object_w_action_innovation_rms",
            "condition_interaction": "object_w_condition_interaction_rms",
            "state_innov": "object_w_state_innovation_rms",
            "object_key_innov": "object_w_object_key_innovation_rms",
            "object_value_innov": "object_w_object_value_innovation_rms",
            "typed_innov": "object_w_typed_innovation_rms",
            "typed_contribution": "object_w_typed_contribution_rms",
            "semantic_contribution": "object_w_semantic_contribution_rms",
            "appearance_contribution": "object_w_appearance_contribution_rms",
            "geometry_contribution": "object_w_geometry_contribution_rms",
            "w1_delta": "object_w1_semantic_delta_rms",
            "w1_transport": "object_w1_transport_rms",
            "w1_interval_cos": "object_w1_interval_adjacent_cosine",
            "w1_object_cos": "object_w1_object_pair_cosine",
            "w2_delta": "object_w2_semantic_delta_rms",
            "w2_transport": "object_w2_transport_rms",
            "w2_interval_cos": "object_w2_interval_adjacent_cosine",
            "w2_object_cos": "object_w2_object_pair_cosine",
            "teacher_visibility": "object_teacher_visibility",
            "teacher_visibility_change": "object_teacher_visibility_change",
            "teacher_persistence_change": "object_teacher_persistence_change",
            "teacher_null": "object_teacher_null_probability",
            "teacher_sem_max": "object_teacher_semantic_max",
            "teacher_sem_margin": "object_teacher_semantic_margin",
            "teacher_app_max": "object_teacher_appearance_max",
            "teacher_app_margin": "object_teacher_appearance_margin",
            "teacher_geom_margin": "object_teacher_geometry_margin",
            "teacher_uncert": "object_teacher_uncertainty",
            "teacher_delta": "object_teacher_semantic_delta_rms",
            "teacher_transport": "object_teacher_transport_rms",
            "teacher_covariance": "object_teacher_covariance_rms",
            "teacher_current_support": "object_teacher_current_loss_support",
            "teacher_future_selector": "object_teacher_future_selector_validity",
            "teacher_identity_error": "object_teacher_successor_delta_identity_max_abs",
            "teacher_supports": "object_teacher_supports_per_interval",
            "content_loss": "object_future_content",
            "successor_loss": "object_future_successor",
            "semantic_loss": "object_future_semantic",
            "transport_loss": "object_future_transport",
            "covariance_loss": "object_future_covariance",
            "visibility_loss": "object_future_visibility",
            "persistence_loss": "object_future_persistence",
            "uncertainty_loss": "object_future_uncertainty",
            "transition_loss": "object_future_transition",
            "pred_cos": "object_future_prediction_adjacent_cosine",
            "target_cos": "object_future_target_adjacent_cosine",
            "pred_var": "object_future_prediction_interval_variation",
            "target_var": "object_future_target_interval_variation",
        },
        "object_policy": {
            # Schema-2/3 ancestry remains parseable below; active schema-4 logs
            # expose the semantic and geometry routes independently.
            "semantic_score": "object_p2_semantic_score_abs",
            "semantic_score_max": "object_p2_semantic_score_max_abs",
            "geometry_score": "object_p2_geometry_score_abs",
            "geometry_score_max": "object_p2_geometry_score_max_abs",
            "address_score": "object_p2_address_score_abs",
            "transport_score": "object_p2_transport_score_abs",
            "semantic_logit_max": "object_p2_semantic_logit_max_abs",
            "geometry_logit_max": "object_p2_geometry_logit_max_abs",
            "semantic_H": "object_p2_semantic_posterior_entropy",
            "geometry_H": "object_p2_geometry_posterior_entropy",
            "semantic_max": "object_p2_semantic_posterior_max",
            "geometry_max": "object_p2_geometry_posterior_max",
            "semantic_null": "object_p2_semantic_null_mass",
            "geometry_null": "object_p2_geometry_null_mass",
            "calibration": "object_p2_selector_calibration",
            "relative_status_abs": "object_p2_relative_status_abs",
            "relative_status_mean": "object_p2_relative_status_mean",
            "content_score": "object_p2_content_score_abs",
            "content_score_max": "object_p2_content_score_max_abs",
            "intent_score": "object_p2_intent_score_abs",
            "intent_score_max": "object_p2_intent_score_max_abs",
            "coordinate_score": "object_p2_coordinate_score_abs",
            "coordinate_score_max": "object_p2_coordinate_score_max_abs",
            "combined_logit_max": "object_p2_combined_logit_max_abs",
            "tau_content": "object_p2_temperature_content",
            "tau_intent": "object_p2_temperature_intent",
            "tau_coordinate": "object_p2_temperature_coordinate",
            "posterior_H": "object_p2_posterior_entropy",
            "posterior_max": "object_p2_posterior_max",
            "null": "object_p2_null_mass",
            "semantic_mass": "object_p2_semantic_value_mass",
            "geometry_mass": "object_p2_geometry_value_mass",
            "status_mass": "object_p2_status_value_mass",
            "semantic_h4_8_mass": "object_p2_semantic_interval_0_mass",
            "semantic_h8_16_mass": "object_p2_semantic_interval_1_mass",
            "semantic_h16_32_mass": "object_p2_semantic_interval_2_mass",
            "semantic_h32_48_mass": "object_p2_semantic_interval_3_mass",
            "geometry_h4_8_mass": "object_p2_geometry_interval_0_mass",
            "geometry_h8_16_mass": "object_p2_geometry_interval_1_mass",
            "geometry_h16_32_mass": "object_p2_geometry_interval_2_mass",
            "geometry_h32_48_mass": "object_p2_geometry_interval_3_mass",
            "h4_8_mass": "object_p2_interval_0_mass",
            "h8_16_mass": "object_p2_interval_1_mass",
            "h16_32_mass": "object_p2_interval_2_mass",
            "h32_48_mass": "object_p2_interval_3_mass",
            "effect_precontract": "object_p2_effect_precontract_rms",
            "effect": "object_p2_effect_rms",
            "contract_min": "object_p2_contract_min",
            "consequence_effect": "object_consequence_effect_rms",
            "interaction": "object_consequence_interaction_rms",
            "consequence_ratio": "object_consequence_ratio",
            "p3_factual": "object_p3_factual_rms",
            "p3_precision": "object_p3_precision_rms",
            "p3_effect": "object_p3_effect_rms",
            "p3_temporal": "object_p3_temporal_rms",
            "p3_state_change": "object_p3_state_change_rms",
            "p3_centered_detail": "object_p3_centered_detail_rms",
            "p3_consequence_innov": "object_p3_consequence_innovation_rms",
            "p3_precision_null": "object_p3_precision_null_mass",
        },
    }
)

_V119_INTERVAL_NAMES = ("h4_8", "h8_16", "h16_32", "h32_48")
_V119_EFFECT_ERROR_FIELDS = {
    "teacher_reliability": "teacher_reliability",
    "successor": "successor",
    "semantic": "semantic",
    "transport": "transport",
    "covariance": "transport_covariance",
    "persistence": "persistence_change",
    "visibility": "visibility_change",
    "uncertainty": "uncertainty_calibration",
    "reliability": "reliability_calibration",
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
    "gripper_trajectory",
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
    # Capability-owned mainline: keep the decisive G/S/W/P and bottom
    # boundaries visible in the text report.  The lossless metric index has
    # always retained them; without this projection a healthy schema-19 run
    # still looked like an RMSE-only run when audited from nohup/JSONL.
    "object_grounding_reconstruction_mse",
    "object_grounding_prototype_mse",
    "object_grounding_spatial_refinement_mse",
    "object_grounding_object_content_pair_cosine",
    "object_grounding_object_chart_pair_overlap",
    "object_grounding_typed_consistency",
    "object_grounding_camera_coordinate_variation",
    "object_grounding_g3_parent_l1",
    "object_intent_goal_attention_entropy",
    "object_intent_interval_variation",
    "object_intent_public_interval_variation",
    "object_intent_policy_interval_variation",
    "object_intent_interval_object_key_variation",
    "object_intent_interval_object_value_variation",
    "object_intent_temporal_variation",
    "object_intent_goal_innovation_rms",
    "object_intent_history_innovation_rms",
    "object_intent_object_innovation_rms",
    "object_intent_action_innovation_rms",
    "object_intent_state_change_evidence_rms",
    "object_intent_typed_carrier_ratio",
    "object_intent_typed_action_context_rms",
    "object_intent_semantic_route_raw_rms",
    "object_intent_semantic_relevance_mass",
    "object_intent_semantic_null_mass",
    "object_intent_semantic_selected_value_rms",
    "object_intent_semantic_object_variation",
    "object_intent_semantic_interval_variation",
    "object_intent_semantic_action_context_rms",
    "object_intent_semantic_score_abs",
    "object_intent_semantic_temperature",
    "object_intent_appearance_route_raw_rms",
    "object_intent_appearance_relevance_mass",
    "object_intent_appearance_null_mass",
    "object_intent_appearance_selected_value_rms",
    "object_intent_appearance_object_variation",
    "object_intent_appearance_interval_variation",
    "object_intent_appearance_action_context_rms",
    "object_intent_appearance_score_abs",
    "object_intent_appearance_temperature",
    "object_intent_geometry_route_raw_rms",
    "object_intent_geometry_relevance_mass",
    "object_intent_geometry_null_mass",
    "object_intent_geometry_selected_value_rms",
    "object_intent_geometry_object_variation",
    "object_intent_geometry_interval_variation",
    "object_intent_geometry_action_context_rms",
    "object_intent_geometry_score_abs",
    "object_intent_geometry_temperature",
    "object_w_physical_action_condition_rms",
    "object_w_physical_action_delta_rms",
    "object_w_physical_action_interval_variation",
    "object_w_physical_action_carrier_rms",
    "object_w_goal_direct_ingress",
    "object_w_coarse_hidden_direct_ingress",
    "object_action_world_initial_materialization_count",
    "object_action_world_initial_tag_identity_error",
    "object_action_world_refinement_count",
    "object_action_world_refinement_pre_action_interval_rms",
    "object_action_world_refinement_post_action_interval_rms",
    "object_w_object_key_innovation_rms",
    "object_w_object_value_innovation_rms",
    "object_w_intent_object_interaction_rms",
    "object_w_action_object_interaction_rms",
    "object_w_typed_contribution_rms",
    "object_w_semantic_contribution_rms",
    "object_w_semantic_contribution_interval_variation",
    "object_w_semantic_contribution_object_variation",
    "object_w_semantic_input_relevance_mass",
    "object_w_semantic_input_value_rms",
    "object_w_semantic_input_interval_variation",
    "object_w_semantic_input_object_variation",
    "object_w_appearance_contribution_rms",
    "object_w_appearance_contribution_interval_variation",
    "object_w_appearance_contribution_object_variation",
    "object_w_appearance_input_relevance_mass",
    "object_w_appearance_input_value_rms",
    "object_w_appearance_input_interval_variation",
    "object_w_appearance_input_object_variation",
    "object_w_geometry_contribution_rms",
    "object_w_geometry_contribution_interval_variation",
    "object_w_geometry_contribution_object_variation",
    "object_w_geometry_input_relevance_mass",
    "object_w_geometry_input_value_rms",
    "object_w_geometry_input_interval_variation",
    "object_w_geometry_input_object_variation",
    "object_w_prediction_interval_variation",
    "object_w1_semantic_delta_rms",
    "object_w1_transport_rms",
    "object_w1_interval_adjacent_cosine",
    "object_w1_object_pair_cosine",
    "object_w2_semantic_delta_rms",
    "object_w2_transport_rms",
    "object_w2_interval_adjacent_cosine",
    "object_w2_object_pair_cosine",
    "object_action_world_refinement_action_interval_delta_rms",
    "object_action_world_refinement_pre_action_delta_rms",
    "object_action_world_refinement_post_action_delta_rms",
    "object_action_world_refinement_pre_semantic_delta_rms",
    "object_action_world_refinement_post_semantic_delta_rms",
    "object_action_world_refinement_semantic_delta_change_rms",
    "object_action_world_refinement_pre_transport_rms",
    "object_action_world_refinement_post_transport_rms",
    "object_action_world_refinement_transport_change_rms",
    "object_action_world_refinement_tag_identity_error",
    "object_teacher_reliability",
    "object_teacher_semantic_delta_rms",
    "object_teacher_transport_rms",
    "object_teacher_covariance_rms",
    "object_teacher_current_loss_support",
    "object_teacher_future_selector_validity",
    "object_teacher_successor_delta_identity_max_abs",
    "loss_future_current_loss_support",
    "loss_future_prediction_selector_validity",
    "loss_future_target_selector_validity",
    "sampling_update_time_first",
    "sampling_update_time_last",
    "sampling_endpoint_head_time",
    "sampling_velocity_update_calls",
    "sampling_endpoint_head_calls",
    "sampling_outer_world_refinement",
    "sampling_outer_proposal_action_rms",
    "sampling_outer_refined_action_rms",
    "sampling_outer_refined_action_delta_rms",
    "sampling_outer_final_world_action_interval_rms",
    "sampling_outer_final_world_action_delta_rms",
    "sampling_outer_final_world_action_interval_mismatch_rms",
    "sampling_outer_final_world_action_delta_mismatch_rms",
    "object_teacher_adjacent_cosine",
    "object_teacher_interval_variation",
    "p1_query_chart_variation",
    "p1_query_coordinate_variation",
    "object_p2_semantic_posterior_entropy",
    "object_p2_geometry_posterior_entropy",
    "object_p2_semantic_value_mass",
    "object_p2_geometry_value_mass",
    "object_p2_geometry_address_correction_rms",
    "object_p2_geometry_address_k_center_error",
    "object_p2_successor_innovation_rms",
    "object_consequence_effect_rms",
    "object_consequence_interaction_rms",
    "object_p3_precision_base_rms",
    "object_p3_precision_consequence_interaction_rms",
    "object_p3_temporal_base_rms",
    "object_p3_temporal_consequence_interaction_rms",
    "controlled_transition_spatial_value_variation",
    "controlled_transition_delta_gain",
    "bottom_capacity_mean",
    "bottom_capacity_block_std",
    "bottom_expected_depth",
    "bottom_controller_common_ratio",
    "bottom_controller_private_ratio",
    "bottom_protected_update_rms",
    "bottom_controlled_transition_value_rms",
    "gripper_private_gate_rms",
    "gripper_private_state_delta_rms",
    "gripper_trajectory_mask_fraction",
    "gripper_trajectory_absolute_loss",
    "gripper_trajectory_delta_loss",
    "gripper_trajectory_transition",
    "gripper_trajectory_persistence",
    "gripper_trajectory_transition_mask_fraction",
    "gripper_trajectory_persistence_mask_fraction",
    "gripper_trajectory_branch_disagreement_rms",
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
    # Schema-19 optimizer owners.  These are post-clip role norms by design;
    # the global pre-clip norm remains the shared scale/instability sentinel.
    "gradient_global_preclip_l2",
    "gradient_postclip_observation_l2",
    "gradient_postclip_grounding_host_l2",
    "gradient_postclip_grounder_l2",
    "gradient_postclip_intent_l2",
    "gradient_postclip_coarse_action_l2",
    "gradient_postclip_plan_recognizer_l2",
    "gradient_postclip_history_proposal_l2",
    "gradient_postclip_dynamics_l2",
    "gradient_postclip_controlled_transition_l2",
    "gradient_postclip_p1_factual_l2",
    "gradient_postclip_p2_effect_reader_l2",
    "gradient_postclip_consequence_l2",
    "gradient_postclip_p3_compiler_l2",
    "gradient_postclip_v120_canvas_seed_l2",
    "gradient_postclip_v120_layer_contracts_l2",
    "gradient_postclip_bottom_query_l2",
    "gradient_postclip_bottom_evidence_adapter_l2",
    "gradient_postclip_bottom_policy_bridge_l2",
    # Historical pre-recovery role names remain readable when comparing old logs.
    "gradient_postclip_bottom_protected_reader_l2",
    "gradient_postclip_bottom_evidence_compiler_l2",
    "gradient_postclip_bottom_organizer_l2",
    "gradient_postclip_bottom_mmdit_l2",
    "gradient_postclip_bottom_capacity_l2",
    "gradient_postclip_bottom_execution_l2",
    "gradient_postclip_bottom_heads_l2",
    # Schema24 exposes the actual three-stage V120 lifecycle.  Keep the old
    # postclip rows above so historical runs remain comparable, but never
    # relabel a post-global value as a raw owner gradient.
    *(
        f"gradient_{stage}_{role}_l2"
        for stage in ("raw", "postlocal", "postglobal")
        for role in (
            "observation",
            "grounding",
            "grounder",
            "intent",
            "coarse_action",
            "plan_recognizer",
            "history_proposal",
            "dynamics",
            "controlled_transition",
            "p1_factual",
            "p2_effect_reader",
            "consequence",
            "p3_compiler",
            "v120_canvas_seed",
            "v120_layer_contracts",
            "bottom_query",
            "bottom_evidence_adapter",
            "bottom_policy_bridge",
            "bottom_organizer",
            "bottom_mmdit",
            "bottom_capacity",
            "bottom_execution",
            "bottom_heads",
        )
    ),
    "gradient_raw_global_l2",
    "gradient_postlocal_global_l2",
    "gradient_postglobal_global_l2",
    "gradient_raw_bottom_decoder_l2",
    "gradient_postlocal_bottom_decoder_l2",
    "gradient_postglobal_bottom_decoder_l2",
    "gradient_tensor_p2_semantic_effect_rms",
    "gradient_tensor_p2_geometry_effect_rms",
    "gradient_tensor_p2_geometry_address_correction_rms",
    "gradient_tensor_w_semantic_fact_ingress_rms",
    "gradient_tensor_w_appearance_fact_ingress_rms",
    "gradient_tensor_w_geometry_fact_ingress_rms",
    "gradient_tensor_w_physical_action_condition_rms",
    "gradient_parameter_gripper_private_gate_weight_rms",
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
    "action_band_1_4_rmse",
    "action_band_5_12_rmse",
    "action_band_13_24_rmse",
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
    "validation_ablation_coverage",
    "validation_diagnostic_primary_rmse_physical",
    "validation_proposal_zero_rmse_physical",
    "validation_proposal_zero_mse_gain_vs_primary_physical",
    "validation_proposal_zero_action_delta_rmse_physical",
    "validation_execution_no_updates_rmse_physical",
    "validation_execution_no_updates_mse_gain_vs_primary_physical",
    "validation_execution_no_updates_action_delta_rmse_physical",
    "validation_execution_full_updates_rmse_physical",
    "validation_execution_full_updates_mse_gain_vs_primary_physical",
    "validation_execution_full_updates_action_delta_rmse_physical",
    "validation_sampling_diagnostic_coverage",
    "validation_proposal_ablation_coverage",
    "validation_execution_ablation_coverage",
    "validation_proposal_primary_rmse_physical",
    "validation_execution_primary_rmse_physical",
    "validation_execution_hard_rmse_physical",
    "validation_execution_hard_mse_gain_vs_primary_physical",
    "validation_execution_hard_action_delta_rmse_physical",
    "validation_execution_neutral_rmse_physical",
    "validation_execution_neutral_mse_gain_vs_primary_physical",
    "validation_execution_neutral_action_delta_rmse_physical",
    "validation_execution_full_capacity_rmse_physical",
    "validation_execution_full_capacity_mse_gain_vs_primary_physical",
    "validation_execution_full_capacity_action_delta_rmse_physical",
    "validation_execution_three_basis_reduction_rmse_physical",
    "validation_execution_three_basis_reduction_mse_gain_vs_primary_physical",
    "validation_execution_three_basis_reduction_action_delta_rmse_physical",
    "balanced_score",
    "deploy_eligible",
    "sample_evidence_z_zero_condition_delta",
    "sample_evidence_z_shuffle_condition_delta",
    # Lossless mainline validation names.  Historical aliases above keep
    # cross-version tables comparable; these retain units, horizon bands and
    # causal intervention semantics needed to diagnose a recovery run.
    "validation_action_rmse_normalized",
    "validation_action_rmse_physical",
    "validation_first_rmse_normalized",
    "validation_first_rmse_physical",
    "validation_first8_rmse_normalized",
    "validation_first8_rmse_physical",
    "validation_tail_rmse_normalized",
    "validation_tail_rmse_physical",
    "validation_tail_first_ratio_normalized",
    "validation_tail_first_ratio_physical",
    "validation_band_1_4_rmse_normalized",
    "validation_band_1_4_rmse_physical",
    "validation_band_5_12_rmse_normalized",
    "validation_band_5_12_rmse_physical",
    "validation_band_13_24_rmse_normalized",
    "validation_band_13_24_rmse_physical",
    "validation_arm_rmse_normalized",
    "validation_arm_rmse_physical",
    "validation_gripper_rmse_normalized",
    "validation_gripper_rmse_physical",
    "validation_decoded_gripper_event_precision",
    "validation_decoded_gripper_event_recall",
    "validation_decoded_gripper_event_f1",
    "validation_decoded_gripper_event_ratio",
    "validation_event_head_precision",
    "validation_event_head_recall",
    "validation_event_head_f1",
    "validation_motion_head_precision",
    "validation_motion_head_recall",
    "validation_motion_head_f1",
    "validation_proposal_zero_rmse_normalized",
    "validation_execution_no_updates_rmse_normalized",
    "validation_execution_full_updates_rmse_normalized",
    "validation_execution_primary_rmse_normalized",
    "validation_execution_hard_rmse_normalized",
    "validation_execution_neutral_rmse_normalized",
    "validation_execution_full_capacity_rmse_normalized",
    "validation_execution_three_basis_reduction_rmse_normalized",
    # Schema28 validation prefixes deployment-only outer-closure diagnostics
    # so they cannot be confused with the teacher-forced loss forward.
    "validation_deploy_sampling_outer_world_refinement",
    "validation_deploy_sampling_outer_proposal_action_rms",
    "validation_deploy_sampling_outer_refined_action_rms",
    "validation_deploy_sampling_outer_refined_action_delta_rms",
    "validation_deploy_sampling_outer_final_world_action_interval_rms",
    "validation_deploy_sampling_outer_final_world_action_delta_rms",
    "validation_deploy_sampling_outer_final_world_action_interval_mismatch_rms",
    "validation_deploy_sampling_outer_final_world_action_delta_mismatch_rms",
    "validation_deploy_object_action_world_refinement_count",
    "validation_deploy_object_action_world_refinement_pre_action_interval_rms",
    "validation_deploy_object_action_world_refinement_post_action_interval_rms",
    "validation_deploy_object_action_world_refinement_action_interval_delta_rms",
    "validation_deploy_object_action_world_refinement_pre_action_delta_rms",
    "validation_deploy_object_action_world_refinement_post_action_delta_rms",
    "validation_deploy_object_action_world_refinement_pre_semantic_delta_rms",
    "validation_deploy_object_action_world_refinement_post_semantic_delta_rms",
    "validation_deploy_object_action_world_refinement_semantic_delta_change_rms",
    "validation_deploy_object_action_world_refinement_pre_transport_rms",
    "validation_deploy_object_action_world_refinement_post_transport_rms",
    "validation_deploy_object_action_world_refinement_transport_change_rms",
    "validation_deploy_object_action_world_refinement_tag_identity_error",
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
    runtime_points: list[dict[str, float]] = field(default_factory=list)
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


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = min(max(float(quantile), 0.0), 1.0) * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


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
        horizon_group_suffix = {
            "future_scale": "target_scale",
            "future_norm_scale": "normalization_scale",
            "future_rel": "reliability",
            "future_direction": "active_direction",
            "future_active": "active_loss",
        }.get(key)
        if horizon_group_suffix is not None:
            for item in raw.split("/"):
                if ":" not in item:
                    continue
                offset, value_raw = item.split(":", 1)
                value = _number(value_raw)
                if value is not None and offset.isdigit():
                    metrics[f"flow_jepa_future_horizon_{offset}_{horizon_group_suffix}"] = value
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


def _parse_v119_effect_error(line: str) -> dict[str, float]:
    """Decode one interval-qualified V119 target-normalized error row."""

    tokens = {match.group("key"): match.group("value").rstrip(",;") for match in TOKEN_RE.finditer(line)}
    interval = tokens.get("interval")
    if interval not in _V119_INTERVAL_NAMES:
        return {}
    metrics: dict[str, float] = {}
    for display_name, field_name in _V119_EFFECT_ERROR_FIELDS.items():
        value = _number(tokens.get(display_name, ""))
        if value is None:
            continue
        if display_name == "teacher_reliability":
            canonical = f"grounded_future_effect_teacher_reliability_{interval}"
        else:
            canonical = (
                f"grounded_future_effect_{field_name}_{interval}"
                "_target_normalized_error"
            )
        metrics[canonical] = value
    return metrics


def _parse_v120_dynamics_error(line: str) -> dict[str, float]:
    """Decode one interval-qualified V120 object-dynamics error row."""

    tokens = {
        match.group("key"): match.group("value").rstrip(",;")
        for match in TOKEN_RE.finditer(line)
    }
    interval = tokens.get("interval")
    if interval not in _V119_INTERVAL_NAMES:
        return {}
    metrics: dict[str, float] = {}
    for field_name in (
        "successor",
        "semantic",
        "transport",
        "covariance",
        "visibility",
        "persistence",
        "uncertainty",
    ):
        value = _number(tokens.get(field_name, ""))
        if value is not None:
            metrics[
                f"object_future_{field_name}_{interval}_normalized_error"
            ] = value
    return metrics


_MAINLINE_ALIASES: dict[str, str] = {
    "loss_total": "loss",
    "loss_action_flow": "physical_flow",
    "loss_action_flow_v120_comparable": "physical_flow",
    "loss_action_flow_event_balanced_audit": "physical_flow_event_balanced",
    "loss_action_flow_native": "physical_flow_native",
    "loss_action_arm_flow": "arm_fm_per_dim",
    "loss_action_gripper_flow": "gripper_fm_field",
    "loss_action_gripper_flow_event_balanced_audit": (
        "gripper_fm_field_event_balanced"
    ),
    "loss_action_gripper_flow_unweighted": "gripper_fm_field",
    "loss_decoded_action": "decoded_action",
    "loss_decoded_action_v120_comparable": "decoded_action",
    "loss_decoded_action_event_balanced_audit": "decoded_action_event_balanced",
    "loss_gripper_trajectory": "gripper_trajectory",
    "loss_future_successor": "flow_jepa_future_prediction",
    "loss_future_semantic_delta": "flow_jepa_future_change",
    "loss_flow_warp": "flow_jepa_warp_loss",
    "loss_flow_identity_advantage": "flow_jepa_identity_advantage_loss",
    "loss_flow_static_identity": "flow_jepa_static_identity_loss",
    "loss_flow_cycle": "flow_jepa_cycle_loss",
    "bottom_capacity_mean": "evidence_mmd_it_capacity_ratio",
    "bottom_expected_depth": "evidence_mmd_it_effective_depth",
    "bottom_execution_cost_audit": "evidence_mmd_it_execution_cost",
    "loss_execution_value": "evidence_mmd_it_execution_value_loss",
    "loss_execution_value_target_spread": (
        "evidence_mmd_it_execution_value_target_spread"
    ),
    "loss_execution_value_predicted_spread": (
        "evidence_mmd_it_execution_value_predicted_spread"
    ),
    "loss_execution_value_predicted_standardized_spread": (
        "evidence_mmd_it_execution_value_predicted_standardized_spread"
    ),
    "loss_execution_value_target_spread_p25": (
        "evidence_mmd_it_execution_value_target_spread_p25"
    ),
    "loss_execution_value_target_spread_p50": (
        "evidence_mmd_it_execution_value_target_spread_p50"
    ),
    "loss_execution_value_target_spread_p75": (
        "evidence_mmd_it_execution_value_target_spread_p75"
    ),
    "loss_execution_value_correlation": (
        "evidence_mmd_it_execution_value_correlation"
    ),
    "loss_execution_value_pairwise_accuracy": (
        "evidence_mmd_it_execution_value_pairwise_accuracy"
    ),
    "loss_execution_value_decision_accuracy": (
        "evidence_mmd_it_execution_value_decision_accuracy"
    ),
    "loss_execution_value_common_mode_ratio": (
        "evidence_mmd_it_execution_value_common_mode_ratio"
    ),
    "loss_execution_candidate_coverage": (
        "evidence_mmd_it_execution_candidate_coverage"
    ),
    "loss_execution_terminal_identity_error": (
        "evidence_mmd_it_terminal_identity_velocity_error"
    ),
    "loss_execution_terminal_target_cost_margin": (
        "evidence_mmd_it_terminal_target_cost_margin"
    ),
    "loss_execution_terminal_predicted_cost_margin": (
        "evidence_mmd_it_terminal_predicted_cost_margin"
    ),
    "loss_execution_terminal_target_preferred_fraction": (
        "evidence_mmd_it_terminal_target_preferred_fraction"
    ),
    "validation_sampling_diagnostic_coverage": "eval_sampling_diagnostic_coverage",
    "validation_proposal_ablation_coverage": "eval_proposal_ablation_coverage",
    "validation_execution_ablation_coverage": "eval_execution_ablation_coverage",
    "validation_execution_primary_rmse_physical": (
        "execution_ablation_primary_full_rmse"
    ),
    "validation_execution_hard_rmse_physical": (
        "execution_ablation_hard_full_rmse"
    ),
    "validation_execution_neutral_rmse_physical": (
        "execution_ablation_neutral_full_rmse"
    ),
    "validation_execution_full_capacity_rmse_physical": (
        "execution_ablation_full_capacity_full_rmse"
    ),
    "validation_execution_three_basis_reduction_rmse_physical": (
        "execution_ablation_three_basis_reduction_full_rmse"
    ),
}


def _canonicalize_mainline_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Keep lossless mainline names while adding cross-version audit aliases."""

    metrics = {
        str(name): number
        for name, value in values.items()
        if (number := _number(value)) is not None
    }
    for source, target in _MAINLINE_ALIASES.items():
        if source in metrics:
            metrics.setdefault(target, metrics[source])

    # Schema 19 trained the event-balanced rows and carried the V120 rows as
    # explicit comparables. Schema 20 restored the V120 rows as the formal
    # objective. Prefer the semantic comparable whenever it is present, while
    # retaining the older formal row under its event-balanced audit name.
    if "loss_action_flow_v120_comparable" in metrics:
        metrics["physical_flow"] = metrics["loss_action_flow_v120_comparable"]
        metrics.setdefault(
            "physical_flow_event_balanced",
            metrics.get(
                "loss_action_flow_event_balanced_audit",
                metrics.get(
                    "loss_action_flow",
                    metrics["loss_action_flow_v120_comparable"],
                ),
            ),
        )
    if "loss_decoded_action_v120_comparable" in metrics:
        metrics["decoded_action"] = metrics["loss_decoded_action_v120_comparable"]
        metrics.setdefault(
            "decoded_action_event_balanced",
            metrics.get(
                "loss_decoded_action_event_balanced_audit",
                metrics.get(
                    "loss_decoded_action",
                    metrics["loss_decoded_action_v120_comparable"],
                ),
            ),
        )
    if "loss_action_gripper_flow_unweighted" in metrics:
        metrics["gripper_fm_field"] = metrics[
            "loss_action_gripper_flow_unweighted"
        ]
        metrics.setdefault(
            "gripper_fm_field_event_balanced",
            metrics.get(
                "loss_action_gripper_flow_event_balanced_audit",
                metrics.get(
                    "loss_action_gripper_flow",
                    metrics["loss_action_gripper_flow_unweighted"],
                ),
            ),
        )

    # Pre-schema-19-recovery mainline logs did not serialize the separate
    # V120-comparable event-unweighted rows.  Preserve their historical parser
    # behavior, while new logs use the unambiguous aliases above.
    if "physical_flow" not in metrics and "loss_action_flow" in metrics:
        metrics["physical_flow"] = metrics["loss_action_flow"]
    if "gripper_fm_field" not in metrics and "loss_action_gripper_flow" in metrics:
        metrics["gripper_fm_field"] = metrics["loss_action_gripper_flow"]
    if "decoded_action" not in metrics and "loss_decoded_action" in metrics:
        metrics["decoded_action"] = metrics["loss_decoded_action"]

    # Historical validation summaries use physical action units.  Preserve
    # both native mainline names and the old aliases, falling back to
    # normalized units only for a dataset that has no physical normalizer.
    validation_aliases = {
        "action": "full_rmse",
        "first": "first_rmse",
        "first8": "first8_rmse",
        "tail": "tail_rmse",
        "arm": "arm_full_rmse",
        "gripper": "gripper_full_rmse",
    }
    for stem, target in validation_aliases.items():
        physical = f"validation_{stem}_rmse_physical"
        normalized = f"validation_{stem}_rmse_normalized"
        if physical in metrics:
            metrics.setdefault(target, metrics[physical])
        elif normalized in metrics:
            metrics.setdefault(target, metrics[normalized])
    if "validation_tail_first_ratio_physical" in metrics:
        metrics.setdefault(
            "tail_first_ratio", metrics["validation_tail_first_ratio_physical"]
        )
    elif "validation_tail_first_ratio_normalized" in metrics:
        metrics.setdefault(
            "tail_first_ratio", metrics["validation_tail_first_ratio_normalized"]
        )
    event_aliases = {
        "validation_decoded_gripper_event_precision": "gripper_event_precision",
        "validation_decoded_gripper_event_recall": "gripper_event_recall",
        "validation_decoded_gripper_event_f1": "gripper_event_f1",
        "validation_decoded_gripper_event_ratio": "gripper_event_ratio",
        "validation_decoded_gripper_events_predicted": "gripper_pred_events",
        "validation_decoded_gripper_events_target": "gripper_target_events",
        "validation_event_head_precision": "event_head_precision",
        "validation_event_head_recall": "event_head_recall",
        "validation_event_head_f1": "event_head_f1",
        "validation_event_head_events_predicted": "event_head_pred_events",
        "validation_event_head_events_target": "event_head_target_events",
        "validation_motion_head_precision": "motion_head_precision",
        "validation_motion_head_recall": "motion_head_recall",
        "validation_motion_head_f1": "motion_head_f1",
        "validation_gripper_event_precision_normalized": "gripper_event_precision",
        "validation_gripper_event_recall_normalized": "gripper_event_recall",
        "validation_gripper_event_f1_normalized": "gripper_event_f1",
        "validation_gripper_event_ratio_normalized": "gripper_event_ratio",
        "validation_gripper_events_predicted_normalized": "gripper_pred_events",
        "validation_gripper_events_target_normalized": "gripper_target_events",
        "validation_event_head_precision_normalized": "event_head_precision",
        "validation_event_head_recall_normalized": "event_head_recall",
        "validation_event_head_f1_normalized": "event_head_f1",
        "validation_event_head_events_predicted_normalized": "event_head_pred_events",
        "validation_event_head_events_target_normalized": "event_head_target_events",
        "validation_motion_precision_normalized": "motion_head_precision",
        "validation_motion_recall_normalized": "motion_head_recall",
        "validation_motion_f1_normalized": "motion_head_f1",
    }
    for source, target in event_aliases.items():
        if source in metrics:
            metrics.setdefault(target, metrics[source])
    return metrics


def _parse_mainline_tokens(line: str) -> dict[str, float]:
    values = {
        match.group("key"): match.group("value").rstrip(",;")
        for match in TOKEN_RE.finditer(line)
    }
    return _canonicalize_mainline_metrics(values)


def _ingest_json(run: ParsedRun, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    kind = payload.get("kind")
    if kind == "train" and isinstance(payload.get("metrics"), Mapping):
        run.batch_points.append(
            BatchPoint(
                int(payload.get("epoch", 0)),
                int(payload.get("batch", 0)),
                _canonicalize_mainline_metrics(payload["metrics"]),
                "mainline",
            )
        )
        return
    if kind == "epoch":
        train = payload.get("train")
        validation = payload.get("validation", payload.get("val"))
        run.epoch_records.append(
            {
                "epoch": int(payload.get("epoch", 0)),
                "global_step": int(payload.get("step", payload.get("global_step", 0))),
                "train": _canonicalize_mainline_metrics(train)
                if isinstance(train, Mapping)
                else {},
                "validation": _canonicalize_mainline_metrics(validation)
                if isinstance(validation, Mapping)
                else {},
            }
        )
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
                        f"{exception_match.group('type')}: {exception_match.group('message')}"
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
            if line.startswith(("[mainline-runtime]", "[mainline-memory]")):
                run.runtime_points.append(_parse_mainline_tokens(line))
                continue
            if line.startswith("[mainline-train]"):
                flush_v94()
                metrics = _parse_mainline_tokens(line)
                epoch = int(metrics.pop("epoch", 0.0))
                batch = int(metrics.pop("batch", 0.0))
                metrics.pop("step", None)
                pending_v94 = BatchPoint(epoch, batch, metrics, "mainline")
                continue
            mainline_train_detail = re.match(r"^\[mainline-train-[^\]]+\]", line)
            if mainline_train_detail is not None and pending_v94 is not None:
                detail = _parse_mainline_tokens(line)
                for name in ("epoch", "batch", "step"):
                    detail.pop(name, None)
                pending_v94.metrics.update(detail)
                continue
            if line.startswith("[mainline-val]"):
                flush_v94()
                flush_epoch()
                metrics = _parse_mainline_tokens(line)
                epoch = int(metrics.pop("epoch", 0.0))
                global_step = int(metrics.pop("step", 0.0))
                metrics.pop("batch", None)
                pending_epoch = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train": {},
                    "validation": metrics,
                }
                continue
            mainline_val_detail = re.match(r"^\[mainline-val-[^\]]+\]", line)
            if mainline_val_detail is not None and pending_epoch is not None:
                detail = _parse_mainline_tokens(line)
                for name in ("epoch", "batch", "step"):
                    detail.pop(name, None)
                pending_epoch["validation"].update(detail)
                continue
            mainline_header = MAINLINE_HEADER_RE.match(line)
            if mainline_header is not None:
                run.headers.append(line)
                config = _parse_header_body(mainline_header.group("body"))
                for key, value in config.items():
                    run.header_config.setdefault(key, value)
                continue
            if line.startswith("[v39-layer]"):
                flush_v94()
                metrics = _parse_legacy_tokens(line)
                epoch = int(metrics.pop("epoch", 0.0))
                batch = int(metrics.pop("batch", 0.0))
                if epoch or batch:
                    run.batch_points.append(BatchPoint(epoch, batch, metrics, "v39-layer"))
                continue
            if line.startswith(_compact_prefixes("train") + ("[v95-stage1-train]",)):
                flush_v94()
                metrics = _parse_v94_tokens(line, "train")
                epoch = int(metrics.pop("epoch", 0.0))
                batch = int(metrics.pop("batch", 0.0))
                if line.startswith("[v95-stage1-train]") and "loss" in metrics:
                    metrics.setdefault("loss_group_representation", metrics["loss"])
                    source = "v95-stage1"
                else:
                    version_match = re.match(r"^\[(v\d+)-", line)
                    if version_match is None:
                        raise RuntimeError("compact train row lost its version prefix")
                    source = version_match.group(1)
                pending_v94 = BatchPoint(epoch, batch, metrics, source)
                continue
            if line.startswith(_compact_prefixes("exec")) and pending_v94 is not None:
                pending_v94.metrics.update(_parse_v94_tokens(line, "exec"))
                continue
            if line.startswith(_compact_prefixes("grad") + ("[v95-stage1-grad]",)) and (
                pending_v94 is not None
            ):
                pending_v94.metrics.update(_parse_v94_tokens(line, "grad"))
                continue
            if line.startswith(_compact_prefixes("repr") + ("[v95-stage1-repr]",)) and (
                pending_v94 is not None
            ):
                pending_v94.metrics.update(_parse_v94_tokens(line, "repr"))
                continue
            v119_auxiliary = re.match(
                r"^\[v119-(ground|intent|effect|policy)\]",
                line,
            )
            if v119_auxiliary is not None and pending_v94 is not None:
                family = v119_auxiliary.group(1)
                pending_v94.metrics.update(_parse_v94_tokens(line, family))
                continue
            if line.startswith("[v119-effect-error]") and pending_v94 is not None:
                pending_v94.metrics.update(_parse_v119_effect_error(line))
                continue
            object_auxiliary = re.match(
                r"^\[(v120|v121|v122)-(ground|intent|dynamics|policy)\]",
                line,
            )
            if object_auxiliary is not None and pending_v94 is not None:
                family = f"object_{object_auxiliary.group(2)}"
                pending_v94.metrics.update(_parse_v94_tokens(line, family))
                continue
            if line.startswith(
                (
                    "[v120-dynamics-error]",
                    "[v121-dynamics-error]",
                    "[v122-dynamics-error]",
                )
            ) and pending_v94 is not None:
                pending_v94.metrics.update(_parse_v120_dynamics_error(line))
                continue
            if (
                line.startswith(
                    tuple(
                        f"[v{version}-balance]"
                        for version in COMPACT_POLICY_VERSIONS
                        if version >= 101
                    )
                )
                and pending_v94 is not None
            ):
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
            if line.startswith(_compact_prefixes("epoch")):
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
            if line.startswith(_compact_prefixes("val")) and pending_epoch is not None:
                pending_epoch["val"].update(_parse_v94_tokens(line, "val"))
                continue
            if line.startswith(_compact_prefixes("probe")) and pending_epoch is not None:
                pending_epoch["val"].update(_parse_v94_tokens(line, "probe"))
                continue
            init_match = INIT_COUNT_RE.match(line)
            if init_match:
                run.init_counts[init_match.group("label")] = int(init_match.group("count"))
                continue
            header_match = HEADER_RE.match(line)
            if header_match and not line.startswith(
                ("[v39-init]",) + tuple(f"[v{version}-" for version in COMPACT_POLICY_VERSIONS)
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
    mainline_paths: dict[str, tuple[str, ...]] = {
        "capability": ("identity", "manifest", "capability"),
        "architecture_schema": ("identity", "manifest", "schema"),
        "layout": ("identity", "manifest", "layout"),
        "layout_schema": ("identity", "manifest", "layout_schema"),
        "seed": ("config", "data", "seed"),
        "batch_size": ("config", "optimizer", "batch_size"),
        "data_root": ("config", "data", "raw_hdf5_root"),
        "train_episode_count": ("config", "data", "train_episodes"),
        "val_episode_count": ("config", "data", "val_episodes"),
        "action_normalizer_fingerprint": (
            "normalizer_fingerprints",
            "action_v120",
        ),
        "action_normalizer_sha256": (
            "identity",
            "dataset",
            "action_normalizer_sha256",
        ),
        "rank": ("config", "bottom", "operator_rank"),
        "groups": ("config", "bottom", "operator_groups"),
        "depth_logit_init": ("config", "bottom", "operator_depth_logit_init"),
        "warmup": ("config", "optimizer", "warmup_steps"),
        "gripper_trajectory_weight": (
            "config",
            "objectives",
            "gripper_trajectory",
        ),
    }
    for key in normalized:
        path = mainline_paths.get(key)
        if path is None:
            continue
        value: Any = run.context
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
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


def _window_stats_through_batch(
    run: ParsedRun,
    key: str,
    *,
    epoch: int,
    batch: int,
    tail: int,
) -> dict[str, Any]:
    """Summarize one metric at a fixed aligned training boundary.

    Recovery audits must compare the same optimization age.  Reusing the
    complete-run tail for V120 while the candidate has only reached epoch 1
    silently compares batch ~2200 with epoch 8 and makes every early gate
    meaningless.
    """

    values = [
        point.metrics[key]
        for point in run.batch_points
        if (point.epoch, point.batch) <= (int(epoch), int(batch))
        and key in point.metrics
        and math.isfinite(point.metrics[key])
    ]
    if not values:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "tail_median": None,
            "change": None,
        }
    window = max(1, min(int(tail), max(len(values) // 4, 1)))
    first = _median(values[:window])
    last = _median(values[-window:])
    return {
        "count": len(values),
        "first": first,
        "last": last,
        "tail_median": _median(values[-int(tail) :]),
        "change": _relative_change(first, last),
    }


LOSS_COMPONENTS: tuple[tuple[str, str, str | None, str], ...] = (
    ("flow", "physical_flow", None, "action"),
    ("proposal", "proposal", "proposal_loss_weight", "action"),
    # Schema26 replaces the categorical head with continuous event-and-after
    # trajectory closure. Keep it distinct from historical event objectives.
    (
        "gripper_trajectory",
        "gripper_trajectory",
        "gripper_trajectory_weight",
        "action",
    ),
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
        out_of_memory = [error for error in run.fatal_errors if "outofmemoryerror" in error.lower()]
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
    mainline_proposal_primary = latest_val.get(
        "validation_proposal_primary_rmse_physical"
    )
    mainline_proposal_gain = latest_val.get(
        "validation_proposal_zero_mse_gain_vs_primary_physical"
    )
    mainline_proposal_delta = latest_val.get(
        "validation_proposal_zero_action_delta_rmse_physical"
    )
    if (
        mainline_proposal_primary is not None
        and mainline_proposal_gain is not None
        and mainline_proposal_delta is not None
        and mainline_proposal_gain
        > max(1e-8, 0.01 * float(mainline_proposal_primary) ** 2)
        and mainline_proposal_delta > 1e-5
    ):
        _add_finding(
            findings,
            "warning",
            "proposal",
            "proposal-zero-better",
            "Removing the clean proposal improves matched-noise action MSE.",
            gain=mainline_proposal_gain,
            action_delta_rmse=mainline_proposal_delta,
            primary_rmse=mainline_proposal_primary,
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

    # Preserve diagnosis of pre-fidelity mainline logs whose two ambiguous
    # execution names shared the proposal subset. New logs use the exact V120
    # modes above through canonical aliases and never enter this branch.
    mainline_primary = latest_val.get("validation_diagnostic_primary_rmse_physical")
    mainline_coverage = latest_val.get("validation_ablation_coverage")
    mainline_interventions = (
        (
            "proposal_zero",
            "proposal",
            "proposal-zero-better",
            "Removing the clean proposal improves matched-noise action MSE.",
        ),
        (
            "execution_no_updates",
            "execution",
            "bottom-no-updates-better",
            "Removing all Evidence-MMDiT residual updates improves matched-noise action MSE.",
        ),
        (
            "execution_full_updates",
            "execution",
            "bottom-full-updates-better",
            "Full bottom execution improves over the learned capacity/continuation controller.",
        ),
    )
    if mainline_primary is not None:
        primary_mse = float(mainline_primary) ** 2
        material_gain = max(1e-8, 0.01 * primary_mse)
        for stem, category, code, message in mainline_interventions:
            gain = latest_val.get(
                f"validation_{stem}_mse_gain_vs_primary_physical"
            )
            delta = latest_val.get(
                f"validation_{stem}_action_delta_rmse_physical"
            )
            if (
                gain is not None
                and delta is not None
                and gain > material_gain
                and delta > 1e-5
            ):
                _add_finding(
                    findings,
                    "warning",
                    category,
                    code,
                    message,
                    gain=gain,
                    action_delta_rmse=delta,
                    primary_rmse=mainline_primary,
                    coverage=mainline_coverage,
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
            value
            for point in open_points
            if (
                value := point.metrics.get(
                    "evidence_mmd_it_capacity_gate_mass",
                    point.metrics.get("evidence_mmd_it_capacity_ratio"),
                )
            )
            is not None
        ]
        depths = [
            value
            for point in open_points
            if (
                value := point.metrics.get(
                    "evidence_mmd_it_effective_basis_mass",
                    point.metrics.get("evidence_mmd_it_effective_depth"),
                )
            )
            is not None
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


def _performance_summary(run: ParsedRun) -> dict[str, Any]:
    """Summarize wall-time and CUDA evidence without conflating their sources.

    Mainline JSONL supplies per-window wall time and per-epoch CUDA peaks.  A
    captured console supplies the same data through explicit runtime rows.
    Historical V120 logs expose ``seconds_per_batch`` on their gradient rows.
    The priority below therefore preserves like-for-like window measurements
    when available and falls back to epoch wall time only when necessary.
    """

    def epoch_values(name: str) -> list[float]:
        values: list[float] = []
        for record in run.epoch_records:
            for section in (_train_section(record), _val_section(record)):
                value = section.get(name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
        return values

    def runtime_values(name: str) -> list[float]:
        values: list[float] = []
        for point in run.runtime_points:
            value = point.get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        return values

    def distribution(values: Sequence[float], *, source: str) -> dict[str, Any]:
        return {
            "count": len(values),
            "source": source,
            "median": _median(values),
            "p90": _percentile(values, 0.90),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }

    seconds_sources = (
        ("window", _series(run, "runtime_window_seconds_per_batch")),
        ("legacy-window", _series(run, "seconds_per_batch")),
        ("console-epoch", runtime_values("runtime_seconds_per_batch")),
        ("epoch", epoch_values("runtime_seconds_per_batch")),
    )
    seconds_source, seconds_values = next(
        ((source, values) for source, values in seconds_sources if values),
        ("missing", []),
    )
    sample_sources = (
        ("window", _series(run, "runtime_window_samples_per_second")),
        ("console-epoch", runtime_values("runtime_samples_per_second")),
        ("epoch", epoch_values("runtime_samples_per_second")),
    )
    sample_source, sample_values = next(
        ((source, values) for source, values in sample_sources if values),
        ("missing", []),
    )

    def peak(name: str) -> float | None:
        values = runtime_values(name) + epoch_values(name)
        return max(values) if values else None

    return {
        "seconds_per_batch": distribution(seconds_values, source=seconds_source),
        "samples_per_second": distribution(sample_values, source=sample_source),
        "cuda_peak_allocated_gib": peak("runtime_cuda_peak_allocated_gib"),
        "cuda_peak_reserved_gib": peak("runtime_cuda_peak_reserved_gib"),
        "cuda_device_used_gib": peak("runtime_cuda_device_used_gib"),
        "cuda_peak_process_estimate_gib": peak(
            "runtime_cuda_peak_process_estimate_gib"
        ),
    }


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
    aligned_batch_2200 = {
        key: stats
        for key in all_batch_keys
        if (
            stats := _window_stats_through_batch(
                run,
                key,
                epoch=1,
                batch=2200,
                tail=tail,
            )
        )["count"]
    }
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
                        "gripper_trajectory",
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
            "context_schema": run.context.get("schema")
            or _config_value(run, "architecture_schema"),
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
            "capability": _config_value(run, "capability"),
            "architecture_schema": _config_value(run, "architecture_schema"),
            "layout": _config_value(run, "layout"),
            "layout_schema": _config_value(run, "layout_schema"),
            "decoder": _config_value(run, "decoder", "final_action_decoder"),
            "training_stage": _config_value(run, "training_stage", "experiment_stage"),
            "seed": _config_value(run, "seed"),
            "batch_size": _config_value(run, "batch_size"),
            "data_root": _config_value(run, "data_root"),
            "train_episode_count": _config_value(run, "train_episode_count"),
            "val_episode_count": _config_value(run, "val_episode_count"),
            "condition_mode": _config_value(run, "condition_mode"),
            "stage1_checkpoint": _config_value(run, "stage1_checkpoint"),
            "stage1_initialization_enabled": _config_value(
                run,
                "stage1_initialization_enabled",
                "require_flow_jepa_stage1_checkpoint",
            ),
            "fresh_run": _config_value(run, "fresh", "fresh_run"),
            "action_normalizer_fingerprint": _config_value(run, "action_normalizer_fingerprint"),
            "action_normalizer_sha256": _config_value(run, "action_normalizer_sha256"),
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
        "module_parameters": (
            dict(run.context.get("module_parameters", {}))
            if isinstance(run.context.get("module_parameters"), Mapping)
            else {}
        ),
        "loss_budget": _loss_budget(run),
        "trajectories": trajectories,
        "structure": structure,
        "gradients": gradients,
        "performance": _performance_summary(run),
        "metric_index": metric_index,
        "aligned_batch_2200": aligned_batch_2200,
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
    performance = summary.get("performance", {})
    seconds = performance.get("seconds_per_batch", {})
    if seconds.get("median") is not None:
        lines.append(
            "performance: "
            f"seconds_per_batch median={_format_number(seconds.get('median'), 3)} "
            f"p90={_format_number(seconds.get('p90'), 3)} "
            f"source={seconds.get('source')} count={seconds.get('count')}"
        )
    process_peak = performance.get("cuda_peak_process_estimate_gib")
    if process_peak is not None:
        lines.append(
            "cuda memory: "
            f"process_peak_estimate={_format_number(process_peak, 3)} GiB "
            f"reserved_peak={_format_number(performance.get('cuda_peak_reserved_gib'), 3)} GiB"
        )
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
                "gripper_event_f1",
                "event_head_f1",
                "motion_head_f1",
                "proposal_utility_mse_gain",
                "validation_action_rmse_normalized",
                "validation_band_1_4_rmse_physical",
                "validation_band_5_12_rmse_physical",
                "validation_band_13_24_rmse_physical",
                "validation_proposal_zero_mse_gain_vs_primary_physical",
                "validation_execution_hard_mse_gain_vs_primary_physical",
                "validation_execution_neutral_mse_gain_vs_primary_physical",
                "validation_execution_full_capacity_mse_gain_vs_primary_physical",
                "validation_execution_three_basis_reduction_mse_gain_vs_primary_physical",
                "validation_sampling_diagnostic_coverage",
                "validation_proposal_ablation_coverage",
                "validation_execution_ablation_coverage",
            )
            metrics = " ".join(
                f"{key}={_format_number(val[key])}" for key in selected if key in val
            )
            lines.append(
                f"  epoch={record['epoch']} step={record['global_step']} {metrics}".rstrip()
            )
    if summary["structure"]:
        lines.append("active structure/controller tail medians:")
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
                "seconds_per_batch": summary.get("performance", {})
                .get("seconds_per_batch", {})
                .get("median"),
                "cuda_peak_process_estimate_gib": summary.get("performance", {}).get(
                    "cuda_peak_process_estimate_gib"
                ),
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


def _first_metric(
    values: Mapping[str, Any], names: Sequence[str]
) -> tuple[str | None, float | None]:
    for name in names:
        value = values.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return name, float(value)
    return None, None


def _epoch_metric_series(
    summary: Mapping[str, Any], names: Sequence[str]
) -> list[float]:
    result: list[float] = []
    for record in summary.get("epochs", []):
        val = record.get("val", {})
        if not isinstance(val, Mapping):
            continue
        _, value = _first_metric(val, names)
        if value is not None:
            result.append(value)
    return result


def _is_v120_normalizer_fingerprint(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 12 and all(character in "0123456789abcdef" for character in text)


def _recovery_assessment(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one complete candidate against the frozen V120 behavior.

    This is intentionally strict and descriptive.  It does not decide that a
    new architecture is bad because one stochastic scalar moved; it decides
    whether the user's stronger claim -- *at least V120* under the same public
    experiment identity -- has actually been demonstrated.
    """

    checks: list[dict[str, Any]] = []

    def record(
        name: str,
        status: str,
        *,
        baseline_value: Any = None,
        candidate_value: Any = None,
        detail: str = "",
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "detail": detail,
            }
        )

    baseline_manifest = baseline.get("manifest", {})
    candidate_manifest = candidate.get("manifest", {})
    identity_keys = (
        "seed",
        "batch_size",
        "data_root",
        "train_episode_count",
        "val_episode_count",
    )
    for key in identity_keys:
        old = baseline_manifest.get(key)
        new = candidate_manifest.get(key)
        if old is None or new is None:
            record(
                f"identity/{key}",
                "incomplete",
                baseline_value=old,
                candidate_value=new,
                detail="serialized public experiment identity is missing",
            )
        else:
            record(
                f"identity/{key}",
                "pass" if old == new else "fail",
                baseline_value=old,
                candidate_value=new,
            )
    old_fingerprint = baseline_manifest.get("action_normalizer_fingerprint")
    new_fingerprint = candidate_manifest.get("action_normalizer_fingerprint")
    if old_fingerprint is None or new_fingerprint is None:
        fingerprint_status = "incomplete"
    elif not (
        _is_v120_normalizer_fingerprint(old_fingerprint)
        and _is_v120_normalizer_fingerprint(new_fingerprint)
    ):
        fingerprint_status = "incomplete"
    else:
        fingerprint_status = (
            "pass"
            if str(old_fingerprint).lower() == str(new_fingerprint).lower()
            else "fail"
        )
    record(
        "identity/action_normalizer",
        fingerprint_status,
        baseline_value=old_fingerprint,
        candidate_value=new_fingerprint,
        detail=(
            "requires the exact 12-character V120-compatible MD5; "
            "the separately serialized SHA-256 is not comparable"
        ),
    )

    baseline_coverage = baseline.get("coverage", {})
    candidate_coverage = candidate.get("coverage", {})
    baseline_epochs = int(baseline_coverage.get("epoch_records") or 0)
    candidate_epochs = int(candidate_coverage.get("epoch_records") or 0)
    record(
        "coverage/all_epochs",
        "pass" if baseline_epochs >= 8 and candidate_epochs >= baseline_epochs else "fail",
        baseline_value=baseline_epochs,
        candidate_value=candidate_epochs,
        detail="recovery requires every V120 epoch, not one best checkpoint",
    )
    old_rows = int(baseline_coverage.get("batch_rows") or 0)
    new_rows = int(candidate_coverage.get("batch_rows") or 0)
    record(
        "coverage/train_rows",
        "pass" if old_rows > 0 and new_rows >= old_rows else "fail",
        baseline_value=old_rows,
        candidate_value=new_rows,
    )
    old_metric_count = int(
        baseline.get("observability", {}).get("batch_metric_count") or 0
    )
    new_metric_count = int(
        candidate.get("observability", {}).get("batch_metric_count") or 0
    )
    record(
        "coverage/active_metric_count",
        "pass" if old_metric_count > 0 and new_metric_count >= old_metric_count else "fail",
        baseline_value=old_metric_count,
        candidate_value=new_metric_count,
        detail="count is only a coverage floor; semantic checks below remain mandatory",
    )
    fatal_count = len(candidate_coverage.get("fatal_errors") or []) + int(
        candidate_coverage.get("traceback_count") or 0
    )
    record(
        "health/no_fatal_error",
        "pass" if fatal_count == 0 else "fail",
        baseline_value=0,
        candidate_value=fatal_count,
    )

    baseline_performance = baseline.get("performance", {})
    candidate_performance = candidate.get("performance", {})
    old_seconds = baseline_performance.get("seconds_per_batch", {})
    new_seconds = candidate_performance.get("seconds_per_batch", {})
    for statistic, ratio_limit in (
        ("median", RECOVERY_MEDIAN_SECONDS_RATIO_MAX),
        ("p90", RECOVERY_P90_SECONDS_RATIO_MAX),
    ):
        old_value = old_seconds.get(statistic)
        new_value = new_seconds.get(statistic)
        old_valid = (
            isinstance(old_value, (int, float))
            and math.isfinite(float(old_value))
            and float(old_value) > 0.0
        )
        new_valid = isinstance(new_value, (int, float)) and math.isfinite(
            float(new_value)
        )
        allowed = float(old_value) * ratio_limit if old_valid else None
        if not old_valid or not new_valid:
            status = "incomplete"
        else:
            assert allowed is not None
            status = "pass" if float(new_value) <= allowed else "fail"
        record(
            f"performance/seconds_per_batch_{statistic}",
            status,
            baseline_value={
                "seconds": old_value,
                "source": old_seconds.get("source"),
                "allowed_ratio": ratio_limit,
                "allowed_seconds": allowed,
            },
            candidate_value={
                "seconds": new_value,
                "source": new_seconds.get("source"),
            },
            detail=(
                "wall time includes model compute and data wait; recovery runs must use "
                "the same batch size, GPU class and logging cadence"
            ),
        )

    process_peak = candidate_performance.get("cuda_peak_process_estimate_gib")
    if not isinstance(process_peak, (int, float)) or not math.isfinite(
        float(process_peak)
    ):
        memory_status = "incomplete"
    else:
        memory_status = (
            "pass"
            if float(process_peak) <= RECOVERY_CUDA_PROCESS_GIB_MAX
            else "fail"
        )
    record(
        "performance/cuda_peak_process_gib",
        memory_status,
        baseline_value=RECOVERY_CUDA_PROCESS_GIB_MAX,
        candidate_value=process_peak,
        detail=(
            "batch-eight dedicated-GPU process estimate must stay within the 22 GiB "
            "production envelope"
        ),
    )

    # Schema24's source-level recovery gate is intentionally evaluated at the
    # same optimization age (epoch 1, batch <=2200), before a complete long
    # run exists.  Report the values without inventing pass/fail thresholds
    # when the frozen Schema23 audit series was not supplied.  The complete
    # eight-epoch gate below remains authoritative for release.
    early_metrics = (
        ("g3_parent_l1", ("object_grounding_g3_parent_l1",)),
        (
            "object_pair_cosine",
            ("object_grounding_object_content_pair_cosine",),
        ),
        (
            "p1_spatial_variation",
            (
                "object_p1_spatial_variation",
                "p1_query_chart_variation",
                "p1_spatial_var",
            ),
        ),
        ("p2_null_mass", ("object_p2_null_mass",)),
        (
            "p2_effect_rms",
            (
                "object_p2_effect_precontract_rms",
                "object_p2_effect_rms",
                "object_consequence_effect_rms",
            ),
        ),
    )
    baseline_early = baseline.get("aligned_batch_2200", {})
    candidate_early = candidate.get("aligned_batch_2200", {})
    parent_early = parent.get("aligned_batch_2200", {}) if parent is not None else {}

    def first_early_value(source: Mapping[str, Any], names: tuple[str, ...]) -> Any:
        for name in names:
            value = source.get(name, {}).get("tail_median")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return None

    for label, names in early_metrics:
        old = first_early_value(baseline_early, names)
        new = first_early_value(candidate_early, names)
        parent_value = first_early_value(parent_early, names)
        if old is None or new is None or (parent is not None and parent_value is None):
            status = "incomplete"
            closure = None
        elif parent is None:
            status = "pass"
            closure = None
        else:
            assert parent_value is not None
            parent_distance = abs(parent_value - old)
            candidate_distance = abs(new - old)
            tolerance = max(1e-8, abs(old) * 1e-6)
            if parent_distance <= tolerance:
                closure = 1.0 if candidate_distance <= tolerance else -candidate_distance
                status = "pass" if candidate_distance <= tolerance else "fail"
            else:
                closure = 1.0 - candidate_distance / parent_distance
                status = "pass" if closure >= 0.50 else "fail"
        record(
            f"early_batch_2200/{label}",
            status,
            baseline_value={"v120": old, "schema23": parent_value},
            candidate_value={"schema24": new, "gap_closure": closure},
            detail=(
                "same-age metric; when Schema23 is supplied, Schema24 must "
                "close at least 50% of its absolute distance to V120"
            ),
        )

    candidate_schema = candidate_manifest.get("architecture_schema")
    schema26 = (
        isinstance(candidate_schema, (int, float))
        and int(candidate_schema) >= 26
    )
    validation_metrics: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str], ...] = (
        ("action_rmse", ("full_rmse",), ("full_rmse",), "lower"),
        ("first_rmse", ("first_rmse",), ("first_rmse",), "lower"),
        ("first8_rmse", ("first8_rmse",), ("first8_rmse",), "lower"),
        ("tail_rmse", ("tail_rmse",), ("tail_rmse",), "lower"),
        ("arm_rmse", ("arm_full_rmse",), ("arm_full_rmse",), "lower"),
        ("gripper_rmse", ("gripper_full_rmse",), ("gripper_full_rmse",), "lower"),
        (
            "band_1_4_rmse",
            ("action_band_1_4_rmse",),
            ("validation_band_1_4_rmse_physical", "action_band_1_4_rmse"),
            "lower",
        ),
        (
            "band_5_12_rmse",
            ("action_band_5_12_rmse",),
            ("validation_band_5_12_rmse_physical", "action_band_5_12_rmse"),
            "lower",
        ),
        (
            "band_13_24_rmse",
            ("action_band_13_24_rmse",),
            ("validation_band_13_24_rmse_physical", "action_band_13_24_rmse"),
            "lower",
        ),
        (
            "decoded_gripper_f1",
            ("gripper_f1", "gripper_event_f1"),
            ("gripper_event_f1", "gripper_f1"),
            "higher",
        ),
        ("motion_head_f1", ("motion_head_f1",), ("motion_head_f1",), "higher"),
    )
    if not schema26:
        validation_metrics += (
            ("event_head_f1", ("event_head_f1",), ("event_head_f1",), "higher"),
        )
    for label, old_names, new_names, direction in validation_metrics:
        old_values = _epoch_metric_series(baseline, old_names)
        new_values = _epoch_metric_series(candidate, new_names)
        if (
            baseline_epochs < 1
            or candidate_epochs < 1
            or len(old_values) < baseline_epochs
            or len(new_values) < candidate_epochs
        ):
            record(
                f"validation/{label}",
                "incomplete",
                baseline_value=len(old_values),
                candidate_value=len(new_values),
                detail="metric must exist at every completed epoch",
            )
            continue
        old_final = old_values[-1]
        new_final = new_values[-1]
        old_mean = statistics.fmean(old_values)
        new_mean = statistics.fmean(new_values)
        if direction == "lower":
            passed = new_final <= old_final and new_mean <= old_mean
        else:
            passed = new_final >= old_final and new_mean >= old_mean
        record(
            f"validation/{label}",
            "pass" if passed else "fail",
            baseline_value={"final": old_final, "mean": old_mean},
            candidate_value={"final": new_final, "mean": new_mean},
            detail=f"{direction} is better; both final and eight-epoch mean must recover",
        )

    old_event_ratio = _epoch_metric_series(baseline, ("gripper_event_ratio",))
    new_event_ratio = _epoch_metric_series(candidate, ("gripper_event_ratio",))
    if (
        baseline_epochs < 1
        or candidate_epochs < 1
        or len(old_event_ratio) < baseline_epochs
        or len(new_event_ratio) < candidate_epochs
    ):
        record(
            "validation/gripper_event_rate_calibration",
            "incomplete",
            baseline_value=len(old_event_ratio),
            candidate_value=len(new_event_ratio),
        )
    else:
        old_error = statistics.fmean(abs(value - 1.0) for value in old_event_ratio)
        new_error = statistics.fmean(abs(value - 1.0) for value in new_event_ratio)
        record(
            "validation/gripper_event_rate_calibration",
            "pass" if new_error <= old_error else "fail",
            baseline_value=old_error,
            candidate_value=new_error,
            detail="mean absolute distance from the target event ratio 1.0",
        )

    if schema26:
        trajectory = candidate.get("trajectories", {}).get(
            "gripper_trajectory", {}
        )
        trajectory_count = trajectory.get("count")
        trajectory_tail = trajectory.get("tail_median")
        record(
            "objective/gripper_trajectory_observed",
            (
                "pass"
                if isinstance(trajectory_count, (int, float))
                and int(trajectory_count) > 0
                and isinstance(trajectory_tail, (int, float))
                and math.isfinite(float(trajectory_tail))
                else "incomplete"
            ),
            candidate_value={
                "count": trajectory_count,
                "tail_median": trajectory_tail,
            },
            detail=(
                "Schema26 replaces the categorical event head with the active "
                "continuous post-event gripper trajectory objective"
            ),
        )

    train_metrics = (
        "physical_flow",
        "physical_flow_native",
        "arm_fm_per_dim",
        "gripper_fm_field",
        "decoded_action",
    )
    for name in train_metrics:
        old = baseline.get("trajectories", {}).get(name, {}).get("tail_median")
        new = candidate.get("trajectories", {}).get(name, {}).get("tail_median")
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            status = "incomplete"
        else:
            status = "pass" if float(new) <= float(old) else "fail"
        record(
            f"train_tail/{name}",
            status,
            baseline_value=old,
            candidate_value=new,
            detail="training scale cannot substitute for validation recovery",
        )

    structure = candidate.get("structure", {})
    intent_interval_key = (
        "object_intent_public_interval_variation"
        if "object_intent_public_interval_variation" in structure
        else "object_intent_interval_variation"
    )
    required_structure = (
        "object_grounding_object_content_pair_cosine",
        intent_interval_key,
        "object_w_prediction_interval_variation",
        "object_w1_object_pair_cosine",
        "object_w2_object_pair_cosine",
        "object_teacher_reliability",
        "p1_query_chart_variation",
        "object_p2_successor_innovation_rms",
        "object_p3_precision_base_rms",
        "bottom_capacity_mean",
    )
    for name in required_structure:
        value = structure.get(name, {}).get("tail_median")
        status = (
            "pass"
            if isinstance(value, (int, float)) and math.isfinite(float(value))
            else "incomplete"
        )
        if name.endswith("object_content_pair_cosine") or name.endswith(
            "object_pair_cosine"
        ):
            if status == "pass" and float(value) >= 0.999:
                status = "fail"
        record(
            f"structure/{name}",
            status,
            candidate_value=value,
            detail="cosine checks reject exact object-slot publicization",
        )

    gradients = candidate.get("gradients", {})
    schema24 = (
        isinstance(candidate_schema, (int, float))
        and int(candidate_schema) >= 24
    ) or any(name.startswith("gradient_raw_") for name in gradients)
    owner_rows = (
        "grounder",
        "intent",
        "dynamics",
        "p1_factual",
        "p2_effect_reader",
        "p3_compiler",
        "v120_canvas_seed",
        "v120_layer_contracts",
        "bottom_evidence_adapter",
        "bottom_policy_bridge",
        "bottom_capacity",
        "bottom_execution",
    )
    required_gradients = (
        tuple(
            f"gradient_{stage}_{owner}_l2"
            for stage in ("raw", "postlocal", "postglobal")
            for owner in owner_rows
        )
        if schema24
        else tuple(f"gradient_postclip_{owner}_l2" for owner in owner_rows)
    )
    for name in required_gradients:
        value = gradients.get(name, {}).get("tail_median")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            status = "incomplete"
        else:
            status = "pass" if float(value) > 1e-12 else "fail"
        record(f"gradient/{name}", status, candidate_value=value)

    if schema24:
        for name in (
            "gradient_postlocal_bottom_decoder_l2",
            "gradient_postglobal_global_l2",
        ):
            value = gradients.get(name, {}).get("tail_median")
            status = (
                "pass"
                if isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) <= 1.0001
                else ("incomplete" if value is None else "fail")
            )
            record(
                f"gradient_bound/{name}",
                status,
                baseline_value=1.0,
                candidate_value=value,
            )

    latest_val = (
        candidate.get("epochs", [])[-1].get("val", {})
        if candidate.get("epochs")
        else {}
    )
    exact_v120_ablations = "validation_execution_ablation_coverage" in latest_val
    if exact_v120_ablations:
        coverage_stems = [
            "sampling_diagnostic",
            "proposal_ablation",
            "execution_ablation",
        ]
        if schema26:
            coverage_stems.append("p2_intervention")
        for stem in coverage_stems:
            coverage = latest_val.get(f"validation_{stem}_coverage")
            record(
                f"causal_ablation/{stem}_coverage",
                "pass"
                if isinstance(coverage, (int, float)) and float(coverage) > 0.0
                else "incomplete",
                candidate_value=coverage,
            )
        proposal_primary = latest_val.get("validation_proposal_primary_rmse_physical")
        proposal_allowed = (
            0.01 * float(proposal_primary) ** 2
            if isinstance(proposal_primary, (int, float))
            and math.isfinite(float(proposal_primary))
            else None
        )
        proposal_gain = latest_val.get(
            "validation_proposal_zero_mse_gain_vs_primary_physical"
        )
        record(
            "causal_ablation/proposal_zero",
            (
                "incomplete"
                if proposal_allowed is None or not isinstance(proposal_gain, (int, float))
                else ("pass" if float(proposal_gain) <= proposal_allowed else "fail")
            ),
            baseline_value=proposal_allowed,
            candidate_value=proposal_gain,
            detail="positive gain means removing the path improves action",
        )
        execution_primary = latest_val.get(
            "validation_execution_primary_rmse_physical"
        )
        execution_allowed = (
            0.01 * float(execution_primary) ** 2
            if isinstance(execution_primary, (int, float))
            and math.isfinite(float(execution_primary))
            else None
        )
        for mode in (
            "hard",
            "neutral",
            "full_capacity",
            "three_basis_reduction",
        ):
            gain = latest_val.get(
                f"validation_execution_{mode}_mse_gain_vs_primary_physical"
            )
            record(
                f"causal_ablation/execution_{mode}",
                (
                    "incomplete"
                    if execution_allowed is None or not isinstance(gain, (int, float))
                    else ("pass" if float(gain) <= execution_allowed else "fail")
                ),
                baseline_value=execution_allowed,
                candidate_value=gain,
                detail="positive gain means forcing the ablation improves action",
            )
    else:
        coverage = latest_val.get("validation_ablation_coverage")
        record(
            "causal_ablation/coverage",
            "pass"
            if isinstance(coverage, (int, float)) and float(coverage) > 0.0
            else "incomplete",
            candidate_value=coverage,
        )
        primary = latest_val.get("validation_diagnostic_primary_rmse_physical")
        allowed_gain = (
            0.01 * float(primary) ** 2
            if isinstance(primary, (int, float)) and math.isfinite(float(primary))
            else None
        )
        for stem in (
            "proposal_zero",
            "execution_no_updates",
            "execution_full_updates",
        ):
            gain = latest_val.get(f"validation_{stem}_mse_gain_vs_primary_physical")
            if allowed_gain is None or not isinstance(gain, (int, float)):
                status = "incomplete"
            else:
                status = "pass" if float(gain) <= allowed_gain else "fail"
            record(
                f"causal_ablation/{stem}",
                status,
                baseline_value=allowed_gain,
                candidate_value=gain,
                detail="positive gain means removing/forcing the path improves action",
            )

    failed = sum(item["status"] == "fail" for item in checks)
    incomplete = sum(item["status"] == "incomplete" for item in checks)
    status = "fail" if failed else ("incomplete" if incomplete else "pass")
    return {
        "baseline": baseline.get("label"),
        "parent": parent.get("label") if parent is not None else None,
        "candidate": candidate.get("label"),
        "status": status,
        "failed": failed,
        "incomplete": incomplete,
        "checks": checks,
    }


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
        merged.runtime_points.extend(run.runtime_points)
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
    unique_runtime: dict[tuple[float, float, tuple[str, ...]], dict[str, float]] = {}
    for point in merged.runtime_points:
        metric_names = tuple(sorted(name for name in point if name not in {"epoch", "step"}))
        key = (float(point.get("epoch", 0.0)), float(point.get("step", 0.0)), metric_names)
        unique_runtime[key] = point
    merged.runtime_points = list(unique_runtime.values())
    return merged


def parse_run_input(path: Path) -> ParsedRun:
    if path.is_file():
        return parse_log(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = (
        "nohup.log",
        "metrics.jsonl",
        "v39_policy_epochs.jsonl",
        "train.log",
    )
    matched = [path / name for name in candidates if (path / name).is_file()]
    if not matched:
        matched = sorted(path.glob("*.log")) + sorted(path.glob("*.txt"))
    if not matched:
        raise FileNotFoundError(f"no supported logs under {path}")
    merged = _merge_runs(
        [parse_log(item, label=path.name) for item in matched],
        path=path,
        label=path.name,
    )
    context_path = path / "run_context.json"
    if context_path.is_file():
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid mainline run context: {context_path}") from error
        if not isinstance(context, Mapping):
            raise ValueError(f"mainline run context must be an object: {context_path}")
        merged.context = dict(context)
    return merged


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ClearVLA policy logs across legacy and compact formats."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="log files or run directories")
    parser.add_argument("--tail", type=int, default=20, help="tail window in logged batches")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="write report instead of stdout")
    parser.add_argument(
        "--recovery-baseline",
        type=Path,
        help="frozen complete V120 log/run used for strict recovery assessment",
    )
    parser.add_argument(
        "--recovery-parent",
        type=Path,
        help=(
            "rejected Schema23 log/run used only for the aligned batch-2200 "
            "absolute-gap closure gate"
        ),
    )
    parser.add_argument(
        "--require-recovery",
        action="store_true",
        help="exit nonzero unless every candidate proves complete V120 recovery",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tail < 1:
        raise SystemExit("--tail must be positive")
    if args.require_recovery and args.recovery_baseline is None:
        raise SystemExit("--require-recovery needs --recovery-baseline")
    if args.recovery_parent is not None and args.recovery_baseline is None:
        raise SystemExit("--recovery-parent needs --recovery-baseline")
    try:
        runs = [parse_run_input(path) for path in args.logs]
        baseline_run = (
            parse_run_input(args.recovery_baseline)
            if args.recovery_baseline is not None
            else None
        )
        parent_run = (
            parse_run_input(args.recovery_parent)
            if args.recovery_parent is not None
            else None
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"log/run input is invalid: {exc}", file=sys.stderr)
        return 2
    if not runs:
        print("no supported log files found", file=sys.stderr)
        return 2
    summaries = [build_summary(run, tail=args.tail) for run in runs]
    baseline_summary = (
        build_summary(baseline_run, tail=args.tail) if baseline_run is not None else None
    )
    parent_summary = (
        build_summary(parent_run, tail=args.tail) if parent_run is not None else None
    )
    recovery = (
        [
            _recovery_assessment(baseline_summary, summary, parent_summary)
            for summary in summaries
        ]
        if baseline_summary is not None
        else []
    )
    payload = {
        "runs": summaries,
        "comparison": _comparison(summaries),
        "recovery_baseline": baseline_summary,
        "recovery_parent": parent_summary,
        "recovery": recovery,
    }
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
                    f"sec/batch={_format_number(row['seconds_per_batch'], 3)} "
                    f"cuda_peak={_format_number(row['cuda_peak_process_estimate_gib'], 3)}GiB "
                    f"critical={row['critical']} warnings={row['warnings']}\n"
                )
            rendered = rendered.rstrip()
        if recovery:
            rendered += "\n\n=== V120 recovery assessment ===\n"
            for assessment in recovery:
                rendered += (
                    f"{assessment['candidate']} vs {assessment['baseline']}: "
                    f"status={assessment['status']} failed={assessment['failed']} "
                    f"incomplete={assessment['incomplete']}\n"
                )
                for check in assessment["checks"]:
                    rendered += (
                        f"  [{check['status']}] {check['name']}: "
                        f"baseline={check['baseline']} candidate={check['candidate']}"
                    )
                    if check["detail"]:
                        rendered += f" ({check['detail']})"
                    rendered += "\n"
            rendered = rendered.rstrip()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.require_recovery and any(item["status"] != "pass" for item in recovery):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
