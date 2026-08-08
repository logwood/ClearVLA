from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    V39PolicyTrainerConfig,
    _attach_intent_frame_progress_audit,
    _attach_v94_loss_ledger,
    _evidence_epoch_log_line,
    _evidence_serial_log_line,
    _layer_contract_aux_scale,
    _needs_action_counterfactuals,
    _needs_future_targets,
    _nonfinite_gradient_report,
    _prepare_run_directory,
    _saved_flow_jepa_hierarchy,
    motion_head_metrics,
)
from clearvla.policy.system import _keep_sampling_diagnostic
from clearvla.tools.summarize_v101_action_path_probe import _model_path_version


def test_nonfinite_gradient_report_names_the_corrupt_parameter() -> None:
    module = torch.nn.Linear(3, 2)
    module.weight.grad = torch.zeros_like(module.weight)
    module.bias.grad = torch.zeros_like(module.bias)
    module.weight.grad[0, 1] = torch.nan
    report = _nonfinite_gradient_report(module)
    assert "weight[shape=(2, 3),nan=1" in report
    assert "bias" not in report


def test_v94_layer_contract_aux_schedule_is_independent_and_constant():
    trainer = V39PolicyTrainerConfig(
        midcut_aux_loss_weight=0.03,
        midcut_aux_final_ratio=0.15,
        midcut_aux_decay_epochs=4,
        layer_contract_aux_loss_weight=0.03,
        layer_contract_aux_final_ratio=1.0,
        layer_contract_aux_decay_epochs=0,
    )
    assert _layer_contract_aux_scale(trainer, 1) == 0.03
    assert _layer_contract_aux_scale(trainer, 4) == 0.03
    assert _layer_contract_aux_scale(trainer, 40) == 0.03


def test_intent_frame_progress_is_detached_audit_only() -> None:
    backward_loss = torch.tensor(2.0, requires_grad=True)
    losses = {"loss": backward_loss}
    sample = {"frame_progress": torch.tensor([0.25, 0.75])}
    output = {
        "flow_jepa_intent_progress_coordinate_per_sample": torch.tensor(
            [[0.50], [0.50]], requires_grad=True
        )
    }

    _attach_intent_frame_progress_audit(losses, sample, output)

    assert losses["loss"] is backward_loss
    assert torch.isclose(losses["flow_jepa_frame_progress"], torch.tensor(0.50))
    assert torch.isclose(
        losses["flow_jepa_intent_frame_progress_gap"], torch.tensor(0.0)
    )
    assert torch.isclose(
        losses["flow_jepa_intent_frame_progress_mae"], torch.tensor(0.25)
    )
    assert not losses["flow_jepa_intent_frame_progress_mae"].requires_grad


def test_v95_future_teacher_does_not_imply_counterfactual_policy_graphs():
    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        contract_mode="layer_adapter",
        future_latent_loss_start_epoch=1,
        flow_jepa_future_loss_weight=0.10,
        flow_jepa_stage_loss_weight=0.02,
        rollout_contrast_loss_weight=0.0,
        layer_contract_aux_loss_weight=0.0,
        layer_contrast_loss_weight=0.03,
    )
    assert _needs_future_targets(trainer, 1)
    assert not _needs_action_counterfactuals(trainer, 1)

    aux_trainer = replace(trainer, layer_contract_aux_loss_weight=0.03)
    assert _needs_action_counterfactuals(aux_trainer, 1)

    contrast_trainer = replace(trainer, rollout_contrast_loss_weight=0.01)
    assert _needs_action_counterfactuals(contrast_trainer, 1)


def test_v99_history_only_identity_advantage_does_not_request_future_frames():
    trainer = V39PolicyTrainerConfig(
        training_stage="policy",
        future_latent_loss_start_epoch=1,
        flow_jepa_identity_advantage_loss_weight=0.02,
        rollout_dynamics_loss_weight=0.0,
        rollout_contrast_loss_weight=0.0,
        rollout_variance_loss_weight=0.0,
        rollout_norm_loss_weight=0.0,
        rollout_milestone_delta_match_weight=0.0,
        midcut_aux_loss_weight=0.0,
        layer_contract_aux_loss_weight=0.0,
    )
    assert not _needs_future_targets(trainer, 1)


def test_v94_loss_ledger_reconstructs_the_backward_scalar():
    trainer = V39PolicyTrainerConfig(
        latent_cvae_mmdit_execution_value_loss_weight=0.5,
    )
    losses = {
        "loss": torch.tensor(6.87),
        "physical_flow": torch.tensor(2.0),
        "proposal": torch.tensor(1.0),
        "event": torch.tensor(2.0),
        "motion": torch.tensor(3.0),
        "rollout_dynamics": torch.tensor(4.0),
        "rollout_milestone_delta_match": torch.tensor(5.0),
        "evidence_mmd_it_execution_value_loss": torch.tensor(6.0),
    }
    _attach_v94_loss_ledger(
        losses,
        trainer,
        enable_future_loss=True,
        layer_aux_contribution=torch.tensor(0.7),
    )
    assert torch.isclose(losses["loss_group_action"], torch.tensor(2.30))
    assert torch.isclose(losses["loss_group_rollout"], torch.tensor(0.87))
    assert torch.isclose(losses["loss_group_execution"], torch.tensor(3.0))
    assert torch.isclose(losses["loss_group_layer"], torch.tensor(0.7))
    assert torch.isclose(losses["loss_ledger_sum"], losses["loss"])
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6


def test_v100_ledger_accounts_for_change_and_static_identity_objectives() -> None:
    trainer = V39PolicyTrainerConfig(
        flow_jepa_future_loss_weight=0.10,
        flow_jepa_future_change_loss_weight=0.02,
        flow_jepa_identity_advantage_loss_weight=0.02,
        flow_jepa_static_identity_loss_weight=0.01,
        proposal_loss_weight=0.0,
        event_loss_weight=0.0,
        arm_motion_loss_weight=0.0,
        rollout_dynamics_loss_weight=0.0,
        rollout_milestone_delta_match_weight=0.0,
    )
    losses = {
        "loss": torch.tensor(2.058),
        "physical_flow": torch.tensor(2.0),
        "flow_jepa_future_prediction": torch.tensor(0.4),
        "flow_jepa_future_change": torch.tensor(0.5),
        "flow_jepa_identity_advantage_loss": torch.tensor(0.3),
        "flow_jepa_static_identity_loss": torch.tensor(0.2),
    }
    _attach_v94_loss_ledger(losses, trainer, enable_future_loss=True)
    assert torch.isclose(losses["loss_group_representation"], torch.tensor(0.058))
    assert torch.isclose(losses["loss_ledger_sum"], losses["loss"])
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6


def test_v105_address_objective_requests_teacher_and_closes_exact_ledger() -> None:
    trainer = V39PolicyTrainerConfig(
        flow_jepa_horizon_address_loss_weight=0.02,
        proposal_loss_weight=0.0,
        event_loss_weight=0.0,
        arm_motion_loss_weight=0.0,
        rollout_dynamics_loss_weight=0.0,
        rollout_milestone_delta_match_weight=0.0,
    )
    assert _needs_future_targets(trainer, 1)
    losses = {
        "loss": torch.tensor(2.01),
        "physical_flow": torch.tensor(2.0),
        "flow_jepa_horizon_address": torch.tensor(0.5),
    }
    _attach_v94_loss_ledger(losses, trainer, enable_future_loss=True)
    assert torch.isclose(
        losses["loss_contrib_flow_jepa_horizon_address"],
        torch.tensor(0.01),
    )
    assert torch.isclose(
        losses["loss_group_representation"],
        torch.tensor(0.01),
    )
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6


def test_v106_interval_objective_requests_teacher_and_closes_exact_ledger() -> None:
    trainer = V39PolicyTrainerConfig(
        flow_jepa_interval_stage_loss_weight=0.02,
        proposal_loss_weight=0.0,
        event_loss_weight=0.0,
        arm_motion_loss_weight=0.0,
        rollout_dynamics_loss_weight=0.0,
        rollout_milestone_delta_match_weight=0.0,
    )
    assert _needs_future_targets(trainer, 1)
    losses = {
        "loss": torch.tensor(2.01),
        "physical_flow": torch.tensor(2.0),
        "flow_jepa_interval_stage": torch.tensor(0.5),
    }
    _attach_v94_loss_ledger(losses, trainer, enable_future_loss=True)
    assert torch.isclose(
        losses["loss_contrib_flow_jepa_interval_stage"],
        torch.tensor(0.01),
    )
    assert torch.isclose(
        losses["loss_group_representation"],
        torch.tensor(0.01),
    )
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6


def test_v95_representation_ledger_and_console_names_are_explicit():
    trainer = V39PolicyTrainerConfig(
        flow_jepa_future_loss_weight=0.10,
        flow_jepa_stage_loss_weight=0.02,
        flow_jepa_warp_loss_weight=0.03,
        flow_jepa_cycle_loss_weight=0.01,
    )
    losses = {
        "loss": torch.tensor(2.063),
        "physical_flow": torch.tensor(2.0),
        "flow_jepa_future_prediction": torch.tensor(0.4),
        "flow_jepa_stage_prediction": torch.tensor(0.3),
        "flow_jepa_warp_loss": torch.tensor(0.5),
        "flow_jepa_cycle_loss": torch.tensor(0.2),
    }
    _attach_v94_loss_ledger(losses, trainer, enable_future_loss=True)
    assert torch.isclose(losses["loss_group_action"], torch.tensor(2.0))
    assert torch.isclose(losses["loss_group_representation"], torch.tensor(0.063))
    assert torch.isclose(losses["loss_ledger_sum"], losses["loss"])
    assert abs(float(losses["loss_ledger_residual"])) < 1e-6

    line = _evidence_serial_log_line(
        {
            **{key: float(value) for key, value in losses.items()},
            "flow_jepa_patch_flow_magnitude": 0.8,
            "flow_jepa_confidence_mean": 0.7,
            "flow_jepa_occlusion_fraction": 0.2,
            "grad_flow_dino_evidence": 0.01,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v95-train]" in line
    assert "[v95-repr]" in line
    assert "representation:0.06300" in line
    assert "window_pred=0.40000" in line
    assert "stage_pred=0.30000" in line
    assert "confidence=0.700" in line
    assert "flow_dino=1.00e-02" in line

    epoch_line = _evidence_epoch_log_line(
        epoch=1,
        global_step=100,
        train={
            "loss": 2.057,
            "physical_flow": 2.0,
            "flow_jepa_future_prediction": 0.4,
            "loss_group_representation": 0.057,
        },
        val={"full_rmse": 0.1},
    )
    assert "[v95-epoch]" in epoch_line
    assert "[v95-val]" in epoch_line
    assert "representation:0.05700" in epoch_line


def test_v96_console_exposes_late_precision_without_stage_semantics():
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "flow_jepa_late_bottleneck": 1.0,
            "flow_jepa_future_prediction": 0.3,
            "flow_jepa_future_change_direction": 0.2,
            "flow_jepa_native_grid_size": 24.0,
            "flow_jepa_coarse_grid_size": 8.0,
            "flow_jepa_detail_gate_mean": 0.375,
            "flow_jepa_detail_effective_comparisons": 600.0,
            "flow_jepa_detail_candidate_comparisons": 1600.0,
            "flow_jepa_address_flow_mass": 0.62,
            "flow_jepa_address_fallback_mass": 0.38,
            "grad_flow_dino_sparse_fine": 0.02,
            "grad_flow_dino_address_reader": 0.03,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v96-train]" in line
    assert "[v96-repr]" in line
    assert "future_pred=0.30000" in line and "change_dir=0.20000" in line
    assert "native_grid=24" in line and "coarse_grid=8" in line
    assert "detail_gate_mean=0.375" in line
    assert "detail_weighted_cmp=600" in line and "detail_candidate_cmp=1600" in line
    assert "address_flow_mass=0.620" in line
    assert "address_fallback_mass=0.380" in line
    assert "fine_flow=2.00e-02" in line and "address_reader=3.00e-02" in line
    assert "stage_pred=" not in line and "stage_h=" not in line


def test_v98_console_exposes_dino_seeded_raw_reader_and_332_gradient_contract():
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "flow_jepa_late_bottleneck": 1.0,
            "flow_jepa_raw_image_enabled": 1.0,
            "flow_jepa_future_prediction": 0.3,
            "flow_jepa_raw_high_grid_size": 84.0,
            "flow_jepa_raw_mid_grid_size": 42.0,
            "flow_jepa_raw_coarse_grid_size": 8.0,
            "flow_jepa_raw_flow_magnitude": 3.2,
            "flow_jepa_raw_flow_grid_magnitude": 0.27,
            "flow_jepa_raw_seed_reliability": 0.18,
            "flow_jepa_raw_boundary_penalty": 0.012,
            "flow_jepa_raw_valid_fraction": 0.99,
            "flow_jepa_raw_detail_precision_mean": 0.91,
            "flow_jepa_future_horizon_4": 0.22,
            "flow_jepa_future_horizon_48": 0.44,
            "flow_jepa_raw_address_flow_mass": 0.61,
            "flow_jepa_raw_address_fallback_mass": 0.39,
            "flow_jepa_grounding_block_count": 3.0,
            "flow_jepa_world_block_count": 3.0,
            "flow_jepa_policy_block_count": 2.0,
            "evidence_top_policy_workspace_scale": 0.10,
            "evidence_top_policy_workspace_update_norm": 0.31,
            "grad_evidence_top_policy_workspace_lift": 0.07,
            "grad_flow_dino_raw_high_flow": 0.02,
            "grad_flow_dino_semantic_coarse_flow": 0.01,
            "grad_flow_dino_raw_address_reader": 0.03,
            "grad_dit_grounding_blocks": 0.04,
            "grad_dit_world_blocks": 0.05,
            "grad_dit_policy_blocks": 0.06,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v98-train]" in line and "[v98-repr]" in line
    assert "raw_high_grid=84" in line and "raw_coarse_grid=8" in line
    assert "raw_flow_grid=0.270" in line and "seed_reliability=0.180" in line
    assert "raw_boundary=0.0120" in line and "raw_valid=0.990" in line
    assert "raw_precision=0.910" in line
    assert "future_h4=0.22000" in line and "future_h48=0.44000" in line
    assert "grounding_blocks=3" in line
    assert "world_blocks=3" in line and "policy_blocks=2" in line
    assert "raw_high_flow=2.00e-02" in line
    assert "semantic_coarse_flow=1.00e-02" in line
    assert "raw_address_reader=3.00e-02" in line
    assert "top_policy_scale=0.100" in line and "top_policy_update=0.310" in line
    assert "top_policy_lift=7.00e-02" in line


def test_v99_console_exposes_identity_baseline_and_nonduplicate_address_contract():
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "flow_jepa_late_bottleneck": 1.0,
            "flow_jepa_raw_image_enabled": 1.0,
            "flow_jepa_zero_flow_guard": 1.0,
            "flow_jepa_future_prediction": 0.3,
            "flow_jepa_identity_advantage_loss": 0.021,
            "flow_jepa_raw_identity_warp_error": 0.12,
            "flow_jepa_raw_warp_gain_over_zero": 0.03,
            "flow_jepa_raw_moving_warp_gain": 0.08,
            "flow_jepa_raw_static_warp_gain": 0.002,
            "flow_jepa_raw_moving_correlation_entropy": 0.62,
            "flow_jepa_raw_moving_correlation_margin": 0.14,
            "flow_jepa_raw_observable_motion_fraction": 0.18,
            "flow_jepa_raw_address_center_separation": 0.42,
            "flow_jepa_raw_address_lane_value_difference": 0.31,
            "flow_jepa_raw_address_logit_advantage": 0.22,
            "flow_jepa_raw_address_zero_flow_value_delta": 0.14,
            "flow_jepa_raw_address_shuffled_flow_value_delta": 0.19,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v99-train]" in line and "[v99-repr]" in line
    assert "identity_adv=0.02100" in line
    assert "zero_warp=0.1200" in line and "warp_gain=+0.0300" in line
    assert "moving_gain=+0.0800" in line and "static_gain=+0.0020" in line
    assert "moving_corr_entropy=0.620" in line
    assert "moving_corr_margin=0.140" in line
    assert "motion_visible=0.180" in line
    assert "address_separation=0.420" in line
    assert "address_value_delta=0.310" in line
    assert "address_logit_gain=+0.220" in line
    assert "address_zero_delta=0.140" in line
    assert "address_shuffle_delta=0.190" in line


def test_v100_console_uses_additive_detail_semantics() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "flow_jepa_strict_role_visual_path": 1.0,
            "flow_jepa_raw_additive_detail_path": 1.0,
            "flow_jepa_future_prediction": 0.3,
            "flow_jepa_future_change_direction": 0.4,
            "flow_jepa_future_change": 0.5,
            "flow_jepa_static_identity_loss": 0.02,
            "flow_jepa_raw_address_flow_mass": 0.37,
            "flow_jepa_raw_address_fallback_mass": 0.63,
            "flow_jepa_raw_address_entropy": 0.81,
            "flow_jepa_raw_address_logit_advantage": 0.22,
            "flow_jepa_raw_detail_fused_with_latest_dino": 1.0,
            "flow_jepa_refined_evidence_token_count": 320.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v100-train]" in line and "[v100-repr]" in line
    assert "change_obj=0.50000" in line
    assert "static_identity=0.02000" in line
    assert "raw_detail_share=0.370" in line
    assert "raw_base_share=0.630" in line
    assert "detail_address_entropy=0.810" in line
    assert "detail_address_concentration=+0.220" in line
    assert "raw_dino_fused=1" in line
    assert "refined_visual_tokens=320" in line
    assert "raw_address_flow=" not in line
    assert "raw_address_fallback=" not in line


def test_v101_console_exposes_temporal_and_coordinate_contract() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "temporal_balance_active": 1.0,
            "flow_jepa_strict_role_visual_path": 1.0,
            "flow_jepa_raw_additive_detail_path": 1.0,
            "flow_jepa_raw_detail_fused_with_latest_dino": 0.0,
            "flow_jepa_raw_detail_fused_with_source_dino": 1.0,
            "physical_flow_no_information_balance": 0.79,
            "trajectory_information_score": 0.12,
            "trajectory_information_weight_min": 1.0,
            "trajectory_information_weight_max": 1.0,
            "trajectory_information_effective_fraction": 1.0,
            "action_horizon_weight_first": 0.955,
            "action_horizon_weight_tail": 1.091,
            "action_band_1_4_physical_flow": 0.70,
            "action_band_5_12_physical_flow": 0.79,
            "action_band_13_24_physical_flow": 0.88,
            "condition_action_history_keep": 0.875,
            "condition_goal_keep": 1.0,
            "condition_proposal_keep": 0.75,
            "flow_jepa_teacher_mask_past_fraction": 0.25,
            "flow_jepa_teacher_mask_change_fraction": 0.50,
            "flow_jepa_teacher_mask_uniform_fraction": 0.25,
            "flow_jepa_teacher_mask_selected_change_ratio": 1.24,
            "evidence_top_policy_workspace_fixed_fusion": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v101-train]" in line and "[v101-balance]" in line
    assert "raw_source_dino_fused=1" in line
    assert "raw_dino_fused=" not in line
    assert "action_h1_4=0.700000" in line
    assert "action_h13_24=0.880000" in line
    assert "teacher_change_quota=0.500" in line
    assert "history_keep=0.875" in line
    assert "top_policy_fixed_fusion=1" in line


def test_v104_console_names_and_exposes_the_three_structural_contracts() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_bounded_flow_coordinates": 1.0,
        "flow_jepa_sequential_horizon_memory": 1.0,
        "role_residual_contract_enabled": 1.0,
        "flow_jepa_raw_mid_boundary_compression": 0.12,
        "flow_jepa_raw_high_boundary_compression": 0.21,
        "flow_jepa_motion_evidence_flow_magnitude": 0.32,
        "flow_jepa_future_query_adjacent_cosine": 0.72,
        "flow_jepa_perceptual_history_entropy": 0.64,
        "flow_jepa_perceptual_history_latest_mass": 0.58,
        "flow_jepa_horizon_transition_update_rms": 0.18,
        "flow_jepa_horizon_transition_state_delta": 0.22,
        "role_residual_raw_rms": 3.2,
        "role_residual_bounded_rms": 0.42,
        "role_residual_compression": 0.71,
        "attnres_world_to_policy_raw_value_rms": 7.0,
        "attnres_world_to_policy_value_rms": 0.95,
        "attnres_world_to_policy_value_compression": 0.81,
        "evidence_policy_delta_attnres_raw_value_rms": 4.0,
        "evidence_policy_delta_attnres_value_rms": 0.90,
        "evidence_policy_delta_attnres_value_compression": 0.76,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v104-train]" in line and "[v104-repr]" in line
    assert "mid_bound_compress=0.120" in line
    assert "high_bound_compress=0.210" in line
    assert "query_adj_cos=0.720" in line
    assert "history_entropy=0.640" in line
    assert "horizon_state_delta=0.220" in line
    assert "role_raw_rms=3.200" in line
    assert "role_write_rms=0.420" in line
    assert "w2p_value_compress=0.810" in line
    assert "bottom_value_compress=0.760" in line


def test_partial_v104_ablation_is_not_named_as_the_formal_v104_contract() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_predictive_change_contract": 1.0,
        "flow_jepa_soft_address_lattice": 1.0,
        "evidence_policy_delta_bridge_enabled": 1.0,
        "flow_jepa_bounded_flow_coordinates": 1.0,
        "flow_jepa_sequential_horizon_memory": 0.0,
        "role_residual_contract_enabled": 0.0,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v103-train]" in line
    assert "[v104-train]" not in line


def test_v105_console_exposes_reliable_future_and_horizon_address_health() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_bounded_flow_coordinates": 1.0,
        "flow_jepa_sequential_horizon_memory": 1.0,
        "role_residual_contract_enabled": 1.0,
        "flow_jepa_horizon_soft_address": 1.0,
        "flow_jepa_future_reliable_normalization": 1.0,
        "flow_jepa_horizon_address_supervision_active": 1.0,
        "flow_jepa_future_raw_delta_loss": 0.02,
        "flow_jepa_future_reliable_normalized_loss": 0.31,
        "flow_jepa_future_change_reliability": 0.44,
        "flow_jepa_future_current_reference_scale": 0.80,
        "flow_jepa_future_normalization_scale": 0.06,
        "flow_jepa_future_horizon_4_target_scale": 0.02,
        "flow_jepa_future_horizon_4_normalization_scale": 0.06,
        "flow_jepa_future_horizon_4_reliability": 0.25,
        "flow_jepa_horizon_address": 0.12,
        "flow_jepa_horizon_address_teacher_reliability": 0.27,
        "flow_jepa_horizon_address_teacher_entropy": 0.61,
        "flow_jepa_horizon_address_predicted_entropy": 0.74,
        "flow_jepa_horizon_address_update_rms": 0.08,
        "flow_jepa_horizon_address_update_ratio": 0.03,
        "flow_jepa_horizon_address_route_entropy": 0.69,
        "flow_jepa_horizon_address_route_max": 0.18,
        "flow_jepa_horizon_address_fine_entropy": 0.57,
        "flow_jepa_horizon_address_variation": 0.11,
        "flow_jepa_horizon_address_cross_cell_distance": 0.42,
        "grad_flow_dino_horizon_address": 0.004,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v105-train]" in line and "[v105-repr]" in line
    assert "future_raw_delta=0.02000" in line
    assert "future_reliable_norm=0.31000" in line
    assert "future_reference_scale=0.800" in line
    assert "future_normalization_scale=0.060" in line
    assert "future_norm_scale=4:0.060" in line
    assert "horizon_address_loss=0.12000" in line
    assert "horizon_address_teacher_rel=0.270" in line
    assert "horizon_address_route_entropy=0.690" in line
    assert "horizon_address=4.00e-03" in line


def test_partial_v105_without_address_supervision_remains_v104() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 1.0,
            "physical_flow": 0.8,
            "flow_jepa_bounded_flow_coordinates": 1.0,
            "flow_jepa_sequential_horizon_memory": 1.0,
            "role_residual_contract_enabled": 1.0,
            "flow_jepa_horizon_soft_address": 1.0,
            "flow_jepa_future_reliable_normalization": 1.0,
            "flow_jepa_horizon_address_supervision_active": 0.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v104-train]" in line
    assert "[v105-train]" not in line


def test_v106_console_names_interval_and_variance_safe_health() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_interval_stage_enabled": 1.0,
        "flow_jepa_variance_safe_routing": 1.0,
        "flow_jepa_complete_numerical_contract": 1.0,
        "flow_jepa_interval_stage": 0.12,
        "flow_jepa_interval_stage_raw": 0.03,
        "flow_jepa_interval_stage_normalized": 0.07,
        "flow_jepa_interval_stage_direction": 0.11,
        "flow_jepa_interval_stage_direction_floor_min": 0.025,
        "flow_jepa_interval_stage_endpoint": 0.09,
        "flow_jepa_interval_stage_target_scale": 0.41,
        "flow_jepa_interval_stage_reliability": 0.52,
        "flow_jepa_interval_stage_written_delta_rms": 0.02,
        "flow_jepa_interval_stage_carrier_ratio": 0.01,
        "flow_jepa_interval_stage_norm_denominator_min": 0.25,
        "flow_jepa_interval_stage_horizon_4_loss": 0.08,
        "flow_jepa_interval_stage_horizon_4_reliability": 0.61,
        "flow_jepa_interval_stage_horizon_4_write_rms": 0.014,
        "flow_jepa_future_direction_floor_min": 0.018,
        "flow_jepa_horizon_address_value_precontract_rms": 0.33,
        "flow_jepa_horizon_address_value_contraction": 0.04,
        "flow_jepa_horizon_address_value_channel_std": 0.19,
        "attnres_ground_to_world_query_norm_denominator_min": 0.25,
        "attnres_world_to_policy_query_norm_denominator_min": 0.27,
        "evidence_policy_delta_attnres_query_norm_denominator_min": 0.26,
        "evidence_protected_detail_basis_query_norm_denominator_min": 0.25,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v106-train]" in line and "[v106-repr]" in line
    assert "interval_stage_loss=0.12000" in line
    assert "interval_stage_target_scale=0.410" in line
    assert "interval_stage_write=0.020" in line
    assert "interval_stage_direction_floor=2.500e-02" in line
    assert "future_direction_floor=1.800e-02" in line
    assert "interval_h4=l:0.0800/r:0.610/w:0.014" in line
    assert "horizon_address_value_rms=0.330" in line
    assert "g2w_query_norm_denom=0.250" in line
    assert "detail_query_norm_denom=0.250" in line


def test_v107_console_exposes_completed_address_and_actual_role_writes() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_policy_multi_glimpse_address": 1.0,
        "flow_jepa_horizon_cell_fine_address": 1.0,
        "flow_jepa_interval_stage_typed_value": 1.0,
        "role_residual_contract_after_gate": 1.0,
        "flow_jepa_address_policy_glimpse_count": 4.0,
        "flow_jepa_address_policy_glimpse_route_variation": 0.12,
        "attnres_world_to_policy_interval_stage_source_mass": 0.08,
        "role_residual_raw_rms": 4.8,
        "role_residual_proposed_rms": 0.31,
        "role_residual_written_rms": 0.29,
        "role_residual_compression": 0.04,
        "role_residual_grounding_written_rms_max": 0.30,
        "role_residual_world_written_rms_max": 0.32,
        "role_residual_policy_written_rms_max": 0.27,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v107-train]" in line and "[v107-repr]" in line
    assert "horizon_fine_cell_specific=1" in line
    assert "late_detail_glimpses=4" in line
    assert "late_detail_glimpse_var=0.120" in line
    assert "interval_stage_typed_mass=0.080" in line
    assert "role_raw_rms=4.800" in line
    assert "role_proposal_rms=0.310" in line
    assert "role_write_rms=0.290" in line
    assert "role_write_max_g=0.300" in line
    assert "role_write_max_w=0.320" in line
    assert "role_write_max_p=0.270" in line


def test_v108_console_exposes_online_address_boundary_path() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_online_horizon_address": 1.0,
        "flow_jepa_online_horizon_address_write_rms": 0.041,
        "flow_jepa_online_address_boundary_seed_adjacent_cosine": 0.72,
        "flow_jepa_online_address_boundary_post_g3_adjacent_cosine": 0.78,
        "flow_jepa_online_address_boundary_post_address_adjacent_cosine": 0.69,
        "flow_jepa_online_address_boundary_post_w1_adjacent_cosine": 0.73,
        "flow_jepa_online_address_boundary_post_w2_adjacent_cosine": 0.76,
        "flow_jepa_online_address_boundary_post_w3_adjacent_cosine": 0.81,
        "flow_jepa_online_address_boundary_post_interval_adjacent_cosine": 0.79,
        "flow_jepa_online_address_boundary_post_w3_cumulative_address_projection": 0.64,
        "flow_jepa_online_address_boundary_post_interval_cumulative_address_projection": 0.59,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v108-train]" in line and "[v108-repr]" in line
    assert "online_horizon_address=1" in line
    assert "online_address_write=0.041" in line
    assert "horizon_cos_g3=0.780" in line
    assert "horizon_cos_address=0.690" in line
    assert "horizon_cos_w3=0.810" in line
    assert "horizon_cos_interval=0.790" in line
    assert "address_projection_interval=0.590" in line


def test_v109_console_exposes_typed_progressive_address_path() -> None:
    row = {
        "loss": 1.0,
        "physical_flow": 0.8,
        "flow_jepa_progressive_grounding_address": 1.0,
        "flow_jepa_progressive_g1_coarse_entropy": 0.74,
        "flow_jepa_progressive_g1_coarse_max": 0.21,
        "flow_jepa_progressive_g2_fine_entropy": 0.63,
        "flow_jepa_progressive_g2_center_shift": 0.18,
        "flow_jepa_progressive_g3_coarse_bias_rms": 0.42,
        "flow_jepa_progressive_g3_summary_rms": 0.23,
        "flow_jepa_progressive_world_posterior_entropy": 0.69,
        "flow_jepa_progressive_world_horizon_variation": 0.12,
        "flow_jepa_progressive_world_source_prior_max": 0.27,
        "flow_jepa_progressive_world_source_horizon_variation": 0.08,
        "flow_jepa_progressive_policy_prior_active": 1.0,
        "flow_jepa_progressive_policy_world_prior_rms": 0.36,
    }
    line = _evidence_serial_log_line(
        row,
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert "[v109-train]" in line and "[v109-repr]" in line
    assert "progressive_address=1" in line
    assert "g1_coarse_entropy=0.740" in line
    assert "g2_center_shift=0.180" in line
    assert "g3_summary=0.230" in line
    assert "world_horizon_variation=0.120" in line
    assert "world_source_max=0.270" in line
    assert "world_source_variation=0.080" in line
    assert "policy_address_prior=1" in line
    assert "policy_world_prior=0.360" in line


def test_v104_model_path_schema_is_supported_by_the_probe_summarizer() -> None:
    assert (
        _model_path_version("clearvla-v104-model-path-intervention-v3")
        == "v104"
    )
    assert (
        _model_path_version("clearvla-v105-model-path-intervention-v4")
        == "v105"
    )
    assert (
        _model_path_version("clearvla-v106-model-path-intervention-v5")
        == "v106"
    )
    assert (
        _model_path_version("clearvla-v107-model-path-intervention-v6")
        == "v107"
    )
    assert (
        _model_path_version("clearvla-v108-model-path-intervention-v7")
        == "v108"
    )
    assert (
        _model_path_version("clearvla-v109-model-path-intervention-v8")
        == "v109"
    )
    assert (
        _model_path_version("clearvla-v110-model-path-intervention-v9")
        == "v110"
    )
    assert (
        _model_path_version("clearvla-v111-model-path-intervention-v10")
        == "v111"
    )
    assert (
        _model_path_version("clearvla-v112-model-path-intervention-v11")
        == "v112"
    )
    assert (
        _model_path_version("clearvla-v113-model-path-intervention-v12")
        == "v113"
    )
    assert (
        _model_path_version("clearvla-v113-model-path-intervention-v13")
        == "v113"
    )


def test_v95_resume_resolves_explicit_and_derived_hierarchy_identically():
    derived = _saved_flow_jepa_hierarchy(
        {
            "future_anchors": 3,
            "action_horizon": 24,
            "flow_jepa_window_offsets": (),
            "flow_jepa_stage_offset": 0,
        }
    )
    explicit = _saved_flow_jepa_hierarchy(
        {
            "future_anchors": 3,
            "action_horizon": 24,
            "flow_jepa_window_offsets": (8, 16, 24),
            "flow_jepa_stage_offset": 25,
        }
    )
    assert derived == ((8, 16, 24), 25)
    assert explicit == derived


def test_v94_evidence_console_log_uses_only_present_active_fields():
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "motion": 0.2,
            "loss_group_action": 0.32,
            "loss_group_rollout": 0.08,
            "loss_contrib_flow": 0.30,
            "loss_contrib_event": 0.02,
            "loss_ledger_residual": 0.0,
            "evidence_mmd_it_capacity_gate_mass": 0.91,
            "evidence_mmd_it_effective_basis_mass": 29.0,
            "evidence_mmd_it_terminal_prior_weight": 0.25,
            "evidence_mmd_it_dynamic_route_next_fraction": 0.4,
            "evidence_mmd_it_hard_route_next_fraction": 0.25,
            "grad_evidence_mmdit_execution_controller": 0.0,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=0.5,
    )
    assert line.count("\n") == 2
    assert "[v94-train]" in line
    assert "[v94-exec]" in line
    assert "[v94-grad]" in line
    assert "route=soft:0.400/hard:0.250/gap:+0.150" in line
    assert "top_contrib=flow:0.30000/event:0.02000" in line
    assert "exec_controller=0.00e+00" in line
    assert "loss_total=0.400000" in line
    assert "flow_loss=0.300000" in line
    assert "capacity_gate_mass=0.91000" in line
    assert "effective_basis_mass=29.000" in line
    assert "terminal_prior=0.250" in line
    assert "global_preclip=1.00e+00" in line
    assert "latent_cvae" not in line
    assert "hierarchical_mmdit" not in line


def test_v114_console_log_reports_actual_p1_execution_contract() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "flow_jepa_p1_shared_factual": 1.0,
            "flow_jepa_typed_p2_utility_precision": 1.0,
            "flow_jepa_p1_query_rows": 24.0,
            "flow_jepa_p2_query_rows": 96.0,
            "flow_jepa_address_query_chunk_actual": 4.0,
            "flow_jepa_typed_p1_activation_checkpoint": 1.0,
            "flow_jepa_typed_p1_activation_checkpoint_active": 1.0,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=4.0,
    )
    assert "[v114-train]" in line
    assert "[v114-repr]" in line
    assert "p1_query_rows=24" in line
    assert "p2_query_rows=96" in line
    assert "p1_query_chunk=4" in line
    assert "p1_checkpoint_configured=1" in line
    assert "p1_checkpoint_active=1" in line


def test_v119_console_log_reports_grounded_capability_rows() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "grounded_intent_effect_active": 1.0,
            "grounded_g2_g3_semantic_owner_l1": 0.05,
            "grounded_s_interval_goal_attention_entropy": 0.80,
            "grounded_w1_semantic_rms": 0.11,
            "grounded_w2_semantic_rms": 0.22,
            "grounded_p2_effect_read_rms": 0.13,
            "grad_grounded_world_shared_inputs": 1e-3,
            "grad_grounded_world_w1_blocks": 2e-3,
            "grad_grounded_world_w2_blocks": 3e-3,
            "grad_grounded_world_shared_heads": 4e-3,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=4.0,
    )
    assert "[v119-train]" in line
    assert "[v119-ground] active=1" in line
    assert "[v119-intent]" in line
    assert "[v119-effect]" in line
    assert "[v119-policy]" in line
    assert "grounded_w_inputs=1.00e-03" in line
    assert "grounded_w1_blocks=2.00e-03" in line
    assert "grounded_w2_blocks=3.00e-03" in line
    assert "grounded_w_shared_heads=4.00e-03" in line


def test_v116_console_log_reports_supervised_effect_and_terminal_semantics() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "flow_jepa_supervised_effect_mainline_active": 1.0,
            "native_velocity_mse": 0.2,
            "arm_tangent_mse": 0.18,
            "arm_null_mse": 0.02,
            "gripper_tangent_mse": 0.25,
            "gripper_null_mse": 0.03,
            "event_reweight_delta": -0.01,
            "flow_jepa_future_effect_w1_current_loss": 0.04,
            "flow_jepa_future_effect_w2_successor_loss": 0.05,
            "flow_jepa_p2_structured_effect_read_rms": 0.12,
            "flow_jepa_w1_typed_condition_proposal_mass": 0.24,
            "flow_jepa_phase_terminal_mass": 0.08,
            "flow_jepa_execution_terminal_probability": 0.08,
            "evidence_execution_terminal_external_bias": -0.01,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=4.0,
    )
    assert "[v116-train]" in line
    assert "native_velocity_mse=0.200000" in line
    assert "event_reweight_delta=-1.000e-02" in line
    assert "effect_w1_current_loss=0.0400" in line
    assert "effect_w2_successor_loss=0.0500" in line
    assert "p2_effect_read=0.120" in line
    assert "w1_proposal_mass=0.240" in line
    assert "execution_terminal=0.080" in line
    assert "execution_terminal_bias=-0.010" in line


def test_v117_console_log_reports_three_slot_intent_effect_semantics() -> None:
    line = _evidence_serial_log_line(
        {
            "loss": 0.4,
            "physical_flow": 0.3,
            "flow_jepa_stateless_intent_controller_active": 1.0,
            "flow_jepa_future_effect_semantic_loss": 0.04,
            "flow_jepa_future_effect_w1_semantic_loss": 0.03,
            "flow_jepa_future_effect_w2_semantic_loss": 0.06,
            "flow_jepa_future_effect_relative_transition_loss": 0.02,
            "flow_jepa_p2_structured_effect_read_rms": 0.12,
            "flow_jepa_p2_structured_effect_slot_variation": 0.04,
            "flow_jepa_p2_effect_near_mass": 0.45,
            "flow_jepa_p2_effect_mid_mass": 0.35,
            "flow_jepa_p2_effect_late_mass": 0.20,
            "flow_jepa_intent_progress_coordinate": 0.55,
            "flow_jepa_frame_progress": 0.40,
            "flow_jepa_intent_frame_progress_gap": 0.15,
            "flow_jepa_intent_frame_progress_mae": 0.17,
            "flow_jepa_intent_window_selector_max": 0.60,
            "flow_jepa_intent_window_selector_entropy": 0.75,
            "flow_jepa_intent_observation_steps": 5.0,
            "grad": 1.0,
        },
        epoch=1,
        batch_index=20,
        learning_rate=1e-4,
        seconds_per_batch=4.0,
    )
    assert "[v117-train]" in line
    assert "effect_semantic_loss=0.0400" in line
    assert "effect_w1_semantic_loss=0.0300" in line
    assert "effect_w2_semantic_loss=0.0600" in line
    assert "effect_relative_transition_loss=0.0200" in line
    assert "p2_effect_slot_var=0.040" in line
    assert "p2_effect_near=0.450" in line
    assert "p2_effect_mid=0.350" in line
    assert "p2_effect_late=0.200" in line
    assert "intent_progress=0.550" in line
    assert "frame_progress=0.400" in line
    assert "progress_gap=+0.150" in line
    assert "progress_mae=0.170" in line
    assert "intent_selector_entropy=0.750" in line
    assert "intent_observation_steps=5" in line


def test_v94_sampling_diagnostics_keep_z_probe_and_drop_inactive_families():
    assert _keep_sampling_diagnostic("evidence_z_zero_condition_delta", evidence_active=True)
    assert _keep_sampling_diagnostic("evidence_z_shuffle_condition_delta", evidence_active=True)
    assert _keep_sampling_diagnostic("evidence_mmd_it_block_2_update_norm", evidence_active=True)
    assert not _keep_sampling_diagnostic("latent_cvae_primary_z_effect_norm", evidence_active=True)


def test_v94_epoch_console_keeps_event_rate_and_count_evidence():
    line = _evidence_epoch_log_line(
        epoch=1,
        global_step=100,
        train={"loss": 0.4, "physical_flow": 0.3},
        val={
            "full_rmse": 0.1,
            "first_rmse": 0.05,
            "first8_rmse": 0.07,
            "tail_rmse": 0.12,
            "tail_first_ratio": 2.4,
            "arm_full_rmse": 0.08,
            "gripper_full_rmse": 0.2,
            "gripper_event_ratio": 4.0,
            "gripper_pred_events": 400.0,
            "gripper_target_events": 100.0,
            "event_head_pred_events": 150.0,
            "event_head_target_events": 100.0,
        },
    )
    assert "tail_first_ratio=2.400" in line
    assert "grip_event_ratio=4.000" in line
    assert "grip_events_pred=400" in line
    assert "event_head_events_pred=150" in line


def test_motion_head_validation_metrics_report_the_active_auxiliary_head():
    metrics = motion_head_metrics(
        [np.array([[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32)],
        [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)],
    )
    assert metrics["motion_head_accuracy"] == 1.0
    assert metrics["motion_head_precision"] == 1.0
    assert metrics["motion_head_recall"] == 1.0
    assert metrics["motion_head_f1"] == 1.0


def test_v94_run_directory_rejects_accidental_append_without_resume():
    manifest = {
        "schema": "clearvla-run-contract-manifest-v1",
        "fingerprint": "contract-a",
        "contract": {},
    }
    with TemporaryDirectory() as directory:
        out_dir = Path(directory)
        _prepare_run_directory(out_dir=out_dir, manifest=manifest, resume=None)
        assert (out_dir / "run_manifest.json").exists()
        (out_dir / "v39_policy_epochs.jsonl").write_text("{}\n", encoding="utf-8")
        try:
            _prepare_run_directory(out_dir=out_dir, manifest=manifest, resume=None)
        except FileExistsError as error:
            assert "choose a new OUT_DIR" in str(error)
        else:
            raise AssertionError("an existing run must not be appended without --resume")
