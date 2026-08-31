import inspect
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

import clearvla.mainline.model.routing as routing_module
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import (
    ActionSupervision,
    AuditMetadata,
    CurrentObservation,
    FutureSupervision,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
    TrainingBatch,
)
from clearvla.mainline.runtime.evaluation import (
    MatchedCoreAttributionAccumulator,
    MatchedP2InterventionAccumulator,
    ValidationAccumulator,
    _gripper_event_class,
    _post_event_distance,
)
from clearvla.mainline.runtime.logging import (
    DeviceMetricAccumulator,
    JsonlRunLogger,
    active_metrics,
    archival_metrics,
    validate_resume_metric_boundary,
)
from clearvla.mainline.train import (
    _diagnostic_batch_indices,
    _prepare_output_directory,
    _validate,
)


def test_active_logging_keeps_every_current_top_owner() -> None:
    values = {
        "object_intent_public_interval_variation": 0.1,
        "object_plan_recognition_loss": 0.2,
        "object_coarse_action_rms": 0.3,
        "object_teacher_reliability": 0.4,
        "object_w_typed_contribution_rms": 0.5,
        "object_w1_semantic_delta_rms": 0.6,
        "object_w2_semantic_delta_rms": 0.7,
        "condition_goal_keep": 0.95,
        "condition_action_history_keep": 0.90,
        "condition_proposal_keep": 0.75,
        "learning_rate_bottom_decoder": 5.6e-5,
        "inactive_ancestry_metric": 0.8,
    }
    filtered = active_metrics(values)
    assert set(filtered) == set(values) - {"inactive_ancestry_metric"}


def test_active_logging_suppresses_inactive_exact_zero() -> None:
    filtered = active_metrics(
        {
            "object_w_typed_contribution_rms": 0.0,
            "loss_ledger_gap": 0.0,
            "gradient_postglobal_p1_factual_l2": 0.0,
            "object_grounding_mass_conservation_error": 0.0,
            "object_p2_terminal_common_residual_identity_error": 0.0,
            "gradient_tensor_p2_semantic_effect_rms": 0.0,
        }
    )
    assert filtered == {
        "loss_ledger_gap": 0.0,
        "gradient_postglobal_p1_factual_l2": 0.0,
        "object_grounding_mass_conservation_error": 0.0,
        "object_p2_terminal_common_residual_identity_error": 0.0,
        "gradient_tensor_p2_semantic_effect_rms": 0.0,
    }


def test_archival_logging_keeps_active_exact_zero_but_not_ancestry() -> None:
    archived = archival_metrics(
        {
            "object_w_typed_contribution_rms": 0.0,
            "object_p2_effect_precontract_rms": 0.0,
            "observation_flow_rms": 0.0,
            "gripper_private_gate_rms": 0.0,
            "gripper_private_state_delta_rms": 0.0,
            "inactive_ancestry_metric": 0.0,
        }
    )
    assert archived == {
        "object_w_typed_contribution_rms": 0.0,
        "object_p2_effect_precontract_rms": 0.0,
        "observation_flow_rms": 0.0,
        "gripper_private_gate_rms": 0.0,
        "gripper_private_state_delta_rms": 0.0,
    }


def test_decision_console_prioritizes_task_objective_path_and_coverage() -> None:
    values = {
        "loss_total": 1.2,
        "loss_action_flow": 0.8,
        "loss_action_flow_native": 0.7,
        "loss_action_flow_first8": 0.6,
        "loss_action_flow_tail": 0.9,
        "loss_action_flow_band_13_24": 0.91,
        "runtime_window_seconds_per_batch": 1.1,
        # Exact V120-comparable aliases remain archived but should not repeat
        # the same task quantities on the console.
        "loss_action_flow_v120_comparable": 0.8,
    }
    line = JsonlRunLogger.compact_line(
        "train",
        epoch=1,
        batch=20,
        step=20,
        metrics=values,
    )
    for name in set(values) - {"loss_action_flow_v120_comparable"}:
        assert f"{name}=" in line
    assert "loss_action_flow_v120_comparable=" not in line

    details = JsonlRunLogger.diagnostic_lines(
        "train",
        epoch=1,
        batch=20,
        step=20,
        metrics={
            "loss_group_action": 0.8,
            "loss_contrib_action_flow": 0.8,
            "loss_ledger_gap": 0.0,
            "loss_contribution_gap": 0.0,
            "object_grounding_object_content_pair_cosine": 0.34,
            "object_p2_semantic_effect_rms": 0.35,
            "object_p3_protected_policy_precision_rms": 0.41,
            "flow_jepa_address_coarse_variance_min": 0.01,
            "object_p2_terminal_has_null": 0.0,
            "gradient_raw_observation_l2": 0.42,
            "gradient_raw_global_l2": 0.43,
            "gradient_tensor_p3_temporal_rms": 0.44,
            # These remain in metrics.jsonl but are deliberately absent from
            # the bounded console decision surface.
            "object_grounding_candidate_key_rms": 0.31,
            "object_intent_semantic_route_raw_rms": 0.33,
            "object_w_semantic_common_contribution_rms": 0.35,
            "bottom_block_1_executed_update_rms": 0.52,
        },
    )
    joined = "\n".join(details)
    assert len(details) == 5
    assert "loss_group_action=0.8" in joined
    assert "loss_contrib_action_flow=0.8" in joined
    assert "loss_ledger_gap=0" in joined
    assert "object_grounding_object_content_pair_cosine=0.34" in joined
    assert "object_p2_semantic_effect_rms=0.35" in joined
    assert "object_p3_protected_policy_precision_rms=0.41" in joined
    assert "flow_jepa_address_coarse_variance_min=0.01" in joined
    assert "object_p2_terminal_has_null=0" in joined
    assert "gradient_raw_observation_l2=0.42" in joined
    assert "gradient_tensor_p3_temporal_rms=0.44" in joined
    assert "object_grounding_candidate_key_rms=" not in joined
    assert "object_intent_semantic_route_raw_rms=" not in joined
    assert "object_w_semantic_common_contribution_rms=" not in joined
    assert "bottom_block_1_executed_update_rms=" not in joined

    validation = {
        "validation_action_rmse_physical": 0.1,
        "validation_first8_rmse_physical": 0.05,
        "validation_tail_rmse_physical": 0.12,
        "validation_band_13_24_rmse_physical": 0.13,
        "validation_action_rmse_normalized": 0.2,
        "validation_band_13_24_rmse_normalized": 0.23,
        "validation_gripper_band_1_4_rmse_physical": 0.03,
        "validation_gripper_band_5_12_rmse_physical": 0.07,
        "validation_gripper_band_13_24_rmse_physical": 0.19,
        "validation_gripper_post_event_1_2_rmse_physical": 0.11,
        "validation_gripper_post_event_3_6_rmse_physical": 0.15,
        "validation_gripper_post_event_7_plus_rmse_physical": 0.21,
        "validation_gripper_post_event_rows_1_2": 12.0,
        "validation_gripper_post_event_rows_3_6": 18.0,
        "validation_gripper_post_event_rows_7_plus": 24.0,
        "validation_gripper_absolute_branch_band_13_24_rmse_physical": 0.18,
        "validation_gripper_delta_branch_band_13_24_rmse_physical": 0.22,
        "validation_gripper_branch_disagreement_band_13_24_rms_physical": 0.08,
        "validation_gripper_branch_decode_identity_max_abs": 0.0,
        "validation_decoded_gripper_event_f1": 0.4,
        "validation_decoded_gripper_events_predicted": 7.0,
        "validation_decoded_gripper_events_target": 8.0,
        "validation_motion_head_f1": 0.6,
        "validation_proposal_ablation_coverage": 0.1,
        "validation_p2_intervention_coverage": 0.1,
        "validation_p2_intervention_semantic_far_zero_gripper_band_13_24_mse_gain_vs_primary_physical": -0.003,
        "validation_p2_intervention_semantic_far_zero_gripper_band_13_24_action_delta_rmse_physical": 0.04,
        "validation_p2_intervention_semantic_far_zero_post_event_1_2_mse_gain_vs_primary_physical": -0.01,
        "validation_proposal_zero_mse_gain_vs_primary_physical": -0.01,
        "validation_execution_ablation_coverage": 0.05,
        "validation_execution_full_capacity_mse_gain_vs_primary_physical": -0.02,
        "object_p2_effect_postcontract_rms": 0.3,
        "validation_deploy_sampling_outer_world_refinement": 1.0,
        "validation_deploy_sampling_outer_proposal_action_rms": 0.31,
        "validation_deploy_sampling_outer_refined_action_rms": 0.32,
        "validation_deploy_sampling_outer_refined_action_delta_rms": 0.04,
        "validation_deploy_object_action_world_refinement_count": 1.0,
        "validation_deploy_object_action_world_refinement_action_interval_delta_rms": 0.05,
        "validation_deploy_object_action_world_refinement_semantic_delta_change_rms": 0.06,
        "validation_deploy_object_action_world_refinement_transport_change_rms": 0.07,
        "validation_deploy_sampling_outer_final_world_action_interval_mismatch_rms": 0.08,
        "validation_deploy_sampling_outer_final_world_action_delta_mismatch_rms": 0.09,
        "validation_action_estimator_match_coverage": 0.1,
        "validation_action_estimator_to_full_interval_action_rms": 0.03,
        "validation_action_estimator_to_full_interval_action_ratio_vs_coarse": 0.6,
        "validation_action_estimator_full_update_direction_cosine": 0.7,
        "validation_action_estimator_to_full_semantic_rms": 0.02,
        "validation_action_estimator_to_full_semantic_ratio_vs_coarse": 0.5,
        "validation_action_estimator_to_full_transport_rms": 0.004,
        "validation_action_estimator_to_full_transport_ratio_vs_coarse": 0.8,
        "validation_action_estimator_extra_path_runtime_seconds": 0.01,
        "validation_action_estimator_extra_path_live_allocation_gib": 0.0,
        "validation_core_attribution_coverage": 0.1,
        "validation_core_attribution_primary_vs_explicit_none_normalized_action_max_abs": 0.0,
        "validation_core_attribution_primary_vs_explicit_none_normalized_bit_exact": 1.0,
        "validation_core_attribution_world_vs_consequence_neutral_normalized_action_max_abs": 0.0,
        "validation_core_attribution_world_vs_consequence_neutral_normalized_bit_exact": 1.0,
        "validation_core_attribution_wrong_action_world_donor_valid_fraction": 1.0,
        "validation_core_attribution_wrong_action_world_donor_valid_rows": 8.0,
        "validation_core_attribution_wrong_action_world_donor_total_rows": 8.0,
        "validation_core_attribution_controlled_transition_delta_neutral_band_13_24_mse_gain_vs_primary_physical": -0.004,
        "validation_core_attribution_controlled_transition_delta_neutral_band_13_24_action_delta_rmse_physical": 0.02,
        "validation_core_attribution_controlled_transition_delta_neutral_gripper_band_13_24_mse_gain_vs_primary_physical": -0.006,
        "validation_core_attribution_controlled_transition_delta_neutral_gripper_band_13_24_action_delta_rmse_physical": 0.03,
    }
    validation_line = JsonlRunLogger.compact_line(
        "val", epoch=1, batch=None, step=100, metrics=validation
    )
    for name in (
        "validation_action_rmse_physical",
        "validation_first8_rmse_physical",
        "validation_tail_rmse_physical",
        "validation_band_13_24_rmse_physical",
        "validation_action_rmse_normalized",
        "validation_band_13_24_rmse_normalized",
    ):
        assert f"{name}=" in validation_line
    validation_details = "\n".join(
        JsonlRunLogger.diagnostic_lines(
            "val", epoch=1, batch=None, step=100, metrics=validation
        )
    )
    assert "validation_decoded_gripper_events_predicted=7" in validation_details
    assert "validation_decoded_gripper_events_target=8" in validation_details
    assert "validation_motion_head_f1=0.6" in validation_details
    assert "validation_proposal_ablation_coverage=0.1" in validation_details
    assert "validation_execution_ablation_coverage=0.05" in validation_details
    assert "object_p2_effect_postcontract_rms=0.3" in validation_details
    assert "[mainline-val-closure]" in validation_details
    assert (
        "validation_deploy_sampling_outer_final_world_action_interval_mismatch_rms=0.08"
        in validation_details
    )
    assert (
        "validation_deploy_sampling_outer_final_world_action_delta_mismatch_rms=0.09"
        in validation_details
    )
    assert "[mainline-val-action-estimator-match]" in validation_details
    assert "validation_action_estimator_match_coverage=0.1" in validation_details
    assert (
        "validation_action_estimator_full_update_direction_cosine=0.7"
        in validation_details
    )
    assert "validation_gripper_band_13_24_rmse_physical=0.19" in validation_details
    assert "validation_gripper_post_event_7_plus_rmse_physical=0.21" in validation_details
    assert "validation_gripper_post_event_rows_7_plus=24" in validation_details
    assert (
        "validation_gripper_absolute_branch_band_13_24_rmse_physical=0.18"
        in validation_details
    )
    assert "validation_p2_intervention_coverage=0.1" in validation_details
    assert (
        "validation_p2_intervention_semantic_far_zero_gripper_band_13_24_"
        "mse_gain_vs_primary_physical=-0.003" in validation_details
    )
    assert "[mainline-val-core-attribution-id]" in validation_details
    assert "validation_core_attribution_coverage=0.1" in validation_details
    assert (
        "validation_core_attribution_primary_vs_explicit_none_normalized_bit_exact=1"
        in validation_details
    )
    assert "[mainline-val-core-attribution-effect]" in validation_details
    assert (
        "validation_core_attribution_controlled_transition_delta_neutral_"
        "band_13_24_mse_gain_vs_primary_physical=-0.004" in validation_details
    )


def test_gradient_tensor_hooks_observe_backward_without_changing_gradient() -> None:
    register = getattr(routing_module, "register_gradient_rms_metric")
    register_axis = getattr(routing_module, "register_gradient_axis_rms_metrics")
    value = torch.randn(2, 3, requires_grad=True)
    coefficient = torch.tensor([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    metrics: dict[str, torch.Tensor] = {}
    register(value, metrics, "gradient_tensor_value_rms")
    register_axis(
        value,
        metrics,
        ("gradient_tensor_axis_0_rms", "gradient_tensor_axis_1_rms", "gradient_tensor_axis_2_rms"),
        dim=-1,
    )
    (value * coefficient).sum().backward()
    torch.testing.assert_close(value.grad, coefficient)
    torch.testing.assert_close(
        metrics["gradient_tensor_value_rms"],
        coefficient.square().mean().sqrt(),
    )
    for index in range(3):
        torch.testing.assert_close(
            metrics[f"gradient_tensor_axis_{index}_rms"],
            coefficient[:, index].square().mean().sqrt(),
        )
    try:
        register_axis(value, {}, ("only_one",), dim=-1)
    except ValueError as error:
        assert "axis" in str(error)
    else:
        raise AssertionError("axis diagnostics must match the producer ABI")


def test_finite_spike_attribution_names_exact_owner_and_channel_abi() -> None:
    from clearvla.mainline.training.gradient_audit import (
        build_finite_gradient_spike_report,
    )

    class ObservationOwner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.delta_head = torch.nn.Linear(2, 6, bias=False)

    class Owners(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observation = ObservationOwner()
            self.bottom = torch.nn.Linear(2, 2, bias=False)

    owners = Owners()
    owners.observation.delta_head.weight.grad = torch.arange(1, 13).float().reshape(6, 2)
    owners.bottom.weight.grad = torch.ones_like(owners.bottom.weight)
    report = build_finite_gradient_spike_report(
        owners.named_parameters(),
        global_norm=30.0,
        audit_threshold=5.0,
        optimizer_group_name=lambda name: f"group:{name}",
    )
    assert report.max_l2.parameter_name == "observation.delta_head.weight"
    assert report.max_abs.parameter_name == "observation.delta_head.weight"
    assert report.flow_delta_head_channel_l2 is not None
    assert len(report.flow_delta_head_channel_l2) == 6

    owners.bottom.weight.grad = torch.full_like(owners.bottom.weight, 100.0)
    bottom_report = build_finite_gradient_spike_report(
        owners.named_parameters(),
        global_norm=201.0,
        audit_threshold=5.0,
        optimizer_group_name=lambda name: f"group:{name}",
    )
    assert bottom_report.max_l2.parameter_name == "bottom.weight"
    assert bottom_report.flow_delta_head_channel_l2 is None


def test_gradient_preclip_window_owns_weighted_mean_max_and_current() -> None:
    from clearvla.mainline.training.gradient_audit import (
        GradientPreclipWindowAccumulator,
    )

    window = GradientPreclipWindowAccumulator()
    window.update(2.0, weight=2.0, batch_offset=1, global_step=10)
    window.update(5.0, weight=1.0, batch_offset=2, global_step=11)
    window.update(3.0, weight=3.0, batch_offset=3, global_step=12)
    values = window.materialize()
    assert values["gradient_window_preclip_l2_mean"] == 3.0
    assert values["gradient_window_preclip_l2_max"] == 5.0
    assert values["gradient_window_preclip_l2_current"] == 3.0
    assert values["gradient_window_preclip_l2_max_batch_offset"] == 2.0
    assert values["gradient_window_preclip_l2_max_global_step"] == 11.0


def test_epoch_tail_training_window_is_persisted_with_gradient_ownership(tmp_path) -> None:
    import json

    from clearvla.mainline.train import _emit_training_window
    from clearvla.mainline.training.gradient_audit import (
        GradientPreclipWindowAccumulator,
    )

    logger = JsonlRunLogger(tmp_path)
    window_metrics = DeviceMetricAccumulator()
    window_metrics.update({"loss_total": torch.tensor(2.5)}, weight=2.0)
    gradient_window = GradientPreclipWindowAccumulator()
    gradient_window.update(
        4.0,
        weight=2.0,
        batch_offset=1,
        global_step=17,
    )
    values = _emit_training_window(
        logger=logger,
        config=ExperimentConfig(),
        window_metrics=window_metrics,
        gradient_window=gradient_window,
        epoch=3,
        batch=7,
        step=17,
        window_seconds=2.0,
        window_samples=2,
        window_batches=1,
        learning_rate=1.0e-4,
        boundary="epoch_tail",
    )
    row = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["kind"] == "train"
    assert row["window_boundary"] == "epoch_tail"
    assert row["window_batches"] == 1
    assert values["gradient_window_preclip_l2_max"] == 4.0


def test_periodic_console_is_sparser_than_lossless_jsonl(tmp_path, capsys) -> None:
    import json

    from clearvla.mainline.train import _emit_training_window
    from clearvla.mainline.training.gradient_audit import (
        GradientPreclipWindowAccumulator,
    )

    logger = JsonlRunLogger(tmp_path)
    config = ExperimentConfig()

    def emit(batch: int) -> str:
        window_metrics = DeviceMetricAccumulator()
        window_metrics.update(
            {
                "loss_total": torch.tensor(2.5),
                "loss_action_flow": torch.tensor(1.5),
                "loss_group_action": torch.tensor(1.5),
                "gradient_raw_global_l2": torch.tensor(2.0),
            }
        )
        gradient_window = GradientPreclipWindowAccumulator()
        gradient_window.update(
            2.0,
            weight=1.0,
            batch_offset=1,
            global_step=batch,
        )
        _emit_training_window(
            logger=logger,
            config=config,
            window_metrics=window_metrics,
            gradient_window=gradient_window,
            epoch=1,
            batch=batch,
            step=batch,
            window_seconds=1.0,
            window_samples=8,
            window_batches=config.runtime.log_every,
            learning_rate=1.0e-4,
            boundary="periodic",
        )
        return capsys.readouterr().out

    assert emit(40) == ""
    health = emit(100)
    assert "[mainline-train]" in health
    assert "[mainline-train-objective]" not in health
    decision = emit(200)
    assert "[mainline-train]" in decision
    assert "[mainline-train-objective]" in decision
    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["batch"] for row in rows] == [40, 100, 200]
    assert all("loss_group_action" in row["metrics"] for row in rows)


def test_gripper_event_metric_rejects_the_opposite_event_direction() -> None:
    target_class = _gripper_event_class(torch.tensor([0.2, -0.2, 0.0]))
    opposite_class = _gripper_event_class(torch.tensor([-0.2, 0.2, 0.0]))
    target_event = target_class != 0
    predicted_event = opposite_class != 0
    true_positive = target_event & (opposite_class == target_class)
    assert target_event.sum() == 2
    assert predicted_event.sum() == 2
    assert true_positive.sum() == 0


def test_device_metric_accumulator_preserves_dynamic_key_weighting() -> None:
    accumulator = DeviceMetricAccumulator()
    accumulator.update({"always": torch.tensor(2.0)}, weight=2.0)
    accumulator.update(
        {"always": torch.tensor(4.0), "diagnostic": torch.tensor(7.0)},
        weight=1.0,
    )
    result = accumulator.materialize()
    assert abs(result["always"] - 8.0 / 3.0) < 1e-6
    assert result["diagnostic"] == 7.0


def test_validation_diagnostic_budget_is_spread_over_the_full_loader() -> None:
    assert _diagnostic_batch_indices(planned_batches=181, budget=4) == {
        1,
        61,
        121,
        181,
    }
    assert _diagnostic_batch_indices(planned_batches=5, budget=0) == set(
        range(1, 6)
    )
    assert _diagnostic_batch_indices(planned_batches=0, budget=4) == set()


def test_fresh_output_directory_cannot_reuse_an_existing_run(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        _prepare_output_directory(output, exact_resume=False)
    except ValueError as error:
        assert "require an empty output directory" in str(error)
    else:
        raise AssertionError("fresh runs must not append to an existing metric stream")


def test_exact_resume_requires_context_in_a_nonempty_output_directory(
    tmp_path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "orphan.txt").write_text("unowned", encoding="utf-8")
    try:
        _prepare_output_directory(output, exact_resume=True)
    except ValueError as error:
        assert "without run_context.json" in str(error)
    else:
        raise AssertionError("resume must not append to an unowned output directory")


def test_exact_resume_metric_stream_must_end_at_checkpoint(tmp_path) -> None:
    output = tmp_path / "run"
    logger = JsonlRunLogger(output)
    logger.write("train", epoch=2, batch=20, step=120, metrics={})
    try:
        validate_resume_metric_boundary(
            output,
            checkpoint_epoch=1,
            checkpoint_step=100,
        )
    except ValueError as error:
        assert "committed epoch row" in str(error)
    else:
        raise AssertionError("partial next-epoch logging must reject exact resume")

    (output / "metrics.jsonl").write_text(
        '{"kind":"epoch","epoch":2,"step":200}\n',
        encoding="utf-8",
    )
    try:
        validate_resume_metric_boundary(
            output,
            checkpoint_epoch=1,
            checkpoint_step=100,
        )
    except ValueError as error:
        assert "metrics/checkpoint boundary differs" in str(error)
    else:
        raise AssertionError("an older checkpoint must not append after newer metrics")


def test_exact_resume_metric_stream_accepts_matching_or_new_output(tmp_path) -> None:
    output = tmp_path / "run"
    logger = JsonlRunLogger(output)
    logger.write("epoch", epoch=3, step=300, train={}, validation={})
    validate_resume_metric_boundary(
        output,
        checkpoint_epoch=3,
        checkpoint_step=300,
    )
    validate_resume_metric_boundary(
        tmp_path / "new-run",
        checkpoint_epoch=3,
        checkpoint_step=300,
    )


def test_validation_reports_explicit_normalized_and_physical_units() -> None:
    config = ExperimentConfig()
    dims = config.dimensions
    batch = 1
    normalized = torch.zeros(batch, dims.action_horizon, dims.action_dim)
    history = ObservableHistory(
        state=torch.zeros(batch, dims.state_dim),
        action_state=torch.zeros(batch, dims.action_dim),
        state_history=torch.zeros(batch, dims.state_history_length, dims.state_dim),
        executed_action_history=torch.zeros(batch, dims.executed_history_length, dims.action_dim),
    )
    training_batch = TrainingBatch(
        online=OnlinePolicyInput(
            observation=CurrentObservation(
                dino_history=torch.zeros(
                    batch,
                    dims.visual_history_length,
                    dims.num_cameras,
                    dims.patches_per_camera,
                    dims.visual_token_dim,
                ),
                raw_rgb=torch.zeros(
                    batch,
                    dims.visual_history_length,
                    dims.num_cameras,
                    3,
                    32,
                    32,
                ),
            ),
            history=history,
            goal=GoalCondition(
                tokens=torch.zeros(batch, 1, dims.goal_token_dim),
                mask=torch.ones(batch, 1, dtype=torch.bool),
            ),
        ),
        action_target=ActionSupervision(
            normalized=normalized,
            raw_units=torch.zeros_like(normalized),
            current_raw_units=torch.zeros(batch, dims.action_dim),
            gripper_transition_boundary=torch.zeros(batch, dims.action_dim),
            gripper_transition_boundary_raw_units=torch.zeros(
                batch, dims.action_dim
            ),
        ),
        future=FutureSupervision(
            dino_supports=torch.zeros(
                batch,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float16,
            ),
            action_sequence=torch.zeros(batch, 48, dims.action_dim),
            state_sequence=torch.zeros(batch, 48, dims.state_dim),
            offsets=torch.arange(4, 49, 4)[None],
        ),
        audit=AuditMetadata(),
    )
    scale = np.full((1, dims.action_dim), 2.0, dtype=np.float32)
    normalizer = ArrayNormalizer(
        offset=np.zeros_like(scale),
        scale=scale,
        mean=np.zeros_like(scale),
        std=np.ones_like(scale),
        minimum=-np.ones_like(scale),
        maximum=np.ones_like(scale),
        mode="zscore",
    )
    accumulator = ValidationAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
    )
    accumulator.update(torch.ones_like(normalized), training_batch)
    metrics = accumulator.means()
    gripper_band_names = (
        "normalized_gripper_band_1_4",
        "normalized_gripper_band_5_12",
        "normalized_gripper_band_13_24",
    )
    assert sum(accumulator.element_count[name] for name in gripper_band_names) == (
        accumulator.element_count["normalized_gripper"]
    )
    torch.testing.assert_close(
        sum(accumulator.square_error[name] for name in gripper_band_names),
        accumulator.square_error["normalized_gripper"],
    )
    assert metrics["validation_action_rmse_normalized"] == 1.0
    assert metrics["validation_action_rmse_physical"] == 0.5
    assert metrics["validation_band_1_4_rmse_normalized"] == 1.0
    assert metrics["validation_band_5_12_rmse_normalized"] == 1.0
    assert metrics["validation_band_13_24_rmse_normalized"] == 1.0
    assert metrics["validation_band_13_24_rmse_physical"] == 0.5
    assert metrics["validation_gripper_band_1_4_rmse_normalized"] == 1.0
    assert metrics["validation_gripper_band_5_12_rmse_physical"] == 0.5
    assert metrics["validation_gripper_band_13_24_rmse_physical"] == 0.5
    assert "validation_decoded_gripper_event_ratio" in metrics
    assert "validation_motion_head_f1" in metrics
    assert "validation_action_rmse" not in metrics

    core = MatchedCoreAttributionAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
        gripper_event_threshold=config.objectives.gripper_event_threshold,
        arm_motion_threshold=config.objectives.arm_motion_threshold,
    )
    primary = torch.zeros_like(normalized)
    shifted = torch.ones_like(normalized)
    core.update_primary(primary, training_batch)
    core.update(
        "explicit_none",
        primary_action=primary,
        counterfactual_action=primary,
        batch=training_batch,
        boundary_metrics={"intervention_active": torch.tensor(0.0)},
    )
    core.update(
        "wrong_action_world",
        primary_action=primary,
        counterfactual_action=shifted,
        batch=training_batch,
        boundary_metrics={
            "donor_valid_rows": torch.tensor(1.0),
            "donor_total_rows": torch.tensor(1.0),
            "donor_valid_fraction": torch.tensor(1.0),
            "retained_support_identity_max_abs": torch.tensor(0.0),
        },
    )
    core.update_identity("primary_vs_explicit_none", primary, primary)
    core_metrics = core.means()
    assert core_metrics["validation_core_attribution_primary_batches"] == 1.0
    assert (
        core_metrics[
            "validation_core_attribution_primary_decoded_gripper_events_target"
        ]
        == 0.0
    )
    assert (
        core_metrics[
            "validation_core_attribution_wrong_action_world_decoded_gripper_"
            "events_predicted"
        ]
        >= 0.0
    )
    assert (
        core_metrics[
            "validation_core_attribution_explicit_none_action_delta_rmse_physical"
        ]
        == 0.0
    )
    assert (
        core_metrics[
            "validation_core_attribution_wrong_action_world_action_delta_rmse_physical"
        ]
        == 0.5
    )
    assert (
        core_metrics[
            "validation_core_attribution_wrong_action_world_gripper_band_13_24_"
            "action_delta_rmse_physical"
        ]
        == 0.5
    )
    assert (
        core_metrics[
            "validation_core_attribution_primary_vs_explicit_none_normalized_"
            "action_max_abs"
        ]
        == 0.0
    )
    assert (
        core_metrics[
            "validation_core_attribution_primary_vs_explicit_none_normalized_bit_exact"
        ]
        == 1.0
    )
    assert (
        core_metrics[
            "validation_core_attribution_wrong_action_world_retained_support_"
            "identity_max_abs"
        ]
        == 0.0
    )
    try:
        core.update(
            "explicit_none",
            primary_action=primary,
            counterfactual_action=primary,
            batch=training_batch,
        )
    except ValueError as error:
        assert "one primary update" in str(error)
    else:
        raise AssertionError("a counterfactual cannot double-count one primary batch")


def test_post_event_distance_resets_and_excludes_pre_event_rows() -> None:
    target_event = torch.tensor(
        [[False, False, True, False, False, True, False]],
        dtype=torch.bool,
    )
    torch.testing.assert_close(
        _post_event_distance(target_event),
        torch.tensor([[-1, -1, 0, 1, 2, 0, 1]]),
    )


def test_validation_reports_gripper_persistence_between_target_events() -> None:
    config = ExperimentConfig()
    dims = config.dimensions
    raw_target = torch.zeros(1, dims.action_horizon, dims.action_dim)
    raw_target[:, 2:8, -1] = 1.0
    prediction = raw_target.clone()
    prediction[:, 3:5, -1] += 1.0
    prediction[:, 5:8, -1] += 2.0
    prediction[:, 9:11, -1] += 3.0
    prediction[:, 11:15, -1] += 4.0
    prediction[:, 15:24, -1] += 5.0
    history = ObservableHistory(
        state=torch.zeros(1, dims.state_dim),
        action_state=torch.zeros(1, dims.action_dim),
        state_history=torch.zeros(1, dims.state_history_length, dims.state_dim),
        executed_action_history=torch.zeros(
            1, dims.executed_history_length, dims.action_dim
        ),
    )
    batch = TrainingBatch(
        online=OnlinePolicyInput(
            observation=CurrentObservation(
                dino_history=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    dims.patches_per_camera,
                    dims.visual_token_dim,
                ),
                raw_rgb=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    3,
                    32,
                    32,
                ),
            ),
            history=history,
            goal=GoalCondition(
                tokens=torch.zeros(1, 1, dims.goal_token_dim),
                mask=torch.ones(1, 1, dtype=torch.bool),
            ),
        ),
        action_target=ActionSupervision(
            normalized=raw_target,
            raw_units=raw_target,
            current_raw_units=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary_raw_units=torch.zeros(1, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.zeros(
                1,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float16,
            ),
            action_sequence=torch.zeros(1, 48, dims.action_dim),
            state_sequence=torch.zeros(1, 48, dims.state_dim),
            offsets=torch.arange(4, 49, 4)[None],
        ),
        audit=AuditMetadata(),
    )
    identity = np.ones((1, dims.action_dim), dtype=np.float32)
    normalizer = ArrayNormalizer(
        offset=np.zeros_like(identity),
        scale=identity,
        mean=np.zeros_like(identity),
        std=identity,
        minimum=-identity,
        maximum=identity,
        mode="identity",
    )
    accumulator = ValidationAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
    )
    accumulator.update(prediction, batch)
    metrics = accumulator.means()
    assert metrics["validation_gripper_post_event_rows_1_2"] == 4.0
    assert metrics["validation_gripper_post_event_rows_3_6"] == 7.0
    assert metrics["validation_gripper_post_event_rows_7_plus"] == 9.0
    assert np.isclose(
        metrics["validation_gripper_post_event_1_2_rmse_physical"],
        np.sqrt(5.0),
    )
    assert np.isclose(
        metrics["validation_gripper_post_event_3_6_rmse_physical"],
        np.sqrt(76.0 / 7.0),
    )
    assert metrics["validation_gripper_post_event_7_plus_rmse_physical"] == 5.0


def test_validation_first_gripper_transition_uses_the_profile_command_boundary() -> None:
    horizon = 24
    target = torch.zeros(1, horizon, 7)
    target[..., -1] = 1.0
    current_qpos_boundary = torch.zeros(1, 7)
    previous_command_boundary = torch.zeros(1, 7)
    previous_command_boundary[..., -1] = 1.0
    batch = SimpleNamespace(
        action_target=SimpleNamespace(
            normalized=target,
            raw_units=target,
            current_raw_units=current_qpos_boundary,
            gripper_transition_boundary=previous_command_boundary,
            gripper_transition_boundary_raw_units=previous_command_boundary,
        ),
        online=SimpleNamespace(
            history=SimpleNamespace(action_state=current_qpos_boundary),
        ),
    )
    identity = ArrayNormalizer.fit_identity(
        [np.asarray([[0.0] * 7, [1.0] * 7], dtype=np.float32)]
    )
    accumulator = ValidationAccumulator.from_action_normalizer(
        identity,
        device=torch.device("cpu"),
        gripper_event_threshold=0.1,
    )
    accumulator.update(target.clone(), cast(TrainingBatch, batch))
    metrics = accumulator.means()
    assert metrics["validation_decoded_gripper_events_target"] == 0.0
    assert metrics["validation_decoded_gripper_events_predicted"] == 0.0

    target_with_transition = target.clone()
    target_with_transition[:, 4:, -1] = 0.0
    batch.action_target.normalized = target_with_transition
    batch.action_target.raw_units = target_with_transition
    accumulator = ValidationAccumulator.from_action_normalizer(
        identity,
        device=torch.device("cpu"),
        gripper_event_threshold=0.1,
    )
    accumulator.update(
        target_with_transition.clone(), cast(TrainingBatch, batch)
    )
    assert accumulator.means()["validation_decoded_gripper_events_target"] == 1.0


def test_validation_gripper_branches_and_event_context_reconstruct_band_error() -> None:
    config = ExperimentConfig()
    dims = config.dimensions
    raw_target = torch.zeros(1, dims.action_horizon, dims.action_dim)
    raw_target[:, 3:10, -1] = 1.0
    raw_target[:, 10:, -1] = 0.25
    absolute = torch.linspace(-0.2, 0.7, dims.action_horizon)[None, :, None]
    cumulative = torch.linspace(0.4, -0.5, dims.action_horizon)[None, :, None]
    prediction = raw_target.clone()
    prediction[..., -1:] = 0.75 * absolute + 0.25 * cumulative
    physical_field = torch.zeros(1, dims.action_horizon, 18)
    physical_field[..., 12:13] = absolute
    cumulative_boundary = torch.cat(
        (torch.zeros(1, 1, 1), cumulative[:, :-1]),
        dim=1,
    )
    physical_field[..., 13:14] = cumulative - cumulative_boundary
    history = ObservableHistory(
        state=torch.zeros(1, dims.state_dim),
        action_state=torch.zeros(1, dims.action_dim),
        state_history=torch.zeros(1, dims.state_history_length, dims.state_dim),
        executed_action_history=torch.zeros(
            1, dims.executed_history_length, dims.action_dim
        ),
    )
    batch = TrainingBatch(
        online=OnlinePolicyInput(
            observation=CurrentObservation(
                dino_history=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    dims.patches_per_camera,
                    dims.visual_token_dim,
                ),
                raw_rgb=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    3,
                    32,
                    32,
                ),
            ),
            history=history,
            goal=GoalCondition(
                tokens=torch.zeros(1, 1, dims.goal_token_dim),
                mask=torch.ones(1, 1, dtype=torch.bool),
            ),
        ),
        action_target=ActionSupervision(
            normalized=raw_target,
            raw_units=raw_target,
            current_raw_units=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary_raw_units=torch.zeros(1, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.zeros(
                1,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float16,
            ),
            action_sequence=torch.zeros(1, 48, dims.action_dim),
            state_sequence=torch.zeros(1, 48, dims.state_dim),
            offsets=torch.arange(4, 49, 4)[None],
        ),
        audit=AuditMetadata(),
    )
    identity = np.ones((1, dims.action_dim), dtype=np.float32)
    normalizer = ArrayNormalizer(
        offset=np.zeros_like(identity),
        scale=identity,
        mean=np.zeros_like(identity),
        std=identity,
        minimum=-identity,
        maximum=identity,
        mode="identity",
    )
    accumulator = ValidationAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
    )
    accumulator.update(
        prediction,
        batch,
        physical_field=physical_field,
        gripper_decode_delta_blend=0.25,
    )
    metrics = accumulator.means()
    for band_name, band_slice in (
        ("1_4", slice(0, 4)),
        ("5_12", slice(4, 12)),
        ("13_24", slice(12, 24)),
    ):
        target = raw_target[:, band_slice, -1:]
        expected_absolute = (absolute[:, band_slice] - target).square().mean().sqrt()
        expected_cumulative = (cumulative[:, band_slice] - target).square().mean().sqrt()
        expected_disagreement = (
            absolute[:, band_slice] - cumulative[:, band_slice]
        ).square().mean().sqrt()
        assert np.isclose(
            metrics[
                f"validation_gripper_absolute_branch_band_{band_name}_rmse_physical"
            ],
            float(expected_absolute),
        )
        assert np.isclose(
            metrics[
                f"validation_gripper_delta_branch_band_{band_name}_rmse_physical"
            ],
            float(expected_cumulative),
        )
        assert np.isclose(
            metrics[
                f"validation_gripper_branch_disagreement_band_{band_name}_rms_physical"
            ],
            float(expected_disagreement),
        )
        total_rows = 0.0
        total_square_error = 0.0
        for context_name in (
            "before_any_event",
            "event",
            "post_1_2",
            "post_3_6",
            "post_7_plus",
        ):
            stem = f"validation_gripper_band_{band_name}_{context_name}"
            rows = metrics[f"{stem}_rows"]
            total_rows += rows
            total_square_error += metrics[f"{stem}_rmse_physical"] ** 2 * rows
        expected_rows = float(band_slice.stop - band_slice.start)
        assert total_rows == expected_rows
        assert np.isclose(
            total_square_error / expected_rows,
            metrics[f"validation_gripper_band_{band_name}_rmse_physical"] ** 2,
        )

    paired = MatchedP2InterventionAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
        gripper_event_threshold=config.objectives.gripper_event_threshold,
        arm_motion_threshold=config.objectives.arm_motion_threshold,
    )
    paired.update(
        "semantic_far_zero",
        primary_action=raw_target,
        counterfactual_action=prediction,
        batch=batch,
    )
    paired_metrics = paired.means()
    assert paired_metrics["validation_p2_intervention_semantic_far_zero_batches"] == 1.0
    assert (
        paired_metrics[
            "validation_p2_intervention_semantic_far_zero_gripper_band_13_24_"
            "mse_gain_vs_primary_physical"
        ]
        < 0.0
    )
    assert (
        paired_metrics[
            "validation_p2_intervention_semantic_far_zero_gripper_band_13_24_"
            "action_delta_rmse_physical"
        ]
        > 0.0
    )


def test_p2_replay_reuses_primary_noise_and_clears_the_eval_seam() -> None:
    source = inspect.getsource(_validate)
    assert "initial_physical_noise=prediction.initial_physical_noise" in source
    assert "for mode in reader.INTERVENTION_MODES" in source
    assert "finally:" in source
    assert "reader.clear_eval_intervention()" in source
    assert "core_attribution.update_primary(prediction.action, batch)" in source
    assert "for mode in CORE_ATTRIBUTION_MODES" in source
    assert "counterfactual_cache = refined_cache" in source
    assert "consequence_module.clear_eval_intervention()" in source
    assert "transition_module.clear_eval_intervention()" in source
    assert '"primary_vs_explicit_none"' in source
    assert '"world_vs_consequence_neutral"' in source


def test_validation_keeps_decoded_events_and_motion_head_semantically_separate() -> None:
    config = ExperimentConfig()
    dims = config.dimensions
    normalized = torch.zeros(1, dims.action_horizon, dims.action_dim)
    raw = torch.zeros_like(normalized)
    # The demonstration closes the gripper at row zero and moves the arm in
    # physical-field space.  The decoded action below remains a complete hold.
    raw[0, 0:, -1] = 0.2
    history = ObservableHistory(
        state=torch.zeros(1, dims.state_dim),
        action_state=torch.zeros(1, dims.action_dim),
        state_history=torch.zeros(1, dims.state_history_length, dims.state_dim),
        executed_action_history=torch.zeros(
            1, dims.executed_history_length, dims.action_dim
        ),
    )
    training_batch = TrainingBatch(
        online=OnlinePolicyInput(
            observation=CurrentObservation(
                dino_history=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    dims.patches_per_camera,
                    dims.visual_token_dim,
                ),
                raw_rgb=torch.zeros(
                    1,
                    dims.visual_history_length,
                    dims.num_cameras,
                    3,
                    32,
                    32,
                ),
            ),
            history=history,
            goal=GoalCondition(
                tokens=torch.zeros(1, 1, dims.goal_token_dim),
                mask=torch.ones(1, 1, dtype=torch.bool),
            ),
        ),
        action_target=ActionSupervision(
            normalized=normalized,
            raw_units=raw,
            current_raw_units=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary=torch.zeros(1, dims.action_dim),
            gripper_transition_boundary_raw_units=torch.zeros(1, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.zeros(
                1,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float16,
            ),
            action_sequence=torch.zeros(1, 48, dims.action_dim),
            state_sequence=torch.zeros(1, 48, dims.state_dim),
            offsets=torch.arange(4, 49, 4)[None],
        ),
        audit=AuditMetadata(),
    )
    identity = np.ones((1, dims.action_dim), dtype=np.float32)
    normalizer = ArrayNormalizer(
        offset=np.zeros_like(identity),
        scale=identity,
        mean=np.zeros_like(identity),
        std=identity,
        minimum=-identity,
        maximum=identity,
        mode="identity",
    )
    accumulator = ValidationAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
    )
    motion_logits = torch.full((1, dims.action_horizon), -10.0)
    motion_logits[:, 0] = 10.0
    motion_target = torch.zeros(1, dims.action_horizon, dtype=torch.bool)
    motion_target[:, 0] = True
    accumulator.update(
        torch.zeros_like(normalized),
        training_batch,
        motion_logits=motion_logits,
        motion_target=motion_target,
    )
    metrics = accumulator.means()
    assert metrics["validation_decoded_gripper_event_f1"] == 0.0
    assert not any("event_head" in name for name in metrics)
    assert metrics["validation_motion_head_f1"] == 1.0
    assert metrics["validation_decoded_motion_f1"] == 0.0
