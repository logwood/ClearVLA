import numpy as np
import torch

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
from clearvla.mainline.runtime.evaluation import ValidationAccumulator, _gripper_event_class
from clearvla.mainline.runtime.logging import (
    DeviceMetricAccumulator,
    JsonlRunLogger,
    active_metrics,
    archival_metrics,
    validate_resume_metric_boundary,
)
from clearvla.mainline.train import _diagnostic_batch_indices, _prepare_output_directory


def test_active_logging_keeps_every_current_top_owner() -> None:
    values = {
        "object_intent_interval_variation": 0.1,
        "object_plan_recognition_loss": 0.2,
        "object_coarse_action_rms": 0.3,
        "object_teacher_reliability": 0.4,
        "object_w_typed_innovation_rms": 0.5,
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
            "object_w_typed_innovation_rms": 0.0,
            "loss_ledger_gap": 0.0,
            "gradient_postglobal_p1_factual_l2": 0.0,
            "object_grounding_mass_conservation_error": 0.0,
        }
    )
    assert filtered == {
        "loss_ledger_gap": 0.0,
        "gradient_postglobal_p1_factual_l2": 0.0,
        "object_grounding_mass_conservation_error": 0.0,
    }


def test_archival_logging_keeps_active_exact_zero_but_not_ancestry() -> None:
    archived = archival_metrics(
        {
            "object_w_typed_innovation_rms": 0.0,
            "object_p2_effect_precontract_rms": 0.0,
            "observation_flow_rms": 0.0,
            "inactive_ancestry_metric": 0.0,
        }
    )
    assert archived == {
        "object_w_typed_innovation_rms": 0.0,
        "object_p2_effect_precontract_rms": 0.0,
        "observation_flow_rms": 0.0,
    }


def test_compact_logging_exposes_the_active_failure_boundaries() -> None:
    values = {
        "loss_future_successor": 0.12,
        "object_grounding_object_content_pair_cosine": 0.34,
        "object_intent_interval_variation": 0.56,
        "object_w_intent_object_interaction_rms": 0.07,
        "object_w_action_object_interaction_rms": 0.08,
        "object_w2_interval_adjacent_cosine": 0.78,
        "loss_action_flow_band_13_24": 0.9,
    }
    line = JsonlRunLogger.compact_line(
        "train",
        epoch=1,
        batch=20,
        step=20,
        metrics=values,
    )
    for name in values:
        assert f"{name}=" in line

    details = JsonlRunLogger.diagnostic_lines(
        "train",
        epoch=1,
        batch=20,
        step=20,
        metrics={
            **values,
            "loss_action_gripper_event_flow": 1.1,
            "loss_flow_recent_warp": 0.2,
            "loss_flow_earlier_warp": 0.3,
            "object_grounding_candidate_key_rms": 0.31,
            "object_w_typed_innovation_rms": 0.32,
            "object_w2_interval_0_semantic_delta_rms": 0.321,
            "object_teacher_interval_0_semantic_delta_rms": 0.322,
            "loss_future_interval_0_semantic_delta": 0.323,
            "object_p2_intent_score_max_abs": 0.33,
            "p1_completed_fact_rms": 0.4,
            "object_p3_effect_rms": 0.41,
            "bottom_capacity_mean": 0.5,
            "bottom_controller_common_ratio": 0.51,
            "bottom_block_1_executed_update_rms": 0.52,
            "validation_sampling_diagnostic_coverage": 0.09,
            "validation_proposal_ablation_coverage": 0.09,
            "validation_execution_ablation_coverage": 0.04,
            "validation_proposal_zero_mse_gain_vs_primary_physical": -0.01,
            "validation_execution_full_capacity_mse_gain_vs_primary_physical": -0.02,
        },
    )
    joined = "\n".join(details)
    assert "loss_action_gripper_event_flow=1.1" in joined
    assert "loss_flow_recent_warp=0.2" in joined
    assert "loss_flow_earlier_warp=0.3" in joined
    assert "object_grounding_candidate_key_rms=0.31" in joined
    assert "object_w_typed_innovation_rms=0.32" in joined
    assert "object_w2_interval_0_semantic_delta_rms=0.321" in joined
    assert "object_teacher_interval_0_semantic_delta_rms=0.322" in joined
    assert "loss_future_interval_0_semantic_delta=0.323" in joined
    assert "object_p2_intent_score_max_abs=0.33" in joined
    assert "p1_completed_fact_rms=0.4" in joined
    assert "object_p3_effect_rms=0.41" in joined
    assert "bottom_capacity_mean=0.5" in joined
    assert "bottom_controller_common_ratio=0.51" in joined
    assert "bottom_block_1_executed_update_rms=0.52" in joined
    assert "validation_sampling_diagnostic_coverage=0.09" in joined
    assert "validation_proposal_ablation_coverage=0.09" in joined
    assert "validation_execution_ablation_coverage=0.04" in joined
    assert "validation_proposal_zero_mse_gain_vs_primary_physical=-0.01" in joined
    assert "validation_execution_full_capacity_mse_gain_vs_primary_physical=-0.02" in joined


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
    assert metrics["validation_action_rmse_normalized"] == 1.0
    assert metrics["validation_action_rmse_physical"] == 0.5
    assert metrics["validation_band_1_4_rmse_normalized"] == 1.0
    assert metrics["validation_band_5_12_rmse_normalized"] == 1.0
    assert metrics["validation_band_13_24_rmse_normalized"] == 1.0
    assert metrics["validation_band_13_24_rmse_physical"] == 0.5
    assert "validation_decoded_gripper_event_ratio" in metrics
    assert "validation_motion_head_f1" in metrics
    assert "validation_action_rmse" not in metrics


def test_validation_keeps_decoded_events_and_auxiliary_heads_semantically_separate() -> None:
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
    event_logits = torch.zeros(1, dims.action_horizon, 3)
    event_logits[..., 0] = 5.0
    event_logits[:, 0, 2] = 10.0
    motion_logits = torch.full((1, dims.action_horizon), -10.0)
    motion_logits[:, 0] = 10.0
    motion_target = torch.zeros(1, dims.action_horizon, dtype=torch.bool)
    motion_target[:, 0] = True
    accumulator.update(
        torch.zeros_like(normalized),
        training_batch,
        event_logits=event_logits,
        motion_logits=motion_logits,
        motion_target=motion_target,
    )
    metrics = accumulator.means()
    assert metrics["validation_decoded_gripper_event_f1"] == 0.0
    assert metrics["validation_event_head_f1"] == 1.0
    assert metrics["validation_event_head_close_f1"] == 1.0
    assert metrics["validation_motion_head_f1"] == 1.0
    assert metrics["validation_decoded_motion_f1"] == 0.0
