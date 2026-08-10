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
)
from clearvla.mainline.train import _prepare_output_directory


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
        "inactive_ancestry_metric": 0.8,
    }
    filtered = active_metrics(values)
    assert set(filtered) == set(values) - {"inactive_ancestry_metric"}


def test_active_logging_suppresses_inactive_exact_zero() -> None:
    filtered = active_metrics(
        {
            "object_w_typed_innovation_rms": 0.0,
            "loss_ledger_gap": 0.0,
        }
    )
    assert filtered == {"loss_ledger_gap": 0.0}


def test_compact_logging_exposes_the_schema17_failure_boundaries() -> None:
    values = {
        "loss_future_successor": 0.12,
        "object_grounding_object_content_pair_cosine": 0.34,
        "object_intent_interval_variation": 0.56,
        "object_w_intent_object_interaction_rms": 0.07,
        "object_w_action_object_interaction_rms": 0.08,
        "object_w2_interval_adjacent_cosine": 0.78,
        "action_flow_balanced_band_13_24": 0.9,
        "action_gripper_event_flow": 1.1,
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
                dino_tokens=torch.zeros(
                    batch,
                    dims.num_cameras,
                    dims.patches_per_camera,
                    dims.visual_token_dim,
                ),
                raw_rgb=torch.zeros(
                    batch,
                    dims.raw_pair_length,
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
    assert "validation_action_rmse" not in metrics
