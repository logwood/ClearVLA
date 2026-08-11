from dataclasses import fields

import pytest
import torch

from clearvla.mainline.config import ExperimentConfig
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


def _batch(batch: int = 2) -> TrainingBatch:
    cfg = ExperimentConfig()
    dims = cfg.dimensions
    online = OnlinePolicyInput(
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
                336,
                336,
            ),
        ),
        history=ObservableHistory(
            state=torch.zeros(batch, dims.state_dim),
            action_state=torch.zeros(batch, dims.action_dim),
            state_history=torch.zeros(batch, dims.state_history_length, dims.state_dim),
            executed_action_history=torch.zeros(
                batch, dims.executed_history_length, dims.action_dim
            ),
        ),
        goal=GoalCondition(
            tokens=torch.zeros(batch, 7, dims.goal_token_dim),
            mask=torch.ones(batch, 7, dtype=torch.bool),
        ),
    )
    return TrainingBatch(
        online=online,
        action_target=ActionSupervision(
            normalized=torch.zeros(batch, dims.action_horizon, dims.action_dim),
            raw_units=torch.zeros(batch, dims.action_horizon, dims.action_dim),
            current_raw_units=torch.zeros(batch, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.zeros(
                batch,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                dtype=torch.float32,
            ),
            action_sequence=torch.zeros(batch, 48, dims.action_dim),
            state_sequence=torch.zeros(batch, 48, dims.state_dim),
            offsets=torch.arange(1, dims.future_supports + 1)[None].expand(batch, -1),
        ),
    )


def test_online_input_cannot_carry_future_or_action_target() -> None:
    names = {field.name for field in fields(OnlinePolicyInput)}
    assert names == {"observation", "history", "goal"}
    assert not names & {
        "future",
        "future_training_pack",
        "target_visual",
        "target_action",
        "allow_future_training_evidence",
    }


def test_training_partitions_validate_without_value_reductions() -> None:
    batch = _batch()
    batch.validate(ExperimentConfig())


def test_future_teacher_accepts_cache_float_but_requires_exact_support_axis() -> None:
    batch = _batch()
    future = FutureSupervision(
        dino_supports=batch.future.dino_supports.to(torch.bfloat16),
        action_sequence=batch.future.action_sequence,
        state_sequence=batch.future.state_sequence,
        offsets=batch.future.offsets,
    )
    future.validate(ExperimentConfig())

    integer_cache = FutureSupervision(
        dino_supports=batch.future.dino_supports.to(torch.int16),
        action_sequence=batch.future.action_sequence,
        state_sequence=batch.future.state_sequence,
        offsets=batch.future.offsets,
    )
    with pytest.raises(TypeError, match="floating cache dtype"):
        integer_cache.validate(ExperimentConfig())

    short = FutureSupervision(
        dino_supports=batch.future.dino_supports[:, :-1],
        action_sequence=batch.future.action_sequence,
        state_sequence=batch.future.state_sequence,
        offsets=batch.future.offsets[:, :-1],
    )
    with pytest.raises(ValueError, match="support count"):
        short.validate(ExperimentConfig())


def test_audit_metadata_is_not_an_online_forward_field() -> None:
    assert "frame_progress" not in {field.name for field in fields(OnlinePolicyInput)}


def test_audit_metadata_requires_one_detached_cpu_row_per_sample() -> None:
    batch = _batch()
    valid = TrainingBatch(
        online=batch.online,
        action_target=batch.action_target,
        future=batch.future,
        audit=AuditMetadata(frame_progress=torch.tensor([0.1, 0.9])),
    )
    valid.validate(ExperimentConfig())
    broken = TrainingBatch(
        online=batch.online,
        action_target=batch.action_target,
        future=batch.future,
        audit=AuditMetadata(frame_progress=torch.tensor([0.1])),
    )
    with pytest.raises(ValueError, match="audit frame_progress"):
        broken.validate(ExperimentConfig())
