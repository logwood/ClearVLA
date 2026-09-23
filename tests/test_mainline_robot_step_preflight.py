"""Adjacent response supervision agrees with its causal command source."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_operation_expectation import _batch

from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.robot_execution import ExecutedRobotStep
from clearvla.mainline.training.engine import validate_finite_training_batch


def _with_step(batch: TrainingBatch, step: ExecutedRobotStep) -> TrainingBatch:
    return replace(
        batch,
        online=replace(
            batch.online, history=replace(batch.online.history, executed_robot_step=step)
        ),
    )


@pytest.mark.parametrize("offsets", [[0, -1, 0], [-4, -1, 0], [-1, 0, 1]])
def test_wrong_temporal_transfer_rejected(offsets: list[int]) -> None:
    batch = _batch()
    step = batch.online.history.executed_robot_step
    assert step is not None
    bad = replace(step, offsets=step.offsets.new_tensor(offsets)[None].expand_as(step.offsets))
    with pytest.raises(ValueError, match="one-step"):
        validate_finite_training_batch(_with_step(batch, bad))


def test_command_disagreement_with_same_t_minus_one_is_rejected() -> None:
    batch = _batch()
    step = batch.online.history.executed_robot_step
    assert step is not None
    with pytest.raises(ValueError, match="executed command"):
        validate_finite_training_batch(
            _with_step(batch, replace(step, command=step.command + 0.25))
        )


def test_claimed_transfer_cannot_disagree_with_source_history_support() -> None:
    batch = _batch()
    h = batch.online.history
    assert h.timing is not None
    mask = h.timing.action_executed.clone()
    mask[:, -1] = False
    changed = replace(
        batch,
        online=replace(
            batch.online, history=replace(h, timing=replace(h.timing, action_executed=mask))
        ),
    )
    with pytest.raises(ValueError, match="execution support"):
        validate_finite_training_batch(changed)


def test_unobserved_transfer_cannot_have_observed_offsets() -> None:
    batch = _batch(0)
    step = batch.online.history.executed_robot_step
    assert step is not None
    bad = replace(step, offsets=step.offsets.new_tensor([[-1, -1, 0]]).expand_as(step.offsets))
    with pytest.raises(ValueError, match="one-step"):
        validate_finite_training_batch(_with_step(batch, bad))


def test_reset_unknown_payload_remains_legal_and_unchanged() -> None:
    batch = _batch(0)
    step = batch.online.history.executed_robot_step
    assert step is not None
    bad = replace(
        step,
        previous_state=torch.full_like(step.previous_state, float("nan")),
        command=torch.full_like(step.command, float("nan")),
    )
    validate_finite_training_batch(_with_step(batch, bad))
    assert torch.isnan(bad.command).all() and torch.isnan(bad.previous_state).all()


def test_consistent_observed_and_history_dropped_transfers_remain_legal() -> None:
    batch = _batch()
    validate_finite_training_batch(batch)
    h = batch.online.history
    assert h.timing is not None and h.executed_robot_step is not None
    keep = torch.zeros(batch.online.batch, dtype=torch.bool)
    changed = replace(
        h,
        timing=h.timing.without_actions(keep),
        executed_robot_step=h.executed_robot_step.without_actions(keep),
    )
    validate_finite_training_batch(replace(batch, online=replace(batch.online, history=changed)))


@pytest.mark.parametrize("source", ["history", "response"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_command_keeps_numeric_owner_before_agreement(
    source: str, value: float,
) -> None:
    batch = _batch()
    history = batch.online.history
    step = history.executed_robot_step
    assert step is not None
    if source == "history":
        commands = history.executed_action_history.clone()
        commands[:, -1, 0] = value
        changed = replace(history, executed_action_history=commands)
        pattern = "non-finite.*online.executed_history"
    else:
        command = step.command.clone()
        command[:, 0] = value
        changed = replace(history, executed_robot_step=replace(step, command=command))
        pattern = "observed robot step payload must be finite"
    bad = replace(batch, online=replace(batch.online, history=changed))
    with pytest.raises(ValueError, match=pattern):
        validate_finite_training_batch(bad)
