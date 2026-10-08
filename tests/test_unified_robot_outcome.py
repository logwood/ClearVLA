from dataclasses import replace

import pytest
import torch

from clearvla.mainline.model.robot_execution import RobotExecutionObserver
from clearvla.mainline.robot_execution import ExecutedRobotStep


def test_prediction_error_is_not_used_as_the_only_observed_outcome():
    m = RobotExecutionObserver(state_dim=3, action_dim=2, hidden=8)
    with torch.no_grad():
        for p in m.response.parameters():
            p.zero_()
    step = ExecutedRobotStep(
        torch.zeros(1, 3), torch.ones(1, 2), torch.tensor([True]), torch.tensor([[-1, -1, 0]])
    )
    unchanged, loss = m.observe(step, torch.zeros(1, 3))
    assert loss == 0 and unchanged.innovation.count_nonzero() == 0
    moved, loss2 = m.observe(step, torch.ones(1, 3))
    assert moved.observed_delta is not None and moved.predicted_delta is not None
    torch.testing.assert_close(moved.innovation, moved.observed_delta - moved.predicted_delta)
    assert moved.observed_delta.count_nonzero() == 3 and not moved.observed_delta.requires_grad
    moved.validate(batch=1, state_dim=3, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="retained together"):
        replace(moved, predicted_delta=None).validate(
            batch=1, state_dim=3, device=torch.device("cpu")
        )
    loss2.backward()
    head = m.response[-1]
    assert isinstance(head, torch.nn.Linear) and head.bias is not None
    assert head.bias.grad is not None and head.bias.grad.abs().sum() > 0


def test_missing_robot_step_quarantines_all_three_quantities():
    m = RobotExecutionObserver(state_dim=3, action_dim=2, hidden=8)
    step = ExecutedRobotStep(
        torch.full((1, 3), float("nan")),
        torch.full((1, 2), float("nan")),
        torch.tensor([False]),
        torch.zeros(1, 3, dtype=torch.long),
    )
    feedback, loss = m.observe(step, torch.full((1, 3), float("nan")))
    for value in (feedback.observed_delta, feedback.predicted_delta, feedback.innovation):
        assert value is not None
        assert torch.isfinite(value).all() and value.count_nonzero() == 0
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
