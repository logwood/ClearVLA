"""Preflight owns supported source values, including every online codec boundary."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_operation_expectation import _batch
from test_mainline_policy import _batch as legacy_batch
from test_mainline_policy import _config as legacy_config

from clearvla.mainline.training.engine import validate_finite_training_batch


@pytest.mark.parametrize("field", ["action_state", "codec_gripper_boundary", "state"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_observable_codec_boundaries_are_in_finite_audit(field: str, value: float) -> None:
    batch = _batch()
    history = batch.online.history
    bad_value = torch.full_like(getattr(history, field), value)
    bad = replace(batch, online=replace(batch.online, history=replace(history, **{field: bad_value})))
    with pytest.raises(ValueError, match="online." + field):
        validate_finite_training_batch(bad)


@pytest.mark.parametrize("kind", ["state", "command"])
def test_reset_padding_is_not_misclassified_as_supported_data(kind: str) -> None:
    batch = _batch(0)
    history = batch.online.history
    assert history.timing is not None
    field, mask = (("state_history", history.timing.state_observed) if kind == "state"
                   else ("executed_action_history", history.timing.action_executed))
    poison = torch.where(mask[..., None], getattr(history, field), float("nan"))
    before = poison.clone()
    changed = replace(batch, online=replace(batch.online, history=replace(history, **{field: poison})))
    validate_finite_training_batch(changed)
    torch.testing.assert_close(poison, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ["state", "command"])
def test_real_history_nonfinite_values_are_not_quarantined(kind: str) -> None:
    batch = _batch()
    history = batch.online.history
    field = "state_history" if kind == "state" else "executed_action_history"
    value = getattr(history, field).clone()
    value[:, -1] = float("nan")
    bad = replace(batch, online=replace(batch.online, history=replace(history, **{field: value})))
    with pytest.raises(ValueError, match="non-finite"):
        validate_finite_training_batch(bad)


@pytest.mark.parametrize("tail", [False, True])
def test_policy_and_world_share_recorded_prefix_under_both_support_contracts(tail: bool) -> None:
    batch = _batch() if tail else legacy_batch(legacy_config())
    source = batch.future.action_sequence.clone()
    source[:, :24] = batch.action_target.normalized
    batch = replace(batch, future=replace(batch.future, action_sequence=source))
    validate_finite_training_batch(batch)
    wrong = source.clone()
    wrong[:, 0, 0] += 1.0
    bad = replace(batch, future=replace(batch.future, action_sequence=wrong))
    with pytest.raises(ValueError, match="policy and world supervision disagree"):
        validate_finite_training_batch(bad)


def test_bad_source_time_is_rejected_before_cross_plane_comparison() -> None:
    batch = legacy_batch(legacy_config())
    offsets = batch.future.offsets.clone()
    offsets[:, 5] = 25
    with pytest.raises(ValueError, match="physical time grid"):
        validate_finite_training_batch(replace(batch, future=replace(batch.future, offsets=offsets)))
