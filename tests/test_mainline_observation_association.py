"""Shared frozen law preserves metric and label/observation separation."""

from dataclasses import fields
from unittest.mock import patch

import pytest
import torch
from test_mainline_structural_contracts import _local_facts

from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.observation_association import ObjectObservationAssociation
from clearvla.mainline.model.teacher import ObjectFutureTeacher


def _case(seed=0):
    torch.manual_seed(seed)
    facts, _ = DenseObjectGrounder(hidden=16, content_dim=8, route_dim=4, objects=4, iterations=1)(
        _local_facts(content=8, route=4, hidden=16)
    )
    teacher = ObjectFutureTeacher(
        content_dim=8, key_dim=4, future_time_grid_mode="control_aligned_24_v1"
    )
    return facts, teacher, torch.randn(1, 6, 1, 2, 2, 8), torch.tensor([[4, 8, 12, 16, 20, 24]])


@pytest.mark.parametrize("seed", [0, 71])
@pytest.mark.parametrize("bf16", [False, True])
def test_measurement_and_first_teacher_interval_exact_same_law(seed, bf16):
    facts, t, frames, offsets = _case(seed)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        m = t.measure_observations(facts=facts, observations=frames, relative_offsets=offsets)
        target, _ = t(facts=facts, future_supports=frames, future_offsets=offsets)
    torch.testing.assert_close(
        target.semantic_delta[:, :1],
        (m.successor_per_support - m.current_reference)[:, :1],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        target.transport_mean[:, :1], m.transport_per_support[:, :1], atol=0, rtol=0
    )
    assert all(not getattr(m, f.name).requires_grad for f in fields(m))
    assert m.candidate_posterior.dtype == torch.float32 and all(
        not p.requires_grad for p in t.parameters()
    )


def test_single_observed_support_no_future_grid_requirement():
    facts, t, frames, offsets = _case()
    m = t.measure_observations(
        facts=facts, observations=frames[:, :1], relative_offsets=offsets[:, :1]
    )
    assert m.successor_per_support.shape == (1, 1, 4, 8) and m.candidate_posterior.shape == (
        1,
        1,
        4,
        1,
        2,
        2,
    )
    with pytest.raises(ValueError):
        t(facts=facts, future_supports=frames[:, :1], future_offsets=offsets[:, :1])


def test_observation_entry_never_opens_future_forward():
    facts, t, frames, offsets = _case()
    with patch.object(t, "forward", side_effect=AssertionError("future entry")):
        m = t.measure_observations(
            facts=facts, observations=frames[:, :1], relative_offsets=offsets[:, :1]
        )
    assert torch.isfinite(m.successor_per_support).all()
    assert isinstance(t, ObjectObservationAssociation)
    assert set(t.state_dict()) == {"semantic_content_key.weight", "appearance_content_key.weight"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_source_availability_before_observation_reads(bad):
    facts, t, frames, offsets = _case()
    frames = frames[:, :1].clone().fill_(bad)
    absent = torch.zeros(1, 1, dtype=torch.bool)
    m = t.measure_observations(
        facts=facts, observations=frames, relative_offsets=offsets[:, :1], observed=absent
    )
    assert torch.isfinite(m.successor_per_support).all()
    with pytest.raises(ValueError, match="non-finite"):
        t.measure_observations(
            facts=facts, observations=frames, relative_offsets=offsets[:, :1], observed=~absent
        )
