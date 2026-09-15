from __future__ import annotations

import torch
from test_mainline_bspine import _arm_only_spine
from test_mainline_flow_schedule import _cache, _config, _Model, _time_and_state_field

from clearvla.mainline.runtime import sampling
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from clearvla.tools.replay_bspine_schedule import (
    Q5,
    UNIFORM,
    _tensor_digest,
    detail_removed,
    integrate_cache,
    schedule_override,
)


def test_replay_integrator_matches_accepted_uniform_and_q5() -> None:
    noise = torch.randn(1, 24, 18)
    for boundaries, schedule in (
        (UNIFORM, DeploymentFlowSchedule.uniform_five()),
        (Q5, DeploymentFlowSchedule.same_nfe_power_five()),
    ):
        actual = integrate_cache(
            _Model(_time_and_state_field), _cache(), _config(), generator=None,
            initial_physical_noise=noise, collect_diagnostics=False,
            dtype=torch.float32, boundaries=boundaries,
        )
        expected = sampling.sample_cached_action(
            _Model(_time_and_state_field), _cache(), _config(),
            initial_physical_noise=noise, flow_schedule=schedule,
        )
        assert torch.equal(actual.action, expected.action)
        assert torch.equal(actual.physical_field, expected.physical_field)
        assert torch.equal(actual.initial_physical_noise, noise)


def test_detail_ablation_preserves_weights_and_restores_hooks() -> None:
    spine = _arm_only_spine(hidden=8)
    with torch.no_grad():
        for parameter in spine.parameters():
            parameter.normal_(std=0.01)
    physical = torch.randn(2, 24, 18)
    before = _tensor_digest(spine.state_dict())
    original, _ = spine(physical)
    controls, _, _ = spine.decompose(physical)
    expected = torch.einsum(
        "tk,bkh->bth", spine.synthesis,
        spine.coarse_lifts["arm_absolute"](controls[..., :6])
        + spine.coarse_lifts["arm_delta"](controls[..., 6:]),
    )
    with detail_removed(spine, True):
        coarse, _ = spine(physical)
    assert torch.equal(coarse, expected)
    restored, _ = spine(physical)
    assert torch.equal(restored, original)
    assert _tensor_digest(spine.state_dict()) == before
    assert all(not module._forward_hooks for module in spine.detail_lifts.values())


def test_replay_schedule_hook_is_restored_after_exception() -> None:
    original = sampling._integrate_cache
    try:
        with schedule_override(sampling, Q5):
            assert sampling._integrate_cache is not original
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert sampling._integrate_cache is original
