from __future__ import annotations

import pytest
import torch

from clearvla.action_solvers.flow_solver import (
    PanelRecord,
    ReplayAttachment,
    ScheduleSpec,
    SolverSpec,
    candidate_by_name,
    euler_update,
    heun_update,
    integrate,
    rk4_update,
    run_two_pass,
)


def _spec(method: str = "euler", steps: int = 5) -> SolverSpec:
    return SolverSpec(
        schedule=ScheduleSpec.uniform(steps),
        method=method,  # type: ignore[arg-type]
    )


def test_euler_uses_left_endpoints_and_keeps_endpoint_head_separate() -> None:
    state = torch.zeros(2, 3)
    calls: list[tuple[float, tuple[int, ...]]] = []

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        calls.append((float(time[0]), tuple(value.shape)))
        return torch.ones_like(value)

    endpoint_calls: list[float] = []

    def endpoint(value: torch.Tensor, time: torch.Tensor, cache: object) -> str:
        endpoint_calls.append(float(time[0]))
        return "head"

    trace = integrate(
        state,
        field,
        _spec(),
        object(),
        endpoint_head=endpoint,
        retain_trajectory=True,
        retain_velocities=True,
    )
    torch.testing.assert_close(trace.final_state, torch.ones_like(state))
    assert [time for time, _ in calls] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert endpoint_calls == [1.0]
    assert trace.nfe == 5
    assert trace.endpoint_calls == 1
    assert trace.total_dynamic_calls == 6
    assert trace.states is not None and len(trace.states) == 6
    assert trace.effective_velocities is not None and len(trace.effective_velocities) == 5


def test_dense_jump_uses_the_terminal_jump_without_changing_nfe() -> None:
    state = torch.zeros(1, 1)
    times: list[float] = []

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        times.append(float(time[0]))
        return torch.ones_like(value)

    trace = integrate(
        state,
        field,
        SolverSpec(schedule=ScheduleSpec.dense_jump(5, 0.5)),
        object(),
    )
    torch.testing.assert_close(trace.final_state, torch.ones_like(state))
    assert times == [0.0, 0.125, 0.25, 0.375, 0.5]
    assert trace.nfe == 5
    assert trace.endpoint_calls == 0


def test_heun_counts_two_velocity_evaluations_per_interval_and_improves_linear_field() -> None:
    state = torch.ones(1, 1)
    calls: list[float] = []

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        calls.append(float(time[0]))
        return value

    trace = integrate(
        state,
        field,
        _spec("heun", 4),
        object(),
        retain_velocities=True,
        record_corrections=True,
    )
    expected = (1.0 + 0.25 + 0.5 * 0.25**2) ** 4
    assert abs(float(trace.final_state.item()) - expected) < 1e-6
    assert trace.nfe == 8
    assert len(calls) == 8
    assert trace.correction_rms is not None and len(trace.correction_rms) == 4


def test_updates_preserve_gradients_and_reject_bad_field_outputs() -> None:
    state = torch.randn(2, 3, requires_grad=True)

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return 2.0 * value

    trace = integrate(state, field, _spec("euler", 3), object())
    trace.final_state.square().mean().backward()
    assert state.grad is not None
    assert bool(torch.isfinite(state.grad).all())

    def wrong_shape(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return value[..., :1]

    with pytest.raises(ValueError, match="shape"):
        integrate(torch.zeros(2, 3), wrong_shape, _spec(), object())

    def nonfinite(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.full_like(value, float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite"):
        integrate(torch.zeros(2, 3), nonfinite, _spec(), object())


def test_interval_helpers_validate_bounds() -> None:
    state = torch.zeros(1, 2)

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.ones_like(value)

    next_state, velocity = euler_update(state, field, object(), t0=0.2, t1=0.4)
    torch.testing.assert_close(next_state, torch.full_like(state, 0.2))
    torch.testing.assert_close(velocity, torch.ones_like(state))
    next_state, start, end = heun_update(state, field, object(), t0=0.2, t1=0.4)
    torch.testing.assert_close(next_state, torch.full_like(state, 0.2))
    torch.testing.assert_close(start, end)
    with pytest.raises(ValueError, match="0 <= t0"):
        euler_update(state, field, object(), t0=0.4, t1=0.2)


def test_rk4_is_a_four_nfe_dense_reference_for_a_linear_field() -> None:
    state = torch.ones(1, 1)

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return value

    next_state, k1, k2, k3, k4 = rk4_update(
        state,
        field,
        object(),
        t0=0.0,
        t1=1.0,
    )
    expected = 1.0 + (1.0 + 2.0 * 1.5 + 2.0 * 1.75 + 2.75) / 6.0
    assert abs(float(next_state.item()) - expected) < 3e-7
    for value, target in zip((k1, k2, k3, k4), (1.0, 1.5, 1.75, 2.75)):
        torch.testing.assert_close(value, torch.full_like(value, target))


def test_step_doubling_cost_is_recorded_separately_from_deployed_nfe() -> None:
    from clearvla.action_solvers.flow_solver import euler_step_doubling

    state = torch.zeros(1, 1)

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.ones_like(value)

    report = euler_step_doubling(state, field, object(), t0=0.0, t1=0.5)
    assert report.diagnostic_nfe == 3


def test_higher_order_and_finer_uniform_oracles_converge_on_exponential_flow() -> None:
    state = torch.ones(1, 1, dtype=torch.float64)

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return value

    exact = torch.full_like(state, float(torch.exp(torch.tensor(1.0))))
    e5 = integrate(state, field, _spec("euler", 5), object()).final_state
    e10 = integrate(state, field, _spec("euler", 10), object()).final_state
    h5 = integrate(state, field, _spec("heun", 5), object()).final_state
    rk4 = integrate(state, field, _spec("rk4", 5), object()).final_state
    errors = [
        float((candidate - exact).abs().item())
        for candidate in (e5, e10, h5, rk4)
    ]
    assert errors[1] < errors[0]
    assert errors[2] < errors[0]
    assert errors[3] < errors[2]
    assert errors[3] < 1.0e-4


def test_panel_row_exposes_plan_and_attachment_fingerprints() -> None:
    candidate = candidate_by_name("E5/E5")

    def field(value: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.zeros_like(value)

    attachment = ReplayAttachment.from_sections(identity={"sample_id": "s"})
    result = run_two_pass(
        torch.zeros(1, 1),
        field,
        candidate.plan,
        {},
        lambda state, endpoint, cache: {},
        endpoint_head=lambda state, time, cache: None,
    )
    row = PanelRecord(candidate, result, attachment).row()
    assert row["two_pass_solver_fingerprint"] == candidate.plan.fingerprint
    assert row["replay_attachment_schema_version"] == 1
    assert row["replay_attachment_fingerprint"] == attachment.fingerprint
