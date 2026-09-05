from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from clearvla.action_solvers.flow_solver.spec import ScheduleSpec
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.flow_schedule import (
    DeploymentFlowSchedule,
    resolve_deployment_flow_schedule,
)
from clearvla.mainline.runtime.sampling import (
    SamplingResult,
    _integrate_cache,
    sample_cached_action,
    sample_refined_cached_action_with_cache,
)


@dataclass(frozen=True)
class _Condition:
    interval_action: torch.Tensor
    interval_delta: torch.Tensor


@dataclass(frozen=True)
class _Top:
    bias: float
    action_condition: _Condition


@dataclass(frozen=True)
class _Cache:
    history: SimpleNamespace
    top: _Top

    def validate(self, config: ExperimentConfig) -> None:
        assert self.history.batch == 1
        assert config.runtime.inference_steps == 5


class _Outlet:
    physical_dim = 18

    def sample_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return torch.randn(
            batch,
            24,
            self.physical_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def finalize(
        self,
        value: torch.Tensor,
        action_state: torch.Tensor,
        *,
        codec_gripper_boundary: torch.Tensor | None,
        command_logits: torch.Tensor | None,
    ) -> SimpleNamespace:
        action = value[..., :7]
        return SimpleNamespace(
            deployed_action=action,
            command_logits=command_logits,
            command=None,
            continuous_action=action,
        )

    def sampling_metrics(self, output: SimpleNamespace) -> dict[str, torch.Tensor]:
        return {
            "sampling_gripper_output_mode_code": output.deployed_action.new_zeros(()),
        }

    def world_condition_action_from_deployed(
        self,
        action: torch.Tensor,
        *,
        command: torch.Tensor | None,
    ) -> torch.Tensor:
        return action

    def world_condition_from_horizon_action(
        self,
        action: torch.Tensor,
        action_state: torch.Tensor,
    ) -> _Condition:
        interval_action = action[:, :4]
        return _Condition(
            interval_action=interval_action,
            interval_delta=torch.zeros_like(interval_action),
        )


class _World:
    def __init__(self) -> None:
        self.rebuild_calls = 0

    def refine_deployment_world(
        self,
        top: _Top,
        *,
        action_condition: _Condition,
        collect_diagnostics: bool,
    ) -> tuple[_Top, dict[str, torch.Tensor]]:
        self.rebuild_calls += 1
        return _Top(bias=2.0, action_condition=action_condition), {}


class _Model:
    def __init__(
        self,
        field: Callable[[torch.Tensor, torch.Tensor, _Cache], torch.Tensor],
    ) -> None:
        self.field = field
        self.outlet_adapter = _Outlet()
        self.world = _World()
        self.calls: list[dict[str, object]] = []
        self.training = True

    def eval(self) -> _Model:
        self.training = False
        return self

    def velocity(
        self,
        cache: _Cache,
        *,
        noisy_action_field: torch.Tensor,
        time: torch.Tensor,
        execution_mode: str,
        deployment_fastpath: bool,
        collect_diagnostics: bool,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "cache": cache,
                "state": noisy_action_field.detach().clone(),
                "time": time.detach().clone(),
                "execution_mode": execution_mode,
                "deployment_fastpath": deployment_fastpath,
            }
        )
        velocity = self.field(noisy_action_field, time, cache)
        bottom = SimpleNamespace(
            physical_velocity=velocity,
            motion_logits=torch.zeros(
                noisy_action_field.shape[0],
                2,
                device=noisy_action_field.device,
            ),
            gripper_command_logits=None,
        )
        return SimpleNamespace(bottom=bottom, metrics={})


def _cache() -> _Cache:
    action_state = torch.zeros(1, 7)
    condition = _Condition(
        interval_action=torch.zeros(1, 4, 7),
        interval_delta=torch.zeros(1, 4, 7),
    )
    history = SimpleNamespace(
        batch=1,
        state=torch.zeros(1, 1),
        action_state=action_state,
        codec_gripper_boundary=None,
    )
    return _Cache(history=history, top=_Top(bias=1.0, action_condition=condition))


def _config() -> ExperimentConfig:
    base = ExperimentConfig()
    return replace(base, runtime=replace(base.runtime, compute_dtype="fp32"))


def _time_and_state_field(
    state: torch.Tensor,
    time: torch.Tensor,
    cache: _Cache,
) -> torch.Tensor:
    return 0.125 * state + time[:, None, None] + cache.top.bias


def test_uniform_default_and_explicit_schedule_match_legacy_update_bit_exactly() -> None:
    config = _config()
    initial = torch.linspace(-1.0, 1.0, 24 * 18).reshape(1, 24, 18)
    implicit = sample_cached_action(
        _Model(_time_and_state_field),
        _cache(),
        config,
        initial_physical_noise=initial,
        dtype=torch.float32,
    )
    explicit = sample_cached_action(
        _Model(_time_and_state_field),
        _cache(),
        config,
        initial_physical_noise=initial,
        dtype=torch.float32,
        flow_schedule=DeploymentFlowSchedule.uniform_five(),
    )

    legacy = initial
    legacy_dt = 1.0 / float(config.runtime.inference_steps)
    legacy_times = (
        torch.arange(config.runtime.inference_steps, dtype=torch.float32) * legacy_dt
    )
    for time in legacy_times:
        velocity = 0.125 * legacy + time + 1.0
        legacy = legacy + legacy_dt * velocity

    assert torch.equal(implicit.physical_field, explicit.physical_field)
    assert torch.equal(implicit.physical_field, legacy)
    assert torch.equal(implicit.step_times, legacy_times)
    assert implicit.step_sizes == (legacy_dt,) * 5


def test_q5_schedule_uses_each_physical_interval_delta() -> None:
    config = _config()
    plan = DeploymentFlowSchedule.same_nfe_power_five()
    boundaries = plan.refined.boundaries

    def time_field(
        state: torch.Tensor,
        time: torch.Tensor,
        cache: _Cache,
    ) -> torch.Tensor:
        return time[:, None, None].expand_as(state)

    result = sample_cached_action(
        _Model(time_field),
        _cache(),
        config,
        initial_physical_noise=torch.zeros(1, 24, 18),
        dtype=torch.float32,
        flow_schedule=plan,
    )
    expected = sum(
        left * (right - left)
        for left, right in zip(boundaries, boundaries[1:])
    )
    torch.testing.assert_close(
        result.physical_field,
        torch.full_like(result.physical_field, expected),
    )
    expected_step_sizes = tuple(
        right - left for left, right in zip(boundaries, boundaries[1:])
    )
    assert result.step_sizes == pytest.approx(expected_step_sizes)
    assert expected != pytest.approx(0.2 * sum(boundaries[:-1]))


def test_two_pass_uses_distinct_grids_same_noise_one_w_rebuild_and_no_history() -> None:
    config = _config()
    proposal_boundaries = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)
    refined_boundaries = (0.0, 0.20, 0.35, 0.55, 0.80, 1.0)
    plan = DeploymentFlowSchedule.custom(
        proposal_boundaries,
        refined_boundaries,
        label="different-grids",
    )
    model = _Model(
        lambda state, time, cache: torch.full_like(state, cache.top.bias)
    )
    initial = torch.linspace(-0.5, 0.5, 24 * 18).reshape(1, 24, 18)
    result, refined_cache = sample_refined_cached_action_with_cache(
        model,
        _cache(),
        config,
        initial_physical_noise=initial,
        dtype=torch.float32,
        flow_schedule=plan,
        deployment_fastpath=True,
    )

    assert len(model.calls) == 12
    proposal_calls = model.calls[:6]
    refined_calls = model.calls[6:]
    assert [float(call["time"][0]) for call in proposal_calls] == pytest.approx(
        (*proposal_boundaries[:-1], 1.0)
    )
    assert [float(call["time"][0]) for call in refined_calls] == pytest.approx(
        (*refined_boundaries[:-1], 1.0)
    )
    assert torch.equal(proposal_calls[0]["state"], initial)
    assert torch.equal(refined_calls[0]["state"], initial)
    assert proposal_calls[0]["cache"] is not refined_calls[0]["cache"]
    assert refined_calls[0]["cache"] is refined_cache
    assert model.world.rebuild_calls == 1
    assert all(call["deployment_fastpath"] is True for call in model.calls)
    assert result.flow_schedule_identity == plan.identity
    assert result.flow_schedule_pass_role == "refined"
    assert result.step_sizes == pytest.approx(
        tuple(
            right - left
            for left, right in zip(refined_boundaries, refined_boundaries[1:])
        )
    )


def test_single_pass_defaults_to_refined_grid_and_can_select_proposal_grid() -> None:
    config = _config()
    plan = DeploymentFlowSchedule.custom(
        (0.0, 0.05, 0.15, 0.35, 0.65, 1.0),
        (0.0, 0.20, 0.35, 0.55, 0.80, 1.0),
        label="single-pass-role",
    )
    noise = torch.zeros(1, 24, 18)
    refined = sample_cached_action(
        _Model(_time_and_state_field),
        _cache(),
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
        flow_schedule=plan,
    )
    proposal = sample_cached_action(
        _Model(_time_and_state_field),
        _cache(),
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
        flow_schedule=plan,
        pass_role="proposal",
    )
    configured = replace(
        config,
        runtime=replace(
            config.runtime,
            deployment_flow_schedule=plan.to_dict(),
        ),
    )
    configured_refined = sample_cached_action(
        _Model(_time_and_state_field),
        _cache(),
        configured,
        initial_physical_noise=noise,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        refined.step_times,
        torch.tensor(plan.refined.query_times, dtype=torch.float32),
    )
    torch.testing.assert_close(
        proposal.step_times,
        torch.tensor(plan.proposal.query_times, dtype=torch.float32),
    )
    assert torch.equal(configured_refined.step_times, refined.step_times)
    assert configured_refined.flow_schedule_identity == plan.identity
    assert refined.flow_schedule_pass_role == "refined"
    assert proposal.flow_schedule_pass_role == "proposal"


def test_q5_schedule_round_trip_identity_and_invalid_nodes_fail_closed() -> None:
    q5 = DeploymentFlowSchedule.same_nfe_power_five()
    expected_boundaries = tuple((index / 5.0) ** 1.25 for index in range(5)) + (
        1.0,
    )
    assert q5.proposal.boundaries == expected_boundaries
    assert q5.refined.boundaries == expected_boundaries
    assert q5.candidate_id == "Q5/Q5"
    restored = DeploymentFlowSchedule.from_dict(q5.to_dict())
    assert restored == q5
    assert restored.fingerprint == q5.fingerprint
    assert q5.fingerprint == (
        "3080a10e487ebba47e873303ec05c8e9828efc033094bb194ac57f69b9a0a1e9"
    )
    assert restored.candidate_id == "Q5/Q5"
    assert restored.identity["physical_nfe"] == 10
    assert restored.identity["endpoint_head_calls"] == 2
    assert restored.identity["world_rebuilds"] == 1
    assert restored.identity["candidate_id"] == "Q5/Q5"
    assert resolve_deployment_flow_schedule(q5.to_dict()) == q5

    with pytest.raises(ValueError, match="strictly increasing"):
        DeploymentFlowSchedule.custom((0.0, 0.1, 0.3, 0.3, 0.8, 1.0))
    with pytest.raises(ValueError, match="Q5 exponent"):
        DeploymentFlowSchedule.same_nfe_power_five(0.0)
    with pytest.raises(ValueError, match="finite"):
        DeploymentFlowSchedule.custom((0.0, 0.1, 0.3, float("nan"), 0.8, 1.0))
    with pytest.raises(ValueError, match="exactly 5"):
        DeploymentFlowSchedule(
            proposal=ScheduleSpec.uniform(4),
            refined=ScheduleSpec.uniform(5),
        )
    with pytest.raises(TypeError, match="must be a mapping"):
        DeploymentFlowSchedule.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing fields"):
        DeploymentFlowSchedule.from_dict({"proposal": q5.proposal.to_dict()})
    invalid_version = q5.to_dict()
    invalid_version["schema_version"] = True
    with pytest.raises(TypeError, match="schema_version"):
        DeploymentFlowSchedule.from_dict(invalid_version)
    invalid_version = q5.to_dict()
    invalid_version["schema_version"] = 1.5
    with pytest.raises(TypeError, match="schema_version"):
        DeploymentFlowSchedule.from_dict(invalid_version)
    invalid_nested_version = q5.to_dict()
    invalid_nested_version["proposal"]["schema_version"] = True
    with pytest.raises(TypeError, match="proposal schedule schema_version"):
        DeploymentFlowSchedule.from_dict(invalid_nested_version)
    with pytest.raises(ValueError, match="pass_role"):
        sample_cached_action(
            _Model(_time_and_state_field),
            _cache(),
            _config(),
            initial_physical_noise=torch.zeros(1, 24, 18),
            dtype=torch.float32,
            pass_role="single",  # type: ignore[arg-type]
        )


def test_hot_loop_does_not_import_the_generic_per_node_sync_checks() -> None:
    source = inspect.getsource(_integrate_cache)
    assert "torch.isfinite" not in source
    assert ".item()" not in source
    assert "bool(" not in source


def test_sampling_result_keeps_the_historical_constructor_surface() -> None:
    tensor = torch.zeros(1)
    result = SamplingResult(
        action=tensor,
        physical_field=tensor,
        motion_logits=tensor,
        initial_physical_noise=tensor,
        step_times=tensor,
        metrics={},
    )
    assert result.step_sizes == ()
    assert result.flow_schedule_identity is None
    assert result.flow_schedule_pass_role == "legacy_unspecified"


def test_fastpath_switch_does_not_change_schedule_or_numerics() -> None:
    config = _config()
    plan = DeploymentFlowSchedule.same_nfe_power_five()
    noise = torch.linspace(-0.25, 0.25, 24 * 18).reshape(1, 24, 18)
    regular_model = _Model(_time_and_state_field)
    fast_model = _Model(_time_and_state_field)
    regular = sample_cached_action(
        regular_model,
        _cache(),
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
        flow_schedule=plan,
        deployment_fastpath=False,
    )
    fast = sample_cached_action(
        fast_model,
        _cache(),
        config,
        initial_physical_noise=noise,
        dtype=torch.float32,
        flow_schedule=plan,
        deployment_fastpath=True,
    )
    assert torch.equal(regular.physical_field, fast.physical_field)
    assert torch.equal(regular.step_times, fast.step_times)
    assert regular.step_sizes == fast.step_sizes
    assert all(call["deployment_fastpath"] is False for call in regular_model.calls)
    assert all(call["deployment_fastpath"] is True for call in fast_model.calls)
