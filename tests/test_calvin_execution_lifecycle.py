"""Official loop-shaped calls over real bridge/history; fake physics only."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from test_bridge_instruction_boundary import AnchoredPolicy
from test_calvin_chunked_execution import _calvin_observation

from clearvla.benchmarks.bridge import PolicyBridge, RemotePolicyClient
from clearvla.benchmarks.calvin_eval import (
    CalvinBridgeModel,
    CalvinExecutionEnvironment,
    validate_bridge_health,
)
from clearvla.simulation.contracts import ExecutedCommand, PolicyObservation


class LocalClient:
    def __init__(self, bridge: PolicyBridge) -> None:
        self.bridge = bridge
        self.requests: list[tuple[bool, bool]] = []

    def act(
        self,
        observation: PolicyObservation,
        instruction: str,
        *,
        reset: bool,
        begin_instruction: bool = False,
    ) -> np.ndarray:
        self.requests.append((reset, begin_instruction))
        return self.bridge.act(
            observation, instruction, reset=reset, begin_instruction=begin_instruction
        )

    def observe(self, observation: PolicyObservation) -> int:
        return self.bridge.observe(observation)


class Physics:
    def __init__(self) -> None:
        self.t = 0
        self.resets = 0
        self.steps = 0
        self.mode = "ok"
        self.buffer: np.ndarray | None = None

    def reset(self, **kwargs: Any) -> None:
        del kwargs
        self.t = 0
        self.resets += 1

    def get_obs(self) -> dict[str, object]:
        return _calvin_observation(self.t)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict[str, Any]]:
        self.steps += 1
        if self.mode == "raise":
            raise RuntimeError("physics failed after admission")
        requested = action.copy()
        info: dict[str, Any] = {"privileged_scene": "not-for-policy"}
        if self.mode in {"mutate", "receipt"}:
            action[:6] *= 0.5
        if self.mode == "receipt":
            info["clearvla_command_receipt"] = ExecutedCommand(requested, action)
        if self.mode == "wrong_receipt":
            info["clearvla_command_receipt"] = ExecutedCommand(np.zeros(7), action)
        if self.mode == "invalid_applied":
            info["clearvla_command_receipt"] = ExecutedCommand(action, np.full(7, 2.0))
        self.buffer = action
        self.t += 1
        return self.get_obs(), 0.0, False, info


def setup(rows: int = 1):
    policy = AnchoredPolicy()
    bridge = PolicyBridge(policy)
    client = LocalClient(bridge)
    model = CalvinBridgeModel(
        cast(RemotePolicyClient, client), execute_rows=rows, require_step_receipt=True
    )
    physics = Physics()
    env = CalvinExecutionEnvironment(physics, model)
    return policy, bridge, client, model, physics, env


@pytest.mark.parametrize("same_text", [False, True])
@pytest.mark.parametrize("rows", [1, 4])
def test_subtasks_preserve_physical_history_and_restart_anchor(same_text: bool, rows: int) -> None:
    policy, bridge, client, model, physics, env = setup(rows)
    env.reset()
    for task in ("push red", "push red" if same_text else "push blue"):
        model.reset()  # official rollout invokes this without resetting env
        for _ in range(3):
            action = model.step(env.get_obs(), task)
            assert model.action_audit()["steps"] == physics.t  # only acknowledged
            env.step(action)
    assert physics.resets == 1 and policy.resets == 1
    assert bridge.history.time_index == 5
    assert policy.starts == ["push red" if same_text else "push blue"]
    assert policy.anchors[0] == 0 and policy.anchors[-1] == 3
    assert sum(int(reset) for reset, _ in client.requests) == 1
    assert model.action_audit()["steps"] == 6
    snapshot = policy.snapshots[-1]
    assert snapshot.previous_state is not None
    np.testing.assert_array_equal(
        snapshot.executed_action_history[-1], model.executed_actions[snapshot.time_index - 1]
    )
    env.reset()
    model.reset()
    model.step(env.get_obs(), "push red")
    assert policy.resets == 2 and bridge.history.time_index == 0
    assert policy.snapshots[-1].previous_state is None


def test_language_change_discards_old_unexecuted_chunk_rows() -> None:
    policy, bridge, _, model, _, env = setup(4)
    env.reset()
    env.step(model.step(env.get_obs(), "red"))
    env.step(model.step(env.get_obs(), "blue"))
    assert len(model.raw_chunks) == 2
    assert model.executed_chunk_rows == [0, 0]
    assert bridge.history.time_index == 1 and policy.starts == ["blue"]


def test_execution_record_occurs_only_after_successful_step() -> None:
    _, _, _, model, physics, env = setup()
    env.reset()
    action = model.step(env.get_obs(), "red")
    before = action.copy()
    assert model.action_arrays()["executed"].shape == (0, 7)
    assert model.action_audit()["steps"] == 0
    env.step(action)
    assert physics.buffer is not None
    physics.buffer[:] = -7  # later backend buffer reuse cannot rewrite history
    action[:] = -8
    np.testing.assert_array_equal(model.executed_actions[0], before)
    np.testing.assert_array_equal(model._previous_action, before)
    with pytest.raises(RuntimeError, match="pending policy command"):
        model.confirm_execution(ExecutedCommand(before, before))


@pytest.mark.parametrize("mode", ["raise", "mutate", "wrong_receipt", "invalid_applied"])
def test_unconfirmed_transition_is_not_recorded_and_requires_physical_reset(mode: str) -> None:
    _, bridge, _, model, physics, env = setup()
    env.reset()
    action = model.step(env.get_obs(), "red")
    physics.mode = mode
    with pytest.raises((RuntimeError, ValueError)):
        env.step(action)
    assert model.executed_actions == [] and bridge.history.time_index == 0
    np.testing.assert_array_equal(model._previous_action, np.zeros(7))
    with pytest.raises(RuntimeError):
        model.step(env.get_obs(), "red")
    with pytest.raises(RuntimeError, match="environment reset"):
        model.reset()  # a new instruction does not erase uncertain execution
    physics.mode = "ok"
    env.reset()
    env.step(model.step(env.get_obs(), "red"))
    assert len(model.executed_actions) == 1


def test_explicit_applied_command_owns_next_history_not_the_proposal() -> None:
    policy, _, _, model, physics, env = setup()
    env.reset()
    proposal = model.step(env.get_obs(), "red")
    physics.mode = "receipt"
    env.step(proposal)
    applied = proposal.copy()
    applied[:6] *= 0.5
    model.step(env.get_obs(), "red")
    np.testing.assert_array_equal(model.executed_actions[0], applied)
    np.testing.assert_array_equal(policy.snapshots[-1].action_state, applied)
    np.testing.assert_array_equal(policy.snapshots[-1].executed_action_history[-1], applied)
    assert not np.array_equal(proposal, applied)


def test_wrong_submission_is_rejected_before_physics_but_valid_retry_is_allowed() -> None:
    _, _, _, model, physics, env = setup()
    env.reset()
    action = model.step(env.get_obs(), "red")
    with pytest.raises(ValueError, match="different submitted"):
        env.step(action * 0)
    assert physics.steps == 0
    env.step(action)
    assert physics.steps == 1


def test_chunk_feedback_is_once_per_physical_step() -> None:
    _, bridge, _, model, _, env = setup(4)
    env.reset()
    model.plan(env.get_obs(), "red")
    env.step(model.execute_planned_row(0))
    with pytest.raises(RuntimeError, match="new observation"):
        model.execute_planned_row(1)
    model.observe(env.get_obs())
    with pytest.raises(RuntimeError, match="duplicate observation"):
        model.observe(env.get_obs())
    with pytest.raises(ValueError, match="once and in order"):
        model.execute_planned_row(0)
    env.step(model.execute_planned_row(1))
    assert bridge.history.time_index == 1 and len(model.executed_actions) == 2


def test_health_refuses_old_bridge_for_instruction_aware_evaluation() -> None:
    health = {
        "status": "ok",
        "mode": "smoke-zero",
        "protocol": {"version": 2, "observe_only": True},
    }
    with pytest.raises(ValueError, match="instruction_boundary"):
        validate_bridge_health(health, allow_smoke_policy=True, require_instruction_boundary=True)


def test_wrapper_cannot_silently_use_assumed_execution_mode() -> None:
    _, _, client, _, physics, _ = setup()
    model = CalvinBridgeModel(cast(RemotePolicyClient, client))
    with pytest.raises(ValueError, match="receipt-enabled"):
        CalvinExecutionEnvironment(physics, model)


@pytest.mark.parametrize("value", [True, 1.5, 0, 25])
def test_execute_count_is_an_exact_control_count(value: Any) -> None:
    _, _, client, _, _, _ = setup()
    with pytest.raises(ValueError, match="execute_rows"):
        CalvinBridgeModel(cast(RemotePolicyClient, client), execute_rows=value)


@pytest.mark.parametrize("value", [True, 0.5])
def test_row_number_is_not_silently_truncated(value: Any) -> None:
    _, _, _, model, _, env = setup()
    env.reset()
    model.plan(env.get_obs(), "red")
    with pytest.raises(ValueError, match="integer control"):
        model.execute_planned_row(value)


def test_plan_failure_cannot_duplicate_a_remote_history_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bridge, client, model, _, env = setup()
    env.reset()
    env.step(model.step(env.get_obs(), "red"))
    original = client.act

    def bad(*args: Any, **kwargs: Any) -> np.ndarray:
        original(*args, **kwargs)
        return np.empty((0, 7), dtype=np.float32)

    monkeypatch.setattr(client, "act", bad)
    with pytest.raises(ValueError, match="empty"):
        model.step(env.get_obs(), "red")
    assert bridge.history.time_index == 1
    with pytest.raises(RuntimeError, match="environment reset"):
        model.step(env.get_obs(), "red")
    assert bridge.history.time_index == 1


def test_failed_env_reset_is_not_a_successful_timeline_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, model, physics, env = setup()
    env.reset()
    env.step(model.step(env.get_obs(), "red"))

    def bad(**kwargs: Any) -> None:
        raise RuntimeError("reset failed")

    monkeypatch.setattr(physics, "reset", bad)
    with pytest.raises(RuntimeError, match="reset failed"):
        env.reset()
    with pytest.raises(RuntimeError, match="environment reset"):
        model.step(env.get_obs(), "red")


def test_bridge_task_event_reaches_actual_checkpoint_input_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import torch
    from test_mainline_instruction_reference import _adapter

    import clearvla.simulation.clearvla_policy as module

    policy, _, _ = _adapter(monkeypatch)
    captured: list[Any] = []

    def sample(_model: Any, online: Any, *_args: Any, **_kwargs: Any) -> Any:
        captured.append(online)
        return SimpleNamespace(action=torch.zeros(1, 24, 7), gripper_command=torch.ones(1, 24))

    monkeypatch.setattr(module, "sample_action", sample)
    bridge = PolicyBridge(policy)
    client = LocalClient(bridge)
    model = CalvinBridgeModel(cast(RemotePolicyClient, client), require_step_receipt=True)
    env = CalvinExecutionEnvironment(Physics(), model)
    env.reset()
    for _ in range(2):
        env.step(model.step(env.get_obs(), "push block"))
    model.reset()
    env.step(model.step(env.get_obs(), "push block"))
    assert [x.instruction_reference.age_steps.item() for x in captured] == [0, 1, 0]
    assert bridge.history.time_index == 2
    assert captured[-1].history.timing.action_executed[0, -1]
    assert captured[-1].history.state.shape == (1, 10)
    assert captured[-1].history.action_state.shape == (1, 7)
