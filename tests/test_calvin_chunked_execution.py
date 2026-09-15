from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from clearvla.benchmarks.bridge import PolicyBridge, SmokeZeroPolicy, policy_observation
from clearvla.benchmarks.calvin_eval import (
    CalvinBridgeModel,
    execute_calvin_action,
    validate_bridge_health,
)


def _policy_observation(step: int, action: np.ndarray):
    return policy_observation(
        top=np.full((3, 4, 3), step, dtype=np.uint8),
        wrist=np.full((2, 3, 3), step + 20, dtype=np.uint8),
        state=np.full(7, step, dtype=np.float32),
        action_state=np.asarray(action, dtype=np.float32),
    )


def _calvin_observation(step: int) -> dict[str, object]:
    return {
        "rgb_obs": {
            "rgb_static": np.full((3, 4, 3), step, dtype=np.uint8),
            "rgb_gripper": np.full((2, 3, 3), step + 20, dtype=np.uint8),
        },
        "robot_obs": np.full(15, step, dtype=np.float32),
    }


class _CountingPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.snapshots: list[Any] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, history, instruction: str) -> np.ndarray:
        self.snapshots.append(history)
        return np.zeros((24, 7), dtype=np.float32)


class _FakeClient:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks
        self.act_calls: list[tuple[Any, str, bool]] = []
        self.observe_calls: list[Any] = []

    def act(self, observation, instruction: str, *, reset: bool) -> np.ndarray:
        self.act_calls.append((observation, instruction, reset))
        return self.chunks[len(self.act_calls) - 1].copy()

    def observe(self, observation) -> int:
        self.observe_calls.append(observation)
        return len(self.observe_calls)


def test_observe_only_advances_physical_history_without_policy_inference() -> None:
    policy = _CountingPolicy()
    bridge = PolicyBridge(policy)
    reset_action = np.zeros(7, dtype=np.float32)

    bridge.act(_policy_observation(0, reset_action), "open drawer", reset=True)
    for step in range(1, 8):
        executed = np.full(7, step, dtype=np.float32)
        assert bridge.observe(_policy_observation(step, executed)) == step

    final_executed = np.full(7, 8, dtype=np.float32)
    bridge.act(
        _policy_observation(8, final_executed),
        "open drawer",
        reset=False,
    )

    assert policy.reset_calls == 1
    assert len(policy.snapshots) == 2
    snapshot = policy.snapshots[-1]
    assert snapshot.time_index == 8
    np.testing.assert_array_equal(snapshot.rgb_history["top"][:, 0, 0, 0], [0, 4, 8])
    np.testing.assert_array_equal(snapshot.state_history[:, 0], [0.0, 4.0, 8.0])
    np.testing.assert_array_equal(
        snapshot.executed_action_history[:, 0],
        [0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0],
    )


def test_observe_only_requires_initialized_history() -> None:
    bridge = PolicyBridge(_CountingPolicy())
    with pytest.raises(RuntimeError, match="initialized"):
        bridge.observe(_policy_observation(1, np.ones(7, dtype=np.float32)))


def test_chunked_execution_requires_observe_only_protocol_preflight() -> None:
    health = PolicyBridge(SmokeZeroPolicy()).health()
    validate_bridge_health(
        health,
        allow_smoke_policy=True,
        require_observe_only=True,
    )
    without_protocol = dict(health)
    without_protocol.pop("protocol")
    with pytest.raises(ValueError, match="observe-only"):
        validate_bridge_health(
            without_protocol,
            allow_smoke_policy=True,
            require_observe_only=True,
        )


def test_calvin_model_records_chunk_rows_and_intermediate_observations() -> None:
    first = np.stack(
        [np.full(7, 0.1 * (row + 1), dtype=np.float32) for row in range(24)]
    )
    first[:, 6] = np.where(np.arange(24) % 2 == 0, -0.25, 0.25)
    second = np.full((24, 7), 0.75, dtype=np.float32)
    client = _FakeClient([first, second])
    model = CalvinBridgeModel(client)  # type: ignore[arg-type]

    model.plan(_calvin_observation(0), "open drawer")
    for row in range(4):
        expected = execute_calvin_action(first[row])
        np.testing.assert_allclose(model.execute_planned_row(row), expected)
        if row < 3:
            assert model.observe(_calvin_observation(row + 1)) == row + 1
            np.testing.assert_allclose(
                client.observe_calls[-1].action_state,
                expected,
            )

    model.plan(_calvin_observation(4), "open drawer")
    model.execute_planned_row(0)
    arrays = model.action_arrays()

    assert [call[2] for call in client.act_calls] == [True, False]
    np.testing.assert_array_equal(arrays["executed_plan_index"], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(arrays["executed_chunk_row"], [0, 1, 2, 3, 0])
    np.testing.assert_array_equal(arrays["observed_history_time_index"], [1, 2, 3])
    np.testing.assert_array_equal(arrays["observed_after_executed_steps"], [1, 2, 3])
    np.testing.assert_allclose(arrays["raw_executed"][:4], first[:4])
    np.testing.assert_allclose(arrays["raw_executed"][4], second[0])
    assert model.action_audit()["planning_decisions"] == 2
    assert model.action_audit()["steps"] == 5


def test_calvin_step_keeps_one_row_receding_horizon_compatibility() -> None:
    first = np.arange(24 * 7, dtype=np.float32).reshape(24, 7) / 100.0
    second = -first
    client = _FakeClient([first, second])
    model = CalvinBridgeModel(client)  # type: ignore[arg-type]

    model.step(_calvin_observation(0), "open drawer")
    model.step(_calvin_observation(1), "open drawer")
    arrays = model.action_arrays()

    np.testing.assert_allclose(arrays["raw_executed"], arrays["raw_first"])
    np.testing.assert_array_equal(arrays["executed_plan_index"], [0, 1])
    np.testing.assert_array_equal(arrays["executed_chunk_row"], [0, 0])
    assert arrays["observed_history_time_index"].shape == (0,)
    assert [call[2] for call in client.act_calls] == [True, False]


def test_calvin_step_chunks_chain_style_calls_without_duplicate_boundary_observe() -> None:
    first = np.stack(
        [np.full(7, 0.1 * (row + 1), dtype=np.float32) for row in range(24)]
    )
    second = np.full((24, 7), 0.75, dtype=np.float32)
    client = _FakeClient([first, second])
    model = CalvinBridgeModel(client, execute_rows=4)  # type: ignore[arg-type]

    for step in range(5):
        model.step(_calvin_observation(step), "open drawer")

    arrays = model.action_arrays()
    np.testing.assert_array_equal(arrays["executed_plan_index"], [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(arrays["executed_chunk_row"], [0, 1, 2, 3, 0])
    np.testing.assert_array_equal(arrays["observed_history_time_index"], [1, 2, 3])
    np.testing.assert_array_equal(arrays["observed_after_executed_steps"], [1, 2, 3])
    assert len(client.observe_calls) == 3
    assert [call[2] for call in client.act_calls] == [True, False]
    assert model.action_audit()["planning_decisions"] == 2
