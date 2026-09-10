from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from clearvla.benchmarks.bridge import PolicyBridge, SmokeZeroPolicy, policy_observation
from clearvla.simulation.contracts import CAMERA_NAMES, PolicyObservation
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.policies import EnvironmentRandomPolicy, HoldPolicy


def _observation(step: int, action_state: np.ndarray | None = None) -> PolicyObservation:
    if action_state is None:
        action_state = np.full(7, step, dtype=np.float32)
    return policy_observation(
        top=np.full((4, 5, 3), step, dtype=np.uint8),
        wrist=np.full((2, 3, 3), step + 20, dtype=np.uint8),
        state=np.full(7, step, dtype=np.float32),
        action_state=action_state,
    )


class _RecordingPolicy:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.calls: list[tuple[Any, str]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, history: Any, instruction: str) -> np.ndarray:
        self.calls.append((history, instruction))
        return np.zeros((24, 7), dtype=np.float32)


def test_policy_observation_preserves_native_camera_shapes_and_copies_history() -> None:
    top = np.zeros((4, 5, 3), dtype=np.uint8)
    wrist = np.ones((2, 3, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    action_state = -state
    observation = policy_observation(
        top=top,
        wrist=wrist,
        state=state,
        action_state=action_state,
    )
    assert tuple(observation.rgb) == CAMERA_NAMES
    assert observation.rgb["top"].shape == (4, 5, 3)
    assert observation.rgb["wrist"].shape == (2, 3, 3)

    history = CausalHistory()
    history.reset(observation)
    top[...] = 255
    wrist[...] = 255
    state[...] = 99
    action_state[...] = 99
    snapshot = history.snapshot()
    assert not np.any(snapshot.rgb_history["top"] == 255)
    assert not np.any(snapshot.rgb_history["wrist"] == 255)
    np.testing.assert_array_equal(snapshot.state, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(
        snapshot.action_state,
        -np.arange(7, dtype=np.float32),
    )


def test_causal_history_uses_only_reset_padding_and_executed_actions() -> None:
    reset_action = np.full(7, -9.0, dtype=np.float32)
    history = CausalHistory()
    history.reset(_observation(0, reset_action), reset_action=reset_action)
    for step in range(1, 9):
        action = np.full(7, step, dtype=np.float32)
        history.append(action, _observation(step, action))

    snapshot = history.snapshot()
    assert snapshot.time_index == 8
    np.testing.assert_array_equal(
        snapshot.rgb_history["top"][:, 0, 0, 0],
        [0, 4, 8],
    )
    np.testing.assert_array_equal(snapshot.state_history[:, 0], [0.0, 4.0, 8.0])
    np.testing.assert_array_equal(
        snapshot.executed_action_history[:, 0],
        [-9.0, -9.0, -9.0, 1.0, 3.0, 5.0, 7.0, 8.0],
    )


def test_policy_bridge_separates_planning_from_observe_only_history_updates() -> None:
    policy = _RecordingPolicy()
    bridge = PolicyBridge(policy)
    assert bridge.health()["history_time_index"] is None

    first = bridge.act(_observation(0), "put the bowl on the plate", reset=True)
    assert first.shape == (24, 7)
    assert policy.reset_calls == 1
    assert len(policy.calls) == 1
    assert policy.calls[-1][0].time_index == 0

    assert bridge.observe(_observation(1)) == 1
    assert len(policy.calls) == 1
    bridge.act(_observation(2), "put the bowl on the plate", reset=False)
    assert len(policy.calls) == 2
    second_snapshot = policy.calls[-1][0]
    assert second_snapshot.time_index == 2
    np.testing.assert_array_equal(
        second_snapshot.executed_action_history[-2:, 0],
        [1.0, 2.0],
    )

    bridge.act(_observation(0), "put the bowl on the plate", reset=True)
    assert policy.reset_calls == 2
    assert policy.calls[-1][0].time_index == 0


def test_bridge_health_marks_smoke_policy_and_observe_protocol_explicitly() -> None:
    health = PolicyBridge(SmokeZeroPolicy()).health()
    assert health["status"] == "ok"
    assert health["mode"] == "smoke-zero"
    assert health["protocol"] == {"version": 2, "observe_only": True}
    assert health["initialized"] is False


def test_dependency_light_smoke_policies_validate_and_emit_full_chunks() -> None:
    history = CausalHistory()
    history.reset(_observation(0))
    snapshot = history.snapshot()
    action = np.arange(7, dtype=np.float32)
    hold = HoldPolicy(action, horizon=3)
    np.testing.assert_array_equal(
        hold.act(snapshot, "move"),
        np.repeat(action[None], 3, axis=0),
    )

    random = EnvironmentRandomPolicy(
        lambda rng: rng.uniform(-1.0, 1.0, size=7),
        seed=123,
        horizon=2,
    )
    first = random.act(snapshot, "move")
    random.reset()
    np.testing.assert_array_equal(first, random.act(snapshot, "move"))
    assert first.shape == (2, 7)

    with pytest.raises(ValueError, match="non-empty"):
        hold.act(snapshot, "  ")


def test_policy_observation_rejects_implicit_camera_or_action_recharting() -> None:
    with pytest.raises(ValueError, match="ordered"):
        PolicyObservation(
            rgb={
                "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
                "top": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            state=np.zeros(7, dtype=np.float32),
            action_state=np.zeros(7, dtype=np.float32),
        ).validate()
    with pytest.raises(ValueError, match=r"shape \[7\]"):
        _observation(0, np.zeros(6, dtype=np.float32))
