"""Regressions for the finite-window online history, without a model or assets."""
from __future__ import annotations

import numpy as np
import pytest

from clearvla.simulation.contracts import CAMERA_NAMES, PolicyObservation
from clearvla.simulation.history import (
    EXECUTED_ACTION_OFFSETS,
    STATE_OFFSETS,
    VISUAL_OFFSETS,
    CausalHistory,
)


def observation(time: int) -> PolicyObservation:
    return PolicyObservation(
        rgb={
            "top": np.full((3, 5, 3), time % 256, dtype=np.uint8),
            "wrist": np.full((2, 4, 3), (time + 17) % 256, dtype=np.uint8),
        },
        state=np.arange(7, dtype=np.float32) + time * 11,
        action_state=np.arange(7, dtype=np.float32) - time * 7,
    )


def action(time: int) -> np.ndarray:
    return np.arange(7, dtype=np.float32) + time * 3 + 0.25


def assert_snapshot(history: CausalHistory, time: int, reset_action: np.ndarray) -> None:
    snapshot = history.snapshot()
    assert snapshot.time_index == time == history.time_index
    current = observation(time)
    np.testing.assert_array_equal(snapshot.state, current.state)
    np.testing.assert_array_equal(snapshot.action_state, current.action_state)
    for camera in CAMERA_NAMES:
        expected = np.stack(
            [observation(max(time + offset, 0)).rgb[camera] for offset in VISUAL_OFFSETS]
        )
        np.testing.assert_array_equal(snapshot.rgb_history[camera], expected)
    np.testing.assert_array_equal(
        snapshot.state_history,
        np.stack([observation(max(time + offset, 0)).state for offset in STATE_OFFSETS]),
    )
    np.testing.assert_array_equal(
        snapshot.executed_action_history,
        np.stack([
            reset_action if time + offset < 0 else action(time + offset)
            for offset in EXECUTED_ACTION_OFFSETS
        ]),
    )


@pytest.mark.parametrize("operation", ["time", "snapshot", "append"])
def test_requires_reset(operation: str) -> None:
    history = CausalHistory()
    with pytest.raises(RuntimeError):
        if operation == "time":
            _ = history.time_index
        elif operation == "snapshot":
            history.snapshot()
        else:
            history.append(action(0), observation(1))


@pytest.mark.parametrize("explicit_boundary", [False, True])
def test_every_sparse_window_matches_absolute_timeline(explicit_boundary: bool) -> None:
    history = CausalHistory()
    boundary = np.full(7, -31, dtype=np.float32) if explicit_boundary else observation(0).action_state
    history.reset(observation(0), reset_action=boundary if explicit_boundary else None)
    assert_snapshot(history, 0, boundary)
    # Includes both retention boundaries, image-value wraparound, and late windows.
    for now in range(1, 301):
        history.append(action(now - 1), observation(now))
        assert_snapshot(history, now, boundary)


def test_retention_is_bounded_for_long_episodes() -> None:
    history = CausalHistory()
    history.reset(observation(0))
    for now in range(1, 2001):
        history.append(action(now - 1), observation(now))
    assert history.time_index == 2000
    assert len(history._observations) == 1 - min(*VISUAL_OFFSETS, *STATE_OFFSETS)
    assert len(history._actions) == -min(EXECUTED_ACTION_OFFSETS)
    assert_snapshot(history, 2000, observation(0).action_state)


def test_reset_after_eviction_restarts_padding() -> None:
    history = CausalHistory()
    history.reset(observation(0))
    for now in range(1, 80):
        history.append(action(now - 1), observation(now))
    boundary = np.full(7, 51, dtype=np.float32)
    history.reset(observation(0), reset_action=boundary)
    boundary[:] = 99
    assert len(history._observations) == 1 and len(history._actions) == 0
    assert_snapshot(history, 0, np.full(7, 51, dtype=np.float32))
    history.append(action(0), observation(1))
    assert_snapshot(history, 1, np.full(7, 51, dtype=np.float32))


def test_input_and_snapshot_mutations_do_not_change_retained_evidence() -> None:
    history = CausalHistory()
    initial = observation(0)
    history.reset(initial)
    initial.rgb["top"][:] = 222
    initial.state[:] = 222
    initial.action_state[:] = 222
    executed, following = action(0), observation(1)
    history.append(executed, following)
    executed[:] = 222
    following.rgb["wrist"][:] = 222
    following.state[:] = 222
    snap = history.snapshot()
    snap.rgb_history["top"][:] = 223
    snap.state[:] = 223
    snap.state_history[:] = 223
    snap.executed_action_history[:] = 223
    assert_snapshot(history, 1, observation(0).action_state)


@pytest.mark.parametrize("kind", ["bad_action", "bad_observation", "bad_reset"])
def test_invalid_input_does_not_mutate_timeline(kind: str) -> None:
    history = CausalHistory()
    history.reset(observation(0))
    history.append(action(0), observation(1))
    with pytest.raises(ValueError):
        if kind == "bad_action":
            history.append(np.full(7, np.nan, dtype=np.float32), observation(2))
        elif kind == "bad_observation":
            bad = observation(2)
            bad.state[0] = np.inf
            history.append(action(1), bad)
        else:
            history.reset(observation(0), reset_action=np.zeros(8))
    assert_snapshot(history, 1, observation(0).action_state)


def test_access_cannot_read_future_or_evicted_evidence() -> None:
    history = CausalHistory()
    history.reset(observation(0))
    for now in range(1, 50):
        history.append(action(now - 1), observation(now))
    with pytest.raises(IndexError):
        history._observation_at(50)
    with pytest.raises(IndexError):
        history._action_at(49)
    with pytest.raises(IndexError):
        history._observation_at(1)
    with pytest.raises(IndexError):
        history._action_at(1)
