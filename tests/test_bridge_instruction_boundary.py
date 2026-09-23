"""Real bridge/history and wire protocol; no simulator or trained skill claims."""

from __future__ import annotations

import io
import urllib.request
from typing import Any

import numpy as np
import pytest
from test_calvin_chunked_execution import _policy_observation

from clearvla.benchmarks.bridge import PolicyBridge, RemotePolicyClient, _event_flag


class AnchoredPolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.starts: list[str] = []
        self.anchors: list[int] = []
        self.snapshots: list[Any] = []
        self.anchor: int | None = None
        self.rng = np.random.default_rng(5)

    def reset(self) -> None:
        self.resets += 1
        self.anchor = None
        self.rng = np.random.default_rng(5)

    def begin_instruction(self, text: str) -> None:
        self.starts.append(text)
        self.anchor = None

    def act(self, history: Any, instruction: str) -> np.ndarray:
        del instruction
        if self.anchor is None:
            self.anchor = int(history.time_index)
        self.anchors.append(self.anchor)
        self.snapshots.append(history)
        return self.rng.standard_normal((24, 7)).astype(np.float32)


def test_instruction_restart_keeps_history_previous_command_and_rng() -> None:
    policy = AnchoredPolicy()
    bridge = PolicyBridge(policy)
    zero = np.zeros(7, dtype=np.float32)
    first = bridge.act(_policy_observation(0, zero), "push red", reset=True)
    a = np.full(7, 0.25, dtype=np.float32)
    bridge.observe(_policy_observation(1, a))
    second = bridge.act(_policy_observation(2, a), "push red", reset=False, begin_instruction=True)
    bridge.act(_policy_observation(3, a), "push red", reset=False)
    assert policy.resets == 1 and policy.starts == ["push red"]
    assert policy.anchors == [0, 2, 2]
    assert bridge.history.time_index == 3
    last = policy.snapshots[-1]
    np.testing.assert_array_equal(last.previous_state, np.full(7, 2, dtype=np.float32))
    np.testing.assert_array_equal(last.executed_action_history[-1], a)
    assert not np.array_equal(first, second)  # instruction reset did not reseed
    restarted = bridge.act(_policy_observation(0, zero), "push red", reset=True)
    assert bridge.history.time_index == 0 and policy.resets == 2
    np.testing.assert_array_equal(first, restarted)


def test_unsupported_event_and_empty_instruction_do_not_mutate_history() -> None:
    from test_calvin_chunked_execution import _CountingPolicy

    bridge = PolicyBridge(_CountingPolicy())
    obs = _policy_observation(0, np.zeros(7, dtype=np.float32))
    bridge.act(obs, "push", reset=True)
    with pytest.raises(ValueError, match="instruction boundary"):
        bridge.act(obs, "push", reset=False, begin_instruction=True)
    with pytest.raises(ValueError, match="non-empty"):
        bridge.act(obs, "  ", reset=False)
    assert bridge.history.time_index == 0
    assert bridge.health()["protocol"]["instruction_boundary"] is False


@pytest.mark.parametrize(
    "value",
    [
        np.array([2], dtype=np.uint8),
        np.array([np.nan]),
        np.array(["false"]),
        np.array([1, 0], dtype=np.uint8),
        np.array(1, dtype=np.uint8),
    ],
)
def test_wire_rejects_ambiguous_event(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="bridge reset"):
        _event_flag(value, name="reset")


@pytest.mark.parametrize("value", [False, True])
def test_wire_roundtrip_and_capability(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    captured: dict[str, np.ndarray] = {}
    output = io.BytesIO()
    np.save(output, np.ones((24, 7), dtype=np.float32), allow_pickle=False)

    def transport(request: urllib.request.Request, **kwargs: Any) -> io.BytesIO:
        del kwargs
        assert request.data is not None
        assert isinstance(request.data, (bytes, bytearray, memoryview))
        with np.load(io.BytesIO(request.data), allow_pickle=False) as data:
            captured.update({key: data[key].copy() for key in data})
        return io.BytesIO(output.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", transport)
    out = RemotePolicyClient().act(
        _policy_observation(0, np.zeros(7, dtype=np.float32)),
        "push red",
        reset=False,
        begin_instruction=value,
    )
    assert out.shape == (24, 7)
    assert _event_flag(captured["begin_instruction"], name="begin_instruction") is value
    assert not _event_flag(captured["reset"], name="reset")
    health = PolicyBridge(AnchoredPolicy()).health()
    assert health["protocol"]["version"] == 3
    assert health["protocol"]["instruction_boundary"] is True
