"""Controller receipts must own recorded labels/history after native step."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from test_stackcube_v2 import descriptor

from clearvla.simulation.contracts import (
    ExecutedCommand,
    PolicyObservation,
    ResetResult,
    StepResult,
)
from clearvla.simulation.dataset import EpisodeRecorder
from clearvla.simulation.rollout import run_episode


def _obs(command: np.ndarray) -> PolicyObservation:
    return PolicyObservation(
        {"top": np.zeros((8, 8, 3), np.uint8), "wrist": np.zeros((8, 8, 3), np.uint8)},
        np.zeros(7, np.float32),
        np.asarray(command, np.float32).copy(),
    )


def test_receipt_copies_both_native_buffers() -> None:
    a = np.ones(7, np.float32)
    b = a * 0.5
    r = ExecutedCommand(a, b)
    a[:] = 5
    b[:] = 4
    np.testing.assert_array_equal(r.submitted, np.ones(7))
    np.testing.assert_array_equal(r.applied, np.full(7, 0.5))
    with pytest.raises(ValueError):
        r.applied[0] = 8


@pytest.mark.parametrize("which", ["submitted", "applied"])
@pytest.mark.parametrize("bad", ["shape", "nan", "inf"])
def test_invalid_receipts_rejected(which: str, bad: str) -> None:
    kw = dict(submitted=np.zeros(7, np.float32), applied=np.zeros(7, np.float32))
    kw[which] = (
        np.zeros(6, dtype=np.float32)
        if bad == "shape"
        else np.full(7, np.nan if bad == "nan" else np.inf, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        ExecutedCommand(**kw)


def test_receipt_allows_documented_internal_change_only() -> None:
    request = np.ones(7, np.float32)
    applied = request * 0.3
    result = StepResult(
        _obs(applied), 0.0, False, False, command_receipt=ExecutedCommand(request, applied)
    )
    resolved = result.executed_command(request)
    np.testing.assert_array_equal(resolved, applied)
    assert result.command_receipt is not None
    resolved[:] = 77
    np.testing.assert_array_equal(result.command_receipt.applied, applied)
    with pytest.raises(ValueError, match="different submitted"):
        result.executed_command(request + 1)
    with pytest.raises(ValueError, match="acknowledged"):
        replace(result, observation=_obs(request)).validate()


def test_legacy_adapter_fails_closed_on_unacknowledged_change() -> None:
    req = np.ones(7, np.float32)
    np.testing.assert_array_equal(
        StepResult(_obs(req), 0.0, False, False).executed_command(req), req
    )
    with pytest.raises(ValueError, match="without a receipt"):
        StepResult(_obs(req * 0.5), 0.0, False, False).executed_command(req)


def test_recorder_cannot_store_requested_label_instead_of_ack(tmp_path: Path) -> None:
    request = np.ones(7, np.float32)
    ack = request * 0.5
    result = StepResult(_obs(ack), 0.0, False, False, command_receipt=ExecutedCommand(request, ack))
    rec = EpisodeRecorder(
        tmp_path,
        descriptor=descriptor(),
        episode_id="unit",
        seed=0,
        instruction="push",
        collector="test",
    )
    with pytest.raises(ValueError, match="acknowledged"):
        rec.append(_obs(np.zeros(7)), request, result)
    assert not rec._actions
    rec.append(_obs(np.zeros(7)), ack, result)
    result.observation.action_state[:] = 99
    np.testing.assert_array_equal(rec._steps[0].observation.action_state, ack)


class _Env:
    descriptor = descriptor()

    def __init__(self, receipt: bool = True, bad_clip: bool = False):
        self.steps = 0
        self.receipt = receipt
        self.bad_clip = bad_clip

    def reset(self, *, seed):
        return ResetResult(_obs(np.zeros(7, np.float32)))

    def hold_action(self, observation):
        return np.zeros(7, np.float32)

    def clip_action(self, action):
        if self.bad_clip:
            return np.full(7, np.nan)
        # Deliberately mutates its input to test proposal isolation.
        action[:] = np.clip(action, -1, 1)
        return action

    def step(self, action):
        request = action.copy()
        applied = request * 0.25
        action[:] = 99  # adapter buffer ownership must not rewrite submitted trace
        self.steps += 1
        return StepResult(
            _obs(applied),
            0.0,
            False,
            self.steps >= 3,
            command_receipt=ExecutedCommand(request, applied) if self.receipt else None,
        )

    def action_bounds(self):
        return -np.ones(7, np.float32), np.ones(7, np.float32)

    def sample_action(self, rng):
        return rng.uniform(-1, 1, 7).astype(np.float32)

    def close(self):
        pass


class _Policy:
    def __init__(self):
        self.histories = []
        self.chunk = np.full((24, 7), 2, np.float32)

    def reset(self):
        self.histories = []

    def act(self, history, instruction):
        self.histories.append(history)
        return self.chunk


def test_real_rollout_history_trace_and_HDF5_all_use_acknowledged_row(tmp_path: Path) -> None:
    p = _Policy()
    env = _Env()
    report = run_episode(
        env,
        p,
        seed=0,
        instruction="push",
        max_steps=3,
        record_dir=tmp_path,
        episode_id="receipt",
        record_video=False,
    )
    assert env.steps == 3
    np.testing.assert_array_equal(p.chunk, np.full((24, 7), 2))
    for h in p.histories[1:]:
        np.testing.assert_array_equal(h.executed_action_history[-1], np.full(7, 0.25))
        np.testing.assert_array_equal(h.action_state, np.full(7, 0.25))
        assert h.previous_state is not None
    import json

    with h5py.File(tmp_path / "receipt.hdf5") as f:
        action_data = f["action"]
        metrics_data = f["sim/metrics_json"]
        assert isinstance(action_data, h5py.Dataset) and isinstance(metrics_data, h5py.Dataset)
        np.testing.assert_array_equal(action_data[:], np.full((3, 7), 0.25))
        metrics = [json.loads(x) for x in metrics_data]
    assert metrics[0]["telemetry_command_receipt"] is True
    np.testing.assert_array_equal(metrics[0]["telemetry_submitted_native_action"], np.ones(7))
    np.testing.assert_array_equal(metrics[0]["telemetry_policy_raw_action_chunk"], p.chunk)
    assert report["steps"] == 3


@pytest.mark.parametrize("bad_clip", [True, False])
def test_rollout_rejects_bad_command_before_history_commit(bad_clip: bool) -> None:
    env = _Env(receipt=False, bad_clip=bad_clip)
    p = _Policy()
    with pytest.raises(ValueError):
        run_episode(
            env, p, seed=0, instruction="push", max_steps=3, record_dir=None, episode_id="x"
        )
    assert len(p.histories) == 1 and env.steps == (0 if bad_clip else 1)


@pytest.mark.parametrize("changed_at", ["clip", "receipt"])
def test_residual_runner_rejects_remapped_action_before_replay(changed_at: str) -> None:
    from test_residual_sac import Environment, Reader, config

    from clearvla.rl.runner import run_episode as run_residual_episode
    from clearvla.rl.sac import ResidualSAC

    class RemapEnvironment(Environment):
        def clip_action(self, action):
            if changed_at == "clip":
                action[:] *= 0.25
            return action

        def step(self, action):
            requested = action.copy()
            applied = action * 0.25
            self.actions.append(applied)
            return StepResult(
                self.observation(),
                0.0,
                False,
                False,
                command_receipt=ExecutedCommand(requested, applied),
            )

    env = RemapEnvironment()
    learner = ResidualSAC(8, config())
    reader = Reader()
    learner.act = lambda features, deterministic=False: np.zeros(7, np.float32)
    with pytest.raises(ValueError, match="residual action map"):
        run_residual_episode(env, reader, learner, seed=0, step_budget=2, mode="train")
    assert learner.replay.size == 0 and reader.calls == 1
    assert len(env.actions) == (0 if changed_at == "clip" else 1)


def test_real_maniskill_adapter_freezes_boundary_before_backend_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv

    env = object.__new__(ManiSkillStackCubeEnv)
    accepted = []

    def step(command):
        accepted.append(command.copy())
        command[:] = 99
        return {}, 0.0, False, False, {}

    env._env = SimpleNamespace(
        action_space=SimpleNamespace(
            low=-np.ones((1, 7), np.float32), high=np.ones((1, 7), np.float32), shape=(1, 7)
        ),
        step=step,
    )
    monkeypatch.setattr(env, "_observation", lambda value: _obs(env._last_action))
    monkeypatch.setattr(env, "_evaluation_metrics", lambda *a, **kw: {})
    request = np.full(7, 2.0, np.float32)
    result = env.step(request)
    np.testing.assert_array_equal(request, np.full(7, 2))
    np.testing.assert_array_equal(accepted[0], np.ones((1, 7)))
    np.testing.assert_array_equal(result.executed_command(request), np.ones(7))
    np.testing.assert_array_equal(env._last_action, np.ones(7))
