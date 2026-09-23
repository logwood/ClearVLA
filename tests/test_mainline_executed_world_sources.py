"""Real loader/history/adapter parity with explicit synthetic visual transport."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_executed_world import _config
from test_mainline_instruction_reference import _adapter, _dataset, _Tokens
from test_mainline_timed_history import _observation
from torch.utils.data import default_collate

from clearvla.benchmarks.bridge import PolicyBridge, SmokeZeroPolicy
from clearvla.mainline.data.dataset import CachedTokenPolicyWindowDataset
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.executed_world import EXECUTED_WORLD_FEEDBACK
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory


def _source():
    data = _dataset(0)
    data.config = replace(
        data.config,
        world_horizon=24,
        robot_feedback_mode="one_step_proprioceptive_v1",
        world_feedback_mode=EXECUTED_WORLD_FEEDBACK,
    )
    data.config.validate()
    return data


def _history(data, end):
    h = CausalHistory(executed_world=True)
    e = data.episodes[0]
    for step in range(end + 1):
        obs = PolicyObservation(
            rgb={c: np.full((32, 32, 3), step, dtype=np.uint8) for c in data.camera_names},
            state=e.states_raw[step],
            action_state=e.action_states_raw[step],
        )
        if step == 0:
            h.reset(obs)
        else:
            h.append(e.actions_raw[step - 1], obs)
    return h


def _policy(monkeypatch):
    p, _, calls = _adapter(monkeypatch)
    data = _source()
    monkeypatch.setattr(
        p,
        "bundle",
        SimpleNamespace(
            config=_config(),
            model=None,
            action_normalizer=data.action_normalizer,
            state_normalizer=data.state_normalizer,
            language=p.bundle.language,
        ),
    )

    def encode(rgb, _):
        calls.append(1)
        images = np.stack([rgb[c] for c in data.camera_names], axis=1)
        tokens = (
            torch.tensor(images[:, :, 0, 0, 0])
            .float()[:, :, None, None]
            .expand(3, 2, 64, 16)
            .clone()
        )
        return tokens, images.copy()

    monkeypatch.setattr(p, "encoder", SimpleNamespace(encode=encode))
    return p, data, calls


@pytest.mark.parametrize("center", [0, 1, 3, 4, 5, 8, 12, 16, 24, 40])
def test_online_offline_executed_source_identical(monkeypatch, center):
    policy, data, calls = _policy(monkeypatch)
    if center:
        policy.act_with_input(_history(data, 0).snapshot(), "push the block")
    _, online = policy.act_with_input(_history(data, center).snapshot(), "push the block")
    store = _Tokens()
    dataset = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, store))
    raw = dataset[center]
    batch = to_training_batch(
        default_collate([raw]),
        goal=GoalTemplate(torch.zeros(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        config=_config(),
        device=torch.device("cpu"),
    )
    batch.validate(_config())
    off = batch.online.history.executed_world_window
    on = online.history.executed_world_window
    assert off is not None and on is not None
    for k in (
        "dino_history",
        "raw_rgb",
        "state",
        "action_state",
        "commands",
        "observed",
        "visual_offsets",
    ):
        torch.testing.assert_close(getattr(off, k), getattr(on, k), atol=0, rtol=0)
    assert on.observed.item() == (center >= 4)
    assert len(calls) == (1 if center == 0 else 2 if center < 4 else 3)
    assert store.keys is not None and store.keys[-4].tolist() == [0, 0]
    if center >= 4:
        assert store.keys[-3:, 1].tolist() == [max(center - 12, 0), max(center - 8, 0), center - 4]
        expected = torch.from_numpy(
            data.action_normalizer.encode(data.episodes[0].actions_raw[center - 4 : center])
        )
        torch.testing.assert_close(on.commands[0], expected, atol=0, rtol=0)
    else:
        assert not on.commands.count_nonzero() and not on.dino_history.count_nonzero()
    for online_source, window in ((online, on), (batch.online, off)):
        ClearVLAMainlinePolicy._admit_executed_world_source(
            cast(Any, SimpleNamespace(config=_config())), online_source, window
        )


def test_extended_history_bounded_and_detached_snapshots():
    h = CausalHistory(executed_world=True)
    h.reset(_observation(0))
    for step in range(1, 1001):
        h.append(np.full(7, step - 1, dtype=np.float32), _observation(step))
    assert len(h._observations) == 13 and len(h._actions) == 24
    w = h.snapshot().executed_world
    assert w is not None and w.anchor_index == 996
    np.testing.assert_array_equal(w.commands[:, 0], [996, 997, 998, 999])
    np.testing.assert_array_equal(w.rgb_history["top"][:, 0, 0, 0], np.array([988, 992, 996]) % 255)
    w.state[:] = 90000
    w.commands[:] = -20
    w.rgb_history["top"][:] = 0
    n = h.snapshot().executed_world
    assert n is not None and n.state[0] == 996 and n.commands[0, 0] == 996
    assert n.rgb_history["top"][0, 0, 0, 0] == 988 % 255
    h.reset(_observation(0))
    n = h.snapshot().executed_world
    assert n is not None and not n.observed and not n.commands.any()


def test_bridge_selects_extended_without_legacy_changes(monkeypatch):
    p, _, _ = _policy(monkeypatch)
    b = PolicyBridge(p)
    old = PolicyBridge(SmokeZeroPolicy())
    assert b.history.executed_world and b.history._observations.maxlen == 13
    assert not old.history.executed_world and old.history._observations.maxlen == 9


def test_instruction_restart_preserves_physical_window_and_rng(monkeypatch):
    p, d, _ = _policy(monkeypatch)
    snap = _history(d, 20).snapshot()
    _, one = p.act_with_input(snap, "push the block")
    rng = p._generator.get_state().clone()
    p.begin_instruction("push the block")
    assert torch.equal(rng, p._generator.get_state())
    _, two = p.act_with_input(snap, "push the block")
    a, b = one.history.executed_world_window, two.history.executed_world_window
    assert a is not None and b is not None
    for k in ("commands", "state", "dino_history", "visual_offsets"):
        torch.testing.assert_close(getattr(a, k), getattr(b, k), atol=0, rtol=0)
    assert two.instruction_reference is not None and two.instruction_reference.age_steps.item() == 0


def test_missing_window_rejected_before_anchor_encoder_rng(monkeypatch):
    p, d, calls = _policy(monkeypatch)
    s = replace(_history(d, 16).snapshot(), executed_world=None)
    rng = p._generator.get_state().clone()
    with pytest.raises(ValueError, match="extended confirmed-command"):
        p.act_with_input(s, "push the block")
    assert (
        not calls and p._instruction_anchor is None and torch.equal(rng, p._generator.get_state())
    )


@pytest.mark.parametrize("which", ["commands", "state", "images", "clock"])
def test_inconsistent_raw_history_rejected(which):
    d = _source()
    s = _history(d, 20).snapshot()
    w = s.executed_world
    assert w is not None
    if which == "commands":
        v = w.commands.copy()
        v[0] += 0.1
        w = replace(w, commands=v)
    elif which == "state":
        w = replace(w, state=w.state + 0.1)
    elif which == "images":
        images = {c: v.copy() for c, v in w.rgb_history.items()}
        images["top"][1] += 1
        w = replace(w, rgb_history=images)
    else:
        w = replace(w, visual_offsets=np.array([-8, -3, 0], dtype=np.int64))
    with pytest.raises(ValueError):
        replace(s, executed_world=w).validate()
