"""Actual-engine heldout probe: numerical fixtures, not robot results."""
from __future__ import annotations

import copy
import gc
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from test_mainline_annotation_goal import _config
from test_mainline_state_features import _model_engine

from clearvla.mainline.runtime.learning_probe import (
    probe_fixed_batch_learning,
    probe_heldout_learning,
)
from clearvla.mainline.runtime.qualification import synthetic_batch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_mainline_runtime.py"


def _inputs():
    c = _config()
    train, normalizer = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"), seed=901)
    val, _ = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"), seed=902)
    return train, val, normalizer


@pytest.fixture(scope="module")
def actual():
    train, val, normalizer = _inputs()
    torch.manual_seed(871)
    model, engine = _model_engine(_config())
    model.configure_action_normalizer(normalizer)
    return model, engine, train, val


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_rejects_bad_budget_before_either_model_pass(actual, budget):
    _, engine, train, val = actual
    with patch.object(engine, "eval_step", side_effect=AssertionError("must not execute")):
        with pytest.raises(ValueError, match="updates"):
            probe_heldout_learning(engine, train, val, updates=cast(int, budget))


def test_same_online_source_is_not_accepted_as_validation(actual):
    _, engine, train, _ = actual
    for val in (train, replace(train)):
        with pytest.raises(ValueError, match="reuse"):
            probe_heldout_learning(engine, train, val, updates=1)


def test_train_updates_never_receive_validation_and_trace_owns_completed_evaluations(actual):
    _, engine, train, val = actual
    snapshots: list[dict[str, Any]] = []
    start = engine.global_step
    with patch.object(engine, "train_step", wraps=engine.train_step) as update:
        trace = probe_heldout_learning(engine, train, val, updates=2, on_progress=snapshots.append)
    assert all(call.args[0] is train for call in update.call_args_list)
    assert update.call_count == 2
    assert trace["completed_updates"] == 2 and trace["validation_optimizer_updates"] == 0
    assert not trace["test_split_used"]
    for split in ("train", "val"):
        assert trace["before"][split]["completed_optimizer_updates"] == start
        assert trace["after"][split]["completed_optimizer_updates"] == start + 2
        assert trace["after"][split]["annotated_goal"]
    assert snapshots[0]["before"] == {}
    assert snapshots[-1]["status"] == "completed"
    json.dumps(trace, allow_nan=False)


def test_validation_failure_after_updates_cannot_become_success(actual):
    _, engine, train, val = actual
    start = engine.global_step
    original = engine.eval_step
    snapshots: list[dict[str, Any]] = []
    calls = 0
    def fail_final_validation(batch, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("injected validation failure")
        return original(batch, **kwargs)
    with patch.object(engine, "eval_step", side_effect=fail_final_validation):
        with pytest.raises(RuntimeError, match="injected"):
            probe_heldout_learning(engine, train, val, updates=1, on_progress=snapshots.append)
    assert engine.global_step == start + 1
    assert snapshots[-1]["status"] == "failed"
    assert "train" in snapshots[-1]["after"] and "val" not in snapshots[-1]["after"]
    assert snapshots[-1]["completed_updates"] == 1


def _assert_equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_equal_tree(x, y)
    else:
        assert a == b


def test_adding_validation_does_not_change_the_actual_training_trajectory():
    train, val, normalizer = _inputs()
    snapshots = []
    for heldout in (False, True):
        torch.manual_seed(9003)
        model, engine = _model_engine(_config())
        model.configure_action_normalizer(normalizer)
        engine.train_flow_generator = torch.Generator().manual_seed(143)
        engine.train_condition_generator = torch.Generator().manual_seed(144)
        if heldout:
            probe_heldout_learning(engine, train, val, updates=2, evaluation_seed=919)
        else:
            probe_fixed_batch_learning(engine, train, updates=2, evaluation_seed=919)
        snapshots.append(copy.deepcopy({
            "model": model.state_dict(), "optimizer": engine.optimizer.state_dict(),
            "schedule": engine.schedule.state_dict(), "global": torch.get_rng_state(),
            "flow": engine.train_flow_generator.get_state(), "condition": engine.train_condition_generator.get_state(),
        }))
        del model, engine
        gc.collect()
    _assert_equal_tree(snapshots[0], snapshots[1])


@pytest.mark.parametrize("args", [
    ["--operation", "fit-heldout", "--updates", "1"],
    ["--operation", "fit-heldout", "--source", "real"],
    ["--operation", "fit-heldout", "--source", "real", "--updates", "0"],
    ["--operation", "fit-heldout", "--source", "real", "--updates", "1", "--inference-step", "1200"],
    ["--operation", "inspect", "--train-indices", "0"],
    ["--operation", "fit-heldout", "--source", "synthetic", "--updates", "1", "--val-indices", "0"],
    ["--operation", "fit-heldout", "--source", "real", "--updates", "1", "--val-indices", "0", "0", "--batch-size", "2"],
])
def test_cli_rejects_ambiguous_or_invalid_probe_before_creating_output(args, tmp_path):
    out = tmp_path / "probe"
    result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(out), *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and not out.exists()


def _command(tmp_path: Path, source: str):
    c = _config()
    c = replace(c, data=replace(c.data, raw_hdf5_root=str(tmp_path / "absent")), optimizer=replace(c.optimizer, warmup_steps=7))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(c.as_dict()))
    return [sys.executable, str(SCRIPT), "--config", str(cfg), "--output", str(tmp_path / "result"), "--operation", "fit-heldout", "--source", source, "--updates", "2", "--raw-side", "32", "--allow-runtime-mismatch", "--timeout", "600"]


def test_missing_real_paths_block_before_model_no_synthetic_fallback(tmp_path):
    result = subprocess.run(_command(tmp_path, "real"), capture_output=True, text=True, timeout=30)
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert result.returncode == 2 and report["status"] == "blocked"
    assert report["arguments"]["operation"] == "fit-heldout"
    assert "heldout_learning" not in report and "model_constructed" not in report


def test_cli_two_updates_and_independent_synthetic_validation_are_explicit(tmp_path):
    result = subprocess.run(_command(tmp_path, "synthetic"), capture_output=True, text=True, timeout=620,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert result.returncode == 0, (result.stderr, report)
    assert report["model_completed"] and not report["formal_training"]
    assert report["heldout_admission"]["status"] == "not-applicable"
    assert report["probe_schedule"]["state"]["curve"]["warmup_steps"] == 7
    assert report["probe_schedule"]["warmup_exceeds_budget"]
    assert report["heldout_learning"]["completed_updates"] == 2
    assert report["heldout_learning"]["validation_optimizer_updates"] == 0
    assert report["batch_support"]["val"]["endpoint_visual_observed_fraction"] == 1
    assert not list((tmp_path / "result").glob("*.pt"))
