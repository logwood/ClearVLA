"""Actual engine/CLI replay qualification; synthetic inputs are explicitly scoped."""

from __future__ import annotations

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
from test_mainline_annotation_goal import _batch, _config
from test_mainline_state_features import _model_engine

from clearvla.mainline.runtime.learning_probe import (
    evaluate_fixed_batch,
    probe_fixed_batch_learning,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_mainline_runtime.py"


@pytest.fixture(scope="module")
def actual():
    torch.manual_seed(19311)
    batch, normalizer = _batch()
    model, engine = _model_engine(_config())
    model.configure_action_normalizer(normalizer)
    return model, engine, batch


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_budget_rejected_before_model_work(bad: object, actual) -> None:
    _, engine, batch = actual
    before = engine.global_step
    with patch.object(engine, "eval_step", side_effect=AssertionError("should not evaluate")):
        with pytest.raises(ValueError, match="updates"):
            probe_fixed_batch_learning(engine, batch, updates=cast(int, bad))
    assert engine.global_step == before


@pytest.mark.parametrize("bad", [-1, True, 2**64, 2.5])
def test_invalid_seed_rejected_before_model_work(bad: object, actual) -> None:
    _, engine, batch = actual
    with patch.object(engine, "eval_step", side_effect=AssertionError("should not evaluate")):
        with pytest.raises(ValueError, match="seed"):
            probe_fixed_batch_learning(engine, batch, updates=1, evaluation_seed=cast(int, bad))


def test_repeatable_evaluation_preserves_global_and_training_random_streams(actual) -> None:
    model, engine, batch = actual
    with (
        patch.object(engine, "train_flow_generator", torch.Generator().manual_seed(19)),
        patch.object(engine, "train_condition_generator", torch.Generator().manual_seed(20)),
    ):
        before = torch.random.get_rng_state().clone()
        assert engine.train_flow_generator is not None and engine.train_condition_generator is not None
        flow = engine.train_flow_generator.get_state().clone()
        condition = engine.train_condition_generator.get_state().clone()
        modes = [m.training for m in model.modules()]
        a = evaluate_fixed_batch(engine, batch, seed=371)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert torch.equal(engine.train_flow_generator.get_state(), flow)
        assert torch.equal(engine.train_condition_generator.get_state(), condition)
        assert modes == [m.training for m in model.modules()]
        # Ambient random draws must not change the fixed-noise measurement.
        torch.rand(17)
        before_second = torch.random.get_rng_state().clone()
        b = evaluate_fixed_batch(engine, batch, seed=371)
        assert a == b
        assert torch.equal(torch.random.get_rng_state(), before_second)


def test_trace_uses_actual_updates_and_keeps_immutable_progress_snapshots(actual) -> None:
    _, engine, batch = actual
    progress: list[dict[str, Any]] = []
    start = engine.global_step
    trace = probe_fixed_batch_learning(engine, batch, updates=2, on_progress=progress.append)
    assert trace["status"] == "completed" and trace["completed_updates"] == 2
    assert engine.global_step == start + 2
    assert len(trace["steps"]) == 2
    assert trace["before"]["completed_optimizer_updates"] == start
    assert trace["after"]["completed_optimizer_updates"] == start + 2
    assert trace["before"]["annotated_goal"] and trace["after"]["annotated_goal"]
    assert trace["loss_change"] == trace["after"]["loss"] - trace["before"]["loss"]
    assert progress[0]["steps"] == [] and progress[0]["status"] == "running"
    assert progress[-1]["status"] == "completed"
    json.dumps(trace, allow_nan=False)


def test_later_failure_retains_one_actual_completed_update(actual) -> None:
    _, engine, batch = actual
    progress: list[dict[str, Any]] = []
    original = engine.train_step
    count = 0
    start = engine.global_step

    def fail_second(*args: Any, **kwargs: Any):
        nonlocal count
        count += 1
        if count == 2:
            raise FloatingPointError("injected failure before second update")
        return original(*args, **kwargs)

    with patch.object(engine, "train_step", side_effect=fail_second):
        with pytest.raises(FloatingPointError, match="injected failure"):
            probe_fixed_batch_learning(engine, batch, updates=2, on_progress=progress.append)
    assert engine.global_step == start + 1
    assert progress[-1]["status"] == "failed"
    assert progress[-1]["completed_updates"] == 1
    assert len(progress[-1]["steps"]) == 1
    assert "after" not in progress[-1]


@pytest.mark.parametrize(
    "args",
    [
        ["--operation", "fit-batch", "--updates", "1"],
        ["--operation", "fit-batch", "--source", "synthetic"],
        ["--operation", "fit-batch", "--source", "real", "--updates", "0"],
        ["--operation", "inspect", "--updates", "2"],
        ["--operation", "fit-batch", "--source", "synthetic", "--updates", "1", "--inference-step", "333"],
    ],
)
def test_cli_requires_explicit_source_budget_and_real_training_clock(
    args: list[str], tmp_path: Path
) -> None:
    output = tmp_path / "probe"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), *args],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert not output.exists()


def _command(tmp_path: Path, source: str) -> list[str]:
    config = _config()
    # An explicit test config; the CLI must not silently shorten production warmup.
    config = replace(config, optimizer=replace(config.optimizer, warmup_steps=7))
    if source == "real":
        config = replace(config, data=replace(config.data, raw_hdf5_root=str(tmp_path / "absent")))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    return [
        sys.executable, str(SCRIPT), "--config", str(path), "--output", str(tmp_path / "probe"),
        "--operation", "fit-batch", "--source", source, "--updates", "2", "--raw-side", "32",
        "--allow-runtime-mismatch", "--timeout", "600",
    ]


def test_real_data_absence_blocks_before_model_without_synthetic_fallback(tmp_path: Path) -> None:
    result = subprocess.run(_command(tmp_path, "real"), capture_output=True, text=True, timeout=30)
    report = json.loads((tmp_path / "probe/report.json").read_text())
    assert result.returncode == 2 and report["status"] == "blocked"
    assert report["arguments"]["source"] == "real"
    assert report["model_executed"] is False and "model_constructed" not in report
    assert "raw_hdf5_root" in report["missing"]


def test_cli_completes_two_actual_goal_model_updates_with_original_warmup(tmp_path: Path) -> None:
    result = subprocess.run(
        _command(tmp_path, "synthetic"), capture_output=True, text=True, timeout=620,
        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
    )
    report = json.loads((tmp_path / "probe/report.json").read_text())
    assert result.returncode == 0, (result.stderr, report)
    assert report["status"] == "passed" and report["model_completed"]
    assert not report["formal_training"] and report["dimensions_unchanged"]
    assert report["completed_optimizer_updates"] == 2
    assert report["probe_schedule"]["warmup_exceeds_budget"]
    assert report["config"]["optimizer"]["warmup_steps"] == 7
    trace = report["fixed_batch_learning"]
    assert trace["status"] == "completed" and len(trace["steps"]) == 2
    assert trace["before"]["annotated_goal"] and trace["after"]["annotated_goal"]
    assert trace["before"]["completed_optimizer_updates"] == 0
    assert trace["after"]["completed_optimizer_updates"] == 2
    assert not (tmp_path / "probe/sampled-action.npy").exists()
    assert not list((tmp_path / "probe").glob("*.pt"))
