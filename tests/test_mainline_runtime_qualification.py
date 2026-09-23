"""A bounded actual-runtime probe cannot substitute fixtures for missing real data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from test_mainline_endpoint_supervision import _config

from clearvla.mainline.runtime.qualification import (
    declared_runtime,
    qualification_config,
    required_data_paths,
    synthetic_batch,
    version_satisfies,
)
from clearvla.mainline.training.engine import validate_finite_training_batch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_mainline_runtime.py"


@pytest.mark.parametrize(
    "version,expected",
    [
        ("2.11.0", True),
        ("2.11.0+cu130", True),
        ("2.10.0+cpu", False),
        ("2.12.0", False),
        ("2.11.0rc1", False),
        ("not-a-version", False),
    ],
)
def test_runtime_admission_is_explicit(version: str, expected: bool) -> None:
    assert version_satisfies(version, ">=2.11,<2.12") is expected


def test_runtime_does_not_install_or_treat_override_as_a_supported_version() -> None:
    result = declared_runtime(ROOT, "3.10.20", "2.6.0+cu124")
    assert result["match"] is False
    c = _config()
    explicit = qualification_config(c, "bf16")
    assert explicit.runtime.compute_dtype == "bf16"
    assert (
        explicit.dimensions == c.dimensions
        and explicit.top == c.top
        and explicit.bottom == c.bottom
    )
    with pytest.raises(ValueError):
        version_satisfies("2.11.0", "~=2.11")


@pytest.mark.parametrize("patches", [64, 256])
def test_fixture_keeps_requested_dimensions_and_source_support(patches: int) -> None:
    c = _config()
    c = replace(c, dimensions=replace(c.dimensions, patches_per_camera=patches))
    rng = torch.get_rng_state().clone()
    b, _ = synthetic_batch(c, count=2, raw_side=32, device=torch.device("cpu"))
    b.validate(c)
    validate_finite_training_batch(b)
    assert torch.equal(rng, torch.get_rng_state())
    assert b.online.observation.dino_history.shape[3] == patches
    assert b.action_target.support is b.future.support
    assert b.online.history.executed_robot_step is not None
    assert torch.equal(
        b.online.history.executed_robot_step.command,
        b.online.history.executed_action_history[:, -1],
    )


@pytest.mark.parametrize("count,side", [(True, 32), (0, 32), (1, 31), (1, 32.0)])
def test_bad_synthetic_geometry_or_count_is_rejected(count: Any, side: Any) -> None:
    with pytest.raises(ValueError):
        synthetic_batch(_config(), count=count, raw_side=side, device=torch.device("cpu"))


def run(
    tmp_path: Path, *args: str, timeout: int = 65
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    cfg = _config()
    # All named resources deliberately absent; real mode cannot silently generate fixtures.
    cfg = replace(cfg, data=replace(cfg.data, raw_hdf5_root=str(tmp_path / "absent-episodes")))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.as_dict()))
    out = tmp_path / "probe"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(path),
            "--output",
            str(out),
            "--raw-side",
            "32",
            "--timeout",
            str(timeout - 10),
            *args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    report = json.loads((out / "report.json").read_text())
    return result, report


def test_missing_real_data_is_blocked_not_replaced(tmp_path: Path) -> None:
    proc, report = run(
        tmp_path, "--operation", "data", "--source", "real", "--allow-runtime-mismatch"
    )
    assert proc.returncode == report["returncode"] == 2
    assert report["status"] == "blocked" and report["missing"]
    assert report["model_executed"] is False
    assert report["formal_training"] is False
    assert not any(x for x in required_data_paths(_config()).values() if x["path"] == "synthetic")


def test_inspection_is_not_model_execution(tmp_path: Path) -> None:
    proc, report = run(tmp_path, "--allow-runtime-mismatch")
    assert proc.returncode == 0 and report["status"] == "passed"
    assert report["scope"] == "environment inspection only"
    assert report["model_executed"] is False


def test_bad_device_has_no_fallback(tmp_path: Path) -> None:
    proc, report = run(tmp_path, "--device", "meta", "--allow-runtime-mismatch")
    assert proc.returncode == 2 and report["status"] == "blocked"
    assert report["model_executed"] is False


def test_timeout_cannot_be_reported_as_success(tmp_path: Path) -> None:
    proc, report = run(tmp_path, "--timeout", "0.01", "--allow-runtime-mismatch")
    assert proc.returncode == report["returncode"] == 124
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    "operation,step", [("inference", 0), ("inference", 333), ("inference", 1200), ("train-step", 0)]
)
def test_actual_cumulative_production_classes_complete_bounded_probe(
    operation: str, step: int, tmp_path: Path
) -> None:
    proc, report = run(
        tmp_path,
        "--operation",
        operation,
        "--inference-step",
        str(step),
        "--allow-runtime-mismatch",
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr, report)
    assert (
        report["model_constructed"]
        and report["model_completed"]
        and report["model_forward_attempted"]
    )
    assert report["formal_training"] is False and report["dimensions_unchanged"]
    assert report["status"] == "passed"
    if operation == "train-step":
        assert report["completed_optimizer_updates"] == 1
    else:
        assert report["action_shape"] == [1, 24, 7]
        b = _config().bottom
        phase = min(
            max((step - b.execution_warmup_steps) / max(b.execution_transition_steps, 1), 0.0), 1.0
        )
        assert report["inference_phase_clock"]["completed_updates"] == step
        # The production schedule owns an FP32 buffer; compare its declared
        # representation, not Python double arithmetic with a looser tolerance.
        assert report["inference_phase_clock"]["phase"] == float(
            torch.tensor(phase, dtype=torch.float32)
        )
        assert "not actual completed training" in report["inference_phase_clock"]["scope"]
    before = (tmp_path / "probe/report.json").read_bytes()
    again = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "probe")],
        capture_output=True,
        timeout=10,
    )
    assert again.returncode != 0
    assert before == (tmp_path / "probe/report.json").read_bytes()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux parent-death contract")
@pytest.mark.parametrize("abrupt", [False, True])
def test_interrupted_probe_never_leaves_its_worker_alive(abrupt: bool, tmp_path: Path) -> None:
    import signal
    import time

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(_config().as_dict()))
    output = tmp_path / "interrupted"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    parent = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(cfg),
            "--output",
            str(output),
            "--operation",
            "inference",
            "--allow-runtime-mismatch",
            "--raw-side",
            "32",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = None
    try:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            report_path, process_path = output / "report.json", output / "process.json"
            if report_path.exists() and process_path.exists():
                report = json.loads(report_path.read_text())
                worker = json.loads(process_path.read_text())["worker_pid"]
                if report.get("parent_death_guard"):
                    break
            time.sleep(0.01)
        else:
            raise AssertionError("worker did not acknowledge parent binding")
        parent.send_signal(signal.SIGKILL if abrupt else signal.SIGTERM)
        parent.wait(timeout=10)
        assert parent.returncode != 0
        until = time.monotonic() + 5
        while time.monotonic() < until:
            status = Path(f"/proc/{worker}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("orphan worker still executing")
        report = json.loads((output / "report.json").read_text())
        assert report["status"] != "passed"
        if not abrupt:
            assert report["status"] == "failed"
            assert report["returncode"] == 128 + signal.SIGTERM
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker is not None:
            try:
                os.kill(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_relative_config_is_resolved_before_child_changes_directory(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_config().as_dict()))
    out = tmp_path / "relative-probe"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            "config.json",
            "--output",
            str(out),
            "--allow-runtime-mismatch",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert json.loads((out / "report.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("operation,step", [("inference", -1), ("inspect", 1), ("train-step", 1)])
def test_emulated_inference_clock_cannot_fake_completed_training(
    operation: str, step: int, tmp_path: Path
) -> None:
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--operation",
            operation,
            "--inference-step",
            str(step),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "inference-step" in result.stderr
    assert not output.exists()
