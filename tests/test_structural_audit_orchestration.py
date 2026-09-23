"""Exercise the real shell gate with stubbed expensive child processes.

These are orchestration tests, not replacements for production neural tests.
The baseline must be measured independently, never used to skip current code.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_gate(tmp_path: Path, baseline: int, current: int, static: int = 0) -> tuple[subprocess.CompletedProcess[str], Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "tests/test_mainline_example.py").write_text("# inventory fixture\n")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/check_structural_rebuild.sh",
        scripts / "check_structural_rebuild.sh",
    )
    old = tmp_path / "baseline"
    (old / "clearvla").mkdir(parents=True)
    (old / "tests").mkdir()
    (old / "tests/test_mainline_example.py").write_text("# old inventory fixture\n")
    changes = tmp_path / "changes.txt"
    changes.write_text("tests/test_mainline_example.py\n")
    output = tmp_path / "reports"
    bins = tmp_path / "bin"
    bins.mkdir()
    fake_python = bins / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['-']:\n"
        "    sys.stdin.read()\n"
        "    print(json.dumps({'runtime': 'orchestration-stub'}))\n"
        "elif args and args[0] == '-':\n"
        f"    os.execv({sys.executable!r}, [{sys.executable!r}] + args)\n"
        "elif args[:2] == ['-m', 'compileall']:\n"
        "    pass\n"
        "elif args and args[0].endswith('audit_structural_static.py'):\n"
        "    sys.exit(int(os.environ['STUB_STATIC_EXIT']))\n"
        "elif args and args[0].endswith('run_structural_tests.py'):\n"
        "    label = args[args.index('--label') + 1]\n"
        "    with pathlib.Path(os.environ['STUB_CALLS']).open('a') as stream:\n"
        "        stream.write(label + '\\n')\n"
        "    print(json.dumps({'label': label, 'stub': True}))\n"
        "    sys.exit(int(os.environ['STUB_' + label.upper() + '_EXIT']))\n"
        "else:\n"
        "    raise RuntimeError('unrecognized orchestration invocation: ' + repr(args))\n"
    )
    fake_python.chmod(0o755)
    for name in ("ruff", "pyright"):
        path = bins / name
        path.write_text("#!/bin/sh\necho orchestration-stub\n")
        path.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=str(bins) + os.pathsep + env.get("PATH", ""),
        CLEARVLA_AUDIT_BASELINE=str(old),
        CLEARVLA_AUDIT_CHANGED_FILES=str(changes),
        STUB_CALLS=str(tmp_path / "calls.txt"),
        STUB_BASELINE_EXIT=str(baseline),
        STUB_CURRENT_EXIT=str(current),
        STUB_STATIC_EXIT=str(static),
    )
    result = subprocess.run(
        ["bash", str(scripts / "check_structural_rebuild.sh"), str(output)],
        cwd=repo, env=env, text=True, capture_output=True, check=False, timeout=15,
    )
    return result, output


@pytest.mark.parametrize("baseline,current", [(0, 0), (1, 0), (0, 1), (3, 5), (137, 0), (0, 137)])
def test_baseline_failure_does_not_skip_current_and_never_turns_green(
    tmp_path: Path, baseline: int, current: int,
) -> None:
    result, output = _run_gate(tmp_path, baseline, current)
    assert (tmp_path / "calls.txt").read_text().splitlines() == ["baseline", "current"]
    report = json.loads((output / "regression-exits.json").read_text())
    assert report == {
        "baseline_exit": baseline,
        "current_exit": current,
        "both_inventories_attempted": True,
        "passed": baseline == current == 0,
    }
    assert result.returncode == (0 if baseline == current == 0 else 1), result.stderr
    assert '"label": "baseline"' in (output / "baseline-tests.txt").read_text()
    assert '"label": "current"' in (output / "current-tests.txt").read_text()


def test_static_preparation_failure_is_not_falsely_marked_as_neural_acceptance(tmp_path: Path) -> None:
    result, output = _run_gate(tmp_path, 0, 0, static=7)
    assert result.returncode == 7
    assert not (tmp_path / "calls.txt").exists()
    assert not (output / "regression-exits.json").exists()
