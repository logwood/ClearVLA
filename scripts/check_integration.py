#!/usr/bin/env python3
"""Run asset-free integration regressions using this exact Python interpreter."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAINLINE_PATTERNS = (
    "test_mainline_*.py", "test_calvin*.py", "test_integration*.py",
    "test_simulation*.py", "test_root_boundary*.py", "test_libero*.py",
    "test_benchmark*.py", "test_training*.py", "test_deployment*.py",
    "test_window_boundaries.py", "test_rdt_multitask_prep.py",
    "test_object_intent_dynamics_323.py", "test_repository_convergence_audit.py",
    "test_clearvla_workspace.py",
)
NUMERICS = (
    "clearvla/action_representations/bspline/tests",
    "clearvla/action_representations/composite/tests",
    "clearvla/action_solvers/flow_solver/tests",
    "tests/test_physical_action_codec.py", "tests/test_temporal_dct.py",
)


def selected_tests(suite: str) -> list[str]:
    if suite == "numerics":
        paths = list(NUMERICS)
    else:
        paths = []
        for pattern in MAINLINE_PATTERNS:
            matches = sorted((ROOT / "tests").glob(pattern))
            if not matches:
                raise FileNotFoundError(f"integration test selection has no match: {pattern}")
            paths.extend(str(path.relative_to(ROOT)) for path in matches)
    for name in paths:
        if not (ROOT / name).exists():
            raise FileNotFoundError(f"required integration test is missing: {name}")
    return list(dict.fromkeys(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("mainline", "numerics"), default="mainline")
    parser.add_argument("--junitxml")
    args, extra = parser.parse_known_args()
    command = [sys.executable, "-m", "pytest", "-q"]
    if args.junitxml:
        command.append("--junitxml=" + args.junitxml)
    return subprocess.call(command + selected_tests(args.suite) + extra, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
