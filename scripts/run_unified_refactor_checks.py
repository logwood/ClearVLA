#!/usr/bin/env python3
"""Read-only source and regression gates for the isolated refactor.

Artifacts distinguish synthesized model fixtures from real dataset training.
Every selected test file runs in a fresh process; failures and timeouts survive.
No production checkpoint, optimizer, branch, or dataset is mutated here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from run_structural_tests import run_inventory

PATTERNS = (
    "test_mainline*.py",
    "test_causal*.py",
    "test_unified*.py",
    "test_full_token_binding.py",
    "test_identity*.py",
    "test_canonical_transport.py",
    "test_log_transport_adjoint.py",
    "test_observed_correspondence.py",
    "test_source_consistent_measurement.py",
    "test_physical_chart_metadata.py",
    "test_calvin*.py",
)


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def source_fingerprint(root: Path) -> dict[str, str]:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in files
        if name and (root / name).is_file()
    }


def syntax_audit(root: Path) -> dict[str, object]:
    rows = []
    for name in source_fingerprint(root):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        except (SyntaxError, UnicodeError) as error:
            rows.append(dict(file=name, status="failed", error=str(error)))
            continue
        # An inventory, not a claim to statically prove dynamic autograd.
        detaches = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"detach", "detach_"}
        ]
        rows.append(dict(file=name, status="parsed", explicit_detach_lines=detaches))
    return dict(
        passed=all(row["status"] == "parsed" for row in rows),
        files=rows,
        scope="all tracked Python syntax and explicit stop-gradient inventory; dynamic routes tested separately",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--timeout-per-file", type=float, default=600.0)
    parser.add_argument(
        "--inventory",
        choices=("all", "primary"),
        default="all",
        help="all includes tracked adapter/legacy/nested solver tests",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError("audit output must be outside the source checkout")
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid test shard")
    out.mkdir(parents=True, exist_ok=True)
    import torch

    before = source_fingerprint(root)
    write(out / "source-before.json", before)
    write(
        out / "runtime.json",
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            scope="CPU synthetic model fixtures and file-format fixtures; no private-data training",
        ),
    )
    syntax = syntax_audit(root)
    write(out / "static-dataflow-inventory.json", syntax)
    collect = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "collection.log").write_text(collect.stdout + collect.stderr)
    files = sorted(
        {
            p.relative_to(root).as_posix()
            for pattern in PATTERNS
            for p in (root / "tests").glob(pattern)
        }
    )
    primary = files
    if args.inventory == "all":
        tracked = (
            subprocess.check_output(["git", "ls-files", "-z", "*test_*.py"], cwd=root)
            .decode()
            .split("\0")
        )
        files = sorted(
            name
            for name in tracked
            if name and Path(name).name.startswith("test_") and (root / name).is_file()
        )
    if not files:
        raise ValueError("no tracked regression tests selected")
    selected = files[args.shard :: args.shards]
    write(
        out / "inventory.json",
        dict(
            scope=args.inventory,
            primary_inventory=primary,
            total_inventory=files,
            shard=args.shard,
            shards=args.shards,
            selected=selected,
        ),
    )
    result = run_inventory(root, selected, out, "current", args.timeout_per_file)
    after = source_fingerprint(root)
    intact = before == after
    write(out / "source-after.json", after)
    write(
        out / "real-data-gate.json",
        dict(
            status="not_run",
            passed=False,
            reason="This read-only CPU gate never substitutes synthetic tensors for source datasets/checkpoints. Use run_unified_real_data_check.py with admitted assets.",
        ),
    )
    passed = bool(syntax["passed"]) and collect.returncode == 0 and result == 0 and intact
    write(
        out / "gate-summary.json",
        dict(
            cpu_passed=passed,
            syntax_passed=syntax["passed"],
            collection_exit=collect.returncode,
            regression_exit=result,
            source_unchanged=intact,
            real_data_passed=False,
            all_user_requested_gates_complete=False,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
