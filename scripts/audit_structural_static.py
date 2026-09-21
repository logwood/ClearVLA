#!/usr/bin/env python3
"""Compare source diagnostics with the pinned base, without hiding old debt.

Ruff must be clean on every changed Python file. Pyright errors are compared
by file/definition/rule/message (line numbers can move) with multiplicity retained. New
files cannot inherit errors from another file. Warnings are reported, not
reclassified as errors or silently suppressed. This is a scoped regression
gate, not a claim that the complete historical repository is type-clean.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


def run_json(command: list[str], root: Path, log: Path) -> Any:
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    log.write_text(result.stdout, encoding="utf-8")
    log.with_suffix(".stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode not in (0, 1):
        raise RuntimeError(f"diagnostic process failed ({result.returncode}): {command[0]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"diagnostic did not produce valid JSON: {log}") from exc


@lru_cache(maxsize=512)
def source_owners(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Anchor diagnostics to their actual enclosing definition, not a line budget."""
    owners: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, parent: str) -> None:
        name = parent
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{parent}.{node.name}" if parent else node.name
            owners.append((node.lineno, node.end_lineno or node.lineno, name))
        for child in ast.iter_child_nodes(node):
            visit(child, name)

    visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), "")
    return tuple(owners)


def type_counter(payload: dict[str, Any], root: Path) -> collections.Counter[tuple[str, ...]]:
    rows: collections.Counter[tuple[str, ...]] = collections.Counter()
    for item in payload["generalDiagnostics"]:
        path = Path(item["file"]).resolve()
        filename = path.relative_to(root.resolve()).as_posix()
        line = int(item["range"]["start"]["line"]) + 1
        owners = [(hi - lo, name) for lo, hi, name in source_owners(path) if lo <= line <= hi]
        owner = min(owners)[1] if owners else "<module>"
        rows[(filename, owner, item["severity"], item.get("rule", ""), item["message"])] += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=Path, required=True)
    args = parser.parse_args()
    baseline, current, output = (
        args.baseline.resolve(),
        args.current.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    files = args.files.read_text(encoding="utf-8").splitlines()
    if not files or any(Path(f).is_absolute() or ".." in Path(f).parts for f in files):
        raise ValueError("changed file inventory must contain relative repository paths")
    files = sorted(set(f for f in files if f.endswith(".py") and (current / f).is_file()))
    old_files = [f for f in files if (baseline / f).is_file()]
    if not files or not old_files:
        raise ValueError("this milestone requires both inherited and newly reviewed source")
    current_lint = run_json(
        ["ruff", "check", "--output-format=json", *files], current, output / "ruff-changed.json"
    )
    # Preserve the full old/new lint inventory as evidence. Only the changed
    # source gate is blocking; unrelated historical errors are not waived away.
    for name, root in (("baseline", baseline), ("current", current)):
        run_json(
            ["ruff", "check", "--output-format=json", "clearvla", "tests"],
            root,
            output / f"ruff-{name}-inventory.json",
        )
    old = run_json(
        ["pyright", "--outputjson", *old_files], baseline, output / "pyright-baseline.json"
    )
    new = run_json(["pyright", "--outputjson", *files], current, output / "pyright-current.json")
    added = type_counter(new, current) - type_counter(old, baseline)
    added_rows = [
        dict(file=k[0], owner=k[1], severity=k[2], rule=k[3], message=k[4], count=v)
        for k, v in sorted(added.items())
    ]
    errors = sum(count for key, count in added.items() if key[2] == "error")
    missing_imports = sum(count for key, count in added.items() if key[3] == "reportMissingImports")
    report = {
        "scope": "changed production/test/review-tool Python files; inherited diagnostics retained",
        "baseline": str(baseline),
        "current": str(current),
        "files": files,
        "ruff_changed_diagnostics": len(current_lint),
        "pyright_baseline": old["summary"],
        "pyright_current": new["summary"],
        "new_type_errors": errors,
        "new_missing_imports": missing_imports,
        "added_diagnostics": added_rows,
        "passed": not current_lint and errors == 0 and missing_imports == 0,
    }
    (output / "static-delta.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "ruff_changed_diagnostics", "new_type_errors")}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
