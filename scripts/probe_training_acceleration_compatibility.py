"""Read-only preflight for reusing the training acceleration layer.

This probe deliberately parses source files instead of importing a branch.  It
can therefore compare branches with incompatible dependency environments and
reports whether a branch exposes the stable training surface, needs a thin
adapter backport, or is a legacy implementation outside the current backend.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def _symbols(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
    return symbols


def _digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def inspect_worktree(root: Path) -> dict[str, Any]:
    engine = root / "clearvla/mainline/training/engine.py"
    policy = root / "clearvla/mainline/model/policy.py"
    graph = root / "clearvla/mainline/training/cuda_graph.py"
    contract = root / "clearvla/mainline/training/acceleration_contract.py"
    adapter = root / "clearvla/mainline/training/acceleration_adapters.py"
    legacy = root / "clearvla/policy/system.py"

    engine_symbols = _symbols(engine)
    policy_symbols = _symbols(policy)
    graph_symbols = _symbols(graph)
    mainline_surface = {
        "engine_forward": "_forward" in engine_symbols,
        "engine_train_step": "train_step" in engine_symbols,
        "policy_set_training_step": "set_training_step" in policy_symbols,
    }
    backend_surface = {
        "cuda_graph_runner": "CudaGraphTrainingStepRunner" in graph_symbols,
        "portable_contract": contract.is_file(),
        "mainline_adapter": adapter.is_file(),
        "policy_adapter_hook": "get_training_acceleration_adapter" in policy_symbols,
    }
    if all(mainline_surface.values()):
        if all(backend_surface.values()):
            classification = "ready-portable-mainline"
            recommendation = "run the common CUDA-Graph/eager gate"
        elif backend_surface["cuda_graph_runner"]:
            classification = "mainline-needs-contract-backport"
            recommendation = "backport the contract and add one version adapter"
        else:
            classification = "mainline-needs-backend-backport"
            recommendation = "backport the shared backend and one version adapter"
    elif legacy.is_file():
        classification = "legacy-surface-needs-bridge"
        recommendation = "write a legacy TrainingStepSurface adapter before benchmarking"
    else:
        classification = "unsupported-surface"
        recommendation = "inspect the branch's training entry point manually"

    digest_paths = (engine, policy, graph, contract, adapter, legacy)
    return {
        "root": str(root.resolve()),
        "classification": classification,
        "recommendation": recommendation,
        "mainline_surface": mainline_surface,
        "backend_surface": backend_surface,
        "source_digest": _digest(digest_paths),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit implementation-level training acceleration compatibility"
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    reports = [inspect_worktree(root) for root in args.roots]
    if args.pretty:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
