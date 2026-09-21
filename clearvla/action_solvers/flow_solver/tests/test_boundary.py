from __future__ import annotations

import ast
from pathlib import Path

import clearvla.action_solvers.flow_solver as flow_solver


def test_solver_lane_has_no_direct_mainline_imports() -> None:
    package_dir = Path(flow_solver.__file__).resolve().parent
    for source_path in package_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert all(
                not module.startswith("clearvla.mainline") for module in modules
            ), f"{source_path.name} imports the mainline directly"
