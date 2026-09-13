#!/usr/bin/env python3
"""Run the bounded forward/reverse ownership audit for CALVIN binding v2.

This is a structural gate, not a benchmark score.  It checks the properties
that can be established without a four-cell ablation: semantic targets have no
K-slot labels, the forward ingress stops at S/coarse, and every new parameter
has exactly one optimizer owner.  Dynamic permutation and VJP checks live in
``tests/test_calvin_object_binding.py`` and are run by the same gate command.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clearvla.mainline.calvin_binding_contract import (
    CALVIN_OBJECT_BINDING_COMPONENT,
    CALVIN_OBJECT_BINDING_CONFIG,
    CALVIN_OBJECT_BINDING_ROLE_NAMES,
    CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
    CALVIN_OBJECT_BINDING_TARGET_COUNT,
)
from clearvla.mainline.config import load_config
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.types import ActionIntentDock
from clearvla.mainline.training.optimizer import build_optimizer


def _source_checks(root: Path) -> dict[str, object]:
    bridge_source = (root / "clearvla/mainline/model/calvin_object_binding.py").read_text(
        encoding="utf-8"
    )
    sidecar_source = (root / "clearvla/mainline/data/calvin_object_binding.py").read_text(
        encoding="utf-8"
    )
    # These strings identify the rejected design or a privileged path.  A
    # v1 mention is allowed only in the explicit rejection/docstring branch.
    forbidden_bridge = {
        "scene_obs": "privileged simulator input",
        "pointer_targets": "fixed-slot sidecar field",
        "slot_seed": "per-slot role parameter",
    }
    bridge_ast = ast.parse(bridge_source)
    # Inspect identifiers rather than prose/docstrings: the module documents
    # why scene_obs and the old pointer_targets field are rejected.
    identifiers = {
        node.id
        for node in ast.walk(bridge_ast)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(bridge_ast)
        if isinstance(node, ast.Attribute)
    }
    forbidden_hits = {
        name: name in identifiers for name in forbidden_bridge
    }
    if any(forbidden_hits.values()):
        raise AssertionError(f"forbidden executable bridge identifiers: {forbidden_hits}")
    sidecar_ast = ast.parse(sidecar_source)
    sidecar_identifiers = {
        node.id for node in ast.walk(sidecar_ast) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(sidecar_ast) if isinstance(node, ast.Attribute)
    }
    if "pointer_targets" in sidecar_identifiers:
        raise AssertionError("sidecar implementation still exposes pointer_targets")
    action_fields = {field.name for field in fields(ActionIntentDock)}
    if {"object_binding_pointer", "object_binding_selected_geometry"} & action_fields:
        raise AssertionError("diagnostic binding tensors leaked into ActionIntentDock")
    return {
        "bridge_has_no_privileged_or_fixed_slot_markers": True,
        "action_dock_fields": sorted(action_fields),
    }


def _model_checks(root: Path) -> dict[str, object]:
    config = load_config(root / "configs/mainline/calvin_object_binding_formal_v2.json")
    policy = ClearVLAMainlinePolicy(config)
    selection = ComponentSelection.from_config(config)
    if selection.intent != CALVIN_OBJECT_BINDING_COMPONENT:
        raise AssertionError("CALVIN v2 did not select the binding component")
    optimizer, ownership = build_optimizer(policy, config)
    binding_names = {
        name for name, _ in policy.named_parameters() if "calvin_object_binding" in name
    }
    owned_names = {
        name
        for group in optimizer.param_groups
        if group.get("name", "").startswith("calvin_object_binding/")
        for name in group.get("parameter_names", ())
    }
    expected_legacy = {
        name.replace("intent.calvin_object_binding.", "calvin_object_binding.", 1)
        for name in binding_names
    }
    if owned_names != expected_legacy:
        raise AssertionError("binding parameters escaped or duplicated their optimizer owner")
    return {
        "component": selection.intent,
        "config_binding": config.top.calvin_object_binding,
        "binding_parameter_count": len(binding_names),
        "optimizer_binding_role_count": ownership.role_counts.get(
            "calvin_object_binding", 0
        ),
        "optimizer_binding_groups": [
            str(group["name"])
            for group in optimizer.param_groups
            if str(group.get("name", "")).startswith("calvin_object_binding/")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "component": CALVIN_OBJECT_BINDING_COMPONENT,
        "config_identity": CALVIN_OBJECT_BINDING_CONFIG,
        "sidecar_schema": CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
        "role_names": list(CALVIN_OBJECT_BINDING_ROLE_NAMES),
        "target_width": CALVIN_OBJECT_BINDING_TARGET_COUNT,
        "source": _source_checks(args.repo.resolve()),
    }
    if not args.skip_model:
        report["model"] = _model_checks(args.repo.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
