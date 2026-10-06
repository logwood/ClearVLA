#!/usr/bin/env python3
"""Deep target-swap trace for P3 source mixing and bottom consumption.

The intervention is identical to probe_target_binding_swap.py.  This version
keeps the causal path read-only and records the named source axes around the
P3 coordinator, task target/scene compilation, typed bottom evidence lanes,
execution-controller gates, and each bottom block update.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch

from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.mainline.model.horizon_coordination import TypedHorizonCoordinator
from clearvla.mainline.model.compiler import ObjectPolicyPlanCompiler
from clearvla.mainline.model.bottom import TypedEvidenceCompiler, ReadOnlyEvidenceMMDiTBlock, ExecutionController
from clearvla.mainline.runtime.sampling import deployment_cache, sample_cached_action
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from probes.probe_g_slot_identity_nodes import (
    array, compare, history_with_world, intent_fields, install_wrappers,
    restore_wrappers, summary,
)
from probes.probe_target_binding_swap import _entropy, _swap_wrapper

DEFAULT_INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
)

def _record(current: dict[str, Any], name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        current[name] = array(value)

def install_deep_wrappers(model: torch.nn.Module, current: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    """Patch the actual restored V120 instances after the legacy node hooks."""
    saved: list[tuple[Any, str, Any]] = []
    def patch(owner: Any, name: str, fn: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)
    planner = model.policy_compiler.plan_compiler
    coordinator = getattr(planner, "coordinator", None)
    if coordinator is not None:
        old = coordinator.forward
        coord_obj = coordinator
        def coordinator_forward(inputs: Any, _old=old):
            for name in (
                "action", "current_fact", "policy_precision", "semantic_effect",
                "geometry_feature_effect", "task_temporal", "observed_change",
            ):
                value = getattr(inputs, name, None)
                _record(current, "p3_input_" + name, value)
                if name in getattr(coord_obj, "SOURCE_NAMES", ()):
                    _record(current, "p3_projected_" + name, coord_obj.sources[name](value))
            out = _old(inputs)
            _record(current, "p3_coord_temporal_deep", out[0])
            _record(current, "p3_coord_state_change_deep", out[1])
            _record(current, "p3_coord_private_deep", out[2])
            return out
        patch(coordinator, "forward", coordinator_forward)
    old = planner.forward
    def planner_forward(*args: Any, _old=old, **kwargs: Any):
        task_execution = kwargs.get("task_execution")
        if task_execution is not None:
            _record(current, "p3_task_target", task_execution.target)
            _record(current, "p3_task_scene", task_execution.scene)
        return _old(*args, **kwargs)
    patch(planner, "forward", planner_forward)

    stage = getattr(model, "execution_bottom", None)
    decoder = getattr(stage, "decoder", None) if stage is not None else None
    if decoder is None:
        raise RuntimeError("deep probe could not locate restored execution decoder")

    old = decoder.evidence_adapter.forward
    def adapter_forward(*args: Any, _old=old, **kwargs: Any):
        out = _old(*args, **kwargs)
        for name, value in out.source_tokens.items():
            _record(current, "bottom_view_source_" + name, value)
        _record(current, "bottom_view_selector_tokens", out.tokens)
        _record(current, "bottom_view_value_tokens", out.value_tokens)
        _record(current, "bottom_view_intent_tokens", out.intent_tokens)
        for name, value in out.summaries.items():
            _record(current, "bottom_view_summary_" + name, value)
        return out
    patch(decoder.evidence_adapter, "forward", adapter_forward)

    old = decoder.organizer.forward
    def organizer_forward(*args: Any, _old=old, **kwargs: Any):
        if args:
            view = args[0]
            if hasattr(view, "value_tokens"):
                _record(current, "bottom_organizer_view_values", view.value_tokens)
                _record(current, "bottom_organizer_view_intent", view.intent_tokens)
        out = _old(*args, **kwargs)
        if isinstance(out, dict):
            for name in ("condition_hidden", "latent_token", "global_condition", "time_hidden", "scan", "intent_context"):
                _record(current, "bottom_organizer_" + name, out.get(name))
        return out
    patch(decoder.organizer, "forward", organizer_forward)

    old = decoder._read_policy_delta_bank
    def policy_delta_forward(action_query: torch.Tensor, bank: Any, *, collect_diagnostics: bool = True, _old=old):
        _record(current, "bottom_role_value_all", bank.values)
        names = tuple(getattr(bank, "source_names", ()))
        for index, name in enumerate(names):
            if index < int(bank.values.shape[1]):
                _record(current, "bottom_role_value_" + str(name), bank.values[:, index])
        _record(current, "bottom_role_protected_detail", bank.protected_detail)
        _record(current, "bottom_role_protected_precision", bank.protected_policy_precision)
        out = _old(action_query, bank, collect_diagnostics=collect_diagnostics)
        _record(current, "bottom_role_optional_update", out[0])
        _record(current, "bottom_role_precision_update", out[1])
        return out
    patch(decoder, "_read_policy_delta_bank", policy_delta_forward)

    for index, block in enumerate(decoder.blocks):
        old = block.forward
        def block_forward(*args: Any, _old=old, _index=index, **kwargs: Any):
            action = kwargs.get("action", args[0] if len(args) > 0 else None)
            evidence = kwargs.get("evidence_tokens", args[1] if len(args) > 1 else None)
            condition = kwargs.get("global_condition", args[2] if len(args) > 2 else None)
            call_key = f"_bottom_block_{_index}_calls"
            call_index = int(current.get(call_key, 0))
            current[call_key] = call_index + 1
            prefix = f"bottom_block_{_index}_call_{call_index}_"
            _record(current, prefix + "action_in", action)
            _record(current, prefix + "evidence", evidence)
            _record(current, prefix + "condition", condition)
            out = _old(*args, **kwargs)
            if isinstance(out, tuple):
                _record(current, prefix + "action_out", out[0])
            elif isinstance(out, torch.Tensor):
                _record(current, prefix + "action_out", out)
            return out
        patch(block, "forward", block_forward)
    return saved

def _run_one(*, policy: Any, history: Any, instruction: str, current: dict[str, Any],
             binder_forward: Any, model: torch.nn.Module, swap: bool) -> dict[str, Any]:
    policy.reset()
    current.clear()
    TaskConditionedTargetBinder.forward = binder_forward
    _, online = policy.act_with_input(history, instruction)
    config = policy.bundle.config
    with torch.no_grad():
        cache, _ = deployment_cache(model, online, config, collect_diagnostics=True, dtype=torch.bfloat16)
        current.update(intent_fields(cache.top.intent))
        noise_generator = torch.Generator(device=online.device).manual_seed(12345)
        noise = torch.randn(
            1, config.dimensions.action_horizon, model.outlet_adapter.physical_dim,
            device=online.device, dtype=torch.float32, generator=noise_generator,
        )
        sampled = sample_cached_action(
            model, cache, config, initial_physical_noise=noise,
            collect_diagnostics=True, dtype=torch.bfloat16, pass_role="refined",
        )
    nodes = {name: value for name, value in current.items() if isinstance(value, np.ndarray)}
    nodes["final_native_action"] = sampled.action[0].detach().float().cpu().numpy()
    nodes["final_physical_field"] = sampled.physical_field[0].detach().float().cpu().numpy()
    if sampled.gripper_command is not None:
        nodes["final_gripper_command"] = sampled.gripper_command[0].detach().float().cpu().numpy()
    return {"swap": bool(swap), "pair": current.get("swap_pair"), "nodes": nodes, "metrics": current.get("velocity_metrics", {})}

def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=device, t5_condition=args.t5_condition, seed=0)
    model = policy.bundle.model
    current: dict[str, Any] = {}
    base_saved = install_wrappers(model, current)
    deep_saved = install_deep_wrappers(model, current)
    installed_binder_forward = TaskConditionedTargetBinder.forward
    swapped_binder_forward = _swap_wrapper(installed_binder_forward, current)
    records: list[dict[str, Any]] = []
    try:
        for layout in args.layouts:
            history = history_with_world(args.observation_dir / f"{layout}.npz")
            for instruction in args.instructions:
                baseline = _run_one(policy=policy, history=history, instruction=instruction, current=current,
                                    binder_forward=installed_binder_forward, model=model, swap=False)
                swapped = _run_one(policy=policy, history=history, instruction=instruction, current=current,
                                   binder_forward=swapped_binder_forward, model=model, swap=True)
                before = baseline["nodes"]["s_binding_mass"]
                after = swapped["nodes"]["s_binding_mass"]
                pair = [int(x) for x in np.asarray(swapped["pair"]).reshape(-1)]
                real_before = np.asarray(before)[0]
                real_after = np.asarray(after)[0]
                names = sorted(set(baseline["nodes"]).intersection(swapped["nodes"]))
                records.append({
                    "layout": layout,
                    "instruction": instruction,
                    "pair": pair,
                    "binding_invariants": {
                        "real_mass_sum_abs_diff": float(abs(real_before.sum() - real_after.sum())),
                        "null_mass_abs_diff": float(abs(float(np.asarray(baseline["nodes"]["s_binding_null_mass"]).reshape(-1)[0]) - float(np.asarray(swapped["nodes"]["s_binding_null_mass"]).reshape(-1)[0]))),
                        "sorted_real_mass_linf": float(np.max(np.abs(np.sort(real_before) - np.sort(real_after)))),
                        "entropy_abs_diff": float(abs(_entropy(real_before) - _entropy(real_after))),
                        "pair_exchange_linf": float(max(abs(real_after[pair[0]] - real_before[pair[1]]), abs(real_after[pair[1]] - real_before[pair[0]]))),
                    },
                    "node_differences": {name: compare(swapped["nodes"][name], baseline["nodes"][name]) for name in names},
                    "node_summaries": {
                        "baseline": {name: summary(baseline["nodes"][name]) for name in names},
                        "swapped": {name: summary(swapped["nodes"][name]) for name in names},
                    },
                })
    finally:
        for owner, name, old in reversed(deep_saved):
            setattr(owner, name, old)
        restore_wrappers(base_saved)
    result = {
        "schema": "clearvla-target-binding-deep-v1",
        "checkpoint": str(args.checkpoint),
        "observation_dir": str(args.observation_dir),
        "layouts": list(args.layouts),
        "instructions": list(args.instructions),
        "intervention": {
            "evidence_changed": False, "instruction_changed": False, "noise_seed": 12345,
            "null_mass_changed": False, "real_mass_multiset_changed": False,
            "operation": "swap_two_highest_supported_real_binding_masses",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layouts", nargs="+", default=["standard"])
    parser.add_argument("--instructions", nargs="+", default=list(DEFAULT_INSTRUCTIONS))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "records": len(result["records"])}, indent=2))

if __name__ == "__main__":
    main()

