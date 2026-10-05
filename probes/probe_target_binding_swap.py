#!/usr/bin/env python3
"""Counterfactual target-binding swap through the complete online action path.

The probe keeps the observation, instruction, G evidence, null mass and the
multiset of real-object masses fixed.  It swaps the two highest supported
real-object masses after the shared binder and lets the ordinary S/P2/P3/W
consumer path run normally.  This is read-only instrumentation; it never
changes checkpoint parameters or training state.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.mainline.runtime.sampling import deployment_cache, sample_cached_action
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from probes.probe_g_slot_identity_nodes import (
    array,
    compare,
    history_with_world,
    intent_fields,
    install_wrappers,
    restore_wrappers,
    summary,
)


DEFAULT_INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
)


def _entropy(mass: np.ndarray) -> float:
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    values = np.clip(values, 1e-30, None)
    return float(-(values * np.log(values)).sum())


def _swap_wrapper(base_forward: Any, current: dict[str, Any]):
    def forward(
        self: Any,
        task: torch.Tensor,
        objects: torch.Tensor,
        supported: torch.Tensor,
        *,
        history: torch.Tensor,
        view_support: torch.Tensor | None = None,
    ):
        result = base_forward(
            self,
            task,
            objects,
            supported,
            history=history,
            view_support=view_support,
        )
        if int(result.log_probability.shape[0]) != 1:
            raise ValueError("the binding-swap probe requires batch size one")
        legal = result.supported[0]
        candidate = torch.where(legal)[0]
        if int(candidate.numel()) < 2:
            raise ValueError("binding-swap probe found fewer than two real objects")
        masses = result.mass[0, candidate]
        pair = candidate[torch.topk(masses, k=2, largest=True, sorted=True).indices]
        first, second = int(pair[0]), int(pair[1])
        permutation = torch.arange(
            result.log_probability.shape[1] - 1,
            device=result.log_probability.device,
        )
        permutation[first], permutation[second] = second, first
        real_log_probability = result.log_probability[..., :-1][:, permutation]
        swapped = replace(
            result,
            log_probability=torch.cat(
                (real_log_probability, result.log_probability[..., -1:]), dim=-1
            ),
        )
        current["swap_pair"] = np.asarray([first, second], dtype=np.int64)
        current["swap_mass_before"] = array(result.mass)
        current["swap_mass_after"] = array(swapped.mass)
        current["swap_null_before"] = array(result.null_mass)
        current["swap_null_after"] = array(swapped.null_mass)
        return swapped

    return forward


def _run_one(
    *,
    policy: ClearVLACheckpointPolicy,
    history: Any,
    instruction: str,
    current: dict[str, Any],
    binder_forward: Any,
    model: torch.nn.Module,
    swap: bool,
) -> dict[str, Any]:
    policy.reset()
    current.clear()
    TaskConditionedTargetBinder.forward = binder_forward
    _, online = policy.act_with_input(history, instruction)
    config = policy.bundle.config
    with torch.no_grad():
        cache, _ = deployment_cache(
            model,
            online,
            config,
            collect_diagnostics=True,
            dtype=torch.bfloat16,
        )
        current.update(intent_fields(cache.top.intent))
        noise_generator = torch.Generator(device=online.device).manual_seed(12345)
        noise = torch.randn(
            1,
            config.dimensions.action_horizon,
            model.outlet_adapter.physical_dim,
            device=online.device,
            dtype=torch.float32,
            generator=noise_generator,
        )
        sampled = sample_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=noise,
            collect_diagnostics=True,
            dtype=torch.bfloat16,
            pass_role="refined",
        )
    nodes = {
        name: value
        for name, value in current.items()
        if isinstance(value, np.ndarray)
    }
    nodes["final_native_action"] = sampled.action[0].detach().float().cpu().numpy()
    nodes["final_physical_field"] = (
        sampled.physical_field[0].detach().float().cpu().numpy()
    )
    if sampled.gripper_command is not None:
        nodes["final_gripper_command"] = (
            sampled.gripper_command[0].detach().float().cpu().numpy()
        )
    required = (
        "s_binding_mass",
        "s_typed_common_value",
        "s_typed_interval_residual_value",
        "p2_selected_semantic_value",
        "p2_selected_geometry_value",
        "p2_selected_selected_target_value",
        "p2_effect_semantic",
        "p2_effect_geometry",
        "p3_temporal",
        "final_native_action",
    )
    missing = [name for name in required if name not in nodes]
    if missing:
        raise RuntimeError(f"binding-swap capture missing {missing}; swap={swap}")
    return {
        "swap": bool(swap),
        "pair": current.get("swap_pair"),
        "nodes": nodes,
        "metrics": current.get("velocity_metrics", {}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
        seed=0,
    )
    model = policy.bundle.model
    current: dict[str, Any] = {}
    saved = install_wrappers(model, current)
    installed_binder_forward = TaskConditionedTargetBinder.forward
    swapped_binder_forward = _swap_wrapper(installed_binder_forward, current)
    records: list[dict[str, Any]] = []
    try:
        for layout in args.layouts:
            history = history_with_world(args.observation_dir / f"{layout}.npz")
            for instruction in args.instructions:
                baseline = _run_one(
                    policy=policy,
                    history=history,
                    instruction=instruction,
                    current=current,
                    binder_forward=installed_binder_forward,
                    model=model,
                    swap=False,
                )
                swapped = _run_one(
                    policy=policy,
                    history=history,
                    instruction=instruction,
                    current=current,
                    binder_forward=swapped_binder_forward,
                    model=model,
                    swap=True,
                )
                before = baseline["nodes"]["s_binding_mass"]
                after = swapped["nodes"]["s_binding_mass"]
                pair = [int(x) for x in np.asarray(swapped["pair"]).reshape(-1)]
                real_before = np.asarray(before)[0]
                real_after = np.asarray(after)[0]
                records.append(
                    {
                        "layout": layout,
                        "instruction": instruction,
                        "pair": pair,
                        "binding_invariants": {
                            "real_mass_sum_before": float(real_before.sum()),
                            "real_mass_sum_after": float(real_after.sum()),
                            "real_mass_sum_abs_diff": float(
                                abs(real_before.sum() - real_after.sum())
                            ),
                            "null_mass_before": float(
                                np.asarray(baseline["nodes"]["s_binding_null_mass"])
                                .reshape(-1)[0]
                            ),
                            "null_mass_after": float(
                                np.asarray(swapped["nodes"]["s_binding_null_mass"])
                                .reshape(-1)[0]
                            ),
                            "null_mass_abs_diff": float(
                                abs(
                                    np.asarray(
                                        baseline["nodes"]["s_binding_null_mass"]
                                    ).reshape(-1)[0]
                                    - np.asarray(
                                        swapped["nodes"]["s_binding_null_mass"]
                                    ).reshape(-1)[0]
                                )
                            ),
                            "sorted_real_mass_linf": float(
                                np.max(
                                    np.abs(
                                        np.sort(real_before) - np.sort(real_after)
                                    )
                                )
                            ),
                            "entropy_before": _entropy(real_before),
                            "entropy_after": _entropy(real_after),
                            "pair_exchange_linf": float(
                                max(
                                    abs(real_after[pair[0]] - real_before[pair[1]]),
                                    abs(real_after[pair[1]] - real_before[pair[0]]),
                                )
                            ),
                        },
                        "node_summaries": {
                            "baseline": {
                                name: summary(value)
                                for name, value in baseline["nodes"].items()
                                if name
                                in {
                                    "s_typed_common_value",
                                    "s_typed_interval_residual_value",
                                    "p2_selected_selected_target_value",
                                    "p2_effect_semantic",
                                    "p3_temporal",
                                    "final_native_action",
                                }
                            },
                            "swapped": {
                                name: summary(value)
                                for name, value in swapped["nodes"].items()
                                if name
                                in {
                                    "s_typed_common_value",
                                    "s_typed_interval_residual_value",
                                    "p2_selected_selected_target_value",
                                    "p2_effect_semantic",
                                    "p3_temporal",
                                    "final_native_action",
                                }
                            },
                        },
                        "node_differences": {
                            name: compare(swapped["nodes"][name], baseline["nodes"][name])
                            for name in (
                                "s_public_interval_carrier",
                                "s_typed_common_value",
                                "s_typed_interval_residual_value",
                                "p2_selected_semantic_value",
                                "p2_selected_geometry_value",
                                "p2_selected_selected_target_value",
                                "p2_effect_semantic",
                                "p2_effect_geometry",
                                "p3_temporal",
                                "bottom_policy_delta_update",
                                "bottom_policy_precision_update",
                                "final_physical_field",
                                "final_native_action",
                            )
                        },
                    }
                )
    finally:
        restore_wrappers(saved)
    result = {
        "schema": "clearvla-target-binding-mass-swap-v1",
        "checkpoint": str(args.checkpoint),
        "observation_dir": str(args.observation_dir),
        "layouts": list(args.layouts),
        "instructions": list(args.instructions),
        "intervention": {
            "evidence_changed": False,
            "instruction_changed": False,
            "noise_seed": 12345,
            "null_mass_changed": False,
            "real_mass_multiset_changed": False,
            "operation": "swap_two_highest_supported_real_binding_masses",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
