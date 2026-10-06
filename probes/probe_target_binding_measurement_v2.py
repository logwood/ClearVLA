#!/usr/bin/env python3
"""Measurement-correct target-binding trace; instrumentation only.

The fixed-W one-pass probe and the complete proposal -> W rebuild -> refined
sampler are reported separately. Bottom policy reads are split into raw P3
lanes, P1 precision, their scaled bridge, and protected detail. Velocity
events retain their sampling pass and integration index.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import clearvla.mainline.runtime.sampling as sampling_runtime
from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from probes.probe_g_slot_identity_nodes import (
    array, compare, history_with_world, intent_fields, install_wrappers,
    restore_wrappers, summary,
)

DEFAULT_INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
)


def _entropy(mass: np.ndarray) -> float:
    values = np.clip(np.asarray(mass, dtype=np.float64).reshape(-1), 1e-30, None)
    return float(-(values * np.log(values)).sum())


def _scalar_metrics(mapping: Any) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    return {
        str(key): float(value.detach().float().item())
        for key, value in mapping.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }


def _record(current: dict[str, Any], name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        current[name] = array(value)


def _node_arrays(current: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: value for name, value in current.items()
        if isinstance(value, np.ndarray) and not name.startswith("_")
    }


def _trace_summary(current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    prefixes = ("s_", "p1_", "p2_", "p3_", "bottom_", "decoder_", "physical_")
    return {
        name: summary(value)
        for name, value in current.items()
        if isinstance(value, np.ndarray)
        and not name.startswith("_")
        and name.startswith(prefixes)
        and "_call_" not in name
    }


def _final_nodes(current: dict[str, Any], sampled: Any) -> dict[str, np.ndarray]:
    nodes = _node_arrays(current)
    nodes["final_native_action"] = sampled.action[0].detach().float().cpu().numpy()
    nodes["final_physical_field"] = (
        sampled.physical_field[0].detach().float().cpu().numpy()
    )
    if sampled.gripper_command is not None:
        nodes["final_gripper_command"] = (
            sampled.gripper_command[0].detach().float().cpu().numpy()
        )
    return nodes


def install_sampling_trace(model: torch.nn.Module, current: dict[str, Any]):
    saved: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, fn: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)

    old_integrate = sampling_runtime._integrate_cache

    def integrate_wrapper(*args: Any, _old=old_integrate, **kwargs: Any):
        pass_role = str(kwargs.get("pass_role", "unknown"))
        label = current.get("_sampling_label")
        current["_sampling_stage"] = (
            f"{label}:{pass_role}" if label else pass_role
        )
        current["_sampling_call_index"] = 0
        current["_sampling_integration_index"] = int(
            current.get("_sampling_integration_count", 0)
        )
        current["_sampling_integration_count"] = (
            current["_sampling_integration_index"] + 1
        )
        return _old(*args, **kwargs)

    patch(sampling_runtime, "_integrate_cache", integrate_wrapper)
    old_refine = sampling_runtime.refine_cached_world

    def refine_wrapper(*args: Any, _old=old_refine, **kwargs: Any):
        previous = current.get("_sampling_stage")
        label = current.get("_sampling_label")
        current["_sampling_stage"] = (
            f"{label}:world_refine" if label else "world_refine"
        )
        try:
            return _old(*args, **kwargs)
        finally:
            current["_sampling_stage"] = previous or "world_refine_done"

    patch(sampling_runtime, "refine_cached_world", refine_wrapper)
    old_velocity = model.velocity

    def velocity_wrapper(*args: Any, _old=old_velocity, **kwargs: Any):
        stage = str(current.get("_sampling_stage", "unscoped"))
        call_index = int(current.get("_sampling_call_index", 0))
        current["_sampling_call_index"] = call_index + 1
        current["_sampling_step"] = call_index
        time = kwargs.get("time")
        if isinstance(time, torch.Tensor):
            current["_sampling_time"] = float(time.detach().float().reshape(-1)[0])
        output = _old(*args, **kwargs)
        if hasattr(output, "bottom"):
            _record(current, "physical_velocity", output.bottom.physical_velocity)
            tensors = getattr(output.bottom, "decoder_tensors", {})
            if isinstance(tensors, dict):
                for name in (
                    "pred_velocity", "evidence_mmd_it_prefix_pred_velocity",
                    "evidence_mmd_it_action_update",
                    "evidence_mmd_it_attention_update",
                    "evidence_mmd_it_ffn_update",
                    "evidence_mmd_it_self_update",
                    "evidence_mmd_it_evidence_update",
                ):
                    _record(current, "decoder_" + name, tensors.get(name))
            current["velocity_metrics"] = _scalar_metrics(
                getattr(output, "metrics", None)
            )
        events = current.get("_trace_events")
        if isinstance(events, list):
            events.append({
                "stage": stage,
                "integration_index": int(
                    current.get("_sampling_integration_index", -1)
                ),
                "integration_call_index": call_index,
                "time": current.get("_sampling_time"),
                "node_summaries": _trace_summary(current),
                "metrics": dict(current.get("velocity_metrics", {})),
            })
        return output

    patch(model, "velocity", velocity_wrapper)
    return saved


def install_deep_wrappers(
    model: torch.nn.Module,
    current: dict[str, Any],
    *,
    original_decoder_read: Any,
):
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
                    _record(
                        current, "p3_projected_" + name,
                        coord_obj.sources[name](value),
                    )
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
        if args and hasattr(args[0], "value_tokens"):
            _record(current, "bottom_organizer_view_values", args[0].value_tokens)
            _record(current, "bottom_organizer_view_intent", args[0].intent_tokens)
        out = _old(*args, **kwargs)
        if isinstance(out, dict):
            for name in (
                "condition_hidden", "latent_token", "global_condition",
                "time_hidden", "scan", "intent_context",
            ):
                _record(current, "bottom_organizer_" + name, out.get(name))
        return out

    patch(decoder.organizer, "forward", organizer_forward)

    route_state: dict[str, Any] = {
        "p3_lane": 0, "detail_call": 0, "precision_pending": False,
    }
    p3_reader = getattr(decoder, "policy_delta_attnres", None)
    detail_reader = getattr(decoder, "protected_detail_basis_attnres", None)
    if p3_reader is None:
        raise RuntimeError("deep probe could not locate P3 policy-delta reader")
    old_p3_forward = p3_reader.forward
    old_detail_forward = (
        detail_reader.forward if detail_reader is not None else None
    )

    def p3_forward(*args: Any, _old=old_p3_forward, **kwargs: Any):
        out = _old(*args, **kwargs)
        lane = int(route_state["p3_lane"])
        _record(current, f"bottom_p3_lane_{lane}_read", out[0])
        route_state["p3_lane"] = lane + 1
        return out

    patch(p3_reader, "forward", p3_forward)
    if detail_reader is not None:

        def detail_forward(*args: Any, _old=old_detail_forward, **kwargs: Any):
            out = _old(*args, **kwargs)
            name = (
                "bottom_p1_precision_read"
                if route_state["precision_pending"]
                else "bottom_protected_detail_read"
            )
            route_state["precision_pending"] = False
            route_state["detail_call"] = int(route_state["detail_call"]) + 1
            _record(current, name, out[0])
            return out

        patch(detail_reader, "forward", detail_forward)

    def policy_delta_forward(
        action_query: torch.Tensor,
        bank: Any,
        *,
        collect_diagnostics: bool = True,
        _old=original_decoder_read,
    ):
        _record(current, "bottom_role_value_all", bank.values)
        for index, name in enumerate(tuple(getattr(bank, "source_names", ()))):
            if index < int(bank.values.shape[1]):
                _record(
                    current, "bottom_role_value_" + str(name),
                    bank.values[:, index],
                )
        _record(current, "bottom_role_protected_detail", bank.protected_detail)
        _record(
            current, "bottom_role_protected_precision",
            bank.protected_policy_precision,
        )
        route_state["p3_lane"] = 0
        route_state["detail_call"] = 0
        route_state["precision_pending"] = (
            bank.protected_policy_precision is not None
        )
        out = _old(
            action_query, bank, collect_diagnostics=collect_diagnostics
        )
        _record(current, "bottom_policy_bridge_combined_update", out[0])
        _record(current, "bottom_protected_detail_update", out[1])
        return out

    patch(decoder, "_read_policy_delta_bank", policy_delta_forward)
    for index, block in enumerate(decoder.blocks):
        old = block.forward

        def block_forward(*args: Any, _old=old, _index=index, **kwargs: Any):
            action = kwargs.get("action", args[0] if args else None)
            evidence = kwargs.get(
                "evidence_tokens", args[1] if len(args) > 1 else None
            )
            condition = kwargs.get(
                "global_condition", args[2] if len(args) > 2 else None
            )
            key = f"_bottom_block_{_index}_calls"
            call_index = int(current.get(key, 0))
            current[key] = call_index + 1
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


def _select_physical_pair(
    cache: Any,
    *,
    min_coordinate_separation: float,
) -> dict[str, Any]:
    belief = cache.top.belief
    binding = cache.top.intent.target_binding
    camera_weights = (
        belief.camera_validity.float() * belief.camera_support.float()
    )
    denominator = camera_weights.sum(dim=2).clamp_min(1.0e-8)
    coordinates = (
        (belief.camera_coordinates.float() * camera_weights).sum(dim=2)
        / denominator
    )[0].detach().float()
    legal = binding.supported[0].to(dtype=torch.bool)
    legal = (
        legal
        & (denominator[0, :, 0] > 1.0e-8)
        & (belief.validity[0, :, 0] > 0)
        & torch.isfinite(coordinates).all(dim=-1)
    )
    candidates = torch.where(legal)[0]
    if int(candidates.numel()) < 2:
        raise RuntimeError("fewer than two supported finite physical K slots")
    best: tuple[float, int, int] | None = None
    for i in range(int(candidates.numel())):
        for j in range(i + 1, int(candidates.numel())):
            first, second = int(candidates[i]), int(candidates[j])
            distance = float(
                torch.linalg.vector_norm(
                    coordinates[first] - coordinates[second]
                ).item()
            )
            candidate = (distance, first, second)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    distance, first, second = best
    if distance < float(min_coordinate_separation):
        raise RuntimeError(
            "maximum supported pair separation "
            f"{distance} is below {min_coordinate_separation}"
        )
    return {
        "pair": [first, second],
        "coordinates": coordinates.detach().cpu().tolist(),
        "pair_coordinates": [
            coordinates[first].detach().cpu().tolist(),
            coordinates[second].detach().cpu().tolist(),
        ],
        "coordinate_separation": distance,
        "candidate_indices": [
            int(value) for value in candidates.detach().cpu()
        ],
        "selection": "farthest_supported_coordinates",
        "min_coordinate_separation": float(min_coordinate_separation),
    }


def _swap_wrapper(base_forward: Any, current: dict[str, Any], pair_info: dict[str, Any]):
    first, second = (int(value) for value in pair_info["pair"])

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
            self, task, objects, supported,
            history=history, view_support=view_support,
        )
        if int(result.log_probability.shape[0]) != 1:
            raise ValueError("binding swap requires batch size one")
        if int(result.log_probability.shape[1] - 1) <= max(first, second):
            raise ValueError("selected pair is outside binding K axis")
        if not bool(result.supported[0, first]) or not bool(
            result.supported[0, second]
        ):
            raise ValueError("selected pair is not legal in the swapped call")
        permutation = torch.arange(
            result.log_probability.shape[1] - 1,
            device=result.log_probability.device,
        )
        permutation[first], permutation[second] = second, first
        real = result.log_probability[..., :-1][:, permutation]
        swapped = replace(
            result,
            log_probability=torch.cat(
                (real, result.log_probability[..., -1:]), dim=-1
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
    policy: Any,
    history: Any,
    instruction: str,
    current: dict[str, Any],
    binder_forward: Any,
    model: torch.nn.Module,
    swap: bool,
    swap_pair: dict[str, Any] | None,
    mode: str,
    min_coordinate_separation: float,
) -> dict[str, Any]:
    policy.reset()
    current.clear()
    trace_events: list[dict[str, Any]] = []
    current["_trace_events"] = trace_events
    TaskConditionedTargetBinder.forward = binder_forward
    _, online = policy.act_with_input(history, instruction)
    # act_with_input performs the policy's ordinary deployment sample before
    # returning the OnlinePolicyInput. It is not one of the measured paths;
    # discard its trace and start a fresh event scope for the explicit cache
    # and sampler calls below.
    trace_events.clear()
    current.clear()
    current["_trace_events"] = trace_events
    current["_sampling_integration_count"] = 0
    config = policy.bundle.config
    with torch.no_grad():
        current["_sampling_stage"] = "encode_online"
        cache, _ = sampling_runtime.deployment_cache(
            model, online, config, collect_diagnostics=True,
            dtype=torch.bfloat16,
        )
        current.update(intent_fields(cache.top.intent))
        pair_info = (
            _select_physical_pair(
                cache,
                min_coordinate_separation=min_coordinate_separation,
            )
            if swap_pair is None else dict(swap_pair)
        )
        current["_pair_info"] = pair_info
        static_summaries = {
            name: summary(value) for name, value in _node_arrays(current).items()
        }
        noise = torch.randn(
            1, config.dimensions.action_horizon,
            model.outlet_adapter.physical_dim,
            device=online.device, dtype=torch.float32,
            generator=torch.Generator(device=online.device).manual_seed(12345),
        )
        paths: dict[str, Any] = {}
        if mode in {"fixed", "both"}:
            start = len(trace_events)
            current["_sampling_label"] = "fixed_w_refined"
            current["_sampling_stage"] = "fixed_w_refined"
            fixed = sampling_runtime.sample_cached_action(
                model, cache, config, initial_physical_noise=noise,
                collect_diagnostics=True, dtype=torch.bfloat16,
                pass_role="refined",
            )
            paths["fixed_w_refined"] = {
                "event_start": start, "event_end": len(trace_events),
                "nodes": _final_nodes(current, fixed),
                "metrics": _scalar_metrics(fixed.metrics),
            }
        if mode in {"full", "both"}:
            start = len(trace_events)
            current["_sampling_label"] = "full_proposal"
            full, refined_cache = (
                sampling_runtime.sample_refined_cached_action_with_cache(
                    model, cache, config,
                    initial_physical_noise=noise,
                    collect_diagnostics=True, dtype=torch.bfloat16,
                )
            )
            paths["full_proposal_w_refined"] = {
                "event_start": start, "event_end": len(trace_events),
                "nodes": _final_nodes(current, full),
                "metrics": _scalar_metrics(full.metrics),
                "refined_cache_rebuilt": refined_cache is not cache,
            }
    selected_path = (
        "full_proposal_w_refined"
        if "full_proposal_w_refined" in paths else "fixed_w_refined"
    )
    selected = paths[selected_path]
    path_comparisons: dict[str, Any] = {}
    if "fixed_w_refined" in paths and "full_proposal_w_refined" in paths:
        path_comparisons["fixed_vs_full_action"] = compare(
            paths["fixed_w_refined"]["nodes"]["final_native_action"],
            paths["full_proposal_w_refined"]["nodes"]["final_native_action"],
        )
        path_comparisons["fixed_vs_full_physical_field"] = compare(
            paths["fixed_w_refined"]["nodes"]["final_physical_field"],
            paths["full_proposal_w_refined"]["nodes"]["final_physical_field"],
        )
    return {
        "swap": bool(swap),
        "pair_info": pair_info,
        "selected_path": selected_path,
        "nodes": selected["nodes"],
        "path_nodes": paths,
        "path_comparisons": path_comparisons,
        "paths": {
            name: {key: value for key, value in path.items() if key != "nodes"}
            for name, path in paths.items()
        },
        "static_summaries": static_summaries,
        "trace_events": trace_events,
        "metrics": selected["metrics"],
    }


def _compare_path_actions(left: dict[str, Any], right: dict[str, Any]):
    return {
        name: compare(
            left["path_nodes"][name]["nodes"]["final_native_action"],
            right["path_nodes"][name]["nodes"]["final_native_action"],
        )
        for name in sorted(set(left["path_nodes"]) & set(right["path_nodes"]))
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(
        args.checkpoint, device=device, t5_condition=args.t5_condition, seed=0
    )
    model = policy.bundle.model
    current: dict[str, Any] = {}
    base_saved = install_wrappers(model, current)
    base_read = next(
        old for owner, name, old in base_saved
        if name == "_read_policy_delta_bank"
    )
    deep_saved = install_deep_wrappers(
        model, current, original_decoder_read=base_read
    )
    sampling_saved = install_sampling_trace(model, current)
    installed_binder_forward = TaskConditionedTargetBinder.forward
    records: list[dict[str, Any]] = []
    try:
        for layout in args.layouts:
            history = history_with_world(args.observation_dir / f"{layout}.npz")
            for instruction in args.instructions:
                baseline = _run_one(
                    policy=policy, history=history, instruction=instruction,
                    current=current, binder_forward=installed_binder_forward,
                    model=model, swap=False, swap_pair=None, mode=args.mode,
                    min_coordinate_separation=args.min_coordinate_separation,
                )
                repeats = []
                for repeat_index in range(max(0, int(args.repeats) - 1)):
                    repeated = _run_one(
                        policy=policy, history=history, instruction=instruction,
                        current=current, binder_forward=installed_binder_forward,
                        model=model, swap=False,
                        swap_pair=baseline["pair_info"], mode=args.mode,
                        min_coordinate_separation=args.min_coordinate_separation,
                    )
                    repeats.append({
                        "repeat_index": repeat_index + 1,
                        "selected_path": repeated["selected_path"],
                        "action_error_vs_baseline": _compare_path_actions(
                            baseline, repeated
                        ),
                        "trace_event_count": len(repeated["trace_events"]),
                    })
                swapped_binder = _swap_wrapper(
                    installed_binder_forward, current, baseline["pair_info"]
                )
                swapped = _run_one(
                    policy=policy, history=history, instruction=instruction,
                    current=current, binder_forward=swapped_binder,
                    model=model, swap=True,
                    swap_pair=baseline["pair_info"], mode=args.mode,
                    min_coordinate_separation=args.min_coordinate_separation,
                )
                before = np.asarray(baseline["nodes"]["s_binding_mass"])[0]
                after = np.asarray(swapped["nodes"]["s_binding_mass"])[0]
                pair = [int(x) for x in baseline["pair_info"]["pair"]]
                names = sorted(
                    name
                    for name in set(baseline["nodes"]).intersection(swapped["nodes"])
                    if "_call_" not in name
                )
                records.append({
                    "layout": layout,
                    "instruction": instruction,
                    "pair": pair,
                    "pair_info": baseline["pair_info"],
                    "measurement": {
                        "mode": args.mode,
                        "fixed_w_refined_is_one_pass": True,
                        "full_proposal_w_refined_is_two_pass": True,
                        "repeats": int(args.repeats),
                        "trace_event_count_baseline": len(baseline["trace_events"]),
                        "trace_event_count_swapped": len(swapped["trace_events"]),
                    },
                    "binding_invariants": {
                        "real_mass_sum_before": float(before.sum()),
                        "real_mass_sum_after": float(after.sum()),
                        "real_mass_sum_abs_diff": float(abs(before.sum() - after.sum())),
                        "null_mass_before": float(
                            np.asarray(
                                baseline["nodes"]["s_binding_null_mass"]
                            ).reshape(-1)[0]
                        ),
                        "null_mass_after": float(
                            np.asarray(
                                swapped["nodes"]["s_binding_null_mass"]
                            ).reshape(-1)[0]
                        ),
                        "null_mass_abs_diff": float(abs(
                            np.asarray(
                                baseline["nodes"]["s_binding_null_mass"]
                            ).reshape(-1)[0]
                            - np.asarray(
                                swapped["nodes"]["s_binding_null_mass"]
                            ).reshape(-1)[0]
                        )),
                        "sorted_real_mass_linf": float(
                            np.max(np.abs(np.sort(before) - np.sort(after)))
                        ),
                        "entropy_before": _entropy(before),
                        "entropy_after": _entropy(after),
                        "entropy_abs_diff": float(
                            abs(_entropy(before) - _entropy(after))
                        ),
                        "pair_exchange_linf": float(max(
                            abs(after[pair[0]] - before[pair[1]]),
                            abs(after[pair[1]] - before[pair[0]]),
                        )),
                    },
                    "baseline": {
                        "selected_path": baseline["selected_path"],
                        "path_measurements": baseline["paths"],
                        "path_comparisons": baseline["path_comparisons"],
                        "static_summaries": baseline["static_summaries"],
                        "trace_events": baseline["trace_events"],
                    },
                    "swapped": {
                        "selected_path": swapped["selected_path"],
                        "path_measurements": swapped["paths"],
                        "path_comparisons": swapped["path_comparisons"],
                        "static_summaries": swapped["static_summaries"],
                        "trace_events": swapped["trace_events"],
                    },
                    "repeat_checks": repeats,
                    "node_differences": {
                        name: compare(swapped["nodes"][name], baseline["nodes"][name])
                        for name in names
                    },
                    "node_summaries": {
                        "baseline": {
                            name: summary(baseline["nodes"][name]) for name in names
                        },
                        "swapped": {
                            name: summary(swapped["nodes"][name]) for name in names
                        },
                    },
                })
    finally:
        for owner, name, old in reversed(sampling_saved):
            setattr(owner, name, old)
        for owner, name, old in reversed(deep_saved):
            setattr(owner, name, old)
        restore_wrappers(base_saved)
    result = {
        "schema": "clearvla-target-binding-measurement-v2",
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
            "operation": "swap_farthest_supported_physical_coordinate_pair",
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
    parser.add_argument("--mode", choices=("fixed", "full", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-coordinate-separation", type=float, default=0.15)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "records": len(result["records"])}, indent=2))


if __name__ == "__main__":
    main()
