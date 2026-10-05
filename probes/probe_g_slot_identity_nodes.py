#!/usr/bin/env python3
"""Node-level instruction-difference trace for the G/S/P1/P2/P3 path.

This is an instrumentation-only probe.  It uses one fixed checkpoint, one
standard CALVIN observation and the same noise/time for four instructions.  It
records tensors at every declared owner boundary and reports pairwise RMSE,
cosine and relative RMSE so attenuation can be localized without changing the
model or checkpoint.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from clearvla.mainline.model.compiler import ObjectFutureEffectReader, ObjectPolicyPlanCompiler, ZeroPreservingObjectConsequence
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from scripts.probe_calvin_internal_layers_npz import _history

DEFAULT_INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
)


def history_with_world(path: Path):
    base = _history(path)
    world = ExecutedWorldSnapshot(
        anchor_index=0,
        rgb_history=base.rgb_history,
        state=base.state.copy(),
        action_state=base.action_state.copy(),
        commands=base.executed_action_history[:4].copy(),
        visual_offsets=np.zeros(3, dtype=np.int64),
        observed=True,
    )
    return replace(
        base,
        time_index=4,
        previous_state=base.state.copy(),
        executed_world=world,
    )


def array(value: Any) -> np.ndarray | None:
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().float().cpu().numpy()


def finite(value: Any) -> bool:
    x = np.asarray(value)
    return bool(np.isfinite(x).all())


def rms(value: Any) -> float:
    x = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x))))


def summary(value: Any) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(x.shape),
        "rms": rms(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "finite": bool(np.isfinite(x).all()),
    }


def compare(left: Any, right: Any) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    d = a - b
    rmse = float(np.sqrt(np.mean(np.square(d))))
    ar = float(np.sqrt(np.mean(np.square(a))))
    br = float(np.sqrt(np.mean(np.square(b))))
    af = a.reshape(-1)
    bf = b.reshape(-1)
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    cosine = float(np.dot(af, bf) / denom) if denom > 0 else 1.0
    return {
        "rmse": rmse,
        "cosine": cosine,
        "left_rms": ar,
        "right_rms": br,
        "relative_rmse_left": rmse / max(ar, 1e-12),
    }


def scalar_metrics(mapping: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(mapping, dict):
        return out
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            name = str(key)
            if name.startswith(("flow_jepa_", "p1_", "object_p2_", "object_p3_", "object_consequence_", "gradient_tensor_p1")):
                out[name] = float(value.detach().float().item())
    return out


def binding_fields(state: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    binding = getattr(state, "target_binding", None)
    if binding is not None:
        out["binding_mass"] = array(binding.mass)
        out["binding_null_mass"] = array(binding.null_mass)
        out["binding_log_probability"] = array(binding.log_probability)
    return out


def intent_fields(state: Any) -> dict[str, np.ndarray]:
    names = (
        "protected_goal_set",
        "public_interval_carrier",
        "policy_interval_context",
        "object_tokens",
        "interval_object_attention",
        "typed_relevance_mass",
        "typed_relevance_value",
        "typed_common_value",
        "typed_interval_residual_value",
        "typed_policy_components",
        "target_object_address_logit",
    )
    out = {name: value for name in ("s_" + name for name in names) if False}
    result: dict[str, np.ndarray] = {}
    for name in names:
        value = array(getattr(state, name, None))
        if value is not None:
            result["s_" + name] = value
    result.update({"s_" + name: value for name, value in binding_fields(state).items()})
    return result


def capture_selected(selected: Any) -> dict[str, np.ndarray]:
    names = (
        "semantic_common_value",
        "semantic_residual_value",
        "semantic_value",
        "geometry_common_value",
        "geometry_residual_value",
        "geometry_value",
        "selected_s_context",
        "selected_target_value",
        "key",
        "support",
    )
    result: dict[str, np.ndarray] = {}
    for name in names:
        value = array(getattr(selected, name, None))
        if value is not None:
            result["p2_selected_" + name] = value
    return result


def capture_effect(effect: Any, prefix: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in ("semantic", "geometry"):
        value = array(getattr(effect, name, None))
        if value is not None:
            result[f"{prefix}_{name}"] = value
    return result


def install_wrappers(model: torch.nn.Module, current: dict[str, Any], *, phase_zero: bool = False, binder_task_zero: bool = False) -> list[tuple[Any, str, Any]]:
    """Patch bound instance methods; return (owner, name, old) for restore."""
    saved: list[tuple[Any, str, Any]] = []

    def patch(owner: Any, name: str, fn: Any) -> None:
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)

    from clearvla.mainline.model.intent import StatelessObjectIntentOrganizer
    organizer_cls = StatelessObjectIntentOrganizer
    old = organizer_cls.forward
    def organizer_forward(self, *args: Any, _old=old, **kwargs: Any):
        state = _old(self, *args, **kwargs)
        current.update(intent_fields(state))
        return state
    patch(organizer_cls, "forward", organizer_forward)

    from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
    binder_cls = TaskConditionedTargetBinder
    old = binder_cls.forward
    def binder_forward(self, task, objects, supported, *, history, view_support=None, _old=old):
        per_view = objects.ndim == 4
        if per_view:
            legal = view_support & supported[..., None]
            safe = torch.where(legal[..., None], objects, 0.0).flatten(1, 2)
        else:
            safe = torch.where(supported[..., None], objects, 0.0)
        obj = self.objects(safe)
        query = self.query_norm(obj + self.history_query(history)[:, None])
        task_value = self.task_value(task)
        read, _ = self.task_read(query, self.task_norm(task_value), task_value, need_weights=False)
        object_score = self.score(read * torch.tanh(self.compatibility(obj)))[..., 0]
        task_context = task_value.mean(1)[:, None].expand(-1, obj.shape[1], -1)
        task_object_score = self.task_object_score(torch.cat((obj * task_context, obj * torch.tanh(task_context)), dim=-1))[..., 0]
        current["binder_task"] = array(task)
        current["binder_object_score"] = array(object_score)
        current["binder_task_object_score"] = array(task_object_score)
        current["binder_task_context"] = array(task_context)
        if binder_task_zero:
            task = torch.zeros_like(task)
        return _old(self, task, objects, supported, history=history, view_support=view_support)
    patch(binder_cls, "forward", binder_forward)

    p1 = model.p1
    old = p1.build_static
    def p1_build(*args: Any, _old=old, **kwargs: Any):
        if phase_zero:
            kwargs = dict(kwargs)
            kwargs["phase_context"] = torch.zeros_like(kwargs["phase_context"])
        state, metrics = _old(*args, **kwargs)
        value = array(state.protected_detail)
        if value is not None:
            current["p1_static_factual"] = value
        current["p1_static_metrics"] = scalar_metrics(metrics)
        return state, metrics
    patch(p1, "build_static", p1_build)

    old = p1.update_dynamic
    def p1_update(*args: Any, _old=old, **kwargs: Any):
        state, metrics = _old(*args, **kwargs)
        for name in ("factual_base", "policy_query_residual"):
            value = array(getattr(state, name))
            if value is not None:
                current["p1_dynamic_" + name] = value
        current["p1_dynamic_metrics"] = scalar_metrics(metrics)
        return state, metrics
    patch(p1, "update_dynamic", p1_update)

    effect_reader = model.policy_compiler.effect_reader
    old = effect_reader.forward_candidate
    def effect_candidate(*args: Any, _old=old, **kwargs: Any):
        if args:
            value = array(args[0])
            if value is not None:
                current["p2_query"] = value
        out = _old(*args, **kwargs)
        current["p2_effect_metrics"] = scalar_metrics(out[1])
        current.update(capture_effect(out[0], "p2_effect"))
        return out
    patch(effect_reader, "forward_candidate", effect_candidate)

    old = effect_reader.spatial_select
    def spatial_select(*args: Any, _old=old, **kwargs: Any):
        selected, metrics = _old(*args, **kwargs)
        current.update(capture_selected(selected))
        current["p2_spatial_metrics"] = scalar_metrics(metrics)
        return selected, metrics
    patch(effect_reader, "spatial_select", spatial_select)

    old = effect_reader.temporal_terminal
    def temporal_terminal(*args: Any, _old=old, **kwargs: Any):
        out = _old(*args, **kwargs)
        current.update(capture_effect(out[0], "p2_terminal_effect"))
        return out
    patch(effect_reader, "temporal_terminal", temporal_terminal)

    consequence = model.policy_compiler.consequence
    old = consequence.forward
    def consequence_forward(*args: Any, _old=old, **kwargs: Any):
        state, metrics = _old(*args, **kwargs)
        for name in ("factual_base", "protected_consequence"):
            value = array(getattr(state, name))
            if value is not None:
                current["p2_consequence_" + name] = value
        current.update(capture_effect(state.effect, "p2_consequence_effect"))
        current.update(capture_effect(state.interaction, "p2_consequence_interaction"))
        current["p2_consequence_metrics"] = scalar_metrics(metrics)
        return state, metrics
    patch(consequence, "forward", consequence_forward)

    planner = model.policy_compiler.plan_compiler
    old = planner.forward
    def planner_forward(*args: Any, _old=old, **kwargs: Any):
        bank, metrics = _old(*args, **kwargs)
        for name in ("protected_base", "protected_policy_precision", "temporal", "state_change"):
            value = array(getattr(bank, name))
            if value is not None:
                current["p3_" + name] = value
        current["p3_metrics"] = scalar_metrics(metrics)
        return bank, metrics
    patch(planner, "forward", planner_forward)

    return saved


def restore_wrappers(saved: list[tuple[Any, str, Any]]) -> None:
    for owner, name, old in reversed(saved):
        setattr(owner, name, old)


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
    saved = install_wrappers(model, current, phase_zero=args.phase_zero, binder_task_zero=args.binder_task_zero)
    records: list[dict[str, Any]] = []
    try:
        for layout in args.layouts:
            history = history_with_world(args.observation_dir / f"{layout}.npz")
            raw_rows: list[dict[str, Any]] = []
            for instruction in args.instructions:
                current.clear()
                policy.reset()
                _, online = policy.act_with_input(history, instruction)
                with torch.no_grad():
                    cache, static = deployment_cache(
                        model, online, policy.bundle.config, collect_diagnostics=True
                    )
                    current.update(intent_fields(cache.top.intent))
                    noise = torch.randn(
                        1,
                        policy.bundle.config.dimensions.action_horizon,
                        model.outlet_adapter.physical_dim,
                        device=device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(12345),
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        output = model.velocity(
                            cache,
                            noisy_action_field=noise,
                            time=torch.full((1,), 0.4, device=device),
                            collect_diagnostics=True,
                        )
                current["physical_velocity"] = array(output.bottom.physical_velocity)
                current["velocity_metrics"] = scalar_metrics(output.metrics)
                # The organizer wrapper runs during encode_online.  P1/P2/P3
                # wrappers run during velocity.  A missing key is a hard probe
                # failure, not silently treated as zero.
                required = ("s_public_interval_carrier", "p1_static_factual", "p1_dynamic_policy_query_residual", "p2_query", "p2_effect_semantic", "p2_consequence_protected_consequence", "p3_temporal", "physical_velocity")
                missing = [name for name in required if name not in current]
                if missing:
                    raise RuntimeError(f"node capture missing for {layout}/{instruction}: {missing}")
                raw_rows.append({
                    "instruction": instruction,
                    "nodes": {name: value for name, value in current.items() if isinstance(value, np.ndarray)},
                    "metrics": {
                        "static": current.get("p1_static_metrics", {}),
                        "dynamic": current.get("p1_dynamic_metrics", {}),
                        "p2": current.get("p2_effect_metrics", {}),
                        "consequence": current.get("p2_consequence_metrics", {}),
                        "p3": current.get("p3_metrics", {}),
                        "velocity": current.get("velocity_metrics", {}),
                        "g": scalar_metrics(static),
                    },
                })
            base = raw_rows[0]
            node_names = sorted(set(base["nodes"]).intersection(*(set(row["nodes"]) for row in raw_rows[1:])))
            rows = []
            for row in raw_rows:
                rows.append({
                    "instruction": row["instruction"],
                    "nodes": {name: summary(row["nodes"][name]) for name in sorted(row["nodes"])},
                    "metrics": row["metrics"],
                })
            differences = {}
            for row in raw_rows[1:]:
                differences[row["instruction"]] = {name: compare(row["nodes"][name], base["nodes"][name]) for name in node_names}
            records.append({"layout": layout, "rows": rows, "instruction_differences_vs_first": differences})
    finally:
        restore_wrappers(saved)
    result = {
        "schema": "clearvla-g-slot-identity-node-trace-v1",
        "checkpoint": str(args.checkpoint),
        "source": "fixed checkpoint / latest branch inference; instrumentation only",
        "interventions": {"phase_zero": bool(args.phase_zero), "binder_task_zero": bool(args.binder_task_zero)},
        "layouts": records,
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
    parser.add_argument("--phase-zero", action="store_true")
    parser.add_argument("--binder-task-zero", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"output": str(args.output), "layouts": [x["layout"] for x in result["layouts"]]}, indent=2))


if __name__ == "__main__":
    main()



















