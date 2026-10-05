#!/usr/bin/env python3
"""Fixed-checkpoint short audit for the G slot-identity repair.

The probe keeps the deployment ABI intact and records only compact summaries:
G competition/read trajectories, owner mass, S/P1/P2 interval variation,
action output, P2 softmax support, and one backward finite-gradient check.
It uses the same fixed checkpoint and four causal CALVIN layouts/instructions.
"""
from __future__ import annotations
import argparse, gc, json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from clearvla.mainline.model import compiler as compiler_module
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from scripts.probe_calvin_internal_layers_npz import _history

INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
)
LAYOUTS = (
    "standard",
    "blue_on_slider_left",
    "blue_on_slider_right",
    "red_on_slider_left",
)

def _history_with_world(path: Path):
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

def _cos(left: Any, right: Any) -> float:
    a = torch.as_tensor(left).float().reshape(1, -1)
    b = torch.as_tensor(right).float().reshape(1, -1)
    return float(F.cosine_similarity(a, b, dim=-1).item())

def _rms(value: Any) -> float:
    x = torch.as_tensor(value).float()
    return float(x.square().mean().sqrt().item())

def _variation(value: Any, dim: int = 1) -> float:
    x = torch.as_tensor(value).float()
    return float((x - x.mean(dim=dim, keepdim=True)).square().mean().sqrt().item())

def _pair_slots(value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().float()
    a, b = x[:, 1], x[:, 3]
    return {
        "cosine_k2_k4": _cos(a, b),
        "slot_l2": [float(v) for v in x.reshape(x.shape[0], x.shape[1], -1).norm(dim=-1).mean(0).tolist()],
        "finite": bool(torch.isfinite(x).all()),
    }

def _pair_read(value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().float()
    a, b = x[:, 1], x[:, 3]
    return {
        "cosine_k2_k4": _cos(a, b),
        "read_rms": _rms(x),
        "finite": bool(torch.isfinite(x).all()),
    }

def _scalar_metrics(mapping: Any, prefixes: tuple[str, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    if not isinstance(mapping, dict):
        return result
    for key, value in mapping.items():
        if not any(str(key).startswith(prefix) for prefix in prefixes):
            continue
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[str(key)] = float(value.detach().float().item())
    return result

def _tensor_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "rms": _rms(value),
        "variation": _variation(value, 1) if value.ndim >= 2 and value.shape[1] > 1 else 0.0,
        "finite": bool(torch.isfinite(value).all()),
    }

def _instruction_diff(values: list[dict[str, Any]], key: str) -> dict[str, float]:
    first = values[0][key]
    result: dict[str, float] = {}
    for row in values[1:]:
        cur = row[key]
        result[row["instruction"]] = {
            "rmse": float(np.sqrt(np.mean(np.square(np.asarray(cur, dtype=np.float64) - np.asarray(first, dtype=np.float64))))),
            "cosine": _cos(cur, first),
        }
    return result

def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
        seed=0,
    )
    original_competition = DenseObjectGrounder._competition
    records: list[dict[str, Any]] = []
    current_calls: list[dict[str, Any]] = []
    def wrapped(self, *call_args, **call_kwargs):
        slots = call_args[0].detach()
        output = original_competition(self, *call_args, **call_kwargs)
        owner, mass, null_mass, read, owner_log, read_log = output
        current_calls.append({
            "slots": _pair_slots(slots),
            "read": _pair_read(read),
            "owner_finite": bool(torch.isfinite(owner).all()),
            "mass_finite": bool(torch.isfinite(mass).all()),
            "null_mass_rms": _rms(null_mass),
        })
        return output
    DenseObjectGrounder._competition = wrapped
    try:
        with torch.no_grad():
            for layout in args.layouts:
                history = _history_with_world(args.observation_dir / f"{layout}.npz")
                layout_rows: list[dict[str, Any]] = []
                for instruction in args.instructions:
                    current_calls.clear()
                    policy.reset()
                    _, online = policy.act_with_input(history, instruction)
                    cache, static = deployment_cache(
                        policy.bundle.model, online, policy.bundle.config,
                        collect_diagnostics=True,
                    )
                    noise = torch.randn(
                        1,
                        policy.bundle.config.dimensions.action_horizon,
                        policy.bundle.model.outlet_adapter.physical_dim,
                        device=device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=device).manual_seed(12345),
                    )
                    p2_calls: list[dict[str, Any]] = []
                    original_softmax = compiler_module._safe_masked_softmax
                    def observe_softmax(logit, support, *, dim):
                        probability = original_softmax(logit, support, dim=dim)
                        p2_calls.append({
                            "shape": list(probability.shape),
                            "support_fraction": float(support.float().mean().item()),
                            "probability_rms": _rms(probability),
                            "finite": bool(torch.isfinite(probability).all()),
                        })
                        return probability
                    compiler_module._safe_masked_softmax = observe_softmax
                    try:
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            output = policy.bundle.model.velocity(
                                cache,
                                noisy_action_field=noise,
                                time=torch.full((1,), 0.4, device=device),
                                collect_diagnostics=True,
                            )
                    finally:
                        compiler_module._safe_masked_softmax = original_softmax
                    intent = cache.top.intent
                    belief = cache.top.belief
                    dynamics = cache.top.predicted_dynamics
                    physical = output.bottom.physical_velocity.detach().float().cpu().numpy()
                    row = {
                        "instruction": instruction,
                        "g_competition_calls": list(current_calls),
                        "g_final_slot": current_calls[-1]["slots"],
                        "g_final_read": current_calls[-1]["read"],
                        "g_mass_rms": _rms(belief.validity),
                        "g_mass_by_slot": [float(x) for x in belief.validity.detach().float().reshape(-1).tolist()],
                        "g_mass_finite": bool(torch.isfinite(belief.validity).all()),
                        "s_public_interval": _tensor_summary(intent.public_interval_carrier),
                        "s_public_interval_value": intent.public_interval_carrier.detach().float().cpu().numpy(),
                        "s_object_attention": _tensor_summary(intent.interval_object_attention),
                        "s_object_attention_value": intent.interval_object_attention.detach().float().cpu().numpy(),
                        "s_typed_value": _tensor_summary(intent.typed_relevance_value),
                        "p1_metrics": _scalar_metrics(output.metrics, ("gradient_tensor_p1", "p1_")),
                        "p2_metrics": _scalar_metrics(output.metrics, ("object_p2_", "object_consequence_")),
                        "g_metrics": _scalar_metrics(static, ("grounding_g1_", "grounding_g2_", "grounding_g3_")),
                        "p2_softmax": p2_calls,
                        "semantic_delta": dynamics.semantic_delta.detach().float().cpu().numpy(),
                        "physical_velocity": physical,
                        "physical_velocity_rms": float(np.sqrt(np.mean(np.square(physical)))),
                        "physical_velocity_finite": bool(np.isfinite(physical).all()),
                        "output_metrics": _scalar_metrics(output.metrics, ("evidence_", "controlled_", "object_")),
                    }
                    layout_rows.append(row)
                row0 = layout_rows[0]
                layout_diff = {
                    "public_interval_vs_first": _instruction_diff(layout_rows, "s_public_interval_value"),
                    "object_attention_vs_first": _instruction_diff(layout_rows, "s_object_attention_value"),
                    "semantic_delta_vs_first": _instruction_diff(layout_rows, "semantic_delta"),
                    "physical_velocity_vs_first": _instruction_diff(layout_rows, "physical_velocity"),
                }
                records.append({
                    "layout": layout,
                    "rows": [
                        {k: v for k, v in row.items() if not k.endswith("_value") and k not in {"semantic_delta", "physical_velocity"}}
                        for row in layout_rows
                    ],
                    "instruction_differences": layout_diff,
                })
        # One gradient-only read from the last cache; this checks backward
        # finiteness without changing any checkpoint or optimizer state.
        grad_report: dict[str, Any] = {}
        policy.bundle.model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            grad_output = policy.bundle.model.velocity(
                cache,
                noisy_action_field=noise,
                time=torch.full((1,), 0.4, device=device),
                collect_diagnostics=True,
            )
            grad_loss = grad_output.bottom.physical_velocity.float().square().mean()
        grad_loss.backward()
        for name, parameter in policy.bundle.model.named_parameters():
            if any(token in name for token in ("grounder.slot_seed", "grounder.gru", "grounder.update_ffn", "intent.")):
                if parameter.grad is not None:
                    grad_report[name] = {
                        "rms": float(parameter.grad.detach().float().square().mean().sqrt().item()),
                        "finite": bool(torch.isfinite(parameter.grad).all()),
                    }
        result = {
            "schema": "clearvla-g-slot-identity-short-v1",
            "checkpoint": str(args.checkpoint),
            "source": "remote repair worktree",
            "layouts": records,
            "gradient": {
                "loss": float(grad_loss.detach().float().item()),
                "finite": bool(torch.isfinite(grad_loss).all()),
                "parameters": grad_report,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        DenseObjectGrounder._competition = original_competition

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    parser.add_argument("--instructions", nargs="+", default=list(INSTRUCTIONS))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output),
        "layouts": [item["layout"] for item in result["layouts"]],
        "gradient_parameter_count": len(result["gradient"]["parameters"]),
    }, indent=2))

if __name__ == "__main__":
    main()
