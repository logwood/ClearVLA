#!/usr/bin/env python3
"""Low-memory read-only Jacobian probe on a fixed CALVIN observation.

This intentionally measures graph connectivity rather than a training metric:
one fixed observation/noise sample is encoded with gradients enabled, and
random VJPs are taken at the coarse proposal, P2 semantic effect, and bottom
physical velocity.  No optimizer or parameter ``.grad`` field is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

from scripts.probe_calvin_internal_layers_npz import _build_online, _history


def _stats(value: Tensor | None) -> dict[str, float | int]:
    if value is None:
        return {"present": 0, "l2": 0.0, "rms": 0.0, "max_abs": 0.0, "nonzero": 0}
    value = value.detach().float()
    return {
        "present": 1,
        "l2": float(torch.linalg.vector_norm(value)),
        "rms": float(value.square().mean().sqrt()),
        "max_abs": float(value.abs().amax()),
        "nonzero": int(torch.count_nonzero(value)),
    }


def _group_stats(grads: Iterable[Tensor | None], params: Iterable[nn.Parameter]) -> dict[str, float | int]:
    grads = [g.detach().float() for g in grads if g is not None]
    params = list(params)
    square = sum((g.square().sum() for g in grads), torch.zeros((), dtype=torch.float32))
    return {
        "parameter_tensors": len(params),
        "parameter_elements": sum(int(p.numel()) for p in params),
        "gradient_present_tensors": len(grads),
        "gradient_l2": float(square.sqrt()),
        "gradient_rms_over_parameters": float((square / max(sum(int(p.numel()) for p in params), 1)).sqrt()),
    }


def _module_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    organizer = model.intent.organizer
    reader = model.policy_compiler.effect_reader
    modules: dict[str, nn.Module] = {
        "goal_input": organizer.goal_input,
        "goal_read": organizer.goal_read,
        "goal_self": organizer.goal_self,
        "interval_goal": organizer.interval_goal,
        "typed_relevance_queries": organizer.typed_relevance_queries,
        "object_content": organizer.object_content,
        "object_semantic": organizer.object_semantic,
        "object_appearance": organizer.object_appearance,
        "object_geometry": organizer.object_geometry,
        "coarse_action": model.intent.coarse_action,
        "p2_typed_semantic_key": reader.typed_intent_key[0],
        "p2_typed_geometry_key": reader.typed_intent_key[1],
        "p2_semantic_value": reader.semantic_value,
    }
    return {
        name: tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
        for name, module in modules.items()
    }


def _vjp(loss: Tensor, values: tuple[Tensor, ...], *, retain_graph: bool) -> tuple[Tensor | None, ...]:
    return torch.autograd.grad(
        loss,
        values,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    policy = ClearVLACheckpointPolicy(
        args.checkpoint,
        device=device,
        dinov2_model=args.dinov2_model,
        dinov2_local_files_only=True,
        seed=0,
    )
    model = policy.bundle.model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    snapshot = Path(args.observation_dir) / f"{args.layout}.npz"
    history = _history(snapshot)
    instructions = tuple(args.instructions)
    dtype = resolve_compute_dtype(policy.bundle.config, None)
    autocast_enabled = device.type in {"cuda", "cpu"} and dtype in {torch.float16, torch.bfloat16}
    groups = _module_groups(model)
    records: dict[str, object] = {}
    for instruction_index, instruction in enumerate(instructions):
        online = _build_online(policy, history, instruction)
        # Do not use deployment_cache: it intentionally wraps encode_online in
        # no_grad.  This probe needs the language-to-owner graph intact.
        with torch.enable_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                cache, training_state, _ = model.encode_online(
                    online,
                    training_mask=False,
                    geometry_supervision=False,
                    collect_diagnostics=False,
                )
                horizon = int(policy.bundle.config.dimensions.action_horizon)
                noise = torch.Generator(device=device).manual_seed(12345)
                noisy = torch.randn(
                    1,
                    horizon,
                    model.outlet_adapter.physical_dim,
                    device=device,
                    dtype=torch.float32,
                    generator=noise,
                )
                output = model.velocity(
                    cache,
                    noisy_action_field=noisy,
                    time=torch.full((1,), 0.4, device=device, dtype=torch.float32),
                    collect_diagnostics=False,
                )
        intent = training_state.top.intent
        seams = {
            "protected_goal": intent.protected_goal_set,
            "public_interval_carrier": intent.public_interval_carrier,
            "policy_interval_context": intent.policy_interval_context,
            # P2 consumes these two stored nodes directly.  The convenience
            # ``typed_relevance_*`` properties reconstruct a fresh sum and
            # therefore are not valid autograd targets for this probe.
            "typed_common_mass": intent.typed_common_mass,
            "typed_interval_residual_mass": intent.typed_interval_residual_mass,
            "typed_common_value": intent.typed_common_value,
            "typed_interval_residual_value": intent.typed_interval_residual_value,
            "typed_relevance_mass": intent.typed_relevance_mass,
            "typed_relevance_value": intent.typed_relevance_value,
            "typed_policy_components": intent.typed_policy_components,
            "object_tokens": intent.object_tokens,
            "coarse_action": training_state.top.coarse_action.action_prediction,
            "w_semantic_delta": cache.top.predicted_dynamics.semantic_delta,
            "p2_semantic_effect": output.compiled.effect.semantic,
            "physical_velocity": output.bottom.physical_velocity,
        }
        surfaces = {
            "coarse_action": training_state.top.coarse_action.action_prediction,
            "p2_semantic_effect": output.compiled.effect.semantic,
            "physical_velocity": output.bottom.physical_velocity,
        }
        surface_report: dict[str, object] = {}
        for surface_name, surface in surfaces.items():
            generator = torch.Generator(device=device).manual_seed(9000 + instruction_index)
            cotangent = torch.randn(surface.shape, device=device, dtype=surface.dtype, generator=generator)
            scalar = (surface * cotangent).mean()
            seam_values = tuple(seams.values())
            seam_gradients = _vjp(scalar, seam_values, retain_graph=True)
            owner_report: dict[str, object] = {}
            for group_name, params in groups.items():
                gradients = _vjp(scalar, params, retain_graph=True)
                owner_report[group_name] = _group_stats(gradients, params)
            surface_report[surface_name] = {
                "scalar": float(scalar.detach().float()),
                "seams": {
                    name: _stats(value)
                    for name, value in zip(seams, seam_gradients, strict=True)
                },
                "owners": owner_report,
            }
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("read-only probe populated parameter.grad")
        records[instruction] = {
            "surfaces": surface_report,
            "forward_values": {
                name: {
                    "rms": float(value.detach().float().square().mean().sqrt()),
                    "shape": list(value.shape),
                }
                for name, value in surfaces.items()
            },
        }
        del output, cache, training_state
    return {
        "schema": "clearvla-calvin-language-jacobian-npz-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "layout": args.layout,
        "instructions": list(instructions),
        "records": records,
        "scope": {"optimizer_step": False, "parameter_grad_write": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--layout", default="standard")
    parser.add_argument("--instructions", nargs="+", required=True)
    parser.add_argument("--dinov2-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "instructions": list(args.instructions)}))


if __name__ == "__main__":
    main()
