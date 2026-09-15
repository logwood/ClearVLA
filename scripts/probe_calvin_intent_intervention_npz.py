#!/usr/bin/env python3
"""Read-only intervention probe for the CALVIN S -> coarse-action boundary.

This diagnostic consumes the cross-Python observation snapshots used by
``probe_calvin_internal_layers_npz``.  It first builds the normal online cache
and captures the three independent ``CoarseActionIntent`` reads.  It then
replays the same S output with only ``public_object_memory`` changed:

* ``baseline``: the exact S-owned object memory;
* ``object_zero``: all object memory zeroed;
* ``object_mean``: every K row replaced by the K mean;
* ``object_permute``: K rows reversed (a permutation-invariance control).

For each intervention the W candidate is rebuilt from the resulting coarse
proposal.  An optional one-time velocity read can quantify downstream impact,
but no policy parameter, checkpoint, dataset, simulator state, or cache on
disk is modified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.model.types import ActionIntentDock
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

from scripts.probe_calvin_internal_layers import (
    INSTRUCTIONS,
    _build_online,
    _capture_velocity,
)
from scripts.probe_calvin_internal_layers_npz import _fingerprint, _history


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _summary(value: Any, *, include_array: bool = False) -> dict[str, Any]:
    arr = _array(value)
    result: dict[str, Any] = {
        "shape": list(arr.shape),
        "rms": float(np.sqrt(np.mean(np.square(arr, dtype=np.float64)))),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if include_array:
        result["array"] = arr.tolist()
    return result


def _rmse(left: Any, right: Any) -> float:
    a = _array(left).astype(np.float64)
    b = _array(right).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.square(a - b))))


def _capture_coarse(model: Any, dock: ActionIntentDock) -> tuple[Any, dict[str, Any]]:
    """Capture the exact three branch deltas emitted by CoarseActionIntent."""

    module = model.intent.coarse_action
    captured: dict[str, Any] = {}
    handles = []
    for name in ("intent_read", "object_read", "history_read"):
        reader = getattr(module, name)

        def hook(_module: Any, _args: Any, output: Any, *, _name: str = name) -> None:
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError(f"unexpected {_name} output from _CrossRead")
            captured[_name] = {
                "value": _summary(output[0], include_array=True),
                "delta": _summary(output[1], include_array=True),
                "weights": _summary(output[2], include_array=True)
                if len(output) > 2
                else None,
            }

        handles.append(reader.register_forward_hook(hook))
    try:
        result = model.intent.propose_action(dock)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"intent_read", "object_read", "history_read"}:
        raise RuntimeError(f"missing coarse read captures: {sorted(captured)}")
    return result, captured


def _condition_from_coarse(model: Any, coarse: Any, action_state: torch.Tensor) -> Any:
    return model.outlet_adapter.world_condition_from_interval_action(
        coarse.action_prediction,
        action_state,
    )


def _intervention_dock(intent: Any, mode: str) -> ActionIntentDock:
    dock = intent.action_dock()
    memory = dock.public_object_memory
    if mode == "baseline":
        changed = memory
    elif mode == "object_zero":
        changed = torch.zeros_like(memory)
    elif mode == "object_mean":
        changed = memory.mean(dim=1, keepdim=True).expand_as(memory)
    elif mode == "object_permute":
        changed = torch.flip(memory, dims=(1,))
    else:
        raise KeyError(f"unknown intervention {mode!r}")
    return replace(dock, public_object_memory=changed)


def _facts_bundle(cache: Any) -> dict[str, Any]:
    facts = cache.top.belief
    return {
        "coordinates": _summary(facts.camera_coordinates, include_array=True),
        "validity": _summary(facts.validity, include_array=True),
        "semantic": _summary(facts.semantic, include_array=True),
        "appearance": _summary(facts.appearance, include_array=True),
        "geometry": _summary(facts.geometry, include_array=True),
        "object_memory": _summary(cache.top.intent.object_tokens, include_array=True),
    }


def _intervention_record(
    *,
    model: Any,
    cache: Any,
    mode: str,
    runtime_dtype: torch.dtype,
    noise: torch.Tensor,
    run_velocity: bool,
) -> dict[str, Any]:
    dock = _intervention_dock(cache.top.intent, mode)
    autocast_enabled = cache.history.state.device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.autocast(
        device_type=cache.history.state.device.type,
        dtype=runtime_dtype,
        enabled=autocast_enabled,
    ):
        coarse, branches = _capture_coarse(model, dock)
        action_condition = _condition_from_coarse(
            model,
            coarse,
            cache.history.action_state,
        )
        world, world_metrics = model.world.materialize(
            belief=cache.top.belief,
            action_condition=action_condition,
            collect_diagnostics=True,
        )
    result: dict[str, Any] = {
        "branches": branches,
        "coarse_interval_action": _summary(
            coarse.action_prediction, include_array=True
        ),
        "coarse_interval_delta": _summary(
            action_condition.interval_delta, include_array=True
        ),
        "w_semantic_delta": _summary(
            world.dynamics.semantic_delta, include_array=True
        ),
        "w_transport_mean": _summary(
            world.dynamics.transport_mean, include_array=True
        ),
        "world_metrics": {
            str(k): float(v.detach().float().cpu().item())
            for k, v in world_metrics.items()
            if isinstance(v, torch.Tensor)
            and v.ndim == 0
            and bool(torch.isfinite(v).all())
        },
    }
    if run_velocity:
        modified_cache = replace(
            cache,
            top=replace(cache.top, candidate_world=world),
        )
        with torch.autocast(
            device_type=cache.history.state.device.type,
            dtype=runtime_dtype,
            enabled=autocast_enabled,
        ):
            output, _posterior = _capture_velocity(
                model,
                modified_cache,
                time_value=0.4,
                noise=noise,
            )
        result["physical_velocity"] = _summary(
            output.bottom.physical_velocity, include_array=True
        )
    return result


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    observation_dir: Path,
    output: Path,
    layouts: tuple[str, ...],
    instructions: tuple[str, ...],
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    run_velocity: bool,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch_device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    records: dict[str, Any] = {}
    modes = ("baseline", "object_zero", "object_mean", "object_permute")
    with torch.no_grad():
        for layout in layouts:
            snapshot = observation_dir / f"{layout}.npz"
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            history = _history(snapshot)
            noise = torch.randn(
                1,
                policy.bundle.config.dimensions.action_horizon,
                policy.bundle.model.outlet_adapter.physical_dim,
                device=policy.device,
                dtype=torch.float32,
                generator=torch.Generator(device=policy.device).manual_seed(12345),
            )
            layout_record: dict[str, Any] = {
                "observation": _fingerprint(snapshot),
                "instructions": {},
            }
            for instruction in instructions:
                online = _build_online(policy, history, instruction)
                cache, static_metrics = deployment_cache(
                    policy.bundle.model,
                    online,
                    policy.bundle.config,
                    collect_diagnostics=True,
                )
                baseline_coarse = cache.top.candidate_world.action_condition.interval_action
                instruction_record: dict[str, Any] = {
                    "facts": _facts_bundle(cache),
                    "static_metrics": {
                        str(k): float(v.detach().float().cpu().item())
                        for k, v in static_metrics.items()
                        if isinstance(v, torch.Tensor)
                        and v.ndim == 0
                        and bool(torch.isfinite(v).all())
                    },
                    "interventions": {},
                }
                for mode in modes:
                    item = _intervention_record(
                        model=policy.bundle.model,
                        cache=cache,
                        mode=mode,
                        runtime_dtype=runtime_dtype,
                        noise=noise,
                        run_velocity=run_velocity,
                    )
                    item["coarse_vs_cache_rmse"] = _rmse(
                        item["coarse_interval_action"]["array"],
                        _array(baseline_coarse),
                    )
                    instruction_record["interventions"][mode] = item
                baseline = instruction_record["interventions"]["baseline"]
                for mode, item in instruction_record["interventions"].items():
                    item["delta_vs_baseline"] = {
                        "coarse_interval_action_rmse": _rmse(
                            item["coarse_interval_action"]["array"],
                            baseline["coarse_interval_action"]["array"],
                        ),
                        "object_read_delta_rmse": _rmse(
                            item["branches"]["object_read"]["delta"]["array"],
                            baseline["branches"]["object_read"]["delta"]["array"],
                        ),
                        "intent_read_delta_rmse": _rmse(
                            item["branches"]["intent_read"]["delta"]["array"],
                            baseline["branches"]["intent_read"]["delta"]["array"],
                        ),
                        "history_read_delta_rmse": _rmse(
                            item["branches"]["history_read"]["delta"]["array"],
                            baseline["branches"]["history_read"]["delta"]["array"],
                        ),
                        "w_semantic_delta_rmse": _rmse(
                            item["w_semantic_delta"]["array"],
                            baseline["w_semantic_delta"]["array"],
                        ),
                        "w_transport_mean_rmse": _rmse(
                            item["w_transport_mean"]["array"],
                            baseline["w_transport_mean"]["array"],
                        ),
                    }
                    if run_velocity and "physical_velocity" in item:
                        item["delta_vs_baseline"]["physical_velocity_rmse"] = _rmse(
                            item["physical_velocity"]["array"],
                            baseline["physical_velocity"]["array"],
                        )
                layout_record["instructions"][instruction] = instruction_record
            records[layout] = layout_record
    result = {
        "schema": "clearvla-calvin-intent-intervention-npz-v1",
        "checkpoint": str(checkpoint),
        "observation_dir": str(observation_dir),
        "layouts": records,
        "instructions": list(instructions),
        "interventions": list(modes),
        "notes": {
            "object_permute": "reverses K rows; no K positional embedding is supplied to _CrossRead",
            "object_zero": "removes the object-memory input while retaining the same learned coarse query",
            "run_velocity": bool(run_velocity),
            "no_mutation": "forward hooks are removed immediately and all outputs are no-grad",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=["standard", "blue_on_slider_left"])
    parser.add_argument("--instructions", nargs="+", default=list(INSTRUCTIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--run-velocity", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        observation_dir=args.observation_dir,
        output=args.output,
        layouts=tuple(str(value) for value in args.layouts),
        instructions=tuple(str(value) for value in args.instructions),
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        run_velocity=bool(args.run_velocity),
    )
    print(json.dumps({"output": str(args.output), "layouts": list(result["layouts"])}, indent=2))


if __name__ == "__main__":
    main()
