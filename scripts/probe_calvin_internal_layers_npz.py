#!/usr/bin/env python3
"""Run the CALVIN internal language probe on cross-Python observation NPZs.

The CALVIN pybullet renderer and the ClearVLA checkpoint intentionally live in
different Python environments on the server.  ``export_calvin_probe_observations``
produces the causal RGB/state snapshots in Python 3.9; this runner consumes
those snapshots in Python 3.12 and invokes only the frozen deployment graph.
No policy parameters, checkpoint, simulator state, or training artifact is
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import HistorySnapshot

from scripts.probe_calvin_internal_layers import (
    INSTRUCTIONS,
    _build_online,
    _capture_velocity,
    _difference_bundle,
    _tensor_bundle,
)


def _history(path: Path) -> HistorySnapshot:
    with np.load(path, allow_pickle=False) as data:
        top = np.asarray(data["top"], dtype=np.uint8)
        wrist = np.asarray(data["wrist"], dtype=np.uint8)
        state = np.asarray(data["state"], dtype=np.float32)
        action_state = np.asarray(data["action_state"], dtype=np.float32)
    if top.ndim != 3 or wrist.ndim != 3 or top.shape[-1] != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"invalid RGB snapshot shapes in {path}")
    if state.shape != (7,) or action_state.shape != (7,):
        raise ValueError(f"invalid state/action shapes in {path}")
    result = HistorySnapshot(
        time_index=0,
        rgb_history={
            "top": np.repeat(top[None, ...], 3, axis=0),
            "wrist": np.repeat(wrist[None, ...], 3, axis=0),
        },
        state=state,
        action_state=action_state,
        state_history=np.repeat(state[None, ...], 3, axis=0),
        executed_action_history=np.repeat(action_state[None, ...], 8, axis=0),
    )
    result.validate()
    return result


def _fingerprint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        result: dict[str, Any] = {}
        for name in ("top", "wrist", "state", "action_state"):
            value = np.ascontiguousarray(data[name])
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        return result


def _raw_values(cache: Any, output: Any, posterior: dict[str, Any]) -> dict[str, np.ndarray]:
    """Keep compact CPU copies for exact pairwise tensor RMSEs."""

    intent = cache.top.intent
    dynamics = cache.top.predicted_dynamics
    condition = cache.top.action_condition
    compiled = output.compiled
    values: dict[str, Any] = {
        "protected_goal": intent.protected_goal_set,
        # Keep the attention maps in the compact pairwise tensor audit.  They
        # are detached diagnostics and are not fed back into deployment.
        "goal_attention": intent.goal_attention,
        "interval_goal_attention": intent.interval_goal_attention,
        "interval_object_attention": intent.interval_object_attention,
        "public_interval_carrier": intent.public_interval_carrier,
        "policy_interval_context": intent.policy_interval_context,
        "typed_relevance_mass": intent.typed_relevance_mass,
        "typed_relevance_value": intent.typed_relevance_value,
        "typed_policy_components": intent.typed_policy_components,
        "coarse_interval_action": condition.interval_action,
        "coarse_interval_delta": condition.interval_delta,
        "w_semantic_delta": dynamics.semantic_delta,
        "w_transport_mean": dynamics.transport_mean,
        "p2_effect_semantic": compiled.effect.semantic,
        "p2_effect_geometry": compiled.effect.geometry,
        "p2_consequence": compiled.consequence.protected_consequence,
        "p3_temporal": compiled.plan.temporal,
        "p3_state_change": compiled.plan.state_change,
        "physical_velocity": output.bottom.physical_velocity,
    }
    values.update(
        {
            f"posterior_{name}": item["probability"]
            for name, item in posterior.items()
            if isinstance(item, dict) and "probability" in item
        }
    )
    return {
        name: (value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32))
        for name, value in values.items()
    }


def _rmse_arrays(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"pairwise tensor shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.square(a - b))))


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
    time_value: float,
) -> dict[str, Any]:
    if not instructions:
        raise ValueError("at least one instruction is required")
    torch_device = torch.device(device)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch_device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    manifest_path = observation_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    records: dict[str, Any] = {}
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    autocast_enabled = torch_device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.no_grad():
        for layout in layouts:
            snapshot = observation_dir / f"{layout}.npz"
            if not snapshot.is_file():
                raise FileNotFoundError(snapshot)
            history = _history(snapshot)
            layout_record: dict[str, Any] = {
                "observation": _fingerprint(snapshot),
                "manifest": manifest.get("layouts", {}).get(layout, {}),
                "instructions": {},
                "pairwise_vs_first": {},
                "pairwise_tensor_rmse_vs_first": {},
            }
            raw_by_instruction: dict[str, dict[str, np.ndarray]] = {}
            noise = torch.randn(
                1,
                policy.bundle.config.dimensions.action_horizon,
                policy.bundle.model.outlet_adapter.physical_dim,
                device=policy.device,
                dtype=torch.float32,
                generator=torch.Generator(device=policy.device).manual_seed(12345),
            )
            for instruction in instructions:
                online = _build_online(policy, history, instruction)
                cache, static_metrics = deployment_cache(
                    policy.bundle.model,
                    online,
                    policy.bundle.config,
                    collect_diagnostics=True,
                )
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=runtime_dtype,
                    enabled=autocast_enabled,
                ):
                    output_value, posterior = _capture_velocity(
                        policy.bundle.model,
                        cache,
                        time_value=time_value,
                        noise=noise,
                    )
                layout_record["instructions"][instruction] = _tensor_bundle(
                    cache,
                    output_value,
                    posterior,
                    static_metrics,
                )
                raw_by_instruction[instruction] = _raw_values(cache, output_value, posterior)
            baseline = layout_record["instructions"][instructions[0]]
            for instruction in instructions[1:]:
                layout_record["pairwise_vs_first"][instruction] = _difference_bundle(
                    layout_record["instructions"][instruction], baseline
                )
                layout_record["pairwise_tensor_rmse_vs_first"][instruction] = {
                    name: _rmse_arrays(values, raw_by_instruction[instructions[0]][name])
                    for name, values in raw_by_instruction[instruction].items()
                }
            records[layout] = layout_record
    result = {
        "schema": "clearvla-calvin-internal-language-npz-v1",
        "checkpoint": str(checkpoint),
        "time_value": float(time_value),
        "observation_dir": str(observation_dir),
        "layouts": records,
        "instructions": list(instructions),
        "notes": {
            "noise": "fixed standard-normal physical field, seed 12345",
            "history": "each exported reset observation is repeated over the causal three-frame history; reset action is repeated for unavailable executed rows",
            "renderer_policy_abi": "CALVIN pybullet snapshot from Python 3.9; policy inference in Python 3.12",
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
    parser.add_argument("--time", type=float, default=0.4)
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
        time_value=float(args.time),
    )
    print(json.dumps({"output": str(args.output), "layouts": list(result["layouts"])}, indent=2))


if __name__ == "__main__":
    main()
