#!/usr/bin/env python3
"""Read the exact causal snapshots from the CALVIN target-y shift probe.

The simulator rollout and checkpoint runtime intentionally use separate Python
environments.  This read-only companion loads the saved causal histories in
the checkpoint environment, evaluates the frozen G/S/W/P graph with identical
noise and flow time for both variants, and reports target-shift sensitivity at
the S object route, P2 semantic/geometry effects, and physical velocity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.mainline.model.types import FlowStepContext
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import HistorySnapshot
from scripts.probe_calvin_internal_layers import (
    _build_online,
    _difference_bundle,
    _tensor_bundle,
    compiler_module,
)
from scripts.probe_calvin_internal_layers_npz import (
    _raw_values,
    _require_calvin_probe_policy,
    _rmse_arrays,
)


SCHEMA = "clearvla-calvin-target-y-shift-closed-loop-v1"
SNAPSHOT_SCHEMA = "clearvla-calvin-causal-history-snapshot-v1"
INTERNAL_SCHEMA = "clearvla-calvin-target-y-shift-internal-v1"
PHASES = ("initial", "high_alignment", "selected_low")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _history(path: Path) -> HistorySnapshot:
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        if version != 1:
            raise ValueError(f"unsupported causal snapshot version {version}")
        result = HistorySnapshot(
            time_index=int(np.asarray(data["time_index"]).reshape(-1)[0]),
            rgb_history={
                "top": np.asarray(data["rgb_top_history"], dtype=np.uint8),
                "wrist": np.asarray(data["rgb_wrist_history"], dtype=np.uint8),
            },
            state=np.asarray(data["state"], dtype=np.float32),
            action_state=np.asarray(data["action_state"], dtype=np.float32),
            state_history=np.asarray(data["state_history"], dtype=np.float32),
            executed_action_history=np.asarray(
                data["executed_action_history"], dtype=np.float32
            ),
        )
    result.validate()
    return result


def _summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"target-y closed-loop summary has invalid schema: {path}")
    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("target-y closed-loop summary must contain two variants")
    return payload


def _snapshot_path(
    rollout_dir: Path,
    variant: Mapping[str, Any],
    phase: str,
) -> tuple[Path, dict[str, Any]]:
    snapshots = variant.get("snapshots")
    if not isinstance(snapshots, Mapping):
        raise ValueError("target-y variant has no causal snapshots")
    record = snapshots.get(phase)
    if not isinstance(record, Mapping) or record.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"target-y variant has invalid {phase} snapshot")
    recorded = Path(str(record["path"]))
    path = recorded if recorded.is_absolute() else rollout_dir / recorded
    if not path.is_file():
        # The summary normally records an absolute remote path.  Permit a
        # relocated result directory by resolving the stable variant layout.
        path = rollout_dir / str(variant["name"]) / "causal_snapshots" / f"{phase}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != record.get("sha256"):
        raise ValueError(f"causal snapshot digest mismatch: {path}")
    return path, dict(record)


def _rmse_report(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
) -> dict[str, float]:
    return {
        name: _rmse_arrays(right[name], left[name])
        for name in (
            "interval_object_attention",
            "typed_relevance_mass",
            "public_interval_carrier",
            "coarse_interval_action",
            "coarse_interval_delta",
            "w_transport_mean",
            "p2_effect_semantic",
            "p2_effect_geometry",
            "p2_consequence",
            "physical_velocity",
            "posterior_spatial_semantic",
            "posterior_spatial_geometry",
        )
    }


def _flow_node(
    closed_loop: Mapping[str, Any], node_index: int
) -> tuple[float, float, float, list[float]]:
    checkpoint = closed_loop.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("closed-loop summary lacks checkpoint identity")
    schedule = checkpoint.get("flow_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("closed-loop checkpoint lacks its flow schedule")
    proposal = schedule.get("proposal")
    refined = schedule.get("refined")
    if not isinstance(proposal, Mapping) or not isinstance(refined, Mapping):
        raise ValueError("closed-loop proposal/refined flow schedules are malformed")
    for role, value in (("proposal", proposal), ("refined", refined)):
        parameters = value.get("parameters")
        candidate = parameters.get("candidate_id") if isinstance(parameters, Mapping) else None
        if candidate != "Q5":
            raise ValueError(
                f"internal probe requires deployed Q5/Q5; {role} is {candidate!r}"
            )
    boundaries = [float(value) for value in proposal.get("boundaries", ())]
    if len(boundaries) < 2 or not np.isfinite(boundaries).all():
        raise ValueError("closed-loop proposal flow boundaries are invalid")
    refined_boundaries = [float(value) for value in refined.get("boundaries", ())]
    if refined_boundaries != boundaries:
        raise ValueError("Q5 proposal/refined boundaries differ unexpectedly")
    steps = len(boundaries) - 1
    if node_index < 0 or node_index >= steps:
        raise ValueError(f"flow node index must be in [0,{steps - 1}]")
    return (
        boundaries[node_index],
        boundaries[node_index + 1] - boundaries[node_index],
        float(node_index) / float(steps),
        boundaries,
    )


def _capture_velocity_at_node(
    model: Any,
    cache: Any,
    *,
    time_value: float,
    step_size: float,
    normalized_index: float,
    noise: torch.Tensor,
) -> tuple[Any, dict[str, Any]]:
    """Capture P2 at one real schedule node under a matched fixed field."""

    captured: list[dict[str, Any]] = []
    original = compiler_module._safe_masked_softmax

    def observer(
        logit: torch.Tensor, support: torch.Tensor, *, dim: int
    ) -> torch.Tensor:
        probability = original(logit, support, dim=dim)
        captured.append(
            {
                "shape": list(probability.shape),
                "dim": int(dim),
                "probability": probability.detach().float().cpu().numpy(),
                "logit": logit.detach().float().cpu().numpy(),
                "support": support.detach().cpu().numpy().astype(bool),
            }
        )
        return probability

    compiler_module._safe_masked_softmax = observer
    try:
        time = torch.full(
            (int(noise.shape[0]),),
            float(time_value),
            device=noise.device,
            dtype=torch.float32,
        )
        context = FlowStepContext.from_step(
            time,
            step_size=float(step_size),
            step_index=float(normalized_index),
            endpoint=False,
        )
        output = model.velocity(
            cache,
            noisy_action_field=noise,
            time=time,
            flow_step_context=context,
            collect_diagnostics=True,
        )
    finally:
        compiler_module._safe_masked_softmax = original
    if len(captured) < 4:
        raise RuntimeError(f"expected at least four P2 softmax calls, got {len(captured)}")
    posterior = {
        "spatial_semantic": captured[0],
        "spatial_geometry": captured[1],
        "temporal": captured[2],
        "temporal_neutral": captured[3],
        "call_count": len(captured),
    }
    return output, posterior


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    rollout_dir: Path,
    output: Path,
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    flow_node_index: int,
) -> dict[str, Any]:
    closed_loop = _summary(rollout_dir / "summary.json")
    variants = [dict(value) for value in closed_loop["variants"]]
    instruction = str(closed_loop["instruction"])
    torch_device = torch.device(device)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch_device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    _require_calvin_probe_policy(policy)
    rollout_checkpoint = closed_loop.get("checkpoint")
    if not isinstance(rollout_checkpoint, Mapping):
        raise ValueError("closed-loop summary lacks checkpoint identity")
    if str(rollout_checkpoint.get("sha256")) != policy.bundle.checkpoint_sha256:
        raise ValueError("internal-probe checkpoint SHA differs from rollout bridge")
    if int(rollout_checkpoint.get("epoch")) != int(policy.bundle.epoch):
        raise ValueError("internal-probe checkpoint epoch differs from rollout bridge")
    if int(rollout_checkpoint.get("global_step")) != int(policy.bundle.global_step):
        raise ValueError("internal-probe checkpoint step differs from rollout bridge")
    time_value, step_size, normalized_index, boundaries = _flow_node(
        closed_loop, int(flow_node_index)
    )
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    autocast_enabled = torch_device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    records: dict[str, Any] = {}
    raw: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    with torch.no_grad():
        for variant in variants:
            name = str(variant["name"])
            records[name] = {}
            raw[name] = {}
            for phase in PHASES:
                snapshot_path, snapshot_record = _snapshot_path(
                    rollout_dir, variant, phase
                )
                history = _history(snapshot_path)
                online = _build_online(policy, history, instruction)
                cache, static_metrics = deployment_cache(
                    policy.bundle.model,
                    online,
                    policy.bundle.config,
                    collect_diagnostics=True,
                )
                noise = torch.randn(
                    1,
                    policy.bundle.config.dimensions.action_horizon,
                    policy.bundle.model.outlet_adapter.physical_dim,
                    device=policy.device,
                    dtype=torch.float32,
                    generator=torch.Generator(device=policy.device).manual_seed(12345),
                )
                with torch.autocast(
                    device_type=torch_device.type,
                    dtype=runtime_dtype,
                    enabled=autocast_enabled,
                ):
                    output_value, posterior = _capture_velocity_at_node(
                        policy.bundle.model,
                        cache,
                        time_value=time_value,
                        step_size=step_size,
                        normalized_index=normalized_index,
                        noise=noise,
                    )
                records[name][phase] = {
                    "snapshot": snapshot_record,
                    "tensors": _tensor_bundle(
                        cache, output_value, posterior, static_metrics
                    ),
                }
                raw[name][phase] = _raw_values(cache, output_value, posterior)

    first_name = str(variants[0]["name"])
    second_name = str(variants[1]["name"])
    pairwise: dict[str, Any] = {}
    for phase in PHASES:
        first_tensors = records[first_name][phase]["tensors"]
        second_tensors = records[second_name][phase]["tensors"]
        phase_comparable = True
        comparison_status = "matched_phase"
        if phase == "selected_low":
            pair = closed_loop["pair"]
            phase_comparable = bool(pair.get("selected_low_phase_comparable", False))
            if str(pair.get("classification", "")).endswith("inconclusive"):
                phase_comparable = False
            if not phase_comparable:
                comparison_status = "closed_loop_phase_inconclusive"
        pairwise[phase] = {
            "phase_comparable": phase_comparable,
            "comparison_status": comparison_status,
            "summary_differences": _difference_bundle(
                second_tensors, first_tensors
            ),
            "exact_tensor_rmse": _rmse_report(
                raw[first_name][phase], raw[second_name][phase]
            ),
        }

    result = {
        "schema": INTERNAL_SCHEMA,
        "diagnostic_not_benchmark_score": True,
        "checkpoint": {
            "path": str(policy.bundle.checkpoint_path.resolve()),
            "sha256": policy.bundle.checkpoint_sha256,
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "rollout": {
            "directory": str(rollout_dir.resolve()),
            "checkpoint": closed_loop["checkpoint"],
            "pair": closed_loop["pair"],
        },
        "instruction": instruction,
        "flow_node": {
            "pass_role": "proposal",
            "cache_semantics": "deployment_cache_before_proposal_ode",
            "index": int(flow_node_index),
            "time": float(time_value),
            "step_size": float(step_size),
            "normalized_index": float(normalized_index),
            "boundaries": boundaries,
            "endpoint": False,
        },
        "fixed_noise_seed": 12345,
        "variant_order": [first_name, second_name],
        "records": records,
        "pairwise": pairwise,
        "interpretation": {
            "interval_object_attention": (
                "S routing sensitivity only; same-index K differences do not "
                "establish moved-target slot attribution"
            ),
            "p2_effect_geometry": "proposal-local compiled P2 geometry sensitivity",
            "physical_velocity": (
                "proposal-local dynamic bottom sensitivity under fixed noise/time"
            ),
            "history": "exact causal history saved by the matching closed-loop rollout",
            "dynamic_scope": (
                "matched proposal-local response at one real Q5 node with "
                "FlowStepContext; not the post-proposal W rebuild, refined pass, "
                "or full deployed ODE trajectory"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--flow-node-index", type=int, default=2)
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        rollout_dir=args.rollout_dir,
        output=args.output,
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        flow_node_index=int(args.flow_node_index),
    )
    print(json.dumps(result["pairwise"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
