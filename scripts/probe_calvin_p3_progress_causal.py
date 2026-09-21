#!/usr/bin/env python3
"""Test P3 progress lanes on frozen no-contact CALVIN history snapshots."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import torch

from clearvla.mainline.runtime.flow_schedule import resolve_deployment_flow_schedule
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

try:
    from scripts.probe_calvin_latest_checkpoint_target_y_replay import ladder
except ImportError:
    from probe_calvin_latest_checkpoint_target_y_replay import ladder  # type: ignore[no-redef]


SCHEMA = "clearvla-calvin-p3-progress-causal-v1"
MODES = ("baseline", "state_change_zero", "temporal_zero", "both_zero")


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _trace_rms(values: list[torch.Tensor]) -> float:
    if not values:
        raise ValueError("empty plan trace")
    flat = torch.cat([value.reshape(-1) for value in values])
    return _rms(flat)


def _run_mode(
    policy: ClearVLACheckpointPolicy,
    cache: Any,
    *,
    initial_noise: torch.Tensor,
    target_k: int,
    mode: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown P3 intervention {mode!r}")
    compiler = policy.bundle.model.policy_compiler.plan_compiler
    original_forward = compiler.forward
    captured_pre: dict[str, list[torch.Tensor]] = {
        "temporal": [],
        "state_change": [],
    }
    captured_used: dict[str, list[torch.Tensor]] = {
        "temporal": [],
        "state_change": [],
    }

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        plan, metrics = original_forward(*args, **kwargs)
        captured_pre["temporal"].append(_clone(plan.temporal))
        captured_pre["state_change"].append(_clone(plan.state_change))
        temporal = plan.temporal
        state_change = plan.state_change
        if mode in {"temporal_zero", "both_zero"}:
            temporal = torch.zeros_like(temporal)
        if mode in {"state_change_zero", "both_zero"}:
            state_change = torch.zeros_like(state_change)
        plan = replace(plan, temporal=temporal, state_change=state_change)
        plan.validate()
        captured_used["temporal"].append(_clone(plan.temporal))
        captured_used["state_change"].append(_clone(plan.state_change))
        return plan, metrics

    compiler.forward = patched_forward  # type: ignore[method-assign]
    try:
        lifecycle = ladder._run_lifecycle(
            policy,
            cache,
            initial_noise=initial_noise,
            target_k=target_k,
            mode="baseline",
        )
    finally:
        compiler.forward = original_forward  # type: ignore[method-assign]
    lifecycle["p3"] = {
        "pre_temporal_rms": _trace_rms(captured_pre["temporal"]),
        "pre_state_change_rms": _trace_rms(captured_pre["state_change"]),
        "used_temporal_rms": _trace_rms(captured_used["temporal"]),
        "used_state_change_rms": _trace_rms(captured_used["state_change"]),
        "calls": len(captured_pre["temporal"]),
    }
    lifecycle["mode"] = mode
    return lifecycle


def run(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    rollout_dir: Path,
    output: Path,
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    variant_name: str,
) -> dict[str, Any]:
    summary = ladder._closed_loop_summary(rollout_dir / "summary.json")
    variants = {str(row["name"]): dict(row) for row in summary["variants"]}
    if variant_name not in variants:
        raise ValueError(f"unknown variant {variant_name!r}")
    variant = variants[variant_name]
    instruction = str(summary["instruction"])
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch.device(device),
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    schedule = resolve_deployment_flow_schedule(
        policy.bundle.config.runtime.deployment_flow_schedule
    )
    if schedule.identity.get("proposal", {}).get("parameters", {}).get(
        "candidate_id"
    ) != "Q5":
        raise ValueError("P3 causal probe requires Q5")

    phases: dict[str, Any] = {}
    with torch.no_grad():
        for phase in ("initial", "selected_low"):
            snapshot, snapshot_record = ladder._snapshot_path(
                rollout_dir, variant, phase
            )
            history = ladder._history(snapshot)
            # This probe only needs the instructed red target. The shared
            # causal-ladder encoder normally insists that both red and blue
            # masks are visible because its original experiment compares
            # target identities. Requiring blue here would make an unrelated
            # visibility detail abort the P3 progress intervention.
            original_colours = ladder.COLORS
            ladder.COLORS = ("red",)
            try:
                cache, training_state, metadata, maps = ladder._encode(
                    policy,
                    history,
                    instruction,
                    runtime_dtype=runtime_dtype,
                )
            finally:
                ladder.COLORS = original_colours
            generator = torch.Generator(device=policy.device).manual_seed(9182026)
            initial_noise = policy.bundle.model.outlet_adapter.sample_noise(
                cache.history.batch,
                device=policy.device,
                dtype=cache.history.action_state.dtype,
                generator=generator,
            )
            target_k = int(maps["red"]["target_k"])
            runs = {
                mode: _run_mode(
                    policy,
                    cache,
                    initial_noise=initial_noise,
                    target_k=target_k,
                    mode=mode,
                )
                for mode in MODES
            }
            baseline = runs["baseline"]
            state_history = cache.history.state_history.detach().float()
            state_sequence = torch.cat(
                (state_history, cache.history.state[:, None].detach().float()),
                dim=1,
            )
            state_delta = state_sequence[:, 1:] - state_sequence[:, :-1]
            phases[phase] = {
                "snapshot": snapshot_record,
                "red_slot_map": metadata["slot_maps"]["red"],
                "observed_state_history_delta_rms": _rms(state_delta),
                "object_transport_prior_rms": _rms(
                    training_state.top.facts.transport_prior
                ),
                "state_change_evidence_rms": _rms(
                    training_state.top.intent.state_change_evidence
                ),
                "baseline": {
                    "proposal": ladder._action_summary(baseline["proposal_action"]),
                    "refined": ladder._action_summary(baseline["refined_action"]),
                    "p3": baseline["p3"],
                },
                "interventions": {
                    mode: {
                        "p3": runs[mode]["p3"],
                        "proposal_delta": ladder._action_delta(
                            runs[mode]["proposal_action"],
                            baseline["proposal_action"],
                        ),
                        "refined_delta": ladder._action_delta(
                            runs[mode]["refined_action"],
                            baseline["refined_action"],
                        ),
                    }
                    for mode in MODES[1:]
                },
            }

    result = {
        "schema": SCHEMA,
        "diagnostic_not_benchmark_score": True,
        "checkpoint": {
            "path": str(policy.bundle.checkpoint_path.resolve()),
            "sha256": policy.bundle.checkpoint_sha256,
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "snapshot_source": {
            "rollout_dir": str(rollout_dir.resolve()),
            "variant": variant_name,
            "instruction": instruction,
            "trajectory_diagnostics": variant.get("trajectory_diagnostics"),
        },
        "flow_schedule": schedule.identity,
        "phases": phases,
        "interpretation_contract": {
            "state_change_zero": (
                "removes only the learned P3 state-change lane after it executes; "
                "no contact rule or task oracle is inserted"
            ),
            "temporal_zero": (
                "matched owner control removing only the learned P3 temporal lane"
            ),
            "selected_low": (
                "historical no-contact snapshot; the source rollout moved the target "
                "less than 0.04 mm over 120 steps"
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
    parser.add_argument("--variant-name", default="target_y_minus_040mm")
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        rollout_dir=args.rollout_dir,
        output=args.output,
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        variant_name=str(args.variant_name),
    )
    print(json.dumps({"schema": result["schema"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
