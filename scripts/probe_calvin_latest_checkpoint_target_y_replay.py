#!/usr/bin/env python3
"""Replay paired target-position snapshots through a newer frozen checkpoint.

This diagnostic reuses observation/history snapshots only. It is not a
benchmark score and deliberately omits already-closed language/identity
counterfactuals. The purpose is to localize target-position sensitivity from
G/S/coarse/W through P2 and the complete matched-noise Q5 action lifecycle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from clearvla.mainline.runtime.flow_schedule import resolve_deployment_flow_schedule
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

try:
    from scripts import probe_calvin_target_y_causal_ladder as ladder
except ImportError:
    # Remote diagnostics stage the replay and causal-ladder scripts beside
    # each other under /tmp without changing the immutable source checkout.
    # Its snapshot loader needs one newer probe-only validation helper; stage
    # that helper under the package name before importing the standalone files.
    helper_path = Path(__file__).with_name("probe_calvin_internal_layers_npz.py")
    if helper_path.is_file():
        helper_name = "scripts.probe_calvin_internal_layers_npz"
        sys.modules.pop(helper_name, None)
        helper_spec = importlib.util.spec_from_file_location(helper_name, helper_path)
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError(f"cannot load staged helper {helper_path}")
        helper_module = importlib.util.module_from_spec(helper_spec)
        sys.modules[helper_name] = helper_module
        helper_spec.loader.exec_module(helper_module)
    import probe_calvin_target_y_causal_ladder as ladder  # type: ignore[no-redef]


SCHEMA = "clearvla-calvin-latest-checkpoint-target-y-replay-v1"


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    delta = left.detach().float() - right.detach().float()
    reference_rms = 0.5 * (_rms(left) + _rms(right))
    delta_rms = _rms(delta)
    return {
        "shape": list(delta.shape),
        "delta_rms": delta_rms,
        "reference_rms": reference_rms,
        "relative_delta": delta_rms / max(reference_rms, 1e-12),
        "max_abs": float(delta.abs().amax().cpu()),
    }


def _trace_delta(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("trace lengths must match and be non-empty")
    left_value = torch.cat([item.reshape(-1) for item in left])
    right_value = torch.cat([item.reshape(-1) for item in right])
    return _tensor_delta(left_value, right_value)


def _source_action(top: Any) -> torch.Tensor:
    condition = top.action_condition
    if hasattr(condition, "source_action"):
        return condition.source_action
    return condition.source_interval_action


def _stage_tensors(runtime: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    top = runtime["training_state"].top
    dynamics = top.predicted_dynamics
    intent = top.intent
    return {
        "g_camera_coordinates": top.facts.camera_coordinates,
        "s_interval_object_attention": intent.interval_object_attention,
        "s_typed_common_value": intent.typed_common_value,
        "s_typed_interval_residual_value": intent.typed_interval_residual_value,
        "coarse_action": top.coarse_action.action_prediction,
        "physical_action_condition": _source_action(top),
        "w_semantic_delta": dynamics.semantic_delta,
        "w_transport_mean": dynamics.transport_mean,
        "w_transport_covariance": dynamics.transport_covariance,
    }


def run(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    rollout_dir: Path,
    output: Path,
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    tensor_output: Path | None,
) -> dict[str, Any]:
    source_summary = ladder._closed_loop_summary(rollout_dir / "summary.json")
    variants = [dict(item) for item in source_summary["variants"]]
    if len(variants) != 2:
        raise ValueError("paired target-y replay requires exactly two variants")
    instruction = str(source_summary["instruction"])
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
        raise ValueError("replay requires the deployed Q5 schedule")

    runtime: dict[str, Any] = {}
    with torch.no_grad():
        for variant in variants:
            name = str(variant["name"])
            snapshot, snapshot_record = ladder._snapshot_path(
                rollout_dir, variant, "initial"
            )
            history = ladder._history(snapshot)
            cache, training_state, metadata, maps = ladder._encode(
                policy,
                history,
                instruction,
                runtime_dtype=runtime_dtype,
            )
            runtime[name] = {
                "cache": cache,
                "facts": training_state.top.facts,
                "training_state": training_state,
                "metadata": metadata,
                "maps": maps,
                "snapshot": snapshot_record,
            }

        first_cache = runtime[str(variants[0]["name"])]["cache"]
        generator = torch.Generator(device=policy.device).manual_seed(9182026)
        initial_noise = policy.bundle.model.outlet_adapter.sample_noise(
            first_cache.history.batch,
            device=policy.device,
            dtype=first_cache.history.action_state.dtype,
            generator=generator,
        )
        baseline: dict[str, Any] = {}
        for variant in variants:
            name = str(variant["name"])
            target_k = int(runtime[name]["maps"]["red"]["target_k"])
            baseline[name] = ladder._run_lifecycle(
                policy,
                runtime[name]["cache"],
                initial_noise=initial_noise,
                target_k=target_k,
                mode="baseline",
            )

        left_name = str(variants[0]["name"])
        right_name = str(variants[1]["name"])
        left_stage = _stage_tensors(runtime[left_name])
        right_stage = _stage_tensors(runtime[right_name])
        stage_delta = {
            name: _tensor_delta(right_stage[name], left_stage[name])
            for name in left_stage
        }
        left_capture = baseline[left_name]["captures"]
        right_capture = baseline[right_name]["captures"]
        for name in (
            "selected_geometry_used",
            "selected_semantic_used",
            "effect_geometry_used",
            "effect_semantic_used",
            "consequence",
        ):
            stage_delta[f"p2_{name}"] = _trace_delta(
                right_capture[name], left_capture[name]
            )

        stage_delta["q5_refined_action"] = _tensor_delta(
            baseline[right_name]["refined_action"],
            baseline[left_name]["refined_action"],
        )
        stage_delta["q5_proposal_action"] = _tensor_delta(
            baseline[right_name]["proposal_action"],
            baseline[left_name]["proposal_action"],
        )

        transplants: dict[str, Any] = {}
        for receiver, donor in ((left_name, right_name), (right_name, left_name)):
            target_k = int(runtime[receiver]["maps"]["red"]["target_k"])
            transplants[receiver] = {
                "donor": donor,
                "geometry_read": ladder._mode_report(
                    ladder._run_lifecycle(
                        policy,
                        runtime[receiver]["cache"],
                        initial_noise=initial_noise,
                        target_k=target_k,
                        mode="geometry_read_transplant",
                        donor=baseline[donor]["captures"],
                    ),
                    baseline[receiver],
                ),
                "geometry_effect": ladder._mode_report(
                    ladder._run_lifecycle(
                        policy,
                        runtime[receiver]["cache"],
                        initial_noise=initial_noise,
                        target_k=target_k,
                        mode="geometry_effect_transplant",
                        donor=baseline[donor]["captures"],
                    ),
                    baseline[receiver],
                ),
            }

        tensor_artifact = None
        if tensor_output is not None:
            tensor_artifact = ladder._save_tensor_artifact(
                tensor_output,
                variants=variants,
                runtime=runtime,
                baseline=baseline,
            )

    result = {
        "schema": SCHEMA,
        "diagnostic_not_benchmark_score": True,
        "observation_replay_checkpoint_mismatch_is_intentional": True,
        "already_closed_language_identity_counterfactuals_omitted": True,
        "checkpoint": {
            "path": str(policy.bundle.checkpoint_path.resolve()),
            "sha256": policy.bundle.checkpoint_sha256,
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "snapshot_source": {
            "rollout_dir": str(rollout_dir.resolve()),
            "checkpoint": source_summary["checkpoint"],
            "instruction": instruction,
        },
        "flow_schedule": schedule.identity,
        "runtime_compute_dtype": str(runtime_dtype).removeprefix("torch."),
        "variants": {
            name: {
                "snapshot": runtime[name]["snapshot"],
                "red_slot_map": runtime[name]["metadata"]["slot_maps"]["red"],
                "s_red_target_attention": runtime[name]["metadata"][
                    "s_target_attention"
                ]["red"],
                "baseline_proposal": ladder._action_summary(
                    baseline[name]["proposal_action"]
                ),
                "baseline_refined": ladder._action_summary(
                    baseline[name]["refined_action"]
                ),
            }
            for name in (left_name, right_name)
        },
        "paired_stage_delta": stage_delta,
        "cross_variant_transplants": transplants,
        "tensor_artifact": tensor_artifact,
        "interpretation_contract": {
            "stage_delta": (
                "same instruction, robot/history protocol and Q5 noise; only the stored "
                "target-y scene differs"
            ),
            "transplants": (
                "replace only the other scene's realized P2 geometry read/effect and "
                "recompute consequence, proposal W rebuild and refined action"
            ),
            "scope": (
                "checkpoint-backed causal replay for structural localization; no color "
                "label or oracle value is supplied to the normal policy path"
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
    parser.add_argument("--tensor-output", type=Path, default=None)
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        rollout_dir=args.rollout_dir,
        output=args.output,
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        tensor_output=args.tensor_output,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "checkpoint": result["checkpoint"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
