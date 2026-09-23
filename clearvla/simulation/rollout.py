"""Run and optionally record one closed-loop simulation episode."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import (
    ACTION_DIM,
    EvaluationState,
    ResetResult,
    SimulationEnvironment,
)
from .dataset import EpisodeRecorder, validate_recorded_episode
from .history import CausalHistory
from .policies import EnvironmentRandomPolicy, HoldPolicy, RolloutPolicy
from .video import export_episode_video

DEFAULT_INSTRUCTIONS = {
    "alicia-proxy": "move the tool center point toward the red target",
    "maniskill-stackcube": "stack the red cube on top of the green cube",
}

# This threshold is deliberately a diagnostic convention, not a runtime gate
# and not a training target.  It matches the continuous-gripper event threshold
# used by the StackCube objective, while keeping the trace useful for any
# policy that emits a seven-dimensional native action chart.
POLICY_TRACE_EVENT_THRESHOLD = 0.10


def _policy_trace_metrics(
    chunk: np.ndarray,
    action: np.ndarray,
    *,
    boundary: np.ndarray,
    history_last_executed: np.ndarray | None = None,
    latency_s: float,
    step_index: int,
) -> dict[str, Any]:
    """Return evaluator-only evidence for one receding-horizon decision.

    The full proposal is intentionally stored outside ``PolicyObservation``.
    It is therefore available for post-hoc attribution without becoming a
    privileged input on the next policy call.  ``action`` is the clipped row
    that actually crossed the environment boundary.
    """

    value = np.asarray(chunk, dtype=np.float32)
    executed = np.asarray(action, dtype=np.float32)
    prior = np.asarray(boundary, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or value.shape[0] < 1:
        raise ValueError("policy trace chunk must be a finite [T,7] array")
    if not np.isfinite(value).all():
        raise ValueError("policy trace chunk contains NaN or infinity")
    if executed.shape != (ACTION_DIM,) or not np.isfinite(executed).all():
        raise ValueError("policy trace executed action must be finite [7]")
    if prior.shape != (ACTION_DIM,) or not np.isfinite(prior).all():
        raise ValueError("policy trace boundary must be finite [7]")
    if history_last_executed is not None:
        history_last = np.asarray(history_last_executed, dtype=np.float32)
        if history_last.shape != (ACTION_DIM,) or not np.isfinite(history_last).all():
            raise ValueError("policy trace history action must be finite [7]")
    else:
        history_last = prior
    if not np.isfinite(latency_s) or latency_s < 0.0:
        raise ValueError("policy trace latency must be finite and non-negative")

    gripper = value[:, -1]
    gripper_boundary = np.concatenate(
        (np.asarray([prior[-1]], dtype=np.float32), gripper[:-1]), axis=0
    )
    gripper_delta = gripper - gripper_boundary
    event_indices = np.flatnonzero(
        np.abs(gripper_delta) >= float(POLICY_TRACE_EVENT_THRESHOLD)
    ).astype(np.int64)
    clipped_delta = value[0] - executed
    raw_oob = np.abs(clipped_delta) > 1e-6
    return {
        "telemetry_policy_trace_schema": "receding_horizon_policy_trace_v1",
        "telemetry_policy_step": int(step_index),
        "telemetry_policy_latency_s": float(latency_s),
        "telemetry_policy_action_horizon": int(value.shape[0]),
        "telemetry_policy_raw_action_chunk": value.tolist(),
        "telemetry_policy_raw_action": value[0].tolist(),
        "telemetry_policy_boundary_action": prior.tolist(),
        "telemetry_policy_history_last_executed_action": history_last.tolist(),
        "telemetry_policy_executed_action": executed.tolist(),
        "telemetry_policy_raw_action_oob_count": int(raw_oob.sum()),
        "telemetry_policy_clip_fraction": float(raw_oob.mean()),
        "telemetry_policy_clip_l2": float(np.linalg.norm(clipped_delta)),
        "telemetry_policy_gripper_boundary": float(prior[-1]),
        "telemetry_policy_gripper_first": float(gripper[0]),
        "telemetry_policy_gripper_min": float(gripper.min()),
        "telemetry_policy_gripper_max": float(gripper.max()),
        "telemetry_policy_gripper_delta_first": float(gripper_delta[0]),
        "telemetry_policy_gripper_delta_max_abs": float(np.abs(gripper_delta).max()),
        "telemetry_policy_gripper_event_count": int(event_indices.size),
        "telemetry_policy_gripper_event_indices": event_indices.tolist(),
    }


def _revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _environment(args: argparse.Namespace) -> SimulationEnvironment:
    if args.environment == "alicia-proxy":
        from .alicia_proxy import AliciaProxyEnv

        return AliciaProxyEnv(
            image_size=args.image_size,
            control_hz=args.control_hz,
            max_episode_steps=args.max_episode_steps,
        )
    if args.environment == "maniskill-stackcube":
        from .maniskill_adapter import ManiSkillStackCubeEnv

        return ManiSkillStackCubeEnv(
            image_size=args.image_size,
            max_episode_steps=args.max_episode_steps,
            sim_backend=args.maniskill_sim_backend,
            render_backend=args.maniskill_render_backend,
            # The active StackCube v2 dataset/checkpoint contract uses the
            # causal fixed-down rotation chart.  Keep the legacy chart
            # selectable explicitly for archived v1 artifacts, but do not let
            # the generic rollout entry point silently feed a v2 checkpoint
            # the incompatible principal chart.
            state_chart=getattr(
                args,
                "maniskill_state_chart",
                "fixed_down_causal_rotvec_v2",
            ),
        )
    raise ValueError(f"unknown simulation environment {args.environment!r}")


def _instruction(environment: str, explicit: str | None) -> str:
    value = DEFAULT_INSTRUCTIONS[environment] if explicit is None else explicit
    if not value.strip():
        raise ValueError("simulation instruction must be non-empty")
    return value


def _policy(
    name: str,
    environment: SimulationEnvironment,
    *,
    initial_observation,
    seed: int,
    args: argparse.Namespace,
) -> RolloutPolicy:
    if name == "hold":
        return HoldPolicy(environment.hold_action(initial_observation))
    if name == "random":
        return EnvironmentRandomPolicy(environment.sample_action, seed=seed)
    if name == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--policy checkpoint requires --checkpoint")
        from .clearvla_policy import ClearVLACheckpointPolicy

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA policy device requested but CUDA is unavailable")
        policy = ClearVLACheckpointPolicy(
            args.checkpoint,
            device=device,
            t5_condition=args.t5_condition,
            dinov2_model=args.dinov2_model,
            dinov2_local_files_only=args.dinov2_local_files_only,
            seed=seed,
        )
        if environment.descriptor.benchmark == "ManiSkill3":
            from .admission import validate_stackcube_descriptor

            validate_stackcube_descriptor(environment.descriptor.to_dict(),
                                           profile=policy.bundle.config.data.data_profile)
        return policy
    raise ValueError(f"unknown smoke policy {name!r}")


def run_episode(
    environment: SimulationEnvironment,
    policy: RolloutPolicy,
    *,
    seed: int,
    instruction: str,
    max_steps: int,
    record_dir: Path | None,
    episode_id: str,
    initial_reset: ResetResult | None = None,
    record_policy_trace: bool | None = None,
    record_video: bool | None = None,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if record_video is None:
        # Every persisted closed-loop episode is video evidence by default.
        # Callers can explicitly disable encoding for a tiny smoke run.
        record_video = record_dir is not None
    if record_video and record_dir is None:
        raise ValueError("record_video=True requires record_dir")
    if initial_reset is None:
        reset = environment.reset(seed=int(seed))
    else:
        if not isinstance(initial_reset, ResetResult):
            raise TypeError("initial_reset must be a ResetResult")
        reset = initial_reset
    reset.observation.validate()
    policy.reset()
    history = CausalHistory()
    history.reset(
        reset.observation,
        reset_action=environment.hold_action(reset.observation),
    )
    recorder = (
        None
        if record_dir is None
        else EpisodeRecorder(
            record_dir,
            descriptor=environment.descriptor,
            episode_id=episode_id,
            seed=seed,
            instruction=instruction,
            collector=type(policy).__name__,
            source_revision=_revision(),
            provenance=(
                {
                    "policy_trace_schema": "receding_horizon_policy_trace_v1",
                    "policy_trace_timing": (
                        "latency measures policy.act only; observation is pre-action; "
                        "physics telemetry is post-action"
                    ),
                }
                if record_policy_trace is not False
                else None
            ),
        )
    )
    if record_policy_trace is None:
        # A recorded closed-loop episode should be attribution-complete by
        # default.  Callers can explicitly opt out for a tiny smoke artifact.
        record_policy_trace = recorder is not None
    observation = reset.observation
    reward_sum = 0.0
    success = bool(reset.evaluation.metrics.get("success", False))
    last_metrics = dict(reset.evaluation.metrics)
    executed = 0
    terminal = False
    truncated = False
    for _ in range(int(max_steps)):
        decision_step = executed
        history_snapshot = history.snapshot()
        if not np.allclose(
            history_snapshot.action_state,
            history_snapshot.executed_action_history[-1],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "causal history action_state must equal its last executed action"
            )
        started = time.perf_counter()
        chunk = np.asarray(policy.act(history_snapshot, instruction), dtype=np.float32)
        latency_s = time.perf_counter() - started
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < 1:
            raise ValueError("policy must return at least one action as [T,7]")
        if not np.isfinite(chunk).all():
            raise ValueError("policy action chunk contains NaN or infinity")
        # Receding-horizon control: execute one action and replan immediately.
        # The recorded target must be the command that actually crosses the
        # controller boundary, never an out-of-range policy proposal.
        # Isolate proposal and submitted buffers from mutating native adapters.
        submitted = np.asarray(environment.clip_action(chunk[0].copy()), dtype=np.float32).copy()
        if submitted.shape != (ACTION_DIM,) or not np.isfinite(submitted).all():
            raise ValueError("clipped policy command must be finite [7]")
        result = environment.step(submitted.copy())
        action = result.executed_command(submitted)
        if record_policy_trace:
            trace = _policy_trace_metrics(
                chunk,
                action,
                boundary=history_snapshot.action_state,
                history_last_executed=history_snapshot.executed_action_history[-1],
                latency_s=latency_s,
                step_index=decision_step,
            )
            result = replace(
                result,
                evaluation=EvaluationState(
                    {**dict(result.evaluation.metrics), **trace,
                     "telemetry_submitted_native_action": submitted.tolist(),
                     "telemetry_command_receipt": result.command_receipt is not None,
                     "telemetry_adapter_command_delta": (action - submitted).tolist()}
                ),
            )
        if recorder is not None:
            recorder.append(observation, action, result)
        history.append(action, result.observation)
        observation = result.observation
        reward_sum += float(result.reward)
        executed += 1
        last_metrics = dict(result.evaluation.metrics)
        success = success or bool(last_metrics.get("success", False))
        terminal = bool(result.terminated)
        truncated = bool(result.truncated)
        if terminal or truncated:
            break
    recorded = None
    if recorder is not None:
        artifact = recorder.finish()
        recorded = {
            "path": str(artifact.path.resolve()),
            **validate_recorded_episode(artifact.path),
        }
        if record_video:
            recorded["video"] = export_episode_video(artifact.path)
            recorded["video_path"] = recorded["video"]["path"]
    return {
        "environment": environment.descriptor.to_dict(),
        "policy": type(policy).__name__,
        "instruction": instruction,
        "seed": int(seed),
        "steps": executed,
        "reward_sum": reward_sum,
        "success": success,
        "terminated": terminal,
        "truncated": truncated,
        "final_evaluator_metrics": last_metrics,
        "recorded_episode": recorded,
        "receding_horizon_execute_steps": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("alicia-proxy", "maniskill-stackcube"),
        default="alicia-proxy",
    )
    parser.add_argument(
        "--policy",
        choices=("hold", "random", "checkpoint"),
        default="hold",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=400,
        help=(
            "stable StackCube environment time limit; independent of the smoke "
            "rollout budget"
        ),
    )
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument(
        "--instruction",
        default=None,
    )
    parser.add_argument("--record-dir", type=Path, default=None)
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "export a side-by-side MP4 next to a recorded episode; defaults to "
            "enabled whenever --record-dir is supplied"
        ),
    )
    parser.add_argument("--episode-id", default="episode_000000")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument(
        "--maniskill-sim-backend",
        default="physx_cpu",
        choices=("physx_cpu", "physx_cuda"),
    )
    parser.add_argument(
        "--maniskill-render-backend",
        default="gpu",
        choices=("gpu", "cpu"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--record-policy-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "persist the full raw action chunk, row-zero clip and policy latency "
            "in evaluator-only sim/metrics_json; defaults to enabled when recording"
        ),
    )
    parser.add_argument(
        "--maniskill-state-chart",
        default="fixed_down_causal_rotvec_v2",
        choices=("principal_rotvec_v1", "fixed_down_causal_rotvec_v2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.max_episode_steps <= 0:
        raise ValueError("--steps and --max-episode-steps must be positive")
    instruction = _instruction(args.environment, args.instruction)
    environment = _environment(args)
    try:
        initial_reset = environment.reset(seed=args.seed)
        initial = initial_reset.observation
        policy = _policy(
            args.policy,
            environment,
            initial_observation=initial,
            seed=args.seed,
            args=args,
        )
        result = run_episode(
            environment,
            policy,
            seed=args.seed,
            instruction=instruction,
            max_steps=args.steps,
            record_dir=args.record_dir,
            episode_id=args.episode_id,
            initial_reset=initial_reset,
            record_policy_trace=args.record_policy_trace,
            record_video=args.record_video,
        )
    finally:
        environment.close()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
