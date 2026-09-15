"""Compare cold-start and expert-prefix handoff LIBERO rollouts.

This is a bounded, deployment-only diagnostic.  It keeps the official LIBERO
reset state, five-step physics warmup, native continuous seven-dimensional
action chart, one-row receding horizon and observe-only bridge protocol.  The
expert branch executes a prefix copied from one converted demonstration and
then hands the *causal* observation history to the same checkpoint policy.
The cold branch uses the policy from the reset state.  No training data,
checkpoint, evaluator, codec or model parameters are modified.

The diagnostic bridge used with this script exposes one additional
``POST /reseed`` endpoint.  It resets only the policy's sampling generator and
leaves the causal history untouched.  Reseeding immediately before the
handoff makes the first post-handoff noise draw matched between the two
branches without introducing a second model path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.io import atomic_json
from clearvla.benchmarks.libero_eval import (
    LIBERO_POLICY_HORIZON,
    _LiberoEpisodeVideoRecorder,
    _set_init_state_observation,
    _step_observation_only,
    _step_with_success,
    execute_libero_action,
    libero_action_contract,
    libero_bridge_identity,
    libero_policy_observation,
    validate_libero_bridge_health,
)
from clearvla.benchmarks.libero_timing_probe import (
    physical_snapshot,
    resolve_contact_geometries,
    restore_snapshot,
)


SCHEMA = "clearvla-libero-expert-handoff-v1"
DEFAULT_EXPERT = (
    "/data/senwang/data/libero/converted/"
    "libero_spatial_task0_full_v1_20260910_terminal_suffix_v2/"
    "libero_spatial_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate_demo_0.hdf5"
)
DEFAULT_STEPS = 32
DEFAULT_PREFIX_STEPS = 12
DEFAULT_WARMUP_STEPS = 5
DEFAULT_IMAGE_SIDE = 128
DEFAULT_FPS = 10.0
DEFAULT_SEED = 10000
DEFAULT_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0
DEFAULT_INIT_STATE = 0
DEFAULT_TARGET_BODY = "akita_black_bowl_1_main"
DEFAULT_PLATE_BODY = "plate_1_main"


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_action(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite [7] action")
    return result.copy()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_expert_prefix(path: Path, *, prefix_steps: int) -> dict[str, Any]:
    """Load and fail closed on a converted demo's causal action prefix."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"expert HDF5 is absent: {path}")
    prefix_steps = _strict_int(prefix_steps, name="prefix_steps", minimum=1)
    with h5py.File(path, "r") as stream:
        if "action" not in stream or "action_state" not in stream:
            raise ValueError("expert HDF5 lacks action/action_state datasets")
        actions = np.asarray(stream["action"], dtype=np.float32)
        action_state = np.asarray(stream["action_state"], dtype=np.float32)
        attrs = {str(key): stream.attrs[key] for key in stream.attrs}
    if actions.ndim != 2 or actions.shape[1:] != (7,) or not np.isfinite(actions).all():
        raise ValueError(f"expert actions must be finite [T,7], got {actions.shape}")
    if action_state.shape != actions.shape or not np.isfinite(action_state).all():
        raise ValueError("expert action_state is not episode-aligned")
    source_count = attrs.get("source_action_count", actions.shape[0])
    source_count = _strict_int(source_count, name="source_action_count", minimum=1)
    if source_count > actions.shape[0] or prefix_steps > source_count:
        raise ValueError(
            "expert prefix must lie in original real-action rows: "
            f"prefix={prefix_steps}, source_count={source_count}, length={actions.shape[0]}"
        )
    if not np.array_equal(action_state[0], np.zeros(7, dtype=np.float32)):
        raise ValueError("expert action_state row zero is not the reset-zero boundary")
    if actions.shape[0] > 1 and not np.array_equal(action_state[1:], actions[:-1]):
        raise ValueError("expert action_state does not equal previous executed action")
    prefix = actions[:prefix_steps].copy()
    # The converted chart is already normalized; execute_libero_action will
    # still perform the same explicit clipping as the evaluator.
    clipped = np.stack([execute_libero_action(row) for row in prefix], axis=0)
    return {
        "path": str(path),
        "sha256": _file_digest(path),
        "source_demo": _decode_attr(attrs.get("source_demo")),
        "converter_schema": _decode_attr(attrs.get("converter_schema")),
        "observation_alignment": _decode_attr(attrs.get("observation_alignment")),
        "source_action_count": int(source_count),
        "dataset_length": int(actions.shape[0]),
        "prefix_steps": int(prefix_steps),
        "prefix_raw": prefix.tolist(),
        "prefix_executed": clipped.tolist(),
        "prefix_clip_count": int(np.any(prefix != clipped, axis=1).sum()),
    }


def _decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def reseed_bridge(endpoint: str, *, timeout: float) -> dict[str, Any]:
    """Reset only the bridge policy RNG, preserving its causal history."""

    request = urllib.request.Request(
        endpoint.rstrip("/") + "/reseed", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # urllib wraps HTTP errors with useful text
        raise RuntimeError(f"diagnostic bridge reseed failed: {error}") from error
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"diagnostic bridge reseed returned invalid payload: {payload!r}")
    return payload


def _resolve_task_metadata(suite: Any, task_id: int) -> tuple[str, str]:
    task = suite.get_task(int(task_id))
    name = getattr(task, "name", None)
    instruction = getattr(task, "language", None)
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = getattr(task, "language_instruction", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"task {task_id} has no name")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"task {task_id} has no language instruction")
    return name.strip(), " ".join(instruction.split())


def _policy_plan(
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    previous_action: np.ndarray,
    *,
    instruction: str,
    reset: bool,
    image_side: int,
    quat_to_axisangle: Any,
) -> tuple[Any, np.ndarray, np.ndarray]:
    policy_input = libero_policy_observation(
        observation,
        previous_action,
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    expected = np.zeros(7, dtype=np.float32) if reset else previous_action
    if not np.array_equal(np.asarray(policy_input.action_state), expected):
        raise AssertionError("policy action_state is not reset-zero/previous action")
    chunk = np.asarray(
        client.act(policy_input, instruction, reset=bool(reset)), dtype=np.float32
    )
    if chunk.shape != (LIBERO_POLICY_HORIZON, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"bridge must return finite [{LIBERO_POLICY_HORIZON},7]")
    raw = chunk[0].copy()
    return policy_input, raw, execute_libero_action(raw)


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rollout contains no rows")
    first = rows[0]
    last = rows[-1]
    distances = [
        float(row["post"]["eef_target_distance"]["xyz_m"])
        for row in rows
        if row.get("post", {}).get("eef_target_distance") is not None
    ]
    return {
        "steps": int(len(rows)),
        "success": bool(last.get("success", False)),
        "handoff_step": int(first.get("handoff_step", 0) or 0),
        "initial_eef_xyz_m": first["pre"]["eef_xyz_m"],
        "final_eef_xyz_m": last["post"]["eef_xyz_m"],
        "initial_target_xyz_m": first["pre"]["target_xyz_m"],
        "final_target_xyz_m": last["post"]["target_xyz_m"],
        "initial_eef_target_xyz_m": float(
            first["pre"]["eef_target_distance"]["xyz_m"]
        ),
        "final_eef_target_xyz_m": float(
            last["post"]["eef_target_distance"]["xyz_m"]
        ),
        "minimum_eef_target_xyz_m": min(distances) if distances else None,
        "gripper_positive_count": int(
            sum(float(row["executed_action"][6]) > 0.05 for row in rows)
        ),
        "gripper_negative_count": int(
            sum(float(row["executed_action"][6]) < -0.05 for row in rows)
        ),
        "arm_action_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray([row["executed_action"][:6] for row in rows])
                    )
                )
            )
        ),
        "contact_steps": {
            "gripper_object": [
                int(row["step"])
                for row in rows
                if int(row["post"]["contacts"].get("gripper_object_contact_count", 0))
                > 0
            ],
            "object_plate": [
                int(row["step"])
                for row in rows
                if int(row["post"]["contacts"].get("object_plate_contact_count", 0))
                > 0
            ],
        },
    }


def _run_cold(
    *,
    env: Any,
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    instruction: str,
    geometry: Mapping[str, Any],
    target_body: str,
    plate_body: str,
    steps: int,
    prefix_steps: int,
    warmup_steps: int,
    image_side: int,
    quat_to_axisangle: Any,
    video_dir: Path,
    task_id: int,
    init_state: int,
    fps: float,
    bridge_identity: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    previous = np.zeros(7, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    success = False
    recorder = _LiberoEpisodeVideoRecorder(
        video_dir,
        task_id=task_id,
        episode=init_state,
        instruction=instruction,
        max_steps=steps,
        warmup_steps=warmup_steps,
        fps=fps,
        metadata={
            "diagnostic": "expert_handoff_cold",
            "handoff_prefix_steps": int(prefix_steps),
            "bridge_identity": dict(bridge_identity),
            "action_contract": libero_action_contract(),
            "contact_geometry": dict(geometry),
            "matched_handoff_noise": True,
        },
    )
    recorder.capture_ready(observation)
    try:
        for step in range(1, steps + 1):
            pre = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            if step == prefix_steps + 1:
                reseed = reseed_bridge(client.endpoint, timeout=timeout)
            else:
                reseed = None
            policy_input, raw, executed = _policy_plan(
                client,
                observation,
                previous,
                instruction=instruction,
                reset=step == 1,
                image_side=image_side,
                quat_to_axisangle=quat_to_axisangle,
            )
            observation, step_success = _step_with_success(env, executed.copy())
            success = bool(success or step_success)
            post = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            rows.append(
                {
                    "step": step,
                    "phase": "policy",
                    "handoff_step": prefix_steps + 1,
                    "action_state_input": previous.tolist(),
                    "policy_time_index_before": int(
                        client.health().get("history_time_index") or 0
                    ),
                    "reseed_before_step": reseed,
                    "policy_state_input": np.asarray(
                        policy_input.state, dtype=np.float32
                    ).tolist(),
                    "top_rgb_sha256": _array_digest(np.asarray(policy_input.rgb["top"])),
                    "wrist_rgb_sha256": _array_digest(
                        np.asarray(policy_input.rgb["wrist"])
                    ),
                    "raw_action_chunk": raw.tolist(),
                    "raw_action_row": raw.tolist(),
                    "executed_action": executed.tolist(),
                    "clipped_dims": np.flatnonzero(raw != executed).astype(int).tolist(),
                    "pre": pre,
                    "post": post,
                    "success": bool(success),
                }
            )
            # Keep the diagnostic video aligned with every executed action;
            # the READY frame is followed by one frame per physical step.
            recorder.capture_step(observation, step=step)
            previous = executed.copy()
        audit = _action_audit(rows)
        video = recorder.finish(success=success, steps=len(rows), action_audit=audit)
    except BaseException as error:
        recorder.abort(error)
        raise
    return {
        "mode": "cold_start_policy_only",
        "success": bool(success),
        "steps": len(rows),
        "rows": rows,
        "summary": _summary_rows(rows),
        "action_audit": audit,
        "video": video,
    }


def _run_expert_handoff(
    *,
    env: Any,
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    instruction: str,
    expert: Mapping[str, Any],
    geometry: Mapping[str, Any],
    target_body: str,
    plate_body: str,
    steps: int,
    prefix_steps: int,
    warmup_steps: int,
    image_side: int,
    quat_to_axisangle: Any,
    video_dir: Path,
    task_id: int,
    init_state: int,
    fps: float,
    bridge_identity: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    if steps <= prefix_steps:
        raise ValueError("steps must be greater than prefix_steps")
    expert_actions = np.asarray(expert["prefix_executed"], dtype=np.float32)
    # Initialize the bridge history without using a policy action in the
    # environment.  The diagnostic server's reseed resets only its generator.
    initial_input = libero_policy_observation(
        observation,
        np.zeros(7, dtype=np.float32),
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    ignored = client.act(initial_input, instruction, reset=True)
    if np.asarray(ignored).shape != (LIBERO_POLICY_HORIZON, 7):
        raise ValueError("bridge initialization returned an invalid action chunk")

    previous = np.zeros(7, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    success = False
    recorder = _LiberoEpisodeVideoRecorder(
        video_dir,
        task_id=task_id,
        episode=init_state,
        instruction=instruction,
        max_steps=steps,
        warmup_steps=warmup_steps,
        fps=fps,
        metadata={
            "diagnostic": "expert_handoff_prefix_then_policy",
            "prefix_steps": int(prefix_steps),
            "expert_source": dict(expert),
            "bridge_identity": dict(bridge_identity),
            "action_contract": libero_action_contract(),
            "contact_geometry": dict(geometry),
            "matched_handoff_noise": True,
        },
    )
    recorder.capture_ready(observation)
    try:
        # Execute the expert prefix.  Every released post-action observation
        # except the last is appended through /observe; the last is appended
        # exactly once by the first policy /act at the handoff.
        for index in range(prefix_steps):
            step = index + 1
            action = _finite_action(expert_actions[index], name=f"expert action {step}")
            pre = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            observation, step_success = _step_with_success(env, action.copy())
            success = bool(success or step_success)
            post = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            if step < prefix_steps:
                append_input = libero_policy_observation(
                    observation,
                    action,
                    quat_to_axisangle=quat_to_axisangle,
                    expected_image_side=image_side,
                )
                time_index = client.observe(append_input)
            else:
                time_index = None
            rows.append(
                {
                    "step": step,
                    "phase": "expert_prefix",
                    "handoff_step": prefix_steps + 1,
                    "action_state_input": previous.tolist(),
                    "expert_action": action.tolist(),
                    "raw_action_row": action.tolist(),
                    "executed_action": action.tolist(),
                    "clipped_dims": [],
                    "bridge_observe_time_index_after": time_index,
                    "pre": pre,
                    "post": post,
                    "success": bool(success),
                }
            )
            recorder.capture_step(observation, step=step)
            previous = action.copy()

        reseed_payload = reseed_bridge(client.endpoint, timeout=timeout)
        for step in range(prefix_steps + 1, steps + 1):
            pre = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            policy_input, raw, executed = _policy_plan(
                client,
                observation,
                previous,
                instruction=instruction,
                # The bridge is already initialized by the ignored reset call;
                # this first call appends the final expert observation exactly
                # once and all later calls append normal policy transitions.
                reset=False,
                image_side=image_side,
                quat_to_axisangle=quat_to_axisangle,
            )
            observation, step_success = _step_with_success(env, executed.copy())
            success = bool(success or step_success)
            post = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
            )
            rows.append(
                {
                    "step": step,
                    "phase": "policy_after_handoff",
                    "handoff_step": prefix_steps + 1,
                    "action_state_input": previous.tolist(),
                    "policy_time_index_before": int(
                        client.health().get("history_time_index") or 0
                    ),
                    "reseed_before_handoff": reseed_payload if step == prefix_steps + 1 else None,
                    "policy_state_input": np.asarray(
                        policy_input.state, dtype=np.float32
                    ).tolist(),
                    "top_rgb_sha256": _array_digest(np.asarray(policy_input.rgb["top"])),
                    "wrist_rgb_sha256": _array_digest(
                        np.asarray(policy_input.rgb["wrist"])
                    ),
                    "raw_action_chunk": raw.tolist(),
                    "raw_action_row": raw.tolist(),
                    "executed_action": executed.tolist(),
                    "clipped_dims": np.flatnonzero(raw != executed).astype(int).tolist(),
                    "pre": pre,
                    "post": post,
                    "success": bool(success),
                }
            )
            recorder.capture_step(observation, step=step)
            previous = executed.copy()
        audit = _action_audit(rows)
        audit["expert_prefix_steps"] = int(prefix_steps)
        video = recorder.finish(success=success, steps=len(rows), action_audit=audit)
    except BaseException as error:
        recorder.abort(error)
        raise
    return {
        "mode": "expert_prefix_then_policy",
        "success": bool(success),
        "steps": len(rows),
        "rows": rows,
        "summary": _summary_rows(rows),
        "action_audit": audit,
        "video": video,
    }


def _action_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions = np.asarray([row["executed_action"] for row in rows], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("handoff action audit lost [7] alignment")
    return {
        "steps": int(actions.shape[0]),
        "executed_min": actions.min(axis=0).tolist(),
        "executed_max": actions.max(axis=0).tolist(),
        "arm_rms": float(np.sqrt(np.mean(np.square(actions[:, :6])))),
        "gripper_values": actions[:, 6].tolist(),
        "gripper_positive_count": int(np.count_nonzero(actions[:, 6] > 0.05)),
        "gripper_negative_count": int(np.count_nonzero(actions[:, 6] < -0.05)),
        "expert_rows": int(sum(row.get("phase") == "expert_prefix" for row in rows)),
        "policy_rows": int(sum(row.get("phase") != "expert_prefix" for row in rows)),
    }


def _paired_comparison(
    cold: Mapping[str, Any], expert: Mapping[str, Any], *, prefix_steps: int
) -> dict[str, Any]:
    cold_rows = list(cold["rows"])
    expert_rows = list(expert["rows"])
    if len(cold_rows) != len(expert_rows):
        raise ValueError("cold and expert rollouts have different lengths")
    cold_handoff = cold_rows[prefix_steps - 1]
    expert_handoff = expert_rows[prefix_steps - 1]
    cold_final = cold_rows[-1]
    expert_final = expert_rows[-1]
    cold_handoff_eef = np.asarray(cold_handoff["post"]["eef_xyz_m"], dtype=np.float64)
    expert_handoff_eef = np.asarray(expert_handoff["post"]["eef_xyz_m"], dtype=np.float64)
    cold_final_eef = np.asarray(cold_final["post"]["eef_xyz_m"], dtype=np.float64)
    expert_final_eef = np.asarray(expert_final["post"]["eef_xyz_m"], dtype=np.float64)
    cold_handoff_target = np.asarray(cold_handoff["post"]["target_xyz_m"], dtype=np.float64)
    expert_handoff_target = np.asarray(expert_handoff["post"]["target_xyz_m"], dtype=np.float64)
    return {
        "prefix_steps": int(prefix_steps),
        "cold_success": bool(cold.get("success", False)),
        "expert_success": bool(expert.get("success", False)),
        "handoff_state_delta_eef_xyz_m": (expert_handoff_eef - cold_handoff_eef).tolist(),
        "handoff_state_delta_target_xyz_m": (
            expert_handoff_target - cold_handoff_target
        ).tolist(),
        "handoff_cold_eef_target_xyz_m": float(
            cold_handoff["post"]["eef_target_distance"]["xyz_m"]
        ),
        "handoff_expert_eef_target_xyz_m": float(
            expert_handoff["post"]["eef_target_distance"]["xyz_m"]
        ),
        "handoff_eef_target_improvement_m": float(
            cold_handoff["post"]["eef_target_distance"]["xyz_m"]
            - expert_handoff["post"]["eef_target_distance"]["xyz_m"]
        ),
        "final_cold_eef_target_xyz_m": float(
            cold_final["post"]["eef_target_distance"]["xyz_m"]
        ),
        "final_expert_eef_target_xyz_m": float(
            expert_final["post"]["eef_target_distance"]["xyz_m"]
        ),
        "final_eef_target_improvement_m": float(
            cold_final["post"]["eef_target_distance"]["xyz_m"]
            - expert_final["post"]["eef_target_distance"]["xyz_m"]
        ),
        "final_eef_delta_expert_minus_cold_m": (
            expert_final_eef - cold_final_eef
        ).tolist(),
        "policy_action_after_handoff_delta_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(
                            expert_rows[prefix_steps]["executed_action"], dtype=np.float64
                        )
                        - np.asarray(
                            cold_rows[prefix_steps]["executed_action"], dtype=np.float64
                        )
                    )
                )
            )
        ),
        "expert_prefix_action_rms": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(
                            [row["executed_action"] for row in expert_rows[:prefix_steps]],
                            dtype=np.float64,
                        )
                    )
                )
            )
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.output}")
    if args.video_dir.exists():
        raise FileExistsError(f"refusing existing video directory: {args.video_dir}")
    if args.steps <= args.prefix_steps or args.prefix_steps <= 0:
        raise ValueError("steps must be greater than a positive prefix_steps")
    expert = load_expert_prefix(args.expert_hdf5, prefix_steps=args.prefix_steps)

    client = RemotePolicyClient(endpoint=args.endpoint, timeout=args.timeout)
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=False)
    bridge_identity = libero_bridge_identity(health_before, allow_smoke_policy=False)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    suite = benchmark_map[str(args.suite)](task_order_index=0)
    task_name, instruction = _resolve_task_metadata(suite, args.task_id)
    init_states = np.asarray(suite.get_task_init_states(args.task_id), dtype=np.float64)
    if args.init_state >= init_states.shape[0]:
        raise ValueError("requested init_state is absent")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=False)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "complete": False,
        "scope": "bounded expert-prefix handoff versus cold policy rollout; not official score",
        "suite": str(args.suite),
        "task_id": int(args.task_id),
        "task_name": task_name,
        "instruction": instruction,
        "init_state": int(args.init_state),
        "steps": int(args.steps),
        "prefix_steps": int(args.prefix_steps),
        "warmup_steps": int(args.warmup_steps),
        "image_side": int(args.image_side),
        "seed": int(args.seed),
        "endpoint": str(args.endpoint),
        "checkpoint": str(args.checkpoint),
        "bridge_identity": bridge_identity,
        "expert": expert,
        "protocol": {
            "cold": "policy-only receding horizon from fixed init/warmup",
            "expert": "execute converted expert prefix; append observations via /observe; policy starts at handoff",
            "handoff_step": int(args.prefix_steps + 1),
            "execute_rows": 1,
            "prediction_horizon": LIBERO_POLICY_HORIZON,
            "matched_handoff_noise": "diagnostic /reseed resets policy generator only, history retained",
        },
        "cases": {},
    }
    atomic_json(args.output, payload)

    env = OffScreenRenderEnv(
        bddl_file_name=str(suite.get_task_bddl_file_path(args.task_id)),
        camera_heights=int(args.image_side),
        camera_widths=int(args.image_side),
    )
    try:
        env.seed(int(args.seed))
        env.reset()
        observation = _set_init_state_observation(env, init_states[args.init_state])
        zero = np.zeros(7, dtype=np.float32)
        for _ in range(int(args.warmup_steps)):
            observation = _step_observation_only(env, zero.copy())
        snapshot = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        geometry = resolve_contact_geometries(
            env,
            target_body=args.target_body,
            plate_body=args.plate_body,
        )

        cold_observation, _ = restore_snapshot(env, snapshot=snapshot, seed=args.seed)
        cold = _run_cold(
            env=env,
            client=client,
            observation=cold_observation,
            instruction=instruction,
            geometry=geometry,
            target_body=args.target_body,
            plate_body=args.plate_body,
            steps=args.steps,
            prefix_steps=args.prefix_steps,
            warmup_steps=args.warmup_steps,
            image_side=args.image_side,
            quat_to_axisangle=quat2axisangle,
            video_dir=args.video_dir / "cold_start",
            task_id=args.task_id,
            init_state=args.init_state,
            fps=args.video_fps,
            bridge_identity=bridge_identity,
            timeout=args.timeout,
        )
        expert_observation, _ = restore_snapshot(env, snapshot=snapshot, seed=args.seed)
        # The bridge's reset=True path clears its history and generator before
        # the expert prefix is initialized.  The endpoint /reseed is used only
        # immediately before the policy handoff.
        expert_rollout = _run_expert_handoff(
            env=env,
            client=client,
            observation=expert_observation,
            instruction=instruction,
            expert=expert,
            geometry=geometry,
            target_body=args.target_body,
            plate_body=args.plate_body,
            steps=args.steps,
            prefix_steps=args.prefix_steps,
            warmup_steps=args.warmup_steps,
            image_side=args.image_side,
            quat_to_axisangle=quat2axisangle,
            video_dir=args.video_dir / "expert_prefix",
            task_id=args.task_id,
            init_state=args.init_state,
            fps=args.video_fps,
            bridge_identity=bridge_identity,
            timeout=args.timeout,
        )
        payload["cases"][str(args.init_state)] = {
            "init_state": int(args.init_state),
            "snapshot_length": int(snapshot.size),
            "snapshot_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
            "contact_geometry": geometry,
            "cold_start": cold,
            "expert_prefix": expert_rollout,
            "comparison": _paired_comparison(
                cold, expert_rollout, prefix_steps=args.prefix_steps
            ),
        }
        atomic_json(args.output, payload)
    finally:
        env.close()

    health_after = client.health()
    validate_libero_bridge_health(health_after, allow_smoke_policy=False)
    if libero_bridge_identity(health_after, allow_smoke_policy=False) != bridge_identity:
        raise RuntimeError("bridge identity changed during expert handoff probe")
    payload["bridge_health_before"] = health_before
    payload["bridge_health_after"] = health_after
    payload["complete"] = True
    atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18780")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-hdf5", type=Path, default=Path(DEFAULT_EXPERT))
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init-state", type=int, default=DEFAULT_INIT_STATE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--prefix-steps", type=int, default=DEFAULT_PREFIX_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--image-side", type=int, default=DEFAULT_IMAGE_SIDE)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-body", default=DEFAULT_TARGET_BODY)
    parser.add_argument("--plate-body", default=DEFAULT_PLATE_BODY)
    args = parser.parse_args()
    if not math.isfinite(float(args.timeout)) or args.timeout <= 0.0:
        raise ValueError("timeout must be finite and positive")
    if not math.isfinite(float(args.video_fps)) or args.video_fps <= 0.0:
        raise ValueError("video_fps must be finite and positive")
    result = run_probe(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "video_dir": str(args.video_dir),
                "comparison": result["cases"][str(args.init_state)]["comparison"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
