"""Bounded LIBERO closed-loop probe for spatial robustness.

This is intentionally separate from the official suite evaluator.  It keeps
the official fixed-init protocol and bridge contract, but applies a small,
explicit y translation to one free-joint object before ``set_init_state`` and
executes only a bounded number of receding-horizon actions.  The probe is
diagnostic: its success rate must not be reported as an official LIBERO score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bridge import RemotePolicyClient
from .io import atomic_json
from .libero_eval import (
    LiberoBridgePolicy,
    _LiberoEpisodeVideoRecorder,
    _set_init_state_observation,
    _step_observation_only,
    _step_with_success,
    _suite_task_assets,
    libero_action_contract,
    libero_bridge_identity,
    validate_libero_bridge_health,
)


SHORT_PROBE_SCHEMA = "clearvla-libero-short-closed-loop-probe-v3"
DEFAULT_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0
DEFAULT_OBJECT_JOINT = "akita_black_bowl_1_joint0"
DEFAULT_OFFSETS_M = (-0.03, 0.03)
DEFAULT_INIT_INDICES = (0, 1)
DEFAULT_STEPS = 8
DEFAULT_WARMUP_STEPS = 5
DEFAULT_IMAGE_SIDE = 128
DEFAULT_FPS = 10.0


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _resolve_free_joint_y(env: Any, joint_name: str) -> tuple[int, int]:
    """Return the qpos address and y coordinate for one free joint.

    MuJoCo free joints are seven-dimensional ``[x,y,z,quat]`` qpos blocks.
    Resolving the address from the model keeps the probe robust to unrelated
    joint-order changes and prevents silently editing a robot joint.
    """

    model = getattr(getattr(env, "sim", None), "model", None)
    if model is None:
        raise ValueError("LIBERO environment has no MuJoCo model")
    resolver = getattr(model, "joint_name2id", None)
    if not callable(resolver):
        raise ValueError("LIBERO MuJoCo model lacks joint_name2id")
    try:
        joint_id = int(resolver(str(joint_name)))
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"LIBERO joint {joint_name!r} is not present") from error
    if joint_id < 0:
        raise ValueError(f"LIBERO joint {joint_name!r} is not present")
    try:
        joint_type = int(model.jnt_type[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"LIBERO joint metadata is invalid for {joint_name!r}") from error
    # MuJoCo mjJNT_FREE == 0.  Do not permit a scalar hinge or robot joint to
    # be interpreted as an xyz object pose.
    if joint_type != 0:
        raise ValueError(
            f"LIBERO probe joint {joint_name!r} is not free (type={joint_type})"
        )
    return qpos_address, qpos_address + 1


def _resolve_flattened_qpos_offset(env: Any, *, state_width: int) -> int:
    """Locate qpos inside the exact flattened-state format consumed by LIBERO.

    Released LIBERO init states are ``MjSimState.flatten()`` rows, not bare
    qpos.  In mujoco-py the scalar simulation time precedes qpos, so directly
    using ``jnt_qposadr`` would edit the preceding coordinate.  Resolve and
    validate the layout against the live simulator instead of baking in that
    one-element prefix.
    """

    sim = getattr(env, "sim", None)
    getter = getattr(sim, "get_state", None)
    data = getattr(sim, "data", None)
    if not callable(getter) or data is None or not hasattr(data, "qpos"):
        raise ValueError("LIBERO environment lacks a readable MuJoCo state")
    state = getter()
    flattener = getattr(state, "flatten", None)
    if not callable(flattener):
        raise ValueError("LIBERO MuJoCo state cannot be flattened")
    flattened = np.asarray(flattener(), dtype=np.float64).reshape(-1)
    qpos = np.asarray(data.qpos, dtype=np.float64).reshape(-1)
    if flattened.size != int(state_width):
        raise ValueError(
            "LIBERO fixed init-state width differs from live flattened state: "
            f"init={state_width}, live={flattened.size}"
        )
    candidates = [
        start
        for start in range(flattened.size - qpos.size + 1)
        if np.array_equal(flattened[start : start + qpos.size], qpos)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "could not uniquely locate qpos in the LIBERO flattened state: "
            f"candidates={candidates}"
        )
    return int(candidates[0])


def apply_free_joint_y_offset(
    state: np.ndarray,
    *,
    qpos_size: int,
    qpos_flat_offset: int,
    y_qpos_index: int,
    offset_m: float,
) -> tuple[np.ndarray, float, float]:
    """Copy one fixed init state and translate exactly one free-joint y pose."""

    value = np.asarray(state, dtype=np.float64)
    if value.ndim != 1 or value.size < int(qpos_size):
        raise ValueError(
            f"LIBERO init state must be a flat vector with at least {qpos_size} values"
        )
    qpos_index = int(y_qpos_index)
    flat_offset = int(qpos_flat_offset)
    if not 0 <= qpos_index < int(qpos_size):
        raise ValueError("LIBERO y qpos index is outside qpos")
    flat_index = flat_offset + qpos_index
    if flat_offset < 0 or not 0 <= flat_index < value.size:
        raise ValueError("LIBERO y qpos index is outside flattened state")
    delta = _finite_float(offset_m, name="offset_m")
    result = value.copy()
    before = float(result[flat_index])
    result[flat_index] = before + delta
    after = float(result[flat_index])
    if not np.isfinite(result).all():
        raise ValueError("LIBERO offset produced a non-finite init state")
    return result, before, after


def _task_metadata(suite: Any, task_id: int) -> tuple[str, str]:
    task = suite.get_task(int(task_id))
    name = getattr(task, "name", None)
    instruction = getattr(task, "language", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"LIBERO task {task_id} has no name")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = getattr(task, "language_instruction", None)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"LIBERO task {task_id} has no language instruction")
    return name, instruction


def _trace_observation(
    env: Any,
    observation: Mapping[str, Any],
    *,
    object_qpos_index: int,
    step: int,
    phase: str,
    action: np.ndarray | None = None,
    success: bool | None = None,
) -> dict[str, Any]:
    """Capture compact physical state alongside each diagnostic action.

    The official evaluator intentionally keeps only score-facing fields.  The
    short probe is diagnostic, so it records the EEF position, gripper qpos,
    and the free-joint object xyz directly from the simulator.  This makes a
    claim about approach, drift, or gripper oscillation checkable without
    inferring state from compressed video pixels.
    """

    qpos = np.asarray(getattr(getattr(env, "sim", None), "data", None).qpos)
    start = int(object_qpos_index)
    if start < 0 or start + 3 > qpos.size:
        raise ValueError("object free-joint qpos slice is outside the simulator qpos")

    def vector(name: str, shape: tuple[int, ...]) -> list[float] | None:
        value = observation.get(name)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            return None
        return [float(item) for item in array.reshape(-1)]

    row: dict[str, Any] = {
        "step": int(step),
        "phase": str(phase),
        "object_xyz_m": [float(item) for item in qpos[start : start + 3]],
        "object_y_m": float(qpos[start + 1]),
        "eef_pos_m": vector("robot0_eef_pos", (3,)),
        "gripper_qpos": vector("robot0_gripper_qpos", (2,)),
    }
    if action is not None:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("trace action must be one finite [7] row")
        row["action"] = [float(item) for item in value]
    else:
        row["action"] = None
    if success is not None:
        row["success"] = bool(success)
    return row


def _trace_summary(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive bounded, human-readable motion gauges from a trace."""

    action_rows = [row for row in trajectory if isinstance(row.get("action"), list)]
    actions = (
        np.asarray([row["action"] for row in action_rows], dtype=np.float64)
        if action_rows
        else np.empty((0, 7), dtype=np.float64)
    )
    object_y = np.asarray(
        [float(row["object_y_m"]) for row in trajectory], dtype=np.float64
    )
    eef_rows = [row for row in trajectory if isinstance(row.get("eef_pos_m"), list)]
    eef = (
        np.asarray([row["eef_pos_m"] for row in eef_rows], dtype=np.float64)
        if eef_rows
        else np.empty((0, 3), dtype=np.float64)
    )
    result: dict[str, Any] = {
        "action_count": int(actions.shape[0]),
        "gripper_commands": (
            [] if actions.shape[0] == 0 else [float(value) for value in actions[:, 6]]
        ),
        "gripper_sign_changes": 0,
        "arm_command_l2_mean": 0.0,
        "arm_command_l2_max": 0.0,
        "object_y_delta_from_ready_m": 0.0,
        "eef_displacement_from_ready_m": 0.0,
    }
    if actions.shape[0] > 0:
        arm_norm = np.linalg.norm(actions[:, :6], axis=1)
        result["arm_command_l2_mean"] = float(np.mean(arm_norm))
        result["arm_command_l2_max"] = float(np.max(arm_norm))
        signs = np.sign(actions[:, 6])
        signs[np.abs(actions[:, 6]) < 0.05] = 0.0
        nonzero = signs[signs != 0.0]
        if nonzero.size > 1:
            result["gripper_sign_changes"] = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    if object_y.size > 1:
        result["object_y_delta_from_ready_m"] = float(object_y[-1] - object_y[0])
    if eef.shape[0] > 1:
        result["eef_displacement_from_ready_m"] = float(np.linalg.norm(eef[-1] - eef[0]))
    return result


def run_short_probe(
    *,
    suite_name: str,
    task_id: int,
    output: Path,
    video_dir: Path,
    endpoint: str,
    timeout: float,
    checkpoint: str | Path,
    episodes_init_indices: Sequence[int] = DEFAULT_INIT_INDICES,
    offsets_m: Sequence[float] = DEFAULT_OFFSETS_M,
    max_steps: int = DEFAULT_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    image_side: int = DEFAULT_IMAGE_SIDE,
    video_fps: float = DEFAULT_FPS,
    object_joint_name: str = DEFAULT_OBJECT_JOINT,
    allow_smoke_policy: bool = False,
) -> dict[str, Any]:
    """Run the bounded probe and atomically write its JSON summary."""

    output = Path(output).expanduser()
    video_dir = Path(video_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    if video_dir.exists() and not video_dir.is_dir():
        raise FileExistsError(f"probe video path is not a directory: {video_dir}")
    if not str(object_joint_name).strip():
        raise ValueError("object_joint_name must be non-empty")
    normalized_suite = str(suite_name).strip().lower()
    task_id = _positive_int(task_id + 1, name="task_id+1") - 1
    max_steps = _positive_int(max_steps, name="max_steps")
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    image_side = _positive_int(image_side, name="image_side")
    timeout = _finite_float(timeout, name="timeout")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    video_fps = _finite_float(video_fps, name="video_fps")
    if video_fps <= 0.0:
        raise ValueError("video_fps must be positive")

    init_indices = tuple(int(value) for value in episodes_init_indices)
    if not init_indices or len(init_indices) != len(set(init_indices)):
        raise ValueError("episodes_init_indices must be non-empty and unique")
    if any(value < 0 for value in init_indices):
        raise ValueError("episodes_init_indices must be non-negative")
    offsets = tuple(_finite_float(value, name="offset_m") for value in offsets_m)
    if not offsets or len(offsets) != len(set(offsets)):
        raise ValueError("offsets_m must be non-empty and unique")

    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=allow_smoke_policy)
    bridge_identity = libero_bridge_identity(
        health_before, allow_smoke_policy=allow_smoke_policy
    )
    # Import the external simulator only after bridge identity has passed.
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    if normalized_suite not in benchmark_map:
        raise ValueError(f"unknown LIBERO suite {normalized_suite!r}")
    suite = benchmark_map[normalized_suite](task_order_index=0)
    if task_id < 0 or task_id >= int(suite.get_num_tasks()):
        raise ValueError(f"task_id {task_id} is outside suite {normalized_suite}")
    task_name, instruction = _task_metadata(suite, task_id)
    init_states = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
    if init_states.ndim != 2 or any(index >= init_states.shape[0] for index in init_indices):
        raise ValueError("requested init index is absent from LIBERO fixed init states")
    task_assets, _ = _suite_task_assets(
        suite, selected=[task_id], episodes_per_task=max(init_indices) + 1
    )
    bddl_path = str(task_assets[task_id]["bddl_path"])
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=image_side,
        camera_widths=image_side,
    )
    bridge_policy = LiberoBridgePolicy(
        client,
        quat_to_axisangle=quat2axisangle,
        expected_image_side=image_side,
    )
    rows: list[dict[str, Any]] = []
    try:
        env.seed(10000)
        qpos_size = int(env.sim.data.qpos.size)
        object_qpos_index, y_qpos_index = _resolve_free_joint_y(env, object_joint_name)
        qpos_flat_offset = _resolve_flattened_qpos_offset(
            env,
            state_width=int(init_states.shape[1]),
        )
        y_flat_state_index = qpos_flat_offset + y_qpos_index
        ordinal = 0
        for offset_m in offsets:
            for init_index in init_indices:
                env.reset()
                shifted, y_before, y_after = apply_free_joint_y_offset(
                    init_states[init_index],
                    qpos_size=qpos_size,
                    qpos_flat_offset=qpos_flat_offset,
                    y_qpos_index=y_qpos_index,
                    offset_m=offset_m,
                )
                observation = _set_init_state_observation(env, shifted)
                applied_y = float(env.sim.data.qpos[y_qpos_index])
                if not np.isclose(applied_y, y_after, rtol=0.0, atol=1e-10):
                    raise RuntimeError(
                        "LIBERO flattened-state offset did not reach the requested "
                        f"object y coordinate: expected={y_after}, actual={applied_y}"
                    )
                zero = np.zeros(7, dtype=np.float32)
                for _ in range(warmup_steps):
                    observation = _step_observation_only(env, zero.copy())
                trajectory: list[dict[str, Any]] = [
                    _trace_observation(
                        env,
                        observation,
                        object_qpos_index=object_qpos_index,
                        step=0,
                        phase="ready",
                    )
                ]
                video = _LiberoEpisodeVideoRecorder(
                    video_dir,
                    task_id=task_id,
                    episode=ordinal,
                    instruction=instruction,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    fps=video_fps,
                    metadata={
                        "suite": normalized_suite,
                        "task_name": task_name,
                        "task_id": task_id,
                        "init_state": init_index,
                        "object_joint_name": object_joint_name,
                        "object_qpos_index": object_qpos_index,
                        "y_qpos_index": y_qpos_index,
                        "qpos_flat_offset": qpos_flat_offset,
                        "y_flat_state_index": y_flat_state_index,
                        "offset_m": offset_m,
                        "y_before_m": y_before,
                        "y_after_m": y_after,
                        "y_applied_before_warmup_m": applied_y,
                        "bridge_identity": dict(bridge_identity),
                        "checkpoint": str(checkpoint),
                        "action_contract": libero_action_contract(),
                    },
                )
                video.capture_ready(observation)
                bridge_policy.reset()
                success = False
                steps = 0
                try:
                    while steps < max_steps and not success:
                        steps += 1
                        action = bridge_policy.step(observation, instruction)
                        observation, step_success = _step_with_success(env, action)
                        success = success or step_success
                        trajectory.append(
                            _trace_observation(
                                env,
                                observation,
                                object_qpos_index=object_qpos_index,
                                step=steps,
                                phase="policy",
                                action=action,
                                success=step_success,
                            )
                        )
                        video.capture_step(observation, step=steps)
                    audit = bridge_policy.action_audit()
                    if int(audit.get("steps", -1)) != steps:
                        raise AssertionError("bridge action count differs from env steps")
                    video_report = video.finish(
                        success=success,
                        steps=steps,
                        action_audit=audit,
                    )
                except BaseException as error:
                    video.abort(error)
                    raise
                rows.append(
                    {
                        "ordinal": ordinal,
                        "offset_m": offset_m,
                        "init_state": init_index,
                        "y_qpos_index": y_qpos_index,
                        "qpos_flat_offset": qpos_flat_offset,
                        "y_flat_state_index": y_flat_state_index,
                        "y_before_m": y_before,
                        "y_after_m": y_after,
                        "y_applied_before_warmup_m": applied_y,
                        "steps": steps,
                        "success": bool(success),
                        "action_audit": audit,
                        "trajectory": trajectory,
                        "trace_summary": _trace_summary(trajectory),
                        "video": video_report,
                    }
                )
                ordinal += 1
    finally:
        env.close()
    health_after = client.health()
    validate_libero_bridge_health(health_after, allow_smoke_policy=allow_smoke_policy)
    if libero_bridge_identity(health_after, allow_smoke_policy=allow_smoke_policy) != bridge_identity:
        raise RuntimeError("bridge identity changed during short probe")
    result = {
        "schema": SHORT_PROBE_SCHEMA,
        "diagnostic_only": True,
        "complete": True,
        "suite": normalized_suite,
        "task_id": task_id,
        "task_name": task_name,
        "instruction": instruction,
        "checkpoint": str(checkpoint),
        "endpoint": endpoint,
        "bridge_identity": bridge_identity,
        "bridge_health_before": health_before,
        "bridge_health_after": health_after,
        "object_joint_name": object_joint_name,
        "qpos_flat_offset": qpos_flat_offset,
        "y_qpos_index": y_qpos_index,
        "y_flat_state_index": y_flat_state_index,
        "offsets_m": list(offsets),
        "init_indices": list(init_indices),
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "image_side": image_side,
        "video_fps": video_fps,
        "action_contract": libero_action_contract(),
        "rollouts": rows,
        "success_count": int(sum(bool(row["success"]) for row in rows)),
        "rollout_count": len(rows),
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-indices", type=int, nargs="+", default=list(DEFAULT_INIT_INDICES))
    parser.add_argument("--offsets-m", type=float, nargs="+", default=list(DEFAULT_OFFSETS_M))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--image-side", type=int, default=DEFAULT_IMAGE_SIDE)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--object-joint-name", default=DEFAULT_OBJECT_JOINT)
    parser.add_argument("--allow-smoke-policy", action="store_true")
    args = parser.parse_args()
    result = run_short_probe(
        suite_name=args.suite,
        task_id=args.task_id,
        output=args.output,
        video_dir=args.video_dir,
        endpoint=args.endpoint,
        timeout=args.timeout,
        checkpoint=args.checkpoint,
        episodes_init_indices=args.init_indices,
        offsets_m=args.offsets_m,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        image_side=args.image_side,
        video_fps=args.video_fps,
        object_joint_name=args.object_joint_name,
        allow_smoke_policy=args.allow_smoke_policy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "SHORT_PROBE_SCHEMA",
    "apply_free_joint_y_offset",
    "run_short_probe",
]
