"""Run paired short closed-loop LIBERO rollouts after a target-only shift.

This is a diagnostic, not an official benchmark score.  For every selected
official initial state the environment is reset and warmed up once, then its
complete MuJoCo state is snapshotted.  Baseline, positive-shift, and
negative-shift rollouts restore that same state, reset the OSC controller goal,
and execute only row 1 of a freshly replanned 24-row action chunk.  The bridge
policy is reset at the start of every variant, which reuses the checkpoint
sampler's noise sequence across the three paired trajectories.

The script deliberately lives outside the training and formal evaluator paths.
It does not alter the checkpoint, action codec, gripper boundary, or simulator
controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.io import atomic_json
from clearvla.benchmarks.libero_eval import (
    LIBERO_POLICY_HORIZON,
    _LiberoEpisodeVideoRecorder,
    _set_init_state_observation,
    _step_observation_only,
    _step_with_success,
    _suite_task_metadata,
    execute_libero_action,
    libero_action_contract,
    libero_bridge_identity,
    libero_policy_observation,
    validate_libero_bridge_health,
)


def _finite_vector(values: Sequence[float], *, size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _sim(env: Any) -> Any:
    inner = getattr(env, "env", env)
    simulator = getattr(inner, "sim", None)
    if simulator is None:
        raise ValueError("LIBERO environment does not expose a MuJoCo simulator")
    return simulator


def _body_position(env: Any, name: str) -> np.ndarray:
    simulator = _sim(env)
    body_id = int(simulator.model.body_name2id(str(name)))
    value = np.asarray(simulator.data.body_xpos[body_id], dtype=np.float64).copy()
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"body {name!r} has an invalid world position")
    return value


def _body_free_joint_qpos_address(env: Any, name: str) -> int:
    simulator = _sim(env)
    model = simulator.model
    body_id = int(model.body_name2id(str(name)))
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != 0:
        raise ValueError(f"body {name!r} is not rooted at a free joint")
    address = int(model.jnt_qposadr[joint_id])
    if address < 0:
        raise ValueError(f"body {name!r} has an invalid qpos address")
    return address


def _observation_after_forward(env: Any) -> Mapping[str, Any]:
    """Refresh observables after restoring/editing qpos without a physics step."""

    post_process = getattr(env, "_post_process", None)
    update_observables = getattr(env, "_update_observables", None)
    inner = getattr(env, "env", env)
    getter = getattr(inner, "_get_observations", None)
    if not all(callable(value) for value in (post_process, update_observables, getter)):
        raise ValueError("LIBERO environment lacks the direct observation refresh path")
    post_process()
    update_observables(force=True)
    observation = getter()
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO direct observation refresh returned a non-mapping")
    return observation


def _reset_controller_goals(env: Any) -> int:
    """Reset action-integrated OSC goals after restoring a MuJoCo snapshot."""

    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        raise ValueError("LIBERO environment does not expose its robot controllers")
    count = 0
    for robot in robots:
        controller = getattr(robot, "controller", None)
        reset_goal = getattr(controller, "reset_goal", None)
        if not callable(reset_goal):
            raise ValueError("LIBERO OSC controller lacks reset_goal")
        reset_goal()
        count += 1
    return count


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def _distance(eef: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(eef, dtype=np.float64)
    return {
        "xy_m": float(np.linalg.norm(delta[:2])),
        "xyz_m": float(np.linalg.norm(delta)),
    }


def _initial_fingerprint(
    observation: Mapping[str, Any],
    *,
    image_side: int,
    quat_to_axisangle: Any,
) -> dict[str, Any]:
    zero = np.zeros(7, dtype=np.float32)
    policy_input = libero_policy_observation(
        observation,
        zero,
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    return {
        "top_rgb_sha256": _array_digest(np.asarray(policy_input.rgb["top"])),
        "wrist_rgb_sha256": _array_digest(np.asarray(policy_input.rgb["wrist"])),
        "policy_state": np.asarray(policy_input.state, dtype=np.float32).tolist(),
        "action_state": np.asarray(policy_input.action_state, dtype=np.float32).tolist(),
    }


def _restore_variant(
    env: Any,
    *,
    base_sim_state: np.ndarray,
    qpos_address: int,
    shift_xyz_m: np.ndarray,
    environment_seed: int,
) -> tuple[Mapping[str, Any], int]:
    resetter = getattr(env, "reset", None)
    seeder = getattr(env, "seed", None)
    if not callable(resetter) or not callable(seeder):
        raise ValueError("LIBERO environment lacks deterministic reset/seed methods")
    # A flattened MuJoCo state does not own every robosuite controller and
    # observable buffer.  Clear those hidden caches first, then write back the
    # exact paired post-warmup physics snapshot below.
    seeder(int(environment_seed))
    resetter()
    setter = getattr(env, "set_state", None)
    if not callable(setter):
        raise ValueError("LIBERO environment lacks flattened-state restoration")
    setter(np.array(base_sim_state, dtype=np.float64, copy=True))
    simulator = _sim(env)
    simulator.data.qpos[qpos_address : qpos_address + 3] += shift_xyz_m
    simulator.forward()
    controller_count = _reset_controller_goals(env)
    observation = _observation_after_forward(env)
    return observation, controller_count


def _passive_layout_stability(
    env: Any,
    *,
    base_sim_state: np.ndarray,
    qpos_address: int,
    shift_xyz_m: np.ndarray,
    target_body: str,
    plate_body: str,
    steps: int,
    environment_seed: int,
) -> dict[str, Any]:
    """Measure whether a shifted object passively resolves a collision."""

    observation, controller_count = _restore_variant(
        env,
        base_sim_state=base_sim_state,
        qpos_address=qpos_address,
        shift_xyz_m=shift_xyz_m,
        environment_seed=environment_seed,
    )
    initial_target = _body_position(env, target_body)
    initial_plate = _body_position(env, plate_body)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    rows: list[dict[str, Any]] = []
    zero = np.zeros(7, dtype=np.float32)
    max_target_displacement = 0.0
    max_plate_displacement = 0.0
    for step_index in range(int(steps)):
        observation = _step_observation_only(env, zero.copy())
        target = _body_position(env, target_body)
        plate = _body_position(env, plate_body)
        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        target_displacement = float(np.linalg.norm(target - initial_target))
        plate_displacement = float(np.linalg.norm(plate - initial_plate))
        max_target_displacement = max(max_target_displacement, target_displacement)
        max_plate_displacement = max(max_plate_displacement, plate_displacement)
        rows.append(
            {
                "step": int(step_index + 1),
                "target_xyz_m": target.tolist(),
                "plate_xyz_m": plate.tolist(),
                "eef_xyz_m": eef.tolist(),
                "target_displacement_m": target_displacement,
                "plate_displacement_m": plate_displacement,
                "eef_displacement_m": float(np.linalg.norm(eef - initial_eef)),
            }
        )
    return {
        "requested_shift_xyz_m": shift_xyz_m.tolist(),
        "controller_count": int(controller_count),
        "steps": int(steps),
        "initial_target_xyz_m": initial_target.tolist(),
        "initial_plate_xyz_m": initial_plate.tolist(),
        "initial_eef_xyz_m": initial_eef.tolist(),
        "max_target_displacement_m": float(max_target_displacement),
        "max_plate_displacement_m": float(max_plate_displacement),
        "rows": rows,
    }


def _action_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = np.asarray([row["raw_action_row"] for row in rows], dtype=np.float32)
    executed = np.asarray([row["executed_action"] for row in rows], dtype=np.float32)
    if raw.shape != executed.shape or raw.ndim != 2 or raw.shape[1] != 7:
        raise AssertionError("closed-loop action rows lost alignment")
    clipped = raw != executed
    return {
        "steps": int(len(rows)),
        "raw_min": raw.min(axis=0).tolist(),
        "raw_max": raw.max(axis=0).tolist(),
        "executed_min": executed.min(axis=0).tolist(),
        "executed_max": executed.max(axis=0).tolist(),
        "raw_oob_count_per_dim": clipped.sum(axis=0).astype(np.int64).tolist(),
        "raw_oob_row_count": int(np.any(clipped, axis=1).sum()),
        "clipped_row_count": int(np.any(clipped, axis=1).sum()),
        "action_state_matches_executed": all(
            np.array_equal(
                np.asarray(row["action_state_input"], dtype=np.float32),
                (
                    np.zeros(7, dtype=np.float32)
                    if index == 0
                    else executed[index - 1]
                ),
            )
            for index, row in enumerate(rows)
        ),
        "action_state_input_count": int(len(rows)),
    }


def _run_variant(
    *,
    env: Any,
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    instruction: str,
    target_body: str,
    plate_body: str,
    shift_xyz_m: np.ndarray,
    steps: int,
    image_side: int,
    quat_to_axisangle: Any,
    video_dir: Path,
    task_id: int,
    init_state: int,
    warmup_steps: int,
    bridge_identity: Mapping[str, Any],
    repeat_first_input: bool,
) -> dict[str, Any]:
    previous_action = np.zeros(7, dtype=np.float32)
    initial_target = _body_position(env, target_body)
    initial_plate = _body_position(env, plate_body)
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    rows: list[dict[str, Any]] = []
    first_repeat: dict[str, float] | None = None
    success = False
    recorder = _LiberoEpisodeVideoRecorder(
        video_dir,
        task_id=task_id,
        episode=init_state,
        instruction=instruction,
        max_steps=steps,
        warmup_steps=warmup_steps,
        fps=10.0,
        metadata={
            "diagnostic": "target_shift_short_closed_loop",
            "init_state": int(init_state),
            "requested_shift_xyz_m": shift_xyz_m.tolist(),
            "bridge_identity": dict(bridge_identity),
            "action_contract": libero_action_contract(),
        },
    )
    recorder.capture_ready(observation)
    try:
        for step_index in range(int(steps)):
            pre_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
            pre_target = _body_position(env, target_body)
            pre_plate = _body_position(env, plate_body)
            policy_input = libero_policy_observation(
                observation,
                previous_action,
                quat_to_axisangle=quat_to_axisangle,
                expected_image_side=image_side,
            )
            raw_chunk = np.asarray(
                client.act(
                    policy_input,
                    instruction,
                    reset=step_index == 0,
                ),
                dtype=np.float32,
            )
            if raw_chunk.shape != (LIBERO_POLICY_HORIZON, 7):
                raise ValueError(
                    "LIBERO bridge must return exactly one [24,7] action chunk, "
                    f"got {raw_chunk.shape}"
                )
            if not np.isfinite(raw_chunk).all():
                raise ValueError("LIBERO bridge returned a non-finite action chunk")
            if step_index == 0 and repeat_first_input:
                repeated = np.asarray(
                    client.act(policy_input, instruction, reset=True),
                    dtype=np.float32,
                )
                if repeated.shape != raw_chunk.shape:
                    raise ValueError("same-input repeat changed the action shape")
                delta = repeated.astype(np.float64) - raw_chunk.astype(np.float64)
                first_repeat = {
                    "max_abs_delta": float(np.max(np.abs(delta))),
                    "rms_delta": float(np.sqrt(np.mean(np.square(delta)))),
                }
                raw_chunk = repeated
            raw_action = raw_chunk[0].copy()
            executed = execute_libero_action(raw_action)
            action_state_input = previous_action.copy()
            observation, step_success = _step_with_success(env, executed.copy())
            success = success or bool(step_success)
            previous_action = executed.copy()
            post_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
            post_target = _body_position(env, target_body)
            post_plate = _body_position(env, plate_body)
            recorder.capture_step(observation, step=step_index + 1)
            rows.append(
                {
                    "step": int(step_index + 1),
                    "history_time_index_expected": int(step_index),
                    "action_state_input": action_state_input.tolist(),
                    "policy_state_input": np.asarray(
                        policy_input.state, dtype=np.float32
                    ).tolist(),
                    "top_rgb_sha256": _array_digest(
                        np.asarray(policy_input.rgb["top"])
                    ),
                    "wrist_rgb_sha256": _array_digest(
                        np.asarray(policy_input.rgb["wrist"])
                    ),
                    "raw_action_chunk": raw_chunk.tolist(),
                    "clipped_action_chunk": np.clip(raw_chunk, -1.0, 1.0).tolist(),
                    "raw_action_row": raw_action.tolist(),
                    "executed_action": executed.tolist(),
                    "clipped_dims": np.flatnonzero(raw_action != executed)
                    .astype(np.int64)
                    .tolist(),
                    "pre": {
                        "eef_xyz_m": pre_eef.tolist(),
                        "target_xyz_m": pre_target.tolist(),
                        "plate_xyz_m": pre_plate.tolist(),
                        "eef_target_distance": _distance(pre_eef, pre_target),
                    },
                    "post": {
                        "eef_xyz_m": post_eef.tolist(),
                        "target_xyz_m": post_target.tolist(),
                        "plate_xyz_m": post_plate.tolist(),
                        "eef_target_distance": _distance(post_eef, post_target),
                    },
                    "success": bool(success),
                }
            )
        audit = _action_audit(rows)
        video_report = recorder.finish(
            success=success,
            steps=len(rows),
            action_audit=audit,
        )
    except BaseException as error:
        recorder.abort(error)
        raise

    final_target = _body_position(env, target_body)
    final_plate = _body_position(env, plate_body)
    final_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    initial_distance = _distance(initial_eef, initial_target)
    final_distance = _distance(final_eef, final_target)
    min_xy = min(float(row["post"]["eef_target_distance"]["xy_m"]) for row in rows)
    min_xyz = min(float(row["post"]["eef_target_distance"]["xyz_m"]) for row in rows)
    health_after = client.health()
    return {
        "requested_shift_xyz_m": shift_xyz_m.tolist(),
        "initial": {
            "eef_xyz_m": initial_eef.tolist(),
            "target_xyz_m": initial_target.tolist(),
            "plate_xyz_m": initial_plate.tolist(),
            "eef_target_distance": initial_distance,
        },
        "final": {
            "eef_xyz_m": final_eef.tolist(),
            "target_xyz_m": final_target.tolist(),
            "plate_xyz_m": final_plate.tolist(),
            "eef_target_distance": final_distance,
        },
        "summary": {
            "steps": int(len(rows)),
            "success": bool(success),
            "eef_displacement_xyz_m": (final_eef - initial_eef).tolist(),
            "target_displacement_xyz_m": (final_target - initial_target).tolist(),
            "plate_displacement_xyz_m": (final_plate - initial_plate).tolist(),
            "xy_distance_reduction_m": float(initial_distance["xy_m"] - final_distance["xy_m"]),
            "xyz_distance_reduction_m": float(initial_distance["xyz_m"] - final_distance["xyz_m"]),
            "minimum_xy_distance_m": float(min_xy),
            "minimum_xyz_distance_m": float(min_xyz),
        },
        "same_input_first_plan_repeat": first_repeat,
        "action_audit": audit,
        "bridge_history_time_index_after": health_after.get("history_time_index"),
        "video": {
            **video_report,
            "directory": str(video_dir),
        },
        "rows": rows,
    }


def _paired_comparison(
    baseline: Mapping[str, Any],
    shifted: Mapping[str, Any],
) -> dict[str, Any]:
    base_rows = list(baseline["rows"])
    shifted_rows = list(shifted["rows"])
    if len(base_rows) != len(shifted_rows) or not base_rows:
        raise ValueError("paired closed-loop rows have different lengths")
    actual_shift = (
        np.asarray(shifted["initial"]["target_xyz_m"], dtype=np.float64)
        - np.asarray(baseline["initial"]["target_xyz_m"], dtype=np.float64)
    )
    per_step: list[dict[str, Any]] = []
    raw_deltas: list[np.ndarray] = []
    executed_deltas: list[np.ndarray] = []
    for index, (base, moved) in enumerate(zip(base_rows, shifted_rows)):
        raw_delta = np.asarray(moved["raw_action_row"], dtype=np.float64) - np.asarray(
            base["raw_action_row"], dtype=np.float64
        )
        executed_delta = np.asarray(
            moved["executed_action"], dtype=np.float64
        ) - np.asarray(base["executed_action"], dtype=np.float64)
        base_eef = np.asarray(base["post"]["eef_xyz_m"], dtype=np.float64)
        moved_eef = np.asarray(moved["post"]["eef_xyz_m"], dtype=np.float64)
        eef_response = moved_eef - base_eef
        raw_deltas.append(raw_delta)
        executed_deltas.append(executed_delta)
        per_step.append(
            {
                "step": int(index + 1),
                "raw_action_delta": raw_delta.tolist(),
                "executed_action_delta": executed_delta.tolist(),
                "executed_xyz_response_proxy_m": (0.05 * executed_delta[:3]).tolist(),
                "executed_xyz_response_xy_cosine_to_target_shift": _cosine(
                    executed_delta[:2], actual_shift[:2]
                ),
                "eef_response_xyz_m": eef_response.tolist(),
                "eef_response_xy_cosine_to_target_shift": _cosine(
                    eef_response[:2], actual_shift[:2]
                ),
            }
        )
    raw = np.stack(raw_deltas)
    executed = np.stack(executed_deltas)
    base_final_eef = np.asarray(baseline["final"]["eef_xyz_m"], dtype=np.float64)
    moved_final_eef = np.asarray(shifted["final"]["eef_xyz_m"], dtype=np.float64)
    final_eef_response = moved_final_eef - base_final_eef
    base_eef_rows = np.asarray(
        [row["post"]["eef_xyz_m"] for row in base_rows], dtype=np.float64
    )
    moved_eef_rows = np.asarray(
        [row["post"]["eef_xyz_m"] for row in shifted_rows], dtype=np.float64
    )
    trajectory_response = moved_eef_rows - base_eef_rows
    return {
        "actual_initial_target_shift_xyz_m": actual_shift.tolist(),
        "raw_action_delta_rms": float(np.sqrt(np.mean(np.square(raw)))),
        "raw_arm_delta_rms": float(np.sqrt(np.mean(np.square(raw[:, :6])))),
        "raw_arm_xyz_delta_rms": float(np.sqrt(np.mean(np.square(raw[:, :3])))),
        "raw_arm_rotation_delta_rms": float(
            np.sqrt(np.mean(np.square(raw[:, 3:6])))
        ),
        "raw_gripper_delta_rms": float(np.sqrt(np.mean(np.square(raw[:, 6])))),
        "executed_action_delta_rms": float(np.sqrt(np.mean(np.square(executed)))),
        "final_eef_response_xyz_m": final_eef_response.tolist(),
        "final_eef_response_xy_cosine_to_target_shift": _cosine(
            final_eef_response[:2], actual_shift[:2]
        ),
        "trajectory_response_xyz_rms_m": float(
            np.sqrt(np.mean(np.square(trajectory_response)))
        ),
        "shifted_xy_distance_reduction_m": shifted["summary"][
            "xy_distance_reduction_m"
        ],
        "shifted_xyz_distance_reduction_m": shifted["summary"][
            "xyz_distance_reduction_m"
        ],
        "per_step": per_step,
    }


def _opposite_direction_check(
    plus: Mapping[str, Any],
    minus: Mapping[str, Any],
) -> dict[str, Any]:
    plus_steps = list(plus["per_step"])
    minus_steps = list(minus["per_step"])
    if len(plus_steps) != len(minus_steps):
        raise ValueError("positive and negative comparisons have different lengths")
    per_step = []
    for plus_row, minus_row in zip(plus_steps, minus_steps):
        plus_action = np.asarray(
            plus_row["executed_action_delta"], dtype=np.float64
        )
        minus_action = np.asarray(
            minus_row["executed_action_delta"], dtype=np.float64
        )
        plus_eef = np.asarray(plus_row["eef_response_xyz_m"], dtype=np.float64)
        minus_eef = np.asarray(minus_row["eef_response_xyz_m"], dtype=np.float64)
        per_step.append(
            {
                "step": int(plus_row["step"]),
                "plus_vs_negative_minus_executed_xy_cosine": _cosine(
                    plus_action[:2], -minus_action[:2]
                ),
                "plus_vs_negative_minus_eef_xy_cosine": _cosine(
                    plus_eef[:2], -minus_eef[:2]
                ),
            }
        )
    plus_final = np.asarray(plus["final_eef_response_xyz_m"], dtype=np.float64)
    minus_final = np.asarray(minus["final_eef_response_xyz_m"], dtype=np.float64)
    return {
        "final_plus_vs_negative_minus_eef_xy_cosine": _cosine(
            plus_final[:2], -minus_final[:2]
        ),
        "per_step": per_step,
    }


def _fingerprint_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    state_delta = np.asarray(after["policy_state"], dtype=np.float64) - np.asarray(
        before["policy_state"], dtype=np.float64
    )
    action_delta = np.asarray(after["action_state"], dtype=np.float64) - np.asarray(
        before["action_state"], dtype=np.float64
    )
    return {
        "top_rgb_same": before["top_rgb_sha256"] == after["top_rgb_sha256"],
        "wrist_rgb_same": before["wrist_rgb_sha256"] == after["wrist_rgb_sha256"],
        "policy_state_max_abs_delta": float(np.max(np.abs(state_delta))),
        "action_state_max_abs_delta": float(np.max(np.abs(action_delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-states", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--stability-steps", type=int, default=2)
    parser.add_argument("--stability-only", action="store_true")
    parser.add_argument(
        "--max-passive-target-displacement-m",
        type=float,
        default=0.002,
    )
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--target-body", default="akita_black_bowl_1_main")
    parser.add_argument("--plate-body", default="plate_1_main")
    parser.add_argument(
        "--symmetric-shift-xyz-m",
        type=float,
        nargs=3,
        default=(0.030, 0.0, 0.0),
    )
    args = parser.parse_args()

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    if (
        args.steps <= 0
        or args.stability_steps <= 0
        or args.warmup_steps < 0
        or args.image_side <= 0
    ):
        raise ValueError("steps, warmup steps, and image side are invalid")
    passive_limit = float(args.max_passive_target_displacement_m)
    if not np.isfinite(passive_limit) or passive_limit <= 0.0:
        raise ValueError("passive target-displacement limit must be positive")
    if not args.init_states or len(set(args.init_states)) != len(args.init_states):
        raise ValueError("init states must be unique and non-empty")
    shift = _finite_vector(
        args.symmetric_shift_xyz_m,
        size=3,
        name="symmetric shift",
    )
    if float(np.linalg.norm(shift)) <= 0.0:
        raise ValueError("symmetric shift must be non-zero")
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    video_root = (
        args.video_dir.expanduser()
        if args.video_dir is not None
        else output.parent / f"{output.stem}_videos"
    )
    if video_root.exists():
        raise FileExistsError(f"refusing existing probe video directory: {video_root}")

    client = RemotePolicyClient(endpoint=args.endpoint, timeout=float(args.timeout))
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=False)
    bridge_identity = libero_bridge_identity(health_before, allow_smoke_policy=False)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    suite = benchmark_map[str(args.suite)](task_order_index=0)
    task = suite.get_task(int(args.task_id))
    task_name, instruction = _suite_task_metadata(task, task_id=int(args.task_id))
    init_states = np.asarray(
        suite.get_task_init_states(int(args.task_id)), dtype=np.float64
    )
    if min(args.init_states) < 0 or max(args.init_states) >= len(init_states):
        raise ValueError("an init-state index is outside the official inventory")

    payload: dict[str, Any] = {
        "schema": "clearvla-libero-target-shift-short-closed-loop-v3",
        "scope": "diagnostic-only; target-qpos intervention, not benchmark score",
        "complete": False,
        "suite": str(args.suite),
        "task_id": int(args.task_id),
        "task_name": task_name,
        "instruction": instruction,
        "official_init_states": [int(value) for value in args.init_states],
        "target_body": str(args.target_body),
        "plate_body": str(args.plate_body),
        "symmetric_shift_xyz_m": shift.tolist(),
        "closed_loop_steps": int(args.steps),
        "stability_only": bool(args.stability_only),
        "passive_stability_steps": int(args.stability_steps),
        "max_passive_target_displacement_m": passive_limit,
        "warmup_steps": int(args.warmup_steps),
        "image_side": int(args.image_side),
        "environment_seed": int(args.seed),
        "bridge_identity": bridge_identity,
        "matched_noise_contract": (
            "each variant starts with reset=True and has equal planning length; "
            "ClearVLACheckpointPolicy.reset reseeds its owned generator"
        ),
        "execution_contract": (
            "receding horizon; predict 24 rows, clip and execute only row 1, "
            "feed the executed row back as action_state"
        ),
        "environment_restore_contract": (
            "one official reset/warmup creates each init-state snapshot; before "
            "every stability or policy variant a deterministic env.reset clears "
            "robosuite controller/observable caches, then the same complete "
            "post-warmup MuJoCo state is restored, only target xyz qpos is edited, "
            "sim.forward and OSC reset_goal run, and observables are refreshed"
        ),
        "cases": {},
    }
    atomic_json(output, payload)

    env = OffScreenRenderEnv(
        bddl_file_name=str(suite.get_task_bddl_file_path(int(args.task_id))),
        camera_heights=int(args.image_side),
        camera_widths=int(args.image_side),
    )
    try:
        for init_index in args.init_states:
            env.seed(int(args.seed))
            env.reset()
            initial = np.array(init_states[int(init_index)], dtype=np.float64, copy=True)
            observation = _set_init_state_observation(env, initial)
            zero = np.zeros(7, dtype=np.float32)
            for _ in range(int(args.warmup_steps)):
                observation = _step_observation_only(env, zero.copy())
            base_sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            qpos_address = _body_free_joint_qpos_address(env, args.target_body)

            shifts = (
                ("baseline", np.zeros(3, dtype=np.float64)),
                ("plus", shift),
                ("minus", -shift),
            )
            stability = {
                label: _passive_layout_stability(
                    env,
                    base_sim_state=base_sim_state,
                    qpos_address=qpos_address,
                    shift_xyz_m=variant_shift,
                    target_body=str(args.target_body),
                    plate_body=str(args.plate_body),
                    steps=int(args.stability_steps),
                    environment_seed=int(args.seed),
                )
                for label, variant_shift in shifts
            }
            unstable = {
                label: report["max_target_displacement_m"]
                for label, report in stability.items()
                if float(report["max_target_displacement_m"]) > passive_limit
            }
            if unstable:
                payload["cases"][str(int(init_index))] = {
                    "official_init_state": int(init_index),
                    "target_free_joint_qpos_address": int(qpos_address),
                    "passive_layout_stability": stability,
                    "rejected_unstable_variants": unstable,
                }
                atomic_json(output, payload)
                raise RuntimeError(
                    "passive layout stability gate rejected shifted variants for "
                    f"init {init_index}: {unstable}; limit={passive_limit}"
                )
            if args.stability_only:
                payload["cases"][str(int(init_index))] = {
                    "official_init_state": int(init_index),
                    "target_free_joint_qpos_address": int(qpos_address),
                    "passive_layout_stability": stability,
                    "rejected_unstable_variants": {},
                }
                atomic_json(output, payload)
                print(
                    json.dumps(
                        {
                            "progress": "stability_complete",
                            "init_state": int(init_index),
                            "max_target_displacement_m": {
                                label: report["max_target_displacement_m"]
                                for label, report in stability.items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            baseline_observation, controller_count = _restore_variant(
                env,
                base_sim_state=base_sim_state,
                qpos_address=qpos_address,
                shift_xyz_m=np.zeros(3, dtype=np.float64),
                environment_seed=int(args.seed),
            )
            initial_baseline_fingerprint = _initial_fingerprint(
                baseline_observation,
                image_side=int(args.image_side),
                quat_to_axisangle=quat2axisangle,
            )
            variants: dict[str, Any] = {}
            for label, variant_shift in shifts:
                variant_observation, restored_controllers = _restore_variant(
                    env,
                    base_sim_state=base_sim_state,
                    qpos_address=qpos_address,
                    shift_xyz_m=variant_shift,
                    environment_seed=int(args.seed),
                )
                if restored_controllers != controller_count:
                    raise AssertionError("OSC controller inventory changed between variants")
                variants[label] = _run_variant(
                    env=env,
                    client=client,
                    observation=variant_observation,
                    instruction=instruction,
                    target_body=str(args.target_body),
                    plate_body=str(args.plate_body),
                    shift_xyz_m=variant_shift,
                    steps=int(args.steps),
                    image_side=int(args.image_side),
                    quat_to_axisangle=quat2axisangle,
                    video_dir=video_root / f"init_{int(init_index):03d}" / label,
                    task_id=int(args.task_id),
                    init_state=int(init_index),
                    warmup_steps=int(args.warmup_steps),
                    bridge_identity=bridge_identity,
                    repeat_first_input=label == "baseline",
                )
                print(
                    json.dumps(
                        {
                            "progress": "variant_complete",
                            "init_state": int(init_index),
                            "variant": label,
                            "xyz_distance_reduction_m": variants[label]["summary"][
                                "xyz_distance_reduction_m"
                            ],
                            "clipped_rows": variants[label]["action_audit"][
                                "clipped_row_count"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            control_observation, restored_controllers = _restore_variant(
                env,
                base_sim_state=base_sim_state,
                qpos_address=qpos_address,
                shift_xyz_m=np.zeros(3, dtype=np.float64),
                environment_seed=int(args.seed),
            )
            control_fingerprint = _initial_fingerprint(
                control_observation,
                image_side=int(args.image_side),
                quat_to_axisangle=quat2axisangle,
            )
            restore_control = _fingerprint_delta(
                initial_baseline_fingerprint, control_fingerprint
            )
            restored_state = np.asarray(env.get_sim_state(), dtype=np.float64)
            restore_control["sim_state_max_abs_delta"] = float(
                np.max(np.abs(restored_state - base_sim_state))
            )
            restore_control["controller_count"] = int(restored_controllers)
            if not (
                restore_control["top_rgb_same"]
                and restore_control["wrist_rgb_same"]
                and restore_control["policy_state_max_abs_delta"] == 0.0
                and restore_control["action_state_max_abs_delta"] == 0.0
                and restore_control["sim_state_max_abs_delta"] == 0.0
            ):
                raise RuntimeError(
                    f"environment restore control failed for init {init_index}: "
                    f"{restore_control}"
                )

            plus_comparison = _paired_comparison(
                variants["baseline"], variants["plus"]
            )
            minus_comparison = _paired_comparison(
                variants["baseline"], variants["minus"]
            )
            case = {
                "official_init_state": int(init_index),
                "target_free_joint_qpos_address": int(qpos_address),
                "passive_layout_stability": stability,
                "restore_control": restore_control,
                "variants": variants,
                "comparisons": {
                    "plus_vs_baseline": plus_comparison,
                    "minus_vs_baseline": minus_comparison,
                    "opposite_direction": _opposite_direction_check(
                        plus_comparison, minus_comparison
                    ),
                },
            }
            payload["cases"][str(int(init_index))] = case
            atomic_json(output, payload)
            print(
                json.dumps(
                    {
                        "init_state": int(init_index),
                        "baseline_xyz_reduction_m": variants["baseline"]["summary"][
                            "xyz_distance_reduction_m"
                        ],
                        "plus_xyz_reduction_m": variants["plus"]["summary"][
                            "xyz_distance_reduction_m"
                        ],
                        "minus_xyz_reduction_m": variants["minus"]["summary"][
                            "xyz_distance_reduction_m"
                        ],
                        "plus_final_response_cosine": plus_comparison[
                            "final_eef_response_xy_cosine_to_target_shift"
                        ],
                        "minus_final_response_cosine": minus_comparison[
                            "final_eef_response_xy_cosine_to_target_shift"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    health_after = client.health()
    if libero_bridge_identity(
        health_after, allow_smoke_policy=False
    ) != bridge_identity:
        raise RuntimeError("bridge identity changed during closed-loop probe")
    payload["bridge_health_after"] = health_after
    payload["complete"] = True
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "video_dir": None if args.stability_only else str(video_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
