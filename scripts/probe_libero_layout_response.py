"""Probe whether a LIBERO policy responds equivariantly to a target-only shift.

This is a bounded diagnostic, not a benchmark score.  The environment is reset
exactly once, advanced through the official zero-action warmup, and snapshotted.
Every variant restores that same complete MuJoCo state and changes only the
target free joint's xyz qpos before rendering.  Every policy request is sent
with ``reset=True`` so the checkpoint policy also reuses its sampling noise.
Repeated reads of the unmodified snapshot measure the remaining render/
observation floor without confounding the target intervention with env.reset().
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.io import atomic_json
from clearvla.benchmarks.libero_eval import (
    _set_init_state_observation,
    _step_observation_only,
    _suite_task_metadata,
    execute_libero_action,
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
    if joint_id < 0:
        raise ValueError(f"body {name!r} has no joint")
    # MuJoCo's joint-type enum is free=0, ball=1, slide=2, hinge=3.
    if int(model.jnt_type[joint_id]) != 0:
        raise ValueError(f"body {name!r} is not rooted at a free joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    if qpos_address < 0:
        raise ValueError(f"body {name!r} has an invalid qpos address")
    return qpos_address


def _observation_after_forward(env: Any) -> Mapping[str, Any]:
    """Refresh LIBERO observables after a qpos-only intervention.

    This mirrors ``ControlEnv.regenerate_obs_from_state`` after its state write,
    but deliberately performs no reset and no physics step.
    """

    post_process = getattr(env, "_post_process", None)
    update_observables = getattr(env, "_update_observables", None)
    inner = getattr(env, "env", env)
    getter = getattr(inner, "_get_observations", None)
    if (
        not callable(post_process)
        or not callable(update_observables)
        or not callable(getter)
    ):
        raise ValueError("LIBERO environment lacks the direct observation refresh path")
    post_process()
    update_observables(force=True)
    observation = getter()
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO direct observation refresh returned a non-mapping")
    return observation


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_delta_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    if left_value.shape != right_value.shape:
        raise ValueError("array-delta inputs have different shapes")
    delta = right_value - left_value
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "changed_fraction": float(np.mean(delta != 0.0)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None
    return float(np.dot(left, right) / denominator)


def _variant(
    *,
    env: Any,
    base_sim_state: np.ndarray,
    qpos_address: int,
    requested_shift_xyz_m: np.ndarray,
    target_body: str,
    plate_body: str,
    instruction: str,
    client: RemotePolicyClient,
    quat_to_axisangle: Any,
    image_side: int,
    repeat_same_policy_input: bool = False,
) -> dict[str, Any]:
    simulator = _sim(env)
    setter = getattr(env, "set_state", None)
    if not callable(setter):
        raise ValueError("LIBERO environment lacks flattened-state restoration")
    setter(np.array(base_sim_state, dtype=np.float64, copy=True))
    simulator.data.qpos[qpos_address : qpos_address + 3] += requested_shift_xyz_m
    simulator.forward()
    observation = _observation_after_forward(env)

    zero = np.zeros(7, dtype=np.float32)
    target_position = _body_position(env, target_body)
    plate_position = _body_position(env, plate_body)
    eef_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    policy_input = libero_policy_observation(
        observation,
        zero,
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    raw_chunk = np.asarray(
        client.act(policy_input, instruction, reset=True), dtype=np.float32
    )
    if raw_chunk.shape != (24, 7) or not np.isfinite(raw_chunk).all():
        raise ValueError(f"bridge returned an invalid action chunk {raw_chunk.shape}")
    clipped_chunk = np.stack(
        [execute_libero_action(row) for row in raw_chunk], axis=0
    ).astype(np.float32)
    same_input_repeat = None
    if repeat_same_policy_input:
        same_input_repeat = np.asarray(
            client.act(policy_input, instruction, reset=True), dtype=np.float32
        )
        if same_input_repeat.shape != raw_chunk.shape:
            raise ValueError("same-input bridge repeat changed action shape")
    top_rgb = np.asarray(policy_input.rgb["top"]).copy()
    wrist_rgb = np.asarray(policy_input.rgb["wrist"]).copy()
    result = {
        "requested_shift_xyz_m": requested_shift_xyz_m.tolist(),
        "target_position_xyz_m": target_position.tolist(),
        "plate_position_xyz_m": plate_position.tolist(),
        "eef_position_xyz_m": eef_position.tolist(),
        "target_minus_eef_xyz_m": (target_position - eef_position).tolist(),
        "top_rgb_sha256": _array_digest(top_rgb),
        "wrist_rgb_sha256": _array_digest(wrist_rgb),
        "policy_state": np.asarray(policy_input.state, dtype=np.float32).tolist(),
        "raw_action_chunk": raw_chunk.tolist(),
        "clipped_action_chunk": clipped_chunk.tolist(),
        "_top_rgb": top_rgb,
        "_wrist_rgb": wrist_rgb,
    }
    if same_input_repeat is not None:
        result["same_input_repeat_raw_action_chunk"] = same_input_repeat.tolist()
    return result


def _approach_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    chunk = np.asarray(row["clipped_action_chunk"], dtype=np.float64)
    target_offset = np.asarray(row["target_minus_eef_xyz_m"], dtype=np.float64)
    result: dict[str, Any] = {}
    for horizon in (1, 4, 8, 24):
        # Official OSC_POSE maps a unit xyz command to a 0.05 m requested
        # displacement.  The sum is an intent proxy, not a simulator replay.
        command_xyz_m = 0.05 * chunk[:horizon, :3].sum(axis=0)
        result[str(horizon)] = {
            "cumulative_xyz_command_m": command_xyz_m.tolist(),
            "xy_cosine_to_current_target": _cosine(
                command_xyz_m[:2], target_offset[:2]
            ),
        }
    return result


def _response_summary(
    baseline: Mapping[str, Any],
    shifted: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_raw = np.asarray(baseline["raw_action_chunk"], dtype=np.float64)
    baseline_clipped = np.asarray(
        baseline["clipped_action_chunk"], dtype=np.float64
    )
    shifted_raw = np.asarray(shifted["raw_action_chunk"], dtype=np.float64)
    baseline_clipped = np.asarray(
        baseline["clipped_action_chunk"], dtype=np.float64
    )
    shifted_clipped = np.asarray(shifted["clipped_action_chunk"], dtype=np.float64)
    actual_shift = np.asarray(
        shifted["target_position_xyz_m"], dtype=np.float64
    ) - np.asarray(baseline["target_position_xyz_m"], dtype=np.float64)
    raw_delta = shifted_raw - baseline_raw
    clipped_delta = shifted_clipped - baseline_clipped
    bands: dict[str, Any] = {}
    for horizon in (1, 4, 8, 24):
        response_xyz_m = 0.05 * clipped_delta[:horizon, :3].sum(axis=0)
        shift_xy = actual_shift[:2]
        response_xy = response_xyz_m[:2]
        denominator = float(np.dot(shift_xy, shift_xy))
        gain = (
            None
            if denominator <= 1e-12
            else float(np.dot(response_xy, shift_xy) / denominator)
        )
        bands[str(horizon)] = {
            "cumulative_xyz_response_m": response_xyz_m.tolist(),
            "xy_cosine_to_target_shift": _cosine(response_xy, shift_xy),
            "xy_projected_gain": gain,
        }
    return {
        "actual_target_shift_xyz_m": actual_shift.tolist(),
        "raw_action_delta_rms": float(np.sqrt(np.mean(np.square(raw_delta)))),
        "raw_arm_delta_rms": float(np.sqrt(np.mean(np.square(raw_delta[:, :6])))),
        "raw_arm_xyz_delta_rms": float(
            np.sqrt(np.mean(np.square(raw_delta[:, :3])))
        ),
        "raw_arm_rotation_delta_rms": float(
            np.sqrt(np.mean(np.square(raw_delta[:, 3:6])))
        ),
        "raw_gripper_delta_rms": float(
            np.sqrt(np.mean(np.square(raw_delta[:, 6])))
        ),
        "clipped_action_delta_rms": float(
            np.sqrt(np.mean(np.square(clipped_delta)))
        ),
        "clipped_arm_delta_rms": float(
            np.sqrt(np.mean(np.square(clipped_delta[:, :6])))
        ),
        "clipped_arm_xyz_delta_rms": float(
            np.sqrt(np.mean(np.square(clipped_delta[:, :3])))
        ),
        "clipped_arm_rotation_delta_rms": float(
            np.sqrt(np.mean(np.square(clipped_delta[:, 3:6])))
        ),
        "clipped_gripper_delta_rms": float(
            np.sqrt(np.mean(np.square(clipped_delta[:, 6])))
        ),
        "top_rgb_delta": _array_delta_summary(
            np.asarray(baseline["_top_rgb"]), np.asarray(shifted["_top_rgb"])
        ),
        "wrist_rgb_delta": _array_delta_summary(
            np.asarray(baseline["_wrist_rgb"]),
            np.asarray(shifted["_wrist_rgb"]),
        ),
        "policy_state_max_abs_delta": float(
            np.max(
                np.abs(
                    np.asarray(baseline["policy_state"], dtype=np.float64)
                    - np.asarray(shifted["policy_state"], dtype=np.float64)
                )
            )
        ),
        "bands": bands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--target-body", default="akita_black_bowl_1_main"
    )
    parser.add_argument("--plate-body", default="plate_1_main")
    parser.add_argument(
        "--symmetric-shift-xyz-m",
        type=float,
        nargs=3,
        default=(0.010, 0.0, 0.0),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {args.output}")
    shift = _finite_vector(
        args.symmetric_shift_xyz_m,
        size=3,
        name="symmetric shift",
    )
    if float(np.linalg.norm(shift)) <= 0.0:
        raise ValueError("symmetric shift must be non-zero")
    if args.warmup_steps < 0 or args.image_side <= 0:
        raise ValueError("warmup steps and image side are invalid")

    client = RemotePolicyClient(endpoint=args.endpoint, timeout=float(args.timeout))
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=False)
    bridge_identity = libero_bridge_identity(
        health_before, allow_smoke_policy=False
    )

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    suite = benchmark_map[str(args.suite)](task_order_index=0)
    task = suite.get_task(int(args.task_id))
    _, instruction = _suite_task_metadata(task, task_id=int(args.task_id))
    init_states = np.asarray(
        suite.get_task_init_states(int(args.task_id)), dtype=np.float64
    )
    if not 0 <= int(args.init_state) < len(init_states):
        raise ValueError("init-state index is outside the official inventory")
    initial_state = np.array(init_states[int(args.init_state)], copy=True)
    env = OffScreenRenderEnv(
        bddl_file_name=str(suite.get_task_bddl_file_path(int(args.task_id))),
        camera_heights=int(args.image_side),
        camera_widths=int(args.image_side),
    )
    try:
        env.seed(10000)
        env.reset()
        observation = _set_init_state_observation(env, initial_state)
        zero = np.zeros(7, dtype=np.float32)
        for _ in range(int(args.warmup_steps)):
            observation = _step_observation_only(env, zero.copy())
        base_sim_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        qpos_address = _body_free_joint_qpos_address(env, args.target_body)
        common = {
            "env": env,
            "base_sim_state": base_sim_state,
            "qpos_address": qpos_address,
            "target_body": str(args.target_body),
            "plate_body": str(args.plate_body),
            "instruction": instruction,
            "client": client,
            "quat_to_axisangle": quat2axisangle,
            "image_side": int(args.image_side),
        }
        baseline = _variant(
            requested_shift_xyz_m=np.zeros(3, dtype=np.float64),
            repeat_same_policy_input=True,
            **common,
        )
        plus = _variant(requested_shift_xyz_m=shift, **common)
        minus = _variant(requested_shift_xyz_m=-shift, **common)
        baseline_repeat = _variant(
            requested_shift_xyz_m=np.zeros(3, dtype=np.float64), **common
        )
    finally:
        env.close()

    baseline_raw = np.asarray(baseline["raw_action_chunk"], dtype=np.float64)
    baseline_clipped = np.asarray(
        baseline["clipped_action_chunk"], dtype=np.float64
    )
    repeat_raw = np.asarray(
        baseline_repeat["raw_action_chunk"], dtype=np.float64
    )
    repeat_delta = repeat_raw - baseline_raw
    same_input_repeat_raw = np.asarray(
        baseline["same_input_repeat_raw_action_chunk"], dtype=np.float64
    )
    same_input_repeat_delta = same_input_repeat_raw - baseline_raw
    plus_response = _response_summary(baseline, plus)
    minus_response = _response_summary(baseline, minus)
    baseline_floor_rms = float(np.sqrt(np.mean(np.square(repeat_delta))))
    baseline_raw_rms = float(np.sqrt(np.mean(np.square(baseline_raw))))
    for response in (plus_response, minus_response):
        response_rms = float(response["raw_action_delta_rms"])
        response["raw_action_signal_to_baseline_floor"] = (
            None if baseline_floor_rms <= 0.0 else response_rms / baseline_floor_rms
        )
        response["raw_action_response_to_baseline_rms"] = (
            None if baseline_raw_rms <= 0.0 else response_rms / baseline_raw_rms
        )
        response["raw_action_exceeds_10x_baseline_floor"] = bool(
            response_rms > 10.0 * baseline_floor_rms
        )
    opposite_response: dict[str, Any] = {}
    plus_clipped = np.asarray(plus["clipped_action_chunk"], dtype=np.float64)
    minus_clipped = np.asarray(minus["clipped_action_chunk"], dtype=np.float64)
    for horizon in (1, 4, 8, 24):
        plus_xyz = 0.05 * (
            plus_clipped[:horizon, :3] - baseline_clipped[:horizon, :3]
        ).sum(axis=0)
        minus_xyz = 0.05 * (
            minus_clipped[:horizon, :3] - baseline_clipped[:horizon, :3]
        ).sum(axis=0)
        opposite_response[str(horizon)] = {
            "plus_vs_negative_minus_xy_cosine": _cosine(
                plus_xyz[:2], -minus_xyz[:2]
            ),
            "plus_xyz_response_m": plus_xyz.tolist(),
            "minus_xyz_response_m": minus_xyz.tolist(),
        }
    health_after = client.health()
    if libero_bridge_identity(
        health_after, allow_smoke_policy=False
    ) != bridge_identity:
        raise RuntimeError("bridge identity changed during layout-response probe")

    payload = {
        "schema": "clearvla-libero-layout-response-probe-v3",
        "scope": (
            "diagnostic-only; one-reset target-qpos intervention, not benchmark score"
        ),
        "suite": str(args.suite),
        "task_id": int(args.task_id),
        "instruction": instruction,
        "official_init_state": int(args.init_state),
        "target_body": str(args.target_body),
        "plate_body": str(args.plate_body),
        "target_free_joint_qpos_address": int(qpos_address),
        "warmup_steps": int(args.warmup_steps),
        "image_side": int(args.image_side),
        "bridge_identity": bridge_identity,
        "matched_noise_contract": (
            "every variant uses reset=True; checkpoint policy reseeds its generator"
        ),
        "environment_intervention_contract": (
            "one env reset and official warmup; every variant restores the same "
            "complete post-warmup sim state, edits only target xyz qpos, calls "
            "sim.forward, and refreshes observables without a physics step"
        ),
        "baseline_repeat": {
            "same_policy_input_raw_action_max_abs_delta": float(
                np.max(np.abs(same_input_repeat_delta))
            ),
            "same_policy_input_raw_action_delta_rms": float(
                np.sqrt(np.mean(np.square(same_input_repeat_delta)))
            ),
            "raw_action_max_abs_delta": float(np.max(np.abs(repeat_delta))),
            "raw_action_delta_rms": float(
                np.sqrt(np.mean(np.square(repeat_delta)))
            ),
            "top_rgb_same": (
                baseline["top_rgb_sha256"]
                == baseline_repeat["top_rgb_sha256"]
            ),
            "wrist_rgb_same": (
                baseline["wrist_rgb_sha256"]
                == baseline_repeat["wrist_rgb_sha256"]
            ),
            "top_rgb_delta": _array_delta_summary(
                baseline["_top_rgb"], baseline_repeat["_top_rgb"]
            ),
            "wrist_rgb_delta": _array_delta_summary(
                baseline["_wrist_rgb"], baseline_repeat["_wrist_rgb"]
            ),
            "policy_state_max_abs_delta": float(
                np.max(
                    np.abs(
                        np.asarray(baseline["policy_state"], dtype=np.float64)
                        - np.asarray(
                            baseline_repeat["policy_state"], dtype=np.float64
                        )
                    )
                )
            ),
        },
        "approach": {
            "baseline": _approach_summary(baseline),
            "plus": _approach_summary(plus),
            "minus": _approach_summary(minus),
        },
        "response": {
            "plus": plus_response,
            "minus": minus_response,
            "opposite_direction_check": opposite_response,
        },
        "variants": {
            "baseline": baseline,
            "plus": plus,
            "minus": minus,
            "baseline_repeat": baseline_repeat,
        },
    }
    # Pixel arrays are needed only for the in-process repeat comparison.  Keep
    # the durable artifact compact and leave the exact uint8 content identified
    # by SHA-256.
    for variant in payload["variants"].values():
        variant.pop("_top_rgb", None)
        variant.pop("_wrist_rgb", None)
    atomic_json(args.output, payload)
    print(
        {
            "output": str(args.output),
            "same_input_repeat_max_abs": payload["baseline_repeat"][
                "same_policy_input_raw_action_max_abs_delta"
            ],
            "rerender_repeat_max_abs": payload["baseline_repeat"][
                "raw_action_max_abs_delta"
            ],
            "plus_arm_delta_rms": plus_response["clipped_arm_delta_rms"],
            "minus_arm_delta_rms": minus_response["clipped_arm_delta_rms"],
            "plus_h8_cosine": plus_response["bands"]["8"][
                "xy_cosine_to_target_shift"
            ],
            "minus_h8_cosine": minus_response["bands"]["8"][
                "xy_cosine_to_target_shift"
            ],
        }
    )


if __name__ == "__main__":
    main()
