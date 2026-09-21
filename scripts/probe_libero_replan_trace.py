"""Trace LIBERO failure phases under chunked receding-horizon execution.

This is a deployment-only diagnostic.  It deliberately does not change the
formal LIBERO evaluator: the official path still executes one row from every
24-row prediction.  With ``--execute-rows 4`` this probe executes four rows
from a plan and appends the three intermediate observations through the
bridge's observe-only endpoint before requesting the next plan.

The durable report separates the useful causal prefix from the post-failure
tail.  A failed episode is not treated as 200 equally informative action
rows: event-aligned fields identify contact, lift, release and placement, and
``effective_analysis_end_step`` stops at the first conservative *phase-failure*
marker plus a short diagnostic window.  A marker is not a proof that recovery
is impossible, so raw rows remain available and later recovery events are
recorded explicitly rather than silently discarded.
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
    _body_position,
    physical_snapshot,
    resolve_contact_geometries,
    restore_snapshot,
)

SCHEMA = "clearvla-libero-replan-trace-v1"
DEFAULT_ENDPOINT = "http://127.0.0.1:18780"
DEFAULT_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0
DEFAULT_STEPS = 120
DEFAULT_EXECUTE_ROWS = 4
DEFAULT_WARMUP_STEPS = 5
DEFAULT_IMAGE_SIDE = 128
DEFAULT_SEED = 10000
DEFAULT_TIMEOUT = 180.0
DEFAULT_TAIL_WINDOW = 8
DEFAULT_GRASP_DEADLINE = 80
DEFAULT_PLACEMENT_GRACE = 80
# MuJoCo contact can flicker for one frame at a time.  Three consecutive
# absent rows is enough to mark a likely release while still tolerating a
# single-frame contact gap; this is only a diagnostic phase marker.
DEFAULT_RELEASE_GRACE = 3
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


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _task_metadata(suite: Any, task_id: int) -> tuple[str, str]:
    task = suite.get_task(int(task_id))
    name = getattr(task, "name", None)
    instruction = getattr(task, "language", None)
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = getattr(task, "language_instruction", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"LIBERO task {task_id} has no name")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"LIBERO task {task_id} has no language instruction")
    return " ".join(name.split()), " ".join(instruction.split())


def _finite_chunk(value: object) -> np.ndarray:
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.shape != (LIBERO_POLICY_HORIZON, 7) or not np.isfinite(chunk).all():
        raise ValueError(
            f"LIBERO bridge must return finite [{LIBERO_POLICY_HORIZON},7] chunk; "
            f"got {chunk.shape}"
        )
    return chunk.copy()


def _contact_count(row: Mapping[str, Any], key: str) -> int:
    contacts = row.get("post", {}).get("contacts", {})
    try:
        return int(contacts.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _target_z(row: Mapping[str, Any]) -> float:
    displacement = row.get("post", {}).get("target_displacement_xyz_m", [0.0, 0.0, 0.0])
    try:
        return float(displacement[2])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _gripper(row: Mapping[str, Any]) -> float:
    action = row.get("executed_action", [0.0] * 7)
    try:
        return float(action[6])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _target_plate_xy_distance(row: Mapping[str, Any]) -> float | None:
    try:
        target = np.asarray(row["post"]["target_xyz_m"], dtype=np.float64)
        plate = np.asarray(row["post"]["plate_xyz_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if target.shape != (3,) or plate.shape != (3,):
        return None
    distance = float(np.linalg.norm(target[:2] - plate[:2]))
    return distance if math.isfinite(distance) else None


def _target_plate_z_delta(row: Mapping[str, Any]) -> float | None:
    try:
        target = float(row["post"]["target_xyz_m"][2])
        plate = float(row["post"]["plate_xyz_m"][2])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    delta = target - plate
    return delta if math.isfinite(delta) else None


def _first_release_command_step(
    rows: Sequence[Mapping[str, Any]], first_grasp: int | None
) -> int | None:
    if first_grasp is None:
        return None
    return _first_step(
        rows,
        lambda row: int(row["step"]) > int(first_grasp) and _gripper(row) < -0.05,
    )


def _first_descent_step(
    rows: Sequence[Mapping[str, Any]], first_lift: int | None
) -> int | None:
    """Describe lowering; this is not itself a failure label."""

    if first_lift is None:
        return None
    running_max = -float("inf")
    lifted_seen = False
    for row in rows:
        step = int(row["step"])
        if step < int(first_lift):
            continue
        z = _target_z(row)
        running_max = max(running_max, z)
        if z >= 0.01:
            lifted_seen = True
        if lifted_seen and running_max - z >= 0.005:
            return step
    return None


def _first_official_on_proxy_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Mirror LIBERO's object ``On`` geometry using the recorded state."""

    def on_row(row: Mapping[str, Any]) -> bool:
        xy = _target_plate_xy_distance(row)
        z_delta = _target_plate_z_delta(row)
        return bool(
            _contact_count(row, "object_plate_contact_count") > 0
            and xy is not None
            and z_delta is not None
            and xy < 0.03
            # LIBERO's ``On(bowl, plate)`` dispatches through the plate state,
            # so the implementation checks plate_z <= bowl_z.
            and z_delta >= 0.0
        )

    return _first_step(
        rows,
        on_row,
    )


def _first_step(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["step"])
    return None


def _failure_markers(
    rows: Sequence[Mapping[str, Any]],
    *,
    grasp_deadline: int,
    placement_grace: int,
    release_grace: int,
) -> list[dict[str, Any]]:
    """Find conservative phase-failure markers, not benchmark scores."""

    if not rows:
        return []
    first_grasp = _first_step(rows, lambda row: _contact_count(row, "gripper_object_contact_count") > 0)
    first_plate = _first_step(rows, lambda row: _contact_count(row, "object_plate_contact_count") > 0)
    markers: list[dict[str, Any]] = []

    if first_grasp is None and int(rows[-1]["step"]) >= int(grasp_deadline):
        markers.append(
            {
                "step": int(grasp_deadline),
                "kind": "no_grasp_by_deadline",
                "detail": "no gripper-object contact through the configured approach deadline",
            }
        )

    # Require a short run of absent contacts.  Single-step contact flicker is
    # common in MuJoCo and should not be called a release by itself.  Even a
    # sustained loss can later be recovered, hence this remains a phase marker.
    if first_grasp is not None and (first_plate is None or first_grasp < first_plate):
        absent = 0
        for row in rows:
            step = int(row["step"])
            if step <= first_grasp:
                continue
            if first_plate is not None and step >= first_plate:
                break
            if _contact_count(row, "gripper_object_contact_count") > 0:
                absent = 0
                continue
            absent += 1
            if absent >= int(release_grace):
                marker_step = step - int(release_grace) + 1
                markers.append(
                    {
                        "step": int(marker_step),
                        "kind": "premature_release_before_plate",
                        "detail": f"gripper-object contact absent for {release_grace} consecutive rows before plate contact",
                    }
                )
                break

    if first_grasp is not None and first_plate is None:
        deadline = first_grasp + int(placement_grace)
        if int(rows[-1]["step"]) >= deadline:
            markers.append(
                {
                    "step": int(deadline),
                    "kind": "no_plate_contact_after_grasp",
                    "detail": "no object-plate contact within the configured post-grasp grace window",
                }
            )

    if first_plate is not None:
        # A negative command is the continuous-chart release direction.  This
        # marker only says that no such command was observed after contact; it
        # does not claim that a command necessarily caused physical release.
        release_step = _first_step(rows, lambda row: int(row["step"]) >= first_plate and _gripper(row) < -0.05)
        deadline = first_plate + int(placement_grace)
        if release_step is None and int(rows[-1]["step"]) >= deadline:
            markers.append(
                {
                    "step": int(deadline),
                    "kind": "no_release_command_after_plate",
                    "detail": "no negative continuous-gripper command within the post-plate grace window",
                }
            )
    return sorted(markers, key=lambda item: (int(item["step"]), str(item["kind"])))


def _late_recovery_events(
    rows: Sequence[Mapping[str, Any]],
    marker: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Record meaningful late events after the analysis-focus boundary.

    This protects the censoring convention from being mistaken for a claim
    that a rollout could not recover after a bad phase.  The detailed raw
    trace remains the primary evidence for any such recovery.
    """

    if marker is None:
        return []
    marker_step = int(marker["step"])
    later = [row for row in rows if int(row["step"]) > marker_step]
    if not later:
        return []
    predicates: tuple[tuple[str, Any], ...] = (
        (
            "gripper_object_contact_after_marker",
            lambda row: _contact_count(row, "gripper_object_contact_count") > 0,
        ),
        (
            "object_plate_contact_after_marker",
            lambda row: _contact_count(row, "object_plate_contact_count") > 0,
        ),
        (
            "negative_gripper_command_after_marker",
            lambda row: _gripper(row) < -0.05,
        ),
        ("success_after_marker", lambda row: bool(row.get("success", False))),
    )
    events: list[dict[str, Any]] = []
    for kind, predicate in predicates:
        step = _first_step(later, predicate)
        if step is not None:
            events.append({"kind": kind, "step": int(step)})
    return events


def summarize_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    success: bool,
    tail_window: int,
    grasp_deadline: int,
    placement_grace: int,
    release_grace: int,
) -> dict[str, Any]:
    """Return event-aligned metrics with an explicit censored tail."""

    if not rows:
        return {
            "total_steps": 0,
            "effective_analysis_end_step": 0,
            "tail_steps_deemphasized": 0,
            "tail_focus_reason": "empty_trace",
            "first_phase_failure_marker": None,
            "late_recovery_events": [],
            "failure_markers": [],
        }
    markers = _failure_markers(
        rows,
        grasp_deadline=grasp_deadline,
        placement_grace=placement_grace,
        release_grace=release_grace,
    )
    first_marker = markers[0] if markers else None
    total_steps = int(rows[-1]["step"])
    if success:
        end_step = total_steps
        censor_reason = "success_or_early_stop"
    elif first_marker is not None:
        end_step = min(total_steps, int(first_marker["step"]) + int(tail_window))
        censor_reason = "first_phase_failure_marker_plus_tail_window"
    else:
        end_step = total_steps
        censor_reason = "no_marker_observed_raw_horizon_retained"
    effective = [row for row in rows if int(row["step"]) <= end_step]
    actions = np.asarray([row.get("executed_action", [0.0] * 7) for row in effective], dtype=np.float64)
    arm_rms = float(np.sqrt(np.mean(np.square(actions[:, :6])))) if len(actions) else 0.0
    gripper_rms = float(np.sqrt(np.mean(np.square(actions[:, 6])))) if len(actions) else 0.0
    first_grasp = _first_step(rows, lambda row: _contact_count(row, "gripper_object_contact_count") > 0)
    first_plate = _first_step(rows, lambda row: _contact_count(row, "object_plate_contact_count") > 0)
    first_lift = _first_step(rows, lambda row: _target_z(row) >= 0.01)
    release_step = _first_release_command_step(rows, first_grasp)
    first_descent = _first_descent_step(rows, first_lift)
    first_on_proxy = _first_official_on_proxy_step(rows)
    release_row = next(
        (row for row in rows if release_step is not None and int(row["step"]) == release_step),
        None,
    )
    final_row = rows[-1]
    plate_after_release = _first_step(
        rows,
        lambda row: release_step is not None
        and int(row["step"]) >= int(release_step)
        and _contact_count(row, "object_plate_contact_count") > 0,
    )
    max_lift = max((_target_z(row) for row in rows), default=0.0)
    return {
        "total_steps": total_steps,
        "effective_analysis_end_step": int(end_step),
        "tail_steps_deemphasized": int(max(0, total_steps - end_step)),
        "tail_focus_reason": censor_reason,
        "first_phase_failure_marker": first_marker,
        "failure_markers": markers,
        "late_recovery_events": _late_recovery_events(rows, first_marker),
        "first_gripper_object_contact_step": first_grasp,
        "first_object_plate_contact_step": first_plate,
        "first_lift_over_1cm_step": first_lift,
        "first_descent_step_after_lift": first_descent,
        "first_release_command_after_grasp_step": release_step,
        "first_plate_contact_after_release_step": plate_after_release,
        "first_official_on_proxy_step": first_on_proxy,
        "placement_geometry": {
            "xy_distance_at_release_m": (
                None if release_row is None else _target_plate_xy_distance(release_row)
            ),
            "z_delta_at_release_m": (
                None if release_row is None else _target_plate_z_delta(release_row)
            ),
            "xy_distance_at_final_m": _target_plate_xy_distance(final_row),
            "z_delta_at_final_m": _target_plate_z_delta(final_row),
            "on_predicate_xy_threshold_m": 0.03,
            "descent_is_not_failure_marker": True,
        },
        "maximum_target_lift_m": float(max_lift),
        "effective_arm_rms": arm_rms,
        "effective_gripper_rms": gripper_rms,
        "effective_row_count": int(len(effective)),
        "event_order": {
            "grasp_before_plate": bool(first_grasp is not None and (first_plate is None or first_grasp <= first_plate)),
            "plate_before_release_command": bool(
                first_plate is not None
                and _first_step(rows, lambda row: int(row["step"]) >= first_plate and _gripper(row) < -0.05) is not None
            ),
        },
        "thresholds": {
            "tail_window_steps": int(tail_window),
            "grasp_deadline_step": int(grasp_deadline),
            "placement_grace_steps": int(placement_grace),
            "contact_absence_release_steps": int(release_grace),
            "lift_threshold_m": 0.01,
            "descent_report_threshold_m": 0.005,
        },
    }


def _run_case(
    *,
    env: Any,
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    instruction: str,
    geometry: Mapping[str, Any],
    target_body: str,
    plate_body: str,
    steps: int,
    execute_rows: int,
    image_side: int,
    quat_to_axisangle: Any,
    initial_target_xyz: np.ndarray,
    initial_plate_xyz: np.ndarray,
    tail_window: int,
    grasp_deadline: int,
    placement_grace: int,
    release_grace: int,
) -> dict[str, Any]:
    previous = np.zeros(7, dtype=np.float32)
    planned: np.ndarray | None = None
    plan_index = -1
    next_chunk_row = 0
    rows: list[dict[str, Any]] = []
    success = False
    observe_indices: list[int] = []

    for step in range(1, int(steps) + 1):
        pre = physical_snapshot(
            env,
            observation,
            target_body=target_body,
            plate_body=plate_body,
            geometry=geometry,
            initial_target_xyz=initial_target_xyz,
            initial_plate_xyz=initial_plate_xyz,
        )
        reset = planned is None or next_chunk_row >= int(execute_rows)
        if reset:
            policy_input = libero_policy_observation(
                observation,
                previous,
                quat_to_axisangle=quat_to_axisangle,
                expected_image_side=image_side,
            )
            chunk = _finite_chunk(
                client.act(policy_input, instruction, reset=(step == 1))
            )
            planned = chunk
            plan_index += 1
            next_chunk_row = 0
            observe_time_index = None
        else:
            # This is the post-action observation for the preceding row.  It
            # must enter bridge history before executing the next planned row.
            policy_input = libero_policy_observation(
                observation,
                previous,
                quat_to_axisangle=quat_to_axisangle,
                expected_image_side=image_side,
            )
            observe_time_index = int(client.observe(policy_input))
            observe_indices.append(observe_time_index)
        assert planned is not None
        chunk_row = int(next_chunk_row)
        raw = planned[chunk_row].copy()
        executed = execute_libero_action(raw)
        action_state_input = previous.copy()
        observation, step_success = _step_with_success(env, executed.copy())
        success = bool(success or step_success)
        post = physical_snapshot(
            env,
            observation,
            target_body=target_body,
            plate_body=plate_body,
            geometry=geometry,
            initial_target_xyz=initial_target_xyz,
            initial_plate_xyz=initial_plate_xyz,
        )
        rows.append(
            {
                "step": int(step),
                "plan_index": int(plan_index),
                "chunk_row": int(chunk_row),
                "replan_boundary": bool(reset),
                "observe_time_index": observe_time_index,
                "action_state_input": action_state_input.tolist(),
                "policy_state_input": np.asarray(policy_input.state, dtype=np.float32).tolist(),
                "raw_action_row": raw.tolist(),
                "executed_action": executed.tolist(),
                "clipped_dims": np.flatnonzero(raw != executed).astype(int).tolist(),
                # Store the complete plan only once, at its boundary.  This
                # preserves exact replan evidence without duplicating 24 rows.
                "raw_action_chunk": planned.tolist() if reset else None,
                "pre": pre,
                "post": post,
                "success": bool(step_success),
            }
        )
        previous = executed.copy()
        next_chunk_row += 1
        if success:
            break

    summary = summarize_trace(
        rows,
        success=success,
        tail_window=tail_window,
        grasp_deadline=grasp_deadline,
        placement_grace=placement_grace,
        release_grace=release_grace,
    )
    actions = np.asarray([row["executed_action"] for row in rows], dtype=np.float64)
    return {
        "execute_rows": int(execute_rows),
        "success": bool(success),
        "steps": int(len(rows)),
        "plan_count": int(plan_index + 1),
        "observe_only_count": int(len(observe_indices)),
        "observe_time_indices": observe_indices,
        "action_audit": {
            "executed_min": actions.min(axis=0).tolist() if len(actions) else [],
            "executed_max": actions.max(axis=0).tolist() if len(actions) else [],
            "arm_rms_raw_horizon": float(np.sqrt(np.mean(np.square(actions[:, :6])))) if len(actions) else 0.0,
            "gripper_values": actions[:, 6].tolist() if len(actions) else [],
            "clipped_row_count": int(sum(bool(row["clipped_dims"]) for row in rows)),
            "action_state_matches_previous_executed": all(
                np.array_equal(
                    np.asarray(row["action_state_input"], dtype=np.float32),
                    np.zeros(7, dtype=np.float32) if index == 0 else actions[index - 1].astype(np.float32),
                )
                for index, row in enumerate(rows)
            ),
        },
        "summary": summary,
        "rows": rows,
    }


def _reference_success_map(path: Path | None) -> dict[int, bool]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result: dict[int, bool] = {}
    # Official LIBERO output nests rollouts under ``tasks``; accept the older
    # top-level list as well so a diagnostic never silently loses labels.
    rollout_sources: list[Any] = []
    top_level = payload.get("rollouts", [])
    if isinstance(top_level, list):
        rollout_sources.append(top_level)
    tasks = payload.get("tasks", {})
    if isinstance(tasks, Mapping):
        for task in tasks.values():
            if isinstance(task, Mapping) and isinstance(task.get("rollouts"), list):
                rollout_sources.append(task["rollouts"])
    for rollouts in rollout_sources:
        for rollout in rollouts:
            if isinstance(rollout, Mapping) and "init_state" in rollout:
                result[int(rollout["init_state"])] = bool(
                    rollout.get("success", False)
                )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    init_states_requested = [int(value) for value in args.init_states]
    if not init_states_requested or len(set(init_states_requested)) != len(init_states_requested):
        raise ValueError("init states must be non-empty and unique")
    steps = _strict_int(args.steps, name="steps", minimum=1)
    execute_rows = _strict_int(args.execute_rows, name="execute_rows", minimum=1)
    if execute_rows > LIBERO_POLICY_HORIZON:
        raise ValueError("execute_rows cannot exceed the policy horizon")
    warmup_steps = _strict_int(args.warmup_steps, name="warmup_steps", minimum=0)
    image_side = _strict_int(args.image_side, name="image_side", minimum=1)
    tail_window = _strict_int(args.tail_window, name="tail_window", minimum=0)
    grasp_deadline = _strict_int(args.grasp_deadline, name="grasp_deadline", minimum=1)
    placement_grace = _strict_int(args.placement_grace, name="placement_grace", minimum=1)
    release_grace = _strict_int(args.release_grace, name="release_grace", minimum=1)
    timeout = _finite_float(args.timeout, name="timeout")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")

    client = RemotePolicyClient(endpoint=str(args.endpoint), timeout=timeout)
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=False)
    bridge_identity = libero_bridge_identity(health_before, allow_smoke_policy=False)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    suite_name = str(args.suite).strip().lower()
    if suite_name not in benchmark_map:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}")
    suite = benchmark_map[suite_name](task_order_index=0)
    task_id = _strict_int(args.task_id, name="task_id", minimum=0)
    task_name, instruction = _task_metadata(suite, task_id)
    init_states = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
    if init_states.ndim != 2 or min(init_states_requested) < 0 or max(init_states_requested) >= len(init_states):
        raise ValueError("an init state is outside the official LIBERO inventory")
    bddl_path = str(suite.get_task_bddl_file_path(task_id))
    reference_success = _reference_success_map(
        Path(args.reference_result).expanduser() if args.reference_result else None
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "complete": False,
        "scope": "LIBERO physical phase/failure diagnostic; never an official score",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task_name,
        "instruction": instruction,
        "init_states": init_states_requested,
        "steps": steps,
        "execute_rows": execute_rows,
        "warmup_steps": warmup_steps,
        "image_side": image_side,
        "seed": int(args.seed),
        "endpoint": str(args.endpoint),
        "checkpoint": str(args.checkpoint),
        "reference_result": (
            None if args.reference_result is None else str(args.reference_result)
        ),
        "reference_success": {str(k): bool(v) for k, v in reference_success.items()},
        "target_body": str(args.target_body),
        "plate_body": str(args.plate_body),
        "bridge_identity": bridge_identity,
        "action_contract": libero_action_contract(),
        "execution_contract": (
            f"diagnostic executes {execute_rows} rows from each [24,7] plan; "
            "rows between replans append current observations through /observe; "
            "the official evaluator remains execute_rows=1"
        ),
        "initialization_contract": (
            "post-warmup state continues directly from set_init_state (official path)"
            if not args.restore_snapshot
            else "post-warmup state is restored through restore_snapshot (non-official paired diagnostic)"
        ),
        "censoring_contract": {
            "tail_window_steps": tail_window,
            "grasp_deadline_step": grasp_deadline,
            "placement_grace_steps": placement_grace,
            "contact_absence_release_steps": release_grace,
            "late_failed_tail": "deemphasized after first phase-failure marker plus tail window; retained verbatim in rows with late-recovery events",
        },
        "cases": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=image_side,
        camera_widths=image_side,
    )
    try:
        # Match the official evaluator: seed the environment once, then reset
        # each fixed init state without reseeding between episodes.
        env.seed(int(args.seed))
        for init_index in init_states_requested:
            env.reset()
            observation = _set_init_state_observation(env, init_states[init_index])
            zero = np.zeros(7, dtype=np.float32)
            for _ in range(warmup_steps):
                observation = _step_observation_only(env, zero.copy())
            snapshot = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            geometry = resolve_contact_geometries(
                env,
                target_body=str(args.target_body),
                plate_body=str(args.plate_body),
            )
            ready_target = _body_position(env, str(args.target_body))
            ready_plate = _body_position(env, str(args.plate_body))
            if args.restore_snapshot:
                observation, controller_count = restore_snapshot(
                    env, snapshot=snapshot, seed=int(args.seed)
                )
            else:
                # The official evaluator continues directly from the fixed
                # init state after warmup.  A flattened MuJoCo snapshot does
                # not include every wrapper/controller field, so restoring it
                # can change the subsequent policy trajectory.
                controller_count = 0
            case = _run_case(
                env=env,
                client=client,
                observation=observation,
                instruction=instruction,
                geometry=geometry,
                target_body=str(args.target_body),
                plate_body=str(args.plate_body),
                steps=steps,
                execute_rows=execute_rows,
                image_side=image_side,
                quat_to_axisangle=quat2axisangle,
                initial_target_xyz=ready_target,
                initial_plate_xyz=ready_plate,
                tail_window=tail_window,
                grasp_deadline=grasp_deadline,
                placement_grace=placement_grace,
                release_grace=release_grace,
            )
            payload["cases"][str(init_index)] = {
                "init_state": int(init_index),
                "reference_success": reference_success.get(int(init_index)),
                "controller_count": int(controller_count),
                "snapshot_length": int(snapshot.size),
                "snapshot_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
                "ready_target_xyz_m": ready_target.tolist(),
                "ready_plate_xyz_m": ready_plate.tolist(),
                "contact_geometry": geometry,
                "result": case,
            }
            atomic_json(output, payload)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    health_after = client.health()
    validate_libero_bridge_health(health_after, allow_smoke_policy=False)
    if libero_bridge_identity(health_after, allow_smoke_policy=False) != bridge_identity:
        raise RuntimeError("bridge identity changed during replan trace")
    payload["bridge_health_before"] = health_before
    payload["bridge_health_after"] = health_after
    payload["complete"] = True
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, default=None)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init-states", type=int, nargs="+", default=[0, 1, 12, 17])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--execute-rows", type=int, default=DEFAULT_EXECUTE_ROWS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--restore-snapshot",
        action="store_true",
        help=(
            "restore the post-warmup MuJoCo snapshot before tracing; this is "
            "non-official and is intended only for paired diagnostic variants"
        ),
    )
    parser.add_argument("--image-side", type=int, default=DEFAULT_IMAGE_SIDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    parser.add_argument("--grasp-deadline", type=int, default=DEFAULT_GRASP_DEADLINE)
    parser.add_argument("--placement-grace", type=int, default=DEFAULT_PLACEMENT_GRACE)
    parser.add_argument("--release-grace", type=int, default=DEFAULT_RELEASE_GRACE)
    parser.add_argument("--target-body", default=DEFAULT_TARGET_BODY)
    parser.add_argument("--plate-body", default=DEFAULT_PLATE_BODY)
    args = parser.parse_args()
    result = run(args)
    compact = {}
    for key, case in result["cases"].items():
        summary = case["result"]["summary"]
        compact[key] = {
            "success": case["result"]["success"],
            "steps": case["result"]["steps"],
            "first_phase_failure": summary.get("first_phase_failure_marker"),
            "effective_end": summary.get("effective_analysis_end_step"),
            "tail_deemphasized": summary.get("tail_steps_deemphasized"),
        }
    print(json.dumps({"output": str(args.output), "cases": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
