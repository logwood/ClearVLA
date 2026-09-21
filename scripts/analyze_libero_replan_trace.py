"""Create an event-aligned, placement-aware report from a LIBERO trace.

The input is the diagnostic JSON produced by ``probe_libero_replan_trace.py``.
This tool is intentionally offline: it does not contact the bridge or the
simulator and never treats the diagnostic cadence as an official benchmark.
It separates normal lowering from a placement failure and applies the official
LIBERO ``On`` geometry (contact, plate_z <= bowl_z, XY center distance < 3 cm)
to the recorded body states.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "clearvla-libero-replan-analysis-v1"
ON_XY_THRESHOLD_M = 0.03


def _first(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["step"])
    return None


def _contact(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(row["post"]["contacts"].get(key, 0))
    except (KeyError, TypeError, ValueError):
        return 0


def _action_gripper(row: Mapping[str, Any]) -> float:
    try:
        return float(row["executed_action"][6])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


GRIPPER_CLOSE_THRESHOLD = 0.05
GRIPPER_OPEN_THRESHOLD = -0.05


def _target_z(row: Mapping[str, Any]) -> float:
    try:
        return float(row["post"]["target_displacement_xyz_m"][2])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0


def _xy_distance(row: Mapping[str, Any]) -> float | None:
    try:
        target = np.asarray(row["post"]["target_xyz_m"], dtype=np.float64)
        plate = np.asarray(row["post"]["plate_xyz_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if target.shape != (3,) or plate.shape != (3,):
        return None
    result = float(np.linalg.norm(target[:2] - plate[:2]))
    return result if math.isfinite(result) else None


def _z_delta(row: Mapping[str, Any]) -> float | None:
    try:
        target = float(row["post"]["target_xyz_m"][2])
        plate = float(row["post"]["plate_xyz_m"][2])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    result = target - plate
    return result if math.isfinite(result) else None


def _on_proxy(row: Mapping[str, Any]) -> bool:
    xy = _xy_distance(row)
    z_delta = _z_delta(row)
    # LIBERO's On(bowl, plate) dispatches to plate.check_ontop(bowl):
    # plate_z <= bowl_z, contact, and bowl/plate XY distance < 0.03 m.
    return bool(
        _contact(row, "object_plate_contact_count") > 0
        and xy is not None
        and z_delta is not None
        and xy < ON_XY_THRESHOLD_M
        and z_delta >= 0.0
    )


def _first_lift(rows: Sequence[Mapping[str, Any]]) -> int | None:
    return _first(rows, lambda row: _target_z(row) >= 0.01)


def _first_descent(rows: Sequence[Mapping[str, Any]], lift: int | None) -> int | None:
    if lift is None:
        return None
    running_max = -float("inf")
    seen_lift = False
    for row in rows:
        if int(row["step"]) < int(lift):
            continue
        z = _target_z(row)
        running_max = max(running_max, z)
        seen_lift = seen_lift or z >= 0.01
        if seen_lift and running_max - z >= 0.005:
            return int(row["step"])
    return None


def _first_close_command(rows: Sequence[Mapping[str, Any]]) -> int | None:
    return _first(rows, lambda row: _action_gripper(row) > GRIPPER_CLOSE_THRESHOLD)


def _first_closed_contact(rows: Sequence[Mapping[str, Any]]) -> int | None:
    return _first(
        rows,
        lambda row: _action_gripper(row) > GRIPPER_CLOSE_THRESHOLD
        and _contact(row, "gripper_object_contact_count") > 0,
    )


def _first_open_after_lift(rows: Sequence[Mapping[str, Any]], lift: int | None) -> int | None:
    """Find an actual release-direction command after the object moved up.

    Negative gripper values before a positive close/lift phase are simply the
    approach/open command, not a release.  Requiring the lift boundary keeps
    the event aligned with the physical phase we are diagnosing.
    """

    if lift is None:
        return None
    return _first(
        rows,
        lambda row: int(row["step"]) > int(lift)
        and _action_gripper(row) < GRIPPER_OPEN_THRESHOLD,
    )


def _first_open_after_contact(rows: Sequence[Mapping[str, Any]], grasp: int | None) -> int | None:
    if grasp is None:
        return None
    return _first(
        rows,
        lambda row: int(row["step"]) > int(grasp)
        and _action_gripper(row) < GRIPPER_OPEN_THRESHOLD,
    )


def _first_contact_loss_after_lift(
    rows: Sequence[Mapping[str, Any]], lift: int | None
) -> int | None:
    if lift is None:
        return None
    return _first(
        rows,
        lambda row: int(row["step"]) > int(lift)
        and _contact(row, "gripper_object_contact_count") <= 0,
    )


def _row_at(rows: Sequence[Mapping[str, Any]], step: int | None) -> Mapping[str, Any] | None:
    if step is None:
        return None
    return next((row for row in rows if int(row["step"]) == int(step)), None)


def _metric_at(rows: Sequence[Mapping[str, Any]], step: int | None, fn) -> float | None:
    row = _row_at(rows, step)
    return None if row is None else fn(row)


def _range(values: Sequence[float | None]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(finite)),
        "max": float(max(finite)),
        "mean": float(np.mean(finite)),
    }


def _reference_map(path: Path | None) -> dict[int, bool]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, bool] = {}
    sources: list[Any] = []
    if isinstance(payload.get("rollouts"), list):
        sources.append(payload["rollouts"])
    tasks = payload.get("tasks", {})
    if isinstance(tasks, Mapping):
        for task in tasks.values():
            if isinstance(task, Mapping) and isinstance(task.get("rollouts"), list):
                sources.append(task["rollouts"])
    for rollouts in sources:
        for rollout in rollouts:
            if isinstance(rollout, Mapping) and "init_state" in rollout:
                result[int(rollout["init_state"])] = bool(rollout.get("success", False))
    return result


def _case_summary(
    init_state: int,
    case: Mapping[str, Any],
    reference_success: Mapping[int, bool],
) -> dict[str, Any]:
    result = case.get("result", {})
    rows = list(result.get("rows", []))
    if not rows:
        raise ValueError(f"case {init_state} has no trace rows")
    grasp = _first(rows, lambda row: _contact(row, "gripper_object_contact_count") > 0)
    close_command = _first_close_command(rows)
    closed_contact = _first_closed_contact(rows)
    lift = _first_lift(rows)
    first_open_after_contact = _first_open_after_contact(rows, grasp)
    release = _first_open_after_lift(rows, lift)
    descent = _first_descent(rows, lift)
    contact_loss_after_lift = _first_contact_loss_after_lift(rows, lift)
    first_plate = _first(rows, lambda row: _contact(row, "object_plate_contact_count") > 0)
    plate_after_lift = _first(
        rows,
        lambda row: lift is not None
        and int(row["step"]) >= int(lift)
        and _contact(row, "object_plate_contact_count") > 0,
    )
    plate_after_release = _first(
        rows,
        lambda row: release is not None
        and int(row["step"]) >= int(release)
        and _contact(row, "object_plate_contact_count") > 0,
    )
    on_step = _first(rows, _on_proxy)
    release_row = _row_at(rows, release)
    open_after_contact_row = _row_at(rows, first_open_after_contact)
    final_row = rows[-1]
    post_release = [row for row in rows if release is not None and int(row["step"]) >= int(release)]
    post_release_xy = [_xy_distance(row) for row in post_release]
    post_lift_xy = [
        _xy_distance(row)
        for row in rows
        if lift is not None and int(row["step"]) >= int(lift)
    ]
    max_lift_row = max(rows, key=_target_z)
    xy_release = None if release_row is None else _xy_distance(release_row)
    xy_final = _xy_distance(final_row)
    phase: str
    if grasp is None:
        phase = "no_grasp_observed"
    elif closed_contact is None:
        phase = "contact_without_close_command"
    elif lift is None:
        phase = "closed_contact_without_lift"
    elif on_step is not None:
        phase = "official_on_predicate_observed"
    elif contact_loss_after_lift is not None and (
        release is None or contact_loss_after_lift < release
    ):
        phase = "grasp_lost_before_release"
    elif release is not None and xy_release is not None and xy_release >= ON_XY_THRESHOLD_M:
        phase = "released_off_center"
    elif plate_after_release is not None:
        phase = "plate_contact_without_on"
    elif plate_after_lift is not None:
        phase = "plate_contact_before_release_only"
    else:
        phase = "no_placement_contact_observed"
    actions = np.asarray([row.get("executed_action", [0.0] * 7) for row in rows], dtype=np.float64)
    clipped_rows = int(sum(bool(row.get("clipped_dims")) for row in rows))
    return {
        "init_state": int(init_state),
        "reference_official_success": reference_success.get(int(init_state)),
        "diagnostic_success": bool(result.get("success", False)),
        "steps": int(len(rows)),
        "phase_label": phase,
        "event_steps": {
            "first_gripper_object_contact": grasp,
            "first_close_command": close_command,
            "first_closed_gripper_object_contact": closed_contact,
            "first_lift_over_1cm": lift,
            "first_descent_after_lift": descent,
            "first_open_command_after_contact": first_open_after_contact,
            "first_release_command_after_lift": release,
            # Keep the historical key as an alias for downstream readers.
            "first_release_command_after_grasp": release,
            "first_gripper_object_contact_loss_after_lift": contact_loss_after_lift,
            "first_object_plate_contact_any": first_plate,
            "first_object_plate_contact_after_lift": plate_after_lift,
            "first_object_plate_contact_after_release": plate_after_release,
            "first_official_on_proxy": on_step,
        },
        "placement_geometry": {
            "xy_distance_at_max_lift_m": _metric_at(rows, int(max_lift_row["step"]), _xy_distance),
            "xy_distance_at_release_m": xy_release,
            "xy_distance_at_first_open_after_contact_m": (
                None
                if open_after_contact_row is None
                else _xy_distance(open_after_contact_row)
            ),
            "xy_distance_min_after_lift_m": min(
                (float(value) for value in post_lift_xy if value is not None),
                default=None,
            ),
            "xy_distance_min_after_release_m": min(
                (float(value) for value in post_release_xy if value is not None),
                default=None,
            ),
            "xy_distance_at_final_m": xy_final,
            "z_delta_at_release_m": None if release_row is None else _z_delta(release_row),
            "z_delta_at_first_open_after_contact_m": (
                None
                if open_after_contact_row is None
                else _z_delta(open_after_contact_row)
            ),
            "z_delta_at_final_m": _z_delta(final_row),
            "on_xy_threshold_m": ON_XY_THRESHOLD_M,
        },
        "transport": {
            "initial_target_xyz_m": case.get("ready_target_xyz_m"),
            "initial_plate_xyz_m": case.get("ready_plate_xyz_m"),
            "target_xyz_at_max_lift_m": max_lift_row["post"].get("target_xyz_m"),
            "plate_xyz_at_max_lift_m": max_lift_row["post"].get("plate_xyz_m"),
            "target_xy_transport_to_max_lift_m": (
                None
                if case.get("ready_target_xyz_m") is None
                else (
                    np.asarray(max_lift_row["post"]["target_xyz_m"][:2], dtype=np.float64)
                    - np.asarray(case["ready_target_xyz_m"][:2], dtype=np.float64)
                ).tolist()
            ),
        },
        "action_summary": {
            "plan_count": int(result.get("plan_count", 0)),
            "observe_only_count": int(result.get("observe_only_count", 0)),
            "arm_rms": float(np.sqrt(np.mean(np.square(actions[:, :6])))),
            "gripper_rms": float(np.sqrt(np.mean(np.square(actions[:, 6])))),
            "positive_gripper_fraction": float(np.mean(actions[:, 6] > 0.05)),
            "negative_gripper_fraction": float(np.mean(actions[:, 6] < -0.05)),
            "clipped_row_fraction": float(clipped_rows / max(len(rows), 1)),
            "action_state_matches_previous_executed": bool(
                result.get("action_audit", {}).get("action_state_matches_previous_executed", False)
            ),
        },
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def count(predicate) -> int:
        return int(sum(bool(predicate(case)) for case in cases))

    return {
        "case_count": int(len(cases)),
        "diagnostic_successes": count(lambda case: case["diagnostic_success"]),
        "official_reference_successes": count(lambda case: case["reference_official_success"] is True),
        "diagnostic_reference_disagreements": count(
            lambda case: case["reference_official_success"] is not None
            and case["diagnostic_success"] != case["reference_official_success"]
        ),
        "phase_counts": {
            phase: count(lambda case, phase=phase: case["phase_label"] == phase)
            for phase in sorted({case["phase_label"] for case in cases})
        },
        "event_step_ranges": {
            key: _range([case["event_steps"].get(key) for case in cases])
            for key in (
                "first_gripper_object_contact",
                "first_close_command",
                "first_closed_gripper_object_contact",
                "first_lift_over_1cm",
                "first_descent_after_lift",
                "first_open_command_after_contact",
                "first_release_command_after_lift",
                "first_release_command_after_grasp",
                "first_gripper_object_contact_loss_after_lift",
                "first_official_on_proxy",
            )
        },
        "placement_xy_ranges_m": {
            key: _range([case["placement_geometry"].get(key) for case in cases])
            for key in (
                "xy_distance_at_first_open_after_contact_m",
                "xy_distance_at_release_m",
                "xy_distance_min_after_release_m",
                "xy_distance_at_final_m",
            )
        },
    }


def analyze(trace_path: Path, reference_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "clearvla-libero-replan-trace-v1":
        raise ValueError("trace schema is not clearvla-libero-replan-trace-v1")
    reference = _reference_map(reference_path)
    cases = [
        _case_summary(int(init_state), case, reference)
        for init_state, case in sorted(payload.get("cases", {}).items(), key=lambda item: int(item[0]))
    ]
    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "trace": str(trace_path),
        "reference_result": None if reference_path is None else str(reference_path),
        "execution_contract": payload.get("execution_contract"),
        "on_predicate_contract": {
            "contact": True,
            "plate_z_le_bowl_z": True,
            "xy_center_distance_m_lt": ON_XY_THRESHOLD_M,
        },
        "tail_policy": {
            "late_rows_are_retained": True,
            "late_rows_are_not_used_as_primary_failure_evidence": True,
            "normal_descent_is_not_a_failure_marker": True,
        },
        "aggregate": _aggregate(cases),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze(args.trace, args.reference)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        "aggregate": report["aggregate"],
        "cases": [
            {
                "init_state": case["init_state"],
                "reference": case["reference_official_success"],
                "diagnostic": case["diagnostic_success"],
                "phase": case["phase_label"],
                "events": case["event_steps"],
                "xy_release_m": case["placement_geometry"]["xy_distance_at_release_m"],
                "xy_final_m": case["placement_geometry"]["xy_distance_at_final_m"],
            }
            for case in report["cases"]
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
