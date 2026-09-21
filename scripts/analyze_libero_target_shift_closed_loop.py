#!/usr/bin/env python3
"""Summarize the target-shift LIBERO closed-loop probe.

``probe_libero_layout_closed_loop.py`` deliberately records a nested, replayable
trace rather than an evaluator score.  This companion keeps that schema intact
and produces a compact audit containing the complete EEF/target distance
curves, gripper command/event timing, target/plate motion, clipping, and the
paired +/- layout response.  It is intentionally separate from the official
LIBERO evaluator and never treats a target-shift rollout as a benchmark score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROBE_SCHEMA = "clearvla-libero-target-shift-short-closed-loop-v3"
AUDIT_SCHEMA = "clearvla-libero-target-shift-closed-loop-audit-v1"
VARIANTS = ("baseline", "plus", "minus")
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
ARM_DIM = 6
GRIPPER_HOLD_THRESHOLD = 0.05
GRIPPER_STRONG_THRESHOLD = 0.5


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _float(value: Any, *, name: str = "value") -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    value = _array(values, name="quantiles").reshape(-1)
    if value.size == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(value.size),
        "mean": _float(value.mean()),
        "p50": _float(np.quantile(value, 0.50)),
        "p95": _float(np.quantile(value, 0.95)),
        "max": _float(value.max()),
    }


def _norm_rows(rows: Sequence[Mapping[str, Any]], key: str, width: int) -> np.ndarray:
    result = _array([row[key] for row in rows], name=key, ndim=2)
    if result.shape[1] != width:
        raise ValueError(f"{key} must have width {width}, got {result.shape}")
    return result


def _event_label(value: float, threshold: float = GRIPPER_HOLD_THRESHOLD) -> str:
    if value >= threshold:
        return "closed"
    if value <= -threshold:
        return "open"
    return "hold"


def _first_at_or_beyond(values: np.ndarray, threshold: float, *, positive: bool) -> int | None:
    mask = values >= threshold if positive else values <= threshold
    indices = np.flatnonzero(mask)
    return None if indices.size == 0 else int(indices[0] + 1)


def _gripper_summary(raw: np.ndarray, executed: np.ndarray) -> dict[str, Any]:
    raw_grip = raw[:, 6]
    grip = executed[:, 6]
    labels = [_event_label(float(value)) for value in grip]
    transitions = [
        {
            "step": int(index + 1),
            "from": labels[index - 1],
            "to": labels[index],
            "value": _float(grip[index]),
        }
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]
    non_hold = [(index + 1, label) for index, label in enumerate(labels) if label != "hold"]
    non_hold_transitions = [
        {"step": int(step), "from": non_hold[index - 1][1], "to": label}
        for index, (step, label) in enumerate(non_hold)
        if index > 0 and label != non_hold[index - 1][1]
    ]
    return {
        "mapping": {
            "positive": "closed",
            "negative": "open",
            "near_zero": "hold",
            "threshold": GRIPPER_HOLD_THRESHOLD,
        },
        "raw": [float(value) for value in raw_grip],
        "executed": [float(value) for value in grip],
        "executed_labels": labels,
        "executed_counts": {
            "closed": int(sum(label == "closed" for label in labels)),
            "open": int(sum(label == "open" for label in labels)),
            "hold": int(sum(label == "hold" for label in labels)),
        },
        "transitions_including_hold": transitions,
        "transitions_excluding_hold": non_hold_transitions,
        "first_closed_step": _first_at_or_beyond(grip, GRIPPER_HOLD_THRESHOLD, positive=True),
        "first_strong_closed_step": _first_at_or_beyond(
            grip, GRIPPER_STRONG_THRESHOLD, positive=True
        ),
        "first_open_step": _first_at_or_beyond(grip, -GRIPPER_HOLD_THRESHOLD, positive=False),
        "first_strong_open_step": _first_at_or_beyond(
            grip, -GRIPPER_STRONG_THRESHOLD, positive=False
        ),
        "raw_min": _float(raw_grip.min()),
        "raw_max": _float(raw_grip.max()),
        "executed_min": _float(grip.min()),
        "executed_max": _float(grip.max()),
        "raw_rows_at_or_above_plus_one": int(np.count_nonzero(raw_grip >= 1.0)),
        "raw_rows_at_or_below_minus_one": int(np.count_nonzero(raw_grip <= -1.0)),
        "executed_rows_at_plus_one": int(np.count_nonzero(grip >= 1.0)),
        "executed_rows_at_minus_one": int(np.count_nonzero(grip <= -1.0)),
    }


def _trajectory_summary(variant: Mapping[str, Any]) -> dict[str, Any]:
    rows = variant.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("variant has no rows")
    initial = variant.get("initial")
    final = variant.get("final")
    if not isinstance(initial, Mapping) or not isinstance(final, Mapping):
        raise ValueError("variant lacks initial/final state")
    raw = _norm_rows(rows, "raw_action_row", 7)
    executed = _norm_rows(rows, "executed_action", 7)
    if raw.shape != executed.shape:
        raise ValueError("raw and executed action rows are not aligned")
    # ``post`` is a mapping, so keep the extraction explicit for a clear schema error.
    eef_rows = _array([row["post"]["eef_xyz_m"] for row in rows], name="post EEF", ndim=2)
    target_rows = _array(
        [row["post"]["target_xyz_m"] for row in rows], name="post target", ndim=2
    )
    plate_rows = _array(
        [row["post"]["plate_xyz_m"] for row in rows], name="post plate", ndim=2
    )
    if eef_rows.shape[1] != 3 or target_rows.shape != eef_rows.shape or plate_rows.shape != eef_rows.shape:
        raise ValueError("post state vectors must all be [steps,3]")
    eef0 = _array(initial["eef_xyz_m"], name="initial EEF").reshape(3)
    target0 = _array(initial["target_xyz_m"], name="initial target").reshape(3)
    plate0 = _array(initial["plate_xyz_m"], name="initial plate").reshape(3)
    eef_curve = np.vstack((eef0, eef_rows))
    target_curve = np.vstack((target0, target_rows))
    plate_curve = np.vstack((plate0, plate_rows))
    eef_target_delta = eef_curve - target_curve
    xyz_distance = np.linalg.norm(eef_target_delta, axis=1)
    xy_distance = np.linalg.norm(eef_target_delta[:, :2], axis=1)
    target_delta = target_curve - target_curve[0]
    plate_delta = plate_curve - plate_curve[0]
    eef_delta = eef_curve - eef_curve[0]
    clipped = np.any(np.abs(raw - executed) > 0.0, axis=1)
    audit = variant.get("action_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("variant has no action_audit")
    return {
        "steps": int(len(rows)),
        "success": bool(variant.get("summary", {}).get("success", False)),
        "requested_shift_xyz_m": [float(value) for value in variant["requested_shift_xyz_m"]],
        "eef_target_distance_curve": {
            "xyz_m": [float(value) for value in xyz_distance],
            "xy_m": [float(value) for value in xy_distance],
            "initial_xyz_m": _float(xyz_distance[0]),
            "final_xyz_m": _float(xyz_distance[-1]),
            "initial_xy_m": _float(xy_distance[0]),
            "final_xy_m": _float(xy_distance[-1]),
            "xyz_reduction_m": _float(xyz_distance[0] - xyz_distance[-1]),
            "xy_reduction_m": _float(xy_distance[0] - xy_distance[-1]),
            "minimum_xyz_m": _float(xyz_distance.min()),
            "minimum_xy_m": _float(xy_distance.min()),
            "minimum_xyz_step": int(np.argmin(xyz_distance)),
            "minimum_xy_step": int(np.argmin(xy_distance)),
        },
        "eef_xyz_curve_m": eef_curve.tolist(),
        "target_xyz_curve_m": target_curve.tolist(),
        "plate_xyz_curve_m": plate_curve.tolist(),
        "eef_motion": {
            "final_delta_xyz_m": eef_delta[-1].tolist(),
            "final_displacement_m": _float(np.linalg.norm(eef_delta[-1])),
            "maximum_displacement_m": _float(np.linalg.norm(eef_delta, axis=1).max()),
        },
        "target_motion": {
            "final_delta_xyz_m": target_delta[-1].tolist(),
            "final_displacement_m": _float(np.linalg.norm(target_delta[-1])),
            "maximum_displacement_m": _float(np.linalg.norm(target_delta, axis=1).max()),
            "maximum_xy_displacement_m": _float(np.linalg.norm(target_delta[:, :2], axis=1).max()),
            "maximum_abs_z_delta_m": _float(np.abs(target_delta[:, 2]).max()),
        },
        "plate_motion": {
            "final_delta_xyz_m": plate_delta[-1].tolist(),
            "final_displacement_m": _float(np.linalg.norm(plate_delta[-1])),
            "maximum_displacement_m": _float(np.linalg.norm(plate_delta, axis=1).max()),
            "maximum_xy_displacement_m": _float(np.linalg.norm(plate_delta[:, :2], axis=1).max()),
            "maximum_abs_z_delta_m": _float(np.abs(plate_delta[:, 2]).max()),
        },
        "arm_command": {
            "l2_per_step": np.linalg.norm(executed[:, :ARM_DIM], axis=1).tolist(),
            "scalar_rms_per_dim": np.sqrt(np.mean(np.square(executed[:, :ARM_DIM]), axis=0)).tolist(),
            "mean_l2": _float(np.linalg.norm(executed[:, :ARM_DIM], axis=1).mean()),
        },
        "gripper": _gripper_summary(raw, executed),
        "clipping": {
            "reported_raw_oob_row_count": int(audit.get("raw_oob_row_count", -1)),
            "reported_clipped_row_count": int(audit.get("clipped_row_count", -1)),
            "recomputed_oob_row_count": int(clipped.sum()),
            "raw_oob_count_per_dim": [int(value) for value in audit.get("raw_oob_count_per_dim", [])],
            "executed_abs_max": [float(value) for value in audit.get("executed_max", [])],
            "raw_abs_max": np.max(np.abs(raw), axis=0).tolist(),
            "audit_matches_recomputed": bool(int(clipped.sum()) == int(audit.get("clipped_row_count", -2))),
        },
        "same_input_first_plan_repeat": variant.get("same_input_first_plan_repeat"),
        "action_state_matches_executed": bool(audit.get("action_state_matches_executed", False)),
        "per_step": [
            {
                "step": int(index + 1),
                "xyz_distance_m": _float(xyz_distance[index + 1]),
                "xy_distance_m": _float(xy_distance[index + 1]),
                "eef_xyz_m": eef_rows[index].tolist(),
                "target_xyz_m": target_rows[index].tolist(),
                "plate_xyz_m": plate_rows[index].tolist(),
                "arm_command_l2": _float(np.linalg.norm(executed[index, :ARM_DIM])),
                "gripper_raw": _float(raw[index, 6]),
                "gripper_executed": _float(executed[index, 6]),
                "gripper_label": _event_label(float(executed[index, 6])),
                "clipped": bool(clipped[index]),
                "clipped_dims": [int(value) for value in rows[index].get("clipped_dims", [])],
            }
            for index in range(len(rows))
        ],
    }


def _response_summary(case: Mapping[str, Any], name: str) -> dict[str, Any]:
    comparison = case.get("comparisons", {}).get(name)
    if not isinstance(comparison, Mapping):
        raise ValueError(f"missing comparison {name}")
    shift = _array(comparison["actual_initial_target_shift_xyz_m"], name="target shift").reshape(3)
    final = _array(comparison["final_eef_response_xyz_m"], name="EEF response").reshape(3)
    per_step = comparison.get("per_step")
    if not isinstance(per_step, list) or not per_step:
        raise ValueError(f"comparison {name} has no per-step rows")
    action_delta = _array(
        [row["executed_action_delta"] for row in per_step],
        name="executed action delta",
        ndim=2,
    )
    eef_delta = _array(
        [row["eef_response_xyz_m"] for row in per_step],
        name="EEF response curve",
        ndim=2,
    )
    target_norm_sq = float(np.dot(shift[:2], shift[:2]))
    projected_gain = None if target_norm_sq <= 1e-12 else float(np.dot(final[:2], shift[:2]) / target_norm_sq)
    return {
        "comparison_source": name,
        "actual_target_shift_xyz_m": shift.tolist(),
        "final_eef_response_xyz_m": final.tolist(),
        "final_eef_response_xy_projected_gain": projected_gain,
        "final_eef_response_xy_cosine": comparison.get("final_eef_response_xy_cosine_to_target_shift"),
        "final_eef_response_has_target_xy_sign": bool(projected_gain is not None and projected_gain > 0.0),
        "raw_action_delta_rms": _float(comparison["raw_action_delta_rms"]),
        "executed_action_delta_rms": _float(comparison["executed_action_delta_rms"]),
        "raw_arm_delta_rms": _float(comparison["raw_arm_delta_rms"]),
        "raw_gripper_delta_rms": _float(comparison["raw_gripper_delta_rms"]),
        "trajectory_response_xyz_rms_m": _float(comparison["trajectory_response_xyz_rms_m"]),
        "executed_action_delta_per_step": action_delta.tolist(),
        "eef_response_xyz_per_step_m": eef_delta.tolist(),
        "eef_response_norm_per_step_m": np.linalg.norm(eef_delta, axis=1).tolist(),
    }


def _case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    variants = case.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("case has no variants")
    variant_reports = {
        name: _trajectory_summary(variants[name])
        for name in VARIANTS
        if name in variants
    }
    if set(variant_reports) != set(VARIANTS):
        raise ValueError("case does not contain baseline/plus/minus variants")
    stability = case.get("passive_layout_stability")
    if not isinstance(stability, Mapping):
        raise ValueError("case has no passive layout stability")
    stability_report = {
        name: {
            "max_target_displacement_m": _float(stability[name]["max_target_displacement_m"]),
            "max_plate_displacement_m": _float(stability[name]["max_plate_displacement_m"]),
            "steps": int(stability[name]["steps"]),
        }
        for name in VARIANTS
    }
    return {
        "official_init_state": int(case["official_init_state"]),
        "target_free_joint_qpos_address": int(case["target_free_joint_qpos_address"]),
        "restore_control": dict(case.get("restore_control", {})),
        "passive_layout_stability": stability_report,
        "variants": variant_reports,
        "shift_response": {
            "plus_vs_baseline": _response_summary(case, "plus_vs_baseline"),
            "minus_vs_baseline": _response_summary(case, "minus_vs_baseline"),
            "opposite_direction": dict(case.get("comparisons", {}).get("opposite_direction", {})),
        },
    }


def _aggregate(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = [
        (case, case["variants"][variant]) for case in cases for variant in VARIANTS
    ]
    reductions_xyz = [float(row[1]["eef_target_distance_curve"]["xyz_reduction_m"]) for row in rows]
    reductions_xy = [float(row[1]["eef_target_distance_curve"]["xy_reduction_m"]) for row in rows]
    success_count = sum(bool(row[1]["success"]) for row in rows)
    clipped = [int(row[1]["clipping"]["recomputed_oob_row_count"]) for row in rows]
    close_steps = [
        row[1]["gripper"]["first_strong_closed_step"]
        for row in rows
        if row[1]["gripper"]["first_strong_closed_step"] is not None
    ]
    open_steps = [
        row[1]["gripper"]["first_strong_open_step"]
        for row in rows
        if row[1]["gripper"]["first_strong_open_step"] is not None
    ]
    responses = [
        case["shift_response"][name]
        for case in cases
        for name in ("plus_vs_baseline", "minus_vs_baseline")
    ]
    signs = [bool(row["final_eef_response_has_target_xy_sign"]) for row in responses]
    return {
        "rollouts": int(len(rows)),
        "successes": int(success_count),
        "success_rate": float(success_count / len(rows)) if rows else 0.0,
        "xyz_distance_reduction_m": _quantiles(reductions_xyz),
        "xy_distance_reduction_m": _quantiles(reductions_xy),
        "clipped_rows_total": int(sum(clipped)),
        "clipped_rows_per_rollout": clipped,
        "first_strong_close_step": _quantiles(close_steps),
        "first_strong_open_step": _quantiles(open_steps),
        "shift_response_pairs": int(len(responses)),
        "shift_response_target_xy_sign_correct": int(sum(signs)),
        "shift_response_projected_gain": _quantiles(
            [float(row["final_eef_response_xy_projected_gain"]) for row in responses if row["final_eef_response_xy_projected_gain"] is not None]
        ),
    }


def analyze(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one target-shift probe is required")
    reports: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != PROBE_SCHEMA:
            raise ValueError(f"{path} is not a target-shift v3 probe: {payload.get('schema')!r}")
        if payload.get("complete") is not True:
            raise ValueError(f"{path} is incomplete")
        cases = payload.get("cases")
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError(f"{path} has no completed cases")
        case_reports = [_case_summary(cases[key]) for key in sorted(cases, key=int)]
        all_cases.extend(case_reports)
        identity = dict(payload.get("bridge_identity", {}))
        identities.append(identity)
        reports.append(
            {
                "path": str(Path(path).resolve()),
                "suite": payload.get("suite"),
                "task_id": int(payload["task_id"]),
                "task_name": payload.get("task_name"),
                "instruction": payload.get("instruction"),
                "official_init_states": [int(value) for value in payload.get("official_init_states", [])],
                "closed_loop_steps": int(payload["closed_loop_steps"]),
                "warmup_steps": int(payload["warmup_steps"]),
                "symmetric_shift_xyz_m": [float(value) for value in payload["symmetric_shift_xyz_m"]],
                "bridge_identity": identity,
                "passive_limit_m": _float(payload["max_passive_target_displacement_m"]),
                "cases": case_reports,
            }
        )
    identity_consistent = bool(identities) and all(identity == identities[0] for identity in identities[1:])
    return {
        "schema": AUDIT_SCHEMA,
        "diagnostic_only": True,
        "probe_schema": PROBE_SCHEMA,
        "coordinate_contract": {
            "target_shift_axis": "y (actual initial target delta is recorded per pair)",
            "gripper_mapping": "positive=closed, negative=open, near-zero=hold",
            "gripper_hold_threshold": GRIPPER_HOLD_THRESHOLD,
            "gripper_strong_threshold": GRIPPER_STRONG_THRESHOLD,
            "action_execution": "first row of each 24-row chunk, componentwise clip [-1,1], executed row becomes next action_state",
        },
        "identity": {
            "all_probe_bridge_identities_match": identity_consistent,
            "bridge_identities": identities,
        },
        "reports": reports,
        "aggregate": _aggregate(all_cases),
        "interpretation": [
            "Distance curves include the ready state at step 0 and each post-action state; they are approach diagnostics, not task scores.",
            "Target/plate displacement is measured from simulator body positions. Small XY motion with a repeatable Z settling component is not evidence of a successful grasp or push.",
            "A positive projected +/- shift gain means the final EEF response has the target-shift sign; it is a sensitivity check, not proof of useful closed-loop control.",
            "Gripper command events describe the command stream. This probe does not record gripper joint qpos/contact force, so command timing must not be reported as physical grasp timing.",
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = analyze(args.probes)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["analyze"]
