#!/usr/bin/env python3
"""Audit the coordinate-corrected LIBERO short closed-loop probe.

The v3 probe records simulator-side bowl coordinates, EEF state, gripper
joint positions, and the exact normalized command sent at every policy step.
This read-only analyzer turns those traces into compact, reproducible gauges:
EEF-to-bowl distance, object motion/contact proxies, gripper timing/clipping,
and the action response to the paired ``-0.03/+0.03 m`` layout shift.

This is a diagnostic report, not an official LIBERO score.  In particular, a
small object displacement is evidence that the object did not move in the
bounded horizon; it is not a proof of non-contact in every geometry.
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


SCHEMA = "clearvla-libero-short-closed-loop-probe-v3"
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
ARM_DIM = 6
GRIPPER_DIM = 6
OBJECT_STATIONARY_TOLERANCE_M = 1e-6


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite report value: {value!r}")
    return result


def _quantiles(value: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = _finite_array(value, name="quantiles").reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(array.size),
        "mean": _float(array.mean()),
        "min": _float(array.min()),
        "p50": _float(np.quantile(array, 0.50)),
        "p95": _float(np.quantile(array, 0.95)),
        "max": _float(array.max()),
    }


def _rmse(value: np.ndarray) -> float:
    array = _finite_array(value, name="rmse")
    return _float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0


def _sign_changes(values: np.ndarray, *, threshold: float = 0.05) -> int:
    signs = np.sign(np.asarray(values, dtype=np.float64).reshape(-1))
    signs[np.abs(values.reshape(-1)) < float(threshold)] = 0.0
    signs = signs[signs != 0.0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _require_vector(row: Mapping[str, Any], key: str, *, size: int) -> np.ndarray:
    value = row.get(key)
    if value is None:
        raise ValueError(f"trace row has no {key}")
    array = _finite_array(value, name=key).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{key} must have shape [{size}], got {array.shape}")
    return array


def _rollout_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ValueError("rollout trajectory must contain ready plus policy rows")
    ready = trajectory[0]
    if not isinstance(ready, Mapping) or ready.get("phase") != "ready":
        raise ValueError("first trajectory row is not the ready state")
    policy_rows = trajectory[1:]
    actions = _finite_array(
        [_require_vector(item, "action", size=7) for item in policy_rows],
        name="actions",
        ndim=2,
    )
    if actions.shape[1] != 7:
        raise ValueError(f"actions must be [N,7], got {actions.shape}")
    eef = _finite_array(
        [_require_vector(item, "eef_pos_m", size=3) for item in trajectory],
        name="eef_pos_m",
        ndim=2,
    )
    bowl = _finite_array(
        [_require_vector(item, "object_xyz_m", size=3) for item in trajectory],
        name="object_xyz_m",
        ndim=2,
    )
    gripper_qpos = _finite_array(
        [_require_vector(item, "gripper_qpos", size=2) for item in trajectory],
        name="gripper_qpos",
        ndim=2,
    )
    distances = np.linalg.norm(eef - bowl, axis=1)
    xy_distances = np.linalg.norm((eef - bowl)[:, :2], axis=1)
    object_delta = bowl - bowl[0]
    object_displacement = np.linalg.norm(object_delta, axis=1)
    object_xy_displacement = np.linalg.norm(object_delta[:, :2], axis=1)
    object_abs_z_delta = np.abs(object_delta[:, 2])
    arm = actions[:, :ARM_DIM]
    gripper = actions[:, GRIPPER_DIM]
    audit = row.get("action_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("rollout has no action_audit")
    return {
        "ordinal": int(row["ordinal"]),
        "offset_m": _float(row["offset_m"]),
        "init_state": int(row["init_state"]),
        "steps": int(row["steps"]),
        "success": bool(row["success"]),
        "requested_y": {
            "before_m": _float(row["y_before_m"]),
            "after_m": _float(row["y_after_m"]),
            "applied_before_warmup_m": _float(row["y_applied_before_warmup_m"]),
            "application_error_m": _float(
                row["y_applied_before_warmup_m"] - row["y_after_m"]
            ),
        },
        "initial": {
            "eef_xyz_m": eef[0].tolist(),
            "bowl_xyz_m": bowl[0].tolist(),
            "eef_bowl_distance_m": _float(distances[0]),
            "eef_bowl_xy_distance_m": _float(xy_distances[0]),
            "gripper_qpos": gripper_qpos[0].tolist(),
        },
        "final": {
            "eef_xyz_m": eef[-1].tolist(),
            "bowl_xyz_m": bowl[-1].tolist(),
            "eef_bowl_distance_m": _float(distances[-1]),
            "eef_bowl_xy_distance_m": _float(xy_distances[-1]),
            "gripper_qpos": gripper_qpos[-1].tolist(),
        },
        "distance": {
            "xyz_m": _quantiles(distances),
            "xy_m": _quantiles(xy_distances),
            "initial_to_final_reduction_m": _float(distances[0] - distances[-1]),
            "minimum_xyz_step": int(np.argmin(distances)),
            "minimum_xy_step": int(np.argmin(xy_distances)),
        },
        "eef_motion": {
            "final_displacement_m": _float(np.linalg.norm(eef[-1] - eef[0])),
            "maximum_displacement_m": _float(
                np.linalg.norm(eef - eef[0], axis=1).max()
            ),
            "per_axis_final_delta_m": (eef[-1] - eef[0]).tolist(),
        },
        "object_motion": {
            "final_displacement_m": _float(object_displacement[-1]),
            "maximum_displacement_m": _float(object_displacement.max()),
            "maximum_xy_displacement_m": _float(object_xy_displacement.max()),
            "maximum_abs_z_delta_m": _float(object_abs_z_delta.max()),
            "per_axis_final_delta_m": object_delta[-1].tolist(),
            "y_final_delta_m": _float(bowl[-1, 1] - bowl[0, 1]),
            "xy_stationary_at_tolerance": bool(
                object_xy_displacement.max() <= OBJECT_STATIONARY_TOLERANCE_M
            ),
            "contact_displacement_proxy": bool(
                object_xy_displacement.max() > 1e-4
            ),
        },
        "arm_commands": {
            "l2": _quantiles(np.linalg.norm(arm, axis=1)),
            "mean_abs_per_dim": np.mean(np.abs(arm), axis=0).tolist(),
            "scalar_rms_per_dim": np.sqrt(np.mean(np.square(arm), axis=0)).tolist(),
        },
        "gripper_commands": {
            "min": _float(gripper.min()),
            "max": _float(gripper.max()),
            "mean": _float(gripper.mean()),
            "sign_changes_threshold_0p05": _sign_changes(gripper),
            "rows_at_or_above_plus_0p999": int(np.count_nonzero(gripper >= 0.999)),
            "rows_at_or_below_minus_0p999": int(np.count_nonzero(gripper <= -0.999)),
            "first_row_above_plus_0p5": (
                None
                if not np.any(gripper >= 0.5)
                else int(np.flatnonzero(gripper >= 0.5)[0] + 1)
            ),
            "qpos_opening_width_start_m": _float(np.abs(gripper_qpos[0]).sum()),
            "qpos_opening_width_end_m": _float(np.abs(gripper_qpos[-1]).sum()),
        },
        "clipping": {
            "raw_oob_row_count": int(audit.get("raw_oob_row_count", 0)),
            "raw_oob_count_per_dim": [
                int(value) for value in audit.get("raw_oob_count_per_dim", [0] * 7)
            ],
            "executed_abs_max": [
                _float(value) for value in audit.get("executed_abs_max", [0.0] * 7)
            ],
        },
    }


def _pair_shift_response(
    summaries: Sequence[Mapping[str, Any]],
    executed_rows: Mapping[tuple[float, int], np.ndarray],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    summary_by_condition = {
        (float(item["offset_m"]), int(item["init_state"])): item
        for item in summaries
    }
    init_values = sorted({int(item["init_state"]) for item in summaries})
    for init_state in init_values:
        negative = executed_rows.get((-0.03, init_state))
        positive = executed_rows.get((0.03, init_state))
        if negative is None or positive is None:
            continue
        if negative.shape != positive.shape:
            raise ValueError("paired offset traces have different action shapes")
        delta = positive - negative
        arm_delta = delta[:, :ARM_DIM]
        negative_summary = summary_by_condition[(-0.03, init_state)]
        positive_summary = summary_by_condition[(0.03, init_state)]
        negative_initial_eef = _finite_array(
            negative_summary["initial"]["eef_xyz_m"], name="negative initial EEF"
        )
        positive_initial_eef = _finite_array(
            positive_summary["initial"]["eef_xyz_m"], name="positive initial EEF"
        )
        target_shift = _finite_array(
            positive_summary["initial"]["bowl_xyz_m"], name="positive bowl"
        ) - _finite_array(
            negative_summary["initial"]["bowl_xyz_m"], name="negative bowl"
        )
        final_eef_response = _finite_array(
            positive_summary["final"]["eef_xyz_m"], name="positive final EEF"
        ) - _finite_array(
            negative_summary["final"]["eef_xyz_m"], name="negative final EEF"
        )
        target_y_shift = _float(target_shift[1])
        if abs(target_y_shift) <= 1e-12:
            raise ValueError("paired offsets did not produce a target y shift")
        final_eef_y_response = _float(final_eef_response[1])
        results.append(
            {
                "init_state": int(init_state),
                "steps": int(delta.shape[0]),
                "initial_eef_pair_max_abs_delta_m": _float(
                    np.max(np.abs(positive_initial_eef - negative_initial_eef))
                ),
                "target_shift_positive_minus_negative_xyz_m": target_shift.tolist(),
                "final_eef_response_positive_minus_negative_xyz_m": (
                    final_eef_response.tolist()
                ),
                "final_eef_y_response_m": final_eef_y_response,
                "final_eef_y_response_to_target_y_shift_ratio": _float(
                    final_eef_y_response / target_y_shift
                ),
                "final_eef_y_response_has_target_sign": bool(
                    final_eef_y_response * target_y_shift > 0.0
                ),
                "executed_action_delta_positive_minus_negative": delta.tolist(),
                "arm_scalar_rmse_normalized": _rmse(arm_delta),
                "arm_l2_delta_per_step": np.linalg.norm(arm_delta, axis=1).tolist(),
                "mean_abs_delta_per_dim": np.mean(np.abs(delta), axis=0).tolist(),
                "gripper_delta_mean": _float(delta[:, 6].mean()),
                "gripper_sign_disagreement_fraction": _float(
                    np.mean(np.signbit(positive[:, 6]) != np.signbit(negative[:, 6]))
                ),
            }
        )
    return results


def _aggregate(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> np.ndarray:
        result: list[float] = []
        for item in summaries:
            value: Any = item
            for key in path:
                value = value[key]
            result.append(float(value))
        return np.asarray(result, dtype=np.float64)

    return {
        "rollouts": int(len(summaries)),
        "successes": int(sum(bool(item["success"]) for item in summaries)),
        "arm_l2_mean": _quantiles(values(("arm_commands", "l2", "mean"))),
        "eef_displacement_m": _quantiles(values(("eef_motion", "final_displacement_m"))),
        "initial_eef_bowl_distance_m": _quantiles(
            values(("initial", "eef_bowl_distance_m"))
        ),
        "minimum_eef_bowl_distance_m": _quantiles(
            values(("distance", "xyz_m", "min"))
        ),
        "object_max_displacement_m": _quantiles(
            values(("object_motion", "maximum_displacement_m"))
        ),
        "object_max_xy_displacement_m": _quantiles(
            values(("object_motion", "maximum_xy_displacement_m"))
        ),
        "object_max_abs_z_delta_m": _quantiles(
            values(("object_motion", "maximum_abs_z_delta_m"))
        ),
        "object_xy_stationary_rollouts": int(
            sum(bool(item["object_motion"]["xy_stationary_at_tolerance"]) for item in summaries)
        ),
        "object_contact_proxy_rollouts": int(
            sum(bool(item["object_motion"]["contact_displacement_proxy"]) for item in summaries)
        ),
        "gripper_mean": _quantiles(values(("gripper_commands", "mean"))),
        "gripper_sign_changes": _quantiles(
            values(("gripper_commands", "sign_changes_threshold_0p05"))
        ),
        "gripper_plus_saturation_rows": int(
            sum(int(item["gripper_commands"]["rows_at_or_above_plus_0p999"]) for item in summaries)
        ),
        "raw_oob_rows": int(sum(int(item["clipping"]["raw_oob_row_count"]) for item in summaries)),
    }


def analyze(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one probe JSON is required")
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a v3 probe: {payload.get('schema')!r}")
        if not payload.get("complete"):
            raise ValueError(f"{path} is incomplete")
        if int(payload.get("qpos_flat_offset", -1)) != 1:
            raise ValueError(f"{path} does not carry the expected qpos offset=1")
        rollouts = payload.get("rollouts")
        if not isinstance(rollouts, list) or not rollouts:
            raise ValueError(f"{path} has no rollouts")
        summaries = [_rollout_summary(row) for row in rollouts]
        executed_rows = {
            (float(item["offset_m"]), int(item["init_state"])): _finite_array(
                [trace["action"] for trace in rollouts[index]["trajectory"][1:]],
                name="executed actions",
                ndim=2,
            )
            for index, item in enumerate(summaries)
        }
        paired = _pair_shift_response(summaries, executed_rows)
        all_actions = np.concatenate(list(executed_rows.values()), axis=0)
        arm_scalar_rms = _rmse(all_actions[:, :ARM_DIM])
        paired_arm_rmse = np.asarray(
            [item["arm_scalar_rmse_normalized"] for item in paired],
            dtype=np.float64,
        )
        reports.append(
            {
                "path": str(Path(path).resolve()),
                "checkpoint": str(payload.get("checkpoint", "")),
                "checkpoint_sha256": payload.get("bridge_identity", {}).get(
                    "checkpoint_sha256"
                ),
                "qpos_flat_offset": int(payload["qpos_flat_offset"]),
                "y_qpos_index": int(payload["y_qpos_index"]),
                "rollouts": summaries,
                "aggregate": _aggregate(summaries),
                "paired_offset_response": paired,
                "paired_offset_response_aggregate": {
                    "pairs": int(len(paired)),
                    "typical_arm_scalar_rms_normalized": arm_scalar_rms,
                    "arm_scalar_rmse_normalized": _quantiles(
                        paired_arm_rmse
                    ),
                    "arm_response_rmse_to_typical_arm_rms_ratio": (
                        0.0
                        if not paired
                        else _float(paired_arm_rmse.mean() / max(arm_scalar_rms, 1e-12))
                    ),
                    "final_eef_y_response_m": _quantiles(
                        [item["final_eef_y_response_m"] for item in paired]
                    ),
                    "final_eef_y_response_to_target_y_shift_ratio": _quantiles(
                        [
                            item["final_eef_y_response_to_target_y_shift_ratio"]
                            for item in paired
                        ]
                    ),
                    "final_eef_y_response_correct_sign_pairs": int(
                        sum(
                            bool(item["final_eef_y_response_has_target_sign"])
                            for item in paired
                        )
                    ),
                    "mean_abs_delta_per_dim": (
                        np.mean(
                            np.asarray(
                                [item["mean_abs_delta_per_dim"] for item in paired],
                                dtype=np.float64,
                            ),
                            axis=0,
                        ).tolist()
                        if paired
                        else [0.0] * 7
                    ),
                },
            }
        )
    return {
        "schema": "clearvla-libero-short-closed-loop-audit-v1",
        "diagnostic_only": True,
        "probe_schema": SCHEMA,
        "coordinate_contract": {
            "qpos_flat_offset": 1,
            "y_qpos_index": 10,
            "y_flat_state_index": 11,
            "object_stationary_tolerance_m": OBJECT_STATIONARY_TOLERANCE_M,
        },
        "reports": reports,
        "interpretation": [
            "The v3 state edit is valid only when applied_y equals requested y_after within 1e-10 m.",
            "The bowl contact proxy uses XY displacement (the table-plane motion); a repeatable Z-only drift is treated as simulator settling, not a push.",
            "Offset-response RMSE measures policy action sensitivity to the paired layout shift; it does not by itself prove useful closed-loop control.",
            "The eight-step horizon is a reach/approach diagnostic, not a complete task-success evaluation.",
        ],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
