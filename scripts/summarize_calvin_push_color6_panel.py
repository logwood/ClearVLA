#!/usr/bin/env python3
"""Summarize the paired 18-case CALVIN push-color closed-loop panel.

The input is the downloaded numeric evidence only: ``result.json``,
``trajectory.npz``, and the separately generated aligned-replan audits.  The
report is descriptive and does not infer a unique internal cause from rollout
behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "clearvla-calvin-push-color6-panel-summary-v1"
VARIANTS = ("camera-v1-eval", "sequence-v1-eval")
BLOCK_ORDER = ("block_blue", "block_red", "block_pink")
TARGET_INDEX = {name: index for index, name in enumerate(BLOCK_ORDER)}
X_ALIGNMENT_M = 0.02
WORKSPACE_SPLIT_M = 0.14


def _float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"expected finite scalar, got {value!r}")
    return result


def _median(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return None if not rows else float(np.median(np.asarray(rows, dtype=np.float64)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _first_true(value: np.ndarray) -> int | None:
    rows = np.flatnonzero(value)
    return None if not len(rows) else int(rows[0])


def _load_case(
    evidence_root: Path,
    replan_root: Path,
    *,
    variant: str,
    case_dir: Path,
) -> dict[str, Any]:
    result_path = case_dir / "result.json"
    trajectory_path = case_dir / "trajectory.npz"
    replan_path = replan_root / variant / f"{case_dir.name}.json"
    result = _read_json(result_path)
    replan = _read_json(replan_path)
    if not bool(result.get("complete")):
        raise ValueError(f"incomplete rollout: {result_path}")
    case = result["case"]
    push = result["push_diagnostics"]
    target = str(push["target_object"])
    if target not in TARGET_INDEX:
        raise ValueError(f"unknown target {target!r}")
    target_index = TARGET_INDEX[target]
    with np.load(trajectory_path, allow_pickle=False) as archive:
        tcp = np.asarray(archive["robot_obs_trajectory"], dtype=np.float64)[:, :3]
        objects = np.asarray(archive["object_positions"], dtype=np.float64)
        contacts = np.asarray(archive["object_robot_contact"], dtype=np.bool_)
    if objects.ndim != 3 or objects.shape[1:] != (3, 3):
        raise ValueError(f"unexpected object_positions shape {objects.shape}")
    if contacts.shape != objects.shape[:2] or tcp.shape != (objects.shape[0], 3):
        raise ValueError(f"trajectory arrays are not aligned in {trajectory_path}")
    target_positions = objects[:, target_index]
    target_contacts = contacts[:, target_index]
    relative = tcp - target_positions
    spatial = np.linalg.norm(relative, axis=1)
    planar = np.linalg.norm(relative[:, :2], axis=1)
    first_contact = _first_true(target_contacts)
    pre_contact_stop = len(relative) if first_contact is None else first_contact + 1
    pre_indices = np.arange(pre_contact_stop, dtype=np.int64)
    nearest_pre = int(pre_indices[np.argmin(spatial[:pre_contact_stop])])
    x_aligned = np.flatnonzero(np.abs(relative[:, 0]) <= X_ALIGNMENT_M)
    first_x_aligned = None if not len(x_aligned) else int(x_aligned[0])
    if len(x_aligned):
        min_abs_y_when_x_aligned = float(np.min(np.abs(relative[x_aligned, 1])))
    else:
        min_abs_y_when_x_aligned = None
    checkpoint = result["checkpoint"]
    task = str(case["task"])
    color = target.removeprefix("block_")
    direction = str(push["direction"])
    initial_x = _float(push["initial_position_m"][0])
    workspace = "near" if initial_x < WORKSPACE_SPLIT_M else "far"
    failure_modes = [str(value) for value in push.get("failure_modes", [])]
    aligned = replan["aligned_replans"]
    return {
        "variant": variant,
        "case": case_dir.name,
        "case_id": int(case["case_id"]),
        "task": task,
        "trial": int(case["trial"]),
        "color": color,
        "direction": direction,
        "workspace": workspace,
        "target_initial_x_m": initial_x,
        "target_initial_y_m": _float(push["initial_position_m"][1]),
        "physical_reset_sha256": str(case["physical_reset_sha256"]),
        "checkpoint_sha256": str(checkpoint["sha256"]),
        "source_commit": str(checkpoint["git_commit"]),
        "success": bool(result["success"]),
        "progress_at_least_100mm": _float(push["target_final_signed_progress_m"]) >= 0.1,
        "direction_correct_1cm": bool(push["direction_correct_1cm"]),
        "target_contact_steps": int(push["target_contact_steps"]),
        "target_contact_positive_progress_steps": int(
            push["target_contact_positive_progress_steps"]
        ),
        "first_target_contact_step": first_contact,
        "target_final_signed_progress_mm": 1000.0
        * _float(push["target_final_signed_progress_m"]),
        "target_max_signed_progress_mm": 1000.0
        * _float(push["target_max_signed_progress_m"]),
        "collateral_objects_2cm": [
            str(value) for value in push.get("collateral_objects_2cm", [])
        ],
        "failure_modes": failure_modes,
        "steps": int(result["steps"]),
        "approach": {
            "nearest_pre_contact_step": nearest_pre,
            "nearest_pre_contact_spatial_mm": 1000.0 * float(spatial[nearest_pre]),
            "nearest_pre_contact_planar_mm": 1000.0 * float(planar[nearest_pre]),
            "nearest_pre_contact_relative_xyz_mm": (
                1000.0 * relative[nearest_pre]
            ).tolist(),
            "first_x_aligned_step": first_x_aligned,
            "min_abs_y_when_x_aligned_mm": (
                None
                if min_abs_y_when_x_aligned is None
                else 1000.0 * min_abs_y_when_x_aligned
            ),
        },
        "replan": {
            "aligned_arm_rmse": _float(aligned["aligned_arm_scalar_rmse_normalized"]),
            "replan_to_same_plan_four_step_ratio": _float(
                aligned["replan_to_same_plan_four_step_delta_rmse_ratio"]
            ),
            "near_translation_negative_fraction": _float(
                aligned["near_old_row4_vs_new_row0_translation_cosine"][
                    "negative_fraction"
                ]
            ),
        },
        "paths": {
            "result": str(result_path.resolve()),
            "trajectory": str(trajectory_path.resolve()),
            "replan": str(replan_path.resolve()),
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(str(value) for value in row["failure_modes"])
    no_contact = [row for row in rows if int(row["target_contact_steps"]) == 0]
    contact = [row for row in rows if int(row["target_contact_steps"]) > 0]
    return {
        "cases": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "progress_at_least_100mm": sum(
            bool(row["progress_at_least_100mm"]) for row in rows
        ),
        "direction_correct_1cm": sum(bool(row["direction_correct_1cm"]) for row in rows),
        "any_target_contact": len(contact),
        "no_target_contact": len(no_contact),
        "collateral_over_2cm": sum(bool(row["collateral_objects_2cm"]) for row in rows),
        "failure_modes": dict(sorted(failures.items())),
        "median_final_signed_progress_mm": _median(
            _float(row["target_final_signed_progress_mm"]) for row in rows
        ),
        "median_aligned_replan_arm_rmse": _median(
            _float(row["replan"]["aligned_arm_rmse"]) for row in rows
        ),
        "median_replan_to_same_plan_four_step_ratio": _median(
            _float(row["replan"]["replan_to_same_plan_four_step_ratio"])
            for row in rows
        ),
        "median_nearest_pre_contact_spatial_mm": _median(
            _float(row["approach"]["nearest_pre_contact_spatial_mm"]) for row in rows
        ),
        "median_nearest_pre_contact_spatial_mm_no_contact": _median(
            _float(row["approach"]["nearest_pre_contact_spatial_mm"])
            for row in no_contact
        ),
        "median_nearest_pre_contact_abs_y_mm_no_contact": _median(
            abs(_float(row["approach"]["nearest_pre_contact_relative_xyz_mm"][1]))
            for row in no_contact
        ),
    }


def _group(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return {
        "/".join(group_key): _aggregate(group_rows)
        for group_key, group_rows in sorted(grouped.items())
    }


def _paired(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[int(row["case_id"])][str(row["variant"])] = row
    pairs: list[dict[str, Any]] = []
    for case_id, arms in sorted(by_case.items()):
        if set(arms) != set(VARIANTS):
            raise ValueError(f"case {case_id} is missing a paired arm")
        camera = arms["camera-v1-eval"]
        sequence = arms["sequence-v1-eval"]
        if camera["physical_reset_sha256"] != sequence["physical_reset_sha256"]:
            raise ValueError(f"case {case_id} reset identities differ")
        pairs.append(
            {
                "case_id": case_id,
                "case": camera["case"],
                "camera_success": bool(camera["success"]),
                "sequence_success": bool(sequence["success"]),
                "camera_contact": int(camera["target_contact_steps"]) > 0,
                "sequence_contact": int(sequence["target_contact_steps"]) > 0,
                "camera_progress_mm": _float(camera["target_final_signed_progress_mm"]),
                "sequence_progress_mm": _float(sequence["target_final_signed_progress_mm"]),
            }
        )
    return {
        "physical_reset_identity_all_match": True,
        "pairs": pairs,
        "both_success": sum(row["camera_success"] and row["sequence_success"] for row in pairs),
        "camera_only_success": sum(
            row["camera_success"] and not row["sequence_success"] for row in pairs
        ),
        "sequence_only_success": sum(
            row["sequence_success"] and not row["camera_success"] for row in pairs
        ),
        "neither_success": sum(
            not row["camera_success"] and not row["sequence_success"] for row in pairs
        ),
    }


def summarize(evidence_root: Path, replan_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rollout_root = evidence_root / variant / "rollouts"
        case_dirs = sorted(
            path for path in rollout_root.iterdir() if (path / "result.json").is_file()
        )
        if len(case_dirs) != 18:
            raise ValueError(f"{variant} requires exactly 18 completed cases, found {len(case_dirs)}")
        rows.extend(
            _load_case(
                evidence_root,
                replan_root,
                variant=variant,
                case_dir=case_dir,
            )
            for case_dir in case_dirs
        )
    checkpoints = {
        variant: sorted(
            {row["checkpoint_sha256"] for row in rows if row["variant"] == variant}
        )
        for variant in VARIANTS
    }
    if any(len(values) != 1 for values in checkpoints.values()):
        raise ValueError("a variant mixed checkpoint identities")
    return {
        "schema": SCHEMA,
        "date": "2026-09-20",
        "diagnostic_not_benchmark_score": True,
        "coordinate_contract": {
            "tcp_xyz": "robot_obs_trajectory[:,0:3]",
            "object_order": list(BLOCK_ORDER),
            "x_alignment_threshold_m": X_ALIGNMENT_M,
            "workspace_split_m": WORKSPACE_SPLIT_M,
            "approach_warning": (
                "TCP-center to block-center geometry; gripper fingers can contact while the "
                "TCP center remains several centimetres away"
            ),
        },
        "checkpoints": checkpoints,
        "aggregate_by_variant": {
            variant: _aggregate([row for row in rows if row["variant"] == variant])
            for variant in VARIANTS
        },
        "aggregate_by_variant_workspace": _group(rows, ("variant", "workspace")),
        "aggregate_by_variant_workspace_direction": _group(
            rows, ("variant", "workspace", "direction")
        ),
        "aggregate_by_variant_color": _group(rows, ("variant", "color")),
        "paired": _paired(rows),
        "cases": rows,
        "limitations": [
            "The two checkpoints are paired on identical physical resets but are not a single-factor architecture ablation.",
            "Replan statistics mix changed observations/history and fresh sampler noise; fixed-snapshot probes are needed for causality.",
            "TCP-center distance is not a collision distance and does not encode finger orientation or contact normal.",
            "Eighteen selected first-task pushes diagnose this panel; they are not the official CALVIN benchmark distribution.",
        ],
    }


def _ratio(successes: int, cases: int) -> str:
    return f"{successes}/{cases}"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate_by_variant"]
    workspace = summary["aggregate_by_variant_workspace"]
    paired = summary["paired"]
    lines = [
        "# 2026-09-20 CALVIN push-color6 closed-loop numeric audit: complete 18 + 18 panel",
        "",
        "This is a numeric diagnostic over autonomous reset rollouts with one executed row per replan. Videos are deliberately left to the user's visual review. Internal causality is not assigned from these rollout statistics alone.",
        "",
        "## Completion and failure decomposition",
        "",
        "| arm | success | >=100 mm signed progress | any target contact | direction correct >=10 mm | collateral >20 mm | median progress (mm) | median replan RMSE | median replan / same-plan-4-step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = aggregate[variant]
        label = variant.removesuffix("-v1-eval")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _ratio(row["successes"], row["cases"]),
                    _ratio(row["progress_at_least_100mm"], row["cases"]),
                    _ratio(row["any_target_contact"], row["cases"]),
                    _ratio(row["direction_correct_1cm"], row["cases"]),
                    _ratio(row["collateral_over_2cm"], row["cases"]),
                    _fmt(row["median_final_signed_progress_mm"]),
                    _fmt(row["median_aligned_replan_arm_rmse"], 4),
                    _fmt(row["median_replan_to_same_plan_four_step_ratio"], 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Workspace split",
            "",
            "The selected panel has 10 far-x targets (about 0.23 m) and 8 near-x targets (about 0.05 m) per arm.",
            "",
            "| arm/workspace | success | contact | direction correct | median progress (mm) | no-contact closest TCP-center distance (mm) | no-contact closest |y| (mm) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        for band in ("far", "near"):
            row = workspace[f"{variant}/{band}"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{variant.removesuffix('-v1-eval')}/{band}",
                        _ratio(row["successes"], row["cases"]),
                        _ratio(row["any_target_contact"], row["cases"]),
                        _ratio(row["direction_correct_1cm"], row["cases"]),
                        _fmt(row["median_final_signed_progress_mm"]),
                        _fmt(row["median_nearest_pre_contact_spatial_mm_no_contact"]),
                        _fmt(row["median_nearest_pre_contact_abs_y_mm_no_contact"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Paired-reset facts",
            "",
            f"All 18 camera/sequence pairs share the same physical reset identity. Outcomes: both succeed {paired['both_success']}, camera-only {paired['camera_only_success']}, sequence-only {paired['sequence_only_success']}, neither {paired['neither_success']}.",
            "",
            "Sequence reaches target contact more often, while camera converts more near-workspace cases into official completion. This separates at least two behavioral deficits—approach/contact acquisition and post-contact direction/stability—but does not by itself identify the responsible module.",
            "",
            "## Full case table",
            "",
            "| arm | case | workspace | success | contacts / positive | final progress mm | failures | collateral | closest pre-contact relative xyz mm | replan RMSE | ratio |",
            "|---|---|---|---:|---:|---:|---|---|---|---:|---:|",
        ]
    )
    for row in summary["cases"]:
        relative = row["approach"]["nearest_pre_contact_relative_xyz_mm"]
        failures = ",".join(row["failure_modes"]) or "-"
        collateral = ",".join(row["collateral_objects_2cm"]) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]).removesuffix("-v1-eval"),
                    str(row["case"]),
                    str(row["workspace"]),
                    "yes" if row["success"] else "no",
                    f"{row['target_contact_steps']} / {row['target_contact_positive_progress_steps']}",
                    _fmt(row["target_final_signed_progress_mm"]),
                    failures,
                    collateral,
                    f"({_fmt(relative[0], 1)}, {_fmt(relative[1], 1)}, {_fmt(relative[2], 1)})",
                    _fmt(row["replan"]["aligned_arm_rmse"], 4),
                    _fmt(row["replan"]["replan_to_same_plan_four_step_ratio"], 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Far-workspace completion is 0/10 in both arms, so the defect is broader than a single color. Contact is not sufficient: several contacted targets move in the wrong direction or stop below threshold.",
            "- Replanning changes the overlapping arm plan by roughly twice a same-plan four-step change in both arms. This is descriptive instability; changed feedback and fresh noise remain mixed here.",
            "- Closest TCP-to-block offsets are center-to-center values. A 30–50 mm offset can coexist with finger contact, so these numbers should be aligned with video/contact traces rather than treated as a hard collision threshold.",
            "- These rollouts localize behavioral regimes. H2/H3/H4/H6/H11 fixed-snapshot and VJP probes are still required before changing architecture.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--replan-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evidence_root.resolve(), args.replan_root.resolve())
    _atomic_text(
        args.json_output.resolve(),
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(args.markdown_output.resolve(), markdown(result))
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "json": str(args.json_output.resolve()),
                "markdown": str(args.markdown_output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
