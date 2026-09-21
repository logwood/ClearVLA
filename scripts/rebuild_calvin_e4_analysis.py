"""Rebuild and verify the final 34-case CALVIN E4 rollout analysis.

This script treats the saved official oracle result as the only success label.
It validates every rollout artifact before emitting behavior-level summaries;
it does not infer internal G/S/W/P2 mechanisms from rollout trajectories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "clearvla-calvin-e4-final-analysis-v1"
BLOCK_ORDER = ("block_blue", "block_red", "block_pink")
BLOCK_COLORS = {name: name.removeprefix("block_") for name in BLOCK_ORDER}
MOTION_EPSILON_M = 0.001
DIRECTION_EPSILON_M = 0.01
COLLATERAL_EPSILON_M = 0.02


class AnalysisError(ValueError):
    """Raised when the saved panel violates its artifact contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def finite_list(value: np.ndarray) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    require(np.isfinite(array).all(), "non-finite value in emitted vector")
    return [float(item) for item in array.reshape(-1)]


def first_true(mask: np.ndarray, *, start: int = 1) -> int | None:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    indices = np.flatnonzero(values[start:])
    return int(indices[0] + start) if indices.size else None


def longest_true_run(mask: np.ndarray, *, start: int = 1) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=np.bool_).reshape(-1)[start:]:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def case_name(case: Mapping[str, Any]) -> str:
    return f"{int(case['case_id']):03d}_{case['task']}_t{int(case['trial'])}"


def task_family(task: str) -> str:
    if task.startswith("push_"):
        return "push"
    if task.startswith("lift_"):
        return "lift"
    if "drawer" in task:
        return "drawer"
    if "slider" in task:
        return "slider"
    if task.startswith("turn_on_"):
        return "switch"
    return "other"


def target_object(task: str) -> str | None:
    if not (task.startswith("push_") or task.startswith("lift_")):
        return None
    for color in ("blue", "red", "pink"):
        if f"_{color}_" in f"_{task}_":
            return f"block_{color}"
    raise AnalysisError(f"block task has no recognized color: {task}")


def push_direction(task: str) -> tuple[str, int] | None:
    if not task.startswith("push_"):
        return None
    if task.endswith("_left"):
        return "left", -1
    if task.endswith("_right"):
        return "right", 1
    raise AnalysisError(f"push task has no direction: {task}")


def fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def file_record(path: Path, root: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_posix(path, root),
        "bytes": int(path.stat().st_size),
    }
    if include_sha256:
        record["sha256"] = sha256_file(path)
    return record


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    expected = payload.pop("manifest_sha256", None)
    require(isinstance(expected, str), "manifest_sha256 is absent")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate,duration,width,height,codec_name",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    require(
        completed.returncode == 0,
        f"ffprobe failed for {path}: {completed.stderr.strip()}",
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    require(len(streams) == 1, f"expected one video stream: {path}")
    stream = streams[0]
    frame_text = stream.get("nb_read_frames", stream.get("nb_frames"))
    require(frame_text not in (None, "N/A"), f"ffprobe did not count frames: {path}")
    frame_count = int(frame_text)
    return {
        "frame_count": frame_count,
        "declared_frame_count": (
            int(stream["nb_frames"])
            if stream.get("nb_frames") not in (None, "N/A")
            else None
        ),
        "avg_frame_rate": str(stream.get("avg_frame_rate")),
        "duration_seconds": float(stream["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": str(stream["codec_name"]),
    }


def load_human_annotations(path: Path | None) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = load_json(path)
    require(isinstance(payload, dict), "legacy annotations must be a JSON object")
    rows = payload.get("rows")
    require(isinstance(rows, list), "legacy annotations rows must be a list")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        case_id = int(row["case_id"])
        require(case_id not in by_id, f"duplicate human annotation case {case_id}")
        status = str(row.get("video_review_status", "unknown"))
        source = str(row.get("label_source", "unknown"))
        by_id[case_id] = {
            "legacy_row_present": True,
            "status": status,
            "source": source,
            "label": row.get("video_label"),
            "label_zh": row.get("video_label_zh"),
            "note": row.get("video_note"),
            "video_reviewed": status == "confirmed_video",
            "scope": "human rollout-phenomenon annotation; not an internal-mechanism fact",
        }
    dictionary = payload.get("label_dictionary", {})
    require(isinstance(dictionary, dict), "label_dictionary must be an object")
    return by_id, dictionary


def missing_human_annotation() -> dict[str, Any]:
    return {
        "legacy_row_present": False,
        "status": "not_available",
        "source": None,
        "label": None,
        "label_zh": None,
        "note": None,
        "video_reviewed": False,
        "scope": "no legacy human annotation imported",
    }


def object_metrics(
    positions: np.ndarray,
    contacts: np.ndarray,
) -> dict[str, Any]:
    displacement = positions - positions[:1]
    planar = np.linalg.norm(displacement[:, :2], axis=-1)
    spatial = np.linalg.norm(displacement, axis=-1)
    contact_indices = np.flatnonzero(contacts[1:]) + 1
    return {
        "initial_position_m": finite_list(positions[0]),
        "final_position_m": finite_list(positions[-1]),
        "final_displacement_m": finite_list(displacement[-1]),
        "final_planar_displacement_m": float(planar[-1]),
        "final_spatial_displacement_m": float(spatial[-1]),
        "max_abs_x_displacement_m": float(np.abs(displacement[:, 0]).max()),
        "max_planar_displacement_m": float(planar.max()),
        "max_spatial_displacement_m": float(spatial.max()),
        "max_upward_z_excursion_m": float(displacement[:, 2].max()),
        "max_downward_z_excursion_m": float((-displacement[:, 2]).max()),
        "contact": {
            "initial": bool(contacts[0]),
            "first_step": int(contact_indices[0]) if contact_indices.size else None,
            "last_step": int(contact_indices[-1]) if contact_indices.size else None,
            "count_steps": int(contact_indices.size),
            "longest_consecutive_steps": longest_true_run(contacts),
        },
    }


def tcp_metrics(robot_obs: np.ndarray, steps: int) -> dict[str, Any]:
    tcp = np.asarray(robot_obs[:, :3], dtype=np.float64)
    displacement = tcp - tcp[:1]
    first30_index = min(30, steps)
    return {
        "initial_position_m": finite_list(tcp[0]),
        "final_position_m": finite_list(tcp[-1]),
        "net_displacement_m": finite_list(displacement[-1]),
        "net_displacement_norm_m": float(np.linalg.norm(displacement[-1])),
        "path_length_m": float(np.linalg.norm(np.diff(tcp, axis=0), axis=-1).sum()),
        "first30_step_index": int(first30_index),
        "first30_displacement_m": finite_list(displacement[first30_index]),
        "first30_displacement_norm_m": float(
            np.linalg.norm(displacement[first30_index])
        ),
        "xyz_min_m": finite_list(tcp.min(axis=0)),
        "xyz_max_m": finite_list(tcp.max(axis=0)),
    }


def action_metrics(raw_chunks: np.ndarray, executed: np.ndarray, latency: np.ndarray) -> dict[str, Any]:
    raw_first = raw_chunks[:, 0]
    gripper = executed[:, 6]
    raw_oob_mask = np.any(np.abs(raw_first) > 1.0, axis=1)
    raw_arm_oob_mask = np.any(np.abs(raw_first[:, :6]) > 1.0, axis=1)
    chunk_oob_mask = np.any(np.abs(raw_chunks) > 1.0, axis=2)
    switches = int(np.count_nonzero(np.diff(gripper)))
    return {
        "planning_decisions": int(executed.shape[0]),
        "raw_executed_oob_rows": int(raw_oob_mask.sum()),
        "raw_executed_arm_oob_rows": int(raw_arm_oob_mask.sum()),
        "raw_all_chunk_oob_rows": int(chunk_oob_mask.sum()),
        "executed_arm_boundary_rows": int(
            np.any(np.abs(executed[:, :6]) >= 0.999, axis=1).sum()
        ),
        "executed_arm_abs_max_per_dim": finite_list(
            np.abs(executed[:, :6]).max(axis=0)
        ),
        "gripper_open_steps": int((gripper > 0).sum()),
        "gripper_close_steps": int((gripper < 0).sum()),
        "gripper_switches": switches,
        "gripper_open_fraction": float(np.mean(gripper > 0)),
        "latency_seconds": {
            "mean": float(np.mean(latency)),
            "p50": float(np.quantile(latency, 0.50)),
            "p90": float(np.quantile(latency, 0.90)),
            "p95": float(np.quantile(latency, 0.95)),
            "p99": float(np.quantile(latency, 0.99)),
            "max": float(np.max(latency)),
        },
    }


def push_metrics(
    task: str,
    tcp: np.ndarray,
    positions: np.ndarray,
    contacts: np.ndarray,
) -> dict[str, Any]:
    direction_value = push_direction(task)
    require(direction_value is not None, f"not a push task: {task}")
    direction, sign = direction_value
    target = target_object(task)
    require(target is not None, f"push target absent: {task}")
    target_index = BLOCK_ORDER.index(target)
    target_position = positions[:, target_index]
    target_displacement = target_position - target_position[:1]
    target_contact = contacts[:, target_index]
    signed_progress = float(sign) * target_displacement[:, 0]
    opposite_progress = -signed_progress
    max_signed = float(signed_progress.max())
    final_signed = float(signed_progress[-1])
    max_opposite = float(opposite_progress.max())
    first_contact = first_true(target_contact)
    first_motion = first_true(np.abs(target_displacement[:, 0]) > MOTION_EPSILON_M)
    first_positive = first_true(signed_progress > MOTION_EPSILON_M)
    step_signed_progress = float(sign) * np.diff(
        target_displacement[:, 0], prepend=target_displacement[0, 0]
    )
    contact_positive = target_contact & (step_signed_progress > MOTION_EPSILON_M)
    pre_contact_step = first_contact - 1 if first_contact is not None else None
    pre_contact_relative = (
        tcp[pre_contact_step] - target_position[pre_contact_step]
        if pre_contact_step is not None
        else None
    )
    considered_end = first_contact if first_contact is not None else len(tcp)
    pre_contact_distances = np.linalg.norm(
        tcp[:considered_end] - target_position[:considered_end], axis=-1
    )
    closest_index = int(np.argmin(pre_contact_distances))
    per_object_max_planar = np.linalg.norm(
        (positions - positions[:1])[:, :, :2], axis=-1
    ).max(axis=0)
    return {
        "direction": direction,
        "world_x_sign": int(sign),
        "target_object": target,
        "target_final_signed_progress_m": final_signed,
        "target_max_signed_progress_m": max_signed,
        "target_max_opposite_progress_m": max_opposite,
        "target_reversal_from_best_m": float(max_signed - final_signed),
        "direction_correct_1cm": bool(
            max_signed >= DIRECTION_EPSILON_M and max_signed > max_opposite
        ),
        "first_target_contact_step": first_contact,
        "first_target_motion_step": first_motion,
        "first_positive_progress_step": first_positive,
        "first_contact_positive_progress_step": first_true(contact_positive),
        "target_contact_steps": int(target_contact[1:].sum()),
        "target_contact_positive_progress_steps": int(contact_positive[1:].sum()),
        "pre_contact_step": pre_contact_step,
        "pre_contact_tcp_relative_target_xyz_m": (
            finite_list(pre_contact_relative)
            if pre_contact_relative is not None
            else None
        ),
        "closest_pre_contact_step": closest_index,
        "closest_pre_contact_tcp_target_distance_m": float(
            pre_contact_distances[closest_index]
        ),
        "closest_pre_contact_tcp_relative_target_xyz_m": finite_list(
            tcp[closest_index] - target_position[closest_index]
        ),
        "target_max_upward_z_excursion_m": float(
            target_displacement[:, 2].max()
        ),
        "target_max_downward_z_excursion_m": float(
            (-target_displacement[:, 2]).max()
        ),
        "target_is_most_moved_planar": bool(
            per_object_max_planar[target_index]
            >= per_object_max_planar.max() - 1e-9
        ),
        "target_is_most_moved_planar_1cm": bool(
            per_object_max_planar[target_index]
            >= per_object_max_planar.max() - 1e-9
            and per_object_max_planar[target_index] >= DIRECTION_EPSILON_M
        ),
    }


def computed_flags(
    *,
    task: str,
    success: bool,
    steps: int,
    tcp: Mapping[str, Any],
    objects: Mapping[str, Any],
    target: str | None,
    push: Mapping[str, Any] | None,
) -> dict[str, bool]:
    flags: dict[str, bool] = {
        "official_success": bool(success),
        "official_success_within_180_steps": bool(success and steps <= 180),
        "any_block_contact": any(
            int(metrics["contact"]["count_steps"]) > 0
            for metrics in objects.values()
        ),
        "any_block_max_spatial_motion_ge_2cm": any(
            float(metrics["max_spatial_displacement_m"])
            >= COLLATERAL_EPSILON_M
            for metrics in objects.values()
        ),
    }
    if target is not None:
        target_metrics = objects[target]
        non_targets = [name for name in BLOCK_ORDER if name != target]
        flags.update(
            {
                "target_contact": int(target_metrics["contact"]["count_steps"]) > 0,
                "target_max_planar_motion_ge_1cm": float(
                    target_metrics["max_planar_displacement_m"]
                )
                >= DIRECTION_EPSILON_M,
                "target_final_z_drop_ge_3cm": float(
                    target_metrics["final_displacement_m"][2]
                )
                <= -0.03,
                "non_target_max_spatial_motion_ge_2cm": any(
                    float(objects[name]["max_spatial_displacement_m"])
                    >= COLLATERAL_EPSILON_M
                    for name in non_targets
                ),
                "non_target_contact": any(
                    int(objects[name]["contact"]["count_steps"]) > 0
                    for name in non_targets
                ),
            }
        )
    if push is not None:
        sign = int(push["world_x_sign"])
        first30_x = float(tcp["first30_displacement_m"][0])
        flags.update(
            {
                "push_direction_correct_1cm": bool(push["direction_correct_1cm"]),
                "push_reversal_from_best_ge_1cm": float(
                    push["target_reversal_from_best_m"]
                )
                >= DIRECTION_EPSILON_M,
                "tcp_first30_moves_with_commanded_world_x": sign * first30_x > 0,
                "tcp_first30_moves_against_commanded_world_x": sign * first30_x < 0,
                "target_is_most_moved_planar": bool(
                    push["target_is_most_moved_planar"]
                ),
            }
        )
    return flags


def verify_case(
    *,
    case: Mapping[str, Any],
    case_dir: Path,
    video_root: Path,
    manifest_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    summary_report: Mapping[str, Any],
    human_annotation: Mapping[str, Any],
    ffprobe: str | None,
    artifact_root: Path,
    source_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = case_name(case)
    require(case_dir.name == name, f"case directory mismatch for {name}")
    expected_direct_files = {
        "initial_static.png",
        "initial_wrist.png",
        "final_static.png",
        "final_wrist.png",
        "result.json",
        "trajectory.npz",
    }
    direct_files = {path.name for path in case_dir.iterdir() if path.is_file()}
    direct_dirs = {path.name for path in case_dir.iterdir() if path.is_dir()}
    require(
        direct_files == expected_direct_files,
        f"unexpected direct files in {name}: {sorted(direct_files)}",
    )
    require(direct_dirs == {"video"}, f"unexpected case directories in {name}: {sorted(direct_dirs)}")
    for image_name in (
        "initial_static.png",
        "initial_wrist.png",
        "final_static.png",
        "final_wrist.png",
    ):
        image_path = case_dir / image_name
        require(image_path.stat().st_size > 0, f"empty image: {image_path}")

    result_path = case_dir / "result.json"
    result = load_json(result_path)
    require(result == summary_report, f"summary/result mismatch for {name}")
    require(result["case"] == case, f"manifest/result case mismatch for {name}")
    require(bool(result.get("complete")), f"case not complete: {name}")
    require(bool(result.get("history_aligned")), f"history not aligned: {name}")
    require(
        result.get("manifest_sha256") == manifest_sha256,
        f"manifest identity mismatch: {name}",
    )
    result_checkpoint = result.get("checkpoint", {})
    for key, expected in checkpoint_identity.items():
        require(
            result_checkpoint.get(key) == expected,
            f"checkpoint {key} mismatch: {name}",
        )
    steps = int(result["steps"])
    success = bool(result["success"])
    require(1 <= steps <= int(result["max_steps"]), f"invalid steps: {name}")
    require(success or steps == int(result["max_steps"]), f"early failed case: {name}")

    trajectory_path = case_dir / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as trace:
        required_arrays = {
            "raw_chunks",
            "raw_first",
            "raw_executed",
            "executed",
            "executed_plan_index",
            "executed_chunk_row",
            "latency_seconds",
            "robot_obs_trajectory",
            "scene_obs_trajectory",
            "object_positions",
            "object_robot_contact",
            "oracle_success",
            "target_surface_contact_preserved",
        }
        require(required_arrays <= set(trace.files), f"trajectory keys missing: {name}")
        raw_chunks = np.asarray(trace["raw_chunks"])
        raw_first = np.asarray(trace["raw_first"])
        raw_executed = np.asarray(trace["raw_executed"])
        executed = np.asarray(trace["executed"])
        latency = np.asarray(trace["latency_seconds"])
        robot_obs = np.asarray(trace["robot_obs_trajectory"])
        scene_obs = np.asarray(trace["scene_obs_trajectory"])
        positions = np.asarray(trace["object_positions"])
        contacts = np.asarray(trace["object_robot_contact"])
        oracle = np.asarray(trace["oracle_success"], dtype=np.bool_)
        surface = np.asarray(
            trace["target_surface_contact_preserved"], dtype=np.bool_
        )
        require(
            raw_chunks.ndim == 3
            and raw_chunks.shape[0] == steps
            and raw_chunks.shape[2] == 7,
            f"raw chunk shape mismatch: {name} {raw_chunks.shape}",
        )
        require(raw_first.shape == (steps, 7), f"raw_first shape mismatch: {name}")
        require(
            raw_executed.shape == (steps, 7),
            f"raw_executed shape mismatch: {name}",
        )
        require(executed.shape == (steps, 7), f"executed shape mismatch: {name}")
        require(latency.shape == (steps,), f"latency shape mismatch: {name}")
        require(robot_obs.shape == (steps + 1, 15), f"robot shape mismatch: {name}")
        require(scene_obs.shape == (steps + 1, 24), f"scene shape mismatch: {name}")
        require(
            positions.shape == (steps + 1, len(BLOCK_ORDER), 3),
            f"object position shape mismatch: {name}",
        )
        require(
            contacts.shape == (steps + 1, len(BLOCK_ORDER)),
            f"object contact shape mismatch: {name}",
        )
        require(oracle.shape == (steps + 1,), f"oracle shape mismatch: {name}")
        require(surface.shape == (steps + 1,), f"surface shape mismatch: {name}")
        for array_name, array in (
            ("raw_chunks", raw_chunks),
            ("raw_first", raw_first),
            ("raw_executed", raw_executed),
            ("executed", executed),
            ("latency", latency),
            ("robot_obs", robot_obs),
            ("scene_obs", scene_obs),
            ("positions", positions),
        ):
            require(np.isfinite(array).all(), f"non-finite {array_name}: {name}")
        require(np.array_equal(raw_first, raw_chunks[:, 0]), f"raw_first mismatch: {name}")
        require(
            np.array_equal(raw_executed, raw_chunks[:, 0]),
            f"not row-zero execution: {name}",
        )
        expected_executed = np.clip(raw_chunks[:, 0], -1.0, 1.0)
        expected_executed[:, 6] = np.where(raw_chunks[:, 0, 6] >= 0, 1.0, -1.0)
        require(
            np.array_equal(executed, expected_executed),
            f"executed decoder mismatch: {name}",
        )
        require(
            set(np.unique(executed[:, 6]).tolist()) <= {-1.0, 1.0},
            f"non-binary gripper: {name}",
        )
        require(
            np.array_equal(trace["executed_plan_index"], np.arange(steps)),
            f"plan index mismatch: {name}",
        )
        require(
            np.all(trace["executed_chunk_row"] == 0),
            f"executed nonzero chunk row: {name}",
        )
        require(not bool(oracle[0]), f"oracle true at reset: {name}")
        require(bool(oracle[-1]) == success, f"oracle/result mismatch: {name}")
        require(not bool(oracle[:-1].any()), f"oracle true before terminal step: {name}")

        tcp = tcp_metrics(robot_obs, steps)
        action = action_metrics(raw_chunks, executed, latency)
        objects = {
            block: object_metrics(positions[:, index], contacts[:, index])
            for index, block in enumerate(BLOCK_ORDER)
        }
        target = target_object(str(case["task"]))
        push = (
            push_metrics(
                str(case["task"]),
                np.asarray(robot_obs[:, :3], dtype=np.float64),
                positions,
                contacts,
            )
            if task_family(str(case["task"])) == "push"
            else None
        )
        flags = computed_flags(
            task=str(case["task"]),
            success=success,
            steps=steps,
            tcp=tcp,
            objects=objects,
            target=target,
            push=push,
        )

        official_push = result.get("push_diagnostics", {})
        official_failure_modes: list[str] = []
        if push is not None:
            require(isinstance(official_push, dict) and official_push, f"missing push diagnostics: {name}")
            official_failure_modes = [
                str(value) for value in official_push.get("failure_modes", [])
            ]
            comparison_keys = {
                "target_final_signed_progress_m": "target_final_signed_progress_m",
                "target_max_signed_progress_m": "target_max_signed_progress_m",
                "target_max_opposite_progress_m": "target_max_opposite_progress_m",
                "first_target_contact_step": "first_target_contact_step",
                "first_target_motion_step": "first_target_motion_step",
                "target_contact_steps": "target_contact_steps",
                "target_is_most_moved_planar_1cm": "target_is_most_moved_planar_1cm",
            }
            for rebuilt_key, saved_key in comparison_keys.items():
                rebuilt = push[rebuilt_key]
                saved = official_push[saved_key]
                if isinstance(rebuilt, float):
                    require(
                        math.isclose(rebuilt, float(saved), rel_tol=0.0, abs_tol=1e-12),
                        f"saved push diagnostic mismatch {saved_key}: {name}",
                    )
                else:
                    require(rebuilt == saved, f"saved push diagnostic mismatch {saved_key}: {name}")
        else:
            require(official_push == {}, f"non-push case has push diagnostics: {name}")

        row = {
            "case": name,
            "case_id": int(case["case_id"]),
            "task": str(case["task"]),
            "task_family": task_family(str(case["task"])),
            "trial": int(case["trial"]),
            "instruction": str(case["instruction"]),
            "initial_state": case["initial_state"],
            "physical_reset_sha256": str(case["physical_reset_sha256"]),
            "source_sequence_index": int(case["source_sequence_index"]),
            "source_sequence": list(case["source_sequence"]),
            "official_success": success,
            "steps": steps,
            "official_success_within_180_steps": bool(success and steps <= 180),
            "target_object": target,
            "tcp": tcp,
            "action": action,
            "objects": objects,
            "push": push,
            "official_push_failure_modes": official_failure_modes,
            "target_surface_contact_preserved": (
                {
                    "defined_for_push": True,
                    "final": bool(surface[-1]),
                    "fraction_after_reset": float(surface[1:].mean()),
                    "interpretation_limit": (
                        "contact-pair persistence only; it does not rule out visual or physical penetration"
                    ),
                }
                if push is not None
                else {
                    "defined_for_push": False,
                    "final": None,
                    "fraction_after_reset": None,
                    "interpretation_limit": "not a target metric for this task family",
                }
            ),
            "computed_flags": flags,
            "human_video_annotation": dict(human_annotation),
            "artifact_paths": {
                "result": relative_posix(result_path, artifact_root),
                "trajectory": relative_posix(trajectory_path, artifact_root),
                "video": relative_posix(video_root / f"{name}.mp4", artifact_root),
            },
        }

    result_video = result.get("video", {})
    require(bool(result_video.get("completed")), f"video incomplete: {name}")
    require(
        int(result_video.get("frame_count")) == steps + 7,
        f"result video frame count mismatch: {name}",
    )
    video_dir = case_dir / "video"
    require(video_dir.is_dir(), f"missing video sidecar directory: {name}")
    sidecars = list(video_dir.glob("*.json"))
    require(len(sidecars) == 1, f"expected one video sidecar: {name}")
    require(not list(video_dir.glob("*.mp4")), f"canonical data contains MP4: {name}")
    sidecar = load_json(sidecars[0])
    require(bool(sidecar.get("completed")), f"video sidecar incomplete: {name}")
    require(
        int(sidecar.get("frame_count")) == steps + 7,
        f"video sidecar frame count mismatch: {name}",
    )
    require(
        int(sidecar.get("sequence_index")) == int(case["case_id"]),
        f"video sequence index mismatch: {name}",
    )
    require(sidecar == result_video, f"result/video sidecar mismatch: {name}")
    flat_video = video_root / f"{name}.mp4"
    require(flat_video.is_file() and flat_video.stat().st_size > 0, f"missing flat video: {name}")
    video_sha = sha256_file(flat_video)
    source_sync: dict[str, Any] | None = None
    if source_root is not None:
        source_case = source_root / "shard_0" / name
        require(source_case.is_dir(), f"source-sync case missing: {name}")
        compared_non_video_files: list[dict[str, Any]] = []
        for canonical in sorted(case_dir.rglob("*")):
            if not canonical.is_file():
                continue
            relative = canonical.relative_to(case_dir)
            source = source_case / relative
            require(source.is_file(), f"source-sync file missing: {name}/{relative}")
            canonical_sha = sha256_file(canonical)
            source_sha = sha256_file(source)
            require(
                canonical_sha == source_sha,
                f"canonical/source-sync hash mismatch: {name}/{relative}",
            )
            compared_non_video_files.append(
                {
                    "path": relative.as_posix(),
                    "bytes": int(source.stat().st_size),
                    "sha256": source_sha,
                }
            )
        source_video = source_case / "video" / str(result_video["video"])
        require(source_video.is_file(), f"source-sync nested video missing: {name}")
        source_video_sha = sha256_file(source_video)
        require(
            source_video_sha == video_sha,
            f"flat/source-sync video hash mismatch: {name}",
        )
        source_sync = {
            "verified": True,
            "non_video_files_compared": len(compared_non_video_files),
            "non_video_files": compared_non_video_files,
            "nested_video_path": relative_posix(source_video, source_root),
            "nested_video_bytes": int(source_video.stat().st_size),
            "nested_video_sha256": source_video_sha,
            "nested_video_sha_matches_flat": True,
        }
    video_probe = probe_video(flat_video, ffprobe) if ffprobe is not None else None
    if video_probe is not None:
        require(
            int(video_probe["frame_count"]) == steps + 7,
            f"decoded video frame count mismatch: {name}",
        )
    inventory = {
        "case": name,
        "case_id": int(case["case_id"]),
        "data_files": [
            file_record(path, artifact_root)
            for path in sorted(case_dir.rglob("*"))
            if path.is_file()
        ],
        "flat_video": {
            "path": relative_posix(flat_video, artifact_root),
            "bytes": int(flat_video.stat().st_size),
            "sha256": video_sha,
            "ffprobe": video_probe,
            "frame_count_matches_steps_plus_7": (
                bool(video_probe and int(video_probe["frame_count"]) == steps + 7)
                if video_probe is not None
                else None
            ),
        },
        "source_sync": source_sync,
    }
    return row, inventory


def grouped_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        trials = len(group)
        successes = sum(bool(row["official_success"]) for row in group)
        within_180 = sum(
            bool(row["official_success_within_180_steps"]) for row in group
        )
        return {
            "trials": trials,
            "successes": successes,
            "success_rate": fraction(successes, trials),
            "successes_within_180_steps": within_180,
            "successes_after_180_steps": successes - within_180,
            "mean_steps": float(np.mean([int(row["steps"]) for row in group])),
        }

    by_task_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_family_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_rows[str(row["task"])].append(row)
        by_family_rows[str(row["task_family"])].append(row)
    return {
        "by_task": {task: summarize(group) for task, group in by_task_rows.items()},
        "by_family": {
            family: summarize(group) for family, group in by_family_rows.items()
        },
    }


def build_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    label_dictionary: Mapping[str, Any],
) -> dict[str, Any]:
    grouped = grouped_statistics(rows)
    successes = [row for row in rows if row["official_success"]]
    push_rows = [row for row in rows if row["task_family"] == "push"]
    push_by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("left", "right"):
        subset = [row for row in push_rows if row["push"]["direction"] == direction]
        target_contacts = sum(bool(row["computed_flags"]["target_contact"]) for row in subset)
        push_by_direction[direction] = {
            "trials": len(subset),
            "successes": sum(bool(row["official_success"]) for row in subset),
            "target_contact_cases": target_contacts,
            "target_contact_rate": fraction(target_contacts, len(subset)),
            "direction_correct_1cm_cases": sum(
                bool(row["push"]["direction_correct_1cm"]) for row in subset
            ),
            "mean_tcp_first30_dx_m": float(
                np.mean([row["tcp"]["first30_displacement_m"][0] for row in subset])
            ),
            "tcp_first30_negative_dx_cases": sum(
                float(row["tcp"]["first30_displacement_m"][0]) < 0 for row in subset
            ),
            "tcp_first30_positive_dx_cases": sum(
                float(row["tcp"]["first30_displacement_m"][0]) > 0 for row in subset
            ),
        }
    push_by_color: dict[str, dict[str, Any]] = {}
    for color in ("blue", "red", "pink"):
        subset = [
            row
            for row in push_rows
            if row["target_object"] == f"block_{color}"
        ]
        push_by_color[color] = {
            "trials": len(subset),
            "successes": sum(bool(row["official_success"]) for row in subset),
            "target_contact_cases": sum(
                bool(row["computed_flags"]["target_contact"]) for row in subset
            ),
            "direction_correct_1cm_cases": sum(
                bool(row["push"]["direction_correct_1cm"]) for row in subset
            ),
        }
    push_target_contacts = sum(
        bool(row["computed_flags"]["target_contact"]) for row in push_rows
    )
    push_failure_modes = Counter(
        mode
        for row in push_rows
        for mode in row["official_push_failure_modes"]
    )
    target_y = [
        float(row["objects"][row["target_object"]]["initial_position_m"][1])
        for row in push_rows
    ]
    non_block_rows = [
        row for row in rows if row["task_family"] not in {"push", "lift"}
    ]
    lift_rows = [row for row in rows if row["task_family"] == "lift"]
    lift_summary = {
        "trials": len(lift_rows),
        "successes": sum(bool(row["official_success"]) for row in lift_rows),
        "target_contact_cases": sum(
            bool(row["computed_flags"]["target_contact"]) for row in lift_rows
        ),
        "target_is_most_moved_planar_cases": [
            int(row["case_id"])
            for row in lift_rows
            if row["target_object"] is not None
            and float(
                row["objects"][row["target_object"]]["max_planar_displacement_m"]
            )
            >= max(
                float(row["objects"][name]["max_planar_displacement_m"])
                for name in BLOCK_ORDER
            )
            - 1e-9
        ],
        "target_not_most_moved_planar_cases": [
            int(row["case_id"])
            for row in lift_rows
            if row["target_object"] is not None
            and float(
                row["objects"][row["target_object"]]["max_planar_displacement_m"]
            )
            < max(
                float(row["objects"][name]["max_planar_displacement_m"])
                for name in BLOCK_ORDER
            )
            - 1e-9
        ],
    }
    incidental_cases = [
        {
            "case_id": int(row["case_id"]),
            "case": row["case"],
            "task": row["task"],
            "contacted_blocks": [
                name
                for name in BLOCK_ORDER
                if int(row["objects"][name]["contact"]["count_steps"]) > 0
            ],
            "moved_blocks_ge_2cm": [
                name
                for name in BLOCK_ORDER
                if float(row["objects"][name]["max_spatial_displacement_m"])
                >= COLLATERAL_EPSILON_M
            ],
        }
        for row in non_block_rows
        if row["computed_flags"]["any_block_contact"]
        or row["computed_flags"]["any_block_max_spatial_motion_ge_2cm"]
    ]
    human_status = Counter(
        str(row["human_video_annotation"]["status"]) for row in rows
    )
    human_labels = Counter(
        str(row["human_video_annotation"]["label"])
        for row in rows
        if row["human_video_annotation"]["label"] is not None
    )
    action_decisions = sum(int(row["action"]["planning_decisions"]) for row in rows)
    raw_oob_rows = sum(int(row["action"]["raw_executed_oob_rows"]) for row in rows)
    weighted_latency_sum = sum(
        float(row["action"]["latency_seconds"]["mean"])
        * int(row["action"]["planning_decisions"])
        for row in rows
    )
    push_surface_drop_cases = [
        {
            "case_id": int(row["case_id"]),
            "case": row["case"],
            "target_final_z_delta_m": float(
                row["objects"][row["target_object"]]["final_displacement_m"][2]
            ),
            "surface_contact_preserved_fraction": float(
                row["target_surface_contact_preserved"]["fraction_after_reset"]
            ),
        }
        for row in push_rows
        if row["computed_flags"].get("target_final_z_drop_ge_3cm", False)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": {
            "panel": manifest["statistical_scope"],
            "official_success_source": "saved CALVIN oracle trace and result.json",
            "mechanism_limit": (
                "rollout behavior cannot by itself localize a fault to G, S, W, P2, or the bottom action generator"
            ),
            "direction_causality_limit": (
                "left/right cases use different resets; this is not a fixed-reset, matched-noise language counterfactual"
            ),
            "target_y_limit": (
                "all push target initial y values are nearly identical, so this panel cannot estimate target-y tracking gain"
            ),
            "surface_metric_limit": (
                "surface_contact_preserved records contact-pair persistence and cannot rule out penetration"
            ),
        },
        "run_contract": {
            "model_contract": manifest["model_contract"],
            "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_epoch": int(checkpoint["metadata"]["epoch"]),
            "checkpoint_global_step": int(checkpoint["metadata"]["global_step"]),
            "checkpoint_source_commit": checkpoint["source_commit"],
            "max_steps": int(manifest["max_steps"]),
            "execute_rows": int(manifest["execute_rows"]),
        },
        "panel": {
            "trials": len(rows),
            "successes": len(successes),
            "success_rate": fraction(len(successes), len(rows)),
            "successes_within_180_steps": sum(
                bool(row["official_success_within_180_steps"]) for row in rows
            ),
            "successes_after_180_steps": sum(
                bool(row["official_success"])
                and not bool(row["official_success_within_180_steps"])
                for row in rows
            ),
            "successful_cases": [
                {
                    "case_id": int(row["case_id"]),
                    "case": row["case"],
                    "task": row["task"],
                    "steps": int(row["steps"]),
                }
                for row in successes
            ],
        },
        **grouped,
        "push": {
            "trials": len(push_rows),
            "successes": sum(bool(row["official_success"]) for row in push_rows),
            "target_contact_cases": push_target_contacts,
            "target_contact_rate": fraction(push_target_contacts, len(push_rows)),
            "by_direction": push_by_direction,
            "by_color": push_by_color,
            "target_initial_y_m": {
                "min": float(min(target_y)),
                "max": float(max(target_y)),
                "mean": float(np.mean(target_y)),
                "range": float(max(target_y) - min(target_y)),
            },
            "target_final_z_drop_ge_3cm_cases": push_surface_drop_cases,
            "target_not_most_moved_planar_cases": [
                int(row["case_id"])
                for row in push_rows
                if not bool(row["push"]["target_is_most_moved_planar"])
            ],
            "reversal_from_best_ge_1cm_cases": [
                int(row["case_id"])
                for row in push_rows
                if bool(row["computed_flags"]["push_reversal_from_best_ge_1cm"])
            ],
            "official_failure_modes": dict(sorted(push_failure_modes.items())),
        },
        "lift": lift_summary,
        "non_block_task_block_interactions": {
            "terminology": "incidental/collateral only; these are not target-binding labels",
            "cases": incidental_cases,
        },
        "action_integrity": {
            "planning_decisions": action_decisions,
            "raw_executed_oob_rows": raw_oob_rows,
            "raw_executed_oob_fraction": fraction(raw_oob_rows, action_decisions),
            "all_executed_rows_match_clipped_raw_first": True,
            "all_gripper_commands_binary": True,
            "mean_policy_latency_seconds": float(
                weighted_latency_sum / action_decisions
            ),
        },
        "human_video_annotations": {
            "status_counts": dict(sorted(human_status.items())),
            "label_counts": dict(sorted(human_labels.items())),
            "label_dictionary": dict(label_dictionary),
            "confirmed_video_cases": [
                int(row["case_id"])
                for row in rows
                if bool(row["human_video_annotation"]["video_reviewed"])
            ],
            "note": (
                "human labels remain separate from computed trajectory flags; success references marked reference_not_video_reviewed are not counted as video-reviewed"
            ),
        },
        "thresholds": {
            "motion_epsilon_m": MOTION_EPSILON_M,
            "direction_epsilon_m": DIRECTION_EPSILON_M,
            "collateral_epsilon_m": COLLATERAL_EPSILON_M,
        },
        "saved_summary_consistency": {
            "successes": int(summary["successes"]),
            "trials": int(summary["trials"]),
            "matched": int(summary["successes"]) == len(successes)
            and int(summary["trials"]) == len(rows),
        },
    }


def mm(value_m: float) -> str:
    return f"{1000.0 * value_m:.1f} mm"


def case_by_id(rows: Sequence[Mapping[str, Any]], case_id: int) -> Mapping[str, Any]:
    matches = [row for row in rows if int(row["case_id"]) == case_id]
    require(len(matches) == 1, f"expected case {case_id}")
    return matches[0]


def render_summary(rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> str:
    panel = stats["panel"]
    push = stats["push"]
    left = push["by_direction"]["left"]
    right = push["by_direction"]["right"]
    by_task = stats["by_task"]
    case010 = case_by_id(rows, 10)
    case011 = case_by_id(rows, 11)
    case018 = case_by_id(rows, 18)
    case020 = case_by_id(rows, 20)
    case016 = case_by_id(rows, 16)
    case032 = case_by_id(rows, 32)
    c010_blue = case010["objects"]["block_blue"]
    c011_blue = case011["objects"]["block_blue"]
    c011_red = case011["objects"]["block_red"]
    c020_pink = case020["objects"]["block_pink"]
    successful_lines = "\n".join(
        f"- `{item['case']}`：{item['steps']} 步"
        for item in panel["successful_cases"]
    )
    task_lines = "\n".join(
        f"- `{task}`：{values['successes']}/{values['trials']}"
        for task, values in by_task.items()
    )
    z_drop_lines = "\n".join(
        f"- case {item['case_id']:03d}：目标最终 z 变化 {mm(item['target_final_z_delta_m'])}，"
        f"surface-contact preserved={item['surface_contact_preserved_fraction']:.3f}"
        for item in push["target_final_z_drop_ge_3cm_cases"]
    )
    return f"""# CALVIN E4 最终 34-case 重建分析

## 结论

这次面板已完整闭合：34/34 个案例、轨迹和视频均通过一致性检查。官方 CALVIN oracle 成功 **{panel['successes']}/{panel['trials']} ({panel['success_rate']:.1%})**；180 步内成功 {panel['successes_within_180_steps']} 个，另有 {panel['successes_after_180_steps']} 个成功发生在 180 步以后。

最强的行为证据不是“机械臂不会产生动作”，而是**动作通道与当前任务对象/方向的绑定不稳定**：push 仅 {push['successes']}/{push['trials']} 成功，目标接触 {push['target_contact_cases']}/{push['trials']}；左推目标接触 {left['target_contact_cases']}/{left['trials']}，右推只有 {right['target_contact_cases']}/{right['trials']}。这能支持继续检查对象条件化几何和其动作消费路径，但闭环 rollout 本身不能把故障定位到 G、S、W、P2 或底层流场中的某一层。

## 官方结果

{task_lines}

成功案例：

{successful_lines}

## 方向与目标绑定

- 左推：成功 {left['successes']}/{left['trials']}，目标接触 {left['target_contact_cases']}/{left['trials']}。
- 右推：成功 {right['successes']}/{right['trials']}，目标接触 {right['target_contact_cases']}/{right['trials']}。
- 9 个右推案例在前 30 步的 TCP world-x 位移有 {right['tcp_first30_negative_dx_cases']}/9 为负，平均 {mm(right['mean_tcp_first30_dx_m'])}；即早期动作一致朝右推目标方向的反向移动。
- 18 个 push 目标初始 y 的范围仅 {mm(push['target_initial_y_m']['range'])}（均值 {push['target_initial_y_m']['mean']:.6f} m）。因此这个面板不能估计目标 y 平移跟随增益，也不能替代固定 reset、固定噪声的目标位置因果实验。
- 官方 push 诊断中的失败模式计数为：`no_target_contact` {push['official_failure_modes'].get('no_target_contact', 0)}、`insufficient_progress` {push['official_failure_modes'].get('insufficient_progress', 0)}、`target_contact_no_motion` {push['official_failure_modes'].get('target_contact_no_motion', 0)}、`non_target_collateral_motion` {push['official_failure_modes'].get('non_target_collateral_motion', 0)}。这些是轨迹层面的筛查标签，不是内部模块归因。
- 按目标颜色看，blue 为 {push['by_color']['blue']['successes']}/{push['by_color']['blue']['trials']} 成功且目标接触 {push['by_color']['blue']['target_contact_cases']}/{push['by_color']['blue']['trials']}，red 为 {push['by_color']['red']['successes']}/{push['by_color']['red']['trials']} 且接触 {push['by_color']['red']['target_contact_cases']}/{push['by_color']['red']['trials']}，pink 为 {push['by_color']['pink']['successes']}/{push['by_color']['pink']['trials']} 且接触 {push['by_color']['pink']['target_contact_cases']}/{push['by_color']['pink']['trials']}。

## 具体失败证据

- case 010 指令是 lift red，但红块未形成有效目标接触；蓝块接触 {c010_blue['contact']['count_steps']} 步，最大空间位移 {c010_blue['max_spatial_displacement_m']:.3f} m。
- case 011 指令是 lift blue；蓝块接触 {c011_blue['contact']['count_steps']} 步、最大空间位移 {c011_blue['max_spatial_displacement_m']:.3f} m，红块接触 {c011_red['contact']['count_steps']} 步且最大空间位移 {c011_red['max_spatial_displacement_m']:.3f} m。
- 4 个 lift 案例中目标接触 {stats['lift']['target_contact_cases']}/4；case 011 的目标不是平面位移最大的物块，说明“发生了接触”也不能直接等同于目标身份绑定正确。
- case 020 是 open drawer，却接触粉块 {c020_pink['contact']['count_steps']} 步并使其最大空间位移达到 {c020_pink['max_spatial_displacement_m']:.3f} m。这里必须称为 incidental/collateral block interaction，不能称为非块任务的“目标绑定”。
- case 016 目标最终位移为 dx={case016['objects'][case016['target_object']]['final_displacement_m'][0]:+.3f} m、dy={case016['objects'][case016['target_object']]['final_displacement_m'][1]:+.3f} m；case 032 为 dx={case032['objects'][case032['target_object']]['final_displacement_m'][0]:+.3f} m、dy={case032['objects'][case032['target_object']]['final_displacement_m'][1]:+.3f} m，二者都不是纯 world-x 推送。
- case 018 最大正确推进 {case018['push']['target_max_signed_progress_m']:.3f} m，最终只保留 {case018['push']['target_final_signed_progress_m']:.3f} m，回撤 {case018['push']['target_reversal_from_best_m']:.3f} m。

目标最终 z 下降至少 30 mm、但 contact-pair persistence 仍为真的 push 案例：

{z_drop_lines}

这说明 `surface_contact_preserved` 只验证接触对仍存在，不能用来排除过压或穿透。

## 动作与部署完整性

- 本面板冻结于 epoch {stats['run_contract']['checkpoint_epoch']}、global step {stats['run_contract']['checkpoint_global_step']}、commit `{stats['run_contract']['checkpoint_source_commit']}`、checkpoint SHA256 `{stats['run_contract']['checkpoint_sha256']}`。
- 部署契约是 `{stats['run_contract']['model_contract']}`；这是 original direct-command、uniform five-step 版本，不是不均匀步长版本。
- {stats['action_integrity']['planning_decisions']} 次决策中，raw 首执行行只有 {stats['action_integrity']['raw_executed_oob_rows']} 行越界；所有实际执行动作都严格等于 `clip(raw_chunks[:, 0])`，夹爪均为二值命令。因此轻微 raw 越界不是当前主要失败解释。
- 成功/失败完全取自 `result.json` 与 `oracle_success`；没有用 rollout 长度猜测成功。

## 证据边界

- 34 个案例是预选、独立 reset 的小面板，不是官方 1000-chain 成绩。
- 左右任务来自不同 reset，未固定采样噪声，左右差异是强行为线索，但不是严格语言因果实验。
- 人工视频标签单独保存在 `human_video_annotation`；`attention_drift`、`target_not_seen_or_bound`、`wrong_side` 等只能作为人工观察现象，不能当作内部机制事实。
- 成功参考中未实际观看视频的案例仍标为 `reference_not_video_reviewed`，没有伪装成视频复核。

## 可复现产物

- `cases.json` / `cases.jsonl`：34 个案例的完整结构化指标。
- `cases.csv`：便于筛选的扁平表。
- `statistics.json`：任务、方向、接触、动作完整性和证据边界汇总。
- `inventory.json`：文件、哈希、视频解码和帧数核验。
- `provenance.json`：冻结 checkpoint、manifest 和重建命令。
"""


def sanitize_csv_text(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value).replace("\\n", " ").replace("\r", " ").replace("\n", " ")


def flatten_case_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "case_id": row["case_id"],
        "case": row["case"],
        "task": row["task"],
        "task_family": row["task_family"],
        "trial": row["trial"],
        "instruction": row["instruction"],
        "source_sequence_index": row["source_sequence_index"],
        "official_success": row["official_success"],
        "steps": row["steps"],
        "official_success_within_180_steps": row[
            "official_success_within_180_steps"
        ],
        "target_object": row["target_object"],
        "official_push_failure_modes": ";".join(
            row["official_push_failure_modes"]
        ),
        "tcp_net_dx_m": row["tcp"]["net_displacement_m"][0],
        "tcp_net_dy_m": row["tcp"]["net_displacement_m"][1],
        "tcp_net_dz_m": row["tcp"]["net_displacement_m"][2],
        "tcp_path_length_m": row["tcp"]["path_length_m"],
        "tcp_first30_dx_m": row["tcp"]["first30_displacement_m"][0],
        "tcp_first30_dy_m": row["tcp"]["first30_displacement_m"][1],
        "tcp_first30_dz_m": row["tcp"]["first30_displacement_m"][2],
        "gripper_open_steps": row["action"]["gripper_open_steps"],
        "gripper_close_steps": row["action"]["gripper_close_steps"],
        "gripper_switches": row["action"]["gripper_switches"],
        "raw_executed_oob_rows": row["action"]["raw_executed_oob_rows"],
        "human_status": row["human_video_annotation"]["status"],
        "human_source": row["human_video_annotation"]["source"],
        "human_label": row["human_video_annotation"]["label"],
        "human_video_reviewed": row["human_video_annotation"]["video_reviewed"],
        "human_note": row["human_video_annotation"]["note"],
        "computed_true_flags": ";".join(
            key for key, value in row["computed_flags"].items() if bool(value)
        ),
        "video_file": row["artifact_paths"]["video"],
        "result_file": row["artifact_paths"]["result"],
        "trajectory_file": row["artifact_paths"]["trajectory"],
    }
    for block in BLOCK_ORDER:
        prefix = BLOCK_COLORS[block]
        metrics = row["objects"][block]
        final = metrics["final_displacement_m"]
        flat.update(
            {
                f"{prefix}_final_dx_m": final[0],
                f"{prefix}_final_dy_m": final[1],
                f"{prefix}_final_dz_m": final[2],
                f"{prefix}_max_planar_displacement_m": metrics[
                    "max_planar_displacement_m"
                ],
                f"{prefix}_max_spatial_displacement_m": metrics[
                    "max_spatial_displacement_m"
                ],
                f"{prefix}_max_upward_z_excursion_m": metrics[
                    "max_upward_z_excursion_m"
                ],
                f"{prefix}_max_downward_z_excursion_m": metrics[
                    "max_downward_z_excursion_m"
                ],
                f"{prefix}_first_contact_step": metrics["contact"]["first_step"],
                f"{prefix}_last_contact_step": metrics["contact"]["last_step"],
                f"{prefix}_contact_steps": metrics["contact"]["count_steps"],
                f"{prefix}_longest_contact_run": metrics["contact"][
                    "longest_consecutive_steps"
                ],
            }
        )
    push = row["push"]
    for key in (
        "direction",
        "world_x_sign",
        "target_final_signed_progress_m",
        "target_max_signed_progress_m",
        "target_max_opposite_progress_m",
        "target_reversal_from_best_m",
        "direction_correct_1cm",
        "first_target_contact_step",
        "first_target_motion_step",
        "first_positive_progress_step",
        "target_contact_steps",
        "target_is_most_moved_planar",
    ):
        flat[f"push_{key}"] = push.get(key) if push is not None else None
    return {key: sanitize_csv_text(value) for key, value in flat.items()}


def emit_outputs(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    summary_markdown: str,
    provenance: Mapping[str, Any],
    inventory_base: Mapping[str, Any],
) -> None:
    cases_path = output_dir / "cases.json"
    jsonl_path = output_dir / "cases.jsonl"
    csv_path = output_dir / "cases.csv"
    statistics_path = output_dir / "statistics.json"
    summary_path = output_dir / "summary.md"
    provenance_path = output_dir / "provenance.json"
    atomic_write_text(cases_path, json_text(list(rows)))
    atomic_write_text(
        jsonl_path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
    )
    flattened = [flatten_case_for_csv(row) for row in rows]
    require(bool(flattened), "no CSV rows")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=list(flattened[0].keys()),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(flattened)
    csv_text = csv_buffer.getvalue()
    require("\\n" not in csv_text, "CSV contains a literal backslash-n sequence")
    atomic_write_text(csv_path, csv_text)
    atomic_write_text(statistics_path, json_text(statistics))
    atomic_write_text(summary_path, summary_markdown)
    atomic_write_text(provenance_path, json_text(provenance))

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    require(len(jsonl_lines) == len(rows), "JSONL physical line count mismatch")
    require(len(csv_lines) == len(rows) + 1, "CSV physical line count mismatch")
    require(
        all(isinstance(json.loads(line), dict) for line in jsonl_lines),
        "invalid JSONL row",
    )

    generated = []
    for path in (
        cases_path,
        jsonl_path,
        csv_path,
        statistics_path,
        summary_path,
        provenance_path,
    ):
        generated.append(
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    inventory = dict(inventory_base)
    inventory["generated_outputs"] = generated
    supporting_documents = []
    for path in (output_dir / "README.md", output_dir / "decision_memo.md"):
        if path.is_file():
            supporting_documents.append(
                {
                    "path": path.name,
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    inventory["supporting_documents"] = supporting_documents
    inventory["output_checks"] = {
        "cases_json_count": len(load_json(cases_path)),
        "jsonl_physical_lines": len(jsonl_lines),
        "csv_physical_lines": len(csv_lines),
        "csv_has_literal_backslash_n": "\\n" in csv_path.read_text(encoding="utf-8"),
    }
    atomic_write_text(output_dir / "inventory.json", json_text(inventory))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-annotations", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Optional untouched final-sync root. When supplied, every canonical "
            "case file and flat MP4 is hash-compared with its source copy."
        ),
    )
    parser.add_argument("--expected-cases", type=int, default=34)
    parser.add_argument(
        "--skip-ffprobe",
        action="store_true",
        help="Skip full video frame counting (not recommended for final acceptance).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve() if args.source_root is not None else None
    require(data_root.is_dir(), f"data root missing: {data_root}")
    require(video_root.is_dir(), f"video root missing: {video_root}")
    if source_root is not None:
        require(source_root.is_dir(), f"source-sync root missing: {source_root}")
    artifact_root = data_root.parent
    require(
        output_dir == artifact_root / "analysis",
        "output-dir must be the sibling analysis directory for reproducible paths",
    )
    ffprobe = None if args.skip_ffprobe else shutil.which("ffprobe")
    if not args.skip_ffprobe:
        require(ffprobe is not None, "ffprobe is required unless --skip-ffprobe is set")

    required_root_json = (
        "summary.json",
        "manifest.json",
        "frozen_checkpoint.json",
        "controller_status.json",
        "bridge_health_0.json",
        "launch.json",
        "startup_verification.json",
        "official_source_provenance.json",
    )
    for filename in required_root_json:
        require((data_root / filename).is_file(), f"missing root metadata: {filename}")
    manifest = load_json(data_root / "manifest.json")
    summary = load_json(data_root / "summary.json")
    checkpoint = load_json(data_root / "frozen_checkpoint.json")
    controller = load_json(data_root / "controller_status.json")
    bridge = load_json(data_root / "bridge_health_0.json")
    launch = load_json(data_root / "launch.json")
    progress = load_json(data_root / "shard_0" / "progress.json")
    completed = load_json(data_root / "shard_0" / "completed.json")

    digest = manifest_digest(manifest)
    require(digest == manifest["manifest_sha256"], "manifest digest mismatch")
    require(summary["manifest_sha256"] == digest, "summary manifest mismatch")
    require(bool(summary["complete"]), "summary is not complete")
    require(int(summary["trials"]) == args.expected_cases, "summary case count mismatch")
    require(len(manifest["cases"]) == args.expected_cases, "manifest case count mismatch")
    require(bool(completed["complete"]), "completed.json is not complete")
    require(str(progress["status"]) == "complete", "progress.json is not complete")
    require(str(controller["status"]) == "complete", "controller is not complete")
    require(bool(controller["owned_children_reaped"]), "controller children were not reaped")
    require(str(bridge["status"]) == "ok", "bridge status is not ok")
    require(
        bridge["deployment"]["action"]["arm_flow_mode"] == "relative_command_direct",
        "unexpected action flow mode",
    )
    require(
        "uniform five-step" in str(manifest["model_contract"]),
        "this panel is not the expected uniform five-step deployment",
    )

    cases = sorted(manifest["cases"], key=lambda item: int(item["case_id"]))
    ids = [int(case["case_id"]) for case in cases]
    require(ids == list(range(args.expected_cases)), "case IDs are missing or duplicated")
    summary_reports = {
        int(report["case"]["case_id"]): report for report in summary["results"]
    }
    require(len(summary_reports) == args.expected_cases, "duplicate/missing summary results")
    expected_case_names = {case_name(case) for case in cases}
    shard_root = data_root / "shard_0"
    actual_case_names = {path.name for path in shard_root.iterdir() if path.is_dir()}
    require(actual_case_names == expected_case_names, "case directory set does not match manifest")
    partial_or_error = [
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and ("partial" in path.name.lower() or path.name.lower() == "error.json")
    ]
    require(not partial_or_error, f"partial/error artifacts present: {partial_or_error}")
    actual_videos = {path.stem for path in video_root.glob("*.mp4")}
    require(actual_videos == expected_case_names, "flat video set does not match manifest")

    source_root_file_checks: list[dict[str, Any]] = []
    if source_root is not None:
        source_root_files = {
            path.name: path
            for path in source_root.iterdir()
            if path.is_file()
        }
        canonical_root_files = {
            path.name: path
            for path in data_root.iterdir()
            if path.is_file()
        }
        require(
            set(canonical_root_files) == set(source_root_files),
            "canonical/source-sync root metadata file sets differ",
        )
        for filename in sorted(canonical_root_files):
            canonical = canonical_root_files[filename]
            source = source_root_files[filename]
            canonical_sha = sha256_file(canonical)
            source_sha = sha256_file(source)
            require(
                canonical_sha == source_sha,
                f"canonical/source-sync root hash mismatch: {filename}",
            )
            source_root_file_checks.append(
                {
                    "path": filename,
                    "bytes": int(source.stat().st_size),
                    "sha256": source_sha,
                }
            )
        source_control_files = {
            path.name: path
            for path in (source_root / "shard_0").iterdir()
            if path.is_file()
        }
        canonical_control_files = {
            path.name: path
            for path in (data_root / "shard_0").iterdir()
            if path.is_file()
        }
        require(
            set(canonical_control_files) == set(source_control_files),
            "canonical/source-sync shard control file sets differ",
        )
        for filename in sorted(canonical_control_files):
            canonical = canonical_control_files[filename]
            source = source_control_files[filename]
            require(
                sha256_file(canonical) == sha256_file(source),
                f"canonical/source-sync control hash mismatch: {filename}",
            )

    checkpoint_identity = {
        "sha256": checkpoint["sha256"],
        "epoch": int(checkpoint["metadata"]["epoch"]),
        "global_step": int(checkpoint["metadata"]["global_step"]),
        "git_commit": checkpoint["source_commit"],
    }
    require(summary["checkpoint"] == checkpoint, "summary/frozen checkpoint mismatch")
    annotations_path = (
        args.legacy_annotations.resolve() if args.legacy_annotations is not None else None
    )
    human_by_id, label_dictionary = load_human_annotations(annotations_path)

    rows: list[dict[str, Any]] = []
    case_inventory: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        row, inventory = verify_case(
            case=case,
            case_dir=shard_root / case_name(case),
            video_root=video_root,
            manifest_sha256=digest,
            checkpoint_identity=checkpoint_identity,
            summary_report=summary_reports[case_id],
            human_annotation=human_by_id.get(case_id, missing_human_annotation()),
            ffprobe=ffprobe,
            artifact_root=artifact_root,
            source_root=source_root,
        )
        rows.append(row)
        case_inventory.append(inventory)

    require(
        sum(bool(row["official_success"]) for row in rows) == int(summary["successes"]),
        "rebuilt success total does not match summary",
    )
    require(
        dict(Counter(str(row["task"]) for row in rows)) == manifest["task_counts"],
        "task coverage does not match manifest",
    )
    stats = build_statistics(
        rows,
        manifest=manifest,
        summary=summary,
        checkpoint=checkpoint,
        label_dictionary=label_dictionary,
    )
    require(stats["panel"]["successes"] == 6, "unexpected final success total")
    require(stats["panel"]["trials"] == 34, "unexpected final panel size")
    require(stats["push"]["target_contact_cases"] == 11, "unexpected push contact total")
    require(
        stats["push"]["by_direction"]["left"]["target_contact_cases"] == 9,
        "unexpected left-push contact total",
    )
    require(
        stats["push"]["by_direction"]["right"]["target_contact_cases"] == 2,
        "unexpected right-push contact total",
    )

    script_path = Path(__file__).resolve()
    source_argument = f' --source-root "{source_root}"' if source_root is not None else ""
    command = (
        ".venv\\Scripts\\python.exe scripts\\rebuild_calvin_e4_analysis.py "
        "--data-root artifacts\\calvin_e4_20260919_download\\data "
        "--video-root artifacts\\calvin_e4_20260919_download\\videos "
        "--output-dir artifacts\\calvin_e4_20260919_download\\analysis "
        "--legacy-annotations "
        "artifacts\\calvin_e4_20260919_download\\analysis\\legacy_26case\\"
        "video_trajectory_annotations.json"
        + source_argument
    )
    generated_utc = datetime.now(timezone.utc).isoformat()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "data_root": relative_posix(data_root, artifact_root),
        "video_root": relative_posix(video_root, artifact_root),
        "legacy_annotations": (
            relative_posix(annotations_path, artifact_root)
            if annotations_path is not None
            else None
        ),
        "source_sync_root": str(source_root) if source_root is not None else None,
        "source_sync_hash_comparison": source_root is not None,
        "rebuild_script": str(script_path),
        "rebuild_script_sha256": sha256_file(script_path),
        "reproduction_command_from_workspace_root": command,
        "manifest_sha256": digest,
        "checkpoint": checkpoint,
        "model_contract": manifest["model_contract"],
        "controller": {
            "status": controller["status"],
            "owned_children_reaped": controller["owned_children_reaped"],
            "planned_trials": controller["planned_trials"],
        },
        "bridge": {
            "status": bridge["status"],
            "initialized": bridge["initialized"],
            "initialized_interpretation": (
                "final health snapshot was written after controller cleanup; false is not used as a startup-failure signal"
            ),
            "mode": bridge["mode"],
            "arm_flow_mode": bridge["deployment"]["action"]["arm_flow_mode"],
            "architecture_manifest_digest": bridge["deployment"]["architecture"][
                "manifest_digest"
            ],
        },
        "launch": {
            "model_commit": launch["model_commit"],
            "torch": launch["torch"],
            "transformers": launch["transformers"],
            "gpu": launch["gpu"],
            "shards": launch["shards"],
        },
        "success_semantics": (
            "official result.json success cross-checked against oracle_success; never inferred from trajectory length"
        ),
    }
    root_files = [
        file_record(path, artifact_root)
        for path in sorted(data_root.iterdir())
        if path.is_file()
    ]
    inventory_base = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "complete": True,
        "expected_cases": args.expected_cases,
        "verified_cases": len(rows),
        "data_file_count": sum(1 for path in data_root.rglob("*") if path.is_file()),
        "flat_video_count": len(actual_videos),
        "flat_video_bytes": sum(path.stat().st_size for path in video_root.glob("*.mp4")),
        "ffprobe_enabled": ffprobe is not None,
        "no_partial_or_error_files": True,
        "manifest_case_directories_exact": True,
        "flat_video_set_exact": True,
        "source_sync_hash_comparison": source_root is not None,
        "root_files": root_files,
        "source_root_files": source_root_file_checks,
        "cases": case_inventory,
    }
    summary_markdown = render_summary(rows, stats)
    emit_outputs(
        output_dir=output_dir,
        rows=rows,
        statistics=stats,
        summary_markdown=summary_markdown,
        provenance=provenance,
        inventory_base=inventory_base,
    )
    print(
        json.dumps(
            {
                "complete": True,
                "cases": len(rows),
                "successes": stats["panel"]["successes"],
                "push_target_contacts": stats["push"]["target_contact_cases"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
