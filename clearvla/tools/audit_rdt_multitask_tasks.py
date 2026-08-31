"""Task-complete offline audit for selecting a bounded RDT multitask scope.

Every ``rdt_data`` task is reduced from all of its HDF5 episodes.  The tool
reads native action/qpos arrays and camera headers, and can decode a fixed
number of deterministic high/right-wrist frames per episode.  Directory names
remain identity only; the scalar HDF5 instruction is the sole semantic text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from clearvla.data.hdf5_episode import (
    decode_hdf5_instruction,
    episode_identity,
    find_hdf5_files,
)
from clearvla.data.split import RDT_SPLIT_NAMES, RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH
from clearvla.vision.image_io import decode_image_value

AUDIT_SCHEMA = "clearvla-rdt-multitask-task-audit-v1"
CAMERA_KEYS = {
    "high": "observations/images/cam_high",
    "right_wrist": "observations/images/cam_right_wrist",
}
ARM_INDICES = {
    "left": np.arange(0, 6, dtype=np.int64),
    "right": np.arange(7, 13, dtype=np.int64),
}
GRIPPER_INDICES = {"left": 6, "right": 13}
ARM_STEP_THRESHOLDS = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
GRIPPER_DELTA_THRESHOLDS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
QUANTILES = (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: Iterable[np.ndarray]) -> dict[str, float]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    arrays = [value for value in arrays if value.size]
    if not arrays:
        return {f"p{int(round(q * 100)):02d}": 0.0 for q in QUANTILES}
    merged = np.concatenate(arrays)
    return {
        f"p{int(round(q * 100)):02d}": float(value)
        for q, value in zip(QUANTILES, np.quantile(merged, QUANTILES), strict=True)
    }


def _fraction_at_thresholds(
    values: Iterable[np.ndarray], thresholds: Iterable[float]
) -> dict[str, float]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    total = sum(int(value.size) for value in arrays)
    return {
        f"ge_{threshold:g}": float(
            sum(int(np.count_nonzero(value >= threshold)) for value in arrays)
            / max(total, 1)
        )
        for threshold in thresholds
    }


def _decode_indices(length: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    return tuple(
        sorted(
            {
                int(round(value))
                for value in np.linspace(0, max(length - 1, 0), min(count, length))
            }
        )
    )


def _load_split(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        raise ValueError("split manifest must contain a split mapping")
    membership: dict[str, str] = {}
    for split in RDT_SPLIT_NAMES:
        rows = payload["splits"].get(split)
        if not isinstance(rows, list):
            raise ValueError(f"split manifest is missing {split!r}")
        for identity in rows:
            text = str(identity)
            if text in membership:
                raise ValueError(f"episode occurs in more than one split: {text}")
            membership[text] = split
    return payload, membership


def _task_row(
    root: Path,
    task_id: str,
    paths: list[Path],
    *,
    split_membership: dict[str, str],
    excluded_too_short: dict[str, int],
    sampled_decode_frames: int,
) -> dict[str, Any]:
    instructions: set[str] = set()
    episode_rows: list[dict[str, Any]] = []
    action_widths: Counter[int] = Counter()
    qpos_widths: Counter[int] = Counter()
    finite = True
    camera_header_complete = True
    decode_errors: list[dict[str, object]] = []
    sampled_decoded = 0
    arm_action_steps: dict[str, list[np.ndarray]] = defaultdict(list)
    arm_qpos_steps: dict[str, list[np.ndarray]] = defaultdict(list)
    arm_action_values: dict[str, list[np.ndarray]] = defaultdict(list)
    arm_qpos_values: dict[str, list[np.ndarray]] = defaultdict(list)
    gripper_values: dict[str, list[np.ndarray]] = defaultdict(list)
    gripper_abs_deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    gripper_segment_lengths: dict[str, list[np.ndarray]] = defaultdict(list)
    split_counts = {
        name: 0 for name in ("train", "val", "test", "excluded_too_short")
    }
    split_windows = {
        name: 0 for name in ("train", "val", "test", "excluded_too_short")
    }

    for path in paths:
        identity, partition, observed_task = episode_identity(root, path)
        if partition != "rdt_data" or observed_task != task_id:
            raise AssertionError("task grouping changed during audit")
        split = split_membership.get(identity)
        if split not in {"train", "val", "test"}:
            if identity not in excluded_too_short:
                raise ValueError(f"task episode has no split/exclusion identity: {identity}")
            split = "excluded_too_short"
        with h5py.File(path, "r") as handle:
            action_ds = handle.get("action")
            qpos_ds = handle.get("observations/qpos")
            instruction_ds = handle.get("instruction")
            if not isinstance(action_ds, h5py.Dataset) or action_ds.ndim != 2:
                raise ValueError(f"{path}: action must be [T,D]")
            if not isinstance(qpos_ds, h5py.Dataset) or qpos_ds.ndim != 2:
                raise ValueError(f"{path}: qpos must be [T,D]")
            if not isinstance(instruction_ds, h5py.Dataset):
                raise ValueError(f"{path}: scalar instruction is missing")
            action = np.asarray(action_ds, dtype=np.float32)
            qpos = np.asarray(qpos_ds, dtype=np.float32)
            if action.shape != qpos.shape or action.shape[0] <= 0:
                raise ValueError(f"{path}: action/qpos shapes differ or are empty")
            length, action_width = (int(value) for value in action.shape)
            qpos_width = int(qpos.shape[1])
            action_widths[action_width] += 1
            qpos_widths[qpos_width] += 1
            episode_finite = bool(np.isfinite(action).all() and np.isfinite(qpos).all())
            finite = finite and episode_finite
            instruction = decode_hdf5_instruction(instruction_ds[()])
            instructions.add(instruction)
            camera_rows: dict[str, object] = {}
            for camera, key in CAMERA_KEYS.items():
                dataset = handle.get(key)
                header_ok = bool(
                    isinstance(dataset, h5py.Dataset)
                    and dataset.ndim == 1
                    and int(dataset.shape[0]) == length
                    and dataset.dtype.kind in {"S", "O"}
                )
                camera_header_complete = camera_header_complete and header_ok
                camera_rows[camera] = {
                    "key": key,
                    "header_complete": header_ok,
                    "rows": int(dataset.shape[0]) if isinstance(dataset, h5py.Dataset) else 0,
                    "dtype": str(dataset.dtype) if isinstance(dataset, h5py.Dataset) else None,
                }
                if not header_ok:
                    continue
                for frame in _decode_indices(length, sampled_decode_frames):
                    try:
                        decoded = decode_image_value(dataset[frame])
                        if (
                            decoded.ndim != 3
                            or decoded.shape[-1] != 3
                            or decoded.dtype != np.uint8
                            or not decoded.size
                        ):
                            raise ValueError(
                                f"decoded RGB has invalid shape/dtype {decoded.shape}/{decoded.dtype}"
                            )
                        sampled_decoded += 1
                    except Exception as exc:
                        decode_errors.append(
                            {
                                "episode_id": identity,
                                "camera": camera,
                                "frame": frame,
                                "error": repr(exc),
                            }
                        )

        windows = max(length - RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH + 1, 0)
        if split == "excluded_too_short" and (
            windows != 0 or excluded_too_short[identity] != length
        ):
            raise ValueError(f"short-episode exclusion is stale: {identity}")
        split_counts[split] += 1
        split_windows[split] += windows
        episode_rows.append(
            {
                "episode_id": identity,
                "split": split,
                "length": length,
                "valid_windows": windows,
                "finite_native_action_qpos": episode_finite,
                "camera": camera_rows,
            }
        )
        if action_width != 14 or qpos_width != 14:
            continue
        boundary = np.concatenate((qpos[:1], action[:-1]), axis=0)
        for side, indices in ARM_INDICES.items():
            arm_action_values[side].append(action[:, indices])
            arm_qpos_values[side].append(qpos[:, indices])
            arm_action_steps[side].append(
                np.linalg.norm(np.diff(action[:, indices], axis=0), axis=1)
            )
            arm_qpos_steps[side].append(
                np.linalg.norm(np.diff(qpos[:, indices], axis=0), axis=1)
            )
        for side, index in GRIPPER_INDICES.items():
            values = action[:, index]
            delta = np.abs(action[:, index] - boundary[:, index])
            gripper_values[side].append(values)
            gripper_abs_deltas[side].append(delta)
            moving = delta > 0.0
            if moving.size:
                change = np.flatnonzero(np.diff(moving.astype(np.int8)) != 0) + 1
                bounds = np.concatenate(([0], change, [len(moving)]))
                gripper_segment_lengths[side].append(np.diff(bounds).astype(np.float64))

    if len(instructions) != 1:
        raise ValueError(f"task {task_id!r} has multiple HDF5 instructions")
    instruction = next(iter(instructions))
    activity: dict[str, Any] = {}
    for side in ("left", "right"):
        action_steps = arm_action_steps[side]
        qpos_steps = arm_qpos_steps[side]
        action_energy = sum(float(np.square(row, dtype=np.float64).sum()) for row in action_steps)
        qpos_energy = sum(float(np.square(row, dtype=np.float64).sum()) for row in qpos_steps)
        action_rows = sum(int(row.size) for row in action_steps)
        qpos_rows = sum(int(row.size) for row in qpos_steps)
        action_values = np.concatenate(arm_action_values[side], axis=0)
        qpos_values = np.concatenate(arm_qpos_values[side], axis=0)
        activity[side] = {
            "joint_action_step_l2": {
                "rms": float(np.sqrt(action_energy / max(action_rows, 1))),
                "quantiles": _quantiles(action_steps),
                "fraction": _fraction_at_thresholds(action_steps, ARM_STEP_THRESHOLDS),
            },
            "joint_qpos_step_l2": {
                "rms": float(np.sqrt(qpos_energy / max(qpos_rows, 1))),
                "quantiles": _quantiles(qpos_steps),
                "fraction": _fraction_at_thresholds(qpos_steps, ARM_STEP_THRESHOLDS),
            },
            "joint_action_range_per_dim": np.ptp(action_values, axis=0).tolist(),
            "joint_qpos_range_per_dim": np.ptp(qpos_values, axis=0).tolist(),
            "gripper_action_value": _quantiles(gripper_values[side]),
            "gripper_boundary_abs_delta": {
                "quantiles": _quantiles(gripper_abs_deltas[side]),
                "fraction": _fraction_at_thresholds(
                    gripper_abs_deltas[side], GRIPPER_DELTA_THRESHOLDS
                ),
                "total_variation": float(
                    sum(float(row.sum(dtype=np.float64)) for row in gripper_abs_deltas[side])
                ),
            },
            "gripper_constant_activity_segment_length": _quantiles(
                gripper_segment_lengths[side]
            ),
        }
    left_action_energy = activity["left"]["joint_action_step_l2"]["rms"] ** 2
    right_action_energy = activity["right"]["joint_action_step_l2"]["rms"] ** 2
    left_qpos_energy = activity["left"]["joint_qpos_step_l2"]["rms"] ** 2
    right_qpos_energy = activity["right"]["joint_qpos_step_l2"]["rms"] ** 2
    activity["left_to_right_joint_step_energy_ratio"] = {
        "action": float(left_action_energy / max(right_action_energy, np.finfo(float).tiny)),
        "qpos": float(left_qpos_energy / max(right_qpos_energy, np.finfo(float).tiny)),
    }
    return {
        "task_id": task_id,
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "source_episode_count": len(episode_rows),
        "eligible_episode_count": len(episode_rows) - split_counts["excluded_too_short"],
        "split_episode_counts": split_counts,
        "valid_window_count": sum(row["valid_windows"] for row in episode_rows),
        "split_valid_window_counts": split_windows,
        "length": _quantiles(
            [np.asarray([row["length"] for row in episode_rows], dtype=np.float64)]
        ),
        "native_profile": {
            "action_widths": {str(key): value for key, value in sorted(action_widths.items())},
            "qpos_widths": {str(key): value for key, value in sorted(qpos_widths.items())},
            "finite_all": finite,
            "canonical_layout": "left_joints_0_5,left_gripper_6,right_joints_7_12,right_gripper_13",
        },
        "camera_audit": {
            "required": list(CAMERA_KEYS),
            "header_complete_all": camera_header_complete,
            "sampled_decode_frames_per_episode_camera": sampled_decode_frames,
            "sampled_decoded": sampled_decoded,
            "decode_errors": decode_errors,
        },
        "activity": activity,
        "episodes": episode_rows,
    }


def audit_rdt_multitask_tasks(
    root: Path,
    *,
    split_manifest: Path,
    sampled_decode_frames: int = 1,
) -> dict[str, Any]:
    source = root.expanduser().resolve()
    split_path = split_manifest.expanduser().resolve()
    split, membership = _load_split(split_path)
    raw_excluded = split.get("excluded_too_short")
    if not isinstance(raw_excluded, list):
        raise ValueError("split manifest is missing short-episode exclusions")
    excluded_too_short = {
        str(row["episode_id"]): int(row["length"])
        for row in raw_excluded
        if isinstance(row, dict)
    }
    if len(excluded_too_short) != len(raw_excluded):
        raise ValueError("split manifest has malformed or duplicate short exclusions")
    grouped: dict[str, list[Path]] = defaultdict(list)
    source_paths = find_hdf5_files(source, "**/*.hdf5")
    for path in source_paths:
        identity, partition, task = episode_identity(source, path)
        if identity not in membership and not any(
            str(row.get("episode_id")) == identity
            for row in split.get("excluded_too_short", [])
        ):
            raise ValueError(f"episode is absent from split manifest: {identity}")
        if partition == "rdt_data":
            grouped[task].append(path)
    if len(grouped) != 302:
        raise ValueError(f"expected 302 eligible rdt_data tasks, found {len(grouped)}")
    tasks = [
        _task_row(
            source,
            task,
            sorted(paths, key=lambda path: episode_identity(source, path)[0]),
            split_membership=membership,
            excluded_too_short=excluded_too_short,
            sampled_decode_frames=int(sampled_decode_frames),
        )
        for task, paths in sorted(grouped.items())
    ]
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "root": str(source),
        "source_episode_count": len(source_paths),
        "rdt_data_task_count": len(tasks),
        "eligible_rdt_data_episode_count": sum(
            row["eligible_episode_count"] for row in tasks
        ),
        "split_manifest": {
            "path": str(split_path),
            "file_sha256": _file_sha256(split_path),
            "manifest_sha256": str(split.get("manifest_sha256", "")),
            "source_episode_inventory_sha256": str(
                split.get("source_episode_inventory_sha256", "")
            ),
            "episode_inventory_sha256": str(split.get("episode_inventory_sha256", "")),
        },
        "sampled_decode_frames_per_episode_camera": int(sampled_decode_frames),
        "tasks": tasks,
    }
    report["audit_sha256"] = _canonical_digest(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit every RDT task for multitask selection")
    parser.add_argument("root", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--sampled-decode-frames", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.sampled_decode_frames < 0:
        raise ValueError("sampled decode frame count must be non-negative")
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite task audit: {destination}")
    report = audit_rdt_multitask_tasks(
        args.root,
        split_manifest=args.split_manifest,
        sampled_decode_frames=args.sampled_decode_frames,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(destination), "audit_sha256": report["audit_sha256"]}))


if __name__ == "__main__":
    main()


__all__ = ["AUDIT_SCHEMA", "audit_rdt_multitask_tasks"]
