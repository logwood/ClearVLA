"""Read-only inventory for the hierarchical RDT fine-tuning dataset.

The tool deliberately inspects only scalar metadata, action/qpos arrays, and
image dataset headers.  It never decodes or copies RGB/depth payloads.  Its
output is a data-contract audit, not a model-quality or task-success metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

ACTION_KEY = "action"
QPOS_KEY = "observations/qpos"
BASE_ACTION_KEY = "base_action"
INSTRUCTION_KEY = "instruction"
CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
RGB_KEYS = tuple(f"observations/images/{name}" for name in CAMERA_NAMES)
DEPTH_KEYS = tuple(f"observations/images_depth/{name}" for name in CAMERA_NAMES)
LEFT_JOINTS = np.arange(0, 6, dtype=np.int64)
RIGHT_JOINTS = np.arange(7, 13, dtype=np.int64)
GRIPPER_INDICES = {"left": 6, "right": 13}
QUANTILES = (0.0, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0)
AUDIT_EVENT_THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)


def discover_hdf5(root: Path) -> list[Path]:
    """Return a deterministic recursive inventory without following aliases."""

    source = Path(root).expanduser()
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes found recursively under {source}")
    return paths


def episode_id(root: Path, path: Path) -> str:
    """Stable root-relative ID; nested task names remain part of identity."""

    relative = path.resolve().relative_to(root.resolve()).with_suffix("")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid root-relative episode path: {relative}")
    return relative.as_posix()


def _decode_instruction(value: object) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        result = bytes(value).decode("utf-8")
    elif isinstance(value, str):
        result = value
    else:
        raise TypeError(f"instruction must be a scalar UTF-8 string, got {type(value)!r}")
    if not result.strip():
        raise ValueError("instruction must not be empty")
    return result


def _counter_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda row: str(row[0]))}


def _quantile_dict(values: Iterable[np.ndarray]) -> dict[str, float]:
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    rows = [value for value in rows if value.size]
    if not rows:
        return {f"p{int(round(q * 100)):02d}": 0.0 for q in QUANTILES}
    merged = np.concatenate(rows)
    quantile = np.quantile(merged, QUANTILES)
    return {
        f"p{int(round(q * 100)):02d}": float(value)
        for q, value in zip(QUANTILES, quantile, strict=True)
    }


def _rmse(sum_square: np.ndarray, rows: int) -> list[float]:
    if rows <= 0:
        return [0.0 for _ in sum_square]
    return np.sqrt(sum_square / float(rows)).astype(np.float64).tolist()


def _task_declared_instructions(task_dir: Path) -> tuple[set[str], list[str]]:
    declared: set[str] = set()
    errors: list[str] = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("instruction"), str):
                declared.add(str(payload["instruction"]))
        except Exception as exc:  # reported as audit evidence
            errors.append(f"{path}: {exc!r}")
    return declared, errors


def audit_rdt_ft_data(
    root: Path,
    *,
    max_episodes: int = 0,
    schema_only: bool = False,
) -> dict[str, Any]:
    """Audit hierarchy, language, camera availability, and native 14-D charts."""

    source = Path(root).expanduser().resolve()
    paths = discover_hdf5(source)
    if int(max_episodes) < 0:
        raise ValueError("max_episodes must be non-negative")
    selected = paths if int(max_episodes) == 0 else paths[: int(max_episodes)]

    ids: list[str] = []
    stem_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    task_episode_counts: Counter[str] = Counter()
    length_rows: list[int] = []
    action_qpos_shapes: Counter[tuple[int, int]] = Counter()
    rgb_storage: Counter[tuple[str, tuple[int, ...]]] = Counter()
    depth_presence: Counter[tuple[str, ...]] = Counter()
    task_instructions: dict[str, set[str]] = defaultdict(set)
    instruction_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    base_action_max_abs = 0.0
    base_action_nonzero_episodes = 0

    gripper_value: dict[str, list[np.ndarray]] = {name: [] for name in GRIPPER_INDICES}
    gripper_abs_delta: dict[str, list[np.ndarray]] = {
        name: [] for name in GRIPPER_INDICES
    }
    arm_step_rms: dict[str, list[float]] = {"left": [], "right": []}
    current_sum_square = np.zeros(14, dtype=np.float64)
    current_rows = 0
    next_sum_square = np.zeros(14, dtype=np.float64)
    next_rows = 0

    for path in selected:
        relative = path.relative_to(source)
        identity = episode_id(source, path)
        ids.append(identity)
        stem_counts[path.stem] += 1
        partition = relative.parts[0] if len(relative.parts) >= 3 else "<flat>"
        task = relative.parent.as_posix()
        partition_counts[partition] += 1
        task_episode_counts[task] += 1
        try:
            with h5py.File(path, "r") as handle:
                missing = [
                    key
                    for key in (ACTION_KEY, QPOS_KEY, INSTRUCTION_KEY, *RGB_KEYS)
                    if key not in handle
                ]
                if missing:
                    raise KeyError(f"missing required datasets: {missing}")
                action_dataset = handle[ACTION_KEY]
                qpos_dataset = handle[QPOS_KEY]
                if action_dataset.ndim != 2 or qpos_dataset.ndim != 2:
                    raise ValueError("action and qpos must both be rank-two")
                if tuple(action_dataset.shape) != tuple(qpos_dataset.shape):
                    raise ValueError(
                        f"action/qpos shape mismatch: {action_dataset.shape} != {qpos_dataset.shape}"
                    )
                length, action_dim = (int(value) for value in action_dataset.shape)
                qpos_dim = int(qpos_dataset.shape[1])
                if length < 1:
                    raise ValueError("episode is empty")
                length_rows.append(length)
                action_qpos_shapes[(action_dim, qpos_dim)] += 1
                instruction = _decode_instruction(handle[INSTRUCTION_KEY][()])
                task_instructions[task].add(instruction)
                instruction_counts[instruction] += 1

                for key in RGB_KEYS:
                    dataset = handle[key]
                    if dataset.ndim != 1 or int(dataset.shape[0]) != length:
                        raise ValueError(f"{key} must be a byte row per frame")
                    if dataset.dtype.kind not in {"S", "O"}:
                        raise TypeError(f"{key} must store encoded bytes, got {dataset.dtype}")
                    rgb_storage[(dataset.dtype.kind, int(dataset.ndim))] += 1
                present_depth = tuple(
                    name
                    for name, key in zip(CAMERA_NAMES, DEPTH_KEYS, strict=True)
                    if key in handle
                )
                depth_presence[present_depth] += 1
                for name, key in zip(CAMERA_NAMES, DEPTH_KEYS, strict=True):
                    if key not in handle:
                        continue
                    dataset = handle[key]
                    if dataset.ndim != 1 or int(dataset.shape[0]) != length:
                        raise ValueError(f"{key} must be a byte row per frame")
                    if dataset.dtype.kind not in {"S", "O"}:
                        raise TypeError(f"{key} must store encoded bytes, got {dataset.dtype}")

                if BASE_ACTION_KEY in handle:
                    base = np.asarray(handle[BASE_ACTION_KEY], dtype=np.float32)
                    if base.ndim != 2 or int(base.shape[0]) != length:
                        raise ValueError("base_action must align with episode frames")
                    maximum = float(np.abs(base).max(initial=0.0))
                    base_action_max_abs = max(base_action_max_abs, maximum)
                    base_action_nonzero_episodes += int(maximum > 0.0)

                if schema_only:
                    continue
                if action_dim != 14 or qpos_dim != 14:
                    raise ValueError(
                        f"numeric bimanual audit requires action/qpos width 14, got {action_dim}/{qpos_dim}"
                    )
                action = np.asarray(action_dataset, dtype=np.float32)
                qpos = np.asarray(qpos_dataset, dtype=np.float32)
                if not np.isfinite(action).all() or not np.isfinite(qpos).all():
                    raise ValueError("action/qpos contains NaN or infinity")
                current_residual = action.astype(np.float64) - qpos.astype(np.float64)
                current_sum_square += np.square(current_residual).sum(axis=0)
                current_rows += length
                if length > 1:
                    next_residual = action[:-1].astype(np.float64) - qpos[1:].astype(
                        np.float64
                    )
                    next_sum_square += np.square(next_residual).sum(axis=0)
                    next_rows += length - 1
                boundary = np.concatenate((qpos[:1], action[:-1]), axis=0)
                delta = action - boundary
                for name, index in GRIPPER_INDICES.items():
                    gripper_value[name].append(action[:, index].copy())
                    gripper_abs_delta[name].append(np.abs(delta[:, index]).copy())
                for name, indices in (("left", LEFT_JOINTS), ("right", RIGHT_JOINTS)):
                    if length > 1:
                        step = np.diff(action[:, indices], axis=0).astype(np.float64)
                        arm_step_rms[name].append(float(np.sqrt(np.square(step).mean())))
                    else:
                        arm_step_rms[name].append(0.0)
        except Exception as exc:  # keep a decision-complete inventory
            errors.append({"episode_id": identity, "error": repr(exc)})

    duplicate_ids = sorted(identity for identity, count in Counter(ids).items() if count > 1)
    multiple_task_instructions = {
        task: sorted(values)
        for task, values in sorted(task_instructions.items())
        if len(values) != 1
    }
    json_mismatch: dict[str, dict[str, list[str]]] = {}
    json_errors: list[str] = []
    tasks_without_declared_json = 0
    for task, values in sorted(task_instructions.items()):
        declared, task_errors = _task_declared_instructions(source / task)
        json_errors.extend(task_errors)
        if not declared:
            tasks_without_declared_json += 1
        elif not values.issubset(declared):
            json_mismatch[task] = {
                "hdf5_only": sorted(values - declared),
                "json_declared": sorted(declared),
            }

    lengths = np.asarray(length_rows, dtype=np.float64)
    report: dict[str, Any] = {
        "schema": "clearvla-rdt-ft-data-audit-v1",
        "root": str(source),
        "discovered_episodes": len(paths),
        "audited_episodes": len(selected),
        "limited": len(selected) != len(paths),
        "episode_identity": {
            "unique": len(set(ids)),
            "duplicates": duplicate_ids,
            "duplicate_stems": {
                stem: int(count) for stem, count in sorted(stem_counts.items()) if count > 1
            },
        },
        "source_partitions": _counter_dict(partition_counts),
        "tasks": {
            "count": len(task_episode_counts),
            "episode_count_min": min(task_episode_counts.values(), default=0),
            "episode_count_median": float(
                np.median(list(task_episode_counts.values())) if task_episode_counts else 0.0
            ),
            "episode_count_max": max(task_episode_counts.values(), default=0),
        },
        "length": {
            "min": int(lengths.min()) if lengths.size else 0,
            "p10": float(np.quantile(lengths, 0.10)) if lengths.size else 0.0,
            "median": float(np.median(lengths)) if lengths.size else 0.0,
            "p90": float(np.quantile(lengths, 0.90)) if lengths.size else 0.0,
            "max": int(lengths.max(initial=0.0)) if lengths.size else 0,
        },
        "action_qpos_widths": _counter_dict(action_qpos_shapes),
        "rgb": {
            "required_cameras": list(CAMERA_NAMES),
            "storage_headers": _counter_dict(rgb_storage),
        },
        "depth": {"availability_sets": _counter_dict(depth_presence)},
        "base_action": {
            "maximum_abs": base_action_max_abs,
            "nonzero_episodes": base_action_nonzero_episodes,
        },
        "language": {
            "unique_original_instructions": len(instruction_counts),
            "tasks_with_multiple_hdf5_instructions": multiple_task_instructions,
            "hdf5_vs_json_mismatch": json_mismatch,
            "tasks_without_declared_json": tasks_without_declared_json,
            "json_errors": json_errors,
            "instruction_inventory_sha256": hashlib.sha256(
                json.dumps(
                    sorted(instruction_counts.items()),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "errors": errors,
        "error_count": len(errors),
    }
    if not schema_only:
        gripper: dict[str, Any] = {}
        for name in GRIPPER_INDICES:
            absolute_delta_rows = gripper_abs_delta[name]
            total = sum(int(row.size) for row in absolute_delta_rows)
            threshold_fraction = {}
            for threshold in AUDIT_EVENT_THRESHOLDS:
                count = sum(int((row >= threshold).sum()) for row in absolute_delta_rows)
                threshold_fraction[f"ge_{threshold:g}"] = float(count / max(total, 1))
            gripper[name] = {
                "action_value": _quantile_dict(gripper_value[name]),
                "boundary_abs_delta": _quantile_dict(absolute_delta_rows),
                "boundary_abs_delta_fraction": threshold_fraction,
            }
        report["numeric"] = {
            "action_minus_qpos_current_rmse_per_dim": _rmse(
                current_sum_square, current_rows
            ),
            "action_t_minus_qpos_t_plus_1_rmse_per_dim": _rmse(
                next_sum_square, next_rows
            ),
            "arm_command_step_rms_per_episode": {
                name: _quantile_dict([np.asarray(values, dtype=np.float64)])
                for name, values in arm_step_rms.items()
            },
            "gripper": gripper,
        }
    return report


def _text_report(report: dict[str, Any]) -> str:
    lines = [
        f"schema={report['schema']}",
        (
            f"episodes={report['audited_episodes']}/{report['discovered_episodes']} "
            f"tasks={report['tasks']['count']} errors={report['error_count']}"
        ),
        f"partitions={json.dumps(report['source_partitions'], sort_keys=True)}",
        (
            "identity="
            f"unique:{report['episode_identity']['unique']} "
            f"duplicates:{len(report['episode_identity']['duplicates'])} "
            f"duplicate_stems:{len(report['episode_identity']['duplicate_stems'])}"
        ),
        (
            "language="
            f"unique:{report['language']['unique_original_instructions']} "
            "multi_task_instruction:"
            f"{len(report['language']['tasks_with_multiple_hdf5_instructions'])} "
            f"json_mismatch:{len(report['language']['hdf5_vs_json_mismatch'])}"
        ),
        f"depth={json.dumps(report['depth']['availability_sets'], sort_keys=True)}",
        f"action_qpos_widths={json.dumps(report['action_qpos_widths'], sort_keys=True)}",
    ]
    if "numeric" in report:
        for side in ("left", "right"):
            row = report["numeric"]["gripper"][side]
            lines.append(
                f"gripper_{side}_value={json.dumps(row['action_value'], sort_keys=True)}"
            )
            lines.append(
                f"gripper_{side}_abs_delta={json.dumps(row['boundary_abs_delta'], sort_keys=True)}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only hierarchical RDT fine-tuning data audit"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return success even when malformed episodes are reported",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_rdt_ft_data(
        args.root,
        max_episodes=args.max_episodes,
        schema_only=args.schema_only,
    )
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_text_report(report))
    if report["error_count"] and not args.allow_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = ["audit_rdt_ft_data", "discover_hdf5", "episode_id"]
