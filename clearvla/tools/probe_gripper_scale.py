"""Read-only amplitude audit for continuous and binary gripper outlets.

This probe deliberately does not construct the policy or change any training
configuration.  It measures the source action chart, the fitted affine action
normalizer, and the six-channel compatibility field used by the shared codec.
For a binary CALVIN outlet it also reports the field that is actually exposed
to dynamic model consumers (which is zero by contract) and the normalized
distance between the two native commands.  The result is evidence for a
normalization decision, not a proposed loss weight.

Typical remote usage::

    python clearvla/tools/probe_gripper_scale.py \
      --calvin-root /data/senwang/data/calvin/converted/abc_d_full \
      --calvin-split-manifest /data/senwang/data/calvin/converted/abc_d_full_raw_overlay_v2_splits.json \
      --pen-root /data/liang.zhang/dataset/grab_pen_single/grab_pen_single \
      --pen-train-count 63

The scan is HDF5 read-only.  No cache, checkpoint, or report is overwritten;
``--output`` refuses to replace an existing file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


ACTION_DIM = 7
ARM_DIM = 6
ARM_INDICES = slice(0, ARM_DIM)
GRIPPER_INDEX = 6
CALVIN_ARM_PHYSICAL_SCALE = np.asarray(
    [0.02, 0.02, 0.02, 0.05, 0.05, 0.05], dtype=np.float64
)
GRIPPER_FIELD_NAMES = (
    "absolute",
    "delta",
    "anchor_delta",
    "previous",
    "abs_delta",
    "positive_delta",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", type=Path, required=True)
    parser.add_argument("--pen-root", type=Path, required=True)
    parser.add_argument("--calvin-split-manifest", type=Path)
    parser.add_argument("--pen-train-count", type=int, default=63)
    parser.add_argument("--calvin-max-files", type=int, default=0)
    parser.add_argument("--pen-max-files", type=int, default=0)
    parser.add_argument("--min-std", type=float, default=1e-2)
    parser.add_argument("--output", type=Path)
    return parser


def _files(root: Path, limit: int) -> list[Path]:
    paths = sorted(root.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"no *.hdf5 files under {root}")
    if int(limit) > 0:
        paths = paths[: int(limit)]
    return paths


def _manifest_train_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("train"), list):
        raise ValueError("split manifest must contain a list at splits.train")
    # Manifests use either the HDF5 stem or an explicit .hdf5 filename.
    return {Path(str(value)).stem for value in splits["train"]}


def _read_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if "action" not in handle or "observations" not in handle:
            raise ValueError(f"{path}: missing action/observations")
        action = np.asarray(handle["action"], dtype=np.float32)
        if "action_state" in handle:
            action_state = np.asarray(handle["action_state"], dtype=np.float32)
        elif "state" in handle:
            # Legacy Pen episodes use qpos/state as the action-chart boundary.
            action_state = np.asarray(handle["state"], dtype=np.float32)
        elif "observations" in handle and "qpos" in handle["observations"]:
            action_state = np.asarray(handle["observations"]["qpos"], dtype=np.float32)
        else:
            raise ValueError(f"{path}: no action-state or state array")
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"{path}: action must be [N,7], got {action.shape}")
    if action_state.shape != action.shape:
        raise ValueError(
            f"{path}: action_state must align with action, got {action_state.shape}"
        )
    if not np.isfinite(action).all() or not np.isfinite(action_state).all():
        raise ValueError(f"{path}: non-finite action/state values")
    return action, action_state


def _concat_rows(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for path in paths:
        action, state = _read_episode(path)
        actions.append(action)
        states.append(state)
    if not actions:
        raise ValueError("cannot concatenate an empty episode set")
    return np.concatenate(actions, axis=0), np.concatenate(states, axis=0)


def _fit_zscore(actions: np.ndarray, *, min_std: float) -> dict[str, np.ndarray]:
    mean = actions.mean(axis=0, keepdims=True).astype(np.float64)
    std = np.maximum(actions.std(axis=0, keepdims=True), float(min_std)).astype(np.float64)
    return {
        "mean": mean,
        "std": std,
        "offset": -mean / std,
        "scale": 1.0 / std,
        "minimum": actions.min(axis=0, keepdims=True).astype(np.float64),
        "maximum": actions.max(axis=0, keepdims=True).astype(np.float64),
    }


def _quantiles(value: np.ndarray) -> dict[str, float]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "p01": float(np.quantile(flat, 0.01)),
        "p05": float(np.quantile(flat, 0.05)),
        "p25": float(np.quantile(flat, 0.25)),
        "p50": float(np.quantile(flat, 0.50)),
        "p75": float(np.quantile(flat, 0.75)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
    }


def _scalar_stats(value: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "quantiles": _quantiles(flat),
    }


def _vector_stats(value: np.ndarray) -> dict[str, Any]:
    """Report per-coordinate and per-row RMS without inventing units."""

    matrix = np.asarray(value, dtype=np.float64).reshape(-1, ARM_DIM)
    if matrix.size == 0:
        raise ValueError("cannot summarize an empty arm vector")
    return {
        "rows": int(matrix.shape[0]),
        "vector_rms_per_coordinate": float(np.sqrt(np.mean(np.square(matrix)))),
        "row_l2_rms": float(np.sqrt(np.mean(np.sum(np.square(matrix), axis=1)))),
        "per_channel": [
            _scalar_stats(matrix[:, index]) for index in range(ARM_DIM)
        ],
    }


def _field(action: np.ndarray, action_state: np.ndarray, normalizer: dict[str, np.ndarray]) -> np.ndarray:
    normalized = action * normalizer["scale"] + normalizer["offset"]
    state = action_state * normalizer["scale"] + normalizer["offset"]
    previous = np.concatenate(
        (state[:1, GRIPPER_INDEX], normalized[:-1, GRIPPER_INDEX]), axis=0
    )
    grip = normalized[:, GRIPPER_INDEX]
    delta = grip - previous
    anchor_delta = grip - state[0, GRIPPER_INDEX]
    return np.stack(
        (grip, delta, anchor_delta, previous, np.abs(delta), np.maximum(delta, 0.0)),
        axis=-1,
    )


def _episode_field_stats(
    paths: Iterable[Path], normalizer: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fields: list[np.ndarray] = []
    raw_deltas: list[np.ndarray] = []
    event_rows = 0
    transitions = 0
    total_rows = 0
    unique_values: set[float] = set()
    for path in paths:
        action, state = _read_episode(path)
        fields.append(_field(action, state, normalizer))
        raw_previous = np.concatenate(
            (state[:1, GRIPPER_INDEX], action[:-1, GRIPPER_INDEX]), axis=0
        )
        raw_delta = action[:, GRIPPER_INDEX] - raw_previous
        raw_deltas.append(raw_delta)
        event_rows += int(np.count_nonzero(np.abs(raw_delta) >= 0.1))
        transitions += int(np.count_nonzero(np.abs(raw_delta) > 1e-8))
        total_rows += int(action.shape[0])
        unique_values.update(float(v) for v in np.unique(action[:, GRIPPER_INDEX]))
    field = np.concatenate(fields, axis=0)
    delta = np.concatenate(raw_deltas, axis=0)
    return field, delta, {
        "rows": total_rows,
        "event_rate_threshold_0.1": event_rows / max(total_rows, 1),
        "transition_rate": transitions / max(total_rows, 1),
        "unique_count": len(unique_values),
        "unique_sample": sorted(unique_values)[:32],
    }


def _episode_arm_stats(
    paths: Iterable[Path], normalizer: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    raw_actions: list[np.ndarray] = []
    normalized_actions: list[np.ndarray] = []
    raw_deltas: list[np.ndarray] = []
    normalized_deltas: list[np.ndarray] = []
    for path in paths:
        action, state = _read_episode(path)
        normalized = action * normalizer["scale"] + normalizer["offset"]
        normalized_state = state * normalizer["scale"] + normalizer["offset"]
        raw_previous = np.concatenate((state[:1, ARM_INDICES], action[:-1, ARM_INDICES]), axis=0)
        normalized_previous = np.concatenate(
            (normalized_state[:1, ARM_INDICES], normalized[:-1, ARM_INDICES]), axis=0
        )
        raw_actions.append(action[:, ARM_INDICES])
        normalized_actions.append(normalized[:, ARM_INDICES])
        raw_deltas.append(action[:, ARM_INDICES] - raw_previous)
        normalized_deltas.append(normalized[:, ARM_INDICES] - normalized_previous)
    return {
        "raw_action": np.concatenate(raw_actions, axis=0),
        "normalized_action": np.concatenate(normalized_actions, axis=0),
        "raw_adjacent_delta": np.concatenate(raw_deltas, axis=0),
        "normalized_adjacent_delta": np.concatenate(normalized_deltas, axis=0),
    }


def _dataset_report(
    *,
    name: str,
    all_paths: list[Path],
    fit_paths: list[Path],
    min_std: float,
    binary_command: bool,
    arm_motion_mode: str,
) -> dict[str, Any]:
    fit_action, _fit_state = _concat_rows(fit_paths)
    normalizer = _fit_zscore(fit_action, min_std=min_std)
    action, state = _concat_rows(all_paths)
    normalized = action * normalizer["scale"] + normalizer["offset"]
    field, raw_delta, event_meta = _episode_field_stats(all_paths, normalizer)
    arm = _episode_arm_stats(all_paths, normalizer)
    grip_raw = action[:, GRIPPER_INDEX]
    grip_normalized = normalized[:, GRIPPER_INDEX]
    command_values = sorted(float(v) for v in np.unique(grip_raw))
    report: dict[str, Any] = {
        "name": name,
        "files": len(all_paths),
        "fit_files": len(fit_paths),
        "rows": int(action.shape[0]),
        "raw_gripper": _scalar_stats(grip_raw),
        "normalized_gripper": _scalar_stats(grip_normalized),
        "raw_gripper_delta": _scalar_stats(raw_delta),
        "raw_arm_action": _vector_stats(arm["raw_action"]),
        "normalized_arm_action": _vector_stats(arm["normalized_action"]),
        "raw_arm_adjacent_delta": _vector_stats(arm["raw_adjacent_delta"]),
        "normalized_arm_adjacent_delta": _vector_stats(
            arm["normalized_adjacent_delta"]
        ),
        "arm_motion_mode": str(arm_motion_mode),
        "raw_arm_motion_signal": _vector_stats(
            arm["raw_action"] if arm_motion_mode == "relative_command" else arm["raw_adjacent_delta"]
        ),
        "normalized_arm_motion_signal": _vector_stats(
            arm["normalized_action"]
            if arm_motion_mode == "relative_command"
            else arm["normalized_adjacent_delta"]
        ),
        "normalizer_gripper": {
            key: float(value[0, GRIPPER_INDEX]) for key, value in normalizer.items()
        },
        "gripper_field": {
            field_name: _scalar_stats(field[:, index])
            for index, field_name in enumerate(GRIPPER_FIELD_NAMES)
        },
        "events": event_meta,
        "binary_command": bool(binary_command),
        "native_unique_values": command_values[:32],
    }
    if binary_command:
        scale = float(normalizer["scale"][0, GRIPPER_INDEX])
        offset = float(normalizer["offset"][0, GRIPPER_INDEX])
        report["binary_chart"] = {
            "class_zero_native_command": -1.0,
            "class_one_native_command": 1.0,
            "normalized_raw_zero": offset,
            "normalized_minus_one": -scale + offset,
            "normalized_plus_one": scale + offset,
            "normalized_command_separation": 2.0 * scale,
            "model_consumed_compatibility_field_rms": 0.0,
        }
        physical_arm = arm["raw_action"] * CALVIN_ARM_PHYSICAL_SCALE[None, :]
        report["calvin_physical_arm_effect"] = _vector_stats(physical_arm)
    else:
        report["binary_chart"] = None
        report["calvin_physical_arm_effect"] = None
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_once(path: Path, payload: str) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n", encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    if args.pen_train_count <= 0:
        raise ValueError("--pen-train-count must be positive")
    if args.min_std <= 0.0 or not math.isfinite(args.min_std):
        raise ValueError("--min-std must be finite and positive")
    calvin_paths = _files(args.calvin_root, args.calvin_max_files)
    pen_paths = _files(args.pen_root, args.pen_max_files)
    calvin_train_names = _manifest_train_names(args.calvin_split_manifest)
    if calvin_train_names is None:
        calvin_fit_paths = calvin_paths
        calvin_fit_source = "all_selected_files"
    else:
        calvin_fit_paths = [path for path in calvin_paths if path.stem in calvin_train_names]
        if not calvin_fit_paths:
            raise ValueError("CALVIN split manifest did not match any selected HDF5 files")
        calvin_fit_source = "manifest_train"
    pen_fit_paths = pen_paths[: args.pen_train_count]
    payload = {
        "schema": "clearvla-gripper-amplitude-probe-v1",
        "read_only": True,
        "normalizer": "zscore",
        "normalizer_fit": {
            "calvin": calvin_fit_source,
            "pen": "ordered_selected_files_prefix",
            "min_std": float(args.min_std),
        },
        "field_contract": {
            "channels": list(GRIPPER_FIELD_NAMES),
            "binary_dynamic_input_rms": 0.0,
            "note": "CALVIN compatibility field is not a formal objective or dynamic input",
        },
        "datasets": {
            "calvin": _dataset_report(
                name="calvin",
                all_paths=calvin_paths,
                fit_paths=calvin_fit_paths,
                min_std=args.min_std,
                binary_command=True,
                arm_motion_mode="relative_command",
            ),
            "pen": _dataset_report(
                name="pen",
                all_paths=pen_paths,
                fit_paths=pen_fit_paths,
                min_std=args.min_std,
                binary_command=False,
                arm_motion_mode="adjacent_action_delta",
            ),
        },
    }
    calvin = payload["datasets"]["calvin"]
    pen = payload["datasets"]["pen"]
    payload["comparison"] = {
        "raw_gripper_rms_ratio_calvin_over_pen": calvin["raw_gripper"]["rms"]
        / max(pen["raw_gripper"]["rms"], 1e-12),
        "normalized_gripper_rms_ratio_calvin_over_pen": calvin["normalized_gripper"]["rms"]
        / max(pen["normalized_gripper"]["rms"], 1e-12),
        "compatibility_field_rms_ratio_calvin_over_pen": {
            key: calvin["gripper_field"][key]["rms"]
            / max(pen["gripper_field"][key]["rms"], 1e-12)
            for key in GRIPPER_FIELD_NAMES
        },
        "raw_arm_motion_vector_rms_ratio_calvin_over_pen": calvin[
            "raw_arm_motion_signal"
        ]["vector_rms_per_coordinate"]
        / max(pen["raw_arm_motion_signal"]["vector_rms_per_coordinate"], 1e-12),
        "normalized_arm_motion_vector_rms_ratio_calvin_over_pen": calvin[
            "normalized_arm_motion_signal"
        ]["vector_rms_per_coordinate"]
        / max(
            pen["normalized_arm_motion_signal"]["vector_rms_per_coordinate"],
            1e-12,
        ),
        "calvin_raw_arm_motion_to_gripper_rms": calvin["raw_arm_motion_signal"][
            "vector_rms_per_coordinate"
        ]
        / max(calvin["raw_gripper"]["rms"], 1e-12),
        "calvin_physical_arm_effect_to_gripper_rms": calvin[
            "calvin_physical_arm_effect"
        ]["vector_rms_per_coordinate"]
        / max(calvin["raw_gripper"]["rms"], 1e-12),
        "interpretation": (
            "These ratios describe chart/data amplitude only. They do not justify "
            "matching binary CE to continuous MSE or changing a loss weight."
        ),
    }
    rendered = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        _write_once(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
