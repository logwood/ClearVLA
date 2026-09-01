"""Train-only right-gripper evidence for a bounded RDT task selection.

The artifact deliberately does not adopt an event threshold.  It reports the
continuous source charts, exact adjacent-command deltas consumed by the
24-step information sampler, and explicitly *descriptive* counterfactual
activity statistics using only the selected train lane.  Validation/test rows
are never opened for threshold selection.  Observed qpos remains reported as
a physical-boundary audit but never becomes a command-transition label.

The older v1 audit mixed the first command delta with converted qpos.  Its
numeric ``candidate_thresholds`` are therefore retained only as historical
evidence and are never emitted by this producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import LoadedEpisode, episode_identity, find_hdf5_files
from clearvla.data.multitask_selection import RDT_MULTITASK_SELECTION_SCHEMA
from clearvla.data.split import RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH

GRIPPER_AUDIT_SCHEMA = "clearvla-rdt-multitask-gripper-train-audit-v3"
PROFILE_NAME = "rdt_right_arm_action_chart_v1"
POLICY_HORIZON = 24
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 1.0)
DESCRIPTIVE_QUANTILES = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)


def _canonical_digest(value: object, *, ensure_ascii: bool) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
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


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _merge(values: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    rows = [value for value in rows if value.size]
    return np.concatenate(rows) if rows else np.empty((0,), dtype=np.float64)


def _quantile_key(value: float) -> str:
    scaled = value * 100.0
    return f"p{scaled:g}".replace(".", "_")


def _series_stats(values: Iterable[np.ndarray]) -> dict[str, object]:
    merged = _merge(values)
    if not merged.size:
        raise ValueError("cannot summarize an empty gripper series")
    if not np.isfinite(merged).all():
        raise ValueError("gripper evidence contains NaN or infinity")
    quantiles = np.quantile(merged, QUANTILES)
    return {
        "count": int(merged.size),
        "mean": float(merged.mean(dtype=np.float64)),
        "std": float(merged.std(dtype=np.float64)),
        "rms": float(np.sqrt(np.square(merged, dtype=np.float64).mean(dtype=np.float64))),
        "quantiles": {
            _quantile_key(q): float(value)
            for q, value in zip(QUANTILES, quantiles, strict=True)
        },
    }


def _true_run_lengths(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not values.size:
        return np.empty((0,), dtype=np.int64)
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts).astype(np.int64)


def _verified_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RDT_MULTITASK_SELECTION_SCHEMA:
        raise ValueError("unsupported RDT multitask selection schema")
    recorded = str(payload.get("selection_sha256", ""))
    digest_payload = dict(payload)
    digest_payload.pop("selection_sha256", None)
    if recorded != _canonical_digest(digest_payload, ensure_ascii=False):
        raise ValueError("RDT multitask selection content digest is inconsistent")
    return payload


def _episode_evidence(
    *,
    root: Path,
    path: Path,
    episode_id: str,
    task_id: str,
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    live_id, partition, live_task = episode_identity(root, path)
    if live_id != episode_id or partition != "rdt_data" or live_task != task_id:
        raise ValueError(f"selected train episode identity changed: {episode_id}")
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle["action"], dtype=np.float32)
        qpos = np.asarray(handle["observations/qpos"], dtype=np.float32)
    if action.shape != qpos.shape or action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"selected train episode is not aligned native 14D: {episode_id}")
    if not np.isfinite(action).all() or not np.isfinite(qpos).all():
        raise ValueError(f"selected train episode contains NaN or infinity: {episode_id}")
    if (
        int(manifest_row.get("length", -1)) != int(action.shape[0])
        or str(manifest_row.get("action_storage_sha256", "")) != _array_sha256(action)
        or str(manifest_row.get("qpos_storage_sha256", "")) != _array_sha256(qpos)
    ):
        raise ValueError(f"selected train episode numeric identity is stale: {episode_id}")

    native = LoadedEpisode(
        path=path,
        episode_id=episode_id,
        source_partition="rdt_data",
        task_id=task_id,
        action_key="action",
        camera_keys={},
        actions_raw=action,
        state_key="observations/qpos",
        states_raw=qpos,
        action_states_raw=qpos,
        source_action_dim=14,
        source_state_dim=14,
    )
    profile = resolve_action_state_profile(PROFILE_NAME)
    projected = profile.project_episode(native)
    if projected.states_raw is None or projected.action_states_raw is None:
        raise AssertionError("right-arm projection lost an observed state chart")
    command = np.asarray(projected.actions_raw[:, 6], dtype=np.float64)
    qpos_native = np.asarray(projected.states_raw[:, 6], dtype=np.float64)
    qpos_action_chart = np.asarray(projected.action_states_raw[:, 6], dtype=np.float64)
    step_delta = np.diff(command)
    command_gap = command - qpos_action_chart

    low_center = 24
    high_center = int(len(command) - 49)
    if high_center < low_center:
        raise ValueError(f"selected train episode cannot form a typed window: {episode_id}")
    sampler_rows: list[np.ndarray] = []
    window_max_abs: list[float] = []
    for center in range(low_center, high_center + 1):
        policy = command[center : center + POLICY_HORIZON]
        delta = np.concatenate(
            ((policy[:1] - command[center - 1]), np.diff(policy))
        )
        sampler_rows.append(delta)
        window_max_abs.append(float(np.abs(delta).max()))
    expected_windows = int(len(command) - RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH + 1)
    if len(sampler_rows) != expected_windows or int(
        manifest_row.get("valid_windows", -1)
    ) != expected_windows:
        raise ValueError(f"selected train episode typed-window identity is stale: {episode_id}")
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "rows": int(len(command)),
        "valid_windows": expected_windows,
        "command": command,
        "qpos_native": qpos_native,
        "qpos_action_chart": qpos_action_chart,
        "step_delta": step_delta,
        "command_gap": command_gap,
        "sampler_delta": np.concatenate(sampler_rows),
        "window_max_abs": np.asarray(window_max_abs, dtype=np.float64),
    }


def _scope_stats(rows: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "episode_count": len(rows),
        "row_count": sum(int(row["rows"]) for row in rows),
        "valid_window_count": sum(int(row["valid_windows"]) for row in rows),
        "continuous_right_gripper": {
            "action_command_native_chart": _series_stats(row["command"] for row in rows),
            "qpos_native_chart": _series_stats(row["qpos_native"] for row in rows),
            "qpos_converted_to_action_chart": _series_stats(
                row["qpos_action_chart"] for row in rows
            ),
            "command_step_delta": _series_stats(row["step_delta"] for row in rows),
            "command_minus_same_row_qpos_action_chart": _series_stats(
                row["command_gap"] for row in rows
            ),
            "exact_sampler_boundary_delta": _series_stats(
                row["sampler_delta"] for row in rows
            ),
        },
    }


def _descriptive_activity_stats(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, object]:
    task_ids = []
    for row in rows:
        if row["task_id"] not in task_ids:
            task_ids.append(row["task_id"])
    task_records: list[dict[str, object]] = []
    all_active_runs: list[np.ndarray] = []
    all_inactive_runs: list[np.ndarray] = []
    for task_id in task_ids:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        steps = _merge(row["step_delta"] for row in task_rows)
        active_runs = [
            _true_run_lengths(np.abs(row["step_delta"]) >= threshold) for row in task_rows
        ]
        inactive_runs = [
            _true_run_lengths(np.abs(row["step_delta"]) < threshold) for row in task_rows
        ]
        all_active_runs.extend(active_runs)
        all_inactive_runs.extend(inactive_runs)
        windows = _merge(row["window_max_abs"] for row in task_rows)
        significant = np.abs(steps) >= threshold
        task_records.append(
            {
                "task_id": task_id,
                "transition_rows": int(steps.size),
                "active_transition_rows": int(np.count_nonzero(significant)),
                "positive_active_transition_rows": int(
                    np.count_nonzero(steps >= threshold)
                ),
                "negative_active_transition_rows": int(
                    np.count_nonzero(steps <= -threshold)
                ),
                "counterfactual_activity_segment_count": int(
                    sum(int(value.size) for value in active_runs)
                ),
                "counterfactual_activity_segment_length": (
                    None
                    if not any(value.size for value in active_runs)
                    else _series_stats(active_runs)
                ),
                "counterfactual_persistence_segment_count": int(
                    sum(int(value.size) for value in inactive_runs)
                ),
                "counterfactual_persistence_segment_length": _series_stats(
                    inactive_runs
                ),
                "typed_windows": int(windows.size),
                "counterfactual_active_windows": int(
                    np.count_nonzero(windows >= threshold)
                ),
                "counterfactual_active_window_fraction": float(
                    np.mean(windows >= threshold)
                ),
            }
        )
    all_steps = _merge(row["step_delta"] for row in rows)
    all_windows = _merge(row["window_max_abs"] for row in rows)
    return {
        "raw_abs_adjacent_command_delta_quantile": float(threshold),
        "source_units": "raw_action_command_chart",
        "semantic_status": "descriptive_only_not_a_training_threshold",
        "eligible_for_threshold_adoption": False,
        "transition_rows": int(all_steps.size),
        "active_transition_rows": int(
            np.count_nonzero(np.abs(all_steps) >= threshold)
        ),
        "positive_active_transition_rows": int(
            np.count_nonzero(all_steps >= threshold)
        ),
        "negative_active_transition_rows": int(
            np.count_nonzero(all_steps <= -threshold)
        ),
        "counterfactual_activity_segment_count": int(
            sum(int(value.size) for value in all_active_runs)
        ),
        "counterfactual_activity_segment_length": (
            None
            if not any(value.size for value in all_active_runs)
            else _series_stats(all_active_runs)
        ),
        "counterfactual_persistence_segment_count": int(
            sum(int(value.size) for value in all_inactive_runs)
        ),
        "counterfactual_persistence_segment_length": _series_stats(
            all_inactive_runs
        ),
        "typed_windows": int(all_windows.size),
        "counterfactual_active_windows": int(
            np.count_nonzero(all_windows >= threshold)
        ),
        "counterfactual_active_window_fraction": float(
            np.mean(all_windows >= threshold)
        ),
        "tasks": task_records,
    }


def audit_rdt_multitask_gripper(
    *,
    data_root: Path,
    selection_manifest: Path,
) -> dict[str, object]:
    root = data_root.expanduser().resolve()
    selection_path = selection_manifest.expanduser().resolve()
    selection = _verified_selection(selection_path)
    task_order = [str(value) for value in selection.get("task_order", [])]
    tasks_value = selection.get("tasks")
    if len(task_order) != 8 or not isinstance(tasks_value, list) or len(tasks_value) != 8:
        raise ValueError("gripper audit requires the adopted exact eight-task selection")
    if [str(value.get("task_id", "")) for value in tasks_value] != task_order:
        raise ValueError("selection task records do not follow task_order")
    model_abi = selection.get("model_abi")
    if not isinstance(model_abi, dict) or (
        model_abi.get("action_profile") != PROFILE_NAME
        or int(model_abi.get("action_dim", -1)) != 7
        or int(model_abi.get("action_horizon", -1)) != POLICY_HORIZON
    ):
        raise ValueError("selection model ABI differs from the gripper audit boundary")

    train_ids = [str(value) for value in selection.get("splits", {}).get("train", [])]
    if not train_ids or len(set(train_ids)) != len(train_ids):
        raise ValueError("selection train episode identities are missing or duplicated")
    manifest_rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for task in tasks_value:
        task_id = str(task["task_id"])
        episodes = task.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"selection task has no episode evidence: {task_id}")
        for row in episodes:
            if not isinstance(row, dict) or row.get("split") != "train":
                continue
            episode_id = str(row.get("episode_id", ""))
            if episode_id in manifest_rows:
                raise ValueError(f"duplicate train episode evidence: {episode_id}")
            manifest_rows[episode_id] = (task_id, row)
    if list(manifest_rows) != train_ids:
        raise ValueError("task episode evidence does not reproduce the selected train order")

    paths = {
        episode_identity(root, path)[0]: path
        for path in find_hdf5_files(root, "**/*.hdf5")
    }
    rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, object]] = []
    for episode_id in train_ids:
        task_id, manifest_row = manifest_rows[episode_id]
        path = paths.get(episode_id)
        if path is None:
            raise FileNotFoundError(f"selected train episode disappeared: {episode_id}")
        evidence = _episode_evidence(
            root=root,
            path=path,
            episode_id=episode_id,
            task_id=task_id,
            manifest_row=manifest_row,
        )
        rows.append(evidence)
        episode_records.append(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "rows": evidence["rows"],
                "valid_windows": evidence["valid_windows"],
                "action_storage_sha256": manifest_row["action_storage_sha256"],
                "qpos_storage_sha256": manifest_row["qpos_storage_sha256"],
            }
        )

    sampler_abs = np.abs(_merge(row["sampler_delta"] for row in rows))
    descriptive_values = np.quantile(sampler_abs, DESCRIPTIVE_QUANTILES)
    descriptive_quantiles = [
        {
            "source_quantile": _quantile_key(q),
            **_descriptive_activity_stats(rows, float(value)),
        }
        for q, value in zip(DESCRIPTIVE_QUANTILES, descriptive_values, strict=True)
    ]
    per_task = [
        {
            "task_id": task_id,
            **_scope_stats([row for row in rows if row["task_id"] == task_id]),
        }
        for task_id in task_order
    ]
    profile = resolve_action_state_profile(PROFILE_NAME)
    payload: dict[str, object] = {
        "schema": GRIPPER_AUDIT_SCHEMA,
        "selection_manifest": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(selection_path),
            "selection_sha256": selection["selection_sha256"],
        },
        "fit_scope": "selected_train_split_only_no_val_test_or_external_test_rows_opened",
        "task_order": task_order,
        "train_episode_count": len(rows),
        "train_episode_inventory_sha256": _canonical_digest(train_ids, ensure_ascii=True),
        "train_episode_rows_sha256": _canonical_digest(episode_records, ensure_ascii=True),
        "action_profile": {
            **profile.as_dict(),
            "sha256": profile.digest(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        "sampler_consumer": {
            "policy_horizon": POLICY_HORIZON,
            "first_delta": "action_command[center]-action_command[center-1]",
            "remaining_deltas": "action_command[t]-action_command[t-1]",
            "window_activity_counterfactual": (
                "any_abs_delta_greater_than_or_equal_to_descriptive_quantile"
            ),
            "qpos_role": "physical_decode_boundary_only_not_event_or_activity_label",
        },
        "overall": _scope_stats(rows),
        "tasks": per_task,
        "descriptive_activity_quantiles": descriptive_quantiles,
        "threshold_decision": {
            "status": "unresolved_source_semantics",
            "adopted_value": None,
            "descriptive_source": "selected_train_split_adjacent_command_delta_quantiles_only",
            "descriptive_values_are_thresholds": False,
            "validation_or_test_used": False,
            "pen_0_1_inherited": False,
            "formal_shuffled_training_ready": False,
            "blocker": (
                "The source has no discrete event/activity label. Quantiles are descriptive "
                "counterfactual activity summaries only; no value is eligible for shuffled "
                "training until a source-backed semantic rule is explicitly adopted."
            ),
        },
        "rejected_legacy_candidates": {
            "status": "rejected",
            "legacy_schema": "clearvla-rdt-multitask-gripper-train-audit-v1",
            "reason": (
                "The legacy first sampler row mixed action command with converted qpos; "
                "its candidate values are not comparable to adjacent-command deltas."
            ),
            "use_for_training": False,
        },
    }
    payload["gripper_audit_sha256"] = _canonical_digest(payload, ensure_ascii=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit selected-train RDT right-gripper descriptive activity statistics"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite gripper audit: {destination}")
    payload = audit_rdt_multitask_gripper(
        data_root=args.data_root,
        selection_manifest=args.selection_manifest,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(destination),
                "gripper_audit_sha256": payload["gripper_audit_sha256"],
                "train_episode_count": payload["train_episode_count"],
                "threshold_status": payload["threshold_decision"]["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "GRIPPER_AUDIT_SCHEMA",
    "audit_rdt_multitask_gripper",
]
