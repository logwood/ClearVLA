"""Content-verified bounded task selection layered on an RDT split manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .split import RDT_SPLIT_NAMES

RDT_MULTITASK_SELECTION_SCHEMA = "clearvla-rdt-multitask-selection-v1"
RDT_MULTITASK_INTERNAL_SPLITS = ("train", "val", "test")


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rdt_multitask_selection_manifest(
    path: str | Path,
    *,
    episode_names: Sequence[str],
    task_names: Sequence[str],
    instructions: Sequence[str | None],
    base_splits: Mapping[str, Sequence[int]],
    base_split_metadata: Mapping[str, object],
    expected_task_count: int = 8,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Validate and resolve a task selection without creating a model input.

    Task identity determines dataset membership only.  The returned indices
    are CPU-side split metadata; no task row or ID is added to a sample.
    ``external_test`` remains exactly the base manifest lane and is never
    admitted to the selected internal train/validation/test scope.
    """

    count = len(episode_names)
    if len(task_names) != count or len(instructions) != count:
        raise ValueError("episode/task/instruction identity lengths must match")
    if set(base_splits) != set(RDT_SPLIT_NAMES):
        raise ValueError("base split must contain the four RDT lanes")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"RDT task selection manifest does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != RDT_MULTITASK_SELECTION_SCHEMA:
        raise ValueError("unsupported RDT multitask selection schema")
    recorded_digest = str(payload.get("selection_sha256", ""))
    digest_payload = dict(payload)
    digest_payload.pop("selection_sha256", None)
    if recorded_digest != _canonical_digest(digest_payload):
        raise ValueError("RDT multitask selection content digest is inconsistent")

    task_order_value = payload.get("task_order")
    if not isinstance(task_order_value, list) or any(
        not isinstance(value, str) or not value for value in task_order_value
    ):
        raise TypeError("RDT multitask task_order must be non-empty task IDs")
    task_order = tuple(task_order_value)
    if len(task_order) != int(expected_task_count) or len(set(task_order)) != len(task_order):
        raise ValueError(
            f"RDT multitask selection must contain exactly {expected_task_count} unique tasks"
        )
    policy = payload.get("policy")
    required_policy = {
        "task_count",
        "task_id_usage",
        "model_conditioning",
        "episode_scope",
        "external_test",
    }
    if not isinstance(policy, dict) or set(policy) != required_policy:
        raise ValueError("RDT multitask selection policy is incomplete")
    if (
        int(policy["task_count"]) != len(task_order)
        or policy["task_id_usage"] != "cpu_audit_sampling_logging_metadata_only"
        or policy["model_conditioning"] is not False
        or policy["episode_scope"] != "all_typed_window_eligible_episodes_per_selected_task"
        or policy["external_test"] != "preserved_external_only_not_selected_or_tuned"
    ):
        raise ValueError("RDT multitask selection policy differs from the adopted boundary")

    base_identity = payload.get("base_split_manifest")
    if not isinstance(base_identity, dict):
        raise ValueError("selection must identify its base split manifest")
    for field in (
        "file_sha256",
        "manifest_sha256",
        "source_episode_inventory_sha256",
        "episode_inventory_sha256",
    ):
        if str(base_identity.get(field, "")) != str(base_split_metadata.get(field, "")):
            raise ValueError(f"selection base split {field} differs from the live manifest")

    names = [str(value) for value in episode_names]
    tasks = [str(value) for value in task_names]
    texts = [None if value is None else str(value) for value in instructions]
    if len(set(names)) != count:
        raise ValueError("episode identities must be unique")
    base_membership: dict[int, str] = {}
    for split in RDT_SPLIT_NAMES:
        for index_value in base_splits[split]:
            index = int(index_value)
            if not 0 <= index < count or index in base_membership:
                raise ValueError("base split indices are invalid or overlapping")
            base_membership[index] = split
    if len(base_membership) != count:
        raise ValueError("base split does not cover every eligible episode")

    observed_task_instructions: dict[str, set[str]] = {}
    for task, text in zip(tasks, texts, strict=True):
        if text is None:
            raise ValueError("selected RDT inventory requires HDF5 instructions")
        observed_task_instructions.setdefault(task, set()).add(text)
    task_records_value = payload.get("tasks")
    if not isinstance(task_records_value, list) or len(task_records_value) != len(task_order):
        raise ValueError("selection must contain one record per ordered task")
    task_records: dict[str, dict[str, object]] = {}
    for expected_task, value in zip(task_order, task_records_value, strict=True):
        if not isinstance(value, dict) or str(value.get("task_id", "")) != expected_task:
            raise ValueError("selection task records must follow task_order exactly")
        if expected_task in task_records:
            raise ValueError("selection task records cannot repeat")
        live_instructions = observed_task_instructions.get(expected_task)
        if live_instructions is None or len(live_instructions) != 1:
            raise ValueError(f"selected task has missing or ambiguous instruction: {expected_task}")
        instruction = next(iter(live_instructions))
        if str(value.get("instruction", "")) != instruction:
            raise ValueError(f"selection instruction differs from HDF5: {expected_task}")
        if str(value.get("instruction_sha256", "")) != hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest():
            raise ValueError(f"selection instruction digest is stale: {expected_task}")
        role = value.get("left_role_audit")
        if not isinstance(role, dict) or role.get("required_support_or_collaboration") is not False:
            raise ValueError(f"selected task lacks a negative left-role audit: {expected_task}")
        task_records[expected_task] = value

    selected_task_set = set(task_order)
    expected_selected: dict[str, list[int]] = {
        split: [
            index
            for index in base_splits[split]
            if tasks[int(index)] in selected_task_set
        ]
        for split in RDT_MULTITASK_INTERNAL_SPLITS
    }
    for split, indices in expected_selected.items():
        indices.sort(key=lambda index: (task_order.index(tasks[index]), names[index]))
        if not indices:
            raise ValueError(f"selected internal split {split!r} is empty")
    selected_payload = payload.get("splits")
    if not isinstance(selected_payload, dict) or set(selected_payload) != set(
        RDT_MULTITASK_INTERNAL_SPLITS
    ):
        raise ValueError("selection must serialize exactly train/val/test")
    name_to_index = {name: index for index, name in enumerate(names)}
    resolved: dict[str, list[int]] = {}
    for split in RDT_MULTITASK_INTERNAL_SPLITS:
        values = selected_payload[split]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise TypeError(f"selection split {split!r} must be episode identities")
        expected_names = [names[index] for index in expected_selected[split]]
        if values != expected_names:
            raise ValueError(f"selection split {split!r} differs from base task filtering")
        resolved[split] = [name_to_index[value] for value in values]

    for task in task_order:
        record = task_records[task]
        expected_task_splits = {
            split: [names[index] for index in expected_selected[split] if tasks[index] == task]
            for split in RDT_MULTITASK_INTERNAL_SPLITS
        }
        if any(not values for values in expected_task_splits.values()):
            raise ValueError(f"selected task lacks a non-empty internal split: {task}")
        if record.get("splits") != expected_task_splits:
            raise ValueError(f"selected task split identity is stale: {task}")

    external_indices = [int(value) for value in base_splits["external_test"]]
    external_names = [names[index] for index in external_indices]
    external_identity = payload.get("external_test_identity")
    expected_external = {
        "episode_count": len(external_names),
        "episode_inventory_sha256": _canonical_digest(external_names),
        "membership": "base_manifest_external_test_only",
        "selected": False,
        "used_for_training_or_tuning": False,
    }
    if external_identity != expected_external:
        raise ValueError("selection external_test identity differs from the base lane")

    all_splits = {**resolved, "external_test": external_indices}
    metadata: dict[str, object] = {
        "schema": RDT_MULTITASK_SELECTION_SCHEMA,
        "path": str(source.resolve()),
        "file_sha256": _file_sha256(source),
        "selection_sha256": recorded_digest,
        "task_order": list(task_order),
        "task_count": len(task_order),
        "split_counts": {name: len(values) for name, values in all_splits.items()},
        "internal_episode_inventory_sha256": _canonical_digest(
            [names[index] for split in RDT_MULTITASK_INTERNAL_SPLITS for index in resolved[split]]
        ),
        "external_test_identity": expected_external,
        "model_conditioning": False,
        "task_id_usage": "cpu_audit_sampling_logging_metadata_only",
    }
    return all_splits, metadata


__all__ = [
    "RDT_MULTITASK_INTERNAL_SPLITS",
    "RDT_MULTITASK_SELECTION_SCHEMA",
    "load_rdt_multitask_selection_manifest",
]
