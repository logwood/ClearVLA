"""Shared, dependency-light contracts for external benchmark datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from clearvla.data.hdf5_episode import (
    LIBERO_E8_STATE_NORMALIZER_REFERENCE,
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
    load_episode,
)
from clearvla.data.instructions import instruction_key
from clearvla.data.split import EPISODE_SPLIT_MANIFEST_SCHEMA

BENCHMARK_DATASET_SCHEMA = "clearvla-external-benchmark-dataset-v1"
BENCHMARK_EPISODE_SCHEMA = "clearvla-external-benchmark-episode-v1"
LIBERO_CONVERTER_SCHEMA = "clearvla-libero-converter-v1"
LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA = (
    "clearvla-libero-causal-aligned-converter-v2"
)
LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA = (
    "clearvla-libero-terminal-replay-converter-v2"
)
LIBERO_CONVERTER_SCHEMAS = frozenset(
    {
        LIBERO_CONVERTER_SCHEMA,
        LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA,
        LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA,
    }
)


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    """Read one contract JSON object and turn parse failures into clear errors."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _attribute_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return ""
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _validate_split_payload(splits: Mapping[str, Any], *, path: Path) -> dict[str, list[str]]:
    raw = splits.get("splits")
    if not isinstance(raw, Mapping):
        raise ValueError(f"benchmark split manifest has no splits mapping: {path}")
    result: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        values = raw.get(name)
        if not isinstance(values, list):
            raise ValueError(f"benchmark split {name!r} must be a JSON list: {path}")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"benchmark split {name!r} contains an invalid episode id: {path}")
        if len(values) != len(set(values)):
            raise ValueError(f"benchmark split {name!r} contains duplicate episode ids: {path}")
        if not values:
            raise ValueError(f"benchmark split {name!r} is empty: {path}")
        result[name] = [str(value) for value in values]
    return result


def _validate_instruction_payload(
    inventory: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, str]:
    if inventory.get("schema") != "clearvla-instruction-inventory-v1":
        raise ValueError(f"instruction inventory schema differs: {path}")
    rows = inventory.get("instructions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"instruction inventory must contain a non-empty list: {path}")
    result: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"instruction inventory row {index} is not an object: {path}")
        key = row.get("key")
        text = row.get("instruction")
        if not isinstance(key, str) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"instruction inventory row {index} is malformed: {path}")
        if key in result and result[key] != text:
            raise ValueError(f"instruction inventory key {key!r} has conflicting text: {path}")
        if instruction_key(text) != key:
            raise ValueError(
                f"instruction inventory row {index} is not content-addressed: {path}"
            )
        result[key] = text
    return result


def _validate_rgb_datasets(
    stream: h5py.File,
    *,
    length: int,
    path: Path,
) -> None:
    for key in ("observations/images/cam_high", "observations/images/cam_right_wrist"):
        dataset = stream.get(key)
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{path} lacks required RGB dataset {key!r}")
        if (
            dataset.ndim != 4
            or int(dataset.shape[0]) != int(length)
            or int(dataset.shape[-1]) != 3
            or min(int(dataset.shape[1]), int(dataset.shape[2])) <= 0
        ):
            raise ValueError(
                f"{path} RGB dataset {key!r} must be [T,H,W,3] aligned to {length}, "
                f"got {dataset.shape}"
            )
        if dataset.dtype != np.dtype(np.uint8):
            raise ValueError(f"{path} RGB dataset {key!r} must be uint8")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_hdf5(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_split_manifest(
    root: Path,
    splits: Mapping[str, list[str]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "schema": EPISODE_SPLIT_MANIFEST_SCHEMA,
        "splits": {name: list(splits[name]) for name in ("train", "val", "test")},
    }
    if metadata is not None:
        overlap = set(payload).intersection(metadata)
        if overlap:
            raise ValueError(f"split metadata collides with required fields: {sorted(overlap)}")
        payload.update(dict(metadata))
    path = root / "splits.json"
    atomic_json(path, payload)
    return path


def write_instruction_inventory(root: Path, instructions: Mapping[str, str]) -> Path:
    payload = {
        "schema": "clearvla-instruction-inventory-v1",
        "instructions": [
            {"key": key, "instruction": instructions[key]}
            for key in sorted(instructions)
        ],
    }
    path = root / "instructions.json"
    atomic_json(path, payload)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def audit_benchmark_dataset(root: str | Path) -> dict[str, Any]:
    source = Path(root)
    manifest_path = source / "dataset_manifest.json"
    split_path = source / "splits.json"
    instruction_path = source / "instructions.json"
    if not manifest_path.is_file() or not split_path.is_file() or not instruction_path.is_file():
        raise FileNotFoundError("benchmark root lacks manifest, splits, or instructions")
    manifest = _json_object(manifest_path, name="benchmark dataset manifest")
    splits_payload = _json_object(split_path, name="benchmark split manifest")
    inventory_payload = _json_object(instruction_path, name="instruction inventory")
    if manifest.get("schema") != BENCHMARK_DATASET_SCHEMA:
        raise ValueError("benchmark dataset manifest schema differs")
    benchmark = _attribute_text(manifest.get("benchmark", "")).strip().upper()
    if benchmark not in {"CALVIN", "LIBERO"}:
        raise ValueError(f"unsupported benchmark dataset identity {benchmark!r}")
    is_calvin = benchmark == "CALVIN"
    is_libero = benchmark == "LIBERO"
    if is_calvin and (
        manifest.get("converter_schema") != "clearvla-calvin-converter-v2"
        or manifest.get("split_unit") != "source-trajectory"
    ):
        raise ValueError("CALVIN requires converter-v2 trajectory/terminal contract")
    if is_libero:
        converter_schema = str(manifest.get("converter_schema", ""))
        if converter_schema not in LIBERO_CONVERTER_SCHEMAS:
            raise ValueError(
                f"unsupported LIBERO converter schema {converter_schema!r}"
            )
        expected_manifest = {
            "data_profile": "libero_relative_7d_v1",
            "arm_flow_mode": "relative_command_adapter",
            "gripper_output_mode": "continuous",
            "split_unit": "episode",
            "controller": "OSC_POSE",
            "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
            "evaluator_language_verified": True,
        }
        if converter_schema == LIBERO_CONVERTER_SCHEMA:
            expected_manifest.update(
                {
                    "valid_center_start": 24,
                    "valid_center_end": "length - 49",
                }
            )
        elif converter_schema == LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA:
            expected_manifest.update(
                {
                    "window_boundary_contract": "causal_prefix_v1",
                    "valid_center_start": 0,
                    "valid_center_end": "length - 49",
                    "strict_valid_center_start": 24,
                    "strict_valid_center_end": "length - 49",
                    "state_normalizer_reference_semantics": (
                        LIBERO_E8_STATE_NORMALIZER_REFERENCE
                    ),
                }
            )
        else:
            expected_manifest.update(
                {
                    "window_boundary_contract": (
                        "causal_prefix_terminal_suffix_v2"
                    ),
                    "valid_center_start": 0,
                    "valid_center_end": "source_action_count - 1",
                    "strict_valid_center_start": 24,
                    "strict_valid_center_end": "source_action_count - 49",
                    "state_normalizer_reference_semantics": (
                        LIBERO_E8_STATE_NORMALIZER_REFERENCE
                    ),
                    "terminal_padding_mode": (
                        LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                    ),
                    "terminal_suffix_rows": 48,
                }
            )
        missing = sorted(name for name in expected_manifest if name not in manifest)
        if missing:
            raise ValueError(
                "LIBERO dataset manifest is missing required outlet fields: "
                + ", ".join(missing)
            )
        for name, expected in expected_manifest.items():
            if manifest.get(name) != expected:
                raise ValueError(
                    f"LIBERO dataset manifest {name}={manifest.get(name)!r} "
                    f"does not match {expected!r}"
                )
        suites = manifest.get("suites")
        if not isinstance(suites, list) or not suites or any(
            not isinstance(value, str) or not value.strip() for value in suites
        ) or len(suites) != len(set(suites)):
            raise ValueError("LIBERO dataset manifest suites must be non-empty and unique")
        action_range = manifest.get("action_range")
        if action_range != [-1.0, 1.0]:
            raise ValueError("LIBERO dataset manifest action_range must be [-1,1]")
        control_hz = manifest.get("control_hz")
        if (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not np.isfinite(float(control_hz))
            or float(control_hz) != 20.0
        ):
            raise ValueError("LIBERO dataset manifest control_hz must be 20")
        tasks = manifest.get("tasks")
        task_count = manifest.get("task_count")
        if (
            not isinstance(tasks, list)
            or not isinstance(task_count, int)
            or isinstance(task_count, bool)
            or task_count != len(tasks)
            or task_count <= 0
        ):
            raise ValueError("LIBERO dataset manifest task inventory is malformed")
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                raise ValueError(f"LIBERO dataset manifest task row {index} is malformed")
            for field in (
                "suite",
                "task_file",
                "instruction_key",
                "demos",
                "evaluator_task_name",
                "evaluator_instruction",
            ):
                if field not in task:
                    raise ValueError(
                        f"LIBERO dataset manifest task row {index} lacks {field!r}"
                    )
            if not isinstance(task.get("suite"), str) or not str(
                task.get("suite")
            ).strip():
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} has an invalid suite"
                )
            task_file = task.get("task_file")
            if not isinstance(task_file, str) or not task_file.strip():
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} has an invalid task_file"
                )
            task_key = task.get("instruction_key")
            if not isinstance(task_key, str) or len(task_key) != 64 or any(
                character not in "0123456789abcdef" for character in task_key.lower()
            ):
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} has an invalid instruction_key"
                )
            demos = task.get("demos")
            if not isinstance(demos, Mapping) or set(demos) != {
                "train",
                "val",
                "test",
            }:
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} has invalid demos"
                )
            for split_name, count in demos.items():
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    raise ValueError(
                        f"LIBERO dataset manifest task row {index} {split_name} count is invalid"
                    )
            if sum(int(count) for count in demos.values()) <= 0:
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} has no demonstrations"
                )
            evaluator_task_name = task.get("evaluator_task_name")
            if not isinstance(evaluator_task_name, str) or not evaluator_task_name.strip():
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} evaluator task name is invalid"
                )
            evaluator_instruction = task.get("evaluator_instruction")
            if not isinstance(evaluator_instruction, str) or not evaluator_instruction.strip():
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} evaluator instruction is invalid"
                )
            if instruction_key(evaluator_instruction) != task_key:
                raise ValueError(
                    f"LIBERO dataset manifest task row {index} evaluator instruction key differs"
                )
            if "bddl_language_instruction" in task:
                alias = task.get("bddl_language_instruction")
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError(
                        f"LIBERO dataset manifest task row {index} BDDL language alias is invalid"
                    )
    if splits_payload.get("schema") != EPISODE_SPLIT_MANIFEST_SCHEMA:
        raise ValueError("benchmark split manifest schema differs")
    split_values = _validate_split_payload(splits_payload, path=split_path)
    instructions = _validate_instruction_payload(inventory_payload, path=instruction_path)

    expected_names = [
        str(name)
        for split in ("train", "val", "test")
        for name in split_values[split]
    ]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("benchmark split membership overlaps")
    files = {path.stem: path for path in source.glob("*.hdf5")}
    if set(files) != set(expected_names):
        raise ValueError("HDF5 inventory and split manifest differ")
    steps = 0
    valid_windows = 0
    strict_valid_windows = 0
    keys: set[str] = set()
    name_to_split = {
        str(name): split
        for split in ("train", "val", "test")
        for name in split_values[split]
    }
    trajectory_splits: dict[str, set[str]] = {
        name: set() for name in ("train", "val", "test")
    }
    for name in expected_names:
        path = files[name]
        recorded_evaluator_instruction: str | None = None
        with h5py.File(path, "r") as stream:
            if _attribute_text(stream.attrs.get("schema", "")) != BENCHMARK_EPISODE_SCHEMA:
                raise ValueError(f"{path} has the wrong benchmark episode schema")
            if is_libero:
                episode_attrs = {
                    "benchmark": "LIBERO",
                    "converter_schema": converter_schema,
                    "data_profile": "libero_relative_7d_v1",
                    "arm_flow_mode": "relative_command_adapter",
                    "gripper_output_mode": "continuous",
                    "controller": "OSC_POSE",
                }
                for attr_name, expected in episode_attrs.items():
                    actual = _attribute_text(stream.attrs.get(attr_name, "")).strip()
                    if actual != expected:
                        raise ValueError(
                            f"{path} {attr_name}={actual!r} does not match {expected!r}"
                        )
                # The evaluator-facing instruction is a separate boundary
                # from the BDDL declaration.  If the converter recorded both,
                # verify that only the canonical episode text can enter the
                # language inventory; a simulator wording alias may differ.
                evaluator_instruction = stream.attrs.get("evaluator_instruction")
                if evaluator_instruction is not None:
                    evaluator_instruction = _attribute_text(evaluator_instruction)
                    if not evaluator_instruction.strip():
                        raise ValueError(f"{path} evaluator instruction is empty")
                    recorded_evaluator_instruction = evaluator_instruction
                bddl_instruction = stream.attrs.get("bddl_language_instruction")
                if bddl_instruction is not None and not _attribute_text(
                    bddl_instruction
                ).strip():
                    raise ValueError(f"{path} BDDL language provenance is empty")
            raw_trajectory_id = stream.attrs.get("source_trajectory_id", "")
            if isinstance(raw_trajectory_id, bytes):
                raw_trajectory_id = raw_trajectory_id.decode("utf-8")
            trajectory_id = str(raw_trajectory_id).strip()
            action_dataset = stream.get("action")
            if not isinstance(action_dataset, h5py.Dataset):
                raise ValueError(f"{path} lacks an action dataset")
            if is_libero:
                if action_dataset.ndim != 2 or tuple(action_dataset.shape[1:]) != (7,):
                    raise ValueError(f"{path} LIBERO action must be [T,7]")
                _validate_rgb_datasets(
                    stream,
                    length=int(action_dataset.shape[0]),
                    path=path,
                )
                if converter_schema == LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA:
                    raw_terminal = stream.attrs.get("terminal_state_index")
                    if raw_terminal is None:
                        raise ValueError(f"{path} has no terminal_state_index")
                    terminal = int(raw_terminal)
                    for key in (
                        "observations/images/cam_high",
                        "observations/images/cam_right_wrist",
                    ):
                        rgb = stream[key]
                        terminal_rows = np.asarray(rgb[terminal:])
                        expected_rows = np.repeat(
                            terminal_rows[:1], len(terminal_rows), axis=0
                        )
                        if not np.array_equal(terminal_rows, expected_rows):
                            raise ValueError(
                                f"{path} terminal RGB dataset {key!r} is not repeated"
                            )
        episode = load_episode(
            path,
            cameras=("top", "wrist"),
            action_key="action",
            action_state_key="action_state",
            state_key="state",
            camera_key_overrides={
                "top": "observations/images/cam_high",
                "wrist": "observations/images/cam_right_wrist",
            },
        )
        if episode.language_key not in instructions:
            raise ValueError(f"{path} instruction is absent from inventory")
        if episode.instruction != instructions[episode.language_key]:
            raise ValueError(f"{path} instruction text differs from inventory")
        if is_libero:
            if (
                recorded_evaluator_instruction is not None
                and recorded_evaluator_instruction != episode.instruction
            ):
                raise ValueError(
                    f"{path} evaluator instruction differs from episode text"
                )
        if not np.array_equal(
            episode.action_states_raw[0],
            np.zeros_like(episode.action_states_raw[0]),
        ):
            raise ValueError(f"{path} action_state[0] is not the reset zero action")
        if not np.array_equal(
            episode.action_states_raw[1:],
            episode.actions_raw[:-1],
        ):
            raise ValueError(
                f"{path} action_state[t] is not the previous executed action"
            )
        if episode.valid_center_start is None or episode.valid_center_end is None:
            raise ValueError(f"{path} has no explicit valid-center contract")
        if is_libero:
            if episode.length < 73:
                raise ValueError(f"{path} LIBERO episode is shorter than 73 rows")
            if (
                np.min(episode.actions_raw) < -1.00001
                or np.max(episode.actions_raw) > 1.00001
            ):
                raise ValueError(f"{path} LIBERO actions are outside normalized [-1,1]")
            if episode.states_raw is None or episode.states_raw.shape[1] != 7:
                raise ValueError(f"{path} LIBERO state must be [T,7]")
            if np.min(episode.states_raw[:, -1]) < -1e-6:
                raise ValueError(f"{path} LIBERO gripper opening width is negative")
            if converter_schema == LIBERO_CONVERTER_SCHEMA:
                if episode.valid_center_start != 24:
                    raise ValueError(f"{path} LIBERO valid_center_start must be 24")
                if episode.valid_center_end != episode.length - 49:
                    raise ValueError(
                        f"{path} LIBERO valid_center_end must equal length-49"
                    )
                if (
                    episode.strict_valid_center_start is not None
                    or episode.state_normalizer_reference_raw is not None
                ):
                    raise ValueError(f"{path} legacy LIBERO v1 has causal-v2 metadata")
            elif converter_schema == LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA:
                if (
                    episode.valid_center_start != 0
                    or episode.valid_center_end != episode.length - 49
                    or episode.strict_valid_center_start != 24
                    or episode.strict_valid_center_end != episode.length - 49
                ):
                    raise ValueError(f"{path} causal LIBERO prefix bounds differ")
                reference = episode.state_normalizer_reference_raw
                if reference is None or not np.array_equal(
                    episode.states_raw[1:], reference[:-1]
                ):
                    raise ValueError(
                        f"{path} causal state rows are not the shifted legacy rows"
                    )
                if episode.terminal_state_index is not None:
                    raise ValueError(f"{path} causal prefix unexpectedly has a suffix")
            else:
                terminal = episode.terminal_state_index
                source_count = episode.source_action_count
                if terminal is None or source_count is None or terminal != source_count:
                    raise ValueError(f"{path} terminal replay source indices differ")
                if (
                    episode.terminal_padding_mode
                    != LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
                    or episode.length != source_count + 48
                    or episode.valid_center_start != 0
                    or episode.valid_center_end != source_count - 1
                    or episode.strict_valid_center_start != 24
                    or episode.strict_valid_center_end != source_count - 49
                ):
                    raise ValueError(f"{path} terminal replay boundary contract differs")
                reference = episode.state_normalizer_reference_raw
                if reference is None or not np.array_equal(
                    episode.states_raw[1:source_count], reference[:-1]
                ):
                    raise ValueError(
                        f"{path} terminal replay real state rows are not causally shifted"
                    )
                if not np.array_equal(episode.states_raw[source_count], reference[-1]):
                    raise ValueError(
                        f"{path} recorded terminal state differs from the legacy final row"
                    )
                terminal_actions = episode.actions_raw[source_count:]
                if not np.array_equal(
                    terminal_actions[:, :6], np.zeros_like(terminal_actions[:, :6])
                ):
                    raise ValueError(f"{path} absorbing arm commands are not zero")
                if not np.array_equal(
                    terminal_actions[:, 6],
                    np.full_like(
                        terminal_actions[:, 6],
                        episode.actions_raw[source_count - 1, 6],
                    ),
                ):
                    raise ValueError(f"{path} absorbing gripper command is not held")
                if not np.array_equal(
                    episode.states_raw[source_count:],
                    np.repeat(
                        episode.states_raw[source_count : source_count + 1],
                        episode.length - source_count,
                        axis=0,
                    ),
                ):
                    raise ValueError(f"{path} terminal replay state is not repeated")
        if is_calvin:
            if not trajectory_id:
                raise ValueError(f"{path} has no source trajectory identity")
            trajectory_splits[name_to_split[name]].add(trajectory_id)
            terminal = episode.terminal_state_index
            if (
                terminal is None
                or episode.terminal_padding_mode
                != RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
            ):
                raise ValueError(f"{path} has no CALVIN absorbing terminal contract")
            if episode.valid_center_end != terminal - 24:
                raise ValueError(
                    f"{path} does not cover every real terminal action in a 24-row target"
                )
            if episode.length - terminal - 1 != 24:
                raise ValueError(f"{path} has the wrong CALVIN absorbing suffix length")
            terminal_actions = episode.actions_raw[terminal:]
            if not np.array_equal(
                terminal_actions[:, :6], np.zeros_like(terminal_actions[:, :6])
            ):
                raise ValueError(f"{path} absorbing arm commands are not zero")
            if not np.array_equal(
                terminal_actions[:, 6],
                np.full_like(terminal_actions[:, 6], episode.actions_raw[terminal - 1, 6]),
            ):
                raise ValueError(f"{path} absorbing gripper command is not held")
            if not np.array_equal(
                episode.states_raw[terminal:],
                np.repeat(
                    episode.states_raw[terminal : terminal + 1],
                    episode.length - terminal,
                    axis=0,
                ),
            ):
                raise ValueError(f"{path} absorbing terminal state is not repeated")
        steps += episode.length
        strict_first = max(
            24,
            episode.valid_center_start,
            24
            if episode.strict_valid_center_start is None
            else episode.strict_valid_center_start,
        )
        strict_last = min(
            episode.length - 49,
            episode.valid_center_end
            if episode.strict_valid_center_end is None
            else episode.strict_valid_center_end,
        )
        if strict_first > strict_last:
            raise ValueError(f"{path} has no causal -24...+48 training window")
        strict_valid_windows += strict_last - strict_first + 1
        if is_libero and converter_schema != LIBERO_CONVERTER_SCHEMA:
            first_center = episode.valid_center_start
            last_center = episode.valid_center_end
        else:
            first_center, last_center = strict_first, strict_last
        valid_windows += last_center - first_center + 1
        keys.add(str(episode.language_key))
    if is_calvin and any(
        trajectory_splits[left].intersection(trajectory_splits[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("CALVIN source trajectory appears in more than one split")
    return {
        "schema": BENCHMARK_DATASET_SCHEMA,
        "benchmark": benchmark,
        "root": str(source.resolve()),
        "episodes": len(expected_names),
        "steps": steps,
        "valid_windows": valid_windows,
        "strict_valid_windows": strict_valid_windows,
        "instructions": len(instructions),
        "episode_instructions": len(keys),
        "splits": {
            name: len(split_values[name]) for name in ("train", "val", "test")
        },
        "manifest_sha256": _sha256(manifest_path),
        "splits_sha256": _sha256(split_path),
        "instructions_sha256": _sha256(instruction_path),
    }


__all__ = [
    "BENCHMARK_DATASET_SCHEMA",
    "BENCHMARK_EPISODE_SCHEMA",
    "LIBERO_CAUSAL_ALIGNED_CONVERTER_SCHEMA",
    "LIBERO_CONVERTER_SCHEMA",
    "LIBERO_CONVERTER_SCHEMAS",
    "LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA",
    "atomic_hdf5",
    "atomic_json",
    "audit_benchmark_dataset",
    "write_instruction_inventory",
    "write_split_manifest",
]
