"""Convert official LIBERO demonstrations into isolated ClearVLA episodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clearvla.data.instructions import instruction_key, normalize_instruction

from .common import (
    BENCHMARK_DATASET_SCHEMA,
    BENCHMARK_EPISODE_SCHEMA,
    atomic_hdf5,
    atomic_json,
    audit_benchmark_dataset,
    write_instruction_inventory,
    write_split_manifest,
)

LIBERO_CONVERTER_SCHEMA = "clearvla-libero-converter-v1"
LIBERO_ACTION_DIM = 7
LIBERO_STATE_DIM = 7
LIBERO_MIN_EPISODE_LENGTH = 73
LIBERO_DATASET_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)


@dataclass(frozen=True)
class _BDDLTaskRecord:
    """The two independent language surfaces in one official task file.

    LIBERO's evaluator sends ``Task.language`` to the policy.  The benchmark
    constructs that value from the task filename, while the BDDL parser keeps
    the ``:language`` declaration for simulator metadata.  They are related
    provenance, but are not interchangeable instruction identities.
    """

    suite: str
    task_name: str
    path: Path
    evaluator_instruction: str
    bddl_instruction: str | None


def _attribute_text(value: object, *, name: str) -> str:
    """Decode one scalar HDF5 attribute without turning missing values into text.

    ``str(None)`` is a particularly dangerous fallback here: it produces a
    seemingly valid instruction named ``"None"`` and lets a bad source reach
    the language bank.  Keep the attribute boundary strict and report the
    source problem before any episode is written.
    """

    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"LIBERO {name} must be a scalar attribute")
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"LIBERO {name} is not valid UTF-8") from error
    if not isinstance(value, str):
        raise ValueError(f"LIBERO {name} must be scalar UTF-8 text")
    return value


def _dataset(group: h5py.Group, key: str, *, name: str) -> np.ndarray:
    value = group.get(key)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"LIBERO HDF5 group lacks dataset {name}={key!r}")
    return np.asarray(value)


def _finite_float_dataset(
    group: h5py.Group,
    key: str,
    *,
    shape_tail: tuple[int, ...],
    name: str,
) -> np.ndarray:
    value = _dataset(group, key, name=name)
    if value.ndim != 1 + len(shape_tail) or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(
            f"LIBERO {name} must have shape [T,{','.join(str(v) for v in shape_tail)}], "
            f"got {value.shape}"
        )
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"LIBERO {name} must be numeric") from error
    if not np.isfinite(result).all():
        raise ValueError(f"LIBERO {name} contains non-finite values")
    return result


def _rgb_dataset(group: h5py.Group, key: str, *, name: str) -> np.ndarray:
    value = _dataset(group, key, name=name)
    if (
        value.ndim != 4
        or value.shape[-1] != 3
        or min(int(value.shape[1]), int(value.shape[2])) <= 0
    ):
        raise ValueError(f"LIBERO {name} must have shape [T,H,W,3], got {value.shape}")
    if value.dtype != np.dtype(np.uint8):
        raise ValueError(f"LIBERO {name} must be uint8, got {value.dtype}")
    return np.asarray(value)


def _natural(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _instruction(data: h5py.Group) -> str:
    raw = data.attrs.get("problem_info")
    if raw is None:
        raise ValueError("LIBERO HDF5 data group has no problem_info")
    try:
        problem = json.loads(_attribute_text(raw, name="problem_info"))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("LIBERO HDF5 problem_info is not valid JSON") from error
    if not isinstance(problem, dict):
        raise ValueError("LIBERO HDF5 problem_info must be a JSON object")
    value = problem.get("language_instruction")
    if isinstance(value, list):
        if not value or any(not isinstance(row, str) for row in value):
            raise ValueError(
                "LIBERO problem_info.language_instruction must be non-empty text"
            )
        value = "".join(value)
    if not isinstance(value, str):
        raise ValueError(
            "LIBERO problem_info.language_instruction must be non-empty text"
        )
    return normalize_instruction(value)


def _task_language_from_filename(path: Path | str) -> str:
    """Reproduce LIBERO's ``grab_language_from_filename`` exactly at the edge.

    The official benchmark does *not* use the BDDL ``:language`` value for
    ``Task.language``.  Uppercase LIBERO-100 names carry a scene prefix that
    is removed; the lowercase LIBERO-90 names are already task slugs.  Keep
    this conversion local to the evaluator boundary instead of making the
    shared language bank learn two unrelated identities for one task.
    """

    name = Path(path).name
    if not name:
        raise ValueError("LIBERO BDDL filename must be non-empty")
    stem = name[:-5] if name.lower().endswith(".bddl") else name
    if not stem:
        raise ValueError(f"LIBERO BDDL filename has no task stem: {name!r}")
    if stem[0].isupper():
        # This is the safe equivalent of the official SCENE1/SCENE10 slices.
        # It also handles future scene numbers without an off-by-one branch.
        scene = re.search(r"SCENE\d+_", stem)
        if scene is not None:
            stem = stem[scene.end() :]
    return normalize_instruction(stem.replace("_", " "))


def _read_bddl_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"LIBERO BDDL file is not readable UTF-8: {path}") from error


def _bddl_language_declaration(path: Path) -> str | None:
    """Parse the simulator's ``:language`` group without requiring bddl.

    Official files use an unquoted Lisp-style group such as
    ``(:language Pick the black bowl ...)``.  Some older exports used a
    quoted scalar.  This small scanner handles both forms, ignores semicolon
    comments and rejects a declaration that is truncated or contains nested
    groups.  It deliberately returns ``None`` when the declaration is absent:
    old BDDL files can still be evaluated from their task filename.
    """

    text = _read_bddl_text(path)

    # Locate the marker outside strings/comments.  A regex alone would accept
    # a commented-out declaration as if it were simulator metadata.
    marker_start: int | None = None
    index = 0
    in_string: str | None = None
    escaped = False
    in_comment = False
    while index < len(text):
        character = text[index]
        if in_comment:
            if character == "\n":
                in_comment = False
            index += 1
            continue
        if in_string is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == in_string:
                in_string = None
            index += 1
            continue
        if character == ";":
            in_comment = True
            index += 1
            continue
        if character in {'"', "'"}:
            in_string = character
            index += 1
            continue
        if character == "(":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            token_start = cursor
            while cursor < len(text) and not text[cursor].isspace() and text[cursor] not in "()":
                cursor += 1
            if text[token_start:cursor].lower() == ":language":
                marker_start = cursor
                break
        index += 1
    if marker_start is None:
        return None

    cursor = marker_start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text):
        raise ValueError(f"LIBERO BDDL :language declaration is malformed: {path}")

    quote = text[cursor] if text[cursor] in {'"', "'"} else None
    if quote is not None:
        cursor += 1
        value_start = cursor
        escaped = False
        while cursor < len(text):
            character = text[cursor]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                raw = text[value_start:cursor]
                cursor += 1
                # Permit the same semicolon comments that the unquoted
                # official form permits between the scalar and its closing
                # group delimiter.
                while cursor < len(text):
                    if text[cursor].isspace():
                        cursor += 1
                        continue
                    if text[cursor] == ";":
                        newline = text.find("\n", cursor + 1)
                        cursor = len(text) if newline < 0 else newline + 1
                        continue
                    break
                if cursor >= len(text) or text[cursor] != ")":
                    raise ValueError(
                        f"LIBERO BDDL :language declaration is malformed: {path}"
                    )
                if quote == '"':
                    try:
                        value = json.loads(f'"{raw}"')
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"LIBERO BDDL :language string is malformed: {path}"
                        ) from error
                else:
                    value = raw.replace("\\'", "'").replace("\\\\", "\\")
                return normalize_instruction(value)
            cursor += 1
        raise ValueError(f"LIBERO BDDL :language declaration is malformed: {path}")

    # Unquoted official form: consume the rest of this group.  Parentheses
    # would indicate a nested expression rather than language words.  Build
    # the value while scanning instead of slicing the original text at the
    # end: a semicolon comment can occur between two language lines, and a
    # final ``split(';', 1)`` would otherwise discard every word after it.
    value: list[str] = []
    in_comment = False
    while cursor < len(text):
        character = text[cursor]
        if in_comment:
            if character in "\r\n":
                in_comment = False
                # Preserve a token boundary even when the comment marker was
                # attached directly to the preceding word.
                value.append(" ")
            cursor += 1
            continue
        if character == ";":
            in_comment = True
            cursor += 1
            continue
        if character == "(":
            raise ValueError(f"LIBERO BDDL :language declaration is malformed: {path}")
        if character == ")":
            return normalize_instruction("".join(value))
        value.append(character)
        cursor += 1
    raise ValueError(f"LIBERO BDDL :language declaration is malformed: {path}")


def _bddl_instruction(path: Path) -> str:
    """Return the BDDL declaration, with the filename as an old-file fallback.

    This helper intentionally represents simulator provenance, *not* the
    canonical evaluator instruction.  Call :func:`_task_language_from_filename`
    for the latter.  Keeping the legacy fallback preserves support for old
    BDDL exports and makes the distinction explicit at call sites.
    """

    declared = _bddl_language_declaration(path)
    return _task_language_from_filename(path) if declared is None else declared


def _bddl_task_records(
    bddl_root: Path,
    *,
    suites: tuple[str, ...],
) -> list[_BDDLTaskRecord]:
    if (
        not suites
        or any(not isinstance(value, str) or not value.strip() for value in suites)
        or len({value.lower() for value in suites}) != len(suites)
    ):
        raise ValueError("LIBERO BDDL suites must be non-empty and unique")
    records: list[_BDDLTaskRecord] = []
    seen: set[tuple[str, str]] = set()
    for suite in suites:
        root = bddl_root / suite
        if not root.is_dir():
            raise FileNotFoundError(f"LIBERO BDDL suite directory does not exist: {root}")
        # ``Path.glob`` is case-sensitive on the Linux evaluator host but the
        # official archives have appeared with both ``.bddl`` and ``.BDDL``
        # suffixes.  Match the extension explicitly while retaining a stable
        # natural filename order.
        paths = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_file() and path.suffix.lower() == ".bddl"
            ),
            key=lambda item: _natural(item.name),
        )
        if not paths:
            raise ValueError(f"LIBERO BDDL suite has no .bddl files: {root}")
        for path in paths:
            task_name = path.stem
            identity = (suite.lower(), task_name.lower())
            if identity in seen:
                raise ValueError(f"duplicate LIBERO BDDL task name: {suite}/{task_name}")
            seen.add(identity)
            records.append(
                _BDDLTaskRecord(
                    suite=suite,
                    task_name=task_name,
                    path=path,
                    evaluator_instruction=_task_language_from_filename(path),
                    bddl_instruction=_bddl_language_declaration(path),
                )
            )
    if not records:
        raise ValueError(f"no LIBERO BDDL tasks found under {bddl_root}")
    return records


def evaluator_instruction_inventory(
    bddl_root: Path,
    *,
    suites: tuple[str, ...],
) -> dict[str, str]:
    """Return canonical ``Task.language`` text keyed by instruction identity.

    The BDDL ``:language`` declaration is intentionally not used here: it is
    simulator metadata and can differ in wording from the text sent by the
    official evaluator.  Use :func:`_bddl_instruction` when that declaration
    itself is needed for provenance/audit.
    """

    result: dict[str, str] = {}
    for record in _bddl_task_records(Path(bddl_root), suites=suites):
        instruction = record.evaluator_instruction
        key = instruction_key(instruction)
        previous = result.get(key)
        if previous is not None and previous != instruction:
            raise ValueError(
                "LIBERO evaluator instruction key resolves to conflicting text: "
                f"{previous!r} versus {instruction!r}"
            )
        result[key] = instruction
    if not result:
        raise ValueError(f"no LIBERO evaluator instructions found under {bddl_root}")
    return result


def bddl_instruction_inventory(
    bddl_root: Path,
    *,
    suites: tuple[str, ...],
) -> dict[str, str]:
    """Compatibility alias for :func:`evaluator_instruction_inventory`.

    The old name was conceptually overloaded: this inventory contains the
    filename-derived evaluator language, not the simulator's BDDL wording.
    Keep the alias for callers that imported the initial boundary module, but
    make the ownership explicit for all new call sites.
    """

    return evaluator_instruction_inventory(bddl_root, suites=suites)


def _opening_width(gripper: np.ndarray) -> np.ndarray:
    if gripper.ndim != 2 or gripper.shape[1] != 2:
        raise ValueError("LIBERO gripper_states must be [T,2]")
    return np.abs(gripper).sum(axis=1, keepdims=True).astype(np.float32)


def _write_episode(
    path: Path,
    *,
    group: h5py.Group,
    instruction: str,
    suite: str,
    task_file: str,
    demo: str,
    evaluator_task_name: str | None = None,
    bddl_instruction: str | None = None,
) -> None:
    actions = _finite_float_dataset(
        group,
        "actions",
        shape_tail=(LIBERO_ACTION_DIM,),
        name="actions",
    )
    obs = group.get("obs")
    if not isinstance(obs, h5py.Group):
        raise ValueError("LIBERO HDF5 demo lacks an obs group")
    top = _rgb_dataset(obs, "agentview_rgb", name="obs.agentview_rgb")
    wrist = _rgb_dataset(obs, "eye_in_hand_rgb", name="obs.eye_in_hand_rgb")
    ee = _finite_float_dataset(
        obs,
        "ee_states",
        shape_tail=(6,),
        name="obs.ee_states",
    )
    gripper = _finite_float_dataset(
        obs,
        "gripper_states",
        shape_tail=(2,),
        name="obs.gripper_states",
    )
    length = int(actions.shape[0])
    if (
        ee.shape[0] != length
        or top.shape[0] != length
        or wrist.shape[0] != length
        or gripper.shape[0] != length
    ):
        raise ValueError("LIBERO observations and actions lost timestep alignment")
    if length < LIBERO_MIN_EPISODE_LENGTH:
        raise ValueError(f"LIBERO trajectory T={length} cannot support -24...+48")
    if np.min(actions) < -1.00001 or np.max(actions) > 1.00001:
        raise ValueError("LIBERO action lies outside the official normalized [-1,1] chart")
    state = np.concatenate((ee, _opening_width(gripper)), axis=1).astype(np.float32)
    action_state = np.concatenate((np.zeros((1, 7), np.float32), actions[:-1]), axis=0)
    if action_state.shape != actions.shape or not np.array_equal(
        action_state[0], np.zeros((LIBERO_ACTION_DIM,), dtype=np.float32)
    ) or not np.array_equal(action_state[1:], actions[:-1]):
        raise AssertionError("LIBERO action_state must be zero then previous action")
    key = instruction_key(instruction)

    def writer(temporary: Path) -> None:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = BENCHMARK_EPISODE_SCHEMA
            stream.attrs["benchmark"] = "LIBERO"
            stream.attrs["instruction"] = instruction
            stream.attrs["language_key"] = key
            stream.attrs["suite"] = suite
            stream.attrs["task_file"] = task_file
            stream.attrs["source_demo"] = demo
            if evaluator_task_name is not None:
                stream.attrs["evaluator_task_name"] = evaluator_task_name
            if bddl_instruction is not None:
                stream.attrs["bddl_language_instruction"] = bddl_instruction
            # This is deliberately separate from the BDDL declaration above:
            # the official evaluator sends the filename-derived Task.language.
            stream.attrs["evaluator_instruction"] = instruction
            stream.attrs["converter_schema"] = LIBERO_CONVERTER_SCHEMA
            stream.attrs["data_profile"] = "libero_relative_7d_v1"
            stream.attrs["arm_flow_mode"] = "relative_command_adapter"
            stream.attrs["gripper_output_mode"] = "continuous"
            stream.attrs["controller"] = "OSC_POSE"
            stream.attrs["camera_names"] = np.asarray(
                ("agentview_rgb", "eye_in_hand_rgb"), dtype="S"
            )
            stream.attrs["valid_center_start"] = 24
            stream.attrs["valid_center_end"] = length - 49
            stream.attrs["action_state_semantics"] = (
                "previous executed native action; reset row is zero"
            )
            stream.attrs["action_semantics"] = (
                "official normalized OSC_POSE xyz/axis-angle delta plus gripper command"
            )
            stream.attrs["state_semantics"] = (
                "EEF xyz/axis-angle plus sum(abs(robot0_gripper_qpos)) opening width"
            )
            stream.create_dataset("action", data=actions)
            stream.create_dataset("action_state", data=action_state)
            stream.create_dataset("state", data=state)
            images = stream.require_group("observations/images")
            images.create_dataset(
                "cam_high", data=top, compression="gzip", compression_opts=4, shuffle=True
            )
            images.create_dataset(
                "cam_right_wrist",
                data=wrist,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

    atomic_hdf5(path, writer)


def _split_demo(index: int, count: int) -> str:
    if count < 3:
        raise ValueError("each LIBERO task needs at least three usable demos")
    val_count = max(1, int(round(count * 0.1)))
    test_count = max(1, int(round(count * 0.1)))
    if val_count + test_count >= count:
        val_count = test_count = 1
    if index >= count - test_count:
        return "test"
    if index >= count - test_count - val_count:
        return "val"
    return "train"


def _demo_length(data: h5py.Group, demo: str) -> int:
    value = data.get(demo)
    if not isinstance(value, h5py.Group):
        raise ValueError(f"LIBERO data/{demo} must be an HDF5 group")
    actions = value.get("actions")
    if not isinstance(actions, h5py.Dataset):
        raise ValueError(f"LIBERO data/{demo} has no actions dataset")
    if actions.ndim != 2 or tuple(actions.shape[1:]) != (LIBERO_ACTION_DIM,):
        raise ValueError(
            f"LIBERO data/{demo}/actions must be [T,7], got {actions.shape}"
        )
    return int(actions.shape[0])


def _source_task_name(path: Path) -> str:
    """Return the official task stem from ``<task>_demo.hdf5``."""

    stem = path.stem
    if stem.lower().endswith("_demo"):
        stem = stem[:-5]
    if not stem:
        raise ValueError(f"LIBERO source filename has no task name: {path.name}")
    return stem


def _index_bddl_tasks(
    records: list[_BDDLTaskRecord],
) -> tuple[
    dict[tuple[str, str], _BDDLTaskRecord],
    dict[tuple[str, str], list[_BDDLTaskRecord]],
]:
    """Index BDDL records by exact task identity and canonical language key."""

    by_task: dict[tuple[str, str], _BDDLTaskRecord] = {}
    by_instruction: dict[tuple[str, str], list[_BDDLTaskRecord]] = {}
    for record in records:
        task_key = (record.suite.lower(), record.task_name.lower())
        if task_key in by_task:
            raise ValueError(
                "duplicate LIBERO evaluator task identity: "
                f"{record.suite}/{record.task_name}"
            )
        by_task[task_key] = record
        instruction_key_value = instruction_key(record.evaluator_instruction)
        by_instruction.setdefault(
            (record.suite.lower(), instruction_key_value), []
        ).append(record)
    return by_task, by_instruction


def convert_libero(
    source: Path,
    output: Path,
    *,
    suites: tuple[str, ...],
    limit_tasks: int | None = None,
    limit_demos_per_task: int | None = None,
    evaluator_bddl_root: Path | None = None,
) -> dict[str, Any]:
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"LIBERO source directory does not exist: {source}")
    suites = tuple(
        value.strip() if isinstance(value, str) else "" for value in suites
    )
    if (
        not suites
        or len({value.lower() for value in suites}) != len(suites)
        or any(not value for value in suites)
    ):
        raise ValueError("LIBERO suite names must be non-empty")
    if "libero_100" in {value.lower() for value in suites}:
        raise ValueError(
            "LIBERO_100 is an archive union; convert libero_90 and libero_10 "
            "as explicit suite roots"
        )
    unsupported_suites = [value for value in suites if value not in LIBERO_DATASET_SUITES]
    if unsupported_suites:
        raise ValueError(
            "unsupported LIBERO suite roots "
            f"{unsupported_suites}; choices={list(LIBERO_DATASET_SUITES)}"
        )
    if evaluator_bddl_root is None:
        raise ValueError(
            "LIBERO conversion requires evaluator_bddl_root so policy language "
            "can be verified against filename-derived Task.language"
        )
    evaluator_bddl_root = Path(evaluator_bddl_root).expanduser()
    if not evaluator_bddl_root.is_dir():
        raise FileNotFoundError(
            f"LIBERO evaluator BDDL root does not exist: {evaluator_bddl_root}"
        )
    if limit_tasks is not None and (
        isinstance(limit_tasks, bool) or not isinstance(limit_tasks, int) or int(limit_tasks) <= 0
    ):
        raise ValueError("limit_tasks must be positive")
    if limit_demos_per_task is not None and (
        isinstance(limit_demos_per_task, bool)
        or not isinstance(limit_demos_per_task, int)
        or int(limit_demos_per_task) <= 0
    ):
        raise ValueError("limit_demos_per_task must be positive")
    # ``Path.exists()`` is false for a dangling link.  Check the link bit
    # independently so an interrupted/attacker-created link cannot be
    # atomically replaced by the conversion root below.
    if output.is_symlink():
        raise FileExistsError(
            f"refusing LIBERO conversion output that is a symlink: {output}"
        )
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"refusing LIBERO conversion output that is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(f"refusing a non-empty LIBERO conversion root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, Path]] = []
    for suite in suites:
        suite_root = source / suite
        if not suite_root.is_dir():
            raise FileNotFoundError(f"LIBERO suite directory does not exist: {suite_root}")
        files.extend(
            (suite, path)
            for path in sorted(
                (
                    candidate
                    for candidate in suite_root.iterdir()
                    if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}
                ),
                key=lambda item: _natural(item.name),
            )
        )
    files.sort(key=lambda row: (row[0], _natural(row[1].name)))
    if limit_tasks is not None:
        files = files[:limit_tasks]
    if not files:
        raise FileNotFoundError("no LIBERO task HDF5 files selected")
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    instructions: dict[str, str] = {}
    evaluator_records: list[_BDDLTaskRecord] = []
    evaluator_by_task: dict[tuple[str, str], _BDDLTaskRecord] = {}
    evaluator_by_instruction: dict[tuple[str, str], list[_BDDLTaskRecord]] = {}
    # Resolve evaluator language before writing any output episode.  The
    # canonical text is filename-derived Task.language; the BDDL declaration
    # is retained only as simulator provenance.
    evaluator_records = _bddl_task_records(
        evaluator_bddl_root,
        suites=suites,
    )
    evaluator_by_task, evaluator_by_instruction = _index_bddl_tasks(
        evaluator_records
    )
    evaluator_instruction_keys = {
        instruction_key(record.evaluator_instruction) for record in evaluator_records
    }
    evaluator_alias_count = sum(
        record.bddl_instruction is not None for record in evaluator_records
    )
    skipped_short: list[str] = []
    tasks: list[dict[str, Any]] = []
    emitted_episode_names: set[str] = set()
    staging_name = output.name or "libero"
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{staging_name}.staging-",
            dir=str(output.parent),
        )
    )
    committed = False
    try:
        for suite, source_file in files:
            with h5py.File(source_file, "r") as stream:
                data = stream.get("data")
                if not isinstance(data, h5py.Group):
                    raise ValueError(f"{source_file} has no HDF5 data group")
                instruction = _instruction(data)
                demos = sorted(data.keys(), key=_natural)
                if not demos:
                    raise ValueError(f"{source_file} has no demonstrations")
                usable = [
                    demo
                    for demo in demos
                    if _demo_length(data, demo) >= LIBERO_MIN_EPISODE_LENGTH
                ]
                skipped_short.extend(
                    f"{suite}/{source_file.name}:{demo}"
                    for demo in demos
                    if demo not in usable
                )
                if limit_demos_per_task is not None:
                    usable = usable[:limit_demos_per_task]
                if len(usable) < 3:
                    raise ValueError(
                        f"{source_file} has fewer than three usable demonstrations"
                    )
                task_name = _source_task_name(source_file)
                task_slug = task_name.lower()
                key = instruction_key(instruction)
                evaluator_record: _BDDLTaskRecord | None = None
                if evaluator_records:
                    evaluator_record = evaluator_by_task.get(
                        (suite.lower(), task_name.lower())
                    )
                    if evaluator_record is None:
                        # A few legacy mirrors renamed the HDF5 file while
                        # preserving its task text.  Permit that only when a
                        # single canonical BDDL task can identify it; never
                        # select arbitrarily among duplicate language rows.
                        candidates = evaluator_by_instruction.get(
                            (suite.lower(), key), []
                        )
                        if len(candidates) == 1:
                            evaluator_record = candidates[0]
                        elif not candidates:
                            raise ValueError(
                                "LIBERO demo instruction is absent from the "
                                "evaluator BDDL Task.language inventory: "
                                f"{suite}/{source_file.name}"
                            )
                        else:
                            raise ValueError(
                                "LIBERO source task cannot be uniquely matched to "
                                "the evaluator BDDL inventory: "
                                f"{suite}/{source_file.name}"
                            )
                    expected_key = instruction_key(
                        evaluator_record.evaluator_instruction
                    )
                    if key != expected_key:
                        raise ValueError(
                            "LIBERO HDF5 language differs from the official "
                            "evaluator Task.language (filename-derived): "
                            f"{suite}/{source_file.name}"
                        )
                task_rows = {"train": 0, "val": 0, "test": 0}
                for index, demo in enumerate(usable):
                    name = f"{suite}_{task_slug}_{demo}"
                    identity = name.casefold()
                    if identity in emitted_episode_names:
                        raise ValueError(
                            "LIBERO source tasks produce duplicate episode identity: "
                            f"{name}"
                        )
                    emitted_episode_names.add(identity)
                    _write_episode(
                        staging / f"{name}.hdf5",
                        group=data[demo],
                        instruction=instruction,
                        suite=suite,
                        task_file=source_file.name,
                        demo=demo,
                        evaluator_task_name=(
                            evaluator_record.task_name
                            if evaluator_record is not None
                            else None
                        ),
                        bddl_instruction=(
                            evaluator_record.bddl_instruction
                            if evaluator_record is not None
                            else None
                        ),
                    )
                    split = _split_demo(index, len(usable))
                    splits[split].append(name)
                    task_rows[split] += 1
                instructions[key] = instruction
                task_row: dict[str, Any] = {
                    "suite": suite,
                    "task_file": source_file.name,
                    "task_name": task_name,
                    "instruction_key": key,
                    "demos": task_rows,
                }
                if evaluator_record is not None:
                    task_row["evaluator_task_name"] = evaluator_record.task_name
                    task_row["evaluator_instruction"] = (
                        evaluator_record.evaluator_instruction
                    )
                    if evaluator_record.bddl_instruction is not None:
                        task_row["bddl_language_instruction"] = (
                            evaluator_record.bddl_instruction
                        )
                tasks.append(task_row)
        manifest = {
            "schema": BENCHMARK_DATASET_SCHEMA,
            "converter_schema": LIBERO_CONVERTER_SCHEMA,
            "benchmark": "LIBERO",
            # These are part of the converted-root boundary, not model
            # topology. Recording them here lets config construction reject a
            # silent fallback to the Pen action/state chart before any cache or
            # training work.
            "data_profile": "libero_relative_7d_v1",
            "arm_flow_mode": "relative_command_adapter",
            "gripper_output_mode": "continuous",
            "source": str(source.resolve()),
            "suites": list(suites),
            "split_unit": "episode",
            "control_hz": 20,
            "controller": "OSC_POSE",
            "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
            "valid_center_start": 24,
            "valid_center_end": "length - 49",
            "valid_center_semantics": (
                "all -24...-1 executed-action history rows and +48 future rows "
                "remain inside each converted demonstration"
            ),
            "action_state_semantics": (
                "action_state[0] is zero; action_state[t] is the previous "
                "executed native LIBERO action"
            ),
            "action_range": [-1.0, 1.0],
            "task_count": len(tasks),
            "tasks": tasks,
            "evaluator_language_verified": True,
            "evaluator_language_source": str(evaluator_bddl_root.resolve()),
            "evaluator_language_contract": (
                "official Task.language is filename-derived; BDDL :language is "
                "simulator provenance only"
            ),
            "evaluator_task_count": len(evaluator_records),
            "evaluator_instruction_count": len(evaluator_instruction_keys),
            "bddl_language_declaration_count": evaluator_alias_count,
            "instruction_inventory_count": len(instructions),
            "skipped_short": skipped_short,
            "internal_demo_split": (
                "deterministic per-task 80/10/10; official score is rollout-only"
            ),
        }
        atomic_json(staging / "dataset_manifest.json", manifest)
        write_split_manifest(staging, splits)
        # Only canonical HDF5/evaluator Task.language text enters the language
        # inventory.  BDDL wording is never silently substituted into policy
        # inputs.
        write_instruction_inventory(staging, instructions)
        report = audit_benchmark_dataset(staging)

        # Commit the complete, audited root in one filesystem operation.  An
        # existing empty directory is accepted for compatibility, but any
        # non-empty destination is still protected against overwrite.
        if output.is_symlink():
            raise FileExistsError(
                f"refusing LIBERO conversion output that is a symlink: {output}"
            )
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError(
                    f"refusing a non-empty LIBERO conversion root: {output}"
                )
            output.rmdir()
        os.replace(staging, output)
        committed = True
        report["root"] = str(output.resolve())
        return report
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=LIBERO_DATASET_SUITES[:3] + ("libero_10",),
        help=(
            "Official extracted suite directories. The libero_100 archive expands "
            "to libero_90 and libero_10; use libero_90 explicitly for pretraining."
        ),
    )
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--limit-demos-per-task", type=int, default=None)
    parser.add_argument(
        "--evaluator-bddl-root",
        type=Path,
        required=True,
        help="LIBERO checkout's libero/libero/bddl_files directory",
    )
    args = parser.parse_args()
    result = convert_libero(
        args.source,
        args.output,
        suites=tuple(args.suites),
        limit_tasks=args.limit_tasks,
        limit_demos_per_task=args.limit_demos_per_task,
        evaluator_bddl_root=args.evaluator_bddl_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LIBERO_ACTION_DIM",
    "LIBERO_CONVERTER_SCHEMA",
    "LIBERO_DATASET_SUITES",
    "LIBERO_MIN_EPISODE_LENGTH",
    "LIBERO_STATE_DIM",
    "bddl_instruction_inventory",
    "convert_libero",
    "evaluator_instruction_inventory",
]
