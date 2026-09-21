"""Convert official CALVIN language slices into ClearVLA HDF5 episodes."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clearvla.data.hdf5_episode import (
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
)
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

CALVIN_CONVERTER_SCHEMA = "clearvla-calvin-converter-v2"
CALVIN_HISTORY_HORIZON = 24
CALVIN_POLICY_HORIZON = 24
CALVIN_WORLD_HORIZON = 48
CALVIN_TERMINAL_PADDING_FRAMES = CALVIN_WORLD_HORIZON - CALVIN_POLICY_HORIZON


@dataclass(frozen=True)
class _SourceTrajectory:
    source_split: str
    index: int
    start: int
    end: int

    @property
    def identity(self) -> str:
        return f"{self.source_split}:{self.index:06d}:{self.start}-{self.end}"


@dataclass(frozen=True)
class _AnnotationRow:
    index: int
    start: int
    end: int
    instruction: str
    task: str
    trajectory: _SourceTrajectory


def _frame_pattern(root: Path) -> tuple[str, int, str]:
    candidates = sorted((*root.glob("*.npz"), *root.glob("*.pkl")))
    for path in candidates:
        match = re.match(r"^(.*?)(\d+)(\.(?:npz|pkl))$", path.name)
        if match:
            return match.group(1), len(match.group(2)), match.group(3)
    raise FileNotFoundError(f"no indexed CALVIN frame files found under {root}")


def _load_frame(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}
    import pickle

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"CALVIN frame {path} is not a mapping")
    return {str(name): np.asarray(value) for name, value in payload.items()}


def _source_trajectories(
    split_root: Path,
    *,
    source_split: str,
) -> list[_SourceTrajectory]:
    path = split_root / "ep_start_end_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN source trajectory inventory is absent: {path}")
    values = np.asarray(np.load(path), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError(f"CALVIN source trajectory inventory must be [N,2], got {values.shape}")
    trajectories = [
        _SourceTrajectory(
            source_split=str(source_split),
            index=int(index),
            start=int(start),
            end=int(end),
        )
        for index, (start, end) in enumerate(values.tolist())
    ]
    if any(row.start < 0 or row.end < row.start for row in trajectories):
        raise ValueError(f"CALVIN source trajectory inventory is invalid: {path}")
    ordered = sorted(trajectories, key=lambda row: (row.start, row.end, row.index))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start <= previous.end:
            raise ValueError(
                "CALVIN source trajectories overlap: "
                f"{previous.identity!r} and {current.identity!r}"
            )
    return ordered


def _annotation_rows(
    split_root: Path,
    *,
    source_split: str,
) -> list[_AnnotationRow]:
    path = split_root / "lang_annotations" / "auto_lang_ann.npy"
    if not path.is_file():
        path = split_root / "auto_lang_ann.npy"
    payload = np.load(path, allow_pickle=True).item()
    indices = payload["info"]["indx"]
    annotations = payload["language"]["ann"]
    tasks = payload["language"].get("task", ["unknown"] * len(indices))
    if not len(indices) == len(annotations) == len(tasks):
        raise ValueError("CALVIN language indices/annotations/tasks differ in length")
    trajectories = _source_trajectories(split_root, source_split=source_split)
    rows: list[_AnnotationRow] = []
    for index, ((start, end), instruction, task) in enumerate(
        zip(indices, annotations, tasks, strict=True)
    ):
        start = int(start)
        end = int(end)
        owners = [
            trajectory
            for trajectory in trajectories
            if trajectory.start <= start and end <= trajectory.end
        ]
        if len(owners) != 1:
            raise ValueError(
                f"CALVIN annotation {index} [{start},{end}] belongs to "
                f"{len(owners)} source trajectories"
            )
        rows.append(
            _AnnotationRow(
                index=int(index),
                start=start,
                end=end,
                instruction=normalize_instruction(str(instruction)),
                task=str(task),
                trajectory=owners[0],
            )
        )
    return rows


def _bounded_trajectory_rows(
    rows: list[_AnnotationRow],
    limit: int | None,
) -> list[_AnnotationRow]:
    """Keep a bounded smoke representative without collapsing to one trajectory."""

    if limit is None or len(rows) <= int(limit):
        return list(rows)
    if int(limit) <= 0:
        raise ValueError("limit_per_split must be positive")
    grouped: dict[str, list[_AnnotationRow]] = {}
    for row in rows:
        grouped.setdefault(row.trajectory.identity, []).append(row)
    selected: list[_AnnotationRow] = []
    positions = {name: 0 for name in grouped}
    while len(selected) < int(limit):
        advanced = False
        for name in sorted(grouped):
            position = positions[name]
            if position < len(grouped[name]):
                selected.append(grouped[name][position])
                positions[name] = position + 1
                advanced = True
                if len(selected) == int(limit):
                    break
        if not advanced:
            break
    return sorted(selected, key=lambda row: row.index)


def validation_instruction_inventory(path: Path) -> dict[str, str]:
    """Read CALVIN's official task-to-validation-expression YAML.

    The official file is intentionally simple (one ``task: [\"text\"]`` row
    per task).  Keeping this parser dependency-light lets dataset conversion
    run in the current ClearVLA environment without importing CALVIN's legacy
    Hydra stack.  It fails closed if the official file changes shape.
    """

    if not path.is_file():
        raise FileNotFoundError(f"CALVIN validation annotation file is absent: {path}")
    instructions: dict[str, str] = {}
    task_names: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(
                f"{path}:{line_number}: expected one task-to-expression row"
            )
        task, raw_values = line.split(":", 1)
        task = task.strip()
        if not task or task in task_names:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate task key")
        try:
            values = ast.literal_eval(raw_values.strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: validation expressions are not a literal list"
            ) from error
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(
                f"{path}:{line_number}: validation task must own a non-empty list"
            )
        task_names.add(task)
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    f"{path}:{line_number}: validation expression must be text"
                )
            instruction = normalize_instruction(value)
            instructions[instruction_key(instruction)] = instruction
    if not instructions:
        raise ValueError(f"CALVIN validation instruction inventory is empty: {path}")
    return instructions


def _write_episode(
    path: Path,
    *,
    frames: list[dict[str, np.ndarray]],
    instruction: str,
    task: str,
    source_split: str,
    source_start: int,
    source_end: int,
    source_annotation_index: int,
    source_trajectory: _SourceTrajectory,
    context_start: int,
    language_start_local: int,
    language_end_local: int,
) -> None:
    required = {"rgb_static", "rgb_gripper", "rel_actions", "robot_obs"}
    missing = sorted(required.difference(frames[0]))
    if missing:
        raise KeyError(f"CALVIN frame lacks required keys: {missing}")
    actions = np.stack([row["rel_actions"] for row in frames]).astype(np.float32)
    robot = np.stack([row["robot_obs"] for row in frames]).astype(np.float32)
    top = np.stack([row["rgb_static"] for row in frames]).astype(np.uint8)
    wrist = np.stack([row["rgb_gripper"] for row in frames]).astype(np.uint8)
    if actions.shape != (len(frames), 7) or robot.shape[1] < 7:
        raise ValueError("CALVIN action must be [T,7] and robot_obs at least [T,7]")
    terminal_state_index = int(language_end_local)
    if not (
        CALVIN_HISTORY_HORIZON <= int(language_start_local)
        <= terminal_state_index - CALVIN_POLICY_HORIZON
        < len(frames)
    ):
        raise ValueError(
            "CALVIN annotation cannot own complete causal history and policy target: "
            f"start={language_start_local} end={terminal_state_index} T={len(frames)}"
        )

    # ``auto_lang_ann`` ends at the terminal observation.  The last real
    # transition action is therefore at end-1.  Keep every such action inside
    # a 24-row policy target, then materialize exactly the official CALVIN
    # absorbing convention for the remaining 48-row Teacher horizon: repeat
    # terminal state/RGB, zero relative arm motion and hold the binary gripper.
    terminal_action = np.zeros((7,), dtype=np.float32)
    terminal_action[-1] = actions[terminal_state_index - 1, -1]
    actions[terminal_state_index] = terminal_action
    actions = np.concatenate(
        (
            actions,
            np.repeat(
                terminal_action[None], CALVIN_TERMINAL_PADDING_FRAMES, axis=0
            ),
        ),
        axis=0,
    )
    robot = np.concatenate(
        (
            robot,
            np.repeat(
                robot[terminal_state_index : terminal_state_index + 1],
                CALVIN_TERMINAL_PADDING_FRAMES,
                axis=0,
            ),
        ),
        axis=0,
    )
    top = np.concatenate(
        (
            top,
            np.repeat(
                top[terminal_state_index : terminal_state_index + 1],
                CALVIN_TERMINAL_PADDING_FRAMES,
                axis=0,
            ),
        ),
        axis=0,
    )
    wrist = np.concatenate(
        (
            wrist,
            np.repeat(
                wrist[terminal_state_index : terminal_state_index + 1],
                CALVIN_TERMINAL_PADDING_FRAMES,
                axis=0,
            ),
        ),
        axis=0,
    )
    state = robot[:, :7].copy()
    action_state = np.concatenate((np.zeros((1, 7), np.float32), actions[:-1]), axis=0)
    key = instruction_key(instruction)

    def writer(temporary: Path) -> None:
        with h5py.File(temporary, "w") as stream:
            stream.attrs["schema"] = BENCHMARK_EPISODE_SCHEMA
            stream.attrs["benchmark"] = "CALVIN"
            stream.attrs["instruction"] = instruction
            stream.attrs["language_key"] = key
            stream.attrs["task"] = task
            stream.attrs["source_split"] = source_split
            stream.attrs["source_start"] = int(source_start)
            stream.attrs["source_end"] = int(source_end)
            stream.attrs["source_annotation_index"] = int(source_annotation_index)
            stream.attrs["source_trajectory_id"] = source_trajectory.identity
            stream.attrs["source_trajectory_index"] = int(source_trajectory.index)
            stream.attrs["source_trajectory_start"] = int(source_trajectory.start)
            stream.attrs["source_trajectory_end"] = int(source_trajectory.end)
            stream.attrs["context_start"] = int(context_start)
            stream.attrs["valid_center_start"] = int(language_start_local)
            stream.attrs["valid_center_end"] = int(
                terminal_state_index - CALVIN_POLICY_HORIZON
            )
            stream.attrs["terminal_state_index"] = terminal_state_index
            stream.attrs["terminal_padding_mode"] = (
                RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
            )
            stream.attrs["terminal_padding_frames"] = CALVIN_TERMINAL_PADDING_FRAMES
            stream.attrs["converter_schema"] = CALVIN_CONVERTER_SCHEMA
            stream.attrs["action_semantics"] = (
                "CALVIN normalized relative world-frame TCP xyz/euler plus binary gripper"
            )
            stream.attrs["state_semantics"] = (
                "TCP xyz/world-frame euler/gripper opening from robot_obs[:7]"
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


def convert_calvin(
    source: Path,
    output: Path,
    *,
    train_source_split: str = "training",
    heldout_source_split: str = "validation",
    val_fraction: float = 0.1,
    split_seed: int = 0,
    task_filter: str | None = None,
    limit_per_split: int | None = None,
    validation_annotations: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing a non-empty CALVIN conversion root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    instructions: dict[str, str] = {}
    source_rows: dict[str, int] = {}
    source_trajectory_counts: dict[str, int] = {}
    output_trajectory_ids: dict[str, set[str]] = {
        name: set() for name in ("train", "val", "test")
    }
    for source_split in (train_source_split, heldout_source_split):
        split_root = source / source_split
        prefix, digits, suffix = _frame_pattern(split_root)
        rows = _annotation_rows(split_root, source_split=source_split)
        eligible = []
        for row in rows:
            if task_filter is not None and row.task != str(task_filter):
                continue
            context_start = max(
                int(row.trajectory.start),
                int(row.start) - CALVIN_HISTORY_HORIZON,
            )
            language_start_local = int(row.start) - context_start
            language_end_local = int(row.end) - context_start
            # A legal centre must own both sides of the fixed causal window.
            # The previous predicate only enforced the 24-row policy suffix:
            # an annotation beginning at local frame 18 could pass because
            # its terminal was far enough away, then fail later in
            # ``_write_episode`` when the required -24 history was absent.
            # Filter that row here so conversion is deterministic and never
            # partially writes a dataset before discovering the boundary.
            if (
                language_start_local >= CALVIN_HISTORY_HORIZON
                and language_start_local
                <= language_end_local - CALVIN_POLICY_HORIZON
            ):
                eligible.append(row)
        eligible = _bounded_trajectory_rows(eligible, limit_per_split)
        source_rows[source_split] = len(eligible)
        eligible_trajectory_ids = sorted(
            {row.trajectory.identity for row in eligible}
        )
        source_trajectory_counts[source_split] = len(eligible_trajectory_ids)
        if source_split == train_source_split:
            if len(eligible_trajectory_ids) < 2:
                raise ValueError(
                    "CALVIN training source needs at least two eligible source "
                    "trajectories for trajectory-disjoint train/val"
                )
            generator = np.random.default_rng(int(split_seed))
            shuffled = list(
                np.asarray(eligible_trajectory_ids, dtype=object)[
                    generator.permutation(len(eligible_trajectory_ids))
                ].tolist()
            )
            val_count = int(round(len(shuffled) * float(val_fraction)))
            val_count = min(max(1, val_count), len(shuffled) - 1)
            validation_trajectories = set(shuffled[:val_count])
        else:
            validation_trajectories = set()

        for row in eligible:
            context_start = max(
                int(row.trajectory.start),
                int(row.start) - CALVIN_HISTORY_HORIZON,
            )
            frames = [
                _load_frame(split_root / f"{prefix}{frame:0{digits}d}{suffix}")
                for frame in range(context_start, int(row.end) + 1)
            ]
            name = f"calvin_{source_split}_{row.index:06d}"
            _write_episode(
                output / f"{name}.hdf5",
                frames=frames,
                instruction=row.instruction,
                task=row.task,
                source_split=source_split,
                source_start=row.start,
                source_end=row.end,
                source_annotation_index=row.index,
                source_trajectory=row.trajectory,
                context_start=context_start,
                language_start_local=row.start - context_start,
                language_end_local=row.end - context_start,
            )
            key = instruction_key(row.instruction)
            instructions[key] = row.instruction
            if source_split == heldout_source_split:
                destination = "test"
            elif row.trajectory.identity in validation_trajectories:
                destination = "val"
            else:
                destination = "train"
            splits[destination].append(name)
            output_trajectory_ids[destination].add(row.trajectory.identity)
    evaluator_instructions: dict[str, str] = {}
    if validation_annotations is not None:
        evaluator_instructions = validation_instruction_inventory(
            validation_annotations
        )
        instructions.update(evaluator_instructions)
    if any(not splits[name] for name in splits):
        raise ValueError("CALVIN conversion needs non-empty train/val/test splits")
    if any(
        output_trajectory_ids[left].intersection(output_trajectory_ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise AssertionError("CALVIN source trajectory appears in more than one split")
    manifest = {
        "schema": BENCHMARK_DATASET_SCHEMA,
        "converter_schema": CALVIN_CONVERTER_SCHEMA,
        "benchmark": "CALVIN",
        "protocol": "ABC-to-D" if "ABC" in source.name else source.name,
        "source": str(source.resolve()),
        "source_rows": source_rows,
        "source_trajectory_counts": source_trajectory_counts,
        "output_trajectory_counts": {
            name: len(output_trajectory_ids[name])
            for name in ("train", "val", "test")
        },
        "split_unit": "source-trajectory",
        "split_seed": int(split_seed),
        "train_validation_fraction": float(val_fraction),
        "task_filter": None if task_filter is None else str(task_filter),
        "control_hz": 30,
        "cameras": ["rgb_static", "rgb_gripper"],
        "action_key": "rel_actions",
        "state_key": "robot_obs[:7]",
        "language_source": "lang_annotations/auto_lang_ann.npy",
        "evaluator_language_source": (
            None
            if validation_annotations is None
            else str(validation_annotations.resolve())
        ),
        "evaluator_instruction_count": len(evaluator_instructions),
        "instruction_inventory_count": len(instructions),
        "valid_center_semantics": (
            "all -24 history remains inside the source trajectory; every real action "
            "through annotated terminal-1 enters a 24-row policy target; the residual "
            "+48 Teacher suffix uses CALVIN absorbing terminal padding"
        ),
        "terminal_padding_mode": RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
        "terminal_padding_frames": CALVIN_TERMINAL_PADDING_FRAMES,
    }
    atomic_json(output / "dataset_manifest.json", manifest)
    write_split_manifest(
        output,
        splits,
        metadata={
            "task_filter": "" if task_filter is None else str(task_filter),
            "split_unit": "source-trajectory",
            "split_seed": int(split_seed),
            "train_validation_fraction": float(val_fraction),
            "trajectory_counts": {
                name: len(output_trajectory_ids[name])
                for name in ("train", "val", "test")
            },
        },
    )
    write_instruction_inventory(output, instructions)
    return audit_benchmark_dataset(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-source-split", default="training")
    parser.add_argument("--heldout-source-split", default="validation")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of eligible training-source trajectories reserved for validation",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--task-filter",
        default=None,
        help="convert only one CALVIN task (for example open_drawer)",
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--validation-annotations",
        type=Path,
        default=None,
        help=(
            "Official calvin_models/conf/annotations/new_playtable_validation.yaml; "
            "its rollout expressions are added to the same content-addressed bank"
        ),
    )
    args = parser.parse_args()
    result = convert_calvin(
        args.source,
        args.output,
        train_source_split=args.train_source_split,
        heldout_source_split=args.heldout_source_split,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        task_filter=args.task_filter,
        limit_per_split=args.limit_per_split,
        validation_annotations=args.validation_annotations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
