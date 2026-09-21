"""Direct, read-only access to the official CALVIN frame inventory.

The formal ClearVLA training path deliberately consumes the converted HDF5
ABI.  This module is the alternative boundary for experiments that need to
read the original CALVIN trajectory files without materialising another HDF5
root.  It mirrors the converter's causal contract (24 history rows, a 24-row
policy target and 48 rows of Teacher support) and performs the same
trajectory-disjoint split.

The reader never writes to the source tree.  Images and numeric fields are
loaded one frame at a time and kept in a bounded process-local LRU.  The
official ``auto_lang_ann.npy`` is an object array, so loading it necessarily
uses pickle; callers should point this reader only at a trusted CALVIN
download.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import h5py

from clearvla.data.hdf5_episode import (
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
    LoadedEpisode,
)
from clearvla.data.instructions import instruction_key, normalize_instruction
from clearvla.data.split import EPISODE_SPLIT_MANIFEST_SCHEMA

from .calvin import (
    CALVIN_HISTORY_HORIZON,
    CALVIN_POLICY_HORIZON,
    CALVIN_TERMINAL_PADDING_FRAMES,
    CALVIN_WORLD_HORIZON,
)
from .io import atomic_json

RAW_CALVIN_READER_SCHEMA = "clearvla-calvin-raw-reader-v1"
_REQUIRED_FRAME_KEYS = ("rel_actions", "robot_obs", "rgb_static", "rgb_gripper")


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow only NumPy's ndarray reconstruction primitives and builtins."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"CALVIN annotation uses a non-whitelisted global {module}.{name}"
            )
        return super().find_class(module, name)


def _load_object_npy(path: Path) -> object:
    """Read an object ``.npy`` with a restricted, NumPy-only unpickler."""

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NumPy header version {version} in {path}")
        if not dtype.hasobject:
            raise ValueError(f"expected an object-array CALVIN annotation file: {path}")
        value = _RestrictedUnpickler(stream).load()
    if not isinstance(value, np.ndarray) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"invalid object-array payload in CALVIN annotation file: {path}")
    return value.item() if value.shape == () else value


def _frame_pattern(root: Path) -> tuple[str, int, str]:
    candidates = sorted((*root.glob("*.npz"), *root.glob("*.pkl")))
    for path in candidates:
        match = re.match(r"^(.*?)(\d+)(\.(?:npz|pkl))$", path.name)
        if match:
            return match.group(1), len(match.group(2)), match.group(3)
    raise FileNotFoundError(f"no indexed CALVIN frame files found under {root}")


def _load_frame_mapping(path: Path) -> dict[str, np.ndarray]:
    """Load one trusted source frame and detach arrays from the file handle."""

    if path.suffix == ".npz":
        # CALVIN frame files contain numeric arrays; keep this path explicitly
        # pickle-free even though the language annotation file is an object
        # array (handled in ``_load_annotations`` below).
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]).copy() for name in payload.files}
    with path.open("rb") as stream:
        payload = _RestrictedUnpickler(stream).load()
    if not isinstance(payload, Mapping):
        raise ValueError(f"CALVIN frame {path} is not a mapping")
    return {str(name): np.asarray(value).copy() for name, value in payload.items()}


def _validate_frame(mapping: Mapping[str, np.ndarray], path: Path) -> "RawCalvinFrame":
    missing = [name for name in _REQUIRED_FRAME_KEYS if name not in mapping]
    if missing:
        raise KeyError(f"CALVIN frame {path} lacks required keys: {missing}")
    action = np.asarray(mapping["rel_actions"], dtype=np.float32)
    robot = np.asarray(mapping["robot_obs"], dtype=np.float32)
    static = np.asarray(mapping["rgb_static"], dtype=np.uint8)
    gripper = np.asarray(mapping["rgb_gripper"], dtype=np.uint8)
    if action.shape != (7,):
        raise ValueError(f"{path}: rel_actions must have shape [7], got {action.shape}")
    if robot.ndim != 1 or robot.shape[0] < 7:
        raise ValueError(f"{path}: robot_obs must be a vector with at least 7 values")
    for name, image in (("rgb_static", static), ("rgb_gripper", gripper)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{path}: {name} must be RGB HWC, got {image.shape}")
    if not np.isfinite(action).all() or not np.isfinite(robot).all():
        raise ValueError(f"{path}: action/state contains non-finite values")
    return RawCalvinFrame(
        rel_actions=np.ascontiguousarray(action),
        robot_obs=np.ascontiguousarray(robot),
        rgb_static=np.ascontiguousarray(static),
        rgb_gripper=np.ascontiguousarray(gripper),
    )


@dataclass(frozen=True)
class RawCalvinFrame:
    """The policy-visible subset of one original CALVIN frame."""

    rel_actions: np.ndarray
    robot_obs: np.ndarray
    rgb_static: np.ndarray
    rgb_gripper: np.ndarray


@dataclass(frozen=True)
class RawCalvinTrajectory:
    """One source trajectory from ``ep_start_end_ids.npy``."""

    source_split: str
    index: int
    start: int
    end: int
    frame_root: Path
    frame_prefix: str
    frame_digits: int
    frame_suffix: str

    @property
    def identity(self) -> str:
        return f"{self.source_split}:{self.index:06d}:{self.start}-{self.end}"

    @property
    def length(self) -> int:
        return int(self.end - self.start + 1)

    def frame_path(self, global_index: int) -> Path:
        index = int(global_index)
        if index < self.start or index > self.end:
            raise IndexError(
                f"frame {index} is outside trajectory [{self.start},{self.end}]"
            )
        path = self.frame_root / f"{self.frame_prefix}{index:0{self.frame_digits}d}{self.frame_suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"missing CALVIN source frame: {path}")
        return path


class RawCalvinFrameStore:
    """Bounded process-local frame cache used by direct readers."""

    def __init__(self, *, capacity: int = 256) -> None:
        if int(capacity) < 0:
            raise ValueError("raw frame cache capacity must be non-negative")
        self.capacity = int(capacity)
        self._frames: OrderedDict[tuple[str, int], RawCalvinFrame] = OrderedDict()

    def load(self, trajectory: RawCalvinTrajectory, global_index: int) -> RawCalvinFrame:
        path = trajectory.frame_path(int(global_index))
        key = (str(path.resolve()), int(global_index))
        if self.capacity > 0 and key in self._frames:
            frame = self._frames.pop(key)
            self._frames[key] = frame
            return frame
        frame = _validate_frame(_load_frame_mapping(path), path)
        if self.capacity > 0:
            self._frames[key] = frame
            while len(self._frames) > self.capacity:
                self._frames.popitem(last=False)
        return frame

    def clear(self) -> None:
        self._frames.clear()


@dataclass(frozen=True)
class RawCalvinAnnotation:
    index: int
    start: int
    end: int
    instruction: str
    task: str
    trajectory: RawCalvinTrajectory


@dataclass(frozen=True)
class RawCalvinEpisode:
    """A language-labelled segment exposed as a virtual converted episode.

    ``terminal_state_index`` is the first synthetic action row.  The source
    terminal frame itself remains the last real observation and is repeated
    for the 24-row absorbing suffix, exactly like the converter output.
    """

    annotation: RawCalvinAnnotation
    context_start: int
    language_start_local: int
    language_end_local: int
    frame_store: RawCalvinFrameStore

    @property
    def episode_id(self) -> str:
        return f"calvin_{self.annotation.trajectory.source_split}_{self.annotation.index:06d}"

    @property
    def instruction(self) -> str:
        return self.annotation.instruction

    @property
    def task(self) -> str:
        return self.annotation.task

    @property
    def source_trajectory_id(self) -> str:
        return self.annotation.trajectory.identity

    @property
    def terminal_state_index(self) -> int:
        return int(self.language_end_local)

    @property
    def terminal_padding_mode(self) -> str:
        return RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING

    @property
    def length(self) -> int:
        return self.terminal_state_index + CALVIN_TERMINAL_PADDING_FRAMES + 1

    @property
    def valid_center_start(self) -> int:
        return int(self.language_start_local)

    @property
    def valid_center_end(self) -> int:
        return self.terminal_state_index - CALVIN_POLICY_HORIZON

    def _source_frame(self, local_index: int) -> RawCalvinFrame:
        if not 0 <= int(local_index) <= self.terminal_state_index:
            raise IndexError(f"source-local frame {local_index} is outside the segment")
        global_index = self.context_start + int(local_index)
        return self.frame_store.load(self.annotation.trajectory, global_index)

    def frame(self, local_index: int) -> RawCalvinFrame:
        """Read one virtual-segment frame, including the absorbing suffix."""

        index = int(local_index)
        if index < 0 or index >= self.length:
            raise IndexError(f"virtual frame {index} is outside [0,{self.length})")
        source_index = min(index, self.terminal_state_index)
        source = self._source_frame(source_index)
        if index < self.terminal_state_index:
            return source

        # At the annotated terminal observation and throughout the synthetic
        # suffix, keep the terminal RGB/state and issue zero arm motion while
        # holding the last real binary gripper command.
        previous = self._source_frame(self.terminal_state_index - 1)
        action = np.zeros((7,), dtype=np.float32)
        action[-1] = previous.rel_actions[-1]
        return RawCalvinFrame(
            rel_actions=action,
            robot_obs=source.robot_obs.copy(),
            rgb_static=source.rgb_static.copy(),
            rgb_gripper=source.rgb_gripper.copy(),
        )

    def arrays(self, indices: Sequence[int] | np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Materialise selected virtual rows with converted-ABI field names."""

        if indices is None:
            rows = np.arange(self.length, dtype=np.int64)
        else:
            rows = np.asarray(indices, dtype=np.int64)
            if rows.ndim != 1 or len(rows) == 0:
                raise ValueError("indices must be a non-empty one-dimensional sequence")
            if int(rows.min()) < 0 or int(rows.max()) >= self.length:
                raise IndexError("virtual frame index is outside the episode")
        frames = [self.frame(int(index)) for index in rows.tolist()]
        action = np.stack([frame.rel_actions for frame in frames]).astype(np.float32)
        robot = np.stack([frame.robot_obs for frame in frames]).astype(np.float32)
        static = np.stack([frame.rgb_static for frame in frames]).astype(np.uint8)
        gripper = np.stack([frame.rgb_gripper for frame in frames]).astype(np.uint8)
        # ``action_state`` is the previous executed command, with a reset zero
        # row at the virtual segment origin.
        # ``indices`` is often a sparse history/support selection, so the
        # previous-command row must be resolved in virtual-episode coordinates
        # rather than by shifting the selected subset.
        action_state_rows = []
        for index in rows.tolist():
            if int(index) == 0:
                action_state_rows.append(np.zeros((7,), dtype=np.float32))
            else:
                action_state_rows.append(self.frame(int(index) - 1).rel_actions)
        action_state = np.stack(action_state_rows).astype(np.float32)
        return {
            "action": np.ascontiguousarray(action),
            "action_state": np.ascontiguousarray(action_state),
            "state": np.ascontiguousarray(robot[:, :7]),
            "robot_obs": np.ascontiguousarray(robot),
            "rgb_static": np.ascontiguousarray(static),
            "rgb_gripper": np.ascontiguousarray(gripper),
        }

    def valid_centers(self, *, stride: int = 1) -> range:
        if int(stride) <= 0:
            raise ValueError("window stride must be positive")
        low = max(CALVIN_HISTORY_HORIZON, self.valid_center_start)
        high = min(self.length - CALVIN_WORLD_HORIZON - 1, self.valid_center_end)
        if high < low:
            return range(0)
        return range(low, high + 1, int(stride))

    def window(self, center: int) -> dict[str, Any]:
        """Return one raw, unnormalised ClearVLA training-shaped window."""

        center = int(center)
        if center not in self.valid_centers():
            raise ValueError(
                f"center {center} is outside [{self.valid_center_start},{self.valid_center_end}] "
                "or lacks the complete causal/future window"
            )
        history_state_indices = np.asarray((center - 8, center - 4, center), dtype=np.int64)
        executed_indices = np.asarray(
            [center + value for value in (-24, -16, -12, -8, -6, -4, -2, -1)],
            dtype=np.int64,
        )
        future_image_indices = np.arange(
            center + 4, center + CALVIN_WORLD_HORIZON + 1, 4, dtype=np.int64
        )
        action_indices = np.arange(
            center, center + CALVIN_WORLD_HORIZON, dtype=np.int64
        )
        future_state_indices = np.arange(
            center + 1, center + CALVIN_WORLD_HORIZON + 1, dtype=np.int64
        )
        history = self.arrays(history_state_indices)
        executed = self.arrays(executed_indices)
        future = self.arrays(action_indices)
        future_support = self.arrays(future_image_indices)
        future_state = self.arrays(future_state_indices)
        return {
            "schema": RAW_CALVIN_READER_SCHEMA,
            "episode_id": self.episode_id,
            "source_trajectory_id": self.source_trajectory_id,
            "instruction": self.instruction,
            "language_key": instruction_key(self.instruction),
            "task": self.task,
            "center": center,
            "state": history["state"][-1],
            "state_raw": history["state"][-1],
            "action_state": history["action_state"][-1],
            "action_state_raw": history["action_state"][-1],
            "history_state": history["state"],
            "executed_action_history": executed["action"],
            "action": future["action"],
            "policy_action": future["action"][:CALVIN_POLICY_HORIZON],
            "policy_action_raw": future["action"][:CALVIN_POLICY_HORIZON],
            "future_state": future_state["state"],
            "history_rgb_static": history["rgb_static"],
            "history_rgb_gripper": history["rgb_gripper"],
            # The mainline Teacher consumes the twelve 4-frame support rows;
            # the complete 48-row action/state arrays remain available above.
            "future_rgb_static": future_support["rgb_static"],
            "future_rgb_gripper": future_support["rgb_gripper"],
            "history_indices": history_state_indices,
            "future_indices": future_image_indices,
        }


def _source_trajectories(split_root: Path, source_split: str) -> tuple[RawCalvinTrajectory, ...]:
    path = split_root / "ep_start_end_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN source trajectory inventory is absent: {path}")
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError(f"CALVIN source trajectory inventory must be [N,2], got {values.shape}")
    prefix, digits, suffix = _frame_pattern(split_root)
    rows = tuple(
        RawCalvinTrajectory(
            source_split=str(source_split),
            index=int(index),
            start=int(start),
            end=int(end),
            frame_root=split_root,
            frame_prefix=prefix,
            frame_digits=digits,
            frame_suffix=suffix,
        )
        for index, (start, end) in enumerate(values.tolist())
    )
    ordered = sorted(rows, key=lambda row: (row.start, row.end, row.index))
    for row in ordered:
        if row.start < 0 or row.end < row.start:
            raise ValueError(f"invalid CALVIN source trajectory: {row.identity}")
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start <= previous.end:
            raise ValueError(
                f"CALVIN source trajectories overlap: {previous.identity!r} and {current.identity!r}"
            )
    return tuple(ordered)


def _load_annotations(
    split_root: Path,
    trajectories: Sequence[RawCalvinTrajectory],
) -> tuple[RawCalvinAnnotation, ...]:
    path = split_root / "lang_annotations" / "auto_lang_ann.npy"
    if not path.is_file():
        path = split_root / "auto_lang_ann.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN language annotation file is absent: {path}")
    # This is the official CALVIN object-array format.  The restricted loader
    # admits only NumPy ndarray reconstruction and builtin containers, while
    # rejecting arbitrary globals from an untrusted pickle payload.
    payload = _load_object_npy(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"CALVIN annotation payload is not a mapping: {path}")
    info = payload.get("info")
    language = payload.get("language")
    if not isinstance(info, Mapping) or not isinstance(language, Mapping):
        raise ValueError(f"CALVIN annotation payload lacks info/language mappings: {path}")
    indices = info.get("indx")
    annotations = language.get("ann")
    tasks = language.get("task")
    if indices is None or annotations is None:
        raise ValueError(f"CALVIN annotation payload lacks indx/ann arrays: {path}")
    if tasks is None:
        tasks = ["unknown"] * len(annotations)
    if not len(indices) == len(annotations) == len(tasks):
        raise ValueError("CALVIN language indices/annotations/tasks differ in length")
    rows: list[RawCalvinAnnotation] = []
    for index, ((start, end), instruction, task) in enumerate(
        zip(indices, annotations, tasks, strict=True)
    ):
        start = int(start)
        end = int(end)
        owners = [trajectory for trajectory in trajectories if trajectory.start <= start and end <= trajectory.end]
        if len(owners) != 1:
            raise ValueError(
                f"CALVIN annotation {index} [{start},{end}] belongs to {len(owners)} trajectories"
            )
        rows.append(
            RawCalvinAnnotation(
                index=int(index),
                start=start,
                end=end,
                instruction=normalize_instruction(str(instruction)),
                task=str(task),
                trajectory=owners[0],
            )
        )
    return tuple(rows)


def _round_robin_limit(rows: Sequence[RawCalvinAnnotation], limit: int | None) -> list[RawCalvinAnnotation]:
    if limit is None or len(rows) <= int(limit):
        return list(rows)
    if int(limit) <= 0:
        raise ValueError("limit must be positive")
    grouped: dict[str, list[RawCalvinAnnotation]] = {}
    for row in rows:
        grouped.setdefault(row.trajectory.identity, []).append(row)
    positions = {key: 0 for key in grouped}
    selected: list[RawCalvinAnnotation] = []
    while len(selected) < int(limit):
        advanced = False
        for key in sorted(grouped):
            position = positions[key]
            if position < len(grouped[key]):
                selected.append(grouped[key][position])
                positions[key] = position + 1
                advanced = True
                if len(selected) == int(limit):
                    break
        if not advanced:
            break
    return sorted(selected, key=lambda row: row.index)


class CalvinRawReader:
    """Read CALVIN source trajectories without creating converted files."""

    def __init__(
        self,
        source: Path,
        *,
        train_source_split: str = "training",
        heldout_source_split: str = "validation",
        val_fraction: float = 0.1,
        split_seed: int = 0,
        task_filter: str | None = None,
        limit_per_split: int | None = None,
        frame_cache_capacity: int = 256,
    ) -> None:
        source = Path(source)
        if not 0.0 < float(val_fraction) < 1.0:
            raise ValueError("val_fraction must be in (0,1)")
        if not source.is_dir():
            raise FileNotFoundError(f"CALVIN source root is absent: {source}")
        self.source = source
        self.train_source_split = str(train_source_split)
        self.heldout_source_split = str(heldout_source_split)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.task_filter = None if task_filter is None else str(task_filter)
        self.limit_per_split = None if limit_per_split is None else int(limit_per_split)
        if self.limit_per_split is not None and self.limit_per_split <= 0:
            raise ValueError("limit_per_split must be positive")
        self.frame_store = RawCalvinFrameStore(capacity=frame_cache_capacity)
        self._episodes: dict[str, tuple[RawCalvinEpisode, ...]] = {}
        self._build()

    def _build(self) -> None:
        source_rows: dict[str, list[RawCalvinAnnotation]] = {}
        for split in (self.train_source_split, self.heldout_source_split):
            split_root = self.source / split
            trajectories = _source_trajectories(split_root, split)
            annotations = _load_annotations(split_root, trajectories)
            eligible: list[RawCalvinAnnotation] = []
            for row in annotations:
                if self.task_filter is not None and row.task != self.task_filter:
                    continue
                context_start = max(row.trajectory.start, row.start - CALVIN_HISTORY_HORIZON)
                language_start_local = row.start - context_start
                language_end_local = row.end - context_start
                if (
                    language_start_local >= CALVIN_HISTORY_HORIZON
                    and language_start_local <= language_end_local - CALVIN_POLICY_HORIZON
                ):
                    eligible.append(row)
            source_rows[split] = _round_robin_limit(eligible, self.limit_per_split)

        train_rows = source_rows[self.train_source_split]
        trajectory_ids = sorted({row.trajectory.identity for row in train_rows})
        if len(trajectory_ids) < 2:
            raise ValueError("CALVIN training source needs at least two eligible trajectories")
        generator = np.random.default_rng(self.split_seed)
        shuffled = list(np.asarray(trajectory_ids, dtype=object)[generator.permutation(len(trajectory_ids))].tolist())
        validation_count = min(
            max(1, int(round(len(shuffled) * self.val_fraction))),
            len(shuffled) - 1,
        )
        validation_trajectories = set(shuffled[:validation_count])

        by_split: dict[str, list[RawCalvinEpisode]] = {"train": [], "val": [], "test": []}
        for source_split, rows in source_rows.items():
            for row in rows:
                context_start = max(row.trajectory.start, row.start - CALVIN_HISTORY_HORIZON)
                start_local = row.start - context_start
                end_local = row.end - context_start
                segment = RawCalvinEpisode(
                    annotation=row,
                    context_start=context_start,
                    language_start_local=start_local,
                    language_end_local=end_local,
                    frame_store=self.frame_store,
                )
                if source_split == self.heldout_source_split:
                    destination = "test"
                elif row.trajectory.identity in validation_trajectories:
                    destination = "val"
                else:
                    destination = "train"
                by_split[destination].append(segment)
        if any(not rows for rows in by_split.values()):
            raise ValueError("CALVIN raw reader needs non-empty train/val/test splits")
        self._episodes = {name: tuple(rows) for name, rows in by_split.items()}

    def episodes(self, split: str) -> tuple[RawCalvinEpisode, ...]:
        try:
            return self._episodes[str(split)]
        except KeyError as error:
            raise ValueError(f"unknown CALVIN reader split {split!r}") from error

    def iter_episodes(self, split: str) -> Iterator[RawCalvinEpisode]:
        yield from self.episodes(split)

    def iter_windows(
        self,
        split: str,
        *,
        stride: int = 1,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        emitted = 0
        for episode in self.episodes(split):
            for center in episode.valid_centers(stride=stride):
                if limit is not None and emitted >= int(limit):
                    return
                yield episode.window(center)
                emitted += 1

    def report(self) -> dict[str, Any]:
        trajectory_counts = {
            split: len({episode.source_trajectory_id for episode in self.episodes(split)})
            for split in ("train", "val", "test")
        }
        return {
            "schema": RAW_CALVIN_READER_SCHEMA,
            "source": str(self.source.resolve()),
            "task_filter": self.task_filter,
            "split_unit": "source-trajectory",
            "episodes": {split: len(self.episodes(split)) for split in ("train", "val", "test")},
            "source_trajectories": trajectory_counts,
            "windows": {
                split: sum(len(episode.valid_centers()) for episode in self.episodes(split))
                for split in ("train", "val", "test")
            },
            "trajectory_overlap": {
                f"{left}_{right}": len(
                    {
                        episode.source_trajectory_id for episode in self.episodes(left)
                    }.intersection(
                        {episode.source_trajectory_id for episode in self.episodes(right)}
                    )
                )
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            },
        }

    def cached_prefix_split_manifest(self, cached_hdf5_root: Path) -> dict[str, Any]:
        """Select raw-contract episodes backed by an existing v1 cache prefix.

        The old full CALVIN conversion owns useful decoded/DINO rows but not
        the corrected terminal contract.  This manifest keeps only episode
        identities present in that immutable prefix; the loader later obtains
        split/terminal truth from this raw reader and adds the absorbing tail
        in memory.  Missing newly eligible annotations are reported rather
        than silently claimed as part of the experiment.
        """

        root = Path(cached_hdf5_root)
        if not root.is_dir():
            raise FileNotFoundError(f"CALVIN cached-prefix HDF5 root is absent: {root}")
        cached_paths = {
            path.stem: path
            for pattern in ("*.hdf5", "*.h5")
            for path in root.glob(pattern)
            if path.is_file()
        }
        if not cached_paths:
            raise FileNotFoundError(f"no cached-prefix HDF5 episodes under {root}")

        # converter-v1 renumbered annotations after filtering, so a cache
        # filename is not a raw annotation identity.  Match the immutable
        # source tuple and serialize the cache->raw mapping explicitly.
        raw_by_signature: dict[tuple[str, int, int, int, str, str], RawCalvinEpisode] = {}
        for split in ("train", "val", "test"):
            for raw in self.episodes(split):
                signature = (
                    raw.annotation.trajectory.source_split,
                    int(raw.annotation.start),
                    int(raw.annotation.end),
                    int(raw.context_start),
                    str(raw.task),
                    str(raw.instruction),
                )
                if signature in raw_by_signature:
                    raise ValueError(f"duplicate raw CALVIN source signature: {signature!r}")
                raw_by_signature[signature] = raw

        def _attr(handle: h5py.File, name: str) -> object:
            value = handle.attrs[name]
            if isinstance(value, (bytes, np.bytes_)):
                return bytes(value).decode("utf-8")
            return value

        cache_to_raw: dict[str, str] = {}
        unmatched_cached: list[str] = []
        for cache_id, path in sorted(cached_paths.items()):
            required = (
                "source_split",
                "source_start",
                "source_end",
                "context_start",
                "task",
                "instruction",
            )
            try:
                with h5py.File(path, "r") as handle:
                    if any(name not in handle.attrs for name in required):
                        unmatched_cached.append(cache_id)
                        continue
                    signature = (
                        str(_attr(handle, "source_split")),
                        int(_attr(handle, "source_start")),
                        int(_attr(handle, "source_end")),
                        int(_attr(handle, "context_start")),
                        str(_attr(handle, "task")),
                        str(_attr(handle, "instruction")),
                    )
            except (OSError, ValueError, TypeError):
                unmatched_cached.append(cache_id)
                continue
            raw = raw_by_signature.get(signature)
            if raw is None:
                unmatched_cached.append(cache_id)
            else:
                cache_to_raw[cache_id] = raw.episode_id

        raw_to_cache = {raw_id: cache_id for cache_id, raw_id in cache_to_raw.items()}

        split_rows: dict[str, list[str]] = {}
        selected_episodes: list[RawCalvinEpisode] = []
        raw_names: set[str] = set()
        for split in ("train", "val", "test"):
            rows = list(self.episodes(split))
            raw_names.update(row.episode_id for row in rows)
            selected = [row for row in rows if row.episode_id in raw_to_cache]
            if not selected:
                raise ValueError(f"cached CALVIN prefix has no {split} episodes")
            split_rows[split] = [raw_to_cache[row.episode_id] for row in selected]
            selected_episodes.extend(selected)

        selected_tasks = sorted({row.task for row in selected_episodes})
        train_tasks = {
            row.task for row in self.episodes("train") if row.episode_id in raw_to_cache
        }
        missing_train_tasks = sorted(set(selected_tasks).difference(train_tasks))
        if missing_train_tasks:
            raise ValueError(
                "cached CALVIN prefix has evaluation-only tasks: "
                f"{missing_train_tasks}"
            )
        task_counts = {
            split: {
                task: sum(
                    row.task == task
                    for row in self.episodes(split)
                    if row.episode_id in raw_to_cache
                )
                for task in selected_tasks
            }
            for split in ("train", "val", "test")
        }
        return {
            "schema": EPISODE_SPLIT_MANIFEST_SCHEMA,
            "task_filter": "" if self.task_filter is None else self.task_filter,
            "split_unit": "source-trajectory",
            "split_seed": self.split_seed,
            "train_validation_fraction": self.val_fraction,
            "raw_source": str(self.source.resolve()),
            "cached_prefix_root": str(root.resolve()),
            "terminal_overlay": RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
            "terminal_padding_frames": CALVIN_TERMINAL_PADDING_FRAMES,
            "task_order": selected_tasks,
            "task_counts": task_counts,
            "selected_episode_count": len(selected_episodes),
            "raw_eligible_episode_count": len(raw_names),
            "raw_eligible_without_cached_prefix_count": len(raw_names.difference(raw_to_cache)),
            "cached_prefix_without_raw_eligibility_count": len(unmatched_cached),
            "raw_eligible_without_cached_prefix_examples": sorted(
                raw_names.difference(raw_to_cache)
            )[:16],
            "cached_prefix_without_raw_eligibility_examples": sorted(
                unmatched_cached
            )[:16],
            "raw_episode_map": cache_to_raw,
            "splits": split_rows,
        }


def virtualize_calvin_cached_prefix(
    episodes: Sequence[LoadedEpisode],
    reader: CalvinRawReader,
    *,
    expected_splits: Mapping[str, Sequence[str]],
    raw_episode_map: Mapping[str, str] | None = None,
) -> tuple[list[LoadedEpisode], dict[str, Any]]:
    """Apply raw trajectory/success-tail truth to cached v1 HDF5 prefixes.

    Numeric source rows and their visual/DINO caches remain read-only.  The
    returned episodes replace the annotated terminal command and add the same
    24-row absorbing suffix as converter-v2.  Cache readers are told exactly
    how many real-prefix rows exist and may only repeat the terminal visual row
    for that declared suffix.
    """

    if not episodes:
        raise ValueError("CALVIN cached-prefix overlay requires episodes")
    if set(expected_splits) != {"train", "val", "test"}:
        raise ValueError("CALVIN raw overlay requires train/val/test split rows")
    raw_by_name: dict[str, RawCalvinEpisode] = {}
    raw_split: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for raw in reader.episodes(split):
            if raw.episode_id in raw_by_name:
                raise ValueError(f"duplicate raw CALVIN episode {raw.episode_id!r}")
            raw_by_name[raw.episode_id] = raw
            raw_split[raw.episode_id] = split

    expected_by_name: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for value in expected_splits[split]:
            name = str(value)
            if name in expected_by_name:
                raise ValueError(f"CALVIN overlay split duplicates {name!r}")
            expected_by_name[name] = split
    loaded_names = {episode.episode_id for episode in episodes}
    if loaded_names != set(expected_by_name):
        missing = sorted(set(expected_by_name).difference(loaded_names))
        extra = sorted(loaded_names.difference(expected_by_name))
        raise ValueError(
            "CALVIN cached-prefix inventory differs from its split manifest: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    cache_to_raw = {
        str(cache_id): str(raw_id)
        for cache_id, raw_id in (raw_episode_map or {}).items()
    }
    overlaid: list[LoadedEpisode] = []
    task_counts = {
        split: Counter() for split in ("train", "val", "test")
    }
    for episode in episodes:
        raw_id = cache_to_raw.get(episode.episode_id, episode.episode_id)
        raw = raw_by_name.get(raw_id)
        if raw is None:
            raise ValueError(
                f"cached episode {episode.episode_id!r} maps to unknown raw episode {raw_id!r}"
            )
        expected_split = expected_by_name[episode.episode_id]
        if raw_split[raw_id] != expected_split:
            raise ValueError(
                f"cached episode {episode.episode_id!r} is assigned to "
                f"{expected_split!r}, raw trajectory policy assigns "
                f"{raw_split[raw_id]!r}"
            )
        if episode.cache_frame_count is not None:
            raise ValueError("CALVIN cached-prefix episode is already virtualized")
        terminal = int(raw.terminal_state_index)
        physical_length = int(episode.length)
        if physical_length != terminal + 1:
            raise ValueError(
                f"{episode.episode_id}: cached prefix T={physical_length} does not "
                f"end at raw terminal index {terminal}"
            )
        if episode.instruction != raw.instruction or episode.task_id != raw.task:
            raise ValueError(
                f"{episode.episode_id}: cached instruction/task differs from raw source"
            )
        for name, cached, source in (
            ("source_start", episode.source_start, raw.annotation.start),
            ("source_end", episode.source_end, raw.annotation.end),
            ("context_start", episode.context_start, raw.context_start),
        ):
            if cached is None or int(cached) != int(source):
                raise ValueError(
                    f"{episode.episode_id}: cached {name}={cached} differs from raw {source}"
                )
        actions = np.asarray(episode.actions_raw, dtype=np.float32).copy()
        states_value = episode.states_raw
        if states_value is None:
            raise ValueError(f"{episode.episode_id}: CALVIN prefix has no state array")
        states = np.asarray(states_value, dtype=np.float32).copy()
        if actions.shape != (physical_length, 7) or states.shape != (
            physical_length,
            7,
        ):
            raise ValueError(
                f"{episode.episode_id}: CALVIN prefix must use aligned [T,7] arrays"
            )
        terminal_action = np.zeros((7,), dtype=np.float32)
        terminal_action[-1] = actions[terminal - 1, -1]
        actions[terminal] = terminal_action
        actions = np.concatenate(
            (
                actions,
                np.repeat(
                    terminal_action[None], CALVIN_TERMINAL_PADDING_FRAMES, axis=0
                ),
            ),
            axis=0,
        )
        states = np.concatenate(
            (
                states,
                np.repeat(
                    states[terminal : terminal + 1],
                    CALVIN_TERMINAL_PADDING_FRAMES,
                    axis=0,
                ),
            ),
            axis=0,
        )
        action_states = np.concatenate(
            (np.zeros((1, 7), dtype=np.float32), actions[:-1]), axis=0
        )
        if len(actions) != raw.length:
            raise AssertionError("CALVIN virtual suffix length differs from raw reader")
        overlaid.append(
            replace(
                episode,
                actions_raw=np.ascontiguousarray(actions),
                states_raw=np.ascontiguousarray(states),
                action_states_raw=np.ascontiguousarray(action_states),
                valid_center_start=int(raw.valid_center_start),
                valid_center_end=int(raw.valid_center_end),
                terminal_state_index=terminal,
                terminal_padding_mode=RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
                cache_frame_count=physical_length,
                source_trajectory_id=raw.source_trajectory_id,
            )
        )
        task_counts[expected_split][raw.task] += 1

    task_order = sorted({task for counts in task_counts.values() for task in counts})
    missing_train = [task for task in task_order if task_counts["train"][task] == 0]
    if missing_train:
        raise ValueError(f"CALVIN raw overlay has tasks absent from train: {missing_train}")
    report = {
        "schema": "clearvla-calvin-cached-prefix-overlay-v1",
        "raw_reader_schema": RAW_CALVIN_READER_SCHEMA,
        "source": str(reader.source.resolve()),
        "split_unit": "source-trajectory",
        "terminal_padding_mode": RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
        "terminal_padding_frames": CALVIN_TERMINAL_PADDING_FRAMES,
        "cached_prefix_visual_reuse": "real_rows_plus_terminal_repeat_only",
        "episodes": {
            split: int(sum(task_counts[split].values()))
            for split in ("train", "val", "test")
        },
        "task_order": task_order,
        "task_counts": {
            split: {task: int(task_counts[split][task]) for task in task_order}
            for split in ("train", "val", "test")
        },
    }
    return overlaid, report


def _sample_summary(window: Mapping[str, Any]) -> dict[str, Any]:
    arrays = {
        key: value
        for key, value in window.items()
        if isinstance(value, np.ndarray)
    }
    return {
        "episode_id": window["episode_id"],
        "source_trajectory_id": window["source_trajectory_id"],
        "center": int(window["center"]),
        "instruction": window["instruction"],
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "policy_action_terminal_row": np.asarray(window["policy_action"])[-1].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--window-limit", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument(
        "--cached-prefix-root",
        type=Path,
        default=None,
        help="Existing converter-v1 HDF5/cache identity to select for a raw overlay",
    )
    parser.add_argument(
        "--split-manifest-output",
        type=Path,
        default=None,
        help="Write a trajectory-disjoint cached-prefix manifest for mainline training",
    )
    args = parser.parse_args()
    reader = CalvinRawReader(
        args.source,
        task_filter=args.task_filter,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        limit_per_split=args.limit_per_split,
    )
    output: dict[str, Any] = {"report": reader.report()}
    if args.split_manifest_output is not None:
        if args.cached_prefix_root is None:
            parser.error("--split-manifest-output requires --cached-prefix-root")
        manifest = reader.cached_prefix_split_manifest(args.cached_prefix_root)
        atomic_json(args.split_manifest_output, manifest)
        output["split_manifest"] = {
            key: value for key, value in manifest.items() if key != "splits"
        }
    if args.split is not None:
        windows = reader.iter_windows(
            args.split,
            stride=args.window_stride,
            limit=None if args.window_limit <= 0 else args.window_limit,
        )
        output["samples"] = [_sample_summary(window) for window in windows]
    print(json.dumps(output, indent=2, sort_keys=True))


__all__ = [
    "RAW_CALVIN_READER_SCHEMA",
    "CalvinRawReader",
    "RawCalvinAnnotation",
    "RawCalvinEpisode",
    "RawCalvinFrame",
    "RawCalvinFrameStore",
    "RawCalvinTrajectory",
    "virtualize_calvin_cached_prefix",
]


if __name__ == "__main__":
    main()
