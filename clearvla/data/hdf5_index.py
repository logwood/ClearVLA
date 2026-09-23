"""Immutable metadata index and lazy row reader for HDF5 episodes.

This module is deliberately separate from the mainline loader.  It provides a
loader-only experiment for measuring cold admission and proving that selected
rows are byte-identical to the existing eager path before any training config
can opt in.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import h5py
import numpy as np

from .hdf5_episode import (
    _manifest_hdf5_candidates,
    decode_hdf5_instruction,
    episode_identity,
)
from .index_contract import FingerprintMode, RowRequest, SourceFingerprint
from .instructions import instruction_key, normalize_instruction
from .schema import (
    ACTION_ALIASES,
    ACTION_STATE_ALIASES,
    CAMERA_ALIASES,
    STATE_ALIASES,
    resolve_key,
)

INDEX_SCHEMA = "clearvla-hdf5-episode-index-v2"
LEGACY_INDEX_SCHEMA = "clearvla-hdf5-episode-index-v1"


def _validate_relative_file(value: str) -> str:
    relative = str(value)
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ":" in relative
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ValueError(f"indexed HDF5 file must be root-relative: {relative!r}")
    return relative


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported HDF5 attribute type: {type(value)!r}")


def _attr(attrs: Mapping[str, object], key: str, default: object = None) -> object:
    return _json_scalar(attrs[key]) if key in attrs else default


@dataclass(frozen=True)
class HDF5EpisodeIndexEntry:
    """Metadata needed to validate and lazily read one episode."""

    episode_id: str
    relative_path: str
    action_key: str
    state_key: str | None
    action_state_key: str | None
    camera_keys: dict[str, str]
    action_shape: tuple[int, int]
    state_shape: tuple[int, int]
    action_state_shape: tuple[int, int]
    camera_lengths: dict[str, int]
    metadata: dict[str, object]
    file_size: int
    mtime_ns: int
    # Keep the actual suffix in the index.  The identity intentionally omits
    # it, but a backend must not guess between .h5 and .hdf5 at read time.
    relative_file: str = ""
    source_fingerprint: SourceFingerprint | None = None
    field_shapes: dict[str, tuple[int, ...]] | None = None
    field_dtypes: dict[str, str] | None = None

    @property
    def path(self) -> Path:
        return Path(self.relative_path)

    def to_json(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "relative_path": self.relative_path,
            "action_key": self.action_key,
            "state_key": self.state_key,
            "action_state_key": self.action_state_key,
            "camera_keys": dict(self.camera_keys),
            "action_shape": list(self.action_shape),
            "state_shape": list(self.state_shape),
            "action_state_shape": list(self.action_state_shape),
            "camera_lengths": dict(self.camera_lengths),
            "metadata": dict(self.metadata),
            "file_size": self.file_size,
            "mtime_ns": self.mtime_ns,
            "relative_file": self.relative_file,
            "source_fingerprint": (
                None
                if self.source_fingerprint is None
                else self.source_fingerprint.to_json()
            ),
            "field_shapes": {
                key: list(shape) for key, shape in (self.field_shapes or {}).items()
            },
            "field_dtypes": dict(self.field_dtypes or {}),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "HDF5EpisodeIndexEntry":
        action_key = str(payload["action_key"])
        state_key = None if payload.get("state_key") is None else str(payload["state_key"])
        action_state_key = (
            None
            if payload.get("action_state_key") is None
            else str(payload["action_state_key"])
        )
        action_shape = tuple(int(v) for v in payload["action_shape"])
        state_shape = tuple(int(v) for v in payload["state_shape"])
        action_state_shape = tuple(int(v) for v in payload["action_state_shape"])
        camera_keys = {str(k): str(v) for k, v in dict(payload["camera_keys"]).items()}
        camera_lengths = {str(k): int(v) for k, v in dict(payload["camera_lengths"]).items()}
        field_shapes_payload = dict(payload.get("field_shapes", {}))
        field_shapes = (
            {
                str(key): tuple(int(value) for value in shape)
                for key, shape in field_shapes_payload.items()
            }
            if field_shapes_payload
            else {
                action_key: action_shape,
                **({state_key: state_shape} if state_key is not None else {}),
                **(
                    {action_state_key: action_state_shape}
                    if action_state_key is not None
                    else {}
                ),
                **{key: (length,) for key, length in camera_lengths.items()},
            }
        )
        return cls(
            episode_id=str(payload["episode_id"]),
            relative_path=_validate_relative_file(str(payload["relative_path"])),
            action_key=action_key,
            state_key=state_key,
            action_state_key=action_state_key,
            camera_keys=camera_keys,
            action_shape=action_shape,
            state_shape=state_shape,
            action_state_shape=action_state_shape,
            camera_lengths=camera_lengths,
            metadata=dict(payload.get("metadata", {})),
            file_size=int(payload["file_size"]),
            mtime_ns=int(payload["mtime_ns"]),
            relative_file=(
                ""
                if not payload.get("relative_file")
                else _validate_relative_file(str(payload["relative_file"]))
            ),
            source_fingerprint=(
                None
                if payload.get("source_fingerprint") is None
                else SourceFingerprint.from_json(dict(payload["source_fingerprint"]))
            ),
            field_shapes=field_shapes,
            field_dtypes={
                str(key): str(value)
                for key, value in dict(payload.get("field_dtypes", {})).items()
            },
        )


def _resolve_keys(
    datasets: Sequence[str],
    *,
    action_key: str,
    state_key: str | None,
    action_state_key: str | None,
    cameras: Sequence[str],
    camera_key_overrides: Mapping[str, str] | None,
) -> tuple[str, str | None, str | None, dict[str, str]]:
    resolved_action = resolve_key(datasets, action_key, ACTION_ALIASES, required=True)
    assert resolved_action is not None
    resolved_state = (
        resolve_key(datasets, state_key, STATE_ALIASES, required=True)
        if state_key
        else None
    )
    resolved_action_state = (
        resolve_key(datasets, action_state_key, ACTION_STATE_ALIASES, required=True)
        if action_state_key
        else None
    )
    overrides = camera_key_overrides or {}
    camera_keys: dict[str, str] = {}
    for camera in cameras:
        requested = overrides.get(camera)
        aliases = CAMERA_ALIASES.get(camera, ())
        if not aliases and requested is None:
            raise KeyError(f"unknown camera name={camera!r}")
        key = resolve_key(datasets, requested, aliases, required=True)
        assert key is not None
        camera_keys[camera] = key
    return resolved_action, resolved_state, resolved_action_state, camera_keys


def _entry_for_path(
    root: Path,
    path: Path,
    *,
    cameras: Sequence[str],
    action_key: str,
    state_key: str | None,
    action_state_key: str | None,
    camera_key_overrides: Mapping[str, str] | None,
    fingerprint_mode: FingerprintMode,
) -> HDF5EpisodeIndexEntry:
    identity, _partition, task_id = episode_identity(root, path)
    before = SourceFingerprint.from_path(path, mode=fingerprint_mode)
    with h5py.File(path, "r") as handle:
        # Resolve the same finite alias candidates as the eager loader without
        # visititems(): unrelated image datasets/groups need not be traversed.
        candidates = [action_key, state_key, action_state_key]
        candidates.extend(ACTION_ALIASES + STATE_ALIASES + ACTION_STATE_ALIASES)
        for camera in cameras:
            candidates.extend(CAMERA_ALIASES.get(camera, ()))
            candidates.append((camera_key_overrides or {}).get(camera))
        names = {
            variant
            for candidate in candidates if candidate
            for variant in (candidate, candidate.lstrip("/"), candidate.replace(".", "/").lstrip("/"))
        }
        datasets = {name: {} for name in names if isinstance(handle.get(name), h5py.Dataset)}
        resolved_action, resolved_state, resolved_action_state, camera_keys = _resolve_keys(
            datasets, action_key=action_key, state_key=state_key,
            action_state_key=action_state_key, cameras=cameras,
            camera_key_overrides=camera_key_overrides,
        )
        action = handle[resolved_action]
        if action.ndim != 2 or min(action.shape) < 1:
            raise ValueError(f"{path}: action must be rank-2, got {action.shape}")
        action_shape = tuple(int(v) for v in action.shape)
        state = handle[resolved_state] if resolved_state is not None else action
        action_state = (
            handle[resolved_action_state]
            if resolved_action_state is not None
            else state
        )
        if state.shape != action.shape or action_state.shape != action.shape:
            raise ValueError(f"{path}: action/state shapes are not aligned")
        state_shape = tuple(int(v) for v in state.shape)
        action_state_shape = tuple(int(v) for v in action_state.shape)
        camera_lengths: dict[str, int] = {}
        field_shapes: dict[str, tuple[int, ...]] = {
            resolved_action: action_shape,
            **({resolved_state: state_shape} if resolved_state is not None else {}),
            **(
                {resolved_action_state: action_state_shape}
                if resolved_action_state is not None
                else {}
            ),
        }
        field_dtypes: dict[str, str] = {
            resolved_action: np.dtype(action.dtype).str,
        }
        if resolved_state is not None:
            field_dtypes[resolved_state] = np.dtype(state.dtype).str
        if resolved_action_state is not None:
            field_dtypes[resolved_action_state] = np.dtype(action_state.dtype).str
        for camera, key in camera_keys.items():
            dataset = handle.get(key)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1:
                raise TypeError(f"{path}: camera={camera!r} is not a sequence")
            camera_lengths[camera] = int(dataset.shape[0])
            field_shapes[key] = tuple(int(value) for value in dataset.shape)
            field_dtypes[key] = np.dtype(dataset.dtype).str
        if any(length != int(action.shape[0]) for length in camera_lengths.values()):
            raise ValueError(f"{path}: camera lengths do not match action length")
        raw_instruction = handle.get("instruction")
        instruction = (
            decode_hdf5_instruction(raw_instruction[()])
            if isinstance(raw_instruction, h5py.Dataset)
            else _attr(handle.attrs, "instruction")
        )
        raw_language_key = _attr(handle.attrs, "language_key")
        language_key = (
            None if raw_language_key is None else str(raw_language_key).strip()
        )
        raw_task = _attr(handle.attrs, "task")
        task = task_id if raw_task is None else str(raw_task).strip()
        metadata_instruction = (
            None
            if instruction is None
            else (
                instruction
                if isinstance(raw_instruction, h5py.Dataset)
                else normalize_instruction(instruction)
            )
        )
        if metadata_instruction is not None:
            expected_language_key = instruction_key(metadata_instruction)
            if language_key is None:
                language_key = expected_language_key
            elif language_key != expected_language_key:
                raise ValueError(f"{path}: language_key does not identify its instruction")
        elif language_key is not None:
            raise ValueError(f"{path}: language_key exists without an instruction")
        metadata = {
            "instruction": metadata_instruction,
            "language_key": language_key,
            "task": task,
            "valid_center_start": _attr(handle.attrs, "valid_center_start"),
            "valid_center_end": _attr(handle.attrs, "valid_center_end"),
            "strict_valid_center_start": _attr(handle.attrs, "strict_valid_center_start"),
            "strict_valid_center_end": _attr(handle.attrs, "strict_valid_center_end"),
            "terminal_state_index": _attr(handle.attrs, "terminal_state_index"),
            "terminal_padding_mode": _attr(handle.attrs, "terminal_padding_mode", ""),
            "source_action_count": _attr(handle.attrs, "source_action_count"),
            "source_start": _attr(handle.attrs, "source_start"),
            "source_end": _attr(handle.attrs, "source_end"),
            "context_start": _attr(handle.attrs, "context_start"),
            "source_trajectory_id": _attr(handle.attrs, "source_trajectory_id", ""),
        }
    before.validate(path)
    return HDF5EpisodeIndexEntry(
        episode_id=identity,
        relative_path=path.resolve().relative_to(root.resolve()).with_suffix("").as_posix(),
        action_key=resolved_action,
        state_key=resolved_state,
        action_state_key=resolved_action_state,
        camera_keys=camera_keys,
        action_shape=action_shape,
        state_shape=state_shape,
        action_state_shape=action_state_shape,
        camera_lengths=camera_lengths,
        metadata=metadata,
        file_size=int(before.size),
        mtime_ns=int(before.mtime_ns),
        relative_file=path.resolve().relative_to(root.resolve()).as_posix(),
        source_fingerprint=before,
        field_shapes=field_shapes,
        field_dtypes=field_dtypes,
    )


def build_hdf5_episode_index(
    root: Path,
    pattern: str,
    *,
    cameras: Sequence[str],
    episode_names: Sequence[str] | None = None,
    action_key: str = "action",
    state_key: str | None = None,
    action_state_key: str | None = None,
    camera_key_overrides: Mapping[str, str] | None = None,
    fingerprint_mode: FingerprintMode = "stat",
) -> list[HDF5EpisodeIndexEntry]:
    """Build metadata without reading full action/state arrays."""

    requested = None if episode_names is None else tuple(str(v) for v in episode_names)
    direct = _manifest_hdf5_candidates(root, pattern, requested) if requested else None
    paths = direct if direct is not None else sorted(root.glob(pattern))
    entries = [
        _entry_for_path(
            root,
            path,
            cameras=cameras,
            action_key=action_key,
            state_key=state_key,
            action_state_key=action_state_key,
            camera_key_overrides=camera_key_overrides,
            fingerprint_mode=fingerprint_mode,
        )
        for path in paths
        if path.is_file()
    ]
    if requested is not None:
        observed = {entry.episode_id for entry in entries}
        missing = sorted(set(requested).difference(observed))
        if missing:
            raise FileNotFoundError(f"requested episodes are absent: {missing[:8]}")
        by_id = {entry.episode_id: entry for entry in entries}
        # Match the existing direct-manifest loader, which sorts the resolved
        # paths before constructing LoadedEpisode objects.  A future loader
        # may expose manifest order explicitly, but this unit must be byte and
        # identity compatible with the current path first.
        entries = [by_id[episode_id] for episode_id in sorted(requested)]
    if not entries:
        raise RuntimeError("no HDF5 episode index entries were built")
    return entries


def save_hdf5_episode_index(
    path: Path,
    *,
    root: Path,
    pattern: str,
    entries: Sequence[HDF5EpisodeIndexEntry],
    contract: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "schema": INDEX_SCHEMA,
        "root": str(root.resolve()),
        "pattern": pattern,
        "contract": dict(contract or {}),
        "entries": [entry.to_json() for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_hdf5_episode_index(
    path: Path,
    *,
    verify_files: bool = True,
    expected_contract: Mapping[str, object] | None = None,
) -> tuple[Path, list[HDF5EpisodeIndexEntry]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {INDEX_SCHEMA, LEGACY_INDEX_SCHEMA}:
        raise ValueError(f"unsupported HDF5 episode index schema: {payload.get('schema')!r}")
    contract = dict(payload.get("contract", {}))
    if expected_contract is not None:
        mismatches = {
            key: (expected, contract.get(key))
            for key, expected in expected_contract.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"indexed reader contract mismatch: {mismatches}")
    root = Path(str(payload["root"]))
    entries = [HDF5EpisodeIndexEntry.from_json(item) for item in payload["entries"]]
    if verify_files:
        for entry in entries:
            candidate = root / entry.relative_file if entry.relative_file else root / (entry.relative_path + ".hdf5")
            if not candidate.is_file() and not entry.relative_file:
                candidate = root / (entry.relative_path + ".h5")
            if not candidate.is_file():
                raise FileNotFoundError(f"indexed episode is missing: {entry.episode_id}")
            if entry.source_fingerprint is not None:
                entry.source_fingerprint.validate(candidate)
            else:
                stat = candidate.stat()
                if int(stat.st_size) != entry.file_size or int(stat.st_mtime_ns) != entry.mtime_ns:
                    raise ValueError(f"indexed episode changed since admission: {entry.episode_id}")
    return root, entries


class HDF5RowReader:
    """Process-local bounded HDF5 row reader for worker-safe lazy access.

    The reader owns no handles at construction time.  A handle is opened on
    the first request in each worker process and evicted by LRU order.  When a
    process is forked or a worker is recreated, the PID change drops every
    inherited handle before opening a new one.  This keeps HDF5 objects out of
    pickled dataset state and makes the same interface usable by other storage
    backends.
    """

    def __init__(self, root: Path, *, max_open_files: int = 16) -> None:
        if int(max_open_files) <= 0:
            raise ValueError("max_open_files must be positive")
        self.root = Path(root)
        self.max_open_files = int(max_open_files)
        self._pid = os.getpid()
        self._handles: OrderedDict[str, h5py.File] = OrderedDict()

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        self.close()
        self._pid = pid

    def _path_for(self, entry: HDF5EpisodeIndexEntry) -> Path:
        if entry.relative_file:
            relative_file = _validate_relative_file(entry.relative_file)
            path = self.root / relative_file
        else:
            relative_path = _validate_relative_file(entry.relative_path)
            path = self.root / (relative_path + ".hdf5")
        if not path.is_file() and not entry.relative_file:
            path = self.root / (relative_path + ".h5")
        if not path.is_file():
            raise FileNotFoundError(f"indexed episode is missing: {entry.episode_id}")
        return path

    def _validate_source(self, path: Path, entry: HDF5EpisodeIndexEntry) -> None:
        if entry.source_fingerprint is not None:
            entry.source_fingerprint.validate(path)
        else:
            stat = path.stat()
            if int(stat.st_size) != entry.file_size or int(stat.st_mtime_ns) != entry.mtime_ns:
                raise ValueError(f"indexed episode changed since admission: {entry.episode_id}")

    def _handle_for(self, path: Path, entry: HDF5EpisodeIndexEntry) -> h5py.File:
        self._reset_after_fork()
        self._validate_source(path, entry)
        key = str(path.resolve())
        handle = self._handles.pop(key, None)
        if handle is None or not handle.id.valid:
            if handle is not None:
                handle.close()
            handle = h5py.File(path, "r")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _old_key, old_handle = self._handles.popitem(last=False)
            old_handle.close()
        return handle

    def read_rows(
        self,
        entry: HDF5EpisodeIndexEntry,
        dataset_key: str,
        rows: Sequence[int],
        *,
        require_finite: bool = True,
        target_dtype: str | np.dtype | None = None,
    ) -> np.ndarray:
        request = RowRequest.of(dataset_key, rows)
        indices = np.asarray(request.rows, dtype=np.int64)
        shapes = entry.field_shapes or {}
        if dataset_key not in shapes:
            raise KeyError(f"dataset key was not admitted in the index: {dataset_key!r}")
        length = int(shapes[dataset_key][0])
        if int(indices.max()) >= length:
            raise IndexError(
                f"requested HDF5 rows [{int(indices.min())},{int(indices.max())}] "
                f"outside field={dataset_key!r} length={length}"
            )
        path = self._path_for(entry)
        handle = self._handle_for(path, entry)
        dataset = handle.get(dataset_key)
        if not isinstance(dataset, h5py.Dataset):
            raise KeyError(f"indexed HDF5 dataset key is absent: {dataset_key!r}")
        expected_shape = (entry.field_shapes or {}).get(dataset_key)
        actual_shape = tuple(int(value) for value in dataset.shape)
        shape_matches = actual_shape == expected_shape or (
            expected_shape is not None
            and len(expected_shape) == 1
            and actual_shape[:1] == expected_shape
        )
        if expected_shape is not None and not shape_matches:
            raise ValueError(f"indexed HDF5 dataset shape changed: {entry.episode_id}:{dataset_key}")
        expected_dtype = (entry.field_dtypes or {}).get(dataset_key)
        if expected_dtype is not None and np.dtype(dataset.dtype).str != expected_dtype:
            raise ValueError(f"indexed HDF5 dataset dtype changed: {entry.episode_id}:{dataset_key}")

        unique, inverse = np.unique(indices, return_inverse=True)
        # Consecutive windows use one hyperslab; arbitrary repeated history
        # rows use one unique read and scatter back to the caller's order.
        if int(unique[-1]) - int(unique[0]) + 1 == len(unique):
            values = np.asarray(dataset[int(unique[0]):int(unique[-1]) + 1])[inverse]
        else:
            values = np.asarray(dataset[unique])[inverse]
        if target_dtype is not None:
            values = np.asarray(values, dtype=np.dtype(target_dtype))
        if require_finite and np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
            raise ValueError(f"selected HDF5 rows contain non-finite values: {entry.episode_id}:{dataset_key}")
        return values

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        if hasattr(self, "_handles"):
            self._handles.clear()

    def __getstate__(self) -> dict[str, object]:
        # h5py handles are process-owned and must never cross a DataLoader
        # worker boundary.  The next process lazily reopens its own handles.
        return {"root": self.root, "max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.root = Path(state["root"])
        self.max_open_files = int(state["max_open_files"])
        self._pid = os.getpid()
        self._handles = OrderedDict()

    def __enter__(self) -> "HDF5RowReader":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass



def read_hdf5_rows(root: Path, entry: HDF5EpisodeIndexEntry, dataset_key: str, rows: Sequence[int]) -> np.ndarray:
    """Read selected rows lazily; no file handle crosses a worker boundary."""
    with HDF5RowReader(root, max_open_files=1) as reader:
        return reader.read_rows(entry, dataset_key, rows)


__all__ = [
    "HDF5EpisodeIndexEntry",
    "HDF5RowReader",
    "INDEX_SCHEMA",
    "LEGACY_INDEX_SCHEMA",
    "build_hdf5_episode_index",
    "load_hdf5_episode_index",
    "read_hdf5_rows",
    "save_hdf5_episode_index",
]
