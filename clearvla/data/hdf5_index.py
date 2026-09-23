"""Immutable metadata index and lazy row reader for HDF5 episodes.

This module is deliberately separate from the mainline loader.  It provides a
loader-only experiment for measuring cold admission and proving that selected
rows are byte-identical to the existing eager path before any training config
can opt in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .hdf5_episode import (
    _manifest_hdf5_candidates,
    decode_hdf5_instruction,
    episode_identity,
)
from .schema import (
    ACTION_ALIASES,
    ACTION_STATE_ALIASES,
    CAMERA_ALIASES,
    STATE_ALIASES,
    list_hdf5_datasets,
    resolve_key,
)


INDEX_SCHEMA = "clearvla-hdf5-episode-index-v1"


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
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "HDF5EpisodeIndexEntry":
        return cls(
            episode_id=str(payload["episode_id"]),
            relative_path=str(payload["relative_path"]),
            action_key=str(payload["action_key"]),
            state_key=(None if payload.get("state_key") is None else str(payload["state_key"])),
            action_state_key=(
                None
                if payload.get("action_state_key") is None
                else str(payload["action_state_key"])
            ),
            camera_keys={str(k): str(v) for k, v in dict(payload["camera_keys"]).items()},
            action_shape=tuple(int(v) for v in payload["action_shape"]),
            state_shape=tuple(int(v) for v in payload["state_shape"]),
            action_state_shape=tuple(int(v) for v in payload["action_state_shape"]),
            camera_lengths={str(k): int(v) for k, v in dict(payload["camera_lengths"]).items()},
            metadata=dict(payload.get("metadata", {})),
            file_size=int(payload["file_size"]),
            mtime_ns=int(payload["mtime_ns"]),
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
) -> HDF5EpisodeIndexEntry:
    datasets = list_hdf5_datasets(str(path))
    resolved_action, resolved_state, resolved_action_state, camera_keys = _resolve_keys(
        datasets,
        action_key=action_key,
        state_key=state_key,
        action_state_key=action_state_key,
        cameras=cameras,
        camera_key_overrides=camera_key_overrides,
    )
    with h5py.File(path, "r") as handle:
        action = handle[resolved_action]
        if action.ndim != 2:
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
        for camera, key in camera_keys.items():
            dataset = handle.get(key)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1:
                raise TypeError(f"{path}: camera={camera!r} is not a sequence")
            camera_lengths[camera] = int(dataset.shape[0])
        if any(length != int(action.shape[0]) for length in camera_lengths.values()):
            raise ValueError(f"{path}: camera lengths do not match action length")
        raw_instruction = handle.get("instruction")
        instruction = (
            decode_hdf5_instruction(raw_instruction[()])
            if isinstance(raw_instruction, h5py.Dataset)
            else _attr(handle.attrs, "instruction")
        )
        metadata = {
            "instruction": instruction,
            "language_key": _attr(handle.attrs, "language_key"),
            "task": _attr(handle.attrs, "task"),
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
    identity, _partition, _task = episode_identity(root, path)
    stat = path.stat()
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
        file_size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
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
) -> None:
    payload = {
        "schema": INDEX_SCHEMA,
        "root": str(root.resolve()),
        "pattern": pattern,
        "entries": [entry.to_json() for entry in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_hdf5_episode_index(path: Path, *, verify_files: bool = True) -> tuple[Path, list[HDF5EpisodeIndexEntry]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"unsupported HDF5 episode index schema: {payload.get('schema')!r}")
    root = Path(str(payload["root"]))
    entries = [HDF5EpisodeIndexEntry.from_json(item) for item in payload["entries"]]
    if verify_files:
        for entry in entries:
            candidate = root / (entry.relative_path + ".hdf5")
            if not candidate.is_file():
                candidate = root / (entry.relative_path + ".h5")
            if not candidate.is_file():
                raise FileNotFoundError(f"indexed episode is missing: {entry.episode_id}")
            stat = candidate.stat()
            if int(stat.st_size) != entry.file_size or int(stat.st_mtime_ns) != entry.mtime_ns:
                raise ValueError(f"indexed episode changed since admission: {entry.episode_id}")
    return root, entries


def read_hdf5_rows(root: Path, entry: HDF5EpisodeIndexEntry, dataset_key: str, rows: Sequence[int]) -> np.ndarray:
    """Read selected rows lazily; no file handle crosses a worker boundary."""

    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("rows must be a non-empty rank-1 sequence")
    if int(indices.min()) < 0 or int(indices.max()) >= entry.action_shape[0]:
        raise IndexError("requested HDF5 rows are outside the indexed episode")
    path = root / (entry.relative_path + ".hdf5")
    if not path.is_file():
        path = root / (entry.relative_path + ".h5")
    with h5py.File(path, "r") as handle:
        dataset = handle[dataset_key]
        return np.asarray(dataset[indices])


__all__ = [
    "HDF5EpisodeIndexEntry",
    "INDEX_SCHEMA",
    "build_hdf5_episode_index",
    "load_hdf5_episode_index",
    "read_hdf5_rows",
    "save_hdf5_episode_index",
]
