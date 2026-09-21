from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from .instructions import instruction_key, normalize_instruction
from .schema import (
    ACTION_ALIASES,
    ACTION_STATE_ALIASES,
    CAMERA_ALIASES,
    STATE_ALIASES,
    list_hdf5_datasets,
    resolve_key,
)

RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING = "relative-action-absorbing-v1"
LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING = "libero-terminal-replay-absorbing-v2"
LIBERO_E8_STATE_NORMALIZER_REFERENCE = (
    "legacy-libero-v1-post-action-real-rows-for-e8-continuation-v1"
)
SUPPORTED_TERMINAL_PADDING_MODES = frozenset(
    {
        RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
        LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    }
)


@dataclass
class LoadedEpisode:
    path: Path
    episode_id: str
    source_partition: str
    task_id: str
    action_key: str
    camera_keys: dict[str, str]
    actions_raw: np.ndarray
    instruction: str | None = None
    actions_norm: np.ndarray | None = None
    action_state_key: str | None = None
    state_key: str | None = None
    states_raw: np.ndarray | None = None
    states_norm: np.ndarray | None = None
    # ``action_states_raw`` is observed qpos expressed in the command chart.
    # It equals ``states_raw`` for legacy data and is explicitly converted by
    # an RDT profile when qpos/action gripper source scales differ.
    action_states_raw: np.ndarray | None = None
    language_key: str | None = None
    valid_center_start: int | None = None
    valid_center_end: int | None = None
    strict_valid_center_start: int | None = None
    strict_valid_center_end: int | None = None
    # Fixed-horizon policies may need future Teacher support beyond a task's
    # terminal state.  CALVIN materializes an absorbing suffix: terminal
    # state/images repeat, relative arm commands are zero and the binary
    # gripper command is held.  ``terminal_state_index`` is the first index at
    # which actions are synthetic and the last real state/image index.
    terminal_state_index: int | None = None
    terminal_padding_mode: str = ""
    source_action_count: int | None = None
    # A causally realigned LIBERO episode replaces the legacy post-action
    # observation rows with pre-action rows.  Model-only continuation from E8
    # must nevertheless retain E8's exact affine state chart.  The converter
    # therefore stores the old real-row state array as a small numeric-only
    # reference; it is used only while fitting the state normalizer and never
    # enters a model sample.
    state_normalizer_reference_raw: np.ndarray | None = None
    state_normalizer_reference_semantics: str = ""
    # A CALVIN raw-overlay episode can extend the numeric action/state arrays
    # with a virtual absorbing terminal suffix while retaining an older,
    # verified cache for the real source prefix.  In that case cache readers
    # clamp only the virtual suffix to the cached terminal observation.  The
    # default keeps the historical one-cache-row-per-logical-row contract.
    cache_frame_count: int | None = None
    source_start: int | None = None
    source_end: int | None = None
    context_start: int | None = None
    source_trajectory_id: str = ""
    source_action_dim: int = 0
    source_state_dim: int = 0
    data_profile: str = "source_native"

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def cache_key(self) -> str:
        """Root-relative cache directory; flat data remains stem-compatible."""

        return self.episode_id

    @property
    def length(self) -> int:
        return int(self.actions_raw.shape[0])

    @property
    def cached_frame_count(self) -> int:
        return (
            int(self.length)
            if self.cache_frame_count is None
            else int(self.cache_frame_count)
        )

    def resolve_cached_frame_indices(self, indices: np.ndarray) -> np.ndarray:
        """Map logical visual rows onto a verified physical cache prefix.

        Only the explicit relative-action absorbing contract may reuse a
        shorter cache.  Its logical suffix repeats the terminal RGB/DINO row,
        so no visual information is fabricated and no stale row is silently
        admitted for another dataset type.
        """

        rows = np.asarray(indices, dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError(f"cache indices must be non-empty [H], got {rows.shape}")
        if int(rows.min()) < 0 or int(rows.max()) >= self.length:
            raise IndexError(
                f"logical cache indices [{int(rows.min())},{int(rows.max())}] "
                f"outside episode length={self.length}"
            )
        cached = self.cached_frame_count
        if cached == self.length:
            return rows
        if (
            self.terminal_padding_mode
            != RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
            or self.terminal_state_index is None
            or cached != int(self.terminal_state_index) + 1
            or not 0 < cached < self.length
        ):
            raise ValueError(
                "a shortened visual cache is valid only for an explicit "
                "relative-action absorbing terminal suffix"
            )
        return np.minimum(rows, cached - 1)


def find_hdf5_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {root} with glob={pattern!r}")
    return files


def _validate_episode_id(value: str) -> str:
    identity = str(value)
    pure = PurePosixPath(identity)
    if (
        not identity
        or pure.is_absolute()
        or "\\" in identity
        or not pure.parts
        or ":" in pure.parts[0]
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != identity
    ):
        raise ValueError(f"invalid root-relative episode identity: {identity!r}")
    return identity


def episode_identity(root: Path, path: Path) -> tuple[str, str, str]:
    """Return ``(root-relative id, source partition, task-local id)``."""

    relative = path.resolve().relative_to(root.resolve()).with_suffix("")
    identity = _validate_episode_id(relative.as_posix())
    pure = PurePosixPath(identity)
    # RDT data is partition/task/episode.  Existing flat datasets retain an
    # empty partition/task and exactly their historical stem as cache key.
    source_partition = pure.parts[0] if len(pure.parts) >= 3 else ""
    task_parts = pure.parts[1:-1] if source_partition else pure.parts[:-1]
    task_id = PurePosixPath(*task_parts).as_posix() if task_parts else ""
    return identity, source_partition, task_id


def decode_hdf5_instruction(value: object) -> str:
    """Decode the scalar UTF-8 instruction format used by RDT HDF5 files."""

    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        result = bytes(value).decode("utf-8")
    elif isinstance(value, str):
        result = value
    else:
        raise TypeError(f"instruction must be scalar UTF-8 text, got {type(value)!r}")
    if not result.strip():
        raise ValueError("instruction cannot be empty")
    return result


def load_hdf5_instruction(path: Path) -> str | None:
    """Read an optional scalar instruction from a dataset or root attribute."""

    with h5py.File(path, "r") as handle:
        instruction_dataset = handle.get("instruction")
        if isinstance(instruction_dataset, h5py.Dataset):
            return decode_hdf5_instruction(instruction_dataset[()])
        raw_instruction = handle.attrs.get("instruction")
        return (
            None
            if raw_instruction is None
            else normalize_instruction(decode_hdf5_instruction(raw_instruction))
        )


def load_episode(
    path: Path,
    *,
    episode_id: str | None = None,
    source_partition: str = "",
    task_id: str = "",
    cameras: tuple[str, ...],
    action_key: str = "action",
    action_state_key: str | None = None,
    state_key: str | None = None,
    camera_key_overrides: dict[str, str] | None = None,
) -> LoadedEpisode:
    overrides = camera_key_overrides or {}
    datasets = list_hdf5_datasets(str(path))
    resolved_action = resolve_key(datasets, action_key, ACTION_ALIASES, required=True)
    assert resolved_action is not None
    resolved_state = (
        resolve_key(datasets, state_key, STATE_ALIASES, required=True) if state_key else None
    )
    resolved_action_state = (
        resolve_key(
            datasets,
            action_state_key,
            ACTION_STATE_ALIASES,
            required=True,
        )
        if action_state_key
        else None
    )

    camera_keys: dict[str, str] = {}
    for camera in cameras:
        requested = overrides.get(camera)
        aliases = CAMERA_ALIASES.get(camera, ())
        if not aliases and requested is None:
            raise KeyError(
                f"Unknown camera name={camera!r}; provide an explicit camera key. "
                f"Known aliases={sorted(CAMERA_ALIASES)}"
            )
        key = resolve_key(
            datasets,
            requested,
            aliases,
            required=True,
        )
        assert key is not None
        camera_keys[camera] = key

    with h5py.File(path, "r") as f:
        actions = np.asarray(f[resolved_action], dtype=np.float32)
        states = (
            np.asarray(f[resolved_state], dtype=np.float32)
            if resolved_state is not None
            else actions.copy()
        )
        action_states = (
            np.asarray(f[resolved_action_state], dtype=np.float32)
            if resolved_action_state is not None
            else states.copy()
        )
        camera_lengths: dict[str, int] = {}
        for camera, key in camera_keys.items():
            dataset = f.get(key)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim < 1:
                raise TypeError(f"{path}: camera={camera!r} must be an HDF5 sequence")
            camera_lengths[camera] = int(dataset.shape[0])
        instruction_dataset = f.get("instruction")
        if isinstance(instruction_dataset, h5py.Dataset):
            instruction = decode_hdf5_instruction(instruction_dataset[()])
        else:
            raw_instruction = f.attrs.get("instruction")
            instruction = (
                None
                if raw_instruction is None
                else normalize_instruction(decode_hdf5_instruction(raw_instruction))
            )
        raw_language_key = f.attrs.get("language_key")
        if isinstance(raw_language_key, (bytes, np.bytes_)):
            raw_language_key = bytes(raw_language_key).decode("utf-8")
        language_key = None if raw_language_key is None else str(raw_language_key).strip()
        raw_task = f.attrs.get("task")
        if isinstance(raw_task, (bytes, np.bytes_)):
            raw_task = bytes(raw_task).decode("utf-8")
        if not str(task_id) and raw_task is not None:
            task_id = str(raw_task).strip()
        raw_valid_start = f.attrs.get("valid_center_start")
        raw_valid_end = f.attrs.get("valid_center_end")
        valid_center_start = None if raw_valid_start is None else int(raw_valid_start)
        valid_center_end = None if raw_valid_end is None else int(raw_valid_end)
        raw_strict_valid_start = f.attrs.get("strict_valid_center_start")
        raw_strict_valid_end = f.attrs.get("strict_valid_center_end")
        strict_valid_center_start = (
            None if raw_strict_valid_start is None else int(raw_strict_valid_start)
        )
        strict_valid_center_end = (
            None if raw_strict_valid_end is None else int(raw_strict_valid_end)
        )
        raw_terminal_index = f.attrs.get("terminal_state_index")
        terminal_state_index = (
            None if raw_terminal_index is None else int(raw_terminal_index)
        )
        raw_padding_mode = f.attrs.get("terminal_padding_mode", "")
        if isinstance(raw_padding_mode, (bytes, np.bytes_)):
            raw_padding_mode = bytes(raw_padding_mode).decode("utf-8")
        terminal_padding_mode = str(raw_padding_mode).strip()
        raw_source_action_count = f.attrs.get("source_action_count")
        source_action_count = (
            None if raw_source_action_count is None else int(raw_source_action_count)
        )
        raw_state_normalizer_reference_key = f.attrs.get(
            "state_normalizer_reference_key"
        )
        if isinstance(
            raw_state_normalizer_reference_key, (bytes, np.bytes_)
        ):
            raw_state_normalizer_reference_key = bytes(
                raw_state_normalizer_reference_key
            ).decode("utf-8")
        state_normalizer_reference_key = (
            ""
            if raw_state_normalizer_reference_key is None
            else str(raw_state_normalizer_reference_key).strip()
        )
        if state_normalizer_reference_key:
            reference_dataset = f.get(state_normalizer_reference_key)
            if not isinstance(reference_dataset, h5py.Dataset):
                raise ValueError(
                    f"{path}: state normalizer reference key does not name a dataset"
                )
            state_normalizer_reference_raw = np.asarray(
                reference_dataset, dtype=np.float32
            )
        else:
            state_normalizer_reference_raw = None
        raw_state_normalizer_reference_semantics = f.attrs.get(
            "state_normalizer_reference_semantics", ""
        )
        if isinstance(
            raw_state_normalizer_reference_semantics, (bytes, np.bytes_)
        ):
            raw_state_normalizer_reference_semantics = bytes(
                raw_state_normalizer_reference_semantics
            ).decode("utf-8")
        state_normalizer_reference_semantics = str(
            raw_state_normalizer_reference_semantics
        ).strip()
        raw_source_start = f.attrs.get("source_start")
        raw_source_end = f.attrs.get("source_end")
        raw_context_start = f.attrs.get("context_start")
        raw_source_trajectory_id = f.attrs.get("source_trajectory_id", "")
        if isinstance(raw_source_trajectory_id, (bytes, np.bytes_)):
            raw_source_trajectory_id = bytes(raw_source_trajectory_id).decode("utf-8")

    if actions.ndim != 2:
        raise ValueError(f"{path}: action must have shape [T,D], got {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError(f"{path}: empty action sequence")
    if not np.isfinite(actions).all():
        bad = np.argwhere(~np.isfinite(actions))
        raise ValueError(f"{path}: action contains non-finite values at {bad[:20].tolist()}")
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{path}: state must have shape [T,D] aligned with action, got {states.shape}"
        )
    if states.shape[1] != actions.shape[1]:
        raise ValueError(
            f"{path}: state/action dims must match for unified RDT token space, got {states.shape[1]} != {actions.shape[1]}"
        )
    if not np.isfinite(states).all():
        bad = np.argwhere(~np.isfinite(states))
        raise ValueError(f"{path}: state contains non-finite values at {bad[:20].tolist()}")
    if action_states.ndim != 2 or action_states.shape != actions.shape:
        raise ValueError(
            f"{path}: action_state must have shape [T,D] aligned with action, "
            f"got {action_states.shape}"
        )
    if not np.isfinite(action_states).all():
        bad = np.argwhere(~np.isfinite(action_states))
        raise ValueError(f"{path}: action_state contains non-finite values at {bad[:20].tolist()}")
    if instruction is not None:
        expected_language_key = instruction_key(instruction)
        if language_key is None:
            language_key = expected_language_key
        elif language_key != expected_language_key:
            raise ValueError(f"{path}: language_key does not identify its instruction")
    elif language_key is not None:
        raise ValueError(f"{path}: language_key exists without an instruction")
    if (valid_center_start is None) != (valid_center_end is None):
        raise ValueError(f"{path}: valid center bounds must be both present or both absent")
    if valid_center_start is not None and valid_center_end is not None:
        if not 0 <= valid_center_start <= valid_center_end < int(actions.shape[0]):
            raise ValueError(
                f"{path}: invalid center bounds "
                f"[{valid_center_start},{valid_center_end}] for T={actions.shape[0]}"
            )
    if (strict_valid_center_start is None) != (strict_valid_center_end is None):
        raise ValueError(
            f"{path}: strict valid center bounds must be both present or both absent"
        )
    if strict_valid_center_start is not None and strict_valid_center_end is not None:
        if not 0 <= strict_valid_center_start <= strict_valid_center_end < int(actions.shape[0]):
            raise ValueError(
                f"{path}: invalid strict center bounds "
                f"[{strict_valid_center_start},{strict_valid_center_end}] for T={actions.shape[0]}"
            )
    if (terminal_state_index is None) != (not terminal_padding_mode):
        raise ValueError(
            f"{path}: terminal_state_index and terminal_padding_mode must be declared together"
        )
    if terminal_state_index is not None:
        if terminal_padding_mode not in SUPPORTED_TERMINAL_PADDING_MODES:
            raise ValueError(
                f"{path}: unsupported terminal padding mode {terminal_padding_mode!r}"
            )
        if not 1 <= terminal_state_index < int(actions.shape[0]) - 1:
            raise ValueError(
                f"{path}: terminal state index {terminal_state_index} leaves no absorbing suffix"
            )
        if valid_center_end is not None and valid_center_end >= terminal_state_index:
            raise ValueError(
                f"{path}: valid center end must precede terminal state index"
            )
    if source_action_count is not None:
        if terminal_padding_mode != LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING:
            raise ValueError(
                f"{path}: source_action_count is reserved for LIBERO terminal replay"
            )
        if terminal_state_index != source_action_count:
            raise ValueError(
                f"{path}: LIBERO terminal_state_index must equal source_action_count"
            )
        if int(actions.shape[0]) != source_action_count + 48:
            raise ValueError(f"{path}: LIBERO terminal replay must append exactly 48 rows")
    if state_normalizer_reference_raw is not None:
        reference_count = (
            int(source_action_count)
            if source_action_count is not None
            else int(actions.shape[0])
        )
        if (
            state_normalizer_reference_raw.ndim != 2
            or state_normalizer_reference_raw.shape
            != (reference_count, int(states.shape[1]))
            or not np.isfinite(state_normalizer_reference_raw).all()
        ):
            raise ValueError(
                f"{path}: state normalizer reference must be finite "
                f"[{reference_count},{states.shape[1]}]"
            )
        if (
            state_normalizer_reference_semantics
            != LIBERO_E8_STATE_NORMALIZER_REFERENCE
        ):
            raise ValueError(
                f"{path}: state normalizer reference has unsupported semantics "
                f"{state_normalizer_reference_semantics!r}"
            )
    elif state_normalizer_reference_semantics:
        raise ValueError(
            f"{path}: state normalizer reference semantics exist without a dataset"
        )
    misaligned_cameras = {
        camera: length
        for camera, length in camera_lengths.items()
        if length != int(actions.shape[0])
    }
    if misaligned_cameras:
        raise ValueError(
            f"{path}: camera frame counts must align with action T={actions.shape[0]}, "
            f"got {misaligned_cameras}"
        )

    return LoadedEpisode(
        path=path,
        episode_id=_validate_episode_id(path.stem if episode_id is None else episode_id),
        source_partition=str(source_partition),
        task_id=str(task_id),
        action_key=resolved_action,
        camera_keys=camera_keys,
        actions_raw=actions,
        instruction=instruction,
        action_state_key=resolved_action_state,
        state_key=resolved_state,
        states_raw=states,
        action_states_raw=action_states,
        language_key=language_key,
        valid_center_start=valid_center_start,
        valid_center_end=valid_center_end,
        strict_valid_center_start=strict_valid_center_start,
        strict_valid_center_end=strict_valid_center_end,
        terminal_state_index=terminal_state_index,
        terminal_padding_mode=terminal_padding_mode,
        source_action_count=source_action_count,
        state_normalizer_reference_raw=state_normalizer_reference_raw,
        state_normalizer_reference_semantics=state_normalizer_reference_semantics,
        source_start=None if raw_source_start is None else int(raw_source_start),
        source_end=None if raw_source_end is None else int(raw_source_end),
        context_start=None if raw_context_start is None else int(raw_context_start),
        source_trajectory_id=str(raw_source_trajectory_id).strip(),
        source_action_dim=int(actions.shape[1]),
        source_state_dim=int(states.shape[1]),
    )


def load_episodes(
    root: Path,
    pattern: str,
    *,
    cameras: tuple[str, ...],
    min_length: int,
    action_key: str = "action",
    action_state_key: str | None = None,
    state_key: str | None = None,
    camera_key_overrides: dict[str, str] | None = None,
    episode_names: Sequence[str] | None = None,
) -> tuple[list[LoadedEpisode], list[tuple[str, str]]]:
    episodes: list[LoadedEpisode] = []
    skipped: list[tuple[str, str]] = []
    identity_paths: dict[str, Path] = {}
    candidates = []
    requested_values = (
        None if episode_names is None else tuple(str(value) for value in episode_names)
    )
    requested = None if requested_values is None else set(requested_values)
    if requested is not None and (not requested or len(requested) != len(requested_values)):
        raise ValueError("requested episode identities must be non-empty and unique")
    for path in find_hdf5_files(root, pattern):
        identity, source_partition, task_id = episode_identity(root, path)
        if requested is None or identity in requested:
            candidates.append((path, identity, source_partition, task_id))
    if requested is not None:
        observed = {identity for _path, identity, _partition, _task in candidates}
        missing = sorted(requested.difference(observed))
        if missing:
            raise FileNotFoundError(
                f"requested HDF5 episodes are absent from {root}: {missing[:8]}"
            )
    for path, identity, source_partition, task_id in candidates:
        try:
            ep = load_episode(
                path,
                episode_id=identity,
                source_partition=source_partition,
                task_id=task_id,
                cameras=cameras,
                action_key=action_key,
                action_state_key=action_state_key,
                state_key=state_key,
                camera_key_overrides=camera_key_overrides,
            )
        except Exception as exc:  # surfaced in CLI metadata
            skipped.append((str(path), repr(exc)))
            continue

        previous = identity_paths.get(ep.episode_id)
        if previous is not None:
            raise RuntimeError(
                "duplicate root-relative episode identity after removing the HDF5 suffix: "
                f"{ep.episode_id!r} maps to both {previous} and {path}"
            )
        identity_paths[ep.episode_id] = path
        if ep.length < min_length:
            skipped.append((str(path), f"too_short_T={ep.length}, min_length={min_length}"))
        else:
            episodes.append(ep)

    if not episodes:
        raise RuntimeError(f"No usable episodes. skipped examples={skipped[:5]}")
    return episodes, skipped


def resolve_too_short_episode_exclusions(
    root: Path,
    skipped: list[tuple[str, str]],
    *,
    expected_minimum_length: int,
) -> dict[str, int]:
    """Turn only exact typed-window length skips into manifest exclusions."""

    minimum_length = int(expected_minimum_length)
    if minimum_length <= 0:
        raise ValueError("expected minimum episode length must be positive")
    exclusions: dict[str, int] = {}
    unexpected: list[tuple[str, str]] = []
    for skipped_path, reason in skipped:
        match = re.fullmatch(r"too_short_T=(\d+), min_length=(\d+)", str(reason))
        if match is None or int(match.group(2)) != minimum_length:
            unexpected.append((str(skipped_path), str(reason)))
            continue
        identity, _, _ = episode_identity(Path(root), Path(skipped_path))
        if identity in exclusions:
            raise RuntimeError(f"duplicate skipped episode identity: {identity!r}")
        exclusions[identity] = int(match.group(1))
    if unexpected:
        raise RuntimeError(
            "formal RDT loading rejects malformed/unreadable episodes rather than "
            f"treating them as manifest exclusions: {unexpected[:5]}"
        )
    return exclusions


__all__ = [
    "LIBERO_E8_STATE_NORMALIZER_REFERENCE",
    "LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING",
    "RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING",
    "SUPPORTED_TERMINAL_PADDING_MODES",
    "LoadedEpisode",
    "decode_hdf5_instruction",
    "episode_identity",
    "find_hdf5_files",
    "load_hdf5_instruction",
    "load_episode",
    "load_episodes",
    "resolve_too_short_episode_exclusions",
]
