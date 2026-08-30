from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

from .schema import ACTION_ALIASES, CAMERA_ALIASES, STATE_ALIASES, list_hdf5_datasets, resolve_key


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
    state_key: str | None = None
    states_raw: np.ndarray | None = None
    states_norm: np.ndarray | None = None
    # ``action_states_raw`` is observed qpos expressed in the command chart.
    # It equals ``states_raw`` for legacy data and is explicitly converted by
    # an RDT profile when qpos/action gripper source scales differ.
    action_states_raw: np.ndarray | None = None
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


def load_episode(
    path: Path,
    *,
    episode_id: str | None = None,
    source_partition: str = "",
    task_id: str = "",
    cameras: tuple[str, ...],
    action_key: str = "action",
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
        instruction_dataset = f.get("instruction")
        instruction = (
            decode_hdf5_instruction(instruction_dataset[()])
            if isinstance(instruction_dataset, h5py.Dataset)
            else None
        )

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

    return LoadedEpisode(
        path=path,
        episode_id=_validate_episode_id(path.stem if episode_id is None else episode_id),
        source_partition=str(source_partition),
        task_id=str(task_id),
        action_key=resolved_action,
        camera_keys=camera_keys,
        actions_raw=actions,
        instruction=instruction,
        state_key=resolved_state,
        states_raw=states,
        action_states_raw=states,
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
    state_key: str | None = None,
    camera_key_overrides: dict[str, str] | None = None,
) -> tuple[list[LoadedEpisode], list[tuple[str, str]]]:
    episodes: list[LoadedEpisode] = []
    skipped: list[tuple[str, str]] = []
    identity_paths: dict[str, Path] = {}
    for path in find_hdf5_files(root, pattern):
        try:
            identity, source_partition, task_id = episode_identity(root, path)
            ep = load_episode(
                path,
                episode_id=identity,
                source_partition=source_partition,
                task_id=task_id,
                cameras=cameras,
                action_key=action_key,
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


__all__ = [
    "LoadedEpisode",
    "decode_hdf5_instruction",
    "episode_identity",
    "find_hdf5_files",
    "load_episode",
    "load_episodes",
]
