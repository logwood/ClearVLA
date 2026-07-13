from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .schema import ACTION_ALIASES, CAMERA_ALIASES, STATE_ALIASES, list_hdf5_datasets, resolve_key


@dataclass
class LoadedEpisode:
    path: Path
    action_key: str
    camera_keys: dict[str, str]
    actions_raw: np.ndarray
    actions_norm: np.ndarray | None = None
    state_key: str | None = None
    states_raw: np.ndarray | None = None
    states_norm: np.ndarray | None = None

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def length(self) -> int:
        return int(self.actions_raw.shape[0])


def find_hdf5_files(root: Path, pattern: str) -> list[Path]:
    files = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {root} with glob={pattern!r}")
    return files


def load_episode(
    path: Path,
    *,
    cameras: tuple[str, ...],
    action_key: str = "action",
    state_key: str | None = None,
    camera_key_overrides: dict[str, str] | None = None,
) -> LoadedEpisode:
    overrides = camera_key_overrides or {}
    datasets = list_hdf5_datasets(str(path))
    resolved_action = resolve_key(datasets, action_key, ACTION_ALIASES, required=True)
    assert resolved_action is not None
    resolved_state = resolve_key(datasets, state_key, STATE_ALIASES, required=True) if state_key else None

    camera_keys: dict[str, str] = {}
    for camera in cameras:
        if camera not in CAMERA_ALIASES:
            raise KeyError(f"Unknown camera name={camera!r}. Known cameras={sorted(CAMERA_ALIASES)}")
        key = resolve_key(
            datasets,
            overrides.get(camera),
            CAMERA_ALIASES[camera],
            required=True,
        )
        assert key is not None
        camera_keys[camera] = key

    with h5py.File(path, "r") as f:
        actions = np.asarray(f[resolved_action], dtype=np.float32)
        states = np.asarray(f[resolved_state], dtype=np.float32) if resolved_state is not None else actions.copy()

    if actions.ndim != 2:
        raise ValueError(f"{path}: action must have shape [T,D], got {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError(f"{path}: empty action sequence")
    if not np.isfinite(actions).all():
        bad = np.argwhere(~np.isfinite(actions))
        raise ValueError(f"{path}: action contains non-finite values at {bad[:20].tolist()}")
    if states.ndim != 2 or states.shape[0] != actions.shape[0]:
        raise ValueError(f"{path}: state must have shape [T,D] aligned with action, got {states.shape}")
    if states.shape[1] != actions.shape[1]:
        raise ValueError(f"{path}: state/action dims must match for unified RDT token space, got {states.shape[1]} != {actions.shape[1]}")
    if not np.isfinite(states).all():
        bad = np.argwhere(~np.isfinite(states))
        raise ValueError(f"{path}: state contains non-finite values at {bad[:20].tolist()}")

    return LoadedEpisode(
        path=path,
        action_key=resolved_action,
        camera_keys=camera_keys,
        actions_raw=actions,
        state_key=resolved_state,
        states_raw=states,
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
    for path in find_hdf5_files(root, pattern):
        try:
            ep = load_episode(
                path,
                cameras=cameras,
                action_key=action_key,
                state_key=state_key,
                camera_key_overrides=camera_key_overrides,
            )
            if ep.length < min_length:
                skipped.append((str(path), f"too_short_T={ep.length}, min_length={min_length}"))
            else:
                episodes.append(ep)
        except Exception as exc:  # surfaced in CLI metadata
            skipped.append((str(path), repr(exc)))

    if not episodes:
        raise RuntimeError(f"No usable episodes. skipped examples={skipped[:5]}")
    return episodes, skipped
