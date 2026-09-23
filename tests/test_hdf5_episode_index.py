from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.hdf5_index import (
    build_hdf5_episode_index,
    load_hdf5_episode_index,
    read_hdf5_rows,
    save_hdf5_episode_index,
)


def _write_episode(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        action = np.arange(30, dtype=np.float32).reshape(5, 6) + offset
        state = action + 100.0
        handle.create_dataset("action", data=action)
        handle.create_dataset("state", data=state)
        handle.create_dataset("observations/images/top", data=np.zeros((5, 2, 2, 3), dtype=np.uint8))
        handle.create_dataset("observations/images/wrist", data=np.zeros((5, 2, 2, 3), dtype=np.uint8))
        handle.attrs["instruction"] = "push the block"


def test_index_and_lazy_rows_match_eager_bytes(tmp_path: Path) -> None:
    _write_episode(tmp_path / "episode_b.hdf5", 100.0)
    _write_episode(tmp_path / "episode_a.hdf5", 0.0)
    requested = ("episode_b", "episode_a")

    entries = build_hdf5_episode_index(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        action_key="action",
        state_key="state",
        episode_names=requested,
    )
    assert [entry.episode_id for entry in entries] == ["episode_a", "episode_b"]
    index_path = tmp_path / "index.json"
    save_hdf5_episode_index(
        index_path, root=tmp_path, pattern="*.hdf5", entries=entries
    )
    root, restored = load_hdf5_episode_index(index_path)
    assert root == tmp_path
    assert [entry.episode_id for entry in restored] == ["episode_a", "episode_b"]

    eager, skipped = load_episodes(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        action_key="action",
        state_key="state",
        episode_names=requested,
    )
    assert not skipped
    for entry, episode in zip(restored, eager, strict=True):
        np.testing.assert_array_equal(
            read_hdf5_rows(root, entry, entry.action_key, [0, 2, 4]),
            np.asarray(episode.actions_raw)[[0, 2, 4]],
        )
        np.testing.assert_array_equal(
            read_hdf5_rows(root, entry, entry.state_key or entry.action_key, [1, 3]),
            np.asarray(episode.states_raw)[[1, 3]],
        )


def test_index_rejects_changed_episode(tmp_path: Path) -> None:
    episode = tmp_path / "episode.hdf5"
    _write_episode(episode, 0.0)
    entries = build_hdf5_episode_index(tmp_path, "*.hdf5", cameras=("top", "wrist"))
    index_path = tmp_path / "index.json"
    save_hdf5_episode_index(index_path, root=tmp_path, pattern="*.hdf5", entries=entries)
    with h5py.File(episode, "a") as handle:
        handle["action"][0, 0] = 999.0
    with pytest.raises(ValueError, match="changed since admission"):
        load_hdf5_episode_index(index_path)
