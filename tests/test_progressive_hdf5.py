from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from clearvla.data.progressive_hdf5 import ProgressiveHDF5EpisodeIndex


def _write(path: Path, value: float) -> None:
    with h5py.File(path, "w") as handle:
        action = np.arange(24, dtype=np.float32).reshape(4, 6) + value
        handle.create_dataset("action", data=action)
        handle.create_dataset("state", data=action + 100)
        handle.create_dataset("observations/images/top", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
        handle.create_dataset("observations/images/wrist", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
        handle.attrs["instruction"] = "push the block"


def test_progressive_index_does_not_admit_tail_until_requested(tmp_path: Path) -> None:
    for name, value in (("episode_a", 0), ("episode_b", 10), ("episode_c", 20)):
        _write(tmp_path / f"{name}.hdf5", value)
    with ProgressiveHDF5EpisodeIndex(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        state_key="state",
        episode_names=("episode_c", "episode_a", "episode_b"),
        max_prefetch=1,
    ) as index:
        assert index.ready_count == 0
        assert index.prefetch((0, 1, 2)) == (0,)
        first = index.get(0)
        assert first.episode_id == "episode_a"
        assert index.ready_count == 1
        assert index.get(2).episode_id == "episode_c"
        assert index.ready_count == 2


def test_progressive_index_fails_closed_before_hdf5_open_for_missing_manifest(tmp_path: Path) -> None:
    _write(tmp_path / "episode_a.hdf5", 0)
    with ProgressiveHDF5EpisodeIndex(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        episode_names=("episode_a", "missing"),
    ) as index:
        with pytest.raises(FileNotFoundError):
            index.get(1)


def test_progressive_index_reuses_verified_disk_record(tmp_path: Path) -> None:
    _write(tmp_path / "episode_a.hdf5", 0)
    cache_root = tmp_path / "cache"
    with ProgressiveHDF5EpisodeIndex(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        state_key="state",
        episode_names=("episode_a",),
        cache_root=cache_root,
    ) as index:
        first = index.get(0)
    with ProgressiveHDF5EpisodeIndex(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        state_key="state",
        episode_names=("episode_a",),
        cache_root=cache_root,
    ) as index:
        second = index.get(0)
    assert second == first


def test_progressive_index_rejects_empty_camera_name(tmp_path: Path) -> None:
    _write(tmp_path / "episode_a.hdf5", 0)
    with pytest.raises(ValueError, match="camera names"):
        ProgressiveHDF5EpisodeIndex(
            tmp_path,
            "*.hdf5",
            cameras=("",),
            episode_names=("episode_a",),
        )
