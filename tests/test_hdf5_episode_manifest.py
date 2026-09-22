from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import clearvla.data.hdf5_episode as hdf5_episode
from clearvla.data.hdf5_episode import _manifest_hdf5_candidates, load_episodes


def _write_episode(path: Path, length: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((length, 7), dtype=np.float32))
        handle.create_dataset(
            "observations/images/top",
            data=np.zeros((length, 2, 2, 3), dtype=np.uint8),
        )
        handle.create_dataset(
            "observations/images/wrist",
            data=np.zeros((length, 2, 2, 3), dtype=np.uint8),
        )


def test_manifest_direct_path_does_not_glob_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode(tmp_path / "episode_b.hdf5")
    _write_episode(tmp_path / "episode_a.hdf5")
    (tmp_path / "unrelated.hdf5").write_bytes(b"not an episode")

    def reject_glob(*_args: object, **_kwargs: object):
        raise AssertionError("exact manifest loading must not enumerate the root")

    monkeypatch.setattr(Path, "glob", reject_glob)
    episodes, skipped = load_episodes(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        episode_names=("episode_b", "episode_a"),
    )
    assert not skipped
    assert [episode.episode_id for episode in episodes] == ["episode_a", "episode_b"]


def test_manifest_direct_path_fails_closed_on_missing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_glob(*_args: object, **_kwargs: object):
        raise AssertionError("missing manifest identities must not trigger a scan")

    monkeypatch.setattr(Path, "glob", reject_glob)
    with pytest.raises(FileNotFoundError, match="missing"):
        load_episodes(
            tmp_path,
            "*.hdf5",
            cameras=("top", "wrist"),
            min_length=1,
            episode_names=("missing",),
        )


def test_manifest_direct_path_falls_back_for_nested_identity(tmp_path: Path) -> None:
    assert _manifest_hdf5_candidates(
        tmp_path, "*.hdf5", ("task/episode",)
    ) is None


def test_manifest_direct_path_keeps_exact_suffix(tmp_path: Path) -> None:
    _write_episode(tmp_path / "episode.h5")
    assert _manifest_hdf5_candidates(tmp_path, "*.h5", ("episode",)) == [
        tmp_path / "episode.h5"
    ]


def test_manifest_direct_path_matches_legacy_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_episode(tmp_path / "episode_b.hdf5", length=3)
    _write_episode(tmp_path / "episode_a.hdf5", length=2)
    requested = ("episode_b", "episode_a")

    direct, direct_skipped = load_episodes(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        episode_names=requested,
    )
    monkeypatch.setattr(hdf5_episode, "_manifest_hdf5_candidates", lambda *_args: None)
    legacy, legacy_skipped = load_episodes(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        episode_names=requested,
    )

    assert direct_skipped == legacy_skipped
    assert [episode.episode_id for episode in direct] == [
        episode.episode_id for episode in legacy
    ]
    for direct_episode, legacy_episode in zip(direct, legacy, strict=True):
        assert direct_episode.path == legacy_episode.path
        assert direct_episode.source_partition == legacy_episode.source_partition
        assert direct_episode.task_id == legacy_episode.task_id
        np.testing.assert_array_equal(
            direct_episode.actions_raw, legacy_episode.actions_raw
        )
