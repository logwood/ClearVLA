from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clearvla.benchmarks.calvin_raw import CalvinRawReader


def _raw_source_split(root: Path, *, trajectories: int) -> None:
    root.mkdir(parents=True)
    source_ranges = []
    annotation_indices = []
    annotations = []
    tasks = []
    for trajectory in range(trajectories):
        source_start = trajectory * 70
        source_end = source_start + 69
        source_ranges.append((source_start, source_end))
        annotation_indices.append((source_start + 24, source_start + 57))
        annotations.append("open the drawer")
        tasks.append("open_drawer")
        for frame in range(source_start, source_end + 1):
            action = np.zeros((7,), dtype=np.float32)
            robot = np.full((7,), frame, dtype=np.float32)
            pixel = np.full((2, 2, 3), frame % 251, dtype=np.uint8)
            np.savez(
                root / f"episode_{frame:07d}.npz",
                rel_actions=action,
                robot_obs=robot,
                rgb_static=pixel,
                rgb_gripper=pixel,
            )
    np.save(root / "ep_start_end_ids.npy", np.asarray(source_ranges, dtype=np.int64))
    language = root / "lang_annotations"
    language.mkdir()
    np.save(
        language / "auto_lang_ann.npy",
        {
            "info": {"indx": np.asarray(annotation_indices, dtype=np.int64)},
            "language": {"ann": annotations, "task": tasks},
        },
        allow_pickle=True,
    )


def test_frame_discovery_does_not_use_full_glob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    original_glob = Path.glob
    original_iterdir = Path.iterdir
    split_roots = {source / "training", source / "validation"}

    def reject_frame_glob(path: Path, pattern: str):
        if pattern in {"*.npz", "*.pkl"}:
            raise AssertionError("raw frame discovery must not glob the full inventory")
        return original_glob(path, pattern)

    def expose_one_frame(path: Path):
        if path in split_roots:
            return iter((path / "episode_0000000.npz",))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "glob", reject_frame_glob)
    monkeypatch.setattr(Path, "iterdir", expose_one_frame)
    reader = CalvinRawReader(source, task_filter="open_drawer", split_seed=7)
    assert sum(len(reader.episodes(split)) for split in ("train", "val", "test")) == 5


def test_frame_discovery_ignores_unrelated_indexed_archive(tmp_path: Path) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    decoys = {
        root / "metadata_2026.npz"
        for root in (source / "training", source / "validation")
    }
    for decoy in decoys:
        np.savez(decoy, marker=np.asarray([1], dtype=np.int64))
    original_iterdir = Path.iterdir

    def decoy_first(path: Path):
        entries = tuple(original_iterdir(path))
        return iter(sorted(entries, key=lambda entry: (entry not in decoys, entry.name)))

    # The implementation must reject the decoy by endpoint coverage, not by
    # assuming that the first lexicographic numeric file is a frame.
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(Path, "iterdir", decoy_first)
        reader = CalvinRawReader(source, task_filter="open_drawer", split_seed=7)
        assert sum(len(reader.episodes(split)) for split in ("train", "val", "test")) == 5
    finally:
        monkeypatch.undo()


def test_frame_discovery_fails_closed_on_missing_endpoint(tmp_path: Path) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    (source / "training" / "episode_0000069.npz").unlink()

    with pytest.raises(FileNotFoundError, match="trajectory endpoint 69"):
        CalvinRawReader(source, task_filter="open_drawer", split_seed=7)
