from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from clearvla.tools.build_rdt_split_manifest import build_rdt_split_manifest


def _episode(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.zeros((1, 14), dtype=np.float32))


def test_rdt_per_task_manifest_is_disjoint_stable_and_partition_aware(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task, count in (("one", 1), ("two", 2), ("three", 3), ("many", 10)):
        for index in range(count):
            _episode(root / "rdt_data" / task / f"episode_{index}.hdf5")
    for index in range(2):
        _episode(root / "test" / "external" / f"episode_{index}.hdf5")

    first = build_rdt_split_manifest(root, seed=17)
    second = build_rdt_split_manifest(root, seed=17)
    changed = build_rdt_split_manifest(root, seed=18)

    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert first["episode_count"] == 18
    assert first["split_counts"] == {
        "train": 12,
        "val": 2,
        "test": 2,
        "external_test": 2,
    }
    splits = first["splits"]
    flattened = [episode for values in splits.values() for episode in values]
    assert len(flattened) == len(set(flattened)) == 18
    assert all(value.startswith("test/") for value in splits["external_test"])
    assert all(not value.startswith("test/") for value in splits["train"])
    assert {
        value.rsplit("/", 1)[0] for value in (*splits["val"], *splits["test"])
    }.issubset({value.rsplit("/", 1)[0] for value in splits["train"]})
    for task, count in (("one", 1), ("two", 2)):
        task_ids = {f"rdt_data/{task}/episode_{index}" for index in range(count)}
        assert task_ids.issubset(set(splits["train"]))
        assert task_ids.isdisjoint(set(splits["val"]))
        assert task_ids.isdisjoint(set(splits["test"]))


def test_rdt_manifest_rejects_an_unclassified_partition(tmp_path: Path) -> None:
    root = tmp_path / "rdt-ft-data"
    for index in range(3):
        _episode(root / "rdt_data" / "task" / f"episode_{index}.hdf5")
    _episode(root / "test" / "external" / "episode_0.hdf5")
    _episode(root / "mystery" / "task" / "episode_0.hdf5")

    try:
        build_rdt_split_manifest(root)
    except ValueError as exc:
        assert "unrecognized source partitions" in str(exc)
    else:
        raise AssertionError("unknown source partitions must fail closed")
