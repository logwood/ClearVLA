from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from clearvla.data.cache_selection import load_cache_episode_selection
from clearvla.data.hdf5_episode import episode_identity, find_hdf5_files
from clearvla.data.split import RDT_SPLIT_NAMES, load_rdt_split_manifest
from clearvla.tools.build_rdt_split_manifest import build_rdt_split_manifest


def _episode(path: Path, *, length: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        values = np.zeros((length, 14), dtype=np.float32)
        handle.create_dataset("action", data=values)
        handle.create_dataset("observations/qpos", data=values)
        handle.create_dataset(
            "observations/images/cam_high",
            data=np.zeros((length, 2, 2, 3), dtype=np.uint8),
        )


def test_rdt_per_task_manifest_is_disjoint_stable_and_partition_aware(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task, count in (("one", 1), ("two", 2), ("three", 3), ("many", 10)):
        for index in range(count):
            _episode(root / "rdt_data" / task / f"episode_{index}.hdf5")
    for index in range(2):
        _episode(root / "test" / "external" / f"episode_{index}.hdf5")
    _episode(root / "rdt_data" / "many" / "episode_short.hdf5", length=30)

    first = build_rdt_split_manifest(root, seed=17)
    second = build_rdt_split_manifest(root, seed=17)
    changed = build_rdt_split_manifest(root, seed=18)

    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest_sha256"] != changed["manifest_sha256"]
    assert first["source_episode_count"] == 19
    assert first["episode_count"] == 18
    assert first["minimum_episode_length"] == 73
    assert first["excluded_too_short"] == [
        {"episode_id": "rdt_data/many/episode_short", "length": 30}
    ]
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


def test_cache_selection_verifies_full_manifest_before_bounding_lane(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task in ("a", "b"):
        for index in range(3):
            _episode(root / "rdt_data" / task / f"episode_{index}.hdf5")
    _episode(root / "rdt_data" / "a" / "episode_short.hdf5", length=30)
    _episode(root / "test" / "external" / "episode_0.hdf5")
    payload = build_rdt_split_manifest(root, seed=11)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    selected = load_cache_episode_selection(
        root,
        "**/*.hdf5",
        cameras=("high",),
        action_key="action",
        state_key="observations/qpos",
        camera_key_overrides={"high": "observations/images/cam_high"},
        split_manifest=manifest,
        manifest_split="val",
        max_episodes=1,
        allow_skipped=False,
    )

    assert selected.eligible_episode_count == 7
    assert selected.selected_episode_count_before_limit == 2
    assert [episode.episode_id for episode in selected.episodes] == [
        payload["splits"]["val"][0]
    ]
    assert selected.skipped[0][1] == "too_short_T=30, min_length=73"
    assert selected.manifest_metadata is not None
    assert selected.manifest_metadata["source_episode_count"] == 8


def test_rdt_manifest_uses_identity_order_not_path_component_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task in ("write_board_1", "write_board_1+1"):
        for index in range(3):
            _episode(root / "rdt_data" / task / f"episode_{index}.hdf5")
    _episode(root / "test" / "external" / "episode_0.hdf5")

    path_order_names = [
        episode_identity(root, path)[0]
        for path in find_hdf5_files(root, "**/*.hdf5")
    ]
    assert path_order_names != sorted(path_order_names)

    payload = build_rdt_split_manifest(root, seed=23)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    # Simulate an arbitrary machine-local discovery order.  Accepted split
    # indices must still address the caller's original episode sequence.
    loaded_names = list(reversed(path_order_names))
    split_indices, _ = load_rdt_split_manifest(
        manifest,
        episode_names=loaded_names,
        expected_pattern="**/*.hdf5",
    )
    for split in RDT_SPLIT_NAMES:
        assert [loaded_names[index] for index in split_indices[split]] == payload[
            "splits"
        ][split]

    selected = load_cache_episode_selection(
        root,
        "**/*.hdf5",
        cameras=("high",),
        action_key="action",
        state_key="observations/qpos",
        camera_key_overrides={"high": "observations/images/cam_high"},
        split_manifest=manifest,
        manifest_split="val",
        max_episodes=1,
        allow_skipped=False,
    )
    assert [episode.episode_id for episode in selected.episodes] == [
        payload["splits"]["val"][0]
    ]
