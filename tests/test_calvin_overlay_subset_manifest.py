from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from scripts.build_calvin_overlay_subset_manifest import build_subset_manifest


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "cache"
    root.mkdir()
    splits = {
        "train": ["train_blue", "train_red"],
        "val": ["val_blue", "val_red"],
        "test": ["test_blue", "test_red"],
    }
    tasks = {
        name: ("push_blue_block_left" if name.endswith("blue") else "open_drawer")
        for names in splits.values()
        for name in names
    }
    for name, task in tasks.items():
        with h5py.File(root / f"{name}.hdf5", "w") as handle:
            handle.attrs["task"] = task
    manifest = tmp_path / "full.json"
    payload = {
        "schema": "clearvla-episode-splits-v1",
        "task_filter": "",
        "split_unit": "source-trajectory",
        "split_seed": 0,
        "train_validation_fraction": 0.1,
        "raw_source": "/data/calvin/raw",
        "cached_prefix_root": str(root.resolve()),
        "terminal_overlay": "relative-action-absorbing-v1",
        "terminal_padding_frames": 24,
        "task_order": ["open_drawer", "push_blue_block_left"],
        "task_counts": {
            split: {"open_drawer": 1, "push_blue_block_left": 1}
            for split in ("train", "val", "test")
        },
        "raw_episode_map": {name: f"raw_{name}" for name in tasks},
        "splits": splits,
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, root


def test_build_subset_preserves_original_membership_and_raw_map(tmp_path: Path) -> None:
    manifest, root = _fixture(tmp_path)
    result = build_subset_manifest(
        manifest, root, ["push_blue_block_left"]
    )

    assert result["task_filter"] == ""
    assert result["task_order"] == ["push_blue_block_left"]
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["selected_episode_count"] == 3
    assert result["splits"] == {
        "train": ["train_blue"],
        "val": ["val_blue"],
        "test": ["test_blue"],
    }
    assert result["raw_episode_map"] == {
        "train_blue": "raw_train_blue",
        "val_blue": "raw_val_blue",
        "test_blue": "raw_test_blue",
    }


def test_build_subset_rejects_a_task_absent_from_source_registry(tmp_path: Path) -> None:
    manifest, root = _fixture(tmp_path)
    with pytest.raises(ValueError, match="absent from the source manifest"):
        build_subset_manifest(manifest, root, ["push_pink_block_left"])
