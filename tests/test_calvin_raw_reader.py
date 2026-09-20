from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from clearvla.benchmarks.calvin import convert_calvin
from clearvla.benchmarks.calvin_raw import (
    CalvinRawReader,
    virtualize_calvin_cached_prefix,
)
from clearvla.data.hdf5_episode import load_episode


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
            local = frame - source_start
            action = np.zeros((7,), dtype=np.float32)
            action[0] = local / 100.0
            action[-1] = -1.0 if local < 40 else 1.0
            robot = np.full((7,), local, dtype=np.float32)
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


def _converted_episode(root: Path, name: str):
    return load_episode(
        root / f"{name}.hdf5",
        cameras=("top", "wrist"),
        action_key="action",
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides={
            "top": "observations/images/cam_high",
            "wrist": "observations/images/cam_right_wrist",
        },
    )


def test_raw_reader_streams_frame_pattern_without_globbing_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_raw_reader_ignores_unrelated_indexed_archive_before_frame_chart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(Path, "iterdir", decoy_first)
    reader = CalvinRawReader(source, task_filter="open_drawer", split_seed=7)
    assert sum(len(reader.episodes(split)) for split in ("train", "val", "test")) == 5


def test_raw_reader_frame_pattern_fails_closed_on_missing_trajectory_endpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    (source / "training" / "episode_0000069.npz").unlink()

    with pytest.raises(FileNotFoundError, match="trajectory endpoint 69"):
        CalvinRawReader(source, task_filter="open_drawer", split_seed=7)


def test_raw_reader_matches_converted_segment_and_split(tmp_path: Path) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)

    reader = CalvinRawReader(
        source,
        task_filter="open_drawer",
        val_fraction=1.0 / 3.0,
        split_seed=7,
    )
    report = reader.report()
    assert report["episodes"] == {"train": 2, "val": 1, "test": 2}
    assert report["trajectory_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}

    converted_root = tmp_path / "converted"
    convert_calvin(
        source,
        converted_root,
        val_fraction=1.0 / 3.0,
        split_seed=7,
        task_filter="open_drawer",
    )
    manifest = json.loads((converted_root / "splits.json").read_text(encoding="utf-8"))
    raw_by_annotation = {
        episode.annotation.index: episode
        for split in ("train", "val", "test")
        for episode in reader.episodes(split)
    }
    for name in [name for split in manifest["splits"].values() for name in split]:
        converted = _converted_episode(converted_root, name)
        with h5py.File(converted_root / f"{name}.hdf5", "r") as handle:
            annotation_index = int(handle.attrs["source_annotation_index"])
        raw = raw_by_annotation[annotation_index]
        arrays = raw.arrays()
        np.testing.assert_array_equal(converted.actions_raw, arrays["action"])
        np.testing.assert_array_equal(converted.action_states_raw, arrays["action_state"])
        np.testing.assert_array_equal(converted.states_raw, arrays["state"])
        with h5py.File(converted_root / f"{name}.hdf5", "r") as handle:
            np.testing.assert_array_equal(handle["observations/images/cam_high"][:], arrays["rgb_static"])
            np.testing.assert_array_equal(handle["observations/images/cam_right_wrist"][:], arrays["rgb_gripper"])


def test_raw_reader_window_keeps_sparse_previous_command_and_support_rows(tmp_path: Path) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    reader = CalvinRawReader(source, task_filter="open_drawer", split_seed=7)
    episode = reader.episodes("train")[0]
    assert list(episode.valid_centers()) == list(range(24, 34))

    window = episode.window(33)
    assert window["history_state"].shape == (3, 7)
    assert window["executed_action_history"].shape == (8, 7)
    assert window["action"].shape == (48, 7)
    assert window["policy_action"].shape == (24, 7)
    assert window["future_state"].shape == (48, 7)
    assert window["future_rgb_static"].shape == (12, 2, 2, 3)
    assert window["future_rgb_gripper"].shape == (12, 2, 2, 3)

    # The current action-state is the command at center-1, not the command at
    # the previous selected history row (center-4).
    expected_current = episode.frame(32).rel_actions
    np.testing.assert_array_equal(window["action_state"], expected_current)
    np.testing.assert_array_equal(window["policy_action"][-1], episode.frame(56).rel_actions)
    np.testing.assert_array_equal(window["action"][24:, :6], np.zeros((24, 6), dtype=np.float32))


def test_raw_reader_rejects_short_prefix_before_virtual_episode_creation(tmp_path: Path) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    annotation_path = source / "training" / "lang_annotations" / "auto_lang_ann.npy"
    payload = np.load(annotation_path, allow_pickle=True).item()
    indices = np.asarray(payload["info"]["indx"], dtype=np.int64)
    indices[0, 0] = 18
    payload["info"]["indx"] = indices
    np.save(annotation_path, payload, allow_pickle=True)
    _raw_source_split(source / "validation", trajectories=2)

    reader = CalvinRawReader(source, task_filter="open_drawer", val_fraction=0.5)
    assert sum(len(reader.episodes(split)) for split in ("train", "val", "test")) == 4


def test_cached_prefix_overlay_restores_terminal_truth_without_copying_visual_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task_ABC_D"
    _raw_source_split(source / "training", trajectories=3)
    _raw_source_split(source / "validation", trajectories=2)
    reader = CalvinRawReader(source, task_filter="open_drawer", val_fraction=1.0 / 3.0, split_seed=7)
    converted_root = tmp_path / "converted"
    convert_calvin(
        source,
        converted_root,
        val_fraction=1.0 / 3.0,
        split_seed=7,
        task_filter="open_drawer",
    )
    manifest = json.loads((converted_root / "splits.json").read_text(encoding="utf-8"))
    prefixes = []
    raw_episode_map = {}
    expected_splits = {
        split: list(values) for split, values in manifest["splits"].items()
    }
    for name in [value for split in manifest["splits"].values() for value in split]:
        full = _converted_episode(converted_root, name)
        terminal = int(full.terminal_state_index)
        # Emulate the immutable converter-v1 prefix: real rows through the
        # annotated terminal state, with no terminal metadata or synthetic tail.
        prefixes.append(
            replace(
                full,
                actions_raw=full.actions_raw[: terminal + 1].copy(),
                states_raw=full.states_raw[: terminal + 1].copy(),
                action_states_raw=full.action_states_raw[: terminal + 1].copy(),
                terminal_state_index=None,
                terminal_padding_mode="",
                valid_center_end=None,
                cache_frame_count=None,
            )
        )
    # Exercise converter-v1's post-filter renumbering explicitly: cache ID
    # and raw annotation ID differ, while the immutable source metadata stays
    # the same.
    raw_name = prefixes[0].episode_id
    legacy_name = "legacy_filtered_000000"
    prefixes[0] = replace(prefixes[0], episode_id=legacy_name)
    for split, names in expected_splits.items():
        expected_splits[split] = [legacy_name if name == raw_name else name for name in names]
    raw_episode_map[legacy_name] = raw_name
    overlaid, report = virtualize_calvin_cached_prefix(
        prefixes,
        reader,
        expected_splits=expected_splits,
        raw_episode_map=raw_episode_map,
    )
    assert len(overlaid) == len(prefixes)
    first = overlaid[0]
    assert first.terminal_padding_mode
    assert first.cache_frame_count == terminal + 1
    assert first.length == terminal + 1 + 24
    np.testing.assert_array_equal(first.actions_raw[terminal:, :6], 0.0)
    np.testing.assert_array_equal(
        first.states_raw[terminal:],
        np.repeat(first.states_raw[terminal : terminal + 1], 25, axis=0),
    )
    np.testing.assert_array_equal(
        first.resolve_cached_frame_indices(
            np.asarray([terminal - 1, terminal, terminal + 1, first.length - 1])
        ),
        np.asarray([terminal - 1, terminal, terminal, terminal]),
    )
    assert report["cached_prefix_visual_reuse"] == "real_rows_plus_terminal_repeat_only"
