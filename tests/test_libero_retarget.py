from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from clearvla.benchmarks.libero_retarget import (
    apply_free_joint_translation,
    corrected_osc_action,
    first_sustained,
    gripper_phase_indices,
    translation_schedule,
)
from clearvla.data.action_chart import project_episodes, resolve_action_state_profile
from clearvla.data.hdf5_episode import (
    LIBERO_E8_STATE_NORMALIZER_REFERENCE,
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    load_episodes,
)
from clearvla.data.libero_retarget import (
    LIBERO_RETARGET_EPISODE_SCHEMA,
    LIBERO_RETARGET_NORMALIZER_POLICY,
    LIBERO_RETARGET_OVERLAY_SCHEMA,
    load_libero_retarget_overlay_contract,
    merge_libero_retarget_training_overlay,
)
from clearvla.mainline.data.loading import _normalizers


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_causal_episode(
    path: Path,
    *,
    source_episode_id: str | None = None,
    translation_y: float = 0.0,
) -> None:
    real_count = 80
    total = real_count + 48
    rows = np.arange(total, dtype=np.float32)[:, None]
    actions = np.zeros((total, 7), dtype=np.float32)
    actions[:real_count, :6] = rows[:real_count] / 400.0
    actions[:real_count, 1] += float(translation_y)
    actions[real_count:, 6] = actions[real_count - 1, 6]
    states = actions.copy()
    action_state = np.concatenate(
        (np.zeros((1, 7), dtype=np.float32), actions[:-1]), axis=0
    )
    with h5py.File(path, "w") as stream:
        stream.attrs["instruction"] = "put the black bowl on the plate"
        stream.attrs["task"] = "libero_spatial/task0"
        stream.attrs["converter_schema"] = (
            "clearvla-libero-terminal-replay-converter-v2"
        )
        stream.attrs["state_normalizer_reference_key"] = (
            "normalizer_reference_state"
        )
        stream.attrs["state_normalizer_reference_semantics"] = (
            LIBERO_E8_STATE_NORMALIZER_REFERENCE
        )
        stream.attrs["valid_center_start"] = 0
        stream.attrs["valid_center_end"] = real_count - 1
        stream.attrs["strict_valid_center_start"] = 24
        stream.attrs["strict_valid_center_end"] = real_count - 49
        stream.attrs["terminal_state_index"] = real_count
        stream.attrs["source_action_count"] = real_count
        stream.attrs["terminal_padding_mode"] = (
            LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
        )
        if source_episode_id is not None:
            stream.attrs["retarget_schema"] = LIBERO_RETARGET_EPISODE_SCHEMA
            stream.attrs["retarget_source_episode_id"] = source_episode_id
            stream.attrs["retarget_translation_m"] = np.asarray(
                [0.0, translation_y, 0.0], dtype=np.float64
            )
            stream.attrs["retarget_success"] = True
        stream.create_dataset("action", data=actions)
        stream.create_dataset("action_state", data=action_state)
        stream.create_dataset("state", data=states)
        stream.create_dataset(
            "normalizer_reference_state", data=states[:real_count]
        )
        images = stream.require_group("observations/images")
        images.create_dataset(
            "cam_high", data=np.zeros((total, 2, 3, 3), dtype=np.uint8)
        )
        images.create_dataset(
            "cam_right_wrist", data=np.ones((total, 2, 3, 3), dtype=np.uint8)
        )


def _retarget_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_root = tmp_path / "base"
    overlay_root = tmp_path / "overlay"
    base_root.mkdir()
    overlay_root.mkdir()
    for episode_id in ("source_0", "validation_0", "test_0"):
        _write_causal_episode(base_root / f"{episode_id}.hdf5")
    split = base_root / "splits.json"
    split.write_text(
        json.dumps(
            {
                "schema": "clearvla-episode-splits-v1",
                "splits": {
                    "train": ["source_0"],
                    "val": ["validation_0"],
                    "test": ["test_0"],
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_manifest = base_root / "dataset_manifest.json"
    dataset_manifest.write_text('{"fixture":true}', encoding="utf-8")
    rows = []
    episode_ids = []
    for suffix, translation_y in (("negative", -0.03), ("positive", 0.03)):
        episode_id = f"source_0__retarget_{suffix}"
        path = overlay_root / f"{episode_id}.hdf5"
        _write_causal_episode(
            path,
            source_episode_id="source_0",
            translation_y=translation_y,
        )
        episode_ids.append(episode_id)
        rows.append(
            {
                "episode_id": episode_id,
                "file": path.name,
                "file_sha256": _sha256(path),
                "source_episode_id": "source_0",
                "source_split": "train",
                "success": True,
                "translation_m": [0.0, translation_y, 0.0],
            }
        )
    manifest = overlay_root / "overlay_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": LIBERO_RETARGET_OVERLAY_SCHEMA,
                "scope": (
                    "train-only simulator retarget overlay; validation/test unchanged"
                ),
                "data_profile": "libero_relative_7d_v1",
                "window_boundary_contract": "causal_prefix_terminal_suffix_v2",
                "normalizer_policy": LIBERO_RETARGET_NORMALIZER_POLICY,
                "base_split_manifest_sha256": _sha256(split),
                "base_dataset_manifest_sha256": _sha256(dataset_manifest),
                "episode_count": 2,
                "pair_count": 1,
                "translations_m": [[0.0, -0.03, 0.0], [0.0, 0.03, 0.0]],
                "episodes": rows,
                "pairs": [
                    {
                        "source_episode_id": "source_0",
                        "episode_ids": episode_ids,
                        "admitted": True,
                        "geometry": {
                            "negative_response_sign_correct": True,
                            "positive_response_sign_correct": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return base_root, split, manifest


def test_sustained_gripper_phases_are_ordered() -> None:
    actions = np.zeros((20, 7), dtype=np.float64)
    actions[5:12, 6] = 1.0
    actions[14:, 6] = -1.0
    assert first_sustained(actions[:, 6] > 0.1, rows=3) == 5
    assert gripper_phase_indices(actions) == (5, 14)


def test_gripper_phase_rejects_missing_release() -> None:
    actions = np.zeros((20, 7), dtype=np.float64)
    actions[5:, 6] = 1.0
    with pytest.raises(ValueError, match="no sustained release"):
        gripper_phase_indices(actions)


def test_translation_schedule_preserves_reset_and_receptacle() -> None:
    shift = np.array([0.0, 0.03, 0.0], dtype=np.float64)
    schedule = translation_schedule(
        action_count=30,
        close_index=10,
        release_index=24,
        translation_m=shift,
        settle_fraction=0.25,
    )
    assert schedule.shape == (31, 3)
    assert np.array_equal(schedule[0], np.zeros(3))
    assert np.allclose(schedule[10], shift, rtol=0.0, atol=1.0e-12)
    assert np.array_equal(schedule[24], np.zeros(3))
    assert np.array_equal(schedule[30], np.zeros(3))
    assert np.all(np.diff(schedule[:11, 1]) >= 0.0)
    assert np.all(np.diff(schedule[14:25, 1]) <= 0.0)


def test_free_joint_translation_accounts_for_flat_state_prefix() -> None:
    state = np.arange(20, dtype=np.float64)
    shifted = apply_free_joint_translation(
        state,
        qpos_flat_offset=1,
        qpos_address=4,
        qpos_size=15,
        translation_m=(0.01, -0.02, 0.03),
    )
    expected = state.copy()
    expected[5:8] += np.array([0.01, -0.02, 0.03])
    assert np.array_equal(shifted, expected)
    assert np.array_equal(state, np.arange(20, dtype=np.float64))


def test_corrected_action_is_baseline_preserving_and_metric() -> None:
    original = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 1.0])
    source = np.array([0.1, 0.2, 0.3])
    unchanged, raw_unchanged = corrected_osc_action(
        original,
        source_pre_xyz_m=source,
        actual_pre_xyz_m=source,
        current_translation_m=np.zeros(3),
        next_translation_m=np.zeros(3),
    )
    assert np.array_equal(raw_unchanged, original)
    assert np.array_equal(unchanged, original.astype(np.float32))

    corrected, raw = corrected_osc_action(
        original,
        source_pre_xyz_m=source,
        actual_pre_xyz_m=source + np.array([0.0, -0.01, 0.0]),
        current_translation_m=np.zeros(3),
        next_translation_m=np.array([0.0, 0.005, 0.0]),
    )
    # 15 mm of y correction at 50 mm per normalized unit is +0.3.
    assert raw[1] == pytest.approx(0.2)
    assert corrected[1] == pytest.approx(0.2)
    assert np.array_equal(corrected[3:], original[3:].astype(np.float32))


def test_corrected_action_reports_preclip_saturation() -> None:
    original = np.zeros(7, dtype=np.float64)
    corrected, raw = corrected_osc_action(
        original,
        source_pre_xyz_m=np.zeros(3),
        actual_pre_xyz_m=np.zeros(3),
        current_translation_m=np.zeros(3),
        next_translation_m=np.array([0.10, 0.0, 0.0]),
    )
    assert raw[0] == pytest.approx(2.0)
    assert corrected[0] == pytest.approx(1.0)


def test_retarget_overlay_is_train_only_and_keeps_base_normalizers(
    tmp_path: Path,
) -> None:
    base_root, split, manifest = _retarget_fixture(tmp_path)
    camera_overrides = {
        "top": "observations/images/cam_high",
        "wrist": "observations/images/cam_right_wrist",
    }
    base, skipped = load_episodes(
        base_root,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides=camera_overrides,
    )
    assert skipped == []
    by_id = {episode.episode_id: index for index, episode in enumerate(base)}
    splits = {
        "train": [by_id["source_0"]],
        "val": [by_id["validation_0"]],
        "test": [by_id["test_0"]],
    }
    contract = load_libero_retarget_overlay_contract(
        manifest.parent,
        manifest,
        base_split_manifest=split,
        base_causal_root=base_root,
        base_train_episode_ids=["source_0"],
    )
    overlay, overlay_skipped = load_episodes(
        manifest.parent,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=1,
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides=camera_overrides,
        episode_names=contract.episode_ids,
    )
    assert overlay_skipped == []
    profile = resolve_action_state_profile("libero_relative_7d_v1")
    projected_base = project_episodes(base, profile)
    base_action, base_state = _normalizers(
        projected_base,
        splits["train"],
        mode="zscore",
    )
    merge = merge_libero_retarget_training_overlay(
        projected_base,
        splits,
        project_episodes(overlay, profile),
        contract,
    )

    assert merge.splits["val"] == tuple(splits["val"])
    assert merge.splits["test"] == tuple(splits["test"])
    assert merge.splits["train"][:1] == tuple(splits["train"])
    assert merge.splits["train"][1:] == merge.overlay_episode_indices
    assert merge.metadata["validation_test_membership_unchanged"] is True
    assert [merge.episodes[index].episode_id for index in merge.overlay_episode_indices] == list(
        contract.episode_ids
    )

    # The loader fits before this merge and retains the immutable base indices.
    # Re-evaluating that exact ownership proves the affine chart is bit-identical.
    merged_action, merged_state = _normalizers(
        list(merge.episodes),
        splits["train"],
        mode="zscore",
    )
    assert merged_action.to_dict() == base_action.to_dict()
    assert merged_state.to_dict() == base_state.to_dict()


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("hdf5_digest", "episode digest differs"),
        ("validation_lineage", "source is not in the base train split"),
        ("broken_pair", "references unknown/repeated episodes"),
    ),
)
def test_retarget_overlay_contract_fails_closed(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    base_root, split, manifest = _retarget_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if defect == "hdf5_digest":
        episode_path = manifest.parent / payload["episodes"][0]["file"]
        with h5py.File(episode_path, "r+") as stream:
            stream["action"][0, 0] = 0.75
    elif defect == "validation_lineage":
        payload["episodes"][0]["source_episode_id"] = "validation_0"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    elif defect == "broken_pair":
        first = payload["pairs"][0]["episode_ids"][0]
        payload["pairs"][0]["episode_ids"] = [first, first]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(defect)

    with pytest.raises(ValueError, match=message):
        load_libero_retarget_overlay_contract(
            manifest.parent,
            manifest,
            base_split_manifest=split,
            base_causal_root=base_root,
            base_train_episode_ids=["source_0"],
        )
