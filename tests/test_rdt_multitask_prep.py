from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from clearvla.data.multitask_selection import (
    RDT_MULTITASK_SELECTION_SCHEMA,
    load_rdt_multitask_selection_manifest,
)
from clearvla.mainline.config import DataConfig
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.data.normalizer_artifact import (
    SHARED_NORMALIZER_SCHEMA,
    canonical_digest,
    load_shared_normalizers,
)
from clearvla.tools.audit_rdt_multitask_gripper import _true_run_lengths


def _selection_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_task_selection_filters_internal_lanes_and_preserves_external_test(tmp_path) -> None:
    episode_names = [
        "rdt_data/task_a/episode_0",
        "rdt_data/task_a/episode_1",
        "rdt_data/task_a/episode_2",
        "test/external/episode_0",
    ]
    task_names = ["task_a", "task_a", "task_a", "external"]
    instructions = ["Do task A.", "Do task A.", "Do task A.", "External task."]
    base_splits = {
        "train": [0],
        "val": [1],
        "test": [2],
        "external_test": [3],
    }
    base_metadata = {
        "file_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "source_episode_inventory_sha256": "3" * 64,
        "episode_inventory_sha256": "4" * 64,
    }
    payload = {
        "schema": RDT_MULTITASK_SELECTION_SCHEMA,
        "task_order": ["task_a"],
        "policy": {
            "task_count": 1,
            "task_id_usage": "cpu_audit_sampling_logging_metadata_only",
            "model_conditioning": False,
            "episode_scope": "all_typed_window_eligible_episodes_per_selected_task",
            "external_test": "preserved_external_only_not_selected_or_tuned",
        },
        "base_split_manifest": base_metadata,
        "tasks": [
            {
                "task_id": "task_a",
                "instruction": "Do task A.",
                "instruction_sha256": hashlib.sha256(b"Do task A.").hexdigest(),
                "left_role_audit": {"required_support_or_collaboration": False},
                "splits": {
                    "train": [episode_names[0]],
                    "val": [episode_names[1]],
                    "test": [episode_names[2]],
                },
            }
        ],
        "splits": {
            "train": [episode_names[0]],
            "val": [episode_names[1]],
            "test": [episode_names[2]],
        },
        "external_test_identity": {
            "episode_count": 1,
            "episode_inventory_sha256": _selection_digest([episode_names[3]]),
            "membership": "base_manifest_external_test_only",
            "selected": False,
            "used_for_training_or_tuning": False,
        },
    }
    payload["selection_sha256"] = _selection_digest(payload)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    splits, metadata = load_rdt_multitask_selection_manifest(
        path,
        episode_names=episode_names,
        task_names=task_names,
        instructions=instructions,
        base_splits=base_splits,
        base_split_metadata=base_metadata,
        expected_task_count=1,
    )

    assert splits == base_splits
    assert metadata["model_conditioning"] is False
    assert metadata["task_id_usage"] == "cpu_audit_sampling_logging_metadata_only"
    assert metadata["external_test_identity"]["used_for_training_or_tuning"] is False


def test_shared_normalizer_artifact_is_recomputed_and_train_bound(tmp_path) -> None:
    action = ArrayNormalizer.fit_zscore(
        [np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)]
    )
    state = ArrayNormalizer.fit_zscore(
        [np.asarray([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)]
    )
    train_ids = ["rdt_data/task_a/episode_0"]
    payload = {
        "schema": SHARED_NORMALIZER_SCHEMA,
        "fit_scope": "one_shared_normalizer_over_selected_train_split_only",
        "per_task_normalizers": False,
        "selection_manifest": {"selection_sha256": "a" * 64},
        "action_profile": {"sha256": "b" * 64},
        "train_episode_count": 1,
        "train_episode_inventory_sha256": canonical_digest(train_ids),
        "action": action.to_dict(),
        "state": state.to_dict(),
    }
    payload["normalizer_sha256"] = canonical_digest(payload)
    path = tmp_path / "normalizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded_action, loaded_state, metadata = load_shared_normalizers(
        path,
        expected_selection_sha256="a" * 64,
        expected_profile_sha256="b" * 64,
        expected_train_episode_ids=train_ids,
        computed_action=action,
        computed_state=state,
    )

    assert loaded_action.to_dict() == action.to_dict()
    assert loaded_state.to_dict() == state.to_dict()
    assert metadata["per_task_normalizers"] is False
    assert metadata["train_episode_count"] == 1


def test_gripper_activity_run_lengths_are_episode_local() -> None:
    assert _true_run_lengths(np.asarray([], dtype=bool)).tolist() == []
    assert _true_run_lengths(np.asarray([False, True, True, False, True])).tolist() == [2, 1]


def test_bounded_selection_and_shared_normalizer_are_an_atomic_config_pair() -> None:
    manifest = replace(
        DataConfig(),
        split_mode="manifest",
        split_manifest="split.json",
        train_episodes=0,
        val_episodes=0,
        test_episodes=0,
    )
    with pytest.raises(ValueError, match="requires one shared normalizer"):
        replace(manifest, task_selection_manifest="selection.json").validate()
    with pytest.raises(ValueError, match="artifact cannot be configured"):
        replace(manifest, normalizer_artifact="normalizer.json").validate()
    replace(
        manifest,
        task_selection_manifest="selection.json",
        normalizer_artifact="normalizer.json",
    ).validate()
