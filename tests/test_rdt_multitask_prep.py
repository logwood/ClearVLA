from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from clearvla.data.multitask_selection import (
    RDT_MULTITASK_SELECTION_SCHEMA,
    load_rdt_multitask_selection_manifest,
)
from clearvla.data.samplers import (
    InformationBalancedSamplerConfig,
    TaskBalancedInformationBatchSampler,
    TaskStratifiedBatchSampler,
)
from clearvla.mainline.config import (
    DataConfig,
    ExperimentConfig,
    ObjectiveConfig,
    load_config,
)
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.data.normalizer_artifact import (
    SHARED_NORMALIZER_SCHEMA,
    canonical_digest,
    load_shared_normalizers,
)
from clearvla.mainline.runtime.evaluation import ValidationAccumulator
from clearvla.mainline.runtime.multitask import TaskValidationAccumulators
from clearvla.tools.audit_rdt_multitask_gripper import (
    GRIPPER_AUDIT_SCHEMA,
    _true_run_lengths,
)


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


def test_gripper_audit_labels_quantiles_as_descriptive_only() -> None:
    assert GRIPPER_AUDIT_SCHEMA == "clearvla-rdt-multitask-gripper-train-audit-v3"


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


def test_task_balanced_information_sampler_covers_every_task_per_batch() -> None:
    task_index = np.repeat(np.arange(8, dtype=np.int64), 12)
    motion = np.linspace(0.0, 1.0, num=len(task_index), dtype=np.float32)
    events = np.arange(len(task_index)) % 5 == 0
    config = InformationBalancedSamplerConfig(
        batch_size=8,
        batches_per_epoch=6,
        seed=17,
    )
    first = TaskBalancedInformationBatchSampler(
        motion,
        events,
        task_index,
        tuple(f"task_{index}" for index in range(8)),
        config,
    )
    second = TaskBalancedInformationBatchSampler(
        motion,
        events,
        task_index,
        tuple(f"task_{index}" for index in range(8)),
        config,
    )

    first_batches = list(first)
    assert first_batches == list(second)
    for batch in first_batches:
        assert sorted(task_index[batch].tolist()) == list(range(8))
        assert len(batch) == len(set(batch)) == 8
    first.set_epoch(1)
    assert list(first) != first_batches


def test_task_stratified_validation_panel_is_equal_and_deterministic() -> None:
    task_index = np.repeat(np.arange(3, dtype=np.int64), (7, 11, 5))
    sampler = TaskStratifiedBatchSampler(
        task_index,
        ("a", "b", "c"),
        samples_per_task=4,
        batch_size=5,
    )
    rows = [row for batch in sampler for row in batch]

    assert len(rows) == 12
    assert np.bincount(task_index[rows], minlength=3).tolist() == [4, 4, 4]
    assert len(rows) == len(set(rows))
    assert sampler.summary["selected_samples_per_task"] == {"a": 4, "b": 4, "c": 4}


def test_rdt_threshold_binds_sampler_objective_and_validation_semantics() -> None:
    data = replace(
        DataConfig(),
        data_profile="rdt_right_arm_action_chart_v1",
        split_mode="manifest",
        split_manifest="split.json",
        task_selection_manifest="selection.json",
        normalizer_artifact="normalizer.json",
        train_episodes=0,
        val_episodes=0,
        test_episodes=0,
        sampling_gripper_event_threshold=0.2,
    )
    ExperimentConfig(
        data=data,
        objectives=replace(ObjectiveConfig(), gripper_event_threshold=0.2),
    ).validate()
    with pytest.raises(ValueError, match="must be identical"):
        ExperimentConfig(data=data).validate()
    with pytest.raises(ValueError, match="Pen gripper trajectory threshold"):
        ExperimentConfig(
            objectives=replace(ObjectiveConfig(), gripper_event_threshold=0.2)
        ).validate()


def test_multitask8_config_and_launcher_fail_closed_until_threshold_is_adopted() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/mainline/rdt_multitask8_data_v1.json")
    assert config.data.sampling_gripper_event_threshold is None
    # The generic objective default is retained for the Pen profile, but the
    # non-Pen manifest cannot consume it when the data-side threshold is null.
    assert config.objectives.gripper_event_threshold == 0.10
    launcher = (root / "scripts/train_rdt_multitask.sh").read_text(encoding="utf-8")
    assert 'RDT_GRIPPER_EVENT_THRESHOLD:?Set one explicitly adopted' in launcher
    assert "descriptive audit quantiles are not thresholds" in launcher
    assert "ADOPTED_RDT_GRIPPER_EVENT_THRESHOLD" not in launcher


def test_multitask_validation_reports_micro_macro_and_missing_coverage() -> None:
    normalizer = ArrayNormalizer.fit_identity(
        [np.asarray([[0.0] * 7, [1.0] * 7], dtype=np.float32)]
    )
    accumulators = TaskValidationAccumulators.from_action_normalizer(
        ("a", "b", "missing"),
        normalizer,
        device=torch.device("cpu"),
        gripper_event_threshold=0.1,
        arm_motion_threshold=0.02,
    )
    batch_size = 4
    horizon = 24
    target = torch.zeros(batch_size, horizon, 7)
    current = torch.zeros(batch_size, 7)
    batch = SimpleNamespace(
        action_target=SimpleNamespace(
            normalized=target,
            raw_units=target,
            current_raw_units=current,
            gripper_transition_boundary=current,
            gripper_transition_boundary_raw_units=current,
        ),
        online=SimpleNamespace(
            history=SimpleNamespace(action_state=current),
        ),
    )
    prediction = torch.cat(
        (
            torch.ones(2, horizon, 7),
            torch.full((2, horizon, 7), 3.0),
        ),
        dim=0,
    )
    tasks = torch.tensor([0, 0, 1, 1])
    accumulators.update(tasks, prediction, batch)
    micro = ValidationAccumulator.from_action_normalizer(
        normalizer,
        device=torch.device("cpu"),
        gripper_event_threshold=0.1,
        arm_motion_threshold=0.02,
    )
    micro.update(prediction, batch)
    report = accumulators.report(micro.means())

    assert report["observed_task_count"] == 2
    assert report["task_coverage"] == pytest.approx(2 / 3)
    assert report["missing_tasks"] == ["missing"]
    assert set(report["tasks"]) == {"a", "b"}
    assert report["tasks"]["a"]["validation_action_rmse_physical"] == pytest.approx(1.0)
    assert report["tasks"]["b"]["validation_action_rmse_physical"] == pytest.approx(3.0)
    assert report["micro"]["validation_action_rmse_physical"] == pytest.approx(5**0.5)
    assert report["macro"]["validation_action_rmse_physical"] == pytest.approx(2.0)
