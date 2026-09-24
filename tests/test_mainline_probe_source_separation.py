"""Raw-source holdout admission: real converter/loader and declared fixtures."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_mainline_annotation_endpoint_sources import _converted, _load
from test_mainline_annotation_goal import _batch

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.data.split import EPISODE_SPLIT_MANIFEST_SCHEMA, load_episode_split_manifest
from clearvla.mainline.data.dataset import ObservedWindowRef
from clearvla.mainline.runtime.probe_selection import (
    audit_calvin_holdout,
    batch_support_report,
    select_probe_indices,
)


def _ep(name: str, group: int, start: int = 0, *, partition: str = "training") -> LoadedEpisode:
    return LoadedEpisode(
        path=Path(name + ".hdf5"), episode_id=name, source_partition="", task_id="test",
        action_key="action", camera_keys={}, actions_raw=np.zeros((100, 7), np.float32),
        data_profile="calvin_relative_7d_v1", source_trajectory_id=f"{partition}:{group:06d}:{start}-{start+100}",
        context_start=start, source_start=start+20, source_end=start+90, source_annotation_index=group,
    )


def _audit(episodes, splits=None, metadata=None):
    return audit_calvin_holdout(
        episodes, {"train": (0,), "val": (1,)} if splits is None else splits,
        {"source": "fresh_train_only_fit", "train_episode_count": 1} if metadata is None else metadata,
    )


def test_real_converter_loader_supplies_source_identity_not_flat_directory(tmp_path: Path):
    episode = _load(_converted(tmp_path))
    assert not episode.source_partition
    episode = replace(episode, data_profile="calvin_relative_7d_v1")
    other = _ep("different", 2, 2000)
    report = _audit([episode, other])
    assert report["status"] == "passed"
    assert report["trajectory_counts"] == {"train": 1, "val": 1}
    json.dumps(report, allow_nan=False)


def test_episode_manifest_accepts_different_annotations_but_probe_detects_shared_source(tmp_path: Path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"schema": EPISODE_SPLIT_MANIFEST_SCHEMA, "splits": {"train": ["a"], "val": ["b"], "test": ["c"]}}))
    a, b, c, _ = load_episode_split_manifest(manifest, episode_names=["a", "b", "c"])
    episodes = [_ep("a", 0), _ep("b", 0), _ep("c", 2, 300)]
    report = _audit(episodes, {"train": a, "val": b, "test": c})
    assert report["status"] == "blocked"
    assert any(p["kind"] == "shared_raw_trajectory" for p in report["problems"])


def test_nonoverlapping_annotations_from_one_trajectory_remain_group_leakage():
    a = replace(_ep("a", 0), source_end=40)
    b = replace(_ep("b", 0), context_start=50, source_start=60)
    assert _audit([a, b])["status"] == "blocked"


@pytest.mark.parametrize("offset", [0, 50, 100])
def test_renamed_ids_cannot_hide_raw_interval_overlap_or_shared_boundary(offset: int):
    report = _audit([_ep("a", 0), _ep("b", 1, offset)])
    assert any(p["kind"] == "overlapping_raw_trajectories" for p in report["problems"])


def test_same_numeric_frames_in_distinct_raw_partitions_are_not_overlap():
    assert _audit([_ep("a", 0), _ep("b", 0, partition="validation")])["status"] == "passed"


@pytest.mark.parametrize("field,value", [
    ("source_trajectory_id", ""), ("source_trajectory_id", "unverified"),
    ("source_trajectory_id", "training:0:90-1"), ("context_start", None),
    ("source_start", True), ("source_end", 90.5), ("source_end", 101),
])
def test_unknown_or_contradictory_provenance_is_not_claimed_independent(field, value):
    report = _audit([_ep("a", 0), replace(_ep("b", 1, 200), **{field: value})])
    assert report["status"] == "blocked"


def test_full_normalizer_training_inventory_checked_not_just_selected_training_row():
    eps = [_ep("chosen-train", 0), _ep("other-train", 1, 200), _ep("chosen-val", 1, 200)]
    report = _audit(eps, {"train": (0, 1), "val": (2,)}, {"source": "fresh_train_only_fit", "train_episode_count": 2})
    assert report["status"] == "blocked"


@pytest.mark.parametrize("metadata", [{}, {"source": "all-data"}, {"source": "fresh_train_only_fit", "train_episode_count": 2}])
def test_unverified_normalizer_scope_blocks(metadata):
    assert _audit([_ep("a", 0), _ep("b", 1, 200)], metadata=metadata)["status"] == "blocked"


def test_shared_validated_train_normalizer_accepted():
    metadata = {"fit_scope": "one_shared_normalizer_over_selected_train_split_only", "train_episode_count": 1}
    assert _audit([_ep("a", 0), _ep("b", 1, 200)], metadata=metadata)["status"] == "passed"


def test_alias_path_rejected_even_with_different_source_claim(tmp_path: Path):
    path = tmp_path / "source.hdf5"
    path.touch()
    alias = tmp_path / "alias.hdf5"
    alias.symlink_to(path)
    report = _audit([replace(_ep("a", 0), path=path), replace(_ep("b", 1, 200), path=alias)])
    assert any(p["kind"] == "shared_resolved_path" for p in report["problems"])


@pytest.mark.parametrize("indices", [(0, 0), (True,), (-1,), (9,)])
def test_invalid_split_indices_not_silently_cast(indices):
    assert _audit([_ep("a", 0), _ep("b", 1, 200)], {"train": indices, "val": (1,)})["status"] == "blocked"


def test_default_spreads_sources_and_uses_eligible_centers_without_labels():
    eps = [_ep("a", 0), _ep("b", 1, 200), _ep("c", 2, 400)]
    refs = [ObservedWindowRef(i, t) for i in range(3) for t in (20, 40, 60)]
    rng = torch.get_rng_state().clone()
    indices, records = select_probe_indices(eps, refs, count=3)
    assert indices == [1, 4, 7]
    assert [r["center"] for r in records] == [40, 40, 40]
    assert torch.equal(rng, torch.get_rng_state())
    # Single-batch default need not be the reset/first inventory row.
    assert select_probe_indices(eps, refs, count=1)[0] == [4]
    eps[0].source_annotation_index = None
    assert select_probe_indices(eps, refs, count=3)[0] == indices


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, 9], [True, 1], [0]])
def test_bad_explicit_probe_indices_rejected(indices):
    with pytest.raises(ValueError):
        select_probe_indices([_ep("a", 0)], [ObservedWindowRef(0, t) for t in range(4)], count=2, explicit=indices)


def test_explicit_order_preserved_but_default_never_repeats_source():
    eps = [_ep("a", 0)]
    refs = [ObservedWindowRef(0, t) for t in range(5)]
    assert select_probe_indices(eps, refs, count=2, explicit=[3, 1])[0] == [3, 1]
    with pytest.raises(ValueError, match="distinct"):
        select_probe_indices(eps, refs, count=2)


def test_support_report_exposes_missing_rows_without_promoting_them_to_success():
    batch, _ = _batch()
    report = batch_support_report(batch)
    assert report["world_feedback_observed_fraction"] == 1
    assert report["endpoint_visual_observed_fraction"] == 1
    assert len(report["action_row_support"]) == 24
    assert batch.future.annotation_endpoint is not None
    missing = replace(batch, future=replace(batch.future, annotation_endpoint=replace(
        batch.future.annotation_endpoint, visual_observed=torch.zeros_like(batch.future.annotation_endpoint.visual_observed),
    )))
    assert batch_support_report(missing)["endpoint_visual_observed_fraction"] == 0
    assert "success" not in report
