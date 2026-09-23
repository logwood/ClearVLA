"""Source terminal admission: real converter/loader/dataset, synthetic images/tokens."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, cast

import h5py
import numpy as np
import pytest
import torch
from test_mainline_annotation_goal import _config
from test_mainline_instruction_reference import _dataset, _Tokens
from test_mainline_timed_history import _ImageStore
from torch.utils.data import default_collate

from clearvla.benchmarks.calvin import _SourceTrajectory, _write_episode
from clearvla.data.annotation_endpoint import resolve_annotation_endpoint
from clearvla.data.hdf5_episode import LoadedEpisode, load_episode
from clearvla.mainline.annotation_goal import ANNOTATED_ENDPOINT_GOAL
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.interfaces import OnlinePolicyInput


def _goal_dataset(*, annotation: int | None = 3, end: int = 1080, terminal: int = 80):
    old = _dataset(start=3)
    ep = replace(
        old.episodes[0],
        source_annotation_index=annotation,
        source_end=end,
        terminal_state_index=terminal,
    )
    cfg = replace(
        old.config,
        annotation_goal_mode=ANNOTATED_ENDPOINT_GOAL,
        world_horizon=24,
        robot_feedback_mode=_config().top.robot_feedback_mode,
        world_feedback_mode=_config().top.world_feedback_mode,
    )
    return ObservedStateWindowDataset(
        [ep],
        [0],
        image_store=cast(Any, _ImageStore()),
        camera_names=old.camera_names,
        action_normalizer=old.action_normalizer,
        state_normalizer=old.state_normalizer,
        config=cfg,
    )


@pytest.mark.parametrize("center", [3, 30, 79])
def test_label_endpoint_is_not_current_plus_action_or_world_horizon(center):
    ds = _goal_dataset()
    assert resolve_annotation_endpoint(ds.episodes[0]).index == 80
    tok = _Tokens()
    batch = CachedTokenPolicyWindowDataset(ds, token_store=cast(Any, tok))[center - 3]
    assert tok.keys is not None and tok.keys[-1].tolist() == [0, 80]
    assert batch["annotation_endpoint_declared"]
    assert batch["annotation_endpoint_offset_steps"] == 80 - center
    assert batch["annotation_endpoint_source_indices"].tolist() == [3, 1000, 1003, 1080, center]
    assert torch.all(batch["annotation_endpoint_dino"] == 80)
    assert torch.all(batch["instruction_reference_dino"] == 3)
    # Ordinary future labels still use the original aligned 24-step grid.
    assert batch["target_future_dinov2_tokens"].shape[0] == 6
    typed = to_training_batch(
        default_collate([batch]),
        goal=GoalTemplate(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        config=_config(),
        device=torch.device("cpu"),
    )
    typed.validate(_config())
    assert typed.future.annotation_endpoint is not None
    assert typed.future.annotation_endpoint.offset_steps.item() == 80 - center
    assert "annotation_endpoint" not in {f.name for f in fields(OnlinePolicyInput)}


@pytest.mark.parametrize(
    "missing",
    [
        "source_annotation_index",
        "context_start",
        "source_start",
        "source_end",
        "terminal_state_index",
    ],
)
def test_missing_metadata_never_fabricates_success_or_endpoint(missing):
    ds = _goal_dataset()
    ep = replace(ds.episodes[0], **{missing: None})
    result = resolve_annotation_endpoint(ep)
    assert result.index is None and result.status == "unknown-provenance"


def test_unknown_endpoint_keeps_BC_window_but_masks_label():
    ds = _goal_dataset(annotation=None)
    control = _goal_dataset()
    assert len(ds) == len(control)
    tok = _Tokens()
    raw = CachedTokenPolicyWindowDataset(ds, token_store=cast(Any, tok))[0]
    assert not raw["annotation_endpoint_declared"]
    assert not raw["annotation_endpoint_visual_observed"].any()
    assert raw["annotation_endpoint_dino"].count_nonzero() == 0
    assert raw["annotation_endpoint_source_indices"].tolist() == [-1] * 5
    assert raw["annotation_endpoint_offset_steps"] == 0
    assert ds.boundary_summary()["annotation_endpoint_is_success_label"] is False


@pytest.mark.parametrize("end,terminal", [(1090, 80), (1080, 60)])
def test_censored_prefix_is_unknown_not_a_new_terminal_goal(end, terminal):
    ds = _goal_dataset(end=end, terminal=terminal)
    source = resolve_annotation_endpoint(ds.episodes[0])
    assert source.index is None and source.status == "censored-before-annotation-end"
    raw = ds[0]
    assert not raw["annotation_endpoint_declared"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_annotation_index", True),
        ("source_end", 1080.5),
        ("source_start", -1),
        ("context_start", "1000"),
    ],
)
def test_malformed_metadata_is_not_silently_converted(field, value):
    ep = _goal_dataset().episodes[0]
    with pytest.raises(ValueError):
        resolve_annotation_endpoint(replace(ep, **{field: value}))


def test_storage_boundary_or_task_string_cannot_authorize_goal():
    ep = _goal_dataset().episodes[0]
    result = resolve_annotation_endpoint(
        replace(ep, terminal_padding_mode="none", task_id="success", instruction="done")
    )
    assert result.index is None and result.status == "unverified-terminal-convention"


def test_contradictory_annotation_end_and_real_terminal_is_rejected():
    ep = _goal_dataset().episodes[0]
    with pytest.raises(ValueError):
        resolve_annotation_endpoint(replace(ep, source_end=1079))


def _converted(tmp_path):
    path = tmp_path / "annotation.hdf5"
    frames = [
        dict(
            rgb_static=np.full((4, 4, 3), i, np.uint8),
            rgb_gripper=np.full((4, 4, 3), i, np.uint8),
            rel_actions=np.array([0.1] * 6 + [1.0], np.float32),
            robot_obs=np.array([i / 100] * 6 + [0.04], np.float32),
        )
        for i in range(81)
    ]
    _write_episode(
        path,
        frames=frames,
        instruction="push the block left",
        task="push_red_block_left",
        source_split="training",
        source_start=1024,
        source_end=1080,
        source_annotation_index=37,
        source_trajectory=_SourceTrajectory("training", 0, 1000, 1100),
        context_start=1000,
        language_start_local=24,
        language_end_local=80,
    )
    return path


def _load(path) -> LoadedEpisode:
    return load_episode(
        path, cameras=("top", "wrist"), state_key="state", action_state_key="action_state"
    )


def test_actual_converter_preserved_annotation_id_and_real_endpoint(tmp_path):
    ep = _load(_converted(tmp_path))
    assert ep.source_annotation_index == 37
    assert ep.source_end == 1080 and ep.context_start == 1000
    assert ep.length > 81 and ep.cached_frame_count >= 81
    assert resolve_annotation_endpoint(ep).index == 80
    assert resolve_annotation_endpoint(replace(ep, cache_frame_count=81)).index == 80
    # The repeated absorbing storage suffix is excluded from the goal index.
    assert ep.terminal_state_index != ep.length - 1


@pytest.mark.parametrize(
    "field",
    [
        "source_annotation_index",
        "source_start",
        "source_end",
        "context_start",
        "terminal_state_index",
    ],
)
@pytest.mark.parametrize("bad", [True, 1.5, -1])
def test_real_HDF5_loader_rejects_lossy_source_indices(tmp_path, field, bad):
    path = _converted(tmp_path)
    with h5py.File(path, "a") as f:
        del f.attrs[field]
        f.attrs[field] = bad
    with pytest.raises(ValueError):
        _load(path)


def test_legacy_HDF5_without_annotation_id_still_loads(tmp_path):
    path = _converted(tmp_path)
    with h5py.File(path, "a") as f:
        del f.attrs["source_annotation_index"]
    ep = _load(path)
    assert ep.source_annotation_index is None
    assert resolve_annotation_endpoint(ep).index is None
