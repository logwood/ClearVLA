"""Heldout CLI real-source lane with actual dataset/batch/model and fixture transport.

HDF5 discovery/encoder storage is represented by a declared bundle. This is not
an experiment on user data. Source admission and all following compute are real.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from test_mainline_annotation_endpoint_sources import _goal_dataset
from test_mainline_annotation_goal import _config
from test_mainline_instruction_reference import _Tokens
from test_mainline_timed_history import _ImageStore

from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, MainlineDataBundle

ROOT = Path(__file__).resolve().parents[1]


def _bundle(shared: bool) -> MainlineDataBundle:
    base = _goal_dataset()
    a = replace(base.episodes[0], episode_id="train", path=Path("train.hdf5"), source_trajectory_id="training:000000:1000-1100")
    b = replace(a, episode_id="val", path=Path("val.hdf5")) if shared else replace(
        a, episode_id="val", path=Path("val.hdf5"), source_trajectory_id="training:000001:2000-2100",
        context_start=2000, source_start=2003, source_end=2080,
    )
    episodes = [a, b]
    datasets = {}
    for name, index in (("train", 0), ("val", 1)):
        data = ObservedStateWindowDataset(
            episodes, [index], image_store=cast(Any, _ImageStore()), camera_names=base.camera_names,
            action_normalizer=base.action_normalizer, state_normalizer=base.state_normalizer, config=base.config,
        )
        datasets[name] = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, _Tokens()))
    return MainlineDataBundle(
        episodes=tuple(episodes), splits={"train": (0,), "val": (1,)}, datasets=datasets,
        materialized_episode_indices=(0, 1), action_normalizer=base.action_normalizer,
        state_normalizer=base.state_normalizer,
        goal=GoalTemplate(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        skipped=(), sampling_seed=17, information_uniform_fraction=0.2, information_event_fraction=0.4,
        information_motion_quantile=0.8, gripper_event_threshold=0.1,
        normalizer_metadata={"source": "fresh_train_only_fit", "train_episode_count": 1},
    )


def _run(tmp_path: Path, bundle: MainlineDataBundle, *, invalid: bool):
    spec = importlib.util.spec_from_file_location("heldout_qualifier_test", ROOT / "scripts/qualify_mainline_runtime.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config().as_dict()))
    output = tmp_path / "report"
    output.mkdir()
    (output / "report.json").write_text(json.dumps({"status": "running", "formal_training": False, "model_executed": False}))
    args = argparse.Namespace(
        config=config_path, output=output, operation="fit-heldout", source="real", updates=1,
        device="cpu", dtype="fp32", batch_size=1, raw_side=32, allow_runtime_mismatch=True,
        train_indices=[5], val_indices=[10], inference_step=0,
    )
    with (
        patch.object(module, "bind_worker_to_parent", return_value=False),
        patch("clearvla.mainline.runtime.qualification.required_data_paths", return_value={}),
        patch("clearvla.mainline.data.loading.load_mainline_data", return_value=bundle),
    ):
        if invalid:
            with patch("clearvla.mainline.model.policy.ClearVLAMainlinePolicy", side_effect=AssertionError("must reject before model")):
                code = module.child(args)
        else:
            code = module.child(args)
    return code, json.loads((output / "report.json").read_text())


def test_shared_raw_source_stops_before_model_with_report(tmp_path: Path):
    code, report = _run(tmp_path, _bundle(shared=True), invalid=True)
    assert code == 2 and report["status"] == "blocked"
    assert not report["model_executed"] and "model_constructed" not in report
    assert any(p["kind"] == "shared_raw_trajectory" for p in report["heldout_admission"]["problems"])


def test_disjoint_source_batch_reads_and_model_follow_explicit_split_indices(tmp_path: Path):
    code, report = _run(tmp_path, _bundle(shared=False), invalid=False)
    assert code == 0, report
    assert report["heldout_admission"]["status"] == "passed"
    train = report["probe_selection"]["train"]["rows"][0]
    val = report["probe_selection"]["val"]["rows"][0]
    assert (train["episode_index"], train["dataset_index"], train["center"]) == (0, 5, 8)
    assert (val["episode_index"], val["dataset_index"], val["center"]) == (1, 10, 13)
    assert report["heldout_learning"]["completed_updates"] == 1
    assert report["heldout_learning"]["validation_optimizer_updates"] == 0
    assert report["batch_support"]["train"]["endpoint_visual_observed_fraction"] == 1
    assert report["batch_support"]["val"]["world_feedback_observed_fraction"] == 1
    assert report["probe_schedule"]["state"]["curve"]["warmup_steps"] == _config().optimizer.warmup_steps
    assert not report["formal_training"]
