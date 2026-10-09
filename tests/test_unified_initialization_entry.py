"""Exercise the actual model-only loader with artificial small checkpoints.

These tests do not contain the mature CALVIN checkpoint. Positive fixtures pin
only the expected source-digest constant to a self-consistent artificial source
snapshot; config, source-path, dataset and parameter validation all run normally.
No real-data training, pretrained encoder or behavior admission is claimed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
from test_mainline_checkpoint import _dataset, _reduced_joint_calvin_config
from test_mainline_takeover_admission import _tree_equal

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    CheckpointIdentity,
    SourceSnapshot,
    build_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime import causal_identity_migration as migration
from clearvla.mainline.runtime.checkpoints import (
    load_checkpoint_for_initialization,
    save_checkpoint,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

NEW_MODES = (
    migration.CAUSAL_UNIFIED_VALUES_V1,
    migration.CAUSAL_UNIFIED_REFERENCE_V1,
    migration.CAUSAL_UNIFIED_OUTCOME_V1,
)
MODES = (
    migration.CAUSAL_IDENTITY_AB_V1,
    migration.CAUSAL_UNIFIED_SOURCE_V1,
    *NEW_MODES,
)
CHANGED_PATH = "clearvla/mainline/runtime/checkpoints.py"


def _changed_source(source: SourceSnapshot, path: str) -> SourceSnapshot:
    assert path in dict(source.files)
    rows = tuple(
        (name, hashlib.sha256(("artificial-source:" + name).encode()).hexdigest())
        if name == path else (name, digest)
        for name, digest in source.files
    )
    result = SourceSnapshot(
        rows,
        hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ArtificialCheckpoint:
    path: Path
    config: ExperimentConfig
    identity: CheckpointIdentity
    current_source: SourceSnapshot
    model: ClearVLAMainlinePolicy


@pytest.fixture(scope="module")
def source_checkpoint(tmp_path_factory) -> ArtificialCheckpoint:
    torch.manual_seed(7741)
    root = tmp_path_factory.mktemp("unified_artificial_initialization")
    config = _reduced_joint_calvin_config()
    config = replace(
        config,
        data=replace(config.data, visual_feature_mode="dinov3_online_v1"),
        dimensions=replace(config.dimensions, visual_token_dim=768, patches_per_camera=256),
        observation=replace(config.observation, visual_chart_mode="full_rgb_endpoint_v1"),
    )
    config.validate()
    language = root / "language.pt"
    language.write_bytes(b"artificial-language-identity-only-not-pretrained-weights")
    current_identity = build_checkpoint_identity(
        config,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=replace(_dataset(), raw_root="/artificial/calvin"),
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="7" * 40,
    )
    identity = replace(
        current_identity, source=_changed_source(current_identity.source, CHANGED_PATH)
    )
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    # Seven is a declared artificial clock, not a history of executed updates.
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1, update_origin=7
    )
    model.set_training_step(7)
    path = root / "artificial-source.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, schedule=schedule, config=config,
        identity=identity, epoch=1, global_step=7, best_metric=None,
    )
    return ArtificialCheckpoint(path, config, identity, current_identity.source, model)


def _candidate(source: ArtificialCheckpoint, mode: str, variant: str):
    canonical = variant == "B"
    config = replace(
        source.config,
        top=replace(
            source.config.top,
            entity_transport_gradient_mode="ordinary_bilinear_v1",
            entity_competition_scale_mode="per_observation_v1",
            observation_measurement_mode="source_consistent_v1",
            target_binding_input_mode="full_tokens_views_v1",
            observed_outcome_mode=(
                "robot_world_before_proposal_v2"
                if mode == migration.CAUSAL_UNIFIED_OUTCOME_V1 else "before_proposal_v1"
            ),
            entity_ownership_mode="canonical_image_v1" if canonical else "local_mixture_v1",
            identity_supervision_mode="rgbd_temporal_conditional_v2" if canonical else "none",
            typed_interval_gradient_mode=(
                "legacy_common_surrogate_v1" if mode == migration.CAUSAL_IDENTITY_AB_V1
                else "ordinary_v1"
            ),
            typed_object_value_mode="conditional_object_v1" if mode in NEW_MODES else "legacy_selected_v1",
        ),
        objectives=replace(
            source.config.objectives,
            gripper_command_transition=0.05,
            calvin_frame_weight_mode="motion_event_v1",
            calvin_frame_motion_gain=0.75,
            calvin_frame_event_gain=1.5,
            calvin_frame_event_radius=1,
            calvin_frame_max_weight=3.0,
            identity_correspondence=0.01 if canonical else 0.0,
            identity_source_prediction=0.01 if canonical else 0.0,
        ),
    )
    config.validate()
    identity = replace(
        source.identity,
        config_digest=config.digest(include_paths=False),
        source=source.current_source,
    )
    return config, ClearVLAMainlinePolicy(config), identity


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["A", "B"])
def test_actual_loader_initializes_every_declared_mode_without_optimizer_or_rng_restore(
    source_checkpoint, tmp_path, monkeypatch, mode, variant,
):
    source = source_checkpoint
    # Test-local expected identity only. The production source pin is unchanged,
    # and the unmodified production pin must reject this fixture below.
    monkeypatch.setattr(migration, "SOURCE_DIGEST", source.identity.source.digest)
    config, model, identity = _candidate(source, mode, variant)
    before = copy.deepcopy(model.state_dict())
    saved = source.model.state_dict()
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1, update_origin=7
    )
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    schedule_before = copy.deepcopy(schedule.state_dict())
    rng_before = torch.get_rng_state().clone()
    # These are deliberately invalid continuation records: model-only loading
    # must neither consume them nor touch the new optimizer/schedule/global RNG.
    payload = torch.load(source.path, weights_only=False)
    for key in ("optimizer", "schedule", "rng", "generators"):
        payload[key] = "not-model-initialization-state"
    path = tmp_path / "model-only.pt"
    torch.save(payload, path)
    result = load_checkpoint_for_initialization(
        path, model=model, config=config, identity=identity, model_contract_migration=mode
    )
    assert result.model_contract_migration == mode and result.global_step == 7
    assert result.changed_source_files == (CHANGED_PATH,)
    assert result.normalizer_identity_equal
    actual = model.state_dict()
    added = set(actual) - set(saved)
    removed = set(saved) - set(actual)
    assert len(added) == 5 + (8 if variant == "B" else 0) + (
        3 if mode == migration.CAUSAL_UNIFIED_OUTCOME_V1 else 0
    )
    assert len(removed) == 5 + (1 if variant == "B" else 0)
    for name, value in actual.items():
        torch.testing.assert_close(value, before[name] if name in added else saved[name], rtol=0, atol=0)
    assert actual["intent.organizer.observed_outcome.output.weight"].count_nonzero() == 0
    if mode == migration.CAUSAL_UNIFIED_OUTCOME_V1:
        assert actual["intent.organizer.observed_robot_outcome.output.weight"].count_nonzero() == 0
    _tree_equal(optimizer_before, optimizer.state_dict())
    _tree_equal(schedule_before, schedule.state_dict())
    assert not optimizer.state and torch.equal(rng_before, torch.get_rng_state())


@pytest.mark.parametrize("mode", MODES)
def test_unmodified_source_pin_rejects_artificial_checkpoint_after_mode_admission(
    source_checkpoint, mode,
):
    config, model, identity = _candidate(source_checkpoint, mode, "A")
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="requires the admitted a2d597d2 source identity"):
        load_checkpoint_for_initialization(
            source_checkpoint.path, model=model, config=config, identity=identity,
            model_contract_migration=mode,
        )
    _tree_equal(before, model.state_dict())


@pytest.mark.parametrize("fault,expected", [
    ("unknown_mode", "unknown model-initialization model migration"),
    ("older_source_mode", "explicit initialization migration"),
    ("missing_state", "state inventory"),
    ("extra_state", "state inventory"),
    ("wrong_shape", "incompatible shape"),
    ("wrong_dtype", "incompatible dtype"),
    ("nonfinite_weight", "non-finite"),
    ("nonneutral_robot_head", "must initialize neutral"),
    ("source_path_drift", "escapes the allow-list"),
    ("objective_drift", "outside its declared graph/objective selectors"),
    ("dataset_drift", "exact source inventory proof"),
    ("normalizer_drift", "exact source inventory proof"),
    ("language_drift", "language identity differs"),
])
def test_latest_loader_retains_strict_boundaries_without_model_or_rng_mutation(
    source_checkpoint, tmp_path, monkeypatch, fault, expected,
):
    source = source_checkpoint
    monkeypatch.setattr(migration, "SOURCE_DIGEST", source.identity.source.digest)
    mode = migration.CAUSAL_UNIFIED_OUTCOME_V1
    config, model, identity = _candidate(source, mode, "B")
    payload = torch.load(source.path, weights_only=False)
    common = "grounding.grounder.content_key.1.weight"
    if fault == "unknown_mode":
        mode = "causal_unified_unregistered_v1"
    elif fault == "older_source_mode":
        mode = migration.CAUSAL_UNIFIED_SOURCE_V1
    elif fault == "missing_state":
        payload["model"].pop(common)
    elif fault == "extra_state":
        payload["model"]["unregistered.weight"] = torch.ones(1)
    elif fault == "wrong_shape":
        payload["model"][common] = payload["model"][common].flatten()
    elif fault == "wrong_dtype":
        payload["model"][common] = payload["model"][common].double()
    elif fault == "nonfinite_weight":
        payload["model"][common].flatten()[0] = float("nan")
    elif fault == "nonneutral_robot_head":
        reader = model.intent.organizer.observed_robot_outcome
        assert reader is not None
        with torch.no_grad():
            reader.output.weight.fill_(0.1)
    elif fault == "source_path_drift":
        identity = replace(identity, source=_changed_source(identity.source, "clearvla/mainline/training/losses.py"))
    elif fault == "objective_drift":
        config = replace(config, objectives=replace(config.objectives, annotated_goal=2.0))
        identity = replace(identity, config_digest=config.digest(include_paths=False))
    elif fault == "dataset_drift":
        identity = replace(identity, dataset=replace(identity.dataset, inventory_sha256="a" * 64))
    elif fault == "normalizer_drift":
        identity = replace(identity, dataset=replace(identity.dataset, action_normalizer_sha256="a" * 64))
    elif fault == "language_drift":
        identity = replace(identity, language=replace(identity.language, sha256="a" * 64))
    path = tmp_path / "rejected.pt"
    torch.save(payload, path)
    before = copy.deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=expected):
        load_checkpoint_for_initialization(
            path, model=model, config=config, identity=identity, model_contract_migration=mode
        )
    _tree_equal(before, model.state_dict())
    assert torch.equal(rng_before, torch.get_rng_state())
