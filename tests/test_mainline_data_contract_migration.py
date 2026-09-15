from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    CheckpointIdentity,
    DatasetIdentity,
    SourceSnapshot,
    build_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.checkpoints import (
    DATA_CONTRACT_MIGRATION_SOURCE_PATHS,
    INITIALIZATION_SOURCE_PATHS,
    LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION,
    LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION,
    load_checkpoint_for_initialization,
    save_checkpoint,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule

ROOT = Path(__file__).resolve().parents[1]


def test_data_migration_keeps_standard_initialization_source_allowlist() -> None:
    assert INITIALIZATION_SOURCE_PATHS <= DATA_CONTRACT_MIGRATION_SOURCE_PATHS
    assert "clearvla/mainline/runtime/evaluation.py" in (
        DATA_CONTRACT_MIGRATION_SOURCE_PATHS
    )


def _dataset(
    *,
    state: str | None = None,
    inventory: str | None = None,
) -> DatasetIdentity:
    zero = hashlib.sha256(b"").hexdigest()
    return DatasetIdentity(
        raw_root="/data/libero/causal",
        hdf5_glob="*.hdf5",
        inventory_sha256=inventory or zero,
        state_normalizer_sha256=state or zero,
        action_normalizer_sha256=zero,
        decoded_cache_identity=zero,
        dino_cache_identity=zero,
    )


def _libero_config(contract: str) -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            data_profile="libero_relative_7d_v1",
            window_boundary_contract=contract,
        ),
        bottom=replace(
            base.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
    )
    config.validate()
    return config


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)


def _checkpoint(tmp_path: Path):
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"goal")
    config = _libero_config("strict_complete_v1")
    identity = build_checkpoint_identity(
        config,
        repo_root=ROOT,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    model = _TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=1, total_steps=2, minimum_ratio=0.1
    )
    path = tmp_path / "source.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=8,
        global_step=0,
        best_metric=0.3,
    )
    return path, condition, config, identity, model


def _identity_with_source_change(identity: CheckpointIdentity) -> CheckpointIdentity:
    rows = list(identity.source.files)
    path, _digest = rows[0]
    rows[0] = (path, "f" * 64)
    rows.sort()
    digest = hashlib.sha256(
        __import__("json").dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return replace(
        identity,
        source=SourceSnapshot(files=tuple(rows), digest=digest),
    )


def test_explicit_libero_boundary_migration_accepts_a_and_b_but_is_model_only(
    tmp_path: Path,
) -> None:
    path, condition, _source_config, source_identity, source_model = _checkpoint(tmp_path)
    for contract in (
        "causal_prefix_v1",
        "causal_prefix_terminal_suffix_v2",
    ):
        target_config = _libero_config(contract)
        target_identity = build_checkpoint_identity(
            target_config,
            repo_root=ROOT,
            dataset=_dataset(),
            language=ArtifactIdentity.from_file("t5_goal", condition),
            commit="1" * 40,
        )
        target = _TinyPolicy()
        for value in target.parameters():
            torch.nn.init.zeros_(value)
        state = load_checkpoint_for_initialization(
            path,
            model=target,
            config=target_config,
            identity=target_identity,
            data_contract_migration=LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION,
        )
        assert state.data_contract_migration == (
            LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION
        )
        assert state.normalizer_identity_equal is True
        assert state.saved_window_boundary_contract == "strict_complete_v1"
        assert state.current_window_boundary_contract == contract
        for name, value in source_model.state_dict().items():
            assert torch.equal(target.state_dict()[name], value)


def test_boundary_migration_requires_explicit_flag_and_identical_normalizers(
    tmp_path: Path,
) -> None:
    path, condition, _source_config, _source_identity, source_model = _checkpoint(tmp_path)
    target_config = _libero_config("causal_prefix_v1")
    target_identity = build_checkpoint_identity(
        target_config,
        repo_root=ROOT,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    target = _TinyPolicy()
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="model initialization config differs"):
        load_checkpoint_for_initialization(
            path,
            model=target,
            config=target_config,
            identity=target_identity,
        )
    assert all(torch.equal(target.state_dict()[name], value) for name, value in before.items())

    drift_identity = build_checkpoint_identity(
        target_config,
        repo_root=ROOT,
        dataset=_dataset(state="a" * 64),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    with pytest.raises(ValueError, match="identical action/state normalizers"):
        load_checkpoint_for_initialization(
            path,
            model=target,
            config=target_config,
            identity=drift_identity,
            data_contract_migration=LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION,
        )
    assert all(torch.equal(target.state_dict()[name], value) for name, value in before.items())


def test_boundary_migration_rejects_model_source_drift_before_mutating_model(
    tmp_path: Path,
) -> None:
    path, condition, _source_config, source_identity, _source_model = _checkpoint(tmp_path)
    target_config = _libero_config("causal_prefix_v1")
    current = build_checkpoint_identity(
        target_config,
        repo_root=ROOT,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    drifted = _identity_with_source_change(current)
    target = _TinyPolicy()
    before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with pytest.raises(ValueError, match="source drift escapes the allow-list"):
        load_checkpoint_for_initialization(
            path,
            model=target,
            config=target_config,
            identity=drifted,
            data_contract_migration=LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION,
        )
    assert all(torch.equal(target.state_dict()[name], value) for name, value in before.items())


def test_explicit_libero_retarget_migration_is_model_only_and_normalizer_stable(
    tmp_path: Path,
) -> None:
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"goal")
    base = _libero_config("causal_prefix_terminal_suffix_v2")
    source_config = replace(
        base,
        data=replace(
            base.data,
            split_mode="episode-manifest",
            split_manifest="/data/libero/causal/splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
        ),
    )
    source_config.validate()
    source_identity = build_checkpoint_identity(
        source_config,
        repo_root=ROOT,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    source_model = _TinyPolicy()
    optimizer = torch.optim.AdamW(source_model.parameters(), lr=1.0e-3)
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=1, total_steps=2, minimum_ratio=0.1
    )
    checkpoint = tmp_path / "terminal_suffix.pt"
    save_checkpoint(
        checkpoint,
        model=source_model,
        optimizer=optimizer,
        schedule=schedule,
        config=source_config,
        identity=source_identity,
        epoch=2,
        global_step=0,
        best_metric=0.2,
    )

    target_config = replace(
        source_config,
        data=replace(
            source_config.data,
            libero_retarget_overlay_root="/data/libero/overlay",
            libero_retarget_overlay_manifest=(
                "/data/libero/overlay/overlay_manifest.json"
            ),
        ),
    )
    target_config.validate()
    target_identity = build_checkpoint_identity(
        target_config,
        repo_root=ROOT,
        dataset=_dataset(inventory="f" * 64),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    target = _TinyPolicy()
    for value in target.parameters():
        torch.nn.init.zeros_(value)
    state = load_checkpoint_for_initialization(
        checkpoint,
        model=target,
        config=target_config,
        identity=target_identity,
        data_contract_migration=LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION,
    )
    assert state.data_contract_migration == LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION
    assert state.normalizer_identity_equal is True
    assert state.saved_window_boundary_contract == "causal_prefix_terminal_suffix_v2"
    assert state.current_window_boundary_contract == "causal_prefix_terminal_suffix_v2"
    for name, value in source_model.state_dict().items():
        assert torch.equal(target.state_dict()[name], value)

    drifted_config = replace(
        target_config,
        objectives=replace(
            target_config.objectives,
            decoded_action=target_config.objectives.decoded_action + 0.01,
        ),
    )
    drifted_identity = build_checkpoint_identity(
        drifted_config,
        repo_root=ROOT,
        dataset=_dataset(inventory="e" * 64),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    untouched = _TinyPolicy()
    before = {
        name: value.detach().clone() for name, value in untouched.state_dict().items()
    }
    with pytest.raises(ValueError, match="outside the admitted training overlay"):
        load_checkpoint_for_initialization(
            checkpoint,
            model=untouched,
            config=drifted_config,
            identity=drifted_identity,
            data_contract_migration=LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION,
        )
    assert all(
        torch.equal(untouched.state_dict()[name], value)
        for name, value in before.items()
    )

    normalizer_drift_identity = build_checkpoint_identity(
        target_config,
        repo_root=ROOT,
        dataset=_dataset(state="a" * 64, inventory="d" * 64),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    with pytest.raises(ValueError, match="identical action/state normalizers"):
        load_checkpoint_for_initialization(
            checkpoint,
            model=untouched,
            config=target_config,
            identity=normalizer_drift_identity,
            data_contract_migration=LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION,
        )
    assert all(
        torch.equal(untouched.state_dict()[name], value)
        for name, value in before.items()
    )
