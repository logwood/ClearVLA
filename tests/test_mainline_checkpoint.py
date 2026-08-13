import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    DatasetIdentity,
    active_source_snapshot,
    build_checkpoint_identity,
    checkpoint_identity_from_mapping,
    compare_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.manifest import manifest_from_mapping
from clearvla.mainline.runtime.checkpoints import (
    load_checkpoint_exact,
    migrate_bottom_only,
    save_checkpoint,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


def _dataset() -> DatasetIdentity:
    zero = hashlib.sha256(b"").hexdigest()
    return DatasetIdentity(
        raw_root="/data/liang.zhang/dataset/grab_pen_single/grab_pen_single",
        hdf5_glob="*.hdf5",
        inventory_sha256=zero,
        state_normalizer_sha256=zero,
        action_normalizer_sha256=zero,
        decoded_cache_identity=zero,
        dino_cache_identity=zero,
    )


def test_active_source_snapshot_excludes_legacy_version_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot = active_source_snapshot(root)
    paths = {path for path, _ in snapshot.files}
    assert "clearvla/mainline/model/top.py" in paths
    assert "clearvla/data/hdf5_episode.py" in paths
    assert "clearvla/data/split.py" in paths
    assert "clearvla/vision/decoded_image_store.py" in paths
    assert "clearvla/vision/preprocessing.py" in paths
    assert "clearvla/data/schema.py" in paths
    assert "clearvla/vision/image_io.py" in paths
    assert "clearvla/data/__init__.py" in paths
    assert "clearvla/vision/__init__.py" in paths
    assert "clearvla/mainline/v120_core/time_domain_mmdit.py" in paths
    assert "clearvla/mainline/v120_core/flow_dino_evidence.py" in paths
    assert "clearvla/mainline/v120_core/profile.py" in paths
    assert "clearvla/mainline/model/action_contract.py" in paths
    assert "clearvla/mainline/model/observation_contract.py" in paths
    # These superseded independent rewrites are retained only as source
    # archaeology.  The formal entry point has no dependency on either one.
    assert "clearvla/mainline/model/bottom.py" not in paths
    assert "clearvla/mainline/model/observation.py" not in paths
    assert "clearvla/mainline/v120_core/trunk.py" not in paths
    assert "clearvla/mainline/v120_core/system.py" not in paths
    assert not any("clearvla/mainline/v120_core/legacy/" in path for path in paths)
    assert not any("policy_runtime_v39.py" in path for path in paths)
    assert not any("train_v40_policy.py" in path for path in paths)
    assert not any("scripts/current_v" in path for path in paths)


def test_checkpoint_identity_roundtrip_and_explicit_bottom_migration(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    language = ArtifactIdentity.from_file("t5_goal", condition)
    identity = build_checkpoint_identity(
        ExperimentConfig(),
        repo_root=root,
        dataset=_dataset(),
        language=language,
        commit="1" * 40,
    )
    restored = checkpoint_identity_from_mapping(identity.as_dict())
    assert restored == identity
    exact = compare_checkpoint_identity(identity, restored)
    assert exact.exact_resume
    assert "top" in exact.reusable_components

    changed = type(identity)(
        manifest=identity.manifest,
        manifest_digest=identity.manifest_digest,
        config_digest="2" * 64,
        source=identity.source,
        git_commit=identity.git_commit,
        dataset=identity.dataset,
        language=identity.language,
    )
    report = compare_checkpoint_identity(identity, changed)
    assert not report.exact_resume
    assert report.reusable_components == ("bottom",)
    assert "top" in report.rejected_components

    historical_manifest = copy.deepcopy(identity.manifest)
    # Schema 22 was experimentally rejected.  It remains parseable only for
    # the explicit unchanged-bottom migration path and can never exact-resume
    # into the schema-24 V120-fidelity recovery.
    historical_manifest["schema"] = 22
    components = dict(historical_manifest["components"])
    components["top"] = "object_intent_dynamics_323_schema6"
    historical_manifest["components"] = components
    parsed_historical = manifest_from_mapping(
        historical_manifest,
        require_current_schema=False,
    )
    historical = replace(
        identity,
        manifest=historical_manifest,
        manifest_digest=parsed_historical.digest(),
    )
    restored_historical = checkpoint_identity_from_mapping(
        historical.as_dict(),
        require_current_manifest=False,
    )
    historical_report = compare_checkpoint_identity(restored_historical, identity)
    assert not historical_report.exact_resume
    assert historical_report.reusable_components == ("bottom",)
    assert "top" in historical_report.rejected_components


def test_artifact_relocation_does_not_change_semantic_identity(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / "first" / "goal.pt"
    second = tmp_path / "second" / "goal.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same-t5-condition")
    second.write_bytes(first.read_bytes())
    identity = build_checkpoint_identity(
        ExperimentConfig(),
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", first),
        commit="1" * 40,
    )
    relocated = replace(
        identity,
        language=ArtifactIdentity.from_file("t5_goal", second),
    )
    assert compare_checkpoint_identity(identity, relocated).exact_resume


def test_checkpoint_identity_rejects_non_hex_and_inconsistent_source_hashes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    identity = build_checkpoint_identity(
        ExperimentConfig(),
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    malformed = replace(identity, config_digest="z" * 64)
    try:
        malformed.validate()
    except ValueError as error:
        assert "config digest" in str(error)
    else:
        raise AssertionError("a non-hex config digest must be rejected")
    inconsistent_source = replace(
        identity,
        source=replace(identity.source, digest="0" * 64),
    )
    try:
        inconsistent_source.validate()
    except ValueError as error:
        assert "source snapshot digest" in str(error)
    else:
        raise AssertionError("a source digest unrelated to its rows must be rejected")


def test_exact_resume_restores_owned_generator_states(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    for _ in range(31):
        schedule.step()
    shuffle = torch.Generator().manual_seed(17)
    flow = torch.Generator().manual_seed(19)
    condition_generator = torch.Generator().manual_seed(23)
    torch.rand(5, generator=shuffle)
    torch.rand(7, generator=flow)
    torch.rand(11, generator=condition_generator)
    expected_shuffle = torch.Generator()
    expected_shuffle.set_state(shuffle.get_state())
    expected_flow = torch.Generator()
    expected_flow.set_state(flow.get_state())
    expected_condition = torch.Generator()
    expected_condition.set_state(condition_generator.get_state())
    expected = (
        torch.rand(4, generator=expected_shuffle),
        torch.rand(4, generator=expected_flow),
        torch.rand(4, generator=expected_condition),
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=2,
        global_step=31,
        best_metric=0.2,
        generators={
            "train_loader": shuffle,
            "train_flow": flow,
            "train_condition": condition_generator,
        },
    )
    torch.rand(20, generator=shuffle)
    torch.rand(20, generator=flow)
    torch.rand(20, generator=condition_generator)
    restored = load_checkpoint_exact(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        generators={
            "train_loader": shuffle,
            "train_flow": flow,
            "train_condition": condition_generator,
        },
    )
    assert restored.epoch == 2 and restored.global_step == 31
    assert torch.equal(torch.rand(4, generator=shuffle), expected[0])
    assert torch.equal(torch.rand(4, generator=flow), expected[1])
    assert torch.equal(torch.rand(4, generator=condition_generator), expected[2])


def test_exact_resume_rejection_does_not_partially_mutate_live_model(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    for _ in range(2):
        schedule.step()
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=1,
        global_step=2,
        best_metric=None,
        generators={"train_loader": torch.Generator().manual_seed(3)},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(7.0)
    live_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        load_checkpoint_exact(
            path,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            identity=identity,
            generators={"wrong_owner": torch.Generator().manual_seed(4)},
        )
    except ValueError as error:
        assert "generator ownership differs" in str(error)
    else:
        raise AssertionError("malformed generator ownership must reject exact resume")
    for name, value in model.state_dict().items():
        assert torch.equal(value, live_before[name])


def test_exact_resume_rejects_malformed_optimizer_state_before_model_mutation(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    schedule.step()
    path = tmp_path / "bad-optimizer.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=1,
        best_metric=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["optimizer"]["state"]
    first_parameter_id = next(iter(state))
    state[first_parameter_id]["exp_avg"] = torch.zeros(1)
    torch.save(payload, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    live_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        load_checkpoint_exact(
            path,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            identity=identity,
        )
    except ValueError as error:
        assert "optimizer state 'exp_avg' has an incompatible shape" in str(error)
    else:
        raise AssertionError("malformed optimizer moments must reject exact resume")
    for name, value in model.state_dict().items():
        assert torch.equal(value, live_before[name])


def test_exact_resume_rejects_model_dtype_before_live_state_mutation(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    schedule.step()
    path = tmp_path / "bad-model-dtype.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=1,
        best_metric=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    parameter_name = next(iter(payload["model"]))
    payload["model"][parameter_name] = payload["model"][parameter_name].double()
    torch.save(payload, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)
    live_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        load_checkpoint_exact(
            path,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            config=config,
            identity=identity,
        )
    except ValueError as error:
        assert f"model state {parameter_name!r} has an incompatible dtype" in str(error)
    else:
        raise AssertionError("model dtype mismatch must reject exact resume")
    for name, value in model.state_dict().items():
        assert torch.equal(value, live_before[name])


def test_bottom_migration_accepts_historical_top_with_same_bottom_abi(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bottom = torch.nn.Linear(3, 2)

    source = TinyPolicy()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    path = tmp_path / "historical-top.pt"
    save_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=0,
        best_metric=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    historical_manifest = payload["identity"]["manifest"]
    historical_manifest["schema"] = int(historical_manifest["schema"]) - 1
    historical_manifest["components"]["top"] = "object_intent_dynamics_323_schema6"
    historical_parsed = manifest_from_mapping(
        historical_manifest,
        require_current_schema=False,
    )
    payload["identity"]["manifest_digest"] = historical_parsed.digest()
    torch.save(payload, path)

    target = TinyPolicy()
    with torch.no_grad():
        target.bottom.weight.zero_()
        target.bottom.bias.zero_()
    report = migrate_bottom_only(path, target, identity=identity)
    assert report.loaded == ("bottom.bias", "bottom.weight")
    assert torch.equal(target.bottom.weight, source.bottom.weight)
    assert torch.equal(target.bottom.bias, source.bottom.bias)


def test_bottom_migration_rejects_untyped_shape_only_checkpoint(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    identity = build_checkpoint_identity(
        ExperimentConfig(),
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    path = tmp_path / "legacy.pt"
    torch.save({"model": {"bottom.fake": torch.ones(1)}}, path)
    try:
        migrate_bottom_only(path, torch.nn.Linear(1, 1), identity=identity)
    except ValueError as exc:
        assert "ABI-compatible complete typed bottom" in str(exc)
    else:
        raise AssertionError("shape-only bottom migration must be rejected")


def test_bottom_migration_rejects_incomplete_typed_state_before_mutation(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bottom = torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.Linear(4, 2),
            )

    model = TinyPolicy()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    incomplete = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name != "bottom.1.bias"
    }
    path = tmp_path / "incomplete.pt"
    torch.save(
        {
            "schema": "clearvla-mainline-checkpoint-v4",
            "identity": identity.as_dict(),
            "model": incomplete,
        },
        path,
    )
    try:
        migrate_bottom_only(path, model, identity=identity)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("an incomplete typed bottom must be rejected")
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
