import copy
import hashlib
import json
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
from clearvla.mainline.manifest import LAYOUT_SCHEMA, manifest_from_mapping
from clearvla.mainline.model.component_contracts import (
    ComponentSelection,
    legacy_state_dict,
    modular_to_legacy_name,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import (
    CHECKPOINT_SCHEMA,
    VALIDATION_REPLAY_SOURCE_PATHS,
    load_checkpoint_exact,
    load_checkpoint_for_validation,
    migrate_bottom_only,
    save_checkpoint,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE0_IMPLEMENTATION,
    BSPINE0_SPEC_FINGERPRINT,
)


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


def _reduced_modular_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            action_basis_tokens=4,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
        ),
        observation=replace(
            base.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(
            base.top,
            grounder_iterations=2,
            teacher_key_dim=8,
        ),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=2,
            controller_heads=4,
        ),
        optimizer=replace(base.optimizer, warmup_steps=2),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    config.validate()
    return config


def _reduced_bspine_config() -> ExperimentConfig:
    base = _reduced_modular_config()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE0_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE0_SPEC_FINGERPRINT,
        ),
    )
    config.validate()
    return config


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
    assert "clearvla/mainline/model/component_contracts.py" in paths
    assert "clearvla/mainline/model/components.py" in paths
    assert "clearvla/mainline/runtime/checkpoints.py" in paths
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


def test_active_source_snapshot_canonicalizes_checkout_newlines(tmp_path: Path) -> None:
    snapshots = []
    for name, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
        root = tmp_path / name
        package = root / "clearvla" / "mainline"
        preset = root / "configs" / "mainline" / "object_intent_dynamics_323.json"
        package.mkdir(parents=True)
        preset.parent.mkdir(parents=True)
        (package / "train.py").write_bytes(newline.join((b"from __future__ import annotations", b"VALUE = 1", b"")))
        preset.write_bytes(newline.join((b"{", b'  "schema": 1', b"}", b"")))
        snapshots.append(active_source_snapshot(root))

    assert snapshots[0] == snapshots[1]


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
    # into the current S-owned typed-relevance recovery.
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


def test_schema29_manifest_cannot_exact_resume_schema30(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    current = build_checkpoint_identity(
        ExperimentConfig(),
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="2" * 40,
    )
    schema29_mapping = copy.deepcopy(current.manifest)
    schema29_mapping["schema"] = 29
    components = dict(schema29_mapping["components"])
    components["top"] = (
        "v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_"
        "s_owned_relevance_goal_invariant_physical_action_conditioned_w_single_"
        "consequence_refinement_p2_transport_address_typed_consequence_two_"
        "optional_p3"
    )
    components["training"] = (
        "v120_mirrored_physical_flow_exact_teacher_current_support_raw_transport_"
        "event_transition_persistence_gripper_trajectory_v120_decay_local_global_"
        "clip_detached_endpoint_self_conditioned_w_single_action_loss_rng_matched_"
        "gradient_probes"
    )
    schema29_mapping["components"] = components
    schema29_manifest = manifest_from_mapping(
        schema29_mapping,
        require_current_schema=False,
    )
    schema29 = replace(
        current,
        manifest=schema29_mapping,
        manifest_digest=schema29_manifest.digest(),
    )

    report = compare_checkpoint_identity(schema29, current)
    assert not report.exact_resume
    assert "manifest identity differs" in report.reasons
    assert report.reusable_components == ("bottom",)
    assert "top" in report.rejected_components
    assert "training" in report.rejected_components


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


def test_validation_replay_allows_only_source_identity_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = ExperimentConfig()
    saved_identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )
    rows = list(saved_identity.source.files)
    allowed_path = "clearvla/mainline/model/transition.py"
    assert allowed_path in VALIDATION_REPLAY_SOURCE_PATHS
    allowed_index = next(index for index, row in enumerate(rows) if row[0] == allowed_path)
    rows[allowed_index] = (
        allowed_path,
        hashlib.sha256(b"transition-eval-only-edit").hexdigest(),
    )
    source_rows = tuple(rows)
    source_digest = hashlib.sha256(
        json.dumps(
            source_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    validation_identity = replace(
        saved_identity,
        source=replace(
            saved_identity.source,
            files=source_rows,
            digest=source_digest,
        ),
        git_commit="2" * 40,
    )

    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    path = tmp_path / "validation.pt"
    save_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=saved_identity,
        epoch=8,
        global_step=0,
        best_metric=0.2,
    )
    exact_target = torch.nn.Linear(3, 2)
    exact_optimizer = torch.optim.AdamW(exact_target.parameters(), lr=1e-3)
    exact_schedule = WarmupCosineSchedule(
        exact_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    try:
        load_checkpoint_exact(
            path,
            model=exact_target,
            optimizer=exact_optimizer,
            schedule=exact_schedule,
            config=config,
            identity=validation_identity,
        )
    except ValueError as error:
        assert "exact resume rejected" in str(error)
        assert "source identity differs" in str(error)
    else:
        raise AssertionError("exact resume must reject transition source drift")

    expected = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
    restored = load_checkpoint_for_validation(
        path,
        model=target,
        config=config,
        identity=validation_identity,
    )
    assert restored.epoch == 8
    assert restored.global_step == 0
    assert restored.best_metric == 0.2
    assert restored.changed_source_files == (allowed_path,)
    assert restored.saved_source_digest == saved_identity.source.digest
    assert restored.current_source_digest == validation_identity.source.digest
    for name, value in target.state_dict().items():
        assert torch.equal(value, expected[name])

    rejected_identity = replace(
        validation_identity,
        dataset=replace(validation_identity.dataset, inventory_sha256="3" * 64),
    )
    live_before = {name: value.detach().clone() for name, value in target.state_dict().items()}
    try:
        load_checkpoint_for_validation(
            path,
            model=target,
            config=config,
            identity=rejected_identity,
        )
    except ValueError as error:
        assert "dataset identity differs" in str(error)
    else:
        raise AssertionError("validation replay must reject non-source identity drift")
    for name, value in target.state_dict().items():
        assert torch.equal(value, live_before[name])

    unexpected_rows = list(saved_identity.source.files)
    unexpected_path = "configs/mainline/object_intent_dynamics_323.json"
    unexpected_index = next(
        index for index, row in enumerate(unexpected_rows) if row[0] == unexpected_path
    )
    unexpected_rows[unexpected_index] = (
        unexpected_path,
        hashlib.sha256(b"unexpected-source-edit").hexdigest(),
    )
    unexpected_source_rows = tuple(unexpected_rows)
    unexpected_source_digest = hashlib.sha256(
        json.dumps(
            unexpected_source_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    unexpected_identity = replace(
        saved_identity,
        source=replace(
            saved_identity.source,
            files=unexpected_source_rows,
            digest=unexpected_source_digest,
        ),
    )
    try:
        load_checkpoint_for_validation(
            path,
            model=target,
            config=config,
            identity=unexpected_identity,
        )
    except ValueError as error:
        assert "escapes the validation-only allow-list" in str(error)
        assert unexpected_path in str(error)
    else:
        raise AssertionError("validation replay must reject source drift outside its allow-list")


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


def test_modular_checkpoint_round_trip_and_legacy_layout_gate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    config = _reduced_modular_config()
    identity = build_checkpoint_identity(
        config,
        repo_root=root,
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition),
        commit="1" * 40,
    )

    torch.manual_seed(11)
    source = ClearVLAMainlinePolicy(config)
    source_optimizer, _ = build_optimizer(source, config)
    source_schedule = WarmupCosineSchedule(
        source_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    current_path = tmp_path / "modular.pt"
    save_checkpoint(
        current_path,
        model=source,
        optimizer=source_optimizer,
        schedule=source_schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=0,
        best_metric=None,
    )
    current_payload = torch.load(current_path, map_location="cpu", weights_only=False)
    assert current_payload["component_selection"] == ComponentSelection.from_config(
        config
    ).as_dict()
    assert current_payload["identity"]["manifest"]["layout_schema"] == LAYOUT_SCHEMA
    assert all(
        not str(name).startswith(("top.", "bottom.", "action_codec."))
        for name in current_payload["model"]
    )

    torch.manual_seed(13)
    exact_target = ClearVLAMainlinePolicy(config)
    exact_optimizer, _ = build_optimizer(exact_target, config)
    exact_schedule = WarmupCosineSchedule(
        exact_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    restored = load_checkpoint_exact(
        current_path,
        model=exact_target,
        optimizer=exact_optimizer,
        schedule=exact_schedule,
        config=config,
        identity=identity,
    )
    assert restored.epoch == 0 and restored.global_step == 0
    for name, value in source.state_dict().items():
        assert torch.equal(exact_target.state_dict()[name], value)

    legacy_manifest = copy.deepcopy(identity.manifest)
    legacy_manifest["layout_schema"] = 1
    parsed_legacy_manifest = manifest_from_mapping(
        legacy_manifest,
        require_current_schema=False,
    )
    legacy_identity = replace(
        identity,
        manifest=legacy_manifest,
        manifest_digest=parsed_legacy_manifest.digest(),
    )
    legacy_path = tmp_path / "legacy-layout.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "identity": legacy_identity.as_dict(),
            "config": config.as_dict(),
            "model": legacy_state_dict(source),
            "epoch": 0,
            "global_step": 0,
            "best_metric": None,
        },
        legacy_path,
    )
    try:
        load_checkpoint_exact(
            legacy_path,
            model=exact_target,
            optimizer=exact_optimizer,
            schedule=exact_schedule,
            config=config,
            identity=identity,
        )
    except ValueError as error:
        assert "pre-modular model layout" in str(error)
    else:
        raise AssertionError("legacy layout must never enter exact resume")

    torch.manual_seed(17)
    validation_target = ClearVLAMainlinePolicy(config)
    validation = load_checkpoint_for_validation(
        legacy_path,
        model=validation_target,
        config=config,
        identity=identity,
    )
    assert validation.epoch == 0 and validation.global_step == 0
    for name, value in source.state_dict().items():
        assert torch.equal(validation_target.state_dict()[name], value)

    torch.manual_seed(19)
    migration_target = ClearVLAMainlinePolicy(config)
    report = migrate_bottom_only(
        legacy_path,
        migration_target,
        identity=identity,
        config=config,
    )
    expected_bottom_names = tuple(
        sorted(
            name
            for name in source.state_dict()
            if modular_to_legacy_name(name).startswith("bottom.")
        )
    )
    assert report.loaded == expected_bottom_names
    assert not report.missing
    assert not report.shape_mismatch
    assert not report.dtype_mismatch
    for name in expected_bottom_names:
        assert torch.equal(migration_target.state_dict()[name], source.state_dict()[name])


def test_schema31_bspine_round_trip_and_schema30_exact_resume_rejection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    language = ArtifactIdentity.from_file("t5_goal", condition)
    schema30_config = _reduced_modular_config()
    schema31_config = _reduced_bspine_config()
    schema30_identity = build_checkpoint_identity(
        schema30_config,
        repo_root=root,
        dataset=_dataset(),
        language=language,
        commit="1" * 40,
    )
    schema31_identity = build_checkpoint_identity(
        schema31_config,
        repo_root=root,
        dataset=_dataset(),
        language=language,
        commit="1" * 40,
    )
    assert schema30_identity.manifest["schema"] == 30
    assert schema31_identity.manifest["schema"] == 31
    assert schema30_identity.manifest_digest != schema31_identity.manifest_digest

    torch.manual_seed(31)
    source = ClearVLAMainlinePolicy(schema31_config)
    source_spine = source.execution_bottom.decoder.spine
    assert source_spine is not None
    generator = torch.Generator().manual_seed(32)
    with torch.no_grad():
        for parameter in source_spine.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 1.0e-2
            )
    source_optimizer, _ = build_optimizer(source, schema31_config)
    source_schedule = WarmupCosineSchedule(
        source_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    for _ in range(3):
        source_schedule.step()
    schema31_path = tmp_path / "schema31-bspine.pt"
    save_checkpoint(
        schema31_path,
        model=source,
        optimizer=source_optimizer,
        schedule=source_schedule,
        config=schema31_config,
        identity=schema31_identity,
        epoch=2,
        global_step=3,
        best_metric=0.25,
    )

    torch.manual_seed(33)
    restored_model = ClearVLAMainlinePolicy(schema31_config)
    restored_optimizer, _ = build_optimizer(restored_model, schema31_config)
    restored_schedule = WarmupCosineSchedule(
        restored_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    restored = load_checkpoint_exact(
        schema31_path,
        model=restored_model,
        optimizer=restored_optimizer,
        schedule=restored_schedule,
        config=schema31_config,
        identity=schema31_identity,
    )
    assert restored.epoch == 2 and restored.global_step == 3
    for name, value in source.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], value), name

    torch.manual_seed(34)
    schema30_model = ClearVLAMainlinePolicy(schema30_config)
    schema30_optimizer, _ = build_optimizer(schema30_model, schema30_config)
    schema30_schedule = WarmupCosineSchedule(
        schema30_optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    schema30_path = tmp_path / "schema30.pt"
    save_checkpoint(
        schema30_path,
        model=schema30_model,
        optimizer=schema30_optimizer,
        schedule=schema30_schedule,
        config=schema30_config,
        identity=schema30_identity,
        epoch=0,
        global_step=0,
        best_metric=None,
    )
    before = {
        name: value.detach().clone()
        for name, value in restored_model.state_dict().items()
    }
    try:
        load_checkpoint_exact(
            schema30_path,
            model=restored_model,
            optimizer=restored_optimizer,
            schedule=restored_schedule,
            config=schema31_config,
            identity=schema31_identity,
        )
    except ValueError as error:
        assert "component selection differs" in str(error)
    else:
        raise AssertionError("Schema30 must not exact-resume into Schema31")
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, before[name]), name
