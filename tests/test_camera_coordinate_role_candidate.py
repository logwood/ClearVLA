from __future__ import annotations

import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    DatasetIdentity,
    build_checkpoint_identity,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import (
    WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION,
    WORLD_CAMERA_COORDINATE_ROLE_V1_SOURCE_PATHS,
    load_checkpoint_for_initialization,
    save_checkpoint,
)
from clearvla.mainline.training.optimizer import (
    WarmupCosineSchedule,
    build_optimizer,
)


def _dataset() -> DatasetIdentity:
    zero = hashlib.sha256(b"").hexdigest()
    return DatasetIdentity(
        raw_root="/data/test",
        hdf5_glob="*.hdf5",
        inventory_sha256=zero,
        state_normalizer_sha256=zero,
        action_normalizer_sha256=zero,
        decoded_cache_identity=zero,
        dino_cache_identity=zero,
    )


def _config() -> ExperimentConfig:
    base = ExperimentConfig()
    result = replace(
        base,
        dimensions=replace(
            base.dimensions,
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
        top=replace(base.top, grounder_iterations=2, teacher_key_dim=8),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_heads=4,
        ),
        optimizer=replace(base.optimizer, warmup_steps=2),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    result.validate()
    return result


def test_camera_condition_is_zero_init_and_rng_preserving() -> None:
    root = Path(__file__).resolve().parents[1]
    helpers = runpy.run_path(str(root / "tests/test_mainline_structural_contracts.py"))
    local_facts = helpers["_local_facts"]
    kwargs = dict(hidden=16, content_dim=8, route_dim=4, heads=4)
    torch.manual_seed(76139)
    legacy = ObjectFutureDynamicsCompiler(**kwargs).eval()
    legacy_rng = torch.get_rng_state().clone()
    torch.manual_seed(76139)
    candidate = ObjectFutureDynamicsCompiler(
        **kwargs,
        camera_names=("top", "wrist"),
        camera_condition_mode="coordinate_role_v1",
    ).eval()
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(legacy_rng, candidate_rng)
    old_state = legacy.state_dict()
    new_state = candidate.state_dict()
    assert set(new_state) - set(old_state) == {
        "camera_coordinate_role_condition.weight"
    }
    for name, value in old_state.items():
        torch.testing.assert_close(new_state[name], value, atol=0.0, rtol=0.0)
    assert torch.count_nonzero(
        new_state["camera_coordinate_role_condition.weight"]
    ) == 0

    with torch.no_grad():
        for name in ("delta_head", "transport_head", "covariance_head"):
            value = torch.randn_like(getattr(legacy, name).weight) * 0.1
            getattr(legacy, name).weight.copy_(value)
            getattr(candidate, name).weight.copy_(value)
    facts, _ = DenseObjectGrounder(
        hidden=16,
        content_dim=8,
        route_dim=4,
        objects=4,
        iterations=1,
    )(local_facts(batch=2, cameras=2, content=8, route=4, hidden=16))
    common = torch.randn(2, 4, 3, 16)
    interval = torch.randn(2, 4, 4, 3, 16)
    for dtype in (None, torch.bfloat16):
        context = (
            torch.autocast("cpu", dtype=dtype)
            if dtype is not None
            else torch.autocast("cpu", enabled=False)
        )
        with context:
            left = legacy._field(
                facts=facts,
                typed_common=common,
                typed_interval_innovation=interval,
            )
            right = candidate._field(
                facts=facts,
                typed_common=common,
                typed_interval_innovation=interval,
            )
        for name in ("semantic_delta", "transport_mean", "transport_covariance"):
            torch.testing.assert_close(
                getattr(right, name), getattr(left, name), atol=0.0, rtol=0.0
            )


def test_camera_checkpoint_migration_is_narrow_and_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    condition = tmp_path / "goal.pt"
    condition.write_bytes(b"t5-condition")
    language = ArtifactIdentity.from_file("t5_goal", condition)
    baseline = _config()
    target = replace(
        baseline,
        top=replace(
            baseline.top,
            world_camera_condition_mode="coordinate_role_v1",
        ),
    )
    saved_identity = build_checkpoint_identity(
        baseline,
        repo_root=root,
        dataset=_dataset(),
        language=language,
        commit="a" * 40,
    )
    target_identity = build_checkpoint_identity(
        target,
        repo_root=root,
        dataset=_dataset(),
        language=language,
        commit="a" * 40,
    )
    changed_paths = {
        "clearvla/mainline/config.py",
        "clearvla/mainline/model/dynamics.py",
        "clearvla/mainline/model/policy.py",
        "clearvla/mainline/model/top.py",
        "clearvla/mainline/runtime/checkpoints.py",
        "clearvla/mainline/train.py",
    }
    assert changed_paths == set(WORLD_CAMERA_COORDINATE_ROLE_V1_SOURCE_PATHS)
    rows = tuple(
        (
            name,
            hashlib.sha256(f"pre-b1:{name}".encode()).hexdigest()
            if name in changed_paths
            else digest,
        )
        for name, digest in saved_identity.source.files
    )
    saved_identity = replace(
        saved_identity,
        source=replace(
            saved_identity.source,
            files=rows,
            digest=hashlib.sha256(
                json.dumps(
                    rows,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest(),
        ),
    )
    source = ClearVLAMainlinePolicy(baseline)
    optimizer, _ = build_optimizer(source, baseline)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=4,
        minimum_ratio=0.1,
    )
    checkpoint = tmp_path / "pre-b1.pt"
    save_checkpoint(
        checkpoint,
        model=source,
        optimizer=optimizer,
        schedule=schedule,
        config=baseline,
        identity=saved_identity,
        epoch=1,
        global_step=0,
        best_metric=0.25,
    )
    candidate = ClearVLAMainlinePolicy(target)
    state = load_checkpoint_for_initialization(
        checkpoint,
        model=candidate,
        config=target,
        identity=target_identity,
        model_contract_migration=WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION,
    )
    assert set(state.changed_source_files) == changed_paths
    assert state.model_contract_migration == WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION
    key = "world.dynamics.camera_coordinate_role_condition.weight"
    assert torch.count_nonzero(candidate.state_dict()[key]) == 0
    for name, value in source.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, atol=0.0, rtol=0.0)

    nonzero = ClearVLAMainlinePolicy(target)
    with torch.no_grad():
        nonzero.world.dynamics.camera_coordinate_role_condition.weight.fill_(1.0)
    with pytest.raises(ValueError, match="exact-zero new condition weight"):
        load_checkpoint_for_initialization(
            checkpoint,
            model=nonzero,
            config=target,
            identity=target_identity,
            model_contract_migration=WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION,
        )
