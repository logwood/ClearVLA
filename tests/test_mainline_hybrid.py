from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_mainline_bspine_arm_only_integration import _arm_only_config
from test_mainline_policy import _batch

from clearvla.mainline.config import ExperimentConfig, HybridSolverConfig, config_from_mapping
from clearvla.mainline.manifest import architecture_manifest_for_bspine_implementation
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.hybrid import (
    differentiable_hybrid_rollout,
    integrate_hybrid_pass,
    role_contract,
)
from clearvla.mainline.runtime.logging import archival_metrics
from clearvla.mainline.runtime.sampling import (
    _integrate_cache,
    sample_refined_cached_action_with_cache,
)
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.hybrid import hybrid_rollout_terms
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def hybrid_config(checkpoint_rollout=True):
    base = _arm_only_config()
    return replace(
        base,
        hybrid=HybridSolverConfig(
            enabled=True,
            rollout_loss_weight=1.0,
            checkpoint_rollout=checkpoint_rollout,
        ),
    )


def test_hybrid_identity_is_opt_in_and_round_trips():
    baseline = ExperimentConfig()
    assert "hybrid" not in baseline.as_dict()
    assert config_from_mapping(baseline.as_dict()) == baseline
    config = hybrid_config()
    assert config_from_mapping(config.as_dict()) == config
    manifest = architecture_manifest_for_bspine_implementation(
        config.bottom.bspine_implementation,
        hybrid_solver=True,
    )
    assert manifest.schema == 32
    assert "hybrid_v1" in manifest.components.training
    assert "schema32" in ComponentSelection.from_config(config).objectives
    with pytest.raises(ValueError):
        replace(baseline, hybrid=config.hybrid).validate()
    with pytest.raises(ValueError):
        HybridSolverConfig(enabled=True).validate()
    with pytest.raises(TypeError):
        HybridSolverConfig(enabled="false").validate()


def test_hybrid_euler_pass_preserves_existing_sampling_math():
    torch.manual_seed(323201)
    config = hybrid_config(False)
    model = ClearVLAMainlinePolicy(config).eval()
    model.set_training_step(1200)
    batch = _batch(config)
    noise = torch.randn(1, 24, 18)
    with torch.no_grad():
        cache, _, _ = model.encode_online(batch.online, geometry_supervision=False)
        reference = _integrate_cache(
            model,
            cache,
            config,
            generator=None,
            initial_physical_noise=noise,
            collect_diagnostics=True,
            dtype=torch.float32,
        )
        candidate = integrate_hybrid_pass(
            model,
            cache,
            config,
            noise,
            method="euler",
            collect_diagnostics=True,
            dtype=torch.float32,
        )
    for name in (
        "action",
        "physical_field",
        "motion_logits",
        "step_times",
        "initial_physical_noise",
    ):
        torch.testing.assert_close(
            getattr(candidate, name), getattr(reference, name), atol=0, rtol=0
        )
    for name in reference.metrics:
        torch.testing.assert_close(candidate.metrics[name], reference.metrics[name], atol=0, rtol=0)


def test_rollout_and_deployment_use_same_shared_field_and_fresh_w():
    torch.manual_seed(323202)
    config = hybrid_config()
    model = ClearVLAMainlinePolicy(config).train()
    model.set_training_step(1200)
    batch = _batch(config)
    with torch.no_grad():
        model.eval()
        cache, _, _ = model.encode_online(batch.online, geometry_supervision=False)
    model.train()
    noise = torch.randn(1, 24, 18)
    original = noise.clone()
    calls = []
    original_velocity = model.velocity

    def capture(*args, **kwargs):
        calls.append(
            (
                args[0],
                kwargs["noisy_action_field"].detach().clone(),
                kwargs["time"].detach().clone(),
                torch.is_grad_enabled(),
                model.training,
            )
        )
        return original_velocity(*args, **kwargs)

    with patch.object(model, "velocity", side_effect=capture):
        rollout = differentiable_hybrid_rollout(model, cache, config, noise, dtype=torch.float32)
    assert model.training
    assert len(calls) == 17
    assert all(row[3] and not row[4] for row in calls)
    assert all(row[0] is cache for row in calls[:6])
    assert all(row[0] is rollout.refined_cache for row in calls[6:])
    assert rollout.refined_cache is not cache
    assert torch.equal(calls[0][1], calls[6][1])
    assert torch.equal(noise, original)
    assert rollout.refined_cache.top.action_condition.interval_action.requires_grad
    with torch.no_grad():
        deployed, _ = sample_refined_cached_action_with_cache(
            model,
            cache,
            config,
            initial_physical_noise=noise,
            dtype=torch.float32,
        )
    for name in ("action", "physical_field", "motion_logits", "step_times"):
        torch.testing.assert_close(
            getattr(deployed, name), getattr(rollout.refined, name), atol=0, rtol=0
        )
    contract = role_contract(model)
    assert contract.identity["state_shape"] == [24, 18]
    assert contract.identity["roles"]["arm_field"]["retain_raw"]
    assert contract.identity["roles"]["continuous_gripper_field"]["chart"] == "identity_raw"
    assert rollout.refined.metrics["hybrid_role_retained_identity_max_abs"].item() == 0
    with pytest.raises(ValueError, match="fastpath"):
        sample_refined_cached_action_with_cache(
            model,
            cache,
            config,
            initial_physical_noise=noise,
            deployment_fastpath=True,
        )


def test_rollout_only_loss_reaches_spine_world_and_online_owners():
    torch.manual_seed(323203)
    config = hybrid_config()
    model = ClearVLAMainlinePolicy(config).eval()
    model.set_training_step(1200)
    # W's output matrices are deliberately zero at cold start, so input VJP
    # there is zero until its ordinary Teacher loss updates those matrices.
    # Audit the connected post-initialization path without changing run init.
    with torch.no_grad():
        for head in (
            model.world.dynamics.delta_head,
            model.world.dynamics.transport_head,
            model.world.dynamics.covariance_head,
        ):
            head.weight.normal_(std=1e-3)
    batch = _batch(config)
    cache, _, _ = model.encode_online(batch.online, geometry_supervision=False)
    noise = torch.randn(1, 24, 18)
    rollout = differentiable_hybrid_rollout(model, cache, config, noise, dtype=torch.float32)
    rollout.proposal.action.retain_grad()
    terms = hybrid_rollout_terms(config, rollout, batch.action_target, batch.online.history)
    terms["hybrid_rollout"].backward()
    assert rollout.proposal.action.grad is not None
    assert rollout.proposal.action.grad.abs().sum() > 0
    for prefix in (
        "observation.",
        "grounding.",
        "intent.",
        "world.",
        "p1.",
        "policy_compiler.",
        "transition.",
        "execution_bottom.decoder.spine.coarse_lifts.",
        "execution_bottom.decoder.spine.detail_lifts.",
        "execution_bottom.decoder.terminal_controller.velocity_head.",
    ):
        grads = [
            p.grad
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        ]
        assert grads, prefix
        assert all(torch.isfinite(g).all() for g in grads), prefix
        assert sum(g.abs().sum().item() for g in grads) > 0, prefix


def test_full_train_step_has_real_hybrid_contributions_and_no_default_drift():
    torch.manual_seed(323204)
    config = hybrid_config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1)
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    engine.global_step = 1200
    result = engine.train_step(_batch(config), collect_diagnostics=True)
    metrics = result.materialize()
    assert torch.isfinite(result.loss)
    assert metrics["hybrid_solver_total_dynamic_calls"] == 17
    assert abs(metrics["loss_contribution_gap"]) < 1e-5
    assert abs(metrics["loss_ledger_gap"]) < 1e-5
    assert metrics["loss_contrib_hybrid_rollout_decoded_action"] > 0
    assert metrics["loss_contrib_hybrid_rollout_transition"] > 0
    assert "hybrid_solver_total_dynamic_calls" in archival_metrics(metrics)


def test_event_free_hold_rows_are_supervised_and_masks_cover_all_rows():
    config = hybrid_config()
    batch = _batch(config)
    zero = torch.zeros_like(batch.action_target.normalized)
    boundary = torch.zeros_like(batch.action_target.gripper_transition_boundary)
    target = replace(
        batch.action_target,
        normalized=zero,
        raw_units=zero,
        gripper_transition_boundary=boundary,
        gripper_transition_boundary_raw_units=boundary,
    )
    history = replace(batch.online.history, codec_gripper_boundary=torch.zeros(1, 1))
    predicted = torch.ones_like(zero, requires_grad=True)
    terms = hybrid_rollout_terms(config, SimpleNamespace(refined_action=predicted), target, history)
    assert terms["hybrid_rollout_hold"] > 0
    assert terms["hybrid_rollout_transition"] == 0
    assert terms["hybrid_rollout_persistence"] == 0
    assert terms["hybrid_rollout_hold_row_fraction"] == 1
    terms["hybrid_rollout"].backward()
    assert predicted.grad[..., -1].abs().sum() > 0

def test_hybrid_checkpoint_and_deployment_abi_round_trip(tmp_path):
    import hashlib
    from pathlib import Path

    import numpy as np
    from test_mainline_checkpoint import _dataset

    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.data.normalizer import ArrayNormalizer
    from clearvla.mainline.runtime.checkpoints import (
        load_checkpoint_for_validation,
        save_checkpoint,
    )
    from clearvla.mainline.runtime.deployment import (
        build_deployment_abi,
        canonical_sha256,
        deployment_config_from_checkpoint,
    )

    config = hybrid_config()
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1)
    identity = build_checkpoint_identity(
        config, repo_root=Path(__file__).resolve().parents[1], dataset=_dataset(),
        language=ArtifactIdentity("test", "<test>", 0, hashlib.sha256(b"").hexdigest()),
    )
    normalizer = ArrayNormalizer.fit_zscore([np.zeros((2, 7), dtype=np.float32)])
    abi = build_deployment_abi(
        config, identity, action_normalizer=normalizer, state_normalizer=normalizer,
        data_profile={"name": "identity_7d_pen", "gripper_transition_boundary": "current_action_state"},
        gripper_indices=(6,), goal_metadata={},
    )
    restored_config = deployment_config_from_checkpoint(config.as_dict(), abi)
    assert restored_config.hybrid == config.hybrid
    assert abi["architecture_manifest"]["schema"] == 32
    corrupt_abi = dict(abi, graph_config=dict(abi["graph_config"]))
    corrupt_abi["graph_config"].pop("hybrid")
    corrupt_abi["graph_config_sha256"] = canonical_sha256(corrupt_abi["graph_config"])
    with pytest.raises(ValueError, match="hybrid"):
        deployment_config_from_checkpoint(config.as_dict(), corrupt_abi)
    filename = tmp_path / "hybrid.pt"
    save_checkpoint(
        filename, model=model, optimizer=optimizer, schedule=schedule,
        config=config, identity=identity, epoch=0, global_step=0, best_metric=None,
    )
    before = hashlib.sha256(filename.read_bytes()).hexdigest()
    fresh = ClearVLAMainlinePolicy(config)
    restored = load_checkpoint_for_validation(filename, model=fresh, config=config, identity=identity)
    assert restored.global_step == 0
    assert hashlib.sha256(filename.read_bytes()).hexdigest() == before
    for key,value in model.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[key], value, atol=0, rtol=0)
