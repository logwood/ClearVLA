"""Completed G3 evidence must reach the real entity binder, not only P1."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_local_ownership import _config as local_config
from test_mainline_state_features import _data, _model_engine, _training
from test_mainline_structural_contracts import _local_facts

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.candidate_support import sample_candidate_expectation


def _config() -> ExperimentConfig:
    cfg = local_config()
    return replace(cfg, top=replace(cfg.top, entity_context_mode="completed_g3_v1"))


def test_context_selection_is_explicit_and_requires_real_support():
    legacy = local_config()
    assert "entity_context_mode" not in cast(dict[str, object], legacy.as_dict()["top"])
    assert config_from_mapping(legacy.as_dict()) == legacy
    cfg = _config()
    assert config_from_mapping(cfg.as_dict()) == cfg
    with pytest.raises(ValueError, match="entity_context"):
        replace(cfg, top=replace(cfg.top, entity_context_mode="unknown")).validate()
    with pytest.raises(ValueError):
        replace(cfg, observation=replace(cfg.observation, candidate_support_mode="moment_local_v1")).validate()


@pytest.mark.parametrize("empty", [False, True])
def test_real_binder_masks_context_before_projection_and_reconstruction(empty):
    torch.manual_seed(227)
    local = _local_facts()
    shape = (*local.content_slots.shape[:-1], 32)
    valid = local.slot_validity.clone()
    valid[..., -1, :] = 0
    if empty:
        valid.zero_()
    context = torch.randn(shape).masked_fill(~valid.bool(), torch.nan).requires_grad_()
    local = replace(local, slot_validity=valid, context_slots=context)
    binder = DenseObjectGrounder(hidden=32, content_dim=16, route_dim=8, entity_context_mode="completed_g3_v1")
    facts, _ = binder(local, collect_diagnostics=False)
    assert torch.isfinite(facts.content).all() and torch.isfinite(facts.reconstruction_error)
    loss = facts.reconstruction_error + facts.content.square().sum()
    loss.backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert torch.count_nonzero(context.grad.masked_select(~valid.bool())) == 0
    if not empty:
        assert context.grad.abs().sum() > 0
        assert binder.context_key is not None
        for p in binder.context_key.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    else:
        assert torch.count_nonzero(context.grad) == 0
    # Learning context never replaces the independently observed target.
    torch.testing.assert_close(facts.dense_chart.dino_content, local.target_dino_content)


def test_context_is_required_not_a_silent_fallback_or_ignored_sidecar():
    local = _local_facts()
    new = DenseObjectGrounder(hidden=32, content_dim=16, route_dim=8, entity_context_mode="completed_g3_v1")
    with pytest.raises(ValueError, match="requires"):
        new(local, collect_diagnostics=False)
    local = replace(local, context_slots=torch.zeros(*local.content_slots.shape[:-1], 32))
    old = DenseObjectGrounder(hidden=32, content_dim=16, route_dim=8)
    with pytest.raises(ValueError, match="undeclared"):
        old(local, collect_diagnostics=False)
    assert local.context_slots is not None
    with pytest.raises(ValueError):
        replace(local, context_slots=local.context_slots[..., :31]).validate()


def test_current_chart_gradient_survives_channel_tiling_with_fixed_locations():
    torch.manual_seed(229)
    chart = torch.randn(1, 2, 3, 3, 257, requires_grad=True)
    coordinates = torch.rand(1, 2, 2, 2, 4, 7, 2) * 2 - 1
    probability = torch.randn(1, 2, 2, 2, 4, 7).softmax(-1)
    out = sample_candidate_expectation(chart, coordinates, probability)
    out.square().sum().backward()
    assert chart.grad is not None and torch.isfinite(chart.grad).all()
    assert (chart.grad.square().sum((0, 1, 2, 3)) > 0).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_completed_third_g_block_reaches_entities_without_policy_bypass(bf16):
    cfg = _config()
    model, _ = _model_engine(cfg)
    batch = _training(_data(), (8,))
    dino = torch.randn(batch.online.observation.dino_history.shape, generator=torch.Generator().manual_seed(227))
    batch = replace(batch, online=replace(batch.online, observation=replace(batch.online.observation, dino_history=dino)))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        _, train, _ = model.encode_online(batch.online)
        local = train.observation.local_facts
        state = train.observation.progressive_state
        assert local.context_slots is not None and local.context_slots.requires_grad
        local.context_slots.retain_grad()
        local.public_scene_base.retain_grad()
        assert state.dynamic_fine_coordinates is not None and state.g2_geometry_probability is not None
        expected = sample_candidate_expectation(local.public_scene_base, state.dynamic_fine_coordinates, state.g2_geometry_probability)
        torch.testing.assert_close(local.context_slots, expected, rtol=0, atol=0)
        assert not local.target_dino_content.requires_grad
        # Entity-only objective excludes dynamic P1/P2/decoder shortcut paths.
        facts = train.top.facts
        objective = facts.content.square().mean() + facts.candidate_assignment.square().mean()
    objective.backward()
    assert local.context_slots.grad is not None and torch.isfinite(local.context_slots.grad).all()
    assert local.context_slots.grad.abs().sum() > 0
    assert local.public_scene_base.grad is not None and local.public_scene_base.grad.abs().sum() > 0
    third = [p.grad for p in model.grounding.blocks[2].parameters() if p.requires_grad and p.grad is not None]
    assert third and all(torch.isfinite(g).all() for g in third)
    assert sum(float(g.abs().sum()) for g in third) > 0
    assert model.grounding.grounder.context_key is not None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.grounding.grounder.context_key.parameters())


@torch.no_grad()
def test_altering_completed_context_changes_real_binding_but_not_observed_target():
    torch.manual_seed(239)
    local = _local_facts()
    context = torch.randn(*local.content_slots.shape[:-1], 32)
    binder = DenseObjectGrounder(hidden=32, content_dim=16, route_dim=8, entity_context_mode="completed_g3_v1")
    a, _ = binder(replace(local, context_slots=context), collect_diagnostics=False)
    b, _ = binder(replace(local, context_slots=context.flip(-1)), collect_diagnostics=False)
    assert not torch.equal(a.candidate_assignment, b.candidate_assignment)
    torch.testing.assert_close(a.dense_chart.dino_content, b.dense_chart.dino_content, rtol=0, atol=0)

def test_real_train_save_reload_and_deployment_keep_g3_context(tmp_path: Path):
    cfg, data = _config(), _data()
    batch = _training(data, (8,))
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    model.eval()
    lang = tmp_path / "language.pt"
    torch.save({"tokens": batch.online.goal.tokens, "mask": batch.online.goal.mask}, lang)
    dataset = replace(
        identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()),
    )
    identity = build_checkpoint_identity(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset,
        language=ArtifactIdentity.from_file("t5_goal", lang),
        commit="6" * 40,
    )
    assert "clearvla/mainline/model/grounding.py" in dict(identity.source.files)
    profile = resolve_action_state_profile("calvin_relative_7d_v1")
    abi = build_deployment_abi(
        cfg,
        identity,
        action_normalizer=data.action_normalizer,
        state_normalizer=data.state_normalizer,
        data_profile={
            **profile.as_dict(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        gripper_indices=(6,),
        goal_metadata={},
    )
    validate_deployment_abi(abi)
    for mutation in ("missing", "wrong"):
        bad = copy.deepcopy(abi)
        obs = cast(dict[str, object], bad["observation"])
        if mutation == "missing":
            del obs["entity_context"]
        else:
            cast(dict[str, object], obs["entity_context"])["read"] = "barycenter"
        with pytest.raises(ValueError):
            validate_deployment_abi(bad)
    p = tmp_path / "model.pt"
    save_checkpoint(
        p,
        model=model,
        optimizer=engine.optimizer,
        schedule=engine.schedule,
        config=cfg,
        identity=identity,
        epoch=0,
        global_step=engine.global_step,
        best_metric=None,
        data_state={
            "action_normalizer": data.action_normalizer.to_dict(),
            "state_normalizer": data.state_normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    m, e = _model_engine(cfg)
    load_checkpoint_exact(
        p, model=m, optimizer=e.optimizer, schedule=e.schedule, config=cfg, identity=identity
    )
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            p,
            model=m,
            optimizer=e.optimizer,
            schedule=e.schedule,
            config=local_config(),
            identity=identity,
        )
    restored = load_deployment_checkpoint(p, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(127))
        b = sample_action(
            restored.model,
            batch.online,
            restored.config,
            generator=torch.Generator().manual_seed(127),
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
