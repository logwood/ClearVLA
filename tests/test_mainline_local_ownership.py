"""M3 contracts and real production-path integration, no learned-score claim."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_candidate_support import _config as support_config
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_state_features import _data, _model_engine, _training

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.local_ownership import (
    COUPLED_LOCAL_OWNERS,
    INDEPENDENT_LOCAL_OWNERS,
    TYPES,
    couple_local_observation,
    refine_local_observation,
)


def _config() -> ExperimentConfig:
    cfg = support_config()
    return replace(
        cfg, observation=replace(cfg.observation, local_ownership_mode=COUPLED_LOCAL_OWNERS)
    )


def test_selection_is_explicit_and_requires_uncompressed_support():
    old = support_config()
    assert "local_ownership_mode" not in cast(dict[str, object], old.as_dict()["observation"])
    assert config_from_mapping(old.as_dict()) == old
    cfg = _config()
    assert config_from_mapping(cfg.as_dict()) == cfg
    with pytest.raises(ValueError, match="support"):
        replace(
            cfg, observation=replace(cfg.observation, candidate_support_mode="moment_local_v1")
        ).validate()
    with pytest.raises(ValueError, match="local_ownership"):
        replace(cfg, observation=replace(cfg.observation, local_ownership_mode="guess")).validate()


def _evidence():
    torch.manual_seed(127)
    parent = torch.randn(2, 3, 4, 7).log_softmax(-1).requires_grad_()
    sources = {name: torch.randn_like(parent, requires_grad=True) for name in TYPES}
    valid = torch.ones_like(parent, dtype=torch.bool)
    return parent, sources, valid


def test_joint_identity_retains_soft_ambiguity_instead_of_stitching_conflicting_reads():
    parent = torch.zeros(1, 1, 2).log_softmax(-1)
    scores = {
        "semantic": torch.zeros_like(parent),
        "appearance": torch.tensor([[[10.0, -10.0]]]),
        "geometry": torch.tensor([[[-10.0, 10.0]]]),
    }
    law = couple_local_observation(scores, parent, torch.ones_like(parent, dtype=torch.bool))
    torch.testing.assert_close(law.candidate_probability, torch.full_like(parent, 0.5))
    # Attribute encoders remain distinct; one location law selects their values.
    appearance_values = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    position_values = torch.tensor([[[0.2], [0.8]]])
    rgb = law.candidate_probability @ appearance_values
    pos = law.candidate_probability @ position_values
    torch.testing.assert_close(rgb, torch.tensor([[[0.5, 0.5]]]))
    torch.testing.assert_close(pos, torch.tensor([[[0.5]]]))
    assert not torch.equal(scores["appearance"].softmax(-1), scores["geometry"].softmax(-1))


def test_candidate_permutation_replication_and_common_gauge_preserve_identity():
    parent, scores, valid = _evidence()
    law = couple_local_observation(scores, parent, valid)
    perm = torch.randperm(7)
    p = couple_local_observation(
        {n: s[..., perm] for n, s in scores.items()}, parent[..., perm], valid[..., perm]
    )
    torch.testing.assert_close(p.candidate_probability, law.candidate_probability[..., perm])
    torch.testing.assert_close(p.hypothesis_probability, law.hypothesis_probability)
    # Splitting each point into two identical samples halves its parent mass,
    # hence does not reward redundant sampling or change the M evidence.
    q = couple_local_observation(
        {n: s.repeat_interleave(2, -1) for n, s in scores.items()},
        parent.repeat_interleave(2, -1) - torch.log(torch.tensor(2.0)),
        valid.repeat_interleave(2, -1),
    )
    torch.testing.assert_close(
        q.candidate_probability.reshape(*parent.shape, 2).sum(-1), law.candidate_probability
    )
    torch.testing.assert_close(q.hypothesis_probability, law.hypothesis_probability)
    shifted = couple_local_observation({n: s + 2.0 for n, s in scores.items()}, parent, valid)
    torch.testing.assert_close(shifted.candidate_probability, law.candidate_probability)
    torch.testing.assert_close(shifted.hypothesis_probability, law.hypothesis_probability)


@pytest.mark.parametrize("empty", [False, True])
def test_invalid_values_do_not_create_mass_or_gradients(empty):
    parent, sources, valid = _evidence()
    valid[..., -1] = False
    valid[:, 0] = False
    if empty:
        valid[:] = False
    parent = parent.detach().masked_fill(~valid, torch.nan).requires_grad_()
    sources = {
        n: s.detach().masked_fill(~valid, torch.nan).requires_grad_() for n, s in sources.items()
    }
    law = couple_local_observation(sources, parent, valid)
    assert (
        torch.isfinite(law.candidate_probability).all()
        and torch.isfinite(law.hypothesis_log_probability).all()
    )
    assert torch.count_nonzero(law.candidate_probability[~valid]) == 0
    torch.testing.assert_close(
        law.hypothesis_probability.sum(-1), torch.ones_like(parent[..., 0, 0])
    )
    loss = law.candidate_probability.square().sum() + law.hypothesis_probability.square().sum()
    loss.backward()
    for value in [parent, *sources.values()]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[~valid]) == 0
    if empty:
        torch.testing.assert_close(
            law.hypothesis_probability, torch.full_like(law.hypothesis_probability, 0.25)
        )


def test_g3_typed_corrections_refine_one_law_and_ignore_missing_hypotheses():
    parent, sources, valid = _evidence()
    law = couple_local_observation(sources, parent, valid)
    support = valid.any(-1)
    support[..., -1] = False
    residuals = {
        n: torch.randn_like(law.hypothesis_log_probability)
        .masked_fill(~support, torch.nan)
        .requires_grad_()
        for n in TYPES
    }
    out = refine_local_observation(law.hypothesis_log_probability, residuals, support)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out.exp()[~support]) == 0
    out.exp().square().sum().backward()
    for value in residuals.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[~support]) == 0
        assert value.grad[support].abs().sum() > 0


@pytest.mark.parametrize("center", [0, 1, 8, 79])
@torch.no_grad()
def test_production_g2_g3_binder_and_teacher_share_location_measure(center):
    cfg = _config()
    batch = _training(_data(), (center,))
    model, _ = _model_engine(cfg)
    cache, training, _ = model.encode_online(batch.online)
    state = training.observation.progressive_state
    assert state.local_ownership_mode == COUPLED_LOCAL_OWNERS
    assert state.g2_semantic_probability is not None
    assert state.g2_appearance_probability is not None and state.g2_geometry_probability is not None
    torch.testing.assert_close(
        state.g2_semantic_probability, state.fine_probability, rtol=0, atol=0
    )
    torch.testing.assert_close(
        state.g2_appearance_probability, state.fine_probability, rtol=0, atol=0
    )
    torch.testing.assert_close(
        state.g2_geometry_probability, state.fine_probability, rtol=0, atol=0
    )
    grounded = state.grounded_fact_set
    assert grounded is not None
    torch.testing.assert_close(
        grounded.semantic_owner_log_probs, grounded.appearance_owner_log_probs, rtol=0, atol=0
    )
    torch.testing.assert_close(
        grounded.semantic_owner_log_probs, grounded.geometry_owner_log_probs, rtol=0, atol=0
    )
    assert not torch.equal(grounded.semantic_slots, grounded.appearance_slots)
    facts = training.top.facts
    torch.testing.assert_close(
        facts.semantic_candidate_assignment, facts.geometry_candidate_assignment, rtol=0, atol=0
    )
    torch.testing.assert_close(
        facts.appearance_candidate_assignment, facts.geometry_candidate_assignment, rtol=0, atol=0
    )
    target, _ = model.build_training_targets(training, batch.future)
    assert target.teacher_dynamics is not None
    target.teacher_dynamics.validate()


@pytest.mark.parametrize("bf16", [False, True])
def test_actual_action_loss_reaches_each_typed_evidence_owner(bf16):
    cfg = _config()
    model, _ = _model_engine(cfg)
    batch = _training(_data(), (8,))
    # The state-chart transport fixture repeats one scalar across all DINO
    # channels; LayerNorm correctly maps that degenerate semantic key to zero.
    # Use nondegenerate, deterministic observation features for connectivity,
    # without modifying production parameters or injecting privileged labels.
    dino = torch.randn(
        batch.online.observation.dino_history.shape, generator=torch.Generator().manual_seed(127)
    )
    batch = replace(
        batch,
        online=replace(
            batch.online, observation=replace(batch.online.observation, dino_history=dino)
        ),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, training, _ = model.encode_online(batch.online)
        state = training.observation.progressive_state
        assert state.fine_probability is not None and state.fine_probability.dtype == torch.float32
        for probability in (
            state.g2_semantic_probability,
            state.g2_appearance_probability,
            state.g2_geometry_probability,
        ):
            assert probability is not None and probability.dtype == torch.float32
            torch.testing.assert_close(probability, state.fine_probability, rtol=0, atol=0)
        output = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = output.bottom.physical_velocity.float().square().mean()
    loss.backward()
    address = model.observation.compiler.encoder.progressive_grounding_address
    assert (
        address is not None
        and address.g2_typed_query is not None
        and address.g3_owner_residual is not None
    )
    for n in TYPES:
        for owner in (address.g2_typed_query[n], address.g3_owner_residual[n]):
            grads = [p.grad for p in owner.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_exact_resume_and_deployment_keep_local_identity(tmp_path: Path):
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
    assert "clearvla/vision/local_ownership.py" in dict(identity.source.files)
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
            del obs["local_ownership"]
        else:
            cast(dict[str, object], obs["local_ownership"])["mode"] = INDEPENDENT_LOCAL_OWNERS
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
            config=support_config(),
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
