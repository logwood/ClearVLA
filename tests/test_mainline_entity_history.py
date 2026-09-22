"""M4c: source-timed past observations enter real K competition.

External HDF5/image/token I/O is replaced by the declared source fixtures only;
encoders, binder, losses, optimizer, checkpoints and sampling are production.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_entity_chart import _config as image_config
from test_mainline_entity_chart import _local_at
from test_mainline_state_features import _data, _model_engine, _training

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.entity_chart import (
    CURRENT_IMAGE_CHART,
    CurrentImageSupport,
    current_image_grid,
)
from clearvla.vision.entity_history import (
    CAUSAL_ENTITY_HISTORY,
    CausalEntityEvidence,
    ObservedEntityHistory,
    entity_history_metadata,
    pull_causal_history,
)
from clearvla.vision.source_time import VisualSourceTime


def _config() -> ExperimentConfig:
    cfg = image_config()
    return replace(cfg, top=replace(cfg.top, entity_history_mode=CAUSAL_ENTITY_HISTORY))


def _history(
    offsets: tuple[int, ...] = (-8, -4, 0), cameras: int = 2, side: int = 5, dim: int = 16
) -> ObservedEntityHistory:
    clock = VisualSourceTime(torch.tensor([offsets]))
    shape = (1, len(offsets), cameras, side, side, dim)
    content = torch.randn(shape, generator=torch.Generator().manual_seed(1021))
    obs = clock.frame_observed[:, :, None, None, None].expand(*shape[:-1]).clone()
    flow = torch.zeros(1, len(offsets) - 1, cameras, 2, side, side)
    status = torch.ones(1, len(offsets) - 1, cameras, 1, side, side)
    return ObservedEntityHistory(content, obs, clock, flow, status, torch.zeros_like(status))


def _support(cameras=2) -> CurrentImageSupport:
    xy = torch.tensor([[-0.6, 0.2], [0.7, -0.4]]).reshape(1, 1, 1, 1, 1, 2, 2)
    xy = xy.expand(1, cameras, 2, 3, 4, 2, 2).clone()
    log = torch.full(xy.shape[:-1], 0.5).log()
    return CurrentImageSupport(xy, log.exp(), torch.ones_like(log, dtype=torch.bool), log)


def _batch(center=8):
    batch = _training(_data(), (center,))
    obs = batch.online.observation
    dino = torch.randn(obs.dino_history.shape, generator=torch.Generator().manual_seed(1023))
    return replace(batch, online=replace(batch.online, observation=replace(obs, dino_history=dino)))


def test_mode_and_consumer_are_explicit():
    old, cfg = image_config(), _config()
    assert "entity_history_mode" not in cast(dict[str, object], old.as_dict()["top"])
    assert config_from_mapping(old.as_dict()) == old
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert (
        load_config("configs/mainline/structural_rebuild_m4c_calvin.json").top.entity_history_mode
        == CAUSAL_ENTITY_HISTORY
    )
    for changes in ({"entity_history_mode": "typo"}, {"entity_chart_mode": "query_lattice_v1"}):
        with pytest.raises(ValueError):
            replace(cfg, top=replace(cfg.top, **changes)).validate()
    with pytest.raises(ValueError):
        replace(
            cfg, observation=replace(cfg.observation, source_time_mode="fixed_history_steps_v1")
        ).validate()
    assert "not_persistent_ids" in str(entity_history_metadata(CAUSAL_ENTITY_HISTORY)["identity"])


@pytest.mark.parametrize("offsets", [(0, 0, 0), (-1, -1, 0), (-8, -4, 0)])
def test_source_time_not_row_index_or_duplicate_pixels(offsets):
    h = _history(offsets)
    a = pull_causal_history(h, h.content)
    torch.testing.assert_close(a.offsets, h.source_time.frame_offsets[:, :-1])
    for i in range(2):
        if h.source_time.frame_observed[0, i]:
            torch.testing.assert_close(a.values[:, i], h.content[:, i])
            assert (a.coverage[:, i] == 1).all()
        else:
            assert torch.count_nonzero(a.coverage[:, i]) == 0
            assert torch.count_nonzero(a.values[:, i]) == 0
    if offsets == (0, 0, 0):
        out = CausalEntityEvidence(16, 8)(h, _support())
        assert torch.count_nonzero(out) == 0


def test_older_map_is_composed_not_latest_flow_times_two():
    h = _history(cameras=1, side=7, dim=2)
    grid = current_image_grid(7, 7, device=torch.device("cpu"))
    content = grid[None, None, None].expand(1, 3, 1, 7, 7, 2).clone()
    flow = h.backward_flow.clone()
    flow[:, 1, :, 0] = 0.2
    # Older x motion varies with the PREVIOUS-image x coordinate.
    flow[:, 0, :, 0] = grid[..., 0] * 0.1
    h = replace(h, content=content, backward_flow=flow)
    a = pull_causal_history(h, content)
    torch.testing.assert_close(
        a.coordinates[0, 1, 0, 3, 3], torch.tensor([0.2, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        a.coordinates[0, 0, 0, 3, 3], torch.tensor([0.22, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        a.values[0, 0, 0, 3, 3], torch.tensor([0.22, 0.0]), atol=1e-6, rtol=0
    )
    # Image-chart flow is per observed pair. Changing dt must NOT rescale maps.
    b = pull_causal_history(
        replace(h, source_time=VisualSourceTime(torch.tensor([[-9, -1, 0]]))), content
    )
    torch.testing.assert_close(a.coordinates, b.coordinates, rtol=0, atol=0)


def test_outside_paths_cannot_reenter_and_missing_content_is_not_stationary_motion():
    h = _history(cameras=1)
    flow = h.backward_flow.clone()
    flow[:, 1, :, 0] = 3
    flow[:, 0, :, 0] = -3
    a = pull_causal_history(replace(h, backward_flow=flow), h.content)
    assert torch.count_nonzero(a.values) == torch.count_nonzero(a.coverage) == 0
    mask = h.observed.clone()
    mask[:, :-1] = False
    empty = replace(h, observed=mask, content=h.content.masked_fill(~mask[..., None], torch.nan))
    out = CausalEntityEvidence(16, 8)(empty, _support(cameras=1))
    assert torch.count_nonzero(out) == 0 and torch.isfinite(out).all()


def test_observation_support_is_not_flow_confidence_or_occlusion():
    h = _history()
    a = pull_causal_history(h, h.content)
    b = pull_causal_history(
        replace(
            h, confidence=torch.zeros_like(h.confidence), occlusion=torch.ones_like(h.occlusion)
        ),
        h.content,
    )
    torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
    torch.testing.assert_close(a.coverage, b.coverage, rtol=0, atol=0)
    assert torch.count_nonzero(b.confidence) == 0
    assert (b.occlusion == 1).all()


def test_masked_interpolation_keeps_coverage_without_dividing_by_it():
    h = _history(cameras=1, side=3, dim=2)
    mask = h.observed.clone()
    mask[:, 1, :, 1, 1] = False
    content = torch.ones_like(h.content)
    content[~mask] = torch.nan
    flow = h.backward_flow.clone()
    flow[:, 1, :, 0] = 0.5
    h = replace(h, observed=mask, content=content, backward_flow=flow)
    a = pull_causal_history(h, content)
    assert a.coverage[0, 1, 0, 1, 1, 0] == 0.5
    torch.testing.assert_close(a.values[0, 1, 0, 1, 1], torch.tensor([0.5, 0.5]))


def test_masked_values_and_duplicate_flow_have_zero_finite_gradients():
    h = _history((-1, -1, 0))
    mask = h.observed.clone()
    mask[:, :-1, :, :, 2] = False
    content = h.content.masked_fill(~mask[..., None], torch.nan).requires_grad_()
    flow = h.backward_flow.clone()
    flow[:, 0] = torch.nan
    flow.requires_grad_()
    h = replace(h, content=content, observed=mask, backward_flow=flow)
    module = CausalEntityEvidence(16, 8)
    out = module(h, _support())
    out.square().sum().backward()
    assert content.grad is not None and flow.grad is not None
    assert torch.isfinite(content.grad).all() and torch.isfinite(flow.grad).all()
    assert torch.count_nonzero(content.grad[~mask]) == 0
    assert torch.count_nonzero(flow.grad[:, 0]) == 0
    assert content.grad[mask].abs().sum() > 0 and flow.grad[:, 1].abs().sum() > 0


def test_camera_permutation_is_camera_local_and_missing_view_cannot_contaminate_other_view():
    h = _history()
    s = _support()
    module = CausalEntityEvidence(16, 8)
    a = module(h, s)
    hp = replace(
        h,
        content=h.content.flip(2),
        observed=h.observed.flip(2),
        backward_flow=h.backward_flow.flip(2),
        confidence=h.confidence.flip(2),
        occlusion=h.occlusion.flip(2),
    )
    sp = CurrentImageSupport(
        s.coordinates.flip(1), s.probability.flip(1), s.valid.flip(1), s.log_probability.flip(1)
    )
    torch.testing.assert_close(module(hp, sp), a.flip(1))
    content = h.content.clone()
    content[:, :-1, 0] *= 3
    b = module(replace(h, content=content), s)
    torch.testing.assert_close(a[:, 1], b[:, 1], rtol=0, atol=0)


def test_full_candidate_expectation_not_empty_barycenter_and_keeps_address_gradients():
    h = _history()
    s = _support()
    module = CausalEntityEvidence(16, 8)
    xy = s.coordinates.clone().requires_grad_()
    logs = torch.randn_like(s.log_probability, requires_grad=True)
    logp = logs.log_softmax(-1)
    s = CurrentImageSupport(xy, logp.exp(), s.valid, logp)
    a = module(h, s)
    # Reference integrates independent point reads, not the helper under test.
    singles: list[torch.Tensor] = []
    for i in range(2):
        si = CurrentImageSupport(
            xy[..., i : i + 1, :],
            torch.ones_like(logp[..., i : i + 1]),
            s.valid[..., i : i + 1],
            torch.zeros_like(logp[..., i : i + 1]),
        )
        singles.append(module(h, si))
    reference = sum(v * logp[..., i, None].exp() for i, v in enumerate(singles))
    torch.testing.assert_close(a, reference, atol=2e-7, rtol=2e-6)
    a.square().sum().backward()
    for value in (xy, logs):
        assert (
            value.grad is not None
            and torch.isfinite(value.grad).all()
            and value.grad.abs().sum() > 0
        )
    invalid = replace(
        s,
        coordinates=torch.full_like(xy, torch.nan),
        probability=torch.zeros_like(logp),
        valid=torch.zeros_like(s.valid),
        log_probability=torch.zeros_like(logp),
    )
    assert torch.count_nonzero(module(h, invalid)) == 0


@pytest.mark.parametrize(
    "kind", ["clock", "observed_duplicate", "shape", "nan_real", "current_not_zero"]
)
def test_malformed_history_is_rejected(kind):
    h = _history()
    if kind == "clock":
        h = replace(h, source_time=VisualSourceTime(torch.tensor([[-8, 1, 0]])))
    if kind == "current_not_zero":
        h = replace(h, source_time=VisualSourceTime(torch.tensor([[-8, -4, 1]])))
    if kind == "observed_duplicate":
        h = replace(h, source_time=VisualSourceTime(torch.tensor([[0, 0, 0]])))
    if kind == "shape":
        h = replace(h, backward_flow=h.backward_flow[:, :1])
    if kind == "nan_real":
        h = replace(h, content=h.content * torch.nan)
    with pytest.raises(ValueError):
        h.validate()


def test_real_binder_requires_temporal_provenance_not_silent_legacy_fallback():
    local = _local_at((0.1, 0.2))
    binder = DenseObjectGrounder(
        hidden=32,
        content_dim=16,
        route_dim=8,
        entity_chart_mode=CURRENT_IMAGE_CHART,
        entity_history_mode=CAUSAL_ENTITY_HISTORY,
    )
    with pytest.raises(ValueError, match="requires"):
        binder(local)
    h = _history(
        cameras=local.target_dino_content.shape[1], side=local.target_dino_content.shape[2]
    )
    content = h.content.clone()
    content[:, -1] = local.target_dino_content
    h = replace(h, content=content)
    local = replace(local, observed_history=h, latest_flow_steps=h.source_time.pair_steps[:, -1])
    facts, _ = binder(local)
    old = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, entity_chart_mode=CURRENT_IMAGE_CHART
    )
    with pytest.raises(ValueError, match="undeclared"):
        old(local)
    # History cannot redefine the actual observation target or source support.
    torch.testing.assert_close(
        facts.dense_chart.dino_content, local.target_dino_content, rtol=0, atol=0
    )
    torch.testing.assert_close(facts.dense_chart.cell_observed, local.cell_observed, rtol=0, atol=0)


@pytest.mark.parametrize("bf16,center", [(False, 0), (False, 1), (False, 8), (True, 8)])
def test_production_history_keys_reach_k_and_both_action_and_world_paths(bf16, center):
    cfg = _config()
    model, engine = _model_engine(cfg)
    batch = _batch(center)
    seen: list[torch.Tensor] = []
    evidence = model.grounding.grounder.history_evidence
    assert evidence is not None

    def record(_m: torch.nn.Module, _i: tuple[object, ...], value: torch.Tensor) -> None:
        seen.append(value)

    handle = evidence.register_forward_hook(record)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, train, _ = model.encode_online(batch.online)
        local = train.observation.local_facts
        assert local.observed_history is not None and local.current_image_support is not None
        torch.testing.assert_close(
            local.observed_history.content[:, -1].to(local.target_dino_content.dtype),
            local.target_dino_content,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            local.observed_history.backward_flow[:, -1],
            -train.observation.grounding.flow.forward,
            rtol=0,
            atol=0,
        )
        assert seen and seen[0].requires_grad
        seen[0].retain_grad()
        # Only entity-side objective: no decoder/P1 gradient can fake this edge.
        loss = train.top.facts.reconstruction_error + train.top.facts.content.square().mean()
    loss.backward()
    assert seen[0].grad is not None and torch.isfinite(seen[0].grad).all()
    params = list(evidence.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    if center:
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        assert seen[0].grad.abs().sum() > 0
    else:
        assert torch.count_nonzero(seen[0]) == 0
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in params)
    handle.remove()
    engine.optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in params]
    engine.train_step(batch)
    assert engine.global_step == 1
    if center:
        assert any(not torch.equal(a, b) for a, b in zip(before, params, strict=True))
    # Dense online history is not exported into the compact W/ODE belief.
    assert not hasattr(cache.top.belief, "observed_history")


@torch.no_grad()
def test_history_is_not_rebuilt_or_mutated_in_ode_and_w_refinement():
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.configure_action_normalizer(_data().action_normalizer)
    model.eval()
    batch = _batch()
    records: list[tuple[ObservedEntityHistory, torch.Tensor]] = []
    evidence = model.grounding.grounder.history_evidence
    assert evidence is not None

    def record(_m: torch.nn.Module, inputs: tuple[object, ...], value: torch.Tensor) -> None:
        history = inputs[0]
        assert isinstance(history, ObservedEntityHistory)
        records.append((history, value.detach().clone()))

    handle = evidence.register_forward_hook(record)
    result = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(1027))
    handle.remove()
    assert len(records) == 1 and result.action.shape == (1, 24, 7)
    h, _ = records[0]
    before = h.source_time.frame_offsets.clone()
    assert batch.online.history.timing is not None
    assert torch.equal(before, batch.online.history.timing.state_offsets)
    # Fresh independent calls with exactly equal observation/noise have no hidden state.
    again = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(1027))
    torch.testing.assert_close(result.action, again.action, atol=0, rtol=0)


def test_checkpoint_identity_reload_and_deployment(tmp_path: Path):
    cfg = _config()
    data = _data()
    batch = _batch()
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
        commit="8" * 40,
    )
    assert "clearvla/vision/entity_history.py" in dict(identity.source.files)
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
    for kind in ["missing", "wrong"]:
        bad = copy.deepcopy(abi)
        obs = cast(dict[str, object], bad["observation"])
        if kind == "missing":
            del obs["entity_history"]
        else:
            cast(dict[str, object], obs["entity_history"])["identity"] = "persistent_k_index"
        with pytest.raises(ValueError, match="entity_history|entity history"):
            validate_deployment_abi(bad)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=engine.optimizer,
        schedule=engine.schedule,
        config=cfg,
        identity=identity,
        epoch=0,
        global_step=1,
        best_metric=None,
        data_state={
            "action_normalizer": data.action_normalizer.to_dict(),
            "state_normalizer": data.state_normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    m, e = _model_engine(cfg)
    load_checkpoint_exact(
        path, model=m, optimizer=e.optimizer, schedule=e.schedule, config=cfg, identity=identity
    )
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=m,
            optimizer=e.optimizer,
            schedule=e.schedule,
            config=image_config(),
            identity=identity,
        )
    deployed = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(1031))
        b = sample_action(
            deployed.model,
            batch.online,
            deployed.config,
            generator=torch.Generator().manual_seed(1031),
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_action_only_gradient_reaches_temporal_grouping_without_teacher(bf16):
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    batch = _batch()
    evidence = model.grounding.grounder.history_evidence
    assert evidence is not None
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = model.encode_online(batch.online)
        output = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = output.bottom.physical_velocity.float().square().mean()
    loss.backward()
    for p in evidence.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@torch.no_grad()
def test_altering_only_past_evidence_changes_actual_k_assignment_not_current_truth():
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    _, train, _ = model.encode_online(_batch().online)
    local = train.observation.local_facts
    history = local.observed_history
    assert history is not None
    previous = history.content.clone()
    previous[:, :-1] = previous[:, :-1].flip(-2)
    modified = replace(local, observed_history=replace(history, content=previous))
    a, _ = model.grounding.grounder(local, collect_diagnostics=False)
    b, _ = model.grounding.grounder(modified, collect_diagnostics=False)
    assert not torch.equal(a.candidate_assignment, b.candidate_assignment)
    torch.testing.assert_close(
        a.dense_chart.dino_content, b.dense_chart.dino_content, rtol=0, atol=0
    )
    torch.testing.assert_close(
        a.dense_chart.candidate_validity, b.dense_chart.candidate_validity, rtol=0, atol=0
    )
    torch.testing.assert_close(
        a.dense_chart.cell_observed, b.dense_chart.cell_observed, rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", ["current_content", "current_mask", "flow_age"])
def test_history_cannot_be_attached_to_a_different_current_observation(kind):
    cfg = _config()
    model, _ = _model_engine(cfg)
    _, train, _ = model.encode_online(_batch().online)
    local = train.observation.local_facts
    history = local.observed_history
    assert history is not None
    if kind == "current_content":
        content = history.content.clone()
        content[:, -1] += 0.2
        bad = replace(local, observed_history=replace(history, content=content))
    elif kind == "current_mask":
        observed = history.observed.clone()
        observed[:, -1] = False
        bad = replace(local, observed_history=replace(history, observed=observed))
    else:
        bad = replace(local, latest_flow_steps=history.source_time.pair_steps[:, -1] + 1)
    with pytest.raises(ValueError, match="disagree"):
        bad.validate()


def test_real_gap_changes_evidence_without_rescaling_the_spatial_correspondence():
    h = _history()
    module = CausalEntityEvidence(16, 8)
    s = _support()
    a = module(h, s)
    changed = replace(h, source_time=VisualSourceTime(torch.tensor([[-32, -16, 0]])))
    b = module(changed, s)
    assert not torch.equal(a, b)
    torch.testing.assert_close(
        pull_causal_history(h, h.content).coordinates,
        pull_causal_history(changed, changed.content).coordinates,
        rtol=0,
        atol=0,
    )
