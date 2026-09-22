"""M6d: one physical clock in the real observation/S/W/Teacher/P/bottom graph.

External visual/language transport is synthetic; no target object IDs, task
success, future observations or hidden simulator state are provided online.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_mainline_instruction_reference import _batch as legacy_batch
from test_mainline_instruction_reference import _dataset, _Tokens
from test_mainline_state_features import _model_engine
from test_mainline_world_robot_relation import _config as legacy_config
from torch.utils.data import default_collate

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.future_time import (
    CONTROL_ALIGNED_FUTURE_TIME,
    CONTROL_INTERVALS,
    LEGACY_FUTURE_TIME,
    resolve_future_time,
)
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import (
    ClearVLAMainlinePolicy,
    OnlinePolicyCache,
    OnlineTrainingState,
)
from clearvla.mainline.model.types import ObjectTopTrainingTargets, PhysicalActionSequenceCondition
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.supervision import FutureLabelSupport
from clearvla.mainline.training.engine import MainlineTrainingEngine, validate_finite_training_batch
from clearvla.mainline.training.losses import future_dynamics_terms


def _config() -> ExperimentConfig:
    c = legacy_config()
    return replace(c, dimensions=replace(c.dimensions, future_supports=6),
                   top=replace(c.top, future_time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME))


def _batch(center: int = 16) -> TrainingBatch:
    b = legacy_batch(center)
    f = b.future
    assert f.support is not None
    support = FutureLabelSupport(f.support.action[:, :24], f.support.state[:, :24], f.support.visual[:, :6])
    f = replace(f, time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME,
        action_sequence=f.action_sequence[:, :24], state_sequence=f.state_sequence[:, :24],
        dino_supports=f.dino_supports[:, :6], offsets=f.offsets[:, :6], support=support)
    return replace(b, future=f, action_target=replace(b.action_target, support=support))


Updated = tuple[ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch, OnlinePolicyCache, OnlineTrainingState, ObjectTopTrainingTargets]


@pytest.fixture(scope="module")
def updated() -> Updated:
    torch.manual_seed(3640)
    m, e = _model_engine(_config())
    b = _batch()
    e.train_step(b, collect_diagnostics=True)  # release zero-init heads by a real update
    m.eval()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online, collect_diagnostics=True)
        targets, _ = m.build_training_targets(state, b.future)
    return m, e, b, cache, state, targets


def test_nonoverlapping_actions_and_successor_observations_have_one_owner():
    grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
    assert grid.bounds == ((0, 4), (4, 8), (8, 16), (16, 24))
    rows = torch.arange(24)
    assert [rows[s].tolist() for s in grid.action_slices(24)] == [
        [0, 1, 2, 3], [4, 5, 6, 7], list(range(8, 16)), list(range(16, 24))]
    dense = grid.selection(torch.arange(1, 25)[None])
    assert dense.sum(1).eq(1).all()
    sparse = grid.selection(torch.tensor([[4, 8, 12, 16, 20, 24]]))
    assert sparse.int().tolist() == [[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]]
    assert not grid.selection(torch.tensor([[0]])).any()  # o[t] is current, not future
    legacy = resolve_future_time()
    assert [rows[s].tolist() for s in legacy.action_slices(24)] == [
        list(range(3, 8)), list(range(7, 16)), list(range(15, 24)), [23]]
    assert legacy.support_offsets == tuple(range(4, 49, 4))


@pytest.mark.parametrize("remaining,expected", [(1, [0,0,0,0]), (4, [1,0,0,0]),
    (8, [1,1,0,0]), (16, [1,1,1,0]), (24, [1,1,1,1])])
def test_tail_support_does_not_shrink_a_physical_interval(remaining, expected):
    g = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
    offsets = torch.tensor([g.support_offsets])
    assert g.interval_observed(offsets, offsets <= remaining).int().tolist() == [expected]


@pytest.mark.parametrize("bad", ["duplicate", "shifted", "missing", "floating"])
def test_sparse_future_chart_cannot_silently_be_reinterpreted(bad):
    b = _batch()
    offsets = b.future.offsets.clone()
    if bad == "duplicate":
        offsets[:, 1] = offsets[:, 0]
    elif bad == "shifted":
        offsets += 1
    elif bad == "missing":
        offsets = offsets[:, :-1]
    else:
        offsets = offsets.float()
    with pytest.raises((ValueError, TypeError)):
        replace(b.future, offsets=offsets).validate(_config())


def test_physical_coarse_interpolation_preserves_irregular_bin_centers():
    g = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
    w = g.row_interpolation()
    assert w.shape == (24, 4) and w.ge(0).all()
    torch.testing.assert_close(w.sum(1), torch.ones(24), rtol=0, atol=0)
    # An independent scalar oracle, not the same vectorized interpolation code.
    knots = [2., 6., 12., 20.]
    for i in range(24):
        x = i + .5
        expected = [0.] * 4
        if x <= 2:
            expected[0] = 1
        elif x >= 20:
            expected[3] = 1
        else:
            j = next(j for j in range(3) if knots[j] <= x <= knots[j+1])
            a = (x-knots[j])/(knots[j+1]-knots[j])
            expected[j], expected[j+1] = 1-a, a
        torch.testing.assert_close(w[i], torch.tensor(expected))
    before = torch.get_rng_state().clone()
    code = g.interval_encoding(32)
    assert torch.equal(before, torch.get_rng_state())
    assert code.shape == (4, 32) and not torch.equal(code[0], code[1])


def test_config_and_extracted_consumers_resolve_the_same_clock(updated: Updated):
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "future_time_grid_mode" not in cast(dict, legacy_config().as_dict()["top"])
    assert load_config("configs/mainline/structural_rebuild_m6d_calvin.json").top.future_time_grid_mode == CONTROL_ALIGNED_FUTURE_TIME
    for top in ({"future_time_grid_mode": "guess"}, {"world_control_mode": "legacy_extrapolation_v1"},
                {"world_supervision_mode": "candidate_legacy_v1"}):
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **top)).validate()
    with pytest.raises(ValueError):
        replace(c, dimensions=replace(c.dimensions, future_supports=12)).validate()
    m, _, b, cache, _, targets = updated
    b.validate(c)
    validate_finite_training_batch(b)
    for resolved in (m.observation.v120_config, m.execution_bottom.core_config):
        assert resolved.flow_jepa_effective_window_offsets == (4, 8, 16, 24)
        assert resolved.flow_jepa_effective_interval_boundaries == (0, 4, 8, 16, 24)
        assert resolved.flow_jepa_target_offsets == (4, 8, 12, 16, 20, 24)
    assert cache.top.predicted_dynamics.control_domain is not None
    assert targets.future_interval_valid is not None
    assert cache.top.predicted_dynamics.control_domain.interval_bounds == CONTROL_INTERVALS
    assert cache.top.predicted_dynamics.control_domain.admitted == (True, True, True, True)
    assert targets.future_interval_valid.all()
    assert targets.plan_recognition is not None
    assert targets.plan_recognition.time_grid_mode == CONTROL_ALIGNED_FUTURE_TIME
    files = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/future_time.py" in files


@pytest.mark.parametrize("index", [0, 15, 76])
def test_real_dataset_cache_and_batch_keep_tail_policy_and_six_visual_supports(index):
    old = _dataset(3)
    data = ObservedStateWindowDataset(old.episodes, [0], image_store=old.image_store,
        camera_names=old.camera_names, action_normalizer=old.action_normalizer,
        state_normalizer=old.state_normalizer, config=replace(old.config, world_horizon=24))
    tokens = _Tokens()
    cached = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, tokens))[index]
    assert cached["target_future_dinov2_tokens"].shape[0] == 6
    assert cached["future_state"].shape[0] == 24
    b = to_training_batch(default_collate([cached]),
        goal=GoalTemplate(torch.randn(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        config=_config(), device=torch.device("cpu"))
    b.validate(_config())
    validate_finite_training_batch(b)
    assert b.future.offsets.tolist() == [[4, 8, 12, 16, 20, 24]]
    assert b.future.support is not None and b.action_target.row_valid is not None
    assert b.online.instruction_reference is not None
    assert b.future.support.action[0, 0]
    assert b.online.instruction_reference.age_steps.item() == index
    if index == 76:
        assert b.action_target.row_valid[0, 0]
        assert not b.future.support.visual.any()


@pytest.mark.parametrize("support_index,owner", [(0,0), (1,1), (2,2), (3,2), (4,3), (5,3)])
def test_teacher_each_sparse_observation_changes_only_its_physical_interval(updated: Updated, support_index, owner):
    m, _, b, _, state, targets = updated
    assert b.future.support is not None
    dino = b.future.dino_supports.clone()
    gen = torch.Generator().manual_seed(3651 + support_index)
    dino[:, support_index] += torch.randn(dino[:, support_index].shape, generator=gen) * 3
    with torch.no_grad():
        changed, _ = m.training_targets.teacher(facts=state.top.facts,
            future_supports=m.observation.teacher_supports(dino), future_offsets=b.future.offsets,
            future_observed=b.future.support.visual)
    base = targets.teacher_dynamics
    assert base is not None
    assert changed.time_grid_mode == CONTROL_ALIGNED_FUTURE_TIME
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        a, z = getattr(base, name), getattr(changed, name)
        for i in range(4):
            if i != owner:
                torch.testing.assert_close(a[:, i], z[:, i], rtol=0, atol=0)
    assert not torch.equal(base.semantic_delta[:, owner], changed.semantic_delta[:, owner])


def test_recognizer_summaries_cover_every_action_and_successor_state_once(updated: Updated):
    m, _, b, _, state, _ = updated
    a = torch.arange(24).float()[None, :, None].expand_as(b.future.action_sequence)
    s = torch.arange(1,25).float()[None, :, None].expand_as(b.future.state_sequence)
    with torch.no_grad():
        t, _ = m.build_training_targets(state, replace(b.future, action_sequence=a, state_sequence=s))
    assert t.plan_recognition is not None
    expected = torch.tensor([[1.5, 5.5, 11.5, 19.5]])
    torch.testing.assert_close(t.plan_recognition.action_summary[..., 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(t.plan_recognition.state_summary[..., 0], expected+1, rtol=0, atol=0)


def test_same_full_candidate_and_observed_controls_produce_all_four_same_worlds(updated: Updated):
    m, _, b, cache, _, _ = updated
    candidate = cache.top.action_condition
    assert isinstance(candidate, PhysicalActionSequenceCondition)
    observed = m.outlet_adapter.observed_world_condition(candidate.source_action, b.online.history.action_state)
    assert observed.horizon == 24 and observed.interval_observed.all()
    with torch.no_grad():
        supervised, _ = m.world.materialize_supervised(belief=cache.top.belief, action_condition=observed)
    for name in ("semantic_delta", "successor_content", "transport_mean", "transport_covariance"):
        torch.testing.assert_close(getattr(supervised.dynamics, name), getattr(cache.top.predicted_dynamics, name), rtol=0, atol=0)


@pytest.mark.parametrize("prefix,intervals", [(4,1), (8,2), (16,3)])
def test_later_controls_cannot_change_earlier_decoded_worlds_after_optimizer_update(updated: Updated, prefix, intervals):
    m, _, b, cache, _, _ = updated
    assert isinstance(cache.top.action_condition, PhysicalActionSequenceCondition)
    a = cache.top.action_condition.source_action.detach().clone()
    a[:, prefix:, :6] += .37
    condition = m.outlet_adapter.world_condition_from_horizon_action(a, b.online.history.action_state)
    with torch.no_grad():
        world, _ = m.world.materialize(belief=cache.top.belief, action_condition=condition)
    before = cache.top.predicted_dynamics
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        torch.testing.assert_close(getattr(before, name)[:, :intervals], getattr(world.dynamics, name)[:, :intervals], rtol=0, atol=0)
    assert not torch.equal(before.semantic_delta[:, intervals:], world.dynamics.semantic_delta[:, intervals:])


@pytest.mark.parametrize("index", [2, 3])
def test_late_W2_fields_have_real_first_action_gradient(updated: Updated, index):
    m, _, _, cache, _, _ = updated
    d = cache.top.predicted_dynamics
    semantic = d.semantic_delta.detach().clone().requires_grad_()
    transport = d.transport_mean.detach().clone().requires_grad_()
    altered = replace(d, semantic_delta=semantic, transport_mean=transport)
    new_cache = replace(cache, top=replace(cache.top, candidate_world=replace(cache.top.candidate_world, dynamics=altered)))
    out = m.velocity(new_cache, noisy_action_field=torch.randn(1,24,18), time=torch.full((1,), .4))
    gs, gt = torch.autograd.grad(out.bottom.physical_velocity[:, 0].square().mean(), (semantic, transport))
    assert torch.isfinite(gs).all() and torch.isfinite(gt).all()
    assert gs[:, index].abs().sum() > 0 and gt[:, index].abs().sum() > 0


def test_action_loss_updates_actual_W2_parameters_not_only_bottom_state_input(updated: Updated):
    m, _, _, cache, _, _ = updated
    world, _ = m.world.materialize(belief=cache.top.belief, action_condition=cache.top.action_condition)
    new_cache = replace(cache, top=replace(cache.top, candidate_world=world))
    out = m.velocity(new_cache, noisy_action_field=torch.randn(1,24,18), time=torch.full((1,), .4))
    parameter = m.world.dynamics.w2.object_attention.in_proj_weight
    grad, = torch.autograd.grad(out.bottom.physical_velocity[:, 0].square().mean(), (parameter,))
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_same_shape_old_clock_is_rejected_at_consumer_boundaries(updated: Updated):
    m, _, b, cache, state, targets = updated
    assert targets.teacher_dynamics is not None
    d = cache.top.predicted_dynamics
    wrong = replace(d, time_grid_mode=LEGACY_FUTURE_TIME, control_domain=None)
    q = torch.zeros_like(cache.factual_dock.protected_detail)
    reader = m.policy_compiler.effect_reader
    with pytest.raises(ValueError, match="time grid"):
        reader.spatial_select(q, wrong, cache.top.intent.policy_dock(), collect_diagnostics=False)
    with pytest.raises(ValueError, match="time grid"):
        replace(cache, top=replace(cache.top, intent=replace(cache.top.intent, time_grid_mode=LEGACY_FUTURE_TIME))).validate(_config())
    with pytest.raises(ValueError, match="time grid"):
        future_dynamics_terms(d, replace(targets.teacher_dynamics, time_grid_mode=LEGACY_FUTURE_TIME), current_loss_support=targets.current_loss_support)
    with pytest.raises(ValueError):
        m.build_training_targets(state, replace(b.future, time_grid_mode=LEGACY_FUTURE_TIME))
    with torch.no_grad():
        selected, _ = reader.spatial_select(q, d, cache.top.intent.policy_dock(), collect_diagnostics=False)
    with pytest.raises(ValueError, match="time grid"):
        reader.temporal_terminal(q, replace(selected, time_grid_mode=LEGACY_FUTURE_TIME), collect_diagnostics=False)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("center", [16, 79])
def test_production_training_diagnostics_and_backward_remain_finite(bf16, center):
    torch.manual_seed(3663)
    m, e = _model_engine(_config())
    b = _batch(center)
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        ledger, _ = e._forward(b, training=False, collect_diagnostics=True,
                             generator=torch.Generator().manual_seed(3664))
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_deployment_keeps_two_world_builds_and_no_teacher(updated: Updated, monkeypatch):
    m, _, b, _, _, _ = updated
    original = m.world.materialize
    worlds = []
    def track(**kwargs):
        w, metrics = original(**kwargs)
        worlds.append(w)
        return w, metrics
    monkeypatch.setattr(m.world, "materialize", track)
    monkeypatch.setattr(m.world, "materialize_supervised", lambda **_: pytest.fail("training controls used online"))
    with torch.no_grad():
        result = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3667))
    assert len(worlds) == 2 and torch.isfinite(result.action).all()
    assert all(w.dynamics.time_grid_mode == CONTROL_ALIGNED_FUTURE_TIME for w in worlds)


def test_exact_checkpoint_ownership_and_clock_ABI(tmp_path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    original = old.build_deployment_abi
    def build(*args, **kwargs):
        abi = cast(dict[str, Any], original(*args, **kwargs))
        grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
        assert abi["future_time_grid"] == grid.metadata()
        assert abi["architecture_manifest"]["intervals"] == [[0,4], [4,8], [8,16], [16,24]]
        assert abi["world_supervision"]["prefix_endpoints"] == [4,8,16,24]
        for kind in ("missing", "grid", "manifest", "supervision", "control"):
            wrong = copy.deepcopy(abi)
            if kind == "missing":
                del wrong["future_time_grid"]
            elif kind == "grid":
                wrong["future_time_grid"]["support_offsets"][-1] = 48
            elif kind == "manifest":
                wrong["architecture_manifest"]["intervals"][-1] = [32,48]
            elif kind == "supervision":
                wrong["world_supervision"]["prefix_endpoints"][-1] = 48
            else:
                wrong["world_control"]["interval_bounds"] = [[4,8],[8,16],[16,32],[32,48]]
            with pytest.raises(ValueError):
                validate_deployment_abi(wrong)
        return abi
    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
