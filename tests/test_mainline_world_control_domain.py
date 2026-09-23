"""M6b control domain. Synthetic image/DINO/T5; real W/P/ODE/loss/optimizer."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_mainline_instruction_reference import _batch
from test_mainline_matched_world_supervision import _config as matched_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.model.types import CandidateWorld, PhysicalActionSequenceCondition
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.world_control import (
    KNOWN_PREFIX_WORLD_CONTROL,
    CandidateControlDomain,
    world_control_metadata,
)


def _config():
    c = matched_config()
    return replace(c, top=replace(c.top, world_control_mode=KNOWN_PREFIX_WORLD_CONTROL))


def _setup():
    torch.manual_seed(3610)
    m, e = _model_engine(_config())
    m.eval()
    return m, e, _batch(16)


@pytest.fixture(scope="module")
def encoded():
    m, _, b = _setup()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
    return m, cache, state, b


@pytest.mark.parametrize("steps", [0, 7, 8, 15, 16, 24, 31, 32, 47, 48])
def test_domain_is_owned_by_actual_prefix_not_horizon_name(steps):
    d = CandidateControlDomain(steps)
    d.validate()
    assert d.admitted == tuple(steps >= e for e in (8, 16, 32, 48))
    x = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).repeat(2, 1, 1)
    selected = x[:, d.admitted]
    expected = selected.mean(1) if selected.shape[1] else torch.zeros_like(x[:, 0])
    torch.testing.assert_close(d.common(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [-1, True, 24.0])
def test_control_prefix_rejects_lossy_metadata(steps):
    with pytest.raises(ValueError):
        CandidateControlDomain(steps).validate()


def test_config_identity_and_source_closure():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    old = matched_config().as_dict()["top"]
    assert isinstance(old, dict) and "world_control_mode" not in old
    for patch in ({"world_control_mode": "guess"}, {"world_action_condition_mode": "interval_mean_v1"},
                  {"world_supervision_mode": "candidate_legacy_v1"}):
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **patch)).validate()
    assert load_config("configs/mainline/structural_rebuild_m6b_calvin.json").top.world_control_mode == c.top.world_control_mode
    root = Path(__file__).resolve().parents[1]
    assert "clearvla/mainline/world_control.py" in dict(active_source_snapshot(root).files)


def test_candidate_and_training_world_have_distinct_support_owners(encoded):
    m, cache, state, b = encoded
    assert cache.top.predicted_dynamics.control_domain == CandidateControlDomain(24)
    with torch.no_grad():
        targets, _ = m.build_training_targets(state, b.future)
    sw = targets.supervised_world
    assert sw is not None and sw.dynamics.control_domain is None
    assert sw.action_condition.interval_observed.all()
    with pytest.raises(ValueError, match="candidate control domain"):
        replace(sw, dynamics=replace(sw.dynamics, control_domain=CandidateControlDomain(24))).validate(action_dim=7)
    d = cache.top.predicted_dynamics
    for domain in (CandidateControlDomain(16), CandidateControlDomain(48)):
        with pytest.raises(ValueError, match="actual action prefix"):
            CandidateWorld(cache.top.action_condition, replace(d, control_domain=domain)).validate(action_dim=7)
    missing = replace(cache, top=replace(cache.top, candidate_world=replace(
        cache.top.candidate_world, dynamics=replace(d, control_domain=None)
    )))
    with pytest.raises(ValueError, match="control domain"):
        missing.validate(_config())


def _nonzero_field(d, *, poison=0.0):
    gen = torch.Generator().manual_seed(3611)
    fields = {}
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        value = getattr(d, name)
        x = torch.randn(value.shape, generator=gen, dtype=value.dtype) * 0.1
        if name == "transport_covariance":
            x = x.abs()
            x[..., 1] = 0
        x[:, 2:] = poison
        fields[name] = x.requires_grad_()
    return replace(d, **fields)


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), 1e30])
def test_unknown_values_keys_common_and_gradients_are_isolated(encoded, poison):
    m, cache, _, _ = encoded
    reader = m.policy_compiler.effect_reader
    query = torch.randn_like(cache.factual_dock.protected_detail).requires_grad_()
    dock = cache.top.intent.policy_dock()
    clean = _nonzero_field(cache.top.predicted_dynamics)
    poisoned = _nonzero_field(cache.top.predicted_dynamics, poison=poison)
    for diagnostics in (False, True):
        a, _ = reader(query, clean, dock, collect_diagnostics=diagnostics)
        z, _ = reader(query, poisoned, dock, collect_diagnostics=diagnostics)
        for name in ("semantic", "geometry"):
            torch.testing.assert_close(getattr(a, name), getattr(z, name), rtol=0, atol=0)
        grads = torch.autograd.grad(z.combined().square().sum(),
            [poisoned.semantic_delta, poisoned.transport_mean, poisoned.transport_covariance, query])
        assert all(torch.isfinite(g).all() for g in grads)
        assert all(torch.count_nonzero(g[:, 2:]) == 0 for g in grads[:3])
        assert sum(g[:, :2].abs().sum() for g in grads[:2]) > 0
    selected, _ = reader.spatial_select(query, poisoned, dock, collect_diagnostics=False)
    assert torch.equal(selected.support, torch.tensor([[[True, True], [True, True], [False, False], [False, False]]]))
    for name in ("key", "semantic_value", "geometry_value", "semantic_common_value", "semantic_residual_value"):
        value = getattr(selected, name)
        assert torch.isfinite(value).all() and torch.count_nonzero(value[:, :, :, 2:]) == 0
    torch.testing.assert_close(poisoned.semantic_common, clean.semantic_delta[:, :2].mean(1), rtol=0, atol=0)
    view = poisoned.policy_view()
    assert view.chart_availability is poisoned.chart_availability
    assert view.camera_chart_availability is poisoned.camera_chart_availability
    assert view.camera_coordinates is poisoned.camera_coordinates


def test_all_unknown_is_neutral_not_a_stationary_world_vote(encoded):
    m, cache, _, _ = encoded
    d = replace(_nonzero_field(cache.top.predicted_dynamics, poison=float("nan")), control_domain=CandidateControlDomain(0))
    q = torch.randn_like(cache.factual_dock.protected_detail).requires_grad_()
    selected, _ = m.policy_compiler.effect_reader.spatial_select(q, d, cache.top.intent.policy_dock(), collect_diagnostics=False)
    assert not selected.support.any()
    effect, _ = m.policy_compiler.effect_reader.temporal_terminal(q, selected, collect_diagnostics=False)
    assert torch.count_nonzero(effect.combined()) == 0
    fact = cache.factual_dock.protected_detail
    consequence, _ = m.policy_compiler.consequence(factual_base=fact, effect=effect, collect_diagnostics=False)
    torch.testing.assert_close(consequence.protected_consequence, fact, rtol=0, atol=0)
    effect.combined().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_object_permutation_retains_domain(encoded):
    _, cache, _, _ = encoded
    d = cache.top.predicted_dynamics
    p = torch.tensor([2, 0, 3, 1])
    shuffled = d.permute(p)
    assert shuffled.control_domain is d.control_domain
    torch.testing.assert_close(shuffled.semantic_common, d.semantic_common[:, p], rtol=0, atol=0)


def test_same_noise_action_is_independent_of_unknown_world_even_after_learning(monkeypatch):
    m, e, b = _setup()
    e.train_step(b, collect_diagnostics=False)
    m.eval()
    counts = []
    original = m.world.materialize
    with torch.no_grad():
        a = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3612))
    def poisoned(**kwargs):
        world, metrics = original(**kwargs)
        assert torch.count_nonzero(world.dynamics.semantic_delta[:, :2]) > 0
        d = world.dynamics
        replacements = {}
        for name in ("semantic_delta", "successor_content", "transport_mean", "transport_covariance"):
            value = getattr(d, name).clone()
            value[:, 2:] = torch.nan
            replacements[name] = value
        counts.append(1)
        return replace(world, dynamics=replace(d, **replacements)), metrics
    monkeypatch.setattr(m.world, "materialize", poisoned)
    with torch.no_grad():
        z = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3612))
    assert len(counts) == 2
    torch.testing.assert_close(a.action, z.action, rtol=0, atol=0)


def _assert_cross_extent_fp32(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare legacy 24/48-row layouts, not a same-layout causality test.

    Per-token RMS math is identical, but full-tensor vs sliced FP32 kernels
    need not be bitwise identical (PyTorch numerical-accuracy contract).
    Sixteen eps times THIS output's reference peak is a regression budget,
    not a proven global error bound or a tolerance for future dependence.
    No fixed absolute floor: replacing a tiny nonzero result with zero fails.
    """
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape and actual.numel() > 0
    assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    scale = float(expected.detach().abs().max())
    torch.testing.assert_close(
        actual, expected, rtol=0.0, atol=16 * torch.finfo(torch.float32).eps * scale,
        equal_nan=False,
    )


@pytest.mark.parametrize("bad", ["zeroed", "sign", "relative_error", "nan", "inf", "dtype"])
def test_cross_extent_numeric_budget_cannot_hide_small_signal_errors(bad: str) -> None:
    expected = torch.tensor([1e-6, -2e-6, 0.0])
    actual = expected.clone()
    if bad == "zeroed":
        actual.zero_()
    elif bad == "sign":
        actual.neg_()
    elif bad == "relative_error":
        actual.mul_(1.001)
    elif bad == "nan":
        actual[0] = torch.nan
    elif bad == "inf":
        actual[0] = torch.inf
    else:
        actual = actual.double()
    with pytest.raises(AssertionError):
        _assert_cross_extent_fp32(actual, expected)


def test_cross_extent_numeric_budget_accepts_only_small_roundoff() -> None:
    expected = torch.tensor([1e-6, -2e-6, 0.0])
    actual = expected * (1 + 4 * torch.finfo(torch.float32).eps)
    _assert_cross_extent_fp32(actual, expected)


def test_zero_reference_has_no_nonzero_absolute_tolerance_floor() -> None:
    expected = torch.zeros(3)
    _assert_cross_extent_fp32(expected.clone(), expected)
    with pytest.raises(AssertionError):
        _assert_cross_extent_fp32(torch.full_like(expected, 1e-30), expected)


def test_matched_known_controls_produce_same_near_physics_after_update():
    m, e, b = _setup()
    e.train_step(b, collect_diagnostics=False)
    m.eval()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
        a = cache.top.action_condition
        assert isinstance(a, PhysicalActionSequenceCondition)
        controls = torch.cat((a.source_action, b.future.action_sequence[:, 24:]), dim=1)
        observed = m.outlet_adapter.observed_world_condition(controls, b.online.history.action_state)
        torch.testing.assert_close(a.physical_fingerprint, observed.physical_fingerprint[:, :24], rtol=0, atol=0)
        torch.testing.assert_close(a.row_end / a.horizon, observed.row_end[:, :24] / observed.control_time_scale, rtol=0, atol=0)
        supervised, _ = m.world.materialize_supervised(belief=state.top.facts, action_condition=observed)
        # Same-extent causality remains exact; no budget can hide future reads.
        later_controls = controls.clone()
        later_controls[:, 24:, 0] += 0.7
        later = m.outlet_adapter.observed_world_condition(later_controls, b.online.history.action_state)
        changed, _ = m.world.materialize_supervised(belief=state.top.facts, action_condition=later)
    assert supervised.dynamics.control_domain is None
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        known = getattr(supervised.dynamics, name)[:, :2]
        _assert_cross_extent_fp32(getattr(cache.top.predicted_dynamics, name)[:, :2], known)
        torch.testing.assert_close(known, getattr(changed.dynamics, name)[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_real_training_keeps_far_W_supervision_and_finite_gradients(bf16):
    m, e, b = _setup()
    # W output heads are zero-initialized. Use an ordinary supervised update,
    # not manually changed weights, before testing an internal-owner VJP.
    e.train_step(b, collect_diagnostics=False)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        ledger, _ = e._forward(b, training=False, collect_diagnostics=True, generator=torch.Generator().manual_seed(3613))
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.world.dynamics.w2.parameters())


def test_exact_checkpoint_and_control_ABI(tmp_path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    original = old.build_deployment_abi
    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert abi["world_control"] == world_control_metadata()
        for kind in ("missing", "changed"):
            wrong = copy.deepcopy(abi)
            if kind == "missing":
                del wrong["world_control"]
            else:
                cast(dict[str, Any], wrong["world_control"])["online_known_prefix"] = 48
            with pytest.raises(ValueError, match="world control"):
                validate_deployment_abi(wrong)
        return abi
    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
