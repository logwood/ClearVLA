"""M6f: real typed P3 horizon coordination, not executed-error feedback.

Existing fixtures replace only external image/DINO/T5 transport. Neural modules,
training losses, optimizer, sampler and checkpoint paths are production code.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, cast

import pytest
import torch
from test_mainline_future_time_grid import _batch
from test_mainline_p2_view_geometry import _config as base_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.future_time import CONTROL_ALIGNED_FUTURE_TIME
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.compiler import (
    ObjectConsequenceState,
    ObjectPolicyPlanCompiler,
    ObjectTypedEffect,
)
from clearvla.mainline.model.horizon_coordination import P3HorizonContext, TypedHorizonCoordinator
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from clearvla.mainline.model.types import PolicyIntentDock
from clearvla.mainline.p3_coordination import (
    POINTWISE_PLAN,
    TYPED_HORIZON_PLAN,
    p3_coordination_metadata,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import MainlineTrainingEngine


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, top=replace(c.top, p3_coordination_mode=TYPED_HORIZON_PLAN))


def _context() -> P3HorizonContext:
    values = {n: torch.randn(2, 24, 3, 16, requires_grad=True) for n in TypedHorizonCoordinator.SOURCE_NAMES}
    return P3HorizonContext(
        action=values["action"], current_fact=values["current_fact"],
        policy_precision=values["policy_precision"], semantic_effect=values["semantic_effect"],
        geometry_feature_effect=values["geometry_feature_effect"], task_temporal=values["task_temporal"],
        observed_change=torch.randn(2, 16, requires_grad=True), time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME,
    )


class CompilerInputs(TypedDict):
    p1_policy_residual: torch.Tensor
    consequence: ObjectConsequenceState
    intent: PolicyIntentDock
    action_query: torch.Tensor


def _compiler_inputs(c: P3HorizonContext) -> CompilerInputs:
    z = torch.zeros_like(c.current_fact)
    consequence = ObjectConsequenceState(c.current_fact,
        ObjectTypedEffect(c.semantic_effect, c.geometry_feature_effect), ObjectTypedEffect(z, z),
        c.current_fact + c.semantic_effect + c.geometry_feature_effect)
    intent = PolicyIntentDock(interval_key=torch.randn(2, 4, 16), temporal_control=c.task_temporal[:, :, 0],
        state_change_evidence=c.observed_change, target_object_address_logit=torch.zeros(2, 4, 4),
        typed_common_value=torch.randn(2, 4, 3, 4), typed_interval_residual_value=torch.randn(2, 4, 4, 3, 4),
        time_grid_mode=c.time_grid_mode)
    return CompilerInputs(p1_policy_residual=c.policy_precision, consequence=consequence,
                intent=intent, action_query=c.action)


def _p3() -> ObjectPolicyPlanCompiler:
    return ObjectPolicyPlanCompiler(hidden=16, horizon=24, basis=3, heads=2,
        future_time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME, coordination_mode=TYPED_HORIZON_PLAN)


Encoded = tuple[ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch, OnlinePolicyCache]


@pytest.fixture(scope="module")
def encoded() -> Encoded:
    torch.manual_seed(6810)
    m, e = _model_engine(_config())
    b = _batch()
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    return m, e, b, cache


def test_configuration_semantics_and_source_closure():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "p3_coordination_mode" not in cast(dict, base_config().as_dict()["top"])
    assert load_config("configs/mainline/structural_rebuild_m6f_calvin.json").top.p3_coordination_mode == TYPED_HORIZON_PLAN
    meta = p3_coordination_metadata()
    assert meta["feedback"] == "none-no-interval-mean-as-one-step-endpoint"
    assert "no-cross-decision" in cast(str, meta["state"])
    sources = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/p3_coordination.py" in sources
    assert "clearvla/mainline/model/horizon_coordination.py" in sources


@pytest.mark.parametrize("fields", [
    {"p3_coordination_mode": "guess"}, {"future_time_grid_mode": "legacy_48_v1"},
    {"p2_geometry_mode": "pooled_transport_v1"}, {"target_binding_mode": "reader_local_v1"},
])
def test_inconsistent_mode_rejected(fields):
    c = _config()
    with pytest.raises(ValueError):
        replace(c, top=replace(c.top, **fields)).validate()


@pytest.mark.parametrize("kwargs", [
    {"hidden": 15, "heads": 2}, {"hidden": 0}, {"horizon": 48}, {"basis": 0}, {"heads": 0},
])
def test_coordinator_shape_and_clock_are_not_guessed(kwargs):
    dims = dict(hidden=16, horizon=24, basis=3, heads=2)
    dims.update(kwargs)
    with pytest.raises(ValueError):
        TypedHorizonCoordinator(**dims)


def test_context_rejects_wrong_time_shape_dtype_and_empty_batch():
    c = _context()
    for bad in (replace(c, time_grid_mode="legacy_48_v1"),
                replace(c, policy_precision=c.policy_precision[:, :-1]),
                replace(c, geometry_feature_effect=c.geometry_feature_effect.long()),
                replace(c, observed_change=c.observed_change[:, None]), replace(c, action=c.action[:0])):
        with pytest.raises(ValueError):
            bad.validate(hidden=16, horizon=24, basis=3)


def test_original_protected_owners_are_returned_not_reconstructed():
    c = _context()
    args = _compiler_inputs(c)
    p3 = _p3()
    plan, _ = p3(**args, collect_diagnostics=False)
    assert plan.protected_base is args["consequence"].protected_consequence
    assert plan.protected_policy_precision is c.policy_precision
    bank = plan.as_policy_role_bank(source_depth=7)
    assert bank.protected_detail is plan.protected_base
    assert bank.protected_policy_precision is plan.protected_policy_precision
    assert bank.source_names == ("p3_temporal", "p3_state_change")
    assert not hasattr(p3, "temporal_lane")


def test_P1_precision_directly_changes_P3_even_with_fixed_P2_effects():
    torch.manual_seed(6811)
    c = _context()
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    a = p3(c)[0]
    b = p3(replace(c, policy_precision=torch.zeros_like(c.policy_precision)))[0]
    assert not torch.equal(a, b)
    gradient, = torch.autograd.grad(a[:, 0].square().sum(), (c.policy_precision,))
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_later_candidate_precision_affects_current_plan_without_future_observations():
    torch.manual_seed(6812)
    c = _context()
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    a = p3(c)[0]
    changed = c.policy_precision.detach().clone()
    changed[:, -1] += torch.randn_like(changed[:, -1])
    b = p3(replace(c, policy_precision=changed))[0]
    assert not torch.equal(a[:, 0], b[:, 0])
    g, = torch.autograd.grad(a[:, 0].square().sum(), (c.policy_precision,))
    assert g[:, -1].abs().sum() > 0


def test_semantic_and_geometry_do_not_cancel_before_source_interpretation():
    torch.manual_seed(6813)
    c = _context()
    zero = torch.zeros_like(c.action)
    opposite = replace(c, **{n: zero for n in TypedHorizonCoordinator.SOURCE_NAMES},
                       observed_change=torch.zeros_like(c.observed_change))
    opposite = replace(opposite, semantic_effect=c.semantic_effect, geometry_feature_effect=-c.semantic_effect)
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    assert torch.count_nonzero(opposite.semantic_effect + opposite.geometry_feature_effect) == 0
    assert torch.count_nonzero(p3(opposite)[0]) > 0


def test_physical_time_is_address_only_and_cannot_manufacture_zero_source_values():
    c = _context()
    zero = replace(c, **{n: torch.zeros_like(c.action) for n in TypedHorizonCoordinator.SOURCE_NAMES},
                   observed_change=torch.zeros_like(c.observed_change))
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    assert torch.equal(p3.control_midpoints, (torch.arange(24).float() + 0.5) / 24)
    assert torch.count_nonzero(p3.control_time_code) > 0
    a, b, context = p3(zero)
    assert torch.count_nonzero(a) == torch.count_nonzero(b) == torch.count_nonzero(context) == 0


@pytest.mark.parametrize("bf16", [False, True])
def test_zero_observed_change_stays_zero_without_dead_observation_derivative(bf16: bool):
    torch.manual_seed(6814)
    c = _context()
    change = torch.zeros_like(c.observed_change, requires_grad=True)
    c = replace(c, observed_change=change)
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        output = p3(c)[1]
    assert torch.count_nonzero(output) == 0
    ga, gp, gc = torch.autograd.grad(output.float().sum(), (c.action, c.policy_precision, change))
    assert torch.count_nonzero(ga) == torch.count_nonzero(gp) == 0
    assert torch.isfinite(gc).all() and gc.abs().sum() > 0


def test_basis_permutation_keeps_corresponding_query_and_never_averages_basis_axis():
    torch.manual_seed(6815)
    c = _context()
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    index = torch.tensor([2, 0, 1])
    other = replace(c, **{n: getattr(c, n)[:, :, index] for n in p3.SOURCE_NAMES})
    a, b = p3(c), p3(other)
    for first, second in zip(a, b):
        torch.testing.assert_close(first[:, :, index], second)
    assert not torch.equal(a[0][:, :, 0], a[0][:, :, 1])


def test_row_permutation_does_not_relabel_physical_time():
    torch.manual_seed(6816)
    c = _context()
    p3 = TypedHorizonCoordinator(hidden=16, horizon=24, basis=3, heads=2)
    changed = replace(c, **{n: getattr(c, n).flip(1) for n in p3.SOURCE_NAMES})
    assert not torch.equal(p3(c)[0].flip(1), p3(changed)[0])


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_do_not_change_P3_values_gradients_RNG_or_input_snapshot(bf16: bool):
    torch.manual_seed(6817)
    c = _context()
    args = _compiler_inputs(c)
    originals = {n: getattr(c, n).detach().clone() for n in TypedHorizonCoordinator.SOURCE_NAMES}
    p3 = _p3()
    results: list[tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]] = []
    for diagnostic in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            plan, _ = p3(**args, collect_diagnostics=diagnostic)
            value = plan.temporal.float().square().sum() + plan.state_change.float().square().sum()
            g = torch.autograd.grad(value, tuple(p3.parameters()))
        results.append((plan.temporal.detach(), plan.state_change.detach(), g, torch.get_rng_state().clone()))
    for x, y in zip(results[0][:2], results[1][:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(results[0][2], results[1][2]):
        assert torch.isfinite(x).all() and x.abs().sum() > 0
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert torch.equal(results[0][3], results[1][3])
    for n, saved in originals.items():
        assert torch.equal(saved, getattr(c, n))


def test_actual_call_uses_original_action_query_and_all_source_owners(encoded: Encoded, monkeypatch: pytest.MonkeyPatch):
    m, _, _, cache = encoded
    p3 = m.policy_compiler.plan_compiler
    captured: list[P3HorizonContext] = []
    queries: list[torch.Tensor] = []
    original_queries: list[torch.Tensor] = []
    original_compile = m.policy_compiler.compile
    def track_compile(context, **kwargs):
        original_queries.append(kwargs["action_query"])
        return original_compile(context, **kwargs)
    monkeypatch.setattr(m.policy_compiler, "compile", track_compile)
    h1 = p3.register_forward_pre_hook(lambda _, args, kw: queries.append(kw["action_query"]), with_kwargs=True)
    assert p3.coordinator is not None
    h2 = p3.coordinator.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    try:
        v = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4))
    finally:
        h1.remove()
        h2.remove()
    assert len(captured) == len(queries) == 1
    assert len(original_queries) == 1
    assert captured[0].action is queries[0] is original_queries[0]
    assert captured[0].policy_precision is v.compiled.plan.protected_policy_precision
    assert captured[0].current_fact is v.compiled.consequence.factual_base


def test_every_new_P3_parameter_is_owned_once_and_reaches_first_action(encoded: Encoded):
    m, e, _, cache = encoded
    p3 = m.policy_compiler.plan_compiler
    owned = [p for g in e.optimizer.param_groups for p in g["params"]]
    assert all(sum(v is p for v in owned) == 1 for p in p3.parameters())
    output = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4))
    names, params = zip(*p3.named_parameters())
    grads = torch.autograd.grad(output.bottom.physical_velocity[:, 0].square().mean(), params)
    for name, g in zip(names, grads):
        assert torch.isfinite(g).all() and g.abs().sum() > 0, name


def test_sampling_has_no_new_world_pass_or_teacher_or_task_state_updates(encoded: Encoded, monkeypatch):
    m, _, b, _ = encoded
    original = m.world.materialize
    calls: list[int] = []
    p3 = m.policy_compiler.plan_compiler
    snapshots = {n: t.clone() for n, t in p3.named_buffers()}
    def track(**kwargs):
        result = original(**kwargs)
        calls.append(1)
        return result
    monkeypatch.setattr(m.world, "materialize", track)
    monkeypatch.setattr(m.world, "materialize_supervised", lambda **_: pytest.fail("supervised W used online"))
    with torch.no_grad():
        a = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(6818))
        b2 = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(6818))
    assert calls == [1, 1, 1, 1]
    torch.testing.assert_close(a.action, b2.action, rtol=0, atol=0)
    for n, t in p3.named_buffers():
        assert torch.equal(t, snapshots[n])


@pytest.mark.parametrize("bf16", [False, True])
def test_real_tail_training_and_diagnostic_backward_are_finite(bf16: bool):
    torch.manual_seed(6819)
    m, e = _model_engine(_config())
    b = _batch(79)
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        ledger, _ = e._forward(b, training=False, collect_diagnostics=True,
                               generator=torch.Generator().manual_seed(6820))
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_exact_checkpoint_reload_and_strict_P3_ABI(tmp_path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    original = old.build_deployment_abi
    def build(*args, **kwargs):
        abi = cast(dict[str, Any], original(*args, **kwargs))
        assert abi["p3_coordination"] == p3_coordination_metadata()
        for kind in ("missing", "clock", "feedback", "query"):
            wrong = copy.deepcopy(abi)
            if kind == "missing":
                del wrong["p3_coordination"]
            else:
                wrong["p3_coordination"][kind] = "wrong"
            with pytest.raises(ValueError, match="P3 coordination"):
                validate_deployment_abi(wrong)
        return abi
    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_legacy_compiler_has_no_new_parameters_or_time_buffers():
    old = ObjectPolicyPlanCompiler(hidden=16, horizon=24, basis=3)
    assert old.coordination_mode == POINTWISE_PLAN and old.coordinator is None
    assert len(list(old.named_parameters())) == 6 and not list(old.named_buffers())


def test_real_loss_and_P3_gradients_are_diagnostic_invariant(encoded: Encoded):
    m, e, b, _ = encoded
    results = []
    for flag in (False, True):
        ledger, _ = e._forward(b, training=False, collect_diagnostics=flag,
            generator=torch.Generator().manual_seed(6830))
        grad = torch.autograd.grad(ledger.total, tuple(m.policy_compiler.plan_compiler.parameters()))
        results.append((ledger.total.detach(), grad, torch.get_rng_state().clone()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, bgrad in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, bgrad, rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])


def test_training_targets_do_not_change_reused_online_P3_context(encoded: Encoded):
    m, _, b, cache = encoded
    field, time = torch.randn(1, 24, 18), torch.tensor([0.4])
    with torch.no_grad():
        before = m.velocity(cache, noisy_action_field=field, time=time).bottom.physical_velocity
        _, state, _ = m.encode_online(b.online)
        changed = replace(b.future, action_sequence=b.future.action_sequence + 0.25)
        m.build_training_targets(state, changed)
        after = m.velocity(cache, noisy_action_field=field, time=time).bottom.physical_velocity
    torch.testing.assert_close(before, after, rtol=0, atol=0)
