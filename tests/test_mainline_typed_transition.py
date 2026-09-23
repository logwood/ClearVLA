"""M6k actual CT and end-to-end consumers, with explicitly synthetic observations."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_bottom_evidence_values import _config as base_config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine
from torch import Tensor, nn

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.action_contract import V120SeedContext
from clearvla.mainline.model.compiler import (
    ObjectConsequenceState,
    ObjectPolicyPlanDeltaBank,
    ObjectTypedEffect,
)
from clearvla.mainline.model.transition import ControlledTransitionDynamics
from clearvla.mainline.model.typed_transition import (
    CONTEXT_SOURCES,
    DYNAMIC_SOURCES,
    TransitionPlanEvidence,
    TypedPlanTransition,
)
from clearvla.mainline.model.types import ControlledTransitionSource, ControlledTransitionState
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.transition_condition import TYPED_TRANSITION, transition_condition_metadata


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, bottom=replace(c.bottom, transition_condition_mode=TYPED_TRANSITION))


def _evidence(*, zero: bool = False) -> TransitionPlanEvidence:
    shape = (1, 24, 4, 16)

    def dynamic() -> Tensor:
        return (torch.zeros(shape) if zero else torch.randn(shape) * 0.1).requires_grad_()

    fact = torch.randn(shape)
    effect = ObjectTypedEffect(dynamic(), dynamic())
    interaction = ObjectTypedEffect(dynamic(), dynamic())
    protected = fact + effect.combined() + interaction.combined()
    consequence = ObjectConsequenceState(fact, effect, interaction, protected)
    return TransitionPlanEvidence(
        dynamic(),
        consequence,
        ObjectPolicyPlanDeltaBank(protected, dynamic(), dynamic(), dynamic()),
        V120SeedContext(torch.randn(1, 1, 16), torch.randn(1, 3, 16), torch.randn(1, 7, 16)),
    )


def _module() -> TypedPlanTransition:
    return TypedPlanTransition(
        hidden=16, heads=4, horizon=24, basis=4, rank=4, action_tokens=8
    ).eval()


def _wrapper() -> ControlledTransitionDynamics:
    return ControlledTransitionDynamics(
        hidden=16,
        heads=4,
        horizon=24,
        basis=4,
        rank=4,
        cameras=2,
        content_dim=16,
        state_dim=10,
        action_dim=7,
        action_tokens=8,
        condition_mode=TYPED_TRANSITION,
    ).eval()


def _replace_source(e: TransitionPlanEvidence, name: str, value: Tensor) -> TransitionPlanEvidence:
    c, p = e.consequence, e.plan
    if name == "action":
        return replace(e, action=value)
    if name in ("precision", "temporal", "change"):
        field = {
            "precision": "protected_policy_precision",
            "temporal": "temporal",
            "change": "state_change",
        }[name]
        return replace(e, plan=replace(p, **{field: value}))
    if name in ("semantic", "geometry"):
        c = replace(c, effect=replace(c.effect, **{name: value}))
    elif name in ("semantic_interaction", "geometry_interaction"):
        c = replace(c, interaction=replace(c.interaction, **{name.split("_")[0]: value}))
    else:
        raise ValueError(name)
    protected = c.factual_base + c.effect.combined() + c.interaction.combined()
    c = replace(c, protected_consequence=protected)
    return replace(e, consequence=c, plan=replace(p, protected_base=protected))


def test_config_roundtrip_and_source_closure() -> None:
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "transition_condition_mode" not in cast(dict, base_config().as_dict()["bottom"])
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6k_calvin.json"
        ).bottom.transition_condition_mode
        == TYPED_TRANSITION
    )
    files = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    for name in ("transition_condition.py", "model/typed_transition.py"):
        assert f"clearvla/mainline/{name}" in files


@pytest.mark.parametrize("kind", ["unknown", "value", "p3", "time", "dropout"])
def test_invalid_graph_rejected(kind: str) -> None:
    c = _config()
    if kind == "unknown":
        c = replace(c, bottom=replace(c.bottom, transition_condition_mode="guess"))
    if kind == "value":
        c = replace(c, bottom=replace(c.bottom, evidence_value_mode="normalized_legacy_v1"))
    if kind == "p3":
        c = replace(c, top=replace(c.top, p3_coordination_mode="pointwise_legacy_v1"))
    if kind == "time":
        c = replace(c, top=replace(c.top, future_time_grid_mode="legacy_48_v1"))
    if kind == "dropout":
        c = replace(c, bottom=replace(c.bottom, controlled_delta_dropout=0.2))
    with pytest.raises(ValueError):
        c.validate()


@pytest.mark.parametrize("kind", ["missing", "owner", "row", "context_batch"])
def test_invalid_dynamic_owners_rejected(kind: str) -> None:
    m, e = _wrapper(), _evidence()
    source = ControlledTransitionSource(torch.randn(1, 512, 16))
    if kind == "missing":
        with pytest.raises(ValueError, match="separate P2"):
            m(source=source, action_query=e.action, plan=e.plan, seed=e.seed)
        return
    if kind == "owner":
        e = replace(e, plan=replace(e.plan, protected_base=e.plan.protected_base.clone()))
    if kind == "row":
        e = replace(e, action=e.action[:, :23])
    if kind == "context_batch":
        e = replace(e, seed=replace(e.seed, state=e.seed.state.repeat(2, 1, 1)))
    with pytest.raises(ValueError):
        m(source=source, action_query=e.action, plan=e.plan, seed=e.seed, consequence=e.consequence)


@pytest.mark.parametrize("bf16", [False, True])
def test_zero_dynamic_input_cannot_become_state_or_type_generated_value(bf16: bool) -> None:
    m, e = _module(), _evidence(zero=True)
    source = torch.randn(1, 512, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        raw = m(source, e)
        assert torch.count_nonzero(raw["rollout_delta_pred"]) == 0
        assert torch.count_nonzero(raw["rollout_action_coeff"]) == 0
        assert torch.count_nonzero(raw["rollout_neutral_coeff"]) == 0
        raw["rollout_delta_pred"].sum().backward()
    assert (
        e.action.grad is not None
        and torch.isfinite(e.action.grad).all()
        and e.action.grad.abs().sum() > 0
    )
    for x in (m.source_identity, m.context_identity, m.action_queries, source):
        assert x.grad is not None and torch.count_nonzero(x.grad) == 0
    for projection in m.context_projections.values():
        for p in projection.parameters():
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0


def test_source_value_bank_retains_every_row_basis_and_source_without_affine_null() -> None:
    m, e = _module(), _evidence()
    keys, values = m.value_bank(e)
    assert keys.shape == values.shape == (1, 8 * 24 * 4, 16)
    assert torch.unique(m.row_address[0], dim=0).shape[0] == 24 * 4
    for i, (name, value) in enumerate(e.dynamic().items()):
        expect = m.value_projections[name](value.flatten(1, 2))
        torch.testing.assert_close(values[:, i * 96 : (i + 1) * 96], expect, rtol=0, atol=0)
        assert cast(nn.Linear, m.value_projections[name]).bias is None


def test_selector_identities_do_not_enter_values() -> None:
    m, e = _module(), _evidence()
    k, v = m.value_bank(e)
    with torch.no_grad():
        m.source_identity.add_(torch.randn_like(m.source_identity) * 2)
    newk, newv = m.value_bank(e)
    torch.testing.assert_close(v, newv, rtol=0, atol=0)
    assert not torch.equal(k, newk)


def test_equal_raw_sums_do_not_force_equal_conditioning() -> None:
    torch.manual_seed(711)
    m, e = _module(), _evidence(zero=True)
    delta = torch.randn_like(e.action)
    changed = _replace_source(_replace_source(e, "action", delta), "precision", -delta)
    assert torch.count_nonzero(changed.action + changed.plan.protected_policy_precision) == 0
    g = torch.randn(1, 512, 16)
    zero = m(g, e)["rollout_delta_pred"]
    actual = m(g, changed)["rollout_delta_pred"]
    assert torch.count_nonzero(zero) == 0 and actual.abs().sum() > 0


@pytest.mark.parametrize("name", DYNAMIC_SOURCES)
def test_each_typed_source_affects_real_CT_output(name: str) -> None:
    torch.manual_seed(712)
    m, e = _module(), _evidence(zero=True)
    source = torch.randn_like(e.action, requires_grad=True)
    e = _replace_source(e, name, source)
    out = m(torch.randn(1, 512, 16), e)["rollout_delta_pred"]
    out.square().mean().backward()
    assert out.abs().sum() > 0
    assert (
        source.grad is not None
        and torch.isfinite(source.grad).all()
        and source.grad.abs().sum() > 0
    )
    projection = cast(nn.Linear, m.value_projections[name])
    assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [1e-3, 1e-5, 1e-7])
def test_small_source_is_not_promoted_to_unit_response(scale: float) -> None:
    torch.manual_seed(713)
    m, e = _module(), _evidence()
    g = torch.randn(1, 512, 16)
    full = m(g, e)["rollout_delta_pred"].norm()
    small = e
    for name, x in e.dynamic().items():
        small = _replace_source(small, name, x * scale)
    tiny = m(g, small)["rollout_delta_pred"].norm()
    assert tiny > 0 and tiny < full * scale * 3


def test_current_context_remains_typed_and_full_length() -> None:
    m, e = _module(), _evidence()
    assert tuple(e.context()) == CONTEXT_SOURCES
    assert [x.shape[1] for x in e.context().values()] == [96, 1, 3, 7]
    out = m(torch.randn(1, 512, 16), e)
    assert out["typed_context_rows"] == 107
    out["rollout_delta_pred"].square().mean().backward()
    for p in m.context_projections.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_plan_time_permutation_is_not_hidden_by_a_mean() -> None:
    torch.manual_seed(714)
    m, e = _module(), _evidence()
    g = torch.randn(1, 512, 16)
    a = m(g, e)["rollout_delta_pred"]
    b = m(g, _replace_source(e, "temporal", e.plan.temporal.flip(1)))["rollout_delta_pred"]
    assert not torch.equal(a, b)


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_same_CT_value_gradient_and_rng(bf16: bool) -> None:
    torch.manual_seed(715)
    m, e = _wrapper(), _evidence()
    g = ControlledTransitionSource(torch.randn(1, 512, 16))
    results = []
    for collect in (False, True):
        before = torch.random.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            out, metrics = m(
                source=g,
                action_query=e.action,
                plan=e.plan,
                seed=e.seed,
                consequence=e.consequence,
                collect_diagnostics=collect,
            )
            grads = torch.autograd.grad(out.value.square().mean(), tuple(m.parameters()))
        assert torch.equal(before, torch.random.get_rng_state())
        assert all(not x.requires_grad for x in metrics.values())
        results.append((out.value, grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, b in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_intervention_executes_network_and_retains_selector() -> None:
    m, e = _wrapper(), _evidence()
    g = ControlledTransitionSource(torch.randn(1, 512, 16))
    calls = []
    assert m.typed_transition is not None
    hook = m.typed_transition.register_forward_hook(lambda *args: calls.append(True))
    m.set_eval_intervention("delta_neutral")
    try:
        out, metrics = m(
            source=g,
            action_query=e.action,
            plan=e.plan,
            seed=e.seed,
            consequence=e.consequence,
            collect_diagnostics=True,
        )
        assert calls == [True] and out.selector is g.selector
        assert (
            torch.count_nonzero(out.value) == 0
            and torch.count_nonzero(out.action_coefficients) == 0
        )
        assert metrics["controlled_transition_learned_neutral"] == 0
        assert metrics["controlled_transition_intervention_first_boundary_delta_rms"] > 0
    finally:
        hook.remove()
        m.clear_eval_intervention()


def test_production_training_and_row_zero_parameter_gradients() -> None:
    torch.manual_seed(716)
    m, engine = _model_engine(_config())
    b = _batch()
    engine.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    v = m.velocity(
        cache,
        noisy_action_field=torch.randn(1, 24, 18),
        time=torch.tensor([0.4]),
        collect_diagnostics=True,
    )
    v.bottom.physical_velocity[:, 0].square().sum().backward()
    optimizer_ids = [id(p) for gr in engine.optimizer.param_groups for p in gr["params"]]
    assert m.transition.typed_transition is not None
    assert not any("neutral" in name for name, _ in m.transition.named_parameters())
    for name, p in m.transition.named_parameters():
        assert optimizer_ids.count(id(p)) == 1
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_sampler_no_new_state_or_world_calls() -> None:
    torch.manual_seed(717)
    c = _config()
    m, e = _model_engine(c)
    b = _batch()
    e.train_step(b)
    m.eval()
    before = {n: x.clone() for n, x in m.named_buffers()}
    calls = []
    assert m.transition.typed_transition is not None
    hook = m.transition.typed_transition.register_forward_hook(lambda *args: calls.append(1))
    try:
        a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(5))
        n = len(calls)
        calls.clear()
        new = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(5))
        assert len(calls) == n and n == 12
        torch.testing.assert_close(a.action, new.action, rtol=0, atol=0)
        assert torch.isfinite(a.action).all()
        for name, x in m.named_buffers():
            torch.testing.assert_close(x, before[name], rtol=0, atol=0)
    finally:
        hook.remove()


def test_checkpoint_exact_reload_and_ABI_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_mainline_target_binding as original

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    build = original.build_deployment_abi

    def checked(*args, **kwargs):
        abi = build(*args, **kwargs)
        assert abi["transition_condition"] == transition_condition_metadata()
        for kind in ("missing", "neutral"):
            bad = copy.deepcopy(abi)
            if kind == "missing":
                del bad["transition_condition"]
            else:
                cast(dict, bad["transition_condition"])["neutral"] = "physical_no_op"
            with pytest.raises(ValueError, match="transition condition"):
                validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(original, "_config", _config)
    monkeypatch.setattr(original, "_batch", _batch)
    monkeypatch.setattr(original, "build_deployment_abi", checked)
    original.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_bottom_rejects_wrong_transition_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    m, _ = _model_engine(_config())
    m.eval()
    b = _batch()
    original = m.transition.forward

    def wrong(*args, **kwargs):
        state, metrics = original(*args, **kwargs)
        return replace(state, condition_mode="summed_legacy_v1"), metrics

    monkeypatch.setattr(m.transition, "forward", wrong)
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        with pytest.raises(ValueError, match="transition value semantics"):
            m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))


def test_no_future_or_world_object_in_CT_inputs() -> None:
    assert set(TransitionPlanEvidence.__dataclass_fields__) == {
        "action",
        "consequence",
        "plan",
        "seed",
    }
    assert set(ControlledTransitionSource.__dataclass_fields__) == {"selector"}
    assert (
        ControlledTransitionState.__dataclass_fields__["condition_mode"].default
        == "summed_legacy_v1"
    )


@pytest.mark.parametrize("bf16", [False, True])
def test_full_velocity_logging_and_gradients_match(bf16: bool) -> None:
    torch.manual_seed(718)
    m, engine = _model_engine(_config())
    b = _batch()
    engine.train_step(b)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    field = torch.randn(1, 24, 18)
    results = []
    parameters = tuple(m.transition.parameters())
    for collect in (False, True):
        before = torch.random.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            v = m.velocity(
                cache,
                noisy_action_field=field,
                time=torch.tensor([0.4]),
                collect_diagnostics=collect,
            )
            value = v.bottom.physical_velocity
            gradients = torch.autograd.grad(value.float().square().mean(), parameters)
        assert torch.equal(before, torch.random.get_rng_state())
        results.append((value, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for a, other in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a, other, rtol=0, atol=0)


def test_BF16_production_optimizer_step_has_finite_CT_updates() -> None:
    torch.manual_seed(719)
    c = _config()
    c = replace(c, runtime=replace(c.runtime, compute_dtype="bf16"))
    m, engine = _model_engine(c)
    before = {name: p.detach().clone() for name, p in m.transition.named_parameters()}
    result = engine.train_step(_batch(), collect_diagnostics=True)
    assert torch.isfinite(result.metrics["loss_total"])
    changed = []
    for name, p in m.transition.named_parameters():
        assert torch.isfinite(p).all()
        changed.append(not torch.equal(p, before[name]))
    assert all(changed)


def test_new_transition_retains_exact_protected_lanes_in_real_bottom_call() -> None:
    torch.manual_seed(720)
    m, _ = _model_engine(_config())
    m.eval()
    seen = []

    def capture(_module: nn.Module, _args: object, kwargs: dict) -> None:
        seen.append(kwargs["plan"])

    hook = m.execution_bottom.register_forward_pre_hook(capture, with_kwargs=True)
    # The production step method bypasses nn.Module.__call__ on the stage.
    # Capture the real decoder's role bank instead, including its references.
    hook.remove()

    def decoder_capture(_module: nn.Module, _args: object, kwargs: dict) -> None:
        if "policy_role_delta_bank" in kwargs:
            seen.append(kwargs["policy_role_delta_bank"])

    hook = m.execution_bottom.decoder.register_forward_pre_hook(decoder_capture, with_kwargs=True)
    try:
        with torch.no_grad():
            cache, _, _ = m.encode_online(_batch().online)
            result = m.velocity(
                cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4])
            )
        assert seen
        bank = seen[0]
        assert bank.protected_detail is result.compiled.plan.protected_base
        assert bank.protected_policy_precision is result.compiled.plan.protected_policy_precision
    finally:
        hook.remove()
