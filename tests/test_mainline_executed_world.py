"""Actual small production graph with synthetic feature transport, not skill."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from test_mainline_instruction_posterior import _config as base_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.executed_world import EXECUTED_WORLD_FEEDBACK
from clearvla.mainline.model.types import (
    ExecutedActionSequenceCondition,
    ObservedActionSequenceCondition,
)
from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.mainline.runtime.sampling import sample_action


def _config():
    c = base_config()
    return replace(c, top=replace(c.top, world_feedback_mode=EXECUTED_WORLD_FEEDBACK))


def _batch(count=1):
    return synthetic_batch(_config(), count=count, raw_side=32, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def production():
    torch.manual_seed(8901)
    b, n = _batch()
    m, e = _model_engine(_config())
    m.configure_action_normalizer(n)
    for _ in range(2):
        e.train_step(b, collect_diagnostics=False)
    m.eval()
    m.set_training_step(1200)  # emulated phase only, not 1200 updates
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
    return m, e, b, cache, state


def _feedback(production):
    prepared = production[3].world_feedback
    assert prepared is not None
    return prepared.feedback


def _read(production):
    reader = production[0].policy_compiler.plan_compiler.world_feedback_read
    assert reader is not None
    return reader


def _layout_close(left, right):
    torch.testing.assert_close(
        left,
        right,
        rtol=0,
        atol=16 * torch.finfo(torch.float32).eps * float(right.detach().abs().max()),
    )


def test_explicit_graph_identity_and_legacy_omission():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    top = base_config().as_dict()["top"]
    assert isinstance(top, dict)
    assert "world_feedback_mode" not in top
    assert (
        load_config(
            "configs/mainline/structural_rebuild_executed_world_calvin.json"
        ).top.world_feedback_mode
        == EXECUTED_WORLD_FEEDBACK
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("world_feedback_mode", "guess"),
        ("world_control_mode", "legacy_extrapolation_v1"),
        ("future_time_grid_mode", "legacy_48_v1"),
        ("entity_chart_mode", "query_lattice_v1"),
        ("world_robot_condition_mode", "implicit_g_only_v1"),
        ("target_binding_mode", "reader_local_v1"),
    ],
)
def test_incoherent_graph_is_rejected(name, value):
    c = _config()
    with pytest.raises(ValueError):
        replace(c, top=replace(c.top, **{name: value})).validate()


def test_actual_training_and_first_command_gradient(production):
    m, e, b, _, _ = production
    assert e.global_step == 2
    r = _read(production)
    cache, _, _ = m.encode_online(b.online)
    out = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
    grad = torch.autograd.grad(
        out.bottom.physical_velocity[:, 0].sum(),
        [r.semantic.weight, r.image.weight, r.query.weight, r.key.weight],
    )
    for g in grad:
        assert torch.isfinite(g).all() and g.count_nonzero() > 0
    fb = cache.world_feedback.feedback
    assert all(
        not getattr(fb, k).requires_grad
        for k in ("semantic", "image", "covariance", "posterior", "null", "past_content")
    )


def test_future_teacher_entry_absent_and_extra_cost_explicit(production):
    m, _, b, _, _ = production
    r = _read(production)
    with (
        patch.object(
            m.training_targets.teacher, "forward", side_effect=AssertionError("future entry opened")
        ),
        patch.object(m.world.dynamics, "forward_w1", wraps=m.world.dynamics.forward_w1) as w1,
        patch.object(m.world.dynamics, "forward_w2", wraps=m.world.dynamics.forward_w2) as w2,
        patch.object(m.observation, "prepare", wraps=m.observation.prepare) as obs,
        patch.object(r, "prepare", wraps=r.prepare) as prepare,
    ):
        sampled = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(sampled.action).all()
    assert (
        obs.call_count == 2
        and w1.call_count == 3
        and w2.call_count == 2
        and prepare.call_count == 1
    )


@pytest.mark.parametrize("train", [False, True])
def test_replay_does_not_update_buffers_parameters_or_clock(production, train):
    m, _, b, cache, _ = production
    try:
        m.train(train)
        before = {k: t.detach().clone() for k, t in m.state_dict().items()}
        m._replay_executed_world(b.online, cache.role_table)
        for k, t in m.state_dict().items():
            torch.testing.assert_close(before[k], t, atol=0, rtol=0)
    finally:
        m.eval()


def test_current_observation_cannot_leak_into_its_W_prediction(production):
    m, _, b, cache, _ = production
    obs = b.online.observation
    dino = obs.dino_history.clone()
    dino[:, -1] = -dino[:, -1]
    changed = replace(b.online, observation=replace(obs, dino_history=dino))
    predictions = []
    original = m.world.dynamics.predict_executed_endpoint

    def capture(**kwargs):
        result = original(**kwargs)
        predictions.append(result)
        return result

    with patch.object(m.world.dynamics, "predict_executed_endpoint", side_effect=capture):
        one = m._replay_executed_world(b.online, cache.role_table)
        two = m._replay_executed_world(changed, cache.role_table)
    for k in ("semantic", "image", "covariance"):
        torch.testing.assert_close(
            getattr(predictions[0], k), getattr(predictions[1], k), atol=0, rtol=0
        )
    assert not torch.equal(one.semantic, two.semantic)


def test_unsampled_t_minus_3_command_is_used(production):
    m, _, b, cache, _ = production
    w = b.online.history.executed_world_window
    assert w is not None
    cmd = w.commands.clone()
    cmd[:, 1, 0] += 0.2
    changed = replace(
        b.online, history=replace(b.online.history, executed_world_window=replace(w, commands=cmd))
    )
    m._admit_executed_world_source(changed, changed.history.executed_world_window)
    one = m._replay_executed_world(b.online, cache.role_table)
    two = m._replay_executed_world(changed, cache.role_table)
    torch.testing.assert_close(one.posterior, two.posterior, atol=0, rtol=0)
    assert not torch.equal(one.semantic, two.semantic)


def test_language_changes_selection_not_measured_discrepancy(production):
    m, _, b, _, _ = production
    changed = replace(b.online, goal=replace(b.online.goal, tokens=-b.online.goal.tokens))
    with torch.no_grad():
        one, _, _ = m.encode_online(b.online)
        two, _, _ = m.encode_online(changed)
    for k in ("semantic", "image", "covariance", "null", "posterior", "past_content"):
        torch.testing.assert_close(
            getattr(one.world_feedback.feedback, k),
            getattr(two.world_feedback.feedback, k),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "which", ["state", "dino_history", "raw_rgb", "commands", "time", "missing"]
)
def test_conflicting_source_rejected_before_neural_computation(production, which):
    m, _, b, _, _ = production
    w = b.online.history.executed_world_window
    assert w is not None
    if which == "missing":
        w = None
    elif which == "time":
        w = replace(w, visual_offsets=torch.tensor([[-8, -3, 0]]))
    else:
        w = replace(w, **{which: getattr(w, which).clone() + 0.1})
    online = replace(b.online, history=replace(b.online.history, executed_world_window=w))
    with patch.object(m.observation, "prepare", side_effect=AssertionError("late admission")):
        with pytest.raises(ValueError):
            m.encode_online(online)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_missing_transition_quarantines_payload_observed_nonfinite_rejected(production, bad):
    m, _, b, _, _ = production
    w = b.online.history.executed_world_window
    assert w is not None
    w = replace(
        w,
        observed=torch.zeros_like(w.observed),
        **{
            k: torch.full_like(getattr(w, k), bad)
            for k in ("state", "action_state", "commands", "dino_history", "raw_rgb")
        },
    )
    online = replace(b.online, history=replace(b.online.history, executed_world_window=w))
    with torch.no_grad():
        cache, _, _ = m.encode_online(online)
    assert cache.world_feedback.value.count_nonzero() == 0
    with pytest.raises(ValueError):
        m.encode_online(
            replace(
                online,
                history=replace(
                    online.history,
                    executed_world_window=replace(w, observed=torch.ones_like(w.observed)),
                ),
            )
        )


def test_history_dropout_invalidates_executed_comparison(production):
    m, _, b, _, _ = production
    c = _config()
    c = replace(c, top=replace(c.top, action_history_condition_dropout=1.0))
    conditioned, _, _, _ = m.conditioning.prepare(
        b.online,
        config=c,
        training=True,
        training_mask=True,
        condition_generator=torch.Generator().manual_seed(0),
    )
    assert not conditioned.history.executed_world_window.observed.any()
    assert b.online.history.executed_world_window.observed.all()
    assert conditioned.history.executed_world_window is not b.online.history.executed_world_window


def test_executed_source_is_not_future_supervision(production):
    m, _, b, _, state = production
    w = b.online.history.executed_world_window
    con = m.outlet_adapter.executed_world_condition(w.commands, w.action_state, w.observed)
    assert isinstance(con, ExecutedActionSequenceCondition) and not isinstance(
        con, ObservedActionSequenceCondition
    )
    assert con.observed[0].tolist() == [True] * 4 + [False] * 20
    future = m.outlet_adapter.observed_world_condition(
        con.source_action, w.action_state, con.observed
    )
    with pytest.raises(ValueError, match="own aligned"):
        m.world.dynamics.predict_executed_endpoint(
            facts=state.top.current_world_belief, action=future
        )


@pytest.mark.parametrize("rows", [2, 5, 24])
def test_no_partial_or_unexecuted_later_controls_in_endpoint(production, rows):
    m, _, b, _, state = production
    w = b.online.history.executed_world_window
    con = m.outlet_adapter.executed_world_condition(w.commands, w.action_state, w.observed)
    con = replace(
        con,
        observed=torch.arange(24)[None] < rows,
        source_action=torch.zeros_like(con.source_action),
        canonical_value=torch.zeros_like(con.canonical_value),
        canonical_delta=torch.zeros_like(con.canonical_delta),
    )
    with pytest.raises(ValueError):
        m.world.dynamics.predict_executed_endpoint(facts=state.top.current_world_belief, action=con)


@pytest.mark.parametrize("case", ["zero", "null", "missing"])
def test_status_null_cannot_create_P3_error_value(production, case):
    r = _read(production)
    fb = _feedback(production)
    facts = production[4].top.facts
    binding = production[3].top.intent.target_binding
    if case == "zero":
        fb = replace(
            fb,
            semantic=torch.zeros_like(fb.semantic),
            image=torch.zeros_like(fb.image),
            covariance=fb.covariance * 7,
        )
    elif case == "null":
        fb = replace(fb, posterior=torch.zeros_like(fb.posterior), null=torch.ones_like(fb.null))
    else:
        fb = replace(
            fb,
            view_observed=torch.zeros_like(fb.view_observed),
            **{
                k: torch.full_like(getattr(fb, k), float("nan"))
                for k in ("semantic", "image", "covariance", "posterior", "past_content", "null")
            },
        )
    context = torch.randn(1, 24, 2, 32, requires_grad=True)
    prepared = r.prepare(fb, facts, binding)
    value = r(prepared, context, binding)
    gradient = torch.autograd.grad(value.sum(), context)[0]
    assert value.count_nonzero() == 0 and gradient.count_nonzero() == 0


def test_past_slots_softly_matched_not_index_paired(production):
    r = _read(production)
    fb = _feedback(production)
    facts = production[4].top.facts
    binding = production[3].top.intent.target_binding
    perm = torch.tensor([2, 0, 3, 1])
    changed = replace(
        fb,
        **{
            k: getattr(fb, k)[:, perm]
            for k in (
                "semantic",
                "image",
                "covariance",
                "null",
                "posterior",
                "past_content",
                "view_observed",
            )
        },
    )
    a = r.prepare(fb, facts, binding).value
    _layout_close(a, r.prepare(changed, facts, binding).value)
    _layout_close(a, r.prepare(fb, facts.permute(perm), binding.permute(perm)).value)


def test_reader_cannot_backpropagate_into_replayed_predictor(production):
    m = production[0]
    r = _read(production)
    prepared = r.prepare(
        _feedback(production), production[4].top.facts, production[3].top.intent.target_binding
    )
    gradients = torch.autograd.grad(
        prepared.value.square().sum(),
        [m.world.dynamics.delta_head.weight, r.semantic.weight],
        allow_unused=True,
    )
    assert gradients[0] is None
    assert (
        gradients[1] is not None
        and torch.isfinite(gradients[1]).all()
        and gradients[1].count_nonzero() > 0
    )


def test_error_changes_actual_first_action_without_reopening_images(production):
    m, _, _, cache, state = production
    r = _read(production)
    fb = _feedback(production)
    new = r.prepare(
        replace(fb, semantic=torch.zeros_like(fb.semantic), image=torch.zeros_like(fb.image)),
        state.top.facts,
        cache.top.intent.target_binding,
    )
    noise, time = torch.randn(1, 24, 18), torch.tensor([0.4])
    with (
        torch.no_grad(),
        patch.object(m.observation, "prepare", side_effect=AssertionError("ODE reopened image")),
    ):
        a = m.velocity(cache, noisy_action_field=noise, time=time).bottom.physical_velocity
        b = m.velocity(
            replace(cache, world_feedback=new), noisy_action_field=noise, time=time
        ).bottom.physical_velocity
    assert not torch.equal(a[:, 0], b[:, 0])


@pytest.mark.parametrize("bf16", [False, True])
def test_existing_W_endpoint_same_layout_exact_and_later_controls_cannot_leak(production, bf16):
    m, _, b, _, state = production
    w = b.online.history.executed_world_window
    belief = state.top.current_world_belief
    assert w is not None and belief is not None
    known = m.outlet_adapter.executed_world_condition(w.commands, w.action_state, w.observed)
    action = known.source_action.clone()
    action[:, 4:] = torch.randn_like(action[:, 4:])
    full = m.outlet_adapter.observed_world_condition(
        action, w.action_state, torch.ones_like(known.observed)
    )
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        endpoint = m.world.dynamics.predict_executed_endpoint(facts=belief, action=known)
        _, prefix, _ = m.world.dynamics.forward_w1(facts=belief, action=known)
        _, extended, _ = m.world.dynamics.forward_w1(facts=belief, action=full)
        torch.testing.assert_close(prefix.common_typed, extended.common_typed, atol=0, rtol=0)
        torch.testing.assert_close(
            prefix.near_interval_innovation[:, :1],
            extended.near_interval_innovation[:, :1],
            atol=0,
            rtol=0,
        )
        same, _ = m.world.dynamics._field_with_diagnostics(
            facts=belief,
            typed_common=prefix.common_typed,
            typed_interval_innovation=prefix.near_interval_innovation,
            diagnostic_prefix=None,
            robot_relation=prefix.robot_relation,
        )
        all_four, _ = m.world.dynamics.forward_w2(facts=belief, w1_state=prefix)
    for actual, layout, full in (
        (endpoint.semantic, same.semantic_delta[:, 0], all_four.semantic_delta[:, 0]),
        (endpoint.image, same.transport_mean[:, 0], all_four.transport_mean[:, 0]),
        (endpoint.covariance, same.transport_covariance[:, 0], all_four.transport_covariance[:, 0]),
    ):
        torch.testing.assert_close(actual, layout, atol=0, rtol=0)
        if not bf16:
            _layout_close(actual, full)
        else:
            assert torch.isfinite(full).all()


def test_exact_checkpoint_owner_and_deployment_roundtrip(tmp_path, monkeypatch):
    import copy

    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert "executed_world" in abi
        for case in ("missing", "changed"):
            wrong = copy.deepcopy(abi)
            if case == "missing":
                del wrong["executed_world"]
            else:
                wrong["executed_world"] = {"old_prediction_is_physical_truth": True}
            with pytest.raises(ValueError, match="executed"):
                validate_deployment_abi(wrong)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", lambda center: _batch()[0])
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
