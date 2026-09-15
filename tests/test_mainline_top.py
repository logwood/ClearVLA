from dataclasses import fields, replace

import pytest
import torch

import clearvla.mainline.model.types as model_types
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model import FactualPrecisionDock, LocalFactSet
from clearvla.mainline.model.top import ObjectIntentDynamicsTop, OnlineTopContext
from clearvla.mainline.v120_core.profile import build_v120_visual_config


def _local_facts(batch: int = 2) -> LocalFactSet:
    cameras, side, local, content, route, hidden = 2, 2, 4, 16, 8, 32
    prefix = (batch, cameras, side, side, local)
    owner = torch.softmax(torch.randn(*prefix), dim=-1)
    return LocalFactSet(
        public_scene_base=torch.randn(batch, cameras, side, side, hidden),
        target_dino_content=torch.randn(batch, cameras, side, side, content),
        cell_observed=torch.ones(batch, cameras, side, side, 1, dtype=torch.bool),
        content_slots=torch.randn(*prefix, content),
        semantic_slots=torch.randn(*prefix, route),
        appearance_slots=torch.randn(*prefix, route),
        geometry_slots=torch.randn(*prefix, route),
        semantic_owner_probs=owner,
        appearance_owner_probs=owner,
        geometry_owner_probs=owner,
        slot_coordinates=torch.tanh(torch.randn(*prefix, 2)),
        slot_support=torch.rand(*prefix),
        slot_validity=torch.ones(*prefix, 1),
        slot_transport_prior=0.1 * torch.randn(*prefix, 2),
    )


def _top() -> ObjectIntentDynamicsTop:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=12,
            action_basis_tokens=2,
        ),
        bottom=replace(base.bottom, controller_heads=4),
    )
    config.validate()
    return ObjectIntentDynamicsTop(
        hidden=32,
        content_dim=16,
        route_dim=8,
        goal_dim=12,
        state_dim=7,
        action_dim=7,
        horizon=24,
        basis=2,
        heads=4,
        teacher_key_dim=8,
        core_config=build_v120_visual_config(config),
    )


def _context(top: ObjectIntentDynamicsTop, batch: int = 2) -> OnlineTopContext:
    context, _ = top.build_online_context(
        local_facts=_local_facts(batch),
        goal_tokens=torch.randn(batch, 6, 12),
        goal_mask=torch.ones(batch, 6, dtype=torch.bool),
        state_history=torch.randn(batch, 3, 7),
        state=torch.randn(batch, 7),
        action_state=torch.randn(batch, 7),
        executed_history=torch.randn(batch, 3, 7),
    )
    return context


def test_online_context_has_prediction_but_no_teacher_or_future_target() -> None:
    names = {field.name for field in fields(OnlineTopContext)}
    assert names == {
        "facts",
        "intent",
        "coarse_action",
        "candidate_world",
    }
    assert not names & {"teacher", "teacher_dynamics", "future_supports", "future_target"}


def test_action_condition_and_candidate_world_remain_one_cache_pair() -> None:
    top = _top()
    context = _context(top)
    deployment = context.deployment_cache()
    assert deployment.action_condition is context.action_condition
    assert deployment.predicted_dynamics is context.predicted_dynamics
    assert deployment.candidate_world is context.candidate_world
    assert context.action_condition.interval_action is context.coarse_action.action_prediction

    copied_condition = replace(
        context.action_condition,
        interval_action=context.action_condition.interval_action.clone(),
    )
    copied_world = replace(
        context.candidate_world,
        action_condition=copied_condition,
    )
    copied = replace(context, candidate_world=copied_world)
    with pytest.raises(ValueError, match="exact proposal tensor"):
        copied.validate(hidden=top.hidden, horizon=top.horizon)

    # The deployment cache has one atomic world field; callers cannot retag a
    # separately stored action while retaining the old dynamics.
    with pytest.raises(TypeError):
        replace(deployment, action_condition=copied_condition)


def test_retired_world_intent_dock_cannot_reconnect_s_or_goal_to_w() -> None:
    context = _context(_top())
    assert not hasattr(model_types, "WorldIntentDock")
    assert not hasattr(context.intent, "world_dock")


def test_teacher_replacement_cannot_change_online_context() -> None:
    torch.manual_seed(7)
    top = _top()
    context = _context(top)
    online_before = context.predicted_dynamics.semantic_delta.detach().clone()
    supports_a = torch.randn(2, 12, 2, 2, 2, 16)
    supports_b = torch.randn_like(supports_a)
    offsets = torch.tensor([4, 6, 8, 10, 12, 16, 20, 24, 32, 38, 44, 48])[None].expand(2, -1)
    action = torch.randn(2, 48, 7)
    state = torch.randn(2, 48, 7)
    target_a, _ = top.build_training_targets(
        context,
        future_supports=supports_a,
        future_offsets=offsets,
        future_action=action,
        future_state=state,
    )
    target_b, _ = top.build_training_targets(
        context,
        future_supports=supports_b,
        future_offsets=offsets,
        future_action=action,
        future_state=state,
    )
    assert torch.equal(online_before, context.predicted_dynamics.semantic_delta)
    assert target_a.teacher_dynamics is not None
    assert target_b.teacher_dynamics is not None
    assert not torch.equal(
        target_a.teacher_dynamics.successor_content,
        target_b.teacher_dynamics.successor_content,
    )


def test_coarse_training_target_matches_runtime_24_row_action_projection() -> None:
    torch.manual_seed(9)
    top = _top()
    context = _context(top)
    supports = torch.randn(2, 12, 2, 2, 2, 16)
    offsets = torch.tensor(
        [4, 6, 8, 10, 12, 16, 20, 24, 32, 38, 44, 48]
    )[None].expand(2, -1)
    future_action = torch.arange(2 * 48 * 7, dtype=torch.float32).reshape(2, 48, 7)
    future_state = torch.randn(2, 48, 7)
    captured: dict[str, model_types.CoarseActionIntentState] = {}

    def capture_supervised(_module, _args, output):
        if output.target is not None:
            captured["coarse"] = output

    handle = top.coarse_action.register_forward_hook(capture_supervised)
    try:
        top.build_training_targets(
            context,
            future_supports=supports,
            future_offsets=offsets,
            future_action=future_action,
            future_state=future_state,
        )
    finally:
        handle.remove()

    supervised = captured["coarse"]
    assert supervised.target is not None
    runtime_condition = model_types.PhysicalActionCondition.from_horizon_action(
        future_action[:, : top.horizon],
        context.action_condition.current_action,
    )
    torch.testing.assert_close(
        supervised.target,
        runtime_condition.interval_action,
        atol=0.0,
        rtol=0.0,
    )


def test_dynamic_p2_p3_consumes_one_materialized_p1_dock() -> None:
    torch.manual_seed(11)
    top = _top()
    context = _context(top)
    batch, horizon, basis, hidden = 2, 24, 2, 32
    dock = FactualPrecisionDock(
        protected_detail=torch.randn(batch, horizon, basis, hidden),
    )
    policy_query_residual = torch.randn_like(
        dock.protected_detail,
        requires_grad=True,
    )
    completed_type = getattr(model_types, "CompletedP1PolicyState", None)
    assert completed_type is not None
    p1_state = completed_type(
        factual_base=dock.protected_detail,
        policy_query_residual=policy_query_residual,
    )
    action_query = torch.randn(batch, horizon, basis, hidden)
    captured: dict[str, torch.Tensor] = {}

    def capture_p2(_module, args, _kwargs):
        captured["p2_query_live"] = args[0]
        captured["p2_query"] = args[0].detach().clone()

    def capture_p3(_module, _args, kwargs):
        captured["p3_query"] = kwargs["action_query"].detach().clone()
        assert "p1_factual_detail" not in kwargs
        captured["p3_policy_residual"] = kwargs[
            "p1_policy_residual"
        ].detach().clone()

    p2_hook = top.effect_reader.register_forward_pre_hook(
        capture_p2,
        with_kwargs=True,
    )
    p3_hook = top.plan_compiler.register_forward_pre_hook(
        capture_p3,
        with_kwargs=True,
    )
    try:
        compiled, _ = top.compile_policy(
            context.deployment_cache(),
            p1_state=p1_state,
            action_query=action_query,
        )
    finally:
        p2_hook.remove()
        p3_hook.remove()
    compiled.validate()
    assert tuple(compiled.plan.protected_base.shape) == (batch, horizon, basis, hidden)
    torch.testing.assert_close(
        captured["p2_query"],
        action_query + dock.protected_detail + policy_query_residual,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        compiled.consequence.factual_base,
        dock.protected_detail,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        compiled.plan.protected_policy_precision,
        policy_query_residual,
        atol=0.0,
        rtol=0.0,
    )
    assert compiled.plan.protected_policy_precision is policy_query_residual
    torch.testing.assert_close(
        captured["p3_policy_residual"],
        policy_query_residual,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured["p3_query"],
        action_query + compiled.consequence.protected_consequence,
        atol=0.0,
        rtol=0.0,
    )
    p2_gradient = torch.autograd.grad(
        captured["p2_query_live"].sum(),
        policy_query_residual,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        p2_gradient,
        torch.ones_like(policy_query_residual),
        atol=0.0,
        rtol=0.0,
    )
    protected_gradient = torch.autograd.grad(
        compiled.plan.protected_policy_precision.square().sum(),
        policy_query_residual,
    )[0]
    torch.testing.assert_close(
        protected_gradient,
        2.0 * policy_query_residual,
        atol=0.0,
        rtol=0.0,
    )
    zero_residual_plan, _ = top.plan_compiler(
        p1_policy_residual=torch.zeros_like(policy_query_residual),
        consequence=compiled.consequence,
        intent=context.intent.policy_dock(),
        action_query=action_query + compiled.consequence.protected_consequence,
        collect_diagnostics=False,
    )
    for name in ("temporal", "state_change"):
        torch.testing.assert_close(
            getattr(compiled.plan, name),
            getattr(zero_residual_plan, name),
            atol=0.0,
            rtol=0.0,
        )
    assert compiled.plan.source_names == ("p3_temporal", "p3_state_change")
    for name in ("factual", "precision", "effect"):
        assert not hasattr(compiled.plan, name)
