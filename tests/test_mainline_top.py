from dataclasses import fields

import torch

from clearvla.mainline.model import LocalFactSet, ObjectFactualDock
from clearvla.mainline.model.top import ObjectIntentDynamicsTop, OnlineTopContext


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
    )


def _context(top: ObjectIntentDynamicsTop, batch: int = 2) -> OnlineTopContext:
    context, _ = top.build_online_context(
        local_facts=_local_facts(batch),
        goal_tokens=torch.randn(batch, 6, 12),
        goal_mask=torch.ones(batch, 6, dtype=torch.bool),
        state_history=torch.randn(batch, 3, 7),
        state=torch.randn(batch, 7),
        executed_history=torch.randn(batch, 3, 7),
    )
    return context


def test_online_context_has_prediction_but_no_teacher_or_future_target() -> None:
    names = {field.name for field in fields(OnlineTopContext)}
    assert names == {"facts", "intent", "coarse_action", "predicted_dynamics"}
    assert not names & {"teacher", "teacher_dynamics", "future_supports", "future_target"}


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


def test_dynamic_p2_p3_consumes_one_materialized_p1_dock() -> None:
    torch.manual_seed(11)
    top = _top()
    context = _context(top)
    batch, horizon, basis, objects, hidden = 2, 24, 2, 4, 32
    fact_by_object = torch.randn(batch, horizon, basis, objects, hidden)
    object_posterior = torch.softmax(torch.randn(batch, horizon, basis, objects + 1), dim=-1)
    chart = context.facts.object_to_chart[:, None, None].expand(-1, horizon, basis, -1, -1, -1, -1)
    coordinates = context.facts.camera_coordinates[:, None, None].expand(
        -1, horizon, basis, -1, -1, -1
    )
    dock = ObjectFactualDock(
        fact_by_object=fact_by_object,
        object_posterior=object_posterior[..., :-1],
        null_posterior=object_posterior[..., -1:],
        chart_posterior=chart,
        camera_coordinates=coordinates,
        aggregate_fact=torch.einsum("btqk,btqkh->btqh", object_posterior[..., :-1], fact_by_object),
    )
    action_query = torch.randn(batch, horizon, basis, hidden)
    captured: dict[str, torch.Tensor] = {}

    def capture_p2(_module, args, _kwargs):
        captured["p2_query"] = args[0].detach().clone()

    def capture_p3(_module, _args, kwargs):
        captured["p3_query"] = kwargs["action_query"].detach().clone()

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
            p1_fact=dock.aggregate_fact,
            action_query=action_query,
        )
    finally:
        p2_hook.remove()
        p3_hook.remove()
    compiled.validate()
    assert tuple(compiled.plan.protected_base.shape) == (batch, horizon, basis, hidden)
    torch.testing.assert_close(
        captured["p2_query"],
        action_query + dock.aggregate_fact,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured["p3_query"],
        action_query + compiled.consequence.protected_consequence,
        atol=0.0,
        rtol=0.0,
    )
