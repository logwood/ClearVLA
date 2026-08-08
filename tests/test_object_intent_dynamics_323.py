from __future__ import annotations

from dataclasses import replace

import torch

from clearvla.experiments.observed_state_lab.policy_runtime_v39 import (
    object_intent_dynamics_terms,
)
from clearvla.policy.grounded_intent_effect import GroundedFactSet
from clearvla.policy.object_intent_dynamics_323 import (
    CoarseActionIntent,
    DenseObjectGrounder,
    FutureObjectDynamics,
    FuturePlanRecognizer,
    ObjectFutureDynamicsCompiler,
    ObjectFutureEffectReader,
    ObjectFutureTeacher,
    ObjectPolicyPlanCompiler,
    ObjectW1WorkingState,
    StatelessObjectIntentOrganizer,
    ZeroPreservingObjectConsequence,
)


def _local_facts(
    *, batch: int = 2, cameras: int = 2, grid: int = 4, slots: int = 4
) -> GroundedFactSet:
    torch.manual_seed(7)
    content = torch.randn(batch, cameras, grid, grid, slots, 16)
    semantic = torch.randn(batch, cameras, grid, grid, slots, 8)
    appearance = torch.randn_like(semantic)
    geometry = torch.randn_like(semantic)
    owner = torch.softmax(
        torch.randn(batch, cameras, grid, grid, slots), dim=-1
    )
    axis = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinate = torch.stack((xx, yy), dim=-1)
    coordinate = coordinate.reshape(1, 1, grid, grid, 1, 2).expand(
        batch, cameras, -1, -1, slots, -1
    )
    coordinate = coordinate + 0.03 * torch.randn_like(coordinate)
    return GroundedFactSet(
        public_scene_base=torch.randn(batch, cameras, grid, grid, 32),
        content_slots=content,
        semantic_slots=semantic,
        appearance_slots=appearance,
        geometry_slots=geometry,
        semantic_owner_probs=owner,
        appearance_owner_probs=torch.softmax(
            torch.randn_like(owner), dim=-1
        ),
        geometry_owner_probs=torch.softmax(
            torch.randn_like(owner), dim=-1
        ),
        slot_coordinates=coordinate,
        slot_support=torch.full(owner.shape, 0.20),
        slot_validity=torch.ones(*owner.shape, 1),
    )


def _online_top():
    local = _local_facts()
    grounder = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, objects=4, iterations=2
    )
    facts, ground_metrics = grounder(local)
    organizer = StatelessObjectIntentOrganizer(
        hidden=32,
        goal_dim=20,
        state_dim=6,
        action_dim=7,
        content_dim=16,
        route_dim=8,
        horizon=24,
        heads=4,
    )
    intent, _ = organizer(
        goal_tokens=torch.randn(2, 9, 20),
        goal_mask=torch.ones(2, 9, dtype=torch.bool),
        state_history=torch.randn(2, 3, 6),
        state=torch.randn(2, 6),
        executed_history=torch.randn(2, 3, 7),
        facts=facts,
        collect_diagnostics=True,
    )
    coarse = CoarseActionIntent(hidden=32, action_dim=7, heads=4)(
        intent, future_action=torch.randn(2, 48, 7)
    )
    compiler = ObjectFutureDynamicsCompiler(
        hidden=32, content_dim=16, route_dim=8, heads=4
    )
    _, w1_state, _ = compiler.forward_w1(
        facts=facts, intent=intent, action=coarse
    )
    dynamics, _ = compiler.forward_w2(
        facts=facts,
        intent=intent,
        action=coarse,
        w1_state=w1_state,
    )
    return grounder, facts, ground_metrics, intent, coarse, dynamics


def test_global_object_grounding_reconstructs_and_g3_starts_as_parent() -> None:
    grounder, facts, metrics, *_ = _online_top()
    facts.validate()
    assert facts.content.shape == (2, 4, 16)
    assert facts.object_to_chart.shape == (2, 4, 2, 4, 4)
    assert float(metrics["object_grounding_g3_parent_l1"]) < 1e-6
    assert float(metrics["object_grounding_mass_conservation_error"]) < 1e-6
    torch.testing.assert_close(
        metrics["object_grounding_null_mass"]
        + 4.0 * metrics["object_grounding_allocation_share_mean"],
        torch.ones_like(metrics["object_grounding_null_mass"]),
        atol=2e-6,
        rtol=2e-6,
    )
    assert bool(((facts.existence >= 0.0) & (facts.existence <= 1.0)).all())
    torch.testing.assert_close(facts.validity, torch.ones_like(facts.validity))
    assert facts.reconstruction_error.isfinite()
    facts.reconstruction_error.backward()
    assert grounder.slot_seed.grad is not None
    assert torch.isfinite(grounder.slot_seed.grad).all()


def test_local_hypothesis_prior_is_not_false_null_or_object_existence() -> None:
    grounder = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, objects=4, iterations=1
    )
    slots = torch.zeros(1, 4, 32)
    candidates = torch.zeros(1, 4, 32)
    validity = torch.ones(1, 4, 1)
    priors = (
        torch.full((1, 4, 1), 0.25),
        torch.tensor([[[0.70], [0.20], [0.08], [0.02]]]),
    )
    for prior in priors:
        owner, assignment, null_assignment, read = grounder._competition(
            slots,
            candidates,
            validity,
            prior,
        )
        # K objects plus null are symmetric here.  Changing the mutually
        # exclusive local-hypothesis prior may redistribute candidate reads,
        # but it cannot manufacture null probability.
        torch.testing.assert_close(
            null_assignment.sum(dim=1),
            torch.full((1,), 0.20),
        )
        torch.testing.assert_close(
            (assignment.sum(dim=-1) + null_assignment).sum(dim=1),
            torch.ones(1),
        )
        allocation_share = assignment.sum(dim=1)
        torch.testing.assert_close(
            allocation_share,
            torch.full((1, 4), 0.20),
        )
        object_vs_null = owner[..., :4] / (
            owner[..., :4] + owner[..., 4, None]
        )
        existence = (
            read * object_vs_null.transpose(1, 2)
        ).sum(dim=-1)
        # Presence is a one-object-vs-null confidence on the object's own
        # support, not its 1/K share of the complete chart.
        torch.testing.assert_close(existence, torch.full((1, 4), 0.50))

    _, assignment, null_assignment, _ = grounder._competition(
        slots,
        candidates,
        torch.zeros_like(validity),
        priors[0],
    )
    assert torch.equal(assignment, torch.zeros_like(assignment))
    torch.testing.assert_close(null_assignment.sum(dim=1), torch.ones(1))


def test_w_uses_physical_object_validity_not_binding_confidence() -> None:
    _, facts, _, _, _, dynamics = _online_top()
    expected = facts.validity[:, None].expand_as(dynamics.validity)
    # W visibility heads start at the neutral visible state, so current
    # physical support must pass through exactly.  Binding confidence remains
    # a separate diagnostic and cannot close the W/P2 path.
    torch.testing.assert_close(dynamics.validity, expected)
    assert not torch.allclose(facts.existence, facts.validity)


def test_teacher_is_no_grad_and_supports_non_identity_transport() -> None:
    _, facts, *_ = _online_top()
    teacher = ObjectFutureTeacher(content_dim=16, key_dim=8)
    supports = torch.randn(2, 12, 2, 4, 4, 16, requires_grad=True)
    offsets = torch.tensor([4, 6, 8, 10, 12, 16, 20, 24, 32, 36, 42, 48])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        target, _ = teacher(
            facts=facts,
            future_supports=supports,
            future_offsets=offsets,
        )
    target.validate()
    assert not target.semantic_delta.requires_grad
    assert target.future_address.shape == (2, 4, 4, 2, 4, 4)
    for value in (
        target.successor_content,
        target.semantic_delta,
        target.transport_mean,
        target.transport_covariance,
        target.visibility,
        target.persistence,
        target.uncertainty,
        target.validity,
        target.future_address,
    ):
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()


def test_object_permutation_is_equivariant_through_teacher_and_p2() -> None:
    _, facts, _, intent, _, dynamics = _online_top()
    permutation = torch.tensor([2, 0, 3, 1])
    teacher = ObjectFutureTeacher(content_dim=16, key_dim=8)
    supports = torch.randn(2, 12, 2, 4, 4, 16)
    offsets = torch.tensor([4, 6, 8, 10, 12, 16, 20, 24, 32, 36, 42, 48])
    target, _ = teacher(
        facts=facts, future_supports=supports, future_offsets=offsets
    )
    permuted_target, _ = teacher(
        facts=facts.permute(permutation),
        future_supports=supports,
        future_offsets=offsets,
    )
    assert torch.allclose(
        permuted_target.semantic_delta,
        target.semantic_delta[:, :, permutation],
        atol=2e-5,
        rtol=2e-5,
    )
    permuted_dynamics = replace(
        dynamics,
        current_reference=dynamics.current_reference[:, permutation],
        successor_content=dynamics.successor_content[:, :, permutation],
        semantic_delta=dynamics.semantic_delta[:, :, permutation],
        transport_mean=dynamics.transport_mean[:, :, permutation],
        transport_covariance=dynamics.transport_covariance[:, :, permutation],
        visibility=dynamics.visibility[:, :, permutation],
        persistence=dynamics.persistence[:, :, permutation],
        uncertainty=dynamics.uncertainty[:, :, permutation],
        validity=dynamics.validity[:, :, permutation],
        future_address=dynamics.future_address[:, :, permutation],
        object_coordinates=dynamics.object_coordinates[:, permutation],
    )
    reader = ObjectFutureEffectReader(hidden=32, content_dim=16)
    action_query = torch.randn(2, 24, 2, 32)
    value, _ = reader(
        action_query, dynamics, intent, collect_diagnostics=True
    )
    permuted_value, _ = reader(
        action_query, permuted_dynamics, intent, collect_diagnostics=True
    )
    assert torch.allclose(value, permuted_value, atol=2e-5, rtol=2e-5)


def test_neutral_effect_is_algebraic_zero_through_consequence() -> None:
    _, facts, _, intent, _, _ = _online_top()
    neutral = FutureObjectDynamics.neutral(facts)
    reader = ObjectFutureEffectReader(hidden=32, content_dim=16)
    action_query = torch.randn(2, 24, 2, 32)
    effect, _ = reader(
        action_query, neutral, intent, collect_diagnostics=True
    )
    assert torch.equal(effect, torch.zeros_like(effect))
    factual = torch.randn_like(effect)
    consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=effect
    )
    assert torch.equal(consequence.interaction, torch.zeros_like(effect))
    assert torch.equal(consequence.protected_consequence, factual)


def test_p2_zero_initialized_content_has_bounded_finite_jacobian() -> None:
    value = torch.zeros(2, 3, 5, requires_grad=True)
    bounded = ObjectFutureEffectReader._bounded_unit(value)
    assert torch.equal(bounded, torch.zeros_like(bounded))
    bounded.sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert float(value.grad.abs().amax()) <= 4.0001


def test_p2_scores_and_temperatures_are_bounded_by_construction() -> None:
    _, _, _, intent, _, dynamics = _online_top()
    dynamics = replace(
        dynamics,
        semantic_delta=torch.randn_like(dynamics.semantic_delta),
        validity=torch.ones_like(dynamics.validity),
    )
    reader = ObjectFutureEffectReader(hidden=32, content_dim=16)
    _, metrics = reader(
        torch.randn(2, 24, 2, 32),
        dynamics,
        intent,
        collect_diagnostics=True,
    )
    for name in (
        "object_p2_content_score_max_abs",
        "object_p2_intent_score_max_abs",
        "object_p2_coordinate_score_max_abs",
    ):
        assert float(metrics[name]) <= 1.0001
    assert float(metrics["object_p2_combined_logit_max_abs"]) <= 12.0001
    for name in (
        "object_p2_temperature_content",
        "object_p2_temperature_intent",
        "object_p2_temperature_coordinate",
    ):
        assert 0.25 <= float(metrics[name]) <= 4.0


def test_w2_reads_both_near_intervals_without_mean_pooling() -> None:
    _, facts, _, intent, coarse, _ = _online_top()
    torch.manual_seed(1204)
    compiler = ObjectFutureDynamicsCompiler(
        hidden=32, content_dim=16, route_dim=8, heads=4
    )
    torch.nn.init.normal_(compiler.delta_head.weight, std=0.05)
    _, state, _ = compiler.forward_w1(
        facts=facts, intent=intent, action=coarse
    )
    contrast = torch.randn_like(state.near[:, :1]) * 0.25
    near_a = torch.cat(
        (state.near[:, :1] + contrast, state.near[:, 1:] - contrast), dim=1
    )
    near_b = torch.cat(
        (state.near[:, :1] + 2.0 * contrast, state.near[:, 1:] - 2.0 * contrast),
        dim=1,
    )
    torch.testing.assert_close(near_a.mean(dim=1), near_b.mean(dim=1))
    field_a, _ = compiler.forward_w2(
        facts=facts,
        intent=intent,
        action=coarse,
        w1_state=ObjectW1WorkingState(
            near=near_a,
            far_base=state.far_base,
            near_field=state.near_field,
        ),
    )
    field_b, _ = compiler.forward_w2(
        facts=facts,
        intent=intent,
        action=coarse,
        w1_state=ObjectW1WorkingState(
            near=near_b,
            far_base=state.far_base,
            near_field=state.near_field,
        ),
    )
    difference = (
        field_a.semantic_delta[:, 2:] - field_b.semantic_delta[:, 2:]
    ).abs().sum()
    assert float(difference.detach()) > 1e-6


def test_recognizer_supervises_online_intent_without_teacher_value_leak() -> None:
    _, _, _, intent, _, dynamics = _online_top()
    recognizer = FuturePlanRecognizer(
        hidden=32, action_dim=7, state_dim=6, content_dim=16, heads=4
    )
    future_action = torch.randn(2, 48, 7)
    future_state = torch.randn(2, 48, 6)
    posterior = recognizer(
        future_action=future_action,
        future_state=future_state,
        teacher=dynamics,
    )
    assert not posterior.interval_targets.requires_grad
    online_loss = torch.nn.functional.smooth_l1_loss(
        intent.interval_queries.float(), posterior.interval_targets.float()
    )
    assert online_loss.isfinite()
    assert posterior.reconstruction_loss.requires_grad


def test_object_loss_uses_existing_future_and_interval_budgets() -> None:
    _, facts, _, intent, coarse, prediction = _online_top()
    teacher_module = ObjectFutureTeacher(content_dim=16, key_dim=8)
    teacher, _ = teacher_module(
        facts=facts,
        future_supports=torch.randn(2, 12, 2, 4, 4, 16),
        future_offsets=torch.tensor(
            [4, 6, 8, 10, 12, 16, 20, 24, 32, 36, 42, 48]
        ),
    )
    output = {
        "pred_physical_velocity": torch.randn(2, 24, 7),
        "object_future_successor_prediction": prediction.successor_content,
        "object_future_successor_target": teacher.successor_content,
        "object_future_semantic_prediction": prediction.semantic_delta,
        "object_future_semantic_target": teacher.semantic_delta,
        "object_future_transport_prediction": prediction.transport_mean,
        "object_future_transport_target": teacher.transport_mean,
        "object_future_covariance_prediction": prediction.transport_covariance,
        "object_future_covariance_target": teacher.transport_covariance,
        "object_future_visibility_prediction": prediction.visibility,
        "object_future_visibility_target": teacher.visibility,
        "object_future_persistence_prediction": prediction.persistence,
        "object_future_persistence_target": teacher.persistence,
        "object_future_uncertainty_prediction": prediction.uncertainty,
        "object_future_uncertainty_target": teacher.uncertainty,
        "object_future_validity_target": teacher.validity,
        "object_fact_existence": facts.existence,
        "object_fact_validity": facts.validity,
        "object_reconstruction_loss_raw": facts.reconstruction_error,
        "object_intent_online_match_loss_raw": intent.interval_queries.square().mean(),
        "object_plan_recognition_loss_raw": intent.temporal_queries.square().mean(),
        "object_coarse_action_loss_raw": coarse.loss,
    }
    terms = object_intent_dynamics_terms(output, require_teacher=True)
    assert terms["object_future_dynamics"].requires_grad
    assert terms["object_intent_structure"].requires_grad
    total = terms["object_future_dynamics"] + terms["object_intent_structure"]
    assert total.isfinite()


def test_p3_has_five_distinct_lanes() -> None:
    _, _, _, intent, _, _ = _online_top()
    factual = torch.randn(2, 24, 2, 32)
    effect = 0.1 * torch.randn_like(factual)
    consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=effect
    )
    bank, _ = ObjectPolicyPlanCompiler(hidden=32, horizon=24, basis=2)(
        p1_fact=factual,
        consequence=consequence,
        intent=intent,
        action_query=torch.randn_like(factual),
    )
    bank.validate()
    assert bank.source_names == (
        "p3_factual",
        "p3_precision",
        "p3_effect",
        "p3_temporal",
        "p3_state_change",
    )
    assert not hasattr(bank, "execution_terminal")
    assert bank.as_policy_role_bank(source_depth=7).values.shape == (
        2,
        5,
        24,
        2,
        32,
    )


def test_p3_precision_and_temporal_lanes_consume_consequence() -> None:
    _, _, _, intent, _, _ = _online_top()
    torch.manual_seed(1205)
    factual = torch.randn(2, 24, 2, 32)
    action_query = torch.randn_like(factual)
    zero_consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=torch.zeros_like(factual)
    )
    changed_consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=0.2 * torch.randn_like(factual)
    )
    compiler = ObjectPolicyPlanCompiler(hidden=32, horizon=24, basis=2)
    zero_bank, _ = compiler(
        p1_fact=factual,
        consequence=zero_consequence,
        intent=intent,
        action_query=action_query,
    )
    changed_bank, _ = compiler(
        p1_fact=factual,
        consequence=changed_consequence,
        intent=intent,
        action_query=action_query,
    )
    precision_delta = (zero_bank.precision - changed_bank.precision).abs().sum()
    temporal_delta = (zero_bank.temporal - changed_bank.temporal).abs().sum()
    assert float(precision_delta.detach()) > 1e-6
    assert float(temporal_delta.detach()) > 1e-6


def test_observable_state_change_is_exact_zero_without_delta_or_transport() -> None:
    local = _local_facts()
    grounder = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, objects=4, iterations=2
    )
    facts, _ = grounder(local)
    facts = replace(facts, transport_prior=torch.zeros_like(facts.transport_prior))
    organizer = StatelessObjectIntentOrganizer(
        hidden=32,
        goal_dim=20,
        state_dim=6,
        action_dim=7,
        content_dim=16,
        route_dim=8,
        horizon=24,
        heads=4,
    )
    state = torch.randn(2, 6)
    intent, _ = organizer(
        goal_tokens=torch.randn(2, 9, 20),
        goal_mask=torch.ones(2, 9, dtype=torch.bool),
        state_history=state[:, None].expand(-1, 3, -1).clone(),
        state=state,
        executed_history=torch.randn(2, 3, 7),
        facts=facts,
        collect_diagnostics=True,
    )
    assert torch.equal(
        intent.state_change_evidence,
        torch.zeros_like(intent.state_change_evidence),
    )

    factual = torch.randn(2, 24, 2, 32)
    consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=0.1 * torch.randn_like(factual)
    )
    bank, _ = ObjectPolicyPlanCompiler(hidden=32, horizon=24, basis=2)(
        p1_fact=factual,
        consequence=consequence,
        intent=intent,
        action_query=torch.randn_like(factual),
    )
    assert torch.equal(bank.state_change, torch.zeros_like(bank.state_change))
    assert not hasattr(bank, "execution_terminal")


def test_state_change_replacement_cannot_modify_other_p3_lanes() -> None:
    _, _, _, intent, _, _ = _online_top()
    factual = torch.randn(2, 24, 2, 32)
    consequence, _ = ZeroPreservingObjectConsequence(32)(
        factual_base=factual, effect=0.1 * torch.randn_like(factual)
    )
    action_query = torch.randn_like(factual)
    compiler = ObjectPolicyPlanCompiler(hidden=32, horizon=24, basis=2)
    zero_intent = replace(
        intent, state_change_evidence=torch.zeros_like(intent.state_change_evidence)
    )
    changed_intent = replace(
        intent, state_change_evidence=torch.randn_like(intent.state_change_evidence)
    )
    zero_bank, _ = compiler(
        p1_fact=factual,
        consequence=consequence,
        intent=zero_intent,
        action_query=action_query,
    )
    changed_bank, _ = compiler(
        p1_fact=factual,
        consequence=consequence,
        intent=changed_intent,
        action_query=action_query,
    )
    for name in ("factual", "precision", "effect", "temporal"):
        torch.testing.assert_close(
            getattr(zero_bank, name), getattr(changed_bank, name), atol=0.0, rtol=0.0
        )
    assert not torch.equal(zero_bank.state_change, changed_bank.state_change)
