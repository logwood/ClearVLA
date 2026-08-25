from __future__ import annotations

from dataclasses import replace

import torch

from clearvla.mainline.model.compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
)
from clearvla.mainline.model.routing import smooth_rms_contract
from clearvla.mainline.model.types import FutureObjectDynamics, PolicyIntentDock


def _dynamics(
    *,
    semantic_delta: torch.Tensor,
    transport_mean: torch.Tensor | None = None,
    camera_available: bool = True,
) -> FutureObjectDynamics:
    batch, intervals, objects, width = semantic_delta.shape
    cameras = 1 if transport_mean is None else int(transport_mean.shape[3])
    if transport_mean is None:
        transport_mean = semantic_delta.new_zeros(
            batch, intervals, objects, cameras, 2
        )
    current = semantic_delta.new_zeros(batch, objects, width)
    camera_coordinates = semantic_delta.new_zeros(batch, objects, cameras, 2)
    camera_availability = semantic_delta.new_full(
        (batch, objects, cameras, 1),
        1.0 if camera_available else 0.0,
    )
    camera_weights = camera_availability.clone()
    return FutureObjectDynamics(
        current_reference=current,
        successor_content=current[:, None] + semantic_delta,
        semantic_delta=semantic_delta,
        transport_mean=transport_mean,
        transport_covariance=torch.zeros(
            batch, intervals, objects, cameras, 3, dtype=torch.float32
        ),
        chart_availability=semantic_delta.new_ones(batch, objects, 1),
        camera_coordinates=camera_coordinates,
        camera_chart_availability=camera_availability,
        camera_weights=camera_weights,
    )


def _intent(
    *,
    batch: int,
    objects: int,
    hidden: int,
    route: int,
    horizon: int,
    scale: float = 1.0,
) -> PolicyIntentDock:
    return PolicyIntentDock(
        common_key=scale * torch.randn(batch, hidden),
        interval_residual_key=scale * torch.randn(batch, 4, hidden),
        typed_common_object_value=scale * torch.randn(batch, objects, 3, route),
        typed_interval_residual_value=(
            scale * torch.randn(batch, 4, objects, 3, route)
        ),
        temporal_control=scale * torch.randn(batch, horizon, hidden),
        state_change_evidence=scale * torch.randn(batch, hidden),
    )


def _consequence(
    *, batch: int, horizon: int, basis: int, hidden: int
) -> ObjectConsequenceState:
    factual = torch.randn(batch, horizon, basis, hidden)
    effect = 0.1 * torch.randn(batch, horizon, basis, 2, hidden)
    interaction = 0.05 * torch.randn_like(effect)
    return ObjectConsequenceState(
        factual_base=factual,
        effect_by_type=effect,
        interaction_by_type=interaction,
        protected_consequence=factual + effect.sum(dim=-2) + interaction.sum(dim=-2),
    )


def test_schema38_w_zero_blocks_all_s_effect_values() -> None:
    torch.manual_seed(3801)
    batch, intervals, objects, width = 2, 4, 3, 6
    horizon, basis, hidden, route = 5, 2, 12, 4
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    dynamics = _dynamics(
        semantic_delta=torch.zeros(batch, intervals, objects, width)
    )
    action = torch.randn(batch, horizon, basis, hidden)
    zero_s = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=0.0,
    )
    large_s = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=50.0,
    )
    zero_read, zero_metrics = reader(
        action, dynamics, zero_s, collect_diagnostics=True
    )
    large_read, large_metrics = reader(
        action, dynamics, large_s, collect_diagnostics=True
    )
    assert torch.count_nonzero(zero_read.effect_by_type) == 0
    assert torch.count_nonzero(large_read.effect_by_type) == 0
    torch.testing.assert_close(
        zero_read.effect_by_type, large_read.effect_by_type, atol=0.0, rtol=0.0
    )
    assert float(zero_metrics["object_p2_independent_s_interval_vote"]) == 0.0
    assert float(large_metrics["object_p2_independent_s_interval_vote"]) == 0.0


def test_schema38_each_type_has_four_equal_intervals_plus_one_null() -> None:
    torch.manual_seed(3802)
    batch, intervals, objects, width = 1, 4, 2, 5
    horizon, basis, hidden, route = 3, 2, 10, 4
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    common = 0.01 * torch.randn(batch, 1, objects, width)
    dynamics = _dynamics(semantic_delta=common.expand(-1, intervals, -1, -1))
    action = torch.zeros(batch, horizon, basis, hidden)
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=10.0,
    )
    read, metrics = reader(action, dynamics, intent, collect_diagnostics=True)
    assert torch.count_nonzero(read.semantic) > 0
    torch.testing.assert_close(
        metrics["object_p2_type_interval_null_mass"],
        torch.tensor(0.2),
        atol=1.0e-7,
        rtol=0.0,
    )
    for name in ("semantic", "geometry"):
        torch.testing.assert_close(
            metrics[f"object_p2_{name}_complete_field_null_mass"],
            torch.tensor(0.2),
            atol=1.0e-7,
            rtol=0.0,
        )
    assert float(metrics["object_p2_selected_complete_field_residual_rms"]) == 0.0
    assert float(metrics["object_p2_complete_field_identity_error"]) == 0.0


def test_schema38_s_cannot_vote_when_w_key_is_zero() -> None:
    torch.manual_seed(3803)
    batch, intervals, objects, width = 1, 4, 1, 5
    horizon, basis, hidden, route = 4, 2, 10, 3
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    for key in reader.source_key:
        key.weight.data.zero_()
    dynamics = _dynamics(
        semantic_delta=0.05 * torch.randn(batch, intervals, objects, width)
    )
    action = torch.randn(batch, horizon, basis, hidden)
    intent_a = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=1.0,
    )
    intent_b = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=100.0,
    )
    read_a, metrics_a = reader(action, dynamics, intent_a, collect_diagnostics=True)
    read_b, metrics_b = reader(action, dynamics, intent_b, collect_diagnostics=True)
    torch.testing.assert_close(
        read_a.effect_by_type, read_b.effect_by_type, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        metrics_a["object_p2_type_interval_null_mass"],
        metrics_b["object_p2_type_interval_null_mass"],
        atol=0.0,
        rtol=0.0,
    )
    assert float(metrics_a["object_p2_s_condition_neutral_posterior_l1"]) == 0.0
    assert float(metrics_b["object_p2_s_condition_neutral_posterior_l1"]) == 0.0


def test_schema38_geometry_unavailable_or_uniform_preserves_semantic_read() -> None:
    torch.manual_seed(3804)
    batch, intervals, objects, width = 1, 4, 3, 6
    horizon, basis, hidden, route = 4, 2, 12, 4
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    semantic = 0.05 * torch.randn(batch, intervals, objects, width)
    unavailable = _dynamics(semantic_delta=semantic, camera_available=False)
    uniform = _dynamics(semantic_delta=semantic, camera_available=True)
    # Unequal K validity is already a semantic object prior. Completely
    # uniform camera geometry must not vote for it a second time.
    unequal_k_validity = torch.tensor([[[0.2], [1.0], [0.6]]])
    unavailable = replace(unavailable, chart_availability=unequal_k_validity)
    uniform = replace(uniform, chart_availability=unequal_k_validity)
    action = torch.randn(batch, horizon, basis, hidden)
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
    )
    unavailable_read, unavailable_metrics = reader(
        action, unavailable, intent, collect_diagnostics=True
    )
    uniform_read, uniform_metrics = reader(
        action, uniform, intent, collect_diagnostics=True
    )
    torch.testing.assert_close(
        unavailable_read.semantic,
        uniform_read.semantic,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert float(
        unavailable_metrics["object_p2_geometry_to_semantic_k_correction_rms"]
    ) == 0.0
    assert float(
        uniform_metrics["object_p2_geometry_to_semantic_k_correction_rms"]
    ) < 1.0e-7

    # The all-unavailable branch must have a finite backward even though its
    # geometry correction and geometry value are algebraically zero.
    unavailable_transport = unavailable.transport_mean.detach().requires_grad_(True)
    unavailable_with_grad = replace(
        unavailable,
        transport_mean=unavailable_transport,
    )
    unavailable_grad_read, _ = reader(
        action, unavailable_with_grad, intent, collect_diagnostics=False
    )
    unavailable_grad_read.physical_sum.square().sum().backward()
    assert unavailable_transport.grad is not None
    assert torch.isfinite(unavailable_transport.grad).all()


def test_schema38_geometry_marginal_refines_semantic_k_and_has_gradient() -> None:
    torch.manual_seed(3805)
    batch, intervals, objects, cameras, width = 1, 4, 2, 1, 4
    horizon, basis, hidden, route = 3, 1, 8, 3
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    with torch.no_grad():
        reader.coordinate_query.weight.zero_()
        reader.coordinate_query.weight[0, 0] = 1.0
        reader.coordinate_query.weight[1, 1] = 1.0
        reader.semantic_value.weight.zero_()
        reader.semantic_value.weight[0, 0] = 1.0
    semantic = torch.zeros(batch, intervals, objects, width)
    semantic[:, :, 0, 0] = 0.1
    semantic[:, :, 1, 0] = -0.1
    transport = torch.zeros(
        batch, intervals, objects, cameras, 2, requires_grad=True
    )
    available = _dynamics(semantic_delta=semantic, transport_mean=transport)
    available = replace(
        available,
        camera_coordinates=torch.tensor([[[[0.0, 0.0]], [[0.8, 0.8]]]]),
    )
    unavailable = replace(
        available,
        camera_chart_availability=torch.zeros_like(
            available.camera_chart_availability
        ),
        camera_weights=torch.zeros_like(available.camera_weights),
    )
    action = torch.zeros(batch, horizon, basis, hidden)
    action[..., 0] = 0.2
    action[..., 1] = 0.2
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=0.0,
    )
    available_read, available_metrics = reader(
        action, available, intent, collect_diagnostics=True
    )
    unavailable_read, _ = reader(
        action, unavailable, intent, collect_diagnostics=False
    )
    assert not torch.allclose(available_read.semantic, unavailable_read.semantic)
    assert float(
        available_metrics[
            "object_p2_geometry_condition_neutral_semantic_posterior_l1"
        ]
    ) > 0.0
    available_read.semantic.square().sum().backward()
    assert transport.grad is not None
    assert torch.isfinite(transport.grad).all()
    assert torch.count_nonzero(transport.grad) > 0


def test_schema38_p2_diagnostics_do_not_change_forward_value() -> None:
    torch.manual_seed(3808)
    batch, intervals, objects, cameras, width = 2, 4, 3, 2, 6
    horizon, basis, hidden, route = 5, 2, 12, 4
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    ).eval()
    dynamics = _dynamics(
        semantic_delta=0.1 * torch.randn(batch, intervals, objects, width),
        transport_mean=0.1
        * torch.randn(batch, intervals, objects, cameras, 2),
    )
    dynamics = replace(
        dynamics,
        camera_coordinates=torch.tanh(
            torch.randn(batch, objects, cameras, 2)
        ),
    )
    action = torch.randn(batch, horizon, basis, hidden)
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
    )
    with_diagnostics, metrics = reader(
        action, dynamics, intent, collect_diagnostics=True
    )
    without_diagnostics, silent = reader(
        action, dynamics, intent, collect_diagnostics=False
    )
    torch.testing.assert_close(
        with_diagnostics.effect_by_type,
        without_diagnostics.effect_by_type,
        atol=0.0,
        rtol=0.0,
    )
    assert metrics
    assert silent == {}


def test_schema38_dynamic_zero_is_static_precision_forward_and_gradient_exact() -> None:
    torch.manual_seed(3806)
    batch, horizon, basis, hidden, objects, route = 1, 6, 2, 12, 2, 4
    compiler = ObjectPolicyPlanCompiler(
        hidden=hidden,
        horizon=horizon,
        basis=basis,
    ).eval()
    consequence = _consequence(
        batch=batch, horizon=horizon, basis=basis, hidden=hidden
    )
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
    )
    fact = torch.randn(batch, horizon, basis, hidden, requires_grad=True)
    action = torch.randn(batch, horizon, basis, hidden, requires_grad=True)
    zero_dynamic = torch.zeros_like(fact)
    bank, _ = compiler(
        p1_factual_detail=fact,
        p1_policy_residual=zero_dynamic,
        consequence=consequence,
        intent=intent,
        action_query=action,
        collect_diagnostics=False,
    )
    manual_precision = compiler.precision_lane(
        torch.tanh(compiler.precision_action(action))
        * compiler.precision_innovation(fact)
    )
    manual_precision = smooth_rms_contract(manual_precision, 0.35)[0]
    torch.testing.assert_close(bank.precision, manual_precision, atol=0.0, rtol=0.0)
    bank_grad = torch.autograd.grad(
        bank.precision.square().sum(), (fact, action), retain_graph=True
    )
    manual_grad = torch.autograd.grad(
        manual_precision.square().sum(), (fact, action)
    )
    for actual, expected in zip(bank_grad, manual_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    nonzero_dynamic = torch.randn_like(fact, requires_grad=True)
    zero_fact_bank, _ = compiler(
        p1_factual_detail=torch.zeros_like(fact),
        p1_policy_residual=nonzero_dynamic,
        consequence=consequence,
        intent=intent,
        action_query=action.detach(),
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(zero_fact_bank.precision) == 0

    dynamic_bank, _ = compiler(
        p1_factual_detail=fact.detach(),
        p1_policy_residual=nonzero_dynamic,
        consequence=consequence,
        intent=intent,
        action_query=action.detach(),
        collect_diagnostics=False,
    )
    assert not torch.equal(dynamic_bank.precision, bank.precision.detach())
    dynamic_bank.precision.square().sum().backward()
    assert nonzero_dynamic.grad is not None
    assert torch.count_nonzero(nonzero_dynamic.grad) > 0
    torch.testing.assert_close(
        dynamic_bank.protected_base,
        consequence.protected_consequence,
        atol=0.0,
        rtol=0.0,
    )
    for name in (
        "effect_semantic",
        "effect_geometry",
        "temporal_semantic",
        "temporal_geometry",
        "state_change",
    ):
        torch.testing.assert_close(
            getattr(dynamic_bank, name), getattr(bank, name), atol=0.0, rtol=0.0
        )


def test_schema38_p2_and_p3_registered_parameters_have_live_gradients() -> None:
    torch.manual_seed(3807)
    batch, intervals, objects, cameras, width = 2, 4, 3, 2, 6
    horizon, basis, hidden, route = 5, 2, 12, 4
    reader = ObjectFutureEffectReader(
        hidden=hidden, content_dim=width, route_dim=route
    )
    semantic = 0.1 * torch.randn(batch, intervals, objects, width)
    transport = 0.1 * torch.randn(batch, intervals, objects, cameras, 2)
    dynamics = _dynamics(
        semantic_delta=semantic,
        transport_mean=transport,
    )
    dynamics = replace(
        dynamics,
        camera_coordinates=torch.tanh(
            torch.randn(batch, objects, cameras, 2)
        ),
        transport_covariance=torch.cat(
            (
                0.05 * torch.ones(batch, intervals, objects, cameras, 1),
                0.01 * torch.randn(batch, intervals, objects, cameras, 1),
                0.05 * torch.ones(batch, intervals, objects, cameras, 1),
            ),
            dim=-1,
        ).float(),
    )
    action = torch.randn(batch, horizon, basis, hidden)
    intent = _intent(
        batch=batch,
        objects=objects,
        hidden=hidden,
        route=route,
        horizon=horizon,
    )
    effect, _ = reader(action, dynamics, intent, collect_diagnostics=False)
    effect.effect_by_type.square().mean().backward()
    for name, parameter in reader.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name

    compiler = ObjectPolicyPlanCompiler(
        hidden=hidden,
        horizon=horizon,
        basis=basis,
    )
    consequence = _consequence(
        batch=batch, horizon=horizon, basis=basis, hidden=hidden
    )
    plan, _ = compiler(
        p1_factual_detail=torch.randn(batch, horizon, basis, hidden),
        p1_policy_residual=torch.randn(batch, horizon, basis, hidden),
        consequence=consequence,
        intent=intent,
        action_query=action,
        collect_diagnostics=False,
    )
    plan_loss = sum(
        getattr(plan, name).square().mean()
        for name in (
            "precision",
            "effect_semantic",
            "effect_geometry",
            "temporal_semantic",
            "temporal_geometry",
            "state_change",
        )
    )
    plan_loss.backward()
    for name, parameter in compiler.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
