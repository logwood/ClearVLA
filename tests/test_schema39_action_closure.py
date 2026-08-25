from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from clearvla.mainline.model.effect_terminal import (
    ObjectFutureEffectTerminal,
    SelectedIntervalEvidence,
)
from clearvla.mainline.model.grounding import _observable_log_read
from clearvla.mainline.model.routing import smooth_rms_contract
from clearvla.mainline.model.types import FutureObjectDynamics, PolicyIntentDock
from clearvla.mainline.training.gradient_audit import (
    build_finite_gradient_spike_report,
)
from clearvla.mainline.v120_core.flow_dino_evidence import (
    _zero_preserving_variance_std,
)


def _intent(
    *,
    batch: int,
    objects: int,
    hidden: int,
    route: int,
    horizon: int,
    scale: float,
) -> PolicyIntentDock:
    return PolicyIntentDock(
        common_key=scale * torch.randn(batch, hidden),
        interval_residual_key=scale * torch.randn(batch, 4, hidden),
        typed_common_object_value=(
            scale * torch.randn(batch, objects, 3, route)
        ),
        typed_interval_residual_value=(
            scale * torch.randn(batch, 4, objects, 3, route)
        ),
        temporal_control=scale * torch.randn(batch, horizon, hidden),
        state_change_evidence=scale * torch.randn(batch, hidden),
    )


def _dynamics(
    *,
    batch: int = 1,
    intervals: int = 4,
    objects: int = 3,
    cameras: int = 2,
    content: int = 5,
    available: bool = True,
) -> FutureObjectDynamics:
    semantic = torch.randn(batch, intervals, objects, content)
    transport = 0.1 * torch.randn(batch, intervals, objects, cameras, 2)
    support_value = 1.0 if available else 0.0
    chart = torch.full((batch, objects, 1), support_value)
    camera = torch.full((batch, objects, cameras, 1), support_value)
    camera_weight = camera / float(max(cameras, 1))
    log_camera_weight = torch.where(
        camera_weight > 0.0,
        torch.where(
            camera_weight > 0.0,
            camera_weight,
            torch.ones_like(camera_weight),
        ).log(),
        torch.zeros_like(camera_weight),
    ).float()
    return FutureObjectDynamics(
        current_reference=torch.zeros(batch, objects, content),
        successor_content=semantic,
        semantic_delta=semantic,
        transport_mean=transport,
        transport_covariance=torch.zeros(
            batch, intervals, objects, cameras, 3, dtype=torch.float32
        ),
        chart_availability=chart,
        log_chart_availability=torch.zeros_like(chart, dtype=torch.float32),
        camera_coordinates=torch.tanh(
            torch.randn(batch, objects, cameras, 2)
        ),
        camera_chart_availability=camera,
        camera_weights=camera_weight,
        log_camera_weight=log_camera_weight,
    )


def test_schema39_s_cannot_change_spatial_selection_or_value() -> None:
    torch.manual_seed(3901)
    batch, horizon, basis, hidden, route = 2, 5, 2, 10, 4
    dynamics = _dynamics(batch=batch, content=6)
    action = torch.randn(batch, horizon, basis, hidden)
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=6,
        route_dim=route,
    ).eval()
    baseline_intent = _intent(
        batch=batch,
        objects=int(dynamics.semantic_delta.shape[2]),
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=0.0,
    )
    changed_intent = _intent(
        batch=batch,
        objects=int(dynamics.semantic_delta.shape[2]),
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=20.0,
    )
    baseline, _ = reader.spatial_select(
        action,
        dynamics,
        baseline_intent,
        collect_diagnostics=False,
    )
    changed, _ = reader.spatial_select(
        action,
        dynamics,
        changed_intent,
        collect_diagnostics=False,
    )
    for name in ("key", "value", "common_value", "residual_value", "support"):
        torch.testing.assert_close(
            getattr(changed, name),
            getattr(baseline, name),
            atol=0.0,
            rtol=0.0,
        )
    assert not torch.equal(changed.selected_s_context, baseline.selected_s_context)


def test_schema39_physical_terminal_has_four_equal_intervals_and_common_once() -> None:
    torch.manual_seed(3902)
    hidden = 4
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=hidden,
        route_dim=2,
    ).eval()
    batch, horizon, basis, intervals, types = 1, 2, 1, 4, 2
    key = torch.zeros(batch, horizon, basis, intervals, types, hidden)
    common = torch.zeros_like(key)
    common[..., 0, :] = 0.05
    common[..., 1, :] = 0.02
    residual = torch.zeros_like(key)
    selected = SelectedIntervalEvidence(
        key=key,
        value=common + residual,
        common_value=common,
        residual_value=residual,
        selected_s_context=torch.randn_like(key),
        support=torch.ones(batch, intervals, types, dtype=torch.bool),
    )
    effect, metrics = reader.temporal_terminal(
        torch.zeros(batch, horizon, basis, hidden),
        selected,
        collect_diagnostics=True,
    )
    _, shared_scale = smooth_rms_contract(
        common[:, :, :, 0].sum(dim=3),
        0.35,
    )
    torch.testing.assert_close(
        effect.effect_by_type,
        common[:, :, :, 0] * shared_scale[..., None, :],
    )
    for interval in range(4):
        torch.testing.assert_close(
            metrics[f"object_p3_interval_{interval}_mass"],
            torch.tensor(0.25),
        )
    assert float(metrics["object_p3_interval_terminal_has_null"]) == 0.0


def test_schema39_distinct_zero_sum_interval_innovations_reach_effect() -> None:
    torch.manual_seed(3903)
    hidden = 4
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=hidden,
        route_dim=2,
    ).eval()
    with torch.no_grad():
        for projection in reader.source_query:
            projection.weight.copy_(torch.eye(hidden))
    key = torch.zeros(1, 1, 1, 4, 2, hidden)
    key[0, 0, 0, :, :, 0] = torch.tensor([1.0, -1.0, 0.5, -0.5])[:, None]
    residual = torch.zeros_like(key)
    residual[0, 0, 0, :, 0, 1] = torch.tensor([0.12, -0.12, 0.06, -0.06])
    assert torch.count_nonzero(residual.sum(dim=3)) == 0
    selected = SelectedIntervalEvidence(
        key=key,
        value=residual,
        common_value=torch.zeros_like(key),
        residual_value=residual,
        selected_s_context=torch.zeros_like(key),
        support=torch.ones(1, 4, 2, dtype=torch.bool),
    )
    action = torch.zeros(1, 1, 1, hidden)
    action[..., 0] = 1.0
    effect, _ = reader.temporal_terminal(
        action,
        selected,
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(effect.semantic) > 0


def test_schema39_w_zero_blocks_s_and_empty_support_is_exact_zero() -> None:
    torch.manual_seed(3904)
    hidden = 6
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=hidden,
        route_dim=2,
    ).eval()
    shape = (1, 3, 2, 4, 2, hidden)
    key = torch.zeros(shape, requires_grad=True)
    value = torch.zeros(shape, requires_grad=True)
    action = torch.randn(1, 3, 2, hidden, requires_grad=True)
    for support_value in (True, False):
        selected = SelectedIntervalEvidence(
            key=key,
            value=value,
            common_value=torch.zeros_like(value),
            residual_value=torch.zeros_like(value),
            selected_s_context=100.0 * torch.randn_like(value),
            support=torch.full((1, 4, 2), support_value, dtype=torch.bool),
        )
        effect, _ = reader.temporal_terminal(
            action,
            selected,
            collect_diagnostics=False,
        )
        assert torch.count_nonzero(effect.effect_by_type) == 0
    effect.effect_by_type.sum().backward()
    for gradient in (action.grad, key.grad, value.grad):
        if gradient is not None:
            assert torch.isfinite(gradient).all()


def test_schema39_s_cannot_cast_an_interval_vote_when_w_key_is_zero() -> None:
    torch.manual_seed(3910)
    hidden = 6
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=hidden,
        route_dim=2,
    ).eval()
    shape = (1, 3, 2, 4, 2, hidden)
    key = torch.zeros(shape)
    residual = torch.randn(shape)
    common = torch.randn(shape)
    support = torch.ones(1, 4, 2, dtype=torch.bool)
    action = torch.randn(1, 3, 2, hidden)

    def run(s_context: torch.Tensor):
        return reader.temporal_terminal(
            action,
            SelectedIntervalEvidence(
                key=key,
                value=common + residual,
                common_value=common,
                residual_value=residual,
                selected_s_context=s_context,
                support=support,
            ),
            collect_diagnostics=True,
        )

    baseline, baseline_metrics = run(torch.zeros(shape))
    changed, changed_metrics = run(100.0 * torch.randn(shape))
    torch.testing.assert_close(
        changed.effect_by_type,
        baseline.effect_by_type,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        changed_metrics["object_p3_interval_posterior_entropy"],
        baseline_metrics["object_p3_interval_posterior_entropy"],
        atol=0.0,
        rtol=0.0,
    )


def test_schema39_log_measure_preserves_tiny_ratios_and_exact_empty_rows() -> None:
    base = torch.tensor(
        [[[math.log(1.0e-30), math.log(2.0e-30)]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    validity = torch.ones_like(base)
    conditional, _, read, availability, log_availability, _, _ = (
        _observable_log_read(base, validity, dim=-1)
    )
    torch.testing.assert_close(
        conditional,
        torch.tensor([[[1.0 / 3.0, 2.0 / 3.0]]]),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(read, conditional, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(availability, torch.ones_like(availability))
    torch.testing.assert_close(log_availability, torch.zeros_like(log_availability))
    (read * torch.tensor([[[1.0, 3.0]]])).sum().backward()
    assert base.grad is not None and torch.isfinite(base.grad).all()

    empty = _observable_log_read(base.detach(), torch.zeros_like(validity), dim=-1)
    for value in (empty[0], empty[2], empty[3]):
        assert torch.count_nonzero(value) == 0
        assert torch.isfinite(value).all()


def test_schema39_p2_camera_measure_never_rounds_through_model_dtype() -> None:
    weight = torch.tensor([[[1.0e-42, 2.0e-42]]], dtype=torch.float32)
    support = weight > 0.0
    probability = ObjectFutureEffectTerminal._conditional_camera_probability(
        weight.log(),
        support,
    )
    assert probability.dtype == torch.float32
    torch.testing.assert_close(
        probability,
        weight / weight.sum(dim=-1, keepdim=True),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    # The same evidence disappears if a consumer reconstructs support from a
    # rounded BF16 probability. P2 must consume the producer-owned FP32 log.
    assert torch.count_nonzero(weight.to(torch.bfloat16)) == 0


def test_schema39_variance_std_is_zero_preserving_with_bounded_jacobian() -> None:
    epsilon = 1.0 / 56.0
    variance = torch.tensor([0.0, 1.0e-10, 0.25], requires_grad=True)
    standard_deviation = _zero_preserving_variance_std(
        variance,
        epsilon=epsilon,
    )
    assert float(standard_deviation[0].detach()) == 0.0
    standard_deviation.sum().backward()
    assert variance.grad is not None and torch.isfinite(variance.grad).all()
    assert float(variance.grad.abs().max()) <= (1.0 / (2.0 * epsilon)) + 1.0e-4
    torch.testing.assert_close(
        standard_deviation[0] + epsilon,
        torch.tensor(epsilon),
    )


class _FlowDeltaOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation = nn.Module()
        self.observation.delta_head = nn.Linear(3, 6, bias=False)


class _MixedSpikeOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observation = nn.Module()
        self.observation.delta_head = nn.Linear(3, 6, bias=False)
        self.observation.aux_delta_head = nn.Linear(3, 6, bias=False)
        self.bottom = nn.Module()
        self.bottom.decoder = nn.Module()
        self.bottom.decoder.velocity_head = nn.Module()
        self.bottom.decoder.velocity_head.arm_abs = nn.Linear(3, 1, bias=False)


def test_schema39_spike_report_splits_flow_and_uncertainty_channels() -> None:
    model = _FlowDeltaOwner()
    gradient = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3)
    model.observation.delta_head.weight.grad = gradient.clone()
    report = build_finite_gradient_spike_report(
        model.named_parameters(),
        global_norm=float(gradient.norm()),
        audit_threshold=5.0,
        optimizer_group_name=lambda _name: "observation/decay",
    )
    assert report.flow_delta_head_channel_l2 is not None
    expected = gradient.square().sum(dim=1).sqrt()
    torch.testing.assert_close(
        torch.tensor(report.flow_delta_head_channel_l2),
        expected,
    )
    values = report.as_dict()
    assert math.isclose(
        values["flow_delta_head_flow_channels_l2"],
        float(gradient[:2].norm()),
        rel_tol=1.0e-6,
    )
    assert math.isclose(
        values["flow_delta_head_uncertainty_channels_l2"],
        float(gradient[2:].norm()),
        rel_tol=1.0e-6,
    )


def test_schema39_spike_report_does_not_attach_flow_split_to_bottom_owner() -> None:
    model = _MixedSpikeOwner()
    model.observation.delta_head.weight.grad = torch.ones_like(
        model.observation.delta_head.weight
    )
    model.observation.aux_delta_head.weight.grad = 2.0 * torch.ones_like(
        model.observation.aux_delta_head.weight
    )
    model.bottom.decoder.velocity_head.arm_abs.weight.grad = 20.0 * torch.ones_like(
        model.bottom.decoder.velocity_head.arm_abs.weight
    )
    report = build_finite_gradient_spike_report(
        model.named_parameters(),
        global_norm=50.0,
        audit_threshold=5.0,
        optimizer_group_name=lambda _name: "test/decay",
    )
    assert (
        report.max_l2.parameter_name
        == "bottom.decoder.velocity_head.arm_abs.weight"
    )
    assert report.flow_delta_head_channel_l2 is None


def test_schema39_spike_report_splits_only_the_owning_observation_head() -> None:
    model = _MixedSpikeOwner()
    owner_gradient = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3)
    model.observation.delta_head.weight.grad = owner_gradient.clone()
    model.observation.aux_delta_head.weight.grad = 0.1 * torch.ones_like(
        model.observation.aux_delta_head.weight
    )
    model.bottom.decoder.velocity_head.arm_abs.weight.grad = torch.zeros_like(
        model.bottom.decoder.velocity_head.arm_abs.weight
    )
    report = build_finite_gradient_spike_report(
        model.named_parameters(),
        global_norm=float(owner_gradient.norm()),
        audit_threshold=5.0,
        optimizer_group_name=lambda _name: "test/decay",
    )
    assert report.max_l2.parameter_name == "observation.delta_head.weight"
    assert report.flow_delta_head_channel_l2 is not None
    torch.testing.assert_close(
        torch.tensor(report.flow_delta_head_channel_l2),
        owner_gradient.square().sum(dim=1).sqrt(),
    )


def test_schema39_camera_permutation_remains_equivariant() -> None:
    torch.manual_seed(3905)
    batch, horizon, basis, hidden, route = 2, 4, 2, 8, 3
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=5,
        route_dim=route,
    ).eval()
    dynamics = _dynamics(batch=batch, cameras=3, content=5)
    intent = _intent(
        batch=batch,
        objects=int(dynamics.semantic_delta.shape[2]),
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=1.0,
    )
    action = torch.randn(batch, horizon, basis, hidden)
    baseline, _ = reader(action, dynamics, intent, collect_diagnostics=False)
    permutation = torch.tensor([2, 0, 1])
    changed = replace(
        dynamics,
        transport_mean=dynamics.transport_mean[:, :, :, permutation],
        transport_covariance=dynamics.transport_covariance[:, :, :, permutation],
        camera_coordinates=dynamics.camera_coordinates[:, :, permutation],
        camera_chart_availability=(
            dynamics.camera_chart_availability[:, :, permutation]
        ),
        camera_weights=dynamics.camera_weights[:, :, permutation],
        log_camera_weight=dynamics.log_camera_weight[:, :, permutation],
    )
    permuted, _ = reader(action, changed, intent, collect_diagnostics=False)
    torch.testing.assert_close(permuted.effect_by_type, baseline.effect_by_type)


def test_schema39_future_dynamics_rejects_rounded_or_non_psd_measures() -> None:
    dynamics = _dynamics(cameras=2)
    with pytest.raises(ValueError, match="chart availability must remain finite FP32"):
        replace(
            dynamics,
            chart_availability=dynamics.chart_availability.to(torch.bfloat16),
        ).validate()

    non_psd = dynamics.transport_covariance.clone()
    non_psd[..., 0] = 0.1
    non_psd[..., 1] = 0.2
    non_psd[..., 2] = 0.1
    with pytest.raises(ValueError, match="covariance must be positive semidefinite"):
        replace(dynamics, transport_covariance=non_psd).validate()

    non_finite = dynamics.transport_covariance.clone()
    non_finite[..., 0] = torch.nan
    with pytest.raises(ValueError, match="covariance must be finite"):
        replace(dynamics, transport_covariance=non_finite).validate()


def test_schema39_effect_terminal_cpu_bfloat16_forward_backward_is_finite() -> None:
    torch.manual_seed(3906)
    batch, horizon, basis, hidden, route = 1, 3, 2, 8, 3
    reader = ObjectFutureEffectTerminal(
        hidden=hidden,
        content_dim=5,
        route_dim=route,
    ).train()
    dynamics = _dynamics(batch=batch, cameras=2, content=5)
    intent = _intent(
        batch=batch,
        objects=int(dynamics.semantic_delta.shape[2]),
        hidden=hidden,
        route=route,
        horizon=horizon,
        scale=1.0,
    )
    action = torch.randn(batch, horizon, basis, hidden, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        effect, metrics = reader(
            action,
            dynamics,
            intent,
            collect_diagnostics=True,
        )
        loss = effect.effect_by_type.float().square().mean()
    assert torch.isfinite(effect.effect_by_type).all()
    assert all(torch.isfinite(value).all() for value in metrics.values())
    loss.backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    for parameter in reader.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
