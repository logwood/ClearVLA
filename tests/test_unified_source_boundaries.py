"""Boundary regressions for the admitted graph, not learned task acceptance."""

from types import SimpleNamespace

import pytest
import torch

from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.vision.canonical_transport import rasterize_source
from clearvla.vision.observed_correspondence import observed_feature_correspondence


def test_canonical_unobserved_local_payload_is_quarantined_before_arithmetic():
    # One real local query and one producer-unobserved query. An atom-valid bit
    # in the latter does not authorize reading its placeholder payload.
    xy = torch.tensor(
        [
            [
                [
                    [
                        [
                            [[0.1, -0.2], [0.6, 0.2]],
                            [[float("nan"), float("nan")], [float("nan"), float("nan")]],
                        ]
                    ]
                ]
            ]
        ],
        requires_grad=True,
    )
    p = torch.tensor([[[[[[0.3, 0.7], [float("nan"), float("nan")]]]]]], requires_grad=True)
    prior = torch.tensor([[[[[1.0, float("nan")]]]]], requires_grad=True)
    validity = torch.tensor([[[[[[1.0], [0.0]]]]]])
    spatial = SimpleNamespace(
        coordinates=xy, probability=p, valid=torch.ones_like(p, dtype=torch.bool)
    )
    source, mass = rasterize_source(spatial, prior, validity, rows=4, columns=5)
    assert torch.isfinite(source).all() and torch.isfinite(mass).all()
    torch.testing.assert_close(mass.sum(), torch.tensor(1.0))
    ((source * torch.arange(20).reshape(1, 1, 1, 20)).sum()).backward()
    for value in (xy, p, prior):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert xy.grad is not None and p.grad is not None and prior.grad is not None
    assert not xy.grad[..., 1, :, :].count_nonzero()
    assert not p.grad[..., 1, :].count_nonzero()
    assert not prior.grad[..., 1].count_nonzero()


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_canonical_grid_requires_positive_integer_dimensions(axis):
    s = SimpleNamespace(
        coordinates=torch.zeros(1, 1, 1, 1, 1, 1, 2),
        probability=torch.ones(1, 1, 1, 1, 1, 1),
        valid=torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
    )
    args = dict(rows=2, columns=2)
    args[axis] = 0
    with pytest.raises(ValueError):
        rasterize_source(s, torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1, 1, 1), **args)


def test_empty_language_axis_is_null_and_backward_is_finite():
    binder = TaskConditionedTargetBinder(16, 4)
    task = torch.empty(2, 0, 16, requires_grad=True)
    obj = torch.randn(2, 4, 16, requires_grad=True)
    result = binder(
        task,
        obj,
        torch.ones(2, 4, dtype=torch.bool),
        history=torch.zeros(2, 16),
        task_mask=torch.empty(2, 0, dtype=torch.bool),
    )
    assert result.mass.count_nonzero() == 0
    torch.testing.assert_close(result.null_mass, torch.ones_like(result.null_mass))
    (result.mass.sum() + result.null_mass.sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in binder.parameters())


def test_view_weighting_does_not_change_under_global_rescaling_of_small_mass():
    torch.manual_seed(29)
    binder = TaskConditionedTargetBinder(16, 4)
    task = torch.randn(1, 3, 16)
    objects = torch.randn(1, 4, 2, 16)
    mass = torch.tensor([[[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]]])
    kw = dict(history=torch.zeros(1, 16), view_support=torch.ones(1, 4, 2, dtype=torch.bool))
    support = torch.ones(1, 4, dtype=torch.bool)
    a = binder(task, objects, support, view_mass=mass, **kw)
    # Still representable FP32 values; denominator-floor clipping must not
    # silently change a conditional per-view distribution.
    b = binder(task, objects, support, view_mass=mass * 1e-39, **kw)
    torch.testing.assert_close(a.mass, b.mass, atol=2e-6, rtol=2e-5)


def test_zero_target_correspondence_returns_unknown_not_empty_distribution():
    a = torch.randn(2, 3, 8, requires_grad=True)
    b = torch.empty(2, 0, 8, requires_grad=True)
    law = observed_feature_correspondence(
        a, b, torch.ones(2, 3, dtype=torch.bool), torch.empty(2, 0, dtype=torch.bool)
    )
    assert law.shape == (2, 3, 1)
    torch.testing.assert_close(law, torch.ones_like(law))
    law.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_correspondence_rejects_nonfinite_scale(scale):
    x = torch.randn(1, 2, 4)
    support = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        observed_feature_correspondence(x, x, support, support, distance_scale=scale)


def test_correspondence_rejects_nonfinite_observed_descriptors():
    x = torch.tensor([[[float("nan"), 1.0], [0.0, 1.0]]])
    support = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        observed_feature_correspondence(x, x, support, support)


def test_log_allocation_retains_underflowed_views_and_ordinary_gradients():
    from clearvla.vision.source_weights import conditional_source_weights

    logs = torch.tensor([[[-1000.0, -1002.0], [float("nan"), float("nan")]]], requires_grad=True)
    support = torch.tensor([[[True, True], [False, False]]])
    probability, log_probability, nonempty = conditional_source_weights(
        None, support, log_mass=logs
    )
    torch.testing.assert_close(
        probability[0, 0], torch.tensor([0.8807971, 0.1192029]), atol=2e-6, rtol=2e-5
    )
    assert not probability[0, 1].count_nonzero() and not nonempty[0, 1]
    (probability[..., 0].sum() + log_probability[..., 0].sum()).backward()
    assert logs.grad is not None
    assert torch.isfinite(logs.grad).all() and not logs.grad[0, 1].count_nonzero()


def test_view_allocation_unknown_is_not_fabricated_as_equal_cameras():
    binder = TaskConditionedTargetBinder(16, 4)
    objects = torch.randn(1, 4, 2, 16)
    result = binder(
        torch.randn(1, 3, 16),
        objects,
        torch.ones(1, 4, dtype=torch.bool),
        history=torch.zeros(1, 16),
        view_support=torch.ones(1, 4, 2, dtype=torch.bool),
        view_mass=torch.zeros(1, 4, 2),
    )
    assert not result.mass.count_nonzero()
    torch.testing.assert_close(result.null_mass, torch.ones_like(result.null_mass))


def test_unsupported_descriptor_nan_has_finite_zero_source_gradient():
    a = torch.full((1, 3, 4), float("nan"), requires_grad=True)
    b = torch.full((1, 3, 4), float("nan"), requires_grad=True)
    support = torch.zeros(1, 3, dtype=torch.bool)
    law = observed_feature_correspondence(a, b, support, support)
    assert law[..., :-1].count_nonzero() == 0
    torch.testing.assert_close(law[..., -1], torch.ones_like(law[..., -1]))
    law.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert not a.grad.count_nonzero() and not b.grad.count_nonzero()
