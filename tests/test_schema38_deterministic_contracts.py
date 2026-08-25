from __future__ import annotations

import math

import torch

from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler


def test_g3_post_tanh_common_gauge_is_forward_and_backward_invariant() -> None:
    """Deleting a K-common logit is an exact softmax reparameterization."""

    torch.manual_seed(38001)
    parent = torch.softmax(torch.randn(3, 11, 4, dtype=torch.float64), dim=-1)
    bounded = (0.5 * torch.tanh(torch.randn_like(parent))).requires_grad_()
    probe = torch.randn_like(parent)

    common = (bounded * parent).sum(dim=-1, keepdim=True)
    historical = torch.softmax(parent.log() + bounded - common, dim=-1)
    direct = torch.softmax(parent.log() + bounded, dim=-1)
    torch.testing.assert_close(historical, direct, atol=2.0e-16, rtol=2.0e-15)

    historical_gradient = torch.autograd.grad(
        (historical * probe).sum(),
        bounded,
        retain_graph=True,
    )[0]
    direct_gradient = torch.autograd.grad((direct * probe).sum(), bounded)[0]
    torch.testing.assert_close(
        historical_gradient,
        direct_gradient,
        atol=3.0e-16,
        rtol=3.0e-14,
    )


def test_covariance_keeps_historical_initial_variance_without_output_floor() -> None:
    compiler = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=8,
        route_dim=4,
        heads=4,
    )
    initial = compiler._covariance_from_raw(compiler.covariance_head.bias)
    historical_floor = (2.0 / 7.0) ** 2
    historical_initial = historical_floor + (
        1.0 - historical_floor
    ) / (1.0 + math.exp(3.0))
    torch.testing.assert_close(
        initial,
        torch.tensor(
            (historical_initial, 0.0, historical_initial),
            dtype=initial.dtype,
        ),
        atol=2.0e-7,
        rtol=2.0e-7,
    )

    # A valid physical covariance below the former one-cell floor is now
    # representable and receives an ordinary gradient toward a zero target.
    raw = torch.tensor((-4.0, -4.0, 0.7), requires_grad=True)
    covariance = compiler._covariance_from_raw(raw)
    xx, xy, yy = covariance.unbind(dim=-1)
    assert 0.0 < float(xx.detach()) < historical_floor
    assert 0.0 < float(yy.detach()) < historical_floor
    assert float((xx * yy - xy.square()).detach()) >= -1.0e-8
    covariance.square().mean().backward()
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert float(raw.grad[0]) > 0.0
    assert float(raw.grad[1]) > 0.0


def test_covariance_serialization_boundary_keeps_parameter_names_and_shapes() -> None:
    source = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=8,
        route_dim=4,
        heads=4,
    )
    target = ObjectFutureDynamicsCompiler(
        hidden=16,
        content_dim=8,
        route_dim=4,
        heads=4,
    )
    covariance_parameters = {
        name: tuple(parameter.shape)
        for name, parameter in source.named_parameters()
        if name.startswith("covariance_head.")
    }
    assert covariance_parameters == {
        "covariance_head.weight": (3, 16),
        "covariance_head.bias": (3,),
    }
    target.load_state_dict(source.state_dict(), strict=True)
    raw = torch.randn(2, 4, 3)
    torch.testing.assert_close(
        source._covariance_from_raw(raw),
        target._covariance_from_raw(raw),
        atol=0.0,
        rtol=0.0,
    )
