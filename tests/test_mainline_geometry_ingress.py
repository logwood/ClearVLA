import torch

from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler


def _compiler(mode: str) -> ObjectFutureDynamicsCompiler:
    return ObjectFutureDynamicsCompiler(
        hidden=8,
        content_dim=8,
        route_dim=4,
        action_dim=7,
        heads=2,
        normalization_floor=0.25,
        geometry_ingress_mode=mode,
    )


def test_geometry_rms_floored_ingress_preserves_zero_and_direction() -> None:
    compiler = _compiler("rms_floored_v1")
    zero = torch.zeros(2, 4, 8)
    normalized, denominator = compiler._geometry_ingress(zero)
    torch.testing.assert_close(normalized, zero, atol=0.0, rtol=0.0)
    torch.testing.assert_close(denominator, torch.full_like(denominator, 0.25))

    value = torch.full((1, 4, 8), 0.05)
    normalized, denominator = compiler._geometry_ingress(value)
    assert torch.isfinite(normalized).all()
    assert torch.all(denominator > 0.25)
    assert normalized.abs().mean() > value.abs().mean()
    torch.testing.assert_close(
        normalized / normalized[..., :1],
        value / value[..., :1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_legacy_geometry_ingress_is_bit_exact() -> None:
    compiler = _compiler("legacy")
    value = torch.randn(2, 4, 8)
    normalized, denominator = compiler._geometry_ingress(value)
    torch.testing.assert_close(normalized, value, atol=0.0, rtol=0.0)
    torch.testing.assert_close(denominator, torch.ones_like(denominator))
