from __future__ import annotations

import torch

from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_SPEC_FINGERPRINT,
    BSpine0,
)

_BASIS_DIGEST = BSPINE0_BASIS_DIGEST


def _spine(*, hidden: int = 32) -> BSpine0:
    return BSpine0(
        horizon=24,
        hidden_size=hidden,
        arm_dim=6,
        gripper_field_dim=6,
        degree=3,
        control_points=12,
        expected_basis_digest=_BASIS_DIGEST,
        expected_spec_fingerprint=BSPINE0_SPEC_FINGERPRINT,
    )


def test_bspine0_fixed_chart_is_exact_and_zero_initialized() -> None:
    spine = _spine()
    physical = torch.randn(3, 24, 18)
    controls, coarse, detail = spine.decompose(physical)
    assert controls.shape == (3, 12, 18)
    assert coarse.shape == detail.shape == physical.shape
    torch.testing.assert_close(coarse + detail, physical, atol=5.0e-7, rtol=0.0)

    tokens, metrics = spine(physical, collect_diagnostics=True)
    assert tokens.shape == (3, 24, 32)
    assert torch.count_nonzero(tokens) == 0
    assert float(metrics["bottom_spine_decomposition_max_abs"]) <= 5.0e-7
    assert float(metrics["bottom_spine_zero_intervention_active"]) == 0.0
    assert sum(parameter.numel() for parameter in spine.parameters()) == 2 * 18 * 32
    assert all(module.bias is None for module in spine.coarse_lifts.values())
    assert all(module.bias is None for module in spine.detail_lifts.values())


def test_bspine0_basis_partition_rank_and_endpoints_are_fixed() -> None:
    spine = _spine()
    synthesis = spine.synthesis
    analysis = spine.analysis

    assert synthesis.dtype == analysis.dtype == torch.float32
    assert tuple(synthesis.shape) == (24, 12)
    assert tuple(analysis.shape) == (12, 24)
    assert int(torch.linalg.matrix_rank(synthesis)) == 12
    torch.testing.assert_close(
        synthesis.sum(dim=-1),
        torch.ones(24),
        atol=5.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        synthesis[0],
        torch.nn.functional.one_hot(torch.tensor(0), num_classes=12).float(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        synthesis[-1],
        torch.nn.functional.one_hot(torch.tensor(11), num_classes=12).float(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        analysis @ synthesis,
        torch.eye(12),
        atol=2.0e-6,
        rtol=0.0,
    )


def test_bspine0_construction_does_not_consume_host_rng() -> None:
    torch.manual_seed(731)
    before = torch.get_rng_state().clone()
    _spine()
    assert torch.equal(torch.get_rng_state(), before)


def test_bspine0_zero_initialization_still_trains_both_views() -> None:
    spine = _spine()
    physical = torch.randn(2, 24, 18)
    tokens, _ = spine(physical)
    probe = torch.randn_like(tokens)
    (tokens * probe).sum().backward()

    for _, parameter in spine.named_parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_bspine0_preserves_reverse_path_after_learning() -> None:
    spine = _spine()
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for parameter in spine.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 1.0e-2
            )
    physical = torch.randn(2, 24, 18, requires_grad=True)
    tokens, metrics = spine(physical, collect_diagnostics=True)
    assert bool(torch.isfinite(tokens).all())
    assert float(metrics["bottom_spine_update_rms"]) > 0.0
    tokens.square().mean().backward()
    assert physical.grad is not None
    assert bool(torch.isfinite(physical.grad).all())
    assert float(physical.grad.abs().sum()) > 0.0


def test_bspine0_coarse_and_detail_views_have_independent_jvp_vjp() -> None:
    generator = torch.Generator().manual_seed(91)
    physical = torch.randn(2, 24, 18, generator=generator)
    tangent = torch.randn(physical.shape, generator=generator)
    probe = torch.randn(2, 24, 16, generator=generator)

    for active_branch, inactive_branch in (
        ("coarse_lifts", "detail_lifts"),
        ("detail_lifts", "coarse_lifts"),
    ):
        spine = _spine(hidden=16)
        with torch.no_grad():
            for parameter in getattr(spine, active_branch).parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 1.0e-2
                )
        getattr(spine, inactive_branch).requires_grad_(False)

        output, jvp = torch.autograd.functional.jvp(
            lambda value: spine(value)[0],
            physical,
            tangent,
            create_graph=False,
            strict=True,
        )
        assert bool(torch.isfinite(output).all())
        assert bool(torch.isfinite(jvp).all())
        assert float(output.abs().sum()) > 0.0
        assert float(jvp.abs().sum()) > 0.0

        vjp_input = physical.detach().requires_grad_(True)
        vjp_output, _ = spine(vjp_input)
        (vjp_output * probe).sum().backward()
        assert vjp_input.grad is not None
        assert bool(torch.isfinite(vjp_input.grad).all())
        assert float(vjp_input.grad.abs().sum()) > 0.0
        active_gradients = [
            parameter.grad
            for parameter in getattr(spine, active_branch).parameters()
        ]
        assert all(gradient is not None for gradient in active_gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in active_gradients)
        assert all(float(gradient.abs().sum()) > 0.0 for gradient in active_gradients)


def test_bspine0_cpu_bf16_autocast_preserves_dtype_and_reverse_path() -> None:
    spine = _spine(hidden=16)
    generator = torch.Generator().manual_seed(92)
    with torch.no_grad():
        for parameter in spine.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
                * 1.0e-2
            )
    physical = torch.randn(2, 24, 18, generator=generator).to(torch.bfloat16)
    physical.requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, metrics = spine(physical, collect_diagnostics=True)
        loss = tokens.float().square().mean()
    assert tokens.dtype == torch.bfloat16
    assert tokens.shape == (2, 24, 16)
    assert all(value.dtype == torch.float32 for value in metrics.values())
    assert bool(torch.isfinite(tokens).all())
    loss.backward()
    assert physical.grad is not None
    assert physical.grad.dtype == torch.bfloat16
    assert bool(torch.isfinite(physical.grad).all())
    assert float(physical.grad.abs().sum()) > 0.0
    for parameter in spine.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_bspine0_zero_intervention_keeps_computation_but_removes_output() -> None:
    spine = _spine()
    with torch.no_grad():
        for parameter in spine.parameters():
            parameter.fill_(0.01)
    physical = torch.randn(2, 24, 18)
    primary, primary_metrics = spine(physical, collect_diagnostics=True)
    zeroed, zeroed_metrics = spine(
        physical,
        zero_output=True,
        collect_diagnostics=True,
    )
    assert float(primary.detach().abs().sum()) > 0.0
    assert torch.count_nonzero(zeroed) == 0
    torch.testing.assert_close(
        primary_metrics["bottom_spine_update_rms"],
        zeroed_metrics["bottom_spine_update_rms"],
    )
    assert float(zeroed_metrics["bottom_spine_zero_intervention_active"]) == 1.0


def test_bspine0_rejects_wrong_serialized_basis_identity() -> None:
    try:
        BSpine0(
            horizon=24,
            hidden_size=32,
            arm_dim=6,
            gripper_field_dim=6,
            degree=3,
            control_points=12,
            expected_basis_digest="0" * 64,
            expected_spec_fingerprint=BSPINE0_SPEC_FINGERPRINT,
        )
    except ValueError as error:
        assert "basis digest" in str(error)
    else:
        raise AssertionError("B-spine accepted the wrong serialized basis identity")
