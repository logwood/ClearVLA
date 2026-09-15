from __future__ import annotations

import torch

from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE_ARM_COARSE_CONTEXT_IMPLEMENTATION,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    ArmCoarseContextBSpine,
)


def _module(hidden: int = 16) -> ArmCoarseContextBSpine:
    return ArmCoarseContextBSpine(
        horizon=24,
        hidden_size=hidden,
        arm_dim=6,
        gripper_field_dim=6,
        degree=3,
        control_points=12,
        expected_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        expected_basis_digest=BSPINE0_BASIS_DIGEST,
        expected_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    )


def test_coarse_context_has_no_detail_owner_and_ignores_gripper_fields() -> None:
    module = _module()
    assert module.implementation_id == BSPINE_ARM_COARSE_CONTEXT_IMPLEMENTATION
    assert tuple(module.detail_lifts) == ()
    assert sum(parameter.numel() for parameter in module.parameters()) == 2 * 6 * 16
    arm = torch.randn(2, 24, 12)
    first = torch.cat((arm, torch.zeros(2, 24, 6)), dim=-1)
    second = torch.cat((arm, torch.randn(2, 24, 6)), dim=-1)
    first_tokens, first_metrics = module(first, collect_diagnostics=True)
    second_tokens, second_metrics = module(second, collect_diagnostics=True)
    torch.testing.assert_close(first_tokens, second_tokens)
    assert float(first_metrics["bottom_spine_detail_path_active"]) == 0.0
    assert float(second_metrics["bottom_spine_gripper_raw_only"]) == 1.0


def test_coarse_context_is_zero_initialized_and_has_cross_time_owner_vjp() -> None:
    module = _module()
    physical = torch.randn(2, 24, 18)
    tokens, _ = module(physical)
    assert torch.count_nonzero(tokens) == 0
    with torch.no_grad():
        for parameter in module.coarse_lifts.parameters():
            parameter.fill_(0.01)
    tokens, metrics = module(physical, collect_diagnostics=True)
    assert bool(torch.isfinite(tokens).all())
    assert float(metrics["bottom_spine_update_rms"]) > 0.0
    # A row-local impulse changes another output row through the coarse chart.
    impulse = torch.zeros_like(physical)
    impulse[:, 0, 0] = 1.0
    changed, _ = module(impulse)
    assert float(changed[:, 1:].detach().abs().sum()) > 0.0
    loss = tokens.square().mean()
    loss.backward()
    gradients = tuple(module.coarse_lifts.parameters())
    assert gradients and all(parameter.grad is not None for parameter in gradients)
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in gradients)
    assert all(float(parameter.grad.abs().sum()) > 0.0 for parameter in gradients)


def test_coarse_context_bf16_preserves_runtime_dtype_and_finite_vjp() -> None:
    module = _module()
    with torch.no_grad():
        for parameter in module.coarse_lifts.parameters():
            parameter.fill_(0.01)
    physical = torch.randn(2, 24, 18).to(torch.bfloat16).requires_grad_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, metrics = module(physical, collect_diagnostics=True)
        loss = tokens.float().square().mean()
    assert tokens.dtype == torch.bfloat16
    assert all(value.dtype == torch.float32 for value in metrics.values())
    loss.backward()
    assert physical.grad is not None
    assert bool(torch.isfinite(physical.grad).all())
