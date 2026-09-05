from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.component_contracts import (
    BSPINE_ARM_COARSE_CONTEXT_EXECUTION_BOTTOM,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.optimizer import build_optimizer
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE_ARM_COARSE_CONTEXT_IMPLEMENTATION,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    ArmCoarseContextBSpine,
    ArmOnlyBSpine,
    validate_bspine_module,
)


def _spine(*, hidden: int = 16) -> ArmCoarseContextBSpine:
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


def _old_arm_only(*, hidden: int = 16) -> ArmOnlyBSpine:
    return ArmOnlyBSpine(
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


def _small_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            action_basis_tokens=2,
            hidden_size=32,
            num_heads=4,
            visual_token_dim=16,
            goal_token_dim=16,
            patches_per_camera=64,
        ),
        observation=replace(
            base.observation,
            feature_dim=16,
            address_route_dim=8,
            flow_iterations=2,
            correlation_radius=1,
            raw_base_channels=8,
        ),
        top=replace(
            base.top,
            grounder_iterations=2,
            teacher_key_dim=8,
            goal_condition_dropout=0.0,
            action_history_condition_dropout=0.0,
        ),
        bottom=replace(
            base.bottom,
            operator_rank=8,
            operator_groups=8,
            controller_tokens=4,
            controller_depth=1,
            controller_heads=4,
        ),
        optimizer=replace(base.optimizer, warmup_steps=2),
        runtime=replace(base.runtime, compute_dtype="fp32"),
    )
    config.validate()
    return config


def _candidate_config() -> ExperimentConfig:
    base = _small_config()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE_ARM_COARSE_CONTEXT_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
            bspine_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        ),
    )
    config.validate()
    return config


def _learn(spine: ArmCoarseContextBSpine, *, seed: int = 1201) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in spine.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 1.0e-2)


def test_arm_coarse_context_is_zero_rng_stable_and_has_no_detail_owner() -> None:
    torch.manual_seed(1200)
    before = torch.get_rng_state().clone()
    spine = _spine(hidden=32)
    assert torch.equal(torch.get_rng_state(), before)
    assert spine.implementation_id == BSPINE_ARM_COARSE_CONTEXT_IMPLEMENTATION
    assert spine.detail_path is False
    assert len(spine.detail_lifts) == 0
    assert sum(parameter.numel() for parameter in spine.parameters()) == 12 * 32
    assert not any("detail" in name for name, _ in spine.named_parameters())
    assert tuple(spine.state_dict()) == (
        "analysis",
        "synthesis",
        "action_group_mask",
        "coarse_lifts.arm_absolute.weight",
        "coarse_lifts.arm_delta.weight",
    )

    physical = torch.randn(2, 24, 18)
    tokens, metrics = spine(physical, collect_diagnostics=True)
    assert torch.count_nonzero(tokens) == 0
    assert float(metrics["bottom_spine_detail_path_active"]) == 0.0
    assert float(metrics["bottom_spine_detail_token_rms"]) == 0.0
    assert float(metrics["bottom_spine_gripper_raw_only"]) == 1.0
    assert float(metrics["bottom_spine_decomposition_max_abs"]) <= 5.0e-7


def test_arm_coarse_context_is_exactly_the_lifted_coarse_projection() -> None:
    spine = _spine(hidden=7)
    _learn(spine)
    physical = torch.randn(2, 24, 18)
    _, coarse, _ = spine.decompose(physical)
    expected = torch.nn.functional.linear(
        coarse[..., :6],
        spine.coarse_lifts["arm_absolute"].weight,
    ) + torch.nn.functional.linear(
        coarse[..., 6:12],
        spine.coarse_lifts["arm_delta"].weight,
    )
    actual, metrics = spine(physical, collect_diagnostics=True)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2.0e-6, rtol=0.0)
    assert float(metrics["bottom_spine_detail_path_active"]) == 0.0


def test_arm_coarse_context_couples_arm_information_across_action_time() -> None:
    spine = _spine(hidden=4)
    with torch.no_grad():
        spine.coarse_lifts["arm_absolute"].weight.zero_()
        spine.coarse_lifts["arm_delta"].weight.zero_()
        spine.coarse_lifts["arm_absolute"].weight[0, 0] = 1.0
    physical = torch.zeros(1, 24, 18)
    source_row = 12
    physical[0, source_row, 0] = 1.0
    tokens, _ = spine(physical)
    affected_rows = torch.nonzero(tokens[0, :, 0].abs() > 1.0e-7).flatten()
    assert affected_rows.numel() > 1
    assert bool((affected_rows != source_row).any())


def test_arm_coarse_context_ignores_gripper_input_and_has_zero_gripper_vjp() -> None:
    spine = _spine(hidden=8)
    _learn(spine, seed=1202)
    generator = torch.Generator().manual_seed(1203)
    physical = torch.randn(2, 24, 18, generator=generator)
    changed = physical.clone()
    changed[..., 12:] = torch.randn(2, 24, 6, generator=generator) * 9.0
    expected, _ = spine(physical)
    actual, _ = spine(changed)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    differentiable = physical.detach().requires_grad_(True)
    output, _ = spine(differentiable)
    output.square().sum().backward()
    assert differentiable.grad is not None
    assert float(differentiable.grad[..., :12].abs().sum()) > 0.0
    assert torch.count_nonzero(differentiable.grad[..., 12:]) == 0


def test_arm_coarse_context_all_parameter_owners_receive_finite_gradient() -> None:
    spine = _spine(hidden=8)
    generator = torch.Generator().manual_seed(1204)
    physical = torch.randn(2, 24, 18, generator=generator)
    probe = torch.randn(2, 24, 8, generator=generator)
    tokens, _ = spine(physical)
    (tokens * probe).sum().backward()
    gradients = [parameter.grad for parameter in spine.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_arm_coarse_context_cpu_bf16_preserves_reverse_path() -> None:
    spine = _spine(hidden=8)
    _learn(spine, seed=1205)
    physical = torch.randn(2, 24, 18).to(torch.bfloat16).requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        tokens, metrics = spine(physical, collect_diagnostics=True)
        loss = tokens.float().square().mean()
    assert tokens.dtype == torch.bfloat16
    assert all(value.dtype == torch.float32 for value in metrics.values())
    loss.backward()
    assert physical.grad is not None
    assert physical.grad.dtype == torch.bfloat16
    assert bool(torch.isfinite(physical.grad).all())
    assert float(physical.grad[..., :12].abs().sum()) > 0.0
    assert torch.count_nonzero(physical.grad[..., 12:]) == 0


def test_arm_coarse_context_identity_and_old_arm_only_state_are_separate() -> None:
    coarse = _spine()
    old = _old_arm_only()
    assert validate_bspine_module(coarse) is coarse
    assert any("detail_lifts" in name for name in old.state_dict())
    _, old_metrics = old(torch.randn(1, 24, 18), collect_diagnostics=True)
    assert "bottom_spine_detail_path_active" not in old_metrics
    with pytest.raises(RuntimeError):
        coarse.load_state_dict(old.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        old.load_state_dict(coarse.state_dict(), strict=True)


def test_arm_coarse_context_factory_preserves_raw_path_rng_and_owner_boundary() -> None:
    baseline_config = _small_config()
    candidate_config = _candidate_config()

    torch.manual_seed(1206)
    baseline = ClearVLAMainlinePolicy(baseline_config)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(1206)
    candidate = ClearVLAMainlinePolicy(candidate_config)
    candidate_rng = torch.get_rng_state().clone()

    assert torch.equal(candidate_rng, baseline_rng)
    assert candidate.selection.execution_bottom == BSPINE_ARM_COARSE_CONTEXT_EXECUTION_BOTTOM
    assert isinstance(
        candidate.execution_bottom.decoder.spine,
        ArmCoarseContextBSpine,
    )
    assert baseline.execution_bottom.decoder.spine is None

    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    extra_parameters = set(candidate_parameters).difference(baseline_parameters)
    assert len(extra_parameters) == 2
    assert all(".spine.coarse_lifts." in name for name in extra_parameters)
    assert not any("detail_lifts" in name for name in extra_parameters)
    for name, parameter in baseline_parameters.items():
        torch.testing.assert_close(
            candidate_parameters[name],
            parameter,
            atol=0.0,
            rtol=0.0,
        )

    physical = torch.randn(
        2,
        baseline_config.dimensions.action_horizon,
        18,
        generator=torch.Generator().manual_seed(1207),
    )
    baseline_raw = baseline.execution_bottom.decoder.noisy_lift(physical)
    candidate_raw = candidate.execution_bottom.decoder.noisy_lift(physical)
    torch.testing.assert_close(candidate_raw, baseline_raw, atol=0.0, rtol=0.0)
    spine_tokens, _ = candidate.execution_bottom.decoder.spine(physical)
    assert torch.count_nonzero(spine_tokens) == 0

    baseline_optimizer, baseline_ownership = build_optimizer(
        baseline,
        baseline_config,
    )
    candidate_optimizer, candidate_ownership = build_optimizer(
        candidate,
        candidate_config,
    )
    baseline_groups = {
        str(group["name"]): tuple(group["parameter_names"])
        for group in baseline_optimizer.param_groups
    }
    candidate_groups = {
        str(group["name"]): tuple(group["parameter_names"])
        for group in candidate_optimizer.param_groups
    }
    spine_group = candidate_groups.pop("bottom_spine/decay")
    assert candidate_groups == baseline_groups
    assert candidate_ownership.role_counts["bottom_spine"] == 2
    assert "bottom_spine" not in baseline_ownership.role_counts
    assert len(spine_group) == 2
    assert all(".spine.coarse_lifts." in name for name in spine_group)
