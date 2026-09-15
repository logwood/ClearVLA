from __future__ import annotations

from dataclasses import replace

import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.component_contracts import (
    BSPINE_ARM_ONLY_EXECUTION_BOTTOM,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.optimizer import build_optimizer
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_IMPLEMENTATION,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    ArmOnlyBSpine,
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


def _arm_only_config() -> ExperimentConfig:
    base = _small_config()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE_ARM_ONLY_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
            bspine_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        ),
    )
    config.validate()
    return config


def test_arm_only_integration_preserves_raw_path_abi_rng_and_optimizer_owners() -> None:
    baseline_config = _small_config()
    arm_only_config = _arm_only_config()

    torch.manual_seed(901)
    baseline = ClearVLAMainlinePolicy(baseline_config)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(901)
    candidate = ClearVLAMainlinePolicy(arm_only_config)
    candidate_rng = torch.get_rng_state().clone()

    assert torch.equal(candidate_rng, baseline_rng)
    assert candidate.selection.execution_bottom == BSPINE_ARM_ONLY_EXECUTION_BOTTOM
    assert isinstance(candidate.execution_bottom.decoder.spine, ArmOnlyBSpine)
    assert baseline.execution_bottom.decoder.spine is None

    baseline_parameters = dict(baseline.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    extra_parameters = set(candidate_parameters).difference(baseline_parameters)
    assert len(extra_parameters) == 4
    assert all(
        name.startswith("execution_bottom.decoder.spine.")
        for name in extra_parameters
    )
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
        generator=torch.Generator().manual_seed(902),
    )
    baseline_raw = baseline.execution_bottom.decoder.noisy_lift(physical)
    candidate_raw = candidate.execution_bottom.decoder.noisy_lift(physical)
    torch.testing.assert_close(candidate_raw, baseline_raw, atol=0.0, rtol=0.0)
    spine_tokens, _ = candidate.execution_bottom.decoder.spine(physical)
    assert spine_tokens.shape == baseline_raw.shape
    assert torch.count_nonzero(spine_tokens) == 0

    assert baseline.execution_bottom.physical_action_dim == 18
    assert candidate.execution_bottom.physical_action_dim == 18
    assert int(baseline.execution_bottom.decoder.config.physical_action_dim) == 18
    assert int(candidate.execution_bottom.decoder.config.physical_action_dim) == 18
    assert (
        tuple(baseline.execution_bottom.decoder.terminal_controller.state_dict())
        == tuple(candidate.execution_bottom.decoder.terminal_controller.state_dict())
    )

    baseline_optimizer, baseline_ownership = build_optimizer(
        baseline,
        baseline_config,
    )
    candidate_optimizer, candidate_ownership = build_optimizer(
        candidate,
        arm_only_config,
    )
    baseline_groups = {
        str(group["name"]): (
            float(group["lr"]),
            float(group["weight_decay"]),
            tuple(group["parameter_names"]),
        )
        for group in baseline_optimizer.param_groups
    }
    candidate_groups = {
        str(group["name"]): (
            float(group["lr"]),
            float(group["weight_decay"]),
            tuple(group["parameter_names"]),
        )
        for group in candidate_optimizer.param_groups
    }
    spine_group = candidate_groups.pop("bottom_spine/decay")
    assert candidate_groups == baseline_groups
    assert set(baseline_ownership.role_counts) == set(
        candidate_ownership.role_counts
    ) - {"bottom_spine"}
    assert candidate_ownership.role_counts["bottom_spine"] == 4
    assert len(spine_group[2]) == 4
    assert len(spine_group[2]) == len(set(spine_group[2]))
    assert all(name.startswith("bottom.decoder.spine.") for name in spine_group[2])
