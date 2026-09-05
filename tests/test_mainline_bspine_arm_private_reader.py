from __future__ import annotations

from dataclasses import replace

import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.interfaces import (
    ActionSupervision,
    CurrentObservation,
    FutureSupervision,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
    TrainingBatch,
)
from clearvla.mainline.manifest import (
    ARM_PRIVATE_READER_BSPINE_ARCHITECTURE_MANIFEST,
    architecture_manifest_for_bspine_implementation,
)
from clearvla.mainline.model.component_contracts import (
    BSPINE_ARM_PRIVATE_READER_EXECUTION_BOTTOM,
    legacy_state_dict,
    map_legacy_state_dict,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.training.optimizer import build_optimizer, gradient_diagnostics
from clearvla.mainline.v120_core.bspine import (
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    BSPINE_ARM_PRIVATE_READER_IMPLEMENTATION,
    ArmPrivateBSpineReader,
    ArmPrivateReaderBSpine,
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


def _private_config() -> ExperimentConfig:
    base = _small_config()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE_ARM_PRIVATE_READER_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
            bspine_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        ),
    )
    config.validate()
    return config


def _batch(config: ExperimentConfig, *, batch: int = 1) -> TrainingBatch:
    dims = config.dimensions
    device = torch.device("cpu")
    online = OnlinePolicyInput(
        observation=CurrentObservation(
            dino_history=torch.randn(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
                device=device,
            ),
            raw_rgb=torch.rand(
                batch,
                dims.visual_history_length,
                dims.num_cameras,
                3,
                48,
                48,
                device=device,
            ),
        ),
        history=ObservableHistory(
            state=torch.randn(batch, dims.state_dim),
            action_state=torch.randn(batch, dims.action_dim),
            codec_gripper_boundary=torch.randn(batch, 1),
            state_history=torch.randn(batch, dims.state_history_length, dims.state_dim),
            executed_action_history=torch.randn(
                batch,
                dims.executed_history_length,
                dims.action_dim,
            ),
        ),
        goal=GoalCondition(
            tokens=torch.randn(batch, 6, dims.goal_token_dim),
            mask=torch.ones(batch, 6, dtype=torch.bool),
        ),
    )
    offsets = torch.tensor(
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        dtype=torch.long,
    )[None].expand(batch, -1)
    return TrainingBatch(
        online=online,
        action_target=ActionSupervision(
            normalized=torch.randn(batch, dims.action_horizon, dims.action_dim),
            raw_units=torch.randn(batch, dims.action_horizon, dims.action_dim),
            current_raw_units=torch.randn(batch, dims.action_dim),
            gripper_transition_boundary=torch.zeros(batch, dims.action_dim),
            gripper_transition_boundary_raw_units=torch.zeros(batch, dims.action_dim),
        ),
        future=FutureSupervision(
            dino_supports=torch.randn(
                batch,
                dims.future_supports,
                dims.num_cameras,
                dims.patches_per_camera,
                dims.visual_token_dim,
            ),
            action_sequence=torch.randn(batch, 48, dims.action_dim),
            state_sequence=torch.randn(batch, 48, dims.state_dim),
            offsets=offsets,
        ),
    )


def _spine(hidden: int = 8) -> ArmPrivateReaderBSpine:
    return ArmPrivateReaderBSpine(
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


def test_private_reader_is_rng_stable_zero_start_and_explicitly_arm_only() -> None:
    torch.manual_seed(6100)
    before = torch.get_rng_state().clone()
    spine = _spine(hidden=8)
    reader = ArmPrivateBSpineReader(hidden_size=8, arm_dim=6)
    assert torch.equal(torch.get_rng_state(), before)
    assert spine.implementation_id == BSPINE_ARM_PRIVATE_READER_IMPLEMENTATION
    assert spine.detail_path is False
    assert tuple(spine.detail_lifts) == ()
    assert tuple(spine.state_dict()) == (
        "analysis",
        "synthesis",
        "action_group_mask",
        "coarse_lifts.arm_absolute.weight",
        "coarse_lifts.arm_delta.weight",
    )
    assert tuple(reader.state_dict()) == ("projection.weight",)
    assert float(reader.projection.weight.detach().abs().sum()) > 0.0
    zero_tokens = torch.zeros(2, 24, 8)
    correction = reader(zero_tokens)
    assert tuple(correction.shape) == (2, 24, 12)
    assert torch.count_nonzero(correction) == 0


def test_private_reader_bf16_storage_keeps_fp32_math_and_reverse_path() -> None:
    reader = ArmPrivateBSpineReader(hidden_size=8, arm_dim=6).to(torch.bfloat16)
    tokens = torch.randn(2, 24, 8).to(torch.bfloat16).requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        correction = reader(tokens)
        loss = correction.float().square().mean()
    assert correction.dtype == torch.bfloat16
    assert bool(torch.isfinite(correction).all())
    loss.backward()
    assert tokens.grad is not None
    assert bool(torch.isfinite(tokens.grad).all())
    assert reader.projection.weight.grad is not None
    assert bool(torch.isfinite(reader.projection.weight.grad).all())
    assert float(reader.projection.weight.grad.abs().sum()) > 0.0


def test_private_reader_gripper_loss_has_zero_reader_and_spine_vjp() -> None:
    spine = _spine(hidden=8)
    reader = ArmPrivateBSpineReader(hidden_size=8, arm_dim=6)
    with torch.no_grad():
        spine.coarse_lifts["arm_absolute"].weight.normal_(std=0.03)
        spine.coarse_lifts["arm_delta"].weight.normal_(std=0.03)
    physical = torch.randn(2, 24, 18)
    spine_tokens, _ = spine(physical)
    correction = reader(spine_tokens)
    # A gripper-only terminal loss cannot see the private arm correction.  The
    # explicit concatenation boundary is therefore checked with autograd, not
    # merely by comparing a detached output slice.
    arm_state = torch.randn(2, 24, 32, requires_grad=True)
    from clearvla.mainline.v120_core.decoder import ActionOnlyPhysicalVelocityHead
    from clearvla.mainline.v120_core.profile import build_v120_policy_config
    from clearvla.mainline.v120_core.time_domain_mmdit import TerminalActionController

    core = build_v120_policy_config()
    core = replace(
        core,
        hidden_size=32,
        action_horizon=24,
        arm_flow_mode="legacy_independent",
        gripper_field_dim=6,
        gripper_field_mode="legacy_handcrafted",
        gripper_output_mode="continuous",
        num_heads=4,
    )
    controller = TerminalActionController(
        action_norm=torch.nn.LayerNorm(32, elementwise_affine=False),
        velocity_head=ActionOnlyPhysicalVelocityHead(core),
        optional_command_head=None,
        optional_event_head=None,
        motion_head=torch.nn.Sequential(torch.nn.LayerNorm(32), torch.nn.Linear(32, 1)),
        arm_dim=6,
    )
    output = controller.read_heads(
        arm_state,
        arm_private_correction=correction,
        collect_diagnostics=False,
    )
    gripper_loss = output.physical_velocity[..., 12:].square().mean()
    grads = torch.autograd.grad(
        gripper_loss,
        tuple(spine.parameters()) + tuple(reader.parameters()),
        allow_unused=True,
    )
    assert all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in grads)


def test_private_reader_factory_preserves_raw_path_and_optimizer_ownership() -> None:
    baseline_config = _small_config()
    private_config = _private_config()
    torch.manual_seed(6110)
    baseline = ClearVLAMainlinePolicy(baseline_config).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(6110)
    private = ClearVLAMainlinePolicy(private_config).eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert private.selection.execution_bottom == BSPINE_ARM_PRIVATE_READER_EXECUTION_BOTTOM
    assert isinstance(private.execution_bottom.decoder.spine, ArmPrivateReaderBSpine)
    assert isinstance(private.execution_bottom.decoder.arm_private_reader, ArmPrivateBSpineReader)
    baseline_parameters = dict(baseline.named_parameters())
    private_parameters = dict(private.named_parameters())
    extra = set(private_parameters).difference(baseline_parameters)
    assert len(extra) == 3
    assert any(".spine.coarse_lifts." in name for name in extra)
    assert "execution_bottom.decoder.arm_private_reader.projection.weight" in extra
    for name, parameter in baseline_parameters.items():
        torch.testing.assert_close(private_parameters[name], parameter, atol=0.0, rtol=0.0)

    baseline_optimizer, baseline_ownership = build_optimizer(baseline, baseline_config)
    private_optimizer, private_ownership = build_optimizer(private, private_config)
    baseline_groups = {
        str(group["name"]): tuple(group["parameter_names"])
        for group in baseline_optimizer.param_groups
    }
    private_groups = {
        str(group["name"]): tuple(group["parameter_names"])
        for group in private_optimizer.param_groups
    }
    private_spine_group = private_groups.pop("bottom_spine/decay")
    assert private_groups == baseline_groups
    assert private_ownership.role_counts["bottom_spine"] == 3
    assert "bottom_spine" not in baseline_ownership.role_counts
    assert len(private_spine_group) == 3
    assert "bottom.decoder.arm_private_reader.projection.weight" in private_spine_group

    # The compatibility ledger sees the optional owner exactly once.
    legacy = legacy_state_dict(private)
    mapped = map_legacy_state_dict(private, legacy)
    assert set(mapped) == set(private.state_dict())

    for parameter in private.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    diagnostics = gradient_diagnostics(private, stage="raw")
    assert diagnostics["gradient_raw_bottom_spine_private_reader_l2"] > 0.0


def test_private_reader_path_changes_arm_velocity_but_not_gripper_or_shared_query() -> None:
    config = _private_config()
    torch.manual_seed(6120)
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        cache, _, _ = model.encode_online(batch.online)
        physical = torch.randn(1, 24, 18, generator=torch.Generator().manual_seed(6121))
        time = torch.full((1,), 0.45)
        baseline = model.velocity(
            cache,
            noisy_action_field=physical,
            time=time,
            collect_diagnostics=True,
        )
        # Open the optional route after construction.  The shared action seed
        # remains raw-only, so only the arm terminal read should move.
        spine = model.execution_bottom.decoder.spine
        reader = model.execution_bottom.decoder.arm_private_reader
        assert spine is not None and reader is not None
        spine.coarse_lifts["arm_absolute"].weight.zero_()
        spine.coarse_lifts["arm_delta"].weight.zero_()
        spine.coarse_lifts["arm_absolute"].weight[0, 0] = 0.05
        reader.projection.weight.zero_()
        reader.projection.weight[0, 0] = 0.20
        changed = model.velocity(
            cache,
            noisy_action_field=physical,
            time=time,
            collect_diagnostics=True,
        )
    velocity_delta = changed.bottom.physical_velocity - baseline.bottom.physical_velocity
    assert float(velocity_delta[..., :12].abs().sum()) > 0.0
    assert torch.count_nonzero(velocity_delta[..., 12:]) == 0
    torch.testing.assert_close(
        changed.bottom.motion_logits,
        baseline.bottom.motion_logits,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        changed.bottom.action_query,
        baseline.bottom.action_query,
        atol=0.0,
        rtol=0.0,
    )
    assert changed.metrics["bottom_spine_arm_private_reader_active"] == 1.0
    assert changed.metrics["bottom_spine_arm_private_correction_rms"] > 0.0


def test_private_reader_config_and_manifest_are_explicit() -> None:
    config = load_config(
        "configs/mainline/object_intent_dynamics_323_pen_bspine_arm_private_reader.json"
    )
    assert config.bottom.bspine_implementation == BSPINE_ARM_PRIVATE_READER_IMPLEMENTATION
    assert "bspine_action_group_mask" in config.as_dict()["bottom"]
    manifest = architecture_manifest_for_bspine_implementation(config.bottom.bspine_implementation)
    assert manifest is ARM_PRIVATE_READER_BSPINE_ARCHITECTURE_MANIFEST
    assert manifest.schema == 31
    assert "private_reader" in manifest.components.bottom
