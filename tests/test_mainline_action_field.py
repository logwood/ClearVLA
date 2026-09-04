from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import ActionSupervision, ObservableHistory
from clearvla.mainline.model.action_codec import (
    PhysicalActionFieldCodec,
    anchor_horizon_weights,
)
from clearvla.mainline.model.policy import PolicyStepOutput
from clearvla.mainline.model.types import PhysicalActionCondition
from clearvla.mainline.training.losses import (
    FlowMatchingState,
    action_terms,
    balanced_binary_command_weights,
    balanced_event_row_weights,
    causal_event_trajectory_mask,
    event_transition_persistence_masks,
    sample_flow_matching,
)
from clearvla.mainline.v120_core.codec import (
    NativeTimePhysicalActionTokenLift as CoreNativeTimePhysicalActionTokenLift,
)
from clearvla.mainline.v120_core.codec import PhysicalActionCodec as CorePhysicalActionCodec
from clearvla.mainline.v120_core.config import V362PolicyConfig as CorePolicyConfig
from clearvla.mainline.v120_core.decoder import ActionOnlyPhysicalVelocityHead


def _codec() -> PhysicalActionFieldCodec:
    return PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
    )


def test_formal_physical_action_field_is_exact_legacy_18d_chart() -> None:
    codec = _codec()
    state = torch.tensor([[0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.25]])
    action = state[:, None].expand(-1, 24, -1).clone()
    action[..., :6] += torch.linspace(0.01, 0.24, 24)[None, :, None]
    action[..., -1] = torch.linspace(0.20, 0.65, 24)
    field = codec.encode(action, state)
    assert tuple(field.shape) == (1, 24, 18)
    parts = codec.split(field)
    boundary = torch.cat((state[:, None], action[:, :-1]), dim=1)
    grip_delta = action[..., -1:] - boundary[..., -1:]
    torch.testing.assert_close(parts.arm_absolute, action[..., :6])
    torch.testing.assert_close(parts.arm_delta, action[..., :6] - boundary[..., :6])
    torch.testing.assert_close(parts.gripper_field[..., 0:1], action[..., -1:])
    torch.testing.assert_close(parts.gripper_field[..., 1:2], grip_delta)
    torch.testing.assert_close(
        parts.gripper_field[..., 2:3],
        action[..., -1:] - state[:, None, -1:],
    )
    torch.testing.assert_close(parts.gripper_field[..., 3:4], boundary[..., -1:])
    torch.testing.assert_close(parts.gripper_field[..., 4:5], grip_delta.abs())
    torch.testing.assert_close(parts.gripper_field[..., 5:6], torch.relu(grip_delta))
    torch.testing.assert_close(codec.decode(field, state), action)


def test_calvin_relative_commands_use_two_direct_arm_branches() -> None:
    codec = _direct_codec()
    state = torch.full((2, 7), 9.0)
    action = torch.randn(2, 24, 7)
    action[..., -1] = 1.0
    field = codec.encode(action, state)
    parts = codec.split(field)

    torch.testing.assert_close(parts.arm_absolute, action[..., :6])
    torch.testing.assert_close(parts.arm_delta, action[..., :6])
    old_temporal_difference = action[..., :6] - torch.cat(
        (state[:, None, :6], action[:, :-1, :6]),
        dim=1,
    )
    assert not torch.allclose(parts.arm_delta, old_temporal_difference)
    torch.testing.assert_close(codec.decode(field, state), action)


def test_calvin_direct_decode_and_projection_never_integrate_second_branch() -> None:
    codec = _direct_codec()
    state = torch.randn(1, 7)
    field = torch.randn(1, 24, codec.physical_dim)
    parts = codec.split(field)
    decoded = codec.decode(field, state)
    expected_arm = 0.75 * parts.arm_absolute + 0.25 * parts.arm_delta
    torch.testing.assert_close(decoded[..., :6], expected_arm)

    native, projected = codec.project_arm_tangent(field[..., :12])
    torch.testing.assert_close(native, expected_arm)
    torch.testing.assert_close(
        projected,
        field[..., :12],
        atol=0.0,
        rtol=0.0,
    )
    temporal = torch.cat((state[:, None, :6], decoded[:, :-1, :6]), dim=1)
    assert not torch.allclose(decoded[..., :6], state[:, None, :6] + parts.arm_delta.cumsum(1))
    assert not torch.allclose(native, decoded[..., :6] - temporal)


def test_calvin_motion_target_is_direct_command_magnitude() -> None:
    codec = _direct_codec()
    state = torch.zeros(1, 7)
    action = torch.zeros(1, 24, 7)
    action[:, :, 0] = 0.03
    magnitude = codec.arm_motion_magnitude(action, state)
    torch.testing.assert_close(magnitude, torch.full_like(magnitude, 0.03))

    legacy = _codec().arm_motion_magnitude(action, state)
    torch.testing.assert_close(legacy[:, 0], torch.tensor([0.03]))
    torch.testing.assert_close(legacy[:, 1:], torch.zeros_like(legacy[:, 1:]))


def test_calvin_profile_and_direct_arm_mode_fail_closed_together() -> None:
    base = ExperimentConfig()
    calvin_data = replace(
        base.data,
        data_profile="calvin_relative_7d_v1",
        split_mode="episode-manifest",
        split_manifest="calvin-test-splits.json",
        train_episodes=0,
        val_episodes=0,
        test_episodes=0,
        sampling_gripper_event_threshold=0.1,
    )
    calvin_objective = replace(
        base.objectives,
        gripper_command=0.1,
        gripper_event_threshold=0.1,
    )
    try:
        replace(
            base,
            data=calvin_data,
            bottom=replace(base.bottom, gripper_output_mode="calvin_binary_command"),
            objectives=calvin_objective,
        ).validate()
    except ValueError as error:
        assert "relative_command_direct" in str(error)
    else:
        raise AssertionError("CALVIN legacy arm chart was accepted")
    try:
        replace(
            base,
            bottom=replace(base.bottom, arm_flow_mode="relative_command_direct"),
        ).validate()
    except ValueError as error:
        assert "CALVIN" in str(error)
    else:
        raise AssertionError("non-CALVIN direct arm chart was accepted")


def test_v120_direct_codec_lift_and_head_keep_two_native_time_branches() -> None:
    config = CorePolicyConfig(
        hidden_size=32,
        num_heads=4,
        gripper_field_dim=6,
        gripper_output_mode="calvin_binary_command",
        arm_flow_mode="relative_command_direct",
    )
    codec = CorePhysicalActionCodec(config)
    state = torch.randn(2, 7)
    action = torch.randn(2, 24, 7)
    physical = codec.encode(action, state)
    torch.testing.assert_close(physical[..., :6], physical[..., 6:12])
    torch.testing.assert_close(codec.decode(physical, state), action)

    random_arm = torch.randn(2, 24, 12)
    native, projected, null = codec.project_arm_tangent(random_arm)
    torch.testing.assert_close(native, 0.75 * random_arm[..., :6] + 0.25 * random_arm[..., 6:])
    assert projected is random_arm
    torch.testing.assert_close(null, torch.zeros_like(null), atol=0.0, rtol=0.0)

    lift = CoreNativeTimePhysicalActionTokenLift(config).eval()
    changed = torch.zeros_like(physical)
    changed[:, 5, 6] = 1.0
    lifted_delta = lift(changed) - lift(torch.zeros_like(changed))
    outside = torch.ones(24, dtype=torch.bool)
    outside[5] = False
    torch.testing.assert_close(
        lifted_delta[:, outside],
        torch.zeros_like(lifted_delta[:, outside]),
        atol=0.0,
        rtol=0.0,
    )

    head = ActionOnlyPhysicalVelocityHead(config)
    tokens = torch.randn(2, 24, 32)
    velocity = head(tokens)
    assert tuple(velocity.shape) == (2, 24, 18)
    assert head.arm_direct
    assert head.arm_abs is not None and head.arm_delta is not None
    velocity[..., :12].square().mean().backward()
    assert head.arm_abs.weight.grad is not None
    assert head.arm_delta.weight.grad is not None


def test_calvin_direct_losses_exclude_acceleration_and_compatibility_gripper() -> None:
    config = _calvin_config()
    codec = _direct_codec()
    state = torch.zeros(1, 7)
    action = torch.zeros(1, 24, 7)
    action[..., 0] = 0.03
    action[..., -1] = 1.0
    target_physical = codec.encode(action, state)
    prediction = target_physical.clone()
    prediction[..., 12:] += 10.0
    command_logits = torch.stack(
        (torch.full((1, 24), -8.0), torch.full((1, 24), 8.0)),
        dim=-1,
    )
    target = ActionSupervision(
        normalized=action,
        raw_units=action,
        current_raw_units=state,
        gripper_transition_boundary=state,
        gripper_transition_boundary_raw_units=state,
    )
    history = cast(
        ObservableHistory,
        SimpleNamespace(
            action_state=state,
            codec_gripper_boundary=state[..., -1:],
        ),
    )
    zero = torch.zeros_like(target_physical)

    def terms_for(physical_velocity: torch.Tensor) -> dict[str, torch.Tensor]:
        output = cast(
            PolicyStepOutput,
            SimpleNamespace(
                bottom=SimpleNamespace(
                    physical_velocity=physical_velocity,
                    motion_logits=torch.zeros(1, 24),
                    gripper_command_logits=command_logits,
                    decoder_tensors={},
                )
            ),
        )
        return action_terms(
            config,
            codec,
            output,
            target,
            history,
            FlowMatchingState(
                time=torch.zeros(1),
                source_physical_noise=zero,
                noisy_physical=zero,
                target_physical=target_physical,
                target_physical_velocity=target_physical,
            ),
        )

    compatibility_only = terms_for(prediction)
    for name in (
        "action_flow",
        "decoded_action",
        "smooth_delta",
        "physical_delta_consistency",
        "gripper_trajectory",
    ):
        torch.testing.assert_close(
            compatibility_only[name],
            torch.tensor(0.0),
            atol=0.0,
            rtol=0.0,
        )
    assert compatibility_only["decoded_action_v120_comparable"] > 0

    branch_mismatch = prediction.clone()
    branch_mismatch[..., 6:12] += 0.5
    mismatched = terms_for(branch_mismatch)
    torch.testing.assert_close(
        mismatched["smooth_delta"],
        torch.tensor(0.0),
        atol=0.0,
        rtol=0.0,
    )
    assert mismatched["physical_delta_consistency"] > 0
    assert mismatched["action_flow"] > 0


def test_flow_matching_and_deployment_share_the_same_physical_field() -> None:
    codec = _codec()
    target = torch.randn(2, 24, 7)
    state = torch.randn(2, 7)
    generator = torch.Generator().manual_seed(17)
    flow = sample_flow_matching(
        target,
        action_state=state,
        codec=codec,
        distribution="v120_mirrored_beta_1_5_1",
        generator=generator,
    )
    assert tuple(flow.source_physical_noise.shape) == (2, 24, 18)
    assert tuple(flow.noisy_physical.shape) == (2, 24, 18)
    torch.testing.assert_close(flow.target_physical, codec.encode(target, state))
    torch.testing.assert_close(
        flow.target_physical_velocity,
        flow.target_physical - flow.source_physical_noise,
    )


def test_pen_legacy_and_calvin_direct_noise_keep_the_historical_rng_draw() -> None:
    seed = 20260904
    expected = torch.randn(2, 24, 18, generator=torch.Generator().manual_seed(seed))
    legacy = _codec().sample_noise(
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    )
    direct = _direct_codec().sample_noise(
        2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    )
    torch.testing.assert_close(legacy, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(direct, expected, atol=0.0, rtol=0.0)
    assert not torch.equal(direct[..., :6], direct[..., 6:12])


def _direct_codec() -> PhysicalActionFieldCodec:
    return PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
        arm_flow_mode="relative_command_direct",
    )


def _calvin_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            data_profile="calvin_relative_7d_v1",
            split_mode="episode-manifest",
            split_manifest="calvin-test-splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
        ),
        bottom=replace(
            base.bottom,
            arm_flow_mode="relative_command_direct",
            gripper_output_mode="calvin_binary_command",
        ),
        objectives=replace(
            base.objectives,
            gripper_command=0.1,
            gripper_event_threshold=0.1,
        ),
    )
    config.validate()
    return config


def test_binary_command_model_input_neutralizes_only_future_gripper_fields() -> None:
    codec = _codec()
    field = torch.randn(2, 24, codec.physical_dim, requires_grad=True)
    conditioned = codec.binary_command_model_input(field)
    arm_channels = 2 * codec.arm_dim

    torch.testing.assert_close(
        conditioned[..., :arm_channels],
        field[..., :arm_channels],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        conditioned[..., arm_channels:],
        torch.zeros_like(conditioned[..., arm_channels:]),
        atol=0.0,
        rtol=0.0,
    )
    conditioned.sum().backward()
    assert field.grad is not None
    torch.testing.assert_close(
        field.grad[..., :arm_channels],
        torch.ones_like(field.grad[..., :arm_channels]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        field.grad[..., arm_channels:],
        torch.zeros_like(field.grad[..., arm_channels:]),
        atol=0.0,
        rtol=0.0,
    )


def test_flow_time_is_the_exact_mirrored_v120_beta_with_owned_rng_order() -> None:
    batch = 8
    seed = 1234
    expected_generator = torch.Generator().manual_seed(seed)
    numerator = torch._standard_gamma(
        torch.full((batch,), 1.5, dtype=torch.float32),
        generator=expected_generator,
    )
    denominator = numerator + torch._standard_gamma(
        torch.ones(batch, dtype=torch.float32),
        generator=expected_generator,
    )
    expected_v120 = numerator / denominator.clamp_min(1e-8)
    expected_mainline = 1.0 - (expected_v120 * 0.999 + 0.001)

    flow = sample_flow_matching(
        torch.randn(batch, 24, 7),
        action_state=torch.randn(batch, 7),
        codec=_codec(),
        distribution="v120_mirrored_beta_1_5_1",
        generator=torch.Generator().manual_seed(seed),
    )
    torch.testing.assert_close(flow.time, expected_mainline)
    assert bool((flow.time >= 0.0).all())
    assert bool((flow.time <= 0.999).all())


def test_anchor_bands_restore_v120_per_row_pressure_and_unit_mean() -> None:
    weight = anchor_horizon_weights(
        horizon=24,
        tail_emphasis=0.20,
        first_step_protection=0.05,
        device=torch.device("cpu"),
    )
    expected = torch.tensor([1.05, 1.0, 1.0, 1.0] + [1.10] * 8 + [1.20] * 12)
    expected = expected / expected.mean()
    torch.testing.assert_close(weight, expected)
    torch.testing.assert_close(weight.mean(), torch.tensor(1.0))
    mass = torch.stack((weight[:4].sum(), weight[4:12].sum(), weight[12:].sum()))
    mass = mass / mass.sum()
    assert bool((mass[1:] > mass[:-1]).all())
    torch.testing.assert_close(
        mass,
        torch.tensor([4.05, 8.80, 14.40]) / 27.25,
    )


def test_event_row_balance_reaches_real_gripper_rows_without_changing_budget() -> None:
    horizon = anchor_horizon_weights(
        horizon=24,
        tail_emphasis=0.20,
        first_step_protection=0.05,
        device=torch.device("cpu"),
    )
    event = torch.zeros(2, 24)
    event[0, 5] = 1.0
    event[1, 19] = 1.0
    weight = balanced_event_row_weights(event, horizon)
    torch.testing.assert_close((weight * horizon[None]).mean(), torch.tensor(1.0))
    event_mean = weight[event.bool()].mean()
    hold_mean = weight[~event.bool()].mean()
    assert event_mean > hold_mean

    no_event = balanced_event_row_weights(torch.zeros_like(event), horizon)
    assert torch.equal(no_event, torch.ones_like(no_event))


def test_binary_command_balance_equalizes_effective_class_mass_and_budget() -> None:
    horizon = anchor_horizon_weights(
        horizon=24,
        tail_emphasis=0.20,
        first_step_protection=0.05,
        device=torch.device("cpu"),
    )
    target = torch.ones(2, 24, dtype=torch.long)
    target[0, :5] = 0
    target[1, 20:] = 0
    weight = balanced_binary_command_weights(target, horizon)
    step = horizon[None].expand_as(weight)
    positive_mass = (weight * step * (target == 1)).sum()
    negative_mass = (weight * step * (target == 0)).sum()

    torch.testing.assert_close(positive_mass, negative_mass)
    torch.testing.assert_close((weight * step).mean(), torch.tensor(1.0))
    assert weight[target == 0].mean() > weight[target == 1].mean()

    one_class = balanced_binary_command_weights(
        torch.ones_like(target),
        horizon,
    )
    torch.testing.assert_close(
        one_class,
        torch.ones_like(one_class),
        atol=0.0,
        rtol=0.0,
    )


def test_continuous_gripper_mask_starts_at_first_event_and_never_reopens() -> None:
    event = torch.zeros(3, 8)
    event[0, 2] = 1.0
    event[0, 6] = 1.0
    event[1, 0] = 1.0
    mask = causal_event_trajectory_mask(event)
    torch.testing.assert_close(
        mask,
        torch.tensor(
            (
                (0, 0, 1, 1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1, 1, 1),
                (0, 0, 0, 0, 0, 0, 0, 0),
            ),
            dtype=torch.float32,
        ),
    )


def test_gripper_transition_and_persistence_masks_are_disjoint_and_complete() -> None:
    event = torch.zeros(3, 8)
    event[0, 2] = 1.0
    event[0, 6] = 1.0
    event[1, 0] = 1.0
    transition, persistence = event_transition_persistence_masks(event)
    assert torch.count_nonzero(transition * persistence) == 0
    torch.testing.assert_close(
        transition + persistence,
        causal_event_trajectory_mask(event),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(transition[2]) == 0
    assert torch.count_nonzero(persistence[2]) == 0


def test_gripper_decode_branches_are_the_exact_deployed_operands() -> None:
    codec = _codec()
    field = torch.zeros(1, 24, codec.physical_dim)
    field[..., 12] = torch.arange(24, dtype=torch.float32)
    field[..., 13] = torch.arange(1, 25, dtype=torch.float32)
    action_state = torch.zeros(1, 7)
    action_state[..., -1] = 3.0
    absolute, cumulative_delta = codec.gripper_decode_branches(field, action_state)
    torch.testing.assert_close(absolute[..., 0], field[..., 12])
    torch.testing.assert_close(
        cumulative_delta[..., 0],
        3.0 + torch.cumsum(field[..., 13], dim=1),
    )
    decoded = codec.decode(field, action_state)
    torch.testing.assert_close(
        decoded[..., -1:],
        0.75 * absolute + 0.25 * cumulative_delta,
    )


def test_deployed_gripper_delta_branch_retains_pre_event_causal_vjp() -> None:
    codec = _codec()
    field = torch.zeros(1, 24, codec.physical_dim, requires_grad=True)
    action_state = torch.zeros(1, 7)
    event = torch.zeros(1, 24)
    event[:, 3] = 1.0
    _, persistence = event_transition_persistence_masks(event)
    _, cumulative_delta = codec.gripper_decode_branches(field, action_state)
    (cumulative_delta[..., 0] * persistence).sum().backward()
    assert field.grad is not None
    # Every post-event deployed value contains the earlier prefix. Removing
    # this VJP would train a target-only reanchoring operation absent at runtime.
    assert torch.count_nonzero(field.grad[:, :4, 13]) > 0
    assert torch.count_nonzero(field.grad[..., 12]) == 0
    assert torch.count_nonzero(field.grad[..., 14:]) == 0


def test_no_event_gripper_trajectory_masks_are_exact_zero() -> None:
    codec = _codec()
    field = torch.randn(2, 24, codec.physical_dim)
    action_state = torch.randn(2, 7)
    event = torch.zeros(2, 24)
    transition, persistence = event_transition_persistence_masks(event)
    _, cumulative_delta = codec.gripper_decode_branches(field, action_state)
    assert torch.count_nonzero(transition) == 0
    assert torch.count_nonzero(persistence) == 0
    assert (cumulative_delta[..., 0] * persistence).sum() == 0


def test_profile_owned_command_boundary_retargets_all_continuous_gripper_codec_paths() -> None:
    codec = _codec()
    action = torch.randn(1, 24, 7)
    action[..., -1] = torch.linspace(-0.8, 1.2, 24)
    current_qpos = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -2.0]])
    previous_command = torch.tensor([[50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 0.5]])
    command_boundary = previous_command[..., -1:]

    field = codec.encode(
        action,
        current_qpos,
        codec_gripper_boundary=command_boundary,
    )
    parts = codec.split(field)
    expected_arm_boundary = torch.cat(
        (current_qpos[:, None, :6], action[:, :-1, :6]), dim=1
    )
    expected_gripper_boundary = torch.cat(
        (command_boundary[:, None], action[:, :-1, -1:]), dim=1
    )
    expected_gripper_delta = action[..., -1:] - expected_gripper_boundary

    torch.testing.assert_close(
        parts.arm_delta,
        action[..., :6] - expected_arm_boundary,
    )
    torch.testing.assert_close(parts.gripper_field[..., :1], action[..., -1:])
    torch.testing.assert_close(
        parts.gripper_field[..., 1:2], expected_gripper_delta
    )
    torch.testing.assert_close(
        parts.gripper_field[..., 2:3],
        action[..., -1:] - command_boundary[:, None],
    )
    torch.testing.assert_close(
        parts.gripper_field[..., 3:4], expected_gripper_boundary
    )
    torch.testing.assert_close(
        parts.gripper_field[..., 4:5], expected_gripper_delta.abs()
    )
    torch.testing.assert_close(
        parts.gripper_field[..., 5:6], torch.relu(expected_gripper_delta)
    )
    decoded = codec.decode(
        field,
        current_qpos,
        codec_gripper_boundary=command_boundary,
    )
    torch.testing.assert_close(decoded, action)
    torch.testing.assert_close(
        codec.delta_consistency(
            field,
            current_qpos,
            action,
            codec_gripper_boundary=command_boundary,
        ),
        torch.zeros(1, 24),
        atol=0.0,
        rtol=0.0,
    )
    flow = sample_flow_matching(
        action,
        action_state=current_qpos,
        codec_gripper_boundary=command_boundary,
        codec=codec,
        distribution="v120_mirrored_beta_1_5_1",
        generator=torch.Generator().manual_seed(29),
    )
    torch.testing.assert_close(flow.target_physical, field, atol=0.0, rtol=0.0)


def test_pen_implicit_and_explicit_current_state_gripper_boundaries_are_bit_exact() -> None:
    codec = _codec()
    action = torch.randn(2, 24, 7)
    current_state = torch.randn(2, 7)
    implicit = codec.encode(action, current_state)
    explicit = codec.encode(
        action,
        current_state,
        codec_gripper_boundary=current_state[..., -1:],
    )

    torch.testing.assert_close(implicit, explicit, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        codec.decode(implicit, current_state),
        codec.decode(
            explicit,
            current_state,
            codec_gripper_boundary=current_state[..., -1:],
        ),
        atol=0.0,
        rtol=0.0,
    )


def test_gripper_persistence_does_not_absorb_a_pre_event_deployed_delta_error() -> None:
    codec = _codec()
    config = ExperimentConfig()
    action = torch.zeros(1, 24, 7)
    action[:, 2:, -1] = 1.0
    current_qpos = torch.zeros(1, 7)
    previous_command = torch.zeros(1, 7)
    target_physical = codec.encode(action, current_qpos)
    predicted_physical = target_physical.clone()
    # The anchored persistence lane starts at the first event.  A pre-event
    # deployed-delta error must therefore remain visible to the physical codec
    # consistency term instead of being absorbed by that persistence target.
    predicted_physical[:, 0, 13] += 0.5
    zero = torch.zeros_like(target_physical)
    output = cast(
        PolicyStepOutput,
        SimpleNamespace(
            bottom=SimpleNamespace(
                physical_velocity=predicted_physical,
                motion_logits=torch.zeros(1, 24),
                decoder_tensors={},
            )
        ),
    )
    target = ActionSupervision(
        normalized=action,
        raw_units=action,
        current_raw_units=current_qpos,
        gripper_transition_boundary=previous_command,
        gripper_transition_boundary_raw_units=previous_command,
    )
    terms = action_terms(
        config,
        codec,
        output,
        target,
        cast(
            ObservableHistory,
            SimpleNamespace(
                action_state=current_qpos,
                codec_gripper_boundary=previous_command[..., -1:],
            ),
        ),
        FlowMatchingState(
            time=torch.zeros(1),
            source_physical_noise=zero,
            noisy_physical=zero,
            target_physical=target_physical,
            target_physical_velocity=target_physical,
        ),
    )
    torch.testing.assert_close(
        terms["gripper_trajectory_transition"],
        torch.tensor(0.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        terms["gripper_trajectory_persistence"],
        torch.tensor(0.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        terms["gripper_trajectory"],
        torch.tensor(0.0),
        atol=0.0,
        rtol=0.0,
    )
    assert terms["physical_delta_consistency"] > 0


def test_horizon_action_condition_uses_deterministic_four_interval_projection() -> None:
    action = torch.arange(2 * 24 * 3, dtype=torch.float32).reshape(2, 24, 3)
    current = torch.tensor([[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]])
    condition = PhysicalActionCondition.from_horizon_action(action, current)
    expected = torch.stack(
        (
            action[:, 3:8].mean(dim=1),
            action[:, 7:16].mean(dim=1),
            action[:, 15:24].mean(dim=1),
            action[:, 23:24].mean(dim=1),
        ),
        dim=1,
    )
    torch.testing.assert_close(condition.interval_action, expected)
    condition.assert_exact_reconstruction()
    assert condition.action_dim == 3
    assert condition.fingerprint.shape == (2, 4, 6)
