from __future__ import annotations

import torch

from clearvla.mainline.model.action_codec import (
    PhysicalActionFieldCodec,
    anchor_horizon_weights,
)
from clearvla.mainline.training.losses import (
    balanced_event_row_weights,
    event_positive_class_weights,
    sample_flow_matching,
)


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


def test_v120_event_positive_boost_is_five_to_one_over_hold() -> None:
    weight = event_positive_class_weights(
        torch.tensor((0, 1, 2)),
        positive_boost=4.0,
    )
    torch.testing.assert_close(weight, torch.tensor((1.0, 5.0, 5.0)))
