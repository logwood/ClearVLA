from __future__ import annotations

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
    anchored_gripper_persistence,
    balanced_event_row_weights,
    causal_event_trajectory_mask,
    event_transition_persistence_masks,
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


def test_gripper_persistence_reanchors_at_every_open_or_close_event() -> None:
    absolute = torch.tensor(
        [[[10.0], [11.0], [12.0], [13.0], [14.0], [20.0], [21.0]]]
    )
    local_delta = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]]
    )
    event = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    reconstructed = anchored_gripper_persistence(
        absolute,
        local_delta,
        event,
    )
    torch.testing.assert_close(
        reconstructed[..., 0],
        torch.tensor([[0.0, 0.0, 12.0, 16.0, 21.0, 20.0, 27.0]]),
        atol=0.0,
        rtol=0.0,
    )


def test_gripper_persistence_has_exact_zero_pre_event_delta_vjp() -> None:
    absolute = torch.randn(1, 8, 1, requires_grad=True)
    local_delta = torch.randn(1, 8, 1, requires_grad=True)
    event = torch.zeros(1, 8)
    event[:, 3] = 1.0
    _, persistence = event_transition_persistence_masks(event)
    reconstructed = anchored_gripper_persistence(
        absolute,
        local_delta,
        event,
    )
    (reconstructed[..., 0] * persistence).sum().backward()
    assert local_delta.grad is not None
    assert absolute.grad is not None
    assert torch.count_nonzero(local_delta.grad[:, :4]) == 0
    assert torch.count_nonzero(local_delta.grad[:, 4:]) > 0
    assert torch.count_nonzero(absolute.grad[:, :3]) == 0
    assert torch.count_nonzero(absolute.grad[:, 3]) > 0
    assert torch.count_nonzero(absolute.grad[:, 4:]) == 0


def test_no_event_gripper_persistence_is_exact_zero() -> None:
    absolute = torch.randn(2, 8, 1)
    local_delta = torch.randn(2, 8, 1)
    event = torch.zeros(2, 8)
    transition, persistence = event_transition_persistence_masks(event)
    reconstructed = anchored_gripper_persistence(
        absolute,
        local_delta,
        event,
    )
    assert torch.count_nonzero(transition) == 0
    assert torch.count_nonzero(persistence) == 0
    assert torch.count_nonzero(reconstructed) == 0


def test_command_event_boundary_does_not_retarget_the_qpos_anchored_codec_delta() -> None:
    codec = _codec()
    config = ExperimentConfig()
    action = torch.zeros(1, 24, 7)
    action[..., -1] = 1.0
    current_qpos = torch.zeros(1, 7)
    previous_command = torch.zeros(1, 7)
    previous_command[..., -1] = 0.5
    target_physical = codec.encode(action, current_qpos)
    zero = torch.zeros_like(target_physical)
    output = cast(
        PolicyStepOutput,
        SimpleNamespace(
            bottom=SimpleNamespace(
                physical_velocity=target_physical,
                motion_logits=torch.zeros(1, 24),
                decoder_tensors={},
            )
        )
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
        cast(ObservableHistory, SimpleNamespace(action_state=current_qpos)),
        FlowMatchingState(
            time=torch.zeros(1),
            source_physical_noise=zero,
            noisy_physical=zero,
            target_physical=target_physical,
            target_physical_velocity=target_physical,
        ),
    )

    torch.testing.assert_close(
        terms["action_gripper_event_rate"], torch.tensor(1.0 / 24.0)
    )
    for name in (
        "action_flow",
        "decoded_action",
        "smooth_delta",
        "physical_delta_consistency",
        "gripper_trajectory",
    ):
        torch.testing.assert_close(terms[name], torch.tensor(0.0), atol=0.0, rtol=0.0)


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
