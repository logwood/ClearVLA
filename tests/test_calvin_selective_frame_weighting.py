from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import ActionSupervision, ObservableHistory
from clearvla.mainline.model.action_codec import (
    PhysicalActionFieldCodec,
    anchor_horizon_weights,
)
from clearvla.mainline.model.components import OutletAdapter
from clearvla.mainline.model.policy import PolicyStepOutput
from clearvla.mainline.training.losses import (
    FlowMatchingState,
    action_terms,
    calvin_selective_frame_weights,
)


def _codec() -> PhysicalActionFieldCodec:
    return PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
    )


def _adapter(*, offset: torch.Tensor | None = None) -> OutletAdapter:
    adapter = OutletAdapter(_codec(), selection="calvin_7d_binary_v1")
    adapter.configure_action_normalizer(
        SimpleNamespace(
            offset=torch.zeros(1, 7) if offset is None else offset,
            scale=torch.ones(1, 7),
        )
    )
    return adapter


def _config(*, selective: bool) -> ExperimentConfig:
    base = ExperimentConfig()
    objectives = replace(
        base.objectives,
        gripper_command=0.1,
        gripper_event_threshold=0.1,
    )
    if selective:
        objectives = replace(
            objectives,
            calvin_frame_weight_mode="motion_event_v1",
            calvin_frame_motion_gain=0.75,
            calvin_frame_event_gain=1.5,
            calvin_frame_event_radius=1,
            calvin_frame_max_weight=3.0,
        )
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
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="calvin_binary_command",
        ),
        objectives=objectives,
    )
    config.validate()
    return config


def test_selective_weights_focus_sparse_motion_and_event_neighbourhood() -> None:
    config = _config(selective=True)
    offset = torch.tensor([[0.7, -0.3, 0.2, 0.4, -0.1, 0.5, 0.0]])
    adapter = _adapter(offset=offset)
    action = offset[:, None].expand(2, 24, -1).clone()
    action[0, 5, 0] += 0.04
    action[1, 10, 1] += 0.04
    event = torch.zeros(2, 24)
    event[:, 12] = 1.0
    horizon = anchor_horizon_weights(
        horizon=24,
        tail_emphasis=config.objectives.horizon_tail_emphasis,
        first_step_protection=config.objectives.horizon_first_step_protection,
        device=action.device,
    )
    weights = calvin_selective_frame_weights(
        config,
        adapter,
        action,
        torch.zeros(2, 7),
        event,
        horizon,
    )

    assert tuple(weights.shape) == (2, 24)
    assert torch.isfinite(weights).all()
    assert float(weights[0, 5]) > float(weights[0, 0])
    assert float(weights[1, 10]) > float(weights[1, 0])
    assert all(float(weights[0, row]) > float(weights[0, 0]) for row in (11, 12, 13))
    weighted_mean = (weights * horizon[None]).sum() / (horizon.sum() * 2.0)
    torch.testing.assert_close(weighted_mean, torch.tensor(1.0), atol=1e-6, rtol=0.0)
    assert float(weights.max()) <= config.objectives.calvin_frame_max_weight


def test_uniform_is_exact_and_selective_profile_rejects_non_calvin_codec() -> None:
    uniform_config = _config(selective=False)
    horizon = torch.ones(24)
    action = torch.randn(2, 24, 7)
    event = torch.zeros(2, 24)
    uniform = calvin_selective_frame_weights(
        uniform_config,
        _adapter(),
        action,
        torch.zeros(2, 7),
        event,
        horizon,
    )
    torch.testing.assert_close(uniform, torch.ones_like(uniform), atol=0.0, rtol=0.0)

    with pytest.raises(ValueError, match="CALVIN"):
        calvin_selective_frame_weights(
            _config(selective=True),
            _codec(),
            action,
            torch.zeros(2, 7),
            event,
            horizon,
        )


def test_selective_action_terms_reweight_rows_and_backpropagate() -> None:
    config = _config(selective=True)
    adapter = _adapter()
    state = torch.zeros(1, 7)
    action = torch.zeros(1, 24, 7)
    action[:, 5, 0] = 0.04
    action[:, 12:, -1] = 1.0
    target_physical = adapter.encode(action, state)
    target = ActionSupervision(
        normalized=action,
        raw_units=action,
        current_raw_units=state,
        gripper_transition_boundary=state,
        gripper_transition_boundary_raw_units=state,
    )
    prediction = (target_physical + 0.1 * torch.randn_like(target_physical)).detach()
    prediction.requires_grad_(True)
    command_logits = torch.zeros(1, 24, 2, requires_grad=True)
    output = cast(
        PolicyStepOutput,
        SimpleNamespace(
            bottom=SimpleNamespace(
                physical_velocity=prediction,
                motion_logits=torch.zeros(1, 24, requires_grad=True),
                gripper_command_logits=command_logits,
                decoder_tensors={},
            )
        ),
    )
    history = cast(
        ObservableHistory,
        SimpleNamespace(action_state=state, codec_gripper_boundary=state[..., -1:]),
    )
    zero = torch.zeros_like(target_physical)
    terms = action_terms(
        config,
        adapter,
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

    assert terms["action_frame_weight_mode_code"] == 1
    assert terms["action_frame_weight_selective_fraction"] > 0
    assert terms["action_flow"] != terms["action_flow_v120_comparable"]
    total = (
        terms["action_flow"]
        + 0.08 * terms["decoded_action"]
        + 0.1 * terms["gripper_command"]
    )
    total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert command_logits.grad is not None and torch.isfinite(command_logits.grad).all()
