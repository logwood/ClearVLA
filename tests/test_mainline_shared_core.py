from dataclasses import replace

import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.action_codec import anchor_horizon_weights
from clearvla.mainline.model.intent import StatelessObjectIntentOrganizer
from clearvla.mainline.training.losses import action_frame_weights


def test_event_motion_frame_weights_keep_all_rows_and_unit_mass() -> None:
    horizon = anchor_horizon_weights(
        horizon=24,
        tail_emphasis=0.20,
        first_step_protection=0.05,
        device=torch.device("cpu"),
    )
    event = torch.zeros(2, 24)
    event[0, 3] = 1.0
    motion = torch.zeros_like(event)
    motion[:, :4] = 1.0
    weight = action_frame_weights(
        event,
        motion,
        horizon,
        mode="event_motion_v1",
        event_gain=0.75,
        motion_gain=0.35,
    )
    assert torch.isfinite(weight).all()
    assert torch.all(weight > 0.0)
    weighted_mean = (weight * horizon[None]).mean()
    torch.testing.assert_close(weighted_mean, torch.ones(()), atol=1e-6, rtol=1e-6)
    assert weight[0, 3] > weight[1, 3]


def test_shared_core_candidate_config_enables_language_and_frame_modes() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        top=replace(base.top, language_conditioning_mode="task_anchor_v1"),
        objectives=replace(
            base.objectives,
            action_frame_weight_mode="event_motion_v1",
            action_frame_event_gain=0.75,
            action_frame_motion_gain=0.35,
        ),
    )
    config.validate()
    assert config.top.language_conditioning_mode == "task_anchor_v1"
    assert config.objectives.action_frame_weight_mode == "event_motion_v1"


def test_legacy_language_mode_does_not_register_an_inactive_owner() -> None:
    legacy = StatelessObjectIntentOrganizer(
        hidden=16,
        goal_dim=16,
        state_dim=7,
        action_dim=7,
        content_dim=8,
        route_dim=4,
        horizon=24,
        heads=4,
        language_conditioning_mode="goal_read",
    )
    assert legacy.language_task_anchor is None

    candidate = StatelessObjectIntentOrganizer(
        hidden=16,
        goal_dim=16,
        state_dim=7,
        action_dim=7,
        content_dim=8,
        route_dim=4,
        horizon=24,
        heads=4,
        language_conditioning_mode="task_anchor_v1",
    )
    assert candidate.language_task_anchor is not None
    torch.testing.assert_close(
        candidate.language_task_anchor.weight,
        torch.zeros_like(candidate.language_task_anchor.weight),
        atol=0.0,
        rtol=0.0,
    )
