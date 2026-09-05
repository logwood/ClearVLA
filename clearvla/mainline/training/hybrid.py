"""Losses on the actually integrated hybrid action, including hold rows."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ExperimentConfig
from ..interfaces import ActionSupervision, ObservableHistory
from ..model.action_codec import anchor_horizon_weights
from ..runtime.hybrid import HybridRolloutResult


def _weighted_masked_mean(value: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    owned_weight = weight * mask.to(dtype=weight.dtype)
    return (value * owned_weight).sum() / owned_weight.sum().clamp_min(1e-8)


def hybrid_rollout_terms(
    config: ExperimentConfig,
    rollout: HybridRolloutResult,
    target: ActionSupervision,
    history: ObservableHistory,
) -> dict[str, Tensor]:
    """Reuse the existing decoded/trajectory budgets on the refined rollout."""
    prediction = rollout.refined_action.float()
    truth = target.normalized.detach().float()
    if prediction.shape != truth.shape:
        raise ValueError("hybrid rollout action and target shapes differ")
    weight = anchor_horizon_weights(
        horizon=config.dimensions.action_horizon,
        tail_emphasis=config.objectives.horizon_tail_emphasis,
        first_step_protection=config.objectives.horizon_first_step_protection,
        device=prediction.device,
    )[None]
    error = F.smooth_l1_loss(prediction, truth, reduction="none")
    decoded = (error.mean(dim=-1) * weight).mean()
    arm = (error[..., :-1].mean(dim=-1) * weight).mean()
    gripper = (error[..., -1] * weight).mean()
    raw = target.raw_units[..., -1].detach().float()
    raw_previous = torch.cat(
        (
            target.gripper_transition_boundary_raw_units[:, -1:].detach().float(),
            raw[:, :-1],
        ),
        dim=1,
    )
    event = (raw - raw_previous).abs() >= config.objectives.gripper_event_threshold
    after = torch.cummax(event.long(), dim=1).values.bool()
    persistence = after & ~event
    hold = ~after
    boundary = history.codec_gripper_boundary.detach().float()
    pred_delta = prediction[..., -1] - torch.cat((boundary, prediction[:, :-1, -1]), dim=1)
    true_delta = truth[..., -1] - torch.cat((boundary, truth[:, :-1, -1]), dim=1)
    delta_error = F.smooth_l1_loss(pred_delta, true_delta, reduction="none")
    state_and_delta_error = 0.5 * (error[..., -1] + delta_error)
    strata = {
        "hybrid_rollout_hold": _weighted_masked_mean(state_and_delta_error, hold, weight),
        "hybrid_rollout_transition": _weighted_masked_mean(state_and_delta_error, event, weight),
        "hybrid_rollout_persistence": _weighted_masked_mean(
            state_and_delta_error, persistence, weight
        ),
    }
    # The three masks partition every row. Equal, explicitly recorded stratum
    # budgets retain no-event supervision instead of dropping all-zero events.
    trajectory = torch.stack(tuple(strata.values())).mean()
    total = (
        config.objectives.decoded_action * decoded
        + config.objectives.gripper_trajectory * trajectory
    )
    return {
        **strata,
        "hybrid_rollout": total,
        "hybrid_rollout_decoded_action": decoded,
        "hybrid_rollout_arm": arm,
        "hybrid_rollout_gripper": gripper,
        "hybrid_rollout_gripper_trajectory": trajectory,
        "hybrid_rollout_hold_row_fraction": hold.float().mean().detach(),
        "hybrid_rollout_event_row_fraction": event.float().mean().detach(),
        "hybrid_rollout_persistence_row_fraction": persistence.float().mean().detach(),
    }


__all__ = ["hybrid_rollout_terms"]
