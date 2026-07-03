from __future__ import annotations

"""Objectives for dynamic predictive world modelling.

There is intentionally no instance-level batch InfoNCE.  Every supervision
term is either a real future target, a real cross-episode local pair, or a
within-sample closed-loop consistency term.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import DynamicPredictiveWorld


@dataclass(frozen=True)
class DynamicWorldLossConfig:
    predictive_weight: float = 1.0
    scene_predictive_weight: float = 0.25
    direction_weight: float = 0.25
    amplitude_weight: float = 0.10
    increment_weight: float = 0.50
    scene_increment_weight: float = 0.10
    teacher_forced_weight: float = 0.25
    scene_teacher_forced_weight: float = 0.10
    descriptor_weight: float = 0.50
    encoder_anchor_weight: float = 0.25
    state_path_weight: float = 0.10
    local_effect_weight: float = 0.25
    local_effect_direction_weight: float = 0.10
    swap_rank_weight: float = 0.0
    swap_margin: float = 0.02
    variance_weight: float = 0.02
    embedding_std_target: float = 0.05
    gripper_transition_boost: float = 3.0
    gripper_transition_threshold: float = 0.10
    gripper_transition_radius: int = 1

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name.endswith("_weight") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.swap_margin < 0 or self.embedding_std_target < 0:
            raise ValueError("swap_margin and embedding_std_target must be non-negative")
        if self.gripper_transition_threshold < 0 or self.gripper_transition_radius < 0:
            raise ValueError("invalid gripper transition configuration")


def _cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_flat = pred.float().reshape(*pred.shape[:-1], -1)
    target_flat = target.float().reshape(*target.shape[:-1], -1)
    return (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()


def _amplitude_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_norm = pred.float().flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-6)
    target_norm = target.float().flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-6)
    return (torch.log(pred_norm) - torch.log(target_norm)).abs().mean()


def _transition_mask(
    target_state_raw: Tensor, current_state_raw: Tensor, gripper_index: int, threshold: float, radius: int
) -> Tensor:
    gripper = target_state_raw[..., gripper_index]
    boundary = torch.cat([current_state_raw[:, None, gripper_index], gripper[:, :-1]], dim=1)
    event = (gripper - boundary).abs() >= float(threshold)
    if radius > 0:
        event_f = event.float()[:, None]
        event = F.max_pool1d(event_f, kernel_size=2 * radius + 1, stride=1, padding=radius)[:, 0] > 0
    return event


def _weighted_state_loss(
    pred: Tensor,
    target: Tensor,
    target_raw: Tensor,
    current_raw: Tensor,
    *,
    gripper_index: int,
    transition_threshold: float,
    transition_radius: int,
    transition_boost: float,
) -> tuple[Tensor, Tensor]:
    event = _transition_mask(
        target_raw, current_raw, gripper_index, transition_threshold, transition_radius
    )
    weight = 1.0 + float(transition_boost) * event.float()
    per_step = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    return (per_step * weight).sum() / weight.sum().clamp_min(1.0), event.float().mean()


def _variance_regularization(*embeddings: Tensor, target_std: float) -> tuple[Tensor, Tensor]:
    losses, stds = [], []
    for embedding in embeddings:
        flat = embedding.float().reshape(-1, embedding.shape[-1])
        std = torch.sqrt(flat.var(dim=0, unbiased=flat.shape[0] > 1) + 1e-4)
        losses.append(F.relu(float(target_std) - std).mean())
        stds.append(std.mean())
    return torch.stack(losses).mean(), torch.stack(stds).mean()


def _masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    valid = valid.to(device=value.device, dtype=value.dtype)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    return (value * valid).sum() / valid.expand_as(value).sum().clamp_min(1.0)


def compute_dynamic_world_losses(
    model: DynamicPredictiveWorld,
    primary: dict[str, Tensor],
    output: dict[str, Tensor],
    *,
    config: DynamicWorldLossConfig,
    pair: dict[str, Tensor] | None = None,
    pair_output: dict[str, Tensor] | None = None,
    pair_valid: Tensor | None = None,
    swapped_output: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    config.validate()
    pred = output["pred_dynamic"]
    target = output["target_dynamic"].detach()
    predictive = F.smooth_l1_loss(pred.float(), target.float())
    direction = _cosine_loss(pred, target)
    amplitude = _amplitude_loss(pred, target)

    pred_scene = output["pred_scene"]
    target_scene = output["target_scene"].detach()
    initial_scene = output["initial_scene"]
    target_initial_scene = output["target_initial_scene"].detach()
    pred_scene_delta = pred_scene - initial_scene[:, None]
    target_scene_delta = target_scene - target_initial_scene[:, None]
    scene_predictive = F.smooth_l1_loss(
        pred_scene_delta.float(), target_scene_delta.float()
    )

    target_initial = output["target_initial_dynamic"].detach()
    pred_sequence = torch.cat([output["initial_dynamic"][:, None], pred], dim=1)
    target_sequence = torch.cat([target_initial[:, None], target], dim=1)
    pred_increment = pred_sequence[:, 1:] - pred_sequence[:, :-1]
    target_increment = target_sequence[:, 1:] - target_sequence[:, :-1]
    increment = F.smooth_l1_loss(pred_increment.float(), target_increment.float())

    pred_scene_sequence = torch.cat([initial_scene[:, None], pred_scene], dim=1)
    target_scene_sequence = torch.cat([target_initial_scene[:, None], target_scene], dim=1)
    scene_increment = F.smooth_l1_loss(
        (pred_scene_sequence[:, 1:] - pred_scene_sequence[:, :-1]).float(),
        (target_scene_sequence[:, 1:] - target_scene_sequence[:, :-1]).float(),
    )

    teacher_forced = F.smooth_l1_loss(
        output["teacher_forced_dynamic"].float(), target.float()
    )
    scene_teacher_forced = F.smooth_l1_loss(
        (output["teacher_forced_scene"] - target_initial_scene[:, None]).float(),
        target_scene_delta.float(),
    )
    world_predictive = predictive + config.scene_predictive_weight * scene_predictive
    world_teacher_forced = teacher_forced + config.scene_predictive_weight * scene_teacher_forced
    descriptor_future = F.smooth_l1_loss(
        output["pred_descriptor"].float(), output["target_descriptor"].float()
    )
    descriptor_current = F.smooth_l1_loss(
        output["initial_descriptor"].float(), output["current_descriptor"].float()
    )
    descriptor = descriptor_future + 0.5 * descriptor_current

    # Encoder-side anchors are only valid while representation learning is active.
    # Predictor experiments load one frozen action-independent representation,
    # so they must not silently move their target space.
    encoder_anchor = pred.new_zeros(())
    online_future_dynamic = output["target_dynamic"].detach()
    if config.encoder_anchor_weight > 0:
        if model.representation_frozen:
            raise ValueError("encoder_anchor_weight must be 0 for a frozen representation")
        _, online_future_dynamic = model.encode_online_future(primary["target_tokens"])
        online_future_descriptor = model.descriptor_prediction(online_future_dynamic)
        encoder_anchor = F.smooth_l1_loss(
            online_future_descriptor.float(), output["target_descriptor"].float()
        )

    state_path, transition_fraction = _weighted_state_loss(
        output["pred_state_path"],
        primary["future_state"],
        primary["future_state_raw"],
        primary["state_raw"],
        gripper_index=model.config.gripper_index,
        transition_threshold=config.gripper_transition_threshold,
        transition_radius=config.gripper_transition_radius,
        transition_boost=config.gripper_transition_boost,
    )

    variance, embedding_std = _variance_regularization(
        output["initial_dynamic"], online_future_dynamic,
        target_std=config.embedding_std_target,
    )
    if model.representation_frozen and config.variance_weight > 0:
        raise ValueError("variance_weight must be 0 for a frozen representation")

    local_effect = pred.new_zeros(())
    local_effect_direction = pred.new_zeros(())
    local_effect_cosine = pred.new_zeros(())
    swap_rank = pred.new_zeros(())
    swap_regret = pred.new_zeros(())
    swap_correct_fraction = pred.new_zeros(())
    valid_fraction = pred.new_zeros(())
    if pair_output is not None and pair_valid is not None:
        valid = pair_valid.to(device=pred.device, dtype=torch.bool)
        valid_fraction = valid.float().mean()
        pair_pred = pair_output["pred_dynamic"]
        pair_target = pair_output["target_dynamic"].detach()
        pred_effect = pred - pair_pred
        target_effect = target - pair_target
        dynamic_local_per = F.smooth_l1_loss(
            pred_effect.float(), target_effect.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))

        pair_pred_scene_delta = (
            pair_output["pred_scene"] - pair_output["initial_scene"][:, None]
        )
        pair_target_scene_delta = (
            pair_output["target_scene"].detach()
            - pair_output["target_initial_scene"].detach()[:, None]
        )
        pred_scene_effect = pred_scene_delta - pair_pred_scene_delta
        target_scene_effect = target_scene_delta - pair_target_scene_delta
        scene_local_per = F.smooth_l1_loss(
            pred_scene_effect.float(), target_scene_effect.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        local_per = dynamic_local_per + config.scene_predictive_weight * scene_local_per
        local_effect = _masked_mean(local_per, valid)

        # Direction uses one shared local-effect vector. sqrt(weight) keeps its
        # squared contribution aligned with the weighted prediction metric.
        scene_scale = float(config.scene_predictive_weight) ** 0.5
        pred_effect_vector = torch.cat(
            [
                pred_effect.float().flatten(start_dim=2),
                scene_scale * pred_scene_effect.float().flatten(start_dim=2),
            ],
            dim=-1,
        )
        target_effect_vector = torch.cat(
            [
                target_effect.float().flatten(start_dim=2),
                scene_scale * target_scene_effect.float().flatten(start_dim=2),
            ],
            dim=-1,
        )
        effect_cosine_per = F.cosine_similarity(
            pred_effect_vector, target_effect_vector, dim=-1
        ).mean(dim=1)
        local_effect_cosine = _masked_mean(effect_cosine_per, valid)
        local_effect_direction = _masked_mean(1.0 - effect_cosine_per, valid)

        if swapped_output is not None:
            correct_dynamic_error = F.smooth_l1_loss(
                pred.float(), target.float(), reduction="none"
            ).mean(dim=(-1, -2, -3))
            correct_scene_error = F.smooth_l1_loss(
                pred_scene_delta.float(), target_scene_delta.float(), reduction="none"
            ).mean(dim=(-1, -2, -3))
            swapped_dynamic_error = F.smooth_l1_loss(
                swapped_output["pred_dynamic"].float(), target.float(), reduction="none"
            ).mean(dim=(-1, -2, -3))
            swapped_scene_delta = swapped_output["pred_scene"] - initial_scene[:, None]
            swapped_scene_error = F.smooth_l1_loss(
                swapped_scene_delta.float(), target_scene_delta.float(), reduction="none"
            ).mean(dim=(-1, -2, -3))
            correct_error = correct_dynamic_error + config.scene_predictive_weight * correct_scene_error
            swapped_error = swapped_dynamic_error + config.scene_predictive_weight * swapped_scene_error
            regret = swapped_error - correct_error
            swap_regret = _masked_mean(regret, valid)
            swap_correct_fraction = _masked_mean((regret > 0).float(), valid)
            swap_rank = _masked_mean(F.relu(float(config.swap_margin) - regret), valid)

    total = (
        config.predictive_weight * predictive
        + config.scene_predictive_weight * scene_predictive
        + config.direction_weight * direction
        + config.amplitude_weight * amplitude
        + config.increment_weight * increment
        + config.scene_increment_weight * scene_increment
        + config.teacher_forced_weight * teacher_forced
        + config.scene_teacher_forced_weight * scene_teacher_forced
        + config.descriptor_weight * descriptor
        + config.encoder_anchor_weight * encoder_anchor
        + config.state_path_weight * state_path
        + config.local_effect_weight * local_effect
        + config.local_effect_direction_weight * local_effect_direction
        + config.swap_rank_weight * swap_rank
        + config.variance_weight * variance
    )
    gate = output["effect_gate"]
    scene_gate = output["scene_effect_gate"]

    dynamic_predictive_by_step = F.smooth_l1_loss(
        pred.float(), target.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    scene_predictive_by_step = F.smooth_l1_loss(
        pred_scene_delta.float(), target_scene_delta.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    dynamic_teacher_by_step = F.smooth_l1_loss(
        output["teacher_forced_dynamic"].float(), target.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    scene_teacher_by_step = F.smooth_l1_loss(
        (output["teacher_forced_scene"] - target_initial_scene[:, None]).float(),
        target_scene_delta.float(),
        reduction="none",
    ).mean(dim=(0, 2, 3))
    world_predictive_by_step = (
        dynamic_predictive_by_step
        + config.scene_predictive_weight * scene_predictive_by_step
    )
    world_teacher_by_step = (
        dynamic_teacher_by_step
        + config.scene_predictive_weight * scene_teacher_by_step
    )
    by_step_metrics: dict[str, Tensor] = {}
    for index, offset in enumerate(model.config.future_offsets):
        by_step_metrics[f"world_predictive_t{offset}"] = world_predictive_by_step[index]
        by_step_metrics[f"world_teacher_forced_t{offset}"] = world_teacher_by_step[index]
        by_step_metrics[f"closed_loop_gap_t{offset}"] = (
            world_predictive_by_step[index] - world_teacher_by_step[index]
        )
        by_step_metrics[f"closed_loop_ratio_t{offset}"] = (
            world_predictive_by_step[index] / world_teacher_by_step[index].clamp_min(1e-8)
        )
        by_step_metrics[f"pred_dynamic_rms_t{offset}"] = (
            pred[:, index].float().square().mean().sqrt()
        )
        by_step_metrics[f"target_dynamic_rms_t{offset}"] = (
            target[:, index].float().square().mean().sqrt()
        )
    max_closed_loop_gap = torch.stack(
        [value for key, value in by_step_metrics.items() if key.startswith("closed_loop_gap_t")]
    ).max()

    return {
        "loss": total,
        "predictive": predictive,
        "scene_predictive": scene_predictive,
        "world_predictive": world_predictive,
        "direction": direction,
        "amplitude": amplitude,
        "increment": increment,
        "scene_increment": scene_increment,
        "teacher_forced": teacher_forced,
        "scene_teacher_forced": scene_teacher_forced,
        "world_teacher_forced": world_teacher_forced,
        "closed_loop_gap": world_predictive.detach() - world_teacher_forced.detach(),
        "max_closed_loop_gap": max_closed_loop_gap.detach(),
        "descriptor": descriptor,
        "descriptor_future": descriptor_future,
        "descriptor_current": descriptor_current,
        "encoder_anchor": encoder_anchor,
        "state_path": state_path,
        "transition_fraction": transition_fraction,
        "variance": variance,
        "embedding_std": embedding_std,
        "local_effect": local_effect,
        "local_effect_direction": local_effect_direction,
        "local_effect_cosine": local_effect_cosine,
        "local_pair_valid_fraction": valid_fraction,
        "swap_rank": swap_rank,
        "swap_regret": swap_regret,
        "swap_correct_fraction": swap_correct_fraction,
        "effect_gate_mean": gate.mean(),
        "effect_gate_std": gate.float().std(unbiased=False),
        "scene_effect_gate_mean": scene_gate.mean(),
        "scene_effect_gate_std": scene_gate.float().std(unbiased=False),
        "pred_dynamic_rms": pred.float().square().mean().sqrt(),
        "pred_scene_delta_rms": pred_scene_delta.float().square().mean().sqrt(),
        "target_scene_delta_rms": target_scene_delta.float().square().mean().sqrt(),
        "target_dynamic_rms": target.float().square().mean().sqrt(),
        **by_step_metrics,
    }


__all__ = ["DynamicWorldLossConfig", "compute_dynamic_world_losses"]
