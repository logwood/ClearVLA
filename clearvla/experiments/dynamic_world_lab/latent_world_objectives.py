from __future__ import annotations

"""Losses for the V34.1 uniform latent world model."""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .latent_world_model import LatentWorldModel


@dataclass(frozen=True)
class LatentWorldLossConfig:
    predictive_weight: float = 1.0
    root_predictive_weight: float = 0.10
    scene_predictive_weight: float = 0.25
    near_hold_predictive_weight: float = 0.03
    near_hold_action_distance: float = 0.01
    teacher_forced_weight: float = 0.10
    increment_weight: float = 0.35
    direction_weight: float = 0.15
    amplitude_weight: float = 0.05
    action_utility_weight: float = 0.30
    action_utility_margin: float = 0.003
    residual_direction_weight: float = 0.15
    local_effect_weight: float = 0.20
    local_effect_direction_weight: float = 0.15
    swap_rank_weight: float = 0.08
    swap_margin: float = 0.005
    state_path_weight: float = 0.18
    current_state_weight: float = 0.10
    local_motion_weight: float = 0.06
    view_descriptor_weight: float = 0.20
    predicted_inverse_action_weight: float = 0.15
    predicted_inverse_delta_weight: float = 0.08
    predicted_inverse_gripper_weight: float = 0.08
    target_inverse_action_weight: float = 0.05
    target_inverse_delta_weight: float = 0.03
    target_inverse_gripper_weight: float = 0.03
    action_probe_weight: float = 0.03
    perception_consistency_weight: float = 0.08
    representation_anchor_weight: float = 0.10
    variance_weight: float = 0.02
    covariance_weight: float = 0.002
    token_diversity_weight: float = 0.01
    embedding_std_target: float = 0.05
    gripper_transition_boost: float = 3.0
    gripper_transition_threshold: float = 0.10
    gripper_transition_radius: int = 1
    informative_action_distance: float = 0.05

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name.endswith("_weight") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            min(
                self.action_utility_margin,
                self.swap_margin,
                self.embedding_std_target,
                self.near_hold_action_distance,
                self.gripper_transition_threshold,
            )
            < 0
        ):
            raise ValueError("loss margins/thresholds must be non-negative")
        if self.gripper_transition_radius < 0:
            raise ValueError("gripper_transition_radius must be non-negative")


def _component_error(
    model: LatentWorldModel,
    pred: Tensor,
    target: Tensor,
    *,
    config: LatentWorldLossConfig,
    reduction: str = "mean",
) -> Tensor:
    pred_split = model.split_world(pred)
    target_split = model.split_world(target)
    root = F.smooth_l1_loss(
        pred_split["root"].float(), target_split["root"].float(), reduction="none"
    )
    scene = F.smooth_l1_loss(
        pred_split["scene"].float(), target_split["scene"].float(), reduction="none"
    )
    dynamic = F.smooth_l1_loss(
        pred_split["dynamic"].float(), target_split["dynamic"].float(), reduction="none"
    )
    if reduction == "none":
        root = root.mean(dim=(-1, -2))
        scene = scene.mean(dim=(-1, -2))
        dynamic = dynamic.mean(dim=(-1, -2))
        return (
            config.root_predictive_weight * root + config.scene_predictive_weight * scene + dynamic
        )
    return (
        config.root_predictive_weight * root.mean()
        + config.scene_predictive_weight * scene.mean()
        + dynamic.mean()
    )


def _legacy_error(
    model: LatentWorldModel,
    pred: Tensor,
    target: Tensor,
    *,
    scene_weight: float,
    reduction: str = "mean",
) -> Tensor:
    pred_split = model.split_world(pred)
    target_split = model.split_world(target)
    scene = F.smooth_l1_loss(
        pred_split["scene"].float(), target_split["scene"].float(), reduction="none"
    )
    dynamic = F.smooth_l1_loss(
        pred_split["dynamic"].float(), target_split["dynamic"].float(), reduction="none"
    )
    if reduction == "none":
        return dynamic.mean(dim=(-1, -2)) + scene_weight * scene.mean(dim=(-1, -2))
    return dynamic.mean() + scene_weight * scene.mean()


def _transition_mask(
    future_raw: Tensor,
    current_raw: Tensor,
    *,
    gripper_index: int,
    threshold: float,
    radius: int,
) -> Tensor:
    gripper = future_raw[..., gripper_index]
    boundary = torch.cat([current_raw[:, None, gripper_index], gripper[:, :-1]], dim=1)
    event = (gripper - boundary).abs() >= float(threshold)
    if radius > 0:
        expanded = event.clone()
        for shift in range(1, radius + 1):
            expanded[:, shift:] |= event[:, :-shift]
            expanded[:, :-shift] |= event[:, shift:]
        event = expanded
    return event


def _weighted_state_loss(
    pred: Tensor,
    target: Tensor,
    target_raw: Tensor,
    current_raw: Tensor,
    *,
    gripper_index: int,
    threshold: float,
    radius: int,
    boost: float,
) -> tuple[Tensor, Tensor]:
    per_dim = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    event = _transition_mask(
        target_raw,
        current_raw,
        gripper_index=gripper_index,
        threshold=threshold,
        radius=radius,
    )
    weight = torch.ones_like(per_dim)
    weight[..., gripper_index] = 1.0 + float(boost) * event.float()
    return (per_dim * weight).sum() / weight.sum().clamp_min(1.0), event


def _variance_loss(world: Tensor, target_std: float) -> tuple[Tensor, Tensor]:
    flat = world.float().reshape(-1, world.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=flat.shape[0] > 1) + 1e-4)
    return F.relu(float(target_std) - std).mean(), std.mean()


def _covariance_loss(world: Tensor) -> Tensor:
    flat = world.float().reshape(-1, world.shape[-1])
    if flat.shape[0] <= 1:
        return flat.new_zeros(())
    flat = flat - flat.mean(dim=0, keepdim=True)
    covariance = flat.T @ flat / float(flat.shape[0] - 1)
    off = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off.square().sum() / max(world.shape[-1], 1)


def _token_diversity(world: Tensor) -> Tensor:
    normalized = F.normalize(world.float(), dim=-1)
    similarity = normalized @ normalized.transpose(-1, -2)
    count = similarity.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool, device=world.device)
    return similarity[..., mask].square().mean()


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _gripper_classes(action_raw: Tensor, state_raw: Tensor, index: int, threshold: float) -> Tensor:
    boundary = torch.cat([state_raw[:, None, index], action_raw[:, :-1, index]], dim=1)
    delta = action_raw[..., index] - boundary
    classes = torch.zeros_like(delta, dtype=torch.long)
    classes[delta < -float(threshold)] = 1
    classes[delta > float(threshold)] = 2
    return classes


def _inverse_losses(
    prefix: str, output: dict[str, Tensor], relative: Tensor, delta: Tensor, classes: Tensor
):
    action = F.smooth_l1_loss(output[f"{prefix}_inverse_action"].float(), relative.float())
    action_delta = F.smooth_l1_loss(output[f"{prefix}_inverse_delta"].float(), delta.float())
    logits = output[f"{prefix}_inverse_gripper_logits"].float()
    gripper = F.cross_entropy(logits.reshape(-1, 3), classes.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == classes).float().mean()
    return action, action_delta, gripper, accuracy


def compute_latent_world_losses(
    model: LatentWorldModel,
    primary: dict[str, Tensor],
    output: dict[str, Tensor],
    *,
    config: LatentWorldLossConfig,
    action_scale: float = 1.0,
    stability_scale: float = 1.0,
    pair_output: dict[str, Tensor] | None = None,
    pair_valid: Tensor | None = None,
    swapped_output: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    config.validate()
    target = output["target_world"].detach()
    target_initial = output["target_initial_world"].detach()
    pred = output["pred_world"]
    hold = output["hold_world"]

    world_predictive = _component_error(model, pred, target, config=config)
    hold_predictive = _component_error(model, hold, target, config=config)
    legacy_full = _legacy_error(model, pred, target, scene_weight=config.scene_predictive_weight)
    legacy_hold = _legacy_error(model, hold, target, scene_weight=config.scene_predictive_weight)
    teacher_forced = _component_error(model, output["teacher_forced_world"], target, config=config)

    pred_sequence = torch.cat([output["initial_world"][:, None], pred], dim=1)
    target_sequence = torch.cat([target_initial[:, None], target], dim=1)
    pred_increment = pred_sequence[:, 1:] - pred_sequence[:, :-1]
    target_increment = target_sequence[:, 1:] - target_sequence[:, :-1]
    increment = F.smooth_l1_loss(pred_increment.float(), target_increment.float())
    increment_cosine_by_step = F.cosine_similarity(
        pred_increment.float().flatten(start_dim=2),
        target_increment.float().flatten(start_dim=2),
        dim=-1,
    )
    direction = 1.0 - increment_cosine_by_step.mean()
    pred_amp = pred_increment.float().square().mean(dim=(-1, -2)).sqrt()
    target_amp = target_increment.float().square().mean(dim=(-1, -2)).sqrt()
    amplitude = F.smooth_l1_loss(pred_amp, target_amp)

    full_per_step = _component_error(model, pred, target, config=config, reduction="none")
    hold_per_step = _component_error(model, hold, target, config=config, reduction="none")
    full_per = full_per_step.mean(dim=1)
    hold_per = hold_per_step.mean(dim=1)
    action_distance = (
        (primary["action"] - output["hold_action"]).float().square().mean(dim=(1, 2)).sqrt()
    )
    event = _transition_mask(
        primary["future_state_raw"],
        primary["state_raw"],
        gripper_index=model.config.gripper_index,
        threshold=config.gripper_transition_threshold,
        radius=config.gripper_transition_radius,
    )
    informative = (action_distance >= float(config.informative_action_distance)) | event.any(dim=1)
    # Hold is a diagnostic counterfactual, not a labelled target.  Detaching its
    # error prevents utility optimization from improving the score by damaging
    # the hold path.
    utility_per = F.relu(float(config.action_utility_margin) + full_per - hold_per.detach())
    action_utility = _masked_mean(utility_per, informative)
    full_vs_hold_gain = (hold_per - full_per).mean()
    informative_gain = _masked_mean(hold_per - full_per, informative)
    near_hold = action_distance <= float(config.near_hold_action_distance)
    near_hold_predictive = _masked_mean(hold_per, near_hold)

    pred_effect = output["action_world_effect"]
    target_motion = target - target_initial[:, None]
    residual_cosine_by_step = F.cosine_similarity(
        pred_effect.float().flatten(start_dim=2),
        target_motion.float().flatten(start_dim=2),
        dim=-1,
    )
    residual_cosine = _masked_mean(residual_cosine_by_step.mean(dim=1), informative)
    residual_direction = torch.where(
        informative.any(), 1.0 - residual_cosine, residual_cosine.new_zeros(())
    )

    state_path, _ = _weighted_state_loss(
        output["pred_state_path"],
        primary["future_state"],
        primary["future_state_raw"],
        primary["state_raw"],
        gripper_index=model.config.gripper_index,
        threshold=config.gripper_transition_threshold,
        radius=config.gripper_transition_radius,
        boost=config.gripper_transition_boost,
    )
    current_state = F.smooth_l1_loss(
        output["current_state_prediction"].float(), primary["state"].float()
    )
    local_motion_target = primary["history_state"][:, -1] - primary["history_state"][:, 0]
    local_motion = F.smooth_l1_loss(
        output["local_motion_prediction"].float(), local_motion_target.float()
    )

    view_current = F.smooth_l1_loss(
        output["current_view_prediction"].float(), output["current_view_target"].float()
    )
    view_future = F.smooth_l1_loss(
        output["pred_view_prediction"].float(), output["future_view_target"].float()
    )
    view_descriptor = view_current + view_future

    action_state = primary["action_state"]
    boundary = torch.cat([action_state[:, None], primary["action"][:, :-1]], dim=1)
    relative_action = primary["action"] - action_state[:, None]
    action_delta = primary["action"] - boundary
    gripper_classes = _gripper_classes(
        primary["action_raw"],
        primary["state_raw"],
        model.config.gripper_index,
        config.gripper_transition_threshold,
    )
    pred_inv_action, pred_inv_delta, pred_inv_gripper, pred_inv_accuracy = _inverse_losses(
        "pred", output, relative_action, action_delta, gripper_classes
    )
    target_inv_action, target_inv_delta, target_inv_gripper, target_inv_accuracy = _inverse_losses(
        "target", output, relative_action, action_delta, gripper_classes
    )

    action_probe = F.smooth_l1_loss(
        output["action_only_probe_prediction"].float(),
        output["action_only_probe_target"].float(),
    )

    representation_anchor = F.smooth_l1_loss(
        output["initial_world"].float(), target_initial.float()
    ) + F.smooth_l1_loss(output["online_future_world"].float(), target.float())
    perception_consistency = pred.new_zeros(())
    if "masked_initial_world" in output:
        perception_consistency = F.smooth_l1_loss(
            output["masked_initial_world"].float(), target_initial.float()
        )
    variance, embedding_std = _variance_loss(
        torch.cat([output["initial_world"], output["online_future_world"].flatten(0, 1)], dim=0),
        config.embedding_std_target,
    )
    covariance = _covariance_loss(output["initial_world"])
    token_diversity = _token_diversity(output["initial_world"])

    local_effect = pred.new_zeros(())
    local_effect_direction = pred.new_zeros(())
    local_effect_cosine = pred.new_zeros(())
    pair_fraction = pred.new_zeros(())
    if pair_output is not None and pair_valid is not None:
        valid = pair_valid.bool().reshape(-1)
        pair_fraction = valid.float().mean()
        predicted_transition = output["pred_world"] - output["initial_world"][:, None]
        pair_predicted_transition = (
            pair_output["pred_world"] - pair_output["initial_world"][:, None]
        )
        target_transition = (
            output["target_world"].detach() - output["target_initial_world"].detach()[:, None]
        )
        pair_target_transition = (
            pair_output["target_world"].detach()
            - pair_output["target_initial_world"].detach()[:, None]
        )
        predicted_difference = predicted_transition - pair_predicted_transition
        target_difference = target_transition - pair_target_transition
        per = F.smooth_l1_loss(
            predicted_difference.float(), target_difference.float(), reduction="none"
        ).mean(dim=(-1, -2, -3))
        cosine = F.cosine_similarity(
            predicted_difference.float().flatten(start_dim=1),
            target_difference.float().flatten(start_dim=1),
            dim=-1,
        )
        local_effect = _masked_mean(per, valid)
        local_effect_cosine = _masked_mean(cosine, valid)
        local_effect_direction = torch.where(
            valid.any(), 1.0 - local_effect_cosine, local_effect_cosine.new_zeros(())
        )

    swap_rank = pred.new_zeros(())
    swap_correct = pred.new_zeros(())
    swap_regret = pred.new_zeros(())
    if swapped_output is not None:
        swapped_per = _component_error(
            model, swapped_output["pred_world"], target, config=config, reduction="none"
        ).mean(dim=1)
        swap_rank = F.relu(float(config.swap_margin) + full_per - swapped_per).mean()
        swap_correct = (full_per < swapped_per).float().mean()
        swap_regret = (full_per - swapped_per).clamp_min(0).mean()

    total = (
        config.predictive_weight * world_predictive
        + config.near_hold_predictive_weight * near_hold_predictive
        + config.teacher_forced_weight * teacher_forced
        + config.increment_weight * increment
        + config.direction_weight * direction
        + config.amplitude_weight * amplitude
        + config.state_path_weight * state_path
        + config.current_state_weight * current_state
        + config.local_motion_weight * local_motion
        + config.view_descriptor_weight * view_descriptor
        + config.action_probe_weight * action_probe
        + float(action_scale)
        * (
            config.action_utility_weight * action_utility
            + config.residual_direction_weight * residual_direction
            + config.predicted_inverse_action_weight * pred_inv_action
            + config.predicted_inverse_delta_weight * pred_inv_delta
            + config.predicted_inverse_gripper_weight * pred_inv_gripper
            + config.target_inverse_action_weight * target_inv_action
            + config.target_inverse_delta_weight * target_inv_delta
            + config.target_inverse_gripper_weight * target_inv_gripper
            + config.local_effect_weight * local_effect
            + config.local_effect_direction_weight * local_effect_direction
            + config.swap_rank_weight * swap_rank
        )
        + float(stability_scale)
        * (
            config.perception_consistency_weight * perception_consistency
            + config.representation_anchor_weight * representation_anchor
            + config.variance_weight * variance
            + config.covariance_weight * covariance
            + config.token_diversity_weight * token_diversity
        )
    )

    result: dict[str, Tensor] = {
        "loss": total,
        "world_predictive": world_predictive,
        "legacy_full": legacy_full,
        "hold_predictive": hold_predictive,
        "legacy_hold": legacy_hold,
        "near_hold_predictive": near_hold_predictive,
        "near_hold_fraction": near_hold.float().mean(),
        "teacher_forced": teacher_forced,
        "closed_loop_gap": world_predictive - teacher_forced,
        "increment": increment,
        "direction": direction,
        "increment_cosine": increment_cosine_by_step.mean(),
        "amplitude": amplitude,
        "action_utility": action_utility,
        "full_vs_hold_gain": full_vs_hold_gain,
        "informative_full_vs_hold_gain": informative_gain,
        "residual_direction": residual_direction,
        "residual_cosine": residual_cosine,
        "state_path": state_path,
        "current_state": current_state,
        "local_motion": local_motion,
        "view_descriptor": view_descriptor,
        "view_descriptor_current": view_current,
        "view_descriptor_future": view_future,
        "pred_inverse_action": pred_inv_action,
        "pred_inverse_delta": pred_inv_delta,
        "pred_inverse_gripper": pred_inv_gripper,
        "pred_inverse_gripper_accuracy": pred_inv_accuracy,
        "target_inverse_action": target_inv_action,
        "target_inverse_delta": target_inv_delta,
        "target_inverse_gripper": target_inv_gripper,
        "target_inverse_gripper_accuracy": target_inv_accuracy,
        # Compatibility aliases point to the predicted-world inverse path.
        "inverse_action": pred_inv_action,
        "inverse_delta": pred_inv_delta,
        "inverse_gripper": pred_inv_gripper,
        "inverse_gripper_accuracy": pred_inv_accuracy,
        "action_only_probe": action_probe,
        "perception_consistency": perception_consistency,
        "representation_anchor": representation_anchor,
        "variance": variance,
        "embedding_std": embedding_std,
        "covariance": covariance,
        "token_diversity": token_diversity,
        "local_effect": local_effect,
        "local_effect_direction": local_effect_direction,
        "local_effect_cosine": local_effect_cosine,
        "local_pair_valid_fraction": pair_fraction,
        "swap_rank": swap_rank,
        "swap_correct_fraction": swap_correct,
        "swap_regret": swap_regret,
        "action_scale": pred.new_tensor(float(action_scale)),
        "stability_scale": pred.new_tensor(float(stability_scale)),
        "pred_world_rms": pred.float().square().mean().sqrt(),
        "target_world_rms": target.float().square().mean().sqrt(),
        "hold_world_rms": hold.float().square().mean().sqrt(),
        "action_effect_rms": pred_effect.float().square().mean().sqrt(),
        "dense_action_effect_rms": output["dense_action_world_effect"]
        .float()
        .square()
        .mean()
        .sqrt(),
    }
    for step, offset in enumerate(model.config.future_offsets):
        result[f"world_predictive_t{offset}"] = full_per_step[:, step].mean()
        result[f"hold_predictive_t{offset}"] = hold_per_step[:, step].mean()
        result[f"full_vs_hold_gain_t{offset}"] = (
            hold_per_step[:, step] - full_per_step[:, step]
        ).mean()
        result[f"residual_cosine_t{offset}"] = _masked_mean(
            residual_cosine_by_step[:, step], informative
        )
        result[f"increment_cosine_t{offset}"] = increment_cosine_by_step[:, step].mean()
    for key in (
        "adaln_gate_abs_mean",
        "adaln_scale_abs_mean",
        "adaln_shift_abs_mean",
        "action_read_joint_rms",
        "world_action_joint_rms",
        "action_signal_rms",
        "world_rms",
        "local_action_rms",
    ):
        if key in output:
            result[key] = output[key]
    return result


__all__ = ["LatentWorldLossConfig", "compute_latent_world_losses", "_legacy_error"]
