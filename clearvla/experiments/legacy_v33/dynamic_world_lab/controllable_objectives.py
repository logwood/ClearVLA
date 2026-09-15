from __future__ import annotations

"""Objectives for V33.6 controllable action/world coupling."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .controllable_model import ControllableDynamicWorld


@dataclass(frozen=True)
class ControllableWorldLossConfig:
    predictive_weight: float = 1.0
    scene_predictive_weight: float = 0.25
    direction_weight: float = 0.25
    amplitude_weight: float = 0.10
    increment_weight: float = 0.50
    scene_increment_weight: float = 0.10
    teacher_forced_weight: float = 0.20
    descriptor_weight: float = 0.40
    state_path_weight: float = 0.10
    prior_state_path_weight: float = 0.05

    residual_weight: float = 1.0
    residual_direction_weight: float = 0.25
    necessity_weight: float = 0.25
    necessity_margin: float = 0.005
    informative_residual_threshold: float = 0.02

    inverse_action_weight: float = 0.20
    inverse_delta_weight: float = 0.10
    inverse_gripper_weight: float = 0.10

    local_effect_weight: float = 0.25
    local_effect_direction_weight: float = 0.15
    swap_rank_weight: float = 0.05
    swap_margin: float = 0.01

    representation_anchor_weight: float = 0.20
    adapter_delta_weight: float = 0.005
    variance_weight: float = 0.02
    embedding_std_target: float = 0.05

    gripper_transition_boost: float = 3.0
    gripper_transition_threshold: float = 0.10
    gripper_transition_radius: int = 1

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name.endswith("_weight") and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.necessity_margin < 0 or self.swap_margin < 0:
            raise ValueError("margins must be non-negative")
        if self.informative_residual_threshold < 0:
            raise ValueError("informative_residual_threshold must be non-negative")
        if self.embedding_std_target <= 0:
            raise ValueError("embedding_std_target must be positive")
        if self.gripper_transition_radius < 0:
            raise ValueError("gripper_transition_radius must be non-negative")


def _cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_flat = pred.float().flatten(start_dim=-2)
    target_flat = target.float().flatten(start_dim=-2)
    return (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1)).mean()


def _amplitude_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred_norm = pred.float().flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-8)
    target_norm = target.float().flatten(start_dim=-2).norm(dim=-1).clamp_min(1e-8)
    return (torch.log(pred_norm) - torch.log(target_norm)).abs().mean()


def _transition_mask(
    target_state_raw: Tensor,
    current_state_raw: Tensor,
    gripper_index: int,
    threshold: float,
    radius: int,
) -> Tensor:
    gripper = target_state_raw[..., gripper_index]
    boundary = torch.cat([current_state_raw[:, None, gripper_index], gripper[:, :-1]], dim=1)
    event = (gripper - boundary).abs() >= float(threshold)
    if radius > 0:
        event = (
            F.max_pool1d(
                event.float()[:, None], kernel_size=2 * radius + 1, stride=1, padding=radius
            )[:, 0]
            > 0
        )
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
    event = _transition_mask(target_raw, current_raw, gripper_index, threshold, radius)
    weight = 1.0 + float(boost) * event.float()
    per_step = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
    return (per_step * weight).sum() / weight.sum().clamp_min(1.0), event


def _masked_mean(value: Tensor, valid: Tensor) -> Tensor:
    valid = valid.to(device=value.device, dtype=value.dtype)
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    return (value * valid).sum() / valid.expand_as(value).sum().clamp_min(1.0)


def _embedding_variance(*embeddings: Tensor, target_std: float) -> tuple[Tensor, Tensor]:
    losses, stds = [], []
    for embedding in embeddings:
        flat = embedding.float().reshape(-1, embedding.shape[-1])
        std = torch.sqrt(flat.var(dim=0, unbiased=flat.shape[0] > 1) + 1e-4)
        losses.append(F.relu(float(target_std) - std).mean())
        stds.append(std.mean())
    return torch.stack(losses).mean(), torch.stack(stds).mean()


def _world_error_per_sample(
    pred_dynamic: Tensor,
    pred_scene: Tensor,
    target_dynamic: Tensor,
    target_scene: Tensor,
    *,
    scene_weight: float,
) -> Tensor:
    dynamic = F.smooth_l1_loss(pred_dynamic.float(), target_dynamic.float(), reduction="none").mean(
        dim=(-1, -2, -3)
    )
    scene = F.smooth_l1_loss(pred_scene.float(), target_scene.float(), reduction="none").mean(
        dim=(-1, -2, -3)
    )
    return dynamic + float(scene_weight) * scene


def _gripper_classes(action_raw: Tensor, state_raw: Tensor, index: int, threshold: float) -> Tensor:
    gripper = action_raw[..., index]
    boundary = torch.cat([state_raw[:, None, index], gripper[:, :-1]], dim=1)
    delta = gripper - boundary
    # 0 hold, 1 open/decrease, 2 close/increase.
    out = torch.zeros_like(delta, dtype=torch.long)
    out[delta <= -float(threshold)] = 1
    out[delta >= float(threshold)] = 2
    return out


def compute_controllable_world_losses(
    model: ControllableDynamicWorld,
    primary: dict[str, Tensor],
    output: dict[str, Tensor],
    *,
    config: ControllableWorldLossConfig,
    phase: str,
    pair_output: dict[str, Tensor] | None = None,
    pair_valid: Tensor | None = None,
    swapped_output: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    config.validate()
    if phase not in {"prior", "effect", "align", "eval"}:
        raise ValueError(f"unsupported phase={phase!r}")

    target_dynamic = output["target_dynamic"].detach()
    target_scene = output["target_scene"].detach()
    pred_dynamic = output["pred_dynamic"]
    pred_scene = output["pred_scene"]
    prior_dynamic = output["prior_pred_dynamic"]
    prior_scene = output["prior_pred_scene"]

    predictive = F.smooth_l1_loss(pred_dynamic.float(), target_dynamic.float())
    scene_predictive = F.smooth_l1_loss(pred_scene.float(), target_scene.float())
    world_predictive = predictive + config.scene_predictive_weight * scene_predictive
    prior_predictive = F.smooth_l1_loss(prior_dynamic.float(), target_dynamic.float())
    prior_scene_predictive = F.smooth_l1_loss(prior_scene.float(), target_scene.float())
    prior_world_predictive = (
        prior_predictive + config.scene_predictive_weight * prior_scene_predictive
    )
    direction = _cosine_loss(pred_dynamic, target_dynamic)
    amplitude = _amplitude_loss(pred_dynamic, target_dynamic)

    initial_dynamic = output["initial_dynamic"]
    initial_scene = output["initial_scene"]
    target_initial_dynamic = output["target_initial_dynamic"].detach()
    target_initial_scene = output["target_initial_scene"].detach()
    pred_sequence = torch.cat([initial_dynamic[:, None], pred_dynamic], dim=1)
    target_sequence = torch.cat([target_initial_dynamic[:, None], target_dynamic], dim=1)
    increment = F.smooth_l1_loss(
        (pred_sequence[:, 1:] - pred_sequence[:, :-1]).float(),
        (target_sequence[:, 1:] - target_sequence[:, :-1]).float(),
    )
    pred_scene_sequence = torch.cat([initial_scene[:, None], pred_scene], dim=1)
    target_scene_sequence = torch.cat([target_initial_scene[:, None], target_scene], dim=1)
    scene_increment = F.smooth_l1_loss(
        (pred_scene_sequence[:, 1:] - pred_scene_sequence[:, :-1]).float(),
        (target_scene_sequence[:, 1:] - target_scene_sequence[:, :-1]).float(),
    )

    teacher_dynamic = output["teacher_forced_dynamic"]
    teacher_scene = output["teacher_forced_scene"]
    teacher_forced = F.smooth_l1_loss(teacher_dynamic.float(), target_dynamic.float())
    teacher_scene_loss = F.smooth_l1_loss(teacher_scene.float(), target_scene.float())
    world_teacher_forced = teacher_forced + config.scene_predictive_weight * teacher_scene_loss

    descriptor_future = F.smooth_l1_loss(
        output["pred_descriptor"].float(), output["target_descriptor"].float()
    )
    descriptor_current = F.smooth_l1_loss(
        output["initial_descriptor"].float(), output["current_descriptor"].float()
    )
    descriptor = descriptor_future + 0.5 * descriptor_current

    state_path, event = _weighted_state_loss(
        output["pred_state_path"],
        primary["future_state"],
        primary["future_state_raw"],
        primary["state_raw"],
        gripper_index=model.config.gripper_index,
        threshold=config.gripper_transition_threshold,
        radius=config.gripper_transition_radius,
        boost=config.gripper_transition_boost,
    )
    prior_state_path, _ = _weighted_state_loss(
        output["prior_state_path"],
        primary["future_state"],
        primary["future_state_raw"],
        primary["state_raw"],
        gripper_index=model.config.gripper_index,
        threshold=config.gripper_transition_threshold,
        radius=config.gripper_transition_radius,
        boost=config.gripper_transition_boost,
    )

    # Directly supervise the action-induced residual relative to a detached,
    # action-free prior.  The prior cannot expand during effect/alignment phases.
    prior_dynamic_detached = prior_dynamic.detach()
    prior_scene_detached = prior_scene.detach()
    predicted_dynamic_residual = pred_dynamic - prior_dynamic_detached
    target_dynamic_residual = target_dynamic - prior_dynamic_detached
    predicted_scene_residual = pred_scene - prior_scene_detached
    target_scene_residual = target_scene - prior_scene_detached
    residual_dynamic = F.smooth_l1_loss(
        predicted_dynamic_residual.float(), target_dynamic_residual.float()
    )
    residual_scene = F.smooth_l1_loss(
        predicted_scene_residual.float(), target_scene_residual.float()
    )
    residual = residual_dynamic + config.scene_predictive_weight * residual_scene
    residual_vector = torch.cat(
        [
            predicted_dynamic_residual.float().flatten(start_dim=2),
            config.scene_predictive_weight**0.5
            * predicted_scene_residual.float().flatten(start_dim=2),
        ],
        dim=-1,
    )
    target_residual_vector = torch.cat(
        [
            target_dynamic_residual.float().flatten(start_dim=2),
            config.scene_predictive_weight**0.5
            * target_scene_residual.float().flatten(start_dim=2),
        ],
        dim=-1,
    )
    residual_cosine_by_step = F.cosine_similarity(residual_vector, target_residual_vector, dim=-1)
    residual_cosine = residual_cosine_by_step.mean()
    residual_direction = 1.0 - residual_cosine

    full_error_per = _world_error_per_sample(
        pred_dynamic,
        pred_scene,
        target_dynamic,
        target_scene,
        scene_weight=config.scene_predictive_weight,
    )
    prior_error_per = _world_error_per_sample(
        prior_dynamic_detached,
        prior_scene_detached,
        target_dynamic,
        target_scene,
        scene_weight=config.scene_predictive_weight,
    )
    residual_strength = target_residual_vector.square().mean(dim=-1).sqrt().mean(dim=1)
    informative = (residual_strength >= float(config.informative_residual_threshold)) | event.any(
        dim=1
    )
    necessity_per = F.relu(float(config.necessity_margin) + full_error_per - prior_error_per)
    necessity = _masked_mean(necessity_per, informative)
    full_vs_prior_gain = (prior_error_per - full_error_per).mean()
    informative_gain = _masked_mean(prior_error_per - full_error_per, informative)

    action_state = primary.get("action_state", primary["state"])
    boundary = torch.cat([action_state[:, None], primary["action"][:, :-1]], dim=1)
    action_delta = primary["action"] - boundary
    relative_action = primary["action"] - action_state[:, None]
    inverse_action = F.smooth_l1_loss(output["inverse_action"].float(), relative_action.float())
    inverse_delta = F.smooth_l1_loss(output["inverse_delta"].float(), action_delta.float())
    gripper_class = _gripper_classes(
        primary["action_raw"],
        primary["state_raw"],
        model.config.gripper_index,
        config.gripper_transition_threshold,
    )
    inverse_gripper = F.cross_entropy(
        output["inverse_gripper_logits"].float().reshape(-1, 3), gripper_class.reshape(-1)
    )
    inverse_gripper_accuracy = (
        (output["inverse_gripper_logits"].argmax(dim=-1) == gripper_class).float().mean()
    )

    current_anchor = F.smooth_l1_loss(
        torch.cat([output["initial_scene"], output["initial_dynamic"]], dim=1).float(),
        torch.cat([output["base_scene"], output["base_dynamic"]], dim=1).detach().float(),
    )
    future_anchor = F.smooth_l1_loss(
        torch.cat([output["online_scene"], output["online_dynamic"]], dim=2).float(),
        torch.cat([output["online_base_scene"], output["online_base_dynamic"]], dim=2)
        .detach()
        .float(),
    )
    representation_anchor = current_anchor + future_anchor
    adapter_delta = 0.5 * (
        output["adapter_delta"].float().square().mean()
        + output["online_adapter_delta"].float().square().mean()
    )
    variance, embedding_std = _embedding_variance(
        output["initial_scene"],
        output["initial_dynamic"],
        output["online_scene"],
        output["online_dynamic"],
        target_std=config.embedding_std_target,
    )

    local_effect = pred_dynamic.new_zeros(())
    local_effect_direction = pred_dynamic.new_zeros(())
    local_effect_cosine = pred_dynamic.new_zeros(())
    pair_valid_fraction = pred_dynamic.new_zeros(())
    swap_rank = pred_dynamic.new_zeros(())
    swap_regret = pred_dynamic.new_zeros(())
    swap_correct_fraction = pred_dynamic.new_zeros(())
    if pair_output is not None and pair_valid is not None:
        valid = pair_valid.to(device=pred_dynamic.device, dtype=torch.bool)
        pair_valid_fraction = valid.float().mean()
        pred_effect = torch.cat(
            [
                (pred_dynamic - pair_output["pred_dynamic"]).float().flatten(start_dim=2),
                config.scene_predictive_weight**0.5
                * (pred_scene - pair_output["pred_scene"]).float().flatten(start_dim=2),
            ],
            dim=-1,
        )
        target_effect = torch.cat(
            [
                (target_dynamic - pair_output["target_dynamic"].detach())
                .float()
                .flatten(start_dim=2),
                config.scene_predictive_weight**0.5
                * (target_scene - pair_output["target_scene"].detach())
                .float()
                .flatten(start_dim=2),
            ],
            dim=-1,
        )
        local_per = F.smooth_l1_loss(pred_effect, target_effect, reduction="none").mean(
            dim=(-1, -2)
        )
        local_effect = _masked_mean(local_per, valid)
        effect_cosine_per = F.cosine_similarity(pred_effect, target_effect, dim=-1).mean(dim=1)
        local_effect_cosine = _masked_mean(effect_cosine_per, valid)
        local_effect_direction = _masked_mean(1.0 - effect_cosine_per, valid)

        if swapped_output is not None:
            swapped_error = _world_error_per_sample(
                swapped_output["pred_dynamic"],
                swapped_output["pred_scene"],
                target_dynamic,
                target_scene,
                scene_weight=config.scene_predictive_weight,
            )
            regret = swapped_error - full_error_per
            swap_regret = _masked_mean(regret, valid)
            swap_correct_fraction = _masked_mean((regret > 0).float(), valid)
            swap_rank = _masked_mean(F.relu(float(config.swap_margin) - regret), valid)

    prior_total = (
        config.predictive_weight * prior_predictive
        + config.scene_predictive_weight * prior_scene_predictive
        + config.state_path_weight * prior_state_path
    )
    full_total = (
        config.predictive_weight * predictive
        + config.scene_predictive_weight * scene_predictive
        + config.direction_weight * direction
        + config.amplitude_weight * amplitude
        + config.increment_weight * increment
        + config.scene_increment_weight * scene_increment
        + config.teacher_forced_weight * world_teacher_forced
        + config.descriptor_weight * descriptor
        + config.state_path_weight * state_path
        + config.prior_state_path_weight * prior_state_path.detach()
        + config.residual_weight * residual
        + config.residual_direction_weight * residual_direction
        + config.necessity_weight * necessity
        + config.inverse_action_weight * inverse_action
        + config.inverse_delta_weight * inverse_delta
        + config.inverse_gripper_weight * inverse_gripper
        + config.local_effect_weight * local_effect
        + config.local_effect_direction_weight * local_effect_direction
        + config.swap_rank_weight * swap_rank
    )
    if phase == "align":
        full_total = (
            full_total
            + config.representation_anchor_weight * representation_anchor
            + config.adapter_delta_weight * adapter_delta
            + config.variance_weight * variance
        )
    total = prior_total if phase == "prior" else full_total

    dynamic_by_step = F.smooth_l1_loss(
        pred_dynamic.float(), target_dynamic.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    scene_by_step = F.smooth_l1_loss(
        pred_scene.float(), target_scene.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    prior_dynamic_by_step = F.smooth_l1_loss(
        prior_dynamic.float(), target_dynamic.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    prior_scene_by_step = F.smooth_l1_loss(
        prior_scene.float(), target_scene.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    teacher_dynamic_by_step = F.smooth_l1_loss(
        teacher_dynamic.float(), target_dynamic.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    teacher_scene_by_step = F.smooth_l1_loss(
        teacher_scene.float(), target_scene.float(), reduction="none"
    ).mean(dim=(0, 2, 3))
    by_step: dict[str, Tensor] = {}
    for index, offset in enumerate(model.config.future_offsets):
        full_step = dynamic_by_step[index] + config.scene_predictive_weight * scene_by_step[index]
        prior_step = (
            prior_dynamic_by_step[index]
            + config.scene_predictive_weight * prior_scene_by_step[index]
        )
        teacher_step = (
            teacher_dynamic_by_step[index]
            + config.scene_predictive_weight * teacher_scene_by_step[index]
        )
        by_step[f"world_predictive_t{offset}"] = full_step
        by_step[f"prior_world_predictive_t{offset}"] = prior_step
        by_step[f"full_vs_prior_gain_t{offset}"] = prior_step - full_step
        by_step[f"world_teacher_forced_t{offset}"] = teacher_step
        by_step[f"closed_loop_gap_t{offset}"] = full_step - teacher_step
        by_step[f"residual_cosine_t{offset}"] = residual_cosine_by_step[:, index].mean()
    max_closed_loop_gap = torch.stack(
        [value for key, value in by_step.items() if key.startswith("closed_loop_gap_t")]
    ).max()

    effect_dynamic = output["action_dynamic_effect"]
    effect_scene = output["action_scene_effect"]
    zero_diag = pred_dynamic.new_zeros(())
    adaln_gate = output.get("action_effect_gate_abs_mean", zero_diag).float().mean()
    adaln_scale = output.get("action_effect_scale_abs_mean", zero_diag).float().mean()
    adaln_shift = output.get("action_effect_shift_abs_mean", zero_diag).float().mean()
    action_read_gate = (
        output.get("action_effect_action_read_gate_abs_mean", zero_diag).float().mean()
    )
    joint_rms = output.get("action_effect_joint_rms", zero_diag).float().mean()
    action_signal_rms = output.get("action_effect_action_signal_rms", zero_diag).float().mean()
    return {
        "loss": total,
        "prior_loss": prior_total,
        "full_loss": full_total,
        "predictive": predictive,
        "scene_predictive": scene_predictive,
        "world_predictive": world_predictive,
        "prior_predictive": prior_predictive,
        "prior_scene_predictive": prior_scene_predictive,
        "prior_world_predictive": prior_world_predictive,
        "full_vs_prior_gain": full_vs_prior_gain,
        "informative_full_vs_prior_gain": informative_gain,
        "informative_fraction": informative.float().mean(),
        "direction": direction,
        "amplitude": amplitude,
        "increment": increment,
        "scene_increment": scene_increment,
        "world_teacher_forced": world_teacher_forced,
        "closed_loop_gap": world_predictive.detach() - world_teacher_forced.detach(),
        "max_closed_loop_gap": max_closed_loop_gap.detach(),
        "descriptor": descriptor,
        "state_path": state_path,
        "prior_state_path": prior_state_path,
        "transition_fraction": event.float().mean(),
        "residual": residual,
        "residual_dynamic": residual_dynamic,
        "residual_scene": residual_scene,
        "residual_direction": residual_direction,
        "residual_cosine": residual_cosine,
        "necessity": necessity,
        "inverse_action": inverse_action,
        "inverse_delta": inverse_delta,
        "inverse_gripper": inverse_gripper,
        "inverse_gripper_accuracy": inverse_gripper_accuracy,
        "representation_anchor": representation_anchor,
        "adapter_delta": adapter_delta,
        "variance": variance,
        "embedding_std": embedding_std,
        "local_effect": local_effect,
        "local_effect_direction": local_effect_direction,
        "local_effect_cosine": local_effect_cosine,
        "local_pair_valid_fraction": pair_valid_fraction,
        "swap_rank": swap_rank,
        "swap_regret": swap_regret,
        "swap_correct_fraction": swap_correct_fraction,
        "action_effect_dynamic_rms": effect_dynamic.float().square().mean().sqrt(),
        "action_effect_scene_rms": effect_scene.float().square().mean().sqrt(),
        "adaln_gate_abs_mean": adaln_gate,
        "adaln_scale_abs_mean": adaln_scale,
        "adaln_shift_abs_mean": adaln_shift,
        "action_read_gate_abs_mean": action_read_gate,
        "action_world_joint_rms": joint_rms,
        "action_signal_rms": action_signal_rms,
        "pred_dynamic_rms": pred_dynamic.float().square().mean().sqrt(),
        "target_dynamic_rms": target_dynamic.float().square().mean().sqrt(),
        **by_step,
    }


__all__ = ["ControllableWorldLossConfig", "compute_controllable_world_losses"]
