from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .world_model import V35ObservedStateWorldModel


@dataclass(frozen=True)
class V35WorldLossConfig:
    latent_weight: float = 1.0
    teacher_forced_weight: float = 0.20
    increment_direction_weight: float = 0.20
    increment_amplitude_weight: float = 0.10
    state_endpoint_weight: float = 0.40
    inverse_pred_weight: float = 0.20
    inverse_target_weight: float = 0.05
    gripper_event_weight: float = 0.15
    region_current_weight: float = 0.10
    region_future_weight: float = 0.15
    masked_consistency_weight: float = 0.10
    variance_weight: float = 0.02
    covariance_weight: float = 0.002
    token_diversity_weight: float = 0.01
    pair_direction_weight: float = 0.10
    swap_rank_weight: float = 0.05
    utility_rank_weight: float = 0.05
    action_probe_weight: float = 0.02
    overshoot_weight: float = 0.0
    overshoot_mode: str = "full"
    overshoot_depth: int = 2
    overshoot_gamma: float = 0.7
    overshoot_cosine_weight: float = 0.10
    consequence_risk_weight: float = 0.02
    consequence_entropy_weight: float = 0.005
    consequence_diversity_weight: float = 0.005
    consequence_risk_temperature: float = 0.5
    transition_weighted_latent_weight: float = 0.0
    transition_window_weight: float = 1.5
    legacy_global_weight: float = 0.25
    gripper_transition_threshold: float = 0.10
    utility_margin: float = 0.002
    swap_margin: float = 0.002
    variance_target: float = 0.05


def component_huber(
    model: V35ObservedStateWorldModel, pred: Tensor, target: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    split_p = model.split_world(pred)
    split_t = model.split_world(target)
    global_loss = F.smooth_l1_loss(split_p["global"], split_t["global"])
    interaction_loss = F.smooth_l1_loss(split_p["interaction"], split_t["interaction"])
    motion_loss = F.smooth_l1_loss(split_p["motion"], split_t["motion"])
    return global_loss, interaction_loss, motion_loss


def legacy_error(
    model: V35ObservedStateWorldModel,
    pred: Tensor,
    target: Tensor,
    *,
    global_weight: float,
    reduction: str = "mean",
) -> Tensor:
    split_p = model.split_world(pred)
    split_t = model.split_world(target)
    global_e = F.smooth_l1_loss(split_p["global"], split_t["global"], reduction="none").mean(
        dim=(-1, -2)
    )
    dynamic_p = torch.cat([split_p["interaction"], split_p["motion"]], dim=-2)
    dynamic_t = torch.cat([split_t["interaction"], split_t["motion"]], dim=-2)
    dynamic_e = F.smooth_l1_loss(dynamic_p, dynamic_t, reduction="none").mean(dim=(-1, -2))
    out = dynamic_e + float(global_weight) * global_e
    return out.mean() if reduction == "mean" else out


def variance_covariance(world: Tensor, target_std: float) -> tuple[Tensor, Tensor, Tensor]:
    flat = world.float().reshape(-1, world.shape[-1])
    std = flat.std(dim=0, unbiased=False)
    variance = F.relu(float(target_std) - std).mean()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = covariance.square().mean()
    token = world.float().reshape(-1, world.shape[-2], world.shape[-1])
    norm = F.normalize(token, dim=-1)
    similarity = norm @ norm.transpose(-1, -2)
    eye = torch.eye(token.shape[-2], device=token.device, dtype=torch.bool)[None]
    diversity = similarity.masked_select(~eye).mean()
    return variance, covariance_loss, diversity


def gripper_classes(action_raw: Tensor, state_raw: Tensor, index: int, threshold: float) -> Tensor:
    boundary = torch.cat([state_raw[:, None, index], action_raw[:, :-1, index]], dim=1)
    delta = action_raw[..., index] - boundary
    classes = torch.zeros_like(delta, dtype=torch.long)
    classes[delta >= threshold] = 2
    classes[delta <= -threshold] = 1
    return classes


def gripper_segment_events(
    action_raw: Tensor, state_raw: Tensor, index: int, threshold: float, *, segment_length: int
) -> Tensor:
    classes = gripper_classes(action_raw, state_raw, index, threshold)
    return (classes != 0).reshape(action_raw.shape[0], -1, segment_length).any(dim=-1)


def weighted_legacy_error(
    model: V35ObservedStateWorldModel,
    pred: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    global_weight: float,
) -> Tensor:
    per = legacy_error(model, pred, target, global_weight=global_weight, reduction="none")
    weights = weights.to(device=per.device, dtype=per.dtype)
    return (per * weights).sum() / weights.sum().clamp_min(1e-6)


def overshoot_alignment_loss(
    model: V35ObservedStateWorldModel,
    pred: Tensor,
    target: Tensor,
    depth_index: Tensor,
    *,
    gamma: float,
    cosine_weight: float,
    global_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if pred.numel() == 0:
        zero = target.new_zeros(())
        return zero, zero, zero, zero
    per = legacy_error(
        model, pred[:, None], target[:, None], global_weight=global_weight, reduction="none"
    )[:, 0]
    depth = depth_index.to(device=per.device, dtype=per.dtype).clamp_min(1)
    weights = float(gamma) ** (depth - 1)
    huber = (per * weights).sum() / weights.sum().clamp_min(1e-6)
    cosine = 1 - F.cosine_similarity(pred.float().flatten(1), target.float().flatten(1), dim=-1)
    cosine = (cosine * weights).sum() / weights.sum().clamp_min(1e-6)
    total = huber + float(cosine_weight) * cosine
    d1 = per[depth_index == 1].mean() if (depth_index == 1).any() else pred.new_zeros(())
    d2 = per[depth_index == 2].mean() if (depth_index == 2).any() else pred.new_zeros(())
    return total, huber, cosine, d2 - d1


def transition_direction_loss(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pred_flat = pred.float().flatten(2)
    target_flat = target.float().flatten(2)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    direction = (1 - cosine).mean()
    pred_amp = pred_flat.square().mean(dim=-1).sqrt()
    target_amp = target_flat.square().mean(dim=-1).sqrt()
    amplitude = F.smooth_l1_loss(pred_amp, target_amp)
    return direction, amplitude, cosine.mean()


def consequence_risk_features(
    model: V35ObservedStateWorldModel,
    pred: Tensor,
    target: Tensor,
    teacher: Tensor,
    segment_events: Tensor,
    action_effect: Tensor,
    *,
    global_weight: float,
) -> tuple[Tensor, Tensor]:
    """Stop-gradient self-diagnostic risk used by consequence attention.

    The target is intentionally detached: consequence slots should learn where
    the current world model drifts or becomes action-sensitive, without letting
    the upstream latent/dynamics path reduce risk by changing the diagnostic.
    """
    with torch.no_grad():
        self_error = legacy_error(
            model, pred.detach(), target.detach(), global_weight=global_weight, reduction="none"
        )
        teacher_error = legacy_error(
            model, teacher.detach(), target.detach(), global_weight=global_weight, reduction="none"
        )
        gap = (self_error - teacher_error).relu()
        effect = action_effect.detach().float().flatten(2).square().mean(dim=-1).sqrt()
        event = segment_events.detach().float()
        features = torch.stack([self_error, gap, effect, event], dim=-1)
        # A compact scalar target for attention supervision.  Normalize per
        # sample so the scorer learns relative consequence within the horizon.
        score = self_error + gap + 0.25 * effect + 0.5 * event
        score = (score - score.mean(dim=-1, keepdim=True)) / score.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
    return features, score


def consequence_attention_regularizers(
    attention: Tensor, risk_score: Tensor, *, temperature: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if attention.numel() == 0:
        zero = risk_score.new_zeros(())
        return zero, zero, zero, zero
    attn_mean = attention.mean(dim=1).clamp_min(1e-8)
    risk_prob = (risk_score / max(float(temperature), 1e-4)).softmax(dim=-1).detach()
    risk_loss = F.kl_div(attn_mean.log(), risk_prob, reduction="batchmean")
    entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1).mean()
    entropy = entropy / torch.log(attention.new_tensor(max(attention.shape[-1], 2)))
    overlap = attention @ attention.transpose(-1, -2)
    if attention.shape[1] > 1:
        eye = torch.eye(attention.shape[1], device=attention.device, dtype=torch.bool)[None]
        diversity = overlap.masked_select(~eye).mean()
    else:
        diversity = attention.new_zeros(())
    peak = attention.max(dim=-1).values.mean()
    return risk_loss, entropy, diversity, peak


def compute_v35_world_losses(
    model: V35ObservedStateWorldModel,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    *,
    config: V35WorldLossConfig,
    masked_world: Tensor | None = None,
    pair_output: dict[str, Tensor] | None = None,
    pair_valid: Tensor | None = None,
    swapped_output: dict[str, Tensor] | None = None,
    stability_scale: float = 1.0,
    action_scale: float = 1.0,
) -> dict[str, Tensor]:
    pred, target = output["pred_world"], output["target_world"]
    global_l, interaction_l, motion_l = component_huber(model, pred, target)
    latent = motion_l + interaction_l + config.legacy_global_weight * global_l
    legacy = legacy_error(model, pred, target, global_weight=config.legacy_global_weight)
    teacher = legacy_error(
        model,
        output["teacher_forced_world"],
        target,
        global_weight=config.legacy_global_weight,
    )
    segment_events = gripper_segment_events(
        sample["action_raw"],
        sample["state_raw"],
        model.config.gripper_index,
        config.gripper_transition_threshold,
        segment_length=model.config.segment_length,
    )
    transition_weights = torch.where(
        segment_events,
        pred.new_full(segment_events.shape, float(config.transition_window_weight)),
        pred.new_ones(segment_events.shape),
    )
    transition_weighted_latent = weighted_legacy_error(
        model, pred, target, transition_weights, global_weight=config.legacy_global_weight
    )
    risk_features, risk_score = consequence_risk_features(
        model,
        pred,
        target,
        output["teacher_forced_world"],
        segment_events,
        output["action_world_effect"],
        global_weight=config.legacy_global_weight,
    )
    overshoot_total = pred.new_zeros(())
    overshoot_huber = pred.new_zeros(())
    overshoot_cosine = pred.new_zeros(())
    overshoot_depth_gap = pred.new_zeros(())
    consequence_risk = pred.new_zeros(())
    consequence_entropy = pred.new_zeros(())
    consequence_diversity = pred.new_zeros(())
    consequence_peak = pred.new_zeros(())
    consequence_expected_start = pred.new_zeros(())
    if float(config.overshoot_weight) > 0 and int(config.overshoot_depth) > 0:
        mode = str(config.overshoot_mode).lower().replace("-", "_")
        if mode in {"consequence", "consequence_query", "learned_consequence"}:
            overshoot = model.consequence_overshooting(
                output["target_initial_world"].detach(),
                output["target_world"].detach(),
                pred,
                output["teacher_forced_world"],
                output["actual_tokens"],
                depth=int(config.overshoot_depth),
                risk_features=risk_features,
                detach_start=True,
                detach_target=True,
            )
            consequence_risk, consequence_entropy, consequence_diversity, consequence_peak = (
                consequence_attention_regularizers(
                    overshoot["consequence_attention"],
                    risk_score,
                    temperature=float(config.consequence_risk_temperature),
                )
            )
            consequence_expected_start = overshoot["consequence_expected_start"].detach()
        elif mode in {"full", "dense", "all_start", "multi_start"}:
            overshoot = model.multi_start_overshooting(
                output["target_initial_world"].detach(),
                output["target_world"].detach(),
                output["actual_tokens"],
                depth=int(config.overshoot_depth),
                detach_start=True,
                detach_target=True,
            )
        elif mode in {"none", "off", "disabled"}:
            overshoot = None
        else:
            raise ValueError(f"unknown overshoot_mode: {config.overshoot_mode}")
        if overshoot is not None:
            overshoot_total, overshoot_huber, overshoot_cosine, overshoot_depth_gap = (
                overshoot_alignment_loss(
                    model,
                    overshoot["overshoot_world"],
                    overshoot["overshoot_target"],
                    overshoot["overshoot_depth_index"],
                    gamma=float(config.overshoot_gamma),
                    cosine_weight=float(config.overshoot_cosine_weight),
                    global_weight=config.legacy_global_weight,
                )
            )

    pred_prev = torch.cat([output["initial_world"][:, None], pred[:, :-1]], dim=1)
    target_prev = torch.cat([output["target_initial_world"][:, None], target[:, :-1]], dim=1)
    pred_inc = pred - pred_prev
    target_inc = target - target_prev
    increment_direction, increment_amplitude, increment_cosine = transition_direction_loss(
        pred_inc, target_inc
    )

    state_endpoint = F.smooth_l1_loss(output["pred_segment_state"], sample["segment_state"])
    target_state_probe = F.smooth_l1_loss(
        output["target_segment_state_prediction"], sample["segment_state"]
    )

    cfg = model.config
    action_segments = sample["action"].reshape(
        sample["action"].shape[0], cfg.num_segments, cfg.segment_length, cfg.action_dim
    )
    pred_inverse = F.smooth_l1_loss(output["pred_inverse_action"], action_segments)
    target_inverse = F.smooth_l1_loss(output["target_inverse_action"], action_segments)
    classes = gripper_classes(
        sample["action_raw"],
        sample["state_raw"],
        cfg.gripper_index,
        config.gripper_transition_threshold,
    ).reshape(sample["action"].shape[0], cfg.num_segments, cfg.segment_length)
    class_weight = torch.tensor([0.25, 1.0, 1.0], device=pred.device, dtype=torch.float32)
    pred_gripper = F.cross_entropy(
        output["pred_inverse_gripper_logits"].float().flatten(0, 2),
        classes.flatten(),
        weight=class_weight,
    )
    target_gripper = F.cross_entropy(
        output["target_inverse_gripper_logits"].float().flatten(0, 2),
        classes.flatten(),
        weight=class_weight,
    )

    current_region = F.smooth_l1_loss(
        output["current_region_prediction"], output["current_region_target"]
    )
    future_region = F.smooth_l1_loss(
        output["pred_region_prediction"], output["future_region_target"]
    )
    masked_consistency = (
        pred.new_zeros(())
        if masked_world is None
        else F.smooth_l1_loss(masked_world, output["initial_world"].detach())
    )
    variance, covariance, diversity = variance_covariance(
        torch.cat([output["initial_world"][:, None], pred], dim=1), config.variance_target
    )

    full_per = legacy_error(
        model, pred, target, global_weight=config.legacy_global_weight, reduction="none"
    ).mean(dim=1)
    hold_per = legacy_error(
        model,
        output["hold_world"],
        target,
        global_weight=config.legacy_global_weight,
        reduction="none",
    ).mean(dim=1)
    utility_rank = F.relu(config.utility_margin + full_per - hold_per.detach()).mean()
    relative_gain = (
        (hold_per.detach() - full_per.detach()) / hold_per.detach().clamp_min(1e-8)
    ).mean()

    pair_direction = pred.new_zeros(())
    local_effect_cosine = pred.new_zeros(())
    if pair_output is not None and pair_valid is not None:
        mask = pair_valid.bool()
        if mask.any():
            pred_transition = pred - output["initial_world"][:, None]
            pair_transition = pair_output["pred_world"] - pair_output["initial_world"][:, None]
            target_transition = target - output["target_initial_world"][:, None]
            pair_target_transition = (
                pair_output["target_world"] - pair_output["target_initial_world"][:, None]
            )
            pred_diff = (pred_transition - pair_transition)[mask].flatten(2)
            target_diff = (target_transition - pair_target_transition)[mask].flatten(2)
            cosine = F.cosine_similarity(pred_diff, target_diff, dim=-1)
            pair_direction = (1 - cosine).mean()
            local_effect_cosine = cosine.mean()

    swap_rank = pred.new_zeros(())
    swap_correct = pred.new_zeros(())
    if swapped_output is not None:
        swapped_per = (
            legacy_error(
                model,
                swapped_output["pred_world"],
                target.detach(),
                global_weight=config.legacy_global_weight,
                reduction="none",
            )
            .mean(dim=1)
            .detach()
        )
        swap_rank = F.relu(config.swap_margin + full_per - swapped_per).mean()
        swap_correct = (full_per.detach() < swapped_per).float().mean()

    action_probe = F.smooth_l1_loss(
        output["action_only_probe_prediction"], output["action_only_probe_target"]
    )
    total = (
        config.latent_weight * latent
        + config.teacher_forced_weight * teacher
        + config.increment_direction_weight * increment_direction
        + config.increment_amplitude_weight * increment_amplitude
        + config.state_endpoint_weight * state_endpoint
        + config.inverse_pred_weight * pred_inverse
        + config.inverse_target_weight * (target_inverse + 0.25 * target_state_probe)
        + config.gripper_event_weight * (pred_gripper + 0.25 * target_gripper)
        + config.region_current_weight * current_region
        + config.region_future_weight * future_region
        + config.masked_consistency_weight * masked_consistency
        + stability_scale
        * (
            config.variance_weight * variance
            + config.covariance_weight * covariance
            + config.token_diversity_weight * diversity
        )
        + action_scale
        * (
            config.utility_rank_weight * utility_rank
            + config.pair_direction_weight * pair_direction
            + config.swap_rank_weight * swap_rank
        )
        + config.action_probe_weight * action_probe
        + config.overshoot_weight * overshoot_total
        + config.consequence_risk_weight * consequence_risk
        + config.consequence_entropy_weight * consequence_entropy
        + config.consequence_diversity_weight * consequence_diversity
        + config.transition_weighted_latent_weight * transition_weighted_latent
    )
    return {
        "loss": total,
        "val_full": legacy,
        "latent": latent,
        "global_predictive": global_l,
        "interaction_predictive": interaction_l,
        "motion_predictive": motion_l,
        "teacher_forced": teacher,
        "increment_direction": increment_direction,
        "increment_amplitude": increment_amplitude,
        "increment_cosine": increment_cosine,
        "state_endpoint": state_endpoint,
        "target_state_probe": target_state_probe,
        "inverse_pred": pred_inverse,
        "inverse_target": target_inverse,
        "gripper_pred": pred_gripper,
        "gripper_target": target_gripper,
        "region_current": current_region,
        "region_future": future_region,
        "masked_consistency": masked_consistency,
        "variance": variance,
        "covariance": covariance,
        "token_diversity": diversity,
        "utility_rank": utility_rank,
        "full_vs_hold_relative_gain": relative_gain.detach(),
        "pair_direction": pair_direction,
        "local_effect_cosine": local_effect_cosine.detach(),
        "swap_rank": swap_rank,
        "swap_correct_fraction": swap_correct.detach(),
        "action_probe": action_probe,
        "overshoot": overshoot_total,
        "overshoot_huber": overshoot_huber,
        "overshoot_cosine": overshoot_cosine,
        "overshoot_depth_gap": overshoot_depth_gap.detach(),
        "consequence_risk": consequence_risk.detach(),
        "consequence_entropy": consequence_entropy.detach(),
        "consequence_diversity": consequence_diversity.detach(),
        "consequence_peak": consequence_peak.detach(),
        "consequence_expected_start": consequence_expected_start.detach(),
        "transition_weighted_latent": transition_weighted_latent,
        "transition_event_rate": segment_events.float().mean().detach(),
        "teacher_self_gap": (legacy.detach() - teacher.detach()),
        "embedding_std": torch.cat(
            [output["initial_world"].detach()[:, None], pred.detach()], dim=1
        )
        .float()
        .std(),
        "action_effect_rms": output["action_world_effect"].detach().float().square().mean().sqrt(),
        "adaln_gate_abs_mean": output["adaln_gate_abs_mean"].detach(),
        "world_action_joint_rms": output["world_action_joint_rms"].detach(),
    }


__all__ = ["V35WorldLossConfig", "legacy_error", "compute_v35_world_losses"]
