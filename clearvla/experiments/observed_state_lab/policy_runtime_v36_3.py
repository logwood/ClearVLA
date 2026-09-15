from __future__ import annotations

"""Training/evaluation runtime for V36.3 transition-aware action latent policy."""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from clearvla.experiments.classic_policy_lab.normalizer import ArrayNormalizer
from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner
from clearvla.experiments.dynamic_world_lab.shared_runtime import (
    encode_current_tokens,
    gripper_transition_metrics,
)

from .policy_v36_3 import V363PolicySystem
from .world_runtime import autocast_context, grad_norm, jsonable, scheduler


@dataclass(frozen=True)
class V363PolicyTrainerConfig:
    epochs: int = 12
    lr: float = 1e-4
    proposal_lr: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1
    proposal_loss_weight: float = 0.05
    first_weight: float = 1.5
    first4_weight: float = 1.3
    first8_weight: float = 1.15
    tail_weight: float = 1.10
    # The historical step weights favour the first action even though sampled
    # validation shows the far horizon learning much more slowly.  New
    # single-stage experiments can opt into a smooth (not banded) tail ramp.
    # Defaults preserve every pre-V101 experiment exactly.
    horizon_weight_mode: str = "legacy"
    horizon_tail_emphasis: float = 0.0
    horizon_first_step_protection: float = 0.0
    # Continuous, target-derived reweighting across windows.  This does not
    # discard static samples or force motion; weights are detached, bounded and
    # renormalized to unit mean inside each batch.
    trajectory_information_weight: float = 0.0
    trajectory_information_min: float = 0.75
    trajectory_information_max: float = 1.50
    event_loss_weight: float = 0.08
    event_positive_weight: float = 6.0
    event_focal_gamma: float = 1.0
    gripper_fm_event_boost: float = 0.0
    gripper_fm_value_weight: float = 1.0
    gripper_fm_delta_weight: float = 1.0
    arm_manifold_null_weight: float = 1.0
    gripper_transition_l1_weight: float = 0.04
    smooth_delta_weight: float = 0.02
    decoded_action_loss_weight: float = 0.04
    physical_delta_consistency_weight: float = 0.03
    transition_gripper_flow_weight: float = 0.08
    event_delta_consistency_weight: float = 0.03
    event_magnitude_weight: float = 0.03
    event_off_delta_weight: float = 0.01
    arm_motion_loss_weight: float = 0.03
    arm_motion_threshold: float = 0.02
    gripper_event_threshold: float = 0.10
    deploy_min_recall: float = 0.40
    deploy_min_event_ratio: float = 0.70
    deploy_max_event_ratio: float = 1.80
    deploy_max_tail_first_ratio: float = 2.60
    eval_inference_steps: int = 5
    log_every: int = 10
    max_train_batches: int = 0
    max_val_batches: int = 0


@torch.no_grad()
def prepare_v363_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V363PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    visual = encode_current_tokens(
        sample,
        conditioner=conditioner,
        model_config=system.world_config,
        camera_names=camera_names,
        device=device,
        dtype=dtype,
    )
    keys = (
        "state",
        "state_raw",
        "action_state",
        "history_state",
        "executed_action_history",
        "executed_action_history_raw",
        "policy_action",
        "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in (
        "state",
        "action_state",
        "history_state",
        "executed_action_history",
        "policy_action",
    ):
        out[key] = out[key].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    return out


def position_weights(config, trainer: V363PolicyTrainerConfig, device: torch.device) -> Tensor:
    mode = str(getattr(trainer, "horizon_weight_mode", "legacy")).strip().lower().replace(
        "-", "_"
    )
    if mode == "legacy":
        weight = torch.full((config.action_horizon,), float(trainer.tail_weight), device=device)
        weight[:8] = float(trainer.first8_weight)
        weight[:4] = float(trainer.first4_weight)
        weight[0] = float(trainer.first_weight)
    elif mode == "smooth_tail":
        emphasis = float(getattr(trainer, "horizon_tail_emphasis", 0.0))
        if emphasis < 0.0:
            raise ValueError("horizon_tail_emphasis must be non-negative")
        if int(config.action_horizon) <= 1:
            progress = torch.zeros((int(config.action_horizon),), device=device)
        else:
            progress = torch.linspace(0.0, 1.0, int(config.action_horizon), device=device)
        # Smoothstep has no discontinuity at first4/first8 and preserves the
        # high-precision near horizon while gradually allocating more gradient
        # to the part that plateaued in validation.
        ramp = progress.square() * (3.0 - 2.0 * progress)
        weight = 1.0 + emphasis * ramp
    elif mode == "anchor_bands":
        emphasis = float(getattr(trainer, "horizon_tail_emphasis", 0.0))
        first_protection = float(getattr(trainer, "horizon_first_step_protection", 0.0))
        if emphasis < 0.0 or first_protection < 0.0:
            raise ValueError("anchor-band emphasis and first-step protection must be non-negative")
        boundaries = tuple(int(value) for value in config.flow_jepa_action_offsets)
        if not boundaries or boundaries[-1] != int(config.action_horizon):
            raise ValueError("anchor_bands requires Flow-JEPA action offsets ending at action_horizon")
        weight = torch.ones((int(config.action_horizon),), device=device)
        start = 0
        denominator = max(len(boundaries) - 1, 1)
        for index, end in enumerate(boundaries):
            band_gain = emphasis * float(index) / float(denominator)
            weight[start:end] = 1.0 + band_gain
            start = end
        weight[0] = weight[0] + first_protection
    else:
        raise ValueError(
            "unknown horizon_weight_mode="
            f"{mode!r}; expected 'legacy', 'smooth_tail', or 'anchor_bands'"
        )
    return weight / weight.mean()


def trajectory_information_weights(
    sample: dict[str, Tensor],
    trainer: V363PolicyTrainerConfig,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return bounded continuous window weights and their motion score.

    The score is the normalized-action RMS change over the policy window.  It
    is a supervision-side sampling statistic only: no model prediction enters
    it and gradients cannot use the weight as a shortcut.  A zero strength (the
    compatibility default) returns exact ones.
    """

    target = sample["policy_action"].to(device=device).detach().float()
    current = sample["action_state"].to(device=device).detach().float()
    if target.ndim != 3 or current.ndim != 2 or target.shape[0] != current.shape[0]:
        raise ValueError("trajectory information expects policy_action [B,T,D] and action_state [B,D]")
    boundary = torch.cat([current[:, None], target[:, :-1]], dim=1)
    score = (target - boundary).square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()
    strength = float(getattr(trainer, "trajectory_information_weight", 0.0))
    if strength <= 0.0:
        return torch.ones_like(score), score
    lower = float(getattr(trainer, "trajectory_information_min", 0.75))
    upper = float(getattr(trainer, "trajectory_information_max", 1.50))
    if lower <= 0.0 or upper < lower:
        raise ValueError("trajectory information bounds require 0 < min <= max")
    relative = score / score.mean().clamp_min(1e-8)
    weight = 1.0 + strength * (relative - 1.0)
    weight = weight.clamp(min=lower, max=upper)
    # Clipping changes the mean; restore a stable objective scale.  Detaching
    # makes the contract explicit even if future datasets expose learnable
    # preprocessing tensors.
    weight = (weight / weight.mean().clamp_min(1e-8)).detach()
    return weight, score


def gripper_event_labels(
    *, target_raw: Tensor, current_raw: Tensor, gripper_index: int, threshold: float
) -> Tensor:
    target_g = target_raw[..., gripper_index].float()
    current_g = current_raw[..., gripper_index].float().reshape(-1, 1)
    boundary = torch.cat([current_g, target_g[:, :-1]], dim=1)
    delta = target_g - boundary
    labels = torch.zeros_like(delta, dtype=torch.long)
    labels = torch.where(delta <= -float(threshold), torch.ones_like(labels), labels)
    labels = torch.where(delta >= float(threshold), torch.full_like(labels, 2), labels)
    return labels


def _focal_cross_entropy(logits: Tensor, labels: Tensor, weights: Tensor, gamma: float) -> Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    if gamma > 0:
        pt = torch.exp(-ce.detach()).clamp(min=1e-6, max=1.0)
        ce = ((1.0 - pt) ** float(gamma)) * ce
    return (ce * weights).mean()


def event_head_metrics(
    logits_rows: list[np.ndarray], target_rows: list[np.ndarray]
) -> dict[str, float]:
    if not logits_rows:
        return {}
    logits = np.concatenate(logits_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    pred = logits.argmax(axis=-1)
    out: dict[str, float] = {"event_head_accuracy": float((pred == target).mean())}
    pos_pred = pred != 0
    pos_target = target != 0
    tp = float(np.logical_and(pos_pred, pos_target).sum())
    fp = float(np.logical_and(pos_pred, ~pos_target).sum())
    fn = float(np.logical_and(~pos_pred, pos_target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    out.update(
        {
            "event_head_precision": float(precision),
            "event_head_recall": float(recall),
            "event_head_f1": float(f1),
            "event_head_pred_events": float(pos_pred.sum()),
            "event_head_target_events": float(pos_target.sum()),
        }
    )
    for label, name in ((1, "open"), (2, "close")):
        p = pred == label
        t = target == label
        ltp = float(np.logical_and(p, t).sum())
        lfp = float(np.logical_and(p, ~t).sum())
        lfn = float(np.logical_and(~p, t).sum())
        lp = ltp / max(ltp + lfp, 1.0)
        lr = ltp / max(ltp + lfn, 1.0)
        lf1 = 2.0 * lp * lr / max(lp + lr, 1e-8)
        out[f"event_head_{name}_precision"] = float(lp)
        out[f"event_head_{name}_recall"] = float(lr)
        out[f"event_head_{name}_f1"] = float(lf1)
    return out


def arm_motion_labels(
    system: V363PolicySystem, target_action: Tensor, action_state: Tensor, threshold: float
) -> Tensor:
    physical = system.codec.encode(target_action, action_state)
    parts = system.codec.split_physical(physical)
    norm = parts["arm_delta"].float().norm(dim=-1)
    return (norm >= float(threshold)).to(target_action.dtype)


def semantic_physical_velocity_error(
    system: V363PolicySystem,
    residual: Tensor,
    *,
    arm_null_weight: float = 1.0,
) -> Tensor:
    """Return per-horizon error with gripper counted as one native dimension."""
    cfg = system.policy_config
    ad = int(cfg.arm_dim)
    gf = int(cfg.gripper_field_dim)
    if residual.ndim < 3 or int(residual.shape[-1]) != int(cfg.physical_action_dim):
        raise ValueError(
            f"physical residual must end in [T,{cfg.physical_action_dim}], got {tuple(residual.shape)}"
        )
    horizon = int(residual.shape[-2])
    flat = residual.reshape(-1, horizon, int(cfg.physical_action_dim))
    if system.codec.uses_arm_manifold:
        arm_native, _, arm_null = system.codec.project_arm_tangent(flat[..., : 2 * ad])
        arm_null_per_dim = 0.5 * (arm_null[..., :ad].square() + arm_null[..., ad : 2 * ad].square())
        arm_error = (arm_native.square() + max(float(arm_null_weight), 0.0) * arm_null_per_dim).sum(
            dim=-1
        )
    else:
        arm_error = 0.5 * (flat[..., :ad].square() + flat[..., ad : 2 * ad].square()).sum(dim=-1)
    gripper_field = flat[..., 2 * ad : 2 * ad + gf]
    if system.codec.uses_parseval_gripper_field:
        native = system.codec.decode_gripper_field(gripper_field)
        null = gripper_field - system.codec.project_gripper_field(gripper_field)
        gripper_error = native[..., 0].square() + null.square().sum(dim=-1)
    else:
        gripper_error = gripper_field.square().mean(dim=-1)
    error = (arm_error + gripper_error) / float(ad + 1)
    return error.reshape(*residual.shape[:-2], horizon)


def _normalized_event_emphasis(
    transition_mask: Tensor,
    position_weight: Tensor,
    boost: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reallocate gripper loss mass toward events without changing its total budget."""

    if transition_mask.ndim != 2:
        raise ValueError(f"transition_mask must be [B,T], got {tuple(transition_mask.shape)}")
    if position_weight.ndim != 1 or int(position_weight.shape[0]) != int(transition_mask.shape[1]):
        raise ValueError(
            f"position_weight must be [T={transition_mask.shape[1]}], got {tuple(position_weight.shape)}"
        )
    mask = transition_mask.to(dtype=torch.float32)
    horizon_weight = position_weight.to(device=mask.device, dtype=torch.float32)[None]
    raw = 1.0 + max(float(boost), 0.0) * mask
    # Normalize in the same horizon metric used by flow matching. Every sample
    # keeps exactly one native gripper dimension of aggregate loss mass.
    normalizer = (raw * horizon_weight).sum(dim=1, keepdim=True) / horizon_weight.sum().clamp_min(
        1e-8
    )
    emphasis = raw / normalizer.detach().clamp_min(1e-8)
    weighted = emphasis * horizon_weight
    event_mass = (weighted * mask).sum() / weighted.sum().clamp_min(1e-8)
    event_mean = (emphasis * mask).sum() / mask.sum().clamp_min(1.0)
    hold = 1.0 - mask
    hold_mean = (emphasis * hold).sum() / hold.sum().clamp_min(1.0)
    return emphasis, event_mass, event_mean, hold_mean


def flow_losses(
    system: V363PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V363PolicyTrainerConfig,
) -> dict[str, Tensor]:
    cfg = system.policy_config
    device = output["pred_physical_velocity"].device
    weight = position_weights(cfg, trainer, device)
    information_weight, information_score = trajectory_information_weights(
        sample,
        trainer,
        device=device,
    )
    labels = gripper_event_labels(
        target_raw=sample["policy_action_raw"].to(device=device),
        current_raw=sample["state_raw"].to(device=device),
        gripper_index=cfg.gripper_index,
        threshold=trainer.gripper_event_threshold,
    )
    velocity_residual = output["pred_physical_velocity"] - output["target_physical_velocity"]
    velocity_error = velocity_residual.square()

    # Flow matching stays on the existing velocity branch.  Arm remains six
    # native dimensions with abs/delta channels.  Gripper is expanded to a
    # larger typed field, then compressed back to one native-dimension error so
    # it has bandwidth without dominating the 7-D action loss.
    ad = int(cfg.arm_dim)
    gf = int(cfg.gripper_field_dim)
    grip_field = velocity_error[..., 2 * ad : 2 * ad + gf]
    arm_abs_error = velocity_error[..., :ad]
    arm_delta_error = velocity_error[..., ad : 2 * ad]
    zero = torch.zeros((), device=device, dtype=velocity_error.dtype)
    if system.codec.uses_arm_manifold:
        arm_native_residual, _, arm_null_residual = system.codec.project_arm_tangent(
            velocity_residual[..., : 2 * ad]
        )
        arm_native_component = arm_native_residual.square()
        arm_null_component = 0.5 * (
            arm_null_residual[..., :ad].square() + arm_null_residual[..., ad : 2 * ad].square()
        )
        arm_null_weight = max(float(trainer.arm_manifold_null_weight), 0.0)
        arm_native_error = arm_native_component + arm_null_weight * arm_null_component

        target_arm = output["target_physical_velocity"][..., : 2 * ad]
        _, _, target_arm_null = system.codec.project_arm_tangent(target_arm)
        arm_target_projection_error = target_arm_null.float().square().mean()
        source_noise = output.get("source_physical_noise")
        if source_noise is not None:
            source_arm = source_noise[..., : 2 * ad]
            projected_source_arm = system.codec.project_arm_field(
                source_arm, sample["action_state"].to(device=device)
            )
            arm_noise_projection_error = (
                (source_arm.float() - projected_source_arm.float()).square().mean()
            )
            arm_noise_abs_std = source_arm[..., :ad].float().std(unbiased=False)
            arm_noise_delta_std = source_arm[..., ad : 2 * ad].float().std(unbiased=False)
        else:
            arm_noise_projection_error = zero.float()
            arm_noise_abs_std = zero.float()
            arm_noise_delta_std = zero.float()
        target_physical = output.get("target_physical")
        if target_physical is not None:
            target_arm_physical = target_physical[..., : 2 * ad]
            arm_target_abs_std = target_arm_physical[..., :ad].float().std(unbiased=False)
            arm_target_delta_std = target_arm_physical[..., ad : 2 * ad].float().std(unbiased=False)
        else:
            arm_target_abs_std = zero.float()
            arm_target_delta_std = zero.float()
    else:
        arm_native_component = 0.5 * (arm_abs_error + arm_delta_error)
        arm_null_component = torch.zeros_like(arm_native_component)
        arm_native_error = arm_native_component
        arm_target_projection_error = zero.float()
        arm_noise_projection_error = zero.float()
        arm_noise_abs_std = zero.float()
        arm_noise_delta_std = zero.float()
        arm_target_abs_std = zero.float()
        arm_target_delta_std = zero.float()
    if system.codec.uses_parseval_gripper_field:
        field_residual = velocity_residual[..., 2 * ad : 2 * ad + gf]
        native_residual = system.codec.decode_gripper_field(field_residual)
        range_residual = system.codec.encode_gripper_field(native_residual)
        null_residual = field_residual - range_residual
        gripper_native_component = native_residual[..., 0].square()
        gripper_null_component = null_residual.square().sum(dim=-1)
        gripper_native_error = gripper_native_component + gripper_null_component
        previous_native = torch.cat([native_residual[:, :1], native_residual[:, :-1]], dim=1)
        gripper_delta_component = (native_residual - previous_native)[..., 0].square()
        target_field = output["target_physical_velocity"][..., 2 * ad : 2 * ad + gf]
        target_projection_error = (
            (target_field - system.codec.project_gripper_field(target_field))
            .float()
            .square()
            .mean()
        )
        target_native = system.codec.decode_gripper_field(target_field)
        target_energy_ratio = (
            target_field.float().square().sum()
            / target_native.float().square().sum().clamp_min(1e-8)
        )
    else:
        grip_channel_weight = torch.ones((gf,), device=device, dtype=velocity_error.dtype)
        grip_channel_weight[0] = max(float(trainer.gripper_fm_value_weight), 0.0)
        grip_channel_weight[1] = max(float(trainer.gripper_fm_delta_weight), 0.0)
        gripper_native_error = (grip_field * grip_channel_weight[None, None]).sum(
            dim=-1
        ) / grip_channel_weight.sum().clamp_min(1e-6)
        gripper_native_component = velocity_error[..., 2 * ad]
        gripper_delta_component = velocity_error[..., 2 * ad + 1]
        gripper_null_component = torch.zeros_like(gripper_native_component)
        target_projection_error = zero.float()
        target_energy_ratio = zero.float()
    gripper_native_error_raw = gripper_native_error
    transition_mask = (labels != 0).to(dtype=velocity_error.dtype, device=device)
    event_boost = max(float(trainer.gripper_fm_event_boost), 0.0)
    event_emphasis, event_loss_mass, event_weight_mean, hold_weight_mean = (
        _normalized_event_emphasis(
            transition_mask,
            weight,
            event_boost,
        )
    )
    gripper_native_error = gripper_native_error_raw * event_emphasis
    physical_error = (arm_native_error.sum(dim=-1) + gripper_native_error) / float(ad + 1)
    uniform_physical_error = (
        physical_error if system.codec.uses_parseval_gripper_field else velocity_error.mean(dim=-1)
    )
    horizon_weight = weight.to(dtype=physical_error.dtype)[None]
    flow_weight = horizon_weight * information_weight.to(dtype=physical_error.dtype)[:, None]
    flow = (physical_error * flow_weight).mean()
    flow_without_information_balance = (physical_error * horizon_weight).mean()
    uniform_flow = (
        uniform_physical_error * weight.to(dtype=uniform_physical_error.dtype)[None]
    ).mean()
    # Cross-version anchor metric. Always derive it from the same orthogonal arm
    # tangent projection and native gripper synthesis, including legacy runs
    # whose optimization objective still lives in independent physical fields.
    # It excludes null components and event emphasis. Comparisons also require
    # an identical action_normalizer_fingerprint.
    anchor_arm_native, _, _ = system.codec.project_arm_tangent(velocity_residual[..., : 2 * ad])
    anchor_gripper_native = system.codec.decode_gripper_field(
        velocity_residual[..., 2 * ad : 2 * ad + gf]
    )[..., 0]
    physical_native_error = (
        anchor_arm_native.square().sum(dim=-1) + anchor_gripper_native.square()
    ) / float(ad + 1)
    physical_flow_native = (physical_native_error * flow_weight).mean()
    physical_flow_native_uniform = physical_native_error.mean()
    arm_flow_per_dim = (arm_native_error.mean(dim=-1) * flow_weight).mean()
    arm_native_flow = (arm_native_component.mean(dim=-1) * flow_weight).mean()
    arm_null_flow = (arm_null_component.mean(dim=-1) * flow_weight).mean()
    # V70 metric overhaul.  The old *_null_ratio keys divide by the RESIDUAL
    # error, a quantity that collapses as training succeeds -- the ratio then
    # rises toward 1 at fixed null and misreports success as pathology.  They
    # are kept for continuity but renamed in spirit: treat them as
    # "null vs remaining error", never as health gauges.  The stable gauges
    # are the RMS (native units, comparable to data delta std and to the
    # fp32 hygiene floor) and the output-energy fraction (scale-free).
    arm_null_ratio = arm_null_flow / (arm_native_flow + arm_null_flow).detach().clamp_min(1e-6)
    arm_null_rms = arm_null_component.detach().float().mean().clamp_min(0.0).sqrt()
    pred_arm = output["pred_physical_velocity"][..., : 2 * ad].detach().float()
    pred_arm_energy = 0.5 * (pred_arm[..., :ad].square() + pred_arm[..., ad:].square()).mean()
    arm_null_output_fraction = (
        arm_null_component.detach().float().mean() / pred_arm_energy.clamp_min(1e-8)
    )
    gripper_field_flow = (gripper_native_error * flow_weight).mean()
    # V70: relative-difficulty ratio.  Both channels are first normalized by
    # their own target energy so a fast-collapsing arm residual no longer
    # inflates the gripper's apparent lag.
    target_v = output["target_physical_velocity"].detach().float()
    target_arm_energy = 0.5 * (
        target_v[..., :ad].square() + target_v[..., ad : 2 * ad].square()
    ).mean().clamp_min(1e-8)
    if system.codec.uses_parseval_gripper_field:
        target_grip_energy = (
            system.codec.decode_gripper_field(target_v[..., 2 * ad : 2 * ad + gf])[..., 0]
            .square()
            .mean()
            .clamp_min(1e-8)
        )
    else:
        target_grip_energy = target_v[..., 2 * ad].square().mean().clamp_min(1e-8)
    gripper_arm_flow_ratio = (gripper_field_flow / target_grip_energy) / (
        arm_flow_per_dim / target_arm_energy
    ).detach().clamp_min(1e-6)
    action_band_metrics: dict[str, Tensor] = {}
    band_start = 0
    action_band_offsets = (
        tuple(int(value) for value in cfg.flow_jepa_action_offsets)
        if int(getattr(cfg, "flow_jepa_enabled", 0))
        else ()
    )
    for band_end in action_band_offsets:
        if band_end <= band_start or band_end > int(physical_error.shape[1]):
            continue
        label = f"{band_start + 1}_{band_end}"
        action_band_metrics[f"action_band_{label}_physical_flow"] = physical_error[
            :, band_start:band_end
        ].mean()
        action_band_metrics[f"action_band_{label}_weight"] = weight[
            band_start:band_end
        ].mean().detach()
        band_start = band_end
    grip_value_flow = (gripper_native_component * flow_weight).mean()
    grip_delta_flow = (gripper_delta_component * flow_weight).mean()
    gripper_null_flow = (gripper_null_component * flow_weight).mean()
    gripper_null_ratio = gripper_null_flow / (
        grip_value_flow + gripper_null_flow
    ).detach().clamp_min(1e-6)
    grip_null_rms = gripper_null_component.detach().float().mean().clamp_min(0.0).sqrt()
    pred_grip_field = output["pred_physical_velocity"][..., 2 * ad : 2 * ad + gf].detach().float()
    pred_grip_energy = pred_grip_field.square().sum(dim=-1).mean()
    grip_null_output_fraction = (
        gripper_null_component.detach().float().mean() / pred_grip_energy.clamp_min(1e-8)
    )
    # V70 (H4 test): decompose gripper null energy by event vs hold steps.
    # If null concentrates at event steps it is a timing-uncertainty signature
    # (informative), not waste -- read it, don't suppress it.
    event_step_mask = transition_mask.to(dtype=torch.float32)
    hold_step_mask = 1.0 - event_step_mask
    grip_null_event_rms = (
        (
            (gripper_null_component.detach().float() * event_step_mask).sum()
            / event_step_mask.sum().clamp_min(1.0)
        )
        .clamp_min(0.0)
        .sqrt()
    )
    grip_null_hold_rms = (
        (
            (gripper_null_component.detach().float() * hold_step_mask).sum()
            / hold_step_mask.sum().clamp_min(1.0)
        )
        .clamp_min(0.0)
        .sqrt()
    )
    grip_null_event_hold_ratio = grip_null_event_rms / grip_null_hold_rms.clamp_min(1e-8)
    event_denom = (
        (transition_mask * weight.to(dtype=physical_error.dtype)[None]).sum().clamp_min(1.0)
    )
    hold_mask_for_flow = (1.0 - transition_mask).to(dtype=physical_error.dtype)
    hold_denom = (
        (hold_mask_for_flow * weight.to(dtype=physical_error.dtype)[None]).sum().clamp_min(1.0)
    )
    # Keep event/hold diagnostics in the native unweighted metric so runs with
    # different emphasis settings remain directly comparable.
    gripper_event_flow = (
        gripper_native_error_raw * transition_mask * weight.to(dtype=physical_error.dtype)[None]
    ).sum() / event_denom
    gripper_hold_flow = (
        gripper_native_error_raw * hold_mask_for_flow * weight.to(dtype=physical_error.dtype)[None]
    ).sum() / hold_denom
    temporal_balance_active = (
        str(getattr(trainer, "horizon_weight_mode", "legacy"))
        .strip()
        .lower()
        .replace("-", "_")
        != "legacy"
        or float(getattr(trainer, "trajectory_information_weight", 0.0)) > 0.0
    )
    auxiliary_step_weight = (
        flow_weight
        if temporal_balance_active
        else torch.ones_like(flow_weight)
    )

    proposal_rows = F.smooth_l1_loss(
        output["proposal_action"],
        sample["policy_action"],
        reduction="none",
    ).mean(dim=-1)
    proposal = (proposal_rows * auxiliary_step_weight).mean()

    flat_labels = labels.reshape(-1)
    flat_logits = output["event_logits"].reshape(-1, 3)
    event_weights = torch.ones_like(flat_labels, dtype=flat_logits.dtype)
    event_weights = event_weights + (flat_labels != 0).to(flat_logits.dtype) * float(
        trainer.event_positive_weight
    )
    event = _focal_cross_entropy(
        flat_logits,
        flat_labels,
        event_weights * auxiliary_step_weight.reshape(-1).to(dtype=event_weights.dtype),
        trainer.event_focal_gamma,
    )

    motion_target = arm_motion_labels(
        system,
        sample["policy_action"].to(device=device),
        sample["action_state"].to(device=device),
        trainer.arm_motion_threshold,
    )
    motion_rows = F.binary_cross_entropy_with_logits(
        output["motion_logits"].float(), motion_target.float(), reduction="none"
    )
    motion = (motion_rows * auxiliary_step_weight).mean()

    transition_mask = transition_mask.to(output["pred_action_estimate"].dtype)
    grip_idx = cfg.gripper_index
    pred_g = output["pred_action_estimate"][..., grip_idx]
    target_g = sample["policy_action"].to(device=device)[..., grip_idx]
    transition_l1 = (
        F.smooth_l1_loss(pred_g, target_g, reduction="none")
        * (1.0 + transition_mask * 8.0)
        * auxiliary_step_weight
    ).mean()

    pred_boundary = torch.cat(
        [sample["action_state"].to(device=device)[:, None], output["pred_action_estimate"][:, :-1]],
        dim=1,
    )
    target_boundary = torch.cat(
        [
            sample["action_state"].to(device=device)[:, None],
            sample["policy_action"].to(device=device)[:, :-1],
        ],
        dim=1,
    )
    pred_delta = output["pred_action_estimate"] - pred_boundary
    target_delta = sample["policy_action"].to(device=device) - target_boundary
    smooth_delta_rows = F.smooth_l1_loss(
        pred_delta, target_delta, reduction="none"
    ).mean(dim=-1)
    smooth_delta = (smooth_delta_rows * auxiliary_step_weight).mean()
    decoded_action_rows = F.smooth_l1_loss(
        output["pred_action_estimate"],
        sample["policy_action"].to(device=device),
        reduction="none",
    ).mean(dim=-1)
    decoded_action = (decoded_action_rows * auxiliary_step_weight).mean()
    physical_delta_consistency_rows = system.codec.delta_consistency_loss(
        output["clean_physical_estimate"],
        sample["action_state"].to(device=device),
        output["pred_action_estimate"],
        reduction="none",
    )
    physical_delta_consistency = (
        physical_delta_consistency_rows * auxiliary_step_weight
    ).mean()

    # V36.3 latent-coupling losses.  These losses supervise the existing final
    # decoded action and the existing typed velocity tensor; they do not create
    # a separate gripper command path.
    transition_gripper_weight = transition_mask * auxiliary_step_weight
    transition_gripper_flow = (
        gripper_native_error_raw * transition_gripper_weight
    ).sum() / transition_gripper_weight.sum().clamp(min=1.0)

    pred_delta_g = pred_delta[..., grip_idx]
    target_delta_g = target_delta[..., grip_idx].detach()
    target_event = labels != 0
    target_sign = torch.zeros_like(pred_delta_g)
    target_sign = torch.where(labels == 2, torch.ones_like(target_sign), target_sign)
    target_sign = torch.where(labels == 1, -torch.ones_like(target_sign), target_sign)
    event_prob = torch.softmax(output["event_logits"].float(), dim=-1)
    signed_event = event_prob[..., 2] - event_prob[..., 1]
    # Same-latent readout/action closure: event readout and final decoded
    # gripper delta must agree in sign on true transitions.
    target_event_weight = target_event.float() * auxiliary_step_weight
    event_delta_consistency = (
        F.softplus(-(signed_event * pred_delta_g.float() * target_sign.float()))
        * target_event_weight
    ).sum() / target_event_weight.sum().clamp(min=1.0)
    # Prevent the smooth channel from swallowing target transitions.  The target
    # magnitude is adaptive because training actions are normalized while event
    # labels are computed in raw Alicia-D units.
    event_magnitude = (
        F.relu(0.70 * target_delta_g.abs() - pred_delta_g.abs()) * target_event_weight
    ).sum() / target_event_weight.sum().clamp(min=1.0)
    # Hold regions should not get event-sized decoded gripper deltas.
    hold_mask = (~target_event).float() * auxiliary_step_weight
    event_off_delta = (pred_delta_g.abs() * hold_mask).sum() / hold_mask.sum().clamp(min=1.0)

    total = (
        flow
        + trainer.proposal_loss_weight * proposal
        + trainer.event_loss_weight * event
        + trainer.arm_motion_loss_weight * motion
        + trainer.gripper_transition_l1_weight * transition_l1
        + trainer.smooth_delta_weight * smooth_delta
        + trainer.decoded_action_loss_weight * decoded_action
        + trainer.physical_delta_consistency_weight * physical_delta_consistency
        + trainer.transition_gripper_flow_weight * transition_gripper_flow
        + trainer.event_delta_consistency_weight * event_delta_consistency
        + trainer.event_magnitude_weight * event_magnitude
        + trainer.event_off_delta_weight * event_off_delta
    )
    pred_event = output["event_logits"].argmax(dim=-1)
    pos_target = labels != 0
    pos_pred = pred_event != 0
    tp = (pos_pred & pos_target).sum().to(torch.float32)
    fp = (pos_pred & ~pos_target).sum().to(torch.float32)
    fn = (~pos_pred & pos_target).sum().to(torch.float32)
    event_precision = tp / torch.clamp(tp + fp, min=1.0)
    event_recall = tp / torch.clamp(tp + fn, min=1.0)
    motion_pred = torch.sigmoid(output["motion_logits"]) >= 0.5
    motion_target_bool = motion_target >= 0.5
    mtp = (motion_pred & motion_target_bool).sum().to(torch.float32)
    mfp = (motion_pred & ~motion_target_bool).sum().to(torch.float32)
    mfn = (~motion_pred & motion_target_bool).sum().to(torch.float32)
    motion_precision = mtp / torch.clamp(mtp + mfp, min=1.0)
    motion_recall = mtp / torch.clamp(mtp + mfn, min=1.0)
    return {
        **action_band_metrics,
        "loss": total,
        "physical_flow": flow,
        "physical_flow_no_information_balance": flow_without_information_balance.detach(),
        "physical_flow_uniform": uniform_flow,
        "physical_flow_native": physical_flow_native.detach(),
        "physical_flow_native_uniform": physical_flow_native_uniform.detach(),
        "arm_fm_per_dim": arm_flow_per_dim,
        "arm_fm_native": arm_native_flow,
        "arm_fm_null": arm_null_flow,
        "arm_fm_null_ratio": arm_null_ratio,
        "arm_fm_null_rms": arm_null_rms,
        "arm_fm_null_output_fraction": arm_null_output_fraction,
        "arm_fm_target_projection_error": arm_target_projection_error,
        "arm_fm_noise_projection_error": arm_noise_projection_error,
        "arm_noise_abs_std": arm_noise_abs_std,
        "arm_noise_delta_std": arm_noise_delta_std,
        "arm_target_abs_std": arm_target_abs_std,
        "arm_target_delta_std": arm_target_delta_std,
        "gripper_fm_field": gripper_field_flow,
        "gripper_arm_fm_ratio": gripper_arm_flow_ratio,
        "gripper_fm_value": grip_value_flow,
        "gripper_fm_delta": grip_delta_flow,
        "gripper_fm_native": grip_value_flow,
        "gripper_fm_null": gripper_null_flow,
        "gripper_fm_null_ratio": gripper_null_ratio,
        "gripper_fm_null_rms": grip_null_rms,
        "gripper_fm_null_output_fraction": grip_null_output_fraction,
        "gripper_fm_null_event_rms": grip_null_event_rms,
        "gripper_fm_null_hold_rms": grip_null_hold_rms,
        "gripper_fm_null_event_hold_ratio": grip_null_event_hold_ratio,
        "gripper_fm_target_projection_error": target_projection_error,
        "gripper_fm_target_energy_ratio": target_energy_ratio,
        "gripper_fm_event": gripper_event_flow,
        "gripper_fm_hold": gripper_hold_flow,
        "gripper_fm_event_rate": transition_mask.float().mean(),
        "gripper_fm_weight_mean": flow_weight.detach().float().mean(),
        "trajectory_information_score": information_score.detach().float().mean(),
        "trajectory_information_weight_min": information_weight.detach().float().min(),
        "trajectory_information_weight_max": information_weight.detach().float().max(),
        "trajectory_information_effective_fraction": (
            information_weight.detach().float().sum().square()
            / (
                float(information_weight.numel())
                * information_weight.detach().float().square().sum().clamp_min(1e-8)
            )
        ),
        "action_horizon_weight_first": weight.detach().float()[0],
        "action_horizon_weight_tail": weight.detach().float()[-1],
        "gripper_fm_event_loss_mass": event_loss_mass.detach().float(),
        "gripper_fm_event_emphasis_mean": event_weight_mean.detach().float(),
        "gripper_fm_hold_emphasis_mean": hold_weight_mean.detach().float(),
        "proposal": proposal,
        "event": event,
        "motion": motion,
        "transition_l1": transition_l1,
        "smooth_delta": smooth_delta,
        "decoded_action": decoded_action,
        "physical_delta_consistency": physical_delta_consistency,
        "transition_gripper_flow": transition_gripper_flow,
        "event_delta_consistency": event_delta_consistency,
        "event_magnitude": event_magnitude,
        "event_off_delta": event_off_delta,
        "first_physical_flow": physical_error[:, 0].mean(),
        "first4_physical_flow": physical_error[:, :4].mean(),
        "first8_physical_flow": physical_error[:, :8].mean(),
        "tail_physical_flow": physical_error[:, 8:].mean(),
        "event_head_precision": event_precision,
        "event_head_recall": event_recall,
        "motion_head_precision": motion_precision,
        "motion_head_recall": motion_recall,
    }


def decode(normalizer: ArrayNormalizer, value: Tensor) -> np.ndarray:
    return normalizer.decode(value.detach().float().cpu().numpy())


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(row) for row in rows)) if rows else set()
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


def normalize_world_ablation_mode(mode: str) -> str:
    normalized = str(mode).replace("-", "_").lower()
    aliases = {
        "": "normal",
        "none": "normal",
        "normal": "normal",
        "zero": "zero",
        "shuffle": "shuffle",
        "batch_mean": "batch_mean",
        "noise": "noise",
    }
    if normalized not in aliases:
        raise ValueError(f"unknown world ablation mode: {mode}")
    return aliases[normalized]


def world_ablation_deltas(
    metrics_by_mode: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    if "normal" not in metrics_by_mode:
        return {}
    normal = metrics_by_mode["normal"]
    rmse_keys = (
        "full_rmse",
        "first_rmse",
        "first4_rmse",
        "first8_rmse",
        "tail_rmse",
        "arm_full_rmse",
        "gripper_full_rmse",
    )
    f_keys = (
        "gripper_precision",
        "gripper_recall",
        "gripper_f1",
        "gripper_close_f1",
        "gripper_open_f1",
    )
    misc_keys = ("gripper_timing_mae_steps", "gripper_event_ratio", "proposal_utility_mse_gain")
    out: dict[str, dict[str, float]] = {}
    for mode, metrics in metrics_by_mode.items():
        if mode == "normal":
            continue
        row: dict[str, float] = {}
        for key in rmse_keys:
            if key in metrics and key in normal:
                row[f"{key}_increase_vs_normal"] = float(metrics[key] - normal[key])
        for key in f_keys:
            if key in metrics and key in normal:
                row[f"{key}_drop_vs_normal"] = float(normal[key] - metrics[key])
        for key in misc_keys:
            if key in metrics and key in normal:
                row[f"{key}_delta_vs_normal"] = float(metrics[key] - normal[key])
        # Compact headline scores used as world-contribution summaries.
        if "tail_rmse" in metrics and "tail_rmse" in normal:
            row["world_tail_reliance"] = float(metrics["tail_rmse"] - normal["tail_rmse"])
        if "gripper_f1" in metrics and "gripper_f1" in normal:
            row["world_event_f1_contribution"] = float(normal["gripper_f1"] - metrics["gripper_f1"])
        out[mode] = row
    return out


@torch.no_grad()
def evaluate_v363_policy(
    *,
    system: V363PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V363PolicyTrainerConfig,
    max_batches: int = 0,
    world_ablation: str = "normal",
    world_ablation_seed_base: int = 914363,
) -> dict[str, float]:
    system.eval()
    world_ablation = normalize_world_ablation_mode(world_ablation)
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    event_logits_rows: list[np.ndarray] = []
    event_target_rows: list[np.ndarray] = []
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        sample = prepare_v363_policy_sample(
            batch,
            conditioner=conditioner,
            system=system,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(36236 + batch_index)
        noise = system.codec.sample_noise(
            sample["policy_action"].shape[0],
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
        )
        with autocast_context(device, dtype):
            ablation_seed = int(world_ablation_seed_base) + int(batch_index)
            pred_pack = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=True,
                return_event_logits=True,
                world_ablation=world_ablation,
                world_ablation_seed=ablation_seed,
            )
            assert isinstance(pred_pack, dict)
            no_proposal = system.sample(
                sample["visual"],
                sample["history_state"],
                sample["executed_action_history"],
                sample["state"],
                steps=trainer.eval_inference_steps,
                noise=noise,
                use_proposal=False,
                world_ablation=world_ablation,
                world_ablation_seed=ablation_seed,
            )
        pred_rows.append(decode(action_normalizer, pred_pack["action"]))
        no_proposal_rows.append(decode(action_normalizer, no_proposal))
        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"],
            current_raw=sample["state_raw"],
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
        )
        event_logits_rows.append(pred_pack["event_logits"].detach().float().cpu().numpy())
        event_target_rows.append(labels.cpu().numpy())
    pred = np.concatenate(pred_rows)
    no_proposal = np.concatenate(no_proposal_rows)
    target = np.concatenate(target_rows)
    current = np.concatenate(current_rows)
    squared = (pred - target) ** 2
    metrics = {
        "full_mse": float(squared.mean()),
        "full_rmse": float(np.sqrt(squared.mean())),
        "first_rmse": float(np.sqrt(squared[:, 0].mean())),
        "first4_rmse": float(np.sqrt(squared[:, :4].mean())),
        "first8_rmse": float(np.sqrt(squared[:, :8].mean())),
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean()))
        if squared.shape[1] > 8
        else float("nan"),
        "arm_full_rmse": float(np.sqrt(squared[..., :-1].mean())),
        "gripper_full_rmse": float(np.sqrt(squared[..., -1].mean())),
        "proposal_utility_mse_gain": float(((no_proposal - target) ** 2).mean() - squared.mean()),
    }
    metrics.update(
        gripper_transition_metrics(
            pred,
            target,
            current,
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
            tolerance=2,
        )
    )
    metrics.update(event_head_metrics(event_logits_rows, event_target_rows))
    metrics["tail_first_ratio"] = float(metrics["tail_rmse"] / max(metrics["first_rmse"], 1e-8))
    metrics["gripper_event_ratio"] = float(
        metrics.get("gripper_pred_events", 0.0)
        / max(metrics.get("gripper_target_events", 0.0), 1.0)
    )
    metrics["world_ablation"] = world_ablation
    return metrics


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def balanced_score(metrics: dict[str, float], trainer: V363PolicyTrainerConfig) -> float:
    full = float(metrics["full_rmse"])
    f1 = float(metrics.get("gripper_f1", 0.0))
    recall = float(metrics.get("gripper_recall", 0.0))
    ratio = float(metrics.get("gripper_event_ratio", 0.0))
    tail_first = float(metrics.get("tail_first_ratio", 999.0))
    ratio_penalty = 0.0 if ratio > 0 else 1.0
    if ratio > 0:
        low = float(trainer.deploy_min_event_ratio)
        high = float(trainer.deploy_max_event_ratio)
        ratio_penalty = max(0.0, math.log(low / ratio)) + max(0.0, math.log(ratio / high))
    return float(
        full
        + 0.03 * (1.0 - f1)
        + 0.05 * max(0.0, float(trainer.deploy_min_recall) - recall)
        + 0.02 * ratio_penalty
        + 0.01 * max(0.0, tail_first - float(trainer.deploy_max_tail_first_ratio))
    )


def is_deploy_eligible(metrics: dict[str, float], trainer: V363PolicyTrainerConfig) -> bool:
    ratio = float(metrics.get("gripper_event_ratio", 0.0))
    return (
        float(metrics.get("gripper_recall", 0.0)) >= float(trainer.deploy_min_recall)
        and float(trainer.deploy_min_event_ratio) <= ratio <= float(trainer.deploy_max_event_ratio)
        and float(metrics.get("tail_first_ratio", 999.0))
        <= float(trainer.deploy_max_tail_first_ratio)
    )


def train_v363_policy(
    *,
    system: V363PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V363PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    system.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        [
            {"params": system.planner.parameters(), "lr": trainer.lr},
            {"params": system.decoder.parameters(), "lr": trainer.lr},
            {"params": system.proposal.parameters(), "lr": trainer.proposal_lr},
        ],
        weight_decay=trainer.weight_decay,
        betas=(trainer.beta1, trainer.beta2),
        eps=trainer.eps,
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    schedule = scheduler(
        optimizer, steps_per_epoch * trainer.epochs, trainer.warmup_steps, trainer.min_lr_ratio
    )
    start_epoch, global_step = 1, 0
    history: list[dict[str, Any]] = []
    best = {
        "full_mse": float("inf"),
        "gripper_f1": -float("inf"),
        "gripper_recall": -float("inf"),
        "balanced": float("inf"),
        "deploy_full_rmse": float("inf"),
    }
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") != "clearvla-v36-3-policy-checkpoint-v1":
            raise ValueError("resume checkpoint is not V36.3 policy")
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        global_step = int(payload["global_step"])
        history = list(payload.get("history", []))
        best.update(payload.get("best", {}))
        restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train()
        rows = []
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            sample = prepare_v363_policy_sample(
                batch,
                conditioner=conditioner,
                system=system,
                camera_names=camera_names,
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                output = system.flow_training_forward(
                    sample["visual"],
                    sample["history_state"],
                    sample["executed_action_history"],
                    sample["state"],
                    sample["policy_action"],
                )
                losses = flow_losses(system, sample, output, trainer)
            losses["loss"].float().backward()
            grad = grad_norm(system.parameters())
            torch.nn.utils.clip_grad_norm_(system.parameters(), trainer.grad_clip)
            optimizer.step()
            schedule.step()
            global_step += 1
            row = {key: float(value.detach().float().cpu()) for key, value in losses.items()}
            row["grad"] = grad
            rows.append(row)
            if trainer.log_every and batch_index % trainer.log_every == 0:
                print(
                    f"[v36.3-transition-latent] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} decode={row['decoded_action']:.6f} cons={row['physical_delta_consistency']:.6f} "
                    f"event={row['event']:.6f} motion={row['motion']:.6f} first={row['first_physical_flow']:.6f} "
                    f"evtR={row['event_head_recall']:.3f} motR={row['motion_head_recall']:.3f} grad={grad:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = mean_rows(rows)
        val_metrics = evaluate_v363_policy(
            system=system,
            loader=val_loader,
            conditioner=conditioner,
            device=device,
            dtype=dtype,
            camera_names=camera_names,
            action_normalizer=action_normalizer,
            trainer=trainer,
            max_batches=trainer.max_val_batches,
        )
        score = balanced_score(val_metrics, trainer)
        deploy_eligible = is_deploy_eligible(val_metrics, trainer)
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        with (out_dir / "v36_3_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics["full_mse"])
        f1 = float(val_metrics.get("gripper_f1", 0.0))
        recall = float(val_metrics.get("gripper_recall", 0.0))
        save = []
        if full < best["full_mse"]:
            best["full_mse"] = full
            save.append("best_full.pt")
        if f1 > best["gripper_f1"]:
            best["gripper_f1"] = f1
            save.append("best_gripper_f1.pt")
        if recall > best["gripper_recall"]:
            best["gripper_recall"] = recall
            save.append("best_gripper_recall.pt")
        if score < best["balanced"]:
            best["balanced"] = score
            save.append("best_balanced.pt")
        if deploy_eligible and float(val_metrics["full_rmse"]) < best["deploy_full_rmse"]:
            best["deploy_full_rmse"] = float(val_metrics["full_rmse"])
            save.append("best_deploy.pt")
        payload = {
            "schema": "clearvla-v36-3-policy-checkpoint-v1",
            "epoch": epoch,
            "global_step": global_step,
            "model": system.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
            "world_config": asdict(system.world_config),
            "policy_config": asdict(system.policy_config),
            "trainer_config": asdict(trainer),
            "action_normalizer": action_normalizer.to_dict(),
            "state_normalizer": state_normalizer.to_dict(),
            "context": context,
            "history": history,
            "best": best,
            "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v36_3_policy_summary.json").write_text(
            json.dumps(
                jsonable(
                    {"schema": "clearvla-v36-3-policy-summary-v1", "best": best, "latest": record}
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "V363PolicyTrainerConfig",
    "prepare_v363_policy_sample",
    "gripper_event_labels",
    "flow_losses",
    "evaluate_v363_policy",
    "normalize_world_ablation_mode",
    "world_ablation_deltas",
    "train_v363_policy",
    "balanced_score",
    "is_deploy_eligible",
]
