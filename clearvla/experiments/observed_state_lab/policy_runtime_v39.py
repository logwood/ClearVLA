from __future__ import annotations

"""Training/evaluation runtime for V39 staged mid-cut latent contract policy."""

import json
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
from clearvla.experiments.dynamic_world_lab.shared_runtime import encode_current_tokens, encode_target_tokens, gripper_transition_metrics

from .policy_v39 import V39PolicySystem
from .policy_runtime_v36_3 import (
    V363PolicyTrainerConfig,
    arm_motion_labels,
    balanced_score,
    decode,
    event_head_metrics,
    flow_losses as v363_flow_losses,
    gripper_event_labels,
    position_weights,
    is_deploy_eligible,
    mean_rows,
)
from .world_runtime import autocast_context, grad_norm, jsonable, scheduler


POLICY_CHECKPOINT_SCHEMAS = frozenset({
    "clearvla-v39-policy-checkpoint-v1",
    "clearvla-v40-policy-checkpoint-v1",
})


@dataclass(frozen=True)
class V39PolicyTrainerConfig(V363PolicyTrainerConfig):
    # V39 staged mid-cut latent-contract objectives. Future
    # target tokens are targets only; no future-noisy latent is fed to the model.
    rollout_dynamics_loss_weight: float = 0.03
    rollout_delta_loss_weight: float = 0.01
    rollout_contrast_loss_weight: float = 0.06
    rollout_contrast_margin: float = 0.02
    # Kept for CLI/checkpoint compatibility; defaults disabled to avoid the
    # future self-denoise shortcut.
    future_latent_loss_weight: float = 0.0
    action_effect_loss_weight: float = 0.0
    future_latent_loss_start_epoch: int = 1
    future_latent_max_batches: int = 0

    # Lightweight CUDA memory accounting. Disabled by default. Set
    # --memory-report-every N to emit a [cuda-mem] line and append JSONL rows
    # every N batches. Set --memory-report-detail 1 to also log stage-level
    # points inside the selected batch. This is monitoring only; it does not
    # change the V38.3 throughput-optimized training/eval semantics.
    memory_report_every: int = 0
    memory_report_detail: int = 0
    memory_report_sync: int = 0

    # Staging.  contract = stop at Z_mid and train only the pre-cut trunk plus
    # simple contract heads.  policy = train the full policy tail, with the
    # pre-cut trunk updated at a lower learning rate and an optional decaying
    # mid-cut preservation loss.
    training_stage: str = "contract"
    upper_lr_scale: float = 0.20
    midcut_head_lr_scale: float = 1.0
    midcut_aux_loss_weight: float = 0.05
    midcut_aux_final_ratio: float = 0.20
    midcut_aux_decay_epochs: int = 4
    midcut_rollout_dynamics_loss_weight: float = 0.03
    midcut_rollout_delta_loss_weight: float = 0.01
    midcut_rollout_contrast_loss_weight: float = 0.03

    # V42 compact latent-CVAE head.  Small KL keeps q(z|latent,target) close
    # to the conditional prior used at inference without letting KL dominate
    # the action flow/chunk reconstruction objective.
    latent_cvae_action_decoder_lr_scale: float = 1.0
    # V42.1: train the deploy/inference prior path directly.  The posterior path
    # remains a weak auxiliary reconstruction target instead of being allowed to
    # carry the main policy loss by looking at the target chunk.
    latent_cvae_kl_weight: float = 5e-4
    latent_cvae_posterior_recon_weight: float = 0.25
    latent_cvae_legacy_anchor_weight: float = 0.03
    latent_cvae_legacy_anchor_decay_steps: int = 2500
    latent_cvae_legacy_anchor_min_weight: float = 0.0
    latent_cvae_adaptive_regularizer_weight: float = 0.002
    latent_cvae_adaptive_route_entropy_weight: float = 0.001
    latent_cvae_trajectory_supervision_weight: float = 0.04
    latent_cvae_trajectory_coeff_weight: float = 0.04
    latent_cvae_trajectory_monotonic_weight: float = 0.01
    # V52: close the hidden-trajectory loophole with a deploy-safe
    # proposal-residual objective.  Coefficient terms supervise the smooth
    # trajectory controls; the bound term is measured in physical residual
    # coordinates so basis-null/high-frequency residuals cannot bypass it.
    latent_cvae_proposal_residual_coeff_weight: float = 0.06
    latent_cvae_proposal_residual_mid_coeff_weight: float = 0.03
    latent_cvae_proposal_residual_bound_weight: float = 0.002
    latent_cvae_proposal_residual_bound_ratio: float = 1.25
    latent_cvae_proposal_residual_coeff_ridge: float = 1e-2
    latent_cvae_proposal_residual_arm_only: int = 1
    # V58 detail residual unfolding.  These losses supervise intermediate
    # full-token detail states; the final policy loss remains the main outlet.
    latent_cvae_micro_supervision_weight: float = 0.05
    latent_cvae_micro_event_weight: float = 0.01
    latent_cvae_micro_monotonic_weight: float = 0.01
    latent_cvae_micro_weight_kl_weight: float = 0.0005
    latent_cvae_micro_coverage_smooth_weight: float = 0.001
    latent_cvae_micro_coverage_floor_weight: float = 0.001
    latent_cvae_micro_coverage_prior_logit_scale: float = 0.25
    latent_cvae_micro_coverage_floor_ratio: float = 0.55
    latent_cvae_micro_learned_weight_max: float = 0.35
    latent_cvae_micro_learned_ramp_steps: int = 2000
    latent_cvae_micro_weight_floor: float = 0.05
    latent_cvae_micro_event_positive_weight: float = 2.0
    latent_cvae_trajectory_smoothness_weight: float = 0.0
    latent_cvae_trajectory_update_smoothness_weight: float = 0.0
    latent_cvae_trajectory_update_energy_weight: float = 0.0
    latent_cvae_trajectory_projection_weight: float = 0.0
    block_action_denoise_lr_scale: float = 1.0
    block_action_denoise_regularizer_weight: float = 0.005
    block_action_denoise_x0_loss_weight: float = 1.0
    # Boundary seam diagnostics for temporal block denoising.  Continuity is
    # enforced structurally by native-action endpoint encoding and sampling
    # projection; these weights are off by default so seam losses remain probes.
    block_action_boundary_delta_weight: float = 0.0
    block_action_boundary_consistency_weight: float = 0.0

    # V39.1. contract_mode=layer_adapter keeps the full DiT active in Stage 1
    # and supervises tiny side adapters at every block instead of stopping at a
    # single hard midcut. Final policy loss is optional and weak in Stage 1.
    contract_mode: str = "midcut"
    layer_contract_loss_weight: float = 1.0
    layer_contract_final_action_loss_weight: float = 0.0
    layer_contract_final_action_lr_scale: float = 0.30
    layerwise_lr_min_scale: float = 0.30
    # Policy-stage migration knob.  The default 0 preserves the old behavior:
    # inherited layer-contract interfaces stay on upper_lr.  When a pre-fix
    # stage1 checkpoint is loaded with dirty adapter weights skipped, set this
    # to 1.0 so the freshly initialized layer interface learns at base LR.
    layer_contract_adapter_policy_lr_scale: float = 0.0

    # V39.2 layer-latent contract.  The per-layer latent/future head is the
    # primary Stage-1 objective.  A single shared flow-matching action probe is
    # a low-weight downstream readability probe; it should not dominate early
    # world-latent formation.
    layer_latent_loss_weight: float = 1.0
    layer_fm_probe_loss_weight: float = 0.0
    layer_event_loss_weight: float = 0.05
    layer_motion_loss_weight: float = 0.03
    layer_decoded_action_loss_weight: float = 0.0
    layer_contrast_loss_weight: float = 0.03

    # V39.3 recurrent milestone consequence regularizers.  These are applied
    # to the layer-contract rollout latent when the model is configured with
    # --layer-recurrent-consequence 1.  They keep the predicted future trajectory
    # from collapsing to a tiny average vector and make each milestone compare
    # against the corresponding sparse future anchor.
    layer_variance_loss_weight: float = 0.05
    layer_norm_loss_weight: float = 0.02
    layer_delta_match_loss_weight: float = 0.15

    # V53-A2: boosting-style layer contract.  Each layer's rollout/milestone
    # prediction is treated as a residual on top of the detached cumulative
    # prediction of the layers below it, and the loss is applied to the
    # cumulative sum.  Layer k therefore learns only what layers < k have not
    # explained, turning the parallel per-layer supervision into a telescoping
    # vertical series.
    layer_boost_residual: int = 0

    # V53.2: weight for the x_t-branch share hinge (model emits
    # latent_cvae_adaptive_noisy_ratio_regularizer when the max is set).
    latent_cvae_noisy_ratio_weight: float = 0.0


def _validate_current_token_tensor(tokens: Tensor, *, system: V39PolicySystem) -> None:
    cfg = system.policy_config
    expected = (
        cfg.visual_history_length,
        cfg.num_cameras,
        cfg.patches_per_camera,
        cfg.visual_token_dim,
    )
    if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"history_dinov2_tokens must be [B,{expected}], got {tuple(tokens.shape)}")


def _validate_target_anchor_token_tensor(tokens: Tensor, *, system: V39PolicySystem) -> None:
    cfg = system.policy_config
    expected_tail = (cfg.num_cameras, cfg.patches_per_camera, cfg.visual_token_dim)
    if tokens.ndim != 5 or tuple(tokens.shape[2:]) != expected_tail:
        raise ValueError(f"target_future_dinov2_tokens must be [B,F,{expected_tail}], got {tuple(tokens.shape)}")
    if int(tokens.shape[1]) < int(cfg.future_anchors):
        raise ValueError(f"target_future_dinov2_tokens has only {tokens.shape[1]} anchors; need {cfg.future_anchors}")


@torch.no_grad()
def encode_target_anchor_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Encode only V38's required future-anchor target tokens.

    V38's residual future-flow target only consumes the last target-history
    frame for the first ``future_anchors`` offsets.  The legacy helper encoded
    all ``num_future × history`` targets; with the default 12 future offsets and
    3 history frames that read 36 frames while V38 needed only 4.  This helper
    returns [B,F,1,C,P,D] so it stays compatible with
    ``TemporalWorldActionDiT.target_future_latent`` without over-reading.
    """
    batch = int(sample["state"].shape[0])
    anchors = int(getattr(model_config, "future_anchors", getattr(model_config, "num_future", 1)))
    if "target_future_dinov2_tokens" in sample:
        tokens = sample["target_future_dinov2_tokens"][:, :anchors]
        _validate_target_anchor_token_tensor(tokens, system=model_config_owner(model_config))
        return tokens.to(device=device, dtype=dtype, non_blocking=True)[:, :, None]
    if "target_history_obs_image" in sample:
        images = sample["target_history_obs_image"][:, :anchors, -1]
        flat = images.reshape(batch * anchors, *images.shape[2:])
        condition = conditioner.encode(flat, camera_names=camera_names)
    else:
        keys = sample["target_history_keys"][:, :anchors, -1, :].reshape(batch * anchors, 2)
        dummy = torch.zeros(batch * anchors, model_config.num_cameras, 3, 1, 1, dtype=torch.float32)
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("V38 future target requires dense DINO tokens")
    dense = condition.dense_tokens
    expected_tokens = model_config.num_cameras * model_config.patches_per_camera
    if dense.ndim != 3 or dense.shape[0] != batch * anchors or dense.shape[1] != expected_tokens or dense.shape[2] != model_config.latent_dim:
        raise ValueError(
            "DINO target-anchor geometry mismatch: "
            f"got {tuple(dense.shape)}, expected ({batch * anchors},{expected_tokens},{model_config.latent_dim})"
        )
    return dense.reshape(
        batch,
        anchors,
        model_config.num_cameras,
        model_config.patches_per_camera,
        model_config.latent_dim,
    ).to(device=device, dtype=dtype)[:, :, None]


def model_config_owner(model_config):
    # Small adapter so token-prefetch validation can reuse the full system-style
    # shape checker.  It intentionally exposes only policy_config.
    class _Owner:
        policy_config = model_config
    return _Owner()


@torch.no_grad()
def prepare_v39_policy_sample(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    system: V39PolicySystem,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    include_target_visual: bool = False,
) -> dict[str, Tensor]:
    if "history_dinov2_tokens" in sample:
        visual = sample["history_dinov2_tokens"]
        _validate_current_token_tensor(visual, system=system)
        visual = visual.to(device=device, dtype=dtype, non_blocking=True)
    else:
        visual = encode_current_tokens(
            sample, conditioner=conditioner, model_config=system.policy_config,
            camera_names=camera_names, device=device, dtype=dtype,
        )
    keys = (
        "state", "state_raw", "action_state", "history_state", "executed_action_history",
        "executed_action_history_raw", "policy_action", "policy_action_raw",
    )
    out = {key: sample[key].to(device=device, non_blocking=True) for key in keys}
    for key in ("state", "action_state", "history_state", "executed_action_history", "policy_action"):
        out[key] = out[key].float()
    compute_dtype = dtype if device.type == "cuda" else torch.float32
    out["visual"] = visual.to(dtype=compute_dtype)
    if include_target_visual:
        target_visual = encode_target_anchor_tokens(
            sample, conditioner=conditioner, model_config=system.policy_config,
            camera_names=camera_names, device=device, dtype=dtype,
        )
        out["target_visual"] = target_visual.to(dtype=compute_dtype)
    return out


def _effect_distance(pred: Tensor, target: Tensor) -> Tensor:
    diff = pred.float() - target.float().detach()
    mse = diff.square().mean(dim=(1, 2))
    pred_n = F.normalize(pred.float(), dim=-1)
    target_n = F.normalize(target.float().detach(), dim=-1)
    cosine = 1.0 - (pred_n * target_n).sum(dim=-1).mean(dim=1)
    return mse + 0.10 * cosine



def _future_grid_count(system_or_output: Any | None, output: dict[str, Tensor]) -> int:
    # Prefer explicit policy config when present; fall back to shape inference.
    cfg = None
    if system_or_output is not None:
        cfg = getattr(system_or_output, "policy_config", None)
    if cfg is not None:
        return int(cfg.num_cameras) * int(cfg.future_grid_size) * int(cfg.future_grid_size)
    target = output.get("rollout_effect_target")
    pred = output.get("rollout_effect_pred")
    ref = target if torch.is_tensor(target) else pred
    if not torch.is_tensor(ref):
        return 1
    # Default geometry in the V39 scripts is two cameras and a 4x4 future grid.
    return 32 if ref.shape[1] % 32 == 0 else max(int(ref.shape[1]), 1)


def _reshape_milestones(x: Tensor, *, grid: int) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"milestone tensor must be [B,K*G,H], got {tuple(x.shape)}")
    grid = max(int(grid), 1)
    usable = (int(x.shape[1]) // grid) * grid
    if usable <= 0:
        raise ValueError(f"cannot infer milestones from shape {tuple(x.shape)} and grid={grid}")
    if usable != int(x.shape[1]):
        x = x[:, :usable]
    return x.reshape(x.shape[0], usable // grid, grid, x.shape[-1])


def latent_variance_loss(output: dict[str, Tensor], *, grid: int) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros((), device=output["pred_physical_velocity"].device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    steps = min(pred.shape[1], target.shape[1])
    pred = pred[:, :steps]
    target = target[:, :steps]
    pred_std = pred.std(dim=(0, 1), unbiased=False).mean().clamp_min(1e-6)
    target_std = target.std(dim=(0, 1), unbiased=False).mean().clamp_min(1e-6)
    return (torch.log(pred_std) - torch.log(target_std)).square()


def latent_norm_loss(output: dict[str, Tensor]) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros((), device=output["pred_physical_velocity"].device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    steps = min(pred.shape[1], target.shape[1])
    pred_norm = pred[:, :steps].norm(dim=-1).mean()
    target_norm = target[:, :steps].norm(dim=-1).mean().detach()
    return F.smooth_l1_loss(pred_norm, target_norm)


def milestone_delta_match_loss(output: dict[str, Tensor], *, grid: int) -> Tensor:
    if "rollout_effect_target" not in output:
        return torch.zeros((), device=output["pred_physical_velocity"].device, dtype=output["pred_physical_velocity"].dtype)
    if "milestone_step_delta_pred" in output:
        pred_delta_flat = output["milestone_step_delta_pred"].float()
        target_delta_flat = _milestone_step_delta_target(output, grid=grid)
        steps = min(pred_delta_flat.shape[1], target_delta_flat.shape[1])
        pred_delta = pred_delta_flat[:, :steps]
        target_delta = target_delta_flat[:, :steps]
    else:
        pred = _reshape_milestones(output["rollout_effect_pred"].float(), grid=grid)
        target = _reshape_milestones(output["rollout_effect_target"].float().detach(), grid=grid)
        k = min(pred.shape[1], target.shape[1])
        pred = pred[:, :k]
        target = target[:, :k]
        z_pred = torch.zeros_like(pred[:, :1])
        z_target = torch.zeros_like(target[:, :1])
        pred_delta = pred - torch.cat([z_pred, pred[:, :-1]], dim=1)
        target_delta = target - torch.cat([z_target, target[:, :-1]], dim=1)
    smooth = F.smooth_l1_loss(pred_delta, target_delta)
    pred_n = F.normalize(pred_delta, dim=-1)
    target_n = F.normalize(target_delta, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine

def _rollout_residual_target(output: dict[str, Tensor]) -> Tensor:
    target = output["rollout_effect_target"].float().detach()
    if "rollout_base_effect_pred" not in output:
        return target
    base = output["rollout_base_effect_pred"].float().detach()
    return target - base


def _milestone_step_delta_target(output: dict[str, Tensor], *, grid: int | None = None) -> Tensor:
    """Return per-milestone target deltas aligned to V40 step-delta predictions."""
    target = _rollout_residual_target(output)
    if grid is None:
        grid = _future_grid_count(None, output)
    target_m = _reshape_milestones(target, grid=int(grid))
    z_target = torch.zeros_like(target_m[:, :1])
    target_delta = target_m - torch.cat([z_target, target_m[:, :-1]], dim=1)
    return target_delta.reshape(target_delta.shape[0], target_delta.shape[1] * target_delta.shape[2], target_delta.shape[-1])


def rollout_delta_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise the action-centered local delta.

    V40 prefers ``milestone_step_delta_pred`` when present.  That makes the
    causal branch learn per-segment action effects instead of using a cumulative
    future residual under a misleading ``delta`` name.  Older checkpoints still
    fall back to ``rollout_delta_pred`` for compatibility.
    """
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    if "milestone_step_delta_pred" in output:
        pred = output["milestone_step_delta_pred"].float()
        target = _milestone_step_delta_target(output)
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
    elif "rollout_delta_pred" in output:
        pred = output["rollout_delta_pred"].float()
        target = _rollout_residual_target(output)
    else:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine

def rollout_dynamics_loss(output: dict[str, Tensor]) -> Tensor:
    """Supervise action-conditioned rollout latent against future residual.

    The target is stop-gradient DINO future-current residual. It is never fed
    as an input to the model, so this cannot become future self-denoising.
    """
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    pred = output["rollout_effect_pred"].float()
    target = output["rollout_effect_target"].float().detach()
    smooth = F.smooth_l1_loss(pred, target)
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    cosine = (1.0 - (pred_n * target_n).sum(dim=-1)).mean()
    return smooth + 0.10 * cosine


def rollout_contrast_loss(output: dict[str, Tensor], *, margin: float = 0.02) -> Tensor:
    """Force real action-controlled local effects to beat counterfactual actions."""
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    state = None
    if "milestone_step_delta_pred_hold_action" in output and "milestone_step_delta_pred_shuffle_action" in output:
        target = _milestone_step_delta_target(output)
        real_pred = output["milestone_step_delta_pred"]
        hold_pred = output["milestone_step_delta_pred_hold_action"]
        shuf_pred = output["milestone_step_delta_pred_shuffle_action"]
        shapes = [real_pred.shape[1], hold_pred.shape[1], shuf_pred.shape[1], target.shape[1]]
        if "milestone_step_delta_pred_shuffle_state" in output:
            shapes.append(output["milestone_step_delta_pred_shuffle_state"].shape[1])
        steps = min(shapes)
        real = _effect_distance(real_pred[:, :steps], target[:, :steps])
        hold = _effect_distance(hold_pred[:, :steps], target[:, :steps])
        shuf = _effect_distance(shuf_pred[:, :steps], target[:, :steps])
        if "milestone_step_delta_pred_shuffle_state" in output:
            state = _effect_distance(output["milestone_step_delta_pred_shuffle_state"][:, :steps], target[:, :steps])
    elif "rollout_delta_pred_hold_action" in output:
        target = _rollout_residual_target(output)
        real = _effect_distance(output["rollout_delta_pred"], target)
        hold = _effect_distance(output["rollout_delta_pred_hold_action"], target)
        shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target)
        if "rollout_delta_pred_shuffle_state" in output:
            state = _effect_distance(output["rollout_delta_pred_shuffle_state"], target)
    else:
        device = output["pred_physical_velocity"].device
        return torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
    m = torch.as_tensor(float(margin), device=real.device, dtype=real.dtype)
    loss = F.relu(m + real - hold) + F.relu(m + real - shuf)
    if state is not None:
        loss = loss + F.relu(m + real - state)
    return loss.mean()

def rollout_diagnostics(output: dict[str, Tensor]) -> dict[str, Tensor]:
    rows: dict[str, Tensor] = {}
    if "rollout_effect_target" not in output:
        device = output["pred_physical_velocity"].device
        z = torch.zeros((), device=device, dtype=output["pred_physical_velocity"].dtype)
        for key in (
            "rollout_distance_real", "rollout_distance_hold", "rollout_distance_shuffle",
            "rollout_delta_hold", "rollout_delta_shuffle", "rollout_full_distance_real",
        ):
            rows[key] = z
        return rows
    target_full = output["rollout_effect_target"].float().detach()
    full_real = _effect_distance(output["rollout_effect_pred"], target_full).mean()
    rows["rollout_full_distance_real"] = full_real.detach()
    rows["rollout_distance_real"] = full_real.detach()
    rows["rollout_base_mse"] = (
        (output.get("rollout_base_effect_pred", output["rollout_effect_pred"]).float() - target_full).square().mean().detach()
    )
    rows["rollout_delta_target_norm"] = _rollout_residual_target(output).detach().float().norm(dim=-1).mean()
    if "rollout_delta_pred" in output:
        target_delta = _rollout_residual_target(output)
        delta_real = _effect_distance(output["rollout_delta_pred"], target_delta).mean()
        rows["rollout_delta_distance_real"] = delta_real.detach()
        # In V38.6 the legacy rollout_distance/delta names refer to the
        # controlled-delta contrast path because that is the causal path being
        # tested by train logs and offline diagnostics.
        rows["rollout_distance_real"] = delta_real.detach()
    pred = output["rollout_effect_pred"].float()
    target = target_full
    steps = min(pred.shape[1], target.shape[1])
    pred_s = pred[:, :steps]
    target_s = target[:, :steps]
    pred_std = pred_s.std(dim=(0, 1), unbiased=False).mean().detach()
    target_std = target_s.std(dim=(0, 1), unbiased=False).mean().detach().clamp_min(1e-6)
    pred_norm = pred_s.norm(dim=-1).mean().detach()
    target_norm = target_s.norm(dim=-1).mean().detach().clamp_min(1e-6)
    rows["rollout_pred_std_norm"] = pred_std
    rows["rollout_target_std_norm"] = target_std
    rows["rollout_pred_std_ratio"] = pred_std / target_std
    rows["rollout_pred_norm_ratio"] = pred_norm / target_norm
    grid = _future_grid_count(None, output)
    try:
        pred_m = _reshape_milestones(pred_s, grid=grid)
        target_m = _reshape_milestones(target_s, grid=grid)
        k = min(pred_m.shape[1], target_m.shape[1])
        pred_m = pred_m[:, :k]
        target_m = target_m[:, :k]
        z_pred = torch.zeros_like(pred_m[:, :1])
        z_target = torch.zeros_like(target_m[:, :1])
        pred_step = pred_m - torch.cat([z_pred, pred_m[:, :-1]], dim=1)
        target_step = target_m - torch.cat([z_target, target_m[:, :-1]], dim=1)
        pred_step_norm = pred_step.norm(dim=-1).mean().detach()
        target_step_norm = target_step.norm(dim=-1).mean().detach().clamp_min(1e-6)
        rows["rollout_milestone_delta_pred_norm"] = pred_step_norm
        rows["rollout_milestone_delta_target_norm"] = target_step_norm
        rows["rollout_milestone_delta_norm_ratio"] = pred_step_norm / target_step_norm
    except Exception:
        pass
    if "milestone_step_delta_pred_hold_action" in output:
        target_step = _milestone_step_delta_target(output)
        steps = min(output["milestone_step_delta_pred"].shape[1], target_step.shape[1])
        real_step = _effect_distance(output["milestone_step_delta_pred"][:, :steps], target_step[:, :steps]).mean()
        hold_step = _effect_distance(output["milestone_step_delta_pred_hold_action"][:, :steps], target_step[:, :steps]).mean()
        shuf_step = _effect_distance(output["milestone_step_delta_pred_shuffle_action"][:, :steps], target_step[:, :steps]).mean()
        rows["step_delta_distance_real"] = real_step.detach()
        rows["step_delta_distance_hold"] = hold_step.detach()
        rows["step_delta_distance_shuffle"] = shuf_step.detach()
        rows["step_delta_hold"] = (hold_step - real_step).detach()
        rows["step_delta_shuffle"] = (shuf_step - real_step).detach()
        rows["step_delta_change_hold"] = (output["milestone_step_delta_pred"].float() - output["milestone_step_delta_pred_hold_action"].float()).square().mean().detach()
        rows["step_delta_change_shuffle"] = (output["milestone_step_delta_pred"].float() - output["milestone_step_delta_pred_shuffle_action"].float()).square().mean().detach()
        if "milestone_step_delta_pred_shuffle_state" in output:
            state_steps = min(output["milestone_step_delta_pred_shuffle_state"].shape[1], target_step.shape[1])
            state_step = _effect_distance(output["milestone_step_delta_pred_shuffle_state"][:, :state_steps], target_step[:, :state_steps]).mean()
            rows["step_delta_state_shuffle"] = (state_step - real_step).detach()
            rows["step_delta_change_state_shuffle"] = (
                output["milestone_step_delta_pred"][:, :state_steps].float()
                - output["milestone_step_delta_pred_shuffle_state"][:, :state_steps].float()
            ).square().mean().detach()
    if "rollout_delta_pred_hold_action" in output:
        target_delta = _rollout_residual_target(output)
        hold = _effect_distance(output["rollout_delta_pred_hold_action"], target_delta).mean()
        shuf = _effect_distance(output["rollout_delta_pred_shuffle_action"], target_delta).mean()
        real = rows["rollout_distance_real"]
        rows["rollout_distance_hold"] = hold.detach()
        rows["rollout_distance_shuffle"] = shuf.detach()
        rows["rollout_delta_hold"] = (hold - real).detach()
        rows["rollout_delta_shuffle"] = (shuf - real).detach()
        rows["rollout_effect_change_hold"] = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_hold_action"].float()).square().mean().detach()
        rows["rollout_effect_change_shuffle"] = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_shuffle_action"].float()).square().mean().detach()
        rows["rollout_full_effect_change_hold"] = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_hold_action"].float()).square().mean().detach()
        rows["rollout_full_effect_change_shuffle"] = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_shuffle_action"].float()).square().mean().detach()
        if "rollout_delta_pred_shuffle_state" in output:
            state = _effect_distance(output["rollout_delta_pred_shuffle_state"], target_delta).mean()
            rows["rollout_delta_state_shuffle"] = (state - real).detach()
            rows["rollout_effect_change_state_shuffle"] = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_shuffle_state"].float()).square().mean().detach()
        if "rollout_effect_pred_shuffle_state" in output:
            rows["rollout_full_effect_change_state_shuffle"] = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_shuffle_state"].float()).square().mean().detach()
    for key in (
        "rollout_coeff_abs_mean",
        "rollout_neutral_coeff_abs_mean",
        "rollout_centered_coeff_abs_mean",
        "rollout_basis_norm",
        "rollout_delta_norm",
        "rollout_base_norm",
        "rollout_delta_gain",
        "milestone_gate_mean",
        "milestone_step_delta_norm",
        "milestone_effect_norm",
        "milestone_effect_std",
        "milestone_effect_gain",
    ):
        if key in output:
            rows[key] = output[key].detach().float()
    return rows

def future_latent_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this is the full base+delta rollout loss, not a
    # future-noisy denoising loss.
    return rollout_dynamics_loss(output)


def action_effect_loss(output: dict[str, Tensor]) -> Tensor:
    # Compatibility alias: this now tracks the controlled action delta path.
    return rollout_delta_loss(output)


def _normalize_horizon_weight(weight: Tensor) -> Tensor:
    return weight / weight.mean(dim=-1, keepdim=True).clamp_min(1e-6)


def _refine_fixed_unfolded_weights(
    *,
    system: V39PolicySystem,
    trainer: V39PolicyTrainerConfig,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fixed coarse-to-fine supervision map for intermediate refine states.

    The final refine state is anchored to the normal policy horizon weighting.
    Earlier states emphasize near-horizon trajectory quality without assigning
    any refine step to a hard physical stage.
    """
    horizon = int(system.policy_config.action_horizon)
    pos = position_weights(system.policy_config, trainer, device).to(dtype=torch.float32)
    idx = torch.arange(horizon, device=device, dtype=torch.float32)
    near = pos * (0.25 + 2.0 * (idx < 4).float() + 0.6 * (idx < 8).float())
    mid = pos * (0.35 + 1.2 * (idx < 8).float() + 0.5 * torch.exp(-((idx - 7.0).square()) / 24.0))
    tail = pos * (0.35 + 1.4 * (idx >= 8).float())
    full = pos
    basis = torch.stack([
        _normalize_horizon_weight(near),
        _normalize_horizon_weight(mid),
        _normalize_horizon_weight(tail),
        _normalize_horizon_weight(full),
    ], dim=0)

    full_mix = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    if steps <= 1:
        mix = full_mix[None]
    else:
        s = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float32)
        near_c = (1.0 - s).square()
        mid_c = torch.exp(-((s - 0.42).square()) / 0.08)
        tail_c = torch.exp(-((s - 0.72).square()) / 0.06)
        full_c = s.square()
        mix = torch.stack([near_c, mid_c, tail_c, full_c], dim=-1).clamp_min(1e-4)
        mix = mix / mix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        terminal = ((s - 0.55) / 0.45).clamp(0.0, 1.0)
        terminal = terminal.square() * (3.0 - 2.0 * terminal)
        mix = (1.0 - terminal[:, None]) * mix + terminal[:, None] * full_mix[None]
        mix[-1] = full_mix
    fixed = torch.einsum("sk,kt->st", mix, basis)
    return _normalize_horizon_weight(fixed).to(dtype=dtype), mix.to(dtype=dtype), basis.to(dtype=dtype)


def micro_refine_supervision_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    global_step: int | None = None,
) -> dict[str, Tensor]:
    pred = output.get("latent_cvae_adaptive_micro_pred_velocity")
    if not isinstance(pred, Tensor) or pred.ndim != 4:
        ref = output["pred_physical_velocity"]
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        return {
            "latent_cvae_micro_supervision": z,
            "latent_cvae_micro_event": z,
            "latent_cvae_micro_monotonic": z,
            "latent_cvae_micro_weight_kl": z,
            "latent_cvae_micro_coverage_smooth": z,
            "latent_cvae_micro_coverage_floor": z,
            "latent_cvae_micro_weight_alpha": z,
            "latent_cvae_micro_step_alpha_mean": z,
            "latent_cvae_micro_weight_final_diff": z,
            "latent_cvae_micro_coverage_tail_mass": z,
            "latent_cvae_micro_weight_first": z,
            "latent_cvae_micro_weight_last": z,
        }
    device = pred.device
    dtype = pred.dtype
    batch, steps, horizon, _ = pred.shape
    target = output["target_physical_velocity"].to(device=device, dtype=dtype)
    fixed, fixed_mix, basis = _refine_fixed_unfolded_weights(
        system=system,
        trainer=trainer,
        steps=steps,
        device=device,
        dtype=dtype,
    )
    fixed_prob = fixed.float().clamp_min(1e-8)
    fixed_prob = fixed_prob / fixed_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = basis[-1].to(device=device, dtype=dtype)
    normal_prob = normal.float().clamp_min(1e-8)
    normal_prob = normal_prob / normal_prob.sum().clamp_min(1e-8)
    prior_scale = max(float(getattr(trainer, "latent_cvae_micro_coverage_prior_logit_scale", 0.25)), 0.0)

    logits = output.get("latent_cvae_adaptive_micro_supervision_logits")
    if isinstance(logits, Tensor) and logits.ndim == 3 and int(logits.shape[1]) == steps and int(logits.shape[2]) == horizon:
        logits_f = logits.to(device=device, dtype=torch.float32)
        prior_logits = prior_scale * fixed_prob.clamp_min(1e-8).log()[None]
        learned_prob = torch.softmax(logits_f + prior_logits, dim=-1)
    elif isinstance(logits, Tensor) and logits.ndim == 3 and int(logits.shape[1]) == steps and int(logits.shape[2]) == 4:
        logits_f = logits.to(device=device, dtype=torch.float32)
        mix_prior = prior_scale * fixed_mix.float().clamp_min(1e-8).log()[None]
        learned_mix = torch.softmax(logits_f + mix_prior, dim=-1).to(dtype=dtype)
        learned = torch.einsum("bsk,kt->bst", learned_mix, basis)
        learned_prob = learned.float().clamp_min(1e-8)
        learned_prob = learned_prob / learned_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        learned_prob = fixed_prob[None].expand(batch, -1, -1)
    learned = (learned_prob * float(horizon)).to(dtype=dtype)

    ramp_steps = max(int(getattr(trainer, "latent_cvae_micro_learned_ramp_steps", 2000)), 1)
    max_alpha = max(float(getattr(trainer, "latent_cvae_micro_learned_weight_max", 0.35)), 0.0)
    step_value = 0 if global_step is None else max(int(global_step), 0)
    alpha_value = min(max_alpha, max_alpha * float(step_value) / float(ramp_steps))
    alpha = torch.as_tensor(alpha_value, device=device, dtype=dtype)
    step_alpha_scale = (1.0 - fixed_mix[:, 3]).clamp(0.0, 1.0).to(device=device, dtype=dtype)
    step_alpha = alpha * step_alpha_scale[None, :, None]
    weights = (1.0 - step_alpha) * fixed[None] + step_alpha * learned
    floor = max(float(getattr(trainer, "latent_cvae_micro_weight_floor", 0.05)), 0.0)
    weights = _normalize_horizon_weight(weights.clamp_min(floor))
    final_normal = normal[None, None].expand(batch, 1, horizon)
    weights = torch.cat([weights[:, :-1], final_normal], dim=1) if steps > 1 else final_normal

    physical_error = (pred - target[:, None]).square().mean(dim=-1)
    micro_flow = (physical_error * weights).mean()
    mono_weight = normal.to(device=device, dtype=dtype)[None, None]
    mono_error = (physical_error * mono_weight).mean(dim=-1)
    monotonic = F.relu(mono_error[:, 1:] - mono_error[:, :-1]).mean() if steps > 1 else torch.zeros((), device=device, dtype=dtype)

    event_logits = output.get("latent_cvae_adaptive_micro_event_logits")
    micro_event = torch.zeros((), device=device, dtype=dtype)
    if isinstance(event_logits, Tensor) and event_logits.ndim == 4:
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"].to(device=device),
            current_raw=sample["state_raw"].to(device=device),
            gripper_index=system.policy_config.gripper_index,
            threshold=trainer.gripper_event_threshold,
        )
        labels = labels[:, None].expand(-1, steps, -1)
        ce = F.cross_entropy(event_logits.reshape(-1, 3).float(), labels.reshape(-1), reduction="none").reshape(batch, steps, horizon)
        pos = (labels != 0).to(dtype=ce.dtype)
        event_weight = 1.0 + pos * max(float(getattr(trainer, "latent_cvae_micro_event_positive_weight", 2.0)) - 1.0, 0.0)
        denom = (event_weight.to(dtype=dtype) * weights).sum().clamp_min(1.0)
        micro_event = (ce.to(dtype=dtype) * event_weight.to(dtype=dtype) * weights).sum() / denom

    fixed_prob_b = fixed_prob[None].expand(batch, -1, -1)
    weight_kl = (
        learned_prob.clamp_min(1e-8)
        * (learned_prob.clamp_min(1e-8).log() - fixed_prob_b.clamp_min(1e-8).log())
    ).sum(dim=-1).mean().to(dtype=dtype)
    smooth = (
        (learned_prob[:, 1:] - learned_prob[:, :-1]).square().sum(dim=-1).mean() * float(horizon)
        if steps > 1 else torch.zeros((), device=device, dtype=dtype)
    )
    avg_prob = learned_prob.mean(dim=1)
    tail_start = min(max(int(getattr(system.policy_config, "rollout_tail_start_step", 8)), 0), max(horizon - 1, 0))
    tail_mask = torch.arange(horizon, device=device) >= tail_start
    tail_mass = avg_prob[:, tail_mask].sum(dim=-1)
    tail_target = normal_prob[tail_mask].sum() * max(float(getattr(trainer, "latent_cvae_micro_coverage_floor_ratio", 0.55)), 0.0)
    coverage_floor = F.relu(tail_target.to(device=device) - tail_mass).square().mean().to(dtype=dtype)
    final_weight_diff = (weights[:, -1] - normal[None]).abs().mean()
    return {
        "latent_cvae_micro_supervision": micro_flow,
        "latent_cvae_micro_event": micro_event,
        "latent_cvae_micro_monotonic": monotonic,
        "latent_cvae_micro_weight_kl": weight_kl,
        "latent_cvae_micro_coverage_smooth": smooth,
        "latent_cvae_micro_coverage_floor": coverage_floor,
        "latent_cvae_micro_weight_alpha": alpha,
        "latent_cvae_micro_step_alpha_mean": step_alpha.mean().detach(),
        "latent_cvae_micro_weight_final_diff": final_weight_diff.detach(),
        "latent_cvae_micro_coverage_tail_mass": tail_mass.detach().mean(),
        "latent_cvae_micro_weight_first": weights[:, 0, :4].mean().detach(),
        "latent_cvae_micro_weight_last": weights[:, -1].mean().detach(),
    }


def _adaptive_trajectory_basis(system: V39PolicySystem, ref: Tensor) -> Tensor | None:
    decoder = getattr(system.planner, "latent_cvae_action_decoder", None)
    velocity_head = getattr(decoder, "velocity_head", None)
    if int(getattr(system.policy_config, "latent_cvae_arm_coeff_output", 0)):
        arm_basis = getattr(velocity_head, "arm_coeff_basis", None)
        if torch.is_tensor(arm_basis) and arm_basis.ndim == 2 and int(arm_basis.numel()) > 0:
            return arm_basis.to(device=ref.device, dtype=ref.dtype)
    basis = getattr(decoder, "trajectory_basis", None)
    if not torch.is_tensor(basis):
        return None
    return basis.to(device=ref.device, dtype=ref.dtype)


def _adaptive_trajectory_analysis(system: V39PolicySystem, ref: Tensor) -> Tensor | None:
    """V53.1: shared [C, T] analysis operator from the decoder.

    Both coefficient supervisions and the model-internal projection must live
    in one coefficient space; this returns the decoder's buffer (ridge
    pseudo-inverse when latent_cvae_trajectory_pinv=1, legacy normalized
    transpose otherwise).
    """
    decoder = getattr(system.planner, "latent_cvae_action_decoder", None)
    velocity_head = getattr(decoder, "velocity_head", None)
    if int(getattr(system.policy_config, "latent_cvae_arm_coeff_output", 0)):
        arm_analysis = getattr(velocity_head, "arm_coeff_analysis", None)
        if torch.is_tensor(arm_analysis) and arm_analysis.ndim == 2 and int(arm_analysis.numel()) > 0:
            return arm_analysis.to(device=ref.device, dtype=ref.dtype)
    analysis = getattr(decoder, "trajectory_analysis", None)
    if not torch.is_tensor(analysis):
        return None
    return analysis.to(device=ref.device, dtype=ref.dtype)


def trajectory_coefficient_supervision_losses(
    system: V39PolicySystem,
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
) -> dict[str, Tensor]:
    pred = output.get("latent_cvae_adaptive_trajectory_pred_velocity")
    ref = output["pred_physical_velocity"]
    z = torch.zeros((), device=ref.device, dtype=ref.dtype)
    if not isinstance(pred, Tensor) or pred.ndim != 4:
        return {
            "latent_cvae_trajectory_supervision": z,
            "latent_cvae_trajectory_coeff_supervision": z,
            "latent_cvae_trajectory_monotonic": z,
            "latent_cvae_trajectory_mid_count": z,
            "latent_cvae_trajectory_coeff_pred_norm": z,
            "latent_cvae_trajectory_coeff_target_norm": z,
        }
    basis = _adaptive_trajectory_basis(system, pred)
    if basis is None:
        return {
            "latent_cvae_trajectory_supervision": z,
            "latent_cvae_trajectory_coeff_supervision": z,
            "latent_cvae_trajectory_monotonic": z,
            "latent_cvae_trajectory_mid_count": z,
            "latent_cvae_trajectory_coeff_pred_norm": z,
            "latent_cvae_trajectory_coeff_target_norm": z,
        }
    device = pred.device
    dtype = pred.dtype
    batch, steps, horizon, _ = pred.shape
    target = output["target_physical_velocity"].to(device=device, dtype=dtype)
    if int(target.shape[1]) != horizon:
        target = F.interpolate(
            target.transpose(1, 2),
            size=horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
    fixed, _, _ = _refine_fixed_unfolded_weights(
        system=system,
        trainer=trainer,
        steps=steps,
        device=device,
        dtype=dtype,
    )
    token_error = (pred - target[:, None]).square().mean(dim=-1)
    trajectory_flow = (token_error * fixed[None]).mean()

    analysis = _adaptive_trajectory_analysis(system, pred)
    if analysis is None:
        analysis = (basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)).transpose(0, 1)
    pred_coeff = torch.einsum("ch,bshp->bscp", analysis, pred)
    target_coeff = torch.einsum("ch,bhp->bcp", analysis, target)
    coeff_error = (pred_coeff - target_coeff[:, None]).square().mean(dim=-1)
    coeff_weight = torch.einsum("sh,hc->sc", fixed.float(), basis.float().clamp_min(0.0))
    coeff_weight = _normalize_horizon_weight(coeff_weight.clamp_min(1e-6)).to(device=device, dtype=dtype)
    coeff_flow = (coeff_error * coeff_weight[None]).mean()

    normal = position_weights(system.policy_config, trainer, device).to(dtype=dtype)
    mono_error = (token_error * normal[None, None]).mean(dim=-1)
    monotonic = F.relu(mono_error[:, 1:] - mono_error[:, :-1]).mean() if steps > 1 else z
    return {
        "latent_cvae_trajectory_supervision": trajectory_flow,
        "latent_cvae_trajectory_coeff_supervision": coeff_flow,
        "latent_cvae_trajectory_monotonic": monotonic,
        "latent_cvae_trajectory_mid_count": torch.as_tensor(float(steps), device=device, dtype=dtype),
        "latent_cvae_trajectory_coeff_pred_norm": pred_coeff.detach().float().norm(dim=-1).mean(),
        "latent_cvae_trajectory_coeff_target_norm": target_coeff.detach().float().norm(dim=-1).mean(),
    }


def _zero_proposal_residual_losses(ref: Tensor) -> dict[str, Tensor]:
    z = torch.zeros((), device=ref.device, dtype=ref.dtype)
    return {
        "latent_cvae_proposal_residual_coeff": z,
        "latent_cvae_proposal_residual_mid_coeff": z,
        "latent_cvae_proposal_residual_bound": z,
        "latent_cvae_proposal_residual_coeff_pred_norm": z,
        "latent_cvae_proposal_residual_coeff_target_norm": z,
        "latent_cvae_proposal_residual_reconstruction": z,
        "latent_cvae_proposal_residual_keep_mean": z,
        "latent_cvae_proposal_residual_target_norm": z,
    }


def _proposal_residual_physical_slice(system: V39PolicySystem, physical_dim: int, *, device: torch.device) -> Tensor:
    """Return the physical channels used by proposal-residual trajectory coefficients."""
    arm_dim = int(system.policy_config.arm_dim)
    end = min(max(2 * arm_dim, 1), int(physical_dim))
    return torch.arange(0, end, device=device, dtype=torch.long)


def proposal_residual_coefficient_losses(
    system: V39PolicySystem,
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
) -> dict[str, Tensor]:
    """Supervise denoising in explicit action-space proposal-residual coefficients.

    The target is target_physical - stopgrad(proposal_physical).  The prediction
    is reconstructed from the actual denoising path, noisy_physical - t * velocity,
    so this does not add an auxiliary answer-copying head.
    """
    ref = output["pred_physical_velocity"]
    required = ("proposal_physical", "target_physical", "clean_physical_estimate", "noisy_physical_action", "time")
    if any(key not in output for key in required):
        return _zero_proposal_residual_losses(ref)
    basis = _adaptive_trajectory_basis(system, ref)
    if basis is None:
        return _zero_proposal_residual_losses(ref)

    device = ref.device
    dtype = ref.dtype
    target = output["target_physical"].to(device=device, dtype=dtype)
    proposal = output["proposal_physical"].to(device=device, dtype=dtype).detach()
    pred_clean = output["clean_physical_estimate"].to(device=device, dtype=dtype)
    if target.shape != proposal.shape or target.shape != pred_clean.shape:
        return _zero_proposal_residual_losses(ref)

    physical_dim = int(target.shape[-1])
    if int(getattr(trainer, "latent_cvae_proposal_residual_arm_only", 1)):
        channel_index = _proposal_residual_physical_slice(system, physical_dim, device=device)
        target = target.index_select(-1, channel_index)
        proposal = proposal.index_select(-1, channel_index)
        pred_clean = pred_clean.index_select(-1, channel_index)

    target_residual = target - proposal
    pred_residual = pred_clean - proposal
    basis_f = basis.to(device=device, dtype=torch.float32)
    if int(getattr(system.policy_config, "latent_cvae_trajectory_pinv", 0)):
        # V53.1: use the decoder's shared analysis operator so proposal-residual
        # coefficients live in the same space as the model projection and the
        # trajectory coefficient supervision.
        shared = _adaptive_trajectory_analysis(system, ref)
        coeff_encoder = shared.float() if shared is not None else torch.linalg.pinv(basis_f)
    else:
        ridge = max(float(getattr(trainer, "latent_cvae_proposal_residual_coeff_ridge", 1e-2)), 0.0)
        if ridge > 0:
            gram = basis_f.transpose(0, 1) @ basis_f
            eye = torch.eye(int(gram.shape[0]), device=device, dtype=torch.float32)
            coeff_encoder = torch.linalg.solve(gram + ridge * eye, basis_f.transpose(0, 1))
        else:
            coeff_encoder = torch.linalg.pinv(basis_f)
    target_coeff = torch.einsum("ch,bhp->bcp", coeff_encoder, target_residual.float()).to(dtype=dtype)
    pred_coeff = torch.einsum("ch,bhp->bcp", coeff_encoder, pred_residual.float()).to(dtype=dtype)

    keep = output.get("proposal_keep")
    if isinstance(keep, Tensor):
        keep_w = keep.to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)
    else:
        keep_w = torch.ones(int(target.shape[0]), device=device, dtype=dtype)
    pos = position_weights(system.policy_config, trainer, device).to(device=device, dtype=dtype)
    if int(pos.shape[0]) != int(basis.shape[0]):
        pos = F.interpolate(pos[None, None], size=int(basis.shape[0]), mode="linear", align_corners=True).reshape(-1)
    coeff_weight = torch.einsum("h,hc->c", pos.float(), basis.float().clamp_min(0.0))
    coeff_weight = (coeff_weight / coeff_weight.mean().clamp_min(1e-6)).to(device=device, dtype=dtype)

    coeff_error = (pred_coeff - target_coeff).square().mean(dim=-1)
    denom = (keep_w[:, None] * coeff_weight[None]).sum().clamp_min(1.0)
    coeff_loss = (coeff_error * coeff_weight[None] * keep_w[:, None]).sum() / denom

    ratio = max(float(getattr(trainer, "latent_cvae_proposal_residual_bound_ratio", 1.25)), 1.0)
    physical_pos = position_weights(system.policy_config, trainer, device).to(device=device, dtype=dtype)
    if int(physical_pos.shape[0]) != int(target_residual.shape[1]):
        physical_pos = F.interpolate(
            physical_pos[None, None],
            size=int(target_residual.shape[1]),
            mode="linear",
            align_corners=True,
        ).reshape(-1)
    physical_pos = physical_pos / physical_pos.mean().clamp_min(1e-6)
    pred_physical_norm = pred_residual.float().norm(dim=-1)
    target_physical_norm = target_residual.detach().float().norm(dim=-1)
    bound = F.relu(pred_physical_norm - ratio * target_physical_norm).square()
    physical_denom = (keep_w[:, None] * physical_pos[None]).sum().clamp_min(1.0)
    bound_loss = (bound.to(dtype=dtype) * physical_pos[None] * keep_w[:, None]).sum() / physical_denom

    recon = torch.einsum("hc,bcp->bhp", basis.to(device=device, dtype=dtype), target_coeff)
    recon_error = (recon - target_residual).square().mean(dim=-1)
    recon_denom = (keep_w[:, None].expand_as(recon_error)).sum().clamp_min(1.0)
    recon_loss = (recon_error * keep_w[:, None]).sum() / recon_denom

    mid_pred = output.get("latent_cvae_adaptive_trajectory_pred_velocity")
    mid_loss = torch.zeros((), device=device, dtype=dtype)
    if isinstance(mid_pred, Tensor) and mid_pred.ndim == 4:
        mid_pred = mid_pred.to(device=device, dtype=dtype)
        batch, steps, horizon, _ = mid_pred.shape
        noisy = output["noisy_physical_action"].to(device=device, dtype=dtype)
        time = output["time"].to(device=device, dtype=dtype).reshape(batch, 1, 1, 1)
        if int(noisy.shape[1]) != horizon:
            noisy = F.interpolate(noisy.transpose(1, 2), size=horizon, mode="linear", align_corners=True).transpose(1, 2)
        mid_clean = noisy[:, None] - time * mid_pred
        mid_proposal = proposal
        if int(mid_proposal.shape[1]) != horizon:
            mid_proposal = F.interpolate(mid_proposal.transpose(1, 2), size=horizon, mode="linear", align_corners=True).transpose(1, 2)
        mid_residual = mid_clean
        if int(getattr(trainer, "latent_cvae_proposal_residual_arm_only", 1)):
            channel_index = _proposal_residual_physical_slice(system, physical_dim, device=device)
            mid_residual = mid_residual.index_select(-1, channel_index)
        mid_residual = mid_residual - mid_proposal[:, None]
        mid_coeff = torch.einsum("ch,bshp->bscp", coeff_encoder, mid_residual.float()).to(dtype=dtype)
        mid_error = (mid_coeff - target_coeff[:, None]).square().mean(dim=-1)
        fixed, _, _ = _refine_fixed_unfolded_weights(
            system=system,
            trainer=trainer,
            steps=steps,
            device=device,
            dtype=dtype,
        )
        step_coeff_weight = torch.einsum("sh,hc->sc", fixed.float(), basis.float().clamp_min(0.0))
        step_coeff_weight = _normalize_horizon_weight(step_coeff_weight.clamp_min(1e-6)).to(device=device, dtype=dtype)
        mid_denom = (keep_w[:, None, None] * step_coeff_weight[None]).sum().clamp_min(1.0)
        mid_loss = (mid_error * step_coeff_weight[None] * keep_w[:, None, None]).sum() / mid_denom

    return {
        "latent_cvae_proposal_residual_coeff": coeff_loss,
        "latent_cvae_proposal_residual_mid_coeff": mid_loss,
        "latent_cvae_proposal_residual_bound": bound_loss,
        "latent_cvae_proposal_residual_coeff_pred_norm": pred_coeff.detach().float().norm(dim=-1).mean(),
        "latent_cvae_proposal_residual_coeff_target_norm": target_coeff.detach().float().norm(dim=-1).mean(),
        "latent_cvae_proposal_residual_reconstruction": recon_loss.detach(),
        "latent_cvae_proposal_residual_keep_mean": keep_w.detach().float().mean(),
        "latent_cvae_proposal_residual_target_norm": target_residual.detach().float().norm(dim=-1).mean(),
    }


def _block_action_boundary_indices(system: V39PolicySystem, device: torch.device) -> Tensor | None:
    if not int(getattr(system.policy_config, "block_action_denoise_matrix", 0)):
        return None
    horizon = int(system.policy_config.action_horizon)
    rows: list[tuple[int, int]] = []
    for item in str(getattr(system.policy_config, "block_action_denoise_blocks", "")).split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        start, end = int(left), int(right)
        if 0 <= start < end <= horizon:
            rows.append((start, end))
    rows = sorted(rows)
    boundaries = [start for start, _ in rows[1:] if 0 < start < horizon]
    if not boundaries:
        return None
    return torch.as_tensor(boundaries, device=device, dtype=torch.long)


def block_action_boundary_continuity_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
) -> dict[str, Tensor]:
    ref = output["pred_physical_velocity"]
    z = torch.zeros((), device=ref.device, dtype=ref.dtype)
    boundary_idx = _block_action_boundary_indices(system, ref.device)
    if boundary_idx is None:
        return {
            "block_action_boundary_delta": z,
            "block_action_boundary_consistency": z,
            "block_action_boundary_count": z,
        }
    pred_action = output.get("pred_action_estimate")
    clean_physical = output.get("clean_physical_estimate")
    if not isinstance(pred_action, Tensor) or not isinstance(clean_physical, Tensor):
        return {
            "block_action_boundary_delta": z,
            "block_action_boundary_consistency": z,
            "block_action_boundary_count": z,
        }
    pred_action = pred_action.to(device=ref.device, dtype=ref.dtype)
    target_action = sample["policy_action"].to(device=ref.device, dtype=ref.dtype)
    prev_idx = boundary_idx - 1
    pred_delta = pred_action[:, boundary_idx] - pred_action[:, prev_idx]
    target_delta = target_action[:, boundary_idx] - target_action[:, prev_idx]
    boundary_delta = F.smooth_l1_loss(pred_delta, target_delta)

    parts = system.codec.split_physical(clean_physical.to(device=ref.device, dtype=ref.dtype))
    physical_native_delta = system.codec.join_action(parts["arm_delta"], parts["gripper_delta"])
    boundary_consistency = F.smooth_l1_loss(pred_delta, physical_native_delta[:, boundary_idx])
    return {
        "block_action_boundary_delta": boundary_delta,
        "block_action_boundary_consistency": boundary_consistency,
        "block_action_boundary_count": torch.as_tensor(float(boundary_idx.numel()), device=ref.device, dtype=ref.dtype),
    }


def flow_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
    global_step: int | None = None,
) -> dict[str, Tensor]:
    losses = v363_flow_losses(system, sample, output, trainer)  # type: ignore[arg-type]
    base_velocity = output.get("latent_cvae_adaptive_spline_base_pred_velocity")
    if isinstance(base_velocity, Tensor) and "target_physical_velocity" in output:
        pred = output["pred_physical_velocity"]
        target = output["target_physical_velocity"].to(device=pred.device, dtype=pred.dtype)
        base = base_velocity.to(device=pred.device, dtype=pred.dtype)
        if base.shape == pred.shape:
            pos_w = position_weights(system.policy_config, trainer, pred.device).to(dtype=pred.dtype)
            base_error = (base - target).square().mean(dim=-1)
            final_error = (pred - target).square().mean(dim=-1)
            base_flow = (base_error * pos_w[None]).mean()
            final_flow = (final_error * pos_w[None]).mean()
            correction = pred - base
            target_norm = target.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
            base_norm = base.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
            losses["latent_cvae_spline_base_flow"] = base_flow.detach().float()
            losses["latent_cvae_spline_final_over_base"] = (final_flow / base_flow.clamp_min(1e-8)).detach().float()
            losses["latent_cvae_spline_improvement"] = ((base_flow - final_flow) / base_flow.clamp_min(1e-8)).detach().float()
            losses["latent_cvae_spline_correction_norm"] = correction.detach().float().norm(dim=-1).mean()
            losses["latent_cvae_spline_correction_to_base"] = (correction.detach().float().norm(dim=-1).mean() / base_norm).detach().float()
            losses["latent_cvae_spline_correction_to_target"] = (correction.detach().float().norm(dim=-1).mean() / target_norm).detach().float()
    dyn = rollout_dynamics_loss(output)
    delta = rollout_delta_loss(output)
    con = rollout_contrast_loss(output, margin=float(trainer.rollout_contrast_margin))
    losses["rollout_dynamics"] = dyn
    losses["rollout_delta"] = delta
    losses["rollout_contrast"] = con
    # Compatibility log names: these no longer correspond to self-denoise.
    losses["future_latent"] = dyn.detach()
    losses["action_effect"] = delta.detach()
    if enable_future_loss and float(trainer.rollout_dynamics_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_dynamics_loss_weight) * dyn
    if enable_future_loss and float(trainer.rollout_delta_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_delta_loss_weight) * delta
    if enable_future_loss and float(trainer.rollout_contrast_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_contrast_loss_weight) * con
    # Disabled by default. Kept only as compatibility knobs; they map to the
    # same dynamics-bound target rather than future-noisy denoise.
    if enable_future_loss and float(trainer.future_latent_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.future_latent_loss_weight) * dyn
    if enable_future_loss and float(trainer.action_effect_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.action_effect_loss_weight) * dyn
    if "latent_cvae_kl" in output:
        kl = output["latent_cvae_kl"]
        weight = float(getattr(trainer, "latent_cvae_kl_weight", 0.0))
        losses["latent_cvae_kl"] = kl.detach().float()
        losses["latent_cvae_kl_weight"] = torch.as_tensor(weight, device=kl.device, dtype=kl.dtype)
        if weight > 0:
            losses["loss"] = losses["loss"] + weight * kl
    if "latent_cvae_kl" in output and "legacy_physical_velocity" in output:
        pred = output["pred_physical_velocity"]
        legacy = output["legacy_physical_velocity"].detach().to(device=pred.device, dtype=pred.dtype)
        anchor_error = (pred - legacy).square().mean(dim=-1)
        pos_w = position_weights(system.policy_config, trainer, pred.device).to(dtype=pred.dtype)
        anchor = (anchor_error * pos_w[None]).mean()
        base_weight = max(float(getattr(trainer, "latent_cvae_legacy_anchor_weight", 0.0)), 0.0)
        min_weight = min(max(float(getattr(trainer, "latent_cvae_legacy_anchor_min_weight", 0.0)), 0.0), base_weight)
        decay_steps = int(getattr(trainer, "latent_cvae_legacy_anchor_decay_steps", 0))
        step_value = 0 if global_step is None else max(int(global_step), 0)
        if decay_steps > 0:
            remain = max(1.0 - float(step_value) / float(decay_steps), 0.0)
            weight_value = min_weight + (base_weight - min_weight) * remain * remain
        else:
            weight_value = base_weight
        anchor_weight = torch.as_tensor(weight_value, device=pred.device, dtype=pred.dtype)
        losses["latent_cvae_legacy_anchor"] = anchor.detach().float()
        losses["latent_cvae_legacy_anchor_weight"] = anchor_weight.detach().float()
        flat_pred = pred.detach().float().flatten(1)
        flat_legacy = legacy.detach().float().flatten(1)
        losses["latent_cvae_legacy_cosine"] = F.cosine_similarity(flat_pred, flat_legacy, dim=-1).mean()
        losses["latent_cvae_legacy_norm_ratio"] = (
            flat_pred.norm(dim=-1) / flat_legacy.norm(dim=-1).clamp_min(1e-6)
        ).mean()
        if weight_value > 0:
            losses["loss"] = losses["loss"] + anchor_weight * anchor

    # V42.1: keep the prior/deploy path as the main output.  The posterior
    # path is a weak auxiliary reconstruction target only; this prevents the
    # target-conditioned z from becoming the only path that learns action.
    if "post_pred_velocity" in output:
        post_pred = output["post_pred_velocity"]
        post_error = (post_pred - output["target_physical_velocity"]).square().mean(dim=-1)
        pos_w = position_weights(system.policy_config, trainer, post_pred.device).to(dtype=post_pred.dtype)
        post_flow = (post_error * pos_w[None]).mean()
        losses["latent_cvae_post_flow"] = post_flow.detach().float()
        post_recon = post_flow
        if "post_pred_action_estimate" in output:
            post_decoded = F.smooth_l1_loss(output["post_pred_action_estimate"], sample["policy_action"].to(device=post_pred.device))
            losses["latent_cvae_post_decoded_action"] = post_decoded.detach().float()
            post_recon = post_recon + float(trainer.decoded_action_loss_weight) * post_decoded
        post_weight = float(getattr(trainer, "latent_cvae_posterior_recon_weight", 0.0))
        losses["latent_cvae_posterior_recon"] = post_recon.detach().float()
        losses["latent_cvae_posterior_recon_weight"] = torch.as_tensor(post_weight, device=post_pred.device, dtype=post_pred.dtype)
        if post_weight > 0:
            losses["loss"] = losses["loss"] + post_weight * post_recon
    if "latent_cvae_adaptive_regularizer" in output:
        reg = output["latent_cvae_adaptive_regularizer"]
        weight = float(getattr(trainer, "latent_cvae_adaptive_regularizer_weight", 0.0))
        losses["latent_cvae_adaptive_regularizer"] = reg.detach().float()
        losses["latent_cvae_adaptive_regularizer_weight"] = torch.as_tensor(weight, device=reg.device, dtype=reg.dtype)
        if weight > 0:
            losses["loss"] = losses["loss"] + weight * reg
    keep_ps = output.get("latent_cvae_adaptive_continue_per_sample")
    if torch.is_tensor(keep_ps) and keep_ps.ndim == 1 and int(keep_ps.shape[0]) > 1 and "target_physical_velocity" in output:
        err_ps = (
            (output["pred_physical_velocity"].detach().float() - output["target_physical_velocity"].detach().float())
            .square().mean(dim=(1, 2))
        )
        kf = keep_ps.detach().float()
        if kf.std() > 1e-8 and err_ps.std() > 1e-8:
            kc = kf - kf.mean()
            ec = err_ps - err_ps.mean()
            losses["latent_cvae_continue_error_corr"] = ((kc * ec).mean() / (kc.std() * ec.std()).clamp_min(1e-8))
        else:
            losses["latent_cvae_continue_error_corr"] = torch.zeros((), device=kf.device)
    if "latent_cvae_adaptive_noisy_ratio_regularizer" in output:
        noisy_reg = output["latent_cvae_adaptive_noisy_ratio_regularizer"]
        noisy_weight = float(getattr(trainer, "latent_cvae_noisy_ratio_weight", 0.0))
        losses["latent_cvae_noisy_ratio_regularizer"] = noisy_reg.detach().float()
        if noisy_weight > 0:
            losses["loss"] = losses["loss"] + noisy_weight * noisy_reg
    if "latent_cvae_adaptive_route_entropy_regularizer" in output:
        route_reg = output["latent_cvae_adaptive_route_entropy_regularizer"]
        route_weight = float(getattr(trainer, "latent_cvae_adaptive_route_entropy_weight", 0.0))
        losses["latent_cvae_adaptive_route_entropy_regularizer"] = route_reg.detach().float()
        losses["latent_cvae_adaptive_route_entropy_weight"] = torch.as_tensor(route_weight, device=route_reg.device, dtype=route_reg.dtype)
        if route_weight > 0:
            losses["loss"] = losses["loss"] + route_weight * route_reg
    if "block_action_denoise_regularizer" in output:
        ba_reg = output["block_action_denoise_regularizer"]
        ba_weight = float(getattr(trainer, "block_action_denoise_regularizer_weight", 0.0))
        losses["block_action_denoise_regularizer"] = ba_reg.detach().float()
        losses["block_action_denoise_regularizer_weight"] = torch.as_tensor(ba_weight, device=ba_reg.device, dtype=ba_reg.dtype)
        if ba_weight > 0:
            losses["loss"] = losses["loss"] + ba_weight * ba_reg
    boundary_losses = block_action_boundary_continuity_losses(system, sample, output)
    for key, value in boundary_losses.items():
        losses[key] = value.detach().float()
    boundary_delta_weight = float(getattr(trainer, "block_action_boundary_delta_weight", 0.0))
    boundary_consistency_weight = float(getattr(trainer, "block_action_boundary_consistency_weight", 0.0))
    if boundary_delta_weight > 0:
        losses["loss"] = losses["loss"] + boundary_delta_weight * boundary_losses["block_action_boundary_delta"]
    if boundary_consistency_weight > 0:
        losses["loss"] = losses["loss"] + boundary_consistency_weight * boundary_losses["block_action_boundary_consistency"]
    trajectory_losses = trajectory_coefficient_supervision_losses(system, output, trainer)
    for key, value in trajectory_losses.items():
        losses[key] = value.detach().float()
    trajectory_weight = float(getattr(trainer, "latent_cvae_trajectory_supervision_weight", 0.0))
    trajectory_coeff_weight = float(getattr(trainer, "latent_cvae_trajectory_coeff_weight", 0.0))
    trajectory_mono_weight = float(getattr(trainer, "latent_cvae_trajectory_monotonic_weight", 0.0))
    if trajectory_weight > 0:
        losses["loss"] = losses["loss"] + trajectory_weight * trajectory_losses["latent_cvae_trajectory_supervision"]
    if trajectory_coeff_weight > 0:
        losses["loss"] = losses["loss"] + trajectory_coeff_weight * trajectory_losses["latent_cvae_trajectory_coeff_supervision"]
    if trajectory_mono_weight > 0:
        losses["loss"] = losses["loss"] + trajectory_mono_weight * trajectory_losses["latent_cvae_trajectory_monotonic"]
    micro_losses = micro_refine_supervision_losses(system, sample, output, trainer, global_step=global_step)
    for key, value in micro_losses.items():
        losses[key] = value.detach().float()
    micro_weight = float(getattr(trainer, "latent_cvae_micro_supervision_weight", 0.0))
    micro_event_weight = float(getattr(trainer, "latent_cvae_micro_event_weight", 0.0))
    micro_mono_weight = float(getattr(trainer, "latent_cvae_micro_monotonic_weight", 0.0))
    micro_kl_weight = float(getattr(trainer, "latent_cvae_micro_weight_kl_weight", 0.0))
    micro_smooth_weight = float(getattr(trainer, "latent_cvae_micro_coverage_smooth_weight", 0.0))
    micro_floor_weight = float(getattr(trainer, "latent_cvae_micro_coverage_floor_weight", 0.0))
    if micro_weight > 0:
        losses["loss"] = losses["loss"] + micro_weight * micro_losses["latent_cvae_micro_supervision"]
    if micro_event_weight > 0:
        losses["loss"] = losses["loss"] + micro_event_weight * micro_losses["latent_cvae_micro_event"]
    if micro_mono_weight > 0:
        losses["loss"] = losses["loss"] + micro_mono_weight * micro_losses["latent_cvae_micro_monotonic"]
    if micro_kl_weight > 0:
        losses["loss"] = losses["loss"] + micro_kl_weight * micro_losses["latent_cvae_micro_weight_kl"]
    if micro_smooth_weight > 0:
        losses["loss"] = losses["loss"] + micro_smooth_weight * micro_losses["latent_cvae_micro_coverage_smooth"]
    if micro_floor_weight > 0:
        losses["loss"] = losses["loss"] + micro_floor_weight * micro_losses["latent_cvae_micro_coverage_floor"]
    proposal_residual_losses = proposal_residual_coefficient_losses(system, output, trainer)
    for key, value in proposal_residual_losses.items():
        losses[key] = value.detach().float()
    proposal_residual_coeff_weight = float(getattr(trainer, "latent_cvae_proposal_residual_coeff_weight", 0.0))
    proposal_residual_mid_weight = float(getattr(trainer, "latent_cvae_proposal_residual_mid_coeff_weight", 0.0))
    proposal_residual_bound_weight = float(getattr(trainer, "latent_cvae_proposal_residual_bound_weight", 0.0))
    if proposal_residual_coeff_weight > 0:
        losses["loss"] = losses["loss"] + proposal_residual_coeff_weight * proposal_residual_losses["latent_cvae_proposal_residual_coeff"]
    if proposal_residual_mid_weight > 0:
        losses["loss"] = losses["loss"] + proposal_residual_mid_weight * proposal_residual_losses["latent_cvae_proposal_residual_mid_coeff"]
    if proposal_residual_bound_weight > 0:
        losses["loss"] = losses["loss"] + proposal_residual_bound_weight * proposal_residual_losses["latent_cvae_proposal_residual_bound"]
    for key, attr in (
        ("latent_cvae_adaptive_trajectory_control_smoothness", "latent_cvae_trajectory_smoothness_weight"),
        ("latent_cvae_adaptive_trajectory_update_smoothness", "latent_cvae_trajectory_update_smoothness_weight"),
        ("latent_cvae_adaptive_trajectory_update_energy", "latent_cvae_trajectory_update_energy_weight"),
        ("latent_cvae_adaptive_trajectory_projection_regularizer", "latent_cvae_trajectory_projection_weight"),
    ):
        if key in output:
            value = output[key]
            weight = float(getattr(trainer, attr, 0.0))
            losses[key] = value.detach().float()
            losses[f"{key}_weight"] = torch.as_tensor(weight, device=value.device, dtype=value.dtype)
            if weight > 0:
                losses["loss"] = losses["loss"] + weight * value
    losses.update(rollout_diagnostics(output))
    if "gate_self" in output:
        losses["gate_self"] = output["gate_self"].detach()
        losses["gate_visual"] = output["gate_visual"].detach()
        losses["gate_rollout"] = output.get("gate_rollout", torch.zeros_like(output["gate_self"])).detach()
        losses["gate_ffn"] = output["gate_ffn"].detach()
    for key in (
        "mod_content_norm", "mod_time_norm", "mod_content_to_time",
        "future_conditioned_action_loss",
        "block_action_denoise_smoothness",
        "block_action_denoise_deviation",
        "block_action_denoise_interaction_norm",
        "block_action_noise_arm_mean",
        "block_action_noise_gripper_mean",
        "block_action_noise_near_mean",
        "block_action_noise_tail_mean",
        "block_action_noise_min",
        "block_action_noise_max",
        "block_action_noise_std",
        "block_action_noise_raw_rms",
        "block_action_noise_rms",
        "block_action_noise_boundary_jump",
        "block_action_loss_arm_mean",
        "block_action_loss_gripper_mean",
        "block_action_x0_near_mean",
        "block_action_x0_tail_mean",
    ):
        if key in output:
            losses[key] = output[key].detach()
    if "rollout_alpha" in output:
        losses["rollout_alpha_mean"] = output["rollout_alpha"].detach().float().mean()
    # V53.2: generic scalar diagnostics passthrough (replaces the hand-kept
    # key list; picks up every latent_cvae_* scalar the model exports).
    for key, value in output.items():
        if (
            key.startswith("latent_cvae")
            and torch.is_tensor(value)
            and value.ndim == 0
            and key not in losses
        ):
            losses[key] = value.detach().float()
    return losses



def _midcut_aux_scale(trainer: V39PolicyTrainerConfig, epoch: int) -> float:
    base = float(getattr(trainer, "midcut_aux_loss_weight", 0.0))
    if base <= 0:
        return 0.0
    decay_epochs = max(int(getattr(trainer, "midcut_aux_decay_epochs", 0)), 0)
    final_ratio = float(getattr(trainer, "midcut_aux_final_ratio", 1.0))
    final_ratio = min(max(final_ratio, 0.0), 1.0)
    if decay_epochs <= 1:
        return base * final_ratio
    progress = min(max((int(epoch) - 1) / float(decay_epochs - 1), 0.0), 1.0)
    ratio = 1.0 + (final_ratio - 1.0) * progress
    return base * ratio


def _midcut_as_primary(output: dict[str, Tensor]) -> dict[str, Tensor]:
    """Build a primary-output view using only Z_mid simple-head predictions."""
    required = ("midcut_pred_physical_velocity", "midcut_event_logits", "midcut_motion_logits")
    if not all(key in output for key in required):
        return output
    fake = dict(output)
    replacements = {
        "pred_physical_velocity": "midcut_pred_physical_velocity",
        "direct_physical_velocity": "midcut_direct_physical_velocity",
        "rollout_residual_velocity": "midcut_rollout_residual_velocity",
        "rollout_alpha": "midcut_rollout_alpha",
        "clean_physical_estimate": "midcut_clean_physical_estimate",
        "pred_action_estimate": "midcut_pred_action_estimate",
        "event_logits": "midcut_event_logits",
        "motion_logits": "midcut_motion_logits",
        "transition_latent": "midcut_transition_latent",
        "rollout_effect_pred": "midcut_rollout_effect_pred",
        "rollout_delta_pred": "midcut_rollout_delta_pred",
        "rollout_base_effect_pred": "midcut_rollout_base_effect_pred",
        "rollout_effect_pred_hold_action": "midcut_rollout_effect_pred_hold_action",
        "rollout_delta_pred_hold_action": "midcut_rollout_delta_pred_hold_action",
        "rollout_base_effect_pred_hold_action": "midcut_rollout_base_effect_pred_hold_action",
        "rollout_effect_pred_shuffle_action": "midcut_rollout_effect_pred_shuffle_action",
        "rollout_delta_pred_shuffle_action": "midcut_rollout_delta_pred_shuffle_action",
        "rollout_base_effect_pred_shuffle_action": "midcut_rollout_base_effect_pred_shuffle_action",
    }
    for dst, src in replacements.items():
        if src in output:
            fake[dst] = output[src]
    return fake


def midcut_contract_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    """Losses for the mid-cut simple-head contract.

    These are deliberately reported separately from the final policy losses so
    Stage 2 can preserve Z_mid without letting the auxiliary objective dominate
    the deployable action decoder.
    """
    fake = _midcut_as_primary(output)
    base = v363_flow_losses(system, sample, fake, trainer)  # type: ignore[arg-type]
    dyn = rollout_dynamics_loss(fake)
    delta = rollout_delta_loss(fake)
    con = rollout_contrast_loss(fake, margin=float(trainer.rollout_contrast_margin))
    total = base["loss"]
    if enable_future_loss:
        total = total + float(getattr(trainer, "midcut_rollout_dynamics_loss_weight", 0.0)) * dyn
        total = total + float(getattr(trainer, "midcut_rollout_delta_loss_weight", 0.0)) * delta
        total = total + float(getattr(trainer, "midcut_rollout_contrast_loss_weight", 0.0)) * con
    out = {f"midcut_{key}": value for key, value in base.items() if torch.is_tensor(value)}
    out["midcut_rollout_dynamics"] = dyn
    out["midcut_rollout_delta"] = delta
    out["midcut_rollout_contrast"] = con
    out["midcut_contract"] = total
    out.update({f"midcut_{key}": value for key, value in rollout_diagnostics(fake).items()})
    return out


def _uses_layer_adapter_contract(trainer: V39PolicyTrainerConfig) -> bool:
    mode = str(getattr(trainer, "contract_mode", "midcut")).lower().replace("-", "_")
    if mode in {"layer", "layers", "adapter", "layer_adapter", "multilayer", "multi_layer", "multi_layer_adapter"}:
        return True
    if mode in {"midcut", "mid_cut"}:
        return False
    raise ValueError(f"unknown contract_mode: {trainer.contract_mode!r}")


def _layer_contract_weight(index: int, count: int) -> float:
    if count <= 1:
        return 1.0
    if count == 6:
        return [0.10, 0.30, 1.00, 1.00, 0.30, 0.10][int(index)]
    if count == 4:
        return [0.30, 1.00, 1.00, 0.30][int(index)]
    if count == 3:
        return [0.30, 1.00, 0.30][int(index)]
    center = 0.5 * float(count - 1)
    distance = abs(float(index) - center) / max(center, 1.0)
    return max(0.10, 1.0 - 0.90 * distance)


def _layer_contract_as_primary(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    entry: dict[str, Tensor],
) -> dict[str, Tensor]:
    fake = dict(output)
    replacements = {
        "pred_physical_velocity": "pred_physical_velocity",
        "direct_physical_velocity": "direct_physical_velocity",
        "rollout_residual_velocity": "rollout_residual_velocity",
        "rollout_alpha": "rollout_alpha",
        "clean_physical_estimate": "clean_physical_estimate",
        "pred_action_estimate": "pred_action_estimate",
        "event_logits": "event_logits",
        "motion_logits": "motion_logits",
        "transition_latent": "transition_latent",
        "rollout_effect_pred": "rollout_effect_pred",
        "rollout_delta_pred": "rollout_delta_pred",
        "rollout_base_effect_pred": "rollout_base_effect_pred",
        "rollout_effect_pred_hold_action": "rollout_effect_pred_hold_action",
        "rollout_delta_pred_hold_action": "rollout_delta_pred_hold_action",
        "rollout_base_effect_pred_hold_action": "rollout_base_effect_pred_hold_action",
        "rollout_effect_pred_shuffle_action": "rollout_effect_pred_shuffle_action",
        "rollout_delta_pred_shuffle_action": "rollout_delta_pred_shuffle_action",
        "rollout_base_effect_pred_shuffle_action": "rollout_base_effect_pred_shuffle_action",
        "milestone_step_delta_pred": "milestone_step_delta_pred",
        "milestone_step_delta_pred_hold_action": "milestone_step_delta_pred_hold_action",
        "milestone_step_delta_pred_shuffle_action": "milestone_step_delta_pred_shuffle_action",
        "milestone_step_delta_pred_shuffle_state": "milestone_step_delta_pred_shuffle_state",
        "rollout_effect_pred_shuffle_state": "rollout_effect_pred_shuffle_state",
        "rollout_delta_pred_shuffle_state": "rollout_delta_pred_shuffle_state",
        "causal_rollout_effect_pred": "causal_rollout_effect_pred",
        "latent_rollout_effect_pred": "latent_rollout_effect_pred",
        "policy_effect_tokens": "policy_effect_tokens",
        "unified_intervention_latent_pred": "unified_intervention_latent_pred",
        "neutral_latent_pred": "neutral_latent_pred",
        "layer_causal_gain": "layer_causal_gain",
        "layer_latent_gain": "layer_latent_gain",
    }
    for dst, src in replacements.items():
        if src in entry:
            fake[dst] = entry[src]
    if "clean_physical_estimate" not in fake and "time" in output and "noisy_physical_action" in output:
        t = output["time"].to(device=fake["pred_physical_velocity"].device, dtype=fake["pred_physical_velocity"].dtype)
        clean = output["noisy_physical_action"] - t[:, None, None] * fake["pred_physical_velocity"]
        fake["clean_physical_estimate"] = clean
        fake["pred_action_estimate"] = system.codec.decode(clean, sample["action_state"].to(device=clean.device, dtype=clean.dtype))
    return fake


def layer_contract_losses(
    system: V39PolicySystem,
    sample: dict[str, Tensor],
    output: dict[str, Tensor],
    trainer: V39PolicyTrainerConfig,
    *,
    enable_future_loss: bool = True,
) -> dict[str, Tensor]:
    """Weighted multi-layer latent/action-probe contract losses.

    V39.2 makes the world/future latent head the primary per-layer objective.
    The shared flow-matching action probe is deliberately low weight and reads
    only the layer latent produced by the tiny adapter.  This avoids six full
    action heads while still testing whether each layer latent is action-usable.
    """
    layers = output.get("layer_contracts")
    if not isinstance(layers, list) or not layers:
        z = torch.zeros((), device=output["pred_physical_velocity"].device, dtype=output["pred_physical_velocity"].dtype)
        return {"loss": z, "layer_contract": z, "layer_contract_weight_sum": z}

    boost_rows: dict[str, list[Tensor]] = {"effect": [], "delta": []}
    if int(getattr(trainer, "layer_boost_residual", 0)):
        # V53-A2: telescoping residual supervision.  cum_k = detach(cum_{k-1})
        # + pred_k for the primary latent predictions and every counterfactual
        # variant, so contrast/dynamics losses stay internally consistent.
        boost_keys = ("rollout_effect_pred", "rollout_delta_pred", "milestone_step_delta_pred")
        suffixes = ("", "_hold_action", "_shuffle_action", "_shuffle_state")
        cum: dict[str, Tensor] = {}
        boosted: list[dict[str, Tensor]] = []
        for entry in layers:
            new_entry = dict(entry)
            for base_key in boost_keys:
                for suffix in suffixes:
                    key = f"{base_key}{suffix}"
                    value = entry.get(key)
                    if not torch.is_tensor(value):
                        continue
                    prev = cum.get(key)
                    if prev is not None and prev.shape == value.shape:
                        cum_value = prev.detach() + value
                    else:
                        cum_value = value
                    cum[key] = cum_value
                    new_entry[key] = cum_value
            residual_norm = entry["rollout_effect_pred"].detach().float().norm(dim=-1).mean() if torch.is_tensor(entry.get("rollout_effect_pred")) else None
            if residual_norm is not None:
                new_entry["boost_effect_residual_norm"] = residual_norm
                boost_rows["effect"].append(residual_norm)
            delta_norm = entry["rollout_delta_pred"].detach().float().norm(dim=-1).mean() if torch.is_tensor(entry.get("rollout_delta_pred")) else None
            if delta_norm is not None:
                new_entry["boost_delta_residual_norm"] = delta_norm
                boost_rows["delta"].append(delta_norm)
            boosted.append(new_entry)
        layers = boosted

    total: Tensor | None = None
    weight_sum = 0.0
    metric_acc: dict[str, Tensor] = {}
    log_rows: dict[str, Tensor] = {}
    count = len(layers)

    w_latent = float(getattr(trainer, "layer_latent_loss_weight", 1.0))
    w_fm = float(getattr(trainer, "layer_fm_probe_loss_weight", 0.02))
    w_event = float(getattr(trainer, "layer_event_loss_weight", 0.05))
    w_motion = float(getattr(trainer, "layer_motion_loss_weight", 0.03))
    w_decoded = float(getattr(trainer, "layer_decoded_action_loss_weight", 0.0))
    w_contrast = float(getattr(trainer, "layer_contrast_loss_weight", 0.0))
    w_var = float(getattr(trainer, "layer_variance_loss_weight", 0.0))
    w_norm = float(getattr(trainer, "layer_norm_loss_weight", 0.0))
    w_delta_match = float(getattr(trainer, "layer_delta_match_loss_weight", 0.0))
    grid = int(system.policy_config.num_cameras) * int(system.policy_config.future_grid_size) * int(system.policy_config.future_grid_size)

    for i, entry in enumerate(layers):
        weight = float(_layer_contract_weight(i, count))
        fake = _layer_contract_as_primary(system, sample, output, entry)
        base = v363_flow_losses(system, sample, fake, trainer)  # type: ignore[arg-type]
        dyn = rollout_dynamics_loss(fake)
        delta = rollout_delta_loss(fake)
        con = rollout_contrast_loss(fake, margin=float(trainer.rollout_contrast_margin))
        var_loss = latent_variance_loss(fake, grid=grid)
        norm_loss = latent_norm_loss(fake)
        delta_match = milestone_delta_match_loss(fake, grid=grid)

        # The latent/future objective is the primary readout contract.  It is
        # only enabled when future target tokens are present.  Action-flow is a
        # shared probe downstream of the latent and should remain lightweight.
        zero = torch.zeros_like(base["physical_flow"])
        layer_total = zero
        if enable_future_loss:
            layer_total = layer_total + w_latent * dyn
            # Delta keeps a separate action-conditioned latent residual alive;
            # default follows the existing midcut delta weight.
            layer_total = layer_total + float(getattr(trainer, "midcut_rollout_delta_loss_weight", 0.0)) * delta
            layer_total = layer_total + w_delta_match * delta_match
            layer_total = layer_total + w_var * var_loss
            layer_total = layer_total + w_norm * norm_loss
            layer_total = layer_total + w_contrast * con
        layer_total = layer_total + w_fm * base["physical_flow"]
        layer_total = layer_total + w_event * base["event"]
        layer_total = layer_total + w_motion * base["motion"]
        if w_decoded > 0:
            layer_total = layer_total + w_decoded * base["decoded_action"]

        total = layer_total * weight if total is None else total + layer_total * weight
        weight_sum += weight
        diag = rollout_diagnostics(fake)
        merged = {
            **{k: v for k, v in base.items() if torch.is_tensor(v)},
            "latent": dyn,
            "rollout_dynamics": dyn,
            "rollout_delta": delta,
            "rollout_contrast": con,
            "latent_variance": var_loss,
            "latent_norm": norm_loss,
            "milestone_delta_match": delta_match,
            **diag,
        }
        for key, value in merged.items():
            if not torch.is_tensor(value):
                continue
            log_key = "layer_base_loss" if key == "loss" else key
            metric_acc[log_key] = metric_acc.get(log_key, torch.zeros_like(value.detach().float())) + value.detach().float() * weight
        log_rows[f"layer{i}_contract"] = layer_total.detach()
        for key in (
            "latent", "rollout_dynamics", "physical_flow", "decoded_action", "event", "motion",
            "rollout_delta", "rollout_delta_shuffle", "rollout_delta_hold", "rollout_contrast",
            "rollout_effect_change_shuffle", "rollout_effect_change_hold",
            "latent_variance", "latent_norm", "milestone_delta_match",
            "rollout_pred_std_ratio", "rollout_pred_norm_ratio", "rollout_milestone_delta_norm_ratio",
            "milestone_gate_mean", "milestone_step_delta_norm", "milestone_effect_norm", "milestone_effect_std",
            "layer_causal_gain", "layer_latent_gain", "step_delta_shuffle", "step_delta_hold",
            "step_delta_change_shuffle", "step_delta_change_hold",
            "step_delta_state_shuffle", "step_delta_change_state_shuffle",
            "rollout_delta_state_shuffle", "rollout_effect_change_state_shuffle",
            "rollout_full_effect_change_state_shuffle",
        ):
            if key in merged:
                log_rows[f"layer{i}_{key}"] = merged[key].detach()
        if "boost_effect_residual_norm" in entry:
            log_rows[f"layer{i}_boost_effect_residual_norm"] = entry["boost_effect_residual_norm"]
        if "boost_delta_residual_norm" in entry:
            log_rows[f"layer{i}_boost_delta_residual_norm"] = entry["boost_delta_residual_norm"]
        if "consequence_zero_base_shift" in entry:
            log_rows[f"layer{i}_czbase"] = entry["consequence_zero_base_shift"].detach()
    assert total is not None
    denom = max(weight_sum, 1e-6)
    contract = total / denom
    out: dict[str, Tensor] = {
        "loss": contract * float(getattr(trainer, "layer_contract_loss_weight", 1.0)),
        "layer_contract": contract,
        "layer_contract_weight_sum": torch.as_tensor(weight_sum, device=contract.device, dtype=contract.dtype),
        "layer_latent_weight": torch.as_tensor(w_latent, device=contract.device, dtype=contract.dtype),
        "layer_fm_probe_weight": torch.as_tensor(w_fm, device=contract.device, dtype=contract.dtype),
        "layer_contrast_weight": torch.as_tensor(w_contrast, device=contract.device, dtype=contract.dtype),
        "layer_variance_weight": torch.as_tensor(w_var, device=contract.device, dtype=contract.dtype),
        "layer_norm_weight": torch.as_tensor(w_norm, device=contract.device, dtype=contract.dtype),
        "layer_delta_match_weight": torch.as_tensor(w_delta_match, device=contract.device, dtype=contract.dtype),
    }
    if boost_rows["effect"]:
        out["layer_boost_effect_residual_norm"] = torch.stack(boost_rows["effect"]).mean()
    if boost_rows["delta"]:
        out["layer_boost_delta_residual_norm"] = torch.stack(boost_rows["delta"]).mean()
    zero_base = [
        entry["consequence_zero_base_shift"]
        for entry in layers
        if torch.is_tensor(entry.get("consequence_zero_base_shift"))
    ]
    if zero_base:
        out["consequence_zero_base_shift"] = torch.stack(zero_base).mean()
    for key, value in metric_acc.items():
        out[key] = value / denom
    out.update(log_rows)
    return out


@torch.no_grad()
def evaluate_v39_policy(
    *,
    system: V39PolicySystem,
    loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    max_batches: int = 0,
    memory_reporter: CudaMemoryReporter | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
) -> dict[str, float]:
    system.eval()
    pred_rows, target_rows, current_rows = [], [], []
    no_proposal_rows = []
    event_logits_rows: list[np.ndarray] = []
    event_target_rows: list[np.ndarray] = []
    contract_eval = _is_contract_stage(trainer) and _uses_layer_adapter_contract(trainer)
    contract_metric_sums: dict[str, float] = {}
    contract_metric_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        report_mem = memory_reporter is not None and memory_reporter.should_report(batch_index)
        if report_mem:
            memory_reporter.reset_peak()
            if memory_reporter.detail:
                memory_reporter.snapshot(tag="eval_batch_start", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
        sample = prepare_v39_policy_sample(
            batch, conditioner=conditioner, system=system, camera_names=camera_names,
            device=device, dtype=dtype, include_target_visual=contract_eval,
        )
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(tag="eval_after_prepare", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
        generator = torch.Generator(device=device)
        generator.manual_seed(37237 + batch_index)
        # V53.5 (#4 fix): the eval starting noise must match the training
        # terminal distribution.  With the block bridge ON training draws
        # native-space noise (sample() scales+encodes it); with it OFF
        # training draws unit Gaussians directly in PHYSICAL space, so eval
        # must too.  Feeding encode(native noise) while training on physical
        # noise puts the sampler off-distribution from the first step.
        noise_dim = (
            system.policy_config.action_dim
            if int(getattr(system.policy_config, "block_action_denoise_matrix", 0))
            else system.policy_config.physical_action_dim
        )
        noise = torch.randn(
            sample["policy_action"].shape[0],
            system.policy_config.action_horizon,
            noise_dim,
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
        )
        contract_losses: dict[str, Tensor] | None = None
        with autocast_context(device, dtype):
            # Action metrics always use deploy-style sampling.  Layer-contract
            # stages additionally run a separately labelled teacher-forced
            # contract evaluation below; its values never enter action metrics.
            stop_midcut_eval = _is_contract_stage(trainer) and not _uses_layer_adapter_contract(trainer)
            pred_pack = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                action_state=sample["action_state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=True, return_event_logits=True,
                stop_at_midcut=stop_midcut_eval,
            )
            assert isinstance(pred_pack, dict)
            no_proposal = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                action_state=sample["action_state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=False,
                stop_at_midcut=stop_midcut_eval,
            )
            if contract_eval:
                contract_output = system.flow_training_forward(
                    sample["visual"], sample["history_state"], sample["executed_action_history"],
                    sample["state"], sample["policy_action"], action_state=sample["action_state"],
                    target_visual=sample["target_visual"], make_counterfactuals=True,
                    stop_at_midcut=False,
                )
                contract_losses = layer_contract_losses(
                    system, sample, contract_output, trainer, enable_future_loss=True,
                )
        if report_mem and memory_reporter.detail:
            memory_reporter.snapshot(tag="eval_after_sample", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step)
        pred_rows.append(decode(action_normalizer, pred_pack["action"]))
        no_proposal_rows.append(decode(action_normalizer, no_proposal))
        target_rows.append(sample["policy_action_raw"].cpu().numpy())
        current_rows.append(sample["state_raw"].cpu().numpy())
        labels = gripper_event_labels(
            target_raw=sample["policy_action_raw"], current_raw=sample["state_raw"],
            gripper_index=system.policy_config.gripper_index, threshold=trainer.gripper_event_threshold,
        )
        event_logits_rows.append(pred_pack["event_logits"].detach().float().cpu().numpy())
        event_target_rows.append(labels.cpu().numpy())
        if contract_losses is not None:
            for key in (
                "loss", "layer_contract", "latent", "rollout_dynamics", "rollout_delta",
                "rollout_contrast", "milestone_delta_match",
            ):
                value = contract_losses.get(key)
                if torch.is_tensor(value):
                    contract_metric_sums[key] = contract_metric_sums.get(key, 0.0) + float(value.detach().float().cpu())
            contract_metric_count += 1
        if report_mem:
            memory_reporter.snapshot(tag="eval_batch_end", phase="eval", epoch=epoch, batch=batch_index, global_step=global_step, print_line=True)
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
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean())) if squared.shape[1] > 8 else float("nan"),
        "arm_full_rmse": float(np.sqrt(squared[..., :-1].mean())),
        "gripper_full_rmse": float(np.sqrt(squared[..., -1].mean())),
        "proposal_utility_mse_gain": float(((no_proposal - target) ** 2).mean() - squared.mean()),
    }
    metrics.update(gripper_transition_metrics(
        pred, target, current, gripper_index=system.policy_config.gripper_index,
        threshold=trainer.gripper_event_threshold, tolerance=2,
    ))
    metrics.update(event_head_metrics(event_logits_rows, event_target_rows))
    metrics["tail_first_ratio"] = float(metrics["tail_rmse"] / max(metrics["first_rmse"], 1e-8))
    metrics["gripper_event_ratio"] = float(metrics.get("gripper_pred_events", 0.0) / max(metrics.get("gripper_target_events", 0.0), 1.0))
    metrics["eval_uses_target_action"] = 0.0
    metrics["eval_teacher_forced"] = 0.0
    metrics["eval_stop_at_midcut"] = float(_is_contract_stage(trainer) and not _uses_layer_adapter_contract(trainer))
    metrics["eval_layer_adapter_contract"] = float(_uses_layer_adapter_contract(trainer))
    metrics["contract_eval_teacher_forced_action"] = float(contract_eval)
    if contract_metric_count:
        for key, value in contract_metric_sums.items():
            metrics[f"contract_{key}"] = value / float(contract_metric_count)
    return metrics


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])






def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024 ** 3)


class CudaMemoryReporter:
    """Small built-in CUDA memory profiler for V38.5 training/eval.

    It records PyTorch allocator state, not an operator-level trace. It is
    cheap enough to keep disabled by default and turn on for short smoke or
    formal runs when diagnosing memory pressure.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        out_dir: Path,
        every: int = 0,
        detail: int = 0,
        sync: int = 0,
    ) -> None:
        self.device = device
        self.enabled = bool(device.type == "cuda" and int(every) > 0 and torch.cuda.is_available())
        self.every = max(int(every), 0)
        self.detail = bool(int(detail))
        self.sync = bool(int(sync))
        self.trace_path = out_dir / "v38_cuda_memory.jsonl"
        if self.enabled:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"schema": "clearvla-v38-cuda-memory-trace-v1", "event": "start", "variant": "v39_staged_midcut_contract"}, separators=(",", ":")) + "\n")

    def should_report(self, batch_index: int) -> bool:
        return self.enabled and self.every > 0 and int(batch_index) % self.every == 0

    def reset_peak(self) -> None:
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(
        self,
        *,
        tag: str,
        epoch: int | None = None,
        batch: int | None = None,
        global_step: int | None = None,
        phase: str = "train",
        print_line: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self.sync:
            torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        row: dict[str, Any] = {
            "schema": "clearvla-v38-cuda-memory-trace-v1",
            "variant": "v39_staged_midcut_contract",
            "phase": phase,
            "tag": tag,
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "allocated_gib": _bytes_to_gib(torch.cuda.memory_allocated(self.device)),
            "reserved_gib": _bytes_to_gib(torch.cuda.memory_reserved(self.device)),
            "max_allocated_gib": _bytes_to_gib(torch.cuda.max_memory_allocated(self.device)),
            "max_reserved_gib": _bytes_to_gib(torch.cuda.max_memory_reserved(self.device)),
            "free_gib": _bytes_to_gib(free_bytes),
            "total_gib": _bytes_to_gib(total_bytes),
        }
        row["used_by_context_gib"] = row["total_gib"] - row["free_gib"]
        if extra:
            row.update(extra)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")
        if print_line:
            print(
                f"[cuda-mem] phase={phase} tag={tag} epoch={epoch} batch={batch} step={global_step} "
                f"alloc={row['allocated_gib']:.3f}GiB reserved={row['reserved_gib']:.3f}GiB "
                f"peak_alloc={row['max_allocated_gib']:.3f}GiB peak_reserved={row['max_reserved_gib']:.3f}GiB "
                f"ctx_used={row['used_by_context_gib']:.3f}/{row['total_gib']:.3f}GiB",
                flush=True,
            )
        return row


def _accumulate_metric_tensors(acc: dict[str, Tensor], losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> None:
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        detached = value.detach().float()
        acc[key] = acc.get(key, torch.zeros((), device=detached.device, dtype=torch.float32)) + detached
    if grad is not None:
        g = grad.detach().float() if torch.is_tensor(grad) else torch.tensor(float(grad))
        acc["grad"] = acc.get("grad", torch.zeros((), device=g.device, dtype=torch.float32)) + g


def _finalize_metric_tensors(acc: dict[str, Tensor], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: float((value / float(count)).detach().cpu()) for key, value in acc.items()}


def _sync_loss_row(losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> dict[str, float]:
    row = {key: float(value.detach().float().cpu()) for key, value in losses.items() if torch.is_tensor(value)}
    if grad is not None:
        row["grad"] = float(grad.detach().float().cpu()) if torch.is_tensor(grad) else float(grad)
    return row


def _module_grad_norm(module: torch.nn.Module, *, reference: Tensor) -> Tensor:
    """Return a detached scalar grad norm for diagnostics only."""
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _attach_grad_diagnostics(losses: dict[str, Tensor], system: V39PolicySystem) -> None:
    """Log whether the contract objective reaches the intended modules.

    These values are diagnostics; they are added after backward and before
    optimizer.step, and never participate in the loss.
    """
    reference = losses["loss"]
    planner = system.planner
    losses["grad_dit_blocks"] = _module_grad_norm(planner.blocks, reference=reference)
    losses["grad_layer_contract_adapters"] = _module_grad_norm(planner.layer_contract_heads, reference=reference)
    if getattr(planner, "layer_fm_probe", None) is not None:
        losses["grad_layer_fm_probe"] = _module_grad_norm(planner.layer_fm_probe, reference=reference)
    if getattr(planner, "layer_consequence_cell", None) is not None:
        losses["grad_layer_consequence_cell"] = _module_grad_norm(planner.layer_consequence_cell, reference=reference)
    losses["grad_midcut_heads"] = _module_grad_norm(planner.midcut_heads, reference=reference)
    final_modules = [
        planner.final_norm,
        planner.direct_physical_head,
        planner.rollout_residual_head,
        planner.controlled_dynamics,
        planner.event_probe,
        planner.motion_probe,
    ]
    if getattr(planner, "latent_cvae_action_decoder", None) is not None:
        losses["grad_latent_cvae_action"] = _module_grad_norm(planner.latent_cvae_action_decoder, reference=reference)
    if getattr(planner, "block_action_denoise", None) is not None:
        losses["grad_block_action_denoise"] = _module_grad_norm(planner.block_action_denoise, reference=reference)
    final = torch.nn.ModuleList(final_modules)
    losses["grad_final_policy_heads"] = _module_grad_norm(final, reference=reference)


def _unique_params(modules: Sequence[torch.nn.Module]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            ident = id(param)
            if ident not in seen:
                seen.add(ident)
                params.append(param)
    return params


def _optimizer_groups(system: V39PolicySystem, trainer: V39PolicyTrainerConfig) -> list[dict[str, Any]]:
    stage = str(getattr(trainer, "training_stage", "contract")).lower().replace("-", "_")
    if stage not in {"contract", "stage1", "policy", "stage2"}:
        raise ValueError("training_stage must be contract/stage1 or policy/stage2")
    cut = int(system.policy_config.midcut_layer)
    planner = system.planner
    groups: list[dict[str, Any]] = []

    if _uses_layer_adapter_contract(trainer) and len(getattr(planner, "layer_contract_heads", [])) > 0:
        shared_modules = [planner.visual_memory, planner.rollout_codec, planner.seed, planner.time, planner.content_mod]
        depth = len(planner.blocks)
        min_scale = float(getattr(trainer, "layerwise_lr_min_scale", 0.30))
        min_scale = min(max(min_scale, 0.01), 1.0)
        if stage in {"contract", "stage1"}:
            groups.append({"params": _unique_params(shared_modules), "lr": trainer.lr * 0.5, "name": "shared_input_low_lr"})
            for i, block in enumerate(planner.blocks):
                frac = 0.0 if depth <= 1 else float(i) / float(depth - 1)
                scale = min_scale + (1.0 - min_scale) * frac
                groups.append({"params": list(block.parameters()), "lr": trainer.lr * scale, "name": f"dit_block_{i}_lr{scale:.2f}"})
            contract_modules = [planner.midcut_norm, planner.midcut_heads, planner.layer_contract_heads]
            if planner.layer_fm_probe is not None:
                contract_modules.append(planner.layer_fm_probe)
            if getattr(planner, "layer_consequence_cell", None) is not None:
                contract_modules.append(planner.layer_consequence_cell)
            event_probe_in_contract = bool(
                int(getattr(system.policy_config, "layer_causal_event_from_effect", 0))
                and float(getattr(trainer, "layer_event_loss_weight", 0.0)) > 0
            )
            if event_probe_in_contract:
                contract_modules.append(planner.event_probe)
            groups.append({"params": _unique_params(contract_modules), "lr": trainer.lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)), "name": "contract_adapters_heads"})
            final_modules = [planner.final_norm, planner.direct_physical_head, planner.rollout_residual_head, planner.controlled_dynamics, planner.motion_probe]
            if not event_probe_in_contract:
                final_modules.append(planner.event_probe)
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                final_modules.append(planner.latent_cvae_action_decoder)
            if float(getattr(trainer, "layer_contract_final_action_loss_weight", 0.0)) > 0:
                groups.append({"params": _unique_params(final_modules), "lr": trainer.lr * float(getattr(trainer, "layer_contract_final_action_lr_scale", 0.30)), "name": "weak_final_policy_probe"})
            if getattr(planner, "block_action_denoise", None) is not None:
                groups.append({
                    "params": list(planner.block_action_denoise.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "block_action_denoise_lr_scale", 1.0)),
                    "name": "block_action_denoise_matrix",
                })
            groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
        else:
            upper_lr = trainer.lr * float(getattr(trainer, "upper_lr_scale", 0.20))
            groups.append({"params": _unique_params(shared_modules), "lr": upper_lr * 0.5, "name": "shared_input_low_lr"})
            for i, block in enumerate(planner.blocks):
                frac = 0.0 if depth <= 1 else float(i) / float(depth - 1)
                lr = upper_lr + (trainer.lr - upper_lr) * frac
                lr = max(lr, trainer.lr * min_scale * 0.25)
                groups.append({"params": list(block.parameters()), "lr": lr, "name": f"dit_block_{i}_policy_layerwise"})
            inherited_contract_lr = upper_lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0))
            groups.append({
                "params": _unique_params([planner.midcut_norm, planner.midcut_heads]),
                "lr": inherited_contract_lr,
                "name": "midcut_contract_heads_low_lr",
            })
            adapter_modules = [planner.layer_contract_heads]
            if planner.layer_fm_probe is not None:
                adapter_modules.append(planner.layer_fm_probe)
            if getattr(planner, "layer_consequence_cell", None) is not None:
                adapter_modules.append(planner.layer_consequence_cell)
            adapter_lr_scale = float(getattr(trainer, "layer_contract_adapter_policy_lr_scale", 0.0))
            adapter_lr = trainer.lr * adapter_lr_scale if adapter_lr_scale > 0 else inherited_contract_lr
            adapter_name = "layer_contract_adapters_reset_lr" if adapter_lr_scale > 0 else "layer_contract_adapters_low_lr"
            groups.append({"params": _unique_params(adapter_modules), "lr": adapter_lr, "name": adapter_name})
            final_modules = [planner.final_norm, planner.direct_physical_head, planner.rollout_residual_head, planner.controlled_dynamics, planner.event_probe, planner.motion_probe]
            groups.append({"params": _unique_params(final_modules), "lr": trainer.lr, "name": "final_policy_heads"})
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                groups.append({
                    "params": list(planner.latent_cvae_action_decoder.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    "name": "latent_cvae_action_decoder",
                })
            if getattr(planner, "block_action_denoise", None) is not None:
                groups.append({
                    "params": list(planner.block_action_denoise.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "block_action_denoise_lr_scale", 1.0)),
                    "name": "block_action_denoise_matrix",
                })
            groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
        return [group for group in groups if len(group["params"]) > 0]

    pre_modules = [
        planner.visual_memory,
        planner.rollout_codec,
        planner.seed,
        planner.time,
        planner.content_mod,
        *list(planner.blocks[:cut]),
        planner.midcut_norm,
    ]
    mid_modules = [planner.midcut_heads]
    post_modules = [
        *list(planner.blocks[cut:]),
        planner.final_norm,
        planner.direct_physical_head,
        planner.rollout_residual_head,
        planner.controlled_dynamics,
        planner.event_probe,
        planner.motion_probe,
    ]
    if stage in {"contract", "stage1"}:
        groups.append({"params": _unique_params(pre_modules), "lr": trainer.lr, "name": "pre_midcut_trunk"})
        groups.append({"params": _unique_params(mid_modules), "lr": trainer.lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)), "name": "midcut_contract_heads"})
        if getattr(planner, "block_action_denoise", None) is not None:
            groups.append({
                "params": list(planner.block_action_denoise.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "block_action_denoise_lr_scale", 1.0)),
                "name": "block_action_denoise_matrix",
            })
        groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
    else:
        upper_lr = trainer.lr * float(getattr(trainer, "upper_lr_scale", 0.20))
        groups.append({"params": _unique_params(pre_modules), "lr": upper_lr, "name": "pre_midcut_trunk_low_lr"})
        groups.append({"params": _unique_params(mid_modules), "lr": upper_lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)), "name": "midcut_contract_heads_low_lr"})
        groups.append({"params": _unique_params(post_modules), "lr": trainer.lr, "name": "post_midcut_policy"})
        if getattr(planner, "latent_cvae_action_decoder", None) is not None:
            groups.append({
                "params": list(planner.latent_cvae_action_decoder.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                "name": "latent_cvae_action_decoder",
            })
        if getattr(planner, "block_action_denoise", None) is not None:
            groups.append({
                "params": list(planner.block_action_denoise.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "block_action_denoise_lr_scale", 1.0)),
                "name": "block_action_denoise_matrix",
            })
        groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
    return [group for group in groups if len(group["params"]) > 0]

def _is_contract_stage(trainer: V39PolicyTrainerConfig) -> bool:
    return str(getattr(trainer, "training_stage", "contract")).lower().replace("-", "_") in {"contract", "stage1"}

def train_v39_policy(
    *,
    system: V39PolicySystem,
    train_loader: DataLoader,
    val_loader: DataLoader,
    conditioner: RDT2Conditioner,
    device: torch.device,
    dtype: torch.dtype,
    camera_names: Sequence[str],
    action_normalizer: ArrayNormalizer,
    state_normalizer: ArrayNormalizer,
    trainer: V39PolicyTrainerConfig,
    out_dir: Path,
    context: dict[str, Any],
    resume: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    memory_reporter = CudaMemoryReporter(
        device=device,
        out_dir=out_dir,
        every=int(getattr(trainer, "memory_report_every", 0)),
        detail=int(getattr(trainer, "memory_report_detail", 0)),
        sync=int(getattr(trainer, "memory_report_sync", 0)),
    )
    system.to(device=device, dtype=torch.float32)
    if memory_reporter.enabled:
        memory_reporter.snapshot(tag="after_model_to_device", phase="setup", print_line=True)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(system, trainer),
        weight_decay=trainer.weight_decay, betas=(trainer.beta1, trainer.beta2), eps=trainer.eps,
    )
    steps_per_epoch = trainer.max_train_batches or len(train_loader)
    schedule = scheduler(optimizer, steps_per_epoch * trainer.epochs, trainer.warmup_steps, trainer.min_lr_ratio)
    start_epoch, global_step = 1, 0
    history: list[dict[str, Any]] = []
    best = {
        "full_mse": float("inf"), "gripper_f1": -float("inf"),
        "gripper_recall": -float("inf"), "balanced": float("inf"),
        "deploy_full_rmse": float("inf"), "layer_contract": float("inf"),
    }
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("schema") not in POLICY_CHECKPOINT_SCHEMAS:
            raise ValueError("resume checkpoint is not V39/V40 policy")
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
        history = list(payload.get("history", [])); best.update(payload.get("best", {})); restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train(); metric_sums: dict[str, Tensor] = {}; metric_count = 0
        include_future = (
            (float(trainer.rollout_dynamics_loss_weight) > 0 or float(trainer.rollout_contrast_loss_weight) > 0
             or float(trainer.future_latent_loss_weight) > 0 or float(trainer.action_effect_loss_weight) > 0
             or float(getattr(trainer, "layer_latent_loss_weight", 0.0)) > 0
             or float(getattr(trainer, "layer_contrast_loss_weight", 0.0)) > 0)
            and epoch >= int(trainer.future_latent_loss_start_epoch)
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            if trainer.max_train_batches and batch_index > trainer.max_train_batches:
                break
            use_future = include_future and (not trainer.future_latent_max_batches or batch_index <= trainer.future_latent_max_batches)
            report_mem = memory_reporter.should_report(batch_index)
            if report_mem:
                memory_reporter.reset_peak()
                if memory_reporter.detail:
                    memory_reporter.snapshot(tag="train_batch_start", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            sample = prepare_v39_policy_sample(
                batch, conditioner=conditioner, system=system, camera_names=camera_names, device=device, dtype=dtype,
                include_target_visual=use_future,
            )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_prepare", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            # Clear every model gradient, including parameters intentionally
            # frozen out of the current optimizer stage.  This prevents stale
            # gradients from accumulating and polluting global grad clipping.
            system.zero_grad(set_to_none=True)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_zero_grad", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            layer_mode = _uses_layer_adapter_contract(trainer)
            stop_midcut = _is_contract_stage(trainer) and not layer_mode
            with autocast_context(device, dtype):
                output = system.flow_training_forward(
                    sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"], sample["policy_action"],
                    action_state=sample["action_state"], target_visual=sample.get("target_visual"), make_counterfactuals=use_future,
                    stop_at_midcut=stop_midcut,
                )
                if _is_contract_stage(trainer) and layer_mode:
                    losses = layer_contract_losses(system, sample, output, trainer, enable_future_loss=use_future)
                    final_weight = float(getattr(trainer, "layer_contract_final_action_loss_weight", 0.0))
                    if final_weight > 0:
                        final_losses = flow_losses(system, sample, output, trainer, enable_future_loss=False, global_step=global_step)
                        losses["loss"] = losses["loss"] + final_weight * final_losses["loss"]
                        losses["final_action_probe"] = final_losses["loss"].detach()
                    losses["stage_contract"] = losses["loss"].detach()
                    losses["layer_adapter_contract"] = torch.as_tensor(1.0, device=losses["loss"].device, dtype=losses["loss"].dtype)
                elif stop_midcut:
                    losses = flow_losses(system, sample, output, trainer, enable_future_loss=use_future, global_step=global_step)
                    losses["stage_contract"] = losses["loss"].detach()
                else:
                    losses = flow_losses(system, sample, output, trainer, enable_future_loss=use_future, global_step=global_step)
                    if layer_mode:
                        aux_losses = layer_contract_losses(system, sample, output, trainer, enable_future_loss=use_future)
                        aux_key = "layer_contract"
                    else:
                        aux_losses = midcut_contract_losses(system, sample, output, trainer, enable_future_loss=use_future)
                        aux_key = "midcut_contract"
                    aux_scale = _midcut_aux_scale(trainer, epoch)
                    total_loss = losses["loss"]
                    if aux_scale > 0:
                        total_loss = total_loss + aux_scale * aux_losses[aux_key]
                    # Merge auxiliary logs without overwriting the deployable
                    # policy loss.  The previous implementation used
                    # losses.update(aux_losses), which could replace
                    # losses["loss"] with a detached/no-grad auxiliary scalar
                    # and crash backward at the start of Stage 2.  Keep other
                    # deployable-policy diagnostics under their original names
                    # as well; layer/midcut auxiliary diagnostics get prefixed
                    # when names collide so pflow remains the deploy path.
                    for key, value in aux_losses.items():
                        if key == "loss":
                            losses[f"aux_{aux_key}_loss"] = value.detach() if torch.is_tensor(value) else value
                        elif key in losses:
                            losses[f"aux_{aux_key}_{key}"] = value.detach() if torch.is_tensor(value) else value
                        else:
                            losses[key] = value
                    losses["loss"] = total_loss
                    losses["midcut_aux_scale"] = torch.as_tensor(aux_scale, device=losses["loss"].device, dtype=losses["loss"].dtype)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_forward_loss", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            losses["loss"].float().backward()
            _attach_grad_diagnostics(losses, system)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_backward", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            grad = torch.nn.utils.clip_grad_norm_(system.parameters(), trainer.grad_clip)
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_clip", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            optimizer.step(); schedule.step(); global_step += 1
            if report_mem:
                memory_reporter.snapshot(tag="train_after_step", epoch=epoch, batch=batch_index, global_step=global_step, print_line=True, extra={"use_future": bool(use_future)})
            _accumulate_metric_tensors(metric_sums, losses, grad=grad)
            metric_count += 1
            if trainer.log_every and batch_index % trainer.log_every == 0:
                row = _sync_loss_row(losses, grad=grad)
                print(
                    f"[v39-layer] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} pflowu={row.get('physical_flow_unweighted', row['physical_flow']):.6f} "
                    f"decode={row['decoded_action']:.6f} rollout={row.get('rollout_dynamics', 0.0):.6f} "
                    f"first8={row.get('first8_physical_flow', 0.0):.6f} tail={row.get('tail_physical_flow', 0.0):.6f} "
                    f"delta={row.get('rollout_delta', 0.0):.6f} contrast={row.get('rollout_contrast', 0.0):.6f} "
                    f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} "
                    f"stdr={row.get('rollout_pred_std_ratio', 0.0):.4f} dnratio={row.get('rollout_milestone_delta_norm_ratio', 0.0):.4f} "
                    f"event={row['event']:.6f} "
                    f"cz={row.get('latent_cvae_prior_z_norm', row.get('latent_cvae_z_norm', 0.0)):.2f} "
                    f"cpz={row.get('latent_cvae_post_z_norm', 0.0):.2f} "
                    f"cmug={row.get('latent_cvae_mu_gap', 0.0):.2f} "
                    f"ckl={row.get('latent_cvae_kl', 0.0):.4f} "
                    f"cpflow={row.get('latent_cvae_post_flow', 0.0):.4f} "
                    f"cstd={row.get('latent_cvae_prior_std', 0.0):.3f} "
                    f"cgate={row.get('latent_cvae_gripper_gate_mean', 0.0):.3f} "
                    f"clmem={row.get('latent_cvae_layer_memory_count', 0.0):.1f} "
                    f"cadu={row.get('latent_cvae_adaptive_refine_update_mean', 0.0):.3f} "
                    f"crmax={row.get('latent_cvae_adaptive_route_max', 0.0):.3f} "
                    f"crent={row.get('latent_cvae_adaptive_route_entropy', 0.0):.3f} "
                    f"creff={row.get('latent_cvae_adaptive_route_effective_slots', 0.0):.2f} "
                    f"cprmax={row.get('latent_cvae_adaptive_progress_max', 0.0):.3f} "
                    f"cprent={row.get('latent_cvae_adaptive_progress_entropy', 0.0):.3f} "
                    f"cpeff={row.get('latent_cvae_adaptive_progress_effective_slots', 0.0):.2f} "
                    f"cprog={row.get('latent_cvae_adaptive_progress_norm', 0.0):.2f} "
                    f"ccont={row.get('latent_cvae_adaptive_continue_mean', 0.0):.3f} "
                    f"ccstd={row.get('latent_cvae_adaptive_continue_std', 0.0):.3f} "
                    f"ccf={row.get('latent_cvae_adaptive_continue_first', 0.0):.3f} "
                    f"ccl={row.get('latent_cvae_adaptive_continue_last', 0.0):.3f} "
                    f"cctc={row.get('latent_cvae_adaptive_continue_time_corr', 0.0):+.3f} "
                    f"ccec={row.get('latent_cvae_continue_error_corr', 0.0):+.3f} "
                    f"cprefix={row.get('latent_cvae_adaptive_prefix_norm', 0.0):.2f} "
                    f"czseed={row.get('latent_cvae_adaptive_progress_seed_norm', 0.0):.3f} "
                    f"czseff={row.get('latent_cvae_adaptive_progress_seed_effective_slots', 0.0):.2f} "
                    f"ctemp={row.get('latent_cvae_adaptive_route_temperature_mean', 0.0):.2f} "
                    f"cfunc={row.get('latent_cvae_adaptive_function_delta_norm', 0.0):.3f} "
                    f"cbasehf={row.get('latent_cvae_adaptive_base_highfreq_norm', 0.0):.3f} "
                    f"cwctl={row.get('latent_cvae_adaptive_coeff_writer_controls', 0.0):.1f} "
                    f"cwgrip={row.get('latent_cvae_adaptive_coeff_writer_include_gripper', 0.0):.0f} "
                    f"cwdir={row.get('latent_cvae_adaptive_coeff_writer_direction_norm', 0.0):.3f} "
                    f"cwraw={row.get('latent_cvae_adaptive_coeff_writer_raw_direction_norm', 0.0):.3f} "
                    f"cwmag={row.get('latent_cvae_adaptive_coeff_writer_magnitude_mean', 0.0):.3f} "
                    f"cwcond={row.get('latent_cvae_adaptive_coeff_writer_condition_norm', 0.0):.3f} "
                    f"cwcgate={row.get('latent_cvae_adaptive_coeff_writer_condition_gate', 0.0):.3f} "
                    f"csbf={row.get('latent_cvae_spline_base_flow', 0.0):.4f} "
                    f"csim={row.get('latent_cvae_spline_improvement', 0.0):+.3f} "
                    f"cscorr={row.get('latent_cvae_spline_correction_to_base', 0.0):.3f} "
                    f"cmdblk={row.get('latent_cvae_adaptive_detail_micro_block_norm', 0.0):.3f} "
                    f"cmdet={row.get('latent_cvae_adaptive_detail_micro_update_norm', 0.0):.3f} "
                    f"cmdgate={row.get('latent_cvae_adaptive_detail_micro_gate_mean', 0.0):.3f} "
                    f"cmdraw={row.get('latent_cvae_adaptive_detail_micro_raw_norm', 0.0):.3f} "
                    f"cmsup={row.get('latent_cvae_micro_supervision', 0.0):.4f} "
                    f"cmevt={row.get('latent_cvae_micro_event', 0.0):.4f} "
                    f"cmmono={row.get('latent_cvae_micro_monotonic', 0.0):.4f} "
                    f"cmwa={row.get('latent_cvae_micro_weight_alpha', 0.0):.3f} "
                    f"cmwfd={row.get('latent_cvae_micro_weight_final_diff', 0.0):.4f} "
                    f"cmwkl={row.get('latent_cvae_micro_weight_kl', 0.0):.4f} "
                    f"cmcs={row.get('latent_cvae_micro_coverage_smooth', 0.0):.4f} "
                    f"cmcf={row.get('latent_cvae_micro_coverage_floor', 0.0):.4f} "
                    f"cmtail={row.get('latent_cvae_micro_coverage_tail_mass', 0.0):.3f} "
                    f"ctctrl={row.get('latent_cvae_adaptive_trajectory_control_norm', 0.0):.3f} "
                    f"ctok={row.get('latent_cvae_adaptive_trajectory_token_norm', 0.0):.3f} "
                    f"ctupd={row.get('latent_cvae_adaptive_trajectory_update_norm', 0.0):.3f} "
                    f"ctctx={row.get('latent_cvae_adaptive_trajectory_context_norm', 0.0):.3f} "
                    f"cterr={row.get('latent_cvae_adaptive_trajectory_projection_error', 0.0):.3f} "
                    f"ctsup={row.get('latent_cvae_trajectory_supervision', 0.0):.4f} "
                    f"ctcoef={row.get('latent_cvae_trajectory_coeff_supervision', 0.0):.4f} "
                    f"ctmono={row.get('latent_cvae_trajectory_monotonic', 0.0):.4f} "
                    f"prcoef={row.get('latent_cvae_proposal_residual_coeff', 0.0):.4f} "
                    f"prmid={row.get('latent_cvae_proposal_residual_mid_coeff', 0.0):.4f} "
                    f"prbd={row.get('latent_cvae_proposal_residual_bound', 0.0):.4f} "
                    f"prn={row.get('latent_cvae_proposal_residual_coeff_pred_norm', 0.0):.3f}/"
                    f"{row.get('latent_cvae_proposal_residual_coeff_target_norm', 0.0):.3f} "
                    f"prkeep={row.get('latent_cvae_proposal_residual_keep_mean', 0.0):.2f} "
                    f"ctsm={row.get('latent_cvae_adaptive_trajectory_control_smoothness', 0.0):.4f} "
                    f"ctusm={row.get('latent_cvae_adaptive_trajectory_update_smoothness', 0.0):.4f} "
                    f"ctue={row.get('latent_cvae_adaptive_trajectory_update_energy', 0.0):.4f} "
                    f"ctpr={row.get('latent_cvae_adaptive_trajectory_projection_regularizer', 0.0):.4f} "
                    f"cstep={row.get('latent_cvae_adaptive_refine_step_bias_norm', 0.0):.3f} "
                    f"ccmax={row.get('latent_cvae_adaptive_capsule_layer_max', 0.0):.3f} "
                    f"ccleff={row.get('latent_cvae_adaptive_capsule_layer_effective_slots', 0.0):.2f} "
                    f"careg={row.get('latent_cvae_adaptive_regularizer', 0.0):.4f} "
                    f"carent={row.get('latent_cvae_adaptive_route_entropy_regularizer', 0.0):.4f} "
                    f"cxgate={row.get('latent_cvae_adaptive_noisy_gate_mean', 0.0):.3f} "
                    f"xnorm={row.get('latent_cvae_adaptive_noisy_branch_norm', 0.0):.3f} "
                    f"xratio={row.get('latent_cvae_adaptive_noisy_branch_ratio', 0.0):.3f} "
                    f"cscan={row.get('latent_cvae_condition_scan_norm', 0.0):.2f} "
                    f"clat={row.get('latent_cvae_condition_lateral_norm', 0.0):.2f} "
                    f"ccanv={row.get('latent_cvae_adaptive_canvas_cross_norm', 0.0):.3f} "
                    f"cvgate={row.get('latent_cvae_adaptive_canvas_gate', 0.0):.3f} "
                    f"czbase={row.get('consequence_zero_base_shift', 0.0):.3f} "
                    f"lboost={row.get('layer_boost_effect_residual_norm', 0.0):.3f} "
                    f"ldres={row.get('layer_boost_delta_residual_norm', 0.0):.3f} "
                    f"cgrad={row.get('grad_latent_cvae_action', 0.0):.3e} "
                    f"grad={row['grad']:.3e} lr={optimizer.param_groups[0]['lr']:.3e}",
                    flush=True,
                )
        train_metrics = _finalize_metric_tensors(metric_sums, metric_count)
        val_metrics = evaluate_v39_policy(
            system=system, loader=val_loader, conditioner=conditioner, device=device, dtype=dtype,
            camera_names=camera_names, action_normalizer=action_normalizer, trainer=trainer,
            max_batches=trainer.max_val_batches, memory_reporter=memory_reporter, epoch=epoch, global_step=global_step,
        )
        score = balanced_score(val_metrics, trainer)  # type: ignore[arg-type]
        deploy_eligible = is_deploy_eligible(val_metrics, trainer)  # type: ignore[arg-type]
        val_metrics["balanced_score"] = score
        val_metrics["deploy_eligible"] = float(deploy_eligible)
        record = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "val": val_metrics}
        history.append(record)
        with (out_dir / "v39_policy_epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")
        full = float(val_metrics["full_mse"])
        f1 = float(val_metrics.get("gripper_f1", 0.0))
        recall = float(val_metrics.get("gripper_recall", 0.0))
        save = []
        select_contract = _is_contract_stage(trainer) and _uses_layer_adapter_contract(trainer)
        if select_contract:
            contract_value = float(val_metrics.get("contract_layer_contract", float("inf")))
            if contract_value < best["layer_contract"]:
                best["layer_contract"] = contract_value
                save.append("best_contract.pt")
        else:
            if full < best["full_mse"]:
                best["full_mse"] = full; save.append("best_full.pt")
            if f1 > best["gripper_f1"]:
                best["gripper_f1"] = f1; save.append("best_gripper_f1.pt")
            if recall > best["gripper_recall"]:
                best["gripper_recall"] = recall; save.append("best_gripper_recall.pt")
            if score < best["balanced"]:
                best["balanced"] = score; save.append("best_balanced.pt")
            if deploy_eligible and float(val_metrics["full_rmse"]) < best["deploy_full_rmse"]:
                best["deploy_full_rmse"] = float(val_metrics["full_rmse"]); save.append("best_deploy.pt")
        payload = {
            "schema": "clearvla-v40-policy-checkpoint-v1", "epoch": epoch, "global_step": global_step,
            "model": system.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
            "policy_config": asdict(system.policy_config), "trainer_config": asdict(trainer),
            "action_normalizer": action_normalizer.to_dict(), "state_normalizer": state_normalizer.to_dict(),
            "context": context, "history": history, "best": best, "rng": rng_state(),
        }
        for name in save:
            torch.save(payload, ckpt_dir / name)
        torch.save(payload, ckpt_dir / "latest.pt")
        (out_dir / "v40_policy_summary.json").write_text(json.dumps(jsonable({"schema": "clearvla-v40-policy-summary-v1", "best": best, "latest": record}), indent=2), encoding="utf-8")
        print(json.dumps(jsonable(record), separators=(",", ":")), flush=True)
    return {"history": history, "best": best}


__all__ = [
    "POLICY_CHECKPOINT_SCHEMAS",
    "V39PolicyTrainerConfig",
    "prepare_v39_policy_sample",
    "encode_target_anchor_tokens",
    "future_latent_loss",
    "action_effect_loss",
    "rollout_dynamics_loss",
    "rollout_delta_loss",
    "rollout_contrast_loss",
    "flow_losses",
    "layer_contract_losses",
    "evaluate_v39_policy",
    "CudaMemoryReporter",
    "train_v39_policy",
]
