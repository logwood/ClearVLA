from __future__ import annotations

"""Training/evaluation runtime for V39 staged mid-cut latent contract policy."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    semantic_physical_velocity_error,
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
    # Main controlled rollout must match not only the future direction but also
    # its energy and inter-milestone change. These mirror the proven layer
    # consequence contracts and prevent a low-amplitude conditional mean from
    # satisfying the world objective.
    rollout_variance_loss_weight: float = 0.05
    rollout_norm_loss_weight: float = 0.02
    rollout_milestone_delta_match_weight: float = 0.15
    # Local fuse for the complete CVAE/MMDiT decoder. Architectural pre-norm is
    # the primary protection; this prevents a decoder-only spike from consuming
    # the global clipping budget and starving the world/rollout trunk.
    latent_cvae_grad_clip: float = 0.0
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

    # Optional safe residual action-flow denoiser.  This module is random at
    # first load but zero-starts behaviorally, so it gets a distinct LR group
    # in policy stage instead of being hidden inside the large trunk group.
    action_flow_residual_lr_scale: float = 1.5
    latent_action_decoder_lr_scale: float = 1.5
    # V42 compact latent-CVAE head.  Small KL keeps q(z|latent,target) close
    # to the conditional prior used at inference without letting KL dominate
    # the action flow/chunk reconstruction objective.
    latent_cvae_action_decoder_lr_scale: float = 1.0
    # The complete V77 host block uses the ordinary decoder rate. Only
    # orthogonal sidecar bases and depth predictors use this rate.
    hierarchical_mmdit_contraction_lr_scale: float = 2.0
    hierarchical_mmdit_shared_base_lr_scale: float = 1.0
    # Weak cost on the selected nested depth.  It begins only after the exact
    # identity warm-up and cannot change basis scale or residual amplitude.
    hierarchical_mmdit_depth_usage_loss_weight: float = 2e-4
    # Target-aware supervision is confined to the training loss. Candidate
    # errors are detached, so only the read-only exit controller receives it.
    hierarchical_mmdit_oracle_route_loss_weight: float = 0.0
    hierarchical_mmdit_oracle_route_relative_tolerance: float = 0.0
    hierarchical_mmdit_oracle_route_warmup_steps: int = 200
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
    latent_cvae_micro_supervision_weight: float = 0.08
    latent_cvae_micro_event_weight: float = 0.02
    latent_cvae_micro_monotonic_weight: float = 0.02
    latent_cvae_micro_weight_kl_weight: float = 0.001
    latent_cvae_micro_coverage_smooth_weight: float = 0.002
    latent_cvae_micro_coverage_floor_weight: float = 0.002
    latent_cvae_micro_coverage_prior_logit_scale: float = 0.25
    latent_cvae_micro_coverage_floor_ratio: float = 0.55
    latent_cvae_micro_learned_weight_max: float = 0.40
    latent_cvae_micro_learned_ramp_steps: int = 2000
    latent_cvae_micro_weight_floor: float = 0.05
    latent_cvae_micro_event_positive_weight: float = 2.0

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
        delta_change_hold = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_hold_action"].float()).square().mean()
        delta_change_shuffle = (output["rollout_delta_pred"].float() - output["rollout_delta_pred_shuffle_action"].float()).square().mean()
        full_change_hold = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_hold_action"].float()).square().mean()
        full_change_shuffle = (output["rollout_effect_pred"].float() - output["rollout_effect_pred_shuffle_action"].float()).square().mean()
        rows["rollout_effect_change_hold"] = delta_change_hold.detach()
        rows["rollout_effect_change_shuffle"] = delta_change_shuffle.detach()
        rows["rollout_full_effect_change_hold"] = full_change_hold.detach()
        rows["rollout_full_effect_change_shuffle"] = full_change_shuffle.detach()
        rows["rollout_hold_cancellation_fraction"] = torch.where(
            delta_change_hold > 1e-8,
            1.0 - full_change_hold / delta_change_hold.clamp_min(1e-8),
            torch.zeros_like(delta_change_hold),
        ).detach()
        rows["rollout_shuffle_cancellation_fraction"] = torch.where(
            delta_change_shuffle > 1e-8,
            1.0 - full_change_shuffle / delta_change_shuffle.clamp_min(1e-8),
            torch.zeros_like(delta_change_shuffle),
        ).detach()
        if "rollout_base_effect_pred" in output:
            base = output["rollout_base_effect_pred"].float()
            rows["rollout_effect_delta_gap"] = (
                output["rollout_effect_pred"].float() - output["rollout_delta_pred"].float()
            ).square().mean().detach()
            if "rollout_base_effect_pred_hold_action" in output:
                rows["rollout_base_change_hold"] = (
                    base - output["rollout_base_effect_pred_hold_action"].float()
                ).square().mean().detach()
            if "rollout_base_effect_pred_shuffle_action" in output:
                rows["rollout_base_change_shuffle"] = (
                    base - output["rollout_base_effect_pred_shuffle_action"].float()
                ).square().mean().detach()
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
        "rollout_decomposition_expansion_ratio",
        "rollout_base_is_fixed_zero",
        "rollout_delta_gain",
        "rollout_deep_update_norm",
        "rollout_deep_token_norm",
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


def _micro_fixed_unfolded_weights(
    *,
    system: V39PolicySystem,
    trainer: V39PolicyTrainerConfig,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fixed coarse-to-fine supervision map for micro refine states.

    The final micro state is anchored to the normal policy horizon weighting.
    Earlier states emphasize near-horizon trajectory quality without assigning
    any micro step to a hard physical stage.
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
        tail_c = torch.exp(-((s - 0.72).square()) / 0.06) * (1.0 - 0.7 * s)
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
        }
    device = pred.device
    dtype = pred.dtype
    batch, steps, horizon, _ = pred.shape
    target = output["target_physical_velocity"].to(device=device, dtype=dtype)
    fixed, fixed_mix, basis = _micro_fixed_unfolded_weights(
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
    max_alpha = max(float(getattr(trainer, "latent_cvae_micro_learned_weight_max", 0.25)), 0.0)
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

    physical_error = semantic_physical_velocity_error(
        system,
        pred - target[:, None],
        arm_null_weight=trainer.arm_manifold_null_weight,
    )
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
        micro_event = (ce.to(dtype=dtype) * event_weight.to(dtype=dtype) * weights).sum() / (event_weight.to(dtype=dtype) * weights).sum().clamp_min(1.0)

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

def _shadow_refinement_probe_metrics(
    system: V39PolicySystem,
    output: dict[str, Tensor],
    *,
    arm_null_weight: float,
) -> dict[str, Tensor]:
    """Measure shadow early-exit quality without feeding target error to routing."""
    shadow_predictions = output.get("refinement_shadow_probe_pred_velocity")
    shadow_active = output.get("refinement_shadow_probe_active")
    fixed_predictions = output.get("refinement_probe_pred_velocity")
    fixed_active = output.get("refinement_probe_active")
    target_velocity = output.get("target_physical_velocity")
    if not all(torch.is_tensor(value) for value in (
        shadow_predictions, shadow_active, fixed_predictions, fixed_active, target_velocity,
    )):
        return {}

    def final_error(predictions: Tensor, active: Tensor) -> Tensor:
        residual = predictions.float() - target_velocity.detach().float()[:, None]
        step_error = semantic_physical_velocity_error(
            system,
            residual,
            arm_null_weight=arm_null_weight,
        ).mean(dim=-1)
        final_index = active.float().sum(dim=1).long().clamp_min(1)
        return step_error.gather(1, final_index[:, None]).squeeze(1)

    with torch.no_grad():
        shadow_error = final_error(shadow_predictions, shadow_active)
        fixed_error = final_error(fixed_predictions, fixed_active)
        shadow_steps = shadow_active.float().sum(dim=1)
        fixed_steps = fixed_active.float().sum(dim=1)
        return {
            "hierarchical_mmdit_shadow_refine_error_final": shadow_error.mean(),
            "hierarchical_mmdit_shadow_refine_error_gap": (
                shadow_error - fixed_error
            ).mean(),
            "hierarchical_mmdit_shadow_refine_error_ratio": (
                shadow_error / fixed_error.clamp_min(1e-8)
            ).mean(),
            "hierarchical_mmdit_shadow_step_saving": (
                fixed_steps - shadow_steps
            ).mean(),
        }


def _oracle_exit_supervision(
    *,
    exit_logits: Tensor,
    candidate_error: Tensor,
    initial_error: Tensor,
    candidate_mask: Tensor,
    relative_tolerance: float,
) -> dict[str, Tensor]:
    """Train an online stop/continue head from detached full-prefix errors."""
    if exit_logits.ndim != 2:
        raise ValueError("exit_logits must be [B,S]")
    expected = tuple(exit_logits.shape)
    if tuple(candidate_error.shape) != expected or tuple(candidate_mask.shape) != expected:
        raise ValueError("oracle route tensors must share [B,S] geometry")
    if tuple(initial_error.shape) != (expected[0],):
        raise ValueError("initial_error must be [B]")
    if float(relative_tolerance) < 0.0:
        raise ValueError("relative_tolerance must be non-negative")

    device = exit_logits.device
    steps = int(exit_logits.shape[1])
    indices = torch.arange(steps, device=device, dtype=torch.long)[None]
    mask = candidate_mask.detach().bool()
    errors = candidate_error.detach().float()
    initial = initial_error.detach().float().abs().clamp_min(1e-6)
    valid = mask.any(dim=1)
    valid_float = valid.float()
    valid_denominator = valid_float.sum().clamp_min(1.0)

    masked_error = torch.where(mask, errors, torch.full_like(errors, float("inf")))
    best_error = masked_error.min(dim=1).values
    safe_best_error = torch.where(valid, best_error, torch.zeros_like(best_error))
    tolerance = float(relative_tolerance) * initial
    near_best = mask & (errors <= best_error[:, None] + tolerance[:, None])
    oracle_index = near_best.float().argmax(dim=1)

    decision_mask = mask & (indices <= oracle_index[:, None])
    stop_target = (indices == oracle_index[:, None]).to(dtype=exit_logits.dtype)
    decision_float = decision_mask.to(dtype=exit_logits.dtype)
    route_loss = (
        F.binary_cross_entropy_with_logits(
            exit_logits.float(), stop_target.float(), reduction="none"
        ) * decision_float.float()
    ).sum() / decision_float.float().sum().clamp_min(1.0)

    with torch.no_grad():
        probabilities = torch.sigmoid(exit_logits.detach().float())
        candidate_depth = mask.long().cumsum(dim=1)
        predicted_stop = mask & (probabilities > 0.5)
        sentinel = torch.full_like(indices, steps)
        first_predicted = torch.where(predicted_stop, indices, sentinel).min(dim=1).values
        last_candidate = torch.where(mask, indices, torch.full_like(indices, -1)).max(dim=1).values
        predicted_index = torch.where(
            first_predicted < steps,
            first_predicted,
            last_candidate.clamp_min(0),
        )
        predicted_error = errors.gather(1, predicted_index[:, None]).squeeze(1)
        oracle_error = errors.gather(1, oracle_index[:, None]).squeeze(1)
        oracle_depth = candidate_depth.gather(
            1, oracle_index[:, None]
        ).squeeze(1).float()
        predicted_depth = candidate_depth.gather(
            1, predicted_index[:, None]
        ).squeeze(1).float()
        mean_probability = (
            probabilities * mask.float()
        ).sum() / mask.float().sum().clamp_min(1.0)

    return {
        "loss": route_loss,
        "target_depth": (oracle_depth * valid_float).sum() / valid_denominator,
        "predicted_depth": (predicted_depth * valid_float).sum() / valid_denominator,
        "depth_accuracy": (
            (predicted_index == oracle_index).float() * valid_float
        ).sum() / valid_denominator,
        "depth_mae": (
            (predicted_depth - oracle_depth).abs() * valid_float
        ).sum() / valid_denominator,
        "best_error": (safe_best_error * valid_float).sum() / valid_denominator,
        "target_error": (oracle_error * valid_float).sum() / valid_denominator,
        "predicted_error": (predicted_error * valid_float).sum() / valid_denominator,
        "predicted_regret": (
            (predicted_error - safe_best_error).clamp_min(0.0) * valid_float
        ).sum() / valid_denominator,
        "stop_probability": mean_probability,
        "valid_fraction": valid_float.mean(),
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
    dyn = rollout_dynamics_loss(output)
    delta = rollout_delta_loss(output)
    con = rollout_contrast_loss(output, margin=float(trainer.rollout_contrast_margin))
    grid = _future_grid_count(system, output)
    rollout_var = latent_variance_loss(output, grid=grid)
    rollout_norm = latent_norm_loss(output)
    rollout_milestone = milestone_delta_match_loss(output, grid=grid)
    losses["rollout_dynamics"] = dyn
    losses["rollout_delta"] = delta
    losses["rollout_contrast"] = con
    losses["rollout_variance"] = rollout_var
    losses["rollout_norm"] = rollout_norm
    losses["rollout_milestone_delta_match"] = rollout_milestone
    # Compatibility log names: these no longer correspond to self-denoise.
    losses["future_latent"] = dyn.detach()
    losses["action_effect"] = delta.detach()
    if enable_future_loss and float(trainer.rollout_dynamics_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_dynamics_loss_weight) * dyn
    if enable_future_loss and float(trainer.rollout_delta_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_delta_loss_weight) * delta
    if enable_future_loss and float(trainer.rollout_contrast_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_contrast_loss_weight) * con
    if enable_future_loss and float(trainer.rollout_variance_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_variance_loss_weight) * rollout_var
    if enable_future_loss and float(trainer.rollout_norm_loss_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_norm_loss_weight) * rollout_norm
    if enable_future_loss and float(trainer.rollout_milestone_delta_match_weight) > 0:
        losses["loss"] = losses["loss"] + float(trainer.rollout_milestone_delta_match_weight) * rollout_milestone
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
        anchor_error = semantic_physical_velocity_error(
            system,
            pred - legacy,
            arm_null_weight=trainer.arm_manifold_null_weight,
        )
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
        post_error = semantic_physical_velocity_error(
            system,
            post_pred - output["target_physical_velocity"],
            arm_null_weight=trainer.arm_manifold_null_weight,
        )
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
    if "latent_cvae_adaptive_route_entropy_regularizer" in output:
        route_reg = output["latent_cvae_adaptive_route_entropy_regularizer"]
        route_weight = float(getattr(trainer, "latent_cvae_adaptive_route_entropy_weight", 0.0))
        losses["latent_cvae_adaptive_route_entropy_regularizer"] = route_reg.detach().float()
        losses["latent_cvae_adaptive_route_entropy_weight"] = torch.as_tensor(route_weight, device=route_reg.device, dtype=route_reg.dtype)
        if route_weight > 0:
            losses["loss"] = losses["loss"] + route_weight * route_reg
    if "hierarchical_mmdit_depth_usage_regularizer" in output:
        depth_reg = output["hierarchical_mmdit_depth_usage_regularizer"]
        depth_weight = float(
            getattr(trainer, "hierarchical_mmdit_depth_usage_loss_weight", 0.0)
        )
        losses["hierarchical_mmdit_depth_usage_regularizer"] = depth_reg.detach().float()
        losses["hierarchical_mmdit_depth_usage_loss_weight"] = torch.as_tensor(
            depth_weight, device=depth_reg.device, dtype=depth_reg.dtype
        )
        if depth_weight > 0.0:
            losses["loss"] = losses["loss"] + depth_weight * depth_reg
    exit_logits = output.get("hierarchical_mmdit_exit_logits")
    probe_predictions = output.get("refinement_probe_pred_velocity")
    probe_active = output.get("refinement_probe_active")
    exit_candidates = output.get("refinement_probe_exit_candidates")
    target_velocity = output.get("target_physical_velocity")
    base_route_weight = max(
        float(trainer.hierarchical_mmdit_oracle_route_loss_weight), 0.0
    )
    if base_route_weight > 0.0 and str(
        system.policy_config.hierarchical_mmdit_schedule_mode
    ) != "fixed":
        raise ValueError(
            "oracle route supervision requires hierarchical_mmdit_schedule_mode=fixed"
        )
    route_inputs = {
        "exit_logits": exit_logits,
        "probe_predictions": probe_predictions,
        "probe_active": probe_active,
        "exit_candidates": exit_candidates,
        "target_velocity": target_velocity,
    }
    if base_route_weight > 0.0:
        missing_route_inputs = [
            name for name, value in route_inputs.items() if not torch.is_tensor(value)
        ]
        if missing_route_inputs:
            raise RuntimeError(
                "oracle route supervision is enabled but decoder probes are missing: "
                + ", ".join(missing_route_inputs)
            )
        with torch.no_grad():
            route_residual = (
                probe_predictions.float()
                - target_velocity.detach().float()[:, None]
            )
            route_horizon_error = semantic_physical_velocity_error(
                system,
                route_residual,
                arm_null_weight=trainer.arm_manifold_null_weight,
            )
            route_position_weight = position_weights(
                system.policy_config, trainer, route_horizon_error.device
            ).to(dtype=route_horizon_error.dtype)
            route_step_error = (
                route_horizon_error * route_position_weight[None, None]
            ).mean(dim=-1)
        route = _oracle_exit_supervision(
            exit_logits=exit_logits,
            candidate_error=route_step_error[:, 1:],
            initial_error=route_step_error[:, 0],
            candidate_mask=probe_active.bool() & exit_candidates.bool(),
            relative_tolerance=float(
                trainer.hierarchical_mmdit_oracle_route_relative_tolerance
            ),
        )
        route_warmup = max(
            int(trainer.hierarchical_mmdit_oracle_route_warmup_steps), 0
        )
        step_value = 0 if global_step is None else max(int(global_step), 0)
        route_weight = (
            base_route_weight
            if system.training and step_value >= route_warmup
            else 0.0
        )
        losses["hierarchical_mmdit_oracle_route_loss"] = route["loss"].detach().float()
        losses["hierarchical_mmdit_oracle_route_weight"] = torch.as_tensor(
            route_weight, device=exit_logits.device, dtype=torch.float32
        )
        losses["hierarchical_mmdit_oracle_position_weighted"] = torch.ones(
            (), device=exit_logits.device, dtype=torch.float32
        )
        for name, value in route.items():
            if name != "loss":
                losses[f"hierarchical_mmdit_oracle_{name}"] = value.detach().float()
        if route_weight > 0.0:
            losses["loss"] = losses["loss"] + route_weight * route["loss"]
    # V70: stable-denominator replacements for the retired xratio gauge.
    # volume parity = noisy token norm vs workspace token norm (1.0 = parity;
    # after the noisy LayerNorm lands this pins to 1 by construction).
    # influence ratio = (attention share x value volume) of noisy vs workspace
    # evidence -- the honest "how much is the action listening to x_t" gauge.
    noisy_vol = output.get("latent_cvae_mmdit_noisy_token_norm")
    ws_vol = output.get("latent_cvae_workspace_token_norm")
    if torch.is_tensor(noisy_vol) and torch.is_tensor(ws_vol):
        losses["mmdit_noisy_volume_parity"] = (
            noisy_vol.detach().float() / ws_vol.detach().float().clamp_min(1e-6)
        )
        noisy_attn = output.get("latent_cvae_mmdit_action_noisy_attention")
        ws_attn = output.get("latent_cvae_mmdit_action_workspace_attention")
        if torch.is_tensor(noisy_attn) and torch.is_tensor(ws_attn):
            losses["mmdit_noisy_influence_ratio"] = (
                noisy_attn.detach().float() * noisy_vol.detach().float()
            ) / (
                ws_attn.detach().float() * ws_vol.detach().float()
            ).clamp_min(1e-8)
    deterministic_intent_decoder = "intent_contract_deterministic" in output
    micro_losses = (
        {}
        if deterministic_intent_decoder
        else micro_refine_supervision_losses(system, sample, output, trainer, global_step=global_step)
    )
    for key, value in micro_losses.items():
        losses[key] = value.detach().float()
    micro_weight = float(getattr(trainer, "latent_cvae_micro_supervision_weight", 0.0))
    micro_event_weight = float(getattr(trainer, "latent_cvae_micro_event_weight", 0.0))
    micro_mono_weight = float(getattr(trainer, "latent_cvae_micro_monotonic_weight", 0.0))
    micro_kl_weight = float(getattr(trainer, "latent_cvae_micro_weight_kl_weight", 0.0))
    micro_smooth_weight = float(getattr(trainer, "latent_cvae_micro_coverage_smooth_weight", 0.0))
    micro_floor_weight = float(getattr(trainer, "latent_cvae_micro_coverage_floor_weight", 0.0))
    if micro_weight > 0 and "latent_cvae_micro_supervision" in micro_losses:
        losses["loss"] = losses["loss"] + micro_weight * micro_losses["latent_cvae_micro_supervision"]
    if micro_event_weight > 0 and "latent_cvae_micro_event" in micro_losses:
        losses["loss"] = losses["loss"] + micro_event_weight * micro_losses["latent_cvae_micro_event"]
    if micro_mono_weight > 0 and "latent_cvae_micro_monotonic" in micro_losses:
        losses["loss"] = losses["loss"] + micro_mono_weight * micro_losses["latent_cvae_micro_monotonic"]
    if micro_kl_weight > 0 and "latent_cvae_micro_weight_kl" in micro_losses:
        losses["loss"] = losses["loss"] + micro_kl_weight * micro_losses["latent_cvae_micro_weight_kl"]
    if micro_smooth_weight > 0 and "latent_cvae_micro_coverage_smooth" in micro_losses:
        losses["loss"] = losses["loss"] + micro_smooth_weight * micro_losses["latent_cvae_micro_coverage_smooth"]
    if micro_floor_weight > 0 and "latent_cvae_micro_coverage_floor" in micro_losses:
        losses["loss"] = losses["loss"] + micro_floor_weight * micro_losses["latent_cvae_micro_coverage_floor"]
    losses.update(rollout_diagnostics(output))
    if "gate_self" in output:
        losses["gate_self"] = output["gate_self"].detach()
        losses["gate_visual"] = output["gate_visual"].detach()
        losses["gate_rollout"] = output.get("gate_rollout", torch.zeros_like(output["gate_self"])).detach()
        losses["gate_ffn"] = output["gate_ffn"].detach()
    for key in (
        "mod_content_norm", "mod_time_norm", "mod_content_to_time",
        "future_conditioned_action_loss",
    ):
        if key in output:
            losses[key] = output[key].detach()
    if "rollout_alpha" in output:
        losses["rollout_alpha_mean"] = output["rollout_alpha"].detach().float().mean()
    for key in (
        "action_flow_residual_norm",
        "action_flow_raw_residual_norm",
        "action_flow_residual_alpha_mean",
        "action_flow_stage_router_entropy",
        "action_flow_stage_router_max",
        "latent_action_stage_router_entropy",
        "latent_action_stage_router_max",
        "latent_action_gripper_gate_mean",
        "latent_action_layer_memory_count",
        "latent_action_temporal_update_mean",
        "latent_action_temporal_near_depth",
        "latent_action_temporal_mid_depth",
        "latent_cvae_kl",
        "latent_cvae_kl_weight",
        "latent_cvae_prior_std",
        "latent_cvae_post_std",
        "latent_cvae_z_norm",
        "latent_cvae_prior_z_norm",
        "latent_cvae_post_z_norm",
        "latent_cvae_mu_gap",
        "latent_cvae_prior_pred_norm",
        "latent_cvae_post_pred_norm",
        "latent_cvae_post_gripper_gate_mean",
        "latent_cvae_condition_norm",
        "latent_cvae_condition_raw_norm",
        "latent_cvae_condition_scan_norm",
        "latent_cvae_condition_lateral_norm",
        "latent_cvae_layer_summary_norm",
        "latent_cvae_transition_source_raw_norm",
        "latent_cvae_transition_condition_norm",
        "latent_cvae_rollout_token_norm",
        "latent_cvae_rollout_token_count",
        "latent_cvae_consequence_scale_mean",
        "latent_cvae_consequence_gate_preference",
        "latent_cvae_consequence_mix_ratio",
        "latent_cvae_posterior_used",
        "latent_cvae_gripper_gate_mean",
        "latent_cvae_layer_memory_count",
        "latent_cvae_adaptive_refine_update_mean",
        "latent_cvae_adaptive_noisy_gate_mean",
        "latent_cvae_adaptive_noisy_branch_norm",
        "latent_cvae_adaptive_noisy_branch_ratio",
        "latent_cvae_adaptive_route_entropy",
        "latent_cvae_adaptive_route_max",
        "latent_cvae_adaptive_route_effective_slots",
        "latent_cvae_adaptive_progress_entropy",
        "latent_cvae_adaptive_progress_max",
        "latent_cvae_adaptive_progress_effective_slots",
        "latent_cvae_adaptive_progress_norm",
        "latent_cvae_adaptive_continue_mean",
        "latent_cvae_adaptive_prefix_norm",
        "latent_cvae_adaptive_progress_seed_entropy",
        "latent_cvae_adaptive_progress_seed_max",
        "latent_cvae_adaptive_progress_seed_effective_slots",
        "latent_cvae_adaptive_progress_seed_norm",
        "latent_cvae_adaptive_route_temperature_mean",
        "latent_cvae_adaptive_route_time_query_norm",
        "latent_cvae_adaptive_semantic_bias_norm",
        "latent_cvae_adaptive_function_delta_norm",
        "latent_cvae_adaptive_base_highfreq_norm",
        "latent_cvae_adaptive_refine_step_bias_norm",
        "latent_cvae_adaptive_capsule_layer_entropy",
        "latent_cvae_adaptive_capsule_layer_max",
        "latent_cvae_adaptive_capsule_layer_effective_slots",
        "latent_cvae_adaptive_condition_strength_mean",
        "latent_cvae_adaptive_condition_strength_std",
        "latent_cvae_adaptive_condition_strength_max",
        "latent_cvae_adaptive_condition_strength_min",
        "latent_cvae_adaptive_condition_residual_norm",
        "latent_cvae_adaptive_context_direction_norm",
        "latent_cvae_adaptive_micro_step_mean",
        "latent_cvae_adaptive_micro_step_std",
        "latent_cvae_adaptive_micro_progress_mean",
        "latent_cvae_adaptive_micro_kp_mean",
        "latent_cvae_adaptive_micro_kd_mean",
        "latent_cvae_adaptive_micro_feedforward_norm",
        "latent_cvae_adaptive_micro_feedback_norm",
        "latent_cvae_adaptive_micro_damping_norm",
        "latent_cvae_adaptive_micro_function_norm",
        "latent_cvae_adaptive_micro_control_norm",
        "latent_cvae_adaptive_micro_update_norm",
        "latent_cvae_adaptive_micro_heun_error",
        "latent_cvae_adaptive_micro_refine_block_norm",
        "latent_cvae_adaptive_micro_controller_norm",
        "latent_cvae_adaptive_regularizer",
        "latent_cvae_adaptive_regularizer_weight",
        "latent_cvae_adaptive_route_entropy_regularizer",
        "latent_cvae_adaptive_route_entropy_weight",
        "latent_cvae_mmdit_action_update_norm",
        "latent_cvae_mmdit_cond_update_norm",
        "latent_cvae_mmdit_action_cond_attention",
        "latent_cvae_mmdit_action_noisy_attention",
        "latent_cvae_mmdit_action_workspace_attention",
        "latent_cvae_mmdit_action_workspace_enrichment",
        "latent_cvae_mmdit_action_low_attention",
        "latent_cvae_mmdit_action_stage_attention",
        "latent_cvae_mmdit_action_low_enrichment",
        "latent_cvae_mmdit_action_stage_enrichment",
        "latent_cvae_mmdit_action_rollout_attention",
        "latent_cvae_mmdit_action_rollout_enrichment",
        "latent_cvae_mmdit_action_token_norm",
        "latent_cvae_mmdit_condition_token_norm",
        "latent_cvae_mmdit_noisy_token_norm",
        # V72: time-stratified x_t/workspace attention (sum+count pairs; the
        # epoch mean of sums divided by the epoch mean of counts recovers the
        # exact stratified mean, robust to empty buckets in small batches).
        "latent_cvae_mmdit_noisy_attn_t0_sum",
        "latent_cvae_mmdit_noisy_attn_t1_sum",
        "latent_cvae_mmdit_noisy_attn_t2_sum",
        "latent_cvae_mmdit_workspace_attn_t0_sum",
        "latent_cvae_mmdit_workspace_attn_t1_sum",
        "latent_cvae_mmdit_workspace_attn_t2_sum",
        "latent_cvae_mmdit_low_attn_t0_sum",
        "latent_cvae_mmdit_low_attn_t1_sum",
        "latent_cvae_mmdit_low_attn_t2_sum",
        "latent_cvae_mmdit_stage_attn_t0_sum",
        "latent_cvae_mmdit_stage_attn_t1_sum",
        "latent_cvae_mmdit_stage_attn_t2_sum",
        "latent_cvae_mmdit_attn_t0_count",
        "latent_cvae_mmdit_attn_t1_count",
        "latent_cvae_mmdit_attn_t2_count",
        "latent_cvae_primary_condition_norm",
        "latent_cvae_primary_z_effect_norm",
        "latent_cvae_workspace_progress_update_norm",
        # V72: echo probe -- fraction of the progress update attributable to
        # the raw action summary input (0 by construction under isolation).
        "latent_cvae_workspace_progress_action_dependence",
        # CR0 probes (do_before_v76): legacy stem activity + z interventions.
        "latent_cvae_legacy_stem_effect_ratio",
        "latent_cvae_z_zero_delta",
        "latent_cvae_z_shuffle_delta",
        "latent_cvae_workspace_token_count",
        "latent_cvae_workspace_token_norm",
        "latent_cvae_workspace_update_norm",
        "latent_cvae_workspace_global_state_norm",
        "latent_cvae_workspace_global_slot_delta_norm",
        "latent_cvae_workspace_global_slot_diversity",
        "latent_cvae_workspace_source_count",
        "latent_cvae_workspace_cached_token_fraction",
        "latent_cvae_workspace_attention_entropy",
        "latent_cvae_workspace_attention_max",
        "latent_cvae_workspace_group_attention_entropy",
        "latent_cvae_workspace_group_effective_sources",
        "latent_cvae_workspace_attention_mass_error",
        "latent_cvae_workspace_action_update_ratio",
        "latent_cvae_workspace_noisy_query_scale",
        "latent_cvae_workspace_progress_query_norm",
        "latent_cvae_workspace_role_geom_attention",
        "latent_cvae_workspace_role_transition_attention",
        "latent_cvae_workspace_role_event_attention",
        "latent_cvae_workspace_role_state_attention",
        "latent_cvae_workspace_role_layer_attention",
        "latent_cvae_workspace_role_global_attention",
        "latent_cvae_workspace_role_geom_token_count",
        "latent_cvae_workspace_role_transition_token_count",
        "latent_cvae_workspace_role_event_token_count",
        "latent_cvae_workspace_role_state_token_count",
        "latent_cvae_workspace_role_layer_token_count",
        "latent_cvae_workspace_role_global_token_count",
        "latent_cvae_workspace_controller_capacity",
        "latent_cvae_workspace_controller_delay",
        "latent_cvae_workspace_controller_temperature",
        "latent_cvae_workspace_controller_role_entropy",
        "latent_cvae_workspace_controller_role_max",
        "latent_cvae_workspace_controller_query_delta_norm",
        "latent_cvae_workspace_controller_workspace_delta_norm",
        "latent_cvae_workspace_controller_role_geom_prob",
        "latent_cvae_workspace_controller_role_transition_prob",
        "latent_cvae_workspace_controller_role_event_prob",
        "latent_cvae_workspace_controller_role_state_prob",
        "latent_cvae_workspace_controller_role_layer_prob",
        "latent_cvae_workspace_controller_role_global_prob",
        "latent_cvae_workspace_controller_role_geom_logit",
        "latent_cvae_workspace_controller_role_transition_logit",
        "latent_cvae_workspace_controller_role_event_logit",
        "latent_cvae_workspace_controller_role_state_logit",
        "latent_cvae_workspace_controller_role_layer_logit",
        "latent_cvae_workspace_controller_role_global_logit",
        "latent_cvae_hierarchical_low_token_count",
        "latent_cvae_hierarchical_low_token_norm",
        "latent_cvae_hierarchical_low_selector_stage_entropy",
        "latent_cvae_hierarchical_low_selector_stage_max",
        "latent_cvae_hierarchical_low_selector_stage_effective_slots",
        "latent_cvae_hierarchical_low_selector_role_norm",
        "latent_cvae_hierarchical_low_selector_content_norm",
        "latent_cvae_hierarchical_stage_token_count",
        "latent_cvae_hierarchical_stage_role_norm",
        "latent_cvae_hierarchical_stage_role_diversity",
        "latent_cvae_hierarchical_stage_content_norm",
        "latent_cvae_hierarchical_stage_content_diversity",
        "latent_cvae_hierarchical_stage_role_content_cosine",
        "latent_cvae_hierarchical_stage_role_output_norm",
        "latent_cvae_hierarchical_stage_content_output_norm",
        "latent_cvae_hierarchical_stage_role_output_fraction",
        "latent_cvae_hierarchical_stage_update_norm",
        "latent_cvae_hierarchical_stage_retain_mean",
        "latent_cvae_hierarchical_stage_promote_attention_entropy",
        "latent_cvae_hierarchical_stage_promote_attention_max",
        "latent_cvae_hierarchical_stage_promoted_norm",
        "latent_cvae_hierarchical_stage_promoted_projected_rms",
        "latent_cvae_hierarchical_stage_promoted_normalized_rms",
        "latent_cvae_hierarchical_stage_promoted_realized_scale",
        "latent_cvae_hierarchical_stage_promote_gate_scale_error",
        "latent_cvae_hierarchical_stage_promote_scale",
        "latent_cvae_hierarchical_manager_stage_attention_entropy",
        "latent_cvae_hierarchical_manager_stage_attention_max",
        "latent_cvae_hierarchical_manager_role_entropy",
        "latent_cvae_hierarchical_manager_role_max",
        "latent_cvae_hierarchical_manager_query_shift_norm",
        "latent_cvae_hierarchical_manager_promote_gate",
        "latent_cvae_hierarchical_manager_low_output_strength",
        "latent_cvae_hierarchical_manager_stage_output_strength",
        "latent_cvae_hierarchical_manager_role_geom_prob",
        "latent_cvae_hierarchical_manager_role_transition_prob",
        "latent_cvae_hierarchical_manager_role_event_prob",
        "latent_cvae_hierarchical_manager_role_state_prob",
        "latent_cvae_hierarchical_manager_role_layer_prob",
        "latent_cvae_hierarchical_manager_role_global_prob",
        "latent_cvae_workspace_layer_attention",
        "latent_cvae_workspace_scan_attention",
        "latent_cvae_workspace_lateral_attention",
        "latent_cvae_workspace_transition_attention",
        "latent_cvae_workspace_transition_delta_attention",
        "latent_cvae_workspace_transition_effect_attention",
        "latent_cvae_workspace_transition_timeline_attention",
        "latent_cvae_workspace_transition_total_attention",
        "latent_cvae_workspace_context_attention",
        "latent_cvae_workspace_visual_attention",
        "latent_cvae_workspace_trajectory_attention",
        "latent_cvae_workspace_rollout_attention",
        "latent_cvae_workspace_capsule_attention",
        "latent_cvae_workspace_progress_attention",
        "latent_cvae_workspace_routed_layer_attention",
        "consequence_self_condition",
        "consequence_self_condition_target_mse",
        "consequence_self_condition_noisy_mse",
        "consequence_preview_flow",
    ):
        if key in output:
            losses[key] = output[key].detach().float()
    probe_predictions = output.get("refinement_probe_pred_velocity")
    probe_active = output.get("refinement_probe_active")
    probe_response = output.get("refinement_probe_action_response_rel")
    probe_pressure = output.get("refinement_probe_stage_pressure_rel")
    probe_stage_ids = output.get("refinement_probe_stage_ids")
    probe_block_ids = output.get("refinement_probe_block_ids")
    target_velocity = output.get("target_physical_velocity")
    if all(torch.is_tensor(value) for value in (
        probe_predictions, probe_active, probe_response, probe_pressure,
        probe_stage_ids, probe_block_ids, target_velocity,
    )):
        with torch.no_grad():
            residual = probe_predictions.float() - target_velocity.detach().float()[:, None]
            step_error = semantic_physical_velocity_error(
                system,
                residual,
                arm_null_weight=trainer.arm_manifold_null_weight,
            ).mean(dim=-1)
            marginal_gain = step_error[:, :-1] - step_error[:, 1:]
            active = probe_active.float()
            denominator = active.sum().clamp_min(1.0)
            losses["hierarchical_mmdit_refine_error_initial"] = step_error[:, 0].mean()
            final_index = active.sum(dim=1).long().clamp_min(1)
            final_error = step_error.gather(1, final_index[:, None]).squeeze(1)
            losses["hierarchical_mmdit_refine_error_final"] = final_error.mean()
            losses["hierarchical_mmdit_refine_gain"] = (marginal_gain * active).sum() / denominator
            losses["hierarchical_mmdit_refine_positive_gain_fraction"] = (
                (marginal_gain > 0).float() * active
            ).sum() / denominator

            def masked_correlation(
                x: Tensor,
                y: Tensor,
                selected: Tensor | None = None,
            ) -> Tensor:
                if selected is None:
                    selected = active.bool()
                x_rows = x[selected].float()
                y_rows = y[selected].float()
                if int(x_rows.numel()) < 2:
                    return torch.zeros((), device=x.device, dtype=torch.float32)
                x_rows = x_rows - x_rows.mean()
                y_rows = y_rows - y_rows.mean()
                scale = x_rows.square().sum().sqrt() * y_rows.square().sum().sqrt()
                return (x_rows * y_rows).sum() / scale.clamp_min(1e-8)

            losses["hierarchical_mmdit_response_gain_corr"] = masked_correlation(
                probe_response, marginal_gain,
            )
            losses["hierarchical_mmdit_pressure_gain_corr"] = masked_correlation(
                probe_pressure, marginal_gain,
            )
            probe_time = output.get("time")
            if torch.is_tensor(probe_time) and tuple(probe_time.shape) == (
                int(active.shape[0]),
            ):
                time_bins = torch.clamp(
                    (probe_time.detach().float().clamp(0.0, 1.0) * 3.0).long(),
                    max=2,
                )
                for time_bin in range(3):
                    selected = active.bool() & (time_bins[:, None] == time_bin)
                    selected_float = selected.float()
                    selected_denominator = selected_float.sum().clamp_min(1.0)
                    losses[
                        f"hierarchical_mmdit_response_gain_corr_t{time_bin}"
                    ] = masked_correlation(probe_response, marginal_gain, selected)
                    losses[
                        f"hierarchical_mmdit_pressure_gain_corr_t{time_bin}"
                    ] = masked_correlation(probe_pressure, marginal_gain, selected)
                    losses[f"hierarchical_mmdit_refine_gain_t{time_bin}"] = (
                        marginal_gain * selected_float
                    ).sum() / selected_denominator
                    losses[
                        f"hierarchical_mmdit_refine_positive_gain_fraction_t{time_bin}"
                    ] = (
                        (marginal_gain > 0).float() * selected_float
                    ).sum() / selected_denominator
            for step_index in range(int(active.shape[1])):
                step_mask = active[:, step_index]
                step_denominator = step_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_step_{step_index}_gain"] = (
                    marginal_gain[:, step_index] * step_mask
                ).sum() / step_denominator
                losses[f"hierarchical_mmdit_step_{step_index}_positive_gain_fraction"] = (
                    (marginal_gain[:, step_index] > 0).float() * step_mask
                ).sum() / step_denominator
            for stage_index in range(
                int(system.policy_config.hierarchical_mmdit_operator_stages)
            ):
                stage_mask = active * (probe_stage_ids == stage_index).float()
                stage_denominator = stage_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_stage_{stage_index}_gain"] = (
                    marginal_gain * stage_mask
                ).sum() / stage_denominator
            for block_index in range(
                int(system.policy_config.hierarchical_mmdit_depth)
            ):
                block_mask = active * (probe_block_ids == block_index).float()
                block_denominator = block_mask.sum().clamp_min(1.0)
                losses[f"hierarchical_mmdit_block_{block_index}_gain"] = (
                    marginal_gain * block_mask
                ).sum() / block_denominator
    losses.update(_shadow_refinement_probe_metrics(
        system,
        output,
        arm_null_weight=trainer.arm_manifold_null_weight,
    ))

    # The deterministic decoder intentionally does not alias its diagnostics
    # into latent_cvae_* names.  Prefix pass-through keeps the new contract
    # auditable without reviving retired CVAE losses or zero-filled log fields.
    for key, value in output.items():
        if (
            key.startswith(("intent_", "owned_", "hierarchical_mmdit_"))
            and torch.is_tensor(value)
            and value.numel() == 1
        ):
            losses[key] = value.detach().float().reshape(())
    clean_noisy_vol = output.get("hierarchical_mmdit_noisy_token_norm")
    clean_cond_vol = output.get("hierarchical_mmdit_condition_token_norm")
    if torch.is_tensor(clean_noisy_vol) and torch.is_tensor(clean_cond_vol):
        losses["hierarchical_mmdit_noisy_volume_parity"] = (
            clean_noisy_vol.detach().float() / clean_cond_vol.detach().float().clamp_min(1e-6)
        )
    clean_noisy_fraction = output.get("hierarchical_mmdit_action_noisy_update_fraction")
    clean_stage_fraction = output.get("hierarchical_mmdit_action_stage_update_fraction")
    clean_low_fraction = output.get("hierarchical_mmdit_action_low_update_fraction")
    if all(torch.is_tensor(value) for value in (
        clean_noisy_fraction, clean_stage_fraction, clean_low_fraction,
    )):
        workspace_fraction = clean_stage_fraction.detach().float() + clean_low_fraction.detach().float()
        losses["hierarchical_mmdit_noisy_to_workspace_update_ratio"] = (
            clean_noisy_fraction.detach().float() / workspace_fraction.clamp_min(1e-8)
        )
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
    # These describe the top-level controlled-dynamics decomposition. A layer
    # consequence entry has a different contract and must not inherit them.
    fake.pop("rollout_decomposition_expansion_ratio", None)
    fake.pop("rollout_base_is_fixed_zero", None)
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
        boundary = sample["action_state"].to(
            device=fake["pred_physical_velocity"].device, dtype=fake["pred_physical_velocity"].dtype
        )
        # V70 (H3 fix): project before decode, matching the training forward
        # and deployment geometry.
        clean = system.codec.project_physical(
            output["noisy_physical_action"] - t[:, None, None] * fake["pred_physical_velocity"],
            boundary,
        )
        fake["clean_physical_estimate"] = clean
        fake["pred_action_estimate"] = system.codec.decode(clean, boundary)
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
    eval_field_null_sse = 0.0
    eval_field_energy = 0.0
    eval_field_null_count = 0
    eval_noise_projection_sse = 0.0
    eval_noise_projection_count = 0
    eval_arm_field_null_sse = 0.0
    eval_arm_field_energy = 0.0
    eval_arm_field_null_count = 0
    eval_arm_noise_projection_sse = 0.0
    eval_arm_noise_projection_count = 0
    contract_eval = _is_contract_stage(trainer) and _uses_layer_adapter_contract(trainer)
    contract_metric_sums: dict[str, float] = {}
    contract_metric_count = 0
    sampling_diagnostic_sums: dict[str, float] = {}
    sampling_diagnostic_counts: dict[str, int] = {}
    shadow_probe_eval = (
        str(getattr(system.policy_config, "final_action_decoder", "legacy"))
        == "hierarchical_mmdit_action"
        and str(getattr(system.policy_config, "hierarchical_mmdit_exhaustion_mode", "off"))
        == "shadow"
    )
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
        noise = system.codec.sample_noise(
            sample["policy_action"].shape[0],
            generator=generator,
            device=device,
            dtype=sample["visual"].dtype,
            action_state=sample["action_state"],
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
            diagnostic_weight = int(pred_pack["action"].shape[0])
            for key, value in pred_pack.items():
                keep_sampling_diagnostic = (
                    key.startswith("sample_latent_cvae_")
                    or key.startswith("sample_intent_")
                    or key.startswith("sample_owned_")
                    or key.startswith("sample_hierarchical_mmdit_")
                    or key.startswith("sample_arm_null_")
                    or key.startswith("sample_grip_null_")
                )
                if keep_sampling_diagnostic and torch.is_tensor(value) and value.numel() == 1:
                    sampling_diagnostic_sums[key] = (
                        sampling_diagnostic_sums.get(key, 0.0)
                        + float(value.float().cpu()) * diagnostic_weight
                    )
                    sampling_diagnostic_counts[key] = sampling_diagnostic_counts.get(key, 0) + diagnostic_weight
            no_proposal = system.sample(
                sample["visual"], sample["history_state"], sample["executed_action_history"], sample["state"],
                action_state=sample["action_state"],
                steps=trainer.eval_inference_steps, noise=noise, use_proposal=False,
                stop_at_midcut=stop_midcut_eval,
            )
            if shadow_probe_eval:
                fork_devices = (
                    [device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda"
                    else []
                )
                with torch.random.fork_rng(devices=fork_devices):
                    probe_seed = 91073 + batch_index
                    torch.manual_seed(probe_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(probe_seed)
                    shadow_output = system.flow_training_forward(
                        sample["visual"],
                        sample["history_state"],
                        sample["executed_action_history"],
                        sample["state"],
                        sample["policy_action"],
                        action_state=sample["action_state"],
                        proposal_dropout=0.0,
                        make_counterfactuals=False,
                        stop_at_midcut=False,
                    )
                shadow_metrics = _shadow_refinement_probe_metrics(
                    system,
                    shadow_output,
                    arm_null_weight=trainer.arm_manifold_null_weight,
                )
                for key, value in shadow_metrics.items():
                    metric_key = f"sample_{key}"
                    sampling_diagnostic_sums[metric_key] = (
                        sampling_diagnostic_sums.get(metric_key, 0.0)
                        + float(value.detach().float().cpu()) * diagnostic_weight
                    )
                    sampling_diagnostic_counts[metric_key] = (
                        sampling_diagnostic_counts.get(metric_key, 0) + diagnostic_weight
                    )
            if system.codec.uses_parseval_gripper_field:
                ad = int(system.policy_config.arm_dim)
                gf = int(system.policy_config.gripper_field_dim)
                noise_field = noise[..., 2 * ad : 2 * ad + gf]
                noise_null = noise_field - system.codec.project_gripper_field(noise_field)
                eval_noise_projection_sse += float(noise_null.float().square().sum().cpu())
                eval_noise_projection_count += int(noise_null.numel())
                pred_field = pred_pack["physical_action"][..., 2 * ad : 2 * ad + gf]
                pred_null = pred_field - system.codec.project_gripper_field(pred_field)
                eval_field_null_sse += float(pred_null.float().square().sum().cpu())
                eval_field_energy += float(pred_field.float().square().sum().cpu())
                eval_field_null_count += int(pred_null.numel())
            if system.codec.uses_arm_manifold:
                ad = int(system.policy_config.arm_dim)
                action_state = sample["action_state"]
                noise_arm = noise[..., : 2 * ad]
                projected_noise_arm = system.codec.project_arm_field(noise_arm, action_state)
                noise_arm_null = noise_arm - projected_noise_arm
                eval_arm_noise_projection_sse += float(noise_arm_null.float().square().sum().cpu())
                eval_arm_noise_projection_count += int(noise_arm_null.numel())
                pred_arm = pred_pack["physical_action"][..., : 2 * ad]
                projected_pred_arm = system.codec.project_arm_field(pred_arm, action_state)
                pred_arm_null = pred_arm - projected_pred_arm
                eval_arm_field_null_sse += float(pred_arm_null.float().square().sum().cpu())
                eval_arm_field_energy += float(pred_arm.float().square().sum().cpu())
                eval_arm_field_null_count += int(pred_arm_null.numel())
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
    arm_squared = squared[..., :-1]
    gripper_squared = squared[..., -1]
    metrics = {
        "full_mse": float(squared.mean()),
        "full_rmse": float(np.sqrt(squared.mean())),
        "first_rmse": float(np.sqrt(squared[:, 0].mean())),
        "first4_rmse": float(np.sqrt(squared[:, :4].mean())),
        "first8_rmse": float(np.sqrt(squared[:, :8].mean())),
        "tail_rmse": float(np.sqrt(squared[:, 8:].mean())) if squared.shape[1] > 8 else float("nan"),
        "arm_full_rmse": float(np.sqrt(arm_squared.mean())),
        "arm_first_rmse": float(np.sqrt(arm_squared[:, 0].mean())),
        "arm_first4_rmse": float(np.sqrt(arm_squared[:, :4].mean())),
        "arm_first8_rmse": float(np.sqrt(arm_squared[:, :8].mean())),
        "arm_tail_rmse": float(np.sqrt(arm_squared[:, 8:].mean())) if arm_squared.shape[1] > 8 else float("nan"),
        "gripper_full_rmse": float(np.sqrt(gripper_squared.mean())),
        "gripper_first_rmse": float(np.sqrt(gripper_squared[:, 0].mean())),
        "gripper_first4_rmse": float(np.sqrt(gripper_squared[:, :4].mean())),
        "gripper_first8_rmse": float(np.sqrt(gripper_squared[:, :8].mean())),
        "gripper_tail_rmse": float(np.sqrt(gripper_squared[:, 8:].mean())) if gripper_squared.shape[1] > 8 else float("nan"),
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
    if system.codec.uses_parseval_gripper_field:
        metrics["eval_gripper_field_projection_mse"] = eval_field_null_sse / max(eval_field_null_count, 1)
        metrics["eval_gripper_field_null_ratio"] = eval_field_null_sse / max(eval_field_energy, 1e-12)
        metrics["eval_gripper_noise_projection_mse"] = eval_noise_projection_sse / max(eval_noise_projection_count, 1)
    if system.codec.uses_arm_manifold:
        metrics["eval_arm_field_projection_mse"] = eval_arm_field_null_sse / max(eval_arm_field_null_count, 1)
        metrics["eval_arm_field_null_ratio"] = eval_arm_field_null_sse / max(eval_arm_field_energy, 1e-12)
        metrics["eval_arm_noise_projection_mse"] = eval_arm_noise_projection_sse / max(
            eval_arm_noise_projection_count, 1
        )
    if contract_metric_count:
        for key, value in contract_metric_sums.items():
            metrics[f"contract_{key}"] = value / float(contract_metric_count)
    for key, value in sampling_diagnostic_sums.items():
        metrics[key] = value / float(max(sampling_diagnostic_counts.get(key, 0), 1))
    metrics["eval_sampling_diagnostic_count"] = float(len(sampling_diagnostic_sums))
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


def _detached_scalar_metric(key: str, value: Tensor) -> Tensor:
    detached = value.detach().float()
    if detached.numel() != 1:
        raise ValueError(
            f"metric {key!r} must contain exactly one element; "
            f"got shape={tuple(detached.shape)}"
        )
    return detached.reshape(())


def _accumulate_metric_tensors(acc: dict[str, Tensor], losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> None:
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        detached = _detached_scalar_metric(key, value)
        acc[key] = acc.get(key, torch.zeros((), device=detached.device, dtype=torch.float32)) + detached
    if grad is not None:
        g = _detached_scalar_metric("grad", grad) if torch.is_tensor(grad) else torch.tensor(float(grad))
        acc["grad"] = acc.get("grad", torch.zeros((), device=g.device, dtype=torch.float32)) + g


def _finalize_metric_tensors(acc: dict[str, Tensor], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {
        key: float((_detached_scalar_metric(key, value) / float(count)).cpu())
        for key, value in acc.items()
    }


def _sync_loss_row(losses: dict[str, Tensor], *, grad: Tensor | float | None = None) -> dict[str, float]:
    row = {
        key: float(_detached_scalar_metric(key, value).cpu())
        for key, value in losses.items()
        if torch.is_tensor(value)
    }
    if grad is not None:
        row["grad"] = (
            float(_detached_scalar_metric("grad", grad).cpu())
            if torch.is_tensor(grad)
            else float(grad)
        )
    return row


def _format_hierarchical_stage_usage(row: dict[str, float]) -> str:
    count = int(round(row.get("hierarchical_mmdit_operator_stage_count", 0.0)))
    if count <= 0:
        prefix = "hierarchical_mmdit_stage_"
        suffix = "_usage"
        indices = []
        for key in row:
            if key.startswith(prefix) and key.endswith(suffix):
                middle = key[len(prefix) : -len(suffix)]
                if middle.isdigit():
                    indices.append(int(middle))
        count = max(indices, default=-1) + 1
    return "/".join(
        f"{row.get(f'hierarchical_mmdit_stage_{index}_usage', 0.0):.2f}"
        for index in range(count)
    ) or "-"


def _format_hierarchical_block_usage(row: dict[str, float]) -> str:
    count = int(round(row.get("hierarchical_mmdit_refine_block_count", 0.0)))
    return "/".join(
        f"{row.get(f'hierarchical_mmdit_block_{index}_usage', 0.0):.2f}"
        for index in range(count)
    ) or "-"


def _owned_serial_log_line(
    row: dict[str, float],
    *,
    epoch: int,
    batch_index: int,
    learning_rate: float,
    seconds_per_batch: float,
) -> str:
    """High-signal batch line for the clean owned-evidence decoder.

    Full scalar diagnostics remain in the epoch JSON.  This line deliberately
    omits retired latent-CVAE fields instead of rendering them as misleading
    zeroes beside the active serial decoder gauges.
    """
    return (
        f"[v39-layer] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
        f"pflow={row['physical_flow']:.6f} "
        f"pflowu={row.get('physical_flow_uniform', row['physical_flow']):.6f} "
        f"pfn={row.get('physical_flow_native', 0.0):.6f} "
        f"afmd={row.get('arm_fm_per_dim', 0.0):.5f} "
        f"gfmf={row.get('gripper_fm_field', 0.0):.5f} "
        f"gfar={row.get('gripper_arm_fm_ratio', 0.0):.3f} "
        f"anull={row.get('arm_fm_null_output_fraction', 0.0):.4f} "
        f"gnull={row.get('gripper_fm_null_output_fraction', 0.0):.4f} "
        f"decode={row.get('decoded_action', 0.0):.6f} "
        f"rollout={row.get('rollout_dynamics', 0.0):.6f} "
        f"rvar={row.get('rollout_variance', 0.0):.4f} "
        f"rnorm={row.get('rollout_norm', 0.0):.4f} "
        f"rstep={row.get('rollout_milestone_delta_match', 0.0):.4f} "
        f"first8={row.get('first8_physical_flow', 0.0):.6f} "
        f"tail={row.get('tail_physical_flow', 0.0):.6f} "
        f"delta={row.get('rollout_delta', 0.0):.6f} "
        f"contrast={row.get('rollout_contrast', 0.0):.6f} "
        f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} "
        f"stdr={row.get('rollout_pred_std_ratio', 0.0):.4f} "
        f"dnratio={row.get('rollout_milestone_delta_norm_ratio', 0.0):.4f} "
        f"event={row.get('event', 0.0):.6f} "
        f"icgs={row.get('intent_global_stage_cosine', 0.0):.3f} "
        f"icgr={row.get('intent_global_read_cosine', 0.0):.3f} "
        f"icsr={row.get('intent_stage_read_cosine', 0.0):.3f} "
        f"icdiv={row.get('intent_global_batch_diversity', 0.0):.2f}/"
        f"{row.get('intent_stage_batch_diversity', 0.0):.2f}/"
        f"{row.get('intent_read_batch_diversity', 0.0):.2f} "
        f"hmdu={row.get('hierarchical_mmdit_action_update_norm', 0.0):.3f} "
        f"hmur={row.get('hierarchical_mmdit_action_update_ratio', 0.0):.3f} "
        f"hmcan={row.get('hierarchical_mmdit_action_serial_cancellation_fraction', 0.0):.3f} "
        f"hmorth={row.get('hierarchical_mmdit_action_serial_cancellation_orthogonal_baseline', 0.0):.3f} "
        f"hmxcan={row.get('hierarchical_mmdit_action_serial_cancellation_excess', 0.0):+.3f} "
        f"hmbdot={row.get('hierarchical_mmdit_action_branch_weighted_cosine', 0.0):+.3f} "
        f"hmcos={row.get('hierarchical_mmdit_action_state_cosine', 0.0):.3f} "
        f"hmbcos={row.get('hierarchical_mmdit_action_noisy_stage_cosine', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_low_cosine', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_low_cosine', 0.0):.3f} "
        f"hmnu={row.get('hierarchical_mmdit_action_noisy_update_norm', 0.0):.3f} "
        f"hmsu={row.get('hierarchical_mmdit_action_stage_update_norm', 0.0):.3f} "
        f"hmlu={row.get('hierarchical_mmdit_action_low_update_norm', 0.0):.3f} "
        f"hmnf={row.get('hierarchical_mmdit_action_noisy_update_fraction', 0.0):.3f} "
        f"hmsf={row.get('hierarchical_mmdit_action_stage_update_fraction', 0.0):.3f} "
        f"hmlf={row.get('hierarchical_mmdit_action_low_update_fraction', 0.0):.3f} "
        f"hmnw={row.get('hierarchical_mmdit_noisy_to_workspace_update_ratio', 0.0):.3f} "
        f"hmdepth={row.get('hierarchical_mmdit_action_noisy_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_depth_ratio', 0.0):.2f} "
        f"hmraw={row.get('hierarchical_mmdit_action_noisy_raw_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_raw_depth_ratio', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_raw_depth_ratio', 0.0):.2f} "
        f"hmedepth={row.get('hierarchical_mmdit_action_noisy_effective_depth', 0.0):.1f}/"
        f"{row.get('hierarchical_mmdit_action_stage_effective_depth', 0.0):.1f}/"
        f"{row.get('hierarchical_mmdit_action_low_effective_depth', 0.0):.1f} "
        f"hmcontract={row.get('hierarchical_mmdit_action_noisy_contraction_ratio', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_contraction_ratio', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_contraction_ratio', 0.0):.3f} "
        f"hmhost={row.get('hierarchical_mmdit_action_noisy_host_update_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_host_update_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_host_update_rms', 0.0):.3f} "
        f"hmcover={row.get('hierarchical_mmdit_action_noisy_subspace_energy_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_subspace_energy_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_subspace_energy_fraction', 0.0):.3f} "
        f"hmremove={row.get('hierarchical_mmdit_action_noisy_removed_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_removed_fraction', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_removed_fraction', 0.0):.3f} "
        f"hmdepthreg={row.get('hierarchical_mmdit_operator_contraction_progress', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_depth_usage_regularizer', 0.0):.4f} "
        f"hmsel={row.get('hierarchical_mmdit_stage_selector_entropy', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_selector_max', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_selector_exploration', 0.0):.2f} "
        f"hmselq={row.get('hierarchical_mmdit_stage_selector_query_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_selector_same_block_query_change', 0.0):.3f} "
        f"hmexit={row.get('hierarchical_mmdit_exit_probability', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_exit_candidate_rate', 0.0):.3f} "
        f"hmoracle={row.get('hierarchical_mmdit_oracle_route_loss', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_oracle_route_weight', 0.0):.3f} "
        f"hmodepth={row.get('hierarchical_mmdit_oracle_target_depth', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_oracle_predicted_depth', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_oracle_depth_accuracy', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_oracle_predicted_regret', 0.0):.4f} "
        f"hmstage={_format_hierarchical_stage_usage(row)} "
        f"hmblock={_format_hierarchical_block_usage(row)} "
        f"hmopgain={row.get('hierarchical_mmdit_action_noisy_operator_gain', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_operator_gain', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_operator_gain', 0.0):.3f} "
        f"hmdir={row.get('hierarchical_mmdit_action_noisy_direction_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_direction_change', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_direction_change', 0.0):.3f} "
        f"hmbcos2={row.get('hierarchical_mmdit_action_noisy_direction_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_direction_cosine', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_direction_cosine', 0.0):+.2f} "
        f"hmbgain={row.get('hierarchical_mmdit_action_noisy_base_data_gain', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_base_data_gain', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_base_data_gain', 0.0):.2f} "
        f"hmbprm={row.get('hierarchical_mmdit_action_noisy_base_parameter_rms', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_stage_base_parameter_rms', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_low_base_parameter_rms', 0.0):.2f} "
        f"hmgate={row.get('hierarchical_mmdit_action_self_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_noisy_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_stage_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_base_gate', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_ffn_base_gate', 0.0):.3f} "
        f"hmgerr={max(row.get(f'hierarchical_mmdit_action_{name}_gate_scale_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hmnrms={row.get('hierarchical_mmdit_action_pre_norm_rms', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_post_norm_rms', 0.0):.3f} "
        f"hmbound={max(row.get(f'hierarchical_mmdit_action_{name}_boundary_identity_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hmnexp={max(row.get(f'hierarchical_mmdit_action_{name}_nonexpansive_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hmnest={max(row.get(f'hierarchical_mmdit_action_{name}_nested_order_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hmbasis={max(row.get(f'hierarchical_mmdit_action_{name}_basis_norm_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e}/"
        f"{max(row.get(f'hierarchical_mmdit_action_{name}_basis_orthogonality_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
        f"hexh={row.get('hierarchical_mmdit_executed_steps', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_action_response_rel', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_rel', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_refine_gain', 0.0):+.4f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_unresolved_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_budget_exhausted_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_final_block', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_final_stage', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_early_exit_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_block_advance_rate', 0.0):.2f}/"
        f"{row.get('hierarchical_mmdit_stage_advance_rate', 0.0):.2f} "
        f"hmuresp={row.get('hierarchical_mmdit_action_response_arm', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_gripper', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_arm_null', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_gripper_null', 0.0):.3f} "
        f"hmuq={row.get('hierarchical_mmdit_action_response_p25', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_p75', 0.0):.3f} "
        f"hmpq={row.get('hierarchical_mmdit_stage_pressure_p25', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_p75', 0.0):.3f} "
        f"hmuT50={row.get('hierarchical_mmdit_action_response_t0_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_t1_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_response_t2_p50', 0.0):.3f} "
        f"hmpT50={row.get('hierarchical_mmdit_stage_pressure_t0_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_t1_p50', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_stage_pressure_t2_p50', 0.0):.3f} "
        f"hmucT={row.get('hierarchical_mmdit_response_gain_corr_t0', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr_t1', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_response_gain_corr_t2', 0.0):+.2f} "
        f"hmpcT={row.get('hierarchical_mmdit_pressure_gain_corr_t0', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t1', 0.0):+.2f}/"
        f"{row.get('hierarchical_mmdit_pressure_gain_corr_t2', 0.0):+.2f} "
        f"hmnmax={row.get('hierarchical_mmdit_action_noisy_attention_max', 0.0):.3f} "
        f"hmsmax={row.get('hierarchical_mmdit_action_stage_attention_max', 0.0):.3f} "
        f"hmlmax={row.get('hierarchical_mmdit_action_low_attention_max', 0.0):.3f} "
        f"hmnfT={row.get('hierarchical_mmdit_noisy_update_fraction_t0_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t0_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_noisy_update_fraction_t1_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t1_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_noisy_update_fraction_t2_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t2_count', 0.0), 1e-6):.3f} "
        f"hmwfT={row.get('hierarchical_mmdit_workspace_update_fraction_t0_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t0_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_workspace_update_fraction_t1_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t1_count', 0.0), 1e-6):.3f}/"
        f"{row.get('hierarchical_mmdit_workspace_update_fraction_t2_sum', 0.0) / max(row.get('hierarchical_mmdit_update_fraction_t2_count', 0.0), 1e-6):.3f} "
        f"halreff={row.get('hierarchical_mmdit_action_low_role_effective_count', 0.0):.2f} "
        f"hal={row.get('hierarchical_mmdit_action_low_role_geom_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_transition_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_event_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_state_attention', 0.0):.3f}/"
        f"{row.get('hierarchical_mmdit_action_low_role_layer_attention', 0.0):.3f} "
        f"olupd={row.get('owned_hierarchical_low_role_geom_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_transition_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_event_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_state_update_norm', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_low_role_layer_update_norm', 0.0):.3f} "
        f"ohsupd={row.get('owned_hierarchical_stage_update_norm', 0.0):.3f} "
        f"ohsret={row.get('owned_hierarchical_stage_retain_mean', 0.0):.3f} "
        f"ohprom={row.get('owned_hierarchical_manager_promote_gate', 0.0):.3f} "
        f"ohprms={row.get('owned_hierarchical_stage_promoted_projected_rms', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_stage_promoted_normalized_rms', 0.0):.3f}/"
        f"{row.get('owned_hierarchical_stage_promoted_realized_scale', 0.0):.3f} "
        f"ohgerr={row.get('owned_hierarchical_stage_promote_gate_scale_error', 0.0):.1e} "
        f"ohmrole={row.get('owned_hierarchical_manager_role_entropy', 0.0):.3f} "
        f"owned={row.get('owned_hierarchical_manager_fixed_output_prior', 0.0):.0f}/"
        f"{row.get('owned_hierarchical_manager_fixed_role_prior', 0.0):.0f}/"
        f"{row.get('owned_hierarchical_low_role_stratified', 0.0):.0f} "
        f"hmdgrad={row.get('grad_hierarchical_mmdit_action', 0.0):.3e} "
        f"hmvgrad={row.get('grad_hierarchical_mmdit_velocity_head', 0.0):.3e} "
        f"icgrad={row.get('grad_intent_contract_compiler', 0.0):.3e} "
        f"owgrad={row.get('grad_owned_workspace', 0.0):.3e} "
        f"hmbgrad={row.get('grad_hierarchical_mmdit_blocks', 0.0):.3e} "
        f"hmbasegrad={row.get('grad_hierarchical_mmdit_shared_base', 0.0):.3e} "
        f"hmwgrad={row.get('grad_hierarchical_mmdit_base_projection', 0.0):.3e} "
        f"hmcopgrad={row.get('grad_hierarchical_mmdit_contractions', 0.0):.3e} "
        f"hmcgrad={row.get('grad_hierarchical_mmdit_contraction_basis', 0.0):.3e}/"
        f"{row.get('grad_hierarchical_mmdit_contraction_depth', 0.0):.3e} "
        f"hmselgrad={row.get('grad_hierarchical_mmdit_stage_selector', 0.0):.3e} "
        f"hmexitgrad={row.get('grad_hierarchical_mmdit_exit_controller', 0.0):.3e}/"
        f"{row.get('grad_hierarchical_mmdit_exit_controller_post_clip', 0.0):.3e} "
        f"hmcmodgrad={row.get('grad_hierarchical_mmdit_content_modulation', 0.0):.3e} "
        f"hmgategrad={row.get('grad_hierarchical_mmdit_host_gates', 0.0):.3e} "
        f"hmdclip={row.get('grad_hierarchical_mmdit_action_post_clip', 0.0):.3e} "
        f"grad={row['grad']:.3e} lr={learning_rate:.3e} spb={seconds_per_batch:.3f}"
    )


def _module_grad_norm(module: torch.nn.Module, *, reference: Tensor) -> Tensor:
    """Return a detached scalar grad norm for diagnostics only."""
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for param in module.parameters():
        if param.grad is None:
            continue
        total = total + param.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _parameter_grad_norm(
    parameters: Iterable[torch.nn.Parameter], *, reference: Tensor
) -> Tensor:
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().pow(2).sum()
    return total.sqrt()


def _linear_row_grad_norm(
    modules: Iterable[torch.nn.Linear],
    *,
    start: int,
    stop: int | None,
    reference: Tensor,
) -> Tensor:
    """Gradient norm for an output-row contract inside shared Linear owners."""
    total = torch.zeros((), device=reference.device, dtype=torch.float32)
    for module in modules:
        weight_grad = module.weight.grad
        if weight_grad is not None:
            total = total + weight_grad.detach().float()[start:stop].pow(2).sum()
        if module.bias is not None and module.bias.grad is not None:
            total = total + module.bias.grad.detach().float()[start:stop].pow(2).sum()
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
    losses["grad_controlled_dynamics"] = _module_grad_norm(planner.controlled_dynamics, reference=reference)
    final_modules = [
        planner.final_norm,
        planner.direct_physical_head,
        planner.rollout_residual_head,
        planner.controlled_dynamics,
        planner.event_probe,
        planner.motion_probe,
    ]
    if getattr(planner, "residual_action_flow_denoiser", None) is not None:
        losses["grad_residual_action_flow"] = _module_grad_norm(planner.residual_action_flow_denoiser, reference=reference)
    if getattr(planner, "latent_main_action_decoder", None) is not None:
        losses["grad_latent_main_action"] = _module_grad_norm(planner.latent_main_action_decoder, reference=reference)
    if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
        decoder = planner.hierarchical_mmdit_action_decoder
        losses["grad_hierarchical_mmdit_action"] = _module_grad_norm(decoder, reference=reference)
        losses["grad_hierarchical_mmdit_velocity_head"] = _module_grad_norm(
            decoder.velocity_head, reference=reference,
        )
        losses["grad_intent_contract_compiler"] = _module_grad_norm(
            decoder.intent_compiler, reference=reference,
        )
        losses["grad_condition_organizer"] = _module_grad_norm(
            decoder.organizer, reference=reference,
        )
        losses["grad_owned_workspace"] = _module_grad_norm(
            decoder.workspace, reference=reference,
        )
        blocks = decoder.blocks
        losses["grad_hierarchical_mmdit_blocks"] = _module_grad_norm(
            blocks, reference=reference,
        )
        shared_base_modules = torch.nn.ModuleList([
            module
            for block in blocks
            for module in (
                block.self_qkv,
                block.self_out,
                block.cross_q,
                block.noisy_kv,
                block.stage_kv,
                block.low_kv,
                block.noisy_out,
                block.stage_out,
                block.low_out,
                block.ffn,
            )
        ])
        losses["grad_hierarchical_mmdit_shared_base"] = _module_grad_norm(
            shared_base_modules, reference=reference,
        )
        losses["grad_hierarchical_mmdit_distinct_base"] = losses[
            "grad_hierarchical_mmdit_shared_base"
        ]
        base_projection_modules = torch.nn.ModuleList([
            module
            for block in blocks
            for module in (
                block.self_out,
                block.noisy_out,
                block.stage_out,
                block.low_out,
                block.ffn.net[2],
            )
        ])
        losses["grad_hierarchical_mmdit_base_projection"] = _module_grad_norm(
            base_projection_modules, reference=reference,
        )
        losses["grad_hierarchical_mmdit_contractions"] = _module_grad_norm(
            decoder.operator_contractions, reference=reference,
        )
        losses["grad_hierarchical_mmdit_contraction_basis"] = _parameter_grad_norm(
            decoder.factor_parameters(), reference=reference,
        )
        losses["grad_hierarchical_mmdit_contraction_depth"] = _parameter_grad_norm(
            (
                parameter
                for bank in decoder.operator_contractions
                for contraction in bank.values()
                for parameter in (contraction.depth_weight, contraction.depth_bias)
            ),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_stage_selector"] = _parameter_grad_norm(
            decoder.stage_selector_parameters(), reference=reference,
        )
        losses["grad_hierarchical_mmdit_exit_controller"] = _parameter_grad_norm(
            decoder.exit_controller_parameters(), reference=reference,
        )
        losses["grad_hierarchical_mmdit_content_modulation"] = _linear_row_grad_norm(
            (block.mod for block in blocks),
            start=0,
            stop=2 * int(decoder.hidden_size),
            reference=reference,
        )
        losses["grad_hierarchical_mmdit_host_gates"] = _linear_row_grad_norm(
            (block.mod for block in blocks),
            start=2 * int(decoder.hidden_size),
            stop=None,
            reference=reference,
        )
    if getattr(planner, "latent_cvae_action_decoder", None) is not None:
        decoder = planner.latent_cvae_action_decoder
        losses["grad_latent_cvae_action"] = _module_grad_norm(decoder, reference=reference)
        hierarchical_workspace = getattr(decoder, "hierarchical_workspace", None)
        legacy_workspace = getattr(decoder, "evidence_workspace", None)
        workspace = hierarchical_workspace if hierarchical_workspace is not None else legacy_workspace
        if workspace is not None:
            losses["grad_latent_cvae_workspace"] = _module_grad_norm(workspace, reference=reference)
            if hierarchical_workspace is not None:
                losses["grad_latent_cvae_hierarchical_workspace"] = _module_grad_norm(
                    hierarchical_workspace, reference=reference,
                )
                losses["grad_latent_cvae_hierarchical_manager"] = _module_grad_norm(
                    hierarchical_workspace.manager, reference=reference,
                )
                low_modules = torch.nn.ModuleList([
                    hierarchical_workspace.condition_query,
                    hierarchical_workspace.low_stage_query,
                    hierarchical_workspace.low_stage_role_key,
                    hierarchical_workspace.low_stage_content_key,
                    hierarchical_workspace.low_stage_role_value,
                    hierarchical_workspace.low_stage_content_value,
                    hierarchical_workspace.low_stage_out,
                    hierarchical_workspace.low_blocks,
                    hierarchical_workspace.low_final_norm,
                ])
                losses["grad_latent_cvae_hierarchical_low"] = _module_grad_norm(
                    low_modules, reference=reference,
                )
                stage_modules = torch.nn.ModuleList([
                    hierarchical_workspace.stage_init,
                    hierarchical_workspace.stage_role_query,
                    hierarchical_workspace.stage_content_query,
                    hierarchical_workspace.stage_condition_query,
                    hierarchical_workspace.stage_low_key,
                    hierarchical_workspace.stage_low_value,
                    hierarchical_workspace.stage_promote_out,
                    hierarchical_workspace.stage_gru,
                    hierarchical_workspace.stage_content_out,
                    hierarchical_workspace.stage_role_out,
                ])
                losses["grad_latent_cvae_hierarchical_stage"] = _module_grad_norm(
                    stage_modules, reference=reference,
                )
            primary_modules: list[torch.nn.Module] = []
            primary_modules.extend(
                block.mod for block in getattr(decoder, "blocks", [])
                if hasattr(block, "mod")
            )
            primary_modules.extend(
                block.action_mod for block in getattr(decoder, "mmdit_blocks", [])
                if hasattr(block, "action_mod")
            )
            primary_modules.extend(
                block.mod for block in (
                    getattr(workspace, "low_blocks", [])
                    if hierarchical_workspace is not None
                    else getattr(workspace, "blocks", [])
                )
                if hasattr(block, "mod")
            )
            if primary_modules:
                losses["grad_latent_cvae_primary_modulation"] = _module_grad_norm(
                    torch.nn.ModuleList(primary_modules),
                    reference=reference,
                )
        else:
            rollout_projection = getattr(decoder, "mmdit_rollout_cond_proj", None)
            if rollout_projection is not None:
                losses["grad_latent_cvae_rollout_condition"] = _module_grad_norm(rollout_projection, reference=reference)
    final = torch.nn.ModuleList(final_modules)
    losses["grad_final_policy_heads"] = _module_grad_norm(final, reference=reference)


def _unique_params(
    sources: Sequence[torch.nn.Module | torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for source in sources:
        candidates = (source,) if isinstance(source, torch.nn.Parameter) else source.parameters()
        for param in candidates:
            ident = id(param)
            if param.requires_grad and ident not in seen:
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
    complete_latent_decoder = (
        getattr(planner, "latent_cvae_action_decoder", None) is not None
        or getattr(planner, "latent_main_action_decoder", None) is not None
        or getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None
    )
    legacy_action_readers = [] if complete_latent_decoder else [planner.direct_physical_head, planner.rollout_residual_head]
    legacy_motion_readers = [] if complete_latent_decoder else [planner.motion_probe]

    def add_hierarchical_decoder_groups(*, lr: float, name: str) -> None:
        decoder = getattr(planner, "hierarchical_mmdit_action_decoder", None)
        if decoder is None:
            return
        factor_params = list(decoder.factor_parameters())
        contraction_control_params = list(decoder.contraction_control_parameters())
        base_scale_params = list(decoder.scale_invariant_base_parameters())
        factor_ids = {id(parameter) for parameter in factor_params}
        contraction_control_ids = {id(parameter) for parameter in contraction_control_params}
        base_scale_ids = {id(parameter) for parameter in base_scale_params}
        owner_sets = (
            factor_ids,
            contraction_control_ids,
            base_scale_ids,
        )
        if any(
            left & right
            for index, left in enumerate(owner_sets)
            for right in owner_sets[index + 1:]
        ):
            raise RuntimeError("hierarchical MMDiT optimizer parameter owners overlap")
        special_ids = set().union(*owner_sets)
        regular_params = [
            parameter for parameter in decoder.parameters()
            if parameter.requires_grad and id(parameter) not in special_ids
        ]
        if regular_params:
            groups.append({"params": regular_params, "lr": lr, "name": name})
        if factor_params:
            groups.append({
                "params": factor_params,
                "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                "weight_decay": 0.0,
                "name": f"{name}_contraction_basis_no_decay",
            })
        if contraction_control_params:
            groups.append({
                "params": contraction_control_params,
                "lr": lr * float(trainer.hierarchical_mmdit_contraction_lr_scale),
                "weight_decay": 0.0,
                "name": f"{name}_contraction_depth_no_decay",
            })
        if base_scale_params:
            groups.append({
                "params": base_scale_params,
                "lr": lr * float(trainer.hierarchical_mmdit_shared_base_lr_scale),
                "weight_decay": 0.0,
                "name": f"{name}_scale_invariant_base_no_decay",
            })

    if _uses_layer_adapter_contract(trainer) and len(getattr(planner, "layer_contract_heads", [])) > 0:
        shared_modules = [
            planner.visual_memory,
            planner.rollout_codec,
            planner.seed,
            planner.time,
            planner.content_mod,
            planner.content_mod_scale,
        ]
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
            final_modules = [planner.final_norm, *legacy_action_readers, planner.controlled_dynamics, *legacy_motion_readers]
            if not event_probe_in_contract:
                final_modules.append(planner.event_probe)
            if getattr(planner, "residual_action_flow_denoiser", None) is not None:
                final_modules.append(planner.residual_action_flow_denoiser)
            if getattr(planner, "latent_main_action_decoder", None) is not None:
                final_modules.append(planner.latent_main_action_decoder)
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                final_modules.append(planner.latent_cvae_action_decoder)
            if float(getattr(trainer, "layer_contract_final_action_loss_weight", 0.0)) > 0:
                groups.append({"params": _unique_params(final_modules), "lr": trainer.lr * float(getattr(trainer, "layer_contract_final_action_lr_scale", 0.30)), "name": "weak_final_policy_probe"})
                add_hierarchical_decoder_groups(
                    lr=trainer.lr * float(getattr(trainer, "layer_contract_final_action_lr_scale", 0.30)),
                    name="weak_hierarchical_mmdit_action_decoder",
                )
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
            final_modules = [planner.final_norm, *legacy_action_readers, planner.controlled_dynamics, planner.event_probe, *legacy_motion_readers]
            groups.append({"params": _unique_params(final_modules), "lr": trainer.lr, "name": "final_policy_heads"})
            if getattr(planner, "residual_action_flow_denoiser", None) is not None:
                groups.append({
                    "params": list(planner.residual_action_flow_denoiser.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "action_flow_residual_lr_scale", 1.5)),
                    "name": "residual_action_flow_denoiser",
                })
            if getattr(planner, "latent_main_action_decoder", None) is not None:
                groups.append({
                    "params": list(planner.latent_main_action_decoder.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "latent_action_decoder_lr_scale", 1.5)),
                    "name": "latent_main_action_decoder",
                })
            if getattr(planner, "latent_cvae_action_decoder", None) is not None:
                groups.append({
                    "params": list(planner.latent_cvae_action_decoder.parameters()),
                    "lr": trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    "name": "latent_cvae_action_decoder",
                })
            if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
                add_hierarchical_decoder_groups(
                    lr=trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                    name="hierarchical_mmdit_action_decoder",
                )
            groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
        return [group for group in groups if len(group["params"]) > 0]

    pre_modules = [
        planner.visual_memory,
        planner.rollout_codec,
        planner.seed,
        planner.time,
        planner.content_mod,
        planner.content_mod_scale,
        *list(planner.blocks[:cut]),
        planner.midcut_norm,
    ]
    mid_modules = [planner.midcut_heads]
    post_modules = [
        *list(planner.blocks[cut:]),
        planner.final_norm,
        *legacy_action_readers,
        planner.controlled_dynamics,
        planner.event_probe,
        *legacy_motion_readers,
    ]
    if stage in {"contract", "stage1"}:
        groups.append({"params": _unique_params(pre_modules), "lr": trainer.lr, "name": "pre_midcut_trunk"})
        groups.append({"params": _unique_params(mid_modules), "lr": trainer.lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)), "name": "midcut_contract_heads"})
        groups.append({"params": list(system.proposal.parameters()), "lr": trainer.proposal_lr, "name": "proposal"})
    else:
        upper_lr = trainer.lr * float(getattr(trainer, "upper_lr_scale", 0.20))
        groups.append({"params": _unique_params(pre_modules), "lr": upper_lr, "name": "pre_midcut_trunk_low_lr"})
        groups.append({"params": _unique_params(mid_modules), "lr": upper_lr * float(getattr(trainer, "midcut_head_lr_scale", 1.0)), "name": "midcut_contract_heads_low_lr"})
        groups.append({"params": _unique_params(post_modules), "lr": trainer.lr, "name": "post_midcut_policy"})
        if getattr(planner, "residual_action_flow_denoiser", None) is not None:
            groups.append({
                "params": list(planner.residual_action_flow_denoiser.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "action_flow_residual_lr_scale", 1.5)),
                "name": "residual_action_flow_denoiser",
            })
        if getattr(planner, "latent_main_action_decoder", None) is not None:
            groups.append({
                "params": list(planner.latent_main_action_decoder.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "latent_action_decoder_lr_scale", 1.5)),
                "name": "latent_main_action_decoder",
            })
        if getattr(planner, "latent_cvae_action_decoder", None) is not None:
            groups.append({
                "params": list(planner.latent_cvae_action_decoder.parameters()),
                "lr": trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                "name": "latent_cvae_action_decoder",
            })
        if getattr(planner, "hierarchical_mmdit_action_decoder", None) is not None:
            add_hierarchical_decoder_groups(
                lr=trainer.lr * float(getattr(trainer, "latent_cvae_action_decoder_lr_scale", 1.0)),
                name="hierarchical_mmdit_action_decoder",
            )
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
    # do_before_v78 M1 quarantine proof: CALLGRAPH_AUDIT=1 turns this run into
    # a one-batch diagnostic -- hooks attach here, the report is written right
    # after the first backward (plus one sampled eval batch), then the run
    # exits.  Zero effect when the env var is absent.
    callgraph_auditor = None
    import os as _os
    if _os.environ.get("CALLGRAPH_AUDIT", "0") == "1":
        from clearvla.tools.callgraph_audit import CallGraphAuditor
        callgraph_auditor = CallGraphAuditor(system)
        callgraph_auditor.attach(first_phase="train")
        print("[callgraph-audit] hooks attached; diagnostic run, will exit after first batch", flush=True)
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
        saved_policy = payload.get("policy_config", {})
        saved_final_decoder = str(saved_policy.get("final_action_decoder", "legacy"))
        current_final_decoder = str(getattr(system.policy_config, "final_action_decoder", "legacy"))
        if saved_final_decoder != current_final_decoder:
            raise ValueError(
                "resume final-action-decoder mismatch: "
                f"checkpoint={saved_final_decoder!r}, current={current_final_decoder!r}"
            )
        if current_final_decoder == "hierarchical_mmdit_action":
            saved_architecture = str(
                saved_policy.get("hierarchical_mmdit_architecture_version", "competitive_v1")
            )
            current_architecture = str(
                getattr(
                    system.policy_config,
                    "hierarchical_mmdit_architecture_version",
                    "post_gate_contraction_sidecar_v11_oracle_router",
                )
            )
            if saved_architecture != current_architecture:
                raise ValueError(
                    "resume hierarchical-MMDiT architecture mismatch: "
                    f"checkpoint={saved_architecture!r}, current={current_architecture!r}; "
                    "use the checkpoint as --stage1-checkpoint to retain the trunk while "
                    "reinitializing the final decoder"
                )
            for field in (
                "hierarchical_mmdit_depth",
                "hierarchical_mmdit_refine_steps",
                "hierarchical_mmdit_low_slots",
                "hierarchical_mmdit_stage_slots",
                "hierarchical_mmdit_noisy_causal",
                "hierarchical_mmdit_noisy_gate_mode",
                "hierarchical_mmdit_output_contract",
                "hierarchical_mmdit_operator_stages",
                "hierarchical_mmdit_operator_rank",
                "hierarchical_mmdit_operator_groups",
                "hierarchical_mmdit_operator_contraction_warmup_steps",
                "hierarchical_mmdit_operator_contraction_transition_steps",
            ):
                saved_value = int(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = int(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                    )
            for field in (
                "hierarchical_mmdit_ffn_expansion",
                "hierarchical_mmdit_noisy_gate_min",
                "hierarchical_mmdit_noisy_gate_power",
                "hierarchical_mmdit_operator_depth_logit_init",
                "hierarchical_mmdit_residual_scale_init",
                "hierarchical_mmdit_residual_scale_max",
                "hierarchical_mmdit_random_prefix_probability",
            ):
                saved_value = float(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = float(getattr(system.policy_config, field))
                if not math.isclose(saved_value, current_value, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value}, current={current_value}"
                    )
            for field in ("hierarchical_mmdit_schedule_mode",):
                saved_value = str(saved_policy.get(field, getattr(system.policy_config, field)))
                current_value = str(getattr(system.policy_config, field))
                if saved_value != current_value:
                    raise ValueError(
                        f"resume {field} mismatch: checkpoint={saved_value!r}, current={current_value!r}"
                    )
            runtime_override_fields = (
                "hierarchical_mmdit_exhaustion_mode",
                "hierarchical_mmdit_action_response_thresholds",
                "hierarchical_mmdit_stage_pressure_thresholds",
                "hierarchical_mmdit_action_response_floor",
                "hierarchical_mmdit_exhaustion_confirm_steps",
            )
            for field in runtime_override_fields:
                saved_value = saved_policy.get(field, getattr(system.policy_config, field))
                current_value = getattr(system.policy_config, field)
                if field.endswith("_thresholds"):
                    saved_value = tuple(float(value) for value in saved_value)
                    current_value = tuple(float(value) for value in current_value)
                if saved_value != current_value:
                    print(
                        f"[v39-resume] runtime refinement override {field}: "
                        f"checkpoint={saved_value!r} current={current_value!r}",
                        flush=True,
                    )
        saved_workspace_tokens = int(saved_policy.get("latent_cvae_horizon_tokens", 24))
        current_workspace_tokens = int(getattr(system.policy_config, "latent_cvae_horizon_tokens", 24))
        if saved_workspace_tokens != current_workspace_tokens:
            raise ValueError(
                "resume workspace-token mismatch: "
                f"checkpoint={saved_workspace_tokens}, current={current_workspace_tokens}"
            )
        saved_hierarchical = int(saved_policy.get("latent_cvae_hierarchical_workspace", 0))
        current_hierarchical = int(getattr(system.policy_config, "latent_cvae_hierarchical_workspace", 0))
        if saved_hierarchical != current_hierarchical:
            raise ValueError(
                "resume hierarchical-workspace mismatch: "
                f"checkpoint={saved_hierarchical}, current={current_hierarchical}"
            )
        if current_hierarchical:
            saved_stage_slots = int(saved_policy.get("latent_cvae_stage_slots", 6))
            current_stage_slots = int(getattr(system.policy_config, "latent_cvae_stage_slots", 6))
            if saved_stage_slots != current_stage_slots:
                raise ValueError(
                    "resume stage-slot mismatch: "
                    f"checkpoint={saved_stage_slots}, current={current_stage_slots}"
                )
        system.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); schedule.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1; global_step = int(payload["global_step"])
        history = list(payload.get("history", [])); best.update(payload.get("best", {})); restore_rng(payload.get("rng"))

    for epoch in range(start_epoch, trainer.epochs + 1):
        system.train(); metric_sums: dict[str, Tensor] = {}; metric_count = 0
        throughput_start = time.perf_counter()
        throughput_batch = 0
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
            hierarchical_decoder = getattr(
                system.planner, "hierarchical_mmdit_action_decoder", None
            )
            if hierarchical_decoder is not None:
                hierarchical_decoder.set_operator_contraction_training_step(global_step)
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
            if not torch.isfinite(losses["loss"].detach()).all():
                raise FloatingPointError(
                    f"non-finite training loss before backward at epoch={epoch} batch={batch_index}"
                )
            losses["loss"].float().backward()
            _attach_grad_diagnostics(losses, system)
            if callgraph_auditor is not None:
                callgraph_auditor.capture_gradients()
                callgraph_auditor.begin_phase("sample")
                with torch.no_grad():
                    evaluate_v39_policy(
                        system=system,
                        loader=val_loader,
                        conditioner=conditioner,
                        device=device,
                        dtype=dtype,
                        camera_names=camera_names,
                        action_normalizer=action_normalizer,
                        trainer=trainer,
                        max_batches=1,
                    )
                callgraph_auditor.detach()
                report_path = callgraph_auditor.write_report(
                    out_dir / "callgraph_audit",
                    context_note=(
                        f"epoch={epoch} batch={batch_index} decoder="
                        + ("hierarchical" if getattr(system.planner, "hierarchical_mmdit_action_decoder", None) is not None else "legacy")
                    ),
                )
                print(f"[callgraph-audit] report written to {report_path}; exiting diagnostic run", flush=True)
                return {"callgraph_audit": str(report_path)}
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_backward", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            latent_decoder = getattr(system.planner, "latent_cvae_action_decoder", None)
            clean_decoder = getattr(system.planner, "hierarchical_mmdit_action_decoder", None)
            decoder_for_local_clip = clean_decoder if clean_decoder is not None else latent_decoder
            exit_controller_params = (
                list(clean_decoder.exit_controller_parameters())
                if clean_decoder is not None else []
            )
            exit_controller_ids = {
                id(parameter) for parameter in exit_controller_params
            }
            latent_clip = float(getattr(trainer, "latent_cvae_grad_clip", 0.0))
            if decoder_for_local_clip is not None and latent_clip > 0:
                local_clip_params = [
                    parameter for parameter in decoder_for_local_clip.parameters()
                    if id(parameter) not in exit_controller_ids
                ]
                torch.nn.utils.clip_grad_norm_(
                    local_clip_params,
                    latent_clip,
                    error_if_nonfinite=True,
                )
                clip_key = (
                    "grad_hierarchical_mmdit_action_post_clip"
                    if clean_decoder is not None else "grad_latent_cvae_action_post_clip"
                )
                losses[clip_key] = _parameter_grad_norm(
                    local_clip_params, reference=losses["loss"],
                )
            main_clip_params = [
                parameter for parameter in system.parameters()
                if id(parameter) not in exit_controller_ids
            ]
            grad = torch.nn.utils.clip_grad_norm_(
                main_clip_params,
                trainer.grad_clip,
                error_if_nonfinite=True,
            )
            if exit_controller_params:
                torch.nn.utils.clip_grad_norm_(
                    exit_controller_params,
                    trainer.grad_clip,
                    error_if_nonfinite=True,
                )
                losses["grad_hierarchical_mmdit_exit_controller_post_clip"] = (
                    _parameter_grad_norm(
                        exit_controller_params, reference=losses["loss"],
                    )
                )
            if report_mem and memory_reporter.detail:
                memory_reporter.snapshot(tag="train_after_clip", epoch=epoch, batch=batch_index, global_step=global_step, extra={"use_future": bool(use_future)})
            optimizer.step(); schedule.step(); global_step += 1
            if report_mem:
                memory_reporter.snapshot(tag="train_after_step", epoch=epoch, batch=batch_index, global_step=global_step, print_line=True, extra={"use_future": bool(use_future)})
            _accumulate_metric_tensors(metric_sums, losses, grad=grad)
            metric_count += 1
            if trainer.log_every and batch_index % trainer.log_every == 0:
                row = _sync_loss_row(losses, grad=grad)
                throughput_now = time.perf_counter()
                throughput_count = max(batch_index - throughput_batch, 1)
                seconds_per_batch = (throughput_now - throughput_start) / float(throughput_count)
                throughput_start = throughput_now
                throughput_batch = batch_index
                print(
                    _owned_serial_log_line(
                        row,
                        epoch=epoch,
                        batch_index=batch_index,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        seconds_per_batch=seconds_per_batch,
                    ) if clean_decoder is not None else (
                    f"[v39-layer] epoch={epoch:03d} batch={batch_index:04d} loss={row['loss']:.6f} "
                    f"pflow={row['physical_flow']:.6f} pflowu={row.get('physical_flow_uniform', row['physical_flow']):.6f} "
                    f"pfn={row.get('physical_flow_native', 0.0):.6f} pfnu={row.get('physical_flow_native_uniform', 0.0):.6f} "
                    f"afmd={row.get('arm_fm_per_dim', 0.0):.5f} gfmf={row.get('gripper_fm_field', 0.0):.5f} "
                    f"afmn={row.get('arm_fm_native', 0.0):.5f} afmnull={row.get('arm_fm_null', 0.0):.5f} "
                    f"afmnrms={row.get('arm_fm_null_rms', 0.0):.4f} afmnf={row.get('arm_fm_null_output_fraction', 0.0):.4f} "
                    f"afmproj={row.get('arm_fm_target_projection_error', 0.0):.2e} "
                    f"afmnoise={row.get('arm_fm_noise_projection_error', 0.0):.2e} "
                    f"anstd={row.get('arm_noise_abs_std', 0.0):.3f}/{row.get('arm_noise_delta_std', 0.0):.3f} "
                    f"atstd={row.get('arm_target_abs_std', 0.0):.3f}/{row.get('arm_target_delta_std', 0.0):.3f} "
                    f"gfar={row.get('gripper_arm_fm_ratio', 0.0):.3f} gfmv={row.get('gripper_fm_value', 0.0):.5f} gfmd={row.get('gripper_fm_delta', 0.0):.5f} "
                    f"gfme={row.get('gripper_fm_event', 0.0):.5f} gfmh={row.get('gripper_fm_hold', 0.0):.5f} "
                    f"gfmem={row.get('gripper_fm_event_loss_mass', 0.0):.3f} "
                    f"gfmew={row.get('gripper_fm_event_emphasis_mean', 0.0):.2f}/"
                    f"{row.get('gripper_fm_hold_emphasis_mean', 0.0):.2f} "
                    f"gfmn={row.get('gripper_fm_native', 0.0):.5f} gfmnull={row.get('gripper_fm_null', 0.0):.5f} "
                    f"gfmnrms={row.get('gripper_fm_null_rms', 0.0):.4f} gfmnf={row.get('gripper_fm_null_output_fraction', 0.0):.4f} "
                    f"gfnehr={row.get('gripper_fm_null_event_hold_ratio', 0.0):.2f} "
                    f"gfmproj={row.get('gripper_fm_target_projection_error', 0.0):.2e} "
                    f"gfmer={row.get('gripper_fm_target_energy_ratio', 0.0):.3f} "
                    f"decode={row['decoded_action']:.6f} rollout={row.get('rollout_dynamics', 0.0):.6f} "
                    f"rvar={row.get('rollout_variance', 0.0):.4f} rnorm={row.get('rollout_norm', 0.0):.4f} "
                    f"rstep={row.get('rollout_milestone_delta_match', 0.0):.4f} "
                    f"first8={row.get('first8_physical_flow', 0.0):.6f} tail={row.get('tail_physical_flow', 0.0):.6f} "
                    f"delta={row.get('rollout_delta', 0.0):.6f} contrast={row.get('rollout_contrast', 0.0):.6f} "
                    f"d_shuffle={row.get('rollout_delta_shuffle', 0.0):.6f} "
                    f"rbase={row.get('rollout_base_norm', 0.0):.3f} "
                    f"rexp={row.get('rollout_decomposition_expansion_ratio', 0.0):.3f} "
                    f"rcancel={row.get('rollout_shuffle_cancellation_fraction', 0.0):.3f} "
                    f"rbleak={row.get('rollout_base_change_shuffle', 0.0):.2e} "
                    f"stdr={row.get('rollout_pred_std_ratio', 0.0):.4f} dnratio={row.get('rollout_milestone_delta_norm_ratio', 0.0):.4f} "
                    f"rdeep={row.get('rollout_deep_update_norm', 0.0):.2f} "
                    f"rdnorm={row.get('rollout_deep_token_norm', 0.0):.2f} "
                    f"event={row['event']:.6f} "
                    f"cz={row.get('latent_cvae_prior_z_norm', row.get('latent_cvae_z_norm', 0.0)):.2f} "
                    f"cpz={row.get('latent_cvae_post_z_norm', 0.0):.2f} "
                    f"cmug={row.get('latent_cvae_mu_gap', 0.0):.2f} "
                    f"ckl={row.get('latent_cvae_kl', 0.0):.4f} "
                    f"cpflow={row.get('latent_cvae_post_flow', 0.0):.4f} "
                    f"cstd={row.get('latent_cvae_prior_std', 0.0):.3f} "
                    f"cgate={row.get('latent_cvae_gripper_gate_mean', 0.0):.3f} "
                    f"clmem={row.get('latent_cvae_layer_memory_count', 0.0):.1f} "
                    f"cscan={row.get('latent_cvae_condition_scan_norm', 0.0):.2f} "
                    f"clat={row.get('latent_cvae_condition_lateral_norm', 0.0):.2f} "
                    f"craw={row.get('latent_cvae_condition_raw_norm', 0.0):.2f} "
                    f"zcond={row.get('latent_cvae_primary_condition_norm', 0.0):.2f} "
                    f"zfx={row.get('latent_cvae_primary_z_effect_norm', 0.0):.2f} "
                    f"clsum={row.get('latent_cvae_layer_summary_norm', 0.0):.2f} "
                    f"ctraw={row.get('latent_cvae_transition_source_raw_norm', 0.0):.2f} "
                    f"ctmem={row.get('latent_cvae_transition_condition_norm', 0.0):.2f} "
                    f"ccscale={row.get('latent_cvae_consequence_scale_mean', 0.0):.3f} "
                    f"ccpref={row.get('latent_cvae_consequence_gate_preference', 0.0):.3f} "
                    f"ccmix={row.get('latent_cvae_consequence_mix_ratio', 0.0):.3f} "
                    f"cadu={row.get('latent_cvae_adaptive_refine_update_mean', 0.0):.3f} "
                    f"cxgate={row.get('latent_cvae_adaptive_noisy_gate_mean', 0.0):.3f} "
                    f"xnorm={row.get('latent_cvae_adaptive_noisy_branch_norm', 0.0):.3f} "
                    f"volpar={row.get('mmdit_noisy_volume_parity', 0.0):.3f} "
                    f"xinfl={row.get('mmdit_noisy_influence_ratio', 0.0):.2f} "
                    f"crmax={row.get('latent_cvae_adaptive_route_max', 0.0):.3f} "
                    f"crent={row.get('latent_cvae_adaptive_route_entropy', 0.0):.3f} "
                    f"creff={row.get('latent_cvae_adaptive_route_effective_slots', 0.0):.2f} "
                    f"cprmax={row.get('latent_cvae_adaptive_progress_max', 0.0):.3f} "
                    f"cprent={row.get('latent_cvae_adaptive_progress_entropy', 0.0):.3f} "
                    f"cpeff={row.get('latent_cvae_adaptive_progress_effective_slots', 0.0):.2f} "
                    f"cprog={row.get('latent_cvae_adaptive_progress_norm', 0.0):.2f} "
                    f"ccont={row.get('latent_cvae_adaptive_continue_mean', 0.0):.3f} "
                    f"cprefix={row.get('latent_cvae_adaptive_prefix_norm', 0.0):.2f} "
                    f"czseed={row.get('latent_cvae_adaptive_progress_seed_norm', 0.0):.3f} "
                    f"czseff={row.get('latent_cvae_adaptive_progress_seed_effective_slots', 0.0):.2f} "
                    f"ctemp={row.get('latent_cvae_adaptive_route_temperature_mean', 0.0):.2f} "
                    f"rtime={row.get('latent_cvae_adaptive_route_time_query_norm', 0.0):.3f} "
                    f"cfunc={row.get('latent_cvae_adaptive_function_delta_norm', 0.0):.3f} "
                    f"cbasehf={row.get('latent_cvae_adaptive_base_highfreq_norm', 0.0):.3f} "
                    f"cstep={row.get('latent_cvae_adaptive_refine_step_bias_norm', 0.0):.3f} "
                    f"ccmax={row.get('latent_cvae_adaptive_capsule_layer_max', 0.0):.3f} "
                    f"ccleff={row.get('latent_cvae_adaptive_capsule_layer_effective_slots', 0.0):.2f} "
                    f"cstr={row.get('latent_cvae_adaptive_condition_strength_mean', 0.0):.3f} "
                    f"ccond={row.get('latent_cvae_adaptive_condition_residual_norm', 0.0):.3f} "
                    f"mdu={row.get('latent_cvae_mmdit_action_update_norm', 0.0):.3f} "
                    f"mdcu={row.get('latent_cvae_mmdit_cond_update_norm', 0.0):.3f} "
                    f"mdca={row.get('latent_cvae_mmdit_action_cond_attention', 0.0):.3f} "
                    f"mdna={row.get('latent_cvae_mmdit_action_noisy_attention', 0.0):.3f} "
                    f"mdat={row.get('latent_cvae_mmdit_action_token_norm', 0.0):.2f} "
                    f"mdct={row.get('latent_cvae_mmdit_condition_token_norm', 0.0):.2f} "
                    f"mdnt={row.get('latent_cvae_mmdit_noisy_token_norm', 0.0):.2f} "
                    f"mdwa={row.get('latent_cvae_mmdit_action_workspace_attention', 0.0):.3f} "
                    f"mdwe={row.get('latent_cvae_mmdit_action_workspace_enrichment', 0.0):.3f} "
                    f"mdla={row.get('latent_cvae_mmdit_action_low_attention', 0.0):.3f} "
                    f"mdsa={row.get('latent_cvae_mmdit_action_stage_attention', 0.0):.3f} "
                    f"mdle={row.get('latent_cvae_mmdit_action_low_enrichment', 0.0):.3f} "
                    f"mdse={row.get('latent_cvae_mmdit_action_stage_enrichment', 0.0):.3f} "
                    f"mdnaT={row.get('latent_cvae_mmdit_noisy_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_noisy_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_noisy_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                    f"mdwaT={row.get('latent_cvae_mmdit_workspace_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_workspace_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_workspace_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                    f"mdlaT={row.get('latent_cvae_mmdit_low_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_low_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_low_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                    f"mdsaT={row.get('latent_cvae_mmdit_stage_attn_t0_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t0_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_stage_attn_t1_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t1_count', 0.0), 1e-6):.3f}"
                    f"/{row.get('latent_cvae_mmdit_stage_attn_t2_sum', 0.0) / max(row.get('latent_cvae_mmdit_attn_t2_count', 0.0), 1e-6):.3f} "
                    f"mdra={row.get('latent_cvae_mmdit_action_rollout_attention', 0.0):.3f} "
                    f"mdre={row.get('latent_cvae_mmdit_action_rollout_enrichment', 0.0):.3f} "
                    f"mdrn={row.get('latent_cvae_rollout_token_norm', 0.0):.2f} "
                    f"mdrc={row.get('latent_cvae_rollout_token_count', 0.0):.0f} "
                    f"wk={row.get('latent_cvae_workspace_token_count', 0.0):.0f} "
                    f"wtok={row.get('latent_cvae_workspace_token_norm', 0.0):.2f} "
                    f"wdelta={row.get('latent_cvae_workspace_update_norm', 0.0):.2f} "
                    f"wgstate={row.get('latent_cvae_workspace_global_state_norm', 0.0):.2f} "
                    f"wgslot={row.get('latent_cvae_workspace_global_slot_delta_norm', 0.0):.3f} "
                    f"wsdiv={row.get('latent_cvae_workspace_global_slot_diversity', 0.0):.3f} "
                    f"wsrc={row.get('latent_cvae_workspace_source_count', 0.0):.0f} "
                    f"wcache={row.get('latent_cvae_workspace_cached_token_fraction', 0.0):.3f} "
                    f"wqscale={row.get('latent_cvae_workspace_noisy_query_scale', 0.0):.3f} "
                    f"went={row.get('latent_cvae_workspace_attention_entropy', 0.0):.3f} "
                    f"wmax={row.get('latent_cvae_workspace_attention_max', 0.0):.3f} "
                    f"wgent={row.get('latent_cvae_workspace_group_attention_entropy', 0.0):.3f} "
                    f"wgeff={row.get('latent_cvae_workspace_group_effective_sources', 0.0):.2f} "
                    f"wmass={row.get('latent_cvae_workspace_attention_mass_error', 0.0):.1e} "
                    f"wupd={row.get('latent_cvae_workspace_action_update_ratio', 0.0):.3f} "
                    f"wprog={row.get('latent_cvae_workspace_progress_attention', 0.0):.3f} "
                    f"wpq={row.get('latent_cvae_workspace_progress_query_norm', 0.0):.3f} "
                    f"wpupd={row.get('latent_cvae_workspace_progress_update_norm', 0.0):.3f} "
                    f"wpact={row.get('latent_cvae_workspace_progress_action_dependence', 0.0):.3f} "
                    f"rstem={row.get('latent_cvae_legacy_stem_effect_ratio', 0.0):.3f} "
                    f"zzero={row.get('latent_cvae_z_zero_delta', 0.0):.3f} "
                    f"zshuf={row.get('latent_cvae_z_shuffle_delta', 0.0):.3f} "
                    f"wscan={row.get('latent_cvae_workspace_scan_attention', 0.0):.3f} "
                    f"wlat={row.get('latent_cvae_workspace_lateral_attention', 0.0):.3f} "
                    f"wtrans={row.get('latent_cvae_workspace_transition_total_attention', row.get('latent_cvae_workspace_transition_attention', 0.0)):.3f} "
                    f"wtraj={row.get('latent_cvae_workspace_trajectory_attention', 0.0):.3f} "
                    f"wroll={row.get('latent_cvae_workspace_rollout_attention', 0.0):.3f} "
                    f"wcaps={row.get('latent_cvae_workspace_capsule_attention', 0.0):.3f} "
                    f"wroute={row.get('latent_cvae_workspace_routed_layer_attention', 0.0):.3f} "
                    f"wgeom={row.get('latent_cvae_workspace_role_geom_attention', 0.0):.3f} "
                    f"wtrn={row.get('latent_cvae_workspace_role_transition_attention', 0.0):.3f} "
                    f"wevt={row.get('latent_cvae_workspace_role_event_attention', 0.0):.3f} "
                    f"wstate={row.get('latent_cvae_workspace_role_state_attention', 0.0):.3f} "
                    f"wrlay={row.get('latent_cvae_workspace_role_layer_attention', 0.0):.3f} "
                    f"wglob={row.get('latent_cvae_workspace_role_global_attention', 0.0):.3f} "
                    f"ctrlcap={row.get('latent_cvae_workspace_controller_capacity', 0.0):.3f} "
                    f"ctrldly={row.get('latent_cvae_workspace_controller_delay', 0.0):.3f} "
                    f"ctrlent={row.get('latent_cvae_workspace_controller_role_entropy', 0.0):.3f} "
                    f"ctrlq={row.get('latent_cvae_workspace_controller_query_delta_norm', 0.0):.3f} "
                    f"hlow={row.get('latent_cvae_hierarchical_low_token_count', 0.0):.0f}/{row.get('latent_cvae_hierarchical_low_token_norm', 0.0):.2f} "
                    f"hlsel={row.get('latent_cvae_hierarchical_low_selector_stage_effective_slots', 0.0):.2f} "
                    f"hsrole={row.get('latent_cvae_hierarchical_stage_token_count', 0.0):.0f}/{row.get('latent_cvae_hierarchical_stage_role_norm', 0.0):.2f} "
                    f"hscont={row.get('latent_cvae_hierarchical_stage_content_norm', 0.0):.2f} "
                    f"hsrdiv={row.get('latent_cvae_hierarchical_stage_role_diversity', 0.0):.2f} "
                    f"hscdiv={row.get('latent_cvae_hierarchical_stage_content_diversity', 0.0):.2f} "
                    f"hsrcos={row.get('latent_cvae_hierarchical_stage_role_content_cosine', 0.0):.3f} "
                    f"hsrfrac={row.get('latent_cvae_hierarchical_stage_role_output_fraction', 0.0):.3f} "
                    f"hsupd={row.get('latent_cvae_hierarchical_stage_update_norm', 0.0):.3f} "
                    f"hsret={row.get('latent_cvae_hierarchical_stage_retain_mean', 0.0):.3f} "
                    f"hsprom={row.get('latent_cvae_hierarchical_stage_promote_scale', 0.0):.3f} "
                    f"hmrole={row.get('latent_cvae_hierarchical_manager_role_entropy', 0.0):.3f} "
                    f"hmprom={row.get('latent_cvae_hierarchical_manager_promote_gate', 0.0):.3f} "
                    f"hmlow={row.get('latent_cvae_hierarchical_manager_low_output_strength', 0.0):.3f} "
                    f"hmstage={row.get('latent_cvae_hierarchical_manager_stage_output_strength', 0.0):.3f} "
                    # V72 (S5 cleanup): dead cm* micro-controller console keys
                    # removed -- micro is config-off AND structurally excluded
                    # under mmdit_refine; the loss-dict keys remain intact for
                    # legacy non-MMDiT configs.
                    f"careg={row.get('latent_cvae_adaptive_regularizer', 0.0):.4f} "
                    f"carent={row.get('latent_cvae_adaptive_route_entropy_regularizer', 0.0):.4f} "
                    f"czbase={row.get('consequence_zero_base_shift', 0.0):.3f} "
                    f"csc={row.get('consequence_self_condition', 0.0):.0f} "
                    f"cscmse={row.get('consequence_self_condition_target_mse', 0.0):.4f} "
                    f"cscnmse={row.get('consequence_self_condition_noisy_mse', 0.0):.4f} "
                    f"cspflow={row.get('consequence_preview_flow', 0.0):.4f} "
                    f"iglob={row.get('intent_global_norm', 0.0):.2f} "
                    f"istage={row.get('intent_stage_contract_norm', 0.0):.2f} "
                    f"iread={row.get('intent_read_contract_norm', 0.0):.2f} "
                    f"icgs={row.get('intent_global_stage_cosine', 0.0):.3f} "
                    f"icgr={row.get('intent_global_read_cosine', 0.0):.3f} "
                    f"icsr={row.get('intent_stage_read_cosine', 0.0):.3f} "
                    f"hmdu={row.get('hierarchical_mmdit_action_update_norm', 0.0):.3f} "
                    f"hmur={row.get('hierarchical_mmdit_action_update_ratio', 0.0):.3f} "
                    f"hmcan={row.get('hierarchical_mmdit_action_serial_cancellation_fraction', 0.0):.3f} "
                    f"hmorth={row.get('hierarchical_mmdit_action_serial_cancellation_orthogonal_baseline', 0.0):.3f} "
                    f"hmxcan={row.get('hierarchical_mmdit_action_serial_cancellation_excess', 0.0):+.3f} "
                    f"hmbdot={row.get('hierarchical_mmdit_action_branch_weighted_cosine', 0.0):+.3f} "
                    f"hmcos={row.get('hierarchical_mmdit_action_state_cosine', 0.0):.3f} "
                    f"hmnu={row.get('hierarchical_mmdit_action_noisy_update_norm', 0.0):.3f} "
                    f"hmsu={row.get('hierarchical_mmdit_action_stage_update_norm', 0.0):.3f} "
                    f"hmlu={row.get('hierarchical_mmdit_action_low_update_norm', 0.0):.3f} "
                    f"hmnf={row.get('hierarchical_mmdit_action_noisy_update_fraction', 0.0):.3f} "
                    f"hmsf={row.get('hierarchical_mmdit_action_stage_update_fraction', 0.0):.3f} "
                    f"hmlf={row.get('hierarchical_mmdit_action_low_update_fraction', 0.0):.3f} "
                    f"hmdepth={row.get('hierarchical_mmdit_action_noisy_depth_ratio', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_depth_ratio', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_low_depth_ratio', 0.0):.2f} "
                    f"hmraw={row.get('hierarchical_mmdit_action_noisy_raw_depth_ratio', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_raw_depth_ratio', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_low_raw_depth_ratio', 0.0):.2f} "
                    f"hmedepth={row.get('hierarchical_mmdit_action_noisy_effective_depth', 0.0):.1f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_effective_depth', 0.0):.1f}/"
                    f"{row.get('hierarchical_mmdit_action_low_effective_depth', 0.0):.1f} "
                    f"hmcontract={row.get('hierarchical_mmdit_action_noisy_contraction_ratio', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_contraction_ratio', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_contraction_ratio', 0.0):.3f} "
                    f"hmhost={row.get('hierarchical_mmdit_action_noisy_host_update_rms', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_host_update_rms', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_host_update_rms', 0.0):.3f} "
                    f"hmcover={row.get('hierarchical_mmdit_action_noisy_subspace_energy_fraction', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_subspace_energy_fraction', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_subspace_energy_fraction', 0.0):.3f} "
                    f"hmremove={row.get('hierarchical_mmdit_action_noisy_removed_fraction', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_removed_fraction', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_removed_fraction', 0.0):.3f} "
                    f"hmdepthreg={row.get('hierarchical_mmdit_operator_contraction_progress', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_depth_usage_regularizer', 0.0):.4f} "
                    f"hmsel={row.get('hierarchical_mmdit_stage_selector_entropy', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_selector_max', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_selector_exploration', 0.0):.2f} "
                    f"hmselq={row.get('hierarchical_mmdit_stage_selector_query_change', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_selector_same_block_query_change', 0.0):.3f} "
                    f"hmexit={row.get('hierarchical_mmdit_exit_probability', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_exit_candidate_rate', 0.0):.3f} "
                    f"hmoracle={row.get('hierarchical_mmdit_oracle_route_loss', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_oracle_route_weight', 0.0):.3f} "
                    f"hmodepth={row.get('hierarchical_mmdit_oracle_target_depth', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_oracle_predicted_depth', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_oracle_depth_accuracy', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_oracle_predicted_regret', 0.0):.4f} "
                    f"hmstage={_format_hierarchical_stage_usage(row)} "
                    f"hmblock={_format_hierarchical_block_usage(row)} "
                    f"hmopgain={row.get('hierarchical_mmdit_action_noisy_operator_gain', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_operator_gain', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_operator_gain', 0.0):.3f} "
                    f"hmdir={row.get('hierarchical_mmdit_action_noisy_direction_change', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_direction_change', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_direction_change', 0.0):.3f} "
                    f"hmbcos2={row.get('hierarchical_mmdit_action_noisy_direction_cosine', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_direction_cosine', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_action_low_direction_cosine', 0.0):+.2f} "
                    f"hmbgain={row.get('hierarchical_mmdit_action_noisy_base_data_gain', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_base_data_gain', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_low_base_data_gain', 0.0):.2f} "
                    f"hmbprm={row.get('hierarchical_mmdit_action_noisy_base_parameter_rms', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_base_parameter_rms', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_low_base_parameter_rms', 0.0):.2f} "
                    f"hmgate={row.get('hierarchical_mmdit_action_self_base_gate', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_noisy_base_gate', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_stage_base_gate', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_low_base_gate', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_ffn_base_gate', 0.0):.3f} "
                    f"hmgerr={max(row.get(f'hierarchical_mmdit_action_{name}_gate_scale_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                    f"hmnrms={row.get('hierarchical_mmdit_action_pre_norm_rms', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_post_norm_rms', 0.0):.3f} "
                    f"hmbound={max(row.get(f'hierarchical_mmdit_action_{name}_boundary_identity_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                    f"hmnexp={max(row.get(f'hierarchical_mmdit_action_{name}_nonexpansive_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                    f"hmnest={max(row.get(f'hierarchical_mmdit_action_{name}_nested_order_violation', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                    f"hmbasis={max(row.get(f'hierarchical_mmdit_action_{name}_basis_norm_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e}/"
                    f"{max(row.get(f'hierarchical_mmdit_action_{name}_basis_orthogonality_error', 0.0) for name in ('self', 'noisy', 'stage', 'low', 'ffn')):.1e} "
                    f"hexh={row.get('hierarchical_mmdit_executed_steps', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_action_response_rel', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_pressure_rel', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_refine_gain', 0.0):+.4f}/"
                    f"{row.get('hierarchical_mmdit_response_gain_corr', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_unresolved_rate', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_budget_exhausted_rate', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_final_block', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_final_stage', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_early_exit_rate', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_block_advance_rate', 0.0):.2f}/"
                    f"{row.get('hierarchical_mmdit_stage_advance_rate', 0.0):.2f} "
                    f"hmuresp={row.get('hierarchical_mmdit_action_response_arm', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_gripper', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_arm_null', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_gripper_null', 0.0):.3f} "
                    f"hmuq={row.get('hierarchical_mmdit_action_response_p25', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_p75', 0.0):.3f} "
                    f"hmpq={row.get('hierarchical_mmdit_stage_pressure_p25', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_pressure_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_pressure_p75', 0.0):.3f} "
                    f"hmuT50={row.get('hierarchical_mmdit_action_response_t0_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_t1_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_action_response_t2_p50', 0.0):.3f} "
                    f"hmpT50={row.get('hierarchical_mmdit_stage_pressure_t0_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_pressure_t1_p50', 0.0):.3f}/"
                    f"{row.get('hierarchical_mmdit_stage_pressure_t2_p50', 0.0):.3f} "
                    f"hmucT={row.get('hierarchical_mmdit_response_gain_corr_t0', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_response_gain_corr_t1', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_response_gain_corr_t2', 0.0):+.2f} "
                    f"hmpcT={row.get('hierarchical_mmdit_pressure_gain_corr_t0', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_pressure_gain_corr_t1', 0.0):+.2f}/"
                    f"{row.get('hierarchical_mmdit_pressure_gain_corr_t2', 0.0):+.2f} "
                    f"hmnmax={row.get('hierarchical_mmdit_action_noisy_attention_max', 0.0):.3f} "
                    f"hmsmax={row.get('hierarchical_mmdit_action_stage_attention_max', 0.0):.3f} "
                    f"hmlmax={row.get('hierarchical_mmdit_action_low_attention_max', 0.0):.3f} "
                    f"halreff={row.get('hierarchical_mmdit_action_low_role_effective_count', 0.0):.2f} "
                    f"halgeo={row.get('hierarchical_mmdit_action_low_role_geom_attention', 0.0):.3f} "
                    f"haltrn={row.get('hierarchical_mmdit_action_low_role_transition_attention', 0.0):.3f} "
                    f"halevt={row.get('hierarchical_mmdit_action_low_role_event_attention', 0.0):.3f} "
                    f"halsta={row.get('hierarchical_mmdit_action_low_role_state_attention', 0.0):.3f} "
                    f"hallay={row.get('hierarchical_mmdit_action_low_role_layer_attention', 0.0):.3f} "
                    f"ogeo={row.get('owned_workspace_role_geom_attention', 0.0):.3f} "
                    f"otrn={row.get('owned_workspace_role_transition_attention', 0.0):.3f} "
                    f"oevt={row.get('owned_workspace_role_event_attention', 0.0):.3f} "
                    f"osta={row.get('owned_workspace_role_state_attention', 0.0):.3f} "
                    f"olay={row.get('owned_workspace_role_layer_attention', 0.0):.3f} "
                    f"ofixed={row.get('owned_hierarchical_manager_fixed_output_prior', 0.0):.0f} "
                    f"ofrole={row.get('owned_hierarchical_manager_fixed_role_prior', 0.0):.0f} "
                    f"ostrat={row.get('owned_hierarchical_low_role_stratified', 0.0):.0f} "
                    f"ohmrole={row.get('owned_hierarchical_manager_role_entropy', 0.0):.3f} "
                    f"ohsupd={row.get('owned_hierarchical_stage_update_norm', 0.0):.3f} "
                    f"ohprom={row.get('owned_hierarchical_manager_promote_gate', 0.0):.3f} "
                    f"ohprms={row.get('owned_hierarchical_stage_promoted_projected_rms', 0.0):.3f}/"
                    f"{row.get('owned_hierarchical_stage_promoted_normalized_rms', 0.0):.3f}/"
                    f"{row.get('owned_hierarchical_stage_promoted_realized_scale', 0.0):.3f} "
                    f"ohgerr={row.get('owned_hierarchical_stage_promote_gate_scale_error', 0.0):.1e} "
                    f"cgrad={row.get('grad_latent_cvae_action', 0.0):.3e} "
                    f"hwgrad={row.get('grad_latent_cvae_hierarchical_workspace', 0.0):.3e} "
                    f"hlgrad={row.get('grad_latent_cvae_hierarchical_low', 0.0):.3e} "
                    f"hsgrad={row.get('grad_latent_cvae_hierarchical_stage', 0.0):.3e} "
                    f"hmgrad={row.get('grad_latent_cvae_hierarchical_manager', 0.0):.3e} "
                    f"cgclip={row.get('grad_latent_cvae_action_post_clip', 0.0):.3e} "
                    f"agrad={row.get('grad_residual_action_flow', 0.0):.3e} "
                    f"rdgrad={row.get('grad_controlled_dynamics', 0.0):.3e} "
                    f"rcgrad={row.get('grad_latent_cvae_rollout_condition', 0.0):.3e} "
                    f"wgrad={row.get('grad_latent_cvae_workspace', 0.0):.3e} "
                    f"zpgrad={row.get('grad_latent_cvae_primary_modulation', 0.0):.3e} "
                    f"hmdgrad={row.get('grad_hierarchical_mmdit_action', 0.0):.3e} "
                    f"hmvgrad={row.get('grad_hierarchical_mmdit_velocity_head', 0.0):.3e} "
                    f"icgrad={row.get('grad_intent_contract_compiler', 0.0):.3e} "
                    f"owgrad={row.get('grad_owned_workspace', 0.0):.3e} "
                    f"hmbgrad={row.get('grad_hierarchical_mmdit_blocks', 0.0):.3e} "
                    f"hmbasegrad={row.get('grad_hierarchical_mmdit_shared_base', 0.0):.3e} "
                    f"hmwgrad={row.get('grad_hierarchical_mmdit_base_projection', 0.0):.3e} "
                    f"hmcopgrad={row.get('grad_hierarchical_mmdit_contractions', 0.0):.3e} "
                    f"hmcgrad={row.get('grad_hierarchical_mmdit_contraction_basis', 0.0):.3e}/"
                    f"{row.get('grad_hierarchical_mmdit_contraction_depth', 0.0):.3e} "
                    f"hmselgrad={row.get('grad_hierarchical_mmdit_stage_selector', 0.0):.3e} "
                    f"hmexitgrad={row.get('grad_hierarchical_mmdit_exit_controller', 0.0):.3e}/"
                    f"{row.get('grad_hierarchical_mmdit_exit_controller_post_clip', 0.0):.3e} "
                    f"hmcmodgrad={row.get('grad_hierarchical_mmdit_content_modulation', 0.0):.3e} "
                    f"hmgategrad={row.get('grad_hierarchical_mmdit_host_gates', 0.0):.3e} "
                    f"hmdclip={row.get('grad_hierarchical_mmdit_action_post_clip', 0.0):.3e} "
                    f"grad={row['grad']:.3e} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"spb={seconds_per_batch:.3f}"),
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
