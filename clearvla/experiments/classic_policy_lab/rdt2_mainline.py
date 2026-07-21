from __future__ import annotations

"""Consolidated ClearVLA RDT2 v28.1 mainline policy.

This is the only active RDT2 policy entry point. It selectively restores the
clean responsibilities that survived earlier experiments:

* full dense visual tokens remain available to the action workspace;
* a history-only GRU predicts a physical-prior-relative trajectory source;
* flow matching predicts only the residual around that learned source;
* first-action and near-prefix exits receive explicit supervision;
* arm flow and gripper openness use separate output semantics;
* gripper outputs remain continuous but are constrained to a fixed calibrated range;
* gripper transition windows receive an explicit auxiliary objective;
* stage-shared low-rank modulation reduces redundant motor-core hypernetworks;
* an optional query-latent visual readout can make bounded first/prefix latent corrections.

Legacy control-interface, progressive, and grounded-motor CLIs remain available
only behind an explicit environment opt-in. They are not part of the active
mainline.
"""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.experiments.residual_flow_lab.flow import (
    ResidualBridgeConfig,
    endpoint_from_velocity,
    sample_residual_bridge,
)
from .rdt2_future_latent import CleanFutureLatentDynamics, sample_future_latent_flow
from .rdt2_fm_reference import (
    Attention,
    CrossAttention,
    FeedForward,
    RMSNorm,
    TimestepEmbedder,
    get_multimodal_pos_embed,
)
from collections import OrderedDict


def _weighted_mse(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != pred.shape[1]:
        raise ValueError(f"weights must be [H={pred.shape[1]}], got {tuple(weights.shape)}")
    return ((pred - target).square() * weights.reshape(1, -1, 1)).mean()


def _relative_error_scale(reference_error: Tensor, zero_error: Tensor) -> Tensor:
    """Stable per-sample scale for demonstration-relative world errors."""

    if reference_error.shape != zero_error.shape:
        raise ValueError("reference_error and zero_error must share shape")
    return torch.maximum(reference_error.detach(), 0.10 * zero_error.detach()).clamp_min(1e-6)


def _conservative_relative_consequence_terms(
    *,
    policy_error: Tensor,
    demo_error: Tensor,
    zero_error: Tensor,
    teacher_error: Tensor,
    relative_margin: float,
    regret_cap: float,
    teacher_weight: float,
    teacher_cap: float,
) -> dict[str, Tensor]:
    """Construct conservative closed-loop transfer terms.

    The demonstrated action is a detached baseline.  Only excess error above
    that baseline is penalized; lower model error is never rewarded, which
    prevents policy optimization from exploiting an imperfect world model.
    """

    if not (policy_error.shape == demo_error.shape == zero_error.shape == teacher_error.shape):
        raise ValueError("closed-loop per-sample errors must share shape")
    if relative_margin < 0 or teacher_weight < 0 or regret_cap <= 0 or teacher_cap <= 0:
        raise ValueError("closed-loop relative settings are invalid")
    scale = _relative_error_scale(demo_error, zero_error)
    policy_minus_demo = policy_error - demo_error.detach()
    relative_regret = policy_minus_demo / scale
    relative_hinge = F.relu(relative_regret - float(relative_margin)).clamp_max(float(regret_cap))
    teacher_relative = (teacher_error / zero_error.detach().clamp_min(1e-6)).clamp_max(
        float(teacher_cap)
    )
    conservative = relative_hinge + float(teacher_weight) * teacher_relative
    return {
        "scale": scale,
        "policy_minus_demo": policy_minus_demo,
        "relative_regret": relative_regret,
        "relative_hinge": relative_hinge,
        "teacher_relative": teacher_relative,
        "conservative": conservative,
    }


def _relative_world_confidence(
    *,
    demo_error: Tensor,
    corrupted_error: Tensor,
    zero_error: Tensor,
    dependency_margin: float,
    world_skill_margin: float,
    confidence_floor: float,
) -> dict[str, Tensor]:
    """Gate policy transfer by action dependence and actual world-model skill."""

    if not (demo_error.shape == corrupted_error.shape == zero_error.shape):
        raise ValueError("world confidence errors must share shape")
    if dependency_margin <= 0 or world_skill_margin <= 0:
        raise ValueError("world confidence margins must be positive")
    if not (0.0 <= confidence_floor <= 1.0):
        raise ValueError("confidence_floor must be in [0,1]")
    scale = _relative_error_scale(demo_error, zero_error)
    dependency_gap = corrupted_error - demo_error
    dependency_relative_gap = dependency_gap / scale
    dependency_confidence = (dependency_relative_gap.detach() / float(dependency_margin)).clamp(
        0.0, 1.0
    )
    world_relative_skill = (
        (zero_error.detach() - demo_error.detach()) / zero_error.detach().clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    world_skill_confidence = (world_relative_skill / float(world_skill_margin)).clamp(0.0, 1.0)
    joint_confidence = dependency_confidence * world_skill_confidence
    joint_confidence = float(confidence_floor) + (1.0 - float(confidence_floor)) * joint_confidence
    return {
        "dependency_gap": dependency_gap,
        "dependency_relative_gap": dependency_relative_gap,
        "dependency_confidence": dependency_confidence,
        "world_relative_skill": world_relative_skill,
        "world_skill_confidence": world_skill_confidence,
        "joint_confidence": joint_confidence,
    }


def _resolve_gripper_index(action_dim: int, gripper_dim_index: int) -> int:
    index = int(gripper_dim_index)
    if index < 0:
        index += int(action_dim)
    if not (0 <= index < int(action_dim)):
        raise ValueError(
            f"gripper_dim_index={gripper_dim_index} is invalid for action_dim={action_dim}"
        )
    return index


def _gripper_transition_mask(
    openness_gt: Tensor,
    *,
    threshold: float,
    radius: int,
    past_last_openness: Tensor | None = None,
) -> Tensor:
    """Mark true continuous-openness transition windows, including chunk boundary.

    ``past_last_openness`` covers the most important boundary event:
    ``history[-1] -> future[0]``. Values are continuous openness in ``[0, 1]``.
    """

    if openness_gt.ndim != 2:
        raise ValueError(f"openness_gt must be [B,H], got {tuple(openness_gt.shape)}")
    if threshold < 0 or radius < 0:
        raise ValueError("gripper transition threshold and radius must be non-negative")
    if past_last_openness is not None and tuple(past_last_openness.shape) != (
        openness_gt.shape[0],
    ):
        raise ValueError(f"past_last_openness must be [B], got {tuple(past_last_openness.shape)}")
    mask = torch.zeros_like(openness_gt, dtype=torch.bool)
    if past_last_openness is not None:
        mask[:, 0] |= (openness_gt[:, 0] - past_last_openness).abs() >= float(threshold)
    if openness_gt.shape[1] > 1:
        changes = (openness_gt[:, 1:] - openness_gt[:, :-1]).abs() >= float(threshold)
        mask[:, 1:] |= changes
        mask[:, :-1] |= changes
    if radius:
        base = mask.clone()
        for offset in range(1, radius + 1):
            mask[:, offset:] |= base[:, :-offset]
            mask[:, :-offset] |= base[:, offset:]
    return mask


def _mean_weighted(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 2 or weights.ndim != 1 or values.shape[1] != weights.shape[0]:
        raise ValueError(
            f"expected values [B,H] and weights [H], got {tuple(values.shape)} and {tuple(weights.shape)}"
        )
    expanded = weights.reshape(1, -1).to(device=values.device, dtype=values.dtype).expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _arm_weighted_mse(
    pred: Tensor, target: Tensor, weights: Tensor, *, gripper_dim_index: int
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share [B,H,D], got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != pred.shape[1]:
        raise ValueError(f"weights must be [H={pred.shape[1]}], got {tuple(weights.shape)}")
    error = (pred - target).square()
    arm_parts = [error[..., :gripper_dim_index], error[..., gripper_dim_index + 1 :]]
    arm_error = torch.cat([part for part in arm_parts if part.shape[-1]], dim=-1)
    if arm_error.shape[-1] == 0:
        return error.new_zeros(())
    return _mean_weighted(arm_error.mean(dim=-1), weights)


def _bounded_gripper_loss(
    predicted_openness: Tensor,
    target_openness: Tensor,
    weights: Tensor,
    *,
    past_last_openness: Tensor,
    transition_boost: float,
    transition_aux_weight: float,
    transition_threshold: float,
    transition_radius: int,
    smooth_weight: float,
) -> dict[str, Tensor]:
    """Continuous gripper supervision with fixed-scale semantics.

    All timesteps receive SmoothL1 supervision. Ground-truth transition windows
    receive additional weight, and a delta-matching term encourages correct
    timing without suppressing legitimate continuous motion.
    """

    if predicted_openness.shape != target_openness.shape or predicted_openness.ndim != 2:
        raise ValueError(
            f"predicted and target openness must share [B,H], got {tuple(predicted_openness.shape)} and {tuple(target_openness.shape)}"
        )
    if tuple(past_last_openness.shape) != (target_openness.shape[0],):
        raise ValueError(f"past_last_openness must be [B], got {tuple(past_last_openness.shape)}")
    if weights.ndim != 1 or weights.shape[0] != target_openness.shape[1]:
        raise ValueError(
            f"weights must be [H={target_openness.shape[1]}], got {tuple(weights.shape)}"
        )
    if (
        min(transition_boost, transition_aux_weight, transition_threshold, smooth_weight) < 0
        or transition_radius < 0
    ):
        raise ValueError("bounded gripper loss settings must be non-negative")
    if not torch.all((predicted_openness >= 0) & (predicted_openness <= 1)):
        raise ValueError("predicted gripper openness must stay in [0, 1]")
    target_clip_fraction = ((target_openness < 0.0) | (target_openness > 1.0)).float().mean()
    target = target_openness.clamp(0.0, 1.0)
    state_error = F.smooth_l1_loss(predicted_openness, target, reduction="none")
    mask = _gripper_transition_mask(
        target,
        threshold=transition_threshold,
        radius=transition_radius,
        past_last_openness=past_last_openness,
    )
    base = (
        weights.reshape(1, -1)
        .to(device=state_error.device, dtype=state_error.dtype)
        .expand_as(state_error)
    )
    weighted = base * (1.0 + float(transition_boost) * mask.to(dtype=state_error.dtype))
    state_loss = (state_error * weighted).sum() / weighted.sum().clamp_min(1.0)
    transition_loss = state_error[mask].mean() if mask.any() else state_error.new_zeros(())
    pred_with_boundary = torch.cat([past_last_openness.unsqueeze(1), predicted_openness], dim=1)
    target_with_boundary = torch.cat([past_last_openness.unsqueeze(1), target], dim=1)
    delta_loss = F.smooth_l1_loss(
        pred_with_boundary[:, 1:] - pred_with_boundary[:, :-1],
        target_with_boundary[:, 1:] - target_with_boundary[:, :-1],
    )
    loss = (
        state_loss
        + float(transition_aux_weight) * transition_loss
        + float(smooth_weight) * delta_loss
    )
    return {
        "loss": loss,
        "state_smooth_l1": state_loss,
        "transition_smooth_l1": transition_loss,
        "delta_smooth_l1": delta_loss,
        "transition_fraction": mask.float().mean(),
        "openness_mean": predicted_openness.mean(),
        "target_clip_fraction": target_clip_fraction,
    }


class ArmOnlyProjection(nn.Module):
    """Train only continuous arm flow; insert a fixed zero gripper velocity."""

    def __init__(self, in_features: int, action_dim: int, gripper_dim_index: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.gripper_dim_index = _resolve_gripper_index(action_dim, gripper_dim_index)
        self.arm = nn.Linear(in_features, self.action_dim - 1)

    @property
    def out_features(self) -> int:
        return self.action_dim

    def zero_init(self) -> None:
        nn.init.zeros_(self.arm.weight)
        nn.init.zeros_(self.arm.bias)

    def forward(self, x: Tensor) -> Tensor:
        arm = self.arm(x)
        zeros = arm.new_zeros((*arm.shape[:-1], 1))
        index = self.gripper_dim_index
        return torch.cat([arm[..., :index], zeros, arm[..., index:]], dim=-1)


class BoundedContinuousGripperHead(nn.Module):
    """Predict continuous openness while preserving the calibrated physical scale.

    A zero-initialized residual starts exactly from the current held openness.
    Positive and negative residuals consume only the available distance to the
    corresponding boundary, so outputs remain in ``[0, 1]`` without binary
    quantization or a free unbounded regression head.
    """

    def __init__(self, hidden_size: int, norm_eps: float, *, residual_scale: float) -> None:
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.residual_scale = float(residual_scale)
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor, base_openness: Tensor) -> Tensor:
        if x.ndim != 3 or tuple(base_openness.shape) != tuple(x.shape[:2]):
            raise ValueError(
                f"expected x [B,H,D] and base_openness [B,H], got {tuple(x.shape)} and {tuple(base_openness.shape)}"
            )
        base = base_openness.clamp(0.0, 1.0)
        delta = (
            torch.tanh(self.fc2(F.silu(self.fc1(self.norm(x)))).squeeze(-1)) * self.residual_scale
        )
        # Consume only the available distance to the corresponding bound. This
        # preserves the current value at initialization and supports every
        # continuous openness inside the calibrated interval.
        available = torch.where(delta >= 0, 1.0 - base, base)
        return (base + delta * available).clamp(0.0, 1.0)


class SplitActionProjection(nn.Module):
    """Separate arm and gripper projections while preserving the original action order."""

    def __init__(self, in_features: int, action_dim: int, gripper_dim_index: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.gripper_dim_index = _resolve_gripper_index(action_dim, gripper_dim_index)
        self.arm = nn.Linear(in_features, self.action_dim - 1)
        self.gripper = nn.Linear(in_features, 1)

    @property
    def out_features(self) -> int:
        return self.action_dim

    def zero_init(self) -> None:
        nn.init.zeros_(self.arm.weight)
        nn.init.zeros_(self.arm.bias)
        nn.init.zeros_(self.gripper.weight)
        nn.init.zeros_(self.gripper.bias)

    def forward(self, x: Tensor) -> Tensor:
        arm = self.arm(x)
        grip = self.gripper(x)
        index = self.gripper_dim_index
        return torch.cat([arm[..., :index], grip, arm[..., index:]], dim=-1)


class MainlineFinalLayer(nn.Module):
    def __init__(
        self, action_dim: int, gripper_dim_index: int, *, hidden_size: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = ArmOnlyProjection(4 * hidden_size, action_dim, gripper_dim_index)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        modulated = self.ffn_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.fc2(F.silu(self.fc1(modulated)))


def _prefix_weights(
    horizon: int,
    *,
    first: float,
    first4: float,
    first8: float,
    tail: float,
    device: torch.device | None = None,
) -> Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min(first, first4, first8, tail) <= 0:
        raise ValueError("prefix weights must be positive")
    weights = torch.full((horizon,), float(tail), device=device)
    weights[: min(8, horizon)] = float(first8)
    weights[: min(4, horizon)] = float(first4)
    weights[0] = float(first)
    return weights / weights.mean()


def _chunk_execution_weights(
    horizon: int,
    *,
    first4: float,
    middle: float,
    late: float,
    tail: float,
    device: torch.device | None = None,
) -> Tensor:
    """Balanced weights for training chunks that will be executed as chunks.

    The old mainline objective was intentionally first-heavy because deployment
    focused on fast/early exits.  Chunk execution needs the entire 24-step plan
    to stay meaningful, so this schedule keeps the middle and late horizons
    comparably important while still leaving a small stabilizing bias for the
    first part of the chunk.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min(first4, middle, late, tail) <= 0:
        raise ValueError("chunk execution weights must be positive")
    weights = torch.full((horizon,), float(tail), device=device)
    weights[: min(20, horizon)] = float(late)
    weights[: min(12, horizon)] = float(middle)
    weights[: min(4, horizon)] = float(first4)
    return weights / weights.mean()


def _horizon_weights(
    horizon: int,
    *,
    mode: str,
    first: float,
    first4: float,
    first8: float,
    tail: float,
    chunk_first4: float,
    chunk_middle: float,
    chunk_late: float,
    chunk_tail: float,
    device: torch.device | None = None,
) -> Tensor:
    if mode == "prefix":
        return _prefix_weights(
            horizon, first=first, first4=first4, first8=first8, tail=tail, device=device
        )
    if mode == "uniform":
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        return torch.ones((horizon,), device=device)
    if mode == "chunk-balanced":
        return _chunk_execution_weights(
            horizon,
            first4=chunk_first4,
            middle=chunk_middle,
            late=chunk_late,
            tail=chunk_tail,
            device=device,
        )
    raise ValueError(f"unknown horizon weight mode: {mode}")


def _arm_delta_matching_loss(
    pred_actions: Tensor,
    target_actions: Tensor,
    weights: Tensor,
    *,
    past_last_action: Tensor,
    gripper_dim_index: int,
) -> Tensor:
    """Match the per-step arm motion of an executable action chunk.

    This is not a generic smoothness penalty.  It compares predicted and target
    action deltas, including the history-to-future boundary, so legitimate
    large movements are encouraged when the demonstration contains them.
    """

    if pred_actions.shape != target_actions.shape or pred_actions.ndim != 3:
        raise ValueError(
            f"pred and target actions must share [B,H,D], got {tuple(pred_actions.shape)} and {tuple(target_actions.shape)}"
        )
    if tuple(past_last_action.shape) != (target_actions.shape[0], target_actions.shape[2]):
        raise ValueError(f"past_last_action must be [B,D], got {tuple(past_last_action.shape)}")
    if weights.ndim != 1 or weights.shape[0] != target_actions.shape[1]:
        raise ValueError(
            f"weights must be [H={target_actions.shape[1]}], got {tuple(weights.shape)}"
        )
    pred_with_boundary = torch.cat([past_last_action.unsqueeze(1), pred_actions], dim=1)
    target_with_boundary = torch.cat([past_last_action.unsqueeze(1), target_actions], dim=1)
    delta_error = F.smooth_l1_loss(
        pred_with_boundary[:, 1:] - pred_with_boundary[:, :-1],
        target_with_boundary[:, 1:] - target_with_boundary[:, :-1],
        reduction="none",
    )
    arm_parts = [delta_error[..., :gripper_dim_index], delta_error[..., gripper_dim_index + 1 :]]
    arm_error = torch.cat([part for part in arm_parts if part.shape[-1]], dim=-1)
    if arm_error.shape[-1] == 0:
        return delta_error.new_zeros(())
    return _mean_weighted(arm_error.mean(dim=-1), weights)


def _close_pre_mask(
    openness_gt: Tensor,
    *,
    threshold: float,
    pre_steps: int,
    past_last_openness: Tensor,
) -> Tensor:
    """Mark the horizon steps immediately before true close transitions."""

    if openness_gt.ndim != 2:
        raise ValueError(f"openness_gt must be [B,H], got {tuple(openness_gt.shape)}")
    if tuple(past_last_openness.shape) != (openness_gt.shape[0],):
        raise ValueError(f"past_last_openness must be [B], got {tuple(past_last_openness.shape)}")
    if threshold < 0 or pre_steps < 0:
        raise ValueError("close phase threshold and pre_steps must be non-negative")
    close = torch.zeros_like(openness_gt, dtype=torch.bool)
    close[:, 0] |= (openness_gt[:, 0] - past_last_openness) >= float(threshold)
    if openness_gt.shape[1] > 1:
        close[:, 1:] |= (openness_gt[:, 1:] - openness_gt[:, :-1]) >= float(threshold)
    mask = torch.zeros_like(close)
    for offset in range(1, pre_steps + 1):
        mask[:, :-offset] |= close[:, offset:]
    return mask


def _masked_arm_endpoint_loss(
    pred_actions: Tensor,
    target_actions: Tensor,
    mask: Tensor,
    *,
    gripper_dim_index: int,
) -> Tensor:
    if pred_actions.shape != target_actions.shape or pred_actions.ndim != 3:
        raise ValueError(
            f"pred and target actions must share [B,H,D], got {tuple(pred_actions.shape)} and {tuple(target_actions.shape)}"
        )
    if tuple(mask.shape) != tuple(target_actions.shape[:2]):
        raise ValueError(f"mask must be [B,H], got {tuple(mask.shape)}")
    error = F.smooth_l1_loss(pred_actions, target_actions, reduction="none")
    arm_parts = [error[..., :gripper_dim_index], error[..., gripper_dim_index + 1 :]]
    arm_error = torch.cat([part for part in arm_parts if part.shape[-1]], dim=-1)
    if arm_error.shape[-1] == 0 or not mask.any():
        return error.new_zeros(())
    return arm_error.mean(dim=-1)[mask].mean()


@dataclass(frozen=True)
class MainlineRDT2FMConfig:
    action_dim: int = 7
    state_dim: int = 7
    prediction_horizon: int = 24
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    num_register_tokens: int = 4
    norm_eps: float = 1e-5
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    use_flash_attn: bool = True
    num_inference_timesteps: int = 5

    # Dense-token plugin contract.  The external-KV path leaves these unset.
    lang_adaptor: str | None = None
    lang_token_dim: int | None = None

    # Explicit local-motion path.
    history_hidden_size: int = 128
    history_layers: int = 1
    prior_residual_scale: float = 1.0
    history_noise_std: float = 0.01

    # Progressive exits.  Values count completed transformer blocks.
    fast_exit_layer: int = 2
    prefix_exit_layer: int = 4
    prefix_length: int = 4
    visual_start_layer: int = 2

    # Stage-shared low-rank modulation.
    modulation_rank: int = 128

    # Optional structured fast visual readout. The stable baseline uses ``none``.
    # ``query-latent`` reads top/wrist patch tokens with camera-specific queries
    # and applies only bounded latent corrections before first/prefix arm heads.
    visual_corrector: str = "none"
    visual_corrector_cameras: int = 2
    visual_top_query_tokens: int = 2
    visual_wrist_query_tokens: int = 4
    visual_query_hidden_size: int = 256
    visual_query_heads: int = 4
    visual_latent_max_scale: float = 0.10
    visual_latent_init_logit: float = -3.0
    visual_top_gate_floor: float = 0.0

    # Clean action-conditioned future-DINO residual dynamics.  The world model
    # has no shared policy parameters.  ``closed-loop`` transfers dynamics
    # supervision only through predicted action values evaluated by a detached
    # world model; future tokens never enter policy hidden states.
    future_latent_variant: str = "none"  # none | world-only | closed-loop
    future_latent_dim: int | None = None
    future_latent_offsets: tuple[int, ...] = (8, 16, 24)
    future_latent_num_cameras: int = 2
    future_latent_grid_size: int = 8
    future_latent_hidden_size: int = 768
    future_latent_depth: int = 6
    future_latent_heads: int = 8
    future_latent_kv_heads: int = 4
    future_latent_modulation_rank: int = 192
    future_world_loss_weight: float = 0.10
    future_endpoint_loss_weight: float = 0.0
    future_motion_weight: float = 1.0
    future_motion_weight_cap: float = 4.0
    future_dependency_loss_weight: float = 0.01

    # Action/future semantic closure. Ground-truth action prefixes and future
    # DINO changes are aligned in a compact space; an inverse head anchors the
    # future representation to explicit action summaries, and a detached
    # semantic evaluator closes predicted action -> future -> action cycles.
    future_action_semantic_dim: int = 256
    future_action_semantic_hidden_size: int = 256
    future_action_semantic_depth: int = 2
    future_action_semantic_heads: int = 4
    future_action_semantic_kv_heads: int = 2
    future_align_loss_weight: float = 0.05
    future_inverse_loss_weight: float = 0.10
    future_current_action_baseline_loss_weight: float = 0.02
    future_action_reconstruction_loss_weight: float = 0.05
    future_embedding_variance_loss_weight: float = 0.02
    future_embedding_covariance_loss_weight: float = 0.005
    future_contrastive_temperature: float = 0.10
    future_structured_nce_weight: float = 0.25
    future_contrastive_transition_boost: float = 1.0
    future_contrastive_duplicate_threshold: float = 1e-6
    future_embedding_std_target: float = 0.05
    future_pred_align_loss_weight: float = 0.05
    future_cycle_loss_weight: float = 0.05
    future_align_margin: float = 0.10
    future_semantic_confidence_margin: float = 0.10
    future_inverse_transition_threshold: float = 0.10
    future_semantic_warmup_steps: int = 3217
    future_semantic_ramp_steps: int = 3217
    future_action_cross_scale: float = 0.0  # <=0 selects 1/sqrt(depth)
    future_semantic_negative_delay: int = 3

    # Relative degradation required when the demonstrated action is replaced
    # by a hard negative.  Relative margins remain meaningful as the absolute
    # world-model MSE falls during training.
    future_dependency_relative_margin: float = 0.03
    future_action_time_power: float = 1.0
    future_action_time_floor: float = 0.10
    future_policy_bridge_time_power: float = 1.0
    future_policy_bridge_time_floor: float = 0.10
    # Closed-loop policy transfer is conservative and demonstration-relative:
    # it never rewards predicted actions for outperforming the demonstration
    # under an imperfect world model.  It only penalizes excess future error,
    # with an optional small teacher-consequence matching term.
    future_consistency_relative_margin: float = 0.02
    future_consistency_regret_cap: float = 2.0
    future_consistency_teacher_weight: float = 0.25
    future_consistency_teacher_cap: float = 1.0
    future_consistency_world_skill_margin: float = 0.10
    future_consistency_confidence_floor: float = 0.0
    future_consistency_weight_cap: float = 4.0
    future_consistency_loss_weight: float = 0.02
    future_consistency_warmup_steps: int = 3217
    future_consistency_ramp_steps: int = 3217
    future_latent_stat_eps: float = 1e-5

    # Horizon objective.  ``prefix`` preserves the v28.2 first-heavy schedule;
    # ``chunk-balanced`` is the v29 training default for policies deployed by
    # executing the whole action chunk.
    horizon_weight_mode: str = "prefix"
    first_position_weight: float = 8.0
    first4_position_weight: float = 4.0
    first8_position_weight: float = 2.0
    tail_position_weight: float = 1.0
    chunk_first4_position_weight: float = 1.5
    chunk_middle_position_weight: float = 1.5
    chunk_late_position_weight: float = 1.5
    chunk_tail_position_weight: float = 1.2
    prior_loss_weight: float = 0.50
    fast_exit_loss_weight: float = 1.00
    prefix_exit_loss_weight: float = 0.50
    full_flow_loss_weight: float = 1.00
    arm_delta_loss_weight: float = 0.0
    align_phase_loss_weight: float = 0.0
    align_phase_pre_steps: int = 8

    # Arm/gripper semantic split. Gripper defaults to the final action dim.
    gripper_dim_index: int = -1
    arm_flow_loss_weight: float = 1.0
    # Bounded-continuous gripper semantics. Raw calibration is checkpointed;
    # normalized endpoints are used inside the policy. Open/close may be in
    # either order, but they must remain distinct.
    gripper_output_mode: str = "bounded-continuous"
    gripper_open_raw: float = 0.0
    gripper_close_raw: float = 1.0
    gripper_open_normalized: float = -1.0
    gripper_close_normalized: float = 1.0
    gripper_openness_residual_scale: float = 1.0
    gripper_state_loss_weight: float = 2.0
    gripper_transition_boost: float = 3.0
    gripper_transition_aux_weight: float = 0.50
    gripper_transition_threshold: float = 0.10
    gripper_transition_radius: int = 1
    gripper_smooth_weight: float = 0.02

    # Residual bridge around the learned history source.
    bridge_clean_probability: float = 0.50
    bridge_mild_probability: float = 0.35
    bridge_strong_probability: float = 0.15
    bridge_mild_noise_std: float = 0.05
    bridge_strong_noise_std: float = 0.15
    bridge_mild_velocity_bias_std: float = 0.02
    bridge_strong_velocity_bias_std: float = 0.06

    def validate(self) -> None:
        positive = (
            self.action_dim,
            self.state_dim,
            self.prediction_horizon,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.num_kv_heads,
            self.history_hidden_size,
            self.history_layers,
            self.fast_exit_layer,
            self.prefix_exit_layer,
            self.prefix_length,
            self.modulation_rank,
            self.visual_corrector_cameras,
            self.visual_top_query_tokens,
            self.visual_wrist_query_tokens,
            self.visual_query_hidden_size,
            self.visual_query_heads,
        )
        if min(positive) <= 0:
            raise ValueError("mainline RDT2-FM dimensions must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if not (self.fast_exit_layer <= self.prefix_exit_layer <= self.depth):
            raise ValueError("require fast_exit_layer <= prefix_exit_layer <= depth")
        if not (0 <= self.visual_start_layer <= self.fast_exit_layer):
            raise ValueError("visual_start_layer must be in [0, fast_exit_layer]")
        if self.prefix_length > self.prediction_horizon:
            raise ValueError("prefix_length cannot exceed prediction_horizon")
        if self.lang_adaptor is not None and self.lang_token_dim is None:
            raise ValueError("lang_token_dim is required when lang_adaptor is enabled")
        if self.prior_residual_scale < 0 or self.history_noise_std < 0:
            raise ValueError("prior scales must be non-negative")
        if self.visual_corrector not in {"none", "query-latent"}:
            raise ValueError("visual_corrector must be 'none' or 'query-latent'")
        if self.visual_corrector == "query-latent" and self.visual_corrector_cameras != 2:
            raise ValueError(
                "query-latent visual corrector currently requires exactly two cameras: top and wrist"
            )
        if self.visual_query_hidden_size % self.visual_query_heads != 0:
            raise ValueError("visual_query_hidden_size must be divisible by visual_query_heads")
        if self.visual_latent_max_scale < 0:
            raise ValueError("visual_latent_max_scale must be non-negative")
        if not (0.0 <= self.visual_top_gate_floor < 1.0):
            raise ValueError("visual_top_gate_floor must be in [0, 1)")
        if self.future_latent_variant not in {"none", "world-only", "closed-loop"}:
            raise ValueError("future_latent_variant must be none, world-only, or closed-loop")
        if self.future_latent_variant != "none":
            if self.future_latent_dim is None or self.future_latent_dim <= 0:
                raise ValueError(
                    "future_latent_dim must be positive when future latent dynamics is enabled"
                )
            if not self.future_latent_offsets or any(
                int(offset) <= 0 for offset in self.future_latent_offsets
            ):
                raise ValueError("future_latent_offsets must be non-empty positive integers")
            if tuple(sorted(set(self.future_latent_offsets))) != tuple(self.future_latent_offsets):
                raise ValueError("future_latent_offsets must be strictly increasing and unique")
            if max(self.future_latent_offsets) > self.prediction_horizon:
                raise ValueError("future_latent_offsets cannot exceed prediction_horizon")
            dims = (
                self.future_latent_num_cameras,
                self.future_latent_grid_size,
                self.future_latent_hidden_size,
                self.future_latent_depth,
                self.future_latent_heads,
                self.future_latent_kv_heads,
                self.future_latent_modulation_rank,
                self.future_action_semantic_dim,
                self.future_action_semantic_hidden_size,
                self.future_action_semantic_depth,
                self.future_action_semantic_heads,
                self.future_action_semantic_kv_heads,
                self.future_semantic_negative_delay,
            )
            if min(dims) <= 0:
                raise ValueError("future latent dimensions must be positive")
            if self.future_latent_hidden_size % self.future_latent_heads != 0:
                raise ValueError(
                    "future_latent_hidden_size must be divisible by future_latent_heads"
                )
            if self.future_latent_heads % self.future_latent_kv_heads != 0:
                raise ValueError("future_latent_heads must be divisible by future_latent_kv_heads")
            if self.future_action_semantic_hidden_size % self.future_action_semantic_heads != 0:
                raise ValueError(
                    "future_action_semantic_hidden_size must be divisible by semantic heads"
                )
            if self.future_action_semantic_heads % self.future_action_semantic_kv_heads != 0:
                raise ValueError("future semantic heads must be divisible by semantic kv heads")
            if self.action_dim != self.state_dim:
                raise ValueError("action-semantic world model requires action_dim == state_dim")
            nonnegative = (
                self.future_world_loss_weight,
                self.future_endpoint_loss_weight,
                self.future_motion_weight,
                self.future_dependency_loss_weight,
                self.future_align_loss_weight,
                self.future_inverse_loss_weight,
                self.future_current_action_baseline_loss_weight,
                self.future_action_reconstruction_loss_weight,
                self.future_embedding_variance_loss_weight,
                self.future_embedding_covariance_loss_weight,
                self.future_structured_nce_weight,
                self.future_contrastive_transition_boost,
                self.future_contrastive_duplicate_threshold,
                self.future_embedding_std_target,
                self.future_pred_align_loss_weight,
                self.future_cycle_loss_weight,
                self.future_align_margin,
                self.future_semantic_confidence_margin,
                self.future_inverse_transition_threshold,
                self.future_semantic_warmup_steps,
                self.future_semantic_ramp_steps,
                self.future_action_cross_scale,
                self.future_dependency_relative_margin,
                self.future_action_time_power,
                self.future_action_time_floor,
                self.future_policy_bridge_time_power,
                self.future_policy_bridge_time_floor,
                self.future_consistency_confidence_floor,
                self.future_consistency_relative_margin,
                self.future_consistency_regret_cap,
                self.future_consistency_teacher_weight,
                self.future_consistency_teacher_cap,
                self.future_consistency_world_skill_margin,
                self.future_consistency_loss_weight,
                self.future_consistency_warmup_steps,
                self.future_consistency_ramp_steps,
            )
            if (
                min(nonnegative) < 0
                or self.future_motion_weight_cap < 1
                or self.future_consistency_regret_cap <= 0
                or self.future_consistency_teacher_cap <= 0
                or self.future_consistency_weight_cap < 1
            ):
                raise ValueError("future latent loss/schedule settings are invalid")
            if self.future_contrastive_temperature <= 0:
                raise ValueError("future_contrastive_temperature must be positive")
            if not (0.0 <= self.future_action_time_floor <= 1.0):
                raise ValueError("future_action_time_floor must be in [0,1]")
            if not (0.0 <= self.future_policy_bridge_time_floor <= 1.0):
                raise ValueError("future_policy_bridge_time_floor must be in [0,1]")
            if not (0.0 <= self.future_consistency_confidence_floor <= 1.0):
                raise ValueError("future_consistency_confidence_floor must be in [0,1]")
            if self.future_latent_stat_eps <= 0:
                raise ValueError("future_latent_stat_eps must be positive")
        if self.horizon_weight_mode not in {"prefix", "uniform", "chunk-balanced"}:
            raise ValueError("horizon_weight_mode must be 'prefix', 'uniform', or 'chunk-balanced'")
        if (
            min(
                self.first_position_weight,
                self.first4_position_weight,
                self.first8_position_weight,
                self.tail_position_weight,
                self.chunk_first4_position_weight,
                self.chunk_middle_position_weight,
                self.chunk_late_position_weight,
                self.chunk_tail_position_weight,
                self.prior_loss_weight,
                self.fast_exit_loss_weight,
                self.prefix_exit_loss_weight,
                self.full_flow_loss_weight,
            )
            <= 0
        ):
            raise ValueError("horizon and branch weights must be positive")
        if (
            self.arm_delta_loss_weight < 0
            or self.align_phase_loss_weight < 0
            or self.align_phase_pre_steps < 0
        ):
            raise ValueError("chunk-execution auxiliary loss settings must be non-negative")
        _resolve_gripper_index(self.action_dim, self.gripper_dim_index)
        if self.gripper_output_mode != "bounded-continuous":
            raise ValueError("active mainline requires gripper_output_mode='bounded-continuous'")
        if (
            self.gripper_open_raw == self.gripper_close_raw
            or self.gripper_open_normalized == self.gripper_close_normalized
        ):
            raise ValueError("gripper open and close calibration endpoints must be distinct")
        if min(self.arm_flow_loss_weight, self.gripper_state_loss_weight) <= 0:
            raise ValueError("arm_flow_loss_weight and gripper_state_loss_weight must be positive")
        if (
            min(
                self.gripper_openness_residual_scale,
                self.gripper_transition_boost,
                self.gripper_transition_aux_weight,
                self.gripper_transition_threshold,
                self.gripper_smooth_weight,
            )
            < 0
            or self.gripper_transition_radius < 0
        ):
            raise ValueError("gripper bounded-continuous settings must be non-negative")
        probabilities = (
            self.bridge_clean_probability,
            self.bridge_mild_probability,
            self.bridge_strong_probability,
        )
        if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("bridge probabilities must be non-negative and sum to 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def bridge_config(self) -> ResidualBridgeConfig:
        return ResidualBridgeConfig(
            clean_probability=self.bridge_clean_probability,
            mild_probability=self.bridge_mild_probability,
            strong_probability=self.bridge_strong_probability,
            mild_noise_std=self.bridge_mild_noise_std,
            strong_noise_std=self.bridge_strong_noise_std,
            mild_velocity_bias_std=self.bridge_mild_velocity_bias_std,
            strong_velocity_bias_std=self.bridge_strong_velocity_bias_std,
        )


class HistoryTrajectoryPrior(nn.Module):
    """Small history-only trajectory source anchored at the physical hold prior."""

    def __init__(self, config: MainlineRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h = int(config.history_hidden_size)
        self.action_in = nn.Linear(config.action_dim, h)
        self.state_in = nn.Linear(config.state_dim, h)
        self.encoder = nn.GRU(h, h, num_layers=config.history_layers, batch_first=True)
        self.future_queries = nn.Parameter(torch.randn(config.prediction_horizon, h) * 0.02)
        heads = 4 if h % 4 == 0 else 1
        self.cross = nn.MultiheadAttention(h, heads, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.residual_head = ArmOnlyProjection(h, config.action_dim, config.gripper_dim_index)
        self.context_proj = nn.Linear(h, config.hidden_size)
        self.residual_head.zero_init()

    def forward(
        self, past_actions: Tensor, state_tokens: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if past_actions.ndim != 3 or past_actions.shape[-1] != cfg.action_dim:
            raise ValueError(
                f"past_actions must be [B,H,{cfg.action_dim}], got {tuple(past_actions.shape)}"
            )
        if state_tokens.ndim != 2 or state_tokens.shape[-1] != cfg.state_dim:
            raise ValueError(
                f"state_tokens must be [B,{cfg.state_dim}], got {tuple(state_tokens.shape)}"
            )
        if tuple(physical_prior.shape[1:]) != (cfg.prediction_horizon, cfg.action_dim):
            raise ValueError(
                f"physical_prior must be [B,{cfg.prediction_horizon},{cfg.action_dim}]"
            )
        history = past_actions
        if self.training and cfg.history_noise_std > 0:
            history = history + torch.randn_like(history) * cfg.history_noise_std
        state = self.state_in(state_tokens)
        encoded, hidden = self.encoder(self.action_in(history))
        query = self.future_queries.unsqueeze(0).expand(history.shape[0], -1, -1) + state.unsqueeze(
            1
        )
        attended, _ = self.cross(query, encoded, encoded, need_weights=False)
        query = self.norm(query + attended)
        delta = torch.tanh(self.residual_head(query)) * cfg.prior_residual_scale
        prior = physical_prior + delta
        context = self.context_proj(hidden[-1] + state)
        return prior, context


class StageModulationBank(nn.Module):
    """Stage-shared low-rank modulation with tiny block-specific affine adapters."""

    def __init__(self, config: MainlineRDT2FMConfig) -> None:
        super().__init__()
        self.config = config
        h, rank = config.hidden_size, config.modulation_rank
        self.trunk = nn.Sequential(nn.SiLU(), nn.Linear(3 * h, rank), nn.SiLU())
        self.stage_heads = nn.ModuleList([nn.Linear(rank, 9 * h) for _ in range(3)])
        self.block_scale = nn.Parameter(torch.ones(config.depth, 9 * h))
        self.block_bias = nn.Parameter(torch.zeros(config.depth, 9 * h))
        for head in self.stage_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _stage(self, layer_idx: int) -> int:
        cfg = self.config
        if layer_idx < cfg.fast_exit_layer:
            return 0
        if layer_idx < cfg.prefix_exit_layer:
            return 1
        return 2

    def prepare(self, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if condition.ndim != 2 or condition.shape[1] != 3 * self.config.hidden_size:
            raise ValueError(
                f"condition must be [B,{3 * self.config.hidden_size}], got {tuple(condition.shape)}"
            )
        latent = self.trunk(condition)
        return tuple(head(latent) for head in self.stage_heads)  # type: ignore[return-value]

    def for_layer(self, prepared: tuple[Tensor, Tensor, Tensor], layer_idx: int) -> Tensor:
        base = prepared[self._stage(layer_idx)]
        return base * self.block_scale[layer_idx] + self.block_bias[layer_idx]


class MainlineRDTBlock(nn.Module):
    def __init__(self, layer_idx: int, config: MainlineRDT2FMConfig) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        core = {
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "norm_eps": config.norm_eps,
            "multiple_of": config.multiple_of,
            "ffn_dim_multiplier": config.ffn_dim_multiplier,
            "use_flash_attn": config.use_flash_attn,
        }
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(core)
        self.cross_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cond_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.cross_attn = CrossAttention(core)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = FeedForward(
            config.hidden_size,
            4 * config.hidden_size,
            config.multiple_of,
            config.ffn_dim_multiplier,
        )

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        modulation: Tensor,
        *,
        c: Tensor | None = None,
        ck: Tensor | None = None,
        cv: Tensor | None = None,
        mask: Tensor | None = None,
        use_cross_attention: bool = True,
    ) -> Tensor:
        if modulation.ndim != 2 or modulation.shape[1] != 9 * self.hidden_size:
            raise ValueError(
                f"modulation must be [B,{9 * self.hidden_size}], got {tuple(modulation.shape)}"
            )
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(9, dim=1)
        h = x + gate_attn.unsqueeze(1) * self.attn(
            self._modulate(self.attn_norm(x), shift_attn, scale_attn)
        )
        if use_cross_attention:
            if c is not None:
                cross = self.cross_attn(
                    self._modulate(self.cross_norm(h), shift_cross, scale_cross),
                    c=self.cond_norm(c),
                    mask=mask,
                )
            else:
                cross = self.cross_attn(
                    self._modulate(self.cross_norm(h), shift_cross, scale_cross),
                    ck=ck,
                    cv=cv,
                    mask=mask,
                )
            h = h + gate_cross.unsqueeze(1) * cross
        return h + gate_mlp.unsqueeze(1) * self.ffn(
            self._modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
        )


class ExitHead(nn.Module):
    """Lightweight native exit head for executable prefixes."""

    def __init__(
        self, hidden_size: int, output_size: int, gripper_dim_index: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.fc1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.fc2 = ArmOnlyProjection(2 * hidden_size, output_size, gripper_dim_index)
        self.fc2.zero_init()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(x))))


class QueryLatentVisualCorrector(nn.Module):
    """Camera-specific query readout with bounded latent-only corrections.

    The module never emits raw action deltas.  It reads full dense patch tokens
    using separate top/wrist learned queries, fuses them with state, history and
    the shallow motor hidden state, then returns small bounded latent residuals
    for the first and near-prefix arm exit heads.
    """

    def __init__(self, config: MainlineRDT2FMConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.query_hidden_size = int(config.visual_query_hidden_size)
        self.top_query_tokens = int(config.visual_top_query_tokens)
        self.wrist_query_tokens = int(config.visual_wrist_query_tokens)
        self.max_scale = float(config.visual_latent_max_scale)
        self.init_logit = float(config.visual_latent_init_logit)
        self.top_gate_floor = float(config.visual_top_gate_floor)
        qh = self.query_hidden_size
        h = self.hidden_size
        self.dense_proj = nn.Linear(h, qh)
        self.top_queries = nn.Parameter(torch.randn(1, self.top_query_tokens, qh) * 0.02)
        self.wrist_queries = nn.Parameter(torch.randn(1, self.wrist_query_tokens, qh) * 0.02)
        self.top_attn = nn.MultiheadAttention(qh, config.visual_query_heads, batch_first=True)
        self.wrist_attn = nn.MultiheadAttention(qh, config.visual_query_heads, batch_first=True)
        self.top_norm = nn.LayerNorm(qh)
        self.wrist_norm = nn.LayerNorm(qh)
        self.top_to_hidden = nn.Linear(qh, h)
        self.wrist_to_hidden = nn.Linear(qh, h)
        fusion_in = 4 * h
        self.top_gate = nn.Sequential(nn.Linear(fusion_in, h), nn.SiLU(), nn.Linear(h, 1))
        self.wrist_gate = nn.Sequential(nn.Linear(fusion_in, h), nn.SiLU(), nn.Linear(h, 1))
        self.fusion = nn.Sequential(nn.Linear(fusion_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.first_delta = nn.Linear(h, h)
        self.prefix_delta = nn.Linear(h, h)
        self.first_alpha = nn.Linear(h, 1)
        self.prefix_alpha = nn.Linear(h, 1)
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.first_delta.weight)
        nn.init.zeros_(self.first_delta.bias)
        nn.init.zeros_(self.prefix_delta.weight)
        nn.init.zeros_(self.prefix_delta.bias)
        nn.init.zeros_(self.first_alpha.weight)
        nn.init.constant_(self.first_alpha.bias, self.init_logit)
        nn.init.zeros_(self.prefix_alpha.weight)
        nn.init.constant_(self.prefix_alpha.bias, self.init_logit)

    @staticmethod
    def _split_camera_tokens(
        dense_condition: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        if dense_condition.ndim != 3:
            raise ValueError(f"dense_condition must be [B,L,D], got {tuple(dense_condition.shape)}")
        if dense_condition.shape[1] % 2:
            raise ValueError("query-latent visual corrector requires equal top/wrist token counts")
        split = dense_condition.shape[1] // 2
        top, wrist = dense_condition[:, :split], dense_condition[:, split:]
        if attention_mask is None:
            return top, wrist, None, None
        if tuple(attention_mask.shape) != tuple(dense_condition.shape[:2]):
            raise ValueError("attention_mask must match dense token [B,L]")
        return top, wrist, attention_mask[:, :split], attention_mask[:, split:]

    @staticmethod
    def _read(
        queries: Tensor,
        tokens: Tensor,
        mask: Tensor | None,
        *,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> Tensor:
        q = queries.expand(tokens.shape[0], -1, -1)
        key_padding_mask = None if mask is None else ~mask.to(dtype=torch.bool)
        out, _ = attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        return norm(q + out).mean(dim=1)

    def forward(
        self,
        *,
        state: Tensor,
        history: Tensor,
        shallow_hidden: Tensor,
        dense_condition: Tensor,
        attention_mask: Tensor | None,
    ) -> "VisualLatentCorrection":
        if state.shape != history.shape or state.shape != shallow_hidden.shape:
            raise ValueError("state, history and shallow_hidden must share [B,H]")
        top_tokens, wrist_tokens, top_mask, wrist_mask = self._split_camera_tokens(
            dense_condition, attention_mask
        )
        top_tokens = self.dense_proj(top_tokens)
        wrist_tokens = self.dense_proj(wrist_tokens)
        top_readout = self._read(
            self.top_queries, top_tokens, top_mask, attn=self.top_attn, norm=self.top_norm
        )
        wrist_readout = self._read(
            self.wrist_queries, wrist_tokens, wrist_mask, attn=self.wrist_attn, norm=self.wrist_norm
        )
        top_hidden = self.top_to_hidden(top_readout)
        wrist_hidden = self.wrist_to_hidden(wrist_readout)
        gate_context = torch.cat(
            [state, history, shallow_hidden, top_hidden + wrist_hidden], dim=-1
        )
        top_gate_raw = torch.sigmoid(self.top_gate(gate_context))
        top_gate = self.top_gate_floor + (1.0 - self.top_gate_floor) * top_gate_raw
        wrist_gate = torch.sigmoid(self.wrist_gate(gate_context))
        visual_hidden = top_gate * top_hidden + wrist_gate * wrist_hidden
        fused = self.fusion(torch.cat([state, history, shallow_hidden, visual_hidden], dim=-1))
        first_alpha = self.max_scale * torch.sigmoid(self.first_alpha(fused))
        prefix_alpha = self.max_scale * torch.sigmoid(self.prefix_alpha(fused))
        first = first_alpha * torch.tanh(self.first_delta(fused))
        prefix = prefix_alpha * torch.tanh(self.prefix_delta(fused))
        hidden_rms = shallow_hidden.square().mean().sqrt().clamp_min(1e-8)
        return VisualLatentCorrection(
            first=first,
            prefix=prefix,
            first_rms=first.square().mean().sqrt(),
            prefix_rms=prefix.square().mean().sqrt(),
            first_alpha_mean=first_alpha.mean(),
            prefix_alpha_mean=prefix_alpha.mean(),
            top_gate_mean=top_gate.mean(),
            wrist_gate_mean=wrist_gate.mean(),
            to_hidden_ratio=first.square().mean().sqrt() / hidden_rms,
        )


@dataclass
class VisualLatentCorrection:
    first: Tensor
    prefix: Tensor
    first_rms: Tensor
    prefix_rms: Tensor
    first_alpha_mean: Tensor
    prefix_alpha_mean: Tensor
    top_gate_mean: Tensor
    wrist_gate_mean: Tensor
    to_hidden_ratio: Tensor

    @staticmethod
    def zeros(reference: Tensor) -> "VisualLatentCorrection":
        scalar = reference.new_zeros(())
        hidden = reference.new_zeros(reference.shape)
        return VisualLatentCorrection(
            first=hidden,
            prefix=hidden,
            first_rms=scalar,
            prefix_rms=scalar,
            first_alpha_mean=scalar,
            prefix_alpha_mean=scalar,
            top_gate_mean=scalar,
            wrist_gate_mean=scalar,
            to_hidden_ratio=scalar,
        )


@dataclass
class MainlineVelocityOutput:
    fast_first: Tensor
    prefix: Tensor
    full: Tensor | None
    fast_gripper_openness: Tensor
    prefix_gripper_openness: Tensor
    full_gripper_openness: Tensor | None
    fast_visual_first_latent_rms: Tensor
    fast_visual_prefix_latent_rms: Tensor
    fast_visual_first_alpha_mean: Tensor
    fast_visual_prefix_alpha_mean: Tensor
    fast_visual_top_gate_mean: Tensor
    fast_visual_wrist_gate_mean: Tensor
    fast_visual_to_hidden_ratio: Tensor


class MainlineRDTCore(nn.Module):
    def __init__(self, config: MainlineRDT2FMConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.t_embedder = TimestepEmbedder(config.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList([MainlineRDTBlock(idx, config) for idx in range(config.depth)])
        self.modulation = StageModulationBank(config)
        self.final_layer = MainlineFinalLayer(
            config.action_dim,
            config.gripper_dim_index,
            hidden_size=config.hidden_size,
            norm_eps=config.norm_eps,
        )
        self.first_exit_head = ExitHead(
            config.hidden_size, config.action_dim, config.gripper_dim_index, config.norm_eps
        )
        self.prefix_exit_head = ExitHead(
            config.hidden_size, config.action_dim, config.gripper_dim_index, config.norm_eps
        )
        self.first_gripper_head = BoundedContinuousGripperHead(
            config.hidden_size,
            config.norm_eps,
            residual_scale=config.gripper_openness_residual_scale,
        )
        self.prefix_gripper_head = BoundedContinuousGripperHead(
            config.hidden_size,
            config.norm_eps,
            residual_scale=config.gripper_openness_residual_scale,
        )
        self.full_gripper_head = BoundedContinuousGripperHead(
            config.hidden_size,
            config.norm_eps,
            residual_scale=config.gripper_openness_residual_scale,
        )
        self.num_register_tokens = config.num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, config.num_register_tokens, config.hidden_size)
        )
        self.x_pos_emb = nn.Parameter(
            torch.zeros(
                1, config.prediction_horizon + config.num_register_tokens, config.hidden_size
            )
        )
        self.state_pos_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # Optional modules must not perturb the shared mainline RNG stream.
        # This keeps same-seed none/query-latent experiments strictly paired.
        if config.visual_corrector == "query-latent":
            with torch.random.fork_rng(devices=[]):
                self.visual_corrector = QueryLatentVisualCorrector(config)
        else:
            self.visual_corrector = None
        self._initialize(dtype)

    def _initialize(self, dtype: torch.dtype) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        visual_modules = (
            set(self.visual_corrector.modules()) if self.visual_corrector is not None else set()
        )
        for module in self.modules():
            if module is self or module in visual_modules:
                continue
            basic(module)
        if self.visual_corrector is not None:
            with torch.random.fork_rng(devices=[]):
                self.visual_corrector.apply(basic)
                self.visual_corrector.zero_init()
        cfg = self.config
        x_pos = get_multimodal_pos_embed(
            cfg.hidden_size,
            OrderedDict(
                [("action", cfg.prediction_horizon), ("register", cfg.num_register_tokens)]
            ),
        )
        state_pos = get_multimodal_pos_embed(cfg.hidden_size, OrderedDict([("state", 1)]))
        self.x_pos_emb.data.copy_(torch.from_numpy(x_pos).float().unsqueeze(0))
        self.state_pos_emb.data.copy_(torch.from_numpy(state_pos).float().unsqueeze(0))
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for head in self.modulation.stage_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        self.final_layer.fc2.zero_init()
        self.first_exit_head.fc2.zero_init()
        self.prefix_exit_head.fc2.zero_init()
        for gripper_head in (
            self.first_gripper_head,
            self.prefix_gripper_head,
            self.full_gripper_head,
        ):
            nn.init.zeros_(gripper_head.fc2.weight)
            nn.init.zeros_(gripper_head.fc2.bias)
        self.to(dtype=dtype)

    def _condition_for_layer(
        self,
        layer_idx: int,
        *,
        dense_condition: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if kv_cache is not None:
            key, value = kv_cache[layer_idx % len(kv_cache)]
            return None, key.transpose(1, 2), value.transpose(1, 2)
        return dense_condition, None, None

    def forward(
        self,
        *,
        x: Tensor,
        timesteps: Tensor,
        state_condition: Tensor,
        history_condition: Tensor,
        dense_condition: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        base_gripper_openness: Tensor,
        stop_after: str = "full",
    ) -> MainlineVelocityOutput:
        cfg = self.config
        if stop_after not in {"fast", "prefix", "full"}:
            raise ValueError(f"unknown stop_after={stop_after!r}")
        if dense_condition is None and kv_cache is None:
            raise ValueError("mainline RDT2-FM requires dense condition tokens or KV cache")
        if tuple(base_gripper_openness.shape) != (x.shape[0], cfg.prediction_horizon):
            raise ValueError(
                f"base_gripper_openness must be [B,{cfg.prediction_horizon}], got {tuple(base_gripper_openness.shape)}"
            )
        time = self.t_embedder(timesteps)
        if time.shape[0] == 1:
            time = time.expand(x.shape[0], -1)
        state = state_condition.squeeze(1) + self.state_pos_emb.squeeze(1)
        if state.shape != history_condition.shape:
            raise ValueError("state and history conditions must both be [B,H]")
        joint = torch.cat([time, state, history_condition], dim=-1)
        x = torch.cat([x, self.register_tokens.expand(x.shape[0], -1, -1)], dim=1) + self.x_pos_emb
        fast = prefix = full = None
        fast_gripper = prefix_gripper = full_gripper = None
        visual_latent: VisualLatentCorrection | None = None
        prepared_modulation = self.modulation.prepare(joint)
        for layer_idx, block in enumerate(self.blocks):
            use_cross = layer_idx >= cfg.visual_start_layer
            c = ck = cv = None
            if use_cross:
                c, ck, cv = self._condition_for_layer(
                    layer_idx, dense_condition=dense_condition, kv_cache=kv_cache
                )
            x = block(
                x,
                self.modulation.for_layer(prepared_modulation, layer_idx),
                c=c,
                ck=ck,
                cv=cv,
                mask=attention_mask,
                use_cross_attention=use_cross,
            )
            completed = layer_idx + 1
            if completed == cfg.fast_exit_layer:
                shallow_hidden = x[:, 0]
                if self.visual_corrector is None:
                    visual_latent = VisualLatentCorrection.zeros(shallow_hidden)
                else:
                    if dense_condition is None:
                        raise ValueError(
                            "query-latent visual corrector requires dense condition tokens, not KV-only conditioning"
                        )
                    visual_latent = self.visual_corrector(
                        state=state,
                        history=history_condition,
                        shallow_hidden=shallow_hidden,
                        dense_condition=dense_condition,
                        attention_mask=attention_mask,
                    )
                fast_hidden = x[:, :1] + visual_latent.first.unsqueeze(1)
                fast = self.first_exit_head(fast_hidden)
                fast_gripper = self.first_gripper_head(x[:, :1], base_gripper_openness[:, :1])
                if stop_after == "fast":
                    break
            if completed == cfg.prefix_exit_layer:
                if visual_latent is None:
                    raise AssertionError("fast visual latent state missing")
                prefix_hidden = x[:, : cfg.prefix_length].clone()
                decay = torch.linspace(
                    1.0, 0.25, cfg.prefix_length, device=x.device, dtype=x.dtype
                ).reshape(1, -1, 1)
                prefix_hidden = prefix_hidden + decay * visual_latent.prefix.unsqueeze(1)
                prefix = self.prefix_exit_head(prefix_hidden)
                prefix_gripper = self.prefix_gripper_head(
                    x[:, : cfg.prefix_length], base_gripper_openness[:, : cfg.prefix_length]
                )
                if stop_after == "prefix":
                    break
        if fast is None:
            raise AssertionError("fast exit was not reached")
        if prefix is None and stop_after != "fast":
            raise AssertionError("prefix exit was not reached")
        if stop_after == "full":
            final_modulation = torch.cat([time, state], dim=-1)
            full_hidden = x[:, : -cfg.num_register_tokens]
            full = self.final_layer(x, final_modulation)[:, : -cfg.num_register_tokens]
            full_gripper = self.full_gripper_head(full_hidden, base_gripper_openness)
        if visual_latent is None:
            raise AssertionError("fast visual latent state missing")
        if fast_gripper is None:
            raise AssertionError("fast gripper head was not reached")
        if prefix_gripper is None and stop_after != "fast":
            raise AssertionError("prefix gripper head was not reached")
        return MainlineVelocityOutput(
            fast_first=fast,
            prefix=fast if prefix is None else prefix,
            full=full,
            fast_gripper_openness=fast_gripper,
            prefix_gripper_openness=fast_gripper if prefix_gripper is None else prefix_gripper,
            full_gripper_openness=full_gripper,
            fast_visual_first_latent_rms=visual_latent.first_rms,
            fast_visual_prefix_latent_rms=visual_latent.prefix_rms,
            fast_visual_first_alpha_mean=visual_latent.first_alpha_mean,
            fast_visual_prefix_alpha_mean=visual_latent.prefix_alpha_mean,
            fast_visual_top_gate_mean=visual_latent.top_gate_mean,
            fast_visual_wrist_gate_mean=visual_latent.wrist_gate_mean,
            fast_visual_to_hidden_ratio=visual_latent.to_hidden_ratio,
        )


class MainlineRDT2FM(nn.Module):
    """Consolidated history-anchored residual-flow action expert."""

    def __init__(
        self,
        config: MainlineRDT2FMConfig = MainlineRDT2FMConfig(),
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.model = MainlineRDTCore(config, dtype=dtype)
        self.lang_adaptor = self._build_adapter(
            config.lang_adaptor, config.lang_token_dim, config.hidden_size
        )
        self.act_adaptor = self._build_adapter("mlp3x_silu", config.action_dim, config.hidden_size)
        self.state_adaptor = self._build_adapter("mlp3x_silu", config.state_dim, config.hidden_size)
        self.history_prior = HistoryTrajectoryPrior(config)
        if config.future_latent_variant != "none":
            if config.future_latent_dim is None:
                raise AssertionError("validated future_latent_dim missing")
            with torch.random.fork_rng(devices=[]):
                self.future_dynamics = CleanFutureLatentDynamics(
                    latent_dim=config.future_latent_dim,
                    action_dim=config.action_dim,
                    state_dim=config.state_dim,
                    action_horizon=config.prediction_horizon,
                    hidden_size=config.future_latent_hidden_size,
                    depth=config.future_latent_depth,
                    modulation_rank=config.future_latent_modulation_rank,
                    num_heads=config.future_latent_heads,
                    num_kv_heads=config.future_latent_kv_heads,
                    norm_eps=config.norm_eps,
                    multiple_of=config.multiple_of,
                    ffn_dim_multiplier=config.ffn_dim_multiplier,
                    use_flash_attn=config.use_flash_attn,
                    future_offsets=config.future_latent_offsets,
                    num_future_frames=len(config.future_latent_offsets),
                    num_cameras=config.future_latent_num_cameras,
                    grid_size=config.future_latent_grid_size,
                    dtype=dtype,
                    stat_eps=config.future_latent_stat_eps,
                    motion_weight=config.future_motion_weight,
                    motion_weight_cap=config.future_motion_weight_cap,
                    semantic_dim=config.future_action_semantic_dim,
                    semantic_hidden_size=config.future_action_semantic_hidden_size,
                    semantic_depth=config.future_action_semantic_depth,
                    semantic_heads=config.future_action_semantic_heads,
                    semantic_kv_heads=config.future_action_semantic_kv_heads,
                    gripper_dim_index=config.gripper_dim_index,
                    inverse_transition_threshold=config.future_inverse_transition_threshold,
                    action_cross_scale=config.future_action_cross_scale,
                    semantic_negative_delay=config.future_semantic_negative_delay,
                )
        else:
            self.future_dynamics = None
        self.bridge = config.bridge_config()
        self.register_buffer(
            "position_weights",
            _horizon_weights(
                config.prediction_horizon,
                mode=config.horizon_weight_mode,
                first=config.first_position_weight,
                first4=config.first4_position_weight,
                first8=config.first8_position_weight,
                tail=config.tail_position_weight,
                chunk_first4=config.chunk_first4_position_weight,
                chunk_middle=config.chunk_middle_position_weight,
                chunk_late=config.chunk_late_position_weight,
                chunk_tail=config.chunk_tail_position_weight,
            ),
            persistent=False,
        )
        self.pred_horizon = config.prediction_horizon
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.to(dtype=dtype)

    @staticmethod
    def _build_adapter(
        kind: str | None, in_features: int | None, out_features: int
    ) -> nn.Module | None:
        if kind is None:
            return None
        if in_features is None:
            raise ValueError(f"in_features required for adapter {kind}")
        if kind == "linear":
            return nn.Linear(in_features, out_features)
        match = re.match(r"^mlp(\d+)x_silu$", kind)
        if not match:
            raise ValueError(f"unknown adapter type: {kind}")
        depth = int(match.group(1))
        modules: list[nn.Module] = [nn.Linear(in_features, out_features)]
        for _ in range(1, depth):
            modules.extend([nn.SiLU(), nn.Linear(out_features, out_features)])
        return nn.Sequential(*modules)

    def _adapt_dense(self, dense_tokens: Tensor | None) -> Tensor | None:
        if dense_tokens is None:
            return None
        return self.lang_adaptor(dense_tokens) if self.lang_adaptor is not None else dense_tokens

    @property
    def future_latent_enabled(self) -> bool:
        return self.future_dynamics is not None

    def compress_current_latents(self, current_tokens: Tensor) -> Tensor:
        if self.future_dynamics is None:
            raise RuntimeError("future latent dynamics is disabled")
        return self.future_dynamics.compress_current(current_tokens)

    def compress_future_latents(self, future_tokens: Tensor) -> Tensor:
        if self.future_dynamics is None:
            raise RuntimeError("future latent dynamics is disabled")
        return self.future_dynamics.compress_future(future_tokens)

    def future_residual_target(
        self, current_compressed: Tensor, future_compressed: Tensor
    ) -> Tensor:
        if self.future_dynamics is None:
            raise RuntimeError("future latent dynamics is disabled")
        return self.future_dynamics.residual_target(current_compressed, future_compressed)

    def set_future_latent_stats(self, mean: Tensor, std: Tensor) -> None:
        if self.future_dynamics is None:
            raise RuntimeError("future latent dynamics is disabled")
        self.future_dynamics.set_residual_stats(mean, std)

    def future_latent_parameters(self):
        return [] if self.future_dynamics is None else list(self.future_dynamics.parameters())

    def policy_parameters(self):
        future_ids = (
            set()
            if self.future_dynamics is None
            else {id(p) for p in self.future_dynamics.parameters()}
        )
        return [p for p in self.parameters() if id(p) not in future_ids]

    def future_shared_parameters(self):
        # Kept as a runtime compatibility name.  The clean world objective has
        # no shared parameters; closed-loop consistency reaches these policy
        # parameters only through predicted action values.
        return self.policy_parameters()

    def future_consistency_scale(self, global_step: int) -> float:
        if self.config.future_latent_variant != "closed-loop":
            return 0.0
        warmup = int(self.config.future_consistency_warmup_steps)
        ramp = int(self.config.future_consistency_ramp_steps)
        if global_step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, max(0.0, (global_step - warmup + 1) / float(ramp)))

    def future_semantic_scale(self, global_step: int) -> float:
        """Ramp predicted-future semantic and cycle supervision only."""
        if self.config.future_latent_variant == "none":
            return 0.0
        warmup = int(self.config.future_semantic_warmup_steps)
        ramp = int(self.config.future_semantic_ramp_steps)
        if global_step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, max(0.0, (global_step - warmup + 1) / float(ramp)))

    def predict_prior(
        self, *, state_tokens: Tensor, past_actions: Tensor, physical_prior: Tensor
    ) -> tuple[Tensor, Tensor]:
        if state_tokens.ndim != 2:
            raise ValueError("state_tokens must be [B,D]")
        return self.history_prior(past_actions, state_tokens, physical_prior)

    @property
    def gripper_dim_index(self) -> int:
        return _resolve_gripper_index(self.config.action_dim, self.config.gripper_dim_index)

    def _gripper_to_openness(self, gripper_value: Tensor) -> Tensor:
        cfg = self.config
        scale = float(cfg.gripper_close_normalized - cfg.gripper_open_normalized)
        return ((gripper_value - float(cfg.gripper_open_normalized)) / scale).clamp(0.0, 1.0)

    def _openness_to_gripper(self, openness: Tensor) -> Tensor:
        cfg = self.config
        return float(cfg.gripper_open_normalized) + openness.clamp(0.0, 1.0) * float(
            cfg.gripper_close_normalized - cfg.gripper_open_normalized
        )

    def _replace_gripper(self, action: Tensor, openness: Tensor, *, length: int) -> Tensor:
        if tuple(openness.shape) != (action.shape[0], length):
            raise ValueError(f"openness must be [B,{length}], got {tuple(openness.shape)}")
        result = action.clone()
        result[:, :length, self.gripper_dim_index] = self._openness_to_gripper(openness)
        return result

    def _velocity(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        residual_state: Tensor,
        timesteps: Tensor,
        dense_tokens: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        stop_after: str,
        learned_prior: Tensor | None = None,
        history_context: Tensor | None = None,
    ) -> tuple[MainlineVelocityOutput, Tensor]:
        if learned_prior is None or history_context is None:
            learned_prior, history_context = self.predict_prior(
                state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
            )
        state = self.state_adaptor(state_tokens.unsqueeze(1))
        # The bounded gripper head must not receive target-derived bridge
        # residuals during training: inference starts from zero gripper
        # residual, so exposing this channel would create a train/inference
        # mismatch and a direct label-leak shortcut.
        arm_residual_state = residual_state.clone()
        arm_residual_state[..., self.gripper_dim_index] = 0
        action = self.act_adaptor(arm_residual_state)
        output = self.model(
            x=action,
            timesteps=timesteps,
            state_condition=state,
            history_condition=history_context,
            dense_condition=self._adapt_dense(dense_tokens),
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            base_gripper_openness=self._gripper_to_openness(
                physical_prior[..., self.gripper_dim_index]
            ),
            stop_after=stop_after,
        )
        return output, learned_prior

    def compute_loss(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        action_gt: Tensor,
        future_latent_tokens: Tensor | None = None,
        dense_tokens: Tensor | None = None,
        kv_cache: list[tuple[Tensor, Tensor]] | None = None,
        attention_mask: Tensor | None = None,
        global_step: int = 0,
        future_flow_generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        learned_prior, history_context = self.predict_prior(
            state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
        )
        bridge = sample_residual_bridge(learned_prior.detach(), action_gt, self.bridge)
        output, _ = self._velocity(
            state_tokens=state_tokens,
            past_actions=past_actions,
            physical_prior=physical_prior,
            residual_state=bridge.residual_state,
            timesteps=bridge.time,
            dense_tokens=dense_tokens,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            stop_after="full",
            learned_prior=learned_prior,
            history_context=history_context,
        )
        if output.full is None:
            raise AssertionError("full output missing")
        weights = self.position_weights.to(device=action_gt.device, dtype=action_gt.dtype)
        grip_idx = self.gripper_dim_index
        prior_arm_mse = _arm_weighted_mse(
            learned_prior, action_gt, weights, gripper_dim_index=grip_idx
        )
        full_arm_mse = _arm_weighted_mse(
            output.full, bridge.target_velocity, weights, gripper_dim_index=grip_idx
        )
        first_arm_mse = _arm_weighted_mse(
            output.fast_first,
            bridge.target_velocity[:, :1],
            weights[:1],
            gripper_dim_index=grip_idx,
        )
        prefix_len = self.config.prefix_length
        prefix_arm_mse = _arm_weighted_mse(
            output.prefix,
            bridge.target_velocity[:, :prefix_len],
            weights[:prefix_len],
            gripper_dim_index=grip_idx,
        )
        predicted_residual_endpoint = endpoint_from_velocity(
            bridge.residual_state, output.full, bridge.time
        )
        predicted_action_endpoint = learned_prior + predicted_residual_endpoint
        arm_delta_loss = _arm_delta_matching_loss(
            predicted_action_endpoint,
            action_gt,
            weights,
            past_last_action=past_actions[:, -1],
            gripper_dim_index=grip_idx,
        )
        if output.full_gripper_openness is None:
            raise AssertionError("full bounded gripper output missing")
        target_openness = self._gripper_to_openness(action_gt[..., grip_idx])
        past_last_openness = self._gripper_to_openness(past_actions[:, -1, grip_idx])
        grip_kwargs = dict(
            past_last_openness=past_last_openness,
            transition_boost=self.config.gripper_transition_boost,
            transition_aux_weight=self.config.gripper_transition_aux_weight,
            transition_threshold=self.config.gripper_transition_threshold,
            transition_radius=self.config.gripper_transition_radius,
            smooth_weight=self.config.gripper_smooth_weight,
        )
        full_grip = _bounded_gripper_loss(
            output.full_gripper_openness, target_openness, weights, **grip_kwargs
        )
        first_grip = _bounded_gripper_loss(
            output.fast_gripper_openness, target_openness[:, :1], weights[:1], **grip_kwargs
        )
        prefix_grip = _bounded_gripper_loss(
            output.prefix_gripper_openness,
            target_openness[:, :prefix_len],
            weights[:prefix_len],
            **grip_kwargs,
        )
        close_align_mask = _close_pre_mask(
            target_openness.clamp(0.0, 1.0),
            threshold=self.config.gripper_transition_threshold,
            pre_steps=self.config.align_phase_pre_steps,
            past_last_openness=past_last_openness,
        )
        align_phase_loss = _masked_arm_endpoint_loss(
            predicted_action_endpoint,
            action_gt,
            close_align_mask,
            gripper_dim_index=grip_idx,
        )

        def branch(arm: Tensor, grip: Tensor) -> Tensor:
            return (
                self.config.arm_flow_loss_weight * arm
                + self.config.gripper_state_loss_weight * grip
            ) / (self.config.arm_flow_loss_weight + self.config.gripper_state_loss_weight)

        policy_total = (
            self.config.prior_loss_weight * prior_arm_mse
            + self.config.fast_exit_loss_weight * branch(first_arm_mse, first_grip["loss"])
            + self.config.prefix_exit_loss_weight * branch(prefix_arm_mse, prefix_grip["loss"])
            + self.config.full_flow_loss_weight * branch(full_arm_mse, full_grip["loss"])
            + self.config.arm_delta_loss_weight * arm_delta_loss
            + self.config.align_phase_loss_weight * align_phase_loss
        )
        future_world_objective = policy_total.new_zeros(())
        future_consistency_objective = policy_total.new_zeros(())
        future_flow_train_objective = policy_total.new_zeros(())
        future_align_train_objective = policy_total.new_zeros(())
        future_inverse_train_objective = policy_total.new_zeros(())
        future_pred_align_train_objective = policy_total.new_zeros(())
        future_cycle_train_objective = policy_total.new_zeros(())
        future_consistency_scale = self.future_consistency_scale(global_step)
        future_semantic_scale = self.future_semantic_scale(global_step)
        future_metrics: dict[str, Tensor] = {}
        if self.future_latent_enabled:
            if future_latent_tokens is None:
                raise ValueError(
                    "future_latent_tokens are required when future latent dynamics is enabled"
                )
            if dense_tokens is None or kv_cache is not None:
                raise ValueError(
                    "future latent dynamics requires dense current-visual conditioning"
                )
            future_module = self.future_dynamics
            if future_module is None:
                raise AssertionError("future dynamics module missing")

            # Frozen DINO observations and demonstrations supervise an isolated
            # world model.  The pure action encoder cannot read current visual
            # tokens; state is used only to form action-relative displacement.
            current_compressed = future_module.compress_current(dense_tokens.detach())
            future_compressed = future_module.compress_future(future_latent_tokens.detach())
            residual_raw = future_module.residual_target(current_compressed, future_compressed)
            residual_normalized = future_module.normalize_residual(residual_raw)
            future_flow = sample_future_latent_flow(
                residual_normalized,
                generator=future_flow_generator,
            )
            motion_weights = future_module.motion_weights(residual_raw)
            world_velocity, gate_metrics, world_aux = future_module.forward_with_aux(
                current_compressed=current_compressed,
                action_chunk=action_gt.detach(),
                state=state_tokens.detach(),
                past_last_action=past_actions[:, -1].detach(),
                future_noisy=future_flow.noisy,
                future_time=future_flow.time,
            )
            future_metrics = future_module.flow_metrics(
                world_velocity,
                future_flow,
                current_compressed=current_compressed,
                future_compressed=future_compressed,
                residual_raw=residual_raw,
                motion_weights=motion_weights,
            )
            future_metrics.update(gate_metrics)

            # --- Ground-truth action <-> future-change semantic closure. ---
            semantic_targets = future_module.build_action_semantic_targets(
                action_gt.detach(),
                state=state_tokens.detach(),
                past_last_action=past_actions[:, -1].detach(),
            )
            action_embedding = world_aux["action_embedding"]
            future_change_embedding = future_module.encode_future_change(residual_normalized)
            inverse_prediction = future_module.inverse_prediction(future_change_embedding)
            inverse_loss, inverse_metrics = future_module.inverse_loss(
                inverse_prediction,
                semantic_targets,
                prefix="future_inverse",
            )
            # The same decoder must recover the action summary from both sides
            # of the shared semantic space.  This anchors the action branch and
            # removes the constant-vector solution that defeated cosine pairing.
            action_reconstruction_prediction = future_module.inverse_prediction(action_embedding)
            action_reconstruction_loss, action_reconstruction_metrics = future_module.inverse_loss(
                action_reconstruction_prediction,
                semantic_targets,
                prefix="future_action_reconstruction",
            )
            current_only_prediction = future_module.current_only_action_prediction(
                current_compressed, state_tokens.detach()
            )
            current_only_loss, current_only_metrics = future_module.inverse_loss(
                current_only_prediction,
                semantic_targets,
                prefix="future_current_only_inverse",
            )

            semantic_negatives, semantic_negative_names = future_module.semantic_negative_actions(
                action_gt.detach(),
                state=state_tokens.detach(),
                current_compressed=current_compressed.detach(),
                past_last_action=past_actions[:, -1].detach(),
            )
            batch_size, negative_count = semantic_negatives.shape[:2]
            negative_flat = semantic_negatives.reshape(
                batch_size * negative_count,
                self.config.prediction_horizon,
                self.config.action_dim,
            )
            state_negative = (
                state_tokens.detach()[:, None]
                .expand(-1, negative_count, -1)
                .reshape(batch_size * negative_count, self.config.state_dim)
            )
            past_negative = (
                past_actions[:, -1]
                .detach()[:, None]
                .expand(-1, negative_count, -1)
                .reshape(batch_size * negative_count, self.config.action_dim)
            )
            negative_embedding = future_module.encode_action_semantics(
                negative_flat,
                past_last_action=past_negative,
                state=state_negative,
            ).reshape(
                batch_size,
                negative_count,
                len(self.config.future_latent_offsets),
                self.config.future_action_semantic_dim,
            )
            negative_difference = (
                semantic_negatives.float() - action_gt.detach().float()[:, None]
            ).square()
            negative_valid = torch.stack(
                [
                    negative_difference[:, :, :offset].mean(dim=(2, 3))
                    > float(self.config.future_contrastive_duplicate_threshold)
                    for offset in self.config.future_latent_offsets
                ],
                dim=-1,
            )
            # Exact/near-identical action chunks are not valid batch
            # negatives.  Masking them avoids punishing semantically equivalent
            # demonstrations while preserving all genuinely different pairs.
            action_pair_difference = (
                action_gt.detach().float()[:, None] - action_gt.detach().float()[None, :]
            ).square()
            duplicate_mask = torch.stack(
                [
                    action_pair_difference[:, :, :offset].mean(dim=(2, 3))
                    <= float(self.config.future_contrastive_duplicate_threshold)
                    for offset in self.config.future_latent_offsets
                ],
                dim=0,
            )
            transition_weight = (
                1.0
                + float(self.config.future_contrastive_transition_boost)
                * semantic_targets["transition"].float()
            )
            align_terms = future_module.contrastive_alignment_terms(
                action_embedding,
                future_change_embedding,
                negative_embedding,
                temperature=float(self.config.future_contrastive_temperature),
                structured_negative_weight=float(self.config.future_structured_nce_weight),
                negative_valid=negative_valid,
                duplicate_mask=duplicate_mask,
                sample_weight=transition_weight,
            )
            align_loss = align_terms["loss"]
            embedding_regularization = future_module.embedding_regularization_terms(
                action_embedding,
                future_change_embedding,
                std_target=float(self.config.future_embedding_std_target),
            )

            # Predicted future residuals are judged by a detached semantic
            # evaluator.  Gradients therefore update only the world prediction
            # and its action-conditioning path, never the semantic anchor.
            remaining = 1.0 - future_flow.time.reshape(-1, 1, 1, 1, 1)
            predicted_residual_normalized = future_flow.noisy + remaining * world_velocity
            predicted_future_embedding = future_module.detached_future_change_embedding(
                predicted_residual_normalized
            )
            predicted_align_terms = future_module.contrastive_alignment_terms(
                action_embedding.detach(),
                predicted_future_embedding,
                temperature=float(self.config.future_contrastive_temperature),
                duplicate_mask=duplicate_mask,
                sample_weight=transition_weight,
            )
            predicted_align_loss = predicted_align_terms["loss"]
            predicted_inverse = future_module.detached_inverse_prediction(
                predicted_future_embedding
            )
            cycle_loss, cycle_metrics = future_module.inverse_loss(
                predicted_inverse,
                semantic_targets,
                prefix="future_pred_action_cycle",
            )
            future_flow_train_objective = future_metrics["future_latent_flow_mse"]
            future_align_train_objective = align_loss
            future_inverse_train_objective = inverse_loss
            future_pred_align_train_objective = predicted_align_loss
            future_cycle_train_objective = cycle_loss
            semantic_objective = (
                self.config.future_align_loss_weight * align_loss
                + self.config.future_inverse_loss_weight * inverse_loss
                + self.config.future_current_action_baseline_loss_weight * current_only_loss
                + self.config.future_action_reconstruction_loss_weight * action_reconstruction_loss
                + self.config.future_embedding_variance_loss_weight
                * embedding_regularization["variance_loss"]
                + self.config.future_embedding_covariance_loss_weight
                * embedding_regularization["covariance_loss"]
                + float(future_semantic_scale)
                * (
                    self.config.future_pred_align_loss_weight * predicted_align_loss
                    + self.config.future_cycle_loss_weight * cycle_loss
                )
            )
            semantic_confidence_raw = (
                align_terms["sample_margin"].detach()
                / max(float(self.config.future_semantic_confidence_margin), 1e-6)
            ).clamp(0.0, 1.0)
            confidence_floor = float(self.config.future_consistency_confidence_floor)
            semantic_confidence = (
                confidence_floor + (1.0 - confidence_floor) * semantic_confidence_raw
            )
            future_metrics.update(inverse_metrics)
            future_metrics.update(action_reconstruction_metrics)
            future_metrics.update(current_only_metrics)
            future_metrics.update(cycle_metrics)
            future_metrics.update(
                {
                    "future_semantic_objective": semantic_objective,
                    "future_semantic_scale": policy_total.new_tensor(float(future_semantic_scale)),
                    "future_align_loss": align_loss,
                    "future_align_symmetric_nce_loss": align_terms["symmetric_nce_loss"],
                    "future_align_action_to_future_nce_loss": align_terms[
                        "action_to_future_nce_loss"
                    ],
                    "future_align_future_to_action_nce_loss": align_terms[
                        "future_to_action_nce_loss"
                    ],
                    "future_align_structured_nce_loss": align_terms["structured_nce_loss"],
                    "future_align_positive_cosine": align_terms["positive_cosine"],
                    "future_align_negative_cosine": align_terms["negative_cosine"],
                    "future_align_batch_negative_cosine": align_terms["batch_negative_cosine"],
                    "future_align_structured_negative_cosine": align_terms[
                        "structured_negative_cosine"
                    ],
                    "future_align_margin": align_terms["margin"],
                    "future_align_action_to_future_top1": align_terms["action_to_future_top1"],
                    "future_align_future_to_action_top1": align_terms["future_to_action_top1"],
                    "future_align_structured_valid_fraction": align_terms[
                        "structured_valid_fraction"
                    ],
                    "future_align_hardest_negative_index": align_terms["hardest_negative_index"],
                    "future_embedding_variance_loss": embedding_regularization["variance_loss"],
                    "future_embedding_covariance_loss": embedding_regularization["covariance_loss"],
                    "future_action_embedding_std": embedding_regularization["action_std"],
                    "future_change_embedding_std": embedding_regularization["future_std"],
                    "future_pred_align_loss": predicted_align_loss,
                    "future_pred_align_action_to_future_top1": predicted_align_terms[
                        "action_to_future_top1"
                    ],
                    "future_pred_align_future_to_action_top1": predicted_align_terms[
                        "future_to_action_top1"
                    ],
                    "future_change_embedding_rms": future_change_embedding.square().mean().sqrt(),
                    "future_pred_change_embedding_rms": predicted_future_embedding.square()
                    .mean()
                    .sqrt(),
                    "future_semantic_confidence_mean": semantic_confidence.mean(),
                    "future_inverse_future_gain": current_only_loss.detach()
                    - inverse_loss.detach(),
                    "future_align_pair_loss": policy_total.new_zeros(()),
                    "future_align_rank_loss": policy_total.new_zeros(()),
                    "future_align_rank_active_fraction": policy_total.new_zeros(()),
                }
            )
            all_negative_cosine = align_terms["all_negative_cosine"]
            for negative_idx, negative_name in enumerate(semantic_negative_names):
                future_metrics[f"future_align_negative_{negative_name}_cosine"] = (
                    all_negative_cosine[:, negative_idx].mean()
                )

            # Retain the old flow-error ranking only as a low-weight auxiliary.
            corrupted_action, corruption_mode = future_module.corrupt_actions(
                action_gt.detach(),
                state=state_tokens.detach(),
                current_compressed=current_compressed.detach(),
            )
            corrupted_velocity, _ = future_module(
                current_compressed=current_compressed,
                action_chunk=corrupted_action,
                state=state_tokens.detach(),
                past_last_action=past_actions[:, -1].detach(),
                future_noisy=future_flow.noisy,
                future_time=future_flow.time,
            )
            correct_per_sample = future_module.weighted_mse(
                world_velocity, future_flow.target_velocity, motion_weights, reduction="none"
            )
            corrupted_per_sample = future_module.weighted_mse(
                corrupted_velocity, future_flow.target_velocity, motion_weights, reduction="none"
            )
            zero_velocity_per_sample = future_module.weighted_mse(
                torch.zeros_like(world_velocity),
                future_flow.target_velocity,
                motion_weights,
                reduction="none",
            )
            confidence_terms = _relative_world_confidence(
                demo_error=correct_per_sample,
                corrupted_error=corrupted_per_sample,
                zero_error=zero_velocity_per_sample,
                dependency_margin=max(float(self.config.future_dependency_relative_margin), 1e-6),
                world_skill_margin=max(
                    float(self.config.future_consistency_world_skill_margin), 1e-6
                ),
                confidence_floor=float(self.config.future_consistency_confidence_floor),
            )
            dependency_gap = confidence_terms["dependency_gap"]
            dependency_relative_gap = confidence_terms["dependency_relative_gap"]
            raw_action_time_weights = (
                (1.0 - future_flow.time.float())
                .pow(float(self.config.future_action_time_power))
                .clamp_min(float(self.config.future_action_time_floor))
            )
            action_time_weights = (
                raw_action_time_weights / raw_action_time_weights.mean().clamp_min(1e-6)
            )
            dependency_loss = (
                F.relu(
                    float(self.config.future_dependency_relative_margin) - dependency_relative_gap
                )
                * action_time_weights.to(dtype=dependency_relative_gap.dtype)
            ).mean()
            action_delta_rms = (
                (action_gt.detach().float() - corrupted_action.float())
                .square()
                .mean(dim=(1, 2))
                .sqrt()
            )
            future_delta_rms = (
                (world_velocity.float() - corrupted_velocity.float())
                .square()
                .mean(dim=(1, 2, 3, 4))
                .sqrt()
            )
            action_jacobian_proxy = (future_delta_rms / action_delta_rms.clamp_min(1e-6)).mean()
            future_world_objective = (
                future_metrics["future_latent_flow_mse"]
                + self.config.future_endpoint_loss_weight
                * future_metrics["future_latent_absolute_endpoint_rmse"].square()
                + self.config.future_dependency_loss_weight * dependency_loss
                + semantic_objective
            )
            future_metrics.update(
                {
                    "future_action_dependency_loss": dependency_loss,
                    "future_action_dependency_gap": dependency_gap.mean(),
                    "future_action_dependency_relative_gap": dependency_relative_gap.mean(),
                    "future_action_corrupted_flow_mse": corrupted_per_sample.mean(),
                    "future_world_demo_flow_mse": correct_per_sample.mean(),
                    "future_world_zero_velocity_flow_mse": zero_velocity_per_sample.mean(),
                    "future_world_relative_skill": confidence_terms["world_relative_skill"].mean(),
                    "future_action_corruption_mode": corruption_mode,
                    "future_action_time_weight_raw_mean": raw_action_time_weights.mean(),
                    "future_action_time_weight_raw_max": raw_action_time_weights.max(),
                    # Finite-difference/secant proxy; logged as a practical Jacobian diagnostic.
                    "future_action_jacobian_rms": action_jacobian_proxy,
                }
            )

            # Closed-loop transfer is allowed only when flow skill, old
            # dependency, and the new action/future semantic margin agree.
            closed_loop_active = future_consistency_scale > 0.0 or not self.training
            if self.config.future_latent_variant == "closed-loop" and closed_loop_active:
                predicted_action_full = self._replace_gripper(
                    predicted_action_endpoint,
                    output.full_gripper_openness,
                    length=self.config.prediction_horizon,
                )
                policy_velocity, _ = future_module.detached_parameter_forward(
                    current_compressed=current_compressed.detach(),
                    action_chunk=predicted_action_full,
                    state=state_tokens.detach(),
                    past_last_action=past_actions[:, -1].detach(),
                    future_noisy=future_flow.noisy.detach(),
                    future_time=future_flow.time.detach(),
                )
                policy_error_per_sample = future_module.weighted_mse(
                    policy_velocity,
                    future_flow.target_velocity.detach(),
                    motion_weights.detach(),
                    reduction="none",
                )
                demo_error_per_sample = correct_per_sample.detach()
                zero_error_per_sample = zero_velocity_per_sample.detach()
                teacher_per_sample = future_module.weighted_mse(
                    policy_velocity,
                    world_velocity.detach(),
                    motion_weights.detach(),
                    reduction="none",
                )
                consequence_terms = _conservative_relative_consequence_terms(
                    policy_error=policy_error_per_sample,
                    demo_error=demo_error_per_sample,
                    zero_error=zero_error_per_sample,
                    teacher_error=teacher_per_sample,
                    relative_margin=float(self.config.future_consistency_relative_margin),
                    regret_cap=float(self.config.future_consistency_regret_cap),
                    teacher_weight=float(self.config.future_consistency_teacher_weight),
                    teacher_cap=float(self.config.future_consistency_teacher_cap),
                )
                policy_minus_demo = consequence_terms["policy_minus_demo"]
                relative_regret = consequence_terms["relative_regret"]
                relative_hinge = consequence_terms["relative_hinge"]
                teacher_relative = consequence_terms["teacher_relative"]
                conservative_per_sample = consequence_terms["conservative"]
                policy_bridge_raw = (
                    (1.0 - bridge.time.float())
                    .pow(float(self.config.future_policy_bridge_time_power))
                    .clamp_min(float(self.config.future_policy_bridge_time_floor))
                )
                policy_bridge_weights = policy_bridge_raw / policy_bridge_raw.mean().clamp_min(1e-6)
                dependency_confidence = confidence_terms["dependency_confidence"]
                world_relative_skill = confidence_terms["world_relative_skill"]
                world_skill_confidence = confidence_terms["world_skill_confidence"]
                flow_joint_confidence = confidence_terms["joint_confidence"]
                joint_confidence = flow_joint_confidence * semantic_confidence
                consequence_weight = (
                    action_time_weights * policy_bridge_weights * joint_confidence
                ).clamp_max(float(self.config.future_consistency_weight_cap))
                consequence_weight = consequence_weight.to(dtype=conservative_per_sample.dtype)
                future_consistency_objective = (conservative_per_sample * consequence_weight).mean()
                future_metrics.update(
                    {
                        "future_policy_flow_mse": (
                            policy_error_per_sample * consequence_weight
                        ).mean(),
                        "future_policy_demo_flow_mse": (
                            demo_error_per_sample * consequence_weight
                        ).mean(),
                        "future_policy_minus_demo_flow_gap": (
                            policy_minus_demo * consequence_weight
                        ).mean(),
                        "future_policy_relative_regret": (
                            relative_regret * consequence_weight
                        ).mean(),
                        "future_policy_relative_hinge": (
                            relative_hinge * consequence_weight
                        ).mean(),
                        "future_policy_teacher_consistency_mse": (
                            teacher_per_sample * consequence_weight
                        ).mean(),
                        "future_policy_teacher_consistency_relative": (
                            teacher_relative * consequence_weight
                        ).mean(),
                        "future_policy_velocity_rms": policy_velocity.square().mean().sqrt(),
                        "future_policy_bridge_time_weight_raw_mean": policy_bridge_raw.mean(),
                        "future_policy_bridge_time_weight_raw_max": policy_bridge_raw.max(),
                        "future_policy_dependency_confidence_mean": dependency_confidence.mean(),
                        "future_policy_world_skill_confidence_mean": world_skill_confidence.mean(),
                        "future_policy_semantic_confidence_mean": semantic_confidence.mean(),
                        "future_policy_flow_joint_confidence_mean": flow_joint_confidence.mean(),
                        "future_policy_joint_confidence_mean": joint_confidence.mean(),
                        "future_policy_dependency_positive_fraction": (
                            dependency_relative_gap.detach() > 0
                        )
                        .float()
                        .mean(),
                        "future_policy_world_skill_positive_fraction": (world_relative_skill > 0)
                        .float()
                        .mean(),
                        "future_policy_relative_hinge_active_fraction": (
                            relative_regret.detach()
                            > float(self.config.future_consistency_relative_margin)
                        )
                        .float()
                        .mean(),
                        "future_policy_consequence_weight_mean": consequence_weight.mean(),
                        "future_policy_consequence_weight_max": consequence_weight.max(),
                    }
                )

        future_objective = (
            self.config.future_world_loss_weight * future_world_objective
            + self.config.future_consistency_loss_weight
            * float(future_consistency_scale)
            * future_consistency_objective
        )
        total = policy_total + future_objective
        return {
            "loss": total,
            "policy_objective": policy_total,
            "future_latent_objective": future_objective,
            "future_latent_weighted_objective": future_objective.detach(),
            "future_world_objective": future_world_objective,
            "future_world_weighted_objective": (
                self.config.future_world_loss_weight * future_world_objective
            ).detach(),
            "future_policy_consistency_objective": future_consistency_objective,
            "future_policy_consistency_weighted_objective": (
                self.config.future_consistency_loss_weight
                * float(future_consistency_scale)
                * future_consistency_objective
            ).detach(),
            "future_policy_consistency_scale": policy_total.new_tensor(
                float(future_consistency_scale)
            ),
            "future_semantic_scale": policy_total.new_tensor(float(future_semantic_scale)),
            "future_flow_train_objective": future_flow_train_objective,
            "future_align_train_objective": future_align_train_objective,
            "future_inverse_train_objective": future_inverse_train_objective,
            "future_pred_align_train_objective": future_pred_align_train_objective,
            "future_cycle_train_objective": future_cycle_train_objective,
            "prior_arm_mse": prior_arm_mse.detach(),
            "full_arm_flow_mse": full_arm_mse.detach(),
            "fast_first_arm_flow_mse": first_arm_mse.detach(),
            "prefix_arm_flow_mse": prefix_arm_mse.detach(),
            "arm_delta_loss": arm_delta_loss.detach(),
            "align_phase_arm_loss": align_phase_loss.detach(),
            "align_phase_fraction": close_align_mask.float().mean().detach(),
            "full_gripper_state_loss": full_grip["state_smooth_l1"].detach(),
            "gripper_transition_loss": full_grip["transition_smooth_l1"].detach(),
            "gripper_delta_loss": full_grip["delta_smooth_l1"].detach(),
            "gripper_transition_fraction": full_grip["transition_fraction"].detach(),
            "gripper_openness_mean": full_grip["openness_mean"].detach(),
            "gripper_target_clip_fraction": full_grip["target_clip_fraction"].detach(),
            "fast_visual_first_latent_rms": output.fast_visual_first_latent_rms.detach(),
            "fast_visual_prefix_latent_rms": output.fast_visual_prefix_latent_rms.detach(),
            "fast_visual_first_alpha_mean": output.fast_visual_first_alpha_mean.detach(),
            "fast_visual_prefix_alpha_mean": output.fast_visual_prefix_alpha_mean.detach(),
            "fast_visual_top_gate_mean": output.fast_visual_top_gate_mean.detach(),
            "fast_visual_wrist_gate_mean": output.fast_visual_wrist_gate_mean.detach(),
            "fast_visual_to_hidden_ratio": output.fast_visual_to_hidden_ratio.detach(),
            "target_arm_residual_rms": torch.cat(
                [
                    bridge.target_residual[..., :grip_idx],
                    bridge.target_residual[..., grip_idx + 1 :],
                ],
                dim=-1,
            )
            .detach()
            .square()
            .mean()
            .sqrt(),
            "source_arm_residual_rms": torch.cat(
                [
                    bridge.source_residual[..., :grip_idx],
                    bridge.source_residual[..., grip_idx + 1 :],
                ],
                dim=-1,
            )
            .detach()
            .square()
            .mean()
            .sqrt(),
            **{key: value.detach() for key, value in future_metrics.items()},
        }

    @torch.no_grad()
    def _integrate(
        self,
        *,
        state_tokens: Tensor,
        past_actions: Tensor,
        physical_prior: Tensor,
        dense_tokens: Tensor | None,
        kv_cache: list[tuple[Tensor, Tensor]] | None,
        attention_mask: Tensor | None,
        steps: int,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        learned_prior, history_context = self.predict_prior(
            state_tokens=state_tokens, past_actions=past_actions, physical_prior=physical_prior
        )
        residual = torch.zeros_like(learned_prior)
        dt = 1.0 / steps
        time = torch.zeros(
            (state_tokens.shape[0],), device=state_tokens.device, dtype=state_tokens.dtype
        )
        if mode == "fast":
            length = 1
        elif mode == "prefix":
            length = self.config.prefix_length
        elif mode == "full":
            length = self.config.prediction_horizon
        else:
            raise ValueError(f"unknown integration mode={mode!r}")
        output = None
        for _ in range(steps):
            output, _ = self._velocity(
                state_tokens=state_tokens,
                past_actions=past_actions,
                physical_prior=physical_prior,
                residual_state=residual,
                timesteps=time,
                dense_tokens=dense_tokens,
                kv_cache=kv_cache,
                attention_mask=attention_mask,
                stop_after=mode,
                learned_prior=learned_prior,
                history_context=history_context,
            )
            velocity = (
                output.fast_first
                if mode == "fast"
                else output.prefix
                if mode == "prefix"
                else output.full
            )
            if velocity is None:
                raise AssertionError("requested progressive velocity missing")
            residual[:, :length] = residual[:, :length] + velocity * dt
            time = time + dt
        if output is None:
            raise AssertionError("integration output missing")
        openness = (
            output.fast_gripper_openness
            if mode == "fast"
            else output.prefix_gripper_openness
            if mode == "prefix"
            else output.full_gripper_openness
        )
        if openness is None:
            raise AssertionError("requested bounded gripper output missing")
        chunk = self._replace_gripper(learned_prior + residual, openness, length=length)
        return chunk, learned_prior

    @torch.no_grad()
    def predict_first_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="fast", **kwargs)
        return chunk[:, 0]

    @torch.no_grad()
    def predict_prefix_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="prefix", **kwargs)
        return chunk[:, : self.config.prefix_length]

    @torch.no_grad()
    def predict_action(self, **kwargs: Any) -> Tensor:
        chunk, _ = self._integrate(mode="full", **kwargs)
        return chunk

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def _resolve_state_dict(source: str | Path | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(source, (str, Path)):
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            payload = source
        if (
            isinstance(payload, dict)
            and "module" in payload
            and isinstance(payload["module"], dict)
        ):
            payload = payload["module"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError("checkpoint must resolve to a state_dict")
        return {str(key).removeprefix("module."): value for key, value in payload.items()}

    def load_compatible_reference_state_dict(
        self, source: str | Path | dict[str, Tensor]
    ) -> dict[str, Any]:
        """Reuse only tensors whose names and shapes remain meaningful.

        Attention, FFN, timestep, positional, adaptor and final-head tensors can
        transfer.  Per-block AdaLN tensors deliberately do not transfer because
        the progressive model replaces them with stage-shared low-rank banks.
        """
        source_state = self._resolve_state_dict(source)
        target_state = self.state_dict()
        matched: dict[str, Tensor] = {}
        skipped_shape: dict[str, dict[str, list[int]]] = {}
        unexpected: list[str] = []
        for key, value in source_state.items():
            if key not in target_state:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped_shape[key] = {
                    "source": list(value.shape),
                    "target": list(target_state[key].shape),
                }
                continue
            matched[key] = value
        self.load_state_dict(matched, strict=False)
        return {
            "matched_tensors": len(matched),
            "source_tensors": len(source_state),
            "target_tensors": len(target_state),
            "missing_target_keys": sorted(set(target_state) - set(matched)),
            "unexpected_source_keys": sorted(unexpected),
            "shape_mismatches": skipped_shape,
        }


__all__ = [
    "MainlineRDT2FM",
    "MainlineRDT2FMConfig",
    "HistoryTrajectoryPrior",
    "SplitActionProjection",
    "ArmOnlyProjection",
    "BoundedContinuousGripperHead",
    "QueryLatentVisualCorrector",
    "VisualLatentCorrection",
    "FutureLatentDynamicsHead",
    "_prefix_weights",
    "_chunk_execution_weights",
    "_horizon_weights",
    "_gripper_transition_mask",
    "_arm_weighted_mse",
    "_arm_delta_matching_loss",
    "_close_pre_mask",
    "_masked_arm_endpoint_loss",
    "_bounded_gripper_loss",
]
