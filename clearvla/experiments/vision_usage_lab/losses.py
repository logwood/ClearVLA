from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import VisionUsageLabOutput


@dataclass(frozen=True)
class VisionUsageLabLossConfig:
    flow_weight: float = 1.0
    endpoint_weight: float = 1.0
    first_weight: float = 1.0
    first4_weight: float = 0.5
    velocity_weight: float = 0.25
    source_weight: float = 0.50
    prefix_weight: float = 0.50
    prefix_teacher_weight: float = 0.25
    streaming_weight: float = 0.50
    streaming_teacher_forced_weight: float = 0.25
    streaming_teacher_weight: float = 0.25
    consistency_weight: float = 0.10
    dynamics_weight: float = 1.0
    dynamics_cosine_weight: float = 0.25
    ranking_weight: float = 0.20
    ranking_margin: float = 0.01
    ranking_demand_boost: float = 1.0
    event_weight: float = 0.10
    demand_weight: float = 0.25
    demand_huber_beta: float = 0.10
    huber_beta: float = 0.03

    def validate(self) -> None:
        weights = (
            self.flow_weight,
            self.endpoint_weight,
            self.first_weight,
            self.first4_weight,
            self.velocity_weight,
            self.source_weight,
            self.prefix_weight,
            self.prefix_teacher_weight,
            self.streaming_weight,
            self.streaming_teacher_forced_weight,
            self.streaming_teacher_weight,
            self.consistency_weight,
            self.dynamics_weight,
            self.dynamics_cosine_weight,
            self.ranking_weight,
            self.ranking_demand_boost,
            self.event_weight,
            self.demand_weight,
        )
        if any(float(value) < 0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative")
        if self.huber_beta <= 0 or self.demand_huber_beta <= 0:
            raise ValueError("Huber beta values must be positive")


@dataclass
class VisionUsageLabLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def detached_floats(self) -> dict[str, float]:
        return {key: float(value.detach().cpu()) for key, value in self.components.items()}


def _smooth_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}")
    value = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1 or weight.ndim != 1 or value.shape != weight.shape:
        raise ValueError("weighted vectors must be aligned [B]")
    return (value * weight).sum() / weight.sum().clamp_min(1e-12)


def action_composite_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: VisionUsageLabLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred and target action chunks must share [B,K,A]")
    full = _smooth_l1_per_sample(pred, target, config.huber_beta)
    first = _smooth_l1_per_sample(pred[:, :1], target[:, :1], config.huber_beta)
    first4 = _smooth_l1_per_sample(
        pred[:, : min(4, pred.shape[1])], target[:, : min(4, target.shape[1])], config.huber_beta
    )
    if pred.shape[1] >= 2:
        velocity = _smooth_l1_per_sample(
            pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], config.huber_beta
        )
    else:
        velocity = torch.zeros_like(full)
    total = (
        config.endpoint_weight * full
        + config.first_weight * first
        + config.first4_weight * first4
        + config.velocity_weight * velocity
    )
    return total, {
        "endpoint_full": full,
        "endpoint_first": first,
        "endpoint_first4": first4,
        "endpoint_velocity": velocity,
    }


def _visual_dynamics_losses(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    config: VisionUsageLabLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_delta.shape != target_delta.shape or pred_delta.ndim != 5:
        raise ValueError(
            f"future visual deltas must share [B,F,V,P,C], got {tuple(pred_delta.shape)} vs {tuple(target_delta.shape)}"
        )
    huber = F.smooth_l1_loss(pred_delta, target_delta, beta=config.huber_beta)
    pred_flat = pred_delta.reshape(-1, pred_delta.shape[-1])
    target_flat = target_delta.reshape(-1, target_delta.shape[-1])
    cosine = 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-6).mean()
    return huber, cosine


def _correction_demand_loss(
    output: VisionUsageLabOutput,
    demand_target: torch.Tensor | None,
    config: VisionUsageLabLossConfig,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if demand_target is None or config.demand_weight <= 0:
        return fallback
    if output.demand_score is None:
        raise ValueError("demand supervision requires demand_score")
    if demand_target.ndim != 1 or demand_target.shape != output.demand_score.shape:
        raise ValueError(
            f"demand_target shape={tuple(demand_target.shape)} != score={tuple(output.demand_score.shape)}"
        )
    return F.smooth_l1_loss(
        output.demand_score,
        demand_target.to(dtype=output.demand_score.dtype),
        beta=config.demand_huber_beta,
    )


def vision_usage_lab_loss(
    *,
    correct: VisionUsageLabOutput,
    target_actions: torch.Tensor,
    target_velocity: torch.Tensor | None,
    target_visual_delta_tokens: torch.Tensor,
    config: VisionUsageLabLossConfig = VisionUsageLabLossConfig(),
    wrong: VisionUsageLabOutput | None = None,
    consistency_output: VisionUsageLabOutput | None = None,
    event_flag: torch.Tensor | None = None,
    demand_target: torch.Tensor | None = None,
    phase: str = "action_flow",
) -> VisionUsageLabLossResult:
    """Losses for learned source, fast executable path and chunk teacher."""
    config.validate()
    if phase not in {"representation_pretrain", "action_flow"}:
        raise ValueError(f"unsupported lab phase={phase!r}")
    if correct.visual_delta_tokens is None:
        raise ValueError("correct output must include visual_delta_tokens")
    dyn_huber, dyn_cosine = _visual_dynamics_losses(
        correct.visual_delta_tokens, target_visual_delta_tokens, config
    )
    zero = torch.zeros((), device=dyn_huber.device, dtype=dyn_huber.dtype)
    source = _smooth_l1_per_sample(correct.learned_source, target_actions, config.huber_beta).mean()
    if correct.fast_prefix is None:
        raise ValueError("correct output must include fast_prefix")
    prefix_target = target_actions[:, : correct.fast_prefix.shape[1]]
    prefix = _smooth_l1_per_sample(correct.fast_prefix, prefix_target, config.huber_beta).mean()
    streaming = zero
    if correct.streaming_actions is not None:
        streaming = _smooth_l1_per_sample(
            correct.streaming_actions, target_actions, config.huber_beta
        ).mean()
    streaming_teacher_forced = zero
    if correct.streaming_teacher_forced_actions is not None:
        streaming_teacher_forced = _smooth_l1_per_sample(
            correct.streaming_teacher_forced_actions,
            target_actions,
            config.huber_beta,
        ).mean()
    streaming_teacher = zero
    flow = zero
    consistency = zero
    action = zero
    prefix_teacher = zero
    ranking = zero
    wrong_action = zero
    ranking_weight_mean = zero
    action_parts: dict[str, torch.Tensor] = {
        "endpoint_full": zero,
        "endpoint_first": zero,
        "endpoint_first4": zero,
        "endpoint_velocity": zero,
    }
    event = zero
    if event_flag is not None and config.event_weight > 0:
        if event_flag.ndim != 1 or event_flag.shape != correct.event_logit.shape:
            raise ValueError(
                f"event_flag shape={tuple(event_flag.shape)} != logit={tuple(correct.event_logit.shape)}"
            )
        event = F.binary_cross_entropy_with_logits(
            correct.event_logit, event_flag.to(dtype=correct.event_logit.dtype)
        )
    demand = _correction_demand_loss(correct, demand_target, config, fallback=zero)

    if phase == "representation_pretrain":
        total = (
            config.source_weight * source
            + config.prefix_weight * prefix
            + config.streaming_weight * streaming
            + config.streaming_teacher_forced_weight * streaming_teacher_forced
            + config.dynamics_weight * dyn_huber
            + config.dynamics_cosine_weight * dyn_cosine
            + config.event_weight * event
            + config.demand_weight * demand
        )
    else:
        if correct.velocity is None or correct.endpoint is None or target_velocity is None:
            raise ValueError("action_flow requires velocity, endpoint, and target_velocity")
        flow = _smooth_l1_per_sample(correct.velocity, target_velocity, config.huber_beta).mean()
        if consistency_output is not None:
            if consistency_output.velocity is None:
                raise ValueError("consistency output must include velocity")
            consistency = _smooth_l1_per_sample(
                consistency_output.velocity, correct.velocity.detach(), config.huber_beta
            ).mean()
        action_per_sample, action_parts = action_composite_per_sample(
            correct.endpoint, target_actions, config
        )
        action = action_per_sample.mean()
        teacher_endpoint = correct.endpoint.detach()
        teacher_prefix = teacher_endpoint[:, : correct.fast_prefix.shape[1]]
        prefix_teacher = _smooth_l1_per_sample(
            correct.fast_prefix, teacher_prefix, config.huber_beta
        ).mean()
        if correct.streaming_actions is not None:
            streaming_teacher = _smooth_l1_per_sample(
                correct.streaming_actions, teacher_endpoint, config.huber_beta
            ).mean()
        if wrong is not None and config.ranking_weight > 0:
            if wrong.endpoint is None:
                raise ValueError("counterfactual output must include endpoint")
            wrong_per_sample, _ = action_composite_per_sample(
                wrong.endpoint, target_actions, config
            )
            wrong_action = wrong_per_sample.mean()
            rank_per_sample = F.relu(config.ranking_margin + action_per_sample - wrong_per_sample)
            if demand_target is None:
                rank_weights = torch.ones_like(rank_per_sample)
            else:
                if demand_target.ndim != 1 or demand_target.shape != rank_per_sample.shape:
                    raise ValueError("demand_target must align with action batch for ranking")
                rank_weights = 1.0 + config.ranking_demand_boost * demand_target.to(
                    dtype=rank_per_sample.dtype
                )
            ranking_weight_mean = rank_weights.mean()
            ranking = _weighted_mean(rank_per_sample, rank_weights)
        total = (
            config.flow_weight * flow
            + config.consistency_weight * consistency
            + action
            + config.source_weight * source
            + config.prefix_weight * prefix
            + config.prefix_teacher_weight * prefix_teacher
            + config.streaming_weight * streaming
            + config.streaming_teacher_forced_weight * streaming_teacher_forced
            + config.streaming_teacher_weight * streaming_teacher
            + config.dynamics_weight * dyn_huber
            + config.dynamics_cosine_weight * dyn_cosine
            + config.ranking_weight * ranking
            + config.event_weight * event
            + config.demand_weight * demand
        )

    components = {
        "total": total,
        "flow": flow,
        "action": action,
        "consistency": consistency,
        "source": source,
        "prefix": prefix,
        "prefix_teacher": prefix_teacher,
        "streaming": streaming,
        "streaming_teacher_forced": streaming_teacher_forced,
        "streaming_teacher": streaming_teacher,
        "dynamics_huber": dyn_huber,
        "dynamics_cosine": dyn_cosine,
        "ranking": ranking,
        "ranking_weight_mean": ranking_weight_mean,
        "wrong_action": wrong_action,
        "event": event,
        "demand": demand,
        "endpoint_full": action_parts["endpoint_full"].mean(),
        "endpoint_first": action_parts["endpoint_first"].mean(),
        "endpoint_first4": action_parts["endpoint_first4"].mean(),
        "endpoint_velocity": action_parts["endpoint_velocity"].mean(),
    }
    return VisionUsageLabLossResult(total=total, components=components)
