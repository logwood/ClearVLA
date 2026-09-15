from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import ResidualFlowLabOutput


@dataclass(frozen=True)
class ResidualFlowLossConfig:
    """Narrow direct-action objective for the residual-flow core.

    Defaults deliberately avoid auxiliary targets.  The flow target and final
    action endpoint are sufficient to test whether visual conditioning improves
    a frozen history source.  Prefix, streaming, router and patch reconstruction
    objectives belong to later deployment stages, not to this core experiment.
    """

    flow_weight: float = 1.0
    endpoint_weight: float = 1.0
    first_weight: float = 0.0
    first4_weight: float = 0.0
    velocity_weight: float = 0.0
    ranking_weight: float = 0.0
    ranking_margin: float = 0.01
    huber_beta: float = 0.03

    def validate(self) -> None:
        weights = (
            self.flow_weight,
            self.endpoint_weight,
            self.first_weight,
            self.first4_weight,
            self.velocity_weight,
            self.ranking_weight,
        )
        if any(float(value) < 0 for value in weights):
            raise ValueError("loss weights must be non-negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative")
        if self.huber_beta <= 0:
            raise ValueError("huber_beta must be positive")


@dataclass
class ResidualFlowLossResult:
    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def detached_floats(self) -> dict[str, float]:
        return {key: float(value.detach().cpu()) for key, value in self.components.items()}


def _smooth_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}")
    value = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return value.reshape(value.shape[0], -1).mean(dim=1)


def source_pretrain_loss(
    learned_source: torch.Tensor, target_actions: torch.Tensor, *, huber_beta: float = 0.03
) -> torch.Tensor:
    """History-only source objective used before residual-flow training."""
    return _smooth_l1_per_sample(learned_source, target_actions, huber_beta).mean()


def action_composite_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: ResidualFlowLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred and target must share [B,K,A]")
    full = _smooth_l1_per_sample(pred, target, config.huber_beta)
    first = _smooth_l1_per_sample(pred[:, :1], target[:, :1], config.huber_beta)
    count = min(4, pred.shape[1])
    first4 = _smooth_l1_per_sample(pred[:, :count], target[:, :count], config.huber_beta)
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


def residual_flow_loss(
    *,
    correct: ResidualFlowLabOutput,
    target_actions: torch.Tensor,
    target_velocity: torch.Tensor,
    config: ResidualFlowLossConfig = ResidualFlowLossConfig(),
    wrong: ResidualFlowLabOutput | None = None,
) -> ResidualFlowLossResult:
    config.validate()
    flow = _smooth_l1_per_sample(
        correct.residual_velocity, target_velocity, config.huber_beta
    ).mean()
    action_per_sample, parts = action_composite_per_sample(
        correct.endpoint_actions, target_actions, config
    )
    action = action_per_sample.mean()
    zero = torch.zeros((), device=flow.device, dtype=flow.dtype)
    wrong_action = zero
    ranking = zero
    if wrong is not None and config.ranking_weight > 0:
        wrong_per_sample, _ = action_composite_per_sample(
            wrong.endpoint_actions, target_actions, config
        )
        wrong_action = wrong_per_sample.mean()
        ranking = F.relu(config.ranking_margin + action_per_sample - wrong_per_sample).mean()
    total = config.flow_weight * flow + action + config.ranking_weight * ranking
    return ResidualFlowLossResult(
        total=total,
        components={
            "total": total,
            "flow": flow,
            "action": action,
            "ranking": ranking,
            "wrong_action": wrong_action,
            "endpoint_full": parts["endpoint_full"].mean(),
            "endpoint_first": parts["endpoint_first"].mean(),
            "endpoint_first4": parts["endpoint_first4"].mean(),
            "endpoint_velocity": parts["endpoint_velocity"].mean(),
        },
    )
