from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ObjectiveName, RDTLiteModel, RDTLiteOutput
from .schedule import CosineDiffusionSchedule, DiffusionScheduleConfig, pi_flow_bridge, sample_pi_time


@dataclass(frozen=True)
class RDTLiteLossConfig:
    objective: ObjectiveName = "rdt_denoise"
    first_weight: float = 0.0
    first4_weight: float = 0.0
    gripper_weight: float = 1.0
    pi_endpoint_weight: float = 0.0
    pi_time_alpha: float = 1.5
    pi_time_beta: float = 1.0

    def validate(self) -> None:
        if self.objective not in ("rdt_denoise", "pi_flow"):
            raise ValueError(f"unsupported objective={self.objective!r}")
        if min(self.first_weight, self.first4_weight, self.gripper_weight, self.pi_endpoint_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.pi_time_alpha <= 0 or self.pi_time_beta <= 0:
            raise ValueError("pi time beta distribution parameters must be positive")


@dataclass(frozen=True)
class RDTLiteLossResult:
    total: torch.Tensor
    prediction: torch.Tensor
    endpoint: torch.Tensor
    noisy_actions: torch.Tensor
    time: torch.Tensor
    components: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, *, gripper_weight: float) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes")
    error = (pred - target).square()
    if error.shape[-1] > 1 and gripper_weight != 1.0:
        weights = torch.ones((error.shape[-1],), device=error.device, dtype=error.dtype)
        weights[-1] = float(gripper_weight)
        error = error * weights
        return error.sum(dim=-1).mean() / weights.sum().clamp_min(1e-12)
    return error.mean()


def _endpoint_terms(endpoint: torch.Tensor, target: torch.Tensor, *, config: RDTLiteLossConfig) -> dict[str, torch.Tensor]:
    first4 = min(4, endpoint.shape[1])
    full = _weighted_mse(endpoint, target, gripper_weight=config.gripper_weight)
    first = _weighted_mse(endpoint[:, :1], target[:, :1], gripper_weight=config.gripper_weight)
    first4_loss = _weighted_mse(endpoint[:, :first4], target[:, :first4], gripper_weight=config.gripper_weight)
    return {"endpoint_full": full, "endpoint_first": first, "endpoint_first4": first4_loss}


def compute_rdt_lite_loss(
    model: RDTLiteModel,
    *,
    state_history: torch.Tensor,
    target_actions: torch.Tensor,
    visual_tokens: torch.Tensor,
    config: RDTLiteLossConfig = RDTLiteLossConfig(),
    diffusion_schedule: CosineDiffusionSchedule | None = None,
) -> RDTLiteLossResult:
    """Narrow direct-action objective in the ActionCodec target space."""

    config.validate()
    if target_actions.ndim != 3:
        raise ValueError("target_actions must be [B,H,A]")
    noise = torch.randn_like(target_actions)
    components: dict[str, torch.Tensor] = {}
    if config.objective == "rdt_denoise":
        schedule = diffusion_schedule or CosineDiffusionSchedule(DiffusionScheduleConfig())
        timesteps = schedule.sample_timesteps(target_actions.shape[0], device=target_actions.device)
        noisy = schedule.add_noise(target_actions, noise, timesteps)
        out: RDTLiteOutput = model(
            state_history=state_history,
            visual_tokens=visual_tokens,
            noisy_actions=noisy,
            time=timesteps.to(dtype=target_actions.dtype),
        )
        endpoint = out.prediction
        endpoint_terms = _endpoint_terms(endpoint, target_actions, config=config)
        clean = endpoint_terms["endpoint_full"]
        total = clean + config.first_weight * endpoint_terms["endpoint_first"] + config.first4_weight * endpoint_terms["endpoint_first4"]
        components.update({"clean_action": clean, **endpoint_terms})
    elif config.objective == "pi_flow":
        time = sample_pi_time(
            target_actions.shape[0],
            device=target_actions.device,
            dtype=target_actions.dtype,
            alpha=config.pi_time_alpha,
            beta=config.pi_time_beta,
        )
        noisy, target_velocity = pi_flow_bridge(target_actions, noise, time)
        out = model(state_history=state_history, visual_tokens=visual_tokens, noisy_actions=noisy, time=time)
        velocity = out.prediction
        flow = _weighted_mse(velocity, target_velocity, gripper_weight=config.gripper_weight)
        expanded_time = time.view(target_actions.shape[0], 1, 1)
        endpoint = noisy - expanded_time * velocity
        endpoint_terms = _endpoint_terms(endpoint, target_actions, config=config)
        total = flow + config.pi_endpoint_weight * endpoint_terms["endpoint_full"]
        total = total + config.first_weight * endpoint_terms["endpoint_first"] + config.first4_weight * endpoint_terms["endpoint_first4"]
        components.update({"flow_velocity": flow, **endpoint_terms})
    else:  # pragma: no cover
        raise ValueError(config.objective)
    components["total"] = total
    return RDTLiteLossResult(total, out.prediction, endpoint, noisy, time if config.objective == "pi_flow" else timesteps, components, out.diagnostics)
