from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionBridgeConfig:
    """Action-informed conditional-flow bridge configuration.

    The source is the motion prior plus temporally smooth noise.  Training
    samples intermediate states on the straight bridge from source to target.
    Inference starts from the clean prior and integrates the learned velocity.
    """

    clean_probability: float = 0.50
    mild_probability: float = 0.35
    strong_probability: float = 0.15
    mild_noise_std: float = 0.05
    strong_noise_std: float = 0.15
    mild_velocity_bias_std: float = 0.02
    strong_velocity_bias_std: float = 0.06
    min_time: float = 0.001
    max_time: float = 0.999

    def validate(self) -> None:
        probabilities = (self.clean_probability, self.mild_probability, self.strong_probability)
        if any(value < 0 for value in probabilities):
            raise ValueError("bridge probabilities must be non-negative")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError(f"bridge probabilities must sum to 1, got {sum(probabilities)}")
        if min(self.mild_noise_std, self.strong_noise_std, self.mild_velocity_bias_std, self.strong_velocity_bias_std) < 0:
            raise ValueError("bridge noise scales must be non-negative")
        if not 0.0 <= self.min_time < self.max_time <= 1.0:
            raise ValueError("bridge time range must satisfy 0 <= min < max <= 1")


@dataclass(frozen=True)
class ActionBridgeBatch:
    source: torch.Tensor         # [B,K,A]
    state: torch.Tensor          # [B,K,A]
    target_velocity: torch.Tensor  # [B,K,A]
    time: torch.Tensor           # [B]
    noise_level: torch.Tensor    # [B], normalized RMS magnitude
    corruption_level: torch.Tensor  # [B] long: 0 clean, 1 mild, 2 strong


def _smooth_noise(reference: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(reference)
    # A tiny temporal smoothing kernel avoids unrealistic independent action jitter.
    channels = reference.shape[-1]
    value = noise.transpose(1, 2)
    value = F.pad(value, (1, 1), mode="replicate")
    value = F.avg_pool1d(value, kernel_size=3, stride=1).transpose(1, 2)
    return value * std[:, None, None]


def sample_action_bridge(
    prior: torch.Tensor,
    target: torch.Tensor,
    config: ActionBridgeConfig = ActionBridgeConfig(),
) -> ActionBridgeBatch:
    config.validate()
    if prior.shape != target.shape or prior.ndim != 3:
        raise ValueError(f"prior and target must share [B,K,A], got {tuple(prior.shape)} vs {tuple(target.shape)}")
    if not torch.isfinite(prior).all() or not torch.isfinite(target).all():
        raise ValueError("prior and target must be finite")
    batch, horizon, _ = prior.shape
    draw = torch.rand((batch,), device=prior.device)
    mild_boundary = config.clean_probability + config.mild_probability
    level = torch.zeros((batch,), dtype=torch.long, device=prior.device)
    level = torch.where(draw >= config.clean_probability, torch.ones_like(level), level)
    level = torch.where(draw >= mild_boundary, torch.full_like(level, 2), level)
    noise_std = torch.zeros((batch,), dtype=prior.dtype, device=prior.device)
    bias_std = torch.zeros_like(noise_std)
    noise_std = torch.where(level == 1, torch.full_like(noise_std, config.mild_noise_std), noise_std)
    noise_std = torch.where(level == 2, torch.full_like(noise_std, config.strong_noise_std), noise_std)
    bias_std = torch.where(level == 1, torch.full_like(bias_std, config.mild_velocity_bias_std), bias_std)
    bias_std = torch.where(level == 2, torch.full_like(bias_std, config.strong_velocity_bias_std), bias_std)
    smooth = _smooth_noise(prior, noise_std)
    velocity_bias = torch.randn((batch, 1, prior.shape[-1]), device=prior.device, dtype=prior.dtype) * bias_std[:, None, None]
    ramp = torch.linspace(0.0, 1.0, horizon, device=prior.device, dtype=prior.dtype)[None, :, None]
    corruption = smooth + ramp * velocity_bias
    source = prior + corruption
    time = torch.empty((batch,), dtype=prior.dtype, device=prior.device).uniform_(config.min_time, config.max_time)
    state = (1.0 - time[:, None, None]) * source + time[:, None, None] * target
    velocity = target - source
    rms = torch.sqrt(torch.mean(corruption.square(), dim=(1, 2)) + 1e-12)
    return ActionBridgeBatch(source, state, velocity, time, rms, level)


def endpoint_from_velocity(state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if state.shape != velocity.shape or state.ndim != 3:
        raise ValueError("state and velocity must share [B,K,A]")
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("time must be [B]")
    return state + (1.0 - time[:, None, None]) * velocity
