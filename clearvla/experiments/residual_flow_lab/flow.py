from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResidualBridgeConfig:
    """Prior-relative conditional-flow bridge.

    The learned history source is treated as an action-space anchor.  Training
    transports a small, temporally smooth residual source to the target
    correction ``future - learned_source``.  Deterministic inference starts at
    zero residual and integrates only the missing correction.
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
    uniform_time_probability: float = 0.30
    source_beta_alpha: float = 1.0
    source_beta_beta: float = 3.0

    def validate(self) -> None:
        probabilities = (self.clean_probability, self.mild_probability, self.strong_probability)
        if any(float(value) < 0 for value in probabilities):
            raise ValueError("bridge probabilities must be non-negative")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError(f"bridge probabilities must sum to 1, got {sum(probabilities)}")
        if (
            min(
                self.mild_noise_std,
                self.strong_noise_std,
                self.mild_velocity_bias_std,
                self.strong_velocity_bias_std,
            )
            < 0
        ):
            raise ValueError("bridge noise scales must be non-negative")
        if not 0.0 <= self.min_time < self.max_time <= 1.0:
            raise ValueError("bridge time range must satisfy 0 <= min < max <= 1")
        if not 0.0 <= self.uniform_time_probability <= 1.0:
            raise ValueError("uniform_time_probability must be in [0,1]")
        if self.source_beta_alpha <= 0 or self.source_beta_beta <= 0:
            raise ValueError("beta parameters must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualBridgeBatch:
    source_residual: torch.Tensor  # [B,K,A]
    residual_state: torch.Tensor  # [B,K,A]
    target_residual: torch.Tensor  # [B,K,A]
    target_velocity: torch.Tensor  # [B,K,A]
    time: torch.Tensor  # [B]
    step_size_hint: torch.Tensor  # [B]
    noise_level: torch.Tensor  # [B]
    corruption_level: torch.Tensor  # [B] long: 0 clean, 1 mild, 2 strong


def _smooth_noise(reference: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(reference)
    value = F.pad(noise.transpose(1, 2), (1, 1), mode="replicate")
    value = F.avg_pool1d(value, kernel_size=3, stride=1).transpose(1, 2)
    return value * std[:, None, None]


def _sample_time(
    batch: int, *, device: torch.device, dtype: torch.dtype, config: ResidualBridgeConfig
) -> torch.Tensor:
    uniform = torch.empty((batch,), device=device, dtype=dtype).uniform_(
        config.min_time, config.max_time
    )
    beta = torch.distributions.Beta(config.source_beta_alpha, config.source_beta_beta)
    source_biased = beta.sample((batch,)).to(device=device, dtype=dtype)
    source_biased = config.min_time + (config.max_time - config.min_time) * source_biased
    use_uniform = torch.rand((batch,), device=device) < config.uniform_time_probability
    return torch.where(use_uniform, uniform, source_biased)


def sample_residual_bridge(
    learned_source: torch.Tensor,
    target_actions: torch.Tensor,
    config: ResidualBridgeConfig = ResidualBridgeConfig(),
) -> ResidualBridgeBatch:
    config.validate()
    if learned_source.shape != target_actions.shape or learned_source.ndim != 3:
        raise ValueError(
            f"learned_source and target_actions must share [B,K,A], got "
            f"{tuple(learned_source.shape)} vs {tuple(target_actions.shape)}"
        )
    if not torch.isfinite(learned_source).all() or not torch.isfinite(target_actions).all():
        raise ValueError("learned_source and target_actions must be finite")
    batch, horizon, _ = learned_source.shape
    target_residual = target_actions - learned_source

    draw = torch.rand((batch,), device=learned_source.device)
    mild_boundary = config.clean_probability + config.mild_probability
    level = torch.zeros((batch,), dtype=torch.long, device=learned_source.device)
    level = torch.where(draw >= config.clean_probability, torch.ones_like(level), level)
    level = torch.where(draw >= mild_boundary, torch.full_like(level, 2), level)

    noise_std = torch.zeros((batch,), dtype=learned_source.dtype, device=learned_source.device)
    bias_std = torch.zeros_like(noise_std)
    noise_std = torch.where(
        level == 1, torch.full_like(noise_std, config.mild_noise_std), noise_std
    )
    noise_std = torch.where(
        level == 2, torch.full_like(noise_std, config.strong_noise_std), noise_std
    )
    bias_std = torch.where(
        level == 1, torch.full_like(bias_std, config.mild_velocity_bias_std), bias_std
    )
    bias_std = torch.where(
        level == 2, torch.full_like(bias_std, config.strong_velocity_bias_std), bias_std
    )

    smooth = _smooth_noise(learned_source, noise_std)
    velocity_bias = (
        torch.randn(
            (batch, 1, learned_source.shape[-1]),
            device=learned_source.device,
            dtype=learned_source.dtype,
        )
        * bias_std[:, None, None]
    )
    ramp = torch.linspace(
        0.0, 1.0, horizon, device=learned_source.device, dtype=learned_source.dtype
    )[None, :, None]
    source_residual = smooth + ramp * velocity_bias
    time = _sample_time(
        batch, device=learned_source.device, dtype=learned_source.dtype, config=config
    )
    residual_state = (1.0 - time[:, None, None]) * source_residual + time[
        :, None, None
    ] * target_residual
    target_velocity = target_residual - source_residual
    noise_level = torch.sqrt(torch.mean(source_residual.square(), dim=(1, 2)) + 1e-12)
    # During training this is a hint, not a routing decision.  A random element
    # from {1,2,4} exposes the velocity field to the deployment step sizes.
    choices = torch.tensor(
        [1.0, 0.5, 0.25], device=learned_source.device, dtype=learned_source.dtype
    )
    step_size_hint = choices[torch.randint(0, len(choices), (batch,), device=learned_source.device)]
    return ResidualBridgeBatch(
        source_residual=source_residual,
        residual_state=residual_state,
        target_residual=target_residual,
        target_velocity=target_velocity,
        time=time,
        step_size_hint=step_size_hint,
        noise_level=noise_level,
        corruption_level=level,
    )


def endpoint_from_velocity(
    state: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    if state.shape != velocity.shape or state.ndim != 3:
        raise ValueError("state and velocity must share [B,K,A]")
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("time must be [B]")
    return state + (1.0 - time[:, None, None]) * velocity
