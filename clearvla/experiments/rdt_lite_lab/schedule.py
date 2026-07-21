from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionScheduleConfig:
    """Noise schedule for the RDT-style direct denoising reference.

    The default follows the official RDT configuration choice
    ``beta_schedule: squaredcos_cap_v2``.  The implementation is self-contained
    so the lightweight reference does not depend on diffusers.
    """

    train_timesteps: int = 1000
    cosine_s: float = 0.008
    max_beta: float = 0.999

    def validate(self) -> None:
        if self.train_timesteps <= 1:
            raise ValueError("train_timesteps must be > 1")
        if self.cosine_s < 0:
            raise ValueError("cosine_s must be non-negative")
        if not 0 < self.max_beta < 1:
            raise ValueError("max_beta must be in (0,1)")


def cosine_beta_schedule(
    config: DiffusionScheduleConfig, *, device: torch.device | None = None
) -> torch.Tensor:
    config.validate()
    steps = int(config.train_timesteps)
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64, device=device)
    alpha_bar = torch.cos(
        ((x / steps + config.cosine_s) / (1.0 + config.cosine_s)) * math.pi * 0.5
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-8, max=config.max_beta).to(dtype=torch.float32)


class CosineDiffusionSchedule:
    """DDPM forward process with deterministic DDIM-style reverse updates."""

    def __init__(self, config: DiffusionScheduleConfig = DiffusionScheduleConfig()) -> None:
        config.validate()
        self.config = config
        betas = cosine_beta_schedule(config)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._betas_cpu = betas
        self._alpha_bars_cpu = alpha_bars

    @property
    def train_timesteps(self) -> int:
        return int(self.config.train_timesteps)

    def _alpha_bars(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._alpha_bars_cpu.to(device=device, dtype=dtype)

    def sample_timesteps(
        self, batch: int, *, device: torch.device, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if batch <= 0:
            raise ValueError("batch must be positive")
        return torch.randint(
            0, self.train_timesteps, (batch,), device=device, generator=generator, dtype=torch.long
        )

    def add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have the same shape")
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError("timesteps must be [B]")
        alpha_bars = self._alpha_bars(device=clean.device, dtype=clean.dtype)
        alpha = alpha_bars[timesteps].view(clean.shape[0], *([1] * (clean.ndim - 1)))
        return alpha.sqrt() * clean + (1.0 - alpha).clamp_min(1e-12).sqrt() * noise

    def inference_timesteps(self, steps: int, *, device: torch.device) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        # Keep unique, monotonically descending integer indices.  For typical
        # 4--10 step probes this preserves the exact requested count.
        values = torch.linspace(self.train_timesteps - 1, 0, steps, device=device)
        indices = values.round().to(dtype=torch.long)
        if len(torch.unique_consecutive(indices)) != len(indices):
            raise ValueError(f"inference steps={steps} produces duplicate schedule indices")
        return indices

    def ddim_step(
        self,
        noisy: torch.Tensor,
        pred_clean: torch.Tensor,
        timestep: int,
        prev_timestep: int | None,
    ) -> torch.Tensor:
        if noisy.shape != pred_clean.shape:
            raise ValueError("noisy and pred_clean must have the same shape")
        alpha_bars = self._alpha_bars(device=noisy.device, dtype=noisy.dtype)
        alpha_t = alpha_bars[int(timestep)]
        if prev_timestep is None:
            return pred_clean
        alpha_prev = alpha_bars[int(prev_timestep)]
        eps = (noisy - alpha_t.sqrt() * pred_clean) / (1.0 - alpha_t).clamp_min(1e-12).sqrt()
        return alpha_prev.sqrt() * pred_clean + (1.0 - alpha_prev).clamp_min(1e-12).sqrt() * eps


def sample_pi_time(
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float = 1.5,
    beta: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the OpenPI-style flow time in ``[0.001, 1.0)``.

    PyTorch's Beta distribution does not accept a generator argument.  To keep
    evaluation determinism simple, training uses the global RNG unless the
    common default Beta(1.5, 1.0) is replaced by a uniform fallback in tests.
    """

    if batch <= 0:
        raise ValueError("batch must be positive")
    if alpha <= 0 or beta <= 0:
        raise ValueError("beta distribution parameters must be positive")
    # The public distribution API uses the active PyTorch RNG state.  The
    # ``generator`` argument is accepted for signature symmetry but is not
    # consumed because torch.distributions.Beta.sample does not expose it.
    del generator
    concentration1 = torch.full((batch,), float(alpha), device=device, dtype=dtype)
    concentration0 = torch.full((batch,), float(beta), device=device, dtype=dtype)
    value = torch.distributions.Beta(concentration1, concentration0).sample()
    return value * 0.999 + 0.001


def pi_flow_bridge(
    actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``x_t`` and target velocity using the public OpenPI convention.

    ``x_t = t * noise + (1 - t) * actions`` and ``u_t = noise - actions``.
    Sampling starts from noise at ``t=1`` and integrates toward ``t=0``.
    """

    if actions.shape != noise.shape:
        raise ValueError("actions and noise must have the same shape")
    if time.ndim != 1 or time.shape[0] != actions.shape[0]:
        raise ValueError("time must be [B]")
    expanded = time.view(actions.shape[0], *([1] * (actions.ndim - 1)))
    return expanded * noise + (1.0 - expanded) * actions, noise - actions
