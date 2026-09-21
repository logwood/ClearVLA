"""SAC is defined on unit residuals, NOT on clipped-action likelihoods."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, outputs),
    )


class ResidualActor(nn.Module):
    def __init__(self, features: int, hidden: int, initial_log_std: float) -> None:
        super().__init__()
        self.net = mlp(features, hidden, 14)
        # Exactly zero deterministic residual initially. Stochastic training
        # deliberately explores and is NOT claimed to be a zero-residual run.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[7:].fill_(initial_log_std)

    def sample(
        self, features: Tensor, *, generator: torch.Generator, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(features).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 1.0)
        noise = (
            torch.zeros_like(mean)
            if deterministic
            else torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        )
        raw = mean + log_std.exp() * noise
        # Stable tanh Jacobian, including at saturated logits.
        log_prob = -0.5 * (noise.square() + 2 * log_std + math.log(2 * math.pi))
        log_prob -= 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return raw.tanh(), log_prob.sum(-1, keepdim=True)


class TwinQ(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.q1 = mlp(features + 7, hidden, 1)
        self.q2 = mlp(features + 7, hidden, 1)

    def forward(self, features: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]:
        value = torch.cat((features, residual), dim=-1)
        return self.q1(value), self.q2(value)


class ResidualActionMap:
    """Native-space execution map; base output has already been decoded.

    The RL MDP action is u in [-1,1]^7. Clipping is part of the environment
    transition. The critic and entropy BOTH use u, never log_prob(a_executed).
    This remains mathematically coherent when several u map to a saturated
    command; saturation still wastes exploration and must be reported.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, fraction: float) -> None:
        self.low = np.asarray(low, dtype=np.float32).copy()
        self.high = np.asarray(high, dtype=np.float32).copy()
        if (
            self.low.shape != (7,)
            or self.high.shape != (7,)
            or not np.isfinite(self.low).all()
            or not np.isfinite(self.high).all()
            or np.any(self.low >= self.high)
        ):
            raise ValueError("native action bounds must be finite ordered [7] vectors")
        if not math.isfinite(fraction) or not 0 < fraction <= 0.25:
            raise ValueError("invalid residual fraction")
        self.radius = (self.high - self.low) * fraction

    def execute(self, base: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, dict]:
        base = np.asarray(base, dtype=np.float32)
        residual = np.asarray(residual, dtype=np.float32)
        if any(v.shape != (7,) or not np.isfinite(v).all() for v in (base, residual)):
            raise ValueError("base and residual must be finite [7]")
        if np.any(np.abs(residual) > 1.0):
            raise ValueError("residual is outside [-1,1]")
        baseline = np.clip(base, self.low, self.high)
        proposal = baseline + self.radius * residual
        executed = np.clip(proposal, self.low, self.high).astype(np.float32)
        return executed, {
            "base_clip_fraction": float(np.mean(baseline != base)),
            "residual_clip_fraction": float(np.mean(proposal != executed)),
            "residual_native_l2": float(np.linalg.norm(executed - baseline)),
        }
