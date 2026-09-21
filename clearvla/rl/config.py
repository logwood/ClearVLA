"""Small, serializable contract for the single-task residual-SAC pilot."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SACConfig:
    hidden: int = 256
    batch_size: int = 128
    replay_capacity: int = 10000
    learning_starts: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    alpha: float = 0.02
    # Fraction of the full native action range, not of dataset z-score std.
    residual_fraction: float = 0.10
    initial_log_std: float = -3.0
    grad_clip: float = 10.0
    seed: int = 0

    def validate(self) -> None:
        for name in ("hidden", "batch_size", "replay_capacity", "learning_starts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size exceeds replay_capacity")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must cover a complete batch")
        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts exceeds replay_capacity")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in (
            "gamma",
            "tau",
            "learning_rate",
            "alpha",
            "residual_fraction",
            "initial_log_std",
            "grad_clip",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.gamma < 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid gamma/tau")
        if self.learning_rate <= 0 or self.alpha <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate, alpha and grad_clip must be positive")
        if not 0 < self.residual_fraction <= 0.25:
            raise ValueError("v1 residual_fraction must be in (0, 0.25]")
        if not -5 <= self.initial_log_std <= 1:
            raise ValueError("initial_log_std must be in [-5, 1]")

    def as_dict(self) -> dict:
        self.validate()
        return asdict(self)
