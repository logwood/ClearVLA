"""Boundary-conditioned source processes for native arm trajectories.

This module owns only the distribution at the noisy endpoint of the arm flow
bridge. It deliberately has no access to task evidence, target actions, flow
time, or controller state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ArmSourceGeometry:
    mode: str
    scale: float
    innovation_weight: float
    velocity_weight: float
    acceleration_weight: float


def _trace_normalized_factor(operator: Tensor) -> Tensor:
    """Normalize an HxH operator to unit mean marginal variance."""

    horizon = int(operator.shape[0])
    energy = operator.square().sum() / float(horizon)
    return operator / energy.clamp_min(torch.finfo(operator.dtype).eps).sqrt()


def _boundary_difference(horizon: int, *, dtype: torch.dtype) -> Tensor:
    difference = torch.eye(horizon, dtype=dtype)
    if horizon > 1:
        difference[1:, :-1] -= torch.eye(horizon - 1, dtype=dtype)
    return difference


class BoundaryConditionedArmSource(nn.Module):
    """Sample a native arm source trajectory conditioned only on current state.

    ``ar1`` preserves the historical conditioned AR(1) process exactly.
    ``boundary_multiscale`` uses a full-rank covariance assembled from position
    innovations and once/twice integrated innovations. Each operator is trace
    normalized before mixing, so the normalized weights describe shape while
    ``scale`` is the sole owner of total stochastic energy.
    """

    MODES = frozenset({"ar1", "boundary_multiscale"})

    def __init__(
        self,
        *,
        horizon: int,
        arm_dim: int,
        mode: str,
        temporal_rho: float,
        scale: float,
        innovation_weight: float,
        velocity_weight: float,
        acceleration_weight: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.arm_dim = int(arm_dim)
        self.mode = str(mode)
        if self.horizon < 1 or self.arm_dim < 1:
            raise ValueError("arm source horizon and dimension must be positive")
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported arm source mode: {self.mode}")

        rho = float(temporal_rho)
        if not 0.0 <= rho < 1.0:
            raise ValueError("arm source temporal_rho must be in [0,1)")
        scale = float(scale)
        if not scale > 0.0:
            raise ValueError("arm source scale must be positive")
        raw_weights = torch.tensor(
            [innovation_weight, velocity_weight, acceleration_weight],
            dtype=torch.float64,
        )
        if bool((raw_weights < 0.0).any()) or not float(raw_weights.sum()) > 0.0:
            raise ValueError("arm source component weights must be non-negative with positive sum")
        if self.mode == "boundary_multiscale" and not float(raw_weights[0]) > 0.0:
            raise ValueError(
                "boundary_multiscale requires positive innovation weight for an explicit full-rank floor"
            )
        weights = raw_weights / raw_weights.sum()
        self.geometry = ArmSourceGeometry(
            mode=self.mode,
            scale=1.0 if self.mode == "ar1" else scale,
            innovation_weight=float(weights[0]),
            velocity_weight=float(weights[1]),
            acceleration_weight=float(weights[2]),
        )

        rows = torch.arange(self.horizon, dtype=torch.float64)[:, None]
        cols = torch.arange(self.horizon, dtype=torch.float64)[None]
        lag = rows - cols
        ar_innovation = (1.0 - rho * rho) ** 0.5 * torch.where(
            lag >= 0,
            torch.as_tensor(rho, dtype=torch.float64) ** lag.clamp_min(0),
            torch.zeros_like(lag),
        )
        ar_state_gain = torch.as_tensor(rho, dtype=torch.float64) ** torch.arange(
            1, self.horizon + 1, dtype=torch.float64
        )

        identity = torch.eye(self.horizon, dtype=torch.float64)
        integrate = torch.tril(torch.ones(self.horizon, self.horizon, dtype=torch.float64))
        integrate_twice = integrate @ integrate
        operators = torch.stack(
            [
                _trace_normalized_factor(identity),
                _trace_normalized_factor(integrate),
                _trace_normalized_factor(integrate_twice),
            ],
            dim=0,
        )
        component_covariances = operators @ operators.transpose(-1, -2)
        multiscale_covariance = (weights[:, None, None] * component_covariances).sum(dim=0) * (
            scale * scale
        )
        # The positive innovation floor makes this covariance strictly SPD.
        multiscale_factor = torch.linalg.cholesky(multiscale_covariance)

        if self.mode == "ar1":
            factor = ar_innovation
            state_gain = ar_state_gain
            covariance = ar_innovation @ ar_innovation.T
        else:
            factor = multiscale_factor
            state_gain = torch.ones(self.horizon, dtype=torch.float64)
            covariance = multiscale_covariance

        difference = _boundary_difference(self.horizon, dtype=torch.float64)
        acceleration = difference @ difference
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        trace = eigenvalues.sum()
        effective_dimension = trace.square() / eigenvalues.square().sum().clamp_min(1e-24)
        condition = eigenvalues[-1] / eigenvalues[0].clamp_min(1e-24)
        delta_covariance = difference @ covariance @ difference.T
        acceleration_covariance = acceleration @ covariance @ acceleration.T

        self.register_buffer("factor", factor.to(torch.float32), persistent=False)
        self.register_buffer("state_gain", state_gain.to(torch.float32), persistent=False)
        self.register_buffer("covariance", covariance.to(torch.float32), persistent=False)
        self.register_buffer(
            "component_covariances",
            component_covariances.to(torch.float32),
            persistent=False,
        )
        self.register_buffer("difference", difference.to(torch.float32), persistent=False)
        self.register_buffer("acceleration", acceleration.to(torch.float32), persistent=False)
        self.register_buffer(
            "covariance_effective_dimension",
            effective_dimension.to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "covariance_condition",
            condition.to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expected_source_rms",
            (torch.trace(covariance) / float(self.horizon)).sqrt().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expected_delta_rms",
            (torch.trace(delta_covariance) / float(self.horizon)).sqrt().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expected_acceleration_rms",
            (torch.trace(acceleration_covariance) / float(self.horizon)).sqrt().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expected_first_step_std",
            covariance[0, 0].sqrt().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expected_terminal_std",
            covariance[-1, -1].sqrt().to(torch.float32),
            persistent=False,
        )

    def conditional_mean(self, state_arm: Tensor) -> Tensor:
        self._check_state(state_arm)
        gain = self.state_gain.to(device=state_arm.device, dtype=torch.float32)
        with torch.autocast(device_type=state_arm.device.type, enabled=False):
            mean = gain[None, :, None] * state_arm.float()[:, None]
        return mean.to(dtype=state_arm.dtype)

    def sample(
        self,
        state_arm: Tensor,
        *,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._check_state(state_arm)
        white = torch.randn(
            int(state_arm.shape[0]),
            self.horizon,
            self.arm_dim,
            device=state_arm.device,
            dtype=dtype,
            generator=generator,
        )
        factor = self.factor.to(device=state_arm.device, dtype=torch.float32)
        mean = self.conditional_mean(state_arm.to(dtype=dtype))
        with torch.autocast(device_type=state_arm.device.type, enabled=False):
            stochastic = torch.einsum("ts,bsd->btd", factor, white.float())
            source = mean.float() + stochastic
        return source.to(dtype=dtype)

    @torch.no_grad()
    def diagnostics(self, source: Tensor, state_arm: Tensor) -> dict[str, Tensor]:
        self._check_source(source)
        self._check_state(state_arm)
        mean = self.conditional_mean(state_arm.to(device=source.device, dtype=source.dtype))
        stochastic = source.float() - mean.float()
        difference = self.difference.to(device=source.device, dtype=torch.float32)
        acceleration = self.acceleration.to(device=source.device, dtype=torch.float32)
        with torch.autocast(device_type=source.device.type, enabled=False):
            delta = torch.einsum("ts,bsd->btd", difference, stochastic)
            second_delta = torch.einsum("ts,bsd->btd", acceleration, stochastic)
        boundary_error = source[:, 0].float() - state_arm.to(
            device=source.device, dtype=torch.float32
        )
        return {
            "arm_source_residual_rms": stochastic.square().mean().sqrt(),
            "arm_source_delta_rms": delta.square().mean().sqrt(),
            "arm_source_acceleration_rms": second_delta.square().mean().sqrt(),
            "arm_source_first_step_rms": boundary_error.square().mean().sqrt(),
            "arm_source_expected_rms": self.expected_source_rms.to(source.device),
            "arm_source_expected_delta_rms": self.expected_delta_rms.to(source.device),
            "arm_source_expected_acceleration_rms": self.expected_acceleration_rms.to(
                source.device
            ),
            "arm_source_expected_first_step_std": self.expected_first_step_std.to(source.device),
            "arm_source_expected_terminal_std": self.expected_terminal_std.to(source.device),
            "arm_source_covariance_effective_dimension": self.covariance_effective_dimension.to(
                source.device
            ),
            "arm_source_covariance_condition": self.covariance_condition.to(source.device),
        }

    def _check_state(self, state_arm: Tensor) -> None:
        if state_arm.ndim != 2 or int(state_arm.shape[-1]) != self.arm_dim:
            raise ValueError(f"state_arm must be [B,{self.arm_dim}], got {tuple(state_arm.shape)}")

    def _check_source(self, source: Tensor) -> None:
        if (
            source.ndim != 3
            or int(source.shape[1]) != self.horizon
            or int(source.shape[2]) != self.arm_dim
        ):
            raise ValueError(
                f"source must be [B,{self.horizon},{self.arm_dim}], got {tuple(source.shape)}"
            )
