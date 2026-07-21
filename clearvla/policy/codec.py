from __future__ import annotations

"""Typed physical action coordinates shared by current and legacy policies."""

from typing import Protocol

import torch
from torch import Tensor, nn

from .source_process import BoundaryConditionedArmSource


def _sinusoidal_position_indices(indices: Tensor, hidden_size: int) -> Tensor:
    """Build deterministic positions without importing the policy primitives."""
    hidden_size = int(hidden_size)
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    half = hidden_size // 2
    if half < 1:
        return torch.zeros(int(indices.numel()), hidden_size, dtype=torch.float32)
    frequency = torch.exp(
        -torch.log(torch.tensor(10000.0, dtype=torch.float64))
        * torch.arange(half, dtype=torch.float64)
        / float(max(half - 1, 1))
    )
    phase = indices.to(torch.float64)[:, None] * frequency[None]
    position = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if int(position.shape[-1]) < hidden_size:
        position = torch.nn.functional.pad(position, (0, hidden_size - int(position.shape[-1])))
    return position[:, :hidden_size].to(torch.float32)


def _orthonormal_dct_matrix(horizon: int) -> Tensor:
    """Build the orthonormal DCT-II matrix used by FAST-style encoding."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("DCT horizon must be positive")
    time = torch.arange(horizon, dtype=torch.float64)[None, :]
    frequency = torch.arange(horizon, dtype=torch.float64)[:, None]
    matrix = torch.cos(torch.pi / float(horizon) * (time + 0.5) * frequency)
    matrix[0] *= float(horizon) ** -0.5
    if horizon > 1:
        matrix[1:] *= (2.0 / float(horizon)) ** 0.5
    return matrix.to(torch.float32)


class TemporalDCT(nn.Module):
    """Exact orthonormal DCT/IDCT over the action horizon.

    FAST additionally quantizes and BPE-encodes coefficients for
    autoregressive VLAs. Those operations deliberately do not belong in the
    continuous flow-matching path.
    """

    def __init__(self, horizon: int) -> None:
        super().__init__()
        horizon = int(horizon)
        self.horizon = horizon
        self.register_buffer(
            "matrix",
            _orthonormal_dct_matrix(horizon),
            persistent=True,
        )

    def _check_input(self, values: Tensor, name: str) -> None:
        if values.ndim < 2 or int(values.shape[-2]) != self.horizon:
            raise ValueError(
                f"{name} must have a horizon dimension of {self.horizon}, got {tuple(values.shape)}"
            )

    @staticmethod
    def _compute_dtype(values: Tensor) -> torch.dtype:
        return torch.float64 if values.dtype == torch.float64 else torch.float32

    def encode(self, values: Tensor) -> Tensor:
        """Map ``[..., time, channels]`` to ``[..., frequency, channels]``."""
        self._check_input(values, "values")
        compute_dtype = self._compute_dtype(values)
        with torch.autocast(device_type=values.device.type, enabled=False):
            matrix = self.matrix.to(device=values.device, dtype=compute_dtype)
            coefficients = torch.einsum("kt,...tc->...kc", matrix, values.to(dtype=compute_dtype))
        return coefficients.to(dtype=values.dtype)

    def decode(self, coefficients: Tensor) -> Tensor:
        """Map ``[..., frequency, channels]`` back to native time."""
        self._check_input(coefficients, "coefficients")
        compute_dtype = self._compute_dtype(coefficients)
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            matrix = self.matrix.to(device=coefficients.device, dtype=compute_dtype)
            values = torch.einsum(
                "tk,...kc->...tc",
                matrix.transpose(0, 1),
                coefficients.to(dtype=compute_dtype),
            )
        return values.to(dtype=coefficients.dtype)

    def forward(self, values: Tensor) -> Tensor:
        return self.encode(values)

    def low_frequency(self, coefficients: Tensor, keep: int) -> Tensor:
        """Return a truncated copy; the source tensor is never modified."""
        self._check_input(coefficients, "coefficients")
        keep = int(keep)
        if keep < 1 or keep > self.horizon:
            raise ValueError(f"keep must be in [1, {self.horizon}], got {keep}")
        if keep == self.horizon:
            return coefficients.clone()
        mask = torch.arange(self.horizon, device=coefficients.device) < keep
        view_shape = [1] * (coefficients.ndim - 2) + [self.horizon, 1]
        return coefficients * mask.view(*view_shape)

    def frequency_energy(self, coefficients: Tensor) -> Tensor:
        """Return mean squared coefficient energy for each frequency."""
        self._check_input(coefficients, "coefficients")
        reduce_dims = tuple(range(coefficients.ndim - 2)) + (coefficients.ndim - 1,)
        return coefficients.float().square().mean(dim=reduce_dims)


class ActionTemporalDCT(nn.Module):
    """Shared time chart with separate arm/gripper analysis boundaries."""

    def __init__(
        self,
        horizon: int,
        *,
        arm_dims: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        gripper_index: int = -1,
    ) -> None:
        super().__init__()
        self.temporal = TemporalDCT(horizon)
        self.horizon = int(horizon)
        self.arm_dims = tuple(int(index) for index in arm_dims)
        self.gripper_index = int(gripper_index)

    def _resolve_gripper(self, action_dim: int) -> int:
        index = self.gripper_index
        if index < 0:
            index += int(action_dim)
        if index < 0 or index >= int(action_dim):
            raise ValueError(
                f"gripper index {self.gripper_index} resolves outside action_dim={action_dim}"
            )
        return index

    def _validate_groups(self, action_dim: int) -> int:
        gripper = self._resolve_gripper(action_dim)
        if len(set(self.arm_dims)) != len(self.arm_dims):
            raise ValueError("arm_dims must not contain duplicates")
        if any(index < 0 or index >= int(action_dim) for index in self.arm_dims):
            raise ValueError(f"arm_dims={self.arm_dims} outside action_dim={action_dim}")
        if gripper in self.arm_dims:
            raise ValueError("gripper dimension must not also be listed in arm_dims")
        return gripper

    def encode(self, action: Tensor) -> Tensor:
        if action.ndim < 3:
            raise ValueError(f"action must have [.., time, channels], got {tuple(action.shape)}")
        self._validate_groups(int(action.shape[-1]))
        return self.temporal.encode(action)

    def decode(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim < 3:
            raise ValueError(
                f"coefficients must have [.., frequency, channels], got {tuple(coefficients.shape)}"
            )
        self._validate_groups(int(coefficients.shape[-1]))
        return self.temporal.decode(coefficients)

    def groups(self, values: Tensor) -> dict[str, Tensor]:
        if values.ndim < 3:
            raise ValueError(f"values must have [.., time, channels], got {tuple(values.shape)}")
        gripper = self._validate_groups(int(values.shape[-1]))
        return {
            "arm": values[..., list(self.arm_dims)],
            "gripper": values[..., [gripper]],
        }

    def group_frequency_energy(self, coefficients: Tensor) -> dict[str, Tensor]:
        return {
            name: self.temporal.frequency_energy(value)
            for name, value in self.groups(coefficients).items()
        }

    def low_frequency(
        self,
        coefficients: Tensor,
        *,
        arm_keep: int | None = None,
        gripper_keep: int | None = None,
    ) -> Tensor:
        """Truncate groups independently without changing the source tensor."""
        if coefficients.ndim < 3:
            raise ValueError(
                f"coefficients must have [.., frequency, channels], got {tuple(coefficients.shape)}"
            )
        action_dim = int(coefficients.shape[-1])
        gripper = self._validate_groups(action_dim)
        for name, keep in (("arm", arm_keep), ("gripper", gripper_keep)):
            if keep is not None and (int(keep) < 1 or int(keep) > self.horizon):
                raise ValueError(f"{name}_keep must be in [1, {self.horizon}], got {keep}")
        mask = torch.ones(
            self.horizon,
            action_dim,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        if arm_keep is not None and int(arm_keep) < self.horizon:
            mask[int(arm_keep) :, list(self.arm_dims)] = 0
        if gripper_keep is not None and int(gripper_keep) < self.horizon:
            mask[int(gripper_keep) :, gripper] = 0
        view_shape = [1] * (coefficients.ndim - 2) + [self.horizon, action_dim]
        return coefficients * mask.view(*view_shape)


class SoftSpectralAperture(nn.Module):
    """Continuous, group-aware bandwidth for coefficient-space refinement.

    The aperture never changes the coefficient state dimension. It only gives
    a refinement level a soft preference over frequency directions. Controller
    output is interpreted as a global budget displacement plus a monotone warp
    of the frequency coordinate. It cannot independently toggle unrelated
    frequencies or violate the low-to-high refinement order. The sigmoid
    front, monotone schedule, and final full-band interpolation avoid the
    discontinuity and zero-gradient behavior of hard K truncation.
    """

    def __init__(
        self,
        horizon: int,
        *,
        arm_channels: int,
        gripper_channels: int,
        arm_start_fraction: float = 0.16,
        gripper_start_fraction: float = 0.33,
        temperature: float = 1.5,
        schedule_power: float = 1.0,
        controller_shift_limit: float = 2.0,
    ) -> None:
        super().__init__()
        horizon = int(horizon)
        arm_channels = int(arm_channels)
        gripper_channels = int(gripper_channels)
        if horizon < 1 or arm_channels < 1 or gripper_channels < 1:
            raise ValueError("spectral aperture dimensions must be positive")
        if not 0.0 < float(arm_start_fraction) <= 1.0:
            raise ValueError("arm_start_fraction must be in (0, 1]")
        if not 0.0 < float(gripper_start_fraction) <= 1.0:
            raise ValueError("gripper_start_fraction must be in (0, 1]")
        if float(temperature) <= 0.0 or float(schedule_power) <= 0.0:
            raise ValueError("spectral temperature and schedule power must be positive")
        if float(controller_shift_limit) < 0.0:
            raise ValueError("controller_shift_limit must be non-negative")
        self.horizon = horizon
        self.arm_channels = arm_channels
        self.gripper_channels = gripper_channels
        self.arm_start_fraction = float(arm_start_fraction)
        self.gripper_start_fraction = float(gripper_start_fraction)
        self.temperature = float(temperature)
        self.schedule_power = float(schedule_power)
        self.controller_shift_limit = float(controller_shift_limit)
        self.register_buffer(
            "frequency_index",
            torch.arange(horizon, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _smooth_progress(progress: Tensor) -> Tensor:
        progress = progress.float().clamp(0.0, 1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    def _frequency_coordinates(
        self,
        shifts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Turn local controller fields into ordered frequency coordinates."""

        batch = int(shifts.shape[0])
        device = shifts.device
        native = self.frequency_index.to(device=device)[None, :, None]
        if self.horizon == 1:
            coordinate = torch.zeros(batch, 1, 2, device=device, dtype=torch.float32)
            unit = torch.ones(batch, device=device, dtype=torch.float32)
            zero = torch.zeros(batch, device=device, dtype=torch.float32)
            return coordinate, unit, unit, zero

        global_shift = shifts.mean(dim=1, keepdim=True)
        local_shift = shifts - global_shift
        interval_logits = 0.5 * (local_shift[:, 1:] + local_shift[:, :-1])
        # The controller limit determines the budget displacement, not an
        # arbitrary coordinate scale. Normalizing it here keeps the warp's
        # conditioning stable when experiments change that limit.
        if self.controller_shift_limit > 0.0:
            interval_logits = (0.5 / self.controller_shift_limit) * interval_logits
        else:
            interval_logits = torch.zeros_like(interval_logits)
        spacing = interval_logits.exp()
        spacing = spacing * (
            float(self.horizon - 1) / spacing.sum(dim=1, keepdim=True).clamp_min(1e-6)
        )
        coordinate = torch.cat(
            [
                torch.zeros(batch, 1, 2, device=device, dtype=torch.float32),
                spacing.cumsum(dim=1),
            ],
            dim=1,
        )
        warp_rms = (coordinate - native).square().mean(dim=(1, 2)).sqrt()
        return (
            coordinate,
            spacing.amin(dim=(1, 2)),
            spacing.amax(dim=(1, 2)),
            warp_rms,
        )

    def forward(
        self,
        progress: Tensor | float,
        *,
        controller_shift: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not torch.is_tensor(progress):
            progress = torch.tensor(float(progress), device=self.frequency_index.device)
        if progress.ndim == 0:
            progress = progress[None]
        if progress.ndim != 1:
            raise ValueError("spectral progress must be scalar or [B]")
        device = progress.device
        p = self._smooth_progress(progress).pow(self.schedule_power)
        arm_start = max(1.0, self.arm_start_fraction * float(self.horizon - 1))
        grip_start = max(1.0, self.gripper_start_fraction * float(self.horizon - 1))
        arm_cutoff = arm_start + (float(self.horizon - 1) - arm_start) * p
        grip_cutoff = grip_start + (float(self.horizon - 1) - grip_start) * p
        shifts = torch.zeros(
            int(progress.shape[0]),
            self.horizon,
            2,
            device=device,
            dtype=torch.float32,
        )
        if controller_shift is not None:
            if tuple(controller_shift.shape) == (int(progress.shape[0]), 2):
                controller_shift = controller_shift[:, None].expand(-1, self.horizon, -1)
            if tuple(controller_shift.shape) != tuple(shifts.shape):
                raise ValueError(
                    "controller_shift must have shape [B,2] or [B,F,2], "
                    f"got {tuple(controller_shift.shape)}"
                )
            shifts = self.controller_shift_limit * torch.tanh(controller_shift.float())
        global_shift = shifts.mean(dim=1)
        frequency_coordinate, spacing_min, spacing_max, warp_rms = self._frequency_coordinates(
            shifts
        )
        arm_front_coordinate = arm_cutoff + global_shift[..., 0]
        grip_front_coordinate = grip_cutoff + global_shift[..., 1]
        temp = max(self.temperature, 1e-3)
        arm_front = torch.sigmoid(
            (arm_front_coordinate[:, None] - frequency_coordinate[..., 0]) / temp
        )
        grip_front = torch.sigmoid(
            (grip_front_coordinate[:, None] - frequency_coordinate[..., 1]) / temp
        )
        # At the final refinement level the aperture is exactly transparent,
        # independent of temperature or controller output.
        arm_mask = (1.0 - p[:, None]) * arm_front + p[:, None]
        grip_mask = (1.0 - p[:, None]) * grip_front + p[:, None]
        token_mask = torch.maximum(arm_mask, grip_mask)
        coefficient_mask = torch.cat(
            [
                arm_mask[..., None].expand(-1, -1, self.arm_channels),
                grip_mask[..., None].expand(-1, -1, self.gripper_channels),
            ],
            dim=-1,
        )
        return {
            "coefficient_mask": coefficient_mask,
            "token_mask": token_mask,
            "arm_mask": arm_mask,
            "gripper_mask": grip_mask,
            "arm_cutoff": arm_front_coordinate,
            "gripper_cutoff": grip_front_coordinate,
            "controller_shift_rms": shifts.square().mean(dim=(1, 2)).sqrt(),
            "controller_global_shift_rms": global_shift.square().mean(dim=-1).sqrt(),
            "frequency_warp_rms": warp_rms,
            "frequency_spacing_min": spacing_min,
            "frequency_spacing_max": spacing_max,
            "progress": p,
        }


class FrequencyPhysicalActionTokenLift(nn.Module):
    """Lift full physical coefficient tokens with frequency semantics.

    Unlike the native-time lift, this class never decodes a gripper frame or
    inserts a time-position assumption. The coefficient vector is already the
    complete flow state; only group projections and frequency positions are
    added before the shared token mixer.
    """

    def __init__(self, config: "PhysicalActionConfig") -> None:
        super().__init__()
        h = int(config.hidden_size)
        arm_channels = 2 * int(config.arm_dim)
        grip_channels = int(config.physical_action_dim) - arm_channels
        if grip_channels < 1:
            raise ValueError("frequency lift requires positive gripper channels")
        self.horizon = int(config.action_horizon)
        self.arm_projection = nn.Linear(arm_channels, h)
        self.gripper_projection = nn.Linear(grip_channels, h)
        self.component_type = nn.Parameter(torch.randn(1, 2, h) * 0.02)
        self.register_buffer(
            "frequency_position",
            _sinusoidal_position_indices(torch.arange(self.horizon, dtype=torch.float32), h)[None],
            persistent=True,
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, h),
        )

    def forward(self, coefficients: Tensor) -> Tensor:
        if coefficients.ndim != 3 or int(coefficients.shape[1]) != self.horizon:
            raise ValueError(
                "frequency physical lift expects [B,horizon,channels], "
                f"got {tuple(coefficients.shape)}"
            )
        arm_channels = int(self.arm_projection.in_features)
        component = self.component_type.to(device=coefficients.device, dtype=coefficients.dtype)
        x = (
            self.arm_projection(coefficients[..., :arm_channels])
            + component[:, 0, None]
            + self.gripper_projection(coefficients[..., arm_channels:])
            + component[:, 1, None]
            + self.frequency_position.to(device=coefficients.device, dtype=coefficients.dtype)
        )
        return self.mix(x)


class PhysicalActionConfig(Protocol):
    action_horizon: int
    hidden_size: int
    gripper_field_dim: int
    gripper_field_mode: str
    arm_flow_mode: str
    arm_noise_temporal_rho: float
    arm_source_mode: str
    arm_source_scale: float
    arm_source_innovation_weight: float
    arm_source_velocity_weight: float
    arm_source_acceleration_weight: float
    physical_decode_delta_blend: float

    @property
    def arm_dim(self) -> int: ...

    @property
    def gripper_index(self) -> int: ...

    @property
    def physical_action_dim(self) -> int: ...

    def validate(self) -> None: ...


class ParsevalGripperTemporalFrame(nn.Module):
    """Causal local redundant gripper coordinates with exact reconstruction.

    Half of the channels are direct views and the rest are short delayed views
    of the same native trajectory. Every source timestep distributes unit
    energy over only valid causal observation slots; the adjoint gathers all
    views back. Consequently Phi.T @ Phi = I exactly, no future value enters an
    earlier field token, and no smoothing/filtering prior is imposed.
    """

    def __init__(self, horizon: int, channels: int) -> None:
        super().__init__()
        horizon = int(horizon)
        channels = int(channels)
        if horizon < 1 or channels < 1:
            raise ValueError("Parseval gripper frame requires positive horizon/channels")
        direct_channels = (channels + 1) // 2
        delayed_channels = channels - direct_channels
        delays = torch.cat(
            [
                torch.zeros(direct_channels, dtype=torch.long),
                torch.arange(1, delayed_channels + 1, dtype=torch.long),
            ]
        ).clamp_max(horizon - 1)
        source = torch.arange(horizon, dtype=torch.long)[:, None]
        valid = source + delays[None] < horizon
        weights = valid.to(torch.float64)
        weights = weights / weights.square().sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        analysis_matrix = torch.zeros(horizon, channels, horizon, dtype=torch.float64)
        for source_step in range(horizon):
            for channel, delay in enumerate(delays.tolist()):
                output_step = source_step + int(delay)
                if output_step < horizon:
                    analysis_matrix[output_step, channel, source_step] = weights[
                        source_step, channel
                    ]
        self.horizon = horizon
        self.channels = channels
        self.register_buffer("delays", delays, persistent=False)
        self.register_buffer("source_weights", weights.to(torch.float32), persistent=False)
        self.register_buffer("analysis_matrix", analysis_matrix.to(torch.float32), persistent=False)

    def analysis(self, gripper: Tensor) -> Tensor:
        if gripper.ndim != 3 or int(gripper.shape[1]) != self.horizon or int(gripper.shape[2]) != 1:
            raise ValueError(f"gripper must be [B,{self.horizon},1], got {tuple(gripper.shape)}")
        output_dtype = gripper.dtype
        source = gripper[..., 0].float()
        matrix = self.analysis_matrix.to(device=gripper.device, dtype=torch.float32)
        # V70: einsum is on the autocast list, so .float() alone does not keep
        # this in fp32 under bf16 autocast; the resulting ulp-level rounding
        # (~4e-3) used to set the floor of every frame-hygiene diagnostic.
        with torch.autocast(device_type=gripper.device.type, enabled=False):
            field = torch.einsum("tcs,bs->btc", matrix, source)
        return field.to(dtype=output_dtype)

    def synthesis(self, field: Tensor) -> Tensor:
        if (
            field.ndim != 3
            or int(field.shape[1]) != self.horizon
            or int(field.shape[2]) != self.channels
        ):
            raise ValueError(
                f"gripper field must be [B,{self.horizon},{self.channels}], got {tuple(field.shape)}"
            )
        output_dtype = field.dtype
        matrix = self.analysis_matrix.to(device=field.device, dtype=torch.float32)
        with torch.autocast(device_type=field.device.type, enabled=False):
            native = torch.einsum("tcs,btc->bs", matrix, field.float())
        return native[..., None].to(dtype=output_dtype)

    def project(self, field: Tensor) -> Tensor:
        return self.analysis(self.synthesis(field))


class PhysicalActionCodec(nn.Module):
    """Deterministic typed action codec for Alicia-D native actions.

    The codec is deliberately not a learned black-box.  It gives the flow a
    better coordinate system while keeping the final command contract native 7-D.
    """

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.gripper_frame = (
            ParsevalGripperTemporalFrame(config.action_horizon, config.gripper_field_dim)
            if str(config.gripper_field_mode) == "parseval_temporal"
            else None
        )
        horizon = int(config.action_horizon)
        difference = torch.eye(horizon, dtype=torch.float64)
        if horizon > 1:
            difference[1:, :-1] -= torch.eye(horizon - 1, dtype=torch.float64)
        gram = torch.eye(horizon, dtype=torch.float64) + difference.T @ difference
        self.register_buffer(
            "arm_difference_matrix", difference.to(torch.float32), persistent=False
        )
        self.register_buffer(
            "arm_projection_inverse", torch.linalg.inv(gram).to(torch.float32), persistent=False
        )

        self.arm_source = BoundaryConditionedArmSource(
            horizon=horizon,
            arm_dim=int(config.arm_dim),
            mode=str(config.arm_source_mode),
            temporal_rho=float(config.arm_noise_temporal_rho),
            scale=float(config.arm_source_scale),
            innovation_weight=float(config.arm_source_innovation_weight),
            velocity_weight=float(config.arm_source_velocity_weight),
            acceleration_weight=float(config.arm_source_acceleration_weight),
        )

    @property
    def arm_dim(self) -> int:
        return self.config.arm_dim

    @property
    def physical_dim(self) -> int:
        return self.config.physical_action_dim

    @property
    def gripper_index(self) -> int:
        return self.config.gripper_index

    @property
    def gripper_field_dim(self) -> int:
        return int(self.config.gripper_field_dim)

    @property
    def uses_parseval_gripper_field(self) -> bool:
        return self.gripper_frame is not None

    @property
    def uses_arm_manifold(self) -> bool:
        return str(self.config.arm_flow_mode) == "manifold_native"

    def split_action(self, action: Tensor) -> tuple[Tensor, Tensor]:
        gi = self.gripper_index
        grip = action[..., gi : gi + 1]
        arm = torch.cat([action[..., :gi], action[..., gi + 1 :]], dim=-1)
        return arm, grip

    def join_action(self, arm: Tensor, grip: Tensor) -> Tensor:
        gi = self.gripper_index
        return torch.cat([arm[..., :gi], grip, arm[..., gi:]], dim=-1)

    def boundary(self, action: Tensor, action_state: Tensor) -> Tensor:
        return torch.cat(
            [action_state[:, None].to(dtype=action.dtype, device=action.device), action[:, :-1]],
            dim=1,
        )

    def encode(self, action: Tensor, action_state: Tensor) -> Tensor:
        """Encode native action chunk into physical flow coordinates."""
        boundary = self.boundary(action, action_state)
        arm, grip = self.split_action(action)
        prev_arm, prev_grip = self.split_action(boundary)
        arm_delta = arm - prev_arm
        grip_field = self.encode_gripper_field(grip, prev_grip=prev_grip, action_state=action_state)
        return torch.cat([arm, arm_delta, grip_field], dim=-1)

    def encode_gripper_field(
        self,
        grip: Tensor,
        *,
        prev_grip: Tensor | None = None,
        action_state: Tensor | None = None,
    ) -> Tensor:
        if self.gripper_frame is not None:
            return self.gripper_frame.analysis(grip)
        if prev_grip is None or action_state is None:
            raise ValueError("legacy gripper field encoding requires prev_grip and action_state")
        return self._encode_gripper_field(grip, prev_grip, action_state)

    def decode_gripper_field(self, field: Tensor) -> Tensor:
        if self.gripper_frame is not None:
            return self.gripper_frame.synthesis(field)
        return field[..., :1]

    def project_gripper_field(self, field: Tensor) -> Tensor:
        if self.gripper_frame is not None:
            return self.gripper_frame.project(field)
        return field

    def encode_gripper_tangent(self, native_velocity: Tensor) -> Tensor:
        """Expand a native-time gripper velocity into the Parseval field."""
        if self.gripper_frame is None:
            raise RuntimeError(
                "gripper tangent expansion requires gripper_field_mode=parseval_temporal"
            )
        return self.gripper_frame.analysis(native_velocity)

    def _arm_difference(self, arm: Tensor) -> Tensor:
        matrix = self.arm_difference_matrix.to(device=arm.device, dtype=torch.float32)
        # V70: keep the consistency-defining arithmetic out of bf16 autocast
        # (einsum autocasts even with .float() inputs).
        with torch.autocast(device_type=arm.device.type, enabled=False):
            delta = torch.einsum("ts,bsd->btd", matrix, arm.float())
        return delta.to(dtype=arm.dtype)

    def encode_arm_tangent(self, native_velocity: Tensor) -> Tensor:
        """Expand native arm velocity into the [absolute, delta] tangent."""
        if (
            native_velocity.ndim != 3
            or int(native_velocity.shape[1]) != int(self.config.action_horizon)
            or int(native_velocity.shape[-1]) != self.arm_dim
        ):
            raise ValueError(
                "native arm velocity must be "
                f"[B,{int(self.config.action_horizon)},{self.arm_dim}], "
                f"got {tuple(native_velocity.shape)}"
            )
        return torch.cat([native_velocity, self._arm_difference(native_velocity)], dim=-1)

    def _sample_native_arm_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        action_state: Tensor,
    ) -> Tensor:
        state_arm, _ = self.split_action(action_state.to(device=device, dtype=dtype))
        return self.arm_source.sample(state_arm, dtype=dtype, generator=generator)

    @torch.no_grad()
    def arm_source_diagnostics(
        self,
        physical_source: Tensor,
        action_state: Tensor,
    ) -> dict[str, Tensor]:
        if not self.uses_arm_manifold:
            return {}
        if physical_source.ndim != 3 or int(physical_source.shape[-1]) != self.physical_dim:
            raise ValueError(
                f"physical_source must be [B,T,{self.physical_dim}], got {tuple(physical_source.shape)}"
            )
        state_arm, _ = self.split_action(
            action_state.to(device=physical_source.device, dtype=physical_source.dtype)
        )
        native_source = physical_source[..., : self.arm_dim]
        return self.arm_source.diagnostics(native_source, state_arm)

    def encode_arm_coordinates(self, arm: Tensor, action_state: Tensor) -> tuple[Tensor, Tensor]:
        state_arm, _ = self.split_action(action_state.to(device=arm.device, dtype=arm.dtype))
        delta = self._arm_difference(arm)
        delta[:, 0] = delta[:, 0] - state_arm
        return arm, delta

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Orthogonally split [absolute, delta] residuals into tangent and null parts."""
        if arm_field.ndim != 3 or int(arm_field.shape[-1]) != 2 * self.arm_dim:
            raise ValueError(
                f"arm_field must be [B,T,{2 * self.arm_dim}], got {tuple(arm_field.shape)}"
            )
        arm_abs = arm_field[..., : self.arm_dim].float()
        arm_delta = arm_field[..., self.arm_dim :].float()
        difference = self.arm_difference_matrix.to(device=arm_field.device, dtype=torch.float32)
        inverse = self.arm_projection_inverse.to(device=arm_field.device, dtype=torch.float32)
        with torch.autocast(device_type=arm_field.device.type, enabled=False):
            rhs = arm_abs + torch.einsum("ts,btd->bsd", difference, arm_delta)
            native = torch.einsum("ts,bsd->btd", inverse, rhs)
            projected_delta = torch.einsum("ts,bsd->btd", difference, native)
        projected = torch.cat([native, projected_delta], dim=-1)
        null = arm_field.float() - projected
        return (
            native.to(dtype=arm_field.dtype),
            projected.to(dtype=arm_field.dtype),
            null.to(dtype=arm_field.dtype),
        )

    def project_arm_field(self, arm_field: Tensor, action_state: Tensor) -> Tensor:
        """Project arm coordinates onto delta == finite_difference(abs, state)."""
        if arm_field.ndim != 3 or int(arm_field.shape[-1]) != 2 * self.arm_dim:
            raise ValueError(
                f"arm_field must be [B,T,{2 * self.arm_dim}], got {tuple(arm_field.shape)}"
            )
        state_arm, _ = self.split_action(
            action_state.to(device=arm_field.device, dtype=arm_field.dtype)
        )
        arm_abs = arm_field[..., : self.arm_dim].float()
        arm_delta = arm_field[..., self.arm_dim :].float()
        difference = self.arm_difference_matrix.to(device=arm_field.device, dtype=torch.float32)
        inverse = self.arm_projection_inverse.to(device=arm_field.device, dtype=torch.float32)
        adjusted_delta = arm_delta.clone()
        adjusted_delta[:, 0] = adjusted_delta[:, 0] + state_arm.float()
        with torch.autocast(device_type=arm_field.device.type, enabled=False):
            rhs = arm_abs + torch.einsum("ts,btd->bsd", difference, adjusted_delta)
            projected_abs = torch.einsum("ts,bsd->btd", inverse, rhs)
            projected_delta = torch.einsum("ts,bsd->btd", difference, projected_abs)
        projected_delta[:, 0] = projected_delta[:, 0] - state_arm.float()
        return torch.cat([projected_abs, projected_delta], dim=-1).to(dtype=arm_field.dtype)

    def project_physical(self, physical: Tensor, action_state: Tensor) -> Tensor:
        if physical.ndim != 3 or int(physical.shape[-1]) != self.physical_dim:
            raise ValueError(
                f"physical must be [B,T,{self.physical_dim}], got {tuple(physical.shape)}"
            )
        arm_field = physical[..., : 2 * self.arm_dim]
        if self.uses_arm_manifold:
            arm_field = self.project_arm_field(arm_field, action_state)
        gripper_field = self.project_gripper_field(physical[..., 2 * self.arm_dim :])
        return torch.cat([arm_field, gripper_field], dim=-1)

    def sample_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        action_state: Tensor | None = None,
    ) -> Tensor:
        shape = (int(batch), int(self.config.action_horizon))
        if not self.uses_arm_manifold and self.gripper_frame is None:
            # Preserve the exact historical RNG path for legacy checkpoints.
            return torch.randn(
                *shape,
                self.physical_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        if self.uses_arm_manifold:
            if action_state is None:
                raise ValueError("manifold_native arm noise requires action_state")
            if int(action_state.shape[0]) != int(batch):
                raise ValueError("action_state batch does not match requested noise batch")
            native_arm_noise = self._sample_native_arm_noise(
                batch,
                device=device,
                dtype=dtype,
                generator=generator,
                action_state=action_state,
            )
            arm_abs, arm_delta = self.encode_arm_coordinates(native_arm_noise, action_state)
            arm_noise = torch.cat([arm_abs, arm_delta], dim=-1)
        else:
            arm_noise = torch.randn(
                *shape,
                2 * self.arm_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        if self.gripper_frame is None:
            grip_noise = torch.randn(
                *shape,
                self.gripper_field_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            return torch.cat([arm_noise, grip_noise], dim=-1)
        native_grip_noise = torch.randn(*shape, 1, device=device, dtype=dtype, generator=generator)
        grip_noise = self.gripper_frame.analysis(native_grip_noise)
        return torch.cat([arm_noise, grip_noise], dim=-1)

    def _encode_gripper_field(
        self, grip: Tensor, prev_grip: Tensor, action_state: Tensor
    ) -> Tensor:
        state_grip = self.split_action(action_state.to(device=grip.device, dtype=grip.dtype))[1]
        delta = grip - prev_grip
        features = [
            grip,
            delta,
            grip - state_grip[:, None],
            prev_grip,
            delta.abs(),
            torch.relu(delta),
            torch.relu(-delta),
            delta - torch.cat([torch.zeros_like(delta[:, :1]), delta[:, :-1]], dim=1),
            self._lag_delta(grip, state_grip, lag=2),
            self._lag_delta(grip, state_grip, lag=4),
            self._future_delta(grip, step=1),
            self._future_delta(grip, step=4),
        ]
        field = torch.cat(features, dim=-1)
        dim = self.gripper_field_dim
        if int(field.shape[-1]) >= dim:
            return field[..., :dim]
        pad = torch.zeros(
            *field.shape[:-1], dim - int(field.shape[-1]), device=field.device, dtype=field.dtype
        )
        return torch.cat([field, pad], dim=-1)

    @staticmethod
    def _lag_delta(grip: Tensor, state_grip: Tensor, *, lag: int) -> Tensor:
        lag = max(int(lag), 1)
        prefix = state_grip[:, None].expand(-1, min(lag, int(grip.shape[1])), -1)
        if int(grip.shape[1]) > lag:
            past = torch.cat([prefix, grip[:, :-lag]], dim=1)
        else:
            past = prefix
        return grip - past[:, : int(grip.shape[1])]

    @staticmethod
    def _future_delta(grip: Tensor, *, step: int) -> Tensor:
        step = max(int(step), 1)
        horizon = int(grip.shape[1])
        if horizon > step:
            future = torch.cat([grip[:, step:], grip[:, -1:].expand(-1, step, -1)], dim=1)
        else:
            future = grip[:, -1:].expand(-1, horizon, -1)
        return future[:, :horizon] - grip

    def split_physical(self, physical: Tensor) -> dict[str, Tensor]:
        ad = self.arm_dim
        gf = self.gripper_field_dim
        gripper_field = physical[..., 2 * ad : 2 * ad + gf]
        gripper_value = self.decode_gripper_field(gripper_field)
        gripper_delta = gripper_value - torch.cat(
            [gripper_value[:, :1], gripper_value[:, :-1]], dim=1
        )
        return {
            "arm_abs": physical[..., :ad],
            "arm_delta": physical[..., ad : 2 * ad],
            "gripper_field": gripper_field,
            "gripper_value": gripper_value,
            "gripper_delta": gripper_delta
            if self.uses_parseval_gripper_field
            else gripper_field[..., 1:2],
            "gripper_extra": gripper_field
            if self.uses_parseval_gripper_field
            else gripper_field[..., 2:],
        }

    def decode(self, physical: Tensor, action_state: Tensor) -> Tensor:
        """Decode physical action representation to native Alicia-D action.

        arm_abs/gripper_value are the primary command.  A small configurable
        blend from integrated deltas lets the local-motion channel influence the
        executed command without turning the decoder into unconstrained IK.
        """
        parts = self.split_physical(physical)
        arm_abs = parts["arm_abs"]
        grip_abs = parts["gripper_value"]
        blend = float(self.config.physical_decode_delta_blend)
        if blend > 0:
            state_arm, state_grip = self.split_action(
                action_state.to(device=physical.device, dtype=physical.dtype)
            )
            arm_from_delta = state_arm[:, None] + torch.cumsum(parts["arm_delta"], dim=1)
            arm = (1.0 - blend) * arm_abs + blend * arm_from_delta
            if self.uses_parseval_gripper_field:
                grip = grip_abs
            else:
                grip_from_delta = state_grip[:, None] + torch.cumsum(parts["gripper_delta"], dim=1)
                grip = (1.0 - blend) * grip_abs + blend * grip_from_delta
        else:
            arm = arm_abs
            grip = grip_abs
        return self.join_action(arm, grip)

    def delta_consistency_loss(
        self, physical: Tensor, action_state: Tensor, decoded_action: Tensor
    ) -> Tensor:
        """Consistency between physical delta channels and decoded action deltas."""
        boundary = self.boundary(
            decoded_action, action_state.to(decoded_action.device, decoded_action.dtype)
        )
        arm, grip = self.split_action(decoded_action)
        prev_arm, prev_grip = self.split_action(boundary)
        actual_delta = torch.cat([arm - prev_arm, grip - prev_grip], dim=-1)
        parts = self.split_physical(physical)
        if self.uses_parseval_gripper_field:
            return torch.nn.functional.smooth_l1_loss(
                actual_delta[..., : self.arm_dim], parts["arm_delta"]
            )
        physical_delta = torch.cat([parts["arm_delta"], parts["gripper_delta"]], dim=-1)
        return torch.nn.functional.smooth_l1_loss(actual_delta, physical_delta)


class DCTFlowCodec(nn.Module):
    """Complete orthonormal coefficient-space flow chart.

    The physical codec owns the action field contract.  This class moves that
    contract, including its linear tangent subspaces, into the DCT coordinates
    used by the flow bridge and sampler.  No learned projection or time-domain
    action decode is involved in the flow path.
    """

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal = TemporalDCT(int(config.action_horizon))
        self.physical = PhysicalActionCodec(config)
        horizon = int(config.action_horizon)
        matrix = self.temporal.matrix.float()
        difference = self.physical.arm_difference_matrix.float()
        arm_operator = matrix @ difference @ matrix.transpose(0, 1)
        arm_gram = (
            torch.eye(horizon, dtype=torch.float32) + arm_operator.transpose(0, 1) @ arm_operator
        )
        self.register_buffer("arm_tangent_operator", arm_operator, persistent=False)
        self.register_buffer(
            "arm_projection_inverse",
            torch.linalg.inv(arm_gram),
            persistent=False,
        )
        self.register_buffer("state_basis", matrix[:, 0], persistent=False)

        if self.physical.gripper_frame is not None:
            analysis = self.physical.gripper_frame.analysis_matrix.float()
            # B maps native gripper DCT coefficients to Parseval-field DCT
            # coefficients: field_coeff[f,c] = B[f,c,k] * native_coeff[k].
            gripper_operator = torch.einsum("ft,tcs,ks->fck", matrix, analysis, matrix)
            flat_operator = gripper_operator.reshape(-1, horizon)
            gripper_gram = flat_operator.transpose(0, 1) @ flat_operator
            self.register_buffer(
                "gripper_tangent_operator",
                gripper_operator,
                persistent=False,
            )
            self.register_buffer(
                "gripper_projection_inverse",
                torch.linalg.inv(gripper_gram),
                persistent=False,
            )
        else:
            self.register_buffer(
                "gripper_tangent_operator",
                torch.empty(0),
                persistent=False,
            )
            self.register_buffer(
                "gripper_projection_inverse",
                torch.empty(0),
                persistent=False,
            )

    @property
    def horizon(self) -> int:
        return int(self.config.action_horizon)

    @property
    def physical_dim(self) -> int:
        return int(self.config.physical_action_dim)

    @property
    def arm_dim(self) -> int:
        return int(self.config.arm_dim)

    @property
    def gripper_field_dim(self) -> int:
        return int(self.config.gripper_field_dim)

    @property
    def uses_arm_manifold(self) -> bool:
        return self.physical.uses_arm_manifold

    @property
    def uses_parseval_gripper_field(self) -> bool:
        return self.physical.uses_parseval_gripper_field

    def _check_coefficients(self, values: Tensor, name: str) -> None:
        if values.ndim != 3:
            raise ValueError(
                f"{name} must be [B,{self.horizon},{self.physical_dim}], got {tuple(values.shape)}"
            )
        if int(values.shape[1]) != self.horizon:
            raise ValueError(
                f"{name} must have frequency dimension {self.horizon}, got {tuple(values.shape)}"
            )
        if int(values.shape[2]) != self.physical_dim:
            raise ValueError(
                f"{name} must have coefficient channels {self.physical_dim}, "
                f"got {tuple(values.shape)}"
            )

    def encode_physical(self, physical: Tensor) -> Tensor:
        if physical.ndim != 3 or int(physical.shape[1]) != self.horizon:
            raise ValueError(
                f"physical must be [B,{self.horizon},{self.physical_dim}], "
                f"got {tuple(physical.shape)}"
            )
        if int(physical.shape[2]) != self.physical_dim:
            raise ValueError(
                f"physical must have channels {self.physical_dim}, got {tuple(physical.shape)}"
            )
        return self.temporal.encode(physical)

    def decode_coefficients(self, coefficients: Tensor) -> Tensor:
        self._check_coefficients(coefficients, "coefficients")
        return self.temporal.decode(coefficients)

    def _expand_arm_tangent(self, native_coefficients: Tensor) -> Tensor:
        output_dtype = native_coefficients.dtype
        with torch.autocast(device_type=native_coefficients.device.type, enabled=False):
            native_fp32 = native_coefficients.float()
            delta_fp32 = torch.einsum(
                "fk,bkd->bfd",
                self.arm_tangent_operator.to(
                    device=native_coefficients.device,
                    dtype=torch.float32,
                ),
                native_fp32,
            )
            return torch.cat([native_fp32, delta_fp32], dim=-1).to(dtype=output_dtype)

    def _expand_gripper_tangent(self, native_coefficients: Tensor) -> Tensor:
        output_dtype = native_coefficients.dtype
        with torch.autocast(device_type=native_coefficients.device.type, enabled=False):
            operator = self.gripper_tangent_operator.to(
                device=native_coefficients.device,
                dtype=torch.float32,
            )
            return torch.einsum(
                "fck,bk->bfc",
                operator,
                native_coefficients[..., 0].float(),
            ).to(dtype=output_dtype)

    def expand_tangent_velocity(
        self,
        arm: Tensor,
        gripper: Tensor,
    ) -> Tensor:
        """Expand native coefficient velocities into physical field coefficients."""
        if arm.ndim != 3 or int(arm.shape[1]) != self.horizon:
            raise ValueError(
                f"arm coefficients must be [B,{self.horizon},C], got {tuple(arm.shape)}"
            )
        if gripper.ndim != 3 or int(gripper.shape[1]) != self.horizon:
            raise ValueError("gripper coefficients must have the same horizon as arm coefficients")
        expected_arm_channels = self.arm_dim if self.uses_arm_manifold else 2 * self.arm_dim
        expected_gripper_channels = (
            1 if self.uses_parseval_gripper_field else self.gripper_field_dim
        )
        if int(arm.shape[2]) != expected_arm_channels:
            raise ValueError(
                f"arm velocity head must emit {expected_arm_channels} channels, "
                f"got {tuple(arm.shape)}"
            )
        if int(gripper.shape[2]) != expected_gripper_channels:
            raise ValueError(
                f"gripper velocity head must emit {expected_gripper_channels} channels, "
                f"got {tuple(gripper.shape)}"
            )
        arm_field = self._expand_arm_tangent(arm) if self.uses_arm_manifold else arm
        grip_field = (
            self._expand_gripper_tangent(gripper) if self.uses_parseval_gripper_field else gripper
        )
        return torch.cat([arm_field, grip_field], dim=-1)

    def _project_arm(self, field: Tensor, state: Tensor | None) -> Tensor:
        arm_dim = self.arm_dim
        arm_abs = field[..., :arm_dim].float()
        arm_delta = field[..., arm_dim : 2 * arm_dim].float()
        operator = self.arm_tangent_operator.to(device=field.device, dtype=torch.float32)
        inverse = self.arm_projection_inverse.to(device=field.device, dtype=torch.float32)
        if state is None:
            state_arm = torch.zeros(
                int(field.shape[0]),
                arm_dim,
                device=field.device,
                dtype=torch.float32,
            )
        else:
            state_arm, _ = self.physical.split_action(
                state.to(device=field.device, dtype=field.dtype)
            )
            state_arm = state_arm.float()
        state_offset = (
            self.state_basis.to(device=field.device, dtype=torch.float32)[None, :, None]
            * state_arm[:, None]
        )
        adjusted_delta = arm_delta + state_offset
        native = torch.einsum(
            "ij,bjd->bid",
            inverse,
            arm_abs + torch.einsum("ji,bjd->bid", operator, adjusted_delta),
        )
        projected_delta = torch.einsum("ij,bjd->bid", operator, native) - state_offset
        return torch.cat([native, projected_delta], dim=-1).to(dtype=field.dtype)

    def _project_gripper(self, field: Tensor) -> Tensor:
        if not self.uses_parseval_gripper_field:
            return field
        operator = self.gripper_tangent_operator.to(device=field.device, dtype=torch.float32)
        inverse = self.gripper_projection_inverse.to(device=field.device, dtype=torch.float32)
        native = torch.einsum(
            "fcj,bfc->bj",
            operator,
            field.float(),
        )
        native = torch.einsum("ij,bj->bi", inverse, native)
        projected = torch.einsum("fcj,bj->bfc", operator, native)
        return projected.to(dtype=field.dtype)

    def project_tangent(self, coefficients: Tensor) -> Tensor:
        """Project a coefficient velocity onto the linear tangent space."""
        self._check_coefficients(coefficients, "coefficients")
        arm_span = 2 * self.arm_dim
        arm = coefficients[..., :arm_span]
        grip = coefficients[..., arm_span : arm_span + self.gripper_field_dim]
        if self.uses_arm_manifold:
            arm = self._project_arm(arm, None)
        if self.uses_parseval_gripper_field:
            grip = self._project_gripper(grip)
        return torch.cat([arm, grip], dim=-1)

    def project_state(self, coefficients: Tensor, action_state: Tensor) -> Tensor:
        """Project a coefficient state onto the action-state affine manifold."""
        self._check_coefficients(coefficients, "coefficients")
        if action_state.ndim != 2 or int(action_state.shape[0]) != int(coefficients.shape[0]):
            raise ValueError(
                "action_state must be [B,action_dim] with the same batch as coefficients"
            )
        arm_span = 2 * self.arm_dim
        arm = coefficients[..., :arm_span]
        grip = coefficients[..., arm_span : arm_span + self.gripper_field_dim]
        if self.uses_arm_manifold:
            arm = self._project_arm(arm, action_state)
        if self.uses_parseval_gripper_field:
            grip = self._project_gripper(grip)
        return torch.cat([arm, grip], dim=-1)

    def sample_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        action_state: Tensor | None = None,
    ) -> Tensor:
        physical = self.physical.sample_noise(
            batch,
            device=device,
            dtype=dtype,
            generator=generator,
            action_state=action_state,
        )
        return self.encode_physical(physical)


class PhysicalActionTokenLift(nn.Module):
    """Typed lift from physical action coordinates to a horizon action token."""

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ad = config.arm_dim
        self.config = config
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        self.arm_abs = nn.Linear(ad, h)
        self.arm_delta = nn.Linear(ad, h)
        if self.parseval_gripper:
            # One projection gives the whole field one semantic bandwidth unit.
            # Avoid field-only LayerNorm: it would erase the native magnitude
            # that the flow state must retain.
            self.grip_field = nn.Linear(int(config.gripper_field_dim), h)
            self.grip_field_input_scale = float(config.gripper_field_dim) ** 0.5
            self.grip_value = None
            self.grip_delta = None
            self.grip_extra = None
            component_count = 3
        else:
            self.grip_field = None
            self.grip_value = nn.Linear(1, h)
            self.grip_delta = nn.Linear(1, h)
            self.grip_extra = nn.Linear(max(int(config.gripper_field_dim) - 2, 1), h)
            component_count = 5
        self.component_type = nn.Parameter(torch.randn(1, component_count, h) * 0.02)
        self.mix = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h)
        )

    def forward(self, physical: Tensor) -> Tensor:
        ad = self.config.arm_dim
        gf = int(self.config.gripper_field_dim)
        arm_abs = physical[..., :ad]
        arm_delta = physical[..., ad : 2 * ad]
        grip_field = physical[..., 2 * ad : 2 * ad + gf]
        comp = self.component_type.to(device=physical.device, dtype=physical.dtype)
        x = self.arm_abs(arm_abs) + comp[:, 0, None] + self.arm_delta(arm_delta) + comp[:, 1, None]
        if self.parseval_gripper:
            assert self.grip_field is not None
            x = x + self.grip_field(grip_field * self.grip_field_input_scale) + comp[:, 2, None]
        else:
            grip_value = grip_field[..., :1]
            grip_delta = grip_field[..., 1:2]
            grip_extra = grip_field[..., 2:]
            if int(grip_extra.shape[-1]) == 0:
                grip_extra = torch.zeros(
                    *grip_field.shape[:-1], 1, device=physical.device, dtype=physical.dtype
                )
            assert (
                self.grip_value is not None
                and self.grip_delta is not None
                and self.grip_extra is not None
            )
            x = (
                x
                + self.grip_value(grip_value)
                + comp[:, 2, None]
                + self.grip_delta(grip_delta)
                + comp[:, 3, None]
                + self.grip_extra(grip_extra)
                + comp[:, 4, None]
            )
        return self.mix(x)


class NativeTimePhysicalActionTokenLift(nn.Module):
    """Lift physical flow state after restoring one-token/one-time semantics.

    Linear manifold coordinates are synthesized before learned projection:
    arm delta is omitted when it is determined by arm absolute state, and the
    delayed Parseval gripper frame is mapped back to its native timeline. The
    legacy independent coordinates remain available only when no exact chart
    exists for them.
    """

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        h = int(config.hidden_size)
        ad = int(config.arm_dim)
        self.config = config
        self.codec = PhysicalActionCodec(config)
        self.arm_manifold = str(config.arm_flow_mode) == "manifold_native"
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        self.arm_native = nn.Linear(ad, h) if self.arm_manifold else None
        self.arm_abs = None if self.arm_manifold else nn.Linear(ad, h)
        self.arm_delta = None if self.arm_manifold else nn.Linear(ad, h)
        if self.parseval_gripper:
            self.grip_native = nn.Linear(1, h)
            self.grip_value = None
            self.grip_delta = None
            self.grip_extra = None
        else:
            self.grip_native = None
            self.grip_value = nn.Linear(1, h)
            self.grip_delta = nn.Linear(1, h)
            self.grip_extra = nn.Linear(max(int(config.gripper_field_dim) - 2, 1), h)
        component_count = 1 + int(not self.arm_manifold) + (1 if self.parseval_gripper else 3)
        self.component_type = nn.Parameter(torch.randn(1, component_count, h) * 0.02)
        self.mix = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, 2 * h),
            nn.SiLU(),
            nn.Linear(2 * h, h),
        )

    def forward(self, physical: Tensor) -> Tensor:
        ad = int(self.config.arm_dim)
        gf = int(self.config.gripper_field_dim)
        if (
            physical.ndim != 3
            or int(physical.shape[1]) != int(self.config.action_horizon)
            or int(physical.shape[-1]) != int(self.config.physical_action_dim)
        ):
            raise ValueError(
                "native-time physical lift received an invalid flow tensor: "
                f"{tuple(physical.shape)}"
            )
        arm_field = physical[..., : 2 * ad]
        grip_field = physical[..., 2 * ad : 2 * ad + gf]
        component = self.component_type.to(device=physical.device, dtype=physical.dtype)
        index = 0
        if self.arm_manifold:
            if self.arm_native is None:
                raise RuntimeError("native arm lift is not initialized")
            # The state itself lives on an action-state-anchored affine
            # manifold. Its absolute half is already the exact native chart;
            # project_arm_tangent is reserved for zero-anchored velocities.
            native_arm = arm_field[..., :ad]
            x = self.arm_native(native_arm) + component[:, index, None]
        else:
            if self.arm_abs is None or self.arm_delta is None:
                raise RuntimeError("legacy arm lift is not initialized")
            arm_abs = arm_field[..., :ad]
            arm_delta = arm_field[..., ad:]
            x = self.arm_abs(arm_abs) + component[:, index, None]
        index += 1
        if self.arm_delta is not None:
            x = x + self.arm_delta(arm_delta) + component[:, index, None]
            index += 1
        if self.parseval_gripper:
            if self.grip_native is None:
                raise RuntimeError("native gripper lift is not initialized")
            native_grip = self.codec.decode_gripper_field(grip_field)
            x = x + self.grip_native(native_grip) + component[:, index, None]
        else:
            if self.grip_value is None or self.grip_delta is None or self.grip_extra is None:
                raise RuntimeError("legacy gripper lift is not initialized")
            grip_extra = grip_field[..., 2:]
            if int(grip_extra.shape[-1]) == 0:
                grip_extra = torch.zeros(
                    *grip_field.shape[:-1],
                    1,
                    device=physical.device,
                    dtype=physical.dtype,
                )
            x = x + self.grip_value(grip_field[..., :1]) + component[:, index, None]
            index += 1
            x = x + self.grip_delta(grip_field[..., 1:2]) + component[:, index, None]
            index += 1
            x = x + self.grip_extra(grip_extra) + component[:, index, None]
        return self.mix(x)


class PhysicalVelocityHead(nn.Module):
    """Typed velocity emitter for physical action flow coordinates."""

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ad = config.arm_dim
        self.config = config
        self.norm = nn.LayerNorm(h)
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        if self.parseval_gripper:
            self.arm_field = nn.Linear(h, 2 * ad)
            self.grip_field = nn.Linear(h, int(config.gripper_field_dim))
        else:
            self.arm_abs = nn.Linear(h, ad)
            self.arm_delta = nn.Linear(h, ad)
            self.grip_value = nn.Linear(h, 1)
            self.grip_delta = nn.Linear(h, 1)
            self.grip_extra = nn.Linear(h, max(int(config.gripper_field_dim) - 2, 0))

    def output_layers(self) -> tuple[nn.Linear, ...]:
        if self.parseval_gripper:
            return self.arm_field, self.grip_field
        return self.arm_abs, self.arm_delta, self.grip_value, self.grip_delta, self.grip_extra

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.norm(tokens)
        if self.parseval_gripper:
            return torch.cat([self.arm_field(x), self.grip_field(x)], dim=-1)
        parts = [self.arm_abs(x), self.arm_delta(x), self.grip_value(x), self.grip_delta(x)]
        if int(self.grip_extra.out_features) > 0:
            parts.append(self.grip_extra(x))
        return torch.cat(parts, dim=-1)


class TransitionAwarePhysicalVelocityHead(nn.Module):
    """V36.2 typed velocity head with in-head gripper latent modulation.

    The arm channels are emitted exactly from the normalized action tokens.  The
    gripper channels are emitted from the same tokens after a zero-initialized
    transition-latent residual.  This is not a separate gripper command head:
    the unique action output is still the typed physical velocity tensor.
    """

    def __init__(self, config: PhysicalActionConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ad = config.arm_dim
        self.config = config
        self.norm = nn.LayerNorm(h)
        self.transition_norm = nn.LayerNorm(h)
        self.gripper_delta = nn.Linear(h, h)
        self.gripper_gate = nn.Linear(h, h)
        nn.init.zeros_(self.gripper_delta.weight)
        nn.init.zeros_(self.gripper_delta.bias)
        nn.init.zeros_(self.gripper_gate.weight)
        nn.init.zeros_(self.gripper_gate.bias)
        self.parseval_gripper = str(config.gripper_field_mode) == "parseval_temporal"
        if self.parseval_gripper:
            self.arm_field = nn.Linear(h, 2 * ad)
            self.grip_field = nn.Linear(h, int(config.gripper_field_dim))
        else:
            self.arm_abs = nn.Linear(h, ad)
            self.arm_delta = nn.Linear(h, ad)
            self.grip_value = nn.Linear(h, 1)
            self.grip_delta = nn.Linear(h, 1)
            self.grip_extra = nn.Linear(h, max(int(config.gripper_field_dim) - 2, 0))

    def output_layers(self) -> tuple[nn.Linear, ...]:
        if self.parseval_gripper:
            return self.arm_field, self.grip_field
        return self.arm_abs, self.arm_delta, self.grip_value, self.grip_delta, self.grip_extra

    def forward(self, tokens: Tensor, transition_latent: Tensor | None = None) -> Tensor:
        x = self.norm(tokens)
        grip_x = x
        if transition_latent is not None:
            z = self.transition_norm(transition_latent)
            grip_x = grip_x + torch.sigmoid(self.gripper_gate(z)) * self.gripper_delta(z)
        if self.parseval_gripper:
            return torch.cat([self.arm_field(x), self.grip_field(grip_x)], dim=-1)
        parts = [
            self.arm_abs(x),
            self.arm_delta(x),
            self.grip_value(grip_x),
            self.grip_delta(grip_x),
        ]
        if int(self.grip_extra.out_features) > 0:
            parts.append(self.grip_extra(grip_x))
        return torch.cat(parts, dim=-1)
