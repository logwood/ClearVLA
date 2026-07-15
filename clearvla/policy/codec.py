from __future__ import annotations

"""Typed physical action coordinates shared by current and legacy policies."""

from typing import Protocol

import torch
from torch import Tensor, nn


class PhysicalActionConfig(Protocol):
    action_horizon: int
    hidden_size: int
    gripper_field_dim: int
    gripper_field_mode: str
    arm_flow_mode: str
    arm_noise_temporal_rho: float
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
        delays = torch.cat([
            torch.zeros(direct_channels, dtype=torch.long),
            torch.arange(1, delayed_channels + 1, dtype=torch.long),
        ]).clamp_max(horizon - 1)
        source = torch.arange(horizon, dtype=torch.long)[:, None]
        valid = source + delays[None] < horizon
        weights = valid.to(torch.float64)
        weights = weights / weights.square().sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        analysis_matrix = torch.zeros(horizon, channels, horizon, dtype=torch.float64)
        for source_step in range(horizon):
            for channel, delay in enumerate(delays.tolist()):
                output_step = source_step + int(delay)
                if output_step < horizon:
                    analysis_matrix[output_step, channel, source_step] = weights[source_step, channel]
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
        if field.ndim != 3 or int(field.shape[1]) != self.horizon or int(field.shape[2]) != self.channels:
            raise ValueError(f"gripper field must be [B,{self.horizon},{self.channels}], got {tuple(field.shape)}")
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
        self.register_buffer("arm_difference_matrix", difference.to(torch.float32), persistent=False)
        self.register_buffer("arm_projection_inverse", torch.linalg.inv(gram).to(torch.float32), persistent=False)

        rho = float(config.arm_noise_temporal_rho)
        rows = torch.arange(horizon, dtype=torch.float64)[:, None]
        cols = torch.arange(horizon, dtype=torch.float64)[None]
        lag = rows - cols
        innovation = (1.0 - rho * rho) ** 0.5 * torch.where(
            lag >= 0,
            torch.as_tensor(rho, dtype=torch.float64) ** lag.clamp_min(0),
            torch.zeros_like(lag),
        )
        state_gain = torch.as_tensor(rho, dtype=torch.float64) ** (
            torch.arange(1, horizon + 1, dtype=torch.float64)
        )
        self.register_buffer(
            "arm_noise_innovation_matrix",
            innovation.to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "arm_noise_state_gain",
            state_gain.to(torch.float32),
            persistent=False,
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
        return torch.cat([action_state[:, None].to(dtype=action.dtype, device=action.device), action[:, :-1]], dim=1)

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

    def _arm_difference(self, arm: Tensor) -> Tensor:
        matrix = self.arm_difference_matrix.to(device=arm.device, dtype=torch.float32)
        # V70: keep the consistency-defining arithmetic out of bf16 autocast
        # (einsum autocasts even with .float() inputs).
        with torch.autocast(device_type=arm.device.type, enabled=False):
            delta = torch.einsum("ts,bsd->btd", matrix, arm.float())
        return delta.to(dtype=arm.dtype)

    def _sample_native_arm_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        action_state: Tensor,
    ) -> Tensor:
        white = torch.randn(
            int(batch), int(self.config.action_horizon), self.arm_dim,
            device=device, dtype=dtype, generator=generator,
        )
        state_arm, _ = self.split_action(action_state.to(device=device, dtype=dtype))
        innovation = self.arm_noise_innovation_matrix.to(device=device, dtype=torch.float32)
        state_gain = self.arm_noise_state_gain.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            correlated = torch.einsum("ts,bsd->btd", innovation, white.float())
            correlated = correlated + state_gain[None, :, None] * state_arm.float()[:, None]
        return correlated.to(dtype=dtype)

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
        state_arm, _ = self.split_action(action_state.to(device=arm_field.device, dtype=arm_field.dtype))
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
            raise ValueError(f"physical must be [B,T,{self.physical_dim}], got {tuple(physical.shape)}")
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
                *shape, self.physical_dim, device=device, dtype=dtype, generator=generator,
            )
        if self.uses_arm_manifold:
            if action_state is None:
                raise ValueError("manifold_native arm noise requires action_state")
            if int(action_state.shape[0]) != int(batch):
                raise ValueError("action_state batch does not match requested noise batch")
            native_arm_noise = self._sample_native_arm_noise(
                batch, device=device, dtype=dtype, generator=generator,
                action_state=action_state,
            )
            arm_abs, arm_delta = self.encode_arm_coordinates(native_arm_noise, action_state)
            arm_noise = torch.cat([arm_abs, arm_delta], dim=-1)
        else:
            arm_noise = torch.randn(
                *shape, 2 * self.arm_dim, device=device, dtype=dtype, generator=generator,
            )
        if self.gripper_frame is None:
            grip_noise = torch.randn(
                *shape, self.gripper_field_dim, device=device, dtype=dtype, generator=generator,
            )
            return torch.cat([arm_noise, grip_noise], dim=-1)
        native_grip_noise = torch.randn(*shape, 1, device=device, dtype=dtype, generator=generator)
        grip_noise = self.gripper_frame.analysis(native_grip_noise)
        return torch.cat([arm_noise, grip_noise], dim=-1)

    def _encode_gripper_field(self, grip: Tensor, prev_grip: Tensor, action_state: Tensor) -> Tensor:
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
        pad = torch.zeros(*field.shape[:-1], dim - int(field.shape[-1]), device=field.device, dtype=field.dtype)
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
            "gripper_delta": gripper_delta if self.uses_parseval_gripper_field else gripper_field[..., 1:2],
            "gripper_extra": gripper_field if self.uses_parseval_gripper_field else gripper_field[..., 2:],
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
            state_arm, state_grip = self.split_action(action_state.to(device=physical.device, dtype=physical.dtype))
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

    def delta_consistency_loss(self, physical: Tensor, action_state: Tensor, decoded_action: Tensor) -> Tensor:
        """Consistency between physical delta channels and decoded action deltas."""
        boundary = self.boundary(decoded_action, action_state.to(decoded_action.device, decoded_action.dtype))
        arm, grip = self.split_action(decoded_action)
        prev_arm, prev_grip = self.split_action(boundary)
        actual_delta = torch.cat([arm - prev_arm, grip - prev_grip], dim=-1)
        parts = self.split_physical(physical)
        if self.uses_parseval_gripper_field:
            return torch.nn.functional.smooth_l1_loss(actual_delta[..., : self.arm_dim], parts["arm_delta"])
        physical_delta = torch.cat([parts["arm_delta"], parts["gripper_delta"]], dim=-1)
        return torch.nn.functional.smooth_l1_loss(actual_delta, physical_delta)


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
        self.mix = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h))

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
                grip_extra = torch.zeros(*grip_field.shape[:-1], 1, device=physical.device, dtype=physical.dtype)
            assert self.grip_value is not None and self.grip_delta is not None and self.grip_extra is not None
            x = (
                x
                + self.grip_value(grip_value) + comp[:, 2, None]
                + self.grip_delta(grip_delta) + comp[:, 3, None]
                + self.grip_extra(grip_extra) + comp[:, 4, None]
            )
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
        parts = [self.arm_abs(x), self.arm_delta(x), self.grip_value(grip_x), self.grip_delta(grip_x)]
        if int(self.grip_extra.out_features) > 0:
            parts.append(self.grip_extra(grip_x))
        return torch.cat(parts, dim=-1)
