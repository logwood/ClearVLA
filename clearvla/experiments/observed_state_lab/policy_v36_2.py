from __future__ import annotations

"""V36.2 physical-action-flow policy.

V36.2 addresses two explicit bottlenecks found in V36.1 static review:

1. The flow coordinate is no longer the raw 24x7 Alicia-D command.  The flow
   runs in a typed physical action representation containing arm absolute
   targets, arm local deltas, gripper value, and gripper delta.  The deployed
   output is still decoded back to the native 7-D Alicia-D command.
2. The action input/output bottleneck is widened at the semantic boundary: noisy
   action tokens are lifted by typed component projections, and velocity is
   emitted by typed component heads instead of a single 7-D linear head.

The world encoder and proposal contract are intentionally kept stable so that
V36.2 isolates the action-coordinate and early action-token bottleneck issues.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .policy import RejectableHistoryProposal, TimeEmbedding
from .world_model import BiasFreeFFN, V35WorldConfig, WorldEvidenceEncoder, sinusoidal_positions


@dataclass(frozen=True)
class V362PolicyConfig:
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 24
    executed_history_length: int = 3
    hidden_size: int = 512
    num_heads: int = 8
    depth: int = 6
    action_decoder_depth: int = 4
    proposal_depth: int = 2
    ffn_expansion: float = 4.0
    proposal_dropout: float = 0.25
    dropout: float = 0.05
    event_tokens: int = 3
    gripper_dim_index: int = -1
    inference_steps: int = 5
    first_execution_steps: int = 4
    mid_execution_steps: int = 8
    physical_decode_delta_blend: float = 0.25
    gripper_field_dim: int = 12
    gripper_field_mode: str = "legacy_handcrafted"
    # Historical runs sampled arm_abs/arm_delta independently. New runs can
    # instead sample one native arm trajectory and map it into the redundant
    # [absolute, delta] coordinates used by the policy.
    arm_flow_mode: str = "legacy_independent"
    arm_noise_temporal_rho: float = 0.0

    def validate(self) -> None:
        if min(
            self.action_dim,
            self.state_dim,
            self.action_horizon,
            self.executed_history_length,
            self.hidden_size,
            self.num_heads,
            self.depth,
            self.action_decoder_depth,
            self.proposal_depth,
            self.event_tokens,
            self.inference_steps,
            self.first_execution_steps,
            self.mid_execution_steps,
        ) <= 0:
            raise ValueError("V36.2 policy dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action/state dimensions must match")
        if not 0 <= self.proposal_dropout < 1:
            raise ValueError("proposal_dropout must be in [0,1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.physical_decode_delta_blend <= 1:
            raise ValueError("physical_decode_delta_blend must be in [0,1]")
        if int(self.gripper_field_dim) < 2:
            raise ValueError("gripper_field_dim must be >= 2")
        if str(self.gripper_field_mode) not in {"legacy_handcrafted", "parseval_temporal"}:
            raise ValueError("gripper_field_mode must be legacy_handcrafted or parseval_temporal")
        if str(self.arm_flow_mode) not in {"legacy_independent", "manifold_native"}:
            raise ValueError("arm_flow_mode must be legacy_independent or manifold_native")
        if not 0.0 <= float(self.arm_noise_temporal_rho) < 1.0:
            raise ValueError("arm_noise_temporal_rho must be in [0,1)")
        if self.first_execution_steps > self.action_horizon:
            raise ValueError("first_execution_steps cannot exceed action_horizon")
        if self.mid_execution_steps > self.action_horizon:
            raise ValueError("mid_execution_steps cannot exceed action_horizon")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.action_dim + self.gripper_dim_index

    @property
    def arm_dim(self) -> int:
        return self.action_dim - 1

    @property
    def physical_action_dim(self) -> int:
        # arm_abs + arm_delta + expanded gripper field. Legacy mode reserves
        # value/delta channels; Parseval mode reconstructs the native gripper
        # trajectory jointly from every field channel.
        return 2 * self.arm_dim + int(self.gripper_field_dim)


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

    def __init__(self, config: V362PolicyConfig) -> None:
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

    def __init__(self, config: V362PolicyConfig) -> None:
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

    def __init__(self, config: V362PolicyConfig) -> None:
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


class HorizonRoleEmbedding(nn.Module):
    """Explicit execution/planning role embedding for horizon tokens."""

    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.execution = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.mid = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.tail = nn.Parameter(torch.randn(1, 1, h) * 0.02)

    def forward(self, batch: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        h = self.execution.shape[-1]
        role = torch.empty(1, self.config.action_horizon, h, device=device, dtype=dtype)
        role[:, : self.config.first_execution_steps] = self.execution.to(device=device, dtype=dtype)
        role[:, self.config.first_execution_steps : self.config.mid_execution_steps] = self.mid.to(device=device, dtype=dtype)
        role[:, self.config.mid_execution_steps :] = self.tail.to(device=device, dtype=dtype)
        return role.expand(batch, -1, -1)


class DiTPlannerBlock(nn.Module):
    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(config.dropout)
        self.mod = nn.Linear(h, 6 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = self.mod(time).chunk(6, dim=-1)
        value = self.n1(x)
        query = self.modulate(value, attn_shift, attn_scale)
        update, _ = self.self_attn(query, query, value, need_weights=False)
        x = x + torch.tanh(attn_gate)[:, None] * self.drop(update)
        update = self.ffn(self.modulate(self.n2(x), ffn_shift, ffn_scale))
        return x + torch.tanh(ffn_gate)[:, None] * self.drop(update)


class PolicyLatentDiTPlannerV362(nn.Module):
    def __init__(self, config: V362PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.proposal_proj = nn.Identity()
        self.role = HorizonRoleEmbedding(config)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.world_type = nn.Parameter(torch.randn(1, world_tokens, h) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.horizon_query = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.event_query = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.event_type = nn.Parameter(torch.randn(1, config.event_tokens, h) * 0.02)
        self.register_buffer("horizon_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList([DiTPlannerBlock(config) for _ in range(config.depth)])
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))
        self.motion_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))

    def _tokens(
        self,
        noisy_physical: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> tuple[Tensor, slice, slice]:
        batch = noisy_physical.shape[0]
        hpos = self.horizon_position.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
        task = self.task_token.expand(batch, -1, -1)
        world_tokens = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        role = self.role(batch, device=noisy_physical.device, dtype=noisy_physical.dtype)
        horizon = self.horizon_query.expand(batch, -1, -1) + hpos + role + self.noisy_physical_lift(noisy_physical)
        event = self.event_query.expand(batch, -1, -1) + self.event_type
        prefix_len = 1 + world_tokens.shape[1] + 1 + executed.shape[1] + proposal.shape[1]
        action_slice = slice(prefix_len, prefix_len + self.config.action_horizon)
        event_start = prefix_len + self.config.action_horizon
        event_slice = slice(event_start, event_start + self.config.event_tokens)
        tokens = torch.cat([task, world_tokens, state_token, executed, proposal, horizon, event], dim=1)
        return tokens, action_slice, event_slice

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype)
        tokens, action_slice, event_slice = self._tokens(noisy_physical, world, state, executed_history, proposal_tokens, proposal_keep)
        time_emb = self.time(time.to(dtype=tokens.dtype))
        for block in self.blocks:
            tokens = block(tokens, time_emb)
        action_tokens = tokens[:, action_slice, :]
        event_tokens = tokens[:, event_slice, :]
        return {
            "planner_tokens": tokens,
            "planner_action_tokens": action_tokens,
            "planner_event_tokens": event_tokens,
            "event_logits": self.event_head(action_tokens),
            "motion_logits": self.motion_head(action_tokens).squeeze(-1),
        }


class ActionExpertBlock(nn.Module):
    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.cn = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True, dropout=config.dropout)
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
        self.drop = nn.Dropout(config.dropout)
        self.mod = nn.Linear(h, 9 * h)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, memory: Tensor, time: Tensor, position: Tensor) -> Tensor:
        sa_s, sa_c, sa_g, ca_s, ca_c, ca_g, ff_s, ff_c, ff_g = self.mod(time).chunk(9, dim=-1)
        value = self.n1(x)
        qk = self.modulate(value, sa_s, sa_c) + position
        update, _ = self.self_attn(qk, qk, value, need_weights=False)
        x = x + torch.tanh(sa_g)[:, None] * self.drop(update)
        query = self.modulate(self.n2(x), ca_s, ca_c) + position
        norm_mem = self.cn(memory)
        update, _ = self.cross(query, norm_mem, norm_mem, need_weights=False)
        x = x + torch.tanh(ca_g)[:, None] * self.drop(update)
        ffn = self.ffn(self.modulate(self.n3(x), ff_s, ff_c))
        return x + torch.tanh(ff_g)[:, None] * self.drop(ffn)


class PlannerConditionedPhysicalActionExpert(nn.Module):
    """Planner-conditioned decoder that emits typed physical-action velocity."""

    def __init__(self, config: V362PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_physical_lift = PhysicalActionTokenLift(config)
        self.planner_action_proj = nn.Linear(h, h)
        self.proposal_proj = nn.Identity()
        self.role = HorizonRoleEmbedding(config)
        self.task_token = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.world_type = nn.Parameter(torch.randn(1, world_tokens, h) * 0.02)
        self.state_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.executed_type = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.proposal_type = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.planner_memory_type = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.register_buffer("action_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList([ActionExpertBlock(config) for _ in range(config.action_decoder_depth)])
        self.out = PhysicalVelocityHead(config)

    def memory(
        self,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        planner_tokens: Tensor,
    ) -> Tensor:
        world_tokens = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        task = self.task_token.expand(world.shape[0], -1, -1)
        planner = planner_tokens + self.planner_memory_type
        return torch.cat([task, world_tokens, state_token, executed, proposal, planner], dim=1)

    def forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        planner_tokens: Tensor,
        planner_action_tokens: Tensor,
    ) -> Tensor:
        batch = noisy_physical.shape[0]
        position = self.action_position.to(device=noisy_physical.device, dtype=noisy_physical.dtype)
        role = self.role(batch, device=noisy_physical.device, dtype=noisy_physical.dtype)
        x = self.noisy_physical_lift(noisy_physical) + self.planner_action_proj(planner_action_tokens) + role
        memory = self.memory(world, state, executed_history, proposal_tokens, proposal_keep, planner_tokens)
        t = self.time(time.to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, memory, t, position)
        return self.out(x)


class V362PolicySystem(nn.Module):
    def __init__(
        self,
        world_config: V35WorldConfig,
        policy_config: V362PolicyConfig,
        world_encoder: WorldEvidenceEncoder,
    ) -> None:
        super().__init__()
        self.world_config = world_config
        self.policy_config = policy_config
        self.world_encoder = world_encoder
        self.world_encoder.requires_grad_(False)
        self.world_encoder.eval()
        self.codec = PhysicalActionCodec(policy_config)
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = PolicyLatentDiTPlannerV362(policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens)
        self.decoder = PlannerConditionedPhysicalActionExpert(policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens)

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_encoder.eval()
        return self

    @torch.no_grad()
    def encode_world(self, visual: Tensor, state_history: Tensor, executed_history: Tensor) -> Tensor:
        return self.world_encoder(visual.float(), state_history.float(), executed_history.float())

    def _policy_forward(
        self,
        noisy_physical: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> dict[str, Tensor]:
        planner = self.planner(noisy_physical, time, world, state, executed_history, proposal_tokens, proposal_keep)
        pred_physical_velocity = self.decoder(
            noisy_physical,
            time,
            world,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            planner["planner_tokens"],
            planner["planner_action_tokens"],
        )
        planner["pred_physical_velocity"] = pred_physical_velocity
        return planner

    def flow_training_forward(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        target_action: Tensor,
        *,
        proposal_dropout: float | None = None,
    ) -> dict[str, Tensor]:
        world = self.encode_world(visual, state_history, executed_history)
        proposal = self.proposal(executed_history)
        target_physical = self.codec.encode(target_action, state)
        noise = self.codec.sample_noise(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        t = torch.rand(target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype)
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(target_physical.dtype)
        policy = self._policy_forward(noisy_physical, t, world, state, executed_history, proposal["tokens"].detach(), keep)
        clean_physical_estimate = noisy_physical - t[:, None, None] * policy["pred_physical_velocity"]
        decoded_action = self.codec.decode(clean_physical_estimate, state)
        return {
            "pred_physical_velocity": policy["pred_physical_velocity"],
            "target_physical_velocity": target_physical_velocity,
            "target_physical": target_physical,
            "clean_physical_estimate": clean_physical_estimate,
            "proposal_action": proposal["action"],
            "world": world,
            "time": t,
            "noisy_physical_action": noisy_physical,
            "pred_action_estimate": decoded_action,
            "event_logits": policy["event_logits"],
            "motion_logits": policy["motion_logits"],
        }

    @torch.no_grad()
    def sample(
        self,
        visual: Tensor,
        state_history: Tensor,
        executed_history: Tensor,
        state: Tensor,
        *,
        steps: int | None = None,
        noise: Tensor | None = None,
        use_proposal: bool = True,
        return_event_logits: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        world = self.encode_world(visual, state_history, executed_history)
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        if noise is None:
            x = self.codec.sample_noise(visual.shape[0], device=visual.device, dtype=visual.dtype)
        else:
            x = noise.clone()
            if x.shape[-1] == self.policy_config.action_dim:
                x = self.codec.encode(x.to(device=visual.device, dtype=visual.dtype), state.to(device=visual.device, dtype=visual.dtype))
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        last: dict[str, Tensor] | None = None
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            last = self._policy_forward(x, t, world, state, executed_history, proposal["tokens"], keep)
            x = x - last["pred_physical_velocity"] / float(steps)
        action = self.codec.decode(x, state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(x, zero_t, world, state, executed_history, proposal["tokens"], keep)
            return {"action": action, "physical_action": x, "event_logits": event["event_logits"], "motion_logits": event["motion_logits"]}
        return action

    def parameter_report(self) -> dict[str, int]:
        report = {
            "frozen_world_encoder": sum(p.numel() for p in self.world_encoder.parameters()),
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "physical_action_codec": sum(p.numel() for p in self.codec.parameters()),
            "latent_dit_planner": sum(p.numel() for p in self.planner.parameters()),
            "physical_action_expert_decoder": sum(p.numel() for p in self.decoder.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "ParsevalGripperTemporalFrame",
    "V362PolicyConfig",
    "PhysicalActionCodec",
    "PhysicalActionTokenLift",
    "PhysicalVelocityHead",
    "HorizonRoleEmbedding",
    "PolicyLatentDiTPlannerV362",
    "PlannerConditionedPhysicalActionExpert",
    "V362PolicySystem",
]
