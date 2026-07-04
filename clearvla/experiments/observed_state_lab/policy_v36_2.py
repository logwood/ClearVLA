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
        # arm_abs + arm_delta + expanded gripper field.  The first two gripper
        # coordinates are value/delta and are the only ones decoded to native
        # actions; the rest are auxiliary flow coordinates for gripper timing.
        return 2 * self.arm_dim + int(self.gripper_field_dim)


class PhysicalActionCodec(nn.Module):
    """Deterministic typed action codec for Alicia-D native actions.

    The codec is deliberately not a learned black-box.  It gives the flow a
    better coordinate system while keeping the final command contract native 7-D.
    """

    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

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
        grip_field = self._encode_gripper_field(grip, prev_grip, action_state)
        return torch.cat([arm, arm_delta, grip_field], dim=-1)

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
        return {
            "arm_abs": physical[..., :ad],
            "arm_delta": physical[..., ad : 2 * ad],
            "gripper_field": gripper_field,
            "gripper_value": gripper_field[..., :1],
            "gripper_delta": gripper_field[..., 1:2],
            "gripper_extra": gripper_field[..., 2:],
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
            grip_from_delta = state_grip[:, None] + torch.cumsum(parts["gripper_delta"], dim=1)
            arm = (1.0 - blend) * arm_abs + blend * arm_from_delta
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
        physical_delta = torch.cat([parts["arm_delta"], parts["gripper_delta"]], dim=-1)
        return torch.nn.functional.smooth_l1_loss(actual_delta, physical_delta)


class PhysicalActionTokenLift(nn.Module):
    """Typed lift from physical action coordinates to a horizon action token."""

    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ad = config.arm_dim
        self.config = config
        self.arm_abs = nn.Linear(ad, h)
        self.arm_delta = nn.Linear(ad, h)
        self.grip_value = nn.Linear(1, h)
        self.grip_delta = nn.Linear(1, h)
        self.grip_extra = nn.Linear(max(int(config.gripper_field_dim) - 2, 1), h)
        self.component_type = nn.Parameter(torch.randn(1, 5, h) * 0.02)
        self.mix = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h * 2), nn.SiLU(), nn.Linear(h * 2, h))

    def forward(self, physical: Tensor) -> Tensor:
        ad = self.config.arm_dim
        gf = int(self.config.gripper_field_dim)
        arm_abs = physical[..., :ad]
        arm_delta = physical[..., ad : 2 * ad]
        grip_field = physical[..., 2 * ad : 2 * ad + gf]
        grip_value = grip_field[..., :1]
        grip_delta = grip_field[..., 1:2]
        grip_extra = grip_field[..., 2:]
        if int(grip_extra.shape[-1]) == 0:
            grip_extra = torch.zeros(*grip_field.shape[:-1], 1, device=physical.device, dtype=physical.dtype)
        comp = self.component_type.to(device=physical.device, dtype=physical.dtype)
        x = (
            self.arm_abs(arm_abs) + comp[:, 0, None]
            + self.arm_delta(arm_delta) + comp[:, 1, None]
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
        self.arm_abs = nn.Linear(h, ad)
        self.arm_delta = nn.Linear(h, ad)
        self.grip_value = nn.Linear(h, 1)
        self.grip_delta = nn.Linear(h, 1)
        self.grip_extra = nn.Linear(h, max(int(config.gripper_field_dim) - 2, 0))

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.norm(tokens)
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
        noise = torch.randn_like(target_physical)
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
            x = torch.randn(
                visual.shape[0],
                self.policy_config.action_horizon,
                self.policy_config.physical_action_dim,
                device=visual.device,
                dtype=visual.dtype,
            )
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
    "V362PolicyConfig",
    "PhysicalActionCodec",
    "PhysicalActionTokenLift",
    "PhysicalVelocityHead",
    "HorizonRoleEmbedding",
    "PolicyLatentDiTPlannerV362",
    "PlannerConditionedPhysicalActionExpert",
    "V362PolicySystem",
]
