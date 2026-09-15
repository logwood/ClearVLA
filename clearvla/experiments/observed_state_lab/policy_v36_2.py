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

import torch
from torch import Tensor, nn

from clearvla.policy.codec import (
    ParsevalGripperTemporalFrame,
    PhysicalActionCodec,
    PhysicalActionTokenLift,
    PhysicalVelocityHead,
)
from clearvla.policy.config import V362PolicyConfig
from clearvla.policy.trunk_primitives import HorizonRoleEmbedding

from .policy import RejectableHistoryProposal, TimeEmbedding
from .world_model import BiasFreeFFN, V35WorldConfig, WorldEvidenceEncoder, sinusoidal_positions


class DiTPlannerBlock(nn.Module):
    def __init__(self, config: V362PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
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
        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = self.mod(time).chunk(
            6, dim=-1
        )
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
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
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
        proposal = (
            self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        )
        role = self.role(batch, device=noisy_physical.device, dtype=noisy_physical.dtype)
        horizon = (
            self.horizon_query.expand(batch, -1, -1)
            + hpos
            + role
            + self.noisy_physical_lift(noisy_physical)
        )
        event = self.event_query.expand(batch, -1, -1) + self.event_type
        prefix_len = 1 + world_tokens.shape[1] + 1 + executed.shape[1] + proposal.shape[1]
        action_slice = slice(prefix_len, prefix_len + self.config.action_horizon)
        event_start = prefix_len + self.config.action_horizon
        event_slice = slice(event_start, event_start + self.config.event_tokens)
        tokens = torch.cat(
            [task, world_tokens, state_token, executed, proposal, horizon, event], dim=1
        )
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
            proposal_keep = torch.ones(
                noisy_physical.shape[0], device=noisy_physical.device, dtype=noisy_physical.dtype
            )
        tokens, action_slice, event_slice = self._tokens(
            noisy_physical, world, state, executed_history, proposal_tokens, proposal_keep
        )
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
        self.self_attn = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.cn = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(
            h, config.num_heads, batch_first=True, dropout=config.dropout
        )
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
        self.register_buffer(
            "action_position",
            sinusoidal_positions(range(1, config.action_horizon + 1), h)[None],
            persistent=True,
        )
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList(
            [ActionExpertBlock(config) for _ in range(config.action_decoder_depth)]
        )
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
        proposal = (
            self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        )
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
        x = (
            self.noisy_physical_lift(noisy_physical)
            + self.planner_action_proj(planner_action_tokens)
            + role
        )
        memory = self.memory(
            world, state, executed_history, proposal_tokens, proposal_keep, planner_tokens
        )
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
        self.planner = PolicyLatentDiTPlannerV362(
            policy_config,
            world_hidden=world_config.hidden_size,
            world_tokens=world_config.world_tokens,
        )
        self.decoder = PlannerConditionedPhysicalActionExpert(
            policy_config,
            world_hidden=world_config.hidden_size,
            world_tokens=world_config.world_tokens,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_encoder.eval()
        return self

    @torch.no_grad()
    def encode_world(
        self, visual: Tensor, state_history: Tensor, executed_history: Tensor
    ) -> Tensor:
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
        planner = self.planner(
            noisy_physical, time, world, state, executed_history, proposal_tokens, proposal_keep
        )
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
        noise = self.codec.sample_noise(
            target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype
        )
        t = torch.rand(
            target_physical.shape[0], device=target_physical.device, dtype=target_physical.dtype
        )
        noisy_physical = (1 - t[:, None, None]) * target_physical + t[:, None, None] * noise
        target_physical_velocity = noise - target_physical
        drop = (
            self.policy_config.proposal_dropout
            if proposal_dropout is None
            else float(proposal_dropout)
        )
        keep = (torch.rand(target_physical.shape[0], device=target_physical.device) >= drop).to(
            target_physical.dtype
        )
        policy = self._policy_forward(
            noisy_physical, t, world, state, executed_history, proposal["tokens"].detach(), keep
        )
        clean_physical_estimate = (
            noisy_physical - t[:, None, None] * policy["pred_physical_velocity"]
        )
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
                x = self.codec.encode(
                    x.to(device=visual.device, dtype=visual.dtype),
                    state.to(device=visual.device, dtype=visual.dtype),
                )
            elif x.shape[-1] != self.policy_config.physical_action_dim:
                raise ValueError("noise must have last dim action_dim or physical_action_dim")
        keep = torch.full(
            (visual.shape[0],),
            1.0 if use_proposal else 0.0,
            device=visual.device,
            dtype=visual.dtype,
        )
        last: dict[str, Tensor] | None = None
        for index in range(steps, 0, -1):
            t = torch.full(
                (visual.shape[0],),
                float(index) / float(steps),
                device=visual.device,
                dtype=visual.dtype,
            )
            last = self._policy_forward(
                x, t, world, state, executed_history, proposal["tokens"], keep
            )
            x = x - last["pred_physical_velocity"] / float(steps)
        action = self.codec.decode(x, state)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(
                x, zero_t, world, state, executed_history, proposal["tokens"], keep
            )
            return {
                "action": action,
                "physical_action": x,
                "event_logits": event["event_logits"],
                "motion_logits": event["motion_logits"],
            }
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
