from __future__ import annotations

"""V36.1 policy-side latent DiT planner with an action-expert decoder.

V36.1 keeps the V35 observed-state world encoder frozen and upgrades only the
bottom policy.  The policy has two explicit stages:

1. A typed latent DiT planner jointly mixes world tokens, state/executed-action
   history, proposal tokens, horizon action queries, and gripper-event scratch
   tokens.
2. A V35-style planner-conditioned action expert decodes the noisy action flow
   through action-token self attention and cross attention into planner/world
   memory.

The continuous 7-D action flow remains unified.  A separate auxiliary
hold/open/close event head supervises sparse gripper transitions without
splitting the arm and gripper execution head.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .policy import RejectableHistoryProposal, TimeEmbedding
from .world_model import BiasFreeFFN, V35WorldConfig, WorldEvidenceEncoder, sinusoidal_positions


@dataclass(frozen=True)
class V36PolicyConfig:
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 24
    executed_history_length: int = 3
    hidden_size: int = 512
    num_heads: int = 8
    depth: int = 6  # typed latent DiT planner depth
    action_decoder_depth: int = 4  # V35-style action expert decoder depth
    proposal_depth: int = 2
    ffn_expansion: float = 4.0
    proposal_dropout: float = 0.25
    dropout: float = 0.05
    event_tokens: int = 3
    gripper_dim_index: int = -1
    inference_steps: int = 5

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
        ) <= 0:
            raise ValueError("V36 policy dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action/state dimensions must match")
        if not 0 <= self.proposal_dropout < 1:
            raise ValueError("proposal_dropout must be in [0,1)")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.action_dim + self.gripper_dim_index


class DiTPlannerBlock(nn.Module):
    """AdaLN-zero DiT block over a typed policy token sequence."""

    def __init__(self, config: V36PolicyConfig) -> None:
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


class PolicyLatentDiTPlanner(nn.Module):
    """Typed latent planner that prepares action/event context tokens.

    This module does not directly execute the continuous flow.  It produces
    refined action tokens and mixed planner memory that are consumed by the
    action expert decoder.
    """

    def __init__(self, config: V36PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_action_proj = nn.Linear(config.action_dim, h)
        self.proposal_proj = nn.Identity()
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
        self.event_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))  # hold/open/close

    def _tokens(
        self,
        noisy_action: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> tuple[Tensor, slice, slice]:
        batch = noisy_action.shape[0]
        hpos = self.horizon_position.to(device=noisy_action.device, dtype=noisy_action.dtype)
        task = self.task_token.expand(batch, -1, -1)
        world_tokens = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        horizon = self.horizon_query.expand(batch, -1, -1) + hpos + self.noisy_action_proj(noisy_action)
        event = self.event_query.expand(batch, -1, -1) + self.event_type
        prefix_len = 1 + world_tokens.shape[1] + 1 + executed.shape[1] + proposal.shape[1]
        action_slice = slice(prefix_len, prefix_len + self.config.action_horizon)
        event_start = prefix_len + self.config.action_horizon
        event_slice = slice(event_start, event_start + self.config.event_tokens)
        tokens = torch.cat([task, world_tokens, state_token, executed, proposal, horizon, event], dim=1)
        return tokens, action_slice, event_slice

    def forward(
        self,
        noisy_action: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_action.shape[0], device=noisy_action.device, dtype=noisy_action.dtype)
        tokens, action_slice, event_slice = self._tokens(
            noisy_action, world, state, executed_history, proposal_tokens, proposal_keep
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
        }


class ActionExpertBlock(nn.Module):
    """V35-style action expert block with planner/world cross attention."""

    def __init__(self, config: V36PolicyConfig) -> None:
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


class PlannerConditionedActionExpert(nn.Module):
    """Continuous unified 7-D action-flow decoder.

    The decoder restores the V35 action-expert contract: the 24 noisy action
    tokens are the execution backbone, updated by action self-attention and
    cross-attention into both raw world/history memory and the DiT planner's
    mixed typed memory.
    """

    def __init__(self, config: V36PolicyConfig, world_hidden: int, world_tokens: int) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.world_proj = nn.Identity() if world_hidden == h else nn.Linear(world_hidden, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.executed_proj = nn.Linear(config.action_dim, h)
        self.noisy_action_proj = nn.Linear(config.action_dim, h)
        self.planner_action_proj = nn.Linear(h, h)
        self.proposal_proj = nn.Identity()
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
        self.blocks = nn.ModuleList([ActionExpertBlock(config) for _ in range(config.action_decoder_depth)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.action_dim))

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
        noisy_action: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
        planner_tokens: Tensor,
        planner_action_tokens: Tensor,
    ) -> Tensor:
        x = self.noisy_action_proj(noisy_action) + self.planner_action_proj(planner_action_tokens)
        position = self.action_position.to(device=x.device, dtype=x.dtype)
        memory = self.memory(world, state, executed_history, proposal_tokens, proposal_keep, planner_tokens)
        t = self.time(time.to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, memory, t, position)
        return self.out(x)


class V36PolicySystem(nn.Module):
    def __init__(
        self,
        world_config: V35WorldConfig,
        policy_config: V36PolicyConfig,
        world_encoder: WorldEvidenceEncoder,
    ) -> None:
        super().__init__()
        self.world_config = world_config
        self.policy_config = policy_config
        self.world_encoder = world_encoder
        self.world_encoder.requires_grad_(False)
        self.world_encoder.eval()
        self.proposal = RejectableHistoryProposal(policy_config)
        self.planner = PolicyLatentDiTPlanner(
            policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens
        )
        self.decoder = PlannerConditionedActionExpert(
            policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_encoder.eval()
        return self

    @torch.no_grad()
    def encode_world(self, visual: Tensor, state_history: Tensor, executed_history: Tensor) -> Tensor:
        return self.world_encoder(visual.float(), state_history.float(), executed_history.float())

    def _policy_forward(
        self,
        noisy_action: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> dict[str, Tensor]:
        planner = self.planner(noisy_action, time, world, state, executed_history, proposal_tokens, proposal_keep)
        pred_velocity = self.decoder(
            noisy_action,
            time,
            world,
            state,
            executed_history,
            proposal_tokens,
            proposal_keep,
            planner["planner_tokens"],
            planner["planner_action_tokens"],
        )
        planner["pred_velocity"] = pred_velocity
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
        noise = torch.randn_like(target_action)
        t = torch.rand(target_action.shape[0], device=target_action.device, dtype=target_action.dtype)
        noisy = (1 - t[:, None, None]) * target_action + t[:, None, None] * noise
        target_velocity = noise - target_action
        drop = self.policy_config.proposal_dropout if proposal_dropout is None else float(proposal_dropout)
        keep = (torch.rand(target_action.shape[0], device=target_action.device) >= drop).to(target_action.dtype)
        policy = self._policy_forward(noisy, t, world, state, executed_history, proposal["tokens"].detach(), keep)
        clean_estimate = noisy - t[:, None, None] * policy["pred_velocity"]
        return {
            "pred_velocity": policy["pred_velocity"],
            "target_velocity": target_velocity,
            "proposal_action": proposal["action"],
            "world": world,
            "time": t,
            "noisy_action": noisy,
            "pred_action_estimate": clean_estimate,
            "event_logits": policy["event_logits"],
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
        x = torch.randn(
            visual.shape[0], self.policy_config.action_horizon, self.policy_config.action_dim,
            device=visual.device, dtype=visual.dtype,
        ) if noise is None else noise.clone()
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        last: dict[str, Tensor] | None = None
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            last = self._policy_forward(x, t, world, state, executed_history, proposal["tokens"], keep)
            x = x - last["pred_velocity"] / float(steps)
        if return_event_logits:
            zero_t = torch.zeros((visual.shape[0],), device=visual.device, dtype=visual.dtype)
            event = self._policy_forward(x, zero_t, world, state, executed_history, proposal["tokens"], keep)["event_logits"]
            return {"action": x, "event_logits": event}
        return x

    def parameter_report(self) -> dict[str, int]:
        report = {
            "frozen_world_encoder": sum(p.numel() for p in self.world_encoder.parameters()),
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "latent_dit_planner": sum(p.numel() for p in self.planner.parameters()),
            "action_expert_decoder": sum(p.numel() for p in self.decoder.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V36PolicyConfig",
    "DiTPlannerBlock",
    "PolicyLatentDiTPlanner",
    "ActionExpertBlock",
    "PlannerConditionedActionExpert",
    "V36PolicySystem",
]
