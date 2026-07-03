from __future__ import annotations

"""V35 unified full-depth action expert.

One shared 7-D flow-matching expert predicts the complete 24-step action chunk.
The world state is supplied by the frozen observed-state encoder. Executed
history produces a detachable proposal token sequence, never a forced residual
or a second execution head.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .world_model import BiasFreeFFN, V35WorldConfig, WorldEvidenceEncoder, sinusoidal_positions


@dataclass(frozen=True)
class V35PolicyConfig:
    action_dim: int = 7
    state_dim: int = 7
    action_horizon: int = 24
    executed_history_length: int = 3
    hidden_size: int = 320
    num_heads: int = 8
    depth: int = 8
    proposal_depth: int = 2
    ffn_expansion: float = 4.0
    proposal_dropout: float = 0.5
    gripper_dim_index: int = -1
    inference_steps: int = 5

    def validate(self) -> None:
        if min(
            self.action_dim, self.state_dim, self.action_horizon, self.executed_history_length,
            self.hidden_size, self.num_heads, self.depth, self.proposal_depth, self.inference_steps,
        ) <= 0:
            raise ValueError("V35 policy dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.action_dim != self.state_dim:
            raise ValueError("action/state dimensions must match")
        if not 0 <= self.proposal_dropout < 1:
            raise ValueError("proposal_dropout must be in [0,1)")

    @property
    def gripper_index(self) -> int:
        return self.gripper_dim_index if self.gripper_dim_index >= 0 else self.action_dim + self.gripper_dim_index


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden))

    def forward(self, t: Tensor) -> Tensor:
        half = self.hidden // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        phase = t[:, None] * freq[None]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.hidden:
            emb = F.pad(emb, (0, self.hidden - emb.shape[-1]))
        return self.net(emb)


class ProposalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden)
        self.mn = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.n2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, int(hidden * expansion)), nn.GELU(), nn.Linear(int(hidden * expansion), hidden)
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        update, _ = self.cross(self.qn(query), self.mn(memory), self.mn(memory), need_weights=False)
        query = query + update
        return query + self.ffn(self.n2(query))


class RejectableHistoryProposal(nn.Module):
    def __init__(self, config: V35PolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.history_proj = nn.Linear(config.action_dim, h)
        self.history_key = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        self.future_query = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.blocks = nn.ModuleList(
            [ProposalBlock(h, config.num_heads, config.ffn_expansion) for _ in range(config.proposal_depth)]
        )
        self.action_head = nn.Linear(h, config.action_dim)

    def forward(self, executed_history: Tensor) -> dict[str, Tensor]:
        memory = self.history_proj(executed_history) + self.history_key
        tokens = self.future_query.expand(executed_history.shape[0], -1, -1)
        for block in self.blocks:
            tokens = block(tokens, memory)
        return {"tokens": tokens, "action": self.action_head(tokens)}


class ExpertBlock(nn.Module):
    def __init__(self, config: V35PolicyConfig) -> None:
        super().__init__()
        h = config.hidden_size
        self.n1 = nn.LayerNorm(h, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.n2 = nn.LayerNorm(h, elementwise_affine=False)
        self.cn = nn.LayerNorm(h)
        self.cross = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.n3 = nn.LayerNorm(h, elementwise_affine=False)
        self.ffn = BiasFreeFFN(h, config.ffn_expansion)
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
        x = x + torch.tanh(sa_g)[:, None] * update
        query = self.modulate(self.n2(x), ca_s, ca_c) + position
        update, _ = self.cross(query, self.cn(memory), self.cn(memory), need_weights=False)
        x = x + torch.tanh(ca_g)[:, None] * update
        ffn = self.ffn(self.modulate(self.n3(x), ff_s, ff_c))
        return x + torch.tanh(ff_g)[:, None] * ffn


class UnifiedLatentActionExpert(nn.Module):
    def __init__(self, config: V35PolicyConfig, world_hidden: int, world_tokens: int) -> None:
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
        self.register_buffer("action_position", sinusoidal_positions(range(1, config.action_horizon + 1), h)[None], persistent=True)
        self.time = TimeEmbedding(h)
        self.blocks = nn.ModuleList([ExpertBlock(config) for _ in range(config.depth)])
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.action_dim))

    def memory(
        self,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor,
    ) -> Tensor:
        world = self.world_proj(world) + self.world_type
        state_token = self.state_proj(state)[:, None] + self.state_type
        executed = self.executed_proj(executed_history) + self.executed_type
        proposal = self.proposal_proj(proposal_tokens) * proposal_keep[:, None, None] + self.proposal_type
        task = self.task_token.expand(world.shape[0], -1, -1)
        return torch.cat([task, world, state_token, executed, proposal], dim=1)

    def forward(
        self,
        noisy_action: Tensor,
        time: Tensor,
        world: Tensor,
        state: Tensor,
        executed_history: Tensor,
        proposal_tokens: Tensor,
        proposal_keep: Tensor | None = None,
    ) -> Tensor:
        if proposal_keep is None:
            proposal_keep = torch.ones(noisy_action.shape[0], device=noisy_action.device, dtype=noisy_action.dtype)
        x = self.noisy_action_proj(noisy_action)
        position = self.action_position.to(device=x.device, dtype=x.dtype)
        memory = self.memory(world, state, executed_history, proposal_tokens, proposal_keep)
        t = self.time(time.to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x, memory, t, position)
        return self.out(x)


class V35PolicySystem(nn.Module):
    def __init__(
        self,
        world_config: V35WorldConfig,
        policy_config: V35PolicyConfig,
        world_encoder: WorldEvidenceEncoder,
    ) -> None:
        super().__init__()
        self.world_config = world_config
        self.policy_config = policy_config
        self.world_encoder = world_encoder
        self.world_encoder.requires_grad_(False)
        self.world_encoder.eval()
        self.proposal = RejectableHistoryProposal(policy_config)
        self.expert = UnifiedLatentActionExpert(
            policy_config, world_hidden=world_config.hidden_size, world_tokens=world_config.world_tokens
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.world_encoder.eval()
        return self

    @torch.no_grad()
    def encode_world(self, visual: Tensor, state_history: Tensor, executed_history: Tensor) -> Tensor:
        return self.world_encoder(visual.float(), state_history.float(), executed_history.float())

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
        predicted = self.expert(
            noisy, t, world, state, executed_history, proposal["tokens"].detach(), keep
        )
        return {
            "pred_velocity": predicted,
            "target_velocity": target_velocity,
            "proposal_action": proposal["action"],
            "world": world,
            "time": t,
            "noisy_action": noisy,
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
    ) -> Tensor:
        world = self.encode_world(visual, state_history, executed_history)
        proposal = self.proposal(executed_history)
        steps = int(steps or self.policy_config.inference_steps)
        x = torch.randn(
            visual.shape[0], self.policy_config.action_horizon, self.policy_config.action_dim,
            device=visual.device, dtype=visual.dtype,
        ) if noise is None else noise.clone()
        keep = torch.full((visual.shape[0],), 1.0 if use_proposal else 0.0, device=visual.device, dtype=visual.dtype)
        for index in range(steps, 0, -1):
            t = torch.full((visual.shape[0],), float(index) / float(steps), device=visual.device, dtype=visual.dtype)
            velocity = self.expert(x, t, world, state, executed_history, proposal["tokens"], keep)
            x = x - velocity / float(steps)
        return x

    def parameter_report(self) -> dict[str, int]:
        report = {
            "frozen_world_encoder": sum(p.numel() for p in self.world_encoder.parameters()),
            "history_proposal": sum(p.numel() for p in self.proposal.parameters()),
            "unified_action_expert": sum(p.numel() for p in self.expert.parameters()),
        }
        report["total"] = sum(p.numel() for p in self.parameters())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return report


__all__ = [
    "V35PolicyConfig", "RejectableHistoryProposal", "UnifiedLatentActionExpert", "V35PolicySystem",
]
