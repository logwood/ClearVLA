from __future__ import annotations

"""Rejectable executed-history proposal used by current policy systems."""

from typing import Protocol

import torch
from torch import Tensor, nn


class ProposalConfig(Protocol):
    action_dim: int
    executed_history_length: int
    action_horizon: int
    hidden_size: int
    num_heads: int
    ffn_expansion: float
    proposal_depth: int


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
    def __init__(self, config: ProposalConfig) -> None:
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
