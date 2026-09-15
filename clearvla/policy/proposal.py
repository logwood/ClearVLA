"""Rejectable executed-history proposal used by current policy systems."""

from __future__ import annotations

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
    action_history_enabled: int
    action_history_recent_tokens: int
    action_history_summary_tokens: int
    executed_action_offsets: tuple[int, ...]


class ProposalBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.qn = nn.LayerNorm(hidden)
        self.mn = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.n2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, int(hidden * expansion)),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden),
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
        self.history_enabled = bool(int(getattr(config, "action_history_enabled", 0)))
        self.history_length = int(config.executed_history_length)
        self.recent_tokens = min(
            int(getattr(config, "action_history_recent_tokens", self.history_length)),
            self.history_length,
        )
        self.summary_tokens = (
            int(getattr(config, "action_history_summary_tokens", 0))
            if self.history_enabled and self.history_length > self.recent_tokens
            else 0
        )
        self.history_proj = nn.Linear(config.action_dim, h)
        self.history_key = nn.Parameter(torch.randn(1, config.executed_history_length, h) * 0.02)
        if self.history_enabled:
            self.history_key.requires_grad_(False)
        self.history_delta_proj = (
            nn.Linear(config.action_dim, h, bias=False) if self.history_enabled else None
        )
        self.history_time = (
            nn.Sequential(
                nn.Linear(h, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
            if self.history_enabled
            else None
        )
        self.history_summary_query = (
            nn.Parameter(torch.randn(1, self.summary_tokens, h) * 0.02)
            if self.summary_tokens > 0
            else None
        )
        self.history_summary_qn = nn.LayerNorm(h) if self.summary_tokens > 0 else None
        self.history_summary_mn = nn.LayerNorm(h) if self.summary_tokens > 0 else None
        self.history_summary_cross = (
            nn.MultiheadAttention(h, config.num_heads, batch_first=True)
            if self.summary_tokens > 0
            else None
        )
        self.history_summary_ffn = (
            nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, int(h * config.ffn_expansion)),
                nn.GELU(),
                nn.Linear(int(h * config.ffn_expansion), h),
            )
            if self.summary_tokens > 0
            else None
        )
        offsets = tuple(
            int(value)
            for value in getattr(
                config,
                "effective_executed_action_offsets",
                tuple(range(-self.history_length, 0)),
            )
        )
        if len(offsets) != self.history_length:
            raise ValueError("executed_action_offsets must match executed_history_length")
        self.register_buffer(
            "history_time_encoding",
            self._sinusoidal_offsets(offsets, h)[None],
            persistent=False,
        )
        self.future_query = nn.Parameter(torch.randn(1, config.action_horizon, h) * 0.02)
        self.blocks = nn.ModuleList(
            [
                ProposalBlock(h, config.num_heads, config.ffn_expansion)
                for _ in range(config.proposal_depth)
            ]
        )
        self.action_head = nn.Linear(h, config.action_dim)

    @staticmethod
    def _sinusoidal_offsets(offsets: tuple[int, ...], dim: int) -> Tensor:
        values = torch.as_tensor(offsets, dtype=torch.float32)
        scale = values.abs().max().clamp_min(1.0)
        values = values / scale
        half = max(dim // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(max(half - 1, 1))
        frequency = torch.exp(-torch.log(torch.tensor(10_000.0)) * exponent)
        angle = values[:, None] * frequency[None] * (2.0 * torch.pi)
        encoding = torch.cat([angle.sin(), angle.cos()], dim=-1)
        if encoding.shape[-1] < dim:
            encoding = torch.nn.functional.pad(encoding, (0, dim - encoding.shape[-1]))
        return encoding[:, :dim]

    @property
    def history_token_count(self) -> int:
        if not self.history_enabled:
            return self.history_length
        return self.recent_tokens + self.summary_tokens

    def encode_history(self, executed_history: Tensor) -> Tensor:
        if executed_history.ndim != 3 or int(executed_history.shape[1]) != self.history_length:
            raise ValueError(
                "executed_history must be [B,executed_history_length,action_dim]"
            )
        projected = self.history_proj(executed_history)
        if not self.history_enabled:
            return projected + self.history_key.to(
                device=projected.device, dtype=projected.dtype
            )
        if self.history_delta_proj is None or self.history_time is None:
            raise RuntimeError("time-aware action history modules are missing")
        delta = torch.cat(
            [torch.zeros_like(executed_history[:, :1]), executed_history[:, 1:] - executed_history[:, :-1]],
            dim=1,
        )
        time = self.history_time(
            self.history_time_encoding.to(device=projected.device, dtype=projected.dtype)
        )
        memory = projected + self.history_delta_proj(delta) + time
        recent = memory[:, -self.recent_tokens :]
        if self.summary_tokens <= 0:
            return recent
        if any(
            module is None
            for module in (
                self.history_summary_query,
                self.history_summary_qn,
                self.history_summary_mn,
                self.history_summary_cross,
                self.history_summary_ffn,
            )
        ):
            raise RuntimeError("action-history summary modules are missing")
        older = memory[:, : -self.recent_tokens]
        query = self.history_summary_query.expand(memory.shape[0], -1, -1).to(
            device=memory.device, dtype=memory.dtype
        )
        normalized_memory = self.history_summary_mn(older)
        update, _ = self.history_summary_cross(
            self.history_summary_qn(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        summary = query + update
        summary = summary + self.history_summary_ffn(summary)
        return torch.cat([summary, recent], dim=1)

    def forward(self, executed_history: Tensor) -> dict[str, Tensor]:
        memory = self.encode_history(executed_history)
        tokens = self.future_query.expand(executed_history.shape[0], -1, -1)
        for block in self.blocks:
            tokens = block(tokens, memory)
        return {
            "tokens": tokens,
            "action": self.action_head(tokens),
            "history_tokens": memory,
        }
