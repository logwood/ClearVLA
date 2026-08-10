"""Causal executed-history proposal preserved from the formal V122 graph."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .types import HistoryActionProposalState


class _ProposalBlock(nn.Module):
    """One query-to-history block; mathematics matches the active proposal."""

    def __init__(self, hidden: int, heads: int, expansion: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(hidden * expansion)),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden),
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.cross(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(query)


class HistoryActionProposal(nn.Module):
    """Eight-row action history -> clean 24-step proposal.

    The 4 recent + 3 summary layout and two proposal blocks are the fixed
    formal-launcher path.  The module is deliberately outside S: S may read
    the complete observable history, while this proposal retains its original
    action-trajectory algorithm and its own supervised head.
    """

    OFFSETS = (-24, -16, -12, -8, -6, -4, -2, -1)

    def __init__(
        self,
        *,
        action_dim: int,
        hidden: int,
        heads: int,
        horizon: int,
        history_length: int,
        recent_tokens: int = 4,
        summary_tokens: int = 3,
        depth: int = 2,
        expansion: float = 4.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.history_length = int(history_length)
        self.recent_tokens = int(recent_tokens)
        self.summary_tokens = int(summary_tokens)
        if self.history_length != len(self.OFFSETS):
            raise ValueError("formal history proposal requires eight action offsets")
        if self.recent_tokens != 4 or self.summary_tokens != 3 or int(depth) != 2:
            raise ValueError("formal history proposal requires 4 recent + 3 summary and depth 2")

        self.history_projection = nn.Linear(action_dim, hidden)
        self.history_delta_projection = nn.Linear(action_dim, hidden, bias=False)
        # Retained for checkpoint/accounting parity.  Time-aware history owns
        # the deterministic sinusoid below, so this old positional key is
        # intentionally frozen exactly as in the formal graph.
        self.history_key = nn.Parameter(
            torch.randn(1, self.history_length, hidden) * 0.02,
            requires_grad=False,
        )
        self.history_time = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.summary_query = nn.Parameter(
            torch.randn(1, self.summary_tokens, hidden) * 0.02
        )
        self.summary_query_norm = nn.LayerNorm(hidden)
        self.summary_memory_norm = nn.LayerNorm(hidden)
        self.summary_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.summary_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(hidden * expansion)),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden),
        )
        self.future_query = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            _ProposalBlock(hidden, heads, expansion) for _ in range(int(depth))
        )
        self.action_head = nn.Linear(hidden, action_dim)
        self.register_buffer(
            "history_time_encoding",
            self._sinusoidal_offsets(self.OFFSETS, hidden)[None],
            persistent=True,
        )

    @property
    def encoded_history_tokens(self) -> int:
        return self.summary_tokens + self.recent_tokens

    @staticmethod
    def _sinusoidal_offsets(offsets: tuple[int, ...], width: int) -> Tensor:
        values = torch.as_tensor(offsets, dtype=torch.float32)
        values = values / values.abs().max().clamp_min(1.0)
        half = max(int(width) // 2, 1)
        exponent = torch.arange(half, dtype=torch.float32) / float(max(half - 1, 1))
        frequency = torch.exp(-math.log(10_000.0) * exponent)
        angle = values[:, None] * frequency[None] * (2.0 * torch.pi)
        encoding = torch.cat((angle.sin(), angle.cos()), dim=-1)
        if int(encoding.shape[-1]) < int(width):
            encoding = torch.nn.functional.pad(
                encoding,
                (0, int(width) - int(encoding.shape[-1])),
            )
        return encoding[:, : int(width)]

    def encode_history(self, executed_history: Tensor) -> Tensor:
        expected = (int(executed_history.shape[0]), self.history_length, self.action_dim)
        if tuple(executed_history.shape) != expected:
            raise ValueError(
                f"executed history must be {expected}, got {tuple(executed_history.shape)}"
            )
        projected = self.history_projection(executed_history)
        delta = torch.cat(
            (
                torch.zeros_like(executed_history[:, :1]),
                executed_history[:, 1:] - executed_history[:, :-1],
            ),
            dim=1,
        )
        time = self.history_time(
            self.history_time_encoding.to(
                device=projected.device,
                dtype=projected.dtype,
            )
        )
        memory = projected + self.history_delta_projection(delta) + time
        older = memory[:, : -self.recent_tokens]
        recent = memory[:, -self.recent_tokens :]
        query = self.summary_query.expand(int(memory.shape[0]), -1, -1).to(
            device=memory.device,
            dtype=memory.dtype,
        )
        normalized_memory = self.summary_memory_norm(older)
        summary_update, _ = self.summary_cross(
            self.summary_query_norm(query),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        summary = query + summary_update
        summary = summary + self.summary_ffn(summary)
        return torch.cat((summary, recent), dim=1)

    def forward(self, executed_history: Tensor) -> HistoryActionProposalState:
        memory = self.encode_history(executed_history)
        tokens = self.future_query.expand(int(executed_history.shape[0]), -1, -1).to(
            device=executed_history.device,
            dtype=executed_history.dtype,
        )
        for block in self.blocks:
            tokens = block(tokens, memory)
        state = HistoryActionProposalState(
            tokens=tokens,
            action_prediction=self.action_head(tokens),
            history_tokens=memory,
        )
        state.validate(
            horizon=self.horizon,
            hidden=self.hidden,
            action_dim=self.action_dim,
            history_tokens=self.encoded_history_tokens,
        )
        return state


__all__ = ["HistoryActionProposal"]
