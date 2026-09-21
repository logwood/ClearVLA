"""S history encoding with separate state/control events on physical time.

No pseudo-synchronous pairing, interpolation, duplicate current state, or
learned task clock is introduced. Only causal event times are model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..temporal import HistoryTiming
from .routing import smooth_rms_contract


def physical_time_encoding(offsets: Tensor, width: int) -> Tensor:
    """Fixed Fourier coordinates in physical control steps (not row indices)."""
    half = max((int(width) + 1) // 2, 1)
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=offsets.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    phase = offsets.float()[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), dim=-1)[..., :width]


@dataclass(frozen=True)
class TimedHistoryEncoding:
    tokens: Tensor
    valid: Tensor
    offsets: Tensor
    kinds: Tensor  # 0=state, 1=executed command
    state_rates: Tensor  # normalized state change per physical control step
    rate_valid: Tensor  # both observed endpoints are present


class _TimedBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, 2 * hidden, bias=False),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden, bias=False),
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        batch, length = valid.shape
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        allowed = causal[None] & valid[:, None, :]
        # A padded query has no evidence role. Give it its own zero key solely
        # to avoid all-masked softmax NaNs; discard it after each residual.
        own_zero = torch.eye(length, device=tokens.device, dtype=torch.bool)[None]
        allowed = torch.where(valid[:, :, None], allowed, own_zero)
        mask = (
            (~allowed)[:, None]
            .expand(-1, self.heads, -1, -1)
            .reshape(batch * self.heads, length, length)
        )
        safe = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        normalized = self.norm(safe)
        update, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        update, _ = smooth_rms_contract(update, 0.35)
        value = torch.where(valid[..., None], safe + update, torch.zeros_like(safe))
        residual, _ = smooth_rms_contract(self.ffn(value), 0.35)
        return torch.where(valid[..., None], value + residual, torch.zeros_like(value))


class TimedHistoryEncoder(nn.Module):
    """Encode actual state samples and executed commands without pairing them."""

    def __init__(self, *, state_dim: int, action_dim: int, hidden: int, heads: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.state_input = nn.Linear(2 * state_dim + 1, hidden, bias=False)
        self.action_input = nn.Linear(action_dim, hidden, bias=False)
        self.kind_embedding = nn.Parameter(torch.randn(2, hidden) * 0.02)
        self.blocks = nn.ModuleList(_TimedBlock(hidden, heads) for _ in range(2))

    def forward(
        self,
        state_history: Tensor,
        state: Tensor,
        executed_history: Tensor,
        timing: HistoryTiming,
    ) -> TimedHistoryEncoding:
        if state_history.ndim != 3 or executed_history.ndim != 3 or state.ndim != 2:
            raise ValueError("timed history requires [B,H,S], [B,S], [B,A,D]")
        batch, states, state_dim = state_history.shape
        if tuple(state.shape) != (batch, state_dim) or executed_history.shape[0] != batch:
            raise ValueError("timed history state and control batch axes differ")
        actions = int(executed_history.shape[1])
        timing.validate(batch=batch, states=states, actions=actions, device=state.device)
        # The separately observed current state owns the final row, not a
        # possibly stale caller copy. Do not append it a second time.
        source = torch.cat((state_history[:, :-1], state[:, None]), dim=1)
        source = torch.where(timing.state_observed[..., None], source, torch.zeros_like(source))
        action = torch.where(
            timing.action_executed[..., None], executed_history, torch.zeros_like(executed_history)
        )
        dt = timing.state_offsets[:, 1:] - timing.state_offsets[:, :-1]
        pair_valid = timing.state_observed[:, 1:] & timing.state_observed[:, :-1] & (dt > 0)
        rate = (source[:, 1:] - source[:, :-1]) / dt.clamp_min(1)[..., None].to(source.dtype)
        rate = torch.where(pair_valid[..., None], rate, torch.zeros_like(rate))
        rates = torch.cat((torch.zeros_like(source[:, :1]), rate), dim=1)
        rate_valid = F.pad(pair_valid, (1, 0), value=False)
        elapsed = F.pad(torch.where(pair_valid, dt, torch.zeros_like(dt)), (1, 0))
        state_token = self.state_input(
            torch.cat((source, rates, elapsed[..., None].to(source.dtype)), dim=-1)
        )
        action_token = self.action_input(action)
        offsets = torch.cat((timing.state_offsets, timing.action_offsets), dim=1)
        kinds = torch.cat(
            (torch.zeros_like(timing.state_offsets), torch.ones_like(timing.action_offsets)), dim=1
        )
        valid = torch.cat((timing.state_observed, timing.action_executed), dim=1)
        tokens = torch.cat((state_token, action_token), dim=1)
        tokens = (
            tokens
            + physical_time_encoding(offsets, self.hidden).to(tokens.dtype)
            + self.kind_embedding[kinds].to(tokens.dtype)
        )
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        # State o[t] precedes command a[t] at equal integer times.
        # State rows were concatenated first, so a stable time sort supplies
        # the tie-break without multiplying int64 source times.
        order = torch.argsort(offsets, dim=1, stable=True)
        tokens = tokens.gather(1, order[..., None].expand(-1, -1, self.hidden))
        valid = valid.gather(1, order)
        offsets = offsets.gather(1, order)
        kinds = kinds.gather(1, order)
        all_rates = torch.cat((rates, source.new_zeros(batch, actions, state_dim)), dim=1)
        all_rates = all_rates.gather(1, order[..., None].expand(-1, -1, state_dim))
        all_rate_valid = torch.cat(
            (rate_valid, torch.zeros_like(timing.action_executed)), dim=1
        ).gather(1, order)
        for block in self.blocks:
            tokens = block(tokens, valid)
        return TimedHistoryEncoding(tokens, valid, offsets, kinds, all_rates, all_rate_valid)
