"""Typed homes for the four active role blocks from the formal V122 graph.

The historical blocks mixed many canvas regions in one class.  Only three
grounding hosts and the first policy host were trainable and executed for the
active capability.  This module preserves their transformer mathematics while
making the legal operands explicit.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from .routing import smooth_rms_contract
from .types import LocalFactSet


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class _GroundingRoleHostBlock(nn.Module):
    """Self/visual/current-state/AdaLN/FFN grounding host."""

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        expansion: float,
        dropout: float,
        maximum_update_rms: float = 0.50,
    ) -> None:
        super().__init__()
        self.maximum_update_rms = float(maximum_update_rms)
        self.self_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.visual_norm = nn.LayerNorm(hidden)
        self.visual_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.visual_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transition_norm = nn.LayerNorm(hidden)
        self.transition_query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.transition_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, int(hidden * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden, bias=False),
        )
        self.modulation = nn.Linear(hidden, 12 * hidden)
        self.role_identity = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.modulation.weight, mean=0.0, std=3e-3)
        nn.init.zeros_(self.modulation.bias)
        with torch.no_grad():
            for index in (2, 5, 8, 11):
                self.modulation.bias[index * hidden : (index + 1) * hidden].fill_(-2.0)

    def _write(self, value: Tensor, proposed: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
        bounded, _ = smooth_rms_contract(
            self.dropout(proposed) * torch.sigmoid(gate)[:, None],
            self.maximum_update_rms,
        )
        return value + bounded, bounded

    def forward(
        self,
        carrier: Tensor,
        *,
        visual_memory: Tensor,
        state_memory: Tensor,
        condition: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        chunks = self.modulation(
            condition
            + self.role_identity.to(device=condition.device, dtype=condition.dtype)
        ).chunk(12, dim=-1)
        sa_shift, sa_scale, sa_gate = chunks[0:3]
        visual_shift, visual_scale, visual_gate = chunks[3:6]
        transition_shift, transition_scale, transition_gate = chunks[6:9]
        ffn_shift, ffn_scale, ffn_gate = chunks[9:12]

        normalized = _modulate(self.self_norm(carrier), sa_shift, sa_scale)
        proposed, _ = self.self_attention(
            normalized,
            normalized,
            self.self_norm(carrier),
            need_weights=False,
        )
        carrier, self_delta = self._write(carrier, proposed, sa_gate)

        query = _modulate(
            self.visual_query_norm(carrier),
            visual_shift,
            visual_scale,
        )
        memory = self.visual_norm(visual_memory)
        proposed, _ = self.visual_attention(
            query,
            memory,
            memory,
            need_weights=False,
        )
        carrier, visual_delta = self._write(carrier, proposed, visual_gate)

        query = _modulate(
            self.transition_query_norm(carrier),
            transition_shift,
            transition_scale,
        )
        memory = self.transition_norm(state_memory)
        proposed, _ = self.transition_attention(
            query,
            memory,
            memory,
            need_weights=False,
        )
        carrier, transition_delta = self._write(
            carrier,
            proposed,
            transition_gate,
        )

        proposed = self.ffn(
            _modulate(self.ffn_norm(carrier), ffn_shift, ffn_scale)
        )
        carrier, ffn_delta = self._write(carrier, proposed, ffn_gate)
        return carrier, {
            "self": self_delta,
            "visual": visual_delta,
            "transition": transition_delta,
            "ffn": ffn_delta,
        }


class TypedGroundingRoleHost(nn.Module):
    """Three active grounding role blocks on the current typed fact chart."""

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        route_dim: int,
        state_dim: int,
        heads: int,
        depth: int = 3,
        expansion: float = 4.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(depth) != 3:
            raise ValueError("active grounding role host requires exactly three blocks")
        self.hidden = int(hidden)
        self.content_projection = nn.Linear(content_dim, hidden, bias=False)
        self.semantic_projection = nn.Linear(route_dim, hidden, bias=False)
        self.appearance_projection = nn.Linear(route_dim, hidden, bias=False)
        self.geometry_projection = nn.Linear(route_dim, hidden, bias=False)
        self.state_projection = nn.Linear(state_dim, hidden, bias=False)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden, elementwise_affine=False),
            nn.Linear(2 * hidden, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.blocks = nn.ModuleList(
            _GroundingRoleHostBlock(
                hidden=hidden,
                heads=heads,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(int(depth))
        )

    def _visual_memory(self, facts: LocalFactSet) -> Tensor:
        typed = (
            self.semantic_projection(facts.semantic_slots)
            + self.appearance_projection(facts.appearance_slots)
            + self.geometry_projection(facts.geometry_slots)
        ) / (3.0**0.5)
        content = self.content_projection(facts.content_slots)
        valid = facts.slot_validity.to(dtype=content.dtype)
        memory = (content + typed) * valid
        return memory.reshape(int(memory.shape[0]), -1, self.hidden)

    def forward(
        self,
        facts: LocalFactSet,
        *,
        state: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[LocalFactSet, dict[str, Tensor]]:
        facts.validate()
        if tuple(state.shape) != (facts.batch, int(self.state_projection.in_features)):
            raise ValueError("grounding role host current state is misaligned")
        original_shape = tuple(facts.public_scene_base.shape)
        carrier = facts.public_scene_base.reshape(facts.batch, -1, self.hidden)
        visual_memory = self._visual_memory(facts)
        state_memory = self.state_projection(state)[:, None]
        metric_rows: dict[str, Tensor] = {}
        for index, block in enumerate(self.blocks, start=1):
            condition = self.condition_projection(
                torch.cat((carrier.mean(dim=1), state_memory[:, 0]), dim=-1)
            )
            carrier, deltas = block(
                carrier,
                visual_memory=visual_memory,
                state_memory=state_memory,
                condition=condition,
            )
            if collect_diagnostics:
                for name, value in deltas.items():
                    metric_rows[f"grounding_host_g{index}_{name}_rms"] = (
                        value.detach().float().square().mean().sqrt()
                    )
        hosted = replace(
            facts,
            public_scene_base=carrier.reshape(original_shape),
        )
        hosted.validate()
        return hosted, metric_rows


class StaticP1RoleHost(nn.Module):
    """The active first policy block, restricted to clean P1 query tokens."""

    def __init__(
        self,
        *,
        hidden: int,
        heads: int,
        expansion: float = 4.0,
        dropout: float = 0.05,
        maximum_update_rms: float = 0.50,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.maximum_update_rms = float(maximum_update_rms)
        self.self_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, int(hidden * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(hidden * expansion), hidden, bias=False),
        )
        # Keep the active block's full AdaLN parameterization.  The disabled
        # visual/transition branches are not reconstructed as frozen baggage.
        self.modulation = nn.Linear(hidden, 12 * hidden)
        self.role_identity = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.modulation.weight, mean=0.0, std=3e-3)
        nn.init.zeros_(self.modulation.bias)
        with torch.no_grad():
            for index in (2, 5, 8, 11):
                self.modulation.bias[index * hidden : (index + 1) * hidden].fill_(-2.0)

    def _write(self, value: Tensor, update: Tensor, gate: Tensor) -> tuple[Tensor, Tensor]:
        bounded, _ = smooth_rms_contract(
            self.dropout(update) * torch.sigmoid(gate)[:, None],
            self.maximum_update_rms,
        )
        return value + bounded, bounded

    def forward(self, query: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if query.ndim != 4 or int(query.shape[-1]) != self.hidden:
            raise ValueError("P1 role host query must be [B,T,Q,H]")
        shape = tuple(query.shape)
        value = query.reshape(int(query.shape[0]), -1, self.hidden)
        condition = value.mean(dim=1) + self.role_identity.to(
            device=value.device,
            dtype=value.dtype,
        )
        chunks = self.modulation(condition).chunk(12, dim=-1)
        normalized = _modulate(self.self_norm(value), chunks[0], chunks[1])
        update, _ = self.self_attention(
            normalized,
            normalized,
            self.self_norm(value),
            need_weights=False,
        )
        value, self_delta = self._write(value, update, chunks[2])
        update = self.ffn(
            _modulate(self.ffn_norm(value), chunks[9], chunks[10])
        )
        value, ffn_delta = self._write(value, update, chunks[11])
        return value.reshape(shape), {
            "p1_host_self_rms": self_delta.detach().float().square().mean().sqrt(),
            "p1_host_ffn_rms": ffn_delta.detach().float().square().mean().sqrt(),
        }


__all__ = ["StaticP1RoleHost", "TypedGroundingRoleHost"]
