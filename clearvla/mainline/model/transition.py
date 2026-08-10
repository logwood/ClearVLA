"""Action-centred controlled transition preserved at the P-to-bottom boundary."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..interfaces import ObservableHistory
from .bottom import canonical_state_history
from .types import (
    ControlledTransitionState,
    FutureObjectDynamics,
    HistoryActionProposalState,
)


class ControlledTransitionDynamics(nn.Module):
    """Low-rank W transition directions controlled by a clean action proposal.

    This is the typed equivalent of the active
    ``ControlledResidualLatentDynamics`` path.  The action-independent base is
    fixed to zero.  Values are produced by
    ``coeff(real proposal) - coeff(neutral context)`` and enter the bottom as
    read-only evidence.
    """

    def __init__(
        self,
        *,
        hidden: int,
        content_dim: int,
        state_dim: int,
        action_dim: int,
        cameras: int,
        heads: int,
        rank: int = 8,
        action_tokens: int = 8,
        neutral_tokens: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.intervals = 4
        self.cameras = int(cameras)
        self.current_projection = nn.Linear(content_dim, hidden, bias=False)
        self.effect_projection = nn.Linear(content_dim, hidden, bias=False)
        geometry_width = self.cameras * (2 + 3) + 2
        self.geometry_projection = nn.Linear(geometry_width, hidden, bias=False)
        self.interval_key = nn.Parameter(
            torch.randn(1, self.intervals, 1, hidden) * 0.02
        )
        self.state_projection = nn.Linear(state_dim, hidden, bias=False)
        self.action_projection = nn.Linear(action_dim, hidden, bias=False)
        self.basis_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, rank * hidden),
        )
        self.action_queries = nn.Parameter(
            torch.randn(1, int(action_tokens), hidden) * 0.02
        )
        self.neutral_queries = nn.Parameter(
            torch.randn(1, int(neutral_tokens), hidden) * 0.02
        )
        self.neutral_bias = nn.Parameter(torch.zeros(1, 1, hidden))
        self.action_memory_norm = nn.LayerNorm(hidden)
        self.action_cross = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transition_query_norm = nn.LayerNorm(hidden)
        self.action_latent_norm = nn.LayerNorm(hidden)
        self.coefficient_cross = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.direct_action_norm = nn.LayerNorm(hidden)
        self.direct_action_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )
        self.delta_dropout = nn.Dropout(dropout)
        self.delta_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        initialized = (
            (self.basis_head, 3e-2),
            (self.coefficient_head, 5e-2),
            (self.direct_action_mlp, 5e-2),
        )
        for head, standard_deviation in initialized:
            output = head[-1]
            if not isinstance(output, nn.Linear) or output.bias is None:
                raise TypeError("controlled-transition heads must end in biased linears")
            nn.init.normal_(output.weight, mean=0.0, std=standard_deviation)
            nn.init.zeros_(output.bias)

    def _transition_tokens(self, dynamics: FutureObjectDynamics) -> Tensor:
        dynamics.validate()
        batch, intervals, objects = dynamics.semantic_delta.shape[:3]
        if intervals != self.intervals:
            raise ValueError("controlled transition requires four W intervals")
        current = self.current_projection(dynamics.current_reference)[:, None]
        semantic = self.effect_projection(dynamics.semantic_delta)
        geometry = torch.cat(
            (
                dynamics.transport_mean.flatten(-2),
                dynamics.transport_covariance.flatten(-2),
                dynamics.visibility,
                dynamics.persistence,
            ),
            dim=-1,
        )
        geometry = self.geometry_projection(geometry)
        transition = current + semantic + geometry
        transition = transition + self.interval_key.to(
            device=transition.device,
            dtype=transition.dtype,
        )
        return transition.reshape(batch, intervals * objects, self.hidden)

    def _context_tokens(self, history: ObservableHistory) -> Tensor:
        state = self.state_projection(canonical_state_history(history))
        action = self.action_projection(history.executed_action_history)
        return torch.cat((state, action), dim=1)

    def _coefficients(
        self,
        transition: Tensor,
        context: Tensor,
        *,
        action_tokens: Tensor | None,
        neutral: bool,
    ) -> Tensor:
        batch, rows = transition.shape[:2]
        if neutral:
            queries = self.neutral_queries.expand(batch, -1, -1).to(
                device=transition.device,
                dtype=transition.dtype,
            )
            memory_source = context
            direct = self.neutral_bias.to(
                device=transition.device,
                dtype=transition.dtype,
            ).expand(batch, rows, -1)
        else:
            if action_tokens is None:
                raise ValueError("real controlled transition requires clean proposal tokens")
            queries = self.action_queries.expand(batch, -1, -1).to(
                device=transition.device,
                dtype=transition.dtype,
            )
            memory_source = torch.cat((context, action_tokens), dim=1)
            direct_action = self.direct_action_mlp(
                self.direct_action_norm(action_tokens).mean(dim=1)
            )
            direct = direct_action[:, None].expand(-1, rows, -1)
        memory = self.action_memory_norm(memory_source)
        latent_action, _ = self.action_cross(
            queries,
            memory,
            memory,
            need_weights=False,
        )
        query = self.transition_query_norm(transition)
        latent = self.action_latent_norm(latent_action)
        action_context, _ = self.coefficient_cross(
            query,
            latent,
            latent,
            need_weights=False,
        )
        return torch.tanh(
            self.coefficient_head(torch.cat((query, action_context + direct), dim=-1))
        )

    def forward(
        self,
        *,
        dynamics: FutureObjectDynamics,
        proposal: HistoryActionProposalState,
        history: ObservableHistory,
        collect_diagnostics: bool = False,
    ) -> tuple[ControlledTransitionState, dict[str, Tensor]]:
        transition = self._transition_tokens(dynamics)
        context = self._context_tokens(history)
        action_coefficients = self._coefficients(
            transition,
            context,
            action_tokens=proposal.tokens,
            neutral=False,
        )
        neutral_coefficients = self._coefficients(
            transition,
            context,
            action_tokens=None,
            neutral=True,
        )
        coefficient_delta = action_coefficients - neutral_coefficients
        batch, rows = transition.shape[:2]
        basis = self.basis_head(transition).reshape(
            batch,
            rows,
            self.rank,
            self.hidden,
        )
        value = torch.einsum("bnr,bnrh->bnh", coefficient_delta, basis)
        value = value / float(self.rank) ** 0.5
        value = self.delta_dropout(
            value
            * self.delta_gain.to(device=value.device, dtype=value.dtype)
        )
        result = ControlledTransitionState(
            selector=transition,
            value=value,
            action_coefficients=action_coefficients,
            neutral_coefficients=neutral_coefficients,
        )
        result.validate(hidden=self.hidden)
        if not collect_diagnostics:
            return result, {}
        return result, {
            "controlled_transition_value_rms": value.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "controlled_transition_basis_rms": basis.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "controlled_transition_centered_coefficient_abs_mean": coefficient_delta.detach()
            .float()
            .abs()
            .mean(),
            "controlled_transition_delta_gain": self.delta_gain.detach().float().abs(),
        }


__all__ = ["ControlledTransitionDynamics"]
