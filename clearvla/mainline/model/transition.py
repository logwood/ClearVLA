"""Action-centred controlled transition preserved at the P-to-bottom boundary."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..interfaces import ObservableHistory
from .bottom import canonical_state_history
from .types import (
    ControlledTransitionState,
    FutureObjectDynamics,
    HistoryActionProposalState,
    ObjectFactSet,
)


class ControlledTransitionDynamics(nn.Module):
    """Low-rank W transition directions controlled by a clean action proposal.

    This is the typed equivalent of the active
    ``ControlledResidualLatentDynamics`` path.  The action-independent base is
    fixed to zero.  Values are produced by
    ``coeff(real proposal) - coeff(zero proposal)`` through the same network
    and enter the bottom as
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
        horizon: int = 24,
        basis: int = 4,
        rank: int = 8,
        action_tokens: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.intervals = 4
        self.cameras = int(cameras)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.current_projection = nn.Linear(content_dim, hidden, bias=False)
        self.effect_projection = nn.Linear(content_dim, hidden, bias=False)
        self.geometry_projection = nn.Linear(7, hidden, bias=False)
        self.interval_key = nn.Parameter(
            torch.randn(1, self.intervals, 1, 1, 1, hidden) * 0.02
        )
        # Retain one transition read per action basis.  Collapsing the full
        # 4*C*8*8 transition chart directly to one row per horizon recreated a
        # premature global bottleneck; keeping H*B rows preserves the same four
        # typed action bases used by P1/P3 without sending all 512 rows through
        # every bottom cross-attention.
        self.pool_query = nn.Parameter(
            torch.randn(1, self.horizon, self.basis, hidden) * 0.02
        )
        self.pool_key = nn.Linear(hidden, hidden, bias=False)
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
        self.action_memory_norm = nn.LayerNorm(hidden)
        # Real and zero-proposal counterfactuals traverse the same
        # deterministic coefficient network.  Attention dropout here would
        # make identical zero proposals differ; optional regularization is
        # applied only after the centered delta has been formed.
        self.action_cross = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=0.0,
        )
        self.transition_query_norm = nn.LayerNorm(hidden)
        self.action_latent_norm = nn.LayerNorm(hidden)
        self.coefficient_cross = nn.MultiheadAttention(
            hidden,
            heads,
            batch_first=True,
            dropout=0.0,
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

    def _transition_tokens(
        self,
        dynamics: FutureObjectDynamics,
        facts: ObjectFactSet,
    ) -> Tensor:
        dynamics.validate()
        facts.validate()
        batch, intervals, objects = dynamics.semantic_delta.shape[:3]
        if intervals != self.intervals:
            raise ValueError("controlled transition requires four W intervals")
        chart = facts.dense_chart
        candidate_weight = (
            chart.candidate_owner_prior.float()
            * chart.candidate_validity[..., 0].float()
        )
        observed_content = torch.einsum(
            "bcyxm,bcyxmd->bcyxd",
            candidate_weight.to(dtype=chart.candidate_content.dtype),
            chart.candidate_content,
        ) / candidate_weight.sum(dim=-1, keepdim=True).to(
            dtype=chart.candidate_content.dtype
        ).clamp_min(1e-6)
        current = self.current_projection(observed_content)[:, None]
        semantic = self.effect_projection(dynamics.semantic_delta)
        geometry = torch.cat(
            (
                dynamics.transport_mean,
                dynamics.transport_covariance,
                dynamics.visibility[:, :, :, None].expand(-1, -1, -1, self.cameras, -1),
                dynamics.persistence[:, :, :, None].expand(-1, -1, -1, self.cameras, -1),
            ),
            dim=-1,
        )
        geometry = self.geometry_projection(geometry)
        address = dynamics.future_address.float().clamp_min(0.0)
        address_mass = address.sum(dim=2).clamp_min(1e-6)
        spatial_semantic = torch.einsum(
            "bikcyx,bikh->bicyxh",
            address.to(dtype=semantic.dtype),
            semantic,
        ) / address_mass[..., None].to(dtype=semantic.dtype)
        spatial_geometry = torch.einsum(
            "bikcyx,bikch->bicyxh",
            address.to(dtype=geometry.dtype),
            geometry,
        ) / address_mass[..., None].to(dtype=geometry.dtype)
        transition = current + spatial_semantic + spatial_geometry
        transition = transition + self.interval_key.to(
            device=transition.device,
            dtype=transition.dtype,
        )
        return transition.reshape(batch, intervals * self.cameras * int(address.shape[-2]) * int(address.shape[-1]), self.hidden)

    def _pool_dense_transition(
        self,
        selector: Tensor,
        value: Tensor,
        action_coefficients: Tensor,
        neutral_coefficients: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        query = torch.nn.functional.normalize(
            self.pool_query.expand(int(selector.shape[0]), -1, -1, -1)
            .reshape(int(selector.shape[0]), self.horizon * self.basis, self.hidden)
            .float(),
            dim=-1,
            eps=0.25,
        )
        key = torch.nn.functional.normalize(
            self.pool_key(selector).float(),
            dim=-1,
            eps=0.25,
        )
        probability = torch.softmax(
            torch.einsum("bth,bnh->btn", query, key) / float(self.hidden) ** 0.5,
            dim=-1,
        )
        pooled_selector = self.pool_query.to(
            device=selector.device, dtype=selector.dtype
        ).reshape(1, self.horizon * self.basis, self.hidden) + torch.einsum(
            "btn,bnh->bth",
            probability.to(dtype=selector.dtype),
            selector,
        )
        pooled_value = torch.einsum(
            "btn,bnh->bth",
            probability.to(dtype=value.dtype),
            value,
        )
        pooled_action = torch.einsum(
            "btn,bnr->btr",
            probability.to(dtype=action_coefficients.dtype),
            action_coefficients,
        )
        pooled_neutral = torch.einsum(
            "btn,bnr->btr",
            probability.to(dtype=neutral_coefficients.dtype),
            neutral_coefficients,
        )
        return pooled_selector, pooled_value, pooled_action, pooled_neutral, probability

    def _context_tokens(self, history: ObservableHistory) -> Tensor:
        state = self.state_projection(canonical_state_history(history))
        action = self.action_projection(history.executed_action_history)
        return torch.cat((state, action), dim=1)

    def _coefficients(
        self,
        transition: Tensor,
        context: Tensor,
        *,
        action_tokens: Tensor,
    ) -> Tensor:
        batch, rows = transition.shape[:2]
        if action_tokens.ndim != 3 or int(action_tokens.shape[0]) != batch:
            raise ValueError("controlled transition action tokens must be [B,A,H]")
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
        facts: ObjectFactSet,
        proposal: HistoryActionProposalState,
        history: ObservableHistory,
        collect_diagnostics: bool = False,
    ) -> tuple[ControlledTransitionState, dict[str, Tensor]]:
        transition = self._transition_tokens(dynamics, facts)
        context = self._context_tokens(history)
        action_coefficients = self._coefficients(
            transition,
            context,
            action_tokens=proposal.tokens,
        )
        neutral_coefficients = self._coefficients(
            transition,
            context,
            action_tokens=torch.zeros_like(proposal.tokens),
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
        (
            pooled_transition,
            pooled_value,
            pooled_action_coefficients,
            pooled_neutral_coefficients,
            pool_probability,
        ) = self._pool_dense_transition(
            transition,
            value,
            action_coefficients,
            neutral_coefficients,
        )
        result = ControlledTransitionState(
            selector=pooled_transition,
            value=pooled_value,
            action_coefficients=pooled_action_coefficients,
            neutral_coefficients=pooled_neutral_coefficients,
        )
        result.validate(hidden=self.hidden)
        if not collect_diagnostics:
            return result, {}
        return result, {
            "controlled_transition_value_rms": pooled_value.detach()
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
            "controlled_transition_dense_rows": value.new_tensor(float(value.shape[1])),
            "controlled_transition_pooled_rows": pooled_value.new_tensor(
                float(pooled_value.shape[1])
            ),
            "controlled_transition_pool_entropy": (
                -(pool_probability.detach().float().clamp_min(1e-8)
                  * pool_probability.detach().float().clamp_min(1e-8).log())
                .sum(dim=-1)
                / math.log(float(max(int(pool_probability.shape[-1]), 2)))
            ).mean(),
            "controlled_transition_spatial_value_variation": value.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
        }


__all__ = ["ControlledTransitionDynamics"]
