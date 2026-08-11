"""V120 512-row action-centred transition at the P-to-bottom boundary.

The spatial transition chart is never pooled before the Evidence-MMDiT.  The
only action-dependent value is produced by the extracted V120
``ControlledResidualLatentDynamics`` as ``coeff(real) - coeff(zero)`` through
one shared network.  Current object/W improvements may change the 512 source
rows, but they cannot replace the mature V120 transition operator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from ..interfaces import ObservableHistory
from ..v120_core.profile import build_v120_policy_config
from ..v120_core.trunk_primitives import (
    ControlledResidualLatentDynamics,
    TrunkPrimitiveConfig,
)
from .action_contract import canonical_state_history
from .types import (
    ControlledTransitionState,
    FutureObjectDynamics,
    HistoryActionProposalState,
    ObjectFactSet,
)


class ControlledTransitionDynamics(nn.Module):
    """Typed W source rows feeding the exact extracted V120 transition core."""

    @property
    def action_queries(self) -> nn.Parameter:
        """Expose the mature core query bank at the typed mainline boundary."""

        return self.v120_transition.action_queries

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
        if self.cameras != 2:
            raise ValueError("the restored V120 transition requires two camera charts")
        self.current_projection = nn.Linear(content_dim, hidden, bias=False)
        self.effect_projection = nn.Linear(content_dim, hidden, bias=False)
        self.geometry_projection = nn.Linear(7, hidden, bias=False)
        self.interval_key = nn.Parameter(
            torch.randn(1, self.intervals, 1, 1, 1, hidden) * 0.02
        )
        self.state_projection = nn.Linear(state_dim, hidden, bias=False)
        self.action_projection = nn.Linear(action_dim, hidden, bias=False)
        reference_config = build_v120_policy_config()
        core_config = replace(
            reference_config,
            hidden_size=self.hidden,
            num_heads=int(heads),
            controlled_delta_rank=self.rank,
            latent_action_tokens=int(action_tokens),
            # The two centered coefficient evaluations must be the same
            # function, including stochastic semantics.  V120's generic 0.05
            # transformer dropout made ``coeff(zero) - coeff(zero)`` non-zero
            # during training because the two calls sampled different masks.
            # The transition-specific dropout knob is zero in the recovered
            # profile, so use it for the coefficient attentions as well.
            dropout=float(dropout),
            controlled_delta_dropout=float(dropout),
            base_effect_hidden=min(
                int(reference_config.base_effect_hidden),
                max(self.hidden, 8),
            ),
        )
        core_config.validate()
        self.v120_transition = ControlledResidualLatentDynamics(
            cast(TrunkPrimitiveConfig, core_config)
        )
        # The integrated path deliberately replaces V120's learned neutral
        # query with the same coefficient network evaluated on an explicit
        # zero proposal.  Keep the mechanically extracted module intact, but
        # remove the superseded neutral-only tensors from optimizer ownership.
        # Leaving them trainable would create silent dead parameters and would
        # make the serialized optimizer contract depend on a disabled branch.
        self.v120_transition.neutral_queries.requires_grad_(False)
        self.v120_transition.neutral_bias.requires_grad_(False)
        self.v120_exact_profile = bool(
            self.hidden == int(reference_config.hidden_size)
            and int(heads) == int(reference_config.num_heads)
            and self.rank == int(reference_config.controlled_delta_rank)
            and int(action_tokens) == int(reference_config.latent_action_tokens)
            and float(dropout) == float(reference_config.controlled_delta_dropout)
        )

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

    def _context_tokens(self, history: ObservableHistory) -> Tensor:
        state = self.state_projection(canonical_state_history(history))
        action = self.action_projection(history.executed_action_history)
        return torch.cat((state, action), dim=1)

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
        batch, rows, hidden = transition.shape
        if rows != 4 * self.cameras * 8 * 8 or hidden != self.hidden:
            raise ValueError("the restored transition must retain all 512 spatial rows")

        # Layer the proven zero-proposal improvement on the extracted V120
        # operator: both operands use the same queries, attention and heads.
        # This removes the old learned-neutral-query alias without replacing
        # the mature basis/coefficient network.
        action_coefficients, _, _ = self.v120_transition._coeff(
            transition,
            context,
            action_tokens=proposal.tokens,
            neutral=False,
        )
        neutral_coefficients, _, _ = self.v120_transition._coeff(
            transition,
            context,
            action_tokens=torch.zeros_like(proposal.tokens),
            neutral=False,
        )
        coefficient_delta = action_coefficients - neutral_coefficients
        basis = self.v120_transition.basis_head(transition).reshape(
            batch,
            rows,
            self.rank,
            self.hidden,
        )
        value = torch.einsum("bnr,bnrh->bnh", coefficient_delta, basis)
        value = value / float(self.rank) ** 0.5
        value = self.v120_transition.delta_drop(
            value
            * self.v120_transition.delta_gain.to(
                device=value.device,
                dtype=value.dtype,
            )
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
            "controlled_transition_delta_gain": self.v120_transition.delta_gain.detach()
            .float()
            .abs(),
            "controlled_transition_dense_rows": value.new_tensor(float(value.shape[1])),
            "controlled_transition_retained_rows": value.new_tensor(float(value.shape[1])),
            "controlled_transition_pool_removed": value.new_zeros(()),
            "controlled_transition_v120_exact_profile": value.new_tensor(
                float(self.v120_exact_profile)
            ),
            "controlled_transition_spatial_value_variation": value.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
        }


__all__ = ["ControlledTransitionDynamics"]
