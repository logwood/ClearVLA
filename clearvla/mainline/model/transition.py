"""Recovered V120 per-ODE action-centred controlled transition."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
from torch import Tensor, nn

from ..v120_core.profile import build_v120_policy_config
from ..v120_core.trunk_primitives import (
    ControlledResidualLatentDynamics,
    TrunkPrimitiveConfig,
)
from .compiler import ObjectPolicyPlanDeltaBank
from .action_contract import V120SeedContext
from .routing import AffineVarianceFlooredCenteredNorm
from .types import (
    ControlledTransitionSource,
    ControlledTransitionState,
    ObjectFactSet,
)


class ControlledTransitionDynamics(nn.Module):
    """Static protected G3 source plus dynamic V120 action/neutral coefficients."""

    @property
    def action_queries(self) -> nn.Parameter:
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
        normalization_floor: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del content_dim, state_dim, action_dim
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.intervals = 4
        self.cameras = int(cameras)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.state_history_rows = 3
        self.executed_rows = 7
        if self.cameras != 2:
            raise ValueError("the recovered V120 transition requires two cameras")
        self.interval_identity = nn.Parameter(
            torch.randn(1, self.intervals, 1, self.hidden) * 0.02
        )
        # This is the extracted V120 terminal trajectory normalization.  It
        # belongs immediately after the P2 residual write and before the
        # controlled action reader.  Moving it earlier or dropping it changes
        # the transition Jacobian and was one source of schema-20 drift.
        self.trajectory_norm = AffineVarianceFlooredCenteredNorm(
            self.hidden,
            float(normalization_floor),
            affine_maximum=4.0,
        )
        reference_config = build_v120_policy_config()
        core_config = replace(
            reference_config,
            hidden_size=self.hidden,
            num_heads=int(heads),
            controlled_delta_rank=self.rank,
            latent_action_tokens=int(action_tokens),
            dropout=float(dropout),
            controlled_delta_dropout=float(dropout),
            base_effect_hidden=min(
                int(reference_config.base_effect_hidden), max(self.hidden, 8)
            ),
        )
        core_config.validate()
        self.v120_transition = ControlledResidualLatentDynamics(
            cast(TrunkPrimitiveConfig, core_config)
        )
        # V120's learned no-op queries are part of the identifiable centered
        # operator.  They must remain trainable; replacing them with the same
        # network evaluated on an all-zero proposal changes the function.
        self.v120_transition.neutral_queries.requires_grad_(True)
        self.v120_transition.neutral_bias.requires_grad_(True)

    def build_source(
        self,
        *,
        facts: ObjectFactSet,
        collect_diagnostics: bool = False,
    ) -> tuple[ControlledTransitionSource, dict[str, Tensor]]:
        """Build the ODE-invariant protected G3 chart exactly once."""

        facts.validate()
        chart = facts.dense_chart.public_scene_base
        if chart.ndim != 5 or tuple(chart.shape[1:4]) != (self.cameras, 8, 8):
            raise ValueError("controlled transition requires the full [C,8,8] G3 chart")
        if int(chart.shape[-1]) != self.hidden:
            raise ValueError("G3 transition chart has an invalid hidden width")
        batch = int(chart.shape[0])
        spatial = chart.reshape(batch, 1, self.cameras * 8 * 8, self.hidden)
        source = spatial + self.interval_identity.to(
            device=chart.device, dtype=chart.dtype
        )
        result = ControlledTransitionSource(
            selector=source.reshape(
                batch, self.intervals * self.cameras * 8 * 8, self.hidden
            )
        )
        result.validate(hidden=self.hidden)
        if not collect_diagnostics:
            return result, {}
        return result, {
            "controlled_transition_source_rms": result.selector.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "controlled_transition_source_spatial_variation": result.selector.detach()
            .float()
            .reshape(batch, self.intervals, self.cameras * 8 * 8, self.hidden)
            .std(dim=2, unbiased=False)
            .mean(),
        }

    def _context_tokens(
        self,
        seed: V120SeedContext,
        plan: ObjectPolicyPlanDeltaBank,
    ) -> Tensor:
        plan.validate()
        seed.validate(
            hidden=self.hidden,
            state_history=self.state_history_rows,
            executed=self.executed_rows,
        )
        dtype = plan.protected_base.dtype
        device = plan.protected_base.device
        # V120 terminal-normalized the complete canvas before constructing
        # this context.  The protected P1/P2 consequence itself remained an
        # explicit typed delta and therefore is not normalized here.
        state = self.trajectory_norm(seed.state)
        state_history = self.trajectory_norm(seed.state_history[:, -1:])
        executed = self.trajectory_norm(seed.executed[:, -1:])
        return torch.cat(
            (
                state.to(device=device, dtype=dtype),
                state_history.to(device=device, dtype=dtype),
                executed.to(device=device, dtype=dtype),
                plan.protected_base.flatten(1, 2),
            ),
            dim=1,
        )

    def forward(
        self,
        *,
        source: ControlledTransitionSource,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        seed: V120SeedContext,
        collect_diagnostics: bool = False,
    ) -> tuple[ControlledTransitionState, dict[str, Tensor]]:
        """Evaluate real minus learned-neutral coefficients for this ODE step."""

        source.validate(hidden=self.hidden)
        expected_action = (
            int(source.selector.shape[0]),
            self.horizon,
            self.basis,
            self.hidden,
        )
        if tuple(action_query.shape) != expected_action:
            raise ValueError("controlled transition action query must be [B,T,Q,H]")
        if tuple(plan.protected_base.shape) != expected_action:
            raise ValueError("controlled transition lost the P1+P2 consequence")
        # V120 passed the complete 24x4 trajectory after P1/P2 and terminal
        # normalization.  Do not basis-reduce this to 24 rows: doing so changes
        # the action cross-attention denominator and erases the factual/effect
        # residual that the transition was meant to condition on.
        trajectory, norm_denominator, norm_gain = self.trajectory_norm.forward_with_denominator(
            action_query + plan.protected_base
        )
        action_tokens = trajectory.flatten(1, 2)
        context = self._context_tokens(seed, plan)
        transition = source.selector
        raw = self.v120_transition(
            transition,
            context,
            action_tokens=action_tokens,
            transition_tokens=transition,
        )
        value = raw["rollout_delta_pred"]
        action_coefficients = raw["rollout_action_coeff"]
        neutral_coefficients = raw["rollout_neutral_coeff"]
        result = ControlledTransitionState(
            selector=transition,
            value=value,
            action_coefficients=action_coefficients,
            neutral_coefficients=neutral_coefficients,
        )
        result.validate(hidden=self.hidden)
        if not collect_diagnostics:
            return result, {}
        basis = raw["rollout_transition_basis"]
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
            "controlled_transition_centered_coefficient_abs_mean": (
                action_coefficients - neutral_coefficients
            )
            .detach()
            .float()
            .abs()
            .mean(),
            "controlled_transition_delta_gain": self.v120_transition.delta_gain.detach()
            .float()
            .abs(),
            "controlled_transition_dense_rows": value.new_tensor(float(value.shape[1])),
            "controlled_transition_retained_rows": value.new_tensor(float(value.shape[1])),
            "controlled_transition_per_ode_action": value.new_ones((), dtype=torch.float32),
            "controlled_transition_learned_neutral": value.new_ones((), dtype=torch.float32),
            "controlled_transition_spatial_value_variation": value.detach()
            .float()
            .std(dim=1, unbiased=False)
            .mean(),
            "controlled_transition_action_token_rows": value.new_tensor(
                float(action_tokens.shape[1]), dtype=torch.float32
            ),
            "controlled_transition_trajectory_norm_denominator_min": (
                norm_denominator.detach().float().amin()
            ),
            "controlled_transition_trajectory_norm_gain": norm_gain.detach().float(),
        }


__all__ = ["ControlledTransitionDynamics"]
