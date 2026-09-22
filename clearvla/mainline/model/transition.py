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
from .action_contract import V120SeedContext
from .compiler import ObjectPolicyPlanDeltaBank
from .routing import AffineVarianceFlooredCenteredNorm
from .types import (
    ControlledTransitionSource,
    ControlledTransitionState,
    PhysicalTransitionInnovation,
)


class ControlledTransitionDynamics(nn.Module):
    """Static protected G3 source plus dynamic V120 action/neutral coefficients."""

    INTERVENTION_MODES = ("delta_neutral",)

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
        physical_field_dim: int,
        cameras: int,
        heads: int,
        horizon: int = 24,
        basis: int = 4,
        rank: int = 8,
        action_tokens: int = 8,
        normalization_floor: float = 0.25,
        dropout: float = 0.0,
        target_action_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        del content_dim, state_dim
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.cameras = int(cameras)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.action_dim = int(action_dim)
        self.physical_field_dim = int(physical_field_dim)
        self.target_action_bottleneck = bool(target_action_bottleneck)
        self.state_history_rows = 3
        self.executed_rows = 7
        if self.cameras != 2:
            raise ValueError("the recovered V120 transition requires two cameras")
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
        self.physical_action_input: nn.Linear | None = (
            nn.Linear(self.action_dim, self.hidden, bias=False)
            if self.target_action_bottleneck
            else None
        )
        self.physical_delta_head: nn.Linear | None = None
        if self.target_action_bottleneck:
            # This is the only public CT readout in the target-action graph.
            # Keep the new public physical correction exactly zero at
            # migration while giving task loss a direct structural path into
            # the private centered transition value.  This does not claim
            # functional equivalence to the retired 512-row bottom ingress.
            self.physical_delta_head = nn.Linear(
                self.hidden,
                self.physical_field_dim,
                bias=False,
            )
            nn.init.zeros_(self.physical_delta_head.weight)
        # Non-persistent validation state.  It is absent from state_dict and
        # does not add a parameter, buffer, optimizer owner or initialization
        # draw to the recovered transition.
        self._eval_intervention = "none"

    def set_eval_intervention(self, mode: str) -> None:
        """Remove real-minus-neutral CT value only at its output boundary."""

        if self.training:
            raise ValueError("controlled-transition interventions are evaluation-only")
        if mode not in self.INTERVENTION_MODES:
            raise ValueError(
                "controlled-transition intervention must be one of "
                + ", ".join(self.INTERVENTION_MODES)
            )
        self._eval_intervention = mode

    def clear_eval_intervention(self) -> None:
        self._eval_intervention = "none"

    def build_source(
        self,
        *,
        g3_rollout: Tensor,
        collect_diagnostics: bool = False,
    ) -> tuple[ControlledTransitionSource, dict[str, Tensor]]:
        """Protect the exact completed G3 anchor rollout for every ODE step.

        V120 passed the final G3 rollout directly to the controlled transition.
        Reconstructing this source from the public chart loses the four anchor
        identities and then invents unrelated interval labels. P1 and the
        transition therefore share this exact tensor boundary.
        """

        expected_rows = 4 * self.cameras * 8 * 8
        if g3_rollout.ndim != 3 or tuple(g3_rollout.shape[1:]) != (
            expected_rows,
            self.hidden,
        ):
            raise ValueError(
                "controlled transition requires the exact [B,4*C*8*8,H] G3 rollout"
            )
        batch = int(g3_rollout.shape[0])
        result = ControlledTransitionSource(selector=g3_rollout)
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
            .reshape(batch, 4, self.cameras * 8 * 8, self.hidden)
            .std(dim=2, unbiased=False)
            .mean(),
            "controlled_transition_source_anchor_variation": result.selector.detach()
            .float()
            .reshape(batch, 4, self.cameras * 8 * 8, self.hidden)
            .std(dim=1, unbiased=False)
            .mean(),
        }

    def _context_tokens(
        self,
        seed: V120SeedContext,
        plan: ObjectPolicyPlanDeltaBank,
        action_proposal: Tensor | None,
    ) -> Tensor:
        plan.validate()
        seed.validate(
            hidden=self.hidden,
            state_history=self.state_history_rows,
            executed=self.executed_rows,
        )
        if self.target_action_bottleneck:
            if self.physical_action_input is None or action_proposal is None:
                raise ValueError("target-action CT requires the physical A0 proposal")
            if tuple(action_proposal.shape) != (
                int(seed.state.shape[0]),
                self.horizon,
                self.action_dim,
            ):
                raise ValueError("target-action CT A0 must be [B,T,A]")
            proposal_context = self.physical_action_input(action_proposal)
            dtype = proposal_context.dtype
            device = proposal_context.device
            policy_context = proposal_context
        else:
            dtype = plan.protected_base.dtype
            device = plan.protected_base.device
            policy_context = plan.protected_base.flatten(1, 2)
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
                policy_context,
            ),
            dim=1,
        )

    def forward(
        self,
        *,
        source: ControlledTransitionSource,
        action_query: Tensor,
        plan: ObjectPolicyPlanDeltaBank,
        action_proposal: Tensor | None = None,
        seed: V120SeedContext,
        collect_diagnostics: bool = False,
    ) -> tuple[
        ControlledTransitionState | PhysicalTransitionInnovation,
        dict[str, Tensor],
    ]:
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
        if tuple(plan.protected_policy_precision.shape) != expected_action:
            raise ValueError("controlled transition lost the dynamic P1 policy residual")
        # V120 passed the complete 24x4 trajectory after P1/P2 and terminal
        # normalization.  Do not basis-reduce this to 24 rows: doing so changes
        # the action cross-attention denominator and erases the factual/effect
        # residual that the transition was meant to condition on.
        if self.target_action_bottleneck:
            if self.physical_action_input is None or action_proposal is None:
                raise ValueError("target-action CT requires the physical A0 proposal")
            if tuple(action_proposal.shape) != (
                expected_action[0],
                self.horizon,
                self.action_dim,
            ):
                raise ValueError("target-action CT A0 must be [B,T,A]")
            proposal_query = self.physical_action_input(action_proposal)[
                :, :, None
            ].expand(-1, -1, self.basis, -1)
            trajectory_source = action_query + proposal_query
        else:
            proposal_query = torch.zeros_like(action_query)
            trajectory_source = (
                action_query
                + plan.protected_base
                + plan.protected_policy_precision
            )
        trajectory, norm_denominator, norm_gain = (
            self.trajectory_norm.forward_with_denominator(trajectory_source)
        )
        action_tokens = trajectory.flatten(1, 2)
        context = self._context_tokens(seed, plan, action_proposal)
        transition = source.selector
        raw = self.v120_transition(
            transition,
            context,
            action_tokens=action_tokens,
            transition_tokens=transition,
        )
        primary_value = raw["rollout_delta_pred"]
        primary_action_coefficients = raw["rollout_action_coeff"]
        neutral_coefficients = raw["rollout_neutral_coeff"]
        mode = self._eval_intervention
        if mode == "none":
            # No identity arithmetic on the primary deployment path.
            value = primary_value
            action_coefficients = primary_action_coefficients
        else:
            if self.training:
                raise ValueError(
                    "controlled-transition interventions are evaluation-only"
                )
            if mode != "delta_neutral":
                raise RuntimeError("controlled-transition intervention state is invalid")
            # The full transition, context and action-token computations above
            # remain live.  Only the real-minus-learned-neutral delta is made
            # algebraically neutral; the protected G3 selector is retained.
            value = torch.zeros_like(primary_value)
            action_coefficients = neutral_coefficients
        spatial_attention: Tensor | None = None
        physical_delta: Tensor | None = None
        if self.target_action_bottleneck:
            if self.physical_delta_head is None:
                raise RuntimeError("target-action CT physical readout is missing")
            # Query rows retain the ODE-time/noisy-action seed, A0 and current
            # state.  Keys and centered values stay on the private G3 chart;
            # only the resulting 24-row physical field crosses the owner
            # boundary.
            state_query = self.trajectory_norm(seed.state).mean(
                dim=1,
                keepdim=True,
            )
            physical_query = trajectory.mean(dim=2) + state_query
            private_key = self.trajectory_norm(transition)
            spatial_logits = torch.einsum(
                "bth,bnh->btn",
                physical_query.float(),
                private_key.float(),
            ) / float(self.hidden) ** 0.5
            spatial_attention = torch.softmax(spatial_logits, dim=-1).to(
                dtype=value.dtype
            )
            physical_value = torch.einsum(
                "btn,bnh->bth",
                spatial_attention,
                value,
            )
            physical_delta = self.physical_delta_head(physical_value)
            result = PhysicalTransitionInnovation(delta_v=physical_delta)
            result.validate(
                horizon=self.horizon,
                physical_dim=self.physical_field_dim,
            )
        else:
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
            "controlled_transition_retained_rows": value.new_tensor(
                float(self.horizon if self.target_action_bottleneck else value.shape[1])
            ),
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
            "controlled_transition_policy_precision_rms": (
                plan.protected_policy_precision.detach()
                .float()
                .square()
                .mean()
                .sqrt()
            ),
            "controlled_transition_physical_proposal_query_rms": proposal_query.detach()
            .float()
            .square()
            .mean()
            .sqrt(),
            "controlled_transition_intervention_active": value.new_tensor(
                float(mode != "none"), dtype=torch.float32
            ),
            "controlled_transition_intervention_network_executed": value.new_ones(
                (), dtype=torch.float32
            ),
            "controlled_transition_intervention_first_boundary_delta_rms": (
                primary_value.detach().float() - value.detach().float()
            )
            .square()
            .mean()
            .sqrt(),
            "controlled_transition_public_physical_delta_rms": (
                value.new_zeros(())
                if physical_delta is None
                else physical_delta.detach().float().square().mean().sqrt()
            ),
            "controlled_transition_spatial_attention_entropy": (
                value.new_zeros(())
                if spatial_attention is None
                else -(
                    spatial_attention.detach().float()
                    * spatial_attention.detach().float().clamp_min(1e-8).log()
                )
                .sum(dim=-1)
                .mean()
            ),
            "controlled_transition_intervention_action_neutral_identity_max_abs": (
                action_coefficients.detach().float()
                - neutral_coefficients.detach().float()
            )
            .abs()
            .amax(),
            "controlled_transition_intervention_selector_identity_max_abs": (
                value.new_zeros(())
                if self.target_action_bottleneck
                else (
                    result.selector.detach().float()
                    - source.selector.detach().float()
                )
            )
            .abs()
            .amax(),
        }


__all__ = ["ControlledTransitionDynamics"]
