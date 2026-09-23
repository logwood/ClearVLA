"""Source-typed low-rank decoder conditioning on the completed G3 chart.

This module never sees physical future labels. It reads a CURRENT noisy plan,
not actions claimed to have happened. Context can select values but cannot
manufacture a dynamic value when all dynamic sources are zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from .action_contract import V120SeedContext, sinusoidal_positions
from .compiler import ObjectConsequenceState, ObjectPolicyPlanDeltaBank

DYNAMIC_SOURCES = (
    "action",
    "precision",
    "semantic",
    "geometry",
    "semantic_interaction",
    "geometry_interaction",
    "temporal",
    "change",
)
CONTEXT_SOURCES = ("fact", "state", "state_history", "executed")


@dataclass(frozen=True)
class TransitionPlanEvidence:
    """Exact upstream owners; no summed protected feature masquerades as fact."""

    action: Tensor
    consequence: ObjectConsequenceState
    plan: ObjectPolicyPlanDeltaBank
    seed: V120SeedContext

    def validate(self, *, hidden: int, horizon: int, basis: int) -> None:
        if self.action.ndim != 4 or tuple(self.action.shape[1:]) != (horizon, basis, hidden):
            raise ValueError("typed transition action must preserve [B,T,Q,H]")
        self.consequence.validate()
        self.plan.validate()
        self.seed.validate(hidden=hidden, state_history=3, executed=7)
        if self.plan.protected_base is not self.consequence.protected_consequence:
            raise ValueError("typed transition requires the same P2/P3 consequence owner")
        expected = self.action.shape
        for name, value in self.dynamic().items():
            if (
                value.shape != expected
                or value.device != self.action.device
                or not value.is_floating_point()
            ):
                raise ValueError(f"typed transition {name} shape/device/dtype mismatch")
        if self.consequence.factual_base.shape != expected:
            raise ValueError("typed transition current fact lost its control row")
        for name, value in self.context().items():
            if (
                value.shape[0] != expected[0]
                or value.device != self.action.device
                or not value.is_floating_point()
            ):
                raise ValueError(f"typed transition {name} context source mismatch")

    def dynamic(self) -> dict[str, Tensor]:
        return {
            "action": self.action,
            "precision": self.plan.protected_policy_precision,
            "semantic": self.consequence.effect.semantic,
            "geometry": self.consequence.effect.geometry,
            "semantic_interaction": self.consequence.interaction.semantic,
            "geometry_interaction": self.consequence.interaction.geometry,
            "temporal": self.plan.temporal,
            "change": self.plan.state_change,
        }

    def context(self) -> dict[str, Tensor]:
        return {
            "fact": self.consequence.factual_base.flatten(1, 2),
            "state": self.seed.state,
            "state_history": self.seed.state_history,
            "executed": self.seed.executed,
        }


class TypedPlanTransition(nn.Module):
    """Two-stage value-preserving attention and source-aligned low-rank write.

    All candidate rows may interact: these are plan features from one causal
    observation, not future observations. Source/type/row codes affect K, never V.
    The original P3 direct bottom lanes remain independently owned.
    """

    row_address: Tensor

    def __init__(
        self, *, hidden: int, heads: int, horizon: int, basis: int, rank: int, action_tokens: int
    ) -> None:
        super().__init__()
        if (
            hidden < 1
            or heads < 1
            or hidden % heads
            or min(horizon, basis, rank, action_tokens) < 1
        ):
            raise ValueError("typed transition dimensions must be positive and head aligned")
        self.hidden, self.horizon, self.basis, self.rank = hidden, horizon, basis, rank
        # Source projections define different value charts before any summation.
        self.value_projections = nn.ModuleDict(
            {n: nn.Linear(hidden, hidden, bias=False) for n in DYNAMIC_SOURCES}
        )
        self.context_projections = nn.ModuleDict(
            {n: nn.Linear(hidden, hidden, bias=False) for n in CONTEXT_SOURCES}
        )
        self.source_identity = nn.Parameter(torch.randn(len(DYNAMIC_SOURCES), hidden) * 0.02)
        self.context_identity = nn.Parameter(torch.randn(len(CONTEXT_SOURCES), hidden) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(1, action_tokens, hidden) * 0.02)
        self.selector_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.context_read = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.action_read = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.coefficient_read = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, bias=False, batch_first=True
        )
        self.coefficient_head = nn.Linear(hidden, rank, bias=False)
        self.basis_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, rank * hidden),
        )
        basis_out = cast(nn.Linear, self.basis_head[-1])
        nn.init.normal_(basis_out.weight, mean=0.0, std=0.03)
        if basis_out.bias is not None:
            nn.init.zeros_(basis_out.bias)
        self.delta_gain = nn.Parameter(torch.tensor(1.0))
        # Only addresses receive row/basis identities. These are not value content.
        if hidden < 2:
            raise ValueError("typed transition needs separate row/basis address coordinates")
        row = sinusoidal_positions(horizon, hidden // 2, device=torch.device("cpu"))[
            :, None
        ].expand(-1, basis, -1)
        basis_code = sinusoidal_positions(basis, hidden - hidden // 2, device=torch.device("cpu"))[
            None
        ].expand(horizon, -1, -1)
        self.register_buffer(
            "row_address",
            torch.cat((row, basis_code), -1).reshape(1, horizon * basis, hidden),
            persistent=False,
        )

    def value_bank(self, evidence: TransitionPlanEvidence) -> tuple[Tensor, Tensor]:
        evidence.validate(hidden=self.hidden, horizon=self.horizon, basis=self.basis)
        keys, values = [], []
        for i, (name, source) in enumerate(evidence.dynamic().items()):
            projection = cast(nn.Linear, self.value_projections[name])
            value = projection(source.to(dtype=projection.weight.dtype).flatten(1, 2))
            key = self.selector_norm(value) + self.source_identity[i].to(dtype=value.dtype)
            key = key + self.row_address.to(device=value.device, dtype=value.dtype)
            keys.append(key)
            values.append(value)
        return torch.cat(keys, 1), torch.cat(values, 1)

    def forward(self, source: Tensor, evidence: TransitionPlanEvidence) -> dict[str, Tensor]:
        evidence.validate(hidden=self.hidden, horizon=self.horizon, basis=self.basis)
        if (
            source.shape != (evidence.action.shape[0], 512, self.hidden)
            or source.device != evidence.action.device
        ):
            raise ValueError("typed transition must retain the complete G3 source chart")
        # Context is deliberately read BEFORE dynamic values; neither value-bank
        # projection nor coefficient V includes current fact/state as an additive shortcut.
        context_keys, context_values = [], []
        for i, (name, x) in enumerate(evidence.context().items()):
            projection = cast(nn.Linear, self.context_projections[name])
            value = projection(x.to(dtype=projection.weight.dtype))
            context_values.append(value)
            context_keys.append(
                self.selector_norm(value) + self.context_identity[i].to(dtype=value.dtype)
            )
        ck, cv = torch.cat(context_keys, 1), torch.cat(context_values, 1)
        queries = self.action_queries.expand(source.shape[0], -1, -1).to(dtype=cv.dtype)
        context, _ = self.context_read(queries, ck, cv, need_weights=False)
        dynamic_keys, dynamic_values = self.value_bank(evidence)
        latent, _ = self.action_read(
            queries + context, dynamic_keys, dynamic_values, need_weights=False
        )
        latent_keys = self.selector_norm(latent) + queries + context
        # No LayerNorm on latent V or coefficient values: zero dynamic source
        # remains exactly zero, and small signals are not promoted to unit size.
        q = self.selector_norm(source).to(dtype=latent.dtype)
        read, _ = self.coefficient_read(q, latent_keys, latent, need_weights=False)
        coeff = torch.tanh(self.coefficient_head(read))
        basis = self.basis_head(source).reshape(source.shape[0], 512, self.rank, self.hidden)
        value = torch.einsum("bnr,bnrh->bnh", coeff, basis) / math.sqrt(self.rank)
        value = value * self.delta_gain.to(dtype=value.dtype)
        return {
            "rollout_delta_pred": value,
            "rollout_action_coeff": coeff,
            "rollout_neutral_coeff": torch.zeros_like(coeff),
            "rollout_transition_basis": basis,
            "latent_action_tokens": latent,
            "typed_dynamic_rows": value.new_tensor(dynamic_values.shape[1]),
            "typed_context_rows": value.new_tensor(cv.shape[1]),
        }
