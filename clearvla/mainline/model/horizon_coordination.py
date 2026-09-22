"""P3 coordination of an entire proposed horizon using separately typed facts.

All rows here are candidate PLAN features computed from the current observation,
not future observed values. Bidirectional row attention therefore is deliberate.
No executed-prefix error, task completion signal or persistent memory is inferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time


@dataclass(frozen=True)
class P3HorizonContext:
    """One immutable observation/plan snapshot with named computational owners.

    The six plan sources are [B,T,Q,H]. Observed change is [B,H]. Semantic and
    geometry are already hidden effects, not endpoint states or physical xy.
    Their source identity is retained until distinct learned projections.
    """

    action: Tensor
    current_fact: Tensor
    policy_precision: Tensor
    semantic_effect: Tensor
    geometry_feature_effect: Tensor
    task_temporal: Tensor
    observed_change: Tensor
    time_grid_mode: str

    def validate(self, *, hidden: int, horizon: int, basis: int) -> None:
        grid = resolve_future_time(self.time_grid_mode)
        if not grid.aligned or grid.horizon != horizon:
            raise ValueError("P3 horizon context needs the declared physical control grid")
        if self.action.ndim != 4:
            raise ValueError("P3 action context must be [B,T,Q,H]")
        shape = (self.action.shape[0], horizon, basis, hidden)
        if min(shape) < 1:
            raise ValueError("P3 context axes must be nonempty")
        for name in TypedHorizonCoordinator.SOURCE_NAMES:
            value = getattr(self, name)
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"P3 {name} must be floating [B,T,Q,H]")
            if value.device != self.action.device:
                raise ValueError(f"P3 {name} lost observation device identity")
        if (
            self.observed_change.shape != (shape[0], hidden)
            or not self.observed_change.is_floating_point()
            or self.observed_change.device != self.action.device
        ):
            raise ValueError("P3 observed change must be floating [B,H] on the same device")
        # Values have already been admitted by their source. Do not add tensor
        # reductions/host synchronization to every existing ODE evaluation.


class TypedHorizonCoordinator(nn.Module):
    """Plan-wide context read with protected carriers kept outside this module.

    Address normalization does not erase source amplitude in the value stream.
    Physical time enters Q/K only; it cannot create an observed-change value.
    There are no inactive legacy P3 parameters, hidden caches or stochastic
    diagnostics. The original two optional output lanes are retained.
    """

    control_midpoints: Tensor
    control_time_code: Tensor

    SOURCE_NAMES = (
        "action", "current_fact", "policy_precision", "semantic_effect",
        "geometry_feature_effect", "task_temporal",
    )

    def __init__(self, *, hidden: int, horizon: int, basis: int, heads: int) -> None:
        super().__init__()
        if hidden < 1 or heads < 1 or hidden % heads or basis < 1:
            raise ValueError("invalid P3 horizon dimensions or attention heads")
        grid = resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME)
        if horizon != grid.horizon:
            raise ValueError("P3 horizon must match its physical time grid")
        self.hidden, self.horizon, self.basis = int(hidden), int(horizon), int(basis)
        self.sources = nn.ModuleDict({
            name: nn.Linear(hidden, hidden, bias=False) for name in self.SOURCE_NAMES
        })
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.key_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.horizon_read = nn.MultiheadAttention(hidden, heads, dropout=0.0, bias=False, batch_first=True)
        self.plan_output = nn.Linear(hidden, hidden, bias=False)
        self.change_value = nn.Linear(hidden, hidden, bias=False)
        self.change_context = nn.Linear(hidden, hidden, bias=False)
        self.change_output = nn.Linear(hidden, hidden, bias=False)
        # Use the same 24-control-step physical unit as S and W. Q bases share
        # a time coordinate; their content still differs and is not averaged.
        times = (torch.arange(horizon, dtype=torch.float32) + 0.5) / grid.horizon
        count = (hidden + 1) // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(count).float() / count)
        angles = 2.0 * math.pi * times[:, None] * frequencies[None]
        code = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :hidden]
        self.register_buffer("control_midpoints", times, persistent=True)
        self.register_buffer("control_time_code", code, persistent=True)

    def forward(self, inputs: P3HorizonContext) -> tuple[Tensor, Tensor, Tensor]:
        inputs.validate(hidden=self.hidden, horizon=self.horizon, basis=self.basis)
        # Separate projections BEFORE fusion: opposite semantic and geometry
        # effects do not undergo a compulsory raw hidden-vector cancellation.
        projected = [self.sources[name](getattr(inputs, name)) for name in self.SOURCE_NAMES]
        context = torch.stack(projected, dim=0).sum(dim=0)
        batch = context.shape[0]
        values = context.reshape(batch, self.horizon * self.basis, self.hidden)
        time_code = self.control_time_code.to(values)[None, :, None].expand(
            batch, -1, self.basis, -1
        ).reshape_as(values)
        query = self.query_norm(values) + time_code
        key = self.key_norm(values) + time_code
        related, _ = self.horizon_read(query, key, values, need_weights=False)
        # Keep the local term as well as the horizon context; no early average
        # of basis queries or of all action rows is substituted for their values.
        coordinated = (values + related).reshape_as(context)
        temporal = self.plan_output(F.silu(coordinated))
        observed = self.change_value(inputs.observed_change)[:, None, None]
        # This lane is exactly zero without observed change, including its
        # gradient to plan/time context. It is not a W prediction residual.
        state_change = self.change_output(
            observed * (1.0 + torch.tanh(self.change_context(coordinated)))
        )
        return temporal, state_change, coordinated
