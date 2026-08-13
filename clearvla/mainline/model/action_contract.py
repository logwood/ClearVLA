"""Typed online action boundary for the restored V120 bottom.

No decoder or alternative bottom implementation belongs here.  The module
owns only the shared causal-state canonicalization, noisy-action query and
typed output record used by P2/P3 and the extracted V120 Evidence-MMDiT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..interfaces import ObservableHistory
from ..v120_core.codec import PhysicalActionTokenLift
from ..v120_core.primitives import sinusoidal_positions as v120_sinusoidal_positions
from ..v120_core.trunk_primitives import HorizonRoleEmbedding


def canonical_state_history(history: ObservableHistory) -> Tensor:
    """Return one causal sequence whose final row is the current state."""

    value = history.state_history
    if value.ndim != 3 or int(value.shape[1]) < 1:
        raise ValueError("state history must contain at least one causal row")
    if int(value.shape[0]) != int(history.state.shape[0]) or int(value.shape[2]) != int(
        history.state.shape[1]
    ):
        raise ValueError("state history and current state do not align")
    if int(value.shape[1]) == 1:
        return history.state[:, None]
    return torch.cat((value[:, :-1], history.state[:, None]), dim=1)


def sinusoidal_positions(length: int, width: int, *, device: torch.device) -> Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = max(width // 2, 1)
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / float(max(half - 1, 1))
    )[None]
    value = torch.cat((torch.sin(position * frequency), torch.cos(position * frequency)), dim=-1)
    if int(value.shape[-1]) < width:
        value = F.pad(value, (0, width - int(value.shape[-1])))
    return value[:, :width]


class TimeCondition(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(hidden, 4 * hidden, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden, hidden, bias=False),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("flow time must be [B]")
        half = max(self.hidden // 2, 1)
        frequency = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=time.device, dtype=torch.float32)
            / float(max(half - 1, 1))
        )
        angle = time.float()[:, None] * frequency[None]
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        if int(embedding.shape[-1]) < self.hidden:
            embedding = F.pad(embedding, (0, self.hidden - int(embedding.shape[-1])))
        network_dtype = next(self.network.parameters()).dtype
        return self.network(embedding[:, : self.hidden].to(dtype=network_dtype))


@dataclass(frozen=True)
class V120SeedContext:
    """Observable canvas rows created beside the noisy-action seed.

    V120 used one ``UnifiedCanvasSeed`` for the noisy trajectory, current
    state, causal state history and compressed executed-action history.  These
    rows are consumed by both the controlled transition and the final evidence
    decoder, so giving each consumer an unrelated projection changes the
    input distribution and breaks the original shared-gradient geometry.
    """

    state: Tensor  # [B,1,H]
    state_history: Tensor  # [B,3,H]
    executed: Tensor  # [B,7,H]

    def validate(self, *, hidden: int, state_history: int, executed: int) -> None:
        if self.state.ndim != 3:
            raise ValueError("V120 seed state must be [B,1,H]")
        batch = int(self.state.shape[0])
        expected = {
            "state": (batch, 1, int(hidden)),
            "state_history": (batch, int(state_history), int(hidden)),
            "executed": (batch, int(executed), int(hidden)),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"V120 seed {name} must be {shape}")


class ActionQueryEncoder(nn.Module):
    """Recovered V120 action and observable-context canvas seed.

    Only the rows that are active in the object mainline are materialized.
    They retain the exact V120 role identities and exact-null executed-history
    transform, while dead task/proposal/register canvas rows stay absent.
    """

    ROLE_TASK = 0
    ROLE_STATE = 1
    ROLE_STATE_HISTORY = 2
    ROLE_EXECUTED = 3
    ROLE_PROPOSAL = 4
    ROLE_NOISY_ACTION = 5
    ROLE_ROLLOUT = 6
    ROLE_REGISTER = 7

    def __init__(self, core_config) -> None:
        super().__init__()
        self.action_dim = int(core_config.physical_action_dim)
        self.hidden = int(core_config.hidden_size)
        self.horizon = int(core_config.action_horizon)
        self.basis = int(core_config.action_basis_tokens)
        self.state_dim = int(core_config.state_dim)
        self.state_history_rows = int(core_config.visual_history_length)
        self.executed_rows = int(core_config.action_history_token_count)
        self.future_anchors = int(core_config.future_anchors)
        self.cameras = int(core_config.num_cameras)
        self.future_grid = int(core_config.future_grid_size)
        self.canvas_registers = int(core_config.canvas_registers)
        self.state_projection = nn.Linear(self.state_dim, self.hidden)
        self.state_history_projection = nn.Linear(self.state_dim, self.hidden)
        self.physical_lift = PhysicalActionTokenLift(core_config)
        self.horizon_role = HorizonRoleEmbedding(core_config)
        self.basis_identity = nn.Parameter(
            torch.randn(1, 1, self.basis, self.hidden) * 0.02
        )
        self.role_embed = nn.Parameter(torch.randn(8, self.hidden) * 0.02)
        self.role_drop = nn.Dropout(float(core_config.role_dropout))
        self.canvas_drop = nn.Dropout(float(core_config.canvas_dropout))
        self.rollout_anchor_type = nn.Parameter(
            torch.randn(1, self.future_anchors, 1, self.hidden) * 0.02
        )
        self.rollout_grid_type = nn.Parameter(
            torch.randn(
                1,
                1,
                self.cameras * self.future_grid * self.future_grid,
                self.hidden,
            )
            * 0.02
        )
        self.registers = nn.Parameter(
            torch.randn(1, self.canvas_registers, self.hidden) * 0.02
        )
        self.action_private_condition = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
        )
        self.shared_condition_mixer = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, 2 * self.hidden),
            nn.SiLU(),
            nn.Linear(2 * self.hidden, self.hidden),
        )
        # V120 exact-null semantics evaluate f(x)-f(0); this affine bias
        # cancels identically and was therefore frozen in the source model.
        final_mixer = self.shared_condition_mixer[-1]
        if not isinstance(final_mixer, nn.Linear):
            raise TypeError("V120 shared condition mixer must end in Linear")
        if final_mixer.bias is not None:
            final_mixer.bias.requires_grad_(False)
        self.state_history_identity = nn.Parameter(
            torch.randn(1, self.state_history_rows, self.hidden) * 0.02
        )
        self.register_buffer(
            "horizon_position",
            v120_sinusoidal_positions(
                range(1, self.horizon + 1), self.hidden
            )[None],
            persistent=True,
        )

    def clean_action_basis_tokens(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """V120 action-basis identity without noisy physical coordinates."""

        if int(batch) < 1:
            raise ValueError("clean action basis requires a positive batch")
        horizon = (
            self.horizon_position.to(device=device, dtype=dtype)
            + self.horizon_role(int(batch), device=device, dtype=dtype)
            + self.role_embed[self.ROLE_NOISY_ACTION]
            .to(device=device, dtype=dtype)
            .reshape(1, 1, -1)
        )
        return horizon[:, :, None] + self.basis_identity.to(
            device=device, dtype=dtype
        )

    def sample_role_table(self, reference: Tensor) -> Tensor:
        """Sample the one V120 role table shared by static G and live action.

        Role dropout is a canvas-level condition. Sampling it independently
        inside G and at every ODE node changes the input distribution and
        defeats the original shared-seed gradient geometry.
        """

        return self.role_drop(
            self.role_embed.to(device=reference.device, dtype=reference.dtype)
        )

    def grounding_canvas(
        self,
        *,
        state: Tensor,
        rollout_init: Tensor,
        role: Tensor,
    ) -> tuple[Tensor, dict[str, slice]]:
        """Build the observation-only V120 G canvas from the shared seed.

        G sees current state, registers and visual rollout only. Empty slices
        make task/history/proposal/noisy-action absence structural instead of
        relying on a learned zero or a later attention mask.
        """

        if state.ndim != 2 or int(state.shape[-1]) != self.state_dim:
            raise ValueError("grounding state must be [B,state_dim]")
        batch = int(state.shape[0])
        future_tokens = self.future_anchors * self.cameras * self.future_grid**2
        if tuple(rollout_init.shape) != (batch, future_tokens, self.hidden):
            raise ValueError(
                "grounding rollout must preserve anchor/camera/spatial identity"
            )
        if tuple(role.shape) != (8, self.hidden):
            raise ValueError("shared V120 role table must be [8,H]")
        device, dtype = rollout_init.device, rollout_init.dtype
        state_token = (
            self.state_projection(state.to(device=device, dtype=dtype))[:, None]
            + role[self.ROLE_STATE]
        )
        spatial = self.cameras * self.future_grid * self.future_grid
        rollout = rollout_init.reshape(
            batch, self.future_anchors, spatial, self.hidden
        )
        rollout = (
            rollout
            + self.rollout_anchor_type.to(device=device, dtype=dtype)
            + self.rollout_grid_type.to(device=device, dtype=dtype)
        ).reshape(batch, future_tokens, self.hidden)
        rollout = rollout + role[self.ROLE_ROLLOUT]
        registers = (
            self.registers.expand(batch, -1, -1).to(device=device, dtype=dtype)
            + role[self.ROLE_REGISTER]
        )
        empty = rollout.new_empty(batch, 0, self.hidden)
        named = (
            ("state", state_token),
            ("state_history", empty),
            ("task", empty),
            ("executed", empty),
            ("proposal", empty),
            ("trajectory", empty),
            ("stage", empty),
            ("rollout", rollout),
            ("registers", registers),
        )
        offset = 0
        slices: dict[str, slice] = {}
        rows: list[Tensor] = []
        for name, value in named:
            rows.append(value)
            slices[name] = slice(offset, offset + int(value.shape[1]))
            offset += int(value.shape[1])
        return self.canvas_drop(torch.cat(rows, dim=1)), slices

    def _action_from_role(self, noisy_action_field: Tensor, role: Tensor) -> Tensor:
        batch = int(noisy_action_field.shape[0])
        action = (
            self.physical_lift(noisy_action_field)
            + self.horizon_position.to(
                device=noisy_action_field.device, dtype=noisy_action_field.dtype
            )
            + self.horizon_role(
                batch,
                device=noisy_action_field.device,
                dtype=noisy_action_field.dtype,
            )
            + role[self.ROLE_NOISY_ACTION]
        )
        return action[:, :, None] + self.basis_identity.to(
            device=action.device, dtype=action.dtype
        )

    def _context_from_role(
        self,
        history: ObservableHistory,
        *,
        executed_memory: Tensor,
        action_history_keep: Tensor,
        role: Tensor,
    ) -> V120SeedContext:
        batch = int(history.state.shape[0])
        expected_memory = (batch, self.executed_rows, self.hidden)
        if tuple(executed_memory.shape) != expected_memory:
            raise ValueError(
                f"V120 compressed action history must be {expected_memory}"
            )
        if tuple(action_history_keep.shape) != (batch,):
            raise ValueError("V120 action-history keep mask must be [B]")
        device = role.device
        dtype = role.dtype
        state = (
            self.state_projection(history.state.to(device=device, dtype=dtype))[:, None]
            + role[self.ROLE_STATE]
        )
        state_history = (
            self.state_history_projection(
                history.state_history.to(device=device, dtype=dtype)
            )
            + self.state_history_identity.to(device=device, dtype=dtype)
            + role[self.ROLE_STATE_HISTORY]
        )
        memory = executed_memory.to(device=device, dtype=dtype)
        conditioned = self.shared_condition_mixer(
            self.action_private_condition(memory)
        )
        null = self.shared_condition_mixer(
            self.action_private_condition(torch.zeros_like(memory))
        )
        executed = (
            (conditioned - null)
            * action_history_keep.to(device=device, dtype=dtype)[:, None, None]
            + role[self.ROLE_EXECUTED]
        )
        context = V120SeedContext(
            state=state,
            state_history=state_history,
            executed=executed,
        )
        context.validate(
            hidden=self.hidden,
            state_history=self.state_history_rows,
            executed=self.executed_rows,
        )
        return context

    def forward_with_context(
        self,
        noisy_action_field: Tensor,
        time: Tensor,
        history: ObservableHistory,
        *,
        executed_memory: Tensor,
        action_history_keep: Tensor,
        role: Tensor | None = None,
    ) -> tuple[Tensor, V120SeedContext]:
        """Build action and context with one shared V120 role-drop sample."""

        if tuple(noisy_action_field.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("noisy physical action field must be [B,T,Aphysical]")
        batch = int(noisy_action_field.shape[0])
        if tuple(time.shape) != (batch,):
            raise ValueError("flow time and noisy action batch do not align")
        if int(history.state.shape[0]) != batch:
            raise ValueError("V120 action and context batches do not align")
        del time
        role = self.sample_role_table(noisy_action_field) if role is None else role
        if tuple(role.shape) != (8, self.hidden):
            raise ValueError("shared V120 role table must be [8,H]")
        return (
            self._action_from_role(noisy_action_field, role),
            self._context_from_role(
                history,
                executed_memory=executed_memory,
                action_history_keep=action_history_keep,
                role=role,
            ),
        )

    def forward(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        if tuple(noisy_action_field.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("noisy physical action field must be [B,T,Aphysical]")
        batch = int(noisy_action_field.shape[0])
        if tuple(time.shape) != (batch,):
            raise ValueError("flow time and noisy action batch do not align")
        # V120 injected time through the downstream modulated blocks and
        # decoder, not by replacing the typed physical seed with one generic
        # affine action+time sum.  The physical flow state itself remains the
        # only action value here; ``time`` is still checked and is consumed by
        # the restored bottom decoder.
        del time
        role = self.role_drop(
            self.role_embed.to(
                device=noisy_action_field.device, dtype=noisy_action_field.dtype
            )
        )
        return self._action_from_role(noisy_action_field, role)

@dataclass(frozen=True)
class BottomOutput:
    physical_velocity: Tensor
    event_logits: Tensor
    motion_logits: Tensor
    action_query: Tensor
    block_updates: tuple[Tensor, ...]
    evidence_tokens: Tensor
    decoder_tensors: dict[str, Tensor] = field(default_factory=lambda: {})

    def validate(self, *, action_dim: int, horizon: int, basis: int, hidden: int) -> None:
        batch = int(self.physical_velocity.shape[0])
        if tuple(self.physical_velocity.shape) != (batch, horizon, action_dim):
            raise ValueError("bottom physical velocity has an invalid shape")
        if tuple(self.event_logits.shape) != (batch, horizon, 3):
            raise ValueError("bottom event logits have an invalid shape")
        if tuple(self.motion_logits.shape) != (batch, horizon):
            raise ValueError("bottom motion logits have an invalid shape")
        if tuple(self.action_query.shape) != (batch, horizon, basis, hidden):
            raise ValueError("bottom action query lost its basis axis")


__all__ = [
    "ActionQueryEncoder",
    "BottomOutput",
    "V120SeedContext",
    "canonical_state_history",
]
