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


class ActionQueryEncoder(nn.Module):
    """Create the sole noisy physical-field query shared by P2/P3/bottom."""

    def __init__(self, *, action_dim: int, hidden: int, horizon: int, basis: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        self.horizon = int(horizon)
        self.basis = int(basis)
        self.action = nn.Linear(action_dim, hidden, bias=False)
        self.time = TimeCondition(hidden)
        self.basis_identity = nn.Parameter(torch.randn(1, 1, basis, hidden) * 0.02)
        self.register_buffer(
            "horizon_position",
            sinusoidal_positions(horizon, hidden, device=torch.device("cpu"))[None, :, None],
            persistent=True,
        )

    def forward(self, noisy_action_field: Tensor, time: Tensor) -> Tensor:
        if tuple(noisy_action_field.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("noisy physical action field must be [B,T,Aphysical]")
        batch = int(noisy_action_field.shape[0])
        if tuple(time.shape) != (batch,):
            raise ValueError("flow time and noisy action batch do not align")
        action = self.action(noisy_action_field)[:, :, None]
        return (
            action
            + self.time(time).to(dtype=action.dtype)[:, None, None]
            + self.horizon_position.to(device=action.device, dtype=action.dtype)
            + self.basis_identity.to(device=action.device, dtype=action.dtype)
        )


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


__all__ = ["ActionQueryEncoder", "BottomOutput", "canonical_state_history"]
