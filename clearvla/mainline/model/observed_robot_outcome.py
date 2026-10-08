"""S-owned measured robot transition, distinct from forecast innovation.

State features and recorded command keep their configured charts: no physical
meters, contact labels, progress scalar or predicted response is invented.
The supported current state distinguishes empty/persistent openings; the measured
change distinguishes a correctly predicted stall from correctly predicted motion.
Both may still be insufficient to identify the task object. This is a candidate
consumer requiring learned-behavior qualification, not a rule for phase changes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..robot_execution import RobotResponseFeedback


class ObservedRobotOutcomeRead(nn.Module):
    """Condition one actually observed transition on existing interval context.

    Observations are detached. Only this reader and the S interval context receive
    task gradients; the response predictor remains owned by its response loss.
    An observed zero transition has an availability witness, while a missing
    predecessor gives exact zero output and gradients. No K selection occurs.
    """

    def __init__(self, *, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (state_dim, action_dim, hidden)):
            raise ValueError("robot outcome dimensions must be positive integers")
        self.state_dim, self.action_dim, self.hidden = state_dim, action_dim, hidden
        # Allocate only in the new selected graph without shifting inherited RNG.
        # CPU construction followed by the caller's normal model.to() is retained.
        with torch.random.fork_rng(devices=[]):
            self.source = nn.Linear(2 * state_dim + action_dim + 1, hidden, bias=False)
            self.context = nn.Linear(hidden, hidden, bias=False)
            self.output = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, feedback: RobotResponseFeedback, context: Tensor) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.hidden or not context.is_floating_point():
            raise ValueError("robot outcome requires interval context [B,I,H]")
        batch = context.shape[0]
        feedback.validate(batch=batch, state_dim=self.state_dim, device=context.device)
        step, current = feedback.source_step, feedback.current_state
        if step is None or current is None or feedback.observed_delta is None:
            raise ValueError("robot outcome requires its measured transition sources")
        step.validate(
            batch=batch,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=context.device,
            strict=True,
        )
        if feedback.observed is not step.observed:
            raise ValueError("robot outcome support must be owned by its executed step")
        if (
            current.shape != (batch, self.state_dim)
            or current.device != context.device
            or not current.is_floating_point()
        ):
            raise ValueError("robot outcome current state has another source chart")
        valid = feedback.observed[:, None]
        now = torch.where(valid, current.detach().float(), 0.0)
        before = torch.where(valid, step.previous_state.detach().float(), 0.0)
        command = torch.where(valid, step.command.detach().float(), 0.0)
        delta = torch.where(valid, feedback.observed_delta.detach().float(), 0.0)
        if not all(bool(torch.isfinite(v).all()) for v in (now, before, command, delta)):
            raise ValueError("robot outcome has nonfinite observed values")
        if not torch.allclose(delta, now - before, atol=2e-6, rtol=1e-5):
            raise ValueError("robot outcome differs from the actual measured state change")
        query = torch.where(valid[:, None], context, 0.0)
        if not bool(torch.isfinite(query).all()):
            raise ValueError("robot outcome has nonfinite supported interval context")
        # The availability witness is a source fact, not success/confidence.
        values = torch.cat((now, command, delta, valid.to(now.dtype)), -1)
        dtype = self.source.weight.dtype
        source = self.source(values.to(dtype))[:, None]
        update = self.output(source * (1.0 + torch.tanh(self.context(query.to(dtype)))))
        return torch.where(valid[:, None], update, 0.0)
