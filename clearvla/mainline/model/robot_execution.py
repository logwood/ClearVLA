"""One causal robot response model and its distinct P3 innovation read.

No current visual/goal input enters the predictor. Policy loss cannot train it
to fabricate a useful latent error: its output is detached before P3 reads it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..robot_execution import ExecutedRobotStep, RobotResponseFeedback


class RobotExecutionObserver(nn.Module):
    def __init__(self, *, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        if min(state_dim, action_dim, hidden) <= 0:
            raise ValueError("robot response dimensions must be positive")
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        # No LayerNorm on feature values: absolute proprioception and command
        # magnitudes must survive. Displacements remain feature differences.
        self.response = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, state_dim)
        )
        self.error_value = nn.Linear(state_dim, hidden, bias=False)
        self.plan_query = nn.Linear(hidden, hidden, bias=False)
        self.error_output = nn.Linear(hidden, hidden, bias=False)

    def observe(
        self, step: ExecutedRobotStep, current_state: Tensor
    ) -> tuple[RobotResponseFeedback, Tensor]:
        batch = current_state.shape[0]
        step.validate(
            batch=batch,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=current_state.device,
        )
        if current_state.shape != (batch, self.state_dim) or not current_state.is_floating_point():
            raise ValueError("robot response current state has wrong feature chart")
        valid = step.observed[:, None]
        # Sanitize BEFORE projections/products and detach source observations.
        previous = torch.where(valid, step.previous_state.detach().float(), 0.0)
        command = torch.where(valid, step.command.detach().float(), 0.0)
        current = torch.where(valid, current_state.detach().float(), 0.0)
        # FP32 input follows parameter dtype outside autocast, as other modules.
        prediction = self.response(
            torch.cat((previous, command), dim=-1).to(self.error_value.weight.dtype)
        )
        prediction = torch.where(valid, prediction.float(), 0.0)
        observed_delta = current - previous
        squared = F.mse_loss(prediction, observed_delta, reduction="none")
        loss = squared.sum() / (step.observed.float().sum().clamp_min(1) * self.state_dim)
        innovation = torch.where(valid, observed_delta - prediction.detach(), 0.0).detach()
        return RobotResponseFeedback(innovation, step.observed, step, current_state), loss

    def read(self, feedback: RobotResponseFeedback, query: Tensor) -> Tensor:
        if query.ndim != 4:
            raise ValueError("robot response read requires [B,T,Q,H]")
        feedback.validate(batch=query.shape[0], state_dim=self.state_dim, device=query.device)
        error = torch.where(feedback.observed[:, None], feedback.innovation, 0.0)
        value = self.error_value(error.to(self.error_value.weight.dtype))[:, None, None]
        # Exact zero input gives zero value AND zero plan-context derivative.
        return self.error_output(value * torch.tanh(self.plan_query(query)))
