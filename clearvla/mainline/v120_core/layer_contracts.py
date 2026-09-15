"""The two active terminal policy layer contracts extracted from V120.

The reference model serialized eight heads for ancestry compatibility, but
strict 3-2-3 ownership froze G/W heads, skipped P3, and exposed only the final
two policy-depth adapters to Evidence-MMDiT. This module keeps their live
rollout/state/event math without materializing unread trajectory probes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import V39PolicyConfig


def _align_milestones(
    tokens: Tensor,
    horizon: int,
    *,
    boundaries: tuple[int, ...],
) -> Tensor:
    if tokens.ndim != 3 or int(tokens.shape[1]) != len(boundaries):
        raise ValueError("milestone tokens and boundaries do not align")
    rows: list[Tensor] = []
    lower = 0
    for index, upper in enumerate(boundaries):
        upper = int(upper)
        if upper <= lower or upper > int(horizon):
            raise ValueError("V120 milestone boundaries are invalid")
        rows.append(tokens[:, index : index + 1].expand(-1, upper - lower, -1))
        lower = upper
    if lower != int(horizon):
        raise ValueError("V120 milestones do not cover the action horizon")
    return torch.cat(rows, dim=1)


def rollout_tokens_to_action_horizon(
    tokens: Tensor,
    config: V39PolicyConfig,
) -> Tensor:
    """Pool the protected G3 chart per anchor and align it to action time."""

    if tokens.ndim != 3:
        raise ValueError("rollout tokens must be [B,F*G,H]")
    grid = (
        int(config.num_cameras)
        * int(config.future_grid_size)
        * int(config.future_grid_size)
    )
    expected = int(config.future_anchors) * grid
    if int(tokens.shape[1]) != expected:
        raise ValueError(
            f"rollout token count must be future_anchors*grid={expected}"
        )
    milestones = tokens.reshape(
        int(tokens.shape[0]),
        int(config.future_anchors),
        grid,
        int(tokens.shape[-1]),
    ).mean(dim=2)
    boundaries = tuple(int(value) for value in config.flow_jepa_action_offsets)
    return _align_milestones(
        milestones[:, : len(boundaries)],
        int(config.action_horizon),
        boundaries=boundaries,
    )


class MidcutContractHeads(nn.Module):
    """Live V120 rollout/state/event readouts behind each layer adapter."""

    def __init__(self, config: V39PolicyConfig) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_size)
        # Construct the removed frozen trajectory heads in their historical
        # positions so every retained readout and downstream module keeps the
        # R1g fresh-run initialization stream. They are temporary unregistered
        # objects and own no runtime or checkpoint state.
        historical_action_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, int(config.physical_action_dim)),
        )
        self.event_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        historical_motion_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.rollout_effect_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        )
        self.rollout_delta_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        )
        self.transition_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden)
        )
        self.future_gain = nn.Parameter(
            torch.tensor(float(config.midcut_future_gain_init), dtype=torch.float32)
        )
        for module in (
            historical_action_head[-1],
            self.event_head[-1],
            historical_motion_head[-1],
        ):
            if not isinstance(module, nn.Linear):
                raise TypeError("V120 contract readout must end in Linear")
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        for module in (
            self.rollout_effect_head[-1],
            self.rollout_delta_head[-1],
            self.transition_head[-1],
        ):
            if not isinstance(module, nn.Linear):
                raise TypeError("V120 contract readout must end in Linear")
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        del historical_action_head, historical_motion_head

    def forward(
        self,
        canvas: Tensor,
        slices: dict[str, slice],
    ) -> dict[str, Tensor]:
        config = self.config
        rollout = canvas[:, slices["rollout"]]
        registers = canvas[:, slices["registers"]]
        gain = self.future_gain.to(device=canvas.device, dtype=canvas.dtype)
        effect = self.rollout_effect_head(rollout) * gain
        delta = self.rollout_delta_head(rollout) * gain
        event_context = rollout_tokens_to_action_horizon(delta, config)
        transition = self.transition_head(delta.mean(dim=1, keepdim=True)).expand(
            -1, int(config.action_horizon), -1
        )
        return {
            "midcut_canvas_tokens": canvas,
            "midcut_rollout_tokens": rollout,
            "midcut_register_tokens": registers,
            "midcut_state_tokens": canvas[:, slices["state"]],
            "midcut_state_history_tokens": canvas[:, slices["state_history"]],
            "midcut_executed_tokens": canvas[:, slices["executed"]],
            "midcut_proposal_tokens": canvas[:, slices["proposal"]],
            "midcut_rollout_effect_pred": effect,
            "midcut_rollout_delta_pred": delta,
            "midcut_rollout_base_effect_pred": torch.zeros_like(effect),
            "midcut_event_logits": self.event_head(event_context),
            "midcut_transition_latent": transition,
            "midcut_rollout_delta_norm": delta.detach().float().norm(dim=-1).mean(),
            "midcut_rollout_effect_norm": effect.detach().float().norm(dim=-1).mean(),
            "midcut_future_gain": gain.detach().float().abs(),
        }


class LayerContractAdapterHeads(nn.Module):
    """Exact V120 bottleneck residual adapter followed by the weak readout."""

    def __init__(self, config: V39PolicyConfig, *, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        hidden = int(config.hidden_size)
        bottleneck = int(config.layer_contract_adapter_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden),
        )
        final = self.adapter[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("V120 layer adapter must end in Linear")
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        self.readout = MidcutContractHeads(config)

    def forward(
        self,
        canvas: Tensor,
        slices: dict[str, slice],
    ) -> dict[str, Tensor]:
        scale = torch.as_tensor(
            float(self.config.layer_contract_residual_scale),
            device=canvas.device,
            dtype=canvas.dtype,
        )
        adapted = canvas + scale * self.adapter(canvas)
        mid = self.readout(adapted, slices)
        output = {
            key[len("midcut_") :]: value
            for key, value in mid.items()
            if key.startswith("midcut_")
        }
        output["layer_index"] = torch.as_tensor(
            self.layer_index,
            device=canvas.device,
            dtype=torch.long,
        )
        return output


__all__ = [
    "LayerContractAdapterHeads",
    "MidcutContractHeads",
    "rollout_tokens_to_action_horizon",
]
