"""Deterministic physical action field used by the formal policy flow.

The frozen V122 launcher resolves to the historical ``legacy_independent`` arm
chart and ``legacy_handcrafted`` six-channel gripper field.  This module owns
only that executed branch; Parseval, manifold and spectral ancestry do not
belong in the independent mainline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ACTION_BAND_ENDS = (4, 12, 24)


@dataclass(frozen=True)
class PhysicalActionFieldParts:
    arm_absolute: Tensor
    arm_delta: Tensor
    gripper_field: Tensor


class PhysicalActionFieldCodec(nn.Module):
    """Exact legacy physical chart with a native seven-dimensional boundary."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        gripper_field_dim: int = 6,
        decode_delta_blend: float = 0.25,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.gripper_field_dim = int(gripper_field_dim)
        self.decode_delta_blend = float(decode_delta_blend)
        if self.action_dim != 7:
            raise ValueError("the formal physical action chart requires native action_dim=7")
        if self.horizon != 24:
            raise ValueError("the formal physical action chart requires horizon=24")
        if self.gripper_field_dim != 6:
            raise ValueError("the resolved V122 legacy gripper field has exactly six channels")
        if self.decode_delta_blend != 0.25:
            raise ValueError("the resolved V122 physical decode blend is exactly 0.25")
        difference = torch.eye(self.horizon, dtype=torch.float64)
        if self.horizon > 1:
            difference[1:, :-1] -= torch.eye(self.horizon - 1, dtype=torch.float64)
        gram = torch.eye(self.horizon, dtype=torch.float64) + difference.T @ difference
        self.register_buffer(
            "arm_difference_matrix", difference.to(torch.float32), persistent=False
        )
        self.register_buffer(
            "arm_projection_inverse", torch.linalg.inv(gram).to(torch.float32), persistent=False
        )

    @property
    def arm_dim(self) -> int:
        return self.action_dim - 1

    @property
    def physical_dim(self) -> int:
        return 2 * self.arm_dim + self.gripper_field_dim

    def _validate_action(self, action: Tensor, action_state: Tensor) -> None:
        if action.ndim != 3 or tuple(action.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError(
                f"native action must be [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if tuple(action_state.shape) != (int(action.shape[0]), self.action_dim):
            raise ValueError("native action state must align as [B,7]")

    def _validate_field(self, field: Tensor, action_state: Tensor) -> None:
        if field.ndim != 3 or tuple(field.shape[1:]) != (self.horizon, self.physical_dim):
            raise ValueError(
                f"physical action field must be [B,{self.horizon},{self.physical_dim}], "
                f"got {tuple(field.shape)}"
            )
        if tuple(action_state.shape) != (int(field.shape[0]), self.action_dim):
            raise ValueError("native action state must align with the physical field")

    @staticmethod
    def _boundary(action: Tensor, action_state: Tensor) -> Tensor:
        return torch.cat(
            (action_state[:, None].to(device=action.device, dtype=action.dtype), action[:, :-1]),
            dim=1,
        )

    def split(self, field: Tensor) -> PhysicalActionFieldParts:
        if field.ndim != 3 or int(field.shape[-1]) != self.physical_dim:
            raise ValueError("physical action field has the wrong final dimension")
        arm = self.arm_dim
        return PhysicalActionFieldParts(
            arm_absolute=field[..., :arm],
            arm_delta=field[..., arm : 2 * arm],
            gripper_field=field[..., 2 * arm :],
        )

    def encode(self, action: Tensor, action_state: Tensor) -> Tensor:
        """Map a normalized native action chunk to the fixed 18-D flow chart."""

        self._validate_action(action, action_state)
        boundary = self._boundary(action, action_state)
        arm = action[..., : self.arm_dim]
        previous_arm = boundary[..., : self.arm_dim]
        grip = action[..., -1:]
        previous_grip = boundary[..., -1:]
        state_grip = action_state[:, None, -1:].to(device=action.device, dtype=action.dtype)
        delta = grip - previous_grip
        # The resolved field width is six, so these are exactly the first six
        # coordinates of the historical legacy_handcrafted codec.
        gripper_field = torch.cat(
            (
                grip,
                delta,
                grip - state_grip,
                previous_grip,
                delta.abs(),
                torch.relu(delta),
            ),
            dim=-1,
        )
        return torch.cat((arm, arm - previous_arm, gripper_field), dim=-1)

    def sample_noise(
        self,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Preserve the legacy independent standard-normal source exactly."""

        return torch.randn(
            int(batch),
            self.horizon,
            self.physical_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def decode(self, field: Tensor, action_state: Tensor) -> Tensor:
        """Decode one physical field to the normalized native action chart."""

        self._validate_field(field, action_state)
        parts = self.split(field)
        state = action_state.to(device=field.device, dtype=field.dtype)
        arm_from_delta = state[:, None, : self.arm_dim] + torch.cumsum(
            parts.arm_delta, dim=1
        )
        grip_absolute = parts.gripper_field[..., :1]
        grip_from_delta = state[:, None, -1:] + torch.cumsum(
            parts.gripper_field[..., 1:2], dim=1
        )
        blend = self.decode_delta_blend
        arm = (1.0 - blend) * parts.arm_absolute + blend * arm_from_delta
        grip = (1.0 - blend) * grip_absolute + blend * grip_from_delta
        return torch.cat((arm, grip), dim=-1)

    def delta_consistency(
        self,
        field: Tensor,
        action_state: Tensor,
        decoded_action: Tensor,
    ) -> Tensor:
        """Return the historical per-step physical/native delta SmoothL1 rows."""

        self._validate_field(field, action_state)
        self._validate_action(decoded_action, action_state)
        boundary = self._boundary(decoded_action, action_state)
        actual_delta = decoded_action - boundary
        parts = self.split(field)
        field_delta = torch.cat(
            (parts.arm_delta, parts.gripper_field[..., 1:2]), dim=-1
        )
        return F.smooth_l1_loss(actual_delta, field_delta, reduction="none").mean(dim=-1)

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor]:
        """Project an independent [absolute,delta] residual onto native arm motion."""

        if arm_field.ndim != 3 or int(arm_field.shape[-1]) != 2 * self.arm_dim:
            raise ValueError("arm field must be [B,T,12]")
        absolute = arm_field[..., : self.arm_dim].float()
        delta = arm_field[..., self.arm_dim :].float()
        difference = self.arm_difference_matrix.to(device=arm_field.device, dtype=torch.float32)
        inverse = self.arm_projection_inverse.to(device=arm_field.device, dtype=torch.float32)
        with torch.autocast(device_type=arm_field.device.type, enabled=False):
            rhs = absolute + torch.einsum("ts,btd->bsd", difference, delta)
            native = torch.einsum("ts,bsd->btd", inverse, rhs)
            projected_delta = torch.einsum("ts,bsd->btd", difference, native)
        projected = torch.cat((native, projected_delta), dim=-1)
        return native.to(dtype=arm_field.dtype), projected.to(dtype=arm_field.dtype)


def anchor_horizon_weights(
    *,
    horizon: int,
    tail_emphasis: float,
    first_step_protection: float,
    device: torch.device,
) -> Tensor:
    """Restore V120's per-row horizon pressure with exact unit mean.

    ``tail_emphasis`` is a mild per-row band multiplier, not a request to give
    the three unequal-length bands equal total mass.  Equal-band allocation
    reduced the 12-row far horizon from roughly 53% of the action objective to
    36% and changed the experiment while claiming only to improve accounting.
    Gripper event/hold balancing is applied separately inside its own channel
    and must not redefine this temporal objective.
    """

    horizon = int(horizon)
    if horizon != ACTION_BAND_ENDS[-1]:
        raise ValueError("anchor bands must end at the 24-step action horizon")
    if float(tail_emphasis) < 0.0 or float(first_step_protection) < 0.0:
        raise ValueError("anchor-band emphasis values must be non-negative")
    weight = torch.empty(horizon, device=device, dtype=torch.float32)
    start = 0
    denominator = max(len(ACTION_BAND_ENDS) - 1, 1)
    for index, end in enumerate(ACTION_BAND_ENDS):
        row_weight = 1.0 + float(tail_emphasis) * float(index) / float(denominator)
        weight[start:end] = row_weight
        start = end
    weight[0] = weight[0] + float(first_step_protection)
    return weight / weight.mean()


__all__ = [
    "ACTION_BAND_ENDS",
    "PhysicalActionFieldCodec",
    "PhysicalActionFieldParts",
    "anchor_horizon_weights",
]
