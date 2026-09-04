"""Deterministic physical action field used by the formal policy flow.

Pen/RDT retain the historical ``legacy_independent`` absolute/delta arm chart.
CALVIN selects an explicit direct relative-command chart while preserving the
same 18-D field ABI. Parseval, manifold and spectral ancestry do not belong in
the independent mainline.
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
    """Explicit 18-D physical chart with a native seven-dimensional boundary."""

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        gripper_field_dim: int = 6,
        decode_delta_blend: float = 0.25,
        arm_flow_mode: str = "legacy_independent",
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.gripper_field_dim = int(gripper_field_dim)
        self.decode_delta_blend = float(decode_delta_blend)
        self.arm_flow_mode = str(arm_flow_mode)
        if self.action_dim != 7:
            raise ValueError("the formal physical action chart requires native action_dim=7")
        if self.horizon != 24:
            raise ValueError("the formal physical action chart requires horizon=24")
        if self.gripper_field_dim != 6:
            raise ValueError("the resolved V122 legacy gripper field has exactly six channels")
        if self.decode_delta_blend != 0.25:
            raise ValueError("the resolved V122 physical decode blend is exactly 0.25")
        if self.arm_flow_mode not in {
            "legacy_independent",
            "relative_command_direct",
        }:
            raise ValueError(
                "arm_flow_mode must be legacy_independent or relative_command_direct"
            )
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

    @property
    def uses_relative_command_direct(self) -> bool:
        return self.arm_flow_mode == "relative_command_direct"

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
    def _resolve_codec_gripper_boundary(
        action_state: Tensor,
        codec_gripper_boundary: Tensor | None,
        *,
        reference: Tensor,
    ) -> Tensor:
        """Return the causal command boundary for the continuous gripper chart.

        Standalone compatibility callers may omit the explicit boundary, in
        which case the historical current-state anchor is retained.  The
        independent mainline always supplies the profile-owned online value so
        RDT command trajectories never use qpos as their codec anchor.
        """

        if codec_gripper_boundary is None:
            value = action_state[:, -1:]
        else:
            value = codec_gripper_boundary
        if tuple(value.shape) != (int(action_state.shape[0]), 1):
            raise ValueError("codec gripper boundary must align as [B,1]")
        return value.to(device=reference.device, dtype=reference.dtype)

    def _boundary(
        self,
        action: Tensor,
        action_state: Tensor,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        gripper = self._resolve_codec_gripper_boundary(
            action_state,
            codec_gripper_boundary,
            reference=action,
        )
        first = torch.cat(
            (
                action_state[:, : self.arm_dim].to(
                    device=action.device,
                    dtype=action.dtype,
                ),
                gripper,
            ),
            dim=-1,
        )
        return torch.cat(
            (first[:, None], action[:, :-1]),
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

    def binary_command_model_input(self, field: Tensor) -> Tensor:
        """Remove the legacy future-gripper field from CALVIN conditioning.

        The six trailing coordinates encode the *target* gripper trajectory
        during flow-matching training.  CALVIN deliberately emits zero
        velocity for those compatibility-only coordinates, so at deployment
        they remain source noise.  Letting the command head or any upstream
        policy block read them would therefore create a train/deploy shortcut.

        Keep the complete 18-D tensor ABI and the arm coordinates unchanged,
        but replace the six future-gripper coordinates with exact zeros at the
        model boundary.  The currently observed gripper state remains
        available through the ordinary state/action-history inputs.
        """

        if field.ndim != 3 or tuple(field.shape[1:]) != (
            self.horizon,
            self.physical_dim,
        ):
            raise ValueError(
                "binary command model input must use the complete physical field"
            )
        arm_channels = 2 * self.arm_dim
        return torch.cat(
            (
                field[..., :arm_channels],
                torch.zeros_like(field[..., arm_channels:]),
            ),
            dim=-1,
        )

    def encode(
        self,
        action: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        """Map a normalized native action chunk to the fixed 18-D flow chart."""

        self._validate_action(action, action_state)
        boundary = self._boundary(action, action_state, codec_gripper_boundary)
        arm = action[..., : self.arm_dim]
        previous_arm = boundary[..., : self.arm_dim]
        grip = action[..., -1:]
        previous_grip = boundary[..., -1:]
        anchor_grip = self._resolve_codec_gripper_boundary(
            action_state,
            codec_gripper_boundary,
            reference=action,
        )[:, None]
        delta = grip - previous_grip
        # The resolved field width is six, so these are exactly the first six
        # coordinates of the historical legacy_handcrafted codec.
        gripper_field = torch.cat(
            (
                grip,
                delta,
                grip - anchor_grip,
                previous_grip,
                delta.abs(),
                torch.relu(delta),
            ),
            dim=-1,
        )
        arm_secondary = arm if self.uses_relative_command_direct else arm - previous_arm
        return torch.cat((arm, arm_secondary, gripper_field), dim=-1)

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

    def decode(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        """Decode one physical field to the normalized native action chart."""

        self._validate_field(field, action_state)
        parts = self.split(field)
        state = action_state.to(device=field.device, dtype=field.dtype)
        if self.uses_relative_command_direct:
            arm_secondary = parts.arm_delta
        else:
            arm_secondary = state[:, None, : self.arm_dim] + torch.cumsum(
                parts.arm_delta, dim=1
            )
        grip_absolute, grip_from_delta = self.gripper_decode_branches(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )
        blend = self.decode_delta_blend
        arm = (1.0 - blend) * parts.arm_absolute + blend * arm_secondary
        grip = (1.0 - blend) * grip_absolute + blend * grip_from_delta
        return torch.cat((arm, grip), dim=-1)

    def gripper_decode_branches(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the two exact continuous operands used by deployment.

        The delta branch has one causal online anchor. Pen/CALVIN use current
        gripper state; RDT uses the previous executed command. Both are
        observable before the chunk and therefore available at deployment.
        """

        self._validate_field(field, action_state)
        parts = self.split(field)
        gripper = self._resolve_codec_gripper_boundary(
            action_state,
            codec_gripper_boundary,
            reference=field,
        )
        absolute = parts.gripper_field[..., :1]
        cumulative_delta = gripper[:, None] + torch.cumsum(
            parts.gripper_field[..., 1:2], dim=1
        )
        return absolute, cumulative_delta

    def delta_consistency(
        self,
        field: Tensor,
        action_state: Tensor,
        decoded_action: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        """Return the chart-specific physical consistency rows."""

        self._validate_field(field, action_state)
        self._validate_action(decoded_action, action_state)
        parts = self.split(field)
        if self.uses_relative_command_direct:
            # Both arm branches are direct views of the same relative command.
            # Do not compare either branch with a temporal command difference.
            return F.smooth_l1_loss(
                parts.arm_absolute,
                parts.arm_delta,
                reduction="none",
            ).mean(dim=-1)
        boundary = self._boundary(
            decoded_action,
            action_state,
            codec_gripper_boundary,
        )
        actual_delta = decoded_action - boundary
        field_delta = torch.cat(
            (parts.arm_delta, parts.gripper_field[..., 1:2]), dim=-1
        )
        return F.smooth_l1_loss(actual_delta, field_delta, reduction="none").mean(dim=-1)

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor]:
        """Return the chart-native arm residual and its compatible projection."""

        if arm_field.ndim != 3 or int(arm_field.shape[-1]) != 2 * self.arm_dim:
            raise ValueError("arm field must be [B,T,12]")
        absolute = arm_field[..., : self.arm_dim].float()
        delta = arm_field[..., self.arm_dim :].float()
        if self.uses_relative_command_direct:
            blend = self.decode_delta_blend
            native = (1.0 - blend) * absolute + blend * delta
            # The two direct branches are independent flow coordinates.  Their
            # disagreement is measured explicitly, not mislabeled as a legacy
            # finite-difference null component.
            return native.to(dtype=arm_field.dtype), arm_field
        difference = self.arm_difference_matrix.to(device=arm_field.device, dtype=torch.float32)
        inverse = self.arm_projection_inverse.to(device=arm_field.device, dtype=torch.float32)
        with torch.autocast(device_type=arm_field.device.type, enabled=False):
            rhs = absolute + torch.einsum("ts,btd->bsd", difference, delta)
            native = torch.einsum("ts,bsd->btd", inverse, rhs)
            projected_delta = torch.einsum("ts,bsd->btd", difference, native)
        projected = torch.cat((native, projected_delta), dim=-1)
        return native.to(dtype=arm_field.dtype), projected.to(dtype=arm_field.dtype)

    def arm_motion_magnitude(self, action: Tensor, action_state: Tensor) -> Tensor:
        """Return the chart-native per-row arm motion magnitude.

        A CALVIN arm row is itself a relative TCP command.  Pen/RDT rows remain
        absolute commands and therefore retain the historical adjacent-delta
        definition exactly.
        """

        self._validate_action(action, action_state)
        arm = action[..., : self.arm_dim]
        if self.uses_relative_command_direct:
            return arm.float().norm(dim=-1)
        boundary = self._boundary(action, action_state)
        return (arm - boundary[..., : self.arm_dim]).float().norm(dim=-1)


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


def binary_gripper_command_from_logits(logits: Tensor) -> Tensor:
    """Map two-class gripper logits to the strict CALVIN command alphabet.

    Class ``0`` maps to the native ``-1`` command and class ``1`` maps to the
    native ``+1`` command.  The returned tensor keeps the leading dimensions
    of ``logits`` and has values exactly in ``{-1,+1}``; no normalizer or
    continuous physical field is involved in this conversion.
    """

    if not isinstance(logits, Tensor) or logits.ndim < 1 or int(logits.shape[-1]) != 2:
        raise ValueError(
            "binary gripper command logits must have a final two-class dimension"
        )
    if not torch.isfinite(logits.float()).all():
        raise ValueError("binary gripper command logits must be finite")
    return logits.argmax(dim=-1).to(dtype=torch.float32).mul(2.0).sub(1.0)


__all__ = [
    "ACTION_BAND_ENDS",
    "PhysicalActionFieldCodec",
    "PhysicalActionFieldParts",
    "anchor_horizon_weights",
    "binary_gripper_command_from_logits",
]
