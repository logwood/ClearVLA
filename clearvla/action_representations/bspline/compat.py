"""Duck-typed bridge for an existing native/physical action codec boundary."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn

from .representation import BSplineActionRepresentation, BSplinePayload


@runtime_checkable
class PhysicalActionCodecProtocol(Protocol):
    """Complete current façade ABI without importing a ClearVLA mainline."""

    action_dim: int
    horizon: int
    gripper_field_dim: int
    decode_delta_blend: float

    @property
    def arm_dim(self) -> int: ...

    @property
    def physical_dim(self) -> int: ...

    @property
    def uses_relative_command_direct(self) -> bool: ...

    def encode(
        self,
        action: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor: ...

    def decode(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor: ...

    def sample_noise(self, *args: Any, **kwargs: Any) -> Tensor: ...

    def split(self, field: Tensor) -> Any: ...

    def binary_command_model_input(self, field: Tensor) -> Tensor: ...

    def gripper_decode_branches(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    def delta_consistency(
        self,
        field: Tensor,
        action_state: Tensor,
        decoded_action: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor: ...

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor]: ...

    def arm_motion_magnitude(self, action: Tensor, action_state: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class NativeActionSplinePayload:
    """Spline arm coordinates paired with exact caller-owned channels."""

    arm: BSplinePayload
    passthrough: Tensor
    action_dim: int
    arm_dim: int

    def to(self, *args: Any, **kwargs: Any) -> NativeActionSplinePayload:
        return NativeActionSplinePayload(
            arm=self.arm.to(*args, **kwargs),
            passthrough=self.passthrough.to(*args, **kwargs),
            action_dim=self.action_dim,
            arm_dim=self.arm_dim,
        )

    def detach(self) -> NativeActionSplinePayload:
        return NativeActionSplinePayload(
            arm=self.arm.detach(),
            passthrough=self.passthrough.detach(),
            action_dim=self.action_dim,
            arm_dim=self.arm_dim,
        )

    def as_state_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.as_state_dict(),
            "passthrough": self.passthrough,
            "action_dim": self.action_dim,
            "arm_dim": self.arm_dim,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> NativeActionSplinePayload:
        expected = {"arm", "passthrough", "action_dim", "arm_dim"}
        if set(value) != expected:
            raise ValueError("invalid native-action B-spline payload keys")
        passthrough = value["passthrough"]
        arm = value["arm"]
        if not isinstance(passthrough, Tensor) or not isinstance(arm, Mapping):
            raise TypeError("serialized native-action payload contains invalid values")
        return cls(
            arm=BSplinePayload.from_state_dict(arm),
            passthrough=passthrough,
            action_dim=int(value["action_dim"]),
            arm_dim=int(value["arm_dim"]),
        )


class PhysicalActionFieldBSplineAdapter(nn.Module):
    """Outer-boundary compatibility façade around an existing physical codec.

    The façade owns no gripper/event semantics.  It transforms only the native
    arm prefix, concatenates the untouched suffix, and delegates physical-field
    construction and decoding to ``physical_codec``.  The safe default accepts
    only ``hierarchical_exact``, where it is a numerical-identity compatibility
    check.  A lossy native projection requires an explicit experimental opt-in
    because it changes targets and decoded actions.

    This class is not the repository plan's B-spine-0 and must not be inserted
    at its Gate-B codec/bottom cut.  Its finite validation can synchronize a
    GPU, and it is not designed for repeated ODE/deployment-bottom calls.
    """

    def __init__(
        self,
        representation: BSplineActionRepresentation,
        physical_codec: PhysicalActionCodecProtocol,
        *,
        allow_experimental_lossy_projection: bool = False,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.physical_codec = physical_codec
        self.allow_experimental_lossy_projection = bool(
            allow_experimental_lossy_projection
        )
        for attribute in (
            "action_dim",
            "horizon",
            "arm_dim",
            "physical_dim",
            "gripper_field_dim",
            "decode_delta_blend",
            "uses_relative_command_direct",
        ):
            if not hasattr(physical_codec, attribute):
                raise TypeError(f"physical_codec is missing required attribute {attribute!r}")
        for method in (
            "encode",
            "decode",
            "sample_noise",
            "split",
            "binary_command_model_input",
            "gripper_decode_branches",
            "delta_consistency",
            "project_arm_tangent",
            "arm_motion_magnitude",
        ):
            if not callable(getattr(physical_codec, method, None)):
                raise TypeError(f"physical_codec is missing required method {method!r}")
        for method in (
            "encode",
            "decode",
            "gripper_decode_branches",
            "delta_consistency",
        ):
            try:
                parameters = inspect.signature(getattr(physical_codec, method)).parameters
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"physical_codec method {method!r} has no inspectable signature"
                ) from error
            boundary = parameters.get("codec_gripper_boundary")
            if (
                boundary is None
                or boundary.kind is not inspect.Parameter.KEYWORD_ONLY
                or boundary.default is not None
            ):
                raise TypeError(
                    f"physical_codec method {method!r} must accept keyword-only "
                    "codec_gripper_boundary: Tensor | None = None"
                )
        if int(physical_codec.horizon) != representation.horizon:
            raise ValueError("physical codec and B-spline horizons differ")
        if int(physical_codec.arm_dim) != representation.arm_dim:
            raise ValueError("physical codec and B-spline arm dimensions differ")
        if int(physical_codec.action_dim) <= int(physical_codec.arm_dim):
            raise ValueError("physical codec must leave at least one passthrough channel")
        if not representation.spec.is_lossless and not self.allow_experimental_lossy_projection:
            raise ValueError(
                "lossy native projection is experimental and changes flow targets/decoded "
                "actions; pass allow_experimental_lossy_projection=True explicitly"
            )

    @property
    def action_dim(self) -> int:
        return int(self.physical_codec.action_dim)

    @property
    def horizon(self) -> int:
        return int(self.physical_codec.horizon)

    @property
    def arm_dim(self) -> int:
        return int(self.physical_codec.arm_dim)

    @property
    def physical_dim(self) -> int:
        return int(self.physical_codec.physical_dim)

    @property
    def gripper_field_dim(self) -> int:
        return int(self.physical_codec.gripper_field_dim)

    @property
    def decode_delta_blend(self) -> float:
        return float(self.physical_codec.decode_delta_blend)

    @property
    def uses_relative_command_direct(self) -> bool:
        return bool(self.physical_codec.uses_relative_command_direct)

    def _validate_native(self, action: Tensor) -> None:
        if action.ndim != 3 or tuple(action.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"native action must be [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if not action.is_floating_point() or not bool(torch.isfinite(action).all()):
            raise ValueError("native action must be finite and floating point")

    def encode_representation(
        self,
        native_action: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> NativeActionSplinePayload:
        """Split native actions using only a caller-declared affine origin.

        ``origin`` is never inferred from ``action_state``.  In CALVIN and
        ManiSkill the latter is the previous executed command rather than an
        absolute TCP pose, so treating it as a spatial anchor would be wrong.
        """

        self._validate_native(native_action)
        arm = self.representation.encode(
            native_action[..., : self.arm_dim],
            times=times,
            origin=origin,
        )
        return NativeActionSplinePayload(
            arm=arm,
            passthrough=native_action[..., self.arm_dim :].clone(),
            action_dim=self.action_dim,
            arm_dim=self.arm_dim,
        )

    def decode_representation(
        self,
        payload: NativeActionSplinePayload,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Recombine decoded arm values with byte-for-byte passthrough values."""

        if not isinstance(payload, NativeActionSplinePayload):
            raise TypeError("payload must be a NativeActionSplinePayload")
        if payload.action_dim != self.action_dim or payload.arm_dim != self.arm_dim:
            raise ValueError("native-action payload dimensions do not match this adapter")
        expected_suffix = (self.horizon, self.action_dim - self.arm_dim)
        if payload.passthrough.ndim != 3 or tuple(payload.passthrough.shape[1:]) != expected_suffix:
            raise ValueError("passthrough tensor must retain the original horizon and suffix width")
        if not payload.passthrough.is_floating_point() or not bool(
            torch.isfinite(payload.passthrough).all()
        ):
            raise ValueError("passthrough tensor must be finite and floating point")
        if int(payload.passthrough.shape[0]) != int(payload.arm.coarse.shape[0]):
            raise ValueError("arm and passthrough payload batches differ")
        if payload.passthrough.device != payload.arm.coarse.device:
            raise ValueError("arm and passthrough payloads must be on the same device")
        dtype = output_dtype or payload.passthrough.dtype
        arm = self.representation.decode(payload.arm, output_dtype=dtype)
        suffix = payload.passthrough.to(dtype=dtype)
        return torch.cat((arm, suffix), dim=-1)

    def project_native(
        self,
        native_action: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> Tensor:
        """Apply only the configured representation decision in native space."""

        payload = self.encode_representation(
            native_action,
            times=times,
            origin=origin,
        )
        return self.decode_representation(payload, output_dtype=native_action.dtype)

    def to_physical(
        self,
        payload: NativeActionSplinePayload,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        native = self.decode_representation(
            payload,
            output_dtype=payload.passthrough.dtype,
        )
        return self.physical_codec.encode(
            native,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def from_physical(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> NativeActionSplinePayload:
        native = self.physical_codec.decode(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )
        return self.encode_representation(
            native,
            times=times,
            origin=origin,
        )

    def encode(
        self,
        action: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        """Codec-compatible native-to-physical entry point."""

        projected = self.project_native(action)
        return self.physical_codec.encode(
            projected,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def decode(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        """Codec-compatible physical-to-native entry point."""

        native = self.physical_codec.decode(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )
        return self.project_native(native)

    # The remaining physical-field behavior stays entirely caller-owned.  The
    # explicit forwarding methods make the façade usable at today's codec slot
    # without a magical __getattr__ that could hide future ABI drift.
    def sample_noise(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.physical_codec.sample_noise(*args, **kwargs)

    def split(self, field: Tensor) -> Any:
        return self.physical_codec.split(field)

    def binary_command_model_input(self, field: Tensor) -> Tensor:
        return self.physical_codec.binary_command_model_input(field)

    def gripper_decode_branches(
        self,
        field: Tensor,
        action_state: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.physical_codec.gripper_decode_branches(
            field,
            action_state,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def delta_consistency(
        self,
        field: Tensor,
        action_state: Tensor,
        decoded_action: Tensor,
        *,
        codec_gripper_boundary: Tensor | None = None,
    ) -> Tensor:
        return self.physical_codec.delta_consistency(
            field,
            action_state,
            decoded_action,
            codec_gripper_boundary=codec_gripper_boundary,
        )

    def project_arm_tangent(self, arm_field: Tensor) -> tuple[Tensor, Tensor]:
        return self.physical_codec.project_arm_tangent(arm_field)

    def arm_motion_magnitude(self, action: Tensor, action_state: Tensor) -> Tensor:
        return self.physical_codec.arm_motion_magnitude(action, action_state)

    def integration_metadata(self) -> dict[str, Any]:
        """Small serializable record for a run-context integration owner."""

        return {
            "adapter": "physical_action_field_bspline_v1",
            "architectural_role": "standalone_outer_boundary_compatibility",
            "bspine0_gate_b_compatible": False,
            "repeated_bottom_call_safe": False,
            "origin_semantics": "explicit_affine_translation_only",
            "codec_gripper_boundary_semantics": "explicit_transparent_forwarding_only",
            "allow_experimental_lossy_projection": (
                self.allow_experimental_lossy_projection
            ),
            "spec": self.representation.spec.to_dict(),
            "spec_fingerprint": self.representation.spec.fingerprint,
            "basis_digest": self.representation.basis_digest,
            "physical_codec_type": type(self.physical_codec).__qualname__,
        }


__all__ = [
    "NativeActionSplinePayload",
    "PhysicalActionCodecProtocol",
    "PhysicalActionFieldBSplineAdapter",
]
