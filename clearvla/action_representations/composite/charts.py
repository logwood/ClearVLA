"""Replaceable temporal charts for continuous action roles."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn

from clearvla.action_representations.bspline import (
    BSplineActionRepresentation,
    BSplinePayload,
)


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_grid(value: Sequence[float], *, name: str) -> tuple[float, ...]:
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} entries must be real numbers")
        result.append(float(item))
    grid = tuple(result)
    if not grid:
        raise ValueError(f"{name} cannot be empty")
    if not all(math.isfinite(item) for item in grid):
        raise ValueError(f"{name} must be finite")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return grid


def _validate_supplied_grid(
    value: Tensor | Sequence[float] | None,
    *,
    expected: tuple[float, ...],
    mismatch_message: str = "times do not match the chart sample grid",
) -> None:
    if value is None:
        return
    # A Python float sequence has no declared tensor dtype.  Letting
    # ``torch.as_tensor`` infer one silently defaults it to the process' torch
    # dtype (normally FP32), which can round a genuinely different value such
    # as 1.00000001 to 1.0 before the authority check.  Treat plain numeric
    # sequences as exact Python floats (FP64); tensor callers retain their
    # explicitly declared dtype so valid FP32/BF16 quantized grids remain
    # supported.
    supplied = (
        torch.as_tensor(value)
        if isinstance(value, Tensor)
        else torch.as_tensor(value, dtype=torch.float64)
    )
    if supplied.dtype == torch.bool or supplied.is_complex():
        raise TypeError("times must be floating-point or integer values")
    if supplied.ndim != 1 or int(supplied.numel()) != len(expected):
        raise ValueError(f"times must contain exactly {len(expected)} values")
    if not bool(torch.isfinite(supplied).all()):
        raise ValueError("times must be finite")
    if supplied.numel() > 1 and not bool((supplied[1:] > supplied[:-1]).all()):
        raise ValueError("times must be strictly increasing")
    if supplied.is_floating_point():
        # Accept exactly the authoritative grid as represented by the caller's
        # dtype.  An epsilon window is unsafe for BF16: on a [0,1] grid it can
        # silently reinterpret a genuinely different endpoint such as 1.039.
        target = torch.tensor(expected, device=supplied.device, dtype=supplied.dtype)
        matches_authority = torch.equal(supplied, target)
    else:
        # Integer grids are accepted only when they equal the real-valued
        # authority exactly; integer quantization is not a time-grid policy.
        actual = supplied.to(dtype=torch.float64)
        target = torch.tensor(expected, device=supplied.device, dtype=torch.float64)
        matches_authority = torch.equal(actual, target)
    if not matches_authority:
        raise ValueError(mismatch_message)


@dataclass(frozen=True)
class RolePayload:
    """One encoded temporal view and its optional exact sampled-row bypass."""

    role_id: str
    chart_kind: str
    chart_fingerprint: str
    chart_state: Mapping[str, Any]
    raw: Tensor | None

    def _map_tensors(self, *, detach: bool, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RolePayload:
        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.detach() if detach else value.to(*args, **kwargs)
            return value

        raw = None if self.raw is None else convert(self.raw)
        return RolePayload(
            role_id=self.role_id,
            chart_kind=self.chart_kind,
            chart_fingerprint=self.chart_fingerprint,
            chart_state={key: convert(value) for key, value in self.chart_state.items()},
            raw=raw,
        )

    def to(self, *args: Any, **kwargs: Any) -> RolePayload:
        return self._map_tensors(detach=False, args=args, kwargs=kwargs)

    def detach(self) -> RolePayload:
        return self._map_tensors(detach=True, args=(), kwargs={})

    def as_state_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "chart_kind": self.chart_kind,
            "chart_fingerprint": self.chart_fingerprint,
            "chart_state": dict(self.chart_state),
            "raw": self.raw,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> RolePayload:
        expected = {
            "role_id",
            "chart_kind",
            "chart_fingerprint",
            "chart_state",
            "raw",
        }
        if set(value) != expected:
            raise ValueError("invalid role payload keys")
        chart_state = value["chart_state"]
        raw = value["raw"]
        if not isinstance(chart_state, Mapping):
            raise TypeError("role chart_state must be a mapping")
        if raw is not None and not isinstance(raw, Tensor):
            raise TypeError("role raw bypass must be a tensor or None")
        role_id = value["role_id"]
        chart_kind = value["chart_kind"]
        chart_fingerprint = value["chart_fingerprint"]
        for name, item in (
            ("role_id", role_id),
            ("chart_kind", chart_kind),
            ("chart_fingerprint", chart_fingerprint),
        ):
            if not isinstance(item, str) or not item.strip():
                raise TypeError(f"role payload {name} must be a non-empty string")
        return cls(
            role_id=role_id,
            chart_kind=chart_kind,
            chart_fingerprint=chart_fingerprint,
            chart_state=dict(chart_state),
            raw=raw,
        )


@runtime_checkable
class TemporalRoleChart(Protocol):
    """Structural ABI consumed by the role-wise composition layer."""

    chart_kind: str
    chart_fingerprint: str
    sample_times: tuple[float, ...]
    horizon: int
    width: int
    is_lossless: bool

    def encode(
        self,
        value: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> Mapping[str, Any]: ...

    def decode(
        self,
        state: Mapping[str, Any],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor: ...

    def validate_state(self, state: Mapping[str, Any]) -> None: ...

    def metadata(self) -> Mapping[str, Any]: ...


class IdentityRoleChart(nn.Module):
    """Exact pass-through chart for event-sensitive continuous trajectories."""

    chart_kind = "identity"

    def __init__(self, *, sample_times: Sequence[float], width: int) -> None:
        super().__init__()
        self.sample_times = _sample_grid(sample_times, name="sample_times")
        self.horizon = len(self.sample_times)
        if type(width) is not int:
            raise TypeError("identity chart width must be an integer")
        self.width = width
        self.is_lossless = True
        if self.width < 1:
            raise ValueError("identity chart width must be positive")
        self.chart_fingerprint = _fingerprint(
            {
                "chart": "clearvla.identity_temporal_role",
                "schema_version": 1,
                "sample_times_hex": [value.hex() for value in self.sample_times],
                "width": self.width,
            }
        )

    def _validate(self, value: Tensor) -> None:
        if value.ndim != 3 or tuple(value.shape[1:]) != (self.horizon, self.width):
            raise ValueError(
                f"identity role value must be [B,{self.horizon},{self.width}]"
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("identity role value must be finite and floating point")

    def encode(
        self,
        value: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> Mapping[str, Any]:
        self._validate(value)
        _validate_supplied_grid(times, expected=self.sample_times)
        if origin is not None:
            raise ValueError("identity chart does not consume an affine origin")
        return {"values": value}

    def validate_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"values"} or not isinstance(state["values"], Tensor):
            raise ValueError("invalid identity role chart state")
        self._validate(state["values"])

    def decode(
        self,
        state: Mapping[str, Any],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        self.validate_state(state)
        value = state["values"]
        assert isinstance(value, Tensor)
        return value if output_dtype is None else value.to(dtype=output_dtype)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "chart_kind": self.chart_kind,
            "chart_fingerprint": self.chart_fingerprint,
            "sample_times": list(self.sample_times),
            "horizon": self.horizon,
            "width": self.width,
            "is_lossless": self.is_lossless,
        }


class BSplineRoleChart(nn.Module):
    """Adapter from the canonical B-spline package to the role chart ABI."""

    chart_kind = "bspline"

    def __init__(self, representation: BSplineActionRepresentation) -> None:
        super().__init__()
        if not isinstance(representation, BSplineActionRepresentation):
            raise TypeError("representation must be a BSplineActionRepresentation")
        self.representation = representation
        self.sample_times = tuple(representation.spec.sample_times)
        self.horizon = representation.horizon
        self.width = representation.arm_dim
        self.is_lossless = representation.spec.is_lossless
        self.chart_fingerprint = _fingerprint(
            {
                "chart": "clearvla.bspline_temporal_role",
                "schema_version": 1,
                "spec_fingerprint": representation.spec.fingerprint,
                "basis_digest": representation.basis_digest,
            }
        )

    def encode(
        self,
        value: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> Mapping[str, Any]:
        return self.representation.encode(
            value,
            times=times,
            origin=origin,
        ).as_state_dict()

    def decode(
        self,
        state: Mapping[str, Any],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        return self.representation.decode(
            BSplinePayload.from_state_dict(state),
            output_dtype=output_dtype,
        )

    def validate_state(self, state: Mapping[str, Any]) -> None:
        # The canonical decoder owns the full shape, identity, finite-value and
        # basis-digest validation. This outer-boundary call may synchronize an
        # accelerator, which is why composite charts are forbidden in ODE loops.
        self.representation.decode(BSplinePayload.from_state_dict(state))

    def metadata(self) -> Mapping[str, Any]:
        return {
            "chart_kind": self.chart_kind,
            "chart_fingerprint": self.chart_fingerprint,
            "sample_times": list(self.sample_times),
            "horizon": self.horizon,
            "width": self.width,
            "is_lossless": self.is_lossless,
            "spec": self.representation.spec.to_dict(),
            "spec_fingerprint": self.representation.spec.fingerprint,
            "basis_digest": self.representation.basis_digest,
        }


__all__ = [
    "BSplineRoleChart",
    "IdentityRoleChart",
    "RolePayload",
    "TemporalRoleChart",
]


