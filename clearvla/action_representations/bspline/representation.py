"""Lossless-capable hierarchical B-spline action representation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .basis import (
    bspline_basis,
    build_basis_bundle,
    structural_diagnostics,
)
from .spec import BSplineSpec


@dataclass(frozen=True)
class BSplinePayload:
    """One encoded arm chunk plus the identities required for safe decoding."""

    coarse: Tensor
    detail: Tensor
    origin: Tensor | None
    schema_version: int
    spec_fingerprint: str
    basis_digest: str
    mode: str

    def to(self, *args: Any, **kwargs: Any) -> BSplinePayload:
        origin = None if self.origin is None else self.origin.to(*args, **kwargs)
        return BSplinePayload(
            coarse=self.coarse.to(*args, **kwargs),
            detail=self.detail.to(*args, **kwargs),
            origin=origin,
            schema_version=self.schema_version,
            spec_fingerprint=self.spec_fingerprint,
            basis_digest=self.basis_digest,
            mode=self.mode,
        )

    def detach(self) -> BSplinePayload:
        origin = None if self.origin is None else self.origin.detach()
        return BSplinePayload(
            coarse=self.coarse.detach(),
            detail=self.detail.detach(),
            origin=origin,
            schema_version=self.schema_version,
            spec_fingerprint=self.spec_fingerprint,
            basis_digest=self.basis_digest,
            mode=self.mode,
        )

    def as_state_dict(self) -> dict[str, Any]:
        """Return a plain torch-serializable payload with no class dependency."""

        return {
            "coarse": self.coarse,
            "detail": self.detail,
            "origin": self.origin,
            "schema_version": self.schema_version,
            "spec_fingerprint": self.spec_fingerprint,
            "basis_digest": self.basis_digest,
            "mode": self.mode,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> BSplinePayload:
        required = {
            "coarse",
            "detail",
            "origin",
            "schema_version",
            "spec_fingerprint",
            "basis_digest",
            "mode",
        }
        missing = required.difference(value)
        extra = set(value).difference(required)
        if missing or extra:
            raise ValueError(
                f"invalid B-spline payload keys; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        coarse = value["coarse"]
        detail = value["detail"]
        origin = value["origin"]
        if not isinstance(coarse, Tensor) or not isinstance(detail, Tensor):
            raise TypeError("serialized coarse and detail values must be tensors")
        if origin is not None and not isinstance(origin, Tensor):
            raise TypeError("serialized origin must be a tensor or None")
        return cls(
            coarse=coarse,
            detail=detail,
            origin=origin,
            schema_version=int(value["schema_version"]),
            spec_fingerprint=str(value["spec_fingerprint"]),
            basis_digest=str(value["basis_digest"]),
            mode=str(value["mode"]),
        )


class BSplineActionRepresentation(nn.Module):
    """Fixed arm-trajectory chart with exact and explicitly lossy modes.

    Matrix construction is deterministic float64 CPU work.  Ordinary runtime
    inputs use an autocast-disabled float32 scope (float64 inputs remain
    float64), so BF16/FP16 callers do not silently perform basis algebra at
    reduced precision.  The transform has no parameters and introduces no
    gradient boundary.  Public validation intentionally performs finite checks
    that may synchronize an accelerator.  This payload API is therefore an
    outer-boundary/offline representation tool, not the tensor-only primitive
    for repeated ODE or deployment-bottom calls.
    """

    _RUNTIME_BUFFER_NAMES = (
        "sample_times",
        "coarse_knots",
        "interpolation_knots",
        "coarse_collocation",
        "interpolation_collocation",
        "coarse_q",
        "coarse_r",
        "detail_q",
        "detail_control_map",
    )

    def __init__(self, spec: BSplineSpec) -> None:
        super().__init__()
        self.spec = spec
        bundle = build_basis_bundle(spec)
        self._canonical_bundle = bundle
        self.basis_digest = bundle.digest
        self._basis_diagnostics = structural_diagnostics(bundle, spec)

        for name in self._RUNTIME_BUFFER_NAMES:
            value = getattr(bundle, name)
            self.register_buffer(name, value.to(torch.float32), persistent=False)

    def _apply(self, fn: Any, recurse: bool = True) -> BSplineActionRepresentation:
        """Move buffers while refusing accidental BF16/FP16 basis storage."""

        super()._apply(fn, recurse=recurse)
        for name in self._RUNTIME_BUFFER_NAMES:
            moved = self._buffers[name]
            assert moved is not None
            canonical = getattr(self._canonical_bundle, name)
            self._buffers[name] = canonical.to(device=moved.device, dtype=torch.float32)
        return self

    @property
    def horizon(self) -> int:
        return self.spec.horizon

    @property
    def arm_dim(self) -> int:
        return self.spec.arm_dim

    @property
    def coordinate_rank(self) -> int:
        return self.spec.coordinate_rank

    @staticmethod
    def _computation_dtype(tensor: Tensor) -> torch.dtype:
        return torch.float64 if tensor.dtype == torch.float64 else torch.float32

    @staticmethod
    def _numerical_scope(tensor: Tensor) -> Any:
        if tensor.device.type in {"cpu", "cuda", "xpu", "mps"}:
            return torch.autocast(device_type=tensor.device.type, enabled=False)
        return nullcontext()

    @staticmethod
    def _require_floating_finite(tensor: Tensor, *, name: str) -> None:
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if not bool(torch.isfinite(tensor.detach()).all()):
            raise ValueError(f"{name} must be finite")

    def _matrix(self, name: str, reference: Tensor, dtype: torch.dtype) -> Tensor:
        # Preserve the canonical CPU float64 matrices for explicit float64
        # audits.  Ordinary runtime calls use the registered float32 buffers.
        value = (
            getattr(self._canonical_bundle, name) if dtype == torch.float64 else getattr(self, name)
        )
        return value.to(device=reference.device, dtype=dtype)

    def _validate_sample_times(self, times: Tensor | Sequence[float] | None) -> None:
        if times is None:
            return
        supplied = torch.as_tensor(times)
        if supplied.ndim != 1 or int(supplied.numel()) != self.horizon:
            raise ValueError(f"times must contain exactly {self.horizon} timestamps")
        if not supplied.is_floating_point():
            supplied = supplied.to(torch.float64)
        if not bool(torch.isfinite(supplied.detach()).all()):
            raise ValueError("times must be finite")
        expected = self._canonical_bundle.sample_times
        actual = supplied.detach().to(device="cpu", dtype=torch.float64)
        source_epsilon = torch.finfo(supplied.dtype).eps
        scale = max(abs(self.spec.sample_times[-1] - self.spec.sample_times[0]), 1.0)
        tolerance = float(source_epsilon) * scale * 8.0
        if not torch.allclose(actual, expected, rtol=0.0, atol=tolerance):
            raise ValueError("times do not match the immutable BSplineSpec sample grid")

    def _validate_arm_trajectory(self, trajectory: Tensor) -> None:
        self._require_floating_finite(trajectory, name="arm_trajectory")
        expected = (self.horizon, self.arm_dim)
        if trajectory.ndim != 3 or tuple(trajectory.shape[1:]) != expected:
            raise ValueError(
                f"arm_trajectory must be [B,{self.horizon},{self.arm_dim}], "
                f"got {tuple(trajectory.shape)}"
            )
        if int(trajectory.shape[0]) < 1:
            raise ValueError("arm_trajectory batch cannot be empty")

    def _validate_origin(self, origin: Tensor | None, reference: Tensor) -> None:
        if origin is None:
            return
        self._require_floating_finite(origin, name="origin")
        expected = (int(reference.shape[0]), self.arm_dim)
        if tuple(origin.shape) != expected:
            raise ValueError(f"origin must have shape {expected}, got {tuple(origin.shape)}")

    def _validate_payload(self, payload: BSplinePayload) -> None:
        if not isinstance(payload, BSplinePayload):
            raise TypeError("payload must be a BSplinePayload")
        if payload.schema_version != self.spec.schema_version:
            raise ValueError("payload schema version does not match this representation")
        if payload.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("payload spec fingerprint does not match this representation")
        if payload.basis_digest != self.basis_digest:
            raise ValueError("payload basis digest does not match this representation")
        if payload.mode != self.spec.mode:
            raise ValueError("payload mode does not match this representation")
        expected_coarse = (self.spec.coarse_rank, self.arm_dim)
        expected_detail = (self.spec.retained_detail_rank, self.arm_dim)
        if payload.coarse.ndim != 3 or tuple(payload.coarse.shape[1:]) != expected_coarse:
            raise ValueError(
                f"payload coarse tensor must be [B,{expected_coarse[0]},{self.arm_dim}]"
            )
        if payload.detail.ndim != 3 or tuple(payload.detail.shape[1:]) != expected_detail:
            raise ValueError(
                f"payload detail tensor must be [B,{expected_detail[0]},{self.arm_dim}]"
            )
        if int(payload.detail.shape[0]) != int(payload.coarse.shape[0]):
            raise ValueError("payload coarse and detail batches differ")
        self._require_floating_finite(payload.coarse, name="payload.coarse")
        self._require_floating_finite(payload.detail, name="payload.detail")
        self._validate_origin(payload.origin, payload.coarse)
        tensors = [payload.coarse, payload.detail]
        if payload.origin is not None:
            tensors.append(payload.origin)
        if any(tensor.device != payload.coarse.device for tensor in tensors):
            raise ValueError("all payload tensors must be on the same device")

    def _new_payload(
        self,
        coarse: Tensor,
        detail: Tensor,
        origin: Tensor | None,
    ) -> BSplinePayload:
        return BSplinePayload(
            coarse=coarse,
            detail=detail,
            origin=origin,
            schema_version=self.spec.schema_version,
            spec_fingerprint=self.spec.fingerprint,
            basis_digest=self.basis_digest,
            mode=self.spec.mode,
        )

    def encode(
        self,
        arm_trajectory: Tensor,
        *,
        times: Tensor | Sequence[float] | None = None,
        origin: Tensor | None = None,
    ) -> BSplinePayload:
        """Encode ``[B,T,D_arm]`` into coarse and retained detail coordinates.

        ``origin`` is an explicit affine coordinate translation.  It does not
        prepend a row or force the first action to equal the supplied state.
        """

        self._validate_arm_trajectory(arm_trajectory)
        self._validate_sample_times(times)
        self._validate_origin(origin, arm_trajectory)
        dtype = self._computation_dtype(arm_trajectory)
        with self._numerical_scope(arm_trajectory):
            centered = arm_trajectory.to(dtype=dtype)
            encoded_origin = None
            if origin is not None:
                encoded_origin = origin.to(
                    device=arm_trajectory.device,
                    dtype=dtype,
                ).clone()
                centered = centered - encoded_origin[:, None, :]
            q_coarse = self._matrix("coarse_q", arm_trajectory, dtype)
            coarse = torch.einsum("tk,btd->bkd", q_coarse, centered)
            retained = self.spec.retained_detail_rank
            if retained:
                q_detail = self._matrix("detail_q", arm_trajectory, dtype)[:, :retained]
                detail = torch.einsum("tl,btd->bld", q_detail, centered)
            else:
                detail = centered.new_empty(int(centered.shape[0]), 0, self.arm_dim)
        return self._new_payload(coarse, detail, encoded_origin)

    def from_coordinates(
        self,
        coordinates: Tensor,
        *,
        origin: Tensor | None = None,
    ) -> BSplinePayload:
        """Package externally produced orthonormal coordinates safely."""

        self._require_floating_finite(coordinates, name="coordinates")
        expected = (self.coordinate_rank, self.arm_dim)
        if coordinates.ndim != 3 or tuple(coordinates.shape[1:]) != expected:
            raise ValueError(
                f"coordinates must be [B,{self.coordinate_rank},{self.arm_dim}], "
                f"got {tuple(coordinates.shape)}"
            )
        self._validate_origin(origin, coordinates)
        if origin is not None and origin.device != coordinates.device:
            raise ValueError("origin and coordinates must be on the same device")
        split = self.spec.coarse_rank
        return self._new_payload(coordinates[:, :split], coordinates[:, split:], origin)

    def coordinates(self, payload: BSplinePayload) -> Tensor:
        """Return the stable ``[coarse, retained-detail]`` coordinate layout."""

        self._validate_payload(payload)
        return torch.cat((payload.coarse, payload.detail), dim=1)

    def decode(
        self,
        payload: BSplinePayload,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Decode on the original sample grid without a continuous solve."""

        self._validate_payload(payload)
        dtype = self._computation_dtype(payload.coarse)
        with self._numerical_scope(payload.coarse):
            coarse = payload.coarse.to(dtype=dtype)
            q_coarse = self._matrix("coarse_q", payload.coarse, dtype)
            result = torch.einsum("tk,bkd->btd", q_coarse, coarse)
            if self.spec.retained_detail_rank:
                detail = payload.detail.to(dtype=dtype)
                q_detail = self._matrix("detail_q", payload.coarse, dtype)[
                    :, : self.spec.retained_detail_rank
                ]
                result = result + torch.einsum("tl,bld->btd", q_detail, detail)
            if payload.origin is not None:
                result = result + payload.origin.to(dtype=dtype)[:, None, :]
        return result if output_dtype is None else result.to(dtype=output_dtype)

    def coarse_control_points(
        self,
        payload: BSplinePayload,
        *,
        absolute: bool = True,
    ) -> Tensor:
        """Return ordinary B-spline controls for the coarse trajectory."""

        self._validate_payload(payload)
        dtype = self._computation_dtype(payload.coarse)
        with self._numerical_scope(payload.coarse):
            r = self._matrix("coarse_r", payload.coarse, dtype)
            controls = torch.linalg.solve_triangular(
                r,
                payload.coarse.to(dtype=dtype),
                upper=True,
            )
            if absolute and payload.origin is not None:
                controls = controls + payload.origin.to(dtype=dtype)[:, None, :]
        return controls

    def _query_tensor(
        self,
        query_times: Tensor | Sequence[float],
        reference: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        if isinstance(query_times, Tensor):
            query = query_times.to(device=reference.device)
        else:
            query = torch.tensor(tuple(query_times), device=reference.device, dtype=dtype)
        if query.ndim != 1 or int(query.numel()) < 1:
            raise ValueError("query_times must be a non-empty one-dimensional grid")
        if not query.is_floating_point():
            query = query.to(torch.float64)
        self._require_floating_finite(query, name="query_times")
        return query.to(dtype=dtype)

    def _coordinate_evaluation_matrix(
        self,
        query_times: Tensor,
        reference: Tensor,
        dtype: torch.dtype,
        *,
        derivative_order: int,
    ) -> Tensor:
        coarse_knots = self._matrix("coarse_knots", reference, dtype)
        coarse_at_query = bspline_basis(
            query_times,
            coarse_knots,
            self.spec.degree,
            derivative_order=derivative_order,
        )
        r = self._matrix("coarse_r", reference, dtype)
        coarse_map = torch.linalg.solve_triangular(
            r.T,
            coarse_at_query.T,
            upper=False,
        ).T
        retained = self.spec.retained_detail_rank
        if not retained:
            return coarse_map
        interpolation_knots = self._matrix("interpolation_knots", reference, dtype)
        interpolation_at_query = bspline_basis(
            query_times,
            interpolation_knots,
            self.spec.degree,
            derivative_order=derivative_order,
        )
        detail_controls = self._matrix("detail_control_map", reference, dtype)[:, :retained]
        return torch.cat((coarse_map, interpolation_at_query @ detail_controls), dim=-1)

    def evaluate(
        self,
        payload: BSplinePayload,
        query_times: Tensor | Sequence[float],
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Evaluate the complete piecewise-polynomial curve at a common grid."""

        self._validate_payload(payload)
        dtype = self._computation_dtype(payload.coarse)
        query = self._query_tensor(query_times, payload.coarse, dtype)
        with self._numerical_scope(payload.coarse):
            coordinate_map = self._coordinate_evaluation_matrix(
                query,
                payload.coarse,
                dtype,
                derivative_order=0,
            )
            values = torch.einsum(
                "qr,brd->bqd",
                coordinate_map,
                torch.cat((payload.coarse, payload.detail), dim=1).to(dtype=dtype),
            )
            if payload.origin is not None:
                values = values + payload.origin.to(dtype=dtype)[:, None, :]
        return values if output_dtype is None else values.to(dtype=output_dtype)

    def derivative(
        self,
        payload: BSplinePayload,
        query_times: Tensor | Sequence[float],
        *,
        order: int = 1,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Evaluate the first or second derivative in ``time_unit`` units."""

        self._validate_payload(payload)
        order = int(order)
        if order not in (1, 2):
            raise ValueError("derivative order must be 1 or 2")
        dtype = self._computation_dtype(payload.coarse)
        query = self._query_tensor(query_times, payload.coarse, dtype)
        with self._numerical_scope(payload.coarse):
            coordinate_map = self._coordinate_evaluation_matrix(
                query,
                payload.coarse,
                dtype,
                derivative_order=order,
            )
            values = torch.einsum(
                "qr,brd->bqd",
                coordinate_map,
                torch.cat((payload.coarse, payload.detail), dim=1).to(dtype=dtype),
            )
        return values if output_dtype is None else values.to(dtype=output_dtype)

    def evaluation_operator(
        self,
        query_times: Tensor | Sequence[float],
        *,
        derivative_order: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Return the linear map from sampled arm rows to queried curve rows."""

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("evaluation operator dtype must be float32 or float64")
        reference = torch.empty((), device=device or "cpu", dtype=dtype)
        query = self._query_tensor(query_times, reference, dtype)
        coordinate_map = self._coordinate_evaluation_matrix(
            query,
            reference,
            dtype,
            derivative_order=int(derivative_order),
        )
        q = torch.cat(
            (
                self._matrix("coarse_q", reference, dtype),
                self._matrix("detail_q", reference, dtype)[:, : self.spec.retained_detail_rank],
            ),
            dim=-1,
        )
        return coordinate_map @ q.T

    def basis_diagnostics(self) -> dict[str, float | int | str | bool]:
        """Return a copy of the immutable structural diagnostics."""

        return dict(self._basis_diagnostics)

    def diagnostics(
        self,
        arm_trajectory: Tensor,
        payload: BSplinePayload | None = None,
        *,
        dense_samples: int = 257,
    ) -> dict[str, Tensor | float | int | str | bool]:
        """Audit reconstruction slices and unclipped dense-curve extrema."""

        self._validate_arm_trajectory(arm_trajectory)
        if payload is None:
            payload = self.encode(arm_trajectory)
        self._validate_payload(payload)
        if int(payload.coarse.shape[0]) != int(arm_trajectory.shape[0]):
            raise ValueError("trajectory and payload batches differ")
        if payload.coarse.device != arm_trajectory.device:
            raise ValueError("trajectory and payload must be on the same device")
        reconstructed = self.decode(payload)
        reference = arm_trajectory.to(device=reconstructed.device, dtype=reconstructed.dtype)
        error = reconstructed - reference

        def rmse(rows: slice) -> Tensor:
            return error[:, rows].float().square().mean().sqrt().detach()

        dense_samples = int(dense_samples)
        if dense_samples < 2:
            raise ValueError("dense_samples must be at least two")
        dense_times = torch.linspace(
            self.spec.sample_times[0],
            self.spec.sample_times[-1],
            dense_samples,
            device=payload.coarse.device,
            dtype=torch.float32,
        )
        dense = self.evaluate(payload, dense_times)
        velocity = self.derivative(payload, dense_times, order=1)
        acceleration = self.derivative(payload, dense_times, order=2)
        sampled_min = reference.amin(dim=1, keepdim=True)
        sampled_max = reference.amax(dim=1, keepdim=True)
        overshoot = torch.maximum(
            (dense - sampled_max).clamp_min(0.0),
            (sampled_min - dense).clamp_min(0.0),
        )

        dtype = self._computation_dtype(arm_trajectory)
        centered = arm_trajectory.to(dtype=dtype)
        if payload.origin is not None:
            centered = centered - payload.origin.to(dtype=dtype)[:, None, :]
        all_detail = self._matrix("detail_q", arm_trajectory, dtype)
        detail_coordinates = torch.einsum("tl,btd->bld", all_detail, centered)
        total_energy = centered.square().sum().clamp_min(torch.finfo(dtype).tiny)
        retained = self.spec.retained_detail_rank
        retained_energy = detail_coordinates[:, :retained].square().sum()
        detail_energy = detail_coordinates.square().sum()
        result: dict[str, Tensor | float | int | str | bool] = dict(self.basis_diagnostics())
        tail_start = min(max(self.horizon // 3, 1), self.horizon - 1)
        result.update(
            {
                "rmse_full": error.float().square().mean().sqrt().detach(),
                "rmse_first": rmse(slice(0, 1)),
                "rmse_first4": rmse(slice(0, min(4, self.horizon))),
                "rmse_tail": rmse(slice(tail_start, self.horizon)),
                "max_abs": error.float().abs().max().detach(),
                "detail_energy_fraction": (detail_energy / total_energy).float().detach(),
                "retained_detail_energy_fraction": (retained_energy / total_energy)
                .float()
                .detach(),
                "dense_overshoot_max_abs": overshoot.float().max().detach(),
                "dense_value_max_abs": dense.float().abs().max().detach(),
                "dense_velocity_max_abs": velocity.float().abs().max().detach(),
                "dense_acceleration_max_abs": acceleration.float().abs().max().detach(),
            }
        )
        return result

    def extra_repr(self) -> str:
        return (
            f"horizon={self.horizon}, arm_dim={self.arm_dim}, "
            f"controls={self.spec.coarse_rank}, degree={self.spec.degree}, "
            f"mode={self.spec.mode!r}, detail={self.spec.retained_detail_rank}, "
            f"digest={self.basis_digest[:12]}"
        )


__all__ = ["BSplineActionRepresentation", "BSplinePayload"]
