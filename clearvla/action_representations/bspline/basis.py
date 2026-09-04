"""Deterministic Torch-only B-spline basis construction and diagnostics."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor

from .spec import BSplineSpec

NUMERICAL_PREFLIGHT_SCHEMA = "clearvla-bspline-runtime-safe-v1"
NUMERICAL_PREFLIGHT_DENSE_SAMPLES = 513

# These are construction-time numerical safety limits, not task-performance
# thresholds or tunable smoothing gains.  They are intentionally fixed by the
# source identity so a caller cannot silently waive them for a runtime object.
_NUMERICAL_PREFLIGHT_LIMITS = {
    "coarse_condition_number": 1_000.0,
    "interpolation_condition_number": 1_000.0,
    "coarse_analysis_operator_2": 1_000.0,
    "interpolation_analysis_operator_2": 1_000.0,
    "dense_sample_linf_operator_norm": 64.0,
    "dense_velocity_linf_operator_norm_per_average_step": 128.0,
    "dense_acceleration_linf_operator_norm_per_average_step2": 512.0,
    "runtime_float32_sample_evaluation_closure_max_abs": 1.0e-3,
    "runtime_float32_dense_sample_linf_operator_norm": 64.0,
    "runtime_float32_dense_velocity_linf_operator_norm_per_average_step": 128.0,
    "runtime_float32_dense_acceleration_linf_operator_norm_per_average_step2": 512.0,
}

_NUMERICAL_PREFLIGHT_REQUIRED_TRUE = (
    "runtime_float32_buffers_finite",
    "runtime_float32_sample_times_strictly_increasing",
    "runtime_float32_knots_nondecreasing",
    "runtime_float32_knot_topology_preserved",
    "runtime_float32_coarse_domain_valid",
    "runtime_float32_interpolation_domain_valid",
    "runtime_float32_operators_finite",
)

_RUNTIME_FLOAT32_BUFFER_NAMES = (
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


@dataclass(frozen=True)
class BSplineBasisBundle:
    """Canonical float64 CPU matrices from which runtime buffers are built."""

    sample_times: Tensor
    coarse_knots: Tensor
    interpolation_knots: Tensor
    coarse_collocation: Tensor
    interpolation_collocation: Tensor
    coarse_q: Tensor
    coarse_r: Tensor
    detail_q: Tensor
    detail_control_map: Tensor
    digest: str

    @property
    def horizon(self) -> int:
        return int(self.sample_times.numel())

    @property
    def coarse_rank(self) -> int:
        return int(self.coarse_q.shape[1])

    @property
    def detail_rank(self) -> int:
        return int(self.detail_q.shape[1])


def _as_float_times(values: Tensor | Iterable[float], *, name: str) -> Tensor:
    if isinstance(values, Tensor):
        result = values
        if not result.is_floating_point():
            result = result.to(dtype=torch.float64)
    else:
        result = torch.tensor(tuple(values), dtype=torch.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.numel() < 1:
        raise ValueError(f"{name} cannot be empty")
    if not bool(torch.isfinite(result.detach()).all()):
        raise ValueError(f"{name} must be finite")
    return result


def open_uniform_knots(
    sample_times: Tensor | Iterable[float],
    num_control_points: int,
    degree: int,
) -> Tensor:
    """Build an open-uniform clamped knot vector over the timestamp domain."""

    times = _as_float_times(sample_times, name="sample_times")
    num_control_points = int(num_control_points)
    degree = int(degree)
    if times.numel() < 2 or not bool((times[1:] > times[:-1]).all()):
        raise ValueError("sample_times must be strictly increasing")
    if degree < 0 or num_control_points < degree + 1:
        raise ValueError("num_control_points must be at least degree + 1")
    interior_count = num_control_points - degree - 1
    pieces = [times[0].repeat(degree + 1)]
    if interior_count:
        interior = torch.linspace(
            times[0],
            times[-1],
            interior_count + 2,
            device=times.device,
            dtype=times.dtype,
        )[1:-1]
        pieces.append(interior)
    pieces.append(times[-1].repeat(degree + 1))
    return torch.cat(pieces)


def not_a_knot_interpolation_knots(
    sample_times: Tensor | Iterable[float],
    degree: int,
) -> Tensor:
    """Return the stable full-rank not-a-knot interpolation knot vector.

    For odd degrees the retained interior knots are sample locations.  For
    even degrees they are adjacent-sample midpoints.  This is the standard
    not-a-knot construction and avoids the poorly conditioned ``K=T``
    open-uniform cubic chart.
    """

    times = _as_float_times(sample_times, name="sample_times")
    degree = int(degree)
    if degree not in (1, 2, 3):
        raise ValueError("interpolation degree must be 1, 2 or 3")
    if times.numel() < degree + 1:
        raise ValueError("not-a-knot interpolation needs at least degree + 1 samples")
    if not bool((times[1:] > times[:-1]).all()):
        raise ValueError("sample_times must be strictly increasing")

    half = (degree + 1) // 2 if degree % 2 else degree // 2
    candidates = times if degree % 2 else 0.5 * (times[:-1] + times[1:])
    interior = candidates[half:-half]
    return torch.cat(
        (
            times[0].repeat(degree + 1),
            interior,
            times[-1].repeat(degree + 1),
        )
    )


def bspline_basis(
    query_times: Tensor | Iterable[float],
    knots: Tensor,
    degree: int,
    *,
    derivative_order: int = 0,
) -> Tensor:
    """Evaluate a B-spline basis or its first two analytical derivatives.

    The right endpoint uses the polynomial's left-hand limit.  Basis values at
    that endpoint are then made exactly clamped (all zero except the final
    basis value), while derivative calls retain the correct left derivative.
    """

    degree = int(degree)
    derivative_order = int(derivative_order)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if derivative_order not in (0, 1, 2):
        raise ValueError("derivative_order must be 0, 1 or 2")
    if not isinstance(knots, Tensor) or knots.ndim != 1:
        raise ValueError("knots must be a one-dimensional tensor")
    if not knots.is_floating_point():
        raise TypeError("knots must use a floating-point dtype")
    if knots.numel() < 2 * degree + 2:
        raise ValueError("knot vector is too short for the requested degree")
    if not bool(torch.isfinite(knots.detach()).all()):
        raise ValueError("knots must be finite")
    if not bool((knots[1:] >= knots[:-1]).all()):
        raise ValueError("knots must be non-decreasing")

    source_query = _as_float_times(query_times, name="query_times")
    comparison_dtype = torch.promote_types(source_query.dtype, knots.dtype)
    comparison_query = source_query.to(device=knots.device, dtype=comparison_dtype)
    original_shape = tuple(source_query.shape)
    comparison_query = comparison_query.reshape(-1)
    domain_start = knots[degree]
    domain_stop = knots[-degree - 1]
    if not bool(domain_stop > domain_start):
        raise ValueError("spline domain must have positive span")
    domain_span = domain_stop - domain_start
    negative_infinity = torch.full_like(domain_start, -torch.inf)
    positive_infinity = torch.full_like(domain_stop, torch.inf)
    lower_ulp = domain_start - torch.nextafter(domain_start, negative_infinity)
    upper_ulp = torch.nextafter(domain_stop, positive_infinity) - domain_stop
    # Half an endpoint ULP admits only source values that round to the runtime
    # endpoint.  Unlike an absolute 1.0 scale floor, this has the same physical
    # units as the time coordinate and cannot swallow a small entire domain.
    lower_tolerance = torch.minimum(0.5 * lower_ulp, 0.5 * domain_span).to(
        dtype=comparison_dtype
    )
    upper_tolerance = torch.minimum(0.5 * upper_ulp, 0.5 * domain_span).to(
        dtype=comparison_dtype
    )
    comparison_start = domain_start.to(dtype=comparison_dtype)
    comparison_stop = domain_stop.to(dtype=comparison_dtype)
    detached = comparison_query.detach()
    if bool(((comparison_start.detach() - detached) > lower_tolerance).any()) or bool(
        ((detached - comparison_stop.detach()) > upper_tolerance).any()
    ):
        raise ValueError("query_times fall outside the closed spline domain")
    query = comparison_query.to(dtype=knots.dtype)
    query = query.clamp(min=domain_start, max=domain_stop)
    at_right_endpoint = query == domain_stop
    left_limit = torch.nextafter(domain_stop, domain_start)
    evaluation_time = torch.where(at_right_endpoint, left_limit, query)

    left_edges = knots[:-1]
    right_edges = knots[1:]
    current = (
        (evaluation_time[:, None] >= left_edges[None, :])
        & (evaluation_time[:, None] < right_edges[None, :])
    ).to(dtype=knots.dtype)
    first = torch.zeros_like(current)
    second = torch.zeros_like(current)

    for recursion_degree in range(1, degree + 1):
        column_count = int(current.shape[-1]) - 1
        basis_columns: list[Tensor] = []
        first_columns: list[Tensor] = []
        second_columns: list[Tensor] = []
        for index in range(column_count):
            left_denominator = knots[index + recursion_degree] - knots[index]
            right_denominator = knots[index + recursion_degree + 1] - knots[index + 1]
            left_nonzero = left_denominator != 0
            right_nonzero = right_denominator != 0
            safe_left = torch.where(
                left_nonzero, left_denominator, torch.ones_like(left_denominator)
            )
            safe_right = torch.where(
                right_nonzero, right_denominator, torch.ones_like(right_denominator)
            )
            left_scale = left_nonzero.to(knots.dtype) / safe_left
            right_scale = right_nonzero.to(knots.dtype) / safe_right
            left_weight = (evaluation_time - knots[index]) * left_scale
            right_weight = (knots[index + recursion_degree + 1] - evaluation_time) * right_scale

            left_basis = current[:, index]
            right_basis = current[:, index + 1]
            left_first = first[:, index]
            right_first = first[:, index + 1]
            left_second = second[:, index]
            right_second = second[:, index + 1]
            basis_columns.append(left_weight * left_basis + right_weight * right_basis)
            first_columns.append(
                left_scale * left_basis
                + left_weight * left_first
                - right_scale * right_basis
                + right_weight * right_first
            )
            second_columns.append(
                2.0 * left_scale * left_first
                + left_weight * left_second
                - 2.0 * right_scale * right_first
                + right_weight * right_second
            )
        current = torch.stack(basis_columns, dim=-1)
        first = torch.stack(first_columns, dim=-1)
        second = torch.stack(second_columns, dim=-1)

    if derivative_order == 0:
        if bool(at_right_endpoint.any()):
            endpoint = torch.zeros_like(current)
            endpoint[:, -1] = 1.0
            current = torch.where(at_right_endpoint[:, None], endpoint, current)
        result = current
    elif derivative_order == 1:
        result = first
    else:
        result = second
    return result.reshape(*original_shape, int(result.shape[-1]))


def canonical_qr_completion(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return unique-sign reduced QR plus a deterministic orthogonal complement."""

    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("matrix must be a floating-point rank-two tensor")
    rows, columns = (int(matrix.shape[0]), int(matrix.shape[1]))
    if rows < columns or columns < 1:
        raise ValueError("matrix must be tall or square with at least one column")
    q_coarse, r_coarse = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r_coarse)
    threshold = torch.finfo(matrix.dtype).eps * max(rows, columns) * 64.0
    if bool((diagonal.abs() <= threshold * diagonal.abs().max().clamp_min(1.0)).any()):
        raise ValueError("B-spline collocation matrix is rank deficient")
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q_coarse = q_coarse * signs[None, :]
    r_coarse = r_coarse * signs[:, None]

    required = rows - columns
    if required == 0:
        return q_coarse, r_coarse, matrix.new_empty(rows, 0)

    complement: list[Tensor] = []
    identity = torch.eye(rows, device=matrix.device, dtype=matrix.dtype)
    for index in range(rows):
        vector = identity[:, index]
        # Two modified Gram-Schmidt passes keep the completion at float64
        # numerical floor without relying on a backend-specific complete QR.
        for _ in range(2):
            vector = vector - q_coarse @ (q_coarse.T @ vector)
            if complement:
                q_detail = torch.stack(complement, dim=-1)
                vector = vector - q_detail @ (q_detail.T @ vector)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) <= threshold:
            continue
        vector = vector / norm
        pivot = int(vector.abs().argmax())
        if float(vector[pivot]) < 0.0:
            vector = -vector
        complement.append(vector)
        if len(complement) == required:
            break
    if len(complement) != required:
        raise RuntimeError("failed to construct the full deterministic detail complement")
    return q_coarse, r_coarse, torch.stack(complement, dim=-1)


def _update_digest(hasher: object, name: str, tensor: Tensor) -> None:
    digest = hasher
    assert hasattr(digest, "update")
    digest.update(name.encode("utf-8"))  # type: ignore[attr-defined]
    digest.update(struct.pack("<I", tensor.ndim))  # type: ignore[attr-defined]
    for dimension in tensor.shape:
        digest.update(struct.pack("<Q", int(dimension)))  # type: ignore[attr-defined]
    for value in tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist():
        digest.update(struct.pack("<d", float(value)))  # type: ignore[attr-defined]


def _build_basis_bundle_unchecked(spec: BSplineSpec) -> BSplineBasisBundle:
    """Build structural matrices before the runtime numerical preflight.

    This helper is deliberately private.  Public runtime construction goes
    through :func:`build_basis_bundle`; diagnostics-only callers use
    :func:`basis_preflight`, which never returns unchecked matrices.
    """

    spec.validate()
    sample_times = torch.tensor(spec.sample_times, dtype=torch.float64, device="cpu")
    coarse_knots = open_uniform_knots(
        sample_times,
        spec.num_control_points,
        spec.degree,
    )
    coarse = bspline_basis(sample_times, coarse_knots, spec.degree)
    q_coarse, r_coarse, q_detail = canonical_qr_completion(coarse)

    interpolation_knots = not_a_knot_interpolation_knots(sample_times, spec.degree)
    interpolation = bspline_basis(sample_times, interpolation_knots, spec.degree)
    singular = torch.linalg.svdvals(interpolation)
    tolerance = torch.finfo(torch.float64).eps * spec.horizon * singular.max() * 64.0
    if float(singular.min()) <= float(tolerance):
        raise ValueError("full interpolation collocation is numerically rank deficient")
    if q_detail.shape[1]:
        detail_control_map = torch.linalg.solve(interpolation, q_detail)
    else:
        detail_control_map = torch.empty(spec.horizon, 0, dtype=torch.float64)

    full_q = torch.cat((q_coarse, q_detail), dim=-1)
    identity = torch.eye(spec.horizon, dtype=torch.float64)
    if float((full_q.T @ full_q - identity).abs().max()) > 1e-11:
        raise RuntimeError("canonical coarse/detail basis is not orthonormal")
    if q_detail.shape[1]:
        closure = interpolation @ detail_control_map
        if float((closure - q_detail).abs().max()) > 1e-11:
            raise RuntimeError("detail interpolation chart does not close on sample times")

    hasher = hashlib.sha256()
    # The numerical basis identity is deliberately separate from the complete
    # spec fingerprint.  Metadata, arm width and the number of retained detail
    # axes do not change these temporal matrices.
    hasher.update(f"{spec.REPRESENTATION_NAME}:canonical_basis_bundle_v1".encode("ascii"))
    for name, tensor in (
        ("sample_times", sample_times),
        ("coarse_knots", coarse_knots),
        ("interpolation_knots", interpolation_knots),
        ("coarse_collocation", coarse),
        ("coarse_q", q_coarse),
        ("coarse_r", r_coarse),
        ("detail_q", q_detail),
        ("detail_control_map", detail_control_map),
    ):
        _update_digest(hasher, name, tensor)

    return BSplineBasisBundle(
        sample_times=sample_times,
        coarse_knots=coarse_knots,
        interpolation_knots=interpolation_knots,
        coarse_collocation=coarse,
        interpolation_collocation=interpolation,
        coarse_q=q_coarse,
        coarse_r=r_coarse,
        detail_q=q_detail,
        detail_control_map=detail_control_map,
        digest=hasher.hexdigest(),
    )


def _numerical_preflight_failures(
    report: dict[str, float | int | str | bool],
) -> tuple[str, ...]:
    failures: list[str] = []
    for invariant in _NUMERICAL_PREFLIGHT_REQUIRED_TRUE:
        if not bool(report[invariant]):
            failures.append(invariant)
    for metric, limit in _NUMERICAL_PREFLIGHT_LIMITS.items():
        actual = float(report[metric])
        if not math.isfinite(actual) or actual > limit:
            failures.append(metric)
    return tuple(failures)


def basis_preflight(spec: BSplineSpec) -> dict[str, float | int | str | bool]:
    """Inspect one spec without creating an unchecked runtime basis object.

    Rank-deficient specifications still raise because no meaningful chart
    exists.  Full-rank specifications that amplify excessively or become
    unsafe in the actual float32 runtime representation return a report with
    ``numerical_preflight_passed=False``; only this diagnostics entry point can
    inspect them without weakening runtime construction.
    """

    bundle = _build_basis_bundle_unchecked(spec)
    return structural_diagnostics(
        bundle,
        spec,
        dense_samples=NUMERICAL_PREFLIGHT_DENSE_SAMPLES,
    )


def build_basis_bundle(spec: BSplineSpec) -> BSplineBasisBundle:
    """Build a canonical basis and fail closed on float64/float32 risk."""

    bundle = _build_basis_bundle_unchecked(spec)
    report = structural_diagnostics(
        bundle,
        spec,
        dense_samples=NUMERICAL_PREFLIGHT_DENSE_SAMPLES,
    )
    failures = _numerical_preflight_failures(report)
    if failures:
        details = ", ".join(
            (
                f"{name}={float(report[name]):.6g}>"
                f"{_NUMERICAL_PREFLIGHT_LIMITS[name]:.6g}"
                if name in _NUMERICAL_PREFLIGHT_LIMITS
                else f"{name}={report[name]!r} (required True)"
            )
            for name in failures
        )
        raise ValueError(
            f"B-spline numerical preflight {NUMERICAL_PREFLIGHT_SCHEMA} failed: "
            f"{details}. This specification is diagnostics-only and cannot "
            "construct a runtime representation."
        )
    return bundle


def coordinate_evaluation_matrix(
    bundle: BSplineBasisBundle,
    query_times: Tensor,
    *,
    degree: int,
    retained_detail_rank: int,
    derivative_order: int = 0,
) -> Tensor:
    """Map retained orthonormal coordinates to curve values at query times."""

    coarse_at_query = bspline_basis(
        query_times,
        bundle.coarse_knots.to(device=query_times.device, dtype=query_times.dtype),
        degree,
        derivative_order=derivative_order,
    )
    r = bundle.coarse_r.to(device=query_times.device, dtype=query_times.dtype)
    coarse_coordinate_map = torch.linalg.solve_triangular(
        r.T,
        coarse_at_query.T,
        upper=False,
    ).T
    retained_detail_rank = int(retained_detail_rank)
    if retained_detail_rank == 0:
        return coarse_coordinate_map
    if not 0 <= retained_detail_rank <= bundle.detail_rank:
        raise ValueError("retained_detail_rank is outside the available complement")
    interpolation_at_query = bspline_basis(
        query_times,
        bundle.interpolation_knots.to(device=query_times.device, dtype=query_times.dtype),
        degree,
        derivative_order=derivative_order,
    )
    detail_controls = bundle.detail_control_map[:, :retained_detail_rank].to(
        device=query_times.device,
        dtype=query_times.dtype,
    )
    detail_coordinate_map = interpolation_at_query @ detail_controls
    return torch.cat((coarse_coordinate_map, detail_coordinate_map), dim=-1)


def _runtime_float32_diagnostics(
    bundle: BSplineBasisBundle,
    spec: BSplineSpec,
    *,
    dense_samples: int,
) -> dict[str, float | int | str | bool]:
    """Inspect the exact float32 buffers and operators used at runtime."""

    runtime = {
        name: getattr(bundle, name).to(dtype=torch.float32)
        for name in _RUNTIME_FLOAT32_BUFFER_NAMES
    }
    buffers_finite = all(bool(torch.isfinite(value).all()) for value in runtime.values())
    sample_times = runtime["sample_times"]
    coarse_knots = runtime["coarse_knots"]
    interpolation_knots = runtime["interpolation_knots"]
    sample_times_increasing = buffers_finite and bool(
        (sample_times[1:] > sample_times[:-1]).all()
    )
    knots_nondecreasing = buffers_finite and bool(
        (coarse_knots[1:] >= coarse_knots[:-1]).all()
    ) and bool((interpolation_knots[1:] >= interpolation_knots[:-1]).all())
    coarse_unique_count = int(torch.unique_consecutive(coarse_knots).numel())
    interpolation_unique_count = int(
        torch.unique_consecutive(interpolation_knots).numel()
    )
    expected_coarse_unique_count = int(
        torch.unique_consecutive(bundle.coarse_knots).numel()
    )
    expected_interpolation_unique_count = int(
        torch.unique_consecutive(bundle.interpolation_knots).numel()
    )
    knot_topology_preserved = buffers_finite and (
        coarse_unique_count == expected_coarse_unique_count
        and interpolation_unique_count == expected_interpolation_unique_count
    )
    coarse_domain_valid = buffers_finite and bool(
        coarse_knots[spec.degree] < coarse_knots[-spec.degree - 1]
    )
    interpolation_domain_valid = buffers_finite and bool(
        interpolation_knots[spec.degree]
        < interpolation_knots[-spec.degree - 1]
    )
    minimum_step = (
        float((sample_times[1:] - sample_times[:-1]).min())
        if buffers_finite
        else math.nan
    )
    domain_span = (
        float(sample_times[-1] - sample_times[0]) if buffers_finite else math.nan
    )
    report: dict[str, float | int | str | bool] = {
        "runtime_dtype": "float32",
        "runtime_float32_buffers_finite": buffers_finite,
        "runtime_float32_sample_times_strictly_increasing": sample_times_increasing,
        "runtime_float32_knots_nondecreasing": knots_nondecreasing,
        "runtime_float32_knot_topology_preserved": knot_topology_preserved,
        "runtime_float32_coarse_domain_valid": coarse_domain_valid,
        "runtime_float32_interpolation_domain_valid": interpolation_domain_valid,
        "runtime_float32_coarse_knot_unique_count": coarse_unique_count,
        "runtime_float32_expected_coarse_knot_unique_count": (
            expected_coarse_unique_count
        ),
        "runtime_float32_interpolation_knot_unique_count": (
            interpolation_unique_count
        ),
        "runtime_float32_expected_interpolation_knot_unique_count": (
            expected_interpolation_unique_count
        ),
        "runtime_float32_minimum_sample_step": minimum_step,
        "runtime_float32_domain_span": domain_span,
        "runtime_float32_operators_finite": False,
        "runtime_float32_sample_evaluation_closure_max_abs": math.inf,
        "runtime_float32_dense_sample_linf_operator_norm": math.inf,
        "runtime_float32_dense_velocity_linf_operator_norm_per_average_step": (
            math.inf
        ),
        "runtime_float32_dense_acceleration_linf_operator_norm_per_average_step2": (
            math.inf
        ),
    }
    structural_runtime_safe = all(
        (
            buffers_finite,
            sample_times_increasing,
            knots_nondecreasing,
            knot_topology_preserved,
            coarse_domain_valid,
            interpolation_domain_valid,
        )
    )
    if not structural_runtime_safe:
        return report

    retained_q = torch.cat(
        (runtime["coarse_q"], runtime["detail_q"]),
        dim=-1,
    )[:, : spec.coordinate_rank]
    try:
        sample_coordinate_map = coordinate_evaluation_matrix(
            bundle,
            sample_times,
            degree=spec.degree,
            retained_detail_rank=spec.retained_detail_rank,
        )
        dense_times = torch.linspace(
            sample_times[0],
            sample_times[-1],
            dense_samples,
            dtype=torch.float32,
        )
        dense_coordinate_maps = {
            order: coordinate_evaluation_matrix(
                bundle,
                dense_times,
                degree=spec.degree,
                retained_detail_rank=spec.retained_detail_rank,
                derivative_order=order,
            )
            for order in (0, 1, 2)
        }
        sample_operator = sample_coordinate_map @ retained_q.T
        dense_operators = {
            order: coordinate_map @ retained_q.T
            for order, coordinate_map in dense_coordinate_maps.items()
        }
    except (RuntimeError, ValueError):
        return report

    operators_finite = bool(torch.isfinite(sample_operator).all()) and all(
        bool(torch.isfinite(operator).all()) for operator in dense_operators.values()
    )
    report["runtime_float32_operators_finite"] = operators_finite
    if not operators_finite:
        return report

    expected_sample_operator = retained_q @ retained_q.T
    average_step = domain_span / float(spec.horizon - 1)
    report["runtime_float32_sample_evaluation_closure_max_abs"] = float(
        (sample_operator - expected_sample_operator).abs().max()
    )
    report["runtime_float32_dense_sample_linf_operator_norm"] = float(
        dense_operators[0].abs().sum(dim=-1).max()
    )
    report["runtime_float32_dense_velocity_linf_operator_norm_per_average_step"] = (
        float(dense_operators[1].abs().sum(dim=-1).max()) * average_step
    )
    report[
        "runtime_float32_dense_acceleration_linf_operator_norm_per_average_step2"
    ] = float(dense_operators[2].abs().sum(dim=-1).max()) * average_step**2
    return report


def structural_diagnostics(
    bundle: BSplineBasisBundle,
    spec: BSplineSpec,
    *,
    dense_samples: int = 513,
) -> dict[str, float | int | str | bool]:
    """Compute fixed canonical and actual-float32 runtime diagnostics."""

    dense_samples = int(dense_samples)
    if dense_samples < 2:
        raise ValueError("dense_samples must be at least two")
    q_full = torch.cat((bundle.coarse_q, bundle.detail_q), dim=-1)
    retained_q = q_full[:, : spec.coordinate_rank]
    identity = torch.eye(spec.horizon, dtype=torch.float64)
    coarse_singular = torch.linalg.svdvals(bundle.coarse_collocation)
    interpolation_singular = torch.linalg.svdvals(bundle.interpolation_collocation)
    dense_times = torch.linspace(
        bundle.sample_times[0],
        bundle.sample_times[-1],
        dense_samples,
        dtype=torch.float64,
    )
    coordinate_maps = {
        order: coordinate_evaluation_matrix(
            bundle,
            dense_times,
            degree=spec.degree,
            retained_detail_rank=spec.retained_detail_rank,
            derivative_order=order,
        )
        for order in (0, 1, 2)
    }
    sample_to_dense = coordinate_maps[0] @ retained_q.T
    sample_to_velocity = coordinate_maps[1] @ retained_q.T
    sample_to_acceleration = coordinate_maps[2] @ retained_q.T
    average_step = float(bundle.sample_times[-1] - bundle.sample_times[0]) / float(
        spec.horizon - 1
    )
    leverage = retained_q.square().sum(dim=-1)
    detail_closure = bundle.interpolation_collocation @ bundle.detail_control_map - bundle.detail_q
    partition = bundle.coarse_collocation.sum(dim=-1)
    report: dict[str, float | int | str | bool] = {
        "schema_version": spec.schema_version,
        "numerical_preflight_schema": NUMERICAL_PREFLIGHT_SCHEMA,
        "basis_digest": bundle.digest,
        "spec_fingerprint": spec.fingerprint,
        "mode": spec.mode,
        "is_lossless": spec.is_lossless,
        "horizon": spec.horizon,
        "arm_dim": spec.arm_dim,
        "degree": spec.degree,
        "coarse_rank": spec.coarse_rank,
        "available_detail_rank": spec.available_detail_rank,
        "retained_detail_rank": spec.retained_detail_rank,
        "detail_selection_policy": "all_or_none",
        "coordinate_rank": spec.coordinate_rank,
        "coarse_condition_number": float(coarse_singular.max() / coarse_singular.min()),
        "interpolation_condition_number": float(
            interpolation_singular.max() / interpolation_singular.min()
        ),
        "coarse_analysis_operator_2": float(1.0 / coarse_singular.min()),
        "interpolation_analysis_operator_2": float(1.0 / interpolation_singular.min()),
        "orthogonality_max_abs": float((q_full.T @ q_full - identity).abs().max()),
        "coarse_partition_max_abs": float((partition - 1.0).abs().max()),
        "coarse_start_endpoint_max_abs": float(
            (bundle.coarse_collocation[0] - torch.eye(1, spec.coarse_rank, dtype=torch.float64)[0])
            .abs()
            .max()
        ),
        "coarse_stop_endpoint_max_abs": float(
            (
                bundle.coarse_collocation[-1]
                - torch.eye(1, spec.coarse_rank, dtype=torch.float64)[0].roll(spec.coarse_rank - 1)
            )
            .abs()
            .max()
        ),
        "detail_interpolation_closure_max_abs": float(
            detail_closure.abs().max() if detail_closure.numel() else 0.0
        ),
        "sample_reconstruction_max_abs": float((retained_q @ retained_q.T - identity).abs().max()),
        "retained_leverage_min": float(leverage.min()),
        "retained_leverage_max": float(leverage.max()),
        "retained_leverage_mean": float(leverage.mean()),
        "dense_sample_linf_operator_norm": float(sample_to_dense.abs().sum(dim=-1).max()),
        "dense_velocity_linf_operator_norm_per_average_step": float(
            sample_to_velocity.abs().sum(dim=-1).max()
        )
        * average_step,
        "dense_acceleration_linf_operator_norm_per_average_step2": float(
            sample_to_acceleration.abs().sum(dim=-1).max()
        )
        * average_step**2,
    }
    report.update(
        _runtime_float32_diagnostics(
            bundle,
            spec,
            dense_samples=dense_samples,
        )
    )
    failures = _numerical_preflight_failures(report)
    report["numerical_preflight_passed"] = not failures
    report["numerical_preflight_failure_count"] = len(failures)
    report["numerical_preflight_failures"] = ",".join(failures)
    for metric, limit in _NUMERICAL_PREFLIGHT_LIMITS.items():
        report[f"max_allowed_{metric}"] = limit
    return report


__all__ = [
    "BSplineBasisBundle",
    "NUMERICAL_PREFLIGHT_DENSE_SAMPLES",
    "NUMERICAL_PREFLIGHT_SCHEMA",
    "basis_preflight",
    "bspline_basis",
    "build_basis_bundle",
    "canonical_qr_completion",
    "coordinate_evaluation_matrix",
    "not_a_knot_interpolation_knots",
    "open_uniform_knots",
    "structural_diagnostics",
]
