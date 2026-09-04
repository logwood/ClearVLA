"""Stable public API for ClearVLA's standalone B-spline action chart."""

from .basis import (
    NUMERICAL_PREFLIGHT_DENSE_SAMPLES,
    NUMERICAL_PREFLIGHT_SCHEMA,
    BSplineBasisBundle,
    basis_preflight,
    bspline_basis,
    build_basis_bundle,
    canonical_qr_completion,
    not_a_knot_interpolation_knots,
    open_uniform_knots,
)
from .compat import (
    NativeActionSplinePayload,
    PhysicalActionCodecProtocol,
    PhysicalActionFieldBSplineAdapter,
)
from .representation import BSplineActionRepresentation, BSplinePayload
from .spec import BSplineSpec, RepresentationMode

__all__ = [
    "BSplineActionRepresentation",
    "BSplineBasisBundle",
    "BSplinePayload",
    "BSplineSpec",
    "NativeActionSplinePayload",
    "NUMERICAL_PREFLIGHT_DENSE_SAMPLES",
    "NUMERICAL_PREFLIGHT_SCHEMA",
    "PhysicalActionCodecProtocol",
    "PhysicalActionFieldBSplineAdapter",
    "RepresentationMode",
    "basis_preflight",
    "bspline_basis",
    "build_basis_bundle",
    "canonical_qr_completion",
    "not_a_knot_interpolation_knots",
    "open_uniform_knots",
]
