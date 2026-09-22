"""Declared geometry-value semantics at the W -> P2 -> P3 boundary."""

from __future__ import annotations

from .future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time

POOLED_TRANSPORT = "pooled_transport_v1"
VIEW_CONDITIONED_TRANSPORT = "view_conditioned_transport_v1"
P2_GEOMETRY_MODES = (POOLED_TRANSPORT, VIEW_CONDITIONED_TRANSPORT)


def validate_camera_names(names: tuple[str, ...]) -> None:
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) or not n for n in names)
    ):
        raise ValueError("P2 geometry requires nonempty unique camera chart names")


def p2_geometry_metadata(camera_names: tuple[str, ...]) -> dict[str, object]:
    validate_camera_names(camera_names)
    return {
        "schema": "p2-view-conditioned-transport-v1",
        "mode": VIEW_CONDITIONED_TRANSPORT,
        "camera_names": list(camera_names),
        "role_basis": sorted(camera_names),
        "input": "current-image-normalized-xy-displacement-per-named-camera",
        "time_grid": resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME).metadata(),
        "transport_statistic": "mean-at-declared-sparse-successor-supports-not-endpoint",
        "covariance_statistic": "mean-within-support-correspondence-covariance-not-task-risk",
        "spatial_query": "separate-learned-image-xy-per-named-view-not-calibrated-projection",
        "value": "zero-preserving-view-conditioned-features-before-camera-or-object-pooling",
        "target": "shared-object-marginal-with-null-no-renormalization",
        "terminal": "features-not-physical-xy-no-second-transport-projection",
        "feedback": "interval-statistics-not-one-step-execution-error",
    }
