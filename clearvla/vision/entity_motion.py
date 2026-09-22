"""Read estimated motion where the current entity actually has image support.

This is a spatial expectation of a causal flow estimate, not a calibrated
physical velocity or an observation-validity decision. K and camera axes are
preserved; the old query-anchor flow is only an explicit legacy control.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .entity_chart import ImageLogMeasure
from .entity_history import ObservedEntityHistory

QUERY_ANCHOR_MOTION = "query_anchor_v1"
CURRENT_ENTITY_MOTION = "current_entity_support_v1"


def entity_motion_metadata(mode: str) -> dict[str, object]:
    if mode not in (QUERY_ANCHOR_MOTION, CURRENT_ENTITY_MOTION):
        raise ValueError("unknown entity motion mode")
    return {
        "mode": mode,
        "value": "estimated_current_minus_previous_normalized_image_xy",
        "spatial_read": (
            "same_current_entity_image_measure_per_camera"
            if mode == CURRENT_ENTITY_MOTION
            else "legacy_query_anchor_then_camera_aggregation"
        ),
        "time": "latest_actual_source_pair_control_steps",
        "missing_pair": "zero_value_and_zero_source_duration_not_static_observation",
        "validity": "unchanged_source_support_not_flow_confidence_or_visibility",
        "camera": "retained_at_G3_boundary_not_world_frame_velocity",
    }


def current_entity_motion(measure: ImageLogMeasure, history: ObservedEntityHistory) -> Tensor:
    """Compute E[current - previous | current entity, camera].

    The inverse map is defined on the current chart, so its negative is exactly
    the displacement associated with each *current* cell. The entity's own
    current image law weights those cells; no query-grid anchor, Gaussian
    approximation or cross-camera average is used. Log conditioning happens
    before exponentiation, including for very small global camera mass.

    Values remain flow estimates even where the estimated previous location
    leaves the image. Such extrapolation is not relabelled as an observed
    correspondence, nor is its learned confidence allowed to mask labels or
    current facts. A physically absent pair, unlike an uncertain estimate, is
    removed using the source clock before arithmetic.
    """
    history.validate()
    probability, _ = measure.normalized((-2, -1))
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("entity motion requires a finite supported spatial law")
    expected = (history.content.shape[0], *history.content.shape[2:5])
    if (probability.shape[0], *probability.shape[2:]) != expected:
        raise ValueError("entity motion and current image must share B/C/Y/X")
    if probability.device != history.content.device:
        raise ValueError("entity motion and its source must share device")
    real_pair = history.source_time.pair_observed[:, -1, None, None, None, None]
    inverse = torch.where(real_pair, history.backward_flow[:, -1], 0.0)
    displacement = -inverse.float().permute(0, 1, 3, 4, 2)
    # This expectation uses the SAME normalized law as current camera centers.
    # Confidence/occlusion, inferred correspondence coverage and the independent
    # reconstruction target are not used to reweight the law.
    return (probability[..., None] * displacement[:, None]).sum((-3, -2))
