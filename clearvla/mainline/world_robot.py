"""Current robot observation carried beside, never mixed into, W controls."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

NO_ROBOT_WORLD = "implicit_g_only_v1"
OBSERVED_ROBOT_VIEWS = "observed_state_views_v1"


def world_robot_metadata(*, state_mode: str, state_dim: int) -> dict[str, object]:
    return {
        "schema": "observed-robot-object-views-v1",
        "mode": OBSERVED_ROBOT_VIEWS,
        "source": "current-observable-history-state",
        "state_features": state_mode,
        "state_dim": state_dim,
        "geometry": "per-camera-image-xy-not-cartesian-relative-pose",
        "motion": "source-timed-image-displacement-per-control-step",
        "fusion": "relational-features-before-camera-reduction",
        "missing_motion": "explicit-zero-gap-indicator-not-stationary-evidence",
        "cache": "same-observation-for-candidate-supervised-and-refined-world",
    }


@dataclass(frozen=True)
class RobotWorldObservation:
    """No goal, action proposal, future row, task memory or fabricated 3D pose."""

    state: Tensor
    feature_mode: str

    def validate(self, *, batch: int, device: torch.device,
                 state_dim: int | None = None, feature_mode: str | None = None,
                 check_finite: bool = False) -> None:
        if self.state.ndim != 2 or self.state.shape[0] != batch or self.state.device != device:
            raise ValueError("W robot observation must be current [B,D] on the belief device")
        if not self.state.is_floating_point():
            raise TypeError("W robot observation must contain floating state features")
        if state_dim is not None and self.state.shape[1] != state_dim:
            raise ValueError("W robot observation state width differs from configured features")
        if not self.feature_mode or (feature_mode is not None and self.feature_mode != feature_mode):
            raise ValueError("W robot observation feature chart differs from configured state")
        # Materialization only; the ODE cache uses structural checks without a
        # host-device synchronization at each numerical solver node.
        if check_finite and not bool(torch.isfinite(self.state).all()):
            raise ValueError("observed current robot state must be finite")
