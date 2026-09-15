"""Explicit 7D ManiSkill state charts; causal and invertible rotation encoding.

The fixed down-facing reference moves Panda's ordinary pi-pitch posture away
from the log cut. Nearest equivalent log branches use only the last observed
rotation. No globally nonsingular 3D SO(3) chart is claimed.
"""

import numpy as np

LEGACY_STATE_CHART = "principal_rotvec_v1"
CONTINUOUS_STATE_CHART = "fixed_down_causal_rotvec_v2"
LEGACY_STATE_SEMANTICS = (
    "[tcp xyz metres, tcp orientation rotation-vector radians, two-finger opening width metres]"
)
CONTINUOUS_STATE_SEMANTICS = "[tcp xyz metres, causal rotation-vector of Rx(pi)^-1 R_tcp radians, two-finger opening width metres]"


def normalized_quaternion(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) <= 1e-12:
        raise ValueError("orientation must be finite nonzero wxyz quaternion")
    return q / np.linalg.norm(q)


def principal_rotvec(value: np.ndarray) -> np.ndarray:
    q = normalized_quaternion(value)
    if q[0] < 0:
        q = -q
    length = np.linalg.norm(q[1:])
    if length <= 1e-10:
        return np.zeros(3, np.float32)
    angle = 2 * np.arctan2(length, np.clip(q[0], -1, 1))
    return (q[1:] * (angle / length)).astype(np.float32)


class CausalRotationChart:
    def __init__(self) -> None:
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def encode(self, quaternion_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = normalized_quaternion(quaternion_wxyz)
        # Hamilton product inverse([0,1,0,0]) * q. Reference is fixed,
        # never a future frame, dataset mean, or per-episode hidden goal.
        relative = np.array([x, -w, z, -y], dtype=np.float64)
        value = principal_rotvec(relative).astype(np.float64)
        angle = np.linalg.norm(value)
        if self.previous is not None:
            if angle > 1e-8:
                axis = value / angle
                turns = int(np.rint((np.dot(self.previous, axis) - angle) / (2 * np.pi)))
                value = axis * (angle + 2 * np.pi * turns)
            else:
                old_norm = np.linalg.norm(self.previous)
                if old_norm > 1e-8:
                    value = self.previous / old_norm * (2 * np.pi * np.rint(old_norm / (2 * np.pi)))
        self.previous = value.copy()
        return value.astype(np.float32)
