"""Source-owned proprioceptive charts, independent of native action coordinates.

Statistics stay fitted on native observed state. A feature chart is an explicit
model input ABI, not an action codec or a calibration inferred from data extrema.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

NATIVE_AFFINE_STATE = "native_affine_v1"
CALVIN_ROTATION6D_STATE = "calvin_tcp_rotation6d_v1"
STATE_FEATURE_MODES = (NATIVE_AFFINE_STATE, CALVIN_ROTATION6D_STATE)


class StateNormalizer(Protocol):
    @property
    def offset(self) -> np.ndarray: ...
    @property
    def scale(self) -> np.ndarray: ...
    def encode(self, value: np.ndarray) -> np.ndarray: ...


def state_feature_width(mode: str, native_width: int) -> int:
    if mode not in STATE_FEATURE_MODES:
        raise ValueError(f"unknown state feature mode: {mode!r}")
    if type(native_width) is not int or native_width < 1:
        raise ValueError("native state width must be a positive integer")
    if mode == CALVIN_ROTATION6D_STATE:
        if native_width != 7:
            raise ValueError("CALVIN rotation feature chart requires native width seven")
        return 10
    return native_width


def validate_state_feature_profile(mode: str, profile: str) -> None:
    if mode not in STATE_FEATURE_MODES:
        raise ValueError(f"unknown state feature mode: {mode!r}")
    if mode == CALVIN_ROTATION6D_STATE and profile != "calvin_relative_7d_v1":
        raise ValueError("CALVIN Euler feature chart cannot be applied to another native chart")


def state_feature_metadata(mode: str, profile: str, native_width: int) -> dict[str, object]:
    validate_state_feature_profile(mode, profile)
    result: dict[str, object] = {
        "contract": mode,
        "profile": profile,
        "native_state_dim": native_width,
        "feature_state_dim": state_feature_width(mode, native_width),
        "normalizer_domain": "native_projected_state_not_action",
        "history_clock_unit": "physical_control_step",
        "seconds_per_step": None,
    }
    if mode == CALVIN_ROTATION6D_STATE:
        result.update(
            {
                "native_order": ["x", "y", "z", "roll", "pitch", "yaw", "opening"],
                "native_position_unit": "m",
                "native_orientation_unit": "rad",
                "native_opening_unit": "m",
                "orientation_matrix": "Rz(yaw) @ Ry(pitch) @ Rx(roll)",
                "feature_order": [
                    "x_affine",
                    "y_affine",
                    "z_affine",
                    "R00",
                    "R10",
                    "R20",
                    "R01",
                    "R11",
                    "R21",
                    "opening_affine",
                ],
                "rotation_encoding": "first_two_matrix_columns_no_affine_rotation_scaling",
                "history_change": "feature_chord_difference_per_physical_control_step_not_twist",
                "inverse_role": "observation_feature_only_not_action_decode",
            }
        )
    return result


def euler_xyz_rotation_columns(euler: np.ndarray) -> np.ndarray:
    """Return columns 0 and 1 of Rz(yaw) Ry(pitch) Rx(roll), in column order.

    Native radians are transformed before normalization. Equivalent Euler
    coordinates describe the same feature, including across +/-pi. No inverse
    Euler chart or independent per-axis angle difference is required.
    """
    a = np.asarray(euler)
    if a.ndim < 1 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("Euler state must be finite [...,3] native radians")
    roll, pitch, yaw = np.moveaxis(a.astype(np.float64, copy=False), -1, 0)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    return np.stack(
        (cy * cp, sy * cp, -sp, cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr), axis=-1
    ).astype(np.float32)


def encode_state_features(
    native: np.ndarray, normalizer: StateNormalizer, *, mode: str, profile: str
) -> np.ndarray:
    """Shared dataset current/history/future and online proprioception ingress.

    Native commands and action-state never enter this function. The legacy
    mode retains the original affine arithmetic. Unknown source charts fail
    rather than treating joint angles or rotvecs as CALVIN Euler orientation.
    """
    validate_state_feature_profile(mode, profile)
    source = np.asarray(native)
    if source.dtype.kind not in "fiu":
        raise ValueError("native state must contain real numeric coordinates")
    value = np.asarray(source, dtype=np.float32)
    if value.ndim < 1 or not np.isfinite(value).all():
        raise ValueError("native state must have a finite final coordinate axis")
    width = int(value.shape[-1])
    state_feature_width(mode, width)
    for label, field in (("offset", normalizer.offset), ("scale", normalizer.scale)):
        a = np.asarray(field)
        if a.shape not in {(width,), (1, width)} or not np.isfinite(a).all():
            raise ValueError(f"state normalizer {label} must belong to native state width {width}")
    if np.any(np.asarray(normalizer.scale) <= 0):
        raise ValueError("state normalizer scales must be finite and positive")
    if mode == NATIVE_AFFINE_STATE:
        result = normalizer.encode(value)
    else:
        # Never compute unused affine Euler values: extreme angular statistics
        # must not overflow an otherwise well-defined rotation representation.
        linear_indices = [0, 1, 2, 6]
        scale = np.asarray(normalizer.scale, dtype=np.float32).reshape(-1)[linear_indices]
        offset = np.asarray(normalizer.offset, dtype=np.float32).reshape(-1)[linear_indices]
        linear = value[..., linear_indices] * scale + offset
        rotation = euler_xyz_rotation_columns(value[..., 3:6])
        result = np.concatenate((linear[..., :3], rotation, linear[..., 3:4]), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("state feature encoding overflowed")
    return np.ascontiguousarray(result).reshape(*value.shape[:-1], state_feature_width(mode, width))
