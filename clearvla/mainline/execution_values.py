"""One outlet-owned arm/gripper metric for execution learning and decisions.

This ranks internal computation candidates, not robot success or a stop command.
Binary command CE belongs to a separate terminal head. Its unused continuous
value coordinate must not influence operation, dwell or termination choices.
The two-coordinate reader/checkpoint layout is retained for compatibility.
"""

from __future__ import annotations

from torch import Tensor

from .gripper_contract import (
    CONTINUOUS_GRIPPER_OUTPUT_MODE,
    VALID_GRIPPER_OUTPUT_MODES,
    is_binary_gripper_mode,
)


def execution_value_component_weights(
    reference: Tensor,
    *,
    arm_dim: int,
    gripper_output_mode: str = CONTINUOUS_GRIPPER_OUTPUT_MODE,
) -> Tensor:
    """Return the existing supervision metric, without a new loss budget."""
    if gripper_output_mode not in VALID_GRIPPER_OUTPUT_MODES:
        raise ValueError("unknown execution-value gripper output mode")
    if is_binary_gripper_mode(gripper_output_mode):
        return reference.new_tensor([1.0, 0.0])
    arm_dim = max(int(arm_dim), 1)
    return reference.new_tensor([float(arm_dim), 1.0]) / float(arm_dim + 1)


def execution_value_score(
    value_field: Tensor,
    *,
    arm_dim: int,
    gripper_output_mode: str = CONTINUOUS_GRIPPER_OUTPUT_MODE,
) -> Tensor:
    """Reduce [B,candidate,horizon,arm/gripper] to the supervised score.

    Do not emulate binary isolation with ``0 * unused``: the compatibility
    coordinate is not part of the decision metric at all. Continuous arithmetic
    is kept in its original expression order and dtype.
    """
    if value_field.ndim != 4 or int(value_field.shape[-1]) != 2:
        raise ValueError("execution value field must be [B,candidate,horizon,2]")
    weight = execution_value_component_weights(
        value_field, arm_dim=arm_dim, gripper_output_mode=gripper_output_mode
    )
    if is_binary_gripper_mode(gripper_output_mode):
        return value_field[..., 0].mean(dim=-1)
    return (value_field * weight).sum(dim=-1).mean(dim=-1)
