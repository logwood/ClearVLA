"""Shared gripper-output ABI predicates.

The action field always keeps the historical six-coordinate gripper lane.  A
binary outlet does *not* change that field's width; it changes ownership of the
deployed command.  Keeping the mode names and predicates in this tiny module
prevents one layer from treating the ManiSkill command head as a CALVIN-only
special case (or, worse, falling back to continuous decoding).
"""

CONTINUOUS_GRIPPER_OUTPUT_MODE = "continuous"
CALVIN_BINARY_GRIPPER_OUTPUT_MODE = "calvin_binary_command"
MANISKILL_BINARY_GRIPPER_OUTPUT_MODE = "maniskill_binary_command"

VALID_GRIPPER_OUTPUT_MODES = frozenset(
    {
        CONTINUOUS_GRIPPER_OUTPUT_MODE,
        CALVIN_BINARY_GRIPPER_OUTPUT_MODE,
        MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
    }
)
BINARY_GRIPPER_OUTPUT_MODES = frozenset(
    {
        CALVIN_BINARY_GRIPPER_OUTPUT_MODE,
        MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
    }
)

CALVIN_BINARY_GRIPPER_SELECTION = "calvin_7d_binary_v1"
MANISKILL_BINARY_GRIPPER_SELECTION = "maniskill_7d_binary_v2"
BINARY_GRIPPER_SELECTIONS = frozenset(
    {CALVIN_BINARY_GRIPPER_SELECTION, MANISKILL_BINARY_GRIPPER_SELECTION}
)


def is_binary_gripper_mode(mode: object) -> bool:
    """Return whether ``mode`` owns a strict native ``{-1,+1}`` command."""

    return str(mode) in BINARY_GRIPPER_OUTPUT_MODES


def is_binary_gripper_selection(selection: object) -> bool:
    """Return whether an outlet selection owns the binary command head."""

    return str(selection) in BINARY_GRIPPER_SELECTIONS


__all__ = [
    "BINARY_GRIPPER_OUTPUT_MODES",
    "BINARY_GRIPPER_SELECTIONS",
    "CALVIN_BINARY_GRIPPER_OUTPUT_MODE",
    "CALVIN_BINARY_GRIPPER_SELECTION",
    "CONTINUOUS_GRIPPER_OUTPUT_MODE",
    "MANISKILL_BINARY_GRIPPER_OUTPUT_MODE",
    "MANISKILL_BINARY_GRIPPER_SELECTION",
    "VALID_GRIPPER_OUTPUT_MODES",
    "is_binary_gripper_mode",
    "is_binary_gripper_selection",
]
