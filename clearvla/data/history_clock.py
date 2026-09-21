"""Source-owned physical-step indices shared by training and online history.

These are observation/control times, never ODE time or a task-progress signal.
An observation at t precedes control a[t]. A negative requested row before the
reset is padding, not an observed state or an executed command.
"""

from __future__ import annotations

import numpy as np

STATE_OFFSETS = (-8, -4, 0)
VISUAL_OFFSETS = STATE_OFFSETS
EXECUTED_ACTION_OFFSETS = (-24, -16, -12, -8, -6, -4, -2, -1)
HISTORY_TIMING_CONTRACT = "physical_step_sparse_history_v1"
HISTORY_TIMING_KEYS = (
    "history_state_offsets",
    "history_action_offsets",
    "history_state_observed",
    "history_action_executed",
)


def sparse_history_clock(
    time_index: int,
    *,
    state_offsets: tuple[int, ...] = STATE_OFFSETS,
    action_offsets: tuple[int, ...] = EXECUTED_ACTION_OFFSETS,
) -> dict[str, np.ndarray]:
    """Describe actual state sources and requested control times relative to now.

    Repeated reset states use their actual source time and observed=False.
    Missing actions keep the requested time but executed=False; no command is
    invented. Absolute episode position is not emitted as a model feature.
    """
    if isinstance(time_index, bool) or not isinstance(time_index, (int, np.integer)):
        raise TypeError("history time_index must be an integer physical step")
    now = int(time_index)
    if now < 0:
        raise ValueError("history time_index cannot precede reset")
    if (
        not state_offsets
        or state_offsets[-1] != 0
        or tuple(sorted(set(state_offsets))) != state_offsets
        or any(isinstance(x, bool) or not isinstance(x, int) for x in state_offsets)
    ):
        raise ValueError("state offsets must be distinct ordered steps ending at zero")
    if (
        not action_offsets
        or action_offsets[-1] >= 0
        or tuple(sorted(set(action_offsets))) != action_offsets
        or any(isinstance(x, bool) or not isinstance(x, int) for x in action_offsets)
    ):
        raise ValueError("executed-action offsets must be distinct ordered past steps")
    states = np.asarray(state_offsets, dtype=np.int64) + now
    actions = np.asarray(action_offsets, dtype=np.int64) + now
    return {
        "history_state_offsets": np.maximum(states, 0) - now,
        "history_action_offsets": actions - now,
        "history_state_observed": states >= 0,
        "history_action_executed": actions >= 0,
    }
