"""Source provenance for fixed-shape future supervision, never policy input.

A source observation o[t] precedes command a[t]; terminal is the final real
observation, not a command. Storage may contain an absorbing suffix, but that
suffix is not observed physical future and is not admitted by this clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FUTURE_SUPPORT_KEYS = (
    "future_action_observed",
    "future_state_observed",
    "future_visual_observed",
)
OBSERVED_FUTURE_CONTRACT = "observed_future_support_v1"


@dataclass(frozen=True)
class FutureSourceRows:
    action_indices: np.ndarray
    state_indices: np.ndarray
    visual_indices: np.ndarray
    action_observed: np.ndarray
    state_observed: np.ndarray
    visual_observed: np.ndarray

    def support_mapping(self) -> dict[str, np.ndarray]:
        return dict(
            zip(
                FUTURE_SUPPORT_KEYS,
                (self.action_observed, self.state_observed, self.visual_observed),
                strict=True,
            )
        )


def future_source_rows(
    center: int,
    terminal: int,
    *,
    world_horizon: int = 48,
    future_offsets: tuple[int, ...] = tuple(range(4, 49, 4)),
) -> FutureSourceRows:
    """Return safe gather indices AND independent observed-label masks.

    Repeating a last valid index is transport padding only. The masks must be
    propagated through the Teacher, all action/auxiliary losses and metrics.
    No row after terminal is inferred from an absorbing storage convention.
    """
    for name, value in (
        ("center", center),
        ("terminal", terminal),
        ("world_horizon", world_horizon),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"future {name} must be an integer control step")
    if not 0 <= int(center) < int(terminal) or world_horizon < 1:
        raise ValueError("future window requires a real current action before terminal")
    if (
        not future_offsets
        or tuple(sorted(set(future_offsets))) != future_offsets
        or any(
            isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= world_horizon
            for x in future_offsets
        )
    ):
        raise ValueError("future observation offsets must be ordered inside the horizon")
    action = int(center) + np.arange(world_horizon, dtype=np.int64)
    state = action + 1
    visual = int(center) + np.asarray(future_offsets, dtype=np.int64)
    return FutureSourceRows(
        action_indices=np.minimum(action, int(terminal) - 1),
        state_indices=np.minimum(state, int(terminal)),
        visual_indices=np.minimum(visual, int(terminal)),
        action_observed=action < terminal,
        state_observed=state <= terminal,
        visual_observed=visual <= terminal,
    )
