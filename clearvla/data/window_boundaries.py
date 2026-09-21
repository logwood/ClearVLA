"""Explicit policy-window boundary contracts shared by data and sampling.

The model always consumes the same causal histories and fixed 24/48-row
horizons.  This module owns only which episode centers are admitted and how
those centers are named for sampling and audit.  Keeping that decision outside
the benchmark converter prevents an old HDF5 lower bound from silently
discarding reset-prefix supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


STRICT_COMPLETE_V1 = "strict_complete_v1"
CAUSAL_PREFIX_V1 = "causal_prefix_v1"
CAUSAL_PREFIX_TERMINAL_SUFFIX_V2 = "causal_prefix_terminal_suffix_v2"

STRICT_REGION = "strict"
PREFIX_REGION = "prefix"
TAIL_REGION = "tail"

WINDOW_BOUNDARY_CONTRACTS = (
    STRICT_COMPLETE_V1,
    CAUSAL_PREFIX_V1,
    CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
)
BOUNDARY_REGIONS = (STRICT_REGION, PREFIX_REGION, TAIL_REGION)
BOUNDARY_REGION_TO_INDEX = {
    name: index for index, name in enumerate(BOUNDARY_REGIONS)
}


@dataclass(frozen=True)
class WindowBoundaryMetadata:
    """Primitive episode metadata needed to resolve legal centers."""

    length: int
    valid_center_start: int | None = None
    valid_center_end: int | None = None
    strict_valid_center_start: int | None = None
    strict_valid_center_end: int | None = None
    terminal_state_index: int | None = None
    source_action_count: int | None = None
    terminal_padding_mode: str = ""

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError("window boundary episode length must be positive")
        for left_name, right_name in (
            ("valid_center_start", "valid_center_end"),
            ("strict_valid_center_start", "strict_valid_center_end"),
        ):
            left = getattr(self, left_name)
            right = getattr(self, right_name)
            if (left is None) != (right is None):
                raise ValueError(f"{left_name} and {right_name} must be declared together")
            if left is not None and right is not None and not (
                0 <= int(left) <= int(right) < self.length
            ):
                raise ValueError(
                    f"invalid {left_name}/{right_name} bounds [{left},{right}] "
                    f"for length={self.length}"
                )
        if self.source_action_count is not None and not (
            1 <= int(self.source_action_count) <= self.length
        ):
            raise ValueError("source_action_count must lie inside the episode")
        if self.terminal_state_index is not None and not (
            1 <= int(self.terminal_state_index) < self.length
        ):
            raise ValueError("terminal_state_index must lie inside the episode")


@dataclass(frozen=True)
class WindowBoundaryPlan:
    """Resolved inclusive center ranges keyed by disjoint semantic region."""

    contract: str
    regions: tuple[tuple[str, int, int], ...]
    source_action_count: int

    def validate(self) -> None:
        if self.contract not in WINDOW_BOUNDARY_CONTRACTS:
            raise ValueError(f"unknown window boundary contract {self.contract!r}")
        if self.source_action_count <= 0:
            raise ValueError("source_action_count must be positive")
        previous_end = -1
        names: list[str] = []
        for name, start, end in self.regions:
            if name not in BOUNDARY_REGIONS or name in names:
                raise ValueError("window boundary regions must be known and unique")
            if not 0 <= int(start) <= int(end):
                raise ValueError(f"invalid {name} region [{start},{end}]")
            if int(start) <= previous_end:
                raise ValueError("window boundary regions must be disjoint and ordered")
            previous_end = int(end)
            names.append(name)
        if not self.regions:
            raise ValueError("window boundary plan has no legal centers")

    def iter_centers(
        self,
        *,
        stride: int = 1,
        allowed_regions: Iterable[str] | None = None,
    ) -> Iterable[tuple[int, str]]:
        if stride <= 0:
            raise ValueError("window-center stride must be positive")
        allowed = None if allowed_regions is None else tuple(str(x) for x in allowed_regions)
        if allowed is not None:
            if not allowed or len(set(allowed)) != len(allowed):
                raise ValueError("allowed boundary regions must be non-empty and unique")
            unknown = sorted(set(allowed) - set(BOUNDARY_REGIONS))
            if unknown:
                raise ValueError(f"unknown allowed boundary regions: {unknown}")
        for name, start, end in self.regions:
            if allowed is None or name in allowed:
                for center in range(int(start), int(end) + 1, int(stride)):
                    yield center, name

    @property
    def region_counts(self) -> dict[str, int]:
        return {
            name: int(end) - int(start) + 1 for name, start, end in self.regions
        }


def _declared_strict_bounds(
    metadata: WindowBoundaryMetadata,
    *,
    computed_start: int,
    computed_end: int,
) -> tuple[int, int]:
    if metadata.strict_valid_center_start is not None:
        assert metadata.strict_valid_center_end is not None
        start = int(metadata.strict_valid_center_start)
        end = int(metadata.strict_valid_center_end)
    elif metadata.valid_center_start is not None:
        assert metadata.valid_center_end is not None
        start = int(metadata.valid_center_start)
        end = int(metadata.valid_center_end)
    else:
        start, end = int(computed_start), int(computed_end)
    start = max(start, int(computed_start))
    end = min(end, int(computed_end))
    if start > end:
        raise ValueError(
            "episode has no complete strict window after declared/computed bounds "
            f"intersection [{start},{end}]"
        )
    return start, end


def resolve_window_boundary_plan(
    metadata: WindowBoundaryMetadata,
    *,
    contract: str,
    strict_computed_start: int,
    complete_future_end: int,
    expected_terminal_padding_mode: str | None = None,
) -> WindowBoundaryPlan:
    """Resolve one fail-closed boundary plan.

    ``strict_computed_start`` is the first center with complete causal history;
    ``complete_future_end`` is the final center whose +48 Teacher support lies
    in the materialized episode.  Both values are calculated by the dataset
    from its typed offsets, while this function reconciles them with explicit
    converter metadata.
    """

    metadata.validate()
    selected = str(contract)
    if selected not in WINDOW_BOUNDARY_CONTRACTS:
        raise ValueError(f"unknown window boundary contract {selected!r}")
    strict_start, strict_end = _declared_strict_bounds(
        metadata,
        computed_start=int(strict_computed_start),
        computed_end=int(complete_future_end),
    )
    source_count = (
        int(metadata.source_action_count)
        if metadata.source_action_count is not None
        else int(metadata.terminal_state_index)
        if metadata.terminal_state_index is not None
        else int(metadata.length)
    )

    if selected == STRICT_COMPLETE_V1:
        plan = WindowBoundaryPlan(
            contract=selected,
            regions=((STRICT_REGION, strict_start, strict_end),),
            source_action_count=source_count,
        )
        plan.validate()
        return plan

    if selected == CAUSAL_PREFIX_V1:
        if metadata.terminal_state_index is not None or metadata.source_action_count is not None:
            raise ValueError(
                "causal_prefix_v1 accepts the original LIBERO trajectory, not a terminal suffix"
            )
        if strict_start <= 0:
            raise ValueError("causal_prefix_v1 requires a non-empty missing-history prefix")
        if strict_end != int(complete_future_end):
            raise ValueError(
                "causal_prefix_v1 requires the declared upper bound to retain every "
                "complete-future center"
            )
        plan = WindowBoundaryPlan(
            contract=selected,
            regions=(
                (PREFIX_REGION, 0, strict_start - 1),
                (STRICT_REGION, strict_start, strict_end),
            ),
            source_action_count=metadata.length,
        )
        plan.validate()
        return plan

    if metadata.terminal_state_index is None or metadata.source_action_count is None:
        raise ValueError(
            "causal_prefix_terminal_suffix_v2 requires terminal_state_index and "
            "source_action_count"
        )
    terminal = int(metadata.terminal_state_index)
    if terminal != source_count:
        raise ValueError("terminal_state_index must equal source_action_count for LIBERO v2")
    if metadata.length != source_count + 48:
        raise ValueError("LIBERO terminal suffix must contain exactly 48 materialized rows")
    if expected_terminal_padding_mode is not None and (
        metadata.terminal_padding_mode != str(expected_terminal_padding_mode)
    ):
        raise ValueError("LIBERO terminal padding mode differs from the requested contract")
    if metadata.valid_center_start != 0 or metadata.valid_center_end != source_count - 1:
        raise ValueError("LIBERO v2 valid centers must cover 0 through the last real action")
    if strict_start != int(strict_computed_start):
        raise ValueError("LIBERO v2 strict prefix bound differs from the typed history")
    expected_strict_end = source_count - 49
    if strict_end != expected_strict_end:
        raise ValueError("LIBERO v2 strict upper bound must equal source_action_count - 49")
    if complete_future_end != source_count - 1:
        raise ValueError("LIBERO v2 suffix does not close the final real action's +48 support")
    tail_start = strict_end + 1
    if tail_start != source_count - 48:
        raise AssertionError("LIBERO v2 tail region no longer owns exactly 48 centers")
    plan = WindowBoundaryPlan(
        contract=selected,
        regions=(
            (PREFIX_REGION, 0, strict_start - 1),
            (STRICT_REGION, strict_start, strict_end),
            (TAIL_REGION, tail_start, source_count - 1),
        ),
        source_action_count=source_count,
    )
    plan.validate()
    return plan


def policy_action_union_coverage(
    centers_by_episode: Mapping[int, Iterable[int]],
    *,
    source_action_counts: Mapping[int, int],
    policy_horizon: int,
    action_offset: int = 0,
) -> dict[str, int | float]:
    """Count unique real source actions touched by the admitted policy targets."""

    if policy_horizon <= 0:
        raise ValueError("policy_horizon must be positive")
    if set(centers_by_episode) != set(source_action_counts):
        raise ValueError("coverage centers and source-action counts identify different episodes")
    total = 0
    covered = 0
    for episode, source_count_value in source_action_counts.items():
        source_count = int(source_count_value)
        if source_count <= 0:
            raise ValueError("source-action counts must be positive")
        rows: set[int] = set()
        for center_value in centers_by_episode[episode]:
            start = int(center_value) + int(action_offset)
            rows.update(
                row
                for row in range(start, start + int(policy_horizon))
                if 0 <= row < source_count
            )
        total += source_count
        covered += len(rows)
    uncovered = total - covered
    return {
        "source_real_action_count": int(total),
        "supervised_real_action_count": int(covered),
        "uncovered_real_action_count": int(uncovered),
        "policy_action_union_coverage_fraction": float(covered / max(total, 1)),
    }


__all__ = [
    "BOUNDARY_REGIONS",
    "BOUNDARY_REGION_TO_INDEX",
    "CAUSAL_PREFIX_TERMINAL_SUFFIX_V2",
    "CAUSAL_PREFIX_V1",
    "PREFIX_REGION",
    "STRICT_COMPLETE_V1",
    "STRICT_REGION",
    "TAIL_REGION",
    "WINDOW_BOUNDARY_CONTRACTS",
    "WindowBoundaryMetadata",
    "WindowBoundaryPlan",
    "policy_action_union_coverage",
    "resolve_window_boundary_plan",
]
