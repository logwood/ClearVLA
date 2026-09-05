"""Versioned schedules and solver-selection contracts.

This module only describes a numerical integration pass.  It deliberately
does not know about ClearVLA models, action codecs, W, outlets or checkpoints.
That separation makes a schedule reusable at the single-pass boundary while
keeping production integration an explicit caller decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping, Sequence

SolverMethod = Literal["euler", "heun", "rk4"]
ScheduleKind = Literal[
    "uniform",
    "dense_jump",
    "late_split",
    "late_warp",
    "cosine",
    "custom",
]
PassRole = Literal["single", "proposal", "refined"]
EndpointPolicy = Literal["external_t1_head"]
CachePolicy = Literal["one_fixed_cache"]
InitialStatePolicy = Literal["reuse_exact_initial"]
CacheBoundaryPolicy = Literal["fresh_cache_reset"]
ParameterValue = str | int | float | bool


def _parameter_items(
    value: Mapping[str, ParameterValue]
    | Sequence[tuple[str, ParameterValue]],
) -> tuple[tuple[str, ParameterValue], ...]:
    """Normalize metadata into a deterministic, hashable representation."""

    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized: list[tuple[str, ParameterValue]] = []
    for key, item in items:
        key = str(key)
        if not key.strip():
            raise ValueError("schedule parameter keys must be non-empty")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"schedule parameter {key!r} must be finite")
        if not isinstance(item, (str, int, float, bool)):
            raise TypeError(
                f"schedule parameter {key!r} must be a scalar JSON value"
            )
        normalized.append((key, item))
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("schedule parameter keys must be unique")
    return tuple(sorted(normalized))


def _parameter_dict(
    value: Sequence[tuple[str, ParameterValue]],
) -> dict[str, ParameterValue]:
    return {key: item for key, item in value}


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float32(value: float) -> float:
    """Round one value exactly as a runtime IEEE-754 float32 would."""

    return struct.unpack("!f", struct.pack("!f", value))[0]


@dataclass(frozen=True)
class ScheduleSpec:
    """Immutable integration boundaries for one pass.

    ``boundaries`` includes both endpoints.  The velocity field is queried at
    every left endpoint except the final ``1.0`` boundary; the caller may use a
    separate endpoint head there.  Thus a schedule with ``N`` intervals has
    exactly ``N`` Euler velocity evaluations.
    """

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    SCHEDULE_NAME: ClassVar[str] = "clearvla.flow_schedule"

    boundaries: tuple[float, ...]
    kind: ScheduleKind = "custom"
    parameters: tuple[tuple[str, ParameterValue], ...] = ()
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "boundaries",
            tuple(float(value) for value in self.boundaries),
        )
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "parameters", _parameter_items(self.parameters))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        self.validate()

    @classmethod
    def uniform(cls, steps: int) -> ScheduleSpec:
        """Build ``steps`` equal intervals on ``[0, 1]``."""

        steps = int(steps)
        if steps < 1:
            raise ValueError("uniform schedule steps must be positive")
        boundaries = tuple(index / float(steps) for index in range(steps)) + (1.0,)
        return cls(
            boundaries=boundaries,
            kind="uniform",
            parameters=(("steps", steps),),
        )

    @classmethod
    def dense_jump(cls, steps: int, t_jump: float = 0.5) -> ScheduleSpec:
        """Build the inference-only Dense-Jump schedule.

        For ``steps > 1`` the first ``steps - 1`` intervals uniformly cover
        ``[0, t_jump]`` and the last interval jumps directly to ``1``.  A
        one-step schedule is the unambiguous ``0 -> 1`` jump regardless of the
        metadata value of ``t_jump``.
        """

        steps = int(steps)
        t_jump = float(t_jump)
        if steps < 1:
            raise ValueError("Dense-Jump steps must be positive")
        if not math.isfinite(t_jump) or not 0.0 < t_jump < 1.0:
            raise ValueError("Dense-Jump t_jump must lie strictly inside (0, 1)")
        if steps == 1:
            boundaries = (0.0, 1.0)
        else:
            prefix = tuple(
                (index * t_jump) / float(steps - 1)
                for index in range(steps)
            )
            boundaries = prefix + (1.0,)
        return cls(
            boundaries=boundaries,
            kind="dense_jump",
            parameters=(("steps", steps), ("t_jump", t_jump)),
        )

    @classmethod
    def late_split(
        cls,
        base_steps: int = 5,
        *,
        split_fraction: float = 0.5,
        interval_index: int | None = None,
    ) -> ScheduleSpec:
        """Split one interval of a uniform grid, defaulting to the last."""

        base_steps = int(base_steps)
        split_fraction = float(split_fraction)
        if base_steps < 1:
            raise ValueError("late-split base_steps must be positive")
        if not math.isfinite(split_fraction) or not 0.0 < split_fraction < 1.0:
            raise ValueError("split_fraction must lie strictly inside (0, 1)")
        index = base_steps - 1 if interval_index is None else int(interval_index)
        if index < 0:
            index += base_steps
        if not 0 <= index < base_steps:
            raise ValueError("interval_index is outside the base schedule")
        base = list(cls.uniform(base_steps).boundaries)
        left, right = base[index], base[index + 1]
        split = left + split_fraction * (right - left)
        boundaries = tuple(base[: index + 1] + [split] + base[index + 1 :])
        return cls(
            boundaries=boundaries,
            kind="late_split",
            parameters=(
                ("base_steps", base_steps),
                ("interval_index", index),
                ("split_fraction", split_fraction),
            ),
        )

    @classmethod
    def late_warp(cls, steps: int = 6, exponent: float = 1.25) -> ScheduleSpec:
        """Build a smooth late-biased grid without a single abrupt split."""

        steps = int(steps)
        exponent = float(exponent)
        if steps < 1:
            raise ValueError("late-warp steps must be positive")
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("late-warp exponent must be positive and finite")
        boundaries = tuple(
            1.0 - (1.0 - index / float(steps)) ** exponent
            for index in range(steps)
        ) + (1.0,)
        return cls(
            boundaries=boundaries,
            kind="late_warp",
            parameters=(("steps", steps), ("exponent", exponent)),
        )

    @classmethod
    def cosine(cls, steps: int = 6) -> ScheduleSpec:
        """Build a diagnostic grid clustered at both endpoints."""

        steps = int(steps)
        if steps < 1:
            raise ValueError("cosine schedule steps must be positive")
        boundaries = tuple(
            0.5 * (1.0 - math.cos(math.pi * index / float(steps)))
            for index in range(steps)
        ) + (1.0,)
        return cls(
            boundaries=boundaries,
            kind="cosine",
            parameters=(("steps", steps),),
        )

    @classmethod
    def custom(
        cls,
        boundaries: Sequence[float],
        *,
        label: str = "custom",
        parameters: Mapping[str, ParameterValue]
        | Sequence[tuple[str, ParameterValue]] = (),
    ) -> ScheduleSpec:
        """Create an explicitly named schedule from caller-owned boundaries."""

        label = str(label)
        normalized_parameters = list(_parameter_items(parameters))
        if label != "custom":
            # Custom schedules may use a descriptive label, but the serialized
            # kind remains ``custom`` so it cannot masquerade as a constructor.
            if any(key == "label" for key, _ in normalized_parameters):
                raise ValueError("custom schedule label is reserved in parameters")
            normalized_parameters.append(("label", label))
        return cls(
            boundaries=tuple(boundaries),
            kind="custom",
            parameters=tuple(normalized_parameters),
        )

    @property
    def interval_count(self) -> int:
        return len(self.boundaries) - 1

    @property
    def query_times(self) -> tuple[float, ...]:
        return self.boundaries[:-1]

    @property
    def step_sizes(self) -> tuple[float, ...]:
        return tuple(
            right - left
            for left, right in zip(self.boundaries, self.boundaries[1:])
        )

    @property
    def runtime_boundaries(self) -> tuple[float, ...]:
        """Boundaries after the float32 conversion used by time conditioning."""

        return tuple(_float32(value) for value in self.boundaries)

    @property
    def runtime_step_sizes(self) -> tuple[float, ...]:
        return tuple(
            right - left
            for left, right in zip(self.runtime_boundaries, self.runtime_boundaries[1:])
        )

    @property
    def max_step(self) -> float:
        return max(self.step_sizes)

    @property
    def min_step(self) -> float:
        return min(self.step_sizes)

    @property
    def is_uniform(self) -> bool:
        first = self.step_sizes[0]
        return all(math.isclose(step, first, rel_tol=1e-12, abs_tol=1e-12) for step in self.step_sizes)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported flow schedule schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if self.kind not in {
            "uniform",
            "dense_jump",
            "late_split",
            "late_warp",
            "cosine",
            "custom",
        }:
            raise ValueError(f"unsupported schedule kind: {self.kind!r}")
        if len(self.boundaries) < 2:
            raise ValueError("a schedule needs at least two boundaries")
        if not all(math.isfinite(value) for value in self.boundaries):
            raise ValueError("schedule boundaries must be finite")
        if self.boundaries[0] != 0.0 or self.boundaries[-1] != 1.0:
            raise ValueError("schedule boundaries must start at 0.0 and end at 1.0")
        if any(
            right <= left
            for left, right in zip(self.boundaries, self.boundaries[1:])
        ):
            raise ValueError("schedule boundaries must be strictly increasing")
        runtime_boundaries = self.runtime_boundaries
        if runtime_boundaries[0] != 0.0 or runtime_boundaries[-1] != 1.0:
            raise ValueError("schedule endpoints must survive float32 conversion")
        if any(
            right <= left
            for left, right in zip(runtime_boundaries, runtime_boundaries[1:])
        ):
            raise ValueError(
                "schedule boundaries collapse or reverse after float32 conversion"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule": self.SCHEDULE_NAME,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "boundaries": list(self.boundaries),
            "parameters": _parameter_dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScheduleSpec:
        data = dict(value)
        allowed = {"schedule", "schema_version", "kind", "boundaries", "parameters"}
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"unknown flow schedule fields: {sorted(unknown)}")
        name = data.pop("schedule", cls.SCHEDULE_NAME)
        if name != cls.SCHEDULE_NAME:
            raise ValueError(f"not a {cls.SCHEDULE_NAME} specification")
        return cls(
            boundaries=tuple(data["boundaries"]),
            kind=data.get("kind", "custom"),
            parameters=data.get("parameters", {}),
            schema_version=data.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class SolverSpec:
    """One numerical method applied inside one fixed cache."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    SOLVER_NAME: ClassVar[str] = "clearvla.flow_solver"

    schedule: ScheduleSpec
    method: SolverMethod = "euler"
    pass_role: PassRole = "single"
    endpoint_policy: EndpointPolicy = "external_t1_head"
    cache_policy: CachePolicy = "one_fixed_cache"
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "pass_role", str(self.pass_role))
        object.__setattr__(self, "endpoint_policy", str(self.endpoint_policy))
        object.__setattr__(self, "cache_policy", str(self.cache_policy))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        self.validate()

    @property
    def intervals(self) -> int:
        return self.schedule.interval_count

    @property
    def nfe_per_pass(self) -> int:
        multiplier = {"euler": 1, "heun": 2, "rk4": 4}[self.method]
        return self.intervals * multiplier

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported flow solver schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if self.method not in {"euler", "heun", "rk4"}:
            raise ValueError(f"unsupported solver method: {self.method!r}")
        if self.pass_role not in {"single", "proposal", "refined"}:
            raise ValueError(f"unsupported pass role: {self.pass_role!r}")
        if self.endpoint_policy != "external_t1_head":
            raise ValueError("the endpoint head must remain an external t=1 readout")
        if self.cache_policy != "one_fixed_cache":
            raise ValueError("one solver pass must use exactly one fixed cache")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.SOLVER_NAME,
            "schema_version": self.schema_version,
            "method": self.method,
            "pass_role": self.pass_role,
            "endpoint_policy": self.endpoint_policy,
            "cache_policy": self.cache_policy,
            "schedule": self.schedule.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SolverSpec:
        data = dict(value)
        allowed = {
            "solver",
            "schema_version",
            "method",
            "pass_role",
            "endpoint_policy",
            "cache_policy",
            "schedule",
        }
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"unknown flow solver fields: {sorted(unknown)}")
        name = data.pop("solver", cls.SOLVER_NAME)
        if name != cls.SOLVER_NAME:
            raise ValueError(f"not a {cls.SOLVER_NAME} specification")
        schedule_data = data["schedule"]
        schedule = (
            schedule_data
            if isinstance(schedule_data, ScheduleSpec)
            else ScheduleSpec.from_dict(schedule_data)
        )
        return cls(
            schedule=schedule,
            method=data.get("method", "euler"),
            pass_role=data.get("pass_role", "single"),
            endpoint_policy=data.get("endpoint_policy", "external_t1_head"),
            cache_policy=data.get("cache_policy", "one_fixed_cache"),
            schema_version=data.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class TwoPassSpec:
    """Explicit proposal → W rebuild → refined solver contract."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    TWO_PASS_NAME: ClassVar[str] = "clearvla.two_pass_flow_solver"

    proposal: SolverSpec
    refined: SolverSpec
    initial_state_policy: InitialStatePolicy = "reuse_exact_initial"
    cache_boundary_policy: CacheBoundaryPolicy = "fresh_cache_reset"
    endpoint_head_calls_per_pass: int = 1
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_state_policy", str(self.initial_state_policy))
        object.__setattr__(self, "cache_boundary_policy", str(self.cache_boundary_policy))
        object.__setattr__(
            self,
            "endpoint_head_calls_per_pass",
            int(self.endpoint_head_calls_per_pass),
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        self.validate()

    @property
    def physical_nfe(self) -> int:
        return self.proposal.nfe_per_pass + self.refined.nfe_per_pass

    @property
    def endpoint_head_calls(self) -> int:
        return 2 * self.endpoint_head_calls_per_pass

    @property
    def total_dynamic_calls(self) -> int:
        return self.physical_nfe + self.endpoint_head_calls

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported two-pass solver schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if self.proposal.pass_role != "proposal":
            raise ValueError("proposal solver must be marked pass_role='proposal'")
        if self.refined.pass_role != "refined":
            raise ValueError("refined solver must be marked pass_role='refined'")
        if self.initial_state_policy != "reuse_exact_initial":
            raise ValueError("the current contract requires the exact initial state restart")
        if self.cache_boundary_policy != "fresh_cache_reset":
            raise ValueError("solver history must reset at the W rebuild boundary")
        if self.endpoint_head_calls_per_pass != 1:
            raise ValueError("the current two-pass contract has one endpoint head per pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "two_pass_solver": self.TWO_PASS_NAME,
            "schema_version": self.schema_version,
            "proposal": self.proposal.to_dict(),
            "refined": self.refined.to_dict(),
            "initial_state_policy": self.initial_state_policy,
            "cache_boundary_policy": self.cache_boundary_policy,
            "endpoint_head_calls_per_pass": self.endpoint_head_calls_per_pass,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TwoPassSpec:
        data = dict(value)
        allowed = {
            "two_pass_solver",
            "schema_version",
            "proposal",
            "refined",
            "initial_state_policy",
            "cache_boundary_policy",
            "endpoint_head_calls_per_pass",
        }
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"unknown two-pass solver fields: {sorted(unknown)}")
        name = data.pop("two_pass_solver", cls.TWO_PASS_NAME)
        if name != cls.TWO_PASS_NAME:
            raise ValueError(f"not a {cls.TWO_PASS_NAME} specification")
        proposal_data = data["proposal"]
        refined_data = data["refined"]
        proposal = (
            proposal_data
            if isinstance(proposal_data, SolverSpec)
            else SolverSpec.from_dict(proposal_data)
        )
        refined = (
            refined_data
            if isinstance(refined_data, SolverSpec)
            else SolverSpec.from_dict(refined_data)
        )
        return cls(
            proposal=proposal,
            refined=refined,
            initial_state_policy=data.get("initial_state_policy", "reuse_exact_initial"),
            cache_boundary_policy=data.get("cache_boundary_policy", "fresh_cache_reset"),
            endpoint_head_calls_per_pass=data.get("endpoint_head_calls_per_pass", 1),
            schema_version=data.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
        )
