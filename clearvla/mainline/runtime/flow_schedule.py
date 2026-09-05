"""Serializable same-NFE deployment schedules for the mainline sampler.

This module is the narrow bridge between the standalone flow-solver schedule
specification and the production sampling loop.  It deliberately does not use
the standalone integrator: production sampling must not introduce per-node
finite checks that synchronize CUDA with the host.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

from clearvla.action_solvers.flow_solver.spec import ScheduleSpec


def _schema_version(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return value


def _schedule_from_value(value: object, *, role: str) -> ScheduleSpec:
    if isinstance(value, ScheduleSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{role} schedule must be a ScheduleSpec or mapping")
    if "schema_version" in value:
        _schema_version(
            value["schema_version"],
            name=f"{role} schedule schema_version",
        )
    return ScheduleSpec.from_dict(value)


@dataclass(frozen=True)
class DeploymentFlowSchedule:
    """The two explicit five-update Euler grids used around one W rebuild."""

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    SCHEDULE_NAME: ClassVar[str] = "clearvla.mainline.deployment_flow_schedule"
    UPDATES_PER_PASS: ClassVar[int] = 5
    POWER_FIVE_CANDIDATE_ID: ClassVar[str] = "Q5"

    proposal: ScheduleSpec
    refined: ScheduleSpec
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _schema_version(
                self.schema_version,
                name="deployment flow schedule schema_version",
            ),
        )
        self.validate()

    @classmethod
    def uniform_five(cls) -> DeploymentFlowSchedule:
        """Return the exact legacy ``0,.2,.4,.6,.8,1`` two-pass grid."""

        schedule = ScheduleSpec.uniform(cls.UPDATES_PER_PASS)
        return cls(proposal=schedule, refined=schedule)

    @classmethod
    def custom(
        cls,
        proposal_boundaries: Sequence[float],
        refined_boundaries: Sequence[float] | None = None,
        *,
        label: str = "custom-five",
    ) -> DeploymentFlowSchedule:
        """Build an explicitly replayable pair of custom five-update grids."""

        proposal = ScheduleSpec.custom(
            proposal_boundaries,
            label=f"{label}-proposal",
        )
        refined = ScheduleSpec.custom(
            proposal_boundaries if refined_boundaries is None else refined_boundaries,
            label=f"{label}-refined",
        )
        return cls(proposal=proposal, refined=refined)

    @classmethod
    def same_nfe_power_five(
        cls,
        exponent: float = 1.25,
    ) -> DeploymentFlowSchedule:
        """Return the stable Q5/Q5 same-NFE power-grid candidate.

        Q5 uses ``t_i = (i / 5) ** exponent`` for both deployment passes.
        It changes only inference integration time and remains opt-in.
        """

        exponent = float(exponent)
        if not math.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("Q5 exponent must be positive and finite")
        boundaries = tuple(
            (index / float(cls.UPDATES_PER_PASS)) ** exponent
            for index in range(cls.UPDATES_PER_PASS)
        ) + (1.0,)
        schedule = ScheduleSpec.custom(
            boundaries,
            label="same-nfe-power-five",
            parameters={
                "candidate_id": cls.POWER_FIVE_CANDIDATE_ID,
                "exponent": exponent,
                "formula": "t_i=(i/5)^exponent",
                "steps": cls.UPDATES_PER_PASS,
            },
        )
        return cls(proposal=schedule, refined=schedule)

    @staticmethod
    def _pass_candidate_id(schedule: ScheduleSpec) -> str:
        parameters = dict(schedule.parameters)
        explicit = parameters.get("candidate_id")
        if explicit is not None:
            return str(explicit)
        if schedule.boundaries == ScheduleSpec.uniform(5).boundaries:
            return "E5"
        return "custom5"

    @property
    def candidate_id(self) -> str:
        """Return a stable proposal/refined candidate label for reports."""

        return (
            f"{self._pass_candidate_id(self.proposal)}/"
            f"{self._pass_candidate_id(self.refined)}"
        )

    @property
    def physical_nfe(self) -> int:
        return self.proposal.interval_count + self.refined.interval_count

    @property
    def endpoint_head_calls(self) -> int:
        return 2

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def identity(self) -> dict[str, Any]:
        """Return JSON-safe schedule identity for deployment/validation output."""

        return {
            **self.to_dict(),
            "candidate_id": self.candidate_id,
            "fingerprint": self.fingerprint,
            "physical_nfe": self.physical_nfe,
            "endpoint_head_calls": self.endpoint_head_calls,
            "world_rebuilds": 1,
        }

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported deployment flow schedule schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        for role, schedule in (
            ("proposal", self.proposal),
            ("refined", self.refined),
        ):
            if not isinstance(schedule, ScheduleSpec):
                raise TypeError(f"{role} schedule must be a ScheduleSpec")
            schedule.validate()
            if schedule.interval_count != self.UPDATES_PER_PASS:
                raise ValueError(
                    f"{role} schedule must contain exactly "
                    f"{self.UPDATES_PER_PASS} physical Euler updates"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_flow_schedule": self.SCHEDULE_NAME,
            "schema_version": self.schema_version,
            "method": "euler",
            "endpoint_policy": "external_t1_head",
            "initial_state_policy": "reuse_exact_initial",
            "cache_boundary_policy": "one_w_rebuild_no_history",
            "proposal": self.proposal.to_dict(),
            "refined": self.refined.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DeploymentFlowSchedule:
        if not isinstance(value, Mapping):
            raise TypeError("deployment flow schedule must be a mapping")
        data = dict(value)
        allowed = {
            "deployment_flow_schedule",
            "schema_version",
            "method",
            "endpoint_policy",
            "initial_state_policy",
            "cache_boundary_policy",
            "proposal",
            "refined",
        }
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(
                f"unknown deployment flow schedule fields: {sorted(unknown)}"
            )
        missing = {"proposal", "refined"}.difference(data)
        if missing:
            raise ValueError(
                f"deployment flow schedule is missing fields: {sorted(missing)}"
            )
        name = data.pop("deployment_flow_schedule", cls.SCHEDULE_NAME)
        if name != cls.SCHEDULE_NAME:
            raise ValueError(f"not a {cls.SCHEDULE_NAME} specification")
        expected_constants = {
            "method": "euler",
            "endpoint_policy": "external_t1_head",
            "initial_state_policy": "reuse_exact_initial",
            "cache_boundary_policy": "one_w_rebuild_no_history",
        }
        for field, expected in expected_constants.items():
            actual = data.pop(field, expected)
            if actual != expected:
                raise ValueError(
                    f"deployment flow schedule {field} must be {expected!r}"
                )
        proposal = _schedule_from_value(
            data.pop("proposal"),
            role="proposal",
        )
        refined = _schedule_from_value(
            data.pop("refined"),
            role="refined",
        )
        schema_version = _schema_version(
            data.pop("schema_version", cls.CURRENT_SCHEMA_VERSION),
            name="deployment flow schedule schema_version",
        )
        return cls(
            proposal=proposal,
            refined=refined,
            schema_version=schema_version,
        )


def resolve_deployment_flow_schedule(
    value: DeploymentFlowSchedule | Mapping[str, Any] | None,
) -> DeploymentFlowSchedule:
    """Resolve an optional serialized schedule without changing old defaults."""

    if value is None:
        return DeploymentFlowSchedule.uniform_five()
    if isinstance(value, DeploymentFlowSchedule):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return DeploymentFlowSchedule.from_dict(value)
    raise TypeError(
        "flow_schedule must be DeploymentFlowSchedule, a serialized mapping, or None"
    )


def legacy_uniform_runtime(
    schedule: ScheduleSpec,
    *,
    steps: int,
) -> bool:
    """Whether a schedule can use the exact historical scalar/time expression."""

    return schedule.boundaries == ScheduleSpec.uniform(steps).boundaries


__all__ = [
    "DeploymentFlowSchedule",
    "legacy_uniform_runtime",
    "resolve_deployment_flow_schedule",
]
