"""Reproducible fixed-checkpoint candidate-panel orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from torch import Tensor

from .diagnostics import CandidateSpec, candidate_matrix
from .integrate import TwoPassResult, run_two_pass
from .protocols import Cache, CacheRebuilder, EndpointHead, TimeFactory, VelocityField

CacheFactory = Callable[[], Cache]


def _json_safe(value: Any, *, path: str = "value", active: set[int] | None = None) -> Any:
    """Normalize a caller-owned value to finite, deterministic JSON data.

    Replay metadata is intentionally kept outside the numerical solver, but a
    replay fingerprint is only meaningful when its inputs have an unambiguous
    serialization.  In particular, tensors and NumPy scalar objects must be
    converted by the experiment owner before they reach this boundary.
    """

    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{path} contains a cyclic mapping")
        active.add(marker)
        try:
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(f"{path} mapping keys must be non-empty strings")
                normalized[key] = _json_safe(item, path=f"{path}.{key}", active=active)
            return {key: normalized[key] for key in sorted(normalized)}
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{path} contains a cyclic sequence")
        active.add(marker)
        try:
            return [
                _json_safe(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            ]
        finally:
            active.remove(marker)
    raise TypeError(
        f"{path} contains {type(value).__qualname__}; expected only JSON scalars, "
        "lists/tuples, and string-keyed mappings"
    )


def _metadata_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# The profile is deliberately strict, but it is applied only when an outer
# owner asks for ``validate_for_u0``.  A generic attachment remains useful for
# local diagnostics and can contain a smaller subset of fields.
U0_REQUIRED_IDENTITY_KEYS = frozenset(
    {
        "code_revision",
        "checkpoint_hash",
        "manifest_schema",
        "config_hash",
        "outlet",
        "action_chart",
        "task_id",
        "sample_id",
        "frame_id",
        "seed",
        "observation_hash",
        "history_hash",
        "language_hash",
        "initial_noise_hash",
        "dtype",
        "device",
        "execution_mode",
    }
)
U0_REQUIRED_SCOPE_KEYS = frozenset(
    {
        "proposal_cache_fingerprint",
        "proposal_world_condition_hash",
        "refined_cache_fingerprint",
        "rebuild_event_id",
        "initial_state_hash",
        "initial_state_equal",
    }
)
U0_REQUIRED_ACCOUNTING_KEYS = frozenset(
    {
        "proposal_physical_nfe",
        "refined_physical_nfe",
        "proposal_endpoint_calls",
        "refined_endpoint_calls",
        "diagnostic_nfe",
        "proposal_wall_time_ms",
        "w_rebuild_wall_time_ms",
        "refined_wall_time_ms",
        "total_wall_time_ms",
        "peak_memory_bytes",
        "batch_size",
        "total_dynamic_calls",
    }
)
_U0_INTEGER_ACCOUNTING_KEYS = frozenset(
    {
        "proposal_physical_nfe",
        "refined_physical_nfe",
        "proposal_endpoint_calls",
        "refined_endpoint_calls",
        "diagnostic_nfe",
        "peak_memory_bytes",
        "batch_size",
        "total_dynamic_calls",
    }
)
_U0_NONNEGATIVE_ACCOUNTING_KEYS = frozenset(
    {
        *_U0_INTEGER_ACCOUNTING_KEYS,
        "proposal_wall_time_ms",
        "w_rebuild_wall_time_ms",
        "refined_wall_time_ms",
        "total_wall_time_ms",
    }
)


def _freeze_section(
    value: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized = tuple((str(key), item) for key, item in items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("replay attachment keys must be unique")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ReplayAttachment:
    """Caller-owned identity and metric fields attached to one replay row.

    Values are intentionally opaque.  The outer experiment owner is
    responsible for converting tensors/arrays to JSON-safe scalars before
    writing JSONL; the solver package never interprets task metrics.
    """

    identity: tuple[tuple[str, Any], ...] = ()
    scope: tuple[tuple[str, Any], ...] = ()
    accounting: tuple[tuple[str, Any], ...] = ()
    numerical: tuple[tuple[str, Any], ...] = ()
    outer_effect: tuple[tuple[str, Any], ...] = ()
    behavior: tuple[tuple[str, Any], ...] = ()
    schema_version: int = 1

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    ATTACHMENT_NAME: ClassVar[str] = "clearvla.flow_replay_attachment"

    def __post_init__(self) -> None:
        for name in (
            "identity",
            "scope",
            "accounting",
            "numerical",
            "outer_effect",
            "behavior",
        ):
            object.__setattr__(self, name, _freeze_section(getattr(self, name)))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported replay attachment schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )

    @classmethod
    def from_sections(
        cls,
        *,
        identity: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        scope: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        accounting: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        numerical: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        outer_effect: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        behavior: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
        schema_version: int = 1,
    ) -> ReplayAttachment:
        return cls(
            _freeze_section(identity),
            _freeze_section(scope),
            _freeze_section(accounting),
            _freeze_section(numerical),
            _freeze_section(outer_effect),
            _freeze_section(behavior),
            schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the backwards-compatible six-section view."""

        return {
            name: dict(getattr(self, name))
            for name in (
                "identity",
                "scope",
                "accounting",
                "numerical",
                "outer_effect",
                "behavior",
            )
        }

    def as_state_dict(self) -> dict[str, Any]:
        """Return the versioned payload used for deterministic fingerprints."""

        return {
            "attachment": self.ATTACHMENT_NAME,
            "schema_version": self.schema_version,
            **self.to_dict(),
        }

    def to_state_dict(self) -> dict[str, Any]:
        """Alias for callers that use the usual ``to_*`` serialization name."""

        return self.as_state_dict()

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> ReplayAttachment:
        """Rebuild an attachment while rejecting unknown/versioned fields."""

        if not isinstance(value, Mapping):
            raise TypeError("replay attachment state must be a mapping")
        allowed = {
            "attachment",
            "schema_version",
            "identity",
            "scope",
            "accounting",
            "numerical",
            "outer_effect",
            "behavior",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown replay attachment fields: {sorted(unknown)}")
        name = value.get("attachment", cls.ATTACHMENT_NAME)
        if name != cls.ATTACHMENT_NAME:
            raise ValueError(f"not a {cls.ATTACHMENT_NAME} attachment")
        sections: dict[str, Mapping[str, Any]] = {}
        for section in (
            "identity",
            "scope",
            "accounting",
            "numerical",
            "outer_effect",
            "behavior",
        ):
            section_value = value.get(section, {})
            if not isinstance(section_value, Mapping):
                raise TypeError(f"replay attachment section {section!r} must be a mapping")
            sections[section] = section_value
        return cls.from_sections(
            identity=sections["identity"],
            scope=sections["scope"],
            accounting=sections["accounting"],
            numerical=sections["numerical"],
            outer_effect=sections["outer_effect"],
            behavior=sections["behavior"],
            schema_version=value.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
        )

    def validate_json_safe(self) -> None:
        """Fail closed unless every attached value has finite JSON semantics."""

        _json_safe(self.as_state_dict(), path="replay_attachment")

    @property
    def fingerprint(self) -> str:
        """Hash the versioned metadata after strict JSON normalization."""

        self.validate_json_safe()
        return _metadata_fingerprint(self.as_state_dict())

    def missing_required_keys(
        self,
        section: str,
        required: Sequence[str] | set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        """Return required keys absent from one named section."""

        if section not in {
            "identity",
            "scope",
            "accounting",
            "numerical",
            "outer_effect",
            "behavior",
        }:
            raise ValueError(f"unknown replay attachment section: {section!r}")
        present = set(dict(getattr(self, section)))
        return tuple(sorted(set(required).difference(present)))

    def empty_required_values(
        self,
        section: str,
        required: Sequence[str] | set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        """Return required fields whose string value is blank or ``None``."""

        if section not in {
            "identity",
            "scope",
            "accounting",
            "numerical",
            "outer_effect",
            "behavior",
        }:
            raise ValueError(f"unknown replay attachment section: {section!r}")
        values = dict(getattr(self, section))
        empty = {
            key
            for key in required
            if key in values
            and (
                values[key] is None
                or (isinstance(values[key], str) and not values[key].strip())
            )
        }
        return tuple(sorted(empty))

    def _validate_u0_value_types(self) -> None:
        """Validate exact-state and accounting types used by the U0 profile."""

        scope = dict(self.scope)
        if scope.get("initial_state_equal") is not True:
            raise ValueError("replay attachment U0 scope requires initial_state_equal=True")
        for key in (
            "proposal_cache_fingerprint",
            "proposal_world_condition_hash",
            "refined_cache_fingerprint",
            "rebuild_event_id",
            "initial_state_hash",
        ):
            value = scope.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"replay attachment U0 scope field {key!r} must be a non-empty string")

        accounting = dict(self.accounting)
        for key in _U0_NONNEGATIVE_ACCOUNTING_KEYS:
            value = accounting.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"replay attachment U0 accounting field {key!r} must be numeric")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"replay attachment U0 accounting field {key!r} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"replay attachment U0 accounting field {key!r} must be non-negative")
        for key in _U0_INTEGER_ACCOUNTING_KEYS:
            value = float(accounting[key])
            if not value.is_integer():
                raise ValueError(f"replay attachment U0 accounting field {key!r} must be integral")
        expected_total = (
            float(accounting["proposal_physical_nfe"])
            + float(accounting["refined_physical_nfe"])
            + float(accounting["proposal_endpoint_calls"])
            + float(accounting["refined_endpoint_calls"])
        )
        actual_total = float(accounting["total_dynamic_calls"])
        if actual_total != expected_total:
            raise ValueError(
                "replay attachment U0 accounting total_dynamic_calls does not "
                "match both-pass physical and endpoint calls"
            )

    def validate_for_u0(
        self,
        *,
        required_identity: Sequence[str] | set[str] | frozenset[str] = U0_REQUIRED_IDENTITY_KEYS,
        required_scope: Sequence[str] | set[str] | frozenset[str] = U0_REQUIRED_SCOPE_KEYS,
        required_accounting: Sequence[str] | set[str] | frozenset[str] = U0_REQUIRED_ACCOUNTING_KEYS,
    ) -> str:
        """Validate the minimum reproducibility contract for an interface replay.

        The returned fingerprint is suitable for a JSONL row.  Numerical,
        outer-effect, and behavior sections stay optional here because they
        belong to later U1--U3 evidence gates.
        """

        self.validate_json_safe()
        missing = {
            "identity": self.missing_required_keys("identity", required_identity),
            "scope": self.missing_required_keys("scope", required_scope),
            "accounting": self.missing_required_keys("accounting", required_accounting),
        }
        empty = {
            "identity": self.empty_required_values("identity", required_identity),
            "scope": self.empty_required_values("scope", required_scope),
            "accounting": self.empty_required_values("accounting", required_accounting),
        }
        missing = {section: values for section, values in missing.items() if values}
        empty = {section: values for section, values in empty.items() if values}
        if missing or empty:
            details = "; ".join(
                f"{section}: {', '.join(values)}" for section, values in missing.items()
            )
            if empty:
                empty_details = "; ".join(
                    f"{section} blank: {', '.join(values)}"
                    for section, values in empty.items()
                )
                details = "; ".join(value for value in (details, empty_details) if value)
            raise ValueError(f"replay attachment is not U0-complete ({details})")
        self._validate_u0_value_types()
        return self.fingerprint

    @property
    def u0_ready(self) -> bool:
        """Whether the default U0 profile is complete and serializable."""

        try:
            self.validate_for_u0()
        except (TypeError, ValueError):
            return False
        return True


@dataclass(frozen=True)
class PanelRecord:
    """One candidate replay and its solver-only summary."""

    candidate: CandidateSpec
    result: TwoPassResult
    attachment: ReplayAttachment | None = None

    @property
    def physical_nfe(self) -> int:
        return self.result.physical_nfe

    @property
    def endpoint_head_calls(self) -> int:
        return self.result.endpoint_head_calls

    @property
    def total_dynamic_calls(self) -> int:
        return self.result.total_dynamic_calls

    def row(self) -> dict[str, Any]:
        """Return compact metadata suitable for a JSONL audit row."""

        row = {
            "candidate": self.candidate.name,
            "tier": self.candidate.tier,
            "purpose": self.candidate.purpose,
            "proposal_solver": self.candidate.plan.proposal.method,
            "proposal_schedule": self.candidate.plan.proposal.schedule.kind,
            "proposal_boundaries": list(self.candidate.plan.proposal.schedule.boundaries),
            "refined_solver": self.candidate.plan.refined.method,
            "refined_schedule": self.candidate.plan.refined.schedule.kind,
            "refined_boundaries": list(self.candidate.plan.refined.schedule.boundaries),
            "proposal_nfe": self.result.proposal.nfe,
            "refined_nfe": self.result.refined.nfe,
            "proposal_endpoint_calls": self.result.proposal.endpoint_calls,
            "refined_endpoint_calls": self.result.refined.endpoint_calls,
            "proposal_total_dynamic_calls": self.result.proposal.total_dynamic_calls,
            "refined_total_dynamic_calls": self.result.refined.total_dynamic_calls,
            "physical_nfe": self.physical_nfe,
            "endpoint_head_calls": self.endpoint_head_calls,
            "total_dynamic_calls": self.total_dynamic_calls,
            "proposal_solver_fingerprint": self.result.proposal.spec.fingerprint,
            "refined_solver_fingerprint": self.result.refined.spec.fingerprint,
            "two_pass_solver_fingerprint": self.candidate.plan.fingerprint,
            "cache_identity_changed": self.result.proposal_cache
            is not self.result.refined_cache,
            "initial_state_reused": bool(
                (self.result.proposal.initial_state == self.result.refined.initial_state)
                .all()
            ),
        }
        if self.attachment is not None:
            row["replay_attachment"] = self.attachment.to_dict()
            row["replay_attachment_schema_version"] = self.attachment.schema_version
            row["replay_attachment_fingerprint"] = self.attachment.fingerprint
        return row

    def with_attachment(self, attachment: ReplayAttachment) -> PanelRecord:
        return PanelRecord(self.candidate, self.result, attachment)


@dataclass(frozen=True)
class PanelResult:
    """Ordered results from a candidate panel."""

    records: tuple[PanelRecord, ...]

    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(record.row() for record in self.records)

    def by_name(self, name: str) -> PanelRecord:
        for record in self.records:
            if record.candidate.name == name:
                return record
        raise KeyError(f"candidate {name!r} is not present in this panel")


def run_candidate_panel(
    initial_state: Tensor,
    field: VelocityField,
    cache_factory: CacheFactory,
    rebuild_cache: CacheRebuilder,
    *,
    endpoint_head: EndpointHead,
    candidates: Iterable[CandidateSpec] | None = None,
    time_factory: TimeFactory | None = None,
    retain_trajectory: bool = False,
    retain_velocities: bool = False,
    record_corrections: bool = False,
    attachment_factory: Callable[[CandidateSpec, TwoPassResult], ReplayAttachment]
    | None = None,
) -> PanelResult:
    """Replay candidates with independent caches and identical initial values.

    ``cache_factory`` is intentionally called once per candidate.  A caller
    that wants exact fixed-checkpoint matching must make that factory
    deterministic and must not draw a different initial noise tensor inside
    it.  This function clones the supplied initial state for each replay so a
    misbehaving field cannot mutate the next candidate's source in place.
    """

    selected = tuple(candidate_matrix() if candidates is None else candidates)
    if not selected:
        raise ValueError("candidate panel cannot be empty")
    records: list[PanelRecord] = []
    cache_ids: set[int] = set()
    integration_kwargs: dict[str, Any] = {
        "endpoint_head": endpoint_head,
        "retain_trajectory": retain_trajectory,
        "retain_velocities": retain_velocities,
        "record_corrections": record_corrections,
    }
    if time_factory is not None:
        integration_kwargs["time_factory"] = time_factory
    for candidate in selected:
        candidate_cache = cache_factory()
        if id(candidate_cache) in cache_ids:
            raise ValueError(
                "cache_factory must return a fresh cache object for every candidate"
            )
        cache_ids.add(id(candidate_cache))
        result = run_two_pass(
            initial_state.clone(),
            field,
            candidate.plan,
            candidate_cache,
            rebuild_cache,
            **integration_kwargs,
        )
        refined_cache_id = id(result.refined_cache)
        if refined_cache_id in cache_ids or refined_cache_id == id(candidate_cache):
            raise ValueError(
                "rebuild_cache must return a fresh cache object for every candidate "
                "and must not alias a proposal cache"
            )
        cache_ids.add(refined_cache_id)
        attachment = (
            attachment_factory(candidate, result) if attachment_factory is not None else None
        )
        records.append(PanelRecord(candidate=candidate, result=result, attachment=attachment))
    return PanelResult(records=tuple(records))


__all__ = [
    "CacheFactory",
    "PanelRecord",
    "PanelResult",
    "ReplayAttachment",
    "run_candidate_panel",
]
