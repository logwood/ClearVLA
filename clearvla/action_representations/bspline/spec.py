"""Versioned configuration for the standalone B-spline action chart."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping, Sequence

RepresentationMode = Literal[
    "hierarchical_exact",
    "compact",
]


@dataclass(frozen=True)
class BSplineSpec:
    """Immutable identity of one B-spline action representation.

    ``sample_times`` are the actual timestamps of the native action rows.  They
    are part of the representation identity; a caller cannot silently feed a
    different control rate to a codec constructed from this specification.
    """

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1
    REPRESENTATION_NAME: ClassVar[str] = "clearvla.bspline_action"

    sample_times: tuple[float, ...]
    arm_dim: int
    num_control_points: int
    degree: int = 3
    mode: RepresentationMode = "hierarchical_exact"
    # Reserved only so an older serialized prototype fails with a targeted
    # error instead of being misread.  Partial detail retention is deliberately
    # unavailable until a principled, fingerprinted ordering exists.
    detail_budget: int | None = None
    time_unit: str = "normalized"
    channel_names: tuple[str, ...] = ()
    channel_units: tuple[str, ...] = ()
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_times", tuple(float(v) for v in self.sample_times))
        object.__setattr__(self, "channel_names", tuple(str(v) for v in self.channel_names))
        object.__setattr__(self, "channel_units", tuple(str(v) for v in self.channel_units))
        object.__setattr__(self, "arm_dim", int(self.arm_dim))
        object.__setattr__(self, "num_control_points", int(self.num_control_points))
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if self.detail_budget is not None:
            object.__setattr__(self, "detail_budget", int(self.detail_budget))
        self.validate()

    @classmethod
    def uniform(
        cls,
        *,
        horizon: int,
        arm_dim: int,
        num_control_points: int,
        degree: int = 3,
        mode: RepresentationMode = "hierarchical_exact",
        detail_budget: int | None = None,
        start: float = 0.0,
        stop: float = 1.0,
        time_unit: str = "normalized",
        channel_names: Sequence[str] = (),
        channel_units: Sequence[str] = (),
    ) -> BSplineSpec:
        """Construct a spec with an explicit uniform timestamp grid."""

        horizon = int(horizon)
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        start = float(start)
        stop = float(stop)
        if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
            raise ValueError("uniform time bounds must be finite with stop > start")
        step = (stop - start) / float(horizon - 1)
        times = tuple(start + step * index for index in range(horizon))
        # Preserve the caller's exact endpoints rather than their accumulated
        # floating-point approximations.
        times = (start, *times[1:-1], stop)
        return cls(
            sample_times=times,
            arm_dim=arm_dim,
            num_control_points=num_control_points,
            degree=degree,
            mode=mode,
            detail_budget=detail_budget,
            time_unit=time_unit,
            channel_names=tuple(channel_names),
            channel_units=tuple(channel_units),
        )

    @property
    def horizon(self) -> int:
        return len(self.sample_times)

    @property
    def coarse_rank(self) -> int:
        return self.num_control_points

    @property
    def available_detail_rank(self) -> int:
        return self.horizon - self.coarse_rank

    @property
    def retained_detail_rank(self) -> int:
        if self.mode == "hierarchical_exact":
            return self.available_detail_rank
        return 0

    @property
    def coordinate_rank(self) -> int:
        return self.coarse_rank + self.retained_detail_rank

    @property
    def is_lossless(self) -> bool:
        return self.coordinate_rank == self.horizon

    def validate(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported B-spline schema {self.schema_version}; "
                f"expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if self.arm_dim < 1:
            raise ValueError("arm_dim must be positive")
        if self.degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2 or 3")
        if self.horizon < self.degree + 1:
            raise ValueError("sample_times must contain at least degree + 1 rows")
        if not all(math.isfinite(value) for value in self.sample_times):
            raise ValueError("sample_times must be finite")
        if any(b <= a for a, b in zip(self.sample_times, self.sample_times[1:])):
            raise ValueError("sample_times must be strictly increasing")
        if self.num_control_points < self.degree + 1:
            raise ValueError("num_control_points must be at least degree + 1")
        if self.num_control_points >= self.horizon:
            raise ValueError(
                "num_control_points must be smaller than the horizon so the "
                "stable detail complement remains present"
            )
        if self.detail_budget is not None:
            raise ValueError(
                "detail_budget is unavailable: the canonical detail complement "
                "has no principled coarse-to-fine ordering"
            )
        if self.mode not in {
            "hierarchical_exact",
            "compact",
        }:
            raise ValueError(f"unsupported representation mode: {self.mode!r}")
        if not self.time_unit.strip():
            raise ValueError("time_unit must be a non-empty label")
        for label, values in (
            ("channel_names", self.channel_names),
            ("channel_units", self.channel_units),
        ):
            if values and len(values) != self.arm_dim:
                raise ValueError(f"{label} must be empty or contain exactly arm_dim entries")
            if any(not value.strip() for value in values):
                raise ValueError(f"{label} entries must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, human-readable configuration."""

        return {
            "representation": self.REPRESENTATION_NAME,
            "schema_version": self.schema_version,
            "sample_times": list(self.sample_times),
            "arm_dim": self.arm_dim,
            "num_control_points": self.num_control_points,
            "degree": self.degree,
            "mode": self.mode,
            "detail_budget": self.detail_budget,
            "time_unit": self.time_unit,
            "channel_names": list(self.channel_names),
            "channel_units": list(self.channel_units),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BSplineSpec:
        """Rebuild a spec while rejecting a different representation family."""

        data = dict(value)
        allowed = {
            "representation",
            "schema_version",
            "sample_times",
            "arm_dim",
            "num_control_points",
            "degree",
            "mode",
            "detail_budget",
            "time_unit",
            "channel_names",
            "channel_units",
        }
        unknown = set(data).difference(allowed)
        if unknown:
            raise ValueError(f"unknown B-spline spec fields: {sorted(unknown)}")
        representation = data.pop("representation", cls.REPRESENTATION_NAME)
        if representation != cls.REPRESENTATION_NAME:
            raise ValueError(f"not a {cls.REPRESENTATION_NAME} specification")
        return cls(
            sample_times=tuple(data["sample_times"]),
            arm_dim=data["arm_dim"],
            num_control_points=data["num_control_points"],
            degree=data.get("degree", 3),
            mode=data.get("mode", "hierarchical_exact"),
            detail_budget=data.get("detail_budget"),
            time_unit=data.get("time_unit", "normalized"),
            channel_names=tuple(data.get("channel_names", ())),
            channel_units=tuple(data.get("channel_units", ())),
            schema_version=data.get("schema_version", cls.CURRENT_SCHEMA_VERSION),
        )

    def identity_dict(self) -> dict[str, Any]:
        """Return the canonical bit-level identity used by ``fingerprint``."""

        value = self.to_dict()
        value["sample_times_hex"] = [number.hex() for number in self.sample_times]
        del value["sample_times"]
        return value

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.identity_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["BSplineSpec", "RepresentationMode"]
