"""Unit-bearing physical reference metadata for action/state charts.

The action/state profile owns the numeric projection and its digest.  This
module deliberately owns a separate, metadata-only description of the units
and nominal ranges that make a raw value interpretable.  It must not be used
as an implicit clipping, normalization, or model-conditioning path.

In particular, a symmetric arm reference has two different useful numbers:
``nominal_abs_limit`` is the maximum absolute magnitude (``pi`` for the Pen
reference), while ``nominal_full_scale``/``nominal_span`` is the
upper-minus-lower span (``2*pi`` for ``[-pi, pi]``).  Keeping both prevents the
common ``pi`` versus ``2*pi`` ambiguity.  Mechanical bounds are separate and
may remain unknown even when a nominal reference is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PHYSICAL_CHART_SCHEMA = "clearvla-physical-chart-v1"


@dataclass(frozen=True)
class PhysicalChannelSpec:
    """Unit and nominal reference range for one projected channel.

    The limits are references, not runtime clamps and not necessarily the
    per-joint mechanical limits.  ``None`` means that the source has not yet
    established a trustworthy limit; it is preferable to inventing a value
    from another robot or from an observed training extrema.
    """

    name: str
    unit: str
    nominal_lower: float | None = None
    nominal_upper: float | None = None
    nominal_abs_limit: float | None = None
    mechanical_lower: float | None = None
    mechanical_upper: float | None = None
    source: str = ""

    @property
    def nominal_full_scale(self) -> float | None:
        """Return the nominal range span, when both endpoints are known."""

        if self.nominal_lower is None or self.nominal_upper is None:
            return None
        return float(self.nominal_upper - self.nominal_lower)

    @property
    def full_scale(self) -> float | None:
        """Short alias for the explicitly defined nominal span."""

        return self.nominal_full_scale

    def validate(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("physical channel names must be non-empty and trimmed")
        if not self.unit or self.unit.strip() != self.unit:
            raise ValueError("physical channel units must be non-empty and trimmed")
        if not self.source or self.source.strip() != self.source:
            raise ValueError("physical channel sources must be non-empty and trimmed")
        for label, lower, upper in (
            ("nominal", self.nominal_lower, self.nominal_upper),
            ("mechanical", self.mechanical_lower, self.mechanical_upper),
        ):
            if (lower is None) != (upper is None):
                raise ValueError(f"{label} channel bounds must be supplied together")
        values = (
            self.nominal_lower,
            self.nominal_upper,
            self.nominal_abs_limit,
            self.mechanical_lower,
            self.mechanical_upper,
        )
        for value in values:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("physical channel references must be finite")
        if self.nominal_lower is not None and self.nominal_upper is not None:
            if float(self.nominal_upper) < float(self.nominal_lower):
                raise ValueError("physical channel upper bound must not precede lower bound")
        if self.mechanical_lower is not None and self.mechanical_upper is not None:
            if float(self.mechanical_upper) < float(self.mechanical_lower):
                raise ValueError("mechanical upper bound must not precede lower bound")
        if self.nominal_abs_limit is not None and float(self.nominal_abs_limit) < 0.0:
            raise ValueError("physical channel absolute limit must be non-negative")

    def _quantity(self, value: float | None) -> dict[str, object] | None:
        if value is None:
            return None
        # Every serialized physical number carries its unit at the point of
        # use.  The sibling ``unit`` field remains a convenient channel-level
        # summary, but consumers need not infer units from field position.
        return {"value": float(value), "unit": self.unit}

    def as_dict(self) -> dict[str, object]:
        self.validate()
        nominal_range: dict[str, object] | None = None
        if self.nominal_lower is not None and self.nominal_upper is not None:
            nominal_range = {
                "lower": self._quantity(self.nominal_lower),
                "upper": self._quantity(self.nominal_upper),
                "span": self._quantity(self.nominal_full_scale),
                "meaning": "upper_minus_lower_span",
            }
        return {
            "name": self.name,
            "unit": self.unit,
            "nominal_abs_limit": self._quantity(self.nominal_abs_limit),
            "nominal_span": self._quantity(self.nominal_full_scale),
            "nominal_range": nominal_range,
            "mechanical_lower": self._quantity(self.mechanical_lower),
            "mechanical_upper": self._quantity(self.mechanical_upper),
            "source": self.source,
        }


@dataclass(frozen=True)
class PhysicalChartSpec:
    """Metadata-only units and nominal references for one chart profile."""

    name: str
    chart_kind: str
    action_channels: tuple[PhysicalChannelSpec, ...]
    state_channels: tuple[PhysicalChannelSpec, ...]
    source: str

    def validate(self) -> None:
        if not self.name or not self.chart_kind or not self.source:
            raise ValueError("physical chart identity and source must be non-empty")
        if not self.action_channels or not self.state_channels:
            raise ValueError("physical chart must describe action and state channels")
        if len(self.action_channels) != len(self.state_channels):
            raise ValueError("action/state physical channel widths must agree")
        for channels in (self.action_channels, self.state_channels):
            names = tuple(channel.name for channel in channels)
            if len(set(names)) != len(names):
                raise ValueError("physical channel names must be unique within a chart")
            for channel in channels:
                channel.validate()

    @property
    def output_dim(self) -> int:
        return len(self.action_channels)

    @property
    def action_units(self) -> tuple[str, ...]:
        return tuple(channel.unit for channel in self.action_channels)

    @property
    def state_units(self) -> tuple[str, ...]:
        return tuple(channel.unit for channel in self.state_channels)

    @staticmethod
    def _summary_unit(units: tuple[str, ...]) -> str:
        return units[0] if units and len(set(units)) == 1 else "mixed"

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": PHYSICAL_CHART_SCHEMA,
            "name": self.name,
            "chart_kind": self.chart_kind,
            "source": self.source,
            "action_unit": self._summary_unit(self.action_units),
            "state_unit": self._summary_unit(self.state_units),
            "action_units": list(self.action_units),
            "state_units": list(self.state_units),
            "action_channels": [channel.as_dict() for channel in self.action_channels],
            "state_channels": [channel.as_dict() for channel in self.state_channels],
        }


def _pen_channels() -> tuple[PhysicalChannelSpec, ...]:
    arm_source = "pen_nominal_reference_rad_not_per_joint_mechanical_limits"
    gripper_source = "pen_nominal_gripper_opening_100_deg"
    arm = tuple(
        PhysicalChannelSpec(
            name=f"joint_{index + 1}",
            unit="rad",
            nominal_lower=-math.pi,
            nominal_upper=math.pi,
            nominal_abs_limit=math.pi,
            source=arm_source,
        )
        for index in range(6)
    )
    gripper_upper = math.radians(100.0)
    gripper = PhysicalChannelSpec(
        name="gripper",
        unit="rad",
        nominal_lower=0.0,
        nominal_upper=gripper_upper,
        nominal_abs_limit=gripper_upper,
        source=gripper_source,
    )
    return (*arm, gripper)


def _unknown_channels(names: tuple[str, ...], *, source: str) -> tuple[PhysicalChannelSpec, ...]:
    return tuple(
        PhysicalChannelSpec(name=name, unit="source_native", source=source)
        for name in names
    )


_PEN_CHANNELS = _pen_channels()
_CALVIN_ACTION_CHANNELS = (
    *tuple(
        PhysicalChannelSpec(
            name=name,
            unit="normalized_relative_command",
            nominal_lower=-1.0,
            nominal_upper=1.0,
            nominal_abs_limit=1.0,
            source=(
                "CALVIN Robot.relative_to_absolute: command multiplied by "
                "max_rel_pos=0.02 m"
            ),
        )
        for name in ("delta_x", "delta_y", "delta_z")
    ),
    *tuple(
        PhysicalChannelSpec(
            name=name,
            unit="normalized_relative_command",
            nominal_lower=-1.0,
            nominal_upper=1.0,
            nominal_abs_limit=1.0,
            source=(
                "CALVIN Robot.relative_to_absolute: command multiplied by "
                "max_rel_orn=0.05 rad"
            ),
        )
        for name in ("delta_roll", "delta_pitch", "delta_yaw")
    ),
    PhysicalChannelSpec(
        name="gripper_command",
        unit="binary_command",
        nominal_lower=-1.0,
        nominal_upper=1.0,
        nominal_abs_limit=1.0,
        source="CALVIN Robot.apply_action requires gripper_action in {-1,+1}",
    ),
)
_CALVIN_STATE_CHANNELS = (
    *tuple(
        PhysicalChannelSpec(
            name=name,
            unit="m",
            source="CALVIN robot_obs[:3] world-frame TCP position",
        )
        for name in ("tcp_x", "tcp_y", "tcp_z")
    ),
    *tuple(
        PhysicalChannelSpec(
            name=name,
            unit="rad",
            source="CALVIN robot_obs[3:6] world-frame TCP Euler orientation",
        )
        for name in ("tcp_roll", "tcp_pitch", "tcp_yaw")
    ),
    PhysicalChannelSpec(
        name="gripper_opening_width",
        unit="m",
        source="CALVIN robot_obs[6] summed finger opening width",
    ),
)
_RDT_LEFT_CHANNEL_NAMES = tuple(f"left_joint_{index + 1}" for index in range(6)) + (
    "left_gripper",
)
_RDT_RIGHT_CHANNEL_NAMES = tuple(f"right_joint_{index + 1}" for index in range(6)) + (
    "right_gripper",
)
_RDT_BIMANUAL_CHANNEL_NAMES = _RDT_LEFT_CHANNEL_NAMES + _RDT_RIGHT_CHANNEL_NAMES


def _chart(
    name: str,
    action_channels: tuple[PhysicalChannelSpec, ...],
    *,
    chart_kind: str,
    source: str,
    state_channels: tuple[PhysicalChannelSpec, ...] | None = None,
) -> PhysicalChartSpec:
    result = PhysicalChartSpec(
        name=name,
        chart_kind=chart_kind,
        action_channels=action_channels,
        state_channels=action_channels if state_channels is None else state_channels,
        source=source,
    )
    result.validate()
    return result


PHYSICAL_CHART_SPECS: dict[str, PhysicalChartSpec] = {
    "identity_7d_pen": _chart(
        "identity_7d_pen",
        _PEN_CHANNELS,
        chart_kind="physical_reference",
        source="Pen action/state chart; nominal references are metadata only",
    ),
    "calvin_relative_7d_v1": _chart(
        "calvin_relative_7d_v1",
        _CALVIN_ACTION_CHANNELS,
        chart_kind="calvin_normalized_relative_tcp_command",
        source=(
            "Official CALVIN calvin_env Robot defaults: max_rel_pos=0.02, "
            "max_rel_orn=0.05, binary gripper"
        ),
        state_channels=_CALVIN_STATE_CHANNELS,
    ),
    "rdt_right_arm_action_chart_v1": _chart(
        "rdt_right_arm_action_chart_v1",
        _unknown_channels(
            _RDT_RIGHT_CHANNEL_NAMES,
            source="RDT right-arm command-chart units and limits are not yet established",
        ),
        chart_kind="source_native_unknown",
        source="RDT right-arm native action/qpos limits are not yet established",
        state_channels=_unknown_channels(
            _RDT_RIGHT_CHANNEL_NAMES,
            source="RDT right-arm qpos-chart units and limits are not yet established",
        ),
    ),
    "rdt_left_arm_action_chart_v1": _chart(
        "rdt_left_arm_action_chart_v1",
        _unknown_channels(
            _RDT_LEFT_CHANNEL_NAMES,
            source="RDT left-arm command-chart units and limits are not yet established",
        ),
        chart_kind="source_native_unknown",
        source="RDT left-arm native action/qpos limits are not yet established",
        state_channels=_unknown_channels(
            _RDT_LEFT_CHANNEL_NAMES,
            source="RDT left-arm qpos-chart units and limits are not yet established",
        ),
    ),
    "rdt_bimanual_action_chart_v1": _chart(
        "rdt_bimanual_action_chart_v1",
        _unknown_channels(
            _RDT_BIMANUAL_CHANNEL_NAMES,
            source="RDT bimanual command-chart units and limits are not yet established",
        ),
        chart_kind="source_native_unknown",
        source="RDT bimanual native action/qpos limits are not yet established",
        state_channels=_unknown_channels(
            _RDT_BIMANUAL_CHANNEL_NAMES,
            source="RDT bimanual qpos-chart units and limits are not yet established",
        ),
    ),
}


def resolve_physical_chart_spec(name: str) -> PhysicalChartSpec:
    """Resolve metadata without changing the numeric action profile."""

    try:
        return PHYSICAL_CHART_SPECS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown physical chart profile {name!r}; "
            f"known={sorted(PHYSICAL_CHART_SPECS)}"
        ) from error


def physical_chart_metadata(name: str) -> dict[str, object]:
    """Return a detached JSON-ready metadata mapping for one profile."""

    return resolve_physical_chart_spec(name).as_dict()


__all__ = [
    "PHYSICAL_CHART_SCHEMA",
    "PHYSICAL_CHART_SPECS",
    "PhysicalChannelSpec",
    "PhysicalChartSpec",
    "physical_chart_metadata",
    "resolve_physical_chart_spec",
]
