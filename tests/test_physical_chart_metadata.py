from __future__ import annotations

import math

import pytest

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.physical_chart import (
    PHYSICAL_CHART_SCHEMA,
    PhysicalChannelSpec,
    physical_chart_metadata,
    resolve_physical_chart_spec,
)


def test_pen_reference_distinguishes_absolute_limit_from_full_scale() -> None:
    spec = resolve_physical_chart_spec("identity_7d_pen")
    assert spec.output_dim == 7
    assert spec.as_dict()["action_unit"] == "rad"
    assert spec.as_dict()["state_unit"] == "rad"
    assert all(channel.unit == "rad" for channel in spec.action_channels)

    arm = spec.action_channels[0]
    assert arm.nominal_abs_limit == pytest.approx(math.pi)
    assert arm.nominal_full_scale == pytest.approx(2.0 * math.pi)
    arm_json = arm.as_dict()
    assert arm_json["nominal_abs_limit"] == {"value": math.pi, "unit": "rad"}
    assert arm_json["nominal_range"]["lower"] == {"value": -math.pi, "unit": "rad"}
    assert arm_json["nominal_range"]["upper"] == {"value": math.pi, "unit": "rad"}
    assert arm_json["nominal_range"]["span"]["unit"] == "rad"
    assert arm_json["nominal_span"] == {"value": 2.0 * math.pi, "unit": "rad"}
    assert arm_json["mechanical_lower"] is None
    assert arm_json["mechanical_upper"] is None

    gripper = spec.action_channels[-1]
    hundred_degrees = math.radians(100.0)
    assert gripper.nominal_abs_limit == pytest.approx(hundred_degrees)
    assert gripper.nominal_full_scale == pytest.approx(hundred_degrees)
    assert gripper.as_dict()["nominal_range"]["upper"] == {
        "value": hundred_degrees,
        "unit": "rad",
    }


def test_unknown_rdt_limits_are_explicitly_source_native() -> None:
    spec = resolve_physical_chart_spec("rdt_right_arm_action_chart_v1")
    assert spec.chart_kind == "source_native_unknown"
    assert spec.as_dict()["action_unit"] == "source_native"
    assert spec.as_dict()["state_unit"] == "source_native"
    assert all(channel.unit == "source_native" for channel in spec.action_channels)
    assert all(channel.nominal_abs_limit is None for channel in spec.action_channels)
    assert all(channel.as_dict()["nominal_range"] is None for channel in spec.action_channels)
    assert spec.action_channels[-1].source != spec.state_channels[-1].source


def test_physical_metadata_is_separate_from_numeric_profile_digest() -> None:
    profile = resolve_action_state_profile("identity_7d_pen")
    assert "physical_chart" not in profile.as_dict()
    assert profile.physical_chart.as_dict()["schema"] == PHYSICAL_CHART_SCHEMA
    assert physical_chart_metadata(profile.name) == profile.physical_chart.as_dict()


def test_channel_rejects_partial_nominal_range() -> None:
    channel = PhysicalChannelSpec(
        name="joint",
        unit="rad",
        nominal_lower=-math.pi,
        source="test",
    )
    with pytest.raises(ValueError, match="bounds must be supplied together"):
        channel.validate()
