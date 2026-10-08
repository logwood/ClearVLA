"""Native array adaptation is not a claim of arbitrary checkpoint/outlet support."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

from clearvla.data.action_chart import project_episodes, resolve_action_state_profile
from clearvla.data.hdf5_episode import load_episode
from clearvla.data.native_contract import (
    EpisodeBoundary,
    NativeArrayProfile,
    NativeChannel,
    Timebase,
    observed_time_indices,
)
from clearvla.data.physical_chart import PhysicalChannelSpec, PhysicalChartSpec


def profile():
    return NativeArrayProfile(
        "fixture-arm-only-v2",
        2,
        3,
        (0, 1),
        (0, 1, 2),
        tuple(NativeChannel("u" + str(i), "rad/s", "joint", "velocity") for i in range(2)),
        tuple(NativeChannel("q" + str(i), "rad", "joint", "position") for i in range(3)),
        "velocity",
        ("external",),
        timebase=Timebase(seconds_per_step=0.02),
    )


def write(path):
    action = np.arange(10, dtype=np.float32).reshape(5, 2) / 100
    previous = np.concatenate((np.zeros((1, 2), np.float32), action[:-1]))
    with h5py.File(path, "w") as h:
        h["action"] = action
        h["measured"] = np.zeros((5, 3), np.float32)
        h["previous"] = previous
        h["camera"] = np.zeros((5, 4, 4, 3), np.uint8)


def test_real_hdf5_reader_keeps_independent_arrays_and_optional_effector(tmp_path):
    p = tmp_path / "sample.hdf5"
    write(p)
    spec = profile()
    ep = spec.read_hdf5(
        p, state_key="measured", previous_command_key="previous", camera_keys={"external": "camera"}
    )
    assert ep.states_raw is not None and ep.action_states_raw is not None
    assert (
        ep.actions_raw.shape == (5, 2)
        and ep.states_raw.shape == (5, 3)
        and ep.action_states_raw.shape == (5, 2)
    )
    assert not spec.effectors
    projected = project_episodes([ep], spec)[0].states_raw
    assert projected is not None and projected.shape == (5, 3)
    assert spec.digest() != replace(spec, command_reference="held_target").digest()
    # Historical data-reader rules are not silently relaxed.
    with pytest.raises(ValueError, match="dims must match"):
        load_episode(
            p,
            cameras=("external",),
            state_key="measured",
            action_state_key="previous",
            camera_key_overrides={"external": "camera"},
        )


def test_native_v2_never_invents_previous_command_or_measured_state(tmp_path):
    p = tmp_path / "sample.hdf5"
    write(p)
    with pytest.raises(ValueError):
        load_episode(
            p,
            cameras=("external",),
            camera_key_overrides={"external": "camera"},
            array_contract="independent-native-v2",
        )
    with h5py.File(p, "a") as h:
        dataset = h["previous"]
        assert isinstance(dataset, h5py.Dataset)
        dataset[3] = 1
    with pytest.raises(ValueError, match="not aligned"):
        profile().read_hdf5(
            p,
            state_key="measured",
            previous_command_key="previous",
            camera_keys={"external": "camera"},
        )


def test_physical_v2_allows_different_widths_without_changing_v1_metadata():
    a = (PhysicalChannelSpec("u", "rad/s", source="fixture"),)
    state = tuple(PhysicalChannelSpec("q" + str(i), "rad", source="fixture") for i in range(2))
    PhysicalChartSpec(
        "fixture", "velocity", a, state, "fixture", schema="clearvla-physical-chart-v2"
    ).validate()
    with pytest.raises(ValueError, match="widths must agree"):
        PhysicalChartSpec("fixture", "velocity", a, state, "fixture").validate()
    old = resolve_action_state_profile("calvin_relative_7d_v1")
    assert old.digest() == "6048e9233150352e761992a89377352b20f84fcbe4d31deb6c66c655af0ffe05"
    assert old.physical_chart.as_dict()["schema"] == "clearvla-physical-chart-v1"


def test_timebase_does_not_confuse_rows_seconds_or_availability():
    assert Timebase(seconds_per_step=0.05).duration_seconds(8) == 0.4
    assert Timebase(seconds_per_step=0.02).duration_seconds(8) == 0.16
    with pytest.raises(ValueError, match="unknown control frequency"):
        Timebase().duration_seconds(8)
    np.testing.assert_array_equal(
        observed_time_indices(np.array([0.0, 1.0, 2.0]), np.array([0.2, 1.5, 2.1]), now=1.2), [0]
    )
    with pytest.raises(ValueError):
        observed_time_indices(np.array([0.0, 1.0]), np.array([0.0, 0.9]), now=1.0)


def test_last_frame_does_not_manufacture_success():
    boundary = EpisodeBoundary(last_observation=True, truncated=True)
    boundary.validate()
    assert boundary.task_success is None and boundary.annotated_goal_observation is None
    simultaneous = replace(boundary, environment_terminal=True)
    simultaneous.validate()
    assert simultaneous.task_success is None
