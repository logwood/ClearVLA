from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from clearvla.benchmarks.common import audit_benchmark_dataset
from clearvla.benchmarks.config import build_benchmark_config
from clearvla.benchmarks.libero import convert_libero
from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.physical_chart import resolve_physical_chart_spec

ROOT = Path(__file__).resolve().parents[1]

def _write_source_task(
    root: Path,
    *,
    instruction: str = "put the red cube on the plate",
    task_name: str = "task",
) -> None:
    suite = root / "libero_spatial"
    suite.mkdir(parents=True)
    path = suite / f"{task_name}_demo.hdf5"
    length = 73
    action = np.zeros((length, 7), dtype=np.float64)
    action[:, :6] = np.linspace(-0.2, 0.2, length)[:, None]
    action[:, 6] = np.linspace(-0.8, 0.8, length)
    ee = np.arange(length * 6, dtype=np.float64).reshape(length, 6) / 100.0
    gripper = np.stack(
        (np.linspace(-0.1, 0.1, length), np.linspace(0.2, 0.4, length)), axis=1
    )
    top = np.zeros((length, 4, 5, 3), dtype=np.uint8)
    wrist = np.full((length, 3, 4, 3), 7, dtype=np.uint8)
    with h5py.File(path, "w") as stream:
        data = stream.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
        for index in range(3):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=action)
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=top)
            obs.create_dataset("eye_in_hand_rgb", data=wrist)
            obs.create_dataset("ee_states", data=ee)
            obs.create_dataset("gripper_states", data=gripper)


def _write_bddl_root(root: Path, *, instruction: str) -> Path:
    bddl = root / "libero_spatial"
    bddl.mkdir(parents=True)
    bddl.joinpath("SCENE1_" + instruction.replace(" ", "_") + ".bddl").write_text(
        "(:problem placeholder)", encoding="utf-8"
    )
    return root

def test_libero_profile_and_chart_declare_native_boundaries() -> None:
    profile = resolve_action_state_profile("libero_relative_7d_v1")
    assert profile.gripper_transition_boundary == "previous_command"
    assert profile.sampling_arm_motion == "relative_command_magnitude"
    chart = resolve_physical_chart_spec(profile.name)
    assert chart.output_dim == 7
    assert chart.action_channels[0].unit == "normalized_relative_command"
    assert chart.action_channels[-1].unit == "normalized_continuous_command"


def test_libero_converter_writes_and_audits_exact_entry_contract(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    instruction = "put the red cube on the plate"
    _write_source_task(source, instruction=instruction)
    bddl = _write_bddl_root(tmp_path / "bddl", instruction=instruction)
    output = tmp_path / "converted"

    report = convert_libero(
        source,
        output,
        suites=("libero_spatial",),
        evaluator_bddl_root=bddl,
    )
    assert report["benchmark"] == "LIBERO"
    assert report["episodes"] == 3
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_profile"] == "libero_relative_7d_v1"
    assert manifest["arm_flow_mode"] == "relative_command_direct"
    assert manifest["gripper_output_mode"] == "continuous"
    assert manifest["controller"] == "OSC_POSE"
    assert manifest["valid_center_end"] == "length - 49"
    assert manifest["evaluator_language_verified"] is True

    episode_path = next(output.glob("*.hdf5"))
    with h5py.File(episode_path, "r") as stream:
        action = np.asarray(stream["action"])
        action_state = np.asarray(stream["action_state"])
        assert action.shape == (73, 7)
        np.testing.assert_array_equal(action_state[0], np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(action_state[1:], action[:-1])
        assert stream.attrs["data_profile"] == "libero_relative_7d_v1"


def test_libero_converter_refuses_missing_instruction_and_bddl_conflict(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _write_source_task(source)
    with h5py.File(source / "libero_spatial" / "task_demo.hdf5", "r+") as stream:
        stream["data"].attrs["problem_info"] = json.dumps({})
    bddl = _write_bddl_root(
        tmp_path / "bddl_missing_instruction",
        instruction="put the red cube on the plate",
    )
    with pytest.raises(ValueError, match="language_instruction"):
        convert_libero(
            source,
            tmp_path / "bad_instruction",
            suites=("libero_spatial",),
            evaluator_bddl_root=bddl,
        )

    source = tmp_path / "raw_conflict"
    _write_source_task(source)
    bddl = _write_bddl_root(tmp_path / "bddl_conflict", instruction="a different command")
    with pytest.raises(ValueError, match="absent from the evaluator BDDL"):
        convert_libero(
            source,
            tmp_path / "bad_bddl",
            suites=("libero_spatial",),
            evaluator_bddl_root=bddl,
        )


def test_libero_converter_requires_official_evaluator_inventory(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    _write_source_task(source)

    with pytest.raises(ValueError, match="requires evaluator_bddl_root"):
        convert_libero(
            source,
            tmp_path / "missing_bddl",
            suites=("libero_spatial",),
        )
    with pytest.raises(ValueError, match="archive union"):
        convert_libero(
            source,
            tmp_path / "union",
            suites=("libero_100",),
            evaluator_bddl_root=tmp_path / "unused",
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    (
        ("nonfinite_action", "contains non-finite"),
        ("out_of_range_action", "outside the official normalized"),
        ("misaligned_state", "lost timestep alignment"),
        ("wrong_rgb_dtype", "must be uint8"),
    ),
)
def test_libero_converter_rejects_malformed_episode_before_atomic_publish(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    source = tmp_path / "raw"
    instruction = "put the red cube on the plate"
    _write_source_task(source, instruction=instruction)
    source_file = source / "libero_spatial" / "task_demo.hdf5"
    with h5py.File(source_file, "r+") as stream:
        demo = stream["data/demo_0"]
        if defect == "nonfinite_action":
            demo["actions"][0, 0] = np.nan
        elif defect == "out_of_range_action":
            demo["actions"][0, 0] = 1.5
        elif defect == "misaligned_state":
            obs = demo["obs"]
            values = np.asarray(obs["ee_states"])[:-1]
            del obs["ee_states"]
            obs.create_dataset("ee_states", data=values)
        elif defect == "wrong_rgb_dtype":
            obs = demo["obs"]
            values = np.asarray(obs["agentview_rgb"], dtype=np.float32)
            del obs["agentview_rgb"]
            obs.create_dataset("agentview_rgb", data=values)
        else:  # pragma: no cover - the parameter table is closed above.
            raise AssertionError(defect)
    bddl = _write_bddl_root(tmp_path / "bddl", instruction=instruction)
    output = tmp_path / "converted"

    with pytest.raises(ValueError, match=message):
        convert_libero(
            source,
            output,
            suites=("libero_spatial",),
            evaluator_bddl_root=bddl,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".converted.staging-*")) == []



