from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import h5py
import numpy as np
import pytest
import torch

from clearvla.benchmarks.config import build_benchmark_config
from clearvla.benchmarks.libero import convert_libero
from clearvla.benchmarks.libero_eval import (
    LIBERO_OFFICIAL_EPISODES_PER_TASK,
    LIBERO_OFFICIAL_MAX_STEPS,
    LIBERO_OFFICIAL_WARMUP_STEPS,
    LIBERO_POLICY_HORIZON,
    LiberoBridgePolicy,
    _step_observation_only,
    execute_libero_action,
    libero_bridge_identity,
    libero_policy_observation,
    validate_libero_bridge_health,
)
from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.physical_chart import resolve_physical_chart_spec
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.model.components import OutletAdapter
from clearvla.mainline.runtime.deployment import (
    CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
    DEPLOYMENT_ABI_SCHEMA,
    canonical_sha256,
    deployment_graph_config,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.evaluation import libero_gripper_sign_projection

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


def _codec() -> PhysicalActionFieldCodec:
    return PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
    )


def test_libero_gripper_sign_projection_is_outlet_local_and_normalizer_aware() -> None:
    prediction = torch.zeros(2, 24, 7)
    prediction[..., :6] = torch.linspace(-0.3, 0.3, 24)[None, :, None]
    physical = torch.zeros(2, 24, 18)
    offset = torch.tensor([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, -0.25]).view(1, 1, 7)
    scale = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5]).view(1, 1, 7)
    # Normalized zero for gripper is its affine offset, not literal zero.
    prediction[0, :, -1] = offset[0, 0, -1] + scale[0, 0, -1] * 0.2
    prediction[1, :, -1] = offset[0, 0, -1] + scale[0, 0, -1] * -0.2
    physical[0, :, -6] = offset[0, 0, -1] + scale[0, 0, -1] * -0.3
    physical[1, :, -6] = offset[0, 0, -1] + scale[0, 0, -1] * 0.3
    boundary = torch.tensor([[-0.75], [0.25]]) * scale[..., -1].reshape(1, 1)
    boundary = boundary + offset[..., -1].reshape(1, 1)

    projected = libero_gripper_sign_projection(
        prediction,
        physical,
        boundary,
        action_offset=offset,
        action_scale=scale,
    )

    assert set(projected) == {"sign_projected", "absolute_sign_projected"}
    for value in projected.values():
        torch.testing.assert_close(value[..., :6], prediction[..., :6])
        raw_gripper = (value[..., -1:] - offset[..., -1:]) / scale[..., -1:]
        assert set(raw_gripper.unique().tolist()) <= {-1.0, 1.0}
    torch.testing.assert_close(
        (projected["sign_projected"][..., -1:] - offset[..., -1:])
        / scale[..., -1:],
        torch.tensor([1.0, -1.0]).view(2, 1, 1).expand(2, 24, 1),
    )
    torch.testing.assert_close(
        (projected["absolute_sign_projected"][..., -1:] - offset[..., -1:])
        / scale[..., -1:],
        torch.tensor([-1.0, 1.0]).view(2, 1, 1).expand(2, 24, 1),
    )


def _observation(step: int = 0) -> dict[str, object]:
    return {
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.2, -0.1], dtype=np.float32),
        "agentview_image": np.full((4, 5, 3), step, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((3, 4, 3), step + 1, dtype=np.uint8),
    }


def _formal_libero_abi() -> dict[str, object]:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(base.data, data_profile="libero_relative_7d_v1"),
        bottom=replace(
            base.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode="continuous",
        ),
    )
    config.validate()
    profile = resolve_action_state_profile("libero_relative_7d_v1")
    profile_payload = {
        **profile.as_dict(),
        "sha256": profile.digest(),
        "gripper_transition_boundary": profile.gripper_transition_boundary,
    }
    graph = deployment_graph_config(config)
    action = {
        "data_profile": profile_payload,
        "gripper_indices": [6],
        "gripper_output_mode": "continuous",
        "arm_flow_mode": "relative_command_adapter",
        "continuous_gripper_codec_boundary": "previous_command",
        "continuous_gripper_codec_boundary_scope": CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
        "receding_horizon_execute_rows": 1,
        "prediction_horizon": 24,
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "normalized_low": [-1.0] * 7,
        "normalized_high": [1.0] * 7,
        "normalizers": {
            "mode": "zscore",
            "action_sha256": "a" * 64,
            "state_sha256": "b" * 64,
        },
    }
    return {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "observation": {
            "camera_names": ["top", "wrist"],
            "visual_offsets": [-8, -4, 0],
            "state_offsets": [-8, -4, 0],
            "executed_action_offsets": [-24, -16, -12, -8, -6, -4, -2, -1],
            "state_dim": 7,
            "action_dim": 7,
            "dinov2": {
                "model": "test",
                "compute_dtype": "fp32",
                "reference_batch_size": 1,
            },
        },
        "action": action,
        "normalizers": {
            "mode": "zscore",
            "action_sha256": "a" * 64,
            "state_sha256": "b" * 64,
        },
        "language": {"sha256": "c" * 64},
    }


class _Client:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self.chunks = chunks
        self.calls: list[object] = []

    def act(self, observation, instruction: str, *, reset: bool) -> np.ndarray:
        self.calls.append((observation, instruction, reset))
        return self.chunks[len(self.calls) - 1]


def _install_fake_libero_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    suite: object,
    env_type: type,
) -> None:
    """Install the minimal external module surface used by ``evaluate_libero``."""

    benchmark_module = types.ModuleType("libero.libero.benchmark")
    benchmark_module.get_benchmark_dict = lambda: {  # type: ignore[attr-defined]
        "libero_spatial": lambda task_order_index=0: suite
    }
    libero_module = types.ModuleType("libero.libero")
    libero_module.benchmark = benchmark_module  # type: ignore[attr-defined]
    env_module = types.ModuleType("libero.libero.envs")
    env_module.OffScreenRenderEnv = env_type  # type: ignore[attr-defined]
    robosuite_transform = types.ModuleType("robosuite.utils.transform_utils")
    robosuite_transform.quat2axisangle = (  # type: ignore[attr-defined]
        lambda quaternion: np.zeros(3, dtype=np.float32)
    )
    robosuite_utils = types.ModuleType("robosuite.utils")
    robosuite_utils.transform_utils = robosuite_transform  # type: ignore[attr-defined]
    robosuite_module = types.ModuleType("robosuite")
    robosuite_module.utils = robosuite_utils  # type: ignore[attr-defined]
    libero_root = types.ModuleType("libero")
    libero_root.libero = libero_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", libero_root)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", env_module)
    monkeypatch.setitem(sys.modules, "robosuite", robosuite_module)
    monkeypatch.setitem(sys.modules, "robosuite.utils", robosuite_utils)
    monkeypatch.setitem(sys.modules, "robosuite.utils.transform_utils", robosuite_transform)


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
    assert manifest["arm_flow_mode"] == "relative_command_adapter"
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


def test_libero_benchmark_config_cannot_fall_back_to_pen_chart(tmp_path: Path) -> None:
    converted = tmp_path / "converted"
    converted.mkdir()
    (converted / "splits.json").write_text(
        json.dumps(
            {
                "schema": "clearvla-episode-splits-v1",
                "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
            }
        ),
        encoding="utf-8",
    )
    (converted / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema": "clearvla-external-benchmark-dataset-v1",
                "converter_schema": "clearvla-libero-converter-v1",
                "benchmark": "LIBERO",
                "data_profile": "libero_relative_7d_v1",
                "arm_flow_mode": "relative_command_adapter",
                "gripper_output_mode": "continuous",
                "controller": "OSC_POSE",
                "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
                "split_unit": "episode",
                "valid_center_start": 24,
                "valid_center_end": "length - 49",
                "evaluator_language_verified": True,
            }
        ),
        encoding="utf-8",
    )
    payload = build_benchmark_config(
        ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
        tmp_path / "libero.json",
        hdf5_root=converted,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
        data_profile="libero_relative_7d_v1",
        arm_flow_mode="relative_command_adapter",
        gripper_output_mode="continuous",
        gripper_event_threshold=0.1,
    )
    assert payload["data"]["data_profile"] == "libero_relative_7d_v1"
    assert payload["bottom"]["arm_flow_mode"] == "relative_command_adapter"

    manifest_path = converted / "dataset_manifest.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["evaluator_language_verified"] = False
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator language must be verified"):
        build_benchmark_config(
            ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
            tmp_path / "unverified.json",
            hdf5_root=converted,
            cache_root=tmp_path / "cache-unverified",
            language_bank=tmp_path / "language-unverified.pt",
            run_root=tmp_path / "run-unverified",
        )

    stale["evaluator_language_verified"] = True
    stale.pop("controller")
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required LIBERO manifest fields"):
        build_benchmark_config(
            ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
            tmp_path / "stale.json",
            hdf5_root=converted,
            cache_root=tmp_path / "cache2",
            language_bank=tmp_path / "language2.pt",
            run_root=tmp_path / "run2",
        )


def test_libero_relative_adapter_preserves_source_and_centers_zero_offset() -> None:
    offset = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.5, 0.25, 0.6]])
    adapter = OutletAdapter(_codec(), selection="libero_7d_continuous_v1")
    adapter.configure_action_normalizer(
        SimpleNamespace(offset=offset, scale=torch.ones(1, 7))
    )
    command = torch.tensor(
        [[[0.1, 0.0, -0.1, 0.2, 0.0, 0.05]]], dtype=torch.float32
    ).expand(2, 4, -1)
    source = offset[:, None].expand(2, 4, -1).clone()
    source[..., :6] += command
    source[..., -1] = torch.tensor([[0.8, 0.8, -0.4, -0.4]])
    current = offset.expand(2, -1).clone()
    current[..., -1] = -0.4
    condition = adapter.world_condition_from_interval_action(source, current)
    torch.testing.assert_close(condition.interval_action[..., :6], torch.cumsum(command, dim=1))
    torch.testing.assert_close(condition.interval_delta[..., :6], command)
    torch.testing.assert_close(
        condition.interval_delta[..., -1],
        source[..., -1]
        - torch.cat((current[:, None, -1], source[:, :-1, -1]), dim=1),
    )
    assert condition.source_interval_action is source


def test_libero_exit_clips_and_reuses_executed_action_as_action_state() -> None:
    raw = np.zeros((LIBERO_POLICY_HORIZON, 7), dtype=np.float32)
    raw[0] = np.asarray([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.5], dtype=np.float32)
    raw[1, 0] = 0.25
    client = _Client([raw, raw])
    policy = LiberoBridgePolicy(
        client, quat_to_axisangle=lambda quaternion: np.asarray(quaternion[1:], dtype=np.float32)
    )
    first = policy.step(_observation(0), "put the cube")
    np.testing.assert_array_equal(first, np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    # A transport client must not be able to mutate the bridge's retained
    # executed-action boundary through a shared request array.
    client.calls[0][0].action_state[...] = -7.0
    second = policy.step(_observation(1), "put the cube")
    np.testing.assert_allclose(client.calls[1][0].action_state, first)
    np.testing.assert_allclose(second, raw[0].clip(-1.0, 1.0))
    assert policy.action_audit()["action_state_matches_executed"] is True


def test_libero_exit_rejects_wrong_chunk_and_nonfinite_observation() -> None:
    client = _Client([np.zeros((1, 7), dtype=np.float32)])
    policy = LiberoBridgePolicy(client, quat_to_axisangle=lambda quaternion: quaternion[1:])
    with pytest.raises(ValueError, match=r"exactly a \[24,7\]"):
        policy.step(_observation(), "put the cube")
    bad = _observation()
    bad["robot0_eef_pos"] = np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="robot0_eef_pos"):
        libero_policy_observation(
            bad,
            np.zeros(7, dtype=np.float32),
            quat_to_axisangle=lambda quaternion: quaternion[1:],
        )
    with pytest.raises(ValueError, match="camera shapes differ"):
        libero_policy_observation(
            _observation(),
            np.zeros(7, dtype=np.float32),
            quat_to_axisangle=lambda quaternion: quaternion[1:],
            expected_image_side=128,
        )


def test_libero_health_rejects_smoke_without_explicit_escape() -> None:
    with pytest.raises(RuntimeError, match="smoke-zero"):
        validate_libero_bridge_health({"status": "ok", "mode": "smoke-zero"})
    validate_libero_bridge_health(
        {"status": "ok", "mode": "smoke-zero"}, allow_smoke_policy=True
    )


def test_libero_action_executor_rejects_nonfinite_rows() -> None:
    with pytest.raises(ValueError, match="finite"):
        execute_libero_action(np.full(7, np.nan, dtype=np.float32))


def test_libero_evaluation_refuses_directory_output_before_external_import(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    from clearvla.benchmarks import libero_eval

    with pytest.raises(FileExistsError, match="not a file"):
        libero_eval.evaluate_libero(
            "libero_spatial",
            output,
            endpoint="http://127.0.0.1:8765",
            timeout=1.0,
            task_ids=[0],
            episodes_per_task=1,
            max_steps=1,
            warmup_steps=0,
            seed=0,
            image_side=4,
            resume=False,
        )


def test_libero_evaluation_refuses_existing_file_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clearvla.benchmarks import libero_eval

    output = tmp_path / "result.json"
    output.write_text('{"sentinel": true}', encoding="utf-8")

    class UnexpectedClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("overwrite refusal must happen before bridge contact")

    monkeypatch.setattr(libero_eval, "RemotePolicyClient", UnexpectedClient)
    with pytest.raises(FileExistsError, match="already exists"):
        libero_eval.evaluate_libero(
            "libero_spatial",
            output,
            endpoint="http://127.0.0.1:8765",
            timeout=1.0,
            task_ids=[0],
            episodes_per_task=1,
            max_steps=1,
            warmup_steps=0,
            seed=0,
            image_side=4,
            resume=False,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}


def test_libero_evaluation_rejects_archive_union_before_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clearvla.benchmarks import libero_eval

    class UnexpectedClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("unsupported suite must be rejected before bridge contact")

    monkeypatch.setattr(libero_eval, "RemotePolicyClient", UnexpectedClient)
    with pytest.raises(ValueError, match="archive union"):
        libero_eval.evaluate_libero(
            "libero_100",
            tmp_path / "result.json",
            endpoint="http://127.0.0.1:8765",
            timeout=1.0,
            task_ids=None,
            episodes_per_task=20,
            max_steps=600,
            warmup_steps=5,
            seed=10000,
            image_side=128,
            resume=False,
        )


def test_libero_resume_requires_complete_executed_action_audit() -> None:
    from clearvla.benchmarks.libero_eval import _validate_rollout_row

    row = {
        "episode": 0,
        "init_state": 0,
        "steps": 1,
        "success": True,
        "action_audit": {
            "steps": 1,
            "action_state_input_count": 1,
            "action_state_matches_executed": True,
            "action_names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            "raw_min": [0.0] * 7,
            "raw_max": [0.0] * 7,
            "raw_oob_count_per_dim": [0] * 7,
            "raw_oob_row_count": 0,
            "executed_min": [0.0] * 7,
            "executed_max": [0.0] * 7,
            "executed_abs_max": [0.0] * 7,
            "clipped_row_count": 0,
        },
    }
    assert _validate_rollout_row(
        row,
        expected_episode=0,
        episodes_per_task=20,
        max_steps=600,
    )["action_audit"] == row["action_audit"]

    missing_audit = dict(row)
    missing_audit.pop("action_audit")
    with pytest.raises(ValueError, match="action_audit must be a mapping"):
        _validate_rollout_row(
            missing_audit,
            expected_episode=0,
            episodes_per_task=20,
            max_steps=600,
        )

    zero_step = dict(row, steps=0)
    with pytest.raises(ValueError, match="steps must be at least 1"):
        _validate_rollout_row(
            zero_step,
            expected_episode=0,
            episodes_per_task=20,
            max_steps=600,
        )

    stale_action_state_count = json.loads(json.dumps(row))
    stale_action_state_count["action_audit"]["action_state_input_count"] = 0
    with pytest.raises(ValueError, match="action_state_input_count must be at least 1"):
        _validate_rollout_row(
            stale_action_state_count,
            expected_episode=0,
            episodes_per_task=20,
            max_steps=600,
        )

    stale_bounds = json.loads(json.dumps(row))
    stale_bounds["action_audit"]["executed_max"][0] = 2.0
    with pytest.raises(ValueError, match=r"executed actions exceed \[-1,1\]"):
        _validate_rollout_row(
            stale_bounds,
            expected_episode=0,
            episodes_per_task=20,
            max_steps=600,
        )

    stale_oob = json.loads(json.dumps(row))
    stale_oob["action_audit"]["raw_oob_row_count"] = 1
    with pytest.raises(ValueError, match="clipping counts are inconsistent"):
        _validate_rollout_row(
            stale_oob,
            expected_episode=0,
            episodes_per_task=20,
            max_steps=600,
        )

def test_libero_bddl_language_declaration_owns_instruction_membership(tmp_path: Path) -> None:
    from clearvla.benchmarks.libero import _bddl_instruction, _task_language_from_filename

    path = tmp_path / "SCENE1_filename_slug.bddl"
    path.write_text(
        '(define (problem x) (:language "put the red cube on the plate") (:domain robosuite))',
        encoding="utf-8",
    )
    assert _bddl_instruction(path) == "put the red cube on the plate"

    # Comments can split the official unquoted Lisp-style declaration across
    # lines.  Text after the comment remains part of the same language group.
    path.write_text(
        "(define (problem x) (:language put the red; object alias\n"
        "cube on the plate) (:domain robosuite))",
        encoding="utf-8",
    )
    assert _bddl_instruction(path) == "put the red cube on the plate"
    assert _task_language_from_filename(path) == "filename slug"

    path.write_text(
        '(define (problem x) (:language "put the red cube on the plate" ; alias\n) '
        '(:domain robosuite))',
        encoding="utf-8",
    )
    assert _bddl_instruction(path) == "put the red cube on the plate"


def test_libero_official_filename_language_is_distinct_from_bddl_alias(
    tmp_path: Path,
) -> None:
    from clearvla.benchmarks.libero import (
        _bddl_instruction,
        evaluator_instruction_inventory,
    )

    task = "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
    source = tmp_path / "raw"
    canonical = "pick up the black bowl on the cookie box and place it on the plate"
    _write_source_task(source, instruction=canonical, task_name=task)
    bddl = tmp_path / "bddl"
    suite = bddl / "libero_spatial"
    suite.mkdir(parents=True)
    bddl_path = suite / f"{task}.bddl"
    alias = "Pick the akita black bowl on the cookies box and place it on the plate"
    bddl_path.write_text(
        f"(define (problem x) (:language {alias}) (:domain robosuite))",
        encoding="utf-8",
    )

    inventory = evaluator_instruction_inventory(bddl, suites=("libero_spatial",))
    assert list(inventory.values()) == [canonical]
    assert _bddl_instruction(bddl_path) == alias
    output = tmp_path / "converted"
    convert_libero(
        source,
        output,
        suites=("libero_spatial",),
        evaluator_bddl_root=bddl,
    )
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluator_language_contract"].startswith(
        "official Task.language is filename-derived"
    )
    assert manifest["tasks"][0]["evaluator_instruction"] == canonical
    assert manifest["tasks"][0]["bddl_language_instruction"] == alias
    instructions = json.loads((output / "instructions.json").read_text(encoding="utf-8"))
    assert [row["instruction"] for row in instructions["instructions"]] == [canonical]


def test_libero_warmup_discards_success_flags() -> None:
    class Env:
        def __init__(self) -> None:
            self.checks = 0

        def step(self, action):
            return ({"ok": True}, 0.0, True, {})

        def check_success(self):
            self.checks += 1
            return True

    env = Env()
    observation = _step_observation_only(env, np.zeros(7, dtype=np.float32))
    assert observation == {"ok": True}
    assert env.checks == 0


def test_libero_evaluator_runs_policy_after_warmup_done_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warmup wrapper flag must not suppress the first formal action."""

    from clearvla.benchmarks import libero_eval

    observation = _observation()
    observation["agentview_image"] = np.zeros((128, 128, 3), dtype=np.uint8)
    observation["robot0_eye_in_hand_image"] = np.zeros(
        (128, 128, 3), dtype=np.uint8
    )

    class Client:
        instances: list["Client"] = []

        def __init__(self, endpoint: str, timeout: float) -> None:
            self.endpoint = endpoint
            self.timeout = timeout
            self.calls: list[tuple[object, str, bool]] = []
            self.__class__.instances.append(self)

        def health(self) -> dict[str, object]:
            return {"status": "ok", "mode": "smoke-zero"}

        def act(self, policy_input, instruction: str, *, reset: bool) -> np.ndarray:
            self.calls.append((policy_input, instruction, reset))
            return np.zeros((LIBERO_POLICY_HORIZON, 7), dtype=np.float32)

    class Env:
        instances: list["Env"] = []

        def __init__(self, **kwargs: object) -> None:
            self.steps = 0
            self.closed = False
            self.__class__.instances.append(self)

        def seed(self, seed: int) -> None:
            assert seed == 10000

        def reset(self) -> None:
            return None

        def set_init_state(self, state: np.ndarray) -> Mapping[str, object]:
            assert state.shape == (4,)
            return observation

        def step(self, action: np.ndarray):
            self.steps += 1
            assert action.shape == (7,)
            # Deliberately report done during both warmup and formal steps.
            return observation, 0.0, True, {}

        def check_success(self) -> bool:
            # The formal step is the first point at which success is queried.
            return self.steps >= 6

        def close(self) -> None:
            self.closed = True

    task = types.SimpleNamespace(name="task", language="do thing")
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text("(define (problem task))", encoding="utf-8")
    suite = types.SimpleNamespace(
        get_num_tasks=lambda: 1,
        get_task=lambda task_id: task,
        get_task_bddl_file_path=lambda task_id: str(bddl_path),
        get_task_init_states=lambda task_id: np.zeros(
            (LIBERO_OFFICIAL_EPISODES_PER_TASK, 4), dtype=np.float64
        ),
    )
    _install_fake_libero_runtime(monkeypatch, suite=suite, env_type=Env)
    monkeypatch.setattr(libero_eval, "RemotePolicyClient", Client)
    writes: list[dict[str, object]] = []
    real_atomic_json = libero_eval.atomic_json

    def recording_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
        writes.append(json.loads(json.dumps(payload)))
        real_atomic_json(path, payload)

    monkeypatch.setattr(libero_eval, "atomic_json", recording_atomic_json)

    result = libero_eval.evaluate_libero(
        "libero_spatial",
        tmp_path / "result.json",
        endpoint="http://127.0.0.1:8765",
        timeout=1.0,
        task_ids=None,
        episodes_per_task=LIBERO_OFFICIAL_EPISODES_PER_TASK,
        max_steps=LIBERO_OFFICIAL_MAX_STEPS,
        warmup_steps=LIBERO_OFFICIAL_WARMUP_STEPS,
        seed=10000,
        image_side=128,
        resume=False,
        allow_smoke_policy=True,
    )
    assert result["tasks"]["0"]["successes"] == LIBERO_OFFICIAL_EPISODES_PER_TASK
    assert result["tasks"]["0"]["rollouts"][0]["steps"] == 1
    assert [row["init_state"] for row in result["tasks"]["0"]["rollouts"]] == list(
        range(LIBERO_OFFICIAL_EPISODES_PER_TASK)
    )
    assert result["official_protocol"] is False
    assert result["complete"] is True
    assert result["completed_task_count"] == 1
    assert writes[0]["complete"] is False
    assert writes[0]["completed_task_count"] == 1
    assert writes[-1]["complete"] is True
    assert len(Client.instances) == 1
    assert len(Client.instances[0].calls) == LIBERO_OFFICIAL_EPISODES_PER_TASK
    assert Client.instances[0].calls[0][1:] == ("do thing", True)
    assert len(Env.instances) == 1
    assert Env.instances[0].steps == LIBERO_OFFICIAL_EPISODES_PER_TASK * (
        LIBERO_OFFICIAL_WARMUP_STEPS + 1
    )
    assert Env.instances[0].closed is True
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["tasks"]["0"]["successes"] == LIBERO_OFFICIAL_EPISODES_PER_TASK
    assert saved["complete"] is True

    resumed = libero_eval.evaluate_libero(
        "libero_spatial",
        tmp_path / "result.json",
        endpoint="http://127.0.0.1:8766",
        timeout=1.0,
        task_ids=None,
        episodes_per_task=LIBERO_OFFICIAL_EPISODES_PER_TASK,
        max_steps=LIBERO_OFFICIAL_MAX_STEPS,
        warmup_steps=LIBERO_OFFICIAL_WARMUP_STEPS,
        seed=10000,
        image_side=128,
        resume=True,
        allow_smoke_policy=True,
    )
    assert resumed["complete"] is True
    assert resumed["bridge_endpoint"] == "http://127.0.0.1:8766"
    assert len(Client.instances) == 2
    # A complete resume revalidates identities but never reruns a task.
    assert len(Env.instances) == 1

    with pytest.raises(ValueError, match="different evaluation protocol"):
        libero_eval.evaluate_libero(
            "libero_spatial",
            tmp_path / "result.json",
            endpoint="http://127.0.0.1:8767",
            timeout=1.0,
            task_ids=None,
            episodes_per_task=LIBERO_OFFICIAL_EPISODES_PER_TASK,
            max_steps=LIBERO_OFFICIAL_MAX_STEPS,
            warmup_steps=LIBERO_OFFICIAL_WARMUP_STEPS,
            seed=10000,
            image_side=64,
            resume=True,
            allow_smoke_policy=True,
        )

    suite.get_task_init_states = lambda task_id: np.ones(  # type: ignore[method-assign]
        (LIBERO_OFFICIAL_EPISODES_PER_TASK, 4), dtype=np.float64
    )
    with pytest.raises(ValueError, match="different evaluation protocol"):
        libero_eval.evaluate_libero(
            "libero_spatial",
            tmp_path / "result.json",
            endpoint="http://127.0.0.1:8768",
            timeout=1.0,
            task_ids=None,
            episodes_per_task=LIBERO_OFFICIAL_EPISODES_PER_TASK,
            max_steps=LIBERO_OFFICIAL_MAX_STEPS,
            warmup_steps=LIBERO_OFFICIAL_WARMUP_STEPS,
            seed=10000,
            image_side=128,
            resume=True,
            allow_smoke_policy=True,
        )


@pytest.mark.parametrize(
    ("failure_point", "error_type", "message", "expected_env_count"),
    (
        ("seed", RuntimeError, "seed failed", 1),
        ("init_states", RuntimeError, "init states failed", 0),
        ("too_few", ValueError, "fewer than the requested", 0),
    ),
)
def test_libero_evaluator_closes_environment_during_setup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    error_type: type[Exception],
    message: str,
    expected_env_count: int,
) -> None:
    from clearvla.benchmarks import libero_eval

    class Client:
        def __init__(self, endpoint: str, timeout: float) -> None:
            pass

        def health(self) -> dict[str, object]:
            return {"status": "ok", "mode": "smoke-zero"}

    class Env:
        instances: list["Env"] = []

        def __init__(self, **kwargs: object) -> None:
            self.closed = False
            self.__class__.instances.append(self)

        def seed(self, seed: int) -> None:
            if failure_point == "seed":
                raise RuntimeError("seed failed")

        def close(self) -> None:
            self.closed = True

    def init_states(task_id: int) -> np.ndarray:
        if failure_point == "init_states":
            raise RuntimeError("init states failed")
        count = 1 if failure_point == "too_few" else 2
        return np.zeros((count, 4), dtype=np.float64)

    task = types.SimpleNamespace(name="task", language="do thing")
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text("(define (problem task))", encoding="utf-8")
    suite = types.SimpleNamespace(
        get_num_tasks=lambda: 1,
        get_task=lambda task_id: task,
        get_task_bddl_file_path=lambda task_id: str(bddl_path),
        get_task_init_states=init_states,
    )
    _install_fake_libero_runtime(monkeypatch, suite=suite, env_type=Env)
    monkeypatch.setattr(libero_eval, "RemotePolicyClient", Client)

    with pytest.raises(error_type, match=message):
        libero_eval.evaluate_libero(
            "libero_spatial",
            tmp_path / "result.json",
            endpoint="http://127.0.0.1:8765",
            timeout=1.0,
            task_ids=None,
            episodes_per_task=2,
            max_steps=1,
            warmup_steps=0,
            seed=0,
            image_side=4,
            resume=False,
            allow_smoke_policy=True,
        )
    assert len(Env.instances) == expected_env_count
    assert all(instance.closed for instance in Env.instances)


def test_libero_deployment_abi_rejects_profile_or_dimension_drift() -> None:
    abi = _formal_libero_abi()
    assert validate_deployment_abi(abi)["action"] == abi["action"]

    stale_profile = json.loads(json.dumps(abi))
    stale_profile["action"]["data_profile"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="profile digest"):
        validate_deployment_abi(stale_profile)

    stale_bounds = json.loads(json.dumps(abi))
    stale_bounds["action"]["normalized_high"][0] = 2.0
    with pytest.raises(ValueError, match="normalized_high"):
        validate_deployment_abi(stale_bounds)

    stale_dims = json.loads(json.dumps(abi))
    stale_dims["observation"]["state_dim"] = 6
    with pytest.raises(ValueError, match="state_dim"):
        validate_deployment_abi(stale_dims)


def test_libero_health_requires_native_profile_digest_and_bounds() -> None:
    abi = _formal_libero_abi()
    health = {
        "status": "ok",
        "mode": "formal-checkpoint",
        "protocol": {"version": 2, "observe_only": True},
        "deployment": {
            "observation": abi["observation"],
            "action": abi["action"],
            "checkpoint": {"sha256": "a" * 64},
            "architecture": {
                "graph_config_sha256": "b" * 64,
                "manifest_digest": "c" * 64,
            },
        },
    }
    validate_libero_bridge_health(health)
    stale = json.loads(json.dumps(health))
    stale["deployment"]["action"]["normalized_low"][6] = -2.0
    with pytest.raises(ValueError, match="normalized_low"):
        validate_libero_bridge_health(stale)
    stale_normalizer = json.loads(json.dumps(health))
    stale_normalizer["deployment"]["action"]["normalizers"]["action_sha256"] = "d" * 64
    validate_libero_bridge_health(stale_normalizer)
    assert libero_bridge_identity(stale_normalizer) != libero_bridge_identity(health)


def test_libero_set_init_state_supports_mutating_official_api() -> None:
    from clearvla.benchmarks.libero_eval import _set_init_state_observation

    expected = _observation()

    class Env:
        def __init__(self) -> None:
            self.state = None

        def set_init_state(self, state):
            self.state = state
            return None

        def get_observation(self):
            return expected

    state = np.zeros(4, dtype=np.float64)
    assert _set_init_state_observation(Env(), state) is expected

    class PrivateEnv:
        def set_init_state(self, state):
            return None

        def _get_observation(self):
            return expected

    assert _set_init_state_observation(PrivateEnv(), state) is expected
