import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import LoadedEpisode, load_episode
from clearvla.data.samplers import InformationBalancedBatchSampler, InformationBalancedSamplerConfig
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.runtime.deployment import _validate_maniskill_profile
from clearvla.mainline.runtime.evaluation import (
    MatchedCoreAttributionAccumulator,
    MatchedP2InterventionAccumulator,
    ValidationAccumulator,
)
from clearvla.rl.features import FrozenBaseReader
from clearvla.simulation.admission import (
    EXPERT_COLLECTOR,
    STACKCUBE_INSTRUCTION,
    STACKCUBE_PROFILE,
    STACKCUBE_REPAIRED_PROFILE,
    STACKCUBE_WINDOW_CONTRACT,
    audit_stackcube_experts,
)
from clearvla.simulation.contracts import (
    EnvironmentDescriptor,
    EvaluationState,
    PolicyObservation,
    ResetResult,
    StepResult,
)
from clearvla.simulation.dataset import EpisodeRecorder
from clearvla.simulation.history import CausalHistory
from clearvla.simulation.maniskill_adapter import _controller_telemetry, _physics_telemetry
from clearvla.simulation.state_chart import (
    CONTINUOUS_STATE_CHART,
    CONTINUOUS_STATE_SEMANTICS,
    CausalRotationChart,
    principal_rotvec,
)


def test_stackcube_rollout_defaults_to_repaired_causal_rotation_chart(monkeypatch):
    from clearvla.simulation.rollout import parse_args

    monkeypatch.setattr(sys, "argv", ["rollout", "--environment", "maniskill-stackcube"])
    args = parse_args()
    assert args.maniskill_state_chart == CONTINUOUS_STATE_CHART


def test_direct_stackcube_environment_default_uses_repaired_chart():
    import inspect

    from clearvla.simulation.maniskill_adapter import ManiSkillStackCubeEnv

    assert (
        inspect.signature(ManiSkillStackCubeEnv).parameters["state_chart"].default
        == CONTINUOUS_STATE_CHART
    )


def quaternion(rotvec):
    angle = np.linalg.norm(rotvec)
    return np.r_[np.cos(angle / 2), np.asarray(rotvec) * np.sin(angle / 2) / max(angle, 1e-12)]


def restore_world_quaternion(local):
    w, x, y, z = quaternion(local)
    return np.array([-x, w, -z, y])  # [0,1,0,0] * local


def test_rotation_chart_removes_down_pose_cut_preserves_rotation_and_causality():
    path = [quaternion([x, 0.002, 0]) for x in np.linspace(np.pi - 0.04, np.pi + 0.04, 81)]
    legacy = np.stack([principal_rotvec(q) for q in path])
    assert np.max(np.linalg.norm(np.diff(legacy, axis=0), axis=1)) > 6
    chart = CausalRotationChart()
    current = np.stack([chart.encode(q if i % 2 else -q) for i, q in enumerate(path)])
    assert np.max(np.linalg.norm(np.diff(current, axis=0), axis=1)) < 0.002
    for value, q in zip(current, path, strict=True):
        assert abs(np.dot(restore_world_quaternion(value), q)) > 1 - 1e-7
    prefix = CausalRotationChart()
    np.testing.assert_array_equal(current[:20], np.stack([prefix.encode(q) for q in path[:20]]))
    chart.reset()
    np.testing.assert_array_equal(chart.encode(path[0]), current[0])


def test_causal_rotation_branch_crossing_and_invalid_quaternion():
    chart = CausalRotationChart()
    encoded = []
    for angle in np.linspace(0, 4 * np.pi, 241):
        local_q = quaternion(np.array([0, 0, angle]))
        w, x, y, z = local_q
        encoded.append(chart.encode(np.array([-x, w, -z, y])))
    np.testing.assert_allclose(np.array(encoded)[:, 2], np.linspace(0, 4 * np.pi, 241), atol=2e-6)
    with pytest.raises(ValueError):
        chart.encode(np.zeros(4))
    with pytest.raises(ValueError):
        chart.encode(np.full(4, np.nan))


class CountingSampler(InformationBalancedBatchSampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.event_counts = []

    def _choice(self, rng, pool, count, selected, fallback):
        if pool is self.event_indices:
            self.event_counts.append(count)
        return super()._choice(rng, pool, count, selected, fallback)


@pytest.mark.parametrize("batch", [2, 4, 8, 16])
def test_fractional_events_preserve_epoch_mass_and_integral_counts(batch):
    config = InformationBalancedSamplerConfig(batch_size=batch, seed=19)
    sampler = CountingSampler(np.linspace(0, 1, 128), np.arange(128) % 5 == 0, config)
    actual = list(sampler)
    assert sum(sampler.event_counts) == int(len(actual) * batch * 0.125)
    assert all(len(row) == len(set(row)) == batch for row in actual)
    assert sampler.summary["effective_dedicated_event_fraction"] == 0.125
    if batch == 4:
        assert sampler.event_counts == [0, 1] * (len(actual) // 2)
    if batch == 8:
        assert set(sampler.event_counts) == {1}
        # Captured from the pre-repair sampler at workspace HEAD, seed19.
        assert hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest() == (
            "b1338c240fe673ddce09758f484a246bc7e6b622d7f1e4e36f37906574ae1ba0"
        )
    other = CountingSampler(sampler.motion_score, sampler.is_event, config)
    assert actual == list(other)
    sampler.set_epoch(1)
    assert actual != list(sampler)


class Images:
    def validate_episode(self, episode):
        pass

    def load_window(self, episode, indices):
        return {
            name: torch.stack([torch.full((3, 4, 4), float(i)) for i in indices])
            for name in ("top", "wrist")
        }


def episode(source=80, hold=48):
    length = source + hold
    actions = np.zeros((length, 7), np.float32)
    actions[:source, :6] = np.arange(source)[:, None] / 100
    actions[:, -1] = 1
    actions[30 : source - 6, -1] = -1
    reset_action = np.r_[np.zeros(6), 1].astype(np.float32)
    states = np.repeat(np.arange(length, dtype=np.float32)[:, None] / 100, 7, axis=1)
    boundary = np.concatenate((reset_action[None], actions[:-1]), axis=0)
    return LoadedEpisode(
        path=Path("fixture.hdf5"),
        episode_id="fixture",
        source_partition="",
        task_id="",
        action_key="action",
        state_key="state",
        action_state_key="action_state",
        camera_keys={},
        actions_raw=actions,
        states_raw=states,
        action_states_raw=boundary,
        valid_center_start=0,
        valid_center_end=source - 1,
        data_profile=STACKCUBE_REPAIRED_PROFILE,
    )


def dataset(ep, padding=True):
    norm = ArrayNormalizer.fit_identity([ep.actions_raw])
    return ObservedStateWindowDataset(
        [ep],
        [0],
        image_store=Images(),
        camera_names=("top", "wrist"),
        state_normalizer=norm,
        action_normalizer=norm,
        config=ObservedStateDatasetConfig(causal_reset_padding=padding),
        gripper_transition_boundary="previous_command",
    )


def observation(ep, i):
    return PolicyObservation(
        rgb={name: np.full((4, 4, 3), i, np.uint8) for name in ("top", "wrist")},
        state=ep.states_raw[i],
        action_state=ep.action_states_raw[i],
    )


def test_all_source_centers_include_approach_grasp_release_as_first_row():
    ep = episode()
    ds = dataset(ep)
    assert [ref.center for ref in ds.refs] == list(range(80))
    for center in (0, 1, 30, 74, 79):
        sample = ds[center]
        np.testing.assert_array_equal(sample["policy_action_raw"][0], ep.actions_raw[center])
        assert sample["future_keys"][-1, 1] == center + 48
    assert ds[74]["gripper_transition_boundary_raw"][-1] == -1
    assert ds[74]["policy_action_raw"][0, -1] == 1
    motion, events = ds.training_information_signals(
        gripper_indices=(6,), event_threshold=0.1, arm_motion="relative_command_magnitude"
    )
    assert np.isfinite(motion).all() and events[30] and events[74]
    from scripts.analyze_mainline_action_baselines import _raw_windows

    target, boundary, previous, _ = _raw_windows(ds)
    np.testing.assert_array_equal(previous[0], ep.action_states_raw[0])
    np.testing.assert_array_equal(target[0, 0], ep.actions_raw[0])
    np.testing.assert_array_equal(boundary[0], ep.action_states_raw[0])


def test_offline_reset_padding_equals_online_history_without_future_action_leak():
    ep = episode()
    ds = dataset(ep)
    history = CausalHistory()
    history.reset(observation(ep, 0))
    for center in range(25):
        actual = ds[center]
        online = history.snapshot()
        np.testing.assert_array_equal(actual["history_state"], online.state_history)
        np.testing.assert_array_equal(
            actual["executed_action_history"], online.executed_action_history
        )
        for index, name in enumerate(("top", "wrist")):
            expected = torch.from_numpy(online.rgb_history[name]).permute(0, 3, 1, 2).float() / 255
            torch.testing.assert_close(
                actual["history_obs_image"][:, index], expected, rtol=0, atol=0
            )
        if center < 24:
            history.append(ep.actions_raw[center], observation(ep, center + 1))
    before = ds[0]
    ep.actions_raw[0] = 0.8
    after = ds[0]
    torch.testing.assert_close(
        before["executed_action_history"], after["executed_action_history"], rtol=0, atol=0
    )
    assert not torch.equal(before["policy_action_raw"], after["policy_action_raw"])


def test_legacy_window_bounds_and_minimum_length_stay_unchanged():
    ep = episode(80, 32)
    assert ObservedStateDatasetConfig().minimum_episode_length == 73
    assert ObservedStateDatasetConfig(causal_reset_padding=True).minimum_episode_length == 49
    assert [r.center for r in dataset(ep, padding=False).refs] == list(range(24, 64))


def descriptor():
    return EnvironmentDescriptor(
        backend="maniskill3",
        benchmark="ManiSkill3",
        task="StackCube-v1",
        robot="panda_wristcam",
        simulator_version="3.0.1",
        state_semantics=CONTINUOUS_STATE_SEMANTICS,
        action_semantics="ManiSkill normalized pd_ee_delta_pose: translation xyz, axis-angle rotation xyz, gripper target",
        action_state_semantics="previous clipped native 7D action",
        camera_semantics={
            "top": "StackCube base_camera external view",
            "wrist": "panda_wristcam hand_camera",
        },
        control_hz=20,
        max_episode_steps=400,
    )


def record(root, hold=48, seed=0):
    ep = episode(80, hold)
    recorder = EpisodeRecorder(
        root,
        descriptor=descriptor(),
        episode_id=f"fixture_{seed:02d}",
        seed=seed,
        instruction=STACKCUBE_INSTRUCTION,
        collector=EXPERT_COLLECTOR,
        valid_center_bounds=(0, 79),
        provenance={
            "source_action_steps": 80,
            "post_success_hold_steps": hold,
            "state_chart": CONTINUOUS_STATE_CHART,
            "window_contract": STACKCUBE_WINDOW_CONTRACT,
            "hold_semantics": "real_env_steps_zero_ee_delta_hold_last_gripper",
        },
    )
    for step in range(len(ep.actions_raw)):
        result = StepResult(
            observation(ep, min(step + 1, len(ep.actions_raw) - 1)),
            1.0,
            False,
            False,
            EvaluationState({"success": step >= 74}),
        )
        recorder.append(observation(ep, step), ep.actions_raw[step], result)
    return recorder.finish().path


def test_recorder_persists_evaluator_physics_telemetry_without_policy_leak(tmp_path):
    root = tmp_path / "telemetry"
    ep = episode(80, 48)
    recorder = EpisodeRecorder(
        root,
        descriptor=descriptor(),
        episode_id="telemetry_00",
        seed=123,
        instruction=STACKCUBE_INSTRUCTION,
        collector="diagnostic",
    )
    metrics = {
        "success": False,
        "telemetry_tcp_pose": [0.0] * 7,
        "telemetry_tcp_linear_velocity": [0.1, 0.2, 0.3],
        "telemetry_cube_a_pose": [1.0] * 7,
        "telemetry_cube_a_linear_velocity": [0.0, 0.0, 0.0],
        "telemetry_cube_a_contact_force": [0.0, 0.0, 0.6],
        "telemetry_cube_a_finger1_contact_force": [0.0, 0.0, 0.0],
        "telemetry_gripper_opening": 0.08,
        "telemetry_gripper_finger_qpos": [0.04, 0.04],
        "telemetry_contact_count": 3,
        "telemetry_executed_action": [0.0] * 6 + [1.0],
    }
    result = StepResult(
        observation(ep, 1),
        0.0,
        False,
        False,
        EvaluationState(metrics),
    )
    recorder.append(observation(ep, 0), ep.actions_raw[0], result)
    path = recorder.finish().path
    with h5py.File(path, "r") as stream:
        saved = json.loads(stream["sim/metrics_json"][0])
        assert "telemetry" not in stream
    assert saved["telemetry_tcp_pose"] == [0.0] * 7
    assert saved["telemetry_cube_a_contact_force"] == [0.0, 0.0, 0.6]
    assert saved["telemetry_gripper_opening"] == pytest.approx(0.08)
    assert saved["telemetry_executed_action"][-1] == pytest.approx(1.0)


def test_v2_admission_requires_all_source_first_rows_and_48_real_future_steps(tmp_path):
    root = tmp_path / "good"
    file = record(root)
    audit = audit_stackcube_experts(root, profile=STACKCUBE_REPAIRED_PROFILE)
    assert audit["expert_admission"] == "stackcube_all_source_first_action_v2"
    with pytest.raises(ValueError):
        audit_stackcube_experts(root, profile=STACKCUBE_PROFILE)
    short = tmp_path / "short"
    record(short, 32)
    with pytest.raises(ValueError, match="support"):
        audit_stackcube_experts(short, profile=STACKCUBE_REPAIRED_PROFILE)
    with h5py.File(file, "a") as h:
        h.attrs["valid_center_start"] = 24
    with pytest.raises(ValueError, match="first-executed"):
        audit_stackcube_experts(root, profile=STACKCUBE_REPAIRED_PROFILE)


def test_v2_preparation_defaults_repaired_and_retains_only_required_json(tmp_path):
    from clearvla.mainline.config import load_config
    from clearvla.rl.prepare_base import prepare

    root = tmp_path / "experts"
    for seed in range(10):
        record(root, seed=seed)
    prepared = tmp_path / "prepared"
    summary = prepare(
        root,
        prepared,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
    )
    assert summary["dataset"]["expert_admission"] == "stackcube_all_source_first_action_v2"
    assert load_config(prepared / "config.json").data.data_profile == STACKCUBE_REPAIRED_PROFILE
    assert {p.name for p in prepared.iterdir()} == {
        "config.json",
        "splits.json",
        "instructions.json",
    }


def test_v2_import_preserves_first_row_bounds_and_no_report_dump(tmp_path):
    from clearvla.simulation.contracts import ResetResult
    from clearvla.simulation.maniskill_import import import_converted_episodes

    ep = episode()
    source = tmp_path / "trajectory.h5"
    with h5py.File(source, "w") as stream:
        stream.create_dataset("traj_0/actions", data=ep.actions_raw[:80])
    metadata = tmp_path / "trajectory.json"
    metadata.write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "StackCube-v1",
                    "env_kwargs": {"control_mode": "pd_ee_delta_pose"},
                },
                "episodes": [
                    {
                        "episode_id": 0,
                        "episode_seed": 0,
                        "elapsed_steps": 80,
                        "success": True,
                        "control_mode": "pd_ee_delta_pose",
                    }
                ],
            }
        )
    )

    class Env:
        descriptor = descriptor()

        def reset(self, *, seed):
            self.index = 0
            return ResetResult(observation(ep, 0))

        def clip_action(self, value):
            return np.asarray(value, np.float32).clip(-1, 1)

        def hold_action(self, obs):
            return np.r_[np.zeros(6), obs.action_state[-1]].astype(np.float32)

        def step(self, action):
            self.index += 1
            obs = observation(ep, min(self.index, 127))
            obs = replace(obs, action_state=action.copy())
            return StepResult(
                obs, 1.0, self.index >= 75, False, EvaluationState({"success": self.index >= 75})
            )

    root = tmp_path / "recorded"
    report = import_converted_episodes(
        Env(),
        trajectory_path=source,
        metadata_path=metadata,
        output_dir=root,
        start_index=0,
        count=1,
        settle_steps=48,
        require_all_success=True,
    )
    assert len(report["accepted"]) == 1
    assert not list(root.glob("import_*.json"))
    loaded = load_episode(
        root / "expert_000000.hdf5",
        cameras=("top", "wrist"),
        state_key="state",
        action_state_key="action_state",
        camera_key_overrides={
            "top": "observations/images/cam_high",
            "wrist": "observations/images/cam_right_wrist",
        },
    )
    assert (loaded.valid_center_start, loaded.valid_center_end) == (0, 79)
    audit_stackcube_experts(root, profile=STACKCUBE_REPAIRED_PROFILE)


def test_new_profile_and_component_are_distinct_without_changing_core_dimensions():
    profile = resolve_action_state_profile(STACKCUBE_REPAIRED_PROFILE)
    old = resolve_action_state_profile(STACKCUBE_PROFILE)
    assert profile.digest() != old.digest()
    assert profile.output_dim == 7 and profile.physical_chart.output_dim == 7
    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        data=replace(cfg.data, data_profile=profile.name),
        bottom=replace(cfg.bottom, arm_flow_mode="relative_command_adapter"),
    )
    cfg.validate()
    with pytest.raises(ValueError, match="stride=1"):
        replace(cfg, data=replace(cfg.data, stride=2)).validate()
    assert ComponentSelection.from_config(cfg).outlet_adapter == "maniskill_7d_continuous_v2"
    meta = {
        **profile.as_dict(),
        "sha256": profile.digest(),
        "gripper_transition_boundary": "previous_command",
    }
    _validate_maniskill_profile(meta)
    meta["state_chart"] = old.state_chart
    with pytest.raises(ValueError):
        _validate_maniskill_profile(meta)
    old_cfg = replace(cfg, data=replace(cfg.data, data_profile=old.name))
    with pytest.raises(ValueError, match="repaired v2"):
        FrozenBaseReader(SimpleNamespace(bundle=SimpleNamespace(config=old_cfg)))


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("maniskill_pd_ee_delta_pose_7d_v1", 1),
        ("maniskill_pd_ee_delta_pose_7d_v2", 1),
        ("calvin_relative_7d_v1", 1),
        ("libero_relative_7d_v1", -1),
        ("identity_7d_pen", -1),
        ("rdt_right_arm_action_chart_v1", -1),
    ],
)
def test_action_profile_owns_native_gripper_open_direction(profile, expected):
    assert resolve_action_state_profile(profile).gripper_open_direction == expected


def test_evaluator_telemetry_keeps_physics_outside_policy_observation():
    class Pose:
        def __init__(self, values):
            self.raw_pose = np.asarray(values, dtype=np.float32)

    class Actor:
        def __init__(self, offset):
            self.pose = Pose(np.arange(7, dtype=np.float32) + offset)

        def get_linear_velocity(self):
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        def get_angular_velocity(self):
            return np.asarray([4.0, 5.0, 6.0], dtype=np.float32)

        def get_net_contact_forces(self):
            return np.asarray([0.0, 0.0, 7.0], dtype=np.float32)

        def get_net_contact_impulses(self):
            return np.asarray([0.0, 0.0, 0.5], dtype=np.float32)

    class Scene:
        def get_pairwise_contact_forces(self, first, second):
            return np.asarray([8.0, 0.0, 0.0], dtype=np.float32)

        def get_contacts(self):
            return [object(), object()]

    class Robot:
        def get_qpos(self):
            return np.asarray([0.0, 0.0, 0.03, 0.04], dtype=np.float32)

        def get_qvel(self):
            return np.asarray([0.0, 0.0, 0.1, 0.2], dtype=np.float32)

    agent = SimpleNamespace(
        tcp=Actor(0.0),
        finger1_link=object(),
        finger2_link=object(),
        robot=Robot(),
    )
    native = SimpleNamespace(
        agent=agent,
        cubeA=Actor(10.0),
        cubeB=Actor(20.0),
        scene=Scene(),
    )
    telemetry = _physics_telemetry(
        native,
        {"extra": {"tcp_pose": np.zeros(7, dtype=np.float32)}},
        np.ones(7, dtype=np.float32),
    )
    assert telemetry["telemetry_tcp_pose"] == list(np.arange(7, dtype=float))
    assert telemetry["telemetry_cube_a_pose"] == list(np.arange(7, dtype=float) + 10)
    assert telemetry["telemetry_cube_b_linear_velocity"] == [1.0, 2.0, 3.0]
    assert telemetry["telemetry_cube_a_contact_force"] == [0.0, 0.0, 7.0]
    assert telemetry["telemetry_cube_a_finger1_contact_force"] == [8.0, 0.0, 0.0]
    assert telemetry["telemetry_gripper_opening"] == pytest.approx(0.07)
    assert telemetry["telemetry_contact_count"] == 2
    assert telemetry["telemetry_executed_action"] == [1.0] * 7


def test_controller_telemetry_marks_cpu_ik_fallback_only_for_nonzero_command():
    class Controller:
        _start_qpos = np.zeros((1, 7), dtype=np.float32)
        _target_qpos = np.zeros((1, 7), dtype=np.float32)

    class Agent:
        controller = SimpleNamespace(controllers={"arm": Controller()})

    native = SimpleNamespace(agent=Agent())
    failed = _controller_telemetry(native, np.r_[np.ones(6), 1.0])
    assert failed["telemetry_ik_fallback"] is True
    assert failed["telemetry_controller_arm_command_norm"] == pytest.approx(np.sqrt(6.0))
    held = _controller_telemetry(native, np.r_[np.zeros(6), 1.0])
    assert held["telemetry_ik_fallback"] is False

    Controller._target_qpos = np.ones((1, 7), dtype=np.float32)
    solved = _controller_telemetry(native, np.r_[np.ones(6), 1.0])
    assert solved["telemetry_ik_fallback"] is False


def test_rollout_trace_records_full_chunk_and_reuses_initial_reset(tmp_path):
    from clearvla.simulation.rollout import run_episode

    class Env:
        descriptor = descriptor()

        def __init__(self):
            self.reset_calls = 0
            self.step_calls = 0

        def _obs(self, action_state):
            return PolicyObservation(
                rgb={name: np.zeros((4, 4, 3), np.uint8) for name in ("top", "wrist")},
                state=np.zeros(7, np.float32),
                action_state=np.asarray(action_state, np.float32).copy(),
            )

        def reset(self, *, seed):
            self.reset_calls += 1
            return ResetResult(self._obs(np.r_[np.zeros(6), 1.0]))

        def hold_action(self, observation):
            return np.r_[np.zeros(6), observation.action_state[-1]].astype(np.float32)

        def clip_action(self, action):
            return np.clip(np.asarray(action, np.float32), -1, 1)

        def action_bounds(self):
            return -np.ones(7, np.float32), np.ones(7, np.float32)

        def sample_action(self, rng):
            return np.zeros(7, np.float32)

        def step(self, action):
            self.step_calls += 1
            value = self.clip_action(action)
            return StepResult(
                self._obs(value),
                0.0,
                False,
                self.step_calls >= 2,
                EvaluationState({"success": False}),
            )

        def close(self):
            pass

    class Policy:
        def reset(self):
            pass

        def act(self, history, instruction):
            history.validate()
            chunk = np.zeros((24, 7), np.float32)
            chunk[0, 0] = 2.0  # row zero is clipped at the environment boundary
            chunk[1, -1] = -0.5  # retained only as a future proposal in this step
            return chunk

    env = Env()
    initial = env.reset(seed=7)
    report = run_episode(
        env,
        Policy(),
        seed=7,
        instruction=STACKCUBE_INSTRUCTION,
        max_steps=2,
        record_dir=tmp_path,
        episode_id="trace_00",
        initial_reset=initial,
    )
    assert env.reset_calls == 1
    assert report["recorded_episode"]["policy_trace"] is True
    with h5py.File(tmp_path / "trace_00.hdf5", "r") as stream:
        rows = [json.loads(value) for value in stream["sim/metrics_json"]]
    assert len(rows) == 2
    assert rows[0]["telemetry_policy_trace_schema"] == "receding_horizon_policy_trace_v1"
    assert np.asarray(rows[0]["telemetry_policy_raw_action_chunk"]).shape == (24, 7)
    assert rows[0]["telemetry_policy_raw_action"] == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert rows[0]["telemetry_policy_raw_action_oob_count"] == 1
    assert rows[0]["telemetry_policy_step"] == 0
    assert rows[1]["telemetry_policy_step"] == 1


def test_event_direction_changes_only_open_close_names_and_propagates_to_matched():
    norm = ArrayNormalizer.fit_identity([np.zeros((2, 7), np.float32)])
    target = torch.ones(1, 24, 7)
    target[..., :6] = 0
    boundary = torch.zeros(1, 7)
    boundary[:, -1] = -1
    history = SimpleNamespace(action_state=boundary, codec_gripper_boundary=boundary[:, -1:])
    action = SimpleNamespace(
        row_valid=None,
        normalized=target,
        raw_units=target,
        current_raw_units=boundary,
        gripper_transition_boundary_raw_units=boundary,
    )
    batch = SimpleNamespace(online=SimpleNamespace(history=history), action_target=action)
    results = []
    for sign in (-1, 1):
        accumulator = ValidationAccumulator.from_action_normalizer(
            norm, device=torch.device("cpu"), gripper_open_direction=sign
        )
        accumulator.update(target, batch)
        results.append(accumulator.means())
    assert (
        results[0]["validation_decoded_gripper_close_f1"]
        == results[1]["validation_decoded_gripper_open_f1"]
        == 1
    )
    assert (
        results[0]["validation_decoded_gripper_event_f1"]
        == results[1]["validation_decoded_gripper_event_f1"]
        == 1
    )
    assert (
        results[0]["validation_decoded_gripper_first_event_f1"]
        == results[1]["validation_decoded_gripper_first_event_f1"]
        == 1
    )
    assert (
        results[0]["validation_decoded_gripper_first_hold_f1"]
        == results[1]["validation_decoded_gripper_first_hold_f1"]
        == 0
    )
    assert (
        results[0]["validation_action_rmse_normalized"]
        == results[1]["validation_action_rmse_normalized"]
        == 0
    )
    for cls in (MatchedP2InterventionAccumulator, MatchedCoreAttributionAccumulator):
        acc = cls.from_action_normalizer(
            norm,
            device=torch.device("cpu"),
            gripper_event_threshold=0.1,
            arm_motion_threshold=0.02,
            gripper_open_direction=1,
        )
        assert acc._new_validation_accumulator().gripper_open_direction == 1
