import json
from copy import deepcopy
from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import load_episode, load_episodes
from clearvla.data.split import load_episode_split_manifest
from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.components import OutletAdapter
from clearvla.mainline.runtime.deployment import (
    CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
    DEPLOYMENT_ABI_SCHEMA,
    canonical_sha256,
    deployment_graph_config,
    validate_deployment_abi,
)
from clearvla.rl.prepare_base import prepare
from clearvla.rl.train import training_seed
from clearvla.simulation.admission import (
    EXPERT_COLLECTOR,
    STACKCUBE_INSTRUCTION,
    STACKCUBE_PROFILE,
    STACKCUBE_REPAIRED_PROFILE,
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
from clearvla.simulation.maniskill_import import import_converted_episodes, recording_step


def descriptor():
    return EnvironmentDescriptor(
        backend="maniskill3",
        benchmark="ManiSkill3",
        task="StackCube-v1",
        robot="panda_wristcam",
        simulator_version="3.0.1",
        control_hz=20,
        max_episode_steps=200,
        state_semantics="[tcp xyz metres, tcp orientation rotation-vector radians, two-finger opening width metres]",
        action_semantics="ManiSkill normalized pd_ee_delta_pose: translation xyz, axis-angle rotation xyz, gripper target",
        action_state_semantics="previous clipped native 7D action",
        camera_semantics={
            "top": "StackCube base_camera external view",
            "wrist": "panda_wristcam hand_camera",
        },
    )


def observation(action, step):
    return PolicyObservation(
        rgb={name: np.full((4, 4, 3), step % 255, np.uint8) for name in ("top", "wrist")},
        state=np.full(7, 0.01 * step, np.float32),
        action_state=action.copy(),
    )


def experts(root, count=10, source_steps=80, hold=32):
    for seed in range(count):
        recorder = EpisodeRecorder(
            root,
            descriptor=descriptor(),
            episode_id=f"expert_{seed:06d}",
            seed=seed,
            instruction=STACKCUBE_INSTRUCTION,
            collector=EXPERT_COLLECTOR,
            provenance={
                "source_action_steps": source_steps,
                "post_success_hold_steps": hold,
                "hold_semantics": "real_env_steps_zero_ee_delta_hold_last_gripper",
            },
        )
        state = observation(np.zeros(7, np.float32), 0)
        for step in range(source_steps + hold):
            action = np.full(7, 0.01, np.float32)
            if step >= source_steps:
                action[:6] = 0
            next_obs = observation(action, step + 1)
            result = StepResult(
                next_obs,
                float(step >= source_steps - 1),
                False,
                step == source_steps + hold - 1,
                EvaluationState({"success": step >= source_steps - 1}),
            )
            recorder.append(state, action, result)
            state = next_obs
        recorder.finish()


def native_config(profile=STACKCUBE_PROFILE):
    cfg = ExperimentConfig()
    return replace(
        cfg,
        data=replace(cfg.data, data_profile=profile),
        bottom=replace(cfg.bottom, arm_flow_mode="relative_command_adapter"),
    )


def test_same_width_profiles_are_distinct_and_old_outlets_stay_selected():
    new = native_config()
    new.validate()
    profile = resolve_action_state_profile(STACKCUBE_PROFILE)
    assert profile.sampling_arm_motion == "relative_command_magnitude"
    assert profile.gripper_transition_boundary == "previous_command"
    assert profile.physical_chart.chart_kind == "maniskill_normalized_pd_ee_delta_pose_command"
    assert profile.digest() != resolve_action_state_profile("libero_relative_7d_v1").digest()
    assert ComponentSelection.from_config(new).outlet_adapter == "maniskill_7d_continuous_v1"
    assert (
        ComponentSelection.from_config(ExperimentConfig()).outlet_adapter == "pen_7d_continuous_v1"
    )
    wrong = replace(new, bottom=replace(new.bottom, arm_flow_mode="legacy_independent"))
    with pytest.raises(ValueError):
        wrong.validate()


def test_codec_unchanged_and_relative_w_chart_is_explicit():
    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    ms = OutletAdapter(codec, selection="maniskill_7d_continuous_v1")
    lb = OutletAdapter(codec, selection="libero_7d_continuous_v1")
    from types import SimpleNamespace

    normal = SimpleNamespace(offset=np.ones((1, 7)), scale=np.full((1, 7), 2.0))
    for outlet in (ms, lb):
        outlet.configure_action_normalizer(normal)
    action = torch.rand(2, 24, 7)
    state = torch.rand(2, 7)
    torch.testing.assert_close(ms.encode(action, state), lb.encode(action, state), rtol=0, atol=0)
    field = ms.encode(action, state)
    torch.testing.assert_close(ms.decode(field, state), lb.decode(field, state), rtol=0, atol=0)
    condition = ms.world_condition_from_horizon_action(torch.ones(2, 24, 7), state)
    assert torch.count_nonzero(condition.interval_action[..., :6]) == 0
    assert field.shape == (2, 24, 18)


def abi(profile_name=STACKCUBE_PROFILE):
    cfg = native_config(profile_name)
    profile = resolve_action_state_profile(profile_name)
    graph = deployment_graph_config(cfg)
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
            "dinov2": {"model": "test", "compute_dtype": "fp32", "reference_batch_size": 1},
        },
        "action": {
            "data_profile": {
                **profile.as_dict(),
                "sha256": profile.digest(),
                "gripper_transition_boundary": "previous_command",
            },
            "gripper_indices": [6],
            "gripper_output_mode": "continuous",
            "arm_flow_mode": "relative_command_adapter",
            "continuous_gripper_codec_boundary": "previous_command",
            "continuous_gripper_codec_boundary_scope": CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
            "receding_horizon_execute_rows": 1,
            "prediction_horizon": 24,
        },
        "normalizers": {"mode": "zscore", "action_sha256": "a" * 64, "state_sha256": "b" * 64},
        "language": {"sha256": "c" * 64},
    }


@pytest.mark.parametrize("profile_name", [STACKCUBE_PROFILE, STACKCUBE_REPAIRED_PROFILE])
def test_deployment_rejects_relabelled_or_recharted_native_profile(profile_name):
    value = abi(profile_name)
    validate_deployment_abi(value)
    validate_deployment_abi(json.loads(json.dumps(value)))
    for field, bad in (
        ("state_chart", "libero"),
        ("action_indices", [0] * 7),
        ("sha256", "0" * 64),
        ("gripper_transition_boundary", "current_action_state"),
    ):
        altered = deepcopy(value)
        altered["action"]["data_profile"][field] = bad
        with pytest.raises(ValueError):
            validate_deployment_abi(altered)


def test_pretrain_preparation_native_split_and_terminal_target_coverage(tmp_path):
    root, output = tmp_path / "experts", tmp_path / "prepared"
    experts(root)
    audit = audit_stackcube_experts(root)
    assert audit["episodes"] == 10 and audit["steps"] == 1120
    summary = prepare(
        root,
        output,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
        profile=STACKCUBE_PROFILE,
    )
    assert summary["split_counts"] == {"train": 8, "val": 1, "test": 1}
    cfg = load_config(output / "config.json")
    assert cfg.data.data_profile == STACKCUBE_PROFILE
    assert cfg.optimizer.epochs == 8 and cfg.optimizer.batch_size == 8
    files = sorted(root.glob("*.hdf5"))
    # HDF5 identities are stem-relative, not filesystem filenames with suffix.
    raw_split = json.loads((output / "splits.json").read_text())
    actual, skipped = load_episodes(
        root,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=73,
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides=cfg.data.camera_key_map(),
        episode_names=sum(raw_split["splits"].values(), []),
    )
    assert len(actual) == 10 and not skipped
    load_episode_split_manifest(
        output / "splits.json", episode_names=[ep.episode_id for ep in actual]
    )
    loaded = load_episode(
        files[0],
        cameras=("top", "wrist"),
        action_key="action",
        state_key="state",
        action_state_key="action_state",
        camera_key_overrides={
            "top": "observations/images/cam_high",
            "wrist": "observations/images/cam_right_wrist",
        },
    )
    projected = resolve_action_state_profile(STACKCUBE_PROFILE).project_episode(loaded)
    assert projected.data_profile == STACKCUBE_PROFILE
    # Original last expert action 79 is covered; without real hold, max
    # supervised row would be (80-49)+23=54 and the release tail disappears.
    assert projected.length - 49 + 23 >= 79
    with pytest.raises(FileExistsError):
        prepare(
            root,
            output,
            cache_root=tmp_path / "cache",
            language_bank=tmp_path / "l.pt",
            run_root=tmp_path / "r",
            profile=STACKCUBE_PROFILE,
        )


@pytest.mark.parametrize(
    "mutation", ["smoke", "missing_provenance", "short_hold", "duplicate_seed", "bad_action_state"]
)
def test_expert_admission_fail_closed(tmp_path, mutation):
    root = tmp_path / "data"
    experts(root, count=2)
    first = root / "expert_000000.hdf5"
    with h5py.File(first, "a") as stream:
        if mutation == "smoke":
            stream.attrs["collector"] = "EnvironmentRandomPolicy"
        elif mutation == "missing_provenance":
            del stream.attrs["provenance_json"]
        elif mutation == "short_hold":
            provenance = json.loads(stream.attrs["provenance_json"])
            provenance["post_success_hold_steps"] = 2
            stream.attrs["provenance_json"] = json.dumps(provenance)
        elif mutation == "duplicate_seed":
            stream.attrs["seed"] = 1
        else:
            stream["action_state"][1] = np.ones(7)
    with pytest.raises(ValueError):
        audit_stackcube_experts(root)


def test_training_seeds_resume_without_reusing_holdouts():
    excluded = {0, 1, 3, 7, 8}
    actual = [training_seed(0, i, excluded) for i in range(10)]
    assert actual == [2, 4, 5, 6, 9, 10, 11, 12, 13, 14]


class ReplayEnv:
    descriptor = descriptor()

    def reset(self, *, seed):
        self.step_count = 0
        return ResetResult(observation(np.zeros(7, np.float32), 0))

    def clip_action(self, action):
        return np.asarray(action, np.float32).clip(-1, 1)

    def hold_action(self, obs):
        result = np.zeros(7, np.float32)
        result[-1] = obs.action_state[-1]
        return result

    def step(self, action):
        self.step_count += 1
        return StepResult(
            observation(action, self.step_count),
            1.0,
            False,
            False,
            EvaluationState({"success": self.step_count >= 80}),
        )


def test_importer_records_real_tail_and_does_not_fake_future_images(tmp_path):
    source = tmp_path / "trajectory.h5"
    with h5py.File(source, "w") as stream:
        stream.create_dataset("traj_0/actions", data=np.full((80, 7), 0.01, np.float32))
    meta = tmp_path / "trajectory.json"
    meta.write_text(
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
    class NativeSuccessTerminatingEnv(ReplayEnv):
        def step(self, action):
            result = super().step(action)
            return replace(result, terminated=bool(result.evaluation.metrics["success"]))

    env = NativeSuccessTerminatingEnv()
    result = import_converted_episodes(
        env,
        trajectory_path=source,
        metadata_path=meta,
        output_dir=tmp_path / "import",
        start_index=0,
        count=1,
        settle_steps=32,
        require_all_success=True,
    )
    assert env.step_count == 112 and result["accepted"][0]["steps"] == 112
    assert audit_stackcube_experts(tmp_path / "import")["episodes"] == 1
    with h5py.File(tmp_path / "import" / "expert_000000.hdf5", "r") as stream:
        assert stream["observations/images/cam_high"][111, 0, 0, 0] == 111
        assert json.loads(stream["sim/metrics_json"][-1])["native_terminated"] is True
        assert not np.asarray(stream["sim/terminated"]).any()


def test_recording_success_continuation_never_changes_rl_failure_or_timeout():
    obs = observation(np.zeros(7, np.float32), 1)
    result = StepResult(obs, 1.0, True, False, EvaluationState({"success": True}))
    assert recording_step(result, continue_success=False) is result
    assert recording_step(result, continue_success=True).terminated is False
    failed = replace(result, evaluation=EvaluationState({"success": True, "fail": True}))
    timed_out = replace(result, truncated=True)
    assert recording_step(failed, continue_success=True) is failed
    assert recording_step(timed_out, continue_success=True) is timed_out
