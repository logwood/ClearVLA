from types import SimpleNamespace
import numpy as np
import pytest

from clearvla.simulation.contracts import (
    PolicyObservation, EvaluationState, ResetResult, StepResult, ExecutedCommand,
)
from scripts.record_maniskill_npz import record_episode, scene_row, REFERENCE_KEYS


class FakeEnv:
    descriptor = SimpleNamespace(control_hz=20)
    def __init__(self):
        self._env = SimpleNamespace(unwrapped=SimpleNamespace(
            agent=SimpleNamespace(robot=SimpleNamespace(
                get_qpos=lambda: np.zeros((1, 9), dtype=np.float32)))))
        self.t = 0

    def observation(self, action):
        return PolicyObservation(
            rgb={k: np.zeros((32, 32, 3), np.uint8) for k in ("top", "wrist")},
            state=np.array([self.t, 0, 0, 0, 0, 0, .04], np.float32),
            action_state=action.copy())

    def metrics(self):
        return {"success": self.t >= 2, "is_cubeA_on_cubeB": self.t >= 2,
                "telemetry_cube_a_pose": [0, 0, .02, 1, 0, 0, 0],
                "telemetry_cube_b_pose": [.1, 0, .02, 1, 0, 0, 0],
                **{f"telemetry_cube_{c}_finger{i}_contact_force": [0, 0, 0]
                   for c in ("a", "b") for i in (1, 2)}}

    def reset(self, *, seed):
        self.t = 0
        return ResetResult(self.observation(np.array([0, 0, 0, 0, 0, 0, 1], np.float32)),
                           EvaluationState(self.metrics()))

    def clip_action(self, action):
        # Deliberately mutates its input to catch loss of the raw policy record.
        np.clip(action, -1, 1, out=action)
        return action

    def step(self, action):
        self.t += 1
        return StepResult(observation=self.observation(action), reward=0,
            terminated=False, truncated=False, evaluation=EvaluationState(self.metrics()),
            command_receipt=ExecutedCommand(action, action))


class FakePolicy:
    requires_executed_world_history = False
    def __init__(self):
        self.seen = []
    def reset(self):
        self.seen = []
    def begin_instruction(self, instruction):
        pass
    def act(self, history, instruction):
        self.seen.append(history)
        chunk = np.zeros((24, 7), np.float32)
        chunk[:, 0] = 2.5
        chunk[:, 1] = np.arange(24) / 100
        chunk[:, -1] = -1
        return chunk


def test_full_trace_preserves_raw_applied_commands_and_replan_history(tmp_path):
    policy = FakePolicy()
    report = record_episode(FakeEnv(), policy, seed=5000000, episode_dir=tmp_path / "episode",
                            max_steps=10, replan_steps=8, record_video=False)
    assert report["success"] and report["steps"] == 10
    assert report["first_success_step"] == 2
    assert report["plan_time_index"] == [0, 8]
    assert [h.time_index for h in policy.seen] == [0, 8]
    assert policy.seen[1].executed_action_history[-1, 0] == 1
    with np.load(tmp_path / "episode/trajectory.npz", allow_pickle=False) as a:
        assert tuple(a.files) == REFERENCE_KEYS
        assert a["robot_obs_trajectory"].shape == (11, 15)
        assert a["object_positions"].shape == (11, 3, 3)
        np.testing.assert_array_equal(a["raw_executed"][:, 0], np.full(10, 2.5))
        np.testing.assert_array_equal(a["executed"][:, 0], np.ones(10))
        np.testing.assert_array_equal(a["executed_chunk_row"], [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
        assert a["robot_obs_trajectory"][0, -1] == 1
        assert np.all(a["robot_obs_trajectory"][1:, -1] == -1)
    assert len((tmp_path / "episode/telemetry.jsonl").read_text().splitlines()) == 11


def test_missing_privileged_scene_data_fails_instead_of_fabricating_zero():
    with pytest.raises(ValueError, match="missing"):
        scene_row({"success": False, "is_cubeA_on_cubeB": False})

