from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import h5py
import numpy as np

from clearvla.benchmarks.common import audit_benchmark_dataset
from clearvla.benchmarks.config import build_benchmark_config
from clearvla.benchmarks.libero import convert_libero
from clearvla.benchmarks.libero_terminal import (
    augment_libero_terminal_suffix,
    convert_libero_causal_prefix,
)
from clearvla.data.hdf5_episode import load_episode
from clearvla.mainline.data.loading import _normalizers

ROOT = Path(__file__).resolve().parents[1]


def _observation(step: int, side: int = 4) -> dict[str, np.ndarray]:
    return {
        "robot0_eef_pos": np.asarray(
            [step / 100.0, 0.2, 0.3], dtype=np.float32
        ),
        "robot0_eef_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(
            [0.01 + step / 10000.0, -0.02], dtype=np.float32
        ),
        "agentview_image": np.full((side, side, 3), step, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full(
            (side, side, 3), step + 1, dtype=np.uint8
        ),
    }


def _state_row(step: int) -> np.ndarray:
    observation = _observation(step)
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            np.zeros(3, dtype=np.float32),
            np.asarray(
                [np.abs(observation["robot0_gripper_qpos"]).sum()],
                dtype=np.float32,
            ),
        )
    )


def _write_raw(root: Path) -> tuple[Path, str]:
    instruction = "put the red cube on the plate"
    suite = root / "libero_spatial"
    suite.mkdir(parents=True)
    raw_path = suite / "task_demo.hdf5"
    length = 73
    actions = np.zeros((length, 7), dtype=np.float64)
    actions[:, 0] = np.linspace(-0.2, 0.2, length)
    actions[:, 6] = np.where(np.arange(length) < 40, 1.0, -1.0)
    with h5py.File(raw_path, "w") as stream:
        data = stream.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": instruction}
        )
        for demo_index in range(3):
            demo = data.create_group(f"demo_{demo_index}")
            demo.create_dataset("actions", data=actions)
            # Raw LIBERO states are pre-action simulator snapshots.  Released
            # obs[t], however, was captured after actions[t].
            states = np.zeros((length, 4), dtype=np.float64)
            states[:, 0] = np.arange(length)
            states[:, 1] = demo_index
            demo.create_dataset("states", data=states)
            rewards = np.zeros(length, dtype=np.uint8)
            dones = np.zeros(length, dtype=np.uint8)
            rewards[-1] = dones[-1] = 1
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            obs = demo.create_group("obs")
            post_steps = np.arange(1, length + 1)
            obs.create_dataset(
                "ee_states",
                data=np.stack([_state_row(int(step))[:6] for step in post_steps]),
            )
            obs.create_dataset(
                "gripper_states",
                data=np.stack(
                    [
                        _observation(int(step))["robot0_gripper_qpos"]
                        for step in post_steps
                    ]
                ),
            )
            obs.create_dataset(
                "agentview_rgb",
                data=np.stack(
                    [_observation(int(step))["agentview_image"] for step in post_steps]
                ),
            )
            obs.create_dataset(
                "eye_in_hand_rgb",
                data=np.stack(
                    [
                        _observation(int(step))["robot0_eye_in_hand_image"]
                        for step in post_steps
                    ]
                ),
            )
    return raw_path, instruction


def _write_bddl(root: Path, instruction: str) -> Path:
    suite = root / "libero_spatial"
    suite.mkdir(parents=True)
    suite.joinpath(instruction.replace(" ", "_") + ".bddl").write_text(
        "(define (problem fake))", encoding="utf-8"
    )
    return root


class _Controller:
    def __init__(self) -> None:
        self.resets = 0

    def reset_goal(self) -> None:
        self.resets += 1


class _Simulator:
    def reset(self) -> None:
        pass

    def forward(self) -> None:
        pass


class _FakeEnv:
    instances: list["_FakeEnv"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.sim = _Simulator()
        self.robots = [SimpleNamespace(controller=_Controller())]
        self.state = np.zeros(4, dtype=np.float64)
        self.observation = _observation(0)
        self.success = False
        self.closed = False
        self.steps = 0
        self.__class__.instances.append(self)

    def seed(self, seed: int) -> None:
        self.seed_value = seed

    def reset(self) -> Mapping[str, np.ndarray]:
        self.success = False
        return self.observation

    def set_init_state(self, state: np.ndarray) -> Mapping[str, np.ndarray]:
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.observation = _observation(int(self.state[0]))
        return self.observation

    def get_sim_state(self) -> np.ndarray:
        return self.state.copy()

    def _post_process(self) -> None:
        pass

    def _update_observables(self, force: bool = False) -> None:
        assert force is True

    def _get_observations(self) -> Mapping[str, np.ndarray]:
        return self.observation

    def step(self, action: np.ndarray):
        assert action.shape == (7,)
        self.steps += 1
        self.success = True
        self.observation = _observation(73)
        return self.observation, 1.0, True, {}

    def check_success(self) -> bool:
        return self.success

    def close(self) -> None:
        self.closed = True


def _quat_to_axisangle(quaternion: np.ndarray) -> np.ndarray:
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])
    return np.zeros(3, dtype=np.float32)


def _episode(path: Path):
    return load_episode(
        path,
        cameras=("top", "wrist"),
        action_key="action",
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides={
            "top": "observations/images/cam_high",
            "wrist": "observations/images/cam_right_wrist",
        },
    )


def test_causal_prefix_and_terminal_conversion_align_rows_and_preserve_e8_chart(
    tmp_path: Path,
) -> None:
    _FakeEnv.instances.clear()
    raw_root = tmp_path / "raw"
    _, instruction = _write_raw(raw_root)
    bddl_root = _write_bddl(tmp_path / "bddl", instruction)
    legacy = tmp_path / "legacy"
    convert_libero(
        raw_root,
        legacy,
        suites=("libero_spatial",),
        evaluator_bddl_root=bddl_root,
    )
    prefix = tmp_path / "prefix"
    terminal = tmp_path / "terminal"
    prefix_report = convert_libero_causal_prefix(
        legacy,
        prefix,
        raw_source=raw_root,
        evaluator_bddl_root=bddl_root,
        env_factory=_FakeEnv,
        quat_to_axisangle=_quat_to_axisangle,
    )
    terminal_report = augment_libero_terminal_suffix(
        legacy,
        terminal,
        raw_source=raw_root,
        evaluator_bddl_root=bddl_root,
        env_factory=_FakeEnv,
        quat_to_axisangle=_quat_to_axisangle,
    )
    assert prefix_report["valid_windows"] == 75
    assert prefix_report["strict_valid_windows"] == 3
    assert terminal_report["valid_windows"] == 219
    assert terminal_report["strict_valid_windows"] == 3
    assert len(_FakeEnv.instances) == 2
    assert all(env.closed for env in _FakeEnv.instances)
    # Terminal replay now executes the complete source episode so OSC
    # controller targets (which are not part of the flattened MuJoCo state)
    # remain causal.  Each fake environment owns three 73-step demos.
    assert all(env.steps == 3 * 73 for env in _FakeEnv.instances)

    name = json.loads((legacy / "splits.json").read_text())["splits"]["train"][0]
    legacy_episode = _episode(legacy / f"{name}.hdf5")
    prefix_episode = _episode(prefix / f"{name}.hdf5")
    terminal_episode = _episode(terminal / f"{name}.hdf5")
    np.testing.assert_array_equal(prefix_episode.states_raw[0], _state_row(0))
    np.testing.assert_array_equal(
        prefix_episode.states_raw[1:], legacy_episode.states_raw[:-1]
    )
    assert prefix_episode.length == 73
    assert prefix_episode.valid_center_start == 0
    assert prefix_episode.valid_center_end == 24
    assert terminal_episode.length == 121
    assert terminal_episode.source_action_count == 73
    assert terminal_episode.terminal_state_index == 73
    np.testing.assert_array_equal(
        terminal_episode.states_raw[:73], prefix_episode.states_raw
    )
    np.testing.assert_array_equal(
        terminal_episode.states_raw[73:],
        np.repeat(legacy_episode.states_raw[-1:], 48, axis=0),
    )
    np.testing.assert_array_equal(
        terminal_episode.actions_raw[73:, :6], np.zeros((48, 6), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        terminal_episode.actions_raw[73:, 6],
        np.full(48, legacy_episode.actions_raw[-1, 6], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_episode.action_states_raw[1:], terminal_episode.actions_raw[:-1]
    )

    legacy_all = [_episode(path) for path in sorted(legacy.glob("*.hdf5"))]
    prefix_all = [_episode(path) for path in sorted(prefix.glob("*.hdf5"))]
    terminal_all = [_episode(path) for path in sorted(terminal.glob("*.hdf5"))]
    train_ids = [0]
    legacy_action, legacy_state = _normalizers(legacy_all, train_ids, mode="zscore")
    prefix_action, prefix_state = _normalizers(prefix_all, train_ids, mode="zscore")
    terminal_action, terminal_state = _normalizers(
        terminal_all, train_ids, mode="zscore"
    )
    assert prefix_action.to_dict() == legacy_action.to_dict() == terminal_action.to_dict()
    assert prefix_state.to_dict() == legacy_state.to_dict() == terminal_state.to_dict()

    with h5py.File(terminal / f"{name}.hdf5", "r") as stream:
        top = np.asarray(stream["observations/images/cam_high"])
        assert int(top[0, 0, 0, 0]) == 0
        assert int(top[1, 0, 0, 0]) == 1
        assert np.all(top[73:] == 73)


def test_causal_manifests_build_their_owned_window_contract(tmp_path: Path) -> None:
    _FakeEnv.instances.clear()
    raw_root = tmp_path / "raw"
    _, instruction = _write_raw(raw_root)
    bddl_root = _write_bddl(tmp_path / "bddl", instruction)
    legacy = tmp_path / "legacy"
    convert_libero(
        raw_root,
        legacy,
        suites=("libero_spatial",),
        evaluator_bddl_root=bddl_root,
    )
    terminal = tmp_path / "terminal"
    augment_libero_terminal_suffix(
        legacy,
        terminal,
        raw_source=raw_root,
        evaluator_bddl_root=bddl_root,
        env_factory=_FakeEnv,
        quat_to_axisangle=_quat_to_axisangle,
    )
    payload = build_benchmark_config(
        ROOT / "configs/mainline/object_intent_dynamics_323.json",
        tmp_path / "config.json",
        hdf5_root=terminal,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
        gripper_event_threshold=0.1,
    )
    assert (
        payload["data"]["window_boundary_contract"]
        == "causal_prefix_terminal_suffix_v2"
    )
    assert audit_benchmark_dataset(terminal)["valid_windows"] == 219
