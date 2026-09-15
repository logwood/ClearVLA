from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from clearvla.benchmarks.calvin_eval import (
    _CalvinVideoRun,
    _record_official_calvin_sequences,
)


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"fake-mp4")
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame).copy())

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    COLOR_RGB2BGR = 1
    INTER_NEAREST = 2
    FONT_HERSHEY_SIMPLEX = 3
    LINE_AA = 4

    def __init__(self) -> None:
        self.writers: list[_FakeWriter] = []

    @staticmethod
    def cvtColor(image: np.ndarray, _code: int) -> np.ndarray:
        return image[..., ::-1].copy()

    @staticmethod
    def resize(image: np.ndarray, size: tuple[int, int], **_kwargs: Any) -> np.ndarray:
        width, height = size
        assert image.shape[:2] == (height, width)
        return image.copy()

    @staticmethod
    def rectangle(image: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return image

    @staticmethod
    def putText(image: np.ndarray, *_args: Any, **_kwargs: Any) -> np.ndarray:
        return image

    @staticmethod
    def VideoWriter_fourcc(*_args: str) -> int:
        return 0

    def VideoWriter(self, path: str, *_args: Any, **_kwargs: Any) -> _FakeWriter:
        writer = _FakeWriter(path)
        self.writers.append(writer)
        return writer


def _observation(step: int) -> dict[str, object]:
    return {
        "rgb_obs": {
            "rgb_static": np.full((320, 320, 3), step, dtype=np.uint8),
            "rgb_gripper": np.full((320, 320, 3), step + 20, dtype=np.uint8),
        },
        "robot_obs": np.full(15, step, dtype=np.float32),
    }


class _FakeEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.step_calls = 0
        self.reset_calls = 0

    def reset(self, **_kwargs: Any) -> None:
        self.step_index = 0
        self.reset_calls += 1

    def get_obs(self) -> dict[str, object]:
        return _observation(self.step_index)

    def step(self, _action: np.ndarray):
        self.step_index += 1
        self.step_calls += 1
        return _observation(self.step_index), 0.0, False, {"step": self.step_index}


class _FakeModel:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, _observation: Any, _instruction: str) -> np.ndarray:
        self.step_calls += 1
        return np.zeros(7, dtype=np.float32)


def _fake_official() -> SimpleNamespace:
    official = SimpleNamespace()

    def rollout(
        env: Any,
        model: Any,
        _task_oracle: Any,
        subtask: str,
        _val_annotations: Any,
        _plans: Any,
        _debug: bool,
    ) -> bool:
        observation = env.get_obs()
        model.reset()
        steps = 1 if subtask == "succeed" else 2
        for _ in range(steps):
            action = model.step(observation, subtask)
            observation, _reward, _done, _info = env.step(action)
        return subtask == "succeed"

    def evaluate_sequence(
        env: Any,
        model: Any,
        task_checker: Any,
        initial_state: Any,
        eval_sequence: Any,
        val_annotations: Any,
        plans: Any,
        debug: bool,
    ) -> int:
        del task_checker
        env.reset(robot_obs=initial_state, scene_obs={})
        successful = 0
        for subtask in eval_sequence:
            if not official.rollout(
                env,
                model,
                None,
                subtask,
                val_annotations,
                plans,
                debug,
            ):
                return successful
            successful += 1
        return successful

    official.rollout = rollout
    official.evaluate_sequence = evaluate_sequence
    return official


def test_inline_video_uses_the_scored_rollout_and_flushes_each_sequence(
    tmp_path: Path,
) -> None:
    official = _fake_official()
    original_rollout = official.rollout
    original_evaluate_sequence = official.evaluate_sequence
    cv2 = _FakeCV2()
    video_run = _CalvinVideoRun(
        tmp_path,
        sequence_limit=1,
        max_subtask_steps=2,
        fps=2.0,
        cv2_module=cv2,
    )
    env = _FakeEnv()
    model = _FakeModel()
    annotations = {
        "succeed": ["complete the first task"],
        "fail": ["attempt the second task"],
    }

    with _record_official_calvin_sequences(official, video_run):
        result = official.evaluate_sequence(
            env,
            model,
            None,
            {"drawer": "closed"},
            ["succeed", "fail"],
            annotations,
            {},
            False,
        )

    assert result == 1
    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert model.reset_calls == 2
    assert model.step_calls == 3
    assert official.rollout is original_rollout
    assert official.evaluate_sequence is original_evaluate_sequence

    video = tmp_path / "sequence_0001.mp4"
    report_path = tmp_path / "sequence_0001.json"
    assert video.is_file()
    assert report_path.is_file()
    assert not (tmp_path / ".sequence_0001.partial.mp4").exists()
    assert cv2.writers[0].released is True

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["completed"] is True
    assert report["official_successful_tasks"] == 1
    assert [task["steps"] for task in report["task_reports"]] == [1, 2]
    assert [task["success"] for task in report["task_reports"]] == [True, False]
    assert video_run.summary()["completed_sequences"] == 1


def test_inline_video_sequence_limit_does_not_change_unrecorded_rollout(
    tmp_path: Path,
) -> None:
    official = _fake_official()
    video_run = _CalvinVideoRun(
        tmp_path,
        sequence_limit=1,
        max_subtask_steps=2,
        fps=2.0,
        cv2_module=_FakeCV2(),
    )
    env = _FakeEnv()
    model = _FakeModel()
    annotations = {"succeed": ["complete the task"]}

    with _record_official_calvin_sequences(official, video_run):
        first = official.evaluate_sequence(
            env, model, None, {}, ["succeed"], annotations, {}, False
        )
        second = official.evaluate_sequence(
            env, model, None, {}, ["succeed"], annotations, {}, False
        )

    assert (first, second) == (1, 1)
    assert env.step_calls == 2
    assert model.step_calls == 2
    assert (tmp_path / "sequence_0001.mp4").is_file()
    assert not (tmp_path / "sequence_0002.mp4").exists()
    assert len(video_run.reports) == 1
