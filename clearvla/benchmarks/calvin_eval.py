"""Official and targeted CALVIN evaluation through the ClearVLA bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast

import numpy as np

from .bridge import RemotePolicyClient, policy_observation
from .io import atomic_json, atomic_npz

CALVIN_EVALUATION_SCHEMA = "clearvla-calvin-lh-mtlc-evaluation-v2"
CALVIN_TARGETED_SCHEMA = "clearvla-calvin-targeted-rollout-audit-v4"
CALVIN_OFFICIAL_SEQUENCES = 1000
CALVIN_OFFICIAL_SUBTASK_STEPS = 360
CALVIN_ACTION_NAMES = (
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
)
CALVIN_ACTION_LOW = np.full(7, -1.0, dtype=np.float32)
CALVIN_ACTION_HIGH = np.full(7, 1.0, dtype=np.float32)
CALVIN_PHYSICAL_SCALE = np.asarray(
    [0.02, 0.02, 0.02, 0.05, 0.05, 0.05],
    dtype=np.float32,
)
CALVIN_DEFAULT_INITIAL_STATE: dict[str, object] = {
    "slider": "right",
    "drawer": "closed",
    "lightbulb": 0,
    "led": 0,
    "red_block": "table",
    "blue_block": "table",
    "pink_block": "slider_right",
    "grasped": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def calvin_action_contract() -> dict[str, object]:
    return {
        "names": list(CALVIN_ACTION_NAMES),
        "normalized_low": CALVIN_ACTION_LOW.tolist(),
        "normalized_high": CALVIN_ACTION_HIGH.tolist(),
        "translation_scale_m_per_step": 0.02,
        "rotation_scale_rad_per_step": 0.05,
        "gripper_execution": "sign_to_binary_minus_or_plus_one",
        "source": (
            "official calvin_env Robot defaults: max_rel_pos=0.02, "
            "max_rel_orn=0.05; apply_action requires gripper in {-1,+1}"
        ),
    }


def execute_calvin_action(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("CALVIN policy action must be one finite [7] row")
    executed = np.clip(value, CALVIN_ACTION_LOW, CALVIN_ACTION_HIGH)
    executed[6] = 1.0 if float(value[6]) >= 0.0 else -1.0
    return executed


def calvin_policy_observation(
    observation: Mapping[str, Any],
    previous_action: np.ndarray,
):
    """Project an official CALVIN observation onto non-privileged evidence."""

    rgb = observation.get("rgb_obs")
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation has no rgb_obs mapping")
    robot = np.asarray(observation.get("robot_obs"), dtype=np.float32)
    if robot.ndim != 1 or robot.shape[0] < 7:
        raise ValueError("CALVIN robot_obs must contain at least seven values")
    return policy_observation(
        top=np.asarray(rgb["rgb_static"]),
        wrist=np.asarray(rgb["rgb_gripper"]),
        state=robot[:7],
        action_state=np.asarray(previous_action, dtype=np.float32),
    )


def _deployment_mapping(health: Mapping[str, Any]) -> Mapping[str, Any]:
    deployment = health.get("deployment")
    if not isinstance(deployment, Mapping):
        raise ValueError("bridge health has no deployment mapping")
    return cast(Mapping[str, Any], deployment)


def validate_bridge_health(
    health: Mapping[str, Any],
    *,
    allow_smoke_policy: bool,
    require_observe_only: bool = False,
) -> None:
    if health.get("status") != "ok":
        raise RuntimeError(f"ClearVLA bridge is unhealthy: {dict(health)}")
    if require_observe_only:
        protocol = health.get("protocol")
        if not isinstance(protocol, Mapping) or protocol.get("observe_only") is not True:
            raise ValueError(
                "chunked CALVIN execution requires bridge observe-only protocol support"
            )
    mode = str(health.get("mode", ""))
    if mode == "smoke-zero":
        if allow_smoke_policy:
            return
        raise RuntimeError("formal CALVIN evaluation refuses the smoke-zero policy")
    if mode != "formal-checkpoint":
        raise RuntimeError(f"unknown ClearVLA bridge mode {mode!r}")
    deployment = _deployment_mapping(health)
    observation = deployment.get("observation")
    action = deployment.get("action")
    if not isinstance(observation, Mapping) or not isinstance(action, Mapping):
        raise ValueError("bridge deployment health lacks observation/action contracts")
    if tuple(observation.get("camera_names", ())) != ("top", "wrist"):
        raise ValueError("CALVIN bridge camera order must be top,wrist")
    profile = action.get("data_profile")
    if not isinstance(profile, Mapping) or profile.get("name") != "calvin_relative_7d_v1":
        raise ValueError("CALVIN evaluation requires the calvin_relative_7d_v1 action chart")
    if str(action.get("gripper_output_mode", "")) != "calvin_binary_command":
        raise ValueError(
            "CALVIN evaluation requires the explicit calvin_binary_command gripper head"
        )
    if str(action.get("arm_flow_mode", "")) != "relative_command_adapter":
        raise ValueError(
            "CALVIN evaluation requires the relative-command action adapter"
        )
    if int(action.get("receding_horizon_execute_rows", 0)) != 1:
        raise ValueError("CALVIN deployment must replan after one executed row")


class CalvinBridgeModel:
    """Duck-typed CalvinBaseModel with complete action/latency auditing."""

    def __init__(self, client: RemotePolicyClient, *, execute_rows: int = 1) -> None:
        if int(execute_rows) <= 0 or int(execute_rows) > 24:
            raise ValueError("CALVIN execute_rows must be in [1,24]")
        self.client = client
        self.execute_rows = int(execute_rows)
        self.raw_chunks: list[np.ndarray] = []
        self.raw_executed_actions: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.executed_plan_indices: list[int] = []
        self.executed_chunk_rows: list[int] = []
        # Explicit local replan provenance.  A row index is only meaningful
        # together with the cumulative environment step at which its plan was
        # created; retaining both prevents a chunk boundary from being
        # misread as a fresh episode phase in audit artifacts.
        self.plan_origin_steps: list[int] = []
        self.executed_plan_origin_steps: list[int] = []
        self.observed_history_time_indices: list[int] = []
        self.observed_after_executed_steps: list[int] = []
        self.latency_seconds: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._previous_action = np.zeros(7, dtype=np.float32)
        self._pending_reset = True
        self._planned_chunk: np.ndarray | None = None
        self._planned_chunk_index: int | None = None
        self._planned_chunk_origin_step = 0
        self._next_chunk_row = 0

    def plan(self, observation: Mapping[str, Any], goal: str) -> np.ndarray:
        """Create one action chunk at the current physical environment time."""

        policy_input = calvin_policy_observation(observation, self._previous_action)
        started = time.perf_counter()
        chunk = self.client.act(
            policy_input,
            str(goal),
            reset=self._pending_reset,
        )
        elapsed = time.perf_counter() - started
        raw = np.asarray(chunk, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
            raise ValueError("CALVIN bridge returned an invalid [T,7] chunk")
        if raw.shape[0] == 0:
            raise ValueError("CALVIN bridge returned an empty action chunk")
        self.raw_chunks.append(raw.copy())
        self.latency_seconds.append(float(elapsed))
        self._planned_chunk = raw.copy()
        self._planned_chunk_index = len(self.raw_chunks) - 1
        self._planned_chunk_origin_step = len(self.executed_actions)
        self.plan_origin_steps.append(int(self._planned_chunk_origin_step))
        self._next_chunk_row = 0
        self._pending_reset = False
        return raw.copy()

    def execute_planned_row(self, row: int) -> np.ndarray:
        """Decode and audit one row from the most recent action chunk."""

        if self._planned_chunk is None or self._planned_chunk_index is None:
            raise RuntimeError("CALVIN action execution requires a preceding plan")
        row_index = int(row)
        if row_index < 0 or row_index >= self._planned_chunk.shape[0]:
            raise IndexError(
                f"planned row {row_index} is outside horizon {self._planned_chunk.shape[0]}"
            )
        raw = self._planned_chunk[row_index]
        action = execute_calvin_action(raw)
        self.raw_executed_actions.append(raw.copy())
        self.executed_actions.append(action.copy())
        self.executed_plan_indices.append(int(self._planned_chunk_index))
        self.executed_chunk_rows.append(row_index)
        self.executed_plan_origin_steps.append(int(self._planned_chunk_origin_step))
        self._previous_action = action.copy()
        self._next_chunk_row = max(int(self._next_chunk_row), row_index + 1)
        return action

    def observe(self, observation: Mapping[str, Any]) -> int:
        """Record an intermediate physical step while retaining the current plan."""

        if self._pending_reset:
            raise RuntimeError("CALVIN observe requires an initialized policy episode")
        if not self.executed_actions:
            raise RuntimeError("CALVIN observe requires an executed action")
        policy_input = calvin_policy_observation(observation, self._previous_action)
        time_index = int(self.client.observe(policy_input))
        self.observed_history_time_indices.append(time_index)
        self.observed_after_executed_steps.append(len(self.executed_actions))
        return time_index

    def step(self, observation: Mapping[str, Any], goal: str) -> np.ndarray:
        """Return one environment action, optionally reusing a planned chunk.

        The official CALVIN evaluator calls ``step`` once per physical
        environment step.  For ``execute_rows > 1`` we retain the current
        policy chunk and append each intervening post-action observation via
        the bridge's observe-only endpoint.  At a chunk boundary we call
        ``act`` directly with the current observation; ``PolicyBridge.act``
        then appends that observation exactly once before replanning.
        """

        chunk = self._planned_chunk
        if (
            chunk is None
            or self._next_chunk_row >= min(self.execute_rows, chunk.shape[0])
        ):
            chunk = self.plan(observation, goal)
            if chunk.shape[0] < self.execute_rows:
                raise ValueError(
                    f"requested {self.execute_rows} executed rows but policy horizon "
                    f"is only {chunk.shape[0]}"
                )
            row = 0
        else:
            # The observation is the result of the previously executed row.
            # Keep the plan and update causal history without another policy
            # inference.  The first row of a fresh chunk is handled by act()
            # above, so no duplicate observation is appended at the boundary.
            self.observe(observation)
            row = int(self._next_chunk_row)
        return self.execute_planned_row(row)

    def action_arrays(self) -> dict[str, np.ndarray]:
        if not self.raw_chunks:
            return {
                "raw_chunks": np.empty((0, 0, 7), dtype=np.float32),
                "raw_first": np.empty((0, 7), dtype=np.float32),
                "raw_executed": np.empty((0, 7), dtype=np.float32),
                "executed": np.empty((0, 7), dtype=np.float32),
                "executed_plan_index": np.empty((0,), dtype=np.int64),
                "executed_chunk_row": np.empty((0,), dtype=np.int64),
                "plan_origin_step": np.empty((0,), dtype=np.int64),
                "executed_plan_origin_step": np.empty((0,), dtype=np.int64),
                "observed_history_time_index": np.empty((0,), dtype=np.int64),
                "observed_after_executed_steps": np.empty((0,), dtype=np.int64),
                "latency_seconds": np.empty((0,), dtype=np.float64),
            }
        horizons = {int(value.shape[0]) for value in self.raw_chunks}
        if len(horizons) != 1:
            raise ValueError(f"policy action horizon changed during rollout: {horizons}")
        raw_chunks = np.stack(self.raw_chunks).astype(np.float32)
        return {
            "raw_chunks": raw_chunks,
            "raw_first": raw_chunks[:, 0],
            "raw_executed": np.stack(self.raw_executed_actions).astype(np.float32),
            "executed": np.stack(self.executed_actions).astype(np.float32),
            "executed_plan_index": np.asarray(
                self.executed_plan_indices, dtype=np.int64
            ),
            "executed_chunk_row": np.asarray(self.executed_chunk_rows, dtype=np.int64),
            "plan_origin_step": np.asarray(self.plan_origin_steps, dtype=np.int64),
            "executed_plan_origin_step": np.asarray(
                self.executed_plan_origin_steps, dtype=np.int64
            ),
            "observed_history_time_index": np.asarray(
                self.observed_history_time_indices, dtype=np.int64
            ),
            "observed_after_executed_steps": np.asarray(
                self.observed_after_executed_steps, dtype=np.int64
            ),
            "latency_seconds": np.asarray(self.latency_seconds, dtype=np.float64),
        }

    def action_audit(self) -> dict[str, object]:
        arrays = self.action_arrays()
        raw_chunks = arrays["raw_chunks"]
        raw_first = arrays["raw_first"]
        raw_executed = arrays["raw_executed"]
        executed = arrays["executed"]
        latency = arrays["latency_seconds"]
        if raw_first.shape[0] == 0:
            return {"steps": 0, "action_names": list(CALVIN_ACTION_NAMES)}
        gripper = executed[:, 6]
        switches = int(np.count_nonzero(gripper[1:] != gripper[:-1]))
        return {
            "steps": int(executed.shape[0]),
            "planning_decisions": int(raw_first.shape[0]),
            "mean_executed_rows_per_plan": float(
                executed.shape[0] / raw_first.shape[0]
            ),
            "action_names": list(CALVIN_ACTION_NAMES),
            "raw_chunk_min": raw_chunks.min(axis=(0, 1)).tolist(),
            "raw_chunk_max": raw_chunks.max(axis=(0, 1)).tolist(),
            "raw_chunk_oob_count_per_dim": (
                (raw_chunks < CALVIN_ACTION_LOW).sum(axis=(0, 1))
                + (raw_chunks > CALVIN_ACTION_HIGH).sum(axis=(0, 1))
            ).tolist(),
            "raw_first_min": raw_first.min(axis=0).tolist(),
            "raw_first_max": raw_first.max(axis=0).tolist(),
            "raw_first_abs_max": np.abs(raw_first).max(axis=0).tolist(),
            "raw_first_oob_count_per_dim": (
                (raw_first < CALVIN_ACTION_LOW).sum(axis=0)
                + (raw_first > CALVIN_ACTION_HIGH).sum(axis=0)
            ).tolist(),
            "raw_first_oob_row_count": int(
                np.any(
                    (raw_first < CALVIN_ACTION_LOW)
                    | (raw_first > CALVIN_ACTION_HIGH),
                    axis=1,
                ).sum()
            ),
            "raw_executed_min": raw_executed.min(axis=0).tolist(),
            "raw_executed_max": raw_executed.max(axis=0).tolist(),
            "raw_executed_abs_max": np.abs(raw_executed).max(axis=0).tolist(),
            "raw_executed_oob_count_per_dim": (
                (raw_executed < CALVIN_ACTION_LOW).sum(axis=0)
                + (raw_executed > CALVIN_ACTION_HIGH).sum(axis=0)
            ).tolist(),
            "raw_executed_oob_row_count": int(
                np.any(
                    (raw_executed < CALVIN_ACTION_LOW)
                    | (raw_executed > CALVIN_ACTION_HIGH),
                    axis=1,
                ).sum()
            ),
            "executed_min": executed.min(axis=0).tolist(),
            "executed_max": executed.max(axis=0).tolist(),
            "executed_abs_max": np.abs(executed).max(axis=0).tolist(),
            "executed_arm_physical_abs_max_per_step": (
                np.abs(executed[:, :6]).max(axis=0) * CALVIN_PHYSICAL_SCALE
            ).tolist(),
            "executed_gripper_open_steps": int((gripper > 0).sum()),
            "executed_gripper_close_steps": int((gripper < 0).sum()),
            "executed_gripper_switches": switches,
            "latency_seconds": {
                "mean": float(latency.mean()),
                "p50": float(np.quantile(latency, 0.50)),
                "p90": float(np.quantile(latency, 0.90)),
                "p99": float(np.quantile(latency, 0.99)),
                "max": float(latency.max()),
            },
        }


def _summary(results: list[int]) -> dict[str, Any]:
    if not results:
        raise ValueError("CALVIN evaluator returned no sequences")
    counts = Counter(int(value) for value in results)
    chain_success = {
        str(length): sum(counts[value] for value in range(length, 6)) / len(results)
        for length in range(1, 6)
    }
    return {
        "average_successful_sequence_length": float(np.mean(results)),
        "chain_success": chain_success,
        "sequence_count": len(results),
    }


def _environment(dataset_root: Path, *, show_gui: bool):
    validation = dataset_root / "validation"
    if not (validation / ".hydra" / "merged_config.yaml").is_file():
        raise FileNotFoundError(
            f"CALVIN validation environment config is absent under {validation}"
        )
    from calvin_env.envs.play_table_env import get_env

    return get_env(
        validation,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        show_gui=bool(show_gui),
    )


def _calvin_video_panel(
    cv2: Any,
    value: Any,
    *,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected CALVIN RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(bgr, (0, 0), (width - 1, 27), (0, 0, 0), thickness=-1)
    cv2.putText(
        bgr,
        title,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr


def _calvin_video_frame(
    cv2: Any,
    observation: Mapping[str, Any],
    *,
    sequence_index: int,
    task_index: int,
    instruction: str,
    step: int,
    max_steps: int,
    status: str,
) -> np.ndarray:
    rgb = observation.get("rgb_obs")
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation has no rgb_obs mapping")
    static = _calvin_video_panel(
        cv2,
        rgb["rgb_static"],
        width=320,
        height=320,
        title="static camera",
    )
    wrist = _calvin_video_panel(
        cv2,
        rgb["rgb_gripper"],
        width=320,
        height=320,
        title="wrist camera",
    )
    panels = np.concatenate([static, wrist], axis=1)
    canvas = np.full((374, panels.shape[1], 3), 20, dtype=np.uint8)
    canvas[:320] = panels
    color = {
        "SUCCESS": (80, 235, 120),
        "FAILED": (60, 80, 235),
        "INTERRUPTED": (60, 180, 235),
    }.get(status, (235, 235, 235))
    cv2.putText(
        canvas,
        (
            f"sequence {sequence_index:04d}  task {task_index}/5  "
            f"step {step:03d}/{max_steps:03d}  {status}"
        ),
        (10, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        instruction,
        (10, 361),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


class _CalvinSequenceVideoRecorder:
    """Stream one official sequence to disk without retaining rollout frames."""

    def __init__(
        self,
        output_dir: Path,
        *,
        sequence_index: int,
        initial_state: Mapping[str, object],
        eval_sequence: list[str],
        max_subtask_steps: int,
        fps: float,
        cv2_module: Any | None = None,
    ) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise RuntimeError(
                    "CALVIN video recording requires OpenCV in the evaluator environment"
                ) from error
        self.cv2 = cv2_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_index = int(sequence_index)
        self.initial_state = dict(initial_state)
        self.eval_sequence = [str(value) for value in eval_sequence]
        self.max_subtask_steps = int(max_subtask_steps)
        self.fps = float(fps)
        stem = f"sequence_{self.sequence_index:04d}"
        self.video_path = self.output_dir / f"{stem}.mp4"
        self.partial_video_path = self.output_dir / f".{stem}.partial.mp4"
        self.report_path = self.output_dir / f"{stem}.json"
        self.partial_report_path = self.output_dir / f".{stem}.partial.json"
        self.writer: Any | None = None
        self.frame_count = 0
        self.task_reports: list[dict[str, object]] = []
        self.current_task: dict[str, object] | None = None
        self.last_observation: Mapping[str, Any] | None = None
        self.task_start_frame_written = False
        self.started = time.perf_counter()
        self.closed = False

    def _write_frame(self, observation: Mapping[str, Any], *, status: str) -> None:
        if self.current_task is None:
            raise RuntimeError("CALVIN video frame has no active task")
        frame = _calvin_video_frame(
            self.cv2,
            observation,
            sequence_index=self.sequence_index,
            task_index=int(self.current_task["task_index"]),
            instruction=str(self.current_task["instruction"]),
            step=int(self.current_task["steps"]),
            max_steps=self.max_subtask_steps,
            status=status,
        )
        if self.writer is None:
            self.writer = self.cv2.VideoWriter(
                str(self.partial_video_path),
                self.cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (int(frame.shape[1]), int(frame.shape[0])),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"could not open CALVIN video writer: {self.video_path}")
        self.writer.write(frame)
        self.frame_count += 1

    def start_task(self, task: str, instruction: str) -> None:
        if self.closed:
            raise RuntimeError("cannot append a task to a closed CALVIN video")
        if self.current_task is not None:
            raise RuntimeError("cannot start a CALVIN task before the preceding task ended")
        self.current_task = {
            "task_index": len(self.task_reports) + 1,
            "task": str(task),
            "instruction": str(instruction),
            "steps": 0,
            "success": None,
            "elapsed_seconds": None,
            "started_at": time.perf_counter(),
        }
        self.task_start_frame_written = False

    def capture_task_start(self, observation: Mapping[str, Any]) -> None:
        if self.task_start_frame_written:
            return
        self.last_observation = observation
        self._write_frame(observation, status="READY")
        self.task_start_frame_written = True

    def capture_step(self, observation: Mapping[str, Any]) -> None:
        if self.current_task is None:
            raise RuntimeError("CALVIN environment stepped without an active recorded task")
        self.current_task["steps"] = int(self.current_task["steps"]) + 1
        self.last_observation = observation
        self._write_frame(observation, status="RUNNING")

    def finish_task(self, success: bool) -> None:
        if self.current_task is None:
            raise RuntimeError("cannot finish a CALVIN video task that was not started")
        self.current_task["success"] = bool(success)
        self.current_task["elapsed_seconds"] = float(
            time.perf_counter() - float(self.current_task.pop("started_at"))
        )
        if self.last_observation is not None:
            status = "SUCCESS" if success else "FAILED"
            hold_frames = max(1, int(round(self.fps * 0.5)))
            for _ in range(hold_frames):
                self._write_frame(self.last_observation, status=status)
        self.task_reports.append(self.current_task)
        self.current_task = None

    def _release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def finish_sequence(self, official_result: int) -> dict[str, object]:
        if self.closed:
            raise RuntimeError("CALVIN sequence video was already closed")
        if self.current_task is not None:
            raise RuntimeError("cannot finish a CALVIN sequence with an active task")
        self._release()
        if not self.partial_video_path.is_file():
            raise RuntimeError("CALVIN sequence completed without any recorded video frames")
        os.replace(self.partial_video_path, self.video_path)
        report: dict[str, object] = {
            "schema": "clearvla-calvin-inline-video-sequence-v1",
            "sequence_index": self.sequence_index,
            "initial_state": self.initial_state,
            "tasks": self.eval_sequence,
            "task_reports": self.task_reports,
            "official_successful_tasks": int(official_result),
            "frame_count": int(self.frame_count),
            "fps": self.fps,
            "video": self.video_path.name,
            "completed": True,
            "elapsed_seconds": float(time.perf_counter() - self.started),
        }
        atomic_json(self.report_path, report)
        self.closed = True
        return report

    def abort_sequence(self, error: BaseException) -> None:
        if self.closed:
            return
        if self.current_task is not None:
            self.current_task["success"] = False
            self.current_task["interrupted"] = True
            self.current_task["elapsed_seconds"] = float(
                time.perf_counter() - float(self.current_task.pop("started_at"))
            )
            self.task_reports.append(self.current_task)
            self.current_task = None
        self._release()
        partial_report: dict[str, object] = {
            "schema": "clearvla-calvin-inline-video-sequence-v1",
            "sequence_index": self.sequence_index,
            "initial_state": self.initial_state,
            "tasks": self.eval_sequence,
            "task_reports": self.task_reports,
            "frame_count": int(self.frame_count),
            "fps": self.fps,
            "video_partial": (
                self.partial_video_path.name if self.partial_video_path.is_file() else None
            ),
            "completed": False,
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": float(time.perf_counter() - self.started),
        }
        atomic_json(self.partial_report_path, partial_report)
        self.closed = True


class _RecordingCalvinEnv:
    """Transparent environment proxy that records the observations already produced."""

    def __init__(self, env: Any, recorder: _CalvinSequenceVideoRecorder) -> None:
        self._env = env
        self.recorder = recorder

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.reset(*args, **kwargs)

    def get_obs(self) -> Any:
        observation = self._env.get_obs()
        self.recorder.capture_task_start(cast(Mapping[str, Any], observation))
        return observation

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        self.recorder.capture_step(cast(Mapping[str, Any], result[0]))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _CalvinVideoRun:
    def __init__(
        self,
        output_dir: Path,
        *,
        sequence_limit: int,
        max_subtask_steps: int,
        fps: float,
        cv2_module: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_limit = int(sequence_limit)
        self.max_subtask_steps = int(max_subtask_steps)
        self.fps = float(fps)
        self.cv2_module = cv2_module
        self.sequence_index = 0
        self.reports: list[dict[str, object]] = []

    def begin_sequence(
        self,
        initial_state: Mapping[str, object],
        eval_sequence: Any,
    ) -> _CalvinSequenceVideoRecorder | None:
        self.sequence_index += 1
        if self.sequence_index > self.sequence_limit:
            return None
        return _CalvinSequenceVideoRecorder(
            self.output_dir,
            sequence_index=self.sequence_index,
            initial_state=initial_state,
            eval_sequence=[str(value) for value in eval_sequence],
            max_subtask_steps=self.max_subtask_steps,
            fps=self.fps,
            cv2_module=self.cv2_module,
        )

    def finish_sequence(
        self,
        recorder: _CalvinSequenceVideoRecorder,
        official_result: int,
    ) -> None:
        self.reports.append(recorder.finish_sequence(official_result))

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "directory": str(self.output_dir.resolve()),
            "fps": self.fps,
            "sequence_limit": self.sequence_limit,
            "completed_sequences": len(self.reports),
            "artifacts": [
                {
                    "sequence_index": int(report["sequence_index"]),
                    "video": str(self.output_dir / str(report["video"])),
                    "report": str(
                        self.output_dir
                        / f"sequence_{int(report['sequence_index']):04d}.json"
                    ),
                }
                for report in self.reports
            ],
        }


@contextmanager
def _record_official_calvin_sequences(
    official: Any,
    video_run: _CalvinVideoRun | None,
) -> Iterator[None]:
    """Patch only official loop boundaries; policy and scoring execute unchanged."""

    if video_run is None:
        yield
        return
    original_evaluate_sequence = official.evaluate_sequence
    original_rollout = official.rollout

    def recording_rollout(
        env: Any,
        model: Any,
        task_oracle: Any,
        subtask: str,
        val_annotations: Any,
        plans: Any,
        debug: bool,
    ) -> Any:
        if not isinstance(env, _RecordingCalvinEnv):
            return original_rollout(
                env,
                model,
                task_oracle,
                subtask,
                val_annotations,
                plans,
                debug,
            )
        instruction = str(val_annotations[subtask][0])
        env.recorder.start_task(str(subtask), instruction)
        try:
            success = original_rollout(
                env,
                model,
                task_oracle,
                subtask,
                val_annotations,
                plans,
                debug,
            )
        except BaseException:
            raise
        else:
            env.recorder.finish_task(bool(success))
            return success

    def recording_evaluate_sequence(
        env: Any,
        model: Any,
        task_checker: Any,
        initial_state: Mapping[str, object],
        eval_sequence: Any,
        val_annotations: Any,
        plans: Any,
        debug: bool,
    ) -> Any:
        recorder = video_run.begin_sequence(initial_state, eval_sequence)
        if recorder is None:
            return original_evaluate_sequence(
                env,
                model,
                task_checker,
                initial_state,
                eval_sequence,
                val_annotations,
                plans,
                debug,
            )
        recording_env = _RecordingCalvinEnv(env, recorder)
        try:
            result = original_evaluate_sequence(
                recording_env,
                model,
                task_checker,
                initial_state,
                eval_sequence,
                val_annotations,
                plans,
                debug,
            )
        except BaseException as error:
            recorder.abort_sequence(error)
            raise
        video_run.finish_sequence(recorder, int(result))
        return result

    setattr(official, "evaluate_sequence", recording_evaluate_sequence)
    setattr(official, "rollout", recording_rollout)
    try:
        yield
    finally:
        setattr(official, "evaluate_sequence", original_evaluate_sequence)
        setattr(official, "rollout", original_rollout)


def evaluate_calvin(
    dataset_root: Path,
    output_dir: Path,
    *,
    endpoint: str,
    timeout: float,
    num_sequences: int,
    max_subtask_steps: int,
    sequence_workers: int,
    show_gui: bool,
    debug: bool,
    execute_rows: int = 1,
    allow_smoke_policy: bool = False,
    record_video: bool = False,
    video_dir: Path | None = None,
    video_fps: float = 12.0,
    video_sequences: int | None = None,
) -> dict[str, Any]:
    if min(num_sequences, max_subtask_steps, sequence_workers) <= 0:
        raise ValueError("CALVIN sequence, step, and worker counts must be positive")
    if execute_rows <= 0 or execute_rows > 24:
        raise ValueError("CALVIN execute_rows must be in [1,24]")
    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("CALVIN video_fps must be finite and positive")
    if video_sequences is not None and (
        video_sequences <= 0 or video_sequences > num_sequences
    ):
        raise ValueError("CALVIN video_sequences must be in [1,num_sequences]")
    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health = client.health()
    validate_bridge_health(
        health,
        allow_smoke_policy=allow_smoke_policy,
        require_observe_only=execute_rows > 1,
    )

    official = importlib.import_module("calvin_agent.evaluation.evaluate_policy")
    env = _environment(dataset_root, show_gui=show_gui)
    model = CalvinBridgeModel(client, execute_rows=execute_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_video_dir = (
        Path(video_dir) if video_dir is not None else output_dir / "videos"
    )
    video_run = (
        _CalvinVideoRun(
            resolved_video_dir,
            sequence_limit=(
                int(video_sequences) if video_sequences is not None else int(num_sequences)
            ),
            max_subtask_steps=max_subtask_steps,
            fps=video_fps,
        )
        if record_video
        else None
    )
    original_get_sequences = official.get_sequences
    original_num_sequences = official.NUM_SEQUENCES
    original_ep_len = official.EP_LEN
    try:
        setattr(
            official,
            "get_sequences",
            lambda count: original_get_sequences(count, num_workers=sequence_workers),
        )
        setattr(official, "NUM_SEQUENCES", int(num_sequences))
        setattr(official, "EP_LEN", int(max_subtask_steps))
        with _record_official_calvin_sequences(official, video_run):
            results = official.evaluate_policy(
                model,
                env,
                epoch="clearvla",
                eval_log_dir=output_dir,
                debug=debug,
                create_plan_tsne=False,
            )
    finally:
        setattr(official, "get_sequences", original_get_sequences)
        setattr(official, "NUM_SEQUENCES", original_num_sequences)
        setattr(official, "EP_LEN", original_ep_len)
        env.close()
    arrays = model.action_arrays()
    atomic_npz(output_dir / "actions.npz", **arrays)
    result = {
        "schema": CALVIN_EVALUATION_SCHEMA,
        "benchmark": "CALVIN LH-MTLC",
        "dataset_root": str(dataset_root.resolve()),
        "bridge_endpoint": endpoint,
        "bridge_health_before": dict(health),
        "bridge_health_after": client.health(),
        "action_contract": calvin_action_contract(),
        "action_audit": model.action_audit(),
        "official_protocol": (
            num_sequences == CALVIN_OFFICIAL_SEQUENCES
            and max_subtask_steps == CALVIN_OFFICIAL_SUBTASK_STEPS
            and execute_rows == 1
        ),
        "evaluator_execute_rows": int(execute_rows),
        "experimental_chunked_execution": bool(execute_rows != 1),
        "max_subtask_steps": int(max_subtask_steps),
        "sequence_workers": int(sequence_workers),
        "video_recording": (
            video_run.summary()
            if video_run is not None
            else {
                "enabled": False,
                "directory": None,
                "fps": float(video_fps),
                "sequence_limit": 0,
                "completed_sequences": 0,
                "artifacts": [],
            }
        ),
        **_summary([int(value) for value in results]),
    }
    atomic_json(output_dir / "clearvla_summary.json", result)
    return result


def _save_rgb(path: Path, value: np.ndarray) -> None:
    from PIL import Image

    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"cannot save non-RGB observation {image.shape} {image.dtype}")
    Image.fromarray(image).save(path)


def _official_task_assets():
    import hydra
    from omegaconf import OmegaConf

    official = importlib.import_module("calvin_agent.evaluation.evaluate_policy")
    conf_dir = Path(official.__file__).resolve().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    oracle = hydra.utils.instantiate(task_cfg)
    annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return official, conf_dir, oracle, annotations


def evaluate_calvin_targeted(
    dataset_root: Path,
    output_dir: Path,
    *,
    task: str,
    instruction: str | None,
    initial_state: Mapping[str, object],
    endpoint: str,
    timeout: float,
    max_steps: int,
    show_gui: bool,
    execute_rows: int = 1,
    allow_smoke_policy: bool = False,
    record_video: bool = False,
    video_dir: Path | None = None,
    video_fps: float = 12.0,
) -> dict[str, Any]:
    if max_steps <= 0 or max_steps > CALVIN_OFFICIAL_SUBTASK_STEPS:
        raise ValueError("targeted CALVIN max_steps must be in [1,360]")
    if execute_rows <= 0 or execute_rows > 24:
        raise ValueError("targeted CALVIN execute_rows must be in [1,24]")
    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("targeted CALVIN video_fps must be finite and positive")
    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_bridge_health(
        health_before,
        allow_smoke_policy=allow_smoke_policy,
        require_observe_only=execute_rows > 1,
    )
    official, conf_dir, oracle, annotations = _official_task_assets()
    if task not in annotations:
        raise KeyError(f"unknown official CALVIN task {task!r}")
    task_instruction = (
        str(annotations[task][0]) if instruction is None else str(instruction)
    )
    if not task_instruction.strip():
        raise ValueError("targeted CALVIN instruction must be non-empty")

    env = _environment(dataset_root, show_gui=show_gui)
    # The targeted path must use the same chunk-retention state machine as the
    # formal evaluator.  Constructing the default one-row bridge here and then
    # manually indexing a fresh plan made execute_rows an outer-loop cosmetic
    # override and reset the action phase at every replan.
    model = CalvinBridgeModel(client, execute_rows=execute_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_video_dir = (
        Path(video_dir) if video_dir is not None else output_dir / "videos"
    )
    video_recorder = (
        _CalvinSequenceVideoRecorder(
            resolved_video_dir,
            sequence_index=1,
            initial_state=initial_state,
            eval_sequence=[task],
            max_subtask_steps=max_steps,
            fps=video_fps,
        )
        if record_video
        else None
    )
    video_report: dict[str, object] | None = None
    robot_states: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        robot_obs, scene_obs = official.get_env_state_for_initial_condition(
            dict(initial_state)
        )
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = env.get_obs()
        initial_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
        _save_rgb(output_dir / "initial_static.png", initial_rgb["rgb_static"])
        _save_rgb(output_dir / "initial_wrist.png", initial_rgb["rgb_gripper"])
        if video_recorder is not None:
            video_recorder.start_task(task, task_instruction)
            video_recorder.capture_task_start(observation)
        model.reset()
        start_info = env.get_info()
        success = False
        environment_steps = 0
        while environment_steps < int(max_steps) and not success:
            robot_states.append(
                np.asarray(observation["robot_obs"], dtype=np.float32).copy()
            )
            # step() retains the current chunk for execute_rows physical steps,
            # appends observe-only history between rows, and replans only at
            # the declared cursor boundary.
            action = model.step(observation, task_instruction)
            observation, _reward, _done, current_info = env.step(action)
            environment_steps += 1
            if video_recorder is not None:
                video_recorder.capture_step(observation)
            current_task = oracle.get_task_info_for_set(
                start_info,
                current_info,
                {task},
            )
            if len(current_task) > 0:
                success = True
        if video_recorder is not None:
            video_recorder.finish_task(success)
            video_report = video_recorder.finish_sequence(1 if success else 0)
        elapsed = time.perf_counter() - started
        final_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
        _save_rgb(output_dir / "final_static.png", final_rgb["rgb_static"])
        _save_rgb(output_dir / "final_wrist.png", final_rgb["rgb_gripper"])
    except BaseException as error:
        if video_recorder is not None:
            video_recorder.abort_sequence(error)
        raise
    finally:
        env.close()

    arrays = model.action_arrays()
    arrays["robot_obs"] = np.stack(robot_states).astype(np.float32)
    atomic_npz(output_dir / "actions.npz", **arrays)
    health_after = client.health()
    deployment = (
        _deployment_mapping(health_after)
        if health_after.get("mode") == "formal-checkpoint"
        else {}
    )
    checkpoint = deployment.get("checkpoint")
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}
    environment_steps = int(arrays["executed"].shape[0])
    planning_decisions = int(arrays["raw_chunks"].shape[0])
    history_time_index = health_after.get("history_time_index")
    expected_history_time_index = environment_steps - 1
    result = {
        "schema": CALVIN_TARGETED_SCHEMA,
        "benchmark": "CALVIN official D environment and task oracle",
        "dataset_root": str(dataset_root.resolve()),
        "task": task,
        "instruction": task_instruction,
        "initial_state": dict(initial_state),
        "success": bool(success),
        "steps": environment_steps,
        "max_steps": int(max_steps),
        "execution": {
            "checkpoint_receding_horizon_execute_rows": 1,
            "evaluator_execute_rows": int(execute_rows),
            "experimental_override": bool(execute_rows != 1),
            "replan_cursor_schema": "calvin_replan_cursor_v1",
            "plan_origin_steps": [int(value) for value in model.plan_origin_steps],
            "planning_decisions": planning_decisions,
            "mean_executed_rows_per_plan": (
                float(environment_steps / planning_decisions)
                if planning_decisions
                else None
            ),
            "intermediate_observations": int(
                arrays["observed_history_time_index"].shape[0]
            ),
            "history_time_index_expected_after": expected_history_time_index,
            "history_time_index_actual_after": history_time_index,
            "history_time_index_aligned": bool(
                history_time_index == expected_history_time_index
            ),
        },
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": (
            float(elapsed / arrays["executed"].shape[0])
            if arrays["executed"].shape[0]
            else None
        ),
        "bridge_endpoint": endpoint,
        "bridge_health_before": dict(health_before),
        "bridge_health_after": dict(health_after),
        "checkpoint": dict(checkpoint_mapping),
        "action_contract": calvin_action_contract(),
        "action_audit": model.action_audit(),
        "video_recording": (
            {
                "enabled": True,
                "directory": str(resolved_video_dir.resolve()),
                "fps": float(video_fps),
                "completed_sequences": 1,
                "video": str(
                    resolved_video_dir / str(cast(dict[str, object], video_report)["video"])
                ),
                "report": str(resolved_video_dir / "sequence_0001.json"),
            }
            if video_report is not None
            else {
                "enabled": False,
                "directory": None,
                "fps": float(video_fps),
                "completed_sequences": 0,
            }
        ),
        "official_assets": {
            "conf_dir": str(conf_dir),
            "task_config_sha256": _sha256(
                conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
            ),
            "annotation_sha256": _sha256(
                conf_dir / "annotations/new_playtable_validation.yaml"
            ),
        },
        "artifacts": {
            "actions": "actions.npz",
            "initial_static": "initial_static.png",
            "initial_wrist": "initial_wrist.png",
            "final_static": "final_static.png",
            "final_wrist": "final_wrist.png",
        },
    }
    atomic_json(output_dir / "result.json", result)
    return result


def _initial_state(value: Path | None) -> Mapping[str, object]:
    if value is None:
        return CALVIN_DEFAULT_INITIAL_STATE
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("initial-state JSON must be a mapping")
    return {str(key): item for key, item in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--num-sequences", type=int, default=CALVIN_OFFICIAL_SEQUENCES)
    parser.add_argument(
        "--max-subtask-steps",
        type=int,
        default=CALVIN_OFFICIAL_SUBTASK_STEPS,
    )
    parser.add_argument("--sequence-workers", type=int, default=8)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--allow-smoke-policy", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record frames from the same scored rollout (no replay)",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Video output directory (default: OUTPUT_DIR/videos)",
    )
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument(
        "--video-sequences",
        type=int,
        default=None,
        help="Record only the first N evaluated sequences (default: all)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Run one official task with the oracle instead of LH-MTLC sequences",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--initial-state-json", type=Path, default=None)
    parser.add_argument(
        "--execute-rows",
        type=int,
        default=1,
        help="Number of consecutive action-chunk rows to execute before replanning",
    )
    args = parser.parse_args()
    if args.task is not None and args.video_sequences not in (None, 1):
        parser.error("--video-sequences must be 1 for a targeted --task rollout")
    if args.task is None:
        result = evaluate_calvin(
            args.dataset_root,
            args.output_dir,
            endpoint=args.endpoint,
            timeout=args.timeout,
            num_sequences=args.num_sequences,
            max_subtask_steps=args.max_subtask_steps,
            sequence_workers=args.sequence_workers,
            show_gui=args.show_gui,
            debug=args.debug,
            execute_rows=args.execute_rows,
            allow_smoke_policy=args.allow_smoke_policy,
            record_video=bool(args.record_video),
            video_dir=args.video_dir,
            video_fps=float(args.video_fps),
            video_sequences=args.video_sequences,
        )
    else:
        result = evaluate_calvin_targeted(
            args.dataset_root,
            args.output_dir,
            task=str(args.task),
            instruction=args.instruction,
            initial_state=_initial_state(args.initial_state_json),
            endpoint=args.endpoint,
            timeout=args.timeout,
            max_steps=args.max_subtask_steps,
            execute_rows=args.execute_rows,
            show_gui=args.show_gui,
            allow_smoke_policy=args.allow_smoke_policy,
            record_video=bool(args.record_video),
            video_dir=args.video_dir,
            video_fps=float(args.video_fps),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CALVIN_EVALUATION_SCHEMA",
    "CALVIN_TARGETED_SCHEMA",
    "CalvinBridgeModel",
    "calvin_action_contract",
    "calvin_policy_observation",
    "evaluate_calvin",
    "evaluate_calvin_targeted",
    "execute_calvin_action",
    "validate_bridge_health",
]
