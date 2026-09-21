"""Evaluate ClearVLA on official LIBERO suites and fixed init states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.instructions import normalize_instruction

from .bridge import RemotePolicyClient, policy_observation
from .io import atomic_json

LIBERO_EVALUATION_SCHEMA = "clearvla-libero-official-evaluation-v1"
LIBERO_OFFICIAL_EPISODES_PER_TASK = 20
LIBERO_OFFICIAL_MAX_STEPS = 600
LIBERO_OFFICIAL_WARMUP_STEPS = 5
LIBERO_OFFICIAL_IMAGE_SIDE = 128
LIBERO_OFFICIAL_SEED = 10000
LIBERO_OFFICIAL_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)
LIBERO_POLICY_HORIZON = 24
LIBERO_ACTION_NAMES = (
    "dx",
    "dy",
    "dz",
    "droll",
    "dpitch",
    "dyaw",
    "gripper",
)
LIBERO_ACTION_LOW = np.full(7, -1.0, dtype=np.float32)
LIBERO_ACTION_HIGH = np.full(7, 1.0, dtype=np.float32)
# The normalized chart is an immutable outlet contract; callers must not be
# able to mutate the process-wide clipping bounds between episodes.
LIBERO_ACTION_LOW.setflags(write=False)
LIBERO_ACTION_HIGH.setflags(write=False)
LIBERO_GRIPPER_CODEC_BOUNDARY_SCOPE = (
    "profile_owned_full_horizon_encode_decode_loss_evaluation"
)
LIBERO_PROFILE_NAME = "libero_relative_7d_v1"


def _libero_video_panel(
    cv2: Any,
    image: object,
    *,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    """Convert one LIBERO RGB camera image into a fixed-size BGR panel."""

    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(
            "LIBERO video camera image must have shape [H,W,3] or [H,W,4]"
        )
    # Robosuite normally returns uint8 RGB.  Accept finite floating images as
    # a convenience for wrappers, while keeping the video boundary explicit.
    if np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all():
            raise ValueError("LIBERO video camera image contains non-finite values")
        scale = 255.0 if float(np.max(value)) <= 1.0 else 1.0
        value = np.rint(np.clip(value * scale, 0.0, 255.0)).astype(np.uint8)
    else:
        value = np.asarray(np.clip(value, 0, 255), dtype=np.uint8)
    value = np.ascontiguousarray(value[..., :3])
    resized = cv2.resize(value, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    panel = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (int(width) - 1, 27), (18, 18, 18), thickness=-1)
    cv2.putText(
        panel,
        str(title),
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(panel)


def _libero_video_frame(
    cv2: Any,
    observation: Mapping[str, Any],
    *,
    task_id: int,
    episode: int,
    instruction: str,
    step: int,
    max_steps: int,
    status: str,
) -> np.ndarray:
    """Build a side-by-side RGB observation frame with rollout metadata."""

    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO video observation must be a mapping")
    top = _libero_video_panel(
        cv2,
        observation.get("agentview_image"),
        width=320,
        height=320,
        title="agentview",
    )
    wrist = _libero_video_panel(
        cv2,
        observation.get("robot0_eye_in_hand_image"),
        width=320,
        height=320,
        title="eye-in-hand",
    )
    panels = np.concatenate((top, wrist), axis=1)
    canvas = np.full((390, panels.shape[1], 3), 20, dtype=np.uint8)
    canvas[: panels.shape[0]] = panels
    color = {
        "SUCCESS": (80, 235, 120),
        "FAILED": (60, 80, 235),
        "RUNNING": (235, 235, 235),
        "READY": (235, 210, 80),
    }.get(str(status), (235, 235, 235))
    cv2.putText(
        canvas,
        (
            f"task {int(task_id):04d}  episode {int(episode):03d}  "
            f"step {int(step):03d}/{int(max_steps):03d}  {str(status)}"
        ),
        (10, 342),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    # A single line is enough for the short official LIBERO instructions and
    # avoids changing the camera aspect ratio when a task has a long name.
    text = " ".join(str(instruction).split())
    max_chars = 112
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    cv2.putText(
        canvas,
        text,
        (10, 366),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return np.ascontiguousarray(canvas)


class _LiberoEpisodeVideoRecorder:
    """Stream one LIBERO episode to MP4 without retaining rollout frames."""

    def __init__(
        self,
        output_dir: Path,
        *,
        task_id: int,
        episode: int,
        instruction: str,
        max_steps: int,
        warmup_steps: int,
        fps: float,
        metadata: Mapping[str, object],
        cv2_module: Any | None = None,
    ) -> None:
        if cv2_module is None:
            try:
                import cv2 as cv2_module
            except ImportError as error:
                raise RuntimeError(
                    "LIBERO video recording requires OpenCV in the evaluator environment"
                ) from error
        self.cv2 = cv2_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = int(task_id)
        self.episode = int(episode)
        self.instruction = str(instruction)
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        self.fps = float(fps)
        self.metadata = dict(metadata)
        stem = f"task_{self.task_id:04d}_episode_{self.episode:03d}"
        self.video_path = self.output_dir / f"{stem}.mp4"
        self.partial_video_path = self.output_dir / f".{stem}.partial.mp4"
        self.report_path = self.output_dir / f"{stem}.json"
        self.partial_report_path = self.output_dir / f".{stem}.partial.json"
        for artifact in (
            self.video_path,
            self.partial_video_path,
            self.report_path,
            self.partial_report_path,
        ):
            if artifact.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing LIBERO video artifact: {artifact}"
                )
        self.writer: Any | None = None
        self.frame_count = 0
        self.last_observation: Mapping[str, Any] | None = None
        self.started = time.perf_counter()
        self.closed = False

    def _write_frame(self, observation: Mapping[str, Any], *, step: int, status: str) -> None:
        frame = _libero_video_frame(
            self.cv2,
            observation,
            task_id=self.task_id,
            episode=self.episode,
            instruction=self.instruction,
            step=int(step),
            max_steps=self.max_steps,
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
                raise RuntimeError(
                    f"could not open LIBERO video writer: {self.video_path}"
                )
        self.writer.write(frame)
        self.frame_count += 1
        self.last_observation = observation

    def capture_ready(self, observation: Mapping[str, Any]) -> None:
        """Record the post-warmup state before the first policy action."""

        if self.closed:
            raise RuntimeError("cannot record a closed LIBERO episode")
        self._write_frame(observation, step=0, status="READY")

    def capture_step(self, observation: Mapping[str, Any], *, step: int) -> None:
        if self.closed:
            raise RuntimeError("cannot record a closed LIBERO episode")
        self._write_frame(observation, step=int(step), status="RUNNING")

    def finish(
        self,
        *,
        success: bool,
        steps: int,
        action_audit: Mapping[str, object],
        hold_seconds: float = 0.5,
    ) -> dict[str, object]:
        if self.closed:
            raise RuntimeError("LIBERO episode video was already closed")
        if self.last_observation is None or self.writer is None:
            raise RuntimeError("LIBERO episode completed without video frames")
        status = "SUCCESS" if bool(success) else "FAILED"
        hold_frames = max(1, int(round(self.fps * float(hold_seconds))))
        for _ in range(hold_frames):
            self._write_frame(self.last_observation, step=int(steps), status=status)
        self.writer.release()
        self.writer = None
        if not self.partial_video_path.is_file():
            raise RuntimeError("LIBERO video writer did not produce an MP4")
        os.replace(self.partial_video_path, self.video_path)
        report: dict[str, object] = {
            "schema": "clearvla-libero-inline-video-episode-v1",
            "task_id": self.task_id,
            "episode": self.episode,
            "instruction": self.instruction,
            "steps": int(steps),
            "warmup_steps": self.warmup_steps,
            "success": bool(success),
            "frame_count": int(self.frame_count),
            "fps": self.fps,
            "resolution": [640, 390],
            "video": self.video_path.name,
            "action_audit": dict(action_audit),
            "metadata": self.metadata,
            "completed": True,
            "elapsed_seconds": float(time.perf_counter() - self.started),
        }
        atomic_json(self.report_path, report)
        self.closed = True
        return report

    def abort(self, error: BaseException) -> None:
        """Release the writer and retain an explicitly marked partial artifact."""

        if self.closed:
            return
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.partial_video_path.is_file():
            aborted_video = self.output_dir / f"{self.video_path.stem}.aborted.mp4"
            try:
                os.replace(self.partial_video_path, aborted_video)
            except OSError:
                aborted_video = self.partial_video_path
        else:
            aborted_video = None
        atomic_json(
            self.partial_report_path,
            {
                "schema": "clearvla-libero-inline-video-episode-v1",
                "task_id": self.task_id,
                "episode": self.episode,
                "completed": False,
                "frame_count": int(self.frame_count),
                "video": None if aborted_video is None else aborted_video.name,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        self.closed = True


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"LIBERO bridge health {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sha256_text(value: object, *, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"LIBERO bridge health {name} must be a SHA-256 digest")
    return text


def _strict_int(value: object, *, name: str, minimum: int | None = None) -> int:
    """Parse JSON integer fields without accepting booleans or lossy floats."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"LIBERO {name} must be an integer")
    result = int(value)
    if minimum is not None and result < int(minimum):
        raise ValueError(f"LIBERO {name} must be at least {minimum}")
    return result


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LIBERO {name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"LIBERO {name} must be finite")
    return result


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"LIBERO {name} must be a sequence")
    return tuple(value)


def _strict_index_sequence(value: object, *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for index, item in enumerate(_sequence(value, name=name)):
        result.append(_strict_int(item, name=f"{name}[{index}]"))
    return tuple(result)


def _strict_numeric_vector(
    value: object,
    *,
    name: str,
    size: int,
) -> np.ndarray:
    rows = _sequence(value, name=name)
    if len(rows) != int(size):
        raise ValueError(f"LIBERO {name} must contain {size} values")
    return np.asarray(
        [_strict_float(item, name=f"{name}[{index}]") for index, item in enumerate(rows)],
        dtype=np.float64,
    )


def _strict_count_vector(
    value: object,
    *,
    name: str,
    size: int,
    maximum: int,
) -> np.ndarray:
    rows = _sequence(value, name=name)
    if len(rows) != int(size):
        raise ValueError(f"LIBERO {name} must contain {size} values")
    counts = np.asarray(
        [
            _strict_int(item, name=f"{name}[{index}]", minimum=0)
            for index, item in enumerate(rows)
        ],
        dtype=np.int64,
    )
    if np.any(counts > int(maximum)):
        raise ValueError(f"LIBERO {name} exceeds rollout steps")
    return counts


def _registered_libero_profile_digest() -> str:
    try:
        return resolve_action_state_profile(LIBERO_PROFILE_NAME).digest()
    except (KeyError, ValueError) as error:
        raise RuntimeError("cannot resolve the registered LIBERO action profile") from error


def _validate_libero_profile(
    profile: Mapping[str, Any],
    *,
    action_gripper_indices: object,
) -> None:
    """Validate the profile metadata emitted by a formal policy bridge."""

    if profile.get("name") != LIBERO_PROFILE_NAME:
        raise ValueError("LIBERO evaluation requires the libero_relative_7d_v1 action chart")
    if profile.get("sha256") != _registered_libero_profile_digest():
        raise ValueError("LIBERO bridge profile digest differs from the registered chart")
    for field, expected in (
        ("source_action_dim", 7),
        ("source_state_dim", 7),
        ("action_chart", "libero_normalized_osc_pose_6d_plus_continuous_gripper"),
        ("state_chart", "libero_eef_6d_plus_gripper_opening_width"),
        ("gripper_transition_boundary", "previous_command"),
    ):
        if field in {"source_action_dim", "source_state_dim"}:
            actual = _strict_int(profile.get(field), name=f"profile.{field}")
        else:
            actual = profile.get(field)
        if actual != expected:
            raise ValueError(
                f"LIBERO bridge profile {field}={actual!r} differs from {expected!r}"
            )
    for field in ("action_indices", "state_indices"):
        indices = _strict_index_sequence(profile.get(field), name=f"profile.{field}")
        if indices != tuple(range(7)):
            raise ValueError(f"LIBERO bridge profile {field} must be native order [0..6]")
    profile_grippers = _strict_index_sequence(
        profile.get("gripper_indices"), name="profile.gripper_indices"
    )
    action_grippers = _strict_index_sequence(
        action_gripper_indices, name="deployment action gripper_indices"
    )
    if profile_grippers != (6,) or action_grippers != (6,):
        raise ValueError("LIBERO bridge gripper indices must be [6]")
    scales = _sequence(
        profile.get("state_to_action_scale"),
        name="profile.state_to_action_scale",
    )
    try:
        numeric_scales = tuple(float(value) for value in scales)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("LIBERO bridge profile state_to_action_scale is not numeric") from error
    if numeric_scales != (1.0,) * 7:
        raise ValueError("LIBERO bridge profile state_to_action_scale must be all ones")


def libero_action_contract() -> dict[str, object]:
    """Return the native LIBERO command contract used by the evaluator."""

    return {
        "profile": LIBERO_PROFILE_NAME,
        "profile_sha256": _registered_libero_profile_digest(),
        "action_dim": 7,
        "state_dim": 7,
        "names": list(LIBERO_ACTION_NAMES),
        "normalized_low": LIBERO_ACTION_LOW.tolist(),
        "normalized_high": LIBERO_ACTION_HIGH.tolist(),
        "arm_semantics": "normalized OSC_POSE xyz/axis-angle relative command",
        "gripper_semantics": "normalized continuous gripper command",
        "gripper_transition_boundary": "previous_command",
        "action_state_semantics": "previous clipped native action; reset row is zero",
        "prediction_horizon": LIBERO_POLICY_HORIZON,
        "receding_horizon_execute_rows": 1,
        "execution_clipping": "componentwise clip to normalized [-1,1] before env.step",
        "source": "official LIBERO robosuite OSC_POSE controller",
    }


def execute_libero_action(raw: np.ndarray) -> np.ndarray:
    """Clip one policy row at the LIBERO normalized-controller boundary.

    The value returned here is the command that actually crosses the
    environment boundary.  Callers must use this same value as the next
    ``action_state``; retaining an unclipped proposal would make the causal
    history disagree with the executed trajectory.
    """

    value = np.asarray(raw, dtype=np.float32)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("LIBERO policy action must be one finite [7] row")
    return np.clip(value, LIBERO_ACTION_LOW, LIBERO_ACTION_HIGH).astype(np.float32)


def _validated_step_observation(env: Any, action: np.ndarray) -> tuple[Any, tuple[Any, ...]]:
    """Run one environment step and validate only its transport shape.

    LIBERO's five zero-action physics warmup steps are deliberately outside
    the policy rollout.  In particular, a wrapper's ``done``/success flag
    during warmup must not become a policy result or suppress the first real
    policy action.  Keep tuple and observation validation shared with the
    normal step path while allowing that caller to discard all evaluator
    flags.
    """

    result = env.step(action)
    if not isinstance(result, (tuple, list)) or len(result) not in (4, 5):
        raise ValueError(
            "LIBERO environment step must return a legacy 4-tuple or Gymnasium 5-tuple"
        )
    observation = result[0]
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO environment returned a non-mapping observation")
    return observation, tuple(result)


def _step_observation_only(env: Any, action: np.ndarray) -> Any:
    """Advance physics without allowing warmup success to end an episode."""

    observation, _ = _validated_step_observation(env, action)
    return observation


def _step_with_success(env: Any, action: np.ndarray) -> tuple[Any, bool]:
    """Step a legacy or Gymnasium LIBERO env and read task success explicitly.

    The official LIBERO BDDL environment currently returns a four-tuple whose
    ``done`` slot is ``_check_success()``.  That is an implementation detail,
    not a safe generic evaluator contract: robosuite's base environment can
    also set ``done`` at its horizon, and newer wrappers may return a
    terminated/truncated five-tuple.  Prefer LIBERO's public ``check_success``
    method whenever it is available and use the step flag only as a fallback
    for minimal test doubles or older wrappers.
    """

    observation, result = _validated_step_observation(env, action)
    if len(result) == 4:
        step_done = bool(result[2])
    else:
        step_done = bool(result[2]) or bool(result[3])
    checker = getattr(env, "check_success", None)
    if not callable(checker):
        # Released LIBERO versions expose the task predicate as the private
        # ``_check_success`` method, while newer wrappers add a public alias.
        # Both are evaluator-only and are preferred over a generic horizon
        # termination flag when available.
        checker = getattr(env, "_check_success", None)
    success = bool(checker()) if callable(checker) else step_done
    return observation, success


def _set_init_state_observation(env: Any, state: np.ndarray) -> Mapping[str, Any]:
    """Apply one fixed LIBERO init state and return the post-set observation.

    The official API has shipped both a returning ``set_init_state`` and a
    mutating version followed by ``get_observation``.  Accepting both keeps
    the evaluator pinned to the same physical state without relying on a
    stale observation returned by ``reset``.
    """

    setter = getattr(env, "set_init_state", None)
    if not callable(setter):
        raise ValueError("LIBERO environment lacks set_init_state")
    returned = setter(state)
    if returned is None:
        getter = getattr(env, "get_observation", None)
        if not callable(getter):
            # A few robosuite releases keep the same accessor private while
            # exposing a mutating ``set_init_state``.  It is still an
            # observation-only read and does not expose evaluator metrics.
            getter = getattr(env, "_get_observation", None)
        if not callable(getter):
            raise ValueError(
                "LIBERO set_init_state returned no observation and the environment "
                "has no get_observation method"
            )
        returned = getter()
    if not isinstance(returned, Mapping):
        raise ValueError("LIBERO set_init_state did not produce a mapping observation")
    return returned


def validate_libero_bridge_health(
    health: Mapping[str, Any],
    *,
    allow_smoke_policy: bool = False,
) -> None:
    """Fail closed unless the bridge exposes the LIBERO outlet contract.

    A loopback HTTP ``200`` only proves that a process is listening.  Formal
    LIBERO scores require the process to be the checkpoint-backed policy with
    the two-camera ordering, relative-command arm adapter and continuous
    gripper boundary that the converted data owns.  The explicit smoke escape
    is retained for transport/environment checks and is never implicit.
    """

    if not isinstance(health, Mapping):
        raise ValueError("LIBERO bridge health must be a mapping")
    if health.get("status") != "ok":
        raise RuntimeError(f"ClearVLA bridge is unhealthy: {dict(health)}")
    mode = str(health.get("mode", ""))
    if mode == "smoke-zero":
        if allow_smoke_policy:
            return
        raise RuntimeError("formal LIBERO evaluation refuses the smoke-zero policy")
    if mode != "formal-checkpoint":
        raise RuntimeError(f"unknown ClearVLA bridge mode {mode!r}")

    # The protocol field is intentionally checked only for a formal policy.
    # This keeps the explicit smoke lane useful with tiny test doubles while
    # ensuring a real score cannot be produced by an old one-way bridge.
    protocol = _mapping(health.get("protocol"), name="protocol")
    if protocol.get("observe_only") is not True:
        raise ValueError("LIBERO bridge must expose the observe-only protocol")
    try:
        protocol_version = _strict_int(protocol.get("version", 0), name="bridge protocol version")
    except ValueError as error:
        raise ValueError("LIBERO bridge protocol version is invalid") from error
    if protocol_version < 2:
        raise ValueError("LIBERO bridge protocol version 2 or newer is required")

    deployment = _mapping(health.get("deployment"), name="deployment")
    observation = _mapping(deployment.get("observation"), name="observation")
    action = _mapping(deployment.get("action"), name="action")
    if tuple(observation.get("camera_names", ())) != ("top", "wrist"):
        raise ValueError("LIBERO bridge camera order must be top,wrist")
    if tuple(observation.get("visual_offsets", ())) != (-8, -4, 0):
        raise ValueError("LIBERO bridge visual history differs from the formal policy")
    if tuple(observation.get("state_offsets", ())) != (-8, -4, 0):
        raise ValueError("LIBERO bridge state history differs from the formal policy")
    if tuple(observation.get("executed_action_offsets", ())) != (
        -24,
        -16,
        -12,
        -8,
        -6,
        -4,
        -2,
        -1,
    ):
        raise ValueError("LIBERO bridge executed-action history differs")
    profile = _mapping(action.get("data_profile"), name="action.data_profile")
    _validate_libero_profile(
        profile,
        action_gripper_indices=action.get("gripper_indices"),
    )
    normalizers = _mapping(action.get("normalizers"), name="action.normalizers")
    if normalizers.get("mode") != "zscore":
        raise ValueError("LIBERO bridge normalizers must use zscore")
    # Normalizer identity is part of the deployed outlet chart.  A bridge
    # with the right model/profile but a different affine action or state
    # transform is not the same evaluation process.
    _sha256_text(
        normalizers.get("action_sha256"),
        name="action.normalizers.action_sha256",
    )
    _sha256_text(
        normalizers.get("state_sha256"),
        name="action.normalizers.state_sha256",
    )
    if str(action.get("gripper_output_mode", "")) != "continuous":
        raise ValueError("LIBERO evaluation requires continuous gripper output")
    if str(action.get("arm_flow_mode", "")) != "relative_command_adapter":
        raise ValueError(
            "LIBERO evaluation requires the relative-command action adapter"
        )
    if str(profile.get("gripper_transition_boundary", "")) != "previous_command":
        raise ValueError(
            "LIBERO evaluation requires a previous-command gripper boundary"
        )
    if str(action.get("continuous_gripper_codec_boundary", "")) != "previous_command":
        raise ValueError("LIBERO deployment gripper boundary is not previous_command")
    if str(action.get("continuous_gripper_codec_boundary_scope", "")) != (
        LIBERO_GRIPPER_CODEC_BOUNDARY_SCOPE
    ):
        raise ValueError("LIBERO deployment gripper codec boundary scope is stale")
    for field, expected in (("state_dim", 7), ("action_dim", 7)):
        actual = _strict_int(observation.get(field), name=f"deployment observation {field}")
        if actual != expected:
            raise ValueError(
                f"LIBERO deployment observation {field} must be {expected}, got {actual}"
            )
    names = _sequence(action.get("names"), name="deployment action names")
    if names != LIBERO_ACTION_NAMES:
        raise ValueError("LIBERO deployment action names/order is stale")
    for field, expected in (
        ("normalized_low", tuple(float(value) for value in LIBERO_ACTION_LOW)),
        ("normalized_high", tuple(float(value) for value in LIBERO_ACTION_HIGH)),
    ):
        values = _sequence(action.get(field), name=f"deployment action {field}")
        try:
            numeric = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"LIBERO deployment action {field} is not numeric") from error
        if numeric != expected:
            raise ValueError(f"LIBERO deployment action {field} differs from [-1,1]")
    try:
        execute_rows = _strict_int(
            action.get("receding_horizon_execute_rows", 0),
            name="deployment execute rows",
        )
        prediction_horizon = _strict_int(
            action.get("prediction_horizon", 0),
            name="deployment prediction horizon",
        )
    except ValueError as error:
        raise ValueError("LIBERO deployment action horizon fields are invalid") from error
    if execute_rows != 1:
        raise ValueError("LIBERO deployment must execute exactly one predicted row")
    if prediction_horizon != 24:
        raise ValueError("LIBERO deployment prediction horizon must be 24")


def libero_bridge_identity(
    health: Mapping[str, Any],
    *,
    allow_smoke_policy: bool = False,
) -> dict[str, str]:
    """Extract the immutable policy identity used by evaluation resume.

    Paths and mutable health counters are intentionally excluded.  A result
    file may be resumed after relocating the checkpoint or restarting the
    bridge, but not after replacing the checkpoint, graph ABI, outlet chart or
    its affine normalizers.
    """

    validate_libero_bridge_health(
        health,
        allow_smoke_policy=allow_smoke_policy,
    )
    if str(health.get("mode", "")) == "smoke-zero":
        return {"mode": "smoke-zero"}
    deployment = _mapping(health.get("deployment"), name="deployment")
    checkpoint = _mapping(deployment.get("checkpoint"), name="checkpoint")
    architecture = _mapping(deployment.get("architecture"), name="architecture")
    action = _mapping(deployment.get("action"), name="action")
    profile = _mapping(action.get("data_profile"), name="action.data_profile")
    checkpoint_sha256 = _sha256_text(
        checkpoint.get("sha256"), name="checkpoint.sha256"
    )
    graph_config_sha256 = _sha256_text(
        architecture.get("graph_config_sha256"),
        name="architecture.graph_config_sha256",
    )
    manifest_digest = _sha256_text(
        architecture.get("manifest_digest"),
        name="architecture.manifest_digest",
    )
    profile_sha256 = _sha256_text(profile.get("sha256"), name="data_profile.sha256")
    normalizers = _mapping(action.get("normalizers"), name="action.normalizers")
    action_normalizer_sha256 = _sha256_text(
        normalizers.get("action_sha256"),
        name="action.normalizers.action_sha256",
    )
    state_normalizer_sha256 = _sha256_text(
        normalizers.get("state_sha256"),
        name="action.normalizers.state_sha256",
    )
    # The digest is not merely copied metadata: it must identify the profile
    # registered by this source tree.  This catches a forged health response
    # before it can be used to resume a score file.
    expected_profile_sha256 = _registered_libero_profile_digest()
    if profile_sha256 != expected_profile_sha256:
        raise ValueError(
            "LIBERO bridge profile digest differs from the registered action chart"
        )
    return {
        "mode": "formal-checkpoint",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_digest": manifest_digest,
        "graph_config_sha256": graph_config_sha256,
        "profile_sha256": profile_sha256,
        "action_normalizer_sha256": action_normalizer_sha256,
        "state_normalizer_sha256": state_normalizer_sha256,
    }


def libero_policy_observation(
    observation: Mapping[str, Any],
    previous_action: np.ndarray,
    *,
    quat_to_axisangle: Callable[[np.ndarray], np.ndarray],
    expected_image_side: int | None = None,
):
    """Project a LIBERO observation to its exact demonstration state chart."""

    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO observation must be a mapping")

    def vector(name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in observation:
            raise ValueError(f"LIBERO observation lacks {name!r}")
        try:
            value = np.asarray(observation[name], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"LIBERO observation {name!r} is not numeric") from error
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(
                f"LIBERO observation {name!r} must be finite with shape {shape}, "
                f"got {value.shape}"
            )
        return value

    position = vector("robot0_eef_pos", (3,))
    quaternion = vector("robot0_eef_quat", (4,))
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-8:
        raise ValueError("LIBERO observation quaternion has zero norm")
    # MuJoCo normally keeps this quaternion unit length, but wrappers can
    # expose a slightly drifted value.  Normalize at the native observation
    # boundary so the axis-angle converter receives a well-defined rotation;
    # a zero/non-finite quaternion remains a hard ingress error above.
    quaternion = quaternion / quaternion_norm
    try:
        orientation = np.asarray(quat_to_axisangle(quaternion), dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("LIBERO quaternion-to-axis-angle conversion failed") from error
    gripper = vector("robot0_gripper_qpos", (2,))
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or orientation.shape != (3,)
        or gripper.shape != (2,)
        or not np.isfinite(orientation).all()
    ):
        raise ValueError("LIBERO EEF/gripper observation shapes changed")
    opening = np.asarray([np.abs(gripper).sum()], dtype=np.float32)
    if "agentview_image" not in observation or "robot0_eye_in_hand_image" not in observation:
        raise ValueError("LIBERO observation lacks both required RGB camera keys")
    top = np.array(observation["agentview_image"], copy=True)
    wrist = np.array(observation["robot0_eye_in_hand_image"], copy=True)
    if expected_image_side is not None:
        side = _strict_int(
            expected_image_side,
            name="expected image side",
            minimum=1,
        )
        expected_shape = (side, side, 3)
        if top.shape != expected_shape or wrist.shape != expected_shape:
            raise ValueError(
                "LIBERO evaluator camera shapes differ from the requested image_side: "
                f"top={top.shape}, wrist={wrist.shape}, expected={expected_shape}"
            )
    state = np.concatenate((position, orientation, opening)).astype(np.float32)
    return policy_observation(
        # Own the arrays that cross the process boundary.  Some HTTP/client
        # implementations reuse or mutate their request buffers; a shared
        # view here could otherwise corrupt the next action-state/history row.
        top=top,
        wrist=wrist,
        state=state.copy(),
        action_state=np.array(previous_action, dtype=np.float32, copy=True),
    )


class LiberoBridgePolicy:
    """Receding-horizon bridge adapter for normalized OSC_POSE actions."""

    def __init__(
        self,
        client: RemotePolicyClient,
        *,
        quat_to_axisangle: Callable[[np.ndarray], np.ndarray],
        expected_image_side: int | None = None,
    ) -> None:
        self.client = client
        self.quat_to_axisangle = quat_to_axisangle
        self.expected_image_side = expected_image_side
        self.reset()

    def reset(self) -> None:
        self._previous_action = np.zeros(7, dtype=np.float32)
        self._pending_reset = True
        self._raw_actions: list[np.ndarray] = []
        self._executed_actions: list[np.ndarray] = []
        self._action_state_inputs: list[np.ndarray] = []

    def step(self, observation: Mapping[str, Any], instruction: str) -> np.ndarray:
        if not isinstance(instruction, str):
            raise ValueError("LIBERO rollout instruction must be non-empty text")
        try:
            normalized_instruction = normalize_instruction(instruction)
        except (TypeError, ValueError) as error:
            raise ValueError("LIBERO rollout instruction must be non-empty text") from error
        if not normalized_instruction:
            raise ValueError("LIBERO rollout instruction must be non-empty")
        policy_input = libero_policy_observation(
            observation,
            self._previous_action,
            quat_to_axisangle=self.quat_to_axisangle,
            expected_image_side=self.expected_image_side,
        )
        expected_action_state = (
            np.zeros(7, dtype=np.float32)
            if self._pending_reset
            else self._previous_action
        )
        if not np.array_equal(
            np.asarray(policy_input.action_state, dtype=np.float32),
            expected_action_state,
        ):
            raise AssertionError(
                "LIBERO action_state input is not the reset zero/previous executed action"
            )
        chunk = self.client.act(
            policy_input,
            normalized_instruction,
            reset=self._pending_reset,
        )
        raw_chunk = np.asarray(chunk, dtype=np.float32)
        if raw_chunk.shape != (LIBERO_POLICY_HORIZON, 7):
            raise ValueError(
                "LIBERO bridge must return exactly a [24,7] action chunk; "
                f"got {raw_chunk.shape}"
            )
        if not np.isfinite(raw_chunk).all():
            raise ValueError("LIBERO bridge returned a non-finite action chunk")
        raw_action = raw_chunk[0].copy()
        action = execute_libero_action(raw_action)
        self._raw_actions.append(raw_action)
        self._executed_actions.append(action.copy())
        self._action_state_inputs.append(expected_action_state.copy())
        self._previous_action = action.copy()
        self._pending_reset = False
        return action

    @property
    def previous_action(self) -> np.ndarray:
        """Return the last clipped command used as the next action-state."""

        return self._previous_action.copy()

    def action_audit(self) -> dict[str, object]:
        """Summarize proposal/execution clipping at the LIBERO boundary."""

        if not self._raw_actions:
            return {
                "steps": 0,
                "action_names": list(LIBERO_ACTION_NAMES),
                "raw_oob_row_count": 0,
                "clipped_row_count": 0,
                "action_state_matches_executed": True,
                "action_state_input_count": 0,
            }
        raw = np.stack(self._raw_actions).astype(np.float32)
        executed = np.stack(self._executed_actions).astype(np.float32)
        if raw.shape != executed.shape or raw.shape[1:] != (7,):
            raise AssertionError("LIBERO proposal/executed action audit lost alignment")
        action_state_inputs = np.stack(self._action_state_inputs).astype(np.float32)
        if action_state_inputs.shape != executed.shape:
            raise AssertionError("LIBERO action-state input audit lost alignment")
        expected_inputs = np.concatenate(
            (np.zeros((1, 7), dtype=np.float32), executed[:-1]),
            axis=0,
        )
        if not np.array_equal(action_state_inputs, expected_inputs):
            raise AssertionError(
                "LIBERO action-state inputs do not match reset zero/previous executed actions"
            )
        if not np.isfinite(executed).all() or not np.all(
            (executed >= LIBERO_ACTION_LOW) & (executed <= LIBERO_ACTION_HIGH)
        ):
            raise AssertionError("LIBERO executed action escaped the normalized bounds")
        return {
            "steps": int(executed.shape[0]),
            "action_names": list(LIBERO_ACTION_NAMES),
            "raw_min": raw.min(axis=0).tolist(),
            "raw_max": raw.max(axis=0).tolist(),
            "raw_oob_count_per_dim": (
                (raw < LIBERO_ACTION_LOW).sum(axis=0)
                + (raw > LIBERO_ACTION_HIGH).sum(axis=0)
            ).tolist(),
            "raw_oob_row_count": int(
                np.any((raw < LIBERO_ACTION_LOW) | (raw > LIBERO_ACTION_HIGH), axis=1).sum()
            ),
            "executed_min": executed.min(axis=0).tolist(),
            "executed_max": executed.max(axis=0).tolist(),
            "executed_abs_max": np.abs(executed).max(axis=0).tolist(),
            "clipped_row_count": int(np.any(raw != executed, axis=1).sum()),
            "action_state_matches_executed": bool(
                np.array_equal(self._previous_action, executed[-1])
            ),
            "action_state_input_count": int(action_state_inputs.shape[0]),
        }


def _validate_rollout_row(
    value: object,
    *,
    expected_episode: int,
    episodes_per_task: int,
    max_steps: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("existing LIBERO rollout row must be a mapping")
    row = cast(Mapping[str, Any], value)
    episode = _strict_int(row.get("episode"), name="rollout episode", minimum=0)
    if episode != int(expected_episode) or episode >= int(episodes_per_task):
        raise ValueError("existing LIBERO rollout episode index is invalid")
    init_state = _strict_int(row.get("init_state"), name="rollout init_state", minimum=0)
    if init_state != int(expected_episode):
        raise ValueError("existing LIBERO rollout fixed init-state index is invalid")
    steps = _strict_int(row.get("steps"), name="rollout steps", minimum=1)
    if steps > int(max_steps):
        raise ValueError("existing LIBERO rollout exceeds max_steps")
    success = row.get("success")
    if not isinstance(success, bool):
        raise ValueError("existing LIBERO rollout success must be boolean")
    audit = row.get("action_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("existing LIBERO rollout action_audit must be a mapping")
    audit_steps = _strict_int(
        audit.get("steps"), name="rollout action_audit steps", minimum=1
    )
    if audit_steps != steps:
        raise ValueError(
            "existing LIBERO rollout action audit does not match executed steps"
        )
    action_state_inputs = _strict_int(
        audit.get("action_state_input_count"),
        name="rollout action_audit action_state_input_count",
        minimum=1,
    )
    if action_state_inputs != steps:
        raise ValueError(
            "existing LIBERO rollout action-state input count does not match steps"
        )
    if audit.get("action_state_matches_executed") is not True:
        raise ValueError(
            "existing LIBERO rollout action_state does not match executed action"
        )
    if tuple(_sequence(audit.get("action_names"), name="rollout action names")) != LIBERO_ACTION_NAMES:
        raise ValueError("existing LIBERO rollout action names/order is invalid")
    raw_min = _strict_numeric_vector(
        audit.get("raw_min"), name="rollout action_audit raw_min", size=7
    )
    raw_max = _strict_numeric_vector(
        audit.get("raw_max"), name="rollout action_audit raw_max", size=7
    )
    executed_min = _strict_numeric_vector(
        audit.get("executed_min"),
        name="rollout action_audit executed_min",
        size=7,
    )
    executed_max = _strict_numeric_vector(
        audit.get("executed_max"),
        name="rollout action_audit executed_max",
        size=7,
    )
    executed_abs_max = _strict_numeric_vector(
        audit.get("executed_abs_max"),
        name="rollout action_audit executed_abs_max",
        size=7,
    )
    if np.any(raw_min > raw_max) or np.any(executed_min > executed_max):
        raise ValueError("existing LIBERO rollout action audit min/max values are invalid")
    if np.any(executed_min < LIBERO_ACTION_LOW) or np.any(
        executed_max > LIBERO_ACTION_HIGH
    ):
        raise ValueError("existing LIBERO rollout executed actions exceed [-1,1]")
    expected_abs_max = np.maximum(np.abs(executed_min), np.abs(executed_max))
    if not np.array_equal(executed_abs_max, expected_abs_max):
        raise ValueError("existing LIBERO rollout executed_abs_max is inconsistent")
    raw_oob_per_dim = _strict_count_vector(
        audit.get("raw_oob_count_per_dim"),
        name="rollout action_audit raw_oob_count_per_dim",
        size=7,
        maximum=steps,
    )
    raw_oob_rows = _strict_int(
        audit.get("raw_oob_row_count"),
        name="rollout action_audit raw_oob_row_count",
        minimum=0,
    )
    clipped_rows = _strict_int(
        audit.get("clipped_row_count"),
        name="rollout action_audit clipped_row_count",
        minimum=0,
    )
    if raw_oob_rows > steps or clipped_rows > steps or raw_oob_rows != clipped_rows:
        raise ValueError("existing LIBERO rollout clipping counts are inconsistent")
    inferred_oob_dims = (raw_min < LIBERO_ACTION_LOW) | (raw_max > LIBERO_ACTION_HIGH)
    if not np.array_equal(raw_oob_per_dim > 0, inferred_oob_dims):
        raise ValueError("existing LIBERO rollout raw out-of-bounds audit is inconsistent")
    if (raw_oob_rows > 0) != bool(np.any(raw_oob_per_dim > 0)):
        raise ValueError("existing LIBERO rollout raw out-of-bounds rows are inconsistent")
    return {
        "episode": episode,
        "init_state": init_state,
        "steps": steps,
        "success": success,
        "action_audit": dict(audit),
    }


def _validate_task_result(
    value: object,
    *,
    task_id: int,
    expected_name: str,
    expected_instruction: str,
    episodes_per_task: int,
    max_steps: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"existing LIBERO task {task_id} result must be a mapping")
    row = cast(Mapping[str, Any], value)
    name = row.get("name")
    instruction = row.get("instruction")
    if not isinstance(name, str) or name != expected_name:
        raise ValueError(f"existing LIBERO task {task_id} name differs from the suite")
    if not isinstance(instruction, str) or instruction.strip() != expected_instruction:
        raise ValueError(
            f"existing LIBERO task {task_id} instruction differs from the suite"
        )
    episodes = _strict_int(row.get("episodes"), name="task episodes", minimum=1)
    if episodes != int(episodes_per_task):
        raise ValueError(f"existing LIBERO task {task_id} episode count differs")
    successes = _strict_int(row.get("successes"), name="task successes", minimum=0)
    if successes > episodes:
        raise ValueError(f"existing LIBERO task {task_id} successes exceed episodes")
    success_rate = _strict_float(row.get("success_rate"), name="task success_rate")
    if not np.isclose(success_rate, successes / episodes, rtol=0.0, atol=1e-12):
        raise ValueError(f"existing LIBERO task {task_id} success_rate is inconsistent")
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != episodes:
        raise ValueError(f"existing LIBERO task {task_id} has invalid rollout rows")
    normalized_rollouts = [
        _validate_rollout_row(
            rollout,
            expected_episode=index,
            episodes_per_task=episodes,
            max_steps=max_steps,
        )
        for index, rollout in enumerate(rollouts)
    ]
    if sum(bool(rollout["success"]) for rollout in normalized_rollouts) != successes:
        raise ValueError(f"existing LIBERO task {task_id} success count is inconsistent")
    return {
        "name": name,
        "instruction": instruction,
        "successes": successes,
        "episodes": episodes,
        "success_rate": success_rate,
        "rollouts": normalized_rollouts,
    }


def _suite_task_metadata(task: Any, *, task_id: int) -> tuple[str, str]:
    raw_name = getattr(task, "name", None)
    raw_instruction = getattr(task, "language", None)
    if not isinstance(raw_instruction, str) or not raw_instruction.strip():
        # LIBERO's task registry used both attribute spellings across releases.
        # Prefer ``language`` but accept the explicit wording field when the
        # former is absent, without inventing text from a filename.
        raw_instruction = getattr(task, "language_instruction", None)
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError(f"LIBERO task {task_id} has no evaluator name")
    if not isinstance(raw_instruction, str) or not raw_instruction.strip():
        raise ValueError(f"LIBERO task {task_id} has no evaluator instruction")
    try:
        instruction = normalize_instruction(raw_instruction)
    except ValueError as error:
        raise ValueError(f"LIBERO task {task_id} has no evaluator instruction") from error
    return raw_name.strip(), instruction


def _file_sha256(path: Path, *, name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"LIBERO {name} is not a readable file: {path}") from error
    return digest.hexdigest()


def _fixed_init_state_digest(states: np.ndarray) -> str:
    """Hash the exact first-N fixed states in a relocation-stable chart."""

    canonical = np.ascontiguousarray(states, dtype="<f8")
    header = json.dumps(
        {"dtype": "float64-le", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _suite_task_assets(
    suite: Any,
    *,
    selected: Sequence[int],
    episodes_per_task: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Preflight and identify the benchmark assets that define each rollout."""

    assets: dict[int, dict[str, Any]] = {}
    identity_rows: list[dict[str, Any]] = []
    for task_id in selected:
        task = suite.get_task(task_id)
        name, instruction = _suite_task_metadata(task, task_id=task_id)
        raw_bddl_path = suite.get_task_bddl_file_path(task_id)
        if not isinstance(raw_bddl_path, (str, Path)):
            raise ValueError(f"LIBERO task {task_id} has no BDDL file path")
        bddl_path = Path(raw_bddl_path).expanduser()
        if not bddl_path.is_file():
            raise ValueError(
                f"LIBERO task {task_id} BDDL path is not a file: {bddl_path}"
            )
        bddl_sha256 = _file_sha256(bddl_path, name=f"task {task_id} BDDL")
        try:
            init_states = np.asarray(
                suite.get_task_init_states(task_id),
                dtype=np.float64,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"LIBERO task {task_id} fixed init states must be numeric"
            ) from error
        if init_states.ndim != 2 or int(init_states.shape[1]) <= 0:
            raise ValueError(
                f"LIBERO task {task_id} fixed init states must be [N,D], "
                f"got {init_states.shape}"
            )
        init_count = int(init_states.shape[0])
        if init_count < int(episodes_per_task):
            raise ValueError(
                f"LIBERO task {task_id} has {init_count} fixed init states, "
                f"fewer than the requested {episodes_per_task} episodes"
            )
        used_states = np.array(
            init_states[: int(episodes_per_task)],
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if not np.isfinite(used_states).all():
            raise ValueError(f"LIBERO task {task_id} fixed init states are non-finite")
        init_states_sha256 = _fixed_init_state_digest(used_states)
        assets[int(task_id)] = {
            "name": name,
            "instruction": instruction,
            "bddl_path": bddl_path,
            "init_states": used_states,
        }
        identity_rows.append(
            {
                "task_id": int(task_id),
                "name": name,
                "instruction": instruction,
                "bddl_file": bddl_path.name,
                "bddl_sha256": bddl_sha256,
                "init_state_shape": list(used_states.shape),
                "init_states_sha256": init_states_sha256,
            }
        )
    return assets, {"task_order_index": 0, "tasks": identity_rows}


def _validate_task_results(
    value: object,
    *,
    suite: Any,
    selected: Sequence[int],
    episodes_per_task: int,
    max_steps: int,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("existing LIBERO result has an invalid tasks mapping")
    allowed = {str(int(task_id)) for task_id in selected}
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        raise ValueError(f"existing LIBERO result contains unselected tasks: {unknown}")
    result: dict[str, dict[str, Any]] = {}
    for key, row in value.items():
        if not isinstance(key, str) or key not in allowed or not key.isdigit():
            raise ValueError("existing LIBERO result task keys are invalid")
        task_id = int(key)
        task = suite.get_task(task_id)
        expected_name, expected_instruction = _suite_task_metadata(task, task_id=task_id)
        result[key] = _validate_task_result(
            row,
            task_id=task_id,
            expected_name=expected_name,
            expected_instruction=expected_instruction,
            episodes_per_task=episodes_per_task,
            max_steps=max_steps,
        )
    return result


def _aggregate_success_rate(task_results: Mapping[str, Mapping[str, Any]]) -> float | None:
    if not task_results:
        return None
    total_successes = 0
    total_episodes = 0
    for row in task_results.values():
        total_successes += _strict_int(row.get("successes"), name="task successes", minimum=0)
        total_episodes += _strict_int(row.get("episodes"), name="task episodes", minimum=1)
    if total_episodes <= 0:
        raise ValueError("LIBERO task results contain no episodes")
    return total_successes / total_episodes


def _initial_payload(
    *,
    suite_name: str,
    task_ids: Sequence[int],
    episodes_per_task: int,
    max_steps: int,
    warmup_steps: int,
    seed: int,
    image_side: int,
    endpoint: str,
    official_protocol: bool,
    bridge_identity: Mapping[str, str],
    suite_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": LIBERO_EVALUATION_SCHEMA,
        "benchmark": "LIBERO",
        "suite": suite_name,
        "task_ids": [int(value) for value in task_ids],
        "episodes_per_task": int(episodes_per_task),
        "max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "seed": int(seed),
        "image_side": int(image_side),
        "bridge_endpoint": endpoint,
        "bridge_identity": dict(bridge_identity),
        "suite_identity": dict(suite_identity),
        "action_contract": libero_action_contract(),
        "official_protocol": bool(official_protocol),
        # ``official_protocol`` describes the requested protocol shape.  A
        # partial atomic checkpoint must remain unmistakable until every task
        # and the final bridge-identity check have completed.
        "complete": False,
        "completed_task_count": 0,
        "tasks": {},
    }


def evaluate_libero(
    suite_name: str,
    output: Path,
    *,
    endpoint: str,
    timeout: float,
    task_ids: Sequence[int] | None,
    episodes_per_task: int,
    max_steps: int,
    warmup_steps: int,
    seed: int,
    image_side: int,
    resume: bool,
    allow_smoke_policy: bool = False,
    record_video: bool = False,
    video_dir: Path | None = None,
    video_fps: float = 10.0,
    video_episodes: int | None = None,
) -> dict[str, Any]:
    output = Path(output).expanduser()
    # ``Path.exists()`` does not report a dangling symlink.  Refuse links
    # before either resume reads or the atomic result write, so an evaluator
    # cannot accidentally follow or replace an unrelated destination.
    if output.is_symlink():
        raise FileExistsError(f"refusing LIBERO evaluation output symlink: {output}")
    if output.exists() and not output.is_file():
        raise FileExistsError(
            f"refusing LIBERO evaluation output that is not a file: {output}"
        )
    if output.exists() and not resume:
        raise FileExistsError(f"LIBERO evaluation output already exists: {output}")
    try:
        episodes_per_task = _strict_int(
            episodes_per_task, name="episodes_per_task", minimum=1
        )
        max_steps = _strict_int(max_steps, name="max_steps", minimum=1)
        warmup_steps = _strict_int(warmup_steps, name="warmup_steps", minimum=0)
        image_side = _strict_int(image_side, name="image_side", minimum=1)
        seed = _strict_int(seed, name="seed", minimum=0)
    except ValueError as error:
        raise ValueError("LIBERO episode, step, image, and warmup bounds are invalid") from error
    if not isinstance(record_video, bool):
        raise ValueError("LIBERO record_video must be boolean")
    try:
        video_fps = float(video_fps)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("LIBERO video_fps must be finite and positive") from error
    if not np.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError("LIBERO video_fps must be finite and positive")
    if video_episodes is not None:
        video_episodes = _strict_int(
            video_episodes, name="video_episodes", minimum=1
        )
    if record_video:
        resolved_video_dir = (
            Path(video_dir).expanduser()
            if video_dir is not None
            else output.parent / f"{output.stem}_videos"
        )
    elif video_dir is not None or video_episodes is not None:
        raise ValueError(
            "LIBERO video_dir/video_episodes require record_video=True"
        )
    else:
        resolved_video_dir = None
    # ``max_steps`` bounds formal policy actions only.  The official five-step
    # physics warmup is a separate pre-roll and must remain legal even for a
    # deliberately tiny bounded smoke (for example, one policy step).
    if not np.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("LIBERO bridge timeout must be finite and positive")

    normalized_suite = str(suite_name).strip().lower()
    if normalized_suite == "libero_100":
        raise ValueError(
            "LIBERO_100 is an archive union, not an official evaluation suite; "
            "evaluate libero_90 and libero_10 separately"
        )
    if normalized_suite not in LIBERO_OFFICIAL_SUITES:
        raise ValueError(
            f"unknown LIBERO suite {suite_name!r}; choices={list(LIBERO_OFFICIAL_SUITES)}"
        )

    # Validate the policy process before importing/constructing the external
    # simulator.  This keeps a formal output directory from being associated
    # with a listening-but-wrong bridge and makes resume identity fail early.
    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_libero_bridge_health(
        health_before,
        allow_smoke_policy=allow_smoke_policy,
    )
    bridge_identity = libero_bridge_identity(
        health_before,
        allow_smoke_policy=allow_smoke_policy,
    )

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    if not isinstance(benchmark_map, Mapping):
        raise ValueError("LIBERO benchmark registry must be a mapping")
    if normalized_suite not in benchmark_map:
        raise ValueError(
            f"official LIBERO suite {normalized_suite!r} is absent from the installed registry"
        )
    suite = benchmark_map[normalized_suite](task_order_index=0)
    if task_ids is None:
        selected = list(range(suite.get_num_tasks()))
    else:
        try:
            selected = [
                _strict_int(value, name="task id", minimum=0) for value in task_ids
            ]
        except ValueError as error:
            raise ValueError("LIBERO task ids must be integer values") from error
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("LIBERO task ids must be unique and non-empty")
    if min(selected) < 0 or max(selected) >= suite.get_num_tasks():
        raise ValueError("LIBERO task id is outside the selected suite")
    task_assets, suite_identity = _suite_task_assets(
        suite,
        selected=selected,
        episodes_per_task=episodes_per_task,
    )
    official_protocol = (
        bridge_identity.get("mode") == "formal-checkpoint"
        and normalized_suite in LIBERO_OFFICIAL_SUITES
        and selected == list(range(suite.get_num_tasks()))
        and episodes_per_task == LIBERO_OFFICIAL_EPISODES_PER_TASK
        and max_steps == LIBERO_OFFICIAL_MAX_STEPS
        and warmup_steps == LIBERO_OFFICIAL_WARMUP_STEPS
        and image_side == LIBERO_OFFICIAL_IMAGE_SIDE
        and seed == LIBERO_OFFICIAL_SEED
    )
    expected = _initial_payload(
        suite_name=normalized_suite,
        task_ids=selected,
        episodes_per_task=episodes_per_task,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        image_side=image_side,
        endpoint=endpoint,
        official_protocol=official_protocol,
        bridge_identity=bridge_identity,
        suite_identity=suite_identity,
    )
    if output.exists():
        try:
            loaded = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing LIBERO result is unreadable JSON") from error
        if not isinstance(loaded, dict):
            raise ValueError("existing LIBERO result must be a JSON mapping")
        result = loaded
        if result.get("schema") != LIBERO_EVALUATION_SCHEMA:
            raise ValueError("existing LIBERO result schema differs")
        if result.get("benchmark") != "LIBERO":
            raise ValueError("existing LIBERO result benchmark differs")
        if result.get("bridge_identity") != bridge_identity:
            raise ValueError(
                "existing LIBERO result belongs to a different checkpoint, graph, "
                "or outlet profile"
            )
        comparable = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "tasks",
                "success_rate",
                "bridge_health_before",
                "bridge_health_after",
                "bridge_endpoint",
                "complete",
                "completed_task_count",
            }
        }
        expected_comparable = {
            key: value
            for key, value in expected.items()
            if key
            not in {
                "tasks",
                "success_rate",
                "bridge_endpoint",
                "complete",
                "completed_task_count",
            }
        }
        if comparable != expected_comparable:
            raise ValueError("existing LIBERO result uses a different evaluation protocol")
        task_results = _validate_task_results(
            result.get("tasks"),
            suite=suite,
            selected=selected,
            episodes_per_task=episodes_per_task,
            max_steps=max_steps,
        )
        completed_task_count = _strict_int(
            result.get("completed_task_count"),
            name="result completed_task_count",
            minimum=0,
        )
        if completed_task_count != len(task_results):
            raise ValueError("existing LIBERO completed-task count is inconsistent")
        complete = result.get("complete")
        if not isinstance(complete, bool):
            raise ValueError("existing LIBERO result complete flag must be boolean")
        if complete:
            if len(task_results) != len(selected):
                raise ValueError("existing LIBERO result claims incomplete tasks are complete")
            stored_health_after = result.get("bridge_health_after")
            if not isinstance(stored_health_after, Mapping):
                raise ValueError(
                    "existing complete LIBERO result lacks final bridge health"
                )
            if libero_bridge_identity(
                stored_health_after,
                allow_smoke_policy=allow_smoke_policy,
            ) != bridge_identity:
                raise ValueError(
                    "existing complete LIBERO result has inconsistent final bridge identity"
                )
        stored_rate = result.get("success_rate")
        if stored_rate is not None:
            expected_rate = _aggregate_success_rate(task_results)
            if not np.isclose(
                _strict_float(stored_rate, name="result success_rate"),
                0.0 if expected_rate is None else expected_rate,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("existing LIBERO result success_rate is inconsistent")
        result["tasks"] = task_results
    else:
        result = expected
        task_results = {}
    # Keep the payload and the mutable task accumulator aliased for a fresh
    # evaluation as well as a resume.  Without this assignment, a successful
    # fresh run wrote an empty ``tasks`` object while only the in-memory
    # accumulator contained the completed rollouts.
    result["tasks"] = task_results
    result["completed_task_count"] = len(task_results)
    result["bridge_endpoint"] = endpoint
    result["bridge_health_before"] = dict(health_before)
    bridge_policy = LiberoBridgePolicy(
        client,
        quat_to_axisangle=quat2axisangle,
        expected_image_side=image_side,
    )
    for task_id in selected:
        task_key = str(task_id)
        task_asset = task_assets[task_id]
        expected_name = str(task_asset["name"])
        instruction = str(task_asset["instruction"])
        if task_key in task_results:
            continue
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_asset["bddl_path"]),
            camera_heights=int(image_side),
            camera_widths=int(image_side),
        )
        try:
            env.seed(int(seed))
            init_states = np.asarray(task_asset["init_states"], dtype=np.float64)
            successes = 0
            episode_rows: list[dict[str, Any]] = []
            for episode in range(int(episodes_per_task)):
                env.reset()
                # Match the official evaluator's deterministic first-N fixed
                # state order.  Repeating a short state inventory would change
                # the benchmark population while preserving the same counts.
                init_index = episode
                initial = np.array(init_states[init_index], dtype=np.float64, copy=True)
                observation = _set_init_state_observation(env, initial)
                zero = np.zeros(7, dtype=np.float32)
                success = False
                video_recorder: _LiberoEpisodeVideoRecorder | None = None
                video_report: dict[str, object] | None = None
                if (
                    record_video
                    and resolved_video_dir is not None
                    and (video_episodes is None or episode < int(video_episodes))
                ):
                    video_recorder = _LiberoEpisodeVideoRecorder(
                        resolved_video_dir,
                        task_id=task_id,
                        episode=episode,
                        instruction=instruction,
                        max_steps=max_steps,
                        warmup_steps=warmup_steps,
                        fps=video_fps,
                        metadata={
                            "suite": normalized_suite,
                            "bddl_path": str(task_asset["bddl_path"]),
                            "init_state": int(init_index),
                            "image_side": int(image_side),
                            "bridge_identity": dict(bridge_identity),
                            "action_contract": libero_action_contract(),
                        },
                    )
                for _ in range(int(warmup_steps)):
                    # Official LIBERO uses these steps only to settle the
                    # simulator after applying the fixed init state.  A
                    # success/done flag raised by a wrapper here is not a
                    # policy result and must not skip the formal rollout.
                    # Pass a fresh row in case an environment wrapper mutates
                    # its action argument in place.
                    observation = _step_observation_only(env, zero.copy())
                if video_recorder is not None:
                    video_recorder.capture_ready(observation)
                bridge_policy.reset()
                steps = 0
                try:
                    while steps < int(max_steps) and not success:
                        steps += 1
                        action = bridge_policy.step(observation, instruction)
                        observation, step_success = _step_with_success(env, action)
                        success = success or step_success
                        if video_recorder is not None:
                            video_recorder.capture_step(observation, step=steps)
                    successes += int(success)
                    action_audit = bridge_policy.action_audit()
                    if int(action_audit.get("steps", -1)) != int(steps):
                        raise AssertionError(
                            "LIBERO action audit step count differs from environment steps"
                        )
                    if action_audit.get("action_state_matches_executed") is not True:
                        raise AssertionError(
                            "LIBERO action_state does not match the last executed action"
                        )
                    if video_recorder is not None:
                        video_report = video_recorder.finish(
                            success=success,
                            steps=steps,
                            action_audit=action_audit,
                        )
                    rollout_row: dict[str, object] = {
                        "episode": episode,
                        "init_state": init_index,
                        "steps": steps,
                        "success": success,
                        "action_audit": action_audit,
                    }
                    if video_report is not None:
                        rollout_row["video"] = video_report
                    episode_rows.append(rollout_row)
                except BaseException as error:
                    if video_recorder is not None:
                        video_recorder.abort(error)
                    raise
        finally:
            env.close()
        task_results[task_key] = {
            "name": expected_name,
            "instruction": instruction,
            "successes": successes,
            "episodes": int(episodes_per_task),
            "success_rate": successes / int(episodes_per_task),
            "rollouts": episode_rows,
        }
        result["complete"] = False
        result["completed_task_count"] = len(task_results)
        result["success_rate"] = _aggregate_success_rate(task_results)
        atomic_json(output, result)
    health_after = client.health()
    validate_libero_bridge_health(
        health_after,
        allow_smoke_policy=allow_smoke_policy,
    )
    if libero_bridge_identity(
        health_after,
        allow_smoke_policy=allow_smoke_policy,
    ) != bridge_identity:
        raise RuntimeError("LIBERO bridge identity changed during evaluation")
    if len(task_results) != len(selected):
        raise AssertionError("LIBERO evaluation ended with incomplete task results")
    result["bridge_health_after"] = dict(health_after)
    result["complete"] = True
    result["completed_task_count"] = len(task_results)
    result["success_rate"] = _aggregate_success_rate(task_results)
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="libero_10",
        choices=(
            *LIBERO_OFFICIAL_SUITES,
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=LIBERO_OFFICIAL_EPISODES_PER_TASK,
    )
    parser.add_argument("--max-steps", type=int, default=LIBERO_OFFICIAL_MAX_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=LIBERO_OFFICIAL_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=LIBERO_OFFICIAL_SEED)
    parser.add_argument("--image-side", type=int, default=LIBERO_OFFICIAL_IMAGE_SIDE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-smoke-policy",
        action="store_true",
        help="Allow the explicit smoke-zero bridge for transport/simulator checks only",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record side-by-side LIBERO camera MP4s for selected episodes",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for rollout MP4/JSON artifacts (requires --record-video)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="Frame rate for recorded rollout videos",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=None,
        help="Record only the first N fixed-init episodes per task",
    )
    args = parser.parse_args()
    result = evaluate_libero(
        args.suite,
        args.output,
        endpoint=args.endpoint,
        timeout=args.timeout,
        task_ids=args.task_ids,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        image_side=args.image_side,
        resume=args.resume,
        allow_smoke_policy=args.allow_smoke_policy,
        record_video=args.record_video,
        video_dir=args.video_dir,
        video_fps=args.video_fps,
        video_episodes=args.video_episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LIBERO_EVALUATION_SCHEMA",
    "LIBERO_OFFICIAL_EPISODES_PER_TASK",
    "LIBERO_OFFICIAL_IMAGE_SIDE",
    "LIBERO_OFFICIAL_MAX_STEPS",
    "LIBERO_OFFICIAL_SEED",
    "LIBERO_OFFICIAL_SUITES",
    "LIBERO_OFFICIAL_WARMUP_STEPS",
    "LIBERO_POLICY_HORIZON",
    "LiberoBridgePolicy",
    "execute_libero_action",
    "evaluate_libero",
    "libero_action_contract",
    "libero_bridge_identity",
    "libero_policy_observation",
    "validate_libero_bridge_health",
]
