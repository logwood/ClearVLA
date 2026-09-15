"""ManiSkill 3 adapter for the first data-producing benchmark task."""

from __future__ import annotations

from importlib.metadata import version
from typing import Any, Mapping, cast

import numpy as np

from .contracts import (
    ACTION_DIM,
    EnvironmentDescriptor,
    EvaluationState,
    PolicyObservation,
    ResetResult,
    StepResult,
)
from .state_chart import (
    CONTINUOUS_STATE_CHART,
    CONTINUOUS_STATE_SEMANTICS,
    LEGACY_STATE_CHART,
    LEGACY_STATE_SEMANTICS,
    CausalRotationChart,
)


def _numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = cast(Any, value).detach().cpu().numpy()
    return np.asarray(value)


def _unbatch(value: Any) -> np.ndarray:
    array = _numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _scalar(value: Any) -> Any:
    array = _numpy(value)
    if array.size != 1:
        raise ValueError(f"expected one scalar, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def _quaternion_to_rotvec(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("TCP quaternion must be one finite wxyz vector")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("TCP quaternion has zero norm")
    q = q / norm
    if q[0] < 0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[0], -1.0, 1.0))
    return (q[1:] / vector_norm * angle).astype(np.float32)


def _numeric_metrics(info: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, value in info.items():
        try:
            array = _numpy(value)
        except Exception:
            continue
        if array.size != 1:
            continue
        scalar = array.reshape(-1)[0].item()
        if isinstance(scalar, (bool, int, float, np.generic)):
            metrics[str(name)] = scalar
    metrics["success"] = bool(metrics.get("success", False))
    return metrics


def _finite_flat(value: Any) -> np.ndarray | None:
    """Return a finite one-dimensional view, or ``None`` when unavailable."""

    try:
        array = _unbatch(value).astype(np.float64, copy=False).reshape(-1)
    except Exception:
        return None
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return array


def _finite_vector(value: Any, width: int) -> list[float] | None:
    array = _finite_flat(value)
    if array is None or int(array.size) < int(width):
        return None
    return [float(item) for item in array[: int(width)]]


def _native_pose(obj: Any) -> list[float] | None:
    if obj is None:
        return None
    pose = getattr(obj, "pose", None)
    return _finite_vector(getattr(pose, "raw_pose", None), 7)


def _native_velocity(obj: Any, method_name: str, property_name: str) -> list[float] | None:
    if obj is None:
        return None
    try:
        method = getattr(obj, method_name, None)
        value = method() if callable(method) else getattr(obj, property_name)
    except Exception:
        return None
    return _finite_vector(value, 3)


def _pair_contact_force(scene: Any, first: Any, second: Any) -> list[float] | None:
    if scene is None or first is None or second is None:
        return None
    try:
        return _finite_vector(scene.get_pairwise_contact_forces(first, second), 3)
    except Exception:
        return None


def _controller_telemetry(native: Any, command: np.ndarray | None) -> dict[str, Any]:
    """Expose bounded controller facts for post-hoc execution attribution.

    ManiSkill's CPU ``PDEEPoseController`` returns ``None`` when its inverse
    kinematics solve fails and then falls back to ``_start_qpos``.  That
    fallback is intentionally an implementation detail of the simulator, so
    it is not part of the policy observation.  When the controller exposes the
    standard private target/start tensors, record only finite scalar summaries
    (and a boolean fallback indicator) in evaluator telemetry.  Older or
    alternate ManiSkill versions simply contribute no fields.
    """

    metrics: dict[str, Any] = {}
    agent = getattr(native, "agent", None)
    controller = getattr(agent, "controller", None)
    if controller is None:
        return metrics
    arm = controller
    children = getattr(controller, "controllers", None)
    if isinstance(children, Mapping):
        arm = children.get("arm") or next(iter(children.values()), None)
    if arm is None:
        return metrics

    target = _finite_flat(getattr(arm, "_target_qpos", None))
    start = _finite_flat(getattr(arm, "_start_qpos", None))
    if target is None or start is None or target.shape != start.shape:
        return metrics
    target_delta = target - start
    target_delta_norm = float(np.linalg.norm(target_delta))
    metrics["telemetry_controller_arm_target_delta_qpos_norm"] = target_delta_norm
    if command is not None:
        command_array = _finite_flat(command)
        if command_array is not None and command_array.size >= 6:
            command_arm_norm = float(np.linalg.norm(command_array[:6]))
            metrics["telemetry_controller_arm_command_norm"] = command_arm_norm
            # On the CPU path an unsolved nonzero pose target is replaced by
            # _start_qpos.  Keep this as a diagnostic flag rather than a hard
            # failure: a zero command legitimately has a zero target delta.
            metrics["telemetry_ik_fallback"] = bool(
                command_arm_norm > 1e-4 and target_delta_norm <= 1e-7
            )
    try:
        mode = getattr(controller, "__class__", type(controller)).__name__
        if mode:
            metrics["telemetry_controller_class"] = str(mode)
    except Exception:
        pass
    return metrics


def _physics_telemetry(
    native: Any,
    observation: Mapping[str, Any] | None,
    command: np.ndarray | None,
) -> dict[str, Any]:
    """Collect bounded evaluator-only physics facts from a ManiSkill step.

    The returned values are merged into ``EvaluationState`` and never enter
    ``PolicyObservation``.  Every field is optional so a ManiSkill minor
    version that omits one accessor does not turn an otherwise valid rollout
    into a policy/data-contract failure.
    """

    metrics: dict[str, Any] = {}

    def put(name: str, value: Any) -> None:
        if value is not None:
            metrics[name] = value

    agent = getattr(native, "agent", None)
    tcp = getattr(agent, "tcp", None)
    cube_a = getattr(native, "cubeA", None)
    cube_b = getattr(native, "cubeB", None)
    scene = getattr(native, "scene", None)

    tcp_pose = _native_pose(tcp)
    if tcp_pose is None and isinstance(observation, Mapping):
        extra = observation.get("extra")
        if isinstance(extra, Mapping):
            tcp_pose = _finite_vector(extra.get("tcp_pose"), 7)
    put("telemetry_tcp_pose", tcp_pose)
    put("telemetry_tcp_linear_velocity", _native_velocity(tcp, "get_linear_velocity", "linear_velocity"))
    put("telemetry_tcp_angular_velocity", _native_velocity(tcp, "get_angular_velocity", "angular_velocity"))

    for label, obj in (("cube_a", cube_a), ("cube_b", cube_b)):
        put(f"telemetry_{label}_pose", _native_pose(obj))
        put(
            f"telemetry_{label}_linear_velocity",
            _native_velocity(obj, "get_linear_velocity", "linear_velocity"),
        )
        put(
            f"telemetry_{label}_angular_velocity",
            _native_velocity(obj, "get_angular_velocity", "angular_velocity"),
        )
        try:
            put(
                f"telemetry_{label}_contact_force",
                _finite_vector(obj.get_net_contact_forces(), 3),
            )
        except Exception:
            pass
        try:
            put(
                f"telemetry_{label}_contact_impulse",
                _finite_vector(obj.get_net_contact_impulses(), 3),
            )
        except Exception:
            pass

    finger1 = getattr(agent, "finger1_link", None)
    finger2 = getattr(agent, "finger2_link", None)
    for label, obj in (("cube_a", cube_a), ("cube_b", cube_b)):
        put(
            f"telemetry_{label}_finger1_contact_force",
            _pair_contact_force(scene, obj, finger1),
        )
        put(
            f"telemetry_{label}_finger2_contact_force",
            _pair_contact_force(scene, obj, finger2),
        )
    if scene is not None:
        try:
            put("telemetry_contact_count", int(len(scene.get_contacts())))
        except Exception:
            pass

    robot = getattr(agent, "robot", None)
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None
    if robot is not None:
        try:
            qpos = _finite_flat(robot.get_qpos())
        except Exception:
            qpos = None
        try:
            qvel = _finite_flat(robot.get_qvel())
        except Exception:
            qvel = None
    if qpos is None and isinstance(observation, Mapping):
        raw_agent = observation.get("agent")
        if isinstance(raw_agent, Mapping):
            qpos = _finite_flat(raw_agent.get("qpos"))
            qvel = _finite_flat(raw_agent.get("qvel"))
    if qpos is not None and qpos.size >= 2:
        finger_qpos = [float(qpos[-2]), float(qpos[-1])]
        put("telemetry_gripper_finger_qpos", finger_qpos)
        put("telemetry_gripper_opening", float(qpos[-2] + qpos[-1]))
    if qvel is not None and qvel.size >= 2:
        put("telemetry_gripper_finger_qvel", [float(qvel[-2]), float(qvel[-1])])
    if command is not None:
        put("telemetry_executed_action", _finite_vector(command, 7))
    metrics.update(_controller_telemetry(native, command))
    return metrics


class ManiSkillStackCubeEnv:
    """Two-camera StackCube-v1 benchmark using native 7D EE delta control."""

    def __init__(
        self,
        *,
        image_size: int = 336,
        max_episode_steps: int = 400,
        sim_backend: str = "physx_cpu",
        render_backend: str = "gpu",
        # New StackCube recordings/checkpoints use the causal fixed-down
        # chart.  The principal chart remains available only when an archived
        # v1 caller opts into it explicitly.
        state_chart: str = CONTINUOUS_STATE_CHART,
    ) -> None:
        if state_chart not in {LEGACY_STATE_CHART, CONTINUOUS_STATE_CHART}:
            raise ValueError("unknown ManiSkill state chart")
        self.state_chart = state_chart
        self._rotation_chart = CausalRotationChart()
        if image_size < 32 or image_size % 16:
            raise ValueError("image_size must be >=32 and divisible by 16")
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ManiSkillStackCubeEnv requires the `simulation-maniskill` extra"
            ) from error
        self._env: Any = gym.make(
            "StackCube-v1",
            robot_uids="panda_wristcam",
            num_envs=1,
            obs_mode="rgb",
            reward_mode="sparse",
            control_mode="pd_ee_delta_pose",
            render_mode="sensors",
            sensor_configs={"width": int(image_size), "height": int(image_size)},
            sim_backend=str(sim_backend),
            render_backend=str(render_backend),
            max_episode_steps=int(max_episode_steps),
            enhanced_determinism=True,
        )
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._max_steps = int(max_episode_steps)
        control_hz = float(getattr(self._env.unwrapped, "control_freq", 20.0))
        self.descriptor = EnvironmentDescriptor(
            backend="maniskill3",
            benchmark="ManiSkill3",
            task="StackCube-v1",
            robot="panda_wristcam",
            simulator_version=version("mani-skill"),
            state_semantics=(CONTINUOUS_STATE_SEMANTICS if state_chart == CONTINUOUS_STATE_CHART
                             else LEGACY_STATE_SEMANTICS),
            action_semantics=(
                "ManiSkill normalized pd_ee_delta_pose: translation xyz, "
                "axis-angle rotation xyz, gripper target"
            ),
            action_state_semantics="previous clipped native 7D action",
            camera_semantics={
                "top": "StackCube base_camera external view",
                "wrist": "panda_wristcam hand_camera",
            },
            control_hz=control_hz,
            max_episode_steps=self._max_steps,
            notes=(
                "task object poses and success flags remain evaluator-only",
                "first benchmark adapter; LIBERO remains an isolated external comparison",
            ),
        )
        self.descriptor.validate()

    def _camera(self, observation: Mapping[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
        sensor_data = observation.get("sensor_data")
        if not isinstance(sensor_data, Mapping):
            raise ValueError("ManiSkill RGB observation has no sensor_data mapping")
        for name in candidates:
            camera = sensor_data.get(name)
            if isinstance(camera, Mapping) and "rgb" in camera:
                image = _unbatch(camera["rgb"])
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                return image.copy()
        raise KeyError(f"none of cameras {candidates} exist; available={tuple(sensor_data)}")

    def _state(self, observation: Mapping[str, Any]) -> np.ndarray:
        extra = observation.get("extra")
        agent = observation.get("agent")
        if not isinstance(extra, Mapping) or not isinstance(agent, Mapping):
            raise ValueError("ManiSkill observation lacks agent/extra state mappings")
        tcp = _unbatch(extra["tcp_pose"]).astype(np.float32)
        qpos = _unbatch(agent["qpos"]).astype(np.float32)
        if tcp.shape != (7,) or qpos.ndim != 1 or qpos.shape[0] < 2:
            raise ValueError(f"unexpected ManiSkill tcp/qpos shapes: {tcp.shape}, {qpos.shape}")
        rotation = (self._rotation_chart.encode(tcp[3:7])
                    if self.state_chart == CONTINUOUS_STATE_CHART
                    else _quaternion_to_rotvec(tcp[3:7]))
        opening = float(qpos[-2] + qpos[-1])
        state = np.concatenate((tcp[:3], rotation, [opening])).astype(np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError("derived ManiSkill policy state is not finite [7]")
        return state

    def _observation(self, value: Mapping[str, Any]) -> PolicyObservation:
        observation = PolicyObservation(
            rgb={
                "top": self._camera(value, ("base_camera", "render_camera")),
                "wrist": self._camera(value, ("hand_camera", "wrist_camera")),
            },
            state=self._state(value),
            action_state=self._last_action.copy(),
        )
        observation.validate()
        return observation

    def _evaluation_metrics(
        self,
        value: Mapping[str, Any],
        info: Mapping[str, Any],
        *,
        command: np.ndarray | None,
    ) -> dict[str, Any]:
        metrics = _numeric_metrics(info)
        native = getattr(self._env, "unwrapped", self._env)
        metrics.update(_physics_telemetry(native, value, command))
        return metrics

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        space = self._env.action_space
        low = _unbatch(space.low).astype(np.float32)
        high = _unbatch(space.high).astype(np.float32)
        if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,):
            raise ValueError(f"unexpected ManiSkill action space {space}")
        if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
            raise ValueError("ManiSkill action bounds must be finite ordered [7] vectors")
        return low.copy(), high.copy()

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("ManiSkill action must be one finite [7] vector")
        low, high = self.action_bounds()
        return np.clip(value, low, high).astype(np.float32)

    def _batched_action(self, action: np.ndarray) -> np.ndarray:
        shape = tuple(int(value) for value in self._env.action_space.shape)
        return action[None] if shape == (1, ACTION_DIM) else action

    def reset(self, *, seed: int) -> ResetResult:
        self._rotation_chart.reset()
        value, info = self._env.reset(seed=int(seed))
        # Zero EE delta and an open gripper are a safe initial action-state.
        self._last_action = self.clip_action(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        return ResetResult(
            self._observation(value),
            EvaluationState(
                self._evaluation_metrics(value, info, command=self._last_action)
            ),
        )

    def step(self, action: np.ndarray) -> StepResult:
        command = self.clip_action(action)
        value, reward, terminated, truncated, info = self._env.step(self._batched_action(command))
        self._last_action = command
        result = StepResult(
            observation=self._observation(value),
            reward=float(_scalar(reward)),
            terminated=bool(_scalar(terminated)),
            truncated=bool(_scalar(truncated)),
            evaluation=EvaluationState(
                self._evaluation_metrics(value, info, command=command)
            ),
        )
        result.validate()
        return result

    def hold_action(self, observation: PolicyObservation) -> np.ndarray:
        observation.validate()
        value = np.zeros(ACTION_DIM, dtype=np.float32)
        value[-1] = float(observation.action_state[-1])
        return self.clip_action(value)

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        low, high = self.action_bounds()
        return rng.uniform(low, high).astype(np.float32)

    def close(self) -> None:
        self._env.close()


__all__ = ["ManiSkillStackCubeEnv"]
