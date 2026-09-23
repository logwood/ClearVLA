"""Primitive MuJoCo Alicia-D proxy used only for control-path smoke tests."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ACTION_DIM,
    EnvironmentDescriptor,
    EvaluationState,
    ExecutedCommand,
    PolicyObservation,
    ResetResult,
    StepResult,
)

ARM_LOW = np.asarray([-2.749, -2.0, -0.5, -2.79, -1.57, -3.14159], dtype=np.float32)
ARM_HIGH = np.asarray([2.749, 2.0, 3.14159, 2.79, 1.57, 3.14159], dtype=np.float32)


class AliciaProxyEnv:
    """Six arm joints plus one Alicia gripper scalar in raw dataset units.

    The first six values are radians.  The final scalar is an explicit
    compatibility convention: 0 is fully open and 100 is fully closed.  This
    is a proxy convention until hardware calibration and the missing meshes are
    supplied; it is serialized in every recorded dataset descriptor.
    """

    def __init__(
        self,
        *,
        image_size: int = 336,
        control_hz: float = 20.0,
        max_episode_steps: int = 200,
    ) -> None:
        if image_size < 32 or image_size % 16:
            raise ValueError("image_size must be >=32 and divisible by 16")
        if control_hz <= 0 or max_episode_steps <= 0:
            raise ValueError("control rate and episode horizon must be positive")
        try:
            import mujoco
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "AliciaProxyEnv requires the `simulation` optional dependencies"
            ) from error
        self._mujoco: Any = mujoco
        xml = Path(__file__).with_name("assets") / "alicia_d_kinematic_proxy.xml"
        self.model = self._mujoco.MjModel.from_xml_path(str(xml))
        self.data = self._mujoco.MjData(self.model)
        self.renderer = self._mujoco.Renderer(
            self.model,
            height=int(image_size),
            width=int(image_size),
        )
        self.control_hz = float(control_hz)
        self._substeps = max(
            1,
            int(round(1.0 / (self.control_hz * float(self.model.opt.timestep)))),
        )
        self._max_steps = int(max_episode_steps)
        self._step = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._tool_site = int(self.model.site("tool0").id)
        self._target_body = int(self.model.body("target").id)
        self._arm_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[int(self.model.joint(f"Joint{index}").id)]
                for index in range(1, 7)
            ],
            dtype=np.int64,
        )
        self._left_qpos_address = int(
            self.model.jnt_qposadr[int(self.model.joint("left_finger").id)]
        )
        self._right_qpos_address = int(
            self.model.jnt_qposadr[int(self.model.joint("right_finger").id)]
        )
        self.descriptor = EnvironmentDescriptor(
            backend="mujoco",
            benchmark="clearvla-control-smoke",
            task="AliciaReachProxy-v0",
            robot="Alicia-D-v5.6-100mm-primitive-proxy",
            simulator_version=version("mujoco"),
            state_semantics="[joint_1..joint_6 radians, gripper_close_percent] measured",
            action_semantics="absolute position target in the same 7D chart",
            action_state_semantics="previous clipped absolute position target",
            camera_semantics={
                "top": "fixed external overview; uncalibrated",
                "wrist": "link6-mounted proxy camera; uncalibrated",
            },
            control_hz=self.control_hz,
            max_episode_steps=self._max_steps,
            kinematic_proxy=True,
            notes=(
                "URDF joint origins/axes/limits retained; STL meshes are absent",
                "controller gains, camera poses and gripper percent mapping are provisional",
                "not a digital twin and not an external benchmark score",
            ),
        )
        self.descriptor.validate()

    @staticmethod
    def _clip_action(action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError("Alicia proxy action must be one finite [7] vector")
        return np.concatenate(
            (np.clip(value[:6], ARM_LOW, ARM_HIGH), np.clip(value[6:], 0.0, 100.0))
        ).astype(np.float32)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        return self._clip_action(action)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.concatenate((ARM_LOW, np.asarray([0.0], dtype=np.float32)))
        high = np.concatenate((ARM_HIGH, np.asarray([100.0], dtype=np.float32)))
        return low, high

    def _measured_state(self) -> np.ndarray:
        arm = np.asarray(self.data.qpos[self._arm_qpos_addresses], dtype=np.float32)
        opening = float(
            self.data.qpos[self._left_qpos_address] - self.data.qpos[self._right_qpos_address]
        )
        close_percent = 100.0 * (1.0 - np.clip(opening / 0.10, 0.0, 1.0))
        return np.concatenate((arm, np.asarray([close_percent], dtype=np.float32)))

    def _set_control(self, action: np.ndarray) -> None:
        command = self._clip_action(action)
        opening = 0.05 * (1.0 - float(command[-1]) / 100.0)
        self.data.ctrl[:] = np.concatenate(
            (command[:6], np.asarray([opening, -opening], dtype=np.float32))
        )
        self._last_action = command

    def _render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        value = np.asarray(self.renderer.render())
        if value.dtype != np.uint8:
            value = np.clip(value, 0, 255).astype(np.uint8)
        return value.copy()

    def _observation(self) -> PolicyObservation:
        observation = PolicyObservation(
            rgb={"top": self._render("top"), "wrist": self._render("wrist")},
            state=self._measured_state(),
            action_state=self._last_action.copy(),
        )
        observation.validate()
        return observation

    def _evaluation(self) -> EvaluationState:
        tool = np.asarray(self.data.site_xpos[self._tool_site], dtype=np.float64)
        target = np.asarray(self.data.xpos[self._target_body], dtype=np.float64)
        distance = float(np.linalg.norm(tool - target))
        return EvaluationState(
            metrics={
                "success": distance <= 0.035,
                "tool_target_distance_m": distance,
            }
        )

    def reset(self, *, seed: int) -> ResetResult:
        rng = np.random.default_rng(int(seed))
        self._mujoco.mj_resetData(self.model, self.data)
        initial = np.asarray([0.0, -0.55, 1.30, 0.0, 0.55, 0.0, 0.0], dtype=np.float32)
        initial[:6] += rng.normal(0.0, 0.01, size=6).astype(np.float32)
        self.data.qpos[self._arm_qpos_addresses] = initial[:6]
        self.data.qpos[self._left_qpos_address] = 0.05
        self.data.qpos[self._right_qpos_address] = -0.05
        self.model.body_pos[self._target_body] = np.asarray(
            [-0.28 + rng.uniform(-0.04, 0.04), rng.uniform(-0.10, 0.04), 0.48],
            dtype=np.float64,
        )
        self._last_action = initial
        self._set_control(initial)
        self._mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return ResetResult(self._observation(), self._evaluation())

    def step(self, action: np.ndarray) -> StepResult:
        submitted = np.asarray(action, dtype=np.float32).copy()
        self._set_control(submitted)
        for _ in range(self._substeps):
            self._mujoco.mj_step(self.model, self.data)
        self._step += 1
        evaluation = self._evaluation()
        success = bool(evaluation.metrics["success"])
        result = StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=success,
            truncated=self._step >= self._max_steps and not success,
            evaluation=evaluation,
            command_receipt=ExecutedCommand(submitted, self._last_action),
        )
        result.validate()
        return result

    def hold_action(self, observation: PolicyObservation) -> np.ndarray:
        observation.validate()
        return self._clip_action(observation.state)

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        measured = self._measured_state()
        delta = rng.normal(0.0, 0.035, size=6).astype(np.float32)
        grip = measured[-1] + float(rng.normal(0.0, 3.0))
        return self._clip_action(np.concatenate((measured[:6] + delta, [grip])))

    def close(self) -> None:
        self.renderer.close()


__all__ = ["ARM_HIGH", "ARM_LOW", "AliciaProxyEnv"]


