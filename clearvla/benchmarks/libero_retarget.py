"""Generate and audit geometry-consistent LIBERO expert retarget rollouts.

This module is deliberately simulator-side.  Moving an object in an image or
MuJoCo state while retaining the original action sequence would create false
supervision.  Instead, a target free joint is translated once in the initial
state and the released expert action is augmented by a metric feedback term
that tracks a smoothly translated copy of the released end-effector path.

The zero-translation lane executes the released expert actions exactly.  A
non-zero translation reaches its full value before the first sustained close,
is retained while the grasp settles, and returns to zero before the sustained
release so the original receptacle remains the placement target.  Generated
rollouts are admission candidates only; callers must require simulator success
and the paired geometric checks before exposing them to training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np

from clearvla.data.hdf5_episode import (
    LIBERO_E8_STATE_NORMALIZER_REFERENCE,
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
)
from clearvla.data.libero_retarget import (
    LIBERO_RETARGET_EPISODE_SCHEMA,
    LIBERO_RETARGET_NORMALIZER_POLICY,
    LIBERO_RETARGET_OVERLAY_SCHEMA,
)

from .common import LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA, atomic_hdf5
from .io import atomic_json
from .libero_short_probe import _resolve_flattened_qpos_offset
from .libero_terminal import (
    LIBERO_CAUSAL_OBSERVATION_ALIGNMENT,
    LIBERO_TERMINAL_SUFFIX_ROWS,
    _default_runtime,
    _observation_arrays,
    _prepare_snapshot,
    _step_result,
)

LIBERO_EXPERT_RETARGET_PROBE_SCHEMA = "clearvla-libero-expert-retarget-probe-v1"
DEFAULT_TARGET_JOINT = "akita_black_bowl_1_joint0"
DEFAULT_TARGET_BODY = "akita_black_bowl_1_main"
DEFAULT_RECEPTACLE_BODY = "plate_1_main"
DEFAULT_TRANSLATIONS_M = ((0.0, -0.03, 0.0), (0.0, 0.03, 0.0))
OSC_POSITION_METRES_PER_UNIT = 0.05


def _finite_array(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}, got {result.shape}")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def first_sustained(mask: np.ndarray, *, start: int = 0, rows: int = 3) -> int | None:
    """Return the first index owning ``rows`` consecutive true values."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("sustained-event mask must be one-dimensional")
    begin = int(start)
    count = int(rows)
    if begin < 0 or count <= 0:
        raise ValueError("sustained-event start/rows are invalid")
    for index in range(begin, max(begin, values.size - count + 1)):
        if bool(values[index : index + count].all()):
            return int(index)
    return None


def gripper_phase_indices(
    actions: np.ndarray,
    *,
    threshold: float = 0.1,
    sustained_rows: int = 3,
) -> tuple[int, int]:
    """Resolve close/release boundaries from the released continuous command."""

    values = _finite_array(actions, name="actions", ndim=2)
    if values.shape[1] != 7 or values.shape[0] < 2:
        raise ValueError("LIBERO expert actions must be [T>=2,7]")
    level = float(threshold)
    if not np.isfinite(level) or level < 0.0:
        raise ValueError("gripper threshold must be finite and non-negative")
    close = first_sustained(
        values[:, 6] > level,
        rows=int(sustained_rows),
    )
    if close is None:
        raise ValueError("expert demonstration has no sustained close event")
    release = first_sustained(
        values[:, 6] < -level,
        start=close + int(sustained_rows),
        rows=int(sustained_rows),
    )
    if release is None:
        raise ValueError("expert demonstration has no sustained release after close")
    if not 0 < close < release < values.shape[0]:
        raise ValueError("expert gripper phase ordering is invalid")
    return close, release


def _smoothstep01(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def translation_schedule(
    *,
    action_count: int,
    close_index: int,
    release_index: int,
    translation_m: Sequence[float],
    settle_fraction: float = 0.20,
) -> np.ndarray:
    """Return a continuous ``[T+1,3]`` EEF reference translation.

    Boundary row ``k`` describes the desired offset before action ``k``.  The
    first row is therefore exactly zero and the row at ``close_index`` is the
    requested target displacement.  After a short grasp-settle interval, the
    offset returns smoothly to zero at the release boundary.
    """

    count = int(action_count)
    close = int(close_index)
    release = int(release_index)
    shift = _finite_array(
        translation_m,
        name="translation_m",
        shape=(3,),
    )
    settle = float(settle_fraction)
    if count <= 1 or not 0 < close < release < count:
        raise ValueError("translation schedule phase indices are invalid")
    if not np.isfinite(settle) or not 0.0 <= settle < 1.0:
        raise ValueError("settle_fraction must be in [0,1)")
    rows = np.arange(count + 1, dtype=np.float64)
    ramp = _smoothstep01(rows / float(close))
    settle_end = min(
        release - 1,
        close + max(1, int(round((release - close) * settle))),
    )
    decay_width = max(1, release - settle_end)
    decay = 1.0 - _smoothstep01((rows - settle_end) / float(decay_width))
    weight = np.where(rows <= close, ramp, np.where(rows <= settle_end, 1.0, decay))
    weight[rows >= release] = 0.0
    result = weight[:, None] * shift[None, :]
    if not np.array_equal(result[0], np.zeros(3, dtype=np.float64)):
        raise AssertionError("retarget schedule must preserve the reset EEF pose")
    if not np.allclose(result[close], shift, rtol=0.0, atol=1.0e-12):
        raise AssertionError("retarget schedule did not reach the target shift at close")
    if not np.array_equal(result[release], np.zeros(3, dtype=np.float64)):
        raise AssertionError("retarget schedule must return to the original receptacle")
    return result


def resolve_free_joint_qpos_address(env: Any, joint_name: str) -> int:
    """Resolve and validate the xyz start address of one MuJoCo free joint."""

    model = getattr(getattr(env, "sim", None), "model", None)
    resolver = getattr(model, "joint_name2id", None)
    if not callable(resolver):
        raise ValueError("LIBERO MuJoCo model lacks joint_name2id")
    try:
        joint_id = int(resolver(str(joint_name)))
        joint_type = int(model.jnt_type[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
    except (IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"LIBERO free joint {joint_name!r} is unavailable") from error
    if joint_id < 0 or joint_type != 0:
        raise ValueError(f"LIBERO joint {joint_name!r} is not a free joint")
    qpos = np.asarray(getattr(getattr(env, "sim", None), "data", None).qpos)
    if not 0 <= qpos_address <= qpos.size - 7:
        raise ValueError("LIBERO free-joint qpos address is outside qpos")
    return qpos_address


def apply_free_joint_translation(
    flattened_state: np.ndarray,
    *,
    qpos_flat_offset: int,
    qpos_address: int,
    qpos_size: int,
    translation_m: Sequence[float],
) -> np.ndarray:
    """Translate exactly the xyz coordinates of one free joint in a flat state."""

    state = _finite_array(flattened_state, name="flattened_state", ndim=1)
    shift = _finite_array(translation_m, name="translation_m", shape=(3,))
    start = int(qpos_flat_offset) + int(qpos_address)
    if int(qpos_flat_offset) < 0 or not 0 <= int(qpos_address) <= int(qpos_size) - 7:
        raise ValueError("free-joint qpos contract is invalid")
    if not 0 <= start <= state.size - 7:
        raise ValueError("free-joint xyz slice is outside flattened state")
    result = state.copy()
    result[start : start + 3] += shift
    return result


def corrected_osc_action(
    original_action: np.ndarray,
    *,
    source_pre_xyz_m: np.ndarray,
    actual_pre_xyz_m: np.ndarray,
    current_translation_m: np.ndarray,
    next_translation_m: np.ndarray,
    gain: float = 1.0,
    metres_per_unit: float = OSC_POSITION_METRES_PER_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a metric path-tracking correction to one normalized OSC action.

    The correction is baseline preserving: with zero translation and an EEF
    exactly on the released pre-action path, the returned action is bitwise the
    original.  The unclipped action is returned as the second value so callers
    can reject saturation-heavy trajectories instead of silently accepting
    them.
    """

    action = _finite_array(original_action, name="original_action", shape=(7,))
    source = _finite_array(source_pre_xyz_m, name="source_pre_xyz_m", shape=(3,))
    actual = _finite_array(actual_pre_xyz_m, name="actual_pre_xyz_m", shape=(3,))
    current = _finite_array(
        current_translation_m,
        name="current_translation_m",
        shape=(3,),
    )
    following = _finite_array(
        next_translation_m,
        name="next_translation_m",
        shape=(3,),
    )
    feedback = float(gain)
    scale = float(metres_per_unit)
    if not np.isfinite(feedback) or feedback < 0.0:
        raise ValueError("retarget feedback gain must be finite and non-negative")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("OSC metres_per_unit must be finite and positive")
    desired_pre = source + current
    correction_m = desired_pre - actual + (following - current)
    unclipped = action.copy()
    unclipped[:3] += feedback * correction_m / scale
    clipped = np.clip(unclipped, -1.0, 1.0)
    return clipped.astype(np.float32), unclipped


def _body_xyz(env: Any, body_name: str) -> np.ndarray:
    sim = getattr(env, "sim", None)
    resolver = getattr(getattr(sim, "model", None), "body_name2id", None)
    if not callable(resolver):
        raise ValueError("LIBERO MuJoCo model lacks body_name2id")
    try:
        body_id = int(resolver(str(body_name)))
        value = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    except (IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"LIBERO body {body_name!r} is unavailable") from error
    return _finite_array(value, name=f"body {body_name}", shape=(3,)).copy()


def _controller_position_scale(env: Any) -> float:
    robots = getattr(env, "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        robots = getattr(getattr(env, "env", env), "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        raise ValueError("LIBERO environment exposes no robot controller")
    controller = getattr(robots[0], "controller", None)
    output_min = _finite_array(
        getattr(controller, "output_min", None),
        name="OSC output_min",
        ndim=1,
    )
    output_max = _finite_array(
        getattr(controller, "output_max", None),
        name="OSC output_max",
        ndim=1,
    )
    input_min = _finite_array(
        getattr(controller, "input_min", None),
        name="OSC input_min",
        ndim=1,
    )
    input_max = _finite_array(
        getattr(controller, "input_max", None),
        name="OSC input_max",
        ndim=1,
    )
    if min(output_min.size, output_max.size, input_min.size, input_max.size) < 3:
        raise ValueError("LIBERO OSC position scale has fewer than three axes")
    scales = (output_max[:3] - output_min[:3]) / (input_max[:3] - input_min[:3])
    if not np.allclose(scales, scales[0], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"LIBERO OSC xyz scales differ: {scales.tolist()}")
    scale = float(scales[0])
    if not np.isclose(scale, OSC_POSITION_METRES_PER_UNIT, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "LIBERO OSC position scale differs from the registered 0.05 m/unit: "
            f"{scale:.12g}"
        )
    return scale


@dataclass(frozen=True)
class RetargetRollout:
    demo_index: int
    translation_m: tuple[float, float, float]
    close_index: int
    release_index: int
    success: bool
    first_success_step: int | None
    clipped_position_fraction: float
    tracking_rmse_m: float
    tracking_max_m: float
    initial_target_translation_error_m: float
    final_target_to_receptacle_m: float
    final_eef_to_receptacle_m: float
    action_l2_delta_mean: float
    action_l2_delta_max: float

    def as_dict(self) -> dict[str, object]:
        return {
            "demo_index": self.demo_index,
            "translation_m": list(self.translation_m),
            "close_index": self.close_index,
            "release_index": self.release_index,
            "success": self.success,
            "first_success_step": self.first_success_step,
            "clipped_position_fraction": self.clipped_position_fraction,
            "tracking_rmse_m": self.tracking_rmse_m,
            "tracking_max_m": self.tracking_max_m,
            "initial_target_translation_error_m": self.initial_target_translation_error_m,
            "final_target_to_receptacle_m": self.final_target_to_receptacle_m,
            "final_eef_to_receptacle_m": self.final_eef_to_receptacle_m,
            "action_l2_delta_mean": self.action_l2_delta_mean,
            "action_l2_delta_max": self.action_l2_delta_max,
        }


@dataclass(frozen=True)
class RetargetTrajectory:
    """One admitted candidate plus the exact arrays needed by mainline data."""

    audit: RetargetRollout
    actions: np.ndarray  # real executed native commands [T,7]
    states: np.ndarray  # causal pre-action policy states [T,7]
    top_rgb: np.ndarray  # causal pre-action images [T,H,W,3]
    wrist_rgb: np.ndarray
    terminal_state: np.ndarray  # genuine post-final-action observation [7]
    terminal_top_rgb: np.ndarray
    terminal_wrist_rgb: np.ndarray
    eef_boundaries_xyz_m: np.ndarray  # pre-action rows plus terminal [T+1,3]


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _task_assets(suite_name: str, task_id: int) -> tuple[Any, str]:
    from libero.libero import benchmark

    benchmark_map = benchmark.get_benchmark_dict()
    if str(suite_name) not in benchmark_map:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}")
    suite = benchmark_map[str(suite_name)](task_order_index=0)
    if not 0 <= int(task_id) < int(suite.n_tasks):
        raise IndexError(f"LIBERO task index {task_id} is outside {suite.n_tasks}")
    return suite.get_task(int(task_id)), str(suite.get_task_bddl_file_path(int(task_id)))


def _run_variant(
    *,
    env: Any,
    quat_to_axisangle: Callable[[np.ndarray], np.ndarray],
    model_xml_postprocessor: Callable[[str, Mapping[str, object]], str],
    raw_demo: h5py.Group,
    demo_index: int,
    translation_m: np.ndarray,
    target_joint: str,
    target_body: str,
    receptacle_body: str,
    image_side: int,
    feedback_gain: float,
    settle_fraction: float,
) -> RetargetTrajectory:
    actions = _finite_array(raw_demo["actions"][...], name="raw actions", ndim=2)
    states = _finite_array(raw_demo["states"][...], name="raw states", ndim=2)
    source_post = _finite_array(
        raw_demo["obs/ee_states"][...],
        name="raw obs.ee_states",
        ndim=2,
    )
    if actions.shape != (states.shape[0], 7) or source_post.shape != (states.shape[0], 6):
        raise ValueError("raw LIBERO action/state/EEF rows are not aligned")
    model_xml = raw_demo.attrs.get("model_file")
    if not isinstance(model_xml, str) or not model_xml.strip():
        raise ValueError("raw LIBERO demonstration has no model_file XML")

    base_observation, _, _ = _prepare_snapshot(
        env,
        states[0],
        model_xml=model_xml,
        model_xml_postprocessor=model_xml_postprocessor,
    )
    base_state, _, _ = _observation_arrays(
        base_observation,
        quat_to_axisangle=quat_to_axisangle,
        image_side=int(image_side),
    )
    qpos_size = int(np.asarray(env.sim.data.qpos).size)
    qpos_address = resolve_free_joint_qpos_address(env, target_joint)
    qpos_flat_offset = _resolve_flattened_qpos_offset(
        env,
        state_width=int(states.shape[1]),
    )
    initial_target = _body_xyz(env, target_body)
    shifted_state = apply_free_joint_translation(
        states[0],
        qpos_flat_offset=qpos_flat_offset,
        qpos_address=qpos_address,
        qpos_size=qpos_size,
        translation_m=translation_m,
    )
    observation, _, _ = _prepare_snapshot(
        env,
        shifted_state,
        model_xml=model_xml,
        model_xml_postprocessor=model_xml_postprocessor,
    )
    shifted_target = _body_xyz(env, target_body)
    applied_error = float(np.linalg.norm((shifted_target - initial_target) - translation_m))
    if applied_error > 1.0e-10:
        raise ValueError(
            "LIBERO retarget translation did not reach the requested body pose: "
            f"error={applied_error:.6g}"
        )
    scale = _controller_position_scale(env)
    close, release = gripper_phase_indices(actions)
    schedule = translation_schedule(
        action_count=actions.shape[0],
        close_index=close,
        release_index=release,
        translation_m=translation_m,
        settle_fraction=settle_fraction,
    )
    source_pre = np.concatenate((base_state[None, :3], source_post[:-1, :3]), axis=0)

    tracking_errors: list[float] = []
    clipping_count = 0
    action_deltas: list[float] = []
    executed_actions: list[np.ndarray] = []
    causal_states: list[np.ndarray] = []
    causal_top: list[np.ndarray] = []
    causal_wrist: list[np.ndarray] = []
    eef_boundaries: list[np.ndarray] = []
    first_success: int | None = None
    success = False
    for step, original in enumerate(actions):
        current_state, current_top, current_wrist = _observation_arrays(
            observation,
            quat_to_axisangle=quat_to_axisangle,
            image_side=int(image_side),
        )
        causal_states.append(current_state.copy())
        causal_top.append(current_top.copy())
        causal_wrist.append(current_wrist.copy())
        eef_boundaries.append(current_state[:3].astype(np.float64, copy=True))
        if np.array_equal(translation_m, np.zeros(3, dtype=np.float64)):
            executed = original.astype(np.float32, copy=True)
            unclipped = original.copy()
        else:
            executed, unclipped = corrected_osc_action(
                original,
                source_pre_xyz_m=source_pre[step],
                actual_pre_xyz_m=current_state[:3],
                current_translation_m=schedule[step],
                next_translation_m=schedule[step + 1],
                gain=feedback_gain,
                metres_per_unit=scale,
            )
        clipping_count += int(bool((np.abs(unclipped[:3]) > 1.0 + 1.0e-12).any()))
        action_deltas.append(float(np.linalg.norm(executed - original)))
        executed_actions.append(executed.copy())
        observation, _reward, _done, step_success = _step_result(env, executed)
        post_state, post_top, post_wrist = _observation_arrays(
            observation,
            quat_to_axisangle=quat_to_axisangle,
            image_side=int(image_side),
        )
        desired_post = source_post[step, :3] + schedule[step + 1]
        tracking_errors.append(float(np.linalg.norm(post_state[:3] - desired_post)))
        success = bool(success or step_success)
        if step_success and first_success is None:
            first_success = int(step)
    target_final = _body_xyz(env, target_body)
    receptacle_final = _body_xyz(env, receptacle_body)
    final_state = post_state
    eef_boundaries.append(final_state[:3].astype(np.float64, copy=True))
    errors = np.asarray(tracking_errors, dtype=np.float64)
    deltas = np.asarray(action_deltas, dtype=np.float64)
    return RetargetTrajectory(
        audit=RetargetRollout(
            demo_index=int(demo_index),
            translation_m=tuple(float(value) for value in translation_m),
            close_index=int(close),
            release_index=int(release),
            success=bool(success),
            first_success_step=first_success,
            clipped_position_fraction=float(clipping_count / actions.shape[0]),
            tracking_rmse_m=float(np.sqrt(np.mean(np.square(errors)))),
            tracking_max_m=float(errors.max()),
            initial_target_translation_error_m=applied_error,
            final_target_to_receptacle_m=float(
                np.linalg.norm(target_final - receptacle_final)
            ),
            final_eef_to_receptacle_m=float(
                np.linalg.norm(final_state[:3] - receptacle_final)
            ),
            action_l2_delta_mean=float(deltas.mean()),
            action_l2_delta_max=float(deltas.max()),
        ),
        actions=np.stack(executed_actions).astype(np.float32),
        states=np.stack(causal_states).astype(np.float32),
        top_rgb=np.stack(causal_top),
        wrist_rgb=np.stack(causal_wrist),
        terminal_state=final_state.astype(np.float32, copy=True),
        terminal_top_rgb=post_top.copy(),
        terminal_wrist_rgb=post_wrist.copy(),
        eef_boundaries_xyz_m=np.stack(eef_boundaries),
    )


def probe_expert_retarget(
    raw_demo_hdf5: str | Path,
    *,
    suite_name: str,
    task_id: int,
    demo_indices: Sequence[int],
    translations_m: Sequence[Sequence[float]],
    target_joint: str = DEFAULT_TARGET_JOINT,
    target_body: str = DEFAULT_TARGET_BODY,
    receptacle_body: str = DEFAULT_RECEPTACLE_BODY,
    image_side: int = 128,
    feedback_gain: float = 1.0,
    settle_fraction: float = 0.20,
) -> dict[str, object]:
    """Run bounded expert retarget variants and return a JSON-safe audit."""

    source = Path(raw_demo_hdf5).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"raw LIBERO demonstration is absent: {source}")
    demos = tuple(int(value) for value in demo_indices)
    if not demos or len(set(demos)) != len(demos) or min(demos) < 0:
        raise ValueError("demo_indices must be non-empty, unique and non-negative")
    translations = tuple(
        _finite_array(value, name="translation", shape=(3,))
        for value in translations_m
    )
    if not translations:
        raise ValueError("at least one retarget translation is required")
    if int(image_side) <= 0:
        raise ValueError("image_side must be positive")

    _task, bddl_path = _task_assets(str(suite_name), int(task_id))
    env_factory, quat_to_axisangle, model_xml_postprocessor = _default_runtime()
    env = env_factory(
        bddl_file_name=bddl_path,
        camera_heights=int(image_side),
        camera_widths=int(image_side),
    )
    rows: list[dict[str, object]] = []
    try:
        with h5py.File(source, "r") as stream:
            data = stream.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError("raw LIBERO HDF5 has no data group")
            for demo_index in demos:
                demo_name = f"demo_{demo_index}"
                raw_demo = data.get(demo_name)
                if not isinstance(raw_demo, h5py.Group):
                    raise ValueError(f"raw LIBERO HDF5 has no data/{demo_name}")
                for translation in (np.zeros(3, dtype=np.float64), *translations):
                    rows.append(
                        _run_variant(
                            env=env,
                            quat_to_axisangle=quat_to_axisangle,
                            model_xml_postprocessor=model_xml_postprocessor,
                            raw_demo=raw_demo,
                            demo_index=demo_index,
                            translation_m=translation,
                            target_joint=str(target_joint),
                            target_body=str(target_body),
                            receptacle_body=str(receptacle_body),
                            image_side=int(image_side),
                            feedback_gain=float(feedback_gain),
                            settle_fraction=float(settle_fraction),
                        ).audit.as_dict()
                    )
    finally:
        env.close()

    baselines = [row for row in rows if np.linalg.norm(row["translation_m"]) == 0.0]
    retargeted = [row for row in rows if np.linalg.norm(row["translation_m"]) > 0.0]
    return {
        "schema": LIBERO_EXPERT_RETARGET_PROBE_SCHEMA,
        "scope": "simulator expert-retarget admission; not an official policy score",
        "raw_demo_hdf5": str(source),
        "suite": str(suite_name),
        "task_id": int(task_id),
        "demo_indices": list(demos),
        "target_joint": str(target_joint),
        "target_body": str(target_body),
        "receptacle_body": str(receptacle_body),
        "controller": {
            "position_metres_per_normalized_unit": OSC_POSITION_METRES_PER_UNIT,
            "feedback_gain": float(feedback_gain),
            "settle_fraction": float(settle_fraction),
            "zero_translation_executes_released_action_exactly": True,
        },
        "rows": rows,
        "summary": {
            "baseline_successes": int(sum(bool(row["success"]) for row in baselines)),
            "baseline_count": len(baselines),
            "retarget_successes": int(sum(bool(row["success"]) for row in retargeted)),
            "retarget_count": len(retargeted),
            "retarget_tracking_rmse_m_mean": float(
                np.mean([float(row["tracking_rmse_m"]) for row in retargeted])
            ),
            "retarget_clipped_position_fraction_max": float(
                max(float(row["clipped_position_fraction"]) for row in retargeted)
            ),
        },
    }


def _json_mapping(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _paired_translations(
    translations_m: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    values = tuple(
        _finite_array(value, name="translation", shape=(3,))
        for value in translations_m
    )
    if len(values) != 2:
        raise ValueError("training overlay requires exactly one symmetric translation pair")
    if float(np.linalg.norm(values[0])) <= 0.0:
        raise ValueError("training overlay translation must be non-zero")
    if not np.allclose(values[0], -values[1], rtol=0.0, atol=1.0e-12):
        raise ValueError("training overlay translations must be exact additive inverses")
    if abs(float(values[0][2])) > 1.0e-12:
        raise ValueError("training overlay currently admits table-plane XY translation only")
    return values[0], values[1]


def _retarget_episode_id(source_episode_id: str, translation_m: np.ndarray) -> str:
    tokens = []
    for axis, value in zip("xyz", translation_m):
        millimetres = float(value) * 1000.0
        sign = "p" if millimetres >= 0.0 else "m"
        magnitude = f"{abs(millimetres):05.1f}".replace(".", "p")
        tokens.append(f"{axis}{sign}{magnitude}")
    return f"{source_episode_id}__retarget_{'_'.join(tokens)}_v1"


def _base_training_episodes(
    *,
    base_root: Path,
    split_manifest: Path,
    demo_indices: Sequence[int],
) -> dict[int, tuple[str, Path]]:
    payload = _json_mapping(split_manifest, name="base split manifest")
    if payload.get("schema") != "clearvla-episode-splits-v1":
        raise ValueError("base split manifest schema differs")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("base split manifest has no splits mapping")
    train = splits.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError("base split manifest has no training episodes")
    requested = set(int(value) for value in demo_indices)
    result: dict[int, tuple[str, Path]] = {}
    for raw_episode_id in train:
        episode_id = str(raw_episode_id)
        path = base_root / f"{episode_id}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"base training episode is absent: {path}")
        with h5py.File(path, "r") as stream:
            raw_source_demo = stream.attrs.get("source_demo")
            converter = stream.attrs.get("converter_schema")
            padding = stream.attrs.get("terminal_padding_mode")
            if isinstance(raw_source_demo, bytes):
                raw_source_demo = raw_source_demo.decode("utf-8")
            if isinstance(converter, bytes):
                converter = converter.decode("utf-8")
            if isinstance(padding, bytes):
                padding = padding.decode("utf-8")
            if converter != LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA:
                raise ValueError(f"base episode is not terminal-suffix LIBERO: {path}")
            if padding != LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING:
                raise ValueError(f"base episode terminal padding differs: {path}")
            source_demo = str(raw_source_demo)
        if not source_demo.startswith("demo_"):
            raise ValueError(f"base episode has malformed source_demo: {path}")
        try:
            demo_index = int(source_demo.removeprefix("demo_"))
        except ValueError as error:
            raise ValueError(f"base episode source_demo is not numeric: {path}") from error
        if demo_index in requested:
            if demo_index in result:
                raise ValueError(f"base training split repeats source demo {demo_index}")
            result[demo_index] = (episode_id, path)
    missing = sorted(requested.difference(result))
    if missing:
        raise ValueError(
            "retarget overlay may use only base training demonstrations; "
            f"missing train demos={missing}"
        )
    return result


def _paired_geometry_audit(
    baseline: RetargetTrajectory,
    variants: Sequence[RetargetTrajectory],
) -> dict[str, float | bool]:
    if len(variants) != 2:
        raise ValueError("paired geometry audit requires two retarget variants")
    translation = np.asarray(variants[1].audit.translation_m, dtype=np.float64)
    magnitude = float(np.linalg.norm(translation))
    if magnitude <= 0.0:
        raise ValueError("paired geometry audit translation is zero")
    unit = translation / magnitude
    ordered = sorted(
        variants,
        key=lambda row: float(np.dot(np.asarray(row.audit.translation_m), unit)),
    )
    negative, positive = ordered
    close = int(baseline.audit.close_index) + 1
    baseline_close = baseline.eef_boundaries_xyz_m[close]
    negative_response = negative.eef_boundaries_xyz_m[close] - baseline_close
    positive_response = positive.eef_boundaries_xyz_m[close] - baseline_close
    negative_projection = float(np.dot(negative_response, unit))
    positive_projection = float(np.dot(positive_response, unit))
    separation = positive_projection - negative_projection
    paired_delta_residual = (
        positive.actions[:, :3]
        + negative.actions[:, :3]
        - 2.0 * baseline.actions[:, :3]
    )
    paired_delta_signal = positive.actions[:, :3] - negative.actions[:, :3]
    residual_rms = float(np.sqrt(np.mean(np.square(paired_delta_residual))))
    signal_rms = float(np.sqrt(np.mean(np.square(paired_delta_signal))))
    return {
        "negative_close_projection_m": negative_projection,
        "positive_close_projection_m": positive_projection,
        "paired_close_separation_m": separation,
        "paired_close_separation_ratio": separation / (2.0 * magnitude),
        "negative_response_sign_correct": negative_projection < 0.0,
        "positive_response_sign_correct": positive_projection > 0.0,
        "arm_delta_antisymmetry_residual_rms": residual_rms,
        "arm_delta_pair_signal_rms": signal_rms,
        "arm_delta_antisymmetry_residual_to_signal": residual_rms
        / max(signal_rms, 1.0e-12),
    }


def _write_retarget_episode(
    path: Path,
    *,
    base_path: Path,
    source_episode_id: str,
    trajectory: RetargetTrajectory,
    target_joint: str,
    target_body: str,
    receptacle_body: str,
    feedback_gain: float,
    settle_fraction: float,
) -> None:
    with h5py.File(base_path, "r") as base:
        attrs = {
            str(name): value
            for name, value in base.attrs.items()
            if not str(name).startswith("terminal_replay_")
            and not str(name).startswith("retarget_")
        }
        reference = np.asarray(base["normalizer_reference_state"], dtype=np.float32)
    real_count = int(trajectory.actions.shape[0])
    if reference.shape != (real_count, 7):
        raise ValueError("base normalizer reference differs from retarget source length")
    absorbing = np.zeros((LIBERO_TERMINAL_SUFFIX_ROWS, 7), dtype=np.float32)
    absorbing[:, 6] = trajectory.actions[-1, 6]
    actions = np.concatenate((trajectory.actions, absorbing), axis=0)
    states = np.concatenate(
        (
            trajectory.states,
            np.repeat(
                trajectory.terminal_state[None],
                LIBERO_TERMINAL_SUFFIX_ROWS,
                axis=0,
            ),
        ),
        axis=0,
    )
    top = np.concatenate(
        (
            trajectory.top_rgb,
            np.repeat(
                trajectory.terminal_top_rgb[None],
                LIBERO_TERMINAL_SUFFIX_ROWS,
                axis=0,
            ),
        ),
        axis=0,
    )
    wrist = np.concatenate(
        (
            trajectory.wrist_rgb,
            np.repeat(
                trajectory.terminal_wrist_rgb[None],
                LIBERO_TERMINAL_SUFFIX_ROWS,
                axis=0,
            ),
        ),
        axis=0,
    )
    action_state = np.concatenate(
        (np.zeros((1, 7), dtype=np.float32), actions[:-1]),
        axis=0,
    )
    if not (
        actions.shape == action_state.shape == states.shape
        and top.shape[0] == wrist.shape[0] == actions.shape[0]
    ):
        raise AssertionError("retarget episode arrays are not row-aligned")

    def writer(temporary: Path) -> None:
        with h5py.File(temporary, "w") as stream:
            for name, value in attrs.items():
                stream.attrs[name] = value
            stream.attrs["converter_schema"] = LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA
            stream.attrs["observation_alignment"] = LIBERO_CAUSAL_OBSERVATION_ALIGNMENT
            stream.attrs["state_normalizer_reference_key"] = "normalizer_reference_state"
            stream.attrs["state_normalizer_reference_semantics"] = (
                LIBERO_E8_STATE_NORMALIZER_REFERENCE
            )
            stream.attrs["valid_center_start"] = 0
            stream.attrs["strict_valid_center_start"] = 24
            stream.attrs["strict_valid_center_end"] = real_count - 49
            stream.attrs["valid_center_end"] = real_count - 1
            stream.attrs["terminal_state_index"] = real_count
            stream.attrs["source_action_count"] = real_count
            stream.attrs["terminal_padding_mode"] = (
                LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
            )
            stream.attrs["retarget_schema"] = LIBERO_RETARGET_EPISODE_SCHEMA
            stream.attrs["retarget_source_episode_id"] = str(source_episode_id)
            stream.attrs["retarget_translation_m"] = np.asarray(
                trajectory.audit.translation_m,
                dtype=np.float64,
            )
            stream.attrs["retarget_target_joint"] = str(target_joint)
            stream.attrs["retarget_target_body"] = str(target_body)
            stream.attrs["retarget_receptacle_body"] = str(receptacle_body)
            stream.attrs["retarget_feedback_gain"] = float(feedback_gain)
            stream.attrs["retarget_settle_fraction"] = float(settle_fraction)
            stream.attrs["retarget_success"] = bool(trajectory.audit.success)
            stream.attrs["retarget_tracking_rmse_m"] = float(
                trajectory.audit.tracking_rmse_m
            )
            stream.attrs["retarget_clipped_position_fraction"] = float(
                trajectory.audit.clipped_position_fraction
            )
            stream.create_dataset("action", data=actions)
            stream.create_dataset("action_state", data=action_state)
            stream.create_dataset("state", data=states)
            stream.create_dataset("normalizer_reference_state", data=reference)
            images = stream.require_group("observations/images")
            images.create_dataset(
                "cam_high",
                data=top,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            images.create_dataset(
                "cam_right_wrist",
                data=wrist,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

    atomic_hdf5(path, writer)


def generate_retarget_training_overlay(
    raw_demo_hdf5: str | Path,
    output_root: str | Path,
    *,
    base_causal_root: str | Path,
    base_split_manifest: str | Path,
    suite_name: str,
    task_id: int,
    demo_indices: Sequence[int],
    translations_m: Sequence[Sequence[float]],
    target_joint: str = DEFAULT_TARGET_JOINT,
    target_body: str = DEFAULT_TARGET_BODY,
    receptacle_body: str = DEFAULT_RECEPTACLE_BODY,
    image_side: int = 128,
    feedback_gain: float = 1.0,
    settle_fraction: float = 0.20,
    max_tracking_rmse_m: float = 0.01,
    max_clipped_position_fraction: float = 0.05,
    max_final_target_to_receptacle_m: float = 0.03,
    min_paired_close_separation_ratio: float = 0.60,
) -> dict[str, object]:
    """Atomically build a train-only, paired LIBERO retarget overlay root."""

    raw_source = Path(raw_demo_hdf5).expanduser().resolve()
    base_root = Path(base_causal_root).expanduser().resolve()
    split_path = Path(base_split_manifest).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not raw_source.is_file():
        raise FileNotFoundError(f"raw LIBERO demonstration is absent: {raw_source}")
    if not base_root.is_dir() or not split_path.is_file():
        raise FileNotFoundError("base causal root/split manifest is absent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite retarget overlay: {destination}")
    demos = tuple(int(value) for value in demo_indices)
    if not demos or len(set(demos)) != len(demos) or min(demos) < 0:
        raise ValueError("demo_indices must be non-empty, unique and non-negative")
    pair = _paired_translations(translations_m)
    thresholds = {
        "max_tracking_rmse_m": float(max_tracking_rmse_m),
        "max_clipped_position_fraction": float(max_clipped_position_fraction),
        "max_final_target_to_receptacle_m": float(
            max_final_target_to_receptacle_m
        ),
        "min_paired_close_separation_ratio": float(
            min_paired_close_separation_ratio
        ),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ValueError("retarget admission thresholds must be finite and non-negative")
    base_episodes = _base_training_episodes(
        base_root=base_root,
        split_manifest=split_path,
        demo_indices=demos,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    task, bddl_path = _task_assets(str(suite_name), int(task_id))
    env_factory, quat_to_axisangle, model_xml_postprocessor = _default_runtime()
    env = env_factory(
        bddl_file_name=bddl_path,
        camera_heights=int(image_side),
        camera_widths=int(image_side),
    )
    episode_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    try:
        instructions = base_root / "instructions.json"
        if not instructions.is_file():
            raise FileNotFoundError(f"base instruction inventory is absent: {instructions}")
        shutil.copy2(instructions, staging / "instructions.json")
        with h5py.File(raw_source, "r") as stream:
            data = stream.get("data")
            if not isinstance(data, h5py.Group):
                raise ValueError("raw LIBERO HDF5 has no data group")
            for demo_index in demos:
                raw_demo = data.get(f"demo_{demo_index}")
                if not isinstance(raw_demo, h5py.Group):
                    raise ValueError(f"raw LIBERO HDF5 has no data/demo_{demo_index}")
                baseline = _run_variant(
                    env=env,
                    quat_to_axisangle=quat_to_axisangle,
                    model_xml_postprocessor=model_xml_postprocessor,
                    raw_demo=raw_demo,
                    demo_index=demo_index,
                    translation_m=np.zeros(3, dtype=np.float64),
                    target_joint=str(target_joint),
                    target_body=str(target_body),
                    receptacle_body=str(receptacle_body),
                    image_side=int(image_side),
                    feedback_gain=float(feedback_gain),
                    settle_fraction=float(settle_fraction),
                )
                variants = [
                    _run_variant(
                        env=env,
                        quat_to_axisangle=quat_to_axisangle,
                        model_xml_postprocessor=model_xml_postprocessor,
                        raw_demo=raw_demo,
                        demo_index=demo_index,
                        translation_m=translation,
                        target_joint=str(target_joint),
                        target_body=str(target_body),
                        receptacle_body=str(receptacle_body),
                        image_side=int(image_side),
                        feedback_gain=float(feedback_gain),
                        settle_fraction=float(settle_fraction),
                    )
                    for translation in pair
                ]
                pair_audit = _paired_geometry_audit(baseline, variants)
                failures: list[str] = []
                if not baseline.audit.success:
                    failures.append("released baseline no longer succeeds")
                for variant in variants:
                    audit = variant.audit
                    if not audit.success:
                        failures.append(f"translation {audit.translation_m} did not succeed")
                    if audit.tracking_rmse_m > thresholds["max_tracking_rmse_m"]:
                        failures.append(f"translation {audit.translation_m} tracking RMSE")
                    if (
                        audit.clipped_position_fraction
                        > thresholds["max_clipped_position_fraction"]
                    ):
                        failures.append(f"translation {audit.translation_m} saturation")
                    if (
                        audit.final_target_to_receptacle_m
                        > thresholds["max_final_target_to_receptacle_m"]
                    ):
                        failures.append(f"translation {audit.translation_m} placement")
                if not bool(pair_audit["negative_response_sign_correct"]):
                    failures.append("negative EEF response sign")
                if not bool(pair_audit["positive_response_sign_correct"]):
                    failures.append("positive EEF response sign")
                if (
                    float(pair_audit["paired_close_separation_ratio"])
                    < thresholds["min_paired_close_separation_ratio"]
                ):
                    failures.append("paired close separation")
                if failures:
                    raise RuntimeError(
                        f"retarget pair demo_{demo_index} failed admission: "
                        + "; ".join(failures)
                    )

                source_episode_id, base_path = base_episodes[demo_index]
                pair_episode_ids: list[str] = []
                for variant in variants:
                    translation = np.asarray(variant.audit.translation_m, dtype=np.float64)
                    episode_id = _retarget_episode_id(source_episode_id, translation)
                    output_path = staging / f"{episode_id}.hdf5"
                    _write_retarget_episode(
                        output_path,
                        base_path=base_path,
                        source_episode_id=source_episode_id,
                        trajectory=variant,
                        target_joint=str(target_joint),
                        target_body=str(target_body),
                        receptacle_body=str(receptacle_body),
                        feedback_gain=float(feedback_gain),
                        settle_fraction=float(settle_fraction),
                    )
                    pair_episode_ids.append(episode_id)
                    episode_rows.append(
                        {
                            "episode_id": episode_id,
                            "file": f"{episode_id}.hdf5",
                            "file_sha256": _sha256_file(output_path),
                            "source_episode_id": source_episode_id,
                            "source_demo": f"demo_{demo_index}",
                            "source_split": "train",
                            **variant.audit.as_dict(),
                        }
                    )
                pair_rows.append(
                    {
                        "source_episode_id": source_episode_id,
                        "source_demo": f"demo_{demo_index}",
                        "episode_ids": pair_episode_ids,
                        "baseline": baseline.audit.as_dict(),
                        "geometry": pair_audit,
                        "admitted": True,
                    }
                )

        base_manifest = base_root / "dataset_manifest.json"
        manifest: dict[str, object] = {
            "schema": LIBERO_RETARGET_OVERLAY_SCHEMA,
            "scope": "train-only simulator retarget overlay; validation/test unchanged",
            "data_profile": "libero_relative_7d_v1",
            "window_boundary_contract": "causal_prefix_terminal_suffix_v2",
            "suite": str(suite_name),
            "task_id": int(task_id),
            "task_name": str(getattr(task, "name", "")),
            "raw_demo_hdf5": str(raw_source),
            "raw_demo_hdf5_sha256": _sha256_file(raw_source),
            "base_causal_root": str(base_root),
            "base_split_manifest": str(split_path),
            "base_split_manifest_sha256": _sha256_file(split_path),
            "base_dataset_manifest_sha256": (
                _sha256_file(base_manifest) if base_manifest.is_file() else None
            ),
            "target_joint": str(target_joint),
            "target_body": str(target_body),
            "receptacle_body": str(receptacle_body),
            "translations_m": [value.tolist() for value in pair],
            "controller": {
                "position_metres_per_normalized_unit": OSC_POSITION_METRES_PER_UNIT,
                "feedback_gain": float(feedback_gain),
                "settle_fraction": float(settle_fraction),
                "zero_translation_executes_released_action_exactly": True,
            },
            "admission_thresholds": thresholds,
            "normalizer_policy": LIBERO_RETARGET_NORMALIZER_POLICY,
            "episode_count": len(episode_rows),
            "pair_count": len(pair_rows),
            "episodes": episode_rows,
            "pairs": pair_rows,
        }
        atomic_json(staging / "overlay_manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        env.close()


def _translation_arg(value: str) -> tuple[float, float, float]:
    parts = str(value).split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("translation must be x,y,z in metres")
    try:
        result = tuple(float(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("translation must contain three floats") from error
    if not np.isfinite(result).all():
        raise argparse.ArgumentTypeError("translation values must be finite")
    return result  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "overlay"), default="probe")
    parser.add_argument("--raw-demo-hdf5", type=Path, required=True)
    parser.add_argument("--base-causal-root", type=Path)
    parser.add_argument("--base-split-manifest", type=Path)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--demo-indices", type=int, nargs="+", default=(0, 1))
    parser.add_argument(
        "--translations-m",
        type=_translation_arg,
        nargs="+",
        default=DEFAULT_TRANSLATIONS_M,
    )
    parser.add_argument("--target-joint", default=DEFAULT_TARGET_JOINT)
    parser.add_argument("--target-body", default=DEFAULT_TARGET_BODY)
    parser.add_argument("--receptacle-body", default=DEFAULT_RECEPTACLE_BODY)
    parser.add_argument("--image-side", type=int, default=128)
    parser.add_argument("--feedback-gain", type=float, default=1.0)
    parser.add_argument("--settle-fraction", type=float, default=0.20)
    parser.add_argument("--max-tracking-rmse-m", type=float, default=0.01)
    parser.add_argument("--max-clipped-position-fraction", type=float, default=0.05)
    parser.add_argument("--max-final-target-to-receptacle-m", type=float, default=0.03)
    parser.add_argument("--min-paired-close-separation-ratio", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    common = {
        "suite_name": args.suite,
        "task_id": args.task_id,
        "demo_indices": args.demo_indices,
        "translations_m": args.translations_m,
        "target_joint": args.target_joint,
        "target_body": args.target_body,
        "receptacle_body": args.receptacle_body,
        "image_side": args.image_side,
        "feedback_gain": args.feedback_gain,
        "settle_fraction": args.settle_fraction,
    }
    if args.mode == "probe":
        report = probe_expert_retarget(args.raw_demo_hdf5, **common)
        atomic_json(args.output, report)
        compact = report["summary"]
    else:
        if args.base_causal_root is None or args.base_split_manifest is None:
            parser.error("overlay mode requires --base-causal-root and --base-split-manifest")
        report = generate_retarget_training_overlay(
            args.raw_demo_hdf5,
            args.output,
            base_causal_root=args.base_causal_root,
            base_split_manifest=args.base_split_manifest,
            max_tracking_rmse_m=args.max_tracking_rmse_m,
            max_clipped_position_fraction=args.max_clipped_position_fraction,
            max_final_target_to_receptacle_m=args.max_final_target_to_receptacle_m,
            min_paired_close_separation_ratio=args.min_paired_close_separation_ratio,
            **common,
        )
        compact = {
            "output_root": str(args.output.resolve()),
            "episode_count": report["episode_count"],
            "pair_count": report["pair_count"],
            "all_pairs_admitted": all(
                bool(row["admitted"]) for row in report["pairs"]
            ),
        }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_RECEPTACLE_BODY",
    "DEFAULT_TARGET_BODY",
    "DEFAULT_TARGET_JOINT",
    "DEFAULT_TRANSLATIONS_M",
    "LIBERO_EXPERT_RETARGET_PROBE_SCHEMA",
    "LIBERO_RETARGET_EPISODE_SCHEMA",
    "LIBERO_RETARGET_OVERLAY_SCHEMA",
    "OSC_POSITION_METRES_PER_UNIT",
    "RetargetRollout",
    "RetargetTrajectory",
    "apply_free_joint_translation",
    "corrected_osc_action",
    "first_sustained",
    "generate_retarget_training_overlay",
    "gripper_phase_indices",
    "probe_expert_retarget",
    "resolve_free_joint_qpos_address",
    "translation_schedule",
]
