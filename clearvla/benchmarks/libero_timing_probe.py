"""Deployment-only LIBERO gripper-timing attribution probe.

The probe is deliberately outside the formal LIBERO evaluator and training
path.  It runs one normal receding-horizon rollout, saves the exact executed
action stream, restores the same post-warmup MuJoCo snapshot, and replays the
same arm commands while suppressing positive (close) gripper commands for a
bounded prefix.  This isolates an early-close timing hypothesis without
changing the shared codec, model, loss, or evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bridge import RemotePolicyClient
from .io import atomic_json
from .libero_eval import (
    LIBERO_POLICY_HORIZON,
    _LiberoEpisodeVideoRecorder,
    _set_init_state_observation,
    _step_observation_only,
    _step_with_success,
    execute_libero_action,
    libero_action_contract,
    libero_bridge_identity,
    libero_policy_observation,
    validate_libero_bridge_health,
)

TIMING_PROBE_SCHEMA = "clearvla-libero-timing-attribution-v1"
DEFAULT_SUITE = "libero_spatial"
DEFAULT_TASK_ID = 0
DEFAULT_INIT_STATE = 0
DEFAULT_STEPS = 32
DEFAULT_DELAY_STEPS = 12
DEFAULT_WARMUP_STEPS = 5
DEFAULT_IMAGE_SIDE = 128
DEFAULT_VIDEO_FPS = 10.0
DEFAULT_CLOSE_THRESHOLD = 0.05
DEFAULT_TARGET_BODY = "akita_black_bowl_1_main"
DEFAULT_PLATE_BODY = "plate_1_main"


def _finite_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if minimum is not None and result < int(minimum):
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_action(value: object, *, name: str = "action") -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be one finite [7] row")
    return result.copy()


def suppress_close_command(
    action: np.ndarray,
    *,
    step_index: int,
    delay_steps: int,
    close_threshold: float = DEFAULT_CLOSE_THRESHOLD,
) -> tuple[np.ndarray, bool]:
    """Suppress only positive close commands during a 1-based prefix.

    LIBERO's normalized OSC gripper chart uses positive values for closing in
    the released task configuration.  The probe intentionally replaces such
    a command with the neutral value ``0`` rather than inventing an open
    command.  Arm dimensions are copied bit-for-bit.
    """

    result = _finite_action(action)
    step = _strict_int(step_index, name="step_index", minimum=1)
    delay = _strict_int(delay_steps, name="delay_steps", minimum=0)
    threshold = _finite_float(close_threshold, name="close_threshold")
    if threshold < 0.0:
        raise ValueError("close_threshold must be non-negative")
    # Account for the small float32 rounding that can turn an exact threshold
    # value (for example ``0.05``) into ``0.0500000007``.
    suppressed = bool(step <= delay and float(result[6]) > threshold + 1e-6)
    if suppressed:
        result[6] = np.float32(0.0)
    return result, suppressed


def _sim(env: Any) -> Any:
    inner = getattr(env, "env", env)
    simulator = getattr(inner, "sim", None)
    if simulator is None:
        raise ValueError("LIBERO environment does not expose a MuJoCo simulator")
    return simulator


def _body_position(env: Any, name: str) -> np.ndarray:
    simulator = _sim(env)
    model = simulator.model
    body_id = int(model.body_name2id(str(name)))
    value = np.asarray(simulator.data.body_xpos[body_id], dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"body {name!r} has an invalid world position")
    return value.copy()


def _body_id_to_name(model: Any, body_id: int) -> str:
    getter = getattr(model, "body_id2name", None)
    if callable(getter):
        value = getter(int(body_id))
        return str(value) if value is not None else f"body_{int(body_id)}"
    # DeepMind MuJoCo exposes one generic ``id2name`` method rather than the
    # mujoco-py convenience methods.  Keep the import lazy so pure helper
    # tests and non-simulator imports do not require MuJoCo at import time.
    generic = getattr(model, "id2name", None)
    if callable(generic):
        try:
            import mujoco

            value = generic(mujoco.mjtObj.mjOBJ_BODY, int(body_id))
            return str(value) if value is not None else f"body_{int(body_id)}"
        except (ImportError, TypeError, ValueError, RuntimeError):
            pass
    return f"body_{int(body_id)}"


def _geom_id_to_name(model: Any, geom_id: int) -> str:
    getter = getattr(model, "geom_id2name", None)
    if callable(getter):
        value = getter(int(geom_id))
        return str(value) if value is not None else f"geom_{int(geom_id)}"
    generic = getattr(model, "id2name", None)
    if callable(generic):
        try:
            import mujoco

            value = generic(mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            return str(value) if value is not None else f"geom_{int(geom_id)}"
        except (ImportError, TypeError, ValueError, RuntimeError):
            pass
    return f"geom_{int(geom_id)}"


def _body_geoms(model: Any, body_id: int) -> set[int]:
    try:
        geom_bodies = np.asarray(model.geom_bodyid, dtype=np.int64).reshape(-1)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("LIBERO MuJoCo model lacks geom_bodyid") from error
    return {int(index) for index in np.flatnonzero(geom_bodies == int(body_id))}


def resolve_contact_geometries(
    env: Any,
    *,
    target_body: str,
    plate_body: str,
    gripper_geom_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Resolve target, receptacle and robot gripper geom sets once.

    Target and plate sets are body-owned.  Gripper geoms are found from the
    released model's stable ``finger``/``gripper`` naming convention, with an
    explicit id override available for a future LIBERO asset variant.  The
    resolved inventory is serialized with the probe so contact claims remain
    auditable.
    """

    simulator = _sim(env)
    model = simulator.model
    target_id = int(model.body_name2id(str(target_body)))
    plate_id = int(model.body_name2id(str(plate_body)))
    target_geoms = _body_geoms(model, target_id)
    plate_geoms = _body_geoms(model, plate_id)
    if not target_geoms or not plate_geoms:
        raise ValueError("target and plate must each own at least one geom")

    if gripper_geom_ids is not None:
        gripper = {int(value) for value in gripper_geom_ids}
        if not gripper or min(gripper) < 0:
            raise ValueError("gripper_geom_ids must be non-empty non-negative ids")
        try:
            ngeom = int(model.ngeom)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("LIBERO MuJoCo model lacks ngeom") from error
        if max(gripper) >= ngeom:
            raise ValueError("gripper_geom_ids contains an id outside the model")
    else:
        try:
            ngeom = int(model.ngeom)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("LIBERO MuJoCo model lacks ngeom") from error
        gripper = set()
        tokens = ("gripper", "finger", "eef")
        geom_bodies = np.asarray(model.geom_bodyid, dtype=np.int64).reshape(-1)
        for geom_id in range(ngeom):
            body_name = _body_id_to_name(model, int(geom_bodies[geom_id])).lower()
            geom_name = _geom_id_to_name(model, geom_id).lower()
            if any(token in body_name or token in geom_name for token in tokens):
                gripper.add(int(geom_id))
        # A malformed naming surface must not silently produce a false
        # no-contact result.  The released spatial task resolves eight geoms.
        if not gripper:
            raise ValueError(
                "could not resolve LIBERO gripper geoms by model names; pass "
                "--gripper-geom-ids explicitly"
            )
    gripper -= target_geoms | plate_geoms
    if not gripper:
        raise ValueError("resolved gripper geoms overlap only target/plate geoms")
    return {
        "target_body": str(target_body),
        "target_body_id": target_id,
        "target_geom_ids": sorted(target_geoms),
        "plate_body": str(plate_body),
        "plate_body_id": plate_id,
        "plate_geom_ids": sorted(plate_geoms),
        "gripper_geom_ids": sorted(gripper),
        "target_geom_names": [_geom_id_to_name(model, i) for i in sorted(target_geoms)],
        "plate_geom_names": [_geom_id_to_name(model, i) for i in sorted(plate_geoms)],
        "gripper_geom_names": [_geom_id_to_name(model, i) for i in sorted(gripper)],
    }


def _contact_pair_row(model: Any, contact: Any, index: int) -> dict[str, Any]:
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
    dist = float(contact.dist)
    if not math.isfinite(dist):
        dist = None  # type: ignore[assignment]
    try:
        geom_bodies = np.asarray(model.geom_bodyid, dtype=np.int64).reshape(-1)
        body1 = int(geom_bodies[geom1])
        body2 = int(geom_bodies[geom2])
    except (AttributeError, IndexError, TypeError, ValueError):
        body1 = -1
        body2 = -1
    return {
        "index": int(index),
        "geom1": geom1,
        "geom2": geom2,
        "geom1_name": _geom_id_to_name(model, geom1),
        "geom2_name": _geom_id_to_name(model, geom2),
        "body1": body1,
        "body2": body2,
        "body1_name": _body_id_to_name(model, body1),
        "body2_name": _body_id_to_name(model, body2),
        "dist_m": dist,
    }


def contact_summary(
    env: Any,
    *,
    target_geom_ids: Sequence[int],
    plate_geom_ids: Sequence[int],
    gripper_geom_ids: Sequence[int],
) -> dict[str, Any]:
    """Return relevant MuJoCo contact pairs and body external-force gauges."""

    simulator = _sim(env)
    model = simulator.model
    data = simulator.data
    target = {int(value) for value in target_geom_ids}
    plate = {int(value) for value in plate_geom_ids}
    gripper = {int(value) for value in gripper_geom_ids}
    try:
        ncon = int(data.ncon)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("LIBERO MuJoCo data lacks ncon") from error
    relevant: list[dict[str, Any]] = []
    for index in range(max(0, ncon)):
        contact = data.contact[index]
        row = _contact_pair_row(model, contact, index)
        pair = {int(row["geom1"]), int(row["geom2"])}
        kind = None
        if pair & gripper and pair & target:
            kind = "gripper_object"
        elif pair & target and pair & plate:
            kind = "object_plate"
        elif pair & gripper and pair & plate:
            kind = "gripper_plate"
        if kind is not None:
            row["kind"] = kind
            relevant.append(row)
    def _kind_rows(kind: str) -> list[dict[str, Any]]:
        return [row for row in relevant if row.get("kind") == kind]

    def _min_distance(rows: Sequence[Mapping[str, Any]]) -> float | None:
        distances = [float(row["dist_m"]) for row in rows if row.get("dist_m") is not None]
        return min(distances) if distances else None

    forces: dict[str, list[float] | None] = {}
    cfrc_ext = getattr(data, "cfrc_ext", None)
    for label in ("target", "plate"):
        # Body ids are supplied by the caller through the geometry ownership;
        # infer them from the first geom when cfrc_ext is available.
        geom_set = target if label == "target" else plate
        inferred_body = None
        try:
            geom_bodies = np.asarray(model.geom_bodyid, dtype=np.int64).reshape(-1)
            if geom_set:
                inferred_body = int(geom_bodies[min(geom_set)])
        except (AttributeError, IndexError, TypeError, ValueError):
            inferred_body = None
        if cfrc_ext is not None and inferred_body is not None:
            try:
                value = np.asarray(cfrc_ext[inferred_body], dtype=np.float64).reshape(-1)
                if value.shape == (6,) and np.isfinite(value).all():
                    forces[label] = [float(item) for item in value]
                else:
                    forces[label] = None
            except (IndexError, TypeError, ValueError):
                forces[label] = None
        else:
            forces[label] = None
    object_rows = _kind_rows("gripper_object")
    plate_rows = _kind_rows("object_plate")
    gripper_plate_rows = _kind_rows("gripper_plate")
    return {
        "ncon": ncon,
        "relevant_pair_count": len(relevant),
        "pairs": relevant,
        "gripper_object_contact_count": len(object_rows),
        "object_plate_contact_count": len(plate_rows),
        "gripper_plate_contact_count": len(gripper_plate_rows),
        "gripper_object_min_dist_m": _min_distance(object_rows),
        "object_plate_min_dist_m": _min_distance(plate_rows),
        "gripper_plate_min_dist_m": _min_distance(gripper_plate_rows),
        "target_cfrc_ext": forces["target"],
        "plate_cfrc_ext": forces["plate"],
    }


def _vector_or_none(value: object, *, size: int) -> list[float] | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.shape != (int(size),) or not np.isfinite(result).all():
        return None
    return [float(item) for item in result]


def _distance(eef: np.ndarray, target: np.ndarray) -> dict[str, float]:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(eef, dtype=np.float64)
    return {
        "xy_m": float(np.linalg.norm(delta[:2])),
        "xyz_m": float(np.linalg.norm(delta)),
    }


def physical_snapshot(
    env: Any,
    observation: Mapping[str, Any],
    *,
    target_body: str,
    plate_body: str,
    geometry: Mapping[str, Any],
    initial_target_xyz: np.ndarray | None = None,
    initial_plate_xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    """Capture simulator state needed for timing attribution."""

    target_xyz = _body_position(env, target_body)
    plate_xyz = _body_position(env, plate_body)
    eef = _vector_or_none(observation.get("robot0_eef_pos"), size=3)
    gripper_qpos = _vector_or_none(observation.get("robot0_gripper_qpos"), size=2)
    row: dict[str, Any] = {
        "eef_xyz_m": eef,
        "gripper_qpos": gripper_qpos,
        "target_xyz_m": target_xyz.tolist(),
        "plate_xyz_m": plate_xyz.tolist(),
        "contacts": contact_summary(
            env,
            target_geom_ids=geometry["target_geom_ids"],
            plate_geom_ids=geometry["plate_geom_ids"],
            gripper_geom_ids=geometry["gripper_geom_ids"],
        ),
    }
    if eef is not None:
        eef_array = np.asarray(eef, dtype=np.float64)
        row["eef_target_distance"] = _distance(eef_array, target_xyz)
        row["eef_plate_distance"] = _distance(eef_array, plate_xyz)
    else:
        row["eef_target_distance"] = None
        row["eef_plate_distance"] = None
    if initial_target_xyz is not None:
        row["target_displacement_xyz_m"] = (
            target_xyz - np.asarray(initial_target_xyz, dtype=np.float64)
        ).tolist()
    if initial_plate_xyz is not None:
        row["plate_displacement_xyz_m"] = (
            plate_xyz - np.asarray(initial_plate_xyz, dtype=np.float64)
        ).tolist()
    return row


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(int(v) for v in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _refresh_observation_after_forward(env: Any) -> Mapping[str, Any]:
    post_process = getattr(env, "_post_process", None)
    update_observables = getattr(env, "_update_observables", None)
    inner = getattr(env, "env", env)
    getter = getattr(inner, "_get_observations", None)
    if not all(callable(value) for value in (post_process, update_observables, getter)):
        raise ValueError("LIBERO environment lacks the direct observation refresh path")
    post_process()
    update_observables(force=True)
    observation = getter()
    if not isinstance(observation, Mapping):
        raise ValueError("LIBERO direct observation refresh returned a non-mapping")
    return observation


def _reset_controller_goals(env: Any) -> int:
    inner = getattr(env, "env", env)
    robots = getattr(inner, "robots", None)
    if not isinstance(robots, (tuple, list)) or not robots:
        raise ValueError("LIBERO environment does not expose its robot controllers")
    count = 0
    for robot in robots:
        controller = getattr(robot, "controller", None)
        reset_goal = getattr(controller, "reset_goal", None)
        if not callable(reset_goal):
            raise ValueError("LIBERO OSC controller lacks reset_goal")
        reset_goal()
        count += 1
    return count


def restore_snapshot(
    env: Any,
    *,
    snapshot: np.ndarray,
    seed: int,
) -> tuple[Mapping[str, Any], int]:
    """Clear wrapper caches and restore the exact post-warmup physics state."""

    seeder = getattr(env, "seed", None)
    resetter = getattr(env, "reset", None)
    setter = getattr(env, "set_state", None)
    if not all(callable(value) for value in (seeder, resetter, setter)):
        raise ValueError("LIBERO environment lacks deterministic state restoration")
    seeder(int(seed))
    resetter()
    setter(np.asarray(snapshot, dtype=np.float64).copy())
    simulator = _sim(env)
    simulator.forward()
    count = _reset_controller_goals(env)
    return _refresh_observation_after_forward(env), count


def _resolve_task_metadata(suite: Any, task_id: int) -> tuple[str, str]:
    task = suite.get_task(int(task_id))
    name = getattr(task, "name", None)
    instruction = getattr(task, "language", None)
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = getattr(task, "language_instruction", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"LIBERO task {task_id} has no name")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"LIBERO task {task_id} has no language instruction")
    return name.strip(), " ".join(instruction.split())


def _plan_action(
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    *,
    previous_action: np.ndarray,
    instruction: str,
    reset: bool,
    image_side: int,
    quat_to_axisangle: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy_input = libero_policy_observation(
        observation,
        previous_action,
        quat_to_axisangle=quat_to_axisangle,
        expected_image_side=image_side,
    )
    expected = np.zeros(7, dtype=np.float32) if reset else previous_action
    if not np.array_equal(np.asarray(policy_input.action_state), expected):
        raise AssertionError("LIBERO policy action_state is not reset-zero/previous action")
    chunk = np.asarray(client.act(policy_input, instruction, reset=bool(reset)), dtype=np.float32)
    if chunk.shape != (LIBERO_POLICY_HORIZON, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"LIBERO bridge must return finite [{LIBERO_POLICY_HORIZON},7]")
    raw = chunk[0].copy()
    executed = execute_libero_action(raw)
    return raw, executed, chunk


def _run_baseline(
    *,
    env: Any,
    client: RemotePolicyClient,
    observation: Mapping[str, Any],
    instruction: str,
    geometry: Mapping[str, Any],
    target_body: str,
    plate_body: str,
    steps: int,
    warmup_steps: int,
    image_side: int,
    video_dir: Path,
    task_id: int,
    episode: int,
    video_fps: float,
    quat_to_axisangle: Any,
    bridge_identity: Mapping[str, Any],
    stop_on_success: bool,
) -> dict[str, Any]:
    previous_action = np.zeros(7, dtype=np.float32)
    initial_target = _body_position(env, target_body)
    initial_plate = _body_position(env, plate_body)
    recorder = _LiberoEpisodeVideoRecorder(
        video_dir,
        task_id=task_id,
        episode=episode,
        instruction=instruction,
        max_steps=steps,
        warmup_steps=warmup_steps,
        fps=video_fps,
        metadata={
            "diagnostic": "timing_attribution_baseline",
            "bridge_identity": dict(bridge_identity),
            "action_contract": libero_action_contract(),
            "contact_geometry": dict(geometry),
        },
    )
    recorder.capture_ready(observation)
    rows: list[dict[str, Any]] = []
    success = False
    try:
        for step_index in range(1, int(steps) + 1):
            pre = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
                initial_target_xyz=initial_target,
                initial_plate_xyz=initial_plate,
            )
            action_state_input = previous_action.copy()
            raw, executed, chunk = _plan_action(
                client,
                observation,
                previous_action=previous_action,
                instruction=instruction,
                reset=step_index == 1,
                image_side=image_side,
                quat_to_axisangle=quat_to_axisangle,
            )
            observation, step_success = _step_with_success(env, executed.copy())
            success = bool(success or step_success)
            post = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
                initial_target_xyz=initial_target,
                initial_plate_xyz=initial_plate,
            )
            recorder.capture_step(observation, step=step_index)
            rows.append(
                {
                    "step": step_index,
                    "action_state_input": action_state_input.tolist(),
                    "raw_action_row": raw.tolist(),
                    "executed_action": executed.tolist(),
                    "raw_action_chunk": chunk.tolist(),
                    "clipped_dims": np.flatnonzero(raw != executed).astype(int).tolist(),
                    "pre": pre,
                    "post": post,
                    "success": bool(step_success),
                }
            )
            previous_action = executed.copy()
            if stop_on_success and success:
                break
        audit = _action_audit(rows)
        video = recorder.finish(success=success, steps=len(rows), action_audit=audit)
    except BaseException as error:
        recorder.abort(error)
        raise
    return {
        "mode": "baseline",
        "success": bool(success),
        "steps": len(rows),
        "rows": rows,
        "action_audit": audit,
        "video": video,
        "initial_target_xyz_m": initial_target.tolist(),
        "initial_plate_xyz_m": initial_plate.tolist(),
        "final_target_xyz_m": rows[-1]["post"]["target_xyz_m"] if rows else initial_target.tolist(),
        "final_plate_xyz_m": rows[-1]["post"]["plate_xyz_m"] if rows else initial_plate.tolist(),
    }


def _run_delayed_replay(
    *,
    env: Any,
    baseline: Mapping[str, Any],
    observation: Mapping[str, Any],
    geometry: Mapping[str, Any],
    target_body: str,
    plate_body: str,
    delay_steps: int,
    close_threshold: float,
    warmup_steps: int,
    video_dir: Path,
    task_id: int,
    episode: int,
    video_fps: float,
    instruction: str,
    bridge_identity: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_rows = list(baseline["rows"])
    initial_target = _body_position(env, target_body)
    initial_plate = _body_position(env, plate_body)
    recorder = _LiberoEpisodeVideoRecorder(
        video_dir,
        task_id=task_id,
        episode=episode,
        instruction=instruction,
        max_steps=len(baseline_rows),
        warmup_steps=warmup_steps,
        fps=video_fps,
        metadata={
            "diagnostic": "timing_attribution_delayed_close_replay",
            "delay_steps": int(delay_steps),
            "close_threshold": float(close_threshold),
            "bridge_identity": dict(bridge_identity),
            "action_contract": libero_action_contract(),
            "contact_geometry": dict(geometry),
            "arm_replay_contract": "baseline executed arm[0:6] copied bit-for-bit",
        },
    )
    recorder.capture_ready(observation)
    previous_action = np.zeros(7, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    success = False
    suppressed_count = 0
    try:
        for source in baseline_rows:
            step_index = _strict_int(source.get("step"), name="baseline step", minimum=1)
            source_action = _finite_action(source.get("executed_action"), name="baseline action")
            action, suppressed = suppress_close_command(
                source_action,
                step_index=step_index,
                delay_steps=delay_steps,
                close_threshold=close_threshold,
            )
            suppressed_count += int(suppressed)
            pre = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
                initial_target_xyz=initial_target,
                initial_plate_xyz=initial_plate,
            )
            observation, step_success = _step_with_success(env, action.copy())
            success = bool(success or step_success)
            post = physical_snapshot(
                env,
                observation,
                target_body=target_body,
                plate_body=plate_body,
                geometry=geometry,
                initial_target_xyz=initial_target,
                initial_plate_xyz=initial_plate,
            )
            recorder.capture_step(observation, step=step_index)
            rows.append(
                {
                    "step": step_index,
                    "source_baseline_step": step_index,
                    "action_state_input": previous_action.tolist(),
                    "source_executed_action": source_action.tolist(),
                    "executed_action": action.tolist(),
                    "close_suppressed": bool(suppressed),
                    "arm_action_exact": bool(np.array_equal(action[:6], source_action[:6])),
                    "pre": pre,
                    "post": post,
                    "success": bool(step_success),
                }
            )
            previous_action = action.copy()
        audit = _action_audit(rows)
        audit["close_suppressed_count"] = int(suppressed_count)
        video = recorder.finish(success=success, steps=len(rows), action_audit=audit)
    except BaseException as error:
        recorder.abort(error)
        raise
    return {
        "mode": "delayed_close_replay",
        "delay_steps": int(delay_steps),
        "close_threshold": float(close_threshold),
        "success": bool(success),
        "steps": len(rows),
        "rows": rows,
        "action_audit": audit,
        "video": video,
        "initial_target_xyz_m": initial_target.tolist(),
        "initial_plate_xyz_m": initial_plate.tolist(),
        "final_target_xyz_m": rows[-1]["post"]["target_xyz_m"] if rows else initial_target.tolist(),
        "final_plate_xyz_m": rows[-1]["post"]["plate_xyz_m"] if rows else initial_plate.tolist(),
    }


def _action_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"steps": 0, "arm_action_rms": 0.0, "gripper_commands": []}
    actions = np.asarray([row["executed_action"] for row in rows], dtype=np.float64)
    return {
        "steps": int(actions.shape[0]),
        "arm_action_rms": float(np.sqrt(np.mean(np.square(actions[:, :6])))),
        "gripper_commands": [float(value) for value in actions[:, 6]],
        "gripper_positive_count": int(np.count_nonzero(actions[:, 6] > 0.05)),
        "gripper_negative_count": int(np.count_nonzero(actions[:, 6] < -0.05)),
        "executed_min": actions.min(axis=0).tolist(),
        "executed_max": actions.max(axis=0).tolist(),
    }


def _first_contact(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for row in rows:
        step = int(row["step"])
        contacts = row.get("post", {}).get("contacts", {})
        if int(contacts.get(key, 0)) > 0:
            return step
    return None


def _minimum_contact_distance(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get("post", {}).get("contacts", {}).get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return min(values) if values else None


def summarize_timing_pair(
    baseline: Mapping[str, Any],
    delayed: Mapping[str, Any],
    *,
    delay_steps: int,
) -> dict[str, Any]:
    """Summarize the causal comparison without treating it as a benchmark."""

    base_rows = list(baseline.get("rows", []))
    delayed_rows = list(delayed.get("rows", []))
    if len(base_rows) != len(delayed_rows):
        raise ValueError("baseline and delayed replay have different step counts")
    if not base_rows:
        raise ValueError("timing pair contains no rows")
    arm_deltas: list[np.ndarray] = []
    action_deltas: list[np.ndarray] = []
    for base, moved in zip(base_rows, delayed_rows):
        base_action = _finite_action(base["executed_action"], name="baseline action")
        moved_action = _finite_action(moved["executed_action"], name="delayed action")
        arm_deltas.append(moved_action[:6].astype(np.float64) - base_action[:6].astype(np.float64))
        action_deltas.append(moved_action.astype(np.float64) - base_action.astype(np.float64))
    arm = np.stack(arm_deltas)
    all_delta = np.stack(action_deltas)
    base_eef = np.asarray([row["post"]["eef_xyz_m"] for row in base_rows], dtype=np.float64)
    delayed_eef = np.asarray([row["post"]["eef_xyz_m"] for row in delayed_rows], dtype=np.float64)
    base_obj = np.asarray([row["post"]["target_xyz_m"] for row in base_rows], dtype=np.float64)
    delayed_obj = np.asarray([row["post"]["target_xyz_m"] for row in delayed_rows], dtype=np.float64)
    return {
        "delay_steps": int(delay_steps),
        "step_count": len(base_rows),
        "baseline_success": bool(baseline.get("success", False)),
        "delayed_success": bool(delayed.get("success", False)),
        "arm_action_delta_max_abs": float(np.max(np.abs(arm))),
        "arm_action_delta_rms": float(np.sqrt(np.mean(np.square(arm)))),
        "full_action_delta_rms": float(np.sqrt(np.mean(np.square(all_delta)))),
        "gripper_action_delta_rms": float(np.sqrt(np.mean(np.square(all_delta[:, 6])))),
        "delayed_close_suppressed_count": int(
            sum(bool(row.get("close_suppressed", False)) for row in delayed_rows)
        ),
        "first_gripper_object_contact": {
            "baseline_step": _first_contact(base_rows, "gripper_object_contact_count"),
            "delayed_step": _first_contact(delayed_rows, "gripper_object_contact_count"),
        },
        "first_object_plate_contact": {
            "baseline_step": _first_contact(base_rows, "object_plate_contact_count"),
            "delayed_step": _first_contact(delayed_rows, "object_plate_contact_count"),
        },
        "minimum_contact_distance_m": {
            "baseline_gripper_object": _minimum_contact_distance(
                base_rows, "gripper_object_min_dist_m"
            ),
            "delayed_gripper_object": _minimum_contact_distance(
                delayed_rows, "gripper_object_min_dist_m"
            ),
            "baseline_object_plate": _minimum_contact_distance(
                base_rows, "object_plate_min_dist_m"
            ),
            "delayed_object_plate": _minimum_contact_distance(
                delayed_rows, "object_plate_min_dist_m"
            ),
        },
        "eef_post_trajectory_delta_rms_m": float(
            np.sqrt(np.mean(np.square(delayed_eef - base_eef)))
        ),
        "final_eef_delta_xyz_m": (delayed_eef[-1] - base_eef[-1]).tolist(),
        "object_post_trajectory_delta_rms_m": float(
            np.sqrt(np.mean(np.square(delayed_obj - base_obj)))
        ),
        "final_object_delta_xyz_m": (delayed_obj[-1] - base_obj[-1]).tolist(),
        "baseline_final_target_displacement_m": float(
            np.linalg.norm(base_obj[-1] - base_obj[0])
        ),
        "delayed_final_target_displacement_m": float(
            np.linalg.norm(delayed_obj[-1] - delayed_obj[0])
        ),
    }


def run_timing_probe(
    *,
    suite_name: str,
    task_id: int,
    init_state: int,
    output: Path,
    video_dir: Path,
    endpoint: str,
    timeout: float,
    checkpoint: str | Path,
    steps: int = DEFAULT_STEPS,
    delay_steps: int = DEFAULT_DELAY_STEPS,
    close_threshold: float = DEFAULT_CLOSE_THRESHOLD,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    image_side: int = DEFAULT_IMAGE_SIDE,
    video_fps: float = DEFAULT_VIDEO_FPS,
    seed: int = 10000,
    target_body: str = DEFAULT_TARGET_BODY,
    plate_body: str = DEFAULT_PLATE_BODY,
    gripper_geom_ids: Sequence[int] | None = None,
    stop_on_success: bool = False,
) -> dict[str, Any]:
    """Run one fixed-snapshot baseline/replay pair and write JSON."""

    output = Path(output).expanduser()
    video_dir = Path(video_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite timing output: {output}")
    if video_dir.exists():
        raise FileExistsError(f"refusing existing timing video directory: {video_dir}")
    suite_name = str(suite_name).strip().lower()
    task_id = _strict_int(task_id, name="task_id", minimum=0)
    init_state = _strict_int(init_state, name="init_state", minimum=0)
    steps = _strict_int(steps, name="steps", minimum=1)
    delay_steps = _strict_int(delay_steps, name="delay_steps", minimum=0)
    if delay_steps > steps:
        raise ValueError("delay_steps cannot exceed steps")
    warmup_steps = _strict_int(warmup_steps, name="warmup_steps", minimum=0)
    image_side = _strict_int(image_side, name="image_side", minimum=1)
    timeout = _finite_float(timeout, name="timeout")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    close_threshold = _finite_float(close_threshold, name="close_threshold")
    if close_threshold < 0.0:
        raise ValueError("close_threshold must be non-negative")
    video_fps = _finite_float(video_fps, name="video_fps")
    if video_fps <= 0.0:
        raise ValueError("video_fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)

    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_libero_bridge_health(health_before, allow_smoke_policy=False)
    bridge_identity = libero_bridge_identity(health_before, allow_smoke_policy=False)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle

    benchmark_map = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_map:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}")
    suite = benchmark_map[suite_name](task_order_index=0)
    if task_id >= int(suite.get_num_tasks()):
        raise ValueError(f"task_id {task_id} is outside suite {suite_name}")
    task_name, instruction = _resolve_task_metadata(suite, task_id)
    init_states = np.asarray(suite.get_task_init_states(task_id), dtype=np.float64)
    if init_states.ndim != 2 or init_state >= init_states.shape[0]:
        raise ValueError("requested init_state is absent from LIBERO fixed states")
    bddl_path = str(suite.get_task_bddl_file_path(task_id))

    payload: dict[str, Any] = {
        "schema": TIMING_PROBE_SCHEMA,
        "diagnostic_only": True,
        "complete": False,
        "scope": "deployment-only causal timing attribution; not an official LIBERO score",
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task_name,
        "instruction": instruction,
        "init_state": init_state,
        "steps": steps,
        "delay_steps": delay_steps,
        "close_threshold": close_threshold,
        "warmup_steps": warmup_steps,
        "image_side": image_side,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "endpoint": endpoint,
        "bridge_identity": bridge_identity,
        "protocol": {
            "snapshot": "one official init state plus five/selected zero-action warmup, restored identically",
            "baseline": "normal bridge inference; execute only row zero of each [24,7] chunk",
            "delayed": "copy baseline executed arm[0:6] exactly; replace positive close prefix commands with zero",
            "success": "evaluator-only check_success after each policy step",
        },
        "cases": {},
    }
    atomic_json(output, payload)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=image_side,
        camera_widths=image_side,
    )
    try:
        env.seed(int(seed))
        env.reset()
        initial_observation = _set_init_state_observation(env, init_states[init_state])
        zero = np.zeros(7, dtype=np.float32)
        observation = initial_observation
        for _ in range(warmup_steps):
            observation = _step_observation_only(env, zero.copy())
        snapshot = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        geometry = resolve_contact_geometries(
            env,
            target_body=target_body,
            plate_body=plate_body,
            gripper_geom_ids=gripper_geom_ids,
        )
        ready_target = _body_position(env, target_body)
        ready_plate = _body_position(env, plate_body)
        ready = physical_snapshot(
            env,
            observation,
            target_body=target_body,
            plate_body=plate_body,
            geometry=geometry,
            initial_target_xyz=ready_target,
            initial_plate_xyz=ready_plate,
        )

        baseline_observation, controller_count = restore_snapshot(
            env, snapshot=snapshot, seed=int(seed)
        )
        baseline = _run_baseline(
            env=env,
            client=client,
            observation=baseline_observation,
            instruction=instruction,
            geometry=geometry,
            target_body=target_body,
            plate_body=plate_body,
            steps=steps,
            warmup_steps=warmup_steps,
            image_side=image_side,
            video_dir=video_dir / "baseline",
            task_id=task_id,
            episode=init_state,
            video_fps=video_fps,
            quat_to_axisangle=quat2axisangle,
            bridge_identity=bridge_identity,
            stop_on_success=bool(stop_on_success),
        )
        delayed_observation, delayed_controller_count = restore_snapshot(
            env, snapshot=snapshot, seed=int(seed)
        )
        if delayed_controller_count != controller_count:
            raise RuntimeError("controller inventory changed between timing variants")
        delayed = _run_delayed_replay(
            env=env,
            baseline=baseline,
            observation=delayed_observation,
            geometry=geometry,
            target_body=target_body,
            plate_body=plate_body,
            delay_steps=delay_steps,
            close_threshold=close_threshold,
            warmup_steps=warmup_steps,
            video_dir=video_dir / "delayed_close",
            task_id=task_id,
            episode=init_state,
            video_fps=video_fps,
            instruction=instruction,
            bridge_identity=bridge_identity,
        )
        comparison = summarize_timing_pair(
            baseline, delayed, delay_steps=delay_steps
        )
        payload["cases"][str(init_state)] = {
            "init_state": init_state,
            "controller_count": controller_count,
            "snapshot_length": int(snapshot.size),
            "snapshot_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
            "contact_geometry": geometry,
            "ready": ready,
            "baseline": baseline,
            "delayed_close": delayed,
            "comparison": comparison,
        }
        atomic_json(output, payload)
    finally:
        env.close()

    health_after = client.health()
    validate_libero_bridge_health(health_after, allow_smoke_policy=False)
    if libero_bridge_identity(health_after, allow_smoke_policy=False) != bridge_identity:
        raise RuntimeError("bridge identity changed during timing probe")
    payload["bridge_health_before"] = health_before
    payload["bridge_health_after"] = health_after
    payload["complete"] = True
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--init-state", type=int, default=DEFAULT_INIT_STATE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--delay-steps", type=int, default=DEFAULT_DELAY_STEPS)
    parser.add_argument("--close-threshold", type=float, default=DEFAULT_CLOSE_THRESHOLD)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--image-side", type=int, default=DEFAULT_IMAGE_SIDE)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--target-body", default=DEFAULT_TARGET_BODY)
    parser.add_argument("--plate-body", default=DEFAULT_PLATE_BODY)
    parser.add_argument("--gripper-geom-ids", type=int, nargs="+", default=None)
    parser.add_argument("--stop-on-success", action="store_true")
    args = parser.parse_args()
    result = run_timing_probe(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state=args.init_state,
        output=args.output,
        video_dir=args.video_dir,
        endpoint=args.endpoint,
        timeout=args.timeout,
        checkpoint=args.checkpoint,
        steps=args.steps,
        delay_steps=args.delay_steps,
        close_threshold=args.close_threshold,
        warmup_steps=args.warmup_steps,
        image_side=args.image_side,
        video_fps=args.video_fps,
        seed=args.seed,
        target_body=args.target_body,
        plate_body=args.plate_body,
        gripper_geom_ids=args.gripper_geom_ids,
        stop_on_success=args.stop_on_success,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "TIMING_PROBE_SCHEMA",
    "contact_summary",
    "physical_snapshot",
    "resolve_contact_geometries",
    "run_timing_probe",
    "summarize_timing_pair",
    "suppress_close_command",
]
