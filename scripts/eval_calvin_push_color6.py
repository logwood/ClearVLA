#!/usr/bin/env python3
"""Run a six-task CALVIN push panel with physical closed-loop diagnostics.

The panel uses initial conditions whose first official CALVIN sequence task is
the requested push.  Every environment step consumes a fresh observation and
executes only row zero of a newly sampled 24-row plan.  Official task-oracle
success is the score; object displacement and contact traces only explain
failures and never replace the oracle.

Run the policy bridge in the checkpoint environment and this evaluator in the
official CALVIN environment.  The bridge is loopback-only by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.calvin_eval import (
    CALVIN_OFFICIAL_SUBTASK_STEPS,
    CalvinBridgeModel,
    _CalvinSequenceVideoRecorder,
    _environment,
    _official_task_assets,
    _save_rgb,
    validate_bridge_health,
)
from clearvla.benchmarks.io import atomic_json, atomic_npz

SCHEMA = "clearvla-calvin-push-color6-closed-loop-v1"
BLOCK_ORDER = ("block_blue", "block_red", "block_pink")
PUSH_THRESHOLD_M = 0.1
MOTION_EPSILON_M = 0.001
DIRECTION_EPSILON_M = 0.01
COLLATERAL_MOTION_EPSILON_M = 0.02
SCREENING_GATE_THRESHOLDS = {
    "minimum_official_success_rate": 2.0 / 6.0,
    "minimum_direction_correct_rate": 5.0 / 6.0,
    "minimum_target_is_most_moved_rate": 5.0 / 6.0,
    "minimum_target_contact_rate": 5.0 / 6.0,
    "maximum_non_target_collateral_motion_rate": 1.0 / 6.0,
    "required_successful_directions": ("left", "right"),
    "minimum_successful_colors": 2,
}


@dataclass(frozen=True)
class PushTaskSpec:
    task: str
    object_name: str
    direction: str
    direction_sign: int


PUSH_TASKS = (
    PushTaskSpec("push_blue_block_left", "block_blue", "left", -1),
    PushTaskSpec("push_blue_block_right", "block_blue", "right", 1),
    PushTaskSpec("push_red_block_left", "block_red", "left", -1),
    PushTaskSpec("push_red_block_right", "block_red", "right", 1),
    PushTaskSpec("push_pink_block_left", "block_pink", "left", -1),
    PushTaskSpec("push_pink_block_right", "block_pink", "right", 1),
)
TASK_BY_NAME = {value.task: value for value in PUSH_TASKS}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _require_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"closed-loop output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _select_first_task_cases(
    sequences: Iterable[tuple[Mapping[str, object], Sequence[str]]],
    *,
    trials_per_task: int,
) -> list[dict[str, Any]]:
    if trials_per_task <= 0:
        raise ValueError("trials_per_task must be positive")
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, (initial_state, tasks) in enumerate(sequences):
        if not tasks:
            continue
        task = str(tasks[0])
        if task not in TASK_BY_NAME or len(selected[task]) >= trials_per_task:
            continue
        selected[task].append(
            {
                "source_sequence_index": int(source_index),
                "task": task,
                "trial": len(selected[task]) + 1,
                "initial_state": {str(key): value for key, value in initial_state.items()},
                "source_sequence": [str(value) for value in tasks],
            }
        )
    missing = {
        spec.task: trials_per_task - len(selected[spec.task])
        for spec in PUSH_TASKS
        if len(selected[spec.task]) < trials_per_task
    }
    if missing:
        raise ValueError(
            "official sequence pool lacks enough first-task push cases: "
            + json.dumps(missing, sort_keys=True)
        )
    return [
        case
        for spec in PUSH_TASKS
        for case in selected[spec.task]
    ]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"CALVIN info {name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _movable_objects(info: Mapping[str, Any]) -> Mapping[str, Any]:
    scene = _mapping(info.get("scene_info"), name="scene_info")
    return _mapping(scene.get("movable_objects"), name="scene_info.movable_objects")


def _robot_uid(info: Mapping[str, Any]) -> int:
    robot = _mapping(info.get("robot_info"), name="robot_info")
    return int(robot["uid"])


def _object_position(info: Mapping[str, Any], object_name: str) -> np.ndarray:
    objects = _movable_objects(info)
    item = _mapping(objects.get(object_name), name=f"movable_objects.{object_name}")
    value = np.asarray(item.get("current_pos"), dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"CALVIN {object_name} current_pos must be finite [3]")
    return value


def _contacts(info: Mapping[str, Any], object_name: str) -> tuple[tuple[Any, ...], ...]:
    objects = _movable_objects(info)
    item = _mapping(objects.get(object_name), name=f"movable_objects.{object_name}")
    raw = item.get("contacts", ())
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"CALVIN {object_name} contacts must be a sequence")
    return tuple(tuple(value) for value in raw)


def _robot_contact(info: Mapping[str, Any], object_name: str) -> bool:
    uid = _robot_uid(info)
    for contact in _contacts(info, object_name):
        if len(contact) > 2 and (int(contact[1]) == uid or int(contact[2]) == uid):
            return True
    return False


def _surface_contacts(info: Mapping[str, Any], object_name: str) -> set[tuple[int, int]]:
    uid = _robot_uid(info)
    result: set[tuple[int, int]] = set()
    for contact in _contacts(info, object_name):
        if len(contact) <= 4:
            continue
        body = int(contact[2])
        link = int(contact[4])
        if body != uid:
            result.add((body, link))
    return result


def _scene_state(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.stack(
        [_object_position(info, name) for name in BLOCK_ORDER]
    ).astype(np.float64)
    contacts = np.asarray(
        [_robot_contact(info, name) for name in BLOCK_ORDER], dtype=np.bool_
    )
    return positions, contacts


def _first_index(mask: np.ndarray, *, include_initial: bool = False) -> int | None:
    value = np.asarray(mask, dtype=np.bool_).reshape(-1)
    start = 0 if include_initial else 1
    indices = np.flatnonzero(value[start:])
    return int(indices[0] + start) if indices.size else None


def _rollout_diagnostics(
    task: str,
    positions: np.ndarray,
    robot_contacts: np.ndarray,
    surface_preserved: np.ndarray,
    *,
    success: bool,
) -> dict[str, Any]:
    spec = TASK_BY_NAME[task]
    value = np.asarray(positions, dtype=np.float64)
    contacts = np.asarray(robot_contacts, dtype=np.bool_)
    surface = np.asarray(surface_preserved, dtype=np.bool_).reshape(-1)
    if value.ndim != 3 or value.shape[1:] != (len(BLOCK_ORDER), 3):
        raise ValueError(f"object positions must be [T,{len(BLOCK_ORDER)},3]")
    if contacts.shape != value.shape[:2] or surface.shape != (value.shape[0],):
        raise ValueError("closed-loop diagnostic traces do not align")

    target_index = BLOCK_ORDER.index(spec.object_name)
    displacement = value - value[:1]
    target_x = displacement[:, target_index, 0]
    signed_progress = float(spec.direction_sign) * target_x
    signed_step_progress = float(spec.direction_sign) * np.diff(
        target_x, prepend=target_x[0]
    )
    opposite_progress = -signed_progress
    target_abs_x = np.abs(target_x)
    per_object_max_abs_x = np.abs(displacement[..., 0]).max(axis=0)
    per_object_max_planar = np.linalg.norm(displacement[..., :2], axis=-1).max(axis=0)
    per_object_max_spatial = np.linalg.norm(displacement, axis=-1).max(axis=0)
    per_object_final_x = displacement[-1, :, 0]
    target_contact = contacts[:, target_index]
    contact_positive_progress = target_contact & (
        signed_step_progress > MOTION_EPSILON_M
    )
    target_is_most_moved = bool(
        per_object_max_planar[target_index] >= per_object_max_planar.max() - 1e-9
        and per_object_max_planar[target_index] >= DIRECTION_EPSILON_M
    )
    non_target_indices = [index for index in range(len(BLOCK_ORDER)) if index != target_index]
    collateral_indices = [
        index
        for index in non_target_indices
        if float(per_object_max_spatial[index]) >= COLLATERAL_MOTION_EPSILON_M
    ]
    largest_non_target = float(per_object_max_spatial[non_target_indices].max())
    collateral_motion = bool(collateral_indices)
    dominant_non_target_motion = bool(
        largest_non_target
        > float(per_object_max_spatial[target_index]) + 0.005
    )
    max_signed = float(signed_progress.max())
    max_opposite = float(opposite_progress.max())
    direction_correct = bool(
        max_signed >= DIRECTION_EPSILON_M and max_signed > max_opposite
    )

    failure_modes: list[str] = []
    if not success:
        if not bool(target_contact.any()):
            failure_modes.append("no_target_contact")
        elif float(target_abs_x.max()) < DIRECTION_EPSILON_M:
            failure_modes.append("target_contact_no_motion")
        if max_opposite >= max(DIRECTION_EPSILON_M, max_signed):
            failure_modes.append("wrong_direction")
        if collateral_motion:
            failure_modes.append("non_target_collateral_motion")
        if DIRECTION_EPSILON_M <= max_signed < PUSH_THRESHOLD_M:
            failure_modes.append("insufficient_progress")
        if max_signed >= PUSH_THRESHOLD_M:
            failure_modes.append("oracle_surface_or_stability_failure")
        if not failure_modes:
            failure_modes.append("no_effective_progress")

    return {
        "task": task,
        "target_object": spec.object_name,
        "direction": spec.direction,
        "world_x_sign": spec.direction_sign,
        "official_threshold_m": PUSH_THRESHOLD_M,
        "official_success": bool(success),
        "steps_recorded": int(value.shape[0] - 1),
        "initial_position_m": value[0, target_index].tolist(),
        "final_position_m": value[-1, target_index].tolist(),
        "target_final_displacement_m": displacement[-1, target_index].tolist(),
        "target_final_signed_progress_m": float(signed_progress[-1]),
        "target_max_signed_progress_m": max_signed,
        "target_max_opposite_progress_m": max_opposite,
        "target_max_abs_x_displacement_m": float(target_abs_x.max()),
        "direction_correct_1cm": direction_correct,
        "first_target_contact_step": _first_index(target_contact),
        "first_target_motion_step": _first_index(target_abs_x > MOTION_EPSILON_M),
        "first_observed_signed_progress_step": _first_index(
            signed_progress > MOTION_EPSILON_M
        ),
        "first_contact_positive_progress_step": _first_index(
            contact_positive_progress
        ),
        "target_contact_steps": int(target_contact[1:].sum()),
        "target_contact_positive_progress_steps": int(
            contact_positive_progress[1:].sum()
        ),
        "surface_contact_preserved_final": bool(surface[-1]),
        "surface_contact_preserved_fraction": float(surface[1:].mean())
        if surface.shape[0] > 1
        else 1.0,
        "per_object_final_x_displacement_m": {
            name: float(per_object_final_x[index])
            for index, name in enumerate(BLOCK_ORDER)
        },
        "per_object_max_abs_x_displacement_m": {
            name: float(per_object_max_abs_x[index])
            for index, name in enumerate(BLOCK_ORDER)
        },
        "per_object_max_planar_displacement_m": {
            name: float(per_object_max_planar[index])
            for index, name in enumerate(BLOCK_ORDER)
        },
        "per_object_max_spatial_displacement_m": {
            name: float(per_object_max_spatial[index])
            for index, name in enumerate(BLOCK_ORDER)
        },
        "target_is_most_moved_planar_1cm": target_is_most_moved,
        "non_target_collateral_motion_2cm": collateral_motion,
        "collateral_objects_2cm": [BLOCK_ORDER[index] for index in collateral_indices],
        "dominant_non_target_motion": dominant_non_target_motion,
        "failure_modes": failure_modes,
    }


def _checkpoint_identity(health: Mapping[str, Any]) -> dict[str, Any]:
    deployment = _mapping(health.get("deployment"), name="bridge.deployment")
    checkpoint = _mapping(deployment.get("checkpoint"), name="deployment.checkpoint")
    architecture = _mapping(
        deployment.get("architecture"), name="deployment.architecture"
    )
    sampling = _mapping(deployment.get("sampling"), name="deployment.sampling")
    policy_seed = sampling.get("policy_seed")
    if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
        raise ValueError("bridge policy seed must be an integer")
    if policy_seed < 0:
        raise ValueError("bridge policy seed must be non-negative")
    if sampling.get("reset_to_policy_seed_each_episode") is not True:
        raise ValueError("bridge policy RNG must reset to its named seed each episode")
    flow_schedule_sha256 = architecture.get("flow_schedule_sha256")
    if not isinstance(flow_schedule_sha256, str) or len(flow_schedule_sha256) != 64:
        raise ValueError("bridge flow schedule identity is absent or malformed")
    flow_schedule = _mapping(
        architecture.get("flow_schedule"), name="deployment.architecture.flow_schedule"
    )
    encoded_schedule = json.dumps(
        flow_schedule,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if hashlib.sha256(encoded_schedule).hexdigest() != flow_schedule_sha256:
        raise ValueError("bridge flow schedule digest is inconsistent")
    result = {
        key: checkpoint.get(key)
        for key in ("path", "sha256", "epoch", "global_step", "git_commit", "source_digest")
    }
    result.update(
        {
            "flow_schedule": dict(flow_schedule),
            "flow_schedule_sha256": flow_schedule_sha256,
            "policy_seed": policy_seed,
            "policy_rng_reset_each_episode": True,
        }
    )
    return result


def _require_rollout_bridge_integrity(
    health: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    executed_steps: int,
) -> int:
    validate_bridge_health(health, allow_smoke_policy=False)
    if _checkpoint_identity(health) != dict(checkpoint):
        raise RuntimeError("policy bridge checkpoint identity changed during panel")
    actual = health.get("history_time_index")
    if isinstance(actual, bool) or not isinstance(actual, int):
        raise RuntimeError(
            f"policy bridge returned invalid history_time_index {actual!r}"
        )
    expected = int(executed_steps) - 1
    if actual != expected:
        raise RuntimeError(
            "policy bridge history is misaligned after rollout: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for report in reports:
        by_task[str(report["task"])].append(report)
    per_task: dict[str, Any] = {}
    for spec in PUSH_TASKS:
        rows = by_task[spec.task]
        successes = [bool(row["success"]) for row in rows]
        diagnostics = [cast(Mapping[str, Any], row["trajectory_diagnostics"]) for row in rows]
        per_task[spec.task] = {
            "trials": len(rows),
            "successes": int(sum(successes)),
            "success_rate": float(np.mean(successes)) if rows else None,
            "direction_correct_rate": float(
                np.mean([bool(item["direction_correct_1cm"]) for item in diagnostics])
            )
            if rows
            else None,
            "target_is_most_moved_rate": float(
                np.mean(
                    [
                        bool(item["target_is_most_moved_planar_1cm"])
                        for item in diagnostics
                    ]
                )
            )
            if rows
            else None,
            "target_contact_rate": float(
                np.mean(
                    [item["first_target_contact_step"] is not None for item in diagnostics]
                )
            )
            if rows
            else None,
            "non_target_collateral_motion_rate": float(
                np.mean(
                    [
                        bool(item["non_target_collateral_motion_2cm"])
                        for item in diagnostics
                    ]
                )
            )
            if rows
            else None,
            "mean_max_signed_progress_m": float(
                np.mean([float(item["target_max_signed_progress_m"]) for item in diagnostics])
            )
            if rows
            else None,
            "failure_mode_counts": {
                mode: sum(
                    mode in cast(list[str], item["failure_modes"])
                    for item in diagnostics
                )
                for mode in sorted(
                    {
                        mode
                        for item in diagnostics
                        for mode in cast(list[str], item["failure_modes"])
                    }
                )
            },
        }
    success_values = [bool(report["success"]) for report in reports]
    diagnostic_values = [
        cast(Mapping[str, Any], report["trajectory_diagnostics"])
        for report in reports
    ]
    return {
        "trials": len(reports),
        "successes": int(sum(success_values)),
        "micro_success_rate": float(np.mean(success_values)) if reports else None,
        "macro_task_success_rate": float(
            np.mean([float(per_task[spec.task]["success_rate"]) for spec in PUSH_TASKS])
        )
        if reports
        else None,
        "direction_correct_rate": float(
            np.mean(
                [bool(item["direction_correct_1cm"]) for item in diagnostic_values]
            )
        )
        if reports
        else None,
        "target_is_most_moved_rate": float(
            np.mean(
                [
                    bool(item["target_is_most_moved_planar_1cm"])
                    for item in diagnostic_values
                ]
            )
        )
        if reports
        else None,
        "target_contact_rate": float(
            np.mean(
                [
                    item["first_target_contact_step"] is not None
                    for item in diagnostic_values
                ]
            )
        )
        if reports
        else None,
        "non_target_collateral_motion_rate": float(
            np.mean(
                [
                    bool(item["non_target_collateral_motion_2cm"])
                    for item in diagnostic_values
                ]
            )
        )
        if reports
        else None,
        "all_six_tasks_attempted": all(bool(by_task[spec.task]) for spec in PUSH_TASKS),
        "all_six_tasks_have_a_success": all(
            bool(per_task[spec.task]["successes"]) for spec in PUSH_TASKS
        ),
        "per_task": per_task,
    }


def _screening_gate(
    reports: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]
) -> dict[str, Any]:
    """Pre-registered triage gate; passing only authorizes a larger panel."""

    if not reports:
        raise ValueError("screening gate requires at least one rollout")
    successes = [report for report in reports if bool(report["success"])]
    successful_directions = {
        TASK_BY_NAME[str(report["task"])].direction for report in successes
    }
    successful_colors = {
        TASK_BY_NAME[str(report["task"])].object_name for report in successes
    }
    thresholds = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in SCREENING_GATE_THRESHOLDS.items()
    }
    checks = {
        "all_six_tasks_attempted": bool(aggregate["all_six_tasks_attempted"]),
        "official_success_rate": float(aggregate["micro_success_rate"])
        >= float(thresholds["minimum_official_success_rate"]),
        "direction_correct_rate": float(aggregate["direction_correct_rate"])
        >= float(thresholds["minimum_direction_correct_rate"]),
        "target_is_most_moved_rate": float(aggregate["target_is_most_moved_rate"])
        >= float(thresholds["minimum_target_is_most_moved_rate"]),
        "target_contact_rate": float(aggregate["target_contact_rate"])
        >= float(thresholds["minimum_target_contact_rate"]),
        "non_target_collateral_motion_rate": float(
            aggregate["non_target_collateral_motion_rate"]
        )
        <= float(thresholds["maximum_non_target_collateral_motion_rate"]),
        "both_directions_have_official_success": successful_directions
        == set(SCREENING_GATE_THRESHOLDS["required_successful_directions"]),
        "at_least_two_colors_have_official_success": len(successful_colors) >= 2,
    }
    passed = all(checks.values())
    borderline = (
        not passed
        and bool(successes)
        and float(aggregate["direction_correct_rate"]) >= 4.0 / 6.0
        and float(aggregate["target_is_most_moved_rate"]) >= 4.0 / 6.0
    )
    return {
        "schema": "clearvla-calvin-push-color6-screening-gate-v1",
        "disposition": "advance" if passed else ("borderline" if borderline else "reject"),
        "passed_for_expanded_panel": passed,
        "thresholds": thresholds,
        "checks": checks,
        "successful_directions": sorted(successful_directions),
        "successful_colors": sorted(successful_colors),
        "scope": (
            "triage only; passing does not establish benchmark performance or "
            "statistical reliability"
        ),
        "next_if_passed": "run at least three fresh official initial conditions per task",
    }


def evaluate_push_color6(
    *,
    dataset_root: Path,
    output_dir: Path,
    endpoint: str,
    timeout: float,
    trials_per_task: int,
    source_sequence_count: int,
    sequence_workers: int,
    max_steps: int,
    record_video: bool,
    video_fps: float,
    expected_policy_seed: int,
) -> dict[str, Any]:
    if trials_per_task <= 0:
        raise ValueError("trials_per_task must be positive")
    if source_sequence_count < len(PUSH_TASKS) * trials_per_task:
        raise ValueError("source_sequence_count is too small for the requested panel")
    if max_steps <= 0 or max_steps > CALVIN_OFFICIAL_SUBTASK_STEPS:
        raise ValueError("max_steps must be in [1,360]")
    if sequence_workers <= 0:
        raise ValueError("sequence_workers must be positive")
    if not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video_fps must be finite and positive")
    if expected_policy_seed < 0:
        raise ValueError("expected_policy_seed must be non-negative")
    _require_new_output_dir(output_dir)

    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_bridge_health(health_before, allow_smoke_policy=False)
    checkpoint = _checkpoint_identity(health_before)
    if int(checkpoint["policy_seed"]) != int(expected_policy_seed):
        raise ValueError(
            "policy bridge seed differs from the requested closed-loop protocol: "
            f"expected {expected_policy_seed}, got {checkpoint['policy_seed']}"
        )

    from calvin_agent.evaluation.multistep_sequences import get_sequences

    source_sequences = get_sequences(
        source_sequence_count, num_workers=sequence_workers
    )
    cases = _select_first_task_cases(
        source_sequences, trials_per_task=trials_per_task
    )
    manifest = {
        "schema": SCHEMA,
        "protocol": {
            "initial_condition_source": (
                "first subtask of official get_sequences(source_sequence_count)"
            ),
            "source_sequence_count": int(source_sequence_count),
            "sequence_workers": int(sequence_workers),
            "trials_per_task": int(trials_per_task),
            "max_steps": int(max_steps),
            "execute_rows": 1,
            "official_sequence_generator_seed": 0,
            "policy_rng_reset_each_trial": True,
            "policy_seed": int(checkpoint["policy_seed"]),
            "expected_policy_seed": int(expected_policy_seed),
            "policy_seed_reused_across_trials": True,
            "flow_schedule_sha256": checkpoint["flow_schedule_sha256"],
            "official_oracle_is_primary_score": True,
            "screening_gate_thresholds": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in SCREENING_GATE_THRESHOLDS.items()
            },
        },
        "tasks": [spec.task for spec in PUSH_TASKS],
        "cases": _jsonable(cases),
    }
    atomic_json(output_dir / "manifest.json", manifest)

    official, conf_dir, oracle, annotations = _official_task_assets()
    env = _environment(dataset_root, show_gui=False)
    reports: list[dict[str, Any]] = []
    started_panel = time.perf_counter()
    try:
        for case_index, case in enumerate(cases, start=1):
            task = str(case["task"])
            spec = TASK_BY_NAME[task]
            initial_state = cast(Mapping[str, object], case["initial_state"])
            instruction = str(annotations[task][0])
            trial = int(case["trial"])
            rollout_dir = output_dir / f"{task}__trial{trial:02d}"
            rollout_dir.mkdir(parents=True, exist_ok=False)
            model = CalvinBridgeModel(client, execute_rows=1)
            recorder = (
                _CalvinSequenceVideoRecorder(
                    rollout_dir / "video",
                    sequence_index=case_index,
                    initial_state=initial_state,
                    eval_sequence=[task],
                    max_subtask_steps=max_steps,
                    fps=video_fps,
                )
                if record_video
                else None
            )
            video_report: dict[str, object] | None = None
            robot_obs, scene_obs = official.get_env_state_for_initial_condition(
                dict(initial_state)
            )
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            observation = env.get_obs()
            start_info = env.get_info()
            initial_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
            _save_rgb(rollout_dir / "initial_static.png", initial_rgb["rgb_static"])
            _save_rgb(rollout_dir / "initial_wrist.png", initial_rgb["rgb_gripper"])
            if recorder is not None:
                recorder.start_task(task, instruction)
                recorder.capture_task_start(observation)

            model.reset()
            robot_trajectory = [
                np.asarray(observation["robot_obs"], dtype=np.float32).copy()
            ]
            action_robot_obs: list[np.ndarray] = []
            first_positions, first_contacts = _scene_state(start_info)
            object_positions = [first_positions]
            object_contacts = [first_contacts]
            start_surface = _surface_contacts(start_info, spec.object_name)
            if not start_surface:
                raise ValueError(
                    f"official case starts {spec.object_name} without surface contact"
                )
            surface_preserved = [True]
            oracle_success = [False]
            success = False
            steps = 0
            started = time.perf_counter()
            try:
                for step in range(1, max_steps + 1):
                    action_robot_obs.append(
                        np.asarray(observation["robot_obs"], dtype=np.float32).copy()
                    )
                    action = model.step(observation, instruction)
                    observation, _reward, _done, current_info = env.step(action)
                    steps = step
                    robot_trajectory.append(
                        np.asarray(observation["robot_obs"], dtype=np.float32).copy()
                    )
                    positions, contacts = _scene_state(current_info)
                    object_positions.append(positions)
                    object_contacts.append(contacts)
                    surface_preserved.append(
                        bool(start_surface <= _surface_contacts(current_info, spec.object_name))
                    )
                    success = bool(
                        oracle.get_task_info_for_set(
                            start_info, current_info, {task}
                        )
                    )
                    oracle_success.append(success)
                    if recorder is not None:
                        recorder.capture_step(observation)
                    if success:
                        break
                health_after_rollout = client.health()
                actual_history = _require_rollout_bridge_integrity(
                    health_after_rollout,
                    checkpoint=checkpoint,
                    executed_steps=steps,
                )
                if recorder is not None:
                    recorder.finish_task(success)
                    video_report = recorder.finish_sequence(1 if success else 0)
            except BaseException as error:
                if recorder is not None:
                    recorder.abort_sequence(error)
                raise
            elapsed = time.perf_counter() - started
            final_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
            _save_rgb(rollout_dir / "final_static.png", final_rgb["rgb_static"])
            _save_rgb(rollout_dir / "final_wrist.png", final_rgb["rgb_gripper"])

            position_array = np.stack(object_positions).astype(np.float64)
            contact_array = np.stack(object_contacts).astype(np.bool_)
            surface_array = np.asarray(surface_preserved, dtype=np.bool_)
            arrays = model.action_arrays()
            arrays.update(
                {
                    "robot_obs_before_action": np.stack(action_robot_obs).astype(
                        np.float32
                    ),
                    "robot_obs_trajectory": np.stack(robot_trajectory).astype(
                        np.float32
                    ),
                    "object_positions": position_array,
                    "object_robot_contact": contact_array,
                    "target_surface_contact_preserved": surface_array,
                    "oracle_success": np.asarray(oracle_success, dtype=np.bool_),
                }
            )
            expected_history = steps - 1
            history_aligned = True
            atomic_npz(rollout_dir / "trajectory.npz", **arrays)
            diagnostics = _rollout_diagnostics(
                task,
                position_array,
                contact_array,
                surface_array,
                success=success,
            )
            report = {
                "schema": SCHEMA,
                "case_index": int(case_index),
                "source_sequence_index": int(case["source_sequence_index"]),
                "source_sequence": list(case["source_sequence"]),
                "task": task,
                "trial": trial,
                "instruction": instruction,
                "initial_state": _jsonable(initial_state),
                "success": bool(success),
                "steps": int(steps),
                "max_steps": int(max_steps),
                "elapsed_seconds": float(elapsed),
                "seconds_per_step": float(elapsed / steps) if steps else None,
                "checkpoint": checkpoint,
                "execution": {
                    "execute_rows": 1,
                    "planning_decisions": int(arrays["raw_chunks"].shape[0]),
                    "history_time_index_expected_after": expected_history,
                    "history_time_index_actual_after": actual_history,
                    "history_time_index_aligned": bool(history_aligned),
                },
                "action_audit": model.action_audit(),
                "trajectory_diagnostics": diagnostics,
                "video": {
                    "enabled": bool(record_video),
                    "path": (
                        str(rollout_dir / "video" / str(video_report["video"]))
                        if video_report is not None
                        else None
                    ),
                },
                "artifacts": {
                    "trajectory": "trajectory.npz",
                    "initial_static": "initial_static.png",
                    "initial_wrist": "initial_wrist.png",
                    "final_static": "final_static.png",
                    "final_wrist": "final_wrist.png",
                },
            }
            atomic_json(rollout_dir / "result.json", report)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "task": task,
                        "trial": trial,
                        "success": success,
                        "steps": steps,
                        "diagnostics": diagnostics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    health_after = client.health()
    validate_bridge_health(health_after, allow_smoke_policy=False)
    final_checkpoint = _checkpoint_identity(health_after)
    if final_checkpoint != checkpoint:
        raise RuntimeError("policy bridge checkpoint identity changed before summary")
    aggregate = _aggregate(reports)
    summary = {
        "schema": SCHEMA,
        "benchmark": "CALVIN official D environment and task oracle",
        "dataset_root": str(dataset_root.resolve()),
        "endpoint": endpoint,
        "checkpoint": checkpoint,
        "checkpoint_identity_stable": True,
        "integrity": {
            "complete": True,
            "checkpoint_identity_stable": True,
            "all_rollout_history_indices_aligned": True,
            "formal_aggregate_valid": True,
        },
        "protocol": manifest["protocol"],
        "official_assets": {
            "config_directory": str(conf_dir),
            "task_oracle": "callbacks/rollout/tasks/new_playtable_tasks.yaml",
            "annotations": "annotations/new_playtable_validation.yaml",
        },
        "rollouts": reports,
        "aggregate": aggregate,
        "screening_gate": _screening_gate(reports, aggregate),
        "elapsed_seconds": float(time.perf_counter() - started_panel),
        "interpretation": {
            "primary_score": "official task-oracle success",
            "right_direction": "+x",
            "left_direction": "-x",
            "diagnostic_only": (
                "contact, displacement, direction, and non-target-motion fields explain "
                "failure but do not override the official oracle"
            ),
            "statistical_unit": "one independently reset official initial condition",
        },
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18772")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--source-sequence-count", type=int, default=1000)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--expected-policy-seed", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_push_color6(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        endpoint=str(args.endpoint),
        timeout=float(args.timeout),
        trials_per_task=int(args.trials_per_task),
        source_sequence_count=int(args.source_sequence_count),
        sequence_workers=int(args.sequence_workers),
        max_steps=int(args.max_steps),
        record_video=bool(args.record_video),
        video_fps=float(args.video_fps),
        expected_policy_seed=int(args.expected_policy_seed),
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
