#!/usr/bin/env python3
"""Run a paired CALVIN target-y shift closed-loop diagnostic.

This is not a benchmark score.  It holds the symbolic initial condition,
robot state, distractor poses, policy checkpoint, instruction, and policy RNG
reset protocol fixed while translating only the requested target object's y
coordinate.  Each rollout saves its physical trajectory, inline video, and
exact causal-history snapshots at reset, high alignment, and the selected
low sweep point for a subsequent read-only internal-layer probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.calvin_eval import (
    CalvinBridgeModel,
    _CalvinSequenceVideoRecorder,
    _environment,
    _official_task_assets,
    _save_rgb,
    calvin_policy_observation,
    validate_bridge_health,
)
from clearvla.benchmarks.io import atomic_json, atomic_npz
from clearvla.simulation.history import CausalHistory, HistorySnapshot
from scripts.eval_calvin_push_color6 import (
    BLOCK_ORDER,
    TASK_BY_NAME,
    _checkpoint_identity,
    _jsonable,
    _require_new_output_dir,
    _require_rollout_bridge_integrity,
    _rollout_diagnostics,
    _scene_state,
    _surface_contacts,
)


SCHEMA = "clearvla-calvin-target-y-shift-closed-loop-v1"
SNAPSHOT_SCHEMA = "clearvla-calvin-causal-history-snapshot-v1"
SCENE_POSITION_SLICES = {
    "block_red": slice(6, 9),
    "block_blue": slice(12, 15),
    "block_pink": slice(18, 21),
}


@dataclass(frozen=True)
class SnapshotCandidate:
    snapshot: HistorySnapshot
    step: int
    tcp_xyz: np.ndarray
    target_xyz: np.ndarray
    relative_xyz: np.ndarray
    kind: str


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON mapping")
    return {str(key): value for key, value in payload.items()}


def _base_positions(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    payload = _load_mapping(path, name="base positions")
    if set(payload) != set(BLOCK_ORDER):
        raise ValueError(
            "base positions must contain exactly " + ", ".join(BLOCK_ORDER)
        )
    result: dict[str, np.ndarray] = {}
    for name in BLOCK_ORDER:
        value = np.asarray(payload[name], dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"base position {name} must be finite [3]")
        result[name] = value.copy()
    return result


def _scene_with_positions(
    scene_obs: np.ndarray,
    positions: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
    value = np.asarray(scene_obs, dtype=np.float64).copy()
    if value.ndim != 1 or value.shape[0] < 24 or not np.isfinite(value).all():
        raise ValueError("CALVIN scene observation must be finite with at least 24 values")
    if positions is not None:
        for object_name, position in positions.items():
            value[SCENE_POSITION_SLICES[object_name]] = np.asarray(
                position, dtype=np.float64
            )
    return value


def _shift_target_y(
    scene_obs: np.ndarray,
    *,
    object_name: str,
    delta_y_m: float,
) -> np.ndarray:
    if object_name not in SCENE_POSITION_SLICES:
        raise KeyError(f"unsupported CALVIN object {object_name!r}")
    if not np.isfinite(delta_y_m):
        raise ValueError("target y shift must be finite")
    value = np.asarray(scene_obs, dtype=np.float64).copy()
    index = SCENE_POSITION_SLICES[object_name].start + 1
    value[index] += float(delta_y_m)
    if not np.isfinite(value).all():
        raise ValueError("shifted CALVIN scene is non-finite")
    return value


def _snapshot_arrays(snapshot: HistorySnapshot) -> dict[str, np.ndarray]:
    snapshot.validate()
    return {
        "schema_version": np.asarray([1], dtype=np.int64),
        "time_index": np.asarray([snapshot.time_index], dtype=np.int64),
        "rgb_top_history": np.asarray(snapshot.rgb_history["top"], dtype=np.uint8),
        "rgb_wrist_history": np.asarray(
            snapshot.rgb_history["wrist"], dtype=np.uint8
        ),
        "state": np.asarray(snapshot.state, dtype=np.float32),
        "action_state": np.asarray(snapshot.action_state, dtype=np.float32),
        "state_history": np.asarray(snapshot.state_history, dtype=np.float32),
        "executed_action_history": np.asarray(
            snapshot.executed_action_history, dtype=np.float32
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _candidate(
    history: CausalHistory,
    *,
    step: int,
    tcp_xyz: np.ndarray,
    target_xyz: np.ndarray,
    kind: str,
) -> SnapshotCandidate:
    return SnapshotCandidate(
        snapshot=history.snapshot(),
        step=int(step),
        tcp_xyz=np.asarray(tcp_xyz, dtype=np.float64).copy(),
        target_xyz=np.asarray(target_xyz, dtype=np.float64).copy(),
        relative_xyz=np.asarray(tcp_xyz, dtype=np.float64)
        - np.asarray(target_xyz, dtype=np.float64),
        kind=str(kind),
    )


def _candidate_report(candidate: SnapshotCandidate) -> dict[str, Any]:
    return {
        "kind": candidate.kind,
        "step": candidate.step,
        "time_index": int(candidate.snapshot.time_index),
        "tcp_xyz_m": candidate.tcp_xyz.tolist(),
        "target_xyz_m": candidate.target_xyz.tolist(),
        "target_relative_xyz_m": candidate.relative_xyz.tolist(),
    }


def _write_snapshot(
    directory: Path,
    *,
    name: str,
    candidate: SnapshotCandidate,
) -> dict[str, Any]:
    path = directory / f"{name}.npz"
    atomic_npz(path, **_snapshot_arrays(candidate.snapshot))
    return {
        "schema": SNAPSHOT_SCHEMA,
        "path": str(path),
        "sha256": _sha256(path),
        **_candidate_report(candidate),
    }


def _variant_name(delta_y_m: float) -> str:
    sign = "plus" if delta_y_m >= 0.0 else "minus"
    return f"target_y_{sign}_{abs(int(round(delta_y_m * 1000.0))):03d}mm"


def _pair_classification(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    a = left["selected_low_snapshot"]
    b = right["selected_low_snapshot"]
    target_delta = float(right["actual_initial_target_xyz_m"][1]) - float(
        left["actual_initial_target_xyz_m"][1]
    )
    if abs(target_delta) <= 1e-9:
        raise ValueError("paired target y positions are identical")
    world_delta = float(b["tcp_xyz_m"][1]) - float(a["tcp_xyz_m"][1])
    relative_delta = float(b["target_relative_xyz_m"][1]) - float(
        a["target_relative_xyz_m"][1]
    )
    gain = world_delta / target_delta
    selection_a = str(a.get("selection", a.get("kind", "")))
    selection_b = str(b.get("selection", b.get("kind", "")))
    phase_comparable = selection_a == selection_b
    if not phase_comparable:
        classification = "phase_mismatch_inconclusive"
    elif selection_a == "low_closest_x" and max(
        abs(float(a["target_relative_xyz_m"][0])),
        abs(float(b["target_relative_xyz_m"][0])),
    ) > 0.020:
        classification = "low_fallback_not_close_enough_inconclusive"
    elif abs(gain) <= 0.25:
        classification = "world_fixed_lane"
    elif abs(gain - 1.0) <= 0.25:
        classification = "target_relative_lane"
    else:
        classification = "partial_or_nonlinear_response"
    return {
        "variant_a": str(left["name"]),
        "variant_b": str(right["name"]),
        "target_y_difference_m": target_delta,
        "selected_low_world_y_difference_m": world_delta,
        "selected_low_relative_y_difference_m": relative_delta,
        "selected_low_world_y_gain_to_target_shift": gain,
        "selected_low_selection_a": selection_a,
        "selected_low_selection_b": selection_b,
        "selected_low_phase_comparable": phase_comparable,
        "classification": classification,
        "threshold_note": "diagnostic labels use |gain|<=0.25 or |gain-1|<=0.25",
    }


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"RMSE shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.square(a - b))))


def run_probe(
    *,
    dataset_root: Path,
    output_dir: Path,
    endpoint: str,
    timeout: float,
    task: str,
    initial_state: Mapping[str, object],
    base_positions: Mapping[str, np.ndarray] | None,
    target_y_shifts_m: Sequence[float],
    max_steps: int,
    video_fps: float,
    expected_policy_seed: int,
) -> dict[str, Any]:
    if task not in TASK_BY_NAME:
        raise KeyError(f"unsupported push task {task!r}")
    if len(target_y_shifts_m) != 2:
        raise ValueError("target-y probe requires exactly two shifts")
    if len({float(value) for value in target_y_shifts_m}) != 2:
        raise ValueError("target-y probe shifts must differ")
    if max_steps <= 0 or max_steps > 360:
        raise ValueError("max_steps must be in [1,360]")
    if not np.isfinite(video_fps) or video_fps <= 0.0:
        raise ValueError("video_fps must be finite and positive")
    _require_new_output_dir(output_dir)

    spec = TASK_BY_NAME[task]
    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health_before = client.health()
    validate_bridge_health(health_before, allow_smoke_policy=False)
    checkpoint = _checkpoint_identity(health_before)
    if int(checkpoint["policy_seed"]) != int(expected_policy_seed):
        raise ValueError(
            "policy seed mismatch: "
            f"expected {expected_policy_seed}, got {checkpoint['policy_seed']}"
        )

    official, conf_dir, oracle, annotations = _official_task_assets()
    instruction = str(annotations[task][0])
    base_robot_obs, official_scene_obs = official.get_env_state_for_initial_condition(
        dict(initial_state)
    )
    base_scene_obs = _scene_with_positions(official_scene_obs, base_positions)
    base_target = base_scene_obs[SCENE_POSITION_SLICES[spec.object_name]].copy()
    env = _environment(dataset_root, show_gui=False)
    reports: list[dict[str, Any]] = []
    action_arrays_by_variant: dict[str, dict[str, np.ndarray]] = {}
    initial_positions_by_variant: dict[str, np.ndarray] = {}
    initial_robot_by_variant: dict[str, np.ndarray] = {}
    started_probe = time.perf_counter()
    try:
        for variant_index, delta_y_m in enumerate(target_y_shifts_m, start=1):
            if variant_index > 1:
                # CALVIN reset restores the physical robot pose but retains the
                # previous episode's non-policy-visible gripper-command latch in
                # robot_obs[-1].  A fresh simulator makes the paired intervention
                # independent of every hidden episode-local state, not merely the
                # seven robot values exposed to ClearVLA.
                env = _environment(dataset_root, show_gui=False)
            name = _variant_name(float(delta_y_m))
            variant_dir = output_dir / name
            variant_dir.mkdir(parents=True, exist_ok=False)
            snapshot_dir = variant_dir / "causal_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            scene_obs = _shift_target_y(
                base_scene_obs,
                object_name=spec.object_name,
                delta_y_m=float(delta_y_m),
            )
            expected_target = base_target.copy()
            expected_target[1] += float(delta_y_m)
            env.reset(
                robot_obs=np.asarray(base_robot_obs, dtype=np.float64).copy(),
                scene_obs=scene_obs.copy(),
            )
            observation = env.get_obs()
            start_info = env.get_info()
            start_positions, start_contacts = _scene_state(start_info)
            initial_robot = np.asarray(
                observation["robot_obs"], dtype=np.float64
            ).copy()
            target_index = BLOCK_ORDER.index(spec.object_name)
            actual_target = start_positions[target_index]
            if not np.allclose(actual_target, expected_target, atol=2e-3, rtol=0.0):
                raise ValueError(
                    f"{name} target reset differs from requested pose: "
                    f"expected {expected_target.tolist()}, got {actual_target.tolist()}"
                )
            start_surface = _surface_contacts(start_info, spec.object_name)
            if not start_surface:
                raise ValueError(f"{name} target starts without surface contact")
            if initial_positions_by_variant:
                reference_positions = next(iter(initial_positions_by_variant.values()))
                reference_robot = next(iter(initial_robot_by_variant.values()))
                non_target = np.ones(len(BLOCK_ORDER), dtype=np.bool_)
                non_target[target_index] = False
                if not np.allclose(
                    start_positions[non_target],
                    reference_positions[non_target],
                    atol=2e-4,
                    rtol=0.0,
                ):
                    raise ValueError("non-target object poses changed between variants")
                if not np.allclose(
                    actual_target[[0, 2]],
                    reference_positions[target_index, [0, 2]],
                    atol=2e-4,
                    rtol=0.0,
                ):
                    raise ValueError("target x/z changed between y-shift variants")
                if not np.allclose(
                    initial_robot, reference_robot, atol=2e-6, rtol=0.0
                ):
                    raise ValueError("robot reset state changed between variants")
            initial_positions_by_variant[name] = start_positions.copy()
            initial_robot_by_variant[name] = initial_robot

            initial_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
            _save_rgb(variant_dir / "initial_static.png", initial_rgb["rgb_static"])
            _save_rgb(variant_dir / "initial_wrist.png", initial_rgb["rgb_gripper"])
            recorder = _CalvinSequenceVideoRecorder(
                variant_dir / "video",
                sequence_index=variant_index,
                initial_state={
                    **dict(initial_state),
                    "diagnostic_target_y_shift_m": float(delta_y_m),
                },
                eval_sequence=[task],
                max_subtask_steps=max_steps,
                fps=video_fps,
            )
            recorder.start_task(task, instruction)
            recorder.capture_task_start(observation)

            model = CalvinBridgeModel(client, execute_rows=1)
            model.reset()
            mirror = CausalHistory()
            previous_action = np.zeros(7, dtype=np.float32)
            robot_trajectory = [
                np.asarray(observation["robot_obs"], dtype=np.float32).copy()
            ]
            action_robot_obs: list[np.ndarray] = []
            object_positions = [start_positions]
            object_contacts = [start_contacts]
            surface_preserved = [True]
            oracle_success = [False]
            candidates: dict[str, SnapshotCandidate] = {}
            previous_relative: np.ndarray | None = None
            entered_low_band = False
            current_info = start_info
            success = False
            steps = 0
            started_variant = time.perf_counter()
            try:
                while steps < max_steps and not success:
                    policy_input = calvin_policy_observation(
                        observation, previous_action
                    )
                    if steps == 0:
                        mirror.reset(
                            policy_input, reset_action=policy_input.action_state
                        )
                    else:
                        mirror.append(policy_input.action_state, policy_input)
                    current_positions, _ = _scene_state(current_info)
                    target_xyz = current_positions[target_index]
                    tcp_xyz = np.asarray(observation["robot_obs"][:3], dtype=np.float64)
                    relative = tcp_xyz - target_xyz
                    if steps == 0:
                        candidates["initial"] = _candidate(
                            mirror,
                            step=steps,
                            tcp_xyz=tcp_xyz,
                            target_xyz=target_xyz,
                            kind="initial",
                        )
                    if not entered_low_band and relative[2] >= 0.040:
                        high = candidates.get("high_alignment")
                        if high is None or np.linalg.norm(relative[:2]) < np.linalg.norm(
                            high.relative_xyz[:2]
                        ):
                            candidates["high_alignment"] = _candidate(
                                mirror,
                                step=steps,
                                tcp_xyz=tcp_xyz,
                                target_xyz=target_xyz,
                                kind="high_alignment",
                            )
                    if relative[2] <= 0.040:
                        entered_low_band = True
                        low = candidates.get("low_closest_x")
                        if low is None or abs(relative[0]) < abs(low.relative_xyz[0]):
                            candidates["low_closest_x"] = _candidate(
                                mirror,
                                step=steps,
                                tcp_xyz=tcp_xyz,
                                target_xyz=target_xyz,
                                kind="low_closest_x",
                            )
                        if (
                            "low_crossing" not in candidates
                            and previous_relative is not None
                            and previous_relative[0] * spec.direction_sign < 0.0
                            and relative[0] * spec.direction_sign >= 0.0
                        ):
                            candidates["low_crossing"] = _candidate(
                                mirror,
                                step=steps,
                                tcp_xyz=tcp_xyz,
                                target_xyz=target_xyz,
                                kind="low_crossing",
                            )
                    previous_relative = relative.copy()

                    action_robot_obs.append(
                        np.asarray(observation["robot_obs"], dtype=np.float32).copy()
                    )
                    action = model.step(observation, instruction)
                    observation, _reward, _done, current_info = env.step(action)
                    previous_action = np.asarray(action, dtype=np.float32).copy()
                    steps += 1
                    robot_trajectory.append(
                        np.asarray(observation["robot_obs"], dtype=np.float32).copy()
                    )
                    positions, contacts = _scene_state(current_info)
                    object_positions.append(positions)
                    object_contacts.append(contacts)
                    surface_preserved.append(
                        bool(
                            start_surface
                            <= _surface_contacts(current_info, spec.object_name)
                        )
                    )
                    success = bool(
                        oracle.get_task_info_for_set(
                            start_info, current_info, {task}
                        )
                    )
                    oracle_success.append(success)
                    recorder.capture_step(observation)
                health_after_rollout = client.health()
                actual_history = _require_rollout_bridge_integrity(
                    health_after_rollout,
                    checkpoint=checkpoint,
                    executed_steps=steps,
                )
                if "high_alignment" not in candidates:
                    raise RuntimeError(
                        f"{name} never produced a high-alignment snapshot"
                    )
                selected_low_key = (
                    "low_crossing"
                    if "low_crossing" in candidates
                    else "low_closest_x"
                )
                if selected_low_key not in candidates:
                    raise RuntimeError(f"{name} never entered the low sweep band")
                recorder.finish_task(success)
                video_report = recorder.finish_sequence(1 if success else 0)
            except BaseException as error:
                recorder.abort_sequence(error)
                raise

            snapshots = {
                "initial": _write_snapshot(
                    snapshot_dir, name="initial", candidate=candidates["initial"]
                ),
                "high_alignment": _write_snapshot(
                    snapshot_dir,
                    name="high_alignment",
                    candidate=candidates["high_alignment"],
                ),
                "selected_low": _write_snapshot(
                    snapshot_dir,
                    name="selected_low",
                    candidate=candidates[selected_low_key],
                ),
            }
            snapshots["selected_low"]["selection"] = selected_low_key
            final_rgb = cast(Mapping[str, np.ndarray], observation["rgb_obs"])
            _save_rgb(variant_dir / "final_static.png", final_rgb["rgb_static"])
            _save_rgb(variant_dir / "final_wrist.png", final_rgb["rgb_gripper"])

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
            atomic_npz(variant_dir / "trajectory.npz", **arrays)
            action_arrays_by_variant[name] = {
                key: np.asarray(arrays[key]).copy()
                for key in ("raw_chunks", "raw_first", "executed")
            }
            diagnostics = _rollout_diagnostics(
                task,
                position_array,
                contact_array,
                surface_array,
                success=success,
            )
            report = {
                "schema": SCHEMA,
                "name": name,
                "task": task,
                "instruction": instruction,
                "diagnostic_not_benchmark_score": True,
                "target_y_shift_m": float(delta_y_m),
                "base_target_xyz_m": base_target.tolist(),
                "requested_target_xyz_m": expected_target.tolist(),
                "actual_initial_target_xyz_m": actual_target.tolist(),
                "initial_state": _jsonable(initial_state),
                "success": bool(success),
                "steps": int(steps),
                "max_steps": int(max_steps),
                "elapsed_seconds": float(time.perf_counter() - started_variant),
                "checkpoint": checkpoint,
                "execution": {
                    "execute_rows": 1,
                    "history_time_index_expected_after": steps - 1,
                    "history_time_index_actual_after": actual_history,
                    "history_time_index_aligned": actual_history == steps - 1,
                },
                "trajectory_diagnostics": diagnostics,
                "snapshots": snapshots,
                "selected_low_snapshot": snapshots["selected_low"],
                "video": {
                    "path": str(variant_dir / "video" / str(video_report["video"])),
                    "frame_count": int(video_report["frame_count"]),
                },
                "artifacts": {
                    "trajectory": str(variant_dir / "trajectory.npz"),
                    "initial_static": str(variant_dir / "initial_static.png"),
                    "final_static": str(variant_dir / "final_static.png"),
                },
            }
            atomic_json(variant_dir / "result.json", report)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "variant": name,
                        "steps": steps,
                        "success": success,
                        "selected_low": report["selected_low_snapshot"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    if len(reports) != 2:
        raise RuntimeError("paired target-y probe did not finish both variants")
    final_health = client.health()
    validate_bridge_health(final_health, allow_smoke_policy=False)
    if _checkpoint_identity(final_health) != checkpoint:
        raise RuntimeError("bridge checkpoint identity changed during target-y probe")
    first, second = reports
    first_arrays = action_arrays_by_variant[str(first["name"])]
    second_arrays = action_arrays_by_variant[str(second["name"])]
    overlap = min(
        int(first_arrays["raw_chunks"].shape[0]),
        int(second_arrays["raw_chunks"].shape[0]),
    )
    pair = _pair_classification(first, second)
    pair["first_plan_arm_rmse"] = _rmse(
        first_arrays["raw_chunks"][0, :, :6],
        second_arrays["raw_chunks"][0, :, :6],
    )
    pair["overlap_raw_chunk_arm_rmse"] = _rmse(
        first_arrays["raw_chunks"][:overlap, :, :6],
        second_arrays["raw_chunks"][:overlap, :, :6],
    )
    pair["overlap_steps"] = overlap
    summary = {
        "schema": SCHEMA,
        "diagnostic_not_benchmark_score": True,
        "dataset_root": str(dataset_root.resolve()),
        "endpoint": endpoint,
        "task": task,
        "instruction": instruction,
        "checkpoint": checkpoint,
        "checkpoint_identity_stable": True,
        "protocol": {
            "only_between_variant_change": "target object y coordinate",
            "target_y_shifts_m": [float(value) for value in target_y_shifts_m],
            "execute_rows": 1,
            "policy_rng_reset_each_variant": True,
            "fresh_simulator_instance_each_variant": True,
            "max_steps": int(max_steps),
            "record_video": True,
            "snapshot_schema": SNAPSHOT_SCHEMA,
        },
        "base_initial_state": _jsonable(initial_state),
        "base_scene_positions_xyz_m": {
            name: base_scene_obs[index].tolist()
            for name, index in SCENE_POSITION_SLICES.items()
        },
        "official_assets": {"config_directory": str(conf_dir)},
        "variants": reports,
        "pair": pair,
        "elapsed_seconds": float(time.perf_counter() - started_probe),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18772")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--task", default="push_red_block_left")
    parser.add_argument("--initial-state-json", type=Path, required=True)
    parser.add_argument("--base-positions-json", type=Path, default=None)
    parser.add_argument(
        "--target-y-shifts-m", type=float, nargs=2, default=(-0.04, 0.04)
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--expected-policy-seed", type=int, default=0)
    args = parser.parse_args()
    result = run_probe(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        endpoint=str(args.endpoint),
        timeout=float(args.timeout),
        task=str(args.task),
        initial_state=_load_mapping(args.initial_state_json, name="initial state"),
        base_positions=_base_positions(args.base_positions_json),
        target_y_shifts_m=tuple(float(value) for value in args.target_y_shifts_m),
        max_steps=int(args.max_steps),
        video_fps=float(args.video_fps),
        expected_policy_seed=int(args.expected_policy_seed),
    )
    print(json.dumps(result["pair"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
