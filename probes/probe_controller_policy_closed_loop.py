#!/usr/bin/env python3
"""Matched CALVIN policy rollout with an explicit diagnostic TCP command anchor.

Stored-target mode preserves the original panel. Measured-TCP mode uses the
backend's existing measured-pose alternative. Replan-TCP anchors a new chunk
at its measured start, then integrates only that chunk. Results are separate.
Every action is freshly predicted at the original eight-row replan boundaries.
Neither mode supplies simulator object identity or controller state to policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient
from clearvla.benchmarks.calvin_eval import (
    CalvinBridgeModel,
    _environment,
    _official_task_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--execute-rows", type=int, default=8)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--anchor", choices=("stored_target", "measured_tcp", "replan_tcp"), required=True)
    args = parser.parse_args()

    if args.max_steps <= 0 or args.execute_rows <= 0:
        raise ValueError("max_steps and execute_rows must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = json.loads(args.state_json.read_text())
    if not isinstance(state, dict):
        raise ValueError("state JSON must contain an initial-condition mapping")

    official, _conf, oracle, annotations = _official_task_assets()
    if args.task not in annotations:
        raise KeyError(f"unknown CALVIN task: {args.task}")
    instruction = str(annotations[args.task][0])
    robot_obs, scene_obs = official.get_env_state_for_initial_condition(state)
    client = RemotePolicyClient(args.endpoint, timeout=600)
    health_before = client.health()
    (args.output_dir / "health_before.json").write_text(
        json.dumps(health_before, indent=2, sort_keys=True) + "\n"
    )
    env = _environment(args.dataset_root, show_gui=False)
    model = CalvinBridgeModel(client, execute_rows=args.execute_rows)
    started = time.perf_counter()
    reports: dict[str, object] = {
        "task": args.task,
        "instruction": instruction,
        "initial_state": state,
        "max_steps": args.max_steps,
        "execute_rows": args.execute_rows,
        "controller_anchor": args.anchor,
        "diagnostic_environment_contract": True,
    }
    # Read-only telemetry: preserve every admitted observation for exact history replay.
    rgb_static_frames, rgb_gripper_frames, scene_states, info_frames, controller_frames = [], [], [], [], []
    applied_goals = []
    original_relative = env.robot.relative_to_absolute
    def relative_with_telemetry(action):
        absolute = original_relative(action)
        applied_goals.append(np.concatenate([
            np.asarray(absolute[0]).reshape(3),
            np.asarray(absolute[1]).reshape(3),
        ]).copy())
        return absolute
    env.robot.relative_to_absolute = relative_with_telemetry
    def capture(observation, info):
        rgb_static_frames.append(np.asarray(observation["rgb_obs"]["rgb_static"], dtype=np.uint8).copy())
        rgb_gripper_frames.append(np.asarray(observation["rgb_obs"]["rgb_gripper"], dtype=np.uint8).copy())
        scene_states.append(np.asarray(observation["scene_obs"], dtype=np.float32).copy())
        info_frames.append(info)
        controller_frames.append({k: np.asarray(getattr(env.robot, k)).copy().tolist()
                                  for k in ("target_pos", "target_orn") if hasattr(env.robot, k)})
    def json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    robot_states: list[np.ndarray] = []
    success = False
    steps = 0
    try:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        assert env.robot.use_target_pose is True
        env.robot.use_target_pose = args.anchor != "measured_tcp"
        observation = env.get_obs()
        start_info = env.get_info()
        model.reset()
        capture(observation, start_info)
        robot_states.append(np.asarray(observation["robot_obs"], dtype=np.float32).copy())
        for step in range(1, args.max_steps + 1):
            action = model.step(observation, instruction)
            if args.anchor == "replan_tcp" and (step - 1) % args.execute_rows == 0:
                # An explicit diagnostic: each newly observed/replanned path is
                # anchored to that same measured state. The backend integrates
                # the next eight rows; earlier plans cannot leave a hidden backlog.
                env.robot.target_pos = np.asarray(observation["robot_obs"][:3]).copy()
                env.robot.target_orn = np.asarray(observation["robot_obs"][3:6]).copy()
            observation, _reward, _done, info = env.step(action.copy())
            robot_states.append(np.asarray(observation["robot_obs"], dtype=np.float32).copy())
            capture(observation, info)
            steps = step
            success = bool(oracle.get_task_info_for_set(start_info, info, {args.task}))
            if step == 1 or step % args.execute_rows == 0 or success:
                print(json.dumps({"status": "running", "step": step, "success": success}), flush=True)
            if success and not args.continue_after_success:
                break
        reports.update(
            {
                "success": success,
                "steps": steps,
                "elapsed_seconds": time.perf_counter() - started,
                "action_audit": model.action_audit(),
                "robot_obs_trajectory": np.stack(robot_states).tolist(),
                "complete": True,
            }
        )
    finally:
        health_after = client.health()
        (args.output_dir / "health_after.json").write_text(
            json.dumps(health_after, indent=2, sort_keys=True) + "\n"
        )
        env.close()
    np.savez_compressed(args.output_dir / "trajectory.npz",
                        rgb_static=np.stack(rgb_static_frames), rgb_gripper=np.stack(rgb_gripper_frames),
                        scene_obs=np.stack(scene_states), robot_obs=np.stack(robot_states),
                        controller_applied_goal=np.stack(applied_goals),
                        **model.action_arrays())
    (args.output_dir / "environment_info.json").write_text(
        json.dumps({"info": info_frames, "controller_targets": controller_frames}, default=json_value) + "\n")
    (args.output_dir / "result.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: reports[k] for k in ("task", "success", "steps", "elapsed_seconds")}))


if __name__ == "__main__":
    main()

