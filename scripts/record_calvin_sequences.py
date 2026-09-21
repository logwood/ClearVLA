#!/usr/bin/env python3
"""Re-run deterministic CALVIN sequences and save readable MP4 rollouts.

This utility is intentionally separate from the benchmark scorer.  It uses the
same official sequence generator and the same ClearVLA bridge, but records one
MP4 and one action archive per sequence so a partial/aborted benchmark run can
still be inspected visually.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from clearvla.benchmarks.calvin_eval import (
    CalvinBridgeModel,
    _environment,
    _official_task_assets,
)
from clearvla.benchmarks.bridge import RemotePolicyClient
from calvin_agent.evaluation.multistep_sequences import get_sequences


def _panel(value: Any, *, width: int, height: int, title: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {image.shape}")
    bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
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


def _frame(
    observation: Mapping[str, Any],
    *,
    sequence_index: int,
    task_index: int,
    task: str,
    step: int,
    max_steps: int,
    success: bool,
    terminal: bool,
) -> np.ndarray:
    rgb = observation["rgb_obs"]
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation has no rgb_obs mapping")
    static = _panel(rgb["rgb_static"], width=320, height=320, title="static camera")
    wrist = _panel(rgb["rgb_gripper"], width=320, height=320, title="wrist camera")
    panels = np.concatenate([static, wrist], axis=1)
    canvas = np.full((374, panels.shape[1], 3), 20, dtype=np.uint8)
    canvas[:320] = panels
    status = "SUCCESS" if success else ("FAILED" if terminal else "running")
    color = (80, 235, 120) if success else ((60, 80, 235) if terminal else (235, 235, 235))
    cv2.putText(
        canvas,
        f"sequence {sequence_index:02d}  task {task_index}/5  step {step:03d}/{max_steps:03d}  {status}",
        (10, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        task,
        (10, 361),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_sequences(
    *,
    dataset_root: Path,
    output_dir: Path,
    endpoint: str,
    num_sequences: int,
    source_sequence_count: int,
    max_subtask_steps: int,
    sequence_workers: int,
    execute_rows: int,
    fps: float,
) -> dict[str, Any]:
    if num_sequences <= 0 or source_sequence_count < num_sequences:
        raise ValueError("source_sequence_count must be >= num_sequences > 0")
    if max_subtask_steps <= 0 or max_subtask_steps > 360:
        raise ValueError("max_subtask_steps must be in [1,360]")
    if execute_rows <= 0 or execute_rows > 24:
        raise ValueError("execute_rows must be in [1,24]")
    if fps <= 0:
        raise ValueError("fps must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    official, conf_dir, oracle, annotations = _official_task_assets()
    # Taking the prefix of get_sequences(1000) reproduces the first seven
    # sequences from the earlier benchmark invocation.  Calling get_sequences(7)
    # would generate a different per-state allocation and therefore a different
    # prefix.
    sequences = get_sequences(source_sequence_count, num_workers=sequence_workers)[:num_sequences]
    manifest = {
        "schema": "clearvla-calvin-video-sequence-manifest-v1",
        "source_sequence_count": int(source_sequence_count),
        "selected_count": int(num_sequences),
        "execute_rows": int(execute_rows),
        "max_subtask_steps": int(max_subtask_steps),
        "sequences": [
            {
                "index": int(index),
                "initial_state": dict(initial_state),
                "tasks": [str(task) for task in eval_sequence],
            }
            for index, (initial_state, eval_sequence) in enumerate(sequences, start=1)
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)

    env = _environment(dataset_root, show_gui=False)
    client = RemotePolicyClient(endpoint=endpoint, timeout=120.0)
    sequence_reports: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    try:
        for sequence_index, (initial_state, eval_sequence) in enumerate(sequences, start=1):
            client_model = CalvinBridgeModel(client, execute_rows=execute_rows)
            robot_obs, scene_obs = official.get_env_state_for_initial_condition(dict(initial_state))
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            observation = env.get_obs()
            writer: cv2.VideoWriter | None = None
            video_path = output_dir / f"sequence_{sequence_index:02d}.mp4"
            action_path = output_dir / f"sequence_{sequence_index:02d}_actions.npz"
            task_reports: list[dict[str, Any]] = []
            sequence_success = 0
            sequence_started = time.perf_counter()
            try:
                first_frame = _frame(
                    observation,
                    sequence_index=sequence_index,
                    task_index=0,
                    task="initial state",
                    step=0,
                    max_steps=max_subtask_steps,
                    success=False,
                    terminal=False,
                )
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (int(first_frame.shape[1]), int(first_frame.shape[0])),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open video writer: {video_path}")
                writer.write(first_frame)

                for task_index, task in enumerate(eval_sequence, start=1):
                    task = str(task)
                    instruction = str(annotations[task][0])
                    client_model.reset()
                    start_info = env.get_info()
                    task_started = time.perf_counter()
                    success = False
                    steps = 0
                    plans_before = len(client_model.raw_chunks)
                    observes_before = len(client_model.observed_history_time_indices)
                    for step in range(1, max_subtask_steps + 1):
                        action = client_model.step(observation, instruction)
                        observation, _reward, _done, current_info = env.step(action)
                        steps = step
                        success = bool(
                            len(oracle.get_task_info_for_set(start_info, current_info, {task})) > 0
                        )
                        writer.write(
                            _frame(
                                observation,
                                sequence_index=sequence_index,
                                task_index=task_index,
                                task=instruction,
                                step=step,
                                max_steps=max_subtask_steps,
                                success=success,
                                terminal=not success and step == max_subtask_steps,
                            )
                        )
                        if success:
                            sequence_success += 1
                            for _ in range(max(1, int(round(fps * 0.5)))):
                                writer.write(
                                    _frame(
                                        observation,
                                        sequence_index=sequence_index,
                                        task_index=task_index,
                                        task=instruction,
                                        step=step,
                                        max_steps=max_subtask_steps,
                                        success=True,
                                        terminal=True,
                                    )
                                )
                            break
                    task_reports.append(
                        {
                            "task": task,
                            "instruction": instruction,
                            "success": bool(success),
                            "steps": int(steps),
                            "planning_decisions": int(len(client_model.raw_chunks) - plans_before),
                            "intermediate_observations": int(
                                len(client_model.observed_history_time_indices) - observes_before
                            ),
                            "elapsed_seconds": float(time.perf_counter() - task_started),
                        }
                    )
                    if not success:
                        break
            finally:
                if writer is not None:
                    writer.release()
            arrays = client_model.action_arrays()
            np.savez_compressed(action_path, **arrays)
            report = {
                "schema": "clearvla-calvin-video-sequence-report-v1",
                "sequence_index": int(sequence_index),
                "initial_state": dict(initial_state),
                "tasks": [str(task) for task in eval_sequence],
                "task_reports": task_reports,
                "successful_tasks": int(sequence_success),
                "sequence_success": bool(sequence_success == len(eval_sequence)),
                "video": video_path.name,
                "actions": action_path.name,
                "elapsed_seconds": float(time.perf_counter() - sequence_started),
            }
            _write_json(output_dir / f"sequence_{sequence_index:02d}.json", report)
            sequence_reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        env.close()

    result = {
        "schema": "clearvla-calvin-video-run-v1",
        "output_dir": str(output_dir),
        "source_sequence_count": int(source_sequence_count),
        "sequence_count": int(num_sequences),
        "execute_rows": int(execute_rows),
        "max_subtask_steps": int(max_subtask_steps),
        "sequence_reports": sequence_reports,
        "elapsed_seconds": float(time.perf_counter() - started_all),
    }
    _write_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18772")
    parser.add_argument("--num-sequences", type=int, default=7)
    parser.add_argument("--source-sequence-count", type=int, default=1000)
    parser.add_argument("--max-subtask-steps", type=int, default=180)
    parser.add_argument("--sequence-workers", type=int, default=4)
    parser.add_argument("--execute-rows", type=int, default=4)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    result = record_sequences(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        endpoint=str(args.endpoint),
        num_sequences=int(args.num_sequences),
        source_sequence_count=int(args.source_sequence_count),
        max_subtask_steps=int(args.max_subtask_steps),
        sequence_workers=int(args.sequence_workers),
        execute_rows=int(args.execute_rows),
        fps=float(args.fps),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
