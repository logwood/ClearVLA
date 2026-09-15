#!/usr/bin/env python3
"""Replay a saved targeted CALVIN action trace and write a viewable MP4.

The replay is deliberately policy-free: it consumes the already executed
actions from ``actions.npz`` and resets the official CALVIN environment to the
same initial condition.  This makes the resulting movie a visualization of a
completed rollout rather than a second inference run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from clearvla.benchmarks.calvin_eval import (
    _environment,
    _official_task_assets,
)

DEFAULT_INITIAL_STATE: dict[str, object] = {
    "slider": "right",
    "drawer": "closed",
    "lightbulb": 0,
    "led": 0,
    "red_block": "table",
    "blue_block": "table",
    "pink_block": "slider_right",
    "grasped": 0,
}


def _read_initial_state(path: Path | None) -> Mapping[str, object]:
    if path is None:
        return DEFAULT_INITIAL_STATE
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("initial-state JSON must be an object")
    return {str(key): item for key, item in value.items()}


def _drawer_state(info: Mapping[str, Any]) -> float:
    scene = info.get("scene_info")
    if not isinstance(scene, Mapping):
        raise ValueError("CALVIN info has no scene_info mapping")
    doors = scene.get("doors")
    if not isinstance(doors, Mapping) or "base__drawer" not in doors:
        raise ValueError("CALVIN scene has no base__drawer state")
    drawer = doors["base__drawer"]
    if not isinstance(drawer, Mapping):
        raise ValueError("CALVIN drawer info must be a mapping")
    value = np.asarray(drawer["current_state"], dtype=np.float64).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError("CALVIN drawer state must be one finite scalar")
    return float(value[0])


def _drawer_contact(info: Mapping[str, Any]) -> tuple[bool, float]:
    scene = info.get("scene_info")
    robot = info.get("robot_info")
    if not isinstance(scene, Mapping) or not isinstance(robot, Mapping):
        raise ValueError("CALVIN info lacks scene or robot mappings")
    fixed = scene.get("fixed_objects")
    if not isinstance(fixed, Mapping) or "table" not in fixed:
        raise ValueError("CALVIN scene has no fixed table object")
    table = fixed["table"]
    if not isinstance(table, Mapping):
        raise ValueError("CALVIN table info must be a mapping")
    links = table.get("links")
    if not isinstance(links, Mapping) or "drawer_link" not in links:
        raise ValueError("CALVIN table has no drawer_link")
    table_uid = int(table["uid"])
    drawer_link = int(links["drawer_link"])
    contacts = robot.get("contacts", ())
    matched: list[Any] = []
    for contact in contacts:
        if len(contact) < 5:
            continue
        table_is_b = int(contact[2]) == table_uid and int(contact[4]) == drawer_link
        table_is_a = int(contact[1]) == table_uid and int(contact[3]) == drawer_link
        if table_is_a or table_is_b:
            matched.append(contact)
    force = max(
        (float(contact[9]) for contact in matched if len(contact) > 9),
        default=0.0,
    )
    return bool(matched), force


def _true_runs(values: np.ndarray) -> list[dict[str, int]]:
    flags = np.asarray(values, dtype=np.bool_).reshape(-1)
    padded = np.r_[False, flags, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [
        {
            "first_step": int(start),
            "last_step": int(end - 1),
            "steps": int(end - start),
        }
        for start, end in zip(starts, ends)
    ]


def _panel(rgb: np.ndarray, *, width: int, height: int, title: str) -> np.ndarray:
    image = np.asarray(rgb)
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
    step: int,
    total: int,
    success: bool,
    action: np.ndarray | None,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    rgb = observation["rgb_obs"]
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation has no rgb_obs mapping")
    static = _panel(
        np.asarray(rgb["rgb_static"]),
        width=panel_width,
        height=panel_height,
        title="static camera",
    )
    wrist = _panel(
        np.asarray(rgb["rgb_gripper"]),
        width=panel_width,
        height=panel_height,
        title="wrist camera",
    )
    canvas = np.concatenate([static, wrist], axis=1)
    status = "SUCCESS" if success else "replay"
    if action is None:
        action_text = "initial state"
    else:
        action_text = (
            "a=["
            + ", ".join(f"{float(value):+.2f}" for value in action[:6])
            + f"], grip={float(action[6]):+.0f}"
        )
    cv2.rectangle(
        canvas,
        (0, panel_height),
        (canvas.shape[1] - 1, panel_height + 46),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.putText(
        canvas,
        f"step {step:03d}/{total:03d}   {status}",
        (10, panel_height + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 235, 120) if success else (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        action_text,
        (10, panel_height + 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def replay(
    *,
    dataset_root: Path,
    actions_path: Path,
    output_path: Path,
    initial_state: Mapping[str, object],
    task: str,
    fps: float,
    show_gui: bool,
) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    with np.load(actions_path, allow_pickle=False) as payload:
        if "executed" not in payload:
            raise KeyError(f"{actions_path} has no executed action array")
        actions = np.asarray(payload["executed"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"executed actions must be finite [N,7], got {actions.shape}")

    official, _conf_dir, oracle, _annotations = _official_task_assets()
    env = _environment(dataset_root, show_gui=show_gui)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    success = False
    step_reached = 0
    tcp_positions: list[np.ndarray] = []
    drawer_states: list[float] = []
    drawer_contacts: list[bool] = []
    drawer_contact_forces: list[float] = []
    try:
        robot_obs, scene_obs = official.get_env_state_for_initial_condition(
            dict(initial_state)
        )
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = env.get_obs()
        start_info = env.get_info()
        tcp_positions.append(
            np.asarray(observation["robot_obs"], dtype=np.float64)[:3].copy()
        )
        drawer_states.append(_drawer_state(start_info))
        initial_contact, initial_force = _drawer_contact(start_info)
        drawer_contacts.append(initial_contact)
        drawer_contact_forces.append(initial_force)
        initial_frame = _frame(
            observation,
            step=0,
            total=int(actions.shape[0]),
            success=False,
            action=None,
            panel_width=320,
            panel_height=320,
        )
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(initial_frame.shape[1]), int(initial_frame.shape[0])),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {output_path}")
        writer.write(initial_frame)
        if show_gui:
            cv2.imshow("CALVIN replay", initial_frame)
            cv2.waitKey(max(1, int(1000 / fps)))

        for index, action in enumerate(actions, start=1):
            observation, _reward, _done, current_info = env.step(action)
            current_task = oracle.get_task_info_for_set(
                start_info,
                current_info,
                {task},
            )
            success = len(current_task) > 0
            step_reached = index
            tcp_positions.append(
                np.asarray(observation["robot_obs"], dtype=np.float64)[:3].copy()
            )
            drawer_states.append(_drawer_state(current_info))
            contact, contact_force = _drawer_contact(current_info)
            drawer_contacts.append(contact)
            drawer_contact_forces.append(contact_force)
            rendered = _frame(
                observation,
                step=index,
                total=int(actions.shape[0]),
                success=success,
                action=action,
                panel_width=320,
                panel_height=320,
            )
            writer.write(rendered)
            if show_gui:
                cv2.imshow("CALVIN replay", rendered)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == 27:
                    break
            if success:
                # Keep a short readable hold on the successful final state.
                for _ in range(max(1, int(round(fps * 1.5)))):
                    writer.write(rendered)
                break
    finally:
        if writer is not None:
            writer.release()
        if show_gui:
            cv2.destroyAllWindows()
        env.close()

    tcp = np.stack(tcp_positions).astype(np.float64)
    drawer = np.asarray(drawer_states, dtype=np.float64)
    displacement = drawer - drawer[0]
    contact_flags = np.asarray(drawer_contacts, dtype=np.bool_)
    contact_forces = np.asarray(drawer_contact_forces, dtype=np.float64)
    contact_steps = np.flatnonzero(contact_flags[1:]) + 1
    motion_steps = np.flatnonzero(np.abs(displacement[1:]) > 1e-6) + 1
    drawer_threshold: float | None
    if task == "open_drawer":
        drawer_threshold = 0.12
        success_steps = np.flatnonzero(displacement[1:] > drawer_threshold) + 1
    elif task == "close_drawer":
        drawer_threshold = -0.12
        success_steps = np.flatnonzero(displacement[1:] < drawer_threshold) + 1
    else:
        drawer_threshold = None
        success_steps = np.empty((0,), dtype=np.int64)
    first_contact_step = int(contact_steps[0]) if contact_steps.size else None
    first_motion_step = int(motion_steps[0]) if motion_steps.size else None
    threshold_step = int(success_steps[0]) if success_steps.size else None
    tcp_segments = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    tcp_net = float(np.linalg.norm(tcp[-1] - tcp[0]))
    diagnostics: dict[str, object] = {
        "drawer_initial_state": float(drawer[0]),
        "drawer_final_state": float(drawer[-1]),
        "drawer_final_displacement": float(displacement[-1]),
        "drawer_max_displacement": float(displacement.max()),
        "drawer_min_displacement": float(displacement.min()),
        "drawer_official_threshold": drawer_threshold,
        "drawer_threshold_step": threshold_step,
        "drawer_first_motion_step": first_motion_step,
        "drawer_first_contact_step": first_contact_step,
        "drawer_contact_steps": int(contact_steps.size),
        "drawer_contact_runs": _true_runs(contact_flags),
        "drawer_max_contact_force": float(contact_forces.max()),
        "tcp_path_m": float(tcp_segments.sum()),
        "tcp_net_displacement_m": tcp_net,
        "tcp_path_to_net_ratio": float(tcp_segments.sum() / tcp_net)
        if tcp_net > 0
        else None,
    }
    if first_contact_step is not None:
        diagnostics.update(
            {
                "drawer_displacement_at_first_contact": float(
                    displacement[first_contact_step]
                ),
                "tcp_at_first_contact": tcp[first_contact_step].tolist(),
                "gripper_command_at_first_contact": float(
                    actions[first_contact_step - 1, 6]
                ),
            }
        )

    result = {
        "schema": "clearvla-calvin-action-replay-video-v1",
        "task": task,
        "dataset_root": str(dataset_root),
        "actions": str(actions_path),
        "video": str(output_path),
        "fps": float(fps),
        "action_count": int(actions.shape[0]),
        "steps_replayed": int(step_reached),
        "success": bool(success),
        "initial_state": dict(initial_state),
        "trajectory_diagnostics": diagnostics,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="open_drawer")
    parser.add_argument("--initial-state-json", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--show-gui", action="store_true")
    args = parser.parse_args()
    result = replay(
        dataset_root=args.dataset_root,
        actions_path=args.actions,
        output_path=args.output,
        initial_state=_read_initial_state(args.initial_state_json),
        task=str(args.task),
        fps=float(args.fps),
        show_gui=bool(args.show_gui),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
