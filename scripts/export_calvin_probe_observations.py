#!/usr/bin/env python3
"""Export fixed CALVIN observations for a cross-Python diagnostic.

CALVIN's official pybullet stack currently runs in the server's Python 3.9
environment, while the ClearVLA checkpoint runs in the Python 3.12 CUDA
environment.  This small utility deliberately writes only rendered RGB/state
snapshots; it does not load a policy or modify simulator assets.  The paired
3.12 internal-layer probe consumes these snapshots so the two ABIs never share
compiled modules in one process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def layout_states() -> dict[str, dict[str, object]]:
    standard = {
        "slider": "right",
        "drawer": "closed",
        "lightbulb": 0,
        "led": 0,
        "red_block": "table",
        "blue_block": "table",
        "pink_block": "slider_right",
        "grasped": 0,
    }
    blue_left = dict(standard)
    blue_left["blue_block"] = "slider_left"
    blue_right = dict(standard)
    blue_right["blue_block"] = "slider_right"
    red_left = dict(standard)
    red_left["red_block"] = "slider_left"
    return {
        "standard": standard,
        "blue_on_slider_left": blue_left,
        "blue_on_slider_right": blue_right,
        "red_on_slider_left": red_left,
    }


def state_for_initial_condition(initial_condition: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Match the official CALVIN evaluator's deterministic initial layout."""

    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    block_slider_left = np.array(
        [-2.40851662e-01, 9.24044687e-02, 4.60990009e-01], dtype=np.float64
    )
    block_slider_right = np.array(
        [7.03416330e-02, 9.24044687e-02, 4.60990009e-01], dtype=np.float64
    )
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01], dtype=np.float64),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01], dtype=np.float64),
    ]
    seed = int.from_bytes(
        hashlib.sha256(repr(sorted(initial_condition.items())).encode()).digest()[:4],
        "little",
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(block_table)
    scene_obs = np.zeros(24, dtype=np.float64)
    if initial_condition.get("slider") == "left":
        scene_obs[0] = 0.28
    if initial_condition.get("drawer") == "open":
        scene_obs[1] = 0.22
    if initial_condition.get("lightbulb") == 1:
        scene_obs[3] = 0.088
    scene_obs[4] = float(initial_condition.get("lightbulb", 0))
    scene_obs[5] = float(initial_condition.get("led", 0))

    def position(name: str, *, table_index: int) -> np.ndarray:
        value = initial_condition.get(name)
        if value == "slider_right":
            return block_slider_right
        if value == "slider_left":
            return block_slider_left
        return block_table[table_index]

    scene_obs[6:9] = position("red_block", table_index=0)
    scene_obs[11] = rng.uniform(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
    if initial_condition.get("blue_block") == "slider_right":
        scene_obs[12:15] = block_slider_right
    elif initial_condition.get("blue_block") == "slider_left":
        scene_obs[12:15] = block_slider_left
    elif initial_condition.get("red_block") == "table":
        scene_obs[12:15] = block_table[1]
    else:
        scene_obs[12:15] = block_table[0]
    scene_obs[17] = rng.uniform(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
    scene_obs[18:21] = position("pink_block", table_index=1)
    scene_obs[23] = rng.uniform(math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
    return robot_obs, scene_obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=list(layout_states()))
    args = parser.parse_args()

    # CALVIN is imported only after argument parsing so this file remains
    # importable in the 3.12 policy environment for metadata inspection.
    from calvin_env.envs.play_table_env import get_env

    validation = args.dataset_root / "validation"
    env = get_env(
        validation,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        show_gui=False,
    )
    records: dict[str, object] = {}
    try:
        all_layouts = layout_states()
        for name in args.layouts:
            if name not in all_layouts:
                raise KeyError(f"unknown layout: {name}")
            initial = all_layouts[name]
            robot_obs, scene_obs = state_for_initial_condition(initial)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            observation = env.get_obs()
            rgb = observation["rgb_obs"]
            top = np.ascontiguousarray(np.asarray(rgb["rgb_static"], dtype=np.uint8))
            wrist = np.ascontiguousarray(np.asarray(rgb["rgb_gripper"], dtype=np.uint8))
            state = np.ascontiguousarray(np.asarray(observation["robot_obs"], dtype=np.float32)[:7])
            action_state = np.zeros(7, dtype=np.float32)
            path = args.output_dir / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, top=top, wrist=wrist, state=state, action_state=action_state)
            records[name] = {
                "initial_state": initial,
                "path": str(path),
                "top_shape": list(top.shape),
                "wrist_shape": list(wrist.shape),
                "state": state.astype(float).tolist(),
                "top_sha256": hashlib.sha256(top.tobytes()).hexdigest(),
                "wrist_sha256": hashlib.sha256(wrist.tobytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "action_state_sha256": hashlib.sha256(
                    action_state.tobytes()
                ).hexdigest(),
            }
    finally:
        env.close()
    manifest = {
        "schema": "calvin-probe-observations-v2",
        "dataset_root": str(args.dataset_root),
        "layouts": records,
        "notes": "Rendered by official CALVIN pybullet environment; no policy inference.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
