#!/usr/bin/env python3
"""Measure CALVIN action sensitivity to language under a fixed observation.

This is a deployment-only diagnostic.  For every layout it resets the bridge
before each instruction, so the RGB/state/history input and policy RNG seed are
held fixed while the instruction changes.  It deliberately records only the
action chunk and the observation fingerprint; it does not alter model weights,
training data, or simulator state beyond the read-only reset used to create an
observation.

The probe is useful for distinguishing three cases:

* left/right and color swaps both change the action: language reaches execution;
* left/right changes but color swaps do not: action semantics work, object
  grounding is weak;
* no instruction changes the action: the deployed language ingress or lookup is
  broken.

Run this on the CALVIN host where the official environment and bridge are
installed.  The bridge must be a formal CALVIN checkpoint bridge, not the
LIBERO bridge or the smoke-zero policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient, policy_observation
from clearvla.benchmarks.calvin_eval import (
    CALVIN_DEFAULT_INITIAL_STATE,
    _environment,
    validate_bridge_health,
)

INSTRUCTIONS = (
    "go push the blue block right",
    "go push the red block right",
    "go push the pink block right",
    "go push the blue block left",
    "go push the red block left",
    "go push the pink block left",
)

OFFICIAL_TASKS = {
    "go push the blue block right": "push_blue_block_right",
    "go push the red block right": "push_red_block_right",
    "go push the pink block right": "push_pink_block_right",
    "go push the blue block left": "push_blue_block_left",
    "go push the red block left": "push_red_block_left",
    "go push the pink block left": "push_pink_block_left",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _fingerprint_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    rgb = observation.get("rgb_obs")
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation lacks rgb_obs")
    state = np.asarray(observation.get("robot_obs"), dtype=np.float32)
    if state.ndim != 1 or state.shape[0] < 7:
        raise ValueError("CALVIN robot_obs must contain at least seven values")
    result: dict[str, Any] = {
        "robot_obs_shape": list(state.shape),
        "robot_obs_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "robot_obs_first7": state[:7].astype(float).tolist(),
    }
    for name in ("rgb_static", "rgb_gripper"):
        image = np.asarray(rgb[name])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"CALVIN {name} must be HxWx3")
        result[f"{name}_shape"] = list(image.shape)
        result[f"{name}_sha256"] = hashlib.sha256(
            np.ascontiguousarray(image).tobytes()
        ).hexdigest()
    return result


def _as_policy_observation(
    observation: Mapping[str, Any], previous_action: np.ndarray
):
    rgb = observation.get("rgb_obs")
    if not isinstance(rgb, Mapping):
        raise ValueError("CALVIN observation lacks rgb_obs")
    robot = np.asarray(observation.get("robot_obs"), dtype=np.float32)
    return policy_observation(
        top=np.asarray(rgb["rgb_static"]),
        wrist=np.asarray(rgb["rgb_gripper"]),
        state=robot[:7],
        action_state=np.asarray(previous_action, dtype=np.float32),
    )


def _layout_states() -> dict[str, dict[str, object]]:
    standard = dict(CALVIN_DEFAULT_INITIAL_STATE)
    # These are official CALVIN symbolic initial-condition names.  They are
    # intentionally explicit in the report so a failed condition is visible
    # instead of being silently replaced by the standard layout.
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


def _state_for_initial_condition(
    initial_condition: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the official CALVIN symbolic layout without Hydra.

    This is the small pure portion of
    ``calvin_agent.evaluation.utils.get_env_state_for_initial_condition``.
    Keeping it here lets the probe run in the inference-only ClearVLA
    environment, which intentionally does not install Hydra.  The coordinates
    and state slots are copied from the official evaluator; the deterministic
    local seed only controls the two table-slot permutation and block yaw.
    """

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
    # Stable per-layout seed; it is not used by the policy and is recorded only
    # through the resulting observation fingerprint.
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


def _chunk_summary(chunk: np.ndarray) -> dict[str, Any]:
    value = np.asarray(chunk, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
        raise ValueError(f"bridge returned invalid action chunk {value.shape}")
    arm = value[:, :6]
    return {
        "shape": list(value.shape),
        "first": value[0].astype(float).tolist(),
        "first_arm_l2": float(np.linalg.norm(arm[0])),
        "mean": value.mean(axis=0).astype(float).tolist(),
        "arm_mean": arm.mean(axis=0).astype(float).tolist(),
        "arm_sum": arm.sum(axis=0).astype(float).tolist(),
        "arm_abs_mean": np.abs(arm).mean(axis=0).astype(float).tolist(),
        "gripper_values": sorted(
            {float(item) for item in value[:, 6].reshape(-1).tolist()}
        ),
        "array": value.astype(float).tolist(),
    }


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    arm = delta[:, :6]
    return {
        "chunk_rmse": float(np.sqrt(np.mean(delta * delta))),
        "first_rmse": float(np.sqrt(np.mean(delta[0] * delta[0]))),
        "arm_chunk_rmse": float(np.sqrt(np.mean(arm * arm))),
        "arm_first_rmse": float(np.sqrt(np.mean(arm[0] * arm[0]))),
        "first_max_abs": float(np.max(np.abs(delta[0]))),
        "arm_first_delta": delta[0, :6].astype(float).tolist(),
        "gripper_first_delta": float(delta[0, 6]),
    }


def _read_state(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"initial state must be a JSON object: {path}")
    return {str(key): item for key, item in payload.items()}


def run_probe(
    *,
    dataset_root: Path,
    endpoint: str,
    output: Path,
    timeout: float,
    repeats: int,
    state_override: Path | None,
) -> dict[str, Any]:
    if repeats <= 0 or repeats > 8:
        raise ValueError("repeats must be in [1,8]")
    client = RemotePolicyClient(endpoint=endpoint, timeout=timeout)
    health = client.health()
    validate_bridge_health(health, allow_smoke_policy=False)
    layouts = _layout_states()
    override = _read_state(state_override)
    if override is not None:
        layouts = {"custom": override}
    env = _environment(dataset_root, show_gui=False)
    records: dict[str, Any] = {}
    try:
        for layout_name, initial_state in layouts.items():
            robot_obs, scene_obs = _state_for_initial_condition(initial_state)
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            observation = env.get_obs()
            fingerprint = _fingerprint_observation(observation)
            base_policy_obs = _as_policy_observation(
                observation, np.zeros(7, dtype=np.float32)
            )
            instructions = list(INSTRUCTIONS)
            layout_record: dict[str, Any] = {
                "initial_state": _jsonable(initial_state),
                "observation": fingerprint,
                "instructions": {},
                "pairwise_vs_blue_right": {},
                "repeat_consistency": {},
                "official_task_for_instruction": dict(OFFICIAL_TASKS),
            }
            chunks: dict[str, list[np.ndarray]] = {}
            for instruction in instructions:
                chunks[instruction] = []
                for repeat in range(repeats):
                    # reset=True is crucial: every call sees the same causal
                    # history and the bridge resets its named policy RNG.
                    chunk = client.act(
                        base_policy_obs,
                        instruction,
                        reset=True,
                    )
                    chunks[instruction].append(np.asarray(chunk, dtype=np.float32))
                layout_record["instructions"][instruction] = _chunk_summary(
                    chunks[instruction][0]
                )
                if repeats > 1:
                    layout_record["repeat_consistency"][instruction] = [
                        _difference(chunks[instruction][0], value)
                        for value in chunks[instruction][1:]
                    ]
            baseline = chunks[INSTRUCTIONS[0]][0]
            for instruction in instructions[1:]:
                layout_record["pairwise_vs_blue_right"][instruction] = _difference(
                    chunks[instruction][0], baseline
                )
            records[layout_name] = layout_record
    finally:
        env.close()

    result = {
        "schema": "clearvla-calvin-language-counterfactual-v1",
        "endpoint": endpoint,
        "dataset_root": str(dataset_root),
        "bridge_health": _jsonable(health),
        "repeats": int(repeats),
        "instructions": list(INSTRUCTIONS),
        "layouts": records,
        "interpretation": {
            "color_grounding_candidate": (
                "compare red/pink versus blue under the same layout; "
                "small arm_first_rmse while left/right is large suggests weak color binding"
            ),
            "language_ingress_candidate": (
                "all pairwise deltas near repeat noise suggests language ingress or lookup failure"
            ),
            "position_shortcut_candidate": (
                "color deltas grow only after the blue object is moved suggests fixed-slot visual prior"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18772")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--state-json", type=Path, default=None)
    args = parser.parse_args()
    result = run_probe(
        dataset_root=args.dataset_root,
        endpoint=str(args.endpoint),
        output=args.output,
        timeout=float(args.timeout),
        repeats=int(args.repeats),
        state_override=args.state_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
