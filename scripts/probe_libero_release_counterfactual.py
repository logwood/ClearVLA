"""Counterfactual release-timing probe for a LIBERO checkpoint.

This diagnostic keeps the expert image/state at a physical release row, but
replaces only the current ``action_state[6]`` (the previous gripper command)
with an open command.  It is intended to test whether the policy has learned
the open continuation but waits for an open command to appear in its history.
No checkpoint, simulator state, or source HDF5 is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from clearvla.benchmarks.bridge import RemotePolicyClient, policy_observation
from clearvla.benchmarks.io import atomic_json


def _run_lengths(values: np.ndarray, threshold: float = 0.05) -> dict[str, int]:
    labels = np.where(values < -threshold, "open", np.where(values > threshold, "close", "hold"))
    result: dict[str, int] = {"open": 0, "close": 0, "hold": 0}
    if labels.size == 0:
        return result
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and labels[end] == labels[start]:
            end += 1
        result[str(labels[start])] = max(result[str(labels[start])], end - start)
        start = end
    return result


def _observation(stream: h5py.File, index: int, *, previous_gripper: float | None = None) -> Any:
    state = np.asarray(stream["state"][index], dtype=np.float32)
    action_state = np.asarray(stream["action_state"][index], dtype=np.float32)
    if previous_gripper is not None:
        action_state = action_state.copy()
        action_state[6] = float(previous_gripper)
    top = np.asarray(stream["observations/images/cam_high"][index], dtype=np.uint8)
    wrist = np.asarray(stream["observations/images/cam_right_wrist"][index], dtype=np.uint8)
    return policy_observation(top=top, wrist=wrist, state=state, action_state=action_state)


def query_release(
    client: RemotePolicyClient,
    stream: h5py.File,
    release_index: int,
    instruction: str,
    *,
    forced_previous_gripper: float,
) -> dict[str, Any]:
    """Teacher-force through the row before release, then alter only row ``release_index``."""

    first = _observation(stream, 0)
    reset_chunk = client.act(first, instruction, reset=True)
    if reset_chunk.shape != (24, 7):
        raise ValueError(f"bridge reset returned {reset_chunk.shape}, expected (24, 7)")
    for cursor in range(1, int(release_index)):
        client.observe(_observation(stream, cursor))
    current_native = _observation(stream, int(release_index))
    current_counterfactual = _observation(
        stream,
        int(release_index),
        previous_gripper=float(forced_previous_gripper),
    )
    # Keep the visual/state row exactly the same; only the gripper component of
    # the current action_state is replaced.  The bridge appends this row once.
    del current_native
    chunk = np.asarray(client.act(current_counterfactual, instruction, reset=False), dtype=np.float32)
    if chunk.shape != (24, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"bridge prediction at row {release_index} has invalid shape {chunk.shape}")
    predicted = np.clip(chunk[:, 6], -1.0, 1.0)
    native_previous = float(np.asarray(stream["action_state"][release_index], dtype=np.float32)[6])
    target = float(np.asarray(stream["action"][release_index], dtype=np.float32)[6])
    return {
        "release_index": int(release_index),
        "native_previous_gripper": native_previous,
        "forced_previous_gripper": float(forced_previous_gripper),
        "target_action_gripper": target,
        "target_delta_native": target - native_previous,
        "predicted_first_gripper": float(predicted[0]),
        "predicted_min_gripper": float(predicted.min()),
        "predicted_max_gripper": float(predicted.max()),
        "predicted_gripper": predicted.astype(float).tolist(),
        "predicted_open_rows": np.flatnonzero(predicted < -0.1).astype(int).tolist(),
        "predicted_open_run_max": int(_run_lengths(predicted)["open"]),
        "predicted_close_run_max": int(_run_lengths(predicted)["close"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18781")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forced-previous-gripper", type=float, default=-1.0)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if not -1.0 <= args.forced_previous_gripper <= 1.0:
        raise ValueError("forced previous gripper must be within [-1, 1]")
    client = RemotePolicyClient(endpoint=str(args.endpoint), timeout=180.0)
    health = client.health()
    if health.get("status") != "ok" or health.get("mode") != "formal-checkpoint":
        raise RuntimeError(f"unexpected bridge health: {health}")
    files: list[dict[str, Any]] = []
    for path in args.paths:
        with h5py.File(path, "r") as stream:
            actions = np.asarray(stream["action"], dtype=np.float32)
            action_state = np.asarray(stream["action_state"], dtype=np.float32)
            source_count = int(stream.attrs.get("source_action_count", actions.shape[0]))
            deltas = actions[:source_count, 6] - action_state[:source_count, 6]
            releases = [int(i) for i in np.flatnonzero(deltas <= -0.1) if int(i) > 0]
            cases = [
                query_release(
                    client,
                    stream,
                    release,
                    str(args.instruction),
                    forced_previous_gripper=float(args.forced_previous_gripper),
                )
                for release in releases
            ]
            files.append(
                {
                    "path": str(path),
                    "source_action_count": source_count,
                    "release_indices": releases,
                    "cases": cases,
                }
            )
    atomic_json(
        args.output,
        {
            "schema": "clearvla-libero-release-counterfactual-probe-v1",
            "endpoint": str(args.endpoint),
            "instruction": str(args.instruction),
            "forced_previous_gripper": float(args.forced_previous_gripper),
            "bridge_health": health,
            "files": files,
        },
    )
    print(json.dumps({"output": str(args.output), "files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
