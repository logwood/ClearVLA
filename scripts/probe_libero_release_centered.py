"""Teacher-forced release-centered probe for a LIBERO checkpoint.

This is a read-only diagnostic.  It walks the causal observations from one or
more converted LIBERO episodes, asks the remote policy for a prediction only at
selected rows around each expert release transition, and records the returned
24-step gripper plan.  No simulator state, checkpoint, or training data is
modified.
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
    """Return longest contiguous open/close runs in a gripper sequence."""

    labels = np.where(values < -threshold, "open", np.where(values > threshold, "close", "hold"))
    result: dict[str, int] = {"open": 0, "close": 0, "hold": 0}
    if labels.size == 0:
        return result
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and labels[end] == labels[start]:
            end += 1
        name = str(labels[start])
        result[name] = max(result[name], end - start)
        start = end
    return result


def _observation(stream: h5py.File, index: int) -> Any:
    state = np.asarray(stream["state"][index], dtype=np.float32)
    action_state = np.asarray(stream["action_state"][index], dtype=np.float32)
    top = np.asarray(stream["observations/images/cam_high"][index], dtype=np.uint8)
    wrist = np.asarray(stream["observations/images/cam_right_wrist"][index], dtype=np.uint8)
    return policy_observation(top=top, wrist=wrist, state=state, action_state=action_state)


def _query_at(
    client: RemotePolicyClient,
    stream: h5py.File,
    index: int,
    instruction: str,
) -> dict[str, Any]:
    """Reset history, teacher-force rows before ``index``, then predict there."""

    first = _observation(stream, 0)
    ignored = client.act(first, instruction, reset=True)
    if ignored.shape != (24, 7):
        raise ValueError(f"bridge reset returned {ignored.shape}, expected (24, 7)")
    for cursor in range(1, int(index)):
        client.observe(_observation(stream, cursor))
    current = _observation(stream, int(index))
    chunk = np.asarray(client.act(current, instruction, reset=False), dtype=np.float32)
    if chunk.shape != (24, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"bridge prediction at row {index} has invalid shape {chunk.shape}")
    predicted = np.clip(chunk[:, 6], -1.0, 1.0)
    return {
        "observation_index": int(index),
        "target_action_gripper": float(np.asarray(stream["action"][index], dtype=np.float32)[6]),
        "target_previous_gripper": float(np.asarray(stream["action_state"][index], dtype=np.float32)[6]),
        "target_delta": float(
            np.asarray(stream["action"][index], dtype=np.float32)[6]
            - np.asarray(stream["action_state"][index], dtype=np.float32)[6]
        ),
        "predicted_first_gripper": float(predicted[0]),
        "predicted_min_gripper": float(predicted.min()),
        "predicted_max_gripper": float(predicted.max()),
        "predicted_gripper": predicted.astype(float).tolist(),
        "predicted_open_rows": np.flatnonzero(predicted < -0.1).astype(int).tolist(),
        "predicted_open_run_max": int(_run_lengths(predicted)["open"]),
        "predicted_close_run_max": int(_run_lengths(predicted)["close"]),
    }


def probe_file(
    client: RemotePolicyClient,
    path: Path,
    *,
    instruction: str,
    radius: int,
) -> dict[str, Any]:
    with h5py.File(path, "r") as stream:
        actions = np.asarray(stream["action"], dtype=np.float32)
        action_state = np.asarray(stream["action_state"], dtype=np.float32)
        source_count = int(stream.attrs.get("source_action_count", actions.shape[0]))
        if source_count <= 0 or source_count > actions.shape[0]:
            raise ValueError(f"invalid source_action_count={source_count} in {path}")
        release_delta = actions[:source_count, 6] - action_state[:source_count, 6]
        releases = np.flatnonzero(release_delta <= -0.1).astype(int).tolist()
        # Row zero is the reset-to-open boundary, not a physical release.
        releases = [int(row) for row in releases if int(row) > 0]
        cases: list[dict[str, Any]] = []
        for release in releases:
            indices = sorted(
                {
                    max(0, release - int(radius)),
                    max(0, release - max(1, int(radius) // 2)),
                    max(0, release - 1),
                    release,
                    min(source_count - 1, release + 1),
                    min(source_count - 1, release + max(1, int(radius) // 2)),
                    min(source_count - 1, release + int(radius)),
                }
            )
            for index in indices:
                result = _query_at(client, stream, index, instruction)
                result["release_index"] = int(release)
                result["offset_from_release"] = int(index - release)
                cases.append(result)
        return {
            "path": str(path),
            "source_action_count": source_count,
            "dataset_length": int(actions.shape[0]),
            "release_indices": releases,
            "radius": int(radius),
            "cases": cases,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18781")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.radius < 1:
        raise ValueError("radius must be positive")
    client = RemotePolicyClient(endpoint=str(args.endpoint), timeout=180.0)
    health = client.health()
    if health.get("status") != "ok" or health.get("mode") != "formal-checkpoint":
        raise RuntimeError(f"unexpected bridge health: {health}")
    results = [
        probe_file(client, Path(path), instruction=str(args.instruction), radius=int(args.radius))
        for path in args.paths
    ]
    atomic_json(
        args.output,
        {
            "schema": "clearvla-libero-release-centered-probe-v1",
            "endpoint": str(args.endpoint),
            "instruction": str(args.instruction),
            "radius": int(args.radius),
            "bridge_health": health,
            "files": results,
        },
    )
    print(json.dumps({"output": str(args.output), "files": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()

