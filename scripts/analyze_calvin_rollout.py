#!/usr/bin/env python3
"""Print task-level smoothness and control diagnostics for a CALVIN trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    return [
        (int(values[start]), int(start), int(end - start))
        for start, end in zip(starts, ends)
    ]


def analyze(path: Path) -> dict[str, object]:
    with np.load(path) as payload:
        executed = np.asarray(payload["executed"], dtype=np.float64)
        raw_first = np.asarray(payload["raw_first"], dtype=np.float64)
        raw_chunks = np.asarray(payload["raw_chunks"], dtype=np.float64)
        latency = np.asarray(payload["latency_seconds"], dtype=np.float64)
    if executed.ndim != 2 or executed.shape[1] != 7:
        raise ValueError(f"executed must be [N,7], got {executed.shape}")
    arm = executed[:, :6]
    delta = np.diff(arm, axis=0)
    xyz = arm[:, :3]
    result: dict[str, object] = {
        "steps": int(executed.shape[0]),
        "executed_arm_mean_abs": np.mean(np.abs(arm), axis=0).tolist(),
        "executed_arm_delta_mean_abs": (
            np.mean(np.abs(delta), axis=0).tolist() if delta.size else [0.0] * 6
        ),
        "executed_arm_delta_p95": (
            np.quantile(np.abs(delta), 0.95, axis=0).tolist()
            if delta.size
            else [0.0] * 6
        ),
        "arm_direction_reversals": (
            np.sum(np.signbit(delta[1:]) != np.signbit(delta[:-1]), axis=0).astype(int).tolist()
            if delta.shape[0] > 1
            else [0] * 6
        ),
        "xyz_abs_command_m": (np.abs(xyz).sum(axis=0) * 0.02).tolist(),
        "xyz_command_path_m": float(np.linalg.norm(xyz, axis=1).sum() * 0.02),
        "xyz_net_command_m": float(np.linalg.norm(xyz.sum(axis=0)) * 0.02),
        "gripper_runs": _runs(np.where(executed[:, 6] >= 0, 1, -1).astype(np.int8)),
        "gripper_switches": int(np.count_nonzero(executed[1:, 6] != executed[:-1, 6])),
        "raw_gripper_abs_max": float(np.abs(raw_chunks[:, :, 6]).max()),
        "raw_gripper_oob_values": int(np.count_nonzero(np.abs(raw_chunks[:, :, 6]) > 1.0)),
        "raw_first_gripper_oob_rows": int(np.count_nonzero(np.abs(raw_first[:, 6]) > 1.0)),
        "latency_mean_seconds": float(latency.mean()),
        "latency_p95_seconds": float(np.quantile(latency, 0.95)),
        "latency_max_seconds": float(latency.max()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.actions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
