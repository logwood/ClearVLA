"""Render a raw LIBERO expert demonstration as a top/wrist MP4.

This is an artifact helper only: it reads an HDF5 demonstration and never
changes the dataset.  It supports the raw LIBERO layout (``data/demo_N/obs``)
and the converted ClearVLA layout (top-level ``observations/images``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def _frames(handle: h5py.File, demo: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "data" in handle:
        group = handle["data"][demo]
        images = group["obs"]
        top = np.asarray(images["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(images["eye_in_hand_rgb"], dtype=np.uint8)
        actions = np.asarray(group["actions"], dtype=np.float32)
    else:
        images = handle["observations"]["images"]
        top = np.asarray(images["cam_high"], dtype=np.uint8)
        wrist = np.asarray(images["cam_right_wrist"], dtype=np.uint8)
        actions = np.asarray(handle["action"], dtype=np.float32)
    if top.shape != wrist.shape or top.ndim != 4 or top.shape[-1] != 3:
        raise ValueError(f"camera arrays must match [T,H,W,3], got {top.shape} and {wrist.shape}")
    if actions.ndim != 2 or actions.shape[0] < top.shape[0] or actions.shape[1] < 7:
        raise ValueError(f"actions do not match camera frames: {actions.shape} vs {top.shape}")
    return top, wrist, actions[: top.shape[0]]


def _transition_rows(gripper: np.ndarray) -> list[int]:
    signs = np.sign(np.asarray(gripper, dtype=np.float32))
    return [int(i) for i in (np.flatnonzero(signs[1:] != signs[:-1]) + 1)]


def render(
    source: Path,
    output: Path,
    metadata: Path,
    *,
    demo: str,
    fps: float,
    hold_frames: int,
    annotate: bool,
) -> dict[str, object]:
    with h5py.File(source, "r") as handle:
        top, wrist, actions = _frames(handle, demo)
    height, width = map(int, top.shape[1:3])
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (2 * width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output}")
    transitions = _transition_rows(actions[:, 6])
    labels: list[str] = []
    try:
        for index in range(int(top.shape[0]) + int(hold_frames)):
            source_index = min(index, int(top.shape[0]) - 1)
            left = cv2.cvtColor(top[source_index], cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(wrist[source_index], cv2.COLOR_RGB2BGR)
            frame = np.concatenate([left, right], axis=1)
            if annotate:
                if index >= top.shape[0]:
                    label = f"demo_11  terminal hold {index - top.shape[0] + 1}/{hold_frames}"
                else:
                    command = float(actions[source_index, 6])
                    phase = "CLOSE" if command > 0.1 else "OPEN" if command < -0.1 else "HOLD"
                    label = f"demo_11  frame {source_index:03d}  gripper {command:+.2f} {phase}"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 18), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (4, 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            if index < top.shape[0]:
                labels.append(label if annotate else "")
    finally:
        writer.release()
    payload = {
        "schema": "clearvla-libero-expert-video-v1",
        "source_hdf5": str(source),
        "source_demo": demo,
        "video": str(output),
        "fps": float(fps),
        "real_observation_frames": int(top.shape[0]),
        "terminal_hold_frames": int(hold_frames),
        "frame_semantics": "raw HDF5 observation frames followed by a visual terminal hold",
        "camera_panels": {"left": "agentview_rgb/cam_high", "right": "eye_in_hand_rgb/cam_right_wrist"},
        "gripper_transition_rows": transitions,
        "gripper_transition_commands": [float(actions[i, 6]) for i in transitions],
        "annotated": bool(annotate),
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--hold-frames", type=int, default=20)
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(render(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
