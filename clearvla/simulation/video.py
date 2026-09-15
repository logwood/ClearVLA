"""Export evaluator episode frames to a small, deterministic MP4 artifact.

The simulator recorder deliberately stores RGB frames in the episode HDF5 so
that the visual evidence and the telemetry always share the same step index.
This module is a presentation-layer helper: it reads those frames after an
episode has been committed and never changes policy observations or simulator
state.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TOP_KEY = "observations/images/cam_high"
WRIST_KEY = "observations/images/cam_right_wrist"


def _control_hz(stream: h5py.File) -> float:
    """Read the recorded control frequency, with the StackCube default."""

    raw = stream.attrs.get("environment_descriptor_json")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw is not None:
        try:
            descriptor = json.loads(str(raw))
            value = float(descriptor.get("control_hz", 20.0))
            if np.isfinite(value) and value > 0.0:
                return value
        except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
            pass
    return 20.0


def _frames(dataset: h5py.Dataset, *, name: str) -> np.ndarray:
    value = np.asarray(dataset)
    if value.ndim != 4 or value.shape[-1] != 3 or value.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty THWC RGB dataset")
    if value.dtype != np.uint8:
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric RGB frames")
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def _output_path(episode: Path, output: str | Path | None) -> Path:
    destination = (
        episode.with_name(f"{episode.stem}_side_by_side.mp4")
        if output is None
        else Path(output)
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _write_cv2(frames: np.ndarray, destination: Path, fps: float) -> str:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "video export requires imageio/imageio-ffmpeg or OpenCV"
        ) from error

    height, width = (int(frames.shape[1]), int(frames.shape[2]))
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"OpenCV could not open MP4 writer for {destination}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return "mp4v"


def _write_imageio(frames: np.ndarray, destination: Path, fps: float) -> str:
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401  # imported to verify the backend
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("imageio/ffmpeg backend is unavailable") from error
    writer = imageio.get_writer(
        str(destination),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        quality=7,
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return "libx264"


def export_episode_video(
    episode_path: str | Path,
    output_path: str | Path | None = None,
    *,
    fps: float | None = None,
    layout: str = "side_by_side",
) -> dict[str, Any]:
    """Export one recorded episode and return compact artifact metadata.

    ``side_by_side`` is the default and places the top camera on the left and
    the wrist camera on the right.  ``top`` and ``wrist`` are useful for quick
    diagnostics.  The destination is never overwritten; encoding is atomic so
    a killed process cannot leave a file that looks complete.
    """

    episode = Path(episode_path)
    if not episode.is_file():
        raise FileNotFoundError(episode)
    if layout not in {"side_by_side", "top", "wrist"}:
        raise ValueError("layout must be side_by_side, top, or wrist")
    destination = _output_path(episode, output_path)

    with h5py.File(episode, "r") as stream:
        top = _frames(stream[TOP_KEY], name=TOP_KEY)
        wrist = _frames(stream[WRIST_KEY], name=WRIST_KEY)
        if top.shape[0] != wrist.shape[0]:
            raise ValueError("top and wrist camera frame counts do not match")
        if top.shape[1:] != wrist.shape[1:]:
            raise ValueError("top and wrist camera frame shapes do not match")
        rate = _control_hz(stream) if fps is None else float(fps)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("fps must be finite and positive")
        if layout == "side_by_side":
            frames = np.concatenate((top, wrist), axis=2)
        elif layout == "top":
            frames = top
        else:
            frames = wrist

    # Encode beside the final destination and atomically rename it.  A named
    # temporary file is used because OpenCV and imageio both require a path.
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".mp4", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    codec = ""
    try:
        try:
            codec = _write_imageio(frames, temporary, rate)
        except RuntimeError:
            codec = _write_cv2(frames, temporary, rate)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "path": str(destination.resolve()),
        "frames": int(frames.shape[0]),
        "fps": float(rate),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "layout": layout,
        "codec": codec,
    }


__all__ = ["export_episode_video"]
