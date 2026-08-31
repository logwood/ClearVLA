"""Render deterministic task-complete RDT contact sheets for manual role audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from clearvla.data.hdf5_episode import (
    decode_hdf5_instruction,
    episode_identity,
    find_hdf5_files,
)
from clearvla.vision.image_io import decode_image_value

CAMERA_KEYS = {
    "high": "observations/images/cam_high",
    "right_wrist": "observations/images/cam_right_wrist",
}


def _indices(length: int, frames: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(round(value))
                for value in np.linspace(0, max(length - 1, 0), min(frames, length))
            }
        )
    )


def _thumbnail(value: object, *, width: int, height: int) -> Image.Image:
    image = Image.fromarray(decode_image_value(value), mode="RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def render_contact_sheets(
    root: Path,
    *,
    task_ids: tuple[str, ...],
    cameras: tuple[str, ...],
    frames_per_episode: int,
    output_dir: Path,
) -> list[dict[str, object]]:
    source = root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Path]] = {task: [] for task in task_ids}
    for path in find_hdf5_files(source, "**/*.hdf5"):
        _identity, partition, task = episode_identity(source, path)
        if partition == "rdt_data" and task in grouped:
            grouped[task].append(path)
    missing = [task for task, paths in grouped.items() if not paths]
    if missing:
        raise ValueError(f"unknown RDT tasks: {missing}")

    reports: list[dict[str, object]] = []
    thumb_width, thumb_height, label_width = 256, 144, 220
    header_height, row_height = 48, thumb_height + 4
    for task in task_ids:
        paths = sorted(grouped[task], key=lambda path: episode_identity(source, path)[0])
        for camera in cameras:
            if camera not in CAMERA_KEYS:
                raise ValueError(f"unsupported contact-sheet camera {camera!r}")
            decoded_rows: list[tuple[str, str, list[Image.Image], list[int]]] = []
            instructions: set[str] = set()
            for path in paths:
                identity, _, _ = episode_identity(source, path)
                with h5py.File(path, "r") as handle:
                    instruction = decode_hdf5_instruction(handle["instruction"][()])
                    instructions.add(instruction)
                    dataset = handle[CAMERA_KEYS[camera]]
                    indices = list(_indices(int(dataset.shape[0]), frames_per_episode))
                    images = [
                        _thumbnail(dataset[index], width=thumb_width, height=thumb_height)
                        for index in indices
                    ]
                decoded_rows.append((identity, path.name, images, indices))
            if len(instructions) != 1:
                raise ValueError(f"task {task!r} owns multiple instructions")
            columns = frames_per_episode
            canvas = Image.new(
                "RGB",
                (
                    label_width + columns * thumb_width,
                    header_height + len(decoded_rows) * row_height,
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            instruction = next(iter(instructions))
            draw.text((4, 4), f"{task} | {camera} | {instruction}", fill="black")
            for row, (identity, _name, images, indices) in enumerate(decoded_rows):
                top = header_height + row * row_height
                draw.text((4, top + 4), identity.split("/")[-1], fill="black")
                for column, (image, index) in enumerate(zip(images, indices, strict=True)):
                    left = label_width + column * thumb_width
                    canvas.paste(image, (left, top))
                    draw.rectangle((left, top, left + 46, top + 16), fill="black")
                    draw.text((left + 2, top + 1), str(index), fill="white")
            destination = output / f"{task}.{camera}.jpg"
            canvas.save(destination, format="JPEG", quality=88, optimize=True)
            reports.append(
                {
                    "task_id": task,
                    "instruction": instruction,
                    "camera": camera,
                    "episodes": len(paths),
                    "frames_per_episode": frames_per_episode,
                    "path": str(destination),
                    "size_bytes": int(destination.stat().st_size),
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
    (output / "contact_sheets.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RDT task-complete contact sheets")
    parser.add_argument("root", type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument(
        "--camera", action="append", choices=tuple(CAMERA_KEYS), default=[]
    )
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames_per_episode <= 0:
        raise ValueError("frames per episode must be positive")
    reports = render_contact_sheets(
        args.root,
        task_ids=tuple(args.task),
        cameras=tuple(args.camera or ("high",)),
        frames_per_episode=args.frames_per_episode,
        output_dir=args.output_dir,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["render_contact_sheets"]
