#!/usr/bin/env python3
"""Build the training-only CALVIN language/object binding sidecar.

The converted CALVIN episodes retain the authoritative task/instruction and
window-boundary metadata as HDF5 root attributes.  This builder turns only
that metadata into a deterministic K=4+null target table; RGB, state and
privileged ``scene_obs`` never enter the policy or the sidecar rows.

The v1 role convention is deliberately explicit and versioned:

* slot 0 = red block;
* slot 1 = blue block;
* slot 2 = pink block;
* slot 3 = reserved non-colour/handle role;
* the final column is null.

Colour-block instructions receive a one-hot role target.  Instructions with
no unambiguous colour (including drawer/slider/light tasks) receive an exact
null target with zero effective coverage, so the bridge is trained only where
the language specifies a colour object.  This avoids pretending that a
drawer handle is a colour-block pointer while retaining those windows for the
ordinary CALVIN action objective and the explicit null compatibility surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from clearvla.mainline.data.calvin_object_binding import (
    CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
    calvin_episode_inventory_digest,
)

SLOT_ROLES = ("red_block", "blue_block", "pink_block", "reserved_non_color")
COLOUR_TO_SLOT = {"red": 0, "blue": 1, "pink": 2}
_COLOUR_RE = re.compile(r"\b(red|blue|pink)\b", re.IGNORECASE)


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_for_text(task: str, instruction: str) -> tuple[np.ndarray, float, float, str]:
    text = f"{task} {instruction}".lower()
    colours = list(
        dict.fromkeys(match.group(1).lower() for match in _COLOUR_RE.finditer(text))
    )
    target = np.zeros(5, dtype=np.float32)
    if len(colours) == 1:
        target[COLOUR_TO_SLOT[colours[0]]] = 1.0
        return target, 1.0, 0.0, f"{colours[0]}_block"
    if len(colours) > 1:
        # Keep a valid soft target for unusual multi-object instructions while
        # discounting it through ambiguity in the bridge loss.
        weight = 1.0 / float(len(colours))
        for colour in colours:
            target[COLOUR_TO_SLOT[colour]] = weight
        return (
            target,
            1.0,
            min(1.0, float(len(colours) - 1) / len(colours)),
            "multi_colour",
        )
    target[-1] = 1.0
    return target, 0.0, 1.0, "null_non_colour"


def build_sidecar(*, hdf5_root: Path, manifest: Path, output: Path) -> dict[str, object]:
    files = sorted(hdf5_root.glob("*.hdf5")) + sorted(hdf5_root.glob("*.h5"))
    files = sorted(set(path.resolve() for path in files))
    if not files:
        raise FileNotFoundError(f"no HDF5 episodes under {hdf5_root}")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    episode_ids: list[str] = []
    centers: list[int] = []
    pointers: list[np.ndarray] = []
    coverages: list[float] = []
    ambiguities: list[float] = []
    roles: list[str] = []
    task_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    for path in files:
        episode_id = path.relative_to(hdf5_root.resolve()).with_suffix("").as_posix()
        with h5py.File(path, "r") as handle:
            attrs = handle.attrs
            task = _text(attrs.get("task", ""))
            instruction = _text(attrs.get("instruction", ""))
            if not task and not instruction:
                raise ValueError(f"{path} has neither task nor instruction metadata")
            length = int(handle["action"].shape[0])
            start_value = attrs.get("valid_center_start", 24)
            end_value = attrs.get("valid_center_end", length - 49)
            start = int(start_value)
            end = int(end_value)
            if not 0 <= start <= end < length:
                raise ValueError(
                    f"{path}: invalid valid center range [{start},{end}] for T={length}"
                )
        pointer, coverage, ambiguity, role = _target_for_text(task, instruction)
        task_counts[task or "<no-task>"] += 1
        role_counts[role] += 1
        for center in range(start, end + 1):
            episode_ids.append(episode_id)
            centers.append(center)
            pointers.append(pointer.copy())
            coverages.append(coverage)
            ambiguities.append(ambiguity)
            roles.append(role)

    if len(set(zip(episode_ids, centers))) != len(episode_ids):
        raise ValueError("generated sidecar has duplicate episode/center keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = calvin_episode_inventory_digest(
        sorted({episode_id for episode_id in episode_ids})
    )
    manifest_digest = _file_sha256(manifest)
    target_stats = {
        "episode_count": len(set(episode_ids)),
        "window_count": len(episode_ids),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "role_window_counts": dict(sorted(Counter(roles).items())),
        "role_episode_counts": dict(sorted(role_counts.items())),
        "slot_roles": list(SLOT_ROLES),
        "target_policy": "colour-role-v1; non-colour exact-null with coverage=0",
    }
    np.savez_compressed(
        output,
        schema=np.asarray(CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA),
        episode_ids=np.asarray(episode_ids, dtype="U"),
        centers=np.asarray(centers, dtype=np.int64),
        pointer_targets=np.asarray(pointers, dtype=np.float32),
        coverage=np.asarray(coverages, dtype=np.float32),
        ambiguity=np.asarray(ambiguities, dtype=np.float32),
        source_digest=np.asarray(source_digest),
        manifest_digest=np.asarray(manifest_digest),
        slot_roles=np.asarray(json.dumps(SLOT_ROLES, separators=(",", ":"))),
        target_stats=np.asarray(
            json.dumps(target_stats, sort_keys=True, separators=(",", ":"))
        ),
        generator=np.asarray("build_calvin_object_binding_sidecar.py:v1"),
    )
    return {
        "schema": CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
        "output": str(output),
        "source_digest": source_digest,
        "manifest_digest": manifest_digest,
        **target_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_sidecar(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
