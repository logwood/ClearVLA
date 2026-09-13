"""Build the training-only CALVIN semantic-role sidecar (v2).

Only authoritative episode/task/instruction metadata and valid causal window
ranges are read.  The builder never assigns a colour to a K object row and
never reads privileged ``scene_obs``.  A color instruction becomes a role
target (red/blue/pink); ambiguous multi-colour text is soft and discounted;
non-colour instructions receive an exact null target with zero effective
coverage.  The ordinary action objective still trains those non-colour rows.
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

from clearvla.mainline.calvin_binding_contract import (
    CALVIN_OBJECT_BINDING_ROLE_NAMES,
    CALVIN_OBJECT_BINDING_TARGET_COUNT,
)
from clearvla.mainline.data.calvin_object_binding import (
    CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
    calvin_episode_inventory_digest,
)

ROLE_TO_INDEX = {
    name.removesuffix("_block"): index
    for index, name in enumerate(CALVIN_OBJECT_BINDING_ROLE_NAMES[:3])
}
_COLOUR_BLOCK_RE = re.compile(
    r"(?<![a-z])(red|blue|pink)(?:[_\s-]+)block(?![a-z])", re.IGNORECASE
)


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


def _manifest_inventory(path: Path) -> tuple[str, ...]:
    """Return exactly the episode inventory consumed by the split manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "clearvla-episode-splits-v1":
        raise ValueError(f"{path} is not a clearvla-episode-splits-v1 manifest")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("episode split manifest must contain train/val/test")
    names: list[str] = []
    for split in ("train", "val", "test"):
        rows = splits[split]
        if not isinstance(rows, list) or not rows or any(
            not isinstance(value, str) or not value for value in rows
        ):
            raise ValueError(f"manifest split {split!r} must contain episode names")
        names.extend(rows)
    if len(names) != len(set(names)):
        raise ValueError("manifest split membership overlaps")
    return tuple(names)


def _target_for_text(task: str, instruction: str) -> tuple[np.ndarray, float, float, str]:
    text = f"{task} {instruction}".lower()
    colours = list(
        dict.fromkeys(
            match.group(1).lower() for match in _COLOUR_BLOCK_RE.finditer(text)
        )
    )
    target = np.zeros(CALVIN_OBJECT_BINDING_TARGET_COUNT, dtype=np.float32)
    if len(colours) == 1:
        target[ROLE_TO_INDEX[colours[0]]] = 1.0
        return target, 1.0, 0.0, f"{colours[0]}_block"
    if len(colours) > 1:
        weight = 1.0 / float(len(colours))
        for colour in colours:
            target[ROLE_TO_INDEX[colour]] = weight
        return (
            target,
            1.0,
            min(1.0, float(len(colours) - 1) / len(colours)),
            "multi_colour",
        )
    # There is no reliable role label for a drawer/handle/light instruction
    # from text alone.  Keep the row in the join for completeness but do not
    # turn the null into a false positive supervised binding.
    target[-1] = 1.0
    return target, 0.0, 1.0, "null_non_colour"


def build_sidecar(*, hdf5_root: Path, manifest: Path, output: Path) -> dict[str, object]:
    files = sorted(hdf5_root.glob("*.hdf5")) + sorted(hdf5_root.glob("*.h5"))
    files = sorted(set(path.resolve() for path in files))
    if not files:
        raise FileNotFoundError(f"no HDF5 episodes under {hdf5_root}")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    inventory = _manifest_inventory(manifest)
    by_id = {
        path.relative_to(hdf5_root.resolve()).with_suffix("").as_posix(): path
        for path in files
    }
    missing = sorted(set(inventory) - set(by_id))
    if missing:
        raise FileNotFoundError(
            f"manifest references {len(missing)} episodes absent from {hdf5_root}; "
            f"first={missing[:4]}"
        )
    files = [by_id[episode_id] for episode_id in inventory]

    episode_ids: list[str] = []
    centers: list[int] = []
    role_targets: list[np.ndarray] = []
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
            start = int(attrs.get("valid_center_start", 24))
            end = int(attrs.get("valid_center_end", length - 49))
            if not 0 <= start <= end < length:
                raise ValueError(
                    f"{path}: invalid valid center range [{start},{end}] for T={length}"
                )
        target, coverage, ambiguity, role = _target_for_text(task, instruction)
        task_counts[task or "<no-task>"] += 1
        role_counts[role] += 1
        for center in range(start, end + 1):
            episode_ids.append(episode_id)
            centers.append(center)
            role_targets.append(target.copy())
            coverages.append(coverage)
            ambiguities.append(ambiguity)
            roles.append(role)

    if len(set(zip(episode_ids, centers))) != len(episode_ids):
        raise ValueError("generated sidecar has duplicate episode/center keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = calvin_episode_inventory_digest(inventory)
    manifest_digest = _file_sha256(manifest)
    target_stats = {
        "episode_count": len(inventory),
        "window_count": len(episode_ids),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "role_window_counts": dict(sorted(Counter(roles).items())),
        "role_episode_counts": dict(sorted(role_counts.items())),
        "role_names": list(CALVIN_OBJECT_BINDING_ROLE_NAMES),
        "target_policy": "instruction-role-v2; non-colour exact-null with coverage=0",
        "privileged_scene_obs_used": False,
    }
    np.savez_compressed(
        output,
        schema=np.asarray(CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA),
        episode_ids=np.asarray(episode_ids, dtype="U"),
        centers=np.asarray(centers, dtype=np.int64),
        role_targets=np.asarray(role_targets, dtype=np.float32),
        coverage=np.asarray(coverages, dtype=np.float32),
        ambiguity=np.asarray(ambiguities, dtype=np.float32),
        source_digest=np.asarray(source_digest),
        manifest_digest=np.asarray(manifest_digest),
        role_names=np.asarray(
            json.dumps(CALVIN_OBJECT_BINDING_ROLE_NAMES, separators=(",", ":"))
        ),
        target_policy=np.asarray(target_stats["target_policy"]),
        target_stats=np.asarray(json.dumps(target_stats, sort_keys=True, separators=(",", ":"))),
        generator=np.asarray("build_calvin_object_binding_sidecar.py:v2"),
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
