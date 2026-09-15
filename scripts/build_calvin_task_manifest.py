"""Build an exact single-task split manifest from a converted CALVIN root.

The HDF5 files stay in their original cache namespace.  Only episode names are
serialized, so decoded/DINO cache fingerprints remain valid and no large data
copy is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py

SCHEMA = "clearvla-episode-splits-v1"


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_manifest(root: Path, task: str) -> dict[str, object]:
    root = root.expanduser().resolve()
    dataset_manifest = json.loads(
        (root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if (
        dataset_manifest.get("converter_schema") != "clearvla-calvin-converter-v2"
        or dataset_manifest.get("split_unit") != "source-trajectory"
    ):
        raise ValueError(
            "CALVIN task manifests require converter-v2 trajectory/terminal data"
        )
    source = root / "splits.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unexpected source split schema in {source}")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != {"train", "val", "test"}:
        raise ValueError("source split manifest must contain train/val/test")
    selected: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    trajectory_ids: dict[str, set[str]] = {
        name: set() for name in ("train", "val", "test")
    }
    for split in selected:
        for raw_name in raw_splits[split]:
            name = str(raw_name)
            path = root / f"{name}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as handle:
                observed = _text(handle.attrs.get("task", ""))
                trajectory_id = _text(
                    handle.attrs.get("source_trajectory_id", "")
                ).strip()
            if observed == task:
                if not trajectory_id:
                    raise ValueError(f"{path} has no source trajectory identity")
                selected[split].append(name)
                trajectory_ids[split].add(trajectory_id)
    if any(not values for values in selected.values()):
        raise ValueError(f"task {task!r} has an empty split: {selected}")
    if any(
        trajectory_ids[left].intersection(trajectory_ids[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError(f"task {task!r} crosses source trajectories between splits")
    return {
        "schema": SCHEMA,
        "task_filter": task,
        "source_split_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_root": str(root),
        "split_unit": "source-trajectory",
        "trajectory_counts": {
            name: len(trajectory_ids[name]) for name in ("train", "val", "test")
        },
        "splits": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.root, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
