"""Build an exact multi-task subset of a CALVIN raw-overlay manifest.

The source manifest already owns the trajectory-disjoint train/val/test split
and the cache-to-raw episode map.  This tool only filters that immutable
membership by the task stored in each cached HDF5 file; it never resplits
trajectories or copies dataset payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py


SCHEMA = "clearvla-episode-splits-v1"
SPLITS = ("train", "val", "test")


def _text(value: object) -> str:
    # h5py may return a NumPy scalar rather than a Python ``bytes`` object.
    # Normalize that scalar before decoding so the filter is stable across
    # NumPy/h5py versions on local and remote hosts.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_source(path: Path) -> tuple[Path, dict[str, object]]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"{source} is not a {SCHEMA} manifest")
    if str(payload.get("task_filter", "")):
        raise ValueError("the source overlay manifest must not already be task-filtered")
    if payload.get("split_unit") != "source-trajectory":
        raise ValueError("CALVIN subset manifests require source-trajectory splits")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != set(SPLITS):
        raise ValueError("source overlay manifest must contain exactly train/val/test")
    names: list[str] = []
    for split in SPLITS:
        rows = raw_splits[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source split {split!r} must be a non-empty list")
        if any(not isinstance(value, str) or not value for value in rows):
            raise TypeError(f"source split {split!r} contains an invalid episode id")
        names.extend(rows)
    if len(names) != len(set(names)):
        raise ValueError("source overlay manifest contains overlapping episode ids")
    return source, payload


def build_subset_manifest(
    source_manifest: Path,
    cached_root: Path,
    tasks: Sequence[str],
) -> dict[str, object]:
    source, payload = _load_source(source_manifest)
    root = cached_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CALVIN cached HDF5 root does not exist: {root}")

    task_order = tuple(str(task).strip() for task in tasks)
    if not task_order or any(not task for task in task_order):
        raise ValueError("at least one non-empty --task is required")
    if len(task_order) != len(set(task_order)):
        raise ValueError("requested CALVIN tasks must be unique")
    source_task_order = payload.get("task_order")
    if not isinstance(source_task_order, list):
        raise ValueError("source overlay manifest has no task_order registry")
    missing_tasks = sorted(set(task_order).difference(map(str, source_task_order)))
    if missing_tasks:
        raise ValueError(f"requested tasks are absent from the source manifest: {missing_tasks}")

    recorded_root = str(payload.get("cached_prefix_root", "")).strip()
    if recorded_root and Path(recorded_root).expanduser().resolve() != root:
        raise ValueError(
            "--cached-root differs from source manifest cached_prefix_root: "
            f"{root} != {recorded_root}"
        )
    raw_episode_map = payload.get("raw_episode_map")
    if not isinstance(raw_episode_map, Mapping):
        raise ValueError("source raw-overlay manifest has no raw_episode_map")

    raw_splits = payload["splits"]
    assert isinstance(raw_splits, Mapping)
    selected: dict[str, list[str]] = {split: [] for split in SPLITS}
    task_counts: dict[str, dict[str, int]] = {
        split: {task: 0 for task in task_order} for split in SPLITS
    }
    selected_map: dict[str, str] = {}
    for split in SPLITS:
        rows = raw_splits[split]
        assert isinstance(rows, list)
        for episode_id in rows:
            path = root / f"{episode_id}.hdf5"
            if not path.is_file():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as handle:
                task = _text(handle.attrs.get("task", "")).strip()
            if task not in task_counts[split]:
                continue
            if episode_id not in raw_episode_map:
                raise ValueError(f"selected episode {episode_id!r} has no raw mapping")
            raw_id = str(raw_episode_map[episode_id]).strip()
            if not raw_id:
                raise ValueError(f"selected episode {episode_id!r} has an empty raw mapping")
            selected[split].append(episode_id)
            selected_map[episode_id] = raw_id
            task_counts[split][task] += 1

    empty = [
        f"{split}:{task}"
        for split in SPLITS
        for task in task_order
        if task_counts[split][task] == 0
    ]
    if empty:
        raise ValueError(f"requested task subsets must cover every split: {empty}")

    source_counts = payload.get("task_counts")
    if isinstance(source_counts, Mapping):
        for split in SPLITS:
            split_counts = source_counts.get(split)
            if not isinstance(split_counts, Mapping):
                raise ValueError(f"source task_counts has no {split!r} mapping")
            expected = {task: int(split_counts.get(task, -1)) for task in task_order}
            if task_counts[split] != expected:
                raise ValueError(
                    f"cached HDF5 task identities disagree with source task_counts[{split!r}]"
                )

    result: dict[str, object] = {
        "schema": SCHEMA,
        "task_filter": "",
        "split_unit": "source-trajectory",
        "source_overlay_manifest": str(source),
        "source_overlay_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_source": str(payload.get("raw_source", "")),
        "cached_prefix_root": str(root),
        "terminal_overlay": str(payload.get("terminal_overlay", "")),
        "terminal_padding_frames": int(payload.get("terminal_padding_frames", 0)),
        "task_order": list(task_order),
        "task_counts": task_counts,
        "split_counts": {split: len(selected[split]) for split in SPLITS},
        "selected_episode_count": sum(len(selected[split]) for split in SPLITS),
        "raw_episode_map": selected_map,
        "splits": selected,
    }
    for name in ("split_seed", "train_validation_fraction"):
        if name in payload:
            result[name] = payload[name]
    if not result["raw_source"]:
        raise ValueError("source raw-overlay manifest has no raw_source")
    if not result["terminal_overlay"] or result["terminal_padding_frames"] <= 0:
        raise ValueError("source raw-overlay manifest has no terminal overlay contract")
    return result


def _write_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cached-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_subset_manifest(args.source_manifest, args.cached_root, args.task)
    _write_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
