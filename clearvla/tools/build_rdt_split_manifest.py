"""Build a deterministic source-partition-aware RDT split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import h5py

from clearvla.data.hdf5_episode import episode_identity, find_hdf5_files
from clearvla.data.split import (
    RDT_SPLIT_MANIFEST_SCHEMA,
    RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
    split_partitioned_episodes_per_task,
)

SPLIT_MANIFEST_SCHEMA = RDT_SPLIT_MANIFEST_SCHEMA


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rdt_split_manifest(
    root: Path,
    *,
    pattern: str = "**/*.hdf5",
    train_partition: str = "rdt_data",
    external_test_partition: str = "test",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    minimum_episode_length: int = RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
) -> dict[str, Any]:
    source = Path(root).expanduser().resolve()
    paths = find_hdf5_files(source, pattern)
    minimum_length = int(minimum_episode_length)
    if minimum_length <= 0:
        raise ValueError("minimum episode length must be positive")
    source_episode_ids: list[str] = []
    source_partitions: list[str] = []
    episode_ids: list[str] = []
    partitions: list[str] = []
    tasks: list[str] = []
    excluded_too_short: list[dict[str, int | str]] = []
    for path in paths:
        identity, partition, task = episode_identity(source, path)
        if len(PurePosixPath(identity).parts) < 3:
            raise ValueError(f"RDT episode is not partition/task/episode: {identity!r}")
        source_episode_ids.append(identity)
        source_partitions.append(partition)
        with h5py.File(path, "r") as handle:
            action = handle.get("action")
            if not isinstance(action, h5py.Dataset) or action.ndim != 2:
                raise ValueError(f"{path}: action must be an HDF5 [T,D] dataset")
            length = int(action.shape[0])
        if length <= 0:
            raise ValueError(f"{path}: action sequence is empty")
        if length < minimum_length:
            excluded_too_short.append({"episode_id": identity, "length": length})
            continue
        episode_ids.append(identity)
        partitions.append(partition)
        tasks.append(task)
    known_partitions = {str(train_partition), str(external_test_partition)}
    unknown_partitions = sorted(set(source_partitions) - known_partitions)
    if unknown_partitions:
        raise ValueError(f"unrecognized source partitions: {unknown_partitions}")
    splits = split_partitioned_episodes_per_task(
        episode_names=episode_ids,
        source_partitions=partitions,
        task_names=tasks,
        train_partition=train_partition,
        external_test_partition=external_test_partition,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )
    split_ids = {
        name: [episode_ids[index] for index in indices]
        for name, indices in splits.items()
    }
    train_tasks = {tasks[index] for index in splits["train"]}
    val_tasks = {tasks[index] for index in splits["val"]}
    test_tasks = {tasks[index] for index in splits["test"]}
    if not val_tasks.issubset(train_tasks) or not test_tasks.issubset(train_tasks):
        raise AssertionError("known-task evaluation contains a task absent from training")
    payload: dict[str, Any] = {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "pattern": str(pattern),
        "minimum_episode_length": minimum_length,
        "policy": {
            "mode": "per-task-episodes",
            "train_partition": str(train_partition),
            "external_test_partition": str(external_test_partition),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "test_frac": round(float(1.0 - train_frac - val_frac), 12),
            "seed": int(seed),
            "tasks_with_fewer_than_three_episodes": "training-only",
        },
        "source_episode_inventory_sha256": _digest(source_episode_ids),
        "source_episode_count": len(source_episode_ids),
        "episode_inventory_sha256": _digest(episode_ids),
        "episode_count": len(episode_ids),
        "excluded_too_short": excluded_too_short,
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "task_counts": {
            "train": len(train_tasks),
            "val": len(val_tasks),
            "test": len(test_tasks),
            "external_test": len({tasks[index] for index in splits["external_test"]}),
        },
        "splits": split_ids,
    }
    payload["manifest_sha256"] = _digest(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic known-task RDT episode split manifest"
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", default="**/*.hdf5")
    parser.add_argument("--train-partition", default="rdt_data")
    parser.add_argument("--external-test-partition", default="test")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--minimum-episode-length",
        type=int,
        default=RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_rdt_split_manifest(
        args.root,
        pattern=args.glob,
        train_partition=args.train_partition,
        external_test_partition=args.external_test_partition,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        minimum_episode_length=args.minimum_episode_length,
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        destination = args.output.expanduser().resolve()
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite split manifest: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"wrote {destination} ({payload['manifest_sha256']})")


if __name__ == "__main__":
    main()


__all__ = ["SPLIT_MANIFEST_SCHEMA", "build_rdt_split_manifest"]
