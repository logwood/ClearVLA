from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Sequence


def split_episode_ids(
    n: int, train_frac: float, val_frac: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Legacy seeded random-fraction split."""
    if n < 3:
        raise ValueError("Need at least 3 episodes for train/val/test split")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    ids = list(range(n))
    random.Random(seed).shuffle(ids)
    n_train = min(max(1, int(round(n * train_frac))), n - 2)
    n_val = min(max(1, int(round(n * val_frac))), n - n_train - 1)
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Natural ordering key: episode_2 precedes episode_10."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def split_episode_ids_ordered(
    n: int,
    *,
    train_episode_count: int,
    val_episode_count: int,
    test_episode_count: int,
    episode_names: Sequence[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic ordered-count split used by the formal 63/5/5 runs.

    Episodes are naturally sorted by name when names are supplied. Training gets
    the first N episodes. Test gets the last K episodes. Validation gets the M
    episodes immediately before the test suffix. Any middle gap is deliberately
    unused.
    """
    if n < 3:
        raise ValueError("Need at least 3 episodes for train/val/test split")
    counts = {
        "train_episode_count": int(train_episode_count),
        "val_episode_count": int(val_episode_count),
        "test_episode_count": int(test_episode_count),
    }
    invalid = {name: value for name, value in counts.items() if value <= 0}
    if invalid:
        raise ValueError(f"ordered-counts split requires positive episode counts, got {invalid}")
    requested = sum(counts.values())
    if requested > n:
        raise ValueError(
            f"ordered-counts split requests {requested} episodes but only {n} are available"
        )
    if episode_names is not None and len(episode_names) != n:
        raise ValueError(f"episode_names length {len(episode_names)} != episode count {n}")

    ids = list(range(n))
    if episode_names is not None:
        ids.sort(key=lambda index: _natural_key(str(episode_names[index])))

    train_end = counts["train_episode_count"]
    test_start = n - counts["test_episode_count"]
    val_start = test_start - counts["val_episode_count"]
    train_ids = ids[:train_end]
    val_ids = ids[val_start:test_start]
    test_ids = ids[test_start:]

    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):
        raise AssertionError("ordered-counts split produced overlapping subsets")
    return train_ids, val_ids, test_ids


def resolve_episode_ids(
    n: int,
    *,
    mode: str,
    train_frac: float,
    val_frac: float,
    seed: int,
    train_episode_count: int | None = None,
    val_episode_count: int | None = None,
    test_episode_count: int | None = None,
    episode_names: Sequence[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve the legacy random split or deterministic ordered-count split."""
    normalized_mode = {
        "random-frac": "random-fraction",
        "random-fraction": "random-fraction",
        "ordered-counts": "ordered-counts",
    }.get(mode)
    if normalized_mode is None:
        raise ValueError(f"unknown episode split mode: {mode!r}")

    counts = (train_episode_count, val_episode_count, test_episode_count)
    if normalized_mode == "random-fraction":
        if any(value not in (None, 0) for value in counts):
            raise ValueError(
                "--train-episode-count/--val-episode-count/--test-episode-count require "
                "--episode-split-mode ordered-counts"
            )
        return split_episode_ids(n, train_frac, val_frac, seed)

    if any(value is None for value in counts):
        raise ValueError(
            "ordered-counts split requires --train-episode-count, "
            "--val-episode-count, and --test-episode-count"
        )
    assert train_episode_count is not None
    assert val_episode_count is not None
    assert test_episode_count is not None
    return split_episode_ids_ordered(
        n,
        train_episode_count=int(train_episode_count),
        val_episode_count=int(val_episode_count),
        test_episode_count=int(test_episode_count),
        episode_names=episode_names,
    )


def split_partitioned_episodes_per_task(
    *,
    episode_names: Sequence[str],
    source_partitions: Sequence[str],
    task_names: Sequence[str],
    train_partition: str,
    external_test_partition: str,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[int]]:
    """Split trajectories within each training task and isolate external test.

    Tasks with fewer than three trajectories stay training-only because they
    cannot supply disjoint train/validation/test evidence.  For every task
    that enters validation or test, at least one trajectory remains in train.
    The task-local RNG prevents adding one task from perturbing every other
    task's membership.
    """

    count = len(episode_names)
    if len(source_partitions) != count or len(task_names) != count:
        raise ValueError("episode, partition, and task identity lengths must match")
    if count < 1:
        raise ValueError("partitioned split requires at least one episode")
    if not train_partition or not external_test_partition:
        raise ValueError("source partition names must be non-empty")
    if train_partition == external_test_partition:
        raise ValueError("training and external-test partitions must differ")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    if not (0.0 < float(val_frac) < 1.0 - float(train_frac)):
        raise ValueError("val_frac must be positive and leave a test remainder")
    if int(seed) < 0:
        raise ValueError("split seed must be non-negative")
    if len(set(str(value) for value in episode_names)) != count:
        raise ValueError("episode identities must be unique")

    known = {str(train_partition), str(external_test_partition)}
    observed = {str(value) for value in source_partitions}
    unknown = sorted(observed - known)
    if unknown:
        raise ValueError(f"unrecognized source partitions: {unknown}")

    grouped: dict[str, list[int]] = defaultdict(list)
    external_test: list[int] = []
    for index, (partition, task) in enumerate(
        zip(source_partitions, task_names, strict=True)
    ):
        partition = str(partition)
        task = str(task)
        if not task:
            raise ValueError(f"episode {episode_names[index]!r} has no task identity")
        if partition == external_test_partition:
            external_test.append(index)
        else:
            grouped[task].append(index)

    if not grouped:
        raise ValueError(f"training partition {train_partition!r} has no episodes")
    if not external_test:
        raise ValueError(
            f"external-test partition {external_test_partition!r} has no episodes"
        )

    result: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
        "external_test": external_test,
    }
    for task in sorted(grouped, key=_natural_key):
        indices = sorted(grouped[task], key=lambda index: _natural_key(episode_names[index]))
        if len(indices) < 3:
            result["train"].extend(indices)
            continue
        task_seed = int.from_bytes(
            hashlib.sha256(f"{int(seed)}\0{task}".encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        random.Random(task_seed).shuffle(indices)
        n_train = min(max(1, int(round(len(indices) * train_frac))), len(indices) - 2)
        n_val = min(
            max(1, int(round(len(indices) * val_frac))),
            len(indices) - n_train - 1,
        )
        result["train"].extend(indices[:n_train])
        result["val"].extend(indices[n_train : n_train + n_val])
        result["test"].extend(indices[n_train + n_val :])

    for ids in result.values():
        ids.sort(key=lambda index: _natural_key(episode_names[index]))
    flattened = [index for ids in result.values() for index in ids]
    if len(flattened) != count or len(set(flattened)) != count:
        raise AssertionError("partitioned split did not cover every episode exactly once")
    if not result["train"] or not result["val"] or not result["test"]:
        raise ValueError("per-task split requires non-empty train, validation, and test sets")
    return result


__all__ = [
    "resolve_episode_ids",
    "split_episode_ids",
    "split_episode_ids_ordered",
    "split_partitioned_episodes_per_task",
]
