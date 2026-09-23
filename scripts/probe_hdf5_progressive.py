#!/usr/bin/env python3
"""Measure demand-driven HDF5 metadata admission on a full manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.hdf5_index import HDF5RowReader
from clearvla.data.progressive_hdf5 import ProgressiveHDF5EpisodeIndex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = tuple(str(value) for value in manifest["splits"][args.split])
    rng = np.random.default_rng(args.seed)
    indices = sorted(
        int(value)
        for value in rng.choice(len(names), size=min(args.sample_count, len(names)), replace=False)
    )

    started = time.perf_counter()
    with ProgressiveHDF5EpisodeIndex(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        state_key="state",
        action_state_key="action_state",
        episode_names=names,
        max_prefetch=4,
        cache_root=args.cache_root,
    ) as index:
        constructor_seconds = time.perf_counter() - started
        first_started = time.perf_counter()
        first_entry = index.get(indices[0])
        first_metadata_seconds = time.perf_counter() - first_started
        index.prefetch(indices[1:])

        selected_names = tuple(names[index_value] for index_value in indices)
        eager_started = time.perf_counter()
        eager, skipped = load_episodes(
            args.root,
            args.pattern,
            cameras=("top", "wrist"),
            min_length=1,
            state_key="state",
            action_state_key="action_state",
            episode_names=selected_names,
        )
        eager_seconds = time.perf_counter() - eager_started
        if skipped or len(eager) != len(indices):
            raise RuntimeError(f"eager selected admission mismatch: {skipped[:3]}")
        eager_by_id = {episode.episode_id: episode for episode in eager}

        row_started = time.perf_counter()
        matched = 0
        with HDF5RowReader(args.root, max_open_files=16) as reader:
            for index_value in indices:
                entry = index.get(index_value)
                episode = eager_by_id[entry.episode_id]
                rows = np.asarray([0, entry.action_shape[0] // 2, entry.action_shape[0] - 1], dtype=np.int64)
                lazy = reader.read_rows(entry, entry.action_key, rows, target_dtype=np.float32)
                if not np.array_equal(lazy, np.asarray(episode.actions_raw)[rows]):
                    raise AssertionError(f"action row mismatch: {entry.episode_id}")
                matched += 1
        row_seconds = time.perf_counter() - row_started
        ready_after = index.ready_count

    print(json.dumps({
        "schema": "clearvla-hdf5-progressive-admission-v1",
        "split": args.split,
        "manifest_episode_count": len(names),
        "selected_episode_count": len(indices),
        "seed": args.seed,
        "constructor_seconds": constructor_seconds,
        "first_metadata_seconds": first_metadata_seconds,
        "eager_selected_admission_seconds": eager_seconds,
        "matched_row_seconds": row_seconds,
        "matched_episode_count": matched,
        "ready_records_before_close": ready_after,
        "first_episode_id": first_entry.episode_id,
        "identity_order_fixed": True,
        "action_row_bytes_equal": True,
    }, indent=2))


if __name__ == "__main__":
    main()
