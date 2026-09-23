#!/usr/bin/env python3
"""Compare eager HDF5 episode admission with the loader-only lazy index path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.hdf5_index import (
    HDF5RowReader,
    build_hdf5_episode_index,
    load_hdf5_episode_index,
    save_hdf5_episode_index,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--episode-names-file", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fingerprint-mode", choices=("stat", "sha256"), default="stat")
    return parser.parse_args()


def main() -> None:
    args = _args()
    episode_names = tuple(
        line.strip()
        for line in args.episode_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not episode_names:
        raise ValueError("episode-names-file is empty")
    started = time.perf_counter()
    entries = build_hdf5_episode_index(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        action_key="action",
        state_key="state",
        action_state_key="action_state",
        episode_names=episode_names,
        fingerprint_mode=args.fingerprint_mode,
    )
    index_build_seconds = time.perf_counter() - started
    save_hdf5_episode_index(
        args.index_out,
        root=args.root,
        pattern=args.pattern,
        entries=entries,
        contract={
            "reader": "hdf5_indexed_lazy_v1",
            "fingerprint_mode": args.fingerprint_mode,
            "camera_names": ["top", "wrist"],
        },
    )
    load_started = time.perf_counter()
    index_root, restored = load_hdf5_episode_index(args.index_out)
    index_load_seconds = time.perf_counter() - load_started

    eager_started = time.perf_counter()
    eager, skipped = load_episodes(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        min_length=1,
        action_key="action",
        state_key="state",
        action_state_key="action_state",
        episode_names=episode_names,
    )
    eager_seconds = time.perf_counter() - eager_started
    if skipped or len(eager) != len(restored):
        raise RuntimeError(f"eager admission mismatch: skipped={skipped[:3]}")

    rng = np.random.default_rng(args.seed)
    checks: list[dict[str, object]] = []
    lazy_started = time.perf_counter()
    with HDF5RowReader(index_root, max_open_files=16) as reader:
        for entry, episode in zip(restored, eager, strict=True):
            rows = np.sort(rng.choice(entry.action_shape[0], size=min(3, entry.action_shape[0]), replace=False))
            lazy_action = reader.read_rows(
                entry, entry.action_key, rows, require_finite=True, target_dtype=np.float32
            )
            lazy_state = reader.read_rows(
                entry,
                entry.state_key or entry.action_key,
                rows,
                require_finite=True,
                target_dtype=np.float32,
            )
            eager_action = np.asarray(episode.actions_raw)[rows]
            eager_state = np.asarray(episode.states_raw)[rows]
            if not np.array_equal(lazy_action, eager_action) or not np.array_equal(
                lazy_state, eager_state
            ):
                raise AssertionError(f"lazy/eager row mismatch for {entry.episode_id}")
            checks.append({"episode_id": entry.episode_id, "rows": rows.tolist()})
    lazy_seconds = time.perf_counter() - lazy_started

    print(
        json.dumps(
            {
                "schema": "clearvla-hdf5-indexed-lazy-probe-v1",
                "episode_count": len(restored),
                "seed": args.seed,
                "fingerprint_mode": args.fingerprint_mode,
                "identity_order_equal": [entry.episode_id for entry in restored]
                == [episode.episode_id for episode in eager],
                "row_bytes_equal": True,
                "index_build_seconds": index_build_seconds,
                "index_reload_seconds": index_load_seconds,
                "eager_admission_seconds": eager_seconds,
                "lazy_selected_rows_seconds": lazy_seconds,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
