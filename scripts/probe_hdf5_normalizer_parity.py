#!/usr/bin/env python3
"""Loader-only HDF5 parity gate for normalizers and episode metadata.

This deliberately stops before image stores, model code, or a DataLoader.  It
checks the part that must be identical before an indexed reader can replace
eager episode admission: identity order, native action/state bytes, terminal
metadata, and the serialized normalizer result.
"""

from __future__ import annotations

import argparse
import hashlib
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
from clearvla.mainline.data.normalizer import ArrayNormalizer


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--episode-names-file", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--fingerprint-mode", choices=("stat", "sha256"), default="stat")
    return parser.parse_args()


def _digest(normalizer: ArrayNormalizer) -> str:
    payload = json.dumps(
        normalizer.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = _args()
    episode_names = tuple(
        line.strip()
        for line in args.episode_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not episode_names:
        raise ValueError("episode-names-file is empty")

    index_started = time.perf_counter()
    entries = build_hdf5_episode_index(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        state_key="state",
        action_state_key="action_state",
        episode_names=episode_names,
        fingerprint_mode=args.fingerprint_mode,
    )
    save_hdf5_episode_index(
        args.index_out,
        root=args.root,
        pattern=args.pattern,
        entries=entries,
        contract={
            "reader": "hdf5_indexed_lazy_v1",
            "fingerprint_mode": args.fingerprint_mode,
            "normalizer": "clearvla-zscore-v1",
        },
    )
    _root, restored = load_hdf5_episode_index(
        args.index_out,
        expected_contract={
            "reader": "hdf5_indexed_lazy_v1",
            "normalizer": "clearvla-zscore-v1",
        },
    )
    index_seconds = time.perf_counter() - index_started

    eager_started = time.perf_counter()
    eager, skipped = load_episodes(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        min_length=1,
        state_key="state",
        action_state_key="action_state",
        episode_names=episode_names,
    )
    eager_seconds = time.perf_counter() - eager_started
    if skipped or len(eager) != len(restored):
        raise RuntimeError(f"eager admission mismatch: skipped={skipped[:3]}")

    lazy_started = time.perf_counter()
    lazy_actions: list[np.ndarray] = []
    lazy_states: list[np.ndarray] = []
    lazy_action_states: list[np.ndarray] = []
    with HDF5RowReader(args.root, max_open_files=16) as reader:
        for entry, episode in zip(restored, eager, strict=True):
            if entry.episode_id != episode.episode_id:
                raise AssertionError(f"identity mismatch: {entry.episode_id} != {episode.episode_id}")
            rows = np.arange(entry.action_shape[0], dtype=np.int64)
            action = reader.read_rows(
                entry, entry.action_key, rows, require_finite=True, target_dtype=np.float32
            )
            state = reader.read_rows(
                entry, entry.state_key or entry.action_key, rows,
                require_finite=True, target_dtype=np.float32,
            )
            action_state = reader.read_rows(
                entry,
                entry.action_state_key or entry.state_key or entry.action_key,
                rows,
                require_finite=True,
                target_dtype=np.float32,
            )
            if not np.array_equal(action, np.asarray(episode.actions_raw)):
                raise AssertionError(f"action bytes differ: {entry.episode_id}")
            if not np.array_equal(state, np.asarray(episode.states_raw)):
                raise AssertionError(f"state bytes differ: {entry.episode_id}")
            if not np.array_equal(action_state, np.asarray(episode.action_states_raw)):
                raise AssertionError(f"action_state bytes differ: {entry.episode_id}")
            if entry.metadata.get("terminal_state_index") != episode.terminal_state_index:
                raise AssertionError(f"terminal metadata differs: {entry.episode_id}")
            lazy_actions.append(action)
            lazy_states.append(state)
            lazy_action_states.append(action_state)
    lazy_seconds = time.perf_counter() - lazy_started

    eager_action_norm = ArrayNormalizer.fit_zscore([np.asarray(ep.actions_raw) for ep in eager])
    eager_state_norm = ArrayNormalizer.fit_zscore([np.asarray(ep.states_raw) for ep in eager])
    lazy_action_norm = ArrayNormalizer.fit_zscore(lazy_actions)
    lazy_state_norm = ArrayNormalizer.fit_zscore(lazy_states)
    action_digest_equal = _digest(eager_action_norm) == _digest(lazy_action_norm)
    state_digest_equal = _digest(eager_state_norm) == _digest(lazy_state_norm)
    if not action_digest_equal or not state_digest_equal:
        raise AssertionError("normalizer digest differs between eager and lazy rows")

    print(json.dumps({
        "schema": "clearvla-hdf5-normalizer-parity-v1",
        "episode_count": len(restored),
        "fingerprint_mode": args.fingerprint_mode,
        "identity_order_equal": [entry.episode_id for entry in restored]
        == [episode.episode_id for episode in eager],
        "action_state_bytes_equal": True,
        "action_normalizer_digest_equal": action_digest_equal,
        "state_normalizer_digest_equal": state_digest_equal,
        "index_and_admission_seconds": index_seconds,
        "eager_admission_seconds": eager_seconds,
        "lazy_rows_seconds": lazy_seconds,
        "action_normalizer_sha256": _digest(lazy_action_norm),
        "state_normalizer_sha256": _digest(lazy_state_norm),
    }, indent=2))


if __name__ == "__main__":
    main()
