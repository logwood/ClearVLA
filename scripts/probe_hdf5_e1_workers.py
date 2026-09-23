#!/usr/bin/env python3
"""Full-manifest HDF5 admission and worker parity gate (no model/training)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from clearvla.data.hdf5_episode import load_episodes
from clearvla.data.hdf5_index import (
    HDF5EpisodeIndexEntry,
    HDF5RowReader,
    build_hdf5_episode_index,
    load_hdf5_episode_index,
    save_hdf5_episode_index,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.hdf5")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


@dataclass(frozen=True)
class RowRef:
    entry_index: int
    episode_id: str
    row: int


class _RowDataset(Dataset[tuple[str, int, np.ndarray]]):
    def __init__(self, root: Path, entries: list[HDF5EpisodeIndexEntry], refs: list[RowRef]) -> None:
        self.root = root
        self.entries = entries
        self.refs = refs
        self._reader: HDF5RowReader | None = None

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> tuple[str, int, np.ndarray]:
        if self._reader is None:
            self._reader = HDF5RowReader(self.root, max_open_files=16)
        ref = self.refs[int(index)]
        entry = self.entries[ref.entry_index]
        value = self._reader.read_rows(
            entry, entry.action_key, [ref.row], require_finite=True, target_dtype=np.float32
        )[0]
        return ref.episode_id, ref.row, value


class _FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, indices: list[int], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batches = [
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def _collect(loader: DataLoader[tuple[str, int, np.ndarray]]) -> list[tuple[str, int, bytes]]:
    result: list[tuple[str, int, bytes]] = []
    for episodes, rows, values in loader:
        for episode, row, value in zip(episodes, rows.tolist(), values, strict=True):
            result.append((str(episode), int(row), np.asarray(value).tobytes()))
    return result


def main() -> None:
    args = _args()
    if args.sample_count <= 0 or args.batch_size <= 0:
        raise ValueError("sample-count and batch-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episode_names = tuple(str(value) for value in manifest["splits"][args.split])
    if len(set(episode_names)) != len(episode_names):
        raise ValueError("manifest contains duplicate episode identities")

    build_started = time.perf_counter()
    entries = build_hdf5_episode_index(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        state_key="state",
        action_state_key="action_state",
        episode_names=episode_names,
        fingerprint_mode="stat",
    )
    build_seconds = time.perf_counter() - build_started
    expected_ids = sorted(episode_names)
    observed_ids = [entry.episode_id for entry in entries]
    if observed_ids != expected_ids:
        raise AssertionError("full-manifest index identity order differs from eager loader")
    save_hdf5_episode_index(
        args.index_out,
        root=args.root,
        pattern=args.pattern,
        entries=entries,
        contract={
            "reader": "hdf5_indexed_lazy_v1",
            "split": args.split,
            "manifest_sha256": "recorded-by-caller",
        },
    )
    reload_started = time.perf_counter()
    _root, restored = load_hdf5_episode_index(args.index_out)
    reload_seconds = time.perf_counter() - reload_started
    if [entry.episode_id for entry in restored] != expected_ids:
        raise AssertionError("reloaded index identity order changed")

    rng = np.random.default_rng(args.seed)
    selected = np.sort(
        rng.choice(len(restored), size=min(args.sample_count, len(restored)), replace=False)
    )
    sample_names = tuple(restored[int(index)].episode_id for index in selected)
    eager_started = time.perf_counter()
    eager, skipped = load_episodes(
        args.root,
        args.pattern,
        cameras=("top", "wrist"),
        min_length=1,
        state_key="state",
        action_state_key="action_state",
        episode_names=sample_names,
    )
    eager_seconds = time.perf_counter() - eager_started
    if skipped or [episode.episode_id for episode in eager] != list(sample_names):
        raise AssertionError(f"sample eager admission mismatch: skipped={skipped[:3]}")

    refs: list[RowRef] = []
    eager_by_id = {episode.episode_id: episode for episode in eager}
    for entry_index in selected.tolist():
        entry = restored[int(entry_index)]
        episode = eager_by_id[entry.episode_id]
        rows = sorted({0, int(entry.action_shape[0] // 2), int(entry.action_shape[0] - 1)})
        with HDF5RowReader(args.root, max_open_files=16) as reader:
            lazy = reader.read_rows(
                entry,
                entry.action_key,
                rows,
                require_finite=True,
                target_dtype=np.float32,
            )
        eager_rows = np.asarray(episode.actions_raw)[rows]
        if not np.array_equal(lazy, eager_rows):
            raise AssertionError(f"sample action rows differ: {entry.episode_id}")
        refs.extend(
            RowRef(int(entry_index), entry.episode_id, int(row))
            for row in rows
        )

    dataset = _RowDataset(args.root, restored, refs)
    sampler_seed = 424242
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(sampler_seed)).tolist()
    batch_sampler = _FixedBatchSampler(order, args.batch_size)
    worker_results: dict[int, list[tuple[str, int, bytes]]] = {}
    worker_seconds: dict[int, float] = {}
    for workers in (0, 4):
        started = time.perf_counter()
        loader_kwargs: dict[str, object] = {
            "batch_sampler": batch_sampler,
            "num_workers": workers,
            "generator": torch.Generator().manual_seed(9917),
            "persistent_workers": False,
        }
        if workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(dataset, **loader_kwargs)
        worker_results[workers] = _collect(loader)
        worker_seconds[workers] = time.perf_counter() - started
    if worker_results[0] != worker_results[4]:
        raise AssertionError("worker 0/4 row identity or bytes differ")

    print(json.dumps({
        "schema": "clearvla-hdf5-e1-admission-workers-v1",
        "split": args.split,
        "manifest_episode_count": len(episode_names),
        "indexed_episode_count": len(restored),
        "sample_episode_count": len(sample_names),
        "sample_row_count": len(refs),
        "full_identity_order_equal": True,
        "sample_action_bytes_equal": True,
        "worker0_worker4_identity_bytes_equal": True,
        "sampler_seed": sampler_seed,
        "batch_size": args.batch_size,
        "build_seconds": build_seconds,
        "reload_seconds": reload_seconds,
        "eager_sample_admission_seconds": eager_seconds,
        "worker_seconds": worker_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
