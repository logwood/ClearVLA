"""Demand-driven HDF5 metadata cache; never a sampler or training admission.

The caller supplies a fixed manifest and already-selected episode indices.
Background readiness cannot add/remove samples or consume any random numbers.
This is a storage primitive: lengths, window population, normalizers and
sampling scores must be frozen separately before a real training loader starts.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

from .hdf5_episode import _validate_episode_id
from .hdf5_index import HDF5EpisodeIndexEntry, _entry_for_path, _validate_relative_file
from .index_contract import FingerprintMode

_CACHE_SCHEMA = "clearvla-progressive-hdf5-record-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


class ProgressiveHDF5EpisodeIndex:
    """No constructor file I/O; one source is admitted on its first demand.

    A bounded optional lookahead follows indices supplied by the caller. A
    demand steals its queued job or runs directly, so it never sits behind a
    queued scan of the manifest. HDF5's library lock may still serialize a
    currently executing background operation. Close cancels queued lookahead
    and waits for at most the one operation already running.

    Spawn/pickle starts a new process-local cache. Fork is supported only when
    no parent background job is in flight; otherwise use spawn (h5py locks
    cannot safely be inherited while active). The optional disk cache stores
    only completed records atomically; it does not certify unvisited episodes.
    """

    def __init__(
        self,
        root: Path,
        pattern: str,
        *,
        cameras: Sequence[str],
        episode_names: Sequence[str],
        action_key: str = "action",
        state_key: str | None = None,
        action_state_key: str | None = None,
        camera_key_overrides: Mapping[str, str] | None = None,
        fingerprint_mode: FingerprintMode = "stat",
        max_prefetch: int = 4,
        cache_root: Path | None = None,
    ) -> None:
        if type(max_prefetch) is not int or max_prefetch < 0:
            raise ValueError("max_prefetch must be a non-negative integer")
        if fingerprint_mode not in {"stat", "sha256"}:
            raise ValueError("unknown fingerprint mode")
        if pattern not in {"*.h5", "*.hdf5", "**/*.h5", "**/*.hdf5"}:
            raise ValueError("progressive admission requires an exact HDF5 suffix")
        identities = tuple(
            _validate_episode_id(_validate_relative_file(str(value)))
            for value in episode_names
        )
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("episode identities must be non-empty and unique")
        if not pattern.startswith("**/") and any("/" in value for value in identities):
            raise ValueError("nested episode identities require a recursive pattern")
        names = tuple(str(value) for value in cameras)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("camera names must be non-empty and unique")
        if not action_key:
            raise ValueError("action_key must be non-empty")
        for key_name, key in (("state_key", state_key), ("action_state_key", action_state_key)):
            if key is not None and not key:
                raise ValueError(f"{key_name} must be non-empty when provided")
        self.root = Path(os.path.abspath(root))  # lexical only; no resolve/stat
        self.pattern = pattern
        self.episode_names = tuple(sorted(identities))
        self.max_prefetch = max_prefetch
        self.cache_root = None if cache_root is None else Path(os.path.abspath(cache_root))
        self._options = {
            "cameras": names,
            "action_key": action_key,
            "state_key": state_key,
            "action_state_key": action_state_key,
            "camera_key_overrides": dict(camera_key_overrides or {}),
            "fingerprint_mode": fingerprint_mode,
        }
        self._contract_digest = _digest({
            "schema": _CACHE_SCHEMA, "root": str(self.root), "pattern": pattern,
            "episode_names": self.episode_names, "options": self._options,
        })
        self._reset_process()

    def _reset_process(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._records: dict[int, HDF5EpisodeIndexEntry] = {}
        self._jobs: dict[int, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def _ensure_process(self) -> None:
        if os.getpid() != self._pid:
            if any(not job.done() for job in self._jobs.values()):
                raise RuntimeError("fork with active HDF5 lookahead is unsafe; use spawn")
            # Never acquire a lock inherited from a different process.
            self._reset_process()
        if self._closed:
            raise RuntimeError("progressive index is closed")

    def __len__(self) -> int:
        return len(self.episode_names)

    def _index(self, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("episode index must be an integer")
        index = operator.index(value)
        if not 0 <= index < len(self):
            raise IndexError(f"episode index outside manifest: {index}")
        return index

    def _path(self, index: int) -> Path:
        suffix = ".hdf5" if self.pattern.endswith(".hdf5") else ".h5"
        path = self.root / (self.episode_names[index] + suffix)
        # Validate only this requested path. Never choose an alternate suffix.
        if not path.is_file():
            raise FileNotFoundError(f"requested HDF5 episode is absent: {path}")
        resolved = path.resolve(strict=True)
        resolved.relative_to(self.root.resolve(strict=True))
        return path

    def _load(self, index: int) -> HDF5EpisodeIndexEntry:
        path = self._path(index)
        cache = None
        if self.cache_root is not None:
            cache = self.cache_root / self._contract_digest / (_digest(self.episode_names[index]) + ".json")
        if cache is not None and cache.is_file():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("schema") != _CACHE_SCHEMA or payload.get("contract") != self._contract_digest:
                raise ValueError("progressive record contract mismatch")
            if payload.get("record_sha256") != _digest(payload["record"]):
                raise ValueError("progressive record checksum mismatch")
            entry = HDF5EpisodeIndexEntry.from_json(payload["record"])
            if entry.episode_id != self.episode_names[index] or entry.relative_file != path.relative_to(self.root).as_posix():
                raise ValueError("progressive record identity mismatch")
            if entry.source_fingerprint is None:
                raise ValueError("progressive record requires a source fingerprint")
            entry.source_fingerprint.validate(path)
            return entry

        entry = _entry_for_path(self.root, path, **self._options)
        if entry.episode_id != self.episode_names[index]:
            raise ValueError("source resolves to a different episode identity")
        if cache is not None:
            record = entry.to_json()
            payload = {
                "schema": _CACHE_SCHEMA, "contract": self._contract_digest,
                "record": record, "record_sha256": _digest(record),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(payload, stream, sort_keys=True, allow_nan=False)
                os.replace(temporary, cache)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        return entry

    def _completed_locked(self) -> None:
        for index, job in list(self._jobs.items()):
            if job.done() and not job.cancelled() and job.exception() is None:
                self._records[index] = job.result()
                del self._jobs[index]

    @property
    def ready_count(self) -> int:
        self._ensure_process()
        with self._lock:
            self._completed_locked()
            return len(self._records)

    @property
    def inflight_count(self) -> int:
        self._ensure_process()
        with self._lock:
            return sum(not job.done() for job in self._jobs.values())

    def prefetch(self, indices: Sequence[int]) -> tuple[int, ...]:
        """Bounded lookahead on already-selected indices; never advances RNG."""
        self._ensure_process()
        values = tuple(self._index(value) for value in indices)
        accepted: list[int] = []
        with self._lock:
            self._completed_locked()
            for index in values:
                if index in self._records or index in self._jobs:
                    continue
                if sum(not job.done() for job in self._jobs.values()) >= self.max_prefetch:
                    break
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hdf5-lookahead")
                self._jobs[index] = self._executor.submit(self._load, index)
                accepted.append(index)
        return tuple(accepted)

    def get(self, index: int) -> HDF5EpisodeIndexEntry:
        self._ensure_process()
        index = self._index(index)
        owner = False
        with self._lock:
            self._completed_locked()
            if index in self._records:
                return self._records[index]
            job = self._jobs.get(index)
            if job is None:
                # Demand-driven path: no background queue is created when the
                # caller asks for an unprefetched episode.  A Future still
                # deduplicates simultaneous requests for the same episode.
                job = Future()
                self._jobs[index] = job
                owner = True
        if owner:
            try:
                job.set_result(self._load(index))
            except BaseException as exc:
                job.set_exception(exc)
        # A queued/running prefetch is the same deterministic job; wait for it
        # rather than cancelling or duplicating HDF5 reads.
        entry = job.result()
        with self._lock:
            self._records[index] = entry
            self._jobs.pop(index, None)
        return entry

    def get_many(self, indices: Sequence[int]) -> list[HDF5EpisodeIndexEntry]:
        return [self.get(index) for index in indices]

    def close(self) -> None:
        if self._pid != os.getpid() or self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def __getstate__(self) -> dict[str, object]:
        return {
            "root": self.root, "pattern": self.pattern, "episode_names": self.episode_names,
            **self._options, "max_prefetch": self.max_prefetch, "cache_root": self.cache_root,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__(**state)

    def __enter__(self) -> "ProgressiveHDF5EpisodeIndex":
        self._ensure_process()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ProgressiveHDF5EpisodeIndex"]
