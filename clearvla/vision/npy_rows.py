"""Bounded, process-local range reads for immutable C-order NPY frame caches.

This reader owns physical reads only. Callers own cache admission, camera order
and logical-to-physical terminal/history indexing. No sampler or RNG is used.
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


@dataclass
class _ArrayFile:
    stream: BinaryIO
    shape: tuple[int, ...]
    dtype: np.dtype
    offset: int

    @property
    def required_size(self) -> int:
        return self.offset + math.prod(self.shape) * self.dtype.itemsize


class NpyRowReader:
    """Read exact ordered rows, merging adjacent physical reads within a call.

    POSIX uses preadv/pread. Windows uses seek/readinto under the same lock;
    this is a portable range-read implementation, never an mmap fallback.
    Handles are discarded on spawn and reopened after fork. The limit is per
    reader, per process; decoded images and tokens have independent budgets.
    """

    def __init__(self, max_open_files: int = 16) -> None:
        if type(max_open_files) is not int or max_open_files <= 0:
            raise ValueError("max_open_files must be a positive integer")
        self.max_open_files = max_open_files
        self._reset_process()

    def _reset_process(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._files: OrderedDict[Path, _ArrayFile] = OrderedDict()

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            # Do not acquire a lock inherited from a parent thread at fork.
            # Closing child descriptor copies does not close the parent's.
            for entry in self._files.values():
                entry.stream.close()
            self._reset_process()

    def __getstate__(self) -> dict[str, int]:
        return {"max_open_files": self.max_open_files}

    def __setstate__(self, state: dict[str, int]) -> None:
        self.__init__(state["max_open_files"])

    @property
    def open_file_count(self) -> int:
        self._ensure_process()
        return len(self._files)

    @property
    def io_method(self) -> str:
        if hasattr(os, "preadv"):
            return "preadv"
        if hasattr(os, "pread"):
            return "pread"
        return "locked_seek_readinto"

    def close(self) -> None:
        self._ensure_process()
        with self._lock:
            for entry in self._files.values():
                entry.stream.close()
            self._files.clear()

    def __enter__(self) -> NpyRowReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Explicit close is the observable cleanup path; interpreter
            # shutdown may already have cleared imported modules/attributes.
            pass

    def _open(self, path: Path) -> _ArrayFile:
        if path in self._files:
            self._files.move_to_end(path)
            return self._files[path]
        if len(self._files) >= self.max_open_files:
            _, previous = self._files.popitem(last=False)
            previous.stream.close()
        stream = path.open("rb", buffering=0)
        try:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY cache version {version}: {path}")
            if (
                fortran_order
                or len(shape) < 2
                or any(size <= 0 for size in shape)
                or dtype.hasobject
                or dtype.fields is not None
                or dtype.kind not in "buifc"
            ):
                raise ValueError(f"C-order nonempty numeric frame array required: {path}")
            entry = _ArrayFile(stream, tuple(shape), dtype, stream.tell())
            if os.fstat(stream.fileno()).st_size < entry.required_size:
                raise EOFError(f"truncated NPY cache: {path}")
        except BaseException:
            stream.close()
            raise
        self._files[path] = entry
        return entry

    @staticmethod
    def _read_into(entry: _ArrayFile, output: np.ndarray, offset: int) -> None:
        view = memoryview(output).cast("B")
        complete = 0
        while complete < len(view):
            try:
                if hasattr(os, "preadv"):
                    count = os.preadv(entry.stream.fileno(), [view[complete:]], offset + complete)
                elif hasattr(os, "pread"):
                    value = os.pread(entry.stream.fileno(), len(view) - complete, offset + complete)
                    count = len(value)
                    view[complete : complete + count] = value
                else:
                    entry.stream.seek(offset + complete)
                    count = entry.stream.readinto(view[complete:])
            except InterruptedError:
                continue
            if not count:
                raise EOFError("NPY cache ended during a range read")
            complete += count

    def read(
        self,
        path: str | Path,
        rows: np.ndarray,
        *,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: np.dtype | type | None = None,
    ) -> np.ndarray:
        indices = np.asarray(rows)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("rows must be a one-dimensional integer array")
        self._ensure_process()
        with self._lock:
            entry = self._open(Path(path).resolve())
            if expected_shape is not None and entry.shape != tuple(expected_shape):
                raise ValueError(f"NPY cache shape changed: {entry.shape} != {expected_shape}")
            if expected_dtype is not None and entry.dtype != np.dtype(expected_dtype):
                raise ValueError(f"NPY cache dtype changed: {entry.dtype} != {expected_dtype}")
            if os.fstat(entry.stream.fileno()).st_size < entry.required_size:
                raise EOFError(f"truncated NPY cache: {path}")
            if indices.size and (indices.min() < 0 or indices.max() >= entry.shape[0]):
                raise IndexError("physical row outside cache; resolve padding in the episode")
            unique, inverse = np.unique(indices.astype(np.int64, copy=False), return_inverse=True)
            values = np.empty((len(unique), *entry.shape[1:]), dtype=entry.dtype)
            if len(unique):
                row_bytes = math.prod(entry.shape[1:]) * entry.dtype.itemsize
                cuts = np.r_[0, np.flatnonzero(np.diff(unique) != 1) + 1, len(unique)]
                for begin, end in zip(cuts[:-1], cuts[1:]):
                    self._read_into(
                        entry,
                        values[begin:end],
                        entry.offset + int(unique[begin]) * row_bytes,
                    )
            # Advanced indexing restores repetitions/order and owns its bytes.
            return values[inverse]


__all__ = ["NpyRowReader"]
