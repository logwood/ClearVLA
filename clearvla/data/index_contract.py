"""Small, dataset-agnostic contracts for indexed sample readers.

The training graph should not need to know whether an episode came from HDF5,
NPY, a shard, or a remote object store.  These types keep source identity and
row requests explicit so a future backend can preserve the same ordering and
fail-closed admission rules without copying HDF5-specific code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import numpy as np

FingerprintMode = Literal["stat", "sha256"]


@dataclass(frozen=True)
class SourceFingerprint:
    """Immutable source identity used to reject stale indexes."""

    mode: FingerprintMode
    size: int
    mtime_ns: int
    sha256: str | None = None

    @classmethod
    def from_path(cls, path: Path, *, mode: FingerprintMode = "stat") -> "SourceFingerprint":
        if mode not in {"stat", "sha256"}:
            raise ValueError(f"unsupported source fingerprint mode: {mode!r}")
        stat = path.stat()
        digest = _sha256_file(path) if mode == "sha256" else None
        return cls(
            mode=mode,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            sha256=digest,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "SourceFingerprint":
        mode = str(payload.get("mode", ""))
        if mode not in {"stat", "sha256"}:
            raise ValueError(f"unsupported source fingerprint mode: {mode!r}")
        digest = payload.get("sha256")
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("source fingerprint sha256 must be a 64-character string")
        if mode == "sha256" and digest is None:
            raise ValueError("sha256 fingerprint mode requires a digest")
        return cls(
            mode=mode,
            size=int(payload["size"]),
            mtime_ns=int(payload["mtime_ns"]),
            sha256=digest,
        )

    def validate(self, path: Path) -> None:
        """Fail closed when the indexed source has changed."""

        current = SourceFingerprint.from_path(path, mode=self.mode)
        if current != self:
            raise ValueError(f"indexed source changed since admission: {path}")


@dataclass(frozen=True)
class RowRequest:
    """A backend-neutral field/row request with explicit order semantics."""

    field: str
    rows: tuple[int, ...]

    @classmethod
    def of(cls, field: str, rows: Sequence[int] | np.ndarray) -> "RowRequest":
        array = np.asarray(rows, dtype=np.int64)
        if array.ndim != 1:
            raise ValueError(f"row request must be rank-1, got shape={array.shape}")
        values = tuple(int(row) for row in array)
        if not field:
            raise ValueError("row request field cannot be empty")
        if not values:
            raise ValueError("row request must contain at least one row")
        if any(row < 0 for row in values):
            raise ValueError("row request rows must be non-negative")
        return cls(field=str(field), rows=values)


class EpisodeRowReader(Protocol):
    """Minimal storage boundary shared by HDF5, NPY and future adapters."""

    def read_rows(
        self,
        record: object,
        field: str,
        rows: Sequence[int] | np.ndarray,
        *,
        require_finite: bool = True,
        target_dtype: str | np.dtype | None = None,
    ) -> np.ndarray:
        """Return owned rows in the caller's original order."""

    def close(self) -> None:
        """Release process-local resources before worker teardown."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EpisodeRowReader", "FingerprintMode", "RowRequest", "SourceFingerprint"]
