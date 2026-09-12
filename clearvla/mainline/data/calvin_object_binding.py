"""Fail-closed CALVIN language-to-object binding target sidecar.

The sidecar is a training-only join keyed by the source episode identity and
causal window center.  It is deliberately independent of the HDF5/DINO cache
digest: converting or relocating the raw data must not silently change the
privileged target provenance, and a missing or duplicate join is an error
before a formal loader can materialize a batch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA = "clearvla-calvin-object-binding-v1"


def calvin_episode_inventory_digest(episode_ids: Iterable[str]) -> str:
    """Digest the sorted source episode identity inventory.

    This matches the compact benchmark split convention and intentionally
    excludes local paths, cache locations and discovery order.
    """

    names = sorted(str(value) for value in episode_ids)
    if not names or len(names) != len(set(names)):
        raise ValueError("CALVIN episode inventory must be non-empty and unique")
    encoded = json.dumps(
        names,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_text(value: object, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"sidecar {name} must be a scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    text = str(scalar)
    if not text.strip():
        raise ValueError(f"sidecar {name} must be non-empty")
    return text


def _normalise_column(value: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape != (rows, 1):
        raise ValueError(f"sidecar {name} must be [{rows},1]")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class CalvinObjectBindingSidecar:
    """Validated, immutable lookup table for one CALVIN target artifact."""

    path: Path
    episode_ids: tuple[str, ...]
    centers: np.ndarray
    pointer_targets: np.ndarray
    coverage: np.ndarray
    ambiguity: np.ndarray
    source_digest: str
    manifest_digest: str
    schema: str = CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA
    _lookup: dict[tuple[str, int], int] | None = None

    def __post_init__(self) -> None:
        rows = len(self.episode_ids)
        if self.schema != CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA:
            raise ValueError(f"unsupported CALVIN binding sidecar schema {self.schema!r}")
        if self.centers.shape != (rows,) or self.centers.dtype != np.int64:
            raise ValueError("sidecar centers must be contiguous int64 [N]")
        if self.pointer_targets.ndim != 2 or self.pointer_targets.shape[0] != rows:
            raise ValueError("sidecar pointer_targets must be [N,K+1]")
        if self.pointer_targets.shape[1] != 5:
            raise ValueError("CALVIN binding sidecar requires K=4 plus null")
        if self.pointer_targets.dtype != np.float32:
            raise TypeError("sidecar pointer_targets must be float32")
        for name, value in (("coverage", self.coverage), ("ambiguity", self.ambiguity)):
            if value.shape != (rows, 1) or value.dtype != np.float32:
                raise TypeError(f"sidecar {name} must be float32 [N,1]")
            if not bool(np.isfinite(value).all()) or bool(((value < 0.0) | (value > 1.0)).any()):
                raise ValueError(f"sidecar {name} must be finite and lie in [0,1]")
        if not bool(np.isfinite(self.pointer_targets).all()) or bool((self.pointer_targets < 0.0).any()):
            raise ValueError("sidecar pointer_targets must be finite and non-negative")
        sums = self.pointer_targets.sum(axis=1)
        if not bool(np.allclose(sums, np.ones_like(sums), atol=2e-4, rtol=2e-4)):
            raise ValueError("sidecar pointer_targets rows must sum to one")
        if any(not str(value).strip() for value in self.episode_ids):
            raise ValueError("sidecar episode_ids must be non-empty")
        if any(int(center) < 0 for center in self.centers.tolist()):
            raise ValueError("sidecar centers must be non-negative")
        if not str(self.source_digest).strip() or not str(self.manifest_digest).strip():
            raise ValueError("sidecar source_digest and manifest_digest are required")
        keys = list(zip(self.episode_ids, self.centers.tolist(), strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError("CALVIN binding sidecar contains duplicate episode/center keys")
        object.__setattr__(
            self,
            "_lookup",
            {key: index for index, key in enumerate(keys)},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_source_digest: str | None = None,
        expected_manifest_digest: str | None = None,
    ) -> "CalvinObjectBindingSidecar":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"CALVIN binding sidecar does not exist: {source}")
        with np.load(source, allow_pickle=False) as data:
            required = {
                "schema",
                "episode_ids",
                "centers",
                "pointer_targets",
                "coverage",
                "ambiguity",
                "source_digest",
                "manifest_digest",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"CALVIN binding sidecar is missing fields: {missing}")
            schema = _scalar_text(data["schema"], name="schema")
            raw_ids = np.asarray(data["episode_ids"])
            if raw_ids.ndim != 1:
                raise ValueError("sidecar episode_ids must be one-dimensional")
            episode_ids = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in raw_ids.tolist()
            )
            rows = len(episode_ids)
            centers = np.asarray(data["centers"], dtype=np.int64)
            if centers.shape != (rows,):
                raise ValueError("sidecar centers must align with episode_ids")
            pointers = np.asarray(data["pointer_targets"], dtype=np.float32)
            if pointers.shape != (rows, 5):
                raise ValueError("sidecar pointer_targets must be [N,5]")
            coverage = _normalise_column(data["coverage"], name="coverage", rows=rows)
            ambiguity = _normalise_column(data["ambiguity"], name="ambiguity", rows=rows)
            source_digest = _scalar_text(data["source_digest"], name="source_digest")
            manifest_digest = _scalar_text(data["manifest_digest"], name="manifest_digest")
        if expected_source_digest is not None and source_digest != str(expected_source_digest):
            raise ValueError("CALVIN binding sidecar source digest does not match the dataset")
        if expected_manifest_digest is not None and manifest_digest != str(expected_manifest_digest):
            raise ValueError("CALVIN binding sidecar manifest digest does not match the dataset")
        return cls(
            path=source,
            episode_ids=episode_ids,
            centers=np.ascontiguousarray(centers),
            pointer_targets=np.ascontiguousarray(pointers),
            coverage=coverage,
            ambiguity=ambiguity,
            source_digest=source_digest,
            manifest_digest=manifest_digest,
            schema=schema,
        )

    @property
    def rows(self) -> int:
        return len(self.episode_ids)

    def require_complete(self, keys: Iterable[tuple[str, int]]) -> None:
        assert self._lookup is not None
        index = self._lookup
        missing = [key for key in keys if (str(key[0]), int(key[1])) not in index]
        if missing:
            preview = ", ".join(f"{episode}:{center}" for episode, center in missing[:4])
            raise KeyError(
                "CALVIN binding sidecar is missing target rows for "
                f"{len(missing)} windows (first: {preview})"
            )

    def lookup(self, episode_id: str, center: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._lookup is not None
        row = self._lookup.get((str(episode_id), int(center)))
        if row is None:
            raise KeyError(f"CALVIN binding target missing for {episode_id!r} center={center}")
        return self.pointer_targets[row], self.coverage[row], self.ambiguity[row]

    def metadata(self) -> dict[str, object]:
        digest = hashlib.sha256()
        digest.update("\n".join(self.episode_ids).encode("utf-8"))
        digest.update(self.centers.tobytes())
        digest.update(self.pointer_targets.tobytes())
        digest.update(self.coverage.tobytes())
        digest.update(self.ambiguity.tobytes())
        return {
            "schema": self.schema,
            "path": str(self.path),
            "rows": self.rows,
            "source_digest": self.source_digest,
            "manifest_digest": self.manifest_digest,
            "target_digest": digest.hexdigest(),
        }


__all__ = [
    "CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA",
    "CalvinObjectBindingSidecar",
    "calvin_episode_inventory_digest",
]
