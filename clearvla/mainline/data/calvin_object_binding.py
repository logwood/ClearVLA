"""Fail-closed CALVIN semantic-role supervision sidecar.

The sidecar is a training-only join keyed by source episode identity and
causal window center.  It stores language-derived *role* targets, never a
claim that a particular anonymous K slot is red/blue/pink.  This distinction
is essential because the global object axis is explicitly relabelable.

The artifact is versioned independently from the model/checkpoint ABI.  The
previous ``v1`` fixed-slot pointer artifact is intentionally rejected rather
than reinterpreted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..calvin_binding_contract import (
    CALVIN_OBJECT_BINDING_ROLE_NAMES,
    CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA,
    CALVIN_OBJECT_BINDING_TARGET_COUNT,
)


def calvin_episode_inventory_digest(episode_ids: Iterable[str]) -> str:
    """Digest the sorted source episode identity inventory."""

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


def _role_names(value: object) -> tuple[str, ...]:
    text = _scalar_text(value, name="role_names")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("sidecar role_names must be JSON") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("sidecar role_names must be a JSON string list")
    if tuple(parsed) != tuple(CALVIN_OBJECT_BINDING_ROLE_NAMES):
        raise ValueError("sidecar role_names do not match the model role ABI")
    return tuple(str(item) for item in parsed)


@dataclass(frozen=True)
class CalvinObjectBindingSidecar:
    """Validated, immutable lookup table for one role-target artifact."""

    path: Path
    episode_ids: tuple[str, ...]
    centers: np.ndarray
    role_targets: np.ndarray
    coverage: np.ndarray
    ambiguity: np.ndarray
    source_digest: str
    manifest_digest: str
    role_names: tuple[str, ...] = CALVIN_OBJECT_BINDING_ROLE_NAMES
    schema: str = CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA
    target_policy: str = ""
    _lookup: dict[tuple[str, int], int] | None = None

    def __post_init__(self) -> None:
        rows = len(self.episode_ids)
        if self.schema != CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA:
            raise ValueError(
                f"unsupported CALVIN binding sidecar schema {self.schema!r}; "
                "the fixed-slot v1 artifact is not admissible"
            )
        if tuple(self.role_names) != tuple(CALVIN_OBJECT_BINDING_ROLE_NAMES):
            raise ValueError("sidecar role_names do not match the model role ABI")
        if self.centers.shape != (rows,) or self.centers.dtype != np.int64:
            raise ValueError("sidecar centers must be contiguous int64 [N]")
        if self.role_targets.ndim != 2 or self.role_targets.shape != (
            rows,
            CALVIN_OBJECT_BINDING_TARGET_COUNT,
        ):
            raise ValueError(
                "sidecar role_targets must be [N,R+1] with the explicit null class"
            )
        if self.role_targets.dtype != np.float32:
            raise TypeError("sidecar role_targets must be float32")
        for name, value in (("coverage", self.coverage), ("ambiguity", self.ambiguity)):
            if value.shape != (rows, 1) or value.dtype != np.float32:
                raise TypeError(f"sidecar {name} must be float32 [N,1]")
            if not bool(np.isfinite(value).all()) or bool(
                ((value < 0.0) | (value > 1.0)).any()
            ):
                raise ValueError(f"sidecar {name} must be finite and lie in [0,1]")
        if not bool(np.isfinite(self.role_targets).all()) or bool(
            (self.role_targets < 0.0).any()
        ):
            raise ValueError("sidecar role_targets must be finite and non-negative")
        sums = self.role_targets.sum(axis=1)
        if not bool(np.allclose(sums, np.ones_like(sums), atol=2e-4, rtol=2e-4)):
            raise ValueError("sidecar role_targets rows must sum to one")
        if any(not str(value).strip() for value in self.episode_ids):
            raise ValueError("sidecar episode_ids must be non-empty")
        if any(int(center) < 0 for center in self.centers.tolist()):
            raise ValueError("sidecar centers must be non-negative")
        if not str(self.source_digest).strip() or not str(self.manifest_digest).strip():
            raise ValueError("sidecar source_digest and manifest_digest are required")
        if not str(self.target_policy).strip().startswith("instruction-role-v2"):
            raise ValueError("sidecar target_policy must identify instruction-role-v2")
        keys = list(zip(self.episode_ids, self.centers.tolist(), strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError("CALVIN binding sidecar contains duplicate episode/center keys")
        object.__setattr__(self, "_lookup", {key: index for index, key in enumerate(keys)})

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
                "role_targets",
                "coverage",
                "ambiguity",
                "source_digest",
                "manifest_digest",
                "role_names",
                "target_policy",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"CALVIN role sidecar is missing fields: {missing}")
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
            role_targets = np.asarray(data["role_targets"], dtype=np.float32)
            if role_targets.shape != (rows, CALVIN_OBJECT_BINDING_TARGET_COUNT):
                raise ValueError("sidecar role_targets have the wrong shape")
            coverage = _normalise_column(data["coverage"], name="coverage", rows=rows)
            ambiguity = _normalise_column(data["ambiguity"], name="ambiguity", rows=rows)
            source_digest = _scalar_text(data["source_digest"], name="source_digest")
            manifest_digest = _scalar_text(data["manifest_digest"], name="manifest_digest")
            role_names = _role_names(data["role_names"])
            target_policy = _scalar_text(data["target_policy"], name="target_policy")
        if expected_source_digest is not None and source_digest != str(expected_source_digest):
            raise ValueError("CALVIN binding sidecar source digest does not match the dataset")
        if expected_manifest_digest is not None and manifest_digest != str(expected_manifest_digest):
            raise ValueError("CALVIN binding sidecar manifest digest does not match the dataset")
        return cls(
            path=source,
            episode_ids=episode_ids,
            centers=np.ascontiguousarray(centers),
            role_targets=np.ascontiguousarray(role_targets),
            coverage=coverage,
            ambiguity=ambiguity,
            source_digest=source_digest,
            manifest_digest=manifest_digest,
            role_names=role_names,
            schema=schema,
            target_policy=target_policy,
        )

    @property
    def rows(self) -> int:
        return len(self.episode_ids)

    def require_complete(self, keys: Iterable[tuple[str, int]]) -> None:
        assert self._lookup is not None
        missing = [
            key for key in keys if (str(key[0]), int(key[1])) not in self._lookup
        ]
        if missing:
            preview = ", ".join(f"{episode}:{center}" for episode, center in missing[:4])
            raise KeyError(
                "CALVIN binding sidecar is missing target rows for "
                f"{len(missing)} windows (first: {preview})"
            )

    def lookup(
        self, episode_id: str, center: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._lookup is not None
        row = self._lookup.get((str(episode_id), int(center)))
        if row is None:
            raise KeyError(f"CALVIN binding target missing for {episode_id!r} center={center}")
        return self.role_targets[row], self.coverage[row], self.ambiguity[row]

    def metadata(self) -> dict[str, object]:
        digest = hashlib.sha256()
        digest.update("\n".join(self.episode_ids).encode("utf-8"))
        digest.update(self.centers.tobytes())
        digest.update(self.role_targets.tobytes())
        digest.update(self.coverage.tobytes())
        digest.update(self.ambiguity.tobytes())
        return {
            "schema": self.schema,
            "path": str(self.path),
            "rows": self.rows,
            "source_digest": self.source_digest,
            "manifest_digest": self.manifest_digest,
            "target_digest": digest.hexdigest(),
            "role_names": list(self.role_names),
            "target_policy": self.target_policy,
        }


__all__ = [
    "CALVIN_OBJECT_BINDING_SIDECAR_SCHEMA",
    "CalvinObjectBindingSidecar",
    "calvin_episode_inventory_digest",
]
