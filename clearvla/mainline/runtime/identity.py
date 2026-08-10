"""Reproducible identities for the data and language used by one run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..checkpoint import ArtifactIdentity, DatasetIdentity
from ..config import ExperimentConfig
from ..data.loading import MainlineDataBundle


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_identity(
    bundle: MainlineDataBundle,
    config: ExperimentConfig,
) -> DatasetIdentity:
    inventory = []
    for episode in bundle.episodes:
        stat = episode.path.stat()
        inventory.append(
            {
                "stem": episode.stem,
                "length": int(episode.length),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    dino_rows = []
    dino_cache_root = Path(config.data.dino_cache)
    decoded_cache_root = Path(config.data.decoded_cache)
    decoded_rows = []
    for episode in bundle.episodes:
        dino_metadata = dino_cache_root / episode.stem / "meta.json"
        decoded_metadata = decoded_cache_root / episode.stem / "meta.json"
        if not dino_metadata.is_file():
            raise FileNotFoundError(f"DINO cache metadata disappeared: {dino_metadata}")
        if not decoded_metadata.is_file():
            raise FileNotFoundError(f"decoded-image cache metadata disappeared: {decoded_metadata}")
        dino_rows.append((episode.stem, hashlib.sha256(dino_metadata.read_bytes()).hexdigest()))
        decoded_rows.append(
            (episode.stem, hashlib.sha256(decoded_metadata.read_bytes()).hexdigest())
        )
    return DatasetIdentity(
        raw_root=str(Path(config.data.raw_hdf5_root)),
        hdf5_glob=config.data.hdf5_glob,
        inventory_sha256=_digest({"episodes": inventory, "splits": bundle.splits}),
        state_normalizer_sha256=_digest(bundle.state_normalizer.to_dict()),
        action_normalizer_sha256=_digest(bundle.action_normalizer.to_dict()),
        decoded_cache_identity=_digest(decoded_rows),
        dino_cache_identity=_digest(dino_rows),
    )


def language_identity(
    bundle: MainlineDataBundle,
    config: ExperimentConfig,
) -> ArtifactIdentity:
    if bundle.goal.metadata.get("source") == "explicit_null_goal_smoke":
        return ArtifactIdentity(
            logical_name="explicit_null_goal_smoke",
            path="<null-goal>",
            size_bytes=0,
            sha256="0" * 64,
        )
    return ArtifactIdentity.from_file("precomputed_t5_condition", config.data.t5_condition)


__all__ = ["dataset_identity", "language_identity"]
