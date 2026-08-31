"""Reproducible identities for the data and language used by one run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..checkpoint import ArtifactIdentity, DatasetIdentity
from ..config import ExperimentConfig
from ..data.loading import MainlineDataBundle
from ..data.normalizer import ArrayNormalizer


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def v120_normalizer_fingerprint(normalizer: ArrayNormalizer) -> str:
    """Return the exact short fingerprint emitted by the frozen V120 run.

    Checkpoint identity continues to own the collision-resistant SHA-256.  A
    second, explicitly named compatibility fingerprint is needed only because
    completed V120 logs serialized this rounded 12-character MD5 and no full
    digest.  Keeping the calculation here prevents an audit tool from
    pretending unlike hash algorithms are comparable.
    """

    def rounded(value: object) -> object:
        array = np.asarray(value)
        if array.dtype.kind in "fc":
            return np.round(array.astype(np.float64), 6).tolist()
        return array.tolist() if array.ndim else value

    payload = json.dumps(
        {
            key: rounded(value)
            for key, value in sorted(normalizer.to_dict().items())
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def dataset_identity(
    bundle: MainlineDataBundle,
    config: ExperimentConfig,
) -> DatasetIdentity:
    inventory = []
    for episode in bundle.episodes:
        stat = episode.path.stat()
        inventory.append(
            {
                "episode_id": episode.episode_id,
                "source_partition": episode.source_partition,
                "task_id": episode.task_id,
                "instruction_sha256": (
                    None if episode.instruction is None else _digest(episode.instruction)
                ),
                "length": int(episode.length),
                "source_action_dim": int(episode.source_action_dim),
                "source_state_dim": int(episode.source_state_dim),
                "data_profile": episode.data_profile,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    dino_rows = []
    dino_cache_root = Path(config.data.dino_cache)
    decoded_cache_root = Path(config.data.decoded_cache)
    decoded_rows = []
    for episode in bundle.episodes:
        dino_metadata = dino_cache_root / episode.cache_key / "meta.json"
        if not dino_metadata.is_file():
            raise FileNotFoundError(f"DINO cache metadata disappeared: {dino_metadata}")
        dino_rows.append(
            (episode.episode_id, hashlib.sha256(dino_metadata.read_bytes()).hexdigest())
        )
        if config.data.image_store_mode == "decoded-cache":
            decoded_metadata = decoded_cache_root / episode.cache_key / "meta.json"
            if not decoded_metadata.is_file():
                raise FileNotFoundError(
                    f"decoded-image cache metadata disappeared: {decoded_metadata}"
                )
            decoded_rows.append(
                (episode.episode_id, hashlib.sha256(decoded_metadata.read_bytes()).hexdigest())
            )
    decoded_identity = (
        _digest(decoded_rows)
        if config.data.image_store_mode == "decoded-cache"
        else _digest(
            {
                "mode": "hdf5-direct",
                "cache_side": config.data.cache_side,
                "camera_keys": config.data.camera_key_map(),
            }
        )
    )
    return DatasetIdentity(
        raw_root=str(Path(config.data.raw_hdf5_root)),
        hdf5_glob=config.data.hdf5_glob,
        inventory_sha256=_digest(
            {
                "episodes": inventory,
                "splits": bundle.splits,
                "data_profile": bundle.data_profile_metadata,
                "split_contract": {
                    name: value
                    for name, value in bundle.split_metadata.items()
                    if name not in {"path", "file_sha256"}
                },
                "normalizer_contract": {
                    name: value
                    for name, value in getattr(bundle, "normalizer_metadata", {}).items()
                    if name not in {"path", "file_sha256"}
                },
            }
        ),
        state_normalizer_sha256=_digest(bundle.state_normalizer.to_dict()),
        action_normalizer_sha256=_digest(bundle.action_normalizer.to_dict()),
        decoded_cache_identity=decoded_identity,
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
    logical_name = str(bundle.goal.metadata.get("source", "precomputed_t5_condition"))
    return ArtifactIdentity.from_file(logical_name, config.data.t5_condition)


__all__ = [
    "dataset_identity",
    "language_identity",
    "v120_normalizer_fingerprint",
]
