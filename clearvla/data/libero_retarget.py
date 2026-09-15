"""Fail-closed contract for train-only LIBERO simulator retarget overlays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .hdf5_episode import (
    LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING,
    LoadedEpisode,
)

LIBERO_RETARGET_OVERLAY_SCHEMA = "clearvla-libero-retarget-training-overlay-v1"
LIBERO_RETARGET_EPISODE_SCHEMA = "clearvla-libero-retarget-episode-v1"
LIBERO_RETARGET_NORMALIZER_POLICY = (
    "base train split only; overlay rows are excluded from fitting"
)
LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA = (
    "clearvla-libero-terminal-replay-converter-v2"
)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, *, name: str) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _translation(value: object, *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite xyz vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class LiberoRetargetOverlayContract:
    root: Path
    manifest: Path
    episode_ids: tuple[str, ...]
    source_episode_ids: tuple[str, ...]
    file_sha256: tuple[tuple[str, str], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class LiberoRetargetOverlayMerge:
    """One base inventory with a manifest-ordered train-only overlay."""

    episodes: tuple[LoadedEpisode, ...]
    splits: dict[str, tuple[int, ...]]
    overlay_episode_indices: tuple[int, ...]
    metadata: dict[str, object]


def load_libero_retarget_overlay_contract(
    root: str | Path,
    manifest: str | Path,
    *,
    base_split_manifest: str | Path,
    base_causal_root: str | Path,
    base_train_episode_ids: Sequence[str],
) -> LiberoRetargetOverlayContract:
    """Validate one immutable paired overlay before any row reaches training."""

    overlay_root = Path(root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    split_path = Path(base_split_manifest).expanduser().resolve()
    base_root = Path(base_causal_root).expanduser().resolve()
    if not overlay_root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("LIBERO retarget overlay root/manifest is absent")
    if manifest_path.parent != overlay_root:
        raise ValueError("LIBERO retarget manifest must live at the overlay root")
    if not split_path.is_file() or not base_root.is_dir():
        raise FileNotFoundError("LIBERO retarget base root/split is absent")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("LIBERO retarget manifest is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema") != LIBERO_RETARGET_OVERLAY_SCHEMA:
        raise ValueError("LIBERO retarget overlay schema differs")
    expected_scalars = {
        "scope": "train-only simulator retarget overlay; validation/test unchanged",
        "data_profile": "libero_relative_7d_v1",
        "window_boundary_contract": "causal_prefix_terminal_suffix_v2",
        "normalizer_policy": LIBERO_RETARGET_NORMALIZER_POLICY,
    }
    for name, expected in expected_scalars.items():
        if payload.get(name) != expected:
            raise ValueError(f"LIBERO retarget overlay {name} differs")
    if payload.get("base_split_manifest_sha256") != _sha256_file(split_path):
        raise ValueError("LIBERO retarget overlay base split digest differs")
    base_manifest = base_root / "dataset_manifest.json"
    declared_base_digest = payload.get("base_dataset_manifest_sha256")
    if declared_base_digest is not None:
        if not base_manifest.is_file() or declared_base_digest != _sha256_file(base_manifest):
            raise ValueError("LIBERO retarget overlay base dataset digest differs")

    base_train = {str(value) for value in base_train_episode_ids}
    if not base_train or len(base_train) != len(tuple(base_train_episode_ids)):
        raise ValueError("base LIBERO training episode identities are invalid")
    rows = payload.get("episodes")
    pairs = payload.get("pairs")
    if not isinstance(rows, list) or not rows or not isinstance(pairs, list) or not pairs:
        raise ValueError("LIBERO retarget overlay must contain episodes and pairs")
    if payload.get("episode_count") != len(rows) or payload.get("pair_count") != len(pairs):
        raise ValueError("LIBERO retarget overlay counts differ")

    episode_ids: list[str] = []
    source_ids: list[str] = []
    row_by_episode: dict[str, Mapping[str, object]] = {}
    digests: list[tuple[str, str]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"LIBERO retarget episode row {index} is malformed")
        episode_id = _text(raw_row.get("episode_id"), name="overlay episode_id")
        source_id = _text(
            raw_row.get("source_episode_id"),
            name="overlay source_episode_id",
        )
        file_name = _text(raw_row.get("file"), name="overlay file")
        if file_name != f"{episode_id}.hdf5" or Path(file_name).name != file_name:
            raise ValueError("LIBERO retarget overlay file identity differs")
        if source_id not in base_train or raw_row.get("source_split") != "train":
            raise ValueError("LIBERO retarget overlay source is not in the base train split")
        if raw_row.get("success") is not True:
            raise ValueError("LIBERO retarget overlay contains a failed rollout")
        translation = _translation(raw_row.get("translation_m"), name="overlay translation")
        if float(np.linalg.norm(translation)) <= 0.0:
            raise ValueError("LIBERO retarget overlay contains a zero translation")
        path = overlay_root / file_name
        if not path.is_file():
            raise FileNotFoundError(f"LIBERO retarget episode is absent: {path}")
        digest = _sha256_file(path)
        if raw_row.get("file_sha256") != digest:
            raise ValueError(f"LIBERO retarget episode digest differs: {path}")
        with h5py.File(path, "r") as stream:
            if (
                _text(stream.attrs.get("converter_schema"), name="converter_schema")
                != LIBERO_TERMINAL_REPLAY_CONVERTER_SCHEMA
                or _text(stream.attrs.get("retarget_schema"), name="retarget_schema")
                != LIBERO_RETARGET_EPISODE_SCHEMA
                or _text(
                    stream.attrs.get("terminal_padding_mode"),
                    name="terminal_padding_mode",
                )
                != LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
            ):
                raise ValueError(f"LIBERO retarget HDF5 contract differs: {path}")
            if _text(
                stream.attrs.get("retarget_source_episode_id"),
                name="retarget_source_episode_id",
            ) != source_id:
                raise ValueError(f"LIBERO retarget HDF5 lineage differs: {path}")
            hdf5_translation = _translation(
                stream.attrs.get("retarget_translation_m"),
                name="HDF5 retarget translation",
            )
            if hdf5_translation != translation:
                raise ValueError(f"LIBERO retarget HDF5 translation differs: {path}")
            if bool(stream.attrs.get("retarget_success")) is not True:
                raise ValueError(f"LIBERO retarget HDF5 is not admitted: {path}")
        if episode_id in row_by_episode:
            raise ValueError("LIBERO retarget overlay repeats an episode")
        episode_ids.append(episode_id)
        source_ids.append(source_id)
        row_by_episode[episode_id] = raw_row
        digests.append((episode_id, digest))

    observed_files = {
        path.stem for path in overlay_root.glob("*.hdf5") if path.is_file()
    }
    if observed_files != set(episode_ids):
        raise ValueError("LIBERO retarget overlay HDF5 inventory differs from manifest")

    seen_sources: set[str] = set()
    paired_episode_ids: set[str] = set()
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, Mapping) or raw_pair.get("admitted") is not True:
            raise ValueError(f"LIBERO retarget pair row {index} is not admitted")
        source_id = _text(
            raw_pair.get("source_episode_id"),
            name="retarget pair source_episode_id",
        )
        raw_ids = raw_pair.get("episode_ids")
        if source_id in seen_sources or not isinstance(raw_ids, list) or len(raw_ids) != 2:
            raise ValueError("LIBERO retarget pair ownership is malformed")
        ids = tuple(str(value) for value in raw_ids)
        if len(set(ids)) != 2 or any(value not in row_by_episode for value in ids):
            raise ValueError("LIBERO retarget pair references unknown/repeated episodes")
        if any(str(row_by_episode[value]["source_episode_id"]) != source_id for value in ids):
            raise ValueError("LIBERO retarget pair crosses source demonstrations")
        translations = [
            np.asarray(row_by_episode[value]["translation_m"], dtype=np.float64)
            for value in ids
        ]
        if not np.allclose(translations[0], -translations[1], rtol=0.0, atol=1.0e-12):
            raise ValueError("LIBERO retarget pair is not symmetric")
        geometry = raw_pair.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("LIBERO retarget pair has no geometry audit")
        if (
            geometry.get("negative_response_sign_correct") is not True
            or geometry.get("positive_response_sign_correct") is not True
        ):
            raise ValueError("LIBERO retarget pair response sign failed")
        seen_sources.add(source_id)
        paired_episode_ids.update(ids)
    if paired_episode_ids != set(episode_ids):
        raise ValueError("LIBERO retarget overlay has unpaired episodes")

    metadata = {
        "schema": LIBERO_RETARGET_OVERLAY_SCHEMA,
        "path": str(manifest_path),
        "file_sha256": _sha256_file(manifest_path),
        "root": str(overlay_root),
        "episode_count": len(episode_ids),
        "pair_count": len(pairs),
        "source_episode_count": len(seen_sources),
        "source_episode_ids": sorted(seen_sources),
        "translations_m": payload.get("translations_m"),
        "controller": payload.get("controller"),
        "admission_thresholds": payload.get("admission_thresholds"),
        "normalizer_policy": LIBERO_RETARGET_NORMALIZER_POLICY,
    }
    return LiberoRetargetOverlayContract(
        root=overlay_root,
        manifest=manifest_path,
        episode_ids=tuple(episode_ids),
        source_episode_ids=tuple(source_ids),
        file_sha256=tuple(digests),
        metadata=metadata,
    )


def merge_libero_retarget_training_overlay(
    base_episodes: Sequence[LoadedEpisode],
    base_splits: Mapping[str, Sequence[int]],
    overlay_episodes: Sequence[LoadedEpisode],
    contract: LiberoRetargetOverlayContract,
) -> LiberoRetargetOverlayMerge:
    """Append an admitted overlay to train without rewriting base membership."""

    base = tuple(base_episodes)
    if not base:
        raise ValueError("LIBERO retarget merge requires base episodes")
    if set(base_splits) != {"train", "val", "test"}:
        raise ValueError("LIBERO retarget merge requires train/val/test base splits")
    normalized_splits = {
        str(name): tuple(int(value) for value in values)
        for name, values in base_splits.items()
    }
    owned_indices: set[int] = set()
    for name in ("train", "val", "test"):
        indices = normalized_splits[name]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"base LIBERO split {name!r} is empty or duplicated")
        if min(indices) < 0 or max(indices) >= len(base):
            raise ValueError(f"base LIBERO split {name!r} has an invalid episode index")
        overlap = owned_indices.intersection(indices)
        if overlap:
            raise ValueError("base LIBERO splits are not trajectory-disjoint")
        owned_indices.update(indices)
    if owned_indices != set(range(len(base))):
        raise ValueError("base LIBERO splits do not cover the episode inventory")

    base_by_id = {episode.episode_id: episode for episode in base}
    if len(base_by_id) != len(base):
        raise ValueError("base LIBERO episode identities are duplicated")
    train_ids = {
        base[index].episode_id for index in normalized_splits["train"]
    }
    declared_sources = set(contract.source_episode_ids)
    if not declared_sources or not declared_sources.issubset(train_ids):
        raise ValueError("LIBERO retarget lineage escapes the base train split")

    overlay_by_id = {episode.episode_id: episode for episode in overlay_episodes}
    if len(overlay_by_id) != len(tuple(overlay_episodes)):
        raise ValueError("LIBERO retarget loaded episodes are duplicated")
    if set(overlay_by_id) != set(contract.episode_ids):
        raise ValueError("LIBERO retarget loaded inventory differs from its manifest")
    collisions = set(overlay_by_id).intersection(base_by_id)
    if collisions:
        raise ValueError(
            "LIBERO retarget episode identity collides with the base inventory: "
            f"{sorted(collisions)[:3]}"
        )

    source_by_overlay = dict(
        zip(contract.episode_ids, contract.source_episode_ids, strict=True)
    )
    ordered_overlay: list[LoadedEpisode] = []
    for episode_id in contract.episode_ids:
        episode = overlay_by_id[episode_id]
        source = base_by_id[source_by_overlay[episode_id]]
        if episode.path.resolve().parent != contract.root:
            raise ValueError("LIBERO retarget episode path escapes its overlay root")
        for name, observed, expected in (
            ("instruction", episode.instruction, source.instruction),
            ("language_key", episode.language_key, source.language_key),
            ("task_id", episode.task_id, source.task_id),
            ("data_profile", episode.data_profile, source.data_profile),
            ("source_action_count", episode.source_action_count, source.source_action_count),
            (
                "state_normalizer_reference_semantics",
                episode.state_normalizer_reference_semantics,
                source.state_normalizer_reference_semantics,
            ),
        ):
            if observed != expected:
                raise ValueError(
                    f"LIBERO retarget {episode_id} {name} differs from its source"
                )
        if (
            episode.terminal_padding_mode
            != LIBERO_TERMINAL_REPLAY_ABSORBING_PADDING
            or episode.terminal_state_index != source.terminal_state_index
            or episode.valid_center_start != source.valid_center_start
            or episode.valid_center_end != source.valid_center_end
            or episode.strict_valid_center_start != source.strict_valid_center_start
            or episode.strict_valid_center_end != source.strict_valid_center_end
        ):
            raise ValueError(
                f"LIBERO retarget {episode_id} window/terminal contract differs"
            )
        ordered_overlay.append(episode)

    overlay_indices = tuple(
        range(len(base), len(base) + len(ordered_overlay))
    )
    merged_splits = dict(normalized_splits)
    merged_splits["train"] = normalized_splits["train"] + overlay_indices
    if (
        merged_splits["val"] != normalized_splits["val"]
        or merged_splits["test"] != normalized_splits["test"]
    ):
        raise AssertionError("LIBERO retarget merge changed validation/test membership")
    metadata = {
        **contract.metadata,
        "base_train_episode_count": len(normalized_splits["train"]),
        "overlay_episode_indices": list(overlay_indices),
        "overlay_episode_ids": list(contract.episode_ids),
        "effective_train_episode_count": len(merged_splits["train"]),
        "validation_test_membership_unchanged": True,
    }
    return LiberoRetargetOverlayMerge(
        episodes=base + tuple(ordered_overlay),
        splits=merged_splits,
        overlay_episode_indices=overlay_indices,
        metadata=metadata,
    )


__all__ = [
    "LIBERO_RETARGET_EPISODE_SCHEMA",
    "LIBERO_RETARGET_NORMALIZER_POLICY",
    "LIBERO_RETARGET_OVERLAY_SCHEMA",
    "LiberoRetargetOverlayContract",
    "LiberoRetargetOverlayMerge",
    "load_libero_retarget_overlay_contract",
    "merge_libero_retarget_training_overlay",
]
