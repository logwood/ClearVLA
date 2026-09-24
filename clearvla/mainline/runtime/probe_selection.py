"""Recorded-source separation for a bounded CALVIN validation probe.

This is experimental admission, never an online policy feature or a replacement
splitter. It checks the loaded provenance, not physical task success or the
truthfulness of arbitrary external metadata. No image/label values select rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch

from clearvla.data.hdf5_episode import LoadedEpisode

from ..data.dataset import ObservedWindowRef
from ..interfaces import TrainingBatch


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source(episode: LoadedEpisode) -> dict[str, Any]:
    # Converter-owned format: <raw split>:<trajectory index>:<start>-<end>.
    # Directory names are NOT source partitions (flat conversions have none).
    match = re.fullmatch(r"([^:]+):(\d+):(\d+)-(\d+)", episode.source_trajectory_id)
    if match is None:
        raise ValueError("missing or malformed source_trajectory_id")
    partition, index_text, lo_text, hi_text = match.groups()
    index, lo, hi = int(index_text), int(lo_text), int(hi_text)
    points = (episode.context_start, episode.source_start, episode.source_end)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in points):
        raise ValueError("missing or malformed context/start/end provenance")
    context, start, end = (int(x) for x in points if x is not None)
    if not lo <= context <= start < end <= hi:
        raise ValueError("annotation/context interval is outside its recorded source trajectory")
    return {
        "episode_id": episode.episode_id,
        "resolved_path": str(episode.path.resolve()),
        "partition": partition,
        "trajectory_index": index,
        "trajectory_start": lo,
        "trajectory_end": hi,
        "context_start": context,
        "source_start": start,
        "source_end": end,
        "annotation_id": None if episode.source_annotation_index is None else int(episode.source_annotation_index),
    }


def audit_calvin_holdout(
    episodes: Sequence[LoadedEpisode],
    splits: Mapping[str, Sequence[int]],
    normalizer_metadata: Mapping[str, object],
) -> dict[str, Any]:
    """Audit ALL fitting/validation episodes, not just the selected probe rows.

    Different annotation files may share the same trajectory. Distinct IDs with
    overlapping raw trajectory intervals are also rejected. This includes the
    full train inventory used to fit the normalizers, so checking only selected
    training windows cannot accidentally certify a contaminated validation set.
    Test membership/labels are not consulted and splits are never rewritten.
    """
    problems: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for split in records:
        ids = splits.get(split, ())
        if not ids:
            problems.append({"kind": "empty_split", "split": split})
        seen: set[int] = set()
        for raw_index in ids:
            if isinstance(raw_index, bool) or not isinstance(raw_index, Integral) or not 0 <= raw_index < len(episodes):
                problems.append({"kind": "invalid_episode_index", "split": split})
                continue
            index = int(raw_index)
            if index in seen:
                problems.append({"kind": "repeated_episode_index", "split": split, "index": index})
                continue
            seen.add(index)
            ep = episodes[index]
            if ep.data_profile != "calvin_relative_7d_v1":
                problems.append({"kind": "unsupported_profile", "episode": ep.episode_id})
                continue
            try:
                records[split].append({**_source(ep), "episode_index": index})
            except ValueError as error:
                problems.append({"kind": "unknown_source", "episode": ep.episode_id, "reason": str(error)})

    fresh = normalizer_metadata.get("source") == "fresh_train_only_fit"
    shared = normalizer_metadata.get("fit_scope") == "one_shared_normalizer_over_selected_train_split_only"
    if not (fresh or shared) or normalizer_metadata.get("train_episode_count") != len(splits.get("train", ())):
        problems.append({"kind": "unverified_train_only_normalizers"})

    # Name/path aliases and trajectory aliases are checked independently.
    for field in ("episode_id", "resolved_path"):
        owners: dict[str, str] = {}
        for split, rows in records.items():
            for row in rows:
                value = str(row[field])
                prior = owners.setdefault(value, split)
                if prior != split:
                    problems.append({"kind": "shared_" + field, "value": value})

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    for split, rows in records.items():
        for row in rows:
            key = (row["partition"], row["trajectory_index"])
            bounds = (row["trajectory_start"], row["trajectory_end"])
            if key not in groups:
                groups[key] = {"bounds": bounds, "splits": set()}
            group = groups[key]
            if bounds != group["bounds"]:
                problems.append({"kind": "inconsistent_trajectory_bounds", "trajectory": list(key)})
            group["splits"].add(split)
    for key, group in groups.items():
        if len(group["splits"]) != 1:
            problems.append({"kind": "shared_raw_trajectory", "trajectory": list(key)})

    # Sweep by raw partition; a shared boundary frame counts as overlap too.
    intervals = sorted((key[0], *group["bounds"], key[1]) for key, group in groups.items())
    active_partition = ""
    maximum_end = -1
    maximum_id = -1
    for partition, lo, hi, index in intervals:
        if partition != active_partition:
            active_partition, maximum_end, maximum_id = partition, -1, -1
        if lo <= maximum_end:
            problems.append({"kind": "overlapping_raw_trajectories", "partition": partition, "indices": [maximum_id, index]})
        if hi > maximum_end:
            maximum_end, maximum_id = hi, index
    return {
        "schema": "clearvla-calvin-probe-source-separation-v1",
        "status": "blocked" if problems else "passed",
        "scope": "declared raw trajectory separation across full train/val inventories; not content deduplication or semantic independence",
        "normalizer_scope": dict(normalizer_metadata),
        "episode_counts": {name: len(rows) for name, rows in records.items()},
        "trajectory_counts": {name: sum(name in g["splits"] for g in groups.values()) for name in records},
        "source_inventory_sha256": _digest(records),
        "problems": problems,
    }


def select_probe_indices(
    episodes: Sequence[LoadedEpisode], refs: Sequence[ObservedWindowRef], *, count: int,
    explicit: Sequence[int] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Default: one central eligible window per evenly spaced source group.

    This selection uses provenance/eligible indices only, NOT labels, reward,
    endpoint availability, contact or the model's outputs. Small batches need
    not cover all tasks or support patterns; the report exposes actual support.
    Explicit indices preserve user order and may share a training trajectory.
    """
    if type(count) is not int or count < 1:
        raise ValueError("probe count must be positive")
    if explicit is not None:
        if len(explicit) != count or len(set(explicit)) != len(explicit):
            raise ValueError("explicit probe indices must be unique and match batch-size")
        if any(type(x) is not int or not 0 <= x < len(refs) for x in explicit):
            raise ValueError("explicit probe index is outside the selected split")
        indices = list(explicit)
    else:
        groups: dict[str, list[int]] = {}
        for i, ref in enumerate(refs):
            groups.setdefault(episodes[ref.episode_idx].source_trajectory_id, []).append(i)
        keys = sorted(groups)
        if len(keys) < count:
            raise ValueError("fewer distinct source trajectories than batch-size; no repeated default rows")
        chosen = [(2 * i + 1) * len(keys) // (2 * count) for i in range(count)]
        indices = [groups[keys[i]][len(groups[keys[i]]) // 2] for i in chosen]
    records = []
    for i in indices:
        ref = refs[i]
        ep = episodes[ref.episode_idx]
        records.append({
            "dataset_index": i, "episode_index": ref.episode_idx, "episode_id": ep.episode_id,
            "source_trajectory_id": ep.source_trajectory_id, "center": ref.center,
            "boundary_region": ref.boundary_region,
        })
    return indices, records


def batch_support_report(batch: TrainingBatch) -> dict[str, Any]:
    """Detached reporting only; support is not a success label or selection rule."""
    support = batch.action_target.support
    world = batch.online.history.executed_world_window
    robot = batch.online.history.executed_robot_step
    endpoint = batch.future.annotation_endpoint
    def fraction(value: torch.Tensor | None) -> float | None:
        return None if value is None else float(value.detach().float().mean().cpu())
    return {
        "samples": batch.online.batch,
        "action_row_support": None if batch.action_target.row_valid is None else batch.action_target.row_valid.detach().float().mean(0).cpu().tolist(),
        "world_feedback_observed_fraction": fraction(None if world is None else world.observed),
        "robot_feedback_observed_fraction": fraction(None if robot is None else robot.observed),
        "endpoint_visual_observed_fraction": fraction(None if endpoint is None else endpoint.visual_observed),
        "endpoint_state_observed_fraction": fraction(None if endpoint is None else endpoint.state_observed),
        "shared_future_support_present": support is not None,
        "scope": "observed/labelled support only; not motion, contact, target progress or success",
    }
