"""Verified episode selection shared by external cache preparation tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hdf5_episode import (
    LoadedEpisode,
    load_episodes,
    resolve_too_short_episode_exclusions,
)
from .multitask_selection import (
    RDT_MULTITASK_INTERNAL_SPLITS,
    load_rdt_multitask_selection_manifest,
)
from .split import (
    RDT_SPLIT_NAMES,
    RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH,
    load_rdt_split_manifest,
)


@dataclass(frozen=True)
class CacheEpisodeSelection:
    """One source-verified cache build scope.

    ``episodes`` contains only the selected lane after the optional episode
    limit.  The manifest, when present, is nevertheless checked against the
    complete source and typed-window-eligible inventories before selection.
    """

    episodes: tuple[LoadedEpisode, ...]
    skipped: tuple[tuple[str, str], ...]
    eligible_episode_count: int
    selected_episode_count_before_limit: int
    manifest_split: str
    max_episodes: int
    manifest_metadata: dict[str, object] | None
    task_selection_metadata: dict[str, object] | None

    def report_metadata(self) -> dict[str, object]:
        return {
            "manifest_split": self.manifest_split,
            "max_episodes": self.max_episodes,
            "eligible_episode_count": self.eligible_episode_count,
            "selected_episode_count_before_limit": (
                self.selected_episode_count_before_limit
            ),
            "selected_episode_ids": [episode.episode_id for episode in self.episodes],
            "manifest": self.manifest_metadata,
            "task_selection": self.task_selection_metadata,
        }


def load_cache_episode_selection(
    root: Path,
    pattern: str,
    *,
    cameras: tuple[str, ...],
    action_key: str,
    state_key: str | None,
    camera_key_overrides: dict[str, str],
    split_manifest: Path | None,
    task_selection_manifest: Path | None = None,
    manifest_split: str,
    max_episodes: int,
    allow_skipped: bool,
) -> CacheEpisodeSelection:
    """Load a legacy inventory or a fully verified RDT manifest selection."""

    split_name = str(manifest_split)
    if split_name not in (*RDT_SPLIT_NAMES, "all"):
        raise ValueError(f"unknown manifest split {split_name!r}")
    limit = int(max_episodes)
    if limit < 0:
        raise ValueError("max episodes must be non-negative")
    if split_manifest is None and split_name != "all":
        raise ValueError("a named manifest split requires --split-manifest")
    if task_selection_manifest is not None and split_manifest is None:
        raise ValueError("a task selection requires its verified base split manifest")

    minimum_length = (
        RDT_TYPED_WINDOW_MIN_EPISODE_LENGTH if split_manifest is not None else 1
    )
    episodes, skipped = load_episodes(
        Path(root),
        str(pattern),
        cameras=cameras,
        min_length=minimum_length,
        action_key=action_key,
        state_key=state_key,
        camera_key_overrides=camera_key_overrides,
    )
    eligible_count = len(episodes)
    manifest_metadata: dict[str, object] | None = None
    task_selection_metadata: dict[str, object] | None = None
    if split_manifest is None:
        if skipped and not allow_skipped:
            raise RuntimeError(f"cache inventory has skipped episodes: {skipped[:5]}")
        selected_indices = list(range(len(episodes)))
    else:
        excluded_too_short = resolve_too_short_episode_exclusions(
            Path(root),
            skipped,
            expected_minimum_length=minimum_length,
        )
        split_indices, manifest_metadata = load_rdt_split_manifest(
            split_manifest,
            episode_names=[episode.episode_id for episode in episodes],
            expected_pattern=str(pattern),
            excluded_too_short=excluded_too_short,
            expected_minimum_episode_length=minimum_length,
        )
        if task_selection_manifest is not None:
            split_indices, task_selection_metadata = (
                load_rdt_multitask_selection_manifest(
                    task_selection_manifest,
                    episode_names=[episode.episode_id for episode in episodes],
                    task_names=[episode.task_id for episode in episodes],
                    instructions=[episode.instruction for episode in episodes],
                    base_splits=split_indices,
                    base_split_metadata=manifest_metadata,
                )
            )
            selected_indices = (
                [
                    index
                    for name in RDT_MULTITASK_INTERNAL_SPLITS
                    for index in split_indices[name]
                ]
                if split_name == "all"
                else list(split_indices[split_name])
            )
        else:
            selected_indices = (
                list(range(len(episodes)))
                if split_name == "all"
                else list(split_indices[split_name])
            )

    selected_count = len(selected_indices)
    if limit:
        selected_indices = selected_indices[:limit]
    selected = tuple(episodes[index] for index in selected_indices)
    if not selected:
        raise ValueError(
            f"cache selection {split_name!r} contains no eligible episodes"
        )
    return CacheEpisodeSelection(
        episodes=selected,
        skipped=tuple((str(path), str(reason)) for path, reason in skipped),
        eligible_episode_count=eligible_count,
        selected_episode_count_before_limit=selected_count,
        manifest_split=split_name,
        max_episodes=limit,
        manifest_metadata=manifest_metadata,
        task_selection_metadata=task_selection_metadata,
    )


__all__ = ["CacheEpisodeSelection", "load_cache_episode_selection"]
