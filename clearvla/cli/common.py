from __future__ import annotations

from pathlib import Path

import torch

from clearvla.data.hdf5_episode import LoadedEpisode, load_episodes
from clearvla.data.normalizer import ZScoreNormalizer
from clearvla.data.split import split_episode_ids


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(name)


def load_and_normalize_episodes(
    *,
    data_root: Path,
    pattern: str,
    cameras: tuple[str, ...],
    action_key: str,
    camera_key_overrides: dict[str, str],
    state_key: str | None = None,
    min_length: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[LoadedEpisode], list[tuple[str, str]], list[int], list[int], list[int], ZScoreNormalizer]:
    episodes, skipped = load_episodes(
        data_root,
        pattern,
        cameras=cameras,
        min_length=min_length,
        action_key=action_key,
        state_key=state_key,
        camera_key_overrides=camera_key_overrides,
    )
    train_ids, val_ids, test_ids = split_episode_ids(len(episodes), train_frac, val_frac, seed)
    normalizer = ZScoreNormalizer.fit([episodes[i].actions_raw for i in train_ids])
    for episode in episodes:
        episode.actions_norm = normalizer.encode(episode.actions_raw)
        assert episode.states_raw is not None
        episode.states_norm = normalizer.encode(episode.states_raw)
    return episodes, skipped, train_ids, val_ids, test_ids, normalizer
