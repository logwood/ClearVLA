from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from clearvla.data.hdf5_episode import LoadedEpisode, load_episodes
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import (
    DinoV2TokenStore as PreparationDinoV2TokenStore,
)
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import (
    save_episode_tokens,
)
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.data.token_store import DinoV2TokenStore
from clearvla.mainline.runtime.identity import dataset_identity
from clearvla.vision.decoded_image_store import (
    DecodedImageStore,
    build_all_decoded_caches,
)
from clearvla.vision.preprocessing import PreprocessConfig

CAMERAS = ("top", "wrist")


def _write_episode(
    path: Path,
    *,
    value: int,
    instruction: str,
    length: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action = np.full((length, 7), float(value), dtype=np.float32)
    qpos = action - 0.25
    high = np.full((length, 3, 5, 3), value, dtype=np.uint8)
    wrist = np.full((length, 3, 5, 3), value + 1, dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action)
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("instruction", data=np.bytes_(instruction))
        handle.create_dataset("observations/images/cam_high", data=high)
        handle.create_dataset("observations/images/cam_left_wrist", data=wrist + 1)
        handle.create_dataset("observations/images/cam_right_wrist", data=wrist)


def _load(root: Path, pattern: str = "**/*.hdf5") -> list[LoadedEpisode]:
    episodes, skipped = load_episodes(
        root,
        pattern,
        cameras=CAMERAS,
        min_length=1,
        action_key="action",
        state_key="qpos",
    )
    assert skipped == []
    return episodes


def _write_token_caches(
    episodes: list[LoadedEpisode],
    cache_dir: Path,
    preprocessing: PreprocessConfig,
) -> None:
    for index, episode in enumerate(episodes):
        tokens = np.full(
            (episode.length, len(CAMERAS), 3, 4),
            float(index + 1),
            dtype=np.float32,
        )
        save_episode_tokens(
            cache_dir=cache_dir,
            episode=episode,
            camera_names=CAMERAS,
            preprocessing=preprocessing,
            dinov2_model="unit-test-dino",
            tokens=tokens,
        )


def test_hierarchical_identity_closes_decoded_dino_and_run_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    _write_episode(
        root / "rdt_data" / "task_a" / "episode_0.hdf5",
        value=10,
        instruction="move object a",
    )
    _write_episode(
        root / "test" / "task_b" / "episode_0.hdf5",
        value=20,
        instruction="move object b",
    )
    episodes = _load(root)

    assert [episode.episode_id for episode in episodes] == [
        "rdt_data/task_a/episode_0",
        "test/task_b/episode_0",
    ]
    assert [episode.source_partition for episode in episodes] == ["rdt_data", "test"]
    assert [episode.task_id for episode in episodes] == ["task_a", "task_b"]
    assert [episode.stem for episode in episodes] == ["episode_0", "episode_0"]
    assert [episode.instruction for episode in episodes] == [
        "move object a",
        "move object b",
    ]

    preprocessing = PreprocessConfig()
    decoded_cache = tmp_path / "decoded"
    dino_cache = tmp_path / "dino"
    metas = build_all_decoded_caches(
        episodes,
        cache_dir=decoded_cache,
        camera_names=CAMERAS,
        preprocessing=preprocessing,
    )
    _write_token_caches(episodes, dino_cache, preprocessing)

    assert [meta.episode_stem for meta in metas] == [
        "rdt_data/task_a/episode_0",
        "test/task_b/episode_0",
    ]
    assert (decoded_cache / "rdt_data" / "task_a" / "episode_0" / "meta.json").is_file()
    assert (decoded_cache / "test" / "task_b" / "episode_0" / "meta.json").is_file()
    assert (dino_cache / "rdt_data" / "task_a" / "episode_0" / "meta.json").is_file()
    assert (dino_cache / "test" / "task_b" / "episode_0" / "meta.json").is_file()

    image_store = DecodedImageStore(
        decoded_cache,
        camera_names=CAMERAS,
        preprocessing=preprocessing,
    )
    first = image_store.load_window(episodes[0], np.asarray([0], dtype=np.int64))
    second = image_store.load_window(episodes[1], np.asarray([0], dtype=np.int64))
    assert int(first["top"][0, 0, 0, 0]) == 10
    assert int(second["top"][0, 0, 0, 0]) == 20

    token_store = DinoV2TokenStore(
        dino_cache,
        episodes=episodes,
        camera_names=CAMERAS,
        preprocessing=preprocessing,
        dinov2_model="unit-test-dino",
    )
    rows = token_store.load_batch([[0, 0], [1, 0]]).numpy()
    assert np.all(rows[0] == 1.0)
    assert np.all(rows[1] == 2.0)
    preparation_store = PreparationDinoV2TokenStore(
        dino_cache,
        episodes=episodes,
        camera_names=CAMERAS,
        preprocessing=preprocessing,
        dinov2_model="unit-test-dino",
    )
    preparation_rows = preparation_store.load_batch([[0, 0], [1, 0]]).numpy()
    assert np.array_equal(preparation_rows, rows)

    normalizer = ArrayNormalizer.fit_zscore([episode.actions_raw for episode in episodes])
    bundle = SimpleNamespace(
        episodes=tuple(episodes),
        splits={"train": (0,), "val": (1,), "test": ()},
        state_normalizer=normalizer,
        action_normalizer=normalizer,
        data_profile_metadata={"name": "identity_7d_pen"},
        split_metadata={"schema": "test"},
    )
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            raw_hdf5_root=str(root),
            hdf5_glob="**/*.hdf5",
            decoded_cache=str(decoded_cache),
            dino_cache=str(dino_cache),
        ),
    )
    identity = dataset_identity(bundle, config)  # type: ignore[arg-type]
    assert len(identity.inventory_sha256) == 64
    assert len(identity.decoded_cache_identity) == 64
    assert len(identity.dino_cache_identity) == 64


def test_flat_episode_keeps_v1_cache_layout_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    _write_episode(
        root / "episode_0.hdf5",
        value=7,
        instruction="grasp the pen",
    )
    episodes = _load(root, "*.hdf5")
    episode = episodes[0]
    assert episode.episode_id == "episode_0"
    assert episode.cache_key == episode.stem == "episode_0"
    assert episode.source_partition == ""
    assert episode.task_id == ""

    preprocessing = PreprocessConfig()
    decoded_cache = tmp_path / "decoded-flat"
    dino_cache = tmp_path / "dino-flat"
    build_all_decoded_caches(
        episodes,
        cache_dir=decoded_cache,
        camera_names=CAMERAS,
        preprocessing=preprocessing,
    )
    _write_token_caches(episodes, dino_cache, preprocessing)

    decoded_meta = json.loads(
        (decoded_cache / "episode_0" / "meta.json").read_text(encoding="utf-8")
    )
    dino_meta = json.loads(
        (dino_cache / "episode_0" / "meta.json").read_text(encoding="utf-8")
    )
    assert decoded_meta["cache_version"] == "decoded-image-v1"
    assert dino_meta["cache_version"] == "rdt2-dinov2-dense-token-v1"
    assert decoded_meta["episode_stem"] == "episode_0"
    assert dino_meta["episode_stem"] == "episode_0"

    named, skipped = load_episodes(
        root,
        "*.hdf5",
        cameras=("high", "left_wrist", "right_wrist"),
        min_length=1,
        action_key="action",
        state_key="qpos",
    )
    assert skipped == []
    assert named[0].camera_keys == {
        "high": "observations/images/cam_high",
        "left_wrist": "observations/images/cam_left_wrist",
        "right_wrist": "observations/images/cam_right_wrist",
    }
    custom, skipped = load_episodes(
        root,
        "*.hdf5",
        cameras=("overhead_rgb",),
        min_length=1,
        action_key="action",
        state_key="qpos",
        camera_key_overrides={
            "overhead_rgb": "observations/images/cam_high",
        },
    )
    assert skipped == []
    assert custom[0].camera_keys == {
        "overhead_rgb": "observations/images/cam_high"
    }


def test_suffix_removal_collision_fails_before_cache_construction(tmp_path: Path) -> None:
    root = tmp_path / "collision"
    _write_episode(root / "episode_0.h5", value=1, instruction="first")
    _write_episode(root / "episode_0.hdf5", value=2, instruction="second")

    with pytest.raises(RuntimeError, match="duplicate root-relative episode identity"):
        _load(root, "*")


def test_three_camera_cache_can_serve_an_ordered_two_camera_view(tmp_path: Path) -> None:
    root = tmp_path / "rdt-ft-data"
    _write_episode(
        root / "rdt_data" / "task" / "episode_0.hdf5",
        value=12,
        instruction="move object",
    )
    cameras = ("high", "left_wrist", "right_wrist")
    episodes, skipped = load_episodes(
        root,
        "**/*.hdf5",
        cameras=cameras,
        min_length=1,
        action_key="action",
        state_key="qpos",
    )
    assert skipped == []
    preprocessing = PreprocessConfig()
    decoded = tmp_path / "decoded-three"
    dino = tmp_path / "dino-three"
    build_all_decoded_caches(
        episodes,
        cache_dir=decoded,
        camera_names=cameras,
        preprocessing=preprocessing,
    )
    tokens = np.zeros((episodes[0].length, 3, 2, 4), dtype=np.float32)
    tokens[:, 0] = 1.0
    tokens[:, 1] = 2.0
    tokens[:, 2] = 3.0
    save_episode_tokens(
        cache_dir=dino,
        episode=episodes[0],
        camera_names=cameras,
        preprocessing=preprocessing,
        dinov2_model="unit-test-dino",
        tokens=tokens,
    )

    # The requested view is deliberately not storage order.  Both decoded
    # images and DINO tokens must reconstruct axes in caller order rather than
    # merely selecting a storage-ordered subset.
    selected = ("right_wrist", "high")
    image_store = DecodedImageStore(
        decoded,
        camera_names=selected,
        preprocessing=preprocessing,
    )
    frames = image_store.load_window(episodes[0], np.asarray([0], dtype=np.int64))
    assert tuple(frames) == selected
    assert int(frames["right_wrist"][0, 0, 0, 0]) == 13
    assert int(frames["high"][0, 0, 0, 0]) == 12
    token_store = DinoV2TokenStore(
        dino,
        episodes=episodes,
        camera_names=selected,
        preprocessing=preprocessing,
        dinov2_model="unit-test-dino",
    )
    rows = token_store.load_batch([[0, 0]]).numpy()
    assert tuple(rows.shape) == (1, 2, 2, 4)
    assert np.all(rows[:, 0] == 3.0)
    assert np.all(rows[:, 1] == 1.0)
