from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from clearvla.benchmarks.calvin import convert_calvin
from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import (
    RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
    LoadedEpisode,
    load_episode,
    load_episodes,
)
from clearvla.data.instructions import instruction_key
from clearvla.data.split import (
    EPISODE_SPLIT_MANIFEST_SCHEMA,
    load_episode_split_manifest,
    load_episode_split_manifest_inventory,
)
from clearvla.mainline.data.dataset import (
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.language import (
    CALVIN_LANGUAGE_BANK_SCHEMA,
    T5_ENCODER_ID,
    load_t5_condition_bank,
)
from clearvla.mainline.data.loading import _normalizers
from clearvla.mainline.data.normalizer import ArrayNormalizer


def _episode(path: Path, *, task: str, instruction: str) -> None:
    length = 73
    action = np.zeros((length, 7), dtype=np.float32)
    action[:, 6] = 1.0
    action_state = np.concatenate((np.zeros((1, 7), dtype=np.float32), action[:-1]), axis=0)
    state = np.full((length, 7), 3.0, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs["instruction"] = instruction
        handle.attrs["language_key"] = instruction_key(instruction)
        handle.attrs["task"] = task
        handle.attrs["valid_center_start"] = 24
        handle.attrs["valid_center_end"] = 24
        handle.create_dataset("action", data=action)
        handle.create_dataset("action_state", data=action_state)
        handle.create_dataset("state", data=state)
        images = handle.require_group("observations/images")
        images.create_dataset("cam_high", data=np.zeros((length, 2, 2, 3), dtype=np.uint8))
        images.create_dataset("cam_right_wrist", data=np.zeros((length, 2, 2, 3), dtype=np.uint8))


class _TinyImageStore:
    def validate_episode(self, _episode: LoadedEpisode) -> None:
        pass

    def load_window(
        self, _episode: LoadedEpisode, indices: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return {
            camera: torch.zeros((len(indices), 3, 2, 2), dtype=torch.uint8)
            for camera in ("top", "wrist")
        }


def _calvin_source_split(
    root: Path,
    *,
    trajectories: int,
    annotation_start_offset: int = 24,
) -> None:
    root.mkdir(parents=True)
    annotation_indices = []
    annotations = []
    tasks = []
    source_trajectories = []
    for trajectory in range(trajectories):
        source_start = trajectory * 70
        source_end = source_start + 69
        source_trajectories.append((source_start, source_end))
        annotation_indices.append(
            (source_start + annotation_start_offset, source_start + 57)
        )
        annotations.append("open the drawer")
        tasks.append("open_drawer")
        for frame in range(source_start, source_end + 1):
            local = frame - source_start
            action = np.zeros((7,), dtype=np.float32)
            action[0] = float(local) / 100.0
            action[-1] = -1.0 if local < 40 else 1.0
            robot = np.full((7,), float(local), dtype=np.float32)
            pixel = np.full((2, 2, 3), frame % 251, dtype=np.uint8)
            np.savez(
                root / f"episode_{frame:07d}.npz",
                rel_actions=action,
                robot_obs=robot,
                rgb_static=pixel,
                rgb_gripper=pixel,
            )
    np.save(
        root / "ep_start_end_ids.npy",
        np.asarray(source_trajectories, dtype=np.int64),
    )
    language = root / "lang_annotations"
    language.mkdir()
    np.save(
        language / "auto_lang_ann.npy",
        {
            "info": {"indx": np.asarray(annotation_indices, dtype=np.int64)},
            "language": {"ann": annotations, "task": tasks},
        },
        allow_pickle=True,
    )


def test_calvin_episode_filter_preserves_explicit_previous_action(tmp_path: Path) -> None:
    _episode(tmp_path / "wanted.hdf5", task="open_drawer", instruction="open the drawer")
    with h5py.File(tmp_path / "unselected_bad.hdf5", "w"):
        pass

    episodes, skipped = load_episodes(
        tmp_path,
        "*.hdf5",
        cameras=("top", "wrist"),
        min_length=73,
        action_key="action",
        action_state_key="action_state",
        state_key="state",
        episode_names=("wanted",),
    )
    assert not skipped
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.task_id == "open_drawer"
    assert episode.valid_center_start == episode.valid_center_end == 24
    assert episode.action_states_raw is not None
    assert episode.states_raw is not None
    assert not np.array_equal(episode.action_states_raw, episode.states_raw)

    projected = resolve_action_state_profile("calvin_relative_7d_v1").project_episode(episode)
    assert projected.action_states_raw is not None
    np.testing.assert_array_equal(projected.action_states_raw, episode.action_states_raw)


def test_calvin_converter_covers_terminal_actions_and_splits_source_trajectories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task_ABC_D"
    _calvin_source_split(source / "training", trajectories=3)
    _calvin_source_split(source / "validation", trajectories=2)
    output = tmp_path / "converted"

    report = convert_calvin(
        source,
        output,
        val_fraction=1.0 / 3.0,
        split_seed=7,
        task_filter="open_drawer",
    )
    assert report["episodes"] == 5
    assert report["splits"] == {"train": 2, "val": 1, "test": 2}

    split_payload = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    assert split_payload["task_filter"] == "open_drawer"
    assert split_payload["split_unit"] == "source-trajectory"
    trajectory_splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        identities = set()
        for name in split_payload["splits"][split]:
            with h5py.File(output / f"{name}.hdf5", "r") as handle:
                identities.add(str(handle.attrs["source_trajectory_id"]))
        trajectory_splits[split] = identities
    assert trajectory_splits["train"].isdisjoint(trajectory_splits["val"])
    assert trajectory_splits["train"].isdisjoint(trajectory_splits["test"])
    assert trajectory_splits["val"].isdisjoint(trajectory_splits["test"])

    name = split_payload["splits"]["train"][0]
    episode = load_episode(
        output / f"{name}.hdf5",
        cameras=("top", "wrist"),
        action_key="action",
        action_state_key="action_state",
        state_key="state",
        camera_key_overrides={
            "top": "observations/images/cam_high",
            "wrist": "observations/images/cam_right_wrist",
        },
    )
    assert episode.terminal_padding_mode == RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING
    assert episode.terminal_state_index == 57
    assert episode.valid_center_start == 24
    assert episode.valid_center_end == 33
    assert episode.length == 82

    terminal = int(episode.terminal_state_index)
    np.testing.assert_array_equal(
        episode.actions_raw[terminal:, :6],
        np.zeros_like(episode.actions_raw[terminal:, :6]),
    )
    np.testing.assert_array_equal(
        episode.actions_raw[terminal:, 6],
        np.full((episode.length - terminal,), episode.actions_raw[terminal - 1, 6]),
    )
    np.testing.assert_array_equal(
        episode.states_raw[terminal:],
        np.repeat(
            episode.states_raw[terminal : terminal + 1],
            episode.length - terminal,
            axis=0,
        ),
    )

    identity = ArrayNormalizer.fit_identity([episode.actions_raw[:terminal]])
    dataset = ObservedStateWindowDataset(
        [episode],
        [0],
        image_store=_TinyImageStore(),  # type: ignore[arg-type]
        camera_names=("top", "wrist"),
        state_normalizer=identity,
        action_normalizer=identity,
        config=ObservedStateDatasetConfig(),
    )
    assert [ref.center for ref in dataset.refs] == list(range(24, 34))
    last = dataset[len(dataset) - 1]
    np.testing.assert_array_equal(
        last["policy_action_raw"].numpy(),
        episode.actions_raw[33:57],
    )
    np.testing.assert_array_equal(
        last["policy_action_raw"][-1].numpy(),
        episode.actions_raw[terminal - 1],
    )
    np.testing.assert_array_equal(
        last["action"][24:, :6].numpy(),
        np.zeros((24, 6), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        last["action"][24:, 6].numpy(),
        np.full((24,), episode.actions_raw[terminal - 1, 6], dtype=np.float32),
    )

    action_normalizer, state_normalizer = _normalizers([episode], [0], mode="zscore")
    expected_action = ArrayNormalizer.fit_zscore([episode.actions_raw[:terminal]])
    expected_state = ArrayNormalizer.fit_zscore([episode.states_raw[: terminal + 1]])
    np.testing.assert_array_equal(action_normalizer.mean, expected_action.mean)
    np.testing.assert_array_equal(action_normalizer.std, expected_action.std)
    np.testing.assert_array_equal(state_normalizer.mean, expected_state.mean)
    np.testing.assert_array_equal(state_normalizer.std, expected_state.std)


def test_calvin_converter_filters_annotations_without_causal_history(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task_ABC_D"
    # The first source trajectory starts its annotation at local frame 18,
    # which cannot provide the required -24 history.  The other two remain
    # valid and still provide disjoint train/val trajectories.
    _calvin_source_split(source / "training", trajectories=3)
    annotation_path = source / "training" / "lang_annotations" / "auto_lang_ann.npy"
    annotation_payload = np.load(annotation_path, allow_pickle=True).item()
    annotation_indices = np.asarray(annotation_payload["info"]["indx"], dtype=np.int64)
    annotation_indices[0, 0] = 18
    annotation_payload["info"]["indx"] = annotation_indices
    np.save(annotation_path, annotation_payload, allow_pickle=True)
    _calvin_source_split(source / "validation", trajectories=2)
    report = convert_calvin(
        source,
        tmp_path / "converted",
        val_fraction=0.5,
        split_seed=0,
        task_filter="open_drawer",
    )
    assert report["episodes"] == 4
    assert report["splits"] == {"train": 1, "val": 1, "test": 2}


def test_calvin_episode_manifest_is_exact_after_task_filter(tmp_path: Path) -> None:
    manifest = tmp_path / "splits.json"
    payload = {
        "schema": EPISODE_SPLIT_MANIFEST_SCHEMA,
        "task_filter": "open_drawer",
        "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _path, loaded, inventory = load_episode_split_manifest_inventory(manifest)
    assert loaded == payload
    assert inventory == ("a", "b", "c")
    train, val, test, metadata = load_episode_split_manifest(
        manifest, episode_names=("c", "a", "b")
    )
    assert (train, val, test) == ([1], [2], [0])
    assert metadata["task_filter"] == "open_drawer"


def test_calvin_content_addressed_t5_bank_loads_fixed_policy_rows(
    tmp_path: Path,
) -> None:
    instruction = "open the drawer"
    mask = torch.tensor([[True, True, False]])
    tokens = torch.randn(1, 3, 4096, dtype=torch.float16)
    tokens[:, 2].zero_()
    path = tmp_path / "bank.pt"
    torch.save(
        {
            "schema": CALVIN_LANGUAGE_BANK_SCHEMA,
            "encoder_model": T5_ENCODER_ID,
            "storage_dtype": "float16",
            "conditions": {
                instruction_key(instruction): {
                    "instruction": instruction,
                    "tokens": tokens,
                    "mask": mask,
                }
            },
        },
        path,
    )
    bank = load_t5_condition_bank(path, max_tokens=4, expected_width=4096)
    assert bank.instructions == (instruction,)
    assert tuple(bank.tokens.shape) == (1, 4, 4096)
    assert tuple(bank.mask.shape) == (1, 4)
    assert bank.mask.tolist() == [[True, True, False, False]]
    assert torch.count_nonzero(bank.tokens[:, 2:]) == 0
