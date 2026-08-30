from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import LoadedEpisode, load_episodes
from clearvla.data.split import load_rdt_split_manifest
from clearvla.experiments.classic_policy_lab.rdt2_dinov2_cache import save_episode_tokens
from clearvla.mainline.config import load_config
from clearvla.mainline.data.language import (
    T5_INSTRUCTION_CACHE_SCHEMA,
    source_instruction_inventory_sha256,
)
from clearvla.mainline.data.loading import (
    load_mainline_data,
    load_mainline_data_for_smoke,
    to_training_batch,
)
from clearvla.tools.build_rdt_split_manifest import build_rdt_split_manifest
from clearvla.tools.build_t5_instruction_cache import build_t5_instruction_cache_payload
from clearvla.vision.preprocessing import PreprocessConfig

ROOT = Path(__file__).resolve().parents[1]


def _native_episode(tmp_path: Path) -> LoadedEpisode:
    action = np.arange(28, dtype=np.float32).reshape(2, 14)
    qpos = action + 100.0
    qpos[:, 6] = 4.7908
    qpos[:, 13] = 4.7888
    action[:, 6] = 11.8997
    action[:, 13] = 13.9231
    return LoadedEpisode(
        path=tmp_path / "episode.hdf5",
        episode_id="rdt_data/task/episode",
        source_partition="rdt_data",
        task_id="task",
        action_key="action",
        camera_keys={"high": "observations/images/cam_high"},
        actions_raw=action,
        state_key="observations/qpos",
        states_raw=qpos,
        action_states_raw=qpos,
        source_action_dim=14,
        source_state_dim=14,
    )


def test_rdt_profiles_keep_command_units_and_convert_qpos_boundary(tmp_path: Path) -> None:
    native = _native_episode(tmp_path)
    right = resolve_action_state_profile("rdt_right_arm_action_chart_v1").project_episode(
        native
    )
    assert tuple(right.actions_raw.shape) == (2, 7)
    assert right.states_raw is not None
    assert tuple(right.states_raw.shape) == (2, 7)
    assert right.actions_raw[0, -1] == pytest.approx(13.9231)
    assert right.states_raw[0, -1] == pytest.approx(4.7888)
    assert right.action_states_raw is not None
    assert right.action_states_raw[0, -1] == pytest.approx(13.9231)
    assert right.data_profile == "rdt_right_arm_action_chart_v1"
    assert right.source_action_dim == right.source_state_dim == 14

    bimanual = resolve_action_state_profile(
        "rdt_bimanual_action_chart_v1"
    ).project_episode(native)
    assert tuple(bimanual.actions_raw.shape) == (2, 14)
    assert bimanual.action_states_raw is not None
    assert bimanual.action_states_raw[0, 6] == pytest.approx(11.8997)
    assert bimanual.action_states_raw[0, 13] == pytest.approx(13.9231)
    assert resolve_action_state_profile(
        "rdt_bimanual_action_chart_v1"
    ).gripper_indices == (6, 13)


def test_rdt_loader_config_names_profile_manifest_and_camera_order() -> None:
    config = load_config(ROOT / "configs" / "mainline" / "rdt_right_arm_data_v1.json")
    assert config.data.data_profile == "rdt_right_arm_action_chart_v1"
    assert config.data.split_mode == "manifest"
    assert config.data.camera_names == ("high", "right_wrist")
    assert config.data.camera_key_map() == {
        "high": "observations/images/cam_high",
        "right_wrist": "observations/images/cam_right_wrist",
    }
    assert config.data.image_store_mode == "hdf5-direct"
    assert config.data.sampling_gripper_event_threshold is None


def test_manifest_loader_resolves_identities_and_rejects_stale_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task, count in (("a", 4), ("b", 4)):
        for index in range(count):
            path = root / "rdt_data" / task / f"episode_{index}.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("action", data=np.zeros((80, 14), dtype=np.float32))
    external = root / "test" / "external" / "episode_0.hdf5"
    external.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(external, "w") as handle:
        handle.create_dataset("action", data=np.zeros((80, 14), dtype=np.float32))

    payload = build_rdt_split_manifest(root, seed=3)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    names = sorted(
        path.relative_to(root).with_suffix("").as_posix()
        for path in root.glob("**/*.hdf5")
    )
    splits, metadata = load_rdt_split_manifest(
        manifest,
        episode_names=names,
        expected_pattern="**/*.hdf5",
    )
    assert set(splits) == {"train", "val", "test", "external_test"}
    assert sum(len(values) for values in splits.values()) == len(names)
    assert metadata["manifest_sha256"] == payload["manifest_sha256"]

    with pytest.raises(ValueError, match="episode count differs|inventory differs"):
        load_rdt_split_manifest(
            manifest,
            episode_names=names[:-1],
            expected_pattern="**/*.hdf5",
        )

    tampered = json.loads(json.dumps(payload))
    external_id = tampered["splits"]["external_test"].pop()
    tampered["splits"]["train"].append(external_id)
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            tampered,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    tampered_manifest = tmp_path / "tampered-split.json"
    tampered_manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic policy"):
        load_rdt_split_manifest(
            tampered_manifest,
            episode_names=names,
            expected_pattern="**/*.hdf5",
        )


def _write_rdt_episode(
    path: Path,
    *,
    instruction: str,
    offset: float,
    length: int = 74,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(length, dtype=np.float32)[:, None]
    action = np.zeros((length, 14), dtype=np.float32)
    qpos = np.zeros((length, 14), dtype=np.float32)
    action[:, :6] = offset + time * 0.001
    action[:, 7:13] = offset + time * 0.002
    qpos[:, :6] = action[:, :6] - 0.01
    qpos[:, 7:13] = action[:, 7:13] - 0.02
    action[:, 6] = 11.8997
    action[:, 13] = 13.9231
    qpos[:, 6] = 4.7908
    qpos[:, 13] = 4.7888
    image = np.full((length, 32, 32, 3), int(offset) % 255, dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action)
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("instruction", data=np.bytes_(instruction))
        handle.create_dataset("observations/images/cam_high", data=image)
        handle.create_dataset("observations/images/cam_left_wrist", data=image + 1)
        handle.create_dataset("observations/images/cam_right_wrist", data=image + 2)


def test_rdt_external_adapter_reaches_a_finite_typed_batch_without_a_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    for task_index, task in enumerate(("task_a", "task_b")):
        for episode_index in range(3):
            _write_rdt_episode(
                root / "rdt_data" / task / f"episode_{episode_index}.hdf5",
                instruction=f"perform {task}",
                offset=float(10 * task_index + episode_index),
            )
    _write_rdt_episode(
        root / "rdt_data" / "task_a" / "episode_short.hdf5",
        instruction="perform task_a",
        offset=4.0,
        length=30,
    )
    _write_rdt_episode(
        root / "test" / "external" / "episode_0.hdf5",
        instruction="perform external",
        offset=30.0,
    )

    manifest_payload = build_rdt_split_manifest(root, seed=5)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    all_cameras = ("high", "left_wrist", "right_wrist")
    episodes, skipped = load_episodes(
        root,
        "**/*.hdf5",
        cameras=all_cameras,
        min_length=1,
        action_key="action",
        state_key="observations/qpos",
    )
    assert skipped == []
    dino = tmp_path / "dino"
    preprocessing = PreprocessConfig(resize_hw=(336, 336), crop_hw=None)
    for episode_index, episode in enumerate(episodes):
        tokens = np.full(
            (episode.length, len(all_cameras), 64, 4),
            float(episode_index + 1),
            dtype=np.float32,
        )
        save_episode_tokens(
            cache_dir=dino,
            episode=episode,
            camera_names=all_cameras,
            preprocessing=preprocessing,
            dinov2_model="unit-test-dino",
            tokens=tokens,
        )

    instructions = tuple(sorted({str(episode.instruction) for episode in episodes}))
    t5_payload = build_t5_instruction_cache_payload(
        instructions=instructions,
        tokens=torch.arange(len(instructions) * 3 * 12, dtype=torch.float32).reshape(
            len(instructions), 3, 12
        )
        + 1,
        attention_mask=torch.ones(len(instructions), 3, dtype=torch.bool),
        model_source="unit-test/google/t5-v1_1-xxl",
        source_episode_count=len(episodes),
        source_instruction_inventory_sha256=source_instruction_inventory_sha256(
            [str(episode.instruction) for episode in episodes]
        ),
    )
    assert t5_payload["schema"] == T5_INSTRUCTION_CACHE_SCHEMA
    t5 = tmp_path / "t5.pt"
    torch.save(t5_payload, t5)

    base = load_config(ROOT / "configs" / "mainline" / "rdt_right_arm_data_v1.json")
    config = replace(
        base,
        data=replace(
            base.data,
            raw_hdf5_root=str(root),
            dino_cache=str(dino),
            t5_condition=str(t5),
            split_manifest=str(manifest),
            dinov2_model="unit-test-dino",
        ),
        dimensions=replace(
            base.dimensions,
            visual_token_dim=4,
            patches_per_camera=64,
            goal_token_dim=12,
            goal_max_tokens=3,
        ),
    )
    config.validate()
    bundle = load_mainline_data(config)
    assert set(bundle.datasets) == {"train", "val", "test", "external_test"}
    assert len(bundle.episodes) == 7
    assert len(bundle.skipped) == 1
    assert "too_short_T=30, min_length=73" in bundle.skipped[0][1]
    assert bundle.split_metadata["source_episode_count"] == 8
    assert bundle.split_metadata["episode_count"] == 7
    assert bundle.split_metadata["excluded_too_short"] == [
        {"episode_id": "rdt_data/task_a/episode_short", "length": 30}
    ]
    assert bundle.data_profile_metadata["name"] == "rdt_right_arm_action_chart_v1"
    assert bundle.gripper_event_threshold is None
    with pytest.raises(ValueError, match="no adopted gripper-event threshold"):
        bundle.loader(
            "train",
            batch_size=1,
            workers=0,
            device=torch.device("cpu"),
        )

    raw = next(
        iter(
            bundle.loader(
                "val",
                batch_size=1,
                workers=0,
                device=torch.device("cpu"),
                shuffle=False,
            )
        )
    )
    typed = to_training_batch(
        raw,
        goal=bundle.goal,
        config=config,
        device=torch.device("cpu"),
    )
    typed.validate(config)
    assert tuple(typed.online.observation.dino_history.shape) == (1, 3, 2, 64, 4)
    assert tuple(typed.action_target.normalized.shape) == (1, 24, 7)
    assert typed.action_target.current_raw_units[0, -1].item() == pytest.approx(13.9231)
    assert bool(torch.isfinite(typed.online.observation.raw_rgb).all())
    assert int(typed.online.goal.mask.sum()) == 3

    bounded_dino = tmp_path / "bounded-dino"
    selected_episode = bundle.episodes[bundle.splits["val"][0]]
    save_episode_tokens(
        cache_dir=bounded_dino,
        episode=selected_episode,
        camera_names=all_cameras,
        preprocessing=preprocessing,
        dinov2_model="unit-test-dino",
        tokens=np.ones(
            (selected_episode.length, len(all_cameras), 64, 4),
            dtype=np.float32,
        ),
    )
    bounded_config = replace(
        config,
        data=replace(config.data, dino_cache=str(bounded_dino)),
    )
    bounded = load_mainline_data_for_smoke(
        bounded_config,
        split="val",
        episode_limit=1,
    )
    assert set(bounded.datasets) == {"val"}
    bounded_raw = next(
        iter(
            bounded.loader(
                "val",
                batch_size=1,
                workers=0,
                device=torch.device("cpu"),
                shuffle=False,
            )
        )
    )
    to_training_batch(
        bounded_raw,
        goal=bounded.goal,
        config=bounded_config,
        device=torch.device("cpu"),
    ).validate(bounded_config)
    with pytest.raises(FileNotFoundError, match="missing DINO token metadata"):
        load_mainline_data(bounded_config)

    stale_payload = dict(t5_payload)
    stale_payload["source_instruction_inventory_sha256"] = "f" * 64
    stale_t5 = tmp_path / "stale-t5.pt"
    torch.save(stale_payload, stale_t5)
    stale_config = replace(
        bounded_config,
        data=replace(bounded_config.data, t5_condition=str(stale_t5)),
    )
    with pytest.raises(ValueError, match="source instruction inventory differs"):
        load_mainline_data_for_smoke(
            stale_config,
            split="val",
            episode_limit=1,
        )
