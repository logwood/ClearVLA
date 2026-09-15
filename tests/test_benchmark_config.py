from __future__ import annotations

import json
from pathlib import Path

import pytest

from clearvla.benchmarks.config import build_benchmark_config
from clearvla.mainline.config import config_from_mapping

ROOT = Path(__file__).resolve().parents[1]


def _converted_root(path: Path) -> Path:
    path.mkdir()
    (path / "splits.json").write_text(
        json.dumps(
            {
                "schema": "clearvla-episode-splits-v1",
                "splits": {"train": ["train_0"], "val": ["val_0"], "test": ["test_0"]},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_benchmark_config_binds_calvin_profile_and_manifest_contract(tmp_path: Path) -> None:
    converted = _converted_root(tmp_path / "converted")
    output = tmp_path / "config.json"
    payload = build_benchmark_config(
        ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
        output,
        hdf5_root=converted,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
        num_workers=0,
        data_profile="calvin_relative_7d_v1",
        arm_flow_mode="relative_command_adapter",
        gripper_output_mode="calvin_binary_command",
        gripper_command_weight=0.1,
        gripper_event_threshold=0.1,
    )
    config = config_from_mapping(payload)
    assert config.data.data_profile == "calvin_relative_7d_v1"
    assert config.data.split_manifest == str((converted / "splits.json").resolve())
    assert config.data.split_mode == "episode-manifest"
    assert (config.data.train_episodes, config.data.val_episodes, config.data.test_episodes) == (
        0,
        0,
        0,
    )
    assert config.bottom.gripper_output_mode == "calvin_binary_command"
    assert config.bottom.arm_flow_mode == "relative_command_adapter"
    assert config.objectives.gripper_command == 0.1
    assert output.is_file()


def test_benchmark_config_refuses_missing_split_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="split manifest"):
        build_benchmark_config(
            ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
            tmp_path / "config.json",
            hdf5_root=tmp_path / "missing",
            cache_root=tmp_path / "cache",
            language_bank=tmp_path / "language.pt",
            run_root=tmp_path / "run",
        )


def test_benchmark_config_binds_libero_retarget_overlay_paths(tmp_path: Path) -> None:
    converted = _converted_root(tmp_path / "converted")
    (converted / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema": "clearvla-external-benchmark-dataset-v1",
                "benchmark": "LIBERO",
                "converter_schema": (
                    "clearvla-libero-terminal-replay-converter-v2"
                ),
                "data_profile": "libero_relative_7d_v1",
                "arm_flow_mode": "relative_command_adapter",
                "gripper_output_mode": "continuous",
                "controller": "OSC_POSE",
                "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
                "split_unit": "episode",
                "evaluator_language_verified": True,
                "valid_center_start": 0,
                "valid_center_end": "source_action_count - 1",
                "strict_valid_center_start": 24,
                "strict_valid_center_end": "source_action_count - 49",
                "window_boundary_contract": (
                    "causal_prefix_terminal_suffix_v2"
                ),
                "observation_alignment": "causal",
                "state_normalizer_reference_semantics": "e8-compatible",
                "terminal_padding_mode": "libero-terminal-replay-absorbing-v2",
                "terminal_suffix_rows": 48,
            }
        ),
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    manifest = overlay / "overlay_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    payload = build_benchmark_config(
        ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json",
        tmp_path / "config.json",
        hdf5_root=converted,
        cache_root=tmp_path / "cache",
        language_bank=tmp_path / "language.pt",
        run_root=tmp_path / "run",
        gripper_event_threshold=0.1,
        libero_retarget_overlay_root=overlay,
    )

    assert payload["data"]["libero_retarget_overlay_root"] == str(
        overlay.resolve()
    )
    assert payload["data"]["libero_retarget_overlay_manifest"] == str(
        manifest.resolve()
    )
