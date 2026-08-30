from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from clearvla.tools.audit_rdt_ft_data import audit_rdt_ft_data, episode_id


def _write_episode(
    path: Path,
    *,
    instruction: str,
    depth: bool,
    offset: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = 5
    qpos = np.zeros((length, 14), dtype=np.float32)
    action = qpos.copy()
    action[:, :6] = offset + np.arange(length, dtype=np.float32)[:, None] * 0.01
    action[:, 6] = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    action[:, 7:13] = offset + np.arange(length, dtype=np.float32)[:, None] * 0.02
    action[:, 13] = np.asarray([0.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=action)
        handle.create_dataset("base_action", data=np.zeros((length, 2), dtype=np.float32))
        handle.create_dataset("instruction", data=np.bytes_(instruction))
        handle.create_dataset("observations/qpos", data=qpos)
        for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            handle.create_dataset(
                f"observations/images/{camera}",
                data=np.asarray([b"jpeg"] * length, dtype="S4"),
            )
            if depth:
                handle.create_dataset(
                    f"observations/images_depth/{camera}",
                    data=np.asarray([b"tiff"] * length, dtype="S4"),
                )


def test_hierarchical_audit_preserves_duplicate_stems_and_source_partition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rdt-ft-data"
    _write_episode(
        root / "rdt_data" / "task_a" / "episode_0.hdf5",
        instruction="move object a",
        depth=True,
        offset=0.0,
    )
    _write_episode(
        root / "rdt_data" / "task_b" / "episode_0.hdf5",
        instruction="move object b",
        depth=False,
        offset=0.1,
    )
    _write_episode(
        root / "test" / "task_a" / "episode_2.hdf5",
        instruction="move object a",
        depth=True,
        offset=0.2,
    )
    (root / "rdt_data" / "task_a" / "expanded_instruction.json").write_text(
        json.dumps({"instruction": "move object a"}),
        encoding="utf-8",
    )
    (root / "rdt_data" / "task_b" / "expanded_instruction.json").write_text(
        json.dumps({"instruction": "move object b"}),
        encoding="utf-8",
    )

    report = audit_rdt_ft_data(root)

    assert report["discovered_episodes"] == 3
    assert report["audited_episodes"] == 3
    assert report["episode_identity"]["unique"] == 3
    assert report["episode_identity"]["duplicates"] == []
    assert report["episode_identity"]["duplicate_stems"] == {"episode_0": 2}
    assert report["source_partitions"] == {"rdt_data": 2, "test": 1}
    assert report["typed_window_eligibility"] == {
        "minimum_episode_length": 73,
        "eligible_episodes": 0,
        "excluded_too_short_count": 3,
        "excluded_too_short": [
            {"episode_id": "rdt_data/task_a/episode_0", "length": 5},
            {"episode_id": "rdt_data/task_b/episode_0", "length": 5},
            {"episode_id": "test/task_a/episode_2", "length": 5},
        ],
    }
    assert report["tasks"]["count"] == 3
    assert report["action_qpos_widths"] == {"(14, 14)": 3}
    assert report["language"]["unique_original_instructions"] == 2
    assert report["language"]["tasks_with_multiple_hdf5_instructions"] == {}
    assert report["language"]["hdf5_vs_json_mismatch"] == {}
    assert report["error_count"] == 0
    assert report["base_action"] == {"maximum_abs": 0.0, "nonzero_episodes": 0}
    assert report["depth"]["availability_sets"] == {
        "()": 1,
        "('cam_high', 'cam_left_wrist', 'cam_right_wrist')": 2,
    }
    assert report["numeric"]["gripper"]["left"]["boundary_abs_delta"]["p100"] == 1.0
    assert report["numeric"]["gripper"]["right"]["boundary_abs_delta"]["p100"] == 2.0


def test_episode_id_is_root_relative_and_suffix_free(tmp_path: Path) -> None:
    root = tmp_path / "root"
    path = root / "rdt_data" / "task" / "episode_0.hdf5"
    path.parent.mkdir(parents=True)
    path.touch()
    assert episode_id(root, path) == "rdt_data/task/episode_0"
