from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from clearvla.benchmarks.libero_timing_probe import (
    contact_summary,
    physical_snapshot,
    summarize_timing_pair,
    suppress_close_command,
)


class _Contact:
    def __init__(self, geom1: int, geom2: int, dist: float) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class _Model:
    ngeom = 5
    geom_bodyid = np.asarray([12, 23, 27, 12, 99], dtype=np.int32)

    @staticmethod
    def body_name2id(name: str) -> int:
        return {"bowl": 23, "plate": 27}[name]

    @staticmethod
    def body_id2name(body_id: int) -> str:
        return {12: "robot0_gripper", 23: "bowl", 27: "plate", 99: "table"}.get(
            body_id, f"body_{body_id}"
        )

    @staticmethod
    def geom_id2name(geom_id: int) -> str:
        return {
            0: "finger_left",
            1: "bowl_geom",
            2: "plate_geom",
            3: "finger_right",
            4: "table_geom",
        }[geom_id]


def _env() -> SimpleNamespace:
    body_xpos = np.zeros((100, 3), dtype=np.float64)
    body_xpos[23] = [0.0, 0.0, 0.0]
    body_xpos[27] = [0.1, 0.2, 0.3]
    data = SimpleNamespace(
        ncon=3,
        contact=[_Contact(0, 1, -0.001), _Contact(1, 2, 0.0), _Contact(3, 4, 0.02)],
        body_xpos=body_xpos,
        cfrc_ext=np.zeros((100, 6), dtype=np.float64),
    )
    data.cfrc_ext[23, 0] = 2.0
    data.cfrc_ext[27, 1] = 3.0
    return SimpleNamespace(sim=SimpleNamespace(model=_Model(), data=data))


def test_suppress_close_only_changes_gripper_prefix() -> None:
    action = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.8], dtype=np.float32)
    delayed, suppressed = suppress_close_command(
        action, step_index=3, delay_steps=4, close_threshold=0.05
    )
    assert suppressed is True
    np.testing.assert_array_equal(delayed[:6], action[:6])
    assert delayed[6] == 0.0
    late, suppressed = suppress_close_command(
        action, step_index=5, delay_steps=4, close_threshold=0.05
    )
    assert suppressed is False
    np.testing.assert_array_equal(late, action)


def test_suppress_close_leaves_open_and_neutral_commands() -> None:
    for value in (-1.0, -0.05, 0.0, 0.05):
        action = np.zeros(7, dtype=np.float32)
        action[6] = value
        delayed, suppressed = suppress_close_command(
            action, step_index=1, delay_steps=10
        )
        assert suppressed is False
        np.testing.assert_array_equal(delayed, action)


def test_contact_summary_classifies_pairs_and_external_forces() -> None:
    report = contact_summary(
        _env(), target_geom_ids=[1], plate_geom_ids=[2], gripper_geom_ids=[0, 3]
    )
    assert report["ncon"] == 3
    assert report["gripper_object_contact_count"] == 1
    assert report["object_plate_contact_count"] == 1
    assert report["gripper_plate_contact_count"] == 0
    assert report["gripper_object_min_dist_m"] == pytest.approx(-0.001)
    assert report["target_cfrc_ext"][0] == pytest.approx(2.0)
    assert report["plate_cfrc_ext"][1] == pytest.approx(3.0)


def test_physical_snapshot_records_distances_and_qpos() -> None:
    env = _env()
    observation = {
        "robot0_eef_pos": np.asarray([0.2, 0.1, 0.5]),
        "robot0_gripper_qpos": np.asarray([0.1, -0.1]),
    }
    geometry = {
        "target_geom_ids": [1],
        "plate_geom_ids": [2],
        "gripper_geom_ids": [0, 3],
    }
    row = physical_snapshot(
        env,
        observation,
        target_body="bowl",
        plate_body="plate",
        geometry=geometry,
        initial_target_xyz=np.asarray([0.0, 0.0, 0.0]),
        initial_plate_xyz=np.asarray([0.0, 0.0, 0.0]),
    )
    assert row["gripper_qpos"] == pytest.approx([0.1, -0.1])
    assert row["eef_target_distance"]["xyz_m"] == pytest.approx(
        np.linalg.norm(np.asarray([0.2, 0.1, 0.5]))
    )
    assert row["target_displacement_xyz_m"] == pytest.approx([0.0, 0.0, 0.0])


def _pair_rows() -> tuple[dict[str, object], dict[str, object]]:
    base_rows = []
    delayed_rows = []
    for step in (1, 2):
        base_action = np.asarray([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)
        delayed_action = base_action.copy()
        if step == 1:
            delayed_action[6] = 0.0
        contacts_base = {
            "gripper_object_contact_count": int(step == 2),
            "object_plate_contact_count": 0,
            "gripper_object_min_dist_m": -0.001 if step == 2 else None,
            "object_plate_min_dist_m": None,
        }
        contacts_delayed = {
            "gripper_object_contact_count": 0,
            "object_plate_contact_count": 0,
            "gripper_object_min_dist_m": None,
            "object_plate_min_dist_m": None,
        }
        base_rows.append(
            {
                "step": step,
                "executed_action": base_action.tolist(),
                "post": {
                    "eef_xyz_m": [float(step), 0.0, 0.0],
                    "target_xyz_m": [0.0, 0.0, 0.0],
                    "contacts": contacts_base,
                },
            }
        )
        delayed_rows.append(
            {
                "step": step,
                "executed_action": delayed_action.tolist(),
                "close_suppressed": step == 1,
                "post": {
                    "eef_xyz_m": [float(step) + 0.1, 0.0, 0.0],
                    "target_xyz_m": [0.0, 0.0, 0.0],
                    "contacts": contacts_delayed,
                },
            }
        )
    return {"rows": base_rows, "success": False}, {"rows": delayed_rows, "success": False}


def test_summarize_timing_pair_reports_arm_identity_and_contact_shift() -> None:
    baseline, delayed = _pair_rows()
    result = summarize_timing_pair(baseline, delayed, delay_steps=1)
    assert result["arm_action_delta_max_abs"] == 0.0
    assert result["delayed_close_suppressed_count"] == 1
    assert result["first_gripper_object_contact"] == {
        "baseline_step": 2,
        "delayed_step": None,
    }
    assert result["gripper_action_delta_rms"] > 0.0
    assert result["eef_post_trajectory_delta_rms_m"] == pytest.approx(
        np.sqrt((0.1**2) / 3.0)
    )


def test_summarize_timing_pair_rejects_mismatched_lengths() -> None:
    baseline, delayed = _pair_rows()
    delayed["rows"] = delayed["rows"][:-1]
    with pytest.raises(ValueError, match="different step counts"):
        summarize_timing_pair(baseline, delayed, delay_steps=1)
