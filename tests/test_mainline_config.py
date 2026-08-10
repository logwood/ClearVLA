from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_config_loads_one_flat_active_preset() -> None:
    config = load_config(ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json")
    config.validate()
    assert config.optimizer.batch_size == 8
    assert config.dimensions.hidden_size == 512
    assert config.dimensions.goal_token_dim == 4096
    # The formal ``dinov2_cache_336`` metadata is a 16x16 token chart.  Do not
    # infer token count from the decoded 336-pixel image side.
    assert config.dimensions.patches_per_camera == 256
    assert config.observation.grid_size == 8
    assert config.observation.flow_reference_frames == 4
    assert config.data.information_uniform_fraction == 0.50
    assert config.data.information_event_fraction == 0.125
    assert config.data.information_motion_quantile == 0.70
    assert config.bottom.gripper_field_dim == 6
    assert config.bottom.physical_decode_delta_blend == 0.25
    assert config.objectives.horizon_tail_emphasis == 0.20
    assert config.objectives.horizon_first_step_protection == 0.05
    assert config.objectives.gripper_event_threshold == 0.10
    assert len(config.digest()) == 64
    assert config.digest() != config.digest(include_paths=True)


def test_mainline_config_rejects_legacy_or_unknown_switches() -> None:
    payload = ExperimentConfig().as_dict()
    payload["flow_jepa_object_intent_dynamics_mainline"] = 1
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown config sections" in str(error)
    else:
        raise AssertionError("legacy graph selectors must not enter the mainline spec")

    payload = json.loads(
        (ROOT / "configs" / "mainline" / "object_intent_dynamics_323.json").read_text(
            encoding="utf-8"
        )
    )
    payload["optimizer"] = {"batch_size": 8, "midcut_aux_loss_weight": 0.1}
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown optimizer fields" in str(error)
    else:
        raise AssertionError("inactive historical losses must not be configurable")

    payload = ExperimentConfig().as_dict()
    payload["observation"]["raw_mid_radius"] = 2
    try:
        config_from_mapping(payload)
    except ValueError as error:
        assert "unknown observation fields" in str(error)
    else:
        raise AssertionError("dead active-looking architecture fields must be rejected")


def test_mainline_config_enforces_fixed_graph_boundaries() -> None:
    config = ExperimentConfig()
    broken = replace(
        config,
        observation=replace(config.observation, microgrid_side=2),
    )
    try:
        broken.validate()
    except ValueError as error:
        assert "four 3x3 factual glimpses" in str(error)
    else:
        raise AssertionError("P1 ownership cannot be changed by a loose flag")
    broken_history = replace(
        config,
        dimensions=replace(config.dimensions, raw_pair_length=3),
    )
    try:
        broken_history.validate()
    except ValueError as error:
        assert "previous/current raw frames" in str(error)
    else:
        raise AssertionError("fixed dataset history cannot masquerade as configurable")
    broken_flow_span = replace(
        config,
        observation=replace(config.observation, flow_reference_frames=1),
    )
    try:
        broken_flow_span.validate()
    except ValueError as error:
        assert "four-frame raw pair" in str(error)
    else:
        raise AssertionError("flow temporal units cannot drift from the raw-pair dataset")
    broken_physical_field = replace(
        config,
        bottom=replace(config.bottom, gripper_field_dim=7),
    )
    try:
        broken_physical_field.validate()
    except ValueError as error:
        assert "six gripper channels" in str(error)
    else:
        raise AssertionError("the formal 18-D physical field cannot drift")


def test_config_identity_ignores_relocation_but_not_data_semantics() -> None:
    config = ExperimentConfig()
    relocated = replace(
        config,
        data=replace(
            config.data,
            raw_hdf5_root="/relocated/raw",
            decoded_cache="/relocated/decoded",
            dino_cache="/relocated/dino",
            t5_condition="/relocated/goal.pt",
            output_dir="/relocated/run",
        ),
    )
    assert config.digest() == relocated.digest()
    changed_stride = replace(config, data=replace(config.data, stride=2))
    changed_camera_key = replace(
        config,
        data=replace(config.data, top_camera_key="observations/images/other"),
    )
    changed_dtype = replace(
        config,
        runtime=replace(config.runtime, compute_dtype="fp32"),
    )
    assert config.digest() != changed_stride.digest()
    assert config.digest() != changed_camera_key.digest()
    assert config.digest() != changed_dtype.digest()
