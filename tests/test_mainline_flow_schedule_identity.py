from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.runtime.deployment import (
    CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
    DEPLOYMENT_ABI_SCHEMA,
    canonical_sha256,
    deployment_flow_schedule,
    deployment_graph_config,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from clearvla.mainline.train import _overrides, _parser, _validation_sampling_config


def _candidate_config() -> ExperimentConfig:
    base = ExperimentConfig()
    schedule = DeploymentFlowSchedule.custom(
        (0.0, 0.125, 0.25, 0.375, 0.5, 1.0),
        (0.0, 0.3, 0.55, 0.75, 0.9, 1.0),
    )
    return replace(
        base,
        runtime=replace(base.runtime, deployment_flow_schedule=schedule.to_dict()),
    )


def _abi(config: ExperimentConfig, *, legacy: bool = False) -> dict:
    graph = deployment_graph_config(config)
    abi = {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "observation": {
            "camera_names": ["top", "wrist"],
            "visual_offsets": [-8, -4, 0],
            "state_offsets": [-8, -4, 0],
            "executed_action_offsets": [-24, -16, -12, -8, -6, -4, -2, -1],
            "dinov2": {"model": "test", "compute_dtype": "fp32", "reference_batch_size": 1},
        },
        "action": {
            "data_profile": {
                "name": "identity_7d_pen",
                "gripper_transition_boundary": "current_action_state",
            },
            "gripper_output_mode": "continuous",
            "arm_flow_mode": "legacy_independent",
            "continuous_gripper_codec_boundary": "current_action_state",
            "continuous_gripper_codec_boundary_scope": CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
            "receding_horizon_execute_rows": 1,
        },
        "normalizers": {"mode": "zscore", "action_sha256": "a" * 64, "state_sha256": "b" * 64},
        "language": {"sha256": "c" * 64},
    }
    if not legacy:
        schedule = deployment_flow_schedule(config).to_dict()
        abi.update(flow_schedule=schedule, flow_schedule_sha256=canonical_sha256(schedule))
    return abi


def test_legacy_config_payload_has_no_new_defaults_and_round_trips() -> None:
    base = ExperimentConfig()
    payload = base.as_dict()
    assert "deployment_flow_schedule" not in payload["runtime"]
    restored = config_from_mapping(json.loads(json.dumps(payload)))
    assert restored.as_dict() == payload
    assert restored.digest() == base.digest()
    assert deployment_flow_schedule(restored) == DeploymentFlowSchedule.uniform_five()
    validate_deployment_abi(_abi(restored, legacy=True))


def test_explicit_schedule_changes_resume_identity_not_training_objective() -> None:
    base = ExperimentConfig()
    config = _candidate_config()
    config.validate()
    assert config.digest() != base.digest()
    assert config.bottom == base.bottom
    assert config.objectives == base.objectives
    assert config.optimizer == base.optimizer
    restored = config_from_mapping(json.loads(json.dumps(config.as_dict())))
    assert restored.digest() == config.digest()
    assert deployment_flow_schedule(restored).physical_nfe == 10
    validate_deployment_abi(_abi(restored))


def test_explicit_schedule_cannot_omit_or_forge_deployment_identity() -> None:
    config = _candidate_config()
    with pytest.raises(ValueError, match="requires deployment identity"):
        validate_deployment_abi(_abi(config, legacy=True))
    corrupt = _abi(config)
    corrupt["flow_schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schedule digest"):
        validate_deployment_abi(corrupt)
    corrupt = _abi(config)
    uniform = DeploymentFlowSchedule.uniform_five().to_dict()
    corrupt.update(flow_schedule=uniform, flow_schedule_sha256=canonical_sha256(uniform))
    with pytest.raises(ValueError, match="differs from graph runtime"):
        validate_deployment_abi(corrupt)


def test_schedule_validation_cannot_silently_add_heun_or_updates() -> None:
    payload = copy.deepcopy(_candidate_config().as_dict())
    payload["runtime"]["deployment_flow_schedule"]["method"] = "heun"
    with pytest.raises(ValueError, match="euler"):
        config_from_mapping(payload)


def test_validation_override_keeps_training_identity_untouched(tmp_path) -> None:
    base = ExperimentConfig()
    original = base.as_dict()
    schedule = deployment_flow_schedule(_candidate_config())
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(schedule.to_dict()), encoding="utf-8")
    replay = _validation_sampling_config(base, path)
    assert base.as_dict() == original
    assert replay.bottom == base.bottom and replay.objectives == base.objectives
    assert deployment_flow_schedule(replay) == schedule
    assert _validation_sampling_config(base, None) is base
    args = _parser().parse_args(["--validation-flow-schedule", str(path)])
    with pytest.raises(ValueError, match="read-only"):
        _overrides(base, args)
    args = _parser().parse_args([
        "--validate-checkpoint", "existing.pt", "--validation-flow-schedule", str(path),
    ])
    assert _overrides(base, args).as_dict() == original
