from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    deployment_config_from_checkpoint,
    validate_deployment_abi,
)


def _config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(base.data, data_profile="libero_relative_7d_v1"),
        bottom=replace(base.bottom, arm_flow_mode="relative_command_direct"),
    )
    config.validate()
    return config


def _profile() -> dict[str, object]:
    profile = resolve_action_state_profile("libero_relative_7d_v1")
    return {
        **profile.as_dict(),
        "sha256": profile.digest(),
        "gripper_transition_boundary": profile.gripper_transition_boundary,
    }


def _abi() -> tuple[ExperimentConfig, dict[str, object]]:
    config = _config()
    identity = SimpleNamespace(
        validate=lambda: None,
        config_digest="a" * 64,
        manifest={"test": "manifest"},
        language=SimpleNamespace(
            logical_name="language",
            sha256="b" * 64,
            size_bytes=1,
        ),
    )
    normalizer = SimpleNamespace(
        mode="zscore",
        to_dict=lambda: {
            "mode": "zscore",
            "offset": [[0.0] * 7],
            "scale": [[1.0] * 7],
        },
    )
    value = build_deployment_abi(
        config,
        identity,
        action_normalizer=normalizer,
        state_normalizer=normalizer,
        data_profile=_profile(),
        gripper_indices=(6,),
        goal_metadata={},
    )
    return config, value


def test_libero_direct_relative_command_deployment_abi_round_trip() -> None:
    config, abi = _abi()
    action = abi["action"]
    assert isinstance(action, dict)
    assert action["arm_flow_mode"] == "relative_command_direct"
    assert action["names"] == [
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    ]
    assert action["normalized_low"] == [-1.0] * 7
    assert action["normalized_high"] == [1.0] * 7
    validate_deployment_abi(abi)
    restored = deployment_config_from_checkpoint(config.as_dict(), abi)
    assert restored.data.data_profile == "libero_relative_7d_v1"
    assert restored.bottom.arm_flow_mode == "relative_command_direct"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda action: action["data_profile"].__setitem__("sha256", "f" * 64), "digest"),
        (lambda action: action.__setitem__("gripper_indices", [5]), "gripper indices"),
        (lambda action: action.__setitem__("normalized_high", [1.0] * 6 + [2.0]), "normalized_high"),
        (lambda action: action.__setitem__("prediction_horizon", 23), "horizon"),
    ),
)
def test_libero_deployment_abi_fails_closed_on_recharting(mutation, message: str) -> None:
    _, abi = _abi()
    stale = json.loads(json.dumps(abi))
    action = stale["action"]
    assert isinstance(action, dict)
    mutation(action)
    with pytest.raises(ValueError, match=message):
        validate_deployment_abi(stale)


def test_libero_profile_refuses_legacy_arm_mode() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(base.data, data_profile="libero_relative_7d_v1"),
    )
    with pytest.raises(ValueError, match="relative_command_direct"):
        config.validate()
