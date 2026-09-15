from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.gripper_contract import MANISKILL_BINARY_GRIPPER_OUTPUT_MODE
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.components import OutletAdapter
from clearvla.mainline.runtime.deployment import (
    CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
    DEPLOYMENT_ABI_SCHEMA,
    canonical_sha256,
    deployment_graph_config,
    validate_deployment_abi,
)
from clearvla.mainline.v120_core.time_domain_mmdit import TerminalActionController


def _maniskill_binary_config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        data=replace(
            base.data,
            data_profile="maniskill_pd_ee_delta_pose_7d_v2",
            split_mode="episode-manifest",
            split_manifest="stackcube-splits.json",
            train_episodes=0,
            val_episodes=0,
            test_episodes=0,
            sampling_gripper_event_threshold=0.1,
        ),
        bottom=replace(
            base.bottom,
            arm_flow_mode="relative_command_adapter",
            gripper_output_mode=MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
        ),
        objectives=replace(
            base.objectives,
            gripper_command=0.1,
            gripper_event_threshold=0.1,
        ),
    )
    config.validate()
    return config


def test_maniskill_v2_binary_mode_selects_its_own_component_pair() -> None:
    config = _maniskill_binary_config()
    selection = ComponentSelection.from_config(config)
    assert selection.terminal_controller == "maniskill_binary_command_v1"
    assert selection.outlet_adapter == "maniskill_7d_binary_v2"


def _binary_deployment_abi() -> dict[str, object]:
    config = _maniskill_binary_config()
    profile = resolve_action_state_profile(config.data.data_profile)
    graph = deployment_graph_config(config)
    return {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "observation": {
            "camera_names": ["top", "wrist"],
            "visual_offsets": [-8, -4, 0],
            "state_offsets": [-8, -4, 0],
            "executed_action_offsets": [-24, -16, -12, -8, -6, -4, -2, -1],
            "state_dim": 7,
            "action_dim": 7,
            "dinov2": {
                "model": "test",
                "compute_dtype": "fp32",
                "reference_batch_size": 1,
            },
        },
        "action": {
            "data_profile": {
                **profile.as_dict(),
                "sha256": profile.digest(),
                "gripper_transition_boundary": "previous_command",
            },
            "gripper_indices": [6],
            "gripper_output_mode": MANISKILL_BINARY_GRIPPER_OUTPUT_MODE,
            "arm_flow_mode": "relative_command_adapter",
            # This names the shared 18-D codec boundary.  It is retained even
            # when the native gripper owner is the private binary head.
            "continuous_gripper_codec_boundary": "previous_command",
            "continuous_gripper_codec_boundary_scope": (
                CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE
            ),
            "receding_horizon_execute_rows": 1,
            "prediction_horizon": 24,
        },
        "normalizers": {
            "mode": "zscore",
            "action_sha256": "a" * 64,
            "state_sha256": "b" * 64,
        },
        "language": {"sha256": "c" * 64},
    }


def test_maniskill_binary_deployment_abi_round_trips_and_rejects_mismatch() -> None:
    value = _binary_deployment_abi()
    validate_deployment_abi(value)
    # The persisted JSON form must be accepted too (tuples become lists).
    validate_deployment_abi(json.loads(json.dumps(value)))

    altered = deepcopy(value)
    altered["action"]["gripper_output_mode"] = "continuous"  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_deployment_abi(altered)

    altered = deepcopy(value)
    altered["action"]["data_profile"]["name"] = (  # type: ignore[index]
        "maniskill_pd_ee_delta_pose_7d_v1"
    )
    with pytest.raises(ValueError):
        validate_deployment_abi(altered)


def test_maniskill_binary_mode_is_not_accepted_for_legacy_profile() -> None:
    base = _maniskill_binary_config()
    legacy = replace(
        base,
        data=replace(base.data, data_profile="maniskill_pd_ee_delta_pose_7d_v1"),
    )
    with pytest.raises(ValueError, match="repaired v2"):
        legacy.validate()


def test_maniskill_binary_outlet_neutralizes_field_and_writes_native_command() -> None:
    adapter = OutletAdapter(
        PhysicalActionFieldCodec(action_dim=7, horizon=24),
        selection="maniskill_7d_binary_v2",
    )
    adapter.configure_action_normalizer(
        SimpleNamespace(offset=np.zeros((1, 7)), scale=np.full((1, 7), 2.0))
    )
    field = torch.randn(1, 24, 18)
    conditioned = adapter.prepare_model_input(field)
    torch.testing.assert_close(conditioned[..., :12], field[..., :12])
    torch.testing.assert_close(conditioned[..., 12:], torch.zeros_like(field[..., 12:]))
    metrics = adapter.conditioning_metrics(
        field,
        conditioned,
        collect_diagnostics=True,
    )
    assert metrics["bottom_binary_gripper_condition_max_abs"].item() == 0.0
    assert metrics["bottom_binary_gripper_condition_neutral"].item() == 1.0
    assert (
        metrics["bottom_maniskill_binary_gripper_condition_max_abs"].item() == 0.0
    )
    assert "bottom_calvin_binary_gripper_condition_max_abs" not in metrics

    action_state = torch.zeros(1, 7)
    logits = torch.zeros(1, 24, 2)
    logits[..., 1] = 1.0
    output = adapter.finalize(
        field,
        action_state,
        codec_gripper_boundary=torch.zeros(1, 1),
        command_logits=logits,
    )
    assert torch.equal(output.command, torch.ones(1, 24))
    assert torch.equal(output.deployed_action[..., -1], torch.ones(1, 24))
    torch.testing.assert_close(output.world_condition_action[..., -1], torch.full((1, 24), 2.0))


class _FakeVelocityHead(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        arm = torch.cat((tokens, tokens[..., :4]), dim=-1)
        return torch.cat(
            (arm, torch.ones_like(tokens[..., :6])), dim=-1
        )

    def forward_with_gripper_state(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(tokens), tokens + 0.5, torch.ones_like(tokens)


def test_binary_candidate_reads_share_the_same_gripper_isolation_boundary() -> None:
    controller = TerminalActionController(
        action_norm=nn.Identity(),
        velocity_head=_FakeVelocityHead(),  # type: ignore[arg-type]
        optional_command_head=nn.Sequential(nn.Linear(8, 2)),
        optional_event_head=None,
        motion_head=nn.Sequential(nn.Linear(8, 1)),
        arm_dim=6,
    ).eval()
    tokens = torch.randn(2, 24, 8)
    candidate = controller.predict_candidate_velocity(tokens)
    torch.testing.assert_close(candidate[..., 12:], torch.zeros_like(candidate[..., 12:]))
    final = controller.read_heads(tokens, collect_diagnostics=False)
    torch.testing.assert_close(final.physical_velocity[..., 12:], torch.zeros_like(final.physical_velocity[..., 12:]))
    assert final.command_logits is not None
