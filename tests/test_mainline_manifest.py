from __future__ import annotations

import hashlib
from dataclasses import replace

import torch

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.manifest import (
    ARCHITECTURE_MANIFEST,
    ARM_ONLY_BSPINE_ARCHITECTURE_MANIFEST,
    BSPINE_ARCHITECTURE_MANIFEST,
    ArchitectureManifest,
    ComponentABI,
    architecture_manifest_for_bspine_implementation,
    manifest_from_mapping,
)
from clearvla.mainline.model.component_contracts import (
    BSPINE_ARM_ONLY_EXECUTION_BOTTOM,
    BSPINE0_EXECUTION_BOTTOM,
    ComponentSelection,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.deployment import (
    CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE,
    DEPLOYMENT_ABI_SCHEMA,
    canonical_sha256,
    deployment_graph_config,
    validate_deployment_abi,
)
from clearvla.mainline.training.optimizer import build_optimizer
from clearvla.mainline.v120_core.bspine import (
    BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
    BSPINE_ARM_ONLY_IMPLEMENTATION,
    BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
    BSPINE0_BASIS_DIGEST,
    BSPINE0_CONTROL_POINTS,
    BSPINE0_DEGREE,
    BSPINE0_IMPLEMENTATION,
    BSPINE0_SPEC_FINGERPRINT,
)


def test_mainline_manifest_round_trip_is_stable() -> None:
    restored = manifest_from_mapping(ARCHITECTURE_MANIFEST.as_dict())
    assert restored == ARCHITECTURE_MANIFEST
    assert restored.layout_schema == 2
    assert restored.digest() == ARCHITECTURE_MANIFEST.digest()
    assert len(restored.digest()) == 64


def test_bspine_selects_schema31_without_relabeling_the_baseline() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE0_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE0_SPEC_FINGERPRINT,
        ),
    )
    config.validate()
    manifest = architecture_manifest_for_bspine_implementation(
        config.bottom.bspine_implementation
    )
    assert ARCHITECTURE_MANIFEST.schema == 30
    assert manifest is BSPINE_ARCHITECTURE_MANIFEST
    assert manifest.schema == 31
    assert manifest.digest() != ARCHITECTURE_MANIFEST.digest()
    assert manifest_from_mapping(manifest.as_dict()) == manifest
    selection = ComponentSelection.from_config(config)
    assert selection.execution_bottom == BSPINE0_EXECUTION_BOTTOM


def test_arm_only_bspine_has_distinct_manifest_and_component_identity() -> None:
    base = ExperimentConfig()
    config = replace(
        base,
        bottom=replace(
            base.bottom,
            bspine_implementation=BSPINE_ARM_ONLY_IMPLEMENTATION,
            bspine_degree=BSPINE0_DEGREE,
            bspine_control_points=BSPINE0_CONTROL_POINTS,
            bspine_basis_digest=BSPINE0_BASIS_DIGEST,
            bspine_spec_fingerprint=BSPINE_ARM_ONLY_SPEC_FINGERPRINT,
            bspine_action_group_mask=BSPINE_ARM_ONLY_ACTION_GROUP_MASK,
        ),
    )
    config.validate()
    manifest = architecture_manifest_for_bspine_implementation(
        config.bottom.bspine_implementation
    )
    assert manifest is ARM_ONLY_BSPINE_ARCHITECTURE_MANIFEST
    assert manifest.schema == BSPINE_ARCHITECTURE_MANIFEST.schema == 31
    assert len(
        {
            ARCHITECTURE_MANIFEST.digest(),
            BSPINE_ARCHITECTURE_MANIFEST.digest(),
            ARM_ONLY_BSPINE_ARCHITECTURE_MANIFEST.digest(),
        }
    ) == 3
    assert manifest.components.bottom != BSPINE_ARCHITECTURE_MANIFEST.components.bottom
    assert manifest_from_mapping(manifest.as_dict()) == manifest
    selection = ComponentSelection.from_config(config)
    assert selection.execution_bottom == BSPINE_ARM_ONLY_EXECUTION_BOTTOM
    assert selection.execution_bottom != BSPINE0_EXECUTION_BOTTOM


def test_mainline_manifest_rejects_version_or_component_drift() -> None:
    old_top = dict(ARCHITECTURE_MANIFEST.as_dict())
    old_top["schema"] = 3
    try:
        manifest_from_mapping(old_top)
    except ValueError as error:
        assert "capability identity" in str(error)
    else:
        raise AssertionError("schema-3 top must not enter the mainline")

    incompatible = ArchitectureManifest(components=ComponentABI(bottom="historical_bottom"))
    # A syntactically valid different component identity is allowed to exist
    # as data, but it cannot equal the active manifest used for resume.
    incompatible.validate()
    assert incompatible != ARCHITECTURE_MANIFEST
    assert incompatible.digest() != ARCHITECTURE_MANIFEST.digest()


def test_mainline_manifest_contains_no_run_label() -> None:
    payload = ARCHITECTURE_MANIFEST.as_dict()
    assert "run_label" not in payload
    assert all(not key.startswith("v1") for key in payload)


def test_mainline_manifest_names_the_current_component_semantics() -> None:
    components = ARCHITECTURE_MANIFEST.components
    assert ARCHITECTURE_MANIFEST.schema == 30
    assert components.observation == (
        "restored_v120_three_frame_flow_dino_progressive_g123_fp32_owner_logs_zero_preserving_variance"
    )
    assert (
        components.top
        == "v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_s_owned_relevance_goal_invariant_physical_action_conditioned_w_single_consequence_refinement_p2_transport_address_typed_consequence_two_optional_p3_schema28_core_recovery"
    )
    assert (
        components.bottom
        == "restored_v120_shared_seed_dynamic_p1_terminal_layer_contracts_lane_local_p3_evidence_mmdit_dense512_execution_fp32_capacity_gripper_private_continuous_field_no_event_head"
    )
    assert (
        components.training
        == "v120_mirrored_physical_flow_exact_teacher_current_support_raw_transport_event_transition_persistence_gripper_trajectory_v120_decay_local_global_clip_physical_w_ingress_gradient_probes_schema28_core_recovery_profile_owned_full_horizon_gripper_codec_boundary"
    )
    assert components.runtime == (
        "cached_observation_progressive_gsw_exact_p1_physical_action_tagged_w_single_refinement_v120_nodes_clean_endpoint_decoded_gripper_events_teacher_isolated_finite_spike_matched_p2_value_address_capacity_metrics_schema28_core_recovery_profile_owned_full_horizon_gripper_codec_boundary_source_native_metrics"
    )


def test_deployment_abi_rejects_pre_boundary_scope_checkpoints() -> None:
    config = ExperimentConfig()
    graph = deployment_graph_config(config)
    profile = {
        "name": "identity_7d_pen",
        "gripper_transition_boundary": "current_action_state",
    }
    action = {
        "data_profile": profile,
        "gripper_indices": [6],
        "gripper_output_mode": config.bottom.gripper_output_mode,
        "arm_flow_mode": config.bottom.arm_flow_mode,
        "continuous_gripper_codec_boundary": "current_action_state",
        "continuous_gripper_codec_boundary_scope": (
            CONTINUOUS_GRIPPER_CODEC_BOUNDARY_SCOPE
        ),
        "receding_horizon_execute_rows": 1,
        "prediction_horizon": config.dimensions.action_horizon,
    }
    abi = {
        "schema": DEPLOYMENT_ABI_SCHEMA,
        "graph_config": graph,
        "graph_config_sha256": canonical_sha256(graph),
        "observation": {
            "camera_names": ["top", "wrist"],
            "visual_offsets": [-8, -4, 0],
            "state_offsets": [-8, -4, 0],
            "executed_action_offsets": [-24, -16, -12, -8, -6, -4, -2, -1],
            "dinov2": {
                "model": "test",
                "compute_dtype": "fp32",
                "reference_batch_size": 1,
            },
        },
        "action": action,
        "normalizers": {
            "mode": "zscore",
            "action_sha256": "a" * 64,
            "state_sha256": "b" * 64,
        },
        "language": {"sha256": "c" * 64},
    }
    assert validate_deployment_abi(abi)["action"] == action

    stale = dict(abi)
    stale["action"] = {
        key: value
        for key, value in action.items()
        if key != "continuous_gripper_codec_boundary_scope"
    }
    try:
        validate_deployment_abi(stale)
    except ValueError as error:
        assert "boundary scope is stale" in str(error)
    else:
        raise AssertionError("pre-boundary deployment ABI was accepted")


def test_schema_30_parameter_inventory_is_explained_by_active_modules() -> None:
    torch.manual_seed(0)
    model = ClearVLAMainlinePolicy(ExperimentConfig())

    def counts(module):
        parameters = tuple(module.parameters())
        return (
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        )

    children = {name: counts(module) for name, module in model.named_children()}
    total, trainable = counts(model)
    assert total == sum(value[0] for value in children.values())
    assert trainable == sum(value[1] for value in children.values())
    assert children["execution_bottom"][0] > 40_000_000
    assert children["p1"][0] > 2_000_000
    assert counts(model.grounding.blocks)[0] > 1_000_000
    assert counts(model.grounding.grounder)[0] > 1_000_000
    assert (total, trainable) == (168_417_179, 152_046_448)
    parameters = tuple(model.parameters())
    assert len(parameters) == 1_385
    assert sum(parameter.requires_grad for parameter in parameters) == 1_063
    state_names = tuple(model.state_dict())
    assert len(state_names) == 1_391
    assert hashlib.sha256("\n".join(state_names).encode()).hexdigest() == (
        "846b1edd7933b796882bcb5a8422816f768110fe9741282ba4435ac45927b7ca"
    )
    assert hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest() == (
        "d3bcc995a57b40e359a6370a4dc3eea1638fa4a210f3082e41f6791a75513c21"
    )
    optimizer, ownership = build_optimizer(model, ExperimentConfig())
    assert len(optimizer.param_groups) == 23
    assert len(ownership.trainable_names) == 1_063
    assert model.execution_bottom.decoder.terminal_controller.optional_event_head is None
    assert not hasattr(model, "decoded_gripper_event_head")
