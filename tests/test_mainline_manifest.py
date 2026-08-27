from __future__ import annotations

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.manifest import (
    ARCHITECTURE_MANIFEST,
    ArchitectureManifest,
    ComponentABI,
    manifest_from_mapping,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy


def test_mainline_manifest_round_trip_is_stable() -> None:
    restored = manifest_from_mapping(ARCHITECTURE_MANIFEST.as_dict())
    assert restored == ARCHITECTURE_MANIFEST
    assert restored.digest() == ARCHITECTURE_MANIFEST.digest()
    assert len(restored.digest()) == 64


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
    assert ARCHITECTURE_MANIFEST.schema == 25
    assert components.observation == (
        "restored_v120_three_frame_flow_dino_progressive_g123_fp32_owner_logs_zero_preserving_variance"
    )
    assert (
        components.top
        == "v120_progressive_g123_dense_grounder_fp32_support_logs_exact_p1_s_owned_k_typed_relevance_four_interval_w_stage_private_p2_terminal_protected_plus_two_optional_p3"
    )
    assert (
        components.bottom
        == "restored_v120_shared_seed_dynamic_p1_terminal_layer_contracts_lane_local_p3_evidence_mmdit_dense512_execution_gripper_private_continuous_state"
    )
    assert (
        components.training
        == "v120_mirrored_physical_flow_exact_teacher_current_support_target_scale_transport_event_boost_v120_decay_local_global_clip_source_gradient_probes"
    )
    assert components.runtime == (
        "cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated_finite_spike_matching_metrics"
    )


def test_schema_25_parameter_inventory_is_explained_by_active_modules() -> None:
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
    assert children["bottom"][0] > 50_000_000
    assert children["factual_reader"][0] > 2_000_000
    assert counts(model.top.grounding_blocks)[0] > 1_000_000
    assert counts(model.top.grounder)[0] > 1_000_000
