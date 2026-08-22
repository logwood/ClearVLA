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
    assert ARCHITECTURE_MANIFEST.schema == 32
    assert components.observation == "restored_v120_three_frame_flow_dino_progressive_g123_bank"
    assert (
        components.top
        == "canonical_decoded_k_content_stateless_state_supervision_w_owned_common_residual_typed_base_interaction_camera_mixture_p2_consequence_p3"
    )
    assert (
        components.bottom
        == "restored_v120_shared_seed_dynamic_p1_four_active_plan_lanes_exact_g3_anchor_transition_evidence_mmdit_dense512_execution"
    )
    assert (
        components.training
        == "v120_mirrored_physical_flow_observed_current_grounding_partial_ot_single_w_future_effect_target_physical_camera_loss_support_event_boost_v120_decay_three_owner_clip"
    )
    assert components.runtime == (
        "cached_observation_progressive_gsw_exact_p1_v120_nodes_clean_endpoint_teacher_isolated_active_ablations_only"
    )


def test_schema_32_parameter_inventory_is_explained_by_active_modules() -> None:
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
