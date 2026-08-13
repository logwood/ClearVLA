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
    assert ARCHITECTURE_MANIFEST.schema == 23
    assert components.observation == "restored_v120_three_frame_flow_dino_raw_local_chart"
    assert (
        components.top
        == "v120_cumulative_intent_four_interval_dynamics_split_support_selector_protected_candidate_p1_five_lane_p3"
    )
    assert (
        components.bottom
        == "restored_v120_shared_seed_dynamic_p1_p1_p2_contracts_evidence_mmdit_dense512_execution"
    )
    assert (
        components.training
        == "v120_mirrored_physical_flow_exact_teacher_current_support_event_boost_exact_role_lr"
    )
    assert components.runtime == (
        "cached_observation_gsw_p1_detail_v120_nodes_clean_endpoint_teacher_isolated"
    )


def test_schema_23_active_parameter_inventory_cannot_silently_shrink() -> None:
    model = ClearVLAMainlinePolicy(ExperimentConfig())

    def counts(module):
        parameters = tuple(module.parameters())
        return (
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        )

    assert counts(model) == (182_724_214, 164_041_578)
    assert {name: counts(module) for name, module in model.named_children()} == {
        "observation": (13_543_661, 3_819_155),
        "action_codec": (0, 0),
        "top": (92_909_001, 92_810_697),
        "history_proposal": (10_014_727, 10_010_631),
        "factual_reader": (2_467_328, 2_467_328),
        "transition": (8_029_833, 7_897_097),
        "bottom": (55_759_664, 47_036_670),
    }
