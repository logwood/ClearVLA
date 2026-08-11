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
    assert ARCHITECTURE_MANIFEST.schema == 20
    assert components.observation == "restored_v120_three_frame_flow_dino_raw_local_chart"
    assert (
        components.top
        == "global_object_intent_four_interval_dynamics_local_p1_additive_p3"
    )
    assert (
        components.bottom
        == "restored_v120_evidence_mmdit_dense512_execution_value_capacity"
    )
    assert (
        components.training
        == "v120_physical_flow_interval_transition_execution_value_role_lr"
    )
    assert components.runtime == "cached_five_step_teacher_isolated_exact_resume_semantic_logging"


def test_schema_20_active_parameter_inventory_cannot_silently_shrink() -> None:
    model = ClearVLAMainlinePolicy(ExperimentConfig())

    def counts(module):
        parameters = tuple(module.parameters())
        return (
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        )

    assert counts(model) == (182_267_215, 167_031_918)
    assert {name: counts(module) for name, module in model.named_children()} == {
        "observation": (13_543_661, 3_819_155),
        "action_codec": (0, 0),
        "top": (98_808_785, 98_710_481),
        "history_proposal": (10_014_727, 10_010_631),
        "factual_reader": (8_763_904, 8_763_904),
        "transition": (8_825_993, 8_690_697),
        "bottom": (42_310_145, 37_037_050),
    }
