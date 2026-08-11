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
    assert ARCHITECTURE_MANIFEST.schema == 19
    assert components.observation == "causal_three_frame_dino_raw_two_flow_pre_g_v5"
    assert (
        components.top
        == "object_intent_dynamics_323_keyed_g_local_p1_additive_p3_v14"
    )
    assert (
        components.bottom
        == "typed_evidence_mmdit_dense_transition4basis_zero_proposal_fullwidth_capacity_v9"
    )
    assert (
        components.training
        == "single_stage_physical_action_v120_role_lr_horizon_event_v12"
    )
    assert components.runtime == "cached_five_step_ode_lossless_semantic_logging_v11"


def test_schema_19_active_parameter_inventory_cannot_silently_shrink() -> None:
    model = ClearVLAMainlinePolicy(ExperimentConfig())

    def counts(module):
        parameters = tuple(module.parameters())
        return (
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        )

    assert counts(model) == (171_940_734, 171_838_334)
    assert {name: counts(module) for name, module in model.named_children()} == {
        "observation": (12_858_245, 12_858_245),
        "action_codec": (0, 0),
        "top": (98_808_785, 98_710_481),
        "history_proposal": (10_014_727, 10_010_631),
        "factual_reader": (8_813_056, 8_813_056),
        "transition": (9_001_993, 9_001_993),
        "bottom": (32_443_928, 32_443_928),
    }
