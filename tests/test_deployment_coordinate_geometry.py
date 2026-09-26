from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import (
    ArtifactIdentity,
    CheckpointIdentity,
    DatasetIdentity,
    SourceSnapshot,
)
from clearvla.mainline.config import ExperimentConfig, ObservationConfig
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.manifest import ARCHITECTURE_MANIFEST
from clearvla.mainline.runtime.deployment import (
    _coordinate_geometry_abi,
    build_deployment_abi,
    canonical_sha256,
    deployment_config_from_checkpoint,
    validate_deployment_abi,
)
from clearvla.mainline.spatial_geometry import ChartSpec


def _build_abi(config: ExperimentConfig) -> dict:
    """Exercise the production builder without requiring external artifacts."""
    digest = "a" * 64
    sources = (("clearvla/mainline/train.py", digest),)
    identity = CheckpointIdentity(
        manifest=ARCHITECTURE_MANIFEST.as_dict(),
        manifest_digest=ARCHITECTURE_MANIFEST.digest(),
        config_digest=config.digest(),
        source=SourceSnapshot(sources, canonical_sha256(sources)),
        git_commit="a" * 40,
        dataset=DatasetIdentity("fixture", "*.hdf5", *([digest] * 5)),
        language=ArtifactIdentity("language", "fixture.pt", 1, digest),
    )
    normalizer = ArrayNormalizer.fit_zscore([np.zeros((2, 7), dtype=np.float32)])
    profile = resolve_action_state_profile(config.data.data_profile)
    profile_metadata = profile.as_dict()
    profile_metadata["gripper_transition_boundary"] = "current_action_state"
    return build_deployment_abi(
        config,
        identity,
        action_normalizer=normalizer,
        state_normalizer=normalizer,
        data_profile=profile_metadata,
        gripper_indices=profile.gripper_indices,
        goal_metadata={},
    )


@pytest.fixture
def canonical_abi() -> dict:
    return _build_abi(ExperimentConfig(
        observation=ObservationConfig(coordinate_contract_mode="canonical_v1")
    ))


def _rehash_geometry(abi: dict) -> None:
    geometry = abi["observation"]["coordinate_geometry"]
    geometry["sha256"] = canonical_sha256({
        key: value for key, value in geometry.items() if key != "sha256"
    })


def test_legacy_deployment_abi_does_not_gain_geometry() -> None:
    config = ExperimentConfig()
    assert _coordinate_geometry_abi(config) == {}
    abi = _build_abi(config)
    assert "coordinate_geometry" not in abi["observation"]
    assert validate_deployment_abi(abi) == abi


def test_legacy_dino_shape_must_match_graph() -> None:
    abi = _build_abi(ExperimentConfig())
    abi["observation"]["dinov2"]["patches_per_camera"] = 64
    with pytest.raises(ValueError, match="DINO patches_per_camera differs"):
        validate_deployment_abi(abi)


def test_canonical_deployment_geometry_is_explicit_and_round_trips(canonical_abi: dict) -> None:
    config = ExperimentConfig(
        observation=ObservationConfig(coordinate_contract_mode="canonical_v1")
    )
    config.validate()
    geometry = _coordinate_geometry_abi(config)

    assert geometry["mode"] == "canonical_rgb_lattice_v1"
    assert geometry["processor"] == {
        "resize_hw": [256, 256],
        "crop_hw": [224, 224],
        "crop_offset_yx": [16, 16],
        "patch_hw": [14, 14],
    }
    charts = geometry["charts"]
    assert isinstance(charts, dict)
    for name in ("outer_rgb", "raw_high", "raw_mid", "dino_patch", "dino_pooled"):
        chart = ChartSpec.from_dict(charts[name])
        assert chart.canonical_frame == "outer_rgb_pixel_centers_v1"

    assert canonical_abi["observation"]["coordinate_geometry"] == geometry
    assert validate_deployment_abi(canonical_abi) == canonical_abi
    restored = deployment_config_from_checkpoint(config.as_dict(), canonical_abi)
    assert restored.observation == config.observation
    assert restored.data.cache_side == config.data.cache_side


def test_geometry_cannot_be_attached_to_legacy_graph(canonical_abi: dict) -> None:
    abi = _build_abi(ExperimentConfig())
    abi["observation"]["coordinate_geometry"] = canonical_abi["observation"]["coordinate_geometry"]
    with pytest.raises(ValueError, match="legacy coordinate contract"):
        validate_deployment_abi(abi)


def test_legacy_graph_cannot_carry_hidden_processor_geometry() -> None:
    abi = _build_abi(ExperimentConfig())
    abi["graph_config"]["observation"]["coordinate_processor_resize_hw"] = [256, 256]
    abi["graph_config_sha256"] = canonical_sha256(abi["graph_config"])
    with pytest.raises(ValueError, match="legacy coordinate contract"):
        validate_deployment_abi(abi)


def test_canonical_graph_cannot_omit_geometry(canonical_abi: dict) -> None:
    del canonical_abi["observation"]["coordinate_geometry"]
    with pytest.raises(ValueError, match="requires deployment geometry"):
        validate_deployment_abi(canonical_abi)


@pytest.mark.parametrize("rehash", [False, True])
def test_processor_tamper_is_rejected_even_with_recomputed_hash(canonical_abi: dict, rehash: bool) -> None:
    canonical_abi["observation"]["coordinate_geometry"]["processor"]["crop_offset_yx"] = [0, 0]
    if rehash:
        _rehash_geometry(canonical_abi)
    with pytest.raises(ValueError, match="digest|differs from graph"):
        validate_deployment_abi(canonical_abi)


@pytest.mark.parametrize("chart", ["outer_rgb", "raw_high", "raw_mid", "dino_patch", "dino_pooled"])
def test_chart_tamper_is_rejected_even_with_recomputed_hash(canonical_abi: dict, chart: str) -> None:
    canonical_abi["observation"]["coordinate_geometry"]["charts"][chart]["origin_xy"][0] += 0.5
    _rehash_geometry(canonical_abi)
    with pytest.raises(ValueError, match="coordinate charts differ"):
        validate_deployment_abi(canonical_abi)


def test_changing_graph_and_processor_does_not_validate_stale_charts(canonical_abi: dict) -> None:
    canonical_abi["graph_config"]["observation"]["coordinate_processor_crop_offset_yx"] = [0, 0]
    canonical_abi["graph_config_sha256"] = canonical_sha256(canonical_abi["graph_config"])
    canonical_abi["observation"]["coordinate_geometry"]["processor"]["crop_offset_yx"] = [0, 0]
    _rehash_geometry(canonical_abi)
    with pytest.raises(ValueError, match="coordinate charts differ"):
        validate_deployment_abi(canonical_abi)


@pytest.mark.parametrize("field,value", [("resize_hw", [224, 224]), ("crop_hw", [224, 224])])
def test_outer_rgb_preprocessing_must_match_geometry(canonical_abi: dict, field: str, value: list[int]) -> None:
    canonical_abi["observation"]["rgb_preprocessing"]["config"][field] = value
    with pytest.raises(ValueError, match="uncropped 336px RGB"):
        validate_deployment_abi(canonical_abi)


def test_outer_rgb_resize_backend_must_match_executable_contract(canonical_abi: dict) -> None:
    current = canonical_abi["observation"]["rgb_preprocessing"]["resize_backend"]
    canonical_abi["observation"]["rgb_preprocessing"]["resize_backend"] = (
        "pillow-bilinear" if current == "opencv-inter-area" else "opencv-inter-area"
    )
    with pytest.raises(ValueError, match="uncropped 336px RGB"):
        validate_deployment_abi(canonical_abi)


@pytest.mark.parametrize("value", [[True, 16], [16.0, 16], [16], None])
def test_processor_requires_exact_integer_pairs(canonical_abi: dict, value: object) -> None:
    canonical_abi["graph_config"]["observation"]["coordinate_processor_crop_offset_yx"] = value
    canonical_abi["graph_config_sha256"] = canonical_sha256(canonical_abi["graph_config"])
    canonical_abi["observation"]["coordinate_geometry"]["processor"]["crop_offset_yx"] = value
    _rehash_geometry(canonical_abi)
    with pytest.raises(ValueError, match="integer|malformed|sequence"):
        validate_deployment_abi(canonical_abi)


@pytest.mark.parametrize("owner", ["graph", "DINO"])
def test_patch_count_must_match_geometry(canonical_abi: dict, owner: str) -> None:
    if owner == "graph":
        canonical_abi["graph_config"]["dimensions"]["patches_per_camera"] = 64
        canonical_abi["graph_config_sha256"] = canonical_sha256(canonical_abi["graph_config"])
    else:
        canonical_abi["observation"]["dinov2"]["patches_per_camera"] = 64
    with pytest.raises(ValueError, match="patch count differs"):
        validate_deployment_abi(canonical_abi)


def test_nondefault_consistent_processor_geometry_is_allowed() -> None:
    config = ExperimentConfig(observation=ObservationConfig(
        coordinate_contract_mode="canonical_v1",
        coordinate_processor_resize_hw=(280, 280),
        coordinate_processor_crop_offset_yx=(28, 28),
    ))
    abi = _build_abi(config)
    assert validate_deployment_abi(deepcopy(abi)) == abi


def test_canonical_config_rejects_processor_chart_count_mismatch() -> None:
    with pytest.raises(ValueError, match="patch chart must match"):
        ExperimentConfig(observation=ObservationConfig(
            coordinate_contract_mode="canonical_v1",
            coordinate_processor_patch_hw=(7, 7),
        )).validate()


def test_canonical_config_rejects_crop_outside_processor_resize() -> None:
    with pytest.raises(ValueError, match="processor geometry is invalid"):
        ExperimentConfig(observation=ObservationConfig(
            coordinate_contract_mode="canonical_v1",
            coordinate_processor_crop_offset_yx=(40, 40),
        )).validate()


def test_canonical_geometry_rejects_non_affine_future_pooling() -> None:
    with pytest.raises(ValueError, match="non-affine"):
        # The chart builder is intentionally fail-closed when a producer's
        # pooling centers cannot be represented by one affine lattice.
        from clearvla.mainline.spatial_geometry import build_dino_charts

        build_dino_charts(
            (336, 336),
            resized_hw=(256, 256),
            crop_hw=(224, 224),
            crop_offset_yx=(16, 16),
            patch_hw=(14, 14),
            pooled_hw=(7, 7),
        )
