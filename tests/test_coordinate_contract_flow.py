from types import SimpleNamespace

import torch

from clearvla.mainline.model.restored_observation import _v120_flow_field
from clearvla.mainline.spatial_geometry import build_dino_charts, build_raw_charts
from clearvla.mainline.v120_core.flow_dino_evidence import (
    FlowDINOEvidencePack,
    _RawPyramidFlow,
    _SoftMultiResolutionAddressCompiler,
)


def _compiler_config() -> SimpleNamespace:
    return SimpleNamespace(
        flow_jepa_grid_size=8,
        num_cameras=2,
        flow_jepa_address_slots=4,
        flow_jepa_address_route_dim=8,
        flow_jepa_raw_reader_radius=1,
        flow_jepa_address_flow_prior_floor=0.2,
        flow_jepa_complete_numerical_contract=0,
        flow_jepa_progressive_grounding_address=1,
        flow_jepa_coordinate_typed_raw_detail=1,
        flow_jepa_routing_norm_floor=0.25,
        visual_token_dim=4,
        flow_jepa_coordinate_contract="canonical_rgb_lattice_v1",
        flow_jepa_coordinate_processor_resize_hw=(32, 32),
        flow_jepa_coordinate_processor_crop_hw=(32, 32),
        flow_jepa_coordinate_processor_crop_offset_yx=(0, 0),
        flow_jepa_coordinate_processor_patch_hw=(4, 4),
    )


def test_canonical_address_bank_uses_real_raw_and_dino_chart_support() -> None:
    config = _compiler_config()
    compiler = _SoftMultiResolutionAddressCompiler(config, raw_dim=4)
    source_dino = torch.randn(1, 2, 8, 8, 4)
    target_dino = torch.randn_like(source_dino)
    source_raw = torch.randn(1, 2, 4, 16, 16)
    target_raw = torch.randn_like(source_raw)
    confidence = torch.ones(1, 2, 1, 16, 16)
    bank, _ = compiler(
        source_dino=source_dino,
        target_dino=target_dino,
        source_raw=source_raw,
        target_raw=target_raw,
        current_rgb=torch.rand(1, 2, 3, 64, 64),
        flow=torch.zeros(1, 2, 2, 16, 16),
        confidence=confidence,
        uncertainty=confidence,
        occlusion=torch.zeros_like(confidence),
        cycle_error=confidence,
    )
    assert bank.coordinate_contract == "canonical_rgb_lattice_v1"
    assert bank.outer_chart is not None and bank.raw_chart is not None
    assert bank.dino_chart is not None
    valid_fraction = bank.fine_valid.float().mean()
    assert bool((valid_fraction > 0).item())
    assert bool((valid_fraction <= 1).item())


def test_canonical_raw_refiner_resamples_seed_in_chart_units() -> None:
    config = SimpleNamespace(
        flow_jepa_zero_flow_guard=1,
        flow_jepa_bounded_flow_coordinates=1,
        flow_jepa_coordinate_contract="canonical_rgb_lattice_v1",
        flow_jepa_complete_numerical_contract=0,
        flow_jepa_visibility_transition_fraction=0.1,
        flow_jepa_raw_activation_checkpoint=0,
        flow_jepa_raw_base_channels=8,
        flow_jepa_feature_dim=8,
        flow_jepa_raw_mid_radius=1,
        flow_jepa_raw_high_radius=1,
        flow_jepa_uncertainty_floor=0.03,
        flow_jepa_correlation_rms_floor=0.1,
    )
    refiner = _RawPyramidFlow(config)
    rgb = torch.rand(1, 2, 2, 3, 64, 64)
    coarse = torch.zeros(2, 2, 4, 4)
    reliability = torch.ones(2, 1, 4, 4)
    context, _, _ = refiner(rgb, coarse, coarse, reliability, reliability)
    assert tuple(context.flow_forward.shape) == (1, 1, 2, 2, 16, 16)


def test_restored_flow_exports_detail_chart_and_canonical_units() -> None:
    batch, cameras, grid, detail_side = 1, 2, 8, 16
    pack = FlowDINOEvidencePack(
        selector_tokens=torch.empty(batch, 0, 4),
        value_tokens=torch.empty(batch, 0, 4),
        key_bias=torch.empty(0),
        stage_query=torch.empty(batch, 0, 4),
        future_queries=torch.empty(batch, 0, 4),
        context_dropout_mask=torch.zeros(
            batch, 3, cameras, grid, grid, dtype=torch.bool
        ),
        future_target_mask=torch.empty(batch, 0),
        patch_flow_forward=torch.ones(batch, 2, cameras, 2, grid, grid),
        patch_flow_backward=torch.ones(batch, 2, cameras, 2, grid, grid),
        flow_confidence=torch.ones(batch, 2, cameras, 1, grid, grid),
        flow_occlusion=torch.zeros(batch, 2, cameras, 1, grid, grid),
        losses={},
        metrics={},
    )
    raw_chart = build_raw_charts((64, 64))["raw_high"]
    dino_chart = build_dino_charts(
        (64, 64),
        resized_hw=(32, 32),
        crop_hw=(32, 32),
        crop_offset_yx=(0, 0),
        patch_hw=(4, 4),
        pooled_hw=(grid, grid),
    )["dino_pooled"]
    field = _v120_flow_field(
        pack,
        -1,
        target_shape=(detail_side, detail_side),
        canonical_raw_chart=raw_chart,
        canonical_dino_chart=dino_chart,
    )
    assert tuple(field.forward.shape) == (batch, cameras, 2, detail_side, detail_side)
    assert tuple(field.backward.shape) == tuple(field.forward.shape)
    assert tuple(field.confidence.shape) == (
        batch,
        cameras,
        1,
        detail_side,
        detail_side,
    )
