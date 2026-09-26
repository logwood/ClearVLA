import torch

from clearvla.mainline.spatial_geometry import (
    area_pool_chart,
    build_raw_charts,
    rgb_chart,
)
from clearvla.mainline.v120_core.flow_dino_evidence import (
    _EarlyMaskedRawContextEncoder,
    _RawPyramidFlow,
)


def test_early_canonical_rgb_carrier_is_resampled_to_dino_centres() -> None:
    kwargs = dict(
        feature_dim=8,
        hidden=8,
        grid=8,
        activation_checkpoint=False,
        processor_resize_hw=(32, 32),
        processor_crop_hw=(32, 32),
        processor_crop_offset_yx=(0, 0),
        processor_patch_hw=(4, 4),
    )
    legacy = _EarlyMaskedRawContextEncoder(**kwargs, coordinate_canonical=False)
    canonical = _EarlyMaskedRawContextEncoder(**kwargs, coordinate_canonical=True)
    canonical.load_state_dict(legacy.state_dict())
    raw = torch.linspace(0.0, 1.0, 64 * 64).reshape(1, 1, 1, 1, 64, 64).repeat(
        1, 2, 1, 3, 1, 1
    )
    mask = torch.zeros(1, 2, 1, 8, 8, dtype=torch.bool)
    with torch.no_grad():
        legacy_value = legacy(raw, mask)
        canonical_value = canonical(raw, mask)
    assert tuple(canonical_value.shape) == (1, 2, 1, 8, 8, 8)
    assert torch.isfinite(canonical_value).all()
    # The two charts have different centers on the 64px outer RGB frame; a
    # nonconstant ramp must therefore exercise the explicit resampling path.
    assert not torch.allclose(legacy_value, canonical_value)


def test_fixed_descriptor_alignment_and_warp_keep_chart_support() -> None:
    rgb = rgb_chart((64, 64))
    source = area_pool_chart(rgb, (16, 16), "descriptor")
    target = build_raw_charts((64, 64))["raw_high"]
    descriptor = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    aligned, support = _RawPyramidFlow._align_fixed_descriptor(
        descriptor, source, target
    )
    assert tuple(aligned.shape[-2:]) == (16, 16)
    assert support.dtype == torch.bool
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = 1.0
    warped, valid = _RawPyramidFlow._supported_descriptor_warp(
        aligned, flow, support
    )
    assert torch.isfinite(warped).all()
    assert valid.shape == (1, 1, 16, 16)
    assert bool(valid[..., 1:].any())
