from __future__ import annotations

import pytest
import torch

from clearvla.mainline.spatial_geometry import (
    ChartSpec,
    area_pool_chart,
    build_dino_charts,
    build_raw_charts,
    convert_covariance,
    convert_points,
    convert_variance,
    convert_vectors,
    resample_flow,
    sample_field,
)


def _chart(name: str, shape: tuple[int, int], origin=(0.0, 0.0), pitch=(1.0, 1.0)) -> ChartSpec:
    h, w = shape
    return ChartSpec(name, shape, origin, pitch, (-0.5, -0.5, origin[0] + pitch[0] * (w - 0.5),
                                                    origin[1] + pitch[1] * (h - 0.5)))


def test_real_raw_and_processor_charts_keep_the_declared_centers() -> None:
    raw = build_raw_charts((336, 336))
    assert raw["raw_high"].origin_xy == (0.0, 0.0)
    assert raw["raw_high"].pitch_xy == (4.0, 4.0)
    assert raw["descriptor_high"].origin_xy == (1.5, 1.5)
    assert raw["descriptor_mid"].origin_xy == (3.5, 3.5)

    dino = build_dino_charts(
        (336, 336), resized_hw=(256, 256), crop_hw=(224, 224),
        crop_offset_yx=(16, 16), patch_hw=(14, 14), pooled_hw=(8, 8),
    )
    assert dino["processor_crop"].origin_xy == pytest.approx((21.15625, 21.15625))
    assert dino["dino_patch"].origin_xy == pytest.approx((29.6875, 29.6875))
    assert dino["dino_patch"].pitch_xy == pytest.approx((18.375, 18.375))
    assert dino["dino_pooled"].origin_xy == pytest.approx((38.875, 38.875))
    assert dino["dino_pooled"].pitch_xy == pytest.approx((36.75, 36.75))
    metadata = dino["dino_pooled"].to_dict()
    assert metadata["center_domain_xyxy"] == pytest.approx([38.875, 38.875, 296.125, 296.125])
    assert ChartSpec.from_dict(metadata) == dino["dino_pooled"]


def test_chart_builders_generalize_to_non_square_runtime_dimensions() -> None:
    raw = build_raw_charts((240, 320))
    assert raw["raw_high"].shape_hw == (60, 80)
    assert raw["raw_mid"].shape_hw == (30, 40)
    assert raw["descriptor_high"].origin_xy == pytest.approx((1.5, 1.5))
    dino = build_dino_charts(
        (240, 320), resized_hw=(256, 341), crop_hw=(224, 224),
        crop_offset_yx=(16, 58), patch_hw=(14, 14), pooled_hw=(8, 8),
    )
    assert dino["processor_crop"].shape_hw == (224, 224)
    assert dino["dino_patch"].shape_hw == (16, 16)
    assert dino["dino_patch"].pitch_xy == pytest.approx((320 / 341 * 14, 240 / 256 * 14))


def test_point_vector_variance_and_covariance_are_distinct_affine_operations() -> None:
    source = _chart("source", (5, 5), origin=(2.0, 4.0), pitch=(2.0, 3.0))
    target = _chart("target", (9, 9), origin=(0.0, 0.0), pitch=(1.0, 1.0))
    point = torch.tensor([[1.0, 2.0]])
    assert torch.equal(convert_points(point, source, target), torch.tensor([[4.0, 10.0]]))
    # Origins do not enter a vector conversion.
    assert torch.equal(convert_vectors(torch.tensor([[2.0, 3.0]]), source, target), torch.tensor([[4.0, 9.0]]))
    assert torch.equal(convert_variance(torch.tensor([[2.0, 3.0]]), source, target), torch.tensor([[8.0, 27.0]]))
    covariance = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]])
    expected = torch.tensor([[[8.0, 6.0], [6.0, 27.0]]])
    assert torch.equal(convert_covariance(covariance, source, target), expected)
    assert torch.allclose(convert_points(convert_points(point, source, target), target, source), point)


def test_nonconstant_flow_resampling_moves_source_position_and_converts_units() -> None:
    source = _chart("flow_source", (4, 4))
    target = _chart("flow_target", (2, 2), origin=(0.5, 0.5), pitch=(2.0, 2.0))
    yy, xx = torch.meshgrid(torch.arange(4, dtype=torch.float32), torch.arange(4, dtype=torch.float32), indexing="ij")
    # A spatially varying field catches implementations that only resize the
    # vector or only multiply its units without remapping source positions.
    flow = torch.stack((2.0 + 0.5 * xx, -1.0 + 2.0 * yy), dim=0).unsqueeze(0)
    result = resample_flow(flow, source, target, source_units="index", target_units="canonical")
    expected = torch.tensor([[[[2.25, 3.25], [2.25, 3.25]], [[0.0, 0.0], [4.0, 4.0]]]])
    assert torch.allclose(result.values, expected, atol=1e-5)
    assert bool(result.support.complete.all())


def test_visible_and_interpolation_support_are_separate_and_producer_support_is_preserved() -> None:
    source = _chart("source", (3, 3))
    field = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    points = torch.tensor([[[[0.0, 0.0], [2.0, 2.0], [2.5, 1.0]]]])
    producer = torch.ones((1, 3, 3), dtype=torch.float32)
    result = sample_field(field, source, points, source_valid=producer)
    # The last point is outside both domains. Padding must not turn it into
    # support, while the first two are valid centers.
    assert result.support.visible.tolist() == [[[True, True, False]]]
    assert result.support.center.tolist() == [[[True, True, False]]]
    assert result.support.kernel_weight.shape == (1, 1, 3)
    assert float(result.support.producer_weight[0, 0, 2]) < 1.0
    assert not bool(result.support.complete[0, 0, 2])
    with pytest.raises(ValueError, match=r"weights in \[0,1\]"):
        sample_field(field, source, points, source_valid=torch.full((1, 3, 3), 2.0))


def test_producer_support_is_kept_independently_for_each_batch_source() -> None:
    source = _chart("source", (2, 2))
    field = torch.ones((2, 1, 2, 2), dtype=torch.float32)
    points = torch.tensor([[[[0.0, 0.0]]], [[[0.0, 0.0]]]])
    producer = torch.stack((torch.ones((2, 2)), torch.zeros((2, 2))))
    result = sample_field(field, source, points, source_valid=producer)
    assert bool(result.support.producer_weight[0, 0, 0])
    assert not bool(result.support.producer_weight[1, 0, 0])
    assert bool(result.support.complete[0, 0, 0])
    assert not bool(result.support.complete[1, 0, 0])


def test_sampling_is_float32_differentiable_and_nonfinite_points_are_quarantined() -> None:
    source = _chart("source", (3, 3))
    field = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_()
    points = torch.tensor([[[[1.0, 1.0], [float("nan"), 0.0]]]], requires_grad=True)
    result = sample_field(field, source, points)
    assert result.values.dtype == torch.float32
    assert float(result.values[0, 0, 0, 0].detach()) == 4.0
    assert float(result.values[0, 0, 0, 1].detach()) == 0.0
    assert not bool(result.support.center[0, 0, 1])
    result.values.sum().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_non_affine_adaptive_pooling_fails_closed() -> None:
    source = _chart("source", (6, 6))
    with pytest.raises(ValueError, match="non-affine"):
        area_pool_chart(source, (4, 4), "bad_pool")
