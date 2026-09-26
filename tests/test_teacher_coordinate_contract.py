import pytest
import torch

from clearvla.mainline.model.teacher import ObjectFutureTeacher
from clearvla.mainline.spatial_geometry import (
    build_dino_charts,
    rgb_chart,
)


def _charts():
    outer = rgb_chart((336, 336))
    dino = build_dino_charts(
        (336, 336),
        resized_hw=(256, 256),
        crop_hw=(224, 224),
        crop_offset_yx=(16, 16),
        patch_hw=(14, 14),
        pooled_hw=(8, 8),
    )["dino_pooled"]
    return outer, dino


def test_teacher_canonical_grid_uses_pooled_centers_in_outer_normalized_frame():
    outer, pooled = _charts()
    teacher = ObjectFutureTeacher(
        content_dim=4,
        key_dim=2,
        coordinate_contract="canonical_rgb_lattice_v1",
        future_chart=pooled,
        outer_chart=outer,
    )
    actual = teacher._future_coordinate_grid(rows=8, columns=8, device=torch.device("cpu"))
    expected = outer.canonical_to_normalized(pooled.lattice())
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
    # The first pooled center is not the endpoint of the outer RGB chart.
    assert not torch.allclose(actual[0, 0], torch.tensor([-1.0, -1.0]))


def test_teacher_legacy_grid_remains_endpoint_normalized():
    teacher = ObjectFutureTeacher(content_dim=4, key_dim=2)
    actual = teacher._future_coordinate_grid(rows=8, columns=8, device=torch.device("cpu"))
    assert torch.equal(actual[0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(actual[-1, -1], torch.tensor([1.0, 1.0]))


def test_teacher_canonical_contract_requires_chart_identity_and_shape():
    with pytest.raises(ValueError, match="future_chart and outer_chart"):
        ObjectFutureTeacher(
            content_dim=4,
            coordinate_contract="canonical_rgb_lattice_v1",
        )
    outer, pooled = _charts()
    teacher = ObjectFutureTeacher(
        content_dim=4,
        coordinate_contract="canonical_rgb_lattice_v1",
        future_chart=pooled,
        outer_chart=outer,
    )
    with pytest.raises(ValueError, match="shape does not match"):
        teacher._future_coordinate_grid(rows=4, columns=4, device=torch.device("cpu"))
