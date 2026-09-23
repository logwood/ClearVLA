"""Positive numeric graph scales must not admit NaN or infinity."""

from dataclasses import replace

import pytest

from clearvla.mainline.config import BottomConfig


@pytest.mark.parametrize(
    "field",
    [
        "residual_scale_max",
        "residual_scale_init",
        "normalization_floor",
        "ffn_expansion",
        "operator_depth_logit_init",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -0.1])
def test_bottom_scales_require_positive_finite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(BottomConfig(), **{field: value}).validate()


@pytest.mark.parametrize("floor", [0.125, 0.75])
def test_finite_scale_changes_are_not_frozen_to_legacy_defaults(floor: float) -> None:
    replace(BottomConfig(), normalization_floor=floor).validate()
