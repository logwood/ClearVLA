from __future__ import annotations

import math

import pytest
import torch

from clearvla.mainline.model.v120_p1 import (
    _FunctionalOwnershipLocalRefiner,
    _UtilityPrecisionLocalRefiner,
)


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.randn(2, 2, 9, 3, requires_grad=True),
        "learned_detail": torch.randn(2, 2, 9, 12, requires_grad=True),
        "coordinates": torch.randn(2, 2, 9, 2, requires_grad=True),
        **{
            name: torch.randn(2, 2, 8, requires_grad=True)
            for name in ("query", "semantic", "appearance", "geometry")
        },
        "future_transport": torch.randn(2, 2, 5, requires_grad=True),
    }


def _refiner() -> _UtilityPrecisionLocalRefiner:
    return _UtilityPrecisionLocalRefiner(width=16, raw_dim=12, route_dim=8, depth=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_neutral_owners_and_empty_pixels_are_exactly_neutral(dtype: torch.dtype) -> None:
    torch.manual_seed(8201)
    refiner, inputs = _refiner(), _inputs()
    records = []
    hook = refiner.delta_router.register_forward_pre_hook(lambda _m, args: records.append(args[:2]))
    try:
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            refiner(**inputs, collect_diagnostics=False)
            neutral = dict(inputs)
            for name in ("semantic", "appearance", "geometry", "future_transport"):
                neutral[name] = torch.zeros_like(inputs[name])
            output, _ = refiner(**neutral, collect_diagnostics=False)
            zero, _ = refiner(
                **{
                    **inputs,
                    "rgb": torch.zeros_like(inputs["rgb"]),
                    "learned_detail": torch.zeros_like(inputs["learned_detail"]),
                },
                collect_diagnostics=False,
            )
    finally:
        hook.remove()
    # Optional selectors cannot alter the protected policy/base read.
    assert torch.equal(records[0][0], records[1][0])
    assert torch.count_nonzero(records[1][1]) == 0
    assert torch.equal(output, records[1][0])
    assert torch.count_nonzero(zero) == 0


def test_p2_lane_differences_are_relative_and_keep_modality_ownership() -> None:
    reads = torch.randn(2, 2, 2, 5, 16, requires_grad=True)
    result = _UtilityPrecisionLocalRefiner._typed_innovation_reads(reads)
    expected = {
        "semantic": reads[:, :, 1, 1] - reads[:, :, 1, 0],
        "appearance": reads[:, :, 0, 2] - reads[:, :, 0, 0],
        "geometry": (
            (reads[:, :, 0, 3] - reads[:, :, 0, 0]) + (reads[:, :, 1, 3] - reads[:, :, 1, 0])
        )
        / math.sqrt(2),
        "horizon": (
            (reads[:, :, 0, 4] - reads[:, :, 0, 0]) - (reads[:, :, 1, 4] - reads[:, :, 1, 0])
        )
        / math.sqrt(2),
    }
    for name in result:
        assert torch.equal(result[name], expected[name])
    shared = torch.randn(2, 2, 2, 1, 16).expand_as(reads)
    assert all(
        torch.count_nonzero(value) == 0
        for value in _UtilityPrecisionLocalRefiner._typed_innovation_reads(shared).values()
    )
    semantic_gradient = torch.autograd.grad(result["semantic"].sum(), reads)[0]
    assert torch.count_nonzero(semantic_gradient[:, :, 0]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2_every_typed_owner_has_finite_vjp_without_more_parameters_or_reads(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(8202)
    refiner, inputs = _refiner(), _inputs()
    parent = _FunctionalOwnershipLocalRefiner(width=16, raw_dim=12, route_dim=8, depth=2)
    assert {k: v.shape for k, v in refiner.state_dict().items()} == {
        k: v.shape for k, v in parent.state_dict().items()
    }
    calls = []
    hooks = [
        attention.register_forward_hook(lambda *_: calls.append(1))
        for attention in (*refiner.token_attn, refiner.read_attn)
    ]
    try:
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            output, _ = refiner(**inputs, collect_diagnostics=True)
        output.float().square().mean().backward()
    finally:
        for hook in hooks:
            hook.remove()
    assert len(calls) == 3
    assert torch.isfinite(output).all()
    for name, parameter in refiner.named_parameters():
        if name.startswith(
            (
                "owner_conditions.",
                "owner_outputs.",
                "geometry_key.",
                "rgb_value.",
                "detail_value.",
                "coordinate_key.",
            )
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    for name, value in inputs.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        assert value.grad.abs().sum() > 0, name


@pytest.mark.parametrize("zero_lane,owner", [("learned_detail", "semantic"), ("rgb", "appearance")])
def test_p2_cannot_borrow_values_from_the_wrong_modality(zero_lane: str, owner: str) -> None:
    inputs = _inputs()
    inputs[zero_lane] = torch.zeros_like(inputs[zero_lane])
    _, metrics = _refiner()(**inputs, collect_diagnostics=True)
    assert metrics[f"flow_jepa_typed_p2_{owner}_contribution_rms"] == 0
