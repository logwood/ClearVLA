"""Spatial/causal contracts of the saved endpoint predictor, not a new model.

These use synthetic DINO feature charts, not image encoders or robot tasks.
The missing-position negative control modifies only an isolated diagnostic copy.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from clearvla.mainline.instruction_reference import InstructionReference
from clearvla.mainline.model.annotation_goal import EndpointGoalPredictor


def _case(seed: int, side: int = 4) -> tuple[
    EndpointGoalPredictor, InstructionReference, InstructionReference, Tensor
]:
    torch.manual_seed(seed)
    model = EndpointGoalPredictor(
        hidden=32, content_dim=16, state_dim=10, heads=4, camera_names=("top", "wrist")
    ).eval()
    n = side * side
    left = torch.zeros(1, 2, n, 16)
    left[..., 1] = 1.0
    right = left.clone()
    for data, cells in ((left, [0, n - 1]), (right, [side - 1, n - side])):
        data[:, :, cells, 1] = 0.0
        data[:, :, cells, 0] = 1.0
    reference = InstructionReference(
        dino=left,
        state=torch.randn(1, 10),
        observed=torch.ones(1, 2, n, dtype=torch.bool),
        age_steps=torch.tensor([9], dtype=torch.long),
    )
    return model, reference, replace(reference, dino=right), torch.randn(1, 5, 32)


@pytest.mark.parametrize("seed", [0, 71, 313, 27301])
def test_same_content_marginal_and_object_centroid_do_not_erase_layout(seed: int) -> None:
    model, left, right, language = _case(seed)
    with torch.no_grad():
        a = model(reference=left, language=language)
        b = model(reference=right, language=language)
    torch.testing.assert_close(left.dino.mean(-2), right.dino.mean(-2), rtol=0, atol=0)
    # The foreground has exactly the same mass and centroid in both layouts.
    torch.testing.assert_close(
        left.dino[..., 0] @ a.coordinates, right.dino[..., 0] @ a.coordinates, rtol=0, atol=0
    )
    # Do not rely on the trivial raw-scene residual: use the inferred spatial
    # law AND a globally pooled prediction with no per-pixel residual path.
    assert not torch.equal(a.log_probability, b.log_probability)
    assert not torch.equal(a.robot_delta, b.robot_delta)


@pytest.mark.parametrize("seed", [0, 71, 313, 27301])
def test_diagnostic_removal_of_position_loses_global_layout_distinction(seed: int) -> None:
    model, left, right, language = _case(seed)
    no_position = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in no_position.position.parameters():
            parameter.zero_()
        a = no_position(reference=left, language=language)
        b = no_position(reference=right, language=language)
    # Whole-tensor attention may round in a different accumulation order.
    # This fixed FP32 budget is not an empirical threshold fitted to this run.
    scale = float(torch.maximum(a.robot_delta.abs().max(), b.robot_delta.abs().max()))
    torch.testing.assert_close(
        a.robot_delta, b.robot_delta, rtol=0, atol=16 * torch.finfo(torch.float32).eps * scale
    )
    first = model.position[0]
    assert isinstance(first, nn.Linear)
    assert first.weight.count_nonzero() > 0  # real model is untouched


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_and_position_have_real_prediction_gradient_owners(dtype: torch.dtype) -> None:
    model, ref, _, language = _case(313)
    x = ref.dino.clone().requires_grad_()
    first = model.position[0]
    assert isinstance(first, nn.Linear)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        p = model(reference=replace(ref, dino=x), language=language)
        loss = p.log_probability[0, 0, 0, 3] + p.robot_delta.square().sum()
    gradients = torch.autograd.grad(loss, (x, first.weight))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.count_nonzero() > 0


@pytest.mark.parametrize("side", [4, 8, 16])
def test_native_chart_size_does_not_reintroduce_layout_invariance(side: int) -> None:
    model, left, right, language = _case(71, side=side)
    with torch.no_grad():
        a = model(reference=left, language=language)
        b = model(reference=right, language=language)
    n = side * side
    assert a.log_probability.shape == (1, 2, n, n + 1)
    assert not torch.equal(a.robot_delta, b.robot_delta)
    torch.testing.assert_close(
        a.log_probability.exp().sum(-1), torch.ones(1, 2, n), rtol=0, atol=8e-7
    )


def test_instruction_age_is_not_a_spatial_or_progress_input() -> None:
    model, ref, _, language = _case(71)
    with torch.no_grad():
        a = model(reference=ref, language=language)
        b = model(reference=replace(ref, age_steps=ref.age_steps + 1000), language=language)
    for name in ("scene", "log_probability", "robot_delta"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)


def test_unknown_reference_cells_are_quarantined_before_learned_reads() -> None:
    model, ref, _, language = _case(313)
    support = ref.observed.clone()
    support[..., ::2] = False
    payload = ref.dino.clone()
    payload[..., ::2, :] = torch.nan
    ref = replace(ref, dino=payload.requires_grad_(), observed=support)
    prediction = model(reference=ref, language=language)
    assert torch.isfinite(prediction.scene).all()
    assert torch.isfinite(prediction.robot_delta).all()
    assert torch.isneginf(prediction.log_probability[..., ::2, :-1]).all()
    assert torch.equal(
        prediction.log_probability[..., ::2, -1],
        torch.zeros_like(prediction.log_probability[..., ::2, -1]),
    )
    gradient = torch.autograd.grad(prediction.scene.sum(), ref.dino)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[..., ::2, :].count_nonzero() == 0
