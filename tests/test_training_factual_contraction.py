from __future__ import annotations

import pytest
import torch

try:
    from torch.utils._triton import has_triton
except ImportError:
    def has_triton() -> bool:
        return False

from clearvla.mainline.training.factual_contraction import (
    _factual_rgb_detail_backward_op,
    _factual_rgb_detail_forward_op,
    atomic_factual_rgb_detail_contraction,
    opaque_atomic_factual_rgb_detail_contraction,
)


def _reference(value_weight: torch.Tensor, rgb_detail: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "bqgcijmku,bcijmkv->bqguv",
        value_weight,
        rgb_detail,
    ).float()


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(406)
    value_weight = torch.randn(
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
        generator=generator,
        dtype=dtype,
    )
    rgb_detail = torch.randn(
        2,
        2,
        2,
        2,
        2,
        3,
        4,
        generator=generator,
        dtype=dtype,
    )
    return value_weight, rgb_detail


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_atomic_factual_contraction_matches_einsum_output_and_vjp_exactly(
    dtype: torch.dtype,
) -> None:
    value_weight, rgb_detail = _inputs(dtype)
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    actual_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    expected = _reference(*expected_inputs)
    actual = atomic_factual_rgb_detail_contraction(*actual_inputs)
    probe = torch.randn(
        expected.shape,
        generator=torch.Generator().manual_seed(407),
        dtype=expected.dtype,
    )
    expected_gradients = torch.autograd.grad((expected * probe).sum(), expected_inputs)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), actual_inputs)

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_opaque_atomic_fallback_matches_preferred_contraction_exactly(
    dtype: torch.dtype,
) -> None:
    value_weight, rgb_detail = _inputs(dtype)
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    actual_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    expected = atomic_factual_rgb_detail_contraction(*expected_inputs)
    actual = opaque_atomic_factual_rgb_detail_contraction(*actual_inputs)
    probe = torch.randn(
        expected.shape,
        generator=torch.Generator().manual_seed(408),
        dtype=expected.dtype,
    )
    expected_gradients = torch.autograd.grad((expected * probe).sum(), expected_inputs)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), actual_inputs)

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    )


def test_atomic_factual_contraction_fake_shapes_and_strides_match_real_ops() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    value_weight, rgb_detail = _inputs(torch.bfloat16)
    real_output, real_left, real_right = _factual_rgb_detail_forward_op(
        value_weight,
        rgb_detail,
    )
    real_gradients = _factual_rgb_detail_backward_op(
        real_left,
        real_right,
        torch.empty_like(real_output),
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        2,
    )

    with FakeTensorMode() as mode:
        fake_weight = mode.from_tensor(value_weight)
        fake_detail = mode.from_tensor(rgb_detail)
        fake_output, fake_left, fake_right = _factual_rgb_detail_forward_op(
            fake_weight,
            fake_detail,
        )
        fake_gradients = _factual_rgb_detail_backward_op(
            fake_left,
            fake_right,
            torch.empty_like(fake_output),
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            2,
        )

    for actual, expected in zip(
        (fake_output, fake_left, fake_right, *fake_gradients),
        (real_output, real_left, real_right, *real_gradients),
        strict=True,
    ):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.stride() == expected.stride()


def test_atomic_factual_contraction_is_one_fullgraph_aot_operator() -> None:
    value_weight, rgb_detail = _inputs(torch.float32)
    compiled = torch.compile(
        atomic_factual_rgb_detail_contraction,
        backend="aot_eager",
        dynamic=False,
        fullgraph=True,
    )
    eager_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    compiled_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    expected = atomic_factual_rgb_detail_contraction(*eager_inputs)
    actual = compiled(*compiled_inputs)
    expected_gradients = torch.autograd.grad(expected.square().sum(), eager_inputs)
    actual_gradients = torch.autograd.grad(actual.square().sum(), compiled_inputs)

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="requires CUDA Inductor",
)
def test_atomic_factual_cuda_bf16_inductor_matches_eager_einsum_exactly() -> None:
    value_weight, rgb_detail = (
        value.cuda().detach() for value in _inputs(torch.bfloat16)
    )
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    actual_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    expected = _reference(*expected_inputs)
    compiled = torch.compile(
        atomic_factual_rgb_detail_contraction,
        dynamic=False,
        fullgraph=True,
    )
    actual = compiled(*actual_inputs)
    probe = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), expected_inputs)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), actual_inputs)

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    )
