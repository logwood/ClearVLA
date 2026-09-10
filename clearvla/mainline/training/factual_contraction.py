"""Exact-order contractions for the opt-in factual training path.

The factual reader's normalized RGB/detail reduction is numerically sensitive:
letting Inductor see the complete einsum can reassociate its BF16 reduction.
This module fixes that order in two forms.  The original opaque implementation
keeps matrix construction and explicit first-order backward inside custom
operators.  The preferred implementation exposes only the layout transforms
and an exact-order ATen BMM to the surrounding compiled factual shell.  That
lets Inductor fuse copies with their producers without seeing an einsum that
it could reassociate.

Nothing imports or installs this path during ordinary model construction.  A
version-owned training acceleration adapter must opt one model instance in.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _validate_inputs(value_weight: Tensor, rgb_detail: Tensor) -> None:
    if value_weight.ndim != 9:
        raise ValueError(
            "factual value weights must have layout "
            "[B,Q,G,C,I,J,M,K,U]"
        )
    if rgb_detail.ndim != 7:
        raise ValueError(
            "factual RGB/detail values must have layout [B,C,I,J,M,K,V]"
        )
    batch, _query, _glimpse, cameras, grid_y, grid_x, slots, candidates, _micro = (
        value_weight.shape
    )
    if tuple(rgb_detail.shape[:-1]) != (
        batch,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
    ):
        raise ValueError(
            "factual value weights and RGB/detail values have incompatible "
            "state layouts"
        )
    if value_weight.device != rgb_detail.device:
        raise ValueError("factual contraction inputs must share one device")
    if value_weight.dtype != rgb_detail.dtype:
        raise ValueError("factual contraction inputs must share one dtype")


def _forward_matrices(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> tuple[Tensor, Tensor]:
    (
        batch,
        query,
        glimpse,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
        micro,
    ) = value_weight.shape
    values = rgb_detail.shape[-1]
    states = cameras * grid_y * grid_x * candidates * slots

    # PyTorch's two-input einsum lowers to exactly these matrices.  In
    # particular K precedes M in the flattened reduction axis.  Flattening M
    # before K is mathematically associative in real arithmetic but changes
    # BF16 accumulation and was the source of the rejected compiled result.
    left = (
        value_weight.permute(0, 1, 2, 8, 3, 4, 5, 7, 6)
        .clone(memory_format=torch.contiguous_format)
        .view(batch, query * glimpse * micro, states)
    )
    right = (
        rgb_detail.permute(0, 1, 2, 3, 5, 4, 6)
        .clone(memory_format=torch.contiguous_format)
        .view(batch, states, values)
    )
    return left, right


def _compiler_visible_forward_matrices(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build the same matrices while leaving their copies compiler-visible."""

    (
        batch,
        query,
        glimpse,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
        micro,
    ) = value_weight.shape
    values = rgb_detail.shape[-1]
    states = cameras * grid_y * grid_x * candidates * slots
    left = (
        value_weight.permute(0, 1, 2, 8, 3, 4, 5, 7, 6)
        .contiguous()
        .view(batch, query * glimpse * micro, states)
    )
    right = (
        rgb_detail.permute(0, 1, 2, 3, 5, 4, 6)
        .contiguous()
        .view(batch, states, values)
    )
    return left, right


@torch.library.custom_op(
    "clearvla_training::factual_rgb_detail_backward",
    mutates_args=(),
)
def _factual_rgb_detail_backward_op(
    left: Tensor,
    right: Tensor,
    output_gradient: Tensor,
    query: int,
    glimpse: int,
    cameras: int,
    grid_y: int,
    grid_x: int,
    slots: int,
    candidates: int,
    micro: int,
) -> tuple[Tensor, Tensor]:
    batch = left.shape[0]
    values = right.shape[-1]
    gradient = output_gradient.to(dtype=left.dtype).reshape(
        batch,
        query * glimpse * micro,
        values,
    )
    right_gradient = torch.bmm(left.transpose(1, 2), gradient)
    left_gradient = torch.bmm(gradient, right.transpose(1, 2))
    value_weight_gradient = left_gradient.reshape(
        batch,
        query,
        glimpse,
        micro,
        cameras,
        grid_y,
        grid_x,
        candidates,
        slots,
    ).permute(0, 1, 2, 4, 5, 6, 8, 7, 3)
    rgb_detail_gradient = right_gradient.reshape(
        batch,
        cameras,
        grid_y,
        grid_x,
        candidates,
        slots,
        values,
    ).permute(0, 1, 2, 3, 5, 4, 6)
    return value_weight_gradient, rgb_detail_gradient


@_factual_rgb_detail_backward_op.register_fake
def _factual_rgb_detail_backward_fake(
    left: Tensor,
    right: Tensor,
    output_gradient: Tensor,
    query: int,
    glimpse: int,
    cameras: int,
    grid_y: int,
    grid_x: int,
    slots: int,
    candidates: int,
    micro: int,
) -> tuple[Tensor, Tensor]:
    del output_gradient
    batch = left.shape[0]
    values = right.shape[-1]
    value_weight_gradient = left.new_empty(
        (
            batch,
            query,
            glimpse,
            micro,
            cameras,
            grid_y,
            grid_x,
            candidates,
            slots,
        )
    ).permute(0, 1, 2, 4, 5, 6, 8, 7, 3)
    rgb_detail_gradient = right.new_empty(
        (
            batch,
            cameras,
            grid_y,
            grid_x,
            candidates,
            slots,
            values,
        )
    ).permute(0, 1, 2, 3, 5, 4, 6)
    return value_weight_gradient, rgb_detail_gradient


@torch.library.custom_op(
    "clearvla_training::factual_rgb_detail_forward",
    mutates_args=(),
)
def _factual_rgb_detail_forward_op(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_inputs(value_weight, rgb_detail)
    batch, query, glimpse = value_weight.shape[:3]
    micro = value_weight.shape[-1]
    values = rgb_detail.shape[-1]
    left, right = _forward_matrices(value_weight, rgb_detail)
    output = torch.bmm(left, right).reshape(
        batch,
        query,
        glimpse,
        micro,
        values,
    )
    return output.float(), left, right


@_factual_rgb_detail_forward_op.register_fake
def _factual_rgb_detail_forward_fake(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    (
        batch,
        query,
        glimpse,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
        micro,
    ) = value_weight.shape
    values = rgb_detail.shape[-1]
    states = cameras * grid_y * grid_x * candidates * slots
    output = value_weight.new_empty(
        (batch, query, glimpse, micro, values),
        dtype=torch.float32,
    )
    left = value_weight.new_empty(
        (batch, query * glimpse * micro, states)
    )
    right = rgb_detail.new_empty((batch, states, values))
    return output, left, right


def _factual_rgb_detail_setup_context(
    ctx: Any,
    inputs: tuple[Tensor, Tensor],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    value_weight, _rgb_detail = inputs
    _result, left, right = output
    ctx.value_weight_shape = tuple(value_weight.shape)
    ctx.save_for_backward(left, right)
    ctx.mark_non_differentiable(left, right)
    ctx.set_materialize_grads(False)


def _factual_rgb_detail_autograd(
    ctx: Any,
    output_gradient: Tensor | None,
    left_gradient: Tensor | None,
    right_gradient: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    if left_gradient is not None or right_gradient is not None:
        raise RuntimeError("saved factual contraction matrices are private outputs")
    if output_gradient is None:
        return None, None
    left, right = ctx.saved_tensors
    (
        _batch,
        query,
        glimpse,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
        micro,
    ) = ctx.value_weight_shape
    return _factual_rgb_detail_backward_op(
        left,
        right,
        output_gradient,
        query,
        glimpse,
        cameras,
        grid_y,
        grid_x,
        slots,
        candidates,
        micro,
    )


_factual_rgb_detail_forward_op.register_autograd(
    _factual_rgb_detail_autograd,
    setup_context=_factual_rgb_detail_setup_context,
)


def opaque_atomic_factual_rgb_detail_contraction(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    """Run the eager-equivalent contraction behind one opaque logical op."""

    output, _left, _right = _factual_rgb_detail_forward_op(
        value_weight,
        rgb_detail,
    )
    return output


def atomic_factual_rgb_detail_contraction(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    """Run exact-order BMM with compiler-visible layout transformations.

    The explicit K-before-M flattening is identical to eager ``einsum``.  BMM
    remains the reduction primitive, so Inductor can fuse only the surrounding
    layout/cast work rather than rewriting the sensitive BF16 contraction.
    """

    _validate_inputs(value_weight, rgb_detail)
    batch, query, glimpse = value_weight.shape[:3]
    micro = value_weight.shape[-1]
    values = rgb_detail.shape[-1]
    left, right = _compiler_visible_forward_matrices(
        value_weight,
        rgb_detail,
    )
    return torch.bmm(left, right).reshape(
        batch,
        query,
        glimpse,
        micro,
        values,
    ).float()


__all__ = [
    "atomic_factual_rgb_detail_contraction",
    "opaque_atomic_factual_rgb_detail_contraction",
]
