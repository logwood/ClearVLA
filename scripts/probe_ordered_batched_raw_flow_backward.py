"""Compare a forward-batched raw-flow sampler with ordered CUDA backward.

This is a laboratory-only probe.  It keeps one grouped ``grid_sample`` in
the forward pass, but its custom backward invokes the stock
``grid_sampler_2d_backward`` once per offset and adds input gradients in the
historical offset order.  The purpose is to measure whether the sparse
parameter tails of the fully batched candidate come from accumulation order
alone, and what speed is left if that order is restored.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
import types
from typing import Iterable

import torch
import torch.nn.functional as F

from clearvla.mainline.v120_core.flow_dino_evidence import (
    _DenseRawFlowRefiner,
    _normalize_grid,
)


class _OrderedBatchedGridSample(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        value: torch.Tensor,
        flat_grid: torch.Tensor,
        batch: int,
        offset_count: int,
    ) -> torch.Tensor:
        _, channels, side, side_b = value.shape
        if side != side_b:
            raise ValueError("ordered sampler requires square values")
        flat_value = (
            value.unsqueeze(1)
            .expand(-1, offset_count, -1, -1, -1)
            .reshape(batch * offset_count, channels, side, side)
        )
        sampled = F.grid_sample(
            flat_value,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        ctx.save_for_backward(value, flat_grid)
        ctx.batch = int(batch)
        ctx.offset_count = int(offset_count)
        return sampled.reshape(batch, offset_count, channels, side, side)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        value, flat_grid = ctx.saved_tensors
        batch = ctx.batch
        offset_count = ctx.offset_count
        grad_flat = grad_output.reshape(
            batch * offset_count,
            grad_output.shape[2],
            grad_output.shape[3],
            grad_output.shape[4],
        )
        grad_value: torch.Tensor | None = None
        grad_grid = torch.empty_like(flat_grid)
        # This is intentionally the same outer order as the historical
        # caller: one stock backward launch per offset, then an ordered add.
        for offset_index in range(offset_count):
            grad_input, grad_coordinates = torch.ops.aten.grid_sampler_2d_backward(
                grad_flat[offset_index::offset_count].contiguous(),
                value,
                flat_grid[offset_index::offset_count].contiguous(),
                0,
                0,
                True,
                (True, True),
            )
            if grad_value is None:
                grad_value = grad_input
            else:
                grad_value = grad_value + grad_input
            grad_grid[offset_index::offset_count] = grad_coordinates
        return grad_value, grad_grid, None, None


class _AutogradOrderedBatchedGridSample(torch.autograd.Function):
    """Same experiment, rebuilding the stock backward graph per offset."""

    @staticmethod
    def forward(ctx, value, flat_grid, batch, offset_count):
        _, channels, side, side_b = value.shape
        flat_value = (
            value.unsqueeze(1)
            .expand(-1, offset_count, -1, -1, -1)
            .reshape(batch * offset_count, channels, side, side_b)
        )
        sampled = F.grid_sample(
            flat_value,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        ctx.save_for_backward(value, flat_grid)
        ctx.batch = int(batch)
        ctx.offset_count = int(offset_count)
        return sampled.reshape(batch, offset_count, channels, side, side_b)

    @staticmethod
    def backward(ctx, grad_output):
        value, flat_grid = ctx.saved_tensors
        batch = ctx.batch
        offset_count = ctx.offset_count
        grad_flat = grad_output.reshape(
            batch * offset_count,
            grad_output.shape[2],
            grad_output.shape[3],
            grad_output.shape[4],
        )
        grad_value = torch.zeros_like(value)
        grad_grid = torch.empty_like(flat_grid)
        # Recreate exactly the primitive operation used by the reference
        # path.  This is intentionally slow and exists only to identify the
        # source of the numerical tail.
        with torch.enable_grad():
            value_live = value.detach().requires_grad_(True)
            for offset_index in range(offset_count):
                grid_live = flat_grid[offset_index::offset_count].detach().requires_grad_(True)
                sampled = F.grid_sample(
                    value_live,
                    grid_live,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                grad_input, grad_coordinates = torch.autograd.grad(
                    sampled,
                    (value_live, grid_live),
                    grad_flat[offset_index::offset_count].contiguous(),
                    retain_graph=False,
                )
                grad_value = grad_value + grad_input
                grad_grid[offset_index::offset_count] = grad_coordinates
        return grad_value, grad_grid, None, None


class _MultiOutputBatchedGridSample(torch.autograd.Function):
    """Return one output per offset so the caller keeps separate branches."""

    @staticmethod
    def forward(ctx, value, flat_grid, batch, offset_count):
        _, channels, side, side_b = value.shape
        flat_value = (
            value.unsqueeze(1)
            .expand(-1, offset_count, -1, -1, -1)
            .reshape(batch * offset_count, channels, side, side_b)
        )
        sampled = F.grid_sample(
            flat_value,
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(batch, offset_count, channels, side, side_b)
        ctx.save_for_backward(value, flat_grid)
        ctx.batch = int(batch)
        ctx.offset_count = int(offset_count)
        return tuple(sampled[:, index] for index in range(offset_count))

    @staticmethod
    def backward(ctx, *grad_outputs):
        value, flat_grid = ctx.saved_tensors
        batch = ctx.batch
        offset_count = ctx.offset_count
        grad_value = torch.zeros_like(value)
        grad_grid = torch.empty_like(flat_grid)
        # Ordered stock backward, now with one independent output branch per
        # offset.  This mirrors the reference caller's list of tensors.
        for offset_index, grad_output in enumerate(grad_outputs):
            grad_input, grad_coordinates = torch.ops.aten.grid_sampler_2d_backward(
                grad_output.contiguous(),
                value,
                flat_grid[offset_index::offset_count].contiguous(),
                0,
                0,
                True,
                (True, True),
            )
            grad_value = grad_value + grad_input
            grad_grid[offset_index::offset_count] = grad_coordinates
        return grad_value, grad_grid, None, None


def _ordered_samples(
    self: _DenseRawFlowRefiner,
    second_feature: torch.Tensor,
    center: torch.Tensor,
    offsets: list[tuple[int, int]],
    search_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, channels, side, side_b = second_feature.shape
    if side != side_b or tuple(center.shape) != (batch, side, side, 2):
        raise ValueError("ordered raw sampling geometry does not align")
    grid_rows = []
    valid_rows = []
    offset_rows = []
    for dx, dy in offsets:
        if self.preserve_uncertain_seed:
            offset = torch.cat(
                (float(dx) * search_scale, float(dy) * search_scale), dim=1
            )
        else:
            offset = second_feature.new_tensor((float(dx), float(dy)))[None, :, None, None]
            offset = offset.expand(batch, -1, side, side)
        coordinates = center + offset.permute(0, 2, 3, 1)
        valid_rows.append(
            (coordinates[..., 0] >= 0.0)
            & (coordinates[..., 0] <= float(side - 1))
            & (coordinates[..., 1] >= 0.0)
            & (coordinates[..., 1] <= float(side - 1))
        )
        offset_rows.append(offset)
        grid_rows.append(_normalize_grid(coordinates, side, side))
    valid = torch.stack(valid_rows, dim=1)
    offset_field = torch.stack(offset_rows, dim=1)
    flat_grid = torch.stack(grid_rows, dim=1).reshape(
        batch * len(offsets), side, side, 2
    )
    with torch.autocast(device_type=second_feature.device.type, enabled=False):
        sampled_outputs = _MultiOutputBatchedGridSample.apply(
            second_feature.float(),
            flat_grid.float(),
            batch,
            len(offsets),
        )
        sampled = torch.stack(sampled_outputs, dim=1)
    return sampled, valid, offset_field


def _outputs(estimate: object) -> tuple[torch.Tensor, ...]:
    values = [
        estimate.flow,
        estimate.information,
        estimate.uncertainty,
        estimate.correlation_entropy,
        estimate.correlation_margin,
        *estimate.iterations,
    ]
    if estimate.boundary_compression is not None:
        values.append(estimate.boundary_compression)
    return tuple(values)


def _inputs(
    batch: int, channels: int, side: int, coarse_side: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(9171)
    return (
        torch.randn(batch, channels, side, side, device=device, generator=generator),
        torch.randn(batch, channels, side, side, device=device, generator=generator),
        torch.randn(batch, 2, coarse_side, coarse_side, device=device, generator=generator),
        torch.rand(batch, 1, coarse_side, coarse_side, device=device, generator=generator),
    )


def _run(
    module: _DenseRawFlowRefiner, inputs: Iterable[torch.Tensor]
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    module.zero_grad(set_to_none=True)
    live = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    outputs = _outputs(module(*live))
    loss = sum(x.float().square().mean() for x in outputs)
    loss.backward()
    input_grads = tuple(
        x.grad.detach().clone() if x.grad is not None else torch.zeros_like(x)
        for x in live
    )
    parameter_grads = tuple(
        p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
        for p in module.parameters()
    )
    return (
        tuple(x.detach().clone() for x in outputs),
        input_grads,
        parameter_grads,
    )


def _max_delta(actual: Iterable[torch.Tensor], expected: Iterable[torch.Tensor]) -> tuple[float, float]:
    max_abs = 0.0
    rel_l2 = 0.0
    for a, e in zip(actual, expected, strict=True):
        delta = (a.float() - e.float()).abs()
        max_abs = max(max_abs, float(delta.max().item()))
        rel_l2 = max(rel_l2, float(delta.norm().div(e.float().norm().clamp_min(1e-30)).item()))
    return max_abs, rel_l2


def _timed(module: _DenseRawFlowRefiner, inputs: tuple[torch.Tensor, ...], warmup: int, steps: int) -> float:
    values: list[float] = []
    for index in range(warmup + steps):
        started = time.perf_counter()
        _run(module, inputs)
        torch.cuda.synchronize(inputs[0].device)
        if index >= warmup:
            values.append(time.perf_counter() - started)
    return statistics.median(values) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--coarse-side", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda")
    reference = _DenseRawFlowRefiner(
        args.channels,
        args.hidden,
        radius=args.radius,
        uncertainty_floor=0.03,
        activation_checkpoint=True,
        preserve_uncertain_seed=True,
        bounded_coordinates=True,
        normalization_floor=0.10,
    ).to(device).train()
    batched = copy.deepcopy(reference)
    batched._batched_offset_sampling = True
    ordered = copy.deepcopy(reference)
    ordered._batched_samples = types.MethodType(_ordered_samples, ordered)
    ordered._batched_offset_sampling = True
    inputs = _inputs(args.batch, args.channels, args.side, args.coarse_side, device)
    expected = _run(reference, inputs)
    current = _run(batched, inputs)
    strict = _run(ordered, inputs)
    print("current_output_max_abs_rel_l2", _max_delta(current[0], expected[0]))
    print("current_input_grad_max_abs_rel_l2", _max_delta(current[1], expected[1]))
    print("current_parameter_grad_max_abs_rel_l2", _max_delta(current[2], expected[2]))
    print("ordered_output_max_abs_rel_l2", _max_delta(strict[0], expected[0]))
    print("ordered_input_grad_max_abs_rel_l2", _max_delta(strict[1], expected[1]))
    print("ordered_parameter_grad_max_abs_rel_l2", _max_delta(strict[2], expected[2]))
    reference_ms = _timed(reference, inputs, args.warmup, args.steps)
    current_ms = _timed(batched, inputs, args.warmup, args.steps)
    ordered_ms = _timed(ordered, inputs, args.warmup, args.steps)
    print(
        "timing_ms",
        {"reference": reference_ms, "batched": current_ms, "ordered_backward": ordered_ms},
    )
    print("speedup", {"batched": reference_ms / current_ms, "ordered_backward": reference_ms / ordered_ms})


if __name__ == "__main__":
    main()
