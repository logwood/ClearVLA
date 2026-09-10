"""Profile exact-order factual contraction layouts on CUDA.

This laboratory probe compares the production custom operator with an
equivalent boundary that exposes the two matrix-layout transforms to
``torch.compile``.  The GEMM and its two VJPs remain opaque and preserve the
same BF16 reduction order.  No model source is patched by this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from clearvla.mainline.training.factual_contraction import (
    opaque_atomic_factual_rgb_detail_contraction as atomic_factual_rgb_detail_contraction,
)


def reference_contraction(value_weight: Tensor, rgb_detail: Tensor) -> Tensor:
    return torch.einsum(
        "bqgcijmku,bcijmkv->bqguv",
        value_weight,
        rgb_detail,
    ).float()


def _visible_matrices(
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
    "clearvla_training_lab::factual_matrix_backward",
    mutates_args=(),
)
def _matrix_backward(
    left: Tensor,
    right: Tensor,
    output_gradient: Tensor,
) -> tuple[Tensor, Tensor]:
    gradient = output_gradient.to(dtype=left.dtype)
    return (
        torch.bmm(gradient, right.transpose(1, 2)),
        torch.bmm(left.transpose(1, 2), gradient),
    )


@_matrix_backward.register_fake
def _matrix_backward_fake(
    left: Tensor,
    right: Tensor,
    output_gradient: Tensor,
) -> tuple[Tensor, Tensor]:
    del output_gradient
    return left.new_empty(left.shape), right.new_empty(right.shape)


@torch.library.custom_op(
    "clearvla_training_lab::factual_matrix_forward",
    mutates_args=(),
)
def _matrix_forward(left: Tensor, right: Tensor) -> Tensor:
    return torch.bmm(left, right).float()


@_matrix_forward.register_fake
def _matrix_forward_fake(left: Tensor, right: Tensor) -> Tensor:
    return left.new_empty(
        (left.shape[0], left.shape[1], right.shape[2]),
        dtype=torch.float32,
    )


def _matrix_setup_context(
    ctx: Any,
    inputs: tuple[Tensor, Tensor],
    output: Tensor,
) -> None:
    del output
    ctx.save_for_backward(*inputs)
    ctx.set_materialize_grads(False)


def _matrix_autograd(
    ctx: Any,
    output_gradient: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    if output_gradient is None:
        return None, None
    left, right = ctx.saved_tensors
    return _matrix_backward(left, right, output_gradient)


_matrix_forward.register_autograd(
    _matrix_autograd,
    setup_context=_matrix_setup_context,
)


def visible_matrix_contraction(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    batch, query, glimpse = value_weight.shape[:3]
    micro = value_weight.shape[-1]
    values = rgb_detail.shape[-1]
    left, right = _visible_matrices(value_weight, rgb_detail)
    return _matrix_forward(left, right).reshape(
        batch,
        query,
        glimpse,
        micro,
        values,
    )


def visible_native_bmm_contraction(
    value_weight: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    """Expose layout and ATen BMM while preserving the eager K-before-M axis."""

    batch, query, glimpse = value_weight.shape[:3]
    micro = value_weight.shape[-1]
    values = rgb_detail.shape[-1]
    left, right = _visible_matrices(value_weight, rgb_detail)
    return torch.bmm(left, right).reshape(
        batch,
        query,
        glimpse,
        micro,
        values,
    ).float()


def _full_contraction(
    contraction: Callable[[Tensor, Tensor], Tensor],
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    local_weight = fine[..., None] * basis
    local_weight = local_weight / local_weight.sum(
        dim=-2,
        keepdim=True,
    ).clamp_min(1e-8)
    joint_weight = route[..., None, None] * local_weight
    return contraction(joint_weight.to(dtype=rgb_detail.dtype), rgb_detail)


def current_full(
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    return _full_contraction(
        atomic_factual_rgb_detail_contraction,
        route,
        fine,
        basis,
        rgb_detail,
    )


def visible_full(
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    return _full_contraction(
        visible_matrix_contraction,
        route,
        fine,
        basis,
        rgb_detail,
    )


def native_bmm_full(
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    return _full_contraction(
        visible_native_bmm_contraction,
        route,
        fine,
        basis,
        rgb_detail,
    )


def reference_full(
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
) -> Tensor:
    return _full_contraction(
        reference_contraction,
        route,
        fine,
        basis,
        rgb_detail,
    )


def _delta(actual: Tensor, expected: Tensor) -> dict[str, float | bool]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().norm().clamp_min(1e-30)
    return {
        "bit_exact": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.max().item()),
        "relative_l2": float(difference.norm().div(denominator).item()),
    }


def _compare_leaf_pair(
    candidate: Callable[[Tensor, Tensor], Tensor],
    expected: Callable[[Tensor, Tensor], Tensor],
    value_weight: Tensor,
    rgb_detail: Tensor,
    probe: Tensor,
) -> dict[str, object]:
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    candidate_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (value_weight, rgb_detail)
    )
    expected_output = expected(*expected_inputs)
    candidate_output = candidate(*candidate_inputs)
    expected_gradients = torch.autograd.grad(
        (expected_output * probe).sum(),
        expected_inputs,
    )
    candidate_gradients = torch.autograd.grad(
        (candidate_output * probe).sum(),
        candidate_inputs,
    )
    return {
        "output": _delta(candidate_output, expected_output),
        "value_weight_gradient": _delta(
            candidate_gradients[0], expected_gradients[0]
        ),
        "rgb_detail_gradient": _delta(
            candidate_gradients[1], expected_gradients[1]
        ),
    }


def _compare_full_pair(
    candidate: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    expected: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
    probe: Tensor,
) -> dict[str, object]:
    expected_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (route, fine, rgb_detail)
    )
    candidate_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        for value in (route, fine, rgb_detail)
    )
    expected_output = expected(
        expected_inputs[0], expected_inputs[1], basis, expected_inputs[2]
    )
    candidate_output = candidate(
        candidate_inputs[0], candidate_inputs[1], basis, candidate_inputs[2]
    )
    expected_gradients = torch.autograd.grad(
        (expected_output * probe).sum(),
        expected_inputs,
    )
    candidate_gradients = torch.autograd.grad(
        (candidate_output * probe).sum(),
        candidate_inputs,
    )
    return {
        "output": _delta(candidate_output, expected_output),
        "route_gradient": _delta(
            candidate_gradients[0], expected_gradients[0]
        ),
        "fine_gradient": _delta(candidate_gradients[1], expected_gradients[1]),
        "rgb_detail_gradient": _delta(
            candidate_gradients[2], expected_gradients[2]
        ),
    }


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "minimum_ms": min(ordered),
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.fmean(ordered)),
        "maximum_ms": max(ordered),
    }


def _timed_leaf(
    function: Callable[[Tensor, Tensor], Tensor],
    value_weight: Tensor,
    rgb_detail: Tensor,
    *,
    warmup: int,
    steps: int,
) -> dict[str, dict[str, float]]:
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    total_samples: list[float] = []
    for index in range(warmup + steps):
        left = value_weight.detach().clone().requires_grad_(True)
        right = rgb_detail.detach().clone().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        forward_stop = torch.cuda.Event(enable_timing=True)
        backward_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = function(left, right)
        forward_stop.record()
        output.square().mean().backward()
        backward_stop.record()
        torch.cuda.synchronize()
        if index >= warmup:
            forward_samples.append(float(start.elapsed_time(forward_stop)))
            backward_samples.append(float(forward_stop.elapsed_time(backward_stop)))
            total_samples.append(float(start.elapsed_time(backward_stop)))
    return {
        "forward": _summary(forward_samples),
        "backward": _summary(backward_samples),
        "total": _summary(total_samples),
    }


def _timed_full(
    function: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    route: Tensor,
    fine: Tensor,
    basis: Tensor,
    rgb_detail: Tensor,
    *,
    warmup: int,
    steps: int,
) -> dict[str, dict[str, float]]:
    forward_samples: list[float] = []
    backward_samples: list[float] = []
    total_samples: list[float] = []
    for index in range(warmup + steps):
        route_step = route.detach().clone().requires_grad_(True)
        fine_step = fine.detach().clone().requires_grad_(True)
        values_step = rgb_detail.detach().clone().requires_grad_(True)
        start = torch.cuda.Event(enable_timing=True)
        forward_stop = torch.cuda.Event(enable_timing=True)
        backward_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = function(route_step, fine_step, basis, values_step)
        forward_stop.record()
        output.square().mean().backward()
        backward_stop.record()
        torch.cuda.synchronize()
        if index >= warmup:
            forward_samples.append(float(start.elapsed_time(forward_stop)))
            backward_samples.append(float(forward_stop.elapsed_time(backward_stop)))
            total_samples.append(float(start.elapsed_time(backward_stop)))
    return {
        "forward": _summary(forward_samples),
        "backward": _summary(backward_samples),
        "total": _summary(total_samples),
    }


def _time_components(
    value_weight: Tensor,
    rgb_detail: Tensor,
    output_gradient: Tensor,
    *,
    warmup: int,
    steps: int,
) -> dict[str, dict[str, float]]:
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
    sample_rows: dict[str, list[float]] = {
        "left_pack": [],
        "right_pack": [],
        "forward_bmm": [],
        "left_vjp_bmm": [],
        "right_vjp_bmm": [],
    }
    for index in range(warmup + steps):
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        events[0].record()
        left = (
            value_weight.permute(0, 1, 2, 8, 3, 4, 5, 7, 6)
            .contiguous()
            .view(batch, query * glimpse * micro, states)
        )
        events[1].record()
        right = (
            rgb_detail.permute(0, 1, 2, 3, 5, 4, 6)
            .contiguous()
            .view(batch, states, values)
        )
        events[2].record()
        output = torch.bmm(left, right)
        events[3].record()
        gradient = output_gradient.to(dtype=left.dtype)
        left_gradient = torch.bmm(gradient, right.transpose(1, 2))
        events[4].record()
        right_gradient = torch.bmm(left.transpose(1, 2), gradient)
        events[5].record()
        torch.cuda.synchronize()
        del output, left_gradient, right_gradient
        if index >= warmup:
            for name, start_index in (
                ("left_pack", 0),
                ("right_pack", 1),
                ("forward_bmm", 2),
                ("left_vjp_bmm", 3),
                ("right_vjp_bmm", 4),
            ):
                sample_rows[name].append(
                    float(events[start_index].elapsed_time(events[start_index + 1]))
                )
    return {name: _summary(samples) for name, samples in sample_rows.items()}


def _compile_options() -> dict[str, bool]:
    return {
        "deterministic": True,
        "force_same_precision": True,
        "emulate_precision_casts": True,
        "fallback_random": True,
        "use_fast_math": False,
        "cuda.use_fast_math": False,
        "split_reductions": False,
        "triton.cooperative_reductions": False,
        "triton.force_cooperative_reductions": False,
        "triton.mix_order_reduction": False,
        "triton.persistent_reductions": False,
        "triton.tile_reductions": False,
        "epilogue_fusion": False,
        "prologue_fusion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    batch = int(args.batch_size)
    query = int(args.query)
    glimpse = 4
    cameras = 2
    grid = 8
    slots = 4
    candidates = 49
    micro = 3
    value_width = 51

    route = torch.softmax(
        torch.randn(
            batch,
            query,
            glimpse,
            cameras,
            grid,
            grid,
            slots,
            device=device,
            dtype=torch.float32,
        ).flatten(3),
        dim=-1,
    ).reshape(batch, query, glimpse, cameras, grid, grid, slots)
    fine = torch.softmax(
        torch.randn(
            *route.shape,
            candidates,
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )
    basis = torch.rand(
        candidates,
        micro,
        device=device,
        dtype=torch.float32,
    )
    rgb_detail = torch.randn(
        batch,
        cameras,
        grid,
        grid,
        slots,
        candidates,
        value_width,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        local_weight = fine[..., None] * basis
        local_weight = local_weight / local_weight.sum(
            dim=-2, keepdim=True
        ).clamp_min(1e-8)
        value_weight = (route[..., None, None] * local_weight).to(
            dtype=torch.bfloat16
        )
    output_gradient = torch.randn(
        batch,
        query * glimpse * micro,
        value_width,
        device=device,
        dtype=torch.float32,
    )

    compiled_current = torch.compile(
        current_full,
        dynamic=False,
        fullgraph=False,
        options=_compile_options(),
    )
    compiled_visible = torch.compile(
        visible_full,
        dynamic=False,
        fullgraph=False,
        options=_compile_options(),
    )
    compiled_native_bmm = torch.compile(
        native_bmm_full,
        dynamic=False,
        fullgraph=False,
        options=_compile_options(),
    )

    comparison_batch = min(batch, 2)
    comparison_query = min(query, 2)
    compare_weight = value_weight[:comparison_batch, :comparison_query]
    compare_values = rgb_detail[:comparison_batch]
    leaf_probe = torch.randn(
        comparison_batch,
        comparison_query,
        glimpse,
        micro,
        value_width,
        device=device,
        dtype=torch.float32,
    )
    full_route = route[:comparison_batch, :comparison_query]
    full_fine = fine[:comparison_batch, :comparison_query]

    comparisons = {
        "visible_vs_reference_leaf": _compare_leaf_pair(
            visible_matrix_contraction,
            reference_contraction,
            compare_weight,
            compare_values,
            leaf_probe,
        ),
        "visible_vs_current_leaf": _compare_leaf_pair(
            visible_matrix_contraction,
            atomic_factual_rgb_detail_contraction,
            compare_weight,
            compare_values,
            leaf_probe,
        ),
        "native_bmm_vs_current_leaf": _compare_leaf_pair(
            visible_native_bmm_contraction,
            atomic_factual_rgb_detail_contraction,
            compare_weight,
            compare_values,
            leaf_probe,
        ),
        "compiled_visible_vs_compiled_current_full": _compare_full_pair(
            compiled_visible,
            compiled_current,
            full_route,
            full_fine,
            basis,
            compare_values,
            leaf_probe,
        ),
        "compiled_native_bmm_vs_compiled_current_full": _compare_full_pair(
            compiled_native_bmm,
            compiled_current,
            full_route,
            full_fine,
            basis,
            compare_values,
            leaf_probe,
        ),
    }

    timings: dict[str, object] = {}
    for name, function in (
        ("current_atomic_leaf", atomic_factual_rgb_detail_contraction),
        ("visible_atomic_leaf", visible_matrix_contraction),
        ("native_bmm_leaf", visible_native_bmm_contraction),
    ):
        torch.cuda.reset_peak_memory_stats()
        timings[name] = {
            "times": _timed_leaf(
                function,
                value_weight,
                rgb_detail,
                warmup=args.warmup,
                steps=args.steps,
            ),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    for name, function in (
        ("compiled_current_full", compiled_current),
        ("compiled_visible_full", compiled_visible),
        ("compiled_native_bmm_full", compiled_native_bmm),
    ):
        torch.cuda.reset_peak_memory_stats()
        timings[name] = {
            "times": _timed_full(
                function,
                route,
                fine,
                basis,
                rgb_detail,
                warmup=args.warmup,
                steps=args.steps,
            ),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }

    components = _time_components(
        value_weight,
        rgb_detail,
        output_gradient,
        warmup=args.warmup,
        steps=args.steps,
    )
    current_median = timings["compiled_current_full"]["times"]["total"][
        "median_ms"
    ]
    visible_median = timings["compiled_visible_full"]["times"]["total"][
        "median_ms"
    ]
    native_bmm_median = timings["compiled_native_bmm_full"]["times"][
        "total"
    ]["median_ms"]
    result = {
        "schema": "clearvla-factual-atomic-layout-probe-v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "shape": {
            "value_weight": list(value_weight.shape),
            "rgb_detail": list(rgb_detail.shape),
        },
        "comparisons": comparisons,
        "timings": timings,
        "components": components,
        "compiled_full_speedup": current_median / visible_median,
        "compiled_native_bmm_full_speedup": current_median / native_bmm_median,
        "timestamp": time.time(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
