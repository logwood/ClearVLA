"""Probe the opt-in batched raw-flow offset sampler.

The raw-flow refiner historically launches one ``grid_sample`` per local
offset.  The version-owned experimental hook groups those offsets into one
batched launch while retaining the same coordinate and reduction order in the
caller.  This script measures the isolated refiner and compares outputs,
input VJPs and parameter VJPs before the change is considered for a full
Pen/RDT gate.

It is a laboratory probe only; it does not modify a checkpoint or install the
hook in the production model.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections.abc import Iterable
from typing import Any

import torch

from clearvla.mainline.v120_core.flow_dino_evidence import _DenseRawFlowRefiner


def _outputs(estimate: Any) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = [
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


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().norm().clamp_min(1e-30)
    return {
        "bit_exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(delta.norm().div(denominator).item()),
    }


def _make_inputs(
    *,
    batch: int,
    channels: int,
    side: int,
    coarse_side: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.randn(
        batch, channels, side, side, device=device, generator=generator
    )
    second = torch.randn(
        batch, channels, side, side, device=device, generator=generator
    )
    coarse_flow = torch.randn(
        batch, 2, coarse_side, coarse_side, device=device, generator=generator
    )
    coarse_reliability = torch.rand(
        batch,
        1,
        coarse_side,
        coarse_side,
        device=device,
        generator=generator,
    )
    return first, second, coarse_flow, coarse_reliability


def _run(
    module: _DenseRawFlowRefiner,
    inputs: Iterable[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], float]:
    module.zero_grad(set_to_none=True)
    live_inputs = tuple(
        value.detach().clone().requires_grad_(True) for value in inputs
    )
    estimate = module(*live_inputs)
    output_values = _outputs(estimate)
    loss = sum(value.float().square().mean() for value in output_values)
    loss.backward()
    input_gradients = tuple(
        value.grad.detach().clone() if value.grad is not None else torch.zeros_like(value)
        for value in live_inputs
    )
    parameter_gradients = tuple(
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for parameter in module.parameters()
    )
    return (
        tuple(value.detach().clone() for value in output_values),
        input_gradients,
        parameter_gradients,
        float(loss.detach().item()),
    )


def _compare(
    reference: _DenseRawFlowRefiner,
    candidate: _DenseRawFlowRefiner,
    inputs: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    expected = _run(reference, inputs)
    actual = _run(candidate, inputs)
    groups: dict[str, list[dict[str, float | bool]]] = {}
    for name, actual_group, expected_group in (
        ("outputs", actual[0], expected[0]),
        ("input_gradients", actual[1], expected[1]),
        ("parameter_gradients", actual[2], expected[2]),
    ):
        groups[name] = [
            _stats(actual_value, expected_value)
            for actual_value, expected_value in zip(
                actual_group, expected_group, strict=True
            )
        ]
    return {
        "loss": _stats(
            torch.tensor(actual[3], device=inputs[0].device),
            torch.tensor(expected[3], device=inputs[0].device),
        ),
        "groups": groups,
    }


def _timed(
    module: _DenseRawFlowRefiner,
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    samples: list[float] = []
    for index in range(warmup + steps):
        started = time.perf_counter()
        _run(module, inputs)
        if inputs[0].is_cuda:
            torch.cuda.synchronize(inputs[0].device)
        if index >= warmup:
            samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return {
        "minimum_ms": min(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "maximum_ms": max(ordered) * 1000.0,
        "iterations_per_second": 1.0 / statistics.fmean(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--coarse-side", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9170)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --device cuda")
    torch.manual_seed(args.seed)
    reference = _DenseRawFlowRefiner(
        args.channels,
        args.hidden,
        radius=args.radius,
        uncertainty_floor=0.03,
        activation_checkpoint=args.activation_checkpoint,
        preserve_uncertain_seed=True,
        bounded_coordinates=True,
        normalization_floor=0.10,
    ).to(device).train()
    candidate = copy.deepcopy(reference)
    candidate._batched_offset_sampling = True
    inputs = _make_inputs(
        batch=args.batch,
        channels=args.channels,
        side=args.side,
        coarse_side=args.coarse_side,
        device=device,
        seed=args.seed + 1,
    )
    comparison = _compare(reference, candidate, inputs)
    timing = {
        "reference": _timed(
            reference,
            inputs,
            warmup=args.warmup,
            steps=args.steps,
        ),
        "batched": _timed(
            candidate,
            inputs,
            warmup=args.warmup,
            steps=args.steps,
        ),
    }
    result: dict[str, object] = {
        "schema": "clearvla-batched-raw-flow-sampling-probe-v1",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "shape": {
            "batch": args.batch,
            "channels": args.channels,
            "hidden": args.hidden,
            "side": args.side,
            "coarse_side": args.coarse_side,
            "radius": args.radius,
            "offset_count": (2 * args.radius + 1) ** 2,
            "activation_checkpoint": args.activation_checkpoint,
        },
        "comparison": comparison,
        "timing": timing,
        "median_speedup": timing["reference"]["median_ms"]
        / timing["batched"]["median_ms"],
        "mean_speedup": timing["reference"]["mean_ms"]
        / timing["batched"]["mean_ms"],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
