"""Measure grouped (rather than all-at-once) raw-flow sampling.

The production candidate batches every local offset.  This probe evaluates
smaller groups, which trade a few more launches for a narrower backward
accumulation fan-in and potentially smaller parameter tails.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time
import types

import torch
import torch.nn.functional as F

from clearvla.mainline.v120_core.flow_dino_evidence import (
    _DenseRawFlowRefiner,
    _normalize_grid,
)


def _grouped_samples(self, second_feature, center, offsets, search_scale):
    batch, channels, side, side_b = second_feature.shape
    if side != side_b or tuple(center.shape) != (batch, side, side, 2):
        raise ValueError("grouped raw sampling geometry does not align")
    offset_field = self._batched_offset_values.view(1, len(offsets), 2, 1, 1)
    if self.preserve_uncertain_seed:
        offset_field = offset_field * search_scale.unsqueeze(1)
    else:
        offset_field = offset_field.expand(batch, -1, -1, side, side)
    coordinates = center.unsqueeze(1) + offset_field.permute(0, 1, 3, 4, 2)
    valid = (
        (coordinates[..., 0] >= 0.0)
        & (coordinates[..., 0] <= float(side - 1))
        & (coordinates[..., 1] >= 0.0)
        & (coordinates[..., 1] <= float(side - 1))
    )
    group_size = int(self._probe_group_size)
    sampled_groups = []
    for start in range(0, len(offsets), group_size):
        stop = min(start + group_size, len(offsets))
        width = stop - start
        flat_grid = _normalize_grid(
            coordinates[:, start:stop].reshape(batch * width, side, side, 2),
            side,
            side,
        )
        flat_value = (
            second_feature.unsqueeze(1)
            .expand(-1, width, -1, -1, -1)
            .reshape(batch * width, channels, side, side)
        )
        with torch.autocast(device_type=second_feature.device.type, enabled=False):
            sampled = F.grid_sample(
                flat_value.float(),
                flat_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        sampled_groups.append(sampled.reshape(batch, width, channels, side, side))
    return torch.cat(sampled_groups, dim=1), valid, offset_field


def _outputs(estimate):
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


def _run(module, inputs):
    module.zero_grad(set_to_none=True)
    live = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    values = _outputs(module(*live))
    sum(x.float().square().mean() for x in values).backward()
    return (
        tuple(x.detach().clone() for x in values),
        tuple(x.grad.detach().clone() for x in live),
        tuple(
            p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for p in module.parameters()
        ),
    )


def _delta(actual, expected):
    max_abs = 0.0
    rel = 0.0
    for a, e in zip(actual, expected, strict=True):
        d = (a.float() - e.float()).abs()
        max_abs = max(max_abs, float(d.max().item()))
        rel = max(rel, float(d.norm().div(e.float().norm().clamp_min(1e-30)).item()))
    return max_abs, rel


def _time(module, inputs, warmup, steps):
    samples = []
    for i in range(warmup + steps):
        started = time.perf_counter()
        _run(module, inputs)
        torch.cuda.synchronize(inputs[0].device)
        if i >= warmup:
            samples.append(time.perf_counter() - started)
    return statistics.median(samples) * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--coarse-side", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--groups", default="1,2,4,5,8,13,25")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(9170)
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
    generator = torch.Generator(device=device).manual_seed(9171)
    inputs = (
        torch.randn(args.batch, args.channels, args.side, args.side, device=device, generator=generator),
        torch.randn(args.batch, args.channels, args.side, args.side, device=device, generator=generator),
        torch.randn(args.batch, 2, args.coarse_side, args.coarse_side, device=device, generator=generator),
        torch.rand(args.batch, 1, args.coarse_side, args.coarse_side, device=device, generator=generator),
    )
    expected = _run(reference, inputs)
    for raw_group in args.groups.split(","):
        group = int(raw_group)
        candidate = copy.deepcopy(reference)
        candidate._batched_offset_sampling = True
        candidate._probe_group_size = group
        candidate._batched_samples = types.MethodType(_grouped_samples, candidate)
        actual = _run(candidate, inputs)
        ref_ms = _time(reference, inputs, args.warmup, args.steps)
        cand_ms = _time(candidate, inputs, args.warmup, args.steps)
        print(
            {
                "group_size": group,
                "launch_groups": (25 + group - 1) // group,
                "output_delta": _delta(actual[0], expected[0]),
                "input_gradient_delta": _delta(actual[1], expected[1]),
                "parameter_gradient_delta": _delta(actual[2], expected[2]),
                "reference_ms": ref_ms,
                "candidate_ms": cand_ms,
                "speedup": ref_ms / cand_ms,
            }
        )


if __name__ == "__main__":
    main()
