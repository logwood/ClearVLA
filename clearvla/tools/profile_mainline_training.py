"""Measure the real mainline training step without validation or checkpoint I/O.

This profiler intentionally uses the same data adapter, model, optimizer and
owned RNG streams as ``clearvla.mainline.train``.  It stops after a bounded
number of batches and writes only a small JSON summary.  It is a measurement
tool, not a second training implementation: the actual update is delegated to
``MainlineTrainingEngine.train_step`` and optional phase timings are supplied
by that method.

The profiler supports an opt-in ``--disable-gradient-spike-audit`` mode.  The
audit is read-only and does not affect gradients, clipping or optimizer state,
but its parameter scan can dominate short GPU runs.  Keeping this switch here
makes the cost visible without changing the production default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.gradient_audit import (
    DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mainline/object_intent_dynamics_323.json")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--decoded-cache", type=Path)
    parser.add_argument("--dino-cache", type=Path)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--disable-gradient-spike-audit", action="store_true")
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data = config.data
    runtime = config.runtime
    optimizer = config.optimizer
    for field_name, value in (
        ("raw_hdf5_root", args.data_root),
        ("decoded_cache", args.decoded_cache),
        ("dino_cache", args.dino_cache),
        ("t5_condition", args.t5_condition),
    ):
        if value is not None:
            data = replace(data, **{field_name: str(value)})
    if args.num_workers is not None:
        data = replace(data, num_workers=int(args.num_workers))
    if args.batch_size is not None:
        optimizer = replace(optimizer, batch_size=int(args.batch_size))
    if args.dtype is not None:
        runtime = replace(runtime, compute_dtype=str(args.dtype))
    result = replace(config, data=data, optimizer=optimizer, runtime=runtime)
    result.validate()
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": float(len(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p90": float(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite profiler value: {result}")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps <= 0 or args.warmup < 0 or args.warmup >= args.steps:
        raise ValueError("require steps > warmup >= 0")
    config = _overrides(load_config(args.config), args)
    _seed(config.data.seed)
    device = _device(args.device)
    dtype = resolve_compute_dtype(config)
    bundle = load_mainline_data(config, allow_null_goal=False)
    loader = bundle.loader(
        "train",
        batch_size=config.optimizer.batch_size,
        workers=config.data.num_workers,
        device=device,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.data.seed + 101),
    )
    iterator = iter(loader)
    model = ClearVLAMainlinePolicy(config).to(device)
    optimizer, _ownership = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=max(args.steps, 1),
        minimum_ratio=config.optimizer.min_lr_ratio,
    )
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        dtype=dtype,
        train_flow_generator=_owned_generator(device, config.data.seed + 102),
        train_condition_generator=_owned_generator(device, config.data.seed + 103),
        gradient_spike_audit_threshold=(
            None
            if args.disable_gradient_spike_audit
            else DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD
        ),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    phase_values: dict[str, list[float]] = {}
    data_wait_values: list[float] = []
    conversion_values: list[float] = []
    step_losses: list[float] = []
    repeated_batch = None
    for step in range(args.steps):
        if args.repeat_batch and repeated_batch is not None:
            raw_batch = repeated_batch
            wait_seconds = 0.0
        else:
            wait_started = time.perf_counter()
            raw_batch = next(iterator)
            wait_seconds = time.perf_counter() - wait_started
            if args.repeat_batch and repeated_batch is None:
                repeated_batch = raw_batch
        convert_started = time.perf_counter()
        batch = to_training_batch(
            raw_batch,
            goal=bundle.goal,
            config=config,
            device=device,
        )
        _sync(device)
        conversion_seconds = time.perf_counter() - convert_started
        phase: dict[str, float] = {}
        handler = None
        if not args.disable_gradient_spike_audit:
            # The handler itself is intentionally empty: the scan cost is
            # measured, while filesystem/logging latency stays out of it.
            def handler(_report: object) -> None:
                return None
        result = engine.train_step(
            batch,
            collect_diagnostics=False,
            gradient_spike_handler=handler,
            phase_timing=phase,
        )
        data_wait_values.append(_finite_float(wait_seconds))
        conversion_values.append(_finite_float(conversion_seconds))
        step_losses.append(float(result.loss))
        if step >= args.warmup:
            for name, value in phase.items():
                phase_values.setdefault(name, []).append(_finite_float(value))

    report: dict[str, object] = {
        "schema": "clearvla-training-acceleration-profile-v1",
        "config": str(args.config),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "batch_size": int(config.optimizer.batch_size),
        "num_workers": int(config.data.num_workers),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "gradient_spike_audit": not args.disable_gradient_spike_audit,
        "repeat_batch": bool(args.repeat_batch),
        "loss_first": step_losses[0],
        "loss_last": step_losses[-1],
        "data_wait_seconds": _summary(data_wait_values[args.warmup :]),
        "batch_conversion_seconds": _summary(conversion_values[args.warmup :]),
        "phases_seconds": {
            name: _summary(values) for name, values in sorted(phase_values.items())
        },
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        report["cuda_peak_allocated_gib"] = float(torch.cuda.max_memory_allocated(index) / 1024**3)
        report["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved(index) / 1024**3)
    return report


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()
