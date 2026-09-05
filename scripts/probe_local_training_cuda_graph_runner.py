"""Measure the checked static-input CUDA Graph runner on the local test model."""

from __future__ import annotations

import statistics
import time

import torch
from probe_local_training_cuda_graph import _engine

from clearvla.mainline.training.cuda_graph import CudaGraphTrainingStepRunner


def _measure(train_step, batch: object, *, warmup: int, steps: int) -> list[float]:
    for _ in range(warmup):
        train_step(batch, collect_diagnostics=False)
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(steps):
        started = time.perf_counter()
        train_step(batch, collect_diagnostics=False)
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return values


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    eager_engine, eager_batch = _engine(device)
    eager = _measure(eager_engine.train_step, eager_batch, warmup=3, steps=10)

    graph_engine, graph_batch = _engine(device)
    runner = CudaGraphTrainingStepRunner(graph_engine)
    graph = _measure(runner.train_step, graph_batch, warmup=3, steps=10)
    eager_median = statistics.median(eager)
    graph_median = statistics.median(graph)
    print("eager_seconds", eager)
    print("graph_runner_seconds", graph)
    print("eager_median_seconds", eager_median)
    print("graph_runner_median_seconds", graph_median)
    print("median_speedup", eager_median / graph_median)
    print("capture_count", runner.capture_count)
    print("capture_setup_seconds", runner.capture_setup_seconds)


if __name__ == "__main__":
    main()
