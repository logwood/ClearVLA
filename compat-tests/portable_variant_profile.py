"""Small, matched remote profile for versioned mainline worktrees.

The script deliberately uses the public mainline training surface and the
shared CUDA-Graph runner.  It is kept outside any model branch so a branch can
be benchmarked without changing its checkpoint or optimizer implementation.
The synthetic batch comes from that branch's own ``tests/test_mainline_policy``
helpers; therefore the tensor ABI is version-local while the timing protocol is
the same for every version.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import random
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The synthetic profile never opens an HDF5 episode.  Keep the harness usable
# on the minimal CUDA runner image where the optional data-loader dependency is
# absent; production/data code is not modified by this shim.
try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    h5py_stub = types.ModuleType("h5py")
    h5py_stub.File = type("File", (), {})
    h5py_stub.Dataset = type("Dataset", (), {})
    sys.modules["h5py"] = h5py_stub


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9170)
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="Compute dtype; auto follows the branch's serialized test config.",
    )
    parser.add_argument(
        "--selective-compile-profile",
        help="Profile resolved by the version's training acceleration adapter.",
    )
    parser.add_argument(
        "--allow-candidate-selective-compile",
        action="store_true",
        help="Allow an equivalence-pending profile for an isolated experiment.",
    )
    parser.add_argument(
        "--context-reuse",
        action="store_true",
        help="Enable the independently gated decoder context-reuse flags.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _move(getattr(value, field.name), device)
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _construct_engine(
    engine_cls: type[Any],
    *,
    model: Any,
    config: Any,
    optimizer: torch.optim.Optimizer,
    schedule: Any,
    device: torch.device,
    seed: int,
    dtype: torch.dtype,
) -> Any:
    signature = inspect.signature(engine_cls)
    kwargs: dict[str, Any] = {
        "model": model,
        "config": config,
        "optimizer": optimizer,
        "schedule": schedule,
        "device": device,
        "dtype": dtype,
    }
    for name, offset in (("train_flow_generator", 1), ("train_condition_generator", 2)):
        if name in signature.parameters:
            kwargs[name] = torch.Generator(device=device).manual_seed(seed + offset)
    # Newer mainline engines expose these as optional hot-path controls.  The
    # inspect guard keeps this harness usable on older schema branches.
    if "skip_postglobal_audit" in signature.parameters:
        kwargs["skip_postglobal_audit"] = True
    if "gradient_spike_audit_threshold" in signature.parameters:
        kwargs["gradient_spike_audit_threshold"] = None
    return engine_cls(**kwargs)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the remote profile")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    from test_mainline_policy import _batch, _config  # type: ignore[import-not-found]

    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.training.acceleration_adapters import (
        MainlineTrainingAccelerationAdapter,
    )
    from clearvla.mainline.training.acceleration_contract import (
        resolve_training_acceleration_adapter,
        resolve_training_compile_plan,
    )
    from clearvla.mainline.training.engine import MainlineTrainingEngine
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

    graph_runner_cls: Any | None = None
    if args.mode == "graph":
        from clearvla.mainline.training.cuda_graph import CudaGraphTrainingStepRunner

        graph_runner_cls = CudaGraphTrainingStepRunner

    _seed(args.seed)
    config = _config()
    # Test configs are intentionally small enough to run on all lab GPUs; the
    # batch dimension is the only controlled throughput variable here.
    dtype = {
        "auto": {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }[str(config.runtime.compute_dtype)],
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]
    model = ClearVLAMainlinePolicy(config).to("cuda").train()
    # Older source snapshots predate the optional model hook.  Attaching the
    # shared adapter in this harness keeps their model source untouched while
    # exercising exactly the same backend and layout discovery.
    adapter = resolve_training_acceleration_adapter(model)
    if adapter.name == "generic-static-v1":
        model.training_acceleration_adapter = MainlineTrainingAccelerationAdapter()
        adapter = resolve_training_acceleration_adapter(model)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=2,
        total_steps=max(args.steps + args.warmup + 2, 8),
        minimum_ratio=0.1,
    )
    engine = _construct_engine(
        MainlineTrainingEngine,
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=schedule,
        device=torch.device("cuda"),
        seed=args.seed + 100,
        dtype=dtype,
    )
    adapter.prepare(engine)
    if args.context_reuse:
        decoder = getattr(getattr(model, "execution_bottom", None), "decoder", None)
        if decoder is None:
            decoder = getattr(getattr(model, "bottom", None), "decoder", None)
        if decoder is None:
            raise RuntimeError("context-reuse requested but decoder was not found")
        decoder._reuse_prepared_block_contexts = True
        decoder._reuse_prepared_controller_context = True
        decoder._reuse_terminal_candidate_velocity = True
    compile_summary: dict[str, Any] | None = None
    if args.selective_compile_profile and args.mode == "eager":
        plan = resolve_training_compile_plan(
            adapter,
            engine,
            args.selective_compile_profile,
        )
        if plan is None:
            raise RuntimeError("selective compile profile resolved to no plan")
        from clearvla.mainline.training.selective_compile import (
            apply_training_compile_plan,
        )

        compile_summary = apply_training_compile_plan(
            engine,
            plan,
            profile=args.selective_compile_profile,
            allow_candidate=args.allow_candidate_selective_compile,
        ).summary()
    batch = _move(_batch(config, batch=args.batch_size), torch.device("cuda"))
    runner: Any = engine if args.mode == "eager" else graph_runner_cls(
        engine,
        compile_profile=args.selective_compile_profile,
        allow_candidate_compile=args.allow_candidate_selective_compile,
    )
    if args.mode == "graph":
        compile_summary = runner.compile_plan_summary

    # Warm-up includes graph capture in graph mode.  It is reported separately
    # and never folded into steady-state speed.
    for _ in range(args.warmup):
        runner.train_step(batch, collect_diagnostics=False)
    torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(args.steps):
        started = time.perf_counter()
        runner.train_step(batch, collect_diagnostics=False)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    ordered = sorted(durations)
    median = float(statistics.median(ordered))
    result: dict[str, Any] = {
        "schema": "clearvla-portable-variant-profile-v1",
        "mode": args.mode,
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "steps": int(args.steps),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seconds": durations,
        "median_seconds_per_step": median,
        "samples_per_second": float(args.batch_size / median),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
        "selective_compile_profile": args.selective_compile_profile,
        "selective_compile_plan": compile_summary,
        "context_reuse": bool(args.context_reuse),
    }
    if graph_runner_cls is not None:
        result["capture_count"] = int(runner.capture_count)
        result["capture_setup_seconds"] = float(runner.capture_setup_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    _run(_parser().parse_args())
