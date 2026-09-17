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
    parser.add_argument(
        "--config-factory",
        default="_config",
        help=(
            "Factory in tests/test_mainline_policy.py used to construct the "
            "version-local profile config (for example _bspine_config for Pen)."
        ),
    )
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
        help=(
            "Profile resolved by the version's training acceleration adapter; "
            "semantic families may be joined with '+'."
        ),
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
    parser.add_argument(
        "--batched-raw-flow-sampling",
        action="store_true",
        help="Enable the opt-in batched local raw-flow offset sampler.",
    )
    parser.add_argument(
        "--atomic-factual-contraction",
        action="store_true",
        help="Enable the adapter-owned exact-order factual contraction.",
    )
    parser.add_argument(
        "--cudnn-benchmark-scope",
        action="append",
        default=[],
        choices=(
            "flow",
            "flow_encoder",
            "flow_context",
            "flow_update",
            "flow_delta",
            "flow_initial",
            "raw_pyramid",
            "raw_mid",
            "raw_high",
            "raw_context",
        ),
        help="Enable cuDNN algorithm search only inside one visual scope.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable global fixed-shape cuDNN algorithm search.",
    )
    parser.add_argument(
        "--cudnn-benchmark-phase",
        choices=("forward",),
        help=(
            "Enable global cuDNN benchmark only during training forward; "
            "backward and idle boundaries use default plans."
        ),
    )
    parser.add_argument(
        "--cudnn-default-prewarm-scope",
        action="append",
        default=[],
        choices=("raw_pyramid",),
        help=(
            "Seed one sensitive scope with default forward/backward plans "
            "before enabling global cuDNN benchmark."
        ),
    )
    parser.add_argument(
        "--cudnn-default-guard-scope",
        action="append",
        default=[],
        choices=(
            "flow",
            "flow_encoder",
            "flow_context",
            "flow_update",
            "flow_delta",
            "flow_initial",
            "raw_pyramid",
            "raw_mid",
            "raw_high",
            "raw_context",
        ),
        help=(
            "Keep one visual owner's convolution forwards on default algorithms "
            "while the engine-level phase policy controls backward."
        ),
    )
    parser.add_argument(
        "--disable-checkpoint-scope",
        action="append",
        default=[],
        choices=("p1", "raw_flow", "raw_pyramid", "raw_mid", "raw_high", "raw_context"),
        help=(
            "Disable one version-adapter-owned activation checkpoint surface; "
            "repeat the flag to select multiple surfaces."
        ),
    )
    parser.add_argument(
        "--raw-mid-checkpoint-save-operation",
        action="append",
        default=[],
        choices=("convolution", "linear", "grid_sample"),
        help="Cache one operator family inside the retained raw-mid checkpoint.",
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
    cudnn_benchmark_scopes = tuple(dict.fromkeys(args.cudnn_benchmark_scope))
    cudnn_default_prewarm_scopes = tuple(
        dict.fromkeys(args.cudnn_default_prewarm_scope)
    )
    cudnn_default_guard_scopes = tuple(
        dict.fromkeys(args.cudnn_default_guard_scope)
    )
    if args.cudnn_benchmark and cudnn_benchmark_scopes:
        raise ValueError(
            "global --cudnn-benchmark cannot be combined with scoped benchmarking"
        )
    if args.cudnn_benchmark_phase and not args.cudnn_benchmark:
        raise ValueError("--cudnn-benchmark-phase requires --cudnn-benchmark")
    if (
        cudnn_default_prewarm_scopes or cudnn_default_guard_scopes
    ) and not args.cudnn_benchmark:
        raise ValueError(
            "cuDNN default prewarm/guard scopes require --cudnn-benchmark"
        )
    torch.backends.cudnn.benchmark = bool(
        args.cudnn_benchmark
        and args.cudnn_benchmark_phase is None
        and not (cudnn_default_prewarm_scopes or cudnn_default_guard_scopes)
    )
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    import test_mainline_policy as policy_test  # type: ignore[import-not-found]

    _batch = policy_test._batch
    try:
        config_factory = getattr(policy_test, args.config_factory)
    except AttributeError as error:
        raise ValueError(
            f"config factory {args.config_factory!r} is absent from "
            f"{root / 'tests' / 'test_mainline_policy.py'}"
        ) from error
    if not callable(config_factory):
        raise TypeError(f"config factory {args.config_factory!r} is not callable")

    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.training.acceleration_adapters import (
        MainlineTrainingAccelerationAdapter,
    )
    from clearvla.mainline.training.acceleration_contract import (
        resolve_training_acceleration_adapter,
        resolve_training_compile_plan,
        set_cudnn_forward_benchmark_phase,
    )
    from clearvla.mainline.training.engine import MainlineTrainingEngine
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer

    graph_runner_cls: Any | None = None
    if args.mode == "graph":
        from clearvla.mainline.training.cuda_graph import CudaGraphTrainingStepRunner

        graph_runner_cls = CudaGraphTrainingStepRunner

    _seed(args.seed)
    config = config_factory()
    if args.dtype != "auto" and str(config.runtime.compute_dtype) != args.dtype:
        config = dataclasses.replace(
            config,
            runtime=dataclasses.replace(
                config.runtime,
                compute_dtype=args.dtype,
            ),
        )
        config.validate()
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
    if args.atomic_factual_contraction:
        hook = getattr(adapter, "set_atomic_factual_contraction", None)
        if not callable(hook):
            raise RuntimeError(
                f"adapter {adapter.name!r} does not expose the atomic factual "
                "contraction"
            )
        changed = int(hook(engine, enabled=True))
        if changed != 1:
            raise RuntimeError(
                "atomic factual contraction did not find its exact boundary"
            )
    checkpoint_scopes = tuple(dict.fromkeys(args.disable_checkpoint_scope))
    if checkpoint_scopes:
        hook = getattr(adapter, "set_activation_checkpointing", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose the "
                "activation-checkpoint policy probe"
            )
        changed = int(hook(engine, enabled=False, scopes=checkpoint_scopes))
        if changed == 0:
            raise RuntimeError(
                "checkpoint disabling was requested but no selected surface exists"
            )
    raw_mid_save_operations = tuple(
        dict.fromkeys(args.raw_mid_checkpoint_save_operation)
    )
    if raw_mid_save_operations:
        if "raw_mid" in checkpoint_scopes or "raw_flow" in checkpoint_scopes:
            raise ValueError(
                "raw-mid selective checkpoint caching cannot be combined with "
                "disabling the raw-mid checkpoint"
            )
        hook = getattr(adapter, "set_selective_checkpoint_save_operations", None)
        if not callable(hook):
            raise RuntimeError(
                f"adapter {adapter.name!r} does not expose selective checkpoint caching"
            )
        changed = int(
            hook(
                engine,
                scope="raw_mid",
                operations=raw_mid_save_operations,
            )
        )
        if changed == 0:
            raise RuntimeError(
                "raw-mid selective checkpoint caching found no supported surface"
            )
    if args.batched_raw_flow_sampling:
        hook = getattr(adapter, "set_batched_raw_flow_sampling", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose the "
                "batched raw-flow sampling probe"
            )
        changed = int(hook(engine, enabled=True))
        if changed == 0:
            raise RuntimeError(
                "batched raw-flow sampling was requested but no refiner was found"
            )
    if cudnn_benchmark_scopes:
        hook = getattr(adapter, "set_cudnn_benchmark_scopes", None)
        if not callable(hook):
            raise RuntimeError(
                f"adapter {adapter.name!r} does not expose scoped cuDNN benchmarking"
            )
        changed = int(
            hook(
                engine,
                enabled=True,
                scopes=cudnn_benchmark_scopes,
            )
        )
        if changed != len(cudnn_benchmark_scopes):
            raise RuntimeError(
                "scoped cuDNN benchmark did not install every requested scope"
            )
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
    cudnn_default_prewarm_setup_seconds = 0.0
    if cudnn_default_prewarm_scopes or cudnn_default_guard_scopes:
        prewarm_started = time.perf_counter()
        if cudnn_default_prewarm_scopes:
            hook = getattr(adapter, "prewarm_cudnn_default_scopes", None)
            if not callable(hook):
                raise RuntimeError(
                    f"adapter {adapter.name!r} does not expose default cuDNN prewarm"
                )
            changed = int(
                hook(
                    engine,
                    batch,
                    scopes=cudnn_default_prewarm_scopes,
                )
            )
            if changed != len(cudnn_default_prewarm_scopes):
                raise RuntimeError(
                    "default cuDNN prewarm did not cover every requested scope"
                )
        additional_guards = tuple(
            scope
            for scope in cudnn_default_guard_scopes
            if scope not in cudnn_default_prewarm_scopes
        )
        if additional_guards:
            guard_hook = getattr(adapter, "set_cudnn_default_scopes", None)
            if not callable(guard_hook):
                raise RuntimeError(
                    f"adapter {adapter.name!r} does not expose default cuDNN guards"
                )
            changed = int(
                guard_hook(
                    engine,
                    enabled=True,
                    scopes=additional_guards,
                )
            )
            if changed != len(additional_guards):
                raise RuntimeError(
                    "default cuDNN guards did not cover every requested scope"
                )
        torch.backends.cudnn.benchmark = args.cudnn_benchmark_phase is None
        cudnn_default_prewarm_setup_seconds = time.perf_counter() - prewarm_started
        torch.cuda.reset_peak_memory_stats()
    if args.cudnn_benchmark_phase == "forward":
        if not set_cudnn_forward_benchmark_phase(engine, enabled=True):
            raise RuntimeError("cuDNN forward benchmark phase was already installed")
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
        "config_factory": str(args.config_factory),
        "seconds": durations,
        "median_seconds_per_step": median,
        "samples_per_second": float(args.batch_size / median),
        "peak_allocated_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "peak_reserved_gib": float(torch.cuda.max_memory_reserved() / 2**30),
        "selective_compile_profile": args.selective_compile_profile,
        "selective_compile_plan": compile_summary,
        "context_reuse": bool(args.context_reuse),
        "batched_raw_flow_sampling": bool(args.batched_raw_flow_sampling),
        "atomic_factual_contraction": bool(args.atomic_factual_contraction),
        "cudnn_benchmark_scopes": list(cudnn_benchmark_scopes),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_benchmark_requested": bool(args.cudnn_benchmark),
        "cudnn_benchmark_phase": args.cudnn_benchmark_phase,
        "cudnn_default_prewarm_scopes": list(cudnn_default_prewarm_scopes),
        "cudnn_default_guard_scopes": list(
            dict.fromkeys(
                (*cudnn_default_prewarm_scopes, *cudnn_default_guard_scopes)
            )
        ),
        "cudnn_default_prewarm_setup_seconds": float(
            cudnn_default_prewarm_setup_seconds
        ),
        "disabled_checkpoint_scopes": list(checkpoint_scopes),
        "raw_mid_checkpoint_save_operations": list(raw_mid_save_operations),
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
