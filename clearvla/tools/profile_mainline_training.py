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
import inspect
import json
import math
import random
import statistics
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.config import ExperimentConfig, load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training.acceleration_adapters import (
    MainlineTrainingAccelerationAdapter,
)
from clearvla.mainline.training.acceleration_contract import (
    resolve_training_acceleration_adapter,
    resolve_training_compile_plan,
    set_cudnn_forward_benchmark_phase,
)
from clearvla.mainline.training.cuda_graph import CudaGraphTrainingStepRunner
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.gradient_audit import (
    DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD,
)
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
from clearvla.mainline.training.prefetch import CudaTrainingBatchPrefetcher
from clearvla.mainline.training.selective_compile import (
    apply_training_compile_plan,
)


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
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help=(
            "Use PyTorch fused AdamW for an isolated optimizer-equivalence "
            "probe; ownership, hyperparameters and checkpoint tensors are retained."
        ),
    )
    parser.add_argument(
        "--retain-postglobal-audit",
        action="store_true",
        help="Retain the baseline audit-only post-global norm on non-diagnostic steps.",
    )
    parser.add_argument(
        "--retain-training-execution-diagnostics",
        action="store_true",
        help=(
            "Retain the historical decoder and execution-loss diagnostics on "
            "every training step for a matched timing baseline."
        ),
    )
    parser.add_argument(
        "--retain-candidate-operation-diagnostics",
        action="store_true",
        help=(
            "Retain audit-only block/controller statistics inside every "
            "candidate operation for a matched timing baseline."
        ),
    )
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument(
        "--cuda-batch-prefetch",
        action="store_true",
        help=(
            "Convert the next pinned worker batch on a private CUDA copy stream "
            "while the current training step runs. This transport-only probe "
            "does not change batch selection or model execution."
        ),
    )
    parser.add_argument(
        "--phase-breakdown",
        action="store_true",
        help="Synchronize each train-step phase for a diagnostic breakdown.",
    )
    parser.add_argument(
        "--cuda-graph-training",
        action="store_true",
        help=(
            "Capture ordinary-batch online encode, forward, loss and backward; "
            "keep diagnostics, clipping, AdamW and scheduling eager."
        ),
    )
    parser.add_argument(
        "--cuda-graph-capture-gradient-lifecycle",
        action="store_true",
        help=(
            "Capture deterministic gradient norm/clipping mutations inside the "
            "CUDA Graph (requires --disable-gradient-spike-audit)."
        ),
    )
    parser.add_argument(
        "--cuda-graph-capture-optimizer-step",
        action="store_true",
        help=(
            "Replay the initialized fused AdamW update from a second CUDA "
            "Graph while keeping scheduling and checkpoint metadata eager."
        ),
    )
    parser.add_argument(
        "--selective-compile-profile",
        help=(
            "Apply a version-adapter-owned selective Inductor plan (for the "
            "mainline adapter: fast|fast-rng|precision|autotune|partitioned "
            "followed "
            "by semantic families such as mainline+intent+factual; stable "
            "single scopes include flow, raw, visual, mainline, mmdit, "
            "conditioning, bridge, intent, factual, factual-atomic, policy "
            "and execution). "
            "Candidate plans also require --allow-candidate-selective-compile."
        ),
    )
    parser.add_argument(
        "--allow-candidate-selective-compile",
        action="store_true",
        help="Authorize a candidate plan for this isolated profiler run.",
    )
    parser.add_argument(
        "--compile-mmdit-blocks",
        action="store_true",
        help="Compile only the stable TimeDomainMMDiT action blocks with Inductor.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument(
        "--compile-forward",
        action="store_true",
        help="Compile the formal engine forward with graph-break fallback.",
    )
    parser.add_argument(
        "--compile-visual-submodules",
        action="store_true",
        help=(
            "Compile stable Flow-DINO/raw-flow visual submodules individually "
            "(experimental; keeps the outer Python graph unchanged)."
        ),
    )
    parser.add_argument(
        "--compile-execution-submodules",
        action="store_true",
        help=(
            "Compile the execution controller/value reader tensor subgraphs "
            "individually (experimental)."
        ),
    )
    parser.add_argument(
        "--compile-mainline-blocks",
        action="store_true",
        help=(
            "Compile repeated grounding/P1/world/transition tensor blocks "
            "individually (experimental; may graph-break)."
        ),
    )
    parser.add_argument(
        "--compile-candidate-prefix",
        action="store_true",
        help=(
            "Compile the attached training candidate-prefix chart as one "
            "experimental subgraph."
        ),
    )
    parser.add_argument(
        "--batched-candidate-prefix",
        action="store_true",
        help=(
            "Batch independent candidate-prefix owners with torch.func.vmap "
            "at the identity contraction boundary (experimental)."
        ),
    )
    parser.add_argument(
        "--reuse-terminal-candidate-velocity",
        action="store_true",
        help=(
            "Reuse the already-computed terminal candidate velocity instead "
            "of evaluating its immediately masked head row (experimental)."
        ),
    )
    parser.add_argument(
        "--candidate-prefix-reuse",
        action="store_true",
        help="Use the experimental attached training candidate-prefix chart.",
    )
    parser.add_argument(
        "--reuse-prepared-block-contexts",
        action="store_true",
        help=(
            "Reuse action-independent MMDiT block context across candidate "
            "operations (experimental equivalence-gated path)."
        ),
    )
    parser.add_argument(
        "--reuse-prepared-controller-context",
        action="store_true",
        help=(
            "Reuse action-independent execution-controller source projections "
            "across decisions (experimental equivalence-gated path)."
        ),
    )
    parser.add_argument(
        "--batched-raw-flow-sampling",
        action="store_true",
        help=(
            "Batch independent local raw-flow offset samplers into one launch "
            "per refiner (experimental; requires its own equivalence gate)."
        ),
    )
    parser.add_argument(
        "--atomic-factual-contraction",
        action="store_true",
        help=(
            "Replace only the normalized factual RGB/detail einsum with its "
            "exact K-before-M BMM training operator (experimental)."
        ),
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
        help=(
            "Enable fixed-shape cuDNN algorithm search only inside one "
            "adapter-owned visual module; repeat to combine scopes."
        ),
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable global fixed-shape cuDNN algorithm search for this probe.",
    )
    parser.add_argument(
        "--cudnn-benchmark-phase",
        choices=("forward",),
        help=(
            "Enable cuDNN benchmark only while the engine executes training "
            "forward; backward and idle boundaries retain default plans."
        ),
    )
    parser.add_argument(
        "--cudnn-default-prewarm-scope",
        action="append",
        default=[],
        choices=("raw_pyramid",),
        help=(
            "Before global cuDNN benchmark, seed one numerically sensitive "
            "scope with the default same-shape forward/backward algorithms."
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
        "--disable-activation-checkpointing",
        action="store_true",
        help=(
            "Disable deterministic raw-flow and late-P1 activation recompute "
            "wrappers for an equivalence-gated memory-for-speed probe."
        ),
    )
    parser.add_argument(
        "--disable-raw-flow-checkpointing",
        action="store_true",
        help="Disable only raw-flow/pyramid activation recomputation.",
    )
    for scope, label in (
        ("raw-pyramid", "raw RGB pyramid"),
        ("raw-mid", "mid-resolution raw-flow refiner"),
        ("raw-high", "high-resolution raw-flow refiner"),
        ("raw-context", "early masked raw-context encoder"),
    ):
        parser.add_argument(
            f"--disable-{scope}-checkpointing",
            action="store_true",
            help=f"Disable only {label} activation recomputation.",
        )
    parser.add_argument(
        "--disable-p1-activation-checkpointing",
        action="store_true",
        help="Disable only late-P1 typed microgrid activation recomputation.",
    )
    parser.add_argument(
        "--raw-mid-checkpoint-save-operation",
        action="append",
        default=[],
        choices=("convolution", "linear", "grid_sample"),
        help=(
            "Keep the raw-mid checkpoint but cache one expensive operator "
            "family during backward recomputation; repeat to combine families."
        ),
    )
    parser.add_argument(
        "--torch-profile-output",
        type=Path,
        help="Write a one-step torch.profiler operator table to this text file.",
    )
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


def _bottom_module(model: ClearVLAMainlinePolicy) -> Any:
    """Resolve the execution bottom across modular and legacy layouts."""

    execution_bottom = getattr(model, "execution_bottom", None)
    if execution_bottom is not None:
        return execution_bottom
    bottom = getattr(model, "bottom", None)
    if bottom is None:
        raise RuntimeError("model has no execution bottom")
    return bottom


def _decoder_module(model: ClearVLAMainlinePolicy) -> Any:
    decoder = getattr(_bottom_module(model), "decoder", None)
    if decoder is None:
        raise RuntimeError("model execution bottom has no decoder")
    return decoder


def _compile_mmdit_blocks(model: ClearVLAMainlinePolicy, *, mode: str) -> float:
    """Compile the repeated action blocks without wrapping the parent module."""

    started = time.perf_counter()
    decoder = _decoder_module(model)
    blocks = tuple(decoder.blocks)
    if not blocks:
        raise RuntimeError("cannot compile an empty MMDiT block list")
    for block in blocks:
        # Assigning the callable on the instance keeps the original Module
        # tree, parameter names, optimizer ownership and checkpoint ABI intact.
        block.forward = torch.compile(  # type: ignore[method-assign]
            block.forward,
            dynamic=False,
            fullgraph=False,
            mode=mode,
        )
    return time.perf_counter() - started


def _compile_visual_submodules(model: ClearVLAMainlinePolicy, *, mode: str) -> float:
    """Compile repeated visual tensor kernels without wrapping the outer graph."""

    started = time.perf_counter()
    encoder = model.observation.compiler.encoder
    modules: list[torch.nn.Module] = [encoder.flow]
    if encoder.raw_flow is not None:
        modules.append(encoder.raw_flow)
    for module in modules:
        # Keep module registration, parameter ownership and checkpoint names
        # unchanged; only replace the callable used by this profiler process.
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            dynamic=False,
            fullgraph=False,
            mode=mode,
        )
    return time.perf_counter() - started


def _compile_execution_submodules(model: ClearVLAMainlinePolicy, *, mode: str) -> float:
    """Compile recurrent execution tensor subgraphs without wrapping the decoder."""

    started = time.perf_counter()
    controller = _decoder_module(model).execution_controller
    if controller is None:
        raise RuntimeError("execution controller is disabled in this profile")
    for module in (controller, controller.value_reader):
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            dynamic=False,
            fullgraph=False,
            mode=mode,
        )
    return time.perf_counter() - started


def _compile_mainline_blocks(model: ClearVLAMainlinePolicy, *, mode: str) -> float:
    """Compile the repeated tensor-only blocks around the online graph."""

    started = time.perf_counter()
    grounding = getattr(model, "grounding", None)
    p1 = getattr(model, "p1", None)
    world = getattr(model, "world", None)
    transition = getattr(model, "transition", None)
    bottom = _bottom_module(model)
    modules: list[torch.nn.Module] = []
    if grounding is not None:
        modules.extend(tuple(grounding.blocks))
    if p1 is not None:
        modules.append(p1.dynamic_policy_block)
    if world is not None:
        modules.extend((world.dynamics.w1, world.dynamics.w2))
    if transition is not None:
        modules.append(transition.v120_transition)
    modules.extend(tuple(getattr(bottom, "layer_contract_heads", ())))
    if not modules:
        raise RuntimeError("model has no resolvable mainline compile blocks")
    for module in modules:
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            dynamic=False,
            fullgraph=False,
            mode=mode,
        )
    return time.perf_counter() - started


def _compile_candidate_prefix(model: ClearVLAMainlinePolicy, *, mode: str) -> float:
    """Compile the prefix chart boundary while leaving decoder dispatch intact."""

    started = time.perf_counter()
    decoder = _decoder_module(model)
    decoder._run_differentiable_native_candidates_prefix_reuse = torch.compile(  # type: ignore[method-assign]
        decoder._run_differentiable_native_candidates_prefix_reuse,
        dynamic=False,
        fullgraph=False,
        mode=mode,
    )
    return time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps <= 0 or args.warmup < 0 or args.warmup >= args.steps:
        raise ValueError("require steps > warmup >= 0")
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
    if cudnn_default_prewarm_scopes or cudnn_default_guard_scopes:
        torch.backends.cudnn.benchmark = False
    elif args.cudnn_benchmark and args.cudnn_benchmark_phase is None:
        torch.backends.cudnn.benchmark = True
    unsupported_graph_compile = (
        args.compile_forward,
        args.compile_candidate_prefix,
    )
    if args.cuda_graph_training and any(unsupported_graph_compile):
        raise ValueError(
            "CUDA Graph training does not support whole-forward or candidate-prefix compile"
        )
    if (
        args.cuda_graph_training
        and args.compile_mmdit_blocks
        and args.compile_mode == "reduce-overhead"
    ):
        raise ValueError(
            "manual CUDA Graph training cannot nest reduce-overhead CUDA Graphs"
        )
    if args.cuda_graph_capture_optimizer_step and not args.cuda_graph_training:
        raise ValueError(
            "--cuda-graph-capture-optimizer-step requires --cuda-graph-training"
        )
    if args.cuda_graph_capture_optimizer_step and not args.fused_adamw:
        raise ValueError(
            "--cuda-graph-capture-optimizer-step requires --fused-adamw"
        )
    legacy_compile_flags = {
        "compile_mmdit_blocks": args.compile_mmdit_blocks,
        "compile_forward": args.compile_forward,
        "compile_visual_submodules": args.compile_visual_submodules,
        "compile_execution_submodules": args.compile_execution_submodules,
        "compile_mainline_blocks": args.compile_mainline_blocks,
        "compile_candidate_prefix": args.compile_candidate_prefix,
    }
    enabled_legacy_compile_flags = tuple(
        name for name, enabled in legacy_compile_flags.items() if enabled
    )
    if args.selective_compile_profile and enabled_legacy_compile_flags:
        raise ValueError(
            "--selective-compile-profile cannot be combined with legacy compile "
            f"flags: {enabled_legacy_compile_flags!r}"
        )
    if args.allow_candidate_selective_compile and not args.selective_compile_profile:
        raise ValueError(
            "--allow-candidate-selective-compile requires "
            "--selective-compile-profile"
        )
    config = _overrides(load_config(args.config), args)
    _seed(config.data.seed)
    device = _device(args.device)
    if args.cuda_batch_prefetch and device.type != "cuda":
        raise ValueError("--cuda-batch-prefetch requires a CUDA device")
    if args.cuda_batch_prefetch and args.phase_breakdown:
        raise ValueError(
            "--phase-breakdown synchronizes the device and would disable "
            "CUDA batch-prefetch overlap"
        )
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
    # Older version snapshots predate the policy-level adapter hook.  Attach
    # the version-neutral mainline adapter at the profiler boundary so those
    # snapshots still receive the same capture-safe preparation as current
    # source, without changing their module tree or checkpoint state.
    adapter = resolve_training_acceleration_adapter(model)
    if adapter.name == "generic-static-v1":
        model.training_acceleration_adapter = MainlineTrainingAccelerationAdapter()
        adapter = resolve_training_acceleration_adapter(model)
    bottom = _bottom_module(model)
    decoder = _decoder_module(model)
    if args.retain_training_execution_diagnostics:
        bottom._retain_training_decoder_diagnostics = True
    if args.retain_candidate_operation_diagnostics:
        decoder._retain_candidate_operation_diagnostics = True
    if args.candidate_prefix_reuse:
        decoder._training_candidate_prefix_reuse = True
    if args.reuse_prepared_block_contexts:
        decoder._reuse_prepared_block_contexts = True
    if args.reuse_prepared_controller_context:
        decoder._reuse_prepared_controller_context = True
    if args.batched_candidate_prefix:
        decoder._batched_candidate_prefix = True
    if args.reuse_terminal_candidate_velocity:
        decoder._reuse_terminal_candidate_velocity = True
    compile_seconds = 0.0
    if args.compile_mmdit_blocks:
        compile_seconds = _compile_mmdit_blocks(model, mode=args.compile_mode)
    visual_compile_seconds = 0.0
    if args.compile_visual_submodules:
        visual_compile_seconds = _compile_visual_submodules(model, mode=args.compile_mode)
    execution_compile_seconds = 0.0
    if args.compile_execution_submodules:
        execution_compile_seconds = _compile_execution_submodules(
            model, mode=args.compile_mode
        )
    mainline_compile_seconds = 0.0
    if args.compile_mainline_blocks:
        mainline_compile_seconds = _compile_mainline_blocks(
            model, mode=args.compile_mode
        )
    candidate_compile_seconds = 0.0
    if args.compile_candidate_prefix:
        candidate_compile_seconds = _compile_candidate_prefix(
            model, mode=args.compile_mode
        )
    optimizer, _ownership = build_optimizer(
        model,
        config,
        fused=True if args.fused_adamw else None,
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=max(args.steps, 1),
        minimum_ratio=config.optimizer.min_lr_ratio,
    )
    # Keep the profiler usable across version snapshots whose engine grew
    # optional hot-path controls at different times.  These controls affect
    # diagnostics only; passing an unknown keyword would otherwise prevent a
    # perfectly valid older implementation from being benchmarked.
    engine_kwargs: dict[str, Any] = {
        "model": model,
        "config": config,
        "optimizer": optimizer,
        "schedule": schedule,
        "device": device,
        "dtype": dtype,
        "train_flow_generator": _owned_generator(device, config.data.seed + 102),
        "train_condition_generator": _owned_generator(device, config.data.seed + 103),
    }
    engine_signature = inspect.signature(MainlineTrainingEngine)
    if "skip_postglobal_audit" in engine_signature.parameters:
        engine_kwargs["skip_postglobal_audit"] = not args.retain_postglobal_audit
    if "gradient_spike_audit_threshold" in engine_signature.parameters:
        engine_kwargs["gradient_spike_audit_threshold"] = (
            None
            if args.disable_gradient_spike_audit
            else DEFAULT_GRADIENT_SPIKE_AUDIT_THRESHOLD
        )
    engine = MainlineTrainingEngine(**engine_kwargs)
    if args.atomic_factual_contraction:
        hook = getattr(adapter, "set_atomic_factual_contraction", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose the "
                "atomic factual-contraction probe"
            )
        changed = int(hook(engine, enabled=True))
        if changed != 1:
            raise RuntimeError(
                "atomic factual contraction was requested but its exact "
                "version-owned boundary was not found"
            )
    checkpoint_scopes = tuple(
        scope
        for scope, selected in (
            ("raw_flow", args.disable_raw_flow_checkpointing or args.disable_activation_checkpointing),
            ("raw_pyramid", args.disable_raw_pyramid_checkpointing),
            ("raw_mid", args.disable_raw_mid_checkpointing),
            ("raw_high", args.disable_raw_high_checkpointing),
            ("raw_context", args.disable_raw_context_checkpointing),
            ("p1", args.disable_p1_activation_checkpointing or args.disable_activation_checkpointing),
        )
        if selected
    )
    if checkpoint_scopes:
        hook = getattr(adapter, "set_activation_checkpointing", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose the "
                "activation-checkpoint policy probe"
            )
        changed = int(
            hook(engine, enabled=False, scopes=checkpoint_scopes)
        )
        if changed == 0:
            raise RuntimeError(
                "activation-checkpoint disabling was requested but no supported "
                "checkpoint surfaces were found"
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
                f"training adapter {adapter.name!r} does not expose selective "
                "checkpoint caching"
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
            raise RuntimeError("batched raw-flow sampling was requested but no refiner was found")
    if cudnn_benchmark_scopes:
        hook = getattr(adapter, "set_cudnn_benchmark_scopes", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose scoped "
                "cuDNN benchmarking"
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
    if args.retain_training_execution_diagnostics:
        engine._retain_training_execution_diagnostics = True
    if args.compile_forward:
        engine._forward = torch.compile(  # type: ignore[method-assign]
            engine._forward,
            dynamic=False,
            fullgraph=False,
            mode=args.compile_mode,
        )
    selective_compile_summary: dict[str, Any] | None = None
    if args.selective_compile_profile and not args.cuda_graph_training:
        adapter = resolve_training_acceleration_adapter(engine.model)
        adapter.prepare(engine)
        plan = resolve_training_compile_plan(
            adapter,
            engine,
            args.selective_compile_profile,
        )
        if plan is None:
            raise RuntimeError("requested selective compile profile resolved to no plan")
        selective_compile_summary = apply_training_compile_plan(
            engine,
            plan,
            profile=args.selective_compile_profile,
            allow_candidate=args.allow_candidate_selective_compile,
        ).summary()
    if args.cudnn_benchmark_phase == "forward":
        if not set_cudnn_forward_benchmark_phase(engine, enabled=True):
            raise RuntimeError("cuDNN forward benchmark phase was already installed")
    step_runner: MainlineTrainingEngine | CudaGraphTrainingStepRunner = engine
    if args.cuda_graph_training:
        step_runner = CudaGraphTrainingStepRunner(
            engine,
            compile_profile=args.selective_compile_profile,
            allow_candidate_compile=args.allow_candidate_selective_compile,
            capture_gradient_lifecycle=args.cuda_graph_capture_gradient_lifecycle,
            capture_optimizer_step=args.cuda_graph_capture_optimizer_step,
        )
        selective_compile_summary = step_runner.compile_plan_summary
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    phase_values: dict[str, list[float]] = {}
    data_wait_values: list[float] = []
    conversion_values: list[float] = []
    step_losses: list[torch.Tensor] = []
    repeated_batch = None
    batch_prefetcher = None
    prefetch_raw_iterator = None
    cudnn_prewarm_pending = bool(
        cudnn_default_prewarm_scopes or cudnn_default_guard_scopes
    )
    cudnn_prewarm_setup_seconds = 0.0
    if args.cuda_batch_prefetch:
        def raw_batches():
            repeated_source = None
            for _index in range(args.steps):
                if args.repeat_batch and repeated_source is not None:
                    raw_batch = repeated_source
                else:
                    raw_batch = next(iterator)
                    if args.repeat_batch:
                        repeated_source = raw_batch
                yield raw_batch

        prefetch_raw_iterator = iter(raw_batches())
    measured_started = 0.0
    torch_profile_table: str | None = None
    for step in range(args.steps):
        if step == args.warmup:
            _sync(device)
            measured_started = time.perf_counter()
        if prefetch_raw_iterator is not None and step == 0:
            # Let the first CUDA Graph capture own only one dynamic input
            # batch.  The copy pipeline is primed immediately after that
            # startup step; all measured steps still receive a prefetched
            # batch when warmup is non-zero.
            wait_started = time.perf_counter()
            raw_batch = next(prefetch_raw_iterator)
            wait_seconds = time.perf_counter() - wait_started
            convert_started = time.perf_counter()
            batch = to_training_batch(
                raw_batch,
                goal=bundle.goal,
                config=config,
                device=device,
            )
            conversion_seconds = time.perf_counter() - convert_started
        elif batch_prefetcher is not None:
            batch = next(batch_prefetcher)
            wait_seconds = batch_prefetcher.last_source_wait_seconds
            conversion_seconds = batch_prefetcher.last_conversion_enqueue_seconds
        else:
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
            if args.phase_breakdown:
                _sync(device)
            conversion_seconds = time.perf_counter() - convert_started
        if cudnn_prewarm_pending:
            prewarm_started = time.perf_counter()
            if cudnn_default_prewarm_scopes:
                hook = getattr(adapter, "prewarm_cudnn_default_scopes", None)
                if not callable(hook):
                    raise RuntimeError(
                        f"training adapter {adapter.name!r} does not expose default "
                        "cuDNN prewarm"
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
                        f"training adapter {adapter.name!r} does not expose default "
                        "cuDNN guards"
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
            torch.backends.cudnn.benchmark = (
                args.cudnn_benchmark_phase is None
            )
            cudnn_prewarm_setup_seconds = time.perf_counter() - prewarm_started
            cudnn_prewarm_pending = False
            torch.cuda.reset_peak_memory_stats(device)
        phase: dict[str, float] = {}
        handler = None
        if not args.disable_gradient_spike_audit:
            # The handler itself is intentionally empty: the scan cost is
            # measured, while filesystem/logging latency stays out of it.
            def handler(_report: object) -> None:
                return None
        should_profile = args.torch_profile_output is not None and step == args.warmup
        profile_context = (
            torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            if should_profile
            else nullcontext()
        )
        step_kwargs: dict[str, Any] = {
            "batch": batch,
            "collect_diagnostics": False,
            "gradient_spike_handler": handler,
        }
        if args.phase_breakdown and "phase_timing" in inspect.signature(
            step_runner.train_step
        ).parameters:
            step_kwargs["phase_timing"] = phase
        with profile_context as profiler:
            result = step_runner.train_step(**step_kwargs)
        if should_profile and profiler is not None:
            torch_profile_table = profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=80
            )
        data_wait_values.append(_finite_float(wait_seconds))
        conversion_values.append(_finite_float(conversion_seconds))
        step_losses.append(result.loss.detach())
        if step >= args.warmup:
            for name, value in phase.items():
                phase_values.setdefault(name, []).append(_finite_float(value))
        # ``step_kwargs`` otherwise keeps the just-consumed GPU batch alive
        # while the prefetcher allocates batch n+2 at the top of the next
        # iteration.  End both aliases here so steady state owns only the
        # current and next transport buffers.
        del step_kwargs, batch
        if prefetch_raw_iterator is not None and batch_prefetcher is None:
            batch_prefetcher = CudaTrainingBatchPrefetcher(
                prefetch_raw_iterator,
                converter=lambda raw_batch: to_training_batch(
                    raw_batch,
                    goal=bundle.goal,
                    config=config,
                    device=device,
                ),
                device=device,
            )

    _sync(device)
    if batch_prefetcher is not None:
        batch_prefetcher.close()
    measured_seconds = time.perf_counter() - measured_started
    measured_steps = int(args.steps - args.warmup)
    measured_samples = measured_steps * int(config.optimizer.batch_size)

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
        "fused_adamw": bool(args.fused_adamw),
        "postglobal_audit": bool(args.retain_postglobal_audit),
        "training_execution_diagnostics": bool(
            args.retain_training_execution_diagnostics
        ),
        "candidate_operation_diagnostics": bool(
            args.retain_candidate_operation_diagnostics
        ),
        "repeat_batch": bool(args.repeat_batch),
        "cuda_batch_prefetch": bool(args.cuda_batch_prefetch),
        "batch_conversion_timing": (
            "copy_stream_enqueue"
            if args.cuda_batch_prefetch
            else "caller_wall_time"
        ),
        "phase_breakdown": bool(args.phase_breakdown),
        "cuda_graph_training": bool(args.cuda_graph_training),
        "cuda_graph_capture_gradient_lifecycle": bool(
            args.cuda_graph_capture_gradient_lifecycle
        ),
        "cuda_graph_capture_optimizer_step": bool(
            args.cuda_graph_capture_optimizer_step
        ),
        "cuda_graph_capture_count": int(
            step_runner.capture_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_replay_count": int(
            step_runner.replay_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_diagnostic_eager_count": int(
            step_runner.diagnostic_eager_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_capture_setup_seconds": _finite_float(
            step_runner.capture_setup_seconds
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0.0
        ),
        "cuda_graph_optimizer_capture_count": int(
            step_runner.optimizer_capture_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_optimizer_replay_count": int(
            step_runner.optimizer_replay_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_optimizer_eager_step_count": int(
            step_runner.optimizer_eager_step_count
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0
        ),
        "cuda_graph_optimizer_capture_setup_seconds": _finite_float(
            step_runner.optimizer_capture_setup_seconds
            if isinstance(step_runner, CudaGraphTrainingStepRunner)
            else 0.0
        ),
        "selective_compile_profile": args.selective_compile_profile,
        "selective_compile_plan": selective_compile_summary,
        "compile_mmdit_blocks": bool(args.compile_mmdit_blocks),
        "compile_forward": bool(args.compile_forward),
        "compile_visual_submodules": bool(args.compile_visual_submodules),
        "compile_execution_submodules": bool(args.compile_execution_submodules),
        "compile_mainline_blocks": bool(args.compile_mainline_blocks),
        "compile_candidate_prefix": bool(args.compile_candidate_prefix),
        "batched_candidate_prefix": bool(args.batched_candidate_prefix),
        "reuse_terminal_candidate_velocity": bool(
            args.reuse_terminal_candidate_velocity
        ),
        "compile_mode": str(args.compile_mode),
        "candidate_prefix_reuse": bool(
            args.candidate_prefix_reuse or args.cuda_graph_training
        ),
        "static_neutral_owner": bool(args.cuda_graph_training),
        "reuse_prepared_block_contexts": bool(args.reuse_prepared_block_contexts),
        "reuse_prepared_controller_context": bool(
            args.reuse_prepared_controller_context
        ),
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
        "cudnn_default_prewarm_setup_seconds": _finite_float(
            cudnn_prewarm_setup_seconds
        ),
        "activation_checkpointing_disabled": bool(
            args.disable_activation_checkpointing
        ),
        "raw_flow_checkpointing_disabled": bool(
            args.disable_raw_flow_checkpointing
            or args.disable_activation_checkpointing
        ),
        "raw_checkpoint_scopes_disabled": [
            scope
            for scope, selected in (
                ("pyramid", args.disable_raw_pyramid_checkpointing),
                ("mid", args.disable_raw_mid_checkpointing),
                ("high", args.disable_raw_high_checkpointing),
                ("context", args.disable_raw_context_checkpointing),
            )
            if selected
        ],
        "raw_mid_checkpoint_save_operations": list(raw_mid_save_operations),
        "p1_activation_checkpointing_disabled": bool(
            args.disable_p1_activation_checkpointing
            or args.disable_activation_checkpointing
        ),
        "compile_setup_seconds": _finite_float(compile_seconds),
        "visual_compile_setup_seconds": _finite_float(visual_compile_seconds),
        "execution_compile_setup_seconds": _finite_float(execution_compile_seconds),
        "mainline_compile_setup_seconds": _finite_float(mainline_compile_seconds),
        "candidate_compile_setup_seconds": _finite_float(candidate_compile_seconds),
        "measured_seconds": _finite_float(measured_seconds),
        "steps_per_second": float(measured_steps / max(measured_seconds, 1e-8)),
        "samples_per_second": float(measured_samples / max(measured_seconds, 1e-8)),
        "loss_first": float(step_losses[0]),
        "loss_last": float(step_losses[-1]),
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
    if torch_profile_table is not None:
        if args.torch_profile_output is None:
            raise RuntimeError("profiler table has no output path")
        args.torch_profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.torch_profile_output.write_text(torch_profile_table + "\n", encoding="utf-8")
        report["torch_profile_output"] = str(args.torch_profile_output)
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
