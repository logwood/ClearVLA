"""Run the shared CUDA-Graph equivalence gate against a version snapshot.

The authoritative gate lives in ``scripts/probe_local_training_cuda_graph_equivalence.py``.
This launcher supplies only the version-local model/config/batch factory, so Pen's
modular layout and RDT's legacy layout exercise the same comparisons without
copying either implementation into the test harness.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import os
import random
import runpy
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config-factory",
        default="_config",
        help=(
            "Factory in tests/test_mainline_policy.py used to construct the "
            "version-local test config (for example _bspine_config for Pen)."
        ),
    )
    parser.add_argument(
        "--selective-compile-profile",
        help="Pass a profile owned by the variant's acceleration adapter.",
    )
    parser.add_argument(
        "--allow-candidate-selective-compile",
        action="store_true",
    )
    parser.add_argument("--context-reuse", action="store_true")
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help="Build both equivalence engines with PyTorch fused AdamW.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help=(
            "Enable fixed-shape cuDNN algorithm benchmarking for this fresh "
            "process. Cross-backend certification uses separately exported "
            "default and benchmark traces."
        ),
    )
    parser.add_argument(
        "--cudnn-deterministic",
        action="store_true",
        help=(
            "Restrict cuDNN to deterministic algorithms for both sides of "
            "the equivalence comparison. This is a test-backend control; it "
            "does not relax any tensor tolerance."
        ),
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Require deterministic PyTorch kernels on both comparison sides. "
            "Unsupported CUDA operators fail loudly instead of weakening the "
            "equivalence threshold."
        ),
    )
    parser.add_argument(
        "--cudnn-benchmark-phase",
        choices=("forward", "backward"),
        help=(
            "Fresh-trace phase policy: enable global benchmark only for "
            "forward or only for backward. Forward may be combined with "
            "default guarded scopes."
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
            "Enable cuDNN algorithm search only inside one adapter-owned "
            "visual scope; repeat to combine scopes."
        ),
    )
    parser.add_argument(
        "--cudnn-default-prewarm-scope",
        action="append",
        default=[],
        choices=("raw_pyramid",),
        help=(
            "Before global cuDNN benchmark, seed one numerically sensitive "
            "scope with default same-shape forward/backward plans. This is "
            "supported only while exporting an independent backend trace."
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
            "Keep one visual owner's convolution forwards on benchmark-disabled "
            "plans; use a phase policy to control backward globally."
        ),
    )
    parser.add_argument(
        "--export-backend-trace",
        type=Path,
        help="Export a complete eager loss/gradient/update trace and exit.",
    )
    parser.add_argument(
        "--backend-trace-steps",
        type=int,
        default=4,
        help="Number of optimizer steps stored by --export-backend-trace.",
    )
    parser.add_argument(
        "--compare-backend-traces",
        type=Path,
        nargs=2,
        metavar=("REFERENCE", "CANDIDATE"),
        help="Compare two traces exported in independent fresh processes.",
    )
    parser.add_argument(
        "--backend-comparison-json",
        type=Path,
        help="Write the complete cross-backend comparison summary.",
    )
    parser.add_argument("--batched-raw-flow-sampling", action="store_true")
    parser.add_argument("--atomic-factual-contraction", action="store_true")
    parser.add_argument(
        "--raw-mid-checkpoint-save-operation",
        action="append",
        default=[],
        choices=("convolution", "linear", "grid_sample"),
    )
    parser.add_argument(
        "--disable-checkpoint-scope",
        action="append",
        default=[],
        choices=("p1", "raw_flow", "raw_pyramid", "raw_mid", "raw_high", "raw_context"),
    )
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
        return {name: _move(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _install_h5py_stub() -> None:
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("h5py")
        stub.File = type("File", (), {})
        stub.Dataset = type("Dataset", (), {})
        sys.modules["h5py"] = stub


def _run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if not (root / "clearvla").is_dir() or not (root / "tests").is_dir():
        raise ValueError(f"not a ClearVLA source snapshot: {root}")
    _install_h5py_stub()
    cudnn_default_prewarm_scopes = tuple(
        dict.fromkeys(args.cudnn_default_prewarm_scope)
    )
    cudnn_default_guard_scopes = tuple(
        dict.fromkeys(args.cudnn_default_guard_scope)
    )
    if args.cudnn_benchmark and args.cudnn_benchmark_scope:
        raise ValueError(
            "global --cudnn-benchmark cannot be combined with scoped benchmarking"
        )
    if args.cudnn_benchmark_phase and not args.cudnn_benchmark:
        raise ValueError("--cudnn-benchmark-phase requires --cudnn-benchmark")
    if args.cudnn_benchmark_phase and args.export_backend_trace is None:
        raise ValueError(
            "--cudnn-benchmark-phase requires --export-backend-trace"
        )
    if (
        cudnn_default_prewarm_scopes or cudnn_default_guard_scopes
    ) and not args.cudnn_benchmark:
        raise ValueError(
            "cuDNN default prewarm/guard scopes require --cudnn-benchmark"
        )
    if (
        cudnn_default_prewarm_scopes or cudnn_default_guard_scopes
    ) and args.export_backend_trace is None:
        raise ValueError(
            "cuDNN default prewarm/guard scopes require --export-backend-trace"
        )
    if args.backend_trace_steps <= 0:
        raise ValueError("--backend-trace-steps must be positive")
    torch.use_deterministic_algorithms(bool(args.deterministic_algorithms))
    torch.backends.cudnn.deterministic = bool(args.cudnn_deterministic)
    # A prewarm candidate must enter the shared exporter with benchmark still
    # disabled.  The exporter seeds the default plans from the engine's real
    # batch, then switches the global flag before the formal trace starts.
    torch.backends.cudnn.benchmark = bool(
        args.cudnn_benchmark
        and args.cudnn_benchmark_phase is None
        and not (cudnn_default_prewarm_scopes or cudnn_default_guard_scopes)
    )
    if args.cudnn_benchmark_phase is not None:
        os.environ["CLEARVLA_EQUIV_CUDNN_BENCHMARK_PHASE"] = (
            args.cudnn_benchmark_phase
        )
    if args.cudnn_benchmark_scope:
        os.environ["CLEARVLA_EQUIV_CUDNN_BENCHMARK_SCOPES"] = ",".join(
            dict.fromkeys(args.cudnn_benchmark_scope)
        )
    if cudnn_default_prewarm_scopes:
        os.environ["CLEARVLA_EQUIV_CUDNN_DEFAULT_PREWARM_SCOPES"] = ",".join(
            cudnn_default_prewarm_scopes
        )
    if cudnn_default_guard_scopes:
        os.environ["CLEARVLA_EQUIV_CUDNN_DEFAULT_GUARD_SCOPES"] = ",".join(
            cudnn_default_guard_scopes
        )
    if args.export_backend_trace is not None:
        os.environ["CLEARVLA_EQUIV_EXPORT_BACKEND_TRACE"] = str(
            args.export_backend_trace
        )
        os.environ["CLEARVLA_EQUIV_BACKEND_TRACE_STEPS"] = str(
            args.backend_trace_steps
        )
    if args.compare_backend_traces is not None:
        os.environ["CLEARVLA_EQUIV_COMPARE_BACKEND_TRACES"] = os.pathsep.join(
            str(path) for path in args.compare_backend_traces
        )
    if args.backend_comparison_json is not None:
        if args.compare_backend_traces is None:
            raise ValueError(
                "--backend-comparison-json requires --compare-backend-traces"
            )
        os.environ["CLEARVLA_EQUIV_BACKEND_COMPARISON_JSON"] = str(
            args.backend_comparison_json
        )
    if args.selective_compile_profile is not None:
        os.environ["CLEARVLA_EQUIV_SELECTIVE_COMPILE_PROFILE"] = (
            args.selective_compile_profile
        )
    if args.allow_candidate_selective_compile:
        os.environ["CLEARVLA_EQUIV_ALLOW_CANDIDATE_SELECTIVE_COMPILE"] = "1"
    if args.context_reuse:
        os.environ["CLEARVLA_EQUIV_CONTEXT_REUSE"] = "1"
    if args.batched_raw_flow_sampling:
        os.environ["CLEARVLA_EQUIV_BATCHED_RAW_FLOW_SAMPLING"] = "1"
    if args.atomic_factual_contraction:
        os.environ["CLEARVLA_EQUIV_ATOMIC_FACTUAL_CONTRACTION"] = "1"
    if args.raw_mid_checkpoint_save_operation:
        os.environ["CLEARVLA_EQUIV_RAW_MID_CHECKPOINT_SAVE_OPERATIONS"] = ",".join(
            dict.fromkeys(args.raw_mid_checkpoint_save_operation)
        )
    checkpoint_environment = {
        "p1": "CLEARVLA_EQUIV_DISABLE_P1_CHECKPOINT",
        "raw_flow": "CLEARVLA_EQUIV_DISABLE_RAW_FLOW_CHECKPOINT",
        "raw_pyramid": "CLEARVLA_EQUIV_DISABLE_RAW_PYRAMID_CHECKPOINT",
        "raw_mid": "CLEARVLA_EQUIV_DISABLE_RAW_MID_CHECKPOINT",
        "raw_high": "CLEARVLA_EQUIV_DISABLE_RAW_HIGH_CHECKPOINT",
        "raw_context": "CLEARVLA_EQUIV_DISABLE_RAW_CONTEXT_CHECKPOINT",
    }
    for scope in args.disable_checkpoint_scope:
        os.environ[checkpoint_environment[scope]] = "1"
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
    )
    from clearvla.mainline.training.cuda_graph import (
        _validated_static_capture_region,
    )
    from clearvla.mainline.training.engine import MainlineTrainingEngine, _autocast
    from clearvla.mainline.training.optimizer import (
        WarmupCosineSchedule,
        build_optimizer,
    )

    def engine_factory(device: torch.device) -> tuple[Any, Any]:
        config = config_factory()
        random.seed(9170)
        np.random.seed(9170)
        torch.manual_seed(9170)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(9170)
        model = ClearVLAMainlinePolicy(config).to(device).train()
        if args.raw_mid_checkpoint_save_operation:
            observation = getattr(model, "observation", None)
            compiler = getattr(observation, "compiler", None)
            encoder = getattr(compiler, "encoder", None)
            if encoder is None:
                encoder = getattr(observation, "encoder", None)
            raw_flow = getattr(encoder, "raw_flow", None)
            raw_mid = getattr(raw_flow, "mid", None)
            if raw_mid is None or not hasattr(raw_mid, "activation_checkpoint"):
                raise RuntimeError(
                    "variant gate cannot locate the raw-mid checkpoint surface"
                )
            # Test-sized configs may normally skip checkpointing.  Enable the
            # same checkpoint on both independently built engines so the
            # reference exercises full recomputation while only the candidate
            # installs selective saved operations.
            raw_mid.activation_checkpoint = True
        adapter = resolve_training_acceleration_adapter(model)
        if adapter.name == "generic-static-v1":
            model.training_acceleration_adapter = MainlineTrainingAccelerationAdapter()
        optimizer_kwargs: dict[str, Any] = {}
        if "fused" in inspect.signature(build_optimizer).parameters:
            optimizer_kwargs["fused"] = True if args.fused_adamw else None
        elif args.fused_adamw:
            raise ValueError(
                "this version's optimizer factory does not support fused AdamW"
            )
        optimizer, _ = build_optimizer(model, config, **optimizer_kwargs)
        schedule = WarmupCosineSchedule(
            optimizer,
            warmup_steps=2,
            total_steps=20,
            minimum_ratio=0.1,
        )
        signature = inspect.signature(MainlineTrainingEngine)
        kwargs: dict[str, Any] = {
            "model": model,
            "config": config,
            "optimizer": optimizer,
            "schedule": schedule,
            "device": device,
            "dtype": torch.float32,
        }
        if "train_flow_generator" in signature.parameters:
            kwargs["train_flow_generator"] = torch.Generator(device=device).manual_seed(
                9171
            )
        if "train_condition_generator" in signature.parameters:
            kwargs["train_condition_generator"] = torch.Generator(
                device=device
            ).manual_seed(9172)
        if "skip_postglobal_audit" in signature.parameters:
            kwargs["skip_postglobal_audit"] = True
        if "gradient_spike_audit_threshold" in signature.parameters:
            kwargs["gradient_spike_audit_threshold"] = None
        engine = MainlineTrainingEngine(**kwargs)
        resolve_training_acceleration_adapter(model).prepare(engine)
        return engine, _move(_batch(config, batch=2), device)

    compatibility_module = types.ModuleType("probe_local_training_cuda_graph")
    compatibility_module._autocast = _autocast
    compatibility_module._batch = _batch
    compatibility_module._engine = engine_factory
    compatibility_module._move = _move
    compatibility_module._source_root = root
    compatibility_module._validated_capture_region = (
        _validated_static_capture_region
    )
    sys.modules["probe_local_training_cuda_graph"] = compatibility_module

    shared_gate = Path(__file__).resolve().parents[1] / "scripts" / (
        "probe_local_training_cuda_graph_equivalence.py"
    )
    sys.argv = [str(shared_gate)]
    print(
        "equivalence_backend "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
        "cudnn_benchmark_scopes="
        f"{os.environ.get('CLEARVLA_EQUIV_CUDNN_BENCHMARK_SCOPES', '')!r} "
        "cudnn_default_prewarm_scopes="
        f"{os.environ.get('CLEARVLA_EQUIV_CUDNN_DEFAULT_PREWARM_SCOPES', '')!r} "
        "cudnn_default_guard_scopes="
        f"{os.environ.get('CLEARVLA_EQUIV_CUDNN_DEFAULT_GUARD_SCOPES', '')!r} "
        "cudnn_benchmark_phase="
        f"{os.environ.get('CLEARVLA_EQUIV_CUDNN_BENCHMARK_PHASE', '')!r} "
        f"cudnn_deterministic={torch.backends.cudnn.deterministic} "
        "deterministic_algorithms="
        f"{torch.are_deterministic_algorithms_enabled()} "
        f"matmul_precision={torch.get_float32_matmul_precision()}",
        flush=True,
    )
    runpy.run_path(str(shared_gate), run_name="__main__")


if __name__ == "__main__":
    _run(_parser().parse_args())
