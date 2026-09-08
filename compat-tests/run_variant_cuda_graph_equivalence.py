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
    if args.selective_compile_profile is not None:
        os.environ["CLEARVLA_EQUIV_SELECTIVE_COMPILE_PROFILE"] = (
            args.selective_compile_profile
        )
    if args.allow_candidate_selective_compile:
        os.environ["CLEARVLA_EQUIV_ALLOW_CANDIDATE_SELECTIVE_COMPILE"] = "1"
    if args.context_reuse:
        os.environ["CLEARVLA_EQUIV_CONTEXT_REUSE"] = "1"
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
        adapter = resolve_training_acceleration_adapter(model)
        if adapter.name == "generic-static-v1":
            model.training_acceleration_adapter = MainlineTrainingAccelerationAdapter()
        optimizer, _ = build_optimizer(model, config)
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
    runpy.run_path(str(shared_gate), run_name="__main__")


if __name__ == "__main__":
    _run(_parser().parse_args())
