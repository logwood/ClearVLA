"""Strict eager-versus-CUDA-graph training equivalence probe.

This is a research gate for the training-acceleration branch, not a production
runtime.  It captures the complete ordinary-batch forward, loss and backward,
then keeps the existing gradient lifecycle, optimizer and scheduler outside
the graph.  Every formal loss surface, raw/clipped gradient, parameter,
optimizer tensor and RNG continuation is compared after each replay.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import probe_local_training_cuda_graph as _probe_module  # noqa: E402
import torch
from probe_local_training_cuda_graph import (  # noqa: E402
    _autocast,
    _batch,
    _engine,
    _move,
    _validated_capture_region,
)

from clearvla.mainline.training.acceleration_contract import (  # noqa: E402
    resolve_training_acceleration_adapter,
)
from clearvla.mainline.training.cuda_graph import (  # noqa: E402
    CudaGraphTrainingStepRunner,
)

_SOURCE_ROOT = Path(getattr(_probe_module, "_source_root", Path.cwd())).resolve()


def _selective_compile_settings() -> tuple[str | None, bool]:
    profile = os.environ.get("CLEARVLA_EQUIV_SELECTIVE_COMPILE_PROFILE")
    legacy_mode = os.environ.get("CLEARVLA_EQUIV_COMPILE_SUBMODULES")
    legacy_profiles = {
        "1": "fast-combined",
        "visual": "fast-visual",
        "mainline": "fast-mainline",
        "mmdit": "fast-mmdit",
    }
    if profile and legacy_mode:
        raise ValueError(
            "set only CLEARVLA_EQUIV_SELECTIVE_COMPILE_PROFILE; the legacy "
            "compile switch cannot be combined with it"
        )
    if legacy_mode:
        if legacy_mode not in legacy_profiles:
            raise ValueError(
                "legacy compile mode cannot be represented by one version-owned "
                f"plan: {legacy_mode!r}"
            )
        profile = legacy_profiles[legacy_mode]
    allow_candidate = (
        os.environ.get("CLEARVLA_EQUIV_ALLOW_CANDIDATE_SELECTIVE_COMPILE") == "1"
    )
    if allow_candidate and not profile:
        raise ValueError(
            "candidate selective-compile authorization requires a compile profile"
        )
    return profile, allow_candidate


def _configure_compiled_graph_engine(engine: Any) -> None:
    """Enable only independently selected non-compile implementation rewrites."""

    adapter = resolve_training_acceleration_adapter(engine.model)
    # The small research engine does not attach a bespoke adapter at
    # construction time.  Install the mainline adapter here so optional
    # implementation-owned probes (including raw-flow batching) use the same
    # contract as the production profiler instead of reaching into the model
    # directly.
    if adapter.name == "generic-static-v1":
        from clearvla.mainline.training.acceleration_adapters import (
            MainlineTrainingAccelerationAdapter,
        )

        engine.model.training_acceleration_adapter = MainlineTrainingAccelerationAdapter()
        adapter = resolve_training_acceleration_adapter(engine.model)
    adapter.prepare(engine)
    if os.environ.get("CLEARVLA_EQUIV_ATOMIC_FACTUAL_CONTRACTION") == "1":
        hook = getattr(adapter, "set_atomic_factual_contraction", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose the atomic "
                "factual-contraction probe"
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
            (
                "raw_flow",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_RAW_FLOW_CHECKPOINT")
                == "1"
                or os.environ.get("CLEARVLA_EQUIV_DISABLE_ACTIVATION_CHECKPOINT")
                == "1",
            ),
            (
                "p1",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_P1_CHECKPOINT") == "1"
                or os.environ.get("CLEARVLA_EQUIV_DISABLE_ACTIVATION_CHECKPOINT")
                == "1",
            ),
            (
                "raw_pyramid",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_RAW_PYRAMID_CHECKPOINT")
                == "1",
            ),
            (
                "raw_mid",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_RAW_MID_CHECKPOINT") == "1",
            ),
            (
                "raw_high",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_RAW_HIGH_CHECKPOINT") == "1",
            ),
            (
                "raw_context",
                os.environ.get("CLEARVLA_EQUIV_DISABLE_RAW_CONTEXT_CHECKPOINT")
                == "1",
            ),
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
        changed = int(hook(engine, enabled=False, scopes=checkpoint_scopes))
        if changed == 0:
            raise RuntimeError(
                "activation-checkpoint disabling was requested but no supported "
                "checkpoint surfaces were found"
            )
    raw_mid_save_operations = tuple(
        name
        for name in os.environ.get(
            "CLEARVLA_EQUIV_RAW_MID_CHECKPOINT_SAVE_OPERATIONS", ""
        ).split(",")
        if name
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
    if os.environ.get("CLEARVLA_EQUIV_BATCHED_RAW_FLOW_SAMPLING") == "1":
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
    cudnn_benchmark_scopes = tuple(
        dict.fromkeys(
            scope
            for scope in os.environ.get(
                "CLEARVLA_EQUIV_CUDNN_BENCHMARK_SCOPES", ""
            ).split(",")
            if scope
        )
    )
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
    context_reuse = os.environ.get("CLEARVLA_EQUIV_CONTEXT_REUSE") == "1"
    if not context_reuse:
        return
    execution_bottom = getattr(engine.model, "execution_bottom", None)
    decoder = getattr(execution_bottom, "decoder", None)
    if decoder is None:
        bottom = getattr(engine.model, "bottom", None)
        decoder = getattr(bottom, "decoder", None)
    if decoder is None:
        raise RuntimeError("context-reuse gate cannot locate the execution decoder")
    decoder._reuse_prepared_block_contexts = True
    decoder._reuse_prepared_controller_context = True
    decoder._reuse_terminal_candidate_velocity = True


def _prewarm_cudnn_default_scopes(
    engine: Any,
    batch: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Seed and guard default plans before a global benchmark trace."""

    prewarm_scopes = tuple(
        dict.fromkeys(
            scope
            for scope in os.environ.get(
                "CLEARVLA_EQUIV_CUDNN_DEFAULT_PREWARM_SCOPES", ""
            ).split(",")
            if scope
        )
    )
    guard_scopes = tuple(
        dict.fromkeys(
            (
                *prewarm_scopes,
                *(
                    scope
                    for scope in os.environ.get(
                        "CLEARVLA_EQUIV_CUDNN_DEFAULT_GUARD_SCOPES", ""
                    ).split(",")
                    if scope
                ),
            )
        )
    )
    if not guard_scopes:
        return (), ()
    if torch.backends.cudnn.benchmark:
        raise RuntimeError(
            "default cuDNN prewarm must run before global benchmark is enabled"
        )
    adapter = resolve_training_acceleration_adapter(engine.model)
    if prewarm_scopes:
        hook = getattr(adapter, "prewarm_cudnn_default_scopes", None)
        if not callable(hook):
            raise RuntimeError(
                f"training adapter {adapter.name!r} does not expose default "
                "cuDNN prewarm"
            )
        changed = int(hook(engine, batch, scopes=prewarm_scopes))
        if changed != len(prewarm_scopes):
            raise RuntimeError(
                "default cuDNN prewarm did not cover every requested scope"
            )
    additional_guards = tuple(
        scope for scope in guard_scopes if scope not in prewarm_scopes
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
    benchmark_phase = os.environ.get("CLEARVLA_EQUIV_CUDNN_BENCHMARK_PHASE")
    torch.backends.cudnn.benchmark = benchmark_phase is None
    print(
        "cudnn_default_prewarm_ok "
        + json.dumps(
            {
                "prewarm_scopes": list(prewarm_scopes),
                "guard_scopes": list(guard_scopes),
                "benchmark_phase": benchmark_phase,
                "global_benchmark": bool(torch.backends.cudnn.benchmark),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return prewarm_scopes, guard_scopes


def _comparison_runner(engine: Any) -> CudaGraphTrainingStepRunner:
    profile, allow_candidate = _selective_compile_settings()
    runner_kwargs: dict[str, Any] = {
        "compile_profile": profile,
        "allow_candidate_compile": allow_candidate,
    }
    capture_gradient_lifecycle = (
        os.environ.get("CLEARVLA_EQUIV_CAPTURE_GRADIENT_LIFECYCLE") == "1"
    )
    runner_parameters = inspect.signature(CudaGraphTrainingStepRunner).parameters
    if "capture_gradient_lifecycle" in runner_parameters:
        runner_kwargs["capture_gradient_lifecycle"] = capture_gradient_lifecycle
    elif capture_gradient_lifecycle:
        raise ValueError(
            "this version's CUDA Graph runner does not support captured "
            "gradient lifecycle"
        )
    runner = CudaGraphTrainingStepRunner(engine, **runner_kwargs)
    if runner.compile_plan_summary is not None:
        print(
            "selective_compile_plan",
            json.dumps(runner.compile_plan_summary, sort_keys=True),
        )
    return runner


def _assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    except AssertionError as error:
        difference = (actual.detach().float() - expected.detach().float()).abs()
        expected_abs = expected.detach().float().abs()
        relative = difference / expected_abs.clamp_min(1e-30)
        outside = ~torch.isclose(
            actual.detach().float(),
            expected.detach().float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        outside_flat = outside.reshape(-1).nonzero().flatten()
        first_flat = int(outside_flat[0]) if outside_flat.numel() else -1
        if first_flat >= 0:
            if outside.ndim == 0:
                first_index = ()
            else:
                first_index_values: list[int] = []
                remainder = first_flat
                for size in reversed(outside.shape):
                    first_index_values.append(remainder % int(size))
                    remainder //= int(size)
                first_index = tuple(reversed(first_index_values))
            first_actual = float(actual.detach().float().reshape(-1)[first_flat])
            first_expected = float(expected.detach().float().reshape(-1)[first_flat])
            first_abs = float(difference.reshape(-1)[first_flat])
        else:
            first_index = ()
            first_actual = float("nan")
            first_expected = float("nan")
            first_abs = float("nan")
        raise AssertionError(
            f"{path}: max_abs={float(difference.max()):.9g} "
            f"max_rel={float(relative.max()):.9g} "
            f"actual_rms={float(actual.detach().float().square().mean().sqrt()):.9g} "
            f"expected_rms={float(expected.detach().float().square().mean().sqrt()):.9g} "
            f"first_outside_index={first_index!r} "
            f"first_actual={first_actual:.9g} first_expected={first_expected:.9g} "
            f"first_abs={first_abs:.9g}"
        ) from error


def _clone_tensor_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _assert_tensor_mapping_close(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    assert tuple(actual) == tuple(expected), (
        path,
        tuple(actual),
        tuple(expected),
    )
    for name in expected:
        _assert_tensor_close(
            actual[name],
            expected[name],
            path=f"{path}.{name}",
            rtol=rtol,
            atol=atol,
        )


def _named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_gradients_close(
    actual: Mapping[str, torch.Tensor | None],
    expected: Mapping[str, torch.Tensor | None],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    assert tuple(actual) == tuple(expected), path
    for name in expected:
        actual_value = actual[name]
        expected_value = expected[name]
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value, f"{path}.{name}"
            continue
        _assert_tensor_close(
            actual_value,
            expected_value,
            path=f"{path}.{name}",
            rtol=rtol,
            atol=atol,
        )


def _assert_parameters_close(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> None:
    for (actual_name, actual_parameter), (
        expected_name,
        expected_parameter,
    ) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        assert actual_name == expected_name
        _assert_tensor_close(
            actual_parameter,
            expected_parameter,
            path=f"parameter.{actual_name}",
            rtol=rtol,
            atol=atol,
        )


def _assert_optimizer_close(
    actual_engine: Any,
    expected_engine: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual_named = dict(actual_engine.model.named_parameters())
    expected_named = dict(expected_engine.model.named_parameters())
    assert tuple(actual_named) == tuple(expected_named)
    for name in expected_named:
        actual_state = actual_engine.optimizer.state.get(actual_named[name], {})
        expected_state = expected_engine.optimizer.state.get(expected_named[name], {})
        assert tuple(actual_state) == tuple(expected_state), f"optimizer.{name}"
        for state_name in expected_state:
            actual_value = actual_state[state_name]
            expected_value = expected_state[state_name]
            if isinstance(expected_value, torch.Tensor):
                assert isinstance(actual_value, torch.Tensor)
                _assert_tensor_close(
                    actual_value,
                    expected_value,
                    path=f"optimizer.{name}.{state_name}",
                    rtol=rtol,
                    atol=atol,
                )
            else:
                assert actual_value == expected_value, f"optimizer.{name}.{state_name}"
    assert len(actual_engine.optimizer.param_groups) == len(
        expected_engine.optimizer.param_groups
    )
    for index, (actual_group, expected_group) in enumerate(
        zip(
            actual_engine.optimizer.param_groups,
            expected_engine.optimizer.param_groups,
            strict=True,
        )
    ):
        for name in ("lr", "weight_decay", "betas", "eps", "name"):
            assert actual_group[name] == expected_group[name], (
                f"optimizer_group[{index}].{name}"
            )


def _tensor_pair_statistics(
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Summarize a tensor surface without hiding later mismatches.

    The strict assertion helpers intentionally stop at the first failed
    tensor.  That is useful as a tripwire, but it cannot calibrate a long-run
    numerical envelope.  This collector reduces every tensor on its own
    device, then transfers only scalar summaries to the host.
    """

    if not pairs:
        return {
            "tensor_count": 0,
            "element_count": 0,
            "different_elements": 0,
            "outside_elements": 0,
            "max_abs": 0.0,
            "max_abs_path": None,
            "max_rel": 0.0,
            "max_rel_path": None,
            "max_tolerance_ratio": 0.0,
            "max_tolerance_ratio_path": None,
            "rms_abs": 0.0,
            "relative_l2": 0.0,
            "first_outside_path": None,
            "first_outside_index": None,
            "first_outside_actual": None,
            "first_outside_expected": None,
            "first_outside_abs": None,
        }

    grouped: dict[
        str,
        dict[str, list[torch.Tensor] | list[str] | int],
    ] = {}
    element_count = 0
    first_outside: dict[str, Any] | None = None
    for path, actual, expected in pairs:
        if (
            tuple(actual.shape) != tuple(expected.shape)
            or actual.dtype != expected.dtype
            or actual.device != expected.device
        ):
            raise AssertionError(
                f"{path}: tensor contract differs "
                f"{tuple(actual.shape)}/{actual.dtype}/{actual.device} != "
                f"{tuple(expected.shape)}/{expected.dtype}/{expected.device}"
            )
        count = int(actual.numel())
        element_count += count
        if count == 0:
            continue
        actual_f = actual.detach().float()
        expected_f = expected.detach().float()
        difference = (actual_f - expected_f).abs()
        expected_abs = expected_f.abs()
        tolerance = expected_abs * float(rtol) + float(atol)
        tolerance_ratio = difference / tolerance.clamp_min(1e-30)
        relative = difference / expected_abs.clamp_min(1e-30)
        outside_mask = difference > tolerance
        outside_count = torch.count_nonzero(outside_mask)
        if first_outside is None and int(outside_count.item()) > 0:
            first_flat = int(outside_mask.reshape(-1).nonzero()[0].item())
            if outside_mask.ndim == 0:
                first_index: tuple[int, ...] = ()
            else:
                first_index_values: list[int] = []
                remainder = first_flat
                for size in reversed(outside_mask.shape):
                    first_index_values.append(remainder % int(size))
                    remainder //= int(size)
                first_index = tuple(reversed(first_index_values))
            first_outside = {
                "first_outside_path": path,
                "first_outside_index": list(first_index),
                "first_outside_actual": float(actual_f.reshape(-1)[first_flat]),
                "first_outside_expected": float(expected_f.reshape(-1)[first_flat]),
                "first_outside_abs": float(difference.reshape(-1)[first_flat]),
            }
        device_key = str(actual.device)
        group = grouped.setdefault(
            device_key,
            {
                "paths": [],
                "max_abs": [],
                "max_rel": [],
                "max_tolerance_ratio": [],
                "difference_square": [],
                "expected_square": [],
                "different_elements": [],
                "outside_elements": [],
                "element_count": 0,
            },
        )
        paths = group["paths"]
        assert isinstance(paths, list)
        paths.append(path)
        for name, value in (
            ("max_abs", difference.amax()),
            ("max_rel", relative.amax()),
            ("max_tolerance_ratio", tolerance_ratio.amax()),
            ("difference_square", difference.square().sum()),
            ("expected_square", expected_f.square().sum()),
            ("different_elements", torch.count_nonzero(difference)),
            ("outside_elements", outside_count),
        ):
            rows = group[name]
            assert isinstance(rows, list)
            rows.append(value)
        group["element_count"] = int(group["element_count"]) + count

    max_abs = -1.0
    max_abs_path: str | None = None
    max_rel = -1.0
    max_rel_path: str | None = None
    max_tolerance_ratio = -1.0
    max_tolerance_ratio_path: str | None = None
    difference_square = 0.0
    expected_square = 0.0
    different_elements = 0
    outside_elements = 0
    for group in grouped.values():
        paths = group["paths"]
        assert isinstance(paths, list)
        for field in ("max_abs", "max_rel", "max_tolerance_ratio"):
            rows = group[field]
            assert isinstance(rows, list)
            packed = torch.stack(rows)
            index = int(packed.argmax().item())
            value = float(packed[index].item())
            if field == "max_abs" and value > max_abs:
                max_abs = value
                max_abs_path = str(paths[index])
            elif field == "max_rel" and value > max_rel:
                max_rel = value
                max_rel_path = str(paths[index])
            elif field == "max_tolerance_ratio" and value > max_tolerance_ratio:
                max_tolerance_ratio = value
                max_tolerance_ratio_path = str(paths[index])
        for field in ("difference_square", "expected_square"):
            rows = group[field]
            assert isinstance(rows, list)
            value = float(torch.stack(rows).sum().item())
            if field == "difference_square":
                difference_square += value
            else:
                expected_square += value
        for field in ("different_elements", "outside_elements"):
            rows = group[field]
            assert isinstance(rows, list)
            value = int(torch.stack(rows).sum().item())
            if field == "different_elements":
                different_elements += value
            else:
                outside_elements += value

    result = {
        "tensor_count": len(pairs),
        "element_count": element_count,
        "different_elements": different_elements,
        "outside_elements": outside_elements,
        "max_abs": max(max_abs, 0.0),
        "max_abs_path": max_abs_path,
        "max_rel": max(max_rel, 0.0),
        "max_rel_path": max_rel_path,
        "max_tolerance_ratio": max(max_tolerance_ratio, 0.0),
        "max_tolerance_ratio_path": max_tolerance_ratio_path,
        "rms_abs": (difference_square / float(max(element_count, 1))) ** 0.5,
        "relative_l2": (difference_square / max(expected_square, 1e-30)) ** 0.5,
    }
    result.update(
        first_outside
        or {
            "first_outside_path": None,
            "first_outside_index": None,
            "first_outside_actual": None,
            "first_outside_expected": None,
            "first_outside_abs": None,
        }
    )
    return result


def _gradient_pair_statistics(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_named = dict(actual.named_parameters())
    expected_named = dict(expected.named_parameters())
    assert tuple(actual_named) == tuple(expected_named)
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    missing_mismatches: list[str] = []
    for name in expected_named:
        actual_value = actual_named[name].grad
        expected_value = expected_named[name].grad
        if actual_value is None or expected_value is None:
            if actual_value is not expected_value:
                missing_mismatches.append(name)
            continue
        pairs.append((name, actual_value, expected_value))
    result = _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)
    result["missing_mismatches"] = missing_mismatches
    return result


def _parameter_pair_statistics(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for (actual_name, actual_parameter), (
        expected_name,
        expected_parameter,
    ) in zip(actual.named_parameters(), expected.named_parameters(), strict=True):
        if actual_name != expected_name:
            raise AssertionError(
                f"model parameter names differ: {actual_name} != {expected_name}"
            )
        pairs.append((actual_name, actual_parameter, expected_parameter))
    return _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)


def _buffer_pair_statistics(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_named = dict(actual.named_buffers())
    expected_named = dict(expected.named_buffers())
    assert tuple(actual_named) == tuple(expected_named)
    return _tensor_pair_statistics(
        [
            (name, actual_named[name], expected_named[name])
            for name in expected_named
        ],
        rtol=rtol,
        atol=atol,
    )


def _optimizer_pair_statistics(
    actual_engine: Any,
    expected_engine: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    actual_named = dict(actual_engine.model.named_parameters())
    expected_named = dict(expected_engine.model.named_parameters())
    assert tuple(actual_named) == tuple(expected_named)
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    scalar_mismatches: list[str] = []
    for name in expected_named:
        actual_state = actual_engine.optimizer.state.get(actual_named[name], {})
        expected_state = expected_engine.optimizer.state.get(expected_named[name], {})
        if tuple(actual_state) != tuple(expected_state):
            scalar_mismatches.append(f"{name}.__state_keys__")
            continue
        for state_name in expected_state:
            actual_value = actual_state[state_name]
            expected_value = expected_state[state_name]
            path = f"{name}.{state_name}"
            if isinstance(expected_value, torch.Tensor):
                if not isinstance(actual_value, torch.Tensor):
                    scalar_mismatches.append(path)
                else:
                    pairs.append((path, actual_value, expected_value))
            elif actual_value != expected_value:
                scalar_mismatches.append(path)
    if len(actual_engine.optimizer.param_groups) != len(
        expected_engine.optimizer.param_groups
    ):
        scalar_mismatches.append("param_groups.__length__")
    else:
        for index, (actual_group, expected_group) in enumerate(
            zip(
                actual_engine.optimizer.param_groups,
                expected_engine.optimizer.param_groups,
                strict=True,
            )
        ):
            for name in ("lr", "weight_decay", "betas", "eps", "name"):
                if actual_group[name] != expected_group[name]:
                    scalar_mismatches.append(f"param_groups[{index}].{name}")
    result = _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)
    result["scalar_mismatches"] = scalar_mismatches
    return result


def _result_pair_statistics(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    metric_provenance: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    assert tuple(actual.metrics) == tuple(expected.metrics)
    if metric_provenance is not None and tuple(metric_provenance) != tuple(
        actual.metrics
    ):
        raise AssertionError("result metric provenance key contract differs")
    metric_names = tuple(
        name
        for name in expected.metrics
        if metric_provenance is None or metric_provenance[name]
    )
    pairs = [
        ("loss", actual.loss, expected.loss),
        ("gradient_norm", actual.gradient_norm, expected.gradient_norm),
        *[
            (f"metrics.{name}", actual.metrics[name], expected.metrics[name])
            for name in metric_names
        ],
    ]
    result = _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)
    result["training_metric_count"] = len(metric_names)
    result["detached_metric_count"] = (
        0 if metric_provenance is None else len(expected.metrics) - len(metric_names)
    )
    result["learning_rate_equal"] = actual.learning_rate == expected.learning_rate
    result["gradient_norm_scalar_abs"] = abs(
        float(actual.gradient_norm_scalar) - float(expected.gradient_norm_scalar)
    )
    return result


def _detached_result_metric_statistics(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    metric_provenance: Mapping[str, bool],
) -> dict[str, Any]:
    if (
        tuple(actual.metrics) != tuple(expected.metrics)
        or tuple(metric_provenance) != tuple(actual.metrics)
    ):
        raise AssertionError("detached metric provenance key contract differs")
    names = tuple(name for name, active in metric_provenance.items() if not active)
    return _tensor_pair_statistics(
        [
            (f"metrics.{name}", actual.metrics[name], expected.metrics[name])
            for name in names
        ],
        rtol=rtol,
        atol=atol,
    )


def _tree_tensor_pairs(
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    if isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor type differs")
        return [(path, actual, expected)]
    if dataclasses.is_dataclass(actual) and not isinstance(actual, type):
        if type(expected) is not type(actual):
            raise AssertionError(f"{path}: dataclass type differs")
        rows: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for field in dataclasses.fields(actual):
            rows.extend(
                _tree_tensor_pairs(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    path=f"{path}.{field.name}",
                )
            )
        return rows
    if isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or tuple(actual) != tuple(expected):
            raise AssertionError(f"{path}: mapping contract differs")
        rows = []
        for name in actual:
            rows.extend(
                _tree_tensor_pairs(
                    actual[name], expected[name], path=f"{path}.{name}"
                )
            )
        return rows
    if isinstance(actual, (list, tuple)):
        if type(expected) is not type(actual) or len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence contract differs")
        rows = []
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            rows.extend(
                _tree_tensor_pairs(
                    actual_item,
                    expected_item,
                    path=f"{path}[{index}]",
                )
            )
        return rows
    if actual != expected:
        raise AssertionError(f"{path}: static value differs")
    return []


def _assert_independent_equal_batches(actual: Any, expected: Any) -> tuple[int, int]:
    pairs = _tree_tensor_pairs(actual, expected, path="batch")
    element_count = 0
    for path, actual_value, expected_value in pairs:
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"{path}: generated batch values differ")
        if (
            actual_value.numel() > 0
            and actual_value.untyped_storage().data_ptr()
            == expected_value.untyped_storage().data_ptr()
        ):
            raise AssertionError(f"{path}: generated batches alias storage")
        element_count += int(actual_value.numel())
    return len(pairs), element_count


def _rng_state(engine: Any, device: torch.device) -> dict[str, torch.Tensor]:
    state = {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(device).clone(),
    }
    if engine.train_flow_generator is not None:
        state["flow"] = engine.train_flow_generator.get_state().clone()
    if engine.train_condition_generator is not None:
        state["condition"] = engine.train_condition_generator.get_state().clone()
    return state


def _set_rng_state(
    engine: Any,
    device: torch.device,
    state: Mapping[str, torch.Tensor],
) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    if engine.train_flow_generator is not None:
        engine.train_flow_generator.set_state(state["flow"])
    if engine.train_condition_generator is not None:
        engine.train_condition_generator.set_state(state["condition"])


def _ledger_snapshot(ledger: Any) -> dict[str, Any]:
    return {
        "total": ledger.total.detach().clone(),
        "groups": _clone_tensor_mapping(ledger.groups),
        "contributions": _clone_tensor_mapping(ledger.contributions),
        "terms": _clone_tensor_mapping(ledger.terms),
        "term_requires_grad": {
            name: bool(value.requires_grad)
            for name, value in ledger.terms.items()
        },
    }


def _assert_ledger_close(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
) -> None:
    _assert_tensor_close(
        actual.total,
        expected["total"],
        path="loss.total",
        rtol=rtol,
        atol=atol,
    )
    for field in ("groups", "contributions", "terms"):
        _assert_tensor_mapping_close(
            getattr(actual, field),
            expected[field],
            path=f"loss.{field}",
            rtol=rtol,
            atol=atol,
        )


def _eager_forward_backward(engine: Any, batch: Any) -> Any:
    engine.model.train()
    engine.model.set_training_step(engine.global_step)
    engine.optimizer.zero_grad(set_to_none=True)
    benchmark_phase = os.environ.get("CLEARVLA_EQUIV_CUDNN_BENCHMARK_PHASE")
    if benchmark_phase not in {None, "forward", "backward"}:
        raise ValueError(f"unknown cuDNN benchmark phase: {benchmark_phase!r}")
    try:
        if benchmark_phase is not None:
            torch.backends.cudnn.benchmark = benchmark_phase == "forward"
        with _autocast(engine.device, engine.dtype):
            ledger, _metrics = engine._forward(
                batch,
                training=True,
                collect_diagnostics=False,
                generator=engine.train_flow_generator,
                condition_generator=engine.train_condition_generator,
            )
        if benchmark_phase is not None:
            torch.backends.cudnn.benchmark = benchmark_phase == "backward"
        ledger.total.backward()
    finally:
        if benchmark_phase is not None:
            torch.backends.cudnn.benchmark = False
    return ledger


def _finish_update(engine: Any) -> tuple[torch.Tensor, float]:
    gradient_norm, _metrics, gradient_norm_scalar = engine._gradient_lifecycle(
        collect_diagnostics=False
    )
    engine.optimizer.step()
    engine.schedule.step()
    engine.global_step += 1
    return gradient_norm.detach().clone(), gradient_norm_scalar


def _cpu_trace_tree(value: Any) -> Any:
    """Clone a nested training-state surface into process-independent CPU storage."""

    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {name: _cpu_trace_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_trace_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_trace_tree(item) for item in value)
    return copy.deepcopy(value)


class _BackendFactualBoundaryCapture:
    """Record the causal P1 boundary without changing the trained graph.

    The factual reader returns ``(updated_trajectory, metrics)``.  The first
    tensor is the exact value consumed by the rest of P1/P2, while the metric
    mapping is detached audit state.  A module hook observes both after any
    version-owned compile wrapper has been installed; the tensor gradient hook
    observes the VJP arriving from the unchanged downstream loss.
    """

    _CANDIDATE_PATHS = (
        "p1.factual_reader",
        "factual_reader",
        "bottom.factual_reader",
    )

    def __init__(self, engine: Any) -> None:
        modules = dict(engine.model.named_modules())
        try:
            self.module_path = next(
                path for path in self._CANDIDATE_PATHS if path in modules
            )
        except StopIteration as error:
            raise RuntimeError(
                "backend trace cannot locate the factual P1 reader"
            ) from error
        self._calls: list[dict[str, Any]] = []
        self._gradients: list[torch.Tensor | None] = []
        self._handle = modules[self.module_path].register_forward_hook(
            self._record_forward
        )

    def begin_step(self) -> None:
        if self._calls or self._gradients:
            raise RuntimeError("factual boundary trace was not consumed")

    def _record_gradient(self, index: int, gradient: torch.Tensor) -> None:
        if self._gradients[index] is not None:
            raise RuntimeError("factual boundary gradient was recorded twice")
        self._gradients[index] = gradient.detach().clone()

    def _record_forward(
        self,
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], torch.Tensor)
            or not isinstance(output[1], Mapping)
        ):
            raise TypeError(
                "factual P1 boundary must return (Tensor, metric mapping)"
            )
        updated, metrics = output
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("factual P1 boundary has no trajectory tensor input")
        incoming = inputs[0]
        if tuple(incoming.shape) != tuple(updated.shape):
            raise ValueError("factual P1 input/output trajectory shapes differ")
        with torch.no_grad():
            row = {
                "output": updated.detach().clone(),
                "update": (updated.detach() - incoming.detach()).clone(),
                "metrics": {
                    name: value.detach().clone()
                    for name, value in metrics.items()
                    if isinstance(value, torch.Tensor)
                },
            }
        index = len(self._calls)
        self._calls.append(row)
        self._gradients.append(None)
        if updated.requires_grad:
            updated.register_hook(
                lambda gradient, index=index: self._record_gradient(
                    index, gradient
                )
            )

    def snapshot(self) -> list[dict[str, Any]]:
        if not self._calls:
            raise RuntimeError("formal forward did not cross the factual P1 boundary")
        rows: list[dict[str, Any]] = []
        for call, gradient in zip(self._calls, self._gradients, strict=True):
            rows.append({**call, "output_gradient": gradient})
        self._calls = []
        self._gradients = []
        return rows

    def close(self) -> None:
        self._handle.remove()


def _export_backend_trace(path: Path, *, steps: int = 4) -> None:
    """Export one fresh trajectory for cross-process backend comparison.

    cuDNN's algorithm cache is process-global.  Comparing a default and a
    benchmark engine inside one process can accidentally reuse the first
    engine's cached plan.  Independent trace processes avoid that ambiguity
    while retaining the complete loss, raw/clipped gradient, parameter,
    optimizer, scheduler and RNG surfaces.  When a selective-compile profile
    is requested, construct its runner only to install the version-owned
    module transforms; execute the exported steps on the ordinary default
    stream.  This isolates compilation arithmetic from CUDA-Graph stream and
    capture effects without changing the eager reference path.
    """

    if path.exists():
        raise FileExistsError(f"backend trace already exists: {path}")
    if steps <= 0:
        raise ValueError("backend trace steps must be positive")
    device = torch.device("cuda")
    engine, prewarm_batch = _engine(device)
    _configure_compiled_graph_engine(engine)
    compile_profile, _allow_candidate = _selective_compile_settings()
    compile_plan_summary: dict[str, Any] | None = None
    if compile_profile is not None:
        compile_runner = _comparison_runner(engine)
        compile_plan_summary = compile_runner.compile_plan_summary
        if compile_plan_summary is None:
            raise RuntimeError(
                "selective compile was requested but no plan was installed"
            )
    (
        cudnn_default_prewarm_scopes,
        cudnn_default_guard_scopes,
    ) = _prewarm_cudnn_default_scopes(
        engine,
        prewarm_batch,
    )
    initial_model = _cpu_trace_tree(engine.model.state_dict())
    initial_optimizer = _cpu_trace_tree(engine.optimizer.state_dict())
    initial_schedule = copy.deepcopy(engine.schedule.state_dict())
    initial_rng = _cpu_trace_tree(_rng_state(engine, device))
    factual_boundary = _BackendFactualBoundaryCapture(engine)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(steps):
            batch = _fresh_long_batch(
                engine.config,
                device,
                batch_size=2,
                index=20_000 + index,
            )
            factual_boundary.begin_step()
            ledger = _eager_forward_backward(engine, batch)
            ledger_snapshot = _cpu_trace_tree(_ledger_snapshot(ledger))
            factual_snapshot = _cpu_trace_tree(factual_boundary.snapshot())
            raw_gradients = _cpu_trace_tree(_named_gradients(engine.model))
            gradient_norm, gradient_norm_scalar = _finish_update(engine)
            torch.cuda.synchronize(device)
            rows.append(
                {
                    "step": index,
                    "ledger": ledger_snapshot,
                    "factual_boundary": factual_snapshot,
                    "raw_gradients": raw_gradients,
                    "gradient_norm": _cpu_trace_tree(gradient_norm),
                    "gradient_norm_scalar": float(gradient_norm_scalar),
                    "clipped_gradients": _cpu_trace_tree(
                        _named_gradients(engine.model)
                    ),
                    "model": _cpu_trace_tree(engine.model.state_dict()),
                    "optimizer": _cpu_trace_tree(engine.optimizer.state_dict()),
                    "schedule": copy.deepcopy(engine.schedule.state_dict()),
                    "global_step": int(engine.global_step),
                    "rng": _cpu_trace_tree(_rng_state(engine, device)),
                }
            )
    finally:
        factual_boundary.close()
    trace = {
        "schema": "clearvla-training-backend-trace-v3",
        "backend": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_benchmark_scopes": [
                scope
                for scope in os.environ.get(
                    "CLEARVLA_EQUIV_CUDNN_BENCHMARK_SCOPES", ""
                ).split(",")
                if scope
            ],
            "cudnn_default_prewarm_scopes": list(
                cudnn_default_prewarm_scopes
            ),
            "cudnn_default_guard_scopes": list(cudnn_default_guard_scopes),
            "cudnn_benchmark_phase": os.environ.get(
                "CLEARVLA_EQUIV_CUDNN_BENCHMARK_PHASE"
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "selective_compile_plan": compile_plan_summary,
            "factual_boundary_path": factual_boundary.module_path,
        },
        "steps": rows,
        "initial_model": initial_model,
        "initial_optimizer": initial_optimizer,
        "initial_schedule": initial_schedule,
        "initial_rng": initial_rng,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        "backend_trace_exported",
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "steps": steps,
                **trace["backend"],
            },
            sort_keys=True,
        ),
    )


def _backend_trace_surface_statistics(
    actual: Any,
    expected: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    pairs = _tree_tensor_pairs(actual, expected, path=path)
    return _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)


def _backend_trace_named_statistics(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Report a mapping summary plus every tensor that actually changed."""

    if tuple(actual) != tuple(expected):
        raise AssertionError(f"{path}: mapping contract differs")
    summary = _backend_trace_surface_statistics(
        actual,
        expected,
        path=path,
        rtol=rtol,
        atol=atol,
    )
    changed: dict[str, Any] = {}
    for name in actual:
        statistics = _backend_trace_surface_statistics(
            actual[name],
            expected[name],
            path=f"{path}.{name}",
            rtol=rtol,
            atol=atol,
        )
        if int(statistics["different_elements"]) > 0:
            changed[name] = statistics
    return {"summary": summary, "changed": changed}


def _backend_trace_ledger_statistics(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Separate formal objectives from detached/raw diagnostic terms."""

    base_fields = ("total", "groups", "contributions", "terms")
    extended_fields = (*base_fields, "term_requires_grad")
    if tuple(actual) not in {base_fields, extended_fields} or tuple(
        expected
    ) != tuple(actual):
        raise AssertionError(f"{path}: loss-ledger contract differs")
    report = {
        "total": _backend_trace_surface_statistics(
            actual["total"],
            expected["total"],
            path=f"{path}.total",
            rtol=rtol,
            atol=atol,
        ),
        "groups": _backend_trace_named_statistics(
            actual["groups"],
            expected["groups"],
            path=f"{path}.groups",
            rtol=rtol,
            atol=atol,
        ),
        "contributions": _backend_trace_named_statistics(
            actual["contributions"],
            expected["contributions"],
            path=f"{path}.contributions",
            rtol=rtol,
            atol=atol,
        ),
        "terms": _backend_trace_named_statistics(
            actual["terms"],
            expected["terms"],
            path=f"{path}.terms",
            rtol=rtol,
            atol=atol,
        ),
    }
    if tuple(actual) == extended_fields:
        actual_provenance = actual["term_requires_grad"]
        expected_provenance = expected["term_requires_grad"]
        if actual_provenance != expected_provenance:
            raise AssertionError(f"{path}: loss-term gradient provenance differs")
        trainable_names = tuple(
            name for name, active in actual_provenance.items() if active
        )
        detached_names = tuple(
            name for name, active in actual_provenance.items() if not active
        )
        report.update(
            {
                "term_requires_grad": actual_provenance,
                "trainable_terms": _backend_trace_named_statistics(
                    {name: actual["terms"][name] for name in trainable_names},
                    {name: expected["terms"][name] for name in trainable_names},
                    path=f"{path}.trainable_terms",
                    rtol=rtol,
                    atol=atol,
                ),
                "detached_terms": _backend_trace_named_statistics(
                    {name: actual["terms"][name] for name in detached_names},
                    {name: expected["terms"][name] for name in detached_names},
                    path=f"{path}.detached_terms",
                    rtol=rtol,
                    atol=atol,
                ),
            }
        )
    else:
        report.update(
            {
                "term_requires_grad": None,
                "trainable_terms": None,
                "detached_terms": None,
            }
        )
    return report


def _backend_trace_factual_statistics(
    actual: list[Mapping[str, Any]],
    expected: list[Mapping[str, Any]],
    *,
    path: str,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    """Split the causal P1 tensor/VJP from its detached metric sidecar."""

    if len(actual) != len(expected):
        raise AssertionError(f"{path}: factual P1 call counts differ")
    required = ("output", "update", "metrics", "output_gradient")
    reports: list[dict[str, Any]] = []
    for index, (actual_call, expected_call) in enumerate(
        zip(actual, expected, strict=True)
    ):
        call_path = f"{path}[{index}]"
        if tuple(actual_call) != required or tuple(expected_call) != required:
            raise AssertionError(f"{call_path}: factual P1 contract differs")
        reports.append(
            {
                "output": _backend_trace_surface_statistics(
                    actual_call["output"],
                    expected_call["output"],
                    path=f"{call_path}.output",
                    rtol=rtol,
                    atol=atol,
                ),
                "update": _backend_trace_surface_statistics(
                    actual_call["update"],
                    expected_call["update"],
                    path=f"{call_path}.update",
                    rtol=rtol,
                    atol=atol,
                ),
                "output_gradient": _backend_trace_surface_statistics(
                    actual_call["output_gradient"],
                    expected_call["output_gradient"],
                    path=f"{call_path}.output_gradient",
                    rtol=rtol,
                    atol=atol,
                ),
                "metrics": _backend_trace_named_statistics(
                    actual_call["metrics"],
                    expected_call["metrics"],
                    path=f"{call_path}.metrics",
                    rtol=rtol,
                    atol=atol,
                ),
            }
        )
    return reports


def _compare_backend_traces(
    reference_path: Path,
    candidate_path: Path,
    *,
    rtol: float,
    atol: float,
) -> None:
    """Compare independent default/benchmark trajectories without cache leakage."""

    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    supported_schemas = {
        "clearvla-training-backend-trace-v1",
        "clearvla-training-backend-trace-v2",
        "clearvla-training-backend-trace-v3",
    }
    if reference.get("schema") not in supported_schemas:
        raise ValueError("reference backend trace has an unsupported schema")
    if candidate.get("schema") != reference.get("schema"):
        raise ValueError("candidate backend trace schema differs")
    reference_rows = reference.get("steps")
    candidate_rows = candidate.get("steps")
    if not isinstance(reference_rows, list) or not isinstance(candidate_rows, list):
        raise TypeError("backend traces must contain step lists")
    if len(reference_rows) != len(candidate_rows):
        raise AssertionError("backend trace step counts differ")

    initial = {
        "model": _backend_trace_surface_statistics(
            candidate["initial_model"],
            reference["initial_model"],
            path="initial.model",
            rtol=0.0,
            atol=0.0,
        ),
        "optimizer": _backend_trace_surface_statistics(
            candidate["initial_optimizer"],
            reference["initial_optimizer"],
            path="initial.optimizer",
            rtol=0.0,
            atol=0.0,
        ),
        "rng": _backend_trace_surface_statistics(
            candidate["initial_rng"],
            reference["initial_rng"],
            path="initial.rng",
            rtol=0.0,
            atol=0.0,
        ),
    }
    if candidate["initial_schedule"] != reference["initial_schedule"]:
        raise AssertionError("initial scheduler states differ")

    trace_has_factual_boundary = reference.get("schema") in {
        "clearvla-training-backend-trace-v2",
        "clearvla-training-backend-trace-v3",
    }
    if trace_has_factual_boundary and (
        candidate["backend"].get("factual_boundary_path")
        != reference["backend"].get("factual_boundary_path")
    ):
        raise AssertionError("factual P1 boundary paths differ")

    reports: list[dict[str, Any]] = []
    failed_surfaces: list[str] = []
    failed_training_surfaces: list[str] = []
    failed_diagnostic_surfaces: list[str] = []
    for index, (actual, expected) in enumerate(
        zip(candidate_rows, reference_rows, strict=True)
    ):
        if int(actual["step"]) != int(expected["step"]):
            raise AssertionError(f"backend trace step identity differs at {index}")
        if int(actual["global_step"]) != int(expected["global_step"]):
            raise AssertionError(f"backend global step differs at {index}")
        if actual["schedule"] != expected["schedule"]:
            raise AssertionError(f"backend scheduler state differs at {index}")
        scalar_difference = abs(
            float(actual["gradient_norm_scalar"])
            - float(expected["gradient_norm_scalar"])
        )
        scalar_tolerance = atol + rtol * abs(
            float(expected["gradient_norm_scalar"])
        )
        surfaces = {
            name: _backend_trace_surface_statistics(
                actual[name],
                expected[name],
                path=f"step[{index}].{name}",
                rtol=(0.0 if name == "rng" else rtol),
                atol=(0.0 if name == "rng" else atol),
            )
            for name in (
                "ledger",
                "raw_gradients",
                "gradient_norm",
                "clipped_gradients",
                "model",
                "optimizer",
                "rng",
            )
        }
        if trace_has_factual_boundary:
            surfaces["factual_boundary"] = _backend_trace_surface_statistics(
                actual["factual_boundary"],
                expected["factual_boundary"],
                path=f"step[{index}].factual_boundary",
                rtol=rtol,
                atol=atol,
            )
            factual_boundary_sections = _backend_trace_factual_statistics(
                actual["factual_boundary"],
                expected["factual_boundary"],
                path=f"step[{index}].factual_boundary",
                rtol=rtol,
                atol=atol,
            )
        else:
            factual_boundary_sections = []
        ledger_sections = _backend_trace_ledger_statistics(
            actual["ledger"],
            expected["ledger"],
            path=f"step[{index}].ledger",
            rtol=rtol,
            atol=atol,
        )
        if scalar_difference > scalar_tolerance:
            failed_surfaces.append(f"step[{index}].gradient_norm_scalar")
            failed_training_surfaces.append(
                f"step[{index}].gradient_norm_scalar"
            )
        for name, statistics in surfaces.items():
            if int(statistics["outside_elements"]) > 0:
                failed_surfaces.append(f"step[{index}].{name}")
                if name not in {"ledger", "factual_boundary"}:
                    failed_training_surfaces.append(f"step[{index}].{name}")
        formal_ledger_sections = (
            ("total", ledger_sections["total"]),
            ("groups", ledger_sections["groups"]["summary"]),
            (
                "contributions",
                ledger_sections["contributions"]["summary"],
            ),
        )
        for name, statistics in formal_ledger_sections:
            if int(statistics["outside_elements"]) > 0:
                failed_training_surfaces.append(
                    f"step[{index}].ledger.{name}"
                )
        trainable_terms = ledger_sections["trainable_terms"]
        detached_terms = ledger_sections["detached_terms"]
        if trainable_terms is not None and int(
            trainable_terms["summary"]["outside_elements"]
        ) > 0:
            failed_training_surfaces.append(
                f"step[{index}].ledger.trainable_terms"
            )
        if detached_terms is not None:
            if int(detached_terms["summary"]["outside_elements"]) > 0:
                failed_diagnostic_surfaces.append(
                    f"step[{index}].ledger.detached_terms"
                )
        elif int(
            ledger_sections["terms"]["summary"]["outside_elements"]
        ) > 0:
            failed_diagnostic_surfaces.append(
                f"step[{index}].ledger.unclassified_terms"
            )
        for call_index, factual_sections in enumerate(
            factual_boundary_sections
        ):
            for name in ("output", "update", "output_gradient"):
                if int(factual_sections[name]["outside_elements"]) > 0:
                    failed_training_surfaces.append(
                        f"step[{index}].factual_boundary[{call_index}].{name}"
                    )
            if int(
                factual_sections["metrics"]["summary"]["outside_elements"]
            ) > 0:
                failed_diagnostic_surfaces.append(
                    f"step[{index}].factual_boundary[{call_index}].metrics"
                )
        reports.append(
            {
                "step": index,
                "gradient_norm_scalar_abs": scalar_difference,
                "gradient_norm_scalar_tolerance": scalar_tolerance,
                "surfaces": surfaces,
                "ledger_sections": ledger_sections,
                "factual_boundary_sections": factual_boundary_sections,
            }
        )

    for name, statistics in initial.items():
        if int(statistics["outside_elements"]) > 0:
            failed_surfaces.append(f"initial.{name}")
            failed_training_surfaces.append(f"initial.{name}")
    summary = {
        "schema": "clearvla-training-backend-comparison-v2",
        "reference": {
            "path": str(reference_path),
            "backend": reference["backend"],
        },
        "candidate": {
            "path": str(candidate_path),
            "backend": candidate["backend"],
        },
        "rtol": rtol,
        "atol": atol,
        "initial": initial,
        "steps": reports,
        "failed_surfaces": failed_surfaces,
        "failed_training_surfaces": failed_training_surfaces,
        "failed_diagnostic_surfaces": failed_diagnostic_surfaces,
        "training_state_accepted": not failed_training_surfaces,
        "strict_observability_accepted": not failed_surfaces,
        # Mathematical training equivalence is owned by the differentiable
        # objective, P1 boundary/VJP, gradients, parameters, optimizer and RNG.
        # Detached metrics remain visible under the stricter observability
        # status, but an eager-vs-eager diagnostic reduction cannot veto an
        # otherwise identical optimizer trajectory.
        "accepted": not failed_training_surfaces,
    }
    output_value = os.environ.get("CLEARVLA_EQUIV_BACKEND_COMPARISON_JSON")
    if output_value:
        output_path = Path(output_value)
        if output_path.exists():
            raise FileExistsError(
                f"backend comparison output already exists: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("backend_trace_comparison", json.dumps(summary, sort_keys=True))
    if failed_training_surfaces:
        raise AssertionError(
            "cross-backend trace exceeded the mathematical training gate: "
            + ", ".join(failed_training_surfaces)
        )
    if failed_diagnostic_surfaces:
        print(
            "backend_detached_observability_drift",
            json.dumps(failed_diagnostic_surfaces),
        )
    print("cudnn_backend_equivalence_ok")


def _capture_forward_backward(engine: Any, batch: Any) -> tuple[torch.cuda.CUDAGraph, Any]:
    engine.model.train()
    engine.model.set_training_step(engine.global_step)
    initial_rng = _rng_state(engine, engine.device)

    # Allocate stable gradient buffers and initialize CUDA libraries.
    engine.optimizer.zero_grad(set_to_none=True)
    with _autocast(engine.device, engine.dtype):
        warm_ledger, warm_metrics = engine._forward(
            batch,
            training=True,
            collect_diagnostics=False,
            generator=engine.train_flow_generator,
            condition_generator=engine.train_condition_generator,
        )
    warm_ledger.total.backward()
    engine.optimizer.zero_grad(set_to_none=False)
    del warm_ledger, warm_metrics
    torch.cuda.synchronize(engine.device)

    graph = torch.cuda.CUDAGraph()
    if engine.train_flow_generator is not None:
        graph.register_generator_state(engine.train_flow_generator)
    if engine.train_condition_generator is not None:
        graph.register_generator_state(engine.train_condition_generator)
    with _validated_capture_region():
        with torch.cuda.graph(graph):
            engine.optimizer.zero_grad(set_to_none=False)
            with _autocast(engine.device, engine.dtype):
                captured_ledger, captured_metrics = engine._forward(
                    batch,
                    training=True,
                    collect_diagnostics=False,
                    generator=engine.train_flow_generator,
                    condition_generator=engine.train_condition_generator,
                )
            captured_ledger.total.backward()
    # The captured outputs and metrics must stay alive for all replays.
    setattr(engine, "_cuda_graph_captured_metrics", captured_metrics)
    _set_rng_state(engine, engine.device, initial_rng)
    torch.cuda.synchronize(engine.device)
    return graph, captured_ledger


def _run_gate(*, start_step: int, steps: int, rtol: float, atol: float) -> None:
    device = torch.device("cuda")
    eager_engine, eager_batch = _engine(device)
    graph_engine, graph_batch = _engine(device)
    _configure_compiled_graph_engine(graph_engine)
    eager_engine.global_step = start_step
    graph_engine.global_step = start_step
    eager_engine.schedule.step_index = start_step
    graph_engine.schedule.step_index = start_step
    eager_engine.schedule._apply_current_ratio()
    graph_engine.schedule._apply_current_ratio()

    _assert_parameters_close(graph_engine.model, eager_engine.model, rtol=0.0, atol=0.0)
    graph_engine.model.set_training_step(graph_engine.global_step)
    # Use the same adapter-backed capture backend as production.  The older
    # direct helper remains available for isolated diagnostics, but comparing
    # it here would test a second graph implementation that is not deployed.
    graph_runner = _comparison_runner(graph_engine)
    graph_runner._ensure_capture(graph_batch)
    graph = graph_runner._graph
    captured_ledger = graph_runner._captured_ledger
    if graph is None or captured_ledger is None:
        raise RuntimeError("CUDA Graph gate capture is incomplete")

    base_rng = _rng_state(graph_engine, device)
    eager_rng = {name: value.clone() for name, value in base_rng.items()}
    graph_rng = {name: value.clone() for name, value in base_rng.items()}
    for local_step in range(steps):
        _set_rng_state(eager_engine, device, eager_rng)
        eager_ledger = _eager_forward_backward(eager_engine, eager_batch)
        expected_ledger = _ledger_snapshot(eager_ledger)
        expected_raw_gradients = _named_gradients(eager_engine.model)
        eager_norm, eager_norm_scalar = _finish_update(eager_engine)
        expected_clipped_gradients = _named_gradients(eager_engine.model)
        eager_rng = _rng_state(eager_engine, device)

        _set_rng_state(graph_engine, device, graph_rng)
        graph_engine.model.set_training_step(graph_engine.global_step)
        graph.replay()
        torch.cuda.synchronize(device)
        _assert_ledger_close(
            captured_ledger,
            expected_ledger,
            rtol=rtol,
            atol=atol,
        )
        _assert_gradients_close(
            _named_gradients(graph_engine.model),
            expected_raw_gradients,
            path="raw_gradient",
            rtol=rtol,
            atol=atol,
        )
        graph_norm, graph_norm_scalar = _finish_update(graph_engine)
        _assert_tensor_close(
            graph_norm,
            eager_norm,
            path="gradient_norm",
            rtol=rtol,
            atol=atol,
        )
        assert abs(graph_norm_scalar - eager_norm_scalar) <= atol + rtol * abs(
            eager_norm_scalar
        )
        _assert_gradients_close(
            _named_gradients(graph_engine.model),
            expected_clipped_gradients,
            path="clipped_gradient",
            rtol=rtol,
            atol=atol,
        )
        _assert_parameters_close(
            graph_engine.model,
            eager_engine.model,
            rtol=rtol,
            atol=atol,
        )
        _assert_optimizer_close(
            graph_engine,
            eager_engine,
            rtol=rtol,
            atol=atol,
        )
        assert graph_engine.global_step == eager_engine.global_step
        assert graph_engine.schedule.step_index == eager_engine.schedule.step_index
        graph_rng = _rng_state(graph_engine, device)
        _assert_tensor_mapping_close(
            graph_rng,
            eager_rng,
            path="rng",
            rtol=0.0,
            atol=0.0,
        )
        print(
            "equivalent_step",
            start_step + local_step,
            "loss",
            float(expected_ledger["total"]),
            "gradient_norm",
            eager_norm_scalar,
        )


def _install_metric_provenance_recorder(engine: Any) -> dict[str, bool]:
    """Record which emitted scalar metrics belong to the training objective.

    ``TrainStepResult.metrics`` is detached by design, so inspecting the result
    after the step cannot distinguish a differentiable raw loss term from an
    audit-only scalar.  Wrap the engine's existing metric materializer and
    retain that provenance while the live ledger is still available.  The
    wrapper does not add tensor work or modify any returned value.
    """

    original = engine._tensor_metrics
    provenance: dict[str, bool] = {}

    def recorded(ledger: Any, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = original(ledger, values)
        current = {name: False for name in result}
        current["loss_total"] = True
        for name in ledger.groups:
            current[f"loss_group_{name}"] = True
        for name in ledger.contributions:
            current[f"loss_contrib_{name}"] = True
        for name, value in ledger.terms.items():
            current[f"loss_{name}"] = bool(value.requires_grad)
        # Model-side values are observations even if their source tensor is
        # attached elsewhere; only LossLedger ownership can enter backward.
        for name in values:
            if name in current:
                current[name] = False
        current["loss_ledger_gap"] = True
        current["loss_contribution_gap"] = True
        if tuple(current) != tuple(result):
            raise AssertionError("training-result metric provenance differs")
        provenance.clear()
        provenance.update(current)
        return result

    engine._tensor_metrics = recorded
    return provenance


def _assert_step_result_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
    path: str = "result",
    actual_metric_provenance: Mapping[str, bool] | None = None,
    expected_metric_provenance: Mapping[str, bool] | None = None,
) -> None:
    _assert_tensor_close(
        actual.loss,
        expected.loss,
        path=f"{path}.loss",
        rtol=rtol,
        atol=atol,
    )
    _assert_tensor_close(
        actual.gradient_norm,
        expected.gradient_norm,
        path=f"{path}.gradient_norm",
        rtol=rtol,
        atol=atol,
    )
    assert actual.learning_rate == expected.learning_rate
    assert actual.gradient_norm_scalar is not None
    assert expected.gradient_norm_scalar is not None
    assert abs(actual.gradient_norm_scalar - expected.gradient_norm_scalar) <= (
        atol + rtol * abs(expected.gradient_norm_scalar)
    )
    if actual_metric_provenance is None and expected_metric_provenance is None:
        _assert_tensor_mapping_close(
            actual.metrics,
            expected.metrics,
            path=f"{path}.metrics",
            rtol=rtol,
            atol=atol,
        )
        return
    if actual_metric_provenance is None or expected_metric_provenance is None:
        raise AssertionError(f"{path}: only one metric provenance map was recorded")
    if (
        tuple(actual_metric_provenance) != tuple(actual.metrics)
        or tuple(expected_metric_provenance) != tuple(expected.metrics)
    ):
        raise AssertionError(f"{path}: metric provenance key contract differs")
    if actual_metric_provenance != expected_metric_provenance:
        raise AssertionError(f"{path}: metric gradient provenance differs")
    training_names = tuple(
        name for name, active in actual_metric_provenance.items() if active
    )
    diagnostic_names = tuple(
        name for name, active in actual_metric_provenance.items() if not active
    )
    _assert_tensor_mapping_close(
        {name: actual.metrics[name] for name in training_names},
        {name: expected.metrics[name] for name in training_names},
        path=f"{path}.training_metrics",
        rtol=rtol,
        atol=atol,
    )
    diagnostic_statistics = _backend_trace_named_statistics(
        {name: actual.metrics[name] for name in diagnostic_names},
        {name: expected.metrics[name] for name in diagnostic_names},
        path=f"{path}.detached_observability",
        rtol=rtol,
        atol=atol,
    )
    if int(diagnostic_statistics["summary"]["outside_elements"]) > 0:
        outside = {
            name: statistics
            for name, statistics in diagnostic_statistics["changed"].items()
            if int(statistics["outside_elements"]) > 0
        }
        print(
            "step_result_detached_observability_drift",
            json.dumps(
                {
                    "path": path,
                    "summary": diagnostic_statistics["summary"],
                    "outside": outside,
                },
                sort_keys=True,
            ),
        )


def _run_production_runner_gate(*, rtol: float, atol: float) -> None:
    """Exercise input copies, eager diagnostics and topology recapture."""

    device = torch.device("cuda")
    eager_engine, first_batch = _engine(device)
    graph_engine, _unused_batch = _engine(device)
    eager_metric_provenance = _install_metric_provenance_recorder(eager_engine)
    graph_metric_provenance = _install_metric_provenance_recorder(graph_engine)
    _configure_compiled_graph_engine(graph_engine)
    runner = _comparison_runner(graph_engine)
    _assert_parameters_close(graph_engine.model, eager_engine.model, rtol=0.0, atol=0.0)

    torch.manual_seed(9340)
    second_batch = _move(_batch(eager_engine.config, batch=2), device)
    # Capture without taking a logical step, then make the first real update a
    # complete eager diagnostic.  The following replay proves that diagnostic
    # zeroing did not invalidate the graph-owned gradient buffers.
    graph_engine.model.train()
    graph_engine.model.set_training_step(graph_engine.global_step)
    runner._ensure_capture(first_batch)
    base_rng = _rng_state(graph_engine, device)
    eager_rng = {name: value.clone() for name, value in base_rng.items()}
    graph_rng = {name: value.clone() for name, value in base_rng.items()}

    cases = (
        (first_batch, True, "eager_diagnostic"),
        (first_batch, False, "post_diagnostic_replay"),
        (second_batch, False, "copied_input"),
        (first_batch, False, "restored_input"),
    )
    for batch, diagnostics, label in cases:
        _set_rng_state(eager_engine, device, eager_rng)
        expected = eager_engine.train_step(
            batch, collect_diagnostics=diagnostics
        )
        eager_rng = _rng_state(eager_engine, device)

        _set_rng_state(graph_engine, device, graph_rng)
        actual = runner.train_step(batch, collect_diagnostics=diagnostics)
        graph_rng = _rng_state(graph_engine, device)

        _assert_step_result_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            path=f"runner.{label}",
            actual_metric_provenance=graph_metric_provenance,
            expected_metric_provenance=eager_metric_provenance,
        )
        _assert_gradients_close(
            _named_gradients(graph_engine.model),
            _named_gradients(eager_engine.model),
            path=f"runner.{label}.clipped_gradient",
            rtol=rtol,
            atol=atol,
        )
        _assert_parameters_close(
            graph_engine.model, eager_engine.model, rtol=rtol, atol=atol
        )
        _assert_optimizer_close(
            graph_engine, eager_engine, rtol=rtol, atol=atol
        )
        _assert_tensor_mapping_close(
            graph_rng,
            eager_rng,
            path=f"runner.{label}.rng",
            rtol=0.0,
            atol=0.0,
        )
        print("runner_equivalent", label, "loss", float(expected.loss))

    assert runner.capture_count == 1


def _run_topology_recapture_gate(*, rtol: float, atol: float) -> None:
    """Switch captures while both trajectories still own identical parameters."""

    device = torch.device("cuda")
    eager_engine, batch = _engine(device)
    graph_engine, _unused_batch = _engine(device)
    eager_metric_provenance = _install_metric_provenance_recorder(eager_engine)
    graph_metric_provenance = _install_metric_provenance_recorder(graph_engine)
    _configure_compiled_graph_engine(graph_engine)
    runner = _comparison_runner(graph_engine)
    graph_engine.model.train()
    graph_engine.model.set_training_step(0)
    runner._ensure_capture(batch)
    assert runner.capture_count == 1

    for engine in (eager_engine, graph_engine):
        engine.global_step = 700
        engine.schedule.step_index = 700
        engine.schedule._apply_current_ratio()
    base_rng = _rng_state(graph_engine, device)
    eager_rng = {name: value.clone() for name, value in base_rng.items()}
    graph_rng = {name: value.clone() for name, value in base_rng.items()}

    _set_rng_state(eager_engine, device, eager_rng)
    expected = eager_engine.train_step(batch, collect_diagnostics=False)
    eager_rng = _rng_state(eager_engine, device)
    _set_rng_state(graph_engine, device, graph_rng)
    actual = runner.train_step(batch, collect_diagnostics=False)
    graph_rng = _rng_state(graph_engine, device)
    _assert_step_result_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        path="runner.active",
        actual_metric_provenance=graph_metric_provenance,
        expected_metric_provenance=eager_metric_provenance,
    )
    _assert_gradients_close(
        _named_gradients(graph_engine.model),
        _named_gradients(eager_engine.model),
        path="runner.active.clipped_gradient",
        rtol=rtol,
        atol=atol,
    )
    _assert_parameters_close(
        graph_engine.model, eager_engine.model, rtol=rtol, atol=atol
    )
    _assert_optimizer_close(graph_engine, eager_engine, rtol=rtol, atol=atol)
    _assert_tensor_mapping_close(
        graph_rng,
        eager_rng,
        path="runner.active.rng",
        rtol=0.0,
        atol=0.0,
    )
    assert runner.capture_count == 2
    print("runner_equivalent active_recapture loss", float(expected.loss))


def _owned_generators(engine: Any) -> dict[str, torch.Generator]:
    generators: dict[str, torch.Generator] = {}
    if engine.train_flow_generator is not None:
        generators["train_flow"] = engine.train_flow_generator
    if engine.train_condition_generator is not None:
        generators["train_condition"] = engine.train_condition_generator
    return generators


def _run_checkpoint_resume_gate(*, rtol: float, atol: float) -> None:
    """Round-trip the real checkpoint API and continue through a fresh capture."""

    from clearvla.mainline.checkpoint import (  # noqa: PLC0415
        ArtifactIdentity,
        DatasetIdentity,
        build_checkpoint_identity,
    )
    from clearvla.mainline.runtime.checkpoints import (  # noqa: PLC0415
        load_checkpoint_exact,
        save_checkpoint,
    )

    device = torch.device("cuda")
    source_engine, first_batch = _engine(device)
    source_metric_provenance = _install_metric_provenance_recorder(source_engine)
    _configure_compiled_graph_engine(source_engine)
    source_runner = _comparison_runner(source_engine)
    source_engine.model.train()
    source_engine.model.set_training_step(source_engine.global_step)

    initial_rng = _rng_state(source_engine, device)
    _set_rng_state(source_engine, device, initial_rng)
    source_runner.train_step(first_batch, collect_diagnostics=False)
    checkpoint_rng = _rng_state(source_engine, device)

    with tempfile.TemporaryDirectory(prefix="clearvla_checkpoint_equiv_") as raw_dir:
        directory = Path(raw_dir)
        language_path = directory / "goal.pt"
        language_path.write_bytes(b"training-acceleration-equivalence")
        zero_digest = hashlib.sha256(b"").hexdigest()
        identity = build_checkpoint_identity(
            source_engine.config,
            repo_root=_SOURCE_ROOT,
            dataset=DatasetIdentity(
                raw_root="/equivalence/fixture",
                hdf5_glob="*.hdf5",
                inventory_sha256=zero_digest,
                state_normalizer_sha256=zero_digest,
                action_normalizer_sha256=zero_digest,
                decoded_cache_identity=zero_digest,
                dino_cache_identity=zero_digest,
            ),
            language=ArtifactIdentity.from_file("t5_goal", language_path),
            commit="e" * 40,
        )
        checkpoint_path = directory / "resume.pt"
        save_checkpoint(
            checkpoint_path,
            model=source_engine.model,
            optimizer=source_engine.optimizer,
            schedule=source_engine.schedule,
            config=source_engine.config,
            identity=identity,
            epoch=3,
            global_step=source_engine.global_step,
            best_metric=0.25,
            data_state={"fixture_cursor": 1},
            generators=_owned_generators(source_engine),
        )

        resumed_engine, _unused_batch = _engine(device)
        resumed_metric_provenance = _install_metric_provenance_recorder(
            resumed_engine
        )
        _configure_compiled_graph_engine(resumed_engine)
        restored = load_checkpoint_exact(
            checkpoint_path,
            model=resumed_engine.model,
            optimizer=resumed_engine.optimizer,
            schedule=resumed_engine.schedule,
            config=resumed_engine.config,
            identity=identity,
            generators=_owned_generators(resumed_engine),
        )
        resumed_engine.global_step = restored.global_step
        assert restored.epoch == 3
        assert restored.global_step == source_engine.global_step
        assert restored.best_metric == 0.25
        _assert_tensor_mapping_close(
            resumed_engine.model.state_dict(),
            source_engine.model.state_dict(),
            path="checkpoint.model",
            rtol=0.0,
            atol=0.0,
        )
        _assert_optimizer_close(
            resumed_engine,
            source_engine,
            rtol=0.0,
            atol=0.0,
        )
        assert resumed_engine.schedule.state_dict() == source_engine.schedule.state_dict()
        restored_rng = _rng_state(resumed_engine, device)
        _assert_tensor_mapping_close(
            restored_rng,
            checkpoint_rng,
            path="checkpoint.rng",
            rtol=0.0,
            atol=0.0,
        )

        source_batch = _fresh_long_batch(
            source_engine.config,
            device,
            batch_size=2,
            index=91,
        )
        resumed_batch = _fresh_long_batch(
            resumed_engine.config,
            device,
            batch_size=2,
            index=91,
        )
        _assert_independent_equal_batches(resumed_batch, source_batch)
        resumed_runner = _comparison_runner(resumed_engine)

        _set_rng_state(source_engine, device, checkpoint_rng)
        expected = source_runner.train_step(
            source_batch,
            collect_diagnostics=False,
        )
        expected_rng = _rng_state(source_engine, device)
        _set_rng_state(resumed_engine, device, checkpoint_rng)
        actual = resumed_runner.train_step(
            resumed_batch,
            collect_diagnostics=False,
        )
        actual_rng = _rng_state(resumed_engine, device)

        _assert_step_result_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            path="checkpoint.continuation",
            actual_metric_provenance=resumed_metric_provenance,
            expected_metric_provenance=source_metric_provenance,
        )
        _assert_gradients_close(
            _named_gradients(resumed_engine.model),
            _named_gradients(source_engine.model),
            path="checkpoint.continuation.clipped_gradient",
            rtol=rtol,
            atol=atol,
        )
        _assert_parameters_close(
            resumed_engine.model,
            source_engine.model,
            rtol=rtol,
            atol=atol,
        )
        _assert_optimizer_close(
            resumed_engine,
            source_engine,
            rtol=rtol,
            atol=atol,
        )
        _assert_tensor_mapping_close(
            actual_rng,
            expected_rng,
            path="checkpoint.continuation.rng",
            rtol=0.0,
            atol=0.0,
        )
        assert resumed_engine.global_step == source_engine.global_step
        assert resumed_engine.schedule.step_index == source_engine.schedule.step_index
        assert source_runner.capture_count == 1
        assert resumed_runner.capture_count == 1
        print(
            "checkpoint_resume_equivalence_ok",
            json.dumps(
                {
                    "checkpoint_bytes": checkpoint_path.stat().st_size,
                    "restored_step": restored.global_step,
                    "continued_step": resumed_engine.global_step,
                    "rng_exact": True,
                    "fresh_capture_count": resumed_runner.capture_count,
                },
                sort_keys=True,
            ),
        )


def _fresh_long_batch(
    config: Any,
    device: torch.device,
    *,
    batch_size: int,
    index: int,
) -> Any:
    """Create a deterministic new batch without perturbing training RNG."""

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    try:
        torch.manual_seed(150000 + int(index))
        return _move(_batch(config, batch=batch_size), device)
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("p50", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def interpolate(fraction: float) -> float:
        position = fraction * float(len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - float(lower)
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "p50": interpolate(0.50),
        "p90": interpolate(0.90),
        "p95": interpolate(0.95),
        "p99": interpolate(0.99),
        "max": ordered[-1],
    }


def _surface_distribution(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"case_count": 0}
    worst_sample_index, worst = max(
        enumerate(rows),
        key=lambda item: float(item[1]["max_tolerance_ratio"]),
    )
    first_outside_case = next(
        (
            (index, row)
            for index, row in enumerate(rows)
            if int(row["outside_elements"]) > 0
        ),
        None,
    )
    outside_cases = [
        {
            "case_index": int(row.get("case_index", sample_index)),
            "sample_index": sample_index,
            "outside_elements": int(row["outside_elements"]),
            "max_abs": float(row["max_abs"]),
            "max_tolerance_ratio": float(row["max_tolerance_ratio"]),
            "max_tolerance_ratio_path": row["max_tolerance_ratio_path"],
            **{
                name: row[name]
                for name in (
                    "first_outside_path",
                    "first_outside_index",
                    "first_outside_actual",
                    "first_outside_expected",
                    "first_outside_abs",
                )
            },
        }
        for sample_index, row in enumerate(rows)
        if int(row["outside_elements"]) > 0
    ]
    return {
        "case_count": len(rows),
        "outside_case_count": sum(int(row["outside_elements"]) > 0 for row in rows),
        "outside_element_count": sum(int(row["outside_elements"]) for row in rows),
        "max_abs": _quantiles([float(row["max_abs"]) for row in rows]),
        "rms_abs": _quantiles([float(row["rms_abs"]) for row in rows]),
        "relative_l2": _quantiles([float(row["relative_l2"]) for row in rows]),
        "max_tolerance_ratio": _quantiles(
            [float(row["max_tolerance_ratio"]) for row in rows]
        ),
        "worst_path": worst["max_tolerance_ratio_path"],
        "worst_case_index": int(worst.get("case_index", worst_sample_index)),
        "worst_sample_index": worst_sample_index,
        "first_outside_case_index": (
            None
            if first_outside_case is None
            else int(
                first_outside_case[1].get(
                    "case_index",
                    first_outside_case[0],
                )
            )
        ),
        "first_outside_sample_index": (
            None if first_outside_case is None else first_outside_case[0]
        ),
        "outside_cases": outside_cases,
        "first_outside": (
            None
            if first_outside_case is None
            else {
                name: first_outside_case[1][name]
                for name in (
                    "first_outside_path",
                    "first_outside_index",
                    "first_outside_actual",
                    "first_outside_expected",
                    "first_outside_abs",
                )
            }
        ),
    }


def _write_optional_summary(summary: Mapping[str, Any]) -> None:
    raw_path = os.environ.get("CLEARVLA_EQUIV_JSON_OUTPUT")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_independent_sample_statistics(
    *,
    cases: int,
    rtol: float,
    atol: float,
) -> None:
    """Measure many one-update cases without allowing numerical drift to compound."""

    if cases < 2:
        raise ValueError("independent equivalence cases must be at least two")
    device = torch.device("cuda")
    batch_size = int(os.environ.get("CLEARVLA_EQUIV_LONG_BATCH_SIZE", "4"))
    if batch_size <= 0:
        raise ValueError("long equivalence batch size must be positive")
    eager_control = os.environ.get("CLEARVLA_EQUIV_LONG_EAGER_CONTROL") == "1"
    eager_engine, _unused_eager_batch = _engine(device)
    comparison_engine, _unused_comparison_batch = _engine(device)
    freeze_observation = os.environ.get("CLEARVLA_EQUIV_FREEZE_OBSERVATION") == "1"
    frozen_parameter_count = 0
    if freeze_observation:
        frozen_counts = []
        for engine in (eager_engine, comparison_engine):
            observation = getattr(engine.model, "observation", None)
            if observation is None:
                raise RuntimeError(
                    "observation-freeze diagnostic cannot locate model.observation"
                )
            parameters = tuple(observation.parameters())
            if not parameters:
                raise RuntimeError(
                    "observation-freeze diagnostic found no observation parameters"
                )
            for parameter in parameters:
                parameter.requires_grad_(False)
            frozen_counts.append(len(parameters))
        if frozen_counts[0] != frozen_counts[1]:
            raise AssertionError(
                "observation-freeze diagnostic changed parameter coverage between engines"
            )
        frozen_parameter_count = frozen_counts[0]
    eager_metric_provenance = _install_metric_provenance_recorder(eager_engine)
    comparison_metric_provenance = _install_metric_provenance_recorder(
        comparison_engine
    )
    if not eager_control:
        _configure_compiled_graph_engine(comparison_engine)
    _assert_parameters_close(
        comparison_engine.model,
        eager_engine.model,
        rtol=0.0,
        atol=0.0,
    )

    eager_initial_model = _clone_tensor_mapping(eager_engine.model.state_dict())
    comparison_initial_model = _clone_tensor_mapping(
        comparison_engine.model.state_dict()
    )
    eager_initial_optimizer = copy.deepcopy(eager_engine.optimizer.state_dict())
    comparison_initial_optimizer = copy.deepcopy(
        comparison_engine.optimizer.state_dict()
    )
    eager_initial_schedule = copy.deepcopy(eager_engine.schedule.state_dict())
    comparison_initial_schedule = copy.deepcopy(
        comparison_engine.schedule.state_dict()
    )
    runner = None if eager_control else _comparison_runner(comparison_engine)
    half = cases // 2
    surfaces: dict[str, list[Mapping[str, Any]]] = {
        "result": [],
        "detached_observability": [],
        "clipped_gradient": [],
        "rng": [],
        "parameters": [],
        "buffers": [],
        "optimizer": [],
    }
    state_stride = max(1, cases // 16)
    batch_tensor_count = 0
    batch_element_count = 0

    def reset_engine(
        engine: Any,
        *,
        model_state: Mapping[str, torch.Tensor],
        optimizer_state: Mapping[str, Any],
        schedule_state: Mapping[str, Any],
        start_step: int,
    ) -> None:
        engine.model.load_state_dict(model_state, strict=True)
        engine.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        engine.schedule.load_state_dict(copy.deepcopy(schedule_state))
        engine.global_step = start_step
        engine.schedule.step_index = start_step
        engine.schedule._apply_current_ratio()
        engine.model.train()
        engine.model.set_training_step(start_step)

    print(
        "independent_sample_config",
        json.dumps(
            {
                "comparison": "eager" if eager_control else "cuda_graph",
                "cases": cases,
                "batch_size": batch_size,
                "sample_slots": cases * batch_size,
                "identity_cases": half,
                "active_cases": cases - half,
                "rtol": rtol,
                "atol": atol,
                "state_check_stride": state_stride,
                "freeze_observation": freeze_observation,
                "frozen_parameter_count": frozen_parameter_count,
            },
            sort_keys=True,
        ),
    )

    for index in range(cases):
        start_step = 0 if index < half else 700
        reset_engine(
            eager_engine,
            model_state=eager_initial_model,
            optimizer_state=eager_initial_optimizer,
            schedule_state=eager_initial_schedule,
            start_step=start_step,
        )
        reset_engine(
            comparison_engine,
            model_state=comparison_initial_model,
            optimizer_state=comparison_initial_optimizer,
            schedule_state=comparison_initial_schedule,
            start_step=start_step,
        )
        eager_batch = _fresh_long_batch(
            eager_engine.config,
            device,
            batch_size=batch_size,
            index=10_000 + index,
        )
        comparison_batch = _fresh_long_batch(
            comparison_engine.config,
            device,
            batch_size=batch_size,
            index=10_000 + index,
        )
        batch_tensor_count, batch_element_count = _assert_independent_equal_batches(
            comparison_batch,
            eager_batch,
        )

        case_seed = 510_000 + index
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)
        if eager_engine.train_flow_generator is not None:
            eager_engine.train_flow_generator.manual_seed(case_seed + 1)
        if eager_engine.train_condition_generator is not None:
            eager_engine.train_condition_generator.manual_seed(case_seed + 2)
        case_rng = _rng_state(eager_engine, device)

        _set_rng_state(eager_engine, device, case_rng)
        expected = eager_engine.train_step(eager_batch, collect_diagnostics=False)
        expected_rng = _rng_state(eager_engine, device)
        _set_rng_state(comparison_engine, device, case_rng)
        if runner is None:
            actual = comparison_engine.train_step(
                comparison_batch,
                collect_diagnostics=False,
            )
        else:
            actual = runner.train_step(
                comparison_batch,
                collect_diagnostics=False,
            )
        actual_rng = _rng_state(comparison_engine, device)

        result_statistics = dict(
            _result_pair_statistics(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                metric_provenance=comparison_metric_provenance,
            ),
            case_index=index,
        )
        if comparison_metric_provenance != eager_metric_provenance:
            raise AssertionError(
                f"independent case {index} changed metric gradient provenance"
            )
        detached_statistics = dict(
            _detached_result_metric_statistics(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                metric_provenance=comparison_metric_provenance,
            ),
            case_index=index,
        )
        gradient_statistics = dict(
            _gradient_pair_statistics(
                comparison_engine.model,
                eager_engine.model,
                rtol=rtol,
                atol=atol,
            ),
            case_index=index,
        )
        rng_statistics = dict(
            _tensor_pair_statistics(
                [
                    (name, actual_rng[name], expected_rng[name])
                    for name in expected_rng
                ],
                rtol=0.0,
                atol=0.0,
            ),
            case_index=index,
        )
        surfaces["result"].append(result_statistics)
        surfaces["detached_observability"].append(detached_statistics)
        surfaces["clipped_gradient"].append(gradient_statistics)
        surfaces["rng"].append(rng_statistics)
        if int(rng_statistics["outside_elements"]) != 0:
            raise AssertionError(f"independent case {index} changed RNG continuation")
        assert comparison_engine.global_step == eager_engine.global_step
        assert comparison_engine.schedule.step_index == eager_engine.schedule.step_index

        if index % state_stride == 0 or index + 1 == cases:
            surfaces["parameters"].append(
                dict(
                    _parameter_pair_statistics(
                        comparison_engine.model,
                        eager_engine.model,
                        rtol=rtol,
                        atol=atol,
                    ),
                    case_index=index,
                )
            )
            surfaces["buffers"].append(
                dict(
                    _buffer_pair_statistics(
                        comparison_engine.model,
                        eager_engine.model,
                        rtol=rtol,
                        atol=atol,
                    ),
                    case_index=index,
                )
            )
            surfaces["optimizer"].append(
                dict(
                    _optimizer_pair_statistics(
                        comparison_engine,
                        eager_engine,
                        rtol=rtol,
                        atol=atol,
                    ),
                    case_index=index,
                )
            )
        if index == 0 or (index + 1) % 16 == 0 or index + 1 == cases:
            print(
                "independent_sample_progress",
                json.dumps(
                    {
                        "case": index,
                        "completed_cases": index + 1,
                        "sample_slots": (index + 1) * batch_size,
                        "phase": "identity" if start_step == 0 else "active",
                        "gradient_max_abs": float(gradient_statistics["max_abs"]),
                        "result_max_abs": float(result_statistics["max_abs"]),
                        "detached_observability_max_abs": float(
                            detached_statistics["max_abs"]
                        ),
                        "rng_exact": int(rng_statistics["outside_elements"]) == 0,
                        "capture_count": 0 if runner is None else runner.capture_count,
                    },
                    sort_keys=True,
                ),
            )

    summary = {
        "comparison": "eager" if eager_control else "cuda_graph",
        "cases": cases,
        "batch_size": batch_size,
        "sample_slots": cases * batch_size,
        "identity_cases": half,
        "active_cases": cases - half,
        "batch_tensor_count": batch_tensor_count,
        "batch_element_count": batch_element_count,
        "capture_count": 0 if runner is None else runner.capture_count,
        "freeze_observation": freeze_observation,
        "frozen_parameter_count": frozen_parameter_count,
        "selective_compile_plan": (
            None if runner is None else runner.compile_plan_summary
        ),
        "rtol": rtol,
        "atol": atol,
        "surfaces": {
            name: _surface_distribution(rows) for name, rows in surfaces.items()
        },
    }
    _write_optional_summary(summary)
    print("independent_sample_summary", json.dumps(summary, sort_keys=True))


def _run_long_stress_gate(*, steps: int, rtol: float, atol: float) -> None:
    """Compare many distinct batches, including the step-200 topology switch."""

    if steps <= 0:
        raise ValueError("long equivalence steps must be positive")
    device = torch.device("cuda")
    batch_size = int(os.environ.get("CLEARVLA_EQUIV_LONG_BATCH_SIZE", "4"))
    if batch_size <= 0:
        raise ValueError("long equivalence batch size must be positive")
    eager_control = os.environ.get("CLEARVLA_EQUIV_LONG_EAGER_CONTROL") == "1"
    collect_failures = os.environ.get("CLEARVLA_EQUIV_LONG_COLLECT") == "1"
    eager_engine, _unused_eager_batch = _engine(device)
    graph_engine, _unused_graph_batch = _engine(device)
    if not eager_control:
        _configure_compiled_graph_engine(graph_engine)
    _assert_parameters_close(graph_engine.model, eager_engine.model, rtol=0.0, atol=0.0)
    first_eager_batch = _fresh_long_batch(
        eager_engine.config,
        device,
        batch_size=batch_size,
        index=0,
    )
    first_graph_batch = _fresh_long_batch(
        graph_engine.config,
        device,
        batch_size=batch_size,
        index=0,
    )
    runner: CudaGraphTrainingStepRunner | None = None
    if eager_control:
        graph_engine.model.train()
    else:
        runner = _comparison_runner(graph_engine)
        graph_engine.model.train()
        graph_engine.model.set_training_step(graph_engine.global_step)
        runner._ensure_capture(first_graph_batch)
    base_rng = _rng_state(graph_engine, device)
    eager_rng = {name: value.clone() for name, value in base_rng.items()}
    graph_rng = {name: value.clone() for name, value in base_rng.items()}
    previous_capture_count = runner.capture_count if runner is not None else 0
    mismatch_steps = 0
    first_mismatches: list[str] = []

    for index in range(steps):
        eager_batch, graph_batch = (
            (first_eager_batch, first_graph_batch)
            if index == 0
            else (
                _fresh_long_batch(
                    eager_engine.config,
                    device,
                    batch_size=batch_size,
                    index=index,
                ),
                _fresh_long_batch(
                    graph_engine.config,
                    device,
                    batch_size=batch_size,
                    index=index,
                ),
            )
        )
        _set_rng_state(eager_engine, device, eager_rng)
        expected = eager_engine.train_step(eager_batch, collect_diagnostics=False)
        eager_rng = _rng_state(eager_engine, device)

        _set_rng_state(graph_engine, device, graph_rng)
        if runner is None:
            actual = graph_engine.train_step(graph_batch, collect_diagnostics=False)
        else:
            actual = runner.train_step(graph_batch, collect_diagnostics=False)
        graph_rng = _rng_state(graph_engine, device)

        try:
            _assert_step_result_close(actual, expected, rtol=rtol, atol=atol)
            _assert_gradients_close(
                _named_gradients(graph_engine.model),
                _named_gradients(eager_engine.model),
                path=f"long.step_{index}.clipped_gradient",
                rtol=rtol,
                atol=atol,
            )
            _assert_parameters_close(
                graph_engine.model,
                eager_engine.model,
                rtol=rtol,
                atol=atol,
            )
            _assert_optimizer_close(
                graph_engine,
                eager_engine,
                rtol=rtol,
                atol=atol,
            )
            assert graph_engine.global_step == eager_engine.global_step
            assert graph_engine.schedule.step_index == eager_engine.schedule.step_index
            _assert_tensor_mapping_close(
                graph_rng,
                eager_rng,
                path=f"long.step_{index}.rng",
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            if not collect_failures:
                raise
            mismatch_steps += 1
            if len(first_mismatches) < 8:
                summary = str(error).splitlines()
                first_mismatches.append(summary[0] if summary else repr(error))
                print("long_mismatch", index, first_mismatches[-1])
        capture_count = runner.capture_count if runner is not None else 0
        if capture_count != previous_capture_count:
            print(
                "long_recapture",
                "step",
                index,
                "capture_count",
                capture_count,
            )
            previous_capture_count = capture_count
        if index == 0 or (index + 1) % 32 == 0 or index + 1 == steps:
            print(
                "long_equivalent_step",
                index,
                "loss",
                float(expected.loss),
                "gradient_norm",
                expected.gradient_norm_scalar,
            )
    print(
        "long_stress_completed" if collect_failures else "long_equivalence_ok",
        "steps",
        steps,
        "batch_size",
        batch_size,
        "sample_slots",
        steps * batch_size,
        "capture_count",
        runner.capture_count if runner is not None else 0,
        "mismatch_steps",
        mismatch_steps,
    )


def _run_long_drift_statistics(*, steps: int, rtol: float, atol: float) -> None:
    """Measure complete long-run drift against an independent eager control.

    Unlike the strict tripwire, this path checks RNG on every update and does
    not stop state inspection after the first failed tensor.  It executes the
    ordinary production ``train_step`` implementations, so the measured
    optimizer/checkpoint state includes the same metric, clipping and schedule
    lifecycle as the throughput path.
    """

    if steps <= 0:
        raise ValueError("long drift-statistics steps must be positive")
    device = torch.device("cuda")
    batch_size = int(os.environ.get("CLEARVLA_EQUIV_LONG_BATCH_SIZE", "4"))
    if batch_size <= 0:
        raise ValueError("long drift-statistics batch size must be positive")
    eager_control = os.environ.get("CLEARVLA_EQUIV_LONG_EAGER_CONTROL") == "1"
    eager_engine, _unused_eager_batch = _engine(device)
    comparison_engine, _unused_comparison_batch = _engine(device)
    eager_metric_provenance = _install_metric_provenance_recorder(eager_engine)
    comparison_metric_provenance = _install_metric_provenance_recorder(
        comparison_engine
    )
    if not eager_control:
        _configure_compiled_graph_engine(comparison_engine)
    _assert_parameters_close(
        comparison_engine.model,
        eager_engine.model,
        rtol=0.0,
        atol=0.0,
    )

    first_eager_batch = _fresh_long_batch(
        eager_engine.config,
        device,
        batch_size=batch_size,
        index=0,
    )
    first_comparison_batch = _fresh_long_batch(
        comparison_engine.config,
        device,
        batch_size=batch_size,
        index=0,
    )
    batch_tensor_count, batch_element_count = _assert_independent_equal_batches(
        first_comparison_batch,
        first_eager_batch,
    )

    runner: CudaGraphTrainingStepRunner | None = None
    if eager_control:
        comparison_engine.model.train()
    else:
        runner = _comparison_runner(comparison_engine)
        comparison_engine.model.train()
        comparison_engine.model.set_training_step(comparison_engine.global_step)
        runner._ensure_capture(first_comparison_batch)
    base_rng = _rng_state(comparison_engine, device)
    eager_rng = {name: value.clone() for name, value in base_rng.items()}
    comparison_rng = {name: value.clone() for name, value in base_rng.items()}

    default_report_steps = {
        0,
        1,
        2,
        3,
        4,
        7,
        15,
        31,
        63,
        127,
        191,
        198,
        199,
        200,
        201,
        223,
        steps - 1,
    }
    configured_report_steps = os.environ.get("CLEARVLA_EQUIV_LONG_REPORT_STEPS")
    if configured_report_steps:
        report_steps = {
            int(value.strip())
            for value in configured_report_steps.split(",")
            if value.strip()
        }
    else:
        report_steps = default_report_steps
    report_steps = {index for index in report_steps if 0 <= index < steps}

    first_outside_step: dict[str, int | None] = {
        "result": None,
        "detached_observability": None,
        "clipped_gradient": None,
        "rng": None,
    }
    first_outside: dict[str, dict[str, Any] | None] = {
        name: None for name in first_outside_step
    }
    outside_step_count = {name: 0 for name in first_outside_step}
    worst: dict[str, dict[str, Any]] = {}
    loss_abs_rows: list[float] = []
    gradient_norm_abs_rows: list[float] = []
    previous_capture_count = runner.capture_count if runner is not None else 0

    def update_trajectory_summary(
        category: str,
        statistics: Mapping[str, Any],
        *,
        step: int,
    ) -> None:
        if int(statistics["outside_elements"]) > 0:
            outside_step_count[category] += 1
            if first_outside_step[category] is None:
                first_outside_step[category] = step
                first_outside[category] = {
                    "step": step,
                    **{
                        name: statistics[name]
                        for name in (
                            "first_outside_path",
                            "first_outside_index",
                            "first_outside_actual",
                            "first_outside_expected",
                            "first_outside_abs",
                        )
                    },
                }
        previous = worst.get(category)
        if previous is None or float(statistics["max_tolerance_ratio"]) > float(
            previous["max_tolerance_ratio"]
        ):
            worst[category] = {
                "step": step,
                "max_tolerance_ratio": float(statistics["max_tolerance_ratio"]),
                "max_tolerance_ratio_path": statistics[
                    "max_tolerance_ratio_path"
                ],
                "max_abs_at_that_step": float(statistics["max_abs"]),
                "max_abs_path_at_that_step": statistics["max_abs_path"],
                "rms_abs_at_that_step": float(statistics["rms_abs"]),
                "relative_l2_at_that_step": float(statistics["relative_l2"]),
            }

    print(
        "long_drift_config",
        json.dumps(
            {
                "comparison": "eager" if eager_control else "cuda_graph",
                "steps": steps,
                "batch_size": batch_size,
                "sample_slots": steps * batch_size,
                "rtol": rtol,
                "atol": atol,
                "batch_tensor_count": batch_tensor_count,
                "batch_element_count": batch_element_count,
                "report_steps": sorted(report_steps),
            },
            sort_keys=True,
        ),
    )

    for index in range(steps):
        eager_batch, comparison_batch = (
            (first_eager_batch, first_comparison_batch)
            if index == 0
            else (
                _fresh_long_batch(
                    eager_engine.config,
                    device,
                    batch_size=batch_size,
                    index=index,
                ),
                _fresh_long_batch(
                    comparison_engine.config,
                    device,
                    batch_size=batch_size,
                    index=index,
                ),
            )
        )
        _assert_independent_equal_batches(comparison_batch, eager_batch)

        _set_rng_state(eager_engine, device, eager_rng)
        expected = eager_engine.train_step(
            eager_batch,
            collect_diagnostics=False,
        )
        eager_rng = _rng_state(eager_engine, device)

        _set_rng_state(comparison_engine, device, comparison_rng)
        if runner is None:
            actual = comparison_engine.train_step(
                comparison_batch,
                collect_diagnostics=False,
            )
        else:
            actual = runner.train_step(
                comparison_batch,
                collect_diagnostics=False,
            )
        comparison_rng = _rng_state(comparison_engine, device)

        result_statistics = _result_pair_statistics(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            metric_provenance=comparison_metric_provenance,
        )
        if comparison_metric_provenance != eager_metric_provenance:
            raise AssertionError(
                f"long step {index} changed metric gradient provenance"
            )
        detached_statistics = _detached_result_metric_statistics(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            metric_provenance=comparison_metric_provenance,
        )
        gradient_statistics = _gradient_pair_statistics(
            comparison_engine.model,
            eager_engine.model,
            rtol=rtol,
            atol=atol,
        )
        rng_statistics = _tensor_pair_statistics(
            [
                (name, comparison_rng[name], eager_rng[name])
                for name in eager_rng
            ],
            rtol=0.0,
            atol=0.0,
        )
        update_trajectory_summary("result", result_statistics, step=index)
        update_trajectory_summary(
            "detached_observability",
            detached_statistics,
            step=index,
        )
        update_trajectory_summary(
            "clipped_gradient",
            gradient_statistics,
            step=index,
        )
        update_trajectory_summary("rng", rng_statistics, step=index)
        loss_abs_rows.append(abs(float(actual.loss) - float(expected.loss)))
        gradient_norm_abs_rows.append(
            abs(
                float(actual.gradient_norm_scalar)
                - float(expected.gradient_norm_scalar)
            )
        )

        capture_count = runner.capture_count if runner is not None else 0
        if capture_count != previous_capture_count:
            print(
                "long_drift_recapture",
                json.dumps(
                    {
                        "step": index,
                        "capture_count": capture_count,
                    },
                    sort_keys=True,
                ),
            )
            previous_capture_count = capture_count

        if index in report_steps:
            parameter_statistics = _parameter_pair_statistics(
                comparison_engine.model,
                eager_engine.model,
                rtol=rtol,
                atol=atol,
            )
            buffer_statistics = _buffer_pair_statistics(
                comparison_engine.model,
                eager_engine.model,
                rtol=rtol,
                atol=atol,
            )
            optimizer_statistics = _optimizer_pair_statistics(
                comparison_engine,
                eager_engine,
                rtol=rtol,
                atol=atol,
            )
            checkpoint = {
                "step": index,
                "updates": index + 1,
                "sample_slots": (index + 1) * batch_size,
                "capture_count": capture_count,
                "result": result_statistics,
                "detached_observability": detached_statistics,
                "clipped_gradient": gradient_statistics,
                "parameters": parameter_statistics,
                "buffers": buffer_statistics,
                "optimizer": optimizer_statistics,
                "rng": rng_statistics,
                "global_step_equal": (
                    comparison_engine.global_step == eager_engine.global_step
                ),
                "schedule_step_equal": (
                    comparison_engine.schedule.step_index
                    == eager_engine.schedule.step_index
                ),
            }
            print(
                "long_drift_checkpoint",
                json.dumps(checkpoint, sort_keys=True),
            )

    summary = {
        "comparison": "eager" if eager_control else "cuda_graph",
        "steps": steps,
        "batch_size": batch_size,
        "sample_slots": steps * batch_size,
        "capture_count": runner.capture_count if runner is not None else 0,
        "capture_setup_seconds": (
            runner.capture_setup_seconds if runner is not None else 0.0
        ),
        "first_outside_step": first_outside_step,
        "first_outside": first_outside,
        "outside_step_count": outside_step_count,
        "selective_compile_plan": (
            None if runner is None else runner.compile_plan_summary
        ),
        "worst": worst,
        "loss_abs_max": max(loss_abs_rows, default=0.0),
        "loss_abs_rms": (
            sum(value * value for value in loss_abs_rows)
            / float(max(len(loss_abs_rows), 1))
        )
        ** 0.5,
        "gradient_norm_abs_max": max(gradient_norm_abs_rows, default=0.0),
        "gradient_norm_abs_rms": (
            sum(value * value for value in gradient_norm_abs_rows)
            / float(max(len(gradient_norm_abs_rows), 1))
        )
        ** 0.5,
        "global_step_equal": (
            comparison_engine.global_step == eager_engine.global_step
        ),
        "schedule_step_equal": (
            comparison_engine.schedule.step_index
            == eager_engine.schedule.step_index
        ),
    }
    print("long_drift_summary", json.dumps(summary, sort_keys=True))
    _write_optional_summary(summary)


def main() -> None:
    # Two independently allocated eager engines on this CUDA stack already
    # differ by up to 2.92435e-7 after one matched update despite identical
    # loss and RNG continuation.  Use a 1e-6 absolute envelope for the GPU
    # gate; the tighter 2e-7 bound remains the CPU rewrite gate.
    gpu_rtol = 2e-5
    gpu_atol = 1e-6
    export_backend_trace = os.environ.get("CLEARVLA_EQUIV_EXPORT_BACKEND_TRACE")
    compare_backend_traces = os.environ.get(
        "CLEARVLA_EQUIV_COMPARE_BACKEND_TRACES"
    )
    if export_backend_trace and compare_backend_traces:
        raise ValueError(
            "backend trace export and comparison modes are mutually exclusive"
        )
    if export_backend_trace:
        _export_backend_trace(
            Path(export_backend_trace),
            steps=int(os.environ.get("CLEARVLA_EQUIV_BACKEND_TRACE_STEPS", "4")),
        )
        return
    if compare_backend_traces:
        paths = tuple(compare_backend_traces.split(os.pathsep))
        if len(paths) != 2 or any(not value for value in paths):
            raise ValueError(
                "backend trace comparison requires exactly two paths"
            )
        _compare_backend_traces(
            Path(paths[0]),
            Path(paths[1]),
            rtol=gpu_rtol,
            atol=gpu_atol,
        )
        return
    # Trace comparison above is deliberately CPU-only (both artifacts are
    # loaded with ``map_location="cpu"``).  Export and live execution still
    # require CUDA, but offline acceptance should remain usable on hosts that
    # do not expose a GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CLEARVLA_EQUIV_CHECKPOINT_ONLY") == "1":
        _run_checkpoint_resume_gate(rtol=gpu_rtol, atol=gpu_atol)
        return
    if os.environ.get("CLEARVLA_EQUIV_RUNNER_ONLY") == "1":
        _run_production_runner_gate(rtol=gpu_rtol, atol=gpu_atol)
        _run_topology_recapture_gate(rtol=gpu_rtol, atol=gpu_atol)
        _run_checkpoint_resume_gate(rtol=gpu_rtol, atol=gpu_atol)
        print("cuda_graph_runner_equivalence_ok")
        return
    if os.environ.get("CLEARVLA_EQUIV_INDEPENDENT_STATS") == "1":
        independent_cases = int(
            os.environ.get("CLEARVLA_EQUIV_INDEPENDENT_CASES", "256")
        )
        _run_independent_sample_statistics(
            cases=independent_cases,
            rtol=gpu_rtol,
            atol=gpu_atol,
        )
        return
    long_steps = int(os.environ.get("CLEARVLA_EQUIV_LONG_STEPS", "0"))
    if os.environ.get("CLEARVLA_EQUIV_LONG_STATS") == "1":
        long_rtol = float(os.environ.get("CLEARVLA_EQUIV_LONG_RTOL", gpu_rtol))
        long_atol = float(os.environ.get("CLEARVLA_EQUIV_LONG_ATOL", gpu_atol))
        _run_long_drift_statistics(
            steps=long_steps or 256,
            rtol=long_rtol,
            atol=long_atol,
        )
        return
    if os.environ.get("CLEARVLA_EQUIV_LONG_ONLY") == "1":
        long_rtol = float(os.environ.get("CLEARVLA_EQUIV_LONG_RTOL", gpu_rtol))
        long_atol = float(os.environ.get("CLEARVLA_EQUIV_LONG_ATOL", gpu_atol))
        _run_long_stress_gate(
            steps=long_steps or 256,
            rtol=long_rtol,
            atol=long_atol,
        )
        return
    # Identity and non-identity execution use different Python topology and
    # therefore intentionally own separate captures.
    # CUDA reduction order may differ at the final few FP32 mantissa bits when
    # the same kernels run on a private graph stream.  Use the branch's
    # pre-existing full-update floating-point gate; do not relax it further.
    _run_gate(start_step=0, steps=2, rtol=gpu_rtol, atol=gpu_atol)
    _run_gate(start_step=700, steps=2, rtol=gpu_rtol, atol=gpu_atol)
    _run_production_runner_gate(rtol=gpu_rtol, atol=gpu_atol)
    _run_topology_recapture_gate(rtol=gpu_rtol, atol=gpu_atol)
    _run_checkpoint_resume_gate(rtol=gpu_rtol, atol=gpu_atol)
    print("cuda_graph_equivalence_ok")


if __name__ == "__main__":
    main()
