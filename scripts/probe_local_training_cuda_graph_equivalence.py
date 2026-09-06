"""Strict eager-versus-CUDA-graph training equivalence probe.

This is a research gate for the training-acceleration branch, not a production
runtime.  It captures the complete ordinary-batch forward, loss and backward,
then keeps the existing gradient lifecycle, optimizer and scheduler outside
the graph.  Every formal loss surface, raw/clipped gradient, parameter,
optimizer tensor and RNG continuation is compared after each replay.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from typing import Any

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


def _configure_compiled_graph_engine(engine: Any) -> None:
    """Opt-in gate for the safe submodule-compile production combination."""

    compile_mode = os.environ.get("CLEARVLA_EQUIV_COMPILE_SUBMODULES")
    context_reuse = os.environ.get("CLEARVLA_EQUIV_CONTEXT_REUSE") == "1"
    valid_modes = {"1", "visual_mainline", "visual", "mainline", "mmdit"}
    if compile_mode not in valid_modes and not context_reuse:
        return
    adapter = resolve_training_acceleration_adapter(engine.model)
    adapter.prepare(engine)
    decoder = engine.model.execution_bottom.decoder
    if context_reuse or compile_mode in valid_modes:
        # These are optional mainline decoder optimizations.  They remain in
        # the probe because compile experiments need their explicit switches;
        # the production Graph backend reaches them only through its adapter.
        decoder._reuse_prepared_block_contexts = True
        decoder._reuse_prepared_controller_context = True
        decoder._reuse_terminal_candidate_velocity = True
    if compile_mode not in valid_modes:
        return
    if compile_mode in {"1", "mmdit"}:
        for block in tuple(decoder.blocks):
            block.forward = torch.compile(  # type: ignore[method-assign]
                block.forward, dynamic=False, fullgraph=False, mode="default"
            )
    if compile_mode in {"1", "visual_mainline", "visual"}:
        encoder = engine.model.observation.compiler.encoder
        visual_modules: list[torch.nn.Module] = [encoder.flow]
        if encoder.raw_flow is not None:
            visual_modules.append(encoder.raw_flow)
        for module in visual_modules:
            module.forward = torch.compile(  # type: ignore[method-assign]
                module.forward, dynamic=False, fullgraph=False, mode="default"
            )
    if compile_mode in {"1", "visual_mainline", "mainline"}:
        mainline_modules: list[torch.nn.Module] = [
            *tuple(engine.model.grounding.blocks),
            engine.model.p1.dynamic_policy_block,
            engine.model.world.dynamics.w1,
            engine.model.world.dynamics.w2,
            engine.model.transition.v120_transition,
            *tuple(engine.model.execution_bottom.layer_contract_heads),
        ]
        for module in mainline_modules:
            module.forward = torch.compile(  # type: ignore[method-assign]
                module.forward, dynamic=False, fullgraph=False, mode="default"
            )


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
        raise AssertionError(
            f"{path}: max_abs={float(difference.max()):.9g} "
            f"max_rel={float(relative.max()):.9g} "
            f"actual_rms={float(actual.detach().float().square().mean().sqrt()):.9g} "
            f"expected_rms={float(expected.detach().float().square().mean().sqrt()):.9g}"
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
        }

    grouped: dict[
        str,
        dict[str, list[torch.Tensor] | list[str] | int],
    ] = {}
    element_count = 0
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
            ("outside_elements", torch.count_nonzero(difference > tolerance)),
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

    return {
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
) -> dict[str, Any]:
    assert tuple(actual.metrics) == tuple(expected.metrics)
    pairs = [
        ("loss", actual.loss, expected.loss),
        ("gradient_norm", actual.gradient_norm, expected.gradient_norm),
        *[
            (f"metrics.{name}", actual.metrics[name], expected.metrics[name])
            for name in expected.metrics
        ],
    ]
    result = _tensor_pair_statistics(pairs, rtol=rtol, atol=atol)
    result["learning_rate_equal"] = actual.learning_rate == expected.learning_rate
    result["gradient_norm_scalar_abs"] = abs(
        float(actual.gradient_norm_scalar) - float(expected.gradient_norm_scalar)
    )
    return result


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


def _ledger_snapshot(ledger: Any) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
    return {
        "total": ledger.total.detach().clone(),
        "groups": _clone_tensor_mapping(ledger.groups),
        "contributions": _clone_tensor_mapping(ledger.contributions),
        "terms": _clone_tensor_mapping(ledger.terms),
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
    with _autocast(engine.device, engine.dtype):
        ledger, _metrics = engine._forward(
            batch,
            training=True,
            collect_diagnostics=False,
            generator=engine.train_flow_generator,
            condition_generator=engine.train_condition_generator,
        )
    ledger.total.backward()
    return ledger


def _finish_update(engine: Any) -> tuple[torch.Tensor, float]:
    gradient_norm, _metrics, gradient_norm_scalar = engine._gradient_lifecycle(
        collect_diagnostics=False
    )
    engine.optimizer.step()
    engine.schedule.step()
    engine.global_step += 1
    return gradient_norm.detach().clone(), gradient_norm_scalar


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
    graph, captured_ledger = _capture_forward_backward(graph_engine, graph_batch)

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


def _assert_step_result_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
) -> None:
    _assert_tensor_close(
        actual.loss,
        expected.loss,
        path="result.loss",
        rtol=rtol,
        atol=atol,
    )
    _assert_tensor_close(
        actual.gradient_norm,
        expected.gradient_norm,
        path="result.gradient_norm",
        rtol=rtol,
        atol=atol,
    )
    assert actual.learning_rate == expected.learning_rate
    assert actual.gradient_norm_scalar is not None
    assert expected.gradient_norm_scalar is not None
    assert abs(actual.gradient_norm_scalar - expected.gradient_norm_scalar) <= (
        atol + rtol * abs(expected.gradient_norm_scalar)
    )
    _assert_tensor_mapping_close(
        actual.metrics,
        expected.metrics,
        path="result.metrics",
        rtol=rtol,
        atol=atol,
    )


def _run_production_runner_gate(*, rtol: float, atol: float) -> None:
    """Exercise input copies, eager diagnostics and topology recapture."""

    device = torch.device("cuda")
    eager_engine, first_batch = _engine(device)
    graph_engine, _unused_batch = _engine(device)
    _configure_compiled_graph_engine(graph_engine)
    runner = CudaGraphTrainingStepRunner(graph_engine)
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

        _assert_step_result_close(actual, expected, rtol=rtol, atol=atol)
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
    _configure_compiled_graph_engine(graph_engine)
    runner = CudaGraphTrainingStepRunner(graph_engine)
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
    _assert_step_result_close(actual, expected, rtol=rtol, atol=atol)
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


def _run_long_stress_gate(*, steps: int, rtol: float, atol: float) -> None:
    """Compare many distinct batches, including the step-200 topology switch."""

    if steps <= 0:
        raise ValueError("long equivalence steps must be positive")
    device = torch.device("cuda")
    batch_size = 4
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
        runner = CudaGraphTrainingStepRunner(graph_engine)
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
    batch_size = 4
    eager_control = os.environ.get("CLEARVLA_EQUIV_LONG_EAGER_CONTROL") == "1"
    eager_engine, _unused_eager_batch = _engine(device)
    comparison_engine, _unused_comparison_batch = _engine(device)
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
        runner = CudaGraphTrainingStepRunner(comparison_engine)
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
        "clipped_gradient": None,
        "rng": None,
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
        "outside_step_count": outside_step_count,
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Two independently allocated eager engines on this CUDA stack already
    # differ by up to 2.92435e-7 after one matched update despite identical
    # loss and RNG continuation.  Use a 1e-6 absolute envelope for the GPU
    # gate; the tighter 2e-7 bound remains the CPU rewrite gate.
    gpu_rtol = 2e-5
    gpu_atol = 1e-6
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
    print("cuda_graph_equivalence_ok")


if __name__ == "__main__":
    main()
