"""Strict eager-versus-CUDA-graph training equivalence probe.

This is a research gate for the training-acceleration branch, not a production
runtime.  It captures the complete ordinary-batch forward, loss and backward,
then keeps the existing gradient lifecycle, optimizer and scheduler outside
the graph.  Every formal loss surface, raw/clipped gradient, parameter,
optimizer tensor and RNG continuation is compared after each replay.
"""

from __future__ import annotations

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
    decoder = engine.model.execution_bottom.decoder
    if context_reuse or compile_mode in valid_modes:
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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Two independently allocated eager engines on this CUDA stack already
    # differ by up to 2.92435e-7 after one matched update despite identical
    # loss and RNG continuation.  Use a 1e-6 absolute envelope for the GPU
    # gate; the tighter 2e-7 bound remains the CPU rewrite gate.
    gpu_rtol = 2e-5
    gpu_atol = 1e-6
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
