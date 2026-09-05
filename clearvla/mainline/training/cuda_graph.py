"""Opt-in CUDA Graph runner for ordinary ClearVLA training batches.

The model, losses, gradient lifecycle, optimizer and scheduler remain owned by
``MainlineTrainingEngine``.  This runner records only its ordinary-batch
forward/loss/backward surface, whose thousands of small CUDA launches are
otherwise repeatedly dispatched from Python.  Diagnostic batches retain the
complete eager path.
"""

from __future__ import annotations

import dataclasses
import gc
import inspect
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from ..interfaces import TrainingBatch
from .engine import MainlineTrainingEngine, TrainStepResult, _autocast
from .gradient_audit import FiniteGradientSpikeReport


class StaticTrainingInputMismatch(ValueError):
    """A batch cannot be copied into the currently captured static slots."""


def _clone_static_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _clone_static_tree(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, dict):
        return {name: _clone_static_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_static_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_static_tree(item) for item in value)
    return value


def _static_tree_signature(value: Any) -> Any:
    if isinstance(value, Tensor):
        return (
            "tensor",
            tuple(int(size) for size in value.shape),
            value.dtype,
            value.device,
        )
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (
                    field.name,
                    _static_tree_signature(getattr(value, field.name)),
                )
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return (
            type(value),
            tuple((name, _static_tree_signature(item)) for name, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_static_tree_signature(item) for item in value)
    return type(value), value


def _copy_static_tree(destination: Any, source: Any, *, path: str = "batch") -> None:
    if isinstance(destination, Tensor):
        if not isinstance(source, Tensor):
            raise StaticTrainingInputMismatch(f"{path} stopped being a tensor")
        if (
            tuple(destination.shape) != tuple(source.shape)
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise StaticTrainingInputMismatch(
                f"{path} tensor contract changed from "
                f"{tuple(destination.shape)}/{destination.dtype}/{destination.device} "
                f"to {tuple(source.shape)}/{source.dtype}/{source.device}"
            )
        destination.copy_(source, non_blocking=True)
        return
    if dataclasses.is_dataclass(destination) and not isinstance(destination, type):
        if type(source) is not type(destination):
            raise StaticTrainingInputMismatch(f"{path} dataclass type changed")
        for field in dataclasses.fields(destination):
            _copy_static_tree(
                getattr(destination, field.name),
                getattr(source, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(destination, Mapping):
        if not isinstance(source, Mapping) or tuple(destination) != tuple(source):
            raise StaticTrainingInputMismatch(f"{path} mapping keys changed")
        for name in destination:
            _copy_static_tree(
                destination[name], source[name], path=f"{path}.{name}"
            )
        return
    if isinstance(destination, (list, tuple)):
        if type(source) is not type(destination) or len(source) != len(destination):
            raise StaticTrainingInputMismatch(f"{path} sequence contract changed")
        for index, (destination_item, source_item) in enumerate(
            zip(destination, source, strict=True)
        ):
            _copy_static_tree(
                destination_item,
                source_item,
                path=f"{path}[{index}]",
            )
        return
    if destination != source:
        raise StaticTrainingInputMismatch(
            f"{path} static value changed from {destination!r} to {source!r}"
        )


@contextmanager
def _validated_static_capture_region():
    """Suppress repeated validators only while recording checked static input.

    The incoming batch and a complete eager warm-up pass are validated before
    capture.  Replay executes no Python, so validations are restored as soon as
    recording ends.  This context is intentionally local to the single-threaded
    training capture boundary.
    """

    originals: list[tuple[type, object]] = []

    def no_validate(_self: object, *_args: object, **_kwargs: object) -> None:
        return None

    seen: set[type] = set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("clearvla.mainline") or module is None:
            continue
        for value in tuple(vars(module).values()):
            if (
                inspect.isclass(value)
                and value not in seen
                and "validate" in value.__dict__
            ):
                seen.add(value)
                originals.append((value, value.__dict__["validate"]))
                setattr(value, "validate", no_validate)
    try:
        yield
    finally:
        for value, original in originals:
            setattr(value, "validate", original)


def _rng_snapshot(
    engine: MainlineTrainingEngine,
) -> dict[str, Tensor]:
    state = {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state(engine.device).clone(),
    }
    if engine.train_flow_generator is not None:
        state["flow"] = engine.train_flow_generator.get_state().clone()
    if engine.train_condition_generator is not None:
        state["condition"] = engine.train_condition_generator.get_state().clone()
    return state


def _restore_rng_snapshot(
    engine: MainlineTrainingEngine,
    state: Mapping[str, Tensor],
) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], engine.device)
    if engine.train_flow_generator is not None:
        engine.train_flow_generator.set_state(state["flow"])
    if engine.train_condition_generator is not None:
        engine.train_condition_generator.set_state(state["condition"])


class CudaGraphTrainingStepRunner:
    """Replay ordinary training steps while preserving eager diagnostics."""

    def __init__(self, engine: MainlineTrainingEngine) -> None:
        if engine.device.type != "cuda":
            raise ValueError("CUDA Graph training requires a CUDA engine")
        self.engine = engine
        decoder = engine.model.execution_bottom.decoder
        # Both rewrites have independent full-update equivalence gates.  They
        # replace dynamic candidate grouping with the same known static chart.
        decoder._static_neutral_owner = True
        decoder._training_candidate_prefix_reuse = True
        # Eager diagnostic steps must zero, rather than discard, the gradient
        # buffers whose addresses are owned by the captured backward graph.
        engine._preserve_static_gradient_buffers = True
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: TrainingBatch | None = None
        self._captured_ledger: Any = None
        self._captured_metrics: dict[str, Tensor] | None = None
        self._capture_key: Any = None
        self.capture_count = 0
        self.capture_setup_seconds = 0.0

    def _topology_key(self) -> str:
        decoder = self.engine.model.execution_bottom.decoder
        return (
            "identity"
            if float(decoder._execution_progress_value) <= 0.0
            else "active"
        )

    def _release_capture(self) -> None:
        if self._graph is not None:
            torch.cuda.synchronize(self.engine.device)
            self._graph.reset()
            # A replacement capture owns a new backward graph.  Drop the old
            # AccumulateGrad buffers before collecting its graph executable so
            # they cannot retain the previous private-stream nodes.
            self.engine.optimizer.zero_grad(set_to_none=True)
        self._graph = None
        self._captured_ledger = None
        self._captured_metrics = None
        self._static_batch = None
        self._capture_key = None
        gc.collect()

    def _capture(self, batch: TrainingBatch) -> None:
        engine = self.engine
        capture_rng = _rng_snapshot(engine)
        started = time.perf_counter()
        try:
            # A normal eager pass owns all value checks and allocates stable
            # gradient buffers before the private capture stream is entered.
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
            for generator in (
                engine.train_flow_generator,
                engine.train_condition_generator,
            ):
                if generator is not None and generator.device.type == "cuda":
                    graph.register_generator_state(generator)
            with _validated_static_capture_region():
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
            self._graph = graph
            self._captured_ledger = captured_ledger
            self._captured_metrics = captured_metrics
            self.capture_count += 1
        finally:
            _restore_rng_snapshot(engine, capture_rng)
            torch.cuda.synchronize(engine.device)
            self.capture_setup_seconds += time.perf_counter() - started

    def _ensure_capture(self, batch: TrainingBatch) -> float:
        batch.validate(self.engine.config)
        capture_key = (self._topology_key(), _static_tree_signature(batch))
        if self._graph is not None and self._capture_key == capture_key:
            if self._static_batch is None:
                raise RuntimeError("CUDA Graph runner lost its static batch")
            started = time.perf_counter()
            _copy_static_tree(self._static_batch, batch)
            return time.perf_counter() - started

        self._release_capture()
        static_batch = _clone_static_tree(batch)
        if not isinstance(static_batch, TrainingBatch):
            raise TypeError("static CUDA Graph input lost TrainingBatch type")
        self._static_batch = static_batch
        self._capture_key = capture_key
        self._capture(static_batch)
        return 0.0

    @staticmethod
    def _snapshot_metrics(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if not values:
            return {}
        names = tuple(values)
        packed = torch.stack([values[name].detach().float() for name in names])
        return {name: packed[index] for index, name in enumerate(names)}

    def train_step(
        self,
        batch: TrainingBatch,
        *,
        collect_diagnostics: bool = False,
        gradient_spike_handler: Callable[[FiniteGradientSpikeReport], None]
        | None = None,
        phase_timing: dict[str, float] | None = None,
    ) -> TrainStepResult:
        if collect_diagnostics:
            return self.engine.train_step(
                batch,
                collect_diagnostics=True,
                gradient_spike_handler=gradient_spike_handler,
                phase_timing=phase_timing,
            )

        engine = self.engine
        timing_started = time.perf_counter()

        def timing_mark(name: str, started: float) -> None:
            if phase_timing is not None:
                torch.cuda.synchronize(engine.device)
                phase_timing[name] = time.perf_counter() - started

        engine.model.train()
        engine.model.set_training_step(engine.global_step)
        setup_before = self.capture_setup_seconds
        copy_started = time.perf_counter()
        copy_seconds = self._ensure_capture(batch)
        timing_mark("cuda_graph_input_copy_seconds", copy_started)
        if phase_timing is not None:
            phase_timing["cuda_graph_capture_setup_seconds"] = (
                self.capture_setup_seconds - setup_before
            )
            phase_timing["cuda_graph_input_copy_enqueue_seconds"] = copy_seconds

        graph = self._graph
        ledger = self._captured_ledger
        captured_metrics = self._captured_metrics
        if graph is None or ledger is None or captured_metrics is None:
            raise RuntimeError("CUDA Graph training capture is incomplete")
        replay_started = time.perf_counter()
        graph.replay()
        timing_mark("cuda_graph_forward_backward_seconds", replay_started)

        gradient_started = time.perf_counter()
        gradient_norm, gradient_metrics, gradient_norm_scalar = (
            engine._gradient_lifecycle(
                collect_diagnostics=False,
                gradient_spike_handler=gradient_spike_handler,
            )
        )
        timing_mark("gradient_lifecycle_seconds", gradient_started)
        learning_rate = float(
            engine.config.optimizer.learning_rate
            * engine.schedule.ratio(engine.schedule.step_index)
        )
        optimizer_started = time.perf_counter()
        engine.optimizer.step()
        engine.schedule.step()
        engine.global_step += 1
        timing_mark("optimizer_seconds", optimizer_started)

        captured_metrics.update(gradient_metrics)
        metrics = self._snapshot_metrics(
            engine._tensor_metrics(ledger, captured_metrics)
        )
        if phase_timing is not None:
            phase_timing["total_seconds"] = time.perf_counter() - timing_started
        return TrainStepResult(
            loss=metrics["loss_total"],
            gradient_norm=gradient_norm.detach().float(),
            learning_rate=learning_rate,
            metrics=metrics,
            gradient_norm_scalar=gradient_norm_scalar,
        )


__all__ = [
    "CudaGraphTrainingStepRunner",
    "StaticTrainingInputMismatch",
]
