"""Opt-in CUDA Graph runner for ordinary ClearVLA training batches.

The model, losses, gradient lifecycle, optimizer and scheduler remain owned by
``MainlineTrainingEngine``.  This runner records its ordinary-batch
forward/loss/backward surface, whose thousands of small CUDA launches are
otherwise repeatedly dispatched from Python.  An independent opt-in graph may
also replay an already-initialized fused AdamW step while the Python scheduler
and checkpoint-visible optimizer metadata remain unchanged.  Diagnostic
batches retain the complete eager path.
"""

from __future__ import annotations

import dataclasses
import gc
import inspect
import math
import sys
import time
from collections.abc import Callable, Hashable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor

from ..interfaces import TrainingBatch
from .acceleration_contract import (
    TrainingAccelerationAdapter,
    resolve_training_acceleration_adapter,
    resolve_training_compile_plan,
)
from .engine import (
    MainlineTrainingEngine,
    NonFiniteGradientError,
    TrainStepResult,
    _autocast,
)
from .gradient_audit import FiniteGradientSpikeReport
from .selective_compile import (
    AppliedTrainingCompilePlan,
    apply_training_compile_plan,
    get_applied_training_compile_plan,
)


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

    def __init__(
        self,
        engine: MainlineTrainingEngine,
        *,
        compile_profile: str | None = None,
        allow_candidate_compile: bool = False,
        capture_gradient_lifecycle: bool = False,
        capture_optimizer_step: bool = False,
    ) -> None:
        if engine.device.type != "cuda":
            raise ValueError("CUDA Graph training requires a CUDA engine")
        self.engine = engine
        # Version-specific capture preparation lives behind this adapter.  The
        # backend therefore remains reusable when a branch replaces the
        # decoder, block stack or even the whole model implementation.
        self._acceleration_adapter: TrainingAccelerationAdapter = (
            resolve_training_acceleration_adapter(engine.model)
        )
        self._acceleration_adapter.prepare(engine)
        existing_compile = get_applied_training_compile_plan(engine)
        self._compile_profile = (
            compile_profile
            if compile_profile is not None
            else (None if existing_compile is None else existing_compile.profile)
        )
        self._allow_candidate_compile = bool(
            allow_candidate_compile
            or (
                existing_compile is not None
                and existing_compile.acceptance == "candidate"
            )
        )
        if capture_gradient_lifecycle and (
            engine.gradient_spike_audit_threshold is not None
        ):
            raise ValueError(
                "captured gradient lifecycle requires the gradient-spike audit "
                "to be disabled"
            )
        self._capture_gradient_lifecycle = bool(capture_gradient_lifecycle)
        if capture_optimizer_step:
            if not isinstance(engine.optimizer, torch.optim.AdamW):
                raise ValueError(
                    "captured optimizer step currently requires torch.optim.AdamW"
                )
            if not engine.optimizer.param_groups or not all(
                bool(group.get("fused", False))
                for group in engine.optimizer.param_groups
            ):
                raise ValueError(
                    "captured optimizer step requires fused AdamW in every group"
                )
            if any(
                bool(group.get("differentiable", False))
                for group in engine.optimizer.param_groups
            ):
                raise ValueError(
                    "captured optimizer step does not support differentiable AdamW"
                )
        self._capture_optimizer_step = bool(capture_optimizer_step)
        self._applied_compile_plan = self._configure_compile_plan()
        self._structure_signature = self._acceleration_adapter.structure_signature(
            engine
        )
        # Eager diagnostic steps must zero, rather than discard, the gradient
        # buffers whose addresses are owned by the captured backward graph.
        engine._preserve_static_gradient_buffers = True
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_batch: TrainingBatch | None = None
        self._captured_ledger: Any = None
        self._captured_metrics: dict[str, Tensor] | None = None
        self._captured_gradient_norm: Tensor | None = None
        self._capture_key: Any = None
        self._optimizer_graph: torch.cuda.CUDAGraph | None = None
        self._optimizer_lr_host: Tensor | None = None
        self._optimizer_lr_device: Tensor | None = None
        self._optimizer_gradient_parameters: tuple[Tensor, ...] = ()
        self.capture_count = 0
        self.replay_count = 0
        self.diagnostic_eager_count = 0
        self.capture_setup_seconds = 0.0
        self.optimizer_capture_count = 0
        self.optimizer_replay_count = 0
        self.optimizer_eager_step_count = 0
        self.optimizer_capture_setup_seconds = 0.0

    def _configure_compile_plan(self) -> AppliedTrainingCompilePlan | None:
        plan = resolve_training_compile_plan(
            self._acceleration_adapter,
            self.engine,
            self._compile_profile,
        )
        if plan is None:
            existing = get_applied_training_compile_plan(self.engine)
            if existing is not None:
                raise RuntimeError(
                    "engine already has a selective compile plan but the CUDA Graph "
                    "runner was configured with compile disabled"
                )
            return None
        return apply_training_compile_plan(
            self.engine,
            plan,
            profile=str(self._compile_profile),
            allow_candidate=self._allow_candidate_compile,
        )

    @property
    def compile_plan_summary(self) -> dict[str, Any] | None:
        if self._applied_compile_plan is None:
            return None
        return self._applied_compile_plan.summary()

    def _topology_key(self) -> Hashable:
        return self._acceleration_adapter.topology_signature(self.engine)

    def _capture_identity(self, batch: TrainingBatch) -> Hashable:
        """Describe every model/input choice that owns captured CUDA work."""

        return (
            self._acceleration_adapter.name,
            self._structure_signature,
            (
                "no-selective-compile"
                if self._applied_compile_plan is None
                else self._applied_compile_plan.signature
            ),
            self._topology_key(),
            _static_tree_signature(batch),
        )

    def invalidate(self, reason: str = "") -> None:
        """Drop the current graph after a versioned model is replaced.

        A graph owns parameter/activation addresses from its capture.  A
        branch that hot-swaps modules, changes widths or changes its training
        topology must call this at a step boundary.  Ordinary optimizer
        updates do not require invalidation because they preserve storage.
        """

        del reason  # retained in the API for callers' audit/logging wrappers
        self._release_capture()
        self._acceleration_adapter = resolve_training_acceleration_adapter(
            self.engine.model
        )
        self._acceleration_adapter.prepare(self.engine)
        self._applied_compile_plan = self._configure_compile_plan()
        self._structure_signature = self._acceleration_adapter.structure_signature(
            self.engine
        )

    def _release_capture(self) -> None:
        if self._graph is not None or self._optimizer_graph is not None:
            torch.cuda.synchronize(self.engine.device)
        if self._optimizer_graph is not None:
            self._optimizer_graph.reset()
        self._optimizer_graph = None
        self._optimizer_lr_host = None
        self._optimizer_lr_device = None
        self._optimizer_gradient_parameters = ()
        if self._graph is not None:
            self._graph.reset()
            # A replacement capture owns a new backward graph.  Drop the old
            # AccumulateGrad buffers before collecting its graph executable so
            # they cannot retain the previous private-stream nodes.
            self.engine.optimizer.zero_grad(set_to_none=True)
        self._graph = None
        self._captured_ledger = None
        self._captured_metrics = None
        self._captured_gradient_norm = None
        self._static_batch = None
        self._capture_key = None
        gc.collect()

    def _optimizer_state_is_capture_ready(self) -> bool:
        """Return whether fused AdamW has initialized every active state row.

        Fused AdamW lazily creates ``step`` and moment tensors.  Recording that
        initialization would put their zero-fill operations in the graph and
        replay them before every update.  The first fresh-run update therefore
        remains the ordinary optimizer call; capture begins only after those
        exact native state tensors exist.  A topology recapture repeats this
        guard so newly active parameters receive the same ordinary lazy-init
        semantics.
        """

        found_gradient = False
        optimizer = self.engine.optimizer
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                found_gradient = True
                state = optimizer.state.get(parameter, {})
                required = {"step", "exp_avg", "exp_avg_sq"}
                if bool(group.get("amsgrad", False)):
                    required.add("max_exp_avg_sq")
                if not required.issubset(state):
                    return False
        return found_gradient

    def _current_optimizer_gradient_parameters(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter
            for group in self.engine.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )

    def _stage_optimizer_learning_rates(self) -> None:
        host = self._optimizer_lr_host
        device = self._optimizer_lr_device
        if host is None or device is None:
            raise RuntimeError("captured optimizer learning-rate storage is absent")
        groups = self.engine.optimizer.param_groups
        if host.numel() != len(groups):
            raise RuntimeError("captured optimizer group count changed")
        for index, group in enumerate(groups):
            learning_rate = group["lr"]
            if isinstance(learning_rate, Tensor):
                raise RuntimeError(
                    "checkpoint-visible optimizer learning rate stopped being a float"
                )
            host[index] = float(learning_rate)
        device.copy_(host, non_blocking=True)

    def _capture_fused_optimizer_step(self) -> None:
        """Record one initialized fused AdamW update without executing it.

        CUDA stream capture records kernels but does not own the formal update;
        the immediate first replay below does.  Temporary device LR scalars are
        FP32 views of one vector.  They reproduce fused AdamW's native scalar
        conversion while every public param-group field is restored before the
        scheduler or checkpoint code can observe it.
        """

        forward_graph = self._graph
        if forward_graph is None:
            raise RuntimeError("optimizer capture requires a forward/backward graph")
        engine = self.engine
        optimizer = engine.optimizer
        gradient_parameters = self._current_optimizer_gradient_parameters()
        if not gradient_parameters:
            raise RuntimeError("captured optimizer found no active gradients")
        groups = optimizer.param_groups
        host = torch.empty(
            len(groups),
            dtype=torch.float32,
            pin_memory=True,
        )
        for index, group in enumerate(groups):
            learning_rate = group["lr"]
            if isinstance(learning_rate, Tensor):
                raise RuntimeError(
                    "captured optimizer requires checkpoint-visible float LRs"
                )
            host[index] = float(learning_rate)
        device = host.to(device=engine.device, non_blocking=False)
        originals = tuple(
            (group["lr"], bool(group.get("capturable", False)))
            for group in groups
        )
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        try:
            for index, group in enumerate(groups):
                group["lr"] = device[index]
                group["capturable"] = True
            # The two graphs always replay in capture order and never overlap,
            # so sharing their allocator pool avoids a second peak workspace.
            with torch.cuda.graph(graph, pool=forward_graph.pool()):
                optimizer.step()
        except Exception:
            graph.reset()
            raise
        finally:
            for group, (learning_rate, capturable) in zip(
                groups,
                originals,
                strict=True,
            ):
                group["lr"] = learning_rate
                group["capturable"] = capturable
        torch.cuda.synchronize(engine.device)
        self._optimizer_graph = graph
        self._optimizer_lr_host = host
        self._optimizer_lr_device = device
        self._optimizer_gradient_parameters = gradient_parameters
        self.optimizer_capture_count += 1
        self.optimizer_capture_setup_seconds += time.perf_counter() - started

    def _optimizer_step(self) -> None:
        """Execute one native or captured optimizer update."""

        if not self._capture_optimizer_step:
            self.engine.optimizer.step()
            self.optimizer_eager_step_count += 1
            return
        graph = self._optimizer_graph
        if graph is None:
            if not self._optimizer_state_is_capture_ready():
                self.engine.optimizer.step()
                self.optimizer_eager_step_count += 1
                return
            self._capture_fused_optimizer_step()
            graph = self._optimizer_graph
            if graph is None:
                raise RuntimeError("fused optimizer capture did not produce a graph")
        current_parameters = self._current_optimizer_gradient_parameters()
        if len(current_parameters) != len(self._optimizer_gradient_parameters) or any(
            current is not captured
            for current, captured in zip(
                current_parameters,
                self._optimizer_gradient_parameters,
                strict=True,
            )
        ):
            raise RuntimeError(
                "captured optimizer active-gradient parameter set changed; "
                "invalidate the runner at the topology boundary"
            )
        self._stage_optimizer_learning_rates()
        graph.replay()
        self.optimizer_replay_count += 1

    def _capture(self, batch: TrainingBatch) -> None:
        engine = self.engine
        # Preparation is idempotent for the active adapter and is repeated at
        # every recapture so a topology-specific branch can re-install its
        # own capture-safe implementation details.
        self._acceleration_adapter.prepare(engine)
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
            # Warm-up activations belong to the ordinary allocator pool and
            # cannot back the private graph pool.  Release only those cached,
            # unreferenced blocks before recording to avoid a transient near-
            # duplicate activation footprint on 24 GiB training GPUs.
            torch.cuda.empty_cache()

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
                    captured_gradient_norm = (
                        engine._captured_gradient_lifecycle()
                        if self._capture_gradient_lifecycle
                        else None
                    )
            self._graph = graph
            self._captured_ledger = captured_ledger
            self._captured_metrics = captured_metrics
            self._captured_gradient_norm = captured_gradient_norm
            self.capture_count += 1
        finally:
            _restore_rng_snapshot(engine, capture_rng)
            torch.cuda.synchronize(engine.device)
            self.capture_setup_seconds += time.perf_counter() - started

    def _ensure_capture(self, batch: TrainingBatch) -> float:
        batch.validate(self.engine.config)
        capture_key = self._capture_identity(batch)
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
            self.diagnostic_eager_count += 1
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
        captured_gradient_norm = self._captured_gradient_norm
        if graph is None or ledger is None or captured_metrics is None:
            raise RuntimeError("CUDA Graph training capture is incomplete")
        replay_started = time.perf_counter()
        graph.replay()
        self.replay_count += 1
        timing_mark("cuda_graph_forward_backward_seconds", replay_started)

        gradient_started = time.perf_counter()
        if captured_gradient_norm is None:
            gradient_norm, gradient_metrics, gradient_norm_scalar = (
                engine._gradient_lifecycle(
                    collect_diagnostics=False,
                    gradient_spike_handler=gradient_spike_handler,
                )
            )
        else:
            if gradient_spike_handler is not None:
                raise RuntimeError(
                    "captured gradient lifecycle cannot run with a spike handler"
                )
            gradient_norm = captured_gradient_norm
            gradient_norm_scalar = float(
                gradient_norm.detach().float().cpu().item()
            )
            if not math.isfinite(gradient_norm_scalar):
                raise NonFiniteGradientError(
                    engine._first_nonfinite_gradient_report(
                        global_norm=gradient_norm
                    )
                )
            gradient_metrics = {}
        timing_mark("gradient_lifecycle_seconds", gradient_started)
        learning_rate = float(
            engine.config.optimizer.learning_rate
            * engine.schedule.ratio(engine.schedule.step_index)
        )
        optimizer_started = time.perf_counter()
        self._optimizer_step()
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
