"""Finite fixed-batch learning measurements, not training/skill acceptance.

Uses the actual engine and a freshly encoded graph on every call. Evaluations
have repeatable random draws without consuming the subsequent training stream.
The caller owns the model, schedule, source admission and process lifetime.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import torch

from ..interfaces import TrainingBatch
from ..training.engine import MainlineTrainingEngine, TrainStepResult


def _metrics(result: TrainStepResult) -> dict[str, Any]:
    loss = float(result.loss.detach())
    if not math.isfinite(loss):
        raise FloatingPointError("non-finite fixed-batch objective")
    goal = {
        name: float(value.detach())
        for name, value in result.metrics.items()
        if name.startswith("loss_annotated_goal_")
        or name == "loss_contrib_annotated_goal"
    }
    if any(not math.isfinite(value) for value in goal.values()):
        raise FloatingPointError("non-finite fixed-batch goal metric")
    return {"loss": loss, "annotated_goal": goal}


def evaluate_fixed_batch(
    engine: MainlineTrainingEngine, batch: TrainingBatch, *, seed: int
) -> dict[str, Any]:
    """Repeatable eval at the real completed-update clock, with no cached graph.

    Only CPU and the active CUDA RNG are forked. Seeding every GPU while
    restoring just one would perturb unrelated random streams.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("evaluation seed must be a nonnegative 64-bit integer")
    device = engine.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("fixed-batch evaluation supports the actual CPU/CUDA engine")
    devices = [] if device.type == "cpu" else [
        torch.cuda.current_device() if device.index is None else device.index
    ]
    training_flags = [(module, module.training) for module in engine.model.modules()]
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            generator = torch.Generator(device=device).manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.set_rng_state(generator.get_state(), device=device)
            result = engine.eval_step(batch, collect_diagnostics=False, generator=generator)
            return {**_metrics(result), "completed_optimizer_updates": engine.global_step}
    finally:
        for module, training in training_flags:
            module.training = training


def probe_fixed_batch_learning(
    engine: MainlineTrainingEngine,
    batch: TrainingBatch,
    *,
    updates: int,
    evaluation_seed: int = 8302,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure before/after loss and every completed update on ONE fixed batch.

    A decreasing loss is neither required for numerical completion nor called
    validation success. No held-out split, early stopping, checkpoint migration,
    task oracle, data replacement, schedule override or forced phase is hidden
    in this helper. An exception is re-raised with partial progress preserved.
    """
    if type(updates) is not int or updates < 1:
        raise ValueError("updates must be a positive integer")
    if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**64:
        raise ValueError("evaluation seed must be a nonnegative 64-bit integer")
    start = engine.global_step
    trace: dict[str, Any] = {
        "status": "running",
        "scope": "one fixed batch; training-set measurement, not held-out/physics acceptance",
        "updates_requested": updates,
        "initial_completed_updates": start,
        "completed_updates": 0,
        "evaluation_seed": evaluation_seed,
        "evaluation_clock": "actual engine completed updates; phase is not held fixed",
        "steps": [],
    }

    def publish(stage: str) -> None:
        trace["stage"] = stage
        trace["completed_updates"] = engine.global_step - start
        if on_progress is not None:
            on_progress(copy.deepcopy(trace))

    try:
        publish("evaluation-before")
        trace["before"] = evaluate_fixed_batch(engine, batch, seed=evaluation_seed)
        publish("evaluation-before-complete")
        for index in range(updates):
            publish(f"update-{index + 1}")
            result = engine.train_step(batch, collect_diagnostics=False)
            if engine.global_step != start + index + 1:
                raise RuntimeError("engine did not report one completed update")
            trace["steps"].append({
                **_metrics(result),
                "completed_optimizer_updates": engine.global_step,
                "learning_rate_used": result.learning_rate,
                "gradient_norm": float(result.gradient_norm.detach()),
            })
            publish(f"update-{index + 1}-complete")
        publish("evaluation-after")
        trace["after"] = evaluate_fixed_batch(engine, batch, seed=evaluation_seed)
        trace["loss_change"] = trace["after"]["loss"] - trace["before"]["loss"]
        trace["status"] = "completed"
        publish("complete")
    except BaseException as error:
        trace.update(status="failed", error=type(error).__name__, message=str(error))
        publish("failed")
        raise
    return trace
