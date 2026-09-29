"""Static P3-compiled task values; never candidate actions or future labels.

The record is runtime provenance, not persistent robot identity. Parameters are
serialized normally; source references and compiler identities are not saved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PROPRIOCEPTIVE_GLOBAL = "proprioceptive_prior_v1"
COMPILED_TASK_GLOBAL = "p3_compiled_task_v1"


def validate_global_condition_mode(mode: str) -> None:
    if mode not in {PROPRIOCEPTIVE_GLOBAL, COMPILED_TASK_GLOBAL}:
        raise ValueError("unknown bottom global condition mode")


def global_task_metadata() -> dict[str, object]:
    return {
        "schema": "p3-compiled-causal-task-global-v1",
        "source": "S-public-interval-carrier-after-causal-task-reference-and-intent-reads",
        "compiler": "P3-bias-free-nonlinear-value-projection",
        "memory": "four-interval-task-values-plus-existing-state-and-executed-memory",
        "lifetime": "one-online-encoding-shared-by-both-passes-and-ODE-nodes",
        "excluded": ["raw-language", "raw-visual", "simulator-id", "future-label",
                     "candidate-W", "noisy-action"],
        "task_adapter": "bias-free-projection-without-type-embedding-value",
        "consumer": "existing-layer-value-scan-and-global-latent-organizer",
        "claim": "task-conditioned-representation-not-guaranteed-task-control",
    }


@dataclass(frozen=True)
class CompiledGlobalTask:
    tokens: Tensor
    source_intervals: Tensor
    compiler_identity: int

    def validate(self, source: Tensor, *, hidden: int, strict: bool = False) -> None:
        if self.source_intervals is not source:
            raise ValueError("compiled global task belongs to another observation/intent")
        if source.ndim != 3 or source.shape[1:] != (4, hidden):
            raise ValueError("global task source must retain four declared intervals")
        if self.tokens.shape != source.shape or self.tokens.device != source.device:
            raise ValueError("compiled global task lost its source shape/device")
        if not self.tokens.is_floating_point() or not source.is_floating_point():
            raise TypeError("compiled global task must be floating point")
        if type(self.compiler_identity) is not int:
            raise TypeError("compiled global task lost its runtime compiler identity")
        # Numerical reductions occur only at creation, not in every ODE call.
        if strict and (not bool(torch.isfinite(source).all())
                       or not bool(torch.isfinite(self.tokens).all())):
            raise ValueError("nonfinite supported global task source/value")


def global_intent_memory(*, mode: str, compiled: CompiledGlobalTask | None,
                         source: Tensor, state: Tensor, executed: Tensor) -> dict[str, Tensor]:
    """Shared admission for integrated and directly restored bottom adapters."""
    validate_global_condition_mode(mode)
    if mode == PROPRIOCEPTIVE_GLOBAL:
        if compiled is not None:
            raise ValueError("compiled global task supplied to an unselected bottom")
        return {"state": state, "executed": executed}
    if compiled is None:
        raise ValueError("selected global task bottom requires P3-compiled values")
    compiled.validate(source, hidden=state.shape[-1])
    if compiled.tokens.shape[0] != state.shape[0] or compiled.tokens.device != state.device:
        raise ValueError("global task and proprioceptive memory have different sources")
    return {"task": compiled.tokens, "state": state, "executed": executed}
