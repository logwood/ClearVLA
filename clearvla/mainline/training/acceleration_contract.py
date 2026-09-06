"""Stable implementation-level contract for training acceleration backends.

The acceleration runtime must survive ordinary network refactors.  It therefore
depends on a small adapter surface (capture preparation, topology identity and
structure identity) instead of importing a particular decoder or block list.
Versioned model implementations can provide their own adapter while sharing
the CUDA-Graph runner and its eager fallback.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from torch import nn


@runtime_checkable
class TrainingAccelerationAdapter(Protocol):
    """Implementation hook consumed by a training acceleration backend."""

    name: str

    def prepare(self, engine: Any) -> None:
        """Enable only implementation-local capture-safe fast paths."""

    def topology_signature(self, engine: Any) -> Hashable:
        """Return the current static execution topology identity."""

    def structure_signature(self, engine: Any) -> Hashable:
        """Return a value that changes when the model structure is replaced."""


def module_structure_signature(module: nn.Module) -> tuple[Hashable, ...]:
    """Describe module classes and tensor contracts, never parameter values.

    This is intentionally independent of the model's semantic names.  It is
    used at capture setup and after an explicit invalidation, so replacing a
    block stack or changing a width cannot silently reuse an old graph.  Normal
    optimizer updates do not change the signature.
    """

    entries: list[Hashable] = []
    for module_name, child in module.named_modules():
        child_type = type(child)
        entries.append(
            (
                "module",
                module_name,
                child_type.__module__,
                child_type.__qualname__,
            )
        )
        for parameter_name, parameter in child.named_parameters(recurse=False):
            entries.append(
                (
                    "parameter",
                    f"{module_name}.{parameter_name}",
                    tuple(int(size) for size in parameter.shape),
                    str(parameter.dtype),
                    bool(parameter.requires_grad),
                )
            )
        for buffer_name, buffer in child.named_buffers(recurse=False):
            entries.append(
                (
                    "buffer",
                    f"{module_name}.{buffer_name}",
                    tuple(int(size) for size in buffer.shape),
                    str(buffer.dtype),
                )
            )
    return tuple(entries)


def _validate_adapter(candidate: Any) -> TrainingAccelerationAdapter:
    required = ("name", "prepare", "topology_signature", "structure_signature")
    missing = tuple(name for name in required if not hasattr(candidate, name))
    if missing:
        raise TypeError(
            "training acceleration adapter is missing required members: "
            + ", ".join(missing)
        )
    return candidate


@dataclass(frozen=True)
class GenericTrainingAccelerationAdapter:
    """Safe fallback for a versioned model without a bespoke adapter."""

    name: str = "generic-static-v1"

    def prepare(self, engine: Any) -> None:
        del engine

    def topology_signature(self, engine: Any) -> Hashable:
        hook = getattr(engine.model, "training_acceleration_topology_signature", None)
        if hook is None:
            return ("static",)
        value = hook()
        if not isinstance(value, Hashable):
            raise TypeError("model topology signature must be hashable")
        return ("model", value)

    def structure_signature(self, engine: Any) -> Hashable:
        hook = getattr(engine.model, "training_acceleration_structure_signature", None)
        if hook is not None:
            value = hook()
            if not isinstance(value, Hashable):
                raise TypeError("model structure signature must be hashable")
            return ("model", value)
        return module_structure_signature(engine.model)


def resolve_training_acceleration_adapter(model: nn.Module) -> TrainingAccelerationAdapter:
    """Resolve a version adapter without importing a versioned model class."""

    factory = getattr(model, "get_training_acceleration_adapter", None)
    if callable(factory):
        candidate = factory()
        if candidate is not None:
            return _validate_adapter(candidate)
    candidate = getattr(model, "training_acceleration_adapter", None)
    if candidate is not None:
        return _validate_adapter(candidate)
    return GenericTrainingAccelerationAdapter()


__all__ = [
    "GenericTrainingAccelerationAdapter",
    "TrainingAccelerationAdapter",
    "module_structure_signature",
    "resolve_training_acceleration_adapter",
]
