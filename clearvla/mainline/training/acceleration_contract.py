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
from typing import Any, Literal, Protocol, runtime_checkable

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


CompileAcceptance = Literal["candidate", "strict-gated"]


@dataclass(frozen=True)
class TrainingEagerBoundary:
    """One callable that must retain its eager numerical implementation.

    ``module_path`` uses the names emitted by ``nn.Module.named_modules`` and
    ``method_name`` defaults to ``forward``.  Method-level boundaries let a
    coarse module retain a numerically sensitive reduction while Inductor
    still fuses the surrounding pointwise/GEMM work.
    """

    module_path: str
    method_name: str = "forward"
    reason: str = "numerically-sensitive"

    def validate(self) -> None:
        if not self.module_path:
            raise ValueError("eager boundary module path cannot be empty")
        if not self.method_name.isidentifier():
            raise ValueError("eager boundary method name is invalid")
        if not self.reason:
            raise ValueError("eager boundary reason cannot be empty")


@dataclass(frozen=True)
class TrainingCompileRegion:
    """A named collection of module forwards compiled with one policy."""

    name: str
    module_paths: tuple[str, ...]
    mode: str | None = None
    dynamic: bool = False
    fullgraph: bool = False
    options: tuple[tuple[str, str | int | bool], ...] = ()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("compile region name cannot be empty")
        if not self.module_paths or any(not path for path in self.module_paths):
            raise ValueError(f"compile region {self.name!r} has no module paths")
        if len(set(self.module_paths)) != len(self.module_paths):
            raise ValueError(f"compile region {self.name!r} repeats a module path")
        if self.mode == "":
            raise ValueError(f"compile region {self.name!r} has an empty mode")
        option_names = tuple(name for name, _value in self.options)
        if any(not name for name in option_names) or len(set(option_names)) != len(
            option_names
        ):
            raise ValueError(f"compile region {self.name!r} has invalid options")
        if self.mode is not None and self.options:
            raise ValueError(
                f"compile region {self.name!r} cannot set both mode and options"
            )


@dataclass(frozen=True)
class TrainingCompilePlan:
    """Version-owned, hashable selective-compilation contract.

    Candidate plans are deliberately rejected by production callers unless
    they explicitly opt into experiments.  Promotion to ``strict-gated`` is a
    source change backed by the full eager/compiled equivalence suite.
    """

    name: str
    regions: tuple[TrainingCompileRegion, ...]
    eager_boundaries: tuple[TrainingEagerBoundary, ...] = ()
    acceptance: CompileAcceptance = "candidate"
    rng_policy: str = "eager-fallback-or-explicit-input"
    numerical_policy: str = "preserve-eager-sensitive-operations"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("compile plan name cannot be empty")
        if self.acceptance not in {"candidate", "strict-gated"}:
            raise ValueError(f"invalid compile-plan acceptance: {self.acceptance!r}")
        if not self.rng_policy or not self.numerical_policy:
            raise ValueError("compile-plan policies cannot be empty")
        if not self.regions:
            raise ValueError(f"compile plan {self.name!r} has no regions")
        region_names: list[str] = []
        compile_paths: list[str] = []
        for region in self.regions:
            region.validate()
            region_names.append(region.name)
            compile_paths.extend(region.module_paths)
        if len(set(region_names)) != len(region_names):
            raise ValueError(f"compile plan {self.name!r} repeats a region name")
        if len(set(compile_paths)) != len(compile_paths):
            raise ValueError(f"compile plan {self.name!r} compiles a module twice")
        boundary_keys: list[tuple[str, str]] = []
        for boundary in self.eager_boundaries:
            boundary.validate()
            boundary_keys.append((boundary.module_path, boundary.method_name))
        if len(set(boundary_keys)) != len(boundary_keys):
            raise ValueError(f"compile plan {self.name!r} repeats an eager boundary")
        direct_conflicts = set(compile_paths).intersection(
            path
            for path, method_name in boundary_keys
            if method_name == "forward"
        )
        if direct_conflicts:
            raise ValueError(
                f"compile plan {self.name!r} both compiles and disables "
                f"{sorted(direct_conflicts)!r}"
            )
        for parent_index, parent in enumerate(compile_paths):
            for child in compile_paths[parent_index + 1 :]:
                if child.startswith(parent + ".") or parent.startswith(child + "."):
                    raise ValueError(
                        f"compile plan {self.name!r} nests compiled modules: "
                        f"{parent!r}, {child!r}"
                    )

    def signature(self) -> tuple[Hashable, ...]:
        """Return the complete execution identity used by CUDA Graph keys."""

        self.validate()
        return (
            "training-compile-plan-v1",
            self.name,
            self.acceptance,
            self.rng_policy,
            self.numerical_policy,
            tuple(
                (
                    region.name,
                    region.module_paths,
                    region.mode,
                    region.dynamic,
                    region.fullgraph,
                    region.options,
                )
                for region in self.regions
            ),
            tuple(
                (boundary.module_path, boundary.method_name, boundary.reason)
                for boundary in self.eager_boundaries
            ),
        )


@runtime_checkable
class TrainingCompilePlanProvider(Protocol):
    """Optional extension implemented only by compile-aware adapters."""

    def compile_plan(self, engine: Any, profile: str) -> TrainingCompilePlan:
        """Resolve a version-local plan for an explicitly requested profile."""


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

    def compile_plan(self, engine: Any, profile: str) -> TrainingCompilePlan:
        """Delegate an optional plan to a model without importing its class."""

        hook = getattr(engine.model, "training_acceleration_compile_plan", None)
        if not callable(hook):
            raise ValueError(
                f"model has no selective compile plan for profile {profile!r}"
            )
        plan = hook(profile)
        if not isinstance(plan, TrainingCompilePlan):
            raise TypeError("model compile-plan hook must return TrainingCompilePlan")
        plan.validate()
        return plan


def resolve_training_compile_plan(
    adapter: TrainingAccelerationAdapter,
    engine: Any,
    profile: str | None,
) -> TrainingCompilePlan | None:
    """Resolve an optional plan while keeping the base adapter ABI stable."""

    if profile is None or profile in {"", "none", "off"}:
        return None
    hook = getattr(adapter, "compile_plan", None)
    if not callable(hook):
        raise ValueError(
            f"training adapter {adapter.name!r} does not support compile profile "
            f"{profile!r}"
        )
    plan = hook(engine, profile)
    if not isinstance(plan, TrainingCompilePlan):
        raise TypeError("adapter compile_plan must return TrainingCompilePlan")
    plan.validate()
    return plan


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
    "CompileAcceptance",
    "GenericTrainingAccelerationAdapter",
    "TrainingAccelerationAdapter",
    "TrainingCompilePlan",
    "TrainingCompilePlanProvider",
    "TrainingCompileRegion",
    "TrainingEagerBoundary",
    "module_structure_signature",
    "resolve_training_acceleration_adapter",
    "resolve_training_compile_plan",
]
