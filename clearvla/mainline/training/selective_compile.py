"""Fail-closed application of version-owned selective compile plans.

The model adapter describes *what* may be compiled.  This module owns the
generic mechanics: path validation, eager numerical barriers, Inductor option
validation, idempotence and an execution signature that CUDA Graph capture can
include.  It never imports a model implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .acceleration_contract import TrainingCompilePlan

_APPLICATION_ATTRIBUTE = "_clearvla_training_compile_application"
_TRANSFORM_ATTRIBUTE = "_clearvla_training_compile_transforms"


@dataclass(frozen=True)
class AppliedTrainingCompilePlan:
    """Audit metadata for one plan installed on one model instance."""

    profile: str
    plan_name: str
    acceptance: str
    rng_policy: str
    numerical_policy: str
    signature: tuple[Hashable, ...]
    signature_sha256: str
    compiled_module_paths: tuple[str, ...]
    eager_boundaries: tuple[tuple[str, str], ...]
    wrapper_setup_seconds: float
    model_identity: int

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "plan_name": self.plan_name,
            "acceptance": self.acceptance,
            "rng_policy": self.rng_policy,
            "numerical_policy": self.numerical_policy,
            "signature_sha256": self.signature_sha256,
            "compiled_module_paths": list(self.compiled_module_paths),
            "eager_boundaries": [list(value) for value in self.eager_boundaries],
            "wrapper_setup_seconds": self.wrapper_setup_seconds,
        }


def _named_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    if len(modules) != sum(1 for _name, _module in model.named_modules()):
        raise RuntimeError("model exposes duplicate named-module paths")
    return modules


def _resolve_modules(
    model: nn.Module,
    plan: TrainingCompilePlan,
) -> tuple[dict[str, nn.Module], tuple[str, ...]]:
    modules = _named_modules(model)
    compile_paths = tuple(
        path for region in plan.regions for path in region.module_paths
    )
    requested_paths = set(compile_paths)
    requested_paths.update(boundary.module_path for boundary in plan.eager_boundaries)
    missing = tuple(sorted(requested_paths.difference(modules)))
    if missing:
        raise ValueError(
            f"compile plan {plan.name!r} references missing modules: {missing!r}"
        )
    for boundary in plan.eager_boundaries:
        if not any(
            boundary.module_path == parent
            or boundary.module_path.startswith(parent + ".")
            for parent in compile_paths
        ):
            raise ValueError(
                f"eager boundary {boundary.module_path}.{boundary.method_name} "
                "is not inside a compiled region"
            )
        method = getattr(modules[boundary.module_path], boundary.method_name, None)
        if not callable(method):
            raise ValueError(
                f"eager boundary is not callable: "
                f"{boundary.module_path}.{boundary.method_name}"
            )
    return modules, compile_paths


def _validate_inductor_options(plan: TrainingCompilePlan) -> None:
    option_names = {
        name for region in plan.regions for name, _value in region.options
    }
    if not option_names:
        return
    try:
        from torch import _inductor

        supported = set(_inductor.list_options())
    except (AttributeError, ImportError) as error:
        raise RuntimeError("Inductor option discovery is unavailable") from error
    unsupported = tuple(sorted(option_names.difference(supported)))
    if unsupported:
        raise ValueError(
            f"compile plan {plan.name!r} requires unsupported Inductor options: "
            f"{unsupported!r}"
        )


def _method_transform_map(module: nn.Module) -> dict[str, tuple[Hashable, ...]]:
    value = getattr(module, _TRANSFORM_ATTRIBUTE, None)
    if value is None:
        value = {}
        setattr(module, _TRANSFORM_ATTRIBUTE, value)
    if not isinstance(value, dict):
        raise TypeError(f"reserved attribute {_TRANSFORM_ATTRIBUTE} is not a mapping")
    return value


def _existing_method_transform_map(
    module: nn.Module,
) -> dict[str, tuple[Hashable, ...]] | None:
    value = getattr(module, _TRANSFORM_ATTRIBUTE, None)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"reserved attribute {_TRANSFORM_ATTRIBUTE} is not a mapping")
    return value


def _install_method(
    module: nn.Module,
    method_name: str,
    replacement: Callable[..., Any],
    *,
    transform_signature: tuple[Hashable, ...],
) -> bool:
    transforms = _method_transform_map(module)
    previous = transforms.get(method_name)
    if previous is not None and previous != transform_signature:
        raise RuntimeError(
            f"module method {type(module).__qualname__}.{method_name} already has "
            "a different training compile transform"
        )
    if previous is None:
        setattr(module, method_name, replacement)
        transforms[method_name] = transform_signature
        return True
    return False


def _preflight_transform(
    module: nn.Module,
    method_name: str,
    transform_signature: tuple[Hashable, ...],
) -> bool:
    """Reject conflicts before any callable on the model is modified."""

    transforms = _existing_method_transform_map(module)
    if transforms is None or method_name not in transforms:
        return True
    if transforms[method_name] != transform_signature:
        raise RuntimeError(
            f"module method {type(module).__qualname__}.{method_name} already has "
            "a different training compile transform"
        )
    return False


def _disable_callable(
    method: Callable[..., Any],
    *,
    reason: str,
) -> Callable[..., Any]:
    disable = torch.compiler.disable
    parameters = inspect.signature(disable).parameters
    kwargs: dict[str, Any] = {"recursive": True}
    if "reason" in parameters:
        kwargs["reason"] = reason
    return disable(method, **kwargs)


def get_applied_training_compile_plan(engine: Any) -> AppliedTrainingCompilePlan | None:
    value = getattr(engine, _APPLICATION_ATTRIBUTE, None)
    if value is None:
        return None
    if not isinstance(value, AppliedTrainingCompilePlan):
        raise TypeError(f"reserved attribute {_APPLICATION_ATTRIBUTE} has invalid type")
    if value.model_identity != id(engine.model):
        return None
    return value


def apply_training_compile_plan(
    engine: Any,
    plan: TrainingCompilePlan,
    *,
    profile: str,
    allow_candidate: bool = False,
) -> AppliedTrainingCompilePlan:
    """Install one selective compile plan without changing model ownership.

    All paths and options are resolved before the first callable is replaced.
    Candidate plans require an explicit experimental opt-in.  A model instance
    cannot switch plans in place because compiled wrappers may own cached
    graphs; callers must construct a fresh model for a different plan.
    """

    plan.validate()
    if not profile:
        raise ValueError("compile profile cannot be empty")
    if plan.acceptance != "strict-gated" and not allow_candidate:
        raise PermissionError(
            f"compile plan {plan.name!r} is only a candidate; "
            "set allow_candidate=True for an isolated experiment"
        )
    existing = get_applied_training_compile_plan(engine)
    signature = plan.signature()
    if existing is not None:
        if existing.profile == profile and existing.signature == signature:
            return existing
        raise RuntimeError(
            "a different selective compile plan is already installed on this model; "
            "construct a fresh model before switching profiles"
        )

    modules, compile_paths = _resolve_modules(engine.model, plan)
    _validate_inductor_options(plan)
    for path in compile_paths:
        if not callable(getattr(modules[path], "forward", None)):
            raise ValueError(f"compiled module has no callable forward: {path!r}")

    boundary_actions: list[
        tuple[Any, str, str, tuple[Hashable, ...], bool]
    ] = []
    for boundary in plan.eager_boundaries:
        module = modules[boundary.module_path]
        transform_signature = (
            "eager-boundary-v1",
            signature,
            boundary.module_path,
            boundary.method_name,
        )
        boundary_actions.append(
            (
                module,
                boundary.method_name,
                boundary.reason,
                transform_signature,
                _preflight_transform(
                    module,
                    boundary.method_name,
                    transform_signature,
                ),
            )
        )

    compile_actions: list[
        tuple[Any, str, Mapping[str, str | int | bool], tuple[Hashable, ...], bool]
    ] = []
    for region in plan.regions:
        options: Mapping[str, str | int | bool] = dict(region.options)
        for path in region.module_paths:
            module = modules[path]
            transform_signature = (
                "compiled-forward-v1",
                signature,
                region.name,
                path,
            )
            compile_actions.append(
                (
                    module,
                    region.name,
                    options,
                    transform_signature,
                    _preflight_transform(module, "forward", transform_signature),
                )
            )

    started = time.perf_counter()
    changed: list[tuple[nn.Module, str, bool, Any]] = []
    transform_snapshots: dict[
        nn.Module,
        tuple[bool, dict[str, tuple[Hashable, ...]] | None],
    ] = {}

    def snapshot_transforms(module: nn.Module) -> None:
        if module in transform_snapshots:
            return
        transforms = _existing_method_transform_map(module)
        transform_snapshots[module] = (
            transforms is not None,
            None if transforms is None else dict(transforms),
        )

    try:
        for module, method_name, reason, transform_signature, needs_install in (
            boundary_actions
        ):
            if not needs_install:
                continue
            original = getattr(module, method_name)
            had_instance_method = method_name in module.__dict__
            original_instance_method = module.__dict__.get(method_name)
            snapshot_transforms(module)
            installed = _install_method(
                module,
                method_name,
                _disable_callable(original, reason=reason),
                transform_signature=transform_signature,
            )
            if installed:
                changed.append(
                    (
                        module,
                        method_name,
                        had_instance_method,
                        original_instance_method,
                    )
                )

        region_by_name = {region.name: region for region in plan.regions}
        for module, region_name, options, transform_signature, needs_install in (
            compile_actions
        ):
            if not needs_install:
                continue
            region = region_by_name[region_name]
            original = module.forward
            had_instance_method = "forward" in module.__dict__
            original_instance_method = module.__dict__.get("forward")
            compiled = torch.compile(
                original,
                dynamic=region.dynamic,
                fullgraph=region.fullgraph,
                mode=region.mode,
                options=dict(options),
            )
            snapshot_transforms(module)
            installed = _install_method(
                module,
                "forward",
                compiled,
                transform_signature=transform_signature,
            )
            if installed:
                changed.append(
                    (
                        module,
                        "forward",
                        had_instance_method,
                        original_instance_method,
                    )
                )
    except Exception:
        for module, method_name, had_instance_method, original in reversed(changed):
            if had_instance_method:
                setattr(module, method_name, original)
            elif method_name in module.__dict__:
                delattr(module, method_name)
        for module, (had_map, original_map) in transform_snapshots.items():
            if had_map:
                setattr(module, _TRANSFORM_ATTRIBUTE, dict(original_map or {}))
            elif hasattr(module, _TRANSFORM_ATTRIBUTE):
                delattr(module, _TRANSFORM_ATTRIBUTE)
        raise

    signature_sha256 = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()
    result = AppliedTrainingCompilePlan(
        profile=profile,
        plan_name=plan.name,
        acceptance=plan.acceptance,
        rng_policy=plan.rng_policy,
        numerical_policy=plan.numerical_policy,
        signature=signature,
        signature_sha256=signature_sha256,
        compiled_module_paths=compile_paths,
        eager_boundaries=tuple(
            (boundary.module_path, boundary.method_name)
            for boundary in plan.eager_boundaries
        ),
        wrapper_setup_seconds=time.perf_counter() - started,
        model_identity=id(engine.model),
    )
    setattr(engine, _APPLICATION_ATTRIBUTE, result)
    return result


__all__ = [
    "AppliedTrainingCompilePlan",
    "apply_training_compile_plan",
    "get_applied_training_compile_plan",
]
