"""Atomic mainline checkpoints with exact-resume and explicit migration."""

from __future__ import annotations

import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from torch import Tensor, nn

from ..checkpoint import (
    CheckpointIdentity,
    checkpoint_identity_from_mapping,
    compare_checkpoint_identity,
)
from ..config import ExperimentConfig, config_from_mapping
from ..training.optimizer import WarmupCosineSchedule

CHECKPOINT_SCHEMA = "clearvla-mainline-checkpoint-v4"


@dataclass(frozen=True)
class RestoredTrainingState:
    epoch: int
    global_step: int
    best_metric: float | None


@dataclass(frozen=True)
class MigrationReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    rejected: tuple[str, ...]


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
    }


def _restore_rng(value: Mapping[str, object]) -> None:
    random.setstate(value["python"])  # type: ignore[arg-type]
    np.random.set_state(value["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(cast(torch.Tensor, value["torch_cpu"]))
    cuda = value.get("torch_cuda", ())
    if torch.cuda.is_available() and isinstance(cuda, (tuple, list)) and cuda:
        torch.cuda.set_rng_state_all(list(cuda))


def _validate_optimizer_state(
    saved_optimizer: Mapping[str, object],
    *,
    optimizer: torch.optim.Optimizer,
    saved_groups: list[object],
    current_groups: list[object],
) -> None:
    """Validate AdamW ownership and tensor shapes before model mutation."""

    saved_state = saved_optimizer.get("state")
    if not isinstance(saved_state, Mapping):
        raise ValueError("exact resume optimizer state is not a mapping")
    parameter_by_saved_id: dict[int, nn.Parameter] = {}
    for saved_group_raw, current_group_raw, live_group in zip(
        saved_groups,
        current_groups,
        optimizer.param_groups,
        strict=True,
    ):
        if not isinstance(saved_group_raw, Mapping) or not isinstance(
            current_group_raw,
            Mapping,
        ):
            raise ValueError("exact resume optimizer group is not a mapping")
        if set(saved_group_raw) != set(current_group_raw):
            raise ValueError("exact resume optimizer group fields differ")
        saved_ids = saved_group_raw.get("params")
        current_ids = current_group_raw.get("params")
        live_parameters = live_group.get("params")
        if (
            not isinstance(saved_ids, (tuple, list))
            or not isinstance(
                current_ids,
                (tuple, list),
            )
            or not isinstance(live_parameters, (tuple, list))
        ):
            raise ValueError("exact resume optimizer parameter list is invalid")
        if not (len(saved_ids) == len(current_ids) == len(live_parameters)):
            raise ValueError("exact resume optimizer parameter count differs")
        for saved_id, parameter in zip(saved_ids, live_parameters, strict=True):
            if isinstance(saved_id, bool) or not isinstance(saved_id, int):
                raise ValueError("exact resume optimizer parameter id is invalid")
            if saved_id in parameter_by_saved_id:
                raise ValueError("exact resume optimizer parameter id is duplicated")
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("live optimizer contains a non-parameter tensor")
            parameter_by_saved_id[saved_id] = parameter

    unknown = set(saved_state).difference(parameter_by_saved_id)
    if unknown:
        raise ValueError("exact resume optimizer state owns unknown parameters")
    for raw_parameter_id, raw_state in saved_state.items():
        if not isinstance(raw_parameter_id, int) or not isinstance(raw_state, Mapping):
            raise ValueError("exact resume optimizer parameter state is invalid")
        parameter = parameter_by_saved_id[raw_parameter_id]
        for state_name, state_value in raw_state.items():
            if not isinstance(state_value, Tensor):
                if not isinstance(state_value, (bool, int, float)):
                    raise ValueError(f"optimizer state {state_name!r} is neither tensor nor scalar")
                continue
            if state_name == "step":
                if state_value.numel() != 1:
                    raise ValueError("optimizer step state must be scalar")
            elif tuple(state_value.shape) != tuple(parameter.shape):
                raise ValueError(f"optimizer state {state_name!r} has an incompatible shape")
            if not bool(torch.isfinite(state_value).all()):
                raise ValueError(f"optimizer state {state_name!r} is non-finite")


def _validate_rng_state(
    value: Mapping[str, object],
    generators: Mapping[str, torch.Generator],
    saved_generators: Mapping[str, object],
) -> None:
    """Exercise RNG states on private generators before restoring live state."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not required.issubset(value):
        raise ValueError("exact resume global RNG ownership is incomplete")
    try:
        random.Random().setstate(value["python"])  # type: ignore[arg-type]
        np.random.RandomState().set_state(value["numpy"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("exact resume host RNG state is invalid") from error
    cpu_state = value["torch_cpu"]
    if not isinstance(cpu_state, Tensor):
        raise ValueError("exact resume CPU RNG state is not a tensor")
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
    except RuntimeError as error:
        raise ValueError("exact resume CPU RNG state is invalid") from error
    cuda_states = value["torch_cuda"]
    if not isinstance(cuda_states, (tuple, list)):
        raise ValueError("exact resume CUDA RNG state is not a sequence")
    if torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("exact resume CUDA RNG device ownership differs")
    for index, state in enumerate(cuda_states):
        if not isinstance(state, Tensor):
            raise ValueError("exact resume CUDA RNG state is not a tensor")
        if torch.cuda.is_available():
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state)
            except RuntimeError as error:
                raise ValueError("exact resume CUDA RNG state is invalid") from error
    for name, generator in generators.items():
        state = saved_generators[name]
        if not isinstance(state, Tensor):
            raise ValueError(f"generator state {name!r} is not a tensor")
        try:
            torch.Generator(device=generator.device).set_state(state)
        except RuntimeError as error:
            raise ValueError(f"generator state {name!r} is invalid") from error


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: WarmupCosineSchedule,
    config: ExperimentConfig,
    identity: CheckpointIdentity,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    data_state: Mapping[str, object] | None = None,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    """Write one recoverable checkpoint without serializing runtime caches."""

    config.validate()
    identity.validate()
    if int(epoch) < 0 or int(global_step) < 0:
        raise ValueError("checkpoint epoch/global step must be non-negative")
    if int(schedule.step_index) != int(global_step):
        raise ValueError("checkpoint schedule step must equal the global step")
    if best_metric is not None and not math.isfinite(float(best_metric)):
        raise ValueError("checkpoint best metric must be finite")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity.as_dict(),
        "config": config.as_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "data_state": None if data_state is None else dict(data_state),
        "rng": _rng_state(),
        "generators": {
            name: generator.get_state() for name, generator in sorted((generators or {}).items())
        },
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint_exact(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: WarmupCosineSchedule,
    config: ExperimentConfig,
    identity: CheckpointIdentity,
    generators: Mapping[str, torch.Generator] | None = None,
) -> RestoredTrainingState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("exact resume payload must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "exact resume requires a v4 mainline checkpoint with complete input identity"
        )
    raw_config = payload.get("config")
    raw_identity = payload.get("identity")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_identity, Mapping):
        raise ValueError("exact resume checkpoint has no typed config/identity")
    saved_config = config_from_mapping(raw_config)
    saved_identity = checkpoint_identity_from_mapping(
        raw_identity,
        require_current_manifest=False,
    )
    report = compare_checkpoint_identity(saved_identity, identity)
    if not report.exact_resume:
        raise ValueError("exact resume rejected: " + "; ".join(report.reasons))
    if saved_config.digest(include_paths=False) != config.digest(include_paths=False):
        raise ValueError("exact resume config differs from the current graph")

    # Validate every ownership boundary before mutating the live training
    # objects.  A malformed generator or optimizer mapping must not leave a
    # half-restored model behind.
    saved_model = payload.get("model")
    saved_optimizer = payload.get("optimizer")
    saved_schedule = payload.get("schedule")
    saved_rng = payload.get("rng")
    if not isinstance(saved_model, Mapping):
        raise ValueError("exact resume checkpoint has no model state mapping")
    current_model = model.state_dict()
    if set(saved_model) != set(current_model):
        raise ValueError("exact resume model parameter ownership differs")
    for name, current_value in current_model.items():
        saved_value = saved_model[name]
        if not isinstance(saved_value, torch.Tensor):
            raise ValueError(f"model state {name!r} is not a tensor")
        if tuple(saved_value.shape) != tuple(current_value.shape):
            raise ValueError(f"model state {name!r} has an incompatible shape")
        if saved_value.dtype != current_value.dtype:
            raise ValueError(f"model state {name!r} has an incompatible dtype")
        if (saved_value.is_floating_point() or saved_value.is_complex()) and not bool(
            torch.isfinite(saved_value).all()
        ):
            raise ValueError(f"model state {name!r} is non-finite")
    if not isinstance(saved_optimizer, Mapping):
        raise ValueError("exact resume checkpoint has no optimizer state mapping")
    current_optimizer = optimizer.state_dict()
    saved_groups = saved_optimizer.get("param_groups")
    current_groups = current_optimizer.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise ValueError("exact resume optimizer groups are invalid")
    for group in [*saved_groups, *current_groups]:
        if (
            not isinstance(group, Mapping)
            or not isinstance(
                group.get("params"),
                (tuple, list),
            )
            or (
                "parameter_names" in group
                and not isinstance(group.get("parameter_names"), (tuple, list))
            )
        ):
            raise ValueError("exact resume optimizer group ownership is invalid")
    saved_group_signature = tuple(
        (
            group.get("name"),
            tuple(group.get("parameter_names", ())),
            len(group.get("params", ())),
        )
        for group in saved_groups
        if isinstance(group, Mapping)
    )
    current_group_signature = tuple(
        (
            group.get("name"),
            tuple(group.get("parameter_names", ())),
            len(group.get("params", ())),
        )
        for group in current_groups
        if isinstance(group, Mapping)
    )
    if len(saved_group_signature) != len(saved_groups) or (
        saved_group_signature != current_group_signature
    ):
        raise ValueError("exact resume optimizer ownership differs")
    _validate_optimizer_state(
        saved_optimizer,
        optimizer=optimizer,
        saved_groups=saved_groups,
        current_groups=current_groups,
    )
    if not isinstance(saved_schedule, Mapping):
        raise ValueError("exact resume checkpoint has no schedule state mapping")
    base_lrs = saved_schedule.get("base_lrs")
    if not isinstance(base_lrs, (tuple, list)) or len(base_lrs) != len(saved_groups):
        raise ValueError("exact resume schedule group ownership differs")
    try:
        restored_schedule_step = int(saved_schedule["step_index"])
        restored_base_lrs = tuple(float(value) for value in base_lrs)
        restored_epoch = int(payload["epoch"])
        restored_step = int(payload["global_step"])
        restored_best = (
            None if payload.get("best_metric") is None else float(payload["best_metric"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("exact resume scalar training state is invalid") from error
    if (
        restored_epoch < 0
        or restored_step < 0
        or restored_schedule_step < 0
        or restored_schedule_step != restored_step
    ):
        raise ValueError("exact resume schedule/global step ownership is inconsistent")
    if not all(math.isfinite(value) and value >= 0.0 for value in restored_base_lrs):
        raise ValueError("exact resume schedule base learning rates are invalid")
    live_base_lrs = tuple(float(value) for value in schedule.base_lrs)
    if len(live_base_lrs) != len(restored_base_lrs) or any(
        not math.isclose(saved, live, rel_tol=0.0, abs_tol=0.0)
        for saved, live in zip(restored_base_lrs, live_base_lrs, strict=True)
    ):
        raise ValueError("exact resume schedule base learning rates differ")
    expected_lr_ratio = schedule.ratio(restored_schedule_step)
    for index, (group, base_lr) in enumerate(zip(saved_groups, restored_base_lrs, strict=True)):
        if not isinstance(group, Mapping):
            raise ValueError("exact resume optimizer group is not a mapping")
        try:
            saved_lr = float(group["lr"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("exact resume optimizer learning rate is invalid") from error
        expected_lr = base_lr * expected_lr_ratio
        if not math.isfinite(saved_lr) or not math.isclose(
            saved_lr,
            expected_lr,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"exact resume optimizer group {index} learning rate differs from schedule"
            )
    if restored_best is not None and not math.isfinite(restored_best):
        raise ValueError("exact resume best metric is non-finite")
    if not isinstance(saved_rng, Mapping):
        raise ValueError("exact resume checkpoint has no global RNG state")
    requested_generators = generators or {}
    saved_generators = payload.get("generators")
    if not isinstance(saved_generators, Mapping):
        raise ValueError("exact resume checkpoint has no generator-state mapping")
    if set(saved_generators) != set(requested_generators):
        raise ValueError("exact resume generator ownership differs from the checkpoint")
    _validate_rng_state(saved_rng, requested_generators, saved_generators)

    model.load_state_dict(saved_model, strict=True)
    optimizer.load_state_dict(cast(dict[str, Any], dict(saved_optimizer)))
    schedule.load_state_dict(dict(saved_schedule))
    for name, generator in requested_generators.items():
        generator.set_state(cast(torch.Tensor, saved_generators[name]))
    _restore_rng(saved_rng)
    return RestoredTrainingState(
        epoch=restored_epoch,
        global_step=restored_step,
        best_metric=restored_best,
    )


def migrate_bottom_only(
    path: str | Path,
    model: nn.Module,
    *,
    identity: CheckpointIdentity,
) -> MigrationReport:
    """Load a verified, ABI-compatible mainline bottom with a full report."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "bottom migration only accepts checkpoint schema v4 with an "
            "ABI-compatible complete typed bottom"
        )
    raw_identity = payload.get("identity")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("bottom migration source has no serialized architecture identity")
    saved_identity = checkpoint_identity_from_mapping(
        raw_identity,
        require_current_manifest=False,
    )
    compatibility = compare_checkpoint_identity(saved_identity, identity)
    if "bottom" not in compatibility.reusable_components:
        raise ValueError("bottom migration rejected: " + "; ".join(compatibility.reasons))
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("migration source has no model state mapping")
    current = model.state_dict()
    selected: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    dtype_mismatch: list[str] = []
    complex_bottom: list[str] = []
    nonfinite_bottom: list[str] = []
    rejected: list[str] = []
    invalid_bottom: list[str] = []
    for raw_name, value in state.items():
        name = str(raw_name)
        if not name.startswith("bottom."):
            rejected.append(name)
            continue
        if name not in current or not isinstance(value, torch.Tensor):
            invalid_bottom.append(name)
        elif tuple(value.shape) != tuple(current[name].shape):
            shape_mismatch.append(name)
        elif value.dtype != current[name].dtype:
            # ``load_state_dict`` silently casts a source tensor to the live
            # parameter dtype.  That is useful for generic transfer learning,
            # but violates this migration's claim that the serialized bottom
            # ABI is *identical* and can also conceal a malformed checkpoint.
            dtype_mismatch.append(name)
        elif value.is_complex() or current[name].is_complex():
            # The active bottom has no complex-valued state.  Keep this an
            # explicit boundary rather than allowing a future matching complex
            # dtype to enter unnoticed.
            complex_bottom.append(name)
        elif value.is_floating_point() and not bool(torch.isfinite(value).all()):
            nonfinite_bottom.append(name)
        else:
            selected[name] = value
    missing = sorted(
        name for name in current if name.startswith("bottom.") and name not in selected
    )
    # ABI compatibility is all-or-nothing.  A partial shape-only load would
    # silently create a third bottom that is neither fresh nor the source
    # checkpoint, defeating the explicit migration boundary.  Validate the
    # complete state before touching the live model.
    if (
        missing
        or shape_mismatch
        or dtype_mismatch
        or complex_bottom
        or nonfinite_bottom
        or invalid_bottom
    ):
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if shape_mismatch:
            details.append(f"shape_mismatch={len(shape_mismatch)}")
        if dtype_mismatch:
            details.append(f"dtype_mismatch={len(dtype_mismatch)}")
        if complex_bottom:
            details.append(f"complex_bottom={len(complex_bottom)}")
        if nonfinite_bottom:
            details.append(f"nonfinite_bottom={len(nonfinite_bottom)}")
        if invalid_bottom:
            details.append(f"invalid_bottom={len(invalid_bottom)}")
        raise ValueError("bottom migration state is incomplete: " + ", ".join(details))
    model.load_state_dict(selected, strict=False)
    return MigrationReport(
        loaded=tuple(sorted(selected)),
        missing=tuple(missing),
        shape_mismatch=tuple(sorted(shape_mismatch)),
        rejected=tuple(sorted(rejected)),
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "MigrationReport",
    "RestoredTrainingState",
    "load_checkpoint_exact",
    "migrate_bottom_only",
    "save_checkpoint",
]
