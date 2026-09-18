"""Atomic mainline checkpoints with exact-resume and explicit migration."""

from __future__ import annotations

import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
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
from ..manifest import LAYOUT_SCHEMA, ArchitectureManifest, manifest_from_mapping
from ..model.component_contracts import (
    ComponentSelection,
    map_legacy_state_dict,
    modular_to_legacy_name,
)
from ..training.optimizer import WarmupCosineSchedule

CHECKPOINT_SCHEMA = "clearvla-mainline-checkpoint-v4"
LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION = (
    "libero_window_boundary_supervision_v1"
)
LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION = (
    "libero_retarget_training_overlay_v1"
)
LIBERO_RELEASE_FIRST_REPAIR_MIGRATION = "libero_release_first_repair_v1"
WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION = (
    "world_camera_coordinate_role_v1"
)
VALIDATION_REPLAY_SOURCE_PATHS = frozenset(
    {
        "clearvla/mainline/model/compiler.py",
        "clearvla/mainline/model/transition.py",
        "clearvla/mainline/runtime/checkpoints.py",
        "clearvla/mainline/runtime/evaluation.py",
        "clearvla/mainline/runtime/logging.py",
        "clearvla/mainline/train.py",
    }
)
INITIALIZATION_SOURCE_PATHS = VALIDATION_REPLAY_SOURCE_PATHS
DATA_CONTRACT_MIGRATION_SOURCE_PATHS = INITIALIZATION_SOURCE_PATHS | frozenset(
    {
        "clearvla/benchmarks/common.py",
        "clearvla/data/hdf5_episode.py",
        "clearvla/data/libero_retarget.py",
        "clearvla/data/samplers.py",
        "clearvla/data/window_boundaries.py",
        "clearvla/mainline/config.py",
        "clearvla/mainline/data/dataset.py",
        "clearvla/mainline/data/loading.py",
        "clearvla/mainline/training/losses.py",
        "clearvla/mainline/runtime/checkpoints.py",
        "clearvla/mainline/train.py",
    }
)
WORLD_CAMERA_COORDINATE_ROLE_V1_SOURCE_PATHS = frozenset(
    {
        "clearvla/mainline/config.py",
        "clearvla/mainline/model/dynamics.py",
        "clearvla/mainline/model/policy.py",
        "clearvla/mainline/model/top.py",
        "clearvla/mainline/runtime/checkpoints.py",
        "clearvla/mainline/train.py",
    }
)
LAYOUT_MIGRATION_REPLAY_SOURCE_PATHS = frozenset(
    {
        "clearvla/mainline/checkpoint.py",
        "clearvla/mainline/manifest.py",
        "clearvla/mainline/model/__init__.py",
        "clearvla/mainline/model/component_contracts.py",
        "clearvla/mainline/model/components.py",
        "clearvla/mainline/model/policy.py",
        "clearvla/mainline/runtime/checkpoints.py",
        "clearvla/mainline/runtime/evaluation.py",
        "clearvla/mainline/runtime/sampling.py",
        "clearvla/mainline/train.py",
        "clearvla/mainline/training/engine.py",
        "clearvla/mainline/training/optimizer.py",
        "clearvla/mainline/v120_core/time_domain_mmdit.py",
    }
)


@dataclass(frozen=True)
class RestoredTrainingState:
    epoch: int
    global_step: int
    best_metric: float | None


@dataclass(frozen=True)
class ValidationReplayState:
    epoch: int
    global_step: int
    best_metric: float | None
    saved_source_digest: str
    current_source_digest: str
    changed_source_files: tuple[str, ...]


@dataclass(frozen=True)
class InitializationState:
    """Metadata for a model-only checkpoint initialization.

    Unlike exact resume, initialization deliberately discards optimizer,
    scheduler, RNG and data-loader continuation state.  The metadata is
    retained in the new run context so that the continuation is auditable.
    """

    epoch: int
    global_step: int
    best_metric: float | None
    saved_source_digest: str
    current_source_digest: str
    changed_source_files: tuple[str, ...]
    data_contract_migration: str | None
    model_contract_migration: str | None
    saved_window_boundary_contract: str
    current_window_boundary_contract: str
    saved_dataset_identity: dict[str, str]
    current_dataset_identity: dict[str, str]
    normalizer_identity_equal: bool


@dataclass(frozen=True)
class MigrationReport:
    loaded: tuple[str, ...]
    missing: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    dtype_mismatch: tuple[str, ...]
    rejected: tuple[str, ...]


def _identity_manifest(identity: CheckpointIdentity) -> ArchitectureManifest:
    return manifest_from_mapping(
        identity.manifest,
        require_current_schema=False,
    )


def _resolved_component_selection(
    model: nn.Module,
    config: ExperimentConfig,
) -> ComponentSelection:
    expected = ComponentSelection.from_config(config)
    live = getattr(model, "selection", None)
    if live is not None and live != expected:
        raise ValueError("live model component selection differs from the config")
    return expected


def _checkpoint_component_selection(
    payload: Mapping[str, object],
    *,
    config: ExperimentConfig,
    manifest: ArchitectureManifest,
) -> ComponentSelection:
    raw = payload.get("component_selection")
    if raw is None and int(manifest.layout_schema) != LAYOUT_SCHEMA:
        # Pre-modular checkpoints did not serialize this object.  The frozen
        # config still resolves its only legal baseline selection.
        return ComponentSelection.from_config(config)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no complete component selection")
    return ComponentSelection.from_mapping(raw, config=config)


def _layout_only_manifest_difference(
    saved: ArchitectureManifest,
    current: ArchitectureManifest,
) -> bool:
    saved_value = saved.as_dict()
    saved_value["layout_schema"] = int(current.layout_schema)
    return saved_value == current.as_dict()


def _state_for_current_layout(
    saved: Mapping[str, object],
    *,
    model: nn.Module,
    saved_layout_schema: int,
) -> Mapping[str, Tensor]:
    if int(saved_layout_schema) == LAYOUT_SCHEMA:
        return cast(Mapping[str, Tensor], saved)
    try:
        return map_legacy_state_dict(model, cast(Mapping[str, Tensor], saved))
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("legacy checkpoint state cannot map to the modular layout") from error


def _legacy_bottom_name(name: str) -> str:
    try:
        return modular_to_legacy_name(name)
    except KeyError:
        # Keep this helper usable by small checkpoint contract fixtures that
        # model a literal ``bottom`` without constructing the 168M graph.
        return name


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
    component_selection = _resolved_component_selection(model, config)
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
        "component_selection": component_selection.as_dict(),
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
    saved_manifest = _identity_manifest(saved_identity)
    if int(saved_manifest.layout_schema) != LAYOUT_SCHEMA:
        raise ValueError(
            "exact resume rejects the pre-modular model layout; use read-only "
            "validation or explicit migration"
        )
    saved_selection = _checkpoint_component_selection(
        payload,
        config=saved_config,
        manifest=saved_manifest,
    )
    current_selection = _resolved_component_selection(model, config)
    if saved_selection != current_selection:
        raise ValueError("exact resume component selection differs")
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


def load_checkpoint_for_validation(
    path: str | Path,
    *,
    model: nn.Module,
    config: ExperimentConfig,
    identity: CheckpointIdentity,
) -> ValidationReplayState:
    """Load a complete model for read-only validation after observation edits.

    Exact resume deliberately rejects every active-source change.  A diagnostic
    replay has a narrower exception: the manifest, resolved semantic config,
    dataset, language artifact and complete model state ABI must still match,
    while the active-source digest may differ.  Optimizer, schedule and RNG
    state are never loaded, so this entry point cannot resume training or
    mutate the serialized checkpoint.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "validation replay requires a v4 mainline checkpoint with complete input identity"
        )
    raw_config = payload.get("config")
    raw_identity = payload.get("identity")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_identity, Mapping):
        raise ValueError("validation replay checkpoint has no typed config/identity")
    saved_config = config_from_mapping(raw_config)
    saved_identity = checkpoint_identity_from_mapping(
        raw_identity,
        require_current_manifest=False,
    )
    saved_manifest = _identity_manifest(saved_identity)
    current_manifest = _identity_manifest(identity)
    saved_selection = _checkpoint_component_selection(
        payload,
        config=saved_config,
        manifest=saved_manifest,
    )
    current_selection = _resolved_component_selection(model, config)
    if saved_selection != current_selection:
        raise ValueError("validation replay component selection differs")
    report = compare_checkpoint_identity(saved_identity, identity)
    legacy_layout_only = (
        int(saved_manifest.layout_schema) != LAYOUT_SCHEMA
        and _layout_only_manifest_difference(saved_manifest, current_manifest)
    )
    allowed_identity_reasons = {"source identity differs"}
    if legacy_layout_only:
        allowed_identity_reasons.add("manifest identity differs")
    rejected_reasons = tuple(
        reason for reason in report.reasons if reason not in allowed_identity_reasons
    )
    if rejected_reasons:
        raise ValueError("validation replay rejected: " + "; ".join(rejected_reasons))
    if saved_config.digest(include_paths=False) != config.digest(include_paths=False):
        raise ValueError("validation replay config differs from the checkpoint graph")
    saved_sources = dict(saved_identity.source.files)
    current_sources = dict(identity.source.files)
    changed_source_files = tuple(
        sorted(
            path
            for path in set(saved_sources).union(current_sources)
            if saved_sources.get(path) != current_sources.get(path)
        )
    )
    allowed_source_paths = VALIDATION_REPLAY_SOURCE_PATHS
    if legacy_layout_only:
        allowed_source_paths = allowed_source_paths.union(
            LAYOUT_MIGRATION_REPLAY_SOURCE_PATHS
        )
    unexpected_source_files = tuple(
        path
        for path in changed_source_files
        if path not in allowed_source_paths
    )
    if unexpected_source_files:
        raise ValueError(
            "validation replay source drift escapes the validation-only allow-list: "
            + ", ".join(unexpected_source_files)
        )

    saved_model = payload.get("model")
    if not isinstance(saved_model, Mapping):
        raise ValueError("validation replay checkpoint has no model state mapping")
    mapped_model = _state_for_current_layout(
        saved_model,
        model=model,
        saved_layout_schema=int(saved_manifest.layout_schema),
    )
    current_model = model.state_dict()
    if set(mapped_model) != set(current_model):
        raise ValueError("validation replay model parameter ownership differs")
    for name, current_value in current_model.items():
        saved_value = mapped_model[name]
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
    try:
        restored_epoch = int(payload["epoch"])
        restored_step = int(payload["global_step"])
        restored_best = (
            None if payload.get("best_metric") is None else float(payload["best_metric"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("validation replay scalar checkpoint state is invalid") from error
    if restored_epoch < 0 or restored_step < 0:
        raise ValueError("validation replay epoch/global step must be non-negative")
    if restored_best is not None and not math.isfinite(restored_best):
        raise ValueError("validation replay best metric is non-finite")

    model.load_state_dict(mapped_model, strict=True)
    return ValidationReplayState(
        epoch=restored_epoch,
        global_step=restored_step,
        best_metric=restored_best,
        saved_source_digest=saved_identity.source.digest,
        current_source_digest=identity.source.digest,
        changed_source_files=changed_source_files,
    )


def _initialization_config_view(config: ExperimentConfig) -> dict[str, object]:
    """Return the semantic config retained when starting a fresh optimizer.

    A model-only continuation is allowed to choose a new optimizer horizon,
    learning rate, batch size and other optimizer controls.  The model,
    objective, data contract and deployment/runtime semantics remain matched;
    only loader/logging limits are ignored as operational controls.
    Paths are already non-semantic for the checkpoint identity and are removed
    here for the same relocation rule used by ``ExperimentConfig.digest``.
    """

    payload = config.as_dict()
    data = dict(cast(Mapping[str, object], payload["data"]))
    for name in (
        "raw_hdf5_root",
        "decoded_cache",
        "dino_cache",
        "t5_condition",
        "output_dir",
        "split_manifest",
        "calvin_raw_source",
        "task_selection_manifest",
        "normalizer_artifact",
        "num_workers",
    ):
        data.pop(name, None)
    payload["data"] = data
    runtime = dict(cast(Mapping[str, object], payload["runtime"]))
    for name in (
        "log_every",
        "max_train_batches",
        "max_val_batches",
        "eval_sampling_diagnostic_batches",
        "eval_proposal_ablation_batches",
        "eval_execution_ablation_batches",
    ):
        runtime.pop(name, None)
    payload["runtime"] = runtime
    # The new run constructs a fresh optimizer and schedule by contract.
    payload.pop("optimizer", None)
    return payload


def _libero_boundary_migration_config_view(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Remove only the admitted LIBERO window-center contract difference."""

    payload = _initialization_config_view(config)
    data = dict(cast(Mapping[str, object], payload["data"]))
    data.pop("window_boundary_contract", None)
    payload["data"] = data
    return payload


def _libero_retarget_migration_config_view(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Remove only the train-only retarget overlay location contract."""

    payload = _initialization_config_view(config)
    data = dict(cast(Mapping[str, object], payload["data"]))
    data.pop("libero_retarget_overlay_root", None)
    data.pop("libero_retarget_overlay_manifest", None)
    payload["data"] = data
    return payload


def _libero_release_repair_config_view(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Remove only the two opt-in release-first repair controls."""

    payload = _initialization_config_view(config)
    data = dict(cast(Mapping[str, object], payload["data"]))
    data.pop("release_first_action_fraction", None)
    payload["data"] = data
    objectives = dict(cast(Mapping[str, object], payload["objectives"]))
    objectives.pop("gripper_first_step_release", None)
    payload["objectives"] = objectives
    return payload


def _world_camera_condition_migration_config_view(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Remove only the admitted opt-in W camera-condition selector."""

    payload = _initialization_config_view(config)
    top = dict(cast(Mapping[str, object], payload["top"]))
    top.pop("world_camera_condition_mode", None)
    payload["top"] = top
    return payload


def load_checkpoint_for_initialization(
    path: str | Path,
    *,
    model: nn.Module,
    config: ExperimentConfig,
    identity: CheckpointIdentity,
    data_contract_migration: str | None = None,
    model_contract_migration: str | None = None,
) -> InitializationState:
    """Load only model parameters as the start of a fresh training run.

    This is intentionally distinct from exact resume and validation replay:
    optimizer, scheduler, RNG and data-loader state are never read or
    restored.  Manifest, model selection, objective/data semantics,
    normalizers and language identity must still match.  Only the narrow
    validation/trainer source allow-list may differ, so an architecture edit
    cannot be hidden behind a continuation label.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "model initialization requires a v4 mainline checkpoint with complete identity"
        )
    raw_config = payload.get("config")
    raw_identity = payload.get("identity")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_identity, Mapping):
        raise ValueError("model initialization checkpoint has no typed config/identity")
    saved_config = config_from_mapping(raw_config)
    saved_identity = checkpoint_identity_from_mapping(
        raw_identity,
        require_current_manifest=False,
    )
    saved_manifest = _identity_manifest(saved_identity)
    if int(saved_manifest.layout_schema) != LAYOUT_SCHEMA:
        raise ValueError(
            "model initialization rejects the pre-modular model layout; use explicit migration"
        )
    saved_selection = _checkpoint_component_selection(
        payload,
        config=saved_config,
        manifest=saved_manifest,
    )
    current_selection = _resolved_component_selection(model, config)
    if saved_selection != current_selection:
        raise ValueError("model initialization component selection differs")

    selected_migration = (
        None
        if data_contract_migration is None
        else str(data_contract_migration).strip()
    )
    selected_model_migration = (
        None
        if model_contract_migration is None
        else str(model_contract_migration).strip()
    )
    if selected_migration is not None and selected_model_migration is not None:
        raise ValueError(
            "data-contract and model-contract initialization migrations "
            "cannot be combined"
        )
    if selected_migration not in {
        None,
        LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION,
        LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION,
        LIBERO_RELEASE_FIRST_REPAIR_MIGRATION,
    }:
        raise ValueError(
            f"unknown model-initialization data migration {selected_migration!r}"
        )
    if selected_model_migration not in {
        None,
        WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION,
    }:
        raise ValueError(
            "unknown model-initialization model migration "
            f"{selected_model_migration!r}"
        )

    report = compare_checkpoint_identity(saved_identity, identity)
    admitted_identity_reasons = {
        "config identity differs",
        "source identity differs",
    }
    if selected_migration is not None:
        admitted_identity_reasons.add("dataset identity differs")
    rejected_reasons = tuple(
        reason
        for reason in report.reasons
        if reason not in admitted_identity_reasons
    )
    if rejected_reasons:
        raise ValueError("model initialization rejected: " + "; ".join(rejected_reasons))
    if selected_model_migration == WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION:
        if (
            saved_config.top.world_camera_condition_mode != "motion_prior_only"
            or config.top.world_camera_condition_mode != "coordinate_role_v1"
        ):
            raise ValueError(
                "W camera-condition migration requires motion_prior_only source "
                "and coordinate_role_v1 target"
            )
        if tuple(saved_config.data.camera_names) != tuple(config.data.camera_names):
            raise ValueError(
                "W camera-condition migration requires the identical declared "
                "camera role order"
            )
        if _world_camera_condition_migration_config_view(
            saved_config
        ) != _world_camera_condition_migration_config_view(config):
            raise ValueError(
                "W camera-condition migration differs outside its one model selector"
            )
        if saved_identity.dataset != identity.dataset:
            raise ValueError(
                "W camera-condition migration requires identical dataset identity"
            )
    elif selected_migration is None:
        if _initialization_config_view(saved_config) != _initialization_config_view(config):
            raise ValueError(
                "model initialization config differs in model/data/objective/runtime semantics"
            )
    elif selected_migration == LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION:
        from clearvla.data.window_boundaries import (
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
            CAUSAL_PREFIX_V1,
            STRICT_COMPLETE_V1,
        )

        if (
            saved_config.data.data_profile != "libero_relative_7d_v1"
            or config.data.data_profile != "libero_relative_7d_v1"
        ):
            raise ValueError("LIBERO boundary migration requires the LIBERO outlet")
        if saved_config.data.window_boundary_contract != STRICT_COMPLETE_V1:
            raise ValueError("LIBERO boundary migration source must use strict_complete_v1")
        if config.data.window_boundary_contract not in {
            CAUSAL_PREFIX_V1,
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        }:
            raise ValueError(
                "LIBERO boundary migration target must be causal_prefix_v1 or "
                "causal_prefix_terminal_suffix_v2"
            )
        if _libero_boundary_migration_config_view(
            saved_config
        ) != _libero_boundary_migration_config_view(config):
            raise ValueError(
                "LIBERO boundary migration differs outside the admitted data-window contract"
            )
        if (
            saved_identity.dataset.action_normalizer_sha256
            != identity.dataset.action_normalizer_sha256
            or saved_identity.dataset.state_normalizer_sha256
            != identity.dataset.state_normalizer_sha256
        ):
            raise ValueError(
                "LIBERO boundary migration requires identical action/state normalizers"
            )
    elif selected_migration == LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION:
        from clearvla.data.window_boundaries import (
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        )

        if selected_migration != LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION:
            raise AssertionError("unreachable data-contract migration")
        if (
            saved_config.data.data_profile != "libero_relative_7d_v1"
            or config.data.data_profile != "libero_relative_7d_v1"
        ):
            raise ValueError("LIBERO retarget migration requires the LIBERO outlet")
        if (
            saved_config.data.window_boundary_contract
            != CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
            or config.data.window_boundary_contract
            != CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
        ):
            raise ValueError(
                "LIBERO retarget migration requires terminal-suffix source and target"
            )
        if (
            saved_config.data.libero_retarget_overlay_root.strip()
            or saved_config.data.libero_retarget_overlay_manifest.strip()
        ):
            raise ValueError("LIBERO retarget migration source already has an overlay")
        if not (
            config.data.libero_retarget_overlay_root.strip()
            and config.data.libero_retarget_overlay_manifest.strip()
        ):
            raise ValueError("LIBERO retarget migration target has no complete overlay")
        if _libero_retarget_migration_config_view(
            saved_config
        ) != _libero_retarget_migration_config_view(config):
            raise ValueError(
                "LIBERO retarget migration differs outside the admitted training overlay"
            )
        if (
            saved_identity.dataset.action_normalizer_sha256
            != identity.dataset.action_normalizer_sha256
            or saved_identity.dataset.state_normalizer_sha256
            != identity.dataset.state_normalizer_sha256
        ):
            raise ValueError(
                "LIBERO retarget migration requires identical action/state normalizers"
            )
        if (
            saved_identity.dataset.raw_root != identity.dataset.raw_root
            or saved_identity.dataset.hdf5_glob != identity.dataset.hdf5_glob
        ):
            raise ValueError(
                "LIBERO retarget migration cannot replace the base dataset root/glob"
            )
        if saved_identity.dataset == identity.dataset:
            raise ValueError("LIBERO retarget migration did not change dataset identity")
    else:
        from clearvla.data.window_boundaries import (
            CAUSAL_PREFIX_TERMINAL_SUFFIX_V2,
        )

        if (
            saved_config.data.data_profile != "libero_relative_7d_v1"
            or config.data.data_profile != "libero_relative_7d_v1"
            or saved_config.data.window_boundary_contract
            != CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
            or config.data.window_boundary_contract
            != CAUSAL_PREFIX_TERMINAL_SUFFIX_V2
        ):
            raise ValueError(
                "LIBERO release-first repair requires terminal-suffix v2 on both sides"
            )
        if float(config.data.release_first_action_fraction) <= 0.0 and float(
            config.objectives.gripper_first_step_release
        ) <= 0.0:
            raise ValueError(
                "LIBERO release-first repair must enable a sampling or loss control"
            )
        if _libero_release_repair_config_view(
            saved_config
        ) != _libero_release_repair_config_view(config):
            raise ValueError(
                "LIBERO release-first repair differs outside its two explicit controls"
            )
        if saved_identity.dataset != identity.dataset:
            raise ValueError(
                "LIBERO release-first repair requires identical dataset identity"
            )

    saved_sources = dict(saved_identity.source.files)
    current_sources = dict(identity.source.files)
    changed_source_files = tuple(
        sorted(
            source_path
            for source_path in set(saved_sources).union(current_sources)
            if saved_sources.get(source_path) != current_sources.get(source_path)
        )
    )
    if selected_model_migration == WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION:
        allowed_source_paths = WORLD_CAMERA_COORDINATE_ROLE_V1_SOURCE_PATHS
    else:
        allowed_source_paths = (
            INITIALIZATION_SOURCE_PATHS
            if selected_migration is None
            else DATA_CONTRACT_MIGRATION_SOURCE_PATHS
        )
    unexpected_source_files = tuple(
        source_path
        for source_path in changed_source_files
        if source_path not in allowed_source_paths
    )
    if unexpected_source_files:
        raise ValueError(
            "model initialization source drift escapes the allow-list: "
            + ", ".join(unexpected_source_files)
        )

    saved_model = payload.get("model")
    if not isinstance(saved_model, Mapping):
        raise ValueError("model initialization checkpoint has no model state mapping")
    mapped_model = _state_for_current_layout(
        saved_model,
        model=model,
        saved_layout_schema=int(saved_manifest.layout_schema),
    )
    current_model = model.state_dict()
    if selected_model_migration == WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION:
        new_condition_key = (
            "world.dynamics.camera_coordinate_role_condition.weight"
        )
        missing = set(current_model) - set(mapped_model)
        unexpected = set(mapped_model) - set(current_model)
        if missing != {new_condition_key} or unexpected:
            raise ValueError(
                "W camera-condition migration must add exactly its one condition weight"
            )
        new_condition = current_model[new_condition_key]
        if not isinstance(new_condition, torch.Tensor) or int(
            torch.count_nonzero(new_condition).item()
        ) != 0:
            raise ValueError(
                "W camera-condition migration requires an exact-zero new condition weight"
            )
        mapped_model = dict(mapped_model)
        mapped_model[new_condition_key] = new_condition.detach().clone()
    elif set(mapped_model) != set(current_model):
        raise ValueError("model initialization parameter ownership differs")
    for name, current_value in current_model.items():
        saved_value = mapped_model[name]
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
    try:
        restored_epoch = int(payload["epoch"])
        restored_step = int(payload["global_step"])
        restored_best = (
            None if payload.get("best_metric") is None else float(payload["best_metric"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model initialization checkpoint scalar state is invalid") from error
    if restored_epoch < 0 or restored_step < 0:
        raise ValueError("model initialization epoch/global step must be non-negative")
    if restored_best is not None and not math.isfinite(restored_best):
        raise ValueError("model initialization best metric is non-finite")

    model.load_state_dict(mapped_model, strict=True)
    return InitializationState(
        epoch=restored_epoch,
        global_step=restored_step,
        best_metric=restored_best,
        saved_source_digest=saved_identity.source.digest,
        current_source_digest=identity.source.digest,
        changed_source_files=changed_source_files,
        data_contract_migration=selected_migration,
        model_contract_migration=selected_model_migration,
        saved_window_boundary_contract=saved_config.data.window_boundary_contract,
        current_window_boundary_contract=config.data.window_boundary_contract,
        saved_dataset_identity=dict(asdict(saved_identity.dataset)),
        current_dataset_identity=dict(asdict(identity.dataset)),
        normalizer_identity_equal=bool(
            saved_identity.dataset.action_normalizer_sha256
            == identity.dataset.action_normalizer_sha256
            and saved_identity.dataset.state_normalizer_sha256
            == identity.dataset.state_normalizer_sha256
        ),
    )


def migrate_bottom_only(
    path: str | Path,
    model: nn.Module,
    *,
    identity: CheckpointIdentity,
    config: ExperimentConfig | None = None,
) -> MigrationReport:
    """Load a verified, ABI-compatible mainline bottom with a full report."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "bottom migration only accepts checkpoint schema v4 with an "
            "ABI-compatible complete typed bottom"
        )
    raw_identity = payload.get("identity")
    raw_config = payload.get("config")
    if not isinstance(raw_identity, Mapping):
        raise ValueError("bottom migration source has no serialized architecture identity")
    saved_config = config_from_mapping(raw_config) if isinstance(raw_config, Mapping) else None
    saved_identity = checkpoint_identity_from_mapping(
        raw_identity,
        require_current_manifest=False,
    )
    saved_manifest = _identity_manifest(saved_identity)
    saved_selection = (
        _checkpoint_component_selection(
            payload,
            config=saved_config,
            manifest=saved_manifest,
        )
        if saved_config is not None
        else None
    )
    live_selection = getattr(model, "selection", None)
    if config is not None:
        current_selection = _resolved_component_selection(model, config)
    elif isinstance(live_selection, ComponentSelection):
        current_selection = live_selection
    else:
        current_selection = None
    if saved_selection is not None and current_selection is not None and (
        saved_selection.execution_bottom != current_selection.execution_bottom
        or saved_selection.terminal_controller != current_selection.terminal_controller
    ):
        raise ValueError("bottom migration component selection is incompatible")
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
    rejected: list[str] = []
    invalid_bottom: list[str] = []
    source_by_legacy_name: dict[str, tuple[str, object]] = {}
    for raw_name, value in state.items():
        name = str(raw_name)
        legacy_name = _legacy_bottom_name(name)
        if not legacy_name.startswith("bottom."):
            rejected.append(name)
            continue
        if legacy_name in source_by_legacy_name:
            invalid_bottom.append(legacy_name)
        else:
            source_by_legacy_name[legacy_name] = (name, value)

    bottom_destinations = {
        name: _legacy_bottom_name(name)
        for name in current
        if _legacy_bottom_name(name).startswith("bottom.")
    }
    missing: list[str] = []
    for destination_name, legacy_name in bottom_destinations.items():
        source_row = source_by_legacy_name.get(legacy_name)
        if source_row is None:
            missing.append(destination_name)
            continue
        source_name, value = source_row
        target = current[destination_name]
        if not isinstance(value, torch.Tensor):
            invalid_bottom.append(source_name)
        elif tuple(value.shape) != tuple(target.shape):
            shape_mismatch.append(source_name)
        elif value.dtype != target.dtype:
            dtype_mismatch.append(source_name)
        elif (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            invalid_bottom.append(source_name)
        else:
            selected[destination_name] = value
    # ABI compatibility is all-or-nothing.  A partial shape-only load would
    # silently create a third bottom that is neither fresh nor the source
    # checkpoint, defeating the explicit migration boundary.  Validate the
    # complete state before touching the live model.
    if missing or shape_mismatch or dtype_mismatch or invalid_bottom:
        details = []
        if missing:
            details.append(f"missing={len(missing)}")
        if shape_mismatch:
            details.append(f"shape_mismatch={len(shape_mismatch)}")
        if dtype_mismatch:
            details.append(f"dtype_mismatch={len(dtype_mismatch)}")
        if invalid_bottom:
            details.append(f"invalid_bottom={len(invalid_bottom)}")
        raise ValueError("bottom migration state is incomplete: " + ", ".join(details))
    model.load_state_dict(selected, strict=False)
    return MigrationReport(
        loaded=tuple(sorted(selected)),
        missing=tuple(missing),
        shape_mismatch=tuple(sorted(shape_mismatch)),
        dtype_mismatch=tuple(sorted(dtype_mismatch)),
        rejected=tuple(sorted(rejected)),
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DATA_CONTRACT_MIGRATION_SOURCE_PATHS",
    "INITIALIZATION_SOURCE_PATHS",
    "InitializationState",
    "LIBERO_RETARGET_TRAINING_OVERLAY_MIGRATION",
    "LIBERO_RELEASE_FIRST_REPAIR_MIGRATION",
    "LIBERO_WINDOW_BOUNDARY_SUPERVISION_MIGRATION",
    "LAYOUT_MIGRATION_REPLAY_SOURCE_PATHS",
    "MigrationReport",
    "RestoredTrainingState",
    "VALIDATION_REPLAY_SOURCE_PATHS",
    "ValidationReplayState",
    "WORLD_CAMERA_COORDINATE_ROLE_V1_MIGRATION",
    "WORLD_CAMERA_COORDINATE_ROLE_V1_SOURCE_PATHS",
    "load_checkpoint_exact",
    "load_checkpoint_for_initialization",
    "load_checkpoint_for_validation",
    "migrate_bottom_only",
    "save_checkpoint",
]
