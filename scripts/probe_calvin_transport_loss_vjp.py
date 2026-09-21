#!/usr/bin/env python3
"""Compare frozen CALVIN transport-loss directions on disjoint validation episodes.

The probe never creates an optimizer or changes model state.  It reproduces
the *actual* mainline training-engine forward (one ``encode_online`` followed
by one formal ``velocity`` call, including the registered refined Q5 schedule
and ``FlowStepContext``), captures the final W2 ``FutureObjectDynamics``
tensors, and compares the active raw transport objective with graph-preserving
normalized and direction-only diagnostic objectives.  Cross-batch gradient
projections distinguish a new direction from a scalar amplification of the
existing loss.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.types import FutureObjectDynamics
from clearvla.mainline.runtime.flow_schedule import DeploymentFlowSchedule
from clearvla.mainline.runtime.identity import dataset_identity, language_identity
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.training import losses as losses_module
from clearvla.mainline.training.engine import _autocast
from clearvla.mainline.training.losses import compose_losses, sample_flow_matching
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

SCHEMA = "clearvla-calvin-transport-loss-vjp-v1"
PRIMARY_OWNERS = ("transport_head", "object_geometry", "camera_role")


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value) % (2**32))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _rng_snapshot() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_numpy = left["numpy"]
    right_numpy = right["numpy"]
    if not (
        left_numpy[0] == right_numpy[0]
        and np.array_equal(left_numpy[1], right_numpy[1])
        and left_numpy[2:] == right_numpy[2:]
    ):
        return False
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"], strict=True)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _owned_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _tensor_bytes(value: Tensor) -> bytes:
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().tobytes()
    return tensor.numpy().tobytes()


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _nonpersistent_buffer_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_buffers(remove_duplicate=False)):
        if name in model.state_dict():
            continue
        digest.update(name.encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _training_flags(model: nn.Module) -> list[bool]:
    return [bool(module.training) for module in model.modules()]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _module_origins(repo_root: Path, model: nn.Module) -> dict[str, dict[str, str]]:
    root = repo_root.expanduser().resolve()
    objects: dict[str, tuple[Any, str]] = {
        "checkpoint_policy": (
            ClearVLACheckpointPolicy,
            "clearvla/simulation/clearvla_policy.py",
        ),
        "training_engine_autocast": (
            _autocast,
            "clearvla/mainline/training/engine.py",
        ),
        "training_losses": (
            losses_module,
            "clearvla/mainline/training/losses.py",
        ),
        "policy_model": (
            model.__class__,
            "clearvla/mainline/model/policy.py",
        ),
        "world_dynamics": (
            model.world.dynamics.__class__,
            "clearvla/mainline/model/dynamics.py",
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (value, expected_relative) in objects.items():
        source = Path(inspect.getfile(value)).resolve()
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"executed module {name!r} is outside --repo-root: {source} vs {root}"
            ) from error
        if relative.as_posix() != expected_relative:
            raise RuntimeError(
                f"executed module {name!r} resolved to the wrong file inside --repo-root: "
                f"expected={expected_relative} actual={relative.as_posix()}"
            )
        result[name] = {
            "path": str(source),
            "relative_path": relative.as_posix(),
            "sha256": _sha256(source),
        }
    return result


def _masked(error: Tensor, weight: Tensor) -> Tensor:
    expanded = torch.nan_to_num(weight.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(
        0.0, 1.0
    )
    while expanded.ndim < error.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(error)
    safe = torch.where(expanded > 0.0, error, torch.zeros_like(error))
    return (safe * expanded).sum() / expanded.sum().clamp_min(1.0)


def _row_losses(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    prediction_f = prediction.float()
    target_f = target.detach().float()
    target_rms = target_f.square().mean(dim=-1, keepdim=True).sqrt()
    scale_floor = (0.25 * target_rms.mean(dim=(0, 2), keepdim=True)).clamp_min(1e-3)
    scale = torch.sqrt(target_rms.square() + scale_floor.square())
    raw = F.smooth_l1_loss(prediction_f, target_f, reduction="none").mean(dim=-1, keepdim=True)
    normalized = F.smooth_l1_loss(
        prediction_f / scale,
        target_f / scale,
        reduction="none",
    ).mean(dim=-1, keepdim=True)
    prediction_direction = prediction_f / torch.sqrt(
        prediction_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
    )
    target_direction = target_f / torch.sqrt(
        target_f.square().mean(dim=-1, keepdim=True) + scale_floor.square()
    )
    direction = 0.5 * (prediction_direction - target_direction).square().mean(dim=-1, keepdim=True)
    return raw, normalized, direction


def _transport_diagnostics(
    prediction: FutureObjectDynamics,
    target: FutureObjectDynamics,
    current_loss_support: Tensor,
) -> dict[str, Tensor]:
    prediction_value = prediction.transport_mean.float()
    target_value = target.transport_mean.detach().float()
    batch, intervals, objects = prediction_value.shape[:3]
    cameras = int(current_loss_support.shape[2])
    expected = (batch, objects, cameras, 1)
    if tuple(current_loss_support.shape) != expected:
        raise ValueError(
            f"transport support must be {expected}, got {tuple(current_loss_support.shape)}"
        )
    support = (
        torch.nan_to_num(current_loss_support.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)[
            :, None
        ]
        .expand(-1, intervals, -1, -1, -1)
        .clamp(0.0, 1.0)
    )
    expanded = support.expand_as(prediction_value)
    valid = expanded > 0.0
    prediction_value = torch.where(valid, prediction_value, torch.zeros_like(prediction_value))
    target_value = torch.where(valid, target_value, torch.zeros_like(target_value))
    count = support.sum(dim=1).clamp_min(1.0)
    prediction_common = (prediction_value * support).sum(dim=1) / count
    target_common = (target_value * support).sum(dim=1) / count
    prediction_innovation = prediction_value - prediction_common[:, None]
    target_innovation = target_value - target_common[:, None]
    common_rows = _row_losses(prediction_common[:, None], target_common[:, None])
    innovation_rows = _row_losses(prediction_innovation, target_innovation)
    names = ("raw", "normalized", "direction")
    result: dict[str, Tensor] = {}
    for index, name in enumerate(names):
        common = _masked(common_rows[index], support[:, :1])
        innovation = _masked(innovation_rows[index], support)
        result[f"{name}_common"] = common
        result[f"{name}_innovation"] = innovation
        result[name] = 0.5 * (common + innovation)
    return result


def _owner_parameters(
    model: nn.Module,
) -> dict[str, tuple[tuple[str, nn.Parameter], ...]]:
    named = tuple(
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    exact_prefixes = {
        "transport_head": "world.dynamics.transport_head.",
        "object_geometry": "world.dynamics.object_geometry.",
        "camera_role": "world.dynamics.camera_coordinate_role_condition.",
        "p2_transport_value": "policy_compiler.effect_reader.transport_value.",
    }
    owners = {
        owner: tuple(row for row in named if row[0].startswith(prefix))
        for owner, prefix in exact_prefixes.items()
    }
    excluded = tuple(exact_prefixes.values()) + (
        "world.dynamics.delta_head.",
        "world.dynamics.covariance_head.",
        "world.dynamics.object_semantic.",
        "world.dynamics.object_appearance.",
    )
    owners["w_shared"] = tuple(
        row
        for row in named
        if row[0].startswith("world.dynamics.") and not row[0].startswith(excluded)
    )
    for required in ("transport_head", "object_geometry", "p2_transport_value", "w_shared"):
        if not owners[required]:
            raise RuntimeError(f"missing required VJP owner {required!r}")
    return owners


def _unique_parameter_boundary(
    owners: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> tuple[
    tuple[tuple[str, nn.Parameter], ...],
    dict[str, tuple[int, ...]],
]:
    unique: list[tuple[str, nn.Parameter]] = []
    by_id: dict[int, int] = {}
    indices: dict[str, tuple[int, ...]] = {}
    for owner, rows in owners.items():
        owner_indices: list[int] = []
        for name, parameter in rows:
            index = by_id.get(id(parameter))
            if index is None:
                index = len(unique)
                by_id[id(parameter)] = index
                unique.append((name, parameter))
            owner_indices.append(index)
        indices[owner] = tuple(owner_indices)
    return tuple(unique), indices


def _gradient_vector(
    rows: Sequence[tuple[str, nn.Parameter]],
    gradients: Sequence[Tensor | None],
) -> Tensor:
    pieces = []
    for (_, parameter), gradient in zip(rows, gradients, strict=True):
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            pieces.append(gradient.detach().float().reshape(-1).cpu())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


def _gradient_stats(
    vector: Tensor,
    rows: Sequence[tuple[str, nn.Parameter]],
    *,
    none_count: int = 0,
) -> dict[str, Any]:
    elements = int(sum(parameter.numel() for _, parameter in rows))
    nonzero = int(torch.count_nonzero(vector)) if vector.numel() else 0
    l2 = float(vector.square().sum().sqrt()) if vector.numel() else 0.0
    return {
        "parameter_tensors": len(rows),
        "parameter_elements": elements,
        "gradient_nonzero_elements": nonzero,
        "gradient_l2": l2,
        "gradient_rms": float(math.sqrt(vector.square().mean().item())) if vector.numel() else 0.0,
        "parameter_l2": float(
            torch.cat([parameter.detach().float().reshape(-1).cpu() for _, parameter in rows])
            .square()
            .sum()
            .sqrt()
        )
        if rows
        else 0.0,
        "gradient_none_tensors": int(none_count),
        "finite": bool(torch.isfinite(vector).all()),
    }


def _cosine(left: Tensor, right: Tensor) -> float | None:
    denominator = float(left.square().sum().sqrt() * right.square().sum().sqrt())
    if denominator <= 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _finite_l2(vector: Tensor, *, label: str) -> float:
    """Return a stable L2 norm and fail closed on any non-finite value."""
    if not bool(torch.isfinite(vector).all()):
        raise RuntimeError(f"non-finite vector in transport VJP audit: {label}")
    value = float(torch.linalg.vector_norm(vector.detach().to(dtype=torch.float64)))
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(
            f"invalid L2 norm in transport VJP audit: {label} value={value}"
        )
    return value


def _require_finite_nonnegative(*, label: str, value: float | None) -> float:
    if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
        raise RuntimeError(
            f"invalid non-negative scalar in transport VJP audit: "
            f"{label} value={value}"
        )
    return float(value)


def _projection(reference: Tensor, direction: Tensor) -> float | None:
    norm = float(direction.square().sum().sqrt())
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return float(torch.dot(reference, direction / norm))


def _transport_references(
    prediction: FutureObjectDynamics,
    target: FutureObjectDynamics,
    current_loss_support: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return graph-preserving physical MSE and non-static direction references.

    These are diagnostic references only.  They do not replace the active
    common/innovation Smooth-L1 objective and they never alter its support.
    The direction mask is the pre-registered target-RMS > 1e-3 mask.
    """

    prediction_value = prediction.transport_mean.float()
    target_value = target.transport_mean.detach().float()
    batch, intervals, objects = prediction_value.shape[:3]
    cameras = int(current_loss_support.shape[2])
    expected = (batch, objects, cameras, 1)
    if tuple(current_loss_support.shape) != expected:
        raise ValueError(
            f"transport support must be {expected}, got {tuple(current_loss_support.shape)}"
        )
    support_rows = (
        torch.nan_to_num(current_loss_support.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)[
            :, None
        ]
        .expand(-1, intervals, -1, -1, -1)
        .clamp(0.0, 1.0)
    )
    support = support_rows.expand_as(prediction_value)
    supported = support > 0.0
    if bool((supported & ~torch.isfinite(prediction_value)).any()):
        raise ValueError("transport prediction has non-finite supported values")
    if bool((supported & ~torch.isfinite(target_value)).any()):
        raise ValueError("transport target has non-finite supported values")
    error = torch.where(
        supported, prediction_value - target_value, torch.zeros_like(prediction_value)
    )
    physical_mse = (error.square() * support).sum() / support.sum().clamp_min(1.0)

    target_rms = target_value.square().mean(dim=-1, keepdim=True).sqrt()
    nonstatic = (target_rms > 1e-3).float()
    direction_support = support * nonstatic
    direction_valid = direction_support > 0.0
    prediction_direction_value = torch.where(
        direction_valid, prediction_value, torch.zeros_like(prediction_value)
    )
    target_direction_value = torch.where(
        direction_valid, target_value, torch.zeros_like(target_value)
    )
    prediction_norm = prediction_direction_value / torch.sqrt(
        prediction_direction_value.square().mean(dim=-1, keepdim=True) + 1e-12
    )
    target_norm = target_direction_value / torch.sqrt(
        target_direction_value.square().mean(dim=-1, keepdim=True) + 1e-12
    )
    direction_error = torch.where(
        direction_valid,
        0.5 * (prediction_norm - target_norm).square(),
        torch.zeros_like(prediction_norm),
    )
    physical_direction = (
        direction_error * direction_support
    ).sum() / direction_support.sum().clamp_min(1.0)
    nonstatic_rows = (support_rows * nonstatic).sum()
    return physical_mse, physical_direction, nonstatic_rows


def _batch_identity(batch: TrainingBatch) -> dict[str, Any]:
    def values(value: Tensor | None) -> list[Any] | None:
        return None if value is None else value.detach().cpu().tolist()

    return {
        "sample_indices": values(batch.audit.sample_index),
        "episode_indices": values(batch.audit.episode_index),
        "frame_progress": values(batch.audit.frame_progress),
        "online_state_sha256": _tensor_sha256(batch.online.history.state),
        "action_target_sha256": _tensor_sha256(batch.action_target.normalized),
        "future_offsets_sha256": _tensor_sha256(batch.future.offsets),
    }


def _episode_disjoint_batches(loader: Any, size: int) -> tuple[object, object]:
    dataset = loader.dataset
    reference_owner = getattr(dataset, "base", dataset)
    refs = getattr(reference_owner, "refs", None)
    collate = loader.collate_fn
    if refs is None or not callable(collate):
        raise RuntimeError("validation dataset does not expose refs/collate")
    by_episode: dict[int, list[int]] = {}
    for sample_index, ref in enumerate(refs):
        by_episode.setdefault(int(ref.episode_idx), []).append(sample_index)
    episode_ids = sorted(by_episode)
    if len(episode_ids) < 2 * int(size):
        raise RuntimeError(
            f"validation split has {len(episode_ids)} episodes, need {2 * int(size)}"
        )

    def samples(ids: Sequence[int]) -> list[int]:
        return [by_episode[episode][len(by_episode[episode]) // 2] for episode in ids]

    first_ids = episode_ids[:size]
    second_ids = episode_ids[size : 2 * size]
    first = collate([dataset[index] for index in samples(first_ids)])
    second = collate([dataset[index] for index in samples(second_ids)])
    return first, second


def _forward_batch(
    *,
    model: nn.Module,
    config: Any,
    batch: TrainingBatch,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    owners: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    """Run the frozen engine's one-pass validation graph and collect VJPs.

    This intentionally does not call ``probe_schema29_observation_vjp``'s
    historical ``_formal_forward`` helper.  That helper belongs to an older
    detached-pass/rebuild protocol and omits the current refined Q5 schedule
    and ``FlowStepContext``.
    """

    capture: dict[str, Any] = {}
    transport_head_calls: list[dict[str, Any]] = []
    original_future = losses_module.future_dynamics_terms
    rng_before = _rng_snapshot()

    def capture_transport_head(
        _module: nn.Module,
        args: tuple[Any, ...],
        result: Any,
    ) -> None:
        if len(args) != 1 or not isinstance(args[0], Tensor) or not isinstance(result, Tensor):
            raise RuntimeError("transport_head hook lost its tensor ABI")
        transport_head_calls.append(
            {
                "input_shape": list(args[0].shape),
                "output_shape": list(result.shape),
                "input_rms": _rms(args[0]),
                "output_rms": _rms(result),
            }
        )

    def wrapped_future(
        prediction: FutureObjectDynamics,
        target: FutureObjectDynamics,
        *,
        current_loss_support: Tensor,
        collect_diagnostics: bool = False,
    ) -> dict[str, Tensor]:
        capture["count"] = int(capture.get("count", 0)) + 1
        capture["prediction"] = prediction
        capture["target"] = target
        capture["support"] = current_loss_support
        return original_future(
            prediction,
            target,
            current_loss_support=current_loss_support,
            collect_diagnostics=collect_diagnostics,
        )

    losses_module.future_dynamics_terms = wrapped_future
    transport_hook = model.world.dynamics.transport_head.register_forward_hook(
        capture_transport_head
    )
    try:
        _seed(seed)
        batch.validate(config)
        model.eval()
        flow_generator = _owned_generator(device, seed + 1)
        with torch.enable_grad(), _autocast(device, dtype):
            cache, training_state, _ = model.encode_online(
                batch.online,
                training_mask=False,
                collect_diagnostics=False,
                condition_generator=None,
            )
            top_targets, _ = model.build_training_targets(
                training_state,
                batch.future,
                collect_diagnostics=False,
            )
            schedule = (
                None
                if config.runtime.deployment_flow_schedule is None
                else DeploymentFlowSchedule.from_dict(
                    config.runtime.deployment_flow_schedule
                ).refined
            )
            flow_state = sample_flow_matching(
                batch.action_target.normalized,
                action_state=batch.online.history.action_state,
                codec_gripper_boundary=batch.online.history.codec_gripper_boundary,
                codec=model.outlet_adapter.codec,
                distribution=config.bottom.flow_time_distribution,
                generator=flow_generator,
                schedule=schedule,
            )
            output = model.velocity(
                cache,
                noisy_action_field=flow_state.noisy_physical,
                time=flow_state.time,
                flow_step_context=flow_state.flow_step_context,
                require_execution_supervision=True,
                collect_diagnostics=False,
            )
            ledger = compose_losses(
                config,
                policy_output=output,
                action_target=batch.action_target,
                history=batch.online.history,
                flow_state=flow_state,
                observation=training_state.observation,
                top_targets=top_targets,
                predicted_dynamics=cache.top.predicted_dynamics,
                action_codec=(
                    model.outlet_adapter
                    if (
                        model.outlet_adapter.is_binary_command
                        or getattr(model.outlet_adapter, "is_relative_command", False)
                    )
                    else model.outlet_adapter.codec
                ),
                collect_diagnostics=False,
            )
    finally:
        transport_hook.remove()
        losses_module.future_dynamics_terms = original_future
        _restore_rng(rng_before)
    if set(capture) != {"count", "prediction", "target", "support"} or capture["count"] != 1:
        raise RuntimeError(
            "future dynamics capture did not execute exactly once: "
            f"keys={sorted(capture)} count={capture.get('count')}"
        )
    if capture["prediction"] is not cache.top.predicted_dynamics:
        raise RuntimeError("captured future dynamics is not the final cache.top W2 prediction")
    if len(transport_head_calls) != 2:
        raise RuntimeError(
            "formal W2 must call transport_head exactly for common and innovation: "
            f"observed={len(transport_head_calls)}"
        )
    step_context = flow_state.flow_step_context
    metadata = {
        "flow_time_mean": float(flow_state.time.detach().float().mean()),
        "flow_time_min": float(flow_state.time.detach().float().amin()),
        "flow_time_max": float(flow_state.time.detach().float().amax()),
        "noisy_physical_rms": _rms(flow_state.noisy_physical),
        "noisy_physical_sha256": _tensor_sha256(flow_state.noisy_physical),
        "flow_step_context": None
        if step_context is None
        else {
            "time_sha256": _tensor_sha256(step_context.time),
            "step_size_sha256": _tensor_sha256(step_context.step_size),
            "normalized_index_sha256": _tensor_sha256(step_context.normalized_index),
            "endpoint_sha256": _tensor_sha256(step_context.endpoint),
        },
        "context_mask_sha256": _tensor_sha256(training_state.observation.grounding.context_mask),
        "context_mask_fraction": float(
            training_state.observation.grounding.context_mask.detach().float().mean()
        ),
        "formal_w_semantic_rms": _rms(cache.top.predicted_dynamics.semantic_delta),
        "formal_w_transport_rms": _rms(cache.top.predicted_dynamics.transport_mean),
        "flow_schedule": None
        if config.runtime.deployment_flow_schedule is None
        else DeploymentFlowSchedule.from_dict(
            config.runtime.deployment_flow_schedule
        ).refined.to_dict(),
        "capture_call_count": int(capture["count"]),
        "transport_head_calls": transport_head_calls,
    }
    diagnostics = _transport_diagnostics(
        capture["prediction"], capture["target"], capture["support"]
    )
    objective_weight = float(config.objectives.future_dynamics) * 0.15
    raw = (
        objective_weight
        * 0.5
        * (ledger.terms["future_transport_common"] + ledger.terms["future_transport_innovation"])
    )
    recomputed_raw = objective_weight * diagnostics["raw"]
    raw_residual = float((raw - recomputed_raw).detach().float().abs())
    raw_tolerance = max(1e-10, 1e-5 * abs(float(raw.detach().float())))
    if raw_residual > raw_tolerance:
        raise RuntimeError(
            "transport raw loss replay does not close: "
            f"residual={raw_residual} tolerance={raw_tolerance}"
        )
    normalized = objective_weight * diagnostics["normalized"]
    direction = objective_weight * 0.10 * diagnostics["direction"]
    candidate = normalized + direction
    physical_mse, physical_direction, nonstatic_rows = _transport_references(
        capture["prediction"], capture["target"], capture["support"]
    )
    support_audit = torch.nan_to_num(
        capture["support"].detach().float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0.0, 1.0)
    semantic = (
        float(config.objectives.future_dynamics) * 0.55 * ledger.terms["future_semantic_delta"]
    )
    covariance = float(config.objectives.future_dynamics) * 0.05 * ledger.terms["future_covariance"]
    transition = ledger.contributions["future_transition"]
    scalars = {
        "raw_common": objective_weight * 0.5 * ledger.terms["future_transport_common"],
        "raw_innovation": objective_weight * 0.5 * ledger.terms["future_transport_innovation"],
        "raw": raw,
        "normalized": normalized,
        "direction": direction,
        "candidate": candidate,
        "physical_mse": physical_mse,
        "physical_direction": physical_direction,
        "action": ledger.groups["action"],
        "execution": ledger.groups["execution"],
        "semantic": semantic,
        "covariance": covariance,
        "transition": transition,
    }
    boundary, owner_indices = _unique_parameter_boundary(owners)
    parameters = tuple(parameter for _, parameter in boundary)
    reports: dict[str, Any] = {}
    retained: dict[str, dict[str, Tensor]] = {}

    def vjp(scalar: Tensor) -> tuple[Tensor | None, ...]:
        if not bool(scalar.requires_grad):
            return tuple(None for _ in parameters)
        return torch.autograd.grad(
            scalar,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    for name, scalar in scalars.items():
        gradients = vjp(scalar)
        if not bool(torch.isfinite(scalar.detach().float())):
            raise RuntimeError(f"non-finite scalar in transport VJP: {name}")
        owner_reports: dict[str, Any] = {}
        retained[name] = {}
        for owner, indices in owner_indices.items():
            rows = tuple(boundary[index] for index in indices)
            owner_gradients = tuple(gradients[index] for index in indices)
            vector = _gradient_vector(rows, owner_gradients)
            if not bool(torch.isfinite(vector).all()):
                raise RuntimeError(f"non-finite transport VJP for scalar={name} owner={owner}")
            owner_reports[owner] = _gradient_stats(
                vector,
                rows,
                none_count=sum(value is None for value in owner_gradients),
            )
            if owner in PRIMARY_OWNERS or owner in {
                "p2_transport_value",
                "w_shared",
            }:
                retained[name][owner] = vector
        reports[name] = {
            "value": float(scalar.detach().float()),
            "owners": owner_reports,
        }
        del gradients
    raw_from_parts: dict[str, float | None] = {}
    scalar_controls: dict[str, dict[str, Any]] = {}
    for owner in (*PRIMARY_OWNERS, "w_shared", "p2_transport_value"):
        raw_vector = retained.get("raw", {}).get(owner)
        common_vector = retained.get("raw_common", {}).get(owner)
        innovation_vector = retained.get("raw_innovation", {}).get(owner)
        candidate_vector = retained.get("candidate", {}).get(owner)
        if raw_vector is None or common_vector is None or innovation_vector is None:
            continue
        parts_error = _finite_l2(
            raw_vector - common_vector - innovation_vector,
            label=f"{owner}.independent_common_plus_innovation_error",
        )
        raw_from_parts[owner] = parts_error
        raw_norm = _finite_l2(raw_vector, label=f"{owner}.raw")
        candidate_norm = (
            _finite_l2(candidate_vector, label=f"{owner}.candidate")
            if candidate_vector is not None
            else 0.0
        )
        scale = candidate_norm / max(raw_norm, 1e-30)
        if not math.isfinite(scale) or scale < 0.0:
            raise RuntimeError(
                f"invalid candidate/raw scale in transport VJP audit: "
                f"owner={owner} value={scale}"
            )
        scalar_vector = raw_vector * scale
        scalar_controls[owner] = {
            "candidate_to_raw_norm_ratio": scale,
            "candidate_scalar_raw_cosine": _cosine(candidate_vector, scalar_vector)
            if candidate_vector is not None
            else None,
            "candidate_scalar_raw_l2_error": float(
                (candidate_vector - scalar_vector).square().sum().sqrt()
            )
            if candidate_vector is not None
            else None,
        }
    # The two component VJPs above are intentionally independent backward
    # executions.  Under BF16 autocast their intermediate adjoints can be
    # rounded before the FP32 parameter-gradient vector is materialized, so
    # adding those two saved vectors is an audit of backward-order sensitivity,
    # not the algebraic closure gate.  Recompute common+innovation in one VJP
    # and compare that joint graph with the active raw objective instead.
    joint_scalar = scalars["raw_common"] + scalars["raw_innovation"]
    if not bool(torch.isfinite(joint_scalar.detach().float())):
        raise RuntimeError("non-finite joint common+innovation transport scalar")
    joint_parts = torch.autograd.grad(
        joint_scalar,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    joint_parts_error: dict[str, float | None] = {}
    retained["raw_joint_parts"] = {}
    for owner, indices in owner_indices.items():
        rows = tuple(boundary[index] for index in indices)
        owner_joint = tuple(joint_parts[index] for index in indices)
        vector = _gradient_vector(rows, owner_joint)
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError(
                f"non-finite joint common+innovation VJP: owner={owner}"
            )
        if owner in PRIMARY_OWNERS or owner in {"p2_transport_value", "w_shared"}:
            retained["raw_joint_parts"][owner] = vector
        reference = retained["raw"].get(owner) if owner in retained["raw"] else None
        if reference is None:
            joint_parts_error[owner] = None
        else:
            joint_parts_error[owner] = _finite_l2(
                vector - reference,
                label=f"{owner}.joint_common_plus_innovation_error",
            )
    del joint_parts
    repeat = (
        torch.autograd.grad(scalars["raw"], parameters, retain_graph=False, allow_unused=True)
        if bool(scalars["raw"].requires_grad)
        else tuple(None for _ in parameters)
    )
    repeat_error: dict[str, float | None] = {}
    repeat_none: dict[str, int] = {}
    retained["raw_repeat"] = {}
    for owner, indices in owner_indices.items():
        rows = tuple(boundary[index] for index in indices)
        owner_repeat = tuple(repeat[index] for index in indices)
        vector = _gradient_vector(rows, owner_repeat)
        if not bool(torch.isfinite(vector).all()):
            raise RuntimeError(f"non-finite repeated raw transport VJP: owner={owner}")
        if owner in PRIMARY_OWNERS or owner in {"p2_transport_value", "w_shared"}:
            retained["raw_repeat"][owner] = vector
        repeat_none[owner] = sum(value is None for value in owner_repeat)
        reference = retained["raw"].get(owner) if owner in retained["raw"] else None
        if reference is None:
            repeat_error[owner] = None
        else:
            repeat_error[owner] = _finite_l2(
                vector - reference,
                label=f"{owner}.raw_repeat_error",
            )
    gradient_parity: dict[str, Any] = {}
    for owner, parts_error in raw_from_parts.items():
        raw_vector = retained["raw"][owner]
        raw_norm = _finite_l2(raw_vector, label=f"{owner}.raw_parity")
        parts_error = _require_finite_nonnegative(
            label=f"{owner}.independent_common_plus_innovation_error",
            value=parts_error,
        )
        repeat_floor = _require_finite_nonnegative(
            label=f"{owner}.raw_repeat_error",
            value=repeat_error.get(owner),
        )
        joint_error = _require_finite_nonnegative(
            label=f"{owner}.joint_common_plus_innovation_error",
            value=joint_parts_error.get(owner),
        )
        tolerance = max(
            5.0 * repeat_floor,
            1e-10,
            1e-5 * raw_norm,
        )
        tolerance = _require_finite_nonnegative(
            label=f"{owner}.joint_parity_tolerance",
            value=tolerance,
        )
        passed = bool(joint_error <= tolerance)
        gradient_parity[owner] = {
            "independent_common_plus_innovation_l2_error": parts_error,
            "independent_backward_relative_error": (
                None
                if parts_error is None
                else float(parts_error) / max(raw_norm, 1e-30)
            ),
            "raw_from_joint_common_innovation_l2_error": joint_error,
            "tolerance": tolerance,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(
                "raw transport VJP does not equal the joint common+innovation VJP: "
                f"owner={owner} joint_error={joint_error} tolerance={tolerance} "
                f"independent_sum_error={parts_error} "
                f"raw_norm={raw_norm} repeat_error={repeat_floor} "
                f"raw_value={float(scalars['raw'].detach().float())} "
                f"common_value={float(scalars['raw_common'].detach().float())} "
                f"innovation_value={float(scalars['raw_innovation'].detach().float())}"
            )
    within_direction: dict[str, Any] = {}
    for owner in (*PRIMARY_OWNERS, "w_shared", "p2_transport_value"):
        if owner not in retained["raw"] or retained["raw"][owner].numel() == 0:
            continue
        raw_vector = retained["raw"][owner]
        candidate_vector = retained["candidate"][owner]
        within_direction[owner] = {
            "raw_candidate_cosine": _cosine(raw_vector, candidate_vector),
            "candidate_to_raw_l2_ratio": float(
                candidate_vector.square().sum().sqrt()
                / raw_vector.square().sum().sqrt().clamp_min(1e-30)
            ),
            "raw_repeat_l2_error": repeat_error[owner],
            "raw_repeat_none_tensors": repeat_none.get(owner),
        }
    result = {
        "batch": _batch_identity(batch),
        "random_seeds": {
            "global": seed,
            "flow": seed + 1,
        },
        "forward": metadata,
        "reference": {
            "nonstatic_target_transport_rows": float(nonstatic_rows.detach().float()),
            "nonstatic_threshold": 1e-3,
            "physical_mse_is_original_coordinate": True,
            "current_support_weight": float(support_audit.sum()),
            "current_supported_camera_rows": int(torch.count_nonzero(support_audit > 0.0)),
            "prediction_transport_rms": _rms(capture["prediction"].transport_mean),
            "target_transport_rms": _rms(capture["target"].transport_mean),
        },
        "transport_raw_replay_residual": raw_residual,
        "transport_raw_replay_tolerance": raw_tolerance,
        "scalars": reports,
        "raw_gradient_parts_l2_error": raw_from_parts,
        "raw_gradient_joint_parts_l2_error": joint_parts_error,
        "raw_gradient_parity": gradient_parity,
        "scalar_control": scalar_controls,
        "within_batch_direction": within_direction,
        "raw_repeat_error": repeat_error,
        "raw_repeat_none_tensors": repeat_none,
    }
    return result, retained


def _cross_batch(
    left: Mapping[str, Mapping[str, Tensor]],
    right: Mapping[str, Mapping[str, Tensor]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    reference_names = (
        "physical_mse",
        "physical_direction",
        "action",
        "execution",
        "semantic",
        "covariance",
        "transition",
    )
    for owner in (*PRIMARY_OWNERS, "w_shared", "p2_transport_value"):
        if owner not in left.get("raw", {}) or owner not in right.get("raw", {}):
            continue
        owner_result: dict[str, Any] = {}
        for reference_name in reference_names:
            reference = right.get(reference_name, {}).get(owner)
            if reference is None:
                continue
            raw_projection = _projection(reference, left["raw"][owner])
            raw_repeat_projection = _projection(reference, left["raw_repeat"][owner])
            candidate_projection = _projection(reference, left["candidate"][owner])
            projection_repeat_floor = (
                None
                if raw_projection is None or raw_repeat_projection is None
                else abs(raw_repeat_projection - raw_projection)
            )
            tau_b = max(
                1e-12,
                5.0 * (0.0 if projection_repeat_floor is None else projection_repeat_floor),
            )
            if (
                raw_projection is not None
                and candidate_projection is not None
                and raw_projection > tau_b
            ):
                relative = (candidate_projection - raw_projection) / raw_projection
            else:
                relative = None
            owner_result[reference_name] = {
                "raw_unit_direction_projection": raw_projection,
                "raw_repeat_unit_direction_projection": raw_repeat_projection,
                "candidate_unit_direction_projection": candidate_projection,
                "projection_repeat_floor": projection_repeat_floor,
                "tau_b": tau_b,
                "candidate_relative_improvement": relative,
                "raw_projection_positive": bool(
                    raw_projection is not None and raw_projection > tau_b
                ),
                "candidate_positive_when_raw_nonpositive": bool(
                    (raw_projection is None or raw_projection <= tau_b)
                    and candidate_projection is not None
                    and candidate_projection > tau_b
                ),
            }
        result[owner] = owner_result
    return result


def run(
    *,
    checkpoint: Path,
    config_path: Path,
    t5_condition: Path | None,
    dinov2_model: Path | None,
    local_files_only: bool,
    output: Path,
    device_name: str,
    episodes_per_batch: int,
    repo_root: Path,
) -> dict[str, Any]:
    device = torch.device(device_name)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    model = policy.bundle.model
    config = load_config(config_path)
    if config.digest(include_paths=False) != policy.bundle.identity.config_digest:
        raise ValueError(
            "training config executable digest does not match checkpoint identity: "
            f"saved={policy.bundle.identity.config_digest} "
            f"current={config.digest(include_paths=False)}"
        )
    source_snapshot = active_source_snapshot(repo_root)
    if source_snapshot.digest != policy.bundle.identity.source.digest:
        raise ValueError(
            "executed source snapshot does not match checkpoint identity: "
            f"saved={policy.bundle.identity.source.digest} current={source_snapshot.digest}"
        )
    module_origins = _module_origins(repo_root, model)
    model.eval()
    if hasattr(model, "set_training_step"):
        model.set_training_step(int(policy.bundle.global_step))
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("VJP probe requires pristine parameter .grad fields")
    data = load_mainline_data(config)
    live_dataset_identity = dataset_identity(data, config)
    saved_dataset_semantic = {
        key: value
        for key, value in asdict(policy.bundle.identity.dataset).items()
        if key != "raw_root"
    }
    live_dataset_semantic = {
        key: value for key, value in asdict(live_dataset_identity).items() if key != "raw_root"
    }
    if live_dataset_semantic != saved_dataset_semantic:
        raise ValueError(
            "live validation data identity does not match checkpoint identity: "
            f"saved={saved_dataset_semantic} current={live_dataset_semantic}"
        )
    live_language_identity = language_identity(data, config)
    saved_language_semantic = {
        "logical_name": policy.bundle.identity.language.logical_name,
        "size_bytes": policy.bundle.identity.language.size_bytes,
        "sha256": policy.bundle.identity.language.sha256,
    }
    live_language_semantic = {
        "logical_name": live_language_identity.logical_name,
        "size_bytes": live_language_identity.size_bytes,
        "sha256": live_language_identity.sha256,
    }
    if live_language_semantic != saved_language_semantic:
        raise ValueError(
            "live language identity does not match checkpoint identity: "
            f"saved={saved_language_semantic} current={live_language_semantic}"
        )
    loader = data.loader(
        "val",
        batch_size=int(episodes_per_batch),
        workers=0,
        device=device,
        generator=torch.Generator().manual_seed(int(config.data.seed) + 6101),
    )
    raw_a, raw_b = _episode_disjoint_batches(loader, int(episodes_per_batch))
    batch_a = to_training_batch(raw_a, goal=data.goal, config=config, device=device)
    batch_b = to_training_batch(raw_b, goal=data.goal, config=config, device=device)
    episodes_a = set(int(value) for value in batch_a.audit.episode_index.tolist())
    episodes_b = set(int(value) for value in batch_b.audit.episode_index.tolist())
    if episodes_a & episodes_b:
        raise RuntimeError("VJP A/B batches are not episode-disjoint")
    owners = _owner_parameters(model)
    rng_before = _rng_snapshot()
    flags_before = _training_flags(model)
    digest_before = _state_digest(model)
    nonpersistent_before = _nonpersistent_buffer_digest(model)
    report_a, gradients_a = _forward_batch(
        model=model,
        config=config,
        batch=batch_a,
        device=device,
        dtype=resolve_compute_dtype(config),
        seed=62001,
        owners=owners,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report_b, gradients_b = _forward_batch(
        model=model,
        config=config,
        batch=batch_b,
        device=device,
        dtype=resolve_compute_dtype(config),
        seed=63001,
        owners=owners,
    )
    digest_after = _state_digest(model)
    nonpersistent_after = _nonpersistent_buffer_digest(model)
    flags_after = _training_flags(model)
    rng_after = _rng_snapshot()
    if digest_before != digest_after:
        raise RuntimeError("frozen transport VJP changed a parameter or buffer")
    if nonpersistent_before != nonpersistent_after:
        raise RuntimeError("frozen transport VJP changed a non-persistent buffer")
    if flags_before != flags_after:
        raise RuntimeError("frozen transport VJP changed module training flags")
    if not _rng_equal(rng_before, rng_after):
        raise RuntimeError("frozen transport VJP did not restore global RNG state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad probe populated persistent .grad fields")
    result = {
        "schema": SCHEMA,
        "diagnostic_not_training": True,
        "harness_valid": True,
        "candidate_package_admitted": False,
        "verdict": "raw_cross_batch_measurements_pending_pre_registered_gate_and_second_variant",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "training_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "digest": config.digest(include_paths=False),
        },
        "source_commit": str(policy.bundle.identity.git_commit),
        "source_snapshot_digest": source_snapshot.digest,
        "executed_module_origins": module_origins,
        "live_dataset_identity": {
            "inventory_sha256": live_dataset_identity.inventory_sha256,
            "state_normalizer_sha256": live_dataset_identity.state_normalizer_sha256,
            "action_normalizer_sha256": live_dataset_identity.action_normalizer_sha256,
            "decoded_cache_identity": live_dataset_identity.decoded_cache_identity,
            "dino_cache_identity": live_dataset_identity.dino_cache_identity,
        },
        "live_language_identity": {
            "logical_name": live_language_identity.logical_name,
            "sha256": live_language_identity.sha256,
        },
        "device": str(device),
        "episodes_per_batch": int(episodes_per_batch),
        "model_state_sha256_before": digest_before,
        "model_state_sha256_after": digest_after,
        "nonpersistent_buffer_sha256_before": nonpersistent_before,
        "nonpersistent_buffer_sha256_after": nonpersistent_after,
        "training_flags_unchanged": flags_before == flags_after,
        "global_rng_restored": _rng_equal(rng_before, rng_after),
        "owners": {
            name: [parameter_name for parameter_name, _ in rows] for name, rows in owners.items()
        },
        "batches": {"A": report_a, "B": report_b},
        "cross_batch": {
            "A_to_B": _cross_batch(gradients_a, gradients_b),
            "B_to_A": _cross_batch(gradients_b, gradients_a),
        },
        "interpretation": {
            "raw": "the checkpoint's actually weighted raw-coordinate transport objective",
            "candidate": "normalized transport plus 0.10 direction, evaluated only as a frozen VJP",
            "positive_projection": "a unit negative-gradient step from the source batch predicts first-order descent of the named reference on the other batch",
            "gate_boundary": "this file records repeat-floor-aware projections but does not itself claim the H6 repair gate or a shared two-variant package passed",
            "guard": "no optimizer, scheduler, parameter perturbation, checkpoint write, or Teacher input to deployment occurred",
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-batch", type=int, default=6)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    if int(args.episodes_per_batch) < 1:
        raise ValueError("--episodes-per-batch must be positive")
    result = run(
        checkpoint=args.checkpoint,
        config_path=args.config,
        t5_condition=args.t5_condition,
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        output=args.output,
        device_name=str(args.device),
        episodes_per_batch=int(args.episodes_per_batch),
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint": result["checkpoint"]["sha256"],
                "cross_batch_owners": sorted(result["cross_batch"]["A_to_B"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
