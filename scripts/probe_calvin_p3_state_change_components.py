#!/usr/bin/env python3
"""Split the frozen CALVIN P3 state-change source into its real components.

This is a checkpoint diagnostic, not a benchmark and not a replacement
controller.  The probe reconstructs the exact S-owned state-change evidence,
checks parity with the cached value, and then changes one pre-fusion component
at a time while running the complete matched-noise Q5 proposal/rebuild/refined
lifecycle.

Colour masks select privileged diagnostic interventions only after a strict
per-camera ownership/null/coordinate gate. They are not learned task grounding.
This bounded script measures total-path sensitivity, not harmfulness or success.
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
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.gripper_contract import is_binary_gripper_mode
from clearvla.mainline.model.routing import smooth_rms_contract
from clearvla.mainline.runtime.flow_schedule import resolve_deployment_flow_schedule
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import (
    refine_cached_world,
    sample_cached_action,
    sample_refined_cached_action_with_cache,
)
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

try:
    from scripts.probe_calvin_latest_checkpoint_target_y_replay import ladder
except ImportError:
    from probe_calvin_latest_checkpoint_target_y_replay import (  # type: ignore[no-redef]
        ladder,
    )


SCHEMA = "clearvla-calvin-p3-state-change-components-v2"
NOISE_SEED = 9182026
FROZEN_CHECKPOINTS = {
    "6e137830dab5920f62ddfbe0af9fc3ac0a1d7c112365a2fc80ed7fc4d962c755": "camera",
    "63581c8f8d45bf9a14a316c6582afacd2e52cecc2671f599dd4ddffc5a0d10c6": "sequence",
}
GLOBAL_MODES = (
    "baseline",
    "self_motion_zero",
    "object_transport_zero",
    "both_zero",
)
OBJECT_MODES = (
    "target_contribution_zero",
    "control_contribution_zero",
    "non_target_contribution_zero",
    "target_conditioned_mean",
    "control_conditioned_mean",
)


def _finite(value: torch.Tensor, name: str) -> None:
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _tree_sha256(value: Any) -> str:
    """Include tensor dtype/shape/bytes; do not cast BF16 or sanitize NaNs."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            digest.update(type(item).__name__.encode())
            for field in fields(item):
                visit(field.name)
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\x00")

    visit(value)
    return digest.hexdigest()


def _model_sha256(model: Any) -> str:
    return _tree_sha256(
        {
            "parameters": dict(model.named_parameters()),
            "buffers_including_nonpersistent": dict(model.named_buffers()),
        }
    )


@contextmanager
def _frozen_model(model: Any):
    flags = [(module, module.training) for module in model.modules()]
    before = _model_sha256(model)
    record = {"model_before_sha256": before}
    model.eval()
    try:
        yield record
    finally:
        for module, flag in flags:
            module.training = flag
        after = _model_sha256(model)
        if before != after:
            raise RuntimeError("diagnostic changed model parameters or buffers")
        record.update(
            model_after_sha256=after,
            parameters_and_all_buffers_unchanged=True,
            training_flags_restored=True,
        )


@contextmanager
def _preserved_rng():
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _parity(candidate: torch.Tensor, reference: torch.Tensor, name: str) -> dict[str, Any]:
    _finite(candidate, name)
    _finite(reference, f"{name} reference")
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise RuntimeError(f"{name}: reconstruction shape/dtype changed")
    left, right = candidate.detach().float(), reference.detach().float()
    delta = left - right
    if not torch.allclose(left, right, atol=1e-7, rtol=1e-5):
        raise RuntimeError(f"{name}: reconstruction failed parity, max_abs={_max_abs(delta)}")
    return {
        "passed": True,
        "bitwise_equal": _tree_sha256(candidate) == _tree_sha256(reference),
        "max_abs": _max_abs(delta),
        "delta_rms": _rms(delta),
        "atol": 1e-7,
        "rtol": 1e-5,
    }


def _same_fields(left: Any, right: Any, *, except_names: set[str]) -> None:
    for field in fields(left):
        if field.name not in except_names and getattr(left, field.name) is not getattr(
            right, field.name
        ):
            raise RuntimeError(f"unrelated cache field replaced: {field.name}")


def _source_identity(policy: Any, repo_root: Path | None) -> dict[str, Any]:
    root = (repo_root or Path(inspect.getfile(active_source_snapshot)).parents[2]).resolve()
    snapshot = active_source_snapshot(root)
    if snapshot.digest != policy.bundle.identity.source.digest:
        raise RuntimeError("executed checkout differs from checkpoint source identity")
    if policy.bundle.checkpoint_sha256 not in FROZEN_CHECKPOINTS:
        raise RuntimeError("this bounded probe admits only the two frozen E1 checkpoints")
    if (int(policy.bundle.epoch), int(policy.bundle.global_step)) != (1, 11012):
        raise RuntimeError("frozen E1/11012 identity changed")
    model = policy.bundle.model
    owners = {
        "policy": (type(model), "clearvla/mainline/model/policy.py"),
        "intent": (type(model.intent.organizer), "clearvla/mainline/model/intent.py"),
        "world": (type(model.world), "clearvla/mainline/model/components.py"),
        "dynamics": (type(model.world.dynamics), "clearvla/mainline/model/dynamics.py"),
        "plan": (type(model.policy_compiler.plan_compiler), "clearvla/mainline/model/compiler.py"),
        "sampling": (sample_cached_action, "clearvla/mainline/runtime/sampling.py"),
        "refinement": (refine_cached_world, "clearvla/mainline/runtime/sampling.py"),
        "native_lifecycle": (
            sample_refined_cached_action_with_cache,
            "clearvla/mainline/runtime/sampling.py",
        ),
        "routing": (smooth_rms_contract, "clearvla/mainline/model/routing.py"),
        "loader": (ClearVLACheckpointPolicy, "clearvla/simulation/clearvla_policy.py"),
    }
    origins = {}
    unwrap = getattr(inspect, "unwrap", lambda value: value)
    for name, (owner, relative) in owners.items():
        unwrapped = unwrap(owner)
        source = Path(inspect.getfile(unwrapped)).resolve()
        if source != (root / relative).resolve():
            raise RuntimeError(f"wrong executed module origin: {name}: {source}")
        origins[name] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "decorated_path": str(Path(inspect.getfile(owner)).resolve()),
            "decorator_unwrapped": unwrapped is not owner,
        }
    return {
        "repo_root": str(root),
        "source_digest": snapshot.digest,
        "checkpoint_arm": FROZEN_CHECKPOINTS[policy.bundle.checkpoint_sha256],
        "module_origins": origins,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshot_helper_sha256": hashlib.sha256(
            Path(inspect.getfile(ladder._snapshot_path)).read_bytes()
        ).hexdigest(),
        "ladder_helper_sha256": hashlib.sha256(
            Path(inspect.getfile(ladder._slot_map)).read_bytes()
        ).hexdigest(),
        "online_input_helper_sha256": hashlib.sha256(
            Path(inspect.getfile(ladder._build_online)).read_bytes()
        ).hexdigest(),
    }


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def _rms(value: torch.Tensor) -> float:
    _finite(value, "RMS input")
    return float(value.detach().float().square().mean().sqrt().cpu())


def _max_abs(value: torch.Tensor) -> float:
    _finite(value, "maximum input")
    return float(value.detach().float().abs().amax().cpu())


def _trace_rms(values: list[torch.Tensor]) -> float:
    if not values:
        raise ValueError("empty P3 trace")
    return _rms(torch.cat([value.reshape(-1) for value in values]))


def _decode_action(policy: Any, action: torch.Tensor, command: torch.Tensor | None) -> np.ndarray:
    _finite(action, "normalized action")
    normalized = action.detach().float().cpu().numpy()
    decoded = policy.bundle.action_normalizer.decode(normalized).astype(np.float32)
    if decoded.ndim != 3 or decoded.shape[0] != 1 or decoded.shape[-1] != 7:
        raise ValueError("expected one CALVIN [1,T,7] action")
    mode = policy.bundle.config.bottom.gripper_output_mode
    if is_binary_gripper_mode(mode):
        if command is None:
            raise ValueError("binary outlet lost terminal gripper command")
        _finite(command, "gripper command")
        values = command.detach().float().cpu().numpy()
        if values.shape != decoded.shape[:-1] or not np.isin(values, [-1.0, 1.0]).all():
            raise ValueError("binary gripper command must be [B,T] in {-1,+1}")
        decoded[..., -1] = values
    elif mode != "continuous":
        raise ValueError(f"unsupported gripper mode {mode!r}")
    if not np.isfinite(decoded).all():
        raise ValueError("nonfinite decoded action")
    return decoded


def _decoded_summary(
    policy: ClearVLACheckpointPolicy,
    action: torch.Tensor,
    command: torch.Tensor | None,
) -> dict[str, Any]:
    decoded = _decode_action(policy, action, command)
    return {
        "semantics": "native command before environment clipping; plan, not an executed trajectory",
        "all_rows": decoded[0].tolist(),
        "first_row": decoded[0, 0].tolist(),
        "first4_mean": decoded[0, :4].mean(axis=0).tolist(),
        "first8_mean": decoded[0, :8].mean(axis=0).tolist(),
        "arm_rms": float(np.sqrt(np.mean(np.square(decoded[..., :-1])))),
    }


def _decoded_delta(
    policy: ClearVLACheckpointPolicy,
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    candidate_command: torch.Tensor | None,
    baseline_command: torch.Tensor | None,
) -> dict[str, Any]:
    left = _decode_action(policy, candidate, candidate_command)
    right = _decode_action(policy, baseline, baseline_command)
    delta = left - right
    return {
        "first_row": delta[0, 0].tolist(),
        "first4_mean": delta[0, :4].mean(axis=0).tolist(),
        "arm_rmse": float(np.sqrt(np.mean(np.square(delta[..., :-1])))),
        "first_xyz_rms": float(np.sqrt(np.mean(np.square(delta[0, 0, :3])))),
        "first_xyz_max_abs": float(np.abs(delta[0, 0, :3]).max()),
        "binary_gripper_flips": int(np.count_nonzero(left[..., -1] != right[..., -1]))
        if is_binary_gripper_mode(policy.bundle.config.bottom.gripper_output_mode)
        else None,
    }


def _identity_row_ok(row: Mapping[str, Any], *, minimum_margin: float) -> bool:
    """Fail closed on missing/invalid color evidence; never fall back to read-chart K."""
    if not math.isfinite(minimum_margin) or minimum_margin <= 0:
        raise ValueError("ownership margin must be finite and positive")
    try:
        ownership = np.asarray(row["ownership_per_camera"], dtype=np.float64)
        validity = np.asarray(row["camera_validity"], dtype=np.float64)
        support = np.asarray(row["camera_support"], dtype=np.float64)
        null = np.asarray(row["null_ownership_per_camera"], dtype=np.float64)
        mass = np.asarray(row["mask_mass"], dtype=np.float64)
        coords = np.asarray(row["camera_coordinates"], dtype=np.float64)
        centroid = np.asarray(row["mask_centroid"], dtype=np.float64)
        scores = np.asarray(row["ownership_scores"], dtype=np.float64)
        if ownership.ndim != 2 or min(ownership.shape) < 1:
            return False
        count, cameras = ownership.shape
        target = int(row["target_k"])
        if not 0 <= target < count or count < 2:
            return False
        if (
            validity.shape != ownership.shape
            or support.shape != ownership.shape
            or null.shape != (cameras,)
            or mass.shape != (cameras,)
            or coords.shape != (count, cameras, 2)
            or centroid.shape != (cameras, 2)
            or scores.shape != (count,)
        ):
            return False
        if not all(
            np.isfinite(v).all()
            for v in (ownership, validity, support, null, mass, coords, centroid, scores)
        ):
            return False
        if (
            not (mass > 0).all()
            or not (validity[target] > 0).all()
            or not (support[target] > 0).all()
            or any(
                (v < 0).any() or (v > 1 + 1e-6).any() for v in (ownership, validity, support, null)
            )
        ):
            return False
        competitors = np.delete(ownership, target, axis=0).max(axis=0)
        if not ((ownership[target] - np.maximum(competitors, null)) >= minimum_margin).all():
            return False
        combined = (ownership * validity).sum(axis=1) / np.maximum(validity.sum(axis=1), 1.0)
        if not np.allclose(scores, combined, atol=1e-6, rtol=1e-6):
            return False
        if (
            int(combined.argmax()) != target
            or combined[target] - np.delete(combined, target).max() < minimum_margin
        ):
            return False
        permutation = [int(k) for k in row["permutation"]]
        if sorted(permutation) != list(range(count)):
            return False
        error = float(row["permutation_score_max_abs_error"])
        margin = float(row["ownership_margin"])
        return bool(
            math.isfinite(error)
            and 0 <= error <= 1e-6
            and math.isfinite(margin)
            and margin >= minimum_margin
            and row["ownership_and_coordinate_agree"]
            and int(row["coordinate_nearest_k"]) == target
            and row["all_valid_cameras_agree"]
            and list(row["ownership_argmax_per_camera"]) == [target] * cameras
            and list(row["coordinate_in_colour_region_per_camera"]) == [True] * cameras
            and int(row["permuted_target_k"]) == permutation.index(target)
            and int(row["expected_permuted_target_k"]) == permutation.index(target)
        )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False


def _encode_with_components(
    policy: Any, history: Any, instruction: str, runtime_dtype: torch.dtype
):
    """One native encode, with observation-only captures of the actual S producer."""
    online = ladder._build_online(policy, history, instruction)
    organizer = policy.bundle.model.intent.organizer
    captures: dict[str, list[torch.Tensor]] = {}

    def save(name: str, value: torch.Tensor) -> None:
        captures.setdefault(name, []).append(value.detach().clone())

    def history_hook(_module: Any, _args: Any, output: Any) -> None:
        save("self_motion", output[0][:, 0])

    def transport_hook(_module: Any, args: Any, output: Any) -> None:
        save("transport_prior", args[0])
        save("transport_tokens", output)

    def fuse_hook(_module: Any, args: Any, _output: Any) -> None:
        save("fuse_input", args[0])

    hooks = [
        organizer.state_change_read.register_forward_hook(history_hook),
        organizer.state_change_transport.register_forward_hook(transport_hook),
        organizer.state_change_fuse.register_forward_hook(fuse_hook),
    ]
    try:
        enabled = policy.device.type in {"cpu", "cuda"} and runtime_dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with (
            torch.no_grad(),
            torch.autocast(device_type=policy.device.type, dtype=runtime_dtype, enabled=enabled),
        ):
            cache, state, _ = policy.bundle.model.encode_online(
                online,
                training_mask=False,
                geometry_supervision=False,
                collect_diagnostics=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
    if set(captures) != {"self_motion", "transport_prior", "transport_tokens", "fuse_input"} or any(
        len(v) != 1 for v in captures.values()
    ):
        raise RuntimeError("state-change producer must run exactly once during encode")
    facts = state.top.facts
    facts.validate()
    fact_valid = torch.nan_to_num(facts.validity.float(), nan=0, posinf=0, neginf=0).clamp(0, 1)
    belief_valid = torch.nan_to_num(
        cache.top.belief.validity.float(), nan=0, posinf=0, neginf=0
    ).clamp(0, 1)
    _parity(fact_valid, belief_valid, "fact/compact-belief validity")
    # Producer-invalid rows are zeroed before Linear, including in this audit.
    fact_prior = torch.where(
        fact_valid > 0, facts.transport_prior, torch.zeros_like(facts.transport_prior)
    )
    belief_prior = torch.where(
        belief_valid > 0,
        cache.top.belief.transport_prior,
        torch.zeros_like(cache.top.belief.transport_prior),
    )
    _parity(fact_prior, belief_prior, "fact/compact-belief supported prior")
    maps = {}
    for colour in ("red", "blue"):
        try:
            row = ladder._slot_map(facts, online, colour)
        except ValueError as error:
            if "mask is empty in at least one camera" not in str(error):
                raise
            maps[colour] = {"colour": colour, "quarantined": str(error)}
            continue
        row["camera_support"] = facts.camera_support[0, ..., 0].detach().float().cpu().tolist()
        _, _, cameras, height, width = facts.object_to_chart.shape
        rgb = online.observation.raw_rgb[0, -1].detach().float()
        mask = torch.nn.functional.interpolate(
            ladder._colour_mask(rgb, colour)[:, None], size=(height, width), mode="area"
        )[:, 0]
        coordinates = facts.camera_coordinates[0, int(row["target_k"])].detach().float()
        inside = []
        for camera in range(cameras):
            xy = coordinates[camera]
            legal = bool(torch.isfinite(xy).all() and (xy.abs() <= 1).all())
            if legal:
                x = int(torch.round((xy[0] + 1) * (width - 1) / 2))
                y = int(torch.round((xy[1] + 1) * (height - 1) / 2))
                legal = bool(mask[camera, y, x] > 0)
            inside.append(legal)
        row["coordinate_in_colour_region_per_camera"] = inside
        maps[colour] = row
    return cache, state, maps, {key: values[0] for key, values in captures.items()}


def _state_change_components(
    policy: ClearVLACheckpointPolicy,
    cache: Any,
    *,
    target_k: int | None,
    control_k: int | None,
    runtime_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Rebuild the exact frozen S state-change algebra without changing weights."""

    model = policy.bundle.model
    organizer = model.intent.organizer
    history = cache.top.intent.history_tokens
    observable = cache.history
    _, observed_state_delta = organizer._paired_history(
        observable.state_history,
        observable.state,
        observable.executed_action_history,
    )
    batch = int(history.shape[0])
    if batch != 1:
        raise ValueError("object-index intervention requires batch one")
    enabled = policy.device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=policy.device.type,
            dtype=runtime_dtype,
            enabled=enabled,
        ),
    ):
        state_change_values = organizer.state_change_input(observed_state_delta)
        self_motion, _ = organizer.state_change_read(
            organizer.state_change_query_norm(
                organizer.state_change_query.to(
                    device=history.device,
                    dtype=history.dtype,
                ).expand(batch, -1, -1)
            ),
            organizer.state_change_key_norm(history),
            state_change_values,
            need_weights=True,
            average_attn_weights=True,
        )
        self_motion = self_motion[:, 0]

        validity = torch.nan_to_num(
            cache.top.belief.validity.to(
                device=history.device,
                dtype=torch.float32,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        object_mask = validity > 0.0
        transport_prior = cache.top.belief.transport_prior.to(
            device=history.device,
            dtype=cache.top.intent.object_tokens.dtype,
        )
        transport_prior = torch.where(
            object_mask.to(device=transport_prior.device),
            transport_prior,
            torch.zeros_like(transport_prior),
        )
        transport_tokens = organizer.state_change_transport(transport_prior)
        transport_validity = validity.to(
            device=transport_tokens.device,
            dtype=transport_tokens.dtype,
        )
        denominator = transport_validity.sum(dim=1).clamp_min(1.0)
        global_transport = (transport_tokens * transport_validity).sum(dim=1) / denominator

        def native_contribution(index: int) -> torch.Tensor:
            return transport_tokens[:, index] * transport_validity[:, index] / denominator

        def conditioned_mean(index: int) -> torch.Tensor:
            local_validity = transport_validity[:, index]
            return transport_tokens[:, index] * local_validity / local_validity.clamp_min(1.0)

        def fuse(self_value: torch.Tensor, transport_value: torch.Tensor) -> torch.Tensor:
            evidence, _ = smooth_rms_contract(
                organizer.state_change_fuse(torch.cat((self_value, transport_value), dim=-1)),
                0.20,
            )
            return evidence

        zero_self = torch.zeros_like(self_motion)
        zero_transport = torch.zeros_like(global_transport)
        evidence = {
            "baseline": fuse(self_motion, global_transport),
            "self_motion_zero": fuse(zero_self, global_transport),
            "object_transport_zero": fuse(self_motion, zero_transport),
            "both_zero": fuse(zero_self, zero_transport),
        }
        object_components = {}
        if target_k is not None and control_k is not None:
            count = int(transport_tokens.shape[1])
            if not (0 <= target_k < count and 0 <= control_k < count and target_k != control_k):
                raise ValueError("object interventions require two distinct valid K indices")
            target_contribution = native_contribution(target_k)
            control_contribution = native_contribution(control_k)
            non_target_contribution = global_transport - target_contribution
            target_mean = conditioned_mean(target_k)
            control_mean = conditioned_mean(control_k)
            object_components = {
                "target_native_contribution": target_contribution,
                "control_native_contribution": control_contribution,
                "non_target_native_contribution": non_target_contribution,
                "target_conditioned_mean": target_mean,
                "control_conditioned_mean": control_mean,
            }
            evidence.update(
                {
                    "target_contribution_zero": fuse(self_motion, non_target_contribution),
                    "control_contribution_zero": fuse(
                        self_motion, global_transport - control_contribution
                    ),
                    "non_target_contribution_zero": fuse(self_motion, target_contribution),
                    "target_conditioned_mean": fuse(self_motion, target_mean),
                    "control_conditioned_mean": fuse(self_motion, control_mean),
                }
            )
    result = {
        "observed_state_delta": observed_state_delta,
        "self_motion": self_motion,
        "transport_prior": transport_prior,
        "transport_tokens": transport_tokens,
        "transport_validity": transport_validity,
        "global_denominator": denominator,
        "per_k_native_contribution": transport_tokens * transport_validity / denominator[:, None],
        "global_transport": global_transport,
        **object_components,
        **{f"evidence__{key}": value for key, value in evidence.items()},
    }
    for name, value in result.items():
        _finite(value, name)
    if torch.count_nonzero(evidence["both_zero"]):
        raise RuntimeError("zero components no longer produce exact zero evidence")
    return result


def _cache_with_evidence(cache: Any, evidence: torch.Tensor, config: Any) -> Any:
    _finite(evidence, "replacement evidence")
    original = cache.top.intent.state_change_evidence
    if (
        evidence.shape != original.shape
        or evidence.dtype != original.dtype
        or evidence.device != original.device
    ):
        raise ValueError("replacement evidence changed shape/dtype/device")
    intent = replace(cache.top.intent, state_change_evidence=evidence)
    intent.validate(
        horizon=config.dimensions.action_horizon,
        hidden=config.dimensions.hidden_size,
    )
    top = replace(cache.top, intent=intent)
    candidate = replace(cache, top=top)
    candidate.validate(config)
    _same_fields(cache.top.intent, intent, except_names={"state_change_evidence"})
    _same_fields(cache.top, top, except_names={"intent"})
    _same_fields(cache, candidate, except_names={"top"})
    return candidate


def _run_mode(
    policy: ClearVLACheckpointPolicy,
    cache: Any,
    *,
    evidence: torch.Tensor | None,
    initial_noise: torch.Tensor,
) -> dict[str, Any]:
    config = policy.bundle.config
    model = policy.bundle.model
    mode_cache = cache if evidence is None else _cache_with_evidence(cache, evidence, config)
    expected = mode_cache.top.intent.state_change_evidence
    expected_hash = _tree_sha256(expected)
    cache_hash = _tree_sha256(cache)
    mode_cache_hash = _tree_sha256(mode_cache)
    noise_hash = _tree_sha256(initial_noise)
    plan_compiler = model.policy_compiler.plan_compiler
    traces: dict[str, list[torch.Tensor]] = {"proposal": [], "refined": []}
    witnesses: list[dict[str, Any]] = []
    active_pass = "proposal"
    pending: dict[str, Any] | None = None
    original_velocity = model.velocity
    had_local_velocity = "velocity" in model.__dict__
    local_velocity = model.__dict__.get("velocity")
    dynamics = model.world.dynamics
    original_w2 = dynamics.forward_w2
    had_local_w2 = "forward_w2" in dynamics.__dict__
    local_w2 = dynamics.__dict__.get("forward_w2")
    w2_calls = 0
    training_flags = [(module, module.training) for module in model.modules()]

    def audited_velocity(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending
        if pending is not None:
            raise RuntimeError("velocity/P3 calls are not one-to-one")
        observed_cache = args[0] if args else kwargs["cache"]
        if observed_cache.top.intent.state_change_evidence is not expected:
            raise RuntimeError("velocity consumed stale state-change evidence")
        context = kwargs.get("flow_step_context")
        if context is None:
            raise RuntimeError("Q5 lost FlowStepContext")
        pending = {
            "pass_role": active_pass,
            "node_index": len(traces[active_pass]),
            "time": _clone(context.time).tolist(),
            "step_size": _clone(context.step_size).tolist(),
            "endpoint": _clone(context.endpoint).tolist(),
            "world_generation": "initial" if active_pass == "proposal" else "rebuilt",
        }
        return original_velocity(*args, **kwargs)

    def plan_pre(_module: Any, _args: Any, kwargs: Any) -> None:
        used = kwargs["intent"].state_change_evidence
        if pending is None or used is not expected or _tree_sha256(used) != expected_hash:
            raise RuntimeError("P3 did not consume the designated new evidence")

    def plan_hook(_module: Any, _args: Any, output: Any) -> None:
        nonlocal pending
        if pending is None:
            raise RuntimeError("P3 call outside audited Q5 node")
        plan = output[0] if isinstance(output, tuple) else output
        _finite(plan.state_change, "P3 state-change lane")
        traces[active_pass].append(_clone(plan.state_change))
        witnesses.append(
            {
                **pending,
                "evidence_sha256": expected_hash,
                "state_change_sha256": _tree_sha256(plan.state_change),
                "state_change_rms": _rms(plan.state_change),
            }
        )
        pending = None

    def audited_w2(*args: Any, **kwargs: Any) -> Any:
        nonlocal w2_calls
        w2_calls += 1
        if kwargs.get("facts") is not mode_cache.top.belief:
            raise RuntimeError("W rebuild did not use the original compact belief")
        return original_w2(*args, **kwargs)

    pre_hook = plan_compiler.register_forward_pre_hook(plan_pre, with_kwargs=True)
    hook = plan_compiler.register_forward_hook(plan_hook)
    model.velocity = audited_velocity
    dynamics.forward_w2 = audited_w2
    try:
        with _preserved_rng(), torch.no_grad():
            proposal = sample_cached_action(
                model,
                mode_cache,
                config,
                initial_physical_noise=initial_noise.clone(),
                collect_diagnostics=False,
                pass_role="proposal",
            )
            world_action = model.outlet_adapter.world_condition_action_from_deployed(
                proposal.action,
                command=proposal.gripper_command,
            )
            refined_cache, _ = refine_cached_world(
                model, mode_cache, world_action, config, collect_diagnostics=True
            )
            _same_fields(mode_cache, refined_cache, except_names={"top"})
            _same_fields(mode_cache.top, refined_cache.top, except_names={"candidate_world"})
            if (
                refined_cache.top.intent is not mode_cache.top.intent
                or refined_cache.top.candidate_world is mode_cache.top.candidate_world
            ):
                raise RuntimeError("rebuild must retain changed S and replace the candidate world")
            active_pass = "refined"
            refined = sample_cached_action(
                model,
                refined_cache,
                config,
                initial_physical_noise=proposal.initial_physical_noise,
                collect_diagnostics=True,
                pass_role="refined",
            )
            for result in (proposal, refined):
                if _tree_sha256(result.initial_physical_noise) != noise_hash:
                    raise RuntimeError("Q5 changed the matched initial noise")
    finally:
        pre_hook.remove()
        hook.remove()
        if had_local_velocity:
            model.velocity = local_velocity
        else:
            del model.velocity
        if had_local_w2:
            dynamics.forward_w2 = local_w2
        else:
            del dynamics.forward_w2
        for module, training in training_flags:
            module.training = training
    if pending is not None or len(witnesses) != 12 or any(len(v) != 6 for v in traces.values()):
        raise RuntimeError("Q5/Q5 must consume the new S at six nodes per pass, twelve total")
    for role in ("proposal", "refined"):
        rows = [row for row in witnesses if row["pass_role"] == role]
        if [row["endpoint"] for row in rows] != [[0.0]] * 5 + [[1.0]]:
            raise RuntimeError("Q5 endpoint consumption changed")
        grid = resolve_deployment_flow_schedule(config.runtime.deployment_flow_schedule).identity[
            role
        ]["boundaries"]
        if len(grid) != 6 or not np.allclose(
            [row["time"][0] for row in rows], grid, atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError("Q5 node times differ from registered boundaries")
        expected_dt = np.r_[np.diff(np.asarray(grid)), 0.0]
        if not np.allclose(
            [row["step_size"][0] for row in rows], expected_dt, atol=1e-7, rtol=1e-6
        ):
            raise RuntimeError("Q5 node step sizes differ from registered boundaries")
    if w2_calls != 1:
        raise RuntimeError(f"expected one W2 rebuild, observed {w2_calls}")
    if (
        _tree_sha256(cache) != cache_hash
        or _tree_sha256(mode_cache) != mode_cache_hash
        or _tree_sha256(initial_noise) != noise_hash
    ):
        raise RuntimeError("lifecycle mutated an input cache or matched noise")
    return {
        "proposal_action": _clone(proposal.action),
        "refined_action": _clone(refined.action),
        "proposal_command": None
        if proposal.gripper_command is None
        else _clone(proposal.gripper_command),
        "refined_command": None
        if refined.gripper_command is None
        else _clone(refined.gripper_command),
        "p3_state_change_rms": _trace_rms(traces["proposal"] + traces["refined"]),
        "p3_per_pass_rms": {role: _trace_rms(values) for role, values in traces.items()},
        "p3_calls": len(witnesses),
        "node_witnesses": witnesses,
        "p3_trace_sha256": _tree_sha256(traces),
        "cache_sha256": cache_hash,
        "world_rebuilds": w2_calls,
        "rebuilt_world_sha256": _tree_sha256(refined_cache.top.candidate_world),
        "initial_noise_sha256": noise_hash,
        "hook_and_velocity_restored": True,
    }


def _sham_report(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("proposal_action", "refined_action", "proposal_command", "refined_command")
    if any(_tree_sha256(candidate[key]) != _tree_sha256(baseline[key]) for key in keys):
        raise RuntimeError("identity sham changed native action inputs or terminal commands")
    for key in ("p3_trace_sha256", "rebuilt_world_sha256", "initial_noise_sha256"):
        if candidate[key] != baseline[key]:
            raise RuntimeError(f"identity sham changed {key}")
    return {
        "passed": True,
        "bitwise_equal": True,
        "native_xyz_epsilon": 0.0,
        "native_xyz_tau": 0.01,
        "action_and_gripper_and_p3_and_world_checked": True,
    }


def _native_reference(policy: Any, cache: Any, initial_noise: torch.Tensor) -> dict[str, Any]:
    """Uninstrumented native outer lifecycle guards the diagnostic's orchestration."""
    model = policy.bundle.model
    flags = [(module, module.training) for module in model.modules()]
    before_cache, before_noise = _tree_sha256(cache), _tree_sha256(initial_noise)
    try:
        with _preserved_rng(), torch.no_grad():
            result, refined_cache = sample_refined_cached_action_with_cache(
                model,
                cache,
                policy.bundle.config,
                initial_physical_noise=initial_noise.clone(),
                collect_diagnostics=True,
            )
    finally:
        for module, flag in flags:
            module.training = flag
    if _tree_sha256(cache) != before_cache or _tree_sha256(initial_noise) != before_noise:
        raise RuntimeError("native reference mutated its cache or noise")
    return {
        "refined_action": _clone(result.action),
        "refined_command": None
        if result.gripper_command is None
        else _clone(result.gripper_command),
        "rebuilt_world_sha256": _tree_sha256(refined_cache.top.candidate_world),
    }


def _mode_report(
    policy: ClearVLACheckpointPolicy,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "p3_state_change_rms": candidate["p3_state_change_rms"],
        "p3_calls": candidate["p3_calls"],
        "p3_per_pass_rms": candidate["p3_per_pass_rms"],
        "node_witnesses": candidate["node_witnesses"],
        "world_rebuilds": candidate["world_rebuilds"],
        "rebuilt_world_sha256": candidate["rebuilt_world_sha256"],
        "proposal": ladder._action_summary(candidate["proposal_action"]),
        "refined": ladder._action_summary(candidate["refined_action"]),
        "proposal_delta_vs_baseline": ladder._action_delta(
            candidate["proposal_action"],
            baseline["proposal_action"],
        ),
        "refined_delta_vs_baseline": ladder._action_delta(
            candidate["refined_action"],
            baseline["refined_action"],
        ),
        "decoded_proposal": _decoded_summary(
            policy, candidate["proposal_action"], candidate["proposal_command"]
        ),
        "decoded_refined": _decoded_summary(
            policy, candidate["refined_action"], candidate["refined_command"]
        ),
        "decoded_refined_delta_vs_baseline": _decoded_delta(
            policy,
            candidate["refined_action"],
            baseline["refined_action"],
            candidate["refined_command"],
            baseline["refined_command"],
        ),
    }


@_preserved_rng()
def run(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    rollout_dir: Path,
    output: Path,
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    variant_name: str,
    minimum_ownership_margin: float,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if not math.isfinite(minimum_ownership_margin) or minimum_ownership_margin <= 0:
        raise ValueError("ownership margin must be finite and positive")
    summary = ladder._closed_loop_summary(rollout_dir / "summary.json")
    variants = {str(row["name"]): dict(row) for row in summary["variants"]}
    if variant_name not in variants:
        raise ValueError(f"unknown variant {variant_name!r}")
    variant = variants[variant_name]
    instruction = str(summary["instruction"])
    if "red" not in instruction.lower().split():
        raise ValueError(
            "this preselected red-target/blue-control probe requires a red instruction"
        )
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch.device(device),
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    source_identity = _source_identity(policy, repo_root)
    schedule = resolve_deployment_flow_schedule(
        policy.bundle.config.runtime.deployment_flow_schedule
    )
    if (
        any(
            schedule.identity.get(role, {}).get("parameters", {}).get("candidate_id") != "Q5"
            for role in ("proposal", "refined")
        )
        or schedule.identity.get("physical_nfe") != 10
        or schedule.identity.get("endpoint_head_calls") != 2
    ):
        raise ValueError("P3 component probe requires Q5/Q5, ten updates and two endpoints")

    phases: dict[str, Any] = {}
    model = policy.bundle.model
    with torch.no_grad(), _frozen_model(model) as invariants:
        for phase in ("initial", "selected_low"):
            snapshot, snapshot_record = ladder._snapshot_path(
                rollout_dir,
                variant,
                phase,
            )
            history = ladder._history(snapshot)
            cache, training_state, maps, producer = _encode_with_components(
                policy, history, instruction, runtime_dtype
            )
            target_map = maps["red"]
            control_map = maps["blue"]
            target_gate = _identity_row_ok(
                target_map,
                minimum_margin=minimum_ownership_margin,
            )
            control_gate = _identity_row_ok(
                control_map,
                minimum_margin=minimum_ownership_margin,
            )
            target_k = int(target_map["target_k"]) if target_gate else None
            control_k = int(control_map["target_k"]) if control_gate else None
            object_identity_gate = bool(target_gate and control_gate and target_k != control_k)
            components = _state_change_components(
                policy,
                cache,
                target_k=target_k if object_identity_gate else None,
                control_k=control_k if object_identity_gate else None,
                runtime_dtype=runtime_dtype,
            )
            cached_evidence = cache.top.intent.state_change_evidence
            producer_parity = {
                name: _parity(components[name], producer[name], name)
                for name in ("self_motion", "transport_prior", "transport_tokens")
            }
            producer_parity["global_transport"] = _parity(
                components["global_transport"],
                producer["fuse_input"].chunk(2, dim=-1)[1],
                "global transport",
            )
            evidence_parity = _parity(
                components["evidence__baseline"], cached_evidence, "cached evidence"
            )

            modes = list(GLOBAL_MODES)
            if object_identity_gate:
                modes.extend(OBJECT_MODES)
            generator = torch.Generator(device=policy.device).manual_seed(NOISE_SEED)
            initial_noise = policy.bundle.model.outlet_adapter.sample_noise(
                cache.history.batch,
                device=policy.device,
                dtype=cache.history.action_state.dtype,
                generator=generator,
            )
            native_reference = _native_reference(policy, cache, initial_noise)
            baseline = _run_mode(policy, cache, evidence=None, initial_noise=initial_noise)
            for key in ("refined_action", "refined_command", "rebuilt_world_sha256"):
                if _tree_sha256(native_reference[key]) != _tree_sha256(baseline[key]):
                    raise RuntimeError(
                        f"instrumented cached baseline differs from native lifecycle: {key}"
                    )
            identity = _run_mode(
                policy,
                cache,
                evidence=cached_evidence.detach().clone(),
                initial_noise=initial_noise,
            )
            identity_sham = _sham_report(identity, baseline)
            if evidence_parity["bitwise_equal"]:
                reconstructed_sham = {**identity_sham, "reused_identity_sham": True}
            else:
                reconstructed = _run_mode(
                    policy,
                    cache,
                    evidence=components["evidence__baseline"],
                    initial_noise=initial_noise,
                )
                reconstructed_sham = {
                    **_sham_report(reconstructed, baseline),
                    "reused_identity_sham": False,
                }
            runs = {
                "baseline": baseline,
                **{
                    mode: _run_mode(
                        policy,
                        cache,
                        evidence=components[f"evidence__{mode}"],
                        initial_noise=initial_noise,
                    )
                    for mode in modes
                    if mode != "baseline"
                },
            }
            # Floating per-K division need not sum bitwise to BF16 sum-then-divide.
            # Audit that identity in FP32 and expose the native rounding residual.
            tokens = components["transport_tokens"].float()
            weights = components["transport_validity"].float()
            denominator = components["global_denominator"].float()
            contributions32 = tokens * weights / denominator[:, None]
            global32 = (tokens * weights).sum(dim=1) / denominator
            closure = _parity(contributions32.sum(dim=1), global32, "FP32 K contribution closure")
            closure["native_rounding_residual_max_abs"] = _max_abs(
                components["per_k_native_contribution"].float().sum(dim=1)
                - components["global_transport"].float()
            )

            def clean_map(value: Any) -> Any:
                if isinstance(value, dict):
                    return {k: clean_map(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [clean_map(v) for v in value]
                return None if isinstance(value, float) and not math.isfinite(value) else value

            phases[phase] = {
                "snapshot": snapshot_record,
                "slot_maps": clean_map(maps),
                "cache_sha256": _tree_sha256(cache),
                "initial_noise": {
                    "seed": NOISE_SEED,
                    "sha256": _tree_sha256(initial_noise),
                    "shape": list(initial_noise.shape),
                    "dtype": str(initial_noise.dtype),
                },
                "identity_gate": {
                    "minimum_ownership_margin": minimum_ownership_margin,
                    "target_pass": target_gate,
                    "control_pass": control_gate,
                    "distinct_k": bool(target_gate and control_gate and target_k != control_k),
                    "object_specific_interventions_run": object_identity_gate,
                    "gate_meaning": "conservative color-region ownership proxy, not certified simulator entity identity",
                    "missing_or_nonfinite_map_fields": "reject object modes; nonfinite audit values serialized as null",
                },
                "component_rms": {
                    name: _rms(value)
                    for name, value in components.items()
                    if not name.startswith("evidence__")
                },
                "component_arrays": {
                    name: _clone(value).tolist()
                    for name, value in components.items()
                    if name
                    in {
                        "self_motion",
                        "global_transport",
                        "transport_prior",
                        "transport_tokens",
                        "transport_validity",
                        "global_denominator",
                        "per_k_native_contribution",
                    }
                },
                "producer_parity": producer_parity,
                "evidence_parity": evidence_parity,
                "native_contribution_closure": closure,
                "identity_sham": identity_sham,
                "native_baseline_sham": {
                    "passed": True,
                    "bitwise_equal": True,
                    "unmodified_native_lifecycle": True,
                },
                "reconstructed_sham": reconstructed_sham,
                "online_metrics": {
                    "object_transport_prior_rms_after_object_mask": _rms(
                        components["transport_prior"]
                    ),
                    "state_change_evidence_rms": _rms(
                        training_state.top.intent.state_change_evidence
                    ),
                },
                "baseline": _mode_report(policy, baseline, baseline),
                "interventions": {
                    mode: _mode_report(policy, runs[mode], baseline)
                    for mode in modes
                    if mode != "baseline"
                },
            }
    result = {
        "schema": SCHEMA,
        "diagnostic_not_benchmark_score": True,
        "checkpoint": {
            "path": str(policy.bundle.checkpoint_path.resolve()),
            "sha256": policy.bundle.checkpoint_sha256,
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "snapshot_source": {
            "rollout_dir": str(rollout_dir.resolve()),
            "variant": variant_name,
            "instruction": instruction,
            "trajectory_diagnostics": variant.get("trajectory_diagnostics"),
        },
        "flow_schedule": schedule.identity,
        "source_identity": source_identity,
        "invariants": {**invariants, "rng_restored_on_exit": True},
        "phases": phases,
        "interpretation_contract": {
            "global_modes": (
                "exact pre-fusion component removals with all other cached facts, "
                "Q5 noise and weights fixed"
            ),
            "native_object_contribution": (
                "one K contribution under the original all-valid-object denominator; "
                "FP32 closure and native BF16 rounding residual are reported separately. "
                "The blue single-K control is distinct from removing ALL non-target K rows."
            ),
            "conditioned_mean": (
                "privileged color-selected K token * validity / max(validity,1). "
                "With binary validity this removes the global K-count dilution; fractional "
                "validity remains a gain, so it is not a mathematical conditional mean. "
                "It changes denominator/amplitude and cannot isolate learned task relevance."
            ),
            "identity_quarantine": (
                "object-specific modes are omitted unless candidate ownership, coordinate "
                "agreement in every camera, null/competitor margin, supported coordinates "
                "inside the color region, permutation and distinct-K gates pass"
            ),
            "verdict_ceiling": "total-path sensitivity only; no expert, reverse donor, matched target-y response or six independent scenes",
            "native_command_units": "CALVIN relative commands before environment clipping, not metres or mm gain",
            "protocol_scope": "two preselected phases, one seed; no D/F localization or harmfulness/candidate-admission test",
            "no_completion_rule": True,
            "weights_updated": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--variant-name", default="target_y_minus_040mm")
    parser.add_argument("--minimum-ownership-margin", type=float, default=0.02)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Frozen checkout; defaults to the executed checkpoint module's root",
    )
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        rollout_dir=args.rollout_dir,
        output=args.output,
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        variant_name=str(args.variant_name),
        minimum_ownership_margin=float(args.minimum_ownership_margin),
        repo_root=args.repo_root,
    )
    print(json.dumps({"schema": result["schema"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
