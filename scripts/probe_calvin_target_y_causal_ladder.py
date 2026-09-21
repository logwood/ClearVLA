#!/usr/bin/env python3
"""Trace target identity through P2 and the complete Q5 deployment lifecycle.

This is a checkpoint diagnostic, not a benchmark or a training path.  A
colour mask is used only to identify the simulator's known target object in
the learned K slots.  The mask is never provided to the policy.  Every causal
intervention runs proposal ODE -> one W rebuild -> refined ODE from the same
initial physical noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from clearvla.mainline.model import compiler as compiler_module
from clearvla.mainline.runtime.flow_schedule import resolve_deployment_flow_schedule
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import (
    refine_cached_world,
    sample_cached_action,
)
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from scripts.probe_calvin_internal_layers import _build_online

try:
    from scripts.probe_calvin_target_y_shift_internal import (
        _history,
        _snapshot_path,
    )
    from scripts.probe_calvin_target_y_shift_internal import (
        _summary as _closed_loop_summary,
    )
except ModuleNotFoundError:
    # The approved remote diagnostic staging directory keeps the two probe
    # files beside each other instead of modifying the source checkout.
    from probe_calvin_target_y_shift_internal import (  # type: ignore[no-redef]
        _history,
        _snapshot_path,
    )
    from probe_calvin_target_y_shift_internal import (
        _summary as _closed_loop_summary,
    )


SCHEMA = "clearvla-calvin-target-y-causal-ladder-v1"
COLORS = ("red", "blue")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _tensor_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"tensor shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return _rms(left.detach().float().cpu() - right.detach().float().cpu())


def _trace_rmse(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    if len(left) != len(right):
        raise ValueError(f"trace length mismatch: {len(left)} vs {len(right)}")
    numerator = 0.0
    count = 0
    for a, b in zip(left, right, strict=True):
        if tuple(a.shape) != tuple(b.shape):
            raise ValueError("trace tensor shape mismatch")
        delta = a.double() - b.double()
        numerator += float(delta.square().sum())
        count += int(delta.numel())
    return float(np.sqrt(numerator / max(count, 1)))


def _colour_mask(rgb: torch.Tensor, colour: str) -> torch.Tensor:
    """Return a conservative policy-input-space mask [C,R,R]."""

    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError("RGB must be [C,3,R,R]")
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if colour == "red":
        mask = (red > 0.55) & ((red - green) > 0.22) & ((red - blue) > 0.22)
    elif colour == "blue":
        mask = (blue > 0.42) & ((blue - red) > 0.18) & ((blue - green) > 0.10)
    else:
        raise ValueError(f"unknown colour {colour!r}")
    return mask.float()


def _slot_map(facts: Any, online: Any, colour: str) -> dict[str, Any]:
    facts.validate()
    read_chart = facts.object_to_chart.detach().float()
    # Physical K identity is the joint local K+null competition.  The
    # object-to-chart tensor is a K-conditioned read posterior and therefore
    # cannot, by itself, say which K owns a coloured pixel.
    ownership = facts.candidate_assignment.detach().float().sum(dim=-1)
    if int(read_chart.shape[0]) != 1 or tuple(ownership.shape) != tuple(read_chart.shape):
        raise ValueError("causal ladder requires batch one")
    _, objects, cameras, height, width = read_chart.shape
    rgb = online.observation.raw_rgb[0, -1].detach().float()
    mask = _colour_mask(rgb, colour)
    resized = F.interpolate(
        mask[:, None], size=(height, width), mode="area"
    )[:, 0]
    if bool((resized.sum(dim=(-2, -1)) <= 0).any()):
        raise ValueError(f"{colour} mask is empty in at least one camera")

    posterior_mass = read_chart[0].sum(dim=(-2, -1)).clamp_min(1e-8)
    read_overlap = (read_chart[0] * resized[None]).sum(dim=(-2, -1)) / posterior_mass
    mask_denominator = resized.sum(dim=(-2, -1)).clamp_min(1e-8)
    ownership_overlap = (
        (ownership[0] * resized[None]).sum(dim=(-2, -1))
        / mask_denominator[None]
    )
    null_ownership = facts.null_assignment.detach().float().sum(dim=-1)[0]
    null_overlap = (
        (null_ownership * resized).sum(dim=(-2, -1)) / mask_denominator
    )
    validity = facts.camera_validity[0, ..., 0].detach().float()
    combined = (ownership_overlap * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)

    grid_y = torch.linspace(-1.0, 1.0, height, device=resized.device)
    grid_x = torch.linspace(-1.0, 1.0, width, device=resized.device)
    y_mesh, x_mesh = torch.meshgrid(grid_y, grid_x, indexing="ij")
    mask_mass = resized.sum(dim=(-2, -1)).clamp_min(1e-8)
    centroid = torch.stack(
        (
            (resized * x_mesh).sum(dim=(-2, -1)) / mask_mass,
            (resized * y_mesh).sum(dim=(-2, -1)) / mask_mass,
        ),
        dim=-1,
    )
    coordinates = facts.camera_coordinates[0].detach().float()
    distance = (coordinates - centroid[None]).square().sum(dim=-1).sqrt()
    distance_combined = (distance * validity).sum(dim=-1) / validity.sum(dim=-1).clamp_min(1.0)

    overlap_order = torch.argsort(combined, descending=True)
    distance_order = torch.argsort(distance_combined)
    target = int(overlap_order[0])
    camera_argmax: list[int] = []
    camera_agreement: list[bool] = []
    for camera in range(cameras):
        legal = validity[:, camera] > 0.0
        camera_score = torch.where(
            legal,
            ownership_overlap[:, camera],
            torch.full_like(ownership_overlap[:, camera], -1.0),
        )
        camera_target = int(camera_score.argmax())
        camera_argmax.append(camera_target)
        camera_agreement.append(
            bool(legal[target]) and camera_target == int(overlap_order[0])
        )
    margin = float(
        (combined[overlap_order[0]] - combined[overlap_order[1]]).cpu()
        if objects > 1
        else combined[overlap_order[0]].cpu()
    )
    permutation = torch.arange(objects - 1, -1, -1, device=read_chart.device)
    permuted = facts.permute(permutation)
    permuted_ownership = permuted.candidate_assignment.detach().float().sum(dim=-1)[0]
    permuted_overlap = (
        (permuted_ownership * resized[None]).sum(dim=(-2, -1))
        / mask_denominator[None]
    )
    permuted_validity = permuted.camera_validity[0, ..., 0].detach().float()
    permuted_combined = (permuted_overlap * permuted_validity).sum(dim=-1) / (
        permuted_validity.sum(dim=-1).clamp_min(1.0)
    )
    expected = combined.index_select(0, permutation)
    permutation_error = float((permuted_combined - expected).abs().amax().cpu())
    expected_target = int((permutation == target).nonzero(as_tuple=False)[0, 0])
    actual_target = int(permuted_combined.argmax())

    return {
        "colour": colour,
        "target_k": target,
        "ownership_scores": combined.cpu().tolist(),
        "ownership_per_camera": ownership_overlap.cpu().tolist(),
        "ownership_argmax_per_camera": camera_argmax,
        "null_ownership_per_camera": null_overlap.cpu().tolist(),
        "object_to_chart_overlap_per_camera": read_overlap.cpu().tolist(),
        "valid_camera_argmax_agrees": camera_agreement,
        "all_valid_cameras_agree": bool(all(camera_agreement)),
        "ownership_margin": margin,
        "coordinate_nearest_k": int(distance_order[0]),
        "coordinate_distances": distance_combined.cpu().tolist(),
        "camera_coordinates": coordinates.cpu().tolist(),
        "mask_centroid": centroid.cpu().tolist(),
        "mask_mass": resized.sum(dim=(-2, -1)).cpu().tolist(),
        "camera_validity": validity.cpu().tolist(),
        "permutation": permutation.cpu().tolist(),
        "permuted_target_k": actual_target,
        "expected_permuted_target_k": expected_target,
        "permutation_score_max_abs_error": permutation_error,
        "ownership_and_coordinate_agree": target == int(distance_order[0]),
    }


def _attention_target_mass(intent: Any, target_k: int) -> dict[str, Any]:
    value = intent.interval_object_attention.detach().float()
    if int(value.shape[-1]) != int(intent.object_tokens.shape[1]):
        raise ValueError("S object attention no longer ends in K")
    target = value[..., target_k]
    return {
        "shape": list(value.shape),
        "target_k": int(target_k),
        "target_mass_mean": float(target.mean().cpu()),
        "target_mass_by_interval": target.reshape(-1, target.shape[-1]).mean(dim=0).cpu().tolist()
        if target.ndim > 1
        else target.cpu().tolist(),
        "argmax_by_interval": value.reshape(-1, value.shape[-2], value.shape[-1])
        .mean(dim=0)
        .argmax(dim=-1)
        .cpu()
        .tolist(),
    }


def _encode(
    policy: ClearVLACheckpointPolicy,
    history: Any,
    instruction: str,
    *,
    runtime_dtype: torch.dtype,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    online = _build_online(policy, history, instruction)
    device = policy.device
    enabled = device.type in {"cuda", "cpu"} and runtime_dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=runtime_dtype, enabled=enabled
    ):
        cache, training_state, _ = policy.bundle.model.encode_online(
            online,
            training_mask=False,
            geometry_supervision=False,
            collect_diagnostics=True,
        )
    facts = training_state.top.facts
    facts.validate()
    maps = {colour: _slot_map(facts, online, colour) for colour in COLORS}
    metadata = {
        "instruction": instruction,
        "slot_maps": maps,
        "s_target_attention": {
            colour: _attention_target_mass(
                training_state.top.intent, int(maps[colour]["target_k"])
            )
            for colour in COLORS
        },
    }
    return cache, training_state, metadata, maps


def _posterior_target_mass(
    values: list[dict[str, torch.Tensor]], target_k: int, cameras: int
) -> dict[str, Any]:
    semantic = [item["semantic"] for item in values]
    geometry = [item["geometry"] for item in values]
    semantic_mass = torch.stack([value[..., target_k].mean() for value in semantic])
    geometry_mass = torch.stack(
        [
            value.reshape(*value.shape[:-1], -1, cameras)[..., target_k, :]
            .sum(dim=-1)
            .mean()
            for value in geometry
        ]
    )
    return {
        "calls": len(values),
        "semantic_target_mass_mean": float(semantic_mass.mean()),
        "geometry_target_mass_mean": float(geometry_mass.mean()),
        "semantic_target_mass_by_call": semantic_mass.tolist(),
        "geometry_target_mass_by_call": geometry_mass.tolist(),
    }


def _action_summary(action: torch.Tensor) -> dict[str, Any]:
    value = action.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "rms": _rms(value),
        "first_row": value[0, 0].tolist(),
        "first4_mean": value[0, :4].mean(dim=0).tolist(),
        "first8_mean": value[0, :8].mean(dim=0).tolist(),
    }


def _action_delta(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, Any]:
    left = candidate.detach().float().cpu()
    right = baseline.detach().float().cpu()
    delta = left - right
    bands = ((0, 4), (4, 12), (12, 24))
    return {
        "rmse": _rms(delta),
        "arm_rmse": _rms(delta[..., :-1]),
        "gripper_rmse": _rms(delta[..., -1:]),
        "y_rmse": _rms(delta[..., 1:2]),
        "first_row_delta": delta[0, 0].tolist(),
        "band_arm_rmse": {
            f"{start + 1}-{end}": _rms(delta[:, start:end, :-1])
            for start, end in bands
        },
    }


def _clone_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().clone()


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _save_tensor_artifact(
    path: Path,
    *,
    variants: list[dict[str, Any]],
    runtime: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {}
    for variant in variants:
        name = str(variant["name"])
        prefix = name.replace("-", "_")
        facts = runtime[name]["facts"]
        dynamics = runtime[name]["cache"].top.predicted_dynamics
        capture = baseline[name]["captures"]
        payload.update(
            {
                f"{prefix}__object_to_chart": _numpy(facts.object_to_chart),
                f"{prefix}__candidate_assignment": _numpy(
                    facts.candidate_assignment
                ),
                f"{prefix}__null_assignment": _numpy(facts.null_assignment),
                f"{prefix}__camera_coordinates": _numpy(facts.camera_coordinates),
                f"{prefix}__camera_transport_prior": _numpy(
                    facts.camera_transport_prior
                ),
                f"{prefix}__camera_support": _numpy(facts.camera_support),
                f"{prefix}__camera_validity": _numpy(facts.camera_validity),
                f"{prefix}__w_transport_mean": _numpy(dynamics.transport_mean),
                f"{prefix}__w_transport_common": _numpy(
                    dynamics.transport_common
                ),
                f"{prefix}__w_transport_interval_innovation": _numpy(
                    dynamics.transport_interval_innovation
                ),
                f"{prefix}__w_transport_covariance": _numpy(
                    dynamics.transport_covariance
                ),
                f"{prefix}__w_camera_chart_availability": _numpy(
                    dynamics.camera_chart_availability
                ),
                f"{prefix}__p2_geometry_spatial_posterior": np.stack(
                    [item["geometry"].numpy() for item in capture["posterior"]],
                    axis=0,
                ),
                f"{prefix}__p2_geometry_selected_preterminal": np.stack(
                    [item.numpy() for item in capture["selected_geometry_used"]],
                    axis=0,
                ),
                f"{prefix}__p2_geometry_selected_common": np.stack(
                    [item.numpy() for item in capture["selected_geometry_common"]],
                    axis=0,
                ),
                f"{prefix}__p2_geometry_selected_residual": np.stack(
                    [item.numpy() for item in capture["selected_geometry_residual"]],
                    axis=0,
                ),
                f"{prefix}__p2_geometry_effect": np.stack(
                    [item.numpy() for item in capture["effect_geometry_used"]],
                    axis=0,
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "keys": sorted(payload),
        "bytes": int(path.stat().st_size),
    }


def _run_lifecycle(
    policy: ClearVLACheckpointPolicy,
    cache: Any,
    *,
    initial_noise: torch.Tensor,
    target_k: int,
    mode: str,
    donor: Mapping[str, list[torch.Tensor]] | None = None,
) -> dict[str, Any]:
    model = policy.bundle.model
    config = policy.bundle.config
    reader = model.policy_compiler.effect_reader
    consequence_module = model.policy_compiler.consequence
    objects = int(cache.top.belief.content.shape[1])
    cameras = int(cache.top.belief.camera_coordinates.shape[2])
    original_softmax = compiler_module._safe_masked_softmax
    original_spatial = reader.spatial_select
    original_forward = reader.forward
    captures: dict[str, Any] = {
        "posterior": [],
        "selected_geometry_pre": [],
        "selected_geometry_used": [],
        "selected_semantic_used": [],
        "effect_geometry_pre": [],
        "effect_geometry_used": [],
        "effect_semantic_used": [],
        "consequence": [],
    }
    pending: dict[str, torch.Tensor] = {}
    spatial_call = 0
    effect_call = 0

    if mode not in {"baseline", "oracle_target", "geometry_read_transplant", "geometry_effect_transplant"}:
        raise ValueError(f"unknown mode {mode!r}")
    if "transplant" in mode and donor is None:
        raise ValueError(f"{mode} requires a donor trace")

    def patched_softmax(
        logit: torch.Tensor, support: torch.Tensor, *, dim: int
    ) -> torch.Tensor:
        probability = original_softmax(logit, support, dim=dim)
        if dim == -1 and logit.ndim == 5 and int(logit.shape[-1]) in {
            objects,
            objects * cameras,
        }:
            kind = "semantic" if int(logit.shape[-1]) == objects else "geometry"
            used = probability
            if mode == "oracle_target":
                restricted = torch.zeros_like(support, dtype=torch.bool)
                if kind == "semantic":
                    restricted[..., target_k] = support[..., target_k]
                else:
                    start = target_k * cameras
                    stop = start + cameras
                    restricted[..., start:stop] = support[..., start:stop]
                if not bool(restricted.any(dim=-1).all()):
                    raise RuntimeError("oracle target K is unsupported in a P2 row")
                used = original_softmax(logit, restricted, dim=-1)
            pending[kind] = _clone_cpu(used)
            if kind == "geometry":
                if "semantic" not in pending:
                    raise RuntimeError("geometry posterior arrived before semantic posterior")
                captures["posterior"].append(
                    {"semantic": pending.pop("semantic"), "geometry": pending.pop("geometry")}
                )
            return used
        return probability

    def patched_spatial(
        action_query: torch.Tensor,
        dynamics: Any,
        intent: Any,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        nonlocal spatial_call
        selected, metrics = original_spatial(
            action_query,
            dynamics,
            intent,
            collect_diagnostics=collect_diagnostics,
        )
        captures["selected_geometry_pre"].append(_clone_cpu(selected.geometry_value))
        if mode == "geometry_read_transplant":
            assert donor is not None
            if spatial_call >= len(donor["selected_geometry_common"]):
                raise RuntimeError("geometry-read donor trace is too short")
            common = donor["selected_geometry_common"][spatial_call].to(
                device=selected.geometry_common_value.device,
                dtype=selected.geometry_common_value.dtype,
            )
            residual = donor["selected_geometry_residual"][spatial_call].to(
                device=selected.geometry_residual_value.device,
                dtype=selected.geometry_residual_value.dtype,
            )
            selected = replace(
                selected,
                geometry_common_value=common,
                geometry_residual_value=residual,
                geometry_value=common + residual,
            )
        captures.setdefault("selected_geometry_common", []).append(
            _clone_cpu(selected.geometry_common_value)
        )
        captures.setdefault("selected_geometry_residual", []).append(
            _clone_cpu(selected.geometry_residual_value)
        )
        captures["selected_geometry_used"].append(_clone_cpu(selected.geometry_value))
        captures["selected_semantic_used"].append(_clone_cpu(selected.semantic_value))
        spatial_call += 1
        return selected, metrics

    def patched_forward(
        action_query: torch.Tensor,
        dynamics: Any,
        intent: Any,
        *,
        collect_diagnostics: bool,
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        nonlocal effect_call
        effect, metrics = original_forward(
            action_query,
            dynamics,
            intent,
            collect_diagnostics=collect_diagnostics,
        )
        captures["effect_geometry_pre"].append(_clone_cpu(effect.geometry))
        if mode == "geometry_effect_transplant":
            assert donor is not None
            if effect_call >= len(donor["effect_geometry_used"]):
                raise RuntimeError("geometry-effect donor trace is too short")
            geometry = donor["effect_geometry_used"][effect_call].to(
                device=effect.geometry.device,
                dtype=effect.geometry.dtype,
            )
            effect = replace(effect, geometry=geometry)
        captures["effect_geometry_used"].append(_clone_cpu(effect.geometry))
        captures["effect_semantic_used"].append(_clone_cpu(effect.semantic))
        effect_call += 1
        return effect, metrics

    def consequence_hook(_module: Any, _args: Any, output: Any) -> None:
        consequence = output[0] if isinstance(output, tuple) else output
        captures["consequence"].append(_clone_cpu(consequence.protected_consequence))

    compiler_module._safe_masked_softmax = patched_softmax
    reader.spatial_select = patched_spatial  # type: ignore[method-assign]
    reader.forward = patched_forward  # type: ignore[method-assign]
    hook = consequence_module.register_forward_hook(consequence_hook)
    try:
        proposal = sample_cached_action(
            model,
            cache,
            config,
            initial_physical_noise=initial_noise.clone(),
            collect_diagnostics=False,
            pass_role="proposal",
        )
        proposal_world_action = model.outlet_adapter.world_condition_action_from_deployed(
            proposal.action,
            command=proposal.gripper_command,
        )
        refined_cache, refinement_metrics = refine_cached_world(
            model,
            cache,
            proposal_world_action,
            config,
            collect_diagnostics=True,
        )
        refined = sample_cached_action(
            model,
            refined_cache,
            config,
            initial_physical_noise=proposal.initial_physical_noise,
            collect_diagnostics=True,
            pass_role="refined",
        )
    finally:
        hook.remove()
        reader.forward = original_forward  # type: ignore[method-assign]
        reader.spatial_select = original_spatial  # type: ignore[method-assign]
        compiler_module._safe_masked_softmax = original_softmax

    if pending:
        raise RuntimeError(f"unpaired spatial posteriors remain: {tuple(pending)}")
    expected_calls = 2 * (int(config.runtime.inference_steps) + 1)
    for name in (
        "posterior",
        "selected_geometry_used",
        "effect_geometry_used",
        "consequence",
    ):
        if len(captures[name]) != expected_calls:
            raise RuntimeError(
                f"{mode} {name} trace has {len(captures[name])}, expected {expected_calls}"
            )
    return {
        "mode": mode,
        "proposal_action": _clone_cpu(proposal.action),
        "refined_action": _clone_cpu(refined.action),
        "captures": captures,
        "posterior_target_mass": _posterior_target_mass(
            captures["posterior"], target_k, cameras
        ),
        "refinement_metrics": {
            str(key): float(value.detach().float().cpu())
            for key, value in refinement_metrics.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        },
    }


def _mode_report(mode: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    capture = mode["captures"]
    reference = baseline["captures"]
    return {
        "mode": mode["mode"],
        "proposal": _action_summary(mode["proposal_action"]),
        "refined": _action_summary(mode["refined_action"]),
        "proposal_delta_vs_baseline": _action_delta(
            mode["proposal_action"], baseline["proposal_action"]
        ),
        "refined_delta_vs_baseline": _action_delta(
            mode["refined_action"], baseline["refined_action"]
        ),
        "posterior_target_mass": mode["posterior_target_mass"],
        "trace_rmse_vs_baseline": {
            name: _trace_rmse(capture[name], reference[name])
            for name in (
                "selected_geometry_pre",
                "selected_geometry_used",
                "selected_semantic_used",
                "effect_geometry_pre",
                "effect_geometry_used",
                "effect_semantic_used",
                "consequence",
            )
        },
    }


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    rollout_dir: Path,
    output: Path,
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
    map_only: bool,
    tensor_output: Path | None,
) -> dict[str, Any]:
    closed_loop = _closed_loop_summary(rollout_dir / "summary.json")
    variants = [dict(item) for item in closed_loop["variants"]]
    if len(variants) != 2:
        raise ValueError("causal ladder requires exactly two target-y variants")
    instruction = str(closed_loop["instruction"])
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch.device(device),
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    if str(closed_loop["checkpoint"]["sha256"]) != policy.bundle.checkpoint_sha256:
        raise ValueError("checkpoint SHA does not match the closed-loop rollout")
    runtime_dtype = resolve_compute_dtype(policy.bundle.config, None)
    schedule = resolve_deployment_flow_schedule(
        policy.bundle.config.runtime.deployment_flow_schedule
    )
    schedule_parameters = schedule.identity.get("proposal", {}).get("parameters", {})
    if schedule_parameters.get("candidate_id") != "Q5":
        raise ValueError("causal ladder requires the deployed Q5 schedule")

    runtime: dict[str, Any] = {}
    records: dict[str, Any] = {}
    with torch.no_grad():
        for variant in variants:
            name = str(variant["name"])
            snapshot, snapshot_record = _snapshot_path(rollout_dir, variant, "initial")
            history = _history(snapshot)
            cache, training_state, metadata, maps = _encode(
                policy,
                history,
                instruction,
                runtime_dtype=runtime_dtype,
            )
            runtime[name] = {
                "cache": cache,
                "history": history,
                "maps": maps,
                "facts": training_state.top.facts,
                "intent": training_state.top.intent,
            }
            records[name] = {
                "snapshot": snapshot_record,
                **metadata,
            }

        result: dict[str, Any] = {
            "schema": SCHEMA,
            "diagnostic_not_benchmark_score": True,
            "oracle_masks_are_diagnostic_only": True,
            "checkpoint": {
                "path": str(policy.bundle.checkpoint_path.resolve()),
                "sha256": policy.bundle.checkpoint_sha256,
                "epoch": int(policy.bundle.epoch),
                "global_step": int(policy.bundle.global_step),
            },
            "rollout": {
                "directory": str(rollout_dir.resolve()),
                "summary_sha256": _sha256(rollout_dir / "summary.json"),
                "instruction": instruction,
            },
            "flow_schedule": schedule.identity,
            "fixed_noise_seed": 9182026,
            "records": records,
            "map_only": bool(map_only),
        }
        mapping_rows = [
            records[str(variant["name"])]["slot_maps"][colour]
            for variant in variants
            for colour in COLORS
        ]
        red_mapping_rows = [
            records[str(variant["name"])]["slot_maps"]["red"]
            for variant in variants
        ]

        def mapping_check(rows: list[dict[str, Any]]) -> dict[str, bool]:
            check = {
                "ownership_coordinate_agreement": bool(
                    all(row["ownership_and_coordinate_agree"] for row in rows)
                ),
                "per_camera_ownership_agreement": bool(
                    all(row["all_valid_cameras_agree"] for row in rows)
                ),
                "permutation_equivariance": bool(
                    all(
                        row["permutation_score_max_abs_error"] <= 1e-7
                        and row["permuted_target_k"]
                        == row["expected_permuted_target_k"]
                        for row in rows
                    )
                ),
            }
            check["pass"] = bool(all(check.values()))
            return check

        result["mapping_gate"] = {
            "red_target_y": mapping_check(red_mapping_rows),
            "all_recorded_colours": mapping_check(mapping_rows),
        }
        if map_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result

        first_cache = runtime[str(variants[0]["name"])]["cache"]
        generator = torch.Generator(device=policy.device).manual_seed(9182026)
        initial_noise = policy.bundle.model.outlet_adapter.sample_noise(
            first_cache.history.batch,
            device=policy.device,
            dtype=first_cache.history.action_state.dtype,
            generator=generator,
        )

        baseline: dict[str, Any] = {}
        identity: dict[str, Any] = {}
        for variant in variants:
            name = str(variant["name"])
            target_k = int(runtime[name]["maps"]["red"]["target_k"])
            baseline[name] = _run_lifecycle(
                policy,
                runtime[name]["cache"],
                initial_noise=initial_noise,
                target_k=target_k,
                mode="baseline",
            )
            repeat = _run_lifecycle(
                policy,
                runtime[name]["cache"],
                initial_noise=initial_noise,
                target_k=target_k,
                mode="baseline",
            )
            identity[name] = {
                "proposal_max_abs": float(
                    (repeat["proposal_action"] - baseline[name]["proposal_action"]).abs().amax()
                ),
                "refined_max_abs": float(
                    (repeat["refined_action"] - baseline[name]["refined_action"]).abs().amax()
                ),
            }
            if identity[name]["refined_max_abs"] > 1e-7:
                raise RuntimeError(f"baseline replay identity failed for {name}")

        resolved_tensor_output = (
            tensor_output
            if tensor_output is not None
            else output.with_name(f"{output.stem}_tensors.npz")
        )
        result["tensor_artifact"] = _save_tensor_artifact(
            resolved_tensor_output,
            variants=variants,
            runtime=runtime,
            baseline=baseline,
        )

        mode_reports: dict[str, Any] = {}
        for index, variant in enumerate(variants):
            name = str(variant["name"])
            donor_name = str(variants[1 - index]["name"])
            target_k = int(runtime[name]["maps"]["red"]["target_k"])
            runs = {
                "baseline": baseline[name],
                "geometry_read_transplant": _run_lifecycle(
                    policy,
                    runtime[name]["cache"],
                    initial_noise=initial_noise,
                    target_k=target_k,
                    mode="geometry_read_transplant",
                    donor=baseline[donor_name]["captures"],
                ),
                "geometry_effect_transplant": _run_lifecycle(
                    policy,
                    runtime[name]["cache"],
                    initial_noise=initial_noise,
                    target_k=target_k,
                    mode="geometry_effect_transplant",
                    donor=baseline[donor_name]["captures"],
                ),
            }
            if result["mapping_gate"]["red_target_y"]["pass"]:
                runs["oracle_target"] = _run_lifecycle(
                    policy,
                    runtime[name]["cache"],
                    initial_noise=initial_noise,
                    target_k=target_k,
                    mode="oracle_target",
                )
            mode_reports[name] = {
                "donor_variant": donor_name,
                "target_k": target_k,
                "oracle_target_status": (
                    "executed"
                    if "oracle_target" in runs
                    else "skipped_unreliable_physical_k_mapping"
                ),
                "modes": {
                    key: _mode_report(value, baseline[name])
                    for key, value in runs.items()
                },
            }

        first_name = str(variants[0]["name"])
        second_name = str(variants[1]["name"])
        cross_variant = {
            "baseline_refined_action": _action_delta(
                baseline[second_name]["refined_action"],
                baseline[first_name]["refined_action"],
            ),
            "baseline_trace_rmse": {
                name: _trace_rmse(
                    baseline[second_name]["captures"][name],
                    baseline[first_name]["captures"][name],
                )
                for name in (
                    "selected_geometry_used",
                    "selected_semantic_used",
                    "effect_geometry_used",
                    "effect_semantic_used",
                    "consequence",
                )
            },
        }

        # Fixed observation/history/noise, with only the target noun changed.
        language_variant = variants[1]
        language_name = str(language_variant["name"])
        blue_instruction = instruction.replace("red", "blue")
        if blue_instruction == instruction:
            raise ValueError("red instruction could not be converted to blue")
        blue_cache, blue_training, blue_metadata, blue_maps = _encode(
            policy,
            runtime[language_name]["history"],
            blue_instruction,
            runtime_dtype=runtime_dtype,
        )
        blue_target_k = int(blue_maps["blue"]["target_k"])
        blue_run = _run_lifecycle(
            policy,
            blue_cache,
            initial_noise=initial_noise,
            target_k=blue_target_k,
            mode="baseline",
        )
        language_swap = {
            "variant": language_name,
            "red_instruction": instruction,
            "blue_instruction": blue_instruction,
            "red_target_k": int(runtime[language_name]["maps"]["red"]["target_k"]),
            "blue_target_k": blue_target_k,
            "blue_metadata": blue_metadata,
            "refined_action_delta": _action_delta(
                blue_run["refined_action"], baseline[language_name]["refined_action"]
            ),
            "red_posterior_target_mass": baseline[language_name]["posterior_target_mass"],
            "blue_posterior_target_mass": blue_run["posterior_target_mass"],
            "s_interval_object_attention_rmse": _tensor_rmse(
                blue_training.top.intent.interval_object_attention,
                runtime[language_name]["intent"].interval_object_attention,
            ),
        }

        result.update(
            {
                "baseline_replay_identity": identity,
                "variants": mode_reports,
                "cross_variant": cross_variant,
                "fixed_scene_language_swap": language_swap,
                "interpretation_contract": {
                    "oracle_target": (
                        "restricts the existing P2 spatial posterior to the simulator-identified "
                        "target K; it adds no value, support, completion rule, or score"
                    ),
                    "geometry_read_transplant": (
                        "injects the other matched variant's actual selected geometry value, then "
                        "recomputes P2 effect, consequence, proposal, W rebuild and refined ODE"
                    ),
                    "geometry_effect_transplant": (
                        "injects the other matched variant's actual P2 geometry effect, then "
                        "recomputes consequence, proposal, W rebuild and refined ODE"
                    ),
                    "scope": (
                        "checkpoint-backed frozen causal diagnostic; oracle/transplant rows are "
                        "not normal-policy scores and do not authorize hard-coded task logic"
                    ),
                },
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--tensor-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        rollout_dir=args.rollout_dir,
        output=args.output,
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        map_only=bool(args.map_only),
        tensor_output=args.tensor_output,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "map_only": result["map_only"],
                "output": str(args.output),
                "records": {
                    name: {
                        colour: value["target_k"]
                        for colour, value in record["slot_maps"].items()
                    }
                    for name, record in result["records"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
