#!/usr/bin/env python3
"""Run matched-noise H2 or H3 causal tests on frozen CALVIN validation windows.

This is a diagnostic harness, not a training entrypoint and not a benchmark.
H2 changes only the S ``_paired_history`` producer.  H3 changes only one
keyword passed to the static P1 factual reader.  Every condition then executes
the normal proposal Q5, one W rebuild, and the refined Q5 with the same initial
physical noise.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.gripper_contract import (
    CALVIN_BINARY_GRIPPER_OUTPUT_MODE,
    CONTINUOUS_GRIPPER_OUTPUT_MODE,
    VALID_GRIPPER_OUTPUT_MODES,
    is_binary_gripper_mode,
)
from clearvla.mainline.runtime.flow_schedule import resolve_deployment_flow_schedule
from clearvla.mainline.runtime.identity import dataset_identity, language_identity
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import (
    refine_cached_world,
    sample_cached_action,
)
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy

SCHEMA = "clearvla-calvin-h2-h3-q5-v1"
FROZEN_CHECKPOINTS = {
    "6e137830dab5920f62ddfbe0af9fc3ac0a1d7c112365a2fc80ed7fc4d962c755": "camera",
    "63581c8f8d45bf9a14a316c6582afacd2e52cecc2671f599dd4ddffc5a0d10c6": "sequence",
}
H2_CONDITIONS = (
    "baseline",
    "identity",
    "action_clock",
    "observed_aligned",
    "action_permutation",
)
H3_CONDITIONS = (
    "baseline",
    "identity",
    "goal_interval",
    "goal_permutation",
    "goal_mean_only",
    "goal_row3_only",
    "history_interval",
    "history_permutation",
    "history_mean_only",
    "history_row3_only",
)
ACTION_PERMUTATION = (1, 2, 3, 4, 5, 6, 0, 7)
INTERVAL_PERMUTATION = (1, 2, 0, 3)
INTERVAL_WEIGHTS = (4.0 / 24.0, 8.0 / 24.0, 12.0 / 24.0, 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, Tensor):
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
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        else:
            digest.update(repr(item).encode())
        digest.update(b"\x00")

    visit(value)
    return digest.hexdigest()


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _max_abs(value: Tensor) -> float:
    return float(value.detach().float().abs().amax())


def _finite(value: Tensor, name: str) -> None:
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be nonempty and finite")


def _clone_cpu(value: Tensor) -> Tensor:
    return value.detach().float().cpu().contiguous()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _model_sha256(model: nn.Module) -> str:
    return _tree_sha256(
        {
            "parameters": dict(model.named_parameters()),
            "buffers_including_nonpersistent": dict(model.named_buffers()),
        }
    )


def _same_fields(left: Any, right: Any, *, except_names: set[str]) -> None:
    if type(left) is not type(right) or not is_dataclass(left):
        raise TypeError("cache identity audit requires matching dataclass instances")
    for field in fields(left):
        if field.name not in except_names and getattr(left, field.name) is not getattr(
            right, field.name
        ):
            raise RuntimeError(f"unrelated cache field replaced: {field.name}")


def _module_origins(repo_root: Path, model: nn.Module) -> dict[str, dict[str, str]]:
    root = repo_root.expanduser().resolve()
    objects: dict[str, tuple[Any, str]] = {
        "checkpoint_policy": (
            ClearVLACheckpointPolicy,
            "clearvla/simulation/clearvla_policy.py",
        ),
        "mainline_policy": (type(model), "clearvla/mainline/model/policy.py"),
        "intent_organizer": (
            type(model.intent.organizer),
            "clearvla/mainline/model/intent.py",
        ),
        "p1_stage": (type(model.p1), "clearvla/mainline/model/components.py"),
        "policy_compiler_stage": (
            type(model.policy_compiler),
            "clearvla/mainline/model/components.py",
        ),
        "p3_plan_compiler": (
            type(model.policy_compiler.plan_compiler),
            "clearvla/mainline/model/compiler.py",
        ),
        "sample_cached_action": (
            sample_cached_action,
            "clearvla/mainline/runtime/sampling.py",
        ),
        "refine_cached_world": (
            refine_cached_world,
            "clearvla/mainline/runtime/sampling.py",
        ),
    }
    result: dict[str, dict[str, str]] = {}
    unwrap = getattr(inspect, "unwrap", lambda value: value)
    for name, (value, expected) in objects.items():
        unwrapped = unwrap(value)
        path = Path(inspect.getfile(unwrapped)).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"{name} was imported outside the frozen checkout: {path}") from exc
        if relative != expected:
            raise RuntimeError(f"{name} source path changed: expected={expected} actual={relative}")
        result[name] = {
            "path": str(path),
            "relative_path": relative,
            "decorated_path": str(Path(inspect.getfile(value)).resolve()),
            "decorator_unwrapped": unwrapped is not value,
        }
    return result


@contextmanager
def _frozen_model(model: nn.Module):
    flags = [(module, bool(module.training)) for module in model.modules()]
    before = _model_sha256(model)
    record: dict[str, Any] = {"model_before_sha256": before}
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


def _source_action(cache: Any) -> Tensor:
    condition = cache.top.action_condition
    if hasattr(condition, "source_action"):
        return condition.source_action
    if hasattr(condition, "source_interval_action"):
        return condition.source_interval_action
    raise TypeError("unknown physical action condition")


def _tensor_delta(candidate: Tensor, baseline: Tensor) -> dict[str, Any]:
    left = candidate.detach().float()
    right = baseline.detach().float()
    if left.shape != right.shape:
        raise ValueError("delta tensors have different shapes")
    delta = left - right
    return {
        "shape": list(delta.shape),
        "rmse": _rms(delta),
        "max_abs": _max_abs(delta),
        "candidate_rms": _rms(left),
        "baseline_rms": _rms(right),
        "bitwise_equal": _tree_sha256(candidate) == _tree_sha256(baseline),
    }


def _cache_tensors(cache: Any) -> dict[str, Tensor]:
    intent = cache.top.intent
    dynamics = cache.top.predicted_dynamics
    result = {
        "s_history": intent.history_tokens,
        "s_public_interval": intent.public_interval_carrier,
        "s_policy_interval": intent.policy_interval_context,
        "s_state_change": intent.state_change_evidence,
        "s_typed_common": intent.typed_common_value,
        "s_typed_interval": intent.typed_interval_residual_value,
        "coarse_action": _source_action(cache),
        "w_semantic_delta": dynamics.semantic_delta,
        "w_transport_mean": dynamics.transport_mean,
        "p1_static_detail": cache.factual_dock.protected_detail,
    }
    for name, value in result.items():
        _finite(value, name)
    return result


def _decode_action(
    policy: ClearVLACheckpointPolicy,
    action: Tensor,
    command: Tensor | None,
) -> np.ndarray:
    normalized = action.detach().float().cpu().numpy()
    decoded = policy.bundle.action_normalizer.decode(normalized).astype(np.float32)
    if decoded.ndim != 3 or decoded.shape[-1] != 7:
        raise ValueError("decoded CALVIN action must be [B,24,7]")
    mode = str(policy.bundle.config.bottom.gripper_output_mode)
    if mode not in VALID_GRIPPER_OUTPUT_MODES:
        raise ValueError(f"unknown gripper output mode {mode!r}")
    if mode not in {
        CONTINUOUS_GRIPPER_OUTPUT_MODE,
        CALVIN_BINARY_GRIPPER_OUTPUT_MODE,
    }:
        raise ValueError(f"non-CALVIN gripper output mode {mode!r} is not admissible")
    if is_binary_gripper_mode(mode):
        if command is None:
            raise ValueError("binary CALVIN output lost its terminal command")
        values = command.detach().float().cpu().numpy()
        if values.shape != decoded.shape[:-1] or not np.isin(values, [-1.0, 1.0]).all():
            raise ValueError("binary terminal command has the wrong schema")
        decoded[..., -1] = values
    elif command is not None:
        raise ValueError("continuous CALVIN output unexpectedly exposed a binary command")
    if not np.isfinite(decoded).all():
        raise ValueError("decoded action contains nonfinite values")
    return decoded


def _decoded_report(
    policy: ClearVLACheckpointPolicy,
    run: Mapping[str, Any],
    expert: np.ndarray,
) -> dict[str, Any]:
    proposal = _decode_action(policy, run["proposal_action"], run["proposal_command"])
    refined = _decode_action(policy, run["refined_action"], run["refined_command"])
    if expert.shape != refined.shape:
        raise ValueError("expert and decoded model action shapes differ")
    first = refined[:, 0]
    expert_first = expert[:, 0]
    xyz_error = np.square(first[:, :3] - expert_first[:, :3]).sum(axis=-1)
    rotation_error = np.square(first[:, 3:6] - expert_first[:, 3:6]).sum(axis=-1)
    gripper_error = first[:, 6] != expert_first[:, 6]
    return {
        "proposal_first_rows": proposal[:, 0].tolist(),
        "refined_first_rows": first.tolist(),
        "expert_first_rows": expert_first.tolist(),
        "xyz_squared_error": xyz_error.tolist(),
        "rotation_squared_error": rotation_error.tolist(),
        "gripper_mismatch": gripper_error.tolist(),
        "xyz_error_mean": float(xyz_error.mean()),
        "rotation_error_mean": float(rotation_error.mean()),
        "gripper_mismatch_count": int(gripper_error.sum()),
    }


def _restore_instance_attribute(
    owner: Any,
    name: str,
    *,
    had_local: bool,
    local_value: Any,
) -> None:
    if had_local:
        setattr(owner, name, local_value)
    elif name in owner.__dict__:
        delattr(owner, name)


def _trace_delta(
    candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(candidate) != len(baseline):
        raise ValueError("trace lengths differ")
    names = (
        "p1_factual_base",
        "p1_policy_residual",
        "p2_semantic",
        "p2_geometry",
        "p2_consequence",
        "p3_temporal",
        "p3_state_change",
    )
    result: dict[str, Any] = {}
    for name in names:
        rows = [
            _tensor_delta(left[name], right[name])
            for left, right in zip(candidate, baseline, strict=True)
        ]
        result[name] = {
            "node_rmse": [row["rmse"] for row in rows],
            "max_node_rmse": max(row["rmse"] for row in rows),
            "max_abs": max(row["max_abs"] for row in rows),
            "all_nodes_bitwise_equal": all(row["bitwise_equal"] for row in rows),
        }
    return result


def _run_q5(
    model: nn.Module,
    cache: Any,
    config: Any,
    *,
    initial_noise: Tensor | None,
    generator: torch.Generator | None,
) -> dict[str, Any]:
    plan_compiler = model.policy_compiler.plan_compiler
    dynamics = model.world.dynamics
    original_velocity = model.velocity
    original_w2 = dynamics.forward_w2
    had_local_velocity = "velocity" in model.__dict__
    local_velocity = model.__dict__.get("velocity")
    had_local_w2 = "forward_w2" in dynamics.__dict__
    local_w2 = dynamics.__dict__.get("forward_w2")
    active_pass = "proposal"
    pending: dict[str, Any] | None = None
    traces: dict[str, list[dict[str, Any]]] = {"proposal": [], "refined": []}
    witnesses: list[dict[str, Any]] = []
    w2_calls = 0
    cache_hash = _tree_sha256(cache)
    supplied_noise_hash = None if initial_noise is None else _tree_sha256(initial_noise)

    def audited_velocity(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending
        if pending is not None:
            raise RuntimeError("velocity/compiler calls are not one-to-one")
        context = kwargs.get("flow_step_context")
        if context is None:
            raise RuntimeError("Q5 lost FlowStepContext")
        pending = {
            "pass_role": active_pass,
            "node_index": len(traces[active_pass]),
            "time": _clone_cpu(context.time).tolist(),
            "step_size": _clone_cpu(context.step_size).tolist(),
            "endpoint": _clone_cpu(context.endpoint).tolist(),
            "world_generation": "initial" if active_pass == "proposal" else "rebuilt",
        }
        return original_velocity(*args, **kwargs)

    def plan_pre(_module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal pending
        if pending is None:
            raise RuntimeError("P3 call occurred outside an audited velocity call")
        consequence = kwargs["consequence"]
        pending["tensors"] = {
            "p1_factual_base": _clone_cpu(consequence.factual_base),
            "p1_policy_residual": _clone_cpu(kwargs["p1_policy_residual"]),
            "p2_semantic": _clone_cpu(consequence.effect.semantic),
            "p2_geometry": _clone_cpu(consequence.effect.geometry),
            "p2_consequence": _clone_cpu(consequence.protected_consequence),
        }

    def plan_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        nonlocal pending
        if pending is None or "tensors" not in pending:
            raise RuntimeError("P3 output lost its audited inputs")
        plan = output[0] if isinstance(output, tuple) else output
        tensors = pending.pop("tensors")
        tensors.update(
            p3_temporal=_clone_cpu(plan.temporal),
            p3_state_change=_clone_cpu(plan.state_change),
        )
        traces[active_pass].append(tensors)
        witnesses.append({**pending})
        pending = None

    def audited_w2(*args: Any, **kwargs: Any) -> Any:
        nonlocal w2_calls
        w2_calls += 1
        if kwargs.get("facts") is not cache.top.belief:
            raise RuntimeError("W rebuild did not use the original compact belief")
        return original_w2(*args, **kwargs)

    with ExitStack() as cleanup:
        pre_hook = plan_compiler.register_forward_pre_hook(plan_pre, with_kwargs=True)
        cleanup.callback(pre_hook.remove)
        hook = plan_compiler.register_forward_hook(plan_hook)
        cleanup.callback(hook.remove)
        model.velocity = audited_velocity
        cleanup.callback(
            _restore_instance_attribute,
            model,
            "velocity",
            had_local=had_local_velocity,
            local_value=local_velocity,
        )
        dynamics.forward_w2 = audited_w2
        cleanup.callback(
            _restore_instance_attribute,
            dynamics,
            "forward_w2",
            had_local=had_local_w2,
            local_value=local_w2,
        )
        with torch.no_grad():
            proposal = sample_cached_action(
                model,
                cache,
                config,
                generator=generator,
                initial_physical_noise=(None if initial_noise is None else initial_noise.clone()),
                collect_diagnostics=False,
                pass_role="proposal",
            )
            world_action = model.outlet_adapter.world_condition_action_from_deployed(
                proposal.action,
                command=proposal.gripper_command,
            )
            refined_cache, _ = refine_cached_world(
                model,
                cache,
                world_action,
                config,
                collect_diagnostics=True,
            )
            _same_fields(cache, refined_cache, except_names={"top"})
            _same_fields(cache.top, refined_cache.top, except_names={"candidate_world"})
            if (
                refined_cache.top.intent is not cache.top.intent
                or refined_cache.factual_dock is not cache.factual_dock
                or refined_cache.transition_source is not cache.transition_source
                or refined_cache.top.candidate_world is cache.top.candidate_world
            ):
                raise RuntimeError("refined cache must retain S/P1/transition and replace only W")
            active_pass = "refined"
            refined = sample_cached_action(
                model,
                refined_cache,
                config,
                initial_physical_noise=proposal.initial_physical_noise,
                collect_diagnostics=True,
                pass_role="refined",
            )
    if (
        pending is not None
        or len(witnesses) != 12
        or any(len(rows) != 6 for rows in traces.values())
    ):
        raise RuntimeError("Q5/Q5 did not produce six audited nodes per pass")
    schedule = resolve_deployment_flow_schedule(config.runtime.deployment_flow_schedule)
    for role in ("proposal", "refined"):
        rows = [row for row in witnesses if row["pass_role"] == role]
        endpoints = [row["endpoint"] for row in rows]
        if not all(
            all(value == (1.0 if index == 5 else 0.0) for value in row)
            for index, row in enumerate(endpoints)
        ):
            raise RuntimeError("Q5 endpoint consumption changed")
        grid = schedule.identity[role]["boundaries"]
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
    noise_hash = _tree_sha256(proposal.initial_physical_noise)
    if supplied_noise_hash is not None and supplied_noise_hash != noise_hash:
        raise RuntimeError("proposal did not preserve supplied initial noise")
    if noise_hash != _tree_sha256(refined.initial_physical_noise):
        raise RuntimeError("refined Q5 did not reuse proposal noise")
    if _tree_sha256(cache) != cache_hash:
        raise RuntimeError("Q5 lifecycle mutated its input cache")
    return {
        "proposal_action": proposal.action.detach().cpu(),
        "refined_action": refined.action.detach().cpu(),
        "proposal_command": None
        if proposal.gripper_command is None
        else proposal.gripper_command.detach().cpu(),
        "refined_command": None
        if refined.gripper_command is None
        else refined.gripper_command.detach().cpu(),
        "initial_noise": proposal.initial_physical_noise.detach(),
        "initial_noise_sha256": noise_hash,
        "traces": traces["proposal"] + traces["refined"],
        "node_witnesses": witnesses,
        "world_rebuilds": w2_calls,
        "rebuilt_world_sha256": _tree_sha256(refined_cache.top.candidate_world),
    }


def _unwrap_dataset(dataset: Any) -> Any:
    seen: set[int] = set()
    current = dataset
    while hasattr(current, "base") and id(current) not in seen:
        seen.add(id(current))
        current = current.base
    required = ("refs", "episodes", "config", "state_normalizer")
    if not all(hasattr(current, name) for name in required):
        raise RuntimeError("validation dataset does not expose raw episode provenance")
    return current


def _instruction_direction(instruction: str) -> str:
    words = instruction.lower().replace("-", " ").split()
    hits = [value for value in ("left", "right") if value in words]
    if len(hits) != 1:
        return "unknown"
    return hits[0]


def _select_samples(
    loader: Any,
    count: int,
) -> tuple[Mapping[str, Tensor], list[dict[str, Any]]]:
    dataset = loader.dataset
    base = _unwrap_dataset(dataset)
    by_episode: dict[int, list[int]] = {}
    for index, ref in enumerate(base.refs):
        by_episode.setdefault(int(ref.episode_idx), []).append(index)
    episode_ids = sorted(by_episode)
    if len(episode_ids) < count:
        raise RuntimeError(f"validation split has {len(episode_ids)} episodes, need {count}")
    by_instruction: dict[str, list[int]] = {}
    for episode in episode_ids:
        instruction = base.episodes[episode].instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError("validation episode lacks a stable instruction identity")
        by_instruction.setdefault(instruction.strip(), []).append(episode)
    selected_episodes: list[int] = []
    depth = 0
    instructions = sorted(by_instruction)
    while len(selected_episodes) < count:
        added = False
        for instruction in instructions:
            rows = by_instruction[instruction]
            if depth < len(rows):
                selected_episodes.append(rows[depth])
                added = True
                if len(selected_episodes) == count:
                    break
        if not added:
            raise RuntimeError("unable to construct the requested episode-disjoint panel")
        depth += 1
    selected_instructions = [
        str(base.episodes[episode].instruction).strip() for episode in selected_episodes
    ]
    directions = {_instruction_direction(value) for value in selected_instructions}
    if count >= 6 and (
        len(set(selected_instructions)) < 3 or not {"left", "right"}.issubset(directions)
    ):
        raise RuntimeError(
            "six-scene admission requires at least three instruction tasks and both directions"
        )
    indices = [by_episode[episode][len(by_episode[episode]) // 2] for episode in selected_episodes]
    samples = [dataset[index] for index in indices]
    raw = loader.collate_fn(samples)
    selection = [
        {
            "sample_index": int(index),
            "episode_index": int(base.refs[index].episode_idx),
            "center_index": int(base.refs[index].center),
            "instruction": str(base.episodes[int(base.refs[index].episode_idx)].instruction),
            "direction": _instruction_direction(
                str(base.episodes[int(base.refs[index].episode_idx)].instruction)
            ),
        }
        for index in indices
    ]
    return raw, selection


def _aligned_history(
    base: Any,
    selection: Sequence[Mapping[str, Any]],
    online: Any,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    cfg = base.config
    offsets = tuple(int(value) for value in cfg.executed_action_offsets)
    if len(offsets) != int(online.history.executed_action_history.shape[1]):
        raise ValueError("executed action offsets do not align with online history")
    action_relative = tuple(
        int(cfg.action_offset) + value - int(cfg.state_offset) for value in offsets
    )
    if max(action_relative) >= 0:
        raise ValueError("observed-aligned H2 requires strictly past executed actions")
    scale = float(max(abs(min(action_relative)), 1))
    states: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    for row in selection:
        episode_index = int(row["episode_index"])
        center = int(row["center_index"])
        episode = base.episodes[episode_index]
        raw_states = episode.states_raw
        if raw_states is None:
            raise ValueError("raw episode is missing observed states")
        current_state_index = center + int(cfg.state_offset)
        requested = [current_state_index + relative for relative in action_relative]
        if min(requested) < 0 or max(requested) > current_state_index:
            raise ValueError("selected H2 window lacks complete causal observed history")
        native = np.asarray(raw_states[np.asarray(requested, dtype=np.int64)], dtype=np.float32)
        encoded = np.asarray(base.state_normalizer.encode(native), dtype=np.float32)
        states.append(encoded)
        provenance.append(
            {
                **dict(row),
                "current_state_index": current_state_index,
                "requested_state_indices": requested,
                "requested_relative_times": list(action_relative),
                "observed_or_padding": ["observed"] * len(requested),
            }
        )
    state_tensor = torch.from_numpy(np.stack(states)).to(
        device=online.history.state.device,
        dtype=online.history.state.dtype,
    )
    delta = torch.cat(
        (torch.zeros_like(state_tensor[:, :1]), state_tensor[:, 1:] - state_tensor[:, :-1]),
        dim=1,
    )
    actions = online.history.executed_action_history
    clock = torch.tensor(action_relative, device=actions.device, dtype=actions.dtype)[None, :, None]
    clock = clock.expand(actions.shape[0], -1, -1) / scale
    paired = torch.cat((state_tensor, actions, delta, clock), dim=-1)
    return (
        paired,
        delta,
        {
            "state_offset": int(cfg.state_offset),
            "action_offset": int(cfg.action_offset),
            "executed_action_offsets": list(offsets),
            "relative_action_clock": list(action_relative),
            "clock_denominator": scale,
            "samples": provenance,
        },
    )


@contextmanager
def _patched_instance_method(owner: Any, name: str, replacement: Callable[..., Any]):
    had_local = name in owner.__dict__
    local = owner.__dict__.get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if had_local:
            setattr(owner, name, local)
        else:
            delattr(owner, name)


def _encode(
    model: nn.Module,
    policy_input: Any,
    config: Any,
    dtype: torch.dtype,
) -> Any:
    enabled = policy_input.device.type in {"cpu", "cuda"} and dtype in {
        torch.bfloat16,
        torch.float16,
    }
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=policy_input.device.type,
            dtype=dtype,
            enabled=enabled,
        ),
    ):
        cache, training_state, _ = model.encode_online(
            policy_input,
            training_mask=False,
            geometry_supervision=False,
            collect_diagnostics=False,
        )
    del training_state
    cache.validate(config)
    return cache


def _h2_caches(
    model: nn.Module,
    batch: Any,
    base: Any,
    selection: Sequence[Mapping[str, Any]],
    config: Any,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    organizer = model.intent.organizer
    original = organizer._paired_history
    legacy_paired, legacy_delta = original(
        batch.online.history.state_history,
        batch.online.history.state,
        batch.online.history.executed_action_history,
    )
    aligned_paired, aligned_delta, provenance = _aligned_history(base, selection, batch.online)
    if aligned_paired.shape != legacy_paired.shape or aligned_delta.shape != legacy_delta.shape:
        raise RuntimeError("aligned H2 replacement changed the frozen tensor schema")
    relative = provenance["relative_action_clock"]
    scale = float(provenance["clock_denominator"])
    action_clock = legacy_paired.clone()
    clock = torch.tensor(relative, device=action_clock.device, dtype=action_clock.dtype)
    action_clock[..., -1] = (clock / scale)[None]
    permutation = torch.tensor(ACTION_PERMUTATION, device=aligned_paired.device, dtype=torch.long)
    action_start = int(batch.online.history.state.shape[-1])
    action_end = action_start + int(batch.online.history.executed_action_history.shape[-1])
    permuted_paired = aligned_paired.clone()
    permuted_paired[..., action_start:action_end] = aligned_paired[
        ..., action_start:action_end
    ].index_select(1, permutation)
    values = {
        "identity": (legacy_paired, legacy_delta),
        "action_clock": (action_clock, legacy_delta),
        "observed_aligned": (aligned_paired, aligned_delta),
        "action_permutation": (permuted_paired, aligned_delta),
    }
    caches = {"baseline": _encode(model, batch.online, config, dtype)}
    captures: dict[str, Any] = {}
    for condition in H2_CONDITIONS[1:]:
        used_paired, used_delta = values[condition]
        calls = 0

        def replacement(
            state_history: Tensor,
            state: Tensor,
            executed_history: Tensor,
            *,
            paired: Tensor = used_paired,
            delta: Tensor = used_delta,
        ) -> tuple[Tensor, Tensor]:
            nonlocal calls
            calls += 1
            if (
                _tree_sha256(state_history) != _tree_sha256(batch.online.history.state_history)
                or _tree_sha256(state) != _tree_sha256(batch.online.history.state)
                or _tree_sha256(executed_history)
                != _tree_sha256(batch.online.history.executed_action_history)
            ):
                raise RuntimeError("H2 wrapper was called for an unexpected online batch")
            return paired, delta

        with _patched_instance_method(organizer, "_paired_history", replacement):
            caches[condition] = _encode(model, batch.online, config, dtype)
        if calls != 1:
            raise RuntimeError(f"H2 {condition} replacement call count was {calls}, expected 1")
        captures[condition] = {
            "paired_sha256": _tree_sha256(used_paired),
            "delta_sha256": _tree_sha256(used_delta),
            "paired_delta_vs_legacy": _tensor_delta(used_paired, legacy_paired),
            "state_change_delta_vs_legacy": _tensor_delta(used_delta, legacy_delta),
        }
    captures["baseline"] = {
        "paired_sha256": _tree_sha256(legacy_paired),
        "delta_sha256": _tree_sha256(legacy_delta),
    }
    return caches, {
        "provenance": provenance,
        "action_permutation": list(ACTION_PERMUTATION),
        "conditions": captures,
    }


def _center_interval(value: Tensor) -> Tensor:
    weights = torch.tensor(INTERVAL_WEIGHTS, device=value.device, dtype=torch.float32)
    common = (value.float() * weights[None, :, None]).sum(dim=1, keepdim=True)
    return (value.float() - common).to(dtype=value.dtype)


def _h3_contexts(base: Tensor, innovation: Tensor, *, lane: str) -> dict[str, Tensor]:
    if base.shape != innovation.shape or tuple(base.shape[1:2]) != (4,):
        raise ValueError(f"H3 {lane} tensors must align as [B,4,H]")
    interval = (base.float() + _center_interval(innovation).float()).to(dtype=base.dtype)
    permutation = torch.tensor(INTERVAL_PERMUTATION, device=innovation.device, dtype=torch.long)
    permuted_innovation = innovation.index_select(1, permutation)
    permuted = (base.float() + _center_interval(permuted_innovation).float()).to(dtype=base.dtype)
    weights = torch.tensor(INTERVAL_WEIGHTS, device=base.device, dtype=torch.float32)
    actual_mean = (interval.float() * weights[None, :, None]).sum(dim=1, keepdim=True)
    mean_only = actual_mean.to(dtype=base.dtype).expand(-1, 4, -1).contiguous()
    row3 = base.clone()
    row3[:, 3] = interval[:, 3]
    return {
        f"{lane}_interval": interval,
        f"{lane}_permutation": permuted,
        f"{lane}_mean_only": mean_only,
        f"{lane}_row3_only": row3,
    }


def _h3_caches(
    model: nn.Module,
    batch: Any,
    config: Any,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    organizer = model.intent.organizer
    captured: dict[str, Tensor] = {}

    def goal_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        captured["goal_innovation"] = output[1].detach()

    def history_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        captured["history_innovation"] = output[1].detach()

    goal_handle = organizer.interval_goal.register_forward_hook(goal_hook)
    history_handle = organizer.interval_history.register_forward_hook(history_hook)
    try:
        baseline = _encode(model, batch.online, config, dtype)
    finally:
        goal_handle.remove()
        history_handle.remove()
    if set(captured) != {"goal_innovation", "history_innovation"}:
        raise RuntimeError("H3 failed to capture the two existing interval innovations")
    dock = baseline.top.intent.factual_dock()
    goal_contexts = _h3_contexts(
        dock.condition_query_context,
        captured["goal_innovation"],
        lane="goal",
    )
    history_contexts = _h3_contexts(
        dock.history_query_context,
        captured["history_innovation"],
        lane="history",
    )
    contexts = {**goal_contexts, **history_contexts}
    p1 = model.p1
    original = p1.build_static
    caches = {"baseline": baseline}
    records: dict[str, Any] = {
        "baseline": {
            "goal_context_sha256": _tree_sha256(dock.condition_query_context),
            "history_context_sha256": _tree_sha256(dock.history_query_context),
        }
    }
    for condition in H3_CONDITIONS[1:]:
        calls = 0
        used: dict[str, Tensor] = {}

        def replacement(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            changed = dict(kwargs)
            if condition.startswith("goal_"):
                changed["condition_query_context"] = contexts[condition]
            elif condition.startswith("history_"):
                changed["history_query_context"] = contexts[condition]
            elif condition != "identity":
                raise RuntimeError(f"unknown H3 condition {condition}")
            used["phase"] = changed["phase_context"].detach()
            used["goal"] = changed["condition_query_context"].detach()
            used["history"] = changed["history_query_context"].detach()
            return original(*args, **changed)

        with _patched_instance_method(p1, "build_static", replacement):
            caches[condition] = _encode(model, batch.online, config, dtype)
        if calls != 1 or set(used) != {"phase", "goal", "history"}:
            raise RuntimeError(f"H3 {condition} static P1 replacement did not execute exactly once")
        records[condition] = {
            "phase_context_sha256": _tree_sha256(used["phase"]),
            "goal_context_sha256": _tree_sha256(used["goal"]),
            "history_context_sha256": _tree_sha256(used["history"]),
            "goal_delta_vs_baseline": _tensor_delta(used["goal"], dock.condition_query_context),
            "history_delta_vs_baseline": _tensor_delta(used["history"], dock.history_query_context),
        }
    return caches, {
        "interval_weights": list(INTERVAL_WEIGHTS),
        "interval_permutation": list(INTERVAL_PERMUTATION),
        "goal_innovation_sha256": _tree_sha256(captured["goal_innovation"]),
        "history_innovation_sha256": _tree_sha256(captured["history_innovation"]),
        "conditions": records,
    }


def _identity_check(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "proposal_action",
        "refined_action",
        "proposal_command",
        "refined_command",
    )
    equal = all(_tree_sha256(candidate[key]) == _tree_sha256(baseline[key]) for key in keys)
    equal = equal and candidate["rebuilt_world_sha256"] == baseline["rebuilt_world_sha256"]
    if not equal:
        raise RuntimeError("identity or negative-control seam changed the complete Q5 result")
    trace = _trace_delta(candidate["traces"], baseline["traces"])
    if not all(value["all_nodes_bitwise_equal"] for value in trace.values()):
        raise RuntimeError("identity or negative-control seam changed an internal Q5 trace")
    return {"passed": True, "bitwise_equal": True, "native_xyz_epsilon": 0.0, "trace": trace}


def _condition_report(
    policy: ClearVLACheckpointPolicy,
    run: Mapping[str, Any],
    baseline: Mapping[str, Any],
    expert: np.ndarray,
) -> dict[str, Any]:
    return {
        "initial_noise_sha256": run["initial_noise_sha256"],
        "world_rebuilds": run["world_rebuilds"],
        "rebuilt_world_sha256": run["rebuilt_world_sha256"],
        "node_witnesses": run["node_witnesses"],
        "proposal_delta_vs_baseline": _tensor_delta(
            run["proposal_action"], baseline["proposal_action"]
        ),
        "refined_delta_vs_baseline": _tensor_delta(
            run["refined_action"], baseline["refined_action"]
        ),
        "internal_trace_delta_vs_baseline": _trace_delta(run["traces"], baseline["traces"]),
        "decoded": _decoded_report(policy, run, expert),
    }


def _gate_summary(
    seed_reports: Mapping[str, Any], condition: str, sample_count: int
) -> dict[str, Any]:
    per_sample: list[dict[str, Any]] = []
    for sample in range(sample_count):
        relative: list[float] = []
        direction: list[float] = []
        rotation_relative: list[float] = []
        new_gripper_errors: list[bool] = []
        for report in seed_reports.values():
            baseline = report["conditions"]["baseline"]["decoded"]
            candidate = report["conditions"][condition]["decoded"]
            base_error = float(baseline["xyz_squared_error"][sample])
            cand_error = float(candidate["xyz_squared_error"][sample])
            relative.append((base_error - cand_error) / max(base_error, 1e-6))
            base_first = np.asarray(baseline["refined_first_rows"][sample], dtype=np.float64)
            cand_first = np.asarray(candidate["refined_first_rows"][sample], dtype=np.float64)
            expert = np.asarray(candidate["expert_first_rows"][sample], dtype=np.float64)
            direction.append(
                float(np.dot(cand_first[:3] - base_first[:3], expert[:3] - base_first[:3]))
            )
            base_rotation = float(baseline["rotation_squared_error"][sample])
            cand_rotation = float(candidate["rotation_squared_error"][sample])
            rotation_relative.append((cand_rotation - base_rotation) / max(base_rotation, 1e-12))
            new_gripper_errors.append(
                bool(candidate["gripper_mismatch"][sample])
                and not bool(baseline["gripper_mismatch"][sample])
            )
        per_sample.append(
            {
                "median_relative_xyz_improvement": float(np.median(relative)),
                "median_direction_dot": float(np.median(direction)),
                "median_relative_rotation_worsening": float(np.median(rotation_relative)),
                "new_gripper_error_any_seed": any(new_gripper_errors),
            }
        )
    correct = sum(row["median_direction_dot"] > 0.0 for row in per_sample)
    return {
        "per_sample": per_sample,
        "direction_correct_count": correct,
        "sample_count": sample_count,
        "median_relative_xyz_improvement": float(
            np.median([row["median_relative_xyz_improvement"] for row in per_sample])
        ),
        "median_relative_rotation_worsening": float(
            np.median([row["median_relative_rotation_worsening"] for row in per_sample])
        ),
        "new_gripper_error_count": sum(row["new_gripper_error_any_seed"] for row in per_sample),
        "pre_sham_gate_only": bool(
            correct >= max(sample_count - 1, 1)
            and np.median([row["median_relative_xyz_improvement"] for row in per_sample]) >= 0.10
            and np.median([row["median_relative_rotation_worsening"] for row in per_sample]) <= 0.05
            and not any(row["new_gripper_error_any_seed"] for row in per_sample)
        ),
    }


def _matched_sham_summary(
    seed_reports: Mapping[str, Any],
    *,
    candidate_name: str,
    sham_names: Sequence[str],
    sample_count: int,
) -> dict[str, Any]:
    native_xyz_directional_advantage_floor = 0.01
    per_sample: list[dict[str, Any]] = []
    for sample in range(sample_count):
        by_sham: dict[str, Any] = {}
        for sham_name in sham_names:
            error_advantages: list[float] = []
            toward_expert_advantages: list[float] = []
            baseline_reference_advantages: list[float] = []
            for report in seed_reports.values():
                baseline = report["conditions"]["baseline"]["decoded"]
                candidate = report["conditions"][candidate_name]["decoded"]
                sham = report["conditions"][sham_name]["decoded"]
                candidate_error = float(candidate["xyz_squared_error"][sample])
                sham_error = float(sham["xyz_squared_error"][sample])
                candidate_first = np.asarray(
                    candidate["refined_first_rows"][sample], dtype=np.float64
                )
                sham_first = np.asarray(sham["refined_first_rows"][sample], dtype=np.float64)
                baseline_first = np.asarray(
                    baseline["refined_first_rows"][sample], dtype=np.float64
                )
                expert_first = np.asarray(
                    candidate["expert_first_rows"][sample], dtype=np.float64
                )
                error_advantages.append(sham_error - candidate_error)
                candidate_over_sham = candidate_first[:3] - sham_first[:3]
                sham_to_expert = expert_first[:3] - sham_first[:3]
                sham_distance = float(np.linalg.norm(sham_to_expert))
                toward_expert_advantages.append(
                    0.0
                    if sham_distance <= 1e-12
                    else float(np.dot(candidate_over_sham, sham_to_expert / sham_distance))
                )
                baseline_to_expert = expert_first[:3] - baseline_first[:3]
                baseline_distance = float(np.linalg.norm(baseline_to_expert))
                baseline_reference_advantages.append(
                    0.0
                    if baseline_distance <= 1e-12
                    else float(
                        np.dot(candidate_over_sham, baseline_to_expert / baseline_distance)
                    )
                )
            median_advantage = float(np.median(error_advantages))
            median_toward_expert = float(np.median(toward_expert_advantages))
            median_baseline_reference = float(np.median(baseline_reference_advantages))
            by_sham[sham_name] = {
                "median_xyz_squared_error_advantage": median_advantage,
                "median_signed_toward_expert_advantage": median_toward_expert,
                "median_signed_baseline_reference_advantage": median_baseline_reference,
                "beats_sham": bool(
                    median_advantage > 0.0
                    and median_toward_expert >= native_xyz_directional_advantage_floor
                ),
            }
        per_sample.append(
            {
                "sample": sample,
                "shams": by_sham,
                "beats_all_shams": all(row["beats_sham"] for row in by_sham.values()),
            }
        )
    pre = _gate_summary(seed_reports, candidate_name, sample_count)
    beats = sum(row["beats_all_shams"] for row in per_sample)
    return {
        "candidate": candidate_name,
        "shams": list(sham_names),
        "native_xyz_directional_advantage_floor": (
            native_xyz_directional_advantage_floor
        ),
        "per_sample": per_sample,
        "beats_all_shams_count": beats,
        "sample_count": sample_count,
        "pre_sham_gate": pre,
        "matched_sham_gate_only": bool(
            pre["pre_sham_gate_only"] and beats >= max(sample_count - 1, 1)
        ),
    }


def run(
    *,
    hypothesis: str,
    checkpoint: Path,
    config_path: Path,
    t5_condition: Path | None,
    dinov2_model: Path | None,
    local_files_only: bool,
    output: Path,
    device_name: str,
    episodes: int,
    seeds: Sequence[int],
    repo_root: Path,
) -> dict[str, Any]:
    hypothesis = hypothesis.upper()
    if hypothesis not in {"H2", "H3"}:
        raise ValueError("hypothesis must be H2 or H3")
    if episodes <= 0 or not seeds:
        raise ValueError("episodes and seeds must be nonempty")
    device = torch.device(device_name)
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=device,
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    if policy.bundle.checkpoint_sha256 not in FROZEN_CHECKPOINTS:
        raise RuntimeError("H2/H3 harness admits only the two frozen E1 checkpoints")
    if (int(policy.bundle.epoch), int(policy.bundle.global_step)) != (1, 11012):
        raise RuntimeError("frozen E1/step-11012 identity changed")
    variant = FROZEN_CHECKPOINTS[policy.bundle.checkpoint_sha256]
    config = load_config(config_path)
    if config.digest(include_paths=False) != policy.bundle.identity.config_digest:
        raise ValueError("external config executable digest differs from checkpoint")
    source = active_source_snapshot(repo_root)
    if source.digest != policy.bundle.identity.source.digest:
        raise ValueError("executed source snapshot differs from checkpoint")
    schedule = resolve_deployment_flow_schedule(config.runtime.deployment_flow_schedule)
    if (
        any(
            schedule.identity.get(role, {}).get("parameters", {}).get("candidate_id") != "Q5"
            for role in ("proposal", "refined")
        )
        or schedule.identity.get("physical_nfe") != 10
        or schedule.identity.get("endpoint_head_calls") != 2
    ):
        raise ValueError("H2/H3 harness requires the registered Q5/Q5 schedule")
    model = policy.bundle.model
    module_origins = _module_origins(repo_root, model)
    data = load_mainline_data(config)
    live_dataset = dataset_identity(data, config)
    saved_dataset = asdict(policy.bundle.identity.dataset)
    if {k: v for k, v in asdict(live_dataset).items() if k != "raw_root"} != {
        k: v for k, v in saved_dataset.items() if k != "raw_root"
    }:
        raise ValueError("live data identity differs from checkpoint identity")
    live_language = language_identity(data, config)
    saved_language = policy.bundle.identity.language
    if (
        live_language.logical_name,
        live_language.size_bytes,
        live_language.sha256,
    ) != (saved_language.logical_name, saved_language.size_bytes, saved_language.sha256):
        raise ValueError("live language identity differs from checkpoint identity")
    loader = data.loader(
        "val",
        batch_size=int(episodes),
        workers=0,
        device=device,
        generator=torch.Generator().manual_seed(int(config.data.seed) + 6203),
    )
    raw, selection = _select_samples(loader, int(episodes))
    batch = to_training_batch(raw, goal=data.goal, config=config, device=device)
    base = _unwrap_dataset(loader.dataset)
    expert = batch.action_target.raw_units.detach().cpu().numpy()
    dtype = resolve_compute_dtype(config)
    conditions = H2_CONDITIONS if hypothesis == "H2" else H3_CONDITIONS
    with _preserved_rng(), _frozen_model(model) as invariants:
        if hypothesis == "H2":
            caches, seam = _h2_caches(model, batch, base, selection, config, dtype)
        else:
            caches, seam = _h3_caches(model, batch, config, dtype)
        cache_hashes = {name: _tree_sha256(cache) for name, cache in caches.items()}
        if cache_hashes["identity"] != cache_hashes["baseline"]:
            raise RuntimeError("identity wrapper changed the encoded policy cache")
        negative_control_names = (
            ("goal_row3_only", "history_row3_only") if hypothesis == "H3" else ()
        )
        for name in negative_control_names:
            if cache_hashes[name] != cache_hashes["baseline"]:
                raise RuntimeError(f"H3 row-3 negative control changed the encoded cache: {name}")
        cache_deltas = {
            condition: {
                name: _tensor_delta(value, _cache_tensors(caches["baseline"])[name])
                for name, value in _cache_tensors(caches[condition]).items()
            }
            for condition in conditions
        }
        seed_reports: dict[str, Any] = {}
        for seed in seeds:
            generator_device = device if device.type == "cuda" else torch.device("cpu")
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))
            baseline = _run_q5(
                model,
                caches["baseline"],
                config,
                initial_noise=None,
                generator=generator,
            )
            runs: dict[str, Any] = {"baseline": baseline}
            reports: dict[str, Any] = {
                "baseline": _condition_report(policy, baseline, baseline, expert)
            }
            for condition in conditions[1:]:
                run_result = _run_q5(
                    model,
                    caches[condition],
                    config,
                    initial_noise=baseline["initial_noise"],
                    generator=None,
                )
                runs[condition] = run_result
                reports[condition] = _condition_report(policy, run_result, baseline, expert)
            identity = _identity_check(runs["identity"], baseline)
            negative_controls = {
                name: _identity_check(runs[name], baseline) for name in negative_control_names
            }
            seed_reports[str(int(seed))] = {
                "initial_noise_sha256": baseline["initial_noise_sha256"],
                "identity": identity,
                "negative_controls": negative_controls,
                "conditions": reports,
            }
    gates = {
        condition: _gate_summary(seed_reports, condition, int(episodes))
        for condition in conditions
        if condition not in {"baseline", "identity"}
    }
    if hypothesis == "H2":
        matched_sham_gates = {
            "observed_aligned": _matched_sham_summary(
                seed_reports,
                candidate_name="observed_aligned",
                sham_names=("action_permutation",),
                sample_count=int(episodes),
            )
        }
    else:
        matched_sham_gates = {
            "goal_interval": _matched_sham_summary(
                seed_reports,
                candidate_name="goal_interval",
                sham_names=("goal_permutation", "goal_mean_only"),
                sample_count=int(episodes),
            ),
            "history_interval": _matched_sham_summary(
                seed_reports,
                candidate_name="history_interval",
                sham_names=("history_permutation", "history_mean_only"),
                sample_count=int(episodes),
            ),
        }
    result = {
        "schema": SCHEMA,
        "hypothesis": hypothesis,
        "variant": variant,
        "diagnostic_not_training": True,
        "harness_valid": True,
        "candidate_package_admitted": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": policy.bundle.checkpoint_sha256,
            "epoch": int(policy.bundle.epoch),
            "global_step": int(policy.bundle.global_step),
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "digest": config.digest(include_paths=False),
        },
        "source_commit": str(policy.bundle.identity.git_commit),
        "source_snapshot_digest": source.digest,
        "executed_module_origins": module_origins,
        "dataset_identity": {
            "inventory_sha256": live_dataset.inventory_sha256,
            "state_normalizer_sha256": live_dataset.state_normalizer_sha256,
            "action_normalizer_sha256": live_dataset.action_normalizer_sha256,
            "decoded_cache_identity": live_dataset.decoded_cache_identity,
            "dino_cache_identity": live_dataset.dino_cache_identity,
        },
        "language_identity": {
            "logical_name": live_language.logical_name,
            "sha256": live_language.sha256,
        },
        "selection": selection,
        "seeds": [int(value) for value in seeds],
        "conditions": list(conditions),
        "seam": seam,
        "cache_sha256": cache_hashes,
        "cache_deltas_vs_baseline": cache_deltas,
        "seed_reports": seed_reports,
        "pre_sham_gates": gates,
        "matched_sham_gates": matched_sham_gates,
        "invariants": invariants,
        "interpretation": {
            "h2": "observed-aligned adds real causal past states; it is not a pure relabel of the sparse online snapshot",
            "h3": "interval conditions reroute existing S interval innovations only into one static P1 direct lane",
            "gate": "pre_sham_gate_only is insufficient for admission; the corresponding permutation/mean sham must also be beaten in both variants",
            "guard": "no optimizer, checkpoint write, future Teacher input, controller rule, or DataLoader change entered the model graph",
        },
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis", choices=("H2", "H3"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=(9182026, 0))
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        hypothesis=args.hypothesis,
        checkpoint=args.checkpoint,
        config_path=args.config,
        t5_condition=args.t5_condition,
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
        output=args.output,
        device_name=args.device,
        episodes=int(args.episodes),
        seeds=tuple(int(value) for value in args.seeds),
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hypothesis": result["hypothesis"],
                "variant": result["variant"],
                "harness_valid": result["harness_valid"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
