#!/usr/bin/env python3
"""Read-only CALVIN probe for the 24-row action -> four-row W seam.

The probe measures how much paired expert/proposal arm error lies in the
row-space and null-space of the exact four-window averaging operator.  A
small, predeclared panel then intervenes only at the W rebuild boundary while
holding the observation cache, proposal, gripper command, Q5 schedule and
refined-pass physical noise fixed.

This is a diagnostic, not a training launcher.  It never updates model state
or writes to the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import default_collate

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.runtime.deployment import deployment_graph_config
from clearvla.mainline.runtime.identity import (
    dataset_identity,
    language_identity,
    v120_normalizer_fingerprint,
)
from clearvla.mainline.runtime.numerics import resolve_compute_dtype
from clearvla.mainline.runtime.sampling import (
    deployment_cache,
    refine_cached_world,
    sample_cached_action,
)
from clearvla.simulation.checkpoint import load_deployment_checkpoint

SCHEMA = "clearvla-calvin-action-temporal-observability-v1"
ARM_DIM = 6
HORIZON = 24
INTERVAL_ROWS_1BASED = ((4, 8), (8, 16), (16, 24), (24, 24))
TIME_BANDS = ((0, 4, "rows_1_4"), (4, 12, "rows_5_12"), (12, 24, "rows_13_24"))
CHANNEL_BANDS = ((0, 3, "xyz"), (3, 6, "rotation"))
INTERVENTION_TASK_ORDINALS = (0, 2, 3, 5)
PROJECTION_EPS = 1.0e-8
CONDITION_TOLERANCE = 2.0e-5
REPLAY_TOLERANCE = 2.0e-5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--panel-batch-size", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def _generator(device: torch.device, seed: int) -> torch.Generator:
    owner = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=owner).manual_seed(int(seed))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_imports_under(repo_root: Path) -> dict[str, str]:
    owners = {
        "active_source_snapshot": active_source_snapshot,
        "load_mainline_data": load_mainline_data,
        "deployment_cache": deployment_cache,
        "load_deployment_checkpoint": load_deployment_checkpoint,
    }
    resolved: dict[str, str] = {}
    for name, owner in owners.items():
        source = Path(inspect.getfile(owner)).resolve()
        if not source.is_relative_to(repo_root):
            raise RuntimeError(
                f"import owner {name} resolved outside frozen repo: {source}"
            )
        resolved[name] = str(source)
    return resolved


def _temporal_matrix(*, device: torch.device | str = "cpu", dtype=torch.float64) -> Tensor:
    matrix = torch.zeros(4, HORIZON, device=device, dtype=dtype)
    for row, (lower, upper) in enumerate(INTERVAL_ROWS_1BASED):
        start = lower - 1
        stop = upper
        matrix[row, start:stop] = 1.0 / float(stop - start)
    return matrix


def _projection_operators(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    matrix = _temporal_matrix(device=device, dtype=torch.float64)
    row_projection = matrix.mT @ torch.linalg.pinv(matrix @ matrix.mT) @ matrix
    null_projection = torch.eye(HORIZON, device=device, dtype=torch.float64) - row_projection
    return matrix, row_projection, null_projection


def _rms(value: Tensor) -> float:
    detached = value.detach().double()
    return float(detached.square().mean().sqrt().cpu()) if detached.numel() else 0.0


def _max_abs(value: Tensor) -> float:
    detached = value.detach().double()
    return float(detached.abs().amax().cpu()) if detached.numel() else 0.0


def _l2(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach().double()).cpu())


def _finite(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().cpu())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-12:
        return None
    return float(numerator / denominator)


def _basic_stats(value: Tensor) -> dict[str, float | int]:
    detached = value.detach().double()
    return {
        "elements": int(detached.numel()),
        "rms": _rms(detached),
        "l2": _l2(detached),
        "max_abs": _max_abs(detached),
        "minimum": float(detached.amin().cpu()),
        "maximum": float(detached.amax().cpu()),
    }


def _partition_stats(value: Tensor) -> dict[str, Any]:
    """Summarize an [...,24,6] arm tensor without erasing key axes."""

    if tuple(value.shape[-2:]) != (HORIZON, ARM_DIM):
        raise ValueError(f"expected [...,{HORIZON},{ARM_DIM}], got {tuple(value.shape)}")
    result: dict[str, Any] = {"all": _basic_stats(value)}
    result["time_bands"] = {
        name: _basic_stats(value[..., start:stop, :])
        for start, stop, name in TIME_BANDS
    }
    result["channels"] = {
        name: _basic_stats(value[..., :, start:stop])
        for start, stop, name in CHANNEL_BANDS
    }
    result["first_three_rows"] = _basic_stats(value[..., :3, :])
    result["row_24"] = _basic_stats(value[..., 23:24, :])
    return result


def _decompose(error: Tensor, row_projection: Tensor) -> tuple[Tensor, Tensor]:
    if tuple(error.shape[-2:]) != (HORIZON, ARM_DIM):
        raise ValueError("paired arm error must end in [24,6]")
    visible = torch.einsum("tu,...ua->...ta", row_projection, error.double())
    invisible = error.double() - visible
    return visible, invisible


def _decomposition_report(error: Tensor, visible: Tensor, invisible: Tensor, matrix: Tensor) -> dict[str, Any]:
    total_energy = float(error.double().square().sum().cpu())
    visible_energy = float(visible.square().sum().cpu())
    invisible_energy = float(invisible.square().sum().cpu())
    return {
        "error": _partition_stats(error.double()),
        "row_space_component": _partition_stats(visible),
        "null_space_component": _partition_stats(invisible),
        "energy_fraction_row_space": _safe_ratio(visible_energy, total_energy),
        "energy_fraction_null_space": _safe_ratio(invisible_energy, total_energy),
        "orthogonality_dot": float((visible * invisible).sum().cpu()),
        "reconstruction_max_abs": _max_abs(error.double() - visible - invisible),
        "matrix_times_null_max_abs": _max_abs(torch.einsum("it,...ta->...ia", matrix, invisible)),
    }


def _selected_manifest(data: Any, split: str = "val") -> list[dict[str, Any]]:
    if not data.is_multitask:
        raise ValueError("temporal observability panel requires a multitask registry")
    dataset = data.datasets[split]
    refs = dataset.base.refs
    task_indices = data.dataset_task_indices(split)
    by_task_episode: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for dataset_index, (ref, task_index) in enumerate(zip(refs, task_indices, strict=True)):
        by_task_episode[int(task_index)][int(ref.episode_idx)].append(int(dataset_index))

    selected: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(data.task_order):
        episodes = sorted(by_task_episode.get(task_index, {}))
        if len(episodes) < 2:
            raise ValueError(f"task {task_name!r} has fewer than two validation episodes")
        choices = ((episodes[0], 1, 3, "first_episode_one_third"), (episodes[-1], 2, 3, "last_episode_two_thirds"))
        for episode_index, numerator, denominator, rule in choices:
            candidates = sorted(
                by_task_episode[task_index][episode_index],
                key=lambda index: (int(refs[index].center), int(index)),
            )
            if not candidates:
                raise RuntimeError("predeclared validation episode has no valid windows")
            ordinal = int(math.floor((len(candidates) - 1) * numerator / denominator))
            dataset_index = candidates[ordinal]
            ref = refs[dataset_index]
            episode = data.episodes[episode_index]
            action_start = int(ref.center + dataset.base.config.action_offset)
            action_end = action_start + HORIZON - 1
            terminal = episode.terminal_state_index
            terminal_padding_rows = 0
            if terminal is not None:
                first_synthetic = max(action_start, int(terminal))
                terminal_padding_rows = max(0, action_end - first_synthetic + 1)
                terminal_padding_rows = min(HORIZON, terminal_padding_rows)
            selected.append(
                {
                    "panel_index": len(selected),
                    "task_index": int(task_index),
                    "task_name": str(task_name),
                    "selection_rule": rule,
                    "episode_index": int(episode_index),
                    "episode_id": str(episode.episode_id),
                    "source_trajectory_id": str(episode.source_trajectory_id),
                    "instruction": None if episode.instruction is None else str(episode.instruction),
                    "dataset_index": int(dataset_index),
                    "episode_window_count": len(candidates),
                    "episode_window_ordinal": int(ordinal),
                    "center": int(ref.center),
                    "boundary_region": str(ref.boundary_region),
                    "action_start": action_start,
                    "action_end": action_end,
                    "terminal_state_index": None if terminal is None else int(terminal),
                    "terminal_padding_rows": int(terminal_padding_rows),
                    "source_start": episode.source_start,
                    "source_end": episode.source_end,
                    "context_start": episode.context_start,
                }
            )
    if len(selected) != 2 * len(data.task_order):
        raise AssertionError("panel selection lost a task or episode")
    episode_keys = {(row["task_index"], row["episode_index"]) for row in selected}
    if len(episode_keys) != len(selected):
        raise AssertionError("panel does not contain distinct task/episode pairs")
    return selected


def _batch(data: Any, config: Any, indices: Sequence[int], device: torch.device):
    raw = default_collate([data.datasets["val"][int(index)] for index in indices])
    return to_training_batch(raw, goal=data.goal, config=config, device=device)


def _centered_arm(action: Tensor, normalizer: Any) -> Tensor:
    offset = action.new_tensor(normalizer.offset).reshape(1, 1, -1)
    return action[..., :ARM_DIM] - offset[..., :ARM_DIM]


def _native_arm(action: Tensor, normalizer: Any) -> Tensor:
    offset = action.new_tensor(normalizer.offset).reshape(1, 1, -1)
    scale = action.new_tensor(normalizer.scale).reshape(1, 1, -1)
    return (action[..., :ARM_DIM] - offset[..., :ARM_DIM]) / scale[..., :ARM_DIM]


def _range_report(action: Tensor, normalizer: Any) -> dict[str, Any]:
    raw = _native_arm(action, normalizer)
    lower = raw.new_tensor(normalizer.minimum[..., :ARM_DIM]).reshape(1, 1, ARM_DIM)
    upper = raw.new_tensor(normalizer.maximum[..., :ARM_DIM]).reshape(1, 1, ARM_DIM)
    outside = (raw < lower) | (raw > upper)
    return {
        "native_arm": _basic_stats(raw),
        "outside_training_range_fraction": float(outside.float().mean().cpu()),
        "outside_training_range_count": int(outside.sum().cpu()),
    }


def _condition(model: Any, cache: Any, world_action: Tensor):
    return model.outlet_adapter.world_condition_from_horizon_action(
        world_action,
        cache.history.action_state,
    )


def _dynamics_delta(candidate: Any, baseline: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        delta = getattr(candidate, name).detach().double() - getattr(baseline, name).detach().double()
        result[name] = {
            "all": _basic_stats(delta),
            "per_interval": [_basic_stats(delta[:, index]) for index in range(int(delta.shape[1]))],
        }
    return result


def _action_report(candidate: Tensor, baseline: Tensor, expert: Tensor, normalizer: Any) -> dict[str, Any]:
    candidate_arm = candidate[..., :ARM_DIM].double()
    baseline_arm = baseline[..., :ARM_DIM].double()
    expert_arm = expert[..., :ARM_DIM].double()
    delta = candidate_arm - baseline_arm
    desired = expert_arm - baseline_arm
    before = baseline_arm - expert_arm
    after = candidate_arm - expert_arm
    dot = float((delta * desired).sum().cpu())
    desired_energy = float(desired.square().sum().cpu())
    cosine_denominator = _l2(delta) * _l2(desired)
    raw_candidate = _native_arm(candidate.float(), normalizer).double()
    raw_baseline = _native_arm(baseline.float(), normalizer).double()
    raw_expert = _native_arm(expert.float(), normalizer).double()
    return {
        "delta_vs_baseline": _partition_stats(delta),
        "first_action_row_delta": _basic_stats(delta[..., :1, :]),
        "arm_rmse_to_expert": _rms(after),
        "baseline_arm_rmse_to_expert": _rms(before),
        "arm_rmse_improvement": _rms(before) - _rms(after),
        "direction_cosine_to_expert_correction": _safe_ratio(dot, cosine_denominator),
        "projection_fraction_of_expert_correction": _safe_ratio(dot, desired_energy),
        "native_arm_rmse_to_expert": _rms(raw_candidate - raw_expert),
        "native_baseline_arm_rmse_to_expert": _rms(raw_baseline - raw_expert),
        "gripper_change_fraction_vs_baseline": float(
            (candidate[..., -1] != baseline[..., -1]).float().mean().cpu()
        ),
    }


def _scalar_metrics(metrics: Mapping[str, Tensor]) -> dict[str, float]:
    return {
        str(name): float(value.detach().float().cpu())
        for name, value in metrics.items()
        if isinstance(value, Tensor) and value.ndim == 0
    }


@torch.inference_mode()
def _proposal(model: Any, cache: Any, config: Any, dtype: torch.dtype, seed: int):
    return sample_cached_action(
        model,
        cache,
        config,
        generator=_generator(cache.history.state.device, seed),
        collect_diagnostics=False,
        dtype=dtype,
        pass_role="proposal",
    )


@torch.inference_mode()
def _branch(
    model: Any,
    cache: Any,
    config: Any,
    dtype: torch.dtype,
    world_action: Tensor,
    initial_noise: Tensor,
) -> tuple[Any, Any, dict[str, Tensor]]:
    refined_cache, metrics = refine_cached_world(
        model,
        cache,
        world_action,
        config,
        collect_diagnostics=True,
        dtype=dtype,
    )
    result = sample_cached_action(
        model,
        refined_cache,
        config,
        initial_physical_noise=initial_noise.clone(),
        collect_diagnostics=False,
        dtype=dtype,
        pass_role="refined",
    )
    return result, refined_cache, metrics


def _panel_pass(
    *,
    data: Any,
    config: Any,
    model: Any,
    dtype: torch.dtype,
    normalizer: Any,
    device: torch.device,
    manifest: Sequence[Mapping[str, Any]],
    matrix: Tensor,
    row_projection: Tensor,
    base_seed: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(manifest), batch_size):
        subset = manifest[start : start + batch_size]
        indices = [int(row["dataset_index"]) for row in subset]
        batch = _batch(data, config, indices, device)
        cache, _ = deployment_cache(model, batch.online, config, dtype=dtype)
        proposal_seed = int(base_seed + 1000 + start)
        proposal = _proposal(model, cache, config, dtype, proposal_seed)
        world = model.outlet_adapter.world_condition_action_from_deployed(
            proposal.action,
            command=proposal.gripper_command,
        )
        error = batch.action_target.normalized[..., :ARM_DIM].double() - world[..., :ARM_DIM].double()
        visible, invisible = _decompose(error, row_projection)
        for local, manifest_row in enumerate(subset):
            report = _decomposition_report(
                error[local : local + 1],
                visible[local : local + 1],
                invisible[local : local + 1],
                matrix,
            )
            rows.append(
                {
                    "manifest": dict(manifest_row),
                    "proposal_seed": proposal_seed,
                    "proposal_batch_size": len(subset),
                    "proposal_batch_row": int(local),
                    "proposal_initial_noise_row_sha256": _tensor_sha256(
                        proposal.initial_physical_noise[local : local + 1]
                    ),
                    "decomposition": report,
                    "proposal_centered_arm": _partition_stats(
                        _centered_arm(world[local : local + 1], normalizer)
                    ),
                    "proposal_range": _range_report(world[local : local + 1], normalizer),
                    "expert_range": _range_report(
                        batch.action_target.normalized[local : local + 1], normalizer
                    ),
                }
            )
        del batch, cache, proposal, world, error, visible, invisible
    return rows


def _intervention_pass(
    *,
    data: Any,
    config: Any,
    model: Any,
    dtype: torch.dtype,
    normalizer: Any,
    device: torch.device,
    manifest: Sequence[Mapping[str, Any]],
    matrix: Tensor,
    row_projection: Tensor,
    base_seed: int,
) -> list[dict[str, Any]]:
    selected = [manifest[2 * ordinal] for ordinal in INTERVENTION_TASK_ORDINALS]
    indices = [int(row["dataset_index"]) for row in selected]
    reports: list[dict[str, Any]] = []

    for seed_ordinal in range(2):
        seed = int(base_seed + 2000 + seed_ordinal)
        batch = _batch(data, config, indices, device)
        cache, _ = deployment_cache(model, batch.online, config, dtype=dtype)
        proposal = _proposal(model, cache, config, dtype, seed)
        proposal_world = model.outlet_adapter.world_condition_action_from_deployed(
            proposal.action,
            command=proposal.gripper_command,
        ).float()
        expert = batch.action_target.normalized.float()
        error = expert[..., :ARM_DIM].double() - proposal_world[..., :ARM_DIM].double()
        visible, invisible = _decompose(error, row_projection)
        visible_rms = visible.square().mean(dim=(-2, -1)).sqrt()
        invisible_rms = invisible.square().mean(dim=(-2, -1)).sqrt()
        comparable = (visible_rms > PROJECTION_EPS) & (invisible_rms > PROJECTION_EPS)
        common = torch.minimum(visible_rms, invisible_rms)
        visible_scale = torch.where(comparable, common / visible_rms.clamp_min(PROJECTION_EPS), torch.zeros_like(common))
        invisible_scale = torch.where(comparable, common / invisible_rms.clamp_min(PROJECTION_EPS), torch.zeros_like(common))
        visible_matched = visible * visible_scale[:, None, None]
        invisible_matched = invisible * invisible_scale[:, None, None]

        candidates: dict[str, Tensor] = {"baseline": proposal_world.clone()}
        for name, delta in (
            ("null_plus", invisible_matched),
            ("null_minus", -invisible_matched),
            ("visible_plus", visible_matched),
            ("visible_minus", -visible_matched),
        ):
            candidate = proposal_world.clone()
            candidate[..., :ARM_DIM] = candidate[..., :ARM_DIM] + delta.float()
            candidates[name] = candidate
        expert_candidate = proposal_world.clone()
        expert_candidate[..., :ARM_DIM] = expert[..., :ARM_DIM]
        candidates["expert_arm_transplant"] = expert_candidate
        if seed_ordinal == 0:
            visible_exact = proposal_world.clone()
            visible_exact[..., :ARM_DIM] = visible_exact[..., :ARM_DIM] + visible.float()
            candidates["visible_exact_equivalence_check"] = visible_exact

        conditions = {
            name: _condition(model, cache, action)
            for name, action in candidates.items()
        }
        baseline_condition = conditions["baseline"]
        branch_outputs: dict[str, tuple[Any, Any, dict[str, Tensor]]] = {}
        for name, candidate in candidates.items():
            branch_outputs[name] = _branch(
                model,
                cache,
                config,
                dtype,
                candidate,
                proposal.initial_physical_noise,
            )
        branch_batch_metrics = {
            name: _scalar_metrics(output[2])
            for name, output in branch_outputs.items()
        }

        baseline_result, baseline_cache, baseline_metrics = branch_outputs["baseline"]
        replay_result, replay_cache, _ = _branch(
            model,
            cache,
            config,
            dtype,
            proposal_world,
            proposal.initial_physical_noise,
        )
        replay_action_residual = _max_abs(replay_result.action - baseline_result.action)
        replay_w_residual = max(
            _max_abs(
                getattr(replay_cache.top.predicted_dynamics, field)
                - getattr(baseline_cache.top.predicted_dynamics, field)
            )
            for field in ("semantic_delta", "transport_mean", "transport_covariance")
        )

        per_sample: list[dict[str, Any]] = []
        for sample_index, manifest_row in enumerate(selected):
            sample_conditions: dict[str, Any] = {}
            sample_branches: dict[str, Any] = {}
            base_action = baseline_result.action[sample_index : sample_index + 1]
            base_dynamics = baseline_cache.top.predicted_dynamics
            for name, candidate in candidates.items():
                candidate_condition = conditions[name]
                action_delta = (
                    candidate[sample_index : sample_index + 1, :, :ARM_DIM]
                    - proposal_world[sample_index : sample_index + 1, :, :ARM_DIM]
                ).double()
                expected_source = torch.einsum("it,bta->bia", matrix, action_delta)
                expected_cumulative = torch.cumsum(expected_source, dim=1)
                actual_source = (
                    candidate_condition.source_interval_action[sample_index : sample_index + 1, :, :ARM_DIM]
                    - baseline_condition.source_interval_action[sample_index : sample_index + 1, :, :ARM_DIM]
                ).double()
                actual_cumulative = (
                    candidate_condition.interval_action[sample_index : sample_index + 1, :, :ARM_DIM]
                    - baseline_condition.interval_action[sample_index : sample_index + 1, :, :ARM_DIM]
                ).double()
                actual_delta = (
                    candidate_condition.interval_delta[sample_index : sample_index + 1, :, :ARM_DIM]
                    - baseline_condition.interval_delta[sample_index : sample_index + 1, :, :ARM_DIM]
                ).double()
                adapter_residual = max(
                    _max_abs(actual_source - expected_source),
                    _max_abs(actual_cumulative - expected_cumulative),
                    _max_abs(actual_delta - expected_source),
                )
                per_row_condition = {
                    "source_interval_action": _basic_stats(
                        candidate_condition.source_interval_action[sample_index : sample_index + 1]
                        - baseline_condition.source_interval_action[sample_index : sample_index + 1]
                    ),
                    "canonical_interval_action": _basic_stats(
                        candidate_condition.interval_action[sample_index : sample_index + 1]
                        - baseline_condition.interval_action[sample_index : sample_index + 1]
                    ),
                    "canonical_interval_delta": _basic_stats(
                        candidate_condition.interval_delta[sample_index : sample_index + 1]
                        - baseline_condition.interval_delta[sample_index : sample_index + 1]
                    ),
                    "fingerprint": _basic_stats(
                        candidate_condition.action_fingerprint[sample_index : sample_index + 1]
                        - baseline_condition.action_fingerprint[sample_index : sample_index + 1]
                    ),
                    "adapter_J_residual_max_abs": float(adapter_residual),
                }
                sample_conditions[name] = per_row_condition
                result, refined_cache, _refinement_metrics = branch_outputs[name]
                branch_finite = all(
                    _finite(value)
                    for value in (
                        candidate[sample_index : sample_index + 1],
                        result.action[sample_index : sample_index + 1],
                        refined_cache.top.predicted_dynamics.semantic_delta[
                            sample_index : sample_index + 1
                        ],
                        refined_cache.top.predicted_dynamics.transport_mean[
                            sample_index : sample_index + 1
                        ],
                        refined_cache.top.predicted_dynamics.transport_covariance[
                            sample_index : sample_index + 1
                        ],
                    )
                )
                sample_branches[name] = {
                    "condition": per_row_condition,
                    "candidate_range": _range_report(
                        candidate[sample_index : sample_index + 1], normalizer
                    ),
                    "world_response": _dynamics_delta(
                        type("DynamicsSlice", (), {
                            field: getattr(refined_cache.top.predicted_dynamics, field)[sample_index : sample_index + 1]
                            for field in ("semantic_delta", "transport_mean", "transport_covariance")
                        })(),
                        type("DynamicsSlice", (), {
                            field: getattr(base_dynamics, field)[sample_index : sample_index + 1]
                            for field in ("semantic_delta", "transport_mean", "transport_covariance")
                        })(),
                    ),
                    "refined_action": _action_report(
                        result.action[sample_index : sample_index + 1],
                        base_action,
                        expert[sample_index : sample_index + 1],
                        normalizer,
                    ),
                    "finite": branch_finite,
                }

            equivalence: dict[str, Any] | None = None
            if seed_ordinal == 0:
                exact_name = "visible_exact_equivalence_check"
                expert_name = "expert_arm_transplant"
                exact_result, exact_cache, _ = branch_outputs[exact_name]
                expert_result, expert_cache, _ = branch_outputs[expert_name]
                equivalence = {
                    "condition_fingerprint_max_abs": _max_abs(
                        conditions[exact_name].action_fingerprint[sample_index : sample_index + 1]
                        - conditions[expert_name].action_fingerprint[sample_index : sample_index + 1]
                    ),
                    "world_semantic_max_abs": _max_abs(
                        exact_cache.top.predicted_dynamics.semantic_delta[sample_index : sample_index + 1]
                        - expert_cache.top.predicted_dynamics.semantic_delta[sample_index : sample_index + 1]
                    ),
                    "world_transport_max_abs": _max_abs(
                        exact_cache.top.predicted_dynamics.transport_mean[sample_index : sample_index + 1]
                        - expert_cache.top.predicted_dynamics.transport_mean[sample_index : sample_index + 1]
                    ),
                    "world_covariance_max_abs": _max_abs(
                        exact_cache.top.predicted_dynamics.transport_covariance[sample_index : sample_index + 1]
                        - expert_cache.top.predicted_dynamics.transport_covariance[sample_index : sample_index + 1]
                    ),
                    "refined_action_max_abs": _max_abs(
                        exact_result.action[sample_index : sample_index + 1]
                        - expert_result.action[sample_index : sample_index + 1]
                    ),
                }
                equivalence["within_tolerance"] = bool(
                    max(float(value) for key, value in equivalence.items() if key.endswith("max_abs"))
                    <= CONDITION_TOLERANCE
                )

            null_condition_max = max(
                sample_conditions[name]["fingerprint"]["max_abs"]
                for name in ("null_plus", "null_minus")
            )
            adapter_residual_max = max(
                sample_conditions[name]["adapter_J_residual_max_abs"]
                for name in sample_conditions
            )
            finite_checks_all_pass = all(
                bool(branch["finite"]) for branch in sample_branches.values()
            )
            per_sample.append(
                {
                    "manifest": dict(manifest_row),
                    "comparable": bool(comparable[sample_index].cpu()),
                    "visible_rms": float(visible_rms[sample_index].cpu()),
                    "invisible_rms": float(invisible_rms[sample_index].cpu()),
                    "matched_intervention_rms": float(common[sample_index].cpu()),
                    "amplitude_policy": "per_sample_min_visible_null_rms_no_clipping_report_range",
                    "null_condition_max_abs": float(null_condition_max),
                    "null_condition_within_tolerance": bool(null_condition_max <= CONDITION_TOLERANCE),
                    "adapter_J_residual_max_abs": float(adapter_residual_max),
                    "adapter_J_within_tolerance": bool(adapter_residual_max <= CONDITION_TOLERANCE),
                    "finite_checks_all_pass": finite_checks_all_pass,
                    "branches": sample_branches,
                    "expert_visible_equivalence": equivalence,
                }
            )

        reports.append(
            {
                "seed": seed,
                "proposal_schedule_identity": proposal.flow_schedule_identity,
                "proposal_step_times": proposal.step_times.detach().float().cpu().tolist(),
                "proposal_step_sizes": list(proposal.step_sizes),
                "proposal_initial_noise_sha256": _tensor_sha256(
                    proposal.initial_physical_noise
                ),
                "proposal_initial_noise_row_sha256": [
                    _tensor_sha256(proposal.initial_physical_noise[index : index + 1])
                    for index in range(int(proposal.initial_physical_noise.shape[0]))
                ],
                "batch_order": [dict(row) for row in selected],
                "branch_batch_refinement_metrics": branch_batch_metrics,
                "baseline_refinement_metrics": _scalar_metrics(baseline_metrics),
                "baseline_replay_action_max_abs": replay_action_residual,
                "baseline_replay_world_max_abs": replay_w_residual,
                "baseline_replay_within_tolerance": bool(
                    max(replay_action_residual, replay_w_residual) <= REPLAY_TOLERANCE
                ),
                "samples": per_sample,
            }
        )
        del batch, cache, proposal, proposal_world, expert, branch_outputs
    return reports


def _aggregate(panel: Sequence[Mapping[str, Any]], interventions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    null_fractions = [
        row["decomposition"]["energy_fraction_null_space"]
        for row in panel
        if row["decomposition"]["energy_fraction_null_space"] is not None
    ]
    visible_fractions = [
        row["decomposition"]["energy_fraction_row_space"]
        for row in panel
        if row["decomposition"]["energy_fraction_row_space"] is not None
    ]
    replay = [bool(row["baseline_replay_within_tolerance"]) for row in interventions]
    equivalence_by_context: dict[int, bool] = {}
    for seed_report in interventions:
        for sample in seed_report["samples"]:
            equivalence = sample["expert_visible_equivalence"]
            if equivalence is not None:
                context = int(sample["manifest"]["panel_index"])
                equivalence_by_context[context] = bool(equivalence["within_tolerance"])

    raw_expert_improvement: list[float] = []
    raw_visible_response: list[float] = []
    raw_null_response: list[float] = []
    valid_expert_improvement: list[float] = []
    valid_visible_response: list[float] = []
    valid_null_response: list[float] = []
    validity_rows: list[dict[str, Any]] = []
    for seed_report in interventions:
        replay_pass = bool(seed_report["baseline_replay_within_tolerance"])
        for sample in seed_report["samples"]:
            branches = sample["branches"]
            expert_value = float(
                branches["expert_arm_transplant"]["refined_action"]["arm_rmse_improvement"]
            )
            visible_values = [
                float(branches[name]["refined_action"]["delta_vs_baseline"]["all"]["rms"])
                for name in ("visible_plus", "visible_minus")
            ]
            null_values = [
                float(branches[name]["refined_action"]["delta_vs_baseline"]["all"]["rms"])
                for name in ("null_plus", "null_minus")
            ]
            raw_expert_improvement.append(expert_value)
            raw_visible_response.extend(visible_values)
            raw_null_response.extend(null_values)

            context = int(sample["manifest"]["panel_index"])
            finite_pass = bool(sample["finite_checks_all_pass"])
            adapter_pass = bool(sample["adapter_J_within_tolerance"])
            null_pass = bool(sample["null_condition_within_tolerance"])
            comparable_pass = bool(sample["comparable"])
            equivalence_pass = bool(equivalence_by_context.get(context, False))
            matched_valid = all(
                (replay_pass, finite_pass, adapter_pass, null_pass, comparable_pass)
            )
            expert_valid = all(
                (replay_pass, finite_pass, adapter_pass, equivalence_pass)
            )
            reasons: list[str] = []
            for passed, reason in (
                (replay_pass, "baseline_replay_failed"),
                (finite_pass, "non_finite_branch"),
                (adapter_pass, "adapter_J_residual_failed"),
                (null_pass, "null_condition_failed"),
                (comparable_pass, "projection_component_too_small"),
                (equivalence_pass, "expert_visible_equivalence_failed"),
            ):
                if not passed:
                    reasons.append(reason)
            validity_rows.append(
                {
                    "seed": int(seed_report["seed"]),
                    "panel_index": context,
                    "matched_sensitivity_valid": matched_valid,
                    "expert_transplant_valid": expert_valid,
                    "invalid_reasons": reasons,
                }
            )
            if matched_valid:
                valid_visible_response.extend(visible_values)
                valid_null_response.extend(null_values)
            if expert_valid:
                valid_expert_improvement.append(expert_value)

    def statistics(values: Iterable[float]) -> dict[str, float | int | None]:
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0:
            return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    return {
        "panel_null_energy_fraction": statistics(null_fractions),
        "panel_row_space_energy_fraction": statistics(visible_fractions),
        "baseline_replay_checks_all_pass": bool(all(replay)),
        "expert_visible_equivalence_checks_all_pass": bool(
            equivalence_by_context and all(equivalence_by_context.values())
        ),
        "validity": {
            "rows": validity_rows,
            "matched_sensitivity_valid_count": sum(
                bool(row["matched_sensitivity_valid"]) for row in validity_rows
            ),
            "expert_transplant_valid_count": sum(
                bool(row["expert_transplant_valid"]) for row in validity_rows
            ),
            "total_count": len(validity_rows),
        },
        "raw_unfiltered": {
            "expert_arm_refined_rmse_improvement": statistics(raw_expert_improvement),
            "matched_visible_refined_response_rms": statistics(raw_visible_response),
            "matched_null_refined_response_rms": statistics(raw_null_response),
        },
        "valid_only": {
            "expert_arm_refined_rmse_improvement": statistics(valid_expert_improvement),
            "matched_visible_refined_response_rms": statistics(valid_visible_response),
            "matched_null_refined_response_rms": statistics(valid_null_response),
            "visible_to_null_refined_response_ratio_of_means": _safe_ratio(
                float(np.mean(valid_visible_response)) if valid_visible_response else 0.0,
                float(np.mean(valid_null_response)) if valid_null_response else 0.0,
            ),
        },
        "causal_summary_valid": bool(
            validity_rows
            and all(row["matched_sensitivity_valid"] for row in validity_rows)
            and all(row["expert_transplant_valid"] for row in validity_rows)
        ),
        "interpretation_scope": (
            "Frozen-model sensitivity of the W action-condition seam and paired expert-reference "
            "error observability; not closed-loop success, W prediction accuracy, or proof that "
            "this limitation is the unique cause of failure."
        ),
    }


def _self_test() -> None:
    matrix, row_projection, null_projection = _projection_operators(torch.device("cpu"))
    if int(torch.linalg.matrix_rank(matrix)) != 4:
        raise AssertionError("temporal matrix rank must be four")
    if _max_abs(matrix[:, :3]) != 0.0:
        raise AssertionError("first three action rows must be invisible to W")
    if not (matrix[0, 7] > 0 and matrix[1, 7] > 0):
        raise AssertionError("row eight must overlap the first two averages")
    if not (matrix[1, 15] > 0 and matrix[2, 15] > 0):
        raise AssertionError("row sixteen must overlap the middle averages")
    if not (matrix[2, 23] > 0 and matrix[3, 23] > 0):
        raise AssertionError("row twenty-four must feed the last two averages")
    sample = torch.randn(3, HORIZON, ARM_DIM, dtype=torch.float64)
    visible, invisible = _decompose(sample, row_projection)
    if _max_abs(sample - visible - invisible) > 1.0e-10:
        raise AssertionError("projection decomposition does not reconstruct")
    if _max_abs(torch.einsum("it,bta->bia", matrix, invisible)) > 1.0e-10:
        raise AssertionError("null component is visible to the temporal matrix")
    if _max_abs(row_projection @ null_projection) > 1.0e-10:
        raise AssertionError("row and null projections are not orthogonal")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _self_test()
    if args.self_test:
        return {"schema": SCHEMA, "self_test": "passed"}
    if int(args.panel_batch_size) <= 0:
        raise ValueError("panel batch size must be positive")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _seed(int(args.seed))
    repo_root = args.repo_root.expanduser().resolve()
    import_owners = _assert_imports_under(repo_root)
    live_source = active_source_snapshot(repo_root)
    git_head = _git_head(repo_root)
    print(f"[preflight] source={live_source.digest} git={git_head}", flush=True)

    bundle = load_deployment_checkpoint(
        args.checkpoint,
        device=device,
        t5_condition=args.t5_condition,
    )
    expected = {
        "checkpoint_sha256": str(args.expected_checkpoint_sha256).lower(),
        "source_digest": str(args.expected_source_digest).lower(),
        "git_commit": str(args.expected_git_commit).lower(),
        "epoch": int(args.expected_epoch),
        "global_step": int(args.expected_global_step),
    }
    actual = {
        "checkpoint_sha256": bundle.checkpoint_sha256.lower(),
        "source_digest": live_source.digest.lower(),
        "git_commit": git_head.lower(),
        "epoch": int(bundle.epoch),
        "global_step": int(bundle.global_step),
    }
    if actual != expected:
        raise RuntimeError(f"explicit checkpoint identity mismatch: expected={expected}, actual={actual}")
    if live_source != bundle.identity.source:
        saved = dict(bundle.identity.source.files)
        current = dict(live_source.files)
        mismatch = sorted(
            set(saved) ^ set(current)
            | {name for name in set(saved) & set(current) if saved[name] != current[name]}
        )
        raise RuntimeError(
            "live source differs from checkpoint identity; mismatches=" + repr(mismatch)
        )
    if git_head != bundle.identity.git_commit:
        raise RuntimeError(
            f"live git head {git_head} differs from checkpoint {bundle.identity.git_commit}"
        )

    config = load_config(args.config)
    if config.digest(include_paths=False) != bundle.identity.config_digest:
        raise ValueError("formal config digest differs from checkpoint identity")
    if deployment_graph_config(config) != deployment_graph_config(bundle.config):
        raise ValueError("formal data config graph differs from checkpoint graph")
    print("[data] loading formal multitask validation bundle", flush=True)
    data = load_mainline_data(config)
    live_dataset_identity = dataset_identity(data, config)
    if live_dataset_identity != bundle.identity.dataset:
        raise ValueError(
            "formal dataset/cache identity differs from checkpoint: "
            f"live={live_dataset_identity!r}, saved={bundle.identity.dataset!r}"
        )
    live_language_identity = language_identity(data, config)
    language_fields = ("logical_name", "size_bytes", "sha256")
    if any(
        getattr(live_language_identity, field) != getattr(bundle.identity.language, field)
        for field in language_fields
    ):
        raise ValueError(
            "formal language identity differs from checkpoint: "
            f"live={live_language_identity!r}, saved={bundle.identity.language!r}"
        )
    if v120_normalizer_fingerprint(data.action_normalizer) != v120_normalizer_fingerprint(
        bundle.action_normalizer
    ):
        raise ValueError("formal action normalizer differs from checkpoint")
    if v120_normalizer_fingerprint(data.state_normalizer) != v120_normalizer_fingerprint(
        bundle.state_normalizer
    ):
        raise ValueError("formal state normalizer differs from checkpoint")

    manifest = _selected_manifest(data)
    manifest_hash = _canonical_sha256(manifest)
    intervention_manifest = [manifest[2 * ordinal] for ordinal in INTERVENTION_TASK_ORDINALS]
    print(
        f"[manifest] frozen {len(manifest)} windows across {len(data.task_order)} tasks; "
        f"sha256={manifest_hash}",
        flush=True,
    )
    model = bundle.model.to(device).eval()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("deployment model unexpectedly owns gradients before the probe")
    dtype = resolve_compute_dtype(config)
    matrix, row_projection, null_projection = _projection_operators(device)
    del null_projection

    print("[panel] decomposing paired expert/proposal errors", flush=True)
    panel = _panel_pass(
        data=data,
        config=config,
        model=model,
        dtype=dtype,
        normalizer=bundle.action_normalizer,
        device=device,
        manifest=manifest,
        matrix=matrix,
        row_projection=row_projection,
        base_seed=int(args.seed),
        batch_size=int(args.panel_batch_size),
    )
    print("[intervention] running four predeclared contexts x two seeds", flush=True)
    interventions = _intervention_pass(
        data=data,
        config=config,
        model=model,
        dtype=dtype,
        normalizer=bundle.action_normalizer,
        device=device,
        manifest=manifest,
        matrix=matrix,
        row_projection=row_projection,
        base_seed=int(args.seed),
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only probe populated model gradients")

    matrix_cpu = matrix.detach().cpu()
    report = {
        "schema": SCHEMA,
        "protocol": {
            "seed": int(args.seed),
            "split": "val",
            "selection": "per task: first episode at 1/3 valid-window ordinal and last episode at 2/3",
            "intervention_task_ordinals": list(INTERVENTION_TASK_ORDINALS),
            "intervention_seeds": [int(args.seed) + 2000, int(args.seed) + 2001],
            "amplitude_policy": "per sample min(row-space RMS, null-space RMS), no clipping, report range",
            "gripper_policy": "retain proposal binary command mapped through checkpoint normalizer",
            "execution_mode": "learned",
            "deployment_fastpath": False,
            "proposal_and_refined_noise_policy": "same proposal initial physical noise reused by every refined branch",
            "training_or_parameter_updates": False,
        },
        "identity": {
            "repo_root": str(repo_root),
            "git_commit": git_head,
            "source_digest": live_source.digest,
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "checkpoint_epoch": int(bundle.epoch),
            "checkpoint_global_step": int(bundle.global_step),
            "import_owners": import_owners,
            "checkpoint_identity": bundle.identity.as_dict(),
            "graph_config": deployment_graph_config(config),
            "action_normalizer_fingerprint": v120_normalizer_fingerprint(data.action_normalizer),
            "state_normalizer_fingerprint": v120_normalizer_fingerprint(data.state_normalizer),
            "live_dataset_identity": {
                name: getattr(live_dataset_identity, name)
                for name in live_dataset_identity.__dataclass_fields__
            },
            "live_language_identity": {
                name: getattr(live_language_identity, name)
                for name in live_language_identity.__dataclass_fields__
            },
            "task_order": list(data.task_order),
        },
        "operator": {
            "matrix": matrix_cpu.tolist(),
            "rank": int(torch.linalg.matrix_rank(matrix_cpu)),
            "nullity": int(HORIZON - torch.linalg.matrix_rank(matrix_cpu)),
            "interval_rows_1based_inclusive": [list(value) for value in INTERVAL_ROWS_1BASED],
            "first_three_columns_zero": bool(torch.count_nonzero(matrix_cpu[:, :3]) == 0),
            "row_8_overlap": [float(matrix_cpu[index, 7]) for index in range(4)],
            "row_16_overlap": [float(matrix_cpu[index, 15]) for index in range(4)],
            "row_24_overlap": [float(matrix_cpu[index, 23]) for index in range(4)],
            "calvin_adapter_linear_fingerprint": "J=[cumsum(M); M] for fixed affine offset/boundary",
        },
        "manifest": {
            "sha256": manifest_hash,
            "rows": manifest,
            "intervention_rows": intervention_manifest,
        },
        "panel": panel,
        "interventions": interventions,
    }
    report["summary"] = _aggregate(panel, interventions)
    return report


def main() -> None:
    args = _parser().parse_args()
    report = run(args)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
