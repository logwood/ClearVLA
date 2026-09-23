"""Bounded production-path qualification with explicitly named input provenance.

Synthetic tensors qualify numerical execution, not datasets, encoders or skills.
No environment is installed or upgraded; external runtime mismatch is recorded.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from clearvla.data.history_clock import sparse_history_clock
from clearvla.data.window_boundaries import OBSERVED_TAIL_V1

from ..config import ExperimentConfig
from ..data.normalizer import ArrayNormalizer
from ..executed_world import ExecutedWorldWindow
from ..future_time import resolve_future_time
from ..instruction_reference import INSTRUCTION_START_REFERENCE, InstructionReference
from ..interfaces import (
    ActionSupervision,
    CurrentObservation,
    FutureSupervision,
    GoalCondition,
    ObservableHistory,
    OnlinePolicyInput,
    TrainingBatch,
)
from ..robot_execution import ExecutedRobotStep
from ..supervision import FutureLabelSupport
from ..temporal import TIMED_HISTORY_ENCODING, HistoryTiming


def version_satisfies(version: str, constraints: str) -> bool:
    """Conservative stable-release numeric comparison; never admit a prerelease."""
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[-\w.]+)?", version)
    if match is None:
        return False
    current = tuple(int(x or 0) for x in match.groups())
    for rule in constraints.split(","):
        bound = re.fullmatch(r"\s*(>=|<=|==|>|<)(\d+)\.(\d+)(?:\.(\d+))?\s*", rule)
        if bound is None:
            raise ValueError(f"unhandled runtime requirement: {constraints}")
        op, *digits = bound.groups()
        target = tuple(int(x or 0) for x in digits)
        if not {
            ">=": current >= target,
            "<=": current <= target,
            "==": current == target,
            ">": current > target,
            "<": current < target,
        }[op]:
            return False
    return True


def declared_runtime(repo: Path, python_version: str, torch_version: str) -> dict[str, Any]:
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    py = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    pt = re.search(r'"torch([<>=][^"]+)"', text)
    if py is None or pt is None:
        raise ValueError("project runtime requirements were not found")
    python_ok = version_satisfies(python_version, py.group(1))
    torch_ok = version_satisfies(torch_version, pt.group(1))
    return {
        "python_requirement": py.group(1),
        "torch_requirement": pt.group(1),
        "python_match": python_ok,
        "torch_match": torch_ok,
        "match": python_ok and torch_ok,
    }


def required_data_paths(config: ExperimentConfig) -> dict[str, dict[str, Any]]:
    result = {}
    for name in (
        "raw_hdf5_root",
        "decoded_cache",
        "dino_cache",
        "t5_condition",
        "split_manifest",
        "calvin_raw_source",
    ):
        raw = getattr(config.data, name)
        if raw is not None and str(raw):
            path = Path(raw).expanduser()
            result[name] = {"path": str(path), "exists": path.exists()}
    return result


def synthetic_batch(
    config: ExperimentConfig, *, count: int, raw_side: int, device: torch.device, seed: int = 8301
) -> tuple[TrainingBatch, ArrayNormalizer]:
    """Normalized fixture at requested dimensions; no image encoder or real data.

    Histories describe physical t=24 with a real t-1 command. Future labels use
    exactly that configuration's grid and one shared source-support record.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("qualification batch count must be a positive integer")
    if type(raw_side) is not int or raw_side < 32 or raw_side % 16:
        raise ValueError("synthetic raw side must be >=32 and divisible by 16")
    dims = config.dimensions
    grid = resolve_future_time(config.top.future_time_grid_mode)
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return (torch.randn(shape, generator=g) * 0.1).to(device)

    def mask(*shape: int) -> Tensor:
        return torch.ones(shape, dtype=torch.bool, device=device)

    state = rand(count, dims.state_dim)
    history = rand(count, dims.state_history_length, dims.state_dim)
    history[:, -1] = state
    executed = rand(count, dims.executed_history_length, dims.action_dim)
    executed[..., -1] = 1.0
    action_state = executed[:, -1].clone()
    dino = rand(
        count,
        dims.visual_history_length,
        dims.num_cameras,
        dims.patches_per_camera,
        dims.visual_token_dim,
    )
    timing = None
    if (
        config.top.history_encoding_mode == TIMED_HISTORY_ENCODING
        or config.observation.source_time_mode == "source_history_steps_v1"
    ):
        timing = HistoryTiming.from_mapping(
            {
                k: torch.from_numpy(v.copy())[None].expand(count, -1).clone().to(device)
                for k, v in sparse_history_clock(24).items()
            }
        )
    robot_step = None
    if config.top.robot_feedback_mode != "none":
        robot_step = ExecutedRobotStep(
            rand(count, dims.state_dim),
            executed[:, -1].clone(),
            mask(count),
            torch.tensor([[-1, -1, 0]], device=device).expand(count, -1).clone(),
        )
    reference = None
    if config.top.instruction_reference_mode == INSTRUCTION_START_REFERENCE:
        reference = InstructionReference(
            dino[:, 0].clone(),
            history[:, 0].clone(),
            mask(count, dims.num_cameras, dims.patches_per_camera),
            torch.full((count,), 8, dtype=torch.long, device=device),
        )
    raw=rand(count,dims.visual_history_length,dims.num_cameras,3,raw_side,raw_side)
    world_window=None
    if config.top.world_feedback_mode!="none":
        commands=rand(count,4,dims.action_dim)
        commands[...,-1]=1.
        if timing is None:
            raise ValueError("synthetic executed world requires physical history chart")
        for j,offset in enumerate((-4,-3,-2,-1)):
            for i,t in enumerate((-24,-16,-12,-8,-6,-4,-2,-1)):
                if offset==t:
                    commands[:,j]=executed[:,i]
        world_window=ExecutedWorldWindow(
            dino_history=torch.cat((rand(count,1,dims.num_cameras,dims.patches_per_camera,dims.visual_token_dim),dino[:,:2]),1),
            raw_rgb=torch.cat((rand(count,1,dims.num_cameras,3,raw_side,raw_side),raw[:,:2]),1),
            state=history[:,1].clone(),action_state=rand(count,dims.action_dim),commands=commands,observed=mask(count),
            visual_offsets=torch.tensor([[-8,-4,0]],device=device).expand(count,-1).clone())
    online = OnlinePolicyInput(
        CurrentObservation(dino,raw),
        ObservableHistory(
            state,
            action_state,
            action_state[:, -1:].clone(),
            history,
            executed,
            timing=timing,
            executed_robot_step=robot_step,
            executed_world_window=world_window,
        ),
        GoalCondition(
            rand(count, dims.goal_max_tokens, dims.goal_token_dim),
            mask(count, dims.goal_max_tokens),
        ),
        instruction_reference=reference,
    )
    support = None
    if config.data.window_boundary_contract == OBSERVED_TAIL_V1:
        support = FutureLabelSupport(
            mask(count, grid.horizon), mask(count, grid.horizon), mask(count, dims.future_supports)
        )
    actions = rand(count, grid.horizon, dims.action_dim)
    actions[..., -1] = torch.where(actions[..., -1] >= 0, 1.0, -1.0)
    future = FutureSupervision(
        rand(
            count,
            dims.future_supports,
            dims.num_cameras,
            dims.patches_per_camera,
            dims.visual_token_dim,
        ),
        actions,
        state[:, None] + rand(count, grid.horizon, dims.state_dim),
        torch.tensor(grid.support_offsets, dtype=torch.long, device=device)[None]
        .expand(count, -1)
        .clone(),
        support=support,
        time_grid_mode=grid.mode,
    )
    target = actions[:, : dims.action_horizon].clone()
    batch = TrainingBatch(
        online,
        ActionSupervision(
            target,
            target.clone(),
            action_state.clone(),
            action_state.clone(),
            action_state.clone(),
            support=support,
        ),
        future,
    )
    batch.validate(config)
    normalizer = ArrayNormalizer.fit_identity(
        [np.stack((-np.ones(dims.action_dim), np.ones(dims.action_dim))).astype(np.float32)]
    )
    return batch, normalizer


def qualification_config(config: ExperimentConfig, dtype: str | None) -> ExperimentConfig:
    # Dtype is the only model-numerics override. All dimensions, graphs, solver
    # schedules and source semantics remain those of the supplied configuration.
    resolved = (
        config
        if dtype is None
        else replace(config, runtime=replace(config.runtime, compute_dtype=dtype))
    )
    resolved.validate()
    return resolved
