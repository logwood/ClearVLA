#!/usr/bin/env python3
"""Audit closed-loop consistency and control cadence in a CALVIN action trace.

The trace contains planned 24-row action chunks and the rows actually sent to
CALVIN. Adjacent plans are aligned in physical environment time before they
are compared: with ``execute_rows=4``, old rows 4:24 and new rows 0:20 describe
the same future control times. Segment resets are inferred from the bridge's
causal-history indices and are excluded from transition statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import numpy as np


TRANSLATION_SCALE_M = 0.02
ROTATION_SCALE_RAD = 0.05
ARM_DIM = 6


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    return [
        (int(values[start]), int(start), int(end - start))
        for start, end in zip(starts, ends)
    ]


def _float(value: float | np.floating) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _rmse(value: np.ndarray, *, axis: int | tuple[int, ...] | None = None):
    value = np.asarray(value, dtype=np.float64)
    result = np.sqrt(np.mean(np.square(value), axis=axis))
    if np.ndim(result) == 0:
        return _float(result)
    return np.asarray(result, dtype=np.float64).tolist()


def _quantiles(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": _float(value.mean()),
        "p50": _float(np.quantile(value, 0.50)),
        "p90": _float(np.quantile(value, 0.90)),
        "p95": _float(np.quantile(value, 0.95)),
        "max": _float(value.max()),
    }


def _cosine_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1, left.shape[-1])
    right = np.asarray(right, dtype=np.float64).reshape(-1, right.shape[-1])
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    valid = denom > 1e-12
    cosine = np.divide(
        np.sum(left * right, axis=-1),
        denom,
        out=np.zeros_like(denom),
        where=valid,
    )
    selected = cosine[valid]
    if selected.size == 0:
        return {
            "valid_rows": 0,
            "mean": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "negative_fraction": 0.0,
        }
    return {
        "valid_rows": int(selected.size),
        "mean": _float(selected.mean()),
        "p10": _float(np.quantile(selected, 0.10)),
        "p50": _float(np.quantile(selected, 0.50)),
        "negative_fraction": _float(np.mean(selected < 0.0)),
    }


def _active_sign_disagreement(
    left: np.ndarray,
    right: np.ndarray,
    *,
    threshold: float = 0.05,
) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64).reshape(-1, left.shape[-1])
    right = np.asarray(right, dtype=np.float64).reshape(-1, right.shape[-1])
    active = (np.abs(left) >= threshold) & (np.abs(right) >= threshold)
    mismatch = np.signbit(left) != np.signbit(right)
    count = active.sum(axis=0)
    fraction = np.divide(
        (active & mismatch).sum(axis=0),
        count,
        out=np.zeros(count.shape, dtype=np.float64),
        where=count > 0,
    )
    return {
        "magnitude_threshold": float(threshold),
        "active_comparisons_per_dim": count.astype(int).tolist(),
        "disagreement_fraction_per_dim": fraction.tolist(),
    }


def _infer_segments(
    history_time: np.ndarray,
    observed_after_steps: np.ndarray,
    total_steps: int,
) -> list[tuple[int, int]]:
    history_time = np.asarray(history_time, dtype=np.int64)
    observed_after_steps = np.asarray(observed_after_steps, dtype=np.int64)
    if history_time.ndim != 1 or observed_after_steps.ndim != 1:
        raise ValueError("history audit arrays must be one-dimensional")
    if history_time.shape != observed_after_steps.shape:
        raise ValueError("history audit arrays do not align")
    if history_time.size == 0:
        return [(0, int(total_steps))]
    reset_rows = np.flatnonzero(history_time[1:] <= history_time[:-1]) + 1
    starts = [0, *(int(observed_after_steps[index]) - 1 for index in reset_rows)]
    ends = [*starts[1:], int(total_steps)]
    segments = list(zip(starts, ends))
    if any(start < 0 or end <= start for start, end in segments):
        raise ValueError(f"inferred invalid segment boundaries: {segments}")
    if segments[0][0] != 0 or segments[-1][1] != int(total_steps):
        raise ValueError("segment boundaries do not cover the complete execution")
    return segments


def _same_segment_transition_mask(
    segments: Iterable[tuple[int, int]], total_steps: int
) -> np.ndarray:
    segment_id = np.full(int(total_steps), -1, dtype=np.int64)
    for index, (start, end) in enumerate(segments):
        segment_id[start:end] = index
    if np.any(segment_id < 0):
        raise ValueError("segments do not cover every executed step")
    return segment_id[1:] == segment_id[:-1]


def _transition_metrics(delta: np.ndarray) -> dict[str, object]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != ARM_DIM:
        raise ValueError("arm transition delta must be [N,6]")
    return {
        "count": int(delta.shape[0]),
        "translation_scalar_rmse_normalized": _rmse(delta[:, :3]),
        "rotation_scalar_rmse_normalized": _rmse(delta[:, 3:]),
        "arm_scalar_rmse_normalized": _rmse(delta),
        "translation_jump_norm_normalized": _quantiles(
            np.linalg.norm(delta[:, :3], axis=-1)
        ),
        "rotation_jump_norm_normalized": _quantiles(
            np.linalg.norm(delta[:, 3:], axis=-1)
        ),
    }


def _segment_summary(
    index: int,
    start: int,
    end: int,
    executed: np.ndarray,
    plan_index: np.ndarray,
    raw_chunks: np.ndarray,
    *,
    max_subtask_steps: int,
    execute_rows: int,
) -> dict[str, object]:
    action = executed[start:end]
    arm = action[:, :ARM_DIM]
    translation = arm[:, :3]
    rotation = arm[:, 3:]
    gripper = np.where(action[:, 6] >= 0.0, 1, -1).astype(np.int8)
    path_m = np.linalg.norm(translation, axis=-1).sum() * TRANSLATION_SCALE_M
    net_m = np.linalg.norm(translation.sum(axis=0)) * TRANSLATION_SCALE_M
    rotation_path = np.linalg.norm(rotation, axis=-1).sum() * ROTATION_SCALE_RAD
    rotation_net = np.linalg.norm(rotation.sum(axis=0)) * ROTATION_SCALE_RAD
    adjacent_translation = _cosine_stats(translation[:-1], translation[1:])
    plans = np.unique(plan_index[start:end])
    if plans.size > 1:
        old = raw_chunks[plans[:-1]]
        new = raw_chunks[plans[1:]]
        overlap = new[:, : 24 - execute_rows] - old[:, execute_rows:]
        overlap_arm = overlap[..., :ARM_DIM]
        near_old = old[:, execute_rows, :ARM_DIM]
        near_new = new[:, 0, :ARM_DIM]
        replan = {
            "pairs": int(plans.size - 1),
            "overlap_arm_scalar_rmse_normalized": _rmse(overlap_arm),
            "overlap_translation_scalar_rmse_normalized": _rmse(overlap_arm[..., :3]),
            "overlap_rotation_scalar_rmse_normalized": _rmse(overlap_arm[..., 3:]),
            "near_translation_cosine": _cosine_stats(near_old[:, :3], near_new[:, :3]),
            "near_rotation_cosine": _cosine_stats(near_old[:, 3:], near_new[:, 3:]),
            "near_gripper_disagreement_fraction": _float(
                np.mean(old[:, execute_rows, 6] != new[:, 0, 6])
            ),
            "overlap_gripper_disagreement_fraction": _float(
                np.mean(old[:, execute_rows:, 6] != new[:, : 24 - execute_rows, 6])
            ),
        }
    else:
        replan = {
            "pairs": 0,
            "overlap_arm_scalar_rmse_normalized": 0.0,
            "overlap_translation_scalar_rmse_normalized": 0.0,
            "overlap_rotation_scalar_rmse_normalized": 0.0,
            "near_translation_cosine": {},
            "near_rotation_cosine": {},
            "near_gripper_disagreement_fraction": 0.0,
            "overlap_gripper_disagreement_fraction": 0.0,
        }
    return {
        "segment": int(index),
        "start_step": int(start),
        "end_step_exclusive": int(end),
        "steps": int(end - start),
        "eval_success": bool(end - start < max_subtask_steps),
        "plans": int(plans.size),
        "translation_command_path_m": _float(path_m),
        "translation_net_command_m": _float(net_m),
        "translation_net_to_path_ratio": _float(net_m / max(path_m, 1e-12)),
        "rotation_command_path_rad": _float(rotation_path),
        "rotation_net_command_rad": _float(rotation_net),
        "rotation_net_to_path_ratio": _float(rotation_net / max(rotation_path, 1e-12)),
        "adjacent_translation_cosine": adjacent_translation,
        "gripper_open_fraction": _float(np.mean(gripper > 0)),
        "gripper_switches": int(np.count_nonzero(gripper[1:] != gripper[:-1])),
        "gripper_runs": _runs(gripper),
        "replan": replan,
    }


def _cohort_summary(
    segments: list[dict[str, object]], *, success: bool
) -> dict[str, object]:
    selected = [row for row in segments if bool(row["eval_success"]) is success]
    if not selected:
        return {"segments": 0}

    def mean(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for row in selected:
            value: object = row
            for key in path:
                if not isinstance(value, dict):
                    raise TypeError(path)
                value = value[key]
            values.append(float(value))
        return _float(np.mean(values))

    return {
        "segments": len(selected),
        "mean_steps": mean(("steps",)),
        "mean_translation_command_path_m": mean(("translation_command_path_m",)),
        "mean_translation_net_to_path_ratio": mean(("translation_net_to_path_ratio",)),
        "mean_adjacent_translation_cosine": mean(
            ("adjacent_translation_cosine", "mean")
        ),
        "mean_adjacent_translation_negative_fraction": mean(
            ("adjacent_translation_cosine", "negative_fraction")
        ),
        "mean_gripper_switches": mean(("gripper_switches",)),
        "mean_replan_overlap_arm_scalar_rmse_normalized": mean(
            ("replan", "overlap_arm_scalar_rmse_normalized")
        ),
        "mean_replan_near_translation_cosine": mean(
            ("replan", "near_translation_cosine", "mean")
        ),
        "mean_replan_near_translation_negative_fraction": mean(
            ("replan", "near_translation_cosine", "negative_fraction")
        ),
        "mean_replan_near_gripper_disagreement_fraction": mean(
            ("replan", "near_gripper_disagreement_fraction")
        ),
    }


def _shared_template_summary(
    executed: np.ndarray,
    segments: list[tuple[int, int]],
    *,
    full_length: int,
) -> dict[str, object]:
    """Measure task-relative structure shared by all full-length segments.

    The bridge policy resets its generator at every subtask, so equal relative
    step indices also receive equal sampler-noise ordinals.  These statistics
    expose shared timing but deliberately do not assign it uniquely to noise,
    common initial robot geometry, or a learned generic phase template.
    """

    selected = [
        executed[start:end]
        for start, end in segments
        if end - start == int(full_length)
    ]
    if len(selected) < 2:
        return {"full_length_segments": len(selected)}
    action = np.stack(selected, axis=0)
    arm = action[..., :ARM_DIM]
    gripper = np.where(action[..., 6] >= 0.0, 1, -1).astype(np.int8)
    segment_count, steps = gripper.shape
    pairs = [
        (left, right)
        for left in range(segment_count)
        for right in range(left + 1, segment_count)
    ]
    aligned_gripper_agreement = _float(
        np.mean([np.mean(gripper[left] == gripper[right]) for left, right in pairs])
    )
    open_fraction = np.mean(gripper > 0, axis=1)
    marginal_gripper_agreement = _float(
        np.mean(
            [
                open_fraction[left] * open_fraction[right]
                + (1.0 - open_fraction[left]) * (1.0 - open_fraction[right])
                for left, right in pairs
            ]
        )
    )
    majority_fraction = np.maximum(
        np.mean(gripper > 0, axis=0), np.mean(gripper < 0, axis=0)
    )
    switch = gripper[:, 1:] != gripper[:, :-1]
    switch_count_by_time = switch.sum(axis=0)
    switch_peaks = sorted(
        (
            {"step": int(index + 1), "segments_switching": int(count)}
            for index, count in enumerate(switch_count_by_time)
            if count > 0
        ),
        key=lambda row: (-row["segments_switching"], row["step"]),
    )[:20]

    def pair_correlations(value: np.ndarray) -> tuple[list[float], float]:
        axis_means: list[float] = []
        all_values: list[float] = []
        for dim in range(value.shape[-1]):
            correlations: list[float] = []
            for left, right in pairs:
                left_value = value[left, :, dim]
                right_value = value[right, :, dim]
                if np.std(left_value) <= 1e-12 or np.std(right_value) <= 1e-12:
                    continue
                correlations.append(
                    _float(np.corrcoef(left_value, right_value)[0, 1])
                )
            axis_means.append(_float(np.mean(correlations)))
            all_values.extend(correlations)
        return axis_means, _float(np.mean(all_values))

    level_axis, level_mean = pair_correlations(arm)
    delta_axis, delta_mean = pair_correlations(np.diff(arm, axis=1))
    return {
        "full_length_segments": int(segment_count),
        "pair_count": int(len(pairs)),
        "steps_per_segment": int(steps),
        "aligned_gripper_pair_agreement": aligned_gripper_agreement,
        "gripper_pair_agreement_expected_from_segment_marginals": (
            marginal_gripper_agreement
        ),
        "aligned_gripper_timing_excess": _float(
            aligned_gripper_agreement - marginal_gripper_agreement
        ),
        "mean_per_step_gripper_majority_fraction": _float(majority_fraction.mean()),
        "gripper_switch_counts_per_segment": switch.sum(axis=1).astype(int).tolist(),
        "top_shared_gripper_switch_times": switch_peaks,
        "aligned_arm_level_pair_correlation_per_dim": level_axis,
        "aligned_arm_level_pair_correlation_mean": level_mean,
        "aligned_arm_step_delta_pair_correlation_per_dim": delta_axis,
        "aligned_arm_step_delta_pair_correlation_mean": delta_mean,
        "interpretation_limit": (
            "Shared timing can reflect repeated sampler noise, common robot-start "
            "geometry, or a learned task-relative phase template. Different-seed "
            "and fixed-noise matched rollouts are required to separate them."
        ),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size == 0:
        raise ValueError("correlation operands must be non-empty and aligned")
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return _float(np.corrcoef(left, right)[0, 1])


def _labelled_pair_comparison(
    left: dict[str, object],
    right: dict[str, object],
    executed: np.ndarray,
    plan_index: np.ndarray,
    raw_chunks: np.ndarray,
    *,
    execute_rows: int,
) -> dict[str, object]:
    left_start = int(left["start_step"])
    right_start = int(right["start_step"])
    shared_steps = min(int(left["steps"]), int(right["steps"]))
    left_action = executed[left_start : left_start + shared_steps]
    right_action = executed[right_start : right_start + shared_steps]
    action_rms = _rmse(
        np.concatenate((left_action[:, :ARM_DIM], right_action[:, :ARM_DIM]), axis=0)
    )
    action_rmse = _rmse(left_action[:, :ARM_DIM] - right_action[:, :ARM_DIM])
    left_plans = np.unique(
        plan_index[left_start : int(left["end_step_exclusive"])]
    )
    right_plans = np.unique(
        plan_index[right_start : int(right["end_step_exclusive"])]
    )
    shared_plans = min(len(left_plans), len(right_plans))
    left_chunks = raw_chunks[left_plans[:shared_plans], :, :ARM_DIM]
    right_chunks = raw_chunks[right_plans[:shared_plans], :, :ARM_DIM]
    chunk_rms = _rmse(np.concatenate((left_chunks, right_chunks), axis=0))
    chunk_rmse = _rmse(left_chunks - right_chunks)
    left_shape = left_chunks - left_chunks.mean(axis=1, keepdims=True)
    right_shape = right_chunks - right_chunks.mean(axis=1, keepdims=True)
    same_ordinal_shape_correlation = _float(
        np.mean(
            [
                _correlation(left_shape[index], right_shape[index])
                for index in range(shared_plans)
            ]
        )
    )
    # The shifted comparison uses the next noise ordinal but aligns physical
    # future time: old rows K:24 and next-plan rows 0:24-K.
    shifted_shape_correlation = (
        _float(
            np.mean(
                [
                    _correlation(
                        left_chunks[index, execute_rows:]
                        - left_chunks[index, execute_rows:].mean(
                            axis=0, keepdims=True
                        ),
                        right_chunks[index + 1, :-execute_rows]
                        - right_chunks[index + 1, :-execute_rows].mean(
                            axis=0, keepdims=True
                        ),
                    )
                    for index in range(shared_plans - 1)
                ]
            )
        )
        if shared_plans > 1
        else 0.0
    )
    return {
        "left_segment": int(left["segment"]),
        "right_segment": int(right["segment"]),
        "left_sequence": int(left["sequence"]),
        "right_sequence": int(right["sequence"]),
        "left_task": str(left["task"]),
        "right_task": str(right["task"]),
        "left_success": bool(left["eval_success"]),
        "right_success": bool(right["eval_success"]),
        "shared_steps": int(shared_steps),
        "executed_arm_scalar_rmse_normalized": action_rmse,
        "executed_arm_rmse_to_action_rms_ratio": _float(
            action_rmse / max(action_rms, 1e-12)
        ),
        "executed_arm_pair_correlation_per_dim": [
            _correlation(left_action[:, dim], right_action[:, dim])
            for dim in range(ARM_DIM)
        ],
        "executed_arm_pair_correlation_mean": _float(
            np.mean(
                [
                    _correlation(left_action[:, dim], right_action[:, dim])
                    for dim in range(ARM_DIM)
                ]
            )
        ),
        "executed_gripper_agreement": _float(
            np.mean(left_action[:, 6] == right_action[:, 6])
        ),
        "shared_plan_ordinals": int(shared_plans),
        "same_noise_ordinal_chunk_arm_scalar_rmse_normalized": chunk_rmse,
        "same_noise_ordinal_chunk_arm_rmse_to_action_rms_ratio": _float(
            chunk_rmse / max(chunk_rms, 1e-12)
        ),
        "same_noise_ordinal_horizon_shape_correlation": (
            same_ordinal_shape_correlation
        ),
        "next_noise_ordinal_time_aligned_horizon_shape_correlation": (
            shifted_shape_correlation
        ),
        "chunk_gripper_agreement": _float(
            np.mean(
                raw_chunks[left_plans[:shared_plans], :, 6]
                == raw_chunks[right_plans[:shared_plans], :, 6]
            )
        ),
    }


def _labelled_comparisons(
    segment_rows: list[dict[str, object]],
    executed: np.ndarray,
    plan_index: np.ndarray,
    raw_chunks: np.ndarray,
    *,
    execute_rows: int,
) -> dict[str, object]:
    if not segment_rows or "task" not in segment_rows[0]:
        return {"available": False}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in segment_rows:
        grouped[str(row["task"])].append(row)
    repeated: list[dict[str, object]] = []
    for task, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        for left, right in combinations(rows, 2):
            comparison = _labelled_pair_comparison(
                left,
                right,
                executed,
                plan_index,
                raw_chunks,
                execute_rows=execute_rows,
            )
            comparison["comparison"] = "same_task"
            comparison["task"] = task
            repeated.append(comparison)
    contrast_specs = (
        ("open_drawer", "close_drawer"),
        ("push_blue_block_left", "push_blue_block_right"),
    )
    contrasts: list[dict[str, object]] = []
    for left_task, right_task in contrast_specs:
        for left, right in product(grouped.get(left_task, ()), grouped.get(right_task, ())):
            comparison = _labelled_pair_comparison(
                left,
                right,
                executed,
                plan_index,
                raw_chunks,
                execute_rows=execute_rows,
            )
            comparison["comparison"] = "semantic_contrast"
            contrasts.append(comparison)
    return {
        "available": True,
        "same_task_pairs": repeated,
        "semantic_contrast_pairs": contrasts,
        "interpretation_limit": (
            "Pairs share sampler-noise ordinals because the generator resets per "
            "subtask. Similarity proves weak rollout-specific correction only when "
            "the paired physical trajectories actually diverged; it does not by "
            "itself separate noise from a task-conditioned open-loop template."
        ),
    }


def analyze(
    path: Path,
    *,
    execute_rows: int = 4,
    max_subtask_steps: int = 180,
    segment_labels: dict[int, dict[str, object]] | None = None,
) -> dict[str, object]:
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "executed",
        "raw_first",
        "raw_chunks",
        "executed_plan_index",
        "executed_chunk_row",
        "observed_history_time_index",
        "observed_after_executed_steps",
        "latency_seconds",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"actions archive is missing {missing}")
    executed = np.asarray(arrays["executed"], dtype=np.float64)
    raw_first = np.asarray(arrays["raw_first"], dtype=np.float64)
    raw_chunks = np.asarray(arrays["raw_chunks"], dtype=np.float64)
    plan_index = np.asarray(arrays["executed_plan_index"], dtype=np.int64)
    chunk_row = np.asarray(arrays["executed_chunk_row"], dtype=np.int64)
    latency = np.asarray(arrays["latency_seconds"], dtype=np.float64)
    if executed.ndim != 2 or executed.shape[1] != 7:
        raise ValueError(f"executed must be [N,7], got {executed.shape}")
    if raw_chunks.ndim != 3 or raw_chunks.shape[1:] != (24, 7):
        raise ValueError(f"raw_chunks must be [P,24,7], got {raw_chunks.shape}")
    if raw_first.shape != (raw_chunks.shape[0], 7):
        raise ValueError("raw_first does not align with raw_chunks")
    if plan_index.shape != (executed.shape[0],) or chunk_row.shape != plan_index.shape:
        raise ValueError("executed plan/row indices do not align with executed actions")
    if not 1 <= int(execute_rows) < raw_chunks.shape[1]:
        raise ValueError("execute_rows must be in [1,23]")
    segments = _infer_segments(
        arrays["observed_history_time_index"],
        arrays["observed_after_executed_steps"],
        executed.shape[0],
    )
    same_segment = _same_segment_transition_mask(segments, executed.shape[0])
    arm_delta = np.diff(executed[:, :ARM_DIM], axis=0)
    boundary = same_segment & (chunk_row[1:] == 0)
    within_chunk = same_segment & (chunk_row[1:] != 0)

    adjacent_plan_pairs: list[tuple[int, int, int]] = []
    for segment_index, (start, end) in enumerate(segments):
        plans = np.unique(plan_index[start:end])
        adjacent_plan_pairs.extend(
            (segment_index, int(old_plan), int(new_plan))
            for old_plan, new_plan in zip(plans[:-1], plans[1:])
        )
    old_index = np.asarray([old for _, old, _ in adjacent_plan_pairs], dtype=np.int64)
    new_index = np.asarray([new for _, _, new in adjacent_plan_pairs], dtype=np.int64)
    old = raw_chunks[old_index]
    new = raw_chunks[new_index]
    overlap_rows = raw_chunks.shape[1] - int(execute_rows)
    old_overlap = old[:, execute_rows:]
    new_overlap = new[:, :overlap_rows]
    overlap_delta = new_overlap - old_overlap
    old_arm = old_overlap[..., :ARM_DIM]
    new_arm = new_overlap[..., :ARM_DIM]
    overlap_arm_delta = overlap_delta[..., :ARM_DIM]
    near_old_arm = old[:, execute_rows, :ARM_DIM]
    near_new_arm = new[:, 0, :ARM_DIM]
    pair_arm_rmse = np.sqrt(np.mean(np.square(overlap_arm_delta), axis=(1, 2)))
    planned_four_step_delta = (
        old[:, execute_rows:, :ARM_DIM] - old[:, :-execute_rows, :ARM_DIM]
    )
    continued_boundary_delta = (
        old[:, execute_rows, :ARM_DIM] - old[:, execute_rows - 1, :ARM_DIM]
    )
    replanned_boundary_delta = (
        new[:, 0, :ARM_DIM] - old[:, execute_rows - 1, :ARM_DIM]
    )
    action_arm_rms = _rmse(np.concatenate((old_arm, new_arm), axis=0))
    overlap_arm_rmse = _rmse(overlap_arm_delta)

    segment_rows = [
        _segment_summary(
            index,
            start,
            end,
            executed,
            plan_index,
            raw_chunks,
            max_subtask_steps=max_subtask_steps,
            execute_rows=execute_rows,
        )
        for index, (start, end) in enumerate(segments)
    ]
    if segment_labels is not None:
        if set(segment_labels) != set(range(len(segment_rows))):
            raise ValueError("segment label manifest does not cover every segment")
        for row in segment_rows:
            label = segment_labels[int(row["segment"])]
            if int(label["steps"]) != int(row["steps"]):
                raise ValueError(
                    f"segment {row['segment']} label length does not match actions"
                )
            if bool(label["eval_success"]) != bool(row["eval_success"]):
                raise ValueError(
                    f"segment {row['segment']} success label does not match length"
                )
            row.update(
                {
                    "sequence": int(label["sequence"]),
                    "task_index": int(label["task_index"]),
                    "task": str(label["task"]),
                }
            )
    gripper = np.where(executed[:, 6] >= 0.0, 1, -1).astype(np.int8)
    gripper_switch = gripper[1:] != gripper[:-1]
    blind_steps = int(np.count_nonzero(chunk_row != 0))
    return {
        "schema": "clearvla-calvin-closed-loop-action-audit-v1",
        "actions_path": str(path.resolve()),
        "cadence": {
            "execute_rows": int(execute_rows),
            "steps": int(executed.shape[0]),
            "planning_decisions": int(raw_chunks.shape[0]),
            "replanned_steps": int(executed.shape[0] - blind_steps),
            "blind_rows_without_new_policy_inference": blind_steps,
            "blind_row_fraction": _float(blind_steps / executed.shape[0]),
            "maximum_observation_age_steps": int(execute_rows - 1),
            "mean_rows_per_plan": _float(executed.shape[0] / raw_chunks.shape[0]),
            "latency_seconds": _quantiles(latency),
        },
        "segmentation": {
            "segments": len(segments),
            "lengths": [int(end - start) for start, end in segments],
            "early_success_segments": [
                int(row["segment"]) for row in segment_rows if row["eval_success"]
            ],
            "reset_boundaries_excluded": len(segments) - 1,
        },
        "executed_control": {
            "arm_mean_abs_per_dim_normalized": np.mean(
                np.abs(executed[:, :ARM_DIM]), axis=0
            ).tolist(),
            "translation_command_path_m": _float(
                np.linalg.norm(executed[:, :3], axis=-1).sum()
                * TRANSLATION_SCALE_M
            ),
            "translation_net_command_m_by_segment_sum": _float(
                sum(float(row["translation_net_command_m"]) for row in segment_rows)
            ),
            "within_chunk_transition": _transition_metrics(arm_delta[within_chunk]),
            "replan_boundary_transition": _transition_metrics(arm_delta[boundary]),
            "boundary_to_within_arm_jump_rmse_ratio": _float(
                _rmse(arm_delta[boundary]) / max(_rmse(arm_delta[within_chunk]), 1e-12)
            ),
            "gripper_open_steps": int(np.count_nonzero(gripper > 0)),
            "gripper_close_steps": int(np.count_nonzero(gripper < 0)),
            "gripper_switches_excluding_resets": int(
                np.count_nonzero(gripper_switch & same_segment)
            ),
            "gripper_switch_fraction_within_chunk": _float(
                np.mean(gripper_switch[within_chunk])
            ),
            "gripper_switch_fraction_at_replan_boundary": _float(
                np.mean(gripper_switch[boundary])
            ),
        },
        "aligned_replans": {
            "plan_pairs": int(len(adjacent_plan_pairs)),
            "aligned_rows_per_pair": int(overlap_rows),
            "aligned_action_arm_rms_normalized": action_arm_rms,
            "aligned_arm_scalar_rmse_normalized": overlap_arm_rmse,
            "aligned_arm_rmse_to_action_rms_ratio": _float(
                overlap_arm_rmse / max(action_arm_rms, 1e-12)
            ),
            "aligned_translation_scalar_rmse_normalized": _rmse(
                overlap_arm_delta[..., :3]
            ),
            "aligned_rotation_scalar_rmse_normalized": _rmse(
                overlap_arm_delta[..., 3:]
            ),
            "aligned_arm_rmse_by_future_row_normalized": _rmse(
                overlap_arm_delta, axis=(0, 2)
            ),
            "pair_arm_rmse_normalized": _quantiles(pair_arm_rmse),
            "aligned_translation_cosine": _cosine_stats(
                old_arm[..., :3], new_arm[..., :3]
            ),
            "aligned_rotation_cosine": _cosine_stats(
                old_arm[..., 3:], new_arm[..., 3:]
            ),
            "near_old_row4_vs_new_row0_arm_scalar_rmse_normalized": _rmse(
                near_new_arm - near_old_arm
            ),
            "near_old_row4_vs_new_row0_translation_cosine": _cosine_stats(
                near_old_arm[:, :3], near_new_arm[:, :3]
            ),
            "near_old_row4_vs_new_row0_rotation_cosine": _cosine_stats(
                near_old_arm[:, 3:], near_new_arm[:, 3:]
            ),
            "active_arm_sign_disagreement": _active_sign_disagreement(
                old_arm, new_arm
            ),
            "near_gripper_disagreement_fraction": _float(
                np.mean(old[:, execute_rows, 6] != new[:, 0, 6])
            ),
            "aligned_gripper_disagreement_fraction": _float(
                np.mean(old_overlap[..., 6] != new_overlap[..., 6])
            ),
            "same_plan_four_step_arm_delta_rmse_normalized": _rmse(
                planned_four_step_delta
            ),
            "replan_to_same_plan_four_step_delta_rmse_ratio": _float(
                overlap_arm_rmse / max(_rmse(planned_four_step_delta), 1e-12)
            ),
            "old_plan_continued_boundary": _transition_metrics(
                continued_boundary_delta
            ),
            "actual_replanned_boundary": _transition_metrics(
                replanned_boundary_delta
            ),
            "actual_to_old_continuation_arm_jump_rmse_ratio": _float(
                _rmse(replanned_boundary_delta)
                / max(_rmse(continued_boundary_delta), 1e-12)
            ),
        },
        "cohort_comparison": {
            "early_eval_success": _cohort_summary(segment_rows, success=True),
            "full_length_failure": _cohort_summary(segment_rows, success=False),
            "warning": (
                "Descriptive only: 3 success and 16 failure segments have different "
                "tasks and horizons; this is not a matched causal comparison."
            ),
        },
        "shared_task_relative_template": _shared_template_summary(
            executed,
            segments,
            full_length=max_subtask_steps,
        ),
        "labelled_pair_comparisons": _labelled_comparisons(
            segment_rows,
            executed,
            plan_index,
            raw_chunks,
            execute_rows=execute_rows,
        ),
        "segments": segment_rows,
        "limitations": [
            "actions.npz contains commanded actions, not actual TCP/object/contact state",
            "aligned replan differences combine changed observations/history with fresh sampler noise",
            "the trace has no execute_rows=1 or fixed-noise counterfactual",
            "commanded path length is not realized robot path length under IK/contact",
        ],
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path)
    parser.add_argument("--execute-rows", type=int, default=4)
    parser.add_argument("--max-subtask-steps", type=int, default=180)
    parser.add_argument("--segment-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    segment_labels = None
    if args.segment_manifest is not None:
        manifest = json.loads(args.segment_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "clearvla-calvin-rollout-segments-v1":
            raise ValueError("unrecognized segment manifest schema")
        rows = manifest.get("segments")
        if not isinstance(rows, list):
            raise ValueError("segment manifest must contain a segments list")
        segment_labels = {int(row["segment"]): row for row in rows}
    result = analyze(
        args.actions,
        execute_rows=args.execute_rows,
        max_subtask_steps=args.max_subtask_steps,
        segment_labels=segment_labels,
    )
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
