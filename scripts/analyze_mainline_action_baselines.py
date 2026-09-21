#!/usr/bin/env python3
"""Measure data-only action baselines on an exact mainline split.

The tool intentionally reads window references and raw episode arrays without
loading RGB or DINO tensors.  It therefore answers whether a sampled policy is
better than simple source-native command baselines without running the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.data.loading import load_mainline_data
from clearvla.mainline.model.action_codec import (
    PhysicalActionFieldCodec,
    anchor_horizon_weights,
)


_BANDS = (("1_4", slice(0, 4)), ("5_12", slice(4, 12)), ("13_24", slice(12, 24)))


def _base_dataset(dataset: object) -> Any:
    base = getattr(dataset, "base", dataset)
    if not hasattr(base, "refs") or not hasattr(base, "episodes"):
        raise TypeError("mainline dataset does not expose raw window references")
    return base


def _raw_windows(
    dataset: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = _base_dataset(dataset)
    targets: list[np.ndarray] = []
    gripper_boundaries: list[np.ndarray] = []
    previous_commands: list[np.ndarray] = []
    action_states: list[np.ndarray] = []
    for ref in base.refs:
        episode = base.episodes[ref.episode_idx]
        if episode.actions_raw is None or episode.action_states_raw is None:
            raise ValueError("baseline analysis requires raw action and action-state arrays")
        action_start = int(ref.center) + int(base.config.action_offset)
        target = np.asarray(
            episode.actions_raw[
                action_start : action_start + int(base.config.policy_horizon)
            ],
            dtype=np.float32,
        )
        boundary = base._gripper_transition_boundary_raw(
            episode,
            state_index=int(ref.center) + int(base.config.state_offset),
            action_start=action_start,
        )
        if action_start == 0 and getattr(base.config, "causal_reset_padding", False):
            previous = np.asarray(episode.action_states_raw[0], dtype=np.float32)
        elif action_start <= 0:
            raise ValueError("baseline window has no preceding command")
        else:
            previous = np.asarray(episode.actions_raw[action_start - 1], dtype=np.float32)
        action_state = np.asarray(
            episode.action_states_raw[
                int(ref.center) + int(base.config.state_offset)
            ],
            dtype=np.float32,
        )
        targets.append(target)
        gripper_boundaries.append(boundary)
        previous_commands.append(previous)
        action_states.append(action_state)
    return (
        np.stack(targets),
        np.stack(gripper_boundaries),
        np.stack(previous_commands),
        np.stack(action_states),
    )


def _rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))


def _surfaces(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    result = {
        "action_rmse": _rmse(error),
        "first_rmse": _rmse(error[:, :1]),
        "first8_rmse": _rmse(error[:, :8]),
        "tail_rmse": _rmse(error[:, 8:]),
        "arm_rmse": _rmse(error[..., :-1]),
        "gripper_rmse": _rmse(error[..., -1:]),
    }
    result["tail_first_ratio"] = result["tail_rmse"] / max(result["first_rmse"], 1e-12)
    for name, band in _BANDS:
        result[f"band_{name}_rmse"] = _rmse(error[:, band])
        result[f"gripper_band_{name}_rmse"] = _rmse(error[:, band, -1:])
    return result


def _event_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    gripper_boundary: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    first = gripper_boundary[:, None, -1:]
    target_previous = np.concatenate((first, target[:, :-1, -1:]), axis=1)
    prediction_previous = np.concatenate((first, prediction[:, :-1, -1:]), axis=1)
    target_delta = target[..., -1:] - target_previous
    prediction_delta = prediction[..., -1:] - prediction_previous
    target_event = np.abs(target_delta) >= float(threshold)
    prediction_event = np.abs(prediction_delta) >= float(threshold)
    target_count = int(target_event.sum())
    prediction_count = int(prediction_event.sum())
    return {
        "events_predicted": prediction_count,
        "events_target": target_count,
        "event_ratio": float(prediction_count / max(target_count, 1)),
    }


def _evaluate(
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
    gripper_boundary: np.ndarray,
    *,
    normalizer: Any,
    event_threshold: float,
) -> dict[str, object]:
    prediction = np.broadcast_to(prediction_raw, target_raw.shape).copy()
    return {
        "source_native": _surfaces(prediction, target_raw),
        "normalized": _surfaces(
            normalizer.encode(prediction),
            normalizer.encode(target_raw),
        ),
        "decoded_gripper_events": _event_counts(
            prediction,
            target_raw,
            gripper_boundary,
            threshold=event_threshold,
        ),
    }


def _expected_zero_velocity_flow(
    target_raw: np.ndarray,
    action_state_raw: np.ndarray,
    gripper_boundary_raw: np.ndarray,
    *,
    normalizer: Any,
    config: Any,
) -> dict[str, float]:
    """Return the analytic expectation for a zero flow-velocity predictor.

    The source process is independent unit Gaussian noise.  Consequently each
    squared field residual contributes ``target_field**2 + 1`` in expectation;
    no Monte Carlo draw or flow-time sample is needed.
    """

    codec = PhysicalActionFieldCodec(
        action_dim=int(config.dimensions.action_dim),
        horizon=int(config.dimensions.action_horizon),
        gripper_field_dim=int(config.bottom.gripper_field_dim),
        decode_delta_blend=float(config.bottom.physical_decode_delta_blend),
    )
    target = torch.from_numpy(normalizer.encode(target_raw))
    action_state = torch.from_numpy(normalizer.encode(action_state_raw))
    gripper_boundary = torch.from_numpy(
        normalizer.encode(gripper_boundary_raw)[..., -1:]
    )
    field = codec.encode(
        target,
        action_state,
        codec_gripper_boundary=gripper_boundary,
    ).float()
    parts = codec.split(field)
    weights = anchor_horizon_weights(
        horizon=int(config.dimensions.action_horizon),
        tail_emphasis=float(config.objectives.horizon_tail_emphasis),
        first_step_protection=float(config.objectives.horizon_first_step_protection),
        device=torch.device("cpu"),
    )[None]
    arm_target_energy = 0.5 * (
        parts.arm_absolute.square() + parts.arm_delta.square()
    )
    gripper_target_energy = parts.gripper_field.square().mean(dim=-1)
    arm_expected = 1.0 + float(
        (arm_target_energy.mean(dim=-1) * weights).mean().item()
    )
    gripper_expected = 1.0 + float(
        (gripper_target_energy * weights).mean().item()
    )
    action_expected = (
        float(codec.arm_dim) * arm_expected + gripper_expected
    ) / float(codec.arm_dim + 1)
    return {
        "action_flow": action_expected,
        "arm_flow_per_dim": arm_expected,
        "gripper_field_flow": gripper_expected,
        # This is only the additive contribution for the zero predictor.  It
        # is not an irreducible optimum because the network observes noisy
        # state and flow time and can infer part of the source realization.
        "unit_gaussian_noise_contribution_to_zero_predictor": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute exact data-only action baselines for a mainline split."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_mainline_data(config)
    if args.split not in bundle.datasets:
        raise KeyError(f"unknown split {args.split!r}")
    train_target, train_gripper_boundary, _, train_action_state = _raw_windows(
        bundle.datasets["train"]
    )
    target, gripper_boundary, previous, action_state = _raw_windows(
        bundle.datasets[args.split]
    )
    horizon = int(target.shape[1])

    zero = np.zeros_like(target)
    hold = zero.copy()
    hold[..., -1] = gripper_boundary[:, None, -1]
    previous_command = np.repeat(previous[:, None], horizon, axis=1)
    normalizer_mean = np.broadcast_to(
        bundle.action_normalizer.mean[:, None, :],
        (1, horizon, int(target.shape[-1])),
    )
    train_pointwise_mean = train_target.mean(axis=0, keepdims=True, dtype=np.float64).astype(
        np.float32
    )
    train_pointwise_median = np.median(train_target, axis=0, keepdims=True).astype(np.float32)

    candidates = {
        "zero_command": zero,
        "physical_hold_zero_arm_previous_gripper": hold,
        "repeat_previous_full_command": previous_command,
        "normalizer_mean_command": normalizer_mean,
        "train_pointwise_mean": train_pointwise_mean,
        "train_pointwise_median": train_pointwise_median,
    }
    result = {
        "config": str(args.config.resolve()),
        "split": str(args.split),
        "windows": int(target.shape[0]),
        "horizon": horizon,
        "action_dim": int(target.shape[-1]),
        "event_threshold_source_native": float(config.objectives.gripper_event_threshold),
        "normalizer": {
            "mode": str(bundle.action_normalizer.mode),
            "mean": bundle.action_normalizer.mean.tolist(),
            "std": bundle.action_normalizer.std.tolist(),
        },
        "flow_matching_zero_velocity_expected": {
            "train": _expected_zero_velocity_flow(
                train_target,
                train_action_state,
                train_gripper_boundary,
                normalizer=bundle.action_normalizer,
                config=config,
            ),
            str(args.split): _expected_zero_velocity_flow(
                target,
                action_state,
                gripper_boundary,
                normalizer=bundle.action_normalizer,
                config=config,
            ),
        },
        "baselines": {
            name: _evaluate(
                value,
                target,
                gripper_boundary,
                normalizer=bundle.action_normalizer,
                event_threshold=float(config.objectives.gripper_event_threshold),
            )
            for name, value in candidates.items()
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
