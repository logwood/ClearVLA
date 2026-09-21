#!/usr/bin/env python3
"""Separate Q5 sampling variance from matched language conditioning effects.

This is a read-only checkpoint probe.  It builds one fixed CALVIN observation
and zero-reset causal history, encodes a small set of instructions once, and
then runs the complete proposal -> W rebuild -> refined Q5 sampler with the
same bank of initial physical noise fields for every instruction.

The report distinguishes:

* deterministic replay error for an identical cache and identical noise;
* within-instruction action spread caused only by the initial flow noise;
* matched-seed action changes caused by changing the instruction;
* the ratio between those effects over early, middle, and late action bands.

No parameter, checkpoint, dataset, or simulator state is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Importing this module first installs its Python-3.9 in-memory compatibility
# loader before any ClearVLA model module is imported on the CALVIN host.
from scripts.probe_calvin_internal_layers import (  # noqa: E402
    ClearVLACheckpointPolicy,
    _as_policy_observation,
    _build_online,
    _environment,
    _fingerprint_observation,
    _layout_states,
    _state_for_initial_condition,
)
from clearvla.mainline.runtime.sampling import (  # noqa: E402
    deployment_cache,
    sample_refined_cached_action,
)
from clearvla.simulation.history import CausalHistory, HistorySnapshot  # noqa: E402


BANDS = {
    "rows_1_4": slice(0, 4),
    "rows_5_12": slice(4, 12),
    "rows_13_24": slice(12, 24),
}


def _rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _pairwise_arm_rmse(actions: np.ndarray) -> dict[str, float]:
    values = [
        _rmse(actions[left, :, :6] - actions[right, :, :6])
        for left, right in combinations(range(actions.shape[0]), 2)
    ]
    return _quantiles(values)


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    arm = actions[:, :, :6].astype(np.float64)
    centered = arm - np.mean(arm, axis=0, keepdims=True)
    first = arm[:, 0]
    signs = np.sign(first)
    majority = np.sign(np.mean(first, axis=0))
    active = np.abs(first) >= 0.05
    disagreement = np.logical_and(active, signs != majority[None])
    active_count = np.maximum(np.sum(active, axis=0), 1)
    return {
        "samples": int(actions.shape[0]),
        "arm_action_rms": _rmse(arm),
        "arm_centered_spread_rms": _rmse(centered),
        "arm_pairwise_rmse": _pairwise_arm_rmse(actions),
        "first_row_arm_mean": np.mean(first, axis=0).tolist(),
        "first_row_arm_std": np.std(first, axis=0).tolist(),
        "first_row_active_sign_disagreement_fraction_per_dim": (
            np.sum(disagreement, axis=0) / active_count
        ).tolist(),
        "action_mean": np.mean(actions, axis=0).tolist(),
        "action_std": np.std(actions, axis=0).tolist(),
    }


def _matched_language_summary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_noise_spread: float,
    right_noise_spread: float,
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"matched action banks differ: {left.shape} versus {right.shape}")
    delta = left[:, :, :6].astype(np.float64) - right[:, :, :6].astype(np.float64)
    noise_reference = 0.5 * (left_noise_spread + right_noise_spread)
    bands: dict[str, Any] = {}
    for name, rows in BANDS.items():
        per_seed = np.sqrt(np.mean(np.square(delta[:, rows]), axis=(1, 2)))
        effect = float(np.mean(per_seed))
        bands[name] = {
            "matched_seed_arm_rmse": _quantiles(per_seed.tolist()),
            "mean_language_effect_to_noise_spread_ratio": float(
                effect / max(noise_reference, 1e-12)
            ),
        }
    mean_delta = np.mean(left[:, :, :6], axis=0) - np.mean(right[:, :, :6], axis=0)
    return {
        "matched_seed_full_arm_rmse": _quantiles(
            np.sqrt(np.mean(np.square(delta), axis=(1, 2))).tolist()
        ),
        "seed_averaged_mean_action_arm_rmse": _rmse(mean_delta),
        "noise_spread_reference_rms": float(noise_reference),
        "full_language_effect_to_noise_spread_ratio": float(
            np.mean(np.sqrt(np.mean(np.square(delta), axis=(1, 2))))
            / max(noise_reference, 1e-12)
        ),
        "first_row_mean_arm_delta": (
            np.mean(left[:, 0, :6], axis=0) - np.mean(right[:, 0, :6], axis=0)
        ).tolist(),
        "bands": bands,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _history_from_snapshot(path: Path) -> HistorySnapshot:
    with np.load(path, allow_pickle=False) as data:
        version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
        if version != 1:
            raise ValueError(f"unsupported causal snapshot version {version}")
        history = HistorySnapshot(
            time_index=int(np.asarray(data["time_index"]).reshape(-1)[0]),
            rgb_history={
                "top": np.asarray(data["rgb_top_history"], dtype=np.uint8),
                "wrist": np.asarray(data["rgb_wrist_history"], dtype=np.uint8),
            },
            state=np.asarray(data["state"], dtype=np.float32),
            action_state=np.asarray(data["action_state"], dtype=np.float32),
            state_history=np.asarray(data["state_history"], dtype=np.float32),
            executed_action_history=np.asarray(
                data["executed_action_history"], dtype=np.float32
            ),
        )
    history.validate()
    return history


def run_probe(
    *,
    checkpoint: Path,
    t5_condition: Path | None,
    dataset_root: Path | None,
    snapshot: Path | None,
    output: Path,
    layout: str,
    instructions: tuple[str, ...],
    seeds: tuple[int, ...],
    device: str,
    dinov2_model: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    if len(instructions) < 2:
        raise ValueError("at least two instructions are required")
    if len(seeds) < 2:
        raise ValueError("at least two noise seeds are required")
    policy = ClearVLACheckpointPolicy(
        checkpoint,
        device=torch.device(device),
        t5_condition=t5_condition,
        dinov2_model=dinov2_model,
        dinov2_local_files_only=local_files_only,
        seed=0,
    )
    env = None
    if snapshot is not None:
        history = _history_from_snapshot(snapshot)
        observation_record: dict[str, Any] = {
            "source": "causal_snapshot",
            "path": str(snapshot),
            "sha256": _sha256(snapshot),
            "time_index": int(history.time_index),
        }
        initial_state: Any = None
    else:
        if dataset_root is None:
            raise ValueError("either --snapshot or --dataset-root is required")
        layouts = _layout_states()
        if layout not in layouts:
            raise KeyError(f"unknown layout {layout!r}")
        env = _environment(dataset_root, show_gui=False)
        robot_obs, scene_obs = _state_for_initial_condition(layouts[layout])
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        observation = env.get_obs()
        policy_observation = _as_policy_observation(
            observation, np.zeros(7, dtype=np.float32)
        )
        timeline = CausalHistory()
        timeline.reset(policy_observation, reset_action=policy_observation.action_state)
        history = timeline.snapshot()
        observation_record = _fingerprint_observation(observation)
        initial_state = layouts[layout]
    try:
        config = policy.bundle.config
        caches: dict[str, Any] = {}
        static_metrics: dict[str, dict[str, float]] = {}
        for instruction in instructions:
            online = _build_online(policy, history, instruction)
            cache, metrics = deployment_cache(
                policy.bundle.model,
                online,
                config,
                collect_diagnostics=True,
            )
            caches[instruction] = cache
            static_metrics[instruction] = {
                key: float(value.detach().float().cpu())
                for key, value in metrics.items()
                if torch.is_tensor(value)
                and value.numel() == 1
                and (
                    key.startswith("object_intent_")
                    or key.startswith("object_p2_")
                    or key.startswith("object_p3_")
                )
            }

        noise_bank: list[torch.Tensor] = []
        for seed in seeds:
            generator = torch.Generator(device=policy.device).manual_seed(int(seed))
            noise_bank.append(
                torch.randn(
                    1,
                    config.dimensions.action_horizon,
                    policy.bundle.model.outlet_adapter.physical_dim,
                    device=policy.device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )

        action_banks: dict[str, np.ndarray] = {}
        schedule_identity: dict[str, Any] | None = None
        for instruction in instructions:
            samples: list[np.ndarray] = []
            for noise in noise_bank:
                sampled = sample_refined_cached_action(
                    policy.bundle.model,
                    caches[instruction],
                    config,
                    initial_physical_noise=noise,
                    collect_diagnostics=False,
                )
                if schedule_identity is None:
                    schedule_identity = sampled.flow_schedule_identity
                samples.append(sampled.action.detach().float().cpu().numpy()[0])
            action_banks[instruction] = np.stack(samples, axis=0)

        baseline_instruction = instructions[0]
        replay_noise = noise_bank[0]
        first = sample_refined_cached_action(
            policy.bundle.model,
            caches[baseline_instruction],
            config,
            initial_physical_noise=replay_noise,
        ).action
        second = sample_refined_cached_action(
            policy.bundle.model,
            caches[baseline_instruction],
            config,
            initial_physical_noise=replay_noise,
        ).action
        replay_max_abs = float(
            (first.detach().float() - second.detach().float()).abs().max().cpu()
        )

        instruction_reports = {
            instruction: {
                **_action_summary(action_banks[instruction]),
                "selected_static_metrics": static_metrics[instruction],
            }
            for instruction in instructions
        }
        pair_reports: dict[str, Any] = {}
        for left_index, left_instruction in enumerate(instructions):
            for right_instruction in instructions[left_index + 1 :]:
                left_report = instruction_reports[left_instruction]
                right_report = instruction_reports[right_instruction]
                key = f"{left_instruction} :: {right_instruction}"
                pair_reports[key] = _matched_language_summary(
                    action_banks[left_instruction],
                    action_banks[right_instruction],
                    left_noise_spread=float(left_report["arm_centered_spread_rms"]),
                    right_noise_spread=float(right_report["arm_centered_spread_rms"]),
                )

        payload = {
            "schema": "clearvla-calvin-q5-noise-language-v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "device": str(policy.device),
            "layout": layout,
            "initial_state": initial_state,
            "observation": observation_record,
            "history": {
                "state_rows": int(history.state_history.shape[0]),
                "action_rows": int(history.executed_action_history.shape[0]),
                "reset_zero_history": bool(history.time_index == 0),
            },
            "noise_seeds": list(seeds),
            "deterministic_replay_max_abs": replay_max_abs,
            "flow_schedule_identity": schedule_identity,
            "instructions": instruction_reports,
            "matched_language_pairs": pair_reports,
            "interpretation": {
                "noise_spread": "same observation, history, language and weights; only initial physical flow noise changes",
                "language_effect": "same observation, history, weights and initial noise; only language changes",
                "ratio": "values below one mean the measured language change is smaller than within-language sampling spread",
            },
        }
        _atomic_json(output, payload)
        return payload
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--t5-condition", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", default="standard")
    parser.add_argument(
        "--instructions",
        nargs="+",
        default=["go push the blue block left", "go push the blue block right"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dinov2-model", type=Path, default=None)
    parser.add_argument("--dinov2-local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_probe(
        checkpoint=args.checkpoint,
        t5_condition=args.t5_condition,
        dataset_root=args.dataset_root,
        snapshot=args.snapshot,
        output=args.output,
        layout=str(args.layout),
        instructions=tuple(str(value) for value in args.instructions),
        seeds=tuple(int(value) for value in args.seeds),
        device=str(args.device),
        dinov2_model=args.dinov2_model,
        local_files_only=bool(args.dinov2_local_files_only),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "deterministic_replay_max_abs": result[
                    "deterministic_replay_max_abs"
                ],
                "pairs": list(result["matched_language_pairs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
