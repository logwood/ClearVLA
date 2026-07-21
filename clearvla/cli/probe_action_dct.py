from __future__ import annotations

"""Probe native action trajectories in an exact DCT chart.

This is a dataset-only diagnostic.  It does not change the policy codec,
quantize coefficients, apply BPE, or truncate the training representation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from clearvla.data.hdf5_episode import find_hdf5_files
from clearvla.data.schema import ACTION_ALIASES, list_hdf5_datasets, resolve_key
from clearvla.policy.codec import ActionTemporalDCT


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _load_action(path: Path, requested_key: str) -> tuple[np.ndarray, str]:
    datasets = list_hdf5_datasets(str(path))
    resolved = resolve_key(datasets, requested_key, ACTION_ALIASES, required=True)
    assert resolved is not None
    with h5py.File(path, "r") as handle:
        action = np.asarray(handle[resolved], dtype=np.float64)
    if action.ndim != 2:
        raise ValueError(f"{path}: action must have [T,D], got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"{path}: action contains non-finite values")
    return action, resolved


def _parse_dims(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("arm dimensions must not be empty")
    return values


def _normalization(
    episodes: list[np.ndarray],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action_dim = int(episodes[0].shape[-1])
    if mode == "none":
        offset = np.zeros(action_dim, dtype=np.float64)
        scale = np.ones(action_dim, dtype=np.float64)
        return offset, scale, {"mode": mode}
    values = np.concatenate(episodes, axis=0)
    low, high = np.quantile(values, [0.01, 0.99], axis=0)
    scale = (high - low) * 0.5
    scale = np.where(scale > 1e-12, scale, 1.0)
    offset = (high + low) * 0.5
    return (
        offset,
        scale,
        {
            "mode": mode,
            "quantile_low": low.tolist(),
            "quantile_high": high.tolist(),
            "mapped_range": [-1.0, 1.0],
        },
    )


def _group_indices(
    action_dim: int,
    arm_dims: tuple[int, ...],
    gripper_index: int,
) -> dict[str, tuple[int, ...]]:
    gripper = int(gripper_index)
    if gripper < 0:
        gripper += action_dim
    if gripper < 0 or gripper >= action_dim:
        raise ValueError(f"gripper index {gripper_index} outside action_dim={action_dim}")
    arm = tuple(int(index) for index in arm_dims)
    if len(set(arm)) != len(arm) or any(index < 0 or index >= action_dim for index in arm):
        raise ValueError(f"invalid arm dimensions {arm} for action_dim={action_dim}")
    if gripper in arm:
        raise ValueError("gripper dimension must not also be an arm dimension")
    return {"arm": arm, "gripper": (gripper,)}


def probe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root)
    files = find_hdf5_files(root, args.glob)
    if int(args.max_episodes) > 0:
        files = files[: int(args.max_episodes)]
    if not files:
        raise RuntimeError(f"no HDF5 files found under {root}")

    episodes: list[np.ndarray] = []
    resolved_keys: set[str] = set()
    skipped: list[dict[str, str]] = []
    for path in files:
        try:
            action, resolved = _load_action(path, str(args.action_key))
            if int(action.shape[0]) < int(args.horizon):
                skipped.append({"path": str(path), "reason": "episode shorter than horizon"})
                continue
            episodes.append(action)
            resolved_keys.add(resolved)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": repr(exc)})
    if not episodes:
        raise RuntimeError(f"no usable episodes under {root}; skipped={skipped[:5]}")

    action_dim = int(episodes[0].shape[-1])
    if any(int(action.shape[-1]) != action_dim for action in episodes):
        raise ValueError("all episodes must have the same action dimension")
    groups = _group_indices(
        action_dim,
        _parse_dims(str(args.arm_dims)),
        int(args.gripper_index),
    )
    offset, scale, normalization = _normalization(episodes, str(args.normalization))
    chart = ActionTemporalDCT(
        int(args.horizon),
        arm_dims=groups["arm"],
        gripper_index=groups["gripper"][0],
    ).eval()
    keep_by_group = {
        "arm": tuple(sorted(set(int(value) for value in args.arm_keep))),
        "gripper": tuple(sorted(set(int(value) for value in args.gripper_keep))),
    }
    for group, keep_values in keep_by_group.items():
        if any(value < 1 or value > int(args.horizon) for value in keep_values):
            raise ValueError(f"all {group} keep values must be in [1, {args.horizon}]")

    error_sums = {group: {str(keep): 0.0 for keep in keep_values} for group in groups}
    error_counts = {group: 0 for group in groups}
    energy_sums = {group: np.zeros(int(args.horizon), dtype=np.float64) for group in groups}
    roundtrip_squared_sum = 0.0
    roundtrip_count = 0
    window_count = 0
    for episode in episodes:
        normalized = (episode - offset[None]) / scale[None]
        stride = max(int(args.stride), 1)
        for start in range(0, int(episode.shape[0]) - int(args.horizon) + 1, stride):
            if int(args.max_windows) > 0 and window_count >= int(args.max_windows):
                break
            window = torch.from_numpy(
                normalized[start : start + int(args.horizon)][None].astype(np.float32)
            )
            with torch.no_grad():
                coefficients = chart.encode(window)
                reconstruction = chart.decode(coefficients)
            original = window
            roundtrip_squared_sum += float((reconstruction - original).square().sum())
            roundtrip_count += int(original.numel())
            for group, indices in groups.items():
                group_coefficients = coefficients[..., list(indices)]
                energy_reduce_dims = tuple(range(group_coefficients.ndim - 2)) + (
                    group_coefficients.ndim - 1,
                )
                energy_sums[group] += (
                    group_coefficients.float().square().mean(dim=energy_reduce_dims).numpy()
                )
                group_original = original[..., list(indices)]
                error_counts[group] += int(group_original.numel())
                for keep in keep_by_group[group]:
                    truncated = chart.low_frequency(
                        coefficients,
                        arm_keep=(keep if group == "arm" else int(args.horizon)),
                        gripper_keep=(keep if group == "gripper" else int(args.horizon)),
                    )
                    with torch.no_grad():
                        reconstructed = chart.decode(truncated)
                    error = (reconstructed[..., list(indices)] - group_original).square()
                    error_sums[group][str(keep)] += float(error.sum())
            window_count += 1
        if int(args.max_windows) > 0 and window_count >= int(args.max_windows):
            break

    frequency_energy_fraction: dict[str, list[float]] = {}
    for group, energy in energy_sums.items():
        total = float(energy.sum())
        frequency_energy_fraction[group] = (energy / max(total, 1e-12)).tolist()

    return {
        "schema": "clearvla-action-dct-probe-v1",
        "config": {
            "data_root": str(root),
            "glob": str(args.glob),
            "action_key": str(args.action_key),
            "horizon": int(args.horizon),
            "stride": max(int(args.stride), 1),
            "arm_dims": list(groups["arm"]),
            "gripper_index": groups["gripper"][0],
            "arm_keep": list(keep_by_group["arm"]),
            "gripper_keep": list(keep_by_group["gripper"]),
            "normalization": str(args.normalization),
        },
        "files": {
            "matched": len(files),
            "loaded": len(episodes),
            "skipped": len(skipped),
            "resolved_action_keys": sorted(resolved_keys),
            "skipped_examples": skipped[:10],
        },
        "normalization": normalization,
        "chart": {
            "kind": "orthonormal_dct_ii",
            "inverse": "transpose",
            "quantization": False,
            "bpe": False,
            "truncated_for_training": False,
        },
        "windows": window_count,
        "roundtrip_rmse": float(np.sqrt(roundtrip_squared_sum / max(roundtrip_count, 1))),
        "reconstruction_rmse": {
            group: {
                f"K={keep}": float(np.sqrt(value / max(error_counts[group], 1)))
                for keep, value in errors.items()
            }
            for group, errors in error_sums.items()
        },
        "frequency_energy_fraction": frequency_energy_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe native action trajectories with an exact DCT chart."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--glob", default="*.hdf5")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--arm-dims", default="0,1,2,3,4,5")
    parser.add_argument("--gripper-index", type=int, default=-1)
    parser.add_argument("--arm-keep", type=int, nargs="+", default=[1, 2, 4, 8, 12, 24])
    parser.add_argument("--gripper-keep", type=int, nargs="+", default=[1, 2, 4, 8, 12, 24])
    parser.add_argument("--normalization", choices=("none", "quantile"), default="quantile")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    result = probe(args)
    text = json.dumps(
        _jsonable(result), ensure_ascii=False, indent=int(args.indent), sort_keys=True
    )
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
