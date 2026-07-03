from __future__ import annotations

"""Shared dense-token encoding and physical gripper metrics for dense-token latent-world models."""

from typing import Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from clearvla.experiments.classic_policy_lab.rdt2_conditioning import RDT2Conditioner


class DenseWorldConfig(Protocol):
    history_length: int
    num_future: int
    num_cameras: int
    patches_per_camera: int
    latent_dim: int


def _reshape_dense_tokens(
    dense: Tensor, *, batch: int, history: int, config: DenseWorldConfig
) -> Tensor:
    expected_tokens = config.num_cameras * config.patches_per_camera
    if dense.ndim != 3 or dense.shape[0] != batch * history:
        raise ValueError(f"conditioner dense tokens have invalid shape {tuple(dense.shape)}")
    if dense.shape[1] != expected_tokens or dense.shape[2] != config.latent_dim:
        raise ValueError(
            "DINO token geometry mismatch: "
            f"got {tuple(dense.shape[1:])}, expected ({expected_tokens},{config.latent_dim})"
        )
    return dense.reshape(
        batch, history, config.num_cameras, config.patches_per_camera, config.latent_dim
    )


@torch.no_grad()
def encode_current_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DenseWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    batch = int(sample["state"].shape[0])
    history = model_config.history_length
    if "history_obs_image" in sample:
        images = sample["history_obs_image"]
        flat_images = images.reshape(batch * history, *images.shape[2:])
        condition = conditioner.encode(flat_images, camera_names=camera_names)
    else:
        keys = sample["history_keys"].reshape(batch * history, 2)
        dummy = torch.zeros(
            batch * history, model_config.num_cameras, 3, 1, 1, dtype=torch.float32
        )
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("latent world requires dense DINO tokens")
    current = _reshape_dense_tokens(
        condition.dense_tokens, batch=batch, history=history, config=model_config
    )
    return current.to(device=device, dtype=dtype)


@torch.no_grad()
def encode_target_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DenseWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    batch = int(sample["state"].shape[0])
    history = model_config.history_length
    future = model_config.num_future
    if "target_history_obs_image" in sample:
        images = sample["target_history_obs_image"]
        flat = images.reshape(batch * future * history, *images.shape[3:])
        condition = conditioner.encode(flat, camera_names=camera_names)
    else:
        keys = sample["target_history_keys"].reshape(batch * future * history, 2)
        dummy = torch.zeros(
            batch * future * history, model_config.num_cameras, 3, 1, 1, dtype=torch.float32
        )
        condition = conditioner.encode(dummy, sample_keys=keys, camera_names=camera_names)
    if condition.dense_tokens is None:
        raise ValueError("latent world requires dense DINO tokens")
    return _reshape_dense_tokens(
        condition.dense_tokens,
        batch=batch * future,
        history=history,
        config=model_config,
    ).reshape(
        batch,
        future,
        history,
        model_config.num_cameras,
        model_config.patches_per_camera,
        model_config.latent_dim,
    ).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_sample_tokens(
    sample: dict[str, Tensor],
    *,
    conditioner: RDT2Conditioner,
    model_config: DenseWorldConfig,
    camera_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    return (
        encode_current_tokens(
            sample,
            conditioner=conditioner,
            model_config=model_config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        ),
        encode_target_tokens(
            sample,
            conditioner=conditioner,
            model_config=model_config,
            camera_names=camera_names,
            device=device,
            dtype=dtype,
        ),
    )


def gripper_transition_metrics(
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    current_raw: np.ndarray,
    *,
    gripper_index: int,
    threshold: float,
    tolerance: int,
) -> dict[str, float]:
    pred_g = np.asarray(pred_raw[..., gripper_index], dtype=np.float64)
    target_g = np.asarray(target_raw[..., gripper_index], dtype=np.float64)
    current_g = np.asarray(current_raw[..., gripper_index], dtype=np.float64).reshape(-1, 1)
    pred_boundary = np.concatenate([current_g, pred_g[:, :-1]], axis=1)
    target_boundary = np.concatenate([current_g, target_g[:, :-1]], axis=1)
    pred_delta = pred_g - pred_boundary
    target_delta = target_g - target_boundary

    def match_direction(direction: int) -> tuple[int, int, int, list[float]]:
        tp = fp = fn = 0
        timing: list[float] = []
        for row in range(len(pred_delta)):
            pred_idx = np.flatnonzero(
                (np.abs(pred_delta[row]) >= threshold) & (np.sign(pred_delta[row]) == direction)
            ).tolist()
            target_idx = np.flatnonzero(
                (np.abs(target_delta[row]) >= threshold) & (np.sign(target_delta[row]) == direction)
            ).tolist()
            dp: list[list[tuple[int, float, tuple[float, ...]]]] = [
                [(0, 0.0, ()) for _ in range(len(target_idx) + 1)]
                for _ in range(len(pred_idx) + 1)
            ]

            def better(
                left: tuple[int, float, tuple[float, ...]],
                right: tuple[int, float, tuple[float, ...]],
            ) -> tuple[int, float, tuple[float, ...]]:
                if left[0] != right[0]:
                    return left if left[0] > right[0] else right
                return left if left[1] <= right[1] else right

            for pred_pos in range(1, len(pred_idx) + 1):
                for target_pos in range(1, len(target_idx) + 1):
                    best = better(dp[pred_pos - 1][target_pos], dp[pred_pos][target_pos - 1])
                    distance = abs(pred_idx[pred_pos - 1] - target_idx[target_pos - 1])
                    if distance <= tolerance:
                        previous = dp[pred_pos - 1][target_pos - 1]
                        best = better(
                            best,
                            (
                                previous[0] + 1,
                                previous[1] + float(distance),
                                previous[2] + (float(distance),),
                            ),
                        )
                    dp[pred_pos][target_pos] = best
            matched, _, matched_timing = dp[-1][-1]
            timing.extend(matched_timing)
            tp += matched
            fp += len(pred_idx) - matched
            fn += len(target_idx) - matched
        return tp, fp, fn, timing

    def summarize(tp: int, fp: int, fn: int, prefix: str) -> dict[str, float]:
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
        return {
            f"{prefix}precision": precision,
            f"{prefix}recall": recall,
            f"{prefix}f1": f1,
            f"{prefix}tp": float(tp),
            f"{prefix}fp": float(fp),
            f"{prefix}fn": float(fn),
        }

    close_tp, close_fp, close_fn, close_timing = match_direction(+1)
    open_tp, open_fp, open_fn, open_timing = match_direction(-1)
    tp, fp, fn = close_tp + open_tp, close_fp + open_fp, close_fn + open_fn
    timing = close_timing + open_timing
    metrics = summarize(tp, fp, fn, "gripper_")
    metrics.update(summarize(close_tp, close_fp, close_fn, "gripper_close_"))
    metrics.update(summarize(open_tp, open_fp, open_fn, "gripper_open_"))
    metrics.update(
        {
            "gripper_pred_events": float(tp + fp),
            "gripper_target_events": float(tp + fn),
            "gripper_timing_mae_steps": float(np.mean(timing)) if timing else float("nan"),
            "gripper_close_timing_mae_steps": float(np.mean(close_timing)) if close_timing else float("nan"),
            "gripper_open_timing_mae_steps": float(np.mean(open_timing)) if open_timing else float("nan"),
        }
    )
    return metrics


__all__ = [
    "encode_current_tokens",
    "encode_target_tokens",
    "encode_sample_tokens",
    "gripper_transition_metrics",
]
