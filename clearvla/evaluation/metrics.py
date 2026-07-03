from __future__ import annotations

from typing import Any

import numpy as np

from clearvla.data.normalizer import ZScoreNormalizer


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _history_replay_for_target(past: np.ndarray, target_horizon: int) -> np.ndarray:
    """Align a recorded history chunk to a future-evaluation horizon.

    A history-replay baseline predicts future step ``k`` with recorded history
    step ``k``. Full chunk evaluations normally provide equal horizons. Prefix
    evaluations can provide a shorter target, so retain the matching history
    prefix. If a caller supplies a shorter history, extend it with its last
    recorded action rather than silently changing the metric shape.
    """
    if past.ndim != 3:
        raise ValueError(f"past must be [B,H,A], got {past.shape}")
    if target_horizon <= 0:
        raise ValueError(f"target_horizon must be positive, got {target_horizon}")
    if past.shape[1] == 0:
        raise ValueError("past history must contain at least one action")
    if past.shape[1] >= target_horizon:
        return past[:, :target_horizon]
    padding = np.repeat(past[:, -1:], target_horizon - past.shape[1], axis=1)
    return np.concatenate([past, padding], axis=1)


def compute_metrics(
    *,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    prior_norm: np.ndarray,
    past_norm: np.ndarray,
    normalizer: ZScoreNormalizer,
) -> dict[str, Any]:
    pred = normalizer.decode(pred_norm)
    target = normalizer.decode(target_norm)
    prior = normalizer.decode(prior_norm)
    past = normalizer.decode(past_norm)
    replay = _history_replay_for_target(past, target.shape[1])

    pred_jump = pred[:, 0] - past[:, -1]
    target_jump = target[:, 0] - past[:, -1]
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    pred_velocity_with_boundary = np.concatenate([pred[:, :1] - past[:, -1:], pred_velocity], axis=1)
    target_velocity_with_boundary = np.concatenate([target[:, :1] - past[:, -1:], target_velocity], axis=1)
    error = pred - target
    prior_error = prior - target
    replay_error = replay - target
    horizon_mse = np.mean(error ** 2, axis=(0, 2))
    horizon_rmse = np.sqrt(horizon_mse)
    horizon_mae = np.mean(np.abs(error), axis=(0, 2))
    first4 = min(4, pred.shape[1])
    first8 = min(8, pred.shape[1])
    per_dim_rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1)))
    per_dim_mae = np.mean(np.abs(error), axis=(0, 1))
    action_std = np.asarray(normalizer.std, dtype=np.float32).reshape(-1)
    if action_std.shape != per_dim_rmse.shape:
        raise ValueError(f"normalizer std shape={action_std.shape} is incompatible with action error shape={per_dim_rmse.shape}")
    per_dim_nrmse = per_dim_rmse / np.maximum(action_std, 1e-8)

    # This project conventionally stores the gripper in the last action
    # dimension (LabEventScoreConfig.gripper_index defaults to -1).  Keep the
    # raw-unit metrics explicit and add degree conversions only as an optional
    # interpretation for datasets whose arm dimensions are radians.
    arm = error[..., :-1] if error.shape[-1] > 1 else None
    gripper = error[..., -1]

    out: dict[str, Any] = {
        "n": int(len(pred)),
        "full_mse": mse_np(pred, target),
        "full_rmse": rmse_np(pred, target),
        "full_mae": mae_np(pred, target),
        "normalized_mae": mae_np(pred_norm, target_norm),
        "first_mse": mse_np(pred[:, 0], target[:, 0]),
        "first_rmse": rmse_np(pred[:, 0], target[:, 0]),
        "first_mae": mae_np(pred[:, 0], target[:, 0]),
        "first4_mse": mse_np(pred[:, :first4], target[:, :first4]),
        "first4_rmse": rmse_np(pred[:, :first4], target[:, :first4]),
        "first4_mae": mae_np(pred[:, :first4], target[:, :first4]),
        "first8_mse": mse_np(pred[:, :first8], target[:, :first8]),
        "first8_rmse": rmse_np(pred[:, :first8], target[:, :first8]),
        "first8_mae": mae_np(pred[:, :first8], target[:, :first8]),
        "delta_mse": mse_np(pred_velocity, target_velocity),
        "delta_rmse": rmse_np(pred_velocity, target_velocity),
        "delta_mae": mae_np(pred_velocity, target_velocity),
        "delta_boundary_mse": mse_np(pred_velocity_with_boundary, target_velocity_with_boundary),
        "delta_boundary_rmse": rmse_np(pred_velocity_with_boundary, target_velocity_with_boundary),
        "delta_boundary_mae": mae_np(pred_velocity_with_boundary, target_velocity_with_boundary),
        "chunk_endpoint_mse": mse_np(pred[:, -1], target[:, -1]),
        "chunk_endpoint_rmse": rmse_np(pred[:, -1], target[:, -1]),
        "chunk_endpoint_mae": mae_np(pred[:, -1], target[:, -1]),
        "boundary_jump_mse": mse_np(pred_jump, target_jump),
        "boundary_jump_rmse": rmse_np(pred_jump, target_jump),
        "boundary_jump_mae": mae_np(pred_jump, target_jump),
        "pred_boundary_jump_norm": float(np.mean(np.linalg.norm(pred_jump, axis=-1))),
        "target_boundary_jump_norm": float(np.mean(np.linalg.norm(target_jump, axis=-1))),
        # Backward-compatible aliases: historical ``prior`` means hold-last.
        "prior_full_mse": mse_np(prior, target),
        "prior_full_rmse": rmse_np(prior, target),
        "prior_full_mae": mae_np(prior, target),
        "prior_normalized_mae": mae_np(prior_norm, target_norm),
        "prior_first_mse": mse_np(prior[:, 0], target[:, 0]),
        "prior_first_rmse": rmse_np(prior[:, 0], target[:, 0]),
        "prior_first_mae": mae_np(prior[:, 0], target[:, 0]),
        "hold_last_full_mse": mse_np(prior, target),
        "hold_last_full_rmse": rmse_np(prior, target),
        "hold_last_full_mae": mae_np(prior, target),
        "history_replay_full_mse": mse_np(replay, target),
        "history_replay_full_rmse": rmse_np(replay, target),
        "history_replay_full_mae": mae_np(replay, target),
        "history_replay_first_mse": mse_np(replay[:, 0], target[:, 0]),
        "history_replay_first_rmse": rmse_np(replay[:, 0], target[:, 0]),
        "per_dim_rmse": per_dim_rmse.tolist(),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_dim_nrmse": per_dim_nrmse.tolist(),
        "per_horizon_rmse": horizon_rmse.tolist(),
        "per_horizon_mae": horizon_mae.tolist(),
        "gripper_dim_index": int(error.shape[-1] - 1),
        "gripper_full_rmse": rmse_np(gripper, np.zeros_like(gripper)),
        "gripper_first_rmse": rmse_np(gripper[:, 0], np.zeros_like(gripper[:, 0])),
        "gripper_first4_rmse": rmse_np(gripper[:, :first4], np.zeros_like(gripper[:, :first4])),
        "gripper_first8_rmse": rmse_np(gripper[:, :first8], np.zeros_like(gripper[:, :first8])),
        "gripper_endpoint_rmse": rmse_np(gripper[:, -1], np.zeros_like(gripper[:, -1])),
        "gripper_delta_rmse": rmse_np(
            pred_velocity_with_boundary[..., -1],
            target_velocity_with_boundary[..., -1],
        ),
        "gripper_full_mae": mae_np(gripper, np.zeros_like(gripper)),
    }
    if arm is not None:
        arm_full_rmse = rmse_np(arm, np.zeros_like(arm))
        arm_first_rmse = rmse_np(arm[:, 0], np.zeros_like(arm[:, 0]))
        arm_first4_rmse = rmse_np(arm[:, :first4], np.zeros_like(arm[:, :first4]))
        arm_first8_rmse = rmse_np(arm[:, :first8], np.zeros_like(arm[:, :first8]))
        arm_delta = pred_velocity_with_boundary[..., :-1] - target_velocity_with_boundary[..., :-1]
        out.update({
            "arm_full_rmse": arm_full_rmse,
            "arm_first_rmse": arm_first_rmse,
            "arm_first4_rmse": arm_first4_rmse,
            "arm_first8_rmse": arm_first8_rmse,
            "arm_endpoint_rmse": rmse_np(arm[:, -1], np.zeros_like(arm[:, -1])),
            "arm_delta_rmse": rmse_np(arm_delta, np.zeros_like(arm_delta)),
            "arm_full_mae": mae_np(arm, np.zeros_like(arm)),
            "arm_full_rmse_deg_if_rad": float(np.degrees(arm_full_rmse)),
            "arm_first_rmse_deg_if_rad": float(np.degrees(arm_first_rmse)),
            "arm_first4_rmse_deg_if_rad": float(np.degrees(arm_first4_rmse)),
            "arm_first8_rmse_deg_if_rad": float(np.degrees(arm_first8_rmse)),
        })
    if out["prior_full_mse"] > 0:
        out["relative_mse_improvement_vs_prior"] = float(1.0 - out["full_mse"] / out["prior_full_mse"])
        out["relative_mse_improvement_vs_hold_last"] = out["relative_mse_improvement_vs_prior"]
    if out["history_replay_full_mse"] > 0:
        out["relative_mse_improvement_vs_history_replay"] = float(1.0 - out["full_mse"] / out["history_replay_full_mse"])
    for step in (1, 2, 4, 8, 12, 16, 20, 24, 25):
        if step <= len(horizon_mse):
            out[f"step_{step}_mse"] = float(horizon_mse[step - 1])
            out[f"step_{step}_rmse"] = float(horizon_rmse[step - 1])
            out[f"step_{step}_mae"] = float(horizon_mae[step - 1])
    return out


def add_stage_metrics(
    rows: dict[str, Any],
    *,
    stage_predictions: list[np.ndarray],
    target_norm: np.ndarray,
    prior_norm: np.ndarray,
    past_norm: np.ndarray,
    normalizer: ZScoreNormalizer,
) -> dict[str, Any]:
    out = dict(rows)
    for idx, pred in enumerate(stage_predictions):
        metrics = compute_metrics(
            pred_norm=pred,
            target_norm=target_norm,
            prior_norm=prior_norm,
            past_norm=past_norm,
            normalizer=normalizer,
        )
        for key, value in metrics.items():
            if key == "n":
                continue
            out[f"refine_{idx}_{key}"] = value
    return out
