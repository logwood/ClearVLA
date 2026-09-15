from __future__ import annotations

import numpy as np


def compute_velocity_np(past: np.ndarray, mode: str = "ema", ema_decay: float = 0.75) -> np.ndarray:
    """Compute velocity from past action sequence.

    Args:
        past: [..., K, D]
    Returns:
        [..., D]
    """
    if past.shape[-2] < 2:
        return np.zeros_like(past[..., -1, :])
    deltas = past[..., 1:, :] - past[..., :-1, :]
    if mode == "last":
        return deltas[..., -1, :]
    if mode == "mean":
        return deltas.mean(axis=-2)
    if mode == "ema":
        n = deltas.shape[-2]
        weights = np.asarray([ema_decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        weights /= weights.sum()
        return np.sum(deltas * weights.reshape(*([1] * (deltas.ndim - 2)), n, 1), axis=-2)
    raise ValueError(f"Unknown velocity mode={mode!r}")


def make_prior_np(
    past: np.ndarray,
    horizon: int,
    *,
    prior: str = "blend",
    prior_beta: float = 0.5,
    velocity_mode: str = "ema",
    ema_decay: float = 0.75,
) -> np.ndarray:
    """Construct future hypothesis from past actions.

    Args:
        past: [..., K, D]
    Returns:
        [..., horizon, D]
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    anchor = past[..., -1:, :]
    action_dim = past.shape[-1]
    steps = np.arange(1, horizon + 1, dtype=np.float32).reshape(
        *([1] * (past.ndim - 2)), horizon, 1
    )

    if prior == "hold":
        return np.broadcast_to(anchor, past.shape[:-2] + (horizon, action_dim)).copy()

    velocity = compute_velocity_np(past, mode=velocity_mode, ema_decay=ema_decay)
    beta = 1.0 if prior in {"velocity", "ema_velocity"} else float(prior_beta)
    if prior not in {"velocity", "ema_velocity", "blend"}:
        raise ValueError(f"Unknown prior={prior!r}")
    return anchor + beta * steps * velocity[..., None, :]
