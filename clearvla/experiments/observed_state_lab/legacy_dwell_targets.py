"""Retained offline target helper used by the archived hierarchical decoder tests.

This does not enter the mainline policy, online inputs, or its training engine.
Coordinates are post-update decisions; unavailable continuations remain masked.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def dwell_value_targets(
    *, prefix_error: Tensor, block_ids: Tensor, active: Tensor, compute_cost: float
) -> dict[str, Tensor]:
    if (
        prefix_error.ndim != 2
        or block_ids.shape != active.shape
        or prefix_error.shape != (active.shape[0], active.shape[1] + 1)
    ):
        raise ValueError("dwell targets require B,N+1 prefix errors and B,N decisions")
    if active.dtype != torch.bool or not math.isfinite(compute_cost) or compute_cost < 0:
        raise ValueError("dwell support and compute cost are invalid")
    if any(x.device != prefix_error.device for x in (block_ids, active)):
        raise ValueError("dwell targets must share a device")
    error = prefix_error.detach()[:, 1:]
    count = active.shape[1]
    continuation = error.clone()
    valid = torch.zeros_like(active)
    operation = torch.zeros_like(block_ids, dtype=torch.long)
    # Archive-only supervision: scan future *label* rows, never online state.
    for i in range(count - 1):
        valid[:, i] = active[:, i] & active[:, i + 1]
        operation[:, i] = torch.where(
            valid[:, i], (block_ids[:, i + 1] != block_ids[:, i]).long(), 0
        )
        offsets = torch.arange(1, count - i, device=error.device, dtype=error.dtype)
        future = error[:, i + 1 :] + compute_cost * offsets
        value = torch.where(active[:, i + 1 :], future, torch.inf).amin(1)
        continuation[:, i] = torch.where(valid[:, i], value, error[:, i])
    return dict(
        exit_target=error,
        continue_action=operation,
        continue_valid=valid,
        continue_target=continuation,
    )
