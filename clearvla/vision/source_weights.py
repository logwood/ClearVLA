"""Conditional weights with an explicit source support and log-mass authority.

Allocation is not physical visibility. An absent/zero allocation cannot invent
an observation, and a finite log mass must not become missing through exp
underflow. This changes neither the K owner nor any source-support mask.
"""

from __future__ import annotations

import torch
from torch import Tensor


def conditional_source_weights(
    mass: Tensor | None,
    supported: Tensor,
    *,
    dim: int = -1,
    log_mass: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return probability, finite masked log probability, and nonempty groups.

    The optional log measure is authoritative; linear mass may also be carried
    as telemetry. An unsupported payload is quarantined before log/exp. -inf
    denotes an actual zero measure; finite very negative values remain valid.
    """
    value = log_mass if log_mass is not None else mass
    if value is None or value.shape != supported.shape or supported.dtype != torch.bool:
        raise ValueError("source allocation requires aligned values and Boolean support")
    if value.device != supported.device or not value.is_floating_point():
        raise ValueError("source allocation and support must share a floating-point device")
    if not -value.ndim <= dim < value.ndim or value.shape[dim] == 0:
        raise ValueError("source allocation needs a nonempty declared reduction axis")
    with torch.autocast(device_type=value.device.type, enabled=False):
        if log_mass is None:
            safe = torch.where(supported, value.float(), 0.0)
            if not bool(torch.isfinite(safe).all()) or bool((safe < 0).any()):
                raise ValueError("source mass must be finite and nonnegative on its support")
            positive = supported & (safe > 0)
            logs = torch.where(positive, torch.where(positive, safe, 1.0).log(), -torch.inf)
        else:
            logs = torch.where(supported, value.float(), -torch.inf)
            if bool((torch.isnan(logs) | torch.isposinf(logs)).any()):
                raise ValueError("supported source log mass cannot be NaN or +inf")
            positive = supported & torch.isfinite(logs)
        nonempty = positive.any(dim, keepdim=True)
        # Avoid an all--inf logsumexp backward; no epsilon changes the measure.
        safe_logs = torch.where(nonempty, logs, 0.0)
        # log_softmax subtracts the shared maximum before summation; the
        # direct logs-logsumexp form loses precision near e.g. -1000.
        log_p = torch.where(positive, torch.log_softmax(safe_logs, dim), 0.0)
        p = torch.where(positive, log_p.exp(), 0.0)
        return p, log_p, nonempty
