"""Conditional per-object S values; consumers, not values, own K probability.

This parameter-free production expression never constructs a task-only object.
It modulates observed typed values by the interval task query, retaining K
until the existing S/P2 consumer reads the one shared binding. A query may
change the value of a source, but cannot synthesize a nonzero absent source.

Historical selected values retain their separate replay mode. Selecting this
contract changes forward values and ordinary gradients and needs retraining;
it does not certify correct object identity or a learned task improvement.
"""

from __future__ import annotations

import torch
from torch import Tensor

LEGACY_SELECTED = "legacy_selected_v1"
CONDITIONAL_OBJECT = "conditional_object_v1"


def validate_typed_object_value_mode(mode: str) -> None:
    if mode not in {LEGACY_SELECTED, CONDITIONAL_OBJECT}:
        raise ValueError("unknown typed object value contract")


def conditional_typed_values(
    route: Tensor, query: Tensor, operation_gate: Tensor, supported: Tensor
) -> Tensor:
    """Return [B,I,K,type,R] without multiplying by any K/null probability.

    ``route`` is [B,K,type,R], ``query`` [B,I,type,R], ``operation_gate``
    [B,I,type], and ``supported`` is the producer-owned Boolean [B,K] mask.
    The bounded modulation lies in [0,2]; the operation gate is produced by
    the caller's sigmoid. No data-dependent normalization, learned mask,
    coordinate assumption, stochastic operation or surrogate VJP is used.
    """
    if route.ndim != 4 or query.ndim != 4:
        raise ValueError("conditional values require object/type and interval/type axes")
    if route.shape[0] != query.shape[0] or route.shape[2:] != query.shape[2:]:
        raise ValueError("conditional source and query types are misaligned")
    if supported.dtype != torch.bool or supported.shape != route.shape[:2]:
        raise ValueError("conditional values require explicit Boolean object support")
    if operation_gate.shape != query.shape[:-1]:
        raise ValueError("operation gate lost its interval/type axes")
    if any(t.device != route.device for t in (query, operation_gate, supported)):
        raise ValueError("conditional typed fields must share their device")
    if not all(t.is_floating_point() for t in (route, query, operation_gate)):
        raise TypeError("conditional typed values require floating-point fields")
    # Quarantine unavailable payload BEFORE multiplication, including backward.
    source = torch.where(supported[..., None, None], route, 0.0)
    return (
        source[:, None]
        * operation_gate[:, :, None, :, None].to(source)
        * (1.0 + torch.tanh(query[:, :, None]).to(source))
    )
