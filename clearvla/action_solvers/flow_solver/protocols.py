"""Structural protocols for integrating an existing instantaneous field.

No protocol imports a ClearVLA implementation.  The mainline can provide a
small adapter around ``model.velocity`` when (and only when) an integration
owner explicitly selects this package.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypeAlias

import torch
from torch import Tensor

Cache: TypeAlias = object
EndpointValue: TypeAlias = Any
TimeFactory: TypeAlias = Callable[[float, Tensor], Tensor]


class VelocityField(Protocol):
    """Instantaneous physical velocity queried at one state and one time."""

    def __call__(self, state: Tensor, time: Tensor, cache: Cache, /) -> Tensor:
        ...


class EndpointHead(Protocol):
    """Non-updating readout evaluated on the final physical field at ``t=1``."""

    def __call__(self, state: Tensor, time: Tensor, cache: Cache, /) -> EndpointValue:
        ...


class CacheRebuilder(Protocol):
    """Build a fresh refined cache after the proposal pass."""

    def __call__(
        self,
        proposal_state: Tensor,
        proposal_endpoint: EndpointValue,
        proposal_cache: Cache,
        /,
    ) -> Cache:
        ...


def default_time_factory(value: float, state: Tensor) -> Tensor:
    """Create the current mainline-compatible scalar/batch time tensor.

    ClearVLA's velocity API expects one time value per batch item.  A scalar
    state is still supported for numerical unit tests and generic callers.
    The time tensor intentionally uses float32, independent of the physical
    state dtype, matching the existing runtime's time conditioning.
    """

    if state.ndim == 0:
        return torch.tensor(value, device=state.device, dtype=torch.float32)
    return torch.full(
        (state.shape[0],),
        value,
        device=state.device,
        dtype=torch.float32,
    )


__all__ = [
    "Cache",
    "CacheRebuilder",
    "EndpointHead",
    "EndpointValue",
    "TimeFactory",
    "VelocityField",
    "default_time_factory",
]
