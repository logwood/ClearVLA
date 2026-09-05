"""Duck-typed adapters for a future mainline integration.

The adapters deliberately avoid importing ``clearvla.mainline``.  They make
the replacement seam small while leaving output extraction, execution mode and
diagnostics explicit at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import Tensor

from .protocols import Cache, EndpointValue

OutputExtractor = Callable[[Any], Tensor]
EndpointExtractor = Callable[[Any], EndpointValue]


@dataclass(frozen=True)
class ExistingModelVelocityAdapter:
    """Adapt a model exposing the current ``velocity`` keyword ABI."""

    model: object
    velocity_extractor: OutputExtractor
    execution_mode: str = "learned"
    collect_diagnostics: bool = False

    def __post_init__(self) -> None:
        if not callable(getattr(self.model, "velocity", None)):
            raise TypeError("model must expose a callable velocity method")
        if not callable(self.velocity_extractor):
            raise TypeError("velocity_extractor must be callable")
        if self.collect_diagnostics:
            raise ValueError(
                "collect_diagnostics=True is unsafe for an unindexed adapter; "
                "use an explicit call-index-aware wrapper and leave this flag false"
            )

    def __call__(self, state: Tensor, time: Tensor, cache: Cache) -> Tensor:
        velocity_method = getattr(self.model, "velocity", None)
        if not callable(velocity_method):
            raise TypeError("model must expose a callable velocity method")
        output = velocity_method(
            cache,
            noisy_action_field=state,
            time=time,
            execution_mode=self.execution_mode,
            collect_diagnostics=self.collect_diagnostics,
        )
        return self.velocity_extractor(output)

    def integration_metadata(self) -> dict[str, object]:
        return {
            "adapter": "existing_model_velocity",
            "velocity_abi": "velocity(cache,noisy_action_field,time,execution_mode,collect_diagnostics)",
            "instantaneous_field": True,
            "mainline_imports": False,
            "cache_scope": "caller-owned-fixed-cache",
            "diagnostics_policy": "disabled-unless-call-index-aware-wrapper",
        }


@dataclass(frozen=True)
class ExistingModelEndpointAdapter:
    """Adapt the separate non-updating ``t=1`` endpoint readout."""

    model: object
    endpoint_extractor: EndpointExtractor
    execution_mode: str = "learned"

    def __post_init__(self) -> None:
        if not callable(getattr(self.model, "velocity", None)):
            raise TypeError("model must expose a callable velocity method")
        if not callable(self.endpoint_extractor):
            raise TypeError("endpoint_extractor must be callable")

    def __call__(self, state: Tensor, time: Tensor, cache: Cache) -> EndpointValue:
        velocity_method = getattr(self.model, "velocity", None)
        if not callable(velocity_method):
            raise TypeError("model must expose a callable velocity method")
        output = velocity_method(
            cache,
            noisy_action_field=state,
            time=time,
            execution_mode=self.execution_mode,
            collect_diagnostics=False,
        )
        return self.endpoint_extractor(output)

    def integration_metadata(self) -> dict[str, object]:
        return {
            "adapter": "existing_model_endpoint_head",
            "endpoint_time": 1.0,
            "updates_physical_state": False,
            "counted_as_physical_nfe": False,
        }


@dataclass(frozen=True)
class SolverBoundary:
    """Explicit labels for the two cache scopes in a proposal/refined run."""

    proposal_cache_label: str = "proposal-fixed-cache"
    refined_cache_label: str = "refined-fresh-cache"
    reset_solver_history_at_rebuild: bool = True

    def validate(self) -> None:
        if not self.proposal_cache_label.strip() or not self.refined_cache_label.strip():
            raise ValueError("cache labels must be non-empty")
        if not self.reset_solver_history_at_rebuild:
            raise ValueError("solver history cannot cross the W rebuild boundary")

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "proposal_cache": self.proposal_cache_label,
            "refined_cache": self.refined_cache_label,
            "solver_history_reset": self.reset_solver_history_at_rebuild,
        }


__all__ = [
    "ExistingModelEndpointAdapter",
    "ExistingModelVelocityAdapter",
    "SolverBoundary",
]
