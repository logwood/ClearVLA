from __future__ import annotations

import pytest
import torch

from clearvla.action_solvers.flow_solver import (
    ExistingModelEndpointAdapter,
    ExistingModelVelocityAdapter,
    SolverBoundary,
)


class _Bottom:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.physical_velocity = velocity
        self.motion_logits = velocity


class _Output:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.bottom = _Bottom(velocity)


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def velocity(self, cache: object, **kwargs: object) -> _Output:
        self.calls.append({"cache": cache, **kwargs})
        value = kwargs["noisy_action_field"]
        assert isinstance(value, torch.Tensor)
        return _Output(value + 1.0)


def test_existing_model_adapters_keep_extraction_explicit() -> None:
    model = _Model()
    velocity = ExistingModelVelocityAdapter(model, lambda output: output.bottom.physical_velocity)
    endpoint = ExistingModelEndpointAdapter(model, lambda output: output.bottom.motion_logits)
    state = torch.zeros(1, 2)
    time = torch.ones(1)
    cache = object()
    torch.testing.assert_close(velocity(state, time, cache), torch.ones_like(state))
    torch.testing.assert_close(endpoint(state, time, cache), torch.ones_like(state))
    assert model.calls[0]["execution_mode"] == "learned"
    assert model.calls[0]["collect_diagnostics"] is False
    assert endpoint.integration_metadata()["updates_physical_state"] is False
    assert velocity.integration_metadata()["instantaneous_field"] is True


def test_solver_boundary_is_fail_closed() -> None:
    boundary = SolverBoundary()
    boundary.validate()
    assert boundary.metadata()["solver_history_reset"] is True
    with pytest.raises(ValueError, match="cannot cross"):
        SolverBoundary(reset_solver_history_at_rebuild=False).validate()


def test_unindexed_model_diagnostics_are_rejected() -> None:
    with pytest.raises(ValueError, match="call-index-aware"):
        ExistingModelVelocityAdapter(
            _Model(),
            lambda output: output.bottom.physical_velocity,
            collect_diagnostics=True,
        )
