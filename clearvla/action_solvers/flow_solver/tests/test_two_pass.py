from __future__ import annotations

import pytest
import torch

from clearvla.action_solvers.flow_solver import (
    ScheduleSpec,
    SolverSpec,
    TwoPassSpec,
    run_two_pass,
)


def _plan() -> TwoPassSpec:
    return TwoPassSpec(
        proposal=SolverSpec(
            schedule=ScheduleSpec.uniform(5),
            pass_role="proposal",
        ),
        refined=SolverSpec(
            schedule=ScheduleSpec.dense_jump(5, 0.5),
            pass_role="refined",
        ),
    )


def test_two_pass_restarts_exact_initial_state_and_resets_cache_scope() -> None:
    initial = torch.zeros(2, 1)
    seen: list[tuple[int, float, float]] = []

    def field(state: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        assert isinstance(cache, dict)
        seen.append((id(cache), float(time[0]), float(cache["bias"])))
        return torch.full_like(state, float(cache["bias"]))

    endpoint_labels: list[str] = []

    def endpoint(state: torch.Tensor, time: torch.Tensor, cache: object) -> str:
        endpoint_labels.append(str(cache["label"]))  # type: ignore[index]
        return str(cache["label"])  # type: ignore[index]

    def rebuild(state: torch.Tensor, endpoint_value: object, cache: object) -> dict[str, object]:
        assert endpoint_value == "proposal"
        assert cache["label"] == "proposal"  # type: ignore[index]
        return {"label": "refined", "bias": 2.0}

    proposal_cache = {"label": "proposal", "bias": 1.0}
    result = run_two_pass(
        initial,
        field,
        _plan(),
        proposal_cache,
        rebuild,
        endpoint_head=endpoint,
    )
    torch.testing.assert_close(result.proposal.initial_state, initial)
    torch.testing.assert_close(result.refined.initial_state, initial)
    assert result.proposal.initial_state is result.refined.initial_state
    assert endpoint_labels == ["proposal", "refined"]
    assert result.physical_nfe == 10
    assert result.endpoint_head_calls == 2
    assert result.total_dynamic_calls == 12
    assert result.refined_cache is not proposal_cache
    proposal_times = [time for cache_id, time, _ in seen if cache_id == id(proposal_cache)]
    refined_times = [time for cache_id, time, _ in seen if cache_id == id(result.refined_cache)]
    assert proposal_times == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert refined_times == pytest.approx([0.0, 0.125, 0.25, 0.375, 0.5])


def test_two_pass_rejects_in_place_cache_reuse() -> None:
    plan = _plan()

    def field(state: torch.Tensor, time: torch.Tensor, cache: object) -> torch.Tensor:
        return torch.zeros_like(state)

    def endpoint(state: torch.Tensor, time: torch.Tensor, cache: object) -> None:
        return None

    with pytest.raises(ValueError, match="fresh cache"):
        run_two_pass(
            torch.zeros(1, 1),
            field,
            plan,
            {},
            lambda state, endpoint_value, cache: cache,
            endpoint_head=endpoint,
        )
