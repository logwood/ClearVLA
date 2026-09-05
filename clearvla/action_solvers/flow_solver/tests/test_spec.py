from __future__ import annotations

import math

import pytest

from clearvla.action_solvers.flow_solver import (
    ScheduleSpec,
    SolverSpec,
    TwoPassSpec,
    candidate_by_name,
    candidate_matrix,
    proposal_shape_matrix,
)


def _pass(schedule: ScheduleSpec, role: str, method: str = "euler") -> SolverSpec:
    return SolverSpec(schedule=schedule, pass_role=role, method=method)  # type: ignore[arg-type]


def test_uniform_schedule_has_left_endpoint_queries_and_exact_endpoints() -> None:
    schedule = ScheduleSpec.uniform(5)
    assert schedule.boundaries == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert schedule.query_times == (0.0, 0.2, 0.4, 0.6, 0.8)
    assert schedule.interval_count == 5
    assert schedule.is_uniform
    assert schedule.runtime_boundaries == pytest.approx(schedule.boundaries)
    assert schedule.runtime_step_sizes == pytest.approx(schedule.step_sizes)
    assert math.isclose(sum(schedule.step_sizes), 1.0)


def test_dense_jump_matches_the_paper_node_definition() -> None:
    schedule = ScheduleSpec.dense_jump(5, 0.5)
    assert schedule.boundaries == (0.0, 0.125, 0.25, 0.375, 0.5, 1.0)
    assert schedule.query_times[-1] == 0.5
    assert schedule.max_step == 0.5
    assert schedule.parameters == (("steps", 5), ("t_jump", 0.5))


def test_shape_builders_are_strictly_increasing_and_have_expected_counts() -> None:
    schedules = (
        ScheduleSpec.late_split(5),
        ScheduleSpec.late_warp(6, 1.25),
        ScheduleSpec.cosine(6),
    )
    for schedule in schedules:
        assert schedule.interval_count == 6
        assert schedule.boundaries[0] == 0.0
        assert schedule.boundaries[-1] == 1.0
        assert all(right > left for left, right in zip(schedule.boundaries, schedule.boundaries[1:]))


def test_schedule_and_solver_round_trip_preserve_identity() -> None:
    schedule = ScheduleSpec.late_warp(6, 1.25)
    restored_schedule = ScheduleSpec.from_dict(schedule.to_dict())
    assert restored_schedule == schedule
    assert restored_schedule.fingerprint == schedule.fingerprint

    solver = SolverSpec(schedule=schedule, method="heun", pass_role="refined")
    restored_solver = SolverSpec.from_dict(solver.to_dict())
    assert restored_solver == solver
    assert restored_solver.fingerprint == solver.fingerprint

    plan = TwoPassSpec(
        proposal=_pass(ScheduleSpec.uniform(5), "proposal"),
        refined=solver,
    )
    restored_plan = TwoPassSpec.from_dict(plan.to_dict())
    assert restored_plan == plan
    assert restored_plan.physical_nfe == 5 + 12


def test_invalid_schedule_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ScheduleSpec(boundaries=(0.0, 0.5, 0.5, 1.0))
    with pytest.raises(ValueError, match="start at 0.0"):
        ScheduleSpec(boundaries=(0.1, 0.5, 1.0))
    with pytest.raises(ValueError, match="finite"):
        ScheduleSpec(boundaries=(0.0, float("nan"), 1.0))
    with pytest.raises(ValueError, match="float32 conversion"):
        ScheduleSpec(boundaries=(0.0, 1.0e-50, 1.0))
    with pytest.raises(ValueError, match="t_jump"):
        ScheduleSpec.dense_jump(5, 1.0)
    with pytest.raises(ValueError, match="split_fraction"):
        ScheduleSpec.late_split(5, split_fraction=0.0)
    with pytest.raises(ValueError, match="unknown flow schedule fields"):
        ScheduleSpec.from_dict({"boundaries": [0.0, 1.0], "unexpected": 1})


def test_solver_and_two_pass_contracts_reject_wrong_ownership() -> None:
    with pytest.raises(ValueError, match="proposal"):
        TwoPassSpec(
            proposal=_pass(ScheduleSpec.uniform(5), "single"),
            refined=_pass(ScheduleSpec.uniform(5), "refined"),
        )
    with pytest.raises(ValueError, match="endpoint"):
        SolverSpec(
            schedule=ScheduleSpec.uniform(5),
            endpoint_policy="not_external",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exact initial state"):
        TwoPassSpec(
            proposal=_pass(ScheduleSpec.uniform(5), "proposal"),
            refined=_pass(ScheduleSpec.uniform(5), "refined"),
            initial_state_policy="independent",  # type: ignore[arg-type]
        )


def test_candidate_matrix_has_matched_cost_and_shape_entries() -> None:
    candidates = candidate_matrix()
    names = {candidate.name for candidate in candidates}
    assert {"E5/E5", "E5/DJ5(.5)", "DJ5(.5)/DJ5(.5)"} <= names
    assert {"E5/E6-uniform", "E5/E6-late", "E5/E6-cosine"} <= names
    assert candidate_by_name("E5/E5").physical_nfe == 10
    assert candidate_by_name("E5/DJ5(.5)").physical_nfe == 10
    assert candidate_by_name("E5/E6-uniform").physical_nfe == 11
    assert candidate_by_name("E5/E6-cosine").physical_nfe == 11
    assert candidate_by_name("E5/H5").physical_nfe == 15
    assert candidate_by_name("E5/RK4-oracle").physical_nfe == 25
    assert "not a dense" in candidate_by_name("E5/RK4-oracle").source_note
    assert candidate_by_name("E5/RK4-5-oracle").name == "E5/RK4-oracle"
    assert candidate_by_name("E5/RK-oracle").name == "E5/RK4-oracle"
    with pytest.raises(KeyError, match="unknown"):
        candidate_by_name("missing")


def test_proposal_shape_matrix_keeps_refined_e5_fixed() -> None:
    controls = proposal_shape_matrix()
    assert {candidate.name for candidate in controls} == {
        "E6-uniform/E5",
        "E6-late/E5",
        "E6-late-warp/E5",
        "E6-cosine/E5",
    }
    assert all(candidate.physical_nfe == 11 for candidate in controls)
    assert all(candidate.plan.refined.intervals == 5 for candidate in controls)
    assert candidate_by_name("E6-late/E5").plan.refined.schedule == ScheduleSpec.uniform(5)
