"""Diagnostics and reproducible candidate panels for the solver lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .integrate import euler_update
from .protocols import Cache, TimeFactory, VelocityField, default_time_factory
from .spec import ScheduleSpec, SolverSpec, TwoPassSpec


def tensor_rms(value: Tensor) -> Tensor:
    """Return an RMS scalar while preserving the input's autograd graph."""

    return value.square().mean().sqrt()


@dataclass(frozen=True)
class StepDoublingReport:
    """Coarse-vs-two-half-step error for one interval."""

    t0: float
    t1: float
    coarse_state: Tensor
    fine_state: Tensor
    absolute_rms: Tensor
    relative_rms: Tensor
    diagnostic_nfe: int = 3

    @property
    def interval_size(self) -> float:
        return self.t1 - self.t0


def euler_step_doubling(
    state: Tensor,
    field: VelocityField,
    cache: Cache,
    *,
    t0: float,
    t1: float,
    time_factory: TimeFactory = default_time_factory,
) -> StepDoublingReport:
    """Compare one Euler step with two half Euler steps.

    This is an audit oracle only; it does not alter the caller's state or
    cache.  The same instantaneous field ABI is used for both trajectories.
    """

    coarse, _ = euler_update(
        state,
        field,
        cache,
        t0=t0,
        t1=t1,
        time_factory=time_factory,
    )
    midpoint = t0 + 0.5 * (t1 - t0)
    first_half, _ = euler_update(
        state,
        field,
        cache,
        t0=t0,
        t1=midpoint,
        time_factory=time_factory,
    )
    fine, _ = euler_update(
        first_half,
        field,
        cache,
        t0=midpoint,
        t1=t1,
        time_factory=time_factory,
    )
    absolute = tensor_rms(fine - coarse)
    denominator = tensor_rms(fine).clamp_min(torch.finfo(fine.dtype).eps)
    return StepDoublingReport(
        t0=float(t0),
        t1=float(t1),
        coarse_state=coarse,
        fine_state=fine,
        absolute_rms=absolute,
        relative_rms=absolute / denominator,
        diagnostic_nfe=3,
    )


def schedule_step_doubling_profile(
    state: Tensor,
    field: VelocityField,
    cache: Cache,
    schedule: ScheduleSpec,
    *,
    time_factory: TimeFactory = default_time_factory,
) -> tuple[StepDoublingReport, ...]:
    """Collect one local Euler error report for each schedule interval."""

    reports: list[StepDoublingReport] = []
    current = state
    for t0, t1 in zip(schedule.boundaries, schedule.boundaries[1:]):
        report = euler_step_doubling(
            current,
            field,
            cache,
            t0=t0,
            t1=t1,
            time_factory=time_factory,
        )
        reports.append(report)
        # Advance the reference trajectory by the coarse step so every report
        # probes the same left-endpoint path as the deployed Euler schedule.
        current = report.coarse_state
    return tuple(reports)


@dataclass(frozen=True)
class CandidateSpec:
    """Named entry in the isolated two-pass solver panel."""

    name: str
    plan: TwoPassSpec
    purpose: str
    tier: Literal["same_cost", "shape_diagnostic", "higher_budget", "oracle"]
    source_note: str = ""

    @property
    def physical_nfe(self) -> int:
        return self.plan.physical_nfe

    @property
    def endpoint_head_calls(self) -> int:
        return self.plan.endpoint_head_calls

    @property
    def total_dynamic_calls(self) -> int:
        return self.plan.total_dynamic_calls


def _pass(schedule: ScheduleSpec, method: str, role: str) -> SolverSpec:
    return SolverSpec(
        schedule=schedule,
        method=method,  # type: ignore[arg-type]
        pass_role=role,  # type: ignore[arg-type]
    )


def _plan(
    proposal_schedule: ScheduleSpec,
    refined_schedule: ScheduleSpec,
    *,
    proposal_method: str = "euler",
    refined_method: str = "euler",
) -> TwoPassSpec:
    return TwoPassSpec(
        proposal=_pass(proposal_schedule, proposal_method, "proposal"),
        refined=_pass(refined_schedule, refined_method, "refined"),
    )


def candidate_matrix() -> tuple[CandidateSpec, ...]:
    """Return the pre-registered, source-independent solver candidates.

    The list is metadata only.  It does not run a model, change a config or
    imply that any candidate is production-ready.
    """

    e5 = ScheduleSpec.uniform(5)
    dj5 = ScheduleSpec.dense_jump(5, 0.5)
    e6 = ScheduleSpec.uniform(6)
    e6_late = ScheduleSpec.late_split(5)
    e6_warp = ScheduleSpec.late_warp(6, 1.25)
    e6_cosine = ScheduleSpec.cosine(6)
    e10 = ScheduleSpec.uniform(10)
    h5 = ScheduleSpec.uniform(5)
    return (
        CandidateSpec(
            "E5/E5",
            _plan(e5, e5),
            "current two-pass reference",
            "same_cost",
        ),
        CandidateSpec(
            "E5/DJ5(.5)",
            _plan(e5, dj5),
            "same-cost front-dense, terminal-jump refined pass",
            "same_cost",
            "Dense-Jump Flow Matching, arXiv:2509.13574",
        ),
        CandidateSpec(
            "DJ5(.5)/DJ5(.5)",
            _plan(dj5, dj5),
            "same-cost Dense-Jump in both passes",
            "same_cost",
            "solver state must reset at the W rebuild boundary",
        ),
        CandidateSpec(
            "E5/E6-uniform",
            _plan(e5, e6),
            "one extra refined Euler update with global uniform refinement",
            "shape_diagnostic",
        ),
        CandidateSpec(
            "E5/E6-late",
            _plan(e5, e6_late),
            "one extra refined update by splitting only the final interval",
            "shape_diagnostic",
        ),
        CandidateSpec(
            "E5/E6-late-warp",
            _plan(e5, e6_warp),
            "one extra refined update with a smooth late-biased grid",
            "shape_diagnostic",
        ),
        CandidateSpec(
            "E5/E6-cosine",
            _plan(e5, e6_cosine),
            "one extra refined update with both-endpoint clustering",
            "shape_diagnostic",
            "optional pressure test when early and late interval curvature are both suspected",
        ),
        CandidateSpec(
            "E5/E10",
            _plan(e5, e10),
            "spend extra accuracy only on the executed refined pass",
            "higher_budget",
        ),
        CandidateSpec(
            "E10/E5",
            _plan(e10, e5),
            "spend extra accuracy only on proposal/W conditioning",
            "higher_budget",
        ),
        CandidateSpec(
            "E10/E10",
            _plan(e10, e10),
            "uniform ten-update control for both passes",
            "higher_budget",
            "π0/π0.5 and SmolVLA-style ten-call comparison",
        ),
        CandidateSpec(
            "E5/H5",
            _plan(e5, h5, refined_method="heun"),
            "matched-NFE refined Heun versus refined Euler-10",
            "higher_budget",
        ),
        CandidateSpec(
            "H5/H5",
            _plan(h5, h5, proposal_method="heun", refined_method="heun"),
            "full matched-NFE Heun control",
            "higher_budget",
        ),
        CandidateSpec(
            "E5/RK4-oracle",
            _plan(e5, e5, refined_method="rk4"),
            "five-interval RK4 higher-order oracle for the refined pass",
            "oracle",
            "diagnostic upper bound; not a dense-grid convergence reference or production budget",
        ),
    )


def proposal_shape_matrix() -> tuple[CandidateSpec, ...]:
    """Return proposal-only shape controls with the refined pass fixed at E5.

    These entries are intentionally kept out of :func:`candidate_matrix` so
    the default panel does not silently multiply the number of expensive
    replays.  They answer the causal question ``does the proposal grid alter
    the rebuilt W enough to matter?`` while holding the executed refined
    solver constant.
    """

    e5 = ScheduleSpec.uniform(5)
    proposal_schedules = (
        ("E6-uniform/E5", ScheduleSpec.uniform(6), "uniform proposal shape"),
        ("E6-late/E5", ScheduleSpec.late_split(5), "late-split proposal shape"),
        (
            "E6-late-warp/E5",
            ScheduleSpec.late_warp(6, 1.25),
            "smooth late-warp proposal shape",
        ),
        (
            "E6-cosine/E5",
            ScheduleSpec.cosine(6),
            "both-endpoint proposal shape",
        ),
    )
    return tuple(
        CandidateSpec(
            name,
            _plan(schedule, e5),
            f"one extra proposal update with {description}; refined E5 is fixed",
            "shape_diagnostic",
            "proposal-only W attribution control",
        )
        for name, schedule, description in proposal_schedules
    )


def candidate_by_name(name: str) -> CandidateSpec:
    """Resolve one registered candidate or fail closed."""

    aliases = {
        # Keep the historical public key readable while offering an explicit
        # name that records the five-interval RK4 density.
        "E5/RK4-5-oracle": "E5/RK4-oracle",
        "E5/RK-oracle": "E5/RK4-oracle",
    }
    canonical_name = aliases.get(name, name)
    for candidate in (*candidate_matrix(), *proposal_shape_matrix()):
        if candidate.name == canonical_name:
            return candidate
    raise KeyError(f"unknown flow solver candidate: {name!r}")


def compare_final_states(reference: Tensor, candidate: Tensor) -> dict[str, Tensor]:
    """Return compact absolute/relative differences for a replay pair."""

    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError("reference and candidate states must have the same shape")
    delta = candidate - reference
    absolute = tensor_rms(delta)
    relative = absolute / tensor_rms(reference).clamp_min(
        torch.finfo(reference.dtype).eps
    )
    return {
        "absolute_rms": absolute,
        "relative_rms": relative,
        "max_abs": delta.abs().amax(),
    }


__all__ = [
    "CandidateSpec",
    "StepDoublingReport",
    "candidate_by_name",
    "candidate_matrix",
    "proposal_shape_matrix",
    "compare_final_states",
    "euler_step_doubling",
    "schedule_step_doubling_profile",
    "tensor_rms",
]
