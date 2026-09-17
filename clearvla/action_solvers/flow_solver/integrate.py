"""Pure PyTorch Euler/Heun integration with explicit pass boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .protocols import (
    Cache,
    CacheRebuilder,
    EndpointHead,
    EndpointValue,
    TimeFactory,
    VelocityField,
    default_time_factory,
)
from .spec import SolverSpec, TwoPassSpec


def _validate_state(state: Tensor, *, name: str) -> None:
    if not isinstance(state, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not state.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _time(value: float, state: Tensor, factory: TimeFactory) -> Tensor:
    result = factory(float(value), state)
    if not isinstance(result, Tensor):
        raise TypeError("time_factory must return a torch.Tensor")
    if not result.is_floating_point():
        raise TypeError("time_factory must return a floating-point tensor")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("time_factory returned non-finite time values")
    return result


def _velocity(
    field: VelocityField,
    state: Tensor,
    time: Tensor,
    cache: Cache,
) -> Tensor:
    value = field(state, time, cache)
    if not isinstance(value, Tensor):
        raise TypeError("velocity field must return a torch.Tensor")
    if tuple(value.shape) != tuple(state.shape):
        raise ValueError(
            "velocity field shape must match state shape: "
            f"got {tuple(value.shape)} for {tuple(state.shape)}"
        )
    if value.device != state.device:
        raise ValueError("velocity field must remain on the state device")
    if not value.is_floating_point():
        raise TypeError("velocity field must return a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("velocity field returned non-finite values")
    # The mainline casts the physical velocity back to the state dtype before
    # its update.  Keep that boundary explicit while preserving autograd.
    return value.to(dtype=state.dtype)


def _rms(value: Tensor) -> Tensor:
    return value.square().mean().sqrt()


def euler_update(
    state: Tensor,
    field: VelocityField,
    cache: Cache,
    *,
    t0: float,
    t1: float,
    time_factory: TimeFactory = default_time_factory,
) -> tuple[Tensor, Tensor]:
    """Perform one validated left-endpoint Euler update."""

    _validate_state(state, name="state")
    t0 = float(t0)
    t1 = float(t1)
    if not (0.0 <= t0 < t1 <= 1.0):
        raise ValueError("Euler interval must satisfy 0 <= t0 < t1 <= 1")
    velocity = _velocity(field, state, _time(t0, state, time_factory), cache)
    next_state = state + (t1 - t0) * velocity
    _validate_state(next_state, name="Euler state")
    return next_state, velocity


def heun_update(
    state: Tensor,
    field: VelocityField,
    cache: Cache,
    *,
    t0: float,
    t1: float,
    time_factory: TimeFactory = default_time_factory,
) -> tuple[Tensor, Tensor, Tensor]:
    """Perform one explicit trapezoidal (improved-Euler) Heun update.

    The second field evaluation is a physical velocity evaluation even when
    ``t1 == 1``.  A separate endpoint head, if requested by ``integrate``, is
    evaluated afterwards on the corrected final state.
    """

    _validate_state(state, name="state")
    t0 = float(t0)
    t1 = float(t1)
    if not (0.0 <= t0 < t1 <= 1.0):
        raise ValueError("Heun interval must satisfy 0 <= t0 < t1 <= 1")
    start_time = _time(t0, state, time_factory)
    start_velocity = _velocity(field, state, start_time, cache)
    predicted = state + (t1 - t0) * start_velocity
    _validate_state(predicted, name="Heun predictor")
    end_velocity = _velocity(
        field,
        predicted,
        _time(t1, predicted, time_factory),
        cache,
    )
    next_state = state + (t1 - t0) * 0.5 * (start_velocity + end_velocity)
    _validate_state(next_state, name="Heun state")
    return next_state, start_velocity, end_velocity


def rk4_update(
    state: Tensor,
    field: VelocityField,
    cache: Cache,
    *,
    t0: float,
    t1: float,
    time_factory: TimeFactory = default_time_factory,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Perform one classical fourth-order Runge–Kutta update.

    RK4 is exposed as a dense-reference/oracle method first.  Its four field
    evaluations are all physical NFE; a separate endpoint head remains outside
    this update.
    """

    _validate_state(state, name="state")
    t0 = float(t0)
    t1 = float(t1)
    if not (0.0 <= t0 < t1 <= 1.0):
        raise ValueError("RK4 interval must satisfy 0 <= t0 < t1 <= 1")
    half = 0.5 * (t1 - t0)
    k1 = _velocity(field, state, _time(t0, state, time_factory), cache)
    first_predictor = state + half * k1
    _validate_state(first_predictor, name="RK4 first predictor")
    k2 = _velocity(
        field,
        first_predictor,
        _time(t0 + half, first_predictor, time_factory),
        cache,
    )
    second_predictor = state + half * k2
    _validate_state(second_predictor, name="RK4 second predictor")
    k3 = _velocity(
        field,
        second_predictor,
        _time(t0 + half, second_predictor, time_factory),
        cache,
    )
    third_predictor = state + (t1 - t0) * k3
    _validate_state(third_predictor, name="RK4 third predictor")
    k4 = _velocity(
        field,
        third_predictor,
        _time(t1, third_predictor, time_factory),
        cache,
    )
    next_state = state + (t1 - t0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    _validate_state(next_state, name="RK4 state")
    return next_state, k1, k2, k3, k4


@dataclass(frozen=True)
class SolverTrace:
    """Result and compact diagnostics for one fixed-cache solver pass."""

    initial_state: Tensor
    final_state: Tensor
    spec: SolverSpec
    nfe: int
    endpoint_calls: int
    endpoint_value: EndpointValue = None
    states: tuple[Tensor, ...] | None = None
    effective_velocities: tuple[Tensor, ...] | None = None
    correction_rms: tuple[Tensor, ...] | None = None

    @property
    def interval_count(self) -> int:
        return self.spec.schedule.interval_count

    @property
    def total_dynamic_calls(self) -> int:
        return self.nfe + self.endpoint_calls

    @property
    def query_times(self) -> tuple[float, ...]:
        return self.spec.schedule.query_times

    @property
    def step_sizes(self) -> tuple[float, ...]:
        return self.spec.schedule.step_sizes

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe scalar metadata without retaining tensors."""

        return {
            "solver_fingerprint": self.spec.fingerprint,
            "schedule_fingerprint": self.spec.schedule.fingerprint,
            "method": self.spec.method,
            "pass_role": self.spec.pass_role,
            "interval_count": self.interval_count,
            "nfe": self.nfe,
            "endpoint_calls": self.endpoint_calls,
            "total_dynamic_calls": self.total_dynamic_calls,
            "max_step": self.spec.schedule.max_step,
            "min_step": self.spec.schedule.min_step,
        }


def integrate(
    initial_state: Tensor,
    field: VelocityField,
    spec: SolverSpec,
    cache: Cache,
    *,
    time_factory: TimeFactory = default_time_factory,
    endpoint_head: EndpointHead | None = None,
    retain_trajectory: bool = False,
    retain_velocities: bool = False,
    record_corrections: bool = False,
) -> SolverTrace:
    """Integrate one schedule inside one immutable cache boundary.

    No solver history is accepted from the caller.  This is intentional: a
    future multistep or pseudo-corrector must create a fresh state for every
    invocation, and the two-pass runner consequently cannot leak history over
    the proposal → W rebuild boundary.
    """

    if not callable(field):
        raise TypeError("field must be callable")
    spec.validate()
    _validate_state(initial_state, name="initial_state")
    current = initial_state
    states: list[Tensor] | None = [current] if retain_trajectory else None
    effective_velocities: list[Tensor] | None = [] if retain_velocities else None
    correction_rms: list[Tensor] | None = [] if record_corrections else None
    nfe = 0

    for t0, t1 in zip(
        spec.schedule.boundaries,
        spec.schedule.boundaries[1:],
    ):
        if spec.method == "euler":
            current, velocity = euler_update(
                current,
                field,
                cache,
                t0=t0,
                t1=t1,
                time_factory=time_factory,
            )
            nfe += 1
            if effective_velocities is not None:
                effective_velocities.append(velocity)
        elif spec.method == "heun":
            current, start_velocity, end_velocity = heun_update(
                current,
                field,
                cache,
                t0=t0,
                t1=t1,
                time_factory=time_factory,
            )
            nfe += 2
            if effective_velocities is not None:
                effective_velocities.append(0.5 * (start_velocity + end_velocity))
            if correction_rms is not None:
                correction_rms.append(_rms(end_velocity - start_velocity))
        else:
            current, k1, k2, k3, k4 = rk4_update(
                current,
                field,
                cache,
                t0=t0,
                t1=t1,
                time_factory=time_factory,
            )
            nfe += 4
            if effective_velocities is not None:
                effective_velocities.append((k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
        if states is not None:
            states.append(current)

    endpoint_value: EndpointValue = None
    endpoint_calls = 0
    if endpoint_head is not None:
        endpoint_value = endpoint_head(
            current,
            _time(1.0, current, time_factory),
            cache,
        )
        endpoint_calls = 1

    return SolverTrace(
        initial_state=initial_state,
        final_state=current,
        spec=spec,
        nfe=nfe,
        endpoint_calls=endpoint_calls,
        endpoint_value=endpoint_value,
        states=tuple(states) if states is not None else None,
        effective_velocities=(
            tuple(effective_velocities) if effective_velocities is not None else None
        ),
        correction_rms=tuple(correction_rms) if correction_rms is not None else None,
    )


@dataclass(frozen=True)
class TwoPassResult:
    """Proposal/refined traces and their explicit cache boundary."""

    proposal: SolverTrace
    refined: SolverTrace
    proposal_cache: Cache
    refined_cache: Cache

    @property
    def initial_state(self) -> Tensor:
        return self.proposal.initial_state

    @property
    def physical_nfe(self) -> int:
        return self.proposal.nfe + self.refined.nfe

    @property
    def endpoint_head_calls(self) -> int:
        return self.proposal.endpoint_calls + self.refined.endpoint_calls

    @property
    def total_dynamic_calls(self) -> int:
        return self.physical_nfe + self.endpoint_head_calls

    def summary(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.summary(),
            "refined": self.refined.summary(),
            "physical_nfe": self.physical_nfe,
            "endpoint_head_calls": self.endpoint_head_calls,
            "total_dynamic_calls": self.total_dynamic_calls,
            "initial_state_reused": bool(
                torch.equal(self.proposal.initial_state, self.refined.initial_state)
            ),
            "cache_identity_changed": self.proposal_cache is not self.refined_cache,
        }


def run_two_pass(
    initial_state: Tensor,
    field: VelocityField,
    plan: TwoPassSpec,
    proposal_cache: Cache,
    rebuild_cache: CacheRebuilder,
    *,
    endpoint_head: EndpointHead,
    time_factory: TimeFactory = default_time_factory,
    retain_trajectory: bool = False,
    retain_velocities: bool = False,
    record_corrections: bool = False,
) -> TwoPassResult:
    """Run proposal → fresh cache rebuild → refined from the same initial state.

    The callback is deliberately the only way to cross the W boundary.  The
    runner does not rebuild a cache itself, and it rejects an in-place cache
    reuse because a multistep solver's state must be reset at that boundary.
    """

    if not callable(rebuild_cache):
        raise TypeError("rebuild_cache must be callable")
    if endpoint_head is None or not callable(endpoint_head):
        raise TypeError("two-pass execution requires an endpoint_head callback")
    plan.validate()
    proposal = integrate(
        initial_state,
        field,
        plan.proposal,
        proposal_cache,
        time_factory=time_factory,
        endpoint_head=endpoint_head,
        retain_trajectory=retain_trajectory,
        retain_velocities=retain_velocities,
        record_corrections=record_corrections,
    )
    refined_cache = rebuild_cache(
        proposal.final_state,
        proposal.endpoint_value,
        proposal_cache,
    )
    if refined_cache is proposal_cache:
        raise ValueError(
            "rebuild_cache must return a fresh cache object so solver history "
            "cannot cross the W rebuild boundary"
        )
    refined = integrate(
        proposal.initial_state,
        field,
        plan.refined,
        refined_cache,
        time_factory=time_factory,
        endpoint_head=endpoint_head,
        retain_trajectory=retain_trajectory,
        retain_velocities=retain_velocities,
        record_corrections=record_corrections,
    )
    return TwoPassResult(
        proposal=proposal,
        refined=refined,
        proposal_cache=proposal_cache,
        refined_cache=refined_cache,
    )


__all__ = [
    "SolverTrace",
    "TwoPassResult",
    "euler_update",
    "heun_update",
    "integrate",
    "rk4_update",
    "run_two_pass",
]
