"""Standalone, versioned flow-solver contracts for ClearVLA.

This package is a diagnostic/provisional lane.  It is not imported by the
mainline sampler and never changes the default five-step deployment path on
its own.
"""

from .compat import (
    ExistingModelEndpointAdapter,
    ExistingModelVelocityAdapter,
    SolverBoundary,
)
from .diagnostics import (
    CandidateSpec,
    StepDoublingReport,
    candidate_by_name,
    candidate_matrix,
    compare_final_states,
    euler_step_doubling,
    proposal_shape_matrix,
    schedule_step_doubling_profile,
    tensor_rms,
)
from .gates import GateName, PromotionGates
from .integrate import (
    SolverTrace,
    TwoPassResult,
    euler_update,
    heun_update,
    integrate,
    rk4_update,
    run_two_pass,
)
from .panel import (
    U0_REQUIRED_ACCOUNTING_KEYS,
    U0_REQUIRED_IDENTITY_KEYS,
    U0_REQUIRED_SCOPE_KEYS,
    CacheFactory,
    PanelRecord,
    PanelResult,
    ReplayAttachment,
    run_candidate_panel,
)
from .protocols import (
    Cache,
    CacheRebuilder,
    EndpointHead,
    EndpointValue,
    TimeFactory,
    VelocityField,
    default_time_factory,
)
from .spec import (
    CacheBoundaryPolicy,
    CachePolicy,
    EndpointPolicy,
    InitialStatePolicy,
    PassRole,
    ScheduleKind,
    ScheduleSpec,
    SolverMethod,
    SolverSpec,
    TwoPassSpec,
)

__all__ = [
    "Cache",
    "CacheBoundaryPolicy",
    "CachePolicy",
    "CacheRebuilder",
    "CacheFactory",
    "CandidateSpec",
    "EndpointHead",
    "EndpointPolicy",
    "EndpointValue",
    "ExistingModelEndpointAdapter",
    "ExistingModelVelocityAdapter",
    "GateName",
    "InitialStatePolicy",
    "PanelRecord",
    "PanelResult",
    "ReplayAttachment",
    "U0_REQUIRED_ACCOUNTING_KEYS",
    "U0_REQUIRED_IDENTITY_KEYS",
    "U0_REQUIRED_SCOPE_KEYS",
    "PassRole",
    "PromotionGates",
    "ScheduleKind",
    "ScheduleSpec",
    "SolverBoundary",
    "SolverMethod",
    "SolverSpec",
    "SolverTrace",
    "StepDoublingReport",
    "TimeFactory",
    "TwoPassResult",
    "TwoPassSpec",
    "VelocityField",
    "candidate_by_name",
    "candidate_matrix",
    "compare_final_states",
    "default_time_factory",
    "euler_step_doubling",
    "proposal_shape_matrix",
    "euler_update",
    "heun_update",
    "integrate",
    "rk4_update",
    "run_two_pass",
    "run_candidate_panel",
    "schedule_step_doubling_profile",
    "tensor_rms",
]
