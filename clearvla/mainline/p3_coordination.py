"""Declared P3 plan-feature semantics; not an executed-outcome observer."""

from __future__ import annotations

from .future_time import CONTROL_ALIGNED_FUTURE_TIME, resolve_future_time

POINTWISE_PLAN = "pointwise_legacy_v1"
TYPED_HORIZON_PLAN = "typed_horizon_v1"
P3_COORDINATION_MODES = (POINTWISE_PLAN, TYPED_HORIZON_PLAN)


def p3_coordination_metadata() -> dict[str, object]:
    return {
        "schema": "p3-typed-horizon-v1",
        "mode": TYPED_HORIZON_PLAN,
        "time_grid": resolve_future_time(CONTROL_ALIGNED_FUTURE_TIME).metadata(),
        "clock": "candidate-control-row-midpoints-not-elapsed-task-or-ode-time",
        "query": "original-action-query-not-summed-protected-consequence",
        "sources": ["action", "current_fact", "policy_precision", "semantic_effect",
                    "geometry_feature_effect", "task_temporal"],
        "scope": "all-proposed-rows-and-bases-within-one-current-observation",
        "value": "source-specific-projections-before-fusion-no-value-normalization",
        "protected": "original-consequence-and-policy-precision-by-reference",
        "change": "S-observed-change-modulates-plan-not-prediction-error",
        "zero_change": "exact-zero-value-and-plan-context-gradient",
        "feedback": "none-no-interval-mean-as-one-step-endpoint",
        "state": "stateless-read-only-through-ode-no-cross-decision-memory",
        "dropout": 0.0,
    }
