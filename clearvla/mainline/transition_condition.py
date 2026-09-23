"""CT is a decoder feature conditioner, not an action-conditioned physical W.

Typed source values and selector context have distinct owners. The new origin
is zero feature input, not a no-op action or a learned physical counterfactual.
"""

from __future__ import annotations

SUMMED_TRANSITION = "summed_legacy_v1"
TYPED_TRANSITION = "typed_plan_v1"
TRANSITION_CONDITION_MODES = (SUMMED_TRANSITION, TYPED_TRANSITION)


def validate_transition_condition_mode(mode: str) -> None:
    if mode not in TRANSITION_CONDITION_MODES:
        raise ValueError(f"unknown transition condition mode: {mode!r}")


def transition_condition_metadata() -> dict[str, object]:
    return {
        "schema": "typed-plan-transition-v1",
        "mode": TYPED_TRANSITION,
        "source": "completed-G3-feature-chart-not-W-state",
        "context": ["current-fact", "current-state-seed", "state-history-seed", "executed-seed"],
        "values": [
            "noisy-action-query",
            "policy-precision",
            "semantic-effect",
            "geometry-effect",
            "semantic-interaction",
            "geometry-interaction",
            "temporal-plan",
            "observed-change",
        ],
        "index": "retain-source-control-row-and-basis-through-latent-attention",
        "selector": "normalized-projected-sources-with-type-and-row-address",
        "value": "source-specific-bias-free-projection-no-value-LayerNorm",
        "coefficients": "bias-free-tanh-read-of-unnormalized-latent-values",
        "neutral": "exact-zero-feature-origin-no-learned-no-op-coefficients",
        "zero": "zero-dynamic-feature-values-imply-zero-CT-write-not-zero-policy",
        "clock": "current-ODE-call-plan-features-with-control-row-identities",
        "state": "stateless-no-future-supervision-no-cross-decision-memory",
        "limits": "not-world-rollout-executed-response-or-calibrated-physical-error",
    }
