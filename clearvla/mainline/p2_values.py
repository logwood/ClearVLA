"""Versioned S-conditioned interpretation of W effect values, not new W states."""
from __future__ import annotations

WORLD_EFFECT_VALUES = "world_values_v1"
CONTEXTUAL_EFFECT_VALUES = "s_conditioned_values_v1"
P2_EFFECT_VALUE_MODES = (WORLD_EFFECT_VALUES, CONTEXTUAL_EFFECT_VALUES)


def p2_effect_value_metadata() -> dict[str, object]:
    return {
        "schema": "p2-s-conditioned-effect-values-v1",
        "mode": CONTEXTUAL_EFFECT_VALUES,
        "source": "W-effect-values-after-shared-K-and-named-view-selection",
        "context": "matching-S-interval-public-plus-typed-context",
        "formula": "base+sum_I(p_I*projected_W_I*tanh(S_I))*tanh(type_channel_gain)",
        "initialization": "zero-channel-gain-exact-old-output-no-extra-RNG",
        "zero": "zero-W-or-null-target-gives-zero-additional-effect",
        "ownership": "no-K-reselection-no-S-write-to-W-no-action-query-value",
        "limits": "learned-feature-interpretation-not-physical-truth-or-task-stop",
    }
