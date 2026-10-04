"""Source-bound role value interpretation; no second target or action bypass."""
from __future__ import annotations
ADDRESS_ONLY_ROLE = "address_only_v1"
CONTEXTUAL_ROLE_VALUES = "source_conditioned_values_v1"
ROLE_VALUE_MODES = (ADDRESS_ONLY_ROLE, CONTEXTUAL_ROLE_VALUES)


def role_value_metadata() -> dict[str, object]:
    return {
        "schema": "task-role-source-conditioned-values-v1",
        "mode": CONTEXTUAL_ROLE_VALUES,
        "consumers": ["S-interval", "coarse", "P1-relation-query", "P2-target-and-gap"],
        "source": "per-object-per-head-attended-values-before-K-mass-and-output-projection",
        "context": "existing-normalized-projected-query-no-raw-language-bypass",
        "formula": "sum_l(p_l*v_l)*(1+tanh(channel_gain)*tanh(projected_query))",
        "algebra": "query-only-channel-modulation-commutes-with-I/C-linear-sum-no-query-candidate-volume",
        "initialization": "zero-H-gain-no-new-RNG-exact-old-value-at-zero",
        "zero": "zero-source-remains-zero-no-K-reselection-null-mass-not-renormalized",
        "limits": "cannot-restore-upstream-lost-geometry-or-separate-identical-object-values",
    }
