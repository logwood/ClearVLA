"""Evidence magnitudes and selector identities have separate bottom owners.

This changes the native evidence adapter and host MMDiT value write, not W's
physical semantics or the execution controller's capacity policy.
"""
from __future__ import annotations

NORMALIZED_EVIDENCE = "normalized_legacy_v1"
MAGNITUDE_EVIDENCE = "magnitude_preserving_v1"
EVIDENCE_VALUE_MODES = (NORMALIZED_EVIDENCE, MAGNITUDE_EVIDENCE)


def validate_evidence_value_mode(mode: str) -> None:
    if mode not in EVIDENCE_VALUE_MODES:
        raise ValueError(f"unknown bottom evidence value mode: {mode!r}")


def bottom_evidence_metadata() -> dict[str, object]:
    return {
        "schema": "bottom-magnitude-evidence-v1",
        "mode": MAGNITUDE_EVIDENCE,
        "selector": "source-normalized-with-type-and-role-prior",
        "value": "bias-free-source-projection-no-bank-normalization-or-type-embedding",
        "layer_value": "clean-intent-attention-not-mixed-layer-selector",
        "host_key": "normalized-selector-with-independent-key-projection",
        "host_value": "independent-bias-free-projection-without-layer-normalization",
        "host_write": "bias-free-output-smooth-unit-RMS-upper-bound-not-unit-rescaling",
        "zero": "zero-values-yield-zero-evidence-attention-write-not-zero-whole-action",
        "scope": "native-evidence-adapter-and-host-MMDiT-cross-write",
        "limits": "CT-is-learned-noisy-plan-feature-not-physical-world-error;controller-policy-separate",
        "state": "stateless-per-existing-ODE-call-no-new-rollout-or-solver-evaluation",
    }
