"""Execution policy reads selectors and values separately; not a physical controller."""

from __future__ import annotations

LEGACY_CONTROLLER_VALUES = "normalized_legacy_v1"
RAW_CONTROLLER_VALUES = "separate_magnitude_v1"


def validate_controller_value_mode(mode: str) -> None:
    if mode not in {LEGACY_CONTROLLER_VALUES, RAW_CONTROLLER_VALUES}:
        raise ValueError(f"unknown controller value mode: {mode!r}")


def controller_value_metadata() -> dict[str, object]:
    return {
        "schema": "execution-selector-value-v1",
        "mode": RAW_CONTROLLER_VALUES,
        "evidence_keys": "explicit-selector-tokens-not-value-content",
        "values": "unnormalized-bias-free-source-and-private-state-projections",
        "candidate_attention": "normalized-QK-raw-V-bias-free-attention",
        "required": "explicit-matching-selector-and-value-streams",
        "limits": "learned-compute-policy-not-evidence-write-or-robot-actuator",
        "unchanged": "capacity-sigmoid-schedule-candidate-law-recurrence-and-host-write-budget",
    }
