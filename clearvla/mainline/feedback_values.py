"""Observed comparison status is not signed world innovation or task progress."""
from __future__ import annotations
INNOVATION_ONLY = "innovation_only_v1"
INNOVATION_AND_STATUS = "innovation_and_status_v1"
FEEDBACK_VALUE_MODES = (INNOVATION_ONLY, INNOVATION_AND_STATUS)


def feedback_value_metadata() -> dict[str, object]:
    return {
        "schema": "executed-comparison-status-values-v1",
        "mode": INNOVATION_AND_STATUS,
        "source": "same-observed-posterior-null-and-current-to-past-soft-match",
        "features": ["target-match-null", "matched-uncertainty-with-unmatched-floor", "match-entropy"],
        "uncertainty": "q+(1-q)*normalized-joint-cell-camera-null-entropy",
        "meaning": "epistemic-status-feature-not-confidence-calibration-or-contact-truth",
        "consumer": "native-P3-world-feedback-reader-independent-of-signed-innovation",
        "formula": "old_output+output(status_projection(features)*(1+tanh(context)))",
        "initialization": "zero-3xH-status-projection-no-new-RNG",
        "support": "actual-executed-window-and-current-observed-target-support-no-loss-gating",
        "limits": "Teacher-labels-unchanged-no-goal-stop-no-hard-recovery-no-measured-target-success",
    }
