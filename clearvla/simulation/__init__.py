"""Deployment-safe simulation boundaries for ClearVLA.

The package deliberately depends on simulator libraries only inside backend
modules.  Importing :mod:`clearvla.simulation` therefore remains safe in the
training environment when the optional simulation extras are absent.
"""

from .contracts import (
    ACTION_DIM,
    CAMERA_NAMES,
    STATE_DIM,
    EnvironmentDescriptor,
    EvaluationState,
    PolicyObservation,
    ResetResult,
    StepResult,
)
from .history import CausalHistory, HistorySnapshot

__all__ = [
    "ACTION_DIM",
    "CAMERA_NAMES",
    "STATE_DIM",
    "CausalHistory",
    "EnvironmentDescriptor",
    "EvaluationState",
    "HistorySnapshot",
    "PolicyObservation",
    "ResetResult",
    "StepResult",
]


