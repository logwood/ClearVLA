"""The capability-named ClearVLA mainline.

This package is the clean vertical implementation of the active policy.  It
does not select behavior by experiment version and it must not import the
historical V39 trainer or the version-switched policy trunk.

The formal ``scripts/smoke_mainline.sh`` and ``scripts/train_mainline.sh``
launchers enter this package directly. Historical V-numbered launchers remain
outside its architecture identity.
"""

from .config import ExperimentConfig, load_config
from .interfaces import OnlinePolicyInput, TrainingBatch
from .manifest import ARCHITECTURE_MANIFEST

__all__ = [
    "ARCHITECTURE_MANIFEST",
    "ExperimentConfig",
    "OnlinePolicyInput",
    "TrainingBatch",
    "load_config",
]
