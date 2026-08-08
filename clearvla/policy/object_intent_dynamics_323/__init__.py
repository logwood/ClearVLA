"""Object-grounded intent/dynamics top for the post-V119 mainline.

The package is intentionally capability-named rather than version-named.  It
does not mutate the historical V119 implementation; the trunk selects this
graph only when ``flow_jepa_object_intent_dynamics_mainline`` is enabled.
"""

from .compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ZeroPreservingObjectConsequence,
)
from .dynamics import ObjectFutureDynamicsCompiler, ObjectW1WorkingState
from .grounding import DenseObjectGrounder
from .intent import (
    CoarseActionIntent,
    FuturePlanRecognizer,
    StatelessObjectIntentOrganizer,
)
from .teacher import ObjectFutureTeacher
from .types import (
    ARCHITECTURE_MANIFEST,
    INTERVAL_BOUNDS,
    INTERVAL_NAMES,
    ArchitectureManifest,
    CoarseActionIntentState,
    DenseFactChart,
    FutureObjectDynamics,
    FuturePlanRecognition,
    ObjectFactSet,
    ObjectIntentState,
    ObjectTopTrainingTargets,
)

__all__ = [
    "ARCHITECTURE_MANIFEST",
    "INTERVAL_BOUNDS",
    "INTERVAL_NAMES",
    "ArchitectureManifest",
    "CoarseActionIntent",
    "CoarseActionIntentState",
    "DenseFactChart",
    "DenseObjectGrounder",
    "FutureObjectDynamics",
    "FuturePlanRecognition",
    "FuturePlanRecognizer",
    "ObjectConsequenceState",
    "ObjectFactSet",
    "ObjectFutureDynamicsCompiler",
    "ObjectW1WorkingState",
    "ObjectFutureEffectReader",
    "ObjectFutureTeacher",
    "ObjectIntentState",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "ObjectTopTrainingTargets",
    "StatelessObjectIntentOrganizer",
    "ZeroPreservingObjectConsequence",
]
