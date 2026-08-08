"""Object-grounded intent/dynamics top for the schema-4 mainline.

The package is intentionally capability-named rather than version-named.  It
rejects historical schema-3 top weights; the trunk selects this graph only
when ``flow_jepa_object_intent_dynamics_mainline`` is enabled.
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
    CAPABILITY_SCHEMA,
    INTERVAL_BOUNDS,
    INTERVAL_NAMES,
    ArchitectureManifest,
    CoarseActionIntentState,
    DenseFactChart,
    FutureObjectDynamics,
    FuturePlanRecognition,
    ObjectFactSet,
    ObjectFactualDock,
    ObjectIntentState,
    ObjectTopTrainingTargets,
    manifest_from_mapping,
)

__all__ = [
    "ARCHITECTURE_MANIFEST",
    "CAPABILITY_SCHEMA",
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
    "ObjectFactualDock",
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
    "manifest_from_mapping",
]
