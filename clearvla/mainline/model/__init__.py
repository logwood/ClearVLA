"""Online model composition for the ClearVLA mainline.

The final package owns observation preparation, G/S/W/P composition and the
single typed ingress into the retained bottom action model.
"""

from .action_contract import BottomDecoderOutput, BottomOutput
from .component_contracts import (
    COMPONENT_ABI_REVISION,
    ComponentSelection,
    DynamicQueryBundle,
    GroundedObservationBundle,
    GroundingSeed,
    OutletActionOutput,
    PolicyCompileResult,
    SharedRoleContext,
    TerminalHeadOutput,
)
from .components import (
    BridgeStage,
    ConditioningStage,
    ExecutionBottomStage,
    GroundingStage,
    IntentStage,
    ObservationStage,
    OutletAdapter,
    P1Stage,
    PolicyCompilerStage,
    TrainingTargetsStage,
    WorldStage,
)
from .compiler import (
    ObjectConsequenceState,
    ObjectFutureEffectReader,
    ObjectPolicyPlanCompiler,
    ObjectPolicyPlanDeltaBank,
    ObjectTypedEffect,
    ZeroPreservingObjectConsequence,
)
from .dynamics import ObjectFutureDynamicsCompiler, ObjectW1WorkingState
from .grounding import DenseObjectGrounder
from .intent import (
    CoarseActionIntent,
    FuturePlanRecognizer,
    StatelessObjectIntentOrganizer,
)
from .observation_contract import ObservationEvidence, PatchFlowField
from .policy import ClearVLAMainlinePolicy, OnlinePolicyCache, PolicyStepOutput
from .proposal import HistoryActionProposal
from .restored_bottom import RestoredV120EvidenceBottom
from .restored_observation import RestoredV120ObservationCompiler
from .teacher import ObjectFutureTeacher
from .top import CompiledPolicyState, ObjectIntentDynamicsTop, OnlineTopContext
from .transition import ControlledTransitionDynamics
from .types import (
    ActionIntentDock,
    CandidateWorld,
    CoarseActionIntentState,
    CompletedP1PolicyState,
    ControlledTransitionState,
    DenseFactChart,
    FactualIntentDock,
    FactualPrecisionDock,
    FutureObjectDynamics,
    FuturePlanRecognition,
    HistoryActionProposalState,
    LocalFactSet,
    ObjectFactSet,
    ObjectIntentState,
    ObjectTopTrainingTargets,
    ObjectWorldBelief,
    P2QueryDock,
    PhysicalActionCondition,
    PolicyIntentDock,
    StatelessIntentBundle,
)

__all__ = [
    "ActionIntentDock",
    "COMPONENT_ABI_REVISION",
    "ComponentSelection",
    "CandidateWorld",
    "BottomOutput",
    "BottomDecoderOutput",
    "BridgeStage",
    "ConditioningStage",
    "ClearVLAMainlinePolicy",
    "CoarseActionIntent",
    "CoarseActionIntentState",
    "CompletedP1PolicyState",
    "CompiledPolicyState",
    "ControlledTransitionDynamics",
    "ControlledTransitionState",
    "DenseFactChart",
    "DenseObjectGrounder",
    "DynamicQueryBundle",
    "ExecutionBottomStage",
    "RestoredV120EvidenceBottom",
    "RestoredV120ObservationCompiler",
    "FutureObjectDynamics",
    "FuturePlanRecognition",
    "FuturePlanRecognizer",
    "FactualIntentDock",
    "HistoryActionProposal",
    "HistoryActionProposalState",
    "GroundedObservationBundle",
    "GroundingSeed",
    "GroundingStage",
    "IntentStage",
    "LocalFactSet",
    "ObjectConsequenceState",
    "FactualPrecisionDock",
    "ObjectFactSet",
    "ObjectWorldBelief",
    "ObjectFutureDynamicsCompiler",
    "ObjectFutureEffectReader",
    "ObjectIntentState",
    "ObjectIntentDynamicsTop",
    "ObjectPolicyPlanCompiler",
    "ObjectPolicyPlanDeltaBank",
    "ObjectTypedEffect",
    "ObjectTopTrainingTargets",
    "ObjectW1WorkingState",
    "ObservationStage",
    "OutletActionOutput",
    "OutletAdapter",
    "P1Stage",
    "PolicyCompileResult",
    "PolicyCompilerStage",
    "PhysicalActionCondition",
    "ObservationEvidence",
    "OnlinePolicyCache",
    "OnlineTopContext",
    "PatchFlowField",
    "P2QueryDock",
    "PolicyStepOutput",
    "PolicyIntentDock",
    "StatelessIntentBundle",
    "SharedRoleContext",
    "TerminalHeadOutput",
    "TrainingTargetsStage",
    "ObjectFutureTeacher",
    "StatelessObjectIntentOrganizer",
    "ZeroPreservingObjectConsequence",
    "WorldStage",
]
