from __future__ import annotations

"""V39 staged mid-cut latent-contract temporal policy.

V39 keeps the V38.6.2 action-centered controlled-residual policy path, but it
adds an explicit *mid-cut contract* inside the DiT block stack.  The first
training stage can stop at the cut and train only intentionally weak heads.  The
second stage resumes from that checkpoint, runs the remaining DiT blocks, and
trains the formal policy head while preserving the mid-cut contract with a small
auxiliary loss.

The important contract is architectural: simple readout heads are attached to a
DiT midpoint, before the final decoder has enough capacity to hide shortcuts.
These heads are not meant to be the deployable policy; they are probes that make
motion/contact/future information readable at Z_mid.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from clearvla.policy.contracts import scaled_contract_view as _scaled_contract_view
from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.decoder import (
    ActionOnlyPhysicalVelocityHead,
    ConditionNeutralActionInitializer,
    HierarchicalMMDiTActionDecoder,
    OwnedHierarchicalActionBlock,
)
from clearvla.policy.evidence import (
    EvidenceMemoryBank,
    HierarchicalEvidenceWorkspace,
    HierarchicalWorkspaceManager,
    OwnedEvidenceMemoryBank,
    PreparedEvidenceMemory,
    SemanticEvidenceWorkspaceBlock,
)
from clearvla.policy.intent import (
    IndependentIntentFusion,
    IntentContractCompiler,
    PolicyConditionOrganizer,
)
from clearvla.policy.legacy.residual import (
    LayeredV37StyleResidualActionFlowDenoiser,
    V37StyleResidualActionBlock,
    V37StyleResidualActionFlowDenoiser,
    _parse_layer_pair_schedule,
)
from clearvla.policy.legacy.latent_main import (
    HierarchicalLatentActionBlock,
    HierarchicalLatentMainActionDecoder,
)
from clearvla.policy.legacy.cvae_workspace import (
    MMDiTConditionLayout,
    SemanticEvidenceWorkspace,
    WorkspaceController,
)
from clearvla.policy.legacy.cvae import (
    AdaptiveCVAEFunctionBank,
    AdaptiveCVAEMicroRefineBlock,
    AdaptiveRecurrentCVAEActionDecoder,
    AdaptiveRecurrentCVAERefinementBlock,
    LatentCVAEActionBlock,
    LatentCVAEActionDecoder,
    LatentCVAEMMDiTBlock,
    _progress_role_basis,
)
from clearvla.policy.trunk import (
    LayerContractAdapterHeads,
    LayerRoleScheduler,
    MidcutContractHeads,
    RecurrentMilestoneConsequenceCell,
    SharedLayerFlowActionProbe,
    TemporalMidcutWorldActionDiT,
    UnifiedInterventionBlock,
    _align_milestone_tokens_to_horizon,
    _rollout_tokens_to_action_horizon,
    _zeros_like_scalar,
)
from clearvla.policy.system import V39PolicySystem

from .policy import RejectableHistoryProposal
from .policy_v36_2 import ParsevalGripperTemporalFrame, PhysicalActionCodec, PhysicalActionTokenLift
from .policy_v36_3 import TransitionAwarePhysicalVelocityHead
from .world_model import BiasFreeFFN, sinusoidal_positions
from .policy_v38 import (
    CanvasPhysicalVelocityHead,
    ControlledResidualLatentDynamics,
    DenseVisualMemory,
    RolloutActionResidualHead,
    RolloutTargetCodec,
    TemporalDynamicsBoundDiTBlock,
    UnifiedCanvasSeed,
    V38PolicyConfig,
)
from .policy import TimeEmbedding

__all__ = [
    "V39PolicyConfig",
    "MidcutContractHeads",
    "LayerContractAdapterHeads",
    "OwnedEvidenceMemoryBank",
    "IntentContractCompiler",
    "PolicyConditionOrganizer",
    "HierarchicalMMDiTActionDecoder",
    "SharedLayerFlowActionProbe",
    "V37StyleResidualActionBlock",
    "V37StyleResidualActionFlowDenoiser",
    "LayerRoleScheduler",
    "UnifiedInterventionBlock",
    "RecurrentMilestoneConsequenceCell",
    "_align_milestone_tokens_to_horizon",
    "_rollout_tokens_to_action_horizon",
    "TemporalMidcutWorldActionDiT",
    "V39PolicySystem",
]
