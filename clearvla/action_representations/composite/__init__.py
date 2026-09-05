"""Stable public API for role-wise continuous and endpoint action views."""

from .charts import (
    BSplineRoleChart,
    IdentityRoleChart,
    RolePayload,
    TemporalRoleChart,
)
from .hybrid_v1 import (
    HYBRID_V1_ACTION_DIM,
    HYBRID_V1_ARM_FIELD_DIM,
    HYBRID_V1_BSPLINE_CONTROL_POINTS,
    HYBRID_V1_BSPLINE_DEGREE,
    HYBRID_V1_GRIPPER_FIELD_DIM,
    HYBRID_V1_HORIZON,
    HYBRID_V1_IDENTITY,
    HYBRID_V1_SCHEMA_VERSION,
    HYBRID_V1_STATE_DIM,
    HybridV1Contract,
    build_hybrid_v1_contract,
)
from .representation import (
    CompositeActionPayload,
    CompositeActionRepresentation,
    DecodedCompositeState,
    EndpointPayload,
)
from .spec import (
    CompositeActionSpec,
    ContinuousRoleSpec,
    DecodeGroupSpec,
    EndpointDistributionKind,
    EndpointPayloadKind,
    EndpointSpec,
    EndpointUsage,
    OwnerKind,
    OwnerRef,
    TemporalAlignment,
    TemporalViewKind,
)

__all__ = [
    "BSplineRoleChart",
    "HYBRID_V1_ACTION_DIM",
    "HYBRID_V1_ARM_FIELD_DIM",
    "HYBRID_V1_BSPLINE_CONTROL_POINTS",
    "HYBRID_V1_BSPLINE_DEGREE",
    "HYBRID_V1_GRIPPER_FIELD_DIM",
    "HYBRID_V1_HORIZON",
    "HYBRID_V1_IDENTITY",
    "HYBRID_V1_SCHEMA_VERSION",
    "HYBRID_V1_STATE_DIM",
    "CompositeActionPayload",
    "CompositeActionRepresentation",
    "CompositeActionSpec",
    "ContinuousRoleSpec",
    "DecodedCompositeState",
    "DecodeGroupSpec",
    "EndpointPayload",
    "EndpointDistributionKind",
    "EndpointPayloadKind",
    "EndpointSpec",
    "EndpointUsage",
    "IdentityRoleChart",
    "HybridV1Contract",
    "OwnerKind",
    "OwnerRef",
    "RolePayload",
    "TemporalRoleChart",
    "TemporalAlignment",
    "TemporalViewKind",
    "build_hybrid_v1_contract",
]

