"""Fixed, caller-owned role boundary for the opt-in ``hybrid-v1`` path.

This module only assembles the standalone temporal charts and their identity.
It does not import the mainline, choose a solver or define a training loss.
The returned representation is an outer-boundary object: callers must keep
chart/payload round-trips outside repeated ODE stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from clearvla.action_representations.bspline import (
    BSplineActionRepresentation,
    BSplineSpec,
)

from .charts import BSplineRoleChart, IdentityRoleChart
from .representation import CompositeActionRepresentation
from .spec import (
    CompositeActionSpec,
    ContinuousRoleSpec,
    DecodeGroupSpec,
    EndpointSpec,
    OwnerRef,
)

HYBRID_V1_IDENTITY = "clearvla.hybrid_v1.role_boundary_v1"
HYBRID_V1_SCHEMA_VERSION = 1
HYBRID_V1_HORIZON = 24
HYBRID_V1_STATE_DIM = 18
HYBRID_V1_ACTION_DIM = 7
HYBRID_V1_ARM_FIELD_DIM = 12
HYBRID_V1_GRIPPER_FIELD_DIM = 6
HYBRID_V1_BSPLINE_DEGREE = 3
HYBRID_V1_BSPLINE_CONTROL_POINTS = 12


@dataclass(frozen=True)
class HybridV1Contract:
    """A ready-to-register representation plus immutable integration metadata.

    ``representation`` owns only role-wise encoding/decoding.  The caller
    remains responsible for the physical codec/outlet finalizer named by the
    spec and for producing endpoint sidecars on the selected clean state.
    """

    representation: CompositeActionRepresentation
    identity: Mapping[str, Any]

    @property
    def spec(self) -> CompositeActionSpec:
        return self.representation.spec


def _identity(
    representation: CompositeActionRepresentation,
) -> dict[str, Any]:
    spec = representation.spec
    charts = dict(representation.integration_metadata()["charts"])
    return {
        "identity": HYBRID_V1_IDENTITY,
        "schema_version": HYBRID_V1_SCHEMA_VERSION,
        "state_shape": [HYBRID_V1_HORIZON, HYBRID_V1_STATE_DIM],
        "action_shape": [HYBRID_V1_HORIZON, HYBRID_V1_ACTION_DIM],
        "continuous_state_contract": "complete_24x18_field",
        "roles": {
            "arm_field": {
                "state_indices": list(range(HYBRID_V1_ARM_FIELD_DIM)),
                "chart": "fixed_cubic_bspline",
                "degree": HYBRID_V1_BSPLINE_DEGREE,
                "control_points": HYBRID_V1_BSPLINE_CONTROL_POINTS,
                "retain_raw": True,
            },
            "continuous_gripper_field": {
                "state_indices": list(
                    range(HYBRID_V1_ARM_FIELD_DIM, HYBRID_V1_STATE_DIM)
                ),
                "chart": "identity_raw",
                "retain_raw": True,
            },
        },
        "charts": charts,
        "typed_endpoint_sidecars_outside_ode": True,
        "endpoint_refresh_policy": "fail_closed_on_state_dtype_or_view_change",
        "single_final_owner_per_decode_group": True,
        "solver_role_awareness": "none",
        "ode_loop_safe": False,
        "default_mainline_enabled": False,
        "spec_fingerprint": spec.fingerprint,
    }


def build_hybrid_v1_contract(
    *,
    codec_id: str,
    normalizer_id: str,
    causal_boundary_id: str,
    outlet_id: str | None = None,
    endpoint_specs: Sequence[EndpointSpec] = (),
    gripper_final_owner: OwnerRef | None = None,
) -> HybridV1Contract:
    """Build the fixed hybrid-v1 role boundary without touching mainline code.

    The arm role is exactly the current 12-channel absolute/adjacent-difference
    field and uses a hierarchical exact cubic ``T=24, K=12`` chart with raw
    rows retained.  The six-channel continuous gripper field is identity/raw.
    Endpoint specs are caller-supplied typed sidecars and never become ODE
    coordinates.  ``gripper_final_owner`` is explicit so Pen/RDT codec-owned
    grippers and CALVIN/other endpoint-owned commands share one ABI.
    """

    arm_representation = BSplineActionRepresentation(
        BSplineSpec.uniform(
            horizon=HYBRID_V1_HORIZON,
            arm_dim=HYBRID_V1_ARM_FIELD_DIM,
            num_control_points=HYBRID_V1_BSPLINE_CONTROL_POINTS,
            degree=HYBRID_V1_BSPLINE_DEGREE,
            mode="hierarchical_exact",
        )
    )
    arm_chart = BSplineRoleChart(arm_representation)
    gripper_chart = IdentityRoleChart(
        sample_times=arm_chart.sample_times,
        width=HYBRID_V1_GRIPPER_FIELD_DIM,
    )
    owner = (
        OwnerRef("codec", codec_id)
        if gripper_final_owner is None
        else gripper_final_owner
    )
    spec = CompositeActionSpec(
        sample_times=arm_chart.sample_times,
        state_dim=HYBRID_V1_STATE_DIM,
        action_dim=HYBRID_V1_ACTION_DIM,
        continuous_roles=(
            ContinuousRoleSpec(
                role_id="arm_field",
                state_indices=tuple(range(HYBRID_V1_ARM_FIELD_DIM)),
                semantic_quantity="arm_absolute_and_adjacent_delta",
                geometry_id="current_euclidean_physical_chart",
                decode_group_id="arm",
                temporal_view_kind="bspline",
                view_spec_fingerprint=arm_chart.chart_fingerprint,
                retain_raw=True,
            ),
            ContinuousRoleSpec(
                role_id="continuous_gripper_field",
                state_indices=tuple(
                    range(HYBRID_V1_ARM_FIELD_DIM, HYBRID_V1_STATE_DIM)
                ),
                semantic_quantity="continuous_gripper_compatibility_field",
                geometry_id="current_euclidean_physical_chart",
                decode_group_id="gripper",
                temporal_view_kind="identity",
                view_spec_fingerprint=gripper_chart.chart_fingerprint,
                retain_raw=True,
            ),
        ),
        endpoint_specs=tuple(endpoint_specs),
        decode_groups=(
            DecodeGroupSpec(
                "arm",
                tuple(range(6)),
                OwnerRef("codec", codec_id),
            ),
            DecodeGroupSpec("gripper", (6,), owner),
        ),
        codec_id=codec_id,
        normalizer_id=normalizer_id,
        causal_boundary_id=causal_boundary_id,
        outlet_id=outlet_id,
    )
    representation = CompositeActionRepresentation(
        spec,
        {
            "arm_field": arm_chart,
            "continuous_gripper_field": gripper_chart,
        },
    )
    return HybridV1Contract(
        representation=representation,
        identity=_identity(representation),
    )


__all__ = [
    "HYBRID_V1_ACTION_DIM",
    "HYBRID_V1_ARM_FIELD_DIM",
    "HYBRID_V1_BSPLINE_CONTROL_POINTS",
    "HYBRID_V1_BSPLINE_DEGREE",
    "HYBRID_V1_GRIPPER_FIELD_DIM",
    "HYBRID_V1_HORIZON",
    "HYBRID_V1_IDENTITY",
    "HYBRID_V1_SCHEMA_VERSION",
    "HYBRID_V1_STATE_DIM",
    "HybridV1Contract",
    "build_hybrid_v1_contract",
]
