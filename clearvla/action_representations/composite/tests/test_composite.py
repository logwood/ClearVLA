from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from clearvla.action_representations.bspline import (
    BSplineActionRepresentation,
    BSplineSpec,
)
from clearvla.action_representations.composite import (
    BSplineRoleChart,
    CompositeActionPayload,
    CompositeActionRepresentation,
    CompositeActionSpec,
    ContinuousRoleSpec,
    DecodeGroupSpec,
    EndpointSpec,
    IdentityRoleChart,
    OwnerRef,
    build_hybrid_v1_contract,
)
from clearvla.action_representations.composite.hybrid_v1 import (
    HYBRID_V1_ACTION_DIM,
    HYBRID_V1_ARM_FIELD_DIM,
    HYBRID_V1_BSPLINE_CONTROL_POINTS,
    HYBRID_V1_BSPLINE_DEGREE,
    HYBRID_V1_GRIPPER_FIELD_DIM,
    HYBRID_V1_HORIZON,
    HYBRID_V1_IDENTITY,
    HYBRID_V1_STATE_DIM,
)


def _current_like() -> tuple[
    CompositeActionSpec,
    dict[str, torch.nn.Module],
]:
    arm_chart = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=12,
                num_control_points=12,
                degree=3,
                mode="hierarchical_exact",
            )
        )
    )
    sample_times = tuple(arm_chart.sample_times)
    gripper_chart = IdentityRoleChart(sample_times=sample_times, width=6)
    spec = CompositeActionSpec(
        sample_times=sample_times,
        state_dim=18,
        action_dim=7,
        continuous_roles=(
            ContinuousRoleSpec(
                role_id="arm_field",
                state_indices=tuple(range(12)),
                semantic_quantity="arm_absolute_and_adjacent_delta",
                geometry_id="euclidean_physical",
                decode_group_id="arm",
                temporal_view_kind="bspline",
                view_spec_fingerprint=arm_chart.chart_fingerprint,
            ),
            ContinuousRoleSpec(
                role_id="continuous_gripper_field",
                state_indices=tuple(range(12, 18)),
                semantic_quantity="continuous_gripper_field",
                geometry_id="euclidean_physical",
                decode_group_id="gripper",
                temporal_view_kind="identity",
                view_spec_fingerprint=gripper_chart.chart_fingerprint,
            ),
        ),
        endpoint_specs=(
            EndpointSpec(
                role_id="binary_gripper_command",
                decode_group_id="gripper",
                semantic_kind="binary_command",
                payload_kind="logits",
                payload_shape=(24, 2),
                axis_names=("action_time", "class"),
                temporal_alignment="action_horizon",
                distribution_kind="categorical",
                vocabulary_id="calvin_gripper_minus_plus_one_v1",
                usage="action_owner",
                producer_id="endpoint_head",
                action_mapping="argmax_to_minus_plus_one",
                boundary_policy="outlet_owned",
            ),
        ),
        decode_groups=(
            DecodeGroupSpec(
                "arm",
                tuple(range(6)),
                OwnerRef("codec", "physical_action_field_v1"),
            ),
            DecodeGroupSpec(
                "gripper",
                (6,),
                OwnerRef("role", "binary_gripper_command"),
            ),
        ),
        codec_id="physical_action_field_v1",
        normalizer_id="test_normalizer",
        causal_boundary_id="profile_owned_gripper_boundary",
    )
    return spec, {
        "arm_field": arm_chart,
        "continuous_gripper_field": gripper_chart,
    }


def test_current_like_roles_and_binary_sidecar_round_trip() -> None:
    spec, charts = _current_like()
    representation = CompositeActionRepresentation(spec, charts)
    state = torch.randn(3, 24, 18)
    command_logits = torch.randn(3, 24, 2)

    payload = representation.encode(
        state,
        endpoints={"binary_gripper_command": command_logits},
    )
    retained = representation.decode(payload, view="retained")
    chart = representation.decode(
        payload,
        view="chart",
        allow_stale_endpoints=True,
    )

    torch.testing.assert_close(retained.continuous_state, state, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        retained.endpoints["binary_gripper_command"],
        command_logits,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(chart.continuous_state, state, atol=1.0e-5, rtol=1.0e-5)
    # The gripper identity lane is bitwise unchanged even in chart view.
    torch.testing.assert_close(
        chart.continuous_state[..., 12:],
        state[..., 12:],
        atol=0.0,
        rtol=0.0,
    )
    assert retained.continuous_state.shape[-1] == 18
    assert retained.requires_endpoint_refresh is False
    assert retained.role_source_equal == {
        "arm_field": True,
        "continuous_gripper_field": True,
    }
    assert chart.requires_endpoint_refresh is True
    assert chart.role_lossless == {
        "arm_field": True,
        "continuous_gripper_field": True,
    }
    assert chart.role_source_equal == {
        "arm_field": False,
        "continuous_gripper_field": True,
    }
    assert command_logits.shape[-1] == 2


def test_payload_serialization_preserves_identity_and_values() -> None:
    spec, charts = _current_like()
    representation = CompositeActionRepresentation(spec, charts)
    state = torch.randn(2, 24, 18)
    logits = torch.randn(2, 24, 2)
    payload = representation.encode(
        state,
        endpoints={"binary_gripper_command": logits},
    )

    restored = CompositeActionPayload.from_state_dict(payload.as_state_dict())
    decoded = representation.decode(restored)
    torch.testing.assert_close(decoded.continuous_state, state, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        decoded.endpoints["binary_gripper_command"], logits, atol=0.0, rtol=0.0
    )


def test_spec_serialization_and_fingerprint_are_deterministic() -> None:
    spec, _ = _current_like()
    restored = CompositeActionSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.fingerprint == spec.fingerprint


def test_role_partition_must_be_exact() -> None:
    spec, _ = _current_like()
    broken = replace(
        spec.continuous_roles[1],
        state_indices=tuple(range(11, 18)),
    )
    with pytest.raises(ValueError, match="partition"):
        replace(spec, continuous_roles=(spec.continuous_roles[0], broken))


def test_lossy_bspline_requires_raw_or_explicit_opt_in() -> None:
    chart = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=4,
                num_control_points=8,
                degree=2,
                mode="compact",
            )
        )
    )
    role = ContinuousRoleSpec(
        role_id="hand_synergy",
        state_indices=(0, 1, 2, 3),
        semantic_quantity="continuous_hand_synergy",
        geometry_id="caller_declared_linear_synergy",
        decode_group_id="hand",
        temporal_view_kind="bspline",
        view_spec_fingerprint=chart.chart_fingerprint,
        retain_raw=False,
        allow_lossy_chart=False,
    )
    spec = CompositeActionSpec(
        sample_times=chart.sample_times,
        state_dim=4,
        action_dim=4,
        continuous_roles=(role,),
        endpoint_specs=(),
        decode_groups=(
            DecodeGroupSpec(
                "hand",
                (0, 1, 2, 3),
                OwnerRef("role", "hand_synergy"),
            ),
        ),
        codec_id="test_hand_codec",
        normalizer_id="test",
        causal_boundary_id="test",
    )
    with pytest.raises(ValueError, match="must retain raw"):
        CompositeActionRepresentation(spec, {"hand_synergy": chart})

    retained_spec = replace(
        spec,
        continuous_roles=(replace(role, retain_raw=True),),
    )
    representation = CompositeActionRepresentation(
        retained_spec,
        {"hand_synergy": chart},
    )
    state = torch.randn(2, 24, 4)
    payload = representation.encode(state)
    retained = representation.decode(payload, view="retained").continuous_state
    chart_value = representation.decode(payload, view="chart").continuous_state
    torch.testing.assert_close(retained, state, atol=0.0, rtol=0.0)
    assert not torch.allclose(chart_value, state)


def test_future_hand_roles_can_use_distinct_temporal_charts() -> None:
    arm = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=6,
                num_control_points=12,
                degree=3,
            )
        )
    )
    fingers = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=15,
                num_control_points=8,
                degree=2,
            )
        )
    )
    spec = CompositeActionSpec(
        sample_times=arm.sample_times,
        state_dim=21,
        action_dim=21,
        continuous_roles=(
            ContinuousRoleSpec(
                "arm",
                tuple(range(6)),
                "arm_joint_position",
                "joint_euclidean",
                "arm",
                "bspline",
                arm.chart_fingerprint,
            ),
            ContinuousRoleSpec(
                "finger_joints",
                tuple(range(6, 21)),
                "dexterous_finger_joint_position",
                "joint_euclidean",
                "hand",
                "bspline",
                fingers.chart_fingerprint,
            ),
        ),
        endpoint_specs=(
            EndpointSpec(
                "contact_mode",
                None,
                "categorical_contact_mode",
                "logits",
                (24, 5),
                ("action_time", "contact_mode"),
                "action_horizon",
                "categorical",
                "five_contact_modes_v1",
                "auxiliary",
                "endpoint_contact_head",
                "caller_owned_contact_mode_mapping",
                "observed_history_only",
            ),
        ),
        decode_groups=(
            DecodeGroupSpec("arm", tuple(range(6)), OwnerRef("role", "arm")),
            DecodeGroupSpec(
                "hand",
                tuple(range(6, 21)),
                OwnerRef("role", "finger_joints"),
            ),
        ),
        codec_id="future_dexterous_codec",
        normalizer_id="future_serialized_normalizer",
        causal_boundary_id="future_observed_hand_state",
    )
    representation = CompositeActionRepresentation(
        spec,
        {"arm": arm, "finger_joints": fingers},
    )
    state = torch.randn(2, 24, 21)
    contact_logits = torch.randn(2, 24, 5)
    decoded = representation.decode(
        representation.encode(state, endpoints={"contact_mode": contact_logits}),
        view="chart",
        allow_stale_endpoints=True,
    )
    torch.testing.assert_close(decoded.continuous_state, state, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(decoded.endpoints["contact_mode"], contact_logits)
    assert arm.chart_fingerprint != fingers.chart_fingerprint
    assert decoded.requires_endpoint_refresh is True


def test_endpoint_payload_kind_is_enforced() -> None:
    spec, charts = _current_like()
    label_spec = replace(
        spec.endpoint_specs[0],
        payload_kind="labels",
        payload_shape=(24,),
        axis_names=("action_time",),
        label_values=(0, 1),
    )
    label_contract = replace(spec, endpoint_specs=(label_spec,))
    representation = CompositeActionRepresentation(label_contract, charts)
    state = torch.randn(1, 24, 18)
    with pytest.raises(TypeError, match="integer tensor"):
        representation.encode(
            state,
            endpoints={"binary_gripper_command": torch.zeros(1, 24)},
        )
    payload = representation.encode(
        state,
        endpoints={
            "binary_gripper_command": torch.zeros(1, 24, dtype=torch.long)
        },
    )
    assert payload.endpoints[0].value.dtype == torch.long


def test_chart_grid_must_match_composite_grid() -> None:
    spec, charts = _current_like()
    wrong_grid = tuple(value * 2.0 for value in spec.sample_times)
    wrong_identity = IdentityRoleChart(sample_times=wrong_grid, width=6)
    wrong_role = replace(
        spec.continuous_roles[1],
        view_spec_fingerprint=wrong_identity.chart_fingerprint,
    )
    wrong_spec = replace(
        spec,
        continuous_roles=(spec.continuous_roles[0], wrong_role),
    )
    with pytest.raises(ValueError, match="sample grid"):
        CompositeActionRepresentation(
            wrong_spec,
            {"arm_field": charts["arm_field"], "continuous_gripper_field": wrong_identity},
        )
    with pytest.raises(ValueError, match="finite"):
        IdentityRoleChart(sample_times=(0.0, float("nan")), width=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        IdentityRoleChart(sample_times=(1.0, 0.0), width=1)


def test_supplied_times_must_match_authoritative_grid() -> None:
    spec, charts = _current_like()
    representation = CompositeActionRepresentation(spec, charts)
    state = torch.randn(1, 24, 18)
    endpoints = {"binary_gripper_command": torch.randn(1, 24, 2)}
    representation.encode(state, endpoints=endpoints, times=None)
    representation.encode(
        state,
        endpoints=endpoints,
        times=torch.tensor(spec.sample_times, dtype=torch.float32),
    )
    representation.encode(
        state,
        endpoints=endpoints,
        times=torch.tensor(spec.sample_times, dtype=torch.float64),
    )
    representation.encode(state, endpoints=endpoints, times=list(spec.sample_times))
    quantized_bfloat16 = torch.tensor(spec.sample_times, dtype=torch.bfloat16)
    representation.encode(
        state,
        endpoints=endpoints,
        times=quantized_bfloat16,
    )
    identity = charts["continuous_gripper_field"]
    assert isinstance(identity, IdentityRoleChart)
    identity.encode(state[..., 12:], times=quantized_bfloat16)

    # Python sequences are interpreted in FP64, so a nearby value cannot be
    # rounded to FP32 and silently accepted as the authoritative endpoint.
    with pytest.raises(ValueError, match="immutable composite sample grid"):
        representation.encode(
            state,
            endpoints=endpoints,
            times=[*spec.sample_times[:-1], spec.sample_times[-1] + 1.0e-8],
        )

    shifted_bfloat16 = quantized_bfloat16 * 1.04
    with pytest.raises(ValueError, match="immutable composite sample grid"):
        representation.encode(
            state,
            endpoints=endpoints,
            times=shifted_bfloat16,
        )
    with pytest.raises(ValueError, match="chart sample grid"):
        identity.encode(state[..., 12:], times=shifted_bfloat16)
    with pytest.raises(ValueError, match="finite"):
        representation.encode(
            state,
            endpoints=endpoints,
            times=torch.full((24,), float("nan")),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        representation.encode(
            state,
            endpoints=endpoints,
            times=torch.tensor(tuple(reversed(spec.sample_times))),
        )
    with pytest.raises(ValueError, match="immutable composite sample grid"):
        representation.encode(
            state,
            endpoints=endpoints,
            times=torch.linspace(0.0, 2.0, 24),
        )


def test_payload_raw_presence_and_chart_state_are_fail_closed() -> None:
    chart = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=4,
                num_control_points=8,
                degree=2,
                mode="compact",
            )
        )
    )
    role = ContinuousRoleSpec(
        role_id="hand",
        state_indices=(0, 1, 2, 3),
        semantic_quantity="hand_joint_position",
        geometry_id="joint_euclidean",
        decode_group_id="hand",
        temporal_view_kind="bspline",
        view_spec_fingerprint=chart.chart_fingerprint,
        retain_raw=True,
    )
    spec = CompositeActionSpec(
        sample_times=chart.sample_times,
        state_dim=4,
        action_dim=4,
        continuous_roles=(role,),
        endpoint_specs=(),
        decode_groups=(
            DecodeGroupSpec(
                "hand", (0, 1, 2, 3), OwnerRef("role", "hand")
            ),
        ),
        codec_id="hand_codec",
        normalizer_id="test",
        causal_boundary_id="observed_hand",
    )
    representation = CompositeActionRepresentation(spec, {"hand": chart})
    payload = representation.encode(torch.randn(1, 24, 4))
    missing_raw = replace(
        payload,
        roles=(replace(payload.roles[0], raw=None),),
    )
    with pytest.raises(ValueError, match="raw-bypass presence"):
        representation.decode(missing_raw)
    broken_chart = replace(
        payload,
        roles=(replace(payload.roles[0], chart_state={}),),
    )
    with pytest.raises(ValueError, match="B-spline payload keys"):
        representation.decode(broken_chart)

    no_raw_role = replace(role, retain_raw=False, allow_lossy_chart=True)
    no_raw_spec = replace(spec, continuous_roles=(no_raw_role,))
    no_raw_representation = CompositeActionRepresentation(no_raw_spec, {"hand": chart})
    no_raw_payload = no_raw_representation.encode(torch.randn(1, 24, 4))
    extra_raw = replace(
        no_raw_payload,
        roles=(replace(no_raw_payload.roles[0], raw=torch.randn(1, 24, 4)),),
    )
    with pytest.raises(ValueError, match="raw-bypass presence"):
        no_raw_representation.decode(extra_raw)


def test_lossy_chart_view_requires_endpoint_refresh_acknowledgement() -> None:
    chart = BSplineRoleChart(
        BSplineActionRepresentation(
            BSplineSpec.uniform(
                horizon=24,
                arm_dim=2,
                num_control_points=6,
                degree=2,
                mode="compact",
            )
        )
    )
    role = ContinuousRoleSpec(
        "arm",
        (0, 1),
        "arm_position",
        "joint_euclidean",
        "arm",
        "bspline",
        chart.chart_fingerprint,
        retain_raw=True,
    )
    endpoint = EndpointSpec(
        "contact_mode",
        None,
        "contact_mode",
        "logits",
        (24, 3),
        ("action_time", "class"),
        "action_horizon",
        "categorical",
        "contact_three_class_v1",
        "auxiliary",
        "contact_head",
        "none",
        "observed_only",
    )
    spec = CompositeActionSpec(
        sample_times=chart.sample_times,
        state_dim=2,
        action_dim=2,
        continuous_roles=(role,),
        endpoint_specs=(endpoint,),
        decode_groups=(
            DecodeGroupSpec("arm", (0, 1), OwnerRef("role", "arm")),
        ),
        codec_id="arm_codec",
        normalizer_id="test",
        causal_boundary_id="observed_arm",
    )
    representation = CompositeActionRepresentation(spec, {"arm": chart})
    payload = representation.encode(
        torch.randn(1, 24, 2),
        endpoints={"contact_mode": torch.randn(1, 24, 3)},
    )
    with pytest.raises(ValueError, match="rerun endpoint producers"):
        representation.decode(payload, view="chart")
    audit = representation.decode(
        payload,
        view="chart",
        allow_stale_endpoints=True,
    )
    assert audit.requires_endpoint_refresh is True
    assert audit.endpoint_binding == "encoded_source_state"
    assert audit.role_lossless == {"arm": False}


def test_nested_deserialization_is_strict_and_does_not_coerce_authority() -> None:
    spec, _ = _current_like()
    serialized = spec.to_dict()
    role = dict(serialized["continuous_roles"][0])
    role["allow_lossy_chart"] = "false"
    serialized["continuous_roles"] = [
        role,
        serialized["continuous_roles"][1],
    ]
    with pytest.raises(TypeError, match="allow_lossy_chart must be a boolean"):
        CompositeActionSpec.from_dict(serialized)

    serialized = spec.to_dict()
    role = dict(serialized["continuous_roles"][0])
    role["unknown_future_contract"] = 123
    serialized["continuous_roles"] = [
        role,
        serialized["continuous_roles"][1],
    ]
    with pytest.raises(ValueError, match="extra=.*unknown_future_contract"):
        CompositeActionSpec.from_dict(serialized)

    serialized = spec.to_dict()
    role = dict(serialized["continuous_roles"][0])
    role["state_indices"] = [0.9, *role["state_indices"][1:]]
    serialized["continuous_roles"] = [
        role,
        serialized["continuous_roles"][1],
    ]
    with pytest.raises(TypeError, match="state_indices entry must be an integer"):
        CompositeActionSpec.from_dict(serialized)


def test_endpoint_dtype_migration_preserves_labels_and_rejects_complex() -> None:
    spec, charts = _current_like()
    label_spec = replace(
        spec.endpoint_specs[0],
        payload_kind="labels",
        payload_shape=(24,),
        axis_names=("action_time",),
        distribution_kind="categorical",
        label_values=tuple(range(300)),
    )
    label_contract = replace(spec, endpoint_specs=(label_spec,))
    representation = CompositeActionRepresentation(label_contract, charts)
    state = torch.randn(1, 24, 18)
    labels = torch.full((1, 24), 257, dtype=torch.long)
    payload = representation.encode(
        state,
        endpoints={"binary_gripper_command": labels},
    )
    moved = payload.to(dtype=torch.bfloat16)
    assert moved.roles[0].raw is not None
    assert moved.roles[0].raw.dtype == torch.bfloat16
    assert moved.endpoints[0].value.dtype == torch.long
    assert torch.equal(moved.endpoints[0].value, labels)
    representation.decode(
        moved,
        output_dtype=torch.float32,
        allow_stale_endpoints=True,
    )
    moved_like = payload.to(torch.zeros((), dtype=torch.float64))
    assert moved_like.roles[0].raw is not None
    assert moved_like.roles[0].raw.dtype == torch.float64
    assert moved_like.endpoints[0].value.dtype == torch.long
    assert torch.equal(moved_like.endpoints[0].value, labels)
    restored = CompositeActionPayload.from_state_dict(moved_like.as_state_dict())
    representation.decode(
        restored,
        output_dtype=torch.float32,
        allow_stale_endpoints=True,
    )
    moved_like_bfloat16 = payload.to(torch.zeros((), dtype=torch.bfloat16))
    assert torch.equal(moved_like_bfloat16.endpoints[0].value, labels)
    moved_device_dtype = payload.to(device=state.device, dtype=torch.bfloat16)
    assert torch.equal(moved_device_dtype.endpoints[0].value, labels)

    with pytest.raises(TypeError, match="integer tensor"):
        representation.encode(
            state,
            endpoints={
                "binary_gripper_command": torch.ones(
                    1, 24, dtype=torch.complex64
                )
            },
        )
    with pytest.raises(ValueError, match="declared vocabulary"):
        representation.encode(
            state,
            endpoints={
                "binary_gripper_command": torch.full(
                    (1, 24), 300, dtype=torch.long
                )
            },
        )


def test_label_vocabulary_validation_has_no_unsigned_overflow_alias() -> None:
    spec, charts = _current_like()
    label_spec = replace(
        spec.endpoint_specs[0],
        payload_kind="labels",
        payload_shape=(24,),
        axis_names=("action_time",),
        distribution_kind="categorical",
        label_values=(-1, 1),
    )
    representation = CompositeActionRepresentation(
        replace(spec, endpoint_specs=(label_spec,)),
        charts,
    )
    state = torch.randn(1, 24, 18)
    signed_labels = torch.tensor([[-1, 1] * 12], dtype=torch.long)
    payload = representation.encode(
        state,
        endpoints={"binary_gripper_command": signed_labels},
    )
    restored = CompositeActionPayload.from_state_dict(payload.as_state_dict())
    decoded = representation.decode(restored)
    assert torch.equal(decoded.endpoints["binary_gripper_command"], signed_labels)

    representation.encode(
        state,
        endpoints={
            "binary_gripper_command": torch.ones(1, 24, dtype=torch.uint8)
        },
    )
    aliased_unsigned = torch.full((1, 24), 255, dtype=torch.uint8)
    with pytest.raises(ValueError, match="declared vocabulary"):
        representation.encode(
            state,
            endpoints={"binary_gripper_command": aliased_unsigned},
        )

    serialized = payload.as_state_dict()
    endpoint_state = dict(serialized["endpoints"][0])
    endpoint_state["value"] = aliased_unsigned
    serialized["endpoints"] = [endpoint_state]
    forged = CompositeActionPayload.from_state_dict(serialized)
    with pytest.raises(ValueError, match="declared vocabulary"):
        representation.decode(forged)


def test_endpoint_freshness_is_invalidated_by_continuous_dtype_changes() -> None:
    spec, charts = _current_like()
    representation = CompositeActionRepresentation(spec, charts)
    state = torch.full((1, 24, 18), 1.003, dtype=torch.float32)
    logits = torch.stack(
        (
            torch.zeros_like(state[..., 0]),
            state[..., 0] - 1.001,
        ),
        dim=-1,
    )
    payload = representation.encode(
        state,
        endpoints={"binary_gripper_command": logits},
    )

    with pytest.raises(ValueError, match="not proven identical"):
        representation.decode(payload, output_dtype=torch.bfloat16)
    decoded_cast = representation.decode(
        payload,
        output_dtype=torch.bfloat16,
        allow_stale_endpoints=True,
    )
    assert decoded_cast.requires_endpoint_refresh is True
    assert not all(decoded_cast.role_source_equal.values())

    moved = payload.to(dtype=torch.bfloat16)
    assert moved.source_state_values_preserved is False
    with pytest.raises(ValueError, match="not proven identical"):
        representation.decode(moved)
    decoded_moved = representation.decode(moved, allow_stale_endpoints=True)
    assert decoded_moved.requires_endpoint_refresh is True
    # The retained rows equal the transformed payload, but the payload records
    # that these values no longer reproduce the endpoint producer's source.
    assert all(decoded_moved.role_source_equal.values())


def test_integration_metadata_keeps_solver_role_agnostic() -> None:
    spec, charts = _current_like()
    metadata = CompositeActionRepresentation(spec, charts).integration_metadata()
    assert metadata["solver_role_awareness"] == "none"
    assert metadata["endpoint_update_semantics"] == "typed_out_of_band_at_clean_endpoint"
    assert metadata["ode_loop_safe"] is False


def test_hybrid_v1_factory_freezes_complete_state_and_role_identity() -> None:
    contract = build_hybrid_v1_contract(
        codec_id="hybrid_v1_codec",
        normalizer_id="hybrid_v1_normalizer",
        causal_boundary_id="hybrid_v1_boundary",
    )
    spec = contract.spec
    assert spec.horizon == HYBRID_V1_HORIZON
    assert spec.state_dim == HYBRID_V1_STATE_DIM
    assert spec.action_dim == HYBRID_V1_ACTION_DIM
    assert spec.continuous_roles[0].state_indices == tuple(range(12))
    assert spec.continuous_roles[0].temporal_view_kind == "bspline"
    assert spec.continuous_roles[0].retain_raw is True
    assert spec.continuous_roles[1].state_indices == tuple(range(12, 18))
    assert spec.continuous_roles[1].width == HYBRID_V1_GRIPPER_FIELD_DIM
    assert spec.continuous_roles[1].temporal_view_kind == "identity"
    assert spec.continuous_roles[1].retain_raw is True
    assert spec.decode_groups[0].final_owner == OwnerRef(
        "codec", "hybrid_v1_codec"
    )
    assert spec.decode_groups[1].final_owner == OwnerRef(
        "codec", "hybrid_v1_codec"
    )

    identity = contract.identity
    assert identity["identity"] == HYBRID_V1_IDENTITY
    assert identity["state_shape"] == [HYBRID_V1_HORIZON, HYBRID_V1_STATE_DIM]
    assert identity["action_shape"] == [HYBRID_V1_HORIZON, HYBRID_V1_ACTION_DIM]
    assert identity["typed_endpoint_sidecars_outside_ode"] is True
    assert identity["single_final_owner_per_decode_group"] is True
    assert identity["solver_role_awareness"] == "none"
    assert identity["ode_loop_safe"] is False
    assert identity["default_mainline_enabled"] is False
    arm_chart = identity["charts"]["arm_field"]
    assert arm_chart["spec"]["degree"] == HYBRID_V1_BSPLINE_DEGREE
    assert arm_chart["spec"]["num_control_points"] == (
        HYBRID_V1_BSPLINE_CONTROL_POINTS
    )
    assert arm_chart["is_lossless"] is True
    assert identity["roles"]["continuous_gripper_field"]["state_indices"] == list(
        range(HYBRID_V1_ARM_FIELD_DIM, HYBRID_V1_STATE_DIM)
    )

    state = torch.randn(2, HYBRID_V1_HORIZON, HYBRID_V1_STATE_DIM)
    decoded = contract.representation.decode(contract.representation.encode(state))
    torch.testing.assert_close(decoded.continuous_state, state, atol=0.0, rtol=0.0)


def test_hybrid_v1_factory_keeps_typed_endpoint_outside_state() -> None:
    endpoint = EndpointSpec(
        role_id="binary_gripper_command",
        decode_group_id="gripper",
        semantic_kind="binary_gripper_command",
        payload_kind="logits",
        payload_shape=(HYBRID_V1_HORIZON, 2),
        axis_names=("action_time", "class"),
        temporal_alignment="action_horizon",
        distribution_kind="categorical",
        vocabulary_id="hybrid_v1_binary_gripper_v1",
        usage="action_owner",
        producer_id="hybrid_v1_endpoint_head",
        action_mapping="argmax_to_minus_plus_one",
        boundary_policy="caller_outlet_owned",
    )
    contract = build_hybrid_v1_contract(
        codec_id="hybrid_v1_codec",
        normalizer_id="hybrid_v1_normalizer",
        causal_boundary_id="hybrid_v1_boundary",
        endpoint_specs=(endpoint,),
        gripper_final_owner=OwnerRef("role", "binary_gripper_command"),
    )
    state = torch.randn(1, HYBRID_V1_HORIZON, HYBRID_V1_STATE_DIM)
    logits = torch.randn(1, HYBRID_V1_HORIZON, 2)
    payload = contract.representation.encode(
        state,
        endpoints={"binary_gripper_command": logits},
    )
    decoded = contract.representation.decode(payload)
    torch.testing.assert_close(decoded.continuous_state, state, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        decoded.endpoints["binary_gripper_command"], logits, atol=0.0, rtol=0.0
    )
    assert decoded.requires_endpoint_refresh is False
    assert contract.spec.decode_groups[1].final_owner == OwnerRef(
        "role", "binary_gripper_command"
    )
