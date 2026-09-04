from __future__ import annotations

import importlib.util
import io
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from clearvla.action_representations.bspline import (
    BSplineActionRepresentation,
    BSplinePayload,
    BSplineSpec,
    NativeActionSplinePayload,
    PhysicalActionFieldBSplineAdapter,
    basis_preflight,
    bspline_basis,
    build_basis_bundle,
    not_a_knot_interpolation_knots,
    open_uniform_knots,
)

_ACTION_CODEC_PATH = Path(__file__).resolve().parents[3] / "mainline" / "model" / "action_codec.py"
_ACTION_CODEC_SPEC = importlib.util.spec_from_file_location(
    "_bspline_compat_action_codec",
    _ACTION_CODEC_PATH,
)
assert _ACTION_CODEC_SPEC is not None and _ACTION_CODEC_SPEC.loader is not None
_ACTION_CODEC_MODULE = importlib.util.module_from_spec(_ACTION_CODEC_SPEC)
sys.modules[_ACTION_CODEC_SPEC.name] = _ACTION_CODEC_MODULE
_ACTION_CODEC_SPEC.loader.exec_module(_ACTION_CODEC_MODULE)
PhysicalActionFieldCodec = _ACTION_CODEC_MODULE.PhysicalActionFieldCodec


def make_spec(
    *,
    mode: str = "hierarchical_exact",
    detail_budget: int | None = None,
    degree: int = 3,
    horizon: int = 24,
    arm_dim: int = 6,
    controls: int = 12,
) -> BSplineSpec:
    return BSplineSpec.uniform(
        horizon=horizon,
        arm_dim=arm_dim,
        num_control_points=controls,
        degree=degree,
        mode=mode,  # type: ignore[arg-type]
        detail_budget=detail_budget,
    )


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_basis_partition_endpoints_and_full_interpolation(degree: int) -> None:
    times = torch.linspace(-0.2, 0.7, 24, dtype=torch.float64)
    coarse_knots = open_uniform_knots(times, 12, degree)
    coarse = bspline_basis(times, coarse_knots, degree)
    torch.testing.assert_close(coarse.sum(dim=-1), torch.ones(24, dtype=torch.float64))
    torch.testing.assert_close(coarse[0], torch.eye(1, 12, dtype=torch.float64)[0])
    expected_last = torch.zeros(12, dtype=torch.float64)
    expected_last[-1] = 1.0
    torch.testing.assert_close(coarse[-1], expected_last)

    interpolation_knots = not_a_knot_interpolation_knots(times, degree)
    interpolation = bspline_basis(times, interpolation_knots, degree)
    assert interpolation.shape == (24, 24)
    assert int(torch.linalg.matrix_rank(interpolation)) == 24
    assert float(torch.linalg.cond(interpolation)) < 5.0
    derivative = bspline_basis(
        times,
        interpolation_knots,
        degree,
        derivative_order=1,
    )
    torch.testing.assert_close(
        derivative.sum(dim=-1),
        torch.zeros(24, dtype=torch.float64),
        atol=2e-12,
        rtol=0.0,
    )


def test_bspline_basis_small_domain_rejects_one_ulp_and_grossly_outside_queries() -> None:
    times = torch.linspace(0.0, 1.0e-6, 24, dtype=torch.float32)
    knots = open_uniform_knots(times, 12, 3)
    domain_start = knots[3]
    domain_stop = knots[-4]
    inside = torch.stack(
        (
            torch.nextafter(domain_start, domain_stop),
            torch.nextafter(domain_stop, domain_start),
        )
    )
    assert bool(torch.isfinite(bspline_basis(inside, knots, 3)).all())

    outside = (
        torch.nextafter(domain_start, torch.full_like(domain_start, -torch.inf)),
        torch.nextafter(domain_stop, torch.full_like(domain_stop, torch.inf)),
        torch.tensor(-5.0e-6, dtype=torch.float32),
        torch.tensor(-1.0e-6, dtype=torch.float32),
        torch.tensor(2.0e-6, dtype=torch.float32),
        torch.tensor(5.0e-6, dtype=torch.float32),
    )
    for query in outside:
        with pytest.raises(ValueError, match="outside the closed spline domain"):
            bspline_basis(query.reshape(1), knots, 3)

    spec = BSplineSpec.uniform(
        horizon=24,
        arm_dim=2,
        num_control_points=12,
        degree=3,
        start=0.0,
        stop=1.0e-6,
        time_unit="s",
    )
    representation = BSplineActionRepresentation(spec)
    payload = representation.encode(torch.randn(1, 24, 2))
    for query in outside:
        with pytest.raises(ValueError, match="outside the closed spline domain"):
            representation.evaluate(payload, query.reshape(1))


def test_v120_shape_structural_numbers_and_digest_are_deterministic() -> None:
    spec = make_spec()
    first = build_basis_bundle(spec)
    second = build_basis_bundle(BSplineSpec.from_dict(spec.to_dict()))
    assert first.digest == second.digest
    torch.testing.assert_close(first.coarse_q, second.coarse_q, atol=0.0, rtol=0.0)
    representation = BSplineActionRepresentation(spec)
    report = representation.basis_diagnostics()
    assert report["basis_digest"] == first.digest
    assert report["is_lossless"] is True
    assert report["coarse_condition_number"] == pytest.approx(4.6703225845, rel=1e-9)
    assert report["interpolation_condition_number"] == pytest.approx(3.8918663598, rel=1e-9)
    assert float(report["orthogonality_max_abs"]) < 1e-12
    assert float(report["dense_sample_linf_operator_norm"]) < 2.0
    assert report["numerical_preflight_passed"] is True
    assert report["runtime_float32_buffers_finite"] is True
    assert report["runtime_float32_sample_times_strictly_increasing"] is True
    assert report["runtime_float32_knot_topology_preserved"] is True
    assert report["runtime_float32_operators_finite"] is True
    assert float(report["runtime_float32_sample_evaluation_closure_max_abs"]) < 1e-5
    assert report["detail_selection_policy"] == "all_or_none"

    compact_other_width = BSplineActionRepresentation(
        BSplineSpec.uniform(
            horizon=24,
            arm_dim=3,
            num_control_points=12,
            degree=3,
            mode="compact",
        )
    )
    assert compact_other_width.basis_digest == representation.basis_digest
    assert compact_other_width.spec.fingerprint != representation.spec.fingerprint


def test_exact_round_trip_coordinates_and_sample_operator() -> None:
    torch.manual_seed(4)
    representation = BSplineActionRepresentation(make_spec())
    arm = torch.randn(5, 24, 6)
    payload = representation.encode(arm, times=torch.linspace(0.0, 1.0, 24))
    reconstructed = representation.decode(payload)
    assert float((reconstructed - arm).detach().abs().max()) < 1e-6
    assert payload.coarse.shape == (5, 12, 6)
    assert payload.detail.shape == (5, 12, 6)
    torch.testing.assert_close(
        representation.from_coordinates(representation.coordinates(payload)).coarse,
        payload.coarse,
    )
    operator = representation.evaluation_operator(spec_times(representation))
    torch.testing.assert_close(
        operator,
        torch.eye(24, dtype=torch.float64),
        atol=2e-14,
        rtol=0.0,
    )
    evaluated = representation.evaluate(payload, spec_times(representation))
    torch.testing.assert_close(evaluated, reconstructed, atol=2e-5, rtol=2e-5)


def spec_times(representation: BSplineActionRepresentation) -> tuple[float, ...]:
    return representation.spec.sample_times


def test_compact_mode_is_explicit_lossy_projection_and_idempotent() -> None:
    torch.manual_seed(5)
    representation = BSplineActionRepresentation(make_spec(mode="compact"))
    arm = torch.randn(3, 24, 6)
    first = representation.decode(representation.encode(arm))
    second = representation.decode(representation.encode(first))
    assert representation.spec.is_lossless is False
    assert representation.coordinate_rank == 12
    assert float((first - arm).square().mean().sqrt()) > 0.1
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)
    assert representation.encode(arm).detail.shape == (3, 0, 6)
    payload = representation.encode(arm)
    query = torch.linspace(0.0, 1.0, 43)
    knots = representation.coarse_knots
    assert isinstance(knots, torch.Tensor)
    basis = bspline_basis(
        query,
        knots,
        representation.spec.degree,
    )
    expected_curve = torch.einsum(
        "qk,bkd->bqd",
        basis,
        representation.coarse_control_points(payload),
    )
    torch.testing.assert_close(
        representation.evaluate(payload, query),
        expected_curve,
        atol=2e-6,
        rtol=2e-6,
    )


def test_partial_detail_budget_fails_closed_without_a_principled_ordering() -> None:
    with pytest.raises(ValueError, match="no principled coarse-to-fine ordering"):
        make_spec(mode="coarse_with_detail_budget", detail_budget=5)


def test_relative_origin_is_explicit_and_round_trips() -> None:
    torch.manual_seed(7)
    representation = BSplineActionRepresentation(make_spec())
    origin = torch.randn(4, 6)
    relative = 0.1 * torch.randn(4, 24, 6)
    arm = origin[:, None, :] + relative
    payload = representation.encode(arm, origin=origin)
    assert payload.origin is not None
    torch.testing.assert_close(payload.origin, origin)
    torch.testing.assert_close(
        representation.decode(payload),
        arm,
        atol=8e-7,
        rtol=8e-7,
    )
    controls_absolute = representation.coarse_control_points(payload)
    controls_relative = representation.coarse_control_points(payload, absolute=False)
    torch.testing.assert_close(
        controls_absolute - controls_relative,
        origin[:, None, :].expand_as(controls_absolute),
    )


def test_arbitrary_nonuniform_grid_horizon_and_arm_dimension() -> None:
    times = (0.0, 0.03, 0.09, 0.16, 0.28, 0.43, 0.61, 0.82, 1.0)
    spec = BSplineSpec(
        sample_times=times,
        arm_dim=3,
        num_control_points=6,
        degree=2,
    )
    representation = BSplineActionRepresentation(spec)
    report = representation.basis_diagnostics()
    assert report["runtime_float32_sample_times_strictly_increasing"] is True
    assert report["runtime_float32_knot_topology_preserved"] is True
    assert report["runtime_float32_operators_finite"] is True
    arm = torch.randn(2, 9, 3, dtype=torch.float64)
    reconstructed = representation.decode(representation.encode(arm, times=times))
    torch.testing.assert_close(reconstructed, arm, atol=2e-14, rtol=2e-14)


def test_clustered_nonuniform_grid_is_diagnostics_only_and_fails_runtime_preflight() -> None:
    times = torch.linspace(0.0, 1.0, 24, dtype=torch.float64) ** 3
    spec = BSplineSpec(
        sample_times=tuple(times.tolist()),
        arm_dim=6,
        num_control_points=12,
        degree=3,
    )
    report = basis_preflight(spec)
    assert report["numerical_preflight_passed"] is False
    assert float(report["coarse_condition_number"]) > 4_000.0
    assert float(report["dense_sample_linf_operator_norm"]) > 2_000.0
    assert (
        float(report["dense_velocity_linf_operator_norm_per_average_step"])
        > 4_000.0
    )
    with pytest.raises(ValueError, match="numerical preflight.*diagnostics-only"):
        build_basis_bundle(spec)
    with pytest.raises(ValueError, match="numerical preflight"):
        BSplineActionRepresentation(spec)


@pytest.mark.parametrize(
    ("start", "stop", "expected_failure"),
    [
        (1.0e8, 1.0e8 + 1.0, "runtime_float32_sample_times_strictly_increasing"),
        (0.0, 1.0e-40, "runtime_float32_operators_finite"),
        (0.0, 1.0e39, "runtime_float32_buffers_finite"),
    ],
)
def test_runtime_float32_time_chart_extremes_fail_before_representation_construction(
    start: float,
    stop: float,
    expected_failure: str,
) -> None:
    spec = BSplineSpec.uniform(
        horizon=24,
        arm_dim=2,
        num_control_points=12,
        degree=3,
        start=start,
        stop=stop,
        time_unit="s",
    )
    report = basis_preflight(spec)
    assert report["numerical_preflight_passed"] is False
    assert expected_failure in str(report["numerical_preflight_failures"])
    with pytest.raises(ValueError, match="runtime_float32"):
        build_basis_bundle(spec)
    with pytest.raises(ValueError, match="runtime_float32"):
        BSplineActionRepresentation(spec)


@pytest.mark.parametrize(
    ("horizon", "degree", "controls", "arm_dim"),
    [
        (4, 1, 2, 1),
        (5, 2, 3, 2),
        (5, 3, 4, 3),
        (8, 1, 5, 7),
        (11, 2, 6, 4),
        (16, 3, 9, 2),
        (31, 3, 15, 5),
    ],
)
def test_exact_property_sweep_across_shapes_and_degrees(
    horizon: int,
    degree: int,
    controls: int,
    arm_dim: int,
) -> None:
    generator = torch.Generator().manual_seed(1000 + horizon + degree)
    intervals = 0.05 + torch.rand(horizon - 1, generator=generator, dtype=torch.float64)
    times = torch.cat((torch.zeros(1, dtype=torch.float64), intervals.cumsum(dim=0)))
    spec = BSplineSpec(
        sample_times=tuple(times.tolist()),
        arm_dim=arm_dim,
        num_control_points=controls,
        degree=degree,
    )
    representation = BSplineActionRepresentation(spec)
    arm64 = torch.randn(3, horizon, arm_dim, generator=generator, dtype=torch.float64)
    arm32 = arm64.float()
    torch.testing.assert_close(
        representation.decode(representation.encode(arm64)),
        arm64,
        atol=3e-14,
        rtol=3e-14,
    )
    torch.testing.assert_close(
        representation.decode(representation.encode(arm32)),
        arm32,
        atol=2e-6,
        rtol=2e-6,
    )
    dense = representation.evaluate(
        representation.encode(arm32),
        torch.linspace(float(times[0]), float(times[-1]), 37),
    )
    assert bool(torch.isfinite(dense).all())


def test_derivatives_match_a_cubic_polynomial_in_physical_time() -> None:
    times = torch.linspace(2.0, 5.0, 17, dtype=torch.float64)
    spec = BSplineSpec(
        sample_times=tuple(times.tolist()),
        arm_dim=4,
        num_control_points=8,
        degree=3,
        time_unit="s",
    )
    representation = BSplineActionRepresentation(spec)
    arm = torch.stack((torch.ones_like(times), times, times.square(), times**3), dim=-1)[None]
    payload = representation.encode(arm)
    query = torch.tensor([2.11, 2.7, 3.4, 4.23, 4.89], dtype=torch.float64)
    expected_value = torch.stack((torch.ones_like(query), query, query.square(), query**3), dim=-1)[
        None
    ]
    expected_first = torch.stack(
        (torch.zeros_like(query), torch.ones_like(query), 2 * query, 3 * query.square()),
        dim=-1,
    )[None]
    expected_second = torch.stack(
        (
            torch.zeros_like(query),
            torch.zeros_like(query),
            2 * torch.ones_like(query),
            6 * query,
        ),
        dim=-1,
    )[None]
    torch.testing.assert_close(
        representation.evaluate(payload, query), expected_value, atol=2e-12, rtol=2e-12
    )
    torch.testing.assert_close(
        representation.derivative(payload, query, order=1),
        expected_first,
        atol=2e-11,
        rtol=2e-11,
    )
    torch.testing.assert_close(
        representation.derivative(payload, query, order=2),
        expected_second,
        atol=2e-10,
        rtol=2e-10,
    )


def test_backward_is_finite_and_exact_chart_preserves_identity_gradient() -> None:
    torch.manual_seed(8)
    representation = BSplineActionRepresentation(make_spec())
    arm = torch.randn(2, 24, 6, requires_grad=True)
    reconstructed = representation.decode(representation.encode(arm))
    loss = reconstructed.square().mean()
    loss.backward()
    assert arm.grad is not None
    assert bool(torch.isfinite(arm.grad).all())
    torch.testing.assert_close(
        arm.grad,
        2.0 * arm.detach() / arm.numel(),
        atol=2e-9,
        rtol=2e-5,
    )


def test_reduced_precision_input_uses_fp32_basis_algebra() -> None:
    torch.manual_seed(9)
    representation = BSplineActionRepresentation(make_spec())
    arm = torch.randn(2, 24, 6).to(torch.bfloat16)
    payload = representation.encode(arm)
    reconstructed = representation.decode(payload)
    assert payload.coarse.dtype == torch.float32
    assert payload.detail.dtype == torch.float32
    assert reconstructed.dtype == torch.float32
    torch.testing.assert_close(reconstructed, arm.float(), atol=1e-6, rtol=1e-6)


def test_module_dtype_cast_cannot_reduce_fixed_basis_precision() -> None:
    torch.manual_seed(91)
    representation = BSplineActionRepresentation(make_spec()).to(dtype=torch.bfloat16)
    assert all(buffer.dtype == torch.float32 for buffer in representation.buffers())
    assert list(representation.parameters()) == []
    assert representation.state_dict() == {}
    arm = torch.randn(2, 24, 6)
    reconstructed = representation.decode(representation.encode(arm))
    torch.testing.assert_close(reconstructed, arm, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_autocast_round_trip_and_backward() -> None:
    torch.manual_seed(92)
    representation = BSplineActionRepresentation(make_spec()).cuda()
    arm = torch.randn(2, 24, 6, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstructed = representation.decode(representation.encode(arm))
        dense = representation.evaluate(
            representation.encode(arm),
            torch.linspace(0.0, 1.0, 51, device="cuda"),
        )
        loss = reconstructed.square().mean() + 0.01 * dense.square().mean()
    loss.backward()
    assert reconstructed.dtype == torch.float32
    assert float((reconstructed - arm).detach().abs().max()) < 1e-6
    assert arm.grad is not None and bool(torch.isfinite(arm.grad).all())


def test_payload_plain_state_serialization_and_identity_rejection() -> None:
    representation = BSplineActionRepresentation(make_spec())
    payload = representation.encode(torch.randn(2, 24, 6))
    stream = io.BytesIO()
    torch.save(payload.as_state_dict(), stream)
    stream.seek(0)
    loaded = BSplinePayload.from_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(representation.decode(loaded), representation.decode(payload))

    other = BSplineActionRepresentation(make_spec(degree=2))
    with pytest.raises(ValueError, match="fingerprint"):
        other.decode(payload)
    corrupt = replace(payload, basis_digest="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        representation.decode(corrupt)


def test_invalid_times_queries_and_modes_fail_closed() -> None:
    representation = BSplineActionRepresentation(make_spec())
    arm = torch.randn(1, 24, 6)
    wrong_times = torch.linspace(0.0, 2.0, 24)
    with pytest.raises(ValueError, match="immutable"):
        representation.encode(arm, times=wrong_times)
    payload = representation.encode(arm)
    with pytest.raises(ValueError, match="outside"):
        representation.evaluate(payload, torch.tensor([-0.01, 0.5]))
    with pytest.raises(ValueError, match="detail_budget"):
        make_spec(mode="compact", detail_budget=2)
    with pytest.raises(ValueError, match="no principled coarse-to-fine ordering"):
        make_spec(mode="coarse_with_detail_budget", detail_budget=12)
    with pytest.raises(ValueError, match="smaller than the horizon"):
        make_spec(controls=24)
    bad_config = make_spec().to_dict()
    bad_config["typo_controls"] = 12
    with pytest.raises(ValueError, match="unknown"):
        BSplineSpec.from_dict(bad_config)


def test_diagnostics_report_critical_slices_and_dense_extrema() -> None:
    torch.manual_seed(10)
    representation = BSplineActionRepresentation(make_spec())
    arm = torch.randn(2, 24, 6)
    report = representation.diagnostics(arm, dense_samples=97)
    for key in (
        "rmse_full",
        "rmse_first",
        "rmse_first4",
        "rmse_tail",
        "max_abs",
        "dense_overshoot_max_abs",
        "dense_velocity_max_abs",
        "dense_acceleration_max_abs",
    ):
        assert key in report
        value = report[key]
        assert isinstance(value, torch.Tensor)
        assert bool(torch.isfinite(value))
    assert float(report["max_abs"]) < 1e-6


def test_current_physical_codec_adapter_is_exact_and_gripper_is_untouched() -> None:
    torch.manual_seed(11)
    base = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    representation = BSplineActionRepresentation(make_spec())
    adapter = PhysicalActionFieldBSplineAdapter(representation, base)
    action = torch.randn(3, 24, 7)
    state = torch.randn(3, 7)

    expected_field = base.encode(action, state)
    actual_field = adapter.encode(action, state)
    torch.testing.assert_close(actual_field, expected_field, atol=2e-6, rtol=2e-6)
    # The six legacy gripper coordinates must be bitwise identical.
    torch.testing.assert_close(
        actual_field[..., 2 * base.arm_dim :],
        expected_field[..., 2 * base.arm_dim :],
        atol=0.0,
        rtol=0.0,
    )

    raw_field = torch.randn(3, 24, 18)
    expected_native = base.decode(raw_field, state)
    actual_native = adapter.decode(raw_field, state)
    torch.testing.assert_close(
        actual_native[..., :6], expected_native[..., :6], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(actual_native[..., 6:], expected_native[..., 6:], atol=0.0, rtol=0.0)
    metadata = adapter.integration_metadata()
    assert metadata["basis_digest"] == representation.basis_digest
    assert metadata["physical_codec_type"] == "PhysicalActionFieldCodec"

    payload = adapter.encode_representation(action)
    torch.testing.assert_close(
        adapter.to_physical(payload, state),
        expected_field,
        atol=2e-6,
        rtol=2e-6,
    )
    recovered_payload = adapter.from_physical(expected_field, state)
    torch.testing.assert_close(
        adapter.decode_representation(recovered_payload),
        action,
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    ("arm_flow_mode", "expected_relative_command_direct"),
    [
        ("legacy_independent", False),
        ("relative_command_direct", True),
    ],
)
def test_current_codec_complete_facade_forwards_explicit_gripper_boundary(
    arm_flow_mode: str,
    expected_relative_command_direct: bool,
) -> None:
    torch.manual_seed(111)
    base = PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        arm_flow_mode=arm_flow_mode,
    )
    representation = BSplineActionRepresentation(make_spec())
    adapter = PhysicalActionFieldBSplineAdapter(representation, base)
    action = torch.randn(2, 24, 7)
    state = torch.randn(2, 7)
    codec_gripper_boundary = torch.tensor([[-3.0], [4.0]], dtype=action.dtype)

    assert adapter.uses_relative_command_direct is base.uses_relative_command_direct
    assert adapter.uses_relative_command_direct is expected_relative_command_direct
    assert (
        adapter.integration_metadata()["codec_gripper_boundary_semantics"]
        == "explicit_transparent_forwarding_only"
    )
    torch.testing.assert_close(
        adapter.arm_motion_magnitude(action, state),
        base.arm_motion_magnitude(action, state),
        atol=0.0,
        rtol=0.0,
    )

    expected_field = base.encode(
        action,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    actual_field = adapter.encode(
        action,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    torch.testing.assert_close(actual_field, expected_field, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual_field[..., 2 * base.arm_dim :],
        expected_field[..., 2 * base.arm_dim :],
        atol=0.0,
        rtol=0.0,
    )
    default_boundary_field = base.encode(action, state)
    assert not torch.equal(
        expected_field[..., 2 * base.arm_dim :],
        default_boundary_field[..., 2 * base.arm_dim :],
    )

    raw_field = torch.randn(2, 24, 18)
    expected_native = base.decode(
        raw_field,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    actual_native = adapter.decode(
        raw_field,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    torch.testing.assert_close(actual_native, expected_native, atol=2e-6, rtol=2e-6)
    expected_branches = base.gripper_decode_branches(
        raw_field,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    actual_branches = adapter.gripper_decode_branches(
        raw_field,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    for actual, expected in zip(actual_branches, expected_branches, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        adapter.delta_consistency(
            raw_field,
            state,
            expected_native,
            codec_gripper_boundary=codec_gripper_boundary,
        ),
        base.delta_consistency(
            raw_field,
            state,
            expected_native,
            codec_gripper_boundary=codec_gripper_boundary,
        ),
        atol=0.0,
        rtol=0.0,
    )

    payload = adapter.encode_representation(action)
    assert payload.arm.origin is None
    torch.testing.assert_close(
        adapter.to_physical(
            payload,
            state,
            codec_gripper_boundary=codec_gripper_boundary,
        ),
        expected_field,
        atol=2e-6,
        rtol=2e-6,
    )
    recovered = adapter.from_physical(
        raw_field,
        state,
        codec_gripper_boundary=codec_gripper_boundary,
    )
    assert recovered.arm.origin is None
    torch.testing.assert_close(
        adapter.decode_representation(recovered),
        expected_native,
        atol=2e-6,
        rtol=2e-6,
    )


def test_adapter_payload_uses_only_an_explicit_affine_origin() -> None:
    torch.manual_seed(12)
    base = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    representation = BSplineActionRepresentation(make_spec())
    adapter = PhysicalActionFieldBSplineAdapter(representation, base)
    action = torch.randn(2, 24, 7)
    origin = torch.randn(2, 6)
    payload = adapter.encode_representation(action, origin=origin)
    assert payload.arm.origin is not None
    torch.testing.assert_close(payload.arm.origin, origin)
    reconstructed = adapter.decode_representation(payload)
    torch.testing.assert_close(reconstructed, action, atol=2e-6, rtol=2e-6)
    metadata = adapter.integration_metadata()
    assert metadata["origin_semantics"] == "explicit_affine_translation_only"
    assert metadata["bspine0_gate_b_compatible"] is False
    assert metadata["repeated_bottom_call_safe"] is False

    stream = io.BytesIO()
    torch.save(payload.as_state_dict(), stream)
    stream.seek(0)
    restored = NativeActionSplinePayload.from_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(
        adapter.decode_representation(restored),
        reconstructed,
    )


def test_compact_adapter_changes_only_arm_semantics() -> None:
    torch.manual_seed(13)
    base = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    representation = BSplineActionRepresentation(make_spec(mode="compact"))
    with pytest.raises(ValueError, match="allow_experimental_lossy_projection"):
        PhysicalActionFieldBSplineAdapter(representation, base)
    adapter = PhysicalActionFieldBSplineAdapter(
        representation,
        base,
        allow_experimental_lossy_projection=True,
    )
    action = torch.randn(2, 24, 7)
    state = torch.randn(2, 7)
    projected = adapter.project_native(action)
    assert float((projected[..., :6] - action[..., :6]).abs().max()) > 0.1
    torch.testing.assert_close(projected[..., 6:], action[..., 6:], atol=0.0, rtol=0.0)
    expected_gripper = base.encode(action, state)[..., 12:]
    actual_gripper = adapter.encode(action, state)[..., 12:]
    torch.testing.assert_close(actual_gripper, expected_gripper, atol=0.0, rtol=0.0)


def test_codec_facade_rejects_an_incomplete_forwarding_contract_at_construction() -> None:
    class EncodeDecodeOnlyCodec:
        action_dim = 7
        horizon = 24
        arm_dim = 6
        physical_dim = 18
        gripper_field_dim = 6
        decode_delta_blend = 0.25

        def encode(self, action: torch.Tensor, action_state: torch.Tensor) -> torch.Tensor:
            return action

        def decode(self, field: torch.Tensor, action_state: torch.Tensor) -> torch.Tensor:
            return field[..., :7]

    representation = BSplineActionRepresentation(make_spec())
    with pytest.raises(TypeError, match="uses_relative_command_direct"):
        PhysicalActionFieldBSplineAdapter(
            representation,
            EncodeDecodeOnlyCodec(),  # type: ignore[arg-type]
        )

    base = PhysicalActionFieldCodec(action_dim=7, horizon=24)

    class MissingArmMotionCodec:
        def __getattr__(self, name: str) -> object:
            if name == "arm_motion_magnitude":
                raise AttributeError(name)
            return getattr(base, name)

    with pytest.raises(TypeError, match="arm_motion_magnitude"):
        PhysicalActionFieldBSplineAdapter(
            representation,
            MissingArmMotionCodec(),  # type: ignore[arg-type]
        )

    class LegacyBoundarySignatureCodec:
        def encode(self, action: torch.Tensor, action_state: torch.Tensor) -> torch.Tensor:
            return base.encode(action, action_state)

        def __getattr__(self, name: str) -> object:
            return getattr(base, name)

    with pytest.raises(TypeError, match="encode.*keyword-only codec_gripper_boundary"):
        PhysicalActionFieldBSplineAdapter(
            representation,
            LegacyBoundarySignatureCodec(),  # type: ignore[arg-type]
        )
