from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from clearvla.mainline.model import types as model_types
from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.model.components import OutletAdapter
from clearvla.mainline.model.dynamics import ObjectFutureDynamicsCompiler
from clearvla.mainline.model.types import (
    ObjectWorldBelief,
    PhysicalActionCondition,
    PhysicalActionSequenceCondition,
)


def _codec() -> PhysicalActionFieldCodec:
    return PhysicalActionFieldCodec(
        action_dim=7,
        horizon=24,
        gripper_field_dim=6,
        decode_delta_blend=0.25,
    )


def _absolute_condition(action: torch.Tensor) -> PhysicalActionSequenceCondition:
    return PhysicalActionSequenceCondition.from_absolute_action(
        action,
        torch.zeros(action.shape[0], action.shape[-1], device=action.device),
        outlet_profile="pen_7d_continuous_v1",
        normalizer_offset=torch.zeros(1, action.shape[-1], device=action.device),
        normalizer_scale=torch.ones(1, action.shape[-1], device=action.device),
    )


def _belief() -> ObjectWorldBelief:
    torch.manual_seed(913)
    batch, objects, cameras = 1, 4, 2
    validity = torch.ones(batch, objects, 1, dtype=torch.float32)
    camera_validity = torch.ones(
        batch,
        objects,
        cameras,
        1,
        dtype=torch.float32,
    )
    return ObjectWorldBelief(
        content=torch.randn(batch, objects, 16),
        semantic=torch.randn(batch, objects, 8),
        appearance=torch.randn(batch, objects, 8),
        geometry=torch.randn(batch, objects, 8),
        camera_coordinates=torch.randn(batch, objects, cameras, 2),
        camera_transport_prior=torch.randn(batch, objects, cameras, 2),
        camera_support=torch.ones_like(camera_validity),
        camera_validity=camera_validity,
        log_camera_validity=torch.zeros_like(camera_validity),
        validity=validity,
        log_validity=torch.zeros_like(validity),
    )


def _sequence_dynamics() -> ObjectFutureDynamicsCompiler:
    torch.manual_seed(914)
    dynamics = ObjectFutureDynamicsCompiler(
        hidden=32,
        content_dim=16,
        route_dim=8,
        action_dim=7,
        heads=4,
        action_condition_mode="sequence_prefix_v1",
    )
    # Fresh W output heads are intentionally zero in the production model.
    # Make this causal test non-degenerate so equality cannot pass vacuously.
    with torch.no_grad():
        torch.nn.init.normal_(dynamics.delta_head.weight, std=0.10)
        torch.nn.init.normal_(dynamics.transport_head.weight, std=0.10)
        torch.nn.init.normal_(dynamics.covariance_head.weight, std=0.10)
    return dynamics


def test_outlet_factory_reuses_verified_normalizer_identity_without_device_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = OutletAdapter(
        _codec(),
        selection="calvin_7d_binary_v1",
        world_action_condition_mode="sequence_prefix_v1",
    )
    adapter.configure_action_normalizer(
        SimpleNamespace(
            offset=torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.5, 0.25, 0.6]]),
            scale=torch.tensor([[1.7, 0.8, 2.1, 1.3, 0.9, 1.4, 0.75]]),
        )
    )
    expected_identity = adapter._normalizer_identity()
    original = model_types.physical_action_normalizer_fingerprint
    calls = 0

    def counted_fingerprint(offset: torch.Tensor, scale: torch.Tensor) -> str:
        nonlocal calls
        calls += 1
        return original(offset, scale)

    monkeypatch.setattr(
        model_types,
        "physical_action_normalizer_fingerprint",
        counted_fingerprint,
    )
    action = torch.randn(2, 24, 7)
    condition = adapter.world_condition_from_horizon_action(
        action,
        torch.randn(2, 7),
    )
    assert isinstance(condition, PhysicalActionSequenceCondition)
    assert condition.normalizer_fingerprint == expected_identity
    assert calls == 0

    condition.assert_exact_contract()
    assert calls == 1


def _world_output(
    dynamics: ObjectFutureDynamicsCompiler,
    belief: ObjectWorldBelief,
    action: torch.Tensor,
):
    _, w1_state, _ = dynamics.forward_w1(
        facts=belief,
        action=_absolute_condition(action),
    )
    return dynamics.forward_w2(facts=belief, w1_state=w1_state)[0]


def _assert_prefix_equal(left, right, intervals: int) -> None:
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        torch.testing.assert_close(
            getattr(left, name)[:, :intervals],
            getattr(right, name)[:, :intervals],
            atol=0.0,
            rtol=0.0,
        )


def test_relative_sequence_adapter_owns_all_24_rows_and_keeps_gradients() -> None:
    offset = torch.tensor([[0.40, -0.20, 0.10, 0.30, -0.50, 0.25, 0.60]])
    adapter = OutletAdapter(
        _codec(),
        selection="calvin_7d_binary_v1",
        world_action_condition_mode="sequence_prefix_v1",
    )
    adapter.configure_action_normalizer(
        SimpleNamespace(offset=offset, scale=torch.ones_like(offset))
    )
    command = torch.linspace(-0.03, 0.04, 24).reshape(1, 24, 1).expand(2, -1, 6)
    source = offset[:, None].expand(2, 24, -1).clone()
    source[..., :6] += command
    source[..., -1] = torch.linspace(-0.8, 0.9, 24)
    source.requires_grad_()
    current = offset.expand(2, -1).clone()
    current[..., -1] = -0.8

    condition = adapter.world_condition_from_action_prediction(source, current)
    assert isinstance(condition, PhysicalActionSequenceCondition)
    assert condition.source_action is source
    assert not hasattr(condition, "interval_action")
    torch.testing.assert_close(
        condition.canonical_value[..., :6],
        torch.cumsum(command, dim=1),
    )
    torch.testing.assert_close(condition.canonical_delta[..., :6], command)
    torch.testing.assert_close(condition.canonical_value[..., -1], source[..., -1])
    condition.assert_exact_contract()
    corrupted_source = replace(
        condition,
        source_action=condition.source_action.detach().clone(),
    )
    corrupted_source.source_action[..., 0].add_(0.25)
    with pytest.raises(ValueError, match="source/delta diverged"):
        corrupted_source.assert_exact_contract()
    condition.canonical_value.square().mean().backward()
    assert source.grad is not None
    assert torch.count_nonzero(source.grad[..., :6]) > 0


def test_sequence_identity_closes_the_old_interval_projection_nullspace() -> None:
    base = torch.zeros(1, 24, 7)
    first_three = base.clone()
    first_three[:, :3, 0] = torch.tensor((0.4, -0.2, 0.1))
    current = torch.zeros(1, 7)

    old_base = PhysicalActionCondition.from_horizon_action(base, current)
    old_first_three = PhysicalActionCondition.from_horizon_action(
        first_three,
        current,
    )
    torch.testing.assert_close(
        old_base.fingerprint,
        old_first_three.fingerprint,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(
        _absolute_condition(base).fingerprint,
        _absolute_condition(first_three).fingerprint,
    )
    shifted_boundary = PhysicalActionSequenceCondition.from_absolute_action(
        base,
        torch.ones_like(current),
        outlet_profile="pen_7d_continuous_v1",
        normalizer_offset=torch.zeros_like(current),
        normalizer_scale=torch.ones_like(current),
    )
    assert not torch.equal(
        _absolute_condition(base).fingerprint,
        shifted_boundary.fingerprint,
    )

    ordered = base.clone()
    reordered = base.clone()
    ordered[:, 3, 1], ordered[:, 4, 1] = 0.5, -0.5
    reordered[:, 3, 1], reordered[:, 4, 1] = -0.5, 0.5
    old_ordered = PhysicalActionCondition.from_horizon_action(ordered, current)
    old_reordered = PhysicalActionCondition.from_horizon_action(reordered, current)
    torch.testing.assert_close(
        old_ordered.fingerprint,
        old_reordered.fingerprint,
        atol=0.0,
        rtol=0.0,
    )
    assert not torch.equal(
        _absolute_condition(ordered).fingerprint,
        _absolute_condition(reordered).fingerprint,
    )


def test_sequence_contract_rejects_detached_source_and_noninteger_prefix() -> None:
    condition = _absolute_condition(torch.zeros(1, 24, 7))
    detached_source = replace(
        condition,
        source_action=torch.ones_like(condition.source_action),
    )
    detached_source.validate(action_dim=7)
    with pytest.raises(ValueError, match="source/value diverged"):
        detached_source.assert_exact_contract()
    with pytest.raises(ValueError, match="complete 24-row prefix"):
        replace(condition, known_prefix_end=24.5).validate(action_dim=7)
    with pytest.raises(TypeError, match="normalizer metadata must remain FP32"):
        replace(
            condition,
            normalizer_offset=condition.normalizer_offset.to(torch.bfloat16),
        ).validate(action_dim=7)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("profile", ("pen_7d_continuous_v1", "calvin_7d_binary_v1"))
def test_sequence_normalizer_identity_survives_amp_without_changing_values(
    dtype: torch.dtype,
    profile: str,
) -> None:
    offset = torch.tensor([[0.40, -0.20, 0.10, 0.30, -0.50, 0.25, 0.60]])
    scale = torch.tensor([[0.70, 1.30, 0.90, 2.10, 0.40, 3.00, 1.70]])
    adapter = OutletAdapter(
        _codec(),
        selection=profile,
        world_action_condition_mode="sequence_prefix_v1",
    )
    adapter.configure_action_normalizer(SimpleNamespace(offset=offset, scale=scale))
    source = torch.linspace(-0.7, 1.1, 2 * 24 * 7).reshape(2, 24, 7).to(dtype)
    source.requires_grad_()
    current = torch.linspace(-0.3, 0.8, 14).reshape(2, 7)
    condition = adapter.world_condition_from_action_prediction(source, current)
    full_precision = adapter.world_condition_from_action_prediction(source.float(), current)
    assert condition.source_action is source
    assert condition.metadata_identity == full_precision.metadata_identity
    assert condition.normalizer_offset.dtype == torch.float32
    assert condition.normalizer_scale.dtype == torch.float32
    torch.testing.assert_close(
        condition.normalizer_offset,
        offset[:, None].expand(2, -1, -1),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        condition.normalizer_scale,
        scale[:, None].expand(2, -1, -1),
        atol=0.0,
        rtol=0.0,
    )

    # Numerical views must retain the pre-fix producer-dtype arithmetic,
    # including offset rounding and the low-precision cumulative sum.
    if adapter.is_relative_command:
        arm_delta = source[..., :6] - offset.to(dtype)[:, None, :6]
        gripper = source[..., 6:]
        gripper_boundary = torch.cat((current.to(dtype)[:, None, 6:], gripper[:, :-1]), dim=1)
        expected_value = torch.cat((torch.cumsum(arm_delta, dim=1), gripper), dim=-1)
        expected_delta = torch.cat((arm_delta, gripper - gripper_boundary), dim=-1)
    else:
        expected_value = source
        boundary = torch.cat((current.to(dtype)[:, None], source[:, :-1]), dim=1)
        expected_delta = source - boundary
    torch.testing.assert_close(condition.canonical_value, expected_value, atol=0.0, rtol=0.0)
    torch.testing.assert_close(condition.canonical_delta, expected_delta, atol=0.0, rtol=0.0)
    adapter.validate_world_condition(condition)
    condition.assert_exact_contract()
    with pytest.raises(ValueError, match="normalizer identity is inconsistent"):
        replace(condition, normalizer_scale=condition.normalizer_scale + 0.25).assert_exact_contract()
    corrupted_delta = condition.canonical_delta.detach().clone()
    corrupted_delta[..., -1].add_(0.25)
    with pytest.raises(ValueError, match="delta reconstruction failed"):
        replace(condition, canonical_delta=corrupted_delta).assert_exact_contract()
    condition.physical_fingerprint.float().square().mean().backward()
    assert source.grad is not None
    assert bool(torch.isfinite(source.grad).all())
    assert torch.count_nonzero(source.grad) > 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_sequence_binary_normalizer_mapping_keeps_legacy_values(dtype: torch.dtype) -> None:
    offset = torch.tensor([[0.40, -0.20, 0.10, 0.30, -0.50, 0.25, 0.60]])
    scale = torch.tensor([[0.70, 1.30, 0.90, 2.10, 0.40, 3.00, 1.70]])
    normalizer = SimpleNamespace(offset=offset, scale=scale)
    sequence = OutletAdapter(
        _codec(),
        selection="calvin_7d_binary_v1",
        world_action_condition_mode="sequence_prefix_v1",
    )
    legacy = OutletAdapter(_codec(), selection="calvin_7d_binary_v1")
    sequence.configure_action_normalizer(normalizer)
    legacy.configure_action_normalizer(normalizer)
    deployed = torch.linspace(-0.7, 1.1, 2 * 24 * 7).reshape(2, 24, 7).to(dtype)
    command = (torch.arange(48).reshape(2, 24) % 2) * 2 - 1
    mapped = sequence.world_condition_action_from_deployed(deployed, command=command)
    legacy_mapped = legacy.world_condition_action_from_deployed(deployed, command=command)
    torch.testing.assert_close(mapped, legacy_mapped, atol=0.0, rtol=0.0)
    torch.testing.assert_close(mapped[..., :6], deployed[..., :6], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        mapped[..., -1],
        command.to(dtype) * scale.to(dtype)[..., -1:] + offset.to(dtype)[..., -1:],
        atol=0.0,
        rtol=0.0,
    )
    condition = sequence.world_condition_from_horizon_action(mapped, torch.zeros(2, 7))
    condition.assert_exact_contract()


def test_sequence_w_is_prefix_causal_through_all_final_outputs() -> None:
    dynamics = _sequence_dynamics()
    belief = _belief()
    torch.manual_seed(915)
    action = torch.randn(1, 24, 7)
    baseline = _world_output(dynamics, belief, action)

    after_eight = action.clone()
    after_eight[:, 8:] += torch.randn_like(after_eight[:, 8:])
    changed_after_eight = _world_output(dynamics, belief, after_eight)
    _assert_prefix_equal(baseline, changed_after_eight, 1)
    assert not torch.equal(
        baseline.semantic_delta[:, 1:],
        changed_after_eight.semantic_delta[:, 1:],
    )

    after_sixteen = action.clone()
    after_sixteen[:, 16:] += torch.randn_like(after_sixteen[:, 16:])
    changed_after_sixteen = _world_output(dynamics, belief, after_sixteen)
    _assert_prefix_equal(baseline, changed_after_sixteen, 2)
    assert not torch.equal(
        baseline.transport_mean[:, 2:],
        changed_after_sixteen.transport_mean[:, 2:],
    )

    # Same old first-interval mean, different order: the new W must respond.
    ordered = torch.zeros_like(action)
    reordered = torch.zeros_like(action)
    ordered[:, 3, 0], ordered[:, 4, 0] = 0.75, -0.75
    reordered[:, 3, 0], reordered[:, 4, 0] = -0.75, 0.75
    ordered_world = _world_output(dynamics, belief, ordered)
    reordered_world = _world_output(dynamics, belief, reordered)
    assert not torch.equal(
        ordered_world.semantic_delta[:, :1],
        reordered_world.semantic_delta[:, :1],
    )
