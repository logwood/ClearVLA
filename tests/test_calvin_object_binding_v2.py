"""Actual bridge/coarse-reader contracts for the opt-in post-read residual."""
from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.model.calvin_object_binding import (
    CALVIN_OBJECT_BINDING_INTENT_V2,
    CalvinObjectBindingBridge,
)
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.sampling import sample_action
from tests.test_mainline_policy import _batch, _calvin_binary_config, _config
from tests.test_mainline_top import _context, _top


@pytest.fixture
def seam():
    torch.manual_seed(617)
    top = _top().eval()
    context = _context(top, batch=1)
    bridge = CalvinObjectBindingBridge(
        hidden=32, route_dim=8, readout_mode="mass_gated_residual_v2"
    ).eval()
    return top, context, bridge


def bind(bridge, context):
    return bridge(
        protected_goal=context.intent.protected_goal_set,
        history_tokens=context.intent.history_tokens,
        object_tokens=context.intent.object_tokens,
        facts=context.facts,
    )[0]


def bound_dock(context, result):
    return replace(
        context.intent.action_dock(),
        selected_object_context=result.selected_context,
        object_binding_pointer=result.pointer,
        object_binding_readout_strength=result.readout_strength,
    )


def test_v2_zero_gate_exactly_restores_public_k_read(seam) -> None:
    top, context, bridge = seam
    with torch.no_grad():
        bridge.context_gate.zero_()
        result = bind(bridge, context)
        baseline = top.coarse_action(context.intent.action_dock()).action_prediction
        corrected = top.coarse_action(bound_dock(context, result)).action_prediction
    assert torch.equal(baseline, corrected)
    assert torch.count_nonzero(result.readout_strength) == 0


def test_v2_all_invalid_slots_have_exact_original_read_fallback(seam) -> None:
    top, context, bridge = seam
    context = replace(context, facts=replace(context.facts, validity=torch.zeros_like(context.facts.validity)))
    with torch.no_grad():
        result = bind(bridge, context)
        baseline = top.coarse_action(context.intent.action_dock()).action_prediction
        corrected = top.coarse_action(bound_dock(context, result)).action_prediction
    assert torch.equal(result.pointer, torch.tensor([[0., 0., 0., 0., 1.]]))
    assert torch.equal(baseline, corrected)
    assert torch.isfinite(result.selected_context).all()


def test_v2_null_mass_attenuates_actual_coarse_innovation(seam) -> None:
    top, context, bridge = seam
    norms = []
    with torch.no_grad():
        baseline = top.coarse_action(context.intent.action_dock()).action_prediction
        for null_logit in (-5., 10.):
            with mock.patch.object(
                bridge.null_scorer, "forward",
                side_effect=lambda value, n=null_logit: value.new_full((value.shape[0], 1), n),
            ):
                result = bind(bridge, context)
            corrected = top.coarse_action(bound_dock(context, result)).action_prediction
            norms.append((corrected - baseline).square().mean().sqrt())
    assert norms[0] > 1e-5
    assert norms[1] < norms[0] * 0.01


def test_v2_action_loss_reaches_query_conditioned_null_scorer(seam) -> None:
    top, context, bridge = seam
    result = bind(bridge, context)
    prediction = top.coarse_action(bound_dock(context, result)).action_prediction
    prediction.square().mean().backward()
    gradient = bridge.null_scorer.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-6


def test_v2_is_equivariant_to_object_slot_permutation(seam) -> None:
    top, context, bridge = seam
    index = torch.tensor([2, 0, 3, 1])
    permuted = replace(context, facts=context.facts.permute(index), intent=context.intent.permute(index))
    with torch.no_grad():
        left, right = bind(bridge, context), bind(bridge, permuted)
        left_action = top.coarse_action(bound_dock(context, left)).action_prediction
        right_action = top.coarse_action(bound_dock(permuted, right)).action_prediction
    torch.testing.assert_close(left.pointer[:, index], right.pointer[:, :4])
    torch.testing.assert_close(left.pointer[:, -1], right.pointer[:, -1])
    torch.testing.assert_close(left.selected_context, right.selected_context)
    torch.testing.assert_close(left.readout_strength, right.readout_strength)
    torch.testing.assert_close(left_action, right_action)


def test_v1_retains_legacy_selected_context_and_parameter_names(seam) -> None:
    _, context, v2 = seam
    legacy = CalvinObjectBindingBridge(hidden=32, route_dim=8).eval()
    legacy.load_state_dict(v2.state_dict(), strict=True)
    left, right = bind(legacy, context), bind(v2, context)
    assert left.readout_strength is None
    torch.testing.assert_close(left.pointer, right.pointer)
    torch.testing.assert_close(left.selected_context, v2.context_gate.tanh() * right.selected_context)
    assert legacy.state_dict().keys() == v2.state_dict().keys()


def test_v2_has_explicit_component_and_config_identity() -> None:
    legacy = load_config("configs/mainline/calvin_object_binding_formal_v1.json")
    candidate = load_config("configs/mainline/calvin_object_binding_residual_v2.json")
    assert ComponentSelection.from_config(candidate).intent == CALVIN_OBJECT_BINDING_INTENT_V2
    assert candidate.digest() != legacy.digest()
    with pytest.raises(ValueError, match="incompatible"):
        ComponentSelection.from_config(legacy).validate(candidate)
    pen = _config()
    with pytest.raises(ValueError, match="incompatible"):
        ClearVLAMainlinePolicy(replace(pen, top=replace(pen.top, calvin_object_binding="calvin_primary_v2")))


def test_v2_runs_the_real_two_pass_sampler_without_teacher() -> None:
    torch.manual_seed(619)
    base = _calvin_binary_config()
    config = replace(base, top=replace(base.top, calvin_object_binding="calvin_primary_v2"))
    model = ClearVLAMainlinePolicy(config).eval()
    online = _batch(config).online
    with (
        mock.patch.object(model, "build_training_targets", side_effect=AssertionError("teacher entered deployment")),
        mock.patch.object(model.world, "refine_deployment_world", wraps=model.world.refine_deployment_world) as refine,
        mock.patch.object(model, "velocity", wraps=model.velocity) as velocity,
    ):
        result = sample_action(model, online, config, generator=torch.Generator().manual_seed(33))
    assert refine.call_count == 1
    assert velocity.call_count == 12  # 5 updates + endpoint, twice; not twelve ODE updates.
    assert result.action.shape == (1, 24, 7)
    assert torch.isfinite(result.action).all()


def test_v2_bf16_bridge_preserves_finite_strength_and_null_fallback(seam) -> None:
    _, context, bridge = seam
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        result = bind(bridge, context)
    assert result.readout_strength is not None and torch.isfinite(result.readout_strength).all()
    torch.testing.assert_close(
        result.readout_strength.float(),
        bridge.context_gate.tanh().float() * result.valid_mass.float(),
        atol=0.005, rtol=0.02,
    )
