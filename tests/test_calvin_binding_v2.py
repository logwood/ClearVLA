"""Action-level null-owner regressions; pointer normalization alone is not enough."""
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.config import load_config
from clearvla.mainline.model.calvin_object_binding import CalvinObjectBindingBridge
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.components import IntentStage
from clearvla.mainline.model.intent import CoarseActionIntent
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from tests.test_mainline_policy import _calvin_binary_config, _config
from tests.test_mainline_top import _context, _top


def _bridge_and_context(v2=True):
    torch.manual_seed(20260917)
    context = _context(_top(), batch=1)
    bridge = CalvinObjectBindingBridge(hidden=32, route_dim=8, preserve_null_mass=v2)
    kwargs = dict(protected_goal=context.intent.protected_goal_set.detach(),
                  history_tokens=context.intent.history_tokens.detach(),
                  object_tokens=context.intent.object_tokens.detach(), facts=context.facts)
    return bridge, context, kwargs


def _dock(context, result):
    return replace(context.intent.action_dock(), selected_object_context=result.selected_context,
                   object_binding_pointer=result.pointer,
                   object_binding_selected_geometry=result.selected_geometry,
                   object_binding_context_scale=result.context_scale)


def test_v2_null_and_blend_owners_receive_action_level_vjp():
    bridge, context, kwargs = _bridge_and_context()
    coarse = CoarseActionIntent(hidden=32, action_dim=7, heads=4)
    with torch.no_grad():
        bridge.null_scorer.weight.zero_()
    first, _ = bridge(**kwargs)
    action = coarse(_dock(context, first)).action_prediction
    null_grad, gate_grad = torch.autograd.grad(action.square().sum(),
                                               (bridge.null_scorer.weight, bridge.context_gate))
    assert null_grad.norm() > 1e-4 and gate_grad.abs() > 1e-4
    query = bridge.goal_norm(kwargs['protected_goal'].float().mean(1) + .25 * kwargs['history_tokens'][:, -1].float())
    with torch.no_grad():
        bridge.null_scorer.weight.copy_(query * (8.0 / query.square().sum()))
    second, _ = bridge(**kwargs)
    refined = coarse(_dock(context, second)).action_prediction
    assert second.pointer[0, -1] > .99
    assert second.context_scale.abs().max() < .01 * first.context_scale.abs().max()
    assert (refined - action).abs().max() > 1e-3


@pytest.mark.parametrize('v2', [False, True])
def test_v1_semantics_remain_explicit_and_all_invalid_v2_has_zero_contribution(v2):
    bridge, context, kwargs = _bridge_and_context(v2)
    result, _ = bridge(**kwargs)
    assert (result.context_scale is not None) == v2
    kwargs['facts'] = replace(context.facts, validity=torch.zeros_like(context.facts.validity))
    result, _ = bridge(**kwargs)
    assert torch.equal(result.pointer, torch.tensor([[0., 0., 0., 0., 1.]]))
    assert torch.count_nonzero(result.selected_context) == 0
    if v2:
        assert torch.count_nonzero(result.context_scale) == 0
        coarse = CoarseActionIntent(hidden=32, action_dim=7, heads=4)
        dock = _dock(context, result)
        # With a zero scale even arbitrary memory must have no influence.
        other = replace(dock, selected_object_context=torch.randn_like(result.selected_context))
        assert torch.equal(coarse(dock).action_prediction, coarse(other).action_prediction)


def test_v2_pointer_and_action_are_slot_permutation_equivariant():
    bridge, context, kwargs = _bridge_and_context()
    result, _ = bridge(**kwargs)
    index = torch.tensor([2, 0, 3, 1])
    kwargs['facts'] = context.facts.permute(index)
    kwargs['object_tokens'] = kwargs['object_tokens'][:, index]
    permuted, _ = bridge(**kwargs)
    torch.testing.assert_close(permuted.pointer, result.pointer[:, [2, 0, 3, 1, 4]])
    torch.testing.assert_close(permuted.selected_context, result.selected_context)
    torch.testing.assert_close(permuted.context_scale, result.context_scale)
    coarse = CoarseActionIntent(hidden=32, action_dim=7, heads=4)
    torch.testing.assert_close(coarse(_dock(context, result)).action_prediction,
                               coarse(_dock(context, permuted)).action_prediction)


def test_v2_component_identity_changes_without_parameter_or_rng_changes():
    base = _calvin_binary_config()
    one = replace(base, top=replace(base.top, calvin_object_binding='calvin_primary_v1'))
    two = replace(base, top=replace(base.top, calvin_object_binding='calvin_primary_v2'))
    assert ComponentSelection.from_config(one).intent != ComponentSelection.from_config(two).intent
    torch.manual_seed(77)
    old = ClearVLAMainlinePolicy(one)
    old_rng = torch.get_rng_state()
    torch.manual_seed(77)
    new = ClearVLAMainlinePolicy(two)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert new.intent.calvin_object_binding.preserve_null_mass
    assert not old.intent.calvin_object_binding.preserve_null_mass
    # Reject the variant for Pen before constructing any module.
    pen = _config()
    with pytest.raises(ValueError):
        ComponentSelection.from_config(replace(pen, top=replace(pen.top, calvin_object_binding='calvin_primary_v2')))
    assert load_config('configs/mainline/calvin_object_binding_v2.json').top.calvin_object_binding == 'calvin_primary_v2'


def test_v2_intent_stage_propagates_scale_to_coarse_dock():
    bridge, context, _ = _bridge_and_context()
    stage = IntentStage(torch.nn.Identity(), CoarseActionIntent(hidden=32, action_dim=7, heads=4), bridge)
    bound, _ = stage.bind_object(context.intent, context.facts)
    assert bound.action_dock().object_binding_context_scale is bound.object_binding_context_scale
    permuted = bound.permute(torch.tensor([3, 2, 1, 0]))
    assert permuted.object_binding_context_scale is bound.object_binding_context_scale
    action = stage.coarse_action(bound.action_dock()).action_prediction
    grad, = torch.autograd.grad(action.square().sum(), bridge.null_scorer.weight)
    assert grad.norm() > 1e-4


def test_optional_binding_owner_is_reported_only_when_constructed():
    from clearvla.mainline.training.optimizer import build_optimizer
    for mode in ('disabled', 'calvin_primary_v1', 'calvin_primary_v2'):
        base = _calvin_binary_config()
        config = replace(base, top=replace(base.top, calvin_object_binding=mode))
        model = ClearVLAMainlinePolicy(config)
        optimizer, ownership = build_optimizer(model, config)
        assert ('calvin_object_binding' in ownership.role_counts) == (mode != 'disabled')
        assert all(count > 0 for count in ownership.role_counts.values())
        actual = [id(p) for group in optimizer.param_groups for p in group['params']]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        assert len(actual) == len(set(actual)) and set(actual) == expected


def test_v1_checkpoint_selection_cannot_be_loaded_as_v2():
    base = _calvin_binary_config()
    one = replace(base, top=replace(base.top, calvin_object_binding='calvin_primary_v1'))
    two = replace(base, top=replace(base.top, calvin_object_binding='calvin_primary_v2'))
    with pytest.raises(ValueError):
        ComponentSelection.from_mapping(ComponentSelection.from_config(one).as_dict(), config=two)


def test_v2_bf16_pointer_scale_and_action_gradient_remain_finite():
    bridge, context, kwargs = _bridge_and_context()
    coarse = CoarseActionIntent(hidden=32, action_dim=7, heads=4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result, _ = bridge(**kwargs)
        action = coarse(_dock(context, result)).action_prediction
    result.validate(batch=1, objects=4, hidden=32)
    grad, = torch.autograd.grad(action.float().square().sum(), bridge.null_scorer.weight)
    assert torch.isfinite(grad).all() and grad.norm() > 1e-4
