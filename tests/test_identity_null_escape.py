"""Prove the joint-null escape and qualify the versioned conditional repair."""
import copy
from dataclasses import replace
import pytest
import torch
from test_source_consistent_measurement import production
from test_identity_supervision import labels
from clearvla.mainline.training import identity


def test_conditional_identity_ignores_null_logit_before_interpolation():
    torch.manual_seed(629)
    logits = torch.randn(1, 5, 2, 2, requires_grad=True)
    legal = torch.ones_like(logits[:, :4], dtype=torch.bool)
    def law(value):
        logs = value.log_softmax(1)[:, :4]
        return identity.conditional_real_law(logs, legal)[0]
    first = law(logits)
    changed = logits.detach().clone()
    changed[:, -1] += torch.tensor([[0., 2.], [6., 40.]])
    second = law(changed)
    torch.testing.assert_close(first, second, atol=5e-7, rtol=3e-6)
    xy = torch.tensor([[[-.25, .31], [.3, -.5]]])
    sampled = identity.sample(first, xy)
    loss = (sampled * torch.arange(4)).sum()
    grad, = torch.autograd.grad(loss, logits)
    assert grad[:, :4].abs().max() > 1e-3
    assert grad[:, -1].abs().max() < 1e-7


def test_missing_producer_cells_stay_unknown_and_have_finite_zero_gradients():
    value = torch.full((1, 4, 2, 2), float('nan'), requires_grad=True)
    law, support = identity.conditional_real_law(value, torch.zeros_like(value, dtype=torch.bool))
    assert torch.count_nonzero(law) == 0 and not support.any()
    law.sum().backward()
    assert torch.equal(value.grad, torch.zeros_like(value))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_actual_identity_objective_cannot_win_by_uniform_abstention(production, monkeypatch, dtype):
    source_model, _, batch, _, _ = production
    if source_model.config.top.entity_ownership_mode != 'canonical_image_v1':
        pytest.skip('canonical identity objective only')
    model = copy.deepcopy(source_model).eval()
    model.zero_grad(set_to_none=True)
    # This compact fixture supplies observed features directly. Full production
    # config / uncached RGB admission is checked by the real BS8 run.
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        _, state, _ = model.encode_online(batch.online)
        facts = state.top.facts
        pair = labels(batch.online)
        model.config = replace(model.config, top=replace(model.config.top, identity_supervision_mode='rgbd_temporal_conditional_v2'))
        original = identity.encode_owners
        v2 = identity.identity_terms(model, batch.online, facts, pair)
        model.config = replace(model.config, top=replace(model.config.top, identity_supervision_mode='rgbd_temporal_v1'))
        v1 = identity.identity_terms(model, batch.online, facts, pair)
        scale = .1
        def abstain(*args, **kwargs):
            slots, logs = original(*args, **kwargs)
            retained = logs[..., :-1] + logs.new_tensor(scale).log()
            null = torch.logaddexp(logs[..., -1:], logs.new_tensor(1 - scale).log() + logs[..., :-1].logsumexp(-1, keepdim=True))
            return slots, torch.cat((retained, null), -1)
        monkeypatch.setattr(identity, 'encode_owners', abstain)
        full = facts.image_ownership
        more_null = torch.cat((full[:, :-1] * scale, full[:, -1:] + (1 - scale) * full[:, :-1].sum(1, keepdim=True)), 1)
        source = replace(facts.current_image_source, log_measure=facts.current_image_source.log_measure + torch.tensor(scale).log())
        shifted = replace(facts, image_ownership=more_null, current_image_source=source)
        v1_shifted = identity.identity_terms(model, batch.online, shifted, pair)
        model.config = replace(model.config, top=replace(model.config.top, identity_supervision_mode='rgbd_temporal_conditional_v2'))
        v2_shifted = identity.identity_terms(model, batch.online, shifted, pair)
        assert v1['identity_correspondence'] > 0
        assert v1_shifted['identity_correspondence'] < .2 * v1['identity_correspondence']
        torch.testing.assert_close(v2['identity_correspondence'], v2_shifted['identity_correspondence'], rtol=2e-5, atol=2e-6)
        for key in ['identity_source_prediction', 'identity_source_removed_prediction_mse']:
            torch.testing.assert_close(v2[key], v2_shifted[key], rtol=1e-5, atol=2e-6)
        loss = v2['identity_correspondence'] + v2['identity_source_prediction']
    loss.backward()
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
    assert model.grounding.grounder.slot_seed.grad.abs().sum() > 0
    assert model.grounding.grounder.content_key[1].weight.grad.abs().sum() > 0
    assert model.grounding.grounder.canonical_decoder.output.weight.grad.abs().sum() > 0


def test_v1_deployment_metadata_is_unchanged_and_v2_is_explicit():
    from clearvla.mainline.causal_identity import causal_identity_metadata
    one = causal_identity_metadata({'identity_supervision_mode': 'rgbd_temporal_v1'})
    two = causal_identity_metadata({'identity_supervision_mode': 'rgbd_temporal_conditional_v2'})
    assert 'correspondence' not in one['identity_supervision']
    assert 'real-K-conditional-before-interpolation' in two['identity_supervision']['correspondence']
    assert one != two
