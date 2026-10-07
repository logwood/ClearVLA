"""Admission of the actual identity loss, including empty-label and leakage cases."""
from dataclasses import replace
import pytest
import torch
from test_source_consistent_measurement import production
from clearvla.mainline.identity_supervision import IdentityCorrespondence
from clearvla.mainline.training.identity import identity_terms, js_divergence
from clearvla.mainline.model.canonical_grounding import encode_owners


def labels(online, valid=True):
    xy=torch.tensor([[-.75,-.5],[.5,.75],[0.,0.]],device=online.device).expand(online.batch,2,-1,-1).clone()
    support=torch.full(xy.shape[:-1],valid,dtype=torch.bool,device=online.device)
    return IdentityCorrespondence(xy,xy.clone(),support,xy.clone(),xy.clone(),support.clone(),torch.tensor([8,16],device=online.device).expand(online.batch,-1))


def canonical(production):
    model,engine,batch,_,_=production
    if model.config.top.entity_ownership_mode!='canonical_image_v1':pytest.skip('identity loss is selected only in B')
    return model,batch


@pytest.mark.parametrize('autocast',[False,True])
def test_actual_loss_backward_and_empty_support(production,autocast):
    model,batch=canonical(production)
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=autocast):
        _,state,_=model.encode_online(batch.online)
        values=identity_terms(model,batch.online,state.top.facts,labels(batch.online))
        loss=values['identity_correspondence']+values['identity_source_prediction']
    assert all(torch.isfinite(value).all() for value in values.values())
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    for parameter in (model.grounding.grounder.slot_seed,model.grounding.grounder.content_key[1].weight,model.grounding.grounder.canonical_decoder.output.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
    with torch.no_grad():
        empty=identity_terms(model,batch.online,state.top.facts,labels(batch.online,False))
    assert empty['identity_correspondence']==0 and empty['identity_source_prediction']==0


def test_source_only_encoder_cannot_read_held_view(production):
    model,batch=canonical(production);module=model.grounding.grounder
    torch.manual_seed(417)
    candidates=torch.randn(4,16,module.hidden)
    mass=torch.ones(4,16);legal=mass.bool()
    with torch.no_grad():
        first,_=encode_owners(module,candidates,mass,legal)
        changed=candidates.clone();changed[1::2]=torch.randn_like(changed[1::2])*10
        second,_=encode_owners(module,changed,mass,legal)
    torch.testing.assert_close(first[::2],second[::2],atol=0,rtol=0)
    assert (first[1::2]-second[1::2]).abs().max()>1e-3


def test_wrong_owner_is_not_a_positive():
    first=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
    assert js_divergence(first,first).sum()==0
    assert (js_divergence(first,first.flip(1))>.69).all()


def test_duplicate_time_cannot_supply_temporal_evidence(production):
    model,batch=canonical(production);pairs=labels(batch.online)
    with pytest.raises(ValueError,match='padded duplicate'):
        replace(pairs,source_frames=torch.tensor([[8,8]])).validate(batch=1,device=batch.online.device)
