"""C6 observed-status value path; no task labels or confidence-derived loss masks."""
from dataclasses import replace
import gc
import pytest
import torch
from test_mainline_annotation_goal import _config as base_config
from test_mainline_state_features import _model_engine
from clearvla.mainline.runtime.qualification import synthetic_batch


def first_config(amp=False):
    c=base_config()
    return replace(c, top=replace(c.top, task_execution_mode="joint_object_scene_v1",
            p2_effect_value_mode="s_conditioned_values_v1"),
        bottom=replace(c.bottom, global_condition_mode="p3_compiled_task_v1"),
        runtime=replace(c.runtime,compute_dtype="bf16" if amp else "fp32"))


def _batch(c):
    return synthetic_batch(c,count=1,raw_side=32,device=torch.device("cpu"),seed=1540)[0]
from clearvla.mainline.feedback_values import INNOVATION_AND_STATUS
from clearvla.mainline.model.executed_world import ExecutedWorldPlanRead


def config(amp=False):
    c=first_config(amp)
    return replace(c,top=replace(c.top,world_feedback_value_mode=INNOVATION_AND_STATUS))


@pytest.fixture(scope='module')
def evidence():
    torch.manual_seed(1530)
    m,e=_model_engine(first_config());m.eval()
    with torch.no_grad():cache,state,_=m.encode_online(_batch(first_config()).online)
    yield cache.world_feedback.feedback, state.top.facts, state.top.intent.target_binding
    del cache,state,m,e
    gc.collect()


def pair(feedback, null):
    valid=feedback.view_observed
    mass=feedback.posterior.sum((-2,-1),keepdim=True)
    p=feedback.posterior/mass.clamp_min(1e-8)*(1-null)
    return replace(feedback,semantic=torch.zeros_like(feedback.semantic),image=torch.zeros_like(feedback.image),
                   posterior=p,null=torch.full_like(feedback.null,null))


@pytest.mark.parametrize('amp',[False,True])
def test_zero_projection_exact_legacy_and_context_zero_status_survives(evidence,amp):
    fb,facts,binding=evidence
    h=32;torch.manual_seed(1532);old=ExecutedWorldPlanRead(hidden=h,content_dim=facts.content.shape[-1],camera_names=fb.camera_names)
    torch.manual_seed(1532);new=ExecutedWorldPlanRead(hidden=h,content_dim=facts.content.shape[-1],camera_names=fb.camera_names,value_mode=INNOVATION_AND_STATUS)
    for n,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[n],rtol=0,atol=0)
    context=torch.randn(facts.batch,24,4,h)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        p=old.prepare(fb,facts,binding);q=new.prepare(fb,facts,binding)
        x=old(p,context,binding);y=new(q,context,binding)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    y.float().square().sum().backward()
    assert new.observed_status_projection.grad is not None and new.observed_status_projection.grad.abs().sum()>0
    with torch.no_grad():
        new.observed_status_projection.normal_(0,.05)
    fb=pair(fb,1.0)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        p=old.prepare(fb,facts,binding);q=new.prepare(fb,facts,binding)
        zero=torch.zeros_like(context)
        x=old(p,zero,binding);y=new(q,zero,binding)
    assert x.count_nonzero()==0 and p.value.count_nonzero()==0
    assert q.value.count_nonzero()==0 and y.abs().max()>0
    assert q.status_features[:,0].min()>0
    assert torch.isfinite(y).all()


def test_null_and_diffuseness_not_interpreted_as_zero_error(evidence):
    fb,facts,binding=evidence
    reader=ExecutedWorldPlanRead(hidden=32,content_dim=facts.content.shape[-1],camera_names=fb.camera_names,value_mode=INNOVATION_AND_STATUS)
    values=[]
    for null in [0.,.5,.99,1.]:
        p=reader.prepare(pair(fb,null),facts,binding)
        assert p.value.count_nonzero()==0
        assert (p.status_features>=-1e-6).all() and (p.status_features<=1+1e-6).all()
        values.append(p.status_features.detach())
    assert not torch.equal(values[0],values[-1])
    assert values[-1][...,1].min()>0
    null=.2
    allowed=fb.view_observed[...,None].expand_as(fb.posterior).float()
    diffuse=allowed/allowed.sum((-2,-1),keepdim=True).clamp_min(1)*(1-null)
    focused=torch.zeros_like(diffuse)
    # One supported cell, with exactly the same null mass and valid source views.
    for b in range(focused.shape[0]):
        for k in range(focused.shape[1]):
            c=int(fb.view_observed[b,k].to(torch.int).argmax())
            focused[b,k,c,0]=1-null
    a=reader.prepare(replace(pair(fb,null),posterior=focused),facts,binding)
    z=reader.prepare(replace(pair(fb,null),posterior=diffuse),facts,binding)
    assert not torch.equal(a.status_features,z.status_features)


def test_reset_support_does_not_become_uncertainty_claim(evidence):
    fb,facts,binding=evidence
    reader=ExecutedWorldPlanRead(hidden=32,content_dim=facts.content.shape[-1],camera_names=fb.camera_names,value_mode=INNOVATION_AND_STATUS)
    with torch.no_grad():reader.observed_status_projection.normal_()
    reset=replace(fb,view_observed=torch.zeros_like(fb.view_observed),window=replace(fb.window,observed=torch.zeros_like(fb.window.observed)),
                  posterior=torch.full_like(fb.posterior,float('nan')),null=torch.full_like(fb.null,float('nan')),
                  semantic=torch.full_like(fb.semantic,float('nan')),image=torch.full_like(fb.image,float('nan')),covariance=torch.full_like(fb.covariance,float('nan')))
    result=reader.prepare(reset,facts,binding)
    assert result.status_features.count_nonzero()==0 and result.value.count_nonzero()==0
    y=reader(result,torch.randn(facts.batch,24,4,32),binding)
    assert y.count_nonzero()==0 and torch.isfinite(y).all()


def test_missing_or_foreign_status_is_rejected(evidence):
    fb,facts,binding=evidence
    reader=ExecutedWorldPlanRead(hidden=32,content_dim=facts.content.shape[-1],camera_names=fb.camera_names,value_mode=INNOVATION_AND_STATUS)
    p=reader.prepare(fb,facts,binding);context=torch.randn(facts.batch,24,4,32)
    with pytest.raises(ValueError,match='status'):reader(replace(p,status_features=None),context,binding)
    with pytest.raises(ValueError,match='another reader'):reader(replace(p,reader_identity=123),context,binding)


def test_same_past_object_relabeling_preserves_status(evidence):
    fb,facts,binding=evidence
    reader=ExecutedWorldPlanRead(hidden=32,content_dim=facts.content.shape[-1],camera_names=fb.camera_names,value_mode=INNOVATION_AND_STATUS)
    p=reader.prepare(fb,facts,binding)
    perm=torch.arange(fb.semantic.shape[1]-1,-1,-1)
    moved=replace(fb,**{n:getattr(fb,n)[:,perm] for n in ['semantic','image','covariance','null','posterior','past_content','view_observed']})
    q=reader.prepare(moved,facts,binding)
    torch.testing.assert_close(p.value,q.value,atol=2e-7,rtol=2e-5)
    torch.testing.assert_close(p.status_features,q.status_features,atol=2e-7,rtol=2e-5)
