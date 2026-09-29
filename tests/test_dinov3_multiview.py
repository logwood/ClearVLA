"""Per-camera values and consumers; deterministic synthetic evidence, not success rates."""
from dataclasses import replace

import pytest
import torch

from test_mainline_structural_contracts import _local_facts
from test_mainline_task_execution import fixture as relation_fixture
from test_mainline_annotation_goal import _config as cumulative_config
from test_mainline_state_features import _model_engine
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.task_execution import JOINT_TASK_EXECUTION
from clearvla.vision.entity_chart import CurrentImageSupport, CURRENT_IMAGE_CHART


def test_grounder_retains_observed_camera_values_not_expanded_global():
    torch.manual_seed(85)
    local = _local_facts(cameras=2)
    # Every candidate in a view shares a distinguishable value; conditional
    # pooling may change its address weights but cannot change its content.
    values = local.content_slots.clone()
    values[:,0] = 2.; values[:,1] = -3.
    prefix = local.slot_validity.shape[:-1]
    xy = torch.zeros(*prefix, 1, 2)
    local = replace(local, content_slots=values,
        current_image_support=CurrentImageSupport(xy,torch.ones(*prefix,1),torch.ones(*prefix,1,dtype=torch.bool),torch.zeros(*prefix,1)))
    g = DenseObjectGrounder(hidden=32,content_dim=16,route_dim=8,
        entity_chart_mode=CURRENT_IMAGE_CHART,object_view_mode='per_camera_values_v1')
    facts,_ = g(local)
    assert facts.camera_content.shape == (1,4,2,16)
    torch.testing.assert_close(facts.camera_content[:,:,0],torch.full((1,4,16),2.))
    torch.testing.assert_close(facts.camera_content[:,:,1],torch.full((1,4,16),-3.))
    p = torch.tensor([2,0,3,1])
    swapped = facts.permute(p)
    torch.testing.assert_close(swapped.camera_content,facts.camera_content[:,p])
    swapped.validate()


@pytest.mark.parametrize('amp',[False,True])
def test_view_specific_relation_changes_only_that_view_before_role_read(amp):
    enc,kw,*_ = relation_fixture()
    old=kw['attributes']
    kw['attributes']=torch.stack((old,old+.3),dim=2).requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        first=enc(**kw)
        second_values=kw['attributes'].clone()
        second_values[:,:,0] += torch.linspace(0,2,16)
        second=enc(**{**kw,'attributes':second_values})
        torch.testing.assert_close(first.values[:,:,:,1],second.values[:,:,:,1],rtol=0,atol=0)
        assert not torch.equal(first.values[:,:,:,0],second.values[:,:,:,0])
        second.values.square().sum().backward()
    assert torch.isfinite(kw['attributes'].grad).all()
    assert (kw['attributes'].grad.abs().sum((0,1,3,4))>0).all()


def test_view_binder_equivariance_count_neutrality_and_no_history_only_target():
    torch.manual_seed(86)
    binder=TaskConditionedTargetBinder(16,4)
    task=torch.randn(2,4,16);objects=torch.randn(2,4,2,16);hist=torch.randn(2,16)
    valid=torch.ones(2,4,dtype=torch.bool);views=torch.ones(2,4,2,dtype=torch.bool)
    a=binder(task,objects,valid,history=hist,view_support=views)
    b=binder(task,objects.flip(2),valid,history=hist,view_support=views.flip(2))
    torch.testing.assert_close(a.mass,b.mass,atol=1e-7,rtol=1e-6)
    duplicate=objects[:,:,:1].expand(-1,-1,2,-1)
    both=binder(task,duplicate,valid,history=hist,view_support=views)
    half=views.clone();half[:,:,1]=False
    one=binder(task,duplicate,valid,history=hist,view_support=half)
    torch.testing.assert_close(both.mass,one.mass,atol=1e-7,rtol=1e-6)
    zero=binder(task*0,objects,valid,history=hist,view_support=views)
    torch.testing.assert_close(zero.mass,zero.mass[:,:1].expand_as(zero.mass),atol=0,rtol=0)
    missing=binder(task,objects*float('nan'),valid,history=hist,view_support=views&False)
    assert missing.mass.count_nonzero()==0
    torch.testing.assert_close(missing.null_mass,torch.ones_like(missing.null_mass))


@pytest.mark.parametrize('amp',[False,True])
def test_per_camera_cumulative_graph_actual_update_and_sample_compact(amp):
    # This is the compact graph composition test. Full256x768 source-entry
    # tests are separate and must not be replaced by this smaller fixture.
    torch.manual_seed(87)
    cfg=cumulative_config()
    cfg=replace(cfg,top=replace(cfg.top,task_execution_mode=JOINT_TASK_EXECUTION,
                               object_view_mode='per_camera_values_v1'),
                runtime=replace(cfg.runtime,compute_dtype='bf16' if amp else 'fp32'))
    cfg.validate()
    batch,norm=synthetic_batch(cfg,count=1,raw_side=32,device=torch.device('cpu'),seed=88)
    model,engine=_model_engine(cfg);model.configure_action_normalizer(norm)
    result=engine.train_step(batch)
    assert torch.isfinite(result.loss) and engine.global_step==1
    model.eval()
    with torch.no_grad():
        a=sample_action(model,batch.online,cfg,generator=torch.Generator().manual_seed(89))
        b=sample_action(model,batch.online,cfg,generator=torch.Generator().manual_seed(89))
    assert torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action,b.action,atol=0,rtol=0)


@pytest.mark.parametrize('side,stride',[(336,4),(336,8),(16,2)])
def test_raw_pyramid_conv_centers_to_full_image_chart(side,stride):
    from clearvla.vision.online_pipeline import rasterize_strided_rgb
    n=side//stride
    x=2*torch.arange(n)*stride/(side-1)-1
    yy,xx=torch.meshgrid(x,x,indexing='ij')
    chart=torch.stack((xx,yy),0)[None].requires_grad_()
    result=rasterize_strided_rgb(chart,image_hw=(side,side),stride=stride)
    target=torch.linspace(-1,1,n)
    torch.testing.assert_close(result[0,0,0,:-1],target[:-1],atol=4*torch.finfo(torch.float32).eps,rtol=0)
    torch.testing.assert_close(result[0,1,:-1,0],target[:-1],atol=4*torch.finfo(torch.float32).eps,rtol=0)
    # The last native center is the declared boundary extension, not pixel335.
    torch.testing.assert_close(result[0,0,0,-1],x[-1])
    result.square().sum().backward()
    assert torch.isfinite(chart.grad).all()
