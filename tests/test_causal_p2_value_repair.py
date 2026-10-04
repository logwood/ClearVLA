"""P2 causal effect interpretation: values, not only interval-address weights."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.model.compiler import ObjectFutureEffectReader, SelectedIntervalEvidence
from clearvla.mainline.p2_values import CONTEXTUAL_EFFECT_VALUES, WORLD_EFFECT_VALUES
from clearvla.mainline.p2_geometry import VIEW_CONDITIONED_TRANSPORT
from clearvla.mainline.future_time import CONTROL_ALIGNED_FUTURE_TIME


def reader(mode=CONTEXTUAL_EFFECT_VALUES):
    return ObjectFutureEffectReader(hidden=16,content_dim=12,route_dim=8,
        target_binding_mode='shared_operation_v1',world_control_mode='known_prefix_v1',
        future_time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME,
        geometry_mode=VIEW_CONDITIONED_TRANSPORT,camera_names=('top','wrist'),
        task_execution_mode='joint_object_scene_v1',heads=2,effect_value_mode=mode)


def selected(*,constant=True,zero=False):
    b,t,q,i,h,d=2,3,2,4,16,12
    sem=torch.randn(b,t,q,1 if constant else i,d).expand(-1,-1,-1,i,-1).clone()
    geo=torch.randn(b,t,q,1 if constant else i,h).expand(-1,-1,-1,i,-1).clone()
    if zero:sem.zero_();geo.zero_()
    return SelectedIntervalEvidence(
        key=torch.randn(b,t,q,i,2,h),semantic_value=sem,
        semantic_common_value=sem,semantic_residual_value=torch.zeros_like(sem),
        geometry_value=geo,geometry_common_value=geo,geometry_residual_value=torch.zeros_like(geo),
        selected_s_context=torch.randn(b,t,q,i,2,h),support=torch.ones(b,i,2,dtype=torch.bool),
        time_grid_mode=CONTROL_ALIGNED_FUTURE_TIME,geometry_mode=VIEW_CONDITIONED_TRANSPORT)


@pytest.mark.parametrize('amp',[False,True])
def test_zero_start_exact_old_values_parameter_initialization_and_rng(amp):
    torch.manual_seed(1001);old=reader(WORLD_EFFECT_VALUES);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(1001);new=reader();assert torch.equal(old_rng,torch.get_rng_state())
    assert set(new.state_dict())-set(old.state_dict())=={'contextual_effect_gain'}
    for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
    s=selected(constant=False);q=torch.randn(2,3,2,16)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        a,am=old.temporal_terminal(q,s,collect_diagnostics=True)
        b,bm=new.temporal_terminal(q,s,collect_diagnostics=True)
    torch.testing.assert_close(a.semantic,b.semantic,atol=0,rtol=0)
    torch.testing.assert_close(a.geometry,b.geometry,atol=0,rtol=0)
    assert torch.count_nonzero(new.contextual_effect_gain)==0


@pytest.mark.parametrize('amp',[False,True])
def test_s_changes_effect_values_when_interval_weights_are_fixed_and_values_identical(amp):
    torch.manual_seed(1003);m=reader();old=reader(WORLD_EFFECT_VALUES)
    old.load_state_dict({k:v for k,v in m.state_dict().items() if k!='contextual_effect_gain'})
    with torch.no_grad():
        m.contextual_effect_gain.fill_(.25)
        for layer in list(m.terminal_query)+list(old.terminal_query):layer.weight.zero_()
    s=selected();source=s.selected_s_context.requires_grad_();q=torch.randn(2,3,2,16)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        a,_=m.temporal_terminal(q,s,collect_diagnostics=False)
        b,_=m.temporal_terminal(q,replace(s,selected_s_context=-source),collect_diagnostics=False)
        x,_=old.temporal_terminal(q,s,collect_diagnostics=False)
        y,_=old.temporal_terminal(q,replace(s,selected_s_context=-source),collect_diagnostics=False)
    torch.testing.assert_close(x.semantic,y.semantic,atol=0,rtol=0)
    torch.testing.assert_close(x.geometry,y.geometry,atol=0,rtol=0)
    assert (a.semantic-b.semantic).abs().max()>.001
    assert (a.geometry-b.geometry).abs().max()>.001
    (a.semantic.float().square().sum()+a.geometry.float().square().sum()).backward()
    assert source.grad is not None and source.grad.abs().max()>1e-5
    assert m.contextual_effect_gain.grad is not None and m.contextual_effect_gain.grad.abs().sum()>0


def test_gain_is_trainable_from_exact_neutral_start():
    torch.manual_seed(1004);m=reader();s=selected();q=torch.randn(2,3,2,16)
    old=m.contextual_effect_gain.detach().clone()
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    a,_=m.temporal_terminal(q,s,collect_diagnostics=False)
    (a.semantic.square().sum()+a.geometry.square().sum()).backward();optimizer.step()
    assert not torch.equal(m.contextual_effect_gain,old)
    assert torch.isfinite(m.contextual_effect_gain).all()


@pytest.mark.parametrize('amp',[False,True])
def test_zero_world_remains_zero_even_with_nonzero_s_and_learned_gain(amp):
    torch.manual_seed(1005);m=reader()
    with torch.no_grad():m.contextual_effect_gain.fill_(2.)
    s=selected(zero=True);q=torch.randn(2,3,2,16)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        a,_=m.temporal_terminal(q,s,collect_diagnostics=False)
    assert torch.count_nonzero(a.semantic)==0 and torch.count_nonzero(a.geometry)==0


def test_missing_intervals_do_not_create_effect_and_output_respects_interventions():
    torch.manual_seed(1006);m=reader();m.contextual_effect_gain.data.fill_(.5)
    s=selected();support=s.support.clone();support[:,1:]=False
    s=replace(s,support=support)
    q=torch.randn(2,3,2,16)
    a,_=m.temporal_terminal(q,s,collect_diagnostics=False)
    c=s.selected_s_context.clone();c[:,:,:,1:]=1234.
    b,_=m.temporal_terminal(q,replace(s,selected_s_context=c),collect_diagnostics=False)
    torch.testing.assert_close(a.semantic,b.semantic,atol=0,rtol=0)
    torch.testing.assert_close(a.geometry,b.geometry,atol=0,rtol=0)
    m.eval(); m.set_eval_intervention('geometry_value_all_zero')
    zero,_=m.temporal_terminal(q,s,collect_diagnostics=False)
    assert torch.count_nonzero(zero.geometry)==0


def test_explicit_configuration_roundtrip_and_original_preset_unchanged():
    old=load_config('configs/mainline/dinov3_online_task_global_calvin.json')
    new=load_config('configs/mainline/dinov3_causal_chain_calvin.json')
    assert 'p2_effect_value_mode' not in old.as_dict()['top']
    assert config_from_mapping(new.as_dict())==new
    assert replace(new,top=old.top,data=old.data)==old
    with pytest.raises(ValueError):replace(new,top=replace(new.top,p2_effect_value_mode='guess')).validate()
    with pytest.raises(ValueError):replace(new,top=replace(new.top,target_binding_mode='reader_local_v1')).validate()
