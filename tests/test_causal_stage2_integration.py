"""C5/C6 actual cumulative policy path; compact CPU, synthetic visual transport."""
from dataclasses import replace
import gc
from pathlib import Path
import pytest
import torch
from test_causal_observed_status import first_config, _batch, _model_engine
from test_mainline_global_task import abi_for
from clearvla.mainline.config import load_config, config_from_mapping
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.runtime.deployment import validate_deployment_abi
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact,save_checkpoint
from clearvla.simulation.checkpoint import load_deployment_checkpoint


def config(amp=False):
    c=first_config(amp)
    return replace(c,top=replace(c.top,task_role_value_mode='source_conditioned_values_v1',world_feedback_value_mode='innovation_and_status_v1'))


def batch(c):
    b=_batch(c);g=b.online.goal
    tokens=torch.randn(g.tokens.shape,generator=torch.Generator().manual_seed(1582),dtype=g.tokens.dtype)
    tokens=torch.where(g.mask[...,None],tokens,0.)
    return replace(b,online=replace(b.online,goal=replace(g,tokens=tokens)))


@pytest.fixture(autouse=True)
def cleanup():
    gc.collect();yield;gc.collect()


@pytest.mark.parametrize('amp',[False,True])
def test_full_zero_start_and_ordinary_training_update(amp):
    c=config(amp);b=batch(c)
    torch.manual_seed(1581);m,e=_model_engine(c)
    torch.manual_seed(1581);old,_=_model_engine(first_config(amp))
    for n,v in old.state_dict().items():torch.testing.assert_close(v,m.state_dict()[n],rtol=0,atol=0)
    m.eval();old.eval()
    dtype=torch.bfloat16 if amp else torch.float32
    with torch.no_grad():
        x=sample_action(m,b.online,c,dtype=dtype,generator=torch.Generator().manual_seed(1591))
        y=sample_action(old,b.online,first_config(amp),dtype=dtype,generator=torch.Generator().manual_seed(1591))
    torch.testing.assert_close(x.action,y.action,rtol=0,atol=0)
    del old,x,y;gc.collect()
    tracked={n:p for n,p in m.named_parameters() if 'contextual_role_gain' in n or 'observed_status_projection' in n}
    assert len(tracked)==10
    owned=[p for group in e.optimizer.param_groups for p in group['params']]
    assert all(sum(p is v for v in owned)==1 and p.requires_grad for p in tracked.values())
    e.train_step(b);r=e.train_step(b)
    assert e.global_step==2 and torch.isfinite(r.loss)
    for n,p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0,n
        assert p.detach().abs().max()>0,n
    m.eval()
    with torch.no_grad():
        x=sample_action(m,b.online,c,dtype=dtype,generator=torch.Generator().manual_seed(1591))
        y=sample_action(m,b.online,c,dtype=dtype,generator=torch.Generator().manual_seed(1591))
    torch.testing.assert_close(x.action,y.action,rtol=0,atol=0)


def test_lifecycle_keeps_status_source_once_per_observation():
    c=config();m,_=_model_engine(c);m.eval();b=batch(c)
    reader=m.policy_compiler.plan_compiler.world_feedback_read
    counts={'prepare':0,'forward':0,'compile':0};records=[]
    original=reader.prepare
    def prepare(*a,**kw):counts['prepare']+=1;return original(*a,**kw)
    reader.prepare=prepare
    h=reader.register_forward_pre_hook(lambda module,args:(records.append(args[0].status_features),counts.update(forward=counts['forward']+1)) and None)
    h2=m.policy_compiler.plan_compiler.global_task_compiler.register_forward_hook(lambda *a:counts.update(compile=counts['compile']+1))
    try:
        with torch.no_grad():x=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1600))
        assert counts=={'prepare':1,'forward':12,'compile':1}
        assert records[0] is not None and all(v is records[0] for v in records)
        assert x.action.shape==(1,24,7)
    finally:reader.prepare=original;h.remove();h2.remove()


@pytest.mark.parametrize('kind',['role_missing','role_changed','status_missing','status_changed'])
def test_abi_rejects_changed_meanings(tmp_path,kind):
    _,abi,_=abi_for(config(),tmp_path);validate_deployment_abi(abi)
    key='task_role_values' if kind.startswith('role') else 'world_feedback_values'
    if kind.endswith('missing'):abi.pop(key)
    else:abi[key]['source']='raw-unknown'
    with pytest.raises(ValueError):validate_deployment_abi(abi)


def test_config_differences_preserve_other_source_and_objective_choices():
    a=load_config('configs/mainline/dinov3_causal_chain_calvin.json')
    b=load_config('configs/mainline/dinov3_causal_chain_stage2_calvin.json')
    restored=replace(b,top=replace(b.top,task_role_value_mode=a.top.task_role_value_mode,world_feedback_value_mode=a.top.world_feedback_value_mode),data=replace(b.data,output_dir=a.data.output_dir))
    assert restored==a and config_from_mapping(b.as_dict())==b
    assert 'world_feedback_value_mode' not in a.as_dict()['top']
    with pytest.raises(ValueError):replace(b,top=replace(b.top,world_feedback_mode='none')).validate()


def test_exact_checkpoint_new_graph_restore_and_deployment(tmp_path):
    torch.manual_seed(1603);c=config();m,e=_model_engine(c);b=batch(c)
    identity,abi,data=abi_for(c,tmp_path);m.configure_action_normalizer(data.action_normalizer)
    e.train_step(b);e.train_step(b);m.eval()
    path=tmp_path/'new.pt'
    save_checkpoint(path,model=m,optimizer=e.optimizer,schedule=e.schedule,config=c,identity=identity,epoch=0,global_step=e.global_step,best_metric=None,data_state={'action_normalizer':data.action_normalizer.to_dict(),'state_normalizer':data.state_normalizer.to_dict(),'deployment_abi':abi})
    with torch.no_grad():expected=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1604)).action
    other,oe=_model_engine(c)
    load_checkpoint_exact(path,model=other,optimizer=oe.optimizer,schedule=oe.schedule,config=c,identity=identity)
    for n,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[n],rtol=0,atol=0)
    del m,e,other,oe;gc.collect()
    loaded=load_deployment_checkpoint(path,device=torch.device('cpu'))
    with torch.no_grad():actual=sample_action(loaded.model,b.online,c,generator=torch.Generator().manual_seed(1604)).action
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
