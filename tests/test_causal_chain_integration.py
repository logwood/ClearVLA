"""Current causal-chain changes on actual policy/engine; compact synthetic data."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import gc

import pytest
import torch
from test_mainline_global_task import config as global_config, abi_for
from test_mainline_task_execution_integration import _batch
from test_mainline_state_features import _model_engine
from clearvla.mainline.runtime.deployment import validate_deployment_abi
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.runtime.checkpoints import save_checkpoint, load_checkpoint_exact
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.mainline.p2_values import CONTEXTUAL_EFFECT_VALUES


def config(amp=False):
    c=global_config(amp)
    return replace(c,top=replace(c.top,p2_effect_value_mode=CONTEXTUAL_EFFECT_VALUES))


@pytest.fixture(autouse=True)
def clean():
    gc.collect()
    yield
    gc.collect()


@pytest.mark.parametrize('amp',[False,True])
def test_real_update_owns_and_updates_binder_and_p2_value_gain(amp):
    torch.manual_seed(1101);c=config(amp);m,e=_model_engine(c);b=_batch()
    tracked={n:p for n,p in m.named_parameters()
             if 'task_object_score' in n or 'contextual_effect_gain' in n}
    assert len(tracked)==2
    owners=[p for group in e.optimizer.param_groups for p in group['params']]
    for p in tracked.values():assert sum(p is x for x in owners)==1 and p.requires_grad
    before={n:p.detach().clone() for n,p in tracked.items()}
    # Candidate W starts at exact zero in the real model. A zero-preserving
    # P2 value interpretation must not invent an effect/gradient there. Let
    # the ORIGINAL W objective learn before testing this downstream owner.
    result=e.train_step(b,collect_diagnostics=True)
    assert torch.count_nonzero(m.policy_compiler.effect_reader.contextual_effect_gain)==0
    result=e.train_step(b,collect_diagnostics=True)
    assert e.global_step==2 and torch.isfinite(result.loss)
    for n,p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
        assert not torch.equal(before[n],p),n
    assert 'gradient_tensor_s_typed_value_predecomposition_rms' in result.metrics
    m.eval()
    with torch.no_grad():
        a=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1102))
        d=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1102))
    assert torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action,d.action,atol=0,rtol=0)


def test_complete_lifecycle_keeps_one_causal_encoding_and_world_rebuild():
    torch.manual_seed(1103);c=config();m,_=_model_engine(c);m.eval()
    counts={'compile':0,'terminal':0}
    def compiled(*_):counts['compile']+=1
    h=m.policy_compiler.plan_compiler.global_task_compiler.register_forward_hook(compiled)
    original=m.policy_compiler.effect_reader.temporal_terminal
    def terminal(*a,**kw):
        counts['terminal']+=1
        return original(*a,**kw)
    m.policy_compiler.effect_reader.temporal_terminal=terminal
    try:
        with torch.no_grad():a=sample_action(m,_batch().online,c,generator=torch.Generator().manual_seed(1104))
        assert counts=={'compile':1,'terminal':12}
        assert a.action.shape==(1,24,7)
    finally:
        h.remove();m.policy_compiler.effect_reader.temporal_terminal=original


@pytest.mark.parametrize('kind',['p2_missing','p2_changed','old_binding'])
def test_abi_rejects_semantic_relabeling(tmp_path,kind):
    _,abi,_=abi_for(config(),tmp_path)
    validate_deployment_abi(abi)
    if kind=='p2_missing':abi.pop('p2_effect_values')
    elif kind=='p2_changed':abi['p2_effect_values']['source']='task-only'
    else:abi['task_execution'].pop('binding_residual')
    with pytest.raises(ValueError):validate_deployment_abi(abi)


def test_new_checkpoint_exact_restore_and_deployment(tmp_path):
    torch.manual_seed(1105);c=config();m,e=_model_engine(c);b=_batch()
    identity,abi,data=abi_for(c,tmp_path);m.configure_action_normalizer(data.action_normalizer)
    e.train_step(b);m.eval()
    ck=tmp_path/'candidate.pt'
    save_checkpoint(ck,model=m,optimizer=e.optimizer,schedule=e.schedule,config=c,identity=identity,
                    epoch=0,global_step=1,best_metric=None,
                    data_state={'action_normalizer':data.action_normalizer.to_dict(),
                                'state_normalizer':data.state_normalizer.to_dict(),'deployment_abi':abi})
    with torch.no_grad():expected=sample_action(m,b.online,c,generator=torch.Generator().manual_seed(1106)).action
    other,oe=_model_engine(c)
    load_checkpoint_exact(ck,model=other,optimizer=oe.optimizer,schedule=oe.schedule,config=c,identity=identity)
    for n,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[n],atol=0,rtol=0)
    del m,e,other,oe;gc.collect()
    deployed=load_deployment_checkpoint(ck,device=torch.device('cpu'))
    with torch.no_grad():actual=sample_action(deployed.model,b.online,c,generator=torch.Generator().manual_seed(1106)).action
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)


def test_zero_start_full_sampling_matches_world_value_mode():
    torch.manual_seed(1111);m,_=_model_engine(config());m.eval()
    torch.manual_seed(1111);old,_=_model_engine(global_config());old.eval()
    b=_batch()
    for n,v in old.state_dict().items():torch.testing.assert_close(m.state_dict()[n],v,atol=0,rtol=0)
    with torch.no_grad():
        a=sample_action(m,b.online,config(),generator=torch.Generator().manual_seed(1112))
        z=sample_action(old,b.online,global_config(),generator=torch.Generator().manual_seed(1112))
    torch.testing.assert_close(a.action,z.action,atol=0,rtol=0)
    torch.testing.assert_close(a.physical_field,z.physical_field,atol=0,rtol=0)
