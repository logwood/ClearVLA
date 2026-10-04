"""C5: source-conditioned role read, without a query-only or second-K policy."""
from dataclasses import replace
import pytest
import torch
from clearvla.mainline.model.task_execution import ObjectRoleRead
from clearvla.mainline.role_values import ADDRESS_ONLY_ROLE, CONTEXTUAL_ROLE_VALUES
from clearvla.mainline.config import config_from_mapping
from test_causal_chain_integration import config as first_config


def config(amp=False):
    c=first_config(amp)
    return replace(c,top=replace(c.top,task_role_value_mode=CONTEXTUAL_ROLE_VALUES))


def inputs(seed=1501):
    torch.manual_seed(seed)
    q=torch.randn(2,6,16,requires_grad=True)
    v=torch.randn(2,1,3,1,16).expand(-1,4,-1,2,-1).clone().requires_grad_()
    k=torch.randn(2,4,3,2,16,requires_grad=True)
    legal=torch.ones(2,4,3,2,dtype=torch.bool)
    mass=torch.tensor([[.1,.6,.1],[.2,.1,.4]])
    return q,k,v,legal,mass


@pytest.mark.parametrize('amp',[False,True])
def test_zero_gain_exact_reference_and_all_old_gradients(amp):
    torch.manual_seed(1502);old=ObjectRoleRead(16,4)
    torch.manual_seed(1502);new=ObjectRoleRead(16,4,role_value_mode=CONTEXTUAL_ROLE_VALUES)
    for n,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[n],rtol=0,atol=0)
    args=inputs();args2=tuple(x.detach().clone().requires_grad_(x.requires_grad) if isinstance(x,torch.Tensor) else x for x in args)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        y=old(*args);z=new(*args2)
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    seed=torch.randn_like(y)
    (y*seed).sum().backward();(z*seed).sum().backward()
    for a,b in zip(args,args2):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    for n,p in old.named_parameters():torch.testing.assert_close(p.grad,dict(new.named_parameters())[n].grad,rtol=0,atol=0)
    assert new.contextual_role_gain.grad is not None and new.contextual_role_gain.grad.abs().sum()>0


@pytest.mark.parametrize('amp',[False,True])
def test_homogeneous_values_no_longer_remove_query_sensitivity(amp):
    torch.manual_seed(1504);m=ObjectRoleRead(16,4,role_value_mode=CONTEXTUAL_ROLE_VALUES)
    q,k,v,s,b=inputs()
    with torch.no_grad():m.contextual_role_gain.fill_(.4)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        y=m(q,k,v,s,b);z=m(-q,k,v,s,b)
    assert (y-z).float().abs().max()>1e-4
    grads=torch.autograd.grad(y.float().square().sum(),(q,v,m.query.weight,m.contextual_role_gain))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)


@pytest.mark.parametrize('empty',['values','mass','support'])
def test_no_source_no_action_value(empty):
    m=ObjectRoleRead(16,4,role_value_mode=CONTEXTUAL_ROLE_VALUES)
    with torch.no_grad():m.contextual_role_gain.fill_(.8)
    q,k,v,s,b=inputs()
    if empty=='values':v=torch.zeros_like(v)
    if empty=='mass':b=torch.zeros_like(b)
    if empty=='support':s.zero_();v=torch.full_like(v,float('nan'));k=torch.full_like(k,float('nan'))
    out=m(q,k,v,s,b)
    assert torch.isfinite(out).all() and torch.count_nonzero(out)==0


def test_permutations_mass_and_invalid_value_quarantine():
    m=ObjectRoleRead(16,4,role_value_mode=CONTEXTUAL_ROLE_VALUES)
    with torch.no_grad():m.contextual_role_gain.fill_(.2)
    q,k,v,s,b=inputs();s[:,:,1,0]=False
    k=k.detach().clone();v=v.detach().clone();k[:,:,1,0]=float('nan');v[:,:,1,0]=float('nan')
    y=m(q,k,v,s,b)
    perm=torch.tensor([2,0,1]);rev=torch.tensor([1,0])
    z=m(q,k[:,:,perm][:,:,:,rev],v[:,:,perm][:,:,:,rev],s[:,:,perm][:,:,:,rev],b[:,perm])
    torch.testing.assert_close(y,z,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(m(q,k,v,s,b*.1),y*.1,atol=1e-7,rtol=2e-6)


def test_selection_is_explicit_and_old_serialization_preserved():
    assert 'task_role_value_mode' not in first_config().as_dict()['top']
    assert config_from_mapping(config().as_dict())==config()
    with pytest.raises(ValueError):replace(config(),top=replace(config().top,task_role_value_mode='random')).validate()
    with pytest.raises(ValueError):replace(config(),top=replace(config().top,task_execution_mode='none')).validate()
