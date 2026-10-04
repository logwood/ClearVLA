"""C7 actual filter, measured source and native-head boundary contracts.

Synthetic laws are controlled examples, not a pretrained persistence guarantee.
"""
from dataclasses import replace
from itertools import product
import pytest
import torch
from clearvla.mainline.command_sequence import prepare_command_boundary
from clearvla.mainline.model.command_sequence import binary_command_filter, ConditionalBinaryCommand
from clearvla.mainline.robot_execution import ExecutedRobotStep


def oracle(emission, energy, initial):
    # Probability-domain dynamic programming, independent of production log-ratio algebra.
    probability = initial
    marginals, pairs = [], []
    for i in range(emission.shape[1]):
        law = torch.softmax(emission[:,i,None,:] + energy[:,i], -1)
        joint = probability[:,:,None] * law
        probability = joint.sum(1)
        marginals.append(probability);pairs.append(joint)
    return torch.stack(marginals,1),torch.stack(pairs,1)


def source(*, positive=True, observed=True, offset=0.,scale=1.):
    c=torch.zeros(2,7);c[:,-1]=(1 if positive else -1)*scale+offset
    o=torch.full((2,),observed,dtype=torch.bool)
    step=ExecutedRobotStep(torch.zeros(2,10),c,o,torch.tensor([[-1,-1,0] if observed else [0,0,0]]).expand(2,-1).clone())
    return step


@pytest.mark.parametrize('amp',[False,True])
@pytest.mark.parametrize('boundary',[0.,.5,1.])
def test_zero_transition_keeps_emissions_and_gradient_neutrality(amp,boundary):
    torch.manual_seed(1710)
    state=torch.randn(2,24,32,requires_grad=True)
    logits=torch.randn(2,24,2,requires_grad=True)
    module=ConditionalBinaryCommand(32)
    before=torch.get_rng_state().clone()
    initial=torch.tensor([1-boundary,boundary]).expand(2,-1)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        out,pair=module(state,logits,initial)
    torch.testing.assert_close(out,logits.float(),rtol=0,atol=0)
    assert torch.equal(before,torch.get_rng_state())
    g=torch.autograd.grad(out.square().sum(),(logits,state,module.transition_weight))
    torch.testing.assert_close(g[0],2*logits,rtol=1e-6,atol=1e-6)
    assert torch.count_nonzero(g[1])==0
    assert torch.isfinite(g[2]).all()
    assert pair.shape==(2,24,2,2) and pair.dtype==torch.float32


@pytest.mark.parametrize('amp',[False,True])
def test_full_probability_law_and_vjps_match_independent_oracle(amp):
    torch.manual_seed(1711)
    e=torch.randn(2,8,2,requires_grad=True)
    t=torch.randn(2,8,2,2,requires_grad=True)
    init=torch.tensor([[.0,1.],[.4,.6]])
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):out,j= binary_command_filter(e,t,init)
    p,k=oracle(e,t,init)
    torch.testing.assert_close(out.softmax(-1),p,atol=3e-7,rtol=2e-6)
    torch.testing.assert_close(j,k,atol=3e-7,rtol=2e-6)
    torch.testing.assert_close(j.sum(2),out.softmax(-1),atol=3e-7,rtol=2e-6)
    torch.testing.assert_close(j.sum(3),torch.cat((init[:,None],p[:,:-1]),1),atol=3e-7,rtol=2e-6)
    cot=torch.randn_like(p)
    a=torch.autograd.grad((out.softmax(-1)*cot).sum(),(e,t),retain_graph=True)
    b=torch.autograd.grad((p*cot).sum(),(e,t))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-7)


def test_enumerated_paths_agree_and_future_rows_cannot_modify_prefix():
    torch.manual_seed(1712);e=torch.randn(1,4,2);t=torch.randn(1,4,2,2);init=torch.tensor([[.3,.7]])
    out,_=binary_command_filter(e,t,init);exact=torch.zeros_like(e)
    conditional=(e[:,:,None,:]+t).softmax(-1)
    for chain in product(range(2),repeat=5):
        prob=init[0,chain[0]]
        for i in range(4):prob=prob*conditional[0,i,chain[i],chain[i+1]]
        for i in range(4):exact[0,i,chain[i+1]]+=prob
    torch.testing.assert_close(out.softmax(-1),exact,rtol=2e-6,atol=3e-7)
    first,_=binary_command_filter(e[:,:2],t[:,:2],init)
    torch.testing.assert_close(out[:,:2],first,atol=0,rtol=0)
    e2=e.clone();e2[:,2:]+=5;t2=t.clone();t2[:,2:]*=3
    x,_=binary_command_filter(e2,t2,init)
    torch.testing.assert_close(out[:,:2],x[:,:2],atol=0,rtol=0)
    tail,_=binary_command_filter(e[:,2:],t[:,2:],first[:,-1].softmax(-1))
    torch.testing.assert_close(out[:,2:].softmax(-1),tail.softmax(-1),atol=3e-7,rtol=2e-6)


def test_learned_transition_can_persist_or_switch_without_fixed_minimum_hold():
    e=torch.zeros(1,24,2);e[0,::2,1]=.01;e[0,1::2,0]=.01
    init=torch.tensor([[0.,1.]])
    stay=torch.eye(2)[None,None].expand(1,24,-1,-1)*6
    flip=(1-torch.eye(2))[None,None].expand(1,24,-1,-1)*6
    no,_=binary_command_filter(e,torch.zeros_like(stay),init)
    hold,_=binary_command_filter(e,stay,init)
    switch,_=binary_command_filter(e,flip,init)
    assert (no.argmax(-1)[:,1:]!=no.argmax(-1)[:,:-1]).sum()==23
    assert (hold.argmax(-1)==1).all()
    assert (switch.argmax(-1)[:,1:]!=switch.argmax(-1)[:,:-1]).sum()==23
    # Evidence can override a persistence preference immediately, even at row zero.
    strong=e.clone();strong[:,0,0]=30
    response,_=binary_command_filter(strong,stay,init)
    assert response.argmax(-1)[0,0]==0


@pytest.mark.parametrize('positive',[False,True])
def test_source_uses_inverse_normalizer_not_tcp_opening_or_future_label(positive):
    s=source(positive=positive,offset=.7,scale=2.3)
    b=prepare_command_boundary(s,offset=torch.tensor(.7),scale=torch.tensor(2.3),fingerprint='checked')
    assert (b.probability[:,int(positive)]==1).all()
    b.validate(s,'checked')
    with pytest.raises(ValueError,match='another executed'):b.validate(replace(s,command=s.command.clone()))
    with pytest.raises(ValueError,match='normalizer'):b.validate(s,'changed')
    s.command.add_(.1)
    with pytest.raises(ValueError,match='mutated'):b.validate(s)


def test_reset_unknown_is_uniform_and_invalid_observed_source_is_rejected():
    s=source(observed=False);s.command.fill_(float('nan'))
    b=prepare_command_boundary(s,offset=torch.tensor(.3),scale=torch.tensor(.8),fingerprint='a')
    torch.testing.assert_close(b.probability,torch.full((2,2),.5),rtol=0,atol=0)
    for kind in ['clock','nonfinite','nonbinary']:
        s=source()
        if kind=='clock':s.offsets[:,0]=-2
        elif kind=='nonfinite':s.command[0,-1]=float('nan')
        else:s.command[0,-1]=.2
        with pytest.raises(ValueError):prepare_command_boundary(s,offset=torch.tensor(0.),scale=torch.tensor(1.),fingerprint='b')


def test_no_rng_or_extra_tensor_state_and_stable_extreme_logits():
    state=torch.get_rng_state().clone();m=ConditionalBinaryCommand(32)
    assert torch.equal(state,torch.get_rng_state()) and len(list(m.parameters()))==1
    e=torch.tensor([[[1000.,-1000.],[-1000.,1000.],[0.,0.]]],requires_grad=True)
    t=torch.full((1,3,2,2),.03,requires_grad=True)
    out,p=binary_command_filter(e,t,torch.tensor([[0.,1.]]))
    assert torch.isfinite(out).all() and torch.isfinite(p).all()
    out.square().mean().backward();assert torch.isfinite(e.grad).all() and torch.isfinite(t.grad).all()
