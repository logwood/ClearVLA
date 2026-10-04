"""Real shared binder: condition REAL-K ranking, not merely real/null mass."""
from __future__ import annotations
from copy import deepcopy

import pytest
import torch
from torch import nn
from clearvla.mainline.model.task_execution import TaskConditionedTargetBinder
from clearvla.mainline.task_execution import task_execution_metadata


def isolated_binder(h=8):
    torch.manual_seed(811)
    model = TaskConditionedTargetBinder(h, 2)
    # Explicit isolation, not pretrained weights: the old nonlinear branch is
    # zeroed so that only the repaired residual's expressiveness is tested.
    with torch.no_grad():
        model.objects.weight.copy_(torch.eye(h))
        model.task_value.weight.copy_(torch.eye(h))
        model.score.weight.zero_()
        model.task_object_score.weight.zero_()
        model.task_object_score.weight[0, 0] = 1
    return model


@pytest.mark.parametrize('amp',[False,True])
def test_task_can_reverse_conditional_object_ranking(amp):
    m=isolated_binder()
    objects=torch.zeros(1,3,8);objects[0,0,0]=1;objects[0,1,0]=-1
    task=torch.zeros(1,4,8);task[...,0]=1
    supported=torch.tensor([[True,True,False]])
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        a=m(task,objects,supported,history=torch.zeros(1,8))
        b=m(-task,objects,supported,history=torch.zeros(1,8))
    assert a.log_probability[0,0]>a.log_probability[0,1]
    assert b.log_probability[0,0]<b.log_probability[0,1]
    assert a.mass[0,2]==0 and b.mass[0,2]==0


@pytest.mark.parametrize('amp',[False,True])
def test_rank_gradient_reaches_task_and_both_cross_halves(amp):
    torch.manual_seed(819)
    m=TaskConditionedTargetBinder(8,2)
    with torch.no_grad():
        m.score.weight.zero_()
        m.task_object_score.weight.fill_(.125)
    task=torch.randn(2,4,8,requires_grad=True)
    objects=torch.randn(2,3,8,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
        b=m(task,objects,torch.ones(2,3,dtype=torch.bool),history=torch.randn(2,8))
        odds=(b.log_probability[:,0]-b.log_probability[:,1]).sum()
    odds.backward()
    assert task.grad is not None and task.grad.abs().sum()>1e-5
    assert objects.grad is not None and objects.grad.abs().sum()>1e-5
    grad=m.task_object_score.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad[:,:8].abs().sum()>1e-5 and grad[:,8:].abs().sum()>1e-5
    # Odds remove the null normalization, so this cannot pass via null alone.
    assert m.null.weight.grad is None or m.null.weight.grad.abs().max()<1e-5


def test_zero_start_keeps_original_binder_value_and_rng():
    torch.manual_seed(827)
    m=TaskConditionedTargetBinder(8,2)
    task,objects,history=torch.randn(2,4,8),torch.randn(2,3,8),torch.randn(2,8)
    support=torch.ones(2,3,dtype=torch.bool)
    before=torch.get_rng_state().clone()
    a=m(task,objects,support,history=history)
    saved=[]
    # At the exact zero start, replacing only the residual with zero is the
    # original neutral nonlinear binder, with the SAME registered parameters.
    h=m.task_object_score.register_forward_hook(lambda mod,args,out: out*0)
    try:b=m(task,objects,support,history=history)
    finally:h.remove()
    torch.testing.assert_close(a.log_probability,b.log_probability,atol=0,rtol=0)
    assert torch.equal(before,torch.get_rng_state())
    assert list(m.task_object_score.weight.shape)==[1,16]


def test_nonzero_weights_do_not_manufacture_identity_from_zero_task():
    m=isolated_binder()
    with torch.no_grad():m.task_object_score.weight.normal_()
    law=m(torch.zeros(2,4,8),torch.randn(2,3,8),torch.ones(2,3,dtype=torch.bool),history=torch.randn(2,8))
    torch.testing.assert_close(law.mass,law.mass[:,:1].expand_as(law.mass),atol=0,rtol=0)


@pytest.mark.parametrize('views',[False,True])
def test_supported_object_permutation_and_missing_values(views):
    torch.manual_seed(831)
    m=isolated_binder();task=torch.randn(2,4,8);history=torch.randn(2,8)
    support=torch.tensor([[True,True,False],[True,False,True]])
    x=torch.randn(2,3,2,8) if views else torch.randn(2,3,8)
    x[~support]=float('nan')
    v=support[:,:,None].expand(-1,-1,2).clone() if views else None
    a=m(task,x,support,history=history,view_support=v)
    perm=torch.tensor([2,0,1])
    b=m(task,x[:,perm],support[:,perm],history=history,view_support=None if v is None else v[:,perm])
    torch.testing.assert_close(a.mass[:,perm],b.mass,atol=2e-7,rtol=2e-6)
    assert torch.isfinite(a.mass).all() and not a.mass[~support].any()
    if views:
        c=m(task,x.flip(2),support,history=history,view_support=v.flip(2))
        torch.testing.assert_close(c.mass,a.mass,atol=2e-7,rtol=2e-6)


def test_old_metadata_cannot_describe_repaired_weight_meaning():
    metadata=task_execution_metadata()
    assert metadata['binding_residual']['schema']=='pointwise-task-object-products-v1'
    old=deepcopy(metadata);del old['binding_residual']
    assert old!=metadata
