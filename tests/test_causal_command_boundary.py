"""Existing binary transition budget must include source boundary -> row zero.

This coverage repair does NOT certify hard command persistence near p=0.5.
"""
from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_calvin_selective_frame_weighting import _adapter, _config
from clearvla.mainline.interfaces import ActionSupervision
from clearvla.mainline.supervision import FutureLabelSupport
from clearvla.mainline.training.losses import FlowMatchingState, action_terms


def terms(probability, *, target_positive=True, boundary_positive=True, valid=None, selective=False):
    cfg=_config(selective=selective);adapter=_adapter()
    b,t=probability.shape
    raw=torch.zeros(b,t,7);raw[...,6]=1.0 if target_positive else -1.0
    raw[:,5,0]=.2
    boundary=torch.zeros(b,7);boundary[:,6]=1.0 if boundary_positive else -1.0
    target=ActionSupervision(normalized=raw,raw_units=raw,current_raw_units=boundary,
        gripper_transition_boundary=boundary,gripper_transition_boundary_raw_units=boundary,
        support=None if valid is None else FutureLabelSupport(
            action=valid, state=valid.clone(), visual=valid[:, [3, 7, 15, 23]].clone()))
    field=adapter.encode(raw,boundary)
    zero=torch.zeros_like(field)
    prediction=(field+torch.linspace(0,.3,t)[None,:,None]).detach().requires_grad_()
    logit=torch.logit(probability).detach().requires_grad_()
    command=torch.stack((torch.zeros_like(logit),logit),-1)
    output=SimpleNamespace(bottom=SimpleNamespace(physical_velocity=prediction,
        motion_logits=torch.zeros(b,t,requires_grad=True),gripper_command_logits=command,decoder_tensors={}))
    history=SimpleNamespace(action_state=boundary,codec_gripper_boundary=boundary[:,-1:])
    flow=FlowMatchingState(time=torch.zeros(b),source_physical_noise=zero,noisy_physical=zero,
        target_physical=field,target_physical_velocity=field,row_valid=valid)
    result=action_terms(cfg,adapter,output,target,history,flow)
    return result,logit


def test_constant_wrong_plan_has_a_boundary_loss_not_zero_interior_only_loss():
    # All predicted rows agree, so the old interior delta term is exactly zero.
    result,logit=terms(torch.full((1,24),.9),target_positive=False,boundary_positive=False)
    assert result['gripper_command_transition']>0
    result['gripper_command_transition'].backward()
    assert logit.grad is not None and logit.grad[0,0]>0
    assert result['gripper_command_boundary_transition_rate']==1
    assert result['gripper_command_boundary_target_rate']==0
    assert result['gripper_command_transition_rate']==0


@pytest.mark.parametrize('sign',[False,True])
def test_correct_persistent_plan_and_event_respect_source_boundary(sign):
    result,_=terms(torch.full((1,24),1-1e-6 if sign else 1e-6),
                   target_positive=sign,boundary_positive=not sign)
    assert result['gripper_command_transition']<1e-8
    assert result['gripper_command_boundary_transition_rate']==1
    assert result['gripper_command_boundary_target_rate']==1
    assert result['gripper_command_transition_rate']==0


def test_only_real_row_zero_retains_gradient_with_one_row_support():
    valid=torch.zeros(1,24,dtype=torch.bool);valid[:,0]=True
    result,logit=terms(torch.full((1,24),.8),target_positive=False,boundary_positive=False,valid=valid)
    result['gripper_command_transition'].backward()
    assert logit.grad[0,0]>0 and torch.count_nonzero(logit.grad[:,1:])==0
    assert torch.isfinite(logit.grad).all()


def test_near_half_alternation_is_visible_even_with_small_soft_delta():
    p=torch.full((1,24),.5001);p[:,1::2]=.4999
    result,_=terms(p,target_positive=True,boundary_positive=True)
    assert result['gripper_command_transition_rate']==1
    assert result['gripper_command_expected_flip_rate']>.49
    assert result['gripper_command_logit_margin_mean']<.001


def test_native_selected_metric_preserves_row_weights_when_support_is_all_real():
    probability=torch.full((1,24),.8)
    a,_=terms(probability,selective=True)
    b,_=terms(probability,valid=torch.ones(1,24,dtype=torch.bool),selective=True)
    torch.testing.assert_close(a['action_flow_native'],b['action_flow_native'],atol=1e-7,rtol=1e-6)
    # This repaired metric is not silently substituted for the formal loss.
    torch.testing.assert_close(a['action_flow'],b['action_flow'],atol=1e-7,rtol=1e-6)
