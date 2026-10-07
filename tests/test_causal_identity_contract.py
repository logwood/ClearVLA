from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from clearvla.mainline.training.optimizer import WarmupCosineSchedule
from clearvla.mainline.train import _initialize_training_clock
from clearvla.mainline.causal_identity import causal_identity_metadata
from clearvla.mainline.runtime.causal_identity_migration import CAUSAL_IDENTITY_AB_V1
from clearvla.mainline.instruction_change import instruction_change_metadata,POSTERIOR_REFERENCE_CHANGE
from clearvla.mainline.executed_world import executed_world_metadata
from test_calvin_optimizer_moment_initialization import Tiny
from test_source_consistent_measurement import production


def test_mature_model_clock_with_fresh_optimizer_warmup_roundtrip():
    model=Tiny(2,2);opt=torch.optim.AdamW(model.parameters(),lr=8e-5)
    schedule=WarmupCosineSchedule(opt,warmup_steps=100,total_steps=1024,minimum_ratio=.1,update_origin=11012)
    engine=SimpleNamespace(global_step=0,model=model)
    source=SimpleNamespace(model_contract_migration=CAUSAL_IDENTITY_AB_V1,global_step=11012)
    _initialize_training_clock(engine,schedule,source,mode='checkpoint')
    assert model.completed==engine.global_step==schedule.step_index==11012
    assert opt.param_groups[0]['lr']==pytest.approx(8e-7)
    state=schedule.state_dict();schedule.load_state_dict(state)
    assert schedule.ratio(11111)==1.
    assert schedule.ratio(12036)==pytest.approx(.1)
    assert not opt.state
    wrong=WarmupCosineSchedule(opt,warmup_steps=100,total_steps=1024,minimum_ratio=.1)
    with pytest.raises(ValueError,match='curve identity'):wrong.load_state_dict(state)


def test_new_metadata_never_claims_learned_matcher_or_online_depth(production):
    model,*_=production
    assert causal_identity_metadata({}) is None
    meta=causal_identity_metadata(model.config.top)
    assert meta['policy_inputs']=='unchanged-RGB-language-observed-proprioception-and-executed-controls'
    assert meta['identity_supervision']['plane']=='training-labels-only'
    change=instruction_change_metadata(('top','wrist'),POSTERIOR_REFERENCE_CHANGE,'source_consistent_v1')
    assert 'learned-null' not in change['matching']
    assert 'before-proposal' in executed_world_metadata('source_consistent_v1','before_proposal_v1')['S']


def test_observed_outcome_is_independent_of_forecast_and_keeps_zero_innovation_outcome(production):
    model,_,batch,cache,state=production
    feedback=cache.world_feedback.feedback
    reader=model.intent.organizer.observed_outcome
    binding=state.top.intent.target_binding
    # Exercise trained non-neutral consumer weights without mutating fixture.
    import copy
    reader=copy.deepcopy(reader)
    torch.manual_seed(199)
    torch.nn.init.normal_(reader.output.weight,std=.05)
    with torch.no_grad():
        base=reader(feedback,state.top.facts,binding)
        forecast=replace(feedback,predicted_semantic=feedback.predicted_semantic+.25,semantic=feedback.semantic-.25)
        same=reader(forecast,state.top.facts,binding)
        torch.testing.assert_close(base,same,atol=0,rtol=0)
        changed=replace(feedback,observed_semantic=feedback.observed_semantic+.25,predicted_semantic=feedback.observed_semantic+.25,semantic=torch.zeros_like(feedback.semantic))
        zero_error=replace(feedback,predicted_semantic=feedback.observed_semantic,semantic=torch.zeros_like(feedback.semantic))
        a=reader(changed,state.top.facts,binding);b=reader(zero_error,state.top.facts,binding)
        assert (a-b).abs().max()>1e-7
