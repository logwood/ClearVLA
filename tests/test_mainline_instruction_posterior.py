"""Actual G3/S/P3 soft source and prepared reads; no skill-success claims."""
from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_mainline_endpoint_supervision import _config as base_config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.instruction_change import (
    POSTERIOR_REFERENCE_CHANGE,
    TYPED_REFERENCE_CHANGE,
    instruction_change_metadata,
)
from clearvla.mainline.model.instruction_change import TypedChangeValueRead
from clearvla.mainline.model.instruction_posterior import (
    PosteriorChangeValueRead,
    PosteriorInstructionChangePlanRead,
    PosteriorInstructionReferenceRead,
)
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.runtime.sampling import sample_action


def _config():
    c=base_config()
    return replace(c,top=replace(c.top,instruction_change_mode=POSTERIOR_REFERENCE_CHANGE))


@pytest.fixture(scope="module")
def production():
    torch.manual_seed(8221)
    m,e=_model_engine(_config())
    b=_batch()
    e.train_step(b,collect_diagnostics=False)
    m.eval()
    with torch.no_grad():
        cache,st,_=m.encode_online(b.online)
    return m,e,b,cache,st


def _values(names=("top","wrist")):
    return PosteriorChangeValueRead(hidden=16,content_dim=16,state_dim=10,camera_names=names)


def _plan():
    return PosteriorInstructionChangePlanRead(hidden=16,content_dim=16,state_dim=10,camera_names=("top","wrist"))


def _ev(production):
    ev=production[3].top.intent.instruction_change
    assert ev is not None and ev.posterior is not None
    return ev


def _with_laws(ev,pnow,pref,rawnow=None,rawref=None):
    """Synthetic observed measures for isolating the actual downstream reader."""
    p=ev.posterior
    assert p is not None
    now=p.current if rawnow is None else rawnow
    ref=p.reference if rawref is None else rawref
    nullnow=1-pnow.sum(-1)
    nullref=1-pref.sum(-1)
    post=replace(p,current=now,reference=ref,current_probability=pnow,reference_probability=pref,
        source_probability=torch.full_like(pnow,1/pnow.shape[-1]),
        current_null=nullnow,reference_null=nullref)
    with torch.autocast('cpu',enabled=False):
        content=torch.einsum('bkcn,bcnd->bkcd',pnow,now.float())-torch.einsum('bkcn,bcnd->bkcd',pref,ref.float())
        image=torch.einsum('bkcn,nd->bkcd',pnow-pref,p.coordinates)
    return replace(ev,posterior=post,reference=replace(ev.reference,dino=ref),content_delta=content,
        image_delta=image,robot_delta=torch.zeros_like(ev.robot_delta))


def test_config_abi_source_and_fact_owners(production):
    c=_config()
    c.validate()
    assert config_from_mapping(c.as_dict())==c
    assert load_config('configs/mainline/structural_rebuild_g3_posterior_calvin.json').top.instruction_change_mode==POSTERIOR_REFERENCE_CHANGE
    assert instruction_change_metadata(('top','wrist'))['mode']==TYPED_REFERENCE_CHANGE
    md=instruction_change_metadata(('top','wrist'),POSTERIOR_REFERENCE_CHANGE)
    assert md['mode']==POSTERIOR_REFERENCE_CHANGE
    paths=dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert 'clearvla/mainline/model/instruction_posterior.py' in paths
    assert 'clearvla/mainline/instruction_posterior.py' in paths
    _,_,b,cache,st=production
    ev=_ev(production)
    assert ev.posterior.source is st.top.facts.current_image_source
    assert ev.posterior.source.spatial is st.top.facts.dense_chart.current_image_support
    assert ev.posterior.current is not ev.posterior.reference
    assert ev.posterior.reference is b.online.instruction_reference.dino
    assert ev.posterior.current_probability.shape[-1]==b.online.observation.dino_history.shape[-2]
    assert cache.top.intent.instruction_plan_values.evidence is ev
    assert cache.top.intent.policy_dock().instruction_plan_values is cache.top.intent.instruction_plan_values


@pytest.mark.parametrize('field,value',[
    ('entity_chart_mode','query_lattice_v1'),('instruction_reference_mode','none'),
    ('p3_coordination_mode','pointwise_legacy_v1'),('target_binding_mode','reader_local_v1')])
def test_incoherent_graph_rejected(field,value):
    c=_config()
    with pytest.raises(ValueError):
        replace(c,top=replace(c.top,**{field:value})).validate()


def test_matcher_is_task_free_and_has_learned_null():
    assert set(inspect.signature(PosteriorInstructionReferenceRead.forward).parameters)=={
        'self','reference','current_dino','current_state','facts','binding'}
    r=PosteriorInstructionReferenceRead(hidden=16,content_dim=16,state_dim=10,camera_names=('top','wrist'))
    assert 'null_key' in dict(r.named_parameters())


@pytest.mark.parametrize('case',['identical','partial','missing','robot'])
@pytest.mark.parametrize('bf16',[False,True])
def test_zero_measured_change_and_missing_support(production,case,bf16):
    m,_,b,cache,st=production
    r=m.intent.organizer.instruction_progress
    raw=b.online.observation.dino_history[:,-1]
    ref=b.online.instruction_reference
    mask=ref.observed.clone()
    if case=='partial':
        mask[...,::2]=False
    if case=='missing':
        mask[:]=False
    start=torch.where(mask[...,None],raw,torch.full_like(raw,float('nan')))
    ref=replace(ref,dino=start,observed=mask,state=b.online.history.state-(.2 if case=='robot' else 0))
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        _,ev=r(reference=ref,current_dino=raw,current_state=b.online.history.state,
            facts=st.top.facts,binding=cache.top.intent.target_binding)
        plan=_plan()
        prepared=plan.prepare(ev)
        assert prepared.joint is not None
        assert prepared.content.count_nonzero()==prepared.image.count_nonzero()==prepared.joint.count_nonzero()==0
        assert ev.posterior.current_probability.dtype==torch.float32
        assert torch.isfinite(prepared.status).all()
        q=torch.randn(1,24,3,16,requires_grad=True)
        out=plan(ev,q,prepared)
    if case!='robot':
        out.float().sum().backward()
        assert out.count_nonzero()==0 and q.grad is not None and q.grad.count_nonzero()==0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in plan.parameters())
    else:
        assert prepared.robot.abs().sum()>0
    if case=='missing':
        assert ev.posterior.current_probability.count_nonzero()==0
        assert ev.posterior.source_probability.count_nonzero()==0
        assert ev.view_weight.count_nonzero()==0


def test_equal_centers_entropies_are_not_equal_distributions(production):
    ev=_ev(production)
    p=ev.posterior
    n=p.current_probability.shape[-1]
    side=math.isqrt(n)
    a=torch.zeros_like(p.current_probability)
    b=torch.zeros_like(a)
    a[...,0]=a[...,-1]=.5
    b[...,side-1]=b[...,n-side]=.5
    raw=torch.zeros_like(p.current)
    contrast=_with_laws(ev,a,b,raw,raw)
    same=_with_laws(ev,a,a,raw,raw)
    assert contrast.content_delta.count_nonzero()==contrast.image_delta.count_nonzero()==0
    def entropy(law):
        return -(law*torch.where(law>0,law,1.).log()).sum(-1)
    torch.testing.assert_close(entropy(a),entropy(b),rtol=0,atol=0)
    # Old consumer sees the same zero content/centroid and identical entropy/status.
    old=TypedChangeValueRead(hidden=16,content_dim=16,state_dim=10,camera_names=ev.camera_names)
    old_a=replace(contrast,posterior=None,match_status=contrast.match_status[...,:6])
    old_b=replace(same,posterior=None,match_status=same.match_status[...,:6])
    assert all(torch.equal(x,y) for x,y in zip(old(old_a),old(old_b)))
    r=_values()
    # Explicit learned-layer fixture, not a production fixed coordinate rule.
    first, last = r.image[0], r.image[2]
    assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        first.weight.zero_()
        first.weight[0]=torch.tensor([1.,1.])
        last.weight.zero_()
        last.weight[0,0]=1.
    actual=r.prepare(contrast).image
    assert actual.abs().sum()>0 and r.prepare(same).image.count_nonzero()==0


def test_joint_read_preserves_content_position_dependence(production):
    ev=_ev(production)
    p=ev.posterior
    law=torch.zeros_like(p.current_probability)
    law[...,0]=law[...,-1]=.5
    raw=torch.zeros_like(p.current)
    raw[...,0,0]=-1.
    raw[...,-1,0]=1.
    start=-raw
    e=_with_laws(ev,law,law,raw,start)
    r=_values()
    with torch.no_grad():
        r.joint_content.weight.zero_()
        r.joint_content.weight[0,0]=1.
        r.joint_image.weight.zero_()
        r.joint_image.weight[0,0]=1.
        r.joint_output.weight.zero_()
        r.joint_output.weight[0,0]=1.
    v=r.prepare(e)
    assert v.content.count_nonzero()==v.image.count_nonzero()==0
    assert v.joint is not None and v.joint.abs().sum()>0


@pytest.mark.parametrize('scale',[0.,.01,.5,1.])
def test_target_null_does_not_force_real_evidence(production,scale):
    ev=_ev(production)
    r=_values()
    orig=r(ev)
    mass=ev.binding.mass*scale
    law=TargetBinding(torch.cat((mass,1-mass.sum(-1,keepdim=True)),-1).log(),ev.binding.supported)
    e=replace(ev,binding=law)
    for a,b in zip(orig,r(e)):
        torch.testing.assert_close(b,a*scale,rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('scale',[0.,.01,.5,1.])
def test_match_null_is_not_renormalized_away(production,scale):
    ev=_ev(production)
    p=ev.posterior
    r=_values()
    base=_with_laws(ev,p.current_probability,p.reference_probability)
    changed=_with_laws(ev,p.current_probability*scale,p.reference_probability*scale)
    a,b=r.prepare(base),r.prepare(changed)
    for name in ('content','image','joint'):
        torch.testing.assert_close(getattr(b,name),getattr(a,name)*scale,rtol=3e-5,atol=2e-7)


def test_K_permutation_and_prepared_dock(production):
    ev=_ev(production)
    r=_values()
    k=torch.tensor([2,0,3,1])
    expected=r(ev)
    other=ev.permute(k,ev.binding.permute(k))
    for a,b in zip(expected,r(other)):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    intent=production[3].top.intent
    new=intent.permute(k)
    new.validate(horizon=24,hidden=intent.object_tokens.shape[-1])
    assert new.instruction_plan_values.evidence is new.instruction_change
    assert new.instruction_plan_values.reader_identity==intent.instruction_plan_values.reader_identity


def test_named_camera_permutation(production):
    ev=_ev(production)
    p=ev.posterior
    r=_values()
    other=_values(('wrist','top'))
    other.load_state_dict(r.state_dict())
    c=torch.tensor([1,0])
    s=p.source
    sp=s.spatial
    new_sp=replace(sp,coordinates=sp.coordinates[:,c],probability=sp.probability[:,c],
        valid=sp.valid[:,c],log_probability=sp.log_probability[:,c])
    new_source=replace(s,log_measure=s.log_measure[:,:,c],supported=s.supported[:,:,c],spatial=new_sp)
    new_ref=replace(ev.reference,dino=p.reference[:,c],observed=ev.reference.observed[:,c])
    post=replace(p,current=p.current[:,c],reference=new_ref.dino,observed=p.observed[:,c],
        source_probability=p.source_probability[:,:,c],current_probability=p.current_probability[:,:,c],
        reference_probability=p.reference_probability[:,:,c],current_null=p.current_null[:,:,c],
        reference_null=p.reference_null[:,:,c],source=new_source)
    e=replace(ev,content_delta=ev.content_delta[:,:,c],image_delta=ev.image_delta[:,:,c],
        match_status=ev.match_status[:,:,c],view_observed=ev.view_observed[:,:,c],view_weight=ev.view_weight[:,:,c],
        camera_names=('wrist','top'),reference=new_ref,posterior=post)
    for a,b in zip(r(ev),other(e)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='chart|camera'):
        r(e)


@pytest.mark.parametrize('bad',['probability_dtype','axis','null_axis','negative','mass','nan','source','reference'])
def test_typed_evidence_rejects_wrong_payload(production,bad):
    e=_ev(production)
    p=e.posterior
    changes={
        'probability_dtype':{'current_probability':p.current_probability.bfloat16()},
        'axis':{'current_probability':p.current_probability[...,:-1]},
        'null_axis':{'current_null':p.current_null[...,None]},
        'negative':{'current_null':p.current_null-2},
        'mass':{'current_null':p.current_null+1},
        'nan':{'current_probability':p.current_probability*float('nan')},
        'source':{'source':replace(p.source,log_measure=p.source.log_measure[:,:1])},
        'reference':{'reference':p.reference.clone()},
    }
    with pytest.raises((ValueError,TypeError)):
        replace(e,posterior=replace(p,**changes[bad])).validate(strict=True)


def test_cache_must_use_same_reader_and_evidence(production):
    e=_ev(production)
    r=_plan()
    v=r.prepare(e)
    q=torch.randn(1,24,3,16)
    with pytest.raises(ValueError,match='prepared'):
        r(e,q)
    with pytest.raises(ValueError,match='source or reader'):
        r(replace(e),q,v)
    with pytest.raises(ValueError,match='source or reader'):
        _plan()(e,q,v)
    cache=production[3]
    for field,value in [('instruction_plan_values',None),('instruction_change',replace(e))]:
        bad=replace(cache,top=replace(cache.top,intent=replace(cache.top.intent,**{field:value})))
        with pytest.raises(ValueError):
            bad.validate(_config())


def test_P3_never_reopens_visual_bank(production,monkeypatch):
    m,_,_,cache,_=production
    r=m.policy_compiler.plan_compiler.instruction_change_read
    def forbidden(*a,**kw):
        raise AssertionError('per-ODE image projection')
    for layer in (r.values.content,r.values.image,r.values.joint_content,r.values.joint_image,r.values.joint_output):
        monkeypatch.setattr(layer,'forward',forbidden)
    with torch.no_grad():
        for time in (0.,.2,1.):
            out=m.velocity(cache,noisy_action_field=torch.randn(1,24,18),time=torch.tensor([time]))
            assert torch.isfinite(out.bottom.physical_velocity).all()


def test_sample_prepares_once_and_is_pure(production,monkeypatch):
    m,_,b,_,_=production
    count={'match':0,'prepare':0,'world':0}
    reader=m.intent.organizer.instruction_progress
    plan=m.policy_compiler.plan_compiler.instruction_change_read
    for mod,name,key in ((reader,'forward','match'),(plan,'prepare','prepare'),(m.world,'materialize','world')):
        assert hasattr(mod,name)
        f=getattr(mod,name)
        def wrap(*a,_f=f,_key=key,**kw):
            count[_key]+=1
            return _f(*a,**kw)
        monkeypatch.setattr(mod,name,wrap)
    buffers={k:v.clone() for k,v in m.named_buffers()}
    with torch.no_grad():
        a=sample_action(m,b.online,m.config,generator=torch.Generator().manual_seed(99))
        first=dict(count)
        z=sample_action(m,b.online,m.config,generator=torch.Generator().manual_seed(99))
    assert first['match']==first['prepare']==1
    assert count['match']==count['prepare']==2
    assert first['world']==2 and count['world']==4
    torch.testing.assert_close(a.action,z.action,rtol=0,atol=0)
    assert all(torch.equal(v,buffers[k]) for k,v in m.named_buffers())


@pytest.mark.parametrize('bf16',[False,True])
def test_actual_update_row_zero_gradients_and_G3_owner(bf16):
    torch.manual_seed(8823)
    m,e=_model_engine(_config())
    b=_batch()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        e.train_step(b,collect_diagnostics=False)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        cache,st,_=m.encode_online(b.online)
        src=st.top.facts.current_image_source
        assert src is not None
        src.log_measure.retain_grad()
        src.spatial.log_probability.retain_grad()
        out=m.velocity(cache,noisy_action_field=torch.randn(1,24,18),time=torch.tensor([.4]))
        loss=out.bottom.physical_velocity[:,0].float().square().mean()
    loss.backward()
    for module in (m.intent.organizer.instruction_progress,m.policy_compiler.plan_compiler.instruction_change_read):
        assert module is not None
        for name,p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    for tensor in (src.log_measure,src.spatial.log_probability):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum()>0
    assert all(p.grad is None for p in m.training_targets.parameters())


def test_intervention_on_each_prepared_type_reaches_first_action(production):
    m,_,_,cache,_=production
    intent=cache.top.intent
    v=intent.instruction_plan_values
    field=torch.randn(1,24,18)
    t=torch.tensor([.4])
    with torch.no_grad():
        baseline=m.velocity(cache,noisy_action_field=field,time=t).bottom.physical_velocity[:,0]
        for name in ('content','image','joint','robot'):
            changed=replace(v,**{name:getattr(v,name)+.25})
            alternate=replace(cache,top=replace(cache.top,intent=replace(intent,instruction_plan_values=changed)))
            out=m.velocity(alternate,noisy_action_field=field,time=t).bottom.physical_velocity[:,0]
            assert not torch.equal(out,baseline),name


def test_current_G3_law_changes_matching_not_target_logits(production):
    m,_,b,cache,st=production
    facts=st.top.facts
    s=facts.current_image_source
    shifted=s.log_measure+torch.linspace(-2,2,s.log_measure.numel()).reshape_as(s.log_measure)
    other=replace(facts,current_image_source=replace(s,log_measure=shifted))
    r=m.intent.organizer.instruction_progress
    kw=dict(reference=b.online.instruction_reference,current_dino=b.online.observation.dino_history[:,-1],
        current_state=b.online.history.state,binding=cache.top.intent.target_binding)
    with torch.no_grad():
        _,a=r(**kw,facts=facts)
        _,z=r(**kw,facts=other)
    assert not torch.equal(a.posterior.source_probability,z.posterior.source_probability)
    assert not torch.equal(a.posterior.reference_probability,z.posterior.reference_probability)
    torch.testing.assert_close(a.binding.log_probability,z.binding.log_probability,rtol=0,atol=0)


def test_future_label_work_leaves_posterior_unchanged(production):
    m,_,b,cache,st=production
    p=_ev(production).posterior
    fields=('source_probability','current_probability','reference_probability','current_null','reference_null')
    original={k:getattr(p,k).clone() for k in fields}
    with torch.no_grad():
        m.build_training_targets(st,replace(b.future,action_sequence=b.future.action_sequence+2))
    assert all(torch.equal(getattr(p,k),v) for k,v in original.items())


def test_exact_checkpoint_and_mode_owned_ABI(tmp_path,monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    original=old.build_deployment_abi
    def build(*a,**kw):
        abi=original(*a,**kw)
        assert abi['instruction_change']==instruction_change_metadata(('top','wrist'),POSTERIOR_REFERENCE_CHANGE)
        for kind in ('missing','legacy','reinterpret'):
            bad=copy.deepcopy(abi)
            if kind=='missing':
                del bad['instruction_change']
            elif kind=='legacy':
                bad['instruction_change']=instruction_change_metadata(('top','wrist'))
            else:
                record=bad['instruction_change']
                assert isinstance(record,dict)
                record['limits']='physical success'
            with pytest.raises(ValueError,match='instruction change'):
                validate_deployment_abi(bad)
        return abi
    monkeypatch.setattr(old,'build_deployment_abi',build)
    monkeypatch.setattr(old,'_config',_config)
    monkeypatch.setattr(old,'_batch',_batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


@pytest.mark.parametrize("patches,bf16", [(64, False), (144, False), (256, False), (256, True)])
def test_complete_native_lattice_reaches_posterior_and_action(patches, bf16):
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    c=_config()
    c=replace(c,dimensions=replace(c.dimensions,patches_per_camera=patches))
    torch.manual_seed(9012)
    m=ClearVLAMainlinePolicy(c).eval()
    b,n=synthetic_batch(c,count=1,raw_side=32,device=torch.device('cpu'))
    m.configure_action_normalizer(n)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        cache,st,_=m.encode_online(b.online)
        evidence=cache.top.intent.instruction_change
        assert evidence is not None
        p=evidence.posterior
        assert p is not None
        assert p.source_probability.shape==p.current_probability.shape==p.reference_probability.shape==(1,4,2,patches)
        assert p.source is st.top.facts.current_image_source
        assert p.source.spatial.coordinates.shape[-2]==patches
        assert st.observation.grounding.context_mask.shape[-2:]==(8,8)
        p.validate(strict=True)
        out=m.velocity(cache,noisy_action_field=torch.randn(1,24,18),time=torch.tensor([.4]))
    assert torch.isfinite(out.bottom.physical_velocity).all()


@pytest.mark.parametrize('bf16',[False,True])
def test_masked_payload_forward_and_weight_VJP_quarantine(production,bf16):
    ev=_ev(production)
    p=ev.posterior
    observed=p.observed.clone()
    observed[...,::2]=False
    now=p.current.clone()
    ref=p.reference.clone()
    now[...,::2,:]=float('nan')
    ref[...,::2,:]=float('nan')
    law=observed[:,None].expand_as(p.source_probability).float()
    law=law/law.sum(-1,keepdim=True)
    p=replace(p,current=now,reference=ref,observed=observed,source_probability=law,
        current_probability=law*.6,reference_probability=law*.7,
        current_null=torch.full_like(p.current_null,.4),reference_null=torch.full_like(p.reference_null,.3))
    e=replace(ev,posterior=p,reference=replace(ev.reference,dino=ref,observed=observed))
    r=_values()
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):
        out=r(e)
        loss=sum(x.float().square().mean() for x in out)
        assert isinstance(loss, torch.Tensor)
    assert all(torch.isfinite(x).all() for x in out)
    loss.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in r.parameters())
    damaged=p.current.clone()
    damaged[...,1,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite observed'):
        replace(e,posterior=replace(p,current=damaged)).validate(strict=True)


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_do_not_change_prepared_values_gradients_or_rng(bf16):
    torch.manual_seed(9081)
    model, _ = _model_engine(_config())
    model.eval()
    batch = _batch()
    results = []
    reader = model.policy_compiler.plan_compiler.instruction_change_read
    assert isinstance(reader, PosteriorInstructionChangePlanRead)
    for diagnostic in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache, _, _ = model.encode_online(batch.online, collect_diagnostics=diagnostic)
            intent = cache.top.intent
            assert intent.instruction_change is not None
            output = reader(
                intent.instruction_change,
                intent.temporal_queries[:, :, None],
                intent.instruction_plan_values,
            )
            gradients = torch.autograd.grad(output.float().square().mean(), tuple(reader.parameters()))
        results.append((output.detach(), gradients, torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    for left, right in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_isolated_new_P3_read_has_direct_G3_source_gradients():
    torch.manual_seed(9103)
    model, _ = _model_engine(_config())
    model.eval()
    cache, state, _ = model.encode_online(_batch().online)
    intent = cache.top.intent
    assert intent.instruction_change is not None
    reader = model.policy_compiler.plan_compiler.instruction_change_read
    assert isinstance(reader, PosteriorInstructionChangePlanRead)
    output = reader(
        intent.instruction_change,
        intent.temporal_queries[:, :, None].detach(),
        intent.instruction_plan_values,
    )
    source = state.top.facts.current_image_source
    assert source is not None
    # This scalar excludes the action decoder/P1/W routes. It isolates the new
    # G3-to-posterior-to-P3 read rather than borrowing an existing P1 gradient.
    gradients = torch.autograd.grad(
        output.square().mean(), (source.log_measure, source.spatial.log_probability)
    )
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
