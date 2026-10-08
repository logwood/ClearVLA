"""Check opt-in address memory on fixed B-v1 actual canonical candidates.

Current causal RGB/history inputs; no losses, optimizer, action intervention,
or features saved to disk. Physical masks score results only. Upstream BF16
preprocessing is retained; isolated canonical G uses FP32 for repeatable tracing.
This is not a reproduction of official CUDA actions or proof of a repair.
"""
import argparse, hashlib, json, copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.model import canonical_grounding as canonical
from clearvla.vision.entity_chart import CanonicalImageReadSource
import clearvla.simulation.clearvla_policy as frontend

class Capture(Exception): pass
def dump(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
def fingerprint(module):
    h=hashlib.sha256()
    for name,p in module.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def variation(value,dim):
    x=value.detach().float()
    common=x.mean(dim,keepdim=True)
    energy=x.square().mean()
    return dict(rms=float(energy.sqrt()),
        centered_rms=float((x-common).square().mean().sqrt()),
        common_energy_fraction=float(common.square().mean()/energy.clamp_min(1e-30)))
def physical(log,mass,legal,shape,spatial,masks,names):
    c,h,w=shape;k=log.shape[-1]-1
    law=log.reshape(1,c,h,w,k+1).permute(0,4,1,2,3)
    support=legal.reshape(1,1,c,h,w).expand(-1,k,-1,-1,-1)
    lm=torch.where(mass>0,mass,1.).log().reshape(1,1,c,h,w)
    src=CanonicalImageReadSource(torch.where(support,law[:,:k]+lm,0.),support,spatial)
    rows=[]
    for cam,mask in enumerate(masks):
        measure=src.on_image(rows=mask.shape[-2],columns=mask.shape[-1])
        joint,_=measure.normalized((1,2,3,4))
        weight=torch.as_tensor(mask,dtype=joint.dtype,device=joint.device)
        regional=torch.einsum('khw,ohw->ok',joint[0,:,cam],weight)
        total=regional.sum(-1);q=regional/total[:,None].clamp_min(torch.finfo(regional.dtype).tiny)
        pairs=[]
        for i in range(len(names)):
            for j in range(i):
                if total[i]>0 and total[j]>0:
                    pairs.append(dict(objects=[names[j],names[i]],tv=float(.5*(q[i]-q[j]).abs().sum())))
        rows.append(dict(camera=cam,object_mass=total.tolist(),K_given_object=q.tolist(),pairs=pairs))
    return rows
def read_stats(read,slots,candidates):
    pairs=[float(.5*(read[:,i]-read[:,j]).abs().sum(-1).mean())
           for i in range(read.shape[1]) for j in range(i)]
    update=read@candidates.float()
    return dict(slot_state=variation(slots,1),read_pair_tv=pairs,
                pooled_update=variation(update,1))
def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','plan','masks','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    p=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cpu'),
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
    model=p.bundle.model;g=model.grounding.grounder
    if p.bundle.config.top.identity_supervision_mode!='rgbd_temporal_v1':
        raise ValueError('This plan is anchored to B-v1 only')
    before=fingerprint(g);versions={n:q._version for n,q in model.named_parameters()}
    original_sample=frontend.sample_action;original_canonical=canonical.canonical_grounding
    original_owner=canonical.encode_owners;holder={};records=[]
    def capture_input(_model,online,*args,**kwargs):holder['online']=online;raise Capture()
    def capture_boundary(module,local,chart,history_context,**kwargs):
        holder.update(local=local,chart=chart,history_context=history_context);raise Capture()
    def capture_candidates(module,candidates,mass,legal):
        holder.update(candidates=candidates.detach().float(),mass=mass.detach(),legal=legal.detach());raise Capture()
    frontend.sample_action=capture_input;canonical.canonical_grounding=capture_boundary
    canonical.encode_owners=capture_candidates
    identity=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=p.bundle.checkpoint_sha256,
        source='3a84399926d50831470518e1b7f105131d3f1056',scope=__doc__,
        candidate_source='working-address-memory',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optimizer_updates=0,production_changed=False,mask_use='scoring-only')
    try:
        for row in json.loads(a.plan.read_text()):
            with np.load(Path(row['case'])/'trajectory.npz',allow_pickle=False) as z:
                data={k:z[k] for k in ('rgb_static','rgb_gripper','robot_obs','executed')}
            p.reset();history=CausalHistory(executed_world=True)
            for t in range(max(row['steps'])+1):
                prev=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
                obs=calvin_policy_observation({'rgb_obs':{k:data[k][t] for k in ('rgb_static','rgb_gripper')},'robot_obs':data['robot_obs'][t]},prev)
                if t==0:history.reset(obs,reset_action=prev)
                else:history.append(prev,obs)
                if t!=0 and t not in row['steps']:continue
                try:p.act_with_input(history.snapshot(),row['instruction'])
                except Capture:pass
                if t not in row['steps']:continue
                with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=p.bundle.config.runtime.compute_dtype=='bf16'):
                    try:model.encode_online(holder['online'],training_mask=False,geometry_supervision=False)
                    except Capture:pass
                local,chart,hist=holder['local'],holder['chart'],holder['history_context']
                shape=tuple(local.current_observed_content.shape[1:4])
                with np.load(a.masks/Path(row['case']).name/('state_%03d.npz'%t),allow_pickle=False) as z:
                    masks=[z['top_masks'],z['wrist_masks']];names=z['object_names'].tolist()
                fields={};hooks=[]
                def feature_hook(name):
                    def f(mod,args,out):
                        fields[name]=out.detach()
                    return f
                for name in ('content_key','semantic_key','appearance_key','geometry_key','coordinate_key','context_key','history_key','candidate_norm'):
                    mod=getattr(g,name,None)
                    if mod is not None:hooks.append(mod.register_forward_hook(feature_hook(name)))
                try:
                    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=p.bundle.config.runtime.compute_dtype=='bf16'):
                        try:original_canonical(g,local,chart,hist,collect_diagnostics=False)
                        except Capture:pass
                finally:
                    for h in hooks:h.remove()
                x,mass,legal=holder['candidates'],holder['mass'],holder['legal']
                reports={}
                for name,out in fields.items():
                    if out.ndim==2:
                        out=out[None,None].expand(1,shape[0],-1,-1)
                    value=out.float().reshape(1,shape[0],-1,out.shape[-1])
                    valid=legal.reshape(1,shape[0],-1,1).float()
                    mean=(value*valid).sum(2,keepdim=True)/valid.sum(2,keepdim=True).clamp_min(1)
                    denominator=(valid.sum()*value.shape[-1]).clamp_min(1)
                    energy=(value.square()*valid).sum()/denominator
                    centered=((value-mean).square()*valid).sum()/denominator
                    reports[name]=dict(shape=list(out.shape),supported_rms=float(energy.sqrt()),
                        within_view_centered_rms=float(centered.sqrt()),
                        centered_energy_fraction=float(centered/energy.clamp_min(1e-30)))
                fields=reports
                stages=[];updates=[];old_comp=g._competition
                def competition(*args,**kwargs):
                    out=old_comp(*args,**kwargs)
                    owner,_,_,read,log,_=out
                    stats=read_stats(read,args[0],args[1])
                    contrast=log[...,:g.objects]-log[...,:g.objects].mean(-1,keepdim=True)
                    safe=torch.where(legal[...,None],contrast,0.)
                    mean=safe.sum(1,keepdim=True)/legal.sum(1,keepdim=True)[...,None].clamp_min(1)
                    residual=torch.where(legal[...,None],contrast-mean,0.)
                    count=(legal.sum()*g.objects).clamp_min(1)
                    stats.update(index=len(stages),mean_null=float((owner[...,-1]*legal).sum()/legal.sum()),
                        K_bias_rms=float(mean.square().mean().sqrt()),
                        spatial_K_interaction_rms=float((residual.square().sum()/count).sqrt()),
                        physical_read=physical(log,mass,legal,shape,chart.current_image_support,masks,names))
                    stages.append(stats);return out
                def gru_hook(mod,args,out):updates.append(dict(kind='gru',output=variation(out.reshape(1,g.objects,-1),1)))
                def ffn_hook(mod,args,out):updates.append(dict(kind='ffn',output=variation(out,1)))
                g._competition=competition
                hooks=[g.gru.register_forward_hook(gru_hook),g.update_ffn.register_forward_hook(ffn_hook)]
                try:
                    with torch.no_grad():slots,log=original_owner(g,x,mass,legal)
                finally:
                    g._competition=old_comp
                    for h in hooks:h.remove()
                with torch.no_grad():repeat_slot,repeat_log=original_owner(g,x,mass,legal)
                assert torch.equal(slots,repeat_slot) and torch.equal(log,repeat_log)
                assert len(stages)==g.iterations+1
                test_g=copy.deepcopy(g)
                test_g.entity_address_memory_mode="conditional_logits_v1"
                test_g.address_memory_gain=torch.nn.Parameter(torch.zeros(g.iterations))
                with torch.no_grad():
                    zero_slots,zero_log=original_owner(test_g,x,mass,legal)
                assert torch.equal(slots,zero_slots) and torch.equal(log,zero_log)
                # One declared small gain for all windows: sensitivity only,
                # never a mask-chosen optimizer target or policy override.
                variants=[]
                for gain in (0.1,):
                    test_g.address_memory_gain.data.fill_(gain)
                    with torch.no_grad():vs,vl=original_owner(test_g,x,mass,legal)
                    variants.append(dict(raw_gain=gain,effective_gain=float(torch.tanh(torch.tensor(gain))),
                        physical=physical(vl,mass,legal,shape,chart.current_image_support,masks,names),
                        owner_l1_mean=float((vl.exp()-log.exp()).abs().sum(-1).mean())))
                test_g.address_memory_gain.data.zero_()
                xs=x.clone().requires_grad_()
                ss,ll=original_owner(test_g,xs,mass,legal)
                gen=torch.Generator().manual_seed(391)
                cotangent=torch.randn(ll.shape,generator=gen)
                loss=(ll.exp()*cotangent).sum()
                gp,gx=torch.autograd.grad(loss,(test_g.address_memory_gain,xs))
                assert torch.isfinite(gp).all() and torch.isfinite(gx).all()
                assert (gp.abs()>0).all()
                record=dict(case_id=row['case_id'],step=t,zero_init_exact=True,
                    synthetic_cotangent_gain_gradient=gp.tolist(),
                    gradient_scope="local G output VJP, not training-loss or action evidence",
                    small_gain_sensitivity=variants,camera_grid=shape,fields=fields,
                    candidate_spatial=variation(x,1),stages=stages,updates=updates,
                    final_physical_read=physical(log,mass,legal,shape,chart.current_image_support,masks,names),
                    exact_owner_repeat=True)
                records.append(record)
                dump(a.output/'results.json',dict(complete=False,identity=identity,records=records))
                print('STRUCTURE',row['case_id'],t,flush=True)
        assert fingerprint(g)==before
        assert all(q._version==versions[n] and q.grad is None for n,q in model.named_parameters())
        dump(a.output/'results.json',dict(complete=True,identity=identity,records=records,parameters_unchanged=True))
    finally:
        frontend.sample_action=original_sample;canonical.canonical_grounding=original_canonical
        canonical.encode_owners=original_owner
if __name__=='__main__':main()
