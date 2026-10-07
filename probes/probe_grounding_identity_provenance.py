"""Trace factual observation support, each G competition and real S/W consumers.

Read-only instrumentation. Component camera-centering is an isolated G diagnosis,
never a production policy or training edit. Masks label report regions only.
"""
from pathlib import Path
import argparse, copy, hashlib, json
import numpy as np
import torch
import torch.nn.functional as F
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.vision.candidate_support import sample_candidate_expectation
from clearvla.vision.entity_chart import pushforward_log_to_current_image
from probes.probe_source_delta_consumer import observe_return
from probe_reconstruction_joint_gradient import clone_tree


def cpu(x): return x.detach().float().cpu().tolist()
def rms(x): return float(x.float().square().mean().sqrt())
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def camera_center(x):
    dims=tuple(range(2,x.ndim-1)); mean=x.mean(dims,keepdim=True)
    return x-mean+mean.mean(1,keepdim=True)


def feature_stats(value, region, base, names):
    # B=1, C, candidates, channels; weights describe physical support, not IDs.
    x=value.float().reshape(1,2,-1,value.shape[-1])[0]
    w=base.reshape(2,-1).float(); reg=region.reshape(2,-1,len(names)).float()
    means=(x*w[...,None]).sum(1)/w.sum(1)[:,None].clamp_min(1e-20)
    result={'rms':rms(x),'camera_mean_difference_rms':rms(means[0]-means[1]),'regions':[]}
    object_means={}
    for c in range(2):
        for j,name in enumerate(names):
            rw=w[c]*reg[c,:,j]
            if float(rw.sum())<1e-7: continue
            mu=(x[c]*rw[:,None]).sum(0)/rw.sum()
            object_means[(c,j)]=mu
            result['regions'].append({'camera':c,'object':name,'mean_from_camera_mean_rms':rms(mu-means[c]),
                'within_region_rms':float(((x[c]-mu).square().mean(-1)*rw).sum().div(rw.sum()).sqrt())})
    pairs=[]
    for c in range(2):
        present=[(j,mu) for (cam,j),mu in object_means.items() if cam==c]
        for i,(j,mu) in enumerate(present):
            for k,other in present[i+1:]: pairs.append({'camera':c,'objects':[names[j],names[k]],'mean_difference_rms':rms(mu-other)})
    result['object_pairs']=pairs
    return result


def distribution_stats(owner, read, logit, shape, coverage, base, names):
    k=read.shape[1]; own=owner[0,:,:k].float().reshape(2,-1,k)
    w=base.reshape(2,-1); reg=coverage.reshape(2,-1,len(names))
    rd=read[0].reshape(k,2,-1).float()
    camera_mass=rd.sum(-1)
    joint=(own*w[...,None]).sum(1).T; joint=joint/joint.sum()
    pk=joint.sum(1,keepdim=True); pc=joint.sum(0,keepdim=True)
    mi=(joint*(joint.clamp_min(1e-30).log()-(pk*pc).clamp_min(1e-30).log())).sum()
    conditional=rd/camera_mass[...,None].clamp_min(1e-30)
    per_camera_pair_tv=[]
    for c in range(2):
        per_camera_pair_tv.append(cpu(torch.stack([.5*(conditional[i,c]-conditional[j,c]).abs().sum() for i in range(k) for j in range(i+1,k)])))
    z=logit[0].float().reshape(2,-1,k); z=z-z.mean(-1,keepdim=True)
    global_k=z.mean((0,1),keepdim=True); camera_k=z.mean(1,keepdim=True)-global_k
    spatial=z-z.mean(1,keepdim=True); total=z.square().mean().clamp_min(1e-30)
    energies={'global_K_bias':float(global_k.square().mean()/total),'camera_K_interaction':float(camera_k.square().mean()/total),'within_camera_K_interaction':float(spatial.square().mean()/total)}
    regions=[]
    for c in range(2):
        for j,name in enumerate(names):
            rw=w[c]*reg[c,:,j]
            if float(rw.sum())<1e-7:continue
            km=(own[c]*rw[:,None]).sum(0); km=km/km.sum().clamp_min(1e-30)
            regions.append({'camera':c,'object':name,'K_given_region':cpu(km),'K_camera_conditioned_region_read':cpu((conditional[:,c]*reg[c,:,j]).sum(-1)),
                'best_candidate_physical_coverage':float(reg[c,:,j].max())})
    return {'camera_K_mi':float(mi),'K_camera_read_mass':cpu(camera_mass),'within_camera_pair_read_tv':per_camera_pair_tv,
        'K2_K4_read_tv':float(.5*(rd[1]-rd[3]).abs().sum()),'logit_energy_fractions':energies,'regions':regions}



def image_read_stats(log_read,read_support,spatial,shape):
    k=log_read.shape[1]
    measure=pushforward_log_to_current_image(log_read.reshape(1,k,*shape),read_support.reshape(1,k,*shape),spatial,rows=16,columns=16)
    full,_=measure.normalized((2,3,4));view,_=measure.normalized((3,4))
    return {'native_image_K2_K4_global_tv':float(.5*(full[:,1]-full[:,3]).abs().sum()),
        'native_image_K2_K4_per_camera_tv':cpu(.5*(view[:,1]-view[:,3]).abs().sum((-1,-2))),
        'native_image_camera_pair_tv':[[float(.5*(view[0,i,c]-view[0,j,c]).abs().sum()) for i in range(k) for j in range(i+1,k)] for c in range(2)]}


def audit(grounder, captured, native, masks, names):
    module=copy.deepcopy(grounder).float().eval()
    module.forward=type(module).forward.__get__(module,type(module))
    facts=clone_tree(captured['local_facts']); shape=facts.content_slots.shape[1:-1]
    support=facts.current_image_support
    # Keep exact source spatial supports; a real-mask expectation is report-only.
    coverage=[]
    for c,m in enumerate(masks):
        chart=torch.as_tensor(m,device=facts.content_slots.device,dtype=torch.float32).permute(1,2,0)[None,None]
        coverage.append(sample_candidate_expectation(chart,support.coordinates[:,c:c+1],support.probability[:,c:c+1]))
    coverage=torch.cat(coverage,1)
    area_chart=torch.stack([F.adaptive_avg_pool2d(torch.as_tensor(m,device=facts.content_slots.device,dtype=torch.float32),facts.target_dino_content.shape[2:4]).permute(1,2,0) for m in masks])[None]
    area_coverage=sample_candidate_expectation(area_chart,support.coordinates,support.probability)
    support_audit=[]
    for c,m in enumerate(masks):
        points=support.coordinates[:,c].reshape(1,-1,256,2)
        raw_samples=F.grid_sample(torch.as_tensor(m,device=points.device,dtype=torch.float32)[None],points,align_corners=True,padding_mode='zeros')
        area_samples=F.grid_sample(area_chart[:,c].permute(0,3,1,2),points,align_corners=True,padding_mode='zeros')
        for j,name in enumerate(names):
            support_audit.append({'camera':c,'object':name,'any_atom_point_coverage':float(raw_samples[0,j].max()),'best_weighted_point_coverage':float(coverage[0,c,...,j].max()),'any_atom_feature_area_coverage':float(area_samples[0,j].max()),'best_weighted_feature_area_coverage':float(area_coverage[0,c,...,j].max())})
    components={}; competitions=[]; hooks=[]; mode='baseline'
    for name in ('content_key','semantic_key','appearance_key','geometry_key','coordinate_key','context_key','history_key'):
        component=getattr(module,name)
        if component is None:continue
        def hook(mod,args,out,name=name):
            if mode=='baseline':components[name]=out
            return camera_center(out) if mode==name else out
        hooks.append(component.register_forward_hook(hook))
    def norm_input(mod,args):return (camera_center(args[0]),) if mode=='all_before_norm' else None
    hooks.append(module.candidate_norm.register_forward_pre_hook(norm_input))
    original=module._competition
    def competition(*args,**kwargs):
        out,local=observe_return(original,args,kwargs)
        competitions.append({'out':out,'local':local,'slots':args[0]})
        return out
    module._competition=competition
    try:
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):
            output,local=observe_return(module.forward,(facts,),{'collect_diagnostics':False})
            base=(local['candidate_prior']*local['valid']).reshape(1,*shape)
            # Exact producer provenance: DINO expectation uses the real G2 support.
            recontent=sample_candidate_expectation(facts.target_dino_content,support.coordinates,support.probability)
            recontext=sample_candidate_expectation(facts.public_scene_base,support.coordinates,support.probability)
            dense=facts.target_dino_content; target_weights=torch.stack([F.adaptive_avg_pool2d(torch.as_tensor(m,device=dense.device,dtype=torch.float32),dense.shape[2:4]).permute(1,2,0) for m in masks])[None]
            native_side=int(np.sqrt(native.shape[-2])); native=native.float().reshape(1,2,native_side,native_side,-1)
            native_weights=torch.stack([F.adaptive_avg_pool2d(torch.as_tensor(m,device=dense.device,dtype=torch.float32),(native_side,native_side)).permute(1,2,0) for m in masks])[None]
            features={'native_dino':feature_stats(native,native_weights,torch.ones(native.shape[:-1],device=dense.device),names),
                'dense_target_dino':feature_stats(dense,target_weights,torch.ones(dense.shape[:-1],device=dense.device),names),
                'candidate_content':feature_stats(facts.content_slots,area_coverage,base,names),
                'candidate_tokens':feature_stats(local['candidates_structured'],area_coverage,base,names)}
            features.update({key:feature_stats(value,area_coverage,base,names) for key,value in components.items()})
            stages=[]
            for j,row in enumerate(competitions):
                own,_,_,read,_,_=row['out']; loc=row['local']
                update=torch.einsum('bkn,bnh->bkh',read,local['candidates'])
                states=row['slots'].float()[0]; units=F.normalize(states,dim=-1); update_units=F.normalize(update[0],dim=-1)
                stages.append({'competition_index':j,'meaning':'learned seed' if j==0 else f'after GRU/FFN/identity update {j}',
                    'slots_pair_cosine':cpu(units@units.T),'read_update_pair_cosine':cpu(update_units@update_units.T),
                    'identity_gain':float(loc['identity_gain']),'physical_image_read':image_read_stats(row['out'][5],local['read_support'],support,shape),**distribution_stats(own,read,loc['logits'],shape,coverage,base,names)})
            final=distribution_stats(local['corrected'],local['read'],local['corrected_log_owner'][...,:module.objects],shape,coverage,base,names)
            final['physical_image_read']=image_read_stats(local['read_log_probability'],local['read_support'],support,shape)
            controls={}
            for mode in list(components)+['all_before_norm']:
                competitions.clear()
                _,changed=observe_return(module.forward,(facts,),{'collect_diagnostics':False})
                controls[mode]={'reconstruction':float(changed['reconstruction_error']),**distribution_stats(changed['corrected'],changed['read'],changed['corrected_log_owner'][...,:module.objects],shape,coverage,base,names)}
            mode='baseline'
            # Algebraic source check: varying K mass alone cannot make unique reads.
            constant=local['candidates'].mean(1,keepdim=True).expand_as(local['candidates'])
            own,_,_,rd,_,_=original(local['slots'],constant,torch.ones_like(local['validity']),local['candidate_prior'],local['candidate_log_prior'],local['candidate_prior_support'])
            region_coverage=[]
            for c in range(2):
                for j,name in enumerate(names):
                    rc=coverage[0,c,...,j]
                    if float(rc.max())>1e-7:region_coverage.append({'camera':c,'object':name,'max':float(rc.max()),'mean':float(rc.mean()),'count_gt_25_percent':int((rc>.25).sum())})
            return {'active_shape':list(facts.content_slots.shape),'support_shape':list(support.coordinates.shape),
                'content_from_actual_support_rms':rms(recontent-facts.content_slots),'context_from_actual_support_rms':rms(recontext-facts.context_slots),
                'fp32_vs_factual_content_rms':rms(output[0].content-captured['facts'].content.float()),
                'features':features,'physical_support_audit':support_audit,'candidate_physical_coverage':region_coverage,'competition_stages':stages,'after_G3_residual':final,
                'component_camera_centering':controls,'separable_owner_counterexample':{'K_read_max_difference':float((rd-rd[:,:1]).abs().max()),'K_real_owner_mean':cpu(own[...,:module.objects].mean(1))},
                'contract':'active restored producer; input provenance plus each actual G iteration; component interventions confined to isolated G; no rollout/optimizer change'}
    finally:
        for h in hooks:h.remove()



def upstream_support(rows,masks,names):
    result=[]
    for row in rows[:2]:
        c=row['locals'];stage=row['stage']
        if stage==1:
            coordinates=c['coordinates'];measures={'G1_base':c['bank'].coarse_base_logits.float().softmax(-1),'G1_after_delta':c['probability'].float()}
            shape=measures['G1_after_delta'].shape
            coordinates=coordinates[:, :, None, None, None].expand(*shape,2)
        else:
            coordinates=c['fine_coordinates'];valid=c['dynamic_fine_valid']
            def law(z):
                z=z.float().masked_fill(~valid,-1e9);out=z.softmax(-1)*valid
                return out/out.sum(-1,keepdim=True).clamp_min(1e-20)
            measures={'G2_parent':law(c['candidates'].parent_log_probability),'G2_scaled_parent':law(c['spatial_prior']),'G2_actual_geometry':c['geometry_probability'].float(),
                'G2_actual_appearance':c['appearance_probability'].float(),'G2_coupled_evidence_without_parent':law(torch.stack(tuple(c['typed_logits'].values())).sum(0)/(len(c['typed_logits'])**.5))}
        out={'stage':stage,'measures':{}}
        for name,p in measures.items():
            detail=[]
            for camera,m in enumerate(masks):
                points=coordinates[:,camera].reshape(1,-1,p.shape[-1],2)
                raw=torch.as_tensor(m,device=p.device,dtype=torch.float32)[None]
                for granularity,chart in [('point',raw),('feature_area_8',F.adaptive_avg_pool2d(raw,(8,8)))]:
                    samples=F.grid_sample(chart,points,align_corners=True,padding_mode='zeros')[0]
                    weight=p[0,camera].reshape(-1,p.shape[-1])
                    expectation=(samples*weight[None]).sum(-1)
                    for j,obj in enumerate(names):
                        detail.append({'camera':camera,'object':obj,'scope':granularity,'best_query_expected_coverage':float(expectation[j].max()),'average_query_coverage':float(expectation[j].mean()),'any_atom_coverage':float(samples[j].max())})
            out['measures'][name]={'coverage':detail,'mean_effective_support':float((1/p.square().sum(-1).clamp_min(1e-20)).mean())}
        result.append(out)
    return result



def base_support(captured,address_call,masks,names):
    """Locate support bias before G1; removed terms are diagnostics, not repairs."""
    c=captured['locals'];bank=address_call['locals']['bank']
    if captured['bank'] is not bank:
        raise RuntimeError('base compiler bank is not the actual G1 bank')
    content=c['content_logits'].float();floor=c['floor_flow_bias'].float();adaptive=c['adaptive_flow_bias'].float()
    total=content+floor+adaptive
    coordinate=bank.coarse_candidate_coordinates.float()
    rows={}
    terms={'actual_base':total,'content_only':content,'geometric_only':floor+adaptive,
           'without_floor':content+adaptive,'without_adaptive':content+floor,'uniform':torch.zeros_like(total)}
    for name,logits in terms.items():
        probability=logits.expand_as(total).softmax(-1);detail=[]
        for cam,m in enumerate(masks):
            raw=torch.as_tensor(m,device=total.device,dtype=torch.float32)[None]
            for scope,chart in [('point',raw),('feature_area_8',F.adaptive_avg_pool2d(raw,(8,8)))]:
                values=F.grid_sample(chart,coordinate[:,cam,None],align_corners=True,padding_mode='zeros')[0,:,0]
                expected=torch.einsum('...n,on->...o',probability[0,cam],values)
                for j,obj in enumerate(names):
                    detail.append({'camera':cam,'object':obj,'scope':scope,'best_query_expected_coverage':float(expected[...,j].max()),'average_query_coverage':float(expected[...,j].mean())})
        rows[name]={'coverage':detail,'mean_effective_support':float((1/probability.square().sum(-1).clamp_min(1e-20)).mean())}
    def contrast(x): return rms(x-x.mean(-1,keepdim=True))
    return {'base_recomposition_rms':rms(total-c['coarse_logits'].float()),
            'serialized_base_rounding_rms':rms(total-bank.coarse_base_logits.float()),
            'content_scale':float(c['content_scale']),'flow_prior_floor':float(c['self'].flow_prior_floor),
            'adaptive_prior_scale':float(c['adaptive_prior_scale']),
            'logit_candidate_contrast_rms':{'content':contrast(content),'floor':contrast(floor),'adaptive':contrast(adaptive)},
            'source_dino_shape':list(c['source_dino'].shape),'source_target_dino_difference_rms':rms(c['source_dino']-c['target_dino']),
            'measures':rows,'scope':'No object labels enter producer; geometric correspondence is not an object selector.'}


def main():
    ap=argparse.ArgumentParser()
    for name in ('checkpoint','plan','masks','output'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
    model=policy.bundle.model;g=model.grounding.grounder;old=g.forward;sources=[];downstream=[]
    base_compiler=model.observation.compiler.encoder.soft_address_compiler;old_base=base_compiler.forward;base_calls=[]
    def build_base(*args,**kwargs):
        out,loc=observe_return(old_base,args,kwargs);base_calls.append({'bank':out[0],'locals':loc});return out
    base_compiler.forward=build_base
    def capture(*args,**kw):
        out,loc=observe_return(old,args,kw);sources.append({'local_facts':loc['local_facts'],'facts':out[0]});return out
    g.forward=capture
    address=model.observation.compiler.encoder.progressive_grounding_address
    old_address=address.update;address_calls=[]
    def advance(*args,**kwargs):
        out,loc=observe_return(old_address,args,kwargs)
        if kwargs['stage'] in (1,2):address_calls.append({'stage':kwargs['stage'],'locals':loc})
        return out
    address.update=advance
    binder=model.intent.organizer.shared_binder
    old_binder=binder.forward
    def bind(*args,**kw):
        out,loc=observe_return(old_binder,args,kw)
        downstream.append({'node':'S.shared_binder','object_shape':list(loc['objects'].shape),'per_view':bool(loc['per_view']),
            'K_mass':cpu(out.mass),'null':cpu(out.null_mass),'object_token_pair_rms':[[rms(loc['objects'][:,i]-loc['objects'][:,j]) for j in range(4)] for i in range(4)]})
        return out
    binder.forward=bind
    dynamics=model.world.dynamics;old_w=dynamics.forward_w1
    def world(*args,**kw):
        out=old_w(*args,**kw);f=kw['facts'];current=sources[0]['facts'] if sources else None
        downstream.append({'node':'W1','type':type(f).__name__,'content_is_current_G':current is not None and f.content is current.content,
            'content_shape':list(f.content.shape),'has_camera_content':getattr(f,'camera_content',None) is not None})
        return out
    dynamics.forward_w1=world
    rows=[]
    try:
        plan=json.loads(a.plan.read_text())
        for cid,state in ((1,24),(5,24),(11,24),(17,136)):
            item=next(x for x in plan if x['case_id']==cid);case=Path(item['case'])
            with np.load(case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in ('rgb_static','rgb_gripper','robot_obs','executed','raw_chunks')}
            policy.reset();history=CausalHistory(executed_world=True)
            for t in range(state+1):
                command=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
                observation=calvin_policy_observation({'rgb_obs':{k:data[k][t] for k in ('rgb_static','rgb_gripper')},'robot_obs':data['robot_obs'][t]},command)
                if t==0:history.reset(observation,reset_action=command)
                else:history.append(command,observation)
                if t not in (0,state):
                    if t%8==0:model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    continue
                sources.clear();downstream.clear();address_calls.clear();base_calls.clear();action,online=policy.act_with_input(history.snapshot(),item['instruction'])
                if t==state:
                    with np.load(a.masks/case.name/f'state_{t:03d}.npz',allow_pickle=False) as z:masks=[z[k+'_masks'].copy() for k in ('top','wrist')];names=z['object_names'].tolist()
                    g.forward=old
                    with torch.inference_mode(False),torch.no_grad():result=audit(g,sources[0],online.observation.dino_history[:,-1],masks,names)
                    g.forward=capture
                    rows.append({'case_id':cid,'state':t,'instruction':item['instruction'],'active_observation_class':type(model.observation.compiler).__name__,'preprocessing':policy.preprocessing.to_dict(),
                        'recorded_arm_rmse':float(np.sqrt(np.mean((action[:8,:6]-data['raw_chunks'][t//8,:8,:6])**2))),'downstream':list(downstream),'local_ownership_mode':address.local_ownership_mode,'candidate_support_mode':address.candidate_support_mode,'upstream_support':upstream_support(address_calls,masks,names),'base_support':base_support(next(c for c in base_calls if c['bank'] is address_calls[0]['locals']['bank']),address_calls[0],masks,names),**result})
                    dump(a.output/'results.json',rows);print(cid,t,'complete',flush=True)
        dump(a.output/'complete.json',{'checkpoint_sha256':policy.bundle.checkpoint_sha256,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'windows':len(rows)})
    finally:g.forward=old;binder.forward=old_binder;dynamics.forward_w1=old_w;address.update=old_address;base_compiler.forward=old_base


if __name__=='__main__':main()
