"""Common A/B factual-source and native-action audit, with no legacy local-K assumptions.

Masks are report-only. Natural instructions share observations, reference and
initial physical noise. Full tensor dumps and altered expert-action labels are
not produced. Run with PYTHONPATH pointing to the checkpoint's source checkout.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,hashlib
import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action


def array(t):return t.detach().float().cpu().numpy()
def rms(t):return float(np.sqrt(np.square(np.asarray(t,dtype=float)).mean()))
def dump(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def physical_read(facts,masks):
    result=[]
    for camera,mask in enumerate(masks):
        measure=facts.current_image_source.on_image(rows=mask.shape[-2],columns=mask.shape[-1])
        conditional,_=measure.normalized((-2,-1));joint,_=measure.normalized((2,3,4))
        read=conditional[0,:,camera];global_read=joint[0,:,camera]
        weight=torch.as_tensor(mask,device=read.device,dtype=read.dtype)
        coverage=torch.einsum('khw,ohw->ko',read,weight)
        law,_=measure.normalized((1,))
        regional=torch.einsum('khw,ohw->ok',law[0,:,camera],weight)/weight.sum((-2,-1))[:,None].clamp_min(1)
        result.append(dict(k_object_read=array(coverage).tolist(),K_given_object=array(regional).tolist(),view_mass=array(global_read.sum((-2,-1))).tolist(),visible_pixels=mask.sum((-2,-1)).tolist()))
    return result


def main():
    q=argparse.ArgumentParser()
    for name in ('checkpoint','plan','masks','output'):q.add_argument('--'+name,type=Path,required=True)
    a=q.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
    policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
    model=policy.bundle.model;g=model.grounding.grounder;captures=[];holder={}
    original_g=g.forward;original_encode=model.encode_online
    import clearvla.simulation.clearvla_policy as frontend
    original_sample=frontend.sample_action
    def ground(*args,**kwargs):
        out=original_g(*args,**kwargs);captures.append(out[0]);return out
    def encode(*args,**kwargs):
        out=original_encode(*args,**kwargs);holder['cache']=out[0];holder['state']=out[1];return out
    def sampled(*args,**kwargs):
        out=original_sample(*args,**kwargs);holder['sampled']=out;return out
    g.forward=ground;model.encode_online=encode;frontend.sample_action=sampled
    plan=json.loads(a.plan.read_text());records=[]
    identity=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=policy.bundle.checkpoint_sha256,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config=policy.bundle.config.top.__dict__,mask_use='audit-only',source='recorded-causal-history-at-fixed-observation',scope='read-support/appearance correspondence, not independently proven semantic identity')
    dump(a.output/'identity.json',identity)
    try:
        for row in plan:
            with np.load(Path(row['case'])/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in ('rgb_static','rgb_gripper','robot_obs','executed')}
            policy.reset();history=CausalHistory(executed_world=True)
            for t in range(max(row['steps'])+1):
                previous=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
                obs=calvin_policy_observation({'rgb_obs':{key:data[key][t] for key in ('rgb_static','rgb_gripper')},'robot_obs':data['robot_obs'][t]},previous)
                if t==0:history.reset(obs,reset_action=previous)
                else:history.append(previous,obs)
                if t not in row['steps']:
                    if t==0:policy.act_with_input(history.snapshot(),row['instruction'])
                    elif t%8==0:model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    continue
                captures.clear();action,online=policy.act_with_input(history.snapshot(),row['instruction'])
                cache=holder['cache'];sample=holder['sampled']
                facts=next(f for f in captures if f.camera_coordinates is cache.top.belief.camera_coordinates)
                with np.load(a.masks/Path(row['case']).name/('state_%03d.npz'%t),allow_pickle=False) as z:masks=[z['top_masks'],z['wrist_masks']];names=z['object_names'].tolist()
                with torch.no_grad():coverage=physical_read(facts,masks)
                entry=dict(case_id=row['case_id'],step=t,object_names=names,coverage=coverage,grounder_calls=len(captures),baseline_first8=action[:8].tolist(),natural=[])
                direction='left' if row['instruction'].endswith('left') else 'right'
                labels=[row['instruction']]+['go push the '+color+' block '+direction for color in ('red','blue','pink') if color not in row['instruction']]
                for text in labels:
                    tokens,mask=policy._goal(text)
                    other=replace(online,goal=replace(online.goal,tokens=tokens.to(policy.device),mask=mask.to(policy.device)))
                    with torch.no_grad():out=sample_action(model,other,policy.bundle.config,initial_physical_noise=sample.initial_physical_noise)
                    new=holder['cache'];mass=array(new.top.intent.target_binding.mass)[0]
                    native=policy.bundle.action_normalizer.decode(array(out.action)[0]);native[:,-1]=array(out.gripper_command)[0]
                    entry['natural'].append(dict(instruction=text,repeat=text==row['instruction'],binding=mass.tolist(),null=float(new.top.intent.target_binding.null_mass[0]),native_first8=native[:8].tolist(),native_arm_difference_rms=rms(native[:8,:6]-action[:8,:6]),selected_object_read=[(mass@np.asarray(view['k_object_read'])).tolist() for view in coverage],G_content_difference_max=float((new.top.belief.content-facts.content).abs().max())))
                with torch.no_grad():
                    raw=facts.dense_chart.dino_content[:,None]
                    offsets=torch.tensor([0,4,8,16,24],device=raw.device)
                    measured=model.training_targets.teacher.measure_observations(facts=facts,observations=raw.expand(-1,5,-1,-1,-1,-1),relative_offsets=offsets)
                    entry['static_contract']=dict(semantic_max=float((measured.successor_per_support-measured.current_reference).abs().max()),image_max=float(measured.transport_per_support.abs().max()),covariance_max=float(measured.covariance_per_support.abs().max()),null=array(measured.null_probability)[0,:, :,0].tolist())
                records.append(entry);dump(a.output/'results.json',dict(identity=identity,records=records,complete=False))
                print('ADMISSION',row['case_id'],t,flush=True)
        dump(a.output/'results.json',dict(identity=identity,records=records,complete=True))
    finally:g.forward=original_g;model.encode_online=original_encode;frontend.sample_action=original_sample


if __name__=='__main__':main()
