"""Local binder interventions on real G sources; no policy/weight modification."""
from pathlib import Path
from dataclasses import replace
import argparse,json,hashlib
import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action
from probes.probe_source_delta_consumer import observe_return

def arr(x):return x.detach().float().cpu().numpy().copy()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
 q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--output',type=Path,required=True);a=q.parse_args()
 a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
 model=policy.bundle.model;g=model.grounding.grounder;s=model.intent.organizer;b=s.shared_binder
 old_g=g.forward;old_s=s.forward;old_b=b.forward;old_encode=model.encode_online
 import clearvla.simulation.clearvla_policy as frontend
 old_sample=frontend.sample_action;sample_holder=[]
 def sample_hook(*args,**kwargs):
  out=old_sample(*args,**kwargs);sample_holder[:]=[out];return out
 frontend.sample_action=sample_hook
 gcalls=[];scalls=[];bcalls=[];caches=[]
 def g_hook(*args,**kwargs):
  out,local=observe_return(old_g,args,kwargs);gcalls.append((out[0],{k:local[k] for k in ['aggregated_content','content','structured_read']}));return out
 def s_hook(*args,**kwargs):
  out,local=observe_return(old_s,args,kwargs);scalls.append({k:local[k] for k in ['facts','objects','typed_object_context','object_validity']});return out
 def b_hook(*args,**kwargs):
  out=old_b(*args,**kwargs);bcalls.append((args,kwargs,out));return out
 def encode_hook(*args,**kwargs):
  out=old_encode(*args,**kwargs);caches[:]=[out[0]];return out
 g.forward=g_hook;s.forward=s_hook;b.forward=b_hook;model.encode_online=encode_hook
 rows=[]
 try:
  plan=json.loads(a.plan.read_text())
  for cid,t in [(2,24),(17,120),(17,136)]:
   case_row=next(r for r in plan if r['case_id']==cid);case=Path(case_row['case']);policy.reset();history=CausalHistory(executed_world=True)
   with np.load(case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
   for step in range(t+1):
    previous=np.zeros(7,np.float32) if step==0 else data['executed'][step-1]
    obs=calvin_policy_observation({'rgb_obs':{k:data[k][step] for k in ['rgb_static','rgb_gripper']},'robot_obs':data['robot_obs'][step]},previous)
    if step==0:
     history.reset(obs,reset_action=previous);policy.act_with_input(history.snapshot(),case_row['instruction'])
    else:history.append(previous,obs)
   gcalls.clear();scalls.clear();bcalls.clear();caches.clear()
   _,online=policy.act_with_input(history.snapshot(),case_row['instruction']);assert policy._instruction_anchor_step==0
   sampled=sample_holder[0]
   belief=caches[0].top.belief
   facts,local=next((f,v) for f,v in gcalls if f.camera_coordinates is belief.camera_coordinates)
   slocal=next(v for v in scalls if v['facts'].camera_coordinates is facts.camera_coordinates)
   assert float(np.max(np.abs(arr(local['content'])-arr(belief.content))))<1e-5
   args,kwargs,original=bcalls[-1];objects=slocal['objects'];typed=slocal['typed_object_context'];valid=slocal['object_validity'][...,None]
   mass=local['structured_read'].float().flatten(3).sum(-1)
   residual=facts.content-local['aggregated_content'];camera=facts.camera_content
   pooled=(camera.float()*mass[...,None]).sum(2)
   camera_valid=(facts.camera_validity[...,0]>0)
   equal=(camera.float()*camera_valid[...,None]).sum(2)/camera_valid.sum(2,keepdim=True).clamp_min(1)
   dtype=objects.dtype
   def object_value(content):
    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
     return torch.where(valid,s.object_content(content.to(facts.content.dtype))+typed,0.)
   rebuilt=object_value(facts.content)
   closure=float(np.max(np.abs(arr(rebuilt)-arr(objects))));assert closure<.002,closure
   variants={'baseline':objects,'remove_content_slot_residual':object_value(local['aggregated_content']),
             'equal_camera_content_keep_slot_residual':object_value(equal+residual.float())}
   record={'case_id':cid,'state':t,'global_camera_mass':arr(mass)[0].tolist(),
       'pooled_camera_content_closure_rmse':rms(arr(pooled)-arr(local['aggregated_content'])),
       'content_visual_rms':rms(arr(local['aggregated_content'])),'content_slot_residual_rms':rms(arr(residual)),
       'S_object_rebuild_max_abs':closure,'rows':[]}
   direction=case_row['instruction'].split()[-1]
   labels=[case_row['instruction']]+['go push the '+color+' block '+direction for color in ['red','blue','pink'] if 'block_'+color!=case_row['target']]
   baseline_mass={}
   for j,label in enumerate(labels):
    if j:
     tok,mask=policy._goal(label);other=replace(online,goal=replace(online.goal,tokens=tok.to(policy.device),mask=mask.to(policy.device)))
     gcalls.clear();scalls.clear();bcalls.clear();caches.clear()
     with torch.no_grad():sample_action(model,other,policy.bundle.config,initial_physical_noise=sampled.initial_physical_noise)
     # Binder is sampled before action noise is consumed; fixed physical noise
     # does not enter this static diagnostic. Use production-protected goals.
     cargs,ckwargs,_=bcalls[-1]
    else:cargs,ckwargs=args,kwargs
    task,supported=cargs[0],cargs[2];hist=ckwargs['history']
    for name,value in variants.items():
     with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
      out,loc=observe_return(old_b,(task,value,supported),{'history':hist})
     bm=arr(out.mass)[0]
     if j==0:baseline_mass[name]=bm
     record['rows'].append({'instruction':label,'variant':name,'mass':bm.tolist(),'null':float(out.null_mass[0]),
       'color_binding_max_abs':float(np.max(np.abs(bm-baseline_mass[name]))),
       'centered_score':(arr(loc['score'])[0]-arr(loc['score'])[0].mean()).tolist(),
       'object_score':arr(loc['object_score'])[0].tolist(),'task_object_score':arr(loc['task_object_score'])[0].tolist()})
   rows.append(record);dump(a.output/'summary.json',{'checkpoint_sha256':policy.bundle.checkpoint_sha256,'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'static binder-only interventions; downstream policy is not re-evaluated under these replacements; no efficacy claim','records':rows})
   print(cid,t,'complete',flush=True)
 finally:g.forward=old_g;s.forward=old_s;b.forward=old_b;model.encode_online=old_encode;frontend.sample_action=old_sample
if __name__=='__main__':main()
