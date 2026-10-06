"""Diagnostic interventions on the real G producer and temporal S boundary."""
from pathlib import Path
from dataclasses import replace
import argparse,json,hashlib,time
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
def mean_camera(x):return x.float().mean(tuple(range(2,x.ndim-1)),keepdim=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-grounder',action='store_true');p.add_argument('--case-ids',type=int,nargs='+',default=[2,4,14,15,17]);a=p.parse_args()
 a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
 model=policy.bundle.model;g=model.grounding.grounder;s=model.intent.organizer;c=model.intent.coarse_action;b=s.shared_binder
 old_g=g.forward;old_s=s.forward;old_c=c.forward;old_encode=model.encode_online;old_candidate=g._candidate_tokens
 import clearvla.simulation.clearvla_policy as front
 old_sample=front.sample_action
 gs=[];ss=[];cs=[];caches=[];samples=[];controls={};component_capture={};key_handles=[];candidate_variant='baseline'
 for name in ['content_key','semantic_key','appearance_key','geometry_key','coordinate_key','context_key','history_key']:
  module=getattr(g,name,None)
  if module is not None:
   key_handles.append(module.register_forward_hook(lambda _m,_i,o,n=name:component_capture.__setitem__(n,o)))
 def candidates(*args,**kwargs):
  out,loc=observe_return(old_candidate,args,kwargs)
  value=loc['value'];offset=None
  if candidate_variant=='all_camera_mean':offset=mean_camera(value)-mean_camera(value).mean(1,keepdim=True)
  elif candidate_variant.endswith('_camera_mean'):
   name=candidate_variant[:-12];part=component_capture[name];offset=mean_camera(part)-mean_camera(part).mean(1,keepdim=True)
   if name in ['semantic_key','appearance_key','geometry_key']:offset=offset/(3**.5)
  if offset is not None:out=g.candidate_norm((value.float()-offset).to(value.dtype))
  component_capture['pre_norm']=value;component_capture['candidate']=out
  return out
 def gh(*args,**kwargs):
  out,loc=observe_return(old_g,args,kwargs)
  gs.append((out[0],args,kwargs,{k:loc[k] for k in ['structured_read','candidate_prior','candidates_structured','chart']},dict(component_capture)))
  return out
 def sh(*args,**kwargs):
  kwargs=dict(kwargs)
  if 's_fields' in controls:
   for name in controls['s_fields']:kwargs[name]=controls['reference_s'][name]
  out,loc=observe_return(old_s,args,kwargs)
  ss.append((out,args,kwargs,{k:loc[k] for k in ['objects','protected_goal','history_context','progress','instruction_change']}))
  return out
 def ch(*args,**kwargs):
  out=old_c(*args,**kwargs);effective=controls.get('coarse',out);cs.append(effective)
  return effective
 def eh(*args,**kwargs):
  out=old_encode(*args,**kwargs);caches[:]=[out[0]];return out
 def sampleh(*args,**kwargs):
  out=old_sample(*args,**kwargs);samples[:]=[out];return out
 old_robot=s.task_relation_encoder.robot.forward
 def robot(*args,**kwargs):
  if controls.get('robot_120'):return old_robot(controls['reference_s']['state'].to(args[0]))
  return old_robot(*args,**kwargs)
 s.task_relation_encoder.robot.forward=robot
 g._candidate_tokens=candidates;g.forward=gh;s.forward=sh;c.forward=ch;model.encode_online=eh;front.sample_action=sampleh
 def clear():gs.clear();ss.clear();cs.clear();caches.clear();samples.clear()
 def native(sample):
  x=policy.bundle.action_normalizer.decode(arr(sample.action)[0]);x[:,-1]=arr(sample.gripper_command)[0];return x
 def record(sample):
  x=native(sample);coarse=policy.bundle.action_normalizer.decode(arr(cs[-1].action_prediction)[0]);cache=caches[0]
  return {'native_first8_mean':x[:8].mean(0).tolist(),'native_first8':x[:8].tolist(),'coarse_first8_mean':coarse[:8].mean(0).tolist(),
    'gripper_open_probability_first8':arr(sample.gripper_command_logits.softmax(-1))[0,:8,1].tolist(),
    'binding':arr(cache.top.intent.target_binding.mass)[0].tolist(),
    'coarse_vs_final_arm_rmse':rms(coarse[:8,:6]-x[:8,:6])}
 rows=[];grows=[];time_start=time.time();previous_s=None;previous_coarse=None
 try:
  plan=json.loads(a.plan.read_text());schedule={2:[24],4:[24],14:[320,328,336,344],15:[328,336,344],17:[112,120,128,136,144]}
  for cid,wanted in schedule.items():
   if cid not in a.case_ids:continue
   row=next(v for v in plan if v['case_id']==cid);case=Path(row['case']);policy.reset();history=CausalHistory(executed_world=True)
   with np.load(case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
   for t in range(max(wanted)+1):
    prev=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
    obs=calvin_policy_observation({'rgb_obs':{k:data[k][t] for k in ['rgb_static','rgb_gripper']},'robot_obs':data['robot_obs'][t]},prev)
    if t==0:history.reset(obs,reset_action=prev)
    else:history.append(prev,obs)
    if t not in wanted:
     if t==0:policy.act_with_input(history.snapshot(),row['instruction'])
     elif t%8==0:
      # Formal deployment draws one field per replan and reuses it for W
      # refinement. Advance that same private generator for skipped plans.
      model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
     continue
    clear();controls.clear();_,online=policy.act_with_input(history.snapshot(),row['instruction']);sample=samples[0];cache=caches[0]
    current_g=next(z for z in gs if z[0].camera_coordinates is cache.top.belief.camera_coordinates)
    current_s=next(z for z in ss if z[2]['facts'].camera_coordinates is cache.top.belief.camera_coordinates)
    current_coarse=cs[-1];base=record(sample);entry={'case_id':cid,'state':t,'baseline':base,'recorded_first8_mean':data['raw_chunks'][t//8,:8].mean(0).tolist(),'recorded_arm_rmse':rms(native(sample)[:8,:6]-data['raw_chunks'][t//8,:8,:6]),'recorded_gripper_mismatches':int(np.count_nonzero(native(sample)[:8,6]!=data['raw_chunks'][t//8,:8,6])),'variants':[]}
    if cid==17 and t==120:previous_s=current_s[2];previous_coarse=current_coarse
    if not a.skip_grounder and (cid,t) in [(2,24),(17,136)]:
     fact,args,kwargs,local,parts=current_g
     rec={'case_id':cid,'state':t,'candidate_parts':{},'variants':[]}
     for name,v in parts.items():
      cm=mean_camera(v);rec['candidate_parts'][name]={'rms':rms(arr(v)),'camera_mean_difference_rms':rms(arr(cm[:,0]-cm[:,1])),
         'within_camera_rms':rms(arr(v.float()-cm))}
     variants=['baseline','all_camera_mean']+[name+'_camera_mean' for name in ['content_key','semantic_key','appearance_key','geometry_key','coordinate_key','context_key','history_key'] if name in parts]
     direction=row['instruction'].split()[-1];labels=['go push the '+color+' block '+direction for color in ['red','blue','pink']]
     for variant in variants:
      candidate_variant=variant
      with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
       fresh,floc=observe_return(old_g,args,kwargs);fresh=fresh[0]
      mass=arr(floc['structured_read'].float().flatten(3).sum(-1))[0];record_g={'variant':variant,'camera_mass':mass.tolist(),'reconstruction_error':float(fresh.reconstruction_error),'language':[]}
      if variant=='baseline':
       record_g['repeat_content_max_abs']=float(np.max(np.abs(arr(fresh.content)-arr(fact.content))))
       assert record_g['repeat_content_max_abs']<.002,record_g
      for label in labels:
       tok,mask=policy._goal(label);kw=dict(current_s[2]);kw.update(facts=fresh,goal_tokens=tok.to(policy.device),goal_mask=mask.to(policy.device))
       with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        result,sloc=observe_return(old_s,(),kw)
        binding,bloc=observe_return(b.forward,(sloc['protected_goal'],sloc['objects'],sloc['object_validity']),{'history':sloc['history_context']})
       score=arr(bloc['score'])[0];record_g['language'].append({'instruction':label,'binding':arr(binding.mass)[0].tolist(),'centered_score':(score-score.mean()).tolist(),
        'object_score':arr(bloc['object_score'])[0].tolist(),'task_object_score':arr(bloc['task_object_score'])[0].tolist()})
      rec['variants'].append(record_g)
     candidate_variant='baseline';grows.append(rec);dump(a.output/'grounder.json',grows)
    if (cid,t)==(17,136):
     variants=[('repeat',[]),('S_relation_robot_120',[]),('S_history_120',['state_history','executed_history','history_timing']),('S_relation_robot_and_history_120',['state_history','executed_history','history_timing']),('coarse_120',[])]
     for name,fields in variants:
      clear();controls.clear()
      if fields:controls.update(s_fields=fields,reference_s=previous_s)
      if 'robot' in name:controls.update(robot_120=True,reference_s=previous_s)
      if name=='coarse_120':controls['coarse']=previous_coarse
      out=sample_action(model,online,policy.bundle.config,initial_physical_noise=sample.initial_physical_noise)
      rr=record(out);rr['variant']=name;entry['variants'].append(rr)
     controls.clear()
    rows.append(entry);dump(a.output/'temporal.json',rows)
    print(cid,t,'complete',round(time.time()-time_start,1),flush=True)
  dump(a.output/'complete.json',{'checkpoint_sha256':policy.bundle.checkpoint_sha256,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'states':len(rows),'elapsed_seconds':time.time()-time_start,
   'scope':'targeted recorded histories; state-0 instruction reference; generator advanced exactly once for every original replan; paired fixed noise; hypothetical boundary interventions are not closed-loop success claims; component camera centering is diagnostic only'})
 finally:
  for h in key_handles:h.remove()
  g._candidate_tokens=old_candidate;g.forward=old_g;s.forward=old_s;c.forward=old_c;model.encode_online=old_encode;front.sample_action=old_sample;s.task_relation_encoder.robot.forward=old_robot

if __name__=='__main__':main()
