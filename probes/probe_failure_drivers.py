"""Paired causal-history/endpoint diagnostics using the formal rollout RNG."""
from pathlib import Path
from dataclasses import replace
import argparse,json,hashlib
import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action

def arr(x):return x.detach().float().cpu().numpy().copy()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-ids',type=int,nargs='+',default=[17,14,15]);a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
 m=policy.bundle.model;b=m.execution_bottom;s=m.intent.organizer;terminal=b.decoder.terminal_controller
 old_step=b.step;old_s=s.forward;old_robot=s.task_relation_encoder.robot.forward;old_heads=terminal.read_heads;old_encode=m.encode_online
 import clearvla.simulation.clearvla_policy as front
 old_sample=front.sample_action;seeds=[];sargs=[];samples=[];head_rows=[];caches=[];control={};reference={};rows=[]
 def step(*args,**kwargs):
  kw=dict(kwargs);seeds.append(kw['seed'])
  if control.get('bottom_fields'):kw['seed']=replace(kw['seed'],**{k:getattr(reference['seed'],k) for k in control['bottom_fields']})
  return old_step(*args,**kw)
 def sh(*args,**kwargs):
  kw=dict(kwargs);sargs.append(kw)
  if control.get('S_history'):
   for k in ['state_history','executed_history','history_timing']:kw[k]=reference['S'][k]
  return old_s(*args,**kw)
 def robot(*args,**kwargs):
  if control.get('S_robot'):return old_robot(reference['S']['state'].to(args[0]))
  return old_robot(*args,**kwargs)
 def heads(*args,**kwargs):
  out=old_heads(*args,**kwargs)
  without=terminal.optional_command_head(args[0])
  head_rows.append({'actual_p_open':arr(out.command_logits.softmax(-1))[0,:8,1].tolist(),'without_private_gate_p_open':arr(without.softmax(-1))[0,:8,1].tolist()})
  return out
 def sampleh(*args,**kwargs):
  out=old_sample(*args,**kwargs);samples[:]=[out];return out
 def encode(*args,**kwargs):
  out=old_encode(*args,**kwargs);caches[:]=[out[0]];return out
 b.step=step;s.forward=sh;s.task_relation_encoder.robot.forward=robot;terminal.read_heads=heads;front.sample_action=sampleh;m.encode_online=encode
 def clear():seeds.clear();sargs.clear();samples.clear();head_rows.clear()
 def summary(out):
  native=policy.bundle.action_normalizer.decode(arr(out.action)[0]);native[:,-1]=arr(out.gripper_command)[0]
  cache=caches[-1];goal=cache.top.intent.annotated_goal;prediction=goal.prediction
  end=(arr(prediction.reference.state)[0,:3]+arr(prediction.robot_delta)[0,:3]-np.asarray(policy.bundle.state_normalizer.offset).reshape(-1)[:3])/np.asarray(policy.bundle.state_normalizer.scale).reshape(-1)[:3]
  remaining=(arr(prediction.robot_delta)[0,:3]-arr(goal.instruction_change.current_state-prediction.reference.state)[0,:3])/np.asarray(policy.bundle.state_normalizer.scale).reshape(-1)[:3]
  return {'native_first8':native[:8].tolist(),'mean':native[:8].mean(0).tolist(),'endpoint':head_rows[-1],'calls':len(seeds),'predicted_endpoint_tcp_xyz':end.tolist(),'predicted_remaining_tcp_xyz':remaining.tolist()}
 try:
  plan=json.loads(a.plan.read_text())
  for cid,wanted in [(17,[120,136]),(14,[328,336]),(15,[336]),(10,[56,72,120,136])]:
   if cid not in a.case_ids:continue
   row=next(r for r in plan if r['case_id']==cid);case=Path(row['case']);history=CausalHistory(executed_world=True);policy.reset()
   with np.load(case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
   for t in range(max(wanted)+1):
    previous=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
    obs=calvin_policy_observation({'rgb_obs':{k:data[k][t] for k in ['rgb_static','rgb_gripper']},'robot_obs':data['robot_obs'][t]},previous)
    if t==0:history.reset(obs,reset_action=previous)
    else:history.append(previous,obs)
    if t not in wanted:
     if t==0:policy.act_with_input(history.snapshot(),row['instruction'])
     elif t%8==0:m.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
     continue
    clear();control.clear();_,online=policy.act_with_input(history.snapshot(),row['instruction']);sample=samples[0];baseline=summary(sample)
    entry={'case_id':cid,'state':t,'baseline':baseline,'recorded_arm_rmse':rms(np.asarray(baseline['native_first8'])[:,:6]-data['raw_chunks'][t//8,:8,:6]),'variants':[]}
    if (cid,t) in [(17,120),(10,56),(10,120)]:reference={'seed':seeds[-1],'S':sargs[-1],'state':t}
    variants=[('repeat',{})]
    if (cid,t) in [(17,136),(10,72),(10,136)]:
     refname=str(reference['state'])
     variants += [('bottom_'+name+'_'+refname,{'bottom_fields':[name]}) for name in ['state','state_history','executed']]
     variants += [('bottom_all_'+refname,{'bottom_fields':['state','state_history','executed']}),('S_and_bottom_'+refname,{'bottom_fields':['state','state_history','executed'],'S_history':True,'S_robot':True}),('restart_instruction_now',{})]
    if cid in [14,15]:variants += [('restart_instruction_now',{}),('closed_command_history',{})]
    for name,settings in variants:
     clear();control.clear();control.update(settings);other=online
     if name=='restart_instruction_now':
      ref=online.instruction_reference
      other=replace(online,instruction_reference=replace(ref,dino=online.observation.dino_history[:,-1],state=online.history.state,age_steps=torch.zeros_like(ref.age_steps)))
     if name=='closed_command_history':
      # Hypothetical closed past commands with all duplicated command sources
      # synchronized. Images/proprioception stay factual: this is a diagnostic
      # source intervention, not a coherent counterfactual physical history.
      raw=np.zeros((1,7),np.float32);raw[:,-1]=-1;closed=float(policy.bundle.action_normalizer.encode(raw)[0,-1])
      ah=online.history.executed_action_history.clone();ah[...,-1]=closed
      ast=online.history.action_state.clone();ast[...,-1]=closed
      def closed_last(x):
       value=x.clone();value[...,-1]=closed;return value
      window=online.history.executed_world_window;robot_step=online.history.executed_robot_step
      other=replace(online,history=replace(online.history,executed_action_history=ah,action_state=ast,codec_gripper_boundary=torch.full_like(online.history.codec_gripper_boundary,closed),
       executed_world_window=replace(window,commands=closed_last(window.commands),action_state=closed_last(window.action_state)),
       executed_robot_step=replace(robot_step,command=closed_last(robot_step.command))))
     out=sample_action(m,other,policy.bundle.config,initial_physical_noise=sample.initial_physical_noise);v=summary(out);v['variant']=name;entry['variants'].append(v)
    control.clear();rows.append(entry);dump(a.output/'summary.json',rows);print(cid,t,'complete',flush=True)
  dump(a.output/'complete.json',{'checkpoint_sha256':policy.bundle.checkpoint_sha256,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'formal RNG advanced for skipped replans; boundary interventions are causal diagnostics, not coherent alternative physical rollouts; no architecture or weights changed'})
 finally:b.step=old_step;s.forward=old_s;s.task_relation_encoder.robot.forward=old_robot;terminal.read_heads=old_heads;front.sample_action=old_sample;m.encode_online=old_encode
if __name__=='__main__':main()
