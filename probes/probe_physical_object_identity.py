"""Match learned K/S reads to actual simulator body masks at selected failure stages.

Stores decision statistics, not a full activation dump. Natural-instruction
interventions keep observation/history/reference and sampling noise fixed.
"""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,importlib.util,json,subprocess,time
import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.mainline.runtime.sampling import sample_action
from probes.probe_source_delta_consumer import observe_return

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(x):return x.detach().float().cpu().numpy().copy()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def difference(a,b):
 d=np.asarray(b,float)-np.asarray(a,float)
 return {"rmse":rms(d),"base_rms":rms(a),"relative_rmse":rms(d)/max(rms(a),1e-12),"max_abs":float(np.max(np.abs(d)))}
def load(path):
 spec=importlib.util.spec_from_file_location("identity_coverage",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def native(policy,sampled):
 value=policy.bundle.action_normalizer.decode(array(sampled.action)[0]).astype(np.float32)
 value[:,-1]=array(sampled.gripper_command)[0]
 return value
def tensors_equalish(a,b):
 return {n:float(np.max(np.abs(array(getattr(a,n))-array(getattr(b,n))))) for n in ("content","semantic","appearance","geometry","camera_coordinates")}

def main():
 q=argparse.ArgumentParser();q.add_argument("--plan",type=Path,required=True);q.add_argument("--masks",type=Path,required=True)
 q.add_argument("--coverage-script",type=Path,required=True);q.add_argument("--checkpoint",type=Path,required=True)
 q.add_argument("--output",type=Path,required=True)
 q.add_argument("--match-rollout-rng",action="store_true")
 a=q.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 coverage=load(a.coverage_script).object_coverage
 policy=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device("cuda:0"),t5_condition=Path("/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt"),
    dinov3_model=Path("/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m"),seed=0)
 model=policy.bundle.model;grounder=model.grounding.grounder
 original_g=grounder.forward;original_encode=model.encode_online
 captures=[];cache_holder=[];current={};saved=[];sample_holder=[]
 def capture_g(*args,**kwargs):
  out,local=observe_return(original_g,args,kwargs)
  captures.append((out[0],{k:local[k] for k in ("chart","camera_read","candidate_prior","validity","candidate_shape")}))
  return out
 def capture_encode(*args,**kwargs):
  out=original_encode(*args,**kwargs);cache_holder[:]=[out[0]]
  intent=out[0].top.intent
  for n in ("protected_goal_set","public_interval_carrier","policy_interval_context","typed_relevance_value"):
   v=getattr(intent,n,None)
   if isinstance(v,torch.Tensor):current["S."+n]=array(v)
  return out
 grounder.forward=capture_g;model.encode_online=capture_encode
 import clearvla.simulation.clearvla_policy as frontend
 original_sample=frontend.sample_action
 def capture_sample(*args,**kwargs):
  out=original_sample(*args,**kwargs);sample_holder[:]=[out];return out
 frontend.sample_action=capture_sample
 def tap(owner,name,label,getter):
  old=getattr(owner,name);saved.append((owner,name,old))
  def wrapper(*args,**kwargs):
   out=old(*args,**kwargs);value=getter(out)
   if isinstance(value,torch.Tensor):current[label]=array(value)
   return out
  setattr(owner,name,wrapper)
 tap(model.intent.organizer.goal_input,"forward","S.goal_input",lambda x:x)
 tap(model.intent.organizer.goal_read,"forward","S.goal_read",lambda x:x[0] if isinstance(x,tuple) else x)
 tap(model.intent.coarse_action,"forward","coarse.action",lambda x:x.action_prediction)
 tap(model.p1,"build_static","P1.protected_detail",lambda x:x[0].protected_detail)
 tap(model.policy_compiler.effect_reader,"forward_candidate","P2.semantic",lambda x:x[0].semantic)
 tap(model.policy_compiler.plan_compiler,"forward","P3.temporal",lambda x:x[0].temporal)
 tap(model.execution_bottom.decoder,"_read_policy_delta_bank","bottom.optional_update",lambda x:x[0])
 plan=json.loads(a.plan.read_text());records=[];interventions=[];started=time.time()
 identity={"checkpoint":str(a.checkpoint),"checkpoint_sha256":policy.bundle.checkpoint_sha256,"probe_sha256":sha(__file__),
  "coverage_script_sha256":sha(a.coverage_script),"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "scope":"frozen targeted snapshots on recorded causal history; instruction reference remains state 0",
  "sampler_scope":("original rollout RNG advanced once at every 8-step replan; natural comparisons share exact initial physical noise"
    if a.match_rollout_rng else "snapshot RNG sequence is not original rollout RNG; natural comparisons share exact initial physical noise"),
  "dynamic_node_scope":"last refined endpoint call; native commands compare full sampled plans",
  "ground_truth_use":"simulator body masks and positions are read-only audit references, never policy input",
  "mask_metric":"per-camera sampling support coverage, not semantic-information percentage or certified K identity"}
 dump(a.output/"identity.json",identity)
 try:
  for row in plan:
   case=Path(row["case"]);dest=a.output/case.name;dest.mkdir();wanted=set(row["steps"])
   with np.load(case/"trajectory.npz",allow_pickle=False) as z:data={k:z[k] for k in z.files}
   history=CausalHistory(executed_world=True);policy.reset();case_records=[]
   for t in range(max(wanted)+1):
    previous=np.zeros(7,np.float32) if t==0 else data["executed"][t-1]
    obs=calvin_policy_observation({"rgb_obs":{"rgb_static":data["rgb_static"][t],"rgb_gripper":data["rgb_gripper"][t]},"robot_obs":data["robot_obs"][t]},previous)
    if t==0:history.reset(obs,reset_action=previous)
    else:history.append(previous,obs)
    if t not in wanted:
     if t==0:
      # An add-on plan may start late; its instruction reference must still
      # be established on the original state 0, not its first probed state.
      policy.act_with_input(history.snapshot(),row["instruction"])
     elif a.match_rollout_rng and t%8==0:
      model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
     continue
    mask=a.masks/case.name/("state_%03d.npz"%t)
    if not mask.exists():raise FileNotFoundError(mask)
    captures.clear();cache_holder.clear();sample_holder.clear();current.clear()
    action,online=policy.act_with_input(history.snapshot(),row["instruction"])
    if policy._instruction_anchor_step != 0:
     raise RuntimeError("targeted probe lost the original instruction-start reference")
    cache=cache_holder[0];sampled=sample_holder[0]
    matches=[(i,f,s) for i,(f,s) in enumerate(captures) if f.camera_coordinates is cache.top.belief.camera_coordinates]
    if len(matches)!=1:raise RuntimeError("current G producer ambiguity")
    which,facts,source=matches[0]
    coords=array(facts.camera_coordinates)[0]
    images=[data["rgb_static"][t],data["rgb_gripper"][t]]
    cover=coverage({"spatial":source["chart"].current_image_support,"camera_read":source["camera_read"]},images,mask,coords)
    prior=(source["candidate_prior"].float()*source["validity"].float()).reshape(1,*source["candidate_shape"])
    prior=prior/prior.sum((-3,-2,-1),keepdim=True).clamp_min(1e-12)
    prior=prior[:,None].expand(-1,4,-1,-1,-1,-1)
    spatial=source["chart"].current_image_support
    local_xy=(spatial.coordinates.float()*spatial.probability[...,None]).sum(-2)
    prior_center=array((prior[...,None]*local_xy[:,None]).sum((-4,-3,-2)))[0]
    prior_cover=coverage({"spatial":spatial,"camera_read":prior},images,mask,prior_center)
    mass=array(cache.top.intent.target_binding.mass)[0]
    condition={"state":t,"case_id":row["case_id"],"target":row["target"],"success":row["success"],
      "coordinates":coords.tolist(),"binding":mass.tolist(),"null":float(cache.top.intent.target_binding.null_mass[0]),
      "current_grounder_call":which,"grounder_calls":len(captures),"coverage":cover,"prior_coverage":prior_cover,
      "recorded_native_first8_mean":data["raw_chunks"][t//8,:8].mean(0).tolist(),
      "task_relation_rms":{n:rms(array(getattr(cache.top.intent.task_relation,n))) for n in ("target","scene") if isinstance(getattr(cache.top.intent.task_relation,n,None),torch.Tensor)},
      "node_rms":{k:rms(v) for k,v in current.items()}}
    condition["recorded_first8_arm_rmse"]=rms(action[:8,:6]-data["raw_chunks"][t//8,:8,:6])
    condition["recorded_first8_gripper_mismatches"]=int(np.count_nonzero(action[:8,6]!=data["raw_chunks"][t//8,:8,6]))
    condition["sampled_native_first8"]=action[:8].tolist()
    condition["binding_weighted_visible_object_read"]=[
       {"camera":c["camera"],"object_names":c["object_names"],"mass":(mass@np.asarray(c["k_object_probability"])).tolist()}
       for c in cover["cameras"]]
    case_records.append(condition);records.append(condition)
    do_natural=(t==24 and row["case_id"] in (2,4,5,9,10,11,14,17)) or (t==136 and row["case_id"]==17) or (t==336 and row["case_id"] in (14,15))
    if do_natural:
     baseline_nodes={k:v.copy() for k,v in current.items()};baseline_action=action.copy()
     direction="left" if row["instruction"].endswith("left") else "right"
     baseline_color=row["target"].split("_")[1]
     language_rows=[]
     instructions=[row["instruction"]]+["go push the "+c+" block "+direction for c in ("red","blue","pink") if c!=baseline_color]+["go push the "+baseline_color+" block "+("right" if direction=="left" else "left")]
     for label in instructions:
      tokens,mask_tokens=policy._goal(label)
      other=replace(online,goal=replace(online.goal,tokens=tokens.to(policy.device),mask=mask_tokens.to(policy.device)))
      captures.clear();cache_holder.clear();current.clear()
      with torch.no_grad():
       out=sample_action(model,other,policy.bundle.config,initial_physical_noise=sampled.initial_physical_noise)
      other_cache=cache_holder[0];other_mass=array(other_cache.top.intent.target_binding.mass)[0]
      out_action=native(policy,out)
      item={"instruction":label,"repeat":label==row["instruction"],"binding":other_mass.tolist(),
       "binding_difference":difference(mass,other_mass),
       "arm_first8_difference":difference(baseline_action[:8,:6],out_action[:8,:6]),
       "gripper_first8_mismatches":int(np.count_nonzero(baseline_action[:8,6]!=out_action[:8,6])),
       "nodes":{n:difference(v,current[n]) for n,v in baseline_nodes.items() if n in current},
       "goal_free_G_max_abs":tensors_equalish(facts,other_cache.top.belief),
       "binding_weighted_visible_object_read":[{"camera":c["camera"],"object_names":c["object_names"],"mass":(other_mass@np.asarray(c["k_object_probability"])).tolist()} for c in cover["cameras"]]}
      language_rows.append(item)
     entry={"case_id":row["case_id"],"state":t,"baseline_instruction":row["instruction"],"records":language_rows}
     interventions.append(entry);dump(dest/("natural_language_state_%03d.json"%t),entry)
    dump(dest/"identity_summary.json",{"case":str(case),"records":case_records})
    dump(a.output/"status.json",{"status":"running","completed_states":len(records),"expected_states":sum(len(x["steps"]) for x in plan),"case":row["case_id"],"state":t,"elapsed_seconds":time.time()-started})
    print("case",row["case_id"],"state",t,"complete",flush=True)
  dump(a.output/"all_identity_summaries.json",records)
  dump(a.output/"all_natural_interventions.json",interventions)
  dump(a.output/"status.json",{"status":"complete","completed_states":len(records),"expected_states":sum(len(x["steps"]) for x in plan),"natural_windows":len(interventions),"elapsed_seconds":time.time()-started})
 finally:
  for owner,name,old in reversed(saved):setattr(owner,name,old)
  grounder.forward=original_g;model.encode_online=original_encode;frontend.sample_action=original_sample
if __name__=="__main__":main()
