"""Natural six-instruction tests of the actual annotation-endpoint predictor."""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
import torch
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from probes.probe_source_delta_consumer import observe_return

def arr(x):return x.detach().float().cpu().numpy().copy()
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,float)**2)))
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
 q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--output',type=Path,required=True);a=q.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(4)
 p=ClearVLACheckpointPolicy(a.checkpoint,device=torch.device('cuda:0'),t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
 s=p.bundle.model.intent.organizer;predictor=s.endpoint_goal;plan=json.loads(a.plan.read_text());rows=[]
 scale=np.asarray(p.bundle.state_normalizer.scale).reshape(-1)[:3];offset=np.asarray(p.bundle.state_normalizer.offset).reshape(-1)[:3]
 for cid in [4,14,17]:
  row=next(x for x in plan if x['case_id']==cid);case=Path(row['case']);p.reset();h=CausalHistory(executed_world=True)
  with np.load(case/'trajectory.npz',allow_pickle=False) as z:
   obs=calvin_policy_observation({'rgb_obs':{k:z[k][0] for k in ['rgb_static','rgb_gripper']},'robot_obs':z['robot_obs'][0]},np.zeros(7,np.float32))
  h.reset(obs,reset_action=np.zeros(7,np.float32));_,online=p.act_with_input(h.snapshot(),row['instruction']);ref=online.instruction_reference
  for precision in ['bf16','fp32']:
   values=[];saved={}
   for color,direction in itertools.product(['red','blue','pink'],['left','right']):
    label=f'go push the {color} block {direction}';tokens,mask=p._goal(label);tokens=tokens.to(p.device);mask=mask.to(p.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
     memory=s.goal_input(torch.where(mask[...,None],tokens,0.));query=s.goal_queries.to(memory).expand(memory.shape[0],-1,-1)
     protected,*_=s.goal_read(query,memory,padding_mask=~mask,diagnostics=False);protected=s.goal_self(protected)
     pred,loc=observe_return(predictor.forward,(),{'reference':ref,'language':protected})
     _,weights=predictor.scene_read(predictor.query_norm(loc['query']),predictor.memory_norm(loc['memory']),predictor.memory_norm(loc['memory']),key_padding_mask=loc['padding'],need_weights=True,average_attn_weights=False)
    n=ref.dino.shape[1]*ref.dino.shape[2];robot_weights=arr(weights)[0,:,-1];scene_weights=arr(weights)[0,:,:-1]
    xyz=(arr(ref.state)[0,:3]+arr(pred.robot_delta)[0,:3]-offset)/scale
    saved[label]={'protected':arr(protected),'scene':arr(pred.scene),'spatial':arr(pred.log_probability.exp()),'xyz':xyz}
    values.append({'instruction':label,'endpoint_tcp_xyz':xyz.tolist(),'robot_query_language_mass_per_head':robot_weights[:,n:n+protected.shape[1]].sum(-1).tolist(),'robot_query_state_mass_per_head':robot_weights[:,-1].tolist(),'scene_query_mean_language_mass':float(scene_weights[:,:,n:n+protected.shape[1]].sum(-1).mean())})
   changes=[]
   for color,direction in itertools.product(['red','blue','pink'],['left','right']):
    first=f'go push the {color} block {direction}'
    for oc,od in [(other,direction) for other in ['red','blue','pink'] if other!=color]+[(color,'left' if direction=='right' else 'right')]:
     second=f'go push the {oc} block {od}';x,y=saved[first],saved[second]
     changes.append({'first':first,'second':second,'change':'color' if oc!=color else 'direction','protected_goal_delta_rms':rms(y['protected']-x['protected']),'predicted_scene_delta_rms':rms(y['scene']-x['scene']),'predicted_spatial_delta_rms':rms(y['spatial']-x['spatial']),'predicted_endpoint_xyz_distance':float(np.linalg.norm(y['xyz']-x['xyz']))})
   rows.append({'case_id':cid,'precision':precision,'instructions':values,'changes':changes});dump(a.output/'summary.json',rows)
  print(cid,'complete',flush=True)
 dump(a.output/'complete.json',{'checkpoint_sha256':p.bundle.checkpoint_sha256,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'same observed instruction-start reference across natural language replacements; fixed trained predictor; FP32 diagnostic vs production BF16; attention weights are descriptive, not independent causal attribution'})
if __name__=='__main__':main()
