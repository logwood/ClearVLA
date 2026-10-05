from __future__ import annotations
import argparse,json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np, torch
import torch.nn.functional as F
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from clearvla.mainline.model.grounding import DenseObjectGrounder
from scripts.probe_calvin_internal_layers_npz import _history

INSTRUCTIONS=('go push the blue block right','go push the red block right','go push the pink block right','go push the blue block left')

def cos(v):
 x=torch.as_tensor(v).float().reshape(1,-1); return float(F.cosine_similarity(x,x.clone(),dim=-1).item())
def pair(v):
 x=v.detach().float(); a=x[:,1].reshape(x.shape[0],-1); b=x[:,3].reshape(x.shape[0],-1)
 return float(F.cosine_similarity(a,b,dim=-1).mean())
def rms(v): return float(torch.as_tensor(v).float().square().mean().sqrt())
def finite(v): return bool(torch.isfinite(v).all())
def scalar_map(m, prefixes):
 out={}
 for k,v in m.items():
  if isinstance(v,torch.Tensor) and v.numel()==1 and any(str(k).startswith(p) for p in prefixes):
   out[str(k)]=float(v.detach().float())
 return out
def main():
 device=torch.device('cuda')
 ckpt=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/short-bs8-gpu0-r9/checkpoints/best.pt')
 obs=Path('/data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003/standard.npz')
 t5=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt')
 policy=ClearVLACheckpointPolicy(ckpt,device=device,t5_condition=t5,seed=0)
 base=_history(obs)
 world=ExecutedWorldSnapshot(anchor_index=0,rgb_history=base.rgb_history,state=base.state.copy(),action_state=base.action_state.copy(),commands=base.executed_action_history[:4].copy(),visual_offsets=np.zeros(3,dtype=np.int64),observed=True)
 history=replace(base,time_index=4,previous_state=base.state.copy(),executed_world=world)
 original=DenseObjectGrounder.forward
 facts=[]
 def wrapped(self,*args,**kwargs):
  out=original(self,*args,**kwargs); facts.append(out[0]); return out
 DenseObjectGrounder.forward=wrapped
 rows=[]
 try:
  with torch.no_grad():
   for instruction in INSTRUCTIONS:
    facts.clear(); policy.reset(); _,online=policy.act_with_input(history,instruction)
    cache,static=deployment_cache(policy.bundle.model,online,policy.bundle.config,collect_diagnostics=True)
    f=facts[-1]; intent=cache.top.intent; binding=intent.target_binding
    chart=f.dense_chart.candidate_content.detach().float()
    cand_spatial=chart.reshape(chart.shape[0],-1,chart.shape[-1])
    row={
     'instruction':instruction,
     'g_candidate_shape':list(chart.shape),
     'g_candidate_rms':rms(chart),
     'g_candidate_cell_variation':float(cand_spatial.std(dim=1,unbiased=False).mean()),
     'g_fact_content_pair_cosine':pair(f.content),
     'g_fact_semantic_pair_cosine':pair(f.semantic),
     'g_fact_geometry_pair_cosine':pair(f.geometry),
     'g_assignment_pair_cosine':pair(f.candidate_assignment),
     'g_object_chart_pair_cosine':pair(f.object_to_chart),
     'g_reconstruction_mse':float(f.reconstruction_error.detach().float()),
     'g_candidate_assignment_per_k':[float(v) for v in f.candidate_assignment.detach().float().sum(dim=(2,3,4,5)).reshape(-1)],
     'g_static':scalar_map(static,('object_grounding_',)),
     'target_binding_mass':binding.mass.detach().float().reshape(-1).tolist() if binding is not None else None,
     'target_binding_null_mass':binding.null_mass.detach().float().reshape(-1).tolist() if binding is not None else None,
     'target_binding_address_logit':intent.target_object_address_logit.detach().float().reshape(-1).tolist(),
     's_object_attention_variation':float((intent.interval_object_attention-intent.interval_object_attention.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),
     's_public_interval_variation':float((intent.public_interval_carrier-intent.public_interval_carrier.mean(dim=1,keepdim=True)).float().square().mean().sqrt()),
     's_object_attention':intent.interval_object_attention.detach().float().reshape(-1).tolist(),
    }
    rows.append(row)
 finally:
  DenseObjectGrounder.forward=original
 result={'schema':'clearvla-second-plan-binding-audit-v1','checkpoint':str(ckpt),'rows':rows}
 out=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/probes/second-plan-binding-audit-20261005.json')
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n')
 print(json.dumps({'output':str(out),'rows':len(rows),'binding_mass':[r['target_binding_mass'] for r in rows],'g_content_cos':[r['g_fact_content_pair_cosine'] for r in rows],'g_assignment_cos':[r['g_assignment_pair_cosine'] for r in rows]},indent=2))
if __name__=='__main__': main()
