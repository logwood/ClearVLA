from pathlib import Path
from dataclasses import replace
import json,numpy as np,torch
from clearvla.mainline.model.grounding import DenseObjectGrounder,_supported_values
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from scripts.probe_calvin_internal_layers_npz import _history
def main():
 d=torch.device('cuda'); ck=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/short-bs8-gpu0-r9/checkpoints/best.pt'); obs=Path('/data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003/standard.npz'); t5=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt')
 p=ClearVLACheckpointPolicy(ck,device=d,t5_condition=t5,seed=0); b=_history(obs); w=ExecutedWorldSnapshot(anchor_index=0,rgb_history=b.rgb_history,state=b.state.copy(),action_state=b.action_state.copy(),commands=b.executed_action_history[:4].copy(),visual_offsets=np.zeros(3,dtype=np.int64),observed=True); h=replace(b,time_index=4,previous_state=b.state.copy(),executed_world=w)
 _,online=p.act_with_input(h,'go push the blue block right'); rows=[]; orig=DenseObjectGrounder._competition
 def wrap(self,*a,**kw):
  slots,candidates,validity,prior,*_=a
  with torch.autocast(device_type='cuda',enabled=False):
   v=torch.nan_to_num(validity.float(),nan=0.0,posinf=0.0,neginf=0.0).reshape(validity.shape[0],validity.shape[1],1).clamp(0,1)
   c=_supported_values(candidates,v).float(); sk=self.slot_norm(slots.float())
   si=self.slot_seed.to(device=c.device,dtype=torch.float32); si=si-si.mean(1,keepdim=True); si=torch.nn.functional.normalize(si,dim=-1,eps=1e-6).expand(slots.shape[0],-1,-1)
   main=torch.einsum('bkh,bnh->bnk',sk,c)/float(self.hidden**0.5); ident=torch.einsum('bkh,bnh->bnk',si,c)/float(self.hidden**0.5)
   rows.append({'main_rms':float(main.square().mean().sqrt()),'identity_rms':float(ident.square().mean().sqrt()),'ratio':float((ident.square().mean()/main.square().clamp_min(1e-12).mean()).sqrt()),'main_abs_max':float(main.abs().max()),'identity_abs_max':float(ident.abs().max()),'slot_norm_rms':float(sk.square().mean().sqrt()),'candidate_norm_rms':float(c.square().mean().sqrt())})
  return orig(self,*a,**kw)
 DenseObjectGrounder._competition=wrap
 try:
  with torch.no_grad(): deployment_cache(p.bundle.model,online,p.bundle.config,collect_diagnostics=True)
 finally: DenseObjectGrounder._competition=orig
 r={'schema':'clearvla-g-competition-logit-scale-v1','rows':rows}
 out=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/probes/g-competition-logit-scale-20261005.json'); out.write_text(json.dumps(r,indent=2)+'\\n'); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
