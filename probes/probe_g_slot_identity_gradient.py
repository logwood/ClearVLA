from __future__ import annotations
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
from clearvla.mainline.runtime.sampling import deployment_cache
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import ExecutedWorldSnapshot
from scripts.probe_calvin_internal_layers_npz import _history
def main():
    ckpt=Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/short-bs8-gpu0-r9/checkpoints/best.pt')
    obs=Path('/data/senwang/clearvla/experiments/dinov3-online-task-global-optimized-20261002/probes/standard-observations-20261003/standard.npz')
    t5=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt')
    device=torch.device('cuda')
    policy=ClearVLACheckpointPolicy(ckpt,device=device,t5_condition=t5,seed=0)
    base=_history(obs)
    world=ExecutedWorldSnapshot(anchor_index=0,rgb_history=base.rgb_history,state=base.state.copy(),action_state=base.action_state.copy(),commands=base.executed_action_history[:4].copy(),visual_offsets=np.zeros(3,dtype=np.int64),observed=True)
    history=replace(base,time_index=4,previous_state=base.state.copy(),executed_world=world)
    _,online=policy.act_with_input(history,'go push the blue block right')
    policy.bundle.model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            cache,_,metrics=policy.bundle.model.encode_online(online,training_mask=False,geometry_supervision=False,collect_diagnostics=True)
            loss=(cache.top.belief.content.float().square().mean()+cache.top.intent.public_interval_carrier.float().square().mean()+cache.top.predicted_dynamics.semantic_delta.float().square().mean())
    loss.backward()
    selected={}
    for name,p in policy.bundle.model.named_parameters():
        if p.grad is not None and (name.startswith('grounding.grounder.') or name.startswith('intent.') or 'progressive_grounding_address' in name):
            selected[name]={'rms':float(p.grad.detach().float().square().mean().sqrt()),'finite':bool(torch.isfinite(p.grad).all())}
    result={'schema':'clearvla-g-slot-identity-gradient-v1','loss':float(loss.detach().float()),'loss_finite':bool(torch.isfinite(loss).all()),'parameter_count':len(selected),'parameters':selected}
    Path('/data/senwang/clearvla/experiments/dinov3-s-interval-repair-20261004/probes/g-slot-identity-gradient-20261005.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\\n')
    print(json.dumps({'loss':result['loss'],'parameter_count':result['parameter_count'],'nonfinite':[k for k,v in selected.items() if not v['finite']]},indent=2))
if __name__=='__main__': main()
