"""Run ordinary two-update training with a read-only first-batch VJP receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from clearvla.mainline.model.component_contracts import legacy_named_parameters
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.spatial_supervision import stackcube_grounding_terms


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--memory-cap-gib',type=float,default=16.)
    args,rest=p.parse_known_args()
    if args.receipt.exists():raise FileExistsError(args.receipt)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Select one available GPU')
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    original=MainlineTrainingEngine._forward_encoded
    complete=False
    def observed(engine,batch,*,encoded,**kw):
        nonlocal complete
        if complete or not engine.model.training:
            return original(engine,batch,encoded=encoded,**kw)
        complete=True
        versions=[p._version for p in engine.model.parameters()]
        rng=torch.cuda.get_rng_state().clone()
        terms=stackcube_grounding_terms(encoded.training_state.top.facts,batch.online.observation.raw_rgb[:,-1])
        owners=[(n,p) for n,p in legacy_named_parameters(engine.model) if p.requires_grad and ('grounder' in n or n.startswith('observation.'))]
        grads=torch.autograd.grad(terms['spatial_grounding'],[p for _,p in owners],allow_unused=True,retain_graph=True)
        rows=[]
        for (name,param),grad in zip(owners,grads):
            if grad is None:continue
            if not torch.isfinite(grad).all():raise FloatingPointError(name)
            rows.append(dict(name=name,rms=float(grad.float().square().mean().sqrt()),max=float(grad.abs().max())))
        if not any('grounder' in x['name'] and x['rms']>0 for x in rows):raise RuntimeError('Missing G binder parameter VJP')
        if not any(x['name'].startswith('observation.') and x['rms']>0 for x in rows):raise RuntimeError('Missing visual producer parameter VJP')
        assert versions==[p._version for p in engine.model.parameters()]
        assert torch.equal(rng,torch.cuda.get_rng_state()),'Spatial audit consumed CUDA RNG'
        arrays={'raw_rgb':batch.online.observation.raw_rgb.detach().float().cpu().numpy(),
                'action_target':batch.action_target.normalized.detach().float().cpu().numpy()}
        velocity=engine.model.velocity;calls=0
        def capture(*a,**k):
            nonlocal calls
            value=velocity(*a,**k)
            arrays['velocity_'+str(calls)]=value.bottom.physical_velocity.detach().float().cpu().numpy()
            arrays['time_'+str(calls)]=k['time'].detach().float().cpu().numpy()
            calls+=1
            return value
        with patch.object(engine.model,'velocity',capture):
            ledger,metrics=original(engine,batch,encoded=encoded,**kw)
        receipt=dict(status='passed',global_step=engine.global_step,
                     terms={k:float(v.detach()) for k,v in terms.items()},
                     owner_vjps=rows,loss_groups={k:float(v.detach()) for k,v in ledger.groups.items()},
                     objective_weight=engine.config.objectives.maniskill_spatial_grounding,
                     parameters_unchanged_by_probe=True,cuda_rng_unchanged_by_probe=True,
                     maximum_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
        np.savez_compressed(args.receipt.with_suffix('.npz'),**arrays)
        print('SPATIAL_PREFLIGHT '+json.dumps({k:v for k,v in receipt.items() if k!='owner_vjps'}),flush=True)
        return ledger,metrics
    sys.argv=['clearvla.mainline.train',*rest]
    with patch.object(MainlineTrainingEngine,'_forward_encoded',observed):
        runpy.run_module('clearvla.mainline.train',run_name='__main__',alter_sys=True)
    if not complete:raise RuntimeError('No ordinary training batch qualified')


if __name__=='__main__':main()
