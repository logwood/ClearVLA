"""Evaluator-only binding interventions on fixed held-out expert observations.

Match existing slots with renderer masks; preserve learned real/null mass.
These are counterfactual action probes, never deployed inputs or task scores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import torch

from clearvla.mainline.model.target_binding import TargetBinding
from analyze_maniskill_spatial_repair import grounding
from probe_maniskill_failure import Capture,array,h5_history,load_policy
from probe_maniskill_spatial_grounding import SpatialCapture
from record_maniskill_npz import atomic_json


def scores(action,target):
    difference=action[:8]-target
    return dict(arm_rmse=float(np.sqrt(np.mean(difference[:,:6]**2))),
        translation_rmse=float(np.sqrt(np.mean(difference[:,:3]**2))),
        rotation_rmse=float(np.sqrt(np.mean(difference[:,3:6]**2))),
        gripper_accuracy=float(np.mean((action[:8,-1]>0)==(target[:,-1]>0))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--dino',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[x for x in json.loads((args.root/'control-heldout/summary.json').read_text())['rows']
        if x['split']=='val' and x['stage'] in ('reset','preclose','lift')]
    assert len(cases)==21
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory)
    rows=[]
    for arm in ('control','candidate'):
        args.checkpoint=args.root/(arm+'-pilot/checkpoints/latest.pt')
        policy=load_policy(args);spatial=SpatialCapture(policy);capture=Capture(policy)
        binder=policy.bundle.model.intent.organizer.shared_binder
        assert binder is not None
        for index,case in enumerate(cases):
            with h5py.File(args.data/'experts'/(case['episode']+'.hdf5')) as h:
                history=h5_history(h,case['step']);target=h['action'][case['step']:case['step']+8]
            with np.load(args.root/'control-heldout'/(case['label']+'.npz'),allow_pickle=False) as source:
                masks=source['masks']
                np.testing.assert_array_equal(np.stack([history.rgb_history[c][-1] for c in ('top','wrist')]),source['rgb'])
                np.testing.assert_array_equal(history.state,source['robot_state'])
            policy.reset()
            with torch.no_grad():baseline,_,tensors=spatial.infer_spatial(history)
            noise=array(spatial.sample.initial_physical_noise)
            saved={k:tensors[k] for k in ('g_density','binding_probability','rgb') if k in tensors}
            saved.update(masks=masks,baseline_action=baseline,target_first8=target)
            file=args.output/(arm+'_'+case['label']+'.npz')
            np.savez_compressed(file,**saved)
            measured=grounding(file)
            slots={name:next(v['slot'] for v in measured['views'] if v['object']==name) for name in ('red','green')}
            results={'baseline':scores(baseline,target)}
            probabilities={'baseline':tensors['binding_probability'].tolist()}
            modes=['none','red','green']
            for mode in modes:
                original=binder.forward
                before={}
                def override(*a,**k):
                    binding=original(*a,**k)
                    before['binding']=array(binding.log_probability.exp())
                    if mode=='none':return binding
                    selected=slots[mode]
                    assert bool(binding.supported[:,selected].all())
                    log=torch.full_like(binding.log_probability,-torch.inf)
                    log[:,selected]=binding.log_probability[:,:-1].logsumexp(-1)
                    log[:,-1]=binding.log_probability[:,-1]
                    value=TargetBinding(log,binding.supported)
                    value.validate(batch=1,objects=log.shape[1]-1,device=log.device)
                    return value
                policy.reset()
                with torch.no_grad(),patch.object(binder,'forward',override):
                    action,_,captured=capture.infer(history)
                g_delta=float(np.abs(captured['g_object_to_chart']-tensors['g_object_to_chart']).max())
                np.testing.assert_allclose(captured['g_object_to_chart'],tensors['g_object_to_chart'],rtol=3e-5,atol=3e-6)
                np.testing.assert_array_equal(array(capture.sample.initial_physical_noise),noise)
                np.testing.assert_array_equal(captured['binding_probability'][:,-1],before['binding'][:,-1])
                assert np.isfinite(action).all()
                results[mode]=scores(action,target)
                results[mode]['action_delta_rmse']=float(np.sqrt(np.mean((action[:8]-baseline[:8])**2)))
                results[mode]['g_chart_max_difference']=g_delta
                if mode!='none':results[mode]['action_delta_from_noop_rmse']=float(np.sqrt(np.mean((action[:8]-saved['none_action'][:8])**2)))
                probabilities[mode]=captured['binding_probability'].tolist()
                saved[mode+'_action']=action
            saved.update({name+'_binding':np.asarray(value) for name,value in probabilities.items()})
            np.savez_compressed(file,**saved)
            row=dict(arm=arm,label=case['label'],episode=case['episode'],stage=case['stage'],slots=slots,
                scores=results,binding=probabilities,grounding=measured,artifact=file.name)
            rows.append(row)
            atomic_json(args.output/'summary.json',dict(note=__doc__,rows=rows,
                admission='G chart agrees at atol=3e-6/rtol=3e-5; initial noise and each call original null mass are exact. Every case records a no-op action repeat; interpret effects against that numerical floor.'))
            print(json.dumps({k:row[k] for k in ('arm','label','scores')}),flush=True)
        atomic_json(args.output/(arm+'-provenance.json'),policy.deployment_health())
        del capture,spatial,policy,binder
        torch.cuda.empty_cache()


if __name__=='__main__':main()
