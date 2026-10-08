"""Compare real training windows to causal deployment inputs, without updates."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np, torch
from torch.utils.data import default_collate
from probe_maniskill_failure import h5_history, load_policy, native
from record_maniskill_npz import atomic_json
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.runtime.identity import v120_normalizer_fingerprint
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.admission import STACKCUBE_INSTRUCTION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','checkpoint','dino','teacher','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    policy=load_policy(a)
    # Inference intentionally reconstructs only deployment data fields. Read
    # the original serialized training context for the exact dataset keys.
    cfg=config_from_mapping(json.loads((a.checkpoint.parent.parent/'run_context.json').read_text())['config'])
    for name in ('top','bottom','dimensions','observation'):
        assert getattr(cfg,name)==getattr(policy.bundle.config,name),name
    bundle=load_mainline_data(cfg)
    fingerprints={}
    for name in ('action_normalizer','state_normalizer'):
        u=v120_normalizer_fingerprint(getattr(bundle,name))
        v=v120_normalizer_fingerprint(getattr(policy.bundle,name))
        assert u==v,(name,u,v)
        fingerprints[name]=u
    cases=[x for x in json.loads((a.teacher/'teacher.json').read_text())['rows'] if x['stage'] in ('reset','approach','close')]
    indices={split:{(bundle.episodes[r.episode_idx].episode_id,int(r.center)):i for i,r in enumerate(ds.base.refs)} for split,ds in bundle.datasets.items()}
    rows=[]
    with torch.no_grad():
        for c in cases:
            i=indices[c['split']][(c['episode'],c['step'])]
            raw=default_collate([bundle.datasets[c['split']][i]])
            batch=to_training_batch(raw,goal=bundle.goal,config=cfg,device=policy.device,visual_encoder=policy.encoder.pipeline)
            with h5py.File(a.data/'experts'/(c['episode']+'.hdf5')) as h:history=h5_history(h,c['step'])
            policy.reset()
            action,online=policy.act_with_input(history,STACKCUBE_INSTRUCTION)
            differences={}
            for block,names in {'observation':('raw_rgb','dino_history'),'history':('state','action_state','state_history','executed_action_history','codec_gripper_boundary'),'goal':('tokens','mask')}.items():
                for name in names:
                    u=getattr(getattr(batch.online,block),name);v=getattr(getattr(online,block),name)
                    assert u.shape==v.shape,(block,name,u.shape,v.shape)
                    d=(u.float()-v.float()).abs()
                    differences[block+'.'+name]={'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                        'training_stride':list(u.stride()),'deployment_stride':list(v.stride())}
            for name in ('state_offsets','action_offsets','state_observed','action_executed'):
                u=getattr(batch.online.history.timing,name);v=getattr(online.history.timing,name)
                torch.testing.assert_close(u,v,atol=0,rtol=0)
            sampled=sample_action(policy.bundle.model,batch.online,cfg,generator=torch.Generator(device=policy.device).manual_seed(0),deployment_fastpath=getattr(policy,'deployment_fastpath',False))
            training_action=native(sampled,policy)
            row={k:c[k] for k in ('split','episode','stage','step')}
            row.update(ingress=differences,action_max_abs=float(np.max(np.abs(training_action-action))))
            if row['action_max_abs']>5e-5:
                # Repeating each exact input distinguishes ingress drift from
                # low-precision / parallel-reduction variation in the model.
                repeats={}
                for label,inp,reference in (('training',batch.online,training_action),('deployment',online,action)):
                    values=[]
                    for _ in range(3):
                        s=sample_action(policy.bundle.model,inp,policy.bundle.config,generator=torch.Generator(device=policy.device).manual_seed(0),deployment_fastpath=getattr(policy,'deployment_fastpath',False))
                        values.append(float(np.max(np.abs(native(s,policy)-reference))))
                    repeats[label]=values
                row['same_input_repeat_max_abs']=repeats
            rows.append(row)
            atomic_json(a.output/'ingress.json',{'normalizer_fingerprints':fingerprints,'rows':rows,'complete':len(rows)==len(cases)})
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
