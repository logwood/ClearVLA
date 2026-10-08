"""Bounded matched training and full closed-loop qualification, one GPU at a time."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();r=args.root
    repo=Path(__file__).resolve().parents[1]
    admission=json.loads((r/'preflight-admission.json').read_text())
    if admission['status']!='passed':raise RuntimeError('Unqualified preflight')
    if (r/'pipeline-status.json').exists():raise FileExistsError('Use a fresh bounded pipeline attempt')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
        raise RuntimeError('Training requires clean committed source')
    state={'status':'running','source_commit':head,'steps':[]}
    def save():
        temp=r/'pipeline-status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(r/'pipeline-status.json')
    def run(name,command):
        state['active_stage']=name;save()
        log=r/(name+'.log')
        if log.exists():raise FileExistsError(log)
        row={'name':name,'started_unix':time.time(),'command':command,'log':str(log)}
        state['steps'].append(row);save()
        with log.open('w') as stream:
            proc=subprocess.Popen(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=proc.pid;save();row['returncode']=proc.wait()
        row['finished_unix']=time.time();save()
        if row['returncode']:raise RuntimeError(name+' failed; inspect its log')
    py=sys.executable
    data='/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2'
    dino='/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'
    source='/data/senwang/clearvla/experiments/maniskill/20261007-stackcube-latest-ffb6c39c/train/checkpoints/best.pt'
    try:
        for arm in ('control','candidate'):
            output=r/(arm+'-pilot')
            run(arm+'-pilot',[py,'-B','-u','scripts/train_mainline_memory_capped.py','--memory-cap-gib','16',
                '--config','configs/mainline/maniskill_spatial_'+arm+'_20261008.json',
                '--output-dir',str(output),'--device','cuda:0','--init-checkpoint',source,
                '--init-model-contract-migration','maniskill_spatial_repair_v1',
                '--init-training-clock','checkpoint','--init-optimizer-state','checkpoint'])
            rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
            epoch=[x for x in rows if x.get('kind')=='epoch']
            if len(epoch)!=1 or epoch[0]['step']!=6648:raise RuntimeError('Incomplete 256-update pilot')
        for arm in ('control','candidate'):
            checkpoint=str(r/(arm+'-pilot/checkpoints/latest.pt'))
            common=['--checkpoint',checkpoint,'--data',data,'--dino',dino]
            for mode in ('heldout','placements'):
                run(arm+'-'+mode,[py,'-B','-u','scripts/probe_maniskill_spatial_grounding.py',mode,
                    *common,'--output',str(r/(arm+'-'+mode))])
            run(arm+'-eval18',[py,'-B','-u','scripts/record_maniskill_npz.py',
                '--checkpoint',checkpoint,'--output-dir',str(r/(arm+'-eval18')),
                '--episodes','18','--eval-seed','5000000','--policy-seed','0',
                '--max-episode-steps','400','--replan-steps','8',
                '--t5-condition',data+'/language/t5_xxl.pt','--dinov3-model',dino])
        state['status']='complete';state['active_stage']=None;save()
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);save();raise


if __name__=='__main__':main()
