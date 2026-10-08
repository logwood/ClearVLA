"""Compare saved same-observation object reads and standard v1/v2 trajectories.

Conditional real-K object differences do not measure absolute allocation mass.
After the first plan, same-clock trajectory differences include state feedback;
this report does not identify a module-level causal rescue.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import statistics
import numpy as np


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values)) if values else dict(count=0)


def factual(path):
    data=json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('incomplete factual probe')
    records={(r['case_id'],r['step']):r for r in data['records']}
    if len(records)!=len(data['records']):
        raise ValueError('duplicate observation')
    return data['identity'],records


def object_pairs(row,camera):
    items={x['object_index']:x for x in row['physical_identity']
           if x['camera']==camera and x['object_supported_pixels']>0 and x['object_source_mass']>0}
    result={}
    for a,b in itertools.combinations(sorted(items),2):
        left,right=items[a],items[b]
        p,q=np.asarray(left['K_distribution']),np.asarray(right['K_distribution'])
        if abs(p.sum()-1)>1e-5 or abs(q.sum()-1)>1e-5:
            raise ValueError('unsupported or unnormalized body-conditional K law')
        result[(a,b)]={'tv':float(np.abs(p-q).sum()/2),'same_argmax':int(p.argmax()==q.argmax())}
    return result


def panel(path):
    d=json.loads((path/'summary.json').read_text())
    if not d['complete'] or d['trials']!=18 or d['execute_rows']!=8 or d['max_steps']!=360:
        raise ValueError('not a complete standard R8 panel')
    return d,{r['index']:r for r in d['records']}


def main():
    ap=argparse.ArgumentParser()
    for name in ('old-factual','new-factual','old-panel','new-panel','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    older,old=factual(a.old_factual);newer,new=factual(a.new_factual)
    if old.keys()!=new.keys():raise ValueError('observations do not match')
    paired={camera:{'old_tv':[],'new_tv':[],'old_same_argmax':[],'new_same_argmax':[]} for camera in (0,1)}
    for key,x in old.items():
        y=new[key]
        if x['object_names']!=y['object_names']:raise ValueError('physical body names differ')
        for c in (0,1):
            for field in ('object_supported_pixels','visible_pixels'):
                if x['coverage'][c][field]!=y['coverage'][c][field]:
                    raise ValueError('different physical or producer support')
            xp,yp=object_pairs(x,c),object_pairs(y,c)
            if xp.keys()!=yp.keys():raise ValueError('different supported object pairs')
            for key2 in xp:
                for version,source in (('old',xp),('new',yp)):
                    for field in ('tv','same_argmax'):
                        paired[c][version+'_'+field].append(source[key2][field])
    osum,op=panel(a.old_panel);nsum,npanel=panel(a.new_panel)
    if osum['seed']!=nsum['seed'] or osum['controller_anchor']!=nsum['controller_anchor']:
        raise ValueError('panel control contract differs')
    lost=sorted(cid for cid in op if op[cid]['success'] and not npanel[cid]['success'])
    gained=sorted(cid for cid in op if not op[cid]['success'] and npanel[cid]['success'])
    rows=[]
    for cid in lost:
        data=[]
        for source in (op,npanel):
            with np.load(Path(source[cid]['result_path']).parent/'trajectory.npz',allow_pickle=False) as z:
                data.append({k:z[k] for k in ('robot_obs','scene_obs','raw_chunks','executed')})
        for k in ('robot_obs','scene_obs'):
            if not np.array_equal(data[0][k][0],data[1][k][0]):raise ValueError('initial state differs')
        length=min(len(x['executed']) for x in data)
        delta=data[1]['raw_chunks'][0,:8,:6]-data[0]['raw_chunks'][0,:8,:6]
        tcp=np.linalg.norm(data[1]['robot_obs'][:length+1,:3]-data[0]['robot_obs'][:length+1,:3],axis=-1)
        grip=np.flatnonzero(data[0]['executed'][:length,6]!=data[1]['executed'][:length,6])
        threshold=np.flatnonzero(tcp>.001)
        rows.append(dict(case=cid,initial_first8_arm_rms_difference=float(np.sqrt(np.mean(delta**2))),
            first_recorded_tcp_difference_over_1mm=int(threshold[0]) if len(threshold) else None,
            first_recorded_gripper_disagreement=int(grip[0]) if len(grip) else None))
    result=dict(complete=True,scope=__doc__,old_identity=older,new_identity=newer,
                aligned_observations=len(old),physical_and_producer_support_identical=True,
                cameras={str(c):{k:stats(v) for k,v in values.items()} for c,values in paired.items()},
                panel=dict(old_successes=osum['successes'],new_successes=nsum['successes'],lost=lost,gained=gained),
                earliest_descriptive_divergence=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('old_identity','new_identity')},indent=2))


if __name__=='__main__':main()
