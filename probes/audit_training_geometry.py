"""Audit selected raw training segments and closed-loop geometric support.

Simulator object positions are audit-only. Proximity is not a contact label.
"""
from pathlib import Path
import argparse, json, hashlib, time
import h5py
import numpy as np
import yaml

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def stats(x):
    x=np.asarray(x, float)
    return dict(zip(['min','p10','p25','median','p75','p90','max'], np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist())) if x.size else {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--rollout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);m=json.loads(a.manifest.read_text());rows=[];windows=[];start_time=time.time()
    colors=['red','blue','pink'];xyz=np.array([6,12,18])[:,None]+np.arange(3)
    scene_ranges=np.load(Path(m['raw_source'])/'training/scene_info.npy',allow_pickle=True).item()
    conf=Path('/home/sen.wang/workspace/robotics/benchmarks/calvin/calvin_env/conf/scene')
    scene_orders={}
    for name in [*scene_ranges,'calvin_scene_D']:
        cfg=yaml.safe_load((conf/(name+'.yaml')).read_text())
        scene_orders[name]=list(cfg['objects']['movable_objects'])
    for split in ['train','val','test']:
        for episode in m['splits'][split]:
            with h5py.File(Path(m['cached_prefix_root'])/(episode+'.hdf5')) as f:
                meta=dict(f.attrs);state=f['state'][:];action=f['action'][:];action_state=f['action_state'][:]
            color=meta['task'].split('_')[1];ci=colors.index(color);direction=-1 if meta['task'].endswith('left') else 1
            begin=int(meta['source_start']);end=int(meta['source_end']);context=int(meta['context_start']);local=begin-context
            scene={}
            scene_name='calvin_scene_D' if meta['source_split']=='validation' else next(name for name,(lo,hi) in scene_ranges.items() if int(lo)<=begin<=end<=int(hi))
            # ABC source observations use their own YAML insertion order.
            # The merged training .hydra config is D and is not this authority.
            order=scene_orders[scene_name]
            raw_xyz=np.array([6+6*order.index('block_'+c) for c in colors])[:,None]+np.arange(3)
            for t in sorted(set(range(begin,end+1,8))|{end}):
                with np.load(Path(m['raw_source'])/meta['source_split']/('episode_%07d.npz'%t),allow_pickle=False) as z:scene[t]=z['scene_obs'][raw_xyz].copy()
            obj0=scene[begin];objend=scene[end];dist=np.linalg.norm(state[local,:2]-obj0[:,:2],axis=-1)
            displace=direction*(objend[ci,0]-obj0[ci,0]);selected=[]
            for t,objs in scene.items():
                j=t-context;remaining=end-t;d=np.linalg.norm(state[j,:2]-objs[:,:2],axis=-1);act=action[j:min(j+8,end-context)]
                if not len(act):continue
                near=bool(d[ci]<.07 and abs(float(state[j,2]-objs[ci,2]))<.09)
                row={'episode':episode,'split':split,'task':meta['task'],'age':t-begin,'remaining':remaining,
                     'target_xy_distance':float(d[ci]),'target_nearest':bool(np.argmin(d)==ci),'near_proxy':near,
                     'target_progress':float(direction*(objs[ci,0]-obj0[ci,0])),
                     'x_command_along_task':float(direction*act[:,0].mean()),'z_command':float(act[:,2].mean()),
                     'gripper_closed_fraction':float(np.mean(act[:,-1]<0)),'tcp_xyz':state[j,:3].tolist()}
                windows.append(row);selected.append(row)
            rows.append({'episode':episode,'split':split,'scene':scene_name,'task':meta['task'],'instruction':meta['instruction'],'source_start':begin,'source_end':end,
                'duration':end-begin,'target_start_xyz':obj0[ci].tolist(),'tcp_start_xyz':state[local,:3].tolist(),
                'start_target_xy_distance':float(dist[ci]),'start_target_nearest':bool(np.argmin(dist)==ci),
                'start_target_nearest_margin':float(np.sort(dist)[1]-np.sort(dist)[0]),
                'net_target_progress':float(displace),'max_target_progress':max([r['target_progress'] for r in selected]+[float(displace)]),
                'start_gripper_closed':bool(action[local,-1]<0),
                'action_state_previous_max_abs':float(np.max(np.abs(action_state[1:]-action[:-1]))),
                'last16_x_along_task':float(direction*action[max(local,end-context-16):end-context,0].mean()),
                'last16_z':float(action[max(local,end-context-16):end-context,2].mean())})
            if len(rows)%250==0:print('episodes',len(rows),'elapsed',round(time.time()-start_time,1),flush=True)
    eval_rows=[]
    for case in sorted(a.rollout.glob('case_*')):
        if not case.is_dir() or not (case/'trajectory.npz').exists():continue
        result=json.loads((case/'result.json').read_text());task=result['task'];ci=colors.index(task.split('_')[1]);direction=-1 if task.endswith('left') else 1
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:
            state=z['robot_obs'];objects=z['scene_obs'][:,xyz];actions=z['executed']
            for t in range(0,len(actions),8):
                d=np.linalg.norm(state[t,:2]-objects[t,:,:2],axis=-1);act=actions[t:t+8]
                eval_rows.append({'case':case.name,'task':task,'success':result['success'],'age':t,
                   'target_xy_distance':float(d[ci]),'target_nearest':bool(np.argmin(d)==ci),
                   'target_progress':float(direction*(objects[t,ci,0]-objects[0,ci,0])),
                   'x_command_along_task':float(direction*act[:,0].mean()),'z_command':float(act[:,2].mean()),
                   'gripper_closed_fraction':float(np.mean(act[:,-1]<0)),
                   'near_proxy':bool(d[ci]<.07 and abs(float(state[t,2]-objects[t,ci,2]))<.09)})
    summary={}
    for split in ['train','val','test']:
        rs=[r for r in rows if r['split']==split];ws=[w for w in windows if w['split']==split];near=[w for w in ws if w['near_proxy']]
        summary[split]={'episodes':len(rs),'duration':stats([r['duration'] for r in rs]),
           'start_target_xy_distance':stats([r['start_target_xy_distance'] for r in rs]),
           'start_target_nearest_fraction':float(np.mean([r['start_target_nearest'] for r in rs])),
           'window_target_nearest_fraction':float(np.mean([r['target_nearest'] for r in ws])),
           'net_target_progress':stats([r['net_target_progress'] for r in rs]),
           'max_target_progress_under_10cm_fraction':float(np.mean([r['max_target_progress']<.10 for r in rs])),
           'near_proxy_windows':len(near),'near_proxy_closed_fraction':float(np.mean([r['gripper_closed_fraction'] for r in near])),
           'last16_z_command':stats([r['last16_z'] for r in rs]),
           'by_task':{task:{'start_distance':stats([r['start_target_xy_distance'] for r in rs if r['task']==task]),'progress':stats([r['net_target_progress'] for r in rs if r['task']==task])} for task in m['task_order']}}
    summary['evaluation_initial']=[r for r in eval_rows if r['age']==0]
    summary['identity']={'manifest':str(a.manifest),'sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scene_orders':scene_orders,'scene_ranges':{k:[int(v) for v in x] for k,x in scene_ranges.items()},'scope':'source-specific ABC/D object order verified from scene_info and simulator YAML; raw observed annotation segments only; no absorbing synthetic suffix; 8-frame sampling; proximity is not contact'}
    dump(a.output/'episodes.json',rows);dump(a.output/'windows.json',windows);dump(a.output/'evaluation_windows.json',eval_rows);dump(a.output/'summary.json',summary)
    print('COMPLETE',json.dumps(summary['train']),flush=True)

if __name__=='__main__':main()
