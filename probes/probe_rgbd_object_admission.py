"""Audit RGB-D correspondences against exact replay body identities.

Body IDs are evaluation labels only. Correspondence receives depth/cameras,
not scene state, color names, G ownership, or simulator object poses.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment, _official_task_assets
from probe_rgbd_correspondence import correspondence, matrix


def audit(depth, views, projections, masks):
    rows=[]
    for source,target in ((0,1),(1,0)):
        for control in ('calibrated','wrong_extrinsic_5cm'):
            transforms=[x.copy() for x in views]
            if control!='calibrated':transforms[target][0,3]+=.05
            law=correspondence(depth,transforms,projections,source,target)
            current=masks[source].reshape(masks[source].shape[0],-1)
            other=masks[target][:,law['v'],law['u']]
            obj=current.any(0);keep=law['accepted']&obj
            correct=(current&other).any(0)
            rows.append(dict(source=source,target=target,control=control,
                object_pixels=int(obj.sum()),accepted_object_pixels=int(keep.sum()),
                correct_object_pixels=int((keep&correct).sum()),
                wrong_object_pixels=int((keep&other.any(0)&~correct).sum()),
                background_pixels=int((keep&~other.any(0)).sum()),
                all_accepted_pixels=int(law['accepted'].sum())))
    return rows


def main():
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--index',type=int)
    a=q.parse_args();a.output.mkdir(exist_ok=False);plan=json.loads(a.plan.read_text())
    if a.index is None:
        results=[]
        for index,row in enumerate(plan):
            out=a.output/('worker_%02d'%index)
            with (a.output/('worker_%02d.log'%index)).open('w') as log:
                done=subprocess.run([sys.executable,'-u',__file__,'--plan',str(a.plan),'--output',str(out),'--index',str(index)],stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:raise RuntimeError('replay failed; see '+str(out)+'.log')
            results.extend(json.loads((out/'results.json').read_text()))
            (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        (a.output/'complete.json').write_text(json.dumps(dict(windows=len(results),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
        return
    row=plan[a.index];case=Path(row['case']);result=json.loads((case/'result.json').read_text())
    with np.load(case/'trajectory.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    official,*_=_official_task_assets();env=_environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'),show_gui=False)
    results=[]
    try:
        robot,scene=official.get_env_state_for_initial_condition(result['initial_state'])
        env.reset(robot_obs=robot,scene_obs=scene);obs=env.get_obs()
        objects={o.name:int(o.uid) for o in env.scene.movable_objects}
        for step in range(max(row['steps'])+1):
            if step:obs,*_=env.step(d['executed'][step-1].copy())
            if step not in row['steps']:continue
            depths=[];masks=[];views=[];projections=[]
            for camera,key in zip(env.cameras,('static','gripper')):
                rgb=obs['rgb_obs']['rgb_'+key]
                if not np.array_equal(rgb,d['rgb_'+key][step]):raise RuntimeError('non-exact RGB replay')
                rgb2,depth=camera.render()
                if not np.array_equal(rgb2,rgb):raise RuntimeError('second render mismatch')
                frame=p.getCameraImage(width=camera.width,height=camera.height,viewMatrix=matrix(camera,'view').T.flatten().tolist(),
                    projectionMatrix=matrix(camera,'projection').T.flatten().tolist(),physicsClientId=camera.cid,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                label=np.asarray(frame[4]).reshape(camera.height,camera.width)
                body=np.where(label<0,-1,label&((1<<24)-1))
                masks.append(np.stack([body==uid for uid in objects.values()]))
                depths.append(depth);views.append(matrix(camera,'view'));projections.append(matrix(camera,'projection'))
            results.append(dict(case=case.name,step=step,directions=audit(depths,views,projections,masks)))
            print(results[-1],flush=True)
        (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:env.close()


if __name__=='__main__':main()
