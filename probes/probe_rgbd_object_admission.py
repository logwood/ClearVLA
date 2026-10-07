"""Audit RGB-D correspondences against exact replay body identities.

Body IDs are evaluation labels only. Correspondence receives depth/cameras,
not scene state, color names, G ownership, or simulator object poses.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment, _official_task_assets
from probe_rgbd_correspondence import correspondence, matrix
from clearvla.vision.observed_flow import observed_rgb_flow
from clearvla.vision.sensor_geometry import camera_views, photometric_support


# The CALVIN renderer uses Python 3.9; importing mainline's package would load
# the Python 3.12 model. Load this training-label leaf without that package.
_label_file=Path(__file__).resolve().parents[1]/'clearvla/mainline/data/identity_correspondence.py'
_spec=importlib.util.spec_from_file_location('identity_label_sampling_audit',_label_file)
_labels=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labels)


def sampled_admission(accepted, current, other, shape, names):
    """Count exactly the production 32x32 label grid; body IDs only audit it.

    Count share is not loss/gradient share. Online training-mask support can
    further filter conditional-v2 identity pairs, but not source prediction.
    """
    y,x=_labels.IdentityLabelProducer._sample(shape)
    index=y*shape[1]+x
    valid=np.asarray(accepted).reshape(-1)[index]
    source=current.reshape(len(names),-1)[:,index]
    target=other.reshape(len(names),-1)[:,index]
    source_obj=source.any(0);target_obj=target.any(0)
    same=(source&target).any(0)
    count=int(valid.sum());objects=int((valid&source_obj).sum())
    rows={name:dict(visible=int(source[i].sum()),
        accepted=int((valid&source[i]).sum()),
        correct=int((valid&source[i]&target[i]).sum()),
        wrong_object=int((valid&source[i]&target_obj&~target[i]).sum()),
        background=int((valid&source[i]&~target_obj).sum()))
        for i,name in enumerate(names)}
    correct=int((valid&same).sum())
    wrong=int((valid&source_obj&target_obj&~same).sum())
    background=int((valid&source_obj&~target_obj).sum())
    assert correct+wrong+background==objects
    assert sum(r['accepted'] for r in rows.values())==objects
    return dict(sample_count=len(index),all_accepted=count,
        accepted_source_object=objects,accepted_source_nonobject=count-objects,
        accepted_target_object=int((valid&target_obj).sum()),
        object_pair_count_fraction=objects/count if count else None,
        correct_object=correct,wrong_object=wrong,object_to_background=background,
        per_object=rows)


def audit(depth, views, projections, masks, rgbs, names):
    rows=[]
    for source,target in ((0,1),(1,0)):
        for control in ('calibrated','wrong_extrinsic_5cm'):
            transforms=[x.copy() for x in views]
            if control!='calibrated':transforms[target][0,3]+=.05
            law=correspondence(depth,transforms,projections,source,target)
            current=masks[source].reshape(masks[source].shape[0],-1)
            other=masks[target][:,law['v'],law['u']]
            accepted=photometric_support(law,rgbs[source],rgbs[target])
            obj=current.any(0);keep=accepted&obj
            correct=(current&other).any(0)
            rows.append(dict(source=source,target=target,control=control,
                object_pixels=int(obj.sum()),accepted_object_pixels=int(keep.sum()),
                correct_object_pixels=int((keep&correct).sum()),
                wrong_object_pixels=int((keep&other.any(0)&~correct).sum()),
                background_pixels=int((keep&~other.any(0)).sum()),
                # Retain this legacy field, but identify its pre-RGB scope.
                all_accepted_pixels=int(law['accepted'].sum()),
                all_depth_accepted_pixels=int(law['accepted'].sum()),
                all_photometric_accepted_pixels=int(accepted.sum()),
                production_sampling=sampled_admission(accepted,current,other,depth[source].shape,names)))
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
        (a.output/'complete.json').write_text(json.dumps(dict(windows=len(results),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            production_label_source_sha256=hashlib.sha256(_label_file.read_bytes()).hexdigest(),
            scope='exact RGB replay; production 32x32 sensor label counts, before model training-mask support; not loss or gradient attribution',
            legacy_all_accepted_pixels='depth acceptance before photometric filtering')))
        return
    row=plan[a.index];case=Path(row['case']);result=json.loads((case/'result.json').read_text())
    calibration=json.loads((Path(__file__).resolve().parents[1]/'clearvla/mainline/assets/calvin_rgbd_joint_geometry_v1.json').read_text())
    with np.load(case/'trajectory.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    official,*_=_official_task_assets();env=_environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'),show_gui=False)
    results=[]
    try:
        robot,scene=official.get_env_state_for_initial_condition(result['initial_state'])
        env.reset(robot_obs=robot,scene_obs=scene);obs=env.get_obs()
        objects={o.name:int(o.uid) for o in env.scene.movable_objects}
        snapshots={}
        required=sorted(set(row['steps']) | {step+offset for step in row['steps'] for offset in (4,8) if step+offset<len(d['rgb_static'])})
        for step in range(max(required)+1):
            if step:obs,*_=env.step(d['executed'][step-1].copy())
            if step not in required:continue
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
            snapshots[step]=masks
            if step not in row['steps']:continue
            derived=camera_views(d['robot_obs'][step],calibration)
            error=max(float(np.max(np.abs(x-y))) for x,y in zip(derived,views))
            # Physics constraints can flex by sub-millimeters during motion.
            # Record this independently; factual match precision, not reset-only
            # matrix agreement, admits the production sensor correspondence.
            results.append(dict(case=case.name,step=step,production_view_max_abs_error=error,directions=audit(depths,derived,projections,masks,[d['rgb_static'][step],d['rgb_gripper'][step]],list(objects))))
            print(dict(case=case.name,step=step,production_view_max_abs_error=error),flush=True)
        for result in results:
            step=result['step'];temporal=[]
            for delta in (4,8):
                if step+delta not in snapshots:continue
                for camera,key in enumerate(('static','gripper')):
                    flow=observed_rgb_flow(d['rgb_'+key][step],d['rgb_'+key][step+delta])
                    masks=snapshots[step][camera];target=snapshots[step+delta][camera]
                    h,w=flow['accepted'].shape
                    xy=flow['xy'];x=np.rint(np.clip(xy[...,0],0,w-1)).astype(int);y=np.rint(np.clip(xy[...,1],0,h-1)).astype(int)
                    other=target[:,y,x];obj=masks.any(0);keep=flow['accepted']&obj;same=(masks&other).any(0)
                    temporal.append(dict(camera=camera,offset=delta,object_pixels=int(obj.sum()),accepted_object_pixels=int(keep.sum()),correct_object_pixels=int((keep&same).sum()),wrong_object_pixels=int((keep&other.any(0)&~same).sum()),background_pixels=int((keep&~other.any(0)).sum()),
                        all_accepted_pixels=int(flow['accepted'].sum()),
                        production_sampling=sampled_admission(flow['accepted'],masks,other,(h,w),list(objects))))
            result['temporal']=temporal
        (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:env.close()


if __name__=='__main__':main()
