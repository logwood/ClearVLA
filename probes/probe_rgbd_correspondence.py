"""Admit RGB-D camera correspondence from observed depth and robot kinematics.

No scene/object pose is used to construct a match. Scene labels, when supplied,
are restricted to audit. Raw calibration must be checked on every scene.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
from clearvla.benchmarks.calvin_eval import _environment, _official_task_assets


def matrix(camera,name):
    alternatives={'view':('viewMatrix','view_matrix'),'projection':('projectionMatrix','projection_matrix')}
    value=next(getattr(camera,n) for n in alternatives[name] if hasattr(camera,n))
    return np.asarray(value).reshape(4,4).T


def correspondence(depths,views,projections,source,target):
    depth=depths[source];h,w=depth.shape
    yy,xx=np.mgrid[:h,:w]
    x=(2*(xx+.5)/w-1).ravel();y=(1-2*(yy+.5)/h).ravel();d=depth.ravel()
    P=projections[source]
    z=(-P[2,2]*d+P[2,3])/d
    clip=np.stack((x,y,z,np.ones_like(z)))
    world=np.linalg.inv(P@views[source])@clip;world/=world[3:]
    target_cam=views[target]@world
    projected=projections[target]@target_cam;projected/=projected[3:]
    th,tw=depths[target].shape
    u=(projected[0]+1)*tw/2-.5;v=(1-projected[1])*th/2-.5
    ui=np.rint(np.clip(u,0,tw-1)).astype(int);vi=np.rint(np.clip(v,0,th-1)).astype(int)
    visible=(u>=0)&(u<=tw-1)&(v>=0)&(v<=th-1)&(-target_cam[2]>0)
    error=np.abs(depths[target][vi,ui]+target_cam[2])
    # Sensor agreement, not allocation/identity confidence or a learned mask.
    accepted=visible & (error<.003)
    return {'u':ui,'v':vi,'visible':visible,'accepted':accepted,'depth_error':error,
            'xy':np.stack((u,v),-1),'source_shape':(h,w)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    root=Path('/data/senwang/data/calvin/raw/task_ABC_D')
    ranges=np.load(root/'training/scene_info.npy',allow_pickle=True).item()
    _official_task_assets()
    rows=[];env=_environment(root,show_gui=False)
    try:
        examples=[]
        for scene,(lo,hi) in ranges.items():
            for f in (.2,.5,.8):examples.append(('training',scene,int(lo+(hi-lo)*f)))
        examples += [('validation','calvin_scene_D',n) for n in (0,20000,40000)]
        for split,scene,index in examples:
            file=root/split/('episode_%07d.npz'%index)
            if not file.exists():continue
            with np.load(file,allow_pickle=False) as z:
                rgb=[z['rgb_static'],z['rgb_gripper']];depth=[z['depth_static'],z['depth_gripper']]
                robot=z['robot_obs']
            env.robot.reset(robot)
            for c in env.cameras:c.render()
            views=[matrix(c,'view') for c in env.cameras];projections=[matrix(c,'projection') for c in env.cameras]
            details=[]
            for source,target in ((0,1),(1,0)):
                corr=correspondence(depth,views,projections,source,target)
                color=rgb[source].reshape(-1,3).astype(float)/255
                other=rgb[target][corr['v'],corr['u']].astype(float)/255
                error=np.abs(color-other).mean(-1);keep=corr['accepted']
                details.append({'source':source,'target':target,'visible_pixels':int(corr['visible'].sum()),
                    'accepted_pixels':int(keep.sum()),'color_mae_quantiles':np.quantile(error[keep],[.1,.5,.9]).tolist() if keep.any() else [],
                    'depth_error_quantiles':np.quantile(corr['depth_error'][corr['visible']],[.1,.5,.9]).tolist() if corr['visible'].any() else []})
            rows.append({'scene':scene,'split':split,'frame':index,'directions':details})
            (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(scene,index,details,flush=True)
        (a.output/'complete.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frames':len(rows)}))
    finally:env.close()


if __name__=='__main__':main()
