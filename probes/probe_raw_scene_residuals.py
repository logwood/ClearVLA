"""Localize incomplete raw-state replay; all labels here remain diagnostic.

RGB equality alone does not certify a simulator identity. This ledger combines
raw/render RGB and depth comparison with body/link interiors, and reports the
unscored remainder. It never modifies the sensor-only label producer.
"""
from pathlib import Path
import argparse,json
import numpy as np
from probe_raw_scene_replay import make_environment,scene_for,digest
from probe_rgbd_correspondence import matrix


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    files={f:s for row in json.loads(a.plan.read_text())['records'] for f,s in zip(row['raw_files'],row['sha256']) if scene_for(f)==a.scene}
    env=make_environment(Path(next(iter(files))).parent,a.scene)
    import pybullet as pb
    report=dict(complete=False,records=[],scope=__doc__,script_sha256=digest(__file__),scene=a.scene)
    for filename,expected in files.items():
        assert digest(filename)==expected
        with np.load(filename,allow_pickle=False) as z:
            data={k:z[k] for k in ('robot_obs','scene_obs','rgb_static','rgb_gripper','depth_static','depth_gripper')}
        env.scene.reset(data['scene_obs']);env.robot.reset(data['robot_obs'])
        actual,_=env.robot.get_observation()
        cameras=[];arrays={}
        for cam,key in zip(env.cameras,('static','gripper')):
            rgb,depth=cam.render()
            frame=pb.getCameraImage(width=cam.width,height=cam.height,
                viewMatrix=matrix(cam,'view').T.flatten().tolist(),projectionMatrix=matrix(cam,'projection').T.flatten().tolist(),
                physicsClientId=cam.cid,flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            np.testing.assert_array_equal(np.asarray(frame[2])[...,:3],rgb)
            label=np.asarray(frame[4]).reshape(cam.height,cam.width)
            body=np.where(label<0,-1,label&((1<<24)-1));link=np.where(label<0,-2,(label>>24)-1)
            rgb_error=np.abs(rgb.astype(float)-data['rgb_'+key]).mean(-1)
            depth_error=np.abs(depth-data['depth_'+key])
            interior=label>=0;padded=np.pad(label,1,constant_values=-999)
            for dy in range(3):
                for dx in range(3):interior&=padded[dy:dy+cam.height,dx:dx+cam.width]==label
            records=[]
            for value in np.unique(label):
                if value<0:continue
                uid=int(value&((1<<24)-1));lid=int((value>>24)-1)
                name=pb.getBodyInfo(uid,physicsClientId=cam.cid)[1].decode()
                lname='base' if lid<0 else pb.getJointInfo(uid,lid,physicsClientId=cam.cid)[12].decode()
                selected=label==value;inside=selected&interior
                records.append(dict(body=uid,link=lid,name=name,link_name=lname,pixels=int(selected.sum()),
                    rgb_unequal=int((selected&(rgb_error>0)).sum()),rgb_mae=float(rgb_error[selected].mean()),
                    depth_mae=float(depth_error[selected].mean()),interior_pixels=int(inside.sum()),
                    exact_rgb_interior=int((inside&(rgb_error==0)).sum()),
                    depth_close_interior=int((inside&(depth_error<1e-6)).sum()),
                    rgb_and_depth_close_interior=int((inside&(rgb_error==0)&(depth_error<1e-6)).sum())))
            # Diagnostic only: an independent scorer must preserve unscored
            # endpoints, rather than silently accepting a partially exact view.
            arrays['diagnostic_body_'+key]=body;arrays['diagnostic_link_'+key]=link
            arrays['diagnostic_rgb_exact_'+key]=rgb_error==0
            arrays['diagnostic_depth_close_'+key]=depth_error<1e-6
            arrays['diagnostic_interior_'+key]=interior
            cameras.append(dict(camera=key,entities=records,rgb_unequal=int((rgb_error>0).sum()),
                depth_max=float(depth_error.max()),rgb_mae=float(rgb_error.mean())))
        target=a.output/(Path(filename).stem+'-diagnostic.npz');np.savez_compressed(target,**arrays)
        report['records'].append(dict(path=filename,raw_sha256=expected,diagnostic_path=str(target),
            diagnostic_sha256=digest(target),robot_state_max_abs=(np.abs(actual-data['robot_obs'])).tolist(),cameras=cameras))
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
