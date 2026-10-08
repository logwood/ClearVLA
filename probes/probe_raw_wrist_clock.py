"""Audit the observation-time wrist camera against joint-reset forward kinematics.

Raw TCP and joints are both observed sensors. Simulator state is used only by
this audit to score renderer agreement, never by a training-label producer.
"""
from pathlib import Path
import argparse,json
import numpy as np
from probe_raw_scene_replay import make_environment,scene_for,digest
from probe_rgbd_correspondence import matrix
from clearvla.vision.sensor_geometry import rigid


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scene',required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    files={f:s for r in json.loads(a.plan.read_text())['records'] for f,s in zip(r['raw_files'],r['sha256']) if scene_for(f)==a.scene}
    env=make_environment(Path(next(iter(files))).parent,a.scene)
    import pybullet as pb
    cam=env.cameras[1];rows=[];transforms=[]
    for filename,expected in files.items():
        assert digest(filename)==expected
        with np.load(filename,allow_pickle=False) as z:
            data={k:z[k] for k in ('robot_obs','scene_obs','rgb_gripper','depth_gripper')}
        env.scene.reset(data['scene_obs']);env.robot.reset(data['robot_obs']);rgb,depth=cam.render()
        tcp_pos,tcp_quat=pb.getLinkState(env.robot.robot_uid,env.robot.tcp_link_id,physicsClientId=env.cid)[:2]
        cam_pos,cam_quat=pb.getLinkState(env.robot.robot_uid,cam.gripper_cam_link,physicsClientId=env.cid)[:2]
        def pose(x,q):
            m=np.eye(4);m[:3,:3]=np.asarray(pb.getMatrixFromQuaternion(q)).reshape(3,3);m[:3,3]=x;return m
        transform=np.linalg.inv(pose(tcp_pos,tcp_quat))@pose(cam_pos,cam_quat);transforms.append(transform)
        # One fixed rigid mount; no fitting to RGB, depth, body IDs or poses.
        corrected=rigid(data['robot_obs'][:3],data['robot_obs'][3:6])@transforms[0]
        view=pb.computeViewMatrix(corrected[:3,3],corrected[:3,3]+corrected[:3,1],-corrected[:3,2])
        image=pb.getCameraImage(width=cam.width,height=cam.height,viewMatrix=view,
            projectionMatrix=cam.projection_matrix,physicsClientId=env.cid,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        new_rgb,new_depth=cam.process_rgbd(image,cam.nearval,cam.farval)
        label=np.asarray(image[4]).reshape(cam.height,cam.width)
        body=np.where(label<0,-1,label&((1<<24)-1));link=np.where(label<0,-2,(label>>24)-1)
        cameras=[]
        for name,r,d in [('joint_fk',rgb,depth),('observed_tcp_rigid',new_rgb,new_depth)]:
            er=np.abs(r.astype(float)-data['rgb_gripper']).mean(-1);ed=np.abs(d-data['depth_gripper'])
            entities=[]
            for uid in np.unique(body):
                if uid<0:continue
                m=body==uid;entities.append(dict(body=int(uid),name=pb.getBodyInfo(int(uid),physicsClientId=env.cid)[1].decode(),
                    pixels=int(m.sum()),rgb_mae=float(er[m].mean()),rgb_unequal=int((m&(er>0)).sum()),depth_mae=float(ed[m].mean())))
            cameras.append(dict(mode=name,rgb_mae=float(er.mean()),rgb_unequal=int((er>0).sum()),
                depth_mae=float(ed.mean()),depth_max=float(ed.max()),entities=entities))
        rows.append(dict(path=filename,controls=cameras,mount_max_abs=float(np.max(np.abs(transform-transforms[0])))))
        report=dict(complete=False,records=rows,mount_tcp_to_camera_link=transforms[0].tolist(),scope=__doc__)
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    # A fresh dynamic step tests the same cached-link/Jacobian-clock distinction
    # without assuming any metadata about how the raw demonstration was saved.
    action=np.array([.1,0,0,0,0,0,1.])
    obs,_,_,_=env.step(action)
    before=matrix(cam,'view').copy();stored=obs['robot_obs'].copy()
    tcp_default=pb.getLinkState(env.robot.robot_uid,env.robot.tcp_link_id,physicsClientId=env.cid)
    tcp_recomputed=pb.getLinkState(env.robot.robot_uid,env.robot.tcp_link_id,physicsClientId=env.cid,computeForwardKinematics=True)
    after_fk=cam.render();after=matrix(cam,'view').copy()
    report.update(complete=True,mount_max_abs=max(r['mount_max_abs'] for r in rows),
        dynamic_step=dict(tcp_default_to_fk_m=float(np.linalg.norm(np.array(tcp_default[0])-tcp_recomputed[0])),
            camera_before_to_fk_max=float(np.max(np.abs(before-after))),
            observed_tcp_default_error=float(np.linalg.norm(stored[:3]-tcp_default[0]))))
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
