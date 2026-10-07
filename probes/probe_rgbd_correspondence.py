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


from clearvla.vision.sensor_geometry import rigid, camera_views, depth_correspondence as correspondence


def extract_joint_calibration(env):
    import xml.etree.ElementTree as ET
    import pybullet as bullet
    import calvin_env
    urdf=Path(env.robot.filename)
    if not urdf.is_absolute():urdf=Path(calvin_env.__file__).resolve().parents[1]/'data'/urdf
    tree=ET.parse(urdf).getroot()
    links={l.attrib['name']:l for l in tree.findall('link')}
    joints={j.find('child').attrib['link']:j for j in tree.findall('joint')}
    infos=[bullet.getJointInfo(env.robot.robot_uid,i,physicsClientId=env.robot.cid) for i in range(bullet.getNumJoints(env.robot.robot_uid,physicsClientId=env.robot.cid))]
    camera_index=env.cameras[1].gripper_cam_link
    link=infos[camera_index][12].decode()
    camera_link=links[link]
    def origin(element):
        if element is None:return np.eye(4)
        return rigid([float(x) for x in element.attrib.get('xyz','0 0 0').split()],[float(x) for x in element.attrib.get('rpy','0 0 0').split()])
    result=[];by_name={info[1].decode():info[0] for info in infos}
    while link in joints:
        joint=joints[link];name=joint.attrib['name'];index=by_name[name];kind=joint.attrib['type']
        source=None;scale=1.
        if kind!='fixed':
            if index in env.robot.arm_joint_ids:source=7+list(env.robot.arm_joint_ids).index(index)
            elif index in env.robot.gripper_joint_ids:source=6;scale=.5
            else:raise RuntimeError('unobserved camera-chain joint '+name)
        axis=joint.find('axis')
        result.append(dict(name=name,type=kind,origin=origin(joint.find('origin')).tolist(),axis=[float(x) for x in (axis.attrib.get('xyz','0 0 1') if axis is not None else '0 0 1').split()],observation_index=source,observation_scale=scale))
        link=joint.find('parent').attrib['link']
    inertial=camera_link.find('inertial');offset=origin(None if inertial is None else inertial.find('origin'))
    axes=np.eye(4);axes[:3,:3]=[[-1,0,0],[0,0,-1],[0,-1,0]]
    base=rigid(env.robot.base_position,bullet.getEulerFromQuaternion(env.robot.base_orientation))
    return dict(schema='calvin_rgbd_joint_geometry_v1',base_pose=base.tolist(),joints=list(reversed(result)),camera_from_link=(offset@axes).tolist(),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    root=Path('/data/senwang/data/calvin/raw/task_ABC_D')
    ranges=np.load(root/'training/scene_info.npy',allow_pickle=True).item()
    _official_task_assets()
    rows=[];extrinsics=[];fk_errors=[];env=_environment(root,show_gui=False)
    joint_calibration=extract_joint_calibration(env)
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
            import pybullet as bullet
            tcp=np.eye(4);tcp[:3,3]=robot[:3]
            tcp[:3,:3]=np.asarray(bullet.getMatrixFromQuaternion(bullet.getQuaternionFromEuler(robot[3:6]))).reshape(3,3)
            extrinsic=np.linalg.inv(tcp)@np.linalg.inv(views[1]);extrinsics.append(extrinsic)
            if len(extrinsics)==1:
                calibration=dict(schema='calvin_rgbd_sensor_geometry_v1',camera_names=['top','wrist'],pixel_center='half-pixel OpenGL; q endpoint coordinates use index/(size-1)',
                    depth_units='metres',source_frame=dict(split=split,scene=scene,index=index),
                    top_world_to_camera=views[0].tolist(),tcp_to_wrist_camera=extrinsic.tolist(),projection=[p.tolist() for p in projections],image_shapes=[list(d.shape) for d in depth])
            joint_calibration.update(top_world_to_camera=views[0].tolist(),projection=[v.tolist() for v in projections],image_shapes=[list(d.shape) for d in depth],camera_names=["top","wrist"],depth_units="metres")
            inferred=camera_views(robot,joint_calibration)
            fk_errors.append(float(np.max(np.abs(inferred[1]-views[1]))))
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
        calibration['verified_frames']=len(extrinsics)
        calibration['extrinsic_max_abs_disagreement']=float(np.max(np.abs(np.stack(extrinsics)-extrinsics[0])))
        (a.output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
        joint_calibration['verified_frames']=len(fk_errors)
        joint_calibration['view_max_abs_disagreement']=max(fk_errors)
        (a.output/'joint-calibration.json').write_text(json.dumps(joint_calibration,indent=2)+'\n')
        (a.output/'complete.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frames':len(rows)}))
    finally:env.close()


if __name__=='__main__':main()
