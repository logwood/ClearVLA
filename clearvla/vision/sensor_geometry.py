"""CPU sensor geometry; no object poses, scene labels, or neural prediction."""
import numpy as np


def photometric_support(law, source_rgb, target_rgb, *, maximum_error=.08):
    """Reject occlusion/color edges using raw sensor values, never learned IDs.

    The depth projection remains the correspondence. Photometry only removes
    uncertain pairs and may lose coverage under view-dependent illumination.
    """
    source=np.asarray(source_rgb,dtype=np.float32).reshape(-1,3)/255.
    target=np.asarray(target_rgb,dtype=np.float32)[law['v'],law['u']]/255.
    error=np.abs(source-target).mean(-1)
    return law['accepted'] & np.isfinite(error) & (error<maximum_error)


def rigid(xyz=(0,0,0),rpy=(0,0,0)):
    rx,ry,rz=rpy;cx,cy,cz=np.cos(rpy);sx,sy,sz=np.sin(rpy)
    rotation=np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                       [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])
    out=np.eye(4);out[:3,:3]=rotation;out[:3,3]=xyz
    return out


def camera_views(robot_observation,calibration):
    """Exact URDF joint chain from the observed seven joint angles."""
    if calibration['schema']!='calvin_rgbd_joint_geometry_v1':
        raise ValueError('unadmitted sensor geometry schema')
    pose=np.asarray(calibration['base_pose'],dtype=float)
    for joint in calibration['joints']:
        pose=pose@np.asarray(joint['origin'],dtype=float)
        index=joint['observation_index']
        if index is not None:
            angle=float(robot_observation[index])*joint.get('observation_scale',1.)
            axis=np.asarray(joint['axis'],dtype=float);axis=axis/np.linalg.norm(axis)
            transform=np.eye(4)
            if joint['type'] in ('revolute','continuous'):
                x,y,z=axis;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                transform[:3,:3]=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
            elif joint['type']=='prismatic':transform[:3,3]=axis*angle
            else:raise ValueError('observed value supplied to fixed joint')
            pose=pose@transform
    pose=pose@np.asarray(calibration['camera_from_link'],dtype=float)
    return [np.asarray(calibration['top_world_to_camera'],dtype=float),np.linalg.inv(pose)]


def depth_correspondence(depths,views,projections,source,target,*,tolerance=.003):
    depth=np.asarray(depths[source]);h,w=depth.shape;yy,xx=np.mgrid[:h,:w]
    x=(2*(xx+.5)/w-1).ravel();y=(1-2*(yy+.5)/h).ravel();d=depth.ravel()
    P=np.asarray(projections[source]);valid_depth=np.isfinite(d)&(d>0);safe=np.where(valid_depth,d,1.)
    z=(-P[2,2]*safe+P[2,3])/safe
    world=np.linalg.inv(P@views[source])@np.stack((x,y,z,np.ones_like(z)));world/=world[3:]
    camera=views[target]@world;clip=np.asarray(projections[target])@camera
    denominator=np.where(np.abs(clip[3:])>1e-12,clip[3:],1.);projected=clip/denominator
    th,tw=depths[target].shape;u=(projected[0]+1)*tw/2-.5;v=(1-projected[1])*th/2-.5
    finite=np.isfinite(u)&np.isfinite(v)
    ui=np.rint(np.clip(np.where(finite,u,0.),0,tw-1)).astype(int);vi=np.rint(np.clip(np.where(finite,v,0.),0,th-1)).astype(int)
    visible=valid_depth&finite&(u>=0)&(u<=tw-1)&(v>=0)&(v<=th-1)&(-camera[2]>0)
    target_depth=np.asarray(depths[target])[vi,ui]
    error=np.abs(target_depth+camera[2]);accepted=visible&np.isfinite(target_depth)&(target_depth>0)&(error<tolerance)
    return dict(u=ui,v=vi,visible=visible,accepted=accepted,depth_error=error,xy=np.stack((u,v),-1),source_shape=(h,w))
