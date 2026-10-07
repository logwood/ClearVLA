"""Read uncached sensor evidence and return small training-only pair labels."""
from pathlib import Path
import json
import numpy as np
import torch
from clearvla.vision.sensor_geometry import camera_views, depth_correspondence, photometric_support
from clearvla.vision.observed_flow import observed_rgb_flow


class IdentityLabelProducer:
    def __init__(self,raw_root):
        self.raw_root=Path(raw_root)
        file=Path(__file__).resolve().parents[1]/'assets/calvin_rgbd_joint_geometry_v1.json'
        self.calibration=json.loads(file.read_text())
        if self.calibration['view_max_abs_disagreement']>5e-6 or self.calibration['verified_frames']<12:
            raise ValueError('sensor geometry has not passed its independent FK admission')

    @staticmethod
    def _sample(shape):
        h,w=shape
        y=np.rint(np.linspace(0,h-1,32)).astype(int);x=np.rint(np.linspace(0,w-1,32)).astype(int)
        yy,xx=np.meshgrid(y,x,indexing='ij')
        return yy.ravel(),xx.ravel()

    @staticmethod
    def _endpoint(xy,shape):
        h,w=shape
        return (np.asarray(xy)*np.array([2/(w-1),2/(h-1)])-1).astype(np.float32)

    def __call__(self,episode,history,image_store):
        # Metadata identifies the raw source, not a task/physical-object label.
        handle=image_store._h5(episode.path)
        split=str(handle.attrs['source_split'])
        context=int(handle.attrs['context_start'])
        if split not in ('training','validation') or context!=episode.context_start:
            raise ValueError('raw correspondence and admitted episode provenance differ')
        previous=context+int(history[-2,1]);current=context+int(history[-1,1])
        frames=[]
        for index in (previous,current):
            path=self.raw_root/split/('episode_%07d.npz'%index)
            with np.load(path,allow_pickle=False) as z:
                # Explicit allowlist: no scene_obs, object poses, or task labels.
                frames.append({key:z[key] for key in ('robot_obs','rgb_static','rgb_gripper','depth_static','depth_gripper')})
        now=frames[1];depths=[now['depth_static'],now['depth_gripper']]
        if [list(d.shape) for d in depths]!=self.calibration['image_shapes']:
            raise ValueError('raw depth resolution differs from admitted sensor calibration')
        views=camera_views(now['robot_obs'],self.calibration)
        out={key:[] for key in ('cross_source','cross_target','cross_valid','temporal_source','temporal_target','temporal_valid')}
        for camera,key in enumerate(('static','gripper')):
            shape=depths[camera].shape;y,x=self._sample(shape);source=np.stack((x,y),-1)
            cross=depth_correspondence(depths,views,self.calibration['projection'],camera,1-camera)
            index=y*shape[1]+x
            rgbs=[now['rgb_static'],now['rgb_gripper']]
            valid=photometric_support(cross,rgbs[camera],rgbs[1-camera])[index]
            target=self._endpoint(cross['xy'][index],depths[1-camera].shape)
            out['cross_source'].append(self._endpoint(source,shape));out['cross_target'].append(np.where(valid[:,None],target,0.));out['cross_valid'].append(valid)
            temporal=observed_rgb_flow(frames[0]['rgb_'+key],now['rgb_'+key])
            valid=temporal['accepted'][y,x]&(current>previous)
            target=self._endpoint(temporal['xy'][y,x],shape)
            out['temporal_source'].append(self._endpoint(source,shape));out['temporal_target'].append(np.where(valid[:,None],target,0.));out['temporal_valid'].append(valid)
        result={'identity_'+key:torch.from_numpy(np.stack(value)) for key,value in out.items()}
        result['identity_source_frames']=torch.tensor([previous,current],dtype=torch.long)
        return result
