"""Frozen RGB correspondence candidate, admitted separately from learned G.

For training supervision/audit only. Forward/backward flow and photometric
agreement reject unsupported matches; none of the thresholds use object labels.
"""
import numpy as np


def observed_rgb_flow(source, target):
    import cv2
    if source.shape!=target.shape or source.ndim!=3 or source.shape[-1]!=3:
        raise ValueError('temporal RGB correspondence requires the same camera chart')
    if source.dtype!=np.uint8 or target.dtype!=np.uint8:
        raise ValueError('frozen RGB flow expects literal uint8 observations')
    a=cv2.cvtColor(source,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    parameters=dict(pyr_scale=.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=0)
    forward=cv2.calcOpticalFlowFarneback(a,b,None,**parameters)
    backward=cv2.calcOpticalFlowFarneback(b,a,None,**parameters)
    h,w=a.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    xy=np.stack((xx,yy),-1)+forward
    back=cv2.remap(backward,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    rgb=cv2.remap(target,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    cycle=np.linalg.norm(forward+back,axis=-1)
    photo=np.abs(source.astype(np.float32)-rgb.astype(np.float32)).mean(-1)/255
    valid=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    accepted=valid&(cycle<.75)&(photo<.08)
    return dict(xy=xy,accepted=accepted,cycle=cycle,photometric=photo)
