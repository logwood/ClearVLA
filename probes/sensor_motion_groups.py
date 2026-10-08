"""Conservative different-motion witness; no physical IDs or slot predictions.

A difference in masks/surfaces alone can split one rigid body. Here negatives
also need two independently fitted, incompatible rigid motions. Fits use depth
and accepted RGB correspondences in each camera's own metric chart; camera
ego-motion is shared and does not require joint/TCP synchronization. Static,
co-moving, under-supported and non-rigid estimates remain unknown. This is a
proposal to audit, not certified physical identity or a production switch.
"""
import numpy as np
from sensor_surface_groups import world_points
from clearvla.vision.observed_flow import observed_rgb_flow


SETTINGS=dict(minimum_points=12,minimum_second_spread_m=.003,
              maximum_self_p90_m=.002,minimum_cross_median_m=.006,
              maximum_depth_stencil_span_m=.01)


def rigid_fit(x,y):
    cx=x.mean(0);cy=y.mean(0)
    u,_,vt=np.linalg.svd((x-cx).T@(y-cy))
    adjust=np.eye(3);adjust[-1,-1]=np.sign(np.linalg.det(u@vt))
    rotation=u@adjust@vt
    return rotation,cy-cx@rotation


def propose_motion_negatives(current,previous,projection,partition,groups):
    """One native camera, current->observed-past; never future or oracle inputs."""
    flow=observed_rgb_flow(current['rgb'],previous['rgb'])
    shape=current['depth'].shape;h,w=shape
    x,current_valid=world_points(current['depth'],np.eye(4),projection)
    xy=np.asarray(flow['xy'],np.float64)
    in_bounds=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    u=np.clip(xy[...,0],0,w-1);v=np.clip(xy[...,1],0,h-1)
    x0=np.floor(u).astype(int);y0=np.floor(v).astype(int)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    values=np.stack([previous['depth'][yy,xx] for yy,xx in ((y0,x0),(y0,x1),(y1,x0),(y1,x1))])
    legal=np.isfinite(values).all(0)&(values>0).all(0)&(np.ptp(values,axis=0)<=SETTINGS['maximum_depth_stencil_span_m'])
    dx=u-x0;dy=v-y0
    weights=np.stack(((1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy))
    d=(np.where(legal[None],values,1.)*weights).sum(0)
    projection=np.asarray(projection,float)
    ndc=np.stack(((2*(u+.5)/w-1).ravel(),(1-2*(v+.5)/h).ravel(),
        ((-projection[2,2]*d+projection[2,3])/d).ravel(),np.ones(h*w)))
    homogeneous=np.linalg.inv(projection)@ndc;y=(homogeneous[:3]/homogeneous[3:]).T
    valid=(flow['accepted']&in_bounds&legal).ravel()&current_valid&np.isfinite(x).all(1)&np.isfinite(y).all(1)
    fits=[];clouds={}
    for group in range(len(groups)):
        ids=np.flatnonzero((partition.ravel()==group)&valid)
        row=dict(group=group,points=len(ids),admitted=False)
        if len(ids)>=SETTINGS['minimum_points']:
            source=x[ids];target=y[ids]
            # Interleaved, deterministic holdout: validation pixels never fit
            # the transform whose reliability they test.
            train=np.arange(len(ids))%2==0;test=~train
            spread=np.linalg.svd(source[train]-source[train].mean(0),compute_uv=False)/np.sqrt(train.sum())
            rotation,offset=rigid_fit(source[train],target[train])
            residual=np.linalg.norm(source[test]@rotation+offset-target[test],axis=1)
            p90=float(np.quantile(residual,.9))
            row.update(second_spread_m=float(spread[1]),heldout_p90_m=p90,
                admitted=bool(spread[1]>=SETTINGS['minimum_second_spread_m'] and p90<=SETTINGS['maximum_self_p90_m']),
                rotation=rotation.tolist(),translation=offset.tolist())
            if row['admitted']:clouds[group]=(source[test],target[test],rotation,offset)
        fits.append(row)
    pairs=[]
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            if not np.all(groups[i]!=groups[j]):continue
            row=dict(groups=[i,j],accepted=False,reason='unknown_motion')
            if i in clouds and j in clouds:
                a,b,ra,ta=clouds[i];c,d,rb,tb=clouds[j]
                error_a=float(np.median(np.linalg.norm(a@rb+tb-b,axis=1)))
                error_b=float(np.median(np.linalg.norm(c@ra+ta-d,axis=1)))
                accepted=min(error_a,error_b)>SETTINGS['minimum_cross_median_m']
                row.update(cross_median_m=[error_a,error_b],accepted=accepted,
                           reason='different_rigid_motion' if accepted else 'co_motion_or_uncertain')
            pairs.append(row)
    return dict(settings=SETTINGS,fits=fits,pairs=pairs,
        accepted_pairs=[r['groups'] for r in pairs if r['accepted']],
        scope=__doc__)
