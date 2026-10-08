"""Build compact auxiliary supervision for a declared, unchanged BS8 exposure.

All original RGB-D/temporal pairs remain in the live dataset. This extra label
pass runs the full frozen mask-prompt budget on the current two cameras, reads
only observed sensor fields, and persists integer/Boolean annotations only.
The time and hardware cost of this pass is part of the experiment cost.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from transformers import Sam2Model,Sam2Processor,pipeline
from probe_frozen_mask_proposals import SETTINGS,propose,sha,dump
from probe_mask_surface_agreement import proposal_partition
from sensor_surface_groups import propose_groups
from sensor_motion_groups import propose_motion_negatives
from clearvla.vision.sensor_geometry import camera_views
from clearvla.mainline.data.identity_correspondence import IdentityLabelProducer
from clearvla.mainline.data.identity_region_annotations import SCHEMA


def conservative_ids(field,xy):
    """Only a single group across every positive-weight interpolation corner."""
    h,w=field.shape;u=(xy[...,0]+1)*(w-1)/2;v=(xy[...,1]+1)*(h-1)/2
    x0=np.floor(u).astype(int);y0=np.floor(v).astype(int)
    inside=(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1)
    x0=np.clip(x0,0,w-1);y0=np.clip(y0,0,h-1);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    dx=u-x0;dy=v-y0;result=field[y0,x0].copy()
    for yy,xx,weight in ((y0,x0,(1-dx)*(1-dy)),(y0,x1,dx*(1-dy)),(y1,x0,(1-dx)*dy),(y1,x1,dx*dy)):
        inside &= (weight<=1e-7)|(field[yy,xx]==result)
    return np.where(inside,result,-1).astype(np.int64)


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','weights','weight-receipt','output','raw-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args()
    if not 0<=a.shard<a.shards:raise ValueError('invalid annotation shard')
    a.output.mkdir(exist_ok=False);plan=json.loads(a.plan.read_text());receipt=json.loads(a.weight_receipt.read_text())
    assert a.weights.resolve()==Path(receipt['path']).resolve()
    for f in receipt['files']:assert sha(a.weights/f['name'])==f['sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model,loading=Sam2Model.from_pretrained(a.weights,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(loading))
    model=model.eval().cuda();model.requires_grad_(False)
    processor=Sam2Processor.from_pretrained(a.weights,local_files_only=True)
    generator=pipeline('mask-generation',model=model,image_processor=processor.image_processor,device=0)
    sensor=IdentityLabelProducer(a.raw_root);cal=sensor.calibration
    report=dict(schema=SCHEMA,supervision_only=True,complete=False,records=[],plan_sha256=sha(a.plan),
        shard=a.shard,shards=a.shards,producer=dict(script_sha256=sha(__file__),weights=receipt,
        settings=SETTINGS,calibration_sha256=sha(Path(__file__).parents[1]/'clearvla/mainline/assets/calvin_rgbd_joint_geometry_v1.json'),
        current_cameras_only=True,original_positive_and_source_budgets='unchanged',scope=__doc__))
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for index,row in enumerate(plan['records']):
        if index%a.shards!=a.shard:continue
        before,now=row['source_frames'];frames=[]
        for path,expected in zip(row['raw_files'],row['sha256']):
            if sha(path)!=expected:raise ValueError('raw sensor bytes changed')
            with np.load(path,allow_pickle=False) as z:
                frames.append({k:z[k] for k in ('robot_obs','rgb_static','rgb_gripper','depth_static','depth_gripper')})
        current=frames[1];rgbs=[current['rgb_static'],current['rgb_gripper']]
        depths=[current['depth_static'],current['depth_gripper']]
        surfaces,metadata=propose_groups(depths,rgbs,camera_views(current['robot_obs'],cal),cal['projection'],
            support_mode='under_tcp_v2',workspace_point=current['robot_obs'][:3],connectivity_mode='geometry_only_v3')
        groups_out=[];adjacency=[];stats=[]
        for c,key in enumerate(('static','gripper')):
            with torch.inference_mode():masks,_=propose(generator,rgbs[c])
            part,groups=proposal_partition(masks,surfaces[c],interior_radius=1)
            witness=propose_motion_negatives(dict(rgb=rgbs[c],depth=depths[c]),
                dict(rgb=frames[0]['rgb_'+key],depth=frames[0]['depth_'+key]),cal['projection'][c],part,groups)
            yy,xx=sensor._sample(part.shape);ids=part[yy,xx];alive=np.unique(ids[ids>=0])
            mapping={int(old):i for i,old in enumerate(alive)}
            selected=np.array([mapping.get(int(g),-1) for g in ids],np.int64)
            adj=np.zeros((len(ids),len(ids)),bool)
            for left,right in witness['accepted_pairs']:
                if left in mapping and right in mapping and now>before:
                    i,j=mapping[left],mapping[right];adj[i,j]=adj[j,i]=True
            groups_out.append(selected);adjacency.append(adj)
            stats.append(dict(groups=len(alive),native_negative_pairs=len(witness['accepted_pairs']),sampled_negative_pairs=int(adj.sum()//2)))
        ordinary=sensor.from_sensor_frames(frames,before,now)
        strata=[]
        for kind in ('cross','temporal'):
            views=[]
            for camera in range(2):
                destination=1-camera if kind=='cross' else camera
                xy=ordinary['identity_'+kind+'_target'][camera].numpy()
                ids=conservative_ids(surfaces[destination],xy)
                ids=np.where(ordinary['identity_'+kind+'_valid'][camera].numpy(),ids,-1)
                views.append(ids)
            strata.append(np.stack(views))
        path=a.output/('labels_%07d.npz'%index)
        np.savez_compressed(path,region_group=np.stack(groups_out),region_different=np.stack(adjacency),
            prediction_region=np.stack(strata),source_frames=np.asarray([before,now],np.int64))
        report['records'].append(dict(source_split=row.get('source_split',row['raw_source_split']),source_frames=[before,now],
            raw_sha256=row['sha256'],labels_path=str(path),labels_sha256=sha(path),cameras=stats))
        report.update(elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(a.output/'manifest.json',report)
        print('ANNOTATED',index,before,now,stats,flush=True)
    report['complete']=True;dump(a.output/'manifest.json',report)


if __name__=='__main__':main()
