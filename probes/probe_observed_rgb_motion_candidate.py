"""Score RGB-only motion candidates against separate rigid geometry.

Evaluation only: no policy, training labels, parameters or recorded commands
are changed. Estimators receive only two uint8 RGB frames. Body/pose-derived
labels are read exclusively by the scorer after estimation. Rejected matches
keep unknown support; report both accepted accuracy and all-visible error with
zero displacement at rejected points so abstention cannot hide missed motion.
DIS is the OpenCV medium preset, without tuning to these audit observations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np

from clearvla.vision.observed_flow import observed_rgb_flow


def stats(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1)
    if not len(values):return None
    if not np.isfinite(values).all():raise ValueError('nonfinite audit value')
    return dict(n=int(values.size),mean=float(values.mean()),maximum=float(values.max()),
                median=float(np.median(values)),p90=float(np.quantile(values,.9)))


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def estimate(source,target,method):
    # Keep oracle labels outside this function's interface.
    if method=='farneback':return observed_rgb_flow(source,target)
    if method!='dis_medium':raise ValueError('unknown estimator')
    if source.shape!=target.shape or source.dtype!=np.uint8 or target.dtype!=np.uint8:
        raise ValueError('RGB shape/dtype contract')
    a=cv2.cvtColor(source,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)
    # Separate estimators prevent cached internal state or supplied flow from
    # providing undeclared tracking history.
    f=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(a,b,None)
    back=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(b,a,None)
    h,w=a.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32)
    xy=np.stack((xx,yy),-1)+f
    returned=cv2.remap(back,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    warped=cv2.remap(target,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    cycle=np.linalg.norm(f+returned,axis=-1)
    photo=np.abs(source.astype(np.float32)-warped.astype(np.float32)).mean(-1)/255
    inside=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    # The existing RGB label generator's sensor-only acceptance rule is held
    # fixed across estimators. It is not a calibrated match probability.
    return dict(xy=xy,accepted=inside&(cycle<.75)&(photo<.08),cycle=cycle,photometric=photo)


def resample(field,xy):
    xy=np.asarray(xy,np.float32)
    return cv2.remap(np.asarray(field,np.float32),xy[:,0,None],xy[:,1,None],
        cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0]


def coarse_transport(field,xy,size):
    h,w=field.shape[:2]
    yy,xx=np.meshgrid(np.linspace(0,h-1,size),np.linspace(0,w-1,size),indexing='ij')
    grid=resample(field,np.stack((xx,yy),-1).reshape(-1,2)).reshape(size,size,2)
    query=xy/np.array([w-1,h-1],np.float32)*(size-1)
    return resample(grid,query)


def score(flow,labels,camera,shape):
    h,w=shape;scale=np.array([(w-1)/2,(h-1)/2],np.float64)
    source=(labels['temporal_source'][camera]+1)*scale
    ix=np.rint(source).astype(int);x,y=ix[:,0],ix[:,1]
    if not np.allclose(source,ix,atol=2e-5):raise ValueError('label grid is not an observed pixel grid')
    target=(labels['audit_rigid_temporal_target'][camera]+1)*scale
    visible=labels['audit_rigid_temporal_valid'][camera].astype(bool)
    body=labels['temporal_source_body'][camera]
    if np.any(visible&(body<0)):raise ValueError('rigid truth must be an admitted body interior')
    pred=flow['xy'][y,x];accepted=flow['accepted'][y,x]
    delta=pred-source;truth=target-source
    yy,xx=np.mgrid[:h,:w];grid=np.stack((xx,yy),-1)
    admitted_field=(flow['xy']-grid)*flow['accepted'][...,None]
    used=delta*accepted[:,None]
    coarse8=coarse_transport(admitted_field,source,8)
    coarse16=coarse_transport(admitted_field,source,16)
    motion=np.linalg.norm(truth,axis=-1);error=np.linalg.norm(pred-target,axis=-1)
    regions={}
    masks=dict(all_visible=visible,moving_gt1px=visible&(motion>1),near_static_le025px=visible&(motion<=.25))
    for name,mask in masks.items():
        keep=mask&accepted
        regions[name]=dict(count=int(mask.sum()),accepted=int(keep.sum()),
            coverage=float(keep.sum()/mask.sum()) if mask.any() else None,
            zero_motion_error_pixels=stats(motion[mask]),
            raw_endpoint_error_pixels=stats(error[mask]),accepted_endpoint_error_pixels=stats(error[keep]),
            unknown_zero_error_pixels=stats(np.linalg.norm(used-truth,axis=-1)[mask]),
            measured_motion_pixels=stats(np.linalg.norm(used,axis=-1)[mask]),
            coarse8_error_pixels=stats(np.linalg.norm(coarse8-truth,axis=-1)[mask]),
            coarse16_error_pixels=stats(np.linalg.norm(coarse16-truth,axis=-1)[mask]),
            accepted_error_gt2px=int((keep&(error>2)).sum()),
            accepted_error_gt5px=int((keep&(error>5)).sum()))
    unknown={}
    for key in ('source_interior','occluded','out_of_view'):
        label='audit_rigid_'+key
        if label in labels:
            mask=labels[label][camera].astype(bool)
            unknown[key]=dict(count=int(mask.sum()),accepted=int((mask&accepted).sum()))
    target_map=labels.get('audit_rigid_target_body_map_'+str(camera))
    if target_map is not None:
        uv=np.rint(np.clip(pred,[0,0],[w-1,h-1])).astype(int)
        found=target_map[uv[:,1],uv[:,0]]
        interior=labels['audit_rigid_source_interior'][camera].astype(bool)
        mask=interior&accepted
        unknown['accepted_interior_body']=dict(count=int(mask.sum()),same=int((mask&(found==body)).sum()),
            wrong_object=int((mask&(found>=0)&(found!=body)).sum()),background=int((mask&(found<0)).sum()))
    return dict(camera=camera,regions=regions,unknown_support=unknown)


def main():
    q=argparse.ArgumentParser()
    for name in ('plan','labels','output'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--methods',nargs='+',choices=('farneback','dis_medium'),default=['farneback','dis_medium'])
    a=q.parse_args();a.output.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    complete=json.loads((a.labels/'complete.json').read_text())
    if complete.get('rigid_simulation_oracle_audit_only') is not True:raise ValueError('independent rigid audit labels required')
    label_rows=json.loads((a.labels/'results.json').read_text())
    index={(r['case'],r['step']):r for r in label_rows}
    if len(index)!=len(label_rows):raise ValueError('duplicate audit window')
    report=dict(identity=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        label_complete=complete,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        opencv_version=cv2.__version__,methods=a.methods,scope=__doc__,policy_changed=False,
        sources=['https://arxiv.org/abs/1603.03590','https://docs.opencv.org/4.10.0/de/d4f/classcv_1_1DISOpticalFlow.html']),
        records=[],repeat_controls=[],complete=False)
    checked=set()
    for row in json.loads(a.plan.read_text()):
        case=Path(row['case'])
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:
            rgb={key:z[key] for key in ('rgb_static','rgb_gripper')}
        for step in row['steps']:
            entry=index[(case.name,step)];path=Path(entry['production_labels'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['production_labels_sha256']:raise ValueError('label hash differs')
            with np.load(path,allow_pickle=False) as z:labels={k:z[k] for k in z.files}
            before,after=map(int,labels['source_frames'])
            if max(before,after)>step or min(before,after)<0:raise ValueError('future or negative frame')
            record=dict(case=str(case),case_id=row.get('case_id'),step=step,source_frames=[before,after],
                label_sha256=entry['production_labels_sha256'],methods={})
            for method in a.methods:
                cameras=[]
                for camera,key in enumerate(('rgb_static','rgb_gripper')):
                    source,target=rgb[key][before],rgb[key][after]
                    start=time.perf_counter();flow=estimate(source,target,method);elapsed=time.perf_counter()-start
                    # Verify the existing method reproduces its sensor labels;
                    # those labels remain distinct from the rigid scoring truth.
                    if method=='farneback':
                        h,w=source.shape[:2];scale=np.array([(w-1)/2,(h-1)/2])
                        pos=np.rint((labels['temporal_source'][camera]+1)*scale).astype(int)
                        x,y=pos[:,0],pos[:,1];valid=flow['accepted'][y,x]&(before!=after)
                        predicted=np.where(valid[:,None],flow['xy'][y,x]/scale-1,0.)
                        if not np.array_equal(valid,labels['temporal_valid'][camera]):raise ValueError('sensor acceptance reproduction failed')
                        if not np.allclose(predicted,labels['temporal_target'][camera],rtol=0,atol=2e-7):raise ValueError('sensor endpoint reproduction failed')
                    scored=score(flow,labels,camera,source.shape[:2]);scored['seconds']=elapsed;cameras.append(scored)
                    if (method,camera) not in checked:
                        repeat=estimate(source,target,method)
                        err=float(np.max(np.abs(repeat['xy']-flow['xy'])))
                        if err!=0 or not np.array_equal(repeat['accepted'],flow['accepted']):raise ValueError('nonrepeatable RGB estimator')
                        same=estimate(source,source,method);h,w=source.shape[:2];yy,xx=np.mgrid[:h,:w]
                        report['repeat_controls'].append(dict(method=method,camera=camera,repeat_max=err,
                            identical_image_motion_pixels=stats(np.linalg.norm(same['xy']-np.stack((xx,yy),-1),axis=-1))))
                        checked.add((method,camera))
                record['methods'][method]=cameras
            report['records'].append(record);dump(a.output/'results.json',report)
            print(json.dumps(dict(case=case.name,step=step,records=len(report['records']))),flush=True)
    report['complete']=True;dump(a.output/'results.json',report)


if __name__=='__main__':main()
