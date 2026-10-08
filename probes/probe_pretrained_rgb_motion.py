"""Frozen official RAFT-small RGB audit; never changes a policy or its labels.

Fixed native-pixel chart, replicate padding to >=128 and multiples of 8,
12 recurrent updates, published C_T_V2 weights, FP32. Forward/backward and
photometric gates remain .75 pixels / .08, as in the existing RGB audit.
Simulator geometry only scores visible/occluded points after estimation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.optical_flow import raft_small
from probe_observed_rgb_motion_candidate import score, stats, dump


class Estimator:
    def __init__(self, weights):
        self.model=raft_small(weights=None).eval().cuda()
        state=torch.load(weights,map_location='cpu',weights_only=True)
        self.model.load_state_dict(state,strict=True)
        self.model.requires_grad_(False)
        self.versions={n:p._version for n,p in self.model.named_parameters()}

    @torch.inference_mode()
    def __call__(self, source, target):
        if source.dtype!=np.uint8 or target.dtype!=np.uint8 or source.shape!=target.shape:
            raise ValueError('RGB contract')
        h,w=source.shape[:2]
        ph=max(128,((h+7)//8)*8)-h;pw=max(128,((w+7)//8)*8)-w
        left=pw//2;top=ph//2
        a=torch.from_numpy(np.stack((source,target))).cuda().permute(0,3,1,2).float()/127.5-1
        a=F.pad(a,(left,pw-left,top,ph-top),mode='replicate')
        b=a.flip(0)
        f=self.model(a,b,num_flow_updates=12)[-1]
        f=f[:,:,top:top+h,left:left+w].permute(0,2,3,1).cpu().numpy()
        forward,back=f
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        xy=np.stack((xx,yy),-1)+forward
        returned=cv2.remap(back,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        warped=cv2.remap(target,xy[...,0],xy[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cycle=np.linalg.norm(forward+returned,axis=-1)
        photo=np.abs(source.astype(np.float32)-warped.astype(np.float32)).mean(-1)/255
        inside=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
        if not np.isfinite(xy).all():raise ValueError('nonfinite motion')
        return dict(xy=xy,accepted=inside&(cycle<.75)&(photo<.08),cycle=cycle,photometric=photo)


def main():
    q=argparse.ArgumentParser()
    for key in ('plan','labels','weights','output'):q.add_argument('--'+key,type=Path,required=True)
    args=q.parse_args();args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('declare deterministic audit runtime')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.set_num_threads(4);cv2.setNumThreads(1)
    weight_hash=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if weight_hash!='01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27':
        raise ValueError('declared official C_T_V2 weight identity differs')
    complete=json.loads((args.labels/'complete.json').read_text())
    if complete.get('rigid_simulation_oracle_audit_only') is not True:raise ValueError('independent geometric scoring required')
    index={(r['case'],r['step']):r for r in json.loads((args.labels/'results.json').read_text())}
    estimator=Estimator(args.weights)
    result=dict(complete=False,records=[],repeat_controls=[],identity=dict(scope=__doc__,
        torch=torch.__version__,torchvision=torchvision.__version__,opencv=cv2.__version__,
        weight_sha256=weight_hash,parameter_load='strict_all_keys',policy_changed=False,
        source='https://docs.pytorch.org/vision/main/_modules/torchvision/models/optical_flow/raft.html',
        paper='https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),label_complete=complete))
    checked=set();torch.cuda.reset_peak_memory_stats()
    for row in json.loads(args.plan.read_text()):
        case=Path(row['case'])
        with np.load(case/'trajectory.npz',allow_pickle=False) as z:
            rgb={k:z[k] for k in ('rgb_static','rgb_gripper')}
        for step in row['steps']:
            entry=index[(case.name,step)];path=Path(entry['production_labels'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['production_labels_sha256']:raise ValueError('label bytes changed')
            with np.load(path,allow_pickle=False) as z:labels={k:z[k] for k in z.files}
            before,after=map(int,labels['source_frames'])
            if min(before,after)<0 or max(before,after)>step:raise ValueError('observed clock violation')
            cams=[]
            for camera,key in enumerate(('rgb_static','rgb_gripper')):
                a,b=rgb[key][before],rgb[key][after]
                torch.cuda.synchronize();start=time.perf_counter();flow=estimator(a,b)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                scored=score(flow,labels,camera,a.shape[:2]);scored['seconds']=elapsed;cams.append(scored)
                if camera not in checked:
                    repeat=estimator(a,b);error=float(np.max(np.abs(flow['xy']-repeat['xy'])))
                    if error!=0 or not np.array_equal(flow['accepted'],repeat['accepted']):raise ValueError('nonrepeatable estimator')
                    same=estimator(a,a);yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
                    result['repeat_controls'].append(dict(camera=camera,repeat_max=error,
                        identical_image_motion_pixels=stats(np.linalg.norm(same['xy']-np.stack((xx,yy),-1),axis=-1))))
                    checked.add(camera)
            result['records'].append(dict(case=case.name,case_id=row.get('case_id'),step=step,
                source_frames=[before,after],methods={'raft_small_C_T_V2':cams}))
            dump(args.output/'results.json',result);print(case.name,step,flush=True)
    if estimator.versions!={n:p._version for n,p in estimator.model.named_parameters()}:raise ValueError('frozen weights changed')
    result.update(complete=True,parameters_unchanged=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    dump(args.output/'results.json',result)


if __name__=='__main__':main()
