"""Sensor-only independent-data qualification of candidate negative regions.

The producer loads only raw RGB, metric depth and robot observations. It never
loads scene_obs, a renderer, audit masks or model ownership. Saved arrays are
compact candidate supervision, not RGB/DINO values. They are not admitted
training labels until the separate scorer and training-support gates pass.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor, pipeline
from probe_frozen_mask_proposals import SETTINGS, propose, sha, dump
from sensor_surface_groups import propose_groups
from probe_mask_surface_agreement import proposal_partition, negative_clique_size
from clearvla.vision.sensor_geometry import camera_views


RAW_KEYS=('robot_obs','rgb_static','rgb_gripper','depth_static','depth_gripper')


def main():
    p=argparse.ArgumentParser()
    for name in ('plan','calibration','weights','weight-receipt','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    plan=json.loads(args.plan.read_text());calibration=json.loads(args.calibration.read_text())
    receipt=json.loads(args.weight_receipt.read_text())
    assert args.weights.resolve()==Path(receipt['path']).resolve()
    for item in receipt['files']:
        assert sha(args.weights/item['name'])==item['sha256']
    files={f:s for row in plan['records'] for f,s in zip(row['raw_files'],row['sha256'])}
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    model,loading=Sam2Model.from_pretrained(args.weights,local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('uninitialized frozen teacher: '+str(loading))
    model=model.eval().cuda();model.requires_grad_(False)
    processor=Sam2Processor.from_pretrained(args.weights,local_files_only=True)
    generator=pipeline('mask-generation',model=model,image_processor=processor.image_processor,device=0)
    report=dict(complete=False,records=[],identity=dict(script_sha256=sha(__file__),
        plan_sha256=sha(args.plan),calibration_sha256=sha(args.calibration),weights=receipt,
        settings=SETTINGS,raw_key_allowlist=RAW_KEYS,production_changed=False,
        scope=__doc__,mask_loading=loading))
    checked=False;torch.cuda.reset_peak_memory_stats()
    for filename,expected in files.items():
        if sha(filename)!=expected:raise ValueError('raw bytes changed')
        with np.load(filename,allow_pickle=False) as z:data={k:z[k] for k in RAW_KEYS}
        views=camera_views(data['robot_obs'],calibration)
        rgbs=[data['rgb_'+c] for c in ('static','gripper')]
        depths=[data['depth_'+c] for c in ('static','gripper')]
        surfaces,metadata=propose_groups(depths,rgbs,views,calibration['projection'],
            support_mode='under_tcp_v2',workspace_point=data['robot_obs'][:3],connectivity_mode='geometry_only_v3')
        arrays={};camera_records=[]
        for c,rgb,surface in zip(('static','gripper'),rgbs,surfaces):
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():masks,scores=propose(generator,rgb)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            if not checked:
                with torch.inference_mode():again,other=propose(generator,rgb)
                np.testing.assert_array_equal(masks,again);np.testing.assert_array_equal(scores,other)
            part,groups=proposal_partition(masks,surface,interior_radius=1)
            arrays['partition_'+c]=part.astype(np.int16)
            arrays['groups_'+c]=groups.astype(np.int16)
            arrays['surface_'+c]=surface.astype(np.int16)
            arrays['masks_'+c]=masks
            camera_records.append(dict(camera=c,rgb_sha256=__import__('hashlib').sha256(rgb.tobytes()).hexdigest(),
                proposals=len(masks),regions=len(groups),support_pixels=int((part>=0).sum()),
                negative_clique_size=negative_clique_size(groups),seconds=elapsed))
        checked=True
        target=args.output/(Path(filename).stem+'-regions.npz');np.savez_compressed(target,**arrays)
        report['records'].append(dict(path=filename,raw_sha256=expected,labels_path=str(target),
            labels_sha256=sha(target),cameras=camera_records,geometry=metadata))
        dump(args.output/'results.json',report)
        print(Path(filename).name,[(x['regions'],x['support_pixels']) for x in camera_records],flush=True)
    report.update(complete=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),repeat_cameras_exact=True)
    dump(args.output/'results.json',report)


if __name__=='__main__':main()
