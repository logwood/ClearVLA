"""Finish the bounded pilot with verified complete artifacts, without touching models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np


def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--wait-timeout-seconds',type=float,default=14400)
    args=p.parse_args();root=args.root.resolve();repo=Path(__file__).resolve().parents[1]
    started=time.time();status=root/'finalization-status.json'
    write_json(status,dict(status='waiting_for_pipeline',started_unix=started))
    try:
        while True:
            pipeline=json.loads((root/'pipeline-status.json').read_text())
            if pipeline['status']=='failed':raise RuntimeError('Pipeline failed: '+str(pipeline.get('error')))
            if pipeline['status']=='complete':break
            if time.time()-started>args.wait_timeout_seconds:raise TimeoutError('Bounded finalizer wait expired')
            time.sleep(15)
        write_json(status,dict(status='analyzing',started_unix=started))
        for name,paths in [('paired',[root/'control-pilot',root/'candidate-pilot'])]:
            with (root/(name+'-training-audit.json')).open('w') as stream:
                subprocess.run([sys.executable,'-B','-m','clearvla.tools.audit_policy_logs',*map(str,paths),'--format','json'],cwd=repo,stdout=stream,check=True)
        subprocess.run([sys.executable,'-B',str(repo/'scripts/analyze_maniskill_spatial_repair.py'),
            '--root',str(root),'--baseline',str(args.baseline)],cwd=repo,check=True)
        shutil.copy2(repo/'configs/mainline/maniskill_spatial_repair_experiment_20261008.json',root/'experiment-specification.json')
        receipts=[]
        for arm,folder in [('original','reference-baseline-eval18'),('control','control-eval18'),('candidate','candidate-eval18')]:
            episodes=sorted((root/folder).glob('episode_*'))
            if len(episodes)!=18:raise RuntimeError('Missing full episodes: '+arm)
            for episode in episodes:
                result=json.loads((episode/'result.json').read_text())
                with np.load(episode/'trajectory.npz',allow_pickle=False) as arrays:
                    assert arrays['executed'].shape==(400,7)
                    assert arrays['robot_obs_trajectory'].shape==(401,15)
                    assert arrays['object_positions'].shape==(401,3,3)
                    assert arrays['raw_chunks'].shape==(50,24,7)
                    for key in arrays.files:
                        if np.issubdtype(arrays[key].dtype,np.number):assert np.isfinite(arrays[key]).all(),key
                capture=cv2.VideoCapture(str(episode/'video.mp4'));frames=0
                while True:
                    ok,frame=capture.read()
                    if not ok:break
                    assert frame.ndim==3 and min(frame.shape[:2])>=336
                    frames+=1
                capture.release()
                assert frames==result['video_frames']==401
                receipts.append(dict(arm=arm,seed=result['seed'],actions=400,observations=401,decoded_video_frames=frames))
        checkpoint_receipts={}
        for arm in ('control','candidate'):
            checkpoint=root/(arm+'-pilot/checkpoints/latest.pt')
            checkpoint_receipts[arm]=dict(path=str(checkpoint),sha256=sha256(checkpoint))
        verification=dict(status='passed',training_source_commit=pipeline['source_commit'],
            analysis_script_sha256=sha256(repo/'scripts/analyze_maniskill_spatial_repair.py'),
            finalizer_script_sha256=sha256(Path(__file__)),checkpoints=checkpoint_receipts,
            trajectories=receipts,full_npz_files=54,full_video_files=54)
        write_json(root/'artifact-verification.json',verification)
        folders=['report','control-heldout','candidate-heldout','control-placements','candidate-placements',
            'control-eval18','candidate-eval18','reference-baseline-eval18']
        files=set()
        for folder in folders:files.update(p for p in (root/folder).rglob('*') if p.is_file())
        for arm in ('control','candidate'):
            files.update(root/(arm+'-pilot')/name for name in ('metrics.jsonl','run_context.json'))
        names=['pipeline-status.json','experiment-specification.json','preflight-admission.json','deployment-smoke-admission.json',
            'label_audit.json','paired-training-audit.json','artifact-verification.json','unit-tests-r3.log',
            'control-preflight-r1.json','candidate-preflight-r1.json','control-preflight-r1.npz','candidate-preflight-r1.npz',
            'control-pilot.log','candidate-pilot.log']
        files.update(root/name for name in names)
        for file in files:
            assert file.is_file() and file.resolve().is_relative_to(root)
        manifest=dict(schema='maniskill-spatial-repair-export-v1',files=[
            dict(path=str(file.relative_to(root)),bytes=file.stat().st_size,sha256=sha256(file)) for file in sorted(files)])
        write_json(root/'export-manifest.json',manifest)
        files.add(root/'export-manifest.json')
        archive=root/'spatial-repair-complete.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as output:
            for file in sorted(files):output.write(file,str(file.relative_to(root)))
        with zipfile.ZipFile(archive) as output:
            bad=output.testzip()
            if bad is not None:raise RuntimeError('Archive CRC failed: '+bad)
        write_json(root/'export-receipt.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha256(archive),
            manifest_files=len(manifest['files']),trajectories=54,videos=54))
        write_json(status,dict(status='complete',started_unix=started,finished_unix=time.time(),
            report=str(root/'report/index.html'),archive=str(archive)))
        print('VERIFIED_EXPORT_COMPLETE',flush=True)
    except BaseException as error:
        write_json(status,dict(status='failed',started_unix=started,error=repr(error)))
        raise


if __name__=='__main__':main()
