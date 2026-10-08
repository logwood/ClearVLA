"""Pinned full-label preparation followed by ONE meaningful BS8 short.

No formal promotion. Every unchanged sampler row is declared before labels;
GPU workers use idle cards only and retain all SAM prompts and source budgets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();r=json.loads(args.receipt.read_text())
    out=Path(r['output']);out.mkdir(exist_ok=False)
    deadline=time.monotonic()+48*3600;started=time.time();active={};done={};launched=set()
    source=Path(r['source']);root=Path(r['experiment_root']);run=r['run_name']
    status_path=out/'status.json';train_status=root/(run+'-status.json')
    def status(state,**fields):
        dump(status_path,dict(state=state,unix_time=time.time(),formal_promoted=False,
            completed_shards=sorted(done),active_shards={str(k):dict(pid=v['process'].pid,gpu=v['gpu']) for k,v in active.items()},**fields))
    def wait():
        if time.monotonic()>deadline:raise TimeoutError('short pipeline exceeded 48 hours')
        time.sleep(30)
    def idle(gpu):
        used=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        return used<512
    def check_source():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=r['source_commit']:
            raise ValueError('immutable training source differs')
        if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
            raise ValueError('immutable training source dirty')
        for path,digest in r['file_sha256'].items():
            if sha(path)!=digest:raise ValueError('pinned pipeline input changed: '+path)
    try:
        check_source()
        status('waiting_mechanical_admission')
        admission=Path(r['mechanical_admission'])
        while not admission.exists():wait()
        decision=json.loads(admission.read_text())
        if decision.get('mechanically_admitted') is not True or decision.get('source')!=r['source_commit']:
            raise ValueError('mechanical/deployment admission rejected')
        plan=json.loads(Path(r['plan']).read_text())
        if not plan['complete'] or plan['batch_size']!=8 or plan['train_batches']!=1024 or plan['val_batches']!=256:
            raise ValueError('meaningful-short exposure differs')
        all_keys={(x['source_split'],*x['source_frames']):x for x in plan['records']}
        if len(all_keys)!=len(plan['records']):raise ValueError('duplicate source clock in exposure')
        workers=int(r['shards'])
        if not 1<=workers<=4:raise ValueError('invalid annotation worker budget')
        while len(done)<workers:
            for shard,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close()
                if code:raise RuntimeError('annotation failed; retain logs for shard '+str(shard))
                file=item['output']/'manifest.json';doc=json.loads(file.read_text())
                expected=[row for i,row in enumerate(plan['records']) if i%workers==shard]
                keys=[(x['source_split'],*x['source_frames']) for x in doc['records']]
                if not doc['complete'] or doc['shard']!=shard or doc['shards']!=workers or len(keys)!=len(set(keys)) or set(keys)!={(x['source_split'],*x['source_frames']) for x in expected}:
                    raise ValueError('shard does not cover its full declared exposure')
                done[shard]=dict(path=str(file),manifest=doc,gpu=item['gpu']);del active[shard]
            busy={item['gpu'] for item in active.values()}
            for shard in range(workers):
                if shard in launched:continue
                gpu=next((g for g in r['gpus'] if g not in busy and idle(g)),None)
                if gpu is None:break
                dest=root/(r['label_prefix']+'-shard'+str(shard));log=(out/('shard%d.log'%shard)).open('x')
                cmd=[r['python'],'-B','-u',str(source/'probes/build_identity_region_annotations.py'),
                    '--plan',r['plan'],'--weights',r['weights'],'--weight-receipt',r['weight_receipt'],
                    '--raw-root',r['raw_root'],'--output',str(dest),'--shard',str(shard),'--shards',str(workers)]
                env=dict(os.environ,PYTHONPATH=r['annotation_dependencies']+':'+str(source),
                    CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                proc=subprocess.Popen(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[shard]=dict(process=proc,gpu=gpu,output=dest,log=log);launched.add(shard);busy.add(gpu)
                dump(out/('shard%d-job.json'%shard),dict(pid=proc.pid,gpu=gpu,command=cmd,unix_time=time.time()))
            status('annotating' if active else 'waiting_idle_gpu_for_annotations')
            if len(done)<workers:wait()
        records=[]
        for shard in range(workers):
            doc=done[shard]['manifest']
            if doc['plan_sha256']!=sha(r['plan']):raise ValueError('annotation plan mismatch')
            for row in doc['records']:
                key=(row['source_split'],*row['source_frames'])
                if row['raw_sha256']!=all_keys[key]['sha256'] or sha(row['labels_path'])!=row['labels_sha256']:
                    raise ValueError('label/raw identity changed')
                records.append(row)
        if len(records)!=len(all_keys):raise ValueError('combined annotation coverage differs')
        manifest=root/(r['label_prefix']+'-manifest.json')
        if manifest.exists():raise FileExistsError(manifest)
        first=done[0]['manifest']
        merged=dict(schema=first['schema'],supervision_only=True,complete=True,records=records,
            plan_sha256=sha(r['plan']),producer=first['producer'],shards=[dict(path=done[s]['path'],sha256=sha(done[s]['path']),gpu=done[s]['gpu']) for s in range(workers)],
            label_wall_seconds=time.time()-started,worker_seconds=sum(done[s]['manifest']['elapsed_seconds'] for s in done),
            cost_scope='Includes separately scheduled full-prompt annotation; no RGB/DINO value cache')
        dump(manifest,merged)
        check_source()
        cfg=json.loads(Path(r['base_config']).read_text())
        if cfg['optimizer']['batch_size']!=8 or cfg['runtime']['max_train_batches']!=1024 or cfg['runtime']['max_val_batches']!=256:
            raise ValueError('base meaningful-short budget differs')
        cfg['data'].update(output_dir=str(root/run),identity_region_manifest=str(manifest),identity_region_manifest_sha256=sha(manifest))
        cfg['top'].update(identity_supervision_mode='rgbd_temporal_regions_v3',identity_region_js_margin=.1)
        cfg['objectives'].update(identity_region_separation=.02,identity_region_prediction=.02)
        config=root/(run+'.config.json')
        if config.exists() or (root/run).exists():raise FileExistsError(config)
        dump(config,cfg)
        status('waiting_idle_training_gpu',gpu=r['training_gpu'],label_manifest=str(manifest))
        while not idle(r['training_gpu']):wait()
        check_source()
        command=[r['python'],'-B','-u','-m','clearvla.mainline.train','--config',str(config),
            '--init-checkpoint',r['init_checkpoint'],'--init-model-contract-migration','causal_identity_ab_v1',
            '--init-training-clock','checkpoint','--device','cuda:0']
        env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=r['training_gpu'],OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        with (root/(run+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            dump(root/(run+'-job.json'),dict(pid=process.pid,command=command,source=r['source_commit'],
                config_sha256=sha(config),annotation_manifest_sha256=sha(manifest),created=time.time(),formal_promoted=False))
            status('training_short',pid=process.pid,config=str(config))
            code=process.wait()
        dump(train_status,dict(status='training_and_offline_finished' if code==0 else 'failed',returncode=code,unix_time=time.time()))
        if code:raise RuntimeError('meaningful short failed; retained log')
        status('complete',scope='One meaningful short and offline validation complete; standard18 and scientific qualification remain; no formal promotion')
    except BaseException as error:
        status('failed',error=repr(error),note='Existing worker logs/results retained; no other job stopped')
        raise


if __name__=='__main__':main()
