"""Wait for a declared run and evaluate its exact final checkpoint on all 18 cases.

This is evaluation only. It never promotes, trains, changes a controller,
overwrites an existing panel, or stops a process it did not start.
"""
from pathlib import Path
import argparse,collections,json,os,socket,subprocess,sys,time,urllib.request


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','run','manifest','output','training-status'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--egl',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--checkpoint-file',choices=('latest.pt','best.pt'),default='latest.pt')
    a=p.parse_args();checkpoint=a.run/'checkpoints'/a.checkpoint_file
    deadline=time.monotonic()+48*3600
    while not a.training_status.exists():
        if time.monotonic()>deadline:raise TimeoutError('training completion not reported within 48 hours')
        time.sleep(30)
    completion=json.loads(a.training_status.read_text())
    if completion.get('status')!='training_and_offline_finished' or completion.get('returncode')!=0:
        raise RuntimeError('training/offline stage failed; retain its original logs')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()!=a.source:
        raise ValueError('evaluation checkout source differs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=a.repo,text=True).strip():raise ValueError('evaluation checkout is dirty')
    import torch
    payload=torch.load(checkpoint,map_location='meta',weights_only=False)
    if payload['epoch']!=a.epoch or payload['global_step']!=a.step or payload['identity']['git_commit']!=a.source:
        raise ValueError('final checkpoint source/position differs')
    if payload['config']['optimizer']['batch_size']!=8:raise ValueError('panel requires the declared BS8 run')
    cases=json.loads(a.manifest.read_text())['cases']
    expected={f'push_{color}_block_{direction}':3 for color in ('blue','red','pink') for direction in ('left','right')}
    if len(cases)!=18 or dict(collections.Counter(x['task'] for x in cases))!=expected:raise ValueError('standard task/seed panel differs')
    with socket.socket() as port_check:port_check.bind(('127.0.0.1',a.port))
    a.output.mkdir(exist_ok=False)
    dump(a.output/'manifest.json',json.loads(a.manifest.read_text()))
    dump(a.output/'checkpoint_identity.json',dict(path=str(checkpoint),source=a.source,epoch=a.epoch,global_step=a.step,training_completion=completion))
    env=dict(os.environ,PYTHONPATH=str(a.repo),CUDA_VISIBLE_DEVICES=a.gpu,EGL_VISIBLE_DEVICES=a.egl,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    t5='/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'
    dino='/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'
    bridge_cmd=[sys.executable,'-B','-u','-m','clearvla.benchmarks.bridge','--checkpoint',str(checkpoint),'--t5-condition',t5,'--dinov3-model',dino,'--host','127.0.0.1','--port',str(a.port),'--device','cuda:0','--seed','0']
    bridge_log=(a.output/'bridge.log').open('w')
    bridge=subprocess.Popen(bridge_cmd,cwd=a.repo,env=env,stdout=bridge_log,stderr=subprocess.STDOUT)
    dump(a.output/'bridge_process.json',dict(pid=bridge.pid,command=bridge_cmd))
    records=[];errors=[]
    try:
        ready=False
        for _ in range(240):
            if bridge.poll() is not None:raise RuntimeError('bridge exited before readiness')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=5) as response:health=json.load(response)
                identity=health['deployment']['checkpoint']
                if health.get('status')=='ok' and identity['path']==str(checkpoint) and identity['global_step']==a.step and identity['git_commit']==a.source:
                    ready=True;dump(a.output/'bridge_health.json',health);break
            except (OSError,KeyError,ValueError):pass
            time.sleep(3)
        if not ready:raise TimeoutError('bridge identity admission timed out')
        for index,case in enumerate(cases,1):
            state=a.output/f'case_{index:02d}_initial_state.json';dump(state,case['initial_state'])
            trial=int(case['trial'])
            destination=a.output/f"case_{index:02d}_{case['task']}_trial{trial:02d}"
            cmd=['/home/sen.wang/.venvs/clearvla-calvin/bin/python','-B',str(a.repo/'probes/probe_controller_policy_closed_loop.py'),'--dataset-root','/data/senwang/data/calvin/raw/task_ABC_D','--state-json',str(state),'--task',case['task'],'--endpoint',f'http://127.0.0.1:{a.port}','--output-dir',str(destination),'--max-steps','360','--execute-rows','8','--anchor','stored_target']
            with (a.output/f'case_{index:02d}_evaluator.log').open('w') as log:done=subprocess.run(cmd,cwd=a.repo,env=env,stdout=log,stderr=subprocess.STDOUT)
            result=destination/'result.json'
            if done.returncode or not result.exists():errors.append(dict(index=index,returncode=done.returncode,result_present=result.exists()))
            else:
                row=json.loads(result.read_text())
                records.append(dict(index=index,task=case['task'],trial=trial,success=bool(row['success']),steps=int(row['steps']),result_path=str(result)))
            dump(a.output/'progress.json',dict(completed=len(records),errors=errors,case_returned=index,unix_time=time.time()))
            print('PANEL',index,records[-1] if records and records[-1]['index']==index else errors[-1],flush=True)
        by_task=collections.defaultdict(lambda:dict(trials=0,successes=0))
        for row in records:by_task[row['task']]['trials']+=1;by_task[row['task']]['successes']+=int(row['success'])
        complete=len(records)==18 and not errors
        summary=dict(schema='clearvla-causal-identity-panel-v1',checkpoint=str(checkpoint),source_commit=a.source,global_step=a.step,seed=0,max_steps=360,execute_rows=8,controller_anchor='stored_target',complete=complete,trials=len(records),expected_trials=18,successes=sum(int(x['success']) for x in records),by_task=dict(by_task),records=records,errors=errors)
        dump(a.output/'summary.json',summary);dump(a.output/'status.json',dict(status='complete' if complete else 'failed',trials=len(records),successes=summary['successes'],errors=errors))
        if not complete:raise RuntimeError('incomplete panel; failed case logs retained')
    finally:
        if bridge.poll() is None:
            bridge.terminate()
            try:bridge.wait(timeout=30)
            except subprocess.TimeoutExpired:bridge.kill();bridge.wait(timeout=10)
        bridge_log.close()


if __name__=='__main__':main()
