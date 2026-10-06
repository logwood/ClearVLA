"""Bind raw-label semantics to the actual 18-rollout controller recurrence."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,hashlib,subprocess,os
import numpy as np

def main():
 q=argparse.ArgumentParser();q.add_argument('--audit',type=Path,required=True);q.add_argument('--rollout',type=Path,required=True);a=q.parse_args();output=a.audit/'controller-anchor';output.mkdir(exist_ok=False)
 episodes=json.loads((a.audit/'training-geometry-r2/episodes.json').read_text());checks=[]
 for r in [v for v in episodes if v['split']=='train'][::43]:
  for t in [r['source_start'],r['source_start']+16,r['source_end']-8]:
   path=Path('/data/senwang/data/calvin/raw/task_ABC_D/training')/('episode_%07d.npz'%t)
   with np.load(path,allow_pickle=False) as z:
    expected=np.clip(z['actions'][:3]-z['robot_obs'][:3],-.02,.02)/.02
    error=float(np.max(np.abs(expected-z['rel_actions'][:3])))
   checks.append({'episode':r['episode'],'frame':t,'max_abs_error':error})
 cases={};recurrences=[]
 for case in sorted(a.rollout.glob('case_*')):
  if not case.is_dir() or not (case/'trajectory.npz').exists():continue
  cid=int(case.name.split('_')[1]);cases[cid]=case;info=json.loads((case/'environment_info.json').read_text());targets=np.array([r['target_pos'] for r in info['controller_targets']])
  with np.load(case/'trajectory.npz',allow_pickle=False) as z:
   command=z['executed'];tcp=z['robot_obs'][:,:3]
   expected=targets[:-1]+.02*command[:,:3]
   recurrence=float(np.max(np.abs(targets[1:]-expected)));gaps=np.linalg.norm(targets-tcp,axis=-1)
   recurrences.append({'case_id':cid,'steps':len(command),'stored_target_recurrence_max_abs_error':recurrence,
      'max_target_tcp_gap':float(gaps.max()),'p95_target_tcp_gap':float(np.quantile(gaps,.95)),'peak_state':int(gaps.argmax())})
 identity={'training_formula':'rel_xyz = clip(absolute_target_xyz - measured_tcp_xyz, -.02, .02)/.02',
   'deployed_formula':'stored_target_next = stored_target_previous + .02 * rel_xyz',
   'raw_training_checks':checks,'raw_training_max_abs_error':max(r['max_abs_error'] for r in checks),'rollout_recurrences':recurrences,
   'training_source':'calvin_env/utils/utils.py:160-171','deployment_source':'calvin_env/robot/robot.py:228-242',
   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (output/'semantics.json').write_text(json.dumps(identity,indent=2)+'\n')
 jobs=[(cid,start,end,anchor) for cid,start,end in [(2,32,120),(9,240,280),(14,328,360),(17,136,168)] for anchor in ['stored_target','measured_tcp']]
 def run(job):
  cid,start,end,anchor=job;name=f'case{cid:02}_{start}_{end}_{anchor}';dest=output/(name+'.json')
  command=['/home/sen.wang/.venvs/clearvla-calvin/bin/python','-u','-m','probes.probe_controller_anchor_replay','--case',str(cases[cid]),'--start',str(start),'--end',str(end),'--anchor',anchor,'--output',str(dest)]
  with open(output/(name+'.log'),'w') as log:
   cp=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
  if cp.returncode:raise RuntimeError(f'{name}: exit {cp.returncode}; retained log')
  row=json.loads(dest.read_text());small={k:v for k,v in row.items() if k not in ['steps','physics']};print(json.dumps(small),flush=True);return small
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
 (output/'replay_summary.json').write_text(json.dumps(results,indent=2)+'\n')
 (output/'complete.json').write_text(json.dumps({'jobs':len(results),'status':'complete','scope':'diagnostic physics replays only; formal evaluator configuration unchanged'},indent=2)+'\n')
if __name__=='__main__':main()
