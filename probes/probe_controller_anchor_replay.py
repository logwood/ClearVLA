"""Compare identical recorded commands under the two CALVIN TCP anchors.

One fresh simulator per invocation. This is not a closed-loop policy score.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment,_official_task_assets

def main():
 q=argparse.ArgumentParser();q.add_argument('--case',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--start',type=int,required=True);q.add_argument('--end',type=int,required=True);q.add_argument('--anchor',choices=['stored_target','measured_tcp'],required=True);a=q.parse_args()
 assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
 result=json.loads((a.case/'result.json').read_text());official,*_=_official_task_assets()
 with np.load(a.case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 env=_environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'),show_gui=False);original_step=p.stepSimulation;relative=env.robot.relative_to_absolute
 action_index=-1;last_applied=None;physics=[];steps=[]
 try:
  robot,scene=official.get_env_state_for_initial_condition(result['initial_state']);env.reset(robot_obs=robot,scene_obs=scene)
  assert env.robot.use_target_pose is True
  target='block_'+result['task'].split('_')[1];uid=next(int(o.uid) for o in env.scene.movable_objects if o.name==target);robot_uid=env.robot.robot_uid;cid=env.robot.cid
  first=np.asarray(p.getBasePositionAndOrientation(uid,physicsClientId=cid)[0]);obs=env.get_obs()
  def applied(*args,**kwargs):
   nonlocal last_applied
   out=relative(*args,**kwargs);last_applied=np.asarray(out[0]).copy();return out
  def physics_step(*args,**kwargs):
   out=original_step(*args,**kwargs)
   if action_index>=a.start:
    contacts=p.getContactPoints(bodyA=uid,physicsClientId=cid);robot_contacts=[c for c in contacts if c[2]==robot_uid]
    physics.append({'step':action_index,'target_z':float(p.getBasePositionAndOrientation(uid,physicsClientId=cid)[0][2]),
     'robot_target_contacts':len(robot_contacts),'max_robot_target_force':max([float(c[9]) for c in robot_contacts]+[0.])})
   return out
  env.robot.relative_to_absolute=applied;p.stepSimulation=physics_step
  for t in range(a.end):
   if t==a.start:
    exact=[np.array_equal(obs['rgb_obs'][k],data[k][t]) for k in ['rgb_static','rgb_gripper']];assert all(exact),('prefix mismatch',t,exact)
    env.robot.use_target_pose=a.anchor=='stored_target'
   action_index=t;command=data['executed'][t].copy();before=np.asarray(obs['robot_obs'][:3]).copy();obs,*_=env.step(command.copy())
   if a.anchor=='stored_target':assert all(np.array_equal(obs['rgb_obs'][k],data[k][t+1]) for k in ['rgb_static','rgb_gripper']),('factual mismatch',t+1)
   if t>=a.start:
    steps.append({'state':t+1,'command':command.tolist(),'target_xyz':list(p.getBasePositionAndOrientation(uid,physicsClientId=cid)[0]),'tcp_xyz':obs['robot_obs'][:3].tolist(),
      'applied_goal_xyz':last_applied.tolist(),'goal_vs_pre_step_tcp':float(np.linalg.norm(last_applied-before)),
      'goal_vs_post_step_tcp':float(np.linalg.norm(last_applied-obs['robot_obs'][:3]))})
  sign=-1 if result['task'].endswith('left') else 1
  out={'case':str(a.case),'window':[a.start,a.end],'anchor':a.anchor,'prefix_exact_rgb':True,'factual_window_exact_rgb':a.anchor=='stored_target',
    'target_signed_dx':float(sign*(steps[-1]['target_xyz'][0]-first[0])),
    'max_goal_tcp_gap':max(s['goal_vs_post_step_tcp'] for s in steps),
    'target_min_z':min(s['target_z'] for s in physics),'max_target_contact_force':max(s['max_robot_target_force'] for s in physics),
    'robot_target_contact_substeps':sum(s['robot_target_contacts']>0 for s in physics),
    'scope':'same factual prefix and fixed recorded arm/gripper commands; only command anchor changes; not policy closed-loop result',
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':steps,'physics':physics}
  a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['steps','physics']}),flush=True)
 finally:p.stepSimulation=original_step;env.robot.relative_to_absolute=relative;env.close()
if __name__=='__main__':main()
