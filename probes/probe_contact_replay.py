"""Fresh-process factual replay and single-command contact-mechanism interventions.

Counterfactuals retain recorded arm commands; they are local physical tests,
not closed-loop evaluations of a modified policy.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment,_official_task_assets

def main():
    q=argparse.ArgumentParser();q.add_argument('--case',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--start',type=int,required=True);q.add_argument('--end',type=int,required=True)
    q.add_argument('--variant',choices=['baseline','gripper_closed','stop_descent'],default='baseline');a=q.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    result=json.loads((a.case/'result.json').read_text());official,*_=_official_task_assets()
    with np.load(a.case/'trajectory.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    recorded=json.loads((a.case/'environment_info.json').read_text())['info']
    env=_environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'),show_gui=False)
    original_step=p.stepSimulation;physics=[];steps=[];action_index=-1
    try:
        robot,scene=official.get_env_state_for_initial_condition(result['initial_state']);env.reset(robot_obs=robot,scene_obs=scene)
        target='block_'+result['task'].split('_')[1];uid=next(int(o.uid) for o in env.scene.movable_objects if o.name==target)
        # The environment owns one physics client in this isolated process.
        cid=env.robot.cid;robot_uid=env.robot.robot_uid
        links={i:p.getJointInfo(robot_uid,i,physicsClientId=cid)[12].decode() for i in range(p.getNumJoints(robot_uid,physicsClientId=cid))}
        def observed_step(*args,**kwargs):
            out=original_step(*args,**kwargs)
            if action_index>=a.start:
                contacts=p.getContactPoints(bodyA=uid,physicsClientId=cid)
                position=p.getBasePositionAndOrientation(uid,physicsClientId=cid)[0]
                physics.append({'action_index':action_index,'target_position':list(position),'contacts':[list(c) for c in contacts]})
            return out
        p.stepSimulation=observed_step;obs=env.get_obs();admitted=False
        for t in range(a.end):
            if t==a.start:
                exact=[np.array_equal(obs['rgb_obs'][k],data[k][t]) for k in ['rgb_static','rgb_gripper']]
                assert all(exact),('prefix RGB mismatch',a.case,t,exact)
                admitted=True
            action_index=t;action=data['executed'][t].copy()
            if t>=a.start:
                if a.variant=='gripper_closed':action[6]=-1
                elif a.variant=='stop_descent':action[2]=max(float(action[2]),0.)
            submitted=action.copy();obs,*_=env.step(action)
            if a.variant=='baseline':
                assert all(np.array_equal(obs['rgb_obs'][k],data[k][t+1]) for k in ['rgb_static','rgb_gripper']),('factual RGB mismatch',t+1)
            if t>=a.start:
                target_pos=p.getBasePositionAndOrientation(uid,physicsClientId=cid)[0]
                steps.append({'state':t+1,'submitted':submitted.tolist(),'target_position':list(target_pos),'robot_obs':obs['robot_obs'].tolist()})
        assert admitted and physics
        first=np.array(recorded[0]['scene_info']['movable_objects'][target]['current_pos']);last=np.array(steps[-1]['target_position'])
        before=np.array(recorded[a.start]['scene_info']['movable_objects'][target]['current_pos']);sign=-1 if result['task'].endswith('left') else 1
        out={'case':str(a.case),'variant':a.variant,'window':[a.start,a.end],'prefix_exact_rgb':True,
             'factual_full_window_exact_rgb':a.variant=='baseline','robot_uid':robot_uid,'target_uid':uid,'robot_links':links,
             'counterfactual_scope':'recorded arm commands after intervention; no policy replanning or efficacy claim',
             'target_total_signed_dx':float(sign*(last[0]-first[0])),'target_window_delta':(last-before).tolist(),
             'target_min_z':float(min(x['target_position'][2] for x in physics)),
             'steps':steps,'physics_substeps':physics}
        a.output.write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:v for k,v in out.items() if k not in ['steps','physics_substeps','robot_links']},indent=2))
    finally:p.stepSimulation=original_step;env.close()
if __name__=='__main__':main()
