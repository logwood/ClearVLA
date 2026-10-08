"""Fixed-control local physical branches for endpoint-head command changes.

All arm controls are the original recorded commands. Only specified binary
gripper rows change. Later controls remain recorded, so this is not a new
closed-loop policy score or a full inference-recoding intervention.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
import pybullet as bullet
from clearvla.benchmarks.calvin_eval import _environment, _official_task_assets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--mode', choices=['baseline', 'repeat', 'changed'])
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text())
    if args.case_index is None:
        report = dict(complete=False, records=[], scope=__doc__, script_sha256=digest(__file__), plan_sha256=digest(args.plan))
        for index, row in enumerate(plan):
            outcomes = {}
            for mode in ['baseline', 'repeat', 'changed']:
                child_dir = args.output / ('case_%02d_%s' % (index, mode))
                with (args.output / ('case_%02d_%s.log' % (index, mode))).open('x') as log:
                    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--plan', str(args.plan),
                        '--output', str(child_dir), '--case-index', str(index), '--mode', mode], stdout=log, stderr=subprocess.STDOUT, check=True)
                outcomes[mode] = json.loads((child_dir/'result.json').read_text())
            for a, b in zip(outcomes['baseline']['states'], outcomes['repeat']['states']):
                if a != b:
                    raise AssertionError('fresh-process simulator baseline repeat changed')
            base, changed = outcomes['baseline']['states'], outcomes['changed']['states']
            report['records'].append(dict(source=row, baseline_repeat_exact=True,
                baseline_max_progress=max(x['target_signed_progress'] for x in base),
                changed_max_progress=max(x['target_signed_progress'] for x in changed),
                baseline_end_progress=base[-1]['target_signed_progress'], changed_end_progress=changed[-1]['target_signed_progress'],
                baseline_target_contact_states=sum(x['target_contact'] for x in base),
                changed_target_contact_states=sum(x['target_contact'] for x in changed),
                initial_rgb_exact=all(v['initial_rgb_exact'] for v in outcomes.values()),
                original_baseline_max_robot_error=outcomes['baseline']['original_max_robot_error'],
                original_baseline_max_scene_error=outcomes['baseline']['original_max_scene_error'],
                baseline_end_rgb_exact=outcomes['baseline']['end_rgb_exact'],
                states={k:v['states'] for k,v in outcomes.items()}))
            (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        report['complete'] = True
        (args.output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return
    row = plan[args.case_index]
    case = Path(row['case'])
    if digest(case/'trajectory.npz') != row['trajectory_sha256']:
        raise ValueError('source trajectory changed')
    source = json.loads((case/'result.json').read_text())
    with np.load(case/'trajectory.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in ('executed', 'rgb_static', 'rgb_gripper', 'robot_obs', 'scene_obs')}
    changes = {int(k):float(v) for k,v in row['command_changes'].items()}
    start, stop = row['state'], row['stop_state']
    if not changes or any(t < start or t >= start+8 for t in changes):
        raise ValueError('intervention must be in the selected executed prefix')
    if stop >= len(data['robot_obs']) or stop > len(data['executed']):
        raise ValueError('branch leaves recorded horizon')
    official, *_ = _official_task_assets()
    env = _environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'), show_gui=False)
    states = []
    original_robot_error = original_scene_error = 0.
    initial_exact = end_exact = False
    try:
        robot, scene = official.get_env_state_for_initial_condition(source['initial_state'])
        env.reset(robot_obs=robot, scene_obs=scene)
        objects = {o.name:int(o.uid) for o in env.scene.movable_objects}
        target = objects[row['target']]
        initial_x = bullet.getBasePositionAndOrientation(target, physicsClientId=env.robot.cid)[0][0]
        obs = env.get_obs()
        for t in range(stop+1):
            if t:
                command = data['executed'][t-1].copy()
                if args.mode == 'changed' and t-1 in changes:
                    command[-1] = changes[t-1]
                obs, *_ = env.step(command)
            if t == start:
                initial_exact = all(np.array_equal(obs['rgb_obs'][k], data[k][t]) for k in ('rgb_static','rgb_gripper'))
                if not initial_exact:
                    raise AssertionError('branch start does not exactly reproduce both recorded RGBs')
            if t < start:
                continue
            if args.mode != 'changed':
                original_robot_error = max(original_robot_error, float(np.abs(np.asarray(obs['robot_obs'])-data['robot_obs'][t]).max()))
                original_scene_error = max(original_scene_error, float(np.abs(np.asarray(obs['scene_obs'])-data['scene_obs'][t]).max()))
            position = bullet.getBasePositionAndOrientation(target, physicsClientId=env.robot.cid)[0]
            contacts = bullet.getContactPoints(bodyA=env.robot.robot_uid, bodyB=target, physicsClientId=env.robot.cid)
            states.append(dict(state=t, target_position=list(position), target_signed_progress=row['direction']*(position[0]-initial_x),
                tcp=list(np.asarray(obs['robot_obs'])[:3].astype(float)), gripper_width=float(obs['robot_obs'][6]), target_contact=bool(contacts)))
        end_exact = all(np.array_equal(obs['rgb_obs'][k],data[k][stop]) for k in ('rgb_static','rgb_gripper'))
        if args.mode != 'changed' and (not end_exact or original_robot_error > 2e-6 or original_scene_error > 2e-6):
            raise AssertionError(('recorded baseline reproduction failed', end_exact, original_robot_error, original_scene_error))
    finally:
        env.close()
    (args.output/'result.json').write_text(json.dumps(dict(mode=args.mode,initial_rgb_exact=initial_exact,end_rgb_exact=end_exact,
        original_max_robot_error=original_robot_error, original_max_scene_error=original_scene_error,states=states),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
