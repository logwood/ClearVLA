"""Exact-replay audit of sensor-only surface groups. No training/policy edit."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import numpy as np
import pybullet as p
from clearvla.benchmarks.calvin_eval import _environment, _official_task_assets
from clearvla.vision.sensor_geometry import camera_views
from probe_rgbd_correspondence import matrix
from sensor_surface_groups import propose_groups, audit_groups, SETTINGS


def dump(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def main():
    q = argparse.ArgumentParser()
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--index', type=int)
    q.add_argument('--support-mode', choices=('dominant_v1','under_tcp_v2'), default='dominant_v1')
    q.add_argument('--export-audit-masks', action='store_true', help='Store oracle body/link masks for a separate scorer only')
    a = q.parse_args(); a.output.mkdir(exist_ok=False)
    plan = json.loads(a.plan.read_text())
    identity = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    proposal_sha256=hashlib.sha256(Path(__file__).with_name('sensor_surface_groups.py').read_bytes()).hexdigest(),
                    plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(), settings=SETTINGS,
                    support_mode=a.support_mode,
                    oracle_masks_exported_for_scoring=a.export_audit_masks,
                    scope=__doc__, production_changed=False,
                    algorithm_source='https://pointclouds.org/documentation/tutorials/cluster_extraction.html')
    report = dict(identity=identity, records=[], complete=False)
    if a.index is None:
        for i in range(len(plan)):
            folder = a.output/('worker_%02d'%i)
            with (a.output/('worker_%02d.log'%i)).open('x') as log:
                child = subprocess.run([sys.executable, '-B', '-u', __file__, '--plan', str(a.plan), '--output', str(folder), '--index', str(i), '--support-mode', a.support_mode] + (['--export-audit-masks'] if a.export_audit_masks else []), stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError('surface replay failed; retained '+str(folder)+'.log')
            value = json.loads((folder/'results.json').read_text())
            if not value['complete']: raise ValueError('incomplete worker')
            report['records'].extend(value['records']); dump(a.output/'results.json', report)
        report['complete'] = True; dump(a.output/'results.json', report)
        return
    row = plan[a.index]; case = Path(row['case'])
    result = json.loads((case/'result.json').read_text())
    calibration = json.loads((Path(__file__).resolve().parents[1]/'clearvla/mainline/assets/calvin_rgbd_joint_geometry_v1.json').read_text())
    with np.load(case/'trajectory.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in ('executed','robot_obs','rgb_static','rgb_gripper')}
    official, *_ = _official_task_assets()
    env = _environment(Path('/data/senwang/data/calvin/raw/task_ABC_D'), show_gui=False)
    try:
        robot, scene = official.get_env_state_for_initial_condition(result['initial_state'])
        env.reset(robot_obs=robot, scene_obs=scene); obs = env.get_obs()
        objects = {int(o.uid):o.name for o in env.scene.movable_objects}
        wanted = sorted(set(row['steps']))
        for step in range(max(wanted)+1):
            if step: obs, *_ = env.step(data['executed'][step-1].copy())
            if step not in wanted: continue
            depths, rgbs, body_maps, link_maps, actual_views = [], [], [], [], []
            for camera, key in zip(env.cameras, ('static','gripper')):
                rgb, depth = camera.render()
                np.testing.assert_array_equal(rgb, data['rgb_'+key][step])
                frame = p.getCameraImage(width=camera.width,height=camera.height,
                    viewMatrix=matrix(camera,'view').T.flatten().tolist(),
                    projectionMatrix=matrix(camera,'projection').T.flatten().tolist(),
                    physicsClientId=camera.cid,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                label = np.asarray(frame[4]).reshape(camera.height,camera.width)
                body_maps.append(np.where(label<0,-1,label&((1<<24)-1)))
                link_maps.append(np.where(label<0,-2,(label>>24)-1))
                depths.append(depth); rgbs.append(rgb); actual_views.append(matrix(camera,'view'))
            views = camera_views(data['robot_obs'][step], calibration)
            # Full body maps remain outside both calls to the sensor producer.
            group_maps, producer = propose_groups(depths, rgbs, views, calibration['projection'],
                support_mode=a.support_mode, workspace_point=data['robot_obs'][step, :3])
            again, other = propose_groups(depths, rgbs, views, calibration['projection'],
                support_mode=a.support_mode, workspace_point=data['robot_obs'][step, :3])
            for x, y in zip(group_maps, again): np.testing.assert_array_equal(x, y)
            if producer != other: raise ValueError('non-deterministic sensor grouping')
            audit = audit_groups(group_maps, body_maps, objects)
            path = a.output/('groups_%03d.npz'%step)
            exported = dict(group_static=group_maps[0], group_gripper=group_maps[1])
            if a.export_audit_masks:
                exported.update(audit_body_static=body_maps[0], audit_body_gripper=body_maps[1],
                                audit_link_static=link_maps[0], audit_link_gripper=link_maps[1])
            np.savez_compressed(path, **exported)
            report['records'].append(dict(case=case.name, case_id=row.get('case_id'), step=step,
                production_view_max_abs_error=max(float(np.abs(x-y).max()) for x,y in zip(views,actual_views)),
                producer=producer,audit=audit,groups_path=str(path),groups_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repeat_equal=True))
            dump(a.output/'results.json', report)
            print('GROUP_AUDIT',case.name,step,producer['accepted_groups'],flush=True)
        report['complete'] = True; dump(a.output/'results.json', report)
    finally:
        env.close()


if __name__ == '__main__': main()
