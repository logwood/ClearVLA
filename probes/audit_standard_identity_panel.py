"""Audit complete standard R8 panels from their saved physical trajectories.

Only recorded states/contact samples are used. No simulator oracle enters a
policy, and a missing sampled contact is not proof of no physical contact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_panel(root):
    summary = json.loads((root / 'summary.json').read_text())
    required = dict(complete=True, trials=18, expected_trials=18, seed=0,
                    execute_rows=8, max_steps=360, controller_anchor='stored_target')
    if any(summary.get(k) != v for k, v in required.items()) or summary.get('errors'):
        raise ValueError('incomplete or nonstandard panel: ' + str(root))
    rows = {int(row['index']): row for row in summary['records']}
    if sorted(rows) != list(range(1, 19)):
        raise ValueError('case indices are not the standard complete 18')
    return summary, rows


def physical_ledger(case, result, arrays):
    telemetry = json.loads((case / 'environment_info.json').read_text())
    infos = telemetry['info']
    steps = len(arrays['executed'])
    if len(infos) != steps + 1 or len(telemetry['controller_targets']) != steps + 1:
        raise ValueError('physical clock differs')
    objects = list(infos[0]['scene_info']['movable_objects'])
    target = 'block_' + result['task'].split('_')[1]
    direction = -1 if result['task'].endswith('left') else 1
    positions = {name: np.asarray([row['scene_info']['movable_objects'][name]['current_pos']
                                  for row in infos]) for name in objects}
    contacts = {}
    for name in objects:
        robot_contacts = np.asarray([any(c[9] > 0 and row['robot_info']['uid'] in (c[1], c[2])
                                    for c in row['scene_info']['movable_objects'][name]['contacts'])
                                    for row in infos])
        indices = np.flatnonzero(robot_contacts)
        contacts[name] = int(indices[0]) if len(indices) else None
    tcp = arrays['robot_obs'][:, :3]
    goals = np.asarray([row['target_pos'] for row in telemetry['controller_targets']])
    gap = np.linalg.norm(goals - tcp, axis=1)
    progress = direction * (positions[target][:, 0] - positions[target][0, 0])
    peak = int(progress.argmax())
    replans = []
    for start in range(0, steps, 8):
        end = min(start + 8, steps)
        cmd = arrays['executed'][start:end]
        replans.append(dict(state=start, target_progress_m=float(progress[start]),
            progress_next8_m=float(progress[end] - progress[start]),
            mean_native_xyz=cmd[:, :3].mean(0).tolist(),
            actual_tcp_delta=(tcp[end] - tcp[start]).tolist(),
            gripper=cmd[:, 6].tolist(), gap_m=float(gap[start])))
    return dict(target=target, first_recorded_robot_contacts=contacts,
        peak_target_progress_m=float(progress[peak]), peak_state=peak,
        final_target_progress_m=float(progress[-1]),
        controller_tcp_gap_at_peak_m=float(gap[peak]),
        max_controller_tcp_gap_m=float(gap.max()),
        gripper_switches=int(np.count_nonzero(np.diff(arrays['executed'][:, 6]))),
        replans=replans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--reference', action='append', default=[], help='NAME=complete panel directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary, rows = load_panel(args.panel)
    references = {}
    for value in args.reference:
        name, root = value.split('=', 1)
        if name in references:
            raise ValueError('duplicate reference name')
        references[name] = (Path(root), *load_panel(Path(root)))
    records = []
    for index, row in sorted(rows.items()):
        case = Path(row['result_path']).parent
        if case.resolve().parent != args.panel.resolve():
            raise ValueError('case path escaped declared panel')
        result = json.loads((case / 'result.json').read_text())
        if not result['complete'] or bool(result['success']) != bool(row['success']):
            raise ValueError('case result/summary mismatch')
        steps = int(result['steps'])
        with np.load(case / 'trajectory.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if not 0 < steps <= 360 or arrays['executed'].shape != (steps, 7):
            raise ValueError('wrong executed clock')
        for key in ('robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper'):
            if len(arrays[key]) != steps + 1:
                raise ValueError('state and action lengths differ')
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError('nonfinite recorded array: ' + key)
        planned = arrays['raw_chunks']
        plan_index, chunk_row = arrays['executed_plan_index'], arrays['executed_chunk_row']
        if planned.ndim != 3 or planned.shape[1:] != (24, 7):
            raise ValueError('recorded plan layout changed')
        if (plan_index.shape != (steps,) or chunk_row.shape != (steps,) or
                not np.array_equal(plan_index, np.arange(steps) // 8) or
                not np.array_equal(chunk_row, np.arange(steps) % 8)):
            raise ValueError('deployment did not execute consecutive R8 prefixes')
        raw_reproduced = planned[plan_index, chunk_row]
        if not np.array_equal(raw_reproduced, arrays['raw_executed']):
            raise ValueError('executed raw prefixes differ from recorded plans')
        prior = {}
        for name, (root, _, old_rows) in references.items():
            old = old_rows[index]
            if old['task'] != row['task'] or old['trial'] != row['trial']:
                raise ValueError('case task/trial changed')
            with np.load(Path(old['result_path']).parent / 'trajectory.npz', allow_pickle=False) as z:
                delta = {key: float(np.max(np.abs(arrays[key][0].astype(np.float64) -
                            z[key][0].astype(np.float64)))) for key in ('robot_obs', 'scene_obs')}
                rgb_equal = all(np.array_equal(arrays[key][0], z[key][0])
                                for key in ('rgb_static', 'rgb_gripper'))
            if any(delta.values()) or not rgb_equal:
                raise ValueError('initial observations differ from ' + name)
            prior[name] = dict(success=old['success'], steps=old['steps'],
                               initial_state_max_difference=delta, initial_rgb_equal=rgb_equal)
        trajectory = case / 'trajectory.npz'
        records.append(dict(index=index, task=row['task'], success=row['success'], steps=steps,
            trajectory=str(trajectory), sha256=sha(trajectory), bytes=trajectory.stat().st_size,
            shapes={key: list(value.shape) for key, value in arrays.items()},
            prior=prior, physical=physical_ledger(case, result, arrays)))
    report = dict(complete=True, standard={k: summary[k] for k in
        ('source_commit', 'global_step', 'seed', 'execute_rows', 'max_steps', 'controller_anchor')},
        successes=summary['successes'], failures=[r['index'] for r in records if not r['success']],
        comparison={name: dict(prior_successes=old_summary['successes'],
          gained=[r['index'] for r in records if r['success'] and not r['prior'][name]['success']],
          lost=[r['index'] for r in records if not r['success'] and r['prior'][name]['success']])
          for name, (_, old_summary, _) in references.items()},
        records=records, script_sha256=sha(__file__), formal_promoted=False,
        scope='Complete physical panel audit; sampled contacts are not every physics substep; '
              'official success does not certify correct color identity, and score differences do not isolate a cause.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    for row in records:
        if not row['success']:
            print('FAILURE', row['index'], {k: v for k, v in row['physical'].items() if k != 'replans'})


if __name__ == '__main__':
    main()
