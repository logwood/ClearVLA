"""Explicitly authorized scheduler takeover; preserve the active control recorder.

This only supports the known control-eval18 boundary of the bounded spatial pilot.
It terminates the scheduler PID alone, never its process group or recorder child.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().split(') ', 1)[1].split()
        return {'state': stat[0], 'ppid': int(stat[1]), 'start_ticks': stat[19],
                'argv': (path / 'cmdline').read_bytes().decode().split('\0')[:-1]}
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scheduler-pid', type=int, required=True)
    p.add_argument('--control-pid', type=int, required=True)
    p.add_argument('--candidate-gpu', required=True)
    p.add_argument('--minimum-free-mib', type=int, default=18000)
    a = p.parse_args()
    r = a.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    status_path = r / 'pipeline-status.json'
    receipt = r / 'report/scheduler-takeover.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    old = process(a.scheduler_pid)
    child = process(a.control_pid)
    if old is None or 'scripts/run_maniskill_spatial_repair.py' not in old['argv'] or str(r) not in old['argv']:
        raise RuntimeError('Scheduler identity changed')
    if child is None or child['ppid'] != a.scheduler_pid or 'scripts/record_maniskill_npz.py' not in child['argv']:
        raise RuntimeError('Active recorder identity changed')
    free = int(subprocess.check_output(['nvidia-smi', '-i', a.candidate_gpu,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < a.minimum_free_mib:
        raise RuntimeError(f'Insufficient free candidate GPU memory: {free} MiB')
    for stage in ['candidate-heldout', 'candidate-placements', 'candidate-eval18']:
        if (r / stage).exists() or (r / (stage + '.log')).exists():
            raise FileExistsError(stage)

    # Freeze only the waiting scheduler before inspecting the atomic state again.
    os.kill(a.scheduler_pid, signal.SIGSTOP)
    takeover_done = False
    try:
        state = json.loads(status_path.read_text())
        if state['status'] != 'running' or state['active_stage'] != 'control-eval18':
            raise RuntimeError('Scheduler passed the supported takeover boundary')
        row = state['steps'][-1]
        if row['name'] != 'control-eval18' or row['pid'] != a.control_pid or 'returncode' in row:
            raise RuntimeError('Control recorder state changed')
        if process(a.scheduler_pid)['start_ticks'] != old['start_ticks']:
            raise RuntimeError('Scheduler PID was reused')
        receipt.parent.mkdir(parents=True, exist_ok=True)
        transfer = {'authorized_by': 'User: Run the panels concurrently',
            'time_unix': time.time(), 'old_scheduler_pid': a.scheduler_pid,
            'adopted_control_pid': a.control_pid, 'control_start_ticks': child['start_ticks'],
            'new_scheduler_pid': os.getpid(), 'candidate_gpu': a.candidate_gpu,
            'candidate_free_mib': free, 'original_pipeline_state': state,
            'contract': 'Same final checkpoints, seeds 5000000..5000017, seed 0, replan 8, 400 steps; active recorder untouched'}
        receipt.write_text(json.dumps(transfer, indent=2) + '\n')
        os.kill(a.scheduler_pid, signal.SIGKILL)
        takeover_done = True
    finally:
        if not takeover_done:
            os.kill(a.scheduler_pid, signal.SIGCONT)

    lock = threading.Lock()
    active = {'control-eval18'}

    def save():
        state['active_stage'] = ','.join(sorted(active)) or None
        state['active_stages'] = sorted(active)
        state['scheduler_takeover'] = str(receipt)
        temp = status_path.with_suffix('.parallel.tmp')
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(status_path)

    def validate_panel(directory):
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['complete'] and manifest['requested_episodes'] == 18
        assert manifest['replan_steps'] == 8 and manifest['max_episode_steps'] == 400
        assert manifest['policy_seed'] == 0 and manifest['execution_policy'] == 'soft'
        assert [x['seed'] for x in manifest['episodes']] == list(range(5000000, 5000018))
        assert all(x['steps'] == 400 and x['video_frames'] == 401 for x in manifest['episodes'])
        return {'complete': True, 'episodes': 18, 'success_count': manifest['success_count']}

    def adopt_control():
        while True:
            now = process(a.control_pid)
            if now is None or now['state'] == 'Z' or now['start_ticks'] != child['start_ticks']:
                break
            time.sleep(5)
        validation = validate_panel(r / 'control-eval18')
        with lock:
            row['returncode'] = None  # Not our child: do not fabricate a process exit code.
            row['completion_validation'] = validation
            row['finished_unix'] = time.time()
            active.remove('control-eval18')
            save()

    def candidate():
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = a.candidate_gpu
        env['PYTHONPATH'] = str(repo)
        py = sys.executable
        data = '/data/senwang/data/clearvla_sim/stackcube_bc_20260910_v2'
        dino = '/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'
        checkpoint = str(r / 'candidate-pilot/checkpoints/latest.pt')
        commands = [(f'candidate-{mode}', [py, '-B', '-u', 'scripts/probe_maniskill_spatial_grounding.py',
            mode, '--checkpoint', checkpoint, '--data', data, '--dino', dino,
            '--output', str(r / ('candidate-' + mode))]) for mode in ['heldout', 'placements']]
        commands.append(('candidate-eval18', [py, '-B', '-u', 'scripts/record_maniskill_npz.py',
            '--checkpoint', checkpoint, '--output-dir', str(r / 'candidate-eval18'),
            '--episodes', '18', '--eval-seed', '5000000', '--policy-seed', '0',
            '--max-episode-steps', '400', '--replan-steps', '8',
            '--t5-condition', data + '/language/t5_xxl.pt', '--dinov3-model', dino]))
        for name, command in commands:
            log = r / (name + '.log')
            entry = {'name': name, 'started_unix': time.time(), 'command': command,
                'log': str(log), 'gpu_uuid': a.candidate_gpu}
            with lock:
                active.add(name)
                state['steps'].append(entry)
                save()
            with log.open('x') as stream:
                proc = subprocess.Popen(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
                with lock:
                    entry['pid'] = proc.pid
                    save()
                result = proc.wait()
            with lock:
                entry['returncode'] = result
                entry['finished_unix'] = time.time()
                active.remove(name)
                save()
            if result:
                raise RuntimeError(name + ' failed; inspect its log')
        validate_panel(r / 'candidate-eval18')

    try:
        with lock:
            save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(adopt_control), pool.submit(candidate)]
            for task in tasks:
                task.result()
        with lock:
            state['status'] = 'complete'
            save()
    except BaseException as error:
        with lock:
            state['status'] = 'failed'
            state['error'] = repr(error)
            save()
        raise


if __name__ == '__main__':
    main()
