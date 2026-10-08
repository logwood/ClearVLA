"""Resume-safe read-only qualification stages; never launch formal training."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    output = Path(receipt['output'])
    output.mkdir(exist_ok=False)
    deadline = time.monotonic() + 48 * 3600
    completed = []

    def status(state, **fields):
        value = dict(state=state, unix_time=time.time(), completed=completed, formal_promoted=False, **fields)
        dump(output / 'status.json', value)
        print(json.dumps(value), flush=True)

    def wait_path(path):
        path = Path(path)
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('qualification dependency did not finish: ' + str(path))
            time.sleep(30)
        return json.loads(path.read_text())

    try:
        for dependency in receipt.get('prerequisites', []):
            status('waiting_prior_evidence', dependency=dependency['path'])
            value = wait_path(dependency['path'])
            while value.get('state') not in {'complete', 'failed'}:
                if time.monotonic() > deadline:
                    raise TimeoutError('prior evidence did not finish: ' + dependency['path'])
                time.sleep(30)
                value = json.loads(Path(dependency['path']).read_text())
            if value.get('state') != 'complete':
                raise RuntimeError('prior evidence failed; inspect retained logs: ' + dependency['path'])
        repo = Path(receipt['production_source'])
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != receipt['train_commit']:
            raise ValueError('production checkout identity differs')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
            raise ValueError('production checkout is dirty')
        for filename, expected in receipt['script_sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError('pinned qualification script changed: ' + filename)
        status('waiting_training_and_offline')
        import torch
        for run in receipt['runs']:
            finished = wait_path(run['training_status'])
            if finished.get('status') != 'training_and_offline_finished' or finished.get('returncode') != 0:
                raise RuntimeError('training/offline stage failed: ' + run['checkpoint'])
            checkpoint = torch.load(run['checkpoint'], map_location='meta', weights_only=False)
            if checkpoint['epoch'] != 1 or checkpoint['global_step'] != receipt['global_step'] or checkpoint['identity']['git_commit'] != receipt['train_commit']:
                raise ValueError('qualification checkpoint does not match declared run')
            if checkpoint['config']['optimizer']['batch_size'] != 8:
                raise ValueError('qualification checkpoint was not trained at BS8')
        env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES=receipt['gpu'],
                   EGL_VISIBLE_DEVICES=receipt['egl'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if receipt.get('deterministic_operator_control'):
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        for stage in receipt['stages']:
            panel_path = stage.get('wait_panel')
            if panel_path:
                status('waiting_standard_18', stage=stage['name'])
                panel = wait_path(panel_path)
                expected = dict(complete=True, trials=18, expected_trials=18, execute_rows=8,
                                max_steps=360, controller_anchor='stored_target', seed=0,
                                source_commit=receipt['train_commit'], global_step=receipt['global_step'],
                                checkpoint=stage['checkpoint'])
                if any(panel.get(k) != v for k, v in expected.items()) or panel.get('errors'):
                    raise ValueError('full standard panel is incomplete or has another identity')
            status('waiting_gpu', stage=stage['name'], gpu=receipt['gpu'])
            while True:
                memory = subprocess.check_output(['nvidia-smi', '-i', receipt['gpu'], '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
                used,total=(int(v.strip()) for v in memory.split(','))
                # Default remains an exclusive model audit. A declared CPU or
                # small EGL-only replay may instead request bounded free RAM;
                # this never admits a formal model job to a shared card.
                required=stage.get('gpu_free_mib')
                if required is not None and (not isinstance(required,int) or required<4096 or required>total):
                    raise ValueError('invalid explicit audit GPU headroom')
                if (required is None and used<512) or (required is not None and total-used>=required):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('qualification GPU stayed occupied')
                time.sleep(30)
            status('running', stage=stage['name'])
            with (output / (stage['name'] + '.log')).open('x') as log:
                child = subprocess.run(stage['command'], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=8 * 3600)
            if child.returncode:
                raise RuntimeError('qualification stage failed; retained log: ' + stage['name'])
            value = json.loads(Path(stage['result']).read_text())
            if stage['contract'] == 'plan18':
                passed = isinstance(value, list) and len(value) == 18
            elif stage['contract'] == 'results_complete':
                passed = value.get('complete') is True and bool(value.get('records'))
            elif stage['contract'] == 'status_complete':
                passed = value.get('status') == 'complete'
            elif stage['contract'] == 'label_export_complete':
                labels = json.loads(Path(stage['result']).with_name('results.json').read_text())
                passed = value.get('windows') == stage['expected_windows'] == len(labels)
                passed = passed and len({(row['case'], row['step']) for row in labels}) == len(labels)
                for row in labels:
                    path = Path(row['production_labels'])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['production_labels_sha256']:
                        raise ValueError('exported audit label identity differs: ' + str(path))
            else:
                raise ValueError('unknown qualification output contract')
            if not passed:
                raise RuntimeError('qualification output incomplete: ' + stage['name'])
            completed.append(stage['name'])
        status('complete', scope='Evidence collection complete; formal promotion requires review of behavior and source controls')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
