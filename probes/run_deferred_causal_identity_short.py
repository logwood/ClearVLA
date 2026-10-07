"""Run one declared qualification short after existing evidence releases its GPU.

This cannot start a formal run or promote a candidate. All inputs and the
standalone helper are pinned in the external experiment receipt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    status_path = Path(receipt['queue_status'])
    if status_path.exists():
        raise FileExistsError('queue already has a status; preserve and inspect it')
    deadline = time.monotonic() + 48 * 3600

    def status(state, **extra):
        temporary = status_path.with_suffix('.tmp')
        value = dict(state=state, unix_time=time.time(), formal_promoted=False, **extra)
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(value), flush=True)

    def wait_json(path):
        while not Path(path).exists():
            if time.monotonic() > deadline:
                raise TimeoutError('missing dependency: ' + str(path))
            time.sleep(30)
        return json.loads(Path(path).read_text())

    try:
        for path, digest in receipt['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('pinned short input changed: ' + path)
        repo = Path(receipt['repo'])
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != receipt['source']:
            raise ValueError('short source differs')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
            raise ValueError('short checkout is dirty')
        config = json.loads(Path(receipt['config']).read_text())
        if config['optimizer']['batch_size'] != 8 or config['optimizer']['epochs'] != 1 or config['runtime']['max_train_batches'] != 1024 or config['runtime']['max_val_batches'] != 256:
            raise ValueError('this launcher is only for the declared 1024-update BS8 qualification')
        if config['top']['identity_supervision_mode'] != 'rgbd_temporal_conditional_v2':
            raise ValueError('queued short must use the repaired correspondence objective')
        if Path(config['data']['output_dir']).exists():
            raise FileExistsError('short output already exists')
        status('waiting_mechanical_qualification')
        result = wait_json(receipt['mechanical_qualification'])
        if result.get('state') != 'complete':
            # The runner writes progress before completion, so wait for a terminal state.
            while result.get('state') not in {'complete', 'failed'}:
                if time.monotonic() > deadline:
                    raise TimeoutError('mechanical qualification did not finish')
                time.sleep(30)
                result = json.loads(Path(receipt['mechanical_qualification']).read_text())
        if result.get('state') != 'complete':
            raise RuntimeError('mechanical qualification failed')
        status('waiting_existing_standard_panel')
        panel = wait_json(receipt['previous_panel'])
        expected = receipt['previous_panel_contract']
        if any(panel.get(k) != v for k, v in expected.items()) or panel.get('errors'):
            raise ValueError('existing standard panel has not completed with its declared identity')
        status('waiting_gpu', gpu=receipt['gpu'])
        while int(subprocess.check_output(['nvidia-smi', '-i', receipt['gpu'], '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)) >= 512:
            if time.monotonic() > deadline:
                raise TimeoutError('GPU not released; preserve existing jobs')
            time.sleep(30)
        for path, digest in receipt['sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('pinned short input changed while waiting: ' + path)
        status('running_short', source=receipt['source'])
        with Path(receipt['log']).open('x') as log:
            run = subprocess.run(['bash', receipt['helper']], cwd=repo, env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=12 * 3600)
        if run.returncode:
            raise RuntimeError('short failed; original log retained')
        completion = wait_json(receipt['training_status'])
        if completion.get('status') != 'training_and_offline_finished' or completion.get('returncode') != 0:
            raise RuntimeError('short/offline completion contract failed')
        status('short_and_offline_complete', note='Standard closed loop and evidence review still required; no formal promotion')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
