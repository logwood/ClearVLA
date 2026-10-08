"""Continue a fully annotated short on an idle GPU, then run its existing audits.

Only placement changes. Fixed production source, config, exposure, annotations,
panel and probe commands remain pinned. No formal run is launched here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    r = json.loads(args.receipt.read_text())
    source, root = Path(r['source']), Path(r['experiment_root'])
    config_path = Path(r['config'])
    config = json.loads(config_path.read_text())
    run = Path(config['data']['output_dir'])
    if (config['optimizer']['batch_size'] != 8 or config['optimizer']['epochs'] != 1
            or config['optimizer']['update_origin'] != 11012
            or config['runtime']['max_train_batches'] != 1024 or config['runtime']['max_val_batches'] != 256
            or config['top']['identity_supervision_mode'] != 'rgbd_temporal_regions_v3'):
        raise ValueError('short exposure or model selection changed')
    if run != root / r['run_name'] or run.exists() or (root / (run.name + '-job.json')).exists():
        raise FileExistsError('short run already exists or has another output')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != r['source_commit']
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()):
        raise ValueError('production source is not its pinned clean checkout')
    for file, expected in r['file_sha256'].items():
        if sha(file) != expected:
            raise ValueError('pinned scheduling input changed: ' + file)
    manifest = json.loads(Path(config['data']['identity_region_manifest']).read_text())
    plan = json.loads(Path(r['plan']).read_text())
    if (manifest.get('complete') is not True or manifest.get('supervision_only') is not True
            or manifest['plan_sha256'] != sha(r['plan'])
            or sha(config['data']['identity_region_manifest']) != config['data']['identity_region_manifest_sha256']):
        raise ValueError('complete annotation identity differs')
    expected = {(x['source_split'], *x['source_frames']): x for x in plan['records']}
    actual = {(x['source_split'], *x['source_frames']): x for x in manifest['records']}
    if len(actual) != len(manifest['records']) or set(actual) != set(expected):
        raise ValueError('annotation coverage differs from declared exposure')
    for key, row in actual.items():
        if row['raw_sha256'] != expected[key]['sha256'] or sha(row['labels_path']) != row['labels_sha256']:
            raise ValueError('annotation bytes or source identity changed')
    if args.validate_only:
        print(json.dumps(dict(validated=True, annotations=len(actual), config_sha256=sha(config_path),
                              source=r['source_commit'], formal_promoted=False)), flush=True)
        return
    out = Path(r['output'])
    out.mkdir(exist_ok=False)
    deadline = time.monotonic() + 48 * 3600

    def status(state, **fields):
        dump(out / 'status.json', dict(state=state, unix_time=time.time(), formal_promoted=False, **fields))

    def choose_gpu(stage):
        status('waiting_idle_gpu', stage=stage)
        while time.monotonic() < deadline:
            reserved = set()
            for item in r.get('active_reservations', []):
                p = Path(item['status'])
                if p.exists() and json.loads(p.read_text()).get('state') not in {'complete', 'failed'}:
                    reserved.add(item['gpu'])
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                                            '--format=csv,noheader,nounits'], text=True).splitlines()
            for row in rows:
                index, uuid, used, total = [x.strip() for x in row.split(',')]
                if uuid in r['allowed_gpus'] and uuid not in reserved and int(used) < 512 and int(total) >= 24000:
                    placement = dict(stage=stage, gpu=uuid, egl=index, unix_time=time.time())
                    dump(out / (stage + '-placement.json'), placement)
                    return placement
            time.sleep(30)
        raise TimeoutError('no idle BS8-capable GPU before scheduling deadline')

    def execute(stage, command, gpu, *, log_path=None):
        env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=gpu['gpu'],
                   EGL_VISIBLE_DEVICES=gpu['egl'], OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (log_path or out / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            inventory = dict(pid=child.pid, command=command, source=r['source_commit'], gpu=gpu['gpu'],
                             egl=gpu['egl'], created=time.time(), config_sha256=sha(config_path), formal_promoted=False)
            dump(out / (stage + '-job.json'), inventory)
            if stage == 'training_short':
                inventory['annotation_manifest_sha256'] = config['data']['identity_region_manifest_sha256']
                dump(root / (run.name + '-job.json'), inventory)
            status('running', stage=stage, pid=child.pid, gpu=gpu['gpu'], egl=gpu['egl'])
            code = child.wait()
        if stage == 'training_short':
            dump(root / (run.name + '-status.json'), dict(status='training_and_offline_finished' if code == 0 else 'failed',
                                                        returncode=code, unix_time=time.time()))
        if code:
            raise RuntimeError('stage failed; preserve its log: ' + stage)

    try:
        gpu = choose_gpu('training_short')
        command = [r['python'], '-B', '-u', '-m', 'clearvla.mainline.train', '--config', str(config_path),
                   '--init-checkpoint', r['init_checkpoint'], '--init-model-contract-migration', 'causal_identity_ab_v1',
                   '--init-training-clock', 'checkpoint', '--device', 'cuda:0']
        execute('training_short', command, gpu, log_path=root / (run.name + '.log'))
        # Existing panel verifies the actual final payload, source, BS8 and clock.
        gpu = choose_gpu('standard18')
        command = [x.replace('{gpu}', gpu['gpu']).replace('{egl}', gpu['egl']) for x in r['panel_command']]
        execute('standard18', command, gpu)
        panel = json.loads(Path(r['panel_summary']).read_text())
        if panel.get('complete') is not True or panel.get('trials') != 18 or panel.get('errors'):
            raise ValueError('standard panel incomplete; do not run dependent qualification')
        # Serial ownership of our GPU stages prevents the pending audits from
        # competing for the card at the train/panel handoff. Every audit stays.
        for item in r['qualifications']:
            stage = item['name']
            gpu = choose_gpu(stage)
            receipt = json.loads(Path(item['template']).read_text())
            receipt.update(gpu=gpu['gpu'], egl=gpu['egl'], output=item['output'])
            receipt['scope'] = receipt.get('scope', '') + '; placement selected by ready short continuation'
            resolved = Path(item['resolved_receipt'])
            if resolved.exists():
                raise FileExistsError(resolved)
            dump(resolved, receipt)
            execute(stage, [r['python'], '-B', '-u', item['runner'], '--receipt', str(resolved)], gpu)
            if json.loads((Path(item['output']) / 'status.json').read_text()).get('state') != 'complete':
                raise ValueError('qualification did not complete: ' + stage)
        status('complete', scope='One meaningful short, its standard18 and all original audits finished; no formal promotion')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
