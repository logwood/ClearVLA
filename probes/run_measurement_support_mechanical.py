"""Two real BS8 updates and saved-checkpoint admission for a support-only repair.

The outer qualification runner schedules an idle GPU. This entry preserves the
ordinary migration, training, backward, optimizer and offline validation paths.
It is mechanical evidence only, never permission for a formal long run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time

import torch
from clearvla.mainline.config import load_config


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--init-checkpoint', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--expected-source', required=True)
    args = p.parse_args()
    config = load_config(args.config)
    out = Path(config.data.output_dir)
    if out.exists() or args.result.exists():
        raise FileExistsError('mechanical artifacts already exist')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if actual != args.expected_source or subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('runtime is not the pinned clean source')
    if (config.top.observation_measurement_mode != 'source_consistent_v2' or config.optimizer.batch_size != 8
            or config.optimizer.update_origin != 11012 or config.runtime.max_train_batches != 2
            or config.runtime.max_val_batches != 2 or config.optimizer.epochs != 1):
        raise ValueError('mechanical exposure or support mode changed')
    command = [sys.executable, '-B', '-u', '-m', 'clearvla.mainline.train', '--config', str(args.config),
        '--init-checkpoint', str(args.init_checkpoint), '--init-model-contract-migration', 'causal_identity_ab_v1',
        '--init-training-clock', 'checkpoint', '--device', 'cuda:0']
    subprocess.run(command, check=True)
    checkpoint = out / 'checkpoints/best.pt'
    saved = torch.load(checkpoint, map_location='meta', weights_only=False)
    if (saved['epoch'] != 1 or saved['global_step'] != 11014 or saved['identity']['git_commit'] != actual
            or saved['identity']['config_digest'] != config.digest()):
        raise ValueError('saved mechanical identity differs')
    records = []
    for line in (out / 'metrics.jsonl').read_text().splitlines():
        value = json.loads(line)
        metrics = value.get('metrics', {})
        if any(isinstance(x, float) and not math.isfinite(x) for x in metrics.values()):
            raise ValueError('nonfinite ordinary train/validation metric')
        if value.get('kind') in {'train', 'epoch'}:
            records.append({k: value.get(k) for k in ('kind', 'phase', 'step')} | {'metrics': {
                k: v for k, v in metrics.items() if k.startswith(('loss_contrib_', 'gradient_window_', 'runtime_window_'))
                or k in {'loss_total', 'loss_contribution_gap', 'runtime_cuda_peak_allocated_gib'}}})
    if not records or not any(r['kind'] == 'epoch' and r['phase'] == 'validation' for r in records):
        raise ValueError('ordinary validation record missing')
    report = dict(complete=True, records=records, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        epoch=1, global_step=11014, source=actual, config_digest=config.digest(),
        optimizer_updates=2, batch_size=8, formal_promoted=False,
        scope='Ordinary training and saved payload admission; cold deployment and behavior remain separate')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    out.with_name(out.name + '-status.json').write_text(json.dumps(dict(
        status='training_and_offline_finished', returncode=0, unix_time=time.time())) + '\n')


if __name__ == '__main__':
    main()
