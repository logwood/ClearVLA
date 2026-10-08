"""Resolve a completed region short's exact annotation config, then audit zero updates.

The separate qualification runner owns waiting, checkpoint/panel admission and
GPU scheduling. This adapter changes only output paths, unused optimizer origin
and one-batch probe limits. It never substitutes common-only identity losses.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys

import torch
from clearvla.mainline.config import config_from_mapping, load_config


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ('config', 'checkpoint', 'probe', 'report'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expected-source', required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    config = load_config(args.config)
    saved = torch.load(args.checkpoint, map_location='meta', weights_only=False)
    payload_config = config_from_mapping(saved['config'])
    if saved['global_step'] != args.expected_step or saved['identity']['git_commit'] != args.expected_source:
        raise ValueError('completed checkpoint source/clock differs from the receipt')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != args.expected_source:
        raise ValueError('VJP runtime must be the immutable training source')
    if config.as_dict() != payload_config.as_dict() or config.digest() != saved['identity']['config_digest']:
        raise ValueError('resolved full training config differs from the actual checkpoint')
    if config.optimizer.batch_size != 8 or config.top.identity_supervision_mode != 'rgbd_temporal_regions_v3':
        raise ValueError('this audit requires the full BS8 region objective')
    manifest = Path(config.data.identity_region_manifest)
    manifest_hash = sha(manifest)
    if manifest_hash != config.data.identity_region_manifest_sha256:
        raise ValueError('completed annotation manifest hash differs')
    annotation = json.loads(manifest.read_text())
    if annotation.get('complete') is not True or annotation.get('supervision_only') is not True:
        raise ValueError('incomplete or non-supervision annotation manifest')
    # A new, deliberately unstepped optimizer has origin zero. The inner VJP
    # probe restores saved execution time before the actual production forward.
    derived = copy.deepcopy(config.as_dict())
    derived['data']['output_dir'] = str(args.report.parent / 'training-entry')
    derived['optimizer']['update_origin'] = 0
    derived['runtime']['max_train_batches'] = 1
    derived['runtime']['max_val_batches'] = 1
    config_from_mapping(derived).validate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    probe_config = args.report.with_name('resolved-full-objective.config.json')
    with probe_config.open('x') as stream:
        stream.write(json.dumps(derived, indent=2) + '\n')
    command = [sys.executable, '-B', '-u', str(args.probe), '--report', str(args.report),
               '--config', str(probe_config), '--init-checkpoint', str(args.checkpoint),
               '--device', 'cuda:0']
    receipt = dict(complete=True, preparation_only=args.prepare_only, optimizer_updates=0,
                   original_config_sha256=sha(args.config), checkpoint_sha256=sha(args.checkpoint),
                   source=args.expected_source, global_step=args.expected_step,
                   original_config_digest=config.digest(), annotation_manifest_sha256=manifest_hash,
                   probe_config_sha256=sha(probe_config), probe_sha256=sha(args.probe),
                   command=command, common_objective_only=False,
                   scope='Full actual region supervision; read-only parameter VJPs, not behavioral qualification')
    with args.report.with_name('resolved-inputs.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    if args.prepare_only:
        return
    subprocess.run(command, check=True)
    result = json.loads(args.report.read_text())
    if (result.get('complete') is not True or result.get('global_step') != args.expected_step
            or result.get('optimizer_updates') != 0 or result.get('training_mask') is not True):
        raise ValueError('full production VJP did not satisfy its read-only contract')


if __name__ == '__main__':
    main()
