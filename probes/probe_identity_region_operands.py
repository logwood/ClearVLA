"""Read actual v3 region operands on fixed, declared BS8 samples.

This is an encoder/identity-objective audit, not a training update or a full
action-loss VJP. CPU FP32 is an explicit diagnostic option; it does not replace
the separate production-BF16/full-loss GPU VJP.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.mainline.training import identity, identity_regions
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkpoint', 'plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--batches', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    if args.batches < 1:
        raise ValueError('at least one real BS8 operand batch is required')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location='meta', weights_only=False)
    config = config_from_mapping(payload['config'])
    if config.top.identity_supervision_mode != 'rgbd_temporal_regions_v3':
        raise ValueError('only the actual trained v3 objective is audited')
    manifest = Path(config.data.identity_region_manifest)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config.data.identity_region_manifest_sha256:
        raise ValueError('training supervision identity changed')
    plan = json.loads(args.plan.read_text())
    exposure = [row for row in plan['exposure'] if row['split'] == 'train'
                and row['trained_or_scored']][:args.batches]
    if len(exposure) != args.batches or any(len(row['sample_indices']) != 8 for row in exposure):
        raise ValueError('requested complete declared BS8 batches are unavailable')
    deployment = load_deployment_checkpoint(args.checkpoint, device=device, t5_condition=config.data.t5_condition)
    model = deployment.model.train()
    model.set_training_step(int(payload['global_step']))
    versions = {name: p._version for name, p in model.named_parameters()}
    bundle = load_mainline_data(config)
    dataset = bundle.datasets['train']
    indices = [i for row in exposure for i in row['sample_indices']]
    loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
    vision = OnlineVisionPipeline.from_config(config, device)
    original = identity_regions.region_separation
    captures = []
    records = []

    def capture(law, group, different, *, normalized_js_margin=.1):
        out = original(law, group, different, normalized_js_margin=normalized_js_margin)
        per_sample = []
        for p, ids, adj in zip(law.float(), group, different):
            good = ids >= 0
            alive, inverse = torch.unique(ids[good], sorted=True, return_inverse=True)
            values = p.new_zeros((len(alive), p.shape[-1]))
            if len(alive):
                values = values.index_add(0, inverse, p[good])
                values = values / torch.bincount(inverse, minlength=len(alive)).to(p)[:, None]
            edges = torch.triu(adj[alive[:, None], alive[None, :]], diagonal=1).nonzero()
            normalized = identity.js_divergence(values[edges[:, 0]], values[edges[:, 1]]) / math.log(2.)
            hinge = torch.relu(1 - normalized / normalized_js_margin)
            per_sample.append(dict(supported_points=int(good.sum()), groups=alive.cpu().tolist(),
                region_law=values.cpu().tolist(), edges=edges.cpu().tolist(),
                normalized_js=normalized.cpu().tolist(), hinge=hinge.cpu().tolist(),
                edge_count=len(edges), active_edges=int((hinge > 0).sum()),
                sample_loss=float(hinge.mean()) if len(edges) else 0.))
        reconstructed = sum(x['sample_loss'] for x in per_sample) / len(per_sample)
        error = abs(float(out['loss']) - reconstructed)
        if error > 3e-7 or sum(x['edge_count'] for x in per_sample) != int(out['supported_pairs']):
            raise ValueError('production region reduction could not be reproduced')
        captures.append(dict(call=len(captures), camera=len(captures) // 2,
            branch='source_only' if len(captures) % 2 == 0 else 'actual_online',
            loss=float(out['loss']), reproduction_error=error, samples=per_sample))
        return out

    identity_regions.region_separation = capture
    start = time.monotonic()
    runtime = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    meta = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=deployment.checkpoint_sha256,
        checkpoint_source=payload['identity']['git_commit'], global_step=payload['global_step'],
        runtime_source=runtime, config_digest=payload['identity']['config_digest'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=str(device), compute_dtype='fp32' if device.type == 'cpu' else config.runtime.compute_dtype,
        batch_size=8, optimizer_updates=0, parameter_vjp=False,
        scope=__doc__, physical_object_labels_used=False)
    dump(args.output / 'identity.json', meta)
    try:
        for number, worker in enumerate(loader):
            batch = to_training_batch(worker, goal=bundle.goal, config=config, device=device, visual_encoder=vision)
            sample_ids = batch.audit.sample_index.cpu().tolist()
            print('ENCODED_INPUT', number + 1, str(batch.online.observation.dino_history.dtype), flush=True)
            if sample_ids != exposure[number]['sample_indices']:
                raise ValueError('actual sample order differs from the declared training batch')
            conditions = {}
            for masked in (True, False):
                captures.clear()
                generator = torch.Generator(device=device).manual_seed(11012 + number)
                torch.manual_seed(11012 + number)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                     enabled=device.type == 'cuda' and config.runtime.compute_dtype == 'bf16', cache_enabled=False):
                    _, state, _ = model.encode_online(batch.online, training_mask=masked,
                        collect_diagnostics=False, condition_generator=generator)
                    terms = identity.identity_terms(model, batch.online, state.top.facts, batch.identity)
                if len(captures) != 4:
                    raise ValueError('source/current-camera region axis changed')
                reconstructed = sum(call['loss'] for call in captures) / 4
                if abs(reconstructed - float(terms['identity_region_separation'])) > 3e-7:
                    raise ValueError('full separation reduction differs')
                conditions['masked' if masked else 'full_observed'] = dict(
                    training_mask=masked, calls=list(captures),
                    raw_separation=float(terms['identity_region_separation']),
                    source_pairs=sum(s['edge_count'] for c in captures if c['branch']=='source_only' for s in c['samples']),
                    online_pairs=sum(s['edge_count'] for c in captures if c['branch']=='actual_online' for s in c['samples']),
                    returned_pair_count=float(terms['identity_region_supported_negative_pairs']))
            if any(p._version != versions[name] or p.grad is not None for name, p in model.named_parameters()):
                raise RuntimeError('read-only region audit altered parameters')
            records.append(dict(batch=number + 1, sample_indices=sample_ids, conditions=conditions))
            dump(args.output / 'results.json', dict(complete=False, records=records, identity=meta))
            print('REGION_OPERANDS', number + 1,
                  {k: {x: v[x] for x in ('source_pairs','online_pairs','raw_separation')}
                   for k,v in conditions.items()}, flush=True)
        dump(args.output / 'results.json', dict(complete=True, records=records, identity=meta,
            elapsed_seconds=time.monotonic()-start, parameters_unchanged=True))
    finally:
        identity_regions.region_separation = original


if __name__ == '__main__':
    main()
