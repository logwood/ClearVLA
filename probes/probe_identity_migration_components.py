"""Decompose v1 -> v2 identity supervision on actual fixed checkpoint evidence.

Real declared BS8 samples, explicit training masking, no model/optimizer update.
This differentiates final source/online K+null logits only. It is not a parameter
VJP, an Adam step, an object-identity oracle, or a closed-loop causal rescue.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import math
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.training import identity as production
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def describe(grad, dim):
    real = grad.narrow(dim, 0, grad.shape[dim] - 1)
    contrast = real - real.mean(dim=dim, keepdim=True)
    null = grad.select(dim, grad.shape[dim] - 1)
    return dict(l2=float(grad.norm()), real_contrast_l2=float(contrast.norm()),
                uniform_null_logit_derivative=float(null.sum()),
                null_logit_l2=float(null.norm()))


def cosine(a, b):
    den = a.double().norm() * b.double().norm()
    return float((a.double() * b.double()).sum() / den) if den > 0 else None


def decompose(batch, facts, source_log, k):
    b, t, c, n, _ = batch.online.observation.dino_history[:, -2:].shape
    side = math.isqrt(n)
    source = source_log.detach().float().requires_grad_()
    law = source.softmax(-1).reshape(b, t, c, side, side, k + 1).permute(0, 1, 2, 5, 3, 4)
    cond = source[..., :k].softmax(-1).reshape(b, t, c, side, side, k).permute(0, 1, 2, 5, 3, 4)
    support = facts.current_image_source.supported.any(1, keepdim=True)
    original = facts.image_ownership.detach().float()
    safe = torch.where(original > 0, original, 1.).log()
    current = safe.detach().requires_grad_()
    null_only = torch.cat((torch.zeros_like(original[:, :k]), torch.ones_like(original[:, k:])), 1)
    online = torch.where(support, current.softmax(1), null_only)
    conditional = torch.where(support, current[:, :k].softmax(1), 0.)
    sample, js = production.sample, production.js_divergence
    values = {name: [] for name in (
        'v1_joint_all', 'joint_supported_old_denominator', 'joint_supported',
        'conditional_supported_old_denominator', 'v2_conditional_supported')}
    records = []
    for kind, when in (('cross', 1), ('temporal', 0)):
        for camera in range(c):
            destination = 1 - camera if kind == 'cross' else camera
            valid = getattr(batch.identity, kind + '_valid')[:, camera]
            x = torch.where(valid[..., None], getattr(batch.identity, kind + '_source')[:, camera], 0.)
            y = torch.where(valid[..., None], getattr(batch.identity, kind + '_target')[:, camera], 0.)
            complete = valid & (sample(support[:, :, destination].float(), y)[..., 0] > 1 - 1e-6)
            if kind == 'cross':
                complete = complete & (sample(support[:, :, camera].float(), x)[..., 0] > 1 - 1e-6)
            joint = js(sample(law[:, when, camera], x), sample(online[:, :, destination], y))
            real = js(sample(cond[:, when, camera], x), sample(conditional[:, :, destination], y))
            if kind == 'cross':
                joint = (joint + js(sample(online[:, :, camera], x), sample(online[:, :, destination], y))) / 2
                real = (real + js(sample(conditional[:, :, camera], x), sample(conditional[:, :, destination], y))) / 2
            count, accepted = valid.sum(), complete.sum()
            def reduce(value, keep, denominator):
                return torch.where(keep, value, 0.).sum() / denominator.clamp_min(1)
            row = {
                'v1_joint_all': reduce(joint, valid, count),
                'joint_supported_old_denominator': reduce(joint, complete, count),
                'joint_supported': reduce(joint, complete, accepted),
                'conditional_supported_old_denominator': reduce(real, complete, count),
                'v2_conditional_supported': reduce(real, complete, accepted)}
            for name, value in row.items():
                values[name].append(value)
            records.append(dict(kind=kind, camera=camera, original_pairs=int(count),
                                supported_pairs=int(accepted), retained_fraction=float(accepted / count.clamp_min(1)),
                                denominator_multiplier=float(count / accepted.clamp_min(1)),
                                raw_losses={name: float(value.detach()) for name, value in row.items()}))
    objectives = {name: torch.stack(parts).mean() for name, parts in values.items()}
    grads, reports = {}, {}
    for index, (name, value) in enumerate(objectives.items()):
        g = torch.autograd.grad(value, (source, current), retain_graph=index < len(objectives) - 1)
        if not all(bool(torch.isfinite(x).all()) for x in g):
            raise ValueError('nonfinite local-logit derivative')
        grads[name] = tuple(x.detach() for x in g)
        reports[name] = dict(raw_loss=float(value.detach()),
                             source_logits=describe(g[0], -1), online_logits=describe(g[1], 1))
    changes = {}
    for left, right in (
        ('v1_joint_all', 'joint_supported_old_denominator'),
        ('joint_supported_old_denominator', 'joint_supported'),
        ('joint_supported', 'v2_conditional_supported'),
        ('v1_joint_all', 'v2_conditional_supported')):
        changes[left + ' -> ' + right] = {
            branch: dict(cosine=cosine(grads[left][i], grads[right][i]),
                         l2_ratio=float(grads[right][i].norm() / grads[left][i].norm().clamp_min(1e-30)),
                         delta_l2=float((grads[right][i] - grads[left][i]).norm()))
            for i, branch in enumerate(('source_logits', 'online_logits'))}
    return dict(components=reports, pair_support=records, changes=changes,
                producer_support=float(support.float().mean()),
                online_joint_reconstruction_max_abs=float((online.detach() - original).abs().max()))


def main():
    parser = argparse.ArgumentParser()
    for name in ('checkpoint', 'plan', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batches', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    payload = torch.load(args.checkpoint, map_location='meta', weights_only=False)
    config = config_from_mapping(payload['config'])
    if config.top.identity_supervision_mode not in {'rgbd_temporal_v1', 'rgbd_temporal_conditional_v2'}:
        raise ValueError('requires v1/v2 checkpoint, not a region objective or a new architecture')
    mode = config.top.identity_supervision_mode
    device = torch.device('cpu')
    deployment = load_deployment_checkpoint(args.checkpoint, device=device, t5_condition=config.data.t5_condition)
    model = deployment.model.train()
    model.set_training_step(int(payload['global_step']))
    versions = {n: p._version for n, p in model.named_parameters()}
    bundle = load_mainline_data(config)
    plan = json.loads(args.plan.read_text())
    exposure = [r for r in plan['exposure'] if r['split'] == 'train' and r['trained_or_scored']][:args.batches]
    if len(exposure) != args.batches or any(len(r['sample_indices']) != 8 for r in exposure):
        raise ValueError('not the declared common BS8 exposure')
    loader = DataLoader(Subset(bundle.datasets['train'], [i for r in exposure for i in r['sample_indices']]),
                        batch_size=8, num_workers=0, shuffle=False)
    vision = OnlineVisionPipeline.from_config(config, device)
    start = time.monotonic()
    records = []
    original = production.encode_owners
    for number, worker in enumerate(loader):
        batch = to_training_batch(worker, goal=bundle.goal, config=config, device=device, visual_encoder=vision)
        ids = batch.audit.sample_index.cpu().tolist()
        if ids != exposure[number]['sample_indices']:
            raise ValueError('training sample identity changed')
        generator = torch.Generator(device=device).manual_seed(11012 + number)
        torch.manual_seed(11012 + number)
        print('INPUT_READY', number + 1, flush=True)
        captured = []
        def capture(*a, **kw):
            value = original(*a, **kw)
            captured.append(value)
            return value
        with torch.no_grad():
            _, state, _ = model.encode_online(batch.online, training_mask=True, collect_diagnostics=False,
                                              condition_generator=generator)
            production.encode_owners = capture
            try:
                actual = production.identity_terms(model, batch.online, state.top.facts, batch.identity)
            finally:
                production.encode_owners = original
        if len(captured) != 1:
            raise ValueError('independent source call count changed')
        result = decompose(batch, state.top.facts, captured[0][1], model.grounding.grounder.objects)
        reproduced = {}
        with torch.no_grad():
            production.encode_owners = lambda *a, **kw: captured[0]
            try:
                for version, key in (('rgbd_temporal_v1', 'v1_joint_all'),
                                     ('rgbd_temporal_conditional_v2', 'v2_conditional_supported')):
                    proxy = SimpleNamespace(grounding=model.grounding, observation=model.observation,
                        config=SimpleNamespace(top=SimpleNamespace(identity_supervision_mode=version)))
                    value = production.identity_terms(proxy, batch.online, state.top.facts, batch.identity)
                    gap = abs(float(value['identity_correspondence']) - result['components'][key]['raw_loss'])
                    if gap > 3e-6:
                        raise ValueError('actual production objective reproduction failed: ' + str(gap))
                    reproduced[version] = dict(loss=float(value['identity_correspondence']), gap=gap,
                        source_prediction=float(value['identity_source_prediction']))
            finally:
                production.encode_owners = original
        if abs(reproduced[mode]['loss'] - float(actual['identity_correspondence'])) > 1e-7:
            raise ValueError('saved-mode production differs')
        if reproduced['rgbd_temporal_v1']['source_prediction'] != reproduced['rgbd_temporal_conditional_v2']['source_prediction']:
            raise ValueError('unmodified source prediction differs')
        if any(p._version != versions[n] or p.grad is not None for n, p in model.named_parameters()):
            raise RuntimeError('read-only probe altered parameters')
        result.update(batch=number+1, sample_indices=ids, production_reproduction=reproduced)
        records.append(result)
        meta = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=deployment.checkpoint_sha256,
            checkpoint_source=payload['identity']['git_commit'], global_step=payload['global_step'],
            runtime_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope=__doc__, batch_size=8, training_mask=True, device='cpu', compute_dtype='fp32',
            parameter_vjp=False, optimizer_updates=0, parameters_unchanged=True)
        dump(args.output/'results.json',dict(complete=False,identity=meta,records=records))
        print('MIGRATION_COMPONENTS',number+1,result['changes']['v1_joint_all -> v2_conditional_supported'],flush=True)
    dump(args.output/'results.json',dict(complete=True,identity=meta,records=records,elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
