"""Trace correspondence pressure at supported and masked production cells.

One real BS8 training-mode observation, no updates. Gradients below are with
respect to final source K+null logits, not an estimate of parameter updates.
"""
from pathlib import Path
import argparse
import json
import math

import torch

from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.training import identity as production
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unmasked-control', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = config_from_mapping(torch.load(args.checkpoint, map_location='meta', weights_only=False)['config'])
    bundle = load_deployment_checkpoint(args.checkpoint, device=torch.device('cuda:0'), t5_condition=config.data.t5_condition)
    model = bundle.model.train()
    versions = {n: p._version for n, p in model.named_parameters()}
    data = load_mainline_data(config)
    from torch.utils.data import DataLoader
    worker = next(iter(DataLoader(data.datasets['train'], batch_size=8, shuffle=False, num_workers=0)))
    vision = OnlineVisionPipeline.from_config(config, torch.device('cuda:0'))
    batch = to_training_batch(worker, goal=data.goal, config=config, device=torch.device('cuda:0'), visual_encoder=vision)
    capture = []
    original = production.encode_owners

    def recorded(*args, **kwargs):
        value = original(*args, **kwargs)
        capture.append(value)
        return value

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.runtime.compute_dtype == 'bf16', cache_enabled=False):
        _, state, _ = model.encode_online(batch.online, training_mask=not args.unmasked_control)
        facts = state.top.facts
        supported_fraction = float(facts.current_image_source.supported.float().mean())
        if not args.unmasked_control and supported_fraction >= 1:
            raise ValueError('requested training-mask probe did not apply the producer mask')
        production.encode_owners = recorded
        try:
            actual = production.identity_terms(model, batch.online, facts, batch.identity)
        finally:
            production.encode_owners = original
    if len(capture) != 1:
        raise ValueError('source encoder call count changed')
    b, t, c, n, _ = batch.online.observation.dino_history[:, -2:].shape
    k = model.grounding.grounder.objects
    side = math.isqrt(n)
    logits = capture[0][1].detach().float().requires_grad_()
    law = logits.softmax(-1).reshape(b, t, c, side, side, k + 1).permute(0, 1, 2, 5, 3, 4)
    conditional = logits[..., :k].softmax(-1).reshape(b, t, c, side, side, k).permute(0, 1, 2, 5, 3, 4)
    real, support = production.conditional_real_law(facts.current_image_source.log_measure, facts.current_image_source.supported)
    sample, js = production.sample, production.js_divergence
    records = []
    joint_total, conditional_total = 0., 0.
    for kind, time in (('cross', 1), ('temporal', 0)):
        for camera in range(c):
            target_camera = 1 - camera if kind == 'cross' else camera
            valid = getattr(batch.identity, kind + '_valid')[:, camera]
            x = torch.where(valid[..., None], getattr(batch.identity, kind + '_source')[:, camera], 0.)
            y = torch.where(valid[..., None], getattr(batch.identity, kind + '_target')[:, camera], 0.)
            complete = valid & (sample(support[:, :, target_camera].float(), y)[..., 0] > 1 - 1e-6)
            if kind == 'cross':
                complete = complete & (sample(support[:, :, camera].float(), x)[..., 0] > 1 - 1e-6)
            source, target = sample(law[:, time, camera], x), sample(facts.image_ownership[:, :, target_camera], y)
            conditional_source, conditional_target = sample(conditional[:, time, camera], x), sample(real[:, :, target_camera], y)
            joint = js(source, target)
            conditional_js = js(conditional_source, conditional_target)
            if kind == 'cross':
                joint = .5 * (joint + js(sample(facts.image_ownership[:, :, camera], x), target))
                conditional_js = .5 * (conditional_js + js(sample(real[:, :, camera], x), conditional_target))
            count = valid.sum().clamp_min(1)
            def reduce(value, mask, denominator=count):
                return torch.where(mask, value, 0.).sum() / denominator
            old_supported = reduce(joint, complete)
            old_missing = reduce(joint, valid & ~complete)
            new = reduce(conditional_js, complete, complete.sum().clamp_min(1))
            derivatives = {}
            for name, loss in [('joint_supported', old_supported), ('joint_missing', old_missing), ('conditional_supported', new)]:
                grad, = torch.autograd.grad(loss, logits, retain_graph=True)
                derivatives[name] = dict(loss=float(loss.detach()), source_uniform_null_logit_derivative=float(grad[..., -1].sum()), source_null_logit_gradient_rms=float(grad[..., -1].square().mean().sqrt()))
            joint_total = joint_total + old_supported + old_missing
            conditional_total = conditional_total + new
            records.append(dict(kind=kind, camera=camera, pairs=int(valid.sum()), fully_supported_pairs=int(complete.sum()), source_null_mean=float(reduce(source[..., -1], valid)), target_null_mean=float(reduce(target[..., -1], valid)), derivatives=derivatives))
    calculated = float((conditional_total if config.top.identity_supervision_mode == 'rgbd_temporal_conditional_v2' else joint_total).detach()) / 4
    gap = abs(calculated - float(actual['identity_correspondence']))
    if gap > 2e-5:
        raise ValueError('probe does not reproduce selected production objective: ' + str(gap))
    if any(p._version != versions[n] for n, p in model.named_parameters()):
        raise RuntimeError('read-only probe changed weights')
    report = dict(complete=True, checkpoint=str(args.checkpoint), checkpoint_sha256=bundle.checkpoint_sha256, source=bundle.identity.git_commit, batch_size=b, updates=0, training_mask=not args.unmasked_control, producer_supported_fraction=supported_fraction, mode='train; explicit encode_online training_mask selector', interpretation='negative uniform-null-logit derivative means direct gradient descent pressure to increase source abstention; not the full parameter optimizer update', production={key: float(value) for key, value in actual.items()}, reproduction_gap=gap, records=records)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
