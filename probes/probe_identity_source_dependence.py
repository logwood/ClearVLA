"""Read-only source-dependence controls for the actual B identity objective.

Uses distinct validation episodes and the production correspondence labels.
No simulator object labels, updates, or tensor dumps. These controls detect
source/slot shortcuts; they do not replace factual physical-object audits.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from clearvla.mainline.config import config_from_mapping
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.grounding import _coordinate_basis
from clearvla.mainline.training import identity as production
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def capture_source(model, online, facts, labels):
    """Capture the independent source encoding made by the actual objective."""
    capture = []
    original = production.encode_owners

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        capture.append(result)
        return result

    production.encode_owners = record
    try:
        actual = production.identity_terms(model, online, facts, labels)
    finally:
        production.encode_owners = original
    if len(capture) != 1:
        raise ValueError('production source-only encoder call count changed')
    return actual, capture[0]


def controls(model, online, facts, labels, *, prepared=None, donor=None, audit_regions=None):
    module = model.grounding.grounder
    actual, captured = prepared if prepared is not None else capture_source(model, online, facts, labels)
    raw = online.observation.dino_history[:, -2:].detach()
    b, times, cameras, cells, width = raw.shape
    side = math.isqrt(cells)
    k = module.objects
    with torch.autocast(device_type=raw.device.type, enabled=False):
        observed = model.observation.compiler.encoder.teacher_norm(raw.float())
    state, log_owner = captured
    state = state.reshape(b, times, cameras, k, module.hidden)
    law = log_owner.exp().reshape(b, times, cameras, side, side, k + 1).permute(0, 1, 2, 5, 3, 4)
    conditional = torch.softmax(log_owner[..., :k].float(), -1).reshape(b, times, cameras, side, side, k).permute(0, 1, 2, 5, 3, 4)
    if donor is not None:
        donor_state, donor_owner = donor[1]
        if donor_state.shape != captured[0].shape or donor_owner.shape != log_owner.shape:
            raise ValueError('independent donor source chart differs')
        donor_state = donor_state.reshape_as(state)
        donor_joint = donor_owner.exp().reshape(b, times, cameras, side, side, k + 1).permute(0, 1, 2, 5, 3, 4)
        donor_conditional = torch.softmax(donor_owner[..., :k].float(), -1).reshape(b, times, cameras, side, side, k).permute(0, 1, 2, 5, 3, 4)
    else:
        donor_state = state.roll(1, 0)
        donor_joint = law.roll(1, 0)
        donor_conditional = conditional.roll(1, 0)
    target = observed[:, -1].reshape(b, cameras, side, side, width).permute(0, 1, 4, 2, 3)
    full = facts.image_ownership
    conditional_mode = model.config.top.identity_supervision_mode == 'rgbd_temporal_conditional_v2'
    comparison_source = conditional if conditional_mode else law
    if conditional_mode:
        source = facts.current_image_source
        comparison_target, full_support = production.conditional_real_law(source.log_measure, source.supported)
        perm = torch.arange(k, device=raw.device).roll(1)
    else:
        comparison_target = full
        perm = torch.cat((torch.arange(k, device=raw.device).roll(1), torch.tensor([k], device=raw.device)))
    records = []
    sample = production.sample
    js = production.js_divergence
    for kind, source_time in (('cross', 1), ('temporal', 0)):
        for camera in range(cameras):
            destination = 1 - camera if kind == 'cross' else camera
            valid = getattr(labels, kind + '_valid')[:, camera]
            source_xy = torch.where(valid[..., None], getattr(labels, kind + '_source')[:, camera], 0.)
            target_xy = torch.where(valid[..., None], getattr(labels, kind + '_target')[:, camera], 0.)
            count = valid.sum()

            def mean(value, admitted=valid):
                return torch.where(admitted, value, 0.).sum() / admitted.sum().clamp_min(1)

            joint_source = sample(law[:, source_time, camera], source_xy)
            joint_target = sample(full[:, :, destination], target_xy)
            source_law = sample(comparison_source[:, source_time, camera], source_xy)
            target_law = sample(comparison_target[:, :, destination], target_xy)
            identity_valid = valid
            if conditional_mode:
                identity_valid = identity_valid & (sample(full_support[:, :, destination].float(), target_xy)[..., 0] > 1 - 1e-6)
            divergence = js(source_law, target_law)
            if kind == 'cross':
                divergence = (divergence + js(sample(comparison_target[:, :, camera], source_xy), target_law)) / 2
                if conditional_mode:
                    identity_valid = identity_valid & (sample(full_support[:, :, camera].float(), source_xy)[..., 0] > 1 - 1e-6)
            q = sample(conditional[:, source_time, camera], source_xy)
            latent = state[:, source_time, camera]
            value, background = module.canonical_decoder(latent)
            position = module.decode_position(_coordinate_basis(target_xy.to(value), 16))
            truth = sample(target[:, destination], target_xy).detach()

            def predict(weights, prototypes):
                return torch.einsum('bnk,bkd->bnd', weights.to(prototypes), prototypes[:, :, destination]) + position + background[destination][None, None]

            prediction = predict(q, value)
            zero_value, _ = module.canonical_decoder(torch.zeros_like(latent))
            donor_value, _ = module.canonical_decoder(donor_state[:, source_time, camera])
            donor_q = sample(donor_conditional[:, source_time, camera], source_xy)
            variants = {
                'actual': prediction,
                'identity_value_zero_q_retained': predict(q, zero_value),
                'whole_source_neutral': predict(torch.full_like(q, 1 / k), zero_value),
                'other_episode_whole_source': predict(donor_q, donor_value),
                'source_values_K_permuted': predict(q, value.roll(1, 1)),
                'common_K_permutation': predict(q.roll(1, -1), value.roll(1, 1)),
            }
            errors = {name: float(mean((v.float() - truth).square().mean(-1))) for name, v in variants.items()}
            permutation_error = float((prediction - variants['common_K_permutation']).abs().max())
            base_js = mean(js(source_law, target_law), identity_valid)
            permuted_js = mean(js(source_law[..., perm], target_law[..., perm]), identity_valid)
            if permutation_error > 2e-2 or float((base_js - permuted_js).abs()) > 1e-5:
                raise ValueError('common K relabeling failed: intervention does not preserve identity algebra')
            records.append(dict(
                kind=kind, source_camera=camera, target_camera=destination, admitted_pairs=int(count),
                identity_supported_pairs=int(identity_valid.sum()),
                production_divergence=float(mean(divergence, identity_valid)), source_target_js=float(base_js),
                target_K_permuted_js=float(mean(js(source_law, target_law[..., perm]), identity_valid)),
                other_episode_source_js=float(mean(js(sample((donor_conditional if conditional_mode else donor_joint)[:, source_time, camera], source_xy), target_law), identity_valid)),
                source_null_mass=float(mean(joint_source[..., -1])), target_null_mass=float(mean(joint_target[..., -1])),
                source_real_entropy=float(mean(-(q * q.clamp_min(1e-12).log()).sum(-1))),
                mse=errors, common_K_prediction_max_error=permutation_error,
                common_K_js_max_error=float((base_js - permuted_js).abs()),
            ))
            if audit_regions is not None:
                body = audit_regions[kind + '_source_body'][:, camera]
                other_body = audit_regions[kind + '_target_body'][:, camera]
                if body.shape != valid.shape or other_body.shape != valid.shape:
                    raise ValueError('audit labels do not index the same admitted pairs')
                regions = {'source_objects': body >= 0, 'source_nonobjects': body < 0}
                regions.update({name: body == index for index, name in enumerate(audit_regions['object_names'])})
                item = {}
                for name, region in regions.items():
                    admitted = valid & region
                    supported = identity_valid & region
                    n = int(admitted.sum())
                    ns = int(supported.sum())
                    error = (prediction.float() - truth).square().mean(-1)
                    item[name] = dict(
                        admitted_pairs=n, identity_supported_pairs=ns,
                        same_body_pairs=int((admitted & (body >= 0) & (body == other_body)).sum()),
                        mse={key: float(mean((value.float() - truth).square().mean(-1), admitted)) if n else None for key, value in variants.items()},
                        source_prediction_group_contribution=float(torch.where(admitted, error, 0.).sum() / count.clamp_min(1)),
                        identity_group_contribution=float(torch.where(supported, divergence, 0.).sum() / identity_valid.sum().clamp_min(1)),
                        source_null_mass=float(mean(joint_source[..., -1], admitted)) if n else None,
                        target_null_mass=float(mean(joint_target[..., -1], admitted)) if n else None,
                        source_target_js=float(mean(js(source_law, target_law), supported)) if ns else None,
                    )
                for field, total in [('source_prediction_group_contribution', errors['actual']), ('identity_group_contribution', float(mean(divergence, identity_valid)))]:
                    gap=abs(item['source_objects'][field]+item['source_nonobjects'][field]-total)
                    if gap>2e-6:raise ValueError('object/nonobject loss partition failed')
                records[-1]['audit_regions']=item
    reproduced = {
        'identity_correspondence': sum(x['production_divergence'] for x in records) / 4,
        'identity_source_prediction': sum(x['mse']['actual'] for x in records) / 4,
        'identity_source_removed_prediction_mse': sum(x['mse']['identity_value_zero_q_retained'] for x in records) / 4,
    }
    gaps = {name: abs(value - float(actual[name])) for name, value in reproduced.items()}
    if max(gaps.values()) > 2e-5:
        raise ValueError('probe did not reproduce actual production objective: ' + str(gaps))
    return dict(identity_supervision_mode=model.config.top.identity_supervision_mode,
                production={k: float(v) for k, v in actual.items()}, reproduction_gaps=gaps, pairs=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--windows-per-episode', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.episodes < args.batch_size or args.episodes % args.batch_size:
        raise ValueError('use whole batches with distinct donor episodes')
    if args.batch_size < 2 or args.windows_per_episode < 1:
        raise ValueError('whole-source shuffle needs at least two distinct episodes')
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0')
    payload = torch.load(args.checkpoint, map_location='meta', weights_only=False)
    config = config_from_mapping(payload['config'])
    if config.top.identity_supervision_mode not in {'rgbd_temporal_v1', 'rgbd_temporal_conditional_v2'}:
        raise ValueError('probe requires the trained B objective')
    deployment = load_deployment_checkpoint(args.checkpoint, device=device, t5_condition=config.data.t5_condition)
    model = deployment.model.eval()
    versions = {name: p._version for name, p in model.named_parameters()}
    data = load_mainline_data(config)
    dataset = data.datasets['val']
    groups = defaultdict(list)
    for index, ref in enumerate(dataset.base.refs):
        groups[int(ref.episode_idx)].append(index)
    available = sorted(groups)
    if len(available) < args.episodes:
        raise ValueError('validation contains too few distinct episodes')
    chosen = [available[i] for i in np.linspace(0, len(available) - 1, args.episodes, dtype=int)]
    indices = []
    for fraction in np.linspace(.25, .75, args.windows_per_episode):
        indices.extend(groups[episode][int((len(groups[episode]) - 1) * fraction)] for episode in chosen)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    vision = OnlineVisionPipeline.from_config(config, device)
    identity = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=deployment.checkpoint_sha256,
        checkpoint_source=deployment.identity.git_commit, global_step=deployment.global_step,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        split='val', episode_indices=chosen, sample_indices=indices,
        scope='held observed source dependence; no physical-object labels, no optimizer updates',
        identity_supervision_mode=config.top.identity_supervision_mode,
        caveat='v1 joint consistency admits null collapse; v2 conditional consistency still admits uniform/single-K collapse; physical-object audits remain required',
    )
    records = []
    dump(args.output / 'identity.json', identity)
    for batch_index, worker in enumerate(loader):
        episodes = worker['episode_idx'].tolist()
        if len(set(episodes)) != len(episodes):
            raise ValueError('whole-source donors are not distinct episodes')
        batch = to_training_batch(worker, goal=data.goal, config=config, device=device, visual_encoder=vision)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.runtime.compute_dtype == 'bf16', cache_enabled=False):
            _, state, _ = model.encode_online(batch.online)
            report = controls(model, batch.online, state.top.facts, batch.identity)
        report.update(batch=batch_index, episodes=episodes, source_frames=batch.identity.source_frames.cpu().tolist())
        records.append(report)
        dump(args.output / 'results.json', dict(identity=identity, complete=False, records=records))
        print('SOURCE_CONTROLS', batch_index, report['production'], flush=True)
        del batch, state
    if any(p._version != versions[name] for name, p in model.named_parameters()):
        raise RuntimeError('read-only probe modified model weights')
    dump(args.output / 'results.json', dict(identity=identity, complete=True, records=records))


if __name__ == '__main__':
    main()
