"""Observed RGB/DINO localization versus an oracle bound on confidence alone.

Frozen production encoder/preprocessing and native 16x16 correspondence only.
Rigid replay labels score outputs after matching; they never enter encoding or
the matching law. The bound independently optimizes the four source-row real
masses in [0,1], holding each row's conditional destination distribution fixed.
This deliberately optimistic bound is NOT an admitted matcher or a policy test.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from clearvla.mainline.config import load_config
from clearvla.simulation.vision import preprocess_rgb_history
from clearvla.vision.preprocessing import PreprocessConfig
from clearvla.vision.online_pipeline import OnlineVisionPipeline
from clearvla.vision.observed_correspondence import observed_feature_correspondence
from clearvla.vision.entity_chart import current_image_grid


def stats(values):
    x = np.asarray(values, dtype=float)
    return None if not x.size else dict(n=int(x.size), mean=float(x.mean()),
        median=float(np.median(x)), p90=float(np.percentile(x, 90)), maximum=float(x.max()))


def hull(points):
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(points) < 2:
        return np.asarray(points)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def half(sequence):
        out = []
        for p in sequence:
            while len(out) >= 2 and cross(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def confidence_bound(vectors, truth):
    # The image of [0,1]^4 is the convex hull of its 16 projected vertices.
    corners = np.asarray(list(itertools.product((0., 1.), repeat=len(vectors)))) @ vectors
    poly = hull(corners)
    if len(poly) == 1:
        return float(np.linalg.norm(poly[0]-truth))
    a, b = poly, np.roll(poly, -1, axis=0)
    edges = b-a
    relative = truth-a
    cross = edges[:, 0]*relative[:, 1]-edges[:, 1]*relative[:, 0]
    if len(poly) > 2 and np.all(cross >= -1e-10):
        return 0.
    lengths = (edges*edges).sum(-1)
    t = np.clip((relative*edges).sum(-1)/np.maximum(lengths, 1e-30), 0., 1.)
    return float(np.linalg.norm(a+t[:, None]*edges-truth, axis=-1).min())


def check_bound():
    assert confidence_bound(np.zeros((4, 2)), np.array([3., 4.])) == 5.
    assert confidence_bound(np.array([[1., 0.], [0., 1.]]), np.array([.2, .8])) == 0.
    assert abs(confidence_bound(np.array([[1., 0.], [0., 1.]]), np.array([2., 2.]))-2**.5) < 1e-12
    assert abs(confidence_bound(np.array([[1., 0.], [1., 0.]]), np.array([.5, 2.]))-2.) < 1e-12
    # Independent convex least-squares check, including nearly collinear input.
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(0)
    maximum = 0.
    for _ in range(100):
        v, y = rng.normal(size=(4, 2)), rng.normal(size=2)*3
        solved = lsq_linear(v.T, y, bounds=(0., 1.), tol=1e-13, max_iter=1000)
        error = abs(confidence_bound(v, y)-np.linalg.norm(v.T@solved.x-y))
        maximum = max(maximum, error)
    assert maximum < 1e-5, maximum
    return maximum


def stencil(xy, side):
    pos = (np.clip(xy, -1, 1)+1)*(side-1)/2
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo+1, side-1)
    frac = pos-lo
    ids = np.stack((lo[:, 1]*side+lo[:, 0], lo[:, 1]*side+hi[:, 0],
                    hi[:, 1]*side+lo[:, 0], hi[:, 1]*side+hi[:, 0]), -1)
    weights = np.stack(((1-frac[:, 0])*(1-frac[:, 1]), frac[:, 0]*(1-frac[:, 1]),
                        (1-frac[:, 0])*frac[:, 1], frac[:, 0]*frac[:, 1]), -1)
    return ids, weights


def score(law, labels, camera, shape):
    law = law.detach().double().cpu().numpy()
    side = int((law.shape[0])**.5)
    grid = current_image_grid(side, side, device=torch.device('cpu')).reshape(-1, 2).double().numpy()
    source = labels['temporal_source'][camera].astype(float)
    target = labels['audit_rigid_temporal_target'][camera].astype(float)
    valid = (labels['audit_rigid_temporal_valid'][camera]
             & (labels['temporal_source_body'][camera] >= 0)
             & (labels['temporal_source_body'][camera] == labels['audit_rigid_temporal_target_body'][camera]))
    if not np.isfinite(target[valid]).all() or np.any(np.abs(target[valid]) > 1+1e-6):
        raise ValueError('visible rigid endpoints must be finite and in view')
    target = np.where(valid[:, None], target, source)
    ids, weights = stencil(source, side)
    real = law[:, :-1].sum(-1)
    conditional = law[:, :-1]/np.maximum(real[:, None], 1e-30)
    delta = np.where(real[:, None] > 0, conditional@grid-grid, 0.)
    pixels = np.array([(shape[1]-1)/2, (shape[0]-1)/2])
    truth = (target-source)*pixels
    basis = delta[ids]*weights[:, :, None]*pixels
    production = (basis*real[ids, None]).sum(1)
    all_real = basis.sum(1)
    # Cross-check the actual pre-aggregation pair delta and source interpolation.
    actual = law[:, :-1]@grid-real[:, None]*grid
    rebuilt = (actual[ids]*weights[:, :, None]).sum(1)*pixels
    assert np.max(np.abs(production-rebuilt)) < 1e-10
    sample_law = (law[ids, :-1]*weights[:, :, None]).sum(1)
    sample_real = sample_law.sum(-1)
    best = sample_law.argmax(-1)
    dest_ids, _ = stencil(target, side)
    rank = np.argsort(np.argsort(-sample_law, axis=-1), axis=-1)+1
    best_true_stencil_rank = np.take_along_axis(rank, dest_ids, -1).min(-1)
    stencil_mass = np.array([sample_law[i, np.unique(j)].sum() for i, j in enumerate(dest_ids)])
    conditional_stencil_mass = stencil_mass/np.maximum(sample_real, 1e-30)
    bound = np.full(len(source), np.nan)
    for i in np.flatnonzero(valid):
        bound[i] = confidence_bound(basis[i], truth[i])
    true_motion = np.linalg.norm(truth, axis=-1)
    production_epe = np.linalg.norm(production-truth, axis=-1)
    all_real_epe = np.linalg.norm(all_real-truth, axis=-1)
    if np.any(bound[valid] > production_epe[valid]+1e-8) or np.any(bound[valid] > all_real_epe[valid]+1e-8):
        raise AssertionError('oracle confidence bound worse than a feasible real-mass assignment')
    metrics = dict(real_mass=sample_real, true_motion_pixels=true_motion,
        measured_motion_pixels=np.linalg.norm(production, axis=-1), production_epe_pixels=production_epe,
        all_real_epe_pixels=all_real_epe, oracle_confidence_only_epe_pixels=bound,
        best_cell_epe_pixels=np.linalg.norm((grid[best]-target)*pixels, axis=-1),
        best_true_stencil_rank=best_true_stencil_rank, conditional_truth_stencil_mass=conditional_stencil_mass)
    regions = {}
    for name, mask in dict(visible_object=valid, moving_object_gt1px=valid&(true_motion>1),
                           static_object_le025px=valid&(true_motion<=.25)).items():
        regions[name] = {k: stats(v[mask]) for k, v in metrics.items()}
        regions[name]['count'] = int(mask.sum())
        regions[name]['confidence_bound_gt2px_count'] = int((mask&(bound>2)).sum())
        regions[name]['best_cell_in_true_stencil_count'] = int((mask&(best_true_stencil_rank == 1)).sum())
    occluded = labels['audit_rigid_occluded'][camera] & labels['audit_rigid_source_interior'][camera]
    return dict(regions=regions, occluded_real_mass=stats(sample_real[occluded]),
                displacement_reproduction_max=float(np.abs(production-rebuilt).max()))


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    for key in ('config', 'plan', 'labels', 'output'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('deterministic CUDA control required')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    cfg = load_config(args.config)
    torch.backends.cuda.matmul.allow_tf32 = cfg.runtime.cuda_tf32
    if cfg.data.dinov3_microbatch != 2 or cfg.data.cache_side != 336:
        raise ValueError('standalone grouping proof only covers production two-camera microbatch=2')
    complete = json.loads((args.labels/'complete.json').read_text())
    if not complete.get('rigid_simulation_oracle_audit_only') or complete.get('pair_mode') != 'current_start':
        raise ValueError('requires independently exact-replayed current/start scoring labels')
    label_rows = {(r['case'], r['step']): r for r in json.loads((args.labels/'results.json').read_text())}
    model = OnlineVisionPipeline.from_config(cfg, torch.device('cuda:0'))
    versions = {n: p._version for n, p in model.named_parameters()}
    preprocessing = PreprocessConfig(resize_hw=(336, 336))
    started = time.monotonic()
    report = dict(complete=False, identity=model.identity(), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope=__doc__,
        bound_numerical_check_max=check_bound(), records=[])
    def encode(rgb, times):
        images = preprocess_rgb_history(dict(top=rgb['rgb_static'][times], wrist=rgb['rgb_gripper'][times]), preprocessing)
        value = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous().float().div(255)
        return model(value)
    with torch.no_grad():
        for row in json.loads(args.plan.read_text()):
            case = Path(row['case'])
            with np.load(case/'trajectory.npz', allow_pickle=False) as z:
                rgb = {k: z[k] for k in ('rgb_static', 'rgb_gripper')}
            for step in row['steps']:
                entry = label_rows[(case.name, step)]
                path = Path(entry['production_labels'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['production_labels_sha256']:
                    raise ValueError('rigid labels changed')
                with np.load(path, allow_pickle=False) as z:
                    labels = {k: z[k] for k in z.files}
                if labels['source_frames'].tolist() != [step, 0]:
                    raise ValueError('current/reference observation clocks differ')
                start = encode(rgb, [0, 0, 0])[-1]
                current = encode(rgb, [max(0, step-8), max(0, step-4), step])[-1]
                repeat = encode(rgb, [max(0, step-8), max(0, step-4), step])[-1]
                torch.testing.assert_close(current, repeat, atol=0., rtol=0.)
                support = torch.ones(current.shape[:-1], dtype=torch.bool, device=current.device)
                law = observed_feature_correspondence(current, start, support, support)
                # Independent algebra for the continuous branch; exact rows use
                # production's explicit identity/ambiguity limit, not division by zero.
                a, b = F.normalize(current.float(), dim=-1).double(), F.normalize(start.float(), dim=-1).double()
                distance = (a.square().sum(-1)[..., None]+b.square().sum(-1)[..., None, :]-2*a@b.transpose(-1, -2)).clamp_min(0)
                continuous = (distance > 1e-12).all(-1)
                odds = (.05/distance.clamp_min(1e-12)).square().mean(-1)
                predicted_mass = odds/(1+odds)
                error = float((predicted_mass[continuous]-law[:, :, :-1].double().sum(-1)[continuous]).abs().max())
                assert error < 2e-6
                record = dict(case=case.name, step=step, label_sha256=entry['production_labels_sha256'], repeat_max=0.,
                    native_chart=[16, 16], real_odds_reproduction_max=error,
                    cameras=[score(law[c], labels, c, rgb[key].shape[1:3]) for c, key in enumerate(('rgb_static', 'rgb_gripper'))])
                report['records'].append(record)
                dump(args.output/'results.json', report)
                print('DESCRIPTOR_BOUND', case.name, step, flush=True)
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise AssertionError('frozen encoder changed')
    report.update(complete=True, parameters_unchanged=True, elapsed_seconds=time.monotonic()-started,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
    dump(args.output/'results.json', report)


if __name__ == '__main__':
    main()
