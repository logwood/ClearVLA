"""Audit the actual observed-feature kernel against independent RGB flow pairs.

Correspondences and body labels only score an already-produced measurement.
They never enter policy inference. Bilinear source-row evaluation preserves the
native measurement grid; descriptor resampling is not substituted for its law.
This is a correlated four-window audit, not general match calibration accuracy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from clearvla.vision.observed_correspondence import observed_feature_correspondence
from clearvla.vision.entity_chart import current_image_grid


def array(value):
    return value.detach().float().cpu().numpy()


def stats(value):
    value = np.asarray(value, dtype=float)
    return dict(n=int(value.size), mean=float(value.mean()), median=float(np.median(value)),
                p10=float(np.percentile(value, 10)), p90=float(np.percentile(value, 90)),
                minimum=float(value.min()), maximum=float(value.max())) if value.size else None


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def localization_controls(law):
    """Change only real-match spread, preserving each row's unknown mass.

    These diagnostic laws never enter policy inference and are not calibrated
    alternatives. They isolate averaging of candidate positions from abstention.
    """
    real = law[..., :-1]
    mass = real.sum(-1, keepdim=True)
    controls = {}
    for count in (1, 5):
        values, indices = real.topk(min(count, real.shape[-1]), dim=-1)
        selected = torch.zeros_like(real).scatter(-1, indices, values)
        selected = selected / selected.sum(-1, keepdim=True).clamp_min(1e-30) * mass
        candidate = torch.cat((selected, law[..., -1:]), -1)
        torch.testing.assert_close(candidate[..., -1], law[..., -1], rtol=0., atol=0.)
        torch.testing.assert_close(candidate.sum(-1), law.sum(-1), rtol=0., atol=2e-6)
        controls['top'+str(count)+'_same_null'] = candidate
    return controls


def score_camera(law, h, w, camera, labels, image_shape):
    device = law.device
    xy = torch.from_numpy(labels['temporal_source'][camera]).to(device)
    channel_law = law[:,camera].permute(0,2,1).reshape(1,h*w+1,h,w)
    sampled = F.grid_sample(channel_law,xy[None,:,None],mode='bilinear',align_corners=True)[0,:,:,0].T
    torch.testing.assert_close(sampled.sum(-1),torch.ones_like(sampled[:,0]),rtol=0.,atol=2e-6)
    grid = current_image_grid(h,w,device=device).reshape(h*w,2)
    real = sampled[:,:-1].sum(-1)
    endpoint = (sampled[:,:-1]@grid)/real[:,None].clamp_min(1e-30)
    truth = torch.from_numpy(labels['temporal_target'][camera]).to(device)
    pixels = torch.tensor([(image_shape[1]-1)/2,(image_shape[0]-1)/2],device=device)
    expected_delta = (sampled[:,:-1]@grid-real[:,None]*xy)*pixels
    true_delta = (truth-xy)*pixels
    endpoint_error = torch.linalg.vector_norm((endpoint-truth)*pixels,dim=-1)
    distance = torch.linalg.vector_norm(true_delta,dim=-1)
    valid = labels['temporal_valid'][camera]
    source_body = labels['temporal_source_body'][camera]
    target_body = labels['temporal_target_body'][camera]
    same_object = (source_body>=0)&(source_body==target_body)
    regions = dict(all_accepted=valid,nonobject=valid&(source_body<0),same_object=valid&same_object)
    for obj,name in enumerate(labels['object_names']):
        regions[str(name)] = valid&same_object&(source_body==obj)
    metrics = {}
    for name,selected in regions.items():
        ids = torch.from_numpy(selected).to(device)
        metrics[name] = dict(count=int(ids.sum()),real_mass=stats(array(real[ids])),
            conditional_endpoint_error_pixels=stats(array(endpoint_error[ids])),
            true_displacement_pixels=stats(array(distance[ids])),
            unconditional_displacement_pixels=stats(array(torch.linalg.vector_norm(expected_delta[ids],dim=-1))),
            unconditional_displacement_error_pixels=stats(array(torch.linalg.vector_norm(expected_delta[ids]-true_delta[ids],dim=-1))))
    return dict(camera=camera,source_body_mismatch_count=int((valid&(source_body>=0)&~same_object).sum()),regions=metrics)


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkpoint', 'plan', 'labels', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--native-control', action='store_true', help='Also audit a native-token kernel; never replace the production measurement')
    parser.add_argument('--localization-controls', action='store_true', help='Measure top1/top5 spreading controls at identical row null; never feed them to policy')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('start this audit with deterministic CUBLAS workspace')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    policy = ClearVLACheckpointPolicy(args.checkpoint, device=torch.device('cuda:0'),
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'), seed=0)
    model = policy.bundle.model
    if model.config.top.observation_measurement_mode != 'source_consistent_v1':
        raise ValueError('this audit requires the actual source-consistent measurement')
    teacher = model.training_targets.teacher
    original = teacher.measure_observations
    captured = []

    def measure(**kwargs):
        result = original(**kwargs)
        captured.append((kwargs, result))
        return result

    teacher.measure_observations = measure
    versions = {name: p._version for name, p in model.named_parameters()}
    labels = json.loads((args.labels/'results.json').read_text())
    index = {(r['case'], r['step']): r for r in labels}
    report = dict(identity=dict(checkpoint=str(args.checkpoint), checkpoint_sha256=policy.bundle.checkpoint_sha256,
        checkpoint_source=policy.bundle.identity.git_commit, measurement_mode=model.config.top.observation_measurement_mode,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope=__doc__,
        label_complete=json.loads((args.labels/'complete.json').read_text())), records=[], complete=False)
    try:
        for row in json.loads(args.plan.read_text()):
            case = Path(row['case'])
            with np.load(case/'trajectory.npz', allow_pickle=False) as z:
                data = {key: z[key] for key in ('rgb_static','rgb_gripper','robot_obs','executed')}
            policy.reset()
            history = CausalHistory(executed_world=True)
            for t in range(max(row['steps'])+1):
                previous = np.zeros(7, np.float32) if t == 0 else data['executed'][t-1]
                obs = calvin_policy_observation({'rgb_obs': {k: data[k][t] for k in ('rgb_static','rgb_gripper')}, 'robot_obs': data['robot_obs'][t]}, previous)
                if t == 0:
                    history.reset(obs, reset_action=previous)
                else:
                    history.append(previous, obs)
                if t not in row['steps']:
                    if t == 0:
                        policy.act_with_input(history.snapshot(), row['instruction'])
                    elif t % 8 == 0:
                        model.outlet_adapter.sample_noise(1, device=policy.device, dtype=torch.float32, generator=policy._generator)
                    continue
                entry = index[(case.name, t)]
                path = Path(entry['production_labels'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry['production_labels_sha256']:
                    raise ValueError('independent label bytes differ from their exact-replay receipt')
                with np.load(path, allow_pickle=False) as z:
                    labels_now = {key:z[key] for key in z.files}
                if labels_now['source_frames'].tolist() != [t-4,t]:
                    raise ValueError('independent temporal labels do not use the observed four-step window')
                captured.clear()
                _, online = policy.act_with_input(history.snapshot(), row['instruction'])
                if len(captured) != 1:
                    raise ValueError('expected exactly one executed observed-world measurement')
                kw, produced = captured[0]
                facts, observations = kw['facts'], kw['observations']
                if observations.shape[1] != 1 or not torch.all(kw['relative_offsets'] == 4):
                    raise ValueError('production observed clock differs from the independent past4 label')
                stored_source = facts.dense_chart.dino_content
                raw = stored_source.float()
                source_mask = facts.dense_chart.cell_observed[...,0]
                b,c,h,w,d = raw.shape
                source = raw.flatten(2,3)
                target = observations[:,0].float().flatten(2,3)
                # W uses the production pooled/normalized measurement chart,
                # not the 16x16 native chart used by instruction-start matching.
                past_grid = model.observation.observation_supports(online.observation.dino_history[:,-2:-1])[:,0]
                now_grid = model.observation.observation_supports(online.observation.dino_history[:,-1:])[:,0]
                torch.testing.assert_close(raw, past_grid.to(stored_source.dtype).float(), rtol=0., atol=0.)
                torch.testing.assert_close(observations[:,0].float(), now_grid.float(), rtol=0., atol=0.)
                if kw['observed'] is not None and not torch.all(kw['observed']):
                    raise ValueError('these independent labels require a fully observed past4 window')
                mask = source_mask.flatten(2)
                with torch.no_grad():
                    law = observed_feature_correspondence(source, target, mask, mask)
                    joint, _ = facts.current_image_source.on_image(rows=h,columns=w).restrict(source_mask[:,None]).normalized((2,3,4))
                    reproduced = torch.einsum('bkcn,bcn->bk',joint.flatten(-2),law[...,-1])
                    torch.testing.assert_close(reproduced, produced.null_probability[:,0,:,0], rtol=0., atol=2e-6)
                record = dict(case_id=row['case_id'],case=case.name,step=t,source_frames=labels_now['source_frames'].tolist(),
                    label_sha256=entry['production_labels_sha256'],production_measurement_chart=[h,w],
                    native_frontend_tokens=int(online.observation.dino_history.shape[-2]),
                    actual_G_measurement_null=array(produced.null_probability[:,0,:,0])[0].tolist(),
                    production_null_reproduction_max=float((reproduced-produced.null_probability[:,0,:,0]).abs().max()),cameras=[])
                for camera,rgb_key in enumerate(('rgb_static','rgb_gripper')):
                    record['cameras'].append(score_camera(law,h,w,camera,labels_now,data[rgb_key].shape[1:3]))
                if args.localization_controls:
                    record['localization_controls'] = {name: [
                        score_camera(candidate,h,w,camera,labels_now,data[key].shape[1:3])
                        for camera,key in enumerate(('rgb_static','rgb_gripper'))]
                        for name,candidate in localization_controls(law).items()}
                if args.native_control:
                    if not torch.all(source_mask):
                        raise ValueError('native diagnostic requires complete observed camera charts')
                    source_native = online.observation.dino_history[:,-2].float()
                    target_native = online.observation.dino_history[:,-1].float()
                    side = int(source_native.shape[-2]**.5)
                    if side*side != source_native.shape[-2]:
                        raise ValueError('native diagnostic chart is not square')
                    support = torch.ones(source_native.shape[:-1],device=policy.device,dtype=torch.bool)
                    native_law = observed_feature_correspondence(source_native,target_native,support,support)
                    record['native_token_control'] = dict(chart=[side,side],scope='alternate measurement only, never fed to the policy',cameras=[
                        score_camera(native_law,side,side,camera,labels_now,data[key].shape[1:3])
                        for camera,key in enumerate(('rgb_static','rgb_gripper'))])
                    if args.localization_controls:
                        record['native_token_control']['localization_controls'] = {name: [
                            score_camera(candidate,side,side,camera,labels_now,data[key].shape[1:3])
                            for camera,key in enumerate(('rgb_static','rgb_gripper'))]
                            for name,candidate in localization_controls(native_law).items()}
                report['records'].append(record)
                dump(args.output/'results.json',report)
                print('OBSERVED_MATCH_CALIBRATION',case.name,t,flush=True)
        if versions != {name:p._version for name,p in model.named_parameters()}:
            raise ValueError('measurement audit changed parameters')
        report.update(complete=True,parameters_unchanged=True)
        dump(args.output/'results.json',report)
    finally:
        teacher.measure_observations = original


if __name__ == '__main__':
    main()
