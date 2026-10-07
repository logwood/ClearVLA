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
    # Production transports pair differences before aggregating source rows.
    # A sampled endpoint minus the query location has an additional source
    # centroid term when neighboring rows carry different confidence.
    row_real = law[:,camera,:,:-1]
    row_delta = row_real@grid-row_real.sum(-1,keepdim=True)*grid
    pair_delta = F.grid_sample(row_delta.permute(0,2,1).reshape(1,2,h,w),
        xy[None,:,None],mode='bilinear',align_corners=True)[0,:,:,0].T*pixels
    conditional_pair_delta = pair_delta/real[:,None].clamp_min(1e-30)
    endpoint_offset = (sampled[:,:-1]@grid-real[:,None]*xy)*pixels
    true_delta = (truth-xy)*pixels
    endpoint_error = torch.linalg.vector_norm((endpoint-truth)*pixels,dim=-1)
    distance = torch.linalg.vector_norm(true_delta,dim=-1)
    valid = labels['temporal_valid'][camera]
    source_body = labels['temporal_source_body'][camera]
    target_body = labels['temporal_target_body'][camera]
    same_object = (source_body>=0)&(source_body==target_body)
    regions = dict(all_accepted=valid,nonobject=valid&(source_body<0),same_object=valid&same_object)
    regions['same_object_motion_gt1px'] = valid&same_object&(array(distance)>1.)
    regions['same_object_motion_le025px'] = valid&same_object&(array(distance)<=.25)
    for obj,name in enumerate(labels['object_names']):
        regions[str(name)] = valid&same_object&(source_body==obj)
    metrics = {}
    for name,selected in regions.items():
        ids = torch.from_numpy(selected).to(device)
        metrics[name] = dict(count=int(ids.sum()),real_mass=stats(array(real[ids])),
            conditional_endpoint_error_pixels=stats(array(endpoint_error[ids])),
            true_displacement_pixels=stats(array(distance[ids])),
            endpoint_offset_pixels=stats(array(torch.linalg.vector_norm(endpoint_offset[ids],dim=-1))),
            production_pair_delta_pixels=stats(array(torch.linalg.vector_norm(pair_delta[ids],dim=-1))),
            production_pair_delta_error_pixels=stats(array(torch.linalg.vector_norm(pair_delta[ids]-true_delta[ids],dim=-1))),
            conditional_pair_delta_error_pixels=stats(array(torch.linalg.vector_norm(conditional_pair_delta[ids]-true_delta[ids],dim=-1))))
    return dict(camera=camera,source_body_mismatch_count=int((valid&(source_body>=0)&~same_object).sum()),regions=metrics)


def score_existing_flow(flow, confidence, occlusion, labels, camera, image_shape):
    """Score the already computed forward flow; no rerun or policy substitution."""
    h,w=flow.shape[-2:];ih,iw=image_shape
    query=torch.from_numpy(labels['temporal_source'][camera]).to(flow.device).float()[None,:,None]
    scale=torch.tensor([(iw-1)/(w-1),(ih-1)/(h-1)],device=flow.device,dtype=torch.float32)[None]
    def sample(value):
        return F.grid_sample(value.float(),query,mode='bilinear',padding_mode='border',align_corners=True)[0,:,:,0].T
    vector=sample(flow)*scale
    reliability=confidence.float()*(1-occlusion.float())
    # Gate before interpolation. This is a diagnostic use of the existing
    # reliability, not a calibrated null probability or production W law.
    weighted=sample(flow.float()*reliability)*scale
    true=(labels['temporal_target'][camera]-labels['temporal_source'][camera])*np.array([(iw-1)/2,(ih-1)/2])
    true=torch.from_numpy(true).to(flow.device).float()
    source_body=labels['temporal_source_body'][camera]
    valid=labels['temporal_valid'][camera]&(source_body>=0)&(source_body==labels['temporal_target_body'][camera])
    magnitude=array(torch.linalg.vector_norm(true,dim=-1))
    masks=dict(same_object=valid,same_object_motion_gt1px=valid&(magnitude>1),
               same_object_motion_le025px=valid&(magnitude<=.25))
    regions={}
    for name,mask in masks.items():
        ids=torch.from_numpy(mask).to(flow.device)
        regions[name]=dict(count=int(mask.sum()),true_displacement_pixels=stats(magnitude[mask]),
            raw_flow_pixels=stats(array(torch.linalg.vector_norm(vector[ids],dim=-1))),
            raw_error_pixels=stats(array(torch.linalg.vector_norm(vector[ids]-true[ids],dim=-1))),
            reliability_weighted_error_pixels=stats(array(torch.linalg.vector_norm(weighted[ids]-true[ids],dim=-1))),
            reliability=stats(array(sample(reliability)[ids])),confidence=stats(array(sample(confidence)[ids])),
            occlusion=stats(array(sample(occlusion)[ids])))
    return dict(camera=camera,grid=[h,w],regions=regions)


def main():
    parser = argparse.ArgumentParser()
    for key in ('checkpoint', 'plan', 'labels', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--native-control', action='store_true', help='Also audit a native-token kernel; never replace the production measurement')
    parser.add_argument('--localization-controls', action='store_true', help='Measure top1/top5 spreading controls at identical row null; never feed them to policy')
    parser.add_argument('--label-source', choices=('sensor','rigid_audit'), default='sensor',
        help='Audit truth only; rigid_audit is simulator geometry and is prohibited as a training/online input')
    parser.add_argument('--existing-flow',action='store_true',help='Capture the existing SEA-RAFT-style RGB and coarse flow without extra model calls')
    parser.add_argument('--reference-rgb-labels',type=Path,help='Audit-only current-to-start rigid labels; test global frozen-DINO initialization for RGB refinement')
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
    prepared_captured=[]
    original_prepare=model.observation.prepare

    def prepare(*items,**kwargs):
        prepared=original_prepare(*items,**kwargs)
        if args.existing_flow:
            pack=prepared.pack;raw=pack.raw_context
            item=dict(dino=prepared.observation.dino_history.detach(),
                coarse=(pack.patch_flow_forward[:,-1].detach().clone(),
                        pack.flow_confidence[:,-1].detach().clone(),pack.flow_occlusion[:,-1].detach().clone()))
            if raw is not None:
                item['raw']=(raw.flow_forward[:,-1].detach().clone(),
                             raw.confidence[:,-1].detach().clone(),raw.occlusion[:,-1].detach().clone())
            prepared_captured.append(item)
        return prepared

    if args.existing_flow:model.observation.prepare=prepare

    def measure(**kwargs):
        result = original(**kwargs)
        captured.append((kwargs, result))
        return result

    teacher.measure_observations = measure
    versions = {name: p._version for name, p in model.named_parameters()}
    labels = json.loads((args.labels/'results.json').read_text())
    index = {(r['case'], r['step']): r for r in labels}
    reference_labels=None
    if args.reference_rgb_labels:
        complete=json.loads((args.reference_rgb_labels/'complete.json').read_text())
        if complete.get('pair_mode')!='current_start' or complete.get('rigid_simulation_oracle_audit_only') is not True:
            raise ValueError('reference RGB control requires independently replayed current-to-start geometry')
        reference_labels={(r['case'],r['step']):r for r in json.loads((args.reference_rgb_labels/'results.json').read_text())}
    report = dict(identity=dict(checkpoint=str(args.checkpoint), checkpoint_sha256=policy.bundle.checkpoint_sha256,
        checkpoint_source=policy.bundle.identity.git_commit, measurement_mode=model.config.top.observation_measurement_mode,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope=__doc__,
        label_complete=json.loads((args.labels/'complete.json').read_text())), records=[], complete=False)
    report['identity']['label_source'] = args.label_source
    if args.label_source == 'rigid_audit' and report['identity']['label_complete'].get('rigid_simulation_oracle_audit_only') is not True:
        raise ValueError('rigid audit requires its explicit simulator-oracle receipt')
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
                sensor_comparison = []
                if args.label_source == 'rigid_audit':
                    for camera,key in enumerate(('rgb_static','rgb_gripper')):
                        rigid_valid=labels_now['audit_rigid_temporal_valid'][camera]
                        sensor_valid=labels_now['temporal_valid'][camera]
                        common=rigid_valid&sensor_valid
                        shape=data[key].shape[1:3]
                        pixels=np.array([(shape[1]-1)/2,(shape[0]-1)/2])
                        difference=(labels_now['temporal_target'][camera]-labels_now['audit_rigid_temporal_target'][camera])*pixels
                        sensor_comparison.append(dict(camera=camera,rigid_visible_points=int(rigid_valid.sum()),
                            common_sensor_points=int(common.sum()),
                            sensor_endpoint_error_pixels=stats(np.linalg.norm(difference[common],axis=-1)),
                            geometry_roundtrip_max=float(labels_now['audit_rigid_roundtrip_max'][camera])))
                    labels_now=dict(labels_now,
                        temporal_target=labels_now['audit_rigid_temporal_target'],
                        temporal_valid=labels_now['audit_rigid_temporal_valid'],
                        temporal_target_body=labels_now['audit_rigid_temporal_target_body'])
                if labels_now['source_frames'].tolist() != [t-4,t]:
                    raise ValueError('independent temporal labels do not use the observed four-step window')
                captured.clear()
                prepared_captured.clear()
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
                    conditional,_ = facts.current_image_source.on_image(rows=h,columns=w).restrict(source_mask[:,None]).normalized((-2,-1))
                    chart = current_image_grid(h,w,device=policy.device).reshape(h*w,2)
                    pair_delta = law[...,:-1]@chart-law[...,:-1].sum(-1,keepdim=True)*chart
                    reproduced_delta = torch.einsum('bkcn,bcnd->bkcd',conditional.flatten(-2),pair_delta)
                    torch.testing.assert_close(reproduced_delta, produced.transport_per_support[:,0], rtol=0., atol=2e-6)
                record = dict(case_id=row['case_id'],case=case.name,step=t,source_frames=labels_now['source_frames'].tolist(),
                    label_sha256=entry['production_labels_sha256'],production_measurement_chart=[h,w],
                    native_frontend_tokens=int(online.observation.dino_history.shape[-2]),
                    actual_G_measurement_null=array(produced.null_probability[:,0,:,0])[0].tolist(),
                    production_pair_delta_reproduction_max=float((reproduced_delta-produced.transport_per_support[:,0]).abs().max()),
                    production_null_reproduction_max=float((reproduced-produced.null_probability[:,0,:,0]).abs().max()),cameras=[])
                if args.label_source == 'rigid_audit':
                    record['sensor_vs_rigid_audit'] = sensor_comparison
                for camera,rgb_key in enumerate(('rgb_static','rgb_gripper')):
                    record['cameras'].append(score_camera(law,h,w,camera,labels_now,data[rgb_key].shape[1:3]))
                if args.existing_flow:
                    matched=[p for p in prepared_captured if p['dino'].shape==online.observation.dino_history.shape
                             and torch.equal(p['dino'],online.observation.dino_history)]
                    if len(matched)!=1:raise ValueError('existing-flow capture must match exactly one current online observation')
                    p=matched[0]
                    if 'raw' in p:
                        high=p['raw'][0];coarse=p['coarse'][0];hc,wc=coarse.shape[-2:];hh,wh=high.shape[-2:]
                        rebuilt=F.interpolate(high.flatten(0,1).float(),size=(hc,wc),mode='bilinear',align_corners=True)
                        rebuilt=rebuilt.reshape_as(coarse)*((wc-1)/(wh-1))
                        torch.testing.assert_close(rebuilt,coarse.float(),atol=2e-6,rtol=0.)
                        record['existing_flow_coarse_reproduction_max']=float((rebuilt-coarse.float()).abs().max())
                    record['existing_learned_flow']={name:[score_existing_flow(
                        values[0][:,camera],values[1][:,camera],values[2][:,camera],labels_now,camera,data[key].shape[1:3])
                        for camera,key in enumerate(('rgb_static','rgb_gripper'))]
                        for name,values in p.items() if name!='dino'}
                if reference_labels is not None:
                    import cv2
                    from probe_observed_rgb_motion_candidate import estimate as rgb_estimate, score as rgb_score
                    cv2.setNumThreads(1)
                    entry=reference_labels[(case.name,t)];label_path=Path(entry['production_labels'])
                    if hashlib.sha256(label_path.read_bytes()).hexdigest()!=entry['production_labels_sha256']:
                        raise ValueError('reference rigid labels changed')
                    with np.load(label_path,allow_pickle=False) as z:ref_labels={k:z[k] for k in z.files}
                    if ref_labels['source_frames'].tolist()!=[t,0]:raise ValueError('current/reference physical clock mismatch')
                    reference=online.instruction_reference
                    if reference is None or not torch.all(reference.age_steps==t):raise ValueError('reference is not the observed instruction start')
                    if not torch.all(reference.observed):raise ValueError('seed audit requires completely observed reference charts')
                    now_dino=online.observation.dino_history[:,-1].float()
                    start_dino=reference.dino.float()
                    side=int(now_dino.shape[-2]**.5)
                    if side*side!=now_dino.shape[-2]:raise ValueError('seed requires the declared full endpoint chart')
                    grid=current_image_grid(side,side,device=policy.device).reshape(-1,2)
                    def seed(first,second,shape):
                        # A global appearance seed is only an initialization,
                        # never a confidence claim, object label or policy law.
                        similarity=F.normalize(first.float(),dim=-1)@F.normalize(second.float(),dim=-1).T
                        delta=grid[similarity.argmax(-1)]-grid
                        field=delta.T.reshape(1,2,side,side)
                        field=F.interpolate(field,size=shape,mode='bilinear',align_corners=True)[0].permute(1,2,0)
                        scale=torch.tensor([(shape[1]-1)/2,(shape[0]-1)/2],device=field.device)
                        return (field*scale).detach().cpu().numpy().astype(np.float32)
                    record['reference_rgb_candidates']=dict(label_sha256=entry['production_labels_sha256'],
                        opencv_version=cv2.__version__,scope='current/start observed DINO initializes RGB DIS; evaluation only; no policy inputs changed',cameras=[])
                    for camera,key in enumerate(('rgb_static','rgb_gripper')):
                        src,dst=data[key][t],data[key][0];shape=src.shape[:2]
                        seeds=(seed(now_dino[0,camera],start_dino[0,camera],shape),seed(start_dino[0,camera],now_dino[0,camera],shape))
                        flow=rgb_estimate(src,dst,'dis_medium',seeds=seeds)
                        repeated=rgb_estimate(src,dst,'dis_medium',seeds=seeds)
                        if not np.array_equal(flow['xy'],repeated['xy']) or not np.array_equal(flow['accepted'],repeated['accepted']):
                            raise ValueError('DINO-seeded RGB control is not repeatable')
                        scored=rgb_score(flow,ref_labels,camera,shape)
                        scored['seed_motion_pixels']=stats(np.linalg.norm(seeds[0],axis=-1))
                        scored['repeat_max']=0.
                        record['reference_rgb_candidates']['cameras'].append(scored)
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
        model.observation.prepare = original_prepare


if __name__ == '__main__':
    main()
