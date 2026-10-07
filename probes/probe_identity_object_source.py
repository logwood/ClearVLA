"""Split the production B source objective on exact-replay physical endpoints.

Body labels only partition already-computed losses. They never reach the model,
source encoder, correspondence construction, or optimizer. No parameter updates.
This is an evaluation-mode objective audit, not masked-training gradient credit.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from clearvla.benchmarks.calvin_eval import calvin_policy_observation
from clearvla.data.history_clock import VISUAL_OFFSETS
from clearvla.mainline.identity_supervision import IdentityCorrespondence
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.history import CausalHistory
from probe_identity_source_dependence import capture_source, controls, dump


def main():
    parser=argparse.ArgumentParser()
    for key in ('checkpoint','plan','labels','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if VISUAL_OFFSETS!=(-8,-4,0):raise ValueError('label exporter expects the actual past4/current visual pair')
    if not (args.labels/'complete.json').exists():raise ValueError('factual label replay is incomplete')
    rows=json.loads(args.plan.read_text())
    factual=json.loads((args.labels/'results.json').read_text())
    label_index={(r['case'],r['step']):r for r in factual}
    requested=[(r,step) for r in rows for step in r['steps']]
    if len(requested)<2 or len({Path(r['case']).name for r,step in requested})!=len(requested):
        raise ValueError('use one window per distinct case for explicit other-case donors')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    policy=ClearVLACheckpointPolicy(args.checkpoint,device=torch.device('cuda:0'),
        t5_condition=Path('/data/senwang/data/calvin/language/abc_d_full_t5_xxl_bank.pt'),
        dinov3_model=Path('/data/senwang/clearvla/third_party/dinov3/hf-vitb16-lvd1689m'),seed=0)
    model=policy.bundle.model
    if model.config.top.identity_supervision_mode not in ('rgbd_temporal_v1','rgbd_temporal_conditional_v2'):
        raise ValueError('object source audit requires a B identity objective')
    versions={name:p._version for name,p in model.named_parameters()}
    grounder=model.grounding.grounder;captured=[];holder={}
    original_ground=grounder.forward;original_encode=model.encode_online

    def ground(*a,**k):
        result=original_ground(*a,**k);captured.append(result[0]);return result

    def encode(*a,**k):
        result=original_encode(*a,**k);holder['cache']=result[0];return result

    grounder.forward=ground;model.encode_online=encode
    prepared=[];records=[]
    identity=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=policy.bundle.checkpoint_sha256,
        checkpoint_source=policy.bundle.identity.git_commit,global_step=policy.bundle.global_step,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        controls_script_sha256=hashlib.sha256(Path(controls.__code__.co_filename).read_bytes()).hexdigest(),
        label_replay=json.loads((args.labels/'complete.json').read_text()),
        scope='frozen evaluation-mode production objective, exact-replay endpoint body partitions; no training-mask/gradient attribution; bilinear DINO features can straddle boundaries',
        mask_use='report only; not a model input or loss weight',formal_promoted=False)
    dump(args.output/'identity.json',identity)
    try:
        for row,step in requested:
            case=Path(row['case']);entry=label_index[(case.name,step)]
            label_path=Path(entry['production_labels'])
            if hashlib.sha256(label_path.read_bytes()).hexdigest()!=entry['production_labels_sha256']:
                raise ValueError('factual correspondence artifact changed')
            with np.load(label_path,allow_pickle=False) as z:
                arrays={key:z[key] for key in z.files}
            if arrays['source_frames'].tolist()!=[max(step-4,0),step]:raise ValueError('source label clock mismatch')
            label_fields={key:torch.from_numpy(arrays[key][None]).to(policy.device) for key in
                ('cross_source','cross_target','cross_valid','temporal_source','temporal_target','temporal_valid','source_frames')}
            labels=IdentityCorrespondence(**label_fields)
            regions={key:torch.from_numpy(arrays[key][None]).to(policy.device) for key in
                ('cross_source_body','cross_target_body','temporal_source_body','temporal_target_body')}
            regions['object_names']=arrays['object_names'].tolist()
            with np.load(case/'trajectory.npz',allow_pickle=False) as z:
                data={key:z[key] for key in ('rgb_static','rgb_gripper','robot_obs','executed')}
            policy.reset();history=CausalHistory(executed_world=True)
            for t in range(step+1):
                previous=np.zeros(7,np.float32) if t==0 else data['executed'][t-1]
                observation=calvin_policy_observation({'rgb_obs':{key:data[key][t] for key in ('rgb_static','rgb_gripper')},'robot_obs':data['robot_obs'][t]},previous)
                if t==0:history.reset(observation,reset_action=previous)
                else:history.append(previous,observation)
                if t not in (0,step):
                    if t%8==0:model.outlet_adapter.sample_noise(1,device=policy.device,dtype=torch.float32,generator=policy._generator)
                    continue
                captured.clear();_,online=policy.act_with_input(history.snapshot(),row['instruction'])
            cache=holder['cache']
            facts=next(f for f in captured if f.camera_coordinates is cache.top.belief.camera_coordinates)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=model.config.runtime.compute_dtype=='bf16',cache_enabled=False):
                source=capture_source(model,online,facts,labels)
            prepared.append(dict(case=case.name,step=step,online=online,facts=facts,labels=labels,regions=regions,source=source))
            print('PREPARED',case.name,step,flush=True)
        for index,current in enumerate(prepared):
            donor=prepared[(index+1)%len(prepared)]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=model.config.runtime.compute_dtype=='bf16',cache_enabled=False):
                result=controls(model,current['online'],current['facts'],current['labels'],prepared=current['source'],donor=donor['source'],audit_regions=current['regions'])
            result.update(case=current['case'],step=current['step'],donor_case=donor['case'],donor_step=donor['step'])
            left=current['online'].observation.dino_history[:,-2:].float()
            right=donor['online'].observation.dino_history[:,-2:].float()
            result['donor_raw_dino_difference_rms_by_time_camera']=(left-right).square().mean((-2,-1)).sqrt().cpu().tolist()
            result['donor_caveat']='Different case is not a different object identity; shared backgrounds and similar object poses can make donors similar. Inspect source differences and physical partitions.'
            records.append(result)
            dump(args.output/'results.json',dict(identity=identity,complete=False,records=records))
        if any(p._version!=versions[name] for name,p in model.named_parameters()):
            raise ValueError('read-only audit changed parameters')
        dump(args.output/'results.json',dict(identity=identity,complete=True,records=records))
    finally:
        grounder.forward=original_ground;model.encode_online=original_encode


if __name__=='__main__':main()
