"""Real RGB -> frozen pretrained DINOv3 -> update -> checkpoint -> deployment.

Requires the user's actual data and authorized HF-format weights. There is no
random-backbone fallback. This proves interface execution, NOT policy quality;
the small number of updates is insufficient for robot-task evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from clearvla.mainline.config import load_config
from clearvla.mainline.checkpoint import build_checkpoint_identity
from clearvla.mainline.data.loading import load_mainline_data_for_smoke, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.checkpoints import save_checkpoint, load_checkpoint_exact
from clearvla.mainline.runtime.identity import dataset_identity, language_identity
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.train import _data_state
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import build_optimizer, WarmupCosineSchedule
from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory
from clearvla.vision.online_pipeline import OnlineVisionPipeline


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mainline/dinov3_online_cumulative_calvin.json'))
    p.add_argument('--model',required=True)
    p.add_argument('--revision',default='')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--dtype',choices=('fp32','bf16'),default='bf16')
    p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--compact',action='store_true',help='Explicit H32 contract test; DINO stays ViT-B/16, 256 patches x 768')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.steps<1 or a.batch_size<1:p.error('steps and batch-size must be positive')
    if a.output.exists():p.error('output exists; refusing to overwrite evidence')
    a.output.mkdir(parents=True)
    report={'scope':'real-pretrained-interface-smoke-not-policy-quality','completed':False,
            'compact':a.compact,'torch':torch.__version__,'device':a.device,'stages':[]}
    def stage(name):
        report['stages'].append(name)
        (a.output/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print('[dinov3-integration]',name,flush=True)
    try:
        device=torch.device(a.device)
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        cfg=load_config(a.config)
        cfg=replace(cfg,data=replace(cfg.data,dinov3_model=a.model,dinov3_revision=a.revision,
                    dinov3_local_files_only=not a.allow_download,dinov3_microbatch=a.microbatch,
                    num_workers=0,output_dir=str(a.output)),runtime=replace(cfg.runtime,compute_dtype=a.dtype))
        if a.compact:
            cfg=replace(cfg,dimensions=replace(cfg.dimensions,hidden_size=32,num_heads=4),
                        bottom=replace(cfg.bottom,controller_heads=4))
        cfg.validate()
        (a.output/'resolved_config.json').write_text(json.dumps(cfg.as_dict(),indent=2)+'\n')
        torch.manual_seed(47)
        bundle=load_mainline_data_for_smoke(cfg,split='train',episode_limit=1)
        bundle=replace(bundle,visual_encoder=OnlineVisionPipeline.from_config(cfg,device))
        report['encoder_identity']=bundle.visual_encoder.identity()
        stage('real_data_and_pretrained_encoder_loaded')
        raw=default_collate([bundle.datasets['train'][i] for i in range(min(a.batch_size,len(bundle.datasets['train'])))])
        before=time.perf_counter()
        batch=to_training_batch(raw,goal=bundle.goal,config=cfg,device=device,visual_encoder=bundle.visual_encoder)
        if device.type=='cuda':torch.cuda.synchronize()
        report['encoding_seconds']=time.perf_counter()-before
        report['visual_requests']=bundle.visual_encoder.last_batch_stats
        del raw
        stage('online_rgb_to_typed_training_batch')
        model=ClearVLAMainlinePolicy(cfg).to(device)
        model.configure_action_normalizer(bundle.action_normalizer)
        optimizer,_=build_optimizer(model,cfg)
        schedule=WarmupCosineSchedule(optimizer,warmup_steps=cfg.optimizer.warmup_steps,
            total_steps=22024,minimum_ratio=cfg.optimizer.min_lr_ratio)
        engine=MainlineTrainingEngine(model=model,config=cfg,optimizer=optimizer,schedule=schedule,
            device=device,dtype=torch.bfloat16 if a.dtype=='bf16' else torch.float32,
            train_flow_generator=torch.Generator(device=device).manual_seed(48),
            train_condition_generator=torch.Generator(device=device).manual_seed(49))
        for _ in range(a.steps):engine.train_step(batch)
        if not all(torch.isfinite(v).all() for v in model.parameters()):raise RuntimeError('nonfinite policy parameters')
        if any(v.grad is not None for v in bundle.visual_encoder.parameters()):raise RuntimeError('frozen encoder received gradients')
        stage('training_updates_completed')
        identity=build_checkpoint_identity(cfg,repo_root=Path(__file__).resolve().parents[1],
            dataset=dataset_identity(bundle,cfg),language=language_identity(bundle,cfg))
        checkpoint=a.output/'interface_smoke.pt'
        save_checkpoint(checkpoint,model=model,optimizer=optimizer,schedule=schedule,config=cfg,identity=identity,
            epoch=0,global_step=engine.global_step,best_metric=None,data_state=_data_state(bundle,cfg,identity))
        load_checkpoint_exact(checkpoint,model=model,optimizer=optimizer,schedule=schedule,config=cfg,identity=identity)
        stage('checkpoint_saved_and_exact_resume_validated')
        # Use a single actual source window for native history reconstruction.
        dataset=bundle.datasets['train'].base
        ref=dataset.refs[0]
        episode=bundle.episodes[ref.episode_idx]
        center=int(ref.center)
        anchor=dataset.instruction_starts.get(ref.episode_idx,center)
        if center!=anchor:raise RuntimeError('smoke expects the first real instruction-start window')
        history=CausalHistory(executed_world=cfg.top.world_feedback_mode!='none')
        for t in range(center+1):
            frames=dataset.image_store.load_window(episode,np.array([t],np.int64))
            observation=PolicyObservation(
                rgb={name:frames[name][0].permute(1,2,0).numpy() for name in cfg.data.camera_names},
                state=episode.states_raw[t].copy(), action_state=episode.action_states_raw[t].copy())
            if t==0:history.reset(observation)
            else:history.append(episode.actions_raw[t-1],observation)
        raw_single=default_collate([bundle.datasets['train'][0]])
        single=to_training_batch(raw_single,goal=bundle.goal,config=cfg,device=device,visual_encoder=bundle.visual_encoder)
        model.eval()
        with torch.no_grad():
            sampled=sample_action(model,single.online,cfg,generator=torch.Generator(device=device).manual_seed(50))
            expected=bundle.action_normalizer.decode(sampled.action[0].float().cpu().numpy()).astype(np.float32)
            if sampled.gripper_command is not None:
                expected[:,-1]=sampled.gripper_command[0].float().cpu().numpy()
        del model,optimizer,schedule,engine,batch,raw_single,single,sampled
        bundle=replace(bundle,visual_encoder=None)
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        policy=ClearVLACheckpointPolicy(checkpoint,device=device,seed=50,
            dinov3_model=a.model,dinov3_allow_download=a.allow_download)
        action,_=policy.act_with_input(history.snapshot(),episode.instruction)
        error=float(np.max(np.abs(action-expected)))
        report['deployment_action_max_abs_difference']=error
        # CUDA bf16 reductions can differ by a few ulps between the in-process
        # training sample and a freshly reconstructed deployment adapter.  The
        # checkpoint-owned dtype and microbatch identity are still required;
        # fp32 keeps the exact zero gate.
        parity_tolerance = 0.0 if a.dtype == 'fp32' else 5.0e-5
        report['deployment_action_parity_tolerance']=parity_tolerance
        if error > parity_tolerance:
            raise RuntimeError('training/deployment action parity failed')
        report['health']=policy.deployment_health()
        report['completed']=True
        if device.type=='cuda':report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
        stage('deployment_adapter_actual_sample_matches_training_sample')
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error)
        stage('failed')
        raise


if __name__=='__main__':main()
