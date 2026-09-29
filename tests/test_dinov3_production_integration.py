"""Actual file/engine/checkpoint/adapter tests using an EXPLICIT marker encoder.

These exercise ClearVLA, not learned DINOv3 behavior. Production loading has no
marker/random fallback. Pretrained/GPU admission is the separate CLI probe.
"""
from __future__ import annotations

import copy
import gc
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import default_collate
from types import SimpleNamespace

from test_mainline_cumulative_data_entry import source_fixture, _observation, _assert_same_tree
from clearvla.mainline.data.loading import load_mainline_data, to_training_batch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.optimizer import build_optimizer, WarmupCosineSchedule
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.vision.dinov3_online import FrozenDinoV3Encoder, patch_centers_in_full_image
from clearvla.vision.online_pipeline import OnlineVisionPipeline, rasterize_full_rgb, resize_endpoint_chart


class SpatialMarker(nn.Module):
    """Camera/time-labelled artificial features, NOT pretrained DINO weights."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(model_type="dinov3_vit", patch_size=16, hidden_size=768, num_register_tokens=4,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072)

    def forward(self, pixel_values):
        rgb = pixel_values.float().mean((-1,-2))
        # retain 3 independent channels, patch position and finite magnitudes
        i = torch.arange(768,device=rgb.device)
        channels = rgb[:,i % 3][:,None,:]
        pos = torch.arange(256,device=rgb.device)[None,:,None] / 256.
        patch = (channels + pos + .01*(i%7)[None,None,:]) * self.weight
        return SimpleNamespace(last_hidden_state=torch.cat((patch[:,:5]*0,patch),1))


def pipeline(dtype=torch.float32):
    e = FrozenDinoV3Encoder(SpatialMarker().to(dtype),microbatch=2,
        identity={"kind":"injected_test_marker_NOT_pretrained"})
    return OnlineVisionPipeline(e)


def fixture(tmp_path, amp=False):
    cfg,_,_=source_fixture(tmp_path)
    shutil.rmtree(cfg.data.dino_cache)
    cfg = replace(cfg,
        data=replace(cfg.data, visual_feature_mode="dinov3_online_v1", dino_cache="",
                     dinov3_model=str(tmp_path/'authorized-model-location')),
        observation=replace(cfg.observation,visual_chart_mode="full_rgb_endpoint_v1"),
        top=replace(cfg.top,object_view_mode="per_camera_values_v1"),
        dimensions=replace(cfg.dimensions,visual_token_dim=768,patches_per_camera=256),
        runtime=replace(cfg.runtime,compute_dtype="bf16" if amp else "fp32"),
    )
    cfg.validate()
    b=load_mainline_data(cfg)
    return cfg,replace(b,visual_encoder=pipeline(torch.bfloat16 if amp else torch.float32))


def engine(cfg,bundle):
    m=ClearVLAMainlinePolicy(cfg)
    m.configure_action_normalizer(bundle.action_normalizer)
    opt,_=build_optimizer(m,cfg)
    sched=WarmupCosineSchedule(opt,warmup_steps=500,total_steps=22024,minimum_ratio=.1)
    e=MainlineTrainingEngine(model=m,config=cfg,optimizer=opt,schedule=sched,
        device=torch.device('cpu'),dtype=torch.bfloat16 if cfg.runtime.compute_dtype=='bf16' else torch.float32,
        train_flow_generator=torch.Generator().manual_seed(10),
        train_condition_generator=torch.Generator().manual_seed(11))
    return m,e


def test_online_formal_loader_never_opens_token_cache_and_preserves_tail(tmp_path, monkeypatch):
    monkeypatch.setattr('clearvla.mainline.data.loading.DinoV2TokenStore',lambda *a,**k:pytest.fail('cache accessed'))
    cfg,b=fixture(tmp_path)
    rows=[b.datasets['train'][0],b.datasets['train'][-1]]
    assert 'history_dinov2_tokens' not in rows[0]
    assert rows[0]['visual_request_rgb'].dtype==torch.uint8
    assert rows[0]['visual_request_rgb'].shape==(14,2,3,336,336)
    batch=to_training_batch(default_collate(rows),goal=b.goal,config=cfg,device=torch.device('cpu'),visual_encoder=b.visual_encoder)
    assert batch.online.observation.dino_history.shape==(2,3,2,256,768)
    assert torch.count_nonzero(batch.future.dino_supports[1])==0
    assert batch.future.annotation_endpoint.declared.all()
    assert batch.future.annotation_endpoint.visual_observed.all()
    assert b.visual_encoder.last_batch_stats['encoded_unique'] < b.visual_encoder.last_batch_stats['supported']
    assert not Path(cfg.data.dino_cache or '/no-token-cache-created').is_file()


def test_conflicting_duplicate_source_pixels_are_rejected(tmp_path):
    cfg,b=fixture(tmp_path)
    raw=default_collate([b.datasets['train'][0]] * 2)
    raw['visual_request_rgb'][1,0,0,0,0,0] += 1
    with pytest.raises(ValueError,match='different RGB'):
        b.visual_encoder.prepare_batch(raw,cfg)


def test_online_rejects_cached_or_missing_producer(tmp_path):
    cfg,b=fixture(tmp_path)
    raw=default_collate([b.datasets['train'][0]])
    with pytest.raises(ValueError,match='resident'):
        to_training_batch(raw,goal=b.goal,config=cfg,device=torch.device('cpu'))
    raw['history_dinov2_tokens']=torch.zeros(1)
    with pytest.raises(ValueError,match='mixed'):
        b.visual_encoder.prepare_batch(raw,cfg)


def test_future_rgb_cannot_change_current_condition(tmp_path):
    cfg,b=fixture(tmp_path)
    # Separate frame ownership; edit future pixels but not causal requests.
    raw=default_collate([b.datasets['train'][0]])
    first=b.visual_encoder.prepare_batch(raw,cfg)
    other={k:v.clone() for k,v in raw.items()}
    other['visual_request_rgb'][:,3:9] = 255 - other['visual_request_rgb'][:,3:9]
    second=b.visual_encoder.prepare_batch(other,cfg)
    for name in ('history_dinov2_tokens','instruction_reference_dino','executed_world_dino'):
        torch.testing.assert_close(first[name],second[name],atol=0,rtol=0)
    assert not torch.equal(first['target_future_dinov2_tokens'],second['target_future_dinov2_tokens'])


def test_raster_is_transformed_not_relabelled_native():
    native=patch_centers_in_full_image(image_hw=(336,336)).reshape(1,256,2).repeat(1,1,384)
    raster=rasterize_full_rgb(native,(336,336)).reshape(1,16,16,768)
    axis=torch.linspace(-1,1,16)
    # Exclude the border where declared nearest-center extension applies.
    torch.testing.assert_close(raster[0,8,1:-1,0],axis[1:-1],atol=3e-7,rtol=0)
    assert not torch.equal(raster.flatten(1,2),native)
    low=resize_endpoint_chart(raster.permute(0,3,1,2),(8,8),'full_rgb_endpoint_v1')
    torch.testing.assert_close(low[0,0,4,1:-1],torch.linspace(-1,1,8)[1:-1],atol=3e-7,rtol=0)


@pytest.mark.parametrize('amp',[False,True])
def test_disk_online_engine_update_and_sampling(tmp_path,amp):
    torch.manual_seed(44)
    cfg,b=fixture(tmp_path,amp)
    batch=to_training_batch(default_collate([b.datasets['train'][0]]),goal=b.goal,config=cfg,
                            device=torch.device('cpu'),visual_encoder=b.visual_encoder)
    m,e=engine(cfg,b)
    tracked={n:p.detach().clone() for n,p in m.named_parameters() if 'task_relation_encoder' in n}
    e.train_step(batch)
    assert e.global_step==1 and e.schedule.step_index==1
    assert any(not torch.equal(v,dict(m.named_parameters())[n]) for n,v in tracked.items())
    assert all(p.grad is None for p in b.visual_encoder.parameters())
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad():
        a=sample_action(m,batch.online,cfg,generator=torch.Generator().manual_seed(4))
        aa=sample_action(m,batch.online,cfg,generator=torch.Generator().manual_seed(4))
    assert a.action.shape==(1,24,7) and torch.isfinite(a.action).all()
    torch.testing.assert_close(a.action,aa.action,atol=0,rtol=0)
    del m,e,b,batch,a,aa,tracked
    gc.collect()


def test_saved_online_identity_and_real_adapter_input(tmp_path,monkeypatch):
    from clearvla.mainline.runtime.identity import dataset_identity, language_identity
    from clearvla.mainline.checkpoint import build_checkpoint_identity
    from clearvla.mainline.train import _data_state
    from clearvla.mainline.runtime.checkpoints import save_checkpoint, load_checkpoint_exact
    from clearvla.mainline.runtime.deployment import validate_deployment_abi, canonical_sha256
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    from clearvla.simulation.history import CausalHistory
    cfg,b=fixture(tmp_path)
    m,e=engine(cfg,b)
    raw=default_collate([b.datasets['train'][0]])
    batch=to_training_batch(raw,goal=b.goal,config=cfg,device=torch.device('cpu'),visual_encoder=b.visual_encoder)
    e.train_step(batch)
    identity=build_checkpoint_identity(cfg,repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset_identity(b,cfg),language=language_identity(b,cfg))
    assert 'clearvla/vision/online_pipeline.py' in dict(identity.source.files)
    state=_data_state(b,cfg,identity)
    abi=state['deployment_abi']
    assert 'dinov2' not in abi['observation']
    validate_deployment_abi(abi)
    bad=copy.deepcopy(abi);bad['observation']['dinov3']['identity']['chart']='wrong'
    with pytest.raises(ValueError):validate_deployment_abi(bad)
    ckpt=tmp_path/'online.pt'
    save_checkpoint(ckpt,model=m,optimizer=e.optimizer,schedule=e.schedule,config=cfg,
                    identity=identity,epoch=1,global_step=1,best_metric=None,data_state=state)
    # Exact restoration into the existing engine verifies optimizer/schedule too.
    load_checkpoint_exact(ckpt,model=m,optimizer=e.optimizer,schedule=e.schedule,config=cfg,identity=identity)
    m.eval()
    with torch.no_grad():
        sampled=sample_action(m,batch.online,cfg,generator=torch.Generator().manual_seed(15))
        expected=b.action_normalizer.decode(sampled.action[0].float().numpy()).astype("float32")
        expected[:,-1]=sampled.gripper_command[0].float().numpy()
    del m,e
    gc.collect()
    monkeypatch.setattr(OnlineVisionPipeline,'from_config',classmethod(lambda cls,c,device: pipeline()))
    policy=ClearVLACheckpointPolicy(ckpt,device=torch.device('cpu'),seed=15)
    episode=b.episodes[b.splits['train'][0]]
    history=CausalHistory(executed_world=True)
    history.reset(_observation(episode,0))
    for i in range(24):history.append(episode.actions_raw[i],_observation(episode,i+1))
    action,online=policy.act_with_input(history.snapshot(),episode.instruction)
    _assert_same_tree(batch.online,online)
    torch.testing.assert_close(torch.from_numpy(action),torch.from_numpy(expected),atol=0,rtol=0)
    health=policy.deployment_health()
    assert health['observation']['dinov3_runtime']['disk_feature_cache'] is False
    del policy
    gc.collect()


def test_online_source_camera_provenance_cannot_be_swapped(tmp_path):
    cfg,b=fixture(tmp_path)
    raw=default_collate([b.datasets['train'][0]])
    raw['visual_request_keys'][...,2] = raw['visual_request_keys'][...,2].flip(-1)
    with pytest.raises(ValueError,match='camera indices'):
        b.visual_encoder.prepare_batch(raw,cfg)


def test_abi_rejects_changed_geometry_even_if_digest_recomputed(tmp_path):
    from clearvla.mainline.checkpoint import build_checkpoint_identity
    from clearvla.mainline.runtime.identity import dataset_identity,language_identity
    from clearvla.mainline.runtime.deployment import validate_deployment_abi,canonical_sha256
    from clearvla.mainline.train import _data_state
    cfg,b=fixture(tmp_path)
    identity=build_checkpoint_identity(cfg,repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset_identity(b,cfg),language=language_identity(b,cfg))
    abi=_data_state(b,cfg,identity)['deployment_abi']
    validate_deployment_abi(abi)
    for key,value in [('camera_names',['wrist','top']),('raster',{})]:
        changed=copy.deepcopy(abi)
        changed['observation']['dinov3']['identity'][key]=value
        changed['observation']['dinov3']['identity_sha256']=canonical_sha256(changed['observation']['dinov3']['identity'])
        with pytest.raises(ValueError,match='identity differs'):
            validate_deployment_abi(changed)
