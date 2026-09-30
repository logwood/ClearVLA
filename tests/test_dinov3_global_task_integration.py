"""Native 336px file ingress / per-view graph with an explicit marker encoder.

Not a pretrained encoder or a production-width CUDA test.
"""
from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import test_dinov3_production_integration as baseline
import torch
from torch.utils.data import default_collate

from clearvla.mainline.data.loading import to_training_batch
from clearvla.mainline.global_task import COMPILED_TASK_GLOBAL
from clearvla.mainline.runtime.sampling import sample_action


def global_fixture(tmp_path, amp=False):
    cfg, bundle = baseline.fixture(tmp_path, amp)
    cfg = replace(cfg, bottom=replace(cfg.bottom, global_condition_mode=COMPILED_TASK_GLOBAL))
    cfg.validate()
    return cfg, bundle


@pytest.mark.parametrize('amp', [False, True])
def test_native_chart_update_with_cumulative_task_global(tmp_path, amp):
    torch.manual_seed(44)
    cfg, b = global_fixture(tmp_path, amp)
    batch = to_training_batch(default_collate([b.datasets['train'][0]]), goal=b.goal,
        config=cfg, device=torch.device('cpu'), visual_encoder=b.visual_encoder)
    assert batch.online.observation.dino_history.shape == (1,3,2,256,768)
    # Role slices must own the canonical layout independently of the larger
    # request canvas that also carries future/reference/world rows.
    assert batch.online.observation.dino_history.is_contiguous()
    assert batch.online.observation.dino_history.stride() == (3*2*256*768, 2*256*768, 256*768, 768, 1)
    assert batch.online.observation.raw_rgb.shape[-2:] == (336,336)
    m,e = baseline.engine(cfg,b)
    tracked = {n:p for n,p in m.named_parameters()
               if 'global_task_compiler.' in n or '.evidence_adapter.intent_proj.task.' in n}
    before = {n:p.detach().clone() for n,p in tracked.items()}
    e.train_step(batch)
    for n,p in tracked.items():
        assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all(), n
        assert not torch.equal(p,before[n]), n
    assert all(p.grad is None for p in b.visual_encoder.parameters())
    m.eval()
    with torch.no_grad():
        result = sample_action(m,batch.online,cfg,generator=torch.Generator().manual_seed(54))
    assert result.action.shape == (1,24,7) and torch.isfinite(result.action).all()
    del m,e,b,batch,tracked,before,result
    gc.collect()


def test_real_adapter_parity_with_compiled_task(tmp_path, monkeypatch):
    # Reuse every existing exact adapter assertion, only selecting the candidate.
    # The encoder remains the test's explicit marker; production has no fallback.
    original = baseline.fixture
    def selected(path, amp=False):
        cfg,b = original(path,amp)
        return replace(cfg,bottom=replace(cfg.bottom,global_condition_mode=COMPILED_TASK_GLOBAL)), b
    monkeypatch.setattr(baseline,'fixture',selected)
    baseline.test_saved_online_identity_and_real_adapter_input(tmp_path,monkeypatch)

@pytest.mark.parametrize('amp', [False, True])
def test_native_chart_forward_without_update(tmp_path, amp):
    """Separate numeric-forward scope; does not supersede the update tests."""
    torch.manual_seed(49)
    cfg, b = global_fixture(tmp_path, amp)
    batch = to_training_batch(default_collate([b.datasets['train'][0]]), goal=b.goal,
        config=cfg, device=torch.device('cpu'), visual_encoder=b.visual_encoder)
    m, e = baseline.engine(cfg, b)
    m.eval()
    with torch.no_grad():
        output = sample_action(m, batch.online, cfg, generator=torch.Generator().manual_seed(52))
    assert output.action.shape == (1,24,7) and torch.isfinite(output.action).all()
    assert e.global_step == 0
    assert all(p.grad is None for p in b.visual_encoder.parameters())
    del m,e,b,batch,output
    gc.collect()


def test_native_fresh_checkpoint_real_adapter(tmp_path, monkeypatch):
    """Fresh-weight save/adapter parity, NOT a train-step or learned-policy test."""
    from pathlib import Path

    from clearvla.mainline.checkpoint import build_checkpoint_identity
    from clearvla.mainline.runtime.checkpoints import save_checkpoint
    from clearvla.mainline.runtime.identity import dataset_identity, language_identity
    from clearvla.mainline.train import _data_state
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    from clearvla.simulation.history import CausalHistory
    from clearvla.vision.online_pipeline import OnlineVisionPipeline
    cfg,b = global_fixture(tmp_path)
    torch.manual_seed(66)
    m,e = baseline.engine(cfg,b)
    batch=to_training_batch(default_collate([b.datasets['train'][0]]),goal=b.goal,config=cfg,
                            device=torch.device('cpu'),visual_encoder=b.visual_encoder)
    identity=build_checkpoint_identity(cfg,repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset_identity(b,cfg),language=language_identity(b,cfg))
    assert 'clearvla/mainline/model/global_task.py' in dict(identity.source.files)
    ckpt=tmp_path/'fresh.pt'
    save_checkpoint(ckpt,model=m,optimizer=e.optimizer,schedule=e.schedule,config=cfg,
                    identity=identity,epoch=0,global_step=0,best_metric=None,data_state=_data_state(b,cfg,identity))
    m.eval()
    with torch.no_grad():
        sampled=sample_action(m,batch.online,cfg,generator=torch.Generator().manual_seed(15))
        expected=b.action_normalizer.decode(sampled.action[0].float().numpy()).astype('float32')
        expected[:,-1]=sampled.gripper_command[0].float().numpy()
    del m,e,sampled
    gc.collect()
    monkeypatch.setattr(OnlineVisionPipeline,'from_config',classmethod(lambda cls,c,device:baseline.pipeline()))
    policy=ClearVLACheckpointPolicy(ckpt,device=torch.device('cpu'),seed=15)
    episode=b.episodes[b.splits['train'][0]]
    history=CausalHistory(executed_world=True)
    history.reset(baseline._observation(episode,0))
    for i in range(24):
        history.append(episode.actions_raw[i], baseline._observation(episode, i + 1))
    action,online=policy.act_with_input(history.snapshot(),episode.instruction)
    baseline._assert_same_tree(batch.online,online)
    torch.testing.assert_close(torch.from_numpy(action),torch.from_numpy(expected),rtol=0,atol=0)
    assert policy.deployment_health()['observation']['dinov3_runtime']['disk_feature_cache'] is False
    del policy,b,batch,online
    gc.collect()
