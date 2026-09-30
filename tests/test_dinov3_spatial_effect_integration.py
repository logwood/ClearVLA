"""Native 336px/HDF5 DINOv3 interface with an explicit marker, never pretrained.

The policy itself, normalizers, cache, sampler, checkpoint and simulator-side
adapter are production implementations. CPU forward and train/update are
separate tests so a resource-limited run cannot re-label the former as training.
"""
from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import default_collate

import test_dinov3_production_integration as baseline
from clearvla.mainline.data.loading import to_training_batch
from clearvla.mainline.global_task import COMPILED_TASK_GLOBAL
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.task_execution import JOINT_SPATIAL_TASK_EXECUTION


def fixture(tmp_path, amp=False):
    cfg, bundle = baseline.fixture(tmp_path, amp)
    cfg = replace(cfg, top=replace(cfg.top, task_execution_mode=JOINT_SPATIAL_TASK_EXECUTION),
                  bottom=replace(cfg.bottom, global_condition_mode=COMPILED_TASK_GLOBAL))
    cfg.validate()
    return cfg, bundle


def batch_for(cfg, bundle):
    batch = to_training_batch(default_collate([bundle.datasets['train'][0]]),
        goal=bundle.goal, config=cfg, device=torch.device('cpu'), visual_encoder=bundle.visual_encoder)
    assert batch.online.observation.dino_history.shape == (1, 3, 2, 256, 768)
    assert batch.online.observation.raw_rgb.shape[-2:] == (336, 336)
    return batch


@pytest.mark.parametrize('amp', [False, True])
def test_native_chart_forward_without_update(tmp_path, amp):
    torch.manual_seed(4731)
    cfg, bundle = fixture(tmp_path, amp)
    batch = batch_for(cfg, bundle)
    model, engine = baseline.engine(cfg, bundle)
    model.eval()
    with torch.no_grad():
        result = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(67))
    assert result.action.shape == (1, 24, 7) and torch.isfinite(result.action).all()
    assert engine.global_step == 0
    assert all(p.grad is None for p in bundle.visual_encoder.parameters())
    del model, engine, bundle, batch, result
    gc.collect()


@pytest.mark.parametrize('amp', [False, True])
def test_native_chart_optimizer_update(tmp_path, amp):
    torch.manual_seed(4732)
    cfg, bundle = fixture(tmp_path, amp)
    batch = batch_for(cfg, bundle)
    model, engine = baseline.engine(cfg, bundle)
    tracked = {n:p for n,p in model.named_parameters() if any(
        key in n for key in ('spatial_context.', 'effect_context_key.',
                             'global_task_compiler.', '.evidence_adapter.intent_proj.task.'))}
    assert tracked
    before = {n:p.detach().clone() for n,p in tracked.items()}
    engine.train_step(batch)
    assert engine.global_step == 1
    for name, param in tracked.items():
        assert param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all(), name
        assert not torch.equal(param, before[name]), name
    assert all(p.grad is None for p in bundle.visual_encoder.parameters())
    del model, engine, bundle, batch, tracked, before
    gc.collect()


def test_native_fresh_checkpoint_exact_adapter_parity(tmp_path, monkeypatch):
    """Fresh model roundtrip/adapter identity, explicitly NOT a learned-policy run."""
    from clearvla.mainline.checkpoint import build_checkpoint_identity
    from clearvla.mainline.runtime.identity import dataset_identity, language_identity
    from clearvla.mainline.runtime.checkpoints import save_checkpoint
    from clearvla.mainline.train import _data_state
    from clearvla.simulation.clearvla_policy import ClearVLACheckpointPolicy
    from clearvla.simulation.history import CausalHistory
    from clearvla.vision.online_pipeline import OnlineVisionPipeline

    cfg, bundle = fixture(tmp_path)
    torch.manual_seed(4777)
    model, engine = baseline.engine(cfg, bundle)
    batch = batch_for(cfg, bundle)
    identity = build_checkpoint_identity(cfg, repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset_identity(bundle, cfg), language=language_identity(bundle, cfg))
    path = tmp_path / 'fresh.pt'
    save_checkpoint(path, model=model, optimizer=engine.optimizer, schedule=engine.schedule,
        config=cfg, identity=identity, epoch=0, global_step=0, best_metric=None,
        data_state=_data_state(bundle, cfg, identity))
    model.eval()
    with torch.no_grad():
        sampled = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(15))
        expected = bundle.action_normalizer.decode(sampled.action[0].float().numpy()).astype('float32')
        expected[:, -1] = sampled.gripper_command[0].float().numpy()
    del model, engine, sampled
    gc.collect()
    monkeypatch.setattr(OnlineVisionPipeline, 'from_config',
                        classmethod(lambda cls, config, device: baseline.pipeline()))
    policy = ClearVLACheckpointPolicy(path, device=torch.device('cpu'), seed=15)
    episode = bundle.episodes[bundle.splits['train'][0]]
    history = CausalHistory(executed_world=True)
    history.reset(baseline._observation(episode, 0))
    for i in range(24):
        history.append(episode.actions_raw[i], baseline._observation(episode, i+1))
    action, online = policy.act_with_input(history.snapshot(), episode.instruction)
    baseline._assert_same_tree(batch.online, online)
    torch.testing.assert_close(torch.from_numpy(action), torch.from_numpy(expected), rtol=0, atol=0)
    assert policy.deployment_health()['observation']['dinov3_runtime']['disk_feature_cache'] is False
    del policy, bundle, batch, online
    gc.collect()
