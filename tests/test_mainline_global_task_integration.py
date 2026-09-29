"""Actual policy/engine/sampler/checkpoint tests with explicit synthetic transport."""
from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import torch

from test_mainline_global_task import abi_for, config
from test_mainline_task_execution_integration import _batch, config as old_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.global_task import CompiledGlobalTask
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.simulation.checkpoint import load_deployment_checkpoint


@pytest.fixture(autouse=True)
def free_graphs():
    gc.collect()
    yield
    gc.collect()


@pytest.mark.parametrize('amp', [False, True])
def test_real_update_gradient_ownership_and_complete_two_pass_lifecycle(amp):
    torch.manual_seed(610)
    c = config(amp)
    model, engine = _model_engine(c)
    batch = _batch()
    tracked = {n:p for n,p in model.named_parameters()
               if 'global_task_compiler.' in n or '.evidence_adapter.intent_proj.task.' in n}
    assert len(tracked) == 3
    parameters = [p for g in engine.optimizer.param_groups for p in g['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    assert all(sum(x is p for x in parameters) == 1 for p in tracked.values())
    before = {n:p.detach().clone() for n,p in tracked.items()}
    engine.train_step(batch)
    assert engine.global_step == 1
    for name,p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        assert not torch.equal(p, before[name]), name
    model.eval()
    compiled, consumed, latent = [], [], []
    compiler = model.policy_compiler.plan_compiler.global_task_compiler
    hook = compiler.register_forward_hook(lambda module, args, out: compiled.append(out))
    z_hook = model.execution_bottom.decoder.organizer.register_forward_hook(
        lambda module, args, out: latent.append(out['latent'].detach().clone()))
    original = model.velocity
    def observe(cache, **kwargs):
        consumed.append(cache.top.intent.compiled_global_task)
        return original(cache, **kwargs)
    model.velocity = observe
    try:
        with torch.no_grad():
            sample = sample_action(model, batch.online, c, generator=torch.Generator().manual_seed(619))
        assert len(compiled) == 1 and len(consumed) == 12
        assert all(value is compiled[0] for value in consumed)
        assert len(latent) == 12
        for value in latent[1:]:
            torch.testing.assert_close(value, latent[0], rtol=0, atol=0)
        assert torch.isfinite(sample.action).all() and sample.action.shape == (1,24,7)
        with torch.no_grad():
            again = sample_action(model, batch.online, c, generator=torch.Generator().manual_seed(619))
        assert len(compiled) == 2 and compiled[0] is not compiled[1]
        torch.testing.assert_close(sample.action, again.action, atol=0, rtol=0)
    finally:
        model.velocity = original
        hook.remove()
        z_hook.remove()


@pytest.mark.parametrize('amp', [False, True])
def test_language_changes_global_with_same_observation_history_and_noise(amp):
    torch.manual_seed(612)
    c = config(amp)
    model, _ = _model_engine(c)
    model.eval()
    batch = _batch()
    source_goal = batch.online.goal
    alternate = replace(batch.online, goal=replace(source_goal, tokens=-source_goal.tokens))
    noise, time = torch.randn(1,24,18), torch.full((1,),.4)
    values = []
    hook = model.execution_bottom.decoder.organizer.register_forward_hook(
        lambda module,args,out: values.append(out['global_condition']))
    try:
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
            cache, _, _ = model.encode_online(batch.online)
            cache2, _, _ = model.encode_online(alternate)
            model.velocity(cache, noisy_action_field=noise, time=time)
            model.velocity(cache2, noisy_action_field=noise, time=time)
        assert not torch.equal(cache.top.intent.compiled_global_task.tokens,
                               cache2.top.intent.compiled_global_task.tokens)
        assert not torch.equal(values[0], values[1])
    finally:
        hook.remove()


def test_task_source_gradient_through_actual_global_only_no_teacher():
    torch.manual_seed(615)
    c = config()
    model, engine = _model_engine(c)
    model.eval()
    batch = _batch()
    goal = replace(batch.online.goal, tokens=batch.online.goal.tokens.clone().requires_grad_())
    cache, _, _ = model.encode_online(replace(batch.online, goal=goal))
    values = []
    handle = model.execution_bottom.decoder.organizer.register_forward_hook(
        lambda module,args,out: values.append(out['global_condition']))
    try:
        model.velocity(cache, noisy_action_field=torch.randn(1,24,18), time=torch.full((1,),.4))
        loss = values[0].float().square().mean()
        g = torch.autograd.grad(loss, (goal.tokens, *model.policy_compiler.plan_compiler.global_task_compiler.parameters()),
                                allow_unused=True)
        assert all(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0 for x in g)
    finally:
        handle.remove()


@pytest.mark.parametrize('mutation', ['missing', 'source', 'producer'])
def test_foreign_or_missing_compilation_is_rejected_before_decoder(mutation):
    model, _ = _model_engine(config())
    model.eval()
    with torch.no_grad():
        cache, _, _ = model.encode_online(_batch().online)
    record = cache.top.intent.compiled_global_task
    assert record is not None
    bad = (None if mutation == 'missing' else
           replace(record, source_intervals=record.source_intervals.clone()) if mutation == 'source'
           else replace(record, compiler_identity=record.compiler_identity + 1))
    altered = replace(cache, top=replace(cache.top, intent=replace(cache.top.intent, compiled_global_task=bad)))
    calls = []
    handle = model.execution_bottom.decoder.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        with pytest.raises(ValueError, match='global task'):
            model.velocity(altered, noisy_action_field=torch.randn(1,24,18), time=torch.full((1,),.4))
        assert not calls
    finally:
        handle.remove()


def test_future_supervision_does_not_mutate_compiled_task():
    c = config()
    model, _ = _model_engine(c)
    model.eval()
    batch = _batch()
    with torch.no_grad():
        cache, state, _ = model.encode_online(batch.online)
        record = cache.top.intent.compiled_global_task
        saved = record.tokens.clone()
        # Future targets use the actual Teacher plane. They cannot update a
        # current-only record, even when the target payload is changed.
        model.build_training_targets(state, batch.future)
        changed = replace(batch.future, state_sequence=batch.future.state_sequence + .1)
        model.build_training_targets(state, changed)
    assert cache.top.intent.compiled_global_task is record
    torch.testing.assert_close(saved, record.tokens, rtol=0, atol=0)


def test_checkpoint_exact_resume_deployment_and_cross_graph_rejection(tmp_path):
    torch.manual_seed(620)
    c = config()
    model, engine = _model_engine(c)
    batch = _batch()
    identity, abi, data = abi_for(c, tmp_path)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    model.eval()
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model=model, optimizer=engine.optimizer, schedule=engine.schedule,
        config=c, identity=identity, epoch=0, global_step=1, best_metric=None,
        data_state={'action_normalizer':data.action_normalizer.to_dict(),
                    'state_normalizer':data.state_normalizer.to_dict(), 'deployment_abi':abi})
    with torch.no_grad():
        expected = sample_action(model, batch.online, c, generator=torch.Generator().manual_seed(621))
    other, oe = _model_engine(c)
    load_checkpoint_exact(path, model=other, optimizer=oe.optimizer, schedule=oe.schedule, config=c, identity=identity)
    # Exact next-update equivalence (including generators/schedule) is checked
    # by the checkpoint inventory; here the new parameters must round-trip.
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_checkpoint_exact(path, model=other, optimizer=oe.optimizer, schedule=oe.schedule,
                              config=old_config(), identity=identity)
    saved = torch.load(path, weights_only=False, map_location='cpu')
    def contains_runtime_record(value):
        if isinstance(value, CompiledGlobalTask): return True
        if isinstance(value, dict): return any(contains_runtime_record(x) for x in value.values())
        if isinstance(value, (list,tuple)): return any(contains_runtime_record(x) for x in value)
        return False
    assert not contains_runtime_record(saved)
    del other, oe, model, engine, saved
    gc.collect()
    deployed = load_deployment_checkpoint(path, device=torch.device('cpu'))
    with torch.no_grad():
        actual = sample_action(deployed.model, batch.online, deployed.config,
                               generator=torch.Generator().manual_seed(621))
    torch.testing.assert_close(expected.action, actual.action, rtol=0, atol=0)


def test_post_warmup_inference_retains_compiled_task_source():
    # Emulates the existing controller clock on fresh weights, not 1200 learned updates.
    torch.manual_seed(629)
    c = config()
    m, _ = _model_engine(c)
    m.set_training_step(1200)
    m.eval()
    records = []
    handle = m.execution_bottom.decoder.organizer.register_forward_hook(
        lambda module,args,out: records.append(out['latent'].detach().clone()))
    try:
        with torch.no_grad():
            sample = sample_action(m,_batch().online,c,generator=torch.Generator().manual_seed(68))
        assert len(records) == 12 and torch.isfinite(sample.action).all()
        assert all(torch.equal(records[0],x) for x in records[1:])
    finally:
        handle.remove()
