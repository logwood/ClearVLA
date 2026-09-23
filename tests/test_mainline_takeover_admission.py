"""Takeover regression: fail before mutating the accepted training state.

Checkpoint tests use a small Linear/AdamW with the actual checkpoint API. Loss
fault injection replaces only loss construction; model/engine/clipping/optimizer
are production classes in the existing reduced structural configuration. This
is not real-data, production-width, GPU, or physical-robot qualification.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_mainline_checkpoint import _dataset
from test_mainline_endpoint_supervision import _config
from test_mainline_state_features import _model_engine

from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.runtime.checkpoints import (
    load_checkpoint_exact,
    load_checkpoint_for_initialization,
    load_checkpoint_for_validation,
    save_checkpoint,
)
from clearvla.mainline.training.losses import LossLedger
from clearvla.mainline.training.optimizer import WarmupCosineSchedule


def _tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _tree_equal(x, y)
    else:
        assert type(a) is type(b) and a == b


@pytest.mark.parametrize('bad', [float('inf'), -float('inf'), float('nan')])
def test_nonfinite_loss_with_finite_derivative_cannot_update(bad, monkeypatch):
    torch.manual_seed(923)
    model, engine = _model_engine(_config())
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(engine.optimizer.state_dict())
    before_schedule = copy.deepcopy(engine.schedule.state_dict())

    def fault(*args, **kwargs):
        total = parameter.float().square().mean() + bad
        # A nonfinite constant has no nonfinite derivative. Gradient-only
        # admission therefore cannot replace checking the scalar objective.
        derivative = torch.autograd.grad(total, parameter, retain_graph=True)[0]
        assert torch.isfinite(derivative).all()
        zero = total.new_zeros(())
        return LossLedger(total, {'action': total, 'representation': zero,
                                 'execution': zero}, {'fault': total}, {}), {}

    monkeypatch.setattr(engine, '_forward', fault)
    with pytest.raises(FloatingPointError, match='non-finite training loss'):
        engine.train_step(None)
    _tree_equal(before_model, model.state_dict())
    _tree_equal(before_optimizer, engine.optimizer.state_dict())
    _tree_equal(before_schedule, engine.schedule.state_dict())
    assert engine.global_step == 0
    assert parameter.grad is None


@pytest.fixture(scope="module")
def checkpoint_identity(tmp_path_factory):
    # The source/config/language identity is immutable within one frozen test
    # process. Build the real source closure once, never mock its validation.
    condition = tmp_path_factory.mktemp("takeover_identity") / "goal.pt"
    condition.write_bytes(b"takeover-synthetic-language-identity")
    return build_checkpoint_identity(
        ExperimentConfig(), repo_root=Path(__file__).resolve().parents[1], dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", condition), commit="1" * 40,
    )


def _checkpoint(tmp_path: Path, identity):
    torch.manual_seed(924)
    config = ExperimentConfig()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=8,
                                    minimum_ratio=.1)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    schedule.step()
    path = tmp_path / 'resume.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, schedule=schedule,
                    config=config, identity=identity, epoch=1, global_step=1,
                    best_metric=None)
    kwargs = dict(model=model, optimizer=optimizer, schedule=schedule,
                  config=config, identity=identity)
    return path, kwargs


def _reject_without_mutation(path, kwargs):
    with torch.no_grad():
        for p in kwargs['model'].parameters():
            p.add_(5.0)
    before = (copy.deepcopy(kwargs['model'].state_dict()),
              copy.deepcopy(kwargs['optimizer'].state_dict()),
              copy.deepcopy(kwargs['schedule'].state_dict()),
              torch.get_rng_state().clone())
    with pytest.raises(ValueError):
        load_checkpoint_exact(path, **kwargs)
    after = (kwargs['model'].state_dict(), kwargs['optimizer'].state_dict(),
             kwargs['schedule'].state_dict(), torch.get_rng_state())
    _tree_equal(before, after)


@pytest.mark.parametrize('field', ['epoch', 'global_step', 'schedule.step_index'])
@pytest.mark.parametrize('bad', [True, 1.5, '1', float('inf')])
def test_exact_resume_rejects_noninteger_clock_before_mutation(tmp_path, checkpoint_identity, field, bad):
    path, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    payload = torch.load(path, weights_only=False)
    if field.startswith('schedule.'):
        payload['schedule']['step_index'] = bad
    else:
        payload[field] = bad
    torch.save(payload, path)
    _reject_without_mutation(path, kwargs)


@pytest.mark.parametrize('field,bad', [
    ('betas', (.8, .99)), ('eps', 1e-3), ('weight_decay', .9),
    ('maximize', True), ('eps', float('nan')),
])
def test_exact_resume_rejects_optimizer_semantic_drift(tmp_path, checkpoint_identity, field, bad):
    path, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    payload = torch.load(path, weights_only=False)
    payload['optimizer']['param_groups'][0][field] = bad
    torch.save(payload, path)
    _reject_without_mutation(path, kwargs)


@pytest.mark.parametrize('field', ['epoch', 'global_step'])
@pytest.mark.parametrize('bad', [True, 1.5, '1'])
def test_save_rejects_coerced_clock_without_creating_checkpoint(tmp_path, checkpoint_identity, field, bad):
    _, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    destination = tmp_path / 'invalid.pt'
    clocks = dict(epoch=1, global_step=1)
    clocks[field] = bad
    with pytest.raises(ValueError):
        save_checkpoint(destination, **kwargs, **clocks, best_metric=None)
    assert not destination.exists()


def test_valid_exact_resume_next_update_is_bit_exact(tmp_path, checkpoint_identity):
    path, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    model, optimizer, schedule = (kwargs[k] for k in ('model', 'optimizer', 'schedule'))

    def step():
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[.2, -.3, .7]])).square().mean().backward()
        optimizer.step()
        schedule.step()

    step()
    expected = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(schedule.state_dict()))
    restored = load_checkpoint_exact(path, **kwargs)
    assert restored.global_step == 1
    step()
    _tree_equal(expected, (model.state_dict(), optimizer.state_dict(), schedule.state_dict()))


@pytest.mark.parametrize("loader", [load_checkpoint_for_validation, load_checkpoint_for_initialization])
@pytest.mark.parametrize("field", ["epoch", "global_step"])
@pytest.mark.parametrize("bad", [True, 1.5, "1", float("inf")])
def test_model_only_load_rejects_coerced_clock_before_mutation(
    tmp_path, checkpoint_identity, loader, field, bad
):
    path, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    payload = torch.load(path, weights_only=False)
    payload[field] = bad
    torch.save(payload, path)
    model = kwargs["model"]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        loader(path, model=model, config=kwargs["config"], identity=kwargs["identity"])
    _tree_equal(before, model.state_dict())


@pytest.mark.parametrize("bad", [True, 1.5, "1", float("inf")])
def test_save_rejects_invalid_schedule_clock(tmp_path, checkpoint_identity, bad):
    _, kwargs = _checkpoint(tmp_path, checkpoint_identity)
    kwargs["schedule"].step_index = bad
    destination = tmp_path / "invalid_schedule.pt"
    with pytest.raises(ValueError):
        save_checkpoint(destination, **kwargs, epoch=1, global_step=1, best_metric=None)
    assert not destination.exists()
