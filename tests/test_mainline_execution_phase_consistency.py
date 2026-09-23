"""One completed-update clock for training, evaluation and deployment.

Inputs use the existing synthetic transport fixture; the model, optimizer,
engine and deployment sampler are production implementations.
"""
from __future__ import annotations

import copy

import pytest
import torch
from test_mainline_endpoint_supervision import _config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import MainlineTrainingEngine

Production = tuple[ExperimentConfig, ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch]


@pytest.fixture
def production() -> Production:
    torch.manual_seed(7101)
    config = _config()
    model, engine = _model_engine(config)
    return config, model, engine, _batch()


def _progress(model: ClearVLAMainlinePolicy) -> float:
    value = model.execution_bottom.decoder.execution_progress
    assert isinstance(value, torch.Tensor)
    return float(value.item())


def _expected(config: ExperimentConfig, step: int) -> float:
    return float(torch.tensor(min(max(
        (step - config.bottom.execution_warmup_steps)
        / config.bottom.execution_transition_steps, 0.0), 1.0)))


@pytest.mark.parametrize("step", [0, 199, 200, 201, 700, 1200])
def test_eval_encoding_uses_completed_update_count(production: Production, step: int) -> None:
    config, model, engine, batch = production
    model.set_training_step(17)
    engine.global_step = step
    engine.encode_eval(batch, collect_diagnostics=False)
    decoder = model.execution_bottom.decoder
    assert _progress(model) == _expected(config, step)
    assert decoder._execution_progress_value == _expected(config, step)


@pytest.mark.parametrize("cached", [False, True])
def test_eval_loss_matches_deployment_phase_with_or_without_cache(production: Production, cached: bool) -> None:
    config, model, engine, batch = production
    engine.global_step = 301
    encoded = engine.encode_eval(batch, collect_diagnostics=False) if cached else None
    model.set_training_step(0)
    result = engine.eval_step(batch, encoded=encoded, collect_diagnostics=False,
                             generator=torch.Generator().manual_seed(715))
    assert torch.isfinite(result.loss)
    assert _progress(model) == _expected(config, 301)


def test_optimizer_boundary_has_same_phase_before_and_after_reload() -> None:
    torch.manual_seed(7101)
    config = _config()
    model, engine = _model_engine(config)
    batch = _batch()
    engine.global_step = 200
    result = engine.train_step(batch, collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert engine.global_step == 201
    assert _progress(model) == _expected(config, 201)
    # The current update's diagnostics are frozen, not an alias to the next phase.
    assert float(result.metrics["evidence_mmd_it_execution_progress"]) == 0.0
    saved = copy.deepcopy(model.state_dict())
    restored, _ = _model_engine(config)
    restored.load_state_dict(saved, strict=True)
    # This is the existing deployment loader's step restoration convention.
    restored.set_training_step(engine.global_step)
    a = sample_action(model, batch.online, config, generator=torch.Generator().manual_seed(71))
    b = sample_action(restored, batch.online, config, generator=torch.Generator().manual_seed(71))
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    torch.testing.assert_close(a.physical_field, b.physical_field, rtol=0, atol=0)


def test_failure_does_not_publish_the_next_phase(production: Production, monkeypatch: pytest.MonkeyPatch) -> None:
    config, model, engine, batch = production
    engine.global_step = 700
    original_schedule_step = engine.schedule.step_index

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("intentional forward failure")

    monkeypatch.setattr(engine, "_forward", fail)
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        engine.train_step(batch)
    assert engine.global_step == 700
    assert engine.schedule.step_index == original_schedule_step
    assert _progress(model) == _expected(config, 700)
