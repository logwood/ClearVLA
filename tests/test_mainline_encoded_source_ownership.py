"""Reusable supervised graphs are tied to their causal observation and producer."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_endpoint_supervision import _config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine


@pytest.fixture(scope="module")
def production():
    torch.manual_seed(998)
    c = _config()
    model, engine = _model_engine(c)
    batch = _batch()
    encoded = engine.encode_eval(batch, collect_diagnostics=False)
    return model, engine, batch, encoded


@pytest.mark.parametrize("field", ["state", "goal", "image", "clock", "same_values_new_owner"])
def test_another_online_observation_is_rejected_before_teacher_or_flow(production, monkeypatch, field):
    model, engine, batch, encoded = production
    source = batch.online
    if field == "state":
        source = replace(source, history=replace(source.history, state=source.history.state + 0.1))
    elif field == "goal":
        source = replace(source, goal=replace(source.goal, tokens=source.goal.tokens + 0.1))
    elif field == "image":
        source = replace(source, observation=replace(source.observation, dino_history=source.observation.dino_history + 0.1))
    elif field == "clock":
        source = replace(source, history=replace(source.history))
    else:
        source = replace(source)
    changed = replace(batch, online=source)

    def forbidden(*a, **kw):
        raise AssertionError("Teacher must not run on a mismatched cache")
    monkeypatch.setattr(model, "build_training_targets", forbidden)
    generator = torch.Generator().manual_seed(881)
    old = generator.get_state().clone()
    with pytest.raises(ValueError, match="another online observation"):
        engine.eval_step(changed, encoded=encoded, generator=generator)
    assert torch.equal(old, generator.get_state())


def test_new_model_and_advanced_step_cannot_reuse_old_graph(production):
    model, engine, batch, encoded = production
    other, _ = _model_engine(_config())
    with pytest.raises(ValueError, match="model instance"):
        encoded.validate_for(batch, model=other, global_step=engine.global_step)
    with pytest.raises(ValueError, match="optimizer step"):
        encoded.validate_for(batch, model=model, global_step=engine.global_step + 1)


def test_train_eval_mismatch_rejected(production):
    model, engine, batch, encoded = production
    try:
        model.train()
        with pytest.raises(ValueError, match="train/eval mode"):
            encoded.validate_for(batch, model=model, global_step=engine.global_step)
    finally:
        model.eval()


def test_policy_and_teacher_planes_must_share_the_same_current_world(production):
    model, engine, batch, encoded = production
    other = engine.encode_eval(batch, collect_diagnostics=False)
    with pytest.raises(ValueError, match="different owners"):
        replace(encoded, training_state=other.training_state).validate_for(batch, model=model, global_step=engine.global_step)
    with pytest.raises(ValueError, match="different owners"):
        replace(encoded, cache=other.cache).validate_for(batch, model=model, global_step=engine.global_step)


def test_changing_future_labels_is_allowed_and_cannot_write_online_cache(production):
    model, engine, batch, encoded = production
    changed = replace(batch, future=replace(batch.future, state_sequence=batch.future.state_sequence + 0.4))
    before = encoded.cache.top.predicted_dynamics.semantic_delta.clone()
    result = engine.eval_step(changed, encoded=encoded, generator=torch.Generator().manual_seed(11))
    assert torch.isfinite(result.loss)
    assert changed.online is batch.online
    torch.testing.assert_close(before, encoded.cache.top.predicted_dynamics.semantic_delta, rtol=0, atol=0)


def test_repeated_valid_evaluation_reuses_cache_and_matches_rng(production):
    model, engine, batch, encoded = production
    a = engine.eval_step(batch, encoded=encoded, collect_diagnostics=False, generator=torch.Generator().manual_seed(45))
    b = engine.eval_step(batch, encoded=encoded, collect_diagnostics=False, generator=torch.Generator().manual_seed(45))
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)


def test_real_optimizer_step_invalidates_previous_cached_eval():
    model, engine = _model_engine(_config())
    batch = _batch()
    encoded = engine.encode_eval(batch, collect_diagnostics=False)
    result = engine.train_step(batch)
    assert torch.isfinite(result.loss)
    with pytest.raises(ValueError, match="optimizer step"):
        engine.eval_step(batch, encoded=encoded)
