"""Padding must be removed before the first language projection, including VJP."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_endpoint_supervision import _config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine

from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import (
    ClearVLAMainlinePolicy,
    OnlinePolicyCache,
    OnlineTrainingState,
)
from clearvla.mainline.training.engine import MainlineTrainingEngine, validate_finite_training_batch

Production = tuple[ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch, tuple[OnlinePolicyCache, OnlineTrainingState]]


@pytest.fixture(scope="module")
def production() -> Production:
    torch.manual_seed(7102)
    model, engine = _model_engine(_config())
    batch = _batch()
    with torch.no_grad():
        cache, state, _ = model.encode_online(batch.online)
    return model, engine, batch, (cache, state)


def _goal_batch(batch: TrainingBatch, *, poison: float, all_padding: bool = False) -> TrainingBatch:
    goal = batch.online.goal
    mask = goal.mask.clone()
    if all_padding:
        mask.zero_()
    else:
        mask[:, -1] = False
    tokens = torch.where(mask[..., None], goal.tokens, poison).detach().requires_grad_()
    return replace(batch, online=replace(batch.online, goal=replace(goal, tokens=tokens, mask=mask)))


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), -float("inf"), 1e30])
@pytest.mark.parametrize("all_padding", [False, True])
def test_supported_preflight_ignores_padding_only(production: Production, poison: float, all_padding: bool) -> None:
    _, _, batch, _ = production
    validate_finite_training_batch(_goal_batch(batch, poison=poison, all_padding=all_padding))


def test_nonfinite_supported_token_is_still_rejected(production: Production) -> None:
    _, _, batch, _ = production
    goal = batch.online.goal
    changed = goal.tokens.clone()
    changed[:, 0] = float("nan")
    bad = replace(batch, online=replace(batch.online, goal=replace(goal, tokens=changed)))
    with pytest.raises(ValueError, match="online.goal"):
        validate_finite_training_batch(bad)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("all_padding", [False, True])
def test_actual_s_values_and_parameter_vjp_match_clean_padding(production: Production, bf16: bool, all_padding: bool) -> None:
    model, _, batch, (cache, state) = production
    organizer = model.intent.organizer
    # Use the real current-object source cached by the production G path.
    facts = state.top.facts
    h = cache.history
    results: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for poison in (0.0, float("nan")):
        changed = _goal_batch(batch, poison=poison, all_padding=all_padding)
        tokens = changed.online.goal.tokens
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output, _ = organizer(
                goal_tokens=tokens, goal_mask=changed.online.goal.mask,
                state_history=h.state_history, state=h.state,
                executed_history=h.executed_action_history, facts=facts,
                collect_diagnostics=False, history_timing=h.timing,
                instruction_reference=batch.online.instruction_reference,
                current_dino=batch.online.observation.dino_history[:, -1],
            )
            loss = output.public_interval_carrier.float().square().mean()
        loss.backward()
        grads = {name: p.grad.clone() for name, p in organizer.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(g).all() for g in grads.values())
        assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
        assert not torch.count_nonzero(tokens.grad[~changed.online.goal.mask])
        results.append((output.public_interval_carrier.detach().clone(), grads))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)


def test_real_optimizer_step_with_masked_nan_is_finite(production: Production) -> None:
    _, _, batch, _ = production
    model, engine = _model_engine(_config())
    result = engine.train_step(_goal_batch(batch, poison=float("nan")), collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
