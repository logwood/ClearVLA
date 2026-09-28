"""Learned task queries may address language; they must not invent its values."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from test_mainline_state_features import _model_engine
from test_mainline_task_execution_integration import _batch, config

from clearvla.mainline.model.intent import _CrossRead


@pytest.mark.parametrize("amp", [False, True])
def test_source_only_read_has_no_query_residual_or_query_ffn(amp):
    torch.manual_seed(44101)
    read = _CrossRead(16, 4, source_values_only=True)
    query = torch.randn(2, 4, 16, requires_grad=True)
    memory = torch.zeros(2, 7, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        value, innovation, _ = read(query, memory)
        empty, empty_delta, _ = read(query, memory,
                                     padding_mask=torch.ones(2, 7, dtype=torch.bool))
    for tensor in (value, innovation, empty, empty_delta):
        assert torch.count_nonzero(tensor) == 0
    value.float().sum().backward()
    assert query.grad is not None and torch.count_nonzero(query.grad) == 0
    assert memory.grad is not None and torch.isfinite(memory.grad).all()
    # This asserts an information-source contract, not zero robot actions.
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        real, _, _ = read(query.detach(), torch.randn_like(memory))
    assert torch.isfinite(real).all() and torch.count_nonzero(real) > 0


@pytest.mark.parametrize("amp", [False, True])
def test_zero_input_language_cannot_be_replaced_by_learned_task_scaffolds(amp):
    torch.manual_seed(4407)
    model, _ = _model_engine(config(amp))
    model.eval()
    batch = _batch()
    neutral_goal = replace(batch.online.goal, tokens=torch.zeros_like(batch.online.goal.tokens))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache, _, _ = model.encode_online(replace(batch.online, goal=neutral_goal))
        ordinary, _, _ = model.encode_online(batch.online)
    intent = cache.top.intent
    assert torch.count_nonzero(intent.protected_goal_set) == 0
    assert intent.task_relation is not None and torch.count_nonzero(intent.task_relation.values) == 0
    assert ordinary.top.intent.task_relation is not None
    assert torch.count_nonzero(ordinary.top.intent.task_relation.values) > 0
    binding = intent.target_binding
    assert binding is not None
    for mass, supported in zip(binding.mass, binding.supported, strict=True):
        available = mass[supported]
        if available.numel() > 1:
            torch.testing.assert_close(available, available[:1].expand_as(available), rtol=0, atol=0)
