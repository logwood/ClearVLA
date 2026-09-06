from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from test_mainline_policy import _batch, _config

from clearvla.mainline.model.intent import (
    StatelessObjectIntentOrganizer,
    _attention_probabilities,
    _CrossRead,
)
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy


def test_current_state_is_replaced_not_appended_and_last_delta_is_retained() -> None:
    history = torch.tensor([[[1.0], [3.0], [999.0]]], requires_grad=True)
    current = torch.tensor([[7.0]], requires_grad=True)
    actions = torch.arange(8.0).reshape(1, 8, 1)
    paired, delta = StatelessObjectIntentOrganizer._paired_history(history, current, actions)
    assert paired.shape == (1, 8, 4)
    assert torch.equal(paired[0, :, 0], torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 7.0]))
    assert torch.equal(paired[..., 1], actions[..., 0])
    assert torch.equal(delta[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0]))
    delta[:, -1].sum().backward()
    assert torch.equal(current.grad, torch.ones_like(current))
    assert torch.equal(history.grad, torch.tensor([[[0.0], [-1.0], [0.0]]]))
    _, stationary = StatelessObjectIntentOrganizer._paired_history(
        torch.ones(1, 3, 1), torch.ones(1, 1), actions
    )
    assert torch.count_nonzero(stationary) == 0


@pytest.mark.parametrize("which", ["state", "action"])
def test_intent_rejects_empty_history(which: str) -> None:
    history = torch.zeros(1, 0 if which == "state" else 3, 7)
    actions = torch.zeros(1, 0 if which == "action" else 8, 7)
    with pytest.raises(ValueError, match="causal row"):
        StatelessObjectIntentOrganizer._paired_history(history, torch.zeros(1, 7), actions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_s_diagnostics_preserve_forward_backward_and_rng(dtype: torch.dtype) -> None:
    torch.manual_seed(8101)
    block = _CrossRead(16, 4).train()
    query = torch.randn(2, 4, 16, requires_grad=True)
    memory = torch.randn(2, 6, 16, requires_grad=True)
    padding = torch.tensor([[False] * 5 + [True], [False] * 3 + [True] * 3])
    saved_rng = torch.get_rng_state().clone()
    outputs, gradients = [], []
    for diagnostic in (False, True):
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            output, innovation, probability = block(
                query, memory, padding_mask=padding, diagnostics=diagnostic
            )
        outputs.append((output, innovation))
        gradients.append(
            torch.autograd.grad(output.square().mean(), (query, memory, *block.parameters()))
        )
        assert torch.equal(torch.get_rng_state(), saved_rng)
    for left, right in zip(outputs[0], outputs[1]):
        assert torch.equal(left, right)
    for left, right in zip(*gradients):
        assert torch.equal(left, right)
    assert probability.dtype == torch.float32 and not probability.requires_grad
    assert (
        torch.count_nonzero(probability.masked_select(padding[:, None].expand_as(probability))) == 0
    )
    torch.testing.assert_close(probability.sum(-1), torch.ones(2, 4))


def test_s_attention_side_probability_matches_fp32_reference() -> None:
    torch.manual_seed(8102)
    attention = torch.nn.MultiheadAttention(16, 4, batch_first=True, bias=False)
    query, memory = torch.randn(2, 3, 16), torch.randn(2, 5, 16)
    padding = torch.tensor([[False] * 4 + [True], [False] * 5])
    _, expected = attention(query, memory, memory, key_padding_mask=padding, need_weights=True)
    actual = _attention_probabilities(attention, query, memory, padding)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_complete_s_keeps_all_non_audit_fields_bit_exact(dtype: torch.dtype) -> None:
    torch.manual_seed(8103)
    config = _config()
    model = ClearVLAMainlinePolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad(), torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        _, training, _ = model.encode_online(batch.online, collect_diagnostics=False)
        inputs = dict(
            goal_tokens=batch.online.goal.tokens,
            goal_mask=batch.online.goal.mask,
            state_history=batch.online.history.state_history,
            state=batch.online.history.state,
            executed_history=batch.online.history.executed_action_history,
            facts=training.top.facts,
        )
        without, _ = model.intent.organizer(**inputs, collect_diagnostics=False)
        with_audit, metrics = model.intent.organizer(**inputs, collect_diagnostics=True)
    for field in fields(without):
        if "attention" not in field.name:
            assert torch.equal(getattr(without, field.name), getattr(with_audit, field.name)), (
                field.name
            )
    assert metrics
