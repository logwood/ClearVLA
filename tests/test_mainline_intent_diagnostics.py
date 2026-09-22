"""Intent logging is a detached observer, never a switch for the value kernel."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from clearvla.mainline.model.intent import _CrossRead, _diagnostic_attention_weights


@pytest.mark.parametrize("autocast_enabled", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_cross_read_diagnostics_preserve_values_and_all_gradients(
    autocast_enabled: bool, masked: bool
) -> None:
    torch.manual_seed(1841)
    reader = _CrossRead(16, 4).eval()
    query = torch.randn(2, 3, 16, requires_grad=True)
    memory = torch.randn(2, 5, 16, requires_grad=True)
    mask = None
    if masked:
        mask = torch.tensor([[False, True, False, True, False], [True] * 5])
        with torch.no_grad():
            memory[mask] = float("nan")
    inputs = (query, memory, *reader.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        quiet, quiet_delta, quiet_weights = reader(query, memory, padding_mask=mask)
    quiet_rng = torch.random.get_rng_state().clone()
    quiet_grads = torch.autograd.grad(quiet.square().mean(), inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        traced, traced_delta, weights = reader(
            query, memory, padding_mask=mask, diagnostics=True
        )
    torch.testing.assert_close(quiet, traced, rtol=0, atol=0)
    torch.testing.assert_close(quiet_delta, traced_delta, rtol=0, atol=0)
    torch.testing.assert_close(quiet_rng, torch.random.get_rng_state(), rtol=0, atol=0)
    traced_grads = torch.autograd.grad(traced.square().mean(), inputs)
    for left, right in zip(quiet_grads, traced_grads, strict=True):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not weights.requires_grad and weights.dtype == torch.float32
    assert torch.count_nonzero(quiet_weights) == 0
    if mask is not None:
        assert torch.count_nonzero(weights[1]) == 0
        assert torch.count_nonzero(weights.masked_select(mask[:, None].expand_as(weights))) == 0
        torch.testing.assert_close(traced[1], query[1], rtol=0, atol=0)
        assert torch.count_nonzero(traced_delta[1]) == 0
    else:
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3), atol=1e-7, rtol=1e-7)


@torch.no_grad()
def test_diagnostic_probabilities_match_independent_multihead_reference() -> None:
    torch.manual_seed(1843)
    attention = nn.MultiheadAttention(16, 4, bias=False, dropout=0.0, batch_first=True).eval()
    query = torch.randn(2, 3, 16)
    key = torch.randn(2, 5, 16)
    padding = torch.tensor([[False, True, False, True, False], [True] * 5])
    key[padding] = float("nan")
    actual = _diagnostic_attention_weights(attention, query, key, padding)
    safe_key = torch.where(padding[..., None], torch.zeros_like(key), key)
    sentinel_mask = padding.clone()
    sentinel_mask[1, 0] = False
    _, expected = attention(
        query, safe_key, safe_key, key_padding_mask=sentinel_mask,
        need_weights=True, average_attn_weights=True,
    )
    assert expected is not None
    expected[1] = 0
    # This is a diagnostic probability reference, not value-path parity.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(actual).all() and not actual.requires_grad


@pytest.mark.parametrize("kind", ["bias", "dropout", "unbatched", "extra_key"])
def test_diagnostic_rejects_attention_variants_it_does_not_implement(kind: str) -> None:
    attention = nn.MultiheadAttention(
        16, 4, bias=kind == "bias", dropout=0.1 if kind == "dropout" else 0.0,
        batch_first=kind != "unbatched", add_zero_attn=kind == "extra_key",
    )
    with pytest.raises(ValueError, match="packed bias-free zero-dropout"):
        _diagnostic_attention_weights(attention, torch.zeros(1, 2, 16), torch.zeros(1, 3, 16))


@pytest.mark.parametrize("kind", ["float_mask", "mask_shape", "key_width", "empty_keys"])
def test_diagnostic_rejects_misaligned_inputs(kind: str) -> None:
    attention = nn.MultiheadAttention(16, 4, bias=False, dropout=0.0, batch_first=True)
    query = torch.zeros(1, 2, 16)
    key = torch.zeros(1, 0 if kind == "empty_keys" else 3, 15 if kind == "key_width" else 16)
    mask = torch.zeros(1, 4 if kind == "mask_shape" else 3, dtype=torch.bool)
    if kind == "float_mask":
        mask = mask.float()
    with pytest.raises(ValueError, match="intent diagnostic"):
        _diagnostic_attention_weights(attention, query, key, mask)
