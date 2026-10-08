"""A single real attention key carries no query-ranking derivative.

These controls preserve the singleton source case; the complete-model
all-parameter-gradient gate uses two older keys so it cannot pass on rounding.
No optimizer-owned parameter is ignored or given an artificial residual.
"""

import pytest
import torch

from clearvla.policy.config import V39PolicyConfig
from clearvla.policy.proposal import RejectableHistoryProposal


@pytest.mark.parametrize("older", [1, 2, 7])
def test_history_summary_ranking_requires_more_than_one_key(older):
    torch.manual_seed(121)
    config = V39PolicyConfig(
        hidden_size=16,
        num_heads=4,
        executed_history_length=older + 1,
        action_history_enabled=1,
        action_history_recent_tokens=1,
        action_history_summary_tokens=1,
    )
    model = RejectableHistoryProposal(config)
    history = torch.randn(2, older + 1, config.action_dim)
    output = model.encode_history(history)
    output.square().mean().backward()
    norm = model.history_summary_qn
    assert norm is not None and norm.weight.grad is not None and norm.bias.grad is not None
    magnitude = norm.weight.grad.abs().max() + norm.bias.grad.abs().max()
    if older == 1:
        # FP32 roundoff is not evidence of an informative query derivative.
        assert magnitude < 1e-7
        with torch.no_grad():
            norm.weight.mul_(2)
            norm.bias.add_(0.5)
        torch.testing.assert_close(model.encode_history(history), output, rtol=1e-6, atol=1e-6)
    else:
        assert magnitude > 1e-4
    assert model.history_proj.weight.grad is not None
    assert model.history_proj.weight.grad.abs().sum() > 0
