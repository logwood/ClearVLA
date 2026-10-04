from __future__ import annotations

import torch

from clearvla.mainline.model.intent import _interval_common_residual


def test_interval_common_residual_preserves_forward_and_common_vjp():
    torch.manual_seed(20261004)
    value = torch.randn(2, 4, 3, 5, requires_grad=True)
    common, residual = _interval_common_residual(value, preserve_common_grad=True)
    torch.testing.assert_close(common[:, None] + residual, value, rtol=0, atol=1e-6)
    loss = (common[:, None] + residual).square().mean()
    common_grad, residual_grad = torch.autograd.grad(loss, (common, residual))
    assert torch.isfinite(common_grad).all() and torch.count_nonzero(common_grad) > 0
    assert torch.isfinite(residual_grad).all() and torch.count_nonzero(residual_grad) > 0
