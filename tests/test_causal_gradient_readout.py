"""Monitor the real typed producer, not the redundant common node's total VJP."""
from __future__ import annotations

import pytest
import torch
from test_mainline_task_execution_integration import _batch, config
from test_mainline_state_features import _model_engine
from clearvla.mainline.model.intent import _interval_common_residual
from clearvla.mainline.model.routing import register_gradient_rms_metric


def test_common_cancellation_is_not_a_missing_upstream_gradient():
    torch.manual_seed(901)
    weight=torch.randn(8,8,requires_grad=True)
    source=torch.randn(2,4,3,8)
    value=source@weight
    common,residual=_interval_common_residual(value)
    observed={}
    register_gradient_rms_metric(value,observed,'producer')
    register_gradient_rms_metric(common,observed,'common_total')
    (common[:,None]+residual).square().mean().backward()
    assert observed['common_total']==0
    assert observed['producer']>0 and weight.grad.abs().sum()>0


@pytest.mark.parametrize('amp',[False,True])
def test_real_s_observes_original_value_and_does_not_change_forward(amp):
    torch.manual_seed(903)
    model,_=_model_engine(config(amp));model.eval()
    batch=_batch();captured=[]
    organizer=model.intent.organizer
    original=organizer._typed_relevance
    def observe(**kwargs):
        result=original(**kwargs)
        result[1].retain_grad();captured.append(result[1])
        return result
    organizer._typed_relevance=observe
    try:
        rng=torch.get_rng_state().clone()
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
            cache,_,metrics=model.encode_online(batch.online,collect_diagnostics=True)
            c=cache.top.intent.typed_common_value
            residual=cache.top.intent.typed_interval_residual_value
            loss=(c[:,None]+residual).float().square().sum()
        loss.backward()
        value=captured[0]
        assert value.grad is not None and value.grad.float().abs().sum()>0
        actual=metrics['gradient_tensor_s_typed_value_predecomposition_rms']
        torch.testing.assert_close(actual,value.grad.float().square().mean().sqrt(),atol=0,rtol=0)
        for i,name in enumerate(('semantic','appearance','geometry')):
            grad=value.grad.select(3,i).float()
            torch.testing.assert_close(metrics[f'gradient_tensor_s_{name}_value_predecomposition_rms'],
                                      grad.square().mean().sqrt(),atol=0,rtol=0)
        # No dropout/RNG changes or replacement of ordinary gradients.
        assert torch.equal(rng,torch.get_rng_state())
        # Removing monitoring preserves the actual S data, not only its shape.
        organizer._typed_relevance=original
        with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp,cache_enabled=False):
            clean,_,no_metrics=model.encode_online(batch.online,collect_diagnostics=False)
        torch.testing.assert_close(clean.top.intent.public_interval_carrier,cache.top.intent.public_interval_carrier,atol=0,rtol=0)
        assert 'gradient_tensor_s_typed_value_predecomposition_rms' not in no_metrics
    finally:
        organizer._typed_relevance=original
