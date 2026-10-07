"""Test actual G/Teacher/S contracts, not only the matching kernel."""
from dataclasses import replace
import pytest
import torch
from test_mainline_instruction_posterior import _config as base_config
from test_mainline_operation_expectation import _batch
from test_mainline_state_features import _model_engine
from clearvla.mainline.model.observation_association import ObjectObservationAssociation


@pytest.fixture(scope='module')
def production():
    torch.manual_seed(7113)
    c=base_config()
    c=replace(c,top=replace(c.top,entity_transport_gradient_mode='ordinary_bilinear_v1',observation_measurement_mode='source_consistent_v1'))
    c.validate();m,e=_model_engine(c);b=_batch()
    m.eval()
    with torch.no_grad():cache,st,_=m.encode_online(b.online)
    return m,e,b,cache,st


def test_complete_static_measurement_at_all_offsets(production):
    m,e,b,cache,st=production;facts=st.top.facts
    association=ObjectObservationAssociation(content_dim=facts.content.shape[-1],camera_names=('top','wrist'),observation_measurement_mode='source_consistent_v1')
    current=facts.dense_chart.dino_content
    offsets=torch.tensor([0,4,8,16,24],device=current.device)
    measurement=association.measure_observations(facts=facts,observations=current[:,None].expand(-1,5,-1,-1,-1,-1),relative_offsets=offsets)
    torch.testing.assert_close(measurement.successor_per_support,measurement.current_reference.expand_as(measurement.successor_per_support),atol=2e-6,rtol=0)
    assert measurement.transport_per_support.abs().max()<2e-6
    assert measurement.covariance_per_support.abs().max()<2e-6
    assert torch.isfinite(measurement.null_probability).all()
    torch.testing.assert_close(measurement.candidate_posterior.flatten(3).sum(-1,keepdim=True)+measurement.null_probability,torch.ones_like(measurement.null_probability),atol=2e-6,rtol=0)


def test_complete_forward_backward_is_finite(production):
    m,e,b,cache,st=production
    m.train();result=e.train_step(b,collect_diagnostics=True)
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
