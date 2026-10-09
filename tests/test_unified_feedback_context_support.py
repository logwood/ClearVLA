"""Absent execution evidence must be removed before the P3 context projection.

Artificial component sources; no learned checkpoint or closed-loop claim.
Known zero innovation stays distinct from missing observation, and finite valid
inputs retain the original numerical forward and ordinary derivatives.
"""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.model.robot_execution import RobotExecutionObserver
from clearvla.mainline.robot_execution import ExecutedRobotStep, RobotResponseFeedback


def robot_case():
    torch.manual_seed(1903)
    module = RobotExecutionObserver(state_dim=3, action_dim=2, hidden=8)
    support = torch.tensor([True, False])
    step = ExecutedRobotStep(
        torch.zeros(2, 3),
        torch.ones(2, 2),
        support,
        torch.tensor([[-1, -1, 0], [0, 0, 0]]),
    )
    return module, step


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("bf16", [False, True])
def test_robot_missing_context_is_quarantined_before_parameter_products(bad, bf16):
    module, step = robot_case()
    other = deepcopy(module)
    feedback, _ = module.observe(step, torch.ones(2, 3))
    clean = torch.randn(2, 4, 2, 8, requires_grad=True)
    dirty = clean.detach().clone()
    dirty[1] = bad
    dirty.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected = other.read(feedback, clean)
        actual = module.read(feedback, dirty)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and actual[1].count_nonzero() == 0
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    assert dirty.grad is not None and clean.grad is not None
    torch.testing.assert_close(dirty.grad, clean.grad, atol=0, rtol=0)
    assert dirty.grad[1].count_nonzero() == 0
    for name in ("error_value", "plan_query", "error_output"):
        got = getattr(module, name).weight.grad
        want = getattr(other, name).weight.grad
        assert got is not None and want is not None and torch.isfinite(got).all()
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("field", ["previous_state", "command", "current_state"])
def test_observer_rejects_nonfinite_observed_payload_before_prediction(field):
    module, step = robot_case()
    current = torch.ones(2, 3)
    if field == "current_state":
        current[0, 0] = torch.nan
    else:
        value = getattr(step, field).clone()
        value[0, 0] = torch.nan
        step = replace(step, **{field: value})
    calls = []
    hook = module.response.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        with pytest.raises(ValueError, match="finite"):
            module.observe(step, current)
    finally:
        hook.remove()
    assert not calls


def test_observer_rejects_wrong_step_clock_before_prediction():
    module, step = robot_case()
    step = replace(step, offsets=torch.tensor([[-4, -4, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match="one-step"):
        module.observe(step, torch.ones(2, 3))


def test_valid_robot_reader_has_original_forward_and_parameter_vjp():
    module, step = robot_case()
    other = deepcopy(module)
    f, _ = module.observe(step, torch.ones(2, 3))
    q = torch.randn(2, 4, 2, 8, requires_grad=True)
    q2 = q.detach().clone().requires_grad_()
    got = module.read(f, q)
    error = torch.where(f.observed[:, None], f.innovation, 0.0)
    wanted = other.error_output(
        other.error_value(error)[:, None, None] * torch.tanh(other.plan_query(q2))
    )
    torch.testing.assert_close(got, wanted, atol=0, rtol=0)
    got.square().sum().backward()
    wanted.square().sum().backward()
    torch.testing.assert_close(q.grad, q2.grad, atol=0, rtol=0)
    for name in ("error_value", "plan_query", "error_output"):
        torch.testing.assert_close(
            getattr(module, name).weight.grad, getattr(other, name).weight.grad, atol=0, rtol=0
        )


def test_robot_context_first_and_second_ordinary_derivatives():
    module, _ = robot_case()
    module.double()
    feedback = RobotResponseFeedback(torch.randn(1, 3, dtype=torch.double), torch.tensor([True]))
    q = torch.randn(1, 2, 1, 8, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: module.read(feedback, x), (q,))
    assert torch.autograd.gradgradcheck(lambda x: module.read(feedback, x), (q,))


@pytest.fixture(scope="module")
def world_sources():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from scripts.check_unified_model_flow import configuration

    config = configuration("small", "A", "conditional_object_v1")
    batch, normalizer = synthetic_batch(config, count=2, raw_side=32, device=torch.device("cpu"))
    module = ClearVLAMainlinePolicy(config).eval()
    module.configure_action_normalizer(normalizer)
    window = batch.online.history.executed_world_window
    assert window is not None
    mask = torch.tensor([True, False])
    window = replace(window, observed=mask)
    online = replace(
        batch.online, history=replace(batch.online.history, executed_world_window=window)
    )
    with torch.no_grad():
        cache, state, _ = module.encode_online(online)
    assert cache.world_feedback is not None
    return cache.world_feedback.feedback, state.top.facts, cache.top.intent.target_binding


@pytest.mark.parametrize("mode", ["innovation_only_v1", "innovation_and_status_v1"])
@pytest.mark.parametrize("bf16", [False, True])
def test_world_missing_context_and_vjp_respect_observed_window(world_sources, mode, bf16):
    from clearvla.mainline.model.executed_world import ExecutedWorldPlanRead

    feedback, facts, binding = world_sources
    reader = ExecutedWorldPlanRead(
        hidden=32,
        content_dim=facts.content.shape[-1],
        camera_names=("top", "wrist"),
        value_mode=mode,
    )
    if reader.observed_status_projection is not None:
        with torch.no_grad():
            reader.observed_status_projection.fill_(0.1)
    with torch.no_grad():
        prepared = reader.prepare(feedback, facts, binding)
    original = deepcopy(reader)
    control = replace(prepared, reader_identity=id(original))
    query = torch.randn(2, 4, 2, 32, requires_grad=True)
    bad = query.detach().clone()
    bad[1] = torch.nan
    bad.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        want = original(control, query, binding)
        got = reader(prepared, bad, binding)
    torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert got[1].count_nonzero() == 0 and torch.isfinite(got).all()
    got.float().square().sum().backward()
    want.float().square().sum().backward()
    torch.testing.assert_close(bad.grad, query.grad, atol=0, rtol=0)
    for name, p in reader.named_parameters():
        expected = dict(original.named_parameters())[name].grad
        if expected is None:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["innovation_only_v1", "innovation_and_status_v1"])
def test_world_finite_context_keeps_original_forward_and_vjp(world_sources, mode):
    from clearvla.mainline.model.executed_world import ExecutedWorldPlanRead

    feedback, facts, binding = world_sources
    reader = ExecutedWorldPlanRead(
        hidden=32,
        content_dim=facts.content.shape[-1],
        camera_names=("top", "wrist"),
        value_mode=mode,
    ).double()
    if reader.observed_status_projection is not None:
        with torch.no_grad():
            reader.observed_status_projection.fill_(0.1)
    with torch.no_grad():
        prepared = reader.prepare(feedback, facts, binding)
    original = deepcopy(reader)
    q = torch.randn(2, 2, 1, 32, dtype=torch.double, requires_grad=True)
    q2 = q.detach().clone().requires_grad_()
    actual = reader(prepared, q, binding)
    context = torch.tanh(original.context(q2))
    wanted = original.output(prepared.value.double()[:, None, None] * context)
    if original.observed_status_projection is not None:
        assert prepared.status_features is not None
        encoded = prepared.status_features.double() @ original.observed_status_projection
        wanted = wanted + original.output(encoded[:, None, None] * (1 + context))
    torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    actual.square().sum().backward()
    wanted.square().sum().backward()
    torch.testing.assert_close(q.grad, q2.grad, atol=0, rtol=0)
    for name, p in reader.named_parameters():
        expected = dict(original.named_parameters())[name].grad
        if expected is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, expected, atol=0, rtol=0)
    assert torch.autograd.gradcheck(lambda x: reader(prepared, x, binding), (q,))
    assert torch.autograd.gradgradcheck(lambda x: reader(prepared, x, binding), (q,))


def test_observed_world_zero_innovation_still_has_uncertainty_status(world_sources):
    from clearvla.mainline.model.executed_world import ExecutedWorldPlanRead

    feedback, facts, binding = world_sources
    reader = ExecutedWorldPlanRead(
        hidden=32,
        content_dim=facts.content.shape[-1],
        camera_names=("top", "wrist"),
        value_mode="innovation_and_status_v1",
    )
    with torch.no_grad():
        reader.observed_status_projection.fill_(0.1)
        reader.output.weight.copy_(torch.eye(32))
        prepared = reader.prepare(feedback, facts, binding)
    assert prepared.status_features is not None
    prepared = replace(prepared, value=torch.zeros_like(prepared.value))
    output = reader(prepared, torch.zeros(2, 2, 1, 32), binding)
    assert output[0].abs().sum() > 0
    assert output[1].count_nonzero() == 0
