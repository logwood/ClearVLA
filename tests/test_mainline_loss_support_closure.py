"""Source-supported losses must ignore unavailable storage before nonlinearities."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from test_mainline_policy import _calvin_binary_config, _config
from torch import Tensor

from clearvla.mainline.model.action_codec import PhysicalActionFieldCodec
from clearvla.mainline.model.policy import PolicyStepOutput
from clearvla.mainline.training.losses import FlowMatchingState, execution_value_terms


def _case(*, binary: bool, tail: int | None = None):
    torch.manual_seed(987)
    c = _calvin_binary_config() if binary else _config()
    codec = PhysicalActionFieldCodec(action_dim=7, horizon=24)
    predicted = torch.randn(2, 3, 4, 24, 2, requires_grad=True)
    candidates = torch.randn(2, 3, 4, 24, 18)
    baseline = candidates[:, :, -1].clone()
    valid = torch.tensor([True, False, True, True]).expand(2, 3, 4).clone()
    mask = None if tail is None else (torch.arange(24)[None] < tail).expand(2, 24)
    target = torch.randn(2, 24, 18)
    f = FlowMatchingState(torch.full((2,), 0.2), target, target, target, target, row_valid=mask)
    return c, codec, predicted, candidates, baseline, valid, f


def _terms(case, *, predicted=None, candidates=None, baseline=None, valid=None, flow=None):
    c, codec, p, a, b, v, f = case
    tensors = {
        "evidence_mmd_it_execution_candidate_value_field": p if predicted is None else predicted,
        "evidence_mmd_it_dwell_candidate_pred_velocity": a if candidates is None else candidates,
        "evidence_mmd_it_execution_candidate_value_mask": v if valid is None else valid,
        "evidence_mmd_it_execution_baseline_pred_velocity": b if baseline is None else baseline,
    }
    output = cast(PolicyStepOutput, SimpleNamespace(bottom=SimpleNamespace(decoder_tensors=tensors)))
    return execution_value_terms(c, codec, output, f if flow is None else flow)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("tail", [None, 1, 7, 24])
def test_unavailable_candidate_storage_is_neutral_in_loss_metrics_and_vjp(binary, tail):
    case = _case(binary=binary, tail=tail)
    _, _, p, a, b, v, f = case
    reference = _terms(case)
    ref_grad = torch.autograd.grad(reference["execution_value"], p)[0]
    rows = torch.ones(2, 24, dtype=torch.bool) if f.row_valid is None else f.row_valid
    support = v[..., None, None] & rows[:, None, None, :, None]
    poison_p = torch.where(support, p.detach(), float("nan")).requires_grad_()
    poison_a = torch.where(support, a, float("inf"))
    poison_b = torch.where(rows[:, None, :, None], b, float("nan"))
    poison_f = replace(f, target_physical_velocity=torch.where(rows[..., None], f.target_physical_velocity, float("nan")))
    actual = _terms(case, predicted=poison_p, candidates=poison_a, baseline=poison_b, flow=poison_f)
    for name in reference:
        assert torch.isfinite(actual[name]).all(), name
        torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0, msg=name)
    grad = torch.autograd.grad(actual["execution_value"], poison_p)[0]
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
    assert torch.count_nonzero(grad.masked_select(~support.expand_as(grad))) == 0


@pytest.mark.parametrize("empty", ["candidates", "labels", "both"])
def test_no_support_has_zero_finite_objective_and_diagnostics(empty):
    case = _case(binary=False, tail=0 if empty != "candidates" else 24)
    _, _, p, a, b, v, f = case
    if empty != "labels":
        v = torch.zeros_like(v)
    poisoned = torch.full_like(p, float("nan"), requires_grad=True)
    actual = _terms(case, predicted=poisoned, candidates=torch.full_like(a, float("nan")), baseline=torch.full_like(b, float("nan")), valid=v)
    assert all(torch.isfinite(value).all() for value in actual.values())
    for name in ("execution_value", "execution_gripper_error_legacy", "execution_terminal_identity_error"):
        assert actual[name] == 0, name
    g = torch.autograd.grad(actual["execution_value"], poisoned)[0]
    assert torch.isfinite(g).all() and torch.count_nonzero(g) == 0


def test_available_nonfinite_values_are_not_hidden():
    case = _case(binary=False)
    p = case[2].detach().clone()
    p[:, :, 0] = float("nan")
    result = _terms(case, predicted=p)
    assert not torch.isfinite(result["execution_value"])


def test_gripper_audit_averages_real_candidates_not_unused_slots():
    case = _case(binary=False)
    full = _terms(case)
    indices = torch.tensor([0, 2, 3])
    reduced = _terms(case, predicted=case[2].index_select(2, indices), candidates=case[3].index_select(2, indices), valid=case[5].index_select(2, indices))
    for name in ("execution_gripper_error_legacy", "execution_gripper_error_deployed", "execution_gripper_error_compatibility", "execution_gripper_error_owned"):
        torch.testing.assert_close(full[name], reduced[name], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("bad", ["float", "shape"])
def test_source_rows_require_boolean_aligned_support(bad):
    case = _case(binary=False, tail=7)
    f = case[-1]
    assert f.row_valid is not None
    rows = f.row_valid.float() if bad == "float" else f.row_valid[:, :-1]
    with pytest.raises(ValueError, match="Boolean"):
        _terms(case, flow=replace(f, row_valid=rows))


@pytest.mark.parametrize("endpoint", [False, True])
@pytest.mark.parametrize("rows", [1, 7, 24])
def test_real_action_loss_quarantines_unknown_predictions_before_decode(endpoint, rows):
    from test_mainline_endpoint_supervision import _config as endpoint_config
    from test_mainline_operation_expectation import _batch
    from test_mainline_state_features import _model_engine

    from clearvla.mainline.endpoint_supervision import EndpointHeadSupervision, endpoint_condition
    from clearvla.mainline.supervision import FutureLabelSupport
    from clearvla.mainline.training.losses import action_terms, sample_flow_matching

    torch.manual_seed(991)
    c = endpoint_config()
    if not endpoint:
        c = replace(c, bottom=replace(c.bottom, endpoint_supervision_mode="interior_heads_v1"))
    m, _ = _model_engine(c)
    m.eval()
    b = _batch()
    n, t, _ = b.action_target.normalized.shape
    mask = (torch.arange(t)[None] < rows).expand(n, t)
    sup = FutureLabelSupport(mask, mask, b.future.offsets <= rows)
    target = replace(b.action_target, support=sup)
    flow = sample_flow_matching(target.normalized, row_valid=mask, action_state=b.online.history.action_state,
        codec_gripper_boundary=b.online.history.codec_gripper_boundary, codec=m.outlet_adapter.codec,
        distribution=c.bottom.flow_time_distribution, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online, collect_diagnostics=False)
        out = m.velocity(cache, noisy_action_field=flow.noisy_physical, time=flow.time, collect_diagnostics=False)

    def calculate(poison: bool):
        def leaf(value: Tensor) -> Tensor:
            safe = value.detach().clone()
            if poison:
                safe = torch.where(mask.reshape(*mask.shape, *((1,) * (safe.ndim-2))), safe, float("nan"))
            return safe.requires_grad_()
        velocity = leaf(out.bottom.physical_velocity)
        motion = leaf(out.bottom.motion_logits)
        assert out.bottom.gripper_command_logits is not None
        command = leaf(out.bottom.gripper_command_logits)
        altered = replace(out, bottom=replace(out.bottom, physical_velocity=velocity, motion_logits=motion, gripper_command_logits=command))
        time, ctx = endpoint_condition(velocity, context_enabled=True)
        ep = EndpointHeadSupervision(altered, time, ctx) if endpoint else None
        terms = action_terms(c, m.outlet_adapter, altered, target, b.online.history, flow, endpoint_supervision=ep)
        objective_names = ("action_flow", "decoded_action", "smooth_delta", "physical_delta_consistency", "gripper_command", "motion")
        loss = sum(terms[key] for key in objective_names)
        assert isinstance(loss, Tensor)
        gradients = torch.autograd.grad(loss, (velocity, motion, command))
        return terms, gradients

    a, ga = calculate(False)
    actual, gb = calculate(True)
    for name in a:
        if a[name].ndim == 0 and torch.isfinite(a[name]):
            assert torch.isfinite(actual[name]), name
            torch.testing.assert_close(actual[name], a[name], rtol=0, atol=0, msg=name)
    for aa, bb in zip(ga, gb, strict=True):
        assert torch.isfinite(bb).all()
        torch.testing.assert_close(aa, bb, rtol=0, atol=0)
