"""Follow-up source-support tests for the recovered task-grounded graph.

Synthetic source fixtures exercise real modules; these do not load trained
CALVIN weights and do not claim task success or calibrated object identity.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from test_mainline_task_execution import fixture

from clearvla.mainline.future_time import CONTROL_INTERVALS
from clearvla.mainline.model.task_execution import TaskOutcomePlanRead
from clearvla.mainline.world_control import CandidateControlDomain


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("missing", ["view", "object_views", "control_interval"])
def test_unavailable_relation_is_quarantined_before_all_parameter_gradients(amp, missing):
    _, _, relation, expected, world = fixture()
    valid = relation.view_observed.clone()
    available = valid[:, None].expand(-1, 4, -1, -1).clone()
    if missing == "view":
        valid[:, 1, 0] = False
        available = valid[:, None].expand_as(available)
    elif missing == "object_views":
        valid[:, 1] = False
        available = valid[:, None].expand_as(available)
    else:
        world = replace(world, control_domain=CandidateControlDomain(8, CONTROL_INTERVALS))
        available[:, 2:] = False
    expected = replace(expected, view_observed=valid)
    raw = relation.values.detach()
    clean_values = raw.masked_fill(~available[..., None], 0).requires_grad_()
    bad_values = raw.masked_fill(~available[..., None], torch.nan).requires_grad_()
    clean = replace(relation, values=clean_values, view_observed=valid)
    bad = replace(relation, values=bad_values, view_observed=valid)
    model = TaskOutcomePlanRead(hidden=16, content_dim=12, heads=4, camera_names=relation.camera_names)
    other = copy.deepcopy(model)
    q = torch.randn(2, 24, 4, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        a = model(q, clean, expected, world)
        b = other(q, bad, expected, world)
        sum(t.float().square().sum() for t in a).backward()
        sum(t.float().square().sum() for t in b).backward()
    for x, y in zip(a, b, strict=True):
        assert torch.isfinite(y).all()
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert bad_values.grad is not None and torch.isfinite(bad_values.grad).all()
    assert torch.count_nonzero(bad_values.grad.masked_select(~available[..., None])) == 0
    for (name, x), (_, y) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        assert x.grad is not None and y.grad is not None, name
        assert torch.isfinite(y.grad).all(), name
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)


@pytest.mark.parametrize("source", ["current_content", "current_state"])
def test_operation_prediction_rejects_equal_shape_relation_from_other_observation(source):
    from test_mainline_state_features import _model_engine
    from test_mainline_task_execution_integration import _batch, config

    torch.manual_seed(17319)
    c = config()
    model, _ = _model_engine(c)
    model.eval()
    with torch.no_grad():
        cache, _, _ = model.encode_online(_batch().online)
        relation = cache.top.intent.task_relation
        assert relation is not None
        altered = replace(relation, **{source: getattr(relation, source).clone()})
        predictor = model.intent.organizer.operation_predictor
        assert predictor is not None
        inputs = dict(task_intervals=torch.randn(1, 4, c.dimensions.hidden_size),
                      facts=cache.top.belief, state=cache.history.state,
                      binding=relation.binding)
        predictor(**inputs, task_relation=relation)
        with pytest.raises(ValueError, match="another current observation"):
            predictor(**inputs, task_relation=altered)


@pytest.mark.parametrize("amp", [False, True])
def test_native_gripper_logits_use_joint_task_owners_after_one_real_update(amp):
    from test_mainline_state_features import _model_engine
    from test_mainline_task_execution_integration import _batch, config

    torch.manual_seed(18471)
    c = config(amp)
    model, engine = _model_engine(c)
    batch = _batch()
    # The inherited command head starts at zero. Its real supervised update
    # must open the upstream path; do not edit weights to manufacture a VJP.
    engine.train_step(batch)
    engine.optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        cache, _, _ = model.encode_online(batch.online)
        step = model.velocity(cache, noisy_action_field=torch.randn(1, 24, 18),
                              time=torch.full((1,), 0.4))
        logits = step.bottom.gripper_command_logits
        assert logits is not None and torch.isfinite(logits).all()
        (logits[:, (0, 23), 1].float() - logits[:, (0, 23), 0].float()).sum().backward()
    for owner in ("shared_binder", "task_relation_encoder", "intent.organizer.interval_object.",
                  "intent.coarse_action.object_read.", "p1.target_read.", "effect_reader.task_execution."):
        ps = [(name, p) for name, p in model.named_parameters() if owner in name]
        assert ps, owner
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in ps), owner
        assert sum(p.grad.float().abs().sum().item() for _, p in ps) > 0, owner
    assert all(p.grad is None for p in model.training_targets.parameters())
