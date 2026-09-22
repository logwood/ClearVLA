"""M6g: exact executed-step support and real P3 use, not visual task progress.

External image/DINO/T5 fixtures are reused. Observer, model, loss, optimizer,
sampling, history, dataset, serialization and deployed adapter are real code.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_future_time_grid import _batch as old_batch
from test_mainline_instruction_reference import _adapter, _dataset, _Tokens
from test_mainline_p3_horizon_coordination import _config as base_config
from test_mainline_state_features import _model_engine
from torch.utils.data import default_collate

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.model.robot_execution import RobotExecutionObserver
from clearvla.mainline.robot_execution import (
    ONE_STEP_ROBOT_FEEDBACK,
    ExecutedRobotStep,
    RobotResponseFeedback,
    robot_execution_metadata,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory


def _config():
    c = base_config()
    return replace(
        c,
        top=replace(c.top, robot_feedback_mode=ONE_STEP_ROBOT_FEEDBACK),
        objectives=replace(c.objectives, robot_response=0.1),
    )


def _batch(center=16):
    b = old_batch(center)
    h = b.online.history
    step = ExecutedRobotStep(
        h.state.clone() - 0.01,
        h.executed_action_history[:, -1].clone(),
        torch.tensor([center > 0]),
        torch.tensor([[-1, -1, 0] if center else [0, 0, 0]]),
    )
    return replace(b, online=replace(b.online, history=replace(h, executed_robot_step=step)))


def _step(known=True):
    return ExecutedRobotStep(
        torch.randn(2, 10, requires_grad=True),
        torch.randn(2, 7, requires_grad=True),
        torch.full((2,), known, dtype=torch.bool),
        torch.tensor([[-1, -1, 0] if known else [0, 0, 0]]).expand(2, -1),
    )


def test_config_roundtrip_and_legacy_serialization():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert (
        load_config("configs/mainline/structural_rebuild_m6g_calvin.json").top.robot_feedback_mode
        == ONE_STEP_ROBOT_FEEDBACK
    )
    assert "robot_feedback_mode" not in cast(dict, base_config().as_dict()["top"])
    assert "robot_response" not in cast(dict, base_config().as_dict()["objectives"])
    assert "not-object" in cast(str, robot_execution_metadata()["scope"])


@pytest.mark.parametrize(
    "change",
    ["unknown", "old_p3", "untimed", "no_objective", "unselected_objective", "native_features"],
)
def test_config_rejects_semantic_mismatches(change):
    c = _config()
    if change == "unknown":
        c = replace(c, top=replace(c.top, robot_feedback_mode="guess"))
    if change == "old_p3":
        c = replace(c, top=replace(c.top, p3_coordination_mode="pointwise_legacy_v1"))
    if change == "untimed":
        c = replace(c, top=replace(c.top, history_encoding_mode="paired_rows_v1"))
    if change == "no_objective":
        c = replace(c, objectives=replace(c.objectives, robot_response=0))
    if change == "unselected_objective":
        c = replace(c, top=replace(c.top, robot_feedback_mode="none"))
    if change == "native_features":
        c = replace(c, top=replace(c.top, state_feature_mode="native_affine_v1"))
    with pytest.raises(ValueError):
        c.validate()


@pytest.mark.parametrize("offset", [[-4, -1, 0], [-1, 0, 0], [-1, -1, 1], [0, 0, 0]])
def test_previous_state_and_command_must_be_the_same_one_step_transition(offset):
    s = replace(_step(), offsets=torch.tensor([offset]).expand(2, -1))
    with pytest.raises(ValueError):
        s.validate(batch=2, state_dim=10, action_dim=7, device=torch.device("cpu"), strict=True)


@pytest.mark.parametrize("kind", ["missing", "command", "last_time", "absent"])
def test_real_encode_rejects_bad_causal_pair_before_neural_work(kind):
    b = _batch()
    h = b.online.history
    step = h.executed_robot_step
    assert step is not None and h.timing is not None
    if kind == "missing":
        h = replace(h, executed_robot_step=None)
    if kind == "command":
        h = replace(h, executed_robot_step=replace(step, command=step.command + 1))
    if kind == "last_time":
        assert h.timing is not None
        h = replace(h, timing=replace(h.timing, action_offsets=h.timing.action_offsets - 1))
    if kind == "absent":
        h = replace(h, executed_robot_step=step.without_actions(torch.zeros(1)))
    m, _ = _model_engine(_config())
    with pytest.raises(ValueError):
        m.encode_online(replace(b.online, history=h))


@pytest.mark.parametrize("bf16", [False, True])
def test_observer_supervision_gradients_and_feedback_detach(bf16):
    torch.manual_seed(6731)
    module = RobotExecutionObserver(state_dim=10, action_dim=7, hidden=16)
    s = _step()
    current = torch.randn(2, 10, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        feedback, loss = module.observe(s, current)
    assert not feedback.innovation.requires_grad
    loss.backward()
    assert current.grad is None and s.previous_state.grad is None and s.command.grad is None
    for p in module.response.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for group in (module.error_value, module.plan_query, module.error_output):
        assert all(p.grad is None for p in group.parameters())


@pytest.mark.parametrize("bf16", [False, True])
def test_unknown_nan_pair_has_zero_value_and_no_parameter_gradient(bf16):
    module = RobotExecutionObserver(state_dim=10, action_dim=7, hidden=16)
    s = _step(False)
    s = replace(
        s,
        previous_state=torch.full_like(s.previous_state, float("nan")),
        command=torch.full_like(s.command, float("nan")),
    )
    s.validate(batch=2, state_dim=10, action_dim=7, device=torch.device("cpu"), strict=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        f, loss = module.observe(s, torch.full((2, 10), float("nan")))
        query = torch.randn(2, 24, 3, 16, requires_grad=True)
        value = module.read(f, query)
    assert torch.count_nonzero(f.innovation) == torch.count_nonzero(value) == loss == 0
    (loss + value.float().sum()).backward()
    assert query.grad is not None and query.grad.count_nonzero() == 0
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in module.parameters())


def test_observed_nonfinite_pair_rejected_not_silently_zeroed():
    s = _step()
    s = replace(s, command=torch.full_like(s.command, float("nan")))
    with pytest.raises(ValueError):
        s.validate(batch=2, state_dim=10, action_dim=7, device=torch.device("cpu"), strict=True)


def test_current_observation_is_not_used_to_predict_itself():
    module = RobotExecutionObserver(state_dim=10, action_dim=7, hidden=16)
    s = _step()
    current = torch.randn(2, 10)
    a, _ = module.observe(s, current)
    b, _ = module.observe(s, current + 3)
    torch.testing.assert_close(b.innovation - a.innovation, torch.full_like(a.innovation, 3))
    assert not torch.equal(
        module.observe(replace(s, command=s.command + 1), current)[0].innovation, a.innovation
    )


def test_zero_error_has_zero_plan_context_gradient_even_for_an_observed_step():
    module = RobotExecutionObserver(state_dim=10, action_dim=7, hidden=16)
    q = torch.randn(2, 24, 3, 16, requires_grad=True)
    f = RobotResponseFeedback(torch.zeros(2, 10), torch.ones(2, dtype=torch.bool))
    x = module.read(f, q)
    x.sum().backward()
    assert x.count_nonzero() == 0 and q.grad is not None and q.grad.count_nonzero() == 0


@pytest.mark.parametrize("center", [0, 1, 16, 79])
def test_dataset_to_training_input_matches_source_predecessor_not_sparse_history(center):
    data = _dataset(0)
    data = ObservedStateWindowDataset(
        data.episodes,
        data.episode_ids,
        image_store=data.image_store,
        camera_names=data.camera_names,
        state_normalizer=data.state_normalizer,
        action_normalizer=data.action_normalizer,
        config=replace(data.config, robot_feedback_mode=ONE_STEP_ROBOT_FEEDBACK, world_horizon=24),
    )
    cached = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, _Tokens()))
    row = cached[center]
    b = to_training_batch(
        default_collate([row]),
        goal=GoalTemplate(torch.zeros(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {}),
        config=_config(),
        device=torch.device("cpu"),
    )
    b.validate(_config())
    step = b.online.history.executed_robot_step
    assert step is not None
    assert step.observed.item() == (center > 0)
    if center:
        torch.testing.assert_close(
            step.previous_state[0], data[center - 1]["state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            step.command[0], row["executed_action_history"][-1], rtol=0, atol=0
        )
        if center >= 4:
            assert not torch.equal(step.previous_state, b.online.history.state_history[:, 1])
    else:
        assert not step.offsets.any()


def test_online_history_retains_adjacent_state_across_instruction_and_capacity():
    hist = CausalHistory()

    def o(i):
        return PolicyObservation(
            rgb={"top": np.zeros((32, 32, 3), np.uint8), "wrist": np.zeros((32, 32, 3), np.uint8)},
            state=np.full(7, i, np.float32),
            action_state=np.zeros(7, np.float32),
        )

    hist.reset(o(0))
    assert hist.snapshot().previous_state is None
    for i in range(1, 38):
        hist.append(np.full(7, i - 1, np.float32), o(i))
        snap = hist.snapshot()
        np.testing.assert_array_equal(snap.previous_state, np.full(7, i - 1))
        assert snap.previous_state is not None
        snap.previous_state[:] = 999
        assert np.all(hist.snapshot().previous_state == i - 1)
    assert len(hist._observations) == 9 and len(hist._actions) == 24
    hist.reset(o(100))
    assert hist.snapshot().previous_state is None


@pytest.mark.parametrize("bf16", [False, True])
def test_production_training_loss_ownership_and_P3_read(bf16):
    torch.manual_seed(6732)
    c = _config()
    b = _batch()
    m, e = _model_engine(c)
    if bf16:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = e.train_step(b, collect_diagnostics=True)
    else:
        result = e.train_step(b, collect_diagnostics=True)
    assert result is not None
    observer = m.policy_compiler.plan_compiler.robot_observer
    assert observer is not None
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, st, _ = m.encode_online(b.online)
        targets, _ = m.build_training_targets(st, b.future)
        assert targets.robot_response_loss is st.robot_response_loss
        out = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
        loss = out.bottom.physical_velocity[:, 0].float().square().mean()
    loss.backward()
    assert all(p.grad is None for p in observer.response.parameters())
    for group in (observer.error_value, observer.plan_query, observer.error_output):
        for p in group.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.training_targets.parameters())


def test_response_computed_once_not_per_ODE_and_cache_is_readonly(monkeypatch):
    torch.manual_seed(6733)
    c = _config()
    b = _batch()
    m, e = _model_engine(c)
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    module = m.policy_compiler.plan_compiler.robot_observer
    assert module is not None
    calls = []
    old = module.observe

    def observe(*args):
        calls.append(args[0])
        return old(*args)

    monkeypatch.setattr(module, "observe", observe)
    with torch.no_grad():
        before = {k: v.clone() for k, v in m.named_buffers()}
        a = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(9))
        assert len(calls) == 1
        other = sample_action(m, b.online, c, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(a.action, other.action, rtol=0, atol=0)
    assert len(calls) == 2 and all(torch.equal(v, before[k]) for k, v in m.named_buffers())


def test_future_labels_cannot_modify_online_feedback_and_drop_history_cannot_bypass():
    c = _config()
    b = _batch()
    m, _ = _model_engine(c)
    m.eval()
    cache, st, _ = m.encode_online(b.online)
    feedback = cache.robot_feedback
    assert feedback is not None
    saved = feedback.innovation.clone()
    m.build_training_targets(st, replace(b.future, action_sequence=b.future.action_sequence + 3))
    assert torch.equal(feedback.innovation, saved)
    c = replace(c, top=replace(c.top, action_history_condition_dropout=0.99))
    m, _ = _model_engine(c)
    m.train()
    cache, st, _ = m.encode_online(
        b.online, training_mask=True, condition_generator=torch.Generator().manual_seed(0)
    )
    assert cache.robot_feedback is not None and cache.robot_feedback.innovation.count_nonzero() == 0
    assert st.robot_response_loss is not None and st.robot_response_loss == 0


def test_cache_rejects_missing_unselected_or_attached_feedback():
    c = _config()
    b = _batch()
    m, _ = _model_engine(c)
    m.eval()
    cache, _, _ = m.encode_online(b.online)
    with pytest.raises(ValueError):
        replace(cache, robot_feedback=None).validate(c)
    f = cache.robot_feedback
    assert f is not None
    with pytest.raises(ValueError):
        replace(
            cache, robot_feedback=replace(f, innovation=f.innovation.requires_grad_())
        ).validate(c)


def test_checkpoint_reload_and_declared_abi(tmp_path: Path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert abi["robot_execution"] == robot_execution_metadata()
        for k in ("missing", "bad_time"):
            bad = copy.deepcopy(abi)
            if k == "missing":
                del bad["robot_execution"]
            else:
                cast(dict, bad["robot_execution"])["clock"] = "four-step-mean"
            with pytest.raises(ValueError, match="robot execution"):
                validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_feedback_cannot_be_repaired_to_another_current_observation():
    c = _config()
    b = _batch()
    m, _ = _model_engine(c)
    m.eval()
    cache, _, _ = m.encode_online(b.online)
    f = cache.robot_feedback
    assert f is not None
    with pytest.raises(ValueError, match="another observation"):
        replace(
            cache,
            robot_feedback=replace(f, current_state=cast(torch.Tensor, f.current_state).clone()),
        ).validate(c)
    with pytest.raises(ValueError, match="another owner"):
        replace(cache, robot_feedback=replace(f, observed=f.observed.clone())).validate(c)


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostic_switch_preserves_feedback_loss_gradients_and_random_state(bf16):
    torch.manual_seed(6734)
    c = _config()
    b = _batch()
    m, _ = _model_engine(c)
    m.eval()
    results = []
    for diagnostic in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache, st, _ = m.encode_online(b.online, collect_diagnostics=diagnostic)
            assert st.robot_response_loss is not None and cache.robot_feedback is not None
            observer = m.policy_compiler.plan_compiler.robot_observer
            assert observer is not None
            params = list(observer.response.parameters())
            grads = torch.autograd.grad(st.robot_response_loss, params)
        results.append(
            (
                cache.robot_feedback.innovation,
                st.robot_response_loss.detach(),
                grads,
                torch.get_rng_state(),
            )
        )
    for i in (0, 1, 3):
        torch.testing.assert_close(results[0][i], results[1][i], rtol=0, atol=0)
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_online_offline_exact_pair_reset_and_repeated_instruction_agree(monkeypatch):
    from test_mainline_instruction_reference import _history

    policy, data, calls = _adapter(monkeypatch)
    # This adapter helper explicitly returns an external transport fixture bundle.
    cast(Any, policy.bundle).config = _config()
    # Reference/visual transport fixture only; actual CausalHistory and adapter.
    _, initial = policy.act_with_input(_history(data, 0), "push block")
    assert (
        initial.history.executed_robot_step is not None
        and not initial.history.executed_robot_step.observed.any()
    )
    _, online = policy.act_with_input(_history(data, 18), "push block")
    h = online.history
    step = h.executed_robot_step
    assert step is not None
    ds = _dataset(0)
    ds = ObservedStateWindowDataset(
        ds.episodes,
        ds.episode_ids,
        image_store=ds.image_store,
        camera_names=ds.camera_names,
        state_normalizer=ds.state_normalizer,
        action_normalizer=ds.action_normalizer,
        config=replace(ds.config, robot_feedback_mode=ONE_STEP_ROBOT_FEEDBACK, world_horizon=24),
    )
    raw = ds[18]
    torch.testing.assert_close(
        step.previous_state[0], raw["robot_step_previous_state"], rtol=0, atol=0
    )
    torch.testing.assert_close(step.command[0], raw["robot_step_command"], rtol=0, atol=0)
    policy.begin_instruction("push block")
    _, repeated = policy.act_with_input(_history(data, 18), "push block")
    assert repeated.history.executed_robot_step is not None
    assert (
        repeated.history.executed_robot_step.observed.all()
    )  # instruction boundary != physical reset
    torch.testing.assert_close(
        repeated.history.executed_robot_step.previous_state, step.previous_state, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="predecessor"):
        policy.act_with_input(replace(_history(data, 18), previous_state=None), "push block")
    assert len(calls) == 4  # encode only; no extra DINO pass for robot predecessor


def test_unsupported_batch_rows_do_not_dilute_response_objective():
    torch.manual_seed(6735)
    module = RobotExecutionObserver(state_dim=10, action_dim=7, hidden=16)
    s = _step()
    current = torch.randn(2, 10)
    partial = replace(
        s, observed=torch.tensor([True, False]), offsets=torch.tensor([[-1, -1, 0], [0, 0, 0]])
    )
    _, masked_loss = module.observe(partial, current)
    first = ExecutedRobotStep(s.previous_state[:1], s.command[:1], s.observed[:1], s.offsets[:1])
    _, one_loss = module.observe(first, current[:1])
    torch.testing.assert_close(masked_loss, one_loss)
    grads = torch.autograd.grad(masked_loss, tuple(module.response.parameters()), retain_graph=True)
    for actual, expected in zip(
        grads, torch.autograd.grad(one_loss, tuple(module.response.parameters()))
    ):
        torch.testing.assert_close(actual, expected)


def test_feedback_changes_current_velocity_without_changing_protected_fact_or_W():
    torch.manual_seed(6736)
    c = _config()
    b = _batch()
    m, e = _model_engine(c)
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        feedback = cache.robot_feedback
        assert feedback is not None
        neutral = replace(
            cache,
            robot_feedback=replace(feedback, innovation=torch.zeros_like(feedback.innovation)),
        )
        z, t = torch.randn(1, 24, 18), torch.tensor([0.35])
        actual = m.velocity(cache, noisy_action_field=z, time=t)
        other = m.velocity(neutral, noisy_action_field=z, time=t)
    assert not torch.equal(
        actual.bottom.physical_velocity[:, 0], other.bottom.physical_velocity[:, 0]
    )
    torch.testing.assert_close(
        actual.compiled.plan.protected_base, other.compiled.plan.protected_base, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.compiled.plan.temporal, other.compiled.plan.temporal, rtol=0, atol=0
    )
    assert cache.top is neutral.top and cache.factual_dock is neutral.factual_dock
