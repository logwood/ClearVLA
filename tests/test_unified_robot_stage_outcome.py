"""Observed robot results enter the real S stage producer, never a success oracle.

Component counterexamples are artificial and use explicitly opened test weights.
Full-model tests use unmodified initialization and ordinary training losses.
"""

import gc
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from clearvla.mainline.model.observed_robot_outcome import ObservedRobotOutcomeRead
from clearvla.mainline.model.robot_execution import RobotExecutionObserver
from clearvla.mainline.robot_execution import ExecutedRobotStep, RobotResponseFeedback

MODE = "robot_world_before_proposal_v2"


def packet(delta=0.0, current=0.4, observed=True):
    mask = torch.tensor([observed])
    now = torch.full((1, 3), current)
    change = torch.full((1, 3), delta)
    step = ExecutedRobotStep(
        now - change,
        torch.tensor([[0.2, -0.3]]),
        mask,
        torch.tensor([[-1, -1, 0] if observed else [0, 0, 0]]),
    )
    return RobotResponseFeedback(
        torch.zeros_like(now),
        mask,
        step,
        now,
        observed_delta=change,
        predicted_delta=change.clone(),
    )


def activated_reader():
    torch.manual_seed(703)
    reader = ObservedRobotOutcomeRead(state_dim=3, action_dim=2, hidden=8)
    # Test-only non-neutral consumer; production always initializes output at zero.
    with torch.no_grad():
        torch.nn.init.normal_(reader.output.weight, std=0.1)
    return reader


def test_zero_innovation_does_not_erase_measured_motion_or_current_opening():
    r = activated_reader()
    q = torch.randn(1, 4, 8)
    stalled = packet(delta=0.0)
    moved = packet(delta=0.1)
    held = packet(delta=0.0, current=0.8)
    assert all(v.innovation.count_nonzero() == 0 for v in (stalled, moved, held))
    a, b, c = (r(v, q) for v in (stalled, moved, held))
    assert (a - b).abs().max() > 1e-5
    assert (a - c).abs().max() > 1e-5
    forecast_only = replace(stalled, predicted_delta=torch.ones(1, 3), innovation=-torch.ones(1, 3))
    torch.testing.assert_close(a, r(forecast_only, q), atol=0, rtol=0)


def test_absent_transition_quarantines_nan_source_and_context_before_projection():
    r = activated_reader()
    f = packet(observed=False)
    assert f.source_step is not None and f.current_state is not None
    f = replace(
        f,
        current_state=torch.full_like(f.current_state, torch.nan),
        source_step=replace(
            f.source_step,
            previous_state=torch.full_like(f.current_state, torch.nan),
            command=torch.full_like(f.source_step.command, torch.nan),
        ),
        observed_delta=torch.full_like(f.current_state, torch.nan),
    )
    q = torch.full((1, 4, 8), torch.nan, requires_grad=True)
    out = r(f, q)
    assert out.count_nonzero() == 0 and torch.isfinite(out).all()
    out.sum().backward()
    assert q.grad is not None and q.grad.count_nonzero() == 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero() == 0
        for p in r.parameters()
    )
    # Zero measured change with an actual predecessor is NOT absence.
    z = packet(delta=0.0, current=0.0)
    assert z.source_step is not None
    z = replace(z, source_step=replace(z.source_step, command=torch.zeros(1, 2)))
    assert r(z, torch.zeros(1, 4, 8)).abs().sum() > 0


def test_wrong_clock_support_and_observed_delta_are_rejected():
    r, f, q = activated_reader(), packet(), torch.ones(1, 4, 8)
    assert f.source_step is not None
    with pytest.raises(ValueError, match="actual measured"):
        r(replace(f, observed_delta=torch.ones(1, 3)), q)
    with pytest.raises(ValueError, match="owned"):
        r(replace(f, observed=f.observed.clone()), q)
    with pytest.raises(ValueError, match="one-step"):
        r(replace(f, source_step=replace(f.source_step, offsets=torch.tensor([[-4, -4, 0]]))), q)
    with pytest.raises(ValueError, match="sources"):
        r(replace(f, source_step=None), q)
    with pytest.raises(ValueError, match="nonfinite"):
        r(replace(f, current_state=torch.full((1, 3), torch.nan)), q)


def test_ordinary_context_derivative_and_no_task_gradient_into_response_predictor():
    r = activated_reader().double()
    q = torch.randn(1, 4, 8, dtype=torch.double, requires_grad=True)
    f = packet()
    assert torch.autograd.gradcheck(lambda x: r(f, x), (q,), atol=2e-5, rtol=2e-4)
    assert torch.autograd.gradgradcheck(lambda x: r(f, x), (q,), atol=2e-5, rtol=2e-4)
    observer = RobotExecutionObserver(state_dim=3, action_dim=2, hidden=8)
    assert f.source_step is not None and f.current_state is not None
    measured, response_loss = observer.observe(f.source_step, f.current_state)
    r.float()(measured, q.float()).square().sum().backward()
    assert all(p.grad is None for p in observer.response.parameters())
    response_loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in observer.response.parameters()
    )


def test_batch_and_interval_permutation_and_constructor_rng_are_explicit():
    torch.manual_seed(221)
    state = torch.random.get_rng_state().clone()
    r = ObservedRobotOutcomeRead(state_dim=3, action_dim=2, hidden=8)
    assert torch.equal(state, torch.random.get_rng_state())
    assert r.output.weight.count_nonzero() == 0
    r = activated_reader()
    f = packet(delta=0.1)
    q = torch.randn(1, 4, 8)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(r(f, q[:, order]), r(f, q)[:, order])
    assert f.source_step is not None and f.current_state is not None
    mask = f.observed.expand(2).clone()
    assert f.observed_delta is not None and f.predicted_delta is not None
    repeat = replace(
        f,
        observed=mask,
        current_state=f.current_state.expand(2, -1).clone(),
        source_step=replace(
            f.source_step,
            observed=mask,
            previous_state=f.source_step.previous_state.expand(2, -1).clone(),
            command=f.source_step.command.expand(2, -1).clone(),
            offsets=f.source_step.offsets.expand(2, -1).clone(),
        ),
        innovation=f.innovation.expand(2, -1).clone(),
        observed_delta=f.observed_delta.expand(2, -1).clone(),
        predicted_delta=f.predicted_delta.expand(2, -1).clone(),
    )
    q2 = torch.cat((q, 20 * q), 0)
    torch.testing.assert_close(r(repeat, q2)[:1], r(f, q), atol=2e-7, rtol=1e-6)


def test_config_ABI_and_initialization_require_the_new_mode(tmp_path):
    from test_mainline_global_task import abi_for

    from clearvla.mainline.config import config_from_mapping, load_config
    from clearvla.mainline.runtime.causal_identity_migration import (
        CAUSAL_INITIALIZATION_MODES,
        CAUSAL_UNIFIED_OUTCOME_V1,
        SOURCE_DIGEST,
        allowed_source_paths,
        validate_selection,
    )
    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    from clearvla.mainline.train import _parser
    from scripts.check_unified_model_flow import configuration

    c = configuration("small", "A", "conditional_object_v1", outcome_mode=MODE)
    assert config_from_mapping(c.as_dict()) == c
    _, abi, _ = abi_for(c, tmp_path)
    validate_deployment_abi(abi)
    damaged = deepcopy(abi)
    meta = damaged["causal_identity"]
    assert isinstance(meta, dict)
    meta.pop("robot_outcome")
    with pytest.raises(ValueError):
        validate_deployment_abi(damaged)
    for change in (
        {"robot_feedback_mode": "none"},
        {"typed_interval_gradient_mode": "legacy_common_surrogate_v1"},
        {"typed_object_value_mode": "legacy_selected_v1"},
    ):
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **change)).validate()
    saved = load_config("configs/mainline/dinov3_causal_repair_calvin_20261006.json")
    current = load_config("configs/mainline/unified_outcomes_b_calvin_check.json")
    for name in CAUSAL_INITIALIZATION_MODES:
        if name == CAUSAL_UNIFIED_OUTCOME_V1:
            validate_selection(saved, current, SOURCE_DIGEST, mode=name)
            assert "clearvla/mainline/model/observed_robot_outcome.py" in allowed_source_paths(name)
        else:
            with pytest.raises(ValueError):
                validate_selection(saved, current, SOURCE_DIGEST, mode=name)
            assert "clearvla/mainline/model/observed_robot_outcome.py" not in allowed_source_paths(
                name
            )
    args = _parser().parse_args(["--init-model-contract-migration", CAUSAL_UNIFIED_OUTCOME_V1])
    assert args.init_optimizer_state == "fresh"


@pytest.mark.parametrize("variant", ["A", "B"])
def test_real_policy_zero_start_parity_and_measured_feedback_before_coarse(variant):
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from clearvla.mainline.runtime.sampling import sample_action
    from scripts.check_unified_model_flow import configuration

    old_c = configuration("small", variant, "conditional_object_v1")
    new_c = replace(old_c, top=replace(old_c.top, observed_outcome_mode=MODE))
    batch, norm = synthetic_batch(new_c, count=1, raw_side=32, device=torch.device("cpu"))
    torch.manual_seed(112)
    old = ClearVLAMainlinePolicy(old_c)
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(112)
    new = ClearVLAMainlinePolicy(new_c)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    for name, p in old.state_dict().items():
        torch.testing.assert_close(p, new.state_dict()[name], atol=0, rtol=0)
    for m in (old, new):
        m.configure_action_normalizer(norm)
        m.eval()
    with torch.no_grad():
        a = sample_action(old, batch.online, old_c, generator=torch.Generator().manual_seed(310))
        b = sample_action(new, batch.online, new_c, generator=torch.Generator().manual_seed(310))
    torch.testing.assert_close(a.action, b.action, atol=0, rtol=0)
    del old, a, b
    gc.collect()
    events = []
    sources = []
    produced = []
    observer = new.policy_compiler.plan_compiler.robot_observer
    assert observer is not None
    observe = observer.observe

    def record_observation(*args, **kwargs):
        value, loss = observe(*args, **kwargs)
        produced.append(value)
        return value, loss

    observer.observe = record_observation
    reader = new.intent.organizer.observed_robot_outcome
    assert reader is not None
    handles = [
        reader.register_forward_pre_hook(
            lambda module, args: (events.append("robot"), sources.append(args[0])) and None
        ),
        new.intent.organizer.interval_self.register_forward_pre_hook(
            lambda *args: events.append("stage")
        ),
        new.intent.coarse_action.register_forward_pre_hook(lambda *args: events.append("coarse")),
    ]
    try:
        with torch.no_grad():
            sample_action(new, batch.online, new_c, generator=torch.Generator().manual_seed(310))
        assert events.count("robot") == events.count("stage") == events.count("coarse") == 1
        assert events.index("robot") < events.index("stage") < events.index("coarse")
        assert len(produced) == 1 and sources[0] is produced[0]
        assert sources[0].source_step is not None
        assert batch.online.history.executed_robot_step is not None
        torch.testing.assert_close(
            sources[0].source_step.command, batch.online.history.executed_robot_step.command
        )
        assert sources[0].current_state is batch.online.history.state
    finally:
        observer.observe = observe
        for h in handles:
            h.remove()
        del new
        gc.collect()


@pytest.mark.parametrize("variant", ["A", "B"])
def test_ordinary_training_updates_every_new_parameter_and_full_sample(variant):
    from scripts.check_unified_model_flow import run

    result = run(
        "small",
        variant,
        updates=2,
        raw_side=32,
        completed_step=1200,
        typed_object_values="conditional_object_v1",
        outcome_mode=MODE,
    )
    entries = {r["name"]: r for r in result["parameter_ledger"][-1]["entries"]}
    new = {n: v for n, v in entries.items() if "observed_robot_outcome." in n}
    assert len(new) == 3
    for name, row in new.items():
        assert row["gradient_status"] == "nonzero", name
        assert row["gradient_l2"] > 0 and row["update_l2"] > 0, name
    assert result["observed_outcome_mode"] == MODE and result["sampled_action_shape"] == [1, 24, 7]
    assert not result["real_data_passed"] and not result["behavior_passed"]


def test_action_loss_reaches_measured_result_consumer_not_response_predictor():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.qualification import synthetic_batch
    from clearvla.mainline.training.engine import MainlineTrainingEngine
    from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
    from scripts.check_unified_model_flow import configuration

    torch.manual_seed(1281)
    c = configuration("small", "B", "conditional_object_v1", outcome_mode=MODE)
    # No-source dropout cannot qualify a positive source-path VJP. This fixture
    # explicitly keeps it, while other tests keep the production dropout settings.
    c = replace(
        c, top=replace(c.top, action_history_condition_dropout=0.0, goal_condition_dropout=0.0)
    )
    batch, normalizer = synthetic_batch(c, count=1, raw_side=32, device=torch.device("cpu"))
    model = ClearVLAMainlinePolicy(c)
    model.configure_action_normalizer(normalizer)
    optimizer, _ = build_optimizer(model, c)
    schedule = WarmupCosineSchedule(
        optimizer, warmup_steps=2, total_steps=1204, minimum_ratio=0.1, update_origin=1200
    )
    engine = MainlineTrainingEngine(
        model=model, config=c, optimizer=optimizer, schedule=schedule, device=torch.device("cpu")
    )
    engine.global_step = 1200
    model.set_training_step(1200)
    engine.train_step(batch)
    engine.train_step(batch)
    model.eval()
    reader = model.intent.organizer.observed_robot_outcome
    observer = model.policy_compiler.plan_compiler.robot_observer
    assert reader is not None and observer is not None
    values = tuple(reader.parameters())
    predictor = tuple(observer.response.parameters())
    ledger, _ = engine._forward(
        batch,
        training=False,
        collect_diagnostics=False,
        generator=torch.Generator().manual_seed(1101),
    )
    grad = torch.autograd.grad(
        ledger.terms["action_arm_flow"], (*values, *predictor), allow_unused=True
    )
    for g in grad[: len(values)]:
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert all(g is None for g in grad[len(values) :])


def test_migration_retains_exact_common_weights_and_rejects_non_neutral_new_head():
    from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
    from clearvla.mainline.runtime.causal_identity_migration import migrate_state
    from scripts.check_unified_model_flow import configuration

    config = configuration("small", "A", "conditional_object_v1", outcome_mode=MODE)
    current = ClearVLAMainlinePolicy(config).state_dict()
    added = {
        name
        for name in current
        if ".observed_outcome." in name or ".observed_robot_outcome." in name
    }
    assert len(added) == 8
    saved = {name: value.clone() for name, value in current.items() if name not in added}
    saved.update(
        {
            f"intent.organizer.instruction_progress.{name}": torch.zeros(1)
            for name in (
                "key.weight",
                "key_position.weight",
                "null_key",
                "query.weight",
                "query_position.weight",
            )
        }
    )
    moved = migrate_state(saved, current, config)
    assert moved.keys() == current.keys()
    for name in current.keys() - added:
        assert moved[name] is saved[name]
    damaged = {
        **current,
        "intent.organizer.observed_robot_outcome.output.weight": torch.ones_like(
            current["intent.organizer.observed_robot_outcome.output.weight"]
        ),
    }
    with pytest.raises(ValueError, match="neutral"):
        migrate_state(saved, damaged, config)
    damaged = {**current}
    damaged.pop("intent.organizer.observed_robot_outcome.context.weight")
    with pytest.raises(ValueError, match="inventory"):
        migrate_state(saved, damaged, config)
