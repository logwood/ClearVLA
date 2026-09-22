"""Production S expectation targets, typed readers, and online/future separation."""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import pytest
import torch
from test_mainline_instruction_change import _batch
from test_mainline_instruction_change import _config as base_config
from test_mainline_state_features import _model_engine
from torch import Tensor

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.operation_expectation import (
    ObjectOperationPredictor,
    OperationExpectationPlanRead,
    OperationExpectationRead,
    OperationExpectationSupervisor,
)
from clearvla.mainline.model.policy import (
    ClearVLAMainlinePolicy,
    OnlinePolicyCache,
    OnlineTrainingState,
)
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.model.types import (
    FutureObjectDynamics,
    ObjectTopTrainingTargets,
    PhysicalActionSequenceCondition,
)
from clearvla.mainline.operation_expectation import (
    OBJECT_OUTCOME_INTENT,
    OperationExpectation,
    operation_expectation_metadata,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.supervision import FutureLabelSupport
from clearvla.mainline.training.engine import MainlineTrainingEngine

Production = tuple[
    ClearVLAMainlinePolicy,
    MainlineTrainingEngine,
    TrainingBatch,
    OnlinePolicyCache,
    OnlineTrainingState,
    ObjectTopTrainingTargets,
]


class LossInputs(TypedDict):
    expectation: OperationExpectation
    teacher: FutureObjectDynamics
    current_loss_support: Tensor
    future_state: Tensor
    label_support: FutureLabelSupport | None
    future_interval_valid: Tensor | None


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, top=replace(c.top, operation_intent_mode=OBJECT_OUTCOME_INTENT))


@pytest.fixture(scope="module")
def production() -> Production:
    torch.manual_seed(6811)
    m, e = _model_engine(_config())
    b = _batch()
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    with torch.no_grad():
        cache, st, _ = m.encode_online(b.online)
        targets, _ = m.build_training_targets(st, b.future, collect_diagnostics=True)
    return m, e, b, cache, st, targets


def _value() -> OperationExpectation:
    torch.manual_seed(6810)
    b, k, c, d, s = 2, 3, 2, 16, 10
    binding = TargetBinding.from_logits(
        torch.randn(b, k), torch.randn(b, 1), torch.ones(b, k, dtype=torch.bool)
    )
    return OperationExpectation(
        torch.randn(b, 4, k, d),
        torch.randn(b, 4, k, c, 2),
        torch.randn(b, 4, s),
        torch.ones(b, k, c, dtype=torch.bool),
        binding,
        torch.randn(b, s),
        torch.randn(b, k, d),
        ("top", "wrist"),
    )


def _reader(names: tuple[str, ...] = ("top", "wrist")) -> OperationExpectationRead:
    return OperationExpectationRead(hidden=16, content_dim=16, state_dim=10, camera_names=names)


def _loss_inputs(production: Production) -> LossInputs:
    _, _, b, cache, _, targets = production
    op = cache.top.intent.operation_expectation
    teacher = targets.teacher_dynamics
    assert op is not None and teacher is not None
    return LossInputs(
        expectation=op,
        teacher=teacher,
        current_loss_support=targets.current_loss_support,
        future_state=b.future.state_sequence,
        label_support=b.future.support,
        future_interval_valid=targets.future_interval_valid,
    )


def test_config_roundtrip_and_source_closure() -> None:
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "operation_intent_mode" not in cast(dict, base_config().as_dict()["top"])
    assert (
        load_config("configs/mainline/structural_rebuild_m6i_calvin.json").top.operation_intent_mode
        == OBJECT_OUTCOME_INTENT
    )
    paths = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/model/operation_expectation.py" in paths
    assert "clearvla/mainline/operation_expectation.py" in paths


@pytest.mark.parametrize("kind", ["unknown", "local", "old_time", "pointwise"])
def test_config_rejects_unsupported_graph(kind: str) -> None:
    fields = {
        "unknown": dict(operation_intent_mode="guess"),
        "local": dict(target_binding_mode="reader_local_v1"),
        "old_time": dict(future_time_grid_mode="legacy_48_v1"),
        "pointwise": dict(p3_coordination_mode="pointwise_legacy_v1"),
    }
    with pytest.raises(ValueError):
        replace(_config(), top=replace(_config().top, **fields[kind])).validate()


@pytest.mark.parametrize(
    "kind", ["view_axis", "robot_axis", "mask_type", "dtype", "camera", "time"]
)
def test_type_rejects_corruption(kind: str) -> None:
    v = _value()
    fields = {
        "view_axis": dict(image_delta=v.image_delta[:, :, :, 0]),
        "robot_axis": dict(robot_delta=v.robot_delta[:, :, :7]),
        "mask_type": dict(view_observed=v.view_observed.float()),
        "dtype": dict(semantic_delta=v.semantic_delta.bfloat16()),
        "camera": dict(camera_names=("top", "top")),
        "time": dict(time_grid_mode="legacy_48_v1"),
    }
    with pytest.raises((TypeError, ValueError)):
        replace(v, **fields[kind]).validate(strict=True)


def test_predictor_does_not_receive_future_labels_or_candidate_action() -> None:
    assert set(inspect.signature(ObjectOperationPredictor.forward).parameters) == {
        "self",
        "task_intervals",
        "facts",
        "state",
        "binding",
    }
    assert not list(OperationExpectationSupervisor().parameters())


def test_no_detached_learned_target_recognizer_in_new_graph(production: Production) -> None:
    m, _, _, cache, _, targets = production
    assert isinstance(m.training_targets.recognizer, OperationExpectationSupervisor)
    assert not list(m.training_targets.recognizer.parameters())
    assert targets.plan_recognition is None and targets.plan_recognition_loss == 0
    assert (
        targets.operation_terms is not None and cache.top.intent.operation_expectation is not None
    )
    assert all(
        not t.requires_grad for name, t in targets.operation_terms.items() if name.endswith("count")
    )
    assert targets.teacher_dynamics is not None
    assert targets.teacher_dynamics.camera_names == ("top", "wrist")


def test_null_mass_applied_only_after_typed_value_read() -> None:
    v, r = _value(), _reader()
    original = r(v)
    mass = v.binding.mass * 0.5
    law = TargetBinding(
        torch.cat((mass, 1 - mass.sum(-1, keepdim=True)), -1).log(), v.binding.supported
    )
    torch.testing.assert_close(r(replace(v, binding=law)), original * 0.5, rtol=1e-6, atol=1e-6)
    zero = torch.cat((torch.zeros_like(mass), torch.ones(mass.shape[0], 1)), -1).log()
    assert r(replace(v, binding=replace(law, log_probability=zero))).count_nonzero() == 0


def test_null_binding_cannot_remove_supervision_or_its_gradient(production: Production) -> None:
    kw = _loss_inputs(production)
    op = kw["expectation"]
    supervisor = OperationExpectationSupervisor()
    prediction = op.semantic_delta.clone().requires_grad_()
    op = replace(op, semantic_delta=prediction)
    ref = supervisor(**{**kw, "expectation": op})
    log = torch.cat(
        (torch.full_like(op.binding.mass, -torch.inf), torch.zeros(op.semantic_delta.shape[0], 1)),
        -1,
    )
    null = replace(op.binding, log_probability=log)
    result = supervisor(**{**kw, "expectation": replace(op, binding=null)})
    for k in ref:
        torch.testing.assert_close(ref[k], result[k], rtol=0, atol=0)
    result["operation_semantic"].backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_object_errors_are_not_averaged_before_loss(production: Production) -> None:
    kw = _loss_inputs(production)
    op = kw["expectation"]
    teacher = kw["teacher"]
    assert teacher.semantic_delta.shape[2] >= 2
    truth = teacher.semantic_delta.clone()
    truth[:, :, 0] = 1.0
    truth[:, :, 1] = -1.0
    wrong = replace(op, semantic_delta=torch.zeros_like(truth))
    result = OperationExpectationSupervisor()(
        **{**kw, "expectation": wrong, "teacher": replace(teacher, semantic_delta=truth)}
    )
    assert result["operation_semantic"] > 0.1


def test_zero_expectations_cannot_be_manufactured_from_plan_context() -> None:
    v = _value()
    v = replace(
        v,
        semantic_delta=torch.zeros_like(v.semantic_delta),
        image_delta=torch.zeros_like(v.image_delta),
        robot_delta=torch.zeros_like(v.robot_delta),
    )
    r = OperationExpectationPlanRead(
        hidden=16, content_dim=16, state_dim=10, camera_names=v.camera_names
    )
    q = torch.randn(2, 24, 3, 16, requires_grad=True)
    out = r(v, q)
    out.sum().backward()
    assert out.count_nonzero() == 0 and q.grad is not None and q.grad.count_nonzero() == 0


def test_camera_and_object_permutation() -> None:
    v, r = _value(), _reader()
    expected = r(v)
    k = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        r(v.permute(k, v.binding.permute(k))), expected, rtol=1e-5, atol=1e-6
    )
    rr = _reader(("wrist", "top"))
    rr.load_state_dict(r.state_dict(), strict=True)
    view = torch.tensor([1, 0])
    p = replace(
        v,
        image_delta=v.image_delta[:, :, :, view],
        view_observed=v.view_observed[:, :, view],
        camera_names=("wrist", "top"),
    )
    torch.testing.assert_close(rr(p), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="camera"):
        r(p)


def test_invalid_view_NaNs_do_not_reach_values_or_gradients() -> None:
    v, r = _value(), _reader()
    valid = v.view_observed.clone()
    valid[:, :, 1] = False
    xy = v.image_delta.clone()
    xy[:, :, :, 1] = float("nan")
    v = replace(v, view_observed=valid, image_delta=xy)
    out = r(v)
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())


def test_visual_and_robot_target_support_are_independent(production: Production) -> None:
    kw = _loss_inputs(production)
    b = production[2]
    assert b.future.support is not None
    invisible = replace(b.future.support, visual=torch.zeros_like(b.future.support.visual))
    no_visual = OperationExpectationSupervisor()(
        **{
            **kw,
            "label_support": invisible,
            "future_interval_valid": torch.zeros(1, 4, dtype=torch.bool),
        }
    )
    assert no_visual["operation_semantic_count"] == no_visual["operation_image_count"] == 0
    assert no_visual["operation_robot_count"] == 4
    absent_state = replace(b.future.support, state=torch.zeros_like(b.future.support.state))
    no_robot = OperationExpectationSupervisor()(
        **{
            **kw,
            "label_support": absent_state,
            "future_state": torch.full_like(b.future.state_sequence, float("nan")),
        }
    )
    assert no_robot["operation_robot_count"] == no_robot["operation_robot"] == 0
    assert no_robot["operation_image_count"] > 0


def test_absent_future_action_does_not_erase_observed_outcome(production: Production) -> None:
    kw = _loss_inputs(production)
    support = kw["label_support"]
    assert support is not None
    no_action = replace(support, action=torch.zeros_like(support.action))
    a = OperationExpectationSupervisor()(**kw)
    b = OperationExpectationSupervisor()(**{**kw, "label_support": no_action})
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_future_target_has_no_gradient_and_no_loss_gate_from_teacher_confidence(
    production: Production,
) -> None:
    kw = _loss_inputs(production)
    op = kw["expectation"]
    teacher = kw["teacher"]
    label = teacher.semantic_delta.clone().requires_grad_()
    state = kw["future_state"].clone().requires_grad_()
    pred = op.semantic_delta.clone().requires_grad_()
    target = replace(teacher, semantic_delta=label)
    loss = OperationExpectationSupervisor()(
        **{
            **kw,
            "teacher": target,
            "future_state": state,
            "expectation": replace(op, semantic_delta=pred),
        }
    )["operation_total"]
    loss.backward()
    assert label.grad is None and state.grad is None and pred.grad is not None


@pytest.mark.parametrize("kind", ["binding", "state", "content", "camera", "missing"])
def test_cache_rejects_wrong_owner(production: Production, kind: str) -> None:
    m, _, _, cache, _, _ = production
    op = cache.top.intent.operation_expectation
    assert op is not None
    if kind == "binding":
        op = replace(op, binding=replace(op.binding))
    elif kind == "state":
        op = replace(op, current_state=op.current_state.clone())
    elif kind == "content":
        op = replace(op, current_content=op.current_content.clone())
    elif kind == "camera":
        op = replace(op, camera_names=("wrist", "top"))
    else:
        op = None
    bad = replace(
        cache, top=replace(cache.top, intent=replace(cache.top.intent, operation_expectation=op))
    )
    with pytest.raises(ValueError):
        bad.validate(m.config)


def test_intent_permutation_and_policy_dock_keep_binding_owner(production: Production) -> None:
    _, _, _, cache, _, _ = production
    p = cache.top.intent.permute(torch.tensor([2, 0, 3, 1]))
    p.validate(horizon=24, hidden=p.object_tokens.shape[-1])
    assert (
        p.operation_expectation is not None and p.operation_expectation.binding is p.target_binding
    )
    assert p.policy_dock().operation_expectation is p.operation_expectation


@pytest.mark.parametrize("bf16", [False, True])
def test_real_optimizer_and_row_zero_action_gradient(bf16: bool) -> None:
    torch.manual_seed(6815)
    m, e = _model_engine(_config())
    b = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        e.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = m.encode_online(b.online)
        v = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
        loss = v.bottom.physical_velocity[:, 0].float().square().mean()
    loss.backward()
    for module in (
        m.intent.organizer.operation_predictor,
        m.intent.organizer.operation_read,
        m.policy_compiler.plan_compiler.operation_read,
    ):
        assert module is not None
        for name, p in module.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
                name
            )
    assert all(p.grad is None for p in m.training_targets.parameters())


def test_each_type_has_independent_direct_P3_effect(production: Production) -> None:
    m, _, _, cache, _, _ = production
    op = cache.top.intent.operation_expectation
    r = m.policy_compiler.plan_compiler.operation_read
    assert op is not None and r is not None
    q = torch.randn(1, 24, 3, m.config.dimensions.hidden_size)
    ref = r(op, q)
    for name in ("semantic_delta", "image_delta", "robot_delta"):
        changed = replace(op, **{name: getattr(op, name) + 0.3})
        assert not torch.equal(r(changed, q), ref), name


def test_expected_values_affect_coarse_actual_consumer(production: Production) -> None:
    m, _, b, _, _, _ = production
    r = m.intent.organizer.operation_read
    assert r is not None
    with torch.no_grad():
        normal, _, _ = m.encode_online(b.online)

        def zero_output(module: torch.nn.Module, args: tuple[object, ...], out: object) -> Tensor:
            assert isinstance(out, Tensor)
            return torch.zeros_like(out)

        handle = r.register_forward_hook(zero_output)
        try:
            zero, _, _ = m.encode_online(b.online)
        finally:
            handle.remove()
    assert isinstance(normal.top.action_condition, PhysicalActionSequenceCondition)
    assert isinstance(zero.top.action_condition, PhysicalActionSequenceCondition)
    assert not torch.equal(
        normal.top.action_condition.source_action, zero.top.action_condition.source_action
    )


def test_future_targets_do_not_change_online_values_or_buffers(production: Production) -> None:
    m, _, b, cache, st, _ = production
    before = {n: v.clone() for n, v in m.named_buffers()}
    noise = torch.randn(1, 24, 18)
    t = torch.tensor([0.4])
    with torch.no_grad():
        v0 = m.velocity(cache, noisy_action_field=noise, time=t).bottom.physical_velocity
        future = replace(
            b.future,
            state_sequence=b.future.state_sequence + 10,
            dino_supports=b.future.dino_supports - 7,
        )
        m.build_training_targets(st, future, collect_diagnostics=True)
        v1 = m.velocity(cache, noisy_action_field=noise, time=t).bottom.physical_velocity
    torch.testing.assert_close(v0, v1, rtol=0, atol=0)
    assert all(torch.equal(v, before[n]) for n, v in m.named_buffers())


def test_sampling_reuses_single_expectation_not_each_ode(production: Production) -> None:
    m, _, b, _, _, _ = production
    count = []
    r = m.intent.organizer.operation_predictor
    assert r is not None
    h = r.register_forward_hook(lambda *args: count.append(1))
    try:
        a = sample_action(m, b.online, m.config, generator=torch.Generator().manual_seed(8))
    finally:
        h.remove()
    assert len(count) == 1 and torch.isfinite(a.action).all()


def test_declared_semantics() -> None:
    meta = operation_expectation_metadata(("top", "wrist"))
    assert "no-target-mass" in str(meta["loss"]) and "not-goal" in str(meta["targets"])
    assert "not-observed-change" in str(meta["consumers"])


def test_robot_expectation_uses_successor_feature_difference(production: Production) -> None:
    kw = _loss_inputs(production)
    state = kw["future_state"]
    current = kw["expectation"].current_state
    row = torch.arange(1, 25).float()[None, :, None]
    future = current[:, None] + row
    target = (
        torch.stack([future[:, lo:hi].mean(1) for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 24))], 1)
        - current[:, None]
    )
    op = replace(kw["expectation"], robot_delta=target)
    terms = OperationExpectationSupervisor()(**{**kw, "expectation": op, "future_state": future})
    assert terms["operation_robot"] == 0
    assert state.shape == future.shape


def test_all_absent_outcome_labels_have_attached_zero(production: Production) -> None:
    kw = _loss_inputs(production)
    op = kw["expectation"]
    teacher = kw["teacher"]
    semantic = op.semantic_delta.clone().requires_grad_()
    image = op.image_delta.clone().requires_grad_()
    robot = op.robot_delta.clone().requires_grad_()
    support = kw["label_support"]
    assert support is not None
    support = replace(
        support, state=torch.zeros_like(support.state), visual=torch.zeros_like(support.visual)
    )
    teacher = replace(
        teacher,
        semantic_delta=torch.full_like(teacher.semantic_delta, float("nan")),
        transport_mean=torch.full_like(teacher.transport_mean, float("nan")),
    )
    result = OperationExpectationSupervisor()(
        **{
            **kw,
            "expectation": replace(
                op, semantic_delta=semantic, image_delta=image, robot_delta=robot
            ),
            "teacher": teacher,
            "future_state": torch.full_like(kw["future_state"], float("nan")),
            "label_support": support,
            "future_interval_valid": torch.zeros((1, 4), dtype=torch.bool),
        }
    )
    result["operation_total"].backward()
    for value in (semantic, image, robot):
        assert value.grad is not None and value.grad.count_nonzero() == 0
    assert result["operation_total"] == 0


def test_optimizer_owns_every_new_parameter_once(production: Production) -> None:
    m, e, _, _, _, _ = production
    ids = [id(p) for g in e.optimizer.param_groups for p in g["params"]]
    for module in (
        m.intent.organizer.operation_predictor,
        m.intent.organizer.operation_read,
        m.policy_compiler.plan_compiler.operation_read,
    ):
        assert module is not None
        assert all(ids.count(id(p)) == 1 for p in module.parameters())


@pytest.mark.parametrize("bf16", [False, True])
def test_diagnostics_leave_expectation_values_and_gradients_unchanged(bf16: bool) -> None:
    torch.manual_seed(6819)
    m, _ = _model_engine(_config())
    m.eval()
    b = _batch()
    results = []
    r = m.policy_compiler.plan_compiler.operation_read
    assert r is not None
    for diagnostics in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            cache, _, _ = m.encode_online(b.online, collect_diagnostics=diagnostics)
            op = cache.top.intent.operation_expectation
            assert op is not None
            out = r(op, cache.top.intent.temporal_queries[:, :, None])
            grad = torch.autograd.grad(out.float().square().mean(), tuple(r.parameters()))
        results.append((out.detach(), grad, torch.get_rng_state()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_checkpoint_exact_reload_and_ABI_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert abi["operation_expectation"] == operation_expectation_metadata(("top", "wrist"))
        for kind in ("missing", "semantics", "camera"):
            bad = copy.deepcopy(abi)
            if kind == "missing":
                del bad["operation_expectation"]
            elif kind == "semantics":
                cast(dict, bad["operation_expectation"])["targets"] = "success-oracle"
            else:
                cast(dict, bad["operation_expectation"])["camera_names"] = ["wrist", "top"]
            with pytest.raises(ValueError, match="operation expectation"):
                validate_deployment_abi(bad)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_full_loss_ledger_logging_is_noninterfering(
    monkeypatch: pytest.MonkeyPatch, grad_enabled: bool
) -> None:
    import test_mainline_instruction_reference as old

    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_real_training_diagnostics_are_executable_and_do_not_change_loss(
        reference_enabled=True, grad_enabled=grad_enabled
    )


@pytest.mark.parametrize("override", [None, True])
def test_supervisor_keeps_source_visual_mask_without_precomputed_interval(
    production: Production, override: bool | None
) -> None:
    kw = _loss_inputs(production)
    support = kw["label_support"]
    assert support is not None
    support = replace(support, visual=torch.zeros_like(support.visual))
    mask = None if override is None else torch.ones((1, 4), dtype=torch.bool)
    values = OperationExpectationSupervisor()(
        **{**kw, "label_support": support, "future_interval_valid": mask}
    )
    assert values["operation_semantic_count"] == values["operation_image_count"] == 0
    assert values["operation_robot_count"] == 4
