"""Actual S/P3 endpoint-goal graph; synthetic transport, not learned task success."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch
from test_mainline_executed_world import _config as base_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.annotation_goal import ANNOTATED_ENDPOINT_GOAL
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.interfaces import OnlinePolicyInput
from clearvla.mainline.model.annotation_goal import (
    reference_observation_law,
    supervise_annotated_goal,
)
from clearvla.mainline.runtime.qualification import synthetic_batch
from clearvla.mainline.runtime.sampling import sample_action


def _config():
    c = base_config()
    return replace(
        c,
        top=replace(c.top, annotation_goal_mode=ANNOTATED_ENDPOINT_GOAL),
        objectives=replace(c.objectives, annotated_goal=0.01),
    )


def _batch(count=1):
    return synthetic_batch(_config(), count=count, raw_side=32, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def production():
    torch.manual_seed(27301)
    b, n = _batch()
    m, e = _model_engine(_config())
    m.configure_action_normalizer(n)
    result = e.train_step(b, collect_diagnostics=False)
    assert e.global_step == 1
    assert torch.isfinite(result.loss)
    assert result.metrics["loss_ledger_gap"] == 0
    torch.testing.assert_close(
        result.metrics["loss_contrib_annotated_goal"],
        0.01 * result.metrics["loss_annotated_goal_total"],
        rtol=0,
        atol=0,
    )
    m.eval()
    m.set_training_step(1200)
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    return m, e, b, cache


def _evidence(production):
    evidence = production[3].top.intent.annotated_goal
    assert evidence is not None
    return evidence


def _reader(production):
    r = production[0].policy_compiler.plan_compiler.annotated_goal_read
    assert r is not None
    return r


def test_identity_is_explicit_and_prior_modes_unchanged():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    before = base_config().as_dict()
    assert isinstance(before["top"], dict) and isinstance(before["objectives"], dict)
    assert "annotation_goal_mode" not in before["top"]
    assert "annotated_goal" not in before["objectives"]
    assert (
        load_config(
            "configs/mainline/structural_rebuild_annotated_goal_calvin.json"
        ).top.annotation_goal_mode
        == ANNOTATED_ENDPOINT_GOAL
    )
    assert "annotation_endpoint" not in {f.name for f in fields(OnlinePolicyInput)}


@pytest.mark.parametrize(
    "field,value",
    [
        ("annotation_goal_mode", "invented"),
        ("instruction_change_mode", "typed_reference_v1"),
        ("target_binding_mode", "reader_local_v1"),
        ("p3_coordination_mode", "pointwise_v1"),
    ],
)
def test_goal_graph_rejects_incompatible_modes(field, value):
    c = _config()
    with pytest.raises((TypeError, ValueError)):
        replace(c, top=replace(c.top, **{field: value})).validate()


@pytest.mark.parametrize("enabled,weight", [(True, 0.0), (False, 0.01)])
def test_objective_budget_cannot_silently_have_no_owner(enabled, weight):
    c = _config()
    with pytest.raises(ValueError):
        replace(
            c,
            top=replace(c.top, annotation_goal_mode=ANNOTATED_ENDPOINT_GOAL if enabled else "none"),
            objectives=replace(c.objectives, annotated_goal=weight),
        ).validate()


def test_typed_label_is_independent_of_action_horizon():
    b, _ = _batch()
    label = b.future.annotation_endpoint
    assert label is not None
    b.validate(_config())
    assert label.offset_steps.item() > _config().dimensions.action_horizon
    assert label.dino.ndim == 4  # not a slice of the fixed-horizon future grid


@pytest.mark.parametrize(
    "kind",
    [
        "clock",
        "provenance",
        "float_indices",
        "undeclared_support",
        "nonfinite_state",
        "nonfinite_visual",
        "missing",
    ],
)
def test_label_admission_checks_sources_before_training(kind):
    b, _ = _batch()
    label = b.future.annotation_endpoint
    assert label is not None
    if kind == "clock":
        ref = b.online.instruction_reference
        assert ref is not None
        b = replace(
            b,
            online=replace(
                b.online, instruction_reference=replace(ref, age_steps=ref.age_steps + 1)
            ),
        )
    else:
        if kind == "provenance":
            label = replace(label, offset_steps=label.offset_steps + 1)
        elif kind == "float_indices":
            label = replace(label, source_indices=label.source_indices.float())
        elif kind == "undeclared_support":
            label = replace(label, declared=torch.zeros_like(label.declared))
        elif kind == "nonfinite_state":
            label = replace(label, state=torch.full_like(label.state, torch.nan))
        elif kind == "nonfinite_visual":
            label = replace(label, dino=torch.full_like(label.dino, torch.nan))
        elif kind == "missing":
            label = None
        b = replace(b, future=replace(b.future, annotation_endpoint=label))
    with pytest.raises((ValueError, TypeError)):
        b.validate(_config())


def test_actual_first_command_gradients_reach_both_S_and_P3(production):
    m, _, b, _ = production
    predictor = m.intent.organizer.endpoint_goal
    sr = m.intent.organizer.endpoint_goal_read
    assert predictor is not None and sr is not None
    pr = _reader(production)
    cache, _, _ = m.encode_online(b.online)
    out = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
    parameters = [
        predictor.content.weight,
        predictor.origin.weight,
        predictor.robot.weight,
        sr.joint_output.weight,
        pr.values.joint_output.weight,
        pr.output.weight,
    ]
    grads = torch.autograd.grad(out.bottom.physical_velocity[:, 0].square().sum(), parameters)
    for g in grads:
        assert torch.isfinite(g).all() and g.count_nonzero() > 0


def test_current_visual_cannot_move_the_start_conditioned_goal(production):
    m, _, b, _ = production
    dino = b.online.observation.dino_history.clone()
    dino[:, -1] = -dino[:, -1]
    online = replace(b.online, observation=replace(b.online.observation, dino_history=dino))
    with torch.no_grad():
        cache, _, _ = m.encode_online(online)
    a, z = _evidence(production), cache.top.intent.annotated_goal
    assert z is not None
    for name in ("scene", "log_probability", "robot_delta"):
        torch.testing.assert_close(
            getattr(a.prediction, name), getattr(z.prediction, name), rtol=0, atol=0
        )
    assert not torch.equal(a.current_probability, z.current_probability)


def test_goal_does_not_read_instruction_age_but_does_read_language(production):
    m, _, _, _ = production
    p = m.intent.organizer.endpoint_goal
    assert p is not None
    ref = _evidence(production).prediction.reference
    language = torch.randn(1, 4, _config().dimensions.hidden_size)
    a = p(reference=ref, language=language)
    b = p(reference=replace(ref, age_steps=ref.age_steps + 1000), language=language)
    z = p(reference=ref, language=-language)
    for name in ("scene", "log_probability", "robot_delta"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=0, rtol=0)
    assert not torch.equal(a.scene, z.scene)


def test_exactly_realized_goal_has_zero_relation_value(production):
    evidence = _evidence(production)
    change = evidence.instruction_change
    p = replace(
        evidence.prediction,
        scene=evidence.current_content,
        log_probability=evidence.current_probability.log(),
        robot_delta=change.current_state - evidence.prediction.reference.state,
    )
    same = replace(evidence, prediction=p, current_probability=p.log_probability.softmax(-1))
    for value in _reader(production).values(same):
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)


def test_null_correspondence_and_target_mass_remain_distinct(production):
    evidence = _evidence(production)
    change = evidence.instruction_change
    posterior = change.posterior
    assert posterior is not None
    null = replace(
        posterior, reference_probability=torch.zeros_like(posterior.reference_probability)
    )
    no_match = replace(evidence, instruction_change=replace(change, posterior=null))
    values = _reader(production).values(no_match)
    for value in values[:3]:
        assert value.count_nonzero() == 0
    # Robot relation can remain observable even when visual association is null.
    assert values[3].count_nonzero() > 0
    no_target = replace(
        change.binding,
        log_probability=torch.cat(
            (
                torch.full_like(change.binding.log_probability[..., :-1], -torch.inf),
                torch.zeros_like(change.binding.log_probability[..., -1:]),
            ),
            -1,
        ),
    )
    values = _reader(production).values(
        replace(evidence, instruction_change=replace(change, binding=no_target))
    )
    assert all(v.count_nonzero() == 0 for v in values)


def test_label_change_updates_supervision_not_online_prediction(production):
    _, _, b, _ = production
    label = b.future.annotation_endpoint
    assert label is not None
    p = _evidence(production).prediction
    original = p.scene.clone()
    a = supervise_annotated_goal(p, label)
    z = supervise_annotated_goal(p, replace(label, dino=-label.dino, state=label.state + 2))
    assert not torch.equal(a["annotated_goal_total"], z["annotated_goal_total"])
    torch.testing.assert_close(p.scene, original, atol=0, rtol=0)


@pytest.mark.parametrize("which", ["visual", "state", "both"])
def test_unsupported_nan_labels_do_not_poison_forward_or_backward(which, production):
    m, _, b, _ = production
    label = b.future.annotation_endpoint
    assert label is not None
    if which in ("visual", "both"):
        label = replace(
            label,
            dino=torch.full_like(label.dino, torch.nan),
            visual_observed=torch.zeros_like(label.visual_observed),
        )
    if which in ("state", "both"):
        label = replace(
            label,
            state=torch.full_like(label.state, torch.nan),
            state_observed=torch.zeros_like(label.state_observed),
        )
    label.validate(_config(), batch=1, device=torch.device("cpu"))
    cache, _, _ = m.encode_online(b.online)
    evidence = cache.top.intent.annotated_goal
    assert evidence is not None
    losses = supervise_annotated_goal(evidence.prediction, label)
    params = tuple(m.intent.organizer.endpoint_goal.parameters())
    grads = torch.autograd.grad(losses["annotated_goal_total"], params, allow_unused=True)
    assert torch.isfinite(losses["annotated_goal_total"])
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    if which == "both":
        assert losses["annotated_goal_total"] == 0
        assert all(g is None or g.count_nonzero() == 0 for g in grads)


def test_unseen_destinations_are_not_negative_labels(production):
    _, _, b, _ = production
    label = b.future.annotation_endpoint
    assert label is not None
    support = label.visual_observed.clone()
    support[..., 1::2] = False
    label = replace(label, visual_observed=support)
    p = _evidence(production).prediction
    logit = p.log_probability.clone()
    logit[..., 1:-1:2] += 10
    changed = replace(p, log_probability=logit.log_softmax(-1))
    a = supervise_annotated_goal(p, label)["annotated_goal_relation"]
    z = supervise_annotated_goal(changed, label)["annotated_goal_relation"]
    torch.testing.assert_close(a, z, atol=1e-6, rtol=0)


def test_endpoint_labels_have_no_backward_owner(production):
    m, _, b, _ = production
    label = b.future.annotation_endpoint
    assert label is not None
    dino = label.dino.clone().requires_grad_()
    state = label.state.clone().requires_grad_()
    cache, _, _ = m.encode_online(b.online)
    evidence = cache.top.intent.annotated_goal
    assert evidence is not None
    loss = supervise_annotated_goal(evidence.prediction, replace(label, dino=dino, state=state))[
        "annotated_goal_total"
    ]
    grads = torch.autograd.grad(loss, (dino, state), allow_unused=True)
    assert grads == (None, None)


def test_training_loss_does_not_use_operated_target_confidence(production):
    _, _, b, _ = production
    label = b.future.annotation_endpoint
    assert label is not None
    evidence = _evidence(production)
    a = supervise_annotated_goal(evidence.prediction, label)
    # Loss API has neither a target binding nor a predicted-success gate.
    assert all(torch.isfinite(v) for v in a.values())
    assert a["annotated_goal_visual_coverage"] == 1 and a["annotated_goal_state_coverage"] == 1


def test_p3_goal_values_cached_once_no_extra_world_or_teacher_calls(production):
    m, _, b, _ = production
    r = _reader(production)
    with (
        patch.object(
            m.training_targets.teacher, "forward", side_effect=AssertionError("future label entry")
        ),
        patch.object(m.world.dynamics, "forward_w1", wraps=m.world.dynamics.forward_w1) as w1,
        patch.object(m.world.dynamics, "forward_w2", wraps=m.world.dynamics.forward_w2) as w2,
        patch.object(r, "prepare", wraps=r.prepare) as prepared,
    ):
        out = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(313))
    assert torch.isfinite(out.action).all()
    assert (w1.call_count, w2.call_count, prepared.call_count) == (3, 2, 1)


def test_prepared_values_reject_another_source_and_reader(production):
    evidence = _evidence(production)
    reader = _reader(production)
    values = reader.prepare(evidence)
    query = torch.zeros(1, 24, 3, _config().dimensions.hidden_size)
    with pytest.raises(ValueError):
        reader(replace(evidence), query, values)
    with pytest.raises(ValueError):
        reader(evidence, query, replace(values, reader_identity=-1))


def test_reference_metric_is_soft_and_preserves_null(production):
    p = _evidence(production).prediction
    q = reference_observation_law(p.reference, p.scene, torch.ones_like(p.reference.observed))
    assert q.dtype == torch.float32 and not q.requires_grad
    assert torch.all(q > 0)
    torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]))
    missing = reference_observation_law(
        p.reference, p.scene, torch.zeros_like(p.reference.observed)
    )
    assert missing[..., :-1].count_nonzero() == 0 and torch.all(missing[..., -1] == 1)


def test_full_graph_CPU_bf16_gradient(production):
    m, _, b, _ = production
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache, _, _ = m.encode_online(b.online)
        out = m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.tensor([0.4]))
    p = m.intent.organizer.endpoint_goal
    assert p is not None
    grads = torch.autograd.grad(
        out.bottom.physical_velocity[:, 0].square().sum(),
        [p.content.weight, p.origin.weight, _reader(production).output.weight],
    )
    for g in grads:
        assert torch.isfinite(g).all() and g.count_nonzero() > 0


def test_exact_checkpoint_and_deployment_metadata(tmp_path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        assert "annotation_goal" in abi
        wrong = copy.deepcopy(abi)
        wrong["annotation_goal"] = {"success_oracle": True}
        with pytest.raises(ValueError):
            validate_deployment_abi(wrong)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", lambda center: _batch()[0])
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_unseen_current_destinations_are_unknown_not_goal_failure(production):
    evidence = _evidence(production)
    support = evidence.current_observed.clone()
    support[..., 1::2] = False
    probability = reference_observation_law(
        evidence.prediction.reference, evidence.current_content, support
    )
    visible = replace(evidence, current_observed=support, current_probability=probability)
    logits = evidence.prediction.log_probability.clone()
    logits[..., 1:-1:2] += 0.5
    scene = evidence.prediction.scene.clone()
    scene[..., 1::2, :] = torch.nan
    hidden_changed = replace(
        visible,
        prediction=replace(
            evidence.prediction, scene=scene, log_probability=logits.log_softmax(-1)
        ),
    )
    reader = _reader(production).values
    for a, b in zip(reader(visible), reader(hidden_changed), strict=True):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(
            a, b, rtol=0, atol=16 * torch.finfo(torch.float32).eps * float(a.detach().abs().max())
        )


def test_equal_centroids_can_still_carry_different_goal_relations(production):
    evidence = _evidence(production)
    n = evidence.prediction.scene.shape[-2]
    side = int(n**0.5)
    a = torch.zeros_like(evidence.current_probability)
    b = torch.zeros_like(a)
    a[..., 0] = a[..., n - 1] = 0.5
    b[..., side - 1] = b[..., n - side] = 0.5
    scene = torch.zeros_like(evidence.prediction.scene)
    prediction = replace(evidence.prediction, scene=scene, log_probability=a.log())
    different = replace(
        evidence, prediction=prediction, current_content=scene, current_probability=b
    )
    same = replace(different, current_probability=a)
    xy = prediction.coordinates
    torch.testing.assert_close(a[..., :-1] @ xy, b[..., :-1] @ xy, rtol=0, atol=0)
    reader = copy.deepcopy(_reader(production).values)
    first, last = reader.image[0], reader.image[2]
    assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        first.weight.zero_()
        first.weight[0] = torch.tensor([1.0, 1.0])
        last.weight.zero_()
        last.weight[0, 0] = 1
    assert reader.prepare(different).image.count_nonzero() > 0
    assert reader.prepare(same).image.count_nonzero() == 0


def test_P3_alone_changes_actual_command_with_other_cached_inputs_fixed(production):
    m, _, _, cache = production
    evidence = _evidence(production)
    p = replace(evidence.prediction, robot_delta=evidence.prediction.robot_delta + 0.25)
    changed = replace(evidence, prediction=p)
    goal_values = _reader(production).prepare(changed)
    intent = replace(cache.top.intent, annotated_goal=changed, annotated_goal_values=goal_values)
    counterfactual = replace(cache, top=replace(cache.top, intent=intent))
    z = torch.randn(1, 24, 18)
    with torch.no_grad():
        a = m.velocity(
            cache, noisy_action_field=z, time=torch.tensor([0.4])
        ).bottom.physical_velocity
        b = m.velocity(
            counterfactual, noisy_action_field=z, time=torch.tensor([0.4])
        ).bottom.physical_velocity
    assert torch.isfinite(b).all() and not torch.equal(a[:, 0], b[:, 0])


def test_current_K_permutation_preserves_reference_goal_and_final_read(production):
    evidence = _evidence(production)
    change = evidence.instruction_change
    index = torch.arange(change.content_delta.shape[1] - 1, -1, -1)
    other = change.permute(index, change.binding.permute(index))
    permuted = evidence.permute(other)
    assert permuted.prediction is evidence.prediction
    reader = _reader(production).values
    for a, b in zip(reader(evidence), reader(permuted), strict=True):
        torch.testing.assert_close(
            a, b, rtol=0, atol=16 * torch.finfo(torch.float32).eps * float(a.detach().abs().max())
        )


@pytest.mark.parametrize("remove", ["goal", "values"])
def test_online_cache_requires_selected_goal_and_its_prepared_owner(remove, production):
    cache = production[3]
    intent = replace(
        cache.top.intent,
        **{"annotated_goal" if remove == "goal" else "annotated_goal_values": None},
    )
    with pytest.raises(ValueError):
        replace(cache, top=replace(cache.top, intent=intent)).validate(_config())
