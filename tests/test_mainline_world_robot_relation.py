"""M6c current robot/view physics boundary; synthetic transport, production net."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_mainline_instruction_reference import _batch
from test_mainline_state_features import _model_engine
from test_mainline_world_control_domain import _config as control_config

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.model.robot_relation import RobotObjectRelationEncoder
from clearvla.mainline.model.types import ObjectWorldBelief, PhysicalActionSequenceCondition
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.world_robot import (
    OBSERVED_ROBOT_VIEWS,
    world_robot_metadata,
)


def _config():
    c = control_config()
    return replace(c, top=replace(c.top, world_robot_condition_mode=OBSERVED_ROBOT_VIEWS,
                                 world_camera_condition_mode="coordinate_role_v1"))


def _setup():
    torch.manual_seed(3660)
    m, e = _model_engine(_config())
    m.eval()
    return m, e, _batch(16)


@pytest.fixture(scope="module")
def encoded():
    m, e, b = _setup()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
    return m, e, b, cache, state


def test_config_identity_sources_and_invalid_combinations():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "world_robot_condition_mode" not in cast(dict, control_config().as_dict()["top"])
    assert load_config("configs/mainline/structural_rebuild_m6c_calvin.json").top.world_robot_condition_mode == OBSERVED_ROBOT_VIEWS
    for bad in ({"world_robot_condition_mode": "foo"}, {"world_control_mode": "legacy_extrapolation_v1"},
                {"world_camera_condition_mode": "motion_prior_only"}, {"entity_motion_mode": "query_anchor_v1"}):
        with pytest.raises(ValueError):
            replace(c, top=replace(c.top, **bad)).validate()
    closure = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/world_robot.py" in closure
    assert "clearvla/mainline/model/robot_relation.py" in closure


def test_same_observation_is_consumed_by_candidate_supervised_and_refinement(monkeypatch):
    m, _, b = _setup()
    encoder = m.world.dynamics.robot_relation_encoder
    assert encoder is not None
    seen = []
    handle = encoder.register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    try:
        with torch.no_grad():
            cache, state, _ = m.encode_online(b.online)
            belief = state.top.current_world_belief
            assert belief is not None and belief.robot_observation is not None
            assert seen == [belief]
            assert cache.top.belief is belief
            assert belief.robot_observation.state is cache.history.state
            assert belief.content is state.top.facts.content
            assert belief.camera_coordinates is state.top.facts.camera_coordinates
            targets, _ = m.build_training_targets(state, b.future)
            assert seen == [belief, belief]
            m.world.refine_deployment_world(cache.top, action_condition=cache.top.action_condition)
            assert seen == [belief, belief, belief]
            assert targets.supervised_world is not None
            before = len(seen)
            m.velocity(cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4))
            assert len(seen) == before  # no neural relation reevaluation inside ODE
    finally:
        handle.remove()


@pytest.mark.parametrize("bad", ["missing", "width", "chart", "nonfinite", "integer", "camera", "time"])
def test_observation_provenance_rejects_missing_or_misdeclared_input(encoded, bad):
    m, _, _, cache, _ = encoded
    belief = cache.top.belief
    observation = belief.robot_observation
    assert observation is not None
    if bad == "missing":
        belief = replace(belief, robot_observation=None)
    if bad == "width":
        belief = replace(belief, robot_observation=replace(observation, state=observation.state[:, :-1]))
    if bad == "chart":
        belief = replace(belief, robot_observation=replace(observation, feature_mode="wrong"))
    if bad == "nonfinite":
        belief = replace(belief, robot_observation=replace(observation, state=observation.state * float("nan")))
    if bad == "integer":
        belief = replace(belief, robot_observation=replace(observation, state=observation.state.long()))
    if bad == "camera":
        # Keep all original view axes consistent except the declared count.
        encoder = RobotObjectRelationEncoder(state_dim=10, state_mode=observation.feature_mode,
            content_dim=belief.content.shape[-1], hidden=32, camera_names=("top",))
    else:
        encoder = m.world.dynamics.robot_relation_encoder
    if bad == "time":
        belief = replace(belief, latest_flow_steps=None)
    assert encoder is not None
    with pytest.raises((ValueError, TypeError)):
        encoder(belief)


def test_caches_cannot_relabel_robot_state_or_current_visual_evidence(encoded):
    _, _, _, cache, state = encoded
    belief = cache.top.belief
    assert belief.robot_observation is not None
    cloned_state = replace(belief.robot_observation, state=belief.robot_observation.state.clone())
    with pytest.raises(ValueError, match="same current robot"):
        replace(cache, top=replace(cache.top, belief=replace(belief, robot_observation=cloned_state))).validate(_config())
    with pytest.raises(ValueError, match="same current G"):
        replace(state, top=replace(state.top, current_world_belief=replace(belief, content=belief.content.clone()))).validate(_config())
    with pytest.raises(ValueError, match="training cache robot"):
        replace(state, top=replace(state.top, current_world_belief=None)).validate(_config())


def test_no_raw_camera_average_is_used_and_reset_is_not_zero_speed_evidence(encoded, monkeypatch):
    m, _, _, cache, _ = encoded
    encoder = m.world.dynamics.robot_relation_encoder
    assert encoder is not None
    belief = cache.top.belief
    monkeypatch.setattr(ObjectWorldBelief, "transport_prior", property(lambda _: pytest.fail("raw view mean reached")))
    monkeypatch.setattr(ObjectWorldBelief, "transport_rate", property(lambda _: pytest.fail("raw view rate mean reached")))
    captured = []
    h = encoder.view_projection.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    try:
        with torch.no_grad():
            a = encoder(replace(belief, latest_flow_steps=torch.zeros(1, dtype=torch.long),
                                camera_transport_prior=torch.full_like(belief.camera_transport_prior, float("nan"))))
            b = encoder(replace(belief, latest_flow_steps=torch.ones(1, dtype=torch.long),
                                camera_transport_prior=torch.zeros_like(belief.camera_transport_prior)))
            m.world.materialize(belief=belief, action_condition=cache.top.action_condition)
    finally:
        h.remove()
    assert torch.isfinite(a.pooled).all()
    assert torch.count_nonzero(captured[0][..., 2:5]) == 0
    assert captured[1][..., 4].gt(0).any()
    assert not torch.equal(a.pooled, b.pooled)


def test_invalid_view_values_and_gradients_are_quarantined(encoded):
    m, _, _, cache, _ = encoded
    encoder = m.world.dynamics.robot_relation_encoder
    assert encoder is not None
    belief = cache.top.belief
    mask = belief.camera_validity.clone()
    mask[:, :, 1] = 0
    coords = belief.camera_coordinates.clone().requires_grad_()
    motion = belief.camera_transport_prior.clone().requires_grad_()
    clean = replace(belief, camera_validity=mask, camera_coordinates=coords, camera_transport_prior=motion)
    poisoned = replace(clean,
        camera_coordinates=torch.where(mask > 0, coords, float("nan")),
        camera_transport_prior=torch.where(mask > 0, motion, float("nan")))
    a, z = encoder(clean), encoder(poisoned)
    torch.testing.assert_close(a.pooled, z.pooled, rtol=0, atol=0)
    for g in torch.autograd.grad(z.pooled.square().sum(), (coords, motion)):
        assert torch.isfinite(g).all() and torch.count_nonzero(g[:, :, 1]) == 0
    absent = replace(poisoned, validity=torch.zeros_like(belief.validity), content=belief.content * float("nan"))
    result = encoder(absent)
    assert torch.count_nonzero(result.pooled) == 0 and torch.count_nonzero(result.per_view) == 0


def test_camera_axes_can_be_permuted_by_declared_roles_not_raw_xy_averaged(encoded):
    m, _, _, cache, _ = encoded
    old = m.world.dynamics.robot_relation_encoder
    assert old is not None
    belief = cache.top.belief
    shuffled = RobotObjectRelationEncoder(state_dim=old.state_dim, state_mode=old.state_mode,
        content_dim=belief.content.shape[-1], hidden=old.state_projection.out_features,
        camera_names=tuple(reversed(old.camera_names)))
    shuffled.load_state_dict(old.state_dict(), strict=True)
    values = {name: getattr(belief, name).flip(2) for name in (
        "camera_coordinates", "camera_transport_prior", "camera_support", "camera_validity", "log_camera_validity")}
    with torch.no_grad():
        a, b = old(belief), shuffled(replace(belief, **values))
    torch.testing.assert_close(a.pooled, b.pooled, rtol=0, atol=0)
    torch.testing.assert_close(a.per_view, b.per_view.flip(2), rtol=0, atol=0)
    # Opposite motions remain distinct inputs in their own named charts.
    motion = torch.zeros_like(belief.camera_transport_prior)
    motion[:, :, 0, 0], motion[:, :, 1, 0] = 1, -1
    with torch.no_grad():
        move = old(replace(belief, camera_transport_prior=motion))
        still = old(replace(belief, camera_transport_prior=torch.zeros_like(motion)))
    assert not torch.equal(move.pooled, still.pooled)


def test_object_permutation_preserves_robot_observation_and_relation(encoded):
    m, _, _, cache, _ = encoded
    encoder = m.world.dynamics.robot_relation_encoder
    assert encoder is not None
    belief = cache.top.belief
    index = torch.arange(belief.objects - 1, -1, -1)
    changed = belief.permute(index)
    assert changed.robot_observation is belief.robot_observation
    with torch.no_grad():
        a, b = encoder(belief), encoder(changed)
    torch.testing.assert_close(a.pooled[:, index], b.pooled, rtol=0, atol=0)
    torch.testing.assert_close(a.per_view[:, index], b.per_view, rtol=0, atol=0)


def test_W1_relation_cannot_be_reused_with_another_observation(encoded):
    m, _, _, cache, _ = encoded
    belief = cache.top.belief
    with torch.no_grad():
        _, state, _ = m.world.dynamics.forward_w1(facts=belief, action=cache.top.action_condition)
        with pytest.raises(ValueError, match="different observation belief"):
            m.world.dynamics.forward_w2(facts=replace(belief, content=belief.content.clone()), w1_state=state)


@pytest.mark.parametrize("bf16", [False, True])
def test_real_optimizer_then_world_and_action_loss_use_current_robot_and_view_parameters(bf16):
    m, e, b = _setup()
    e.train_step(b, collect_diagnostics=False)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        ledger, _ = e._forward(b, training=False, collect_diagnostics=True, generator=torch.Generator().manual_seed(3661))
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    encoder = m.world.dynamics.robot_relation_encoder
    assert encoder is not None
    for name, param in encoder.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert torch.count_nonzero(param.grad) > 0, name
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    # Isolate W's current robot input from G, S, action and all history changes.
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
    belief = cache.top.belief
    assert belief.robot_observation is not None
    robot = belief.robot_observation.state.detach().clone().requires_grad_()
    changed = replace(belief, robot_observation=replace(belief.robot_observation, state=robot))
    world, _ = m.world.materialize(belief=changed, action_condition=cache.top.action_condition)
    # Actual P2/P3/decoder via actual velocity call, same G/S/control, only W changes.
    counterfactual = replace(cache, history=replace(cache.history, state=robot),
        top=replace(cache.top, belief=changed, candidate_world=world))
    output = m.velocity(counterfactual, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4))
    # The state is also available to the ordinary action bridge. Only the
    # new W encoder weight can prove the decoder really consumes THIS route.
    gradient = torch.autograd.grad(output.bottom.physical_velocity.square().sum(),
                                  encoder.state_projection.weight)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    assert changed.robot_observation is not None
    perturbed = replace(changed, robot_observation=replace(changed.robot_observation, state=robot + 0.3))
    with torch.no_grad():
        other, _ = m.world.materialize(belief=perturbed, action_condition=cache.top.action_condition)
    assert not torch.equal(world.dynamics.semantic_delta, other.dynamics.semantic_delta)
    assert not torch.equal(world.dynamics.transport_mean, other.dynamics.transport_mean)


def test_real_matched_controls_keep_near_predictions_equal_and_later_controls_cannot_rewrite_near():
    m, e, b = _setup()
    e.train_step(b, collect_diagnostics=False)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
        candidate = cache.top.action_condition
        assert isinstance(candidate, PhysicalActionSequenceCondition)
        raw = torch.cat((candidate.source_action, b.future.action_sequence[:, 24:]), dim=1)
        a = m.outlet_adapter.observed_world_condition(raw, b.online.history.action_state)
        raw2 = raw.clone()
        raw2[:, 24:, 0] += 0.7
        z = m.outlet_adapter.observed_world_condition(raw2, b.online.history.action_state)
        w, _ = m.world.materialize_supervised(belief=cache.top.belief, action_condition=a)
        q, _ = m.world.materialize_supervised(belief=cache.top.belief, action_condition=z)
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        torch.testing.assert_close(getattr(w.dynamics, name)[:, :2], getattr(q.dynamics, name)[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(getattr(w.dynamics, name)[:, :2], getattr(cache.top.predicted_dynamics, name)[:, :2], rtol=0, atol=0)
    assert not torch.equal(w.dynamics.semantic_delta[:, 2:], q.dynamics.semantic_delta[:, 2:])


def test_future_control_changes_do_not_mutate_robot_or_cached_online_actions(encoded):
    m, _, b, cache, state = encoded
    assert cache.top.belief.robot_observation is not None
    robot = cache.top.belief.robot_observation.state.clone()
    with torch.no_grad():
        a = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3662))
        future = replace(b.future, action_sequence=b.future.action_sequence + 0.4)
        m.build_training_targets(state, future)
        z = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3662))
    torch.testing.assert_close(a.action, z.action, rtol=0, atol=0)
    torch.testing.assert_close(robot, cache.top.belief.robot_observation.state, rtol=0, atol=0)


def test_exact_checkpoint_ownership_and_robot_ABI(tmp_path, monkeypatch):
    import test_mainline_target_binding as original_test

    from clearvla.mainline.runtime.deployment import validate_deployment_abi
    build_original = original_test.build_deployment_abi
    def build(*args, **kwargs):
        abi = build_original(*args, **kwargs)
        assert abi["world_robot"] == world_robot_metadata(state_mode=_config().top.state_feature_mode,
                                                         state_dim=_config().dimensions.state_dim)
        for mode in ("missing", "width", "chart"):
            wrong = copy.deepcopy(abi)
            if mode == "missing":
                del wrong["world_robot"]
            else:
                cast(dict[str, Any], wrong["world_robot"])["state_dim" if mode == "width" else "state_features"] = 777
            with pytest.raises(ValueError, match="robot observation"):
                validate_deployment_abi(wrong)
        return abi
    monkeypatch.setattr(original_test, "build_deployment_abi", build)
    monkeypatch.setattr(original_test, "_config", _config)
    monkeypatch.setattr(original_test, "_batch", _batch)
    original_test.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)
