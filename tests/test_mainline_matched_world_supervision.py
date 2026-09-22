"""Observed-control W supervision is separate from the deployed candidate world.

Image/DINO/T5 payload comes from the existing declared transport fixtures;
G/S/W/P, outlet charts, optimizer, loss routing and checkpoint code are real.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_instruction_reference import _batch
from test_mainline_instruction_reference import _config as reference_config
from test_mainline_state_features import _model_engine

from clearvla.mainline.config import config_from_mapping, load_config
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.model.types import CandidateWorld, PhysicalActionSequenceCondition
from clearvla.mainline.runtime.sampling import sample_action


def _config():
    c = reference_config()
    return replace(c, top=replace(c.top, world_supervision_mode="matched_observed_sequence_v1"))


def _setup():
    torch.manual_seed(3061)
    m, engine = _model_engine(_config())
    m.eval()
    b = _batch(16)
    return m, engine, b


def test_explicit_selection_and_legacy_config_identity():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    legacy_top = reference_config().as_dict()["top"]
    assert isinstance(legacy_top, dict)
    assert "world_supervision_mode" not in legacy_top
    with pytest.raises(ValueError, match="sequence-prefix"):
        replace(c, top=replace(c.top, world_action_condition_mode="interval_mean_v1")).validate()
    with pytest.raises(ValueError, match="world_supervision_mode"):
        replace(c, top=replace(c.top, world_supervision_mode="other")).validate()
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m6a_calvin.json"
        ).top.world_supervision_mode
        == c.top.world_supervision_mode
    )


@pytest.mark.parametrize("prefix", [0, 1, 7, 8, 15, 16, 23, 24, 31, 32, 47, 48])
def test_known_prefix_owns_interval_labels_and_missing_payload_is_quarantined(prefix: int):
    m, _, b = _setup()
    raw = b.future.action_sequence.clone().requires_grad_()
    mask = torch.arange(raw.shape[1])[None] < prefix
    poisoned = torch.where(mask[..., None], raw, torch.full_like(raw, float("nan")))
    a = m.outlet_adapter.observed_world_condition(poisoned, b.online.history.action_state, mask)
    a.validate(action_dim=7)
    assert a.source_action.shape == (1, 48, 7)
    assert torch.isfinite(a.physical_fingerprint).all()
    assert not a.physical_fingerprint.requires_grad
    torch.testing.assert_close(
        a.interval_observed, torch.tensor([[prefix >= t for t in (8, 16, 32, 48)]])
    )
    assert torch.count_nonzero(a.physical_fingerprint[:, prefix:]) == 0
    assert torch.count_nonzero(a.source_action[:, prefix:]) == 0
    clean = m.outlet_adapter.observed_world_condition(
        torch.where(mask[..., None], raw, torch.full_like(raw, 1e30)),
        b.online.history.action_state,
        mask,
    )
    torch.testing.assert_close(a.physical_fingerprint, clean.physical_fingerprint, rtol=0, atol=0)


def test_relative_chart_continues_across_24_row_boundary_and_keeps_real_clock():
    m, _, b = _setup()
    normalizer = ArrayNormalizer.fit_zscore(
        [np.stack((np.arange(7), np.arange(7) + 4)).astype(np.float32)]
    )
    m.configure_action_normalizer(normalizer)
    raw = torch.randn(1, 48, 7, generator=torch.Generator().manual_seed(3062))
    a = m.outlet_adapter.observed_world_condition(raw, b.online.history.action_state)
    first = m.outlet_adapter.world_condition_from_horizon_action(
        raw[:, :24], b.online.history.action_state
    )
    assert isinstance(first, PhysicalActionSequenceCondition)
    arm = raw[..., :6] - first.normalizer_offset[..., :6]
    torch.testing.assert_close(a.canonical_value[..., :6], arm.cumsum(dim=1), atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(a.canonical_delta[..., :6], arm, rtol=0, atol=0)
    torch.testing.assert_close(
        a.canonical_delta[:, 24, 6], raw[:, 24, 6] - raw[:, 23, 6], rtol=0, atol=0
    )
    torch.testing.assert_close(a.canonical_value[:, :24], first.canonical_value, rtol=0, atol=0)
    assert a.row_end[0, 23, 0] == 24 and a.row_end[0, 47, 0] == 48
    assert a.control_time_scale == 24


@pytest.mark.parametrize("bad", ["gap", "known_nan", "wrong_mask", "short"])
def test_invalid_control_provenance_is_rejected(bad: str):
    m, _, b = _setup()
    raw = b.future.action_sequence.clone()
    mask = torch.ones(raw.shape[:2], dtype=torch.bool)
    if bad == "gap":
        mask[:, 7] = False
    if bad == "known_nan":
        raw[:, 0, 0] = float("nan")
    if bad == "wrong_mask":
        mask = mask.float()
    if bad == "short":
        raw, mask = raw[:, :24], mask[:, :24]
    with pytest.raises(ValueError):
        m.outlet_adapter.observed_world_condition(raw, b.online.history.action_state, mask)


@pytest.mark.parametrize("cut,unchanged", [(8, 1), (16, 2), (24, 2), (32, 3)])
def test_physical_action_carrier_is_prefix_causal_and_uses_late_observed_controls(
    cut: int, unchanged: int
):
    m, _, b = _setup()
    with torch.no_grad():
        _, state, _ = m.encode_online(b.online)
        a = m.outlet_adapter.observed_world_condition(
            b.future.action_sequence, b.online.history.action_state
        )
        modified = b.future.action_sequence.clone()
        modified[:, cut:, 0] += 0.4
        z = m.outlet_adapter.observed_world_condition(modified, b.online.history.action_state)
        p, _, _, _ = m.world.dynamics._base(state.top.facts, a, collect_diagnostics=False)
        q, _, _, _ = m.world.dynamics._base(state.top.facts, z, collect_diagnostics=False)
    torch.testing.assert_close(p[:, :unchanged], q[:, :unchanged], rtol=0, atol=0)
    assert not torch.equal(p[:, unchanged:], q[:, unchanged:])


def test_missing_actions_do_not_advance_recurrence():
    m, _, b = _setup()
    with torch.no_grad():
        _, state, _ = m.encode_online(b.online)
        mask = torch.arange(48)[None] < 7
        a = m.outlet_adapter.observed_world_condition(
            b.future.action_sequence, b.online.history.action_state, mask
        )
        states = []
        cell = m.world.dynamics.sequence_action_recurrence
        assert cell is not None
        hook = cell.register_forward_pre_hook(lambda _, args: states.append(args[1].clone()))
        try:
            m.world.dynamics._base(state.top.facts, a, collect_diagnostics=False)
        finally:
            hook.remove()
    assert len(states) == 48
    for value in states[8:]:
        torch.testing.assert_close(value, states[7], rtol=0, atol=0)


def test_supervised_condition_cannot_enter_candidate_world_or_online_adapter():
    m, _, b = _setup()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
        targets, _ = m.build_training_targets(state, b.future)
    supervised = targets.supervised_world
    assert supervised is not None
    with pytest.raises(TypeError):
        m.outlet_adapter.validate_world_condition(cast(Any, supervised.action_condition))
    with pytest.raises(TypeError):
        m.world.materialize(
            belief=state.top.facts, action_condition=cast(Any, supervised.action_condition)
        )
    with pytest.raises(TypeError):
        CandidateWorld(cast(Any, supervised.action_condition), supervised.dynamics).validate(
            action_dim=7
        )
    with pytest.raises(TypeError):
        m.world.refine_deployment_world(
            cache.top, action_condition=cast(Any, supervised.action_condition)
        )


def test_actual_loss_routes_observed_world_and_retains_online_candidate(monkeypatch):
    import clearvla.mainline.training.losses as losses_module

    m, e, b = _setup()
    worlds = []
    original_world = m.world.materialize_supervised
    original_loss = losses_module.future_dynamics_terms

    def capture_world(**kwargs):
        result = original_world(**kwargs)
        worlds.append(result[0])
        return result

    def check_loss(prediction, target, **kwargs):
        assert worlds and prediction is worlds[-1].dynamics
        assert kwargs["interval_valid"].all()
        return original_loss(prediction, target, **kwargs)

    monkeypatch.setattr(m.world, "materialize_supervised", capture_world)
    monkeypatch.setattr(losses_module, "future_dynamics_terms", check_loss)
    ledger, _ = e._forward(
        b, training=False, collect_diagnostics=True, generator=torch.Generator().manual_seed(3063)
    )
    assert len(worlds) == 1 and torch.isfinite(ledger.total)
    ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_supervised_future_loss_has_no_path_to_S_or_label_gradients():
    m, _, b = _setup()
    cache, state, _ = m.encode_online(b.online)
    future = replace(b.future, action_sequence=b.future.action_sequence.clone().requires_grad_())
    tag, candidate = cache.top.action_condition, cache.top.predicted_dynamics
    targets, _ = m.build_training_targets(state, future)
    supervised = targets.supervised_world
    assert supervised is not None
    assert cache.top.action_condition is tag and cache.top.predicted_dynamics is candidate
    supervised.dynamics.semantic_delta.sum().backward()
    assert future.action_sequence.grad is None
    assert all(p.grad is None for p in m.intent.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.world.parameters())


def test_changed_future_targets_do_not_change_online_cache_or_supervised_physics():
    m, _, b = _setup()
    with torch.no_grad():
        cache, state, _ = m.encode_online(b.online)
        field = cache.top.predicted_dynamics.semantic_delta.clone()
        one, _ = m.build_training_targets(state, b.future)
        altered = replace(
            b.future,
            dino_supports=b.future.dino_supports * 0.7,
            state_sequence=b.future.state_sequence * 0.3,
        )
        two, _ = m.build_training_targets(state, altered)
    assert one.supervised_world is not None and two.supervised_world is not None
    torch.testing.assert_close(
        one.supervised_world.dynamics.semantic_delta,
        two.supervised_world.dynamics.semantic_delta,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(cache.top.predicted_dynamics.semantic_delta, field, rtol=0, atol=0)


def test_deployment_does_not_run_observed_control_world(monkeypatch):
    m, _, b = _setup()

    def forbidden(**_):
        raise AssertionError("training W was called by deployment")

    monkeypatch.setattr(m.world, "materialize_supervised", forbidden)
    with torch.no_grad():
        out = sample_action(m, b.online, _config(), generator=torch.Generator().manual_seed(3064))
    assert out.action.shape == (1, 24, 7) and torch.isfinite(out.action).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_actual_training_step_or_bf16_backward_is_finite(bf16: bool):
    m, e, b = _setup()
    if not bf16:
        e.train_step(b, collect_diagnostics=True)
    else:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ledger, _ = e._forward(
                b,
                training=False,
                collect_diagnostics=True,
                generator=torch.Generator().manual_seed(3065),
            )
        assert torch.isfinite(ledger.total)
        ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_exact_checkpoint_and_W_contract_reload(tmp_path: Path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = original(*args, **kwargs)
        world_abi = abi["world_supervision"]
        assert isinstance(world_abi, dict)
        assert world_abi["prefix_endpoints"] == [8, 16, 32, 48]
        for kind in ("missing", "changed"):
            wrong = copy.deepcopy(abi)
            if kind == "missing":
                del wrong["world_supervision"]
            else:
                wrong_world = wrong["world_supervision"]
                assert isinstance(wrong_world, dict)
                wrong_world["online_known_prefix"] = 48
            with pytest.raises(ValueError, match="W supervision"):
                validate_deployment_abi(wrong)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_default_mode_and_new_mode_have_identical_untrained_online_model():
    torch.manual_seed(3066)
    old, _ = _model_engine(reference_config())
    torch.manual_seed(3066)
    new, _ = _model_engine(_config())
    old.eval()
    new.eval()
    assert list(old.state_dict()) == list(new.state_dict())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    b = _batch(16)
    with torch.no_grad():
        a = sample_action(
            old, b.online, reference_config(), generator=torch.Generator().manual_seed(3067)
        )
        z = sample_action(new, b.online, _config(), generator=torch.Generator().manual_seed(3067))
    torch.testing.assert_close(a.action, z.action, rtol=0, atol=0)


def test_final_world_after_a_real_update_retains_control_prefix_causality():
    m, engine, b = _setup()
    # Exercise learned, nonzero output heads rather than a vacuous zero-init law.
    engine.train_step(b, collect_diagnostics=False)
    m.eval()
    with torch.no_grad():
        _, state, _ = m.encode_online(b.online)
        condition = m.outlet_adapter.observed_world_condition(
            b.future.action_sequence, b.online.history.action_state
        )
        original, _ = m.world.materialize_supervised(
            belief=state.top.facts, action_condition=condition
        )
        assert torch.count_nonzero(original.dynamics.semantic_delta) > 0
        for cut, intervals in ((8, 1), (16, 2), (24, 2), (32, 3)):
            changed = b.future.action_sequence.clone()
            changed[:, cut:, 0] += 0.4
            altered_condition = m.outlet_adapter.observed_world_condition(
                changed, b.online.history.action_state
            )
            altered, _ = m.world.materialize_supervised(
                belief=state.top.facts, action_condition=altered_condition
            )
            for field in ("semantic_delta", "transport_mean", "transport_covariance"):
                before = getattr(original.dynamics, field)
                after = getattr(altered.dynamics, field)
                torch.testing.assert_close(
                    before[:, :intervals], after[:, :intervals], rtol=0, atol=0
                )
            assert not torch.equal(
                original.dynamics.semantic_delta[:, intervals:],
                altered.dynamics.semantic_delta[:, intervals:],
            )


def test_tail_keeps_policy_supervision_without_inventing_complete_world_intervals():
    _, engine, _ = _setup()
    tail = _batch(79)  # One actual action remains before the terminal state.
    assert tail.future.support is not None
    assert tail.future.support.action.sum() == 1
    ledger, _ = engine._forward(
        tail,
        training=False,
        collect_diagnostics=True,
        generator=torch.Generator().manual_seed(3068),
    )
    for key in (
        "future_dynamics",
        "future_semantic_delta",
        "future_transport",
        "future_covariance",
        "future_transition",
        "future_observed_intervals",
    ):
        assert ledger.terms[key].item() == 0.0, key
    assert torch.isfinite(ledger.terms["action_flow"])
    assert ledger.terms["action_flow"].item() > 0
    ledger.total.backward()


@pytest.mark.parametrize("selection", ["pen_7d_continuous_v1", "rdt_right_arm_7d_v1"])
def test_absolute_outlet_keeps_the_real_boundary_across_supervised_blocks(selection: str):
    from clearvla.mainline.model.components import OutletAdapter

    model, _, batch = _setup()
    outlet = OutletAdapter(
        model.outlet_adapter.codec,
        selection=selection,
        world_action_condition_mode="sequence_prefix_v1",
    )
    normalizer = ArrayNormalizer.fit_zscore(
        [np.stack((np.arange(7), np.arange(7) + 4)).astype(np.float32)]
    )
    outlet.configure_action_normalizer(normalizer)
    raw = batch.future.action_sequence
    observed = outlet.observed_world_condition(raw, batch.online.history.action_state)
    torch.testing.assert_close(observed.canonical_value, raw, rtol=0, atol=0)
    previous = torch.cat((batch.online.history.action_state[:, None], raw[:, :-1]), dim=1)
    torch.testing.assert_close(observed.canonical_delta, raw - previous, rtol=0, atol=0)
