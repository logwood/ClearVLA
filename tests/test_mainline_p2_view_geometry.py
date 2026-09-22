"""M6e: named-view geometry reaches the real policy, without invented effects.

Synthetic external image/DINO/T5 transport only; neural modules, loss, optimizer,
checkpoint and sampler are production implementations. No object/trial oracle.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_mainline_future_time_grid import _batch
from test_mainline_future_time_grid import _config as base_config
from test_mainline_state_features import _model_engine
from test_mainline_target_binding import _law
from torch import nn

from clearvla.mainline.checkpoint import active_source_snapshot
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.future_time import CONTROL_INTERVALS
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy, OnlinePolicyCache
from clearvla.mainline.model.target_binding import TargetBinding
from clearvla.mainline.model.view_geometry import ViewConditionedTransport
from clearvla.mainline.p2_geometry import (
    POOLED_TRANSPORT,
    VIEW_CONDITIONED_TRANSPORT,
    p2_geometry_metadata,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.world_control import CandidateControlDomain


def _config() -> ExperimentConfig:
    c = base_config()
    return replace(c, top=replace(c.top, p2_geometry_mode=VIEW_CONDITIONED_TRANSPORT))


Encoded = tuple[ClearVLAMainlinePolicy, MainlineTrainingEngine, TrainingBatch, OnlinePolicyCache]


@pytest.fixture(scope="module")
def encoded() -> Encoded:
    torch.manual_seed(6701)
    m, e = _model_engine(_config())
    b = _batch()
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    with torch.no_grad():
        cache, _, _ = m.encode_online(b.online)
    return m, e, b, cache


def test_configuration_and_metadata_do_not_relabel_old_or_endpoint_values():
    c = _config()
    c.validate()
    assert config_from_mapping(c.as_dict()) == c
    assert "p2_geometry_mode" not in cast(dict, base_config().as_dict()["top"])
    assert (
        load_config("configs/mainline/structural_rebuild_m6e_calvin.json").top.p2_geometry_mode
        == VIEW_CONDITIONED_TRANSPORT
    )
    metadata = p2_geometry_metadata(tuple(c.data.camera_names))
    assert "not-endpoint" in cast(str, metadata["transport_statistic"])
    assert "not-task-risk" in cast(str, metadata["covariance_statistic"])
    assert "not-one-step" in cast(str, metadata["feedback"])
    source = dict(active_source_snapshot(Path(__file__).resolve().parents[1]).files)
    assert "clearvla/mainline/model/view_geometry.py" in source
    assert "clearvla/mainline/p2_geometry.py" in source


@pytest.mark.parametrize(
    "change",
    [
        {"p2_geometry_mode": "guess"},
        {"target_binding_mode": "reader_local_v1"},
        {"world_robot_condition_mode": "implicit_g_only_v1"},
        {"future_time_grid_mode": "legacy_48_v1"},
    ],
)
def test_incompatible_configuration_rejected(change):
    c = _config()
    with pytest.raises(ValueError):
        replace(c, top=replace(c.top, **change)).validate()


@pytest.mark.parametrize("names", [(), ("top", "top"), ("top", "")])
def test_view_roles_are_not_guessed_from_number_of_cameras(names):
    with pytest.raises(ValueError):
        ViewConditionedTransport(hidden=8, camera_names=names)


def test_opposite_image_displacements_survive_before_fusion_and_zero_stays_zero():
    torch.manual_seed(6720)
    module = ViewConditionedTransport(hidden=8, camera_names=("top", "wrist"))
    project = nn.Linear(2, 8, bias=False)
    d = torch.tensor([[[[[0.4, 0.0], [-0.4, 0.0]]]]], requires_grad=True)
    xy, cov = torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 3)
    supported = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    context = module.context_features(xy, cov, supported)
    effect = module.condition(project(d), context, supported, value=True)
    zero = module.condition(project(torch.zeros_like(d)), context, supported, value=True)
    # Independent old-path oracle: a shared linear projection of the vector
    # mean must lose these opposite motions. New view-conditioned values need
    # not obey that erroneous identity. This is not a learned-success test.
    assert torch.count_nonzero(project(d.mean(-2))) == 0
    assert effect.mean(-2).abs().sum() > 0
    assert torch.count_nonzero(zero) == 0
    (grad,) = torch.autograd.grad(effect.square().sum(), (d,))
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_camera_query_is_chart_conditioned_not_one_xy_broadcast_to_both_views():
    torch.manual_seed(6721)
    module = ViewConditionedTransport(hidden=8, camera_names=("top", "wrist"))
    project = nn.Linear(8, 2, bias=False)
    q = torch.randn(2, 3, 2, 8, requires_grad=True)
    xy = module.image_queries(q, project)
    assert xy.shape == (2, 3, 2, 2, 2)
    assert not torch.equal(xy[..., 0, :], xy[..., 1, :])
    (grad,) = torch.autograd.grad(xy.square().sum(), (module.query_role.weight,))
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_geometry_module_masks_invalid_payload_before_products_and_gradients():
    module = ViewConditionedTransport(hidden=8, camera_names=("top", "wrist"))
    xy = torch.tensor([[[[0.2, 0.4], [torch.nan, torch.nan]]]], requires_grad=True)
    cov = torch.zeros(1, 4, 1, 2, 3)
    cov[..., 1, :] = torch.nan
    cov.requires_grad_()
    support = torch.ones(1, 4, 1, 2, dtype=torch.bool)
    support[..., 1] = False
    ctx = module.context_features(xy, cov, support)
    base = torch.randn(1, 4, 1, 2, 8)
    base[..., 1, :] = torch.nan
    base.requires_grad_()
    out = module.condition(base, ctx, support, value=True)
    assert torch.isfinite(out).all() and torch.count_nonzero(out[..., 1, :]) == 0
    gradients = torch.autograd.grad(out.square().sum(), (xy, cov, base))
    for g in gradients:
        assert torch.isfinite(g).all() and torch.count_nonzero(g[..., 1, :]) == 0


@pytest.mark.parametrize("names", [(), ("wrist", "top"), ("wrong", "wrist")])
def test_W_chart_identity_is_checked_at_actual_P2_boundary(encoded: Encoded, names):
    m, _, _, cache = encoded
    d = replace(cache.top.predicted_dynamics, camera_names=names)
    q = torch.zeros_like(cache.factual_dock.protected_detail)
    with pytest.raises(ValueError, match="camera"):
        m.policy_compiler.effect_reader(
            q, d, cache.top.intent.policy_dock(), collect_diagnostics=False
        )


def test_true_value_projection_precedes_K_C_pool_and_is_not_repeated_at_terminal(encoded: Encoded):
    m, _, _, cache = encoded
    reader = m.policy_compiler.effect_reader
    shapes = []
    h = reader.transport_value.register_forward_pre_hook(
        lambda _, args: shapes.append(tuple(args[0].shape))
    )
    q = torch.randn_like(cache.factual_dock.protected_detail)
    with torch.no_grad():
        selected, metrics = reader.spatial_select(
            q,
            cache.top.predicted_dynamics,
            cache.top.intent.policy_dock(),
            collect_diagnostics=True,
        )
        effect, terminal_metrics = reader.temporal_terminal(q, selected, collect_diagnostics=True)
    h.remove()
    assert shapes == [tuple(cache.top.predicted_dynamics.transport_mean.shape)]
    assert selected.geometry_value.shape[-1] == q.shape[-1]
    assert selected.geometry_mode == VIEW_CONDITIONED_TRANSPORT
    assert "object_p2_geometry_selected_feature_rms" in metrics
    assert "object_p2_geometry_terminal_feature_rms" in terminal_metrics
    assert "object_p2_geometry_selected_physical_rms" not in metrics
    assert effect.geometry.shape == q.shape
    with pytest.raises(ValueError):
        reader.temporal_terminal(
            q, replace(selected, geometry_mode=POOLED_TRANSPORT), collect_diagnostics=False
        )


@pytest.mark.parametrize("null", [0.0, 0.9, 1.0])
def test_shared_object_and_null_mass_are_applied_after_view_internal_read(encoded: Encoded, null):
    m, _, _, cache = encoded
    reader, d = m.policy_compiler.effect_reader, cache.top.predicted_dynamics
    q = torch.randn_like(cache.factual_dock.protected_detail)
    dock = cache.top.intent.policy_dock()
    with torch.no_grad():
        full, _ = reader.spatial_select(
            q, d, replace(dock, target_binding=_law(0.0)), collect_diagnostics=False
        )
        selected, _ = reader.spatial_select(
            q, d, replace(dock, target_binding=_law(null)), collect_diagnostics=False
        )
        effect, _ = reader.temporal_terminal(q, selected, collect_diagnostics=False)
    torch.testing.assert_close(selected.geometry_value, full.geometry_value * (1 - null))
    torch.testing.assert_close(selected.semantic_value, full.semantic_value * (1 - null))
    if null == 1.0:
        assert torch.count_nonzero(effect.geometry) == torch.count_nonzero(effect.semantic) == 0


def test_zero_W_motion_does_not_turn_position_role_covariance_into_action_effect(encoded: Encoded):
    m, _, _, cache = encoded
    d = cache.top.predicted_dynamics
    d = replace(d, transport_mean=torch.zeros_like(d.transport_mean))
    q = torch.randn_like(cache.factual_dock.protected_detail)
    effect, _ = m.policy_compiler.effect_reader(
        q, d, cache.top.intent.policy_dock(), collect_diagnostics=True
    )
    assert torch.count_nonzero(effect.geometry) == 0
    view = m.policy_compiler.effect_reader.view_geometry
    assert view is not None
    gradients = torch.autograd.grad(
        effect.geometry.sum(), tuple(view.parameters()), allow_unused=True
    )
    assert all(
        g is None or (torch.isfinite(g).all() and torch.count_nonzero(g) == 0) for g in gradients
    )


def test_camera_permutation_follows_names_and_never_relabels_coordinates(encoded: Encoded):
    m, _, _, cache = encoded
    reader, d = m.policy_compiler.effect_reader, cache.top.predicted_dynamics
    view = reader.view_geometry
    assert view is not None
    other = copy.deepcopy(reader)
    other.view_geometry = ViewConditionedTransport(
        hidden=view.hidden, camera_names=tuple(reversed(view.camera_names))
    )
    other.view_geometry.load_state_dict(view.state_dict(), strict=True)
    values = {
        n: getattr(d, n).flip(2 if n.startswith("camera_") or n.startswith("log_camera") else 3)
        for n in (
            "camera_coordinates",
            "camera_chart_availability",
            "log_camera_chart_availability",
            "transport_mean",
            "transport_covariance",
        )
    }
    changed = replace(d, **values, camera_names=tuple(reversed(d.camera_names)))
    q = torch.randn_like(cache.factual_dock.protected_detail)
    with torch.no_grad():
        a, _ = reader(q, d, cache.top.intent.policy_dock(), collect_diagnostics=False)
        b, _ = other(q, changed, cache.top.intent.policy_dock(), collect_diagnostics=False)
    torch.testing.assert_close(a.semantic, b.semantic)
    torch.testing.assert_close(a.geometry, b.geometry)


def test_object_permutation_keeps_W_chart_and_shared_binding_identity(encoded: Encoded):
    m, _, _, cache = encoded
    reader, d = m.policy_compiler.effect_reader, cache.top.predicted_dynamics
    index = torch.tensor([3, 1, 0, 2])
    assert d.permute(index).camera_names == d.camera_names
    q = torch.randn_like(cache.factual_dock.protected_detail)
    with torch.no_grad():
        a, _ = reader(q, d, cache.top.intent.policy_dock(), collect_diagnostics=False)
        b, _ = reader(
            q,
            d.permute(index),
            cache.top.intent.permute(index).policy_dock(),
            collect_diagnostics=False,
        )
    torch.testing.assert_close(a.semantic, b.semantic)
    torch.testing.assert_close(a.geometry, b.geometry)


def test_unknown_control_NaNs_cannot_enter_feature_common_or_interval_keys(encoded: Encoded):
    m, _, _, cache = encoded
    domain = CandidateControlDomain(known_prefix_steps=8, interval_bounds=CONTROL_INTERVALS)
    d = replace(cache.top.predicted_dynamics, control_domain=domain)
    changed = {}
    for name in ("semantic_delta", "transport_mean", "transport_covariance"):
        value = getattr(d, name).clone()
        value[:, 2:] = torch.nan
        changed[name] = value
    q = torch.randn_like(cache.factual_dock.protected_detail)
    reader = m.policy_compiler.effect_reader
    with torch.no_grad():
        a, _ = reader(q, d, cache.top.intent.policy_dock(), collect_diagnostics=True)
        b, _ = reader(
            q, replace(d, **changed), cache.top.intent.policy_dock(), collect_diagnostics=True
        )
    torch.testing.assert_close(a.semantic, b.semantic, rtol=0, atol=0)
    torch.testing.assert_close(a.geometry, b.geometry, rtol=0, atol=0)


def test_missing_view_has_zero_parameter_and_input_contamination_in_real_P2(encoded: Encoded):
    m, _, _, cache = encoded
    d = cache.top.predicted_dynamics
    supported = torch.tensor([[True, False, True, True]])
    law = TargetBinding.from_logits(torch.zeros(1, 4), torch.zeros(1, 1), supported)
    availability = d.chart_availability.clone()
    availability[:, 1] = 0
    camera = d.camera_chart_availability.clone()
    camera[:, 1] = 0
    camera[:, :, 1] = 0
    changes = {}
    for name in ("transport_mean", "transport_covariance"):
        value = getattr(d, name).clone()
        value[:, :, 1] = torch.nan
        value[:, :, :, 1] = torch.nan
        changes[name] = value.requires_grad_()
    xy = d.camera_coordinates.clone()
    xy[:, 1] = torch.nan
    xy[:, :, 1] = torch.nan
    semantic = d.semantic_delta.clone()
    semantic[:, :, 1] = torch.nan
    d = replace(
        d,
        **changes,
        semantic_delta=semantic,
        camera_coordinates=xy,
        chart_availability=availability,
        camera_chart_availability=camera,
    )
    reader = m.policy_compiler.effect_reader
    q = torch.randn_like(cache.factual_dock.protected_detail)
    effect, _ = reader(
        q, d, replace(cache.top.intent.policy_dock(), target_binding=law), collect_diagnostics=False
    )
    params = tuple(reader.parameters())
    grads = torch.autograd.grad(
        effect.combined().square().sum(), tuple(changes.values()) + params, allow_unused=True
    )
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    for g in grads[:2]:
        assert (
            g is not None
            and torch.count_nonzero(g[:, :, 1]) == 0
            and torch.count_nonzero(g[:, :, :, 1]) == 0
        )


def test_new_geometry_parameter_owners_receive_first_action_loss_gradients(encoded: Encoded):
    m, e, _, cache = encoded
    reader = m.policy_compiler.effect_reader
    view = reader.view_geometry
    assert view is not None
    owned = [p for g in e.optimizer.param_groups for p in g["params"]]
    assert all(sum(p is v for v in owned) == 1 for p in view.parameters())
    output = m.velocity(
        cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
    )
    names, params = zip(*view.named_parameters())
    grads = torch.autograd.grad(output.bottom.physical_velocity[:, 0].square().mean(), params)
    for name, g in zip(names, grads):
        assert torch.isfinite(g).all() and g.abs().sum() > 0, name


@pytest.mark.parametrize("bf16", [False, True])
def test_real_diagnostic_value_gradient_parity_in_P2(encoded: Encoded, bf16):
    m, _, _, cache = encoded
    reader = m.policy_compiler.effect_reader
    q = torch.randn_like(cache.factual_dock.protected_detail)
    results = []
    for flag in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            effect, _ = reader(
                q,
                cache.top.predicted_dynamics,
                cache.top.intent.policy_dock(),
                collect_diagnostics=flag,
            )
            gradients = torch.autograd.grad(
                effect.combined().float().square().sum(),
                tuple(reader.parameters()),
                allow_unused=True,
            )
        results.append((effect.combined().detach(), gradients, torch.get_rng_state().clone()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][2], results[1][2])
    for ga, gb in zip(results[0][1], results[1][1]):
        if ga is None:
            assert gb is None
        else:
            assert gb is not None and torch.isfinite(ga).all()
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_deployment_keeps_same_W_count_and_never_reads_training_targets(
    encoded: Encoded, monkeypatch
):
    m, _, b, _ = encoded
    original = m.world.materialize
    calls = []

    def track(**kwargs):
        result = original(**kwargs)
        calls.append(result[0].dynamics.camera_names)
        return result

    monkeypatch.setattr(m.world, "materialize", track)
    monkeypatch.setattr(
        m.world,
        "materialize_supervised",
        lambda **_: pytest.fail("observed future controls used online"),
    )
    with torch.no_grad():
        sampled = sample_action(
            m, b.online, _config(), generator=torch.Generator().manual_seed(6743)
        )
    assert calls == [tuple(_config().data.camera_names)] * 2
    assert sampled.action.shape == (1, 24, 7) and torch.isfinite(sampled.action).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_production_training_and_tail_action_keep_finite_backward(bf16):
    torch.manual_seed(6744)
    m, e = _model_engine(_config())
    b = _batch(79)
    e.train_step(b, collect_diagnostics=True)
    m.eval()
    m.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        ledger, _ = e._forward(
            b,
            training=False,
            collect_diagnostics=True,
            generator=torch.Generator().manual_seed(6745),
        )
    assert torch.isfinite(ledger.total)
    ledger.total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_exact_checkpoint_reload_and_geometry_ABI(tmp_path, monkeypatch):
    import test_mainline_target_binding as old

    from clearvla.mainline.runtime.deployment import validate_deployment_abi

    original = old.build_deployment_abi

    def build(*args, **kwargs):
        abi = cast(dict[str, Any], original(*args, **kwargs))
        assert abi["p2_geometry"] == p2_geometry_metadata(tuple(_config().data.camera_names))
        for kind in ("missing", "roles", "statistic", "terminal"):
            wrong = copy.deepcopy(abi)
            if kind == "missing":
                del wrong["p2_geometry"]
            elif kind == "roles":
                wrong["p2_geometry"]["camera_names"].reverse()
            elif kind == "statistic":
                wrong["p2_geometry"]["transport_statistic"] = "endpoint"
            else:
                wrong["p2_geometry"]["terminal"] = "physical-xy"
            with pytest.raises(ValueError, match="P2 geometry"):
                validate_deployment_abi(wrong)
        return abi

    monkeypatch.setattr(old, "build_deployment_abi", build)
    monkeypatch.setattr(old, "_config", _config)
    monkeypatch.setattr(old, "_batch", _batch)
    old.test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path)


def test_cache_rejects_relabelled_camera_order_before_numerical_sampling(encoded: Encoded):
    _, _, _, cache = encoded
    candidate = cache.top.candidate_world
    changed = replace(
        candidate, dynamics=replace(candidate.dynamics, camera_names=("wrist", "top"))
    )
    with pytest.raises(ValueError, match="camera"):
        replace(cache, top=replace(cache.top, candidate_world=changed)).validate(_config())


def test_all_missing_sources_remain_finite_zero_geometry_in_real_reader(encoded: Encoded):
    m, _, _, cache = encoded
    d = cache.top.predicted_dynamics
    supported = torch.zeros_like(d.chart_availability[..., 0], dtype=torch.bool)
    law = TargetBinding.from_logits(
        torch.full_like(supported, torch.nan, dtype=torch.float32), torch.zeros(1, 1), supported
    )
    d = replace(
        d,
        chart_availability=torch.zeros_like(d.chart_availability),
        camera_chart_availability=torch.zeros_like(d.camera_chart_availability),
        camera_coordinates=torch.full_like(d.camera_coordinates, torch.nan),
        semantic_delta=torch.full_like(d.semantic_delta, torch.nan),
        transport_mean=torch.full_like(d.transport_mean, torch.nan),
        transport_covariance=torch.full_like(d.transport_covariance, torch.nan),
    )
    q = torch.randn_like(cache.factual_dock.protected_detail)
    effect, _ = m.policy_compiler.effect_reader(
        q, d, replace(cache.top.intent.policy_dock(), target_binding=law), collect_diagnostics=True
    )
    assert torch.isfinite(effect.combined()).all()
    assert torch.count_nonzero(effect.combined()) == 0
    gradients = torch.autograd.grad(
        effect.combined().sum(),
        tuple(m.policy_compiler.effect_reader.parameters()),
        allow_unused=True,
    )
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
