"""M5a real production checks; input transport uses declared synthetic fixtures."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_entity_history import _batch
from test_mainline_entity_motion import _config as base_config
from test_mainline_state_features import _data, _model_engine

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.component_contracts import legacy_named_parameters
from clearvla.mainline.model.target_binding import (
    SHARED_TARGET_BINDING,
    BoundTargetRead,
    SharedTargetBinder,
    TargetBinding,
    TargetEvidence,
)
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint


def _config() -> ExperimentConfig:
    cfg = base_config()
    return replace(
        cfg,
        top=replace(
            cfg.top,
            target_binding_mode=SHARED_TARGET_BINDING,
            p2_spatial_intent_mode="post_pool_only",
        ),
    )


def _law(null=0.2):
    mass = torch.tensor([[0.1, 0.2, 0.3, 0.4]]) * (1 - null)
    return TargetBinding(
        torch.cat((mass, torch.tensor([[null]])), -1).log(), torch.ones(1, 4, dtype=torch.bool)
    )


def test_config_selection_replaces_prior_and_retains_old_identity():
    cfg = _config()
    cfg.validate()
    assert config_from_mapping(cfg.as_dict()) == cfg
    old_top = base_config().as_dict()["top"]
    assert isinstance(old_top, dict)
    assert "target_binding_mode" not in old_top
    assert (
        load_config("configs/mainline/structural_rebuild_m5a_calvin.json").top.target_binding_mode
        == SHARED_TARGET_BINDING
    )
    for values in (
        {"target_binding_mode": "guess"},
        {"p2_spatial_intent_mode": "shared_target_prior_v1"},
        {"history_encoding_mode": "paired_rows_v1"},
    ):
        with pytest.raises(ValueError):
            replace(cfg, top=replace(cfg.top, **values)).validate()


@pytest.mark.parametrize("null_logit", [-1000.0, 0.0, 1000.0])
def test_law_quarantines_missing_objects_and_keeps_null(null_logit):
    scores = torch.tensor([[1.0, float("nan"), -1.0]], requires_grad=True)
    null = torch.tensor([[null_logit]], requires_grad=True)
    law = TargetBinding.from_logits(scores, null, torch.tensor([[True, False, True]]))
    law.validate(batch=1, objects=3, device=scores.device)
    torch.testing.assert_close(law.mass.sum(-1) + law.null_mass[:, 0], torch.ones(1))
    assert law.mass[0, 1] == 0
    (law.mass.square().sum() + law.null_mass.square().sum()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all() and scores.grad[0, 1] == 0
    assert null.grad is not None and torch.isfinite(null.grad).all()


def test_all_missing_is_exact_null_with_finite_backward():
    scores = torch.full((2, 4), torch.nan, requires_grad=True)
    law = TargetBinding.from_logits(scores, torch.zeros(2, 1), torch.zeros(2, 4, dtype=torch.bool))
    assert torch.count_nonzero(law.mass) == 0
    torch.testing.assert_close(law.null_mass, torch.ones(2, 1), rtol=0, atol=0)
    law.mass.sum().backward()
    assert scores.grad is not None and torch.count_nonzero(scores.grad) == 0


@pytest.mark.parametrize("kind", ["shape", "dtype", "nonfinite", "unnormalized", "unsupported"])
def test_malformed_law_rejected(kind):
    law = _law()
    if kind == "shape":
        law = replace(law, log_probability=law.log_probability[:, :-1])
    elif kind == "dtype":
        law = replace(law, log_probability=law.log_probability.double())
    elif kind == "nonfinite":
        law = replace(law, log_probability=law.log_probability * torch.nan)
    elif kind == "unnormalized":
        law = replace(law, log_probability=law.log_probability + 1)
    else:
        law = replace(law, supported=torch.zeros_like(law.supported))
    with pytest.raises((ValueError, TypeError)):
        law.validate(batch=1, objects=4, device=torch.device("cpu"))


def _read_fixture():
    torch.manual_seed(819)
    reader = BoundTargetRead(16, 4)
    e = TargetEvidence(
        torch.randn(1, 4, 6, 16, requires_grad=True), torch.ones(1, 4, 6, dtype=torch.bool)
    )
    return reader, torch.randn(1, 3, 2, 16, requires_grad=True), e


def test_internal_read_has_real_query_key_value_gradients():
    reader, query, e = _read_fixture()
    reader(query, e, _law()).square().sum().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0
    for name, p in reader.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_null_is_not_renormalized_and_full_null_exact_zero():
    reader, q, e = _read_fixture()
    a, b = reader(q, e, _law(0.1)), reader(q, e, _law(0.99))
    assert b.norm() < a.norm() * 0.02
    assert torch.count_nonzero(reader(q, e, _law(1.0))) == 0


def test_object_and_internal_source_permutations():
    reader, q, e = _read_fixture()
    law = _law()
    k, source_order = torch.tensor([2, 0, 3, 1]), torch.tensor([4, 3, 5, 0, 2, 1])
    torch.testing.assert_close(reader(q, e.permute(k), law.permute(k)), reader(q, e, law))
    torch.testing.assert_close(
        reader(q, TargetEvidence(e.tokens[:, :, source_order], e.valid[:, :, source_order]), law),
        reader(q, e, law),
    )


def test_invalid_internal_values_and_empty_objects_have_zero_gradients():
    reader, q, e = _read_fixture()
    valid = e.valid.clone()
    valid[:, :, -1] = False
    valid[:, -1] = False
    tokens = e.tokens.detach().masked_fill(~valid[..., None], torch.nan).requires_grad_()
    scores = torch.zeros(1, 4, requires_grad=True)
    law = TargetBinding.from_logits(scores, torch.zeros(1, 1), valid.any(-1))
    reader(q, TargetEvidence(tokens, valid), law).square().sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad[~valid]) == 0
    assert scores.grad is not None and scores.grad[0, -1] == 0


def test_soft_binding_is_task_conditioned_and_object_equivariant():
    torch.manual_seed(827)
    binder = SharedTargetBinder(16)
    objects, context = torch.randn(2, 4, 16), torch.randn(2, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    a, b = binder(context, objects, valid), binder(-context, objects, valid)
    assert not torch.allclose(a.mass, b.mass)
    k = torch.tensor([1, 3, 2, 0])
    torch.testing.assert_close(binder(context, objects[:, k], valid[:, k]).mass, a.mass[:, k])
    assert (a.mass > 0).all() and (a.null_mass > 0).all()


@pytest.mark.parametrize("center", [0, 16, 79])
def test_real_step_updates_new_owners_with_native_outlet(center):
    torch.manual_seed(829)
    model, engine = _model_engine(_config())
    model.configure_action_normalizer(_data().action_normalizer)
    tracked = {
        n: p
        for n, p in model.named_parameters()
        if any(
            key in n
            for key in (
                "shared_binder",
                ".target_read.",
                ".target_coordinate.",
                ".target_state.",
                ".target_view.",
            )
        )
    }
    assert tracked
    before = {n: p.detach().clone() for n, p in tracked.items()}
    params = [p for g in engine.optimizer.param_groups for p in g["params"]]
    assert len(params) == len({id(p) for p in params})
    assert all(any(p is item for item in params) for p in tracked.values())
    assert len(legacy_named_parameters(model)) == len(list(model.parameters()))
    engine.train_step(_batch(center))
    assert engine.global_step == 1
    for n, p in tracked.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n
        assert not torch.equal(before[n], p), n
    for n, p in model.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n


@torch.no_grad()
def test_same_binding_reaches_coarse_p1_p2_once_per_observation(monkeypatch):
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    batch = _batch(16)
    bindings, consumers = [], []
    binder = model.intent.organizer.shared_binder
    assert binder is not None and model.p1.target_read is not None
    h = binder.register_forward_hook(lambda m, args, output: bindings.append(output))
    hooks = [
        r.register_forward_pre_hook(lambda m, args: consumers.append(args[2]))
        for r in (model.intent.coarse_action.object_read, model.p1.target_read)
    ]
    reader = model.policy_compiler.effect_reader
    original = reader.spatial_select

    def track(query, dynamics, intent, **kwargs):
        consumers.append(intent.target_binding)
        return original(query, dynamics, intent, **kwargs)

    monkeypatch.setattr(reader, "spatial_select", track)
    a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(839))
    assert len(bindings) == 1 and len(consumers) > 2
    assert all(item is bindings[0] for item in consumers)
    b = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(839))
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    assert a.action.shape == (1, 24, 7)
    for hook in [h, *hooks]:
        hook.remove()


def _encoded():
    model, _ = _model_engine(_config())
    model.eval()
    cache, _, _ = model.encode_online(_batch(16).online, collect_diagnostics=False)
    return model, cache


def test_current_target_evidence_reaches_p1_independently_of_w():
    model, cache = _encoded()
    e = cache.top.intent.target_evidence
    assert e is not None
    e.tokens.retain_grad()
    cache.factual_dock.protected_detail.square().mean().backward()
    assert e.tokens.grad is not None and e.tokens.grad[..., 4:, :].abs().sum() > 0
    for p in model.world.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    d = cache.top.predicted_dynamics
    neutral = replace(
        d,
        semantic_delta=torch.zeros_like(d.semantic_delta),
        transport_mean=torch.zeros_like(d.transport_mean),
    )
    effect, _ = model.policy_compiler.effect_reader(
        torch.randn_like(cache.factual_dock.protected_detail),
        neutral,
        cache.top.intent.policy_dock(),
        collect_diagnostics=False,
    )
    assert torch.count_nonzero(effect.semantic) == torch.count_nonzero(effect.geometry) == 0
    assert cache.factual_dock.protected_detail.abs().sum() > 0


def test_p2_k_marginal_is_shared_not_independent_spatial_softmax():
    model, cache = _encoded()
    reader = model.policy_compiler.effect_reader
    d, law = cache.top.predicted_dynamics, _law()
    dock = replace(cache.top.intent.policy_dock(), target_binding=law)
    values = torch.eye(4, d.semantic_delta.shape[-1])[None, None].expand(1, 4, -1, -1).clone()
    d = replace(d, semantic_delta=values)
    q = torch.randn_like(cache.factual_dock.protected_detail)
    a, _ = reader.spatial_select(q, d, dock, collect_diagnostics=False)
    b, _ = reader.spatial_select(-3 * q, d, dock, collect_diagnostics=False)
    expected = torch.einsum("bk,bikd->bid", law.mass, values)[:, None, None].expand_as(
        a.semantic_value
    )
    torch.testing.assert_close(a.semantic_value, expected)
    torch.testing.assert_close(b.semantic_value, expected)
    # Camera-constant per-K values independently expose the K marginal.
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    d = replace(d, transport_mean=v[None, None, :, None].expand_as(d.transport_mean))
    a, _ = reader.spatial_select(q, d, dock, collect_diagnostics=False)
    expected_geo = (law.mass[..., None] * v).sum(1)[:, None, None, None].expand_as(a.geometry_value)
    torch.testing.assert_close(a.geometry_value, expected_geo)
    empty, _ = reader(q, d, replace(dock, target_binding=_law(1.0)), collect_diagnostics=False)
    assert torch.count_nonzero(empty.semantic) == torch.count_nonzero(empty.geometry) == 0


def test_p2_permutation_and_missing_binding():
    model, cache = _encoded()
    reader = model.policy_compiler.effect_reader
    d = cache.top.predicted_dynamics
    q = torch.randn_like(cache.factual_dock.protected_detail)
    k = torch.tensor([3, 1, 0, 2])
    a, _ = reader(q, d, cache.top.intent.policy_dock(), collect_diagnostics=False)
    b, _ = reader(
        q, d.permute(k), cache.top.intent.permute(k).policy_dock(), collect_diagnostics=False
    )
    torch.testing.assert_close(a.semantic, b.semantic)
    torch.testing.assert_close(a.geometry, b.geometry)
    with pytest.raises(ValueError, match="binding"):
        reader(
            q,
            d,
            replace(cache.top.intent.policy_dock(), target_binding=None),
            collect_diagnostics=False,
        )


def test_cpu_bfloat16_real_backward():
    model, _ = _model_engine(_config())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache, _, _ = model.encode_online(_batch(16).online, collect_diagnostics=False)
        loss = cache.factual_dock.protected_detail.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert cache.top.intent.target_binding is not None
    assert cache.top.intent.target_binding.log_probability.dtype == torch.float32
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_checkpoint_owners_exact_reload_deployment_and_abi(tmp_path: Path):
    cfg, data = _config(), _data()
    batch = _batch(16)
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    model.eval()
    lang = tmp_path / "language.pt"
    torch.save({"tokens": batch.online.goal.tokens, "mask": batch.online.goal.mask}, lang)
    dataset = replace(
        identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()),
    )
    identity = build_checkpoint_identity(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset,
        language=ArtifactIdentity.from_file("t5_goal", lang),
        commit="e09a6e9917c3b2fb1f9b4dce0083ea3264ad6b06",
    )
    assert "clearvla/mainline/model/target_binding.py" in dict(identity.source.files)
    profile = resolve_action_state_profile("calvin_relative_7d_v1")
    abi = build_deployment_abi(
        cfg,
        identity,
        action_normalizer=data.action_normalizer,
        state_normalizer=data.state_normalizer,
        data_profile={
            **profile.as_dict(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        gripper_indices=(6,),
        goal_metadata={},
    )
    validate_deployment_abi(abi)
    for kind in ("missing", "changed"):
        wrong = copy.deepcopy(abi)
        observation = cast(dict[str, object], wrong["observation"])
        if kind == "missing":
            del observation["operated_target"]
        else:
            contract = cast(dict[str, object], observation["operated_target"])
            contract["persistent_physical_ids"] = True
        with pytest.raises(ValueError, match="target"):
            validate_deployment_abi(wrong)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=engine.optimizer,
        schedule=engine.schedule,
        config=cfg,
        identity=identity,
        epoch=0,
        global_step=1,
        best_metric=None,
        data_state={
            "action_normalizer": data.action_normalizer.to_dict(),
            "state_normalizer": data.state_normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    m, e = _model_engine(cfg)
    load_checkpoint_exact(
        path, model=m, optimizer=e.optimizer, schedule=e.schedule, config=cfg, identity=identity
    )
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=m,
            optimizer=e.optimizer,
            schedule=e.schedule,
            config=base_config(),
            identity=identity,
        )
    restored = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(853))
        b = sample_action(
            restored.model,
            batch.online,
            restored.config,
            generator=torch.Generator().manual_seed(853),
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)


@pytest.mark.parametrize("bf16", [False, True])
def test_action_only_gradient_reaches_binding_and_factual_target_without_teacher(bf16):
    torch.manual_seed(863)
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, _, _ = model.encode_online(_batch(16).online)
        output = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = output.bottom.physical_velocity.float().square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        if any(
            x in name
            for x in ("shared_binder", ".target_read.", ".target_coordinate.", ".target_view.")
        ):
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, (
                name
            )
    for p in model.training_targets.parameters():
        assert p.grad is None


def test_p2_source_invalid_nan_is_quarantined_before_key_and_value_projections():
    model, cache = _encoded()
    d = cache.top.predicted_dynamics
    supported = torch.tensor([[True, False, True, True]])
    binding = TargetBinding.from_logits(torch.zeros(1, 4), torch.zeros(1, 1), supported)
    validity = d.chart_availability.clone()
    validity[:, 1] = 0
    camera_valid = d.camera_chart_availability.clone()
    camera_valid[:, 1] = 0
    semantic = d.semantic_delta.detach().clone()
    semantic[:, :, 1] = torch.nan
    semantic.requires_grad_()
    transport = d.transport_mean.detach().clone()
    transport[:, :, 1] = torch.nan
    transport.requires_grad_()
    coordinate = d.camera_coordinates.clone()
    coordinate[:, 1] = torch.nan
    covariance = d.transport_covariance.clone()
    covariance[:, :, 1] = torch.nan
    d = replace(
        d,
        chart_availability=validity,
        camera_chart_availability=camera_valid,
        semantic_delta=semantic,
        transport_mean=transport,
        camera_coordinates=coordinate,
        transport_covariance=covariance,
    )
    dock = replace(cache.top.intent.policy_dock(), target_binding=binding)
    q = torch.randn_like(cache.factual_dock.protected_detail)
    effect, _ = model.policy_compiler.effect_reader(q, d, dock, collect_diagnostics=False)
    (effect.semantic.square().sum() + effect.geometry.square().sum()).backward()
    for value in (semantic, transport):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[:, :, 1]) == 0


def test_valid_binding_cannot_be_paired_with_a_different_support_packet():
    reader, q, e = _read_fixture()
    support = _law().supported.clone()
    support[:, 2] = False
    binding = TargetBinding.from_logits(torch.zeros(1, 4), torch.zeros(1, 1), support)
    with pytest.raises(ValueError, match="support"):
        reader(q, e, binding)
