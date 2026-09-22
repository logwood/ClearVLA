"""M2 checks the production candidate materializer and its G3/P1 consumers."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_state_features import _data, _model_engine, _training
from test_mainline_visual_source_time import _config as clock_config

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.candidate_support import (
    FULL_POSTERIOR_SUPPORT,
    candidate_support_metadata,
    posterior_candidate_support,
    posterior_microgrid_expectation,
    sample_candidate_expectation,
)


def _config() -> ExperimentConfig:
    cfg = clock_config()
    return replace(
        cfg, observation=replace(cfg.observation, candidate_support_mode=FULL_POSTERIOR_SUPPORT)
    )


def test_complete_posterior_mode_is_explicit_and_serialized():
    old, cfg = clock_config(), _config()
    assert "candidate_support_mode" not in cast(dict[str, object], old.as_dict()["observation"])
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m2_calvin.json"
        ).observation.candidate_support_mode
        == FULL_POSTERIOR_SUPPORT
    )
    assert candidate_support_metadata(FULL_POSTERIOR_SUPPORT)["candidate_count"] == 64
    with pytest.raises(ValueError):
        replace(
            cfg, observation=replace(cfg.observation, candidate_support_mode="invented")
        ).validate()


def test_equal_moments_do_not_collapse_different_spatial_distributions():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    positions = torch.stack((x, torch.zeros_like(x)), -1)[None, None]
    p = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0])
    q = torch.tensor([0.125, 0.0, 0.75, 0.0, 0.125])
    torch.testing.assert_close((p * x).sum(), (q * x).sum())
    torch.testing.assert_close((p * x * x).sum(), (q * x * x).sum())
    support = torch.ones(1, 1, 1, 1, 1)
    source = torch.zeros(1, 1, 1, 1, 2)
    correction = torch.zeros(1, 1, 1, 1, 1, 2)
    results = []
    for prob in (p, q):
        logit = prob.clamp_min(1e-20).log().reshape(1, 1, 1, 1, 1, 5).requires_grad_()
        result = posterior_candidate_support(
            positions=positions,
            logits=logit,
            correction=correction,
            source_centers=source,
            support=support,
        )
        torch.testing.assert_close(
            result.current_coordinates.flatten(0, -2), positions.flatten(0, -2)
        )
        torch.testing.assert_close(
            result.parent_log_probability.exp().sum(-1), torch.ones_like(support)
        )
        # Nonlinear observed content can distinguish distributions with equal moments.
        feature = (result.parent_log_probability.exp() * x.pow(4)).sum()
        feature.backward()
        assert logit.grad is not None and torch.isfinite(logit.grad).all()
        results.append(feature.item())
    assert results[0] == pytest.approx(0.0625) and results[1] == pytest.approx(0.25)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_posterior_gauge_and_candidate_permutation_preserve_correspondence(dtype):
    generator = torch.Generator().manual_seed(9)
    points = (torch.rand(1, 2, 7, 2, generator=generator) * 2 - 1).to(dtype)
    logits = torch.randn(1, 2, 2, 2, 4, 7, generator=generator).to(dtype)
    correction = torch.randn(1, 2, 2, 2, 4, 2, generator=generator).tanh() * 0.2
    support = torch.full((1, 2, 2, 2, 4), 0.1)
    source = torch.zeros(1, 2, 2, 2, 2)
    ref = posterior_candidate_support(
        positions=points,
        logits=logits,
        correction=correction,
        source_centers=source,
        support=support,
    )
    perm = torch.tensor([6, 0, 5, 1, 4, 2, 3])
    test = posterior_candidate_support(
        positions=points[:, :, perm],
        logits=logits[..., perm].float() + 8,
        correction=correction,
        source_centers=source,
        support=support,
    )
    torch.testing.assert_close(ref.current_coordinates[..., perm, :], test.current_coordinates)
    torch.testing.assert_close(ref.parent_log_probability[..., perm], test.parent_log_probability)
    assert (
        ref.current_coordinates.abs().max() <= 1
        and ref.parent_log_probability.dtype == torch.float32
    )


def test_observed_content_is_expected_at_support_not_sampled_at_empty_mean():
    chart = torch.tensor([1.0, 1.0, 99.0, 3.0, 3.0]).reshape(1, 1, 1, 5, 1)
    points = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]).reshape(1, 1, 1, 1, 1, 2, 2).requires_grad_()
    logits = torch.zeros(1, 1, 1, 1, 1, 2, requires_grad=True)
    value = sample_candidate_expectation(chart, points, logits.softmax(-1))
    assert value.item() == 2.0
    value.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert points.grad is not None and torch.isfinite(points.grad).all()
    # Independent current-image read at the mean would hallucinate the center's 99.
    center = F.grid_sample(
        chart.permute(0, 1, 4, 2, 3).flatten(0, 1), torch.zeros(1, 1, 1, 2), align_corners=True
    )
    assert center.item() == 99.0


def _micro_case():
    axis = torch.linspace(-1, 1, 17)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    chart = torch.stack((x, y, x * y))[None, None]
    points = torch.tensor([[-0.6, -0.4], [0.5, 0.3]]).reshape(1, 1, 1, 1, 1, 2, 2)
    measure = torch.tensor([0.3, 0.7]).log().reshape(1, 1, 1, 1, 1, 1, 1, 2)
    center = (
        F.grid_sample(chart.flatten(0, 1), points.reshape(1, 1, 2, 2), align_corners=True)
        .permute(0, 2, 3, 1)
        .reshape(1, 1, 1, 1, 1, 2, 3)
    )
    return chart, points, measure, center


def test_p1_microgrid_is_local_to_each_candidate_not_a_global_thumbnail():
    chart, points, logits, center = _micro_case()
    logits.requires_grad_()
    rgb, detail, coords = posterior_microgrid_expectation(
        torch.ones(1, 1, 1, 1, 1, 1, 1),
        logits,
        torch.ones(1, 1, 1, 1, 1, 2, dtype=torch.bool),
        points,
        torch.full((1, 1, 1, 1, 1), 0.1),
        chart,
        chart,
        center,
        center,
    )
    expectations = []
    expected_coords = []
    for y in (-0.1, 0.0, 0.1):
        for x in (-0.1, 0.0, 0.1):
            xy = points.flatten(0, -2) + torch.tensor([x, y])
            values = torch.stack((xy[:, 0], xy[:, 1], xy[:, 0] * xy[:, 1]), -1)
            expectations.append((values * torch.tensor([0.3, 0.7])[:, None]).sum(0))
            expected_coords.append((xy * torch.tensor([0.3, 0.7])[:, None]).sum(0))
    torch.testing.assert_close(rgb[0, 0, 0], torch.stack(expectations))
    torch.testing.assert_close(detail, rgb)
    torch.testing.assert_close(coords[0, 0, 0], torch.stack(expected_coords))
    (rgb.square().sum() + coords.square().sum()).backward()
    assert (
        logits.grad is not None
        and torch.isfinite(logits.grad).all()
        and logits.grad.abs().sum() > 0
    )


@pytest.mark.parametrize("valid", [True, False])
def test_microgrid_zero_support_and_extreme_logits_have_finite_reverse(valid):
    chart, points, logits, center = _micro_case()
    logits = torch.tensor([1000.0, -1000.0]).reshape_as(logits).requires_grad_()
    out = posterior_microgrid_expectation(
        torch.ones(1, 1, 1, 1, 1, 1, 1),
        logits,
        torch.full((1, 1, 1, 1, 1, 2), valid, dtype=torch.bool),
        points,
        torch.full((1, 1, 1, 1, 1), 0.1),
        chart,
        chart,
        center,
        center,
    )
    torch.stack([t.square().mean() for t in out]).sum().backward()
    assert all(torch.isfinite(t).all() for t in out)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    if not valid:
        assert all(torch.count_nonzero(t) == 0 for t in out)
        assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("center", [0, 1, 8, 79])
def test_actual_encoder_g2_g3_p1_and_teacher_consume_complete_support(center):
    model, _ = _model_engine(_config())
    model.eval()
    batch = _training(_data(), (center,))
    with torch.no_grad():
        cache, training, _ = model.encode_online(batch.online)
        state = training.observation.progressive_state
        assert state.dynamic_fine_values is not None and state.dynamic_fine_coordinates is not None
        assert state.dynamic_fine_values.shape[-2] == 64
        assert state.candidate_support_mode == FULL_POSTERIOR_SUPPORT
        assert state.coarse_logits is not None and state.coarse_logits.dtype == torch.float32
        assert (
            state.fine_log_probability is not None
            and torch.isfinite(state.fine_log_probability).all()
        )
        assert (
            state.g2_geometry_probability is not None
            and state.bank.dense_current_dino_content is not None
        )
        actual = sample_candidate_expectation(
            state.bank.dense_current_dino_content,
            state.dynamic_fine_coordinates,
            state.g2_geometry_probability,
        )
        assert state.grounded_fact_set is not None
        torch.testing.assert_close(actual, state.grounded_fact_set.content_slots)
        targets, _ = model.build_training_targets(training, batch.future)
        assert targets.teacher_dynamics is not None
        cache.validate(_config())
        assert torch.isfinite(targets.teacher_dynamics.semantic_delta).all()


def test_real_g2_uses_parent_distribution_not_just_reused_moments():
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    with torch.no_grad():
        _, training, _ = model.encode_online(_training(_data(), (8,)).online)
        original = training.observation.progressive_state
        organizer = model.observation.compiler.encoder.progressive_grounding_address
        sampler = model.observation.compiler.encoder.soft_address_compiler
        assert organizer is not None and sampler is not None
        assert organizer.g2_typed_query is not None and organizer.g2_typed_rectifier is not None
        for p in organizer.g2_typed_query.parameters():
            p.zero_()
        for p in organizer.g2_typed_rectifier.parameters():
            p.zero_()
        # Hold the entire moment/key/query input fixed; change ONLY the retained parent law.
        assert original.coarse_logits is not None
        outputs = []
        for first, last in ((0, 63), (7, 56)):
            logits = torch.full_like(original.coarse_logits, -60.0)
            logits[..., first] = logits[..., last] = 0.0
            state = replace(original, stage=1, coarse_logits=logits)
            rollout = torch.zeros(
                1, organizer.anchors * organizer.cameras * organizer.grid**2, organizer.hidden
            )
            state = organizer.update(
                state,
                rollout,
                stage=2,
                candidate_sampler=sampler.progressive_fine_candidates,
                collect_diagnostics=False,
            )
            assert state.fine_probability is not None
            probability = state.fine_probability
            assert probability[..., first].min() > 0.49 and probability[..., last].min() > 0.49
            outputs.append(probability.clone())
        assert not torch.equal(*outputs)


@pytest.mark.parametrize("bf16", [False, True])
def test_real_action_velocity_backward_reaches_g1_without_teacher(bf16):
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    batch = _training(_data(), (1,))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, training, _ = model.encode_online(batch.online)
        state = training.observation.progressive_state
        assert state.coarse_logits is not None
        state.coarse_logits.retain_grad()
        output = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = output.bottom.physical_velocity.float().square().mean()
    loss.backward()
    assert state.coarse_logits.grad is not None and state.coarse_logits.grad.abs().sum() > 0
    assert torch.isfinite(state.coarse_logits.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_train_step_and_exact_checkpoint_deployment_keep_new_contract(tmp_path: Path):
    cfg, data = _config(), _data()
    batch = _training(data, (8,))
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    model.eval()
    language = tmp_path / "language.pt"
    torch.save({"tokens": batch.online.goal.tokens, "mask": batch.online.goal.mask}, language)
    dataset = replace(
        identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()),
    )
    identity = build_checkpoint_identity(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset,
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="5" * 40,
    )
    assert "clearvla/vision/candidate_support.py" in dict(identity.source.files)
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
    for mutation in ("missing", 49, True):
        bad = copy.deepcopy(abi)
        obs = cast(dict[str, object], bad["observation"])
        if mutation == "missing":
            del obs["candidate_support"]
        else:
            cast(dict[str, object], obs["candidate_support"])["candidate_count"] = mutation
        with pytest.raises(ValueError):
            validate_deployment_abi(bad)
    path = tmp_path / "posterior-support.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=engine.optimizer,
        schedule=engine.schedule,
        config=cfg,
        identity=identity,
        epoch=0,
        global_step=engine.global_step,
        best_metric=None,
        data_state={
            "action_normalizer": data.action_normalizer.to_dict(),
            "state_normalizer": data.state_normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    restored, re = _model_engine(cfg)
    load_checkpoint_exact(
        path,
        model=restored,
        optimizer=re.optimizer,
        schedule=re.schedule,
        config=cfg,
        identity=identity,
    )
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=restored,
            optimizer=re.optimizer,
            schedule=re.schedule,
            config=clock_config(),
            identity=identity,
        )
    bundle = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(73))
        b = sample_action(
            bundle.model, batch.online, bundle.config, generator=torch.Generator().manual_seed(73)
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    assert a.action.shape == (1, 24, 7)


def test_channel_tiled_content_expectation_matches_full_reference_and_gradients():
    torch.manual_seed(117)
    chart = torch.randn(2, 2, 5, 6, 257, requires_grad=True)
    points = (torch.rand(2, 2, 2, 3, 2, 4, 2) * 1.6 - 0.8).requires_grad_()
    logits = torch.randn(2, 2, 2, 3, 2, 4, requires_grad=True)
    prob = logits.softmax(-1)
    actual = sample_candidate_expectation(chart, points, prob)
    raw = F.grid_sample(
        chart.permute(0, 1, 4, 2, 3).reshape(4, 257, 5, 6),
        points.reshape(4, -1, 4, 2),
        align_corners=True,
    )
    expected = (raw * prob.reshape(4, 1, -1, 4)).sum(-1).transpose(1, 2).reshape(2, 2, 2, 3, 2, 257)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weight = torch.randn_like(actual)
    grads_a = torch.autograd.grad(
        (actual * weight).sum(), (chart, points, logits), retain_graph=True
    )
    grads_b = torch.autograd.grad((expected * weight).sum(), (chart, points, logits))
    for a, b in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("all_invalid", [False, True])
def test_microgrid_quarantines_unsupported_candidate_payload_before_arithmetic(all_invalid):
    chart, points, logits, center = _micro_case()
    valid = torch.tensor([not all_invalid, False]).reshape(1, 1, 1, 1, 1, 2)
    bad_points = torch.where(
        valid[..., None], points, torch.full_like(points, torch.nan)
    ).requires_grad_()
    bad_center = torch.where(
        valid[..., None], center, torch.full_like(center, torch.nan)
    ).requires_grad_()
    radius = torch.full((1, 1, 1, 1, 1), torch.nan if all_invalid else 0.1, requires_grad=True)
    bad_logits = logits.masked_fill(~valid[:, None, None], torch.nan).requires_grad_()
    out = posterior_microgrid_expectation(
        torch.ones(1, 1, 1, 1, 1, 1, 1),
        bad_logits,
        valid,
        bad_points,
        radius,
        chart,
        chart,
        bad_center,
        bad_center,
    )
    assert all(torch.isfinite(x).all() for x in out)
    torch.stack([x.square().sum() for x in out]).sum().backward()
    for x in (bad_points, bad_center, radius, bad_logits):
        assert x.grad is not None and torch.isfinite(x.grad).all()
    assert bad_points.grad is not None and torch.count_nonzero(bad_points.grad[~valid]) == 0
    if all_invalid:
        assert all(torch.count_nonzero(x) == 0 for x in out)


def test_microgrid_query_partition_and_candidate_permutation_preserve_read():
    chart, points, logits, center = _micro_case()
    logits = logits.expand(1, 3, 2, 1, 1, 1, 1, 2).clone()
    logits[:, 1, :, ..., 0] += 0.6
    routes = torch.ones(1, 3, 2, 1, 1, 1, 1)
    valid = torch.ones(1, 1, 1, 1, 1, 2, dtype=torch.bool)
    radius = torch.full((1, 1, 1, 1, 1), 0.1)
    whole = posterior_microgrid_expectation(
        routes, logits, valid, points, radius, chart, chart, center, center
    )
    chunks = [
        posterior_microgrid_expectation(
            routes[:, i : i + 1],
            logits[:, i : i + 1],
            valid,
            points,
            radius,
            chart,
            chart,
            center,
            center,
        )
        for i in range(3)
    ]
    permuted = posterior_microgrid_expectation(
        routes,
        logits.flip(-1),
        valid.flip(-1),
        points.flip(-2),
        radius,
        chart,
        chart,
        center.flip(-2),
        center.flip(-2),
    )
    for i in range(3):
        torch.testing.assert_close(whole[i], torch.cat([c[i] for c in chunks], 1), rtol=0, atol=0)
        torch.testing.assert_close(whole[i], permuted[i], rtol=0, atol=0)


def test_zero_radius_means_the_center_of_each_hypothesis_not_one_global_index():
    chart, points, logits, center = _micro_case()
    output = posterior_microgrid_expectation(
        torch.ones(1, 1, 1, 1, 1, 1, 1),
        logits,
        torch.ones(1, 1, 1, 1, 1, 2, dtype=torch.bool),
        points,
        torch.zeros(1, 1, 1, 1, 1),
        chart,
        chart,
        center,
        center,
    )
    for value in output:
        torch.testing.assert_close(value, value[..., 4:5, :].expand_as(value), rtol=0, atol=0)
