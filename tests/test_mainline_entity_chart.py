"""M4b: actual current-image reads/writes, including production consumers."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_entity_context import _config as context_config
from test_mainline_state_features import _data, _model_engine, _training
from test_mainline_structural_contracts import _local_facts

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.model.grounding import DenseObjectGrounder
from clearvla.mainline.model.types import LocalFactSet
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.entity_chart import (
    CURRENT_IMAGE_CHART,
    QUERY_CHART,
    CurrentImageSupport,
    current_image_centers,
    entity_chart_metadata,
    pushforward_to_current_image,
)


def _config() -> ExperimentConfig:
    cfg = context_config()
    return replace(cfg, top=replace(cfg.top, entity_chart_mode=CURRENT_IMAGE_CHART))


def _support(xy: torch.Tensor, probability: torch.Tensor | None = None) -> CurrentImageSupport:
    if probability is None:
        probability = torch.full(xy.shape[:-1], 1 / xy.shape[-2])
    log_p = torch.where(probability > 0, probability, 1.0).log().float()
    return CurrentImageSupport(xy, probability.float(), probability > 0, log_p)


def _local_at(xy: tuple[float, float], *, empty: bool = False) -> LocalFactSet:
    local = _local_facts()
    prefix = local.slot_validity.shape[:-1]
    coordinates = torch.tensor(xy).expand(*prefix, 2, 2).clone()
    probability = torch.full((*prefix, 2), 0.5)
    valid = torch.ones_like(probability, dtype=torch.bool)
    local_validity = torch.ones_like(local.slot_validity)
    if empty:
        coordinates.fill_(torch.nan)
        probability.zero_()
        valid.zero_()
        local_validity.zero_()
    return replace(
        local,
        slot_validity=local_validity,
        current_image_support=CurrentImageSupport(
            coordinates, probability, valid, torch.where(valid, probability, 1.0).log()
        ),
    )


def test_selection_and_chart_identity_are_explicit():
    old, cfg = context_config(), _config()
    assert "entity_chart_mode" not in cast(dict[str, object], old.as_dict()["top"])
    assert config_from_mapping(old.as_dict()) == old
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert (
        load_config("configs/mainline/structural_rebuild_m4b_calvin.json").top.entity_chart_mode
        == CURRENT_IMAGE_CHART
    )
    for changes in ({"entity_chart_mode": "guess"}, {"entity_context_mode": "candidate_only_v1"}):
        with pytest.raises(ValueError):
            replace(cfg, top=replace(cfg.top, **changes)).validate()
    with pytest.raises(ValueError):
        replace(
            cfg, observation=replace(cfg.observation, local_ownership_mode="independent_typed_v1")
        ).validate()
    assert entity_chart_metadata(CURRENT_IMAGE_CHART)["reverse_lookup_axes"] == "current_camera_y_x"
    assert entity_chart_metadata(QUERY_CHART)["reverse_lookup_axes"] == "query_camera_y_x"


def test_separated_positions_are_not_written_to_their_empty_barycenter():
    xy = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]).reshape(1, 1, 1, 1, 1, 2, 2)
    image = pushforward_to_current_image(
        torch.ones(1, 1, 1, 1, 1, 1), _support(xy), rows=3, columns=5
    )
    assert image[0, 0, 0, 1, 2] == 0
    assert image[0, 0, 0, 1, 0] == image[0, 0, 0, 1, 4] == 0.5
    torch.testing.assert_close(current_image_centers(image), torch.zeros(1, 1, 1, 2))


@pytest.mark.parametrize("rows,cols", [(1, 1), (1, 7), (6, 1), (4, 7)])
def test_mass_camera_and_object_identity_survive_edges(rows, cols):
    xy = (
        torch.tensor([[-1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        .reshape(1, 1, 1, 1, 1, 4, 2)
        .expand(2, 3, 2, 4, 2, 4, 2)
    )
    mass = torch.rand(2, 4, 3, 2, 4, 2, generator=torch.Generator().manual_seed(51))
    actual = pushforward_to_current_image(mass, _support(xy), rows=rows, columns=cols)
    torch.testing.assert_close(actual.sum((-2, -1)), mass.sum((-3, -2, -1)), atol=3e-6, rtol=2e-6)
    order = torch.tensor([2, 0, 1])
    permuted = pushforward_to_current_image(
        mass[:, :, order], _support(xy[:, order]), rows=rows, columns=cols
    )
    torch.testing.assert_close(permuted, actual[:, :, order])


def test_pushforward_is_the_adjoint_of_real_image_reads_and_keeps_gradients():
    gen = torch.Generator().manual_seed(61)
    xy = (torch.rand(1, 2, 2, 3, 2, 5, 2, generator=gen) * 1.7 - 0.85).requires_grad_()
    logits = torch.randn(1, 2, 2, 3, 2, 5, generator=gen, requires_grad=True)
    prob = logits.softmax(-1)
    mass = torch.rand(1, 3, 2, 2, 3, 2, generator=gen, requires_grad=True)
    image_values = torch.randn(1, 3, 2, 5, 7, generator=gen)
    push = pushforward_to_current_image(mass, _support(xy, prob), rows=5, columns=7)
    lhs = (push * image_values).sum()
    # Independent reference: actual grid_sample pullback, not the splat helper.
    values = image_values.permute(0, 2, 1, 3, 4).reshape(2, 3, 5, 7)
    grid = xy.reshape(2, -1, 1, 2)
    pulled = (
        F.grid_sample(values, grid, align_corners=True)
        .reshape(1, 2, 3, 2, 3, 2, 5)
        .permute(0, 2, 1, 3, 4, 5, 6)
    )
    rhs = (pulled * mass[..., None] * prob[:, None]).sum()
    torch.testing.assert_close(lhs, rhs, atol=2e-6, rtol=2e-6)
    a = torch.autograd.grad(lhs, (xy, logits, mass), retain_graph=True)
    b = torch.autograd.grad(rhs, (xy, logits, mass))
    for u, v in zip(a, b, strict=True):
        assert torch.isfinite(u).all() and u.abs().sum() > 0
        torch.testing.assert_close(u, v, atol=4e-6, rtol=4e-5)


def test_reordering_query_candidates_does_not_move_current_image_evidence():
    xy = torch.rand(1, 2, 2, 3, 2, 4, 2) * 2 - 1
    prob = torch.randn(1, 2, 2, 3, 2, 4).softmax(-1)
    mass = torch.rand(1, 4, 2, 2, 3, 2)
    a = pushforward_to_current_image(mass, _support(xy, prob), rows=4, columns=5)
    b = pushforward_to_current_image(
        mass.flip(4), _support(xy.flip(3), prob.flip(3)), rows=4, columns=5
    )
    torch.testing.assert_close(a, b)
    c = pushforward_to_current_image(
        mass,
        _support(xy.repeat_interleave(2, -2), prob.repeat_interleave(2, -1) / 2),
        rows=4,
        columns=5,
    )
    torch.testing.assert_close(a, c)


def test_invalid_payload_is_quarantined_before_indices_products_and_gradients():
    xy = torch.full((1, 1, 1, 1, 2, 3, 2), torch.nan, requires_grad=True)
    prob = torch.zeros(1, 1, 1, 1, 2, 3)
    support = CurrentImageSupport(
        xy, prob, torch.zeros_like(prob, dtype=torch.bool), torch.zeros_like(prob)
    )
    mass = torch.full((1, 4, 1, 1, 1, 2), torch.nan, requires_grad=True)
    out = pushforward_to_current_image(mass, support, rows=4, columns=5)
    assert torch.count_nonzero(out) == 0
    out.square().sum().backward()
    for tensor in (xy, mass):
        assert (
            tensor.grad is not None
            and torch.isfinite(tensor.grad).all()
            and torch.count_nonzero(tensor.grad) == 0
        )
    assert torch.count_nonzero(current_image_centers(out)) == 0


def test_g3_competition_probability_law_is_fp32_under_bf16_autocast():
    torch.manual_seed(67)
    grounder = DenseObjectGrounder(hidden=16, content_dim=8, route_dim=8, objects=3)
    slots = torch.randn(2, 3, 16)
    candidates = torch.randn(2, 5, 16)
    validity = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0]]
    )
    prior = torch.full((2, 5, 1), 0.2)
    log_prior = prior.log()
    prior_support = torch.ones(2, 5, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = grounder._competition(
            slots,
            candidates,
            validity,
            prior,
            log_prior,
            prior_support,
        )
    with torch.autocast("cpu", enabled=False):
        expected = grounder._competition(
            slots,
            candidates,
            validity,
            prior,
            log_prior,
            prior_support,
        )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value.dtype == torch.float32
        torch.testing.assert_close(actual_value, expected_value, rtol=0.0, atol=0.0)


def test_g3_pair_diagnostics_are_fp32_under_bf16_autocast():
    torch.manual_seed(68)
    content = torch.randn(2, 4, 16).to(torch.bfloat16)
    probability = torch.randn(2, 4, 9).softmax(dim=-1).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cosine = DenseObjectGrounder._pair_cosine(content)
        overlap = DenseObjectGrounder._pair_overlap(probability)
    with torch.autocast("cpu", enabled=False):
        expected_cosine = DenseObjectGrounder._pair_cosine(content)
        expected_overlap = DenseObjectGrounder._pair_overlap(probability)
    assert cosine.dtype == torch.float32
    assert overlap.dtype == torch.float32
    torch.testing.assert_close(cosine, expected_cosine, rtol=0.0, atol=0.0)
    torch.testing.assert_close(overlap, expected_overlap, rtol=0.0, atol=0.0)


def test_malformed_support_is_rejected_not_silently_renormalized():
    xy = torch.zeros(1, 1, 1, 1, 1, 2, 2)
    support = _support(xy)
    for invalid in (
        replace(support, coordinates=xy + 2),
        replace(support, probability=support.probability * 0.5),
        replace(support, valid=torch.zeros_like(support.valid)),
        replace(support, probability=support.probability.double()),
    ):
        with pytest.raises((ValueError, TypeError)):
            invalid.validate()
    with pytest.raises(ValueError, match="disagree"):
        support.validate(torch.zeros(1, 1, 1, 1, 1, 1))


@pytest.mark.parametrize("empty", [False, True])
def test_real_binder_writes_in_current_pixels_and_cannot_mask_uncovered_loss(empty):
    torch.manual_seed(73)
    local = _local_at((1.0, -1.0), empty=empty)
    grounder = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, entity_chart_mode=CURRENT_IMAGE_CHART
    )
    facts, _ = grounder(local, collect_diagnostics=True)
    p = facts.object_to_chart
    assert facts.object_chart_mode == CURRENT_IMAGE_CHART
    torch.testing.assert_close(facts.camera_coordinates, current_image_centers(p))
    mask = torch.ones_like(p, dtype=torch.bool)
    mask[..., 0, -1] = False
    assert torch.count_nonzero(p[mask]) == 0
    if empty:
        assert torch.count_nonzero(p) == 0
    else:
        torch.testing.assert_close(p.sum((2, 3, 4)), torch.ones_like(p[:, :, 0, 0, 0]))
    target = local.target_dino_content.detach().float()
    obs = local.cell_observed[..., 0]
    expected = (facts.reconstructed_dino.float() - target).square().mean(-1)[obs].mean()
    torch.testing.assert_close(facts.reconstruction_error, expected)
    assert facts.reconstruction_error > 0
    swapped = facts.permute(torch.tensor([3, 1, 0, 2]))
    swapped.validate()
    assert swapped.object_chart_mode == CURRENT_IMAGE_CHART
    with pytest.raises(ValueError, match="mode"):
        replace(facts, object_chart_mode=QUERY_CHART).validate()
    with pytest.raises(ValueError, match="matching"):
        DenseObjectGrounder(hidden=32, content_dim=16, route_dim=8)(local)


@pytest.mark.parametrize("bf16,center", [(False, 0), (False, 8), (True, 8)])
def test_production_g_teacher_and_action_loss_keep_one_current_image_measure(bf16, center):
    cfg = _config()
    model, engine = _model_engine(cfg)
    batch = _training(_data(), (center,))
    dino = torch.randn(
        batch.online.observation.dino_history.shape, generator=torch.Generator().manual_seed(79)
    )
    batch = replace(
        batch,
        online=replace(
            batch.online, observation=replace(batch.online.observation, dino_history=dino)
        ),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        _, train, _ = model.encode_online(batch.online)
        local, facts = train.observation.local_facts, train.top.facts
        support = local.current_image_support
        assert support is not None and support is facts.dense_chart.current_image_support
        support.coordinates.retain_grad()
        support.log_probability.retain_grad()
        torch.testing.assert_close(
            facts.camera_coordinates,
            current_image_centers(facts.object_to_chart).to(facts.camera_coordinates.dtype),
        )
        objective = facts.reconstruction_error + facts.content.square().mean()
    objective.backward()
    for t in (support.coordinates, support.log_probability):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert not local.target_dino_content.requires_grad
    engine.optimizer.zero_grad(set_to_none=True)
    engine.train_step(batch)
    assert engine.global_step == 1


def test_real_checkpoint_and_online_sampling_retain_image_coordinate_contract(tmp_path: Path):
    cfg, data = _config(), _data()
    batch = _training(data, (8,))
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
        commit="7" * 40,
    )
    assert "clearvla/vision/entity_chart.py" in dict(identity.source.files)
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
    for mutation in ("missing", "wrong"):
        bad = copy.deepcopy(abi)
        obs = cast(dict[str, object], bad["observation"])
        if mutation == "missing":
            del obs["entity_chart"]
        else:
            cast(dict[str, object], obs["entity_chart"])["reverse_lookup_axes"] = "query_camera_y_x"
        with pytest.raises(ValueError):
            validate_deployment_abi(bad)
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
            config=context_config(),
            identity=identity,
        )
    deployed = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(83))
        b = sample_action(
            deployed.model,
            batch.online,
            deployed.config,
            generator=torch.Generator().manual_seed(83),
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)
    assert a.action.shape[-1] == 7


@pytest.mark.parametrize("log_scale", [0.0, -100.0, -1000.0])
def test_underflowed_spatial_mass_has_finite_conditional_gradients(log_scale):
    from clearvla.vision.entity_chart import pushforward_log_to_current_image

    xy = torch.tensor([[-0.7, 0.2], [0.6, -0.1]]).reshape(1, 1, 1, 1, 1, 2, 2).requires_grad_()
    logits = torch.tensor([0.1, -0.2], requires_grad=True)
    log_p = logits.log_softmax(0).reshape(1, 1, 1, 1, 1, 2)
    spatial = CurrentImageSupport(xy, log_p.exp(), torch.ones_like(log_p, dtype=torch.bool), log_p)
    source = torch.tensor([log_scale - 1, log_scale - 2], requires_grad=True)
    log_mass = source.reshape(1, 2, 1, 1, 1, 1)
    measure = pushforward_log_to_current_image(
        log_mass, torch.ones_like(log_mass, dtype=torch.bool), spatial, rows=4, columns=5
    )
    conditional, _ = measure.normalized((1,))
    objective = conditional.square().sum() + measure.camera_centers().square().sum()
    objective.backward()
    for tensor in (source, xy, logits):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    observed = conditional.sum(1)
    torch.testing.assert_close(
        observed[measure.supported.any(1)],
        torch.ones_like(observed[measure.supported.any(1)]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_log_and_linear_pushforward_agree_without_numerical_underflow():
    from clearvla.vision.entity_chart import pushforward_log_to_current_image

    torch.manual_seed(91)
    xy = torch.rand(1, 2, 2, 3, 2, 5, 2) * 1.8 - 0.9
    p = torch.randn(1, 2, 2, 3, 2, 5).softmax(-1)
    mass = torch.rand(1, 4, 2, 2, 3, 2) + 0.1
    spatial = _support(xy, p)
    a = pushforward_to_current_image(mass, spatial, rows=5, columns=7)
    b = pushforward_log_to_current_image(
        mass.log(), torch.ones_like(mass, dtype=torch.bool), spatial, rows=5, columns=7
    )
    torch.testing.assert_close(a, torch.where(b.supported, b.log_mass.exp(), 0.0))
    torch.testing.assert_close(current_image_centers(a), b.camera_centers())


def test_empty_log_segments_have_zero_outputs_and_finite_zero_gradients():
    from clearvla.vision.entity_chart import pushforward_log_to_current_image

    xy = torch.full((1, 1, 1, 1, 1, 2, 2), torch.nan, requires_grad=True)
    logs = torch.zeros(1, 1, 1, 1, 1, 2, requires_grad=True)
    spatial = CurrentImageSupport(
        xy, torch.zeros_like(logs), torch.zeros_like(logs, dtype=torch.bool), logs
    )
    source = torch.full((1, 4, 1, 1, 1, 1), torch.nan, requires_grad=True)
    measure = pushforward_log_to_current_image(
        source, torch.zeros_like(source, dtype=torch.bool), spatial, rows=3, columns=4
    )
    prob, _ = measure.normalized((1,))
    objective = prob.sum() + measure.camera_centers().sum()
    objective.backward()
    assert not measure.supported.any() and torch.count_nonzero(prob) == 0
    for tensor in (xy, logs, source):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_one_globally_suppressed_camera_does_not_become_missing_observation():
    # An adversarial but finite null logit tests semantics, not task performance.
    # This does not claim such weights were learned by a real checkpoint.
    torch.manual_seed(103)
    local = _local_at((0.3, -0.2))
    binder = DenseObjectGrounder(
        hidden=32, content_dim=16, route_dim=8, entity_chart_mode=CURRENT_IMAGE_CHART
    )
    from clearvla.mainline.model.grounding import dense_chart_from_local_facts

    tokens = binder._candidate_tokens(dense_chart_from_local_facts(local))
    with torch.no_grad():
        binder.null_key.copy_(tokens[:, 0].mean((1, 2, 3))[:, None] * 10000)
    facts, _ = binder(local, collect_diagnostics=False)
    assert torch.isfinite(facts.camera_transport_prior).all()
    # Source support stays legal even when selected global camera mass underflows.
    torch.testing.assert_close(facts.camera_validity, torch.ones_like(facts.camera_validity))
    assert facts.current_image_measure is not None
    torch.testing.assert_close(
        facts.camera_coordinates, facts.current_image_measure.camera_centers()
    )
    (facts.reconstruction_error + facts.camera_coordinates.square().sum()).backward()
    assert binder.null_key.grad is not None and torch.isfinite(binder.null_key.grad).all()
