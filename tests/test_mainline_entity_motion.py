"""Actual G3 image ownership must also own its exported motion estimate."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_entity_history import _batch, _history
from test_mainline_entity_history import _config as _history_config
from test_mainline_state_features import _data, _model_engine

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.vision.entity_chart import ImageLogMeasure, current_image_grid
from clearvla.vision.entity_motion import (
    CURRENT_ENTITY_MOTION,
    QUERY_ANCHOR_MOTION,
    current_entity_motion,
    entity_motion_metadata,
)
from clearvla.vision.source_time import VisualSourceTime


def _config() -> ExperimentConfig:
    cfg = _history_config()
    cfg = replace(cfg, top=replace(cfg.top, entity_motion_mode=CURRENT_ENTITY_MOTION))
    cfg.validate()
    return cfg


def _measure() -> ImageLogMeasure:
    support = torch.zeros(1, 2, 2, 5, 5, dtype=torch.bool)
    support[0, 0, :, 1, 1] = True
    support[0, 1, :, 3, 3] = True
    return ImageLogMeasure(torch.zeros(support.shape), support)


def test_affine_motion_is_read_at_actual_entity_location_not_query_anchor():
    h = _history(side=5)
    grid = current_image_grid(5, 5, device=h.content.device)
    flow = h.backward_flow.clone()
    flow[:, -1, :, 0] = -(0.1 + 0.2 * grid[..., 0])
    flow[:, -1, :, 1] = 0.3 * grid[..., 1]
    h = replace(h, backward_flow=flow)
    measure = _measure()
    value = current_entity_motion(measure, h)
    centers = measure.camera_centers()
    expected = torch.stack((0.1 + 0.2 * centers[..., 0], -0.3 * centers[..., 1]), -1)
    torch.testing.assert_close(value, expected)
    assert not torch.equal(value[:, 0], value[:, 1])
    # Source time belongs to rate conversion, not spatial lookup.
    changed_time = replace(h, source_time=VisualSourceTime(torch.tensor([[-9, -1, 0]])))
    torch.testing.assert_close(current_entity_motion(measure, changed_time), value, rtol=0, atol=0)


def test_tiny_camera_mass_is_conditioned_in_log_space_and_views_stay_separate():
    h = _history(side=5)
    flow = h.backward_flow.clone()
    flow[:, -1, 0, 0] = -0.3
    flow[:, -1, 1, 0] = 0.2
    h = replace(h, backward_flow=flow)
    m = _measure()
    logs = m.log_mass.clone()
    logs[:, :, 1] -= 1000
    m = replace(m, log_mass=logs)
    out = current_entity_motion(m, h)
    torch.testing.assert_close(out[0, :, :, 0], torch.tensor([[0.3, -0.2], [0.3, -0.2]]))
    hp = replace(
        h,
        content=h.content.flip(2),
        observed=h.observed.flip(2),
        backward_flow=flow.flip(2),
        confidence=h.confidence.flip(2),
        occlusion=h.occlusion.flip(2),
    )
    mp = ImageLogMeasure(m.log_mass.flip(2), m.supported.flip(2))
    torch.testing.assert_close(current_entity_motion(mp, hp), out.flip(2), rtol=0, atol=0)
    kp = ImageLogMeasure(m.log_mass.flip(1), m.supported.flip(1))
    torch.testing.assert_close(current_entity_motion(kp, h), out.flip(1), rtol=0, atol=0)


def test_confidence_and_correspondence_coverage_do_not_rewrite_motion_law():
    h = _history(side=5)
    # A large estimate can point outside: it is still an estimate, not an
    # admitted historical correspondence or a reason to mask the present.
    h = replace(h, backward_flow=h.backward_flow + 3)
    m = _measure()
    a = current_entity_motion(m, h)
    b = current_entity_motion(m, replace(h, confidence=h.confidence * 0, occlusion=h.occlusion + 1))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, torch.full_like(a, -3), rtol=0, atol=0)


def test_missing_pair_and_empty_entity_are_not_nan_or_fake_uniform_motion():
    h = _history((0, 0, 0), side=5)
    h = replace(h, backward_flow=h.backward_flow * torch.nan)
    out = current_entity_motion(_measure(), h)
    assert torch.isfinite(out).all() and torch.count_nonzero(out) == 0
    m = _measure()
    out = current_entity_motion(
        replace(m, supported=torch.zeros_like(m.supported)), _history(side=5)
    )
    assert torch.count_nonzero(out) == 0


def test_nonuniform_spatial_law_and_flow_both_receive_gradients():
    h = _history(side=5)
    grid = current_image_grid(5, 5, device=h.content.device)
    flow = h.backward_flow.clone()
    flow[:, -1, :, 0] = grid[..., 0]
    flow.requires_grad_()
    logs = torch.randn(1, 2, 2, 5, 5, requires_grad=True)
    m = ImageLogMeasure(logs, torch.ones_like(logs, dtype=torch.bool))
    out = current_entity_motion(m, replace(h, backward_flow=flow))
    out.square().sum().backward()
    for t in (flow, logs):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert flow.grad is not None and torch.count_nonzero(flow.grad[:, 0]) == 0


def test_motion_chart_configuration_has_no_implicit_fallback():
    cfg = _config()
    assert config_from_mapping(cfg.as_dict()) == cfg
    old = _history_config()
    assert old.top.entity_motion_mode == QUERY_ANCHOR_MOTION
    assert (
        entity_motion_metadata(CURRENT_ENTITY_MOTION)["spatial_read"]
        != entity_motion_metadata(QUERY_ANCHOR_MOTION)["spatial_read"]
    )
    assert "entity_motion_mode" not in cast(dict[str, object], old.as_dict()["top"])
    for top in (
        replace(cfg.top, entity_motion_mode="unknown"),
        replace(cfg.top, entity_history_mode="current_only_v1"),
    ):
        with pytest.raises(ValueError):
            replace(cfg, top=top).validate()


@pytest.mark.parametrize("bf16,center", [(False, 0), (False, 8), (True, 8)])
def test_production_facts_world_and_s_consume_the_same_current_motion(bf16: bool, center: int):
    model, engine = _model_engine(_config())
    batch = _batch(center)
    w_inputs: list[torch.Tensor] = []
    s_inputs: list[torch.Tensor] = []

    def record_w(_m: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        w_inputs.append(args[0])

    def record_s(_m: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        s_inputs.append(args[0])

    wh = model.world.dynamics.object_transport_prior.register_forward_pre_hook(record_w)
    sh = model.intent.organizer.state_change_transport.register_forward_pre_hook(record_s)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, train, _ = model.encode_online(batch.online)
        facts = train.top.facts
        h = train.observation.local_facts.observed_history
        assert facts.current_image_measure is not None and h is not None
        expected = current_entity_motion(facts.current_image_measure, h).to(
            facts.camera_transport_prior.dtype
        )
        torch.testing.assert_close(facts.camera_transport_prior, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            cache.top.belief.camera_transport_prior, expected, rtol=0, atol=0
        )
        assert w_inputs and s_inputs
        torch.testing.assert_close(
            w_inputs[0], facts.transport_rate.to(w_inputs[0].dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(
            s_inputs[0], facts.transport_rate.to(s_inputs[0].dtype), rtol=0, atol=0
        )
    wh.remove()
    sh.remove()
    engine.train_step(batch)
    assert engine.global_step == 1
    if center == 0:
        assert facts.latest_flow_steps is not None
        assert torch.count_nonzero(expected) == 0 and (facts.latest_flow_steps == 0).all()


def test_checkpoint_identity_reload_and_deployment(tmp_path: Path):
    cfg = _config()
    data = _data()
    batch = _batch()
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
        commit="8" * 40,
    )
    assert "clearvla/vision/entity_motion.py" in dict(identity.source.files)
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
    for kind in ["missing", "wrong"]:
        bad = copy.deepcopy(abi)
        obs = cast(dict[str, object], bad["observation"])
        if kind == "missing":
            del obs["entity_motion"]
        else:
            cast(dict[str, object], obs["entity_motion"])["spatial_read"] = "stale_query_anchor"
        with pytest.raises(ValueError, match="entity_motion|entity motion"):
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
            config=_history_config(),
            identity=identity,
        )
    deployed = load_deployment_checkpoint(path, device=torch.device("cpu"))
    with torch.no_grad():
        a = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(1031))
        b = sample_action(
            deployed.model,
            batch.online,
            deployed.config,
            generator=torch.Generator().manual_seed(1031),
        )
    torch.testing.assert_close(a.action, b.action, rtol=0, atol=0)


@torch.no_grad()
def test_real_binder_ignores_stale_anchor_motion_in_new_mode_only():
    model, _ = _model_engine(_config())
    model.eval()
    _, train, _ = model.encode_online(_batch().online)
    local = train.observation.local_facts
    assert local.slot_transport_prior is not None
    perturbed = replace(local, slot_transport_prior=local.slot_transport_prior + 500.0)
    a, _ = model.grounding.grounder(local, collect_diagnostics=False)
    b, _ = model.grounding.grounder(perturbed, collect_diagnostics=False)
    for field in (
        "camera_transport_prior",
        "camera_coordinates",
        "candidate_assignment",
        "camera_validity",
    ):
        torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    # The old mode remains an explicit, observable control, not a fallback.
    control, _ = _model_engine(_history_config())
    x, _ = control.grounding.grounder(local, collect_diagnostics=False)
    y, _ = control.grounding.grounder(perturbed, collect_diagnostics=False)
    assert not torch.equal(x.camera_transport_prior, y.camera_transport_prior)


@pytest.mark.parametrize("bf16", [False, True])
def test_action_only_gradient_reaches_exported_current_motion(bf16):
    model, _ = _model_engine(_config())
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        cache, train, _ = model.encode_online(_batch().online)
        motion = train.top.facts.camera_transport_prior
        motion.retain_grad()
        out = model.velocity(
            cache, noisy_action_field=torch.randn(1, 24, 18), time=torch.full((1,), 0.4)
        )
        loss = out.bottom.physical_velocity.float().square().mean()
    loss.backward()
    assert motion.grad is not None
    assert torch.isfinite(motion.grad).all() and motion.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["nonfinite", "grid", "rank"])
def test_mismatched_or_nonfinite_current_law_is_rejected(kind):
    m, h = _measure(), _history(side=5)
    if kind == "nonfinite":
        m = replace(m, log_mass=torch.full_like(m.log_mass, torch.nan))
    elif kind == "grid":
        m = ImageLogMeasure(m.log_mass[..., :-1], m.supported[..., :-1])
    else:
        m = ImageLogMeasure(m.log_mass[:, 0], m.supported[:, 0])
    with pytest.raises(ValueError):
        current_entity_motion(m, h)


def test_parameter_inventory_is_unchanged_by_motion_read_selection():
    torch.manual_seed(2231)
    old, _ = _model_engine(_history_config())
    torch.manual_seed(2231)
    new, _ = _model_engine(_config())
    assert list(old.state_dict()) == list(new.state_dict())
    for name, value in new.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)
    assert [(n, p.requires_grad) for n, p in old.named_parameters()] == [
        (n, p.requires_grad) for n, p in new.named_parameters()
    ]
