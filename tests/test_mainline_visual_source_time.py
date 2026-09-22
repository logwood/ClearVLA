"""Source-timed vision tests exercise the actual raw-flow/G/S/W/Teacher path."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_state_features import _config as state_config
from test_mainline_state_features import _data, _model_engine, _training

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
from clearvla.vision.source_time import (
    SOURCE_VISUAL_TIME,
    VisualSourceTime,
    displacement_rate,
    pair_mean,
    visual_time_metadata,
)


def _config() -> ExperimentConfig:
    cfg = state_config()
    return replace(cfg, observation=replace(cfg.observation, source_time_mode=SOURCE_VISUAL_TIME))


@pytest.mark.parametrize("offsets,frames,pairs,gaps", [
    ([0, 0, 0], [False, False, True], [False, False], [0, 0]),
    ([-1, -1, 0], [False, True, True], [False, True], [0, 1]),
    ([-4, -4, 0], [False, True, True], [False, True], [0, 4]),
    ([-5, -4, 0], [True, True, True], [True, True], [1, 4]),
    ([-8, -4, 0], [True, True, True], [True, True], [4, 4]),
])
def test_source_clock_counts_measurements_not_storage_rows(offsets, frames, pairs, gaps):
    time = VisualSourceTime(torch.tensor([offsets], dtype=torch.long))
    time.validate(batch=1, frames=3, device=torch.device("cpu"), strict=True)
    assert time.frame_observed.tolist() == [frames]
    assert time.pair_observed.tolist() == [pairs]
    assert time.pair_steps.tolist() == [gaps]
    assert time.memory_observed(cameras=2, grid=8).sum() == (sum(frames) + sum(pairs)) * 128


@pytest.mark.parametrize("offsets", [torch.zeros(1, 3), torch.tensor([[0, -1, 0]]),
                                    torch.tensor([[-8, -4, 1]]), torch.zeros(3, dtype=torch.long)])
def test_malformed_source_time_fails_before_visual_encoding(offsets):
    with pytest.raises(ValueError):
        VisualSourceTime(offsets).validate(batch=1, frames=3, device=torch.device("cpu"), strict=True)


def test_duplicate_alias_has_no_value_or_gradient_authority_including_dropout_mask():
    time = VisualSourceTime(torch.tensor([[-1, -1, 0]]))
    x = torch.tensor([[[float("nan")], [3.], [4.]]], requires_grad=True)
    y = time.canonicalize(x)
    torch.testing.assert_close(y, torch.tensor([[[3.], [3.], [4.]]]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[0.], [2.], [1.]]]))
    mask = torch.tensor([[[False], [True], [False]]])
    assert time.canonicalize(mask).tolist() == [[[True], [True], [False]]]


def test_pair_mean_and_motion_rate_keep_source_support_and_units():
    x = torch.tensor([[float("nan"), 7.], [3., 5.]], requires_grad=True)
    observed = torch.tensor([False, True])
    loss = pair_mean(x, observed)
    assert loss.item() == 4.
    loss.backward()
    torch.testing.assert_close(x.grad, torch.tensor([[0., 0.], [.5, .5]]))
    assert pair_mean(x, torch.zeros_like(observed)).item() == 0.
    rate = displacement_rate(torch.tensor([[4., 8.], [1., 2.], [float("nan"), 1.]]),
                             torch.tensor([4, 1, 0]))
    torch.testing.assert_close(rate, torch.tensor([[1., 2.], [1., 2.], [0., 0.]]))


def test_explicit_config_and_metadata_do_not_relabel_old_checkpoints():
    old, cfg = state_config(), _config()
    assert "source_time_mode" not in cast(dict[str, object], old.as_dict()["observation"])
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert load_config("configs/mainline/structural_rebuild_m1d_calvin.json").observation.source_time_mode == SOURCE_VISUAL_TIME
    assert visual_time_metadata(SOURCE_VISUAL_TIME)["seconds_per_step"] is None
    with pytest.raises(ValueError):
        replace(cfg, observation=replace(cfg.observation, source_time_mode="guess_fps")).validate()
    with pytest.raises(ValueError):
        replace(cfg, top=replace(cfg.top, history_encoding_mode="paired_rows_v1")).validate()


def test_real_encoder_requires_its_selected_source_clock():
    data, cfg = _data(), _config()
    model, _ = _model_engine(cfg)
    batch = _training(data, (0,))
    with pytest.raises(ValueError, match="missing actual"):
        model.observation.prepare(batch.online.observation)
    old, _ = _model_engine(state_config())
    with pytest.raises(ValueError, match="silently ignore"):
        old.observation.prepare(batch.online.observation, source_offsets=torch.zeros(1, 3, dtype=torch.long))


@pytest.mark.parametrize("center,steps,memory_rows", [(0, 0, 128), (1, 1, 384), (4, 4, 384), (8, 4, 640)])
def test_production_encode_preserves_time_across_grounding_world_and_teacher(center, steps, memory_rows):
    data, cfg = _data(), _config()
    model, _ = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    model.eval()
    batch = _training(data, (center,))
    with torch.no_grad():
        cache, training, _ = model.encode_online(batch.online)
        bank = training.observation.grounding
        assert bank.latest_flow_steps is not None and bank.visual_memory_observed is not None
        assert bank.latest_flow_steps.tolist() == [steps]
        assert bank.visual_memory_observed.sum() == memory_rows
        assert training.top.facts.latest_flow_steps is not None
        assert cache.top.belief.latest_flow_steps is not None
        assert training.top.facts.latest_flow_steps.tolist() == [steps]
        assert cache.top.belief.latest_flow_steps.tolist() == [steps]
        permutation = torch.arange(training.top.facts.objects - 1, -1, -1)
        permuted_facts = training.top.facts.permute(permutation)
        permuted_world = cache.top.belief.permute(permutation)
        assert permuted_facts.latest_flow_steps is not None and permuted_world.latest_flow_steps is not None
        assert permuted_facts.latest_flow_steps.tolist() == [steps]
        assert permuted_world.latest_flow_steps.tolist() == [steps]
        targets, _ = model.build_training_targets(training, batch.future)
        assert targets.teacher_dynamics is not None
        assert torch.isfinite(targets.teacher_dynamics.semantic_delta).all()
        if center == 0:
            assert bank.native_flow_losses is not None
            assert all(float(v) == 0. for v in bank.native_flow_losses.values())
            assert torch.count_nonzero(training.top.facts.camera_transport_rate) == 0


def test_real_flow_objectives_exclude_duplicate_pairs_in_mixed_batches():
    cfg = _config()
    model, _ = _model_engine(cfg)
    encoder = model.observation.compiler.encoder
    encoder.eval()
    torch.manual_seed(912)
    frames = torch.randn(2, 3, 2, 8, 8, 16)
    support = torch.tensor([[False, False], [True, True]])
    out = encoder._estimate_pairs(frames, support)
    one = encoder._estimate_pairs(frames[1:], support[1:])
    for name, value in out[-1].items():
        torch.testing.assert_close(value, one[-1][name], atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(out[0].flow[:4]) == 0
    raw = torch.rand(2, 3, 2, 3, 32, 32)
    assert encoder.raw_flow is not None
    full = encoder.raw_flow(raw, out[0].flow, out[1].flow,
                            encoder._semantic_seed_reliability(out[0]),
                            encoder._semantic_seed_reliability(out[1]), pair_observed=support)
    actual = encoder.raw_flow(raw[1:], one[0].flow, one[1].flow,
                              encoder._semantic_seed_reliability(one[0]),
                              encoder._semantic_seed_reliability(one[1]), pair_observed=support[1:])
    for name, value in full[1].items():
        torch.testing.assert_close(value, actual[1][name], atol=2e-6, rtol=2e-6)
    for name in ("flow_forward", "flow_backward", "confidence", "observable_motion"):
        assert torch.count_nonzero(getattr(full[0], name)[0]) == 0


def test_production_action_is_invariant_to_reset_alias_payload_and_clock_is_read_only():
    cfg, data = _config(), _data()
    model, _ = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    model.eval()
    batch = _training(data, (1,))
    image, tokens = batch.online.observation.raw_rgb.clone(), batch.online.observation.dino_history.clone()
    image[:, 0], tokens[:, 0] = float("nan"), float("nan")
    changed = replace(batch.online, observation=replace(batch.online.observation, raw_rgb=image, dino_history=tokens))
    timing = batch.online.history.timing
    assert timing is not None
    before = {k: v.clone() for k, v in timing.as_mapping().items()}
    with torch.no_grad():
        first = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(31))
        second = sample_action(model, changed, cfg, generator=torch.Generator().manual_seed(31))
    torch.testing.assert_close(first.action, second.action, atol=0, rtol=0)
    for key, value in before.items():
        torch.testing.assert_close(value, timing.as_mapping()[key], atol=0, rtol=0)
    assert first.action.shape == (1, 24, 7)


def test_real_model_training_reaches_source_time_projection_and_masks_reset_flow_gradients():
    cfg, data = _config(), _data()
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(_training(data, (0, 1, 8, 79)))
    assert engine.global_step == 1
    projection = model.observation.compiler.encoder.source_time_projection
    assert projection is not None and projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert not model.observation.compiler.encoder.history_type.requires_grad
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_teacher_extrapolates_by_observed_duration_not_nominal_slot_spacing():
    from clearvla.mainline.model.teacher import ObjectFutureTeacher
    # Use the actual factory-owned Teacher, not a substitute mathematical model.
    model, _ = _model_engine(_config())
    teachers = [m for m in model.modules() if isinstance(m, ObjectFutureTeacher)]
    assert len(teachers) == 1
    teacher = teachers[0]
    offsets = torch.tensor([[4, 8, 48], [4, 8, 48], [4, 8, 48]])
    result = teacher._flow_horizon_scale(offsets, torch.tensor([1, 4, 0]))
    torch.testing.assert_close(result, torch.tensor([[4., 8., 48.], [1., 2., 12.], [0., 0., 0.]]))


@pytest.mark.parametrize("sequential", [False, True])
def test_future_queries_cannot_read_missing_flow_pairs(sequential: bool):
    model, _ = _model_engine(_config())
    encoder = model.observation.compiler.encoder
    assert encoder.sequential_horizon_memory
    encoder.sequential_horizon_memory = sequential
    shape = (1, len(encoder.window_offsets), encoder.cameras,
             encoder.grid_size, encoder.grid_size, encoder.hidden)
    identity = torch.randn(shape)
    context = torch.randn(1, 1, *shape[2:])
    x = torch.full((1, 2, *shape[2:-1], encoder.MOTION_DIM), float("nan"), requires_grad=True)
    source = VisualSourceTime(torch.zeros(1, 3, dtype=torch.long))
    query, _ = encoder._compose_future_queries(identity, x, context, source)
    control, _ = encoder._compose_future_queries(identity, torch.zeros_like(x), context, source)
    torch.testing.assert_close(query, control, atol=0, rtol=0)
    query.square().mean().backward()
    assert x.grad is not None and torch.count_nonzero(x.grad) == 0
    for p in encoder.future_motion.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    assert torch.isfinite(query).all()


def test_real_grounding_blocks_quarantine_invalid_memory_before_normalization():
    model, _ = _model_engine(_config())
    model.eval()  # Hold dropout fixed while changing only unsupported memory payloads.
    batch = _training(_data(), (1,))
    offsets = batch.online.history.timing
    assert offsets is not None
    prepared = model.observation.prepare(batch.online.observation, source_offsets=offsets.state_offsets)
    role = model.bridge.sample_role_context(prepared.pack.value_tokens)
    canvas, slices = model.bridge.build_grounding_seed(state=batch.online.history.state,
        rollout_init=prepared.pack.future_queries, role=role)
    bank, _ = model.observation.build_grounding_bank(prepared, canvas, slices)
    support = bank.visual_memory_observed
    assert support is not None and (~support).any()
    modulation = torch.zeros(canvas.shape[0], model.config.dimensions.hidden_size)
    bad_keys = torch.where(support[..., None], bank.visual_memory, torch.full_like(bank.visual_memory, float("nan")))
    bad_values = torch.where(support[..., None], bank.visual_value_memory, torch.full_like(bank.visual_value_memory, float("nan")))
    block = model.grounding.blocks[0]
    clean, _ = block(canvas, bank.visual_memory, modulation, slices,
        visual_value_memory=bank.visual_value_memory, visual_key_padding_mask=~support)
    changed, _ = block(canvas, bad_keys, modulation, slices,
        visual_value_memory=bad_values, visual_key_padding_mask=~support)
    torch.testing.assert_close(clean, changed, atol=0, rtol=0)
    assert torch.isfinite(changed).all()


def test_source_timed_encoder_cpu_bfloat16_mixed_support_backward():
    model, _ = _model_engine(_config())
    batch = _training(_data(), (0, 1, 8))
    timing = batch.online.history.timing
    assert timing is not None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prepared = model.observation.prepare(batch.online.observation, source_offsets=timing.state_offsets)
        pack = prepared.pack
        assert pack.memory_observed is not None
        values = torch.where(pack.memory_observed[..., None], pack.value_tokens, torch.zeros_like(pack.value_tokens))
        loss = pack.future_queries.float().square().mean() + values.float().square().mean() + sum(pack.losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    encoder = model.observation.compiler.encoder
    for module in (encoder.source_time_projection, encoder.source_organizer_time_projection):
        assert module is not None and module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_source_timed_checkpoint_and_deployment_roundtrip_with_strict_metadata(tmp_path: Path):
    torch.manual_seed(631)
    data, cfg = _data(), _config()
    batch = _training(data, (1,))
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    model.eval()
    language = tmp_path / "language.pt"
    torch.save({"tokens": batch.online.goal.tokens, "mask": batch.online.goal.mask}, language)
    dataset = replace(identity_fixture(),
        action_normalizer_sha256=canonical_sha256(data.action_normalizer.to_dict()),
        state_normalizer_sha256=canonical_sha256(data.state_normalizer.to_dict()))
    identity = build_checkpoint_identity(cfg, repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset, language=ArtifactIdentity.from_file("t5_goal", language), commit="4"*40)
    assert "clearvla/vision/source_time.py" in dict(identity.source.files)
    profile = resolve_action_state_profile("calvin_relative_7d_v1")
    abi = build_deployment_abi(cfg, identity, action_normalizer=data.action_normalizer,
        state_normalizer=data.state_normalizer,
        data_profile={**profile.as_dict(), "gripper_transition_boundary":profile.gripper_transition_boundary},
        gripper_indices=(6,), goal_metadata={})
    validate_deployment_abi(abi)
    for mutation in ("missing", "seconds", "bool"):
        broken = copy.deepcopy(abi)
        observation = cast(dict[str, object], broken["observation"])
        if mutation == "missing":
            del observation["visual_source_time"]
        else:
            time = cast(dict[str, object], observation["visual_source_time"])
            time["seconds_per_step"] = 1/60 if mutation == "seconds" else False
        with pytest.raises(ValueError):
            validate_deployment_abi(broken)
    path = tmp_path / "source-timed.pt"
    save_checkpoint(path, model=model, optimizer=engine.optimizer, schedule=engine.schedule,
        config=cfg, identity=identity, epoch=0, global_step=engine.global_step, best_metric=None,
        data_state={"action_normalizer":data.action_normalizer.to_dict(),
                    "state_normalizer":data.state_normalizer.to_dict(), "deployment_abi":abi})
    restored, restored_engine = _model_engine(cfg)
    load_checkpoint_exact(path, model=restored, optimizer=restored_engine.optimizer,
        schedule=restored_engine.schedule, config=cfg, identity=identity)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_checkpoint_exact(path, model=restored, optimizer=restored_engine.optimizer,
            schedule=restored_engine.schedule, config=state_config(), identity=identity)
    bundle = load_deployment_checkpoint(path, device=torch.device("cpu"))
    assert bundle.config.observation.source_time_mode == SOURCE_VISUAL_TIME
    assert bundle.state_normalizer.scale.shape == (1, 7)
    with torch.no_grad():
        first = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(92))
        second = sample_action(bundle.model, batch.online, bundle.config,
                               generator=torch.Generator().manual_seed(92))
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    assert first.action.shape == (1, 24, 7) and bundle.global_step == 1
