"""Observed-tail source contracts, exercising production classes (no toy model)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_timed_history import _ImageStore, _timed_batch, _timed_config

from clearvla.data.future_clock import future_source_rows
from clearvla.data.hdf5_episode import RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING, LoadedEpisode
from clearvla.data.window_boundaries import OBSERVED_TAIL_V1
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.data.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.runtime.evaluation import ValidationAccumulator
from clearvla.mainline.supervision import FutureLabelSupport, quarantine, supported_mean
from clearvla.mainline.training.engine import MainlineTrainingEngine, validate_finite_training_batch
from clearvla.mainline.training.losses import sample_flow_matching
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer


def _config() -> ExperimentConfig:
    base = _timed_config(binary=True)
    return replace(base, data=replace(base.data, window_boundary_contract=OBSERVED_TAIL_V1))


def _support(length: int, batch: int = 1) -> FutureLabelSupport:
    action = (torch.arange(48)[None] < length).expand(batch, -1)
    visual = (torch.arange(4, 49, 4)[None] <= length).expand(batch, -1)
    return FutureLabelSupport(action, action.clone(), visual)


def _batch(length: int, invalid: float = 0.0) -> TrainingBatch:
    b = _timed_batch(_config())
    support = _support(length)
    source = b.future.action_sequence.clone()
    source[..., -1] = source[..., -1].sign()
    policy = source[:, :24].clone()

    def fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(
            mask.reshape(*mask.shape, *((1,) * (x.ndim - mask.ndim))),
            x,
            torch.full_like(x, invalid),
        )

    return replace(
        b,
        action_target=replace(
            b.action_target,
            normalized=fill(policy, support.action[:, :24]),
            raw_units=fill(policy, support.action[:, :24]),
            support=support,
        ),
        future=replace(
            b.future,
            action_sequence=fill(source, support.action),
            state_sequence=fill(b.future.state_sequence, support.state),
            dino_supports=fill(b.future.dino_supports, support.visual),
            support=support,
        ),
    )


def _normalizer() -> ArrayNormalizer:
    return ArrayNormalizer.fit_zscore([np.array([[-1.0] * 7, [1.0] * 7], dtype=np.float32)])


def _model_engine(config: ExperimentConfig):
    model = ClearVLAMainlinePolicy(config)
    model.configure_action_normalizer(_normalizer())
    optimizer, _ = build_optimizer(model, config)
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1),
        device=torch.device("cpu"),
    )
    return model, engine


@pytest.mark.parametrize("remaining", [1, 2, 3, 4, 7, 8, 9, 16, 23, 24, 25, 47, 48, 60])
def test_source_clock_counts_real_commands_successors_and_sparse_views(remaining: int):
    clock = future_source_rows(20, 20 + remaining)
    assert int(clock.action_observed.sum()) == min(remaining, 48)
    np.testing.assert_array_equal(clock.action_observed, clock.state_observed)
    assert int(clock.visual_observed.sum()) == min(remaining // 4, 12)
    assert clock.action_indices.max() < 20 + remaining
    assert clock.state_indices.max() <= 20 + remaining
    assert clock.visual_indices.max() <= 20 + remaining
    if remaining == 1:
        assert clock.action_indices[0] == 20 and clock.state_indices[0] == 21
        assert not clock.visual_observed.any()


@pytest.mark.parametrize(
    "center,terminal", [(3, 3), (-1, 3), (4, 3), (True, 3), (0, False), (1.5, 3)]
)
def test_clock_rejects_non_real_current_actions(center: Any, terminal: Any):
    with pytest.raises((ValueError, TypeError)):
        future_source_rows(center, terminal)


def test_support_requires_real_prefix_and_consistent_observation_clock():
    offsets = torch.arange(4, 49, 4)[None]
    s = _support(9)
    s.validate(batch=1, horizon=48, offsets=offsets, device=torch.device("cpu"), strict=True)
    for bad in (
        replace(s, action=torch.zeros_like(s.action)),
        replace(s, visual=torch.ones_like(s.visual)),
        replace(s, state=torch.ones_like(s.state)),
    ):
        with pytest.raises(ValueError):
            bad.validate(
                batch=1, horizon=48, offsets=offsets, device=torch.device("cpu"), strict=True
            )
    assert s.interval_observed(offsets, ((4, 8), (8, 16), (16, 32), (32, 48))).tolist() == [
        [True, False, False, False]
    ]


def _dataset(terminal: int = 80, start: int = 24) -> ObservedStateWindowDataset:
    length = terminal + 25
    state = np.repeat(np.arange(length, dtype=np.float32)[:, None], 7, axis=1)
    action = state.copy()
    action[:, -1] = 1
    # The existing file has fake absorbing storage. A corrupt filler must NOT
    # become a training label or even be read into the source future arrays.
    state[terminal + 1 :] = 99999
    action[terminal:] = 99999
    previous = np.concatenate((np.zeros((1, 7), np.float32), action[:-1]))
    e = LoadedEpisode(
        path=Path("tail.hdf5"),
        episode_id="tail",
        source_partition="",
        task_id="",
        action_key="action",
        camera_keys={},
        actions_raw=action,
        states_raw=state,
        action_states_raw=previous,
        valid_center_start=start,
        valid_center_end=max(start, terminal - 24),
        terminal_state_index=terminal,
        terminal_padding_mode=RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING,
    )
    norm = ArrayNormalizer.fit_identity([state[:terminal]])
    return ObservedStateWindowDataset(
        [e],
        [0],
        image_store=cast(Any, _ImageStore()),
        camera_names=("top", "wrist"),
        state_normalizer=norm,
        action_normalizer=norm,
        config=ObservedStateDatasetConfig(
            window_boundary_contract=OBSERVED_TAIL_V1,
            causal_reset_padding=True,
            emit_history_timing=True,
        ),
    )


@pytest.mark.parametrize("terminal,start", [(80, 24), (29, 24), (8, 0)])
def test_dataset_admits_every_real_current_state_without_fabricating_future(
    terminal: int, start: int
):
    data = _dataset(terminal, start)
    assert [ref.center for ref in data.refs] == list(range(start, terminal))
    last = data[len(data) - 1]
    assert last["policy_action_raw"][0, 0] == terminal - 1
    assert last["future_state"][0, 0] == terminal
    assert last["future_action_observed"].sum() == 1
    assert last["future_state_observed"].sum() == 1
    assert last["future_visual_observed"].sum() == 0
    assert torch.count_nonzero(last["policy_action_raw"][1:]) == 0
    assert torch.count_nonzero(last["future_state"][1:]) == 0
    assert last["future_keys"][:, 1].max() == terminal
    summary = data.boundary_summary()
    assert summary["windows"] == terminal - start
    assert summary["source_current_action_rows"] == terminal - start
    assert summary["supervision_support"] == "real_source_only"


def test_masked_mean_ignores_nan_labels_and_preserves_zero_gradients():
    x = torch.tensor([[[2.0], [float("nan")]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    loss = supported_mean(quarantine(x, mask).square(), mask)
    assert loss == 4
    loss.backward()
    assert x.grad is not None and x.grad.tolist() == [[[4.0], [0.0]]]
    zero = supported_mean(quarantine(x, torch.zeros_like(mask)).square(), torch.zeros_like(mask))
    assert zero == 0 and zero.requires_grad


def test_new_config_is_explicit_and_legacy_has_no_silent_label_support():
    cfg = load_config("configs/mainline/structural_rebuild_m1b_calvin.json")
    assert cfg.data.window_boundary_contract == OBSERVED_TAIL_V1
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert (
        load_config(
            "configs/mainline/structural_rebuild_m1_calvin.json"
        ).data.window_boundary_contract
        != OBSERVED_TAIL_V1
    )
    with pytest.raises(ValueError, match="source support"):
        _timed_batch(_config()).validate(_config())
    b = _batch(9)
    with pytest.raises(ValueError, match="share one"):
        replace(b, action_target=replace(b.action_target, support=_support(9))).validate(_config())


def test_flow_bridge_unknown_rows_are_noise_not_fake_terminal_targets():
    torch.manual_seed(7)
    model, _ = _model_engine(_config())
    b = _batch(3, invalid=float("nan"))
    fm = sample_flow_matching(
        b.action_target.normalized,
        action_state=b.online.history.action_state,
        codec_gripper_boundary=b.online.history.codec_gripper_boundary,
        codec=model.outlet_adapter.codec,
        distribution=_config().bottom.flow_time_distribution,
        row_valid=b.action_target.row_valid,
        generator=torch.Generator().manual_seed(6),
    )
    torch.testing.assert_close(
        fm.noisy_physical[:, 3:], fm.source_physical_noise[:, 3:], rtol=0, atol=0
    )
    assert not torch.count_nonzero(fm.target_physical_velocity[:, 3:])
    assert torch.isfinite(fm.noisy_physical).all()


@pytest.mark.parametrize("remaining", [1, 9, 24, 48])
def test_production_training_forward_backward_masks_all_target_consumers(remaining: int):
    torch.manual_seed(32)
    cfg = _config()
    model, engine = _model_engine(cfg)
    b = _batch(remaining, invalid=float("nan"))
    validate_finite_training_batch(b)
    result = engine.train_step(b, collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert result.metrics["loss_action_observed_label_rows"] == min(remaining, 24)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if remaining < 4:
        assert result.metrics["loss_future_dynamics"] == 0
        assert result.metrics["loss_future_transition"] == 0
        assert result.metrics["loss_intent_online"] == 0
    bad = replace(
        b, action_target=replace(b.action_target, normalized=b.action_target.normalized.clone())
    )
    bad.action_target.normalized[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_finite_training_batch(bad)


def test_validation_counts_only_real_labels_including_batch_selection():
    b = _batch(3, invalid=float("nan"))
    mask = b.action_target.row_valid
    assert mask is not None
    target = quarantine(b.action_target.normalized, mask)
    pred = target.clone()
    pred[:, 3:, :6] = 123456
    pred[..., -1] = 1
    acc = ValidationAccumulator.from_action_normalizer(
        _normalizer(),
        device=torch.device("cpu"),
        gripper_output_mode="calvin_binary_command",
        arm_flow_mode="relative_command_adapter",
    )
    acc.update(pred, b, row_indices=torch.tensor([0]))
    values = acc.means()
    assert values["validation_arm_rmse_physical"] == 0
    assert values["validation_observed_action_rows"] == 3
    assert values["validation_possible_action_rows"] == 24
    assert values["validation_normalized_tail_observed_elements"] == 0
    assert acc.scalar_totals["gripper_command_rows"] == 3
    assert acc.scalar_totals["motion_head_rows"] == 3


def _replace_missing_payload(b: TrainingBatch, filler: float) -> TrainingBatch:
    support = b.future.support
    assert support is not None

    def fill(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shape = (*mask.shape, *((1,) * (x.ndim - mask.ndim)))
        return torch.where(mask.reshape(shape), x, torch.full_like(x, filler))

    action_mask = support.action[:, :24]
    return replace(
        b,
        action_target=replace(
            b.action_target,
            normalized=fill(b.action_target.normalized, action_mask),
            raw_units=fill(b.action_target.raw_units, action_mask),
        ),
        future=replace(
            b.future,
            action_sequence=fill(b.future.action_sequence, support.action),
            state_sequence=fill(b.future.state_sequence, support.state),
            dino_supports=fill(b.future.dino_supports, support.visual),
        ),
    )


def test_unknown_label_payload_cannot_change_any_formal_loss_or_parameter_gradient():
    torch.manual_seed(203)
    cfg = _config()
    model, engine = _model_engine(cfg)
    model.eval()
    b = _batch(9)
    corrupted = _replace_missing_payload(b, float("nan"))
    first, _ = engine._forward(
        b, training=False, collect_diagnostics=True, generator=torch.Generator().manual_seed(71)
    )
    first.total.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    second, _ = engine._forward(
        corrupted,
        training=False,
        collect_diagnostics=True,
        generator=torch.Generator().manual_seed(71),
    )
    for name, value in first.contributions.items():
        torch.testing.assert_close(value, second.contributions[name], rtol=0, atol=0)
    second.total.backward()
    for name, p in model.named_parameters():
        if name in gradients:
            assert p.grad is not None
            torch.testing.assert_close(gradients[name], p.grad, rtol=0, atol=0)


def test_future_label_processing_does_not_mutate_online_sampling_or_state():
    from clearvla.mainline.runtime.sampling import sample_action

    torch.manual_seed(301)
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    b = _batch(9)
    with torch.no_grad():
        before = sample_action(model, b.online, cfg, generator=torch.Generator().manual_seed(7))
        _, train_state, _ = model.encode_online(b.online)
        model.build_training_targets(train_state, _replace_missing_payload(b, float("nan")).future)
        after = sample_action(model, b.online, cfg, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(before.action, after.action, rtol=0, atol=0)
    assert b.online.history.timing is not None
    assert b.online.history.timing.state_offsets.tolist() == [[-8, -4, 0]]


def test_recognizer_does_not_read_unavailable_interval_tokens():
    torch.manual_seed(55)
    cfg = _config()
    model, _ = _model_engine(cfg)
    model.eval()
    b = _batch(9)
    with torch.no_grad():
        _, state, _ = model.encode_online(b.online)
        first, _ = model.build_training_targets(state, b.future)
        identity = model.training_targets.recognizer.interval_identity
        assert isinstance(identity, torch.Tensor)
        identity[:, 1:] = 1e6
        second, _ = model.build_training_targets(state, b.future)
    assert first.plan_recognition is not None and second.plan_recognition is not None
    assert first.plan_recognition.interval_valid is not None
    assert first.plan_recognition.interval_valid.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(
        first.plan_recognition.interval_targets,
        second.plan_recognition.interval_targets,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(first.online_intent_loss, second.online_intent_loss, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_production_cache_loader_and_metric_subset_keep_source_masks(device: str):
    from torch.utils.data import default_collate

    from clearvla.mainline.data.dataset import CachedTokenPolicyWindowDataset
    from clearvla.mainline.data.loading import GoalTemplate, to_training_batch

    class TokenTransport:
        def load_batch(self, keys: torch.Tensor) -> torch.Tensor:
            assert int(keys[:, 1].max()) <= 80
            return keys[:, 1, None, None, None].expand(-1, 2, 64, 16).float()

    data = _dataset()
    cache = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, TokenTransport()))
    raw = default_collate([cache[len(cache) - 7], cache[len(cache) - 1]])
    goal = GoalTemplate(torch.zeros(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {})
    b = to_training_batch(raw, goal=goal, config=_config(), device=torch.device(device))
    assert b.future.support is not None and b.action_target.support is b.future.support
    assert b.future.support.action.sum(dim=1).tolist() == [7, 1]
    validate_finite_training_batch(b)
    pred = b.action_target.normalized.clone()
    pred[..., -1] = 1
    acc = ValidationAccumulator.from_action_normalizer(
        _normalizer(),
        device=torch.device("cpu"),
        gripper_output_mode="calvin_binary_command",
        arm_flow_mode="relative_command_adapter",
    )
    acc.update(pred, b, row_indices=torch.tensor([1]))
    assert acc.means()["validation_observed_action_rows"] == 1
    assert acc.samples == 1
    broken = dict(raw)
    del broken["future_visual_observed"]
    with pytest.raises(KeyError, match="future_visual_observed"):
        to_training_batch(broken, goal=goal, config=_config(), device=torch.device(device))


def test_real_checkpoint_roundtrip_and_no_silent_m1a_exact_resume(tmp_path: Path):
    from test_mainline_checkpoint import _dataset as identity_fixture

    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint

    cfg = _config()
    model, engine = _model_engine(cfg)
    language = tmp_path / "language.pt"
    language.write_bytes(b"provenance fixture; not encoder data")
    identity = build_checkpoint_identity(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=identity_fixture(),
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="2" * 40,
    )
    sources = dict(identity.source.files)
    assert "clearvla/data/future_clock.py" in sources
    assert "clearvla/mainline/supervision.py" in sources
    path = tmp_path / "exact.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=engine.optimizer,
        schedule=engine.schedule,
        config=cfg,
        identity=identity,
        epoch=0,
        global_step=0,
        best_metric=None,
    )
    restored, restored_engine = _model_engine(cfg)
    load_checkpoint_exact(
        path,
        model=restored,
        optimizer=restored_engine.optimizer,
        schedule=restored_engine.schedule,
        config=cfg,
        identity=identity,
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    old = replace(cfg, data=replace(cfg.data, window_boundary_contract="strict_complete_v1"))
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=restored,
            optimizer=restored_engine.optimizer,
            schedule=restored_engine.schedule,
            config=old,
            identity=identity,
        )


def test_masked_support_cannot_be_smuggled_into_legacy_training():
    with pytest.raises(ValueError, match="observed-tail contract"):
        _batch(9).validate(_timed_config(binary=True))


def test_preflight_checks_shared_real_action_truth_not_just_mask_shapes():
    b = _batch(9)
    future = b.future.action_sequence.clone()
    future[0, 0, 0] += 1
    with pytest.raises(ValueError, match="disagree on observed source actions"):
        validate_finite_training_batch(replace(b, future=replace(b.future, action_sequence=future)))


@pytest.mark.parametrize("remaining", [1, 9])
def test_continuous_outlet_uses_same_observed_support_without_fake_hold_labels(remaining: int):
    torch.manual_seed(102)
    base = _config()
    cfg = replace(
        base,
        data=replace(base.data, data_profile="libero_relative_7d_v1"),
        bottom=replace(base.bottom, gripper_output_mode="continuous"),
        objectives=replace(base.objectives, gripper_command=0.0),
    )
    cfg.validate()
    model, engine = _model_engine(cfg)
    b = _batch(remaining, invalid=float("nan"))
    result = engine.train_step(b, collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert result.metrics["loss_action_observed_label_rows"] == remaining
    assert result.metrics["loss_gripper_command"] == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampler_and_progress_do_not_read_absorbing_storage():
    data = _dataset()
    motion, events = data.training_information_signals(
        gripper_indices=(6,), event_threshold=0.1, arm_motion="relative_command_magnitude"
    )
    assert np.isfinite(motion).all()
    # Real gripper is always +1; the corrupt absorbing filler is not an event.
    assert not events.any()
    assert motion[-1] < 100
    last = data[len(data) - 1]
    assert float(last["frame_progress"]) == pytest.approx(55 / 56)
    assert data.boundary_summary()["eligible_labeled_current_rows"] == 56


@pytest.mark.parametrize("remaining", [1, 9])
def test_cpu_bf16_production_tail_forward_backward_is_finite(remaining: int):
    """CPU autocast coverage only; this does not certify CUDA/BF16 kernels."""
    torch.manual_seed(417)
    base = _config()
    cfg = replace(base, runtime=replace(base.runtime, compute_dtype="bf16"))
    model, engine = _model_engine(cfg)
    result = engine.train_step(_batch(remaining, invalid=float("nan")), collect_diagnostics=True)
    assert torch.isfinite(result.loss)
    assert result.metrics["loss_action_observed_label_rows"] == remaining
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_deployment_restore_drops_training_only_support_but_keeps_same_action(tmp_path: Path):
    """Use the real ABI/checkpoint loader; only external T5 bytes are a fixture."""
    from test_mainline_checkpoint import _dataset as identity_fixture

    from clearvla.data.action_chart import resolve_action_state_profile
    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.runtime.checkpoints import save_checkpoint
    from clearvla.mainline.runtime.deployment import build_deployment_abi, canonical_sha256
    from clearvla.mainline.runtime.sampling import sample_action
    from clearvla.simulation.checkpoint import load_deployment_checkpoint

    torch.manual_seed(631)
    cfg = _config()
    model, engine = _model_engine(cfg)
    batch = _batch(9)
    engine.train_step(batch)
    model.eval()
    normalizer = _normalizer()
    language = tmp_path / "language.pt"
    torch.save({"tokens": batch.online.goal.tokens, "mask": batch.online.goal.mask}, language)
    digest = canonical_sha256(normalizer.to_dict())
    dataset = replace(
        identity_fixture(), action_normalizer_sha256=digest, state_normalizer_sha256=digest
    )
    identity = build_checkpoint_identity(
        cfg,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=dataset,
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="3" * 40,
    )
    profile = resolve_action_state_profile(cfg.data.data_profile)
    abi = build_deployment_abi(
        cfg,
        identity,
        action_normalizer=normalizer,
        state_normalizer=normalizer,
        data_profile={
            **profile.as_dict(),
            "gripper_transition_boundary": profile.gripper_transition_boundary,
        },
        gripper_indices=(6,),
        goal_metadata={},
    )
    path = tmp_path / "deployment.pt"
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
            "action_normalizer": normalizer.to_dict(),
            "state_normalizer": normalizer.to_dict(),
            "deployment_abi": abi,
        },
    )
    restored = load_deployment_checkpoint(path, device=torch.device("cpu"))
    # Data-only source support is intentionally not imported into inference.
    assert restored.config.data.window_boundary_contract == "strict_complete_v1"
    assert restored.config.top.history_encoding_mode == cfg.top.history_encoding_mode
    with torch.no_grad():
        first = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(92))
        second = sample_action(
            restored.model,
            batch.online,
            restored.config,
            generator=torch.Generator().manual_seed(92),
        )
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    assert restored.global_step == 1
