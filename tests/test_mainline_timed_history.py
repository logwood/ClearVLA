"""M1 contracts against the actual production data, model and online classes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pytest
import torch
from test_mainline_policy import _batch, _calvin_binary_config, _config

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.data.history_clock import HISTORY_TIMING_KEYS, sparse_history_clock
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.data.dataset import ObservedStateDatasetConfig, ObservedStateWindowDataset
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import OnlinePolicyInput, TrainingBatch
from clearvla.mainline.model.component_contracts import ComponentSelection
from clearvla.mainline.model.policy import ClearVLAMainlinePolicy
from clearvla.mainline.model.proposal import HistoryActionProposal
from clearvla.mainline.model.timed_history import TimedHistoryEncoder
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.temporal import TIMED_HISTORY_ENCODING, HistoryTiming
from clearvla.mainline.training.engine import MainlineTrainingEngine
from clearvla.mainline.training.optimizer import WarmupCosineSchedule, build_optimizer
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory


def _timing(now: int, batch: int = 1) -> HistoryTiming:
    clock = sparse_history_clock(now)
    return HistoryTiming.from_mapping(
        {k: torch.from_numpy(v)[None].expand(batch, -1) for k, v in clock.items()}
    )


def _timed_config(binary: bool = False) -> ExperimentConfig:
    base = _calvin_binary_config() if binary else _config()
    return replace(
        base,
        top=replace(
            base.top,
            history_encoding_mode=TIMED_HISTORY_ENCODING,
            world_action_condition_mode="sequence_prefix_v1",
        ),
    )


def _timed_batch(config: ExperimentConfig, now: int = 32) -> TrainingBatch:
    batch = _batch(config)
    return replace(
        batch,
        online=replace(batch.online, history=replace(batch.online.history, timing=_timing(now))),
    )


def _observation(index: int) -> PolicyObservation:
    return PolicyObservation(
        rgb={k: np.full((32, 32, 3), index % 255, np.uint8) for k in ("top", "wrist")},
        state=np.full(7, index, np.float32),
        action_state=np.full(7, max(index - 1, 0), np.float32),
    )


@pytest.mark.parametrize("now", [0, 1, 4, 8, 24, 25, 1000])
def test_physical_clock_is_causal_and_marks_padding(now: int):
    clock = _timing(now)
    clock.validate(batch=1, states=3, actions=8, device=torch.device("cpu"), strict=True)
    assert clock.state_offsets[0, -1] == 0
    assert bool(clock.state_observed[0, -1])
    assert torch.equal(clock.action_offsets[0], torch.tensor([-24, -16, -12, -8, -6, -4, -2, -1]))
    if now == 0:
        assert clock.state_offsets.tolist() == [[0, 0, 0]]
        assert clock.state_observed.tolist() == [[False, False, True]]
        assert not bool(clock.action_executed.any())
    if now >= 24:
        assert bool(clock.state_observed.all()) and bool(clock.action_executed.all())


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_clock_rejects_invalid_physical_step(bad: Any):
    with pytest.raises((TypeError, ValueError)):
        sparse_history_clock(bad)


def test_clock_rejects_future_and_duplicate_real_times():
    good = _timing(32)
    for broken in (
        replace(good, action_offsets=torch.tensor([[-24, -16, -12, -8, -6, -4, -2, 0]])),
        replace(good, state_offsets=torch.tensor([[-8, 0, 0]])),
        replace(good, state_observed=torch.zeros_like(good.state_observed)),
        replace(
            good,
            state_offsets=torch.zeros_like(good.state_offsets),
            state_observed=torch.tensor([[True, False, True]]),
        ),
    ):
        with pytest.raises(ValueError):
            broken.validate(batch=1, states=3, actions=8, device=torch.device("cpu"), strict=True)


def test_timed_encoder_preserves_sparse_times_and_physical_state_rates():
    encoder = TimedHistoryEncoder(state_dim=1, action_dim=1, hidden=16, heads=4)
    result = encoder(
        torch.tensor([[[16.0], [24.0], [32.0]]]),
        torch.tensor([[32.0]]),
        torch.arange(8.0).reshape(1, 8, 1),
        _timing(32),
    )
    assert result.offsets.tolist() == [[-24, -16, -12, -8, -8, -6, -4, -4, -2, -1, 0]]
    assert result.kinds.tolist() == [[1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]]
    assert result.tokens.shape == (1, 11, 16)
    assert result.state_rates[result.rate_valid].tolist() == [[2.0], [2.0]]
    assert result.offsets[result.kinds == 0].tolist() == [-8, -4, 0]


def test_timed_encoder_masks_padded_nan_and_its_gradient():
    torch.manual_seed(20)
    encoder = TimedHistoryEncoder(state_dim=2, action_dim=2, hidden=16, heads=4)
    states = torch.full((1, 3, 2), float("nan"), requires_grad=True)
    actions = torch.full((1, 8, 2), float("nan"), requires_grad=True)
    current = torch.ones(1, 2, requires_grad=True)
    result = encoder(states, current, actions, _timing(0))
    assert torch.isfinite(result.tokens).all()
    assert result.valid.sum() == 1 and not result.rate_valid.any()
    result.tokens.square().sum().backward()
    assert states.grad is not None and torch.count_nonzero(states.grad) == 0
    assert actions.grad is not None and torch.count_nonzero(actions.grad) == 0
    assert current.grad is not None and torch.isfinite(current.grad).all()
    for parameter in encoder.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_timing_changes_features_without_changing_values_or_order():
    torch.manual_seed(21)
    encoder = TimedHistoryEncoder(state_dim=2, action_dim=2, hidden=16, heads=4).eval()
    states, actions, current = torch.randn(1, 3, 2), torch.randn(1, 8, 2), torch.randn(1, 2)
    first = _timing(32)
    second = replace(
        first, state_offsets=2 * first.state_offsets, action_offsets=2 * first.action_offsets
    )
    a, b = encoder(states, current, actions, first), encoder(states, current, actions, second)
    assert torch.equal(a.kinds, b.kinds)
    assert not torch.allclose(a.tokens, b.tokens)


def test_causal_timed_tokens_do_not_read_later_observations():
    torch.manual_seed(22)
    encoder = TimedHistoryEncoder(state_dim=2, action_dim=2, hidden=16, heads=4).eval()
    states, actions, current = torch.randn(1, 3, 2), torch.randn(1, 8, 2), torch.randn(1, 2)
    a = encoder(states, current, actions, _timing(32))
    b = encoder(states, current + 500, actions, _timing(32))
    torch.testing.assert_close(a.tokens[:, :-1], b.tokens[:, :-1], rtol=0, atol=0)


@pytest.mark.parametrize("now", [0, 2, 8, 32])
def test_history_proposal_quarantines_unexecuted_rows(now: int):
    model = HistoryActionProposal(action_dim=7, hidden=16, heads=4, horizon=24, history_length=8)
    timing = _timing(now)
    original = torch.randn(1, 8, 7)
    contaminated = torch.where(
        timing.action_executed[..., None], original, torch.full_like(original, float("nan"))
    )
    a = model(original, timing=timing)
    b = model(contaminated, timing=timing)
    torch.testing.assert_close(a.action_prediction, b.action_prediction, rtol=0, atol=0)
    assert torch.isfinite(b.history_tokens).all()
    if now == 0:
        assert torch.count_nonzero(b.history_tokens) == 0


def test_online_history_is_bounded_but_keeps_absolute_time_and_copy_isolation():
    history = CausalHistory()
    history.reset(_observation(0))
    for index in range(1, 2001):
        history.append(np.full(7, index - 1, np.float32), _observation(index))
    snap = history.snapshot()
    assert snap.time_index == 2000
    assert len(history._observations) == 9 and len(history._actions) == 24
    np.testing.assert_array_equal(snap.state_history[:, 0], [1992, 1996, 2000])
    np.testing.assert_array_equal(
        snap.executed_action_history[:, 0], [1976, 1984, 1988, 1992, 1994, 1996, 1998, 1999]
    )
    snap.state[:] = -100
    assert history.snapshot().state[0] == 2000
    with pytest.raises(IndexError):
        history._action_at(2000)
    with pytest.raises(IndexError):
        history._observation_at(1900)
    history.reset(_observation(0))
    assert history.time_index == 0
    assert not history.snapshot().timing_arrays()["history_action_executed"].any()


class _ImageStore:
    def validate_episode(self, episode: LoadedEpisode) -> None:
        pass

    def load_window(self, episode: LoadedEpisode, rows: Sequence[int]) -> dict[str, torch.Tensor]:
        return {
            k: torch.stack(
                [torch.full((3, 32, 32), int(row) % 255, dtype=torch.uint8) for row in rows]
            )
            for k in ("top", "wrist")
        }


@pytest.mark.parametrize("center", [0, 1, 4, 8, 24, 40])
def test_dataset_and_online_history_own_identical_time_and_values(center: int):
    length = 100
    state = np.repeat(np.arange(length, dtype=np.float32)[:, None], 7, axis=1)
    action = np.repeat(np.arange(length, dtype=np.float32)[:, None], 7, axis=1)
    previous = np.concatenate((np.zeros((1, 7), np.float32), action[:-1]))
    episode = LoadedEpisode(
        path=Path("clock.hdf5"),
        episode_id="clock",
        source_partition="",
        task_id="",
        action_key="action",
        camera_keys={},
        actions_raw=action,
        states_raw=state,
        action_states_raw=previous,
    )
    norm = ArrayNormalizer.fit_identity([state])
    cfg = ObservedStateDatasetConfig(causal_reset_padding=True, emit_history_timing=True)
    dataset = ObservedStateWindowDataset(
        [episode],
        [0],
        image_store=cast(
            Any, _ImageStore()
        ),  # Only image I/O is replaced by a deterministic fixture.
        camera_names=("top", "wrist"),
        state_normalizer=norm,
        action_normalizer=norm,
        config=cfg,
    )
    index = next(i for i, ref in enumerate(dataset.refs) if ref.center == center)
    sample = dataset[index]
    history = CausalHistory()
    history.reset(_observation(0))
    for i in range(1, center + 1):
        history.append(action[i - 1], _observation(i))
    snap = history.snapshot()
    for key, value in snap.timing_arrays().items():
        np.testing.assert_array_equal(sample[key].numpy(), value)
    np.testing.assert_array_equal(sample["history_state"].numpy(), snap.state_history)
    np.testing.assert_array_equal(
        sample["executed_action_history"].numpy(), snap.executed_action_history
    )


def test_new_mode_requires_source_clock_and_has_distinct_config_and_component_identity():
    base = _config()
    config = _timed_config()
    assert "history_encoding_mode" not in cast(dict[str, object], base.as_dict()["top"])
    assert config_from_mapping(config.as_dict()) == config
    assert (
        ComponentSelection.from_config(base).intent != ComponentSelection.from_config(config).intent
    )
    with pytest.raises(ValueError, match="producer timing"):
        _batch(config).validate(config)
    candidate = load_config("configs/mainline/structural_rebuild_m1_calvin.json")
    assert candidate.top.history_encoding_mode == TIMED_HISTORY_ENCODING


@pytest.mark.parametrize("now", [0, 4, 32])
def test_real_production_model_consumes_clock_and_preserves_masks_through_coarse(now: int):
    config = _timed_config()
    batch = _timed_batch(config, now)
    model = ClearVLAMainlinePolicy(config).eval()
    model.configure_action_normalizer(ArrayNormalizer.fit_identity([np.zeros((2, 7), np.float32)]))
    with torch.no_grad():
        cache, training, _ = model.encode_online(batch.online)
    intent = training.top.intent
    assert intent.history_tokens.shape[1] == 11
    assert intent.history_validity is not None
    assert batch.online.history.timing is not None
    assert int(intent.history_validity.sum()) == int(
        batch.online.history.timing.state_observed.sum()
        + batch.online.history.timing.action_executed.sum()
    )
    assert intent.action_dock().history_validity is intent.history_validity
    assert torch.isfinite(training.top.coarse_action.action_prediction).all()
    assert model.intent.organizer.timed_history is not None
    assert model.intent.organizer.history_input is None
    assert cache.history.timing is not None


def test_new_model_full_training_save_restore_and_online_sampling(tmp_path: Path):
    torch.manual_seed(76)
    config = _timed_config(binary=True)
    batch = _timed_batch(config, 32)
    model = ClearVLAMainlinePolicy(config)
    normalizer = ArrayNormalizer.fit_zscore(
        [np.zeros((2, 7), np.float32), np.ones((2, 7), np.float32)]
    )
    model.configure_action_normalizer(normalizer)
    optimizer, _ = build_optimizer(model, config)
    engine = MainlineTrainingEngine(
        model=model,
        config=config,
        optimizer=optimizer,
        schedule=WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1),
        device=torch.device("cpu"),
    )
    result = engine.train_step(batch)
    assert torch.isfinite(result.loss)
    encoder = model.intent.organizer.timed_history
    assert encoder is not None
    assert encoder.state_input.weight.grad is not None
    assert encoder.action_input.weight.grad is not None
    assert torch.isfinite(encoder.state_input.weight.grad).all()
    assert torch.isfinite(encoder.action_input.weight.grad).all()
    assert torch.count_nonzero(encoder.state_input.weight.grad) > 0
    assert torch.count_nonzero(encoder.action_input.weight.grad) > 0
    path = tmp_path / "roundtrip.pt"
    torch.save(model.state_dict(), path)
    restored = ClearVLAMainlinePolicy(config)
    restored.configure_action_normalizer(normalizer)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.eval()
    model.eval()
    timing = batch.online.history.timing
    assert timing is not None
    before = {key: value.clone() for key, value in timing.as_mapping().items()}
    with torch.no_grad():
        first = sample_action(
            model, batch.online, config, generator=torch.Generator().manual_seed(88)
        )
        second = sample_action(
            restored, batch.online, config, generator=torch.Generator().manual_seed(88)
        )
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    assert first.action.shape == (1, 24, 7) and torch.isfinite(first.action).all()
    for key, value in before.items():
        torch.testing.assert_close(value, timing.as_mapping()[key], rtol=0, atol=0)
    legacy = ClearVLAMainlinePolicy(_calvin_binary_config())
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(restored.state_dict(), strict=True)


@pytest.mark.parametrize("requested_device", ["cpu", "cpu:0"])
def test_training_batch_requires_all_clock_fields_and_transfers_provenance(requested_device: str):
    config = _timed_config()
    b = 2
    source = {
        "history_dinov2_tokens": torch.zeros(b, 3, 2, 64, 16),
        "history_obs_image": torch.zeros(b, 3, 2, 3, 32, 32),
        "state": torch.zeros(b, 7),
        "action_state": torch.zeros(b, 7),
        "gripper_transition_boundary": torch.zeros(b, 7),
        "history_state": torch.zeros(b, 3, 7),
        "executed_action_history": torch.zeros(b, 8, 7),
        "policy_action": torch.zeros(b, 24, 7),
        "policy_action_raw": torch.zeros(b, 24, 7),
        "action_state_raw": torch.zeros(b, 7),
        "gripper_transition_boundary_raw": torch.zeros(b, 7),
        "action": torch.zeros(b, 48, 7),
        "future_state": torch.zeros(b, 48, 7),
        "target_future_dinov2_tokens": torch.zeros(b, 12, 2, 64, 16),
        "target_future_offsets": torch.arange(4, 49, 4)[None].expand(b, -1),
    }
    goal = GoalTemplate(torch.zeros(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {})
    with pytest.raises(KeyError, match="history_state_offsets"):
        to_training_batch(source, goal=goal, config=config, device=torch.device(requested_device))
    source.update(_timing(32, b).as_mapping())
    typed = to_training_batch(
        source, goal=goal, config=config, device=torch.device(requested_device)
    )
    assert typed.online.history.timing is not None
    for name in HISTORY_TIMING_KEYS:
        torch.testing.assert_close(typed.online.history.timing.as_mapping()[name], source[name])
    with pytest.raises(ValueError, match="future state|unexecuted current action"):
        broken = dict(source)
        broken["history_action_offsets"] = torch.zeros(b, 8, dtype=torch.long)
        to_training_batch(broken, goal=goal, config=config, device=torch.device(requested_device))


def test_deployment_abi_timing_cannot_be_removed_or_forged():
    from copy import deepcopy

    from test_maniskill_binary_gripper import _binary_deployment_abi

    from clearvla.mainline.runtime.deployment import canonical_sha256, validate_deployment_abi
    from clearvla.mainline.temporal import HISTORY_TIMING_CONTRACT

    abi = cast(dict[str, Any], _binary_deployment_abi())
    abi["graph_config"]["top"]["history_encoding_mode"] = TIMED_HISTORY_ENCODING
    abi["graph_config_sha256"] = canonical_sha256(abi["graph_config"])
    with pytest.raises(ValueError, match="history timing contract"):
        validate_deployment_abi(abi)
    abi["observation"]["history_timing_contract"] = HISTORY_TIMING_CONTRACT
    validate_deployment_abi(abi)
    bad = deepcopy(abi)
    bad["observation"]["history_timing_contract"] = "uniform_fake_clock"
    with pytest.raises(ValueError, match="history timing contract"):
        validate_deployment_abi(bad)


def test_real_checkpoint_api_roundtrip_and_legacy_mode_rejection(tmp_path: Path):
    from test_mainline_checkpoint import _dataset

    from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
    from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint

    config = _timed_config()
    language = tmp_path / "goal.pt"
    language.write_bytes(b"test provenance only")
    identity = build_checkpoint_identity(
        config,
        repo_root=Path(__file__).resolve().parents[1],
        dataset=_dataset(),
        language=ArtifactIdentity.from_file("t5_goal", language),
        commit="1" * 40,
    )
    model = ClearVLAMainlinePolicy(config)
    optimizer, _ = build_optimizer(model, config)
    schedule = WarmupCosineSchedule(optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1)
    path = tmp_path / "real-api.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        config=config,
        identity=identity,
        epoch=0,
        global_step=0,
        best_metric=None,
    )
    restored = ClearVLAMainlinePolicy(config)
    new_optimizer, _ = build_optimizer(restored, config)
    new_schedule = WarmupCosineSchedule(
        new_optimizer, warmup_steps=2, total_steps=4, minimum_ratio=0.1
    )
    load_checkpoint_exact(
        path,
        model=restored,
        optimizer=new_optimizer,
        schedule=new_schedule,
        config=config,
        identity=identity,
    )
    assert restored.selection.intent == "timestamped_object_intent_v1"
    assert restored.selection.conditioning == "timestamped_observable_history_v1"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    old = replace(config, top=replace(config.top, history_encoding_mode="paired_rows_v1"))
    with pytest.raises(ValueError):
        load_checkpoint_exact(
            path,
            model=restored,
            optimizer=new_optimizer,
            schedule=new_schedule,
            config=old,
            identity=identity,
        )


@pytest.mark.parametrize("now", [0, 4, 32])
@pytest.mark.parametrize("requested_device", ["cpu", "cpu:0"])
def test_production_online_adapter_emits_source_clock(
    now: int, requested_device: str, monkeypatch: pytest.MonkeyPatch
):
    from types import SimpleNamespace

    import clearvla.simulation.clearvla_policy as module

    config = _timed_config()
    history = CausalHistory()
    history.reset(_observation(0))
    for t in range(1, now + 1):
        history.append(np.full(7, t - 1, np.float32), _observation(t))
    norm = ArrayNormalizer.fit_identity([np.zeros((2, 7), np.float32)])
    # Constructor dependencies are fixtures; act_with_input below is the production method.
    policy = cast(Any, module.ClearVLACheckpointPolicy.__new__(module.ClearVLACheckpointPolicy))
    policy.bundle = SimpleNamespace(
        config=config,
        state_normalizer=norm,
        action_normalizer=norm,
        language=SimpleNamespace(
            is_instruction_bank=False,
            tokens=torch.zeros(1, 4, 16),
            mask=torch.ones(1, 4, dtype=torch.bool),
        ),
        model=object(),
    )
    policy.encoder = SimpleNamespace(
        encode=lambda *_: (torch.zeros(3, 2, 64, 16), np.zeros((3, 2, 32, 32, 3), np.uint8))
    )
    policy.preprocessing = None
    policy.device = torch.device(requested_device)
    policy._generator = torch.Generator()
    captured = {}

    def sample(
        _model: object, online: OnlinePolicyInput, _config: ExperimentConfig, **_kwargs: Any
    ):
        captured["online"] = online
        return SimpleNamespace(action=torch.zeros(1, 24, 7), gripper_command=None)

    monkeypatch.setattr(module, "sample_action", sample)
    raw, online = policy.act_with_input(history.snapshot(), "test instruction")
    assert raw.shape == (24, 7) and captured["online"] is online
    assert online.history.timing is not None
    for key, value in history.snapshot().timing_arrays().items():
        np.testing.assert_array_equal(online.history.timing.as_mapping()[key][0].numpy(), value)


def test_condition_dropout_removes_execution_evidence_not_state_observations():
    from unittest import mock

    config = _timed_config()
    config = replace(config, top=replace(config.top, action_history_condition_dropout=0.5))
    model = ClearVLAMainlinePolicy(config).train()
    batch = _timed_batch(config)

    def zero_random(*shape: int, **kwargs: Any):
        kwargs.pop("generator", None)
        return torch.zeros(shape, **kwargs)

    with mock.patch("clearvla.mainline.model.components.torch.rand", side_effect=zero_random):
        conditioned, _, _, keep = model.conditioning.prepare(
            batch.online, config=config, training=True, training_mask=True, condition_generator=None
        )
    assert keep.item() == 0
    assert conditioned.history.timing is not None
    assert not conditioned.history.timing.action_executed.any()
    assert conditioned.history.timing.state_observed.all()
    assert batch.online.history.timing is not None
    assert batch.online.history.timing.action_executed.all()


def test_temporal_sources_are_in_the_active_checkpoint_closure():
    from clearvla.mainline.checkpoint import active_source_snapshot

    snapshot = active_source_snapshot(Path(__file__).resolve().parents[1])
    sources = dict(snapshot.files)
    for path in (
        "clearvla/data/history_clock.py",
        "clearvla/mainline/temporal.py",
        "clearvla/mainline/model/timed_history.py",
        "clearvla/simulation/history.py",
    ):
        assert path in sources


@pytest.mark.parametrize("now", [0, 32])
def test_timestamped_encoder_cpu_bf16_has_finite_reverse_path(now: int):
    encoder = TimedHistoryEncoder(state_dim=7, action_dim=7, hidden=16, heads=4)
    states = torch.randn(2, 3, 7, requires_grad=True)
    actions = torch.randn(2, 8, 7, requires_grad=True)
    current = torch.randn(2, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = encoder(states, current, actions, _timing(now, 2))
        loss = result.tokens.float().square().sum()
    loss.backward()
    assert torch.isfinite(result.tokens).all()
    for parameter in encoder.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("now", [0, 4])
def test_entire_online_encode_is_invariant_to_padded_history_values(now: int):
    config = _timed_config()
    model = ClearVLAMainlinePolicy(config).eval()
    model.configure_action_normalizer(ArrayNormalizer.fit_identity([np.zeros((2, 7), np.float32)]))
    batch = _timed_batch(config, now)
    history = batch.online.history
    timing = history.timing
    assert timing is not None
    changed = replace(
        history,
        state_history=torch.where(
            timing.state_observed[..., None],
            history.state_history,
            torch.full_like(history.state_history, float("nan")),
        ),
        executed_action_history=torch.where(
            timing.action_executed[..., None],
            history.executed_action_history,
            torch.full_like(history.executed_action_history, float("nan")),
        ),
    )
    with torch.no_grad():
        cache_a, training_a, _ = model.encode_online(batch.online)
        cache_b, training_b, _ = model.encode_online(replace(batch.online, history=changed))
    for a, b in (
        (training_a.top.intent.history_tokens, training_b.top.intent.history_tokens),
        (
            training_a.top.coarse_action.action_prediction,
            training_b.top.coarse_action.action_prediction,
        ),
        (cache_a.executed_memory, cache_b.executed_memory),
        (cache_a.factual_dock.protected_detail, cache_b.factual_dock.protected_detail),
    ):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
