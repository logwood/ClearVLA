"""M1c: native observations, model features and commands are distinct charts.

Only image/DINO/T5 transport is replaced where stated; model, window builder,
training engine, deployment adapter and checkpoint readers are production code.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_mainline_checkpoint import _dataset as identity_fixture
from test_mainline_observed_tail import _config as tail_config
from test_mainline_observed_tail import _model_engine, _normalizer
from test_mainline_timed_history import _ImageStore
from torch.utils.data import default_collate

from clearvla.data.action_chart import resolve_action_state_profile
from clearvla.data.hdf5_episode import RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING, LoadedEpisode
from clearvla.data.state_features import (
    CALVIN_ROTATION6D_STATE,
    NATIVE_AFFINE_STATE,
    encode_state_features,
    euler_xyz_rotation_columns,
    state_feature_metadata,
)
from clearvla.data.window_boundaries import OBSERVED_TAIL_V1
from clearvla.mainline.checkpoint import ArtifactIdentity, build_checkpoint_identity
from clearvla.mainline.config import ExperimentConfig, config_from_mapping, load_config
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.loading import GoalTemplate, to_training_batch
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.interfaces import TrainingBatch
from clearvla.mainline.model.timed_history import TimedHistoryEncoder
from clearvla.mainline.runtime.checkpoints import load_checkpoint_exact, save_checkpoint
from clearvla.mainline.runtime.deployment import (
    build_deployment_abi,
    canonical_sha256,
    validate_deployment_abi,
)
from clearvla.mainline.runtime.sampling import sample_action
from clearvla.mainline.training.engine import validate_finite_training_batch
from clearvla.simulation.checkpoint import load_deployment_checkpoint
from clearvla.simulation.contracts import PolicyObservation
from clearvla.simulation.history import CausalHistory

PROFILE = "calvin_relative_7d_v1"


def _config() -> ExperimentConfig:
    base = tail_config()
    cfg = replace(base, dimensions=replace(base.dimensions, state_dim=10),
                  top=replace(base.top, state_feature_mode=CALVIN_ROTATION6D_STATE))
    cfg.validate()
    return cfg


def _encode(value: np.ndarray, normalizer: ArrayNormalizer) -> np.ndarray:
    return encode_state_features(value, normalizer, mode=CALVIN_ROTATION6D_STATE, profile=PROFILE)


def _data() -> ObservedStateWindowDataset:
    t = np.arange(81, dtype=np.float32)
    states = np.stack((.2+.001*t, -.1+.0007*t, .4+.0003*t,
                       .2+.002*t, -.1+.001*t, 3.10+.003*t, .04+.0001*t), axis=-1)
    actions = np.stack([np.sin(.1*t+j)*.2 for j in range(6)] + [np.ones_like(t)], axis=-1)
    boundary = np.concatenate((np.zeros((1, 7), np.float32), actions[:-1]))
    e = LoadedEpisode(path=Path("state-chart.hdf5"), episode_id="state-chart",
                      source_partition="", task_id="", action_key="action", camera_keys={},
                      data_profile=PROFILE, actions_raw=actions, states_raw=states,
                      action_states_raw=boundary, valid_center_start=0, valid_center_end=56,
                      terminal_state_index=80,
                      terminal_padding_mode=RELATIVE_ACTION_ABSORBING_TERMINAL_PADDING)
    return ObservedStateWindowDataset(
        [e], [0], image_store=cast(Any, _ImageStore()), camera_names=("top", "wrist"),
        action_normalizer=_normalizer(), state_normalizer=ArrayNormalizer.fit_zscore([states]),
        config=ObservedStateDatasetConfig(window_boundary_contract=OBSERVED_TAIL_V1,
                  causal_reset_padding=True, emit_history_timing=True,
                  state_feature_mode=CALVIN_ROTATION6D_STATE, state_profile=PROFILE))


class _TokenTransport:
    def load_batch(self, keys: torch.Tensor) -> torch.Tensor:
        return (keys[:, 1, None, None, None].expand(-1, 2, 64, 16).float() / 80).contiguous()


def _training(data: ObservedStateWindowDataset, centers: tuple[int, ...]) -> TrainingBatch:
    cache = CachedTokenPolicyWindowDataset(data, token_store=cast(Any, _TokenTransport()))
    raw = default_collate([cache[i] for i in centers])
    goal = GoalTemplate(torch.zeros(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), {})
    return to_training_batch(raw, goal=goal, config=_config(), device=torch.device("cpu"))


@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7)])
@pytest.mark.parametrize("kind", ["zscore", "limits", "identity"])
def test_native_affine_control_is_exact(shape: tuple[int, ...], kind: str):
    rng = np.random.default_rng(15)
    normalizer = getattr(ArrayNormalizer, "fit_" + kind)([rng.normal(size=(40, 7))])
    x = rng.normal(size=shape)
    actual = encode_state_features(x, normalizer, mode=NATIVE_AFFINE_STATE, profile=PROFILE)
    np.testing.assert_array_equal(actual, normalizer.encode(x))
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


def test_rotation_columns_match_independent_matrix_product_and_preserve_linear_chart():
    rng = np.random.default_rng(901)
    states = rng.normal(size=(30, 7)).astype(np.float32)
    norm = ArrayNormalizer.fit_zscore([states])
    result = _encode(states, norm)
    np.testing.assert_array_equal(result[:, [0, 1, 2, 9]], norm.encode(states)[:, [0, 1, 2, 6]])
    for row, expected in zip(states, result, strict=True):
        r, p, y = row[3:6].astype(np.float64)
        rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        R = rz @ ry @ rx
        np.testing.assert_allclose(expected[3:9], R[:, :2].T.reshape(-1), atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(expected[3:6]), 1, atol=2e-7)
        np.testing.assert_allclose(np.dot(expected[3:6], expected[6:9]), 0, atol=2e-7)


def test_equivalent_euler_charts_and_wrap_boundary_have_consistent_features():
    euler = np.array([[.31, -.27, 2.4]])
    np.testing.assert_allclose(euler_xyz_rotation_columns(euler),
        euler_xyz_rotation_columns(euler + 2*np.pi), atol=1e-7)
    other = np.array([[euler[0, 0]+np.pi, np.pi-euler[0, 1], euler[0, 2]+np.pi]])
    np.testing.assert_allclose(euler_xyz_rotation_columns(euler),
                              euler_xyz_rotation_columns(other), atol=1e-7)
    near = euler_xyz_rotation_columns(np.array([[0, 0, np.pi-1e-5], [0, 0, -np.pi+1e-5]]))
    assert np.linalg.norm(near[1]-near[0]) < 4e-5


def test_rotation_ignores_unused_affine_angle_extrema_without_overflow():
    norm = _normalizer()
    scale, offset = norm.scale.copy(), norm.offset.copy()
    scale[:, 3:6], offset[:, 3:6] = np.finfo(np.float32).max, np.finfo(np.float32).max
    changed = replace(norm, scale=scale, offset=offset)
    states = np.full((3, 7), 4.0, np.float32)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_encode(states, norm), _encode(states, changed))


@pytest.mark.parametrize("value", [np.zeros((1, 6)), np.zeros((1, 10)),
    np.full((1, 7), np.nan), np.full((1, 7), np.inf), np.ones((1, 7), complex),
    np.ones((1, 7), bool), np.array(1.)])
def test_invalid_native_source_is_not_silently_reinterpreted(value: np.ndarray):
    with pytest.raises(ValueError):
        _encode(value, _normalizer())


@pytest.mark.parametrize("field,value", [("scale", np.zeros((1, 7))),
    ("scale", -np.ones((1, 7))), ("offset", np.zeros((1, 10))),
    ("offset", np.full((1, 7), np.nan))])
def test_feature_encoder_requires_native_normalizer(field: str, value: np.ndarray):
    with pytest.raises(ValueError):
        _encode(np.zeros((1, 7)), replace(_normalizer(), **{field: value}))


def test_config_and_metadata_separate_state_and_action_and_reject_unknown_conventions():
    cfg = _config()
    assert config_from_mapping(cfg.as_dict()) == cfg
    assert "state_feature_mode" not in cast(dict[str, object], tail_config().as_dict()["top"])
    loaded = load_config("configs/mainline/structural_rebuild_m1c_calvin.json")
    assert loaded.dimensions.state_dim == 10 and loaded.dimensions.action_dim == 7
    metadata = state_feature_metadata(CALVIN_ROTATION6D_STATE, PROFILE, 7)
    assert metadata["native_state_dim"] == 7 and metadata["feature_state_dim"] == 10
    assert metadata["seconds_per_step"] is None
    with pytest.raises(ValueError):
        replace(cfg, dimensions=replace(cfg.dimensions, state_dim=7)).validate()
    with pytest.raises(ValueError):
        replace(cfg, top=replace(cfg.top, history_encoding_mode="paired_rows_v1")).validate()
    with pytest.raises(ValueError):
        encode_state_features(np.zeros(7), _normalizer(), mode=CALVIN_ROTATION6D_STATE,
                              profile="libero_relative_7d_v1")


def test_dataset_rejects_wrong_native_episode_identity():
    data = _data()
    wrong = replace(data.episodes[0], data_profile="identity_7d_pen")
    with pytest.raises(ValueError, match="projected episode"):
        ObservedStateWindowDataset([wrong], [0], image_store=data.image_store,
            camera_names=data.camera_names, state_normalizer=data.state_normalizer,
            action_normalizer=data.action_normalizer, config=data.config)


@pytest.mark.parametrize("center", [0, 1, 4, 8, 24, 79])
def test_production_dataset_loader_and_online_adapter_use_identical_state_chart(
    center: int, monkeypatch: pytest.MonkeyPatch,
):
    import clearvla.simulation.clearvla_policy as module
    data = _data()
    e = data.episodes[0]
    assert e.states_raw is not None and e.action_states_raw is not None
    states, boundaries = e.states_raw, e.action_states_raw
    history = CausalHistory()
    def obs(i: int) -> PolicyObservation:
        return PolicyObservation(rgb={k: np.full((32, 32, 3), i, np.uint8) for k in ("top", "wrist")},
                                 state=states[i], action_state=boundaries[i])
    history.reset(obs(0))
    for i in range(center):
        history.append(e.actions_raw[i], obs(i+1))
    policy = cast(Any, module.ClearVLACheckpointPolicy.__new__(module.ClearVLACheckpointPolicy))
    policy.bundle = SimpleNamespace(config=_config(), state_normalizer=data.state_normalizer,
        action_normalizer=data.action_normalizer, model=object(),
        language=SimpleNamespace(is_instruction_bank=False, tokens=torch.zeros(1, 4, 16),
                                 mask=torch.ones(1, 4, dtype=torch.bool)))
    policy.encoder = SimpleNamespace(encode=lambda *_: (
        torch.zeros(3, 2, 64, 16), np.zeros((3, 2, 32, 32, 3), np.uint8)))
    policy.preprocessing = None
    policy.device = torch.device("cpu")
    policy._generator = torch.Generator()
    monkeypatch.setattr(module, "sample_action", lambda *_, **__: SimpleNamespace(
        action=torch.zeros(1, 24, 7), gripper_command=torch.ones(1, 24)))
    native_action, online = policy.act_with_input(history.snapshot(), "push block")
    batch = _training(data, (center,))
    for name in ("state", "state_history", "action_state", "executed_action_history",
                 "codec_gripper_boundary"):
        torch.testing.assert_close(getattr(online.history, name), getattr(batch.online.history, name),
                                   atol=0, rtol=0)
    assert online.history.state.shape == (1, 10) and native_action.shape == (24, 7)
    raw = data[center]
    assert raw["state_raw"].shape == (7,) and raw["action_state_raw"].shape == (7,)
    mask = raw["future_state_observed"]
    if not bool(mask.all()):
        assert torch.count_nonzero(raw["future_state"][~mask]) == 0
    expected = _encode(states[min(center+1, 80)], data.state_normalizer)
    np.testing.assert_array_equal(raw["future_state"][0].numpy(), expected)
    validate_finite_training_batch(batch)


def test_timed_feature_difference_does_not_invent_a_wraparound_rotation():
    data = _data()
    b = _training(data, (16,))
    assert b.online.history.timing is not None
    encoder = TimedHistoryEncoder(state_dim=10, action_dim=7, hidden=16, heads=4)
    result = encoder(b.online.history.state_history, b.online.history.state,
                     b.online.history.executed_action_history, b.online.history.timing)
    assert torch.isfinite(result.state_rates).all()
    assert result.state_rates[..., 3:9].abs().max() < .01


@pytest.mark.parametrize("center", [0, 16, 79])
def test_production_model_trains_native7_actions_with_feature10_states(center: int):
    torch.manual_seed(301)
    data, cfg = _data(), _config()
    batch = _training(data, (center,))
    model, engine = _model_engine(cfg)
    model.configure_action_normalizer(data.action_normalizer)
    engine.train_step(batch)
    assert engine.global_step == 1
    history_encoder = model.intent.organizer.timed_history
    assert history_encoder is not None
    parameters = dict(history_encoder.named_parameters())
    # The real ten-dimensional source projection, not a replacement encoder.
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters.values())
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_native_normalizers_exact_training_and_deployment_restore(tmp_path: Path):
    torch.manual_seed(631)
    data, cfg = _data(), _config()
    batch = _training(data, (16,))
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
    assert "clearvla/data/state_features.py" in dict(identity.source.files)
    profile = resolve_action_state_profile(PROFILE)
    abi = build_deployment_abi(cfg, identity, action_normalizer=data.action_normalizer,
        state_normalizer=data.state_normalizer,
        data_profile={**profile.as_dict(), "gripper_transition_boundary":profile.gripper_transition_boundary},
        gripper_indices=(6,), goal_metadata={})
    validate_deployment_abi(abi)
    for key in ("state_features", "state_dim"):
        broken = copy.deepcopy(abi)
        observation = cast(dict[str, object], broken["observation"])
        del observation[key]
        with pytest.raises(ValueError):
            validate_deployment_abi(broken)
    broken = copy.deepcopy(abi)
    observation = cast(dict[str, object], broken["observation"])
    features = cast(dict[str, object], observation["state_features"])
    features["orientation_matrix"] = "Rx @ Ry @ Rz"
    with pytest.raises(ValueError):
        validate_deployment_abi(broken)
    path = tmp_path / "state-features.pt"
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
            schedule=restored_engine.schedule, config=tail_config(), identity=identity)
    bundle = load_deployment_checkpoint(path, device=torch.device("cpu"))
    assert bundle.state_normalizer.scale.shape == (1, 7)
    assert bundle.config.dimensions.state_dim == 10 and bundle.config.dimensions.action_dim == 7
    with torch.no_grad():
        first = sample_action(model, batch.online, cfg, generator=torch.Generator().manual_seed(92))
        second = sample_action(bundle.model, batch.online, bundle.config,
                               generator=torch.Generator().manual_seed(92))
    torch.testing.assert_close(first.action, second.action, rtol=0, atol=0)
    assert first.action.shape == (1, 24, 7)
    assert bundle.global_step == 1
