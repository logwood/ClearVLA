from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch

from clearvla.data.hdf5_episode import LoadedEpisode
from clearvla.data.samplers import InformationBalancedBatchSampler
from clearvla.mainline.config import ExperimentConfig
from clearvla.mainline.data.dataset import (
    CachedTokenPolicyWindowDataset,
    ObservedStateDatasetConfig,
    ObservedStateWindowDataset,
)
from clearvla.mainline.data.language import load_t5_condition, load_t5_condition_bank
from clearvla.mainline.data.loading import (
    GoalTemplate,
    MainlineDataBundle,
    _configure_worker_tensor_sharing,
    to_training_batch,
)
from clearvla.mainline.data.normalizer import ArrayNormalizer
from clearvla.mainline.data.token_store import DinoV2TokenStore
from clearvla.mainline.runtime.identity import v120_normalizer_fingerprint


def _config() -> ExperimentConfig:
    base = ExperimentConfig()
    config = replace(
        base,
        dimensions=replace(
            base.dimensions,
            visual_token_dim=16,
            goal_token_dim=12,
            patches_per_camera=64,
        ),
    )
    config.validate()
    return config


def test_v120_compatibility_normalizer_fingerprint_is_exact() -> None:
    normalizer = ArrayNormalizer.fit_zscore(
        [np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 8.0]], dtype=np.float32)]
    )
    assert v120_normalizer_fingerprint(normalizer) == "b9a2b34d6697"


class _SamplerImageStore:
    def validate_episode(self, _episode: LoadedEpisode) -> None:
        pass


def _information_signal_dataset(
    actions: np.ndarray,
    action_states: np.ndarray,
    normalizer: ArrayNormalizer,
) -> ObservedStateWindowDataset:
    episode = LoadedEpisode(
        path=Path("sampling-test.hdf5"),
        episode_id="sampling-test",
        source_partition="",
        task_id="",
        action_key="action",
        camera_keys={},
        actions_raw=np.asarray(actions, dtype=np.float32),
        states_raw=np.asarray(action_states, dtype=np.float32),
        action_states_raw=np.asarray(action_states, dtype=np.float32),
    )
    return ObservedStateWindowDataset(
        [episode],
        [0],
        image_store=cast(object, _SamplerImageStore()),
        camera_names=("high", "right_wrist"),
        state_normalizer=normalizer,
        action_normalizer=normalizer,
        config=ObservedStateDatasetConfig(),
    )


def test_calvin_sampler_scores_centered_relative_command_not_its_derivative() -> None:
    length = ObservedStateDatasetConfig().minimum_episode_length
    actions = np.zeros((length, 7), dtype=np.float32)
    actions[:, 0] = 0.20
    actions[:, -1] = -1.0
    action_states = actions.copy()
    scale = np.asarray([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5]], dtype=np.float32)
    offset = np.asarray([[0.4, -0.3, 0.2, -0.1, 0.6, -0.5, 0.25]], dtype=np.float32)
    normalizer = ArrayNormalizer(
        offset=offset,
        scale=scale,
        mean=-offset / scale,
        std=1.0 / scale,
        minimum=np.full((1, 7), -1.0, dtype=np.float32),
        maximum=np.full((1, 7), 1.0, dtype=np.float32),
        mode="zscore",
    )
    dataset = _information_signal_dataset(actions, action_states, normalizer)

    derivative_motion, derivative_event = dataset.training_information_signals(
        gripper_indices=(6,),
        event_threshold=0.1,
        arm_motion="adjacent_action_delta",
    )
    command_motion, command_event = dataset.training_information_signals(
        gripper_indices=(6,),
        event_threshold=0.1,
        arm_motion="relative_command_magnitude",
    )

    np.testing.assert_array_equal(derivative_motion, np.zeros(1, dtype=np.float32))
    np.testing.assert_allclose(
        command_motion,
        np.asarray([0.4 / np.sqrt(6.0)], dtype=np.float32),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_array_equal(derivative_event, np.asarray([False]))
    np.testing.assert_array_equal(command_event, derivative_event)

    # A native zero command encodes to the nonzero affine offset but must
    # still be classified as zero motion by the CALVIN sampler.
    actions[:, :6] = 0.0
    action_states[:, :6] = 0.0
    zero_motion, _ = dataset.training_information_signals(
        gripper_indices=(6,),
        event_threshold=0.1,
        arm_motion="relative_command_magnitude",
    )
    np.testing.assert_array_equal(zero_motion, np.zeros(1, dtype=np.float32))


def test_pen_rdt_sampler_keeps_the_historical_full_action_delta_formula() -> None:
    length = ObservedStateDatasetConfig().minimum_episode_length
    rows = np.arange(length, dtype=np.float32)[:, None]
    actions = np.concatenate(
        tuple((rows * (index + 1) / 100.0) for index in range(7)),
        axis=1,
    ).astype(np.float32)
    action_states = actions - 0.03
    normalizer = ArrayNormalizer.fit_zscore([actions])
    dataset = _information_signal_dataset(actions, action_states, normalizer)

    actual, _ = dataset.training_information_signals(
        gripper_indices=(6,),
        event_threshold=100.0,
        arm_motion="adjacent_action_delta",
    )
    center = 24
    normalized = normalizer.encode(actions[center : center + 24]).astype(np.float32)
    state = normalizer.encode(action_states[center]).astype(np.float32)
    boundary = np.concatenate((state[None], normalized[:-1]), axis=0)
    expected = np.asarray(
        [np.sqrt(np.mean(np.square(normalized - boundary), dtype=np.float64))],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_release_first_signal_uses_previous_command_and_open_direction() -> None:
    length = ObservedStateDatasetConfig().minimum_episode_length
    actions = np.zeros((length, 7), dtype=np.float32)
    actions[:, -1] = 1.0
    actions[24, -1] = -1.0
    action_states = actions.copy()
    action_states[24, -1] = 1.0  # observed qpos need not be the command boundary
    normalizer = ArrayNormalizer.fit_zscore([actions])
    dataset = _information_signal_dataset(actions, action_states, normalizer)
    release = dataset.training_release_first_signals(
        gripper_indices=(6,),
        event_threshold=0.1,
        open_direction=-1,
    )
    # The synthetic strict dataset starts at center 24, so its first window
    # consumes action row 24 and sees action row 23 as the previous command.
    assert release.shape == (1,)
    assert bool(release[0])


def test_dataset_batch_is_partitioned_into_online_target_and_teacher_planes() -> None:
    config = _config()
    batch = 2
    raw = {
        "history_dinov2_tokens": torch.randn(batch, 3, 2, 64, 16).half(),
        "history_obs_image": torch.rand(batch, 3, 2, 3, 32, 32),
        "state": torch.randn(batch, 7),
        "state_raw": torch.randn(batch, 7),
        "action_state_raw": torch.randn(batch, 7),
        "action_state": torch.randn(batch, 7),
        "gripper_transition_boundary": torch.randn(batch, 7),
        "gripper_transition_boundary_raw": torch.randn(batch, 7),
        "history_state": torch.randn(batch, 3, 7),
        "executed_action_history": torch.randn(
            batch,
            config.dimensions.executed_history_length,
            7,
        ),
        "policy_action": torch.randn(batch, 24, 7),
        "policy_action_raw": torch.randn(batch, 24, 7),
        "action": torch.randn(batch, 48, 7),
        "future_state": torch.randn(batch, 48, 7),
        "target_future_dinov2_tokens": torch.randn(batch, 12, 2, 64, 16).half(),
        "target_future_offsets": torch.arange(4, 49, 4)[None].expand(batch, -1),
        "sample_index": torch.arange(batch),
        "episode_idx": torch.arange(batch, dtype=torch.long),
        "frame_progress": torch.rand(batch),
    }
    goal_rows = torch.stack((torch.zeros(5, 12), torch.ones(5, 12)), dim=0)
    goal = GoalTemplate(
        tokens=goal_rows,
        mask=torch.ones(2, 5, dtype=torch.bool),
        metadata={"source": "test"},
        episode_condition_indices=torch.tensor([1, 0]),
    )
    typed = to_training_batch(
        raw,
        goal=goal,
        config=config,
        device=torch.device("cpu"),
    )
    typed.validate(config)
    assert not hasattr(typed.online, "future")
    assert typed.future.dino_supports.dtype == torch.float16
    assert torch.equal(typed.action_target.raw_units.cpu(), raw["policy_action_raw"])
    assert torch.equal(
        typed.action_target.current_raw_units.cpu(), raw["action_state_raw"]
    )
    assert torch.equal(
        typed.action_target.gripper_transition_boundary.cpu(),
        raw["gripper_transition_boundary"],
    )
    assert torch.equal(
        typed.online.history.codec_gripper_boundary.cpu(),
        raw["gripper_transition_boundary"][:, -1:],
    )
    assert torch.equal(
        typed.action_target.gripper_transition_boundary_raw_units.cpu(),
        raw["gripper_transition_boundary_raw"],
    )
    assert torch.equal(typed.online.goal.tokens[0].cpu(), goal_rows[1])
    assert torch.equal(typed.online.goal.tokens[1].cpu(), goal_rows[0])
    assert typed.audit.frame_progress is not None
    assert typed.audit.frame_progress.device.type == "cpu"


def test_t5_loader_fails_formal_missing_and_null_is_explicit(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    try:
        load_t5_condition(missing, max_tokens=8, expected_width=12)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("formal language condition must not silently become null")
    tokens, mask, metadata = load_t5_condition(
        missing,
        max_tokens=8,
        expected_width=12,
        allow_null=True,
    )
    assert tuple(tokens.shape) == (1, 1, 12)
    assert mask.item()
    assert metadata["source"] == "explicit_null_goal_smoke"


def test_t5_loader_accepts_one_precomputed_condition(tmp_path) -> None:
    path = tmp_path / "goal.pt"
    torch.save(
        {
            "last_hidden_state": torch.randn(1, 10, 12).half(),
            "attention_mask": torch.tensor([[1] * 9 + [0]]),
        },
        path,
    )
    tokens, mask, metadata = load_t5_condition(
        path,
        max_tokens=8,
        expected_width=12,
    )
    assert tuple(tokens.shape) == (1, 8, 12)
    assert tokens.dtype == torch.float32
    assert tuple(mask.shape) == (1, 8)
    assert metadata["effective_tokens"] == 8
    bank = load_t5_condition_bank(path, max_tokens=8, expected_width=12)
    assert not bank.is_instruction_bank
    assert bank.condition_indices([None, "any legacy single-task text"]).tolist() == [0, 0]
    assert torch.equal(bank.tokens, tokens)
    assert torch.equal(bank.mask, mask)


class _SamplingDataset(CachedTokenPolicyWindowDataset):
    def __init__(self) -> None:
        # This focused loader test owns no image/token stores.  It subclasses
        # the formal cached dataset so the production type boundary remains
        # exercised rather than mocked away.
        pass

    def __len__(self) -> int:
        return 16

    def training_information_signals(
        self,
        *,
        gripper_indices: tuple[int, ...],
        event_threshold: float,
        arm_motion: str = "adjacent_action_delta",
    ) -> tuple[np.ndarray, np.ndarray]:
        assert gripper_indices == (6,)
        assert event_threshold == 0.10
        assert arm_motion == "adjacent_action_delta"
        return (
            np.arange(16, dtype=np.float32),
            np.asarray([index % 4 == 0 for index in range(16)], dtype=bool),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(int(index), dtype=torch.long)}


def test_train_loader_owns_the_resolved_information_balanced_sampler() -> None:
    dataset = _SamplingDataset()
    goal = GoalTemplate(
        tokens=torch.zeros(1, 1, 12),
        mask=torch.ones(1, 1, dtype=torch.bool),
        metadata={"source": "test"},
    )
    bundle = MainlineDataBundle(
        episodes=(),
        splits={"train": (), "val": ()},
        datasets={"train": dataset, "val": dataset},
        materialized_episode_indices=(),
        action_normalizer=cast(ArrayNormalizer, None),
        state_normalizer=cast(ArrayNormalizer, None),
        goal=goal,
        skipped=(),
        sampling_seed=7,
        information_uniform_fraction=0.50,
        information_event_fraction=0.125,
        information_motion_quantile=0.70,
        gripper_event_threshold=0.10,
        gripper_indices=(6,),
    )
    train_loader = bundle.loader(
        "train",
        batch_size=8,
        workers=0,
        device=torch.device("cpu"),
    )
    assert isinstance(train_loader.batch_sampler, InformationBalancedBatchSampler)
    assert train_loader.batch_sampler.summary["uniform_fraction"] == 0.50
    assert train_loader.batch_sampler.summary["event_fraction"] == 0.125
    assert train_loader.batch_sampler.summary["motion_fraction"] == 0.375
    assert len(next(iter(train_loader))["index"]) == 8

    validation_loader = bundle.loader(
        "val",
        batch_size=8,
        workers=0,
        device=torch.device("cpu"),
    )
    assert not isinstance(
        validation_loader.batch_sampler,
        InformationBalancedBatchSampler,
    )


def test_validation_loader_does_not_keep_prefetch_workers_alive() -> None:
    """The one-batch startup preflight must not orphan validation workers."""

    dataset = _SamplingDataset()
    goal = GoalTemplate(
        tokens=torch.zeros(1, 1, 12),
        mask=torch.ones(1, 1, dtype=torch.bool),
        metadata={"source": "test"},
    )
    bundle = MainlineDataBundle(
        episodes=(),
        splits={"train": (), "val": ()},
        datasets={"train": dataset, "val": dataset},
        materialized_episode_indices=(),
        action_normalizer=cast(ArrayNormalizer, None),
        state_normalizer=cast(ArrayNormalizer, None),
        goal=goal,
        skipped=(),
        sampling_seed=7,
        information_uniform_fraction=0.50,
        information_event_fraction=0.125,
        information_motion_quantile=0.70,
        gripper_event_threshold=0.10,
        gripper_indices=(6,),
    )
    train_loader = bundle.loader(
        "train",
        batch_size=2,
        workers=1,
        device=torch.device("cpu"),
    )
    validation_loader = bundle.loader(
        "val",
        batch_size=2,
        workers=1,
        device=torch.device("cpu"),
        shuffle=False,
    )
    assert train_loader.persistent_workers is True
    assert validation_loader.persistent_workers is False
    assert train_loader.prefetch_factor == 1
    assert validation_loader.prefetch_factor == 1


def test_worker_tensor_sharing_fails_closed_on_file_system_strategy() -> None:
    original = torch.multiprocessing.get_sharing_strategy()
    try:
        _configure_worker_tensor_sharing(1)
        assert torch.multiprocessing.get_sharing_strategy() == "file_system"
    finally:
        torch.multiprocessing.set_sharing_strategy(original)


def test_cached_dataset_groups_three_causal_dino_rows_with_future_supports() -> None:
    class Base:
        config = ObservedStateDatasetConfig()

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {
                "history_keys": torch.tensor([[2, 9], [2, 13], [2, 17]]),
                "future_keys": torch.stack((torch.full((12,), 2), torch.arange(21, 69, 4)), dim=-1),
                "future_offsets": torch.arange(4, 49, 4),
                "history_obs_image": torch.zeros(3, 2, 3, 32, 32),
            }

    class Store:
        def __init__(self):
            self.calls = []

        def load_batch(self, keys):
            self.calls.append(keys.clone())
            return torch.zeros(len(keys), 2, 64, 16)

    store = Store()
    dataset = CachedTokenPolicyWindowDataset(
        cast(ObservedStateWindowDataset, Base()),
        token_store=cast(DinoV2TokenStore, store),
    )
    sample = dataset[0]
    assert len(store.calls) == 1
    assert torch.equal(store.calls[0][0], torch.tensor([2, 9]))
    assert tuple(store.calls[0].shape) == (15, 2)
    assert tuple(sample["history_dinov2_tokens"].shape) == (3, 2, 64, 16)
    assert tuple(sample["history_obs_image"].shape) == (3, 2, 3, 32, 32)
    assert tuple(sample["target_future_dinov2_tokens"].shape) == (12, 2, 64, 16)
