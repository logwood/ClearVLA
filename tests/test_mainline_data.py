from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import torch

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
    ) -> tuple[np.ndarray, np.ndarray]:
        assert gripper_indices == (6,)
        assert event_threshold == 0.10
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
